#!/usr/bin/env python3
"""Train rectified flow matching on CIFAR-10."""

import sys
sys.path.insert(0, ".")

import argparse
import json
import logging
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
import yaml
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.flow_matching.flow.rectified_flow import RectifiedFlow
from src.flow_matching.models.unet import UNet
from src.flow_matching.utils.ema import EMA

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def get_cifar10_loader(batch_size, num_workers=4):
    """Load CIFAR-10 with standard augmentation."""
    transform = T.Compose([
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),  # Scale to [-1, 1]
    ])
    dataset = torchvision.datasets.CIFAR10(
        root="data", train=True, download=True, transform=transform,
    )
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True,
    )


def save_samples(flow, step, device, save_dir, num_steps=100, n=64):
    """Generate and save sample grid."""
    samples = flow.sample((n, 3, 32, 32), device, num_steps=num_steps)
    samples = (samples + 1) / 2  # [-1,1] -> [0,1]
    grid = torchvision.utils.make_grid(samples, nrow=8, padding=2)
    save_path = save_dir / f"samples_step{step:07d}.png"
    torchvision.utils.save_image(grid, save_path)
    return save_path


def save_trajectory(flow, step, device, save_dir, num_steps=10):
    """Generate and save ODE trajectory visualization."""
    _, traj = flow.sample((8, 3, 32, 32), device, num_steps=num_steps, return_trajectory=True)
    # Show trajectory at t=0, 0.25, 0.5, 0.75, 1.0
    indices = [0, num_steps // 4, num_steps // 2, 3 * num_steps // 4, num_steps]
    all_frames = []
    for idx in indices:
        frame = (traj[idx] + 1) / 2
        all_frames.append(frame)
    # Stack into grid: rows=timesteps, cols=samples
    grid = torchvision.utils.make_grid(torch.cat(all_frames, dim=0), nrow=8, padding=2)
    save_path = save_dir / f"trajectory_step{step:07d}.png"
    torchvision.utils.save_image(grid, save_path)
    return save_path


def train(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # Directories
    output_dir = Path("output") / f"cifar10_{config['flow']['time_distribution']}"
    output_dir.mkdir(parents=True, exist_ok=True)
    sample_dir = output_dir / "samples"
    sample_dir.mkdir(exist_ok=True)
    ckpt_dir = output_dir / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)

    # Model
    model_cfg = config["model"]
    model = UNet(
        in_channels=3,
        base_channels=model_cfg["base_channels"],
        channel_mults=tuple(model_cfg["channel_mults"]),
        num_res_blocks=model_cfg["num_res_blocks"],
        attn_resolutions=tuple(model_cfg["attn_resolutions"]),
        dropout=model_cfg["dropout"],
        num_heads=model_cfg["num_heads"],
    ).to(device)

    logger.info(f"Model parameters: {model.num_params / 1e6:.2f}M")

    # Flow
    flow = RectifiedFlow(model, time_distribution=config["flow"]["time_distribution"])

    # EMA
    train_cfg = config["training"]
    ema = EMA(model, decay=train_cfg["ema_decay"])

    # Optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=train_cfg["lr"],
        betas=tuple(train_cfg["betas"]),
        weight_decay=train_cfg["weight_decay"],
    )

    # Learning rate warmup
    def lr_lambda(step):
        if step < train_cfg["warmup_steps"]:
            return step / train_cfg["warmup_steps"]
        return 1.0

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Mixed precision
    scaler = GradScaler() if train_cfg["mixed_precision"] else None

    # Data
    loader = get_cifar10_loader(train_cfg["batch_size"], config["data"]["num_workers"])
    data_iter = iter(loader)

    # Resume
    start_step = 0
    resume_path = ckpt_dir / "latest.pt"
    if resume_path.exists():
        logger.info(f"Resuming from {resume_path}")
        ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        ema.load_state_dict(ckpt["ema"])
        start_step = ckpt["step"]
        if scaler and "scaler" in ckpt:
            scaler.load_state_dict(ckpt["scaler"])
        logger.info(f"Resumed from step {start_step}")

    # Training loop
    total_steps = train_cfg["total_steps"]
    log_interval = train_cfg["log_interval"]
    sample_interval = train_cfg["sample_interval"]
    save_interval = train_cfg["save_interval"]

    model.train()
    running_loss = 0.0
    t0 = time.time()

    log_data = []

    for step in range(start_step, total_steps):
        # Get batch
        try:
            batch, _ = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            batch, _ = next(data_iter)

        batch = batch.to(device, non_blocking=True)

        # Forward
        optimizer.zero_grad(set_to_none=True)

        if scaler:
            with autocast(dtype=torch.bfloat16):
                loss, info = flow.compute_loss(batch)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), train_cfg["grad_clip"])
            scaler.step(optimizer)
            scaler.update()
        else:
            loss, info = flow.compute_loss(batch)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), train_cfg["grad_clip"])
            optimizer.step()

        scheduler.step()
        ema.update(model)

        running_loss += info["loss"]

        # Logging
        if (step + 1) % log_interval == 0:
            avg_loss = running_loss / log_interval
            elapsed = time.time() - t0
            steps_per_sec = log_interval / elapsed
            lr = scheduler.get_last_lr()[0]

            log_entry = {
                "step": step + 1,
                "loss": avg_loss,
                "lr": lr,
                "steps_per_sec": steps_per_sec,
                "v_pred_norm": info["v_pred_norm"],
                "v_target_norm": info["v_target_norm"],
            }
            log_data.append(log_entry)

            logger.info(
                f"Step {step+1}/{total_steps} | Loss: {avg_loss:.4f} | "
                f"LR: {lr:.2e} | {steps_per_sec:.1f} steps/s"
            )

            running_loss = 0.0
            t0 = time.time()

        # Generate samples
        if (step + 1) % sample_interval == 0:
            logger.info("Generating samples...")
            # Use EMA model for sampling
            ema_flow = RectifiedFlow(ema.shadow, time_distribution=config["flow"]["time_distribution"])
            path = save_samples(ema_flow, step + 1, device, sample_dir, num_steps=100)
            logger.info(f"  Saved samples to {path}")

            # Also save trajectory
            traj_path = save_trajectory(ema_flow, step + 1, device, sample_dir, num_steps=10)
            logger.info(f"  Saved trajectory to {traj_path}")

        # Save checkpoint
        if (step + 1) % save_interval == 0:
            ckpt = {
                "step": step + 1,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "ema": ema.state_dict(),
                "config": config,
            }
            if scaler:
                ckpt["scaler"] = scaler.state_dict()

            torch.save(ckpt, ckpt_dir / f"step_{step+1:07d}.pt")
            torch.save(ckpt, ckpt_dir / "latest.pt")
            logger.info(f"  Saved checkpoint at step {step+1}")

            # Save training log
            with open(output_dir / "training_log.json", "w") as f:
                json.dump(log_data, f)

    # Final save
    ckpt = {
        "step": total_steps,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "ema": ema.state_dict(),
        "config": config,
    }
    if scaler:
        ckpt["scaler"] = scaler.state_dict()
    torch.save(ckpt, ckpt_dir / "final.pt")
    torch.save(ckpt, ckpt_dir / "latest.pt")

    with open(output_dir / "training_log.json", "w") as f:
        json.dump(log_data, f)

    logger.info(f"Training complete. Checkpoints in {ckpt_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--time-dist", choices=["uniform", "logit_normal", "u_shaped"], default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    if args.steps:
        config["training"]["total_steps"] = args.steps
    if args.time_dist:
        config["flow"]["time_distribution"] = args.time_dist

    train(config)


if __name__ == "__main__":
    main()
