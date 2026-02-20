#!/usr/bin/env python3
"""Reflow: Generate coupling pairs and train 2-rectified flow."""

import sys
sys.path.insert(0, ".")

import argparse
import json
import logging
import time
from pathlib import Path

import torch
import torch.nn as nn
import torchvision
import yaml
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from src.flow_matching.flow.rectified_flow import RectifiedFlow
from src.flow_matching.models.unet import UNet
from src.flow_matching.utils.ema import EMA

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def generate_pairs(checkpoint_path, num_pairs, gen_steps, batch_size, device):
    """Generate (noise, data) coupling pairs from 1-RF model."""
    logger.info(f"Loading model from {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = ckpt["config"]

    model_cfg = config["model"]
    model = UNet(
        in_channels=3,
        base_channels=model_cfg["base_channels"],
        channel_mults=tuple(model_cfg["channel_mults"]),
        num_res_blocks=model_cfg["num_res_blocks"],
        attn_resolutions=tuple(model_cfg["attn_resolutions"]),
        dropout=0.0,
        num_heads=model_cfg["num_heads"],
    ).to(device)

    # Use EMA model
    if "ema" in ckpt:
        model.load_state_dict(ckpt["ema"])
    else:
        model.load_state_dict(ckpt["model"])

    model.eval()
    flow = RectifiedFlow(model)

    logger.info(f"Generating {num_pairs} pairs with {gen_steps} Euler steps...")
    z0, z1 = flow.generate_reflow_pairs(
        num_pairs, (3, 32, 32), device, num_steps=gen_steps, batch_size=batch_size
    )

    return z0, z1, config


def train_reflow(z0, z1, config, output_dir, device, total_steps=400000):
    """Train 2-rectified flow on the generated coupling pairs."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = output_dir / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)
    sample_dir = output_dir / "samples"
    sample_dir.mkdir(exist_ok=True)

    # Model (fresh initialization or from 1-RF checkpoint)
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

    flow = RectifiedFlow(model, time_distribution="uniform")
    ema = EMA(model, decay=0.9999)

    optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)

    def lr_lambda(step):
        if step < 5000:
            return step / 5000
        return 1.0

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = GradScaler()

    # Dataloader from pre-generated pairs
    dataset = TensorDataset(z0, z1)
    loader = DataLoader(dataset, batch_size=128, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    data_iter = iter(loader)

    model.train()
    log_data = []
    running_loss = 0.0
    t0 = time.time()

    for step in range(total_steps):
        try:
            x0_batch, x1_batch = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            x0_batch, x1_batch = next(data_iter)

        x0_batch = x0_batch.to(device, non_blocking=True)
        x1_batch = x1_batch.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with autocast(dtype=torch.bfloat16):
            loss, info = flow.compute_loss(x1_batch, x0=x0_batch)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        scheduler.step()
        ema.update(model)
        running_loss += info["loss"]

        if (step + 1) % 500 == 0:
            avg_loss = running_loss / 500
            elapsed = time.time() - t0
            logger.info(f"[Reflow] Step {step+1}/{total_steps} | Loss: {avg_loss:.4f} | "
                       f"{500/elapsed:.1f} steps/s")
            log_data.append({"step": step+1, "loss": avg_loss})
            running_loss = 0.0
            t0 = time.time()

        if (step + 1) % 25000 == 0:
            ema_flow = RectifiedFlow(ema.shadow)
            samples = ema_flow.sample((64, 3, 32, 32), device, num_steps=10)
            samples = (samples + 1) / 2
            grid = torchvision.utils.make_grid(samples, nrow=8, padding=2)
            torchvision.utils.save_image(grid, sample_dir / f"reflow_step{step+1:07d}.png")

        if (step + 1) % 100000 == 0:
            torch.save({
                "step": step+1, "model": model.state_dict(),
                "ema": ema.state_dict(), "config": config,
            }, ckpt_dir / f"reflow_step_{step+1:07d}.pt")

    # Final save
    torch.save({
        "step": total_steps, "model": model.state_dict(),
        "ema": ema.state_dict(), "config": config,
        "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
    }, ckpt_dir / "reflow_final.pt")

    with open(output_dir / "reflow_log.json", "w") as f:
        json.dump(log_data, f)

    logger.info(f"Reflow training complete. Saved to {ckpt_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="Path to 1-RF checkpoint")
    parser.add_argument("--num-pairs", type=int, default=500000)
    parser.add_argument("--gen-steps", type=int, default=50)
    parser.add_argument("--train-steps", type=int, default=400000)
    parser.add_argument("--output-dir", default="output/cifar10_reflow")
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()

    device = torch.device("cuda")

    # Stage 1: Generate pairs
    z0, z1, config = generate_pairs(
        args.checkpoint, args.num_pairs, args.gen_steps, args.batch_size, device
    )
    logger.info(f"Generated {len(z0)} pairs")

    # Save pairs
    pairs_dir = Path(args.output_dir) / "pairs"
    pairs_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"z0": z0, "z1": z1}, pairs_dir / "reflow_pairs.pt")
    logger.info(f"Saved pairs to {pairs_dir}")

    # Stage 2: Train 2-RF
    train_reflow(z0, z1, config, args.output_dir, device, total_steps=args.train_steps)


if __name__ == "__main__":
    main()
