#!/usr/bin/env python3
"""Evaluate rectified flow model — compute FID scores at various NFE."""

import sys
sys.path.insert(0, ".")

import argparse
import json
import logging
import os
import tempfile
from pathlib import Path

import torch
import torchvision
import torchvision.transforms as T
import yaml
from tqdm import tqdm

from src.flow_matching.flow.rectified_flow import RectifiedFlow
from src.flow_matching.models.unet import UNet
from src.flow_matching.utils.ema import EMA

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def generate_samples(flow, num_samples, device, num_steps, batch_size=512, save_dir=None):
    """Generate samples and optionally save them as individual images."""
    all_samples = []
    remaining = num_samples

    with tqdm(total=num_samples, desc=f"Generating (NFE={num_steps})") as pbar:
        while remaining > 0:
            B = min(batch_size, remaining)
            samples = flow.sample((B, 3, 32, 32), device, num_steps=num_steps)
            samples = (samples + 1) / 2  # [-1,1] -> [0,1]
            samples = samples.clamp(0, 1)

            if save_dir:
                for i, img in enumerate(samples):
                    idx = num_samples - remaining + i
                    torchvision.utils.save_image(img, save_dir / f"{idx:06d}.png")

            all_samples.append(samples.cpu())
            remaining -= B
            pbar.update(B)

    return torch.cat(all_samples)


def compute_fid(generated_dir, dataset_dir=None, device="cuda"):
    """Compute FID between generated samples and CIFAR-10 training set."""
    from pytorch_fid.fid_score import calculate_fid_given_paths

    if dataset_dir is None:
        # Create CIFAR-10 reference images
        dataset_dir = Path("data/cifar10_train_images")
        if not dataset_dir.exists():
            logger.info("Preparing CIFAR-10 reference images for FID...")
            dataset_dir.mkdir(parents=True, exist_ok=True)
            dataset = torchvision.datasets.CIFAR10(root="data", train=True, download=True)
            for i, (img, _) in enumerate(tqdm(dataset, desc="Saving CIFAR-10")):
                img.save(str(dataset_dir / f"{i:06d}.png"))
            logger.info(f"Saved {len(dataset)} reference images")

    fid = calculate_fid_given_paths(
        [str(dataset_dir), str(generated_dir)],
        batch_size=256,
        device=device,
        dims=2048,
    )
    return fid


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint")
    parser.add_argument("--num-samples", type=int, default=50000)
    parser.add_argument("--nfe-list", type=int, nargs="+", default=[1, 2, 5, 10, 25, 50, 100])
    parser.add_argument("--use-ema", action="store_true", default=True)
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--batch-size", type=int, default=512)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load checkpoint
    logger.info(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = ckpt["config"]

    # Build model
    model_cfg = config["model"]
    model = UNet(
        in_channels=3,
        base_channels=model_cfg["base_channels"],
        channel_mults=tuple(model_cfg["channel_mults"]),
        num_res_blocks=model_cfg["num_res_blocks"],
        attn_resolutions=tuple(model_cfg["attn_resolutions"]),
        dropout=0.0,  # No dropout at inference
        num_heads=model_cfg["num_heads"],
    ).to(device)

    if args.use_ema and "ema" in ckpt:
        logger.info("Using EMA model")
        model.load_state_dict(ckpt["ema"])
    else:
        model.load_state_dict(ckpt["model"])

    model.eval()
    flow = RectifiedFlow(model)

    step = ckpt.get("step", 0)
    logger.info(f"Model from step {step}, params: {model.num_params / 1e6:.2f}M")

    # Evaluate at different NFE
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {"step": step, "num_samples": args.num_samples, "fid_scores": {}}

    for nfe in args.nfe_list:
        logger.info(f"\n{'='*50}")
        logger.info(f"Evaluating NFE={nfe}")

        with tempfile.TemporaryDirectory() as tmpdir:
            gen_dir = Path(tmpdir)
            generate_samples(flow, args.num_samples, device, nfe, args.batch_size, gen_dir)

            fid = compute_fid(gen_dir, device=str(device))
            results["fid_scores"][str(nfe)] = fid
            logger.info(f"  FID (NFE={nfe}): {fid:.2f}")

        # Save intermediate results
        with open(output_dir / "fid_results.json", "w") as f:
            json.dump(results, f, indent=2)

    # Summary
    logger.info(f"\n{'='*60}")
    logger.info("FID EVALUATION SUMMARY")
    logger.info(f"{'='*60}")
    logger.info(f"{'NFE':>6} {'FID':>10}")
    logger.info("-" * 20)
    for nfe in sorted(results["fid_scores"].keys(), key=int):
        logger.info(f"{nfe:>6} {results['fid_scores'][nfe]:>10.2f}")

    # Also generate a nice sample grid from best NFE
    best_nfe = min(results["fid_scores"], key=lambda k: results["fid_scores"][k])
    logger.info(f"\nBest: NFE={best_nfe}, FID={results['fid_scores'][best_nfe]:.2f}")

    # Generate display samples
    samples = flow.sample((64, 3, 32, 32), device, num_steps=int(best_nfe))
    samples = (samples + 1) / 2
    grid = torchvision.utils.make_grid(samples, nrow=8, padding=2)
    torchvision.utils.save_image(grid, output_dir / "best_samples.png")

    with open(output_dir / "fid_results.json", "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    main()
