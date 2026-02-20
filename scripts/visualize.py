#!/usr/bin/env python3
"""Generate publication-quality visualizations for rectified flow results."""

import sys
sys.path.insert(0, ".")

import json
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "#fafafa",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "font.family": "sans-serif",
    "font.size": 11,
})


def plot_training_loss(log_path, save_dir):
    """Plot training loss curve."""
    with open(log_path) as f:
        log_data = json.load(f)

    steps = [d["step"] for d in log_data]
    losses = [d["loss"] for d in log_data]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(steps, losses, color="#2196F3", linewidth=1, alpha=0.3)

    # Smoothed
    window = min(50, len(losses) // 10)
    if window > 1:
        smoothed = np.convolve(losses, np.ones(window) / window, mode="valid")
        ax.plot(steps[window - 1:], smoothed, color="#2196F3", linewidth=2, label="Smoothed")

    ax.set_xlabel("Training Step", fontsize=12)
    ax.set_ylabel("CFM Loss (MSE)", fontsize=12)
    ax.set_title("Rectified Flow Training Loss (CIFAR-10)", fontsize=14, fontweight="bold")
    ax.legend()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(save_dir / "training_loss.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("  Saved training_loss.png")


def plot_fid_vs_nfe(fid_path, save_dir):
    """Plot FID vs number of function evaluations."""
    with open(fid_path) as f:
        fid_data = json.load(f)

    fid_scores = fid_data["fid_scores"]
    nfes = sorted([int(k) for k in fid_scores.keys()])
    fids = [fid_scores[str(n)] for n in nfes]

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(nfes, fids, "o-", color="#FF5722", linewidth=2, markersize=8)

    for n, f in zip(nfes, fids):
        ax.annotate(f"{f:.1f}", (n, f), textcoords="offset points",
                   xytext=(0, 10), ha="center", fontsize=9, fontweight="bold")

    ax.set_xlabel("Number of Function Evaluations (NFE)", fontsize=12)
    ax.set_ylabel("FID-50K", fontsize=12)
    ax.set_title("FID vs Sampling Steps (CIFAR-10)", fontsize=14, fontweight="bold")
    ax.set_xscale("log")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(save_dir / "fid_vs_nfe.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("  Saved fid_vs_nfe.png")


def plot_comparison_with_literature(fid_path, save_dir):
    """Compare our results with published baselines."""
    with open(fid_path) as f:
        fid_data = json.load(f)

    our_fid = fid_data["fid_scores"]

    # Published CIFAR-10 FID results
    baselines = [
        ("DDPM", 1000, 3.17),
        ("Score SDE", 2000, 2.20),
        ("1-RF (Liu et al.)", 127, 2.58),
        ("OT-CFM", 100, 3.50),
    ]

    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot baselines
    for name, nfe, fid in baselines:
        ax.scatter(nfe, fid, s=100, c="#9E9E9E", alpha=0.6, edgecolors="gray", zorder=2)
        ax.annotate(name, (nfe, fid), textcoords="offset points",
                   xytext=(5, 5), fontsize=9, color="gray")

    # Plot our results
    nfes = sorted([int(k) for k in our_fid.keys()])
    fids = [our_fid[str(n)] for n in nfes]
    ax.plot(nfes, fids, "o-", color="#2196F3", linewidth=2, markersize=8,
            label="Ours (1-Rectified Flow)", zorder=3)

    for n, f in zip(nfes, fids):
        if n in [1, 5, 10, 50, 100]:
            ax.annotate(f"{f:.1f}", (n, f), textcoords="offset points",
                       xytext=(0, 10), ha="center", fontsize=9, fontweight="bold", color="#2196F3")

    ax.set_xlabel("Number of Function Evaluations (NFE)", fontsize=12)
    ax.set_ylabel("FID-50K", fontsize=12)
    ax.set_title("Rectified Flow vs Baselines (CIFAR-10 Unconditional)", fontsize=14, fontweight="bold")
    ax.set_xscale("log")
    ax.legend(fontsize=11)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(save_dir / "comparison_with_baselines.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("  Saved comparison_with_baselines.png")


def create_nfe_comparison_grid(save_dir):
    """Create a grid showing samples at different NFE values."""
    import torch
    import torchvision

    # Look for sample images at different steps
    sample_dirs = list(Path("output").rglob("samples_step*.png"))
    if not sample_dirs:
        logger.warning("  No sample images found, skipping NFE comparison grid")
        return

    # Use the latest sample
    latest = sorted(sample_dirs)[-1]
    logger.info(f"  Using latest samples: {latest}")

    # Copy to results
    import shutil
    shutil.copy2(str(latest), str(save_dir / "generated_samples.png"))
    logger.info("  Saved generated_samples.png")


def main():
    save_dir = Path("results")
    save_dir.mkdir(parents=True, exist_ok=True)

    # Training loss
    log_paths = list(Path("output").rglob("training_log.json"))
    for log_path in log_paths:
        logger.info(f"Plotting training loss from {log_path}")
        plot_training_loss(log_path, save_dir)

    # FID results
    fid_paths = list(Path("results").glob("fid_results.json")) + list(Path("output").rglob("fid_results.json"))
    for fid_path in fid_paths:
        logger.info(f"Plotting FID results from {fid_path}")
        plot_fid_vs_nfe(fid_path, save_dir)
        plot_comparison_with_literature(fid_path, save_dir)

    # Sample grids
    create_nfe_comparison_grid(save_dir)

    logger.info(f"\nAll visualizations saved to {save_dir}")


if __name__ == "__main__":
    main()
