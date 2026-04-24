#!/usr/bin/env python3
"""Stage progression visualization script.

Creates detailed visualizations showing how optical flow evolves through HQS stages.
Useful for understanding convergence behavior and debugging.
"""
from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle
from tqdm import tqdm

from models import build_model
from data import build_dataset, build_dataloader
from utils import compute_metrics, flow_to_hsv


logger = logging.getLogger(__name__)


def setup_logging(output_dir: str) -> None:
    """Setup logging."""
    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(output_dir, "stage_visualization.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file),
        ],
    )


def visualize_stage_progression(
    model: torch.nn.Module,
    dataloader,
    output_dir: str,
    device: torch.device,
    num_samples: int = 10,
) -> None:
    """
    Visualize optical flow progression through HQS stages.

    Creates a grid showing:
    - Original images
    - Flow at each stage (HSV visualization)
    - EPE at each stage
    - Error heat map at each stage
    """
    model.eval()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sample_count = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Processing")):
            if sample_count >= num_samples:
                break

            img1 = batch["image1"].to(device, non_blocking=True)
            img2 = batch["image2"].to(device, non_blocking=True)
            flow_gt = batch["flow"].to(device, non_blocking=True)
            valid = batch["valid"].to(device, non_blocking=True)

            # Forward pass
            out = model(img1, img2)
            flow_preds = out["flow_preds"]

            B = img1.shape[0]
            for b in range(B):
                if sample_count >= num_samples:
                    break

                # Extract data
                img1_b = img1[b].cpu()
                img2_b = img2[b].cpu()
                flow_gt_b = flow_gt[b].cpu()
                valid_b = valid[b].cpu()
                flow_stages = [fp[b].cpu() for fp in flow_preds]

                # Create visualization
                _create_stage_grid(
                    img1_b, img2_b, flow_gt_b, valid_b, flow_stages,
                    output_dir / f"sample_{sample_count:04d}_stages.png",
                )

                # Create convergence plot
                _create_convergence_plot(
                    flow_gt_b, valid_b, flow_stages,
                    output_dir / f"sample_{sample_count:04d}_convergence.png",
                )

                sample_count += 1

    logger.info(f"Created {sample_count} stage progression visualizations")


def _create_stage_grid(
    img1: torch.Tensor,
    img2: torch.Tensor,
    flow_gt: torch.Tensor,
    valid: torch.Tensor,
    flow_stages: list[torch.Tensor],
    output_path: Path,
) -> None:
    """Create a grid visualization of stages."""
    K = len(flow_stages)
    cols = 3
    rows = K + 2  # +1 for images, +1 for GT

    fig = plt.figure(figsize=(15, 4 * rows))
    gs = gridspec.GridSpec(rows, cols, figure=fig, hspace=0.35, wspace=0.25)

    # Normalize images to [0, 1] for display
    def normalize_img(img):
        img_np = img.permute(1, 2, 0).numpy()
        if img_np.max() > 2.0:
            img_np = img_np / 255.0
        return np.clip(img_np, 0, 1)

    # Row 0: Input images
    ax = fig.add_subplot(gs[0, 0])
    ax.imshow(normalize_img(img1))
    ax.set_title("Image 1 (Reference)", fontsize=11, fontweight="bold")
    ax.axis("off")

    ax = fig.add_subplot(gs[0, 1])
    ax.imshow(normalize_img(img2))
    ax.set_title("Image 2 (Target)", fontsize=11, fontweight="bold")
    ax.axis("off")

    ax = fig.add_subplot(gs[0, 2])
    # Show valid mask
    valid_np = valid.numpy()
    ax.imshow(valid_np, cmap="gray")
    ax.set_title("Valid Mask", fontsize=11, fontweight="bold")
    ax.axis("off")

    # Row 1: Ground truth
    ax = fig.add_subplot(gs[1, 0])
    flow_gt_hsv = flow_to_hsv(flow_gt)
    ax.imshow(flow_gt_hsv)
    metrics_gt = compute_metrics(flow_gt, flow_gt, valid)
    ax.set_title(
        f"Ground Truth Flow\nEPE: {metrics_gt['epe']:.3f}",
        fontsize=11, fontweight="bold"
    )
    ax.axis("off")

    ax = fig.add_subplot(gs[1, 1])
    ax.axis("off")

    ax = fig.add_subplot(gs[1, 2])
    ax.axis("off")

    # Rows 2+: Each stage
    for stage_idx, flow_k in enumerate(flow_stages):
        row = 2 + stage_idx

        # Upsample to GT resolution if needed
        if flow_k.shape != flow_gt.shape:
            scale_h = flow_gt.shape[-2] / flow_k.shape[-2]
            scale_w = flow_gt.shape[-1] / flow_k.shape[-1]
            flow_k = F.interpolate(
                flow_k.unsqueeze(0), size=flow_gt.shape[-2:],
                mode="bilinear", align_corners=True
            ).squeeze(0)
            flow_k[0] *= scale_w
            flow_k[1] *= scale_h

        # Column 0: Flow visualization
        ax = fig.add_subplot(gs[row, 0])
        flow_hsv = flow_to_hsv(flow_k)
        ax.imshow(flow_hsv)
        metrics_k = compute_metrics(flow_k, flow_gt, valid)
        ax.set_title(
            f"Stage {stage_idx + 1}\nEPE: {metrics_k['epe']:.3f}",
            fontsize=11
        )
        ax.axis("off")

        # Column 1: Error map
        ax = fig.add_subplot(gs[row, 1])
        epe = torch.sqrt(((flow_k - flow_gt) ** 2).sum(dim=0))
        epe_np = epe.numpy()
        im = ax.imshow(epe_np, cmap="hot")
        epe_masked = epe_np[valid.numpy() > 0.5]
        ax.set_title(
            f"EPE Map\nMean: {epe_masked.mean():.3f}, "
            f"Max: {epe_masked.max():.1f}",
            fontsize=10
        )
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        # Column 2: Magnitude map
        ax = fig.add_subplot(gs[row, 2])
        mag_k = torch.sqrt((flow_k ** 2).sum(dim=0))
        mag_gt = torch.sqrt((flow_gt ** 2).sum(dim=0))
        mag_diff = (mag_k - mag_gt).abs()
        mag_diff_np = mag_diff.numpy()
        im = ax.imshow(mag_diff_np, cmap="viridis")
        ax.set_title(
            f"Magnitude Diff\nMean: {mag_diff_np.mean():.3f}",
            fontsize=10
        )
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.savefig(output_path, dpi=100, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved → {output_path}")


def _create_convergence_plot(
    flow_gt: torch.Tensor,
    valid: torch.Tensor,
    flow_stages: list[torch.Tensor],
    output_path: Path,
) -> None:
    """Create convergence plot showing EPE over stages."""
    K = len(flow_stages)
    epe_stages = []
    mag_stages = []

    for flow_k in flow_stages:
        # Upsample to GT resolution if needed
        if flow_k.shape != flow_gt.shape:
            scale_h = flow_gt.shape[-2] / flow_k.shape[-2]
            scale_w = flow_gt.shape[-1] / flow_k.shape[-1]
            flow_k = F.interpolate(
                flow_k.unsqueeze(0), size=flow_gt.shape[-2:],
                mode="bilinear", align_corners=True
            ).squeeze(0)
            flow_k[0] *= scale_w
            flow_k[1] *= scale_h

        epe = torch.sqrt(((flow_k - flow_gt) ** 2).sum(dim=0))
        epe_masked = epe[valid.bool()]
        epe_stages.append(epe_masked.mean().item())

        mag = torch.sqrt((flow_k ** 2).sum(dim=0))
        mag_stages.append(mag.mean().item())

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # EPE convergence
    stages = list(range(1, K + 1))
    ax1.plot(stages, epe_stages, "o-", linewidth=2.5, markersize=8, color="red")
    ax1.set_xlabel("HQS Stage", fontsize=12)
    ax1.set_ylabel("Mean EPE (pixels)", fontsize=12)
    ax1.set_title("EPE Convergence", fontsize=13, fontweight="bold")
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(stages)

    # Improvement percentage
    initial_epe = epe_stages[0]
    improvements = [100 * (1 - epe / initial_epe) for epe in epe_stages]
    ax1_twin = ax1.twinx()
    ax1_twin.plot(stages, improvements, "s--", linewidth=2, markersize=6,
                  color="blue", alpha=0.7)
    ax1_twin.set_ylabel("Improvement (%)", fontsize=12, color="blue")
    ax1_twin.tick_params(axis="y", labelcolor="blue")

    # Magnitude tracking
    ax2.plot(stages, mag_stages, "s-", linewidth=2.5, markersize=8, color="green")
    ax2.set_xlabel("HQS Stage", fontsize=12)
    ax2.set_ylabel("Mean Flow Magnitude (pixels)", fontsize=12)
    ax2.set_title("Flow Magnitude Evolution", fontsize=13, fontweight="bold")
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(stages)

    plt.tight_layout()
    plt.savefig(output_path, dpi=100, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved convergence plot → {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize optical flow stage progression"
    )
    parser.add_argument(
        "--config", "-c", required=True,
        help="Path to model config YAML",
    )
    parser.add_argument(
        "--checkpoint", "-ckpt", required=True,
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--data_config", "-dc", required=True,
        help="Path to data config YAML",
    )
    parser.add_argument(
        "--output_dir", "-o", required=True,
        help="Output directory for visualizations",
    )
    parser.add_argument(
        "--num_samples", type=int, default=10,
        help="Number of samples to visualize",
    )
    parser.add_argument(
        "--device", default=None,
        help="Device (cuda/mps/cpu)",
    )
    args = parser.parse_args()

    # Setup
    from omegaconf import OmegaConf
    setup_logging(args.output_dir)
    logger.info("=" * 80)
    logger.info("Stage Progression Visualization")
    logger.info("=" * 80)

    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    logger.info(f"Device: {device}")

    # Load configs
    model_cfg = OmegaConf.load(args.config)
    data_cfg = OmegaConf.load(args.data_config)

    # Build model
    model = build_model(model_cfg).to(device)
    logger.info(f"Model: {model.__class__.__name__}")

    # Load checkpoint
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"], strict=False)
    logger.info(f"Loaded checkpoint: {args.checkpoint}")

    # Build data
    val_cfg = data_cfg.get("val_data") or data_cfg.get("data") or data_cfg
    val_cfg = OmegaConf.merge(val_cfg, {"batch_size": 1})

    val_data = build_dataset(val_cfg, split="val")
    val_loader = build_dataloader(val_data, val_cfg, split="val")
    logger.info(f"Validation set: {len(val_data)} samples")

    # Visualize
    visualize_stage_progression(
        model, val_loader, args.output_dir, device,
        num_samples=args.num_samples,
    )

    logger.info("=" * 80)
    logger.info(f"Visualizations saved to {args.output_dir}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
