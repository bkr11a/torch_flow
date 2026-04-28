#!/usr/bin/env python3
"""Comprehensive evaluation script for optical flow models.

Evaluates model on a dataset and:
1. Saves all predicted flows to disk
2. Saves intermediate stages for analysis
3. Computes metrics (EPE, F1, OFCE, smoothness, photometric loss)
4. Creates visualizations of flows and metrics
5. Saves stage-by-stage progression for convergence analysis
"""
from __future__ import annotations

import argparse
import logging
import json
import os
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

from models import build_model
from data import build_dataset, build_dataloader
from losses import SequenceLoss, SmoothnessLoss, OFCELoss
from utils import (
    compute_metrics,
    read_flow,
    write_flow,
    flow_to_hsv,
    flow_to_hsv_batch,
    create_flow_colorwheel,
)


logger = logging.getLogger(__name__)


def setup_logging(output_dir: str) -> None:
    """Setup logging to console and file."""
    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(output_dir, "evaluation.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file),
        ],
    )


def get_device() -> torch.device:
    """Get the best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class FlowEvaluator:
    """Evaluates optical flow model and saves results."""

    def __init__(
        self,
        model: nn.Module,
        output_dir: str,
        device: torch.device,
        cfg: Optional[Dict] = None,
    ):
        self.model = model
        self.output_dir = Path(output_dir)
        self.device = device
        self.cfg = cfg or {}

        # Create output subdirectories
        (self.output_dir / "flows").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "flows_hsv").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "intermediate_stages").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "gt_flows").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "gt_flows_hsv").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "errors").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "stage_convergence").mkdir(parents=True, exist_ok=True)

        # Prepare loss functions for metric computation
        self.seq_loss = SequenceLoss(
            gamma=self.cfg.get("gamma", 0.85),
            max_flow=self.cfg.get("max_flow", 400.0),
        )
        self.smooth_loss = SmoothnessLoss()
        self.ofce_loss = OFCELoss()

        # Metrics storage
        self.metrics_list: List[Dict] = []
        self.flow_magnitudes: List[float] = []
        self.normalized_stage_profiles: List[np.ndarray] = []

    def evaluate(self, dataloader: DataLoader) -> Dict[str, float]:
        """Evaluate model on dataloader."""
        self.model.eval()
        device = self.device

        aggregate_metrics = {}
        num_batches = 0

        logger.info(f"Starting evaluation on {len(dataloader)} batches...")

        with torch.no_grad():
            pbar = tqdm(dataloader, desc="Evaluating", unit="batch")

            for batch_idx, batch in enumerate(pbar):
                img1 = batch["image1"].to(device, non_blocking=True)
                img2 = batch["image2"].to(device, non_blocking=True)
                flow_gt = batch["flow"].to(device, non_blocking=True)
                valid = batch["valid"].to(device, non_blocking=True)

                # Forward pass
                out = self.model(img1, img2)
                flow_preds = out["flow_preds"]
                flow_low = out.get("flow_low", [flow_preds[-1]])

                # Process each item in batch
                B = img1.shape[0]
                for b in range(B):
                    sample_name = f"batch_{batch_idx:06d}_item_{b:02d}"

                    # Get predictions for this sample
                    flow_pred = flow_preds[-1][b]  # Final stage full-res
                    flow_stages = [fp[b] for fp in flow_preds]  # All stages

                    # Get GT
                    flow_gt_b = flow_gt[b]
                    valid_b = valid[b]

                    # Compute metrics
                    metrics = compute_metrics(flow_pred, flow_gt_b, valid_b)
                    metrics["sample_name"] = sample_name

                    # Add auxiliary losses
                    smooth_loss = self.smooth_loss(
                        flow_pred.unsqueeze(0), img1[b].unsqueeze(0)
                    ).item()
                    metrics["smoothness_loss"] = smooth_loss

                    ofce_loss = self.ofce_loss(
                        img1[b].unsqueeze(0), img2[b].unsqueeze(0),
                        flow_pred.unsqueeze(0)
                    ).item()
                    metrics["ofce_loss"] = ofce_loss

                    # Save flows (numpy .flo format)
                    flow_path = (
                        self.output_dir / "flows" / f"{sample_name}.flo"
                    )
                    write_flow(flow_path, flow_pred.cpu().numpy())

                    # Save GT flow
                    gt_path = (
                        self.output_dir / "gt_flows" / f"{sample_name}_gt.flo"
                    )
                    write_flow(gt_path, flow_gt_b.cpu().numpy())

                    # Save flow magnitudes for visualization scaling
                    mag = torch.sqrt((flow_pred ** 2).sum(dim=0))
                    self.flow_magnitudes.append(mag.max().item())

                    # Save HSV visualizations
                    flow_hsv = flow_to_hsv(flow_pred)
                    import cv2
                    cv2.imwrite(
                        str(self.output_dir / "flows_hsv" / f"{sample_name}.png"),
                        cv2.cvtColor(flow_hsv, cv2.COLOR_RGB2BGR),
                    )

                    flow_gt_hsv = flow_to_hsv(flow_gt_b)
                    cv2.imwrite(
                        str(self.output_dir / "gt_flows_hsv" / f"{sample_name}_gt.png"),
                        cv2.cvtColor(flow_gt_hsv, cv2.COLOR_RGB2BGR),
                    )

                    # Save error map
                    epe = torch.sqrt(((flow_pred - flow_gt_b) ** 2).sum(dim=0))
                    # EPE is a scalar (H, W) map — not a flow vector, so we
                    # cannot pass it to flow_to_hsv (which expects 2 channels).
                    # Clip at 5 px for display, apply a hot colormap.
                    epe_np = epe.cpu().numpy()  # (H, W)
                    epe_norm = np.clip(epe_np / 5.0 * 255.0, 0, 255).astype(np.uint8)
                    epe_bgr = cv2.applyColorMap(epe_norm, cv2.COLORMAP_HOT)
                    cv2.imwrite(
                        str(self.output_dir / "errors" / f"{sample_name}_epe.png"),
                        epe_bgr,
                    )

                    # Save stage-by-stage convergence and collect normalized profile
                    stage_epe = self._compute_stage_mean_epe(flow_stages, flow_gt_b, valid_b)
                    self._save_stage_convergence(sample_name, stage_epe)
                    self.normalized_stage_profiles.append(
                        stage_epe / max(float(stage_epe[0]), 1e-6)
                    )

                    # Save intermediate states for detailed inspection
                    self._save_intermediate_states(
                        sample_name, flow_stages, flow_gt_b, img1[b], img2[b], valid_b
                    )

                    # Accumulate metrics
                    self.metrics_list.append(metrics)
                    for k, v in metrics.items():
                        if k != "sample_name" and not math.isnan(v):
                            aggregate_metrics[k] = (
                                aggregate_metrics.get(k, 0.0) + v
                            )

                    pbar.update(1)

                num_batches += B

        # Average metrics
        for k in aggregate_metrics:
            aggregate_metrics[k] /= num_batches

        # Save detailed metrics
        metrics_path = self.output_dir / "metrics_detailed.json"
        with open(metrics_path, "w") as f:
            json.dump(self.metrics_list, f, indent=2)
        logger.info(f"Saved detailed metrics → {metrics_path}")

        # Save summary
        summary_path = self.output_dir / "metrics_summary.json"
        with open(summary_path, "w") as f:
            json.dump(aggregate_metrics, f, indent=2)
        logger.info(f"Saved summary metrics → {summary_path}")

        # Save colorwheel reference
        wheel = create_flow_colorwheel()
        import cv2
        cv2.imwrite(
            str(self.output_dir / "flow_colorwheel_reference.png"),
            cv2.cvtColor(wheel, cv2.COLOR_RGB2BGR),
        )

        # Save dataset-wide normalized convergence profile (mean ± 2 std).
        self._save_dataset_normalized_convergence()

        logger.info(f"Evaluation complete. Results saved to {self.output_dir}")
        return aggregate_metrics

    def _compute_stage_mean_epe(
        self,
        flow_stages: List[torch.Tensor],
        flow_gt: torch.Tensor,
        valid: torch.Tensor,
    ) -> np.ndarray:
        """Compute per-stage mean EPE against GT, respecting valid mask."""
        epe_stages: List[float] = []

        for flow_k in flow_stages:
            vm = valid.bool()

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
                vm = F.interpolate(
                    vm.float().unsqueeze(0).unsqueeze(0),
                    size=flow_gt.shape[-2:],
                    mode="nearest",
                ).squeeze(0).squeeze(0).bool()

            epe = torch.sqrt(((flow_k - flow_gt) ** 2).sum(dim=0))
            epe_stages.append(epe[vm].mean().item() if vm.any() else epe.mean().item())

        return np.asarray(epe_stages, dtype=np.float32)

    def _save_dataset_normalized_convergence(self) -> None:
        """Save dataset-level normalized stage convergence (mean with ±2std band)."""
        if not self.normalized_stage_profiles:
            logger.warning("No stage profiles collected; skipping dataset convergence plot")
            return

        import matplotlib.pyplot as plt

        max_k = max(profile.shape[0] for profile in self.normalized_stage_profiles)
        profile_mat = np.full((len(self.normalized_stage_profiles), max_k), np.nan, dtype=np.float32)
        for i, profile in enumerate(self.normalized_stage_profiles):
            profile_mat[i, : profile.shape[0]] = profile

        mean_profile = np.nanmean(profile_mat, axis=0)
        std_profile = np.nanstd(profile_mat, axis=0)
        counts = np.sum(~np.isnan(profile_mat), axis=0)
        lower = np.maximum(0.0, mean_profile - 2.0 * std_profile)
        upper = mean_profile + 2.0 * std_profile
        stages = np.arange(1, max_k + 1)

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(stages, mean_profile, "o-", linewidth=2, markersize=6, color="#176087", label="Mean normalized EPE")
        ax.fill_between(stages, lower, upper, color="#6ba4bf", alpha=0.30, label="\u00b12\u03c3")
        ax.set_xlabel("HQS Stage", fontsize=12)
        ax.set_ylabel("Normalized Mean EPE", fontsize=12)
        ax.set_title("Dataset-Wide Normalized Stage Convergence", fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.set_xticks(stages)
        ax.legend()

        plot_path = self.output_dir / "stage_convergence" / "dataset_normalized_convergence.png"
        plt.savefig(plot_path, dpi=120, bbox_inches="tight")
        plt.close()

        stats = {
            "normalization": "per-sample mean EPE divided by stage-1 mean EPE",
            "samples": int(len(self.normalized_stage_profiles)),
            "per_stage": [
                {
                    "stage": int(k + 1),
                    "samples": int(counts[k]),
                    "mean_normalized_epe": float(mean_profile[k]),
                    "std_normalized_epe": float(std_profile[k]),
                    "lower_2std": float(lower[k]),
                    "upper_2std": float(upper[k]),
                }
                for k in range(max_k)
            ],
        }
        stats_path = self.output_dir / "stage_convergence" / "dataset_normalized_convergence.json"
        with open(stats_path, "w") as f:
            json.dump(stats, f, indent=2)

        logger.info(f"Saved dataset normalized convergence plot → {plot_path}")
        logger.info(f"Saved dataset normalized convergence stats → {stats_path}")

    def _save_stage_convergence(
        self,
        sample_name: str,
        epe_stages: np.ndarray,
    ) -> None:
        """Save visualization of how flow improves across stages."""
        import matplotlib.pyplot as plt

        K = int(epe_stages.shape[0])

        # Create convergence plot
        fig, ax = plt.subplots(figsize=(10, 6))
        stages = list(range(1, K + 1))
        ax.plot(stages, epe_stages, "o-", linewidth=2, markersize=8)
        ax.set_xlabel("HQS Stage", fontsize=12)
        ax.set_ylabel("Mean EPE (pixels)", fontsize=12)
        ax.set_title(f"Flow Convergence: {sample_name}", fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.set_xticks(stages)

        plot_path = (
            self.output_dir / "stage_convergence" / f"{sample_name}_convergence.png"
        )
        plt.savefig(plot_path, dpi=100, bbox_inches="tight")
        plt.close()

    def _save_intermediate_states(
        self,
        sample_name: str,
        flow_stages: List[torch.Tensor],
        flow_gt: torch.Tensor,
        img1: torch.Tensor,
        img2: torch.Tensor,
        valid: torch.Tensor,
    ) -> None:
        """Save detailed intermediate states for visual inspection."""
        import cv2
        import matplotlib.pyplot as plt
        from matplotlib.gridspec import GridSpec

        K = len(flow_stages)
        cols = min(3, K)
        rows = math.ceil(K / cols) + 1  # +1 for GT row

        fig = plt.figure(figsize=(5 * cols, 4 * rows))
        gs = GridSpec(rows, cols, figure=fig, hspace=0.3, wspace=0.3)

        # Plot each stage
        for k, flow_k in enumerate(flow_stages):
            ax = fig.add_subplot(gs[k // cols, k % cols])

            # Upsample to GT resolution for visualization
            if flow_k.shape != flow_gt.shape:
                scale_h = flow_gt.shape[-2] / flow_k.shape[-2]
                scale_w = flow_gt.shape[-1] / flow_k.shape[-1]
                flow_k_up = F.interpolate(
                    flow_k.unsqueeze(0), size=flow_gt.shape[-2:],
                    mode="bilinear", align_corners=True
                ).squeeze(0)
                flow_k_up[0] *= scale_w
                flow_k_up[1] *= scale_h
            else:
                flow_k_up = flow_k

            flow_hsv = flow_to_hsv(flow_k_up)
            epe = torch.sqrt(((flow_k_up - flow_gt) ** 2).sum(dim=0))
            epe_mean = epe[valid.bool()].mean().item()

            ax.imshow(flow_hsv)
            ax.set_title(f"Stage {k + 1}\nMean EPE: {epe_mean:.3f}px", fontsize=11)
            ax.axis("off")

        # Plot GT
        ax_gt = fig.add_subplot(gs[rows - 1, :])
        flow_gt_hsv = flow_to_hsv(flow_gt)
        ax_gt.imshow(flow_gt_hsv)
        ax_gt.set_title("Ground Truth", fontsize=12)
        ax_gt.axis("off")

        plot_path = (
            self.output_dir / "intermediate_stages" /
            f"{sample_name}_stages.png"
        )
        plt.savefig(plot_path, dpi=100, bbox_inches="tight")
        plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate optical flow model")
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
        help="Output directory for results",
    )
    parser.add_argument(
        "--device", default=None,
        help="Device (cuda/mps/cpu)",
    )
    parser.add_argument(
        "--batch_size", type=int, default=4,
        help="Batch size for evaluation",
    )
    args = parser.parse_args()

    # Setup
    from omegaconf import OmegaConf
    setup_logging(args.output_dir)
    logger.info("=" * 80)
    logger.info("HQS Optical Flow - Comprehensive Evaluation")
    logger.info("=" * 80)

    device = torch.device(args.device or "cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # Load configs
    model_cfg = OmegaConf.load(args.config)
    data_cfg = OmegaConf.load(args.data_config)

    # Build model
    model = build_model(model_cfg).to(device)
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Load checkpoint
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"], strict=False)
    logger.info(f"Loaded checkpoint from {args.checkpoint}")

    # Build data
    eval_cfg = data_cfg.get("val_data") or data_cfg.get("data") or data_cfg
    eval_cfg = OmegaConf.merge(eval_cfg, {"batch_size": args.batch_size})

    eval_data = build_dataset(eval_cfg, split="val")
    eval_loader = build_dataloader(eval_data, eval_cfg, split="val")
    logger.info(f"Evaluation set: {len(eval_data)} samples")

    # Evaluate
    evaluator = FlowEvaluator(
        model,
        args.output_dir,
        device,
        cfg=model_cfg.loss if hasattr(model_cfg, "loss") else {},
    )
    metrics = evaluator.evaluate(eval_loader)

    # Log final results
    logger.info("=" * 80)
    logger.info("Final Metrics Summary")
    logger.info("=" * 80)
    for k, v in sorted(metrics.items()):
        if not math.isnan(v):
            logger.info(f"  {k:20s}: {v:10.4f}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
