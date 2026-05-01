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
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
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


_WORKER_SMOOTH_LOSS: Optional[SmoothnessLoss] = None
_WORKER_OFCE_LOSS: Optional[OFCELoss] = None


def _get_worker_losses() -> Tuple[SmoothnessLoss, OFCELoss]:
    """Lazily initialize worker-local loss modules."""
    global _WORKER_SMOOTH_LOSS
    global _WORKER_OFCE_LOSS

    if _WORKER_SMOOTH_LOSS is None:
        _WORKER_SMOOTH_LOSS = SmoothnessLoss()
    if _WORKER_OFCE_LOSS is None:
        _WORKER_OFCE_LOSS = OFCELoss()

    return _WORKER_SMOOTH_LOSS, _WORKER_OFCE_LOSS


def _compute_stage_mean_epe_np(
    flow_stages: List[np.ndarray],
    flow_gt: np.ndarray,
    valid: np.ndarray,
) -> np.ndarray:
    """Compute per-stage mean EPE against GT, respecting valid mask."""
    flow_gt_t = torch.from_numpy(flow_gt)
    valid_t = torch.from_numpy(valid).bool()
    epe_stages: List[float] = []

    for flow_k_np in flow_stages:
        flow_k_t = torch.from_numpy(flow_k_np)
        vm = valid_t

        if flow_k_t.shape != flow_gt_t.shape:
            scale_h = flow_gt_t.shape[-2] / flow_k_t.shape[-2]
            scale_w = flow_gt_t.shape[-1] / flow_k_t.shape[-1]
            flow_k_t = F.interpolate(
                flow_k_t.unsqueeze(0),
                size=flow_gt_t.shape[-2:],
                mode="bilinear",
                align_corners=True,
            ).squeeze(0)
            flow_k_t[0] *= scale_w
            flow_k_t[1] *= scale_h
            vm = F.interpolate(
                vm.float().unsqueeze(0).unsqueeze(0),
                size=flow_gt_t.shape[-2:],
                mode="nearest",
            ).squeeze(0).squeeze(0).bool()

        epe = torch.sqrt(((flow_k_t - flow_gt_t) ** 2).sum(dim=0))
        epe_stages.append(epe[vm].mean().item() if vm.any() else epe.mean().item())

    return np.asarray(epe_stages, dtype=np.float32)


def _save_stage_convergence_plot(output_dir: str, sample_name: str, epe_stages: np.ndarray) -> None:
    """Save visualization of how flow improves across stages."""
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    K = int(epe_stages.shape[0])
    stages = list(range(1, K + 1))

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(stages, epe_stages, "o-", linewidth=2, markersize=8)
    ax.set_xlabel("HQS Stage", fontsize=12)
    ax.set_ylabel("Mean EPE (pixels)", fontsize=12)
    ax.set_title(f"Flow Convergence: {sample_name}", fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(stages)

    plot_path = Path(output_dir) / "stage_convergence" / f"{sample_name}_convergence.png"
    plt.savefig(plot_path, dpi=100, bbox_inches="tight")
    plt.close()


def _save_intermediate_states_plot(
    output_dir: str,
    sample_name: str,
    flow_stages: List[np.ndarray],
    flow_gt: np.ndarray,
    valid: np.ndarray,
) -> None:
    """Save detailed intermediate flow states for visual inspection."""
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    flow_gt_t = torch.from_numpy(flow_gt)
    valid_t = torch.from_numpy(valid).bool()

    K = len(flow_stages)
    cols = min(3, K)
    rows = math.ceil(K / cols) + 1

    fig = plt.figure(figsize=(5 * cols, 4 * rows))
    gs = GridSpec(rows, cols, figure=fig, hspace=0.3, wspace=0.3)

    for k, flow_k_np in enumerate(flow_stages):
        ax = fig.add_subplot(gs[k // cols, k % cols])
        flow_k_t = torch.from_numpy(flow_k_np)

        if flow_k_t.shape != flow_gt_t.shape:
            scale_h = flow_gt_t.shape[-2] / flow_k_t.shape[-2]
            scale_w = flow_gt_t.shape[-1] / flow_k_t.shape[-1]
            flow_k_up = F.interpolate(
                flow_k_t.unsqueeze(0),
                size=flow_gt_t.shape[-2:],
                mode="bilinear",
                align_corners=True,
            ).squeeze(0)
            flow_k_up[0] *= scale_w
            flow_k_up[1] *= scale_h
        else:
            flow_k_up = flow_k_t

        flow_hsv = flow_to_hsv(flow_k_up)
        epe = torch.sqrt(((flow_k_up - flow_gt_t) ** 2).sum(dim=0))
        epe_mean = epe[valid_t].mean().item() if valid_t.any() else epe.mean().item()

        ax.imshow(flow_hsv)
        ax.set_title(f"Stage {k + 1}\\nMean EPE: {epe_mean:.3f}px", fontsize=11)
        ax.axis("off")

    ax_gt = fig.add_subplot(gs[rows - 1, :])
    ax_gt.imshow(flow_to_hsv(flow_gt_t))
    ax_gt.set_title("Ground Truth", fontsize=12)
    ax_gt.axis("off")

    plot_path = Path(output_dir) / "intermediate_stages" / f"{sample_name}_stages.png"
    plt.savefig(plot_path, dpi=100, bbox_inches="tight")
    plt.close()


def _process_sample_task(task: Dict) -> Dict:
    """Worker task: compute metrics and save all per-sample artifacts."""
    import cv2

    smooth_loss_fn, ofce_loss_fn = _get_worker_losses()

    sample_name = task["sample_name"]
    output_dir = Path(task["output_dir"])

    flow_pred_t = torch.from_numpy(task["flow_pred"])
    flow_gt_t = torch.from_numpy(task["flow_gt"])
    valid_t = torch.from_numpy(task["valid"])
    img1_t = torch.from_numpy(task["img1"])
    img2_t = torch.from_numpy(task["img2"])
    flow_stages_np = task["flow_stages"]

    metrics = compute_metrics(flow_pred_t, flow_gt_t, valid_t)
    metrics["sample_name"] = sample_name

    metrics["smoothness_loss"] = smooth_loss_fn(
        flow_pred_t.unsqueeze(0), img1_t.unsqueeze(0)
    ).item()
    metrics["ofce_loss"] = ofce_loss_fn(
        img1_t.unsqueeze(0), img2_t.unsqueeze(0), flow_pred_t.unsqueeze(0)
    ).item()

    write_flow(output_dir / "flows" / f"{sample_name}.flo", flow_pred_t.numpy())
    write_flow(output_dir / "gt_flows" / f"{sample_name}_gt.flo", flow_gt_t.numpy())

    mag = torch.sqrt((flow_pred_t ** 2).sum(dim=0))
    flow_mag_max = float(mag.max().item())

    flow_hsv = flow_to_hsv(flow_pred_t)
    cv2.imwrite(
        str(output_dir / "flows_hsv" / f"{sample_name}.png"),
        cv2.cvtColor(flow_hsv, cv2.COLOR_RGB2BGR),
    )

    flow_gt_hsv = flow_to_hsv(flow_gt_t)
    cv2.imwrite(
        str(output_dir / "gt_flows_hsv" / f"{sample_name}_gt.png"),
        cv2.cvtColor(flow_gt_hsv, cv2.COLOR_RGB2BGR),
    )

    epe = torch.sqrt(((flow_pred_t - flow_gt_t) ** 2).sum(dim=0))
    epe_np = epe.numpy()
    epe_norm = np.clip(epe_np / 5.0 * 255.0, 0, 255).astype(np.uint8)
    epe_bgr = cv2.applyColorMap(epe_norm, cv2.COLORMAP_HOT)
    cv2.imwrite(str(output_dir / "errors" / f"{sample_name}_epe.png"), epe_bgr)

    stage_epe = _compute_stage_mean_epe_np(flow_stages_np, task["flow_gt"], task["valid"])
    _save_stage_convergence_plot(str(output_dir), sample_name, stage_epe)
    _save_intermediate_states_plot(
        str(output_dir),
        sample_name,
        flow_stages_np,
        task["flow_gt"],
        task["valid"],
    )

    norm_profile = stage_epe / max(float(stage_epe[0]), 1e-6)

    return {
        "metrics": metrics,
        "flow_mag_max": flow_mag_max,
        "normalized_stage_profile": norm_profile,
    }


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
        postproc_workers: int = 1,
    ):
        self.model = model
        self.output_dir = Path(output_dir)
        self.device = device
        self.cfg = cfg or {}
        self.postproc_workers = max(1, int(postproc_workers))

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

        logger.info(
            f"Starting evaluation on {len(dataloader)} batches with "
            f"{self.postproc_workers} post-processing worker(s)..."
        )

        dataset_len = len(dataloader.dataset) if hasattr(dataloader, "dataset") else None
        pbar = tqdm(total=dataset_len, desc="Evaluating", unit="sample")

        executor: Optional[ProcessPoolExecutor] = None
        if self.postproc_workers > 1:
            executor = ProcessPoolExecutor(
                max_workers=self.postproc_workers,
                mp_context=mp.get_context("spawn"),
            )

        try:
            with torch.no_grad():
                for batch_idx, batch in enumerate(dataloader):
                img1 = batch["image1"].to(device, non_blocking=True)
                img2 = batch["image2"].to(device, non_blocking=True)
                flow_gt = batch["flow"].to(device, non_blocking=True)
                valid = batch["valid"].to(device, non_blocking=True)

                # Forward pass
                out = self.model(img1, img2)
                flow_preds = out["flow_preds"]
                flow_low = out.get("flow_low", [flow_preds[-1]])

                # Move batch tensors once to CPU for per-sample post-processing.
                img1_cpu = img1.detach().cpu()
                img2_cpu = img2.detach().cpu()
                flow_gt_cpu = flow_gt.detach().cpu()
                valid_cpu = valid.detach().cpu()
                flow_preds_cpu = [fp.detach().cpu() for fp in flow_preds]

                B = img1.shape[0]
                tasks: List[Dict] = []
                for b in range(B):
                    sample_name = f"batch_{batch_idx:06d}_item_{b:02d}"

                    tasks.append(
                        {
                            "sample_name": sample_name,
                            "output_dir": str(self.output_dir),
                            "flow_pred": flow_preds_cpu[-1][b].numpy(),
                            "flow_stages": [fp[b].numpy() for fp in flow_preds_cpu],
                            "flow_gt": flow_gt_cpu[b].numpy(),
                            "valid": valid_cpu[b].numpy(),
                            "img1": img1_cpu[b].numpy(),
                            "img2": img2_cpu[b].numpy(),
                        }
                    )

                if executor is None:
                    results = [_process_sample_task(task) for task in tasks]
                else:
                    futures = [executor.submit(_process_sample_task, task) for task in tasks]
                    results = [f.result() for f in futures]

                for result in results:
                    metrics = result["metrics"]
                    self.flow_magnitudes.append(result["flow_mag_max"])
                    self.normalized_stage_profiles.append(result["normalized_stage_profile"])

                    self.metrics_list.append(metrics)
                    for k, v in metrics.items():
                        if k != "sample_name" and not math.isnan(v):
                            aggregate_metrics[k] = (
                                aggregate_metrics.get(k, 0.0) + v
                            )

                pbar.update(B)

                num_batches += B
        finally:
            pbar.close()
            if executor is not None:
                executor.shutdown(wait=True)

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
    parser.add_argument(
        "--postproc_workers", type=int, default=None,
        help="Number of CPU workers for per-sample post-processing. "
             "Defaults to max(1, cpu_count - 1). Use 1 to disable parallelism.",
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
    default_workers = max(1, (os.cpu_count() or 1) - 1)
    postproc_workers = default_workers if args.postproc_workers is None else max(1, args.postproc_workers)
    logger.info(f"Post-processing workers: {postproc_workers}")

    evaluator = FlowEvaluator(
        model,
        args.output_dir,
        device,
        cfg=model_cfg.loss if hasattr(model_cfg, "loss") else {},
        postproc_workers=postproc_workers,
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
