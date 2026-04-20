"""Inference + visualisation script for HQSFlow.

Runs the model on a user-specified image pair (or a random sample from a
dataset) and produces a multi-panel figure saved to disk.

Outputs (all in --output_dir)
------------------------------
flow_pred.png       — Colour-coded predicted flow (Middlebury wheel)
flow_gt.png         — Colour-coded GT flow (if available)
error_map.png       — Per-pixel EPE error map (if GT available)
side_by_side.png    — Seven-panel: img1, img2, flow_gt, flow_pred, error,
                       mag_gt, mag_pred
stage_convergence.png — EPE per HQS stage (convergence plot)
stage_flows/        — Colour flow image at each HQS stage

Usage examples
--------------
# Pair of raw PNG images (no GT)
python verify.py --config configs/default.yaml \
                 --checkpoint checkpoints/hqs_flow/best.pth \
                 --image1 img1.png --image2 img2.png

# Random Sintel sample (with GT, shows error map)
python verify.py --config configs/default.yaml \
                 --checkpoint checkpoints/hqs_flow/best.pth \
                 --dataset sintel --dataset_root datasets/Sintel \
                 --sample_idx 42

# Save outputs to custom directory
python verify.py --config configs/default.yaml \
                 --checkpoint checkpoints/hqs_flow/best.pth \
                 --image1 img1.png --image2 img2.png \
                 --output_dir verify_out/run1
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from omegaconf import OmegaConf

logger = logging.getLogger("verify")


# ---------------------------------------------------------------------------
# Imports that need matplotlib to be present
# ---------------------------------------------------------------------------

try:
    import matplotlib
    matplotlib.use("Agg")          # non-interactive backend — safe on SSH / CI
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    _MPL = True
except ImportError:
    _MPL = False
    logger.warning("matplotlib not available — figures will not be saved.")


# ---------------------------------------------------------------------------
# Flow colour coding (Middlebury wheel)
# ---------------------------------------------------------------------------

def flow_to_rgb(flow: np.ndarray, max_mag: Optional[float] = None) -> np.ndarray:
    """
    Convert flow (H, W, 2) to RGB uint8 using the Middlebury colour wheel.
    If max_mag is None the magnitude is normalised to the 99th percentile.
    """
    u, v = flow[..., 0], flow[..., 1]
    mag   = np.sqrt(u ** 2 + v ** 2)
    angle = np.arctan2(-v, -u) / np.pi        # in [-1, 1]

    if max_mag is None:
        max_mag = float(np.percentile(mag, 99)) + 1e-6

    mag_norm = np.clip(mag / max_mag, 0, 1)

    # Hue from angle, saturation/value from magnitude
    hue = (angle + 1) / 2                     # [0, 1]
    sat = mag_norm
    val = np.ones_like(mag_norm)

    hsv = np.stack([hue, sat, val], axis=-1).astype(np.float32)
    rgb = mcolors.hsv_to_rgb(hsv)             # [0, 1]
    return (rgb * 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Image loading
# ---------------------------------------------------------------------------

def load_image(path: str) -> torch.Tensor:
    """Load an image from *path* and return float32 tensor (C, H, W) in [0,1]."""
    from PIL import Image
    img = Image.open(path).convert("RGB")
    arr = np.array(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1)   # (3, H, W)


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_model(
    model: nn.Module,
    img1: torch.Tensor,
    img2: torch.Tensor,
    device: torch.device,
) -> dict:
    """
    Forward pass.  Returns dict with:
      flow_pred   — (2, H, W) float32 tensor on CPU
      flow_preds  — list of (2, H, W) tensors per stage, on CPU
    """
    x1 = img1.unsqueeze(0).to(device)
    x2 = img2.unsqueeze(0).to(device)

    model.eval()
    out = model(x1, x2)

    return {
        "flow_pred":  out["flow_preds"][-1][0].cpu(),
        "flow_preds": [f[0].cpu() for f in out["flow_preds"]],
    }


# ---------------------------------------------------------------------------
# Visualisation helpers
# ---------------------------------------------------------------------------

def _tensor_to_np(t: torch.Tensor) -> np.ndarray:
    """(C, H, W) → (H, W, C) numpy uint8."""
    arr = t.permute(1, 2, 0).numpy()
    return np.clip(arr * 255, 0, 255).astype(np.uint8)


def _flow_to_np(t: torch.Tensor) -> np.ndarray:
    """(2, H, W) tensor → (H, W, 2) numpy float32."""
    return t.permute(1, 2, 0).numpy()


def epe_map(pred: torch.Tensor, gt: torch.Tensor) -> np.ndarray:
    """Per-pixel EPE as (H, W) float32 array."""
    diff = pred - gt
    return torch.sqrt((diff ** 2).sum(dim=0)).numpy()


# ---------------------------------------------------------------------------
# Figure: side-by-side panels
# ---------------------------------------------------------------------------

def save_side_by_side(
    img1_np: np.ndarray,
    img2_np: np.ndarray,
    flow_pred_np: np.ndarray,
    flow_gt_np: Optional[np.ndarray],
    epe_np: Optional[np.ndarray],
    path: str,
) -> None:
    has_gt = flow_gt_np is not None

    max_mag = None
    if has_gt:
        gt_mag = np.sqrt((flow_gt_np ** 2).sum(-1))
        max_mag = float(np.percentile(gt_mag, 99)) + 1e-6

    pred_rgb = flow_to_rgb(flow_pred_np, max_mag)

    n_cols = 5 if has_gt else 3
    fig, axes = plt.subplots(1, n_cols, figsize=(4 * n_cols, 4))
    if n_cols == 1:
        axes = [axes]

    axes[0].imshow(img1_np);        axes[0].set_title("Image 1");     axes[0].axis("off")
    axes[1].imshow(img2_np);        axes[1].set_title("Image 2");     axes[1].axis("off")
    axes[2].imshow(pred_rgb);       axes[2].set_title("Pred Flow");   axes[2].axis("off")

    if has_gt:
        gt_rgb = flow_to_rgb(flow_gt_np, max_mag)
        axes[3].imshow(gt_rgb);     axes[3].set_title("GT Flow");     axes[3].axis("off")
        im = axes[4].imshow(epe_np, cmap="hot", vmin=0,
                             vmax=float(np.percentile(epe_np, 99)))
        axes[4].set_title("EPE map"); axes[4].axis("off")
        plt.colorbar(im, ax=axes[4], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Side-by-side figure → {path}")


# ---------------------------------------------------------------------------
# Figure: stage convergence
# ---------------------------------------------------------------------------

def save_convergence_plot(
    stage_flows: list,
    gt: Optional[torch.Tensor],
    path: str,
) -> None:
    """Plot EPE vs stage index. If gt unavailable, plot mean flow magnitude."""
    stages = list(range(1, len(stage_flows) + 1))

    if gt is not None:
        epes = []
        for f in stage_flows:
            diff = f - gt
            ep = torch.sqrt((diff ** 2).sum(0)).mean().item()
            epes.append(ep)
        ylabel, values = "Mean EPE", epes
        title = "HQS Stage Convergence (EPE)"
    else:
        mags = [torch.sqrt((f ** 2).sum(0)).mean().item() for f in stage_flows]
        ylabel, values = "Mean Flow Magnitude (px)", mags
        title = "HQS Stage Convergence (Flow Magnitude)"

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(stages, values, "o-", linewidth=2, markersize=6, color="#2563EB")
    ax.set_xlabel("HQS Stage")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(stages)
    ax.grid(True, alpha=0.3)
    if gt is not None:
        ax.axhline(values[-1], color="gray", linestyle="--",
                   alpha=0.5, label=f"Final: {values[-1]:.3f}")
        ax.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    logger.info(f"Convergence plot → {path}")


# ---------------------------------------------------------------------------
# Save per-stage flow images
# ---------------------------------------------------------------------------

def save_stage_flows(
    stage_flows: list,
    out_dir: str,
    max_mag: Optional[float] = None,
) -> None:
    os.makedirs(out_dir, exist_ok=True)
    for i, f in enumerate(stage_flows):
        flow_np = _flow_to_np(f)
        rgb     = flow_to_rgb(flow_np, max_mag)
        img_path = os.path.join(out_dir, f"stage_{i+1:02d}.png")

        fig, ax = plt.subplots(figsize=(5, 4))
        ax.imshow(rgb)
        ax.set_title(f"Stage {i+1}")
        ax.axis("off")
        plt.tight_layout()
        plt.savefig(img_path, dpi=120, bbox_inches="tight")
        plt.close()
    logger.info(f"Stage flow images → {out_dir}/")


# ---------------------------------------------------------------------------
# Dataset sample loading
# ---------------------------------------------------------------------------

def get_dataset_sample(
    dataset_name: str,
    dataset_root: str,
    sample_idx: int,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Return (img1, img2, flow_gt, valid) from a dataset. All (C/2, H, W) tensors."""
    from data import SintelDataset, KITTIDataset, SpringDataset

    if dataset_name == "sintel":
        ds = SintelDataset(root=dataset_root, split="val", dstype="clean")
    elif dataset_name == "kitti":
        ds = KITTIDataset(root=dataset_root, split="val")
    elif dataset_name == "spring":
        ds = SpringDataset(root=dataset_root, split="val")
    else:
        raise ValueError(f"Unknown dataset: {dataset_name!r}")

    sample = ds[sample_idx % len(ds)]
    img1  = sample["image1"].float() / 255.0 if sample["image1"].dtype == torch.uint8 else sample["image1"]
    img2  = sample["image2"].float() / 255.0 if sample["image2"].dtype == torch.uint8 else sample["image2"]
    flow  = sample.get("flow")
    valid = sample.get("valid")
    return img1, img2, flow, valid


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    logging.basicConfig(
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%H:%M:%S",
        level=logging.INFO,
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    parser = argparse.ArgumentParser(description="HQSFlow inference visualisation")
    parser.add_argument("--config",     required=True,  help="Config YAML path")
    parser.add_argument("--checkpoint", required=True,  help="Checkpoint .pth path")
    parser.add_argument("--output_dir", default="verify_out", help="Output directory")
    parser.add_argument("--device",     default=None)
    # Image-pair mode
    parser.add_argument("--image1",     default=None)
    parser.add_argument("--image2",     default=None)
    # Dataset sample mode
    parser.add_argument("--dataset",      default=None, choices=["sintel", "kitti", "spring"])
    parser.add_argument("--dataset_root", default=None)
    parser.add_argument("--sample_idx",   type=int, default=0)
    # Options
    parser.add_argument("--save_stages", action="store_true",
                        help="Save individual per-stage flow images")
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()

    if not _MPL:
        logger.error("matplotlib is required for verify.py. "
                     "Install it: pip install matplotlib")
        sys.exit(1)

    # ── Config ───────────────────────────────────────────────────────────────
    cfg = OmegaConf.load(args.config)
    if args.overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(args.overrides))

    # ── Device ───────────────────────────────────────────────────────────────
    from engine.trainer import get_device, amp_enabled
    device = get_device(args.device)
    logger.info(f"Device: {device}")

    # ── Model ────────────────────────────────────────────────────────────────
    from models import build_model
    model = build_model(cfg).to(device)
    ckpt  = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt.get("model", ckpt), strict=True)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model loaded  ({n_params:,} params)")

    # ── Load images ──────────────────────────────────────────────────────────
    flow_gt: Optional[torch.Tensor] = None
    valid:   Optional[torch.Tensor] = None

    if args.image1 and args.image2:
        img1 = load_image(args.image1)
        img2 = load_image(args.image2)
    elif args.dataset:
        root = args.dataset_root
        if not root:
            raise ValueError("--dataset_root is required when using --dataset")
        img1, img2, flow_gt, valid = get_dataset_sample(
            args.dataset, root, args.sample_idx
        )
        logger.info(f"Loaded sample {args.sample_idx} from {args.dataset}")
    else:
        parser.error("Provide either --image1/--image2 or --dataset/--dataset_root")

    # ── Inference ────────────────────────────────────────────────────────────
    result = run_model(model, img1, img2, device)
    flow_pred  = result["flow_pred"]    # (2, H, W)
    stage_preds = result["flow_preds"]  # list of (2, H, W)
    logger.info(f"Inference done  ({len(stage_preds)} stages)")

    # ── Prepare numpy arrays ─────────────────────────────────────────────────
    img1_np = _tensor_to_np(img1)
    img2_np = _tensor_to_np(img2)
    pred_np = _flow_to_np(flow_pred)
    gt_np   = _flow_to_np(flow_gt) if flow_gt is not None else None
    err_np  = epe_map(flow_pred, flow_gt) if flow_gt is not None else None

    # ── Compute summary stats ────────────────────────────────────────────────
    if flow_gt is not None:
        from utils import compute_metrics
        mask = valid if valid is not None else torch.ones(
            flow_pred.shape[1:], dtype=torch.bool
        )
        metrics = compute_metrics(flow_pred, flow_gt, mask)
        logger.info(
            "Metrics: " + "  ".join(f"{k}={v:.4f}" for k, v in metrics.items())
        )

    # ── Save figures ─────────────────────────────────────────────────────────
    os.makedirs(args.output_dir, exist_ok=True)

    # Side-by-side
    sbs_path = os.path.join(args.output_dir, "side_by_side.png")
    save_side_by_side(img1_np, img2_np, pred_np, gt_np, err_np, sbs_path)

    # Individual images
    max_mag = None
    if gt_np is not None:
        gt_mag  = np.sqrt((gt_np ** 2).sum(-1))
        max_mag = float(np.percentile(gt_mag, 99)) + 1e-6

    for name, flow_arr in (("flow_pred", pred_np),
                           *([("flow_gt", gt_np)] if gt_np is not None else [])):
        rgb = flow_to_rgb(flow_arr, max_mag)
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.imshow(rgb);  ax.axis("off")
        ax.set_title(name.replace("_", " ").title())
        plt.tight_layout()
        plt.savefig(os.path.join(args.output_dir, f"{name}.png"),
                    dpi=150, bbox_inches="tight")
        plt.close()

    if err_np is not None:
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(err_np, cmap="hot", vmin=0,
                       vmax=float(np.percentile(err_np, 99)))
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title("EPE Error Map");  ax.axis("off")
        plt.tight_layout()
        plt.savefig(os.path.join(args.output_dir, "error_map.png"),
                    dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"Error map → {args.output_dir}/error_map.png")

    # Convergence plot
    conv_path = os.path.join(args.output_dir, "stage_convergence.png")
    save_convergence_plot(stage_preds, flow_gt, conv_path)

    # Per-stage flow images (optional)
    if args.save_stages:
        stages_dir = os.path.join(args.output_dir, "stage_flows")
        save_stage_flows(stage_preds, stages_dir, max_mag)

    logger.info(f"\nAll outputs saved to: {args.output_dir}/")


if __name__ == "__main__":
    main()
