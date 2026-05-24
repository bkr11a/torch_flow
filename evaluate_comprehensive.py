#!/usr/bin/env python3
"""Comprehensive evaluation and experiment reporting for optical flow models.

This script evaluates a checkpoint and produces:
1. Per-sample artifacts (flow files, HSV visualizations, error maps)
2. Aggregate metrics (including occlusion-boundary metrics when available)
3. Per-iteration diagnostics tables and plots
4. Qualitative figure panels matching experiment-report expectations
5. Auto-generated experiment markdown report (template-style)
6. TensorFlow-like model summary with per-layer parameter counts
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import logging
import math
import multiprocessing as mp
import os
import platform
import re
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from data import build_dataloader, build_dataset
from losses import OFCELoss, SequenceLoss, SmoothnessLoss
from models import build_model
from utils import compute_metrics, create_flow_colorwheel, flow_to_hsv, write_flow


logger = logging.getLogger(__name__)

_WORKER_SMOOTH_LOSS: Optional[SmoothnessLoss] = None
_WORKER_OFCE_LOSS: Optional[OFCELoss] = None


def _get_worker_losses() -> Tuple[SmoothnessLoss, OFCELoss]:
    global _WORKER_SMOOTH_LOSS
    global _WORKER_OFCE_LOSS
    if _WORKER_SMOOTH_LOSS is None:
        _WORKER_SMOOTH_LOSS = SmoothnessLoss()
    if _WORKER_OFCE_LOSS is None:
        _WORKER_OFCE_LOSS = OFCELoss()
    return _WORKER_SMOOTH_LOSS, _WORKER_OFCE_LOSS


def _safe_float(v: Any, default: float = float("nan")) -> float:
    try:
        x = float(v)
        return x
    except Exception:
        return default


def _extract_int_from_stem(stem: str) -> int:
    m = re.findall(r"\d+", stem)
    if not m:
        return -1
    return int(m[-1])


def _resolve_dataset_record(dataset: Any, global_index: int) -> Optional[Dict[str, Any]]:
    """Resolve raw sample record dict from dataset or ConcatDataset."""
    if hasattr(dataset, "_samples"):
        samples = getattr(dataset, "_samples")
        if 0 <= global_index < len(samples):
            return samples[global_index]
        return None

    if hasattr(dataset, "datasets") and hasattr(dataset, "cumulative_sizes"):
        cum = dataset.cumulative_sizes
        sub_idx = 0
        while sub_idx < len(cum) and global_index >= cum[sub_idx]:
            sub_idx += 1
        if sub_idx >= len(dataset.datasets):
            return None
        prev = 0 if sub_idx == 0 else cum[sub_idx - 1]
        local = global_index - prev
        return _resolve_dataset_record(dataset.datasets[sub_idx], local)

    return None


def _infer_scene_and_sample(record: Optional[Dict[str, Any]], fallback_name: str) -> Dict[str, Any]:
    """Infer scene folder and stable sample id from source file paths."""
    if record is None:
        return {
            "scene": "",
            "sample_id": fallback_name,
            "frame1": fallback_name,
            "frame2": fallback_name,
            "sort_key": -1,
            "has_scene": False,
        }

    img1 = record.get("image1")
    img2 = record.get("image2")
    if not img1:
        return {
            "scene": "",
            "sample_id": fallback_name,
            "frame1": fallback_name,
            "frame2": fallback_name,
            "sort_key": -1,
            "has_scene": False,
        }

    p1 = Path(img1)
    p2 = Path(img2) if img2 else p1
    frame1 = p1.stem
    frame2 = p2.stem

    parent = p1.parent.name
    grand = p1.parent.parent.name if p1.parent.parent is not None else ""
    scene = parent

    if parent in {
        "left", "right", "image_2", "data", "frame_left", "frame_right",
        "flow_FW_left", "flow_BW_left", "flow_FW_right", "flow_BW_right",
    } and grand:
        scene = grand

    if scene in {"clean", "final", "flow", "training", "test", "train", "testing"}:
        scene = ""

    has_scene = bool(scene)

    sample_id = f"{frame1}__{frame2}"
    sample_id = sample_id.replace("/", "_")
    return {
        "scene": scene,
        "sample_id": sample_id,
        "frame1": frame1,
        "frame2": frame2,
        "sort_key": _extract_int_from_stem(frame1),
        "has_scene": has_scene,
    }


def _scene_subdir(base: Path, scene: str) -> Path:
    return base / scene if scene else base


def _qualitative_selection(request: str, dataset_len: int, seed: int) -> Optional[set]:
    req = str(request).strip().lower()
    if req == "all":
        return None

    n = int(req)
    if n <= 0:
        return set()
    n = min(n, dataset_len)

    g = np.random.default_rng(seed)
    chosen = g.choice(dataset_len, size=n, replace=False)
    return set(int(x) for x in chosen.tolist())


def _draw_flow_arrows(flow_xy: np.ndarray, background_rgb: np.ndarray, step: int = 16, scale: float = 1.0) -> np.ndarray:
    """Render quiver-like arrows from a regular pixel grid over background image."""
    import cv2

    h, w, _ = flow_xy.shape
    vis = np.clip(background_rgb * 255.0, 0, 255).astype(np.uint8).copy()

    for y in range(step // 2, h, step):
        for x in range(step // 2, w, step):
            dx = float(flow_xy[y, x, 0]) * scale
            dy = float(flow_xy[y, x, 1]) * scale
            x2 = int(round(x + dx))
            y2 = int(round(y + dy))
            if x2 < 0 or x2 >= w or y2 < 0 or y2 >= h:
                continue
            cv2.arrowedLine(vis, (x, y), (x2, y2), (255, 255, 255), 1, tipLength=0.28)

    return vis


def _build_scene_video(scene_dir: Path, output_path: Path, fps: int = 30) -> bool:
    """Create scene video grid: I1 | I2 / GT flow | Pred flow."""
    import cv2

    img1_dir = scene_dir / "image1"
    img2_dir = scene_dir / "image2"
    gt_dir = scene_dir / "gt_flow_hsv"
    pred_dir = scene_dir / "pred_flow_hsv"
    if not (img1_dir.exists() and img2_dir.exists() and gt_dir.exists() and pred_dir.exists()):
        return False

    frame_files = sorted([p.name for p in img1_dir.glob("*.png")])
    if not frame_files:
        return False

    first = cv2.imread(str(img1_dir / frame_files[0]))
    if first is None:
        return False
    h, w = first.shape[:2]
    out_h, out_w = h * 2, w * 2

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(output_path), fourcc, float(fps), (out_w, out_h))

    for name in frame_files:
        i1 = cv2.imread(str(img1_dir / name))
        i2 = cv2.imread(str(img2_dir / name))
        gt = cv2.imread(str(gt_dir / name))
        pr = cv2.imread(str(pred_dir / name))
        if any(x is None for x in (i1, i2, gt, pr)):
            continue

        i1 = cv2.resize(i1, (w, h), interpolation=cv2.INTER_LINEAR)
        i2 = cv2.resize(i2, (w, h), interpolation=cv2.INTER_LINEAR)
        gt = cv2.resize(gt, (w, h), interpolation=cv2.INTER_LINEAR)
        pr = cv2.resize(pr, (w, h), interpolation=cv2.INTER_LINEAR)

        top = np.concatenate([i1, i2], axis=1)
        bot = np.concatenate([gt, pr], axis=1)
        frame = np.concatenate([top, bot], axis=0)
        writer.write(frame)

    writer.release()
    return True


def _to_rgb_np(img_chw: np.ndarray) -> np.ndarray:
    img = np.transpose(img_chw, (1, 2, 0))
    img = np.clip(img, 0.0, 1.0)
    return img


def _to_uint8_gray(x: np.ndarray, norm_max: float = 1.0) -> np.ndarray:
    y = np.clip(x / max(norm_max, 1e-6), 0.0, 1.0)
    return (255.0 * y).astype(np.uint8)


def _resize_flow_like(flow: torch.Tensor, out_h: int, out_w: int) -> torch.Tensor:
    in_h, in_w = flow.shape[-2:]
    if in_h == out_h and in_w == out_w:
        return flow
    out = F.interpolate(flow.unsqueeze(0), size=(out_h, out_w), mode="bilinear", align_corners=True).squeeze(0)
    out[0] *= float(out_w) / float(in_w)
    out[1] *= float(out_h) / float(in_h)
    return out


def _warp_xy(img: torch.Tensor, flow_xy: torch.Tensor) -> torch.Tensor:
    """Warp image with flow in [dx, dy] order."""
    b, _, h, w = img.shape
    ys, xs = torch.meshgrid(
        torch.arange(h, device=img.device, dtype=img.dtype),
        torch.arange(w, device=img.device, dtype=img.dtype),
        indexing="ij",
    )
    ys = ys.view(1, h, w).expand(b, -1, -1)
    xs = xs.view(1, h, w).expand(b, -1, -1)

    qx = xs + flow_xy[:, 0]
    qy = ys + flow_xy[:, 1]

    gx = 2.0 * qx / (w - 1) - 1.0 if w > 1 else torch.zeros_like(qx)
    gy = 2.0 * qy / (h - 1) - 1.0 if h > 1 else torch.zeros_like(qy)
    grid = torch.stack([gx, gy], dim=-1)
    return F.grid_sample(img, grid, mode="bilinear", padding_mode="border", align_corners=True)


def _compute_stage_mean_epe_np(flow_stages: List[np.ndarray], flow_gt: np.ndarray, valid: np.ndarray) -> np.ndarray:
    flow_gt_t = torch.from_numpy(flow_gt)
    valid_t = torch.from_numpy(valid).bool()
    epe_stages: List[float] = []

    for flow_k_np in flow_stages:
        flow_k_t = torch.from_numpy(flow_k_np)
        vm = valid_t

        if flow_k_t.shape != flow_gt_t.shape:
            flow_k_t = _resize_flow_like(flow_k_t, flow_gt_t.shape[-2], flow_gt_t.shape[-1])
            vm = F.interpolate(
                vm.float().unsqueeze(0).unsqueeze(0),
                size=flow_gt_t.shape[-2:],
                mode="nearest",
            ).squeeze(0).squeeze(0).bool()

        epe = torch.sqrt(((flow_k_t - flow_gt_t) ** 2).sum(dim=0))
        epe_stages.append(epe[vm].mean().item() if vm.any() else epe.mean().item())

    return np.asarray(epe_stages, dtype=np.float32)


def _compute_iteration_metrics_np(
    flow_stages: List[np.ndarray],
    flow_gt: np.ndarray,
    valid: np.ndarray,
    img1: np.ndarray,
    img2: np.ndarray,
    aux_stages: Optional[List[np.ndarray]] = None,
    delta_stages: Optional[List[np.ndarray]] = None,
    coupling_stages: Optional[List[np.ndarray]] = None,
) -> Dict[str, List[float]]:
    flow_gt_t = torch.from_numpy(flow_gt)
    valid_t = torch.from_numpy(valid).bool()
    img1_t = torch.from_numpy(img1)
    img2_t = torch.from_numpy(img2)

    epe_list: List[float] = []
    photo_resid_list: List[float] = []
    update_norm_list: List[float] = []
    w_minus_q_norm_list: List[float] = []
    delta_norm_list: List[float] = []

    prev_flow: Optional[torch.Tensor] = None
    for k, flow_k_np in enumerate(flow_stages):
        flow_k = torch.from_numpy(flow_k_np)
        flow_k = _resize_flow_like(flow_k, flow_gt_t.shape[-2], flow_gt_t.shape[-1])

        epe_map = torch.sqrt(((flow_k - flow_gt_t) ** 2).sum(dim=0))
        epe_k = epe_map[valid_t].mean().item() if valid_t.any() else epe_map.mean().item()
        epe_list.append(float(epe_k))

        warped = _warp_xy(img2_t.unsqueeze(0), flow_k.unsqueeze(0)).squeeze(0)
        photo_resid = (warped - img1_t).abs().mean(dim=0)
        photo_resid_list.append(float(photo_resid.mean().item()))

        if prev_flow is None:
            update_norm_list.append(0.0)
            delta_norm_list.append(0.0)
        else:
            update = torch.sqrt(((flow_k - prev_flow) ** 2).sum(dim=0))
            update_norm_list.append(float(update.mean().item()))
            if delta_stages is not None and k < len(delta_stages):
                d = torch.from_numpy(delta_stages[k])
                d = _resize_flow_like(d, flow_gt_t.shape[-2], flow_gt_t.shape[-1])
                dnorm = torch.sqrt((d ** 2).sum(dim=0)).mean().item()
                delta_norm_list.append(float(dnorm))
            else:
                delta_norm_list.append(float(update.mean().item()))

        if coupling_stages is not None and k < len(coupling_stages):
            c = torch.from_numpy(coupling_stages[k])
            c = _resize_flow_like(c, flow_gt_t.shape[-2], flow_gt_t.shape[-1])
            c_norm = torch.sqrt((c ** 2).sum(dim=0)).mean().item()
            w_minus_q_norm_list.append(float(c_norm))
        elif aux_stages is not None and k < len(aux_stages):
            a = torch.from_numpy(aux_stages[k])
            a = _resize_flow_like(a, flow_gt_t.shape[-2], flow_gt_t.shape[-1])
            wq = torch.sqrt(((flow_k - a) ** 2).sum(dim=0)).mean().item()
            w_minus_q_norm_list.append(float(wq))
        else:
            w_minus_q_norm_list.append(float("nan"))

        prev_flow = flow_k

    return {
        "epe": epe_list,
        "photometric_residual": photo_resid_list,
        "mean_update_norm": update_norm_list,
        "mean_w_minus_q_norm": w_minus_q_norm_list,
        "mean_delta_norm": delta_norm_list,
    }


def _save_stage_convergence_plot(output_dir: str, scene: str, sample_id: str, epe_stages: np.ndarray) -> None:
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    K = int(epe_stages.shape[0])
    xs = list(range(1, K + 1))

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(xs, epe_stages, "o-", linewidth=2, markersize=7)
    ax.set_xlabel("HQS Stage")
    ax.set_ylabel("Mean EPE (px)")
    ax.set_title(f"Convergence: {scene}/{sample_id}")
    ax.grid(True, alpha=0.3)
    ax.set_xticks(xs)
    path = _scene_subdir(Path(output_dir) / "stage_convergence", scene) / f"{sample_id}_convergence.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()


def _save_intermediate_states_plot(
    output_dir: str,
    scene: str,
    sample_id: str,
    flow_stages: List[np.ndarray],
    flow_gt: np.ndarray,
    valid: np.ndarray,
) -> None:
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    flow_gt_t = torch.from_numpy(flow_gt)
    valid_t = torch.from_numpy(valid).bool()
    K = len(flow_stages)
    cols = min(4, K)
    rows = int(math.ceil(K / cols)) + 1

    fig = plt.figure(figsize=(4.5 * cols, 3.8 * rows))
    gs = GridSpec(rows, cols, figure=fig, hspace=0.28, wspace=0.22)

    for k, flow_k_np in enumerate(flow_stages):
        ax = fig.add_subplot(gs[k // cols, k % cols])
        flow_k = _resize_flow_like(torch.from_numpy(flow_k_np), flow_gt_t.shape[-2], flow_gt_t.shape[-1])
        hsv = flow_to_hsv(flow_k)
        epe = torch.sqrt(((flow_k - flow_gt_t) ** 2).sum(dim=0))
        epe_mean = epe[valid_t].mean().item() if valid_t.any() else epe.mean().item()
        ax.imshow(hsv)
        ax.set_title(f"k={k + 1} | EPE={epe_mean:.3f}")
        ax.axis("off")

    ax_gt = fig.add_subplot(gs[rows - 1, :])
    ax_gt.imshow(flow_to_hsv(flow_gt_t))
    ax_gt.set_title("Ground Truth Flow")
    ax_gt.axis("off")

    path = _scene_subdir(Path(output_dir) / "intermediate_stages", scene) / f"{sample_id}_stages.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()


def _save_qualitative_panels(
    output_dir: str,
    scene: str,
    sample_id: str,
    flow_pred: np.ndarray,
    flow_gt: np.ndarray,
    img1: np.ndarray,
    img2: np.ndarray,
    flow_stages: List[np.ndarray],
    delta_stages: Optional[List[np.ndarray]],
    coupling_stages: Optional[List[np.ndarray]],
) -> Dict[str, str]:
    import cv2
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    out_dir = _scene_subdir(Path(output_dir) / "qualitative", scene)
    out_dir.mkdir(parents=True, exist_ok=True)

    flow_pred_t = torch.from_numpy(flow_pred)
    flow_gt_t = torch.from_numpy(flow_gt)
    img1_t = torch.from_numpy(img1)
    img2_t = torch.from_numpy(img2)

    flow_pred_t = _resize_flow_like(flow_pred_t, flow_gt_t.shape[-2], flow_gt_t.shape[-1])
    warped = _warp_xy(img2_t.unsqueeze(0), flow_pred_t.unsqueeze(0)).squeeze(0)

    epe = torch.sqrt(((flow_pred_t - flow_gt_t) ** 2).sum(dim=0)).numpy()
    epe_u8 = _to_uint8_gray(epe, norm_max=max(5.0, float(np.nanpercentile(epe, 95.0))))
    epe_bgr = cv2.applyColorMap(epe_u8, cv2.COLORMAP_HOT)
    epe_rgb = cv2.cvtColor(epe_bgr, cv2.COLOR_BGR2RGB)

    resid = (warped - img1_t).abs().mean(dim=0).numpy()
    resid_u8 = _to_uint8_gray(resid, norm_max=max(0.2, float(np.nanpercentile(resid, 95.0))))
    resid_bgr = cv2.applyColorMap(resid_u8, cv2.COLORMAP_TURBO)
    resid_rgb = cv2.cvtColor(resid_bgr, cv2.COLOR_BGR2RGB)

    overview_path = out_dir / f"{sample_id}_overview.png"
    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    axs = axes.ravel()
    axs[0].imshow(_to_rgb_np(img1))
    axs[0].set_title("Image1")
    axs[1].imshow(_to_rgb_np(img2))
    axs[1].set_title("Image2")
    axs[2].imshow(flow_to_hsv(flow_gt_t))
    axs[2].set_title("GT Flow")
    axs[3].imshow(flow_to_hsv(flow_pred_t))
    axs[3].set_title("Pred Flow")
    axs[4].imshow(epe_rgb)
    axs[4].set_title("Error Map (EPE)")
    axs[5].imshow(_to_rgb_np(warped.numpy()))
    axs[5].set_title("Warped Image2")
    axs[6].imshow(resid_rgb)
    axs[6].set_title("Photometric Residual")
    axs[7].axis("off")
    for ax in axs:
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(overview_path, dpi=120, bbox_inches="tight")
    plt.close()

    def _save_stage_grid(vec_list: List[np.ndarray], out_path: Path, title_prefix: str) -> None:
        K = len(vec_list)
        cols = min(4, K)
        rows = int(math.ceil(K / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 3.8 * rows))
        axes_arr = np.array(axes).reshape(-1)
        for i, vec in enumerate(vec_list):
            ax = axes_arr[i]
            vec_t = torch.from_numpy(vec)
            vec_t = _resize_flow_like(vec_t, flow_gt_t.shape[-2], flow_gt_t.shape[-1])
            ax.imshow(flow_to_hsv(vec_t))
            ax.set_title(f"{title_prefix} {i + 1}")
            ax.axis("off")
        for j in range(K, len(axes_arr)):
            axes_arr[j].axis("off")
        plt.tight_layout()
        plt.savefig(out_path, dpi=120, bbox_inches="tight")
        plt.close()

    flow_stage_path = out_dir / f"{sample_id}_flow_stages.png"
    _save_stage_grid(flow_stages, flow_stage_path, "w^k")

    delta_path = out_dir / f"{sample_id}_delta_stages.png"
    if delta_stages is not None and len(delta_stages) > 0:
        _save_stage_grid(delta_stages, delta_path, "delta w^k")

    coupling_path = out_dir / f"{sample_id}_coupling_stages.png"
    if coupling_stages is not None and len(coupling_stages) > 0:
        _save_stage_grid(coupling_stages, coupling_path, "w^k-q^k")

    paths = {
        "overview": str(overview_path),
        "flow_stages": str(flow_stage_path),
    }
    if delta_path.exists():
        paths["delta_stages"] = str(delta_path)
    if coupling_path.exists():
        paths["coupling_stages"] = str(coupling_path)
    return paths


def _process_sample_task(task: Dict[str, Any]) -> Dict[str, Any]:
    import cv2

    smooth_loss_fn, ofce_loss_fn = _get_worker_losses()

    sample_name = task["sample_name"]
    scene = task.get("scene", "")
    sample_id = task.get("sample_id", sample_name)
    frame1 = task.get("frame1", sample_id)
    output_dir = Path(task["output_dir"])

    flow_pred_t = torch.from_numpy(task["flow_pred"])
    flow_gt_t = torch.from_numpy(task["flow_gt"])
    valid_t = torch.from_numpy(task["valid"])
    img1_t = torch.from_numpy(task["img1"])
    img2_t = torch.from_numpy(task["img2"])
    flow_stages_np: List[np.ndarray] = task["flow_stages"]
    aux_stages_np: Optional[List[np.ndarray]] = task.get("aux_stages")
    delta_stages_np: Optional[List[np.ndarray]] = task.get("delta_stages")
    coupling_stages_np: Optional[List[np.ndarray]] = task.get("coupling_stages")

    occ_np = task.get("occlusion")
    inv_np = task.get("invalid")
    occ_t = torch.from_numpy(occ_np) if occ_np is not None else None
    inv_t = torch.from_numpy(inv_np) if inv_np is not None else None

    metrics = compute_metrics(flow_pred_t, flow_gt_t, valid_t, occ_mask=occ_t, invalid_mask=inv_t)
    metrics["sample_name"] = sample_name
    metrics["scene"] = scene
    metrics["sample_id"] = sample_id

    metrics["smoothness_loss"] = smooth_loss_fn(
        flow_pred_t.unsqueeze(0), img1_t.unsqueeze(0)
    ).item()
    metrics["ofce_loss"] = ofce_loss_fn(
        img1_t.unsqueeze(0), img2_t.unsqueeze(0), flow_pred_t.unsqueeze(0)
    ).item()

    flow_dir = _scene_subdir(output_dir / "flows", scene)
    flow_dir.mkdir(parents=True, exist_ok=True)
    gt_flow_dir = _scene_subdir(output_dir / "gt_flows", scene)
    gt_flow_dir.mkdir(parents=True, exist_ok=True)
    write_flow(flow_dir / f"{sample_id}.flo", flow_pred_t.numpy())
    write_flow(gt_flow_dir / f"{sample_id}_gt.flo", flow_gt_t.numpy())

    mag = torch.sqrt((flow_pred_t ** 2).sum(dim=0))
    flow_mag_max = float(mag.max().item())

    flow_hsv = flow_to_hsv(flow_pred_t)
    flow_hsv_dir = _scene_subdir(output_dir / "flows_hsv", scene)
    flow_hsv_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(
        str(flow_hsv_dir / f"{sample_id}.png"),
        cv2.cvtColor(flow_hsv, cv2.COLOR_RGB2BGR),
    )

    flow_gt_hsv = flow_to_hsv(flow_gt_t)
    gt_hsv_dir = _scene_subdir(output_dir / "gt_flows_hsv", scene)
    gt_hsv_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(
        str(gt_hsv_dir / f"{sample_id}_gt.png"),
        cv2.cvtColor(flow_gt_hsv, cv2.COLOR_RGB2BGR),
    )

    img1_rgb = _to_rgb_np(task["img1"])
    flow_pred_np = np.transpose(task["flow_pred"], (1, 2, 0))
    flow_gt_np = np.transpose(task["flow_gt"], (1, 2, 0))
    # Arrow overlays on both requested backgrounds and for both GT/pred flow.
    arrow_base = output_dir / "flows_arrows"
    pred_img_bg = _draw_flow_arrows(flow_pred_np, img1_rgb)
    gt_img_bg = _draw_flow_arrows(flow_gt_np, img1_rgb)
    pred_hsv_bg = _draw_flow_arrows(flow_pred_np, flow_hsv.astype(np.float32) / 255.0)
    gt_hsv_bg = _draw_flow_arrows(flow_gt_np, flow_gt_hsv.astype(np.float32) / 255.0)

    arrow_targets = [
        ("image_bg/pred", pred_img_bg, f"{sample_id}.png"),
        ("image_bg/gt", gt_img_bg, f"{sample_id}_gt.png"),
        ("flow_hsv_bg/pred", pred_hsv_bg, f"{sample_id}.png"),
        ("flow_hsv_bg/gt", gt_hsv_bg, f"{sample_id}_gt.png"),
    ]
    for sub, arr, name in arrow_targets:
        arr_dir = _scene_subdir(arrow_base / sub, scene)
        arr_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(arr_dir / name), cv2.cvtColor(arr, cv2.COLOR_RGB2BGR))

    epe = torch.sqrt(((flow_pred_t - flow_gt_t) ** 2).sum(dim=0))
    epe_np = epe.numpy()
    epe_norm = np.clip(epe_np / 5.0 * 255.0, 0, 255).astype(np.uint8)
    epe_bgr = cv2.applyColorMap(epe_norm, cv2.COLORMAP_HOT)
    err_dir = _scene_subdir(output_dir / "errors", scene)
    err_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(err_dir / f"{sample_id}_epe.png"), epe_bgr)

    has_scene = bool(task.get("has_scene", False))
    if has_scene:
        scene_frame_root = output_dir / "scene_frames" / scene
        (scene_frame_root / "image1").mkdir(parents=True, exist_ok=True)
        (scene_frame_root / "image2").mkdir(parents=True, exist_ok=True)
        (scene_frame_root / "gt_flow_hsv").mkdir(parents=True, exist_ok=True)
        (scene_frame_root / "pred_flow_hsv").mkdir(parents=True, exist_ok=True)
        cv2.imwrite(
            str(scene_frame_root / "image1" / f"{frame1}.png"),
            cv2.cvtColor((np.clip(img1_rgb, 0.0, 1.0) * 255.0).astype(np.uint8), cv2.COLOR_RGB2BGR),
        )
        cv2.imwrite(
            str(scene_frame_root / "image2" / f"{frame1}.png"),
            cv2.cvtColor((np.clip(_to_rgb_np(task["img2"]), 0.0, 1.0) * 255.0).astype(np.uint8), cv2.COLOR_RGB2BGR),
        )
        cv2.imwrite(
            str(scene_frame_root / "gt_flow_hsv" / f"{frame1}.png"),
            cv2.cvtColor(flow_gt_hsv, cv2.COLOR_RGB2BGR),
        )
        cv2.imwrite(
            str(scene_frame_root / "pred_flow_hsv" / f"{frame1}.png"),
            cv2.cvtColor(flow_hsv, cv2.COLOR_RGB2BGR),
        )

    stage_epe = _compute_stage_mean_epe_np(flow_stages_np, task["flow_gt"], task["valid"])
    _save_stage_convergence_plot(str(output_dir), scene, sample_id, stage_epe)
    _save_intermediate_states_plot(
        str(output_dir),
        scene,
        sample_id,
        flow_stages_np,
        task["flow_gt"],
        task["valid"],
    )

    iter_metrics = _compute_iteration_metrics_np(
        flow_stages_np,
        task["flow_gt"],
        task["valid"],
        task["img1"],
        task["img2"],
        aux_stages=aux_stages_np,
        delta_stages=delta_stages_np,
        coupling_stages=coupling_stages_np,
    )

    qualitative_paths: Dict[str, str] = {}
    if bool(task.get("save_qualitative", False)):
        qualitative_paths = _save_qualitative_panels(
            str(output_dir),
            scene,
            sample_id,
            task["flow_pred"],
            task["flow_gt"],
            task["img1"],
            task["img2"],
            flow_stages_np,
            delta_stages_np,
            coupling_stages_np,
        )

    norm_profile = stage_epe / max(float(stage_epe[0]), 1e-6)

    return {
        "metrics": metrics,
        "flow_mag_max": flow_mag_max,
        "normalized_stage_profile": norm_profile,
        "iter_metrics": iter_metrics,
        "qualitative_paths": qualitative_paths,
        "scene": scene,
        "frame1": frame1,
        "has_scene": has_scene,
    }


def setup_logging(output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(output_dir, "evaluation.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(log_file)],
    )


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _run_command(cmd: List[str], cwd: Optional[str] = None) -> str:
    try:
        p = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, check=True)
        return p.stdout.strip()
    except Exception:
        return ""


def _system_metadata(device: torch.device) -> Dict[str, Any]:
    meta: Dict[str, Any] = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "pytorch": torch.__version__,
        "device": str(device),
        "cpu": platform.processor() or platform.machine(),
        "gpu": None,
        "num_gpus": 0,
        "vram_gb": None,
        "ram_gb": None,
    }

    if torch.cuda.is_available():
        meta["num_gpus"] = torch.cuda.device_count()
        props = torch.cuda.get_device_properties(0)
        meta["gpu"] = props.name
        meta["vram_gb"] = round(props.total_memory / (1024 ** 3), 2)
    elif device.type == "mps":
        meta["gpu"] = "Apple Silicon (MPS)"
        meta["num_gpus"] = 1

    try:
        if hasattr(os, "sysconf"):
            pages = os.sysconf("SC_PHYS_PAGES")
            page_size = os.sysconf("SC_PAGE_SIZE")
            meta["ram_gb"] = round((pages * page_size) / (1024 ** 3), 2)
    except Exception:
        pass

    return meta


def _git_metadata(repo_dir: str) -> Dict[str, str]:
    return {
        "branch": _run_command(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=repo_dir),
        "commit": _run_command(["git", "rev-parse", "HEAD"], cwd=repo_dir),
        "repo_root": _run_command(["git", "rev-parse", "--show-toplevel"], cwd=repo_dir),
    }


def _aggregate_iter_profiles(profiles: List[Dict[str, List[float]]]) -> Dict[str, List[float]]:
    if not profiles:
        return {}
    max_k = max(len(p.get("epe", [])) for p in profiles)
    keys = ["epe", "photometric_residual", "mean_update_norm", "mean_w_minus_q_norm", "mean_delta_norm"]
    out: Dict[str, List[float]] = {k: [] for k in keys}

    for key in keys:
        mat = np.full((len(profiles), max_k), np.nan, dtype=np.float32)
        for i, p in enumerate(profiles):
            vals = p.get(key, [])
            if vals:
                mat[i, : len(vals)] = np.asarray(vals, dtype=np.float32)
        means = np.nanmean(mat, axis=0)
        out[key] = [float(x) for x in means]
    return out


def _save_iteration_summary(output_dir: Path, agg_iter: Dict[str, List[float]]) -> None:
    if not agg_iter:
        return

    json_path = output_dir / "per_iteration_summary.json"
    with open(json_path, "w") as f:
        json.dump(agg_iter, f, indent=2)

    csv_path = output_dir / "per_iteration_summary.csv"
    K = len(agg_iter.get("epe", []))
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["iteration", "epe", "photometric_residual", "mean_update_norm", "mean_w_minus_q_norm", "mean_delta_norm"])
        for i in range(K):
            w.writerow([
                i + 1,
                _safe_float(agg_iter.get("epe", [float("nan")])[i]),
                _safe_float(agg_iter.get("photometric_residual", [float("nan")])[i]),
                _safe_float(agg_iter.get("mean_update_norm", [float("nan")])[i]),
                _safe_float(agg_iter.get("mean_w_minus_q_norm", [float("nan")])[i]),
                _safe_float(agg_iter.get("mean_delta_norm", [float("nan")])[i]),
            ])


def _render_markdown_report(
    output_dir: Path,
    template_path: Optional[str],
    report_context: Dict[str, Any],
) -> Path:
    now = report_context.get("datetime", "")
    exp_id = report_context.get("experiment_id", "HQS-EXP-AUTO")
    exp_title = report_context.get("experiment_title", "Auto-generated comprehensive evaluation")
    status = report_context.get("status", "completed")

    metrics = report_context.get("metrics", {})
    iters = report_context.get("iter", {})
    cfg_path = report_context.get("config_path", "")
    data_cfg_path = report_context.get("data_config_path", "")
    ckpt = report_context.get("checkpoint", "")
    git = report_context.get("git", {})
    system = report_context.get("system", {})
    qualitative_paths = report_context.get("qualitative_paths", [])

    lines: List[str] = []
    lines.append("# E0 - Sanity Check")
    lines.append("")
    lines.append("## Experiment Metadata")
    lines.append("")
    lines.append(f"- Experiment ID: {exp_id}")
    lines.append(f"- Experiment title: {exp_title}")
    lines.append(f"- Date and Time: {now}")
    lines.append("- MLFlow Link: N/A (auto-fill if available)")
    lines.append("- MLFlow Experiment Name: N/A")
    lines.append("- MLFlow Run Name: N/A")
    lines.append("- MLFlow Artefact Store: N/A")
    lines.append(f"- Repository: {git.get('repo_root', '')}")
    lines.append(f"- Branch: {git.get('branch', '')}")
    lines.append(f"- Commit Hash: {git.get('commit', '')}")
    lines.append(f"- Configuration file: {cfg_path}")
    lines.append(f"- Data configuration file: {data_cfg_path}")
    lines.append(f"- Checkpoint: {ckpt}")
    lines.append(
        "- Hardware: "
        f"GPU={system.get('gpu')} | num_gpus={system.get('num_gpus')} | "
        f"VRAM_GB={system.get('vram_gb')} | CPU={system.get('cpu')} | RAM_GB={system.get('ram_gb')}"
    )
    lines.append(f"- Runtime: evaluation_seconds={report_context.get('runtime_sec', float('nan')):.2f}")
    lines.append("- Random seeds: N/A (evaluation-only script)")
    lines.append(f"- Status: {status}")
    lines.append("")

    lines.append("## Research Question")
    lines.append("- Human input required.")
    lines.append("")
    lines.append("## Hypothesis")
    lines.append("- Human input required.")
    lines.append("")
    lines.append("## Claim Being Tested")
    lines.append("- Auto-filled evidence below; choose paper-level claim manually.")
    lines.append("")
    lines.append("## Mathematical Object Under Test")
    lines.append("- Model update decomposition is expected from architecture; fill exact equation manually if needed.")
    lines.append("")

    lines.append("## Experimental Variables")
    lines.append("- Independent variable: checkpoint/config under evaluation.")
    lines.append("- Dependent variables: EPE/F1, speed buckets, occlusion-boundary buckets, iterative diagnostics.")
    lines.append("- Controlled variables: dataset split and evaluation codepath.")
    lines.append("")

    lines.append("## Quantitative Results")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|---|---:|")
    for k in sorted(metrics.keys()):
        v = metrics[k]
        if isinstance(v, (float, int)) and not math.isnan(float(v)):
            lines.append(f"| {k} | {float(v):.6f} |")
    lines.append("")

    if iters:
        lines.append("## Per-iteration Results")
        lines.append("")
        lines.append("| Iteration k | EPE | Photometric residual | Mean update norm | Mean ||w^k-q^k|| | Mean ||delta w^k|| |")
        lines.append("|---:|---:|---:|---:|---:|---:|")
        K = len(iters.get("epe", []))
        for i in range(K):
            epe = _safe_float(iters.get("epe", [float("nan")])[i])
            pr = _safe_float(iters.get("photometric_residual", [float("nan")])[i])
            up = _safe_float(iters.get("mean_update_norm", [float("nan")])[i])
            wq = _safe_float(iters.get("mean_w_minus_q_norm", [float("nan")])[i])
            dn = _safe_float(iters.get("mean_delta_norm", [float("nan")])[i])
            lines.append(f"| {i + 1} | {epe:.6f} | {pr:.6f} | {up:.6f} | {wq:.6f} | {dn:.6f} |")
        lines.append("")

    lines.append("## Qualitative Results")
    lines.append("- Generated figures are saved under `qualitative/`, `intermediate_stages/`, and `stage_convergence/`.")
    if qualitative_paths:
        lines.append("- Example generated figure paths:")
        for p in qualitative_paths[:8]:
            lines.append(f"  - {p}")
    lines.append("")

    lines.append("## Failure Modes")
    lines.append("- Auto-hints:")
    if iters and len(iters.get("epe", [])) >= 2:
        e = iters["epe"]
        if any(e[i] > e[i - 1] for i in range(1, len(e))):
            lines.append("  - Non-monotonic per-iteration EPE detected (possible recurrent drift/oscillation).")
    if metrics.get("d0_10") and metrics.get("epe_matched"):
        try:
            if float(metrics["d0_10"]) > float(metrics["epe_matched"]):
                lines.append("  - Boundary-near errors exceed matched mean EPE (boundary handling may be weak).")
        except Exception:
            pass
    lines.append("")

    lines.append("## Threats to Validity")
    lines.append("- Internal/external/construct/implementation validity require human review.")
    lines.append("")

    lines.append("## Reproducibility Checklist")
    lines.append(f"- Commit hash: {git.get('commit', '')}")
    lines.append(f"- Config file: {cfg_path}")
    lines.append(f"- Data config file: {data_cfg_path}")
    lines.append(f"- Checkpoint path: {ckpt}")
    lines.append("- Environment and hardware: see run_metadata.json")
    lines.append("- Metrics: metrics_summary.json")
    lines.append("- Per-iteration metrics: per_iteration_summary.json")
    lines.append("- Detailed per-sample metrics: metrics_detailed.json")
    lines.append("")

    lines.append("## Conclusion")
    lines.append("- Auto-generated evidence is provided above. Final scientific claim should be manually constrained.")
    lines.append("")

    lines.append("## Next Experiments")
    lines.append("1. Repeat with at least three random seeds.")
    lines.append("2. Run equal-capacity ablations and loss-only baseline.")
    lines.append("3. Add perturbation tests (noise/brightness/blur/JPEG).")
    lines.append("4. Review qualitative residual and delta maps for failure regimes.")

    if template_path and os.path.isfile(template_path):
        lines.append("")
        lines.append("---")
        lines.append("")
        lines.append("## Template Reference")
        lines.append(f"- Source template: {template_path}")

    out_path = output_dir / "experiment_report.md"
    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    return out_path


def _dataset_normalized_convergence(output_dir: Path, profiles: List[np.ndarray]) -> None:
    if not profiles:
        return

    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    max_k = max(p.shape[0] for p in profiles)
    mat = np.full((len(profiles), max_k), np.nan, dtype=np.float32)
    for i, p in enumerate(profiles):
        mat[i, : p.shape[0]] = p

    mean = np.nanmean(mat, axis=0)
    std = np.nanstd(mat, axis=0)
    lo = np.maximum(0.0, mean - 2.0 * std)
    hi = mean + 2.0 * std
    x = np.arange(1, max_k + 1)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(x, mean, "o-", linewidth=2, markersize=6, label="Mean normalized EPE")
    ax.fill_between(x, lo, hi, alpha=0.3, label="+-2sigma")
    ax.set_xlabel("HQS Stage")
    ax.set_ylabel("Normalized Mean EPE")
    ax.set_title("Dataset-Wide Normalized Stage Convergence")
    ax.grid(True, alpha=0.3)
    ax.legend()

    out_plot = output_dir / "stage_convergence" / "dataset_normalized_convergence.png"
    plt.savefig(out_plot, dpi=120, bbox_inches="tight")
    plt.close()

    out_json = output_dir / "stage_convergence" / "dataset_normalized_convergence.json"
    with open(out_json, "w") as f:
        json.dump(
            {
                "samples": int(len(profiles)),
                "mean": [float(v) for v in mean],
                "std": [float(v) for v in std],
                "lower_2std": [float(v) for v in lo],
                "upper_2std": [float(v) for v in hi],
            },
            f,
            indent=2,
        )


def _leaf_modules(model: nn.Module) -> Iterable[Tuple[str, nn.Module]]:
    for name, module in model.named_modules():
        if name == "":
            continue
        if len(list(module.children())) == 0:
            yield name, module


def _output_shape_repr(out: Any) -> str:
    if isinstance(out, torch.Tensor):
        return str(list(out.shape))
    if isinstance(out, (list, tuple)) and out:
        if isinstance(out[0], torch.Tensor):
            return str(list(out[0].shape))
        return f"{type(out).__name__}(len={len(out)})"
    if isinstance(out, dict):
        parts = []
        for k, v in out.items():
            if isinstance(v, torch.Tensor):
                parts.append(f"{k}:{list(v.shape)}")
            elif isinstance(v, list) and v and isinstance(v[0], torch.Tensor):
                parts.append(f"{k}:[{len(v)}x{list(v[0].shape)}]")
            else:
                parts.append(f"{k}:{type(v).__name__}")
        return "{" + ", ".join(parts[:3]) + ("..." if len(parts) > 3 else "") + "}"
    return type(out).__name__


def save_model_summary(
    model: nn.Module,
    device: torch.device,
    output_dir: Path,
    input_shape: Tuple[int, int, int, int],
) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    hooks = []

    for name, module in _leaf_modules(model):
        def _hook(_, __, out, mod_name=name, mod=module):
            params = sum(p.numel() for p in mod.parameters(recurse=False))
            trainable = sum(p.numel() for p in mod.parameters(recurse=False) if p.requires_grad)
            rows.append(
                {
                    "name": mod_name,
                    "type": mod.__class__.__name__,
                    "output_shape": _output_shape_repr(out),
                    "params": int(params),
                    "trainable_params": int(trainable),
                }
            )

        hooks.append(module.register_forward_hook(_hook))

    was_training = model.training
    model.eval()
    x1 = torch.zeros(*input_shape, device=device)
    x2 = torch.zeros(*input_shape, device=device)

    error = ""
    with torch.no_grad():
        try:
            _ = model(x1, x2)
        except Exception as e:
            error = str(e)

    for hndl in hooks:
        hndl.remove()
    if was_training:
        model.train()

    total_params = int(sum(p.numel() for p in model.parameters()))
    total_trainable = int(sum(p.numel() for p in model.parameters() if p.requires_grad))

    summary = {
        "input_shape": list(input_shape),
        "total_params": total_params,
        "trainable_params": total_trainable,
        "non_trainable_params": total_params - total_trainable,
        "layers": rows,
        "forward_error": error,
    }

    json_path = output_dir / "model_summary.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)

    md_path = output_dir / "model_summary.md"
    with open(md_path, "w") as f:
        f.write("# Model Summary\n\n")
        f.write(f"- Input shape: {list(input_shape)}\n")
        f.write(f"- Total params: {total_params:,}\n")
        f.write(f"- Trainable params: {total_trainable:,}\n")
        f.write(f"- Non-trainable params: {total_params - total_trainable:,}\n")
        if error:
            f.write(f"- Forward-pass capture warning: {error}\n")
        f.write("\n| Layer | Type | Output shape | Params | Trainable |\n")
        f.write("|---|---|---|---:|---:|\n")
        for r in rows:
            f.write(
                f"| {r['name']} | {r['type']} | {r['output_shape']} | "
                f"{r['params']:,} | {r['trainable_params']:,} |\n"
            )

    logger.info(f"Saved model summary JSON -> {json_path}")
    logger.info(f"Saved model summary markdown -> {md_path}")
    return {"json": str(json_path), "markdown": str(md_path), **summary}


class FlowEvaluator:
    def __init__(
        self,
        model: nn.Module,
        output_dir: str,
        device: torch.device,
        cfg: Optional[Dict[str, Any]] = None,
        postproc_workers: int = 1,
        qualitative_samples: str = "8",
        qualitative_seed: int = 42,
        save_report: bool = True,
        report_template_path: Optional[str] = None,
        experiment_id: str = "HQS-EXP-AUTO",
        experiment_title: str = "Auto comprehensive evaluation",
        status: str = "completed",
        report_context: Optional[Dict[str, Any]] = None,
    ):
        self.model = model
        self.output_dir = Path(output_dir)
        self.device = device
        self.cfg = cfg or {}
        self.postproc_workers = max(1, int(postproc_workers))
        self.qualitative_samples = str(qualitative_samples)
        self.qualitative_seed = int(qualitative_seed)
        self.save_report = bool(save_report)
        self.report_template_path = report_template_path
        self.experiment_id = experiment_id
        self.experiment_title = experiment_title
        self.status = status
        self.report_context = report_context or {}

        (self.output_dir / "flows").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "flows_hsv").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "intermediate_stages").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "gt_flows").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "gt_flows_hsv").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "errors").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "stage_convergence").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "qualitative").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "flows_arrows").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "gt_flows_arrows").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "scene_frames").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "scene_videos").mkdir(parents=True, exist_ok=True)

        self.seq_loss = SequenceLoss(
            gamma=self.cfg.get("gamma", 0.85),
            max_flow=self.cfg.get("max_flow", 400.0),
        )
        self.smooth_loss = SmoothnessLoss()
        self.ofce_loss = OFCELoss()

        self.metrics_list: List[Dict[str, Any]] = []
        self.flow_magnitudes: List[float] = []
        self.normalized_stage_profiles: List[np.ndarray] = []
        self.iter_profiles: List[Dict[str, List[float]]] = []
        self.qualitative_paths: List[str] = []

    def evaluate(self, dataloader: DataLoader) -> Dict[str, float]:
        self.model.eval()
        device = self.device

        aggregate_metrics: Dict[str, float] = {}
        num_samples = 0
        dataset_len = len(dataloader.dataset) if hasattr(dataloader, "dataset") else 0
        qual_set = _qualitative_selection(self.qualitative_samples, dataset_len, self.qualitative_seed)

        logger.info(
            f"Starting evaluation on {len(dataloader)} batches with "
            f"{self.postproc_workers} post-processing worker(s)..."
        )

        pbar = tqdm(total=dataset_len if dataset_len > 0 else None, desc="Evaluating", unit="sample")

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
                    occ = batch.get("occlusion")
                    inv = batch.get("invalid")
                    occ_cpu = occ.detach().cpu() if occ is not None else None
                    inv_cpu = inv.detach().cpu() if inv is not None else None

                    out = self.model(img1, img2)
                    flow_preds = out.get("flow_preds_raw", out["flow_preds"])
                    flow_final = out["flow_preds"][-1]
                    flow_low = out.get("flow_low_raw", out.get("flow_low", [flow_preds[-1]]))
                    aux_low = out.get("aux_low")
                    delta_low = out.get("delta_low")
                    coupling_low = out.get("coupling_residual_low")

                    img1_cpu = img1.detach().cpu()
                    img2_cpu = img2.detach().cpu()
                    flow_gt_cpu = flow_gt.detach().cpu()
                    valid_cpu = valid.detach().cpu()
                    flow_preds_cpu = [fp.detach().cpu() for fp in flow_preds]
                    flow_final_cpu = flow_final.detach().cpu()
                    flow_low_cpu = [fl.detach().cpu() for fl in flow_low]
                    aux_low_cpu = [a.detach().cpu() for a in aux_low] if aux_low is not None else None
                    delta_low_cpu = [d.detach().cpu() for d in delta_low] if delta_low is not None else None
                    coupling_low_cpu = [c.detach().cpu() for c in coupling_low] if coupling_low is not None else None

                    B = img1.shape[0]
                    tasks: List[Dict[str, Any]] = []
                    for b in range(B):
                        global_idx = num_samples + b
                        src_record = _resolve_dataset_record(dataloader.dataset, global_idx)
                        sample_meta = _infer_scene_and_sample(
                            src_record,
                            fallback_name=f"batch_{batch_idx:06d}_item_{b:02d}",
                        )
                        sample_name = f"batch_{batch_idx:06d}_item_{b:02d}"
                        save_q = True if qual_set is None else (global_idx in qual_set)

                        tasks.append(
                            {
                                "sample_name": sample_name,
                                "scene": sample_meta["scene"],
                                "sample_id": sample_meta["sample_id"],
                                "frame1": sample_meta["frame1"],
                                "frame2": sample_meta["frame2"],
                                "sort_key": sample_meta["sort_key"],
                                "output_dir": str(self.output_dir),
                                "flow_pred": flow_final_cpu[b].numpy(),
                                "flow_stages": [fp[b].numpy() for fp in flow_preds_cpu],
                                "flow_low_stages": [fl[b].numpy() for fl in flow_low_cpu],
                                "aux_stages": [a[b].numpy() for a in aux_low_cpu] if aux_low_cpu is not None else None,
                                "delta_stages": [d[b].numpy() for d in delta_low_cpu] if delta_low_cpu is not None else None,
                                "coupling_stages": [c[b].numpy() for c in coupling_low_cpu] if coupling_low_cpu is not None else None,
                                "flow_gt": flow_gt_cpu[b].numpy(),
                                "valid": valid_cpu[b].numpy(),
                                "occlusion": occ_cpu[b].numpy() if occ_cpu is not None else None,
                                "invalid": inv_cpu[b].numpy() if inv_cpu is not None else None,
                                "img1": img1_cpu[b].numpy(),
                                "img2": img2_cpu[b].numpy(),
                                "save_qualitative": save_q,
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
                        self.iter_profiles.append(result["iter_metrics"])

                        self.metrics_list.append(metrics)
                        for k, v in metrics.items():
                            if k in {"sample_name", "scene", "sample_id"}:
                                continue
                            if isinstance(v, (float, int, np.floating, np.integer)) and not math.isnan(float(v)):
                                aggregate_metrics[k] = aggregate_metrics.get(k, 0.0) + float(v)

                        for p in result.get("qualitative_paths", {}).values():
                            self.qualitative_paths.append(p)

                    pbar.update(B)
                    num_samples += B
        finally:
            pbar.close()
            if executor is not None:
                executor.shutdown(wait=True)

        for k in aggregate_metrics:
            aggregate_metrics[k] /= max(1, num_samples)

        metrics_path = self.output_dir / "metrics_detailed.json"
        with open(metrics_path, "w") as f:
            json.dump(self.metrics_list, f, indent=2)
        logger.info(f"Saved detailed metrics -> {metrics_path}")

        summary_path = self.output_dir / "metrics_summary.json"
        with open(summary_path, "w") as f:
            json.dump(aggregate_metrics, f, indent=2)
        logger.info(f"Saved summary metrics -> {summary_path}")

        wheel = create_flow_colorwheel()
        import cv2
        cv2.imwrite(
            str(self.output_dir / "flow_colorwheel_reference.png"),
            cv2.cvtColor(wheel, cv2.COLOR_RGB2BGR),
        )

        _dataset_normalized_convergence(self.output_dir, self.normalized_stage_profiles)

        self._build_scene_videos()

        agg_iter = _aggregate_iter_profiles(self.iter_profiles)
        _save_iteration_summary(self.output_dir, agg_iter)

        report_data = {
            "metrics": aggregate_metrics,
            "iter": agg_iter,
            "qualitative_paths": self.qualitative_paths,
            "datetime": dt.datetime.now().strftime("%d/%m/%Y - %H:%M:%S"),
            "experiment_id": self.experiment_id,
            "experiment_title": self.experiment_title,
            "status": self.status,
            **self.report_context,
        }

        report_data_path = self.output_dir / "report_data.json"
        with open(report_data_path, "w") as f:
            json.dump(report_data, f, indent=2)

        if self.save_report:
            report_path = _render_markdown_report(
                self.output_dir,
                self.report_template_path,
                report_data,
            )
            logger.info(f"Saved experiment report -> {report_path}")

        logger.info(f"Evaluation complete. Results saved to {self.output_dir}")
        return aggregate_metrics

    def _build_scene_videos(self) -> None:
        scene_root = self.output_dir / "scene_frames"
        if not scene_root.exists():
            return

        built: List[str] = []
        for scene_dir in sorted(scene_root.iterdir()):
            if not scene_dir.is_dir():
                continue
            out = self.output_dir / "scene_videos" / f"{scene_dir.name}.mp4"
            if _build_scene_video(scene_dir, out, fps=30):
                built.append(str(out))

        if built:
            logger.info("Built scene videos:")
            for path in built:
                logger.info(f"  - {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate optical flow model comprehensively")
    parser.add_argument("--config", "-c", required=True, help="Path to model config YAML")
    parser.add_argument("--checkpoint", "-ckpt", required=True, help="Path to model checkpoint")
    parser.add_argument("--data_config", "-dc", required=True, help="Path to data config YAML")
    parser.add_argument("--output_dir", "-o", required=True, help="Output directory for results")
    parser.add_argument("--device", default=None, help="Device (cuda/mps/cpu)")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for evaluation")
    parser.add_argument(
        "--postproc_workers",
        type=int,
        default=None,
        help="Number of CPU workers for per-sample post-processing; default=max(1,cpu_count-1)",
    )

    parser.add_argument("--report_template", default=None, help="Path to markdown template reference")
    parser.add_argument("--experiment_id", default="HQS-EXP-AUTO")
    parser.add_argument("--experiment_title", default="Auto comprehensive evaluation")
    parser.add_argument("--status", default="completed", choices=["planned", "running", "completed", "failed", "superseded"])
    parser.add_argument("--no_report", action="store_true", help="Disable auto experiment_report.md generation")
    parser.add_argument(
        "--qualitative_samples",
        default="8",
        help="Number of random qualitative samples, or 'all' to render the full validation set",
    )
    parser.add_argument(
        "--qualitative_seed",
        type=int,
        default=42,
        help="Random seed used when selecting qualitative samples",
    )

    parser.add_argument("--no_model_summary", action="store_true", help="Disable TensorFlow-like model summary output")
    parser.add_argument("--summary_height", type=int, default=256, help="Dummy input height for model summary")
    parser.add_argument("--summary_width", type=int, default=256, help="Dummy input width for model summary")

    args = parser.parse_args()

    from omegaconf import OmegaConf

    setup_logging(args.output_dir)
    logger.info("=" * 80)
    logger.info("HQS Optical Flow - Comprehensive Evaluation + Report")
    logger.info("=" * 80)

    t0 = time.time()

    device = torch.device(args.device) if args.device else get_device()
    logger.info(f"Device: {device}")

    model_cfg = OmegaConf.load(args.config)
    data_cfg = OmegaConf.load(args.data_config)

    model = build_model(model_cfg).to(device)
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt.get("model", ckpt), strict=False)
    logger.info(f"Loaded checkpoint from {args.checkpoint}")

    eval_cfg = data_cfg.get("val_data") or data_cfg.get("data") or data_cfg
    eval_cfg = OmegaConf.merge(eval_cfg, {"batch_size": args.batch_size})

    eval_data = build_dataset(eval_cfg, split="val")
    eval_loader = build_dataloader(eval_data, eval_cfg, split="val")
    logger.info(f"Evaluation set: {len(eval_data)} samples")

    model_summary_info: Dict[str, Any] = {}
    if not args.no_model_summary:
        model_summary_info = save_model_summary(
            model,
            device,
            Path(args.output_dir),
            input_shape=(1, 3, int(args.summary_height), int(args.summary_width)),
        )

    default_workers = max(1, (os.cpu_count() or 1) - 1)
    postproc_workers = default_workers if args.postproc_workers is None else max(1, args.postproc_workers)
    logger.info(f"Post-processing workers: {postproc_workers}")

    sys_meta = _system_metadata(device)
    git_meta = _git_metadata(str(Path(__file__).resolve().parent))

    report_context = {
        "config_path": os.path.abspath(args.config),
        "data_config_path": os.path.abspath(args.data_config),
        "checkpoint": os.path.abspath(args.checkpoint),
        "system": sys_meta,
        "git": git_meta,
        "dataset_size": len(eval_data),
        "model_summary": {
            "total_params": model_summary_info.get("total_params", int(sum(p.numel() for p in model.parameters()))),
            "trainable_params": model_summary_info.get(
                "trainable_params", int(sum(p.numel() for p in model.parameters() if p.requires_grad))
            ),
            "summary_markdown": model_summary_info.get("markdown", ""),
            "summary_json": model_summary_info.get("json", ""),
        },
    }

    evaluator = FlowEvaluator(
        model,
        args.output_dir,
        device,
        cfg=model_cfg.loss if hasattr(model_cfg, "loss") else {},
        postproc_workers=postproc_workers,
        qualitative_samples=args.qualitative_samples,
        qualitative_seed=args.qualitative_seed,
        save_report=not args.no_report,
        report_template_path=args.report_template,
        experiment_id=args.experiment_id,
        experiment_title=args.experiment_title,
        status=args.status,
        report_context=report_context,
    )

    metrics = evaluator.evaluate(eval_loader)

    runtime = time.time() - t0
    run_meta_path = Path(args.output_dir) / "run_metadata.json"
    run_metadata = {
        "datetime": dt.datetime.now().isoformat(),
        "runtime_sec": float(runtime),
        "args": vars(args),
        "system": sys_meta,
        "git": git_meta,
        "config": os.path.abspath(args.config),
        "data_config": os.path.abspath(args.data_config),
        "checkpoint": os.path.abspath(args.checkpoint),
        "model_summary": model_summary_info,
        "metrics": metrics,
    }
    with open(run_meta_path, "w") as f:
        json.dump(run_metadata, f, indent=2)

    report_data_path = Path(args.output_dir) / "report_data.json"
    if report_data_path.exists():
        with open(report_data_path) as f:
            rd = json.load(f)
        rd["runtime_sec"] = float(runtime)
        with open(report_data_path, "w") as f:
            json.dump(rd, f, indent=2)

    logger.info("=" * 80)
    logger.info("Final Metrics Summary")
    logger.info("=" * 80)
    for k, v in sorted(metrics.items()):
        if not math.isnan(v):
            logger.info(f"  {k:22s}: {v:10.4f}")
    logger.info(f"  {'runtime_sec':22s}: {runtime:10.2f}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
