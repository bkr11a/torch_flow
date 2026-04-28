"""Evaluation script for HQSFlow.

Usage examples
--------------
# Evaluate on Sintel clean + final
python evaluate.py --config configs/default.yaml \
                   --checkpoint checkpoints/hqs_flow/best.pth \
                   --datasets sintel

# Evaluate on all datasets
python evaluate.py --config configs/default.yaml \
                   --checkpoint checkpoints/hqs_flow/best.pth \
                   --datasets sintel kitti spring \
                   --output results/eval.csv

# OmegaConf overrides
python evaluate.py --config configs/default.yaml \
                   --checkpoint checkpoints/hqs_flow/best.pth \
                   --datasets sintel \
                   model.model_backbone.num_hqs_iterations=8
"""
from __future__ import annotations

import argparse
import csv
import logging
import os
import sys
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from omegaconf import OmegaConf

from engine.trainer import get_device, amp_enabled
from models import build_model
from utils import compute_metrics, aggregate_metrics, InputPadder
from data import build_dataset, build_dataloader

logging.basicConfig(
    format="%(asctime)s | %(levelname)-8s | %(name)s: %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("evaluate")


# ---------------------------------------------------------------------------
# Per-dataset evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_loader(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    use_amp: bool,
    label: str,
) -> Dict[str, float]:
    model.eval()
    records: List[Dict[str, float]] = []

    for batch in tqdm(loader, desc=f"  {label}", unit="batch", leave=False, dynamic_ncols=True):
        img1  = batch["image1"].to(device, non_blocking=True)
        img2  = batch["image2"].to(device, non_blocking=True)
        flow  = batch["flow"].to(device, non_blocking=True)
        valid = batch["valid"].to(device, non_blocking=True)
        occ_batch = batch.get("occlusion")

        padder = InputPadder(img1.shape, divisor=8)
        img1, img2 = padder.pad(img1, img2)

        with torch.autocast(device_type=device.type, enabled=use_amp):
            out = model(img1, img2)
        pred = padder.unpad(out["flow_preds"][-1])

        for b in range(pred.shape[0]):
            occ = None
            if occ_batch is not None:
                occ = occ_batch[b].to(device)
            records.append(compute_metrics(pred[b], flow[b], valid[b], occ_mask=occ))

    agg = aggregate_metrics(records)
    logger.info(
        f"[{label}] "
        + "  ".join(f"{k}={v:.4f}" for k, v in agg.items())
    )
    return agg


def _make_loader(root: str, name: str, extra: dict, batch_size: int = 1):
    dcfg = OmegaConf.create({
        "name": name,
        "root": root,
        "batch_size": batch_size,
        "num_workers": 2,
        "crop": False,
        **extra,
    })
    ds = build_dataset(dcfg, split="val")
    return build_dataloader(ds, dcfg, split="val")


# ---------------------------------------------------------------------------
# Multi-dataset evaluation
# ---------------------------------------------------------------------------

def run_sintel(model, device, use_amp, cfg, data_root=None) -> Dict[str, float]:
    root = data_root or cfg.data.get("sintel_root", "datasets/Sintel")
    results: Dict[str, float] = {}
    for pass_name in ("clean", "final"):
        loader = _make_loader(root, "sintel", {"pass_name": pass_name})
        agg = evaluate_loader(model, loader, device, use_amp, f"Sintel-{pass_name}")
        for k, v in agg.items():
            results[f"{pass_name}_{k}"] = v
    return results


def run_kitti(model, device, use_amp, cfg, data_root=None) -> Dict[str, float]:
    root = data_root or cfg.data.get("kitti_root", "datasets/KITTI")
    loader = _make_loader(root, "kitti", {})
    return evaluate_loader(model, loader, device, use_amp, "KITTI-15")


def run_spring(model, device, use_amp, cfg, data_root=None) -> Dict[str, float]:
    root = data_root or cfg.data.get("spring_root", "datasets/Spring")
    loader = _make_loader(root, "spring", {})
    return evaluate_loader(model, loader, device, use_amp, "Spring")


# ---------------------------------------------------------------------------
# Table + CSV
# ---------------------------------------------------------------------------

_COLUMN_SETS = {
    "sintel": [
        ("clean_epe",   "Sintel-C EPE"),
        ("clean_f1",    "Sintel-C F1 %"),
        ("final_epe",   "Sintel-F EPE"),
        ("final_f1",    "Sintel-F F1 %"),
    ],
    "kitti": [
        ("epe",  "KITTI EPE"),
        ("f1",   "KITTI F1-all %"),
    ],
    "spring": [
        ("epe",      "Spring EPE"),
        ("f1",       "Spring F1 %"),
        ("s0_10",    "Spring s0-10"),
        ("s10_40",   "Spring s10-40"),
        ("s40_plus", "Spring s40+"),
    ],
}

_SOTA_REF = {
    "Sintel-C EPE":  "1.43 (RAFT)",
    "Sintel-F EPE":  "2.71 (RAFT)",
    "KITTI F1-all %": "5.10 (RAFT)",
}

_COL_W = [14, 22, 10, 18]


def _row(cells) -> str:
    return "  ".join(str(c).ljust(w) for c, w in zip(cells, _COL_W))


def print_table(all_metrics: Dict[str, Dict[str, float]]) -> None:
    sep = "  ".join("-" * w for w in _COL_W)
    print(_row(["Dataset", "Metric", "Value", "SOTA (ref)"]))
    print(sep)
    for ds_key, metrics in all_metrics.items():
        cols = _COLUMN_SETS.get(ds_key, [(k, k) for k in metrics])
        for metric_key, display in cols:
            val = metrics.get(metric_key)
            val_str  = f"{val:.4f}" if val is not None else "—"
            sota_str = _SOTA_REF.get(display, "—")
            print(_row([ds_key, display, val_str, sota_str]))
    print()


def save_csv(all_metrics: Dict[str, Dict[str, float]], path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    rows = [
        {"dataset": ds, "metric": k, "value": v}
        for ds, metrics in all_metrics.items()
        for k, v in metrics.items()
    ]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["dataset", "metric", "value"])
        w.writeheader()
        w.writerows(rows)
    logger.info(f"Results → {path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate HQSFlow")
    parser.add_argument("--config",     required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--datasets", nargs="+",
        choices=["sintel", "kitti", "spring"],
        default=["sintel"],
    )
    parser.add_argument("--device",       default=None)
    parser.add_argument("--output",       default=None, help="CSV output path")
    parser.add_argument("--sintel_root",  default=None)
    parser.add_argument("--kitti_root",   default=None)
    parser.add_argument("--spring_root",  default=None)
    parser.add_argument("overrides", nargs="*",
                        help="OmegaConf key=value overrides")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    if args.overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(args.overrides))

    device  = get_device(args.device)
    use_amp = amp_enabled(device)
    logger.info(f"Device: {device}  AMP: {use_amp}")

    model = build_model(cfg).to(device)
    ckpt  = torch.load(args.checkpoint, map_location=device, weights_only=False)
    state = ckpt.get("model", ckpt)
    model.load_state_dict(state, strict=True)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Loaded {args.checkpoint}  ({n_params:,} params)")

    runners = {
        "sintel": lambda: run_sintel(model, device, use_amp, cfg, args.sintel_root),
        "kitti":  lambda: run_kitti(model,  device, use_amp, cfg, args.kitti_root),
        "spring": lambda: run_spring(model, device, use_amp, cfg, args.spring_root),
    }

    all_metrics: Dict[str, Dict[str, float]] = {}
    for ds in args.datasets:
        logger.info(f"\n{'─'*50}\nEvaluating: {ds}\n{'─'*50}")
        all_metrics[ds] = runners[ds]()

    print("\n" + "=" * 70)
    print(" HQSFlow  —  Evaluation Results")
    print("=" * 70)
    print_table(all_metrics)
    print("=" * 70 + "\n")

    if args.output:
        save_csv(all_metrics, args.output)


if __name__ == "__main__":
    main()
