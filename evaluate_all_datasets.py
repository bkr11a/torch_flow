#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

from omegaconf import OmegaConf


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate one checkpoint on all configured datasets.")
    p.add_argument("--config", "-c", required=True)
    p.add_argument("--checkpoint", "-ckpt", required=True)
    p.add_argument("--suite", "-s", required=True)
    p.add_argument("--output_dir", "-o", required=True)
    p.add_argument("--device", default=None)
    p.add_argument("--qualitative_samples", default=None)
    p.add_argument("--postproc_workers", type=int, default=None)
    p.add_argument("--max_samples", type=int, default=None)
    p.add_argument("--continue_on_error", action="store_true")
    p.add_argument("--dry_run", action="store_true")

    p.add_argument(
        "--eval_resolution",
        choices=["suite", "safe_crop", "full"],
        default="suite",
        help=(
            "Evaluation resolution policy. "
            "'suite' respects each dataset YAML entry; "
            "'safe_crop' forces eval_center_crop=true; "
            "'full' disables eval cropping for all datasets."
        ),
    )
    p.add_argument(
        "--full_resolution",
        action="store_true",
        help="Alias for --eval_resolution full.",
    )

    return p.parse_args()


def as_dict(cfg) -> Dict[str, Any]:
    d = OmegaConf.to_container(cfg, resolve=True)
    assert isinstance(d, dict)
    return d


def _resolution_policy(args) -> str:
    return "full" if args.full_resolution else str(args.eval_resolution)


def _apply_resolution_policy(data_cfg: Dict[str, Any], policy: str) -> Dict[str, Any]:
    data_cfg = copy.deepcopy(data_cfg)

    if policy == "suite":
        return data_cfg

    if policy == "safe_crop":
        data_cfg["eval_center_crop"] = True
        # Keep suite-defined eval_crop_size if present.
        # If missing, leave it missing so evaluate_comprehensive can fall back
        # to crop_size or no crop.
        return data_cfg

    if policy == "full":
        # Full native-resolution evaluation.
        # This is useful for final qualitative artifacts, but may be very
        # expensive or impossible for all-pairs-correlation models on Spring,
        # HD1K, and KITTI.
        data_cfg["eval_center_crop"] = False
        return data_cfg

    raise ValueError(f"Unknown eval resolution policy: {policy}")


def write_data_config(entry: Dict[str, Any], out: Path, policy: str) -> Dict[str, Any]:
    out.parent.mkdir(parents=True, exist_ok=True)
    data_cfg = _apply_resolution_policy(entry["data"], policy)
    OmegaConf.save(config=OmegaConf.create({"val_data": data_cfg}), f=str(out))
    return data_cfg


def run_one(entry: Dict[str, Any], args, out_root: Path, cfg_root: Path) -> Dict[str, Any]:
    name = str(entry["id"])
    policy = _resolution_policy(args)

    out_dir = out_root / name
    if policy == "full":
        out_dir = out_root / f"{name}_fullres"
    elif policy == "safe_crop":
        out_dir = out_root / f"{name}_safecrop"

    data_cfg_path = cfg_root / f"{name}_{policy}.yaml"
    data_cfg = write_data_config(entry, data_cfg_path, policy=policy)

    batch_size = int(entry.get("batch_size", data_cfg.get("batch_size", 1)))
    qualitative = args.qualitative_samples or str(entry.get("qualitative_samples", "8"))
    max_samples = args.max_samples if args.max_samples is not None else entry.get("max_samples", None)

    cmd = [
        sys.executable, "evaluate_comprehensive.py",
        "--config", args.config,
        "--checkpoint", args.checkpoint,
        "--data_config", str(data_cfg_path),
        "--output_dir", str(out_dir),
        "--batch_size", str(batch_size),
        "--qualitative_samples", str(qualitative),
        "--experiment_id", f"EVAL-{name}-{policy}",
        "--experiment_title", f"Comprehensive evaluation: {name} ({policy})",
    ]

    if args.device:
        cmd += ["--device", args.device]
    if args.postproc_workers is not None:
        cmd += ["--postproc_workers", str(args.postproc_workers)]
    if max_samples is not None:
        cmd += ["--max_samples", str(int(max_samples))]
    if bool(entry.get("no_model_summary", True)):
        cmd += ["--no_model_summary"]

    print("\n" + "=" * 100)
    print(f"Dataset: {name}")
    print(f"Resolution policy: {policy}")
    print(f"eval_center_crop: {data_cfg.get('eval_center_crop', None)}")
    print(f"eval_crop_size: {data_cfg.get('eval_crop_size', None)}")
    print(" ".join(cmd))
    print("=" * 100 + "\n")

    if args.dry_run:
        return {
            "dataset": name,
            "status": "dry_run",
            "resolution_policy": policy,
            "output_dir": str(out_dir),
            "data_config": str(data_cfg_path),
            "command": cmd,
        }

    out_dir.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(cmd, text=True)
    status = "completed" if proc.returncode == 0 else "failed"

    metrics = {}
    metrics_path = out_dir / "metrics_summary.json"
    if metrics_path.exists():
        with open(metrics_path) as f:
            metrics = json.load(f)

    return {
        "dataset": name,
        "status": status,
        "returncode": proc.returncode,
        "resolution_policy": policy,
        "output_dir": str(out_dir),
        "data_config": str(data_cfg_path),
        "supports_scenes": bool(entry.get("supports_scenes", False)),
        "metrics": metrics,
    }


def safe_float(x):
    try:
        return float(x)
    except Exception:
        return float("nan")


def save_tables(out_root: Path, results: List[Dict[str, Any]]) -> None:
    sd = out_root / "summary"
    sd.mkdir(parents=True, exist_ok=True)

    with open(sd / "evaluation_summary.json", "w") as f:
        json.dump(results, f, indent=2)

    keys = sorted({
        k
        for r in results
        for k, v in r.get("metrics", {}).items()
        if isinstance(v, (int, float))
    })

    with open(sd / "evaluation_summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "status", "resolution_policy", "supports_scenes", "output_dir", *keys])
        for r in results:
            m = r.get("metrics", {})
            w.writerow([
                r.get("dataset"),
                r.get("status"),
                r.get("resolution_policy"),
                r.get("supports_scenes"),
                r.get("output_dir"),
                *[m.get(k, "") for k in keys],
            ])


def plot_metric(out_root: Path, results: List[Dict[str, Any]], metric: str, ylabel: str, title: str) -> None:
    import math
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    rows = [
        (r["dataset"], safe_float(r.get("metrics", {}).get(metric)))
        for r in results
        if r.get("status") == "completed"
    ]
    rows = [(d, v) for d, v in rows if not math.isnan(v)]
    if not rows:
        return

    fig, ax = plt.subplots(figsize=(max(8, 0.75 * len(rows)), 5.5))
    ax.bar([d for d, _ in rows], [v for _, v in rows])
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.3)
    ax.tick_params(axis="x", rotation=35)
    fig.tight_layout()
    fig.savefig(out_root / "summary" / f"{metric}_bar.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_group(out_root: Path, results: List[Dict[str, Any]], keys: List[str], filename: str, title: str) -> None:
    import math
    import numpy as np
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    rows = []
    for r in results:
        if r.get("status") != "completed":
            continue
        vals = [safe_float(r.get("metrics", {}).get(k)) for k in keys]
        if not all(math.isnan(v) for v in vals):
            rows.append((r["dataset"], vals))

    if not rows:
        return

    labels = [r[0] for r in rows]
    mat = np.asarray([r[1] for r in rows], dtype=float)
    x = np.arange(len(labels))
    width = 0.8 / max(1, len(keys))

    fig, ax = plt.subplots(figsize=(max(9, 0.9 * len(labels)), 5.8))
    for i, key in enumerate(keys):
        ax.bar(x + (i - (len(keys) - 1) / 2.0) * width, mat[:, i], width, label=key)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_title(title)
    ax.set_ylabel("metric value")
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_root / "summary" / filename, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_plots(out_root: Path, results: List[Dict[str, Any]]) -> None:
    (out_root / "summary").mkdir(parents=True, exist_ok=True)
    plot_metric(out_root, results, "epe", "EPE (px)", "Cross-dataset endpoint error")
    plot_metric(out_root, results, "f1", "F1 / outlier rate (%)", "Cross-dataset F1 / outlier rate")
    plot_group(out_root, results, ["s0_10", "s10_40", "s40_plus"], "speed_bucket_metrics.png", "Speed-stratified performance")
    plot_group(out_root, results, ["d0", "d0_10", "d10_60", "d60_140", "d140_plus"], "occlusion_distance_bucket_metrics.png", "Distance-to-occlusion-boundary performance")


def save_index(out_root: Path, results: List[Dict[str, Any]], args) -> None:
    lines = [
        "# Full Optical-Flow Evaluation Suite",
        "",
        "## Run metadata",
        "",
        f"- Config: `{os.path.abspath(args.config)}`",
        f"- Checkpoint: `{os.path.abspath(args.checkpoint)}`",
        f"- Suite: `{os.path.abspath(args.suite)}`",
        f"- Resolution policy: `{_resolution_policy(args)}`",
        "",
        "## Dataset summary",
        "",
        "| Dataset | Status | Resolution | EPE | F1 | Supports scenes | Output |",
        "|---|---|---|---:|---:|---|---|",
    ]

    for r in results:
        m = r.get("metrics", {})
        epe = safe_float(m.get("epe"))
        f1 = safe_float(m.get("f1"))
        epe_s = "" if epe != epe else f"{epe:.4f}"
        f1_s = "" if f1 != f1 else f"{f1:.4f}"
        lines.append(
            f"| {r.get('dataset')} | {r.get('status')} | {r.get('resolution_policy')} | "
            f"{epe_s} | {f1_s} | {r.get('supports_scenes')} | `{r.get('output_dir')}` |"
        )

    lines += [
        "",
        "## Publication plots",
        "",
        "- `summary/epe_bar.png`",
        "- `summary/f1_bar.png`",
        "- `summary/speed_bucket_metrics.png`",
        "- `summary/occlusion_distance_bucket_metrics.png`",
        "",
        "Each dataset subdirectory is produced by `evaluate_comprehensive.py` and contains flow PNGs, GT flow PNGs, EPE maps, stage convergence, model-internal grids, and scene videos where scenes are resolvable.",
    ]

    with open(out_root / "EVALUATION_INDEX.md", "w") as f:
        f.write("\n".join(lines) + "\n")


def main():
    args = parse_args()
    suite = as_dict(OmegaConf.load(args.suite))
    out_root = Path(args.output_dir)
    cfg_root = out_root / "_data_configs"
    out_root.mkdir(parents=True, exist_ok=True)

    results = []
    for entry in suite.get("datasets", []):
        if not entry.get("enabled", True):
            continue

        result = run_one(entry, args, out_root, cfg_root)
        results.append(result)

        if result.get("status") == "failed" and not args.continue_on_error:
            save_tables(out_root, results)
            save_index(out_root, results, args)
            raise SystemExit(result.get("returncode", 1))

    save_tables(out_root, results)
    save_plots(out_root, results)
    save_index(out_root, results, args)

    print(f"\nEvaluation suite complete: {out_root}")
    print(f"Summary: {out_root / 'summary' / 'evaluation_summary.csv'}")
    print(f"Index:   {out_root / 'EVALUATION_INDEX.md'}")


if __name__ == "__main__":
    main()
