#!/usr/bin/env python3
"""Create detailed scientific plots from a training history.json file."""
from __future__ import annotations

import argparse
import json
import logging
import math
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np


logger = logging.getLogger(__name__)


def setup_logging(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(output_dir / "plot_history_scientific.log"),
        ],
    )


def load_history(path: Path) -> Dict:
    with open(path) as handle:
        return json.load(handle)


def series_from_pairs(values: Sequence[Sequence[float]]) -> Tuple[np.ndarray, np.ndarray]:
    if not values:
        return np.array([]), np.array([])
    array = np.asarray(values, dtype=float)
    return array[:, 0], array[:, 1]


def smooth(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or values.size < window:
        return values
    kernel = np.ones(window, dtype=float) / float(window)
    return np.convolve(values, kernel, mode="same")


def save_line_plot(
    output_path: Path,
    x: np.ndarray,
    y: np.ndarray,
    title: str,
    xlabel: str,
    ylabel: str,
    smooth_window: int,
    color: str,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(x, y, alpha=0.35, linewidth=1.0, color=color, label="raw")
    y_smooth = smooth(y, smooth_window)
    ax.plot(x, y_smooth, linewidth=2.0, color=color, label=f"smooth({smooth_window})")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def extract_val_metrics(val_metrics: Sequence[Dict]) -> Dict[str, List[Tuple[float, float]]]:
    metric_series: Dict[str, List[Tuple[float, float]]] = {}
    for index, record in enumerate(val_metrics):
        step = float(record.get("step", index))
        for key, value in record.items():
            if key == "step" or isinstance(value, bool):
                continue
            if isinstance(value, (int, float)) and math.isfinite(float(value)):
                metric_series.setdefault(key, []).append((step, float(value)))
    return metric_series


def grouped_metric_name(metric_name: str) -> str:
    if metric_name.startswith("epe"):
        return "epe_family"
    if metric_name.startswith("s"):
        return "speed_buckets"
    if metric_name.startswith("d"):
        return "distance_buckets"
    if metric_name.startswith("loss"):
        return "loss_family"
    return "other_metrics"


def save_metric_grid(
    output_path: Path,
    metric_items: Sequence[Tuple[str, List[Tuple[float, float]]]],
    smooth_window: int,
) -> None:
    if not metric_items:
        return
    cols = 2
    rows = int(math.ceil(len(metric_items) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(14, 4.5 * rows))
    axes = np.atleast_1d(axes).reshape(rows, cols)

    for axis, (metric_name, series) in zip(axes.flat, metric_items):
        x, y = series_from_pairs(series)
        axis.plot(x, y, alpha=0.35, linewidth=1.0, color="tab:blue")
        axis.plot(x, smooth(y, smooth_window), linewidth=2.0, color="tab:red")
        axis.set_title(metric_name)
        axis.set_xlabel("Step")
        axis.set_ylabel(metric_name)
        axis.grid(True, alpha=0.3)

    for axis in axes.flat[len(metric_items):]:
        axis.axis("off")

    fig.tight_layout()
    fig.savefig(output_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def save_summary_dashboard(
    output_path: Path,
    history: Dict,
    train_loss: Tuple[np.ndarray, np.ndarray],
    train_epe: Tuple[np.ndarray, np.ndarray],
    val_series: Dict[str, List[Tuple[float, float]]],
    smooth_window: int,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    loss_x, loss_y = train_loss
    if loss_y.size:
        axes[0, 0].plot(loss_x, loss_y, alpha=0.35, color="tab:blue")
        axes[0, 0].plot(loss_x, smooth(loss_y, smooth_window), color="tab:red", linewidth=2.0)
    axes[0, 0].set_title("Training Loss")
    axes[0, 0].set_xlabel("Step")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].grid(True, alpha=0.3)

    epe_x, epe_y = train_epe
    if epe_y.size:
        axes[0, 1].plot(epe_x, epe_y, alpha=0.35, color="tab:green")
        axes[0, 1].plot(epe_x, smooth(epe_y, smooth_window), color="tab:orange", linewidth=2.0)
    axes[0, 1].set_title("Training EPE")
    axes[0, 1].set_xlabel("Step")
    axes[0, 1].set_ylabel("EPE")
    axes[0, 1].grid(True, alpha=0.3)

    for metric_name in ["epe", "f1", "epe_matched", "epe_unmatched"]:
        series = val_series.get(metric_name)
        if not series:
            continue
        x, y = series_from_pairs(series)
        axes[1, 0].plot(x, y, linewidth=2.0, label=metric_name)
    axes[1, 0].set_title("Key Validation Metrics")
    axes[1, 0].set_xlabel("Step")
    axes[1, 0].set_ylabel("Metric Value")
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend()

    axes[1, 1].axis("off")
    summary_lines = [
        f"run_name: {history.get('run_name', '')}",
        f"total_steps: {history.get('total_steps', 0)}",
        f"best_epe: {history.get('best_epe', float('nan'))}",
        f"best_step: {history.get('best_step', 0)}",
        f"train_loss_points: {len(history.get('train_loss', []))}",
        f"train_epe_points: {len(history.get('train_epe', []))}",
        f"val_points: {len(history.get('val_metrics', []))}",
        f"tracked_val_metrics: {len(val_series)}",
    ]
    axes[1, 1].text(
        0.02,
        0.98,
        "\n".join(summary_lines),
        va="top",
        ha="left",
        fontsize=12,
        family="monospace",
    )

    fig.tight_layout()
    fig.savefig(output_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def save_metric_correlations(
    output_path: Path,
    val_series: Dict[str, List[Tuple[float, float]]],
) -> None:
    metric_names = [name for name, series in val_series.items() if len(series) >= 2]
    if len(metric_names) < 2:
        return

    aligned_steps = sorted(set.intersection(*[
        set(step for step, _ in val_series[name]) for name in metric_names
    ]))
    if len(aligned_steps) < 2:
        return

    matrix = []
    for name in metric_names:
        value_map = {step: value for step, value in val_series[name]}
        matrix.append([value_map[step] for step in aligned_steps])
    matrix_np = np.asarray(matrix, dtype=float)
    corr = np.corrcoef(matrix_np)

    fig, ax = plt.subplots(figsize=(10, 8))
    image = ax.imshow(corr, cmap="coolwarm", vmin=-1.0, vmax=1.0)
    ax.set_xticks(np.arange(len(metric_names)))
    ax.set_yticks(np.arange(len(metric_names)))
    ax.set_xticklabels(metric_names, rotation=90)
    ax.set_yticklabels(metric_names)
    ax.set_title("Validation Metric Correlation")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Scientific plotting for training history.json")
    parser.add_argument("--history", required=True, help="Path to history.json")
    parser.add_argument("--output_dir", required=True, help="Directory for plots")
    parser.add_argument("--smooth_window", type=int, default=11, help="Moving-average window")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    setup_logging(output_dir)
    history = load_history(Path(args.history))

    train_loss = series_from_pairs(history.get("train_loss", []))
    train_epe = series_from_pairs(history.get("train_epe", []))
    val_series = extract_val_metrics(history.get("val_metrics", []))

    save_summary_dashboard(
        output_dir / "summary_dashboard.png",
        history,
        train_loss,
        train_epe,
        val_series,
        args.smooth_window,
    )

    if train_loss[1].size:
        save_line_plot(
            output_dir / "train_loss.png",
            train_loss[0],
            train_loss[1],
            title="Training Loss vs Step",
            xlabel="Step",
            ylabel="Loss",
            smooth_window=args.smooth_window,
            color="tab:blue",
        )
    if train_epe[1].size:
        save_line_plot(
            output_dir / "train_epe.png",
            train_epe[0],
            train_epe[1],
            title="Training EPE vs Step",
            xlabel="Step",
            ylabel="EPE",
            smooth_window=args.smooth_window,
            color="tab:green",
        )

    grouped: Dict[str, List[Tuple[str, List[Tuple[float, float]]]]] = {}
    for metric_name, series in sorted(val_series.items()):
        grouped.setdefault(grouped_metric_name(metric_name), []).append((metric_name, series))

    for group_name, metric_items in grouped.items():
        save_metric_grid(output_dir / f"{group_name}.png", metric_items, args.smooth_window)

    save_metric_correlations(output_dir / "val_metric_correlation.png", val_series)

    with open(output_dir / "history_summary.json", "w") as handle:
        json.dump(
            {
                "run_name": history.get("run_name", ""),
                "total_steps": history.get("total_steps", 0),
                "best_epe": history.get("best_epe", float("nan")),
                "best_step": history.get("best_step", 0),
                "tracked_val_metrics": sorted(val_series.keys()),
            },
            handle,
            indent=2,
        )

    logger.info("History plots saved to %s", output_dir)


if __name__ == "__main__":
    main()