#!/usr/bin/env python3
"""Inspect intermediate model activations and public outputs.

This script runs inference on a few validation samples, captures intermediate
module activations via forward hooks, and writes scientific plots for every
selected tensor so model internals can be inspected visually.
"""
from __future__ import annotations

import argparse
import json
import logging
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple, TypeVar

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from omegaconf import OmegaConf
from tqdm import tqdm

from data import build_dataloader, build_dataset
from models import build_model
from utils import flow_to_hsv


logger = logging.getLogger(__name__)


def setup_logging(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(output_dir / "inspect_model_internals.log"),
        ],
    )


@dataclass
class ActivationRecord:
    name: str
    module_type: str
    tensor_index: int
    shape: List[int]
    dtype: str
    min_value: float
    max_value: float
    mean_value: float
    std_value: float
    l2_norm: float
    figure_path: str


def pick_device(device_arg: Optional[str]) -> torch.device:
    if device_arg:
        return torch.device(device_arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def normalise_image(image: torch.Tensor) -> np.ndarray:
    image_np = image.detach().cpu().permute(1, 2, 0).float().numpy()
    if image_np.max() > 2.0:
        image_np = image_np / 255.0
    return np.clip(image_np, 0.0, 1.0)


def extract_tensors(value: Any, prefix: str = "") -> List[Tuple[str, torch.Tensor]]:
    tensors: List[Tuple[str, torch.Tensor]] = []
    if isinstance(value, torch.Tensor):
        tensors.append((prefix or "tensor", value))
    elif isinstance(value, dict):
        for key, item in value.items():
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            tensors.extend(extract_tensors(item, child_prefix))
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            child_prefix = f"{prefix}[{index}]" if prefix else f"[{index}]"
            tensors.extend(extract_tensors(item, child_prefix))
    return tensors


T = TypeVar("T")


def truncate_items(items: Sequence[T], max_items: int) -> Sequence[T]:
    if max_items <= 0:
        return items
    return items[:max_items]


def tensor_stats(tensor: torch.Tensor) -> Dict[str, float]:
    tensor_float = tensor.detach().float().cpu()
    return {
        "min_value": float(tensor_float.min().item()),
        "max_value": float(tensor_float.max().item()),
        "mean_value": float(tensor_float.mean().item()),
        "std_value": float(tensor_float.std(unbiased=False).item()),
        "l2_norm": float(torch.linalg.vector_norm(tensor_float).item()),
    }


def summarise_feature_maps(
    tensor: torch.Tensor,
    max_channels: int,
    batch_index: int = 0,
) -> Tuple[np.ndarray, np.ndarray, List[Tuple[int, np.ndarray]]]:
    if tensor.ndim == 4:
        sample = tensor[batch_index]
    elif tensor.ndim == 3:
        sample = tensor
    else:
        raise ValueError(f"Expected 3D/4D tensor, got shape {tuple(tensor.shape)}")

    sample = sample.detach().float().cpu()
    channel_mean = sample.mean(dim=0).numpy()
    channel_std = sample.std(dim=0, unbiased=False).numpy()
    channel_energy = sample.square().mean(dim=(1, 2))
    topk = min(max_channels, sample.shape[0])
    top_indices = torch.topk(channel_energy, k=topk).indices.tolist()
    selected = [(idx, sample[idx].numpy()) for idx in top_indices]
    return channel_mean, channel_std, selected


def render_tensor_figure(
    tensor: torch.Tensor,
    title: str,
    output_path: Path,
    max_channels: int,
    batch_index: int = 0,
) -> None:
    tensor_cpu = tensor.detach().float().cpu()
    stats = tensor_stats(tensor_cpu)

    if tensor_cpu.ndim in (3, 4):
        mean_map, std_map, selected_channels = summarise_feature_maps(
            tensor_cpu, max_channels=max_channels, batch_index=batch_index
        )
        ncols = max(2, len(selected_channels))
        fig, axes = plt.subplots(2, ncols, figsize=(4 * ncols, 8))
        axes = np.atleast_2d(axes)

        axes[0, 0].imshow(mean_map, cmap="viridis")
        axes[0, 0].set_title("Channel Mean")
        axes[0, 0].axis("off")

        axes[0, 1].imshow(std_map, cmap="magma")
        axes[0, 1].set_title("Channel Std")
        axes[0, 1].axis("off")

        for col in range(2, ncols):
            axes[0, col].axis("off")

        for col, (channel_idx, channel_map) in enumerate(selected_channels):
            axes[1, col].imshow(channel_map, cmap="coolwarm")
            axes[1, col].set_title(f"Channel {channel_idx}")
            axes[1, col].axis("off")
        for col in range(len(selected_channels), ncols):
            axes[1, col].axis("off")

        fig.suptitle(
            (
                f"{title}\nshape={tuple(tensor_cpu.shape)}  "
                f"min={stats['min_value']:.4g}  max={stats['max_value']:.4g}  "
                f"mean={stats['mean_value']:.4g}  std={stats['std_value']:.4g}"
            ),
            fontsize=11,
        )
        fig.tight_layout()
        fig.savefig(output_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        return

    flat = tensor_cpu.reshape(-1).numpy()
    if tensor_cpu.ndim == 2:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].imshow(tensor_cpu.numpy(), cmap="viridis", aspect="auto")
        axes[0].set_title("Heat Map")
        axes[1].hist(flat, bins=80, color="tab:blue", alpha=0.85)
        axes[1].set_title("Value Distribution")
        axes[1].set_xlabel("Value")
        axes[1].set_ylabel("Count")
    else:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].plot(flat, linewidth=1.2)
        axes[0].set_title("Value Trace")
        axes[0].set_xlabel("Index")
        axes[0].set_ylabel("Value")
        axes[1].hist(flat, bins=80, color="tab:orange", alpha=0.85)
        axes[1].set_title("Value Distribution")
        axes[1].set_xlabel("Value")
        axes[1].set_ylabel("Count")

    fig.suptitle(
        (
            f"{title}\nshape={tuple(tensor_cpu.shape)}  "
            f"min={stats['min_value']:.4g}  max={stats['max_value']:.4g}  "
            f"mean={stats['mean_value']:.4g}  std={stats['std_value']:.4g}"
        ),
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def sanitise_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.\-]+", "_", name).strip("_") or "root"


def matches_patterns(name: str, patterns: Sequence[re.Pattern[str]]) -> bool:
    if not patterns:
        return True
    return any(pattern.search(name) for pattern in patterns)


def _extract_int_from_stem(stem: str) -> int:
    numbers = re.findall(r"\d+", stem)
    if not numbers:
        return -1
    return int(numbers[-1])


def _resolve_dataset_record(dataset: Any, global_index: int) -> Optional[Dict[str, Any]]:
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
    if record is None:
        return {
            "scene": "",
            "sample_id": fallback_name,
            "frame1": fallback_name,
            "frame2": fallback_name,
            "sort_key": -1,
            "has_scene": False,
        }

    image1 = record.get("image1")
    image2 = record.get("image2")
    if not image1:
        return {
            "scene": "",
            "sample_id": fallback_name,
            "frame1": fallback_name,
            "frame2": fallback_name,
            "sort_key": -1,
            "has_scene": False,
        }

    p1 = Path(image1)
    p2 = Path(image2) if image2 else p1
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
    sample_id = f"{frame1}__{frame2}".replace("/", "_")
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


def _qualitative_selection(request: str, dataset_len: int, seed: int) -> Optional[Set[int]]:
    req = str(request).strip().lower()
    if req == "all":
        return None

    n = int(req)
    if n <= 0:
        return set()
    n = min(n, dataset_len)

    rng = np.random.default_rng(seed)
    chosen = rng.choice(dataset_len, size=n, replace=False)
    return set(int(x) for x in chosen.tolist())


def register_activation_hooks(
    model: nn.Module,
    include_patterns: Sequence[re.Pattern[str]],
    exclude_patterns: Sequence[re.Pattern[str]],
    leaf_only: bool,
    max_modules: int,
) -> Tuple[List[torch.utils.hooks.RemovableHandle], List[str], Dict[str, Any]]:
    captured: Dict[str, Any] = {}
    handles: List[torch.utils.hooks.RemovableHandle] = []
    selected_names: List[str] = []

    for module_name, module in model.named_modules():
        if not module_name:
            continue
        if leaf_only and any(True for _ in module.children()):
            continue
        if not matches_patterns(module_name, include_patterns):
            continue
        if exclude_patterns and matches_patterns(module_name, exclude_patterns):
            continue
        if len(selected_names) >= max_modules:
            break

        def _make_hook(name: str):
            def _hook(_module: nn.Module, _inputs: Tuple[Any, ...], output: Any) -> None:
                captured[name] = output
            return _hook

        handles.append(module.register_forward_hook(_make_hook(module_name)))
        selected_names.append(module_name)

    return handles, selected_names, captured


def remove_hooks(handles: Iterable[torch.utils.hooks.RemovableHandle]) -> None:
    for handle in handles:
        handle.remove()


def plot_flow_series(
    flow_series: Sequence[torch.Tensor],
    output_path: Path,
    title: str,
    batch_index: int = 0,
) -> None:
    if not flow_series:
        return
    ncols = len(flow_series)
    fig, axes = plt.subplots(1, ncols, figsize=(4 * ncols, 4))
    if ncols == 1:
        axes = [axes]

    for axis, flow in zip(axes, flow_series):
        flow_b = flow[batch_index].detach().cpu()
        axis.imshow(flow_to_hsv(flow_b))
        axis.axis("off")

    for index, axis in enumerate(axes):
        axis.set_title(f"{title} {index + 1}")
    fig.tight_layout()
    fig.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def to_sample_map(tensor: torch.Tensor, batch_index: int = 0) -> Optional[torch.Tensor]:
    sample = tensor.detach().float().cpu()
    if sample.ndim == 4:
        sample = sample[batch_index]
    if sample.ndim == 2:
        return sample
    if sample.ndim == 3:
        return sample
    return None


def collect_stage_tensor_groups(outputs: Dict[str, Any]) -> Dict[str, List[torch.Tensor]]:
    grouped: Dict[str, List[torch.Tensor]] = {}
    for key, value in outputs.items():
        if isinstance(value, torch.Tensor):
            grouped[key] = [value]
        elif isinstance(value, (list, tuple)) and value and all(isinstance(item, torch.Tensor) for item in value):
            grouped[key] = list(value)
    return grouped


def plot_grouped_stage_tensors(
    grouped: Dict[str, List[torch.Tensor]],
    output_path: Path,
    title: str,
    batch_index: int,
    mode: str,
) -> None:
    if not grouped:
        return

    keys = list(grouped.keys())
    nrows = len(keys)
    ncols = max(len(grouped[key]) for key in keys)
    fig_w = min(4.0 * ncols, 24.0)
    fig_h = max(3.0 * nrows, 3.5)
    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_w, fig_h), squeeze=False)

    for row, key in enumerate(keys):
        series = grouped[key]
        for col in range(ncols):
            axis = axes[row, col]
            axis.axis("off")
            if col >= len(series):
                continue

            sample = to_sample_map(series[col], batch_index=batch_index)
            if sample is None:
                continue

            if mode == "flow" and sample.ndim == 3 and sample.shape[0] == 2:
                axis.imshow(flow_to_hsv(sample))
            elif mode == "scalar":
                if sample.ndim == 3 and sample.shape[0] > 1:
                    axis.imshow(sample.mean(dim=0).numpy(), cmap="viridis")
                elif sample.ndim == 3:
                    axis.imshow(sample[0].numpy(), cmap="viridis")
                else:
                    axis.imshow(sample.numpy(), cmap="viridis", aspect="auto")
            elif mode == "feature":
                if sample.ndim != 3:
                    continue
                axis.imshow(sample.mean(dim=0).numpy(), cmap="magma")
            else:
                continue

            axis.set_title(f"{key} [{col}]", fontsize=9)

    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_grouped_public_outputs(
    sample_dir: Path,
    outputs: Dict[str, Any],
    file_prefix: str,
    batch_index: int = 0,
) -> None:
    grouped = collect_stage_tensor_groups(outputs)
    flow_groups: Dict[str, List[torch.Tensor]] = {}
    occupancy_groups: Dict[str, List[torch.Tensor]] = {}
    scalar_groups: Dict[str, List[torch.Tensor]] = {}
    feature_groups: Dict[str, List[torch.Tensor]] = {}

    for key, series in grouped.items():
        sample = to_sample_map(series[0], batch_index=batch_index)
        if sample is None:
            continue
        if sample.ndim == 3 and sample.shape[0] == 2:
            flow_groups[key] = series
        elif "occupancy" in key.lower() and (sample.ndim == 2 or (sample.ndim == 3 and sample.shape[0] == 1)):
            occupancy_groups[key] = series
        elif sample.ndim == 2 or (sample.ndim == 3 and sample.shape[0] == 1):
            scalar_groups[key] = series
        elif sample.ndim == 3:
            feature_groups[key] = series

    plot_grouped_stage_tensors(
        grouped=flow_groups,
        output_path=sample_dir / f"{file_prefix}_stage_flows_grouped.png",
        title="Stage Flow Outputs (HSV)",
        batch_index=batch_index,
        mode="flow",
    )
    plot_grouped_stage_tensors(
        grouped=occupancy_groups,
        output_path=sample_dir / f"{file_prefix}_stage_occupancy_grouped.png",
        title="Stage Occupancy Masks",
        batch_index=batch_index,
        mode="scalar",
    )
    plot_grouped_stage_tensors(
        grouped=scalar_groups,
        output_path=sample_dir / f"{file_prefix}_stage_scalars_grouped.png",
        title="Stage Scalar/Mask Outputs",
        batch_index=batch_index,
        mode="scalar",
    )
    plot_grouped_stage_tensors(
        grouped=feature_groups,
        output_path=sample_dir / f"{file_prefix}_stage_features_grouped.png",
        title="Stage Feature Outputs (Channel Mean)",
        batch_index=batch_index,
        mode="feature",
    )


def plot_master_dashboard(sample_dir: Path, file_prefix: str) -> None:
    panel_paths = [
        sample_dir / f"{file_prefix}_sample_overview.png",
        sample_dir / f"{file_prefix}_stage_flows_grouped.png",
        sample_dir / f"{file_prefix}_stage_occupancy_grouped.png",
        sample_dir / f"{file_prefix}_stage_scalars_grouped.png",
        sample_dir / f"{file_prefix}_stage_features_grouped.png",
    ]
    existing = [path for path in panel_paths if path.exists()]
    if not existing:
        return

    n_panels = len(existing)
    ncols = 2 if n_panels > 1 else 1
    nrows = int(np.ceil(n_panels / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(10 * ncols, 6 * nrows), squeeze=False)

    for index, axis in enumerate(axes.ravel()):
        axis.axis("off")
        if index >= n_panels:
            continue

        panel_path = existing[index]
        image = plt.imread(panel_path)
        axis.imshow(image)
        axis.set_title(panel_path.stem.replace("_", " ").title(), fontsize=11)

    fig.suptitle("Model Internal Debug Dashboard", fontsize=13)
    fig.tight_layout()
    fig.savefig(sample_dir / f"{file_prefix}_master_dashboard.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_sample_overview(
    sample_dir: Path,
    batch: Dict[str, torch.Tensor],
    outputs: Dict[str, Any],
    file_prefix: str,
    batch_index: int = 0,
) -> None:
    has_gt = "flow" in batch and isinstance(batch["flow"], torch.Tensor)
    has_occupancy = (
        isinstance(outputs, dict)
        and isinstance(outputs.get("occupancy_masks"), (list, tuple))
        and len(outputs["occupancy_masks"]) > 0
        and isinstance(outputs["occupancy_masks"][0], torch.Tensor)
    )

    cols = 2 + int(has_gt) + int(has_occupancy)
    fig, axes = plt.subplots(1, cols, figsize=(5 * cols, 4))
    axes = np.atleast_1d(axes)

    image1 = batch["image1"][batch_index]
    image2 = batch["image2"][batch_index]
    axes[0].imshow(normalise_image(image1))
    axes[0].set_title("Image 1")
    axes[0].axis("off")
    axes[1].imshow(normalise_image(image2))
    axes[1].set_title("Image 2")
    axes[1].axis("off")

    col = 2
    if has_gt:
        axes[col].imshow(flow_to_hsv(batch["flow"][batch_index].detach().cpu()))
        axes[col].set_title("Ground Truth Flow")
        axes[col].axis("off")
        col += 1

    if has_occupancy:
        occupancy_series = outputs["occupancy_masks"]
        occupancy_map = to_sample_map(occupancy_series[-1], batch_index=batch_index)
        if occupancy_map is not None:
            if occupancy_map.ndim == 3 and occupancy_map.shape[0] == 1:
                occupancy_map = occupancy_map[0]
            axes[col].imshow(occupancy_map.numpy(), cmap="viridis", vmin=0.0, vmax=1.0)
            axes[col].set_title("Occupancy Mask (latest)")
            axes[col].axis("off")

    fig.tight_layout()
    fig.savefig(sample_dir / f"{file_prefix}_sample_overview.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    if isinstance(outputs, dict):
        plot_grouped_public_outputs(sample_dir, outputs, file_prefix=file_prefix, batch_index=batch_index)
    plot_master_dashboard(sample_dir, file_prefix=file_prefix)


def inspect_sample(
    batch: Dict[str, Any],
    outputs: Any,
    model: nn.Module,
    sample_dir: Path,
    sample_id: str,
    batch_index: int,
    selected_module_names: Sequence[str],
    captured_outputs: Dict[str, Any],
    max_channels: int,
    max_tensors_per_module: int,
) -> None:
    sample_dir.mkdir(parents=True, exist_ok=True)
    plot_sample_overview(sample_dir, batch, outputs, file_prefix=sample_id, batch_index=batch_index)

    activation_dir = sample_dir / "activations" / sample_id
    activation_dir.mkdir(parents=True, exist_ok=True)

    records: List[ActivationRecord] = []
    for module_name in selected_module_names:
        module_output = captured_outputs.get(module_name)
        if module_output is None:
            continue

        tensors = extract_tensors(module_output)
        selected_tensors = truncate_items(tensors, max_tensors_per_module)
        for tensor_index, (_, tensor) in enumerate(selected_tensors):
            if tensor.ndim == 0:
                tensor = tensor.reshape(1)
            figure_name = f"{len(records):04d}_{sanitise_name(module_name)}_{tensor_index}.png"
            figure_path = activation_dir / figure_name
            render_tensor_figure(
                tensor=tensor,
                title=f"{module_name} :: tensor {tensor_index}",
                output_path=figure_path,
                max_channels=max_channels,
                batch_index=batch_index,
            )
            stats = tensor_stats(tensor)
            records.append(
                ActivationRecord(
                    name=module_name,
                    module_type=type(dict(model.named_modules())[module_name]).__name__,
                    tensor_index=tensor_index,
                    shape=list(tensor.shape),
                    dtype=str(tensor.dtype),
                    figure_path=str(figure_path.relative_to(sample_dir)),
                    **stats,
                )
            )

    if isinstance(outputs, dict):
        public_items = list(outputs.items())
    else:
        public_items = [("model_output", outputs)]

    for public_name, public_value in public_items:
        public_tensors = extract_tensors(public_value, public_name)
        public_dir = sample_dir / "public_outputs" / sample_id
        public_dir.mkdir(parents=True, exist_ok=True)
        selected_public_tensors = truncate_items(public_tensors, max_tensors_per_module)
        for tensor_index, (tensor_name, tensor) in enumerate(selected_public_tensors):
            if tensor.ndim == 0:
                tensor = tensor.reshape(1)
            figure_name = f"{sanitise_name(tensor_name)}_{tensor_index}.png"
            render_tensor_figure(
                tensor=tensor,
                title=tensor_name,
                output_path=public_dir / figure_name,
                max_channels=max_channels,
                batch_index=batch_index,
            )

    with open(sample_dir / f"{sample_id}_activation_index.json", "w") as handle:
        json.dump([asdict(record) for record in records], handle, indent=2)


def parse_regex_list(values: Optional[Sequence[str]]) -> List[re.Pattern[str]]:
    return [re.compile(value) for value in values or []]


def move_batch_to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    moved: Dict[str, Any] = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            moved[key] = value.to(device, non_blocking=True)
        else:
            moved[key] = value
    return moved


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect intermediate HQSFlow activations")
    parser.add_argument("--config", "-c", required=True, help="Path to model config YAML")
    parser.add_argument("--checkpoint", "-ckpt", required=True, help="Path to model checkpoint")
    parser.add_argument("--data_config", "-dc", required=True, help="Path to data config YAML")
    parser.add_argument("--output_dir", "-o", required=True, help="Directory for plots")
    parser.add_argument("--device", default=None, help="Device override: cuda, mps, or cpu")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for inspection")
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
    parser.add_argument(
        "--num_samples",
        type=int,
        default=None,
        help="Deprecated alias for qualitative sample count; use --qualitative_samples",
    )
    parser.add_argument(
        "--include",
        nargs="*",
        default=None,
        help="Regex filters for module names to include. Default captures all selected modules.",
    )
    parser.add_argument(
        "--exclude",
        nargs="*",
        default=[r".*dropout.*"],
        help="Regex filters for module names to exclude.",
    )
    parser.add_argument(
        "--max_modules",
        type=int,
        default=0,
        help="Maximum number of hooked modules. Use 0 for all matching modules.",
    )
    parser.add_argument("--max_channels", type=int, default=6, help="Top channels plotted for 3D/4D tensors")
    parser.add_argument(
        "--max_tensors_per_module",
        type=int,
        default=0,
        help="Maximum plotted tensors per module output (0 = all)",
    )
    parser.add_argument(
        "--all_modules",
        action="store_true",
        help="Hook non-leaf modules as well. Default only hooks leaf modules.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    setup_logging(output_dir)
    device = pick_device(args.device)
    logger.info("Device: %s", device)

    model_cfg = OmegaConf.load(args.config)
    data_cfg = OmegaConf.load(args.data_config)

    model = build_model(model_cfg).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    state_dict = ckpt.get("model", ckpt) if isinstance(ckpt, dict) else ckpt
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    include_patterns = parse_regex_list(args.include)
    exclude_patterns = parse_regex_list(args.exclude)
    max_modules = args.max_modules if args.max_modules > 0 else 10**9
    handles, selected_module_names, captured_outputs = register_activation_hooks(
        model=model,
        include_patterns=include_patterns,
        exclude_patterns=exclude_patterns,
        leaf_only=not args.all_modules,
        max_modules=max_modules,
    )
    logger.info("Hooked %d modules", len(selected_module_names))

    val_cfg = data_cfg.get("val_data") or data_cfg.get("data") or data_cfg
    val_cfg = OmegaConf.merge(val_cfg, {"batch_size": args.batch_size})
    dataset = build_dataset(val_cfg, split="val")
    dataloader = build_dataloader(dataset, val_cfg, split="val")
    logger.info("Validation samples available: %d", len(dataset))

    qualitative_request = str(args.num_samples) if args.num_samples is not None else args.qualitative_samples
    qual_set = _qualitative_selection(qualitative_request, len(dataset), args.qualitative_seed)
    target_count = len(dataset) if qual_set is None else len(qual_set)
    logger.info("Qualitative selection: %s (%d samples)", qualitative_request, target_count)

    qualitative_root = output_dir / "qualitative"
    qualitative_root.mkdir(parents=True, exist_ok=True)

    manifest = {
        "config": args.config,
        "checkpoint": args.checkpoint,
        "data_config": args.data_config,
        "device": str(device),
        "batch_size": args.batch_size,
        "qualitative_samples": qualitative_request,
        "qualitative_seed": args.qualitative_seed,
        "hooked_modules": selected_module_names,
        "samples": [],
    }

    try:
        with torch.no_grad():
            processed = 0
            global_offset = 0
            with tqdm(total=target_count, desc="Saving qualitative samples", unit="sample") as sample_pbar:
                for batch_idx, batch in enumerate(tqdm(dataloader, desc="Inspecting", unit="batch")):
                    if processed >= target_count:
                        break
                    batch = move_batch_to_device(batch, device)

                    captured_outputs.clear()
                    outputs = model(batch["image1"], batch["image2"])

                    batch_size = int(batch["image1"].shape[0])
                    for batch_item in range(batch_size):
                        global_idx = global_offset + batch_item
                        if qual_set is not None and global_idx not in qual_set:
                            continue

                        sample_name = f"batch_{batch_idx:06d}_item_{batch_item:02d}"
                        record = _resolve_dataset_record(dataset, global_idx)
                        sample_meta = _infer_scene_and_sample(record, fallback_name=sample_name)
                        scene_dir = _scene_subdir(qualitative_root, sample_meta["scene"])
                        sample_dir = scene_dir

                        inspect_sample(
                            batch=batch,
                            outputs=outputs,
                            model=model,
                            sample_dir=sample_dir,
                            sample_id=sample_meta["sample_id"],
                            batch_index=batch_item,
                            selected_module_names=selected_module_names,
                            captured_outputs=captured_outputs,
                            max_channels=args.max_channels,
                            max_tensors_per_module=args.max_tensors_per_module,
                        )
                        manifest["samples"].append(
                            {
                                "scene": sample_meta["scene"],
                                "sample_id": sample_meta["sample_id"],
                                "path": str((sample_dir / f"{sample_meta['sample_id']}_master_dashboard.png").relative_to(output_dir)),
                            }
                        )
                        processed += 1
                        sample_pbar.update(1)
                        if processed >= target_count:
                            break

                    global_offset += batch_size
    finally:
        remove_hooks(handles)

    with open(output_dir / "manifest.json", "w") as handle:
        json.dump(manifest, handle, indent=2)
    logger.info("Activation plots saved to %s", output_dir)


if __name__ == "__main__":
    main()