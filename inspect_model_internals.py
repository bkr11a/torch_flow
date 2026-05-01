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
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

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


def plot_sample_overview(
    sample_dir: Path,
    batch: Dict[str, torch.Tensor],
    outputs: Dict[str, Any],
    batch_index: int = 0,
) -> None:
    has_gt = "flow" in batch and isinstance(batch["flow"], torch.Tensor)
    cols = 3 if has_gt else 2
    fig, axes = plt.subplots(1, cols, figsize=(5 * cols, 4))
    if cols == 2:
        axes = [axes[0], axes[1]]

    image1 = batch["image1"][batch_index]
    image2 = batch["image2"][batch_index]
    axes[0].imshow(normalise_image(image1))
    axes[0].set_title("Image 1")
    axes[0].axis("off")
    axes[1].imshow(normalise_image(image2))
    axes[1].set_title("Image 2")
    axes[1].axis("off")

    if has_gt:
        axes[2].imshow(flow_to_hsv(batch["flow"][batch_index].detach().cpu()))
        axes[2].set_title("Ground Truth Flow")
        axes[2].axis("off")

    fig.tight_layout()
    fig.savefig(sample_dir / "sample_overview.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

    if "flow_preds" in outputs:
        plot_flow_series(outputs["flow_preds"], sample_dir / "flow_preds.png", "Stage")
    if "flow_preds_raw" in outputs:
        plot_flow_series(outputs["flow_preds_raw"], sample_dir / "flow_preds_raw.png", "Raw Stage")
    if "flow_low" in outputs:
        plot_flow_series(outputs["flow_low"], sample_dir / "flow_low.png", "Low-Res Stage")


def inspect_sample(
    model: nn.Module,
    batch: Dict[str, Any],
    sample_dir: Path,
    selected_module_names: Sequence[str],
    captured_outputs: Dict[str, Any],
    max_channels: int,
    max_tensors_per_module: int,
) -> None:
    image1 = batch["image1"]
    image2 = batch["image2"]
    outputs = model(image1, image2)

    sample_dir.mkdir(parents=True, exist_ok=True)
    plot_sample_overview(sample_dir, batch, outputs)

    activation_dir = sample_dir / "activations"
    activation_dir.mkdir(parents=True, exist_ok=True)

    records: List[ActivationRecord] = []
    for module_name in selected_module_names:
        module_output = captured_outputs.get(module_name)
        if module_output is None:
            continue

        tensors = extract_tensors(module_output)
        for tensor_index, (_, tensor) in enumerate(tensors[:max_tensors_per_module]):
            if tensor.ndim == 0:
                tensor = tensor.reshape(1)
            figure_name = f"{len(records):04d}_{sanitise_name(module_name)}_{tensor_index}.png"
            figure_path = activation_dir / figure_name
            render_tensor_figure(
                tensor=tensor,
                title=f"{module_name} :: tensor {tensor_index}",
                output_path=figure_path,
                max_channels=max_channels,
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
        public_dir = sample_dir / "public_outputs"
        public_dir.mkdir(parents=True, exist_ok=True)
        for tensor_index, (tensor_name, tensor) in enumerate(public_tensors[:max_tensors_per_module * 4]):
            if tensor.ndim == 0:
                tensor = tensor.reshape(1)
            figure_name = f"{sanitise_name(tensor_name)}_{tensor_index}.png"
            render_tensor_figure(
                tensor=tensor,
                title=tensor_name,
                output_path=public_dir / figure_name,
                max_channels=max_channels,
            )

    with open(sample_dir / "activation_index.json", "w") as handle:
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
    parser.add_argument("--config", required=True, help="Path to model config YAML")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
    parser.add_argument("--data_config", required=True, help="Path to data config YAML")
    parser.add_argument("--output_dir", required=True, help="Directory for plots")
    parser.add_argument("--num_samples", type=int, default=2, help="Number of validation samples")
    parser.add_argument("--device", default=None, help="Device override: cuda, mps, or cpu")
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
        default=2,
        help="Maximum plotted tensors per module output",
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
    val_cfg = OmegaConf.merge(val_cfg, {"batch_size": 1})
    dataset = build_dataset(val_cfg, split="val")
    dataloader = build_dataloader(dataset, val_cfg, split="val")
    logger.info("Validation samples available: %d", len(dataset))

    manifest = {
        "config": args.config,
        "checkpoint": args.checkpoint,
        "data_config": args.data_config,
        "device": str(device),
        "hooked_modules": selected_module_names,
        "samples": [],
    }

    try:
        with torch.no_grad():
            processed = 0
            for batch in tqdm(dataloader, desc="Inspecting"):
                if processed >= args.num_samples:
                    break
                batch = move_batch_to_device(batch, device)
                sample_dir = output_dir / f"sample_{processed:04d}"
                inspect_sample(
                    model=model,
                    batch=batch,
                    sample_dir=sample_dir,
                    selected_module_names=selected_module_names,
                    captured_outputs=captured_outputs,
                    max_channels=args.max_channels,
                    max_tensors_per_module=args.max_tensors_per_module,
                )
                manifest["samples"].append(str(sample_dir.relative_to(output_dir)))
                processed += 1
    finally:
        remove_hooks(handles)

    with open(output_dir / "manifest.json", "w") as handle:
        json.dump(manifest, handle, indent=2)
    logger.info("Activation plots saved to %s", output_dir)


if __name__ == "__main__":
    main()