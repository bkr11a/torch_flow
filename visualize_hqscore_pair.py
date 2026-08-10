#!/usr/bin/env python3
"""Export the principal HQSCore inference fields for one image pair.

This script is intended to be run from the repository root.  The minimal
invocation is::

    python visualize_hqscore_pair.py FRAME_1 FRAME_2 CHECKPOINT

The checkpoint's embedded resolved configuration is used when available.
For a weights-only checkpoint, the script falls back to ``configs/default.yaml``
merged with ``configs/dropins/06_hqs_core.yaml``.  ``--config`` and
``--override`` can be supplied to select another compatible HQSCore setup.

The output directory contains six PNG files:

* ``01_input_frame_1.png``
* ``02_input_frame_2.png``
* ``03_initial_flow.png``
* ``04_global_matching_confidence.png``
* ``05_final_validity.png``
* ``06_final_flow.png``

The initial and final flows share one Middlebury colour normalisation.  The
confidence and validity images are 8-bit greyscale maps in which black is zero
and white is one.  Final validity is the effective data validity ``m = b * a``
from the last HQS iteration, rather than learned reliability ``a`` alone.
"""
from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

# Allow unsupported MPS operations to fall back to CPU where PyTorch permits.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from PIL import Image, ImageOps

from models import build_model, resize_flow
from utils import InputPadder, flow_to_color


LOGGER = logging.getLogger("visualize_hqscore_pair")
REPOSITORY_ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run HQSCore on one frame pair and save its inputs, global "
            "initialisation, confidence, final validity, and final flow."
        )
    )
    parser.add_argument("image1", type=Path, help="First/source RGB frame")
    parser.add_argument("image2", type=Path, help="Second/target RGB frame")
    parser.add_argument("checkpoint", type=Path, help="HQSCore .pth checkpoint")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/hqscore_pair"),
        help="Destination directory (default: outputs/hqscore_pair)",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help=(
            "Optional base YAML. If omitted, use the checkpoint configuration; "
            "weights-only checkpoints fall back to configs/default.yaml."
        ),
    )
    parser.add_argument(
        "--override",
        type=Path,
        action="append",
        default=None,
        help=(
            "Optional YAML overlay. May be repeated. For a weights-only "
            "checkpoint, 06_hqs_core.yaml is used when this option is omitted."
        ),
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Inference device: auto, cuda, cuda:0, mps, or cpu (default: auto)",
    )
    parser.add_argument(
        "--max-flow",
        type=float,
        default=None,
        help=(
            "Maximum flow magnitude used by both flow colour maps. The default "
            "is their shared 99th-percentile magnitude."
        ),
    )
    parser.add_argument(
        "--amp",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable/disable CUDA automatic mixed precision (default: enabled on CUDA)",
    )
    parser.add_argument(
        "--non-strict",
        action="store_true",
        help="Permit missing or unexpected checkpoint keys (diagnostic use only)",
    )
    return parser.parse_args()


def _require_file(path: Path, label: str) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"{label} does not exist: {resolved}")
    return resolved


def _resolve_device(requested: str) -> torch.device:
    if requested != "auto":
        device = torch.device(requested)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        if device.type == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested but is not available")
        return device
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _as_config(value: Any) -> DictConfig:
    if isinstance(value, DictConfig):
        return value
    if isinstance(value, Mapping):
        return OmegaConf.create(dict(value))
    raise TypeError(
        "Checkpoint field 'cfg' is not an OmegaConf configuration or mapping"
    )


def _load_configuration(
    args: argparse.Namespace,
    checkpoint: Any,
) -> DictConfig:
    explicit_overrides: Sequence[Path] = args.override or ()

    if args.config is not None:
        config_path = _require_file(args.config, "Config")
        cfg = OmegaConf.load(config_path)
        LOGGER.info("Configuration: %s", config_path)
    elif isinstance(checkpoint, Mapping) and checkpoint.get("cfg") is not None:
        cfg = _as_config(checkpoint["cfg"])
        LOGGER.info("Configuration: embedded resolved checkpoint configuration")
    else:
        config_path = REPOSITORY_ROOT / "configs/default.yaml"
        cfg = OmegaConf.load(_require_file(config_path, "Fallback config"))
        LOGGER.info("Configuration: %s", config_path)
        if not explicit_overrides:
            explicit_overrides = (
                REPOSITORY_ROOT / "configs/dropins/06_hqs_core.yaml",
            )

    for override_path in explicit_overrides:
        resolved = _require_file(override_path, "Config override")
        cfg = OmegaConf.merge(cfg, OmegaConf.load(resolved))
        LOGGER.info("Configuration overlay: %s", resolved)

    return cfg


def _checkpoint_state(checkpoint: Any) -> Mapping[str, torch.Tensor]:
    if not isinstance(checkpoint, Mapping):
        if isinstance(checkpoint, dict):
            return checkpoint
        raise TypeError("Checkpoint must contain a PyTorch state dictionary")

    for key in ("model", "state_dict", "model_state_dict"):
        value = checkpoint.get(key)
        if isinstance(value, Mapping):
            state = dict(value)
            break
    else:
        state = dict(checkpoint)

    # DataParallel checkpoints sometimes add this prefix to every parameter.
    if state and all(str(key).startswith("module.") for key in state):
        state = {str(key)[len("module.") :]: value for key, value in state.items()}
    return state


def _load_rgb(path: Path) -> tuple[Image.Image, torch.Tensor]:
    with Image.open(path) as opened:
        image = ImageOps.exif_transpose(opened).convert("RGB")
        image.load()
    array = np.asarray(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array.copy()).permute(2, 0, 1).contiguous()
    return image, tensor


def _require_tensor(outputs: Mapping[str, Any], key: str) -> torch.Tensor:
    value = outputs.get(key)
    if not isinstance(value, torch.Tensor):
        raise RuntimeError(
            f"HQSCore output '{key}' is missing or is not a tensor. "
            "Use the pgma HQSCore implementation at commit f9533fa9 or later."
        )
    return value


def _require_tensor_list(
    outputs: Mapping[str, Any], key: str
) -> list[torch.Tensor]:
    value = outputs.get(key)
    if not isinstance(value, (list, tuple)) or not value:
        raise RuntimeError(
            f"HQSCore output '{key}' is missing or empty. "
            "Use the pgma HQSCore implementation at commit f9533fa9 or later."
        )
    if not all(isinstance(item, torch.Tensor) for item in value):
        raise RuntimeError(f"HQSCore output '{key}' contains a non-tensor value")
    return list(value)


def _unpad_full_flow(
    low_flow: torch.Tensor,
    padded_size: tuple[int, int],
    padder: InputPadder,
) -> torch.Tensor:
    full = resize_flow(low_flow.float(), padded_size)
    return padder.unpad(full)


def _unpad_scalar_map(
    scalar: torch.Tensor,
    padded_size: tuple[int, int],
    padder: InputPadder,
) -> torch.Tensor:
    full = F.interpolate(
        scalar.float(), size=padded_size, mode="bilinear", align_corners=False
    )
    return padder.unpad(full).clamp(0.0, 1.0)


def _finite_or_raise(tensor: torch.Tensor, label: str) -> None:
    if not bool(torch.isfinite(tensor).all().item()):
        count = int((~torch.isfinite(tensor)).sum().item())
        raise RuntimeError(f"{label} contains {count} NaN/Inf values")


def _flow_array(flow: torch.Tensor) -> np.ndarray:
    if flow.shape != (1, 2, *flow.shape[-2:]):
        raise ValueError(f"Expected flow [1,2,H,W], got {tuple(flow.shape)}")
    return flow[0].permute(1, 2, 0).detach().cpu().numpy().astype(np.float32)


def _scalar_array(scalar: torch.Tensor) -> np.ndarray:
    if scalar.shape != (1, 1, *scalar.shape[-2:]):
        raise ValueError(f"Expected scalar map [1,1,H,W], got {tuple(scalar.shape)}")
    return scalar[0, 0].detach().cpu().numpy().astype(np.float32)


def _shared_flow_scale(
    initial_flow: np.ndarray,
    final_flow: np.ndarray,
    requested: Optional[float],
) -> float:
    if requested is not None:
        if not np.isfinite(requested) or requested <= 0.0:
            raise ValueError("--max-flow must be a finite positive value")
        return float(requested)

    magnitudes = np.concatenate(
        [
            np.linalg.norm(initial_flow, axis=-1).reshape(-1),
            np.linalg.norm(final_flow, axis=-1).reshape(-1),
        ]
    )
    finite = magnitudes[np.isfinite(magnitudes)]
    if finite.size == 0:
        raise RuntimeError("The initial and final flow contain no finite values")
    return max(float(np.percentile(finite, 99.0)), 1.0)


def _save_scalar_png(array: np.ndarray, path: Path) -> None:
    image = np.rint(np.clip(array, 0.0, 1.0) * 255.0).astype(np.uint8)
    Image.fromarray(image, mode="L").save(path)


@torch.inference_mode()
def _run_inference(
    model: torch.nn.Module,
    image1: torch.Tensor,
    image2: torch.Tensor,
    device: torch.device,
    use_amp: bool,
) -> tuple[dict[str, Any], InputPadder, tuple[int, int]]:
    input1 = image1.unsqueeze(0).to(device, non_blocking=True)
    input2 = image2.unsqueeze(0).to(device, non_blocking=True)
    padder = InputPadder(input1.shape, divisor=8)
    input1, input2 = padder.pad(input1, input2)
    padded_size = tuple(int(value) for value in input1.shape[-2:])

    model.eval()
    with torch.autocast(device_type=device.type, enabled=use_amp):
        outputs = model(input1, input2)
    if not isinstance(outputs, dict):
        raise RuntimeError(f"Expected HQSCore to return a dict, got {type(outputs)!r}")
    return outputs, padder, padded_size


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%H:%M:%S",
    )

    image1_path = _require_file(args.image1, "Frame 1")
    image2_path = _require_file(args.image2, "Frame 2")
    checkpoint_path = _require_file(args.checkpoint, "Checkpoint")

    device = _resolve_device(args.device)
    use_amp = device.type == "cuda" if args.amp is None else bool(args.amp)
    if use_amp and device.type != "cuda":
        raise ValueError("--amp is supported only for CUDA inference")
    LOGGER.info("Device: %s | AMP: %s", device, use_amp)

    # The repository's checkpoints contain trusted Python configuration
    # objects, so their established loading convention uses weights_only=False.
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False
    )
    cfg = _load_configuration(args, checkpoint)
    model = build_model(cfg)
    if model.__class__.__name__ != "HQSCore":
        raise ValueError(
            "The resolved configuration did not construct HQSCore; got "
            f"{model.__class__.__name__}. Supply the matching HQSCore config/override."
        )

    state = _checkpoint_state(checkpoint)
    incompatible = model.load_state_dict(state, strict=not args.non_strict)
    if incompatible.missing_keys:
        LOGGER.warning("Missing checkpoint keys: %s", incompatible.missing_keys)
    if incompatible.unexpected_keys:
        LOGGER.warning("Unexpected checkpoint keys: %s", incompatible.unexpected_keys)
    model = model.to(device)
    LOGGER.info(
        "Loaded %s (%s parameters)",
        checkpoint_path,
        f"{sum(parameter.numel() for parameter in model.parameters()):,}",
    )

    input_image1, input_tensor1 = _load_rgb(image1_path)
    input_image2, input_tensor2 = _load_rgb(image2_path)
    if input_tensor1.shape != input_tensor2.shape:
        raise ValueError(
            "Input frames must have identical RGB dimensions; got "
            f"{tuple(input_tensor1.shape)} and {tuple(input_tensor2.shape)}"
        )

    outputs, padder, padded_size = _run_inference(
        model, input_tensor1, input_tensor2, device, use_amp
    )

    # This is q_0 after all-pairs decoding and confidence gating: the state
    # actually supplied to the first HQS iteration.
    initial_low = _require_tensor(outputs, "global_init_flow_xy")
    confidence_low = _require_tensor(outputs, "global_init_confidence")
    validity_lows = _require_tensor_list(outputs, "data_valid_lows")
    final_padded = _require_tensor_list(outputs, "flow_preds")[-1]

    initial_flow = _unpad_full_flow(initial_low, padded_size, padder)
    confidence = _unpad_scalar_map(confidence_low, padded_size, padder)
    final_validity = _unpad_scalar_map(validity_lows[-1], padded_size, padder)
    final_flow = padder.unpad(final_padded.float())

    for tensor, label in (
        (initial_flow, "Initial flow"),
        (confidence, "Global matching confidence"),
        (final_validity, "Final validity"),
        (final_flow, "Final flow"),
    ):
        _finite_or_raise(tensor, label)

    initial_array = _flow_array(initial_flow)
    final_array = _flow_array(final_flow)
    confidence_array = _scalar_array(confidence)
    validity_array = _scalar_array(final_validity)
    common_max_flow = _shared_flow_scale(
        initial_array, final_array, args.max_flow
    )

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    input_image1.save(output_dir / "01_input_frame_1.png")
    input_image2.save(output_dir / "02_input_frame_2.png")
    Image.fromarray(
        flow_to_color(initial_array, max_flow=common_max_flow), mode="RGB"
    ).save(output_dir / "03_initial_flow.png")
    _save_scalar_png(
        confidence_array, output_dir / "04_global_matching_confidence.png"
    )
    _save_scalar_png(validity_array, output_dir / "05_final_validity.png")
    Image.fromarray(
        flow_to_color(final_array, max_flow=common_max_flow), mode="RGB"
    ).save(output_dir / "06_final_flow.png")

    LOGGER.info("Saved six PNG files to %s", output_dir)
    LOGGER.info("Shared flow colour maximum: %.3f px", common_max_flow)
    LOGGER.info(
        "Global confidence range: [%.4f, %.4f] | final validity range: [%.4f, %.4f]",
        float(confidence_array.min()),
        float(confidence_array.max()),
        float(validity_array.min()),
        float(validity_array.max()),
    )


if __name__ == "__main__":
    main()
