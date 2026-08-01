#!/usr/bin/env python3
"""Evaluate one configured optical-flow dataset without binary flow exports.

This entry point intentionally reuses the comprehensive evaluator so metric,
diagnostic, plotting, qualitative, video, model-summary, and report behaviour
remain aligned with :mod:`evaluate_comprehensive`.  It adds support for the
standalone data-configuration layout used by the native-resolution evaluation
configs::

    data:
      name: sintel
      root: /path/to/MPI-Sintel-complete
      batch_size: 1

    evaluation:
      native_resolution: true
      crop: false
      resize: false
      pad_to_stride: 8
      pad_mode: replicate
      write_scene_video: true
      scene_key: scene

Unlike ``evaluate_comprehensive.py``, this script never serialises predicted or
ground-truth flow arrays as binary flow files.  All other comprehensive
evaluation products are retained.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import math
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf

import evaluate_comprehensive as comprehensive
from data import build_dataloader, build_dataset
from models import build_model


logger = logging.getLogger(__name__)

# Keep stable references before installing the selected-dataset behaviour into
# the comprehensive evaluator module.
_COMPREHENSIVE_PROCESS_SAMPLE_TASK = comprehensive._process_sample_task
_COMPREHENSIVE_INFER_SCENE_AND_SAMPLE = comprehensive._infer_scene_and_sample

# These settings are read only in the main process while tasks are assembled.
_SCENE_KEY = "scene"
_WRITE_SCENE_VIDEO = True


def _discard_binary_flow(*_args: Any, **_kwargs: Any) -> None:
    """Deliberately discard a binary-flow export request."""


def _process_sample_task_without_binary_flow(task: Dict[str, Any]) -> Dict[str, Any]:
    """Run comprehensive post-processing with binary flow writing disabled.

    This function is top-level so it remains picklable when
    ``--postproc_workers`` selects the spawn-based process pool.
    """
    comprehensive.write_flow = _discard_binary_flow
    return _COMPREHENSIVE_PROCESS_SAMPLE_TASK(task)


def _normalise_scene_component(value: Any) -> str:
    text = str(value or "").strip().replace("\\", "/")
    text = text.rsplit("/", 1)[-1]
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("._")


def _infer_scene_and_sample(
    record: Optional[Dict[str, Any]],
    fallback_name: str,
) -> Dict[str, Any]:
    """Resolve configured scene metadata and HD1K filename sequences.

    Sintel and Spring are already resolved correctly by the comprehensive
    evaluator from their directory layout.  HD1K encodes the sequence in the
    leading filename field, so it requires an explicit correction to prevent
    all validation frames being merged into one video.
    """
    result = _COMPREHENSIVE_INFER_SCENE_AND_SAMPLE(record, fallback_name)

    configured_scene = None
    if record is not None and _SCENE_KEY:
        configured_scene = record.get(_SCENE_KEY)
    if configured_scene not in (None, ""):
        scene = _normalise_scene_component(configured_scene)
        if scene:
            result["scene"] = scene

    if record is not None and record.get("image1"):
        image1 = Path(str(record["image1"]))
        if image1.parent.name in {"image_2", "image_3"}:
            match = re.match(r"^(\d{6})[_-]\d{6}$", image1.stem)
            if match:
                result["scene"] = f"hd1k_{match.group(1)}"

    result["has_scene"] = bool(result.get("scene")) and _WRITE_SCENE_VIDEO
    return result


def _to_plain_mapping(config: Any) -> Dict[str, Any]:
    if config is None:
        return {}
    if isinstance(config, DictConfig):
        value = OmegaConf.to_container(config, resolve=True)
    elif isinstance(config, Mapping):
        value = dict(config)
    else:
        raise TypeError(
            "The evaluation section must be a YAML mapping, got "
            f"{type(config).__name__}."
        )
    if not isinstance(value, dict):
        raise TypeError("The evaluation section must resolve to a YAML mapping.")
    return value


def _resolve_data_and_evaluation_config(
    data_document: DictConfig,
    cli_batch_size: Optional[int],
) -> Tuple[DictConfig, Dict[str, Any]]:
    """Resolve old and new data-config layouts without ignoring policy keys."""
    if data_document.get("val_data") is not None:
        data_section = data_document.get("val_data")
    elif data_document.get("data") is not None:
        data_section = data_document.get("data")
    else:
        data_section = data_document

    eval_cfg = OmegaConf.create(
        OmegaConf.to_container(data_section, resolve=True)
    )
    evaluation = _to_plain_mapping(data_document.get("evaluation"))

    native_resolution = bool(evaluation.get("native_resolution", False))
    crop_requested = bool(evaluation.get("crop", False))
    resize_requested = bool(evaluation.get("resize", False))

    if native_resolution and (crop_requested or resize_requested):
        raise ValueError(
            "native_resolution=true is incompatible with crop=true or "
            "resize=true. Native evaluation must preserve the source pixel grid."
        )
    if crop_requested or resize_requested:
        raise ValueError(
            "evaluate_selected_dataset.py supports uncropped, unresized "
            "benchmark evaluation only. Set evaluation.crop=false and "
            "evaluation.resize=false."
        )

    if native_resolution:
        # The validation factory already disables augmentation. Removing these
        # fields also makes the native-resolution contract explicit and guards
        # against future loader changes.
        for key in ("crop_size", "image_size", "resize", "height", "width"):
            if key in eval_cfg:
                del eval_cfg[key]

    if cli_batch_size is not None:
        if cli_batch_size <= 0:
            raise ValueError("--batch_size must be greater than zero.")
        eval_cfg.batch_size = int(cli_batch_size)
    elif eval_cfg.get("batch_size") is None:
        # Backwards-compatible fallback for older data-only YAML files.
        eval_cfg.batch_size = 4

    if eval_cfg.get("num_workers") is None:
        eval_cfg.num_workers = 4

    stride_default = 8 if native_resolution else 1
    stride = int(evaluation.get("pad_to_stride", stride_default) or 1)
    if stride <= 0:
        raise ValueError("evaluation.pad_to_stride must be a positive integer.")

    pad_mode = str(evaluation.get("pad_mode", "replicate")).lower()
    if pad_mode not in {"replicate", "reflect", "circular", "constant"}:
        raise ValueError(
            "evaluation.pad_mode must be one of: replicate, reflect, "
            "circular, constant."
        )

    policy = {
        "native_resolution": native_resolution,
        "crop": False,
        "resize": False,
        "pad_to_stride": stride,
        "pad_mode": pad_mode,
        "write_scene_video": bool(evaluation.get("write_scene_video", True)),
        "scene_key": str(evaluation.get("scene_key", "scene") or "scene"),
    }
    return eval_cfg, policy


def _symmetric_padding(height: int, width: int, stride: int) -> Tuple[int, int, int, int]:
    pad_h = (stride - height % stride) % stride
    pad_w = (stride - width % stride) % stride
    top = pad_h // 2
    bottom = pad_h - top
    left = pad_w // 2
    right = pad_w - left
    return left, right, top, bottom


def _unpad_full_resolution_tensors(
    value: Any,
    padded_hw: Tuple[int, int],
    original_hw: Tuple[int, int],
    padding: Tuple[int, int, int, int],
) -> Any:
    """Recursively unpad only tensors that live on the padded image grid."""
    if torch.is_tensor(value):
        if value.ndim >= 2 and tuple(value.shape[-2:]) == padded_hw:
            left, _right, top, _bottom = padding
            height, width = original_hw
            return value[..., top : top + height, left : left + width]
        return value
    if isinstance(value, list):
        return [
            _unpad_full_resolution_tensors(v, padded_hw, original_hw, padding)
            for v in value
        ]
    if isinstance(value, tuple):
        return tuple(
            _unpad_full_resolution_tensors(v, padded_hw, original_hw, padding)
            for v in value
        )
    if isinstance(value, dict):
        return {
            k: _unpad_full_resolution_tensors(v, padded_hw, original_hw, padding)
            for k, v in value.items()
        }
    return value


class StridePaddedModel(nn.Module):
    """Pad native inputs for inference and restore full-grid outputs."""

    def __init__(self, model: nn.Module, stride: int, pad_mode: str) -> None:
        super().__init__()
        self.model = model
        self.stride = max(1, int(stride))
        self.pad_mode = str(pad_mode)

    def forward(self, image1: torch.Tensor, image2: torch.Tensor, *args: Any, **kwargs: Any) -> Any:
        if image1.shape != image2.shape:
            raise ValueError(
                "Evaluation image pair has mismatched shapes: "
                f"{tuple(image1.shape)} and {tuple(image2.shape)}."
            )

        height, width = int(image1.shape[-2]), int(image1.shape[-1])
        padding = _symmetric_padding(height, width, self.stride)
        if not any(padding):
            return self.model(image1, image2, *args, **kwargs)

        pad_kwargs: Dict[str, Any] = {}
        if self.pad_mode == "constant":
            pad_kwargs["value"] = 0.0
        padded1 = F.pad(image1, padding, mode=self.pad_mode, **pad_kwargs)
        padded2 = F.pad(image2, padding, mode=self.pad_mode, **pad_kwargs)
        output = self.model(padded1, padded2, *args, **kwargs)

        padded_hw = (int(padded1.shape[-2]), int(padded1.shape[-1]))
        return _unpad_full_resolution_tensors(
            output,
            padded_hw=padded_hw,
            original_hw=(height, width),
            padding=padding,
        )


class SelectedDatasetEvaluator(comprehensive.FlowEvaluator):
    """Comprehensive evaluator with configuration-controlled scene videos."""

    def __init__(self, *args: Any, write_scene_video: bool = True, **kwargs: Any) -> None:
        self.write_scene_video = bool(write_scene_video)
        super().__init__(*args, **kwargs)

    def _build_scene_videos(self) -> None:
        if not self.write_scene_video:
            logger.info("Scene-video generation disabled by the data configuration.")
            return
        super()._build_scene_videos()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate one selected optical-flow dataset comprehensively without "
            "binary flow exports"
        )
    )
    parser.add_argument("--config", "-c", required=True, help="Path to model config YAML")
    parser.add_argument("--checkpoint", "-ckpt", required=True, help="Path to model checkpoint")
    parser.add_argument("--data_config", "-dc", required=True, help="Path to data config YAML")
    parser.add_argument("--output_dir", "-o", required=True, help="Output directory for results")
    parser.add_argument("--device", default=None, help="Device (cuda/mps/cpu)")
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Override the data-config evaluation batch size",
    )
    parser.add_argument(
        "--postproc_workers",
        type=int,
        default=None,
        help="Number of CPU workers for per-sample post-processing; default=max(1,cpu_count-1)",
    )
    parser.add_argument("--report_template", default=None, help="Path to markdown template reference")
    parser.add_argument("--experiment_id", default="HQS-EXP-AUTO")
    parser.add_argument("--experiment_title", default="Auto comprehensive evaluation")
    parser.add_argument(
        "--status",
        default="completed",
        choices=["planned", "running", "completed", "failed", "superseded"],
    )
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
    return parser


def main() -> None:
    global _SCENE_KEY
    global _WRITE_SCENE_VIDEO

    args = _build_parser().parse_args()
    comprehensive.setup_logging(args.output_dir)
    logger.info("=" * 80)
    logger.info("HQS Optical Flow - Selected Dataset Comprehensive Evaluation")
    logger.info("Binary flow export: disabled")
    logger.info("=" * 80)

    start_time = time.time()
    device = torch.device(args.device) if args.device else comprehensive.get_device()
    logger.info("Device: %s", device)

    model_cfg = OmegaConf.load(args.config)
    data_document = OmegaConf.load(args.data_config)
    eval_cfg, evaluation_policy = _resolve_data_and_evaluation_config(
        data_document,
        cli_batch_size=args.batch_size,
    )

    _SCENE_KEY = evaluation_policy["scene_key"]
    _WRITE_SCENE_VIDEO = evaluation_policy["write_scene_video"]
    comprehensive._infer_scene_and_sample = _infer_scene_and_sample
    comprehensive._process_sample_task = _process_sample_task_without_binary_flow

    model = build_model(model_cfg).to(device)
    model_total_params = int(sum(parameter.numel() for parameter in model.parameters()))
    model_trainable_params = int(
        sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    )
    logger.info(
        "Model parameters: total=%s trainable=%s non_trainable=%s",
        f"{model_total_params:,}",
        f"{model_trainable_params:,}",
        f"{model_total_params - model_trainable_params:,}",
    )

    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint.get("model", checkpoint), strict=False)
    logger.info("Loaded checkpoint from %s", args.checkpoint)

    eval_data = build_dataset(eval_cfg, split="val")
    eval_loader = build_dataloader(eval_data, eval_cfg, split="val")
    logger.info("Evaluation set: %s samples", len(eval_data))
    logger.info(
        "Evaluation policy: native=%s crop=%s resize=%s stride=%s pad_mode=%s "
        "scene_video=%s",
        evaluation_policy["native_resolution"],
        evaluation_policy["crop"],
        evaluation_policy["resize"],
        evaluation_policy["pad_to_stride"],
        evaluation_policy["pad_mode"],
        evaluation_policy["write_scene_video"],
    )

    model_summary_info: Dict[str, Any] = {}
    if not args.no_model_summary:
        model_summary_info = comprehensive.save_model_summary(
            model,
            device,
            Path(args.output_dir),
            input_shape=(1, 3, int(args.summary_height), int(args.summary_width)),
        )

    inference_model: nn.Module = StridePaddedModel(
        model,
        stride=evaluation_policy["pad_to_stride"],
        pad_mode=evaluation_policy["pad_mode"],
    )

    default_workers = max(1, (os.cpu_count() or 1) - 1)
    postproc_workers = (
        default_workers
        if args.postproc_workers is None
        else max(1, int(args.postproc_workers))
    )
    logger.info("Post-processing workers: %s", postproc_workers)

    system_metadata = comprehensive._system_metadata(device)
    git_metadata = comprehensive._git_metadata(str(Path(__file__).resolve().parent))
    report_context = {
        "config_path": os.path.abspath(args.config),
        "data_config_path": os.path.abspath(args.data_config),
        "checkpoint": os.path.abspath(args.checkpoint),
        "system": system_metadata,
        "git": git_metadata,
        "dataset_size": len(eval_data),
        "evaluation_policy": evaluation_policy,
        "binary_flow_export": False,
        "model_summary": {
            "total_params": model_summary_info.get("total_params", model_total_params),
            "trainable_params": model_summary_info.get(
                "trainable_params", model_trainable_params
            ),
            "non_trainable_params": model_summary_info.get(
                "non_trainable_params", model_total_params - model_trainable_params
            ),
            "summary_markdown": model_summary_info.get("markdown", ""),
            "summary_json": model_summary_info.get("json", ""),
        },
    }

    evaluator = SelectedDatasetEvaluator(
        inference_model,
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
        write_scene_video=evaluation_policy["write_scene_video"],
    )

    metrics = evaluator.evaluate(eval_loader)
    runtime = time.time() - start_time

    run_metadata = {
        "datetime": dt.datetime.now().isoformat(),
        "runtime_sec": float(runtime),
        "args": vars(args),
        "system": system_metadata,
        "git": git_metadata,
        "config": os.path.abspath(args.config),
        "data_config": os.path.abspath(args.data_config),
        "checkpoint": os.path.abspath(args.checkpoint),
        "resolved_data_config": OmegaConf.to_container(eval_cfg, resolve=True),
        "evaluation_policy": evaluation_policy,
        "binary_flow_export": False,
        "model_summary": model_summary_info,
        "metrics": metrics,
    }
    run_metadata_path = Path(args.output_dir) / "run_metadata.json"
    with open(run_metadata_path, "w") as file:
        json.dump(run_metadata, file, indent=2)

    report_data_path = Path(args.output_dir) / "report_data.json"
    if report_data_path.exists():
        with open(report_data_path) as file:
            report_data = json.load(file)
        report_data["runtime_sec"] = float(runtime)
        report_data["evaluation_policy"] = evaluation_policy
        report_data["binary_flow_export"] = False
        with open(report_data_path, "w") as file:
            json.dump(report_data, file, indent=2)

    logger.info("=" * 80)
    logger.info("Final Metrics Summary")
    logger.info("=" * 80)
    for key, value in sorted(metrics.items()):
        if not math.isnan(value):
            logger.info("  %-22s: %10.4f", key, value)
    logger.info("  %-22s: %10.2f", "runtime_sec", runtime)
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
