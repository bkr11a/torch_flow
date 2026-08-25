#!/usr/bin/env python3
"""Prepare optical-flow benchmark test-set predictions for leaderboard submission.

Supported submission targets
----------------------------
- MPI Sintel: writes clean/final ``.flo`` trees and optionally invokes Bundler.
- KITTI 2012: writes benchmark-format 16-bit PNG flow and a ZIP archive.
- KITTI 2015: writes benchmark-format 16-bit PNG flow and a ZIP archive.
- Spring: writes the exact ``.flo5`` tree and optionally invokes flow_subsampling.

The script intentionally performs *native-resolution* inference. Inputs are padded
symmetrically to a configurable stride and full-resolution predictions are unpadded
before serialisation. It reuses the repository's model and dataset factories rather
than maintaining a second implementation of model construction.

A JSON manifest is written for every benchmark to record the checkpoint digest,
config, command-line invocation and generated files. This is intended to make a
leaderboard result traceable to the exact experimental artefact that produced it.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf

from data.hd1k import HD1KDataset
from data.kitti import KITTIDataset
from data.sintel import SintelDataset
from data.spring import SpringDataset
from models import build_model


SUPPORTED_BENCHMARKS = ("sintel", "kitti2012", "kitti2015", "spring", "hd1k")


def _sha256(path: Path, chunk_size: int = 4 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _normalise_mapping(value: Any) -> Dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, DictConfig):
        resolved = OmegaConf.to_container(value, resolve=True)
        if not isinstance(resolved, dict):
            raise TypeError("Expected YAML mapping")
        return resolved
    if isinstance(value, Mapping):
        return dict(value)
    raise TypeError(f"Expected mapping, got {type(value).__name__}")


def _symmetric_padding(height: int, width: int, stride: int) -> Tuple[int, int, int, int]:
    stride = max(int(stride), 1)
    pad_h = (stride - height % stride) % stride
    pad_w = (stride - width % stride) % stride
    top = pad_h // 2
    bottom = pad_h - top
    left = pad_w // 2
    right = pad_w - left
    return left, right, top, bottom


def _extract_final_flow(output: Any) -> torch.Tensor:
    """Return a full-resolution ``[B,2,H,W]`` tensor from repository model output."""
    if torch.is_tensor(output):
        flow = output
    elif isinstance(output, Mapping):
        preferred = (
            "flow_final_refined",
            "flow_final",
            "flow",
        )
        flow = None
        for key in preferred:
            value = output.get(key)
            if torch.is_tensor(value):
                flow = value
                break
        if flow is None:
            preds = output.get("flow_preds")
            if isinstance(preds, (list, tuple)) and preds and torch.is_tensor(preds[-1]):
                flow = preds[-1]
        if flow is None:
            raise KeyError(
                "Could not locate final flow. Expected flow_final_refined, flow_final, "
                "flow, or the final flow_preds entry."
            )
    elif isinstance(output, (list, tuple)) and output and torch.is_tensor(output[-1]):
        flow = output[-1]
    else:
        raise TypeError(f"Unsupported model output type: {type(output).__name__}")

    if flow.ndim != 4 or flow.shape[1] != 2:
        raise ValueError(f"Expected final flow [B,2,H,W], got {tuple(flow.shape)}")
    return flow


def _run_native_inference(
    model: torch.nn.Module,
    image1: torch.Tensor,
    image2: torch.Tensor,
    *,
    device: torch.device,
    stride: int,
    pad_mode: str,
    amp: bool,
) -> torch.Tensor:
    if image1.ndim == 3:
        image1 = image1.unsqueeze(0)
    if image2.ndim == 3:
        image2 = image2.unsqueeze(0)
    if image1.shape != image2.shape:
        raise ValueError(f"Input pair shape mismatch: {image1.shape} vs {image2.shape}")

    image1 = image1.to(device=device, dtype=torch.float32, non_blocking=True)
    image2 = image2.to(device=device, dtype=torch.float32, non_blocking=True)
    height, width = int(image1.shape[-2]), int(image1.shape[-1])
    padding = _symmetric_padding(height, width, stride)

    kwargs: Dict[str, Any] = {}
    if pad_mode == "constant":
        kwargs["value"] = 0.0
    if any(padding):
        image1_p = F.pad(image1, padding, mode=pad_mode, **kwargs)
        image2_p = F.pad(image2, padding, mode=pad_mode, **kwargs)
    else:
        image1_p, image2_p = image1, image2

    autocast_enabled = bool(amp and device.type == "cuda")
    with torch.inference_mode():
        with torch.autocast(device_type=device.type, enabled=autocast_enabled):
            output = model(image1_p, image2_p)
        flow = _extract_final_flow(output).float()

    if tuple(flow.shape[-2:]) != tuple(image1_p.shape[-2:]):
        old_h, old_w = flow.shape[-2:]
        flow = F.interpolate(flow, size=image1_p.shape[-2:], mode="bilinear", align_corners=False)
        flow[:, 0] *= image1_p.shape[-1] / float(old_w)
        flow[:, 1] *= image1_p.shape[-2] / float(old_h)

    if any(padding):
        left, _right, top, _bottom = padding
        flow = flow[..., top : top + height, left : left + width]

    if tuple(flow.shape[-2:]) != (height, width):
        raise RuntimeError(
            f"Native prediction grid was not restored: {tuple(flow.shape[-2:])} != {(height, width)}"
        )
    return flow.detach().cpu()


def write_flo(path: Path, flow_hw2: np.ndarray) -> None:
    """Write Middlebury/Sintel ``.flo`` binary format."""
    if flow_hw2.ndim != 3 or flow_hw2.shape[-1] != 2:
        raise ValueError(f"Expected HxWx2 flow, got {flow_hw2.shape}")
    path.parent.mkdir(parents=True, exist_ok=True)
    flow = np.asarray(flow_hw2, dtype="<f4")
    height, width = flow.shape[:2]
    with path.open("wb") as handle:
        np.asarray([202021.25], dtype="<f4").tofile(handle)
        np.asarray([width, height], dtype="<i4").tofile(handle)
        flow.tofile(handle)


def write_kitti_flow_png(path: Path, flow_hw2: np.ndarray) -> None:
    """Write KITTI optical-flow PNG using the official 1/64-pixel encoding."""
    try:
        import cv2
    except ImportError as exc:
        raise ImportError("KITTI submission writing requires OpenCV (cv2).") from exc

    if flow_hw2.ndim != 3 or flow_hw2.shape[-1] != 2:
        raise ValueError(f"Expected HxWx2 flow, got {flow_hw2.shape}")
    if not np.isfinite(flow_hw2).all():
        raise ValueError(f"Prediction contains non-finite values: {path.name}")

    u = np.clip(np.rint(flow_hw2[..., 0] * 64.0 + 2**15), 0, 65535).astype(np.uint16)
    v = np.clip(np.rint(flow_hw2[..., 1] * 64.0 + 2**15), 0, 65535).astype(np.uint16)
    valid = np.ones_like(u, dtype=np.uint16)
    encoded_bgr = np.stack((valid, v, u), axis=-1)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), encoded_bgr):
        raise IOError(f"Failed to write KITTI flow PNG: {path}")


def write_flo5(path: Path, flow_hw2: np.ndarray) -> None:
    """Write Spring ``.flo5`` HDF5 flow file using dataset key ``flow``."""
    try:
        import h5py
    except ImportError as exc:
        raise ImportError("Spring submission writing requires h5py.") from exc

    if flow_hw2.ndim != 3 or flow_hw2.shape[-1] != 2:
        raise ValueError(f"Expected HxWx2 flow, got {flow_hw2.shape}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as handle:
        handle.create_dataset("flow", data=np.asarray(flow_hw2, dtype=np.float32))


def _upsample_spring_grid_without_vector_scaling(flow_hw2: np.ndarray, factor: int) -> np.ndarray:
    """Move HD predictions to Spring's 4K file grid without scaling u/v vectors."""
    factor = int(factor)
    if factor == 1:
        return np.ascontiguousarray(flow_hw2.astype(np.float32))
    if factor <= 0:
        raise ValueError("spring_output_spatial_scale must be >= 1")
    tensor = torch.from_numpy(flow_hw2.transpose(2, 0, 1)).unsqueeze(0).float()
    up = F.interpolate(tensor, scale_factor=factor, mode="bilinear", align_corners=False)
    return np.ascontiguousarray(up[0].permute(1, 2, 0).numpy().astype(np.float32))


def _load_model(model_config: Path, checkpoint: Path, device: torch.device) -> torch.nn.Module:
    cfg = OmegaConf.load(model_config)
    model = build_model(cfg).to(device)
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    state = payload.get("model", payload) if isinstance(payload, Mapping) else payload
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"[checkpoint] missing keys: {len(missing)}", file=sys.stderr)
    if unexpected:
        print(f"[checkpoint] unexpected keys: {len(unexpected)}", file=sys.stderr)
    model.eval()
    return model


def _sample_to_numpy_flow(
    model: torch.nn.Module,
    sample: Mapping[str, torch.Tensor],
    *,
    device: torch.device,
    stride: int,
    pad_mode: str,
    amp: bool,
) -> np.ndarray:
    flow = _run_native_inference(
        model,
        sample["image1"],
        sample["image2"],
        device=device,
        stride=stride,
        pad_mode=pad_mode,
        amp=amp,
    )[0]
    array = flow.permute(1, 2, 0).numpy().astype(np.float32)
    if not np.isfinite(array).all():
        raise FloatingPointError("Model produced a non-finite test-set prediction.")
    return np.ascontiguousarray(array)


def _run_tool(command: Sequence[str], cwd: Path) -> None:
    print("[tool]", " ".join(str(x) for x in command))
    subprocess.run([str(x) for x in command], cwd=str(cwd), check=True)


def _zip_files(files: Sequence[Path], base_dir: Path, zip_path: Path) -> Path:
    import zipfile

    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            archive.write(path, arcname=str(path.relative_to(base_dir)))
    return zip_path


def prepare_sintel(
    *,
    model: torch.nn.Module,
    cfg: Mapping[str, Any],
    output_root: Path,
    device: torch.device,
    stride: int,
    pad_mode: str,
    amp: bool,
) -> Dict[str, Any]:
    root = Path(str(cfg["root"]))
    benchmark_root = output_root / "sintel"
    generated: List[Path] = []
    counts: Dict[str, int] = {}

    for dstype in ("clean", "final"):
        dataset = SintelDataset(root=str(root), split="test", dstype=dstype)
        pass_root = benchmark_root / dstype
        count = 0
        for index in range(len(dataset)):
            sample = dataset[index]
            meta = dataset._samples[index]
            image1 = Path(str(meta["image1"]))
            scene = image1.parent.name
            output_name = image1.with_suffix(".flo").name
            flow = _sample_to_numpy_flow(
                model, sample, device=device, stride=stride, pad_mode=pad_mode, amp=amp
            )
            destination = pass_root / scene / output_name
            write_flo(destination, flow)
            generated.append(destination)
            count += 1
        counts[dstype] = count

    bundled_file: Optional[Path] = None
    bundler = cfg.get("bundler", None)
    if bundler:
        bundler_path = Path(str(bundler)).expanduser()
        if bundler_path.is_file():
            bundled_file = benchmark_root / str(cfg.get("bundle_name", "hqsflow_sintel.lzma"))
            # MPI Sintel Bundler convention: clean-dir final-dir output.lzma
            _run_tool(
                [str(bundler_path), str(benchmark_root / "clean"), str(benchmark_root / "final"), str(bundled_file)],
                cwd=benchmark_root,
            )
        else:
            print(f"[sintel] Bundler not found at {bundler_path}; raw .flo trees are complete.")

    return {
        "benchmark": "sintel",
        "counts": counts,
        "raw_root": str(benchmark_root),
        "bundle": str(bundled_file) if bundled_file is not None else None,
        "files": [str(p) for p in generated],
    }


def prepare_kitti(
    *,
    year: int,
    model: torch.nn.Module,
    cfg: Mapping[str, Any],
    output_root: Path,
    device: torch.device,
    stride: int,
    pad_mode: str,
    amp: bool,
) -> Dict[str, Any]:
    root = Path(str(cfg["root"]))
    key = f"kitti{year}"
    benchmark_root = output_root / key
    benchmark_root.mkdir(parents=True, exist_ok=True)

    dataset = KITTIDataset(
        root=str(root),
        split="test",
        val_size=0,
        camera=cfg.get("camera", None),
        flow_type=str(cfg.get("flow_type", "occ")),
    )
    generated: List[Path] = []
    for index in range(len(dataset)):
        sample = dataset[index]
        meta = dataset._samples[index]
        image1 = Path(str(meta["image1"]))
        flow = _sample_to_numpy_flow(
            model, sample, device=device, stride=stride, pad_mode=pad_mode, amp=amp
        )
        destination = benchmark_root / image1.name
        write_kitti_flow_png(destination, flow)
        generated.append(destination)

    zip_path = output_root / f"{key}_submission.zip"
    _zip_files(generated, benchmark_root, zip_path)
    return {
        "benchmark": key,
        "count": len(generated),
        "raw_root": str(benchmark_root),
        "archive": str(zip_path),
        "files": [str(p) for p in generated],
    }



def prepare_hd1k(
    *,
    model: torch.nn.Module,
    cfg: Mapping[str, Any],
    output_root: Path,
    device: torch.device,
    stride: int,
    pad_mode: str,
    amp: bool,
) -> Dict[str, Any]:
    """Prepare the HD1K challenge split as a legacy KITTI-style PNG archive.

    The current public HCI page still documents HD1K as a benchmark but does not
    expose a clearly verifiable modern standalone upload workflow. This export is
    therefore marked ``legacy_challenge_export`` in the manifest rather than
    claiming that it is directly upload-ready.
    """
    root = Path(str(cfg["root"]))
    benchmark_root = output_root / "hd1k"
    benchmark_root.mkdir(parents=True, exist_ok=True)
    dataset = HD1KDataset(
        root=str(root),
        split="test",
        camera=str(cfg.get("camera", "image_2")),
    )
    generated: List[Path] = []
    for index in range(len(dataset)):
        sample = dataset[index]
        meta = dataset._samples[index]
        image1 = Path(str(meta["image1"]))
        flow = _sample_to_numpy_flow(
            model, sample, device=device, stride=stride, pad_mode=pad_mode, amp=amp
        )
        destination = benchmark_root / image1.with_suffix(".png").name
        write_kitti_flow_png(destination, flow)
        generated.append(destination)
    zip_path = output_root / "hd1k_legacy_challenge_export.zip"
    _zip_files(generated, benchmark_root, zip_path)
    return {
        "benchmark": "hd1k",
        "status": "legacy_challenge_export",
        "count": len(generated),
        "raw_root": str(benchmark_root),
        "archive": str(zip_path),
        "files": [str(p) for p in generated],
    }

def _spring_direction_and_side(meta: Mapping[str, Any]) -> Tuple[str, str, str]:
    image1 = Path(str(meta["image1"]))
    image2 = Path(str(meta["image2"]))
    scene = image1.parent.parent.name
    side_match = re.fullmatch(r"frame_(left|right)", image1.parent.name)
    if side_match is None:
        raise ValueError(f"Cannot infer Spring side from {image1}")
    side = side_match.group(1)

    def frame_number(path: Path) -> int:
        match = re.search(r"(\d+)$", path.stem)
        if match is None:
            raise ValueError(f"Cannot parse Spring frame index from {path.name}")
        return int(match.group(1))

    idx1, idx2 = frame_number(image1), frame_number(image2)
    direction = "FW" if idx2 > idx1 else "BW"
    index_string = re.search(r"(\d+)$", image1.stem).group(1)  # type: ignore[union-attr]
    return scene, direction, f"{side}:{index_string}"


def prepare_spring(
    *,
    model: torch.nn.Module,
    cfg: Mapping[str, Any],
    output_root: Path,
    device: torch.device,
    stride: int,
    pad_mode: str,
    amp: bool,
) -> Dict[str, Any]:
    root = Path(str(cfg["root"]))
    benchmark_root = output_root / "spring"
    spatial_scale = int(cfg.get("output_spatial_scale", 2))

    dataset = SpringDataset(
        root=str(root),
        split="test",
        direction="both",
        side="both",
    )
    generated: List[Path] = []
    per_route: Dict[str, int] = {}
    for index in range(len(dataset)):
        sample = dataset[index]
        meta = dataset._samples[index]
        scene, direction, side_and_index = _spring_direction_and_side(meta)
        side, index_string = side_and_index.split(":", 1)
        route = f"flow_{direction}_{side}"
        flow = _sample_to_numpy_flow(
            model, sample, device=device, stride=stride, pad_mode=pad_mode, amp=amp
        )
        flow = _upsample_spring_grid_without_vector_scaling(flow, spatial_scale)
        destination = benchmark_root / scene / route / f"{route}_{index_string}.flo5"
        write_flo5(destination, flow)
        generated.append(destination)
        per_route[route] = per_route.get(route, 0) + 1

    subsampling_output: Optional[Path] = None
    subsampler = cfg.get("flow_subsampling", None)
    if subsampler:
        exe = Path(str(subsampler)).expanduser()
        if exe.is_file():
            before = set(benchmark_root.glob("*.hdf5"))
            _run_tool([str(exe.resolve()), "."], cwd=benchmark_root)
            after = set(benchmark_root.glob("*.hdf5"))
            new_files = sorted(after - before)
            if new_files:
                subsampling_output = new_files[-1]
        else:
            print(f"[spring] flow_subsampling not found at {exe}; .flo5 tree is complete.")

    return {
        "benchmark": "spring",
        "count": len(generated),
        "per_route": per_route,
        "output_spatial_scale": spatial_scale,
        "raw_root": str(benchmark_root),
        "subsampling_output": str(subsampling_output) if subsampling_output else None,
        "files": [str(p) for p in generated],
    }


def _benchmark_model_settings(
    global_cfg: Mapping[str, Any],
    benchmark_cfg: Mapping[str, Any],
    default_model_config: Path,
    default_checkpoint: Path,
) -> Tuple[Path, Path]:
    model_config = Path(str(benchmark_cfg.get("model_config", default_model_config)))
    checkpoint = Path(str(benchmark_cfg.get("checkpoint", default_checkpoint)))
    return model_config, checkpoint


def _write_manifest(
    benchmark_root: Path,
    *,
    benchmark: str,
    model_config: Path,
    checkpoint: Path,
    result: Mapping[str, Any],
) -> Path:
    manifest = {
        "benchmark": benchmark,
        "created_unix": time.time(),
        "argv": sys.argv,
        "model_config": str(model_config.resolve()),
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": _sha256(checkpoint),
        "result": dict(result),
    }
    benchmark_root.mkdir(parents=True, exist_ok=True)
    path = benchmark_root / "submission_manifest.json"
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-config", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--submission-config", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--benchmarks",
        nargs="+",
        default=list(SUPPORTED_BENCHMARKS),
        choices=SUPPORTED_BENCHMARKS,
    )
    parser.add_argument("--device", default=None, help="cuda, cuda:0, mps, or cpu")
    parser.add_argument("--stride", type=int, default=8)
    parser.add_argument(
        "--pad-mode",
        default="replicate",
        choices=("replicate", "reflect", "constant", "circular"),
    )
    parser.add_argument("--amp", action="store_true", help="Use CUDA autocast during submission inference")
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    if args.stride <= 0:
        raise ValueError("--stride must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    document = OmegaConf.load(args.submission_config)
    submission_cfg = _normalise_mapping(document.get("benchmarks", document))
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    model_cache: Dict[Tuple[str, str], torch.nn.Module] = {}
    results: Dict[str, Any] = {}

    for benchmark in args.benchmarks:
        if benchmark not in submission_cfg:
            raise KeyError(f"Missing benchmark section '{benchmark}' in {args.submission_config}")
        bench_cfg = _normalise_mapping(submission_cfg[benchmark])
        if not bool(bench_cfg.get("enabled", True)):
            print(f"[{benchmark}] disabled; skipping")
            continue

        model_config, checkpoint = _benchmark_model_settings(
            submission_cfg, bench_cfg, args.model_config, args.checkpoint
        )
        cache_key = (str(model_config.resolve()), str(checkpoint.resolve()))
        if cache_key not in model_cache:
            print(f"[{benchmark}] loading model: {model_config}")
            print(f"[{benchmark}] checkpoint: {checkpoint}")
            model_cache[cache_key] = _load_model(model_config, checkpoint, device)
        model = model_cache[cache_key]

        print(f"[{benchmark}] preparing test-set predictions")
        if benchmark == "sintel":
            result = prepare_sintel(
                model=model,
                cfg=bench_cfg,
                output_root=args.output_dir,
                device=device,
                stride=args.stride,
                pad_mode=args.pad_mode,
                amp=args.amp,
            )
        elif benchmark == "kitti2012":
            result = prepare_kitti(
                year=2012,
                model=model,
                cfg=bench_cfg,
                output_root=args.output_dir,
                device=device,
                stride=args.stride,
                pad_mode=args.pad_mode,
                amp=args.amp,
            )
        elif benchmark == "kitti2015":
            result = prepare_kitti(
                year=2015,
                model=model,
                cfg=bench_cfg,
                output_root=args.output_dir,
                device=device,
                stride=args.stride,
                pad_mode=args.pad_mode,
                amp=args.amp,
            )
        elif benchmark == "spring":
            result = prepare_spring(
                model=model,
                cfg=bench_cfg,
                output_root=args.output_dir,
                device=device,
                stride=args.stride,
                pad_mode=args.pad_mode,
                amp=args.amp,
            )
        elif benchmark == "hd1k":
            result = prepare_hd1k(
                model=model,
                cfg=bench_cfg,
                output_root=args.output_dir,
                device=device,
                stride=args.stride,
                pad_mode=args.pad_mode,
                amp=args.amp,
            )
        else:
            raise AssertionError(benchmark)

        root = Path(str(result["raw_root"]))
        manifest_path = _write_manifest(
            root,
            benchmark=benchmark,
            model_config=model_config,
            checkpoint=checkpoint,
            result=result,
        )
        result = dict(result)
        result["manifest"] = str(manifest_path)
        results[benchmark] = result
        print(f"[{benchmark}] complete: {manifest_path}")

    summary = args.output_dir / "submission_summary.json"
    summary.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Submission preparation complete: {summary}")


if __name__ == "__main__":
    main()
