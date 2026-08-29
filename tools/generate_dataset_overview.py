#!/usr/bin/env python3
r"""Generate a publication-grade optical-flow dataset overview figure.

The figure is a two-row, four-column montage.  Every dataset card contains
the source frame I_1, target frame I_2, and a larger ground-truth flow panel.
Sintel Clean and Final are deliberately shown using the same scene/frame so
that their rendering-domain difference is directly comparable.

Default panel order
-------------------
  FlyingChairs | FlyingThings3D | Sintel Clean | Sintel Final
  Spring       | KITTI 2012     | KITTI 2015   | HD1K

The defaults match the dataset roots used by the HQSFlow curriculum.  Roots,
sample indices, output formats, and visual settings can all be overridden from
the command line.

Examples
--------
Generate PDF and 600-dpi PNG with automatic deterministic sample selection::

    python generate_dataset_overview.py

Choose particular entries after inspecting the indices printed by a first run::

    python generate_dataset_overview.py \
        --sample chairs=420 \
        --sample things=1750 \
        --sample sintel_final=214 \
        --sample spring=605 \
        --sample kitti2012=73 \
        --sample kitti2015=118 \
        --sample hd1k=381

Write PDF, PNG, and SVG to a custom base path::

    python generate_dataset_overview.py \
        --output figures/hqsflow_dataset_overview \
        --formats pdf,png,svg

Dependencies
------------
Required: numpy, matplotlib, opencv-python (or opencv-python-headless).
Spring .flo5 files additionally require h5py.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import re
import struct
import sys
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import cv2
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpecFromSubplotSpec
import numpy as np


# ---------------------------------------------------------------------------
# Publication styling
# ---------------------------------------------------------------------------

INPUT_ORANGE = "#F39C34"
FLOW_GREEN = "#33C653"
MID_GREY = "#747474"
LIGHT_GREY = np.array([218, 218, 218], dtype=np.uint8)
TEXT_GREY = "#555555"

PANEL_KEYS = (
    "chairs",
    "things",
    "sintel_clean",
    "sintel_final",
    "spring",
    "kitti2012",
    "kitti2015",
    "hd1k",
)

PANEL_TITLES = {
    "chairs": "FlyingChairs",
    "things": "FlyingThings3D",
    "sintel_clean": "MPI-Sintel Clean",
    "sintel_final": "MPI-Sintel Final",
    "spring": "Spring",
    "kitti2012": "KITTI 2012",
    "kitti2015": "KITTI 2015",
    "hd1k": "HD1K",
}

PANEL_LETTERS = dict(zip(PANEL_KEYS, "abcdefgh"))

DEFAULT_ROOTS = {
    "chairs": Path("/mnt/a/benchmark_data/optical_flow/FlyingChairs_release"),
    "things": Path(
        "/mnt/a/benchmark_data/optical_flow/FlyingThings3D/"
        "FlyingThings3D_subset"
    ),
    "sintel": Path(
        "/mnt/a/benchmark_data/optical_flow/MPI-Sintel/"
        "MPI-Sintel-complete"
    ),
    "spring": Path("/mnt/a/benchmark_data/optical_flow/Spring/spring"),
    "kitti2012": Path(
        "/mnt/a/benchmark_data/optical_flow/KITTI/KITTI 2012/"
        "data_stereo_flow"
    ),
    "kitti2015": Path(
        "/mnt/a/benchmark_data/optical_flow/KITTI/KITTI 2015/"
        "data_scene_flow"
    ),
    "hd1k": Path(
        "/mnt/a/benchmark_data/optical_flow/HD1K/hd1k_full_package"
    ),
}


@dataclass(frozen=True)
class FlowSample:
    """Paths and metadata for one supervised optical-flow pair."""

    dataset: str
    sample_id: str
    image1: Path
    image2: Path
    flow: Path
    flow_encoding: str = "dense"


@dataclass
class LoadedSample:
    """Decoded data used by both selection and rendering."""

    sample: FlowSample
    image1: np.ndarray
    image2: np.ndarray
    flow: np.ndarray
    valid: np.ndarray
    p95_magnitude: float
    valid_fraction: float


def natural_key(value: object) -> List[object]:
    """Natural-sort paths containing frame and sequence numbers."""

    text = str(value)
    return [int(part) if part.isdigit() else part.lower()
            for part in re.split(r"(\d+)", text)]


def sorted_paths(paths: Iterable[Path]) -> List[Path]:
    return sorted(paths, key=natural_key)


def require_directory(path: Path, label: str) -> Path:
    path = path.expanduser().resolve()
    if not path.is_dir():
        raise FileNotFoundError(f"{label} directory not found: {path}")
    return path


def find_image_by_stem(directory: Path, stem: str) -> Optional[Path]:
    for suffix in (".png", ".jpg", ".jpeg", ".ppm", ".webp"):
        candidate = directory / f"{stem}{suffix}"
        if candidate.is_file():
            return candidate
    return None


# ---------------------------------------------------------------------------
# Dataset discovery
# ---------------------------------------------------------------------------

def discover_chairs(root: Path) -> List[FlowSample]:
    root = require_directory(root, "FlyingChairs root")
    data_dir = root / "data" if (root / "data").is_dir() else root
    flows = sorted_paths(data_dir.glob("*_flow.flo"))
    samples: List[FlowSample] = []

    for flow in flows:
        base = flow.name.removesuffix("_flow.flo")
        image1 = find_image_by_stem(data_dir, f"{base}_img1")
        image2 = find_image_by_stem(data_dir, f"{base}_img2")
        if image1 is None or image2 is None:
            continue
        samples.append(FlowSample(
            dataset="chairs",
            sample_id=base,
            image1=image1,
            image2=image2,
            flow=flow,
        ))

    return samples


def _match_things_subset_flow(
    image: Path,
    flow_dir: Path,
    flow_files: Sequence[Path],
    pair_index: int,
    direction: str,
    side: str,
) -> Optional[Path]:
    stem = image.stem
    direction_word = "IntoFuture" if direction == "into_future" else "IntoPast"
    side_letter = side[0].upper()
    candidates: List[Path] = []

    for suffix in (".flo", ".pfm"):
        candidates.append(flow_dir / f"{stem}{suffix}")
        if stem.isdigit():
            candidates.extend([
                flow_dir / f"{int(stem):06d}{suffix}",
                flow_dir / f"{int(stem):07d}{suffix}",
            ])
        candidates.append(
            flow_dir / f"OpticalFlow{direction_word}_{stem}_{side_letter}{suffix}"
        )

    for candidate in candidates:
        if candidate.is_file():
            return candidate

    # Mirrors the repository loader's last-resort pairing convention.
    if pair_index < len(flow_files):
        return flow_files[pair_index]
    return None


def _discover_things_subset(root: Path) -> List[FlowSample]:
    split_root = root / "train"
    image_root = split_root / "image_clean"
    flow_root = split_root / "flow"
    if not image_root.is_dir() or not flow_root.is_dir():
        return []

    samples: List[FlowSample] = []
    for side in ("left", "right"):
        image_dir = image_root / side
        flow_dir = flow_root / side / "into_future"
        if not image_dir.is_dir() or not flow_dir.is_dir():
            continue

        frames = sorted_paths(
            path for path in image_dir.iterdir()
            if path.is_file()
            and path.suffix.lower() in {".png", ".jpg", ".jpeg", ".ppm"}
        )
        flow_files = sorted_paths(
            path for path in flow_dir.iterdir()
            if path.is_file() and path.suffix.lower() in {".flo", ".pfm"}
        )

        for index in range(max(0, len(frames) - 1)):
            flow = _match_things_subset_flow(
                frames[index], flow_dir, flow_files, index, "into_future", side
            )
            if flow is None:
                continue
            samples.append(FlowSample(
                dataset="things",
                sample_id=f"{side}/{frames[index].stem}",
                image1=frames[index],
                image2=frames[index + 1],
                flow=flow,
            ))

    return samples


def _discover_things_full(root: Path) -> List[FlowSample]:
    image_base = root / "frames_cleanpass" / "TRAIN"
    flow_base = root / "optical_flow" / "TRAIN"
    if not image_base.is_dir() or not flow_base.is_dir():
        return []

    samples: List[FlowSample] = []
    for subset_dir in sorted_paths(p for p in image_base.iterdir() if p.is_dir()):
        for sequence_dir in sorted_paths(p for p in subset_dir.iterdir() if p.is_dir()):
            relative_sequence = sequence_dir.relative_to(image_base)
            for side in ("left", "right"):
                image_dir = sequence_dir / side
                flow_dir = flow_base / relative_sequence / "into_future" / side
                if not image_dir.is_dir() or not flow_dir.is_dir():
                    continue
                frames = sorted_paths(image_dir.glob("*.png"))
                for index in range(max(0, len(frames) - 1)):
                    frame = frames[index]
                    flow_name = (
                        f"OpticalFlowIntoFuture_{frame.stem}_{side[0].upper()}.pfm"
                    )
                    flow = flow_dir / flow_name
                    if not flow.is_file():
                        continue
                    samples.append(FlowSample(
                        dataset="things",
                        sample_id=f"{relative_sequence}/{side}/{frame.stem}",
                        image1=frame,
                        image2=frames[index + 1],
                        flow=flow,
                    ))

    return samples


def discover_things(root: Path) -> List[FlowSample]:
    root = require_directory(root, "FlyingThings3D root")
    samples = _discover_things_subset(root)
    return samples if samples else _discover_things_full(root)


def discover_sintel(root: Path, pass_name: str) -> List[FlowSample]:
    root = require_directory(root, "MPI-Sintel root")
    image_base = root / "training" / pass_name
    flow_base = root / "training" / "flow"
    if not image_base.is_dir():
        return []

    samples: List[FlowSample] = []
    for scene_dir in sorted_paths(p for p in image_base.iterdir() if p.is_dir()):
        frames = sorted_paths(scene_dir.glob("*.png"))
        for index in range(max(0, len(frames) - 1)):
            frame = frames[index]
            flow = flow_base / scene_dir.name / f"{frame.stem}.flo"
            if not flow.is_file():
                continue
            key = f"sintel_{pass_name}"
            samples.append(FlowSample(
                dataset=key,
                sample_id=f"{scene_dir.name}/{frame.stem}",
                image1=frame,
                image2=frames[index + 1],
                flow=flow,
            ))

    return samples


def _resolve_spring_root(root: Path) -> Path:
    root = require_directory(root, "Spring root")
    if (root / "train").is_dir():
        return root
    if (root / "spring" / "train").is_dir():
        return root / "spring"
    return root


def discover_spring(root: Path) -> List[FlowSample]:
    root = _resolve_spring_root(root)
    train_root = root / "train"
    if not train_root.is_dir():
        return []

    samples: List[FlowSample] = []
    for scene_dir in sorted_paths(p for p in train_root.iterdir() if p.is_dir()):
        image_dir = scene_dir / "frame_left"
        flow_dir = scene_dir / "flow_FW_left"
        if not image_dir.is_dir() or not flow_dir.is_dir():
            continue
        frames = sorted_paths(
            path for path in image_dir.iterdir()
            if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg"}
        )

        for index in range(max(0, len(frames) - 1)):
            frame = frames[index]
            match = re.search(r"(\d+)$", frame.stem)
            if match is None:
                continue
            frame_index = match.group(1)
            prefix = "flow_FW_left"
            candidates = [
                flow_dir / f"{prefix}_{frame_index}{suffix}"
                for suffix in (".flo5", ".flo", ".pfm")
            ]
            candidates.extend(
                flow_dir / f"{frame.stem}{suffix}"
                for suffix in (".flo5", ".flo", ".pfm")
            )
            flow = next((path for path in candidates if path.is_file()), None)
            if flow is None:
                continue
            samples.append(FlowSample(
                dataset="spring",
                sample_id=f"{scene_dir.name}/{frame.stem}",
                image1=frame,
                image2=frames[index + 1],
                flow=flow,
                flow_encoding="spring",
            ))

    return samples


def _resolve_kitti_root(root: Path) -> Path:
    root = require_directory(root, "KITTI root")
    if (root / "training").is_dir():
        return root
    for nested in ("data_scene_flow", "data_stereo_flow"):
        if (root / nested / "training").is_dir():
            return root / nested
    return root


def discover_kitti(
    root: Path,
    dataset_key: str,
    camera: str,
    flow_type: str = "occ",
) -> List[FlowSample]:
    root = _resolve_kitti_root(root)
    image_dir = root / "training" / camera
    flow_dir = root / "training" / f"flow_{flow_type}"
    if not image_dir.is_dir() or not flow_dir.is_dir():
        return []

    samples: List[FlowSample] = []
    for image1 in sorted_paths(image_dir.glob("*_10.png")):
        base = image1.name.removesuffix("_10.png")
        image2 = image_dir / f"{base}_11.png"
        flow = flow_dir / f"{base}_10.png"
        if not image2.is_file() or not flow.is_file():
            continue
        samples.append(FlowSample(
            dataset=dataset_key,
            sample_id=base,
            image1=image1,
            image2=image2,
            flow=flow,
            flow_encoding="kitti_png",
        ))

    return samples


def _resolve_hd1k_root(root: Path) -> Path:
    root = require_directory(root, "HD1K root")
    if (root / "hd1k_full_package").is_dir():
        return root / "hd1k_full_package"
    return root


def discover_hd1k(root: Path) -> List[FlowSample]:
    root = _resolve_hd1k_root(root)
    image_dir = root / "hd1k_input" / "image_2"
    flow_dir = root / "hd1k_flow_gt" / "flow_occ"
    if not image_dir.is_dir() or not flow_dir.is_dir():
        return []

    samples: List[FlowSample] = []
    for flow in sorted_paths(flow_dir.glob("*.png")):
        if "_" not in flow.stem:
            continue
        prefix, frame_text = flow.stem.rsplit("_", 1)
        if not frame_text.isdigit():
            continue
        image1 = find_image_by_stem(image_dir, flow.stem)
        next_stem = f"{prefix}_{int(frame_text) + 1:0{len(frame_text)}d}"
        image2 = find_image_by_stem(image_dir, next_stem)
        if image1 is None or image2 is None:
            continue
        samples.append(FlowSample(
            dataset="hd1k",
            sample_id=flow.stem,
            image1=image1,
            image2=image2,
            flow=flow,
            flow_encoding="kitti_png",
        ))

    return samples


def discover_all(args: argparse.Namespace) -> Dict[str, List[FlowSample]]:
    return {
        "chairs": discover_chairs(args.chairs_root),
        "things": discover_things(args.things_root),
        "sintel_clean": discover_sintel(args.sintel_root, "clean"),
        "sintel_final": discover_sintel(args.sintel_root, "final"),
        "spring": discover_spring(args.spring_root),
        "kitti2012": discover_kitti(
            args.kitti2012_root, "kitti2012", "image_0", "occ"
        ),
        "kitti2015": discover_kitti(
            args.kitti2015_root, "kitti2015", "image_2", "occ"
        ),
        "hd1k": discover_hd1k(args.hd1k_root),
    }


# ---------------------------------------------------------------------------
# Flow and image loading
# ---------------------------------------------------------------------------

def read_rgb(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def read_flo(path: Path) -> np.ndarray:
    with path.open("rb") as handle:
        magic_bytes = handle.read(4)
        if len(magic_bytes) != 4:
            raise ValueError(f"Truncated .flo header: {path}")
        magic = struct.unpack("<f", magic_bytes)[0]
        if not math.isclose(magic, 202021.25, rel_tol=0.0, abs_tol=1e-3):
            raise ValueError(f"Invalid .flo magic {magic}: {path}")
        width, height = struct.unpack("<ii", handle.read(8))
        if width <= 0 or height <= 0:
            raise ValueError(f"Invalid .flo dimensions {(width, height)}: {path}")
        data = np.frombuffer(handle.read(), dtype="<f4")
    expected = height * width * 2
    if data.size != expected:
        raise ValueError(
            f"Expected {expected} flow values, found {data.size}: {path}"
        )
    return data.reshape(height, width, 2).astype(np.float32)


def _read_non_comment_line(handle) -> bytes:
    while True:
        line = handle.readline()
        if not line:
            raise ValueError("Unexpected end of PFM header")
        stripped = line.strip()
        if stripped and not stripped.startswith(b"#"):
            return stripped


def read_pfm(path: Path) -> np.ndarray:
    with path.open("rb") as handle:
        header = _read_non_comment_line(handle)
        if header not in {b"PF", b"Pf"}:
            raise ValueError(f"Not a PFM file: {path}")
        channels = 3 if header == b"PF" else 1
        width, height = map(int, _read_non_comment_line(handle).split())
        scale = float(_read_non_comment_line(handle))
        dtype = "<f4" if scale < 0 else ">f4"
        data = np.frombuffer(handle.read(), dtype=dtype)
    expected = width * height * channels
    if data.size != expected:
        raise ValueError(
            f"Expected {expected} PFM values, found {data.size}: {path}"
        )
    data = data.reshape(height, width, channels)
    data = np.flipud(data)
    if channels < 2:
        raise ValueError(f"PFM does not contain two flow channels: {path}")
    return np.ascontiguousarray(data[..., :2].astype(np.float32))


def read_flo5(path: Path) -> np.ndarray:
    try:
        import h5py
    except ImportError as exc:
        raise ImportError(
            "Spring .flo5 reading requires h5py. Install it with "
            "`python -m pip install h5py`."
        ) from exc

    with h5py.File(path, "r") as handle:
        dataset = handle.get("flow")
        if dataset is None:
            candidates = []

            def visitor(_name, obj) -> None:
                if hasattr(obj, "shape") and len(obj.shape) == 3:
                    candidates.append(obj)

            handle.visititems(visitor)
            dataset = candidates[0] if candidates else None
        if dataset is None:
            raise ValueError(f"No three-dimensional flow dataset found in {path}")
        flow = np.asarray(dataset, dtype=np.float32)

    if flow.ndim != 3:
        raise ValueError(f"Expected three-dimensional .flo5 data: {path}")
    if flow.shape[-1] in {2, 3}:
        flow = flow[..., :2]
    elif flow.shape[0] in {2, 3}:
        flow = np.transpose(flow[:2], (1, 2, 0))
    else:
        raise ValueError(f"Cannot interpret .flo5 shape {flow.shape}: {path}")
    return np.ascontiguousarray(flow.astype(np.float32))


def read_kitti_flow_png(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    encoded = cv2.imread(
        str(path), cv2.IMREAD_ANYDEPTH | cv2.IMREAD_COLOR
    )
    if encoded is None:
        raise FileNotFoundError(f"Could not read encoded flow PNG: {path}")
    if encoded.ndim != 3 or encoded.shape[2] != 3:
        raise ValueError(f"Expected three-channel flow PNG, got {encoded.shape}: {path}")
    if encoded.dtype != np.uint16:
        raise ValueError(
            f"Expected uint16 KITTI/HD1K flow PNG, got {encoded.dtype}: {path}"
        )

    # OpenCV channel order is BGR: validity, v, u.
    valid = encoded[..., 0].astype(bool)
    u = (encoded[..., 2].astype(np.float32) - 2 ** 15) / 64.0
    v = (encoded[..., 1].astype(np.float32) - 2 ** 15) / 64.0
    flow = np.stack((u, v), axis=-1)
    valid &= np.isfinite(flow).all(axis=-1)
    flow[~valid] = 0.0
    return flow.astype(np.float32), valid


def read_dense_flow(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    suffix = path.suffix.lower()
    if suffix == ".flo":
        flow = read_flo(path)
    elif suffix == ".pfm":
        flow = read_pfm(path)
    elif suffix == ".flo5":
        flow = read_flo5(path)
    else:
        raise ValueError(f"Unsupported dense-flow format {suffix}: {path}")

    valid = np.isfinite(flow).all(axis=-1)
    valid &= np.max(np.abs(flow), axis=-1) < 1e9
    flow = flow.copy()
    flow[~valid] = 0.0
    return flow.astype(np.float32), valid


def align_flow_to_image(
    flow: np.ndarray,
    valid: np.ndarray,
    image_shape: Sequence[int],
    sample: FlowSample,
) -> Tuple[np.ndarray, np.ndarray]:
    image_height, image_width = image_shape[:2]
    flow_height, flow_width = flow.shape[:2]
    if (flow_height, flow_width) == (image_height, image_width):
        return flow, valid

    # Spring distributes UHD flow for HD frames.  Subsampling does not scale
    # the vector values because the vectors are already expressed on the HD
    # image grid in the official release.
    if (
        sample.flow_encoding == "spring"
        and flow_height == 2 * image_height
        and flow_width == 2 * image_width
    ):
        return (
            np.ascontiguousarray(flow[::2, ::2]),
            np.ascontiguousarray(valid[::2, ::2]),
        )

    raise ValueError(
        "Flow/image resolution mismatch for "
        f"{sample.dataset}:{sample.sample_id}: "
        f"image={(image_height, image_width)}, flow={(flow_height, flow_width)}"
    )


def load_sample(sample: FlowSample) -> LoadedSample:
    image1 = read_rgb(sample.image1)
    image2 = read_rgb(sample.image2)
    if sample.flow_encoding == "kitti_png":
        flow, valid = read_kitti_flow_png(sample.flow)
    else:
        flow, valid = read_dense_flow(sample.flow)
    flow, valid = align_flow_to_image(flow, valid, image1.shape, sample)

    if image2.shape[:2] != image1.shape[:2]:
        raise ValueError(
            f"Input-frame size mismatch for {sample.dataset}:{sample.sample_id}: "
            f"{image1.shape[:2]} versus {image2.shape[:2]}"
        )

    magnitudes = np.linalg.norm(flow, axis=-1)
    valid_magnitudes = magnitudes[valid]
    p95 = (
        float(np.percentile(valid_magnitudes, 95.0))
        if valid_magnitudes.size
        else 0.0
    )
    return LoadedSample(
        sample=sample,
        image1=image1,
        image2=image2,
        flow=flow,
        valid=valid,
        p95_magnitude=p95,
        valid_fraction=float(valid.mean()),
    )


# ---------------------------------------------------------------------------
# Deterministic representative-sample selection
# ---------------------------------------------------------------------------

def evenly_spaced_indices(length: int, count: int) -> List[int]:
    if length <= 0:
        return []
    count = max(1, min(length, count))
    return sorted(set(np.linspace(0, length - 1, count).round().astype(int).tolist()))


def percentile_ranks(values: np.ndarray) -> np.ndarray:
    if values.size <= 1:
        return np.ones_like(values, dtype=np.float64)
    order = np.argsort(values, kind="stable")
    ranks = np.empty(values.size, dtype=np.float64)
    ranks[order] = np.linspace(0.0, 1.0, values.size)
    return ranks


def image_quality_features(image: np.ndarray) -> Tuple[float, float]:
    height, width = image.shape[:2]
    scale = min(1.0, 320.0 / max(height, width))
    if scale < 1.0:
        image = cv2.resize(
            image,
            (max(1, round(width * scale)), max(1, round(height * scale))),
            interpolation=cv2.INTER_AREA,
        )
    grey = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32)
    detail = float(cv2.Laplacian(grey, cv2.CV_32F).var())
    exposed = float(((grey > 8.0) & (grey < 247.0)).mean())
    return detail, exposed


def auto_select(
    dataset_key: str,
    samples: Sequence[FlowSample],
    candidate_count: int,
) -> Tuple[int, LoadedSample, List[str]]:
    warnings: List[str] = []
    candidates = evenly_spaced_indices(len(samples), candidate_count)
    records = []

    for index in candidates:
        try:
            loaded = load_sample(samples[index])
            detail, exposure = image_quality_features(loaded.image1)
            records.append({
                "index": index,
                "loaded": loaded,
                "detail": detail,
                "exposure": exposure,
                "motion": loaded.p95_magnitude,
                "valid": loaded.valid_fraction,
            })
        except Exception as exc:  # candidate-level resilience, reported later
            warnings.append(f"candidate {index} could not be read: {exc}")

    if not records:
        detail = "\n".join(f"  - {message}" for message in warnings[:8])
        raise RuntimeError(
            f"No readable automatic-selection candidates for {dataset_key}.\n{detail}"
        )

    details = np.asarray([record["detail"] for record in records], dtype=np.float64)
    exposures = np.asarray([record["exposure"] for record in records], dtype=np.float64)
    motions = np.asarray([record["motion"] for record in records], dtype=np.float64)
    validities = np.asarray([record["valid"] for record in records], dtype=np.float64)

    detail_rank = percentile_ranks(details)
    valid_rank = percentile_ranks(validities)
    motion_rank = percentile_ranks(motions)
    # Prefer meaningful but non-extreme motion: approximately the 65th
    # percentile of the candidates considered for each dataset.
    representative_motion = 1.0 - np.minimum(
        1.0, np.abs(motion_rank - 0.65) / 0.65
    )
    exposure_quality = np.clip((exposures - 0.65) / 0.35, 0.0, 1.0)

    scores = (
        0.35 * detail_rank
        + 0.25 * representative_motion
        + 0.20 * valid_rank
        + 0.20 * exposure_quality
    )
    winner = int(np.argmax(scores))
    record = records[winner]
    return int(record["index"]), record["loaded"], warnings


def parse_sample_overrides(values: Sequence[str]) -> Dict[str, int]:
    overrides: Dict[str, int] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(
                f"Invalid --sample {value!r}; expected DATASET=INDEX"
            )
        key, index_text = value.split("=", 1)
        key = key.strip().lower()
        if key not in PANEL_KEYS:
            raise ValueError(
                f"Unknown sample key {key!r}; choose from {', '.join(PANEL_KEYS)}"
            )
        try:
            overrides[key] = int(index_text)
        except ValueError as exc:
            raise ValueError(f"Sample index is not an integer: {value!r}") from exc
    return overrides


def normalise_manual_index(index: int, length: int, key: str) -> int:
    resolved = index if index >= 0 else length + index
    if resolved < 0 or resolved >= length:
        raise IndexError(
            f"Sample index {index} is outside {key}'s range [-{length}, {length - 1}]"
        )
    return resolved


def match_sintel_counterpart(
    source: LoadedSample,
    target_samples: Sequence[FlowSample],
) -> Tuple[int, LoadedSample]:
    target_by_id = {sample.sample_id: (index, sample)
                    for index, sample in enumerate(target_samples)}
    matched = target_by_id.get(source.sample.sample_id)
    if matched is None:
        raise RuntimeError(
            "Could not match Sintel Clean and Final using scene/frame ID "
            f"{source.sample.sample_id!r}"
        )
    index, sample = matched
    return index, load_sample(sample)


def select_samples(
    discovered: Mapping[str, Sequence[FlowSample]],
    overrides: Mapping[str, int],
    candidate_count: int,
) -> Tuple[Dict[str, int], Dict[str, LoadedSample], List[str]]:
    selected_indices: Dict[str, int] = {}
    selected: Dict[str, LoadedSample] = {}
    warnings: List[str] = []

    for key in PANEL_KEYS:
        if not discovered[key]:
            raise RuntimeError(
                f"No valid {PANEL_TITLES[key]} samples were discovered. "
                "Check the corresponding --*-root argument and dataset layout."
            )

    # Select all non-Sintel panels first.
    for key in PANEL_KEYS:
        if key in {"sintel_clean", "sintel_final"}:
            continue
        samples = discovered[key]
        if key in overrides:
            index = normalise_manual_index(overrides[key], len(samples), key)
            loaded = load_sample(samples[index])
        else:
            index, loaded, selection_warnings = auto_select(
                key, samples, candidate_count
            )
            warnings.extend(f"{key}: {message}" for message in selection_warnings)
        selected_indices[key] = index
        selected[key] = loaded

    # Clean and Final must use the same sequence/frame unless the user
    # explicitly overrides both panels.
    clean_samples = discovered["sintel_clean"]
    final_samples = discovered["sintel_final"]
    if "sintel_clean" in overrides and "sintel_final" in overrides:
        for key, samples in (
            ("sintel_clean", clean_samples),
            ("sintel_final", final_samples),
        ):
            index = normalise_manual_index(overrides[key], len(samples), key)
            selected_indices[key] = index
            selected[key] = load_sample(samples[index])
    elif "sintel_clean" in overrides:
        clean_index = normalise_manual_index(
            overrides["sintel_clean"], len(clean_samples), "sintel_clean"
        )
        clean_loaded = load_sample(clean_samples[clean_index])
        final_index, final_loaded = match_sintel_counterpart(
            clean_loaded, final_samples
        )
        selected_indices.update(
            sintel_clean=clean_index, sintel_final=final_index
        )
        selected.update(
            sintel_clean=clean_loaded, sintel_final=final_loaded
        )
    else:
        if "sintel_final" in overrides:
            final_index = normalise_manual_index(
                overrides["sintel_final"], len(final_samples), "sintel_final"
            )
            final_loaded = load_sample(final_samples[final_index])
        else:
            final_index, final_loaded, selection_warnings = auto_select(
                "sintel_final", final_samples, candidate_count
            )
            warnings.extend(
                f"sintel_final: {message}" for message in selection_warnings
            )
        clean_index, clean_loaded = match_sintel_counterpart(
            final_loaded, clean_samples
        )
        selected_indices.update(
            sintel_clean=clean_index, sintel_final=final_index
        )
        selected.update(
            sintel_clean=clean_loaded, sintel_final=final_loaded
        )

    return selected_indices, selected, warnings


# ---------------------------------------------------------------------------
# Flow visualisation and panel rendering
# ---------------------------------------------------------------------------

def make_colorwheel() -> np.ndarray:
    ry, yg, gc, cb, bm, mr = 15, 6, 4, 11, 13, 6
    count = ry + yg + gc + cb + bm + mr
    wheel = np.zeros((count, 3), dtype=np.float32)
    column = 0
    wheel[column:column + ry, 0] = 255
    wheel[column:column + ry, 1] = np.floor(255 * np.arange(ry) / ry)
    column += ry
    wheel[column:column + yg, 0] = 255 - np.floor(255 * np.arange(yg) / yg)
    wheel[column:column + yg, 1] = 255
    column += yg
    wheel[column:column + gc, 1] = 255
    wheel[column:column + gc, 2] = np.floor(255 * np.arange(gc) / gc)
    column += gc
    wheel[column:column + cb, 1] = 255 - np.floor(255 * np.arange(cb) / cb)
    wheel[column:column + cb, 2] = 255
    column += cb
    wheel[column:column + bm, 2] = 255
    wheel[column:column + bm, 0] = np.floor(255 * np.arange(bm) / bm)
    column += bm
    wheel[column:column + mr, 2] = 255 - np.floor(255 * np.arange(mr) / mr)
    wheel[column:column + mr, 0] = 255
    return wheel


COLORWHEEL = make_colorwheel()


def flow_to_rgb(
    flow: np.ndarray,
    valid: np.ndarray,
    normalisation: float,
) -> np.ndarray:
    scale = max(float(normalisation), 1e-6)
    u = flow[..., 0] / scale
    v = flow[..., 1] / scale
    magnitude = np.sqrt(u ** 2 + v ** 2)

    number_of_colours = COLORWHEEL.shape[0]
    angle = np.arctan2(-v, -u) / np.pi
    fractional_index = (angle + 1.0) * 0.5 * (number_of_colours - 1)
    lower = np.floor(fractional_index).astype(np.int32) % number_of_colours
    upper = (lower + 1) % number_of_colours
    fraction = fractional_index - np.floor(fractional_index)

    colour = (
        (1.0 - fraction[..., None]) * COLORWHEEL[lower]
        + fraction[..., None] * COLORWHEEL[upper]
    )
    clipped_magnitude = np.clip(magnitude, 0.0, 1.0)[..., None]
    colour = 255.0 - clipped_magnitude * (255.0 - colour)
    colour = np.clip(colour, 0, 255).astype(np.uint8)
    colour[~valid] = LIGHT_GREY
    return colour


def make_flow_wheel_image(size: int = 192) -> np.ndarray:
    coordinates = np.linspace(-1.05, 1.05, size, dtype=np.float32)
    x_grid, y_grid = np.meshgrid(coordinates, coordinates)
    flow = np.stack((x_grid, -y_grid), axis=-1)
    valid = (x_grid ** 2 + y_grid ** 2) <= 1.0
    image = flow_to_rgb(flow, valid, 1.0)
    image[~valid] = 255
    return image


def crop_to_aspect(
    array: np.ndarray,
    target_aspect: float,
    mode: str,
    background: Sequence[int] = (245, 245, 245),
) -> np.ndarray:
    height, width = array.shape[:2]
    if height <= 0 or width <= 0:
        raise ValueError(f"Cannot crop empty array with shape {array.shape}")
    current_aspect = width / height

    if mode == "cover":
        if current_aspect > target_aspect:
            new_width = max(1, round(height * target_aspect))
            left = (width - new_width) // 2
            return np.ascontiguousarray(array[:, left:left + new_width])
        new_height = max(1, round(width / target_aspect))
        top = (height - new_height) // 2
        return np.ascontiguousarray(array[top:top + new_height, :])

    if mode != "contain":
        raise ValueError(f"Unknown fit mode: {mode}")

    channels = 1 if array.ndim == 2 else array.shape[2]
    if current_aspect > target_aspect:
        canvas_width = width
        canvas_height = max(height, round(width / target_aspect))
    else:
        canvas_height = height
        canvas_width = max(width, round(height * target_aspect))

    if channels == 1:
        canvas = np.full((canvas_height, canvas_width), background[0], dtype=array.dtype)
    else:
        colour = np.asarray(background[:channels], dtype=array.dtype)
        canvas = np.empty((canvas_height, canvas_width, channels), dtype=array.dtype)
        canvas[...] = colour
    top = (canvas_height - height) // 2
    left = (canvas_width - width) // 2
    canvas[top:top + height, left:left + width] = array
    return canvas


def style_image_axis(axis, border_colour: str) -> None:
    axis.set_xticks([])
    axis.set_yticks([])
    axis.set_facecolor("white")
    for spine in axis.spines.values():
        spine.set_visible(True)
        spine.set_color(border_colour)
        spine.set_linewidth(0.8)


def add_corner_label(axis, text: str, location: str = "left") -> None:
    x = 0.025 if location == "left" else 0.975
    alignment = "left" if location == "left" else "right"
    axis.text(
        x,
        0.94,
        text,
        transform=axis.transAxes,
        ha=alignment,
        va="top",
        fontsize=5.7,
        color="black",
        bbox={
            "boxstyle": "round,pad=0.16",
            "facecolor": "white",
            "edgecolor": "none",
            "alpha": 0.82,
        },
        zorder=5,
    )


def add_group_heading(axis, title: str) -> None:
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.axis("off")
    axis.axhline(0.48, color=MID_GREY, linewidth=0.65, zorder=1)
    axis.text(
        0.5,
        0.48,
        title,
        ha="center",
        va="center",
        fontsize=7.6,
        fontweight="bold",
        bbox={"facecolor": "white", "edgecolor": "none", "pad": 2.0},
        zorder=2,
    )


def render_dataset_card(
    figure: plt.Figure,
    subplot_spec,
    key: str,
    loaded: LoadedSample,
    target_aspect: float,
    fit_mode: str,
) -> None:
    card_grid = GridSpecFromSubplotSpec(
        3,
        2,
        subplot_spec=subplot_spec,
        height_ratios=(0.38, 1.0, 2.02),
        width_ratios=(1.0, 1.0),
        hspace=0.045,
        wspace=0.035,
    )

    title_axis = figure.add_subplot(card_grid[0, :])
    title_axis.axis("off")
    title_axis.text(
        0.5,
        0.70,
        f"({PANEL_LETTERS[key]}) {PANEL_TITLES[key]}",
        ha="center",
        va="center",
        fontsize=7.35,
        fontweight="bold",
        color="black",
    )
    title_axis.text(
        0.5,
        0.04,
        loaded.sample.sample_id,
        ha="center",
        va="bottom",
        fontsize=4.8,
        color=TEXT_GREY,
    )

    image1_axis = figure.add_subplot(card_grid[1, 0])
    image2_axis = figure.add_subplot(card_grid[1, 1])
    flow_axis = figure.add_subplot(card_grid[2, :])

    image1 = crop_to_aspect(loaded.image1, target_aspect, fit_mode)
    image2 = crop_to_aspect(loaded.image2, target_aspect, fit_mode)
    flow_rgb = flow_to_rgb(
        loaded.flow, loaded.valid, loaded.p95_magnitude
    )
    flow_rgb = crop_to_aspect(
        flow_rgb,
        target_aspect,
        fit_mode,
        background=LIGHT_GREY,
    )

    image1_axis.imshow(image1, interpolation="lanczos", rasterized=True)
    image2_axis.imshow(image2, interpolation="lanczos", rasterized=True)
    flow_axis.imshow(flow_rgb, interpolation="nearest", rasterized=True)

    style_image_axis(image1_axis, INPUT_ORANGE)
    style_image_axis(image2_axis, INPUT_ORANGE)
    style_image_axis(flow_axis, FLOW_GREEN)
    add_corner_label(image1_axis, r"$I_1$")
    add_corner_label(image2_axis, r"$I_2$")
    add_corner_label(flow_axis, r"$\mathbf{w}^{\star}$")
    add_corner_label(
        flow_axis,
        rf"$p_{{95}}={loaded.p95_magnitude:.1f}\,\mathrm{{px}}$",
        location="right",
    )


def configure_matplotlib() -> None:
    matplotlib.rcParams.update({
        "font.family": "serif",
        "font.serif": ["STIXGeneral", "Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "axes.unicode_minus": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "savefig.facecolor": "white",
        "figure.facecolor": "white",
    })


def build_figure(
    selected: Mapping[str, LoadedSample],
    width_inches: float,
    target_aspect: float,
    fit_mode: str,
) -> plt.Figure:
    configure_matplotlib()
    figure = plt.figure(figsize=(width_inches, width_inches * 0.70))
    outer = figure.add_gridspec(
        5,
        4,
        left=0.018,
        right=0.992,
        top=0.985,
        bottom=0.035,
        width_ratios=(1, 1, 1, 1),
        height_ratios=(0.085, 1.0, 0.085, 1.0, 0.16),
        hspace=0.105,
        wspace=0.075,
    )

    top_heading = figure.add_subplot(outer[0, :])
    bottom_heading = figure.add_subplot(outer[2, :])
    add_group_heading(top_heading, "Synthetic curriculum and cinematic rendering")
    add_group_heading(
        bottom_heading,
        "High-resolution synthetic and real driving domains",
    )

    for column, key in enumerate(PANEL_KEYS[:4]):
        render_dataset_card(
            figure, outer[1, column], key, selected[key], target_aspect, fit_mode
        )
    for column, key in enumerate(PANEL_KEYS[4:]):
        render_dataset_card(
            figure, outer[3, column], key, selected[key], target_aspect, fit_mode
        )

    legend_grid = GridSpecFromSubplotSpec(
        1,
        3,
        subplot_spec=outer[4, :],
        width_ratios=(1.0, 0.075, 1.0),
        wspace=0.035,
    )
    left_legend = figure.add_subplot(legend_grid[0, 0])
    wheel_axis = figure.add_subplot(legend_grid[0, 1])
    right_legend = figure.add_subplot(legend_grid[0, 2])
    for axis in (left_legend, wheel_axis, right_legend):
        axis.axis("off")

    left_legend.text(
        0.99,
        0.5,
        "Ground-truth flow",
        ha="right",
        va="center",
        fontsize=5.8,
        color=TEXT_GREY,
    )
    wheel_axis.imshow(make_flow_wheel_image(), interpolation="bilinear")
    right_legend.text(
        0.01,
        0.5,
        r"hue: direction; intensity: magnitude ($p_{95}$ normalised per sample)",
        ha="left",
        va="center",
        fontsize=5.8,
        color=TEXT_GREY,
    )

    return figure


# ---------------------------------------------------------------------------
# Output and command-line interface
# ---------------------------------------------------------------------------

def output_base(path: Path) -> Path:
    path = path.expanduser()
    if path.suffix.lower() in {".pdf", ".png", ".svg"}:
        path = path.with_suffix("")
    return path.resolve()


def save_outputs(
    figure: plt.Figure,
    base: Path,
    formats: Sequence[str],
    dpi: int,
) -> List[Path]:
    base.parent.mkdir(parents=True, exist_ok=True)
    written: List[Path] = []
    for output_format in formats:
        destination = base.with_suffix(f".{output_format}")
        save_kwargs = {
            "format": output_format,
            "facecolor": "white",
            "edgecolor": "none",
        }
        if output_format == "png":
            save_kwargs["dpi"] = dpi
        else:
            save_kwargs["dpi"] = 300
        figure.savefig(destination, **save_kwargs)
        written.append(destination)
    return written


def write_manifest(
    base: Path,
    args: argparse.Namespace,
    discovered: Mapping[str, Sequence[FlowSample]],
    selected_indices: Mapping[str, int],
    selected: Mapping[str, LoadedSample],
    outputs: Sequence[Path],
) -> Path:
    manifest_path = base.with_name(f"{base.name}_manifest.json")
    manifest = {
        "figure": {
            "outputs": [str(path) for path in outputs],
            "width_inches": args.width_inches,
            "dpi": args.dpi,
            "fit_mode": args.fit_mode,
            "target_aspect": args.target_aspect,
            "flow_normalisation": "per-sample 95th-percentile magnitude",
            "invalid_flow_colour_rgb": LIGHT_GREY.tolist(),
        },
        "datasets": {},
    }
    for key in PANEL_KEYS:
        loaded = selected[key]
        manifest["datasets"][key] = {
            "panel": PANEL_LETTERS[key],
            "title": PANEL_TITLES[key],
            "discovered_samples": len(discovered[key]),
            "selected_index": selected_indices[key],
            "sample": {
                **{
                    field: str(value) if isinstance(value, Path) else value
                    for field, value in asdict(loaded.sample).items()
                },
                "p95_magnitude_px": loaded.p95_magnitude,
                "valid_fraction": loaded.valid_fraction,
            },
        }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest_path


def parse_formats(value: str) -> List[str]:
    formats = [part.strip().lower() for part in value.split(",") if part.strip()]
    allowed = {"pdf", "png", "svg"}
    unknown = sorted(set(formats) - allowed)
    if unknown:
        raise argparse.ArgumentTypeError(
            f"Unsupported formats {unknown}; choose from pdf,png,svg"
        )
    if not formats:
        raise argparse.ArgumentTypeError("At least one output format is required")
    return formats


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--chairs-root", type=Path, default=DEFAULT_ROOTS["chairs"])
    parser.add_argument("--things-root", type=Path, default=DEFAULT_ROOTS["things"])
    parser.add_argument("--sintel-root", type=Path, default=DEFAULT_ROOTS["sintel"])
    parser.add_argument("--spring-root", type=Path, default=DEFAULT_ROOTS["spring"])
    parser.add_argument(
        "--kitti2012-root", type=Path, default=DEFAULT_ROOTS["kitti2012"]
    )
    parser.add_argument(
        "--kitti2015-root", type=Path, default=DEFAULT_ROOTS["kitti2015"]
    )
    parser.add_argument("--hd1k-root", type=Path, default=DEFAULT_ROOTS["hd1k"])
    parser.add_argument(
        "--sample",
        action="append",
        default=[],
        metavar="DATASET=INDEX",
        help=(
            "Select a discovered sample explicitly; repeat as needed. Keys: "
            + ", ".join(PANEL_KEYS)
        ),
    )
    parser.add_argument(
        "--auto-candidates",
        type=int,
        default=16,
        help="Evenly spaced candidates scored per automatically selected dataset.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("figures/dataset_overview"),
        help="Output base path; any .pdf/.png/.svg suffix is stripped.",
    )
    parser.add_argument(
        "--formats",
        type=parse_formats,
        default=parse_formats("pdf,png"),
        help="Comma-separated output formats: pdf,png,svg (default: pdf,png).",
    )
    parser.add_argument("--dpi", type=int, default=600, help="PNG resolution.")
    parser.add_argument(
        "--width-inches",
        type=float,
        default=7.2,
        help="Full figure width in inches (default targets a two-column figure).",
    )
    parser.add_argument(
        "--target-aspect",
        type=float,
        default=1.80,
        help="Aspect ratio of every image/flow tile.",
    )
    parser.add_argument(
        "--fit-mode",
        choices=("cover", "contain"),
        default="cover",
        help="cover crops centrally; contain preserves the entire native frame.",
    )
    return parser


def validate_arguments(args: argparse.Namespace) -> None:
    if args.auto_candidates < 1:
        raise ValueError("--auto-candidates must be at least one")
    if args.dpi < 72:
        raise ValueError("--dpi must be at least 72")
    if args.width_inches <= 0:
        raise ValueError("--width-inches must be positive")
    if args.target_aspect <= 0:
        raise ValueError("--target-aspect must be positive")


def main() -> None:
    parser = build_argument_parser()
    args = parser.parse_args()

    try:
        validate_arguments(args)
        overrides = parse_sample_overrides(args.sample)
        discovered = discover_all(args)
        print("Discovered supervised pairs:")
        for key in PANEL_KEYS:
            print(f"  {key:14s} {len(discovered[key]):6d}")

        selected_indices, selected, warnings = select_samples(
            discovered, overrides, args.auto_candidates
        )
        if warnings:
            print("\nAutomatic-selection warnings:", file=sys.stderr)
            for warning in warnings[:20]:
                print(f"  - {warning}", file=sys.stderr)
            if len(warnings) > 20:
                print(
                    f"  - ... {len(warnings) - 20} additional warnings omitted",
                    file=sys.stderr,
                )

        print("\nSelected samples:")
        for key in PANEL_KEYS:
            loaded = selected[key]
            print(
                f"  {key:14s} index={selected_indices[key]:6d}  "
                f"id={loaded.sample.sample_id}  "
                f"p95={loaded.p95_magnitude:7.2f}px  "
                f"valid={100.0 * loaded.valid_fraction:6.2f}%"
            )

        figure = build_figure(
            selected,
            width_inches=args.width_inches,
            target_aspect=args.target_aspect,
            fit_mode=args.fit_mode,
        )
        base = output_base(args.output)
        outputs = save_outputs(figure, base, args.formats, args.dpi)
        plt.close(figure)
        manifest = write_manifest(
            base, args, discovered, selected_indices, selected, outputs
        )

        print("\nWritten:")
        for path in outputs:
            print(f"  {path}")
        print(f"  {manifest}")
    except (FileNotFoundError, RuntimeError, ValueError, IndexError, ImportError) as exc:
        parser.exit(2, f"error: {exc}\n")


if __name__ == "__main__":
    main()
