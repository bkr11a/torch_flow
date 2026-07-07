"""Spring optical flow dataset.

Expected extracted layout:

    <root>/
        train/
            <scene>/
                frame_left/
                    frame_left_0001.png
                    frame_left_0002.png
                    ...
                flow_FW_left/
                    flow_FW_left_0001.flo5
                    flow_FW_left_0002.flo5
                    ...
                flow_BW_left/
                frame_right/
                flow_FW_right/
                flow_BW_right/

The loader also accepts root paths one level above if the archive extracts into
a nested `spring/` directory.

Reference: Mehl et al., "Spring: A High-Resolution High-Detail Dataset and
           Benchmark for Scene Flow, Optical Flow and Stereo", CVPR 2023.
           https://spring-benchmark.org
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Callable, List, Optional

import torch
import numpy as np

from .base_dataset import FlowDataset


class SpringDataset(FlowDataset):
    """
    Args:
        root:       Path to Spring root.
        split:      "train" | "val" | "test".
        direction:  "forward" | "backward" | "both".
        side:       "left" | "right" | "both".
        augmentor:  Optional augmentation callable.
        val_scenes: Explicit validation scene names. If None, hold out the
                    last val_fraction of scenes.
        val_fraction: Fraction of scenes held out for validation.
    """

    def __init__(
        self,
        root: str,
        split: str = "train",
        direction: str = "forward",
        side: str = "left",
        augmentor: Optional[Callable] = None,
        val_scenes: Optional[List[str]] = None,
        val_fraction: float = 0.10,
    ) -> None:
        if split not in {"train", "val", "test"}:
            raise ValueError(f"split must be 'train', 'val', or 'test', got {split!r}")

        if direction not in {"forward", "backward", "both"}:
            raise ValueError(
                f"direction must be 'forward', 'backward', or 'both', got {direction!r}"
            )

        if side not in {"left", "right", "both"}:
            raise ValueError(f"side must be 'left', 'right', or 'both', got {side!r}")

        self.original_root = root
        self.root_path = self._resolve_root(root)
        self.direction = direction
        self.side = side
        self.val_scenes = val_scenes
        self.val_fraction = float(val_fraction)

        super().__init__(
            root=str(self.root_path),
            split=split,
            augmentor=augmentor,
            sparse=False,
        )

    @staticmethod
    def _resolve_root(root: str) -> Path:
        root_path = Path(root)

        if (root_path / "train").is_dir() or (root_path / "test").is_dir():
            return root_path

        if (root_path / "spring" / "train").is_dir():
            return root_path / "spring"

        return root_path

    def _collect_samples(self) -> None:
        split_dir = self.root_path / ("test" if self.split == "test" else "train")

        if not split_dir.is_dir():
            raise RuntimeError(f"Spring split dir not found: {split_dir}")

        scene_dirs = sorted([
            p for p in split_dir.iterdir()
            if p.is_dir()
        ])

        if not scene_dirs:
            raise RuntimeError(f"No Spring scene directories found in: {split_dir}")

        val_scenes = self._resolve_val_scenes(scene_dirs)

        for scene_path in scene_dirs:
            scene = scene_path.name
            is_val = scene in val_scenes

            if self.split == "val" and not is_val:
                continue

            if self.split == "train" and is_val:
                continue

            self._collect_scene(scene_path)

    def _resolve_val_scenes(self, scene_dirs: List[Path]) -> set[str]:
        if self.split == "test":
            return set()

        if self.val_scenes is not None:
            return set(self.val_scenes)

        # Hold out the last N scenes. This avoids hard-coding scene names such
        # as 00041/00044, which may not match the extracted archive naming.
        n = len(scene_dirs)

        if n <= 1:
            return set()

        n_val = max(1, int(round(n * self.val_fraction)))
        n_val = min(n_val, n - 1)

        return {p.name for p in scene_dirs[-n_val:]}

    def _sides(self):
        return ["left", "right"] if self.side == "both" else [self.side]

    def _directions(self):
        if self.direction == "both":
            return ["FW", "BW"]

        if self.direction == "forward":
            return ["FW"]

        return ["BW"]

    def _collect_scene(self, scene_path: Path) -> None:
        for side in self._sides():
            img_dir = scene_path / f"frame_{side}"

            if not img_dir.is_dir():
                continue

            frames = sorted([
                p for p in img_dir.iterdir()
                if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg"}
            ])

            if len(frames) < 2:
                continue

            for direction_code in self._directions():
                step = 1 if direction_code == "FW" else -1
                flow_dir = scene_path / f"flow_{direction_code}_{side}"

                if self.split != "test" and not flow_dir.is_dir():
                    continue

                for i, img1 in enumerate(frames):
                    j = i + step

                    if j < 0 or j >= len(frames):
                        continue

                    img2 = frames[j]

                    if self.split == "test":
                        self._samples.append({
                            "image1": str(img1),
                            "image2": str(img2),
                            "flow": None,
                        })
                        continue

                    flow_path = self._match_flow_file(
                        img_path=img1,
                        flow_dir=flow_dir,
                        direction_code=direction_code,
                        side=side,
                    )

                    # For train/val, do not append flow=None. That would create
                    # fake [2,1,1] zero-flow samples.
                    if flow_path is None:
                        continue

                    self._samples.append({
                        "image1": str(img1),
                        "image2": str(img2),
                        "flow": str(flow_path),
                    })

    @staticmethod
    def _frame_index_from_name(path: Path) -> str:
        """
        Examples:
            frame_left_0001.png -> 0001
            0001.png            -> 0001
        """
        stem = path.stem
        m = re.search(r"(\d+)$", stem)

        if m is None:
            raise ValueError(f"Could not parse frame index from Spring filename: {path}")

        return m.group(1)

    @classmethod
    def _match_flow_file(
        cls,
        img_path: Path,
        flow_dir: Path,
        direction_code: str,
        side: str,
    ) -> Optional[Path]:
        idx = cls._frame_index_from_name(img_path)
        flow_prefix = f"flow_{direction_code}_{side}"

        candidates = [
            flow_dir / f"{flow_prefix}_{idx}.flo5",
            flow_dir / f"{flow_prefix}_{idx}.flo",
            flow_dir / f"{flow_prefix}_{idx}.pfm",

            # Extra fallbacks in case the extracted archive uses image-like
            # names or omits the flow prefix.
            flow_dir / f"{img_path.stem}.flo5",
            flow_dir / f"{img_path.stem}.flo",
            flow_dir / f"{img_path.stem}.pfm",
            flow_dir / f"{idx}.flo5",
            flow_dir / f"{idx}.flo",
            flow_dir / f"{idx}.pfm",
        ]

        for p in candidates:
            if p.is_file():
                return p

        return None

    # ------------------------------------------------------------------
    # Flow loading
    # ------------------------------------------------------------------

    def _load_flow(self, path: str) -> np.ndarray:
        if path.endswith(".flo5"):
            return self._load_flow_flo5(path)

        return super()._load_flow(path)

    @staticmethod
    def _subsample_spring_flow_to_image_grid(
        flow: np.ndarray,
        image_shape,
        flow_path: str,
    ) -> np.ndarray:
        """
        Spring stores HD images but UHD ground-truth flow.

        Typical Spring optical-flow training layout:
            image: 1080 x 1920
            flow:  2160 x 3840

        SOTA-style training, e.g. SEA-RAFT-style Spring loading, subsamples the
        UHD flow to the image grid using every second GT value:

            flow = flow[::2, ::2]

        We do NOT scale u/v here.
        """
        img_h, img_w = image_shape[:2]
        flow_h, flow_w = flow.shape[:2]

        if (flow_h, flow_w) == (img_h, img_w):
            return np.ascontiguousarray(flow.astype(np.float32))

        if flow_h == 2 * img_h and flow_w == 2 * img_w:
            return np.ascontiguousarray(flow[::2, ::2].astype(np.float32))

        raise RuntimeError(
            "Spring flow/image resolution mismatch.\n"
            f"  image_shape: {(img_h, img_w)}\n"
            f"  flow_shape:  {(flow_h, flow_w)}\n"
            f"  flow_path:   {flow_path}\n\n"
            "Expected either same resolution or Spring's standard 2x UHD flow."
        )

    def __getitem__(self, idx: int):
        s = self._samples[idx]

        img1 = self._load_image(s["image1"])
        img2 = self._load_image(s["image2"])

        if s.get("flow") is not None:
            flow = self._load_flow(s["flow"])

            # Critical Spring-specific step:
            # .flo5 GT is UHD, images are HD. Bring GT onto image grid.
            flow = self._subsample_spring_flow_to_image_grid(
                flow=flow,
                image_shape=img1.shape,
                flow_path=s["flow"],
            )

            valid = np.ones((flow.shape[0], flow.shape[1]), dtype=bool)
            valid &= np.isfinite(flow[:, :, 0]) & np.isfinite(flow[:, :, 1])
            flow[~valid] = 0.0
        else:
            h, w = img1.shape[:2]
            flow = np.zeros((h, w, 2), dtype=np.float32)
            valid = np.zeros((h, w), dtype=bool)

        sample = {
            "image1": img1,
            "image2": img2,
            "flow": flow,
            "valid": valid,
            "occlusion": None,
            "invalid": None,
        }

        if self.augmentor is not None and self.split == "train":
            sample = self.augmentor(sample)

        return {
            "image1": self._to_tensor(sample["image1"]),
            "image2": self._to_tensor(sample["image2"]),
            "flow": torch.from_numpy(
                sample["flow"].transpose(2, 0, 1).copy()
            ).float(),
            "valid": torch.from_numpy(
                sample["valid"].astype(np.float32)
            ),
            "occlusion": None,
            "invalid": None,
        }

    @staticmethod
    def _load_flow_flo5(path: str) -> np.ndarray:
        """
        Load Spring .flo5 optical-flow files.

        Spring .flo5 files are HDF5-like files in typical Spring releases.
        This loader is intentionally robust to the dataset key name. It prefers
        a dataset named `flow`, but falls back to the first 3D dataset with at
        least two channels.
        """
        try:
            import h5py
        except ImportError as exc:
            raise ImportError(
                "Reading Spring .flo5 files requires h5py. Install it with:\n"
                "  pip install h5py"
            ) from exc

        with h5py.File(path, "r") as f:
            key = None

            if "flow" in f:
                key = "flow"
            else:
                for candidate in f.keys():
                    obj = f[candidate]
                    if hasattr(obj, "shape") and len(obj.shape) >= 3:
                        key = candidate
                        break

            if key is None:
                raise RuntimeError(
                    f"Could not find flow dataset inside .flo5 file: {path}. "
                    f"Available keys: {list(f.keys())}"
                )

            flow = np.asarray(f[key], dtype=np.float32)

        # Accept either H,W,2 or 2,H,W.
        if flow.ndim != 3:
            raise ValueError(f"Expected 3D flow array in {path}, got shape {flow.shape}")

        if flow.shape[-1] >= 2:
            flow = flow[..., :2]
        elif flow.shape[0] >= 2:
            flow = np.transpose(flow[:2], (1, 2, 0))
        else:
            raise ValueError(f"Could not interpret .flo5 flow shape {flow.shape}: {path}")

        return np.ascontiguousarray(flow.astype(np.float32))