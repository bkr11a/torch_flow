"""KITTI optical flow dataset loader for KITTI 2012 and KITTI 2015.

Supported layouts:

KITTI 2015:
    <root>/
        training/
            image_2/
            image_3/
            flow_occ/
            flow_noc/
        testing/
            image_2/
            image_3/

KITTI 2012:
    <root>/
        training/
            image_0/
            image_1/
            flow_occ/
            flow_noc/
        testing/
            image_0/
            image_1/

You may pass either:
    /.../KITTI 2015/data_scene_flow
    /.../KITTI 2015

or:
    /.../KITTI 2012/data_stereo_flow
    /.../KITTI 2012

The loader auto-resolves data_scene_flow / data_stereo_flow if needed.

KITTI flow PNG format:
    cv2 loads the PNG as BGR.
    B channel: validity
    G channel: v
    R channel: u

    u = (R - 2^15) / 64
    v = (G - 2^15) / 64
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Callable, Optional, Tuple

import cv2
import numpy as np
import torch

from .base_dataset import FlowDataset


class KITTIDataset(FlowDataset):
    """
    Supports KITTI-2015 and KITTI-2012.

    Args:
        root:
            Path to KITTI data root. Can be either the parent folder or the
            actual data_scene_flow/data_stereo_flow folder.
        split:
            "train" | "val" | "test".
        augmentor:
            Optional augmentation callable.
        val_size:
            Number of training samples held out for validation.
        camera:
            Optional explicit camera folder. If None, auto-detects:
              - image_2 for KITTI 2015
              - image_0 for KITTI 2012
        flow_type:
            "occ" or "noc". Usually "occ" for standard training/evaluation.
    """

    def __init__(
        self,
        root: str,
        split: str = "train",
        augmentor: Optional[Callable] = None,
        val_size: int = 40,
        camera: Optional[str] = None,
        flow_type: str = "occ",
    ) -> None:
        self.original_root = root
        self.root_path = self._resolve_root(root)
        self.val_size = int(val_size)
        self.camera = camera
        self.flow_type = flow_type

        if split not in {"train", "val", "test"}:
            raise ValueError(f"split must be 'train', 'val', or 'test', got {split!r}")

        if flow_type not in {"occ", "noc"}:
            raise ValueError(f"flow_type must be 'occ' or 'noc', got {flow_type!r}")

        super().__init__(
            root=str(self.root_path),
            split=split,
            augmentor=augmentor,
            sparse=True,
        )

    @staticmethod
    def _resolve_root(root: str) -> Path:
        root_path = Path(root)

        if (root_path / "training").is_dir():
            return root_path

        if (root_path / "data_scene_flow" / "training").is_dir():
            return root_path / "data_scene_flow"

        if (root_path / "data_stereo_flow" / "training").is_dir():
            return root_path / "data_stereo_flow"

        return root_path

    def _detect_camera_dir(self, split_dir: Path) -> Path:
        if self.camera is not None:
            candidate = split_dir / self.camera
            if not candidate.is_dir():
                raise RuntimeError(f"KITTI camera dir not found: {candidate}")
            return candidate

        # KITTI 2015 uses image_2. KITTI 2012 uses image_0.
        for cam in ("image_2", "image_0", "colored_0"):
            candidate = split_dir / cam
            if candidate.is_dir():
                return candidate

        raise RuntimeError(
            f"Could not auto-detect KITTI image dir under {split_dir}. "
            "Expected image_2 for KITTI 2015 or image_0 for KITTI 2012."
        )

    def _flow_dir(self) -> Path:
        return self.root_path / "training" / f"flow_{self.flow_type}"

    def _collect_samples(self) -> None:
        if self.split == "test":
            self._collect_test_samples()
            return

        self._collect_train_val_samples()

    def _collect_test_samples(self) -> None:
        split_dir = self.root_path / "testing"
        img_dir = self._detect_camera_dir(split_dir)

        frames = sorted([
            f for f in os.listdir(img_dir)
            if f.endswith("_10.png")
        ])

        for f in frames:
            base = f.replace("_10.png", "")
            img1 = img_dir / f"{base}_10.png"
            img2 = img_dir / f"{base}_11.png"

            if not img2.is_file():
                continue

            self._samples.append({
                "image1": str(img1),
                "image2": str(img2),
                "flow": None,
            })

    def _collect_train_val_samples(self) -> None:
        split_dir = self.root_path / "training"
        img_dir = self._detect_camera_dir(split_dir)
        flow_dir = self._flow_dir()

        if not img_dir.is_dir():
            raise RuntimeError(f"KITTI image dir not found: {img_dir}")

        if not flow_dir.is_dir():
            raise RuntimeError(f"KITTI flow dir not found: {flow_dir}")

        frames = sorted([
            f for f in os.listdir(img_dir)
            if f.endswith("_10.png")
        ])

        all_samples = []

        for f in frames:
            base = f.replace("_10.png", "")

            img1 = img_dir / f"{base}_10.png"
            img2 = img_dir / f"{base}_11.png"
            flow_path = flow_dir / f"{base}_10.png"

            if not img1.is_file() or not img2.is_file() or not flow_path.is_file():
                continue

            all_samples.append({
                "image1": str(img1),
                "image2": str(img2),
                "flow": str(flow_path),
            })

        if not all_samples:
            raise RuntimeError(
                "KITTI loader found no valid image/flow pairs.\n"
                f"root={self.root_path}\n"
                f"img_dir={img_dir}\n"
                f"flow_dir={flow_dir}\n\n"
                "Inspect with:\n"
                f"  find {img_dir} -maxdepth 1 -type f | head -20\n"
                f"  find {flow_dir} -maxdepth 1 -type f | head -20"
            )

        if self.val_size <= 0:
            self._samples = all_samples
            return

        val_size = min(self.val_size, max(1, len(all_samples) - 1))

        if self.split == "val":
            self._samples = all_samples[-val_size:]
        else:
            self._samples = all_samples[:-val_size]

    # ------------------------------------------------------------------
    # KITTI PNG flow decoding
    # ------------------------------------------------------------------

    @staticmethod
    def _read_flow_png(path: str) -> Tuple[np.ndarray, np.ndarray]:
        img = cv2.imread(path, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_COLOR)

        if img is None:
            raise FileNotFoundError(f"KITTI flow not found: {path}")

        if img.ndim != 3 or img.shape[2] != 3:
            raise ValueError(f"Expected 3-channel KITTI flow PNG, got {img.shape}: {path}")

        # cv2 loads as BGR.
        # KITTI optical-flow PNG:
        #   B channel: validity
        #   G channel: v
        #   R channel: u
        valid = img[:, :, 0].astype(bool)
        u = (img[:, :, 2].astype(np.float32) - 2 ** 15) / 64.0
        v = (img[:, :, 1].astype(np.float32) - 2 ** 15) / 64.0

        flow = np.stack([u, v], axis=-1).astype(np.float32)

        finite = np.isfinite(flow[:, :, 0]) & np.isfinite(flow[:, :, 1])
        valid = valid & finite

        flow[~valid] = 0.0

        return flow, valid

    # Keep this for compatibility, but do not rely on it for validity.
    def _load_flow(self, path: str) -> np.ndarray:
        flow, _ = self._read_flow_png(path)
        return flow

    @staticmethod
    def _sparse_valid(flow: np.ndarray) -> np.ndarray:
        # Not used by this class because __getitem__ preserves the encoded PNG
        # validity mask directly. This fallback exists only for compatibility.
        return np.isfinite(flow[:, :, 0]) & np.isfinite(flow[:, :, 1])

    # ------------------------------------------------------------------
    # Override __getitem__ to preserve encoded validity
    # ------------------------------------------------------------------

    def __getitem__(self, idx: int):
        s = self._samples[idx]

        img1 = self._load_image(s["image1"])
        img2 = self._load_image(s["image2"])

        occ_np = None
        inv_np = None

        if s.get("flow") is not None:
            flow, valid = self._read_flow_png(s["flow"])
        else:
            h, w = img1.shape[:2]
            flow = np.zeros((h, w, 2), dtype=np.float32)
            valid = np.zeros((h, w), dtype=bool)

        sample = {
            "image1": img1,
            "image2": img2,
            "flow": flow,
            "valid": valid,
            "occlusion": occ_np,
            "invalid": inv_np,
            "synthetic_occlusion": None,
        }

        if self.augmentor is not None and self.split == "train":
            sample = self.augmentor(sample)

        def _mask_to_tensor(m):
            return torch.from_numpy(m.astype(np.float32)) if m is not None else None

        return {
            "image1": self._to_tensor(sample["image1"]),
            "image2": self._to_tensor(sample["image2"]),
            "flow": torch.from_numpy(
                sample["flow"].transpose(2, 0, 1).copy()
            ).float(),
            "valid": torch.from_numpy(
                sample["valid"].astype(np.float32)
            ),
            "occlusion": _mask_to_tensor(sample.get("occlusion")),
            "invalid": _mask_to_tensor(sample.get("invalid")),
            "synthetic_occlusion": _mask_to_tensor(
                sample.get("synthetic_occlusion")
            ),
        }
