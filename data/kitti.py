"""KITTI optical flow dataset (2012 and 2015).

Directory layout (KITTI-2015):
    <root>/
        training/
            image_2/  XXXXXX_10.png  XXXXXX_11.png
            flow_occ/ XXXXXX_10.png  (16-bit GT flow, sparse)
        testing/
            image_2/  ...

KITTI uses sparse ground truth (LiDAR-derived), so validity masks are needed.
"""
from __future__ import annotations

import os
from typing import Callable, Optional

import cv2
import numpy as np

from .base_dataset import FlowDataset


class KITTIDataset(FlowDataset):
    """
    Supports KITTI-2015 and KITTI-2012 (same directory layout differences).

    Args:
        root:     Path to the KITTI dataset root (e.g. KITTI/2015).
        split:    "train" | "val" | "test".
        augmentor: Optional augmentation callable.
        val_size:  Number of training samples held out for validation.
    """

    def __init__(
        self,
        root: str,
        split: str = "train",
        augmentor: Optional[Callable] = None,
        val_size: int = 40,
    ) -> None:
        self.val_size = val_size
        super().__init__(root=root, split=split, augmentor=augmentor, sparse=True)

    def _collect_samples(self) -> None:
        if self.split == "test":
            img_dir = os.path.join(self.root, "testing", "image_2")
            frames = sorted([f for f in os.listdir(img_dir) if f.endswith("_10.png")])
            for f in frames:
                base = f.replace("_10.png", "")
                self._samples.append({
                    "image1":    os.path.join(img_dir, f"{base}_10.png"),
                    "image2":    os.path.join(img_dir, f"{base}_11.png"),
                    "flow":      None,
                })
            return

        img_dir  = os.path.join(self.root, "training", "image_2")
        flow_dir = os.path.join(self.root, "training", "flow_occ")

        frames = sorted([f for f in os.listdir(img_dir) if f.endswith("_10.png")])
        all_samples = []
        for f in frames:
            base = f.replace("_10.png", "")
            flow_path = os.path.join(flow_dir, f"{base}_10.png")
            all_samples.append({
                "image1": os.path.join(img_dir, f"{base}_10.png"),
                "image2": os.path.join(img_dir, f"{base}_11.png"),
                "flow":   flow_path,
            })

        if self.split == "val":
            self._samples = all_samples[-self.val_size:]
        else:
            self._samples = all_samples[:-self.val_size]

    # Override flow loading for KITTI's 16-bit PNG format
    def _load_flow(self, path: str) -> np.ndarray:
        img = cv2.imread(path, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"KITTI flow not found: {path}")
        # KITTI PNG encoding: flow = (pixel - 2^15) / 64.0
        # Channel 2 (B in BGR) is the validity mask (1 = valid)
        valid = img[:, :, 0].astype(bool)
        u = (img[:, :, 2].astype(np.float32) - 2 ** 15) / 64.0
        v = (img[:, :, 1].astype(np.float32) - 2 ** 15) / 64.0
        flow = np.stack([u, v], axis=-1)
        flow[~valid] = 0.0
        return flow

    @staticmethod
    def _sparse_valid(flow: np.ndarray) -> np.ndarray:
        # For KITTI, validity is encoded in the PNG; we derive it from non-zero
        return (np.abs(flow[:, :, 0]) + np.abs(flow[:, :, 1])) > 0
