"""Base optical flow dataset class."""
from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate

# Keys whose values may be None (datasets without occlusion/invalid masks).
_NULLABLE_KEYS = {"occlusion", "invalid", "synthetic_occlusion"}


def flow_collate_fn(batch):
    """Custom collate that allows None values for optional mask keys."""
    result = {}
    for key in batch[0].keys():
        values = [sample[key] for sample in batch]
        if key in _NULLABLE_KEYS:
            if all(v is None for v in values):
                result[key] = None
            else:
                # Replace any None with zeros matching the first non-None shape.
                ref = next(v for v in values if v is not None)
                values = [v if v is not None else torch.zeros_like(ref)
                          for v in values]
                result[key] = torch.stack(values)
        else:
            result[key] = default_collate(values)
    return result


class FlowDataset(Dataset, ABC):
    """
    Abstract base for all optical flow datasets.

    Subclasses must implement:
        _collect_samples()  →  populates self._samples list of dicts with
                               keys: "image1", "image2", "flow" (paths).

    Optionally override:
        _load_flow(path)    if the flow format is non-standard.
        _load_valid(...)    to compute validity mask.
    """

    def __init__(
        self,
        root: str,
        split: str = "train",
        augmentor: Optional[Callable] = None,
        sparse: bool = False,
    ) -> None:
        super().__init__()
        self.root      = root
        self.split     = split
        self.augmentor = augmentor
        self.sparse    = sparse

        self._samples: List[Dict[str, Optional[str]]] = []
        self._collect_samples()

        if len(self._samples) == 0:
            raise RuntimeError(
                f"{self.__class__.__name__}: no samples found in {root!r} "
                f"(split={split!r}).  Check dataset path and directory structure."
            )

    # ---------------------------------------------------------------------- #
    # Abstract interface
    # ---------------------------------------------------------------------- #

    @abstractmethod
    def _collect_samples(self) -> None:
        """Populate self._samples."""

    # ---------------------------------------------------------------------- #
    # Data loading helpers
    # ---------------------------------------------------------------------- #

    @staticmethod
    def _load_image(path: str) -> np.ndarray:
        import cv2
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Image not found: {path}")
        return img.astype(np.float32)   # BGR, [0, 255]

    @staticmethod
    def _load_flow_flo(path: str) -> np.ndarray:
        """Load Middlebury .flo format."""
        with open(path, "rb") as f:
            magic = np.frombuffer(f.read(4), dtype=np.float32)
            assert magic == 202021.25, f"Invalid .flo magic: {magic}"
            W, H = np.frombuffer(f.read(8), dtype=np.int32)
            flow = np.frombuffer(f.read(H * W * 8), dtype=np.float32)
        return flow.reshape(H, W, 2)

    @staticmethod
    def _load_flow_pfm(path: str) -> np.ndarray:
        """Load .pfm format (used by Spring)."""
        import re
        with open(path, "rb") as f:
            header = f.readline().decode("utf-8").rstrip()
            if header not in ("PF", "Pf"):
                raise ValueError(f"Not a PFM file: {path}")
            channels = 3 if header == "PF" else 1
            dims = f.readline().decode("utf-8").rstrip()
            W, H = map(int, dims.split())
            scale = float(f.readline().decode("utf-8").rstrip())
            little_endian = scale < 0
            data = np.frombuffer(f.read(), dtype=np.float32)
        data = data.reshape(H, W, channels)
        if little_endian:
            data = data[::-1]          # flip vertically
        return data[:, :, :2]         # u, v only

    @staticmethod
    def _load_mask_png(path: str) -> np.ndarray:
        """Load a single-channel PNG mask (0=valid/visible, 255=occluded/invalid).
        Returns bool array True where the pixel is *usable* (not masked out).
        """
        import cv2
        mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"Mask not found: {path}")
        return mask == 0   # 0 → usable; 255 → discard

    def _load_flow(self, path: str) -> np.ndarray:
        if path.endswith(".flo"):
            return self._load_flow_flo(path)
        if path.endswith(".pfm"):
            return self._load_flow_pfm(path)
        raise ValueError(f"Unknown flow format: {path}")

    # ---------------------------------------------------------------------- #
    # __getitem__
    # ---------------------------------------------------------------------- #

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        s = self._samples[idx]

        img1 = self._load_image(s["image1"])
        img2 = self._load_image(s["image2"])

        flow: Optional[np.ndarray] = None
        valid: Optional[np.ndarray] = None
        occ_np: Optional[np.ndarray] = None
        inv_np: Optional[np.ndarray] = None

        if s.get("flow") is not None:
            flow = self._load_flow(s["flow"])
            if self.sparse:
                valid = self._sparse_valid(flow)
                flow = self._fill_sparse(flow, valid)
            else:
                valid = np.ones((flow.shape[0], flow.shape[1]), dtype=bool)
                valid &= np.isfinite(flow[:, :, 0]) & np.isfinite(flow[:, :, 1])

            # Occlusion describes correspondence validity, not GT-flow
            # validity.  Sintel supplies a valid flow vector for these pixels;
            # retain it for prior/null-space supervision and use the occlusion
            # mask separately as a visibility target.
            if s.get("occlusion") is not None:
                occ_np = ~self._load_mask_png(s["occlusion"])  # True = occluded

            # Apply invalid mask: GT unreliable (blur, reflections, etc.)
            if s.get("invalid") is not None:
                inv_np = ~self._load_mask_png(s["invalid"])    # True = invalid
                valid &= ~inv_np

        sample: Dict = {
            "image1":    img1,
            "image2":    img2,
            "flow":      flow,
            "valid":     valid,
            "occlusion": occ_np,
            "invalid":   inv_np,
            "synthetic_occlusion": None,
        }

        if self.augmentor is not None and self.split == "train":
            sample = self.augmentor(sample)

        def _mask_to_tensor(m: Optional[np.ndarray]) -> Optional[torch.Tensor]:
            return torch.from_numpy(m.astype(np.float32)) if m is not None else None

        return {
            "image1":    self._to_tensor(sample["image1"]),
            "image2":    self._to_tensor(sample["image2"]),
            "flow":      torch.from_numpy(
                             sample["flow"].transpose(2, 0, 1).copy()
                         ).float() if sample["flow"] is not None
                         else torch.zeros(2, 1, 1),
            "valid":     torch.from_numpy(
                             sample["valid"].astype(np.float32)
                         ) if sample["valid"] is not None
                         else torch.ones(1, 1),
            "occlusion": _mask_to_tensor(sample.get("occlusion")),
            "invalid":   _mask_to_tensor(sample.get("invalid")),
            "synthetic_occlusion": _mask_to_tensor(
                sample.get("synthetic_occlusion")
            ),
        }

    def __len__(self) -> int:
        return len(self._samples)

    # ---------------------------------------------------------------------- #
    # Helpers
    # ---------------------------------------------------------------------- #

    @staticmethod
    def _to_tensor(img: np.ndarray) -> torch.Tensor:
        """BGR numpy (H,W,3) → RGB torch (3,H,W) float."""
        img = img[:, :, ::-1].astype(np.float32) / 255.0    # BGR→RGB, [0,1]
        return torch.from_numpy(img.transpose(2, 0, 1).copy())

    @staticmethod
    def _sparse_valid(flow: np.ndarray) -> np.ndarray:
        """Validity mask for sparse flow: non-zero and finite."""
        u, v = flow[:, :, 0], flow[:, :, 1]
        return (np.abs(u) > 0) | (np.abs(v) > 0)

    @staticmethod
    def _fill_sparse(flow: np.ndarray, valid: np.ndarray) -> np.ndarray:
        """Zero-fill invalid flow entries."""
        flow = flow.copy()
        flow[~valid] = 0.0
        return flow
