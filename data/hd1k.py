"""HD1K optical flow dataset loader.

Expected layout:

    <root>/
        hd1k_full_package/
            hd1k_input/
                image_2/
                image_3/
            hd1k_flow_gt/
                flow_occ/
            hd1k_challenge/
                image_2/
                image_3/

or directly:

    <root>/
        hd1k_input/
        hd1k_flow_gt/
        hd1k_challenge/

This loader uses hd1k_input/image_2 and hd1k_flow_gt/flow_occ for
training/validation. The challenge split has images only.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch

from .base_dataset import FlowDataset


class HD1KDataset(FlowDataset):
    """
    HD1K optical flow loader.

    Args:
        root:
            Path to either:
              - /.../HD1K
              - /.../HD1K/hd1k_full_package
        split:
            "train" | "val" | "test"
        camera:
            Usually "image_2". "image_3" exists, but GT flow_occ is normally
            used with image_2.
        val_fraction:
            Fraction of sequence IDs held out for validation when split="val".
        val_num_sequences:
            Optional explicit number of sequence IDs held out for validation.
        augmentor:
            Optional training augmentor.
    """

    def __init__(
        self,
        root: str,
        split: str = "train",
        camera: str = "image_2",
        val_fraction: float = 0.10,
        val_num_sequences: Optional[int] = None,
        augmentor: Optional[Callable] = None,
    ) -> None:
        self.original_root = root
        self.root = self._resolve_root(root)
        self.camera = camera
        self.val_fraction = val_fraction
        self.val_num_sequences = val_num_sequences

        if split not in {"train", "val", "test"}:
            raise ValueError(f"split must be 'train', 'val', or 'test', got {split!r}")

        if camera not in {"image_2", "image_3"}:
            raise ValueError(f"camera must be 'image_2' or 'image_3', got {camera!r}")

        super().__init__(
            root=str(self.root),
            split=split,
            augmentor=augmentor,
            sparse=True,
        )

    @staticmethod
    def _resolve_root(root: str) -> Path:
        root_path = Path(root)

        if (root_path / "hd1k_full_package").is_dir():
            return root_path / "hd1k_full_package"

        return root_path

    # ------------------------------------------------------------------
    # Sample collection
    # ------------------------------------------------------------------

    def _collect_samples(self) -> None:
        if self.split == "test":
            self._collect_challenge_samples()
            return

        self._collect_train_val_samples()

    def _collect_train_val_samples(self) -> None:
        image_dir = Path(self.root) / "hd1k_input" / self.camera
        flow_dir = Path(self.root) / "hd1k_flow_gt" / "flow_occ"

        if not image_dir.is_dir():
            raise RuntimeError(f"HD1K image dir not found: {image_dir}")

        if not flow_dir.is_dir():
            raise RuntimeError(f"HD1K flow dir not found: {flow_dir}")

        flow_files = sorted(flow_dir.glob("*.png"))

        if not flow_files:
            raise RuntimeError(f"No HD1K flow PNGs found in: {flow_dir}")

        all_samples: List[Dict[str, str]] = []

        for flow_path in flow_files:
            parsed = self._parse_sequence_frame(flow_path.stem)
            if parsed is None:
                continue

            seq_id, frame_idx, frame_width, prefix = parsed

            img1 = self._find_image_by_stem(image_dir, flow_path.stem)
            img2_stem = f"{prefix}_{frame_idx + 1:0{frame_width}d}"
            img2 = self._find_image_by_stem(image_dir, img2_stem)

            if img1 is None or img2 is None:
                continue

            all_samples.append({
                "image1": str(img1),
                "image2": str(img2),
                "flow": str(flow_path),
                "seq_id": seq_id,
            })

        if not all_samples:
            raise RuntimeError(
                "HD1K found flow files, but no valid image/flow pairs.\n"
                f"image_dir={image_dir}\n"
                f"flow_dir={flow_dir}\n\n"
                "Inspect filenames with:\n"
                f"  find {image_dir} -maxdepth 1 -type f | head -20\n"
                f"  find {flow_dir} -maxdepth 1 -type f | head -20"
            )

        seq_ids = sorted({s["seq_id"] for s in all_samples})

        if self.val_num_sequences is not None:
            n_val = int(self.val_num_sequences)
        else:
            n_val = max(1, int(round(len(seq_ids) * float(self.val_fraction))))

        n_val = max(1, min(n_val, len(seq_ids) - 1)) if len(seq_ids) > 1 else 1
        val_seq_ids = set(seq_ids[-n_val:])

        for sample in all_samples:
            is_val = sample["seq_id"] in val_seq_ids

            if self.split == "val" and not is_val:
                continue

            if self.split == "train" and is_val:
                continue

            self._samples.append({
                "image1": sample["image1"],
                "image2": sample["image2"],
                "flow": sample["flow"],
            })

    def _collect_challenge_samples(self) -> None:
        image_dir = Path(self.root) / "hd1k_challenge" / self.camera

        if not image_dir.is_dir():
            raise RuntimeError(f"HD1K challenge image dir not found: {image_dir}")

        images = sorted([
            p for p in image_dir.iterdir()
            if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg"}
        ])

        by_seq: Dict[str, List[Path]] = {}

        for img in images:
            parsed = self._parse_sequence_frame(img.stem)
            if parsed is None:
                continue

            seq_id, _, _, _ = parsed
            by_seq.setdefault(seq_id, []).append(img)

        for seq_id in sorted(by_seq):
            frames = sorted(by_seq[seq_id])
            for i in range(len(frames) - 1):
                self._samples.append({
                    "image1": str(frames[i]),
                    "image2": str(frames[i + 1]),
                    "flow": None,
                })

    @staticmethod
    def _parse_sequence_frame(stem: str) -> Optional[Tuple[str, int, int, str]]:
        """
        Parse HD1K-style stems such as:

            000000_000000
            000001_000034

        Returns:
            seq_id, frame_idx, frame_width, prefix
        """
        if "_" not in stem:
            return None

        parts = stem.split("_")
        frame_str = parts[-1]
        prefix = "_".join(parts[:-1])
        seq_id = parts[0]

        if not frame_str.isdigit():
            return None

        return seq_id, int(frame_str), len(frame_str), prefix

    @staticmethod
    def _find_image_by_stem(image_dir: Path, stem: str) -> Optional[Path]:
        for ext in (".png", ".jpg", ".jpeg"):
            p = image_dir / f"{stem}{ext}"
            if p.is_file():
                return p

        return None

    # ------------------------------------------------------------------
    # HD1K / KITTI-style flow PNG decoding
    # ------------------------------------------------------------------

    @staticmethod
    def _read_flow_png(path: str) -> Tuple[np.ndarray, np.ndarray]:
        img = cv2.imread(path, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_COLOR)

        if img is None:
            raise FileNotFoundError(f"HD1K flow PNG not found: {path}")

        # cv2 loads as BGR.
        # KITTI-style optical flow PNG:
        #   channel 0: validity
        #   channel 1: v
        #   channel 2: u
        valid = img[:, :, 0].astype(bool)
        u = (img[:, :, 2].astype(np.float32) - 2 ** 15) / 64.0
        v = (img[:, :, 1].astype(np.float32) - 2 ** 15) / 64.0

        flow = np.stack([u, v], axis=-1)
        flow[~valid] = 0.0

        finite = np.isfinite(flow[:, :, 0]) & np.isfinite(flow[:, :, 1])
        valid = valid & finite

        return flow.astype(np.float32), valid.astype(bool)

    # ------------------------------------------------------------------
    # Override __getitem__ so we preserve the encoded validity mask
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
            # Challenge/test split: no GT.
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
