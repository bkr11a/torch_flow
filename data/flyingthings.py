"""FlyingChairs and FlyingThings3D datasets for pre-training."""
from __future__ import annotations

import os
from typing import Callable, Optional

from .base_dataset import FlowDataset


class FlyingChairsDataset(FlowDataset):
    """
    FlyingChairs – 22,872 image pairs for optical flow pre-training.

    Directory layout:
        <root>/
            data/
                XXXXX_img1.ppm
                XXXXX_img2.ppm
                XXXXX_flow.flo
            train_val.txt   (1=train, 2=val)
    """

    def __init__(
        self,
        root: str,
        split: str = "train",
        augmentor: Optional[Callable] = None,
    ) -> None:
        super().__init__(root=root, split=split, augmentor=augmentor, sparse=False)

    def _collect_samples(self) -> None:
        data_dir = os.path.join(self.root, "data")
        split_file = os.path.join(self.root, "train_val.txt")

        all_flows = sorted([
            f for f in os.listdir(data_dir) if f.endswith("_flow.flo")
        ])

        if os.path.isfile(split_file):
            with open(split_file) as fh:
                labels = [int(l.strip()) for l in fh if l.strip()]
        else:
            # fallback: 90/10 split
            n = len(all_flows)
            labels = [1] * int(0.9 * n) + [2] * (n - int(0.9 * n))

        target_label = 1 if self.split == "train" else 2

        for flow_file, label in zip(all_flows, labels):
            if label != target_label:
                continue
            base = flow_file.replace("_flow.flo", "")
            self._samples.append({
                "image1": os.path.join(data_dir, f"{base}_img1.ppm"),
                "image2": os.path.join(data_dir, f"{base}_img2.ppm"),
                "flow":   os.path.join(data_dir, flow_file),
            })


class FlyingThingsDataset(FlowDataset):
    """
    FlyingThings3D dataset loader.

    Supports both:

    1. Full FlyingThings3D layout:
        <root>/
            frames_cleanpass/TRAIN/...
            frames_finalpass/TRAIN/...
            optical_flow/TRAIN/...

    2. FlyingThings3D subset layout:
        <root>/
            train/
                image_clean/
                    left/
                    right/
                flow/
                    left/
                        into_future/
                        into_past/
                    right/
                        into_future/
                        into_past/
            val/
                image_clean/
                    left/
                    right/
                flow/
                    left/
                        into_future/
                        into_past/
                    right/
                        into_future/
                        into_past/
    """

    def __init__(
        self,
        root: str,
        split: str = "train",
        dstype: str = "clean",
        side: str = "left",
        direction: str = "forward",
        augmentor: Optional[Callable] = None,
    ) -> None:
        self.dstype = dstype
        self.side = side
        self.direction = direction

        if self.side not in {"left", "right", "both"}:
            raise ValueError(f"side must be 'left', 'right', or 'both', got {side!r}")

        if self.direction not in {"forward", "backward", "both"}:
            raise ValueError(
                f"direction must be 'forward', 'backward', or 'both', got {direction!r}"
            )

        super().__init__(root=root, split=split, augmentor=augmentor, sparse=False)

    def _collect_samples(self) -> None:
        if self._looks_like_subset_layout():
            self._collect_subset_samples()
            return

        if self._looks_like_full_layout():
            self._collect_full_samples()
            return

        raise RuntimeError(
            "FlyingThings3D layout not recognised.\n"
            f"root={self.root!r}\n\n"
            "Expected subset layout:\n"
            "  train/image_clean/{left,right}\n"
            "  train/flow/{left,right}/{into_future,into_past}\n\n"
            "or full layout:\n"
            "  frames_cleanpass/TRAIN\n"
            "  optical_flow/TRAIN"
        )

    # ------------------------------------------------------------------
    # Layout detection
    # ------------------------------------------------------------------

    def _looks_like_subset_layout(self) -> bool:
        split_name = self._subset_split_name()
        return (
            os.path.isdir(os.path.join(self.root, split_name, "image_clean"))
            and os.path.isdir(os.path.join(self.root, split_name, "flow"))
        )

    def _looks_like_full_layout(self) -> bool:
        split_name = "TRAIN" if self.split == "train" else "TEST"
        img_base = os.path.join(self.root, f"frames_{self.dstype}pass", split_name)
        flow_base = os.path.join(self.root, "optical_flow", split_name)
        return os.path.isdir(img_base) and os.path.isdir(flow_base)

    # ------------------------------------------------------------------
    # FlyingThings3D subset layout
    # ------------------------------------------------------------------

    def _subset_split_name(self) -> str:
        # Your subset uses train/ and val/.
        return "train" if self.split == "train" else "val"

    def _subset_sides(self):
        return ["left", "right"] if self.side == "both" else [self.side]

    def _subset_directions(self):
        if self.direction == "both":
            return ["into_future", "into_past"]
        if self.direction == "forward":
            return ["into_future"]
        return ["into_past"]

    def _collect_subset_samples(self) -> None:
        from pathlib import Path

        if self.dstype != "clean":
            raise RuntimeError(
                "FlyingThings3D_subset appears to contain image_clean only. "
                f"Requested dstype={self.dstype!r}. Use dstype: clean."
            )

        split_name = self._subset_split_name()
        split_root = Path(self.root) / split_name
        image_root = split_root / "image_clean"
        flow_root = split_root / "flow"

        if not image_root.is_dir():
            raise RuntimeError(f"Subset image root not found: {image_root}")

        if not flow_root.is_dir():
            raise RuntimeError(f"Subset flow root not found: {flow_root}")

        for side in self._subset_sides():
            image_dir = image_root / side

            if not image_dir.is_dir():
                raise RuntimeError(f"Subset image dir not found: {image_dir}")

            frames = sorted([
                p for p in image_dir.iterdir()
                if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg", ".ppm"}
            ])

            if len(frames) < 2:
                raise RuntimeError(f"Need at least 2 images in {image_dir}, found {len(frames)}")

            for direction in self._subset_directions():
                flow_dir = flow_root / side / direction

                if not flow_dir.is_dir():
                    raise RuntimeError(f"Subset flow dir not found: {flow_dir}")

                flow_files = sorted([
                    p for p in flow_dir.iterdir()
                    if p.is_file() and p.suffix.lower() in {".flo", ".pfm"}
                ])

                if not flow_files:
                    raise RuntimeError(f"No flow files found in {flow_dir}")

                for i, img1 in enumerate(frames):
                    if direction == "into_future":
                        img2_idx = i + 1
                    else:
                        img2_idx = i - 1

                    if img2_idx < 0 or img2_idx >= len(frames):
                        continue

                    img2 = frames[img2_idx]
                    flow_path = self._match_subset_flow_file(
                        img1=img1,
                        flow_dir=flow_dir,
                        flow_files=flow_files,
                        pair_index=i,
                    )

                    if flow_path is None:
                        # Do not append flow=None for supervised training.
                        continue

                    self._samples.append({
                        "image1": str(img1),
                        "image2": str(img2),
                        "flow": str(flow_path),
                    })

        if not self._samples:
            raise RuntimeError(
                "FlyingThings3D subset found images/flow folders but no valid pairs.\n"
                f"root={self.root}\n"
                f"split={split_name}\n"
                f"side={self.side}\n"
                f"direction={self.direction}\n\n"
                "Inspect names with:\n"
                f"  find {image_root} -maxdepth 2 -type f | head -20\n"
                f"  find {flow_root} -maxdepth 4 -type f | head -20"
            )

    def _match_subset_flow_file(self, img1, flow_dir, flow_files, pair_index: int):
        """
        Match a flow file to img1.

        Supports common names like:
            0000000.flo
            000000.flo
            frame_0001.flo
            OpticalFlowIntoFuture_000000_L.pfm

        Falls back to sorted flow file order.
        """
        stem = img1.stem

        candidates = []
        for ext in (".flo", ".pfm"):
            candidates.extend([
                flow_dir / f"{stem}{ext}",
                flow_dir / f"{int(stem):06d}{ext}" if stem.isdigit() else flow_dir / f"{stem}{ext}",
                flow_dir / f"{int(stem):07d}{ext}" if stem.isdigit() else flow_dir / f"{stem}{ext}",
                flow_dir / f"OpticalFlowIntoFuture_{stem}_L{ext}",
                flow_dir / f"OpticalFlowIntoPast_{stem}_L{ext}",
            ])

        for p in candidates:
            if p.is_file():
                return p

        # Fallback: assume sorted nth flow corresponds to nth valid image-pair.
        if pair_index < len(flow_files):
            return flow_files[pair_index]

        return None

    # ------------------------------------------------------------------
    # Full FlyingThings3D layout
    # ------------------------------------------------------------------

    def _collect_full_samples(self) -> None:
        split_name = "TRAIN" if self.split == "train" else "TEST"

        img_base = os.path.join(
            self.root,
            f"frames_{self.dstype}pass",
            split_name,
        )
        flow_base = os.path.join(self.root, "optical_flow", split_name)

        if not os.path.isdir(img_base):
            raise RuntimeError(f"FlyingThings3D img dir not found: {img_base}")

        if not os.path.isdir(flow_base):
            raise RuntimeError(f"FlyingThings3D flow dir not found: {flow_base}")

        sides = ["left", "right"] if self.side == "both" else [self.side]

        if self.direction == "both":
            directions = [("into_future", 1), ("into_past", -1)]
        elif self.direction == "forward":
            directions = [("into_future", 1)]
        else:
            directions = [("into_past", -1)]

        for subset in sorted(os.listdir(img_base)):
            subset_img = os.path.join(img_base, subset)
            subset_flow = os.path.join(flow_base, subset)

            if not os.path.isdir(subset_img):
                continue

            for seq in sorted(os.listdir(subset_img)):
                for side in sides:
                    seq_img = os.path.join(subset_img, seq, side)

                    if not os.path.isdir(seq_img):
                        continue

                    frames = sorted([
                        f for f in os.listdir(seq_img)
                        if f.endswith(".png")
                    ])

                    for direction_name, step in directions:
                        seq_flow = os.path.join(
                            subset_flow,
                            seq,
                            direction_name,
                            side,
                        )

                        if not os.path.isdir(seq_flow):
                            continue

                        for i in range(len(frames)):
                            j = i + step
                            if j < 0 or j >= len(frames):
                                continue

                            fnum = frames[i].replace(".png", "")

                            if direction_name == "into_future":
                                flow_name = f"OpticalFlowIntoFuture_{fnum}_{side[0].upper()}.pfm"
                            else:
                                flow_name = f"OpticalFlowIntoPast_{fnum}_{side[0].upper()}.pfm"

                            flow_path = os.path.join(seq_flow, flow_name)

                            if not os.path.isfile(flow_path):
                                continue

                            self._samples.append({
                                "image1": os.path.join(seq_img, frames[i]),
                                "image2": os.path.join(seq_img, frames[j]),
                                "flow": flow_path,
                            })