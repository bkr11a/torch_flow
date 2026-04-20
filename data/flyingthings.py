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
    FlyingThings3D subset used for optical flow training.

    Directory layout (disparity-less subset variant used by most papers):
        <root>/
            frames_cleanpass/
                TRAIN/ TEST/
                    A/ B/ C/  (subsets)
                        XXXX/
                            left/ XXXX.png
            optical_flow/
                TRAIN/ TEST/
                    A/ B/ C/
                        XXXX/
                            into_future/ left/ OpticalFlowIntoFuture_XXXX_L.pfm
                            into_past/   left/ OpticalFlowIntoPast_XXXX_L.pfm
    """

    def __init__(
        self,
        root: str,
        split: str = "train",
        dstype: str = "clean",
        augmentor: Optional[Callable] = None,
    ) -> None:
        self.dstype = dstype
        super().__init__(root=root, split=split, augmentor=augmentor, sparse=False)

    def _collect_samples(self) -> None:
        split_name = "TRAIN" if self.split == "train" else "TEST"
        img_base   = os.path.join(
            self.root,
            f"frames_{self.dstype}pass",
            split_name,
        )
        flow_base = os.path.join(self.root, "optical_flow", split_name)

        if not os.path.isdir(img_base):
            raise RuntimeError(f"FlyingThings3D img dir not found: {img_base}")

        for subset in sorted(os.listdir(img_base)):
            subset_img  = os.path.join(img_base,  subset)
            subset_flow = os.path.join(flow_base, subset)
            if not os.path.isdir(subset_img):
                continue

            for seq in sorted(os.listdir(subset_img)):
                seq_img  = os.path.join(subset_img,  seq, "left")
                seq_flow = os.path.join(subset_flow, seq, "into_future", "left")
                if not os.path.isdir(seq_img):
                    continue

                frames = sorted([f for f in os.listdir(seq_img) if f.endswith(".png")])
                for i in range(len(frames) - 1):
                    fnum = frames[i].replace(".png", "")
                    flow_name = f"OpticalFlowIntoFuture_{fnum}_L.pfm"
                    flow_path = os.path.join(seq_flow, flow_name)
                    self._samples.append({
                        "image1": os.path.join(seq_img, frames[i]),
                        "image2": os.path.join(seq_img, frames[i + 1]),
                        "flow":   flow_path if os.path.isfile(flow_path) else None,
                    })
