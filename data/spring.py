"""Spring optical flow dataset.

Directory layout:
    <root>/
        train/
            <scene>/
                frame_left/  XXXX.png
                flow_FW_left/ XXXX.flo  (or .pfm for full resolution)
                flow_BW_left/
        test/
            <scene>/
                frame_left/  XXXX.png

Reference: Mehl et al., "Spring: A High-Resolution High-Detail Dataset and
           Benchmark for Scene Flow, Optical Flow and Stereo", CVPR 2023.
           https://spring-benchmark.org
"""
from __future__ import annotations

import os
from typing import Callable, List, Optional

from .base_dataset import FlowDataset


class SpringDataset(FlowDataset):
    """
    Args:
        root:       Path to the Spring dataset root.
        split:      "train" | "val" | "test".
        direction:  "forward" | "backward" | "both".
        side:       "left" | "right" | "both".
        augmentor:  Optional augmentation callable.
        val_scenes: Scene names held out for validation.
    """

    DEFAULT_VAL_SCENES = ["00041", "00044"]   # ~10 % of training sequences

    def __init__(
        self,
        root: str,
        split: str = "train",
        direction: str = "forward",
        side: str = "left",
        augmentor: Optional[Callable] = None,
        val_scenes: Optional[List[str]] = None,
    ) -> None:
        self.direction  = direction
        self.side       = side
        self.val_scenes = val_scenes or self.DEFAULT_VAL_SCENES
        super().__init__(root=root, split=split, augmentor=augmentor, sparse=False)

    def _collect_samples(self) -> None:
        split_dir = os.path.join(self.root, "train" if self.split != "test" else "test")
        if not os.path.isdir(split_dir):
            raise RuntimeError(f"Spring split dir not found: {split_dir}")

        sides      = ["left", "right"] if self.side == "both" else [self.side]
        directions = ["FW", "BW"]      if self.direction == "both" else (
                     ["FW"]            if self.direction == "forward" else ["BW"])

        for scene in sorted(os.listdir(split_dir)):
            scene_path = os.path.join(split_dir, scene)
            if not os.path.isdir(scene_path):
                continue
            is_val = scene in self.val_scenes
            if self.split == "val"   and not is_val:
                continue
            if self.split == "train" and is_val:
                continue

            for side in sides:
                img_dir = os.path.join(scene_path, f"frame_{side}")
                if not os.path.isdir(img_dir):
                    continue
                frames = sorted([f for f in os.listdir(img_dir) if f.endswith(".png")])

                for direction in directions:
                    flow_dir = os.path.join(scene_path, f"flow_{direction}_{side}")
                    for i in range(len(frames) - 1):
                        img1_idx = i
                        img2_idx = i + 1 if direction == "FW" else i - 1
                        if img2_idx < 0 or img2_idx >= len(frames):
                            continue

                        flow_name = frames[i].replace(".png", ".flo")
                        flow_path = os.path.join(flow_dir, flow_name)
                        # Spring can also use .pfm
                        if not os.path.isfile(flow_path):
                            flow_path = flow_path.replace(".flo", ".pfm")

                        self._samples.append({
                            "image1": os.path.join(img_dir, frames[img1_idx]),
                            "image2": os.path.join(img_dir, frames[img2_idx]),
                            "flow":   flow_path if (
                                os.path.isfile(flow_path) and self.split != "test"
                            ) else None,
                        })
