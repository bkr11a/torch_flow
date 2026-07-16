"""MPI Sintel optical flow dataset.

Directory layout expected:
    <root>/
        training/
            clean/        <scene>/frame_XXXX.png
            final/        <scene>/frame_XXXX.png
            flow/         <scene>/frame_XXXX.flo
            occlusions/   <scene>/frame_XXXX.png  (0=visible, 255=occluded)
            invalid/      <scene>/frame_XXXX.png  (0=valid, 255=invalid)
        test/
            clean/   <scene>/frame_XXXX.png
            final/   <scene>/frame_XXXX.png

Occlusion and invalid masks
---------------------------
  occlusions/  – pixels in frame1 whose corresponding point is not visible
                 in frame2 (occluded or out of frame). 255 = occluded.
  invalid/     – pixels where the GT flow is unreliable for any reason
                 (motion blur, reflections, etc.). 255 = invalid.

Only ``invalid`` restricts GT-flow supervision.  ``occlusions`` remains a
separate visibility label: Sintel still provides a valid flow vector there,
which is required to train and evaluate the source-conditioned prior.

Reference: Butler et al., "A naturalistic open source movie for optical flow
           evaluation", ECCV 2012.
"""
from __future__ import annotations

import os
from typing import Callable, List, Optional

from .base_dataset import FlowDataset


class SintelDataset(FlowDataset):
    """
    Args:
        root:     Path to the Sintel dataset root.
        split:    "train" | "val" | "test".
        dstype:   "clean" | "final" | "both" (use both passes during training).
        augmentor: Optional augmentation callable.
        val_scenes: Scene names held out for validation (used only if split="val").
    """

    # Canonical Sintel validation set (used in most papers)
    DEFAULT_VAL_SCENES = ["ambush_4", "ambush_6", "bamboo_2", "market_2",
                          "shaman_3", "sleeping_1", "temple_2"]

    def __init__(
        self,
        root: str,
        split: str = "train",
        dstype: str = "clean",
        augmentor: Optional[Callable] = None,
        val_scenes: Optional[List[str]] = None,
        use_occlusions: bool = True,
        use_invalid: bool = True,
    ) -> None:
        self.dstype          = dstype
        self.val_scenes      = val_scenes or self.DEFAULT_VAL_SCENES
        self.use_occlusions  = use_occlusions
        self.use_invalid     = use_invalid
        super().__init__(root=root, split=split, augmentor=augmentor, sparse=False)

    def _collect_samples(self) -> None:
        if self.split == "test":
            self._collect_test()
        else:
            self._collect_train_val()

    def _collect_train_val(self) -> None:
        passes = ["clean", "final"] if self.dstype == "both" else [self.dstype]
        for ds in passes:
            img_dir  = os.path.join(self.root, "training", ds)
            flow_dir = os.path.join(self.root, "training", "flow")

            if not os.path.isdir(img_dir):
                raise RuntimeError(f"Sintel image dir not found: {img_dir}")

            for scene in sorted(os.listdir(img_dir)):
                scene_img  = os.path.join(img_dir,  scene)
                scene_flow = os.path.join(flow_dir, scene)
                if not os.path.isdir(scene_img):
                    continue

                is_val_scene = scene in self.val_scenes
                if self.split == "val" and not is_val_scene:
                    continue
                if self.split == "train" and is_val_scene:
                    continue

                frames = sorted([
                    f for f in os.listdir(scene_img) if f.endswith(".png")
                ])
                occ_dir = os.path.join(
                    self.root, "training", "occlusions", scene
                )
                inv_dir = os.path.join(
                    self.root, "training", "invalid", scene
                )

                for i in range(len(frames) - 1):
                    frame_name = frames[i]
                    occ_path = os.path.join(occ_dir, frame_name)
                    inv_path = os.path.join(inv_dir, frame_name)
                    self._samples.append({
                        "image1":    os.path.join(scene_img,  frame_name),
                        "image2":    os.path.join(scene_img,  frames[i + 1]),
                        "flow":      os.path.join(scene_flow,
                                                  frame_name.replace(".png", ".flo")),
                        "occlusion": occ_path if (
                            self.use_occlusions and os.path.isfile(occ_path)
                        ) else None,
                        "invalid":   inv_path if (
                            self.use_invalid and os.path.isfile(inv_path)
                        ) else None,
                    })

    def _collect_test(self) -> None:
        passes = ["clean", "final"] if self.dstype == "both" else [self.dstype]
        for ds in passes:
            img_dir = os.path.join(self.root, "test", ds)
            if not os.path.isdir(img_dir):
                raise RuntimeError(f"Sintel test dir not found: {img_dir}")

            for scene in sorted(os.listdir(img_dir)):
                scene_path = os.path.join(img_dir, scene)
                if not os.path.isdir(scene_path):
                    continue
                frames = sorted([
                    f for f in os.listdir(scene_path) if f.endswith(".png")
                ])
                for i in range(len(frames) - 1):
                    self._samples.append({
                        "image1":    os.path.join(scene_path, frames[i]),
                        "image2":    os.path.join(scene_path, frames[i + 1]),
                        "flow":      None,
                        "occlusion": None,
                        "invalid":   None,
                    })
