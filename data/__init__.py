"""data/__init__.py – dataset registry and factory."""
from __future__ import annotations

from torch.utils.data import ConcatDataset, DataLoader
from typing import Optional

from .base_dataset import FlowDataset
from .sintel import SintelDataset
from .spring import SpringDataset
from .kitti import KITTIDataset
from .hd1k import HD1KDataset
from .flyingthings import FlyingChairsDataset, FlyingThingsDataset
from .augmentation import FlowAugmentor, SparseFlowAugmentor

__all__ = [
    "FlowDataset",
    "SintelDataset", "SpringDataset", "KITTIDataset", "HD1KDataset",
    "FlyingChairsDataset", "FlyingThingsDataset",
    "FlowAugmentor", "SparseFlowAugmentor",
    "build_dataset", "build_dataloader",
]


def build_dataset(cfg, split: str = "train"):
    """
    Build a dataset (or ConcatDataset for mixed training) from config.

    Expected cfg fields:
        name  : str or list[str]  – dataset name(s)
        root  : str or list[str]  – dataset root(s)
        dstype: str               – "clean" | "final" | "both"  (Sintel/Things)
        crop_size: [H, W]         – augmentation crop size
        min_scale, max_scale: float

    When *name* is a list, datasets are concatenated with equal weighting.
    """
    augmentor = None
    if split == "train" and hasattr(cfg, "crop_size"):
        crop = tuple(cfg.crop_size)  # (H, W)
        augmentor = FlowAugmentor(
            crop_size=crop,
            min_scale=cfg.get("min_scale", -0.2),
            max_scale=cfg.get("max_scale",  0.5),
            detail_crop_prob=cfg.get("detail_crop_prob", 0.0),
        )

    names = cfg.name if isinstance(cfg.name, list) else [cfg.name]
    roots  = cfg.root if isinstance(cfg.root, list) else [cfg.root] * len(names)

    datasets = []
    for name, root in zip(names, roots):
        datasets.append(_build_single(name, root, split, cfg, augmentor))

    if len(datasets) == 1:
        return datasets[0]
    return ConcatDataset(datasets)


def _build_single(name: str, root: str, split: str, cfg, augmentor):
    name = name.lower()
    if name == "sintel":
        return SintelDataset(
            root=root, split=split,
            dstype=cfg.get("dstype", "clean"),
            augmentor=augmentor,
        )
    if name == "spring":
        return SpringDataset(
            root=root,
            split=split,
            direction=cfg.get("direction", "forward"),
            side=cfg.get("side", "left"),
            val_scenes=cfg.get("val_scenes", None),
            val_fraction=cfg.get("val_fraction", 0.10),
            augmentor=augmentor,
        )
    if name in ("kitti", "kitti12", "kitti2012", "kitti15", "kitti2015"):
        sparse_aug = SparseFlowAugmentor(crop_size=tuple(cfg.crop_size)) \
                    if split == "train" and hasattr(cfg, "crop_size") else None
        return KITTIDataset(
            root=root,
            split=split,
            augmentor=sparse_aug,
            val_size=cfg.get("val_size", 40),
            camera=cfg.get("camera", None),
            flow_type=cfg.get("flow_type", "occ"),
        )
    if name == "chairs":
        return FlyingChairsDataset(root=root, split=split, augmentor=augmentor)
    if name == "things":
        return FlyingThingsDataset(
            root=root,
            split=split,
            dstype=cfg.get("dstype", "clean"),
            side=cfg.get("side", "left"),              # "left" | "right" | "both"
            direction=cfg.get("direction", "forward"), # "forward" | "backward" | "both"
            augmentor=augmentor,
        )
    if name in ("hd1k", "h1dk"):
        sparse_aug = SparseFlowAugmentor(crop_size=tuple(cfg.crop_size)) \
                    if split == "train" and hasattr(cfg, "crop_size") else None
        return HD1KDataset(
            root=root,
            split=split,
            camera=cfg.get("camera", "image_2"),
            val_fraction=cfg.get("val_fraction", 0.10),
            val_num_sequences=cfg.get("val_num_sequences", None),
            augmentor=sparse_aug,
        )
    raise ValueError(f"Unknown dataset: {name!r}")


def build_dataloader(dataset, cfg, split: str = "train") -> DataLoader:
    from .base_dataset import flow_collate_fn
    shuffle = (split == "train")
    return DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=shuffle,
        num_workers=cfg.get("num_workers", 4),
        pin_memory=True,
        drop_last=(split == "train"),
        persistent_workers=cfg.get("num_workers", 4) > 0,
        collate_fn=flow_collate_fn,
    )
