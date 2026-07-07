from omegaconf import OmegaConf
from data import build_dataset, build_dataloader

cfg = OmegaConf.create({
    "mode": "weighted_mixed",
    "epoch_size": 64,
    "deterministic": False,
    "seed": 123,
    "include_dataset_id": True,
    "batch_size": 2,
    "num_workers": 0,
    "crop_size": [320, 768],
    "min_scale": -0.2,
    "max_scale": 0.4,
    "detail_crop_prob": 0.0,
    "datasets": [
        {"label": "things_subset", "name": "things", "root": "/mnt/a/benchmark_data/optical_flow/FlyingThings3D/FlyingThings3D_subset", "dstype": "clean", "side": "both", "direction": "forward", "weight": 0.40},
        {"label": "sintel_final", "name": "sintel", "root": "/mnt/a/benchmark_data/optical_flow/MPI-Sintel/MPI-Sintel-complete", "dstype": "final", "weight": 0.25},
        {"label": "spring_left", "name": "spring", "root": "/mnt/a/benchmark_data/optical_flow/Spring/spring", "direction": "forward", "side": "left", "val_fraction": 0.10, "weight": 0.20},
        {"label": "hd1k", "name": "hd1k", "root": "/mnt/a/benchmark_data/optical_flow/HD1K/hd1k_full_package", "camera": "image_2", "val_fraction": 0.10, "weight": 0.15},
    ],
})

ds = build_dataset(cfg, split="train")
print(ds)
loader = build_dataloader(ds, cfg, split="train")
batch = next(iter(loader))
print("image1:", batch["image1"].shape)
print("image2:", batch["image2"].shape)
print("flow:  ", batch["flow"].shape)
print("valid: ", batch["valid"].shape)
print("dataset_id:", batch.get("dataset_id"))
print("valid ratio:", float(batch["valid"].float().mean()))
