
from omegaconf import OmegaConf
from data import build_dataset
import torch
import math


ROOT = "/mnt/a/benchmark_data/optical_flow"


DATASETS = {
    "chairs": {
        "name": "chairs",
        "root": f"{ROOT}/FlyingChairs_release",
        "crop_size": [368, 496],
        "min_scale": -0.1,
        "max_scale": 1.0,
        "detail_crop_prob": 0.0,
        "dstype": "clean",
    },

    "things_subset_forward_both": {
        "name": "things",
        "root": f"{ROOT}/FlyingThings3D/FlyingThings3D_subset",
        "dstype": "clean",
        "side": "both",
        "direction": "forward",
        "crop_size": [400, 720],
        "min_scale": -0.2,
        "max_scale": 0.8,
        "detail_crop_prob": 0.0,
    },

    "things_subset_forward_left": {
        "name": "things",
        "root": f"{ROOT}/FlyingThings3D/FlyingThings3D_subset",
        "dstype": "clean",
        "side": "left",
        "direction": "forward",
        "crop_size": [400, 720],
        "min_scale": -0.2,
        "max_scale": 0.8,
        "detail_crop_prob": 0.0,
    },

    "sintel_clean": {
        "name": "sintel",
        "root": f"{ROOT}/MPI-Sintel/MPI-Sintel-complete",
        "dstype": "clean",
        "crop_size": [368, 768],
        "min_scale": -0.2,
        "max_scale": 0.6,
        "detail_crop_prob": 0.0,
    },

    "sintel_final": {
        "name": "sintel",
        "root": f"{ROOT}/MPI-Sintel/MPI-Sintel-complete",
        "dstype": "final",
        "crop_size": [368, 768],
        "min_scale": -0.2,
        "max_scale": 0.6,
        "detail_crop_prob": 0.0,
    },

    "sintel_both": {
        "name": "sintel",
        "root": f"{ROOT}/MPI-Sintel/MPI-Sintel-complete",
        "dstype": "both",
        "crop_size": [368, 768],
        "min_scale": -0.2,
        "max_scale": 0.6,
        "detail_crop_prob": 0.0,
    },

    "spring_forward_left": {
        "name": "spring",
        "root": f"{ROOT}/Spring",
        "direction": "forward",
        "side": "left",
        "crop_size": [540, 960],
        "min_scale": -0.3,
        "max_scale": 0.5,
        "detail_crop_prob": 0.0,
    },

    "spring_forward_both": {
        "name": "spring",
        "root": f"{ROOT}/Spring",
        "direction": "forward",
        "side": "both",
        "crop_size": [540, 960],
        "min_scale": -0.3,
        "max_scale": 0.5,
        "detail_crop_prob": 0.0,
    },

    "kitti_2012": {
        "name": "kitti",
        "root": f"{ROOT}/KITTI/KITTI 2012",
        "crop_size": [320, 960],
        "min_scale": -0.2,
        "max_scale": 0.4,
        "detail_crop_prob": 0.0,
    },

    "kitti_2015": {
        "name": "kitti",
        "root": f"{ROOT}/KITTI/KITTI 2015",
        "crop_size": [320, 960],
        "min_scale": -0.2,
        "max_scale": 0.4,
        "detail_crop_prob": 0.0,
    },
}


def tensor_stats(x):
    if not isinstance(x, torch.Tensor):
        return "not tensor"

    x = x.detach()
    finite = torch.isfinite(x)

    if finite.any():
        xf = x[finite]
        return {
            "shape": tuple(x.shape),
            "dtype": str(x.dtype),
            "finite_frac": float(finite.float().mean()),
            "min": float(xf.min()),
            "max": float(xf.max()),
            "mean": float(xf.float().mean()),
        }

    return {
        "shape": tuple(x.shape),
        "dtype": str(x.dtype),
        "finite_frac": 0.0,
    }


def print_sample_paths(ds, n=3):
    if not hasattr(ds, "_samples"):
        print("  no _samples attribute")
        return

    print("  first sample paths:")
    for i in range(min(n, len(ds._samples))):
        print(f"    [{i}] {ds._samples[i]}")


def smoke_one(label, cfg_dict):
    print("\n" + "=" * 90)
    print(label)
    print("=" * 90)

    cfg = OmegaConf.create(cfg_dict)

    for split in ["train", "val"]:
        print(f"\n[{split}]")

        try:
            ds = build_dataset(cfg, split=split)
            print("  samples:", len(ds))
            print_sample_paths(ds, n=2)

            s = ds[0]

            print("  image1:", tensor_stats(s["image1"]))
            print("  image2:", tensor_stats(s["image2"]))
            print("  flow:  ", tensor_stats(s["flow"]))
            print("  valid: ", tensor_stats(s["valid"]))

            valid = s["valid"]
            print("  valid ratio:", float(valid.float().mean()))

            occ = s.get("occlusion", None)
            inv = s.get("invalid", None)
            if occ is None:
                print("  occlusion: None")
            else:
                print("  occlusion:", tensor_stats(occ), "ratio:", float(occ.float().mean()))

            if inv is None:
                print("  invalid: None")
            else:
                print("  invalid:", tensor_stats(inv), "ratio:", float(inv.float().mean()))

        except Exception as e:
            print("  FAILED:", type(e).__name__, str(e))


if __name__ == "__main__":
    for label, cfg in DATASETS.items():
        smoke_one(label, cfg)