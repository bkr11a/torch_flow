#!/usr/bin/env python3
"""Construction/capacity audit for OF-A and OF-B ablation variants."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from omegaconf import OmegaConf


def count_trainable(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_total(model):
    return sum(p.numel() for p in model.parameters())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default=".")
    args = parser.parse_args()
    repo = Path(args.repo_root).resolve()
    sys.path.insert(0, str(repo))

    from models import build_model

    base = OmegaConf.load(repo / "configs/default.yaml")
    variants = {
        "A0": "configs/dropins/16_hqs_core_pretrained_gmflow.yaml",
        "A1": "configs/dropins/17_hqs_core_recurrent_control.yaml",
        "A2": "configs/dropins/18_hqs_core_recurrent_control_physics_loss.yaml",
        "A3": "configs/dropins/19_hqs_core_augmented_physics_loss.yaml",
        "A4": "configs/dropins/20_hqs_core_single_state.yaml",
        "B1": "configs/dropins/21_hqs_core_B1_analytic_data_only.yaml",
        "B2": "configs/dropins/22_hqs_core_B2_learned_data_only.yaml",
        "B3": "configs/dropins/23_hqs_core_B3_analytic_proximal_only.yaml",
        "B4": "configs/dropins/24_hqs_core_B4_learned_proximal_only.yaml",
    }

    models = {}
    for key, rel in variants.items():
        cfg = OmegaConf.merge(base, OmegaConf.load(repo / rel))
        model = build_model(cfg)
        models[key] = model
        print(
            f"{key}: total={count_total(model):,} "
            f"trainable={count_trainable(model):,} "
            f"type={type(model).__name__}"
        )
        report = getattr(model, "capacity_match_report", None)
        if report is not None:
            print(f"  capacity_match_report={report}")
        report = getattr(model, "operator_ablation_report", None)
        if report is not None:
            print(f"  operator_ablation_report={report}")

    # Architecture-only controls.
    assert count_trainable(models["A2"]) == count_trainable(models["A1"]), (
        "A2 must be architecture-identical to A1"
    )
    assert count_trainable(models["A3"]) == count_trainable(models["A0"]), (
        "A3 must be architecture-identical to A0"
    )

    # OF-B nominal parameterisation is intentionally identical to A0.
    for key in ("B1", "B2", "B3", "B4"):
        assert count_trainable(models[key]) == count_trainable(models["A0"]), (
            f"{key} nominal trainable count differs from A0"
        )
        assert count_total(models[key]) == count_total(models["A0"]), (
            f"{key} total count differs from A0"
        )

    # A4 is active-capacity matched per cell, with an overall 1% guard.
    a0 = count_trainable(models["A0"])
    a4 = count_trainable(models["A4"])
    rel = abs(a4 - a0) / float(a0)
    print(f"A4/A0 trainable mismatch={100.0 * rel:.5f}%")
    assert rel <= 0.01, "A4 exceeds the 1% trainable-capacity tolerance"

    # Eq. (65) objective weights.
    for key in ("A2", "A3"):
        cfg = OmegaConf.merge(
            base,
            OmegaConf.load(repo / variants[key]),
        )
        assert float(cfg.loss.photo_weight) == 0.20
        assert float(cfg.loss.smooth_weight) == 0.20
        assert float(cfg.loss.ofce_weight) == 0.20

    print("PASS: OF-A/OF-B construction and capacity invariants hold.")


if __name__ == "__main__":
    main()
