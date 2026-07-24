#!/usr/bin/env python3
"""Synthetic construction and forward smoke tests for HQS-LM-OF/SF."""
from __future__ import annotations

import argparse

import torch
from omegaconf import OmegaConf

from models import build_model


def _load(path: str, *, small: bool):
    cfg = OmegaConf.merge(
        OmegaConf.load("configs/default.yaml"),
        OmegaConf.load(path),
    )
    if not small:
        return cfg
    common = {
        "feature_channels": [16, 24, 32, 32],
        "match_channels": [16, 24, 32, 32],
        "context_channels": [16, 16, 24, 24],
        "blocks_per_scale": [1, 1, 1, 1],
        "groups": 8,
        "iterations": [1, 1, 1, 1],
        "jacobi_sweeps": [1, 1, 1, 1],
        "correlation_radii": [1, 1, 1, 1],
        "all_pairs_levels": [1, 2],
        "prior_hidden_channels": 16,
        "reliability_hidden_channels": 8,
        "upsample_hidden_dim": 16,
        "local_corr_channel_chunk": 0,
        "local_corr_checkpoint": False,
    }
    key = (
        "hqs_lm_of"
        if cfg.model.model_type == "hqs_lm_of"
        else "hqs_lm_sf"
    )
    cfg.model[key] = OmegaConf.merge(cfg.model[key], common)
    return cfg


def _inputs(device: torch.device):
    batch, height, width = 1, 32, 48
    image1 = torch.rand(batch, 3, height, width, device=device)
    image2 = torch.rand_like(image1)
    return image1, image2


def optical(device: torch.device, small: bool) -> None:
    cfg = _load("configs/dropins/08_hqs_lm_of.yaml", small=small)
    model = build_model(cfg).to(device).eval()
    image1, image2 = _inputs(device)
    with torch.no_grad():
        output = model(image1, image2)
    final = output["flow_preds"][-1]
    assert final.shape == (1, 2, 32, 48)
    assert torch.isfinite(final).all()
    assert all(
        torch.count_nonzero(value) == 0
        for value in output["learned_data_delta_lows"]
    )
    print(
        "HQS-LM-OF:",
        tuple(final.shape),
        f"stages={len(output['flow_preds'])}",
        f"parameters={model.param_count()['total']:,}",
    )


def scene(device: torch.device, small: bool) -> None:
    cfg = _load(
        "configs/dropins/09_hqs_lm_sf_prototype.yaml",
        small=small,
    )
    model = build_model(cfg).to(device).eval()
    image1, image2 = _inputs(device)
    depth1 = torch.full((1, 1, 32, 48), 10.0, device=device)
    depth2 = depth1.clone()
    intrinsics = torch.tensor(
        [[[50.0, 0.0, 23.5], [0.0, 50.0, 15.5], [0.0, 0.0, 1.0]]],
        device=device,
    )
    transform = torch.eye(4, device=device).unsqueeze(0)
    with torch.no_grad():
        output = model(
            image1,
            image2,
            depth1,
            depth2,
            intrinsics,
            transform,
        )
    final = output["scene_flow_final"]
    induced = output["induced_flow_final"]
    assert final.shape == (1, 3, 32, 48)
    assert induced.shape == (1, 2, 32, 48)
    assert torch.isfinite(final).all()
    assert torch.isfinite(induced).all()
    print(
        "HQS-LM-SF:",
        tuple(final.shape),
        f"stages={len(output['scene_flow_preds'])}",
        f"parameters={model.param_count()['total']:,}",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        choices=("optical", "scene", "both"),
        default="both",
    )
    parser.add_argument(
        "--full-config",
        action="store_true",
        help="Use the complete configured channel and iteration budgets.",
    )
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    device = torch.device(args.device)
    small = not args.full_config
    if args.model in {"optical", "both"}:
        optical(device, small)
    if args.model in {"scene", "both"}:
        scene(device, small)


if __name__ == "__main__":
    main()
