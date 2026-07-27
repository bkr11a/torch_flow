#!/usr/bin/env python3
"""Construction and forward smoke test for HQS-Field-OF."""
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
    cfg.model.hqs_field_of = OmegaConf.merge(
        cfg.model.hqs_field_of,
        {
            "feature_channels": [16, 24, 32, 32],
            "match_channels": [16, 24, 32, 32],
            "context_channels": [16, 16, 24, 24],
            "blocks_per_scale": [1, 1, 1, 1],
            "groups": 8,
            "iterations": [1, 1, 1, 1],
            "graph_sweeps": [1, 1, 1, 1],
            "correlation_radii": [1, 1, 1, 1],
            "all_pairs_levels": [1, 2],
            "num_hypotheses": [2, 2, 2, 2],
            "proposal_embedding_channels": [8, 8, 8, 8],
            "proposal_hidden_channels": [8, 8, 8, 8],
            "correlation_attention_channels": [8, 8, 8, 8],
            "correlation_attention_heads": [2, 2, 2, 2],
            "graph_embedding_channels": [8, 8, 8, 8],
            "graph_dilations": [[1], [1], [1], [1]],
            "feature_transformer_depth": [1, 0, 0, 0],
            "feature_transformer_heads": [4, 4, 4, 4],
            "feature_transformer_max_tokens": [64, 64, 64, 64],
            "local_corr_channel_chunk": 0,
            "local_corr_checkpoint": False,
            "cycle_scales": [16],
            "upsample_hidden_dim": 16,
        },
    )
    return cfg


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/dropins/10_hqs_field_of.yaml",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--full-config", action="store_true")
    args = parser.parse_args()

    device = torch.device(args.device)
    cfg = _load(args.config, small=not args.full_config)
    model = build_model(cfg).to(device).eval()
    image1 = torch.rand(1, 3, 32, 48, device=device)
    image2 = torch.rand_like(image1)
    with torch.no_grad():
        output = model(image1, image2)
    final = output["flow_preds"][-1]
    assert final.shape == (1, 2, 32, 48)
    assert torch.isfinite(final).all()
    assert output["solver"] == "hqs_field_of"
    assert len(output["flow_preds"]) == 4
    assert len(output["hypothesis_proposal_lows"]) == 4
    assert all(
        torch.count_nonzero(value) == 0
        for value in output["learned_data_delta_lows"]
    )
    print(
        "HQS-Field-OF:",
        tuple(final.shape),
        f"stages={len(output['flow_preds'])}",
        f"parameters={model.param_count()['total']:,}",
    )


if __name__ == "__main__":
    main()
