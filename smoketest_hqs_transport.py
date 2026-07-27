#!/usr/bin/env python3
"""Construction and forward smoke tests for HQS-OTOF/HQS-Field-OFv2."""
from __future__ import annotations

import argparse

import torch
from omegaconf import OmegaConf

from models import build_model


OVERLAYS = {
    "hqs_otof": ("configs/dropins/11_hqs_otof.yaml", "hqs_otof"),
    "hqs_field_of_v2": (
        "configs/dropins/12_hqs_field_of_v2.yaml",
        "hqs_field_of_v2",
    ),
}


def _load(model_name: str, *, small: bool):
    path, key = OVERLAYS[model_name]
    cfg = OmegaConf.merge(
        OmegaConf.load("configs/default.yaml"),
        OmegaConf.load(path),
    )
    if not small:
        return cfg, key
    cfg.model[key] = OmegaConf.merge(
        cfg.model[key],
        {
            "feature_channels": [8, 16, 24, 32],
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
            "ot_feature_channels": 16,
            "ot_sinkhorn_iterations": 3,
            "ot_query_chunk_size": 16,
            "ot_num_hypotheses": 2,
            "ot_maximum_tokens": 128,
            "ot_gradient_checkpointing": False,
        },
    )
    if key == "hqs_field_of_v2":
        cfg.model[key].retransport_state_weights = [0.0]
        cfg.model[key].retransport_sinkhorn_iterations = 2
        cfg.model[key].retransport_query_chunk_size = 16
        cfg.model[key].retransport_num_hypotheses = 2
        cfg.model[key].retransport_maximum_tokens = 128
        cfg.model[key].semantic_graph_neighbours = [2, 2]
        cfg.model[key].semantic_graph_maximum_tokens = [64, 128]
        cfg.model[key].semantic_graph_embedding_channels = 8
        cfg.model[key].transport_calibrator_hidden_channels = 8
    cfg.loss.num_stages = 4
    return cfg, key


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        choices=tuple(OVERLAYS),
        default="hqs_otof",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--full-config", action="store_true")
    args = parser.parse_args()

    cfg, key = _load(args.model, small=not args.full_config)
    device = torch.device(args.device)
    model = build_model(cfg).to(device).eval()
    image1 = torch.rand(1, 3, 32, 48, device=device)
    image2 = torch.rand_like(image1)
    with torch.no_grad():
        output = model(image1, image2)
    final = output["flow_preds"][-1]
    expected = sum(int(value) for value in cfg.model[key].iterations)
    assert int(cfg.loss.num_stages) == expected
    assert len(output["flow_preds"]) == expected
    assert len(output["hypothesis_proposal_lows"]) == expected
    assert final.shape == (1, 2, 32, 48)
    assert torch.isfinite(final).all()
    assert torch.isfinite(output["ot_observability"]).all()
    print(
        args.model,
        tuple(final.shape),
        f"stages={expected}",
        f"parameters={model.param_count()['total']:,}",
    )


if __name__ == "__main__":
    main()
