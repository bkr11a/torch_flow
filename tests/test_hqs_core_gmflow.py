from __future__ import annotations

import math
from pathlib import Path

import torch
from omegaconf import OmegaConf

from hqs_pytorch.customML.customModels.HQSCore import HQSCore
from models.hqs_core_components import AllPairsCorrelation
from models.hqs_gmflow_components import (
    GMFlowMatchingFrontEnd,
    GMFlowResidualBlock,
    gmflow_forward_backward_consistency,
)


def _local_key_to_official_gmflow(key: str) -> str:
    if key.startswith("backbone."):
        suffix = key[len("backbone.") :]
        prefixes = {
            "stem.0.": "conv1.",
            "stage2.": "layer1.",
            "stage4.": "layer2.",
            "stage8.": "layer3.",
            "projection.": "conv2.",
        }
        for local, official in prefixes.items():
            if suffix.startswith(local):
                return "backbone." + official + suffix[len(local) :]
    if key.startswith("transformer.blocks."):
        suffix = key[len("transformer.blocks.") :]
        suffix = suffix.replace(".self_attention.", ".self_attn.")
        suffix = suffix.replace(
            ".cross_attention.",
            ".cross_attn_ffn.",
        )
        suffix = suffix.replace(".query.", ".q_proj.")
        suffix = suffix.replace(".key.", ".k_proj.")
        suffix = suffix.replace(".value.", ".v_proj.")
        suffix = suffix.replace(".ffn.", ".mlp.")
        return "transformer.layers." + suffix
    if key.startswith("flow_propagation."):
        suffix = key[len("flow_propagation.") :]
        suffix = suffix.replace("query.", "q_proj.")
        suffix = suffix.replace("key.", "k_proj.")
        return "feature_flow_attn." + suffix
    raise AssertionError(f"No official GMFlow mapping for {key}")


def _write_synthetic_official_checkpoint(
    matcher: GMFlowMatchingFrontEnd,
    path: Path,
    *,
    omit_local_key: str | None = None,
) -> dict[str, torch.Tensor]:
    official = {}
    expected = {}
    torch.manual_seed(2608)
    for local_key, value in matcher.state_dict().items():
        if local_key == omit_local_key:
            continue
        replacement = torch.randn_like(value)
        expected[local_key] = replacement
        official[_local_key_to_official_gmflow(local_key)] = replacement
    # The official full model contains this unrelated head. The importer must
    # ignore it deliberately rather than treating it as a partial-load error.
    official["upsampler.0.weight"] = torch.randn(1)
    torch.save({"model": official}, path)
    return expected


def _tiny_gmflow_core_config():
    return {
        "feature_channels": [16, 24, 32, 32],
        "match_channels": [16, 24, 32, 32],
        "context_channels": [16, 16, 24, 24],
        "blocks_per_scale": [0, 0, 0, 0],
        "groups": 4,
        "matching_pyramid": "gmflow",
        "gmflow_feature_channels": 32,
        "gmflow_transformer_depth": 1,
        "gmflow_ffn_expansion": 2,
        "gmflow_attention_splits": 2,
        "gmflow_gradient_checkpointing": False,
        "gmflow_propagation_query_chunk_size": 8,
        "iterations": [1, 1, 1, 1],
        "jacobi_sweeps": [1, 1, 1, 1],
        "correlation_radii": [1, 1, 1, 1],
        "all_pairs_levels": [1, 2],
        "correlation_embedding_dim": 8,
        "coarse_hidden_dim": 12,
        "fine_hidden_dim": 8,
        "prior_hidden_channels": 8,
        "validity_hidden_dim": 8,
        "upsample_hidden_dim": 8,
        "global_match_scale": 8,
        "global_match_transition_mode": "native_residual",
        "global_decoder": "soft_expectation",
        "global_correlation_mode": "sqrt_dim",
        "global_temperature": 1.0,
        "global_query_chunk_size": 8,
        "global_confidence_gated": False,
        "global_bidirectional": True,
        "global_use_flow_propagation": True,
        "global_propagation_routing": "fb_blend",
        "global_fb_data_gate": True,
        "global_fb_data_gate_scales": [16, 8],
        "global_fb_gate_source": "raw",
    }


def test_gmflow_overlay_selects_exact_matching_path():
    root = Path(__file__).resolve().parents[1]
    cfg = OmegaConf.merge(
        OmegaConf.load(root / "configs/default.yaml"),
        OmegaConf.load(root / "configs/dropins/15_hqs_core_gmflow.yaml"),
    )
    core = cfg.model.hqs_core
    assert cfg.model.model_type == "hqs_core"
    assert core.matching_pyramid == "gmflow"
    assert core.gmflow_transformer_depth == 6
    assert core.gmflow_attention_splits == 2
    assert core.global_match_scale == 8
    assert core.global_correlation_mode == "sqrt_dim"
    assert core.global_temperature == 1.0
    assert core.global_decoder == "soft_expectation"
    assert core.global_bidirectional is True
    assert core.global_fb_data_gate is True
    assert list(core.iterations) == [2, 4, 2, 2]
    assert cfg.loss.num_stages == 10
    assert cfg.loss.global_init_weight == 1.0
    assert cfg.loss.global_propagated_weight == 1.0


def test_pretrained_gmflow_overlay_uses_frozen_raw_routing():
    root = Path(__file__).resolve().parents[1]
    cfg = OmegaConf.merge(
        OmegaConf.load(root / "configs/default.yaml"),
        OmegaConf.load(
            root / "configs/dropins/16_hqs_core_pretrained_gmflow.yaml"
        ),
    )
    core = cfg.model.hqs_core
    assert core.gmflow_released_weights_compatible is True
    assert core.gmflow_pretrained_strict is True
    assert core.gmflow_freeze_pretrained is True
    assert core.global_propagation_routing == "raw"
    assert core.global_use_flow_propagation is False
    assert core.global_fb_data_gate is False
    assert cfg.loss.global_init_weight == 0.0
    assert cfg.loss.global_fb_visibility_weight == 0.0
    assert cfg.training.global_matcher_warmup_steps == 0
    assert cfg.training.amp is False


def test_released_backbone_graph_has_second_pre_addition_relu():
    torch.manual_seed(19)
    block = GMFlowResidualBlock(
        4,
        4,
        stride=1,
        released_weights_compatible=True,
    )
    value = torch.randn(2, 4, 5, 7)
    first = torch.relu(block.norm1(block.conv1(value)))
    second = torch.relu(block.norm2(block.conv2(first)))
    expected = torch.relu(value + second)
    assert torch.allclose(block(value), expected, atol=1e-6)


def test_official_checkpoint_import_is_complete_and_value_preserving(tmp_path):
    source = GMFlowMatchingFrontEnd(
        channels=32,
        transformer_depth=1,
        ffn_expansion=2,
        attention_splits=2,
        gradient_checkpointing=False,
        propagation_query_chunk_size=8,
        released_weights_compatible=True,
    )
    checkpoint = tmp_path / "gmflow_official.pth"
    expected = _write_synthetic_official_checkpoint(source, checkpoint)
    target = GMFlowMatchingFrontEnd(
        channels=32,
        transformer_depth=1,
        ffn_expansion=2,
        attention_splits=2,
        gradient_checkpointing=False,
        propagation_query_chunk_size=8,
        released_weights_compatible=True,
    )
    report = target.load_official_checkpoint(checkpoint, strict=True)
    assert report.loaded_tensor_count == len(expected)
    assert report.loaded_parameter_count == sum(
        value.numel() for value in expected.values()
    )
    assert report.ignored_checkpoint_keys == ("upsampler.0.weight",)
    for key, value in target.state_dict().items():
        assert torch.equal(value, expected[key]), key


def test_official_checkpoint_import_rejects_missing_frontend_tensor(tmp_path):
    matcher = GMFlowMatchingFrontEnd(
        channels=32,
        transformer_depth=1,
        ffn_expansion=2,
        attention_splits=2,
        gradient_checkpointing=False,
        propagation_query_chunk_size=8,
        released_weights_compatible=True,
    )
    checkpoint = tmp_path / "gmflow_incomplete.pth"
    omitted = "transformer.blocks.0.self_attention.query.weight"
    _write_synthetic_official_checkpoint(
        matcher,
        checkpoint,
        omit_local_key=omitted,
    )
    target = GMFlowMatchingFrontEnd(
        channels=32,
        transformer_depth=1,
        ffn_expansion=2,
        attention_splits=2,
        gradient_checkpointing=False,
        propagation_query_chunk_size=8,
        released_weights_compatible=True,
    )
    try:
        target.load_official_checkpoint(checkpoint, strict=True)
    except RuntimeError as error:
        assert omitted in str(error)
    else:
        raise AssertionError("Strict import accepted an incomplete checkpoint")


def test_hqscore_loads_and_permanently_freezes_official_frontend(tmp_path):
    source = GMFlowMatchingFrontEnd(
        channels=32,
        transformer_depth=1,
        ffn_expansion=2,
        attention_splits=2,
        gradient_checkpointing=False,
        propagation_query_chunk_size=8,
        released_weights_compatible=True,
    )
    checkpoint = tmp_path / "gmflow_tiny.pth"
    _write_synthetic_official_checkpoint(source, checkpoint)
    config = _tiny_gmflow_core_config()
    config.update(
        {
            "gmflow_released_weights_compatible": True,
            "gmflow_pretrained_checkpoint": str(checkpoint),
            "gmflow_pretrained_strict": True,
            "gmflow_freeze_pretrained": True,
            "global_use_flow_propagation": False,
            "global_propagation_routing": "raw",
            "global_fb_data_gate": False,
        }
    )
    model = HQSCore({"hqs_core": config})
    assert model.gmflow_pretrained_report is not None
    assert model.permanently_frozen_parameter_prefixes == (
        "gmflow_matcher.",
    )
    assert all(
        not parameter.requires_grad
        for parameter in model.gmflow_matcher.parameters()
    )


def test_scaled_dot_all_pairs_matches_gmflow_equation():
    torch.manual_seed(3)
    source = torch.randn(2, 8, 3, 4)
    target = torch.randn(2, 8, 3, 4)
    correlation = AllPairsCorrelation(
        source,
        target,
        num_levels=1,
        radius=1,
        correlation_mode="sqrt_dim",
    )
    expected = torch.bmm(
        source.float().flatten(2).transpose(1, 2),
        target.float().flatten(2),
    ) / math.sqrt(source.shape[1])
    assert torch.allclose(correlation.base_correlation, expected, atol=1e-6)


def test_bidirectional_soft_match_reuses_transposed_volume():
    torch.manual_seed(7)
    source = torch.randn(1, 8, 2, 3)
    target = torch.randn(1, 8, 2, 3)
    forward_volume = AllPairsCorrelation(
        source,
        target,
        num_levels=1,
        radius=1,
        correlation_mode="sqrt_dim",
    )
    reversed_volume = AllPairsCorrelation(
        target,
        source,
        num_levels=1,
        radius=1,
        correlation_mode="sqrt_dim",
    )
    reverse_from_transpose = forward_volume.global_soft_match(
        temperature=1.0,
        query_chunk_size=2,
        reverse=True,
    )["flow_xy"]
    reverse_from_rebuild = reversed_volume.global_soft_match(
        temperature=1.0,
        query_chunk_size=2,
    )["flow_xy"]
    assert torch.allclose(
        reverse_from_transpose,
        reverse_from_rebuild,
        atol=1e-6,
    )


def test_gmflow_pair_transform_is_swap_symmetric_and_differentiable():
    torch.manual_seed(11)
    matcher = GMFlowMatchingFrontEnd(
        channels=32,
        transformer_depth=1,
        ffn_expansion=2,
        attention_splits=2,
        gradient_checkpointing=False,
        propagation_query_chunk_size=8,
    )
    source = torch.rand(1, 3, 32, 48, requires_grad=True)
    target = torch.rand(1, 3, 32, 48, requires_grad=True)
    forward_source, forward_target = matcher(source, target)
    reverse_source, reverse_target = matcher(target, source)
    assert forward_source.shape == (1, 32, 4, 6)
    assert torch.allclose(forward_source, reverse_target, atol=1e-5)
    assert torch.allclose(forward_target, reverse_source, atol=1e-5)
    loss = forward_source.square().mean() + forward_target.square().mean()
    loss.backward()
    gradients = [
        parameter.grad
        for parameter in matcher.parameters()
        if parameter.requires_grad and parameter.grad is not None
    ]
    assert gradients
    assert all(torch.isfinite(gradient).all() for gradient in gradients)


def test_forward_backward_consistency_detects_out_of_bounds_occlusion():
    forward = torch.zeros(1, 2, 5, 7)
    backward = torch.zeros_like(forward)
    forward[:, 0] = 1.0
    backward[:, 0] = -1.0
    geometry = gmflow_forward_backward_consistency(
        forward,
        backward,
        feature_scale=1,
        alpha=0.01,
        beta_full_resolution=0.5,
        softness_full_resolution=0.05,
    )
    assert geometry["reliability"][..., :-1].mean() > 0.99
    assert torch.count_nonzero(geometry["occlusion"][..., :-1]) == 0
    assert torch.all(geometry["occlusion"][..., -1] == 1)


def test_hqscore_gmflow_forward_contract_and_operator_gate():
    torch.manual_seed(13)
    model = HQSCore({"hqs_core": _tiny_gmflow_core_config()})
    source = torch.rand(1, 3, 32, 48)
    target = torch.rand(1, 3, 32, 48)
    output = model(source, target)
    assert output["prediction_scales"] == [16, 8, 4, 2]
    assert len(output["flow_preds"]) == 4
    assert output["matching_pyramid"] == "gmflow"
    assert output["global_correlation_mode"] == "sqrt_dim"
    assert output["global_bidirectional"] is True
    assert output["global_fb_reliability"].shape == (1, 1, 4, 6)
    assert output["global_fb_occlusion"].shape == (1, 1, 4, 6)
    assert output["gmflow_propagated_flow_yx"].shape == (1, 2, 4, 6)
    assert len(output["global_fb_measurement_reliability_lows"]) == 4
    assert all(torch.isfinite(flow).all() for flow in output["flow_preds"])
    assert all(
        ((gate >= 0.0) & (gate <= 1.0)).all()
        for gate in output["global_fb_measurement_reliability_lows"]
    )
    output["flow_preds"][-1].abs().mean().backward()
    matcher_gradients = [
        parameter.grad
        for parameter in model.gmflow_matcher.parameters()
        if parameter.requires_grad and parameter.grad is not None
    ]
    assert matcher_gradients
    assert all(torch.isfinite(gradient).all() for gradient in matcher_gradients)
