from __future__ import annotations

import inspect
import math
from pathlib import Path

import torch
from omegaconf import OmegaConf

from models import build_model
from models.hqs_lm_components import LMState
from models.hqs_ot_components import (
    BlockwiseDustbinSinkhorn,
    SourceSemanticGraphProximal,
    TransportFieldCell,
    resize_transport_measurement,
    solve_transport_hqs_increment,
)


ROOT = Path(__file__).resolve().parents[1]


def _load(name: str):
    return OmegaConf.merge(
        OmegaConf.load(ROOT / "configs/default.yaml"),
        OmegaConf.load(ROOT / f"configs/dropins/{name}.yaml"),
    )


def _small(cfg, key: str):
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
    return cfg


def test_hqs_otof_overlay_is_static_quarter_resolution_transport():
    cfg = _load("11_hqs_otof")
    assert cfg.model.model_type == "hqs_otof"
    assert cfg.model.hqs_otof.ot_scale == 4
    assert list(cfg.model.hqs_otof.iterations) == [2, 4, 2, 2]
    assert cfg.loss.num_stages == 10
    assert cfg.loss.ot_observability_weight > 0
    assert cfg.training.freeze_backbone_during_global_warmup is True


def test_hqs_field_v2_overlay_retransports_without_initial_state_bias():
    cfg = _load("12_hqs_field_of_v2")
    assert cfg.model.model_type == "hqs_field_of_v2"
    assert cfg.model.hqs_field_of_v2.ot_scale == 4
    assert cfg.model.hqs_field_of_v2.retransport_scale == 8
    weights = list(cfg.model.hqs_field_of_v2.retransport_state_weights)
    assert len(weights) == cfg.model.hqs_field_of_v2.iterations[1]
    assert weights[0] == 0.0
    assert all(right >= left for left, right in zip(weights, weights[1:]))
    assert cfg.loss.num_stages == 10


def test_blockwise_dustbin_sinkhorn_satisfies_multi_batch_marginals():
    matcher = BlockwiseDustbinSinkhorn(
        temperature=0.2,
        sinkhorn_iterations=40,
        query_chunk_size=2,
        local_expectation_radius=0,
        num_hypotheses=2,
        initial_dustbin_score=0.0,
        maximum_tokens=8,
        gradient_checkpointing=False,
    )
    batch = 3
    source = torch.randn(batch, 4, 2, 2)
    target = torch.randn_like(source)
    measurement = matcher(source, target)
    source_flat = torch.nn.functional.normalize(
        source.float(), dim=1
    ).flatten(2).transpose(1, 2)
    target_flat = torch.nn.functional.normalize(
        target.float(), dim=1
    ).flatten(2).transpose(1, 2)
    score = torch.bmm(source_flat, target_flat.transpose(1, 2)) / 0.2
    tokens = 4
    norm = -math.log(2 * tokens)
    real_plan = torch.exp(
        score
        + measurement.dual_source[:, :tokens].unsqueeze(-1)
        + measurement.dual_target[:, :tokens].unsqueeze(1)
        - norm
    )
    source_bin = torch.exp(
        matcher.dustbin_score
        + measurement.dual_source[:, :tokens]
        + measurement.dual_target[:, tokens : tokens + 1]
        - norm
    )
    target_bin = torch.exp(
        matcher.dustbin_score
        + measurement.dual_source[:, tokens : tokens + 1]
        + measurement.dual_target[:, :tokens]
        - norm
    )
    assert torch.allclose(
        real_plan.sum(-1) + source_bin,
        torch.ones(batch, tokens),
        atol=2e-4,
    )
    assert torch.allclose(
        real_plan.sum(1) + target_bin,
        torch.ones(batch, tokens),
        atol=2e-4,
    )


def test_transport_local_expectation_recovers_identity_matching():
    # Four orthogonal descriptors laid out on a 2x2 grid.
    source = torch.eye(4).reshape(1, 4, 2, 2)
    matcher = BlockwiseDustbinSinkhorn(
        temperature=0.01,
        sinkhorn_iterations=30,
        query_chunk_size=2,
        local_expectation_radius=0,
        num_hypotheses=2,
        initial_dustbin_score=-5.0,
        maximum_tokens=8,
        gradient_checkpointing=False,
    )
    result = matcher(source, source)
    assert torch.allclose(result.flow, torch.zeros_like(result.flow), atol=1e-3)
    assert (result.observability > 0.95).all()
    assert (result.confidence > 0.95).all()


def test_cached_and_recomputed_all_pairs_transport_agree():
    torch.manual_seed(7)
    source = torch.randn(1, 4, 2, 3)
    target = torch.randn_like(source)
    common = dict(
        temperature=0.2,
        sinkhorn_iterations=20,
        query_chunk_size=2,
        local_expectation_radius=1,
        num_hypotheses=3,
        initial_dustbin_score=0.25,
        maximum_tokens=8,
        gradient_checkpointing=False,
    )
    cached = BlockwiseDustbinSinkhorn(
        **common, cache_all_pairs_scores=True
    )
    recomputed = BlockwiseDustbinSinkhorn(
        **common, cache_all_pairs_scores=False
    )
    recomputed.load_state_dict(cached.state_dict())
    result_cached = cached(source, target)
    result_recomputed = recomputed(source, target)
    for name in (
        "flow",
        "topk_probabilities",
        "confidence",
        "observability",
        "entropy",
        "precision",
    ):
        assert torch.allclose(
            getattr(result_cached, name),
            getattr(result_recomputed, name),
            atol=1e-6,
            rtol=1e-5,
        )


def test_transport_data_solve_uses_correlated_two_by_two_precision():
    flow = torch.zeros(1, 2, 1, 1)
    proposal = torch.tensor([[[[2.0]], [[-1.0]]]])
    precision = torch.tensor([[[[2.0]], [[0.5]], [[1.0]]]])
    scalar = torch.ones(1, 1, 1, 1)
    result = solve_transport_hqs_increment(
        flow_w=flow,
        flow_z=flow,
        proposal=proposal,
        precision=precision,
        support=scalar,
        beta_map=scalar,
        damping_map=torch.zeros_like(scalar),
    )
    expected = torch.tensor([1.2173913, -0.3043478])
    assert torch.allclose(result.delta.flatten(), expected, atol=1e-6)


def test_transport_precision_respects_flow_units_when_resized():
    source = torch.eye(4).reshape(1, 4, 2, 2)
    matcher = BlockwiseDustbinSinkhorn(
        temperature=0.01,
        sinkhorn_iterations=20,
        query_chunk_size=2,
        local_expectation_radius=0,
        num_hypotheses=2,
        initial_dustbin_score=-5.0,
        maximum_tokens=8,
        gradient_checkpointing=False,
    )
    measurement = matcher(source, source)
    resized = resize_transport_measurement(measurement, (1, 1))
    # Flow units halve on each axis, so inverse covariance scales by four.
    expected = torch.nn.functional.interpolate(
        measurement.precision,
        size=(1, 1),
        mode="bilinear",
        align_corners=False,
    ) * 4.0
    assert torch.allclose(resized.precision, expected, atol=1e-6)


def test_semantic_graph_is_symmetric_nonnegative_and_preserves_constants():
    proximal = SourceSemanticGraphProximal(
        context_channels=4,
        embedding_channels=4,
        groups=1,
        neighbours=2,
        maximum_tokens=16,
    )
    context = torch.randn(1, 4, 3, 3)
    embedding = proximal.prepare(context)
    adjacency = proximal._adjacency(embedding)
    assert (adjacency >= 0).all()
    assert torch.allclose(adjacency, adjacency.transpose(1, 2), atol=1e-7)
    field = torch.ones(1, 2, 3, 3)
    scalar = torch.ones(1, 1, 3, 3) * 0.1
    result = proximal(
        data_state=field,
        previous_proximal=field,
        source_embedding=embedding,
        measurement_support=torch.zeros_like(scalar),
        beta_map=scalar,
        regularisation_map=scalar,
        inertia_map=scalar,
        sweeps=3,
    )
    assert torch.allclose(result, field, atol=1e-5)


def test_semantic_graph_proximal_has_amp_safe_fp32_solve_boundary():
    proximal = SourceSemanticGraphProximal(
        context_channels=4,
        embedding_channels=4,
        groups=1,
        neighbours=2,
        maximum_tokens=16,
    )
    context = torch.randn(2, 4, 3, 3)
    field = torch.ones(2, 2, 3, 3, dtype=torch.bfloat16)
    scalar = torch.ones(2, 1, 3, 3, dtype=torch.bfloat16) * 0.1
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        embedding = proximal.prepare(context)
        adjacency = proximal._adjacency(embedding)
        result = proximal(
            data_state=field,
            previous_proximal=field,
            source_embedding=embedding,
            measurement_support=torch.zeros_like(scalar),
            beta_map=scalar,
            regularisation_map=scalar,
            inertia_map=scalar,
            sweeps=2,
        )
    assert adjacency.dtype == torch.float32
    assert result.dtype == field.dtype
    assert torch.isfinite(result.float()).all()
    assert torch.allclose(result.float(), field.float(), atol=1e-5)


def test_transport_field_cell_has_no_target_vector_bypass_interface():
    parameters = set(inspect.signature(TransportFieldCell.forward).parameters)
    forbidden = {
        "target",
        "target_features",
        "correlation",
        "learned_flow_delta",
        "target_motion",
    }
    assert parameters.isdisjoint(forbidden)


@torch.no_grad()
def test_both_transport_models_construct_and_emit_dynamic_stage_counts():
    cases = (
        ("11_hqs_otof", "hqs_otof", "hqs_otof"),
        (
            "12_hqs_field_of_v2",
            "hqs_field_of_v2",
            "hqs_field_of_v2",
        ),
    )
    image1 = torch.rand(1, 3, 32, 48)
    image2 = torch.rand_like(image1)
    for overlay, key, solver in cases:
        cfg = _small(_load(overlay), key)
        model = build_model(cfg).eval()
        output = model(image1, image2)
        expected = sum(int(value) for value in cfg.model[key].iterations)
        assert output["solver"] == solver
        assert len(output["flow_preds"]) == expected
        assert len(output["hypothesis_proposal_lows"]) == expected
        assert output["flow_preds"][-1].shape == (1, 2, 32, 48)
        assert torch.isfinite(output["flow_preds"][-1]).all()
        assert torch.isfinite(output["ot_observability"]).all()
        assert (
            output["ot_observability"].amin() >= 0
            and output["ot_observability"].amax() <= 1
        )
