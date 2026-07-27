from __future__ import annotations

import inspect
from pathlib import Path

import torch
from omegaconf import OmegaConf

from models.hqs_core_components import AllPairsCorrelation
from models.hqs_field_components import (
    CorrelationMixtureDecoder,
    LearnedCorrelationMixture,
    SourceGraphFieldProximal,
    local_topk_correlation_mixture,
    solve_correlation_mixture_hqs_increment,
)


def _map(value: float, height: int = 1, width: int = 1):
    return torch.full((1, 1, height, width), float(value))


def test_hqs_field_overlay_is_a_complete_ten_stage_switch():
    root = Path(__file__).resolve().parents[1]
    cfg = OmegaConf.merge(
        OmegaConf.load(root / "configs/default.yaml"),
        OmegaConf.load(root / "configs/dropins/10_hqs_field_of.yaml"),
    )
    assert cfg.model.model_type == "hqs_field_of"
    assert list(cfg.model.hqs_field_of.iterations) == [2, 4, 2, 2]
    assert list(cfg.model.hqs_field_of.num_hypotheses) == [4, 4, 4, 4]
    assert cfg.loss.num_stages == 10
    assert cfg.model.hqs_field_of.allow_external_validity_inputs is False
    assert cfg.loss.photo_weight == 0.0
    assert cfg.loss.ofce_weight == 0.0


def _learned_mixture(
    proposals: torch.Tensor,
    *,
    logits: torch.Tensor,
    precision: torch.Tensor,
    matchability: float = 1.0,
) -> LearnedCorrelationMixture:
    batch, modes, _, height, width = proposals.shape
    return LearnedCorrelationMixture(
        proposals=proposals,
        analytic_proposals=proposals.clone(),
        learned_deltas=torch.zeros_like(proposals),
        logits=logits,
        probabilities=torch.softmax(logits, dim=1),
        precision=precision,
        matchability=_map(matchability, height, width).expand(
            batch, -1, -1, -1
        ),
        hidden=torch.zeros(batch, 1, height, width),
        attention_entropy=torch.zeros(batch, 1, height, width),
        attention_peak=torch.ones(batch, 1, height, width),
    )


def test_local_topk_retains_separated_modes_instead_of_soft_averaging():
    correlation = torch.full((1, 9, 1, 1), -20.0)
    # radius=1 channel order is row-major (dy,dx). Channels 3 and 5 are
    # offsets (-1,0) and (+1,0); their mean would be the false zero mode.
    correlation[:, 3] = 10.0
    correlation[:, 5] = 9.0
    current = torch.zeros(1, 2, 1, 1)
    mixture = local_topk_correlation_mixture(
        correlation,
        current,
        radius=1,
        temperature=1.0,
        num_hypotheses=2,
    )
    recovered_x = set(
        float(value)
        for value in mixture.proposals[0, :, 0, 0, 0]
    )
    assert recovered_x == {-1.0, 1.0}
    assert mixture.retained_mass.item() > 0.99


def test_global_topk_match_decodes_map_and_reuses_transpose_for_reverse():
    source = torch.tensor([[[[1.0, 0.0]], [[0.0, 1.0]]]])
    target = torch.tensor([[[[0.0, 1.0]], [[1.0, 0.0]]]])
    correlation = AllPairsCorrelation(
        source, target, num_levels=1, radius=1
    )
    forward = correlation.global_topk_match(
        num_hypotheses=1, temperature=0.01
    )
    reverse = correlation.global_topk_match(
        num_hypotheses=1, temperature=0.01, reverse=True
    )
    assert torch.allclose(
        forward["hypotheses"][0, 0, 0, 0],
        torch.tensor([1.0, -1.0]),
    )
    assert torch.allclose(
        reverse["hypotheses"][0, 0, 0, 0],
        torch.tensor([1.0, -1.0]),
    )


def test_mixture_responsibility_selects_mode_consistent_with_proximal():
    proposals = torch.tensor(
        [[[[[2.0]], [[0.0]]], [[[-2.0]], [[0.0]]]]]
    )
    logits = torch.zeros(1, 2, 1, 1)
    precision = torch.zeros(1, 2, 3, 1, 1)
    precision[:, :, 0] = 4.0
    precision[:, :, 2] = 4.0
    mixture = _learned_mixture(
        proposals, logits=logits, precision=precision
    )
    result = solve_correlation_mixture_hqs_increment(
        flow_w=torch.zeros(1, 2, 1, 1),
        flow_z=torch.tensor([[[[1.5]], [[0.0]]]]),
        mixture=mixture,
        beta_map=_map(1.0),
        damping_map=_map(0.1),
        responsibility_reference="proximal",
    )
    assert result.responsibilities[0, 0, 0, 0] > 0.99
    assert result.delta[0, 0, 0, 0] > 1.0


def test_zero_measurement_precision_reduces_to_damped_hqs_consensus():
    proposals = torch.zeros(1, 2, 2, 1, 1)
    logits = torch.zeros(1, 2, 1, 1)
    precision = torch.zeros(1, 2, 3, 1, 1)
    mixture = _learned_mixture(
        proposals, logits=logits, precision=precision
    )
    result = solve_correlation_mixture_hqs_increment(
        flow_w=torch.tensor([[[[2.0]], [[-1.0]]]]),
        flow_z=torch.zeros(1, 2, 1, 1),
        mixture=mixture,
        beta_map=_map(1.0),
        damping_map=_map(1.0),
    )
    assert torch.allclose(
        result.delta.flatten(),
        torch.tensor([-1.0, 0.5]),
        atol=1e-6,
    )
    assert result.support.item() == 0.0


def test_source_graph_proximal_preserves_constant_field():
    proximal = SourceGraphFieldProximal(
        context_channels=4,
        embedding_channels=4,
        groups=1,
        dilations=(1, 2),
    )
    height, width = 5, 7
    field = torch.ones(1, 2, height, width)
    context = torch.randn(1, 4, height, width)
    embedding = proximal.prepare(context)
    result = proximal(
        data_state=field,
        previous_proximal=field,
        source_embedding=embedding,
        source_guidance=torch.zeros(1, 1, height, width),
        measurement_support=torch.zeros(1, 1, height, width),
        beta_map=_map(0.1, height, width),
        regularisation_map=_map(0.2, height, width),
        inertia_map=_map(0.05, height, width),
        sweeps=3,
    )
    assert torch.allclose(result, field, atol=1e-5)


def test_field_proximal_interface_cannot_receive_target_motion_evidence():
    parameters = set(
        inspect.signature(SourceGraphFieldProximal.forward).parameters
    )
    forbidden = {
        "target",
        "target_image",
        "target_features",
        "correlation",
        "photometric_residual",
        "match_proposal",
    }
    assert parameters.isdisjoint(forbidden)


def test_mixture_full_precision_is_used_in_coupled_solve():
    proposals = torch.tensor(
        [[[[[2.0]], [[-1.0]]]]]
    )
    logits = torch.zeros(1, 1, 1, 1)
    # P = [[2,.5],[.5,1]]. With beta=1 and zero damping,
    # (P+I) delta = P proposal.
    precision = torch.tensor(
        [[[[[2.0]], [[0.5]], [[1.0]]]]]
    )
    mixture = _learned_mixture(
        proposals, logits=logits, precision=precision
    )
    result = solve_correlation_mixture_hqs_increment(
        flow_w=torch.zeros(1, 2, 1, 1),
        flow_z=torch.zeros(1, 2, 1, 1),
        mixture=mixture,
        beta_map=_map(1.0),
        damping_map=_map(0.0),
    )
    expected = torch.tensor([1.2173913, -0.3043478])
    assert torch.allclose(result.delta.flatten(), expected, atol=1e-6)


def test_mixture_decoder_zero_cycle_support_has_finite_backward():
    decoder = CorrelationMixtureDecoder(
        correlation_channels=9,
        context_channels=4,
        num_hypotheses=2,
        embedding_channels=8,
        hidden_channels=8,
        attention_channels=8,
        attention_heads=2,
        groups=1,
    )
    with torch.no_grad():
        # Exercise correlated precision and the bounded attention-gain path.
        decoder.head[-1].bias[5] = 1.0
        decoder.head[-1].bias[11] = -1.0
        decoder.flow_attention.log_correlation_scale.fill_(100.0)
    flow = torch.zeros(1, 2, 2, 2)
    analytic = local_topk_correlation_mixture(
        torch.randn(1, 9, 2, 2),
        flow,
        radius=1,
        temperature=1.0,
        num_hypotheses=2,
    )
    cycle = torch.zeros(1, 2, 1, 2, 2)
    cycle[:, 0] = 1.0
    output = decoder(
        correlation=torch.randn(1, 9, 2, 2),
        analytic=analytic,
        source_context=torch.randn(1, 4, 2, 2),
        flow_w=flow,
        flow_z=flow,
        cycle_support=cycle,
        beta_map=_map(0.1, 2, 2),
        damping_map=_map(0.1, 2, 2),
        hidden=None,
        iteration_fraction=0.0,
    )
    objective = (
        output.proposals.sum()
        + output.precision.sum()
        + output.probabilities.sum()
        + output.matchability.sum()
    )
    objective.backward()
    assert torch.isfinite(output.precision).all()
    p11 = output.precision[:, :, 0]
    p12 = output.precision[:, :, 1]
    p22 = output.precision[:, :, 2]
    assert (p11 * p22 - p12.square() >= -1e-7).all()
    assert all(
        torch.isfinite(parameter.grad).all()
        for parameter in decoder.parameters()
        if parameter.grad is not None
    )
