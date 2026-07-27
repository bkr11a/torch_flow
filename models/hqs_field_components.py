r"""Probabilistic correlation and source-field operators for HQS optical flow.

The module implements the two sub-problems used by
``HQSFieldOpticalFlow``:

1. a multi-hypothesis correspondence data step; and
2. a source-conditioned graph-field proximal step.

The data operator minimises a quadratic majoriser of a correlation-mixture
likelihood coupled to the HQS auxiliary state.  For per-pixel hypotheses
``mu_j``, mixture weights ``pi_j`` and positive-semidefinite precision
matrices ``P_j``, responsibilities are evaluated at the current proximal
state and the data increment solves

.. math::

    \left(\sum_j r_j P_j + (\beta+\tau)I\right)\Delta w =
    -\left[\sum_j r_jP_j(w-\mu_j)+\beta(w-z)\right].

No target-conditioned vector is added after this solve.

The proximal operator is the Jacobi solution of a convex graph quadratic
whose affinities are functions of source-frame features only:

.. math::

    \frac{\beta}{2}\sum_p a_p\|z_p-w_p\|^2
    +\frac{\eta}{2}\|z-z^k\|^2
    +\frac{\lambda}{2}\sum_{(p,q)}A_{pq}(I_1)\|z_p-z_q\|^2.

Target-derived support can change the scalar data anchor ``a_p`` but cannot
provide a motion direction or alter the source-conditioned graph affinities.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .hqs_core_components import ConvGNAct, PositiveSchedule, SeparableConvGRU
from .hqs_lm_components import LMState, bounded_vector, positive_map
from .hqs_lm_learned_measurement import (
    FlowConditionedCorrelationAttention,
)
from .warp import flow_in_bounds_mask


def _logit(probability: float) -> float:
    probability = min(max(float(probability), 1e-5), 1.0 - 1e-5)
    return math.log(probability / (1.0 - probability))


@dataclass
class CorrelationMixture:
    """Analytic correspondence modes extracted from a correlation tensor."""

    proposals: torch.Tensor
    offsets: torch.Tensor
    logits: torch.Tensor
    probabilities: torch.Tensor
    in_bounds: torch.Tensor
    confidence: torch.Tensor
    entropy: torch.Tensor
    margin: torch.Tensor
    peak: torch.Tensor
    retained_mass: torch.Tensor


@dataclass
class LearnedCorrelationMixture:
    """Learned calibration of analytic correspondence modes."""

    proposals: torch.Tensor
    analytic_proposals: torch.Tensor
    learned_deltas: torch.Tensor
    logits: torch.Tensor
    probabilities: torch.Tensor
    precision: torch.Tensor
    matchability: torch.Tensor
    hidden: torch.Tensor
    attention_entropy: torch.Tensor
    attention_peak: torch.Tensor


@dataclass
class MixtureNormalEquation:
    """Result and diagnostics of the analytic mixture/HQS solve."""

    delta: torch.Tensor
    responsibilities: torch.Tensor
    posterior_proposal: torch.Tensor
    collapsed_precision: torch.Tensor
    support: torch.Tensor
    inverse_trace: torch.Tensor
    condition: torch.Tensor
    responsibility_entropy: torch.Tensor


@dataclass
class HQSFieldIterationOutput:
    """One operator-split iteration of the probabilistic field model."""

    state: LMState
    data_delta: torch.Tensor
    posterior_proposal: torch.Tensor
    collapsed_precision: torch.Tensor
    hypotheses: torch.Tensor
    hypothesis_logits: torch.Tensor
    hypothesis_probabilities: torch.Tensor
    hypothesis_responsibilities: torch.Tensor
    hypothesis_precision: torch.Tensor
    learned_hypothesis_deltas: torch.Tensor
    analytic_hypotheses: torch.Tensor
    matchability: torch.Tensor
    measurement_support: torch.Tensor
    cycle_support: torch.Tensor
    responsibility_entropy: torch.Tensor
    inverse_trace: torch.Tensor
    condition: torch.Tensor
    beta: torch.Tensor
    damping: torch.Tensor
    regularisation: torch.Tensor
    proximal_inertia: torch.Tensor
    proposal_hidden: torch.Tensor
    correlation_attention_entropy: torch.Tensor
    correlation_attention_peak: torch.Tensor


def local_topk_correlation_mixture(
    correlation: torch.Tensor,
    current_flow: torch.Tensor,
    *,
    radius: int,
    temperature: float,
    num_hypotheses: int,
) -> CorrelationMixture:
    """Retain the strongest local correlation modes without soft averaging.

    Only the finest correlation-pyramid level is decoded geometrically.  The
    complete multi-level tensor is still consumed by the learned calibration
    network.
    """
    if correlation.ndim != 4:
        raise ValueError(
            f"correlation must be [B,C,H,W], got {tuple(correlation.shape)}"
        )
    if current_flow.ndim != 4 or current_flow.shape[1] != 2:
        raise ValueError(
            f"current_flow must be [B,2,H,W], got {tuple(current_flow.shape)}"
        )
    radius = int(radius)
    kernel = 2 * radius + 1
    channels = kernel * kernel
    hypotheses = int(num_hypotheses)
    if correlation.shape[1] < channels:
        raise ValueError(
            f"Need {channels} finest-level channels, got "
            f"{correlation.shape[1]}"
        )
    if not 1 <= hypotheses <= channels:
        raise ValueError(
            f"num_hypotheses must be in [1,{channels}], got {hypotheses}"
        )

    logits = correlation[:, :channels].float()
    scaled_logits = logits / max(float(temperature), 1e-4)
    probabilities = torch.softmax(scaled_logits, dim=1)
    top_probabilities, indices = probabilities.topk(
        k=hypotheses, dim=1
    )
    top_logits = torch.gather(
        scaled_logits, dim=1, index=indices
    )

    dy, dx = torch.meshgrid(
        torch.arange(
            -radius,
            radius + 1,
            device=correlation.device,
            dtype=torch.float32,
        ),
        torch.arange(
            -radius,
            radius + 1,
            device=correlation.device,
            dtype=torch.float32,
        ),
        indexing="ij",
    )
    offset_table = torch.stack((dx.reshape(-1), dy.reshape(-1)), dim=-1)
    selected_offsets = offset_table[indices.permute(0, 2, 3, 1)]
    selected_offsets = selected_offsets.permute(0, 3, 4, 1, 2)
    selected_offsets = selected_offsets.to(dtype=current_flow.dtype)
    proposals = current_flow.unsqueeze(1) + selected_offsets

    in_bounds = torch.stack(
        [
            flow_in_bounds_mask(proposals[:, index])
            for index in range(hypotheses)
        ],
        dim=1,
    )
    log_channels = math.log(max(channels, 2))
    entropy = -(
        probabilities * probabilities.clamp_min(1e-9).log()
    ).sum(dim=1, keepdim=True) / log_channels
    peak = top_probabilities[:, 0:1]
    if channels > 1:
        two = probabilities.topk(k=2, dim=1).values
        margin = two[:, 0:1] - two[:, 1:2]
    else:
        margin = peak
    retained_mass = top_probabilities.sum(dim=1, keepdim=True)
    margin_ratio = margin / peak.clamp_min(1e-6)
    confidence = (
        0.35 * (1.0 - entropy)
        + 0.35 * margin_ratio
        + 0.30 * retained_mass
    ).clamp(0.0, 1.0)

    return CorrelationMixture(
        proposals=proposals,
        offsets=selected_offsets,
        logits=top_logits,
        probabilities=top_probabilities,
        in_bounds=in_bounds.to(dtype=current_flow.dtype),
        confidence=confidence.to(dtype=current_flow.dtype),
        entropy=entropy.clamp(0.0, 1.0).to(dtype=current_flow.dtype),
        margin=margin.to(dtype=current_flow.dtype),
        peak=peak.to(dtype=current_flow.dtype),
        retained_mass=retained_mass.to(dtype=current_flow.dtype),
    )


def correlation_mixture_from_global(
    result: dict[str, torch.Tensor],
    *,
    reference: torch.Tensor,
) -> CorrelationMixture:
    """Convert ``AllPairsCorrelation.global_topk_match`` output."""
    proposals = result["hypotheses"].to(reference)
    probabilities = result["probabilities"].to(reference)
    logits = result["logits"].to(reference)
    offsets = proposals - torch.zeros_like(proposals)
    hypotheses = proposals.shape[1]
    in_bounds = torch.stack(
        [
            flow_in_bounds_mask(proposals[:, index])
            for index in range(hypotheses)
        ],
        dim=1,
    )
    return CorrelationMixture(
        proposals=proposals,
        offsets=offsets,
        logits=logits,
        probabilities=probabilities,
        in_bounds=in_bounds.to(reference),
        confidence=result["confidence"].to(reference),
        entropy=result["entropy"].to(reference),
        margin=result["margin"].to(reference),
        peak=result["peak"].to(reference),
        retained_mass=result["retained_mass"].to(reference),
    )


class CorrelationMixtureDecoder(nn.Module):
    """Recurrently calibrate correspondence modes and their uncertainty.

    Every learned vector remains a displacement of a correlation-derived
    hypothesis.  The vectors and full precision matrices are consumed only by
    the analytic mixture/HQS solve.
    """

    statistic_channels = 15

    def __init__(
        self,
        *,
        correlation_channels: int,
        context_channels: int,
        num_hypotheses: int,
        embedding_channels: int = 64,
        hidden_channels: int = 96,
        attention_channels: int = 32,
        attention_heads: int = 4,
        groups: int = 8,
        maximum_hypothesis_delta: float = 2.0,
        maximum_logit_adjustment: float = 2.0,
        precision_minimum: float = 0.0,
        precision_maximum: float = 1.0,
        precision_correlation_limit: float = 0.95,
        initial_precision: float = 0.20,
        initial_matchability: float = 0.90,
        confidence_floor: float = 0.02,
        cycle_support_floor: float = 0.02,
        attention_temperature: float = 1.0,
        maximum_attention_correlation_scale: float = 8.0,
    ) -> None:
        super().__init__()
        self.num_hypotheses = int(num_hypotheses)
        self.hidden_channels = int(hidden_channels)
        self.maximum_hypothesis_delta = float(maximum_hypothesis_delta)
        self.maximum_logit_adjustment = float(maximum_logit_adjustment)
        self.precision_minimum = max(float(precision_minimum), 0.0)
        self.precision_maximum = float(precision_maximum)
        self.precision_correlation_limit = min(
            max(float(precision_correlation_limit), 0.0), 0.999
        )
        self.confidence_floor = min(
            max(float(confidence_floor), 0.0), 1.0
        )
        self.cycle_support_floor = min(
            max(float(cycle_support_floor), 0.0), 1.0
        )
        if self.num_hypotheses < 1:
            raise ValueError("num_hypotheses must be positive")
        if self.precision_maximum <= self.precision_minimum:
            raise ValueError(
                "precision_maximum must exceed precision_minimum"
            )

        self.correlation_projection = nn.Sequential(
            ConvGNAct(
                int(correlation_channels),
                int(embedding_channels),
                kernel_size=1,
                groups=groups,
            ),
            ConvGNAct(
                int(embedding_channels),
                int(embedding_channels),
                groups=groups,
            ),
        )
        self.flow_attention = FlowConditionedCorrelationAttention(
            correlation_channels=int(correlation_channels),
            context_channels=int(context_channels),
            statistic_channels=self.statistic_channels,
            attention_channels=int(attention_channels),
            num_heads=int(attention_heads),
            groups=groups,
            temperature=float(attention_temperature),
            maximum_correlation_scale=float(
                maximum_attention_correlation_scale
            ),
        )
        input_channels = (
            int(embedding_channels)
            + int(attention_channels)
            + int(context_channels)
            + self.statistic_channels
            + 2
        )
        self.input_projection = nn.Sequential(
            ConvGNAct(
                input_channels,
                self.hidden_channels,
                kernel_size=1,
                groups=groups,
            ),
            ConvGNAct(
                self.hidden_channels,
                self.hidden_channels,
                groups=groups,
            ),
        )
        self.hidden_initialiser = nn.Conv2d(
            int(context_channels), self.hidden_channels, 3, padding=1
        )
        self.gru = SeparableConvGRU(
            self.hidden_channels, self.hidden_channels
        )
        # Per mode: [dx,dy,logit,p_xx,p_yy,rho], plus matchability.
        self.head = nn.Sequential(
            ConvGNAct(
                self.hidden_channels,
                self.hidden_channels,
                groups=groups,
            ),
            nn.Conv2d(
                self.hidden_channels,
                6 * self.num_hypotheses + 1,
                3,
                padding=1,
            ),
        )
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)
        precision_fraction = (
            (float(initial_precision) - self.precision_minimum)
            / (self.precision_maximum - self.precision_minimum)
        )
        with torch.no_grad():
            for index in range(self.num_hypotheses):
                base = 6 * index
                self.head[-1].bias[base + 3] = _logit(
                    precision_fraction
                )
                self.head[-1].bias[base + 4] = _logit(
                    precision_fraction
                )
            self.head[-1].bias[-1] = _logit(initial_matchability)

    def initialise_hidden(self, source_context: torch.Tensor) -> torch.Tensor:
        with torch.autocast(
            device_type=source_context.device.type, enabled=False
        ):
            return torch.tanh(
                self.hidden_initialiser(source_context.float())
            )

    def forward(
        self,
        *,
        correlation: torch.Tensor,
        analytic: CorrelationMixture,
        source_context: torch.Tensor,
        flow_w: torch.Tensor,
        flow_z: torch.Tensor,
        cycle_support: torch.Tensor,
        beta_map: torch.Tensor,
        damping_map: torch.Tensor,
        hidden: Optional[torch.Tensor],
        iteration_fraction: float,
    ) -> LearnedCorrelationMixture:
        """Decode the calibrated mixture inside an AMP-safe FP32 island."""
        with torch.autocast(
            device_type=correlation.device.type, enabled=False
        ):
            return self._forward_fp32(
                correlation=correlation.float(),
                analytic=CorrelationMixture(
                    proposals=analytic.proposals.float(),
                    offsets=analytic.offsets.float(),
                    logits=analytic.logits.float(),
                    probabilities=analytic.probabilities.float(),
                    in_bounds=analytic.in_bounds.float(),
                    confidence=analytic.confidence.float(),
                    entropy=analytic.entropy.float(),
                    margin=analytic.margin.float(),
                    peak=analytic.peak.float(),
                    retained_mass=analytic.retained_mass.float(),
                ),
                source_context=source_context.float(),
                flow_w=flow_w.float(),
                flow_z=flow_z.float(),
                cycle_support=cycle_support.float(),
                beta_map=beta_map.float(),
                damping_map=damping_map.float(),
                hidden=None if hidden is None else hidden.float(),
                iteration_fraction=float(iteration_fraction),
            )

    def _forward_fp32(
        self,
        *,
        correlation: torch.Tensor,
        analytic: CorrelationMixture,
        source_context: torch.Tensor,
        flow_w: torch.Tensor,
        flow_z: torch.Tensor,
        cycle_support: torch.Tensor,
        beta_map: torch.Tensor,
        damping_map: torch.Tensor,
        hidden: Optional[torch.Tensor],
        iteration_fraction: float,
    ) -> LearnedCorrelationMixture:
        if analytic.proposals.shape[1] != self.num_hypotheses:
            raise ValueError(
                "Analytic mixture mode count does not match decoder: "
                f"{analytic.proposals.shape[1]} vs {self.num_hypotheses}"
            )
        if hidden is None:
            hidden = self.initialise_hidden(source_context)

        top_weights = analytic.probabilities
        top_weights = top_weights / top_weights.sum(
            dim=1, keepdim=True
        ).clamp_min(1e-8)
        mean_proposal = (
            top_weights.unsqueeze(2) * analytic.proposals
        ).sum(dim=1)
        mean_offset = mean_proposal - flow_w
        mean_in_bounds = (
            top_weights.unsqueeze(2) * analytic.in_bounds
        ).sum(dim=1)
        if cycle_support.ndim == 5:
            if cycle_support.shape[1] != self.num_hypotheses:
                raise ValueError(
                    "Per-mode cycle support must match the mixture size"
                )
            cycle_statistic = (
                top_weights.unsqueeze(2) * cycle_support
            ).sum(dim=1)
        elif cycle_support.ndim == 4:
            cycle_statistic = cycle_support
        else:
            raise ValueError(
                "cycle_support must be [B,1,H,W] or [B,K,1,H,W]"
            )
        iteration_map = flow_w.new_full(
            (flow_w.shape[0], 1, *flow_w.shape[-2:]),
            float(iteration_fraction),
        )
        statistics = torch.cat(
            (
                flow_w,
                flow_z - flow_w,
                mean_offset,
                analytic.entropy,
                analytic.margin,
                analytic.peak,
                analytic.retained_mass,
                cycle_statistic,
                mean_in_bounds,
                beta_map,
                damping_map,
                iteration_map,
            ),
            dim=1,
        )
        if statistics.shape[1] != self.statistic_channels:
            raise RuntimeError(
                "Internal mixture statistic mismatch: "
                f"expected {self.statistic_channels}, "
                f"got {statistics.shape[1]}"
            )

        embedded = self.correlation_projection(correlation)
        attended, attention_entropy, attention_peak = self.flow_attention(
            correlation=correlation,
            source_context=source_context,
            statistics=statistics,
            iteration_fraction=iteration_fraction,
        )
        recurrent_input = self.input_projection(
            torch.cat(
                (
                    embedded,
                    attended,
                    source_context,
                    statistics,
                    attention_entropy,
                    attention_peak,
                ),
                dim=1,
            )
        )
        hidden_next = self.gru(recurrent_input, hidden)
        parameters = self.head(hidden_next)

        proposal_list = []
        learned_delta_list = []
        logit_list = []
        precision_list = []
        precision_range = (
            self.precision_maximum - self.precision_minimum
        )
        matchability = torch.sigmoid(parameters[:, -1:])
        confidence_gate = self.confidence_floor + (
            1.0 - self.confidence_floor
        ) * analytic.confidence
        for index in range(self.num_hypotheses):
            base = 6 * index
            learned_delta = bounded_vector(
                parameters[:, base : base + 2],
                self.maximum_hypothesis_delta,
            )
            proposal = analytic.proposals[:, index] + learned_delta
            logit_adjustment = self.maximum_logit_adjustment * torch.tanh(
                parameters[:, base + 2 : base + 3]
            )
            diagonal_x = self.precision_minimum + precision_range * (
                torch.sigmoid(parameters[:, base + 3 : base + 4])
            )
            diagonal_y = self.precision_minimum + precision_range * (
                torch.sigmoid(parameters[:, base + 4 : base + 5])
            )
            rho = self.precision_correlation_limit * torch.tanh(
                parameters[:, base + 5 : base + 6]
            )
            candidate_in_bounds = flow_in_bounds_mask(proposal)
            candidate_cycle = (
                cycle_support[:, index]
                if cycle_support.ndim == 5
                else cycle_support
            )
            cycle_gate = self.cycle_support_floor + (
                1.0 - self.cycle_support_floor
            ) * candidate_cycle.clamp(0.0, 1.0)
            gate = (
                candidate_in_bounds
                * matchability
                * confidence_gate
                * cycle_gate
            )
            base_cross = torch.sqrt(
                (diagonal_x * diagonal_y).clamp_min(1e-8)
            )
            precision = torch.cat(
                (
                    gate * diagonal_x,
                    gate * rho * base_cross,
                    gate * diagonal_y,
                ),
                dim=1,
            )
            proposal_list.append(proposal)
            learned_delta_list.append(learned_delta)
            logit_list.append(
                analytic.logits[:, index : index + 1]
                + logit_adjustment
            )
            precision_list.append(precision)

        proposals = torch.stack(proposal_list, dim=1)
        learned_deltas = torch.stack(learned_delta_list, dim=1)
        logits = torch.cat(logit_list, dim=1)
        probabilities = torch.softmax(logits, dim=1)
        precision = torch.stack(precision_list, dim=1)
        return LearnedCorrelationMixture(
            proposals=proposals,
            analytic_proposals=analytic.proposals,
            learned_deltas=learned_deltas,
            logits=logits,
            probabilities=probabilities,
            precision=precision,
            matchability=matchability,
            hidden=hidden_next,
            attention_entropy=attention_entropy,
            attention_peak=attention_peak,
        )


def solve_correlation_mixture_hqs_increment(
    *,
    flow_w: torch.Tensor,
    flow_z: torch.Tensor,
    mixture: LearnedCorrelationMixture,
    beta_map: torch.Tensor,
    damping_map: torch.Tensor,
    responsibility_reference: str = "proximal",
    maximum_mahalanobis: float = 50.0,
) -> MixtureNormalEquation:
    """Solve the per-pixel mixture-likelihood/HQS quadratic majoriser."""
    if flow_w.shape != flow_z.shape or flow_w.shape[1] != 2:
        raise ValueError("flow_w and flow_z must be aligned [B,2,H,W]")
    if mixture.proposals.ndim != 5 or mixture.proposals.shape[2] != 2:
        raise ValueError("mixture proposals must be [B,K,2,H,W]")
    if mixture.precision.shape[:2] != mixture.proposals.shape[:2]:
        raise ValueError("mixture precision and proposals must align")
    if mixture.precision.shape[2] != 3:
        raise ValueError("mixture precision must store [p11,p12,p22]")

    w = flow_w.float()
    z = flow_z.float()
    proposals = mixture.proposals.float()
    precision = mixture.precision.float()
    p11 = precision[:, :, 0:1]
    p12 = precision[:, :, 1:2]
    p22 = precision[:, :, 2:3]
    if responsibility_reference == "proximal":
        reference = z
    elif responsibility_reference == "data":
        reference = w
    else:
        raise ValueError(
            "responsibility_reference must be 'proximal' or 'data'"
        )

    reference_error = reference.unsqueeze(1) - proposals
    reference_x = reference_error[:, :, 0:1]
    reference_y = reference_error[:, :, 1:2]
    mahalanobis = (
        p11 * reference_x.square()
        + 2.0 * p12 * reference_x * reference_y
        + p22 * reference_y.square()
    ).clamp(min=0.0, max=float(maximum_mahalanobis))
    log_prior = torch.log_softmax(mixture.logits.float(), dim=1).unsqueeze(2)
    responsibilities = torch.softmax(
        log_prior - 0.5 * mahalanobis, dim=1
    )

    weighted_p11 = (responsibilities * p11).sum(dim=1)
    weighted_p12 = (responsibilities * p12).sum(dim=1)
    weighted_p22 = (responsibilities * p22).sum(dim=1)
    beta = beta_map.float()
    damping = damping_map.float()
    h11 = weighted_p11 + beta + damping
    h12 = weighted_p12
    h22 = weighted_p22 + beta + damping

    data_error = w.unsqueeze(1) - proposals
    error_x = data_error[:, :, 0:1]
    error_y = data_error[:, :, 1:2]
    g1 = (
        responsibilities * (p11 * error_x + p12 * error_y)
    ).sum(dim=1)
    g2 = (
        responsibilities * (p12 * error_x + p22 * error_y)
    ).sum(dim=1)
    g1 = g1 + beta * (w[:, 0:1] - z[:, 0:1])
    g2 = g2 + beta * (w[:, 1:2] - z[:, 1:2])

    determinant_raw = h11 * h22 - h12.square()
    determinant_floor = (1e-6 * h11 * h22).clamp_min(1e-8)
    determinant = torch.maximum(determinant_raw, determinant_floor)
    delta_x = -(h22 * g1 - h12 * g2) / determinant
    delta_y = -(-h12 * g1 + h11 * g2) / determinant
    delta = torch.cat((delta_x, delta_y), dim=1)

    posterior_proposal = (
        responsibilities * proposals
    ).sum(dim=1)
    collapsed_precision = torch.cat(
        (weighted_p11, weighted_p12, weighted_p22), dim=1
    )
    measurement_trace = weighted_p11 + weighted_p22
    support = (
        measurement_trace
        / (measurement_trace + 2.0 * (beta + damping)).clamp_min(1e-8)
    ).clamp(0.0, 1.0)
    discriminant = torch.sqrt(
        (h11 - h22).square() + 4.0 * h12.square() + 1e-12
    )
    eigen_max = 0.5 * (h11 + h22 + discriminant)
    eigen_min = 0.5 * (h11 + h22 - discriminant)
    condition = (
        eigen_max / eigen_min.clamp_min(1e-8)
    ).clamp_max(1e6)
    inverse_trace = ((h11 + h22) / determinant).clamp_max(1e6)
    k = max(int(responsibilities.shape[1]), 2)
    responsibility_entropy = -(
        responsibilities
        * responsibilities.clamp_min(1e-9).log()
    ).sum(dim=1) / math.log(k)

    dtype = flow_w.dtype
    return MixtureNormalEquation(
        delta=delta.to(dtype=dtype),
        responsibilities=responsibilities.squeeze(2).to(dtype=dtype),
        posterior_proposal=posterior_proposal.to(dtype=dtype),
        collapsed_precision=collapsed_precision.to(dtype=dtype),
        support=support.to(dtype=dtype),
        inverse_trace=inverse_trace.to(dtype=dtype),
        condition=condition.to(dtype=dtype),
        responsibility_entropy=responsibility_entropy.to(dtype=dtype),
    )


def _shift_with_validity(
    value: torch.Tensor,
    dy: int,
    dx: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return ``value[p+(dy,dx)]`` and a non-wrapping validity mask."""
    shifted = torch.roll(value, shifts=(-int(dy), -int(dx)), dims=(-2, -1))
    mask = value.new_ones((value.shape[0], 1, *value.shape[-2:]))
    if dy > 0:
        mask[..., -dy:, :] = 0.0
    elif dy < 0:
        mask[..., : -dy, :] = 0.0
    if dx > 0:
        mask[..., :, -dx:] = 0.0
    elif dx < 0:
        mask[..., :, : -dx] = 0.0
    return shifted, mask


class SourceGraphFieldProximal(nn.Module):
    """Convex nonlocal field completion with source-only affinities."""

    def __init__(
        self,
        *,
        context_channels: int,
        embedding_channels: int = 32,
        groups: int = 8,
        dilations: Sequence[int] = (1, 2, 4),
        feature_temperature: float = 0.25,
        edge_alpha: float = 10.0,
        anchor_floor: float = 0.05,
    ) -> None:
        super().__init__()
        self.embedding = nn.Sequential(
            ConvGNAct(
                int(context_channels),
                int(embedding_channels),
                kernel_size=1,
                groups=groups,
            ),
            ConvGNAct(
                int(embedding_channels),
                int(embedding_channels),
                groups=groups,
            ),
        )
        values = tuple(sorted({int(value) for value in dilations}))
        if not values or values[0] < 1:
            raise ValueError("graph dilations must be positive")
        self.dilations = values
        self.feature_temperature = max(float(feature_temperature), 1e-4)
        self.edge_alpha = max(float(edge_alpha), 0.0)
        self.anchor_floor = min(max(float(anchor_floor), 0.0), 1.0)

    def prepare(self, source_context: torch.Tensor) -> torch.Tensor:
        with torch.autocast(
            device_type=source_context.device.type, enabled=False
        ):
            return F.normalize(
                self.embedding(source_context.float()), dim=1, eps=1e-6
            )

    def _directions(self) -> Tuple[Tuple[int, int], ...]:
        directions = []
        for dilation in self.dilations:
            directions.extend(
                (
                    (0, dilation),
                    (0, -dilation),
                    (dilation, 0),
                    (-dilation, 0),
                )
            )
            if dilation == 1:
                directions.extend(
                    (
                        (1, 1),
                        (1, -1),
                        (-1, 1),
                        (-1, -1),
                    )
                )
        return tuple(directions)

    def forward(
        self,
        *,
        data_state: torch.Tensor,
        previous_proximal: torch.Tensor,
        source_embedding: torch.Tensor,
        source_guidance: torch.Tensor,
        measurement_support: torch.Tensor,
        beta_map: torch.Tensor,
        regularisation_map: torch.Tensor,
        inertia_map: torch.Tensor,
        sweeps: int,
    ) -> torch.Tensor:
        """Solve the fixed-affinity graph quadratic by Jacobi iteration."""
        if data_state.shape != previous_proximal.shape:
            raise ValueError("data and proximal states must align")
        if source_embedding.shape[-2:] != data_state.shape[-2:]:
            raise ValueError("source embedding and field grid must align")
        if source_guidance.shape[-2:] != data_state.shape[-2:]:
            raise ValueError("source guidance and field grid must align")

        embedding = source_embedding.float()
        guidance = source_guidance.float()
        weights = []
        degree = data_state.new_zeros(
            (data_state.shape[0], 1, *data_state.shape[-2:]),
            dtype=torch.float32,
        )
        for dy, dx in self._directions():
            neighbour_embedding, valid = _shift_with_validity(
                embedding, dy, dx
            )
            neighbour_guidance, _ = _shift_with_validity(
                guidance, dy, dx
            )
            feature_distance = (
                embedding - neighbour_embedding
            ).square().mean(dim=1, keepdim=True)
            guidance_distance = (
                guidance - neighbour_guidance
            ).abs().mean(dim=1, keepdim=True)
            weight = valid * torch.exp(
                -feature_distance / self.feature_temperature
                - self.edge_alpha * guidance_distance
            )
            weights.append((dy, dx, weight))
            degree = degree + weight

        support = measurement_support.float().clamp(0.0, 1.0)
        anchor = beta_map.float() * (
            self.anchor_floor + (1.0 - self.anchor_floor) * support
        )
        inertia = inertia_map.float()
        regularisation = regularisation_map.float()
        data32 = data_state.float()
        previous32 = previous_proximal.float()
        current = previous32
        for _ in range(max(int(sweeps), 1)):
            neighbour_sum = torch.zeros_like(current)
            for dy, dx, weight in weights:
                neighbour, _ = _shift_with_validity(current, dy, dx)
                neighbour_sum = neighbour_sum + weight * neighbour
            numerator = (
                anchor * data32
                + inertia * previous32
                + regularisation * neighbour_sum
            )
            denominator = (
                anchor + inertia + regularisation * degree
            ).clamp_min(1e-6)
            current = numerator / denominator
        return current.to(dtype=data_state.dtype)


class HQSCorrelationFieldCell(nn.Module):
    """One mixture-data and source-graph proximal HQS iteration."""

    def __init__(
        self,
        *,
        correlation_channels: int,
        context_channels: int,
        max_iterations: int,
        radius: int,
        num_hypotheses: int,
        mixture_temperature: float,
        proposal_embedding_channels: int = 64,
        proposal_hidden_channels: int = 96,
        correlation_attention_channels: int = 32,
        correlation_attention_heads: int = 4,
        graph_embedding_channels: int = 32,
        graph_dilations: Sequence[int] = (1, 2, 4),
        groups: int = 8,
        beta_initial: float = 0.10,
        beta_minimum: float = 0.01,
        damping_initial: float = 0.10,
        damping_minimum: float = 0.001,
        lambda_initial: float = 0.08,
        lambda_minimum: float = 0.001,
        proximal_inertia_initial: float = 0.05,
        proximal_inertia_minimum: float = 0.001,
        maximum_hypothesis_delta: float = 2.0,
        maximum_logit_adjustment: float = 2.0,
        precision_minimum: float = 0.0,
        precision_maximum: float = 1.0,
        precision_correlation_limit: float = 0.95,
        initial_precision: float = 0.20,
        initial_matchability: float = 0.90,
        confidence_floor: float = 0.02,
        cycle_support_floor: float = 0.02,
        graph_feature_temperature: float = 0.25,
        graph_edge_alpha: float = 10.0,
        graph_anchor_floor: float = 0.05,
        correlation_attention_temperature: float = 1.0,
        maximum_attention_correlation_scale: float = 8.0,
        responsibility_reference: str = "proximal",
    ) -> None:
        super().__init__()
        self.radius = int(radius)
        self.num_hypotheses = int(num_hypotheses)
        self.max_iterations = int(max_iterations)
        self.mixture_temperature = float(mixture_temperature)
        self.responsibility_reference = str(responsibility_reference)
        self.measurement_decoder = CorrelationMixtureDecoder(
            correlation_channels=correlation_channels,
            context_channels=context_channels,
            num_hypotheses=num_hypotheses,
            embedding_channels=proposal_embedding_channels,
            hidden_channels=proposal_hidden_channels,
            attention_channels=correlation_attention_channels,
            attention_heads=correlation_attention_heads,
            groups=groups,
            maximum_hypothesis_delta=maximum_hypothesis_delta,
            maximum_logit_adjustment=maximum_logit_adjustment,
            precision_minimum=precision_minimum,
            precision_maximum=precision_maximum,
            precision_correlation_limit=precision_correlation_limit,
            initial_precision=initial_precision,
            initial_matchability=initial_matchability,
            confidence_floor=confidence_floor,
            cycle_support_floor=cycle_support_floor,
            attention_temperature=correlation_attention_temperature,
            maximum_attention_correlation_scale=(
                maximum_attention_correlation_scale
            ),
        )
        self.field_proximal = SourceGraphFieldProximal(
            context_channels=context_channels,
            embedding_channels=graph_embedding_channels,
            groups=groups,
            dilations=graph_dilations,
            feature_temperature=graph_feature_temperature,
            edge_alpha=graph_edge_alpha,
            anchor_floor=graph_anchor_floor,
        )
        self.beta_schedule = PositiveSchedule(
            max_iterations, beta_initial, beta_minimum
        )
        self.damping_schedule = PositiveSchedule(
            max_iterations, damping_initial, damping_minimum
        )
        self.lambda_schedule = PositiveSchedule(
            max_iterations, lambda_initial, lambda_minimum
        )
        self.inertia_schedule = PositiveSchedule(
            max_iterations,
            proximal_inertia_initial,
            proximal_inertia_minimum,
        )

    def initialise_proposal_hidden(
        self, source_context: torch.Tensor
    ) -> torch.Tensor:
        return self.measurement_decoder.initialise_hidden(source_context)

    def prepare_source_graph(
        self, source_context: torch.Tensor
    ) -> torch.Tensor:
        return self.field_proximal.prepare(source_context)

    def forward(
        self,
        *,
        correlation: torch.Tensor,
        source_context: torch.Tensor,
        source_guidance: torch.Tensor,
        source_graph_embedding: torch.Tensor,
        cycle_support: torch.Tensor,
        state: LMState,
        iteration: int,
        graph_sweeps: int,
        max_data_delta: float,
        proposal_hidden: Optional[torch.Tensor] = None,
        analytic_mixture: Optional[CorrelationMixture] = None,
    ) -> HQSFieldIterationOutput:
        beta = self.beta_schedule(iteration)
        damping = self.damping_schedule(iteration)
        regularisation = self.lambda_schedule(iteration)
        inertia = self.inertia_schedule(iteration)
        beta_map = positive_map(beta, state.w)
        damping_map = positive_map(damping, state.w)
        lambda_map = positive_map(regularisation, state.w)
        inertia_map = positive_map(inertia, state.w)

        if analytic_mixture is None:
            analytic_mixture = local_topk_correlation_mixture(
                correlation,
                state.w,
                radius=self.radius,
                temperature=self.mixture_temperature,
                num_hypotheses=self.num_hypotheses,
            )
        learned_mixture = self.measurement_decoder(
            correlation=correlation,
            analytic=analytic_mixture,
            source_context=source_context,
            flow_w=state.w,
            flow_z=state.z,
            cycle_support=cycle_support,
            beta_map=beta_map,
            damping_map=damping_map,
            hidden=proposal_hidden,
            iteration_fraction=(
                float(iteration)
                / float(max(self.max_iterations - 1, 1))
            ),
        )
        normal = solve_correlation_mixture_hqs_increment(
            flow_w=state.w,
            flow_z=state.z,
            mixture=learned_mixture,
            beta_map=beta_map,
            damping_map=damping_map,
            responsibility_reference=self.responsibility_reference,
        )
        data_delta = bounded_vector(normal.delta, max_data_delta)
        data_state = state.w + data_delta
        proximal_state = self.field_proximal(
            data_state=data_state,
            previous_proximal=state.z,
            source_embedding=source_graph_embedding,
            source_guidance=source_guidance,
            measurement_support=normal.support,
            beta_map=beta_map,
            regularisation_map=lambda_map,
            inertia_map=inertia_map,
            sweeps=graph_sweeps,
        )
        cycle_diagnostic = (
            cycle_support.max(dim=1).values
            if cycle_support.ndim == 5
            else cycle_support
        )
        return HQSFieldIterationOutput(
            state=LMState(w=data_state, z=proximal_state),
            data_delta=data_delta,
            posterior_proposal=normal.posterior_proposal,
            collapsed_precision=normal.collapsed_precision,
            hypotheses=learned_mixture.proposals,
            hypothesis_logits=learned_mixture.logits,
            hypothesis_probabilities=learned_mixture.probabilities,
            hypothesis_responsibilities=normal.responsibilities,
            hypothesis_precision=learned_mixture.precision,
            learned_hypothesis_deltas=learned_mixture.learned_deltas,
            analytic_hypotheses=learned_mixture.analytic_proposals,
            matchability=learned_mixture.matchability,
            measurement_support=normal.support,
            cycle_support=cycle_diagnostic,
            responsibility_entropy=normal.responsibility_entropy,
            inverse_trace=normal.inverse_trace,
            condition=normal.condition,
            beta=beta,
            damping=damping,
            regularisation=regularisation,
            proximal_inertia=inertia,
            proposal_hidden=learned_mixture.hidden,
            correlation_attention_entropy=(
                learned_mixture.attention_entropy
            ),
            correlation_attention_peak=learned_mixture.attention_peak,
        )


__all__ = [
    "CorrelationMixture",
    "CorrelationMixtureDecoder",
    "HQSCorrelationFieldCell",
    "HQSFieldIterationOutput",
    "LearnedCorrelationMixture",
    "MixtureNormalEquation",
    "SourceGraphFieldProximal",
    "correlation_mixture_from_global",
    "local_topk_correlation_mixture",
    "solve_correlation_mixture_hqs_increment",
]
