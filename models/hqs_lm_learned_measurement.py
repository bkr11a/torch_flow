r"""Learned correspondence measurement for the HQS-LM optical-flow solver.

This module implements the performance-oriented successor to the strict
soft-argmax HQS-LM data operator.  A recurrent decoder consumes the complete
local correlation tensor and predicts:

* a bounded vector residual around the analytic soft correspondence;
* a matchability probability that may approach zero; and
* a full symmetric positive-semidefinite ``2 x 2`` correspondence precision.

The learned vector is an observation inside the normal equation.  It is never
added after the analytic solve.  Target-derived evidence therefore remains
confined to the data operator:

.. math::

    \delta^\star =
    \arg\min_\delta
    \frac12\|W^{1/2}(r + J\delta)\|^2
    + \frac12\|P_\theta^{1/2}
      (\delta-\widehat\delta_\theta)\|^2
    + \frac{\beta}{2}\|w+\delta-z\|^2
    + \frac{\tau}{2}\|\delta\|^2.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

from .hqs_core_components import (
    ConvGNAct,
    PositiveSchedule,
    SeparableConvGRU,
)
from .hqs_lm_components import (
    FeatureLinearisation,
    LMState,
    LocalMatchMeasurement,
    OpticalLMIterationOutput,
    OperatorReliabilityCalibrator,
    SourceOnlyMotionProximal,
    VectorEdgeAwareJacobiProx,
    bounded_vector,
    linearise_feature_warp,
    local_correlation_measurement,
    positive_map,
    solve_optical_lm_increment,
)


def _logit(probability: float) -> float:
    probability = min(max(float(probability), 1e-5), 1.0 - 1e-5)
    return math.log(probability / (1.0 - probability))


@dataclass
class LearnedCorrespondenceMeasurement:
    """Vector observation and full precision decoded from correlation."""

    proposal: torch.Tensor
    proposal_offset: torch.Tensor
    analytic_proposal: torch.Tensor
    learned_proposal_delta: torch.Tensor
    precision: torch.Tensor
    matchability: torch.Tensor
    hidden: torch.Tensor
    attention_entropy: torch.Tensor
    attention_peak: torch.Tensor


class FlowConditionedCorrelationAttention(nn.Module):
    """Attend to correlation hypotheses using the current HQS state.

    The module is evaluated at every HQS iteration.  A query derived from
    source context, the current data/proximal states, residual diagnostics and
    the iteration index attends over every indexed correlation channel.  Its
    output is a measurement feature; it cannot emit or add a flow vector.

    This per-pixel formulation is substantially cheaper than materialising
    ``B*H*W`` sequences of ``K`` full-dimensional tokens.  Learned
    hypothesis keys/values encode the channel ordering, while the raw
    correlation score remains an explicit additive attention logit.
    """

    def __init__(
        self,
        *,
        correlation_channels: int,
        context_channels: int,
        statistic_channels: int,
        attention_channels: int = 32,
        num_heads: int = 4,
        groups: int = 8,
        temperature: float = 1.0,
    ) -> None:
        super().__init__()
        correlation_channels = int(correlation_channels)
        attention_channels = int(attention_channels)
        num_heads = int(num_heads)
        if correlation_channels < 1:
            raise ValueError("correlation_channels must be positive")
        if attention_channels < 1 or num_heads < 1:
            raise ValueError(
                "attention_channels and num_heads must be positive"
            )
        if attention_channels % num_heads != 0:
            raise ValueError(
                "attention_channels must be divisible by num_heads, got "
                f"{attention_channels} and {num_heads}"
            )
        if float(temperature) <= 0.0:
            raise ValueError("attention temperature must be positive")

        self.correlation_channels = correlation_channels
        self.attention_channels = attention_channels
        self.num_heads = num_heads
        self.head_channels = attention_channels // num_heads
        self.temperature = float(temperature)

        query_channels = (
            int(context_channels) + int(statistic_channels) + 1
        )
        self.query_projection = nn.Sequential(
            ConvGNAct(
                query_channels,
                attention_channels,
                kernel_size=1,
                groups=groups,
            ),
            nn.Conv2d(
                attention_channels,
                attention_channels,
                kernel_size=1,
                bias=False,
            ),
        )
        self.hypothesis_keys = nn.Parameter(
            torch.empty(
                num_heads,
                correlation_channels,
                self.head_channels,
            )
        )
        self.hypothesis_values = nn.Parameter(
            torch.empty(
                num_heads,
                correlation_channels,
                self.head_channels,
            )
        )
        self.hypothesis_bias = nn.Parameter(
            torch.zeros(num_heads, correlation_channels)
        )
        self.log_correlation_scale = nn.Parameter(
            torch.zeros(num_heads)
        )
        self.output_projection = nn.Sequential(
            nn.Conv2d(
                attention_channels + num_heads,
                attention_channels,
                kernel_size=1,
                bias=False,
            ),
            nn.GroupNorm(
                max(
                    divisor
                    for divisor in range(
                        1, min(groups, attention_channels) + 1
                    )
                    if attention_channels % divisor == 0
                ),
                attention_channels,
            ),
            nn.SiLU(inplace=True),
        )
        nn.init.xavier_uniform_(self.hypothesis_keys)
        nn.init.xavier_uniform_(self.hypothesis_values)

    def forward(
        self,
        *,
        correlation: torch.Tensor,
        source_context: torch.Tensor,
        statistics: torch.Tensor,
        iteration_fraction: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if correlation.shape[1] != self.correlation_channels:
            raise ValueError(
                "Flow-conditioned attention expected "
                f"{self.correlation_channels} correlation channels, got "
                f"{correlation.shape[1]}"
            )
        iteration_map = correlation.new_full(
            (correlation.shape[0], 1, *correlation.shape[-2:]),
            min(max(float(iteration_fraction), 0.0), 1.0),
        )
        query = self.query_projection(
            torch.cat(
                (source_context, statistics, iteration_map),
                dim=1,
            )
        )
        batch, _, height, width = query.shape
        query = query.float().view(
            batch,
            self.num_heads,
            self.head_channels,
            height,
            width,
        )

        # Per-pixel standardisation preserves the relative shape of the
        # correlation distribution without allowing raw score scale to
        # dominate the learned state-conditioned content logits.
        correlation32 = correlation.float()
        correlation32 = (
            correlation32
            - correlation32.mean(dim=1, keepdim=True)
        ) / (
            correlation32.var(
                dim=1, keepdim=True, unbiased=False
            ).add(1e-6).sqrt()
        )
        content_logits = torch.einsum(
            "bhdxy,hkd->bhkxy",
            query,
            self.hypothesis_keys.float(),
        ) / math.sqrt(float(self.head_channels))
        correlation_scale = self.log_correlation_scale.float().exp().view(
            1, self.num_heads, 1, 1, 1
        )
        logits = (
            content_logits
            + correlation_scale * correlation32.unsqueeze(1)
            + self.hypothesis_bias.float().view(
                1, self.num_heads, self.correlation_channels, 1, 1
            )
        ) / self.temperature
        weights = torch.softmax(logits, dim=2)

        attended = torch.einsum(
            "bhkxy,hkd->bhdxy",
            weights,
            self.hypothesis_values.float(),
        ).reshape(batch, self.attention_channels, height, width)
        attended_score = (
            weights * correlation32.unsqueeze(1)
        ).sum(dim=2)
        entropy = -(
            weights * weights.clamp_min(1e-9).log()
        ).sum(dim=2) / math.log(float(max(self.correlation_channels, 2)))
        peak = weights.max(dim=2).values
        attended = self.output_projection(
            torch.cat((attended, attended_score), dim=1).to(
                dtype=correlation.dtype
            )
        )
        return (
            attended,
            entropy.mean(dim=1, keepdim=True).to(correlation.dtype),
            peak.mean(dim=1, keepdim=True).to(correlation.dtype),
        )


class CorrelationProposalPrecisionDecoder(nn.Module):
    """Recurrently decode full correlation into a probabilistic observation.

    Precision is stored as ``[p_xx, p_xy, p_yy]``.  It is parameterised by
    positive diagonal entries and a bounded correlation coefficient:

    ``p_xy = rho * sqrt(p_xx * p_yy)``, with ``|rho| < 1``.

    This guarantees a positive-semidefinite measurement precision.  The
    positive HQS and LM diagonal terms make the complete normal matrix
    positive definite, including when matchability tends to zero.
    """

    statistic_channels = 15

    def __init__(
        self,
        *,
        correlation_channels: int,
        context_channels: int,
        embedding_channels: int = 64,
        hidden_channels: int = 96,
        groups: int = 8,
        maximum_proposal_delta: float = 2.0,
        precision_minimum: float = 0.0,
        precision_maximum: float = 1.0,
        precision_correlation_limit: float = 0.95,
        initial_precision: float = 0.20,
        initial_matchability: float = 0.90,
        analytic_confidence_floor: float = 0.02,
        attention_channels: int = 32,
        attention_heads: int = 4,
        attention_temperature: float = 1.0,
    ) -> None:
        super().__init__()
        if int(correlation_channels) < 1:
            raise ValueError("correlation_channels must be positive")
        if float(precision_maximum) <= float(precision_minimum):
            raise ValueError(
                "precision_maximum must be greater than precision_minimum"
            )
        self.hidden_channels = int(hidden_channels)
        self.maximum_proposal_delta = float(maximum_proposal_delta)
        self.precision_minimum = max(float(precision_minimum), 0.0)
        self.precision_maximum = float(precision_maximum)
        self.precision_correlation_limit = min(
            max(float(precision_correlation_limit), 0.0), 0.999
        )
        self.analytic_confidence_floor = min(
            max(float(analytic_confidence_floor), 0.0), 1.0
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
            int(context_channels),
            self.hidden_channels,
            3,
            padding=1,
        )
        self.gru = SeparableConvGRU(
            self.hidden_channels, self.hidden_channels
        )
        self.head = nn.Sequential(
            ConvGNAct(
                self.hidden_channels,
                self.hidden_channels,
                groups=groups,
            ),
            # [proposal_x, proposal_y, p_xx, p_yy, rho, matchability]
            nn.Conv2d(self.hidden_channels, 6, 3, padding=1),
        )

        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)
        precision_fraction = (
            (float(initial_precision) - self.precision_minimum)
            / (self.precision_maximum - self.precision_minimum)
        )
        with torch.no_grad():
            self.head[-1].bias[2] = _logit(precision_fraction)
            self.head[-1].bias[3] = _logit(precision_fraction)
            self.head[-1].bias[4] = 0.0
            self.head[-1].bias[5] = _logit(initial_matchability)

    def initialise_hidden(self, source_context: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.hidden_initialiser(source_context))

    def forward(
        self,
        *,
        correlation: torch.Tensor,
        source_context: torch.Tensor,
        linearisation: FeatureLinearisation,
        analytic_measurement: LocalMatchMeasurement,
        flow_w: torch.Tensor,
        flow_z: torch.Tensor,
        beta_map: torch.Tensor,
        damping_map: torch.Tensor,
        hidden: Optional[torch.Tensor],
        iteration_fraction: float = 0.0,
    ) -> LearnedCorrespondenceMeasurement:
        if hidden is None:
            hidden = self.initialise_hidden(source_context)

        residual_abs = linearisation.residual.abs().mean(dim=1, keepdim=True)
        residual_rms = torch.sqrt(
            linearisation.residual.square().mean(dim=1, keepdim=True) + 1e-8
        )
        gradient_rms = torch.sqrt(
            (
                linearisation.jacobian_x.square()
                + linearisation.jacobian_y.square()
            ).mean(dim=1, keepdim=True)
            + 1e-8
        )
        statistics = torch.cat(
            (
                flow_w,
                flow_z - flow_w,
                analytic_measurement.offset,
                residual_abs,
                residual_rms,
                gradient_rms,
                analytic_measurement.confidence,
                analytic_measurement.entropy,
                analytic_measurement.margin,
                linearisation.in_bounds,
                beta_map,
                damping_map,
            ),
            dim=1,
        )
        if statistics.shape[1] != self.statistic_channels:
            raise RuntimeError(
                "Internal learned-measurement statistic mismatch: "
                f"expected {self.statistic_channels}, got {statistics.shape[1]}"
            )
        embedded_correlation = self.correlation_projection(correlation)
        (
            attended_correlation,
            attention_entropy,
            attention_peak,
        ) = self.flow_attention(
            correlation=correlation,
            source_context=source_context,
            statistics=statistics,
            iteration_fraction=iteration_fraction,
        )
        recurrent_input = self.input_projection(
            torch.cat(
                (
                    embedded_correlation,
                    attended_correlation,
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

        learned_delta = bounded_vector(
            parameters[:, 0:2], self.maximum_proposal_delta
        )
        proposal_offset = analytic_measurement.offset + learned_delta
        proposal = flow_w + proposal_offset

        precision_range = self.precision_maximum - self.precision_minimum
        diagonal_x = self.precision_minimum + precision_range * torch.sigmoid(
            parameters[:, 2:3]
        )
        diagonal_y = self.precision_minimum + precision_range * torch.sigmoid(
            parameters[:, 3:4]
        )
        rho = self.precision_correlation_limit * torch.tanh(
            parameters[:, 4:5]
        )
        matchability = torch.sigmoid(parameters[:, 5:6])
        confidence_gate = self.analytic_confidence_floor + (
            1.0 - self.analytic_confidence_floor
        ) * analytic_measurement.confidence.square()
        gate = (
            linearisation.in_bounds * matchability * confidence_gate
        )
        p11 = gate * diagonal_x
        p22 = gate * diagonal_y
        p12 = rho * torch.sqrt((p11 * p22).clamp_min(0.0))
        precision = torch.cat((p11, p12, p22), dim=1)

        return LearnedCorrespondenceMeasurement(
            proposal=proposal,
            proposal_offset=proposal_offset,
            analytic_proposal=analytic_measurement.proposal,
            learned_proposal_delta=learned_delta,
            precision=precision,
            matchability=matchability * linearisation.in_bounds,
            hidden=hidden_next,
            attention_entropy=attention_entropy,
            attention_peak=attention_peak,
        )


class HQSOpticalLearnedMeasurementCell(nn.Module):
    """HQS-LM cell with a learned vector observation inside the LM solve."""

    def __init__(
        self,
        *,
        correlation_channels: int,
        context_channels: int,
        max_iterations: int,
        radius: int,
        match_temperature: float,
        maximum_proposal_delta: float,
        proposal_embedding_channels: int = 64,
        proposal_hidden_channels: int = 96,
        prior_hidden_channels: int = 64,
        reliability_hidden_channels: int = 32,
        groups: int = 8,
        beta_initial: float = 0.10,
        beta_minimum: float = 0.01,
        damping_initial: float = 0.10,
        damping_minimum: float = 0.001,
        lambda_initial: float = 0.08,
        lambda_minimum: float = 0.001,
        edge_alpha: float = 10.0,
        minimum_data_anchor: float = 0.05,
        initial_validity: float = 0.90,
        precision_minimum: float = 0.0,
        precision_maximum: float = 1.0,
        precision_correlation_limit: float = 0.95,
        initial_precision: float = 0.20,
        initial_matchability: float = 0.90,
        analytic_confidence_floor: float = 0.02,
        correlation_attention_channels: int = 32,
        correlation_attention_heads: int = 4,
        correlation_attention_temperature: float = 1.0,
        charbonnier_epsilon: float = 0.03,
        charbonnier_alpha: float = 0.45,
    ) -> None:
        super().__init__()
        self.radius = int(radius)
        self.max_iterations = int(max_iterations)
        self.match_temperature = float(match_temperature)
        self.charbonnier_epsilon = float(charbonnier_epsilon)
        self.charbonnier_alpha = float(charbonnier_alpha)

        self.reliability = OperatorReliabilityCalibrator(
            context_channels=context_channels,
            hidden_channels=reliability_hidden_channels,
            groups=groups,
            initial_validity=initial_validity,
        )
        self.measurement_decoder = CorrelationProposalPrecisionDecoder(
            correlation_channels=correlation_channels,
            context_channels=context_channels,
            embedding_channels=proposal_embedding_channels,
            hidden_channels=proposal_hidden_channels,
            groups=groups,
            maximum_proposal_delta=maximum_proposal_delta,
            precision_minimum=precision_minimum,
            precision_maximum=precision_maximum,
            precision_correlation_limit=precision_correlation_limit,
            initial_precision=initial_precision,
            initial_matchability=initial_matchability,
            analytic_confidence_floor=analytic_confidence_floor,
            attention_channels=correlation_attention_channels,
            attention_heads=correlation_attention_heads,
            attention_temperature=correlation_attention_temperature,
        )
        self.analytic_proximal = VectorEdgeAwareJacobiProx(
            edge_alpha=edge_alpha,
            minimum_data_anchor=minimum_data_anchor,
        )
        self.learned_proximal = SourceOnlyMotionProximal(
            state_channels=2,
            context_channels=context_channels,
            hidden_channels=prior_hidden_channels,
            groups=groups,
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

    def initialise_proposal_hidden(
        self, source_context: torch.Tensor
    ) -> torch.Tensor:
        return self.measurement_decoder.initialise_hidden(source_context)

    def forward(
        self,
        *,
        source_features: torch.Tensor,
        target_features: torch.Tensor,
        target_grad_x: torch.Tensor,
        target_grad_y: torch.Tensor,
        correlation: torch.Tensor,
        source_context: torch.Tensor,
        source_guidance: torch.Tensor,
        state: LMState,
        iteration: int,
        jacobi_sweeps: int,
        max_data_delta: float,
        max_prox_delta: float,
        proposal_hidden: Optional[torch.Tensor] = None,
    ) -> OpticalLMIterationOutput:
        beta = self.beta_schedule(iteration)
        damping = self.damping_schedule(iteration)
        regularisation = self.lambda_schedule(iteration)
        beta_map = positive_map(beta, state.w)
        damping_map = positive_map(damping, state.w)
        lambda_map = positive_map(regularisation, state.w)

        linearisation = linearise_feature_warp(
            source_features,
            target_features,
            target_grad_x,
            target_grad_y,
            state.w,
        )
        analytic_measurement = local_correlation_measurement(
            correlation,
            state.w,
            radius=self.radius,
            temperature=self.match_temperature,
        )
        appearance_validity, _ = self.reliability(
            source_context,
            linearisation,
            analytic_measurement,
            state.w,
            state.z,
        )
        learned_measurement = self.measurement_decoder(
            correlation=correlation,
            source_context=source_context,
            linearisation=linearisation,
            analytic_measurement=analytic_measurement,
            flow_w=state.w,
            flow_z=state.z,
            beta_map=beta_map,
            damping_map=damping_map,
            hidden=proposal_hidden,
            iteration_fraction=(
                float(iteration)
                / float(max(self.max_iterations - 1, 1))
            ),
        )

        normal = solve_optical_lm_increment(
            residual=linearisation.residual,
            jacobian_x=linearisation.jacobian_x,
            jacobian_y=linearisation.jacobian_y,
            appearance_validity=appearance_validity,
            flow_w=state.w,
            flow_z=state.z,
            match_proposal=learned_measurement.proposal,
            match_precision=learned_measurement.precision,
            beta_map=beta_map,
            damping_map=damping_map,
            charbonnier_epsilon=self.charbonnier_epsilon,
            charbonnier_alpha=self.charbonnier_alpha,
        )
        data_delta = bounded_vector(normal.delta, max_data_delta)
        data_state = state.w + data_delta

        match_confidence = (
            learned_measurement.matchability
            * analytic_measurement.confidence
        ).clamp(0.0, 1.0)
        data_confidence = torch.maximum(
            appearance_validity, match_confidence
        ).clamp(0.0, 1.0)
        proximal_anchor = self.analytic_proximal(
            data_state,
            source_guidance,
            beta_map,
            lambda_map,
            data_confidence,
            sweeps=jacobi_sweeps,
        )
        proximal_state = self.learned_proximal(
            proximal_anchor,
            data_state,
            state.z,
            source_context,
            source_guidance,
            data_confidence,
            normal.inverse_trace,
            beta_map,
            lambda_map,
            max_delta=max_prox_delta,
        )

        return OpticalLMIterationOutput(
            state=LMState(w=data_state, z=proximal_state),
            appearance_validity=appearance_validity,
            data_confidence=data_confidence,
            match_proposal=learned_measurement.proposal,
            match_precision=learned_measurement.precision,
            match_confidence=match_confidence,
            match_entropy=analytic_measurement.entropy,
            feature_residual=linearisation.residual,
            data_delta=data_delta,
            proximal_anchor=proximal_anchor,
            inverse_trace=normal.inverse_trace,
            condition=normal.condition,
            beta=beta,
            damping=damping,
            regularisation=regularisation,
            learned_proposal_delta=(
                learned_measurement.learned_proposal_delta
            ),
            proposal_offset=learned_measurement.proposal_offset,
            matchability=learned_measurement.matchability,
            proposal_hidden=learned_measurement.hidden,
            analytic_match_proposal=(
                learned_measurement.analytic_proposal
            ),
            correlation_attention_entropy=(
                learned_measurement.attention_entropy
            ),
            correlation_attention_peak=(
                learned_measurement.attention_peak
            ),
        )


__all__ = [
    "CorrelationProposalPrecisionDecoder",
    "FlowConditionedCorrelationAttention",
    "HQSOpticalLearnedMeasurementCell",
    "LearnedCorrespondenceMeasurement",
]
