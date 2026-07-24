r"""Shared operators for the HQS-LM optical-flow and scene-flow models.

The module enforces the central architectural invariant of the revised
formulation:

* target-frame features and correlation evidence terminate at the data step;
* the data step produces motion only through a damped normal-equation solve;
* the proximal step receives source-frame context, split-state variables and
  scalar conditioning diagnostics, but no target features or correlation
  tensor.

The optical-flow data update is the per-pixel minimiser of

.. math::

    \frac{1}{2}\|W^{1/2}(r + J\delta)\|_2^2
    + \frac{1}{2}\|P_m^{1/2}(w+\delta-\tilde w)\|_2^2
    + \frac{\beta}{2}\|w+\delta-z\|_2^2
    + \frac{\tau}{2}\|\delta\|_2^2.

The corresponding scene-flow model reuses the schedules, correlation
measurement, source-only proximal and field upsampler defined here.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .hqs_core_components import (
    ConvGNAct,
    PositiveSchedule,
    edge_magnitude,
    spatial_gradients,
)
from .warp import backward_warp, flow_in_bounds_mask


def _groups(channels: int, requested: int) -> int:
    for groups in range(min(int(channels), int(requested)), 0, -1):
        if channels % groups == 0:
            return groups
    return 1


def bounded_vector(vector: torch.Tensor, maximum: float) -> torch.Tensor:
    """Smoothly bound each per-pixel vector by ``maximum``."""
    maximum = max(float(maximum), 1e-6)
    norm = torch.sqrt(vector.square().sum(dim=1, keepdim=True) + 1e-12)
    ratio = norm / maximum
    scale = torch.where(
        ratio > 1e-4,
        torch.tanh(ratio) / ratio,
        1.0 - ratio.square() / 3.0,
    )
    return vector * scale


def positive_map(
    value: torch.Tensor,
    reference: torch.Tensor,
) -> torch.Tensor:
    """Broadcast a positive scalar schedule value over a spatial tensor."""
    return value.to(device=reference.device, dtype=reference.dtype).view(
        1, 1, 1, 1
    ).expand(reference.shape[0], 1, *reference.shape[-2:])


@dataclass
class LMState:
    """Interpretable HQS state: data variable ``w`` and proximal variable ``z``."""

    w: torch.Tensor
    z: torch.Tensor


@dataclass
class LocalMatchMeasurement:
    """Analytic local correspondence observation decoded from correlation."""

    proposal: torch.Tensor
    offset: torch.Tensor
    confidence: torch.Tensor
    entropy: torch.Tensor
    margin: torch.Tensor


def local_correlation_measurement(
    correlation: torch.Tensor,
    current_flow: torch.Tensor,
    *,
    radius: int,
    temperature: float,
) -> LocalMatchMeasurement:
    """Decode the finest-level correlation neighbourhood by soft argmax.

    ``AllPairsCorrelation.lookup`` concatenates pyramid levels, whereas
    ``LocalCorrBlock`` returns one level.  The first ``(2r+1)^2`` channels have
    the same row-major ``(dy, dx)`` ordering in both implementations.
    """
    if correlation.ndim != 4:
        raise ValueError(
            f"correlation must be [B,K,H,W], got {tuple(correlation.shape)}"
        )
    if current_flow.ndim != 4 or current_flow.shape[1] != 2:
        raise ValueError(
            f"current_flow must be [B,2,H,W], got {tuple(current_flow.shape)}"
        )
    radius = int(radius)
    kernel = 2 * radius + 1
    channels = kernel * kernel
    if correlation.shape[1] < channels:
        raise ValueError(
            f"Need at least {channels} correlation channels for radius={radius}, "
            f"got {correlation.shape[1]}"
        )

    logits = correlation[:, :channels].float()
    probabilities = torch.softmax(
        logits / max(float(temperature), 1e-4), dim=1
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
    offsets = torch.stack((dx.reshape(-1), dy.reshape(-1)), dim=0).view(
        1, 2, channels, 1, 1
    )
    offset = (probabilities.unsqueeze(1) * offsets).sum(dim=2)

    log_channels = math.log(max(channels, 2))
    entropy = -(
        probabilities * probabilities.clamp_min(1e-9).log()
    ).sum(dim=1, keepdim=True) / log_channels
    if channels > 1:
        top2 = probabilities.topk(k=2, dim=1).values
        peak = top2[:, 0:1]
        margin = top2[:, 0:1] - top2[:, 1:2]
    else:
        peak = probabilities
        margin = probabilities
    margin_ratio = margin / peak.clamp_min(1e-6)
    confidence = (
        0.5 * (1.0 - entropy) + 0.5 * margin_ratio
    ).clamp(0.0, 1.0)

    dtype = current_flow.dtype
    offset = offset.to(dtype=dtype)
    return LocalMatchMeasurement(
        proposal=current_flow + offset,
        offset=offset,
        confidence=confidence.to(dtype=dtype),
        entropy=entropy.clamp(0.0, 1.0).to(dtype=dtype),
        margin=margin.to(dtype=dtype),
    )


@dataclass
class FeatureLinearisation:
    residual: torch.Tensor
    jacobian_x: torch.Tensor
    jacobian_y: torch.Tensor
    in_bounds: torch.Tensor


def linearise_feature_warp(
    source_features: torch.Tensor,
    target_features: torch.Tensor,
    target_grad_x: torch.Tensor,
    target_grad_y: torch.Tensor,
    flow_xy: torch.Tensor,
) -> FeatureLinearisation:
    """Warp target features and their precomputed spatial derivatives."""
    if source_features.shape != target_features.shape:
        raise ValueError(
            "source/target feature shapes differ: "
            f"{tuple(source_features.shape)} vs {tuple(target_features.shape)}"
        )
    return FeatureLinearisation(
        residual=backward_warp(
            target_features, flow_xy, padding_mode="border"
        )
        - source_features,
        jacobian_x=backward_warp(
            target_grad_x, flow_xy, padding_mode="border"
        ),
        jacobian_y=backward_warp(
            target_grad_y, flow_xy, padding_mode="border"
        ),
        in_bounds=flow_in_bounds_mask(flow_xy),
    )


def charbonnier_irls_weight(
    residual: torch.Tensor,
    *,
    epsilon: float = 0.03,
    alpha: float = 0.45,
) -> torch.Tensor:
    """Bounded IRLS weight for a generalised Charbonnier penalty.

    The normalisation makes the weight one at zero residual and prevents very
    small residuals from producing arbitrarily large Hessian entries.
    """
    epsilon = max(float(epsilon), 1e-6)
    alpha = min(max(float(alpha), 0.05), 1.0)
    return (
        1.0 + residual.float().square() / (epsilon * epsilon)
    ).pow(alpha - 1.0).to(dtype=residual.dtype)


@dataclass
class OpticalNormalEquation:
    delta: torch.Tensor
    h11: torch.Tensor
    h12: torch.Tensor
    h22: torch.Tensor
    inverse_trace: torch.Tensor
    condition: torch.Tensor
    robust_weight: torch.Tensor


def solve_optical_lm_increment(
    *,
    residual: torch.Tensor,
    jacobian_x: torch.Tensor,
    jacobian_y: torch.Tensor,
    appearance_validity: torch.Tensor,
    flow_w: torch.Tensor,
    flow_z: torch.Tensor,
    match_proposal: torch.Tensor,
    match_precision: torch.Tensor,
    beta_map: torch.Tensor,
    damping_map: torch.Tensor,
    charbonnier_epsilon: float = 0.03,
    charbonnier_alpha: float = 0.45,
) -> OpticalNormalEquation:
    """Solve the robust multi-channel ``2 x 2`` optical-flow LM system."""
    if residual.shape != jacobian_x.shape or residual.shape != jacobian_y.shape:
        raise ValueError("Residual and feature Jacobians must have equal shapes")
    if flow_w.shape != flow_z.shape or flow_w.shape != match_proposal.shape:
        raise ValueError("Flow, proximal state and match proposal must align")
    if flow_w.shape[1] != 2:
        raise ValueError(f"Expected two flow channels, got {flow_w.shape[1]}")

    channels = max(int(residual.shape[1]), 1)
    robust = charbonnier_irls_weight(
        residual,
        epsilon=charbonnier_epsilon,
        alpha=charbonnier_alpha,
    )
    validity = appearance_validity.to(dtype=residual.dtype).clamp(0.0, 1.0)
    weight = robust * validity
    normaliser = 1.0 / float(channels)

    jx = jacobian_x.float()
    jy = jacobian_y.float()
    r = residual.float()
    weight32 = weight.float()

    px = match_precision[:, 0:1].float()
    py = (
        match_precision[:, 1:2].float()
        if match_precision.shape[1] > 1
        else px
    )
    beta = beta_map.float()
    damping = damping_map.float()

    h11 = normaliser * (weight32 * jx.square()).sum(dim=1, keepdim=True)
    h12 = normaliser * (weight32 * jx * jy).sum(dim=1, keepdim=True)
    h22 = normaliser * (weight32 * jy.square()).sum(dim=1, keepdim=True)
    h11 = h11 + px + beta + damping
    h22 = h22 + py + beta + damping

    g1 = normaliser * (weight32 * jx * r).sum(dim=1, keepdim=True)
    g2 = normaliser * (weight32 * jy * r).sum(dim=1, keepdim=True)
    g1 = (
        g1
        + px * (flow_w[:, 0:1].float() - match_proposal[:, 0:1].float())
        + beta * (flow_w[:, 0:1].float() - flow_z[:, 0:1].float())
    )
    g2 = (
        g2
        + py * (flow_w[:, 1:2].float() - match_proposal[:, 1:2].float())
        + beta * (flow_w[:, 1:2].float() - flow_z[:, 1:2].float())
    )

    determinant = (h11 * h22 - h12.square()).clamp_min(1e-8)
    delta_x = -(h22 * g1 - h12 * g2) / determinant
    delta_y = -(-h12 * g1 + h11 * g2) / determinant
    delta = torch.cat((delta_x, delta_y), dim=1).to(dtype=flow_w.dtype)

    discriminant = torch.sqrt(
        (h11 - h22).square() + 4.0 * h12.square() + 1e-12
    )
    eigen_max = 0.5 * (h11 + h22 + discriminant)
    eigen_min = 0.5 * (h11 + h22 - discriminant)
    condition = eigen_max / eigen_min.clamp_min(1e-8)
    inverse_trace = (h11 + h22) / determinant

    return OpticalNormalEquation(
        delta=delta,
        h11=h11.to(dtype=flow_w.dtype),
        h12=h12.to(dtype=flow_w.dtype),
        h22=h22.to(dtype=flow_w.dtype),
        inverse_trace=inverse_trace.to(dtype=flow_w.dtype),
        condition=condition.to(dtype=flow_w.dtype),
        robust_weight=weight,
    )


class OperatorReliabilityCalibrator(nn.Module):
    """Calibrate scalar appearance validity and match precision.

    The head cannot emit a motion vector.  Its two outputs only scale the
    appearance residual and the analytic correspondence precision.
    """

    statistic_channels = 7

    def __init__(
        self,
        context_channels: int,
        hidden_channels: int = 32,
        groups: int = 8,
        initial_validity: float = 0.90,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            ConvGNAct(
                context_channels + self.statistic_channels,
                hidden_channels,
                groups=groups,
            ),
            ConvGNAct(hidden_channels, hidden_channels, groups=groups),
            nn.Conv2d(hidden_channels, 2, 3, padding=1),
        )
        nn.init.zeros_(self.net[-1].weight)
        initial_validity = min(max(float(initial_validity), 1e-4), 1.0 - 1e-4)
        with torch.no_grad():
            self.net[-1].bias[0] = math.log(
                initial_validity / (1.0 - initial_validity)
            )
            self.net[-1].bias[1] = 0.0

    def forward(
        self,
        source_context: torch.Tensor,
        linearisation: FeatureLinearisation,
        measurement: LocalMatchMeasurement,
        flow_w: torch.Tensor,
        flow_z: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        residual_abs = linearisation.residual.abs().mean(dim=1, keepdim=True)
        residual_rms = torch.sqrt(
            linearisation.residual.square().mean(dim=1, keepdim=True) + 1e-8
        )
        coupling = torch.sqrt(
            (flow_w - flow_z).square().sum(dim=1, keepdim=True) + 1e-8
        )
        statistics = torch.cat(
            (
                residual_abs,
                residual_rms,
                measurement.confidence,
                measurement.entropy,
                measurement.margin,
                coupling,
                linearisation.in_bounds,
            ),
            dim=1,
        )
        logits = self.net(torch.cat((source_context, statistics), dim=1))
        appearance_validity = (
            linearisation.in_bounds * torch.sigmoid(logits[:, 0:1])
        )
        # Bounded calibration prevents the learned scalar from deleting or
        # arbitrarily amplifying the analytic correspondence observation.
        precision_multiplier = 0.25 + 1.5 * torch.sigmoid(logits[:, 1:2])
        return appearance_validity, precision_multiplier


class VectorEdgeAwareJacobiProx(nn.Module):
    """Analytic edge-aware proximal anchor for a vector field."""

    def __init__(
        self,
        edge_alpha: float = 10.0,
        minimum_data_anchor: float = 0.05,
    ) -> None:
        super().__init__()
        self.edge_alpha = float(edge_alpha)
        self.minimum_data_anchor = min(
            max(float(minimum_data_anchor), 0.0), 1.0
        )

    def forward(
        self,
        data_state: torch.Tensor,
        source_guidance: torch.Tensor,
        beta_map: torch.Tensor,
        regularisation_map: torch.Tensor,
        data_confidence: torch.Tensor,
        *,
        sweeps: int,
    ) -> torch.Tensor:
        if int(sweeps) <= 0:
            return data_state
        if source_guidance.shape[1] != 1:
            raise ValueError("source_guidance must have one channel")

        right_weight = torch.zeros_like(source_guidance)
        right_weight[..., :-1] = torch.exp(
            -self.edge_alpha
            * (
                source_guidance[..., 1:] - source_guidance[..., :-1]
            ).abs()
        )
        left_weight = torch.zeros_like(source_guidance)
        left_weight[..., 1:] = right_weight[..., :-1]
        down_weight = torch.zeros_like(source_guidance)
        down_weight[..., :-1, :] = torch.exp(
            -self.edge_alpha
            * (
                source_guidance[..., 1:, :]
                - source_guidance[..., :-1, :]
            ).abs()
        )
        up_weight = torch.zeros_like(source_guidance)
        up_weight[..., 1:, :] = down_weight[..., :-1, :]

        confidence = data_confidence.clamp(0.0, 1.0)
        confidence = self.minimum_data_anchor + (
            1.0 - self.minimum_data_anchor
        ) * confidence
        anchor = beta_map * confidence
        weight_sum = right_weight + left_weight + down_weight + up_weight
        denominator = anchor + regularisation_map * weight_sum

        proximal = data_state
        for _ in range(int(sweeps)):
            right = F.pad(proximal[..., 1:], (0, 1, 0, 0))
            left = F.pad(proximal[..., :-1], (1, 0, 0, 0))
            down = F.pad(proximal[..., 1:, :], (0, 0, 0, 1))
            up = F.pad(proximal[..., :-1, :], (0, 0, 1, 0))
            neighbours = (
                right_weight * right
                + left_weight * left
                + down_weight * down
                + up_weight * up
            )
            proximal = (
                anchor * data_state + regularisation_map * neighbours
            ) / denominator.clamp_min(1e-6)
        return proximal


class _DilatedPriorBlock(nn.Module):
    def __init__(self, channels: int, dilation: int, groups: int) -> None:
        super().__init__()
        self.depthwise = nn.Conv2d(
            channels,
            channels,
            3,
            padding=int(dilation),
            dilation=int(dilation),
            groups=channels,
            bias=False,
        )
        self.pointwise = nn.Conv2d(channels, channels, 1, bias=False)
        self.norm = nn.GroupNorm(_groups(channels, groups), channels)
        self.act = nn.SiLU(inplace=True)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        residual = self.pointwise(self.depthwise(value))
        return self.act(value + self.norm(residual))


class SourceOnlyMotionProximal(nn.Module):
    """Source-conditioned learned motion completion around an analytic prox.

    No target image, target feature, photometric residual or correlation tensor
    is accepted by this interface.  ``data_confidence`` and ``uncertainty`` are
    scalar diagnostics from the analytic solve; they control where the prior is
    needed but cannot supply a target-derived motion direction.
    """

    def __init__(
        self,
        *,
        state_channels: int,
        context_channels: int,
        hidden_channels: int = 64,
        groups: int = 8,
    ) -> None:
        super().__init__()
        self.state_channels = int(state_channels)
        # analytic, data-analytic, previous + context + guidance/diagnostics.
        in_channels = (
            3 * self.state_channels + context_channels + 5
        )
        self.input = ConvGNAct(
            in_channels, hidden_channels, groups=groups
        )
        self.body = nn.Sequential(
            _DilatedPriorBlock(hidden_channels, dilation=1, groups=groups),
            _DilatedPriorBlock(hidden_channels, dilation=2, groups=groups),
            _DilatedPriorBlock(hidden_channels, dilation=4, groups=groups),
        )
        self.output = nn.Conv2d(
            hidden_channels, self.state_channels, 3, padding=1
        )
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(
        self,
        analytic_proximal: torch.Tensor,
        data_state: torch.Tensor,
        previous_proximal: torch.Tensor,
        source_context: torch.Tensor,
        source_guidance: torch.Tensor,
        data_confidence: torch.Tensor,
        uncertainty: torch.Tensor,
        beta_map: torch.Tensor,
        regularisation_map: torch.Tensor,
        *,
        max_delta: float,
    ) -> torch.Tensor:
        if analytic_proximal.shape[1] != self.state_channels:
            raise ValueError(
                f"Expected {self.state_channels} state channels, got "
                f"{analytic_proximal.shape[1]}"
            )
        guidance_edge = edge_magnitude(source_guidance)
        uncertainty = torch.log1p(uncertainty.clamp_min(0.0))
        inputs = torch.cat(
            (
                analytic_proximal,
                data_state - analytic_proximal,
                previous_proximal,
                source_context,
                guidance_edge,
                data_confidence,
                uncertainty,
                beta_map,
                regularisation_map,
            ),
            dim=1,
        )
        residual = float(max_delta) * torch.tanh(
            self.output(self.body(self.input(inputs)))
        )
        return analytic_proximal + residual


@dataclass
class OpticalLMIterationOutput:
    state: LMState
    appearance_validity: torch.Tensor
    data_confidence: torch.Tensor
    match_proposal: torch.Tensor
    match_precision: torch.Tensor
    match_confidence: torch.Tensor
    match_entropy: torch.Tensor
    feature_residual: torch.Tensor
    data_delta: torch.Tensor
    proximal_anchor: torch.Tensor
    inverse_trace: torch.Tensor
    condition: torch.Tensor
    beta: torch.Tensor
    damping: torch.Tensor
    regularisation: torch.Tensor


class HQSOpticalLMCell(nn.Module):
    """One HQS iteration with an analytic multi-channel LM data step."""

    def __init__(
        self,
        *,
        context_channels: int,
        max_iterations: int,
        radius: int,
        match_temperature: float,
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
        match_precision_floor: float = 0.01,
        match_precision_ceiling: float = 1.00,
        charbonnier_epsilon: float = 0.03,
        charbonnier_alpha: float = 0.45,
    ) -> None:
        super().__init__()
        self.radius = int(radius)
        self.match_temperature = float(match_temperature)
        self.match_precision_floor = float(match_precision_floor)
        self.match_precision_ceiling = float(match_precision_ceiling)
        self.charbonnier_epsilon = float(charbonnier_epsilon)
        self.charbonnier_alpha = float(charbonnier_alpha)

        self.reliability = OperatorReliabilityCalibrator(
            context_channels=context_channels,
            hidden_channels=reliability_hidden_channels,
            groups=groups,
            initial_validity=initial_validity,
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

    @staticmethod
    def initialise_state(flow: torch.Tensor) -> LMState:
        return LMState(w=flow, z=flow)

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
        measurement = local_correlation_measurement(
            correlation,
            state.w,
            radius=self.radius,
            temperature=self.match_temperature,
        )
        appearance_validity, precision_multiplier = self.reliability(
            source_context,
            linearisation,
            measurement,
            state.w,
            state.z,
        )
        confidence = measurement.confidence * linearisation.in_bounds
        base_precision = self.match_precision_floor + (
            self.match_precision_ceiling - self.match_precision_floor
        ) * confidence.square()
        match_precision = (
            base_precision * precision_multiplier * linearisation.in_bounds
        )

        normal = solve_optical_lm_increment(
            residual=linearisation.residual,
            jacobian_x=linearisation.jacobian_x,
            jacobian_y=linearisation.jacobian_y,
            appearance_validity=appearance_validity,
            flow_w=state.w,
            flow_z=state.z,
            match_proposal=measurement.proposal,
            match_precision=match_precision,
            beta_map=beta_map,
            damping_map=damping_map,
            charbonnier_epsilon=self.charbonnier_epsilon,
            charbonnier_alpha=self.charbonnier_alpha,
        )
        data_delta = bounded_vector(normal.delta, max_data_delta)
        data_state = state.w + data_delta

        data_confidence = torch.maximum(
            appearance_validity, confidence
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
            match_proposal=measurement.proposal,
            match_precision=match_precision,
            match_confidence=measurement.confidence,
            match_entropy=measurement.entropy,
            feature_residual=linearisation.residual,
            data_delta=data_delta,
            proximal_anchor=proximal_anchor,
            inverse_trace=normal.inverse_trace,
            condition=normal.condition,
            beta=beta,
            damping=damping,
            regularisation=regularisation,
        )


def convex_upsample_field(
    field: torch.Tensor,
    mask_logits: torch.Tensor,
    *,
    rate: int,
    scale_vectors: bool,
) -> torch.Tensor:
    """Convexly upsample a two- or three-dimensional vector field."""
    if field.ndim != 4:
        raise ValueError(f"field must be [B,C,H,W], got {tuple(field.shape)}")
    rate = int(rate)
    if rate < 1:
        raise ValueError(f"rate must be positive, got {rate}")
    batch, channels, height, width = field.shape
    expected = 9 * rate * rate
    if mask_logits.shape != (batch, expected, height, width):
        raise ValueError(
            "Invalid mask shape: expected "
            f"{(batch, expected, height, width)}, got {tuple(mask_logits.shape)}"
        )
    mask = torch.softmax(
        mask_logits.view(batch, 1, 9, rate, rate, height, width),
        dim=2,
    )
    scale = float(rate) if scale_vectors else 1.0
    neighbourhood = F.unfold(
        scale * field, kernel_size=3, padding=1
    ).view(batch, channels, 9, 1, 1, height, width)
    result = torch.sum(mask * neighbourhood, dim=2)
    return result.permute(0, 1, 4, 2, 5, 3).reshape(
        batch, channels, height * rate, width * rate
    )


class SourceGuidedFieldUpsampler(nn.Module):
    """Source-only convex upsampling for optical or metric scene flow."""

    def __init__(
        self,
        *,
        field_channels: int,
        context_channels: int,
        hidden_channels: int = 64,
        groups: int = 8,
        rate: int = 2,
        scale_vectors: bool,
    ) -> None:
        super().__init__()
        self.field_channels = int(field_channels)
        self.rate = int(rate)
        self.scale_vectors = bool(scale_vectors)
        self.mask_head = nn.Sequential(
            ConvGNAct(
                context_channels + self.field_channels + 1,
                hidden_channels,
                groups=groups,
            ),
            ConvGNAct(hidden_channels, hidden_channels, groups=groups),
            nn.Conv2d(hidden_channels, 9 * self.rate * self.rate, 1),
        )
        nn.init.zeros_(self.mask_head[-1].weight)
        nn.init.zeros_(self.mask_head[-1].bias)

    def forward(
        self,
        field: torch.Tensor,
        source_context: torch.Tensor,
        source_guidance: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        mask_logits = self.mask_head(
            torch.cat(
                (source_context, field, edge_magnitude(source_guidance)),
                dim=1,
            )
        )
        return (
            convex_upsample_field(
                field,
                mask_logits,
                rate=self.rate,
                scale_vectors=self.scale_vectors,
            ),
            mask_logits,
        )


__all__ = [
    "FeatureLinearisation",
    "HQSOpticalLMCell",
    "LMState",
    "LocalMatchMeasurement",
    "OperatorReliabilityCalibrator",
    "OpticalLMIterationOutput",
    "OpticalNormalEquation",
    "SourceGuidedFieldUpsampler",
    "SourceOnlyMotionProximal",
    "VectorEdgeAwareJacobiProx",
    "bounded_vector",
    "charbonnier_irls_weight",
    "convex_upsample_field",
    "linearise_feature_warp",
    "local_correlation_measurement",
    "positive_map",
    "solve_optical_lm_increment",
    "spatial_gradients",
]
