"""Optimal-transport measurement operators for structured optical flow.

This module contains the clean-room operators shared by ``HQS-OTOF`` and
``HQS-Field-OFv2``.  The implementation follows the mathematical description
of FlowIt's initial matching stage, but it does not reproduce FlowIt's learned
refinement network:

* transformer-enhanced pyramid features are fused on a selected grid;
* every source/target feature pair participates in a dustbin-augmented,
  entropy-regularised transport problem;
* local expectation around the transport peak produces a flow observation;
* confidence, observability, entropy and covariance are derived from mass;
* flow changes only through an analytic HQS data solve and a source-only
  positive-semidefinite graph proximal.

The all-pairs score volume is computed once; on CUDA it is cached in FP16 by
default. Sinkhorn dual updates and plan decoding remain blockwise in FP32, so
the full correlation and full-precision transport plan are never stored
simultaneously.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .hqs_core_components import ConvGNAct, PositiveSchedule
from .hqs_lm_components import LMState, bounded_vector, positive_map
from .warp import flow_in_bounds_mask, resize_flow


def _logit(probability: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    probability = probability.float().clamp(eps, 1.0 - eps)
    return probability.log() - (1.0 - probability).log()


def _coords(
    height: int,
    width: int,
    *,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    y, x = torch.meshgrid(
        torch.arange(height, device=device, dtype=dtype),
        torch.arange(width, device=device, dtype=dtype),
        indexing="ij",
    )
    return torch.stack((x.reshape(-1), y.reshape(-1)), dim=-1)


@dataclass
class TransportMeasurement:
    """Decoded statistics of a dustbin-augmented transport plan."""

    flow: torch.Tensor
    topk_flows: torch.Tensor
    topk_logits: torch.Tensor
    topk_probabilities: torch.Tensor
    confidence: torch.Tensor
    observability: torch.Tensor
    dustbin_probability: torch.Tensor
    entropy: torch.Tensor
    peak_mass: torch.Tensor
    retained_mass: torch.Tensor
    precision: torch.Tensor
    dual_source: torch.Tensor
    dual_target: torch.Tensor


@dataclass
class TransportNormalEquation:
    """Result of the transport-conditioned analytic HQS data update."""

    delta: torch.Tensor
    proposal: torch.Tensor
    precision: torch.Tensor
    support: torch.Tensor
    inverse_trace: torch.Tensor
    condition: torch.Tensor


@dataclass
class TransportFieldIterationOutput:
    """One transport-data and semantic-field proximal iteration."""

    state: LMState
    data_delta: torch.Tensor
    proposal: torch.Tensor
    precision: torch.Tensor
    measurement_support: torch.Tensor
    inverse_trace: torch.Tensor
    condition: torch.Tensor
    beta: torch.Tensor
    damping: torch.Tensor
    regularisation: torch.Tensor
    proximal_inertia: torch.Tensor


class HierarchicalTransportFusion(nn.Module):
    """Content-gated cross-scale fusion of transformer-enhanced features.

    The input features are expected to have already passed through the shared
    per-scale self/cross-transformers used by ``HQSFieldOpticalFlow``.  Each
    scale is projected into a common space, resized to the requested transport
    grid and fused with content-dependent softmax gates.  The same module is
    applied to both frames, preserving the Siamese matching representation.
    """

    def __init__(
        self,
        channels_by_scale: Dict[int, int],
        *,
        scales: Sequence[int] = (4, 8, 16),
        output_channels: int = 128,
        groups: int = 8,
    ) -> None:
        super().__init__()
        self.scales = tuple(int(scale) for scale in scales)
        if not self.scales:
            raise ValueError("At least one transport fusion scale is required")
        missing = set(self.scales).difference(channels_by_scale)
        if missing:
            raise ValueError(f"Missing feature channels for scales {missing}")
        self.projections = nn.ModuleDict(
            {
                str(scale): ConvGNAct(
                    int(channels_by_scale[scale]),
                    int(output_channels),
                    kernel_size=1,
                    groups=groups,
                )
                for scale in self.scales
            }
        )
        self.gates = nn.ModuleDict(
            {
                str(scale): nn.Conv2d(
                    int(output_channels), 1, kernel_size=1, bias=True
                )
                for scale in self.scales
            }
        )
        self.refine = nn.Sequential(
            ConvGNAct(
                int(output_channels),
                int(output_channels),
                groups=groups,
            ),
            nn.Conv2d(
                int(output_channels),
                int(output_channels),
                kernel_size=3,
                padding=1,
                bias=False,
            ),
        )

    def forward(
        self,
        features: Dict[int, torch.Tensor],
        *,
        target_scale: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        target_scale = int(target_scale)
        if target_scale not in features:
            raise ValueError(f"Missing target-scale feature 1/{target_scale}")
        size = features[target_scale].shape[-2:]
        projected = []
        gate_logits = []
        for scale in self.scales:
            value = self.projections[str(scale)](features[scale])
            if value.shape[-2:] != size:
                value = F.interpolate(
                    value,
                    size=size,
                    mode="bilinear",
                    align_corners=False,
                )
            projected.append(value)
            gate_logits.append(self.gates[str(scale)](value))
        weights = torch.softmax(torch.stack(gate_logits, dim=1), dim=1)
        fused = sum(
            weights[:, index] * value
            for index, value in enumerate(projected)
        )
        fused = fused + self.refine(fused)
        return F.normalize(fused, dim=1, eps=1e-6), weights


class BlockwiseDustbinSinkhorn(nn.Module):
    """All-pairs entropy-regularised transport with an unmatched dustbin.

    The real-real all-pairs correlation matrix is computed once and cached in
    FP16 on CUDA by default. Row and column log-sum-exp reductions and plan
    decoding are blockwise in FP32, so Sinkhorn does not repeatedly recompute
    the expensive dot products or materialise another full-precision plan.
    """

    def __init__(
        self,
        *,
        temperature: float = 0.07,
        sinkhorn_iterations: int = 8,
        query_chunk_size: int = 128,
        local_expectation_radius: int = 2,
        num_hypotheses: int = 4,
        initial_dustbin_score: float = 1.0,
        state_sigma: float = 3.0,
        state_cost_clip: float = 50.0,
        covariance_floor: float = 0.05,
        precision_maximum: float = 4.0,
        maximum_tokens: int = 20000,
        gradient_checkpointing: bool = True,
        cache_all_pairs_scores: bool = True,
        half_precision_score_cache: bool = True,
    ) -> None:
        super().__init__()
        self.temperature = max(float(temperature), 1e-4)
        self.sinkhorn_iterations = max(int(sinkhorn_iterations), 1)
        self.query_chunk_size = max(int(query_chunk_size), 1)
        self.local_expectation_radius = max(
            int(local_expectation_radius), 0
        )
        self.num_hypotheses = max(int(num_hypotheses), 1)
        self.state_sigma = max(float(state_sigma), 1e-4)
        self.state_cost_clip = max(float(state_cost_clip), 0.0)
        self.covariance_floor = max(float(covariance_floor), 1e-5)
        self.precision_maximum = max(float(precision_maximum), 1e-4)
        self.maximum_tokens = max(int(maximum_tokens), 1)
        self.gradient_checkpointing = bool(gradient_checkpointing)
        self.cache_all_pairs_scores = bool(cache_all_pairs_scores)
        self.half_precision_score_cache = bool(
            half_precision_score_cache
        )
        self.dustbin_score = nn.Parameter(
            torch.tensor(float(initial_dustbin_score), dtype=torch.float32)
        )

    def _checkpoint(self, function, *arguments):
        enabled = (
            self.training
            and self.gradient_checkpointing
            and torch.is_grad_enabled()
            and any(
                isinstance(value, torch.Tensor) and value.requires_grad
                for value in arguments
            )
        )
        if enabled:
            return checkpoint(
                function,
                *arguments,
                use_reentrant=False,
                preserve_rng_state=False,
            )
        return function(*arguments)

    def _score_block(
        self,
        source: torch.Tensor,
        target: torch.Tensor,
        target_coordinates: torch.Tensor,
        predicted_target: torch.Tensor,
        state_weight: torch.Tensor,
    ) -> torch.Tensor:
        score = torch.bmm(source, target.transpose(1, 2))
        score = score / self.temperature
        displacement = (
            predicted_target.unsqueeze(2)
            - target_coordinates.view(1, 1, -1, 2)
        )
        state_cost = 0.5 * displacement.square().sum(dim=-1)
        state_cost = state_cost / (self.state_sigma * self.state_sigma)
        if self.state_cost_clip > 0:
            state_cost = state_cost.clamp_max(self.state_cost_clip)
        return score - state_weight * state_cost

    def _row_logsumexp(
        self,
        source_chunk: torch.Tensor,
        target: torch.Tensor,
        target_coordinates: torch.Tensor,
        predicted_target: torch.Tensor,
        state_weight: torch.Tensor,
        dual_target_real: torch.Tensor,
        dual_target_bin: torch.Tensor,
        score_block: Optional[torch.Tensor],
    ) -> torch.Tensor:
        score = (
            score_block.float()
            if score_block is not None
            else self._score_block(
                source_chunk,
                target,
                target_coordinates,
                predicted_target,
                state_weight,
            )
        )
        real = score + dual_target_real.unsqueeze(1)
        bin_column = (
            self.dustbin_score.float() + dual_target_bin
        ).unsqueeze(1).expand(-1, source_chunk.shape[1], -1)
        return torch.logsumexp(torch.cat((real, bin_column), dim=-1), dim=-1)

    def _column_logsumexp(
        self,
        source: torch.Tensor,
        target_chunk: torch.Tensor,
        target_coordinates_chunk: torch.Tensor,
        predicted_target: torch.Tensor,
        state_weight: torch.Tensor,
        dual_source_real: torch.Tensor,
        dual_source_bin: torch.Tensor,
        score_block: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if score_block is None:
            # Evaluate the same score matrix with target columns chunked.
            score = torch.bmm(source, target_chunk.transpose(1, 2))
            score = score / self.temperature
            displacement = (
                predicted_target.unsqueeze(2)
                - target_coordinates_chunk.view(1, 1, -1, 2)
            )
            state_cost = 0.5 * displacement.square().sum(dim=-1)
            state_cost = state_cost / (
                self.state_sigma * self.state_sigma
            )
            if self.state_cost_clip > 0:
                state_cost = state_cost.clamp_max(self.state_cost_clip)
            score = score - state_weight * state_cost
        else:
            score = score_block.float()
        real = score + dual_source_real.unsqueeze(-1)
        bin_row = (
            self.dustbin_score.float() + dual_source_bin
        ).unsqueeze(-1).expand(-1, -1, target_chunk.shape[1])
        return torch.logsumexp(torch.cat((real, bin_row), dim=1), dim=1)

    def _decode_block(
        self,
        source_chunk: torch.Tensor,
        target: torch.Tensor,
        source_coordinates: torch.Tensor,
        target_coordinates: torch.Tensor,
        predicted_target: torch.Tensor,
        state_weight: torch.Tensor,
        dual_source_chunk: torch.Tensor,
        dual_target_real: torch.Tensor,
        dual_target_bin: torch.Tensor,
        normalisation: torch.Tensor,
        height_tensor: torch.Tensor,
        width_tensor: torch.Tensor,
        score_block: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, ...]:
        score = (
            score_block.float()
            if score_block is not None
            else self._score_block(
                source_chunk,
                target,
                target_coordinates,
                predicted_target,
                state_weight,
            )
        )
        log_plan = (
            score
            + dual_source_chunk.unsqueeze(-1)
            + dual_target_real.unsqueeze(1)
            - normalisation
        )
        plan = torch.exp(log_plan.clamp(min=-80.0, max=20.0))
        real_mass = plan.sum(dim=-1)
        dustbin = torch.exp(
            (
                self.dustbin_score.float()
                + dual_source_chunk
                + dual_target_bin
                - normalisation
            ).clamp(min=-80.0, max=20.0)
        )

        hypotheses = min(self.num_hypotheses, plan.shape[-1])
        top_probability, top_index = plan.topk(k=hypotheses, dim=-1)
        top_logit = torch.gather(log_plan, dim=-1, index=top_index)
        top_coordinates = target_coordinates[top_index]
        top_flows = top_coordinates - source_coordinates.unsqueeze(2)

        height = int(height_tensor.item())
        width = int(width_tensor.item())
        peak_index = top_index[..., 0]
        peak_y = torch.div(peak_index, width, rounding_mode="floor")
        peak_x = peak_index.remainder(width)
        radius = self.local_expectation_radius
        offset_y, offset_x = torch.meshgrid(
            torch.arange(
                -radius,
                radius + 1,
                device=plan.device,
                dtype=torch.long,
            ),
            torch.arange(
                -radius,
                radius + 1,
                device=plan.device,
                dtype=torch.long,
            ),
            indexing="ij",
        )
        offset_y = offset_y.reshape(1, 1, -1)
        offset_x = offset_x.reshape(1, 1, -1)
        candidate_y_raw = peak_y.unsqueeze(-1) + offset_y
        candidate_x_raw = peak_x.unsqueeze(-1) + offset_x
        valid = (
            (candidate_y_raw >= 0)
            & (candidate_y_raw < height)
            & (candidate_x_raw >= 0)
            & (candidate_x_raw < width)
        )
        candidate_y = candidate_y_raw.clamp(0, height - 1)
        candidate_x = candidate_x_raw.clamp(0, width - 1)
        candidate_index = candidate_y * width + candidate_x
        local_probability = torch.gather(
            plan, dim=-1, index=candidate_index
        ) * valid.to(plan)
        local_mass = local_probability.sum(dim=-1)
        denominator = local_mass.clamp_min(1e-8)
        expected_x = (
            local_probability * candidate_x.to(plan)
        ).sum(dim=-1) / denominator
        expected_y = (
            local_probability * candidate_y.to(plan)
        ).sum(dim=-1) / denominator
        peak_coordinates = top_coordinates[..., 0, :]
        expected_x = torch.where(
            local_mass > 1e-8, expected_x, peak_coordinates[..., 0]
        )
        expected_y = torch.where(
            local_mass > 1e-8, expected_y, peak_coordinates[..., 1]
        )
        expected = torch.stack((expected_x, expected_y), dim=-1)
        flow = expected - source_coordinates

        conditional = plan / real_mass.unsqueeze(-1).clamp_min(1e-8)
        entropy = -(
            conditional * conditional.clamp_min(1e-9).log()
        ).sum(dim=-1) / math.log(max(plan.shape[-1], 2))

        centered_x = candidate_x.to(plan) - expected_x.unsqueeze(-1)
        centered_y = candidate_y.to(plan) - expected_y.unsqueeze(-1)
        variance_x = (
            local_probability * centered_x.square()
        ).sum(dim=-1) / denominator + self.covariance_floor
        variance_y = (
            local_probability * centered_y.square()
        ).sum(dim=-1) / denominator + self.covariance_floor
        covariance = (
            local_probability * centered_x * centered_y
        ).sum(dim=-1) / denominator
        determinant = (
            variance_x * variance_y - covariance.square()
        ).clamp_min(self.covariance_floor * self.covariance_floor)
        p11 = variance_y / determinant
        p12 = -covariance / determinant
        p22 = variance_x / determinant
        trace = (p11 + p22).clamp_min(1e-8)
        trace_scale = torch.clamp(self.precision_maximum / trace, max=1.0)
        precision = torch.stack((p11, p12, p22), dim=-1)
        # Mass support is applied exactly once by the analytic data solve.
        # Keeping covariance precision separate avoids unintentionally
        # squaring confidence/observability attenuation.
        precision = precision * trace_scale.unsqueeze(-1)

        return (
            flow,
            top_flows,
            top_logit,
            top_probability,
            local_mass.clamp(0.0, 1.0),
            real_mass.clamp(0.0, 1.0),
            dustbin.clamp(0.0, 1.0),
            entropy.clamp(0.0, 1.0),
            top_probability[..., 0],
            top_probability.sum(dim=-1).clamp(0.0, 1.0),
            precision,
        )

    def forward(
        self,
        source_features: torch.Tensor,
        target_features: torch.Tensor,
        *,
        state_flow: Optional[torch.Tensor] = None,
        state_weight: float = 0.0,
    ) -> TransportMeasurement:
        if source_features.shape != target_features.shape:
            raise ValueError(
                "Transport features must have identical shapes, got "
                f"{tuple(source_features.shape)} and "
                f"{tuple(target_features.shape)}"
            )
        if source_features.ndim != 4:
            raise ValueError("Transport features must be [B,C,H,W]")
        batch, _, height, width = source_features.shape
        tokens = height * width
        if tokens > self.maximum_tokens:
            raise RuntimeError(
                f"Transport grid contains {tokens} tokens, exceeding "
                f"maximum_tokens={self.maximum_tokens}. Reduce the training "
                "crop or increase hqs_* .ot_maximum_tokens deliberately."
            )
        if state_flow is not None and state_flow.shape != (
            batch,
            2,
            height,
            width,
        ):
            state_flow = resize_flow(state_flow, (height, width))

        source = F.normalize(
            source_features.float(), dim=1, eps=1e-6
        ).flatten(2).transpose(1, 2)
        target = F.normalize(
            target_features.float(), dim=1, eps=1e-6
        ).flatten(2).transpose(1, 2)
        coordinates = _coords(
            height, width, device=source.device, dtype=torch.float32
        )
        source_coordinates = coordinates.view(1, tokens, 2).expand(
            batch, -1, -1
        )
        if state_flow is None:
            predicted_target = source_coordinates
        else:
            predicted_target = (
                source_coordinates
                + state_flow.float().flatten(2).transpose(1, 2)
            )
        state_weight_tensor = source.new_tensor(float(state_weight))

        normalisation = source.new_tensor(-math.log(2.0 * tokens))
        log_real_mass = normalisation
        log_bin_mass = source.new_tensor(
            math.log(tokens) - math.log(2.0 * tokens)
        )
        dual_source = source.new_zeros(batch, tokens + 1)
        dual_target = source.new_zeros(batch, tokens + 1)
        chunk = self.query_chunk_size
        score_cache = None
        if self.cache_all_pairs_scores:
            score_parts = []
            for start in range(0, tokens, chunk):
                stop = min(start + chunk, tokens)
                score_parts.append(
                    self._score_block(
                        source[:, start:stop],
                        target,
                        coordinates,
                        predicted_target[:, start:stop],
                        state_weight_tensor,
                    )
                )
            score_cache = torch.cat(score_parts, dim=1)
            del score_parts
            if (
                self.half_precision_score_cache
                and score_cache.device.type == "cuda"
            ):
                score_cache = score_cache.to(dtype=torch.float16)

        for _ in range(self.sinkhorn_iterations):
            source_updates = []
            for start in range(0, tokens, chunk):
                stop = min(start + chunk, tokens)
                row_lse = self._checkpoint(
                    self._row_logsumexp,
                    source[:, start:stop],
                    target,
                    coordinates,
                    predicted_target[:, start:stop],
                    state_weight_tensor,
                    dual_target[:, :tokens],
                    dual_target[:, tokens : tokens + 1],
                    (
                        None
                        if score_cache is None
                        else score_cache[:, start:stop]
                    ),
                )
                source_updates.append(log_real_mass - row_lse)
            dustbin_row_lse = torch.logsumexp(
                self.dustbin_score.float() + dual_target, dim=-1
            )
            dual_source = torch.cat(
                (
                    torch.cat(source_updates, dim=1),
                    (log_bin_mass - dustbin_row_lse).unsqueeze(1),
                ),
                dim=1,
            )

            target_updates = []
            for start in range(0, tokens, chunk):
                stop = min(start + chunk, tokens)
                column_lse = self._checkpoint(
                    self._column_logsumexp,
                    source,
                    target[:, start:stop],
                    coordinates[start:stop],
                    predicted_target,
                    state_weight_tensor,
                    dual_source[:, :tokens],
                    dual_source[:, tokens : tokens + 1],
                    (
                        None
                        if score_cache is None
                        else score_cache[:, :, start:stop]
                    ),
                )
                target_updates.append(log_real_mass - column_lse)
            dustbin_column_lse = torch.logsumexp(
                self.dustbin_score.float() + dual_source, dim=-1
            )
            dual_target = torch.cat(
                (
                    torch.cat(target_updates, dim=1),
                    (log_bin_mass - dustbin_column_lse).unsqueeze(1),
                ),
                dim=1,
            )

        decoded = [[] for _ in range(11)]
        height_tensor = source.new_tensor(height, dtype=torch.long)
        width_tensor = source.new_tensor(width, dtype=torch.long)
        for start in range(0, tokens, chunk):
            stop = min(start + chunk, tokens)
            values = self._checkpoint(
                self._decode_block,
                source[:, start:stop],
                target,
                source_coordinates[:, start:stop],
                coordinates,
                predicted_target[:, start:stop],
                state_weight_tensor,
                dual_source[:, start:stop],
                dual_target[:, :tokens],
                dual_target[:, tokens : tokens + 1],
                normalisation,
                height_tensor,
                width_tensor,
                (
                    None
                    if score_cache is None
                    else score_cache[:, start:stop]
                ),
            )
            for destination, value in zip(decoded, values):
                destination.append(value)
        values = [torch.cat(parts, dim=1) for parts in decoded]
        (
            flow,
            topk_flows,
            topk_logits,
            topk_probabilities,
            confidence,
            observability,
            dustbin_probability,
            entropy,
            peak_mass,
            retained_mass,
            precision,
        ) = values

        dtype = source_features.dtype

        def scalar_map(value: torch.Tensor) -> torch.Tensor:
            return value.reshape(batch, 1, height, width).to(dtype=dtype)

        hypotheses = topk_flows.shape[2]
        return TransportMeasurement(
            flow=flow.transpose(1, 2).reshape(
                batch, 2, height, width
            ).to(dtype=dtype),
            topk_flows=topk_flows.permute(0, 2, 3, 1).reshape(
                batch, hypotheses, 2, height, width
            ).to(dtype=dtype),
            topk_logits=topk_logits.permute(0, 2, 1).reshape(
                batch, hypotheses, height, width
            ).to(dtype=dtype),
            topk_probabilities=topk_probabilities.permute(
                0, 2, 1
            ).reshape(
                batch, hypotheses, height, width
            ).to(dtype=dtype),
            confidence=scalar_map(confidence),
            observability=scalar_map(observability),
            dustbin_probability=scalar_map(dustbin_probability),
            entropy=scalar_map(entropy),
            peak_mass=scalar_map(peak_mass),
            retained_mass=scalar_map(retained_mass),
            precision=precision.permute(0, 2, 1).reshape(
                batch, 3, height, width
            ).to(dtype=dtype),
            dual_source=dual_source,
            dual_target=dual_target,
        )


def resize_transport_measurement(
    measurement: TransportMeasurement,
    size: Tuple[int, int],
) -> TransportMeasurement:
    """Resize transport statistics while preserving vector units."""
    size = (int(size[0]), int(size[1]))
    if measurement.flow.shape[-2:] == size:
        return measurement
    input_height, input_width = measurement.flow.shape[-2:]
    scale_x = float(size[1]) / float(input_width)
    scale_y = float(size[0]) / float(input_height)
    topk = torch.stack(
        [
            resize_flow(measurement.topk_flows[:, index], size)
            for index in range(measurement.topk_flows.shape[1])
        ],
        dim=1,
    )

    def linear(value: torch.Tensor) -> torch.Tensor:
        return F.interpolate(
            value, size=size, mode="bilinear", align_corners=False
        )

    precision = linear(measurement.precision).clone()
    precision[:, 0] /= scale_x * scale_x
    precision[:, 1] /= scale_x * scale_y
    precision[:, 2] /= scale_y * scale_y
    return TransportMeasurement(
        flow=resize_flow(measurement.flow, size),
        topk_flows=topk,
        topk_logits=linear(measurement.topk_logits),
        topk_probabilities=linear(measurement.topk_probabilities),
        confidence=linear(measurement.confidence).clamp(0.0, 1.0),
        observability=linear(measurement.observability).clamp(0.0, 1.0),
        dustbin_probability=linear(
            measurement.dustbin_probability
        ).clamp(0.0, 1.0),
        entropy=linear(measurement.entropy).clamp(0.0, 1.0),
        peak_mass=linear(measurement.peak_mass).clamp(0.0, 1.0),
        retained_mass=linear(measurement.retained_mass).clamp(0.0, 1.0),
        precision=precision,
        dual_source=measurement.dual_source,
        dual_target=measurement.dual_target,
    )


def transport_to_correlation_mixture(
    measurement: TransportMeasurement,
    *,
    size: Tuple[int, int],
    num_hypotheses: int,
):
    """Adapt OT modes to the existing analytic mixture-cell interface."""
    from .hqs_field_components import CorrelationMixture

    resized = resize_transport_measurement(measurement, size)
    requested = int(num_hypotheses)
    available = resized.topk_flows.shape[1]
    if requested > available:
        raise ValueError(
            f"OT produced {available} modes but {requested} were requested"
        )
    proposals = resized.topk_flows[:, :requested]
    logits = resized.topk_logits[:, :requested]
    probabilities = resized.topk_probabilities[:, :requested]
    peak = probabilities[:, 0:1]
    if requested > 1:
        margin = peak - probabilities[:, 1:2]
    else:
        margin = peak
    retained = probabilities.sum(dim=1, keepdim=True).clamp(0.0, 1.0)
    in_bounds = torch.stack(
        [
            flow_in_bounds_mask(proposals[:, index])
            for index in range(requested)
        ],
        dim=1,
    )
    return CorrelationMixture(
        proposals=proposals,
        offsets=proposals,
        logits=logits,
        probabilities=probabilities,
        in_bounds=in_bounds.to(proposals),
        confidence=resized.confidence,
        entropy=resized.entropy,
        margin=margin,
        peak=peak,
        retained_mass=retained,
    )


def solve_transport_hqs_increment(
    *,
    flow_w: torch.Tensor,
    flow_z: torch.Tensor,
    proposal: torch.Tensor,
    precision: torch.Tensor,
    support: torch.Tensor,
    beta_map: torch.Tensor,
    damping_map: torch.Tensor,
) -> TransportNormalEquation:
    """Solve a transport-moment data term coupled to the HQS split state."""
    if not (
        flow_w.shape == flow_z.shape == proposal.shape
        and flow_w.shape[1] == 2
    ):
        raise ValueError("Flow states and proposal must align as [B,2,H,W]")
    if precision.shape != (
        flow_w.shape[0],
        3,
        flow_w.shape[2],
        flow_w.shape[3],
    ):
        raise ValueError("precision must be [B,3,H,W]")
    w = flow_w.float()
    z = flow_z.float()
    mu = proposal.float()
    support32 = support.float().clamp(0.0, 1.0)
    p11 = precision[:, 0:1].float() * support32
    p12 = precision[:, 1:2].float() * support32
    p22 = precision[:, 2:3].float() * support32
    beta = beta_map.float()
    damping = damping_map.float()
    h11 = p11 + beta + damping
    h12 = p12
    h22 = p22 + beta + damping
    error_x = w[:, 0:1] - mu[:, 0:1]
    error_y = w[:, 1:2] - mu[:, 1:2]
    g1 = (
        p11 * error_x
        + p12 * error_y
        + beta * (w[:, 0:1] - z[:, 0:1])
    )
    g2 = (
        p12 * error_x
        + p22 * error_y
        + beta * (w[:, 1:2] - z[:, 1:2])
    )
    determinant_raw = h11 * h22 - h12.square()
    determinant = torch.maximum(
        determinant_raw,
        (1e-6 * h11 * h22).clamp_min(1e-8),
    )
    delta_x = -(h22 * g1 - h12 * g2) / determinant
    delta_y = -(-h12 * g1 + h11 * g2) / determinant
    discriminant = torch.sqrt(
        (h11 - h22).square() + 4.0 * h12.square() + 1e-12
    )
    eigen_max = 0.5 * (h11 + h22 + discriminant)
    eigen_min = 0.5 * (h11 + h22 - discriminant)
    measurement_trace = p11 + p22
    effective_support = (
        measurement_trace
        / (measurement_trace + 2.0 * (beta + damping)).clamp_min(1e-8)
    ).clamp(0.0, 1.0)
    dtype = flow_w.dtype
    return TransportNormalEquation(
        delta=torch.cat((delta_x, delta_y), dim=1).to(dtype=dtype),
        proposal=proposal,
        precision=torch.cat((p11, p12, p22), dim=1).to(dtype=dtype),
        support=effective_support.to(dtype=dtype),
        inverse_trace=((h11 + h22) / determinant).clamp_max(1e6).to(
            dtype=dtype
        ),
        condition=(eigen_max / eigen_min.clamp_min(1e-8)).clamp_max(
            1e6
        ).to(dtype=dtype),
    )


class TransportPrecisionCalibrator(nn.Module):
    """Learn scalar transport trust without predicting a flow vector."""

    def __init__(
        self,
        context_channels: int,
        *,
        hidden_channels: int = 64,
        groups: int = 8,
        maximum_log_precision_adjustment: float = 2.0,
        maximum_observability_logit_adjustment: float = 2.0,
    ) -> None:
        super().__init__()
        self.maximum_log_precision_adjustment = float(
            maximum_log_precision_adjustment
        )
        self.maximum_observability_logit_adjustment = float(
            maximum_observability_logit_adjustment
        )
        # Context + data/proximal residual (2) + confidence, observability,
        # entropy and iteration fraction (4).
        self.body = nn.Sequential(
            ConvGNAct(
                int(context_channels) + 6,
                int(hidden_channels),
                groups=groups,
            ),
            ConvGNAct(
                int(hidden_channels),
                int(hidden_channels),
                groups=groups,
            ),
            nn.Conv2d(int(hidden_channels), 2, kernel_size=3, padding=1),
        )
        nn.init.zeros_(self.body[-1].weight)
        nn.init.zeros_(self.body[-1].bias)

    def forward(
        self,
        *,
        source_context: torch.Tensor,
        state: LMState,
        measurement: TransportMeasurement,
        iteration_fraction: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        residual = state.w - measurement.flow
        fraction = state.w.new_full(
            (state.w.shape[0], 1, *state.w.shape[-2:]),
            float(iteration_fraction),
        )
        features = torch.cat(
            (
                source_context,
                residual,
                measurement.confidence,
                measurement.observability,
                measurement.entropy,
                fraction,
            ),
            dim=1,
        )
        adjustment = self.body(features)
        log_precision = (
            torch.tanh(adjustment[:, 0:1])
            * self.maximum_log_precision_adjustment
        )
        observability_adjustment = (
            torch.tanh(adjustment[:, 1:2])
            * self.maximum_observability_logit_adjustment
        )
        calibrated_support = torch.sigmoid(
            _logit(measurement.observability)
            + observability_adjustment.float()
        ).to(measurement.observability)
        calibrated_support = torch.sqrt(
            (
                calibrated_support * measurement.confidence
            ).clamp_min(0.0)
        ).clamp(0.0, 1.0)
        calibrated_precision = (
            measurement.precision.float() * log_precision.float().exp()
        ).to(measurement.precision)
        return calibrated_precision, calibrated_support


class SourceSemanticGraphProximal(nn.Module):
    """Source-only symmetric semantic graph field proximal.

    Non-negative symmetric affinities define a graph Laplacian
    ``L = D - A``.  Consequently the fixed-affinity field subproblem is convex
    whenever the configured anchor or inertia is positive.
    """

    def __init__(
        self,
        *,
        context_channels: int,
        embedding_channels: int = 32,
        groups: int = 8,
        neighbours: int = 8,
        feature_temperature: float = 0.10,
        anchor_floor: float = 0.05,
        maximum_tokens: int = 4096,
        gradient_checkpointing: bool = True,
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
        self.neighbours = max(int(neighbours), 1)
        self.feature_temperature = max(float(feature_temperature), 1e-4)
        self.anchor_floor = min(max(float(anchor_floor), 0.0), 1.0)
        self.maximum_tokens = max(int(maximum_tokens), 1)
        self.gradient_checkpointing = bool(gradient_checkpointing)

    def prepare(self, source_context: torch.Tensor) -> torch.Tensor:
        with torch.autocast(
            device_type=source_context.device.type, enabled=False
        ):
            return F.normalize(
                self.embedding(source_context.float()), dim=1, eps=1e-6
            )

    def _adjacency(self, source_embedding: torch.Tensor) -> torch.Tensor:
        flat = source_embedding.flatten(2).transpose(1, 2)
        similarity = torch.bmm(flat, flat.transpose(1, 2))
        tokens = similarity.shape[-1]
        diagonal = torch.eye(
            tokens, device=similarity.device, dtype=torch.bool
        ).unsqueeze(0)
        similarity_for_topk = similarity.masked_fill(diagonal, -1e4)
        neighbours = min(self.neighbours, max(tokens - 1, 1))
        indices = similarity_for_topk.topk(
            k=neighbours, dim=-1
        ).indices
        selected = torch.gather(similarity, dim=-1, index=indices)
        weights = torch.exp(
            (selected - 1.0) / self.feature_temperature
        )
        adjacency = torch.zeros_like(similarity).scatter(
            dim=-1, index=indices, src=weights
        )
        adjacency = 0.5 * (adjacency + adjacency.transpose(1, 2))
        return adjacency.masked_fill(diagonal, 0.0)

    def forward(
        self,
        *,
        data_state: torch.Tensor,
        previous_proximal: torch.Tensor,
        source_embedding: torch.Tensor,
        measurement_support: torch.Tensor,
        beta_map: torch.Tensor,
        regularisation_map: torch.Tensor,
        inertia_map: torch.Tensor,
        sweeps: int,
    ) -> torch.Tensor:
        if data_state.shape != previous_proximal.shape:
            raise ValueError("Semantic proximal states must align")
        batch, _, height, width = data_state.shape
        tokens = height * width
        if tokens > self.maximum_tokens:
            raise RuntimeError(
                f"Semantic graph has {tokens} tokens, exceeding "
                f"maximum_tokens={self.maximum_tokens}; use this proximal only "
                "on configured coarse scales."
            )
        adjacency = self._adjacency(source_embedding.float())
        degree = adjacency.sum(dim=-1, keepdim=True)
        support = measurement_support.float().flatten(2).transpose(1, 2)
        beta = beta_map.float().flatten(2).transpose(1, 2)
        inertia = inertia_map.float().flatten(2).transpose(1, 2)
        regularisation = (
            regularisation_map.float().flatten(2).transpose(1, 2)
        )
        anchor = beta * (
            self.anchor_floor + (1.0 - self.anchor_floor) * support
        )
        data = data_state.float().flatten(2).transpose(1, 2)
        previous = previous_proximal.float().flatten(2).transpose(1, 2)
        current = previous
        for _ in range(max(int(sweeps), 1)):
            neighbour_sum = torch.bmm(adjacency, current)
            current = (
                anchor * data
                + inertia * previous
                + regularisation * neighbour_sum
            ) / (
                anchor + inertia + regularisation * degree
            ).clamp_min(1e-6)
        return current.transpose(1, 2).reshape(
            batch, 2, height, width
        ).to(data_state)


class TransportFieldCell(nn.Module):
    """Transport-conditioned analytic data solve and semantic field proximal."""

    def __init__(
        self,
        *,
        context_channels: int,
        max_iterations: int,
        groups: int = 8,
        calibrator_hidden_channels: int = 64,
        graph_embedding_channels: int = 32,
        graph_neighbours: int = 8,
        graph_feature_temperature: float = 0.10,
        graph_anchor_floor: float = 0.05,
        graph_maximum_tokens: int = 4096,
        beta_initial: float = 0.10,
        beta_minimum: float = 0.01,
        damping_initial: float = 0.10,
        damping_minimum: float = 0.001,
        lambda_initial: float = 0.08,
        lambda_minimum: float = 0.001,
        proximal_inertia_initial: float = 0.05,
        proximal_inertia_minimum: float = 0.001,
        maximum_log_precision_adjustment: float = 2.0,
        maximum_observability_logit_adjustment: float = 2.0,
    ) -> None:
        super().__init__()
        self.max_iterations = int(max_iterations)
        self.calibrator = TransportPrecisionCalibrator(
            context_channels,
            hidden_channels=calibrator_hidden_channels,
            groups=groups,
            maximum_log_precision_adjustment=(
                maximum_log_precision_adjustment
            ),
            maximum_observability_logit_adjustment=(
                maximum_observability_logit_adjustment
            ),
        )
        self.field_proximal = SourceSemanticGraphProximal(
            context_channels=context_channels,
            embedding_channels=graph_embedding_channels,
            groups=groups,
            neighbours=graph_neighbours,
            feature_temperature=graph_feature_temperature,
            anchor_floor=graph_anchor_floor,
            maximum_tokens=graph_maximum_tokens,
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

    def prepare_source_graph(
        self, source_context: torch.Tensor
    ) -> torch.Tensor:
        return self.field_proximal.prepare(source_context)

    def forward(
        self,
        *,
        measurement: TransportMeasurement,
        source_context: torch.Tensor,
        source_graph_embedding: torch.Tensor,
        state: LMState,
        iteration: int,
        graph_sweeps: int,
        max_data_delta: float,
    ) -> TransportFieldIterationOutput:
        if measurement.flow.shape != state.w.shape:
            measurement = resize_transport_measurement(
                measurement, state.w.shape[-2:]
            )
        beta = self.beta_schedule(iteration)
        damping = self.damping_schedule(iteration)
        regularisation = self.lambda_schedule(iteration)
        inertia = self.inertia_schedule(iteration)
        beta_map = positive_map(beta, state.w)
        damping_map = positive_map(damping, state.w)
        lambda_map = positive_map(regularisation, state.w)
        inertia_map = positive_map(inertia, state.w)
        precision, support = self.calibrator(
            source_context=source_context,
            state=state,
            measurement=measurement,
            iteration_fraction=(
                float(iteration) / float(max(self.max_iterations - 1, 1))
            ),
        )
        normal = solve_transport_hqs_increment(
            flow_w=state.w,
            flow_z=state.z,
            proposal=measurement.flow,
            precision=precision,
            support=support,
            beta_map=beta_map,
            damping_map=damping_map,
        )
        data_delta = bounded_vector(normal.delta, max_data_delta)
        data_state = state.w + data_delta
        proximal_state = self.field_proximal(
            data_state=data_state,
            previous_proximal=state.z,
            source_embedding=source_graph_embedding,
            measurement_support=normal.support,
            beta_map=beta_map,
            regularisation_map=lambda_map,
            inertia_map=inertia_map,
            sweeps=graph_sweeps,
        )
        return TransportFieldIterationOutput(
            state=LMState(w=data_state, z=proximal_state),
            data_delta=data_delta,
            proposal=measurement.flow,
            precision=normal.precision,
            measurement_support=normal.support,
            inverse_trace=normal.inverse_trace,
            condition=normal.condition,
            beta=beta,
            damping=damping,
            regularisation=regularisation,
            proximal_inertia=inertia,
        )


__all__ = [
    "BlockwiseDustbinSinkhorn",
    "HierarchicalTransportFusion",
    "SourceSemanticGraphProximal",
    "TransportFieldCell",
    "TransportFieldIterationOutput",
    "TransportMeasurement",
    "TransportNormalEquation",
    "resize_transport_measurement",
    "solve_transport_hqs_increment",
    "transport_to_correlation_mixture",
]
