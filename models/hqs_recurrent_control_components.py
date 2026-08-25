"""Equal-capacity unconstrained recurrent control for HQSCore ablations.

The control deliberately preserves the *evidence* available to HQSCore while
removing the HQS hypothesis-class restriction. It receives correlation, source
context, the current flow, warped photometric residual/gradients, geometric
in-bounds information, source edges and the learned reliability diagnostic, then
maps them through a generic recurrent update directly to the next flow.

No auxiliary split variable is used semantically, no analytic data solve is
performed, no data/proximal decomposition exists, and reliability is not a hard
multiplicative routing constraint. ``HQSState.q`` is retained only as a parent-
class compatibility alias and is always exactly equal to ``HQSState.w``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Mapping, Optional, Tuple

import torch
import torch.nn as nn

from models.hqs_core_components import (
    ConvGNAct,
    HQSIterationOutput,
    HQSState,
    ResidualGNBlock,
    SeparableConvGRU,
    SharedValidityHead,
    WarpLinearisation,
    edge_magnitude,
)


def count_trainable_parameters(module: nn.Module) -> int:
    return int(sum(p.numel() for p in module.parameters() if p.requires_grad))


def _bounded_vector(vector: torch.Tensor, maximum: float) -> torch.Tensor:
    """Smooth vector-norm limiter matching HQSCore's numerical safeguard."""
    maximum = max(float(maximum), 1e-6)
    norm = torch.sqrt(vector.square().sum(dim=1, keepdim=True) + 1e-12)
    ratio = norm / maximum
    scale = torch.where(
        ratio > 1e-4,
        torch.tanh(ratio) / ratio,
        1.0 - ratio.square() / 3.0,
    )
    return vector * scale


class ActiveCapacityAdapter(nn.Module):
    """Small *active* residual bottleneck used only to close a parameter gap.

    This is intentionally part of the forward path; the equal-capacity control
    does not use dummy or disconnected parameters.
    """

    def __init__(self, channels: int, bottleneck_channels: int) -> None:
        super().__init__()
        bottleneck_channels = int(bottleneck_channels)
        if bottleneck_channels <= 0:
            self.net = nn.Identity()
            self.enabled = False
        else:
            self.net = nn.Sequential(
                nn.Conv2d(channels, bottleneck_channels, 1),
                nn.SiLU(inplace=True),
                nn.Conv2d(bottleneck_channels, channels, 1),
            )
            # Start near identity while preserving an active trainable path.
            nn.init.zeros_(self.net[-1].weight)
            nn.init.zeros_(self.net[-1].bias)
            self.enabled = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.enabled:
            return x
        return x + self.net(x)


class MatchedUnconstrainedRecurrentCell(nn.Module):
    """Generic fused recurrent update used for OF-A1/OF-A2.

    Interface-compatible with :class:`StructuredHQSCell` so the *same* HQSCore
    multiscale front/back-end can be used without duplicating model code.
    """

    def __init__(
        self,
        *,
        correlation_channels: int,
        context_channels: int,
        hidden_channels: int,
        max_iterations: int,
        residual_blocks: int = 2,
        capacity_adapter_channels: int = 0,
        groups: int = 8,
    ) -> None:
        super().__init__()
        self.hidden_channels = int(hidden_channels)
        self.max_iterations = max(int(max_iterations), 1)

        # corr + source context + current flow + residual + Ix/Iy + in-bounds
        # + learned reliability diagnostic + source edge + iteration fraction.
        in_channels = (
            int(correlation_channels)
            + int(context_channels)
            + 2
            + 1
            + 2
            + 1
            + 1
            + 1
            + 1
        )
        self.warp_linearisation = WarpLinearisation()
        self.input_projection = ConvGNAct(
            in_channels,
            self.hidden_channels,
            groups=groups,
        )
        self.hidden_initialiser = nn.Conv2d(
            int(context_channels), self.hidden_channels, 3, padding=1
        )
        self.gru = SeparableConvGRU(self.hidden_channels, self.hidden_channels)
        self.residual_trunk = nn.Sequential(
            *[
                ResidualGNBlock(self.hidden_channels, groups=groups)
                for _ in range(int(residual_blocks))
            ]
        )
        self.capacity_adapter = ActiveCapacityAdapter(
            self.hidden_channels, int(capacity_adapter_channels)
        )
        self.flow_head = nn.Sequential(
            ConvGNAct(self.hidden_channels, self.hidden_channels, groups=groups),
            nn.Conv2d(self.hidden_channels, 2, 3, padding=1),
        )
        nn.init.zeros_(self.flow_head[-1].weight)
        nn.init.zeros_(self.flow_head[-1].bias)

    def initialise_state(
        self,
        source_context: torch.Tensor,
        flow_q: torch.Tensor,
    ) -> HQSState:
        hidden = torch.tanh(self.hidden_initialiser(source_context))
        # q is a compatibility alias only. It is not an independent state.
        return HQSState(w=flow_q, q=flow_q, hidden=hidden)

    def forward(
        self,
        *,
        source_gray: torch.Tensor,
        target_gray: torch.Tensor,
        target_grad_x: torch.Tensor,
        target_grad_y: torch.Tensor,
        correlation: torch.Tensor,
        source_context: torch.Tensor,
        state: HQSState,
        validity_head: SharedValidityHead,
        iteration: int,
        jacobi_sweeps: int,
        max_data_delta: float,
        max_prox_delta: float,
        measurement_reliability: torch.Tensor | None = None,
    ) -> HQSIterationOutput:
        del jacobi_sweeps, max_prox_delta

        # Direct-flow state: do not allow a latent q/w split to emerge through
        # the compatibility interface.
        flow = state.q
        linearisation = self.warp_linearisation(
            source_gray,
            target_gray,
            target_grad_x,
            target_grad_y,
            flow,
        )

        # Reliability remains observable/supervisable so A0 and A1 can receive
        # equivalent auxiliary supervision. Crucially, it is an INPUT FEATURE,
        # not a hard routing constraint on the recurrent flow update.
        reliability = validity_head(
            correlation,
            linearisation,
            flow,
            flow,
        )
        if measurement_reliability is not None:
            # This external measurement diagnostic may still be exposed as an
            # input feature, but it must not impose A0's multiplicative routing.
            reliability_feature = reliability * measurement_reliability.to(
                device=reliability.device,
                dtype=reliability.dtype,
            ).clamp(0.0, 1.0)
        else:
            reliability_feature = reliability

        source_edge = edge_magnitude(source_gray)
        iteration_fraction = torch.full_like(
            reliability,
            float(iteration) / float(max(self.max_iterations - 1, 1)),
        )
        fused = torch.cat(
            (
                correlation,
                source_context,
                flow,
                linearisation.residual,
                linearisation.grad_x,
                linearisation.grad_y,
                linearisation.in_bounds,
                reliability_feature,
                source_edge,
                iteration_fraction,
            ),
            dim=1,
        )
        projected = self.input_projection(fused)
        hidden_next = self.gru(projected, state.hidden)
        features = self.residual_trunk(hidden_next)
        features = self.capacity_adapter(features)
        raw_delta = self.flow_head(features)
        proposal = float(max_data_delta) * torch.tanh(raw_delta)
        delta = _bounded_vector(proposal, max_data_delta)
        flow_next = flow + delta

        # The output schema is retained for evaluator/trainer compatibility.
        # These aliases must not be interpreted as HQS quantities for A1/A2.
        zeros = torch.zeros_like(delta)
        scalar_zero = flow_next.new_zeros(())
        validity_diagnostic = linearisation.in_bounds * reliability
        return HQSIterationOutput(
            state=HQSState(w=flow_next, q=flow_next, hidden=hidden_next),
            reliability=reliability,
            validity=validity_diagnostic,
            analytic_delta=zeros,
            learned_delta=delta,
            data_delta=delta,
            proximal_anchor=flow_next,
            beta=scalar_zero,
            regularisation=scalar_zero,
        )


@dataclass(frozen=True)
class CapacityMatchReport:
    target_parameters: int
    matched_parameters: int
    relative_error: float
    hidden_channels: int
    residual_blocks: int
    capacity_adapter_channels: int

    def as_dict(self) -> Dict[str, object]:
        return {
            "target_parameters": self.target_parameters,
            "matched_parameters": self.matched_parameters,
            "relative_error": self.relative_error,
            "hidden_channels": self.hidden_channels,
            "residual_blocks": self.residual_blocks,
            "capacity_adapter_channels": self.capacity_adapter_channels,
        }


def build_capacity_matched_recurrent_cell(
    *,
    target_parameters: int,
    correlation_channels: int,
    context_channels: int,
    max_iterations: int,
    groups: int,
    hidden_min: int = 48,
    hidden_max: int = 192,
    hidden_step: int = 8,
    residual_blocks_min: int = 0,
    residual_blocks_max: int = 6,
    adapter_max_channels: int = 2048,
    tolerance: float = 0.01,
) -> Tuple[MatchedUnconstrainedRecurrentCell, CapacityMatchReport]:
    """Search for an *active* recurrent architecture within the target budget."""
    target_parameters = int(target_parameters)
    if target_parameters <= 0:
        raise ValueError("target_parameters must be positive")

    best: Optional[Tuple[float, MatchedUnconstrainedRecurrentCell, CapacityMatchReport]] = None

    for hidden in range(int(hidden_min), int(hidden_max) + 1, int(hidden_step)):
        for blocks in range(int(residual_blocks_min), int(residual_blocks_max) + 1):
            # Search adapter widths. First consider no adapter, then estimate
            # the bottleneck needed to close the remaining parameter gap.
            base = MatchedUnconstrainedRecurrentCell(
                correlation_channels=correlation_channels,
                context_channels=context_channels,
                hidden_channels=hidden,
                max_iterations=max_iterations,
                residual_blocks=blocks,
                capacity_adapter_channels=0,
                groups=groups,
            )
            base_count = count_trainable_parameters(base)
            remaining = target_parameters - base_count

            adapter_candidates = {0}
            # Adapter parameter count is approximately 2*h*b + b + h.
            if remaining > hidden:
                estimate = int(round((remaining - hidden) / float(2 * hidden + 1)))
                for candidate in range(max(1, estimate - 3), min(adapter_max_channels, estimate + 3) + 1):
                    adapter_candidates.add(candidate)

            for adapter in sorted(adapter_candidates):
                cell = MatchedUnconstrainedRecurrentCell(
                    correlation_channels=correlation_channels,
                    context_channels=context_channels,
                    hidden_channels=hidden,
                    max_iterations=max_iterations,
                    residual_blocks=blocks,
                    capacity_adapter_channels=adapter,
                    groups=groups,
                )
                count = count_trainable_parameters(cell)
                relative = abs(count - target_parameters) / float(target_parameters)
                report = CapacityMatchReport(
                    target_parameters=target_parameters,
                    matched_parameters=count,
                    relative_error=relative,
                    hidden_channels=hidden,
                    residual_blocks=blocks,
                    capacity_adapter_channels=adapter,
                )
                if best is None or relative < best[0]:
                    best = (relative, cell, report)

    if best is None:
        raise RuntimeError("Capacity search produced no candidate recurrent cell")
    relative, cell, report = best
    if relative > float(tolerance):
        raise RuntimeError(
            "Could not match recurrent-control capacity within tolerance: "
            f"target={target_parameters:,}, best={report.matched_parameters:,}, "
            f"error={100.0 * relative:.3f}% > {100.0 * tolerance:.3f}%"
        )
    return cell, report


__all__ = [
    "CapacityMatchReport",
    "MatchedUnconstrainedRecurrentCell",
    "build_capacity_matched_recurrent_cell",
    "count_trainable_parameters",
]
