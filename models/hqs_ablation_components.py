"""Controlled HQSCore ablation cells for OF-A4 and OF-B1--B4.

The classes in this module deliberately leave the reference ``StructuredHQSCell``
untouched.  They reproduce its operator construction and initialisation, then
apply only the intervention declared by the experiment configuration.

OF-B variants
-------------
``AblatedStructuredHQSCell`` keeps the HQS split state and full reference
parameterisation, while selectively suppressing one analytic/learned operator:

* B1 analytic data only   -> learned data residual and its hidden update disabled.
* B2 learned data only    -> analytic data correction is exactly zero both in the
                             proposal and in the learned-data input.
* B3 analytic prox only   -> learned source-conditioned proximal residual disabled.
* B4 learned prox only    -> Jacobi anchor replaced by the identity ``q_bar = w``.

OF-A4
-----
``SingleStateOperatorCell`` removes the persistent ``w/q`` split.  A single
flow state is propagated between iterations, but the operator ordering remains

    flow -> warp/validity -> data proposal -> proximal regularisation -> flow'.

The coupling feature is removed from the learned data operator.  An active
residual bottleneck is inserted on that learned path and sized automatically so
that the recurrent-cell trainable parameter count remains within the requested
tolerance of the reference HQS cell.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from models.hqs_core_components import (
    ConvGNAct,
    EdgeAwareJacobiProx,
    HQSIterationOutput,
    HQSState,
    PositiveSchedule,
    RecurrentDataOperator,
    SeparableConvGRU,
    SharedValidityHead,
    SourceConditionedProxResidual,
    WarpLinearisation,
    edge_magnitude,
    weighted_analytic_data_delta,
)
from models.hqs_recurrent_control_components import (
    ActiveCapacityAdapter,
    count_trainable_parameters,
)


def _bounded_vector(vector: torch.Tensor, maximum: float) -> torch.Tensor:
    """Smooth vector-norm limiter identical to HQSCore's safeguard."""
    maximum = max(float(maximum), 1e-6)
    norm = torch.sqrt(vector.square().sum(dim=1, keepdim=True) + 1e-12)
    ratio = norm / maximum
    scale = torch.where(
        ratio > 1e-4,
        torch.tanh(ratio) / ratio,
        1.0 - ratio.square() / 3.0,
    )
    return vector * scale


def _analytic_delta(
    *,
    linearisation,
    validity: torch.Tensor,
    beta_map: torch.Tensor,
    max_data_delta: float,
    mode: str,
) -> torch.Tensor:
    """Reference HQSCore analytic data correction."""
    if mode == "weighted_solve":
        delta = weighted_analytic_data_delta(
            linearisation,
            validity,
            beta_map,
        )
    else:
        denominator = (
            linearisation.grad_x.square()
            + linearisation.grad_y.square()
            + beta_map
        ).clamp_min(1e-6)
        delta = torch.cat(
            (
                -linearisation.grad_x
                * linearisation.residual
                / denominator,
                -linearisation.grad_y
                * linearisation.residual
                / denominator,
            ),
            dim=1,
        )
    return _bounded_vector(delta, max_data_delta)


class AblatedStructuredHQSCell(nn.Module):
    """Reference HQS cell with explicit OF-B operator interventions.

    Every reference module remains instantiated so the nominal parameter count
    is unchanged.  Disabled operators are bypassed on the forward graph rather
    than deleted, making the independent variable the operator contribution
    itself rather than a coincident change in model size.
    """

    def __init__(
        self,
        *,
        correlation_channels: int,
        context_channels: int,
        hidden_channels: int,
        max_iterations: int,
        prior_hidden_channels: int = 64,
        groups: int = 8,
        beta_initial: float = 0.10,
        beta_minimum: float = 0.01,
        lambda_initial: float = 0.08,
        lambda_minimum: float = 0.001,
        edge_alpha: float = 10.0,
        analytic_weight: float = 1.0,
        learned_data_weight: float = 1.0,
        analytic_validity_mode: str = "post_gate",
        use_analytic_data: bool = True,
        use_learned_data: bool = True,
        analytic_input_to_learned: bool = True,
        use_analytic_prox: bool = True,
        use_learned_prox: bool = True,
    ) -> None:
        super().__init__()
        self.analytic_weight = float(analytic_weight)
        self.learned_data_weight = float(learned_data_weight)
        self.analytic_validity_mode = str(analytic_validity_mode).lower()
        if self.analytic_validity_mode not in {"post_gate", "weighted_solve"}:
            raise ValueError(
                "analytic_validity_mode must be 'post_gate' or 'weighted_solve'"
            )

        self.use_analytic_data = bool(use_analytic_data)
        self.use_learned_data = bool(use_learned_data)
        self.analytic_input_to_learned = bool(analytic_input_to_learned)
        self.use_analytic_prox = bool(use_analytic_prox)
        self.use_learned_prox = bool(use_learned_prox)

        if not (self.use_analytic_data or self.use_learned_data):
            raise ValueError("At least one data operator must remain active")
        if not (self.use_analytic_prox or self.use_learned_prox):
            raise ValueError("At least one proximal operator must remain active")
        if not self.use_analytic_data and self.analytic_input_to_learned:
            raise ValueError(
                "analytic_input_to_learned cannot be true when analytic data "
                "is disabled; B2 requires an exact zero analytic input"
            )

        # Instantiate exactly the same parameterised operators as A0.
        self.warp_linearisation = WarpLinearisation()
        self.data_operator = RecurrentDataOperator(
            correlation_channels,
            context_channels,
            hidden_channels,
            groups=groups,
        )
        self.analytic_prox = EdgeAwareJacobiProx(edge_alpha=edge_alpha)
        self.learned_prox = SourceConditionedProxResidual(
            context_channels,
            prior_hidden_channels,
            groups=groups,
        )
        self.beta_schedule = PositiveSchedule(
            max_iterations,
            beta_initial,
            beta_minimum,
        )
        self.lambda_schedule = PositiveSchedule(
            max_iterations,
            lambda_initial,
            lambda_minimum,
        )

    def initialise_state(
        self,
        source_context: torch.Tensor,
        flow_q: torch.Tensor,
    ) -> HQSState:
        hidden = self.data_operator.initialise_hidden(source_context)
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
        beta = self.beta_schedule(iteration).to(
            device=state.q.device,
            dtype=state.q.dtype,
        )
        regularisation = self.lambda_schedule(iteration).to(
            device=state.q.device,
            dtype=state.q.dtype,
        )
        beta_map = beta.view(1, 1, 1, 1).expand(
            state.q.shape[0], 1, *state.q.shape[-2:]
        )
        lambda_map = regularisation.view(1, 1, 1, 1).expand_as(beta_map)

        linearisation = self.warp_linearisation(
            source_gray,
            target_gray,
            target_grad_x,
            target_grad_y,
            state.q,
        )
        reliability = validity_head(
            correlation,
            linearisation,
            state.w,
            state.q,
        )
        if measurement_reliability is not None:
            if measurement_reliability.shape != reliability.shape:
                raise ValueError(
                    "measurement_reliability must match the cell grid: "
                    f"{tuple(measurement_reliability.shape)} versus "
                    f"{tuple(reliability.shape)}"
                )
            reliability = reliability * measurement_reliability.to(
                device=reliability.device,
                dtype=reliability.dtype,
            ).clamp(0.0, 1.0)
        validity = linearisation.in_bounds * reliability

        if self.use_analytic_data:
            analytic_delta = _analytic_delta(
                linearisation=linearisation,
                validity=validity,
                beta_map=beta_map,
                max_data_delta=max_data_delta,
                mode=self.analytic_validity_mode,
            )
        else:
            # B2: no analytic correction may enter either the proposal or the
            # learned residual's input tensor.
            analytic_delta = torch.zeros_like(state.q)

        if self.use_learned_data:
            learned_input_analytic = (
                analytic_delta
                if self.analytic_input_to_learned
                else torch.zeros_like(analytic_delta)
            )
            learned_delta, hidden_next = self.data_operator(
                correlation,
                source_context,
                state.w,
                state.q,
                linearisation,
                learned_input_analytic,
                beta_map,
                validity,
                state.hidden,
            )
            learned_proposal = (
                float(max_data_delta) * torch.tanh(learned_delta)
            )
        else:
            # B1: learned data correction is removed, including its recurrent
            # hidden-state evolution.  The data step is therefore analytic only.
            learned_delta = torch.zeros_like(state.q)
            learned_proposal = torch.zeros_like(state.q)
            hidden_next = state.hidden

        if self.analytic_validity_mode == "weighted_solve":
            proposal = (
                self.analytic_weight * analytic_delta
                + self.learned_data_weight * validity * learned_proposal
            )
            data_delta = _bounded_vector(proposal, max_data_delta)
            flow_w_next = state.q + data_delta
        else:
            proposal = (
                self.analytic_weight * analytic_delta
                + self.learned_data_weight * learned_proposal
            )
            data_delta = _bounded_vector(proposal, max_data_delta)
            flow_w_next = state.q + validity * data_delta

        if self.use_analytic_prox:
            proximal_anchor = self.analytic_prox(
                flow_w_next,
                source_gray,
                beta,
                regularisation,
                sweeps=jacobi_sweeps,
            )
        else:
            # B4: the analytic/Jacobi proximal anchor is the identity.  No
            # quantity produced by Jacobi sweeps enters the learned proximal.
            proximal_anchor = flow_w_next

        if self.use_learned_prox:
            q_next = self.learned_prox(
                proximal_anchor,
                flow_w_next,
                state.q,
                source_context,
                edge_magnitude(source_gray),
                beta_map,
                lambda_map,
                max_delta=max_prox_delta,
            )
        else:
            # B3: analytic proximal only.
            q_next = proximal_anchor

        return HQSIterationOutput(
            state=HQSState(w=flow_w_next, q=q_next, hidden=hidden_next),
            reliability=reliability,
            validity=validity,
            analytic_delta=analytic_delta,
            learned_delta=learned_delta,
            data_delta=data_delta,
            proximal_anchor=proximal_anchor,
            beta=beta,
            regularisation=regularisation,
        )


class SingleStateDataOperator(nn.Module):
    """Learned data residual with the persistent HQS coupling feature removed."""

    def __init__(
        self,
        correlation_channels: int,
        context_channels: int,
        hidden_channels: int,
        *,
        capacity_adapter_channels: int = 0,
        groups: int = 8,
    ) -> None:
        super().__init__()
        # corr + source context + current flow + residual + Ix/Iy + analytic
        # delta + beta + validity.  There is no persistent (w-q) input.
        in_channels = (
            int(correlation_channels)
            + int(context_channels)
            + 2
            + 1
            + 2
            + 2
            + 1
            + 1
        )
        self.input_projection = ConvGNAct(
            in_channels,
            hidden_channels,
            groups=groups,
        )
        self.hidden_initialiser = nn.Conv2d(
            context_channels,
            hidden_channels,
            3,
            padding=1,
        )
        self.gru = SeparableConvGRU(hidden_channels, hidden_channels)
        self.capacity_adapter = ActiveCapacityAdapter(
            hidden_channels,
            int(capacity_adapter_channels),
        )
        self.flow_head = nn.Sequential(
            ConvGNAct(hidden_channels, hidden_channels, groups=groups),
            nn.Conv2d(hidden_channels, 2, 3, padding=1),
        )
        nn.init.zeros_(self.flow_head[-1].weight)
        nn.init.zeros_(self.flow_head[-1].bias)

    def initialise_hidden(self, source_context: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.hidden_initialiser(source_context))

    def forward(
        self,
        correlation: torch.Tensor,
        source_context: torch.Tensor,
        flow: torch.Tensor,
        linearisation,
        analytic_delta: torch.Tensor,
        beta_map: torch.Tensor,
        validity: torch.Tensor,
        hidden: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        inputs = torch.cat(
            (
                correlation,
                source_context,
                flow,
                linearisation.residual,
                linearisation.grad_x,
                linearisation.grad_y,
                analytic_delta,
                beta_map,
                validity,
            ),
            dim=1,
        )
        projected = self.input_projection(inputs)
        candidate = self.gru(projected, hidden)
        hidden_gate = validity.to(dtype=candidate.dtype)
        hidden_next = hidden + hidden_gate * (candidate - hidden)
        active_features = self.capacity_adapter(hidden_next)
        return self.flow_head(active_features), hidden_next


class SingleStateOperatorCell(nn.Module):
    """OF-A4 single-persistent-state operator-composed recurrent cell."""

    def __init__(
        self,
        *,
        correlation_channels: int,
        context_channels: int,
        hidden_channels: int,
        max_iterations: int,
        prior_hidden_channels: int = 64,
        groups: int = 8,
        beta_initial: float = 0.10,
        beta_minimum: float = 0.01,
        lambda_initial: float = 0.08,
        lambda_minimum: float = 0.001,
        edge_alpha: float = 10.0,
        analytic_weight: float = 1.0,
        learned_data_weight: float = 1.0,
        analytic_validity_mode: str = "post_gate",
        capacity_adapter_channels: int = 0,
    ) -> None:
        super().__init__()
        self.analytic_weight = float(analytic_weight)
        self.learned_data_weight = float(learned_data_weight)
        self.analytic_validity_mode = str(analytic_validity_mode).lower()
        if self.analytic_validity_mode not in {"post_gate", "weighted_solve"}:
            raise ValueError(
                "analytic_validity_mode must be 'post_gate' or 'weighted_solve'"
            )

        self.warp_linearisation = WarpLinearisation()
        self.data_operator = SingleStateDataOperator(
            correlation_channels,
            context_channels,
            hidden_channels,
            capacity_adapter_channels=capacity_adapter_channels,
            groups=groups,
        )
        self.analytic_prox = EdgeAwareJacobiProx(edge_alpha=edge_alpha)
        self.learned_prox = SourceConditionedProxResidual(
            context_channels,
            prior_hidden_channels,
            groups=groups,
        )
        self.beta_schedule = PositiveSchedule(
            max_iterations,
            beta_initial,
            beta_minimum,
        )
        self.lambda_schedule = PositiveSchedule(
            max_iterations,
            lambda_initial,
            lambda_minimum,
        )

    def initialise_state(
        self,
        source_context: torch.Tensor,
        flow_q: torch.Tensor,
    ) -> HQSState:
        hidden = self.data_operator.initialise_hidden(source_context)
        # Parent HQSCore expects HQSState; w/q are aliases for one flow state.
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
        # There is one persistent flow variable.  q/w equality is enforced at
        # every iteration boundary and is not interpreted as HQS coupling.
        flow = state.q
        beta = self.beta_schedule(iteration).to(
            device=flow.device,
            dtype=flow.dtype,
        )
        regularisation = self.lambda_schedule(iteration).to(
            device=flow.device,
            dtype=flow.dtype,
        )
        beta_map = beta.view(1, 1, 1, 1).expand(
            flow.shape[0], 1, *flow.shape[-2:]
        )
        lambda_map = regularisation.view(1, 1, 1, 1).expand_as(beta_map)

        linearisation = self.warp_linearisation(
            source_gray,
            target_gray,
            target_grad_x,
            target_grad_y,
            flow,
        )
        # SharedValidityHead remains identical to A0, but receives the same
        # flow in both state slots.  Its coupling-norm feature is therefore
        # identically zero, as required when the split state is absent.
        reliability = validity_head(
            correlation,
            linearisation,
            flow,
            flow,
        )
        if measurement_reliability is not None:
            if measurement_reliability.shape != reliability.shape:
                raise ValueError(
                    "measurement_reliability must match the cell grid: "
                    f"{tuple(measurement_reliability.shape)} versus "
                    f"{tuple(reliability.shape)}"
                )
            reliability = reliability * measurement_reliability.to(
                device=reliability.device,
                dtype=reliability.dtype,
            ).clamp(0.0, 1.0)
        validity = linearisation.in_bounds * reliability

        analytic_delta = _analytic_delta(
            linearisation=linearisation,
            validity=validity,
            beta_map=beta_map,
            max_data_delta=max_data_delta,
            mode=self.analytic_validity_mode,
        )
        learned_delta, hidden_next = self.data_operator(
            correlation,
            source_context,
            flow,
            linearisation,
            analytic_delta,
            beta_map,
            validity,
            state.hidden,
        )
        learned_proposal = float(max_data_delta) * torch.tanh(learned_delta)

        if self.analytic_validity_mode == "weighted_solve":
            proposal = (
                self.analytic_weight * analytic_delta
                + self.learned_data_weight * validity * learned_proposal
            )
            data_delta = _bounded_vector(proposal, max_data_delta)
            data_proposal = flow + data_delta
        else:
            proposal = (
                self.analytic_weight * analytic_delta
                + self.learned_data_weight * learned_proposal
            )
            data_delta = _bounded_vector(proposal, max_data_delta)
            data_proposal = flow + validity * data_delta

        proximal_anchor = self.analytic_prox(
            data_proposal,
            source_gray,
            beta,
            regularisation,
            sweeps=jacobi_sweeps,
        )
        flow_next = self.learned_prox(
            proximal_anchor,
            data_proposal,
            flow,
            source_context,
            edge_magnitude(source_gray),
            beta_map,
            lambda_map,
            max_delta=max_prox_delta,
        )

        return HQSIterationOutput(
            state=HQSState(w=flow_next, q=flow_next, hidden=hidden_next),
            reliability=reliability,
            validity=validity,
            analytic_delta=analytic_delta,
            learned_delta=learned_delta,
            data_delta=data_delta,
            proximal_anchor=proximal_anchor,
            beta=beta,
            regularisation=regularisation,
        )


@dataclass(frozen=True)
class SingleStateCapacityReport:
    target_parameters: int
    matched_parameters: int
    relative_error: float
    capacity_adapter_channels: int

    def as_dict(self) -> Dict[str, object]:
        return {
            "target_parameters": self.target_parameters,
            "matched_parameters": self.matched_parameters,
            "relative_error": self.relative_error,
            "capacity_adapter_channels": self.capacity_adapter_channels,
        }


def build_capacity_matched_single_state_cell(
    *,
    target_parameters: int,
    correlation_channels: int,
    context_channels: int,
    hidden_channels: int,
    max_iterations: int,
    prior_hidden_channels: int,
    groups: int,
    beta_initial: float,
    beta_minimum: float,
    lambda_initial: float,
    lambda_minimum: float,
    edge_alpha: float,
    analytic_weight: float,
    learned_data_weight: float,
    analytic_validity_mode: str,
    adapter_max_channels: int = 128,
    tolerance: float = 0.01,
) -> Tuple[SingleStateOperatorCell, SingleStateCapacityReport]:
    """Match A4's active single-state cell to the corresponding A0 cell."""
    target_parameters = int(target_parameters)
    if target_parameters <= 0:
        raise ValueError("target_parameters must be positive")

    common = dict(
        correlation_channels=correlation_channels,
        context_channels=context_channels,
        hidden_channels=hidden_channels,
        max_iterations=max_iterations,
        prior_hidden_channels=prior_hidden_channels,
        groups=groups,
        beta_initial=beta_initial,
        beta_minimum=beta_minimum,
        lambda_initial=lambda_initial,
        lambda_minimum=lambda_minimum,
        edge_alpha=edge_alpha,
        analytic_weight=analytic_weight,
        learned_data_weight=learned_data_weight,
        analytic_validity_mode=analytic_validity_mode,
    )

    best: Optional[Tuple[float, SingleStateOperatorCell, SingleStateCapacityReport]] = None
    for adapter in range(0, int(adapter_max_channels) + 1):
        cell = SingleStateOperatorCell(
            capacity_adapter_channels=adapter,
            **common,
        )
        count = count_trainable_parameters(cell)
        relative = abs(count - target_parameters) / float(target_parameters)
        report = SingleStateCapacityReport(
            target_parameters=target_parameters,
            matched_parameters=count,
            relative_error=relative,
            capacity_adapter_channels=adapter,
        )
        if best is None or relative < best[0]:
            best = (relative, cell, report)

    if best is None:
        raise RuntimeError("Single-state capacity search produced no candidate")
    relative, cell, report = best
    if relative > float(tolerance):
        raise RuntimeError(
            "Could not match single-state cell capacity within tolerance: "
            f"target={target_parameters:,}, best={report.matched_parameters:,}, "
            f"error={100.0 * relative:.4f}% > {100.0 * tolerance:.4f}%"
        )
    return cell, report


__all__ = [
    "AblatedStructuredHQSCell",
    "SingleStateCapacityReport",
    "SingleStateDataOperator",
    "SingleStateOperatorCell",
    "build_capacity_matched_single_state_cell",
]
