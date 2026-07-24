"""Analytic data and proximal cell for HQS-LM scene flow."""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn

from .hqs_core_components import ConvGNAct, PositiveSchedule, edge_magnitude
from .hqs_lm_components import (
    LMState,
    OperatorReliabilityCalibrator,
    SourceOnlyMotionProximal,
    VectorEdgeAwareJacobiProx,
    bounded_vector,
    charbonnier_irls_weight,
    linearise_feature_warp,
    local_correlation_measurement,
    positive_map,
)
from .scene_flow_geometry import (
    DepthLinearisation,
    SceneProjection,
    feature_scene_jacobian,
    linearise_target_depth,
    project_scene_flow,
    projective_scene_jacobian,
)


class GeometryReliabilityCalibrator(nn.Module):
    """Calibrate depth-operator validity without emitting motion."""

    def __init__(
        self,
        *,
        context_channels: int,
        hidden_channels: int = 32,
        groups: int = 8,
        initial_validity: float = 0.90,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            ConvGNAct(
                context_channels + 4,
                hidden_channels,
                groups=groups,
            ),
            ConvGNAct(hidden_channels, hidden_channels, groups=groups),
            nn.Conv2d(hidden_channels, 1, 3, padding=1),
        )
        nn.init.zeros_(self.net[-1].weight)
        initial_validity = min(max(float(initial_validity), 1e-4), 1.0 - 1e-4)
        nn.init.constant_(
            self.net[-1].bias,
            math.log(initial_validity / (1.0 - initial_validity)),
        )

    def forward(
        self,
        source_context: torch.Tensor,
        depth_linearisation: DepthLinearisation,
        source_depth_valid: torch.Tensor,
        source_guidance: torch.Tensor,
    ) -> torch.Tensor:
        residual_abs = depth_linearisation.residual.abs()
        relative_residual = residual_abs / (
            depth_linearisation.warped_depth.abs() + 0.1
        )
        guidance_change = edge_magnitude(source_guidance)
        statistics = torch.cat(
            (
                residual_abs,
                relative_residual,
                source_depth_valid,
                guidance_change,
            ),
            dim=1,
        )
        return depth_linearisation.valid * torch.sigmoid(
            self.net(torch.cat((source_context, statistics), dim=1))
        )


@dataclass
class SceneNormalEquation:
    delta: torch.Tensor
    hessian: torch.Tensor
    inverse_diagonal: torch.Tensor
    inverse_trace: torch.Tensor
    condition: torch.Tensor
    appearance_weight: torch.Tensor
    geometry_weight: torch.Tensor


def solve_scene_lm_increment(
    *,
    appearance_residual: torch.Tensor,
    appearance_jacobian: torch.Tensor,
    appearance_validity: torch.Tensor,
    geometry_residual: torch.Tensor,
    geometry_jacobian: torch.Tensor,
    geometry_validity: torch.Tensor,
    induced_flow: torch.Tensor,
    projective_jacobian: torch.Tensor,
    match_proposal: torch.Tensor,
    match_precision: torch.Tensor,
    scene_flow_w: torch.Tensor,
    scene_flow_z: torch.Tensor,
    beta_map: torch.Tensor,
    damping_map: torch.Tensor,
    feature_weight: float = 1.0,
    geometry_weight: float = 1.0,
    charbonnier_epsilon: float = 0.03,
    charbonnier_alpha: float = 0.45,
    depth_charbonnier_epsilon: float = 0.10,
) -> SceneNormalEquation:
    """Solve the joint appearance, geometry and match ``3 x 3`` LM system."""
    if appearance_jacobian.ndim != 5 or appearance_jacobian.shape[2] != 3:
        raise ValueError("appearance_jacobian must be [B,C,3,H,W]")
    if geometry_jacobian.ndim != 5 or geometry_jacobian.shape[2] != 3:
        raise ValueError("geometry_jacobian must be [B,1,3,H,W]")
    if projective_jacobian.shape[1:3] != (2, 3):
        raise ValueError("projective_jacobian must be [B,2,3,H,W]")
    if scene_flow_w.shape != scene_flow_z.shape or scene_flow_w.shape[1] != 3:
        raise ValueError("Scene-flow split states must be aligned [B,3,H,W]")

    feature_channels = max(int(appearance_residual.shape[1]), 1)
    app_robust = charbonnier_irls_weight(
        appearance_residual,
        epsilon=charbonnier_epsilon,
        alpha=charbonnier_alpha,
    )
    app_weight = (
        float(feature_weight)
        * appearance_validity.clamp(0.0, 1.0)
        * app_robust
        / float(feature_channels)
    ).float()
    geom_robust = charbonnier_irls_weight(
        geometry_residual,
        epsilon=depth_charbonnier_epsilon,
        alpha=charbonnier_alpha,
    )
    geom_weight_map = (
        float(geometry_weight)
        * geometry_validity.clamp(0.0, 1.0)
        * geom_robust
    ).float()

    app_jacobian = appearance_jacobian.float()
    app_residual = appearance_residual.float()
    geometry_jacobian32 = geometry_jacobian.float()
    geometry_residual32 = geometry_residual.float()
    projective32 = projective_jacobian.float()

    hessian = torch.einsum(
        "bcjhw,bckhw,bchw->bjkhw",
        app_jacobian,
        app_jacobian,
        app_weight,
    )
    gradient = torch.einsum(
        "bcjhw,bchw,bchw->bjhw",
        app_jacobian,
        app_residual,
        app_weight,
    )
    hessian = hessian + torch.einsum(
        "bcjhw,bckhw,bchw->bjkhw",
        geometry_jacobian32,
        geometry_jacobian32,
        geom_weight_map,
    )
    gradient = gradient + torch.einsum(
        "bcjhw,bchw,bchw->bjhw",
        geometry_jacobian32,
        geometry_residual32,
        geom_weight_map,
    )

    precision = match_precision.float()
    if precision.shape[1] == 1:
        precision = precision.expand(-1, 2, -1, -1)
    match_residual = induced_flow.float() - match_proposal.float()
    hessian = hessian + torch.einsum(
        "bijhw,bikhw,bihw->bjkhw",
        projective32,
        projective32,
        precision,
    )
    gradient = gradient + torch.einsum(
        "bijhw,bihw,bihw->bjhw",
        projective32,
        match_residual,
        precision,
    )

    beta = beta_map.float()
    damping = damping_map.float()
    coupling = scene_flow_w.float() - scene_flow_z.float()
    gradient = gradient + beta * coupling
    diagonal = beta + damping
    identity_field = torch.eye(
        3, device=hessian.device, dtype=hessian.dtype
    ).view(1, 3, 3, 1, 1)
    hessian = hessian + diagonal.unsqueeze(2) * identity_field

    matrix = hessian.permute(0, 3, 4, 1, 2).contiguous()
    vector = gradient.permute(0, 2, 3, 1).contiguous().unsqueeze(-1)
    matrix = matrix + 1e-6 * identity_field.permute(0, 3, 4, 1, 2)
    cholesky = torch.linalg.cholesky(matrix)
    solution = torch.cholesky_solve(vector, cholesky)
    delta = -solution.squeeze(-1).permute(0, 3, 1, 2)

    inverse = torch.cholesky_inverse(cholesky)
    inverse_diagonal = torch.diagonal(
        inverse, dim1=-2, dim2=-1
    ).permute(0, 3, 1, 2)
    inverse_trace = inverse_diagonal.sum(dim=1, keepdim=True)
    eigenvalues = torch.linalg.eigvalsh(matrix)
    condition = (
        eigenvalues[..., -1] / eigenvalues[..., 0].clamp_min(1e-8)
    ).unsqueeze(1)

    dtype = scene_flow_w.dtype
    return SceneNormalEquation(
        delta=delta.to(dtype=dtype),
        hessian=hessian.to(dtype=dtype),
        inverse_diagonal=inverse_diagonal.to(dtype=dtype),
        inverse_trace=inverse_trace.to(dtype=dtype),
        condition=condition.to(dtype=dtype),
        appearance_weight=app_weight.to(dtype=dtype),
        geometry_weight=geom_weight_map.to(dtype=dtype),
    )


@dataclass
class SceneLMIterationOutput:
    state: LMState
    projection: SceneProjection
    depth_linearisation: DepthLinearisation
    appearance_validity: torch.Tensor
    geometry_validity: torch.Tensor
    data_confidence: torch.Tensor
    match_proposal: torch.Tensor
    match_precision: torch.Tensor
    match_confidence: torch.Tensor
    match_entropy: torch.Tensor
    feature_residual: torch.Tensor
    data_delta: torch.Tensor
    proximal_anchor: torch.Tensor
    inverse_diagonal: torch.Tensor
    inverse_trace: torch.Tensor
    condition: torch.Tensor
    beta: torch.Tensor
    damping: torch.Tensor
    regularisation: torch.Tensor


class HQSSceneLMCell(nn.Module):
    """One calibrated projective HQS-LM scene-flow iteration."""

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
        feature_weight: float = 1.0,
        geometry_weight: float = 1.0,
        charbonnier_epsilon: float = 0.03,
        charbonnier_alpha: float = 0.45,
        depth_charbonnier_epsilon: float = 0.10,
    ) -> None:
        super().__init__()
        self.radius = int(radius)
        self.match_temperature = float(match_temperature)
        self.match_precision_floor = float(match_precision_floor)
        self.match_precision_ceiling = float(match_precision_ceiling)
        self.feature_weight = float(feature_weight)
        self.geometry_weight = float(geometry_weight)
        self.charbonnier_epsilon = float(charbonnier_epsilon)
        self.charbonnier_alpha = float(charbonnier_alpha)
        self.depth_charbonnier_epsilon = float(depth_charbonnier_epsilon)

        self.appearance_reliability = OperatorReliabilityCalibrator(
            context_channels=context_channels,
            hidden_channels=reliability_hidden_channels,
            groups=groups,
            initial_validity=initial_validity,
        )
        self.geometry_reliability = GeometryReliabilityCalibrator(
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
            state_channels=3,
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
        depth1: torch.Tensor,
        depth2: torch.Tensor,
        intrinsics: torch.Tensor,
        transform_21: torch.Tensor,
        state: LMState,
        iteration: int,
        jacobi_sweeps: int,
        max_data_delta: float,
        max_prox_delta: float,
    ) -> SceneLMIterationOutput:
        beta = self.beta_schedule(iteration)
        damping = self.damping_schedule(iteration)
        regularisation = self.lambda_schedule(iteration)
        beta_map = positive_map(beta, state.w)
        damping_map = positive_map(damping, state.w)
        lambda_map = positive_map(regularisation, state.w)

        projection = project_scene_flow(
            state.w, depth1, intrinsics, transform_21
        )
        feature_linearisation = linearise_feature_warp(
            source_features,
            target_features,
            target_grad_x,
            target_grad_y,
            projection.induced_flow,
        )
        measurement = local_correlation_measurement(
            correlation,
            projection.induced_flow,
            radius=self.radius,
            temperature=self.match_temperature,
        )
        appearance_validity, precision_multiplier = (
            self.appearance_reliability(
                source_context,
                feature_linearisation,
                measurement,
                state.w,
                state.z,
            )
        )
        appearance_validity = appearance_validity * projection.valid

        projective_jacobian = projective_scene_jacobian(
            projection.target_points, intrinsics, transform_21
        )
        appearance_jacobian = feature_scene_jacobian(
            feature_linearisation.jacobian_x,
            feature_linearisation.jacobian_y,
            projective_jacobian,
        )
        depth_linearisation = linearise_target_depth(
            depth2,
            projection,
            projective_jacobian,
            transform_21,
        )
        geometry_validity = self.geometry_reliability(
            source_context,
            depth_linearisation,
            (depth1 > 0.0).to(dtype=depth1.dtype),
            source_guidance,
        )

        match_confidence = (
            measurement.confidence
            * projection.valid
            * feature_linearisation.in_bounds
        )
        base_precision = self.match_precision_floor + (
            self.match_precision_ceiling - self.match_precision_floor
        ) * match_confidence.square()
        match_precision = (
            base_precision
            * precision_multiplier
            * projection.valid
            * feature_linearisation.in_bounds
        )

        normal = solve_scene_lm_increment(
            appearance_residual=feature_linearisation.residual,
            appearance_jacobian=appearance_jacobian,
            appearance_validity=appearance_validity,
            geometry_residual=depth_linearisation.residual,
            geometry_jacobian=depth_linearisation.jacobian,
            geometry_validity=geometry_validity,
            induced_flow=projection.induced_flow,
            projective_jacobian=projective_jacobian,
            match_proposal=measurement.proposal,
            match_precision=match_precision,
            scene_flow_w=state.w,
            scene_flow_z=state.z,
            beta_map=beta_map,
            damping_map=damping_map,
            feature_weight=self.feature_weight,
            geometry_weight=self.geometry_weight,
            charbonnier_epsilon=self.charbonnier_epsilon,
            charbonnier_alpha=self.charbonnier_alpha,
            depth_charbonnier_epsilon=self.depth_charbonnier_epsilon,
        )
        data_delta = bounded_vector(normal.delta, max_data_delta)
        data_state = state.w + data_delta
        data_confidence = torch.maximum(
            torch.maximum(appearance_validity, geometry_validity),
            match_confidence,
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

        return SceneLMIterationOutput(
            state=LMState(w=data_state, z=proximal_state),
            projection=projection,
            depth_linearisation=depth_linearisation,
            appearance_validity=appearance_validity,
            geometry_validity=geometry_validity,
            data_confidence=data_confidence,
            match_proposal=measurement.proposal,
            match_precision=match_precision,
            match_confidence=measurement.confidence,
            match_entropy=measurement.entropy,
            feature_residual=feature_linearisation.residual,
            data_delta=data_delta,
            proximal_anchor=proximal_anchor,
            inverse_diagonal=normal.inverse_diagonal,
            inverse_trace=normal.inverse_trace,
            condition=normal.condition,
            beta=beta,
            damping=damping,
            regularisation=regularisation,
        )


__all__ = [
    "GeometryReliabilityCalibrator",
    "HQSSceneLMCell",
    "SceneLMIterationOutput",
    "SceneNormalEquation",
    "solve_scene_lm_increment",
]
