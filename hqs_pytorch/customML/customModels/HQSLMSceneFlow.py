"""HQS-LM-SF: calibrated RGB-D scene-flow prototype.

The model estimates metric scene displacement ``S`` in the source-camera
coordinate system from ``(I1, I2, D1, D2, K, T_21)``.  Its data subproblem
combines:

* multi-channel feature-warp residuals;
* target-depth consistency;
* precision-weighted 2D correspondence observations;
* HQS consensus with a source-conditioned proximal variable.

All motion corrections are produced by a differentiable, damped ``3 x 3``
normal-equation solve.  The learned proximal receives only source-frame image
and depth context plus scalar confidence/conditioning diagnostics.
"""
from __future__ import annotations

from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.correlation import LocalCorrBlock
from models.hqs_core_components import (
    AllPairsCorrelation,
    ConvGNAct,
    HQSCorePyramidEncoder,
    spatial_gradients,
)
from models.hqs_lm_components import (
    LMState,
    SourceGuidedFieldUpsampler,
)
from models.hqs_lm_scene_components import HQSSceneLMCell
from models.scene_flow_geometry import (
    identity_transform,
    lift_flow_to_scene_flow,
    project_scene_flow,
    resize_metric_field,
    scale_intrinsics,
    validate_intrinsics,
    validate_transform,
)
from models.warp import backward_warp


def _cfg_get(cfg, key: str, default):
    if cfg is None:
        return default
    if isinstance(cfg, Mapping):
        return cfg.get(key, default)
    try:
        return cfg.get(key, default)
    except Exception:
        return getattr(cfg, key, default)


def _list(
    cfg,
    key: str,
    default: Sequence,
    cast,
) -> Tuple:
    value = _cfg_get(cfg, key, default)
    result = tuple(cast(item) for item in value)
    if len(result) != len(default):
        raise ValueError(
            f"hqs_lm_sf.{key} must contain {len(default)} entries, got {value}"
        )
    return result


class HQSLMSceneFlow(nn.Module):
    """Four-scale calibrated projective HQS-LM scene-flow estimator."""

    scale_order: Tuple[int, ...] = (16, 8, 4, 2)

    def __init__(self, cfg) -> None:
        super().__init__()
        model_cfg = _cfg_get(cfg, "hqs_lm_sf", cfg)
        self.cfg = model_cfg

        feature_channels = _list(
            model_cfg, "feature_channels", (32, 64, 96, 128), int
        )
        match_channels = _list(
            model_cfg, "match_channels", (64, 96, 128, 128), int
        )
        context_channels = _list(
            model_cfg, "context_channels", (64, 64, 96, 96), int
        )
        blocks_per_scale = _list(
            model_cfg, "blocks_per_scale", (1, 2, 2, 2), int
        )
        groups = int(_cfg_get(model_cfg, "groups", 8))
        depth_context_channels = int(
            _cfg_get(model_cfg, "depth_context_channels", 16)
        )

        iterations = _list(
            model_cfg, "iterations", (2, 4, 2, 2), int
        )
        if any(value < 1 for value in iterations):
            raise ValueError(
                f"All HQS-LM-SF iteration counts must be positive: {iterations}"
            )
        self.iterations_by_scale = dict(zip(self.scale_order, iterations))
        self.num_hqs_iterations = sum(iterations)
        self.jacobi_sweeps = dict(
            zip(
                self.scale_order,
                _list(model_cfg, "jacobi_sweeps", (2, 2, 1, 1), int),
            )
        )
        self.max_data_delta = dict(
            zip(
                self.scale_order,
                _list(
                    model_cfg,
                    "max_data_delta_metres",
                    (1.0, 0.75, 0.50, 0.25),
                    float,
                ),
            )
        )
        self.max_prox_delta = dict(
            zip(
                self.scale_order,
                _list(
                    model_cfg,
                    "max_prox_delta_metres",
                    (0.25, 0.20, 0.10, 0.05),
                    float,
                ),
            )
        )
        self.correlation_radii = dict(
            zip(
                self.scale_order,
                _list(
                    model_cfg, "correlation_radii", (4, 4, 4, 3), int
                ),
            )
        )
        self.match_temperatures = dict(
            zip(
                self.scale_order,
                _list(
                    model_cfg,
                    "match_temperatures",
                    (0.07, 0.07, 0.10, 0.10),
                    float,
                ),
            )
        )
        levels = _list(model_cfg, "all_pairs_levels", (1, 4), int)
        self.all_pairs_levels = {16: levels[0], 8: levels[1]}

        self.feature_encoder = HQSCorePyramidEncoder(
            feature_channels=feature_channels,
            match_channels=match_channels,
            context_channels=context_channels,
            blocks_per_scale=blocks_per_scale,
            groups=groups,
        )
        context_by_scale = dict(zip((2, 4, 8, 16), context_channels))
        self.depth_context = nn.ModuleDict()
        self.context_fusion = nn.ModuleDict()
        for scale in self.scale_order:
            self.depth_context[str(scale)] = nn.Sequential(
                ConvGNAct(
                    2,
                    depth_context_channels,
                    groups=groups,
                ),
                ConvGNAct(
                    depth_context_channels,
                    depth_context_channels,
                    groups=groups,
                ),
            )
            self.context_fusion[str(scale)] = ConvGNAct(
                context_by_scale[scale] + depth_context_channels,
                context_by_scale[scale],
                groups=groups,
            )

        local_chunk = int(
            _cfg_get(model_cfg, "local_corr_channel_chunk", 4)
        )
        local_checkpoint = bool(
            _cfg_get(model_cfg, "local_corr_checkpoint", True)
        )
        self.local_correlations = nn.ModuleDict(
            {
                "4": LocalCorrBlock(
                    radius=self.correlation_radii[4],
                    channel_chunk_size=local_chunk,
                    checkpoint_chunks=local_checkpoint,
                ),
                "2": LocalCorrBlock(
                    radius=self.correlation_radii[2],
                    channel_chunk_size=local_chunk,
                    checkpoint_chunks=local_checkpoint,
                ),
            }
        )

        common = dict(
            prior_hidden_channels=int(
                _cfg_get(model_cfg, "prior_hidden_channels", 64)
            ),
            reliability_hidden_channels=int(
                _cfg_get(model_cfg, "reliability_hidden_channels", 32)
            ),
            groups=groups,
            beta_initial=float(
                _cfg_get(model_cfg, "beta_initial", 0.10)
            ),
            beta_minimum=float(
                _cfg_get(model_cfg, "beta_minimum", 0.01)
            ),
            damping_initial=float(
                _cfg_get(model_cfg, "damping_initial", 0.10)
            ),
            damping_minimum=float(
                _cfg_get(model_cfg, "damping_minimum", 0.001)
            ),
            lambda_initial=float(
                _cfg_get(model_cfg, "lambda_initial", 0.08)
            ),
            lambda_minimum=float(
                _cfg_get(model_cfg, "lambda_minimum", 0.001)
            ),
            edge_alpha=float(_cfg_get(model_cfg, "edge_alpha", 8.0)),
            minimum_data_anchor=float(
                _cfg_get(model_cfg, "minimum_data_anchor", 0.05)
            ),
            initial_validity=float(
                _cfg_get(model_cfg, "initial_validity", 0.90)
            ),
            match_precision_floor=float(
                _cfg_get(model_cfg, "match_precision_floor", 0.01)
            ),
            match_precision_ceiling=float(
                _cfg_get(model_cfg, "match_precision_ceiling", 1.00)
            ),
            feature_weight=float(
                _cfg_get(model_cfg, "feature_weight", 1.0)
            ),
            geometry_weight=float(
                _cfg_get(model_cfg, "geometry_weight", 1.0)
            ),
            charbonnier_epsilon=float(
                _cfg_get(model_cfg, "charbonnier_epsilon", 0.03)
            ),
            charbonnier_alpha=float(
                _cfg_get(model_cfg, "charbonnier_alpha", 0.45)
            ),
            depth_charbonnier_epsilon=float(
                _cfg_get(model_cfg, "depth_charbonnier_epsilon", 0.10)
            ),
        )
        self.cells = nn.ModuleDict(
            {
                str(scale): HQSSceneLMCell(
                    context_channels=context_by_scale[scale],
                    max_iterations=self.iterations_by_scale[scale],
                    radius=self.correlation_radii[scale],
                    match_temperature=self.match_temperatures[scale],
                    **common,
                )
                for scale in self.scale_order
            }
        )
        self.final_upsampler = SourceGuidedFieldUpsampler(
            field_channels=3,
            context_channels=context_by_scale[2],
            hidden_channels=int(
                _cfg_get(model_cfg, "upsample_hidden_dim", 64)
            ),
            groups=groups,
            rate=2,
            # Scene-flow components are metric, not pixels on the current grid.
            scale_vectors=False,
        )

        self.maximum_depth = float(
            _cfg_get(model_cfg, "maximum_depth", 100.0)
        )
        self.depth_guidance_weight = float(
            _cfg_get(model_cfg, "depth_guidance_weight", 0.5)
        )
        self.global_temperature = float(
            _cfg_get(model_cfg, "global_temperature", 0.07)
        )
        self.global_query_chunk_size = int(
            _cfg_get(model_cfg, "global_query_chunk_size", 512)
        )
        self.global_confidence_gated = bool(
            _cfg_get(model_cfg, "global_confidence_gated", True)
        )
        self.global_confidence_floor = float(
            _cfg_get(model_cfg, "global_confidence_floor", 0.05)
        )

        self.register_buffer(
            "image_mean",
            torch.tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "image_std",
            torch.tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1),
            persistent=False,
        )

    @staticmethod
    def _gray(image: torch.Tensor) -> torch.Tensor:
        weights = image.new_tensor((0.2989, 0.5870, 0.1140)).view(
            1, 3, 1, 1
        )
        return (image * weights).sum(dim=1, keepdim=True)

    def _normalise_image(self, image: torch.Tensor) -> torch.Tensor:
        return (image - self.image_mean.to(image)) / self.image_std.to(image)

    def _normalise_depth(self, depth: torch.Tensor) -> torch.Tensor:
        maximum = max(self.maximum_depth, 1e-3)
        return torch.log1p(depth.clamp(0.0, maximum)) / torch.log1p(
            depth.new_tensor(maximum)
        )

    @staticmethod
    def _resize_depth(
        depth: torch.Tensor,
        size: Tuple[int, int],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        valid = (depth > 0.0).to(dtype=depth.dtype)
        numerator = F.interpolate(
            depth * valid,
            size=size,
            mode="bilinear",
            align_corners=False,
        )
        denominator = F.interpolate(
            valid,
            size=size,
            mode="bilinear",
            align_corners=False,
        )
        resized = numerator / denominator.clamp_min(1e-6)
        resized_valid = (denominator > 0.999).to(dtype=depth.dtype)
        return resized * resized_valid, resized_valid

    def _raw_correlation(
        self,
        scale: int,
        induced_flow: torch.Tensor,
        source_features: torch.Tensor,
        target_features: torch.Tensor,
        all_pairs: Optional[AllPairsCorrelation],
    ) -> torch.Tensor:
        if scale in (16, 8):
            if all_pairs is None:
                raise RuntimeError(f"Missing all-pairs correlation at 1/{scale}")
            return all_pairs.lookup(induced_flow)
        return self.local_correlations[str(scale)](
            source_features, target_features, induced_flow
        )

    def forward(
        self,
        image1: torch.Tensor,
        image2: torch.Tensor,
        depth1: torch.Tensor,
        depth2: torch.Tensor,
        intrinsics: torch.Tensor,
        transform_21: Optional[torch.Tensor] = None,
        *,
        scene_flow_init: Optional[torch.Tensor] = None,
        iters: Optional[int] = None,
    ) -> Dict[str, object]:
        if image1.ndim != 4 or image1.shape != image2.shape:
            raise ValueError(
                "HQS-LM-SF expects equal [B,3,H,W] images"
            )
        if image1.shape[1] != 3:
            raise ValueError("HQS-LM-SF expects RGB inputs")
        batch, _, full_height, full_width = image1.shape
        expected_depth = (batch, 1, full_height, full_width)
        if depth1.shape != expected_depth or depth2.shape != expected_depth:
            raise ValueError(
                f"depth1/depth2 must be {expected_depth}, got "
                f"{tuple(depth1.shape)} and {tuple(depth2.shape)}"
            )
        validate_intrinsics(intrinsics, batch)
        if transform_21 is None:
            transform_21 = identity_transform(
                batch, device=image1.device, dtype=image1.dtype
            )
        validate_transform(transform_21, batch)
        if iters is not None and int(iters) != self.num_hqs_iterations:
            raise ValueError(
                "Set hqs_lm_sf.iterations rather than forward(iters=...); "
                f"configured total={self.num_hqs_iterations}"
            )

        image1 = image1.float()
        image2 = image2.float()
        depth1 = depth1.float()
        depth2 = depth2.float()
        intrinsics = intrinsics.to(device=image1.device, dtype=image1.dtype)
        transform_21 = transform_21.to(
            device=image1.device, dtype=image1.dtype
        )

        source_backbone = self.feature_encoder.backbone(
            self._normalise_image(image1)
        )
        target_backbone = self.feature_encoder.backbone(
            self._normalise_image(image2)
        )
        source_match_raw = self.feature_encoder.project_matching(
            source_backbone
        )
        target_match_raw = self.feature_encoder.project_matching(
            target_backbone
        )
        source_match = {
            scale: F.normalize(value, dim=1, eps=1e-6)
            for scale, value in source_match_raw.items()
        }
        target_match = {
            scale: F.normalize(value, dim=1, eps=1e-6)
            for scale, value in target_match_raw.items()
        }
        image_context = self.feature_encoder.project_context(source_backbone)

        source_gray_full = self._gray(image1)
        source_context: Dict[int, torch.Tensor] = {}
        source_guidance: Dict[int, torch.Tensor] = {}
        depth1_pyramid: Dict[int, torch.Tensor] = {}
        depth2_pyramid: Dict[int, torch.Tensor] = {}
        intrinsics_pyramid: Dict[int, torch.Tensor] = {}
        target_gradients: Dict[
            int, Tuple[torch.Tensor, torch.Tensor]
        ] = {}
        for scale in self.scale_order:
            size = source_match[scale].shape[-2:]
            depth1_scale, depth1_valid = self._resize_depth(depth1, size)
            depth2_scale, _ = self._resize_depth(depth2, size)
            depth1_pyramid[scale] = depth1_scale
            depth2_pyramid[scale] = depth2_scale
            intrinsics_pyramid[scale] = scale_intrinsics(
                intrinsics,
                source_size=(full_height, full_width),
                target_size=size,
            )
            depth_signal = self._normalise_depth(depth1_scale)
            depth_features = self.depth_context[str(scale)](
                torch.cat((depth_signal, depth1_valid), dim=1)
            )
            source_context[scale] = self.context_fusion[str(scale)](
                torch.cat((image_context[scale], depth_features), dim=1)
            )
            source_gray = F.interpolate(
                source_gray_full,
                size=size,
                mode="bilinear",
                align_corners=False,
            )
            source_guidance[scale] = (
                source_gray
                + self.depth_guidance_weight * depth_signal
            )
            target_gradients[scale] = spatial_gradients(
                target_match[scale]
            )

        correlation16 = AllPairsCorrelation(
            source_match[16],
            target_match[16],
            num_levels=self.all_pairs_levels[16],
            radius=self.correlation_radii[16],
        )
        global_match = correlation16.global_soft_match(
            temperature=self.global_temperature,
            query_chunk_size=self.global_query_chunk_size,
        )
        if scene_flow_init is not None:
            if scene_flow_init.ndim != 4 or scene_flow_init.shape[1] != 3:
                raise ValueError(
                    "scene_flow_init must be [B,3,H,W] in metric units"
                )
            initial = resize_metric_field(
                scene_flow_init.to(
                    device=image1.device, dtype=image1.dtype
                ),
                source_match[16].shape[-2:],
            )
            lifted_valid = (depth1_pyramid[16] > 0.0).to(image1.dtype)
        else:
            lifted = lift_flow_to_scene_flow(
                global_match["flow_xy"],
                depth1_pyramid[16],
                depth2_pyramid[16],
                intrinsics_pyramid[16],
                transform_21,
            )
            initial = lifted.scene_flow
            lifted_valid = lifted.valid
            if self.global_confidence_gated:
                floor = min(max(self.global_confidence_floor, 0.0), 1.0)
                gate = floor + (
                    1.0 - floor
                ) * global_match["confidence"]
                initial = gate * initial

        state = LMState(w=initial, z=initial)
        scene_flow_predictions: List[torch.Tensor] = []
        induced_flow_predictions: List[torch.Tensor] = []
        scene_flow_lows: List[torch.Tensor] = []
        data_scene_flow_lows: List[torch.Tensor] = []
        induced_flow_lows: List[torch.Tensor] = []
        coupling_lows: List[torch.Tensor] = []
        delta_lows: List[torch.Tensor] = []
        data_delta_lows: List[torch.Tensor] = []
        proximal_anchor_lows: List[torch.Tensor] = []
        appearance_validity_lows: List[torch.Tensor] = []
        geometry_validity_lows: List[torch.Tensor] = []
        data_confidence_lows: List[torch.Tensor] = []
        depth_residual_lows: List[torch.Tensor] = []
        feature_residual_lows: List[torch.Tensor] = []
        match_proposal_lows: List[torch.Tensor] = []
        match_precision_lows: List[torch.Tensor] = []
        inverse_diagonal_lows: List[torch.Tensor] = []
        inverse_trace_lows: List[torch.Tensor] = []
        condition_lows: List[torch.Tensor] = []
        beta_values: List[torch.Tensor] = []
        damping_values: List[torch.Tensor] = []
        lambda_values: List[torch.Tensor] = []
        prediction_scales: List[int] = []

        for scale_index, scale in enumerate(self.scale_order):
            if scale_index == 0:
                all_pairs: Optional[AllPairsCorrelation] = correlation16
            else:
                target_size = source_match[scale].shape[-2:]
                state = LMState(
                    w=resize_metric_field(state.w, target_size),
                    z=resize_metric_field(state.z, target_size),
                )
                if scale == 8:
                    all_pairs = AllPairsCorrelation(
                        source_match[8],
                        target_match[8],
                        num_levels=self.all_pairs_levels[8],
                        radius=self.correlation_radii[8],
                    )
                else:
                    all_pairs = None

            cell = self.cells[str(scale)]
            for iteration in range(self.iterations_by_scale[scale]):
                previous_z = state.z
                current_projection = project_scene_flow(
                    state.w,
                    depth1_pyramid[scale],
                    intrinsics_pyramid[scale],
                    transform_21,
                )
                correlation = self._raw_correlation(
                    scale,
                    current_projection.induced_flow,
                    source_match[scale],
                    target_match[scale],
                    all_pairs,
                )
                grad_x, grad_y = target_gradients[scale]
                update = cell(
                    source_features=source_match[scale],
                    target_features=target_match[scale],
                    target_grad_x=grad_x,
                    target_grad_y=grad_y,
                    correlation=correlation,
                    source_context=source_context[scale],
                    source_guidance=source_guidance[scale],
                    depth1=depth1_pyramid[scale],
                    depth2=depth2_pyramid[scale],
                    intrinsics=intrinsics_pyramid[scale],
                    transform_21=transform_21,
                    state=state,
                    iteration=iteration,
                    jacobi_sweeps=self.jacobi_sweeps[scale],
                    max_data_delta=self.max_data_delta[scale],
                    max_prox_delta=self.max_prox_delta[scale],
                )
                state = update.state

                scene_flow_lows.append(state.z)
                data_scene_flow_lows.append(state.w)
                induced_flow_lows.append(
                    project_scene_flow(
                        state.z,
                        depth1_pyramid[scale],
                        intrinsics_pyramid[scale],
                        transform_21,
                    ).induced_flow
                )
                coupling_lows.append(state.w - state.z)
                delta_lows.append(state.z - previous_z)
                data_delta_lows.append(update.data_delta)
                proximal_anchor_lows.append(update.proximal_anchor)
                appearance_validity_lows.append(
                    update.appearance_validity
                )
                geometry_validity_lows.append(update.geometry_validity)
                data_confidence_lows.append(update.data_confidence)
                depth_residual_lows.append(
                    update.depth_linearisation.residual
                )
                feature_residual_lows.append(update.feature_residual)
                match_proposal_lows.append(update.match_proposal)
                match_precision_lows.append(update.match_precision)
                inverse_diagonal_lows.append(update.inverse_diagonal)
                inverse_trace_lows.append(update.inverse_trace)
                condition_lows.append(update.condition)
                beta_values.append(update.beta)
                damping_values.append(update.damping)
                lambda_values.append(update.regularisation)
                prediction_scales.append(scale)

                full_scene = resize_metric_field(
                    state.z, (full_height, full_width)
                )
                full_projection = project_scene_flow(
                    full_scene, depth1, intrinsics, transform_21
                )
                scene_flow_predictions.append(full_scene)
                induced_flow_predictions.append(
                    full_projection.induced_flow
                )

        raw_scene_flow_predictions = list(scene_flow_predictions)
        final_scene_flow, final_mask_logits = self.final_upsampler(
            state.z,
            source_context[2],
            source_guidance[2],
        )
        if (
            final_scene_flow.shape[-2] >= full_height
            and final_scene_flow.shape[-1] >= full_width
        ):
            final_scene_flow = final_scene_flow[
                ..., :full_height, :full_width
            ]
        if final_scene_flow.shape[-2:] != (full_height, full_width):
            final_scene_flow = resize_metric_field(
                final_scene_flow, (full_height, full_width)
            )
        final_projection = project_scene_flow(
            final_scene_flow, depth1, intrinsics, transform_21
        )
        final_warped_depth = backward_warp(
            depth2,
            final_projection.induced_flow,
            padding_mode="border",
        )
        final_target_depth_valid = backward_warp(
            (depth2 > 0.0).to(dtype=depth2.dtype),
            final_projection.induced_flow,
            padding_mode="zeros",
        )
        final_geometry_valid = final_projection.valid * (
            final_target_depth_valid > 0.999
        ).to(dtype=depth2.dtype)
        final_depth_residual = (
            final_warped_depth - final_projection.target_points[:, 2:3]
        )
        scene_flow_predictions[-1] = final_scene_flow
        induced_flow_predictions[-1] = final_projection.induced_flow

        return {
            "scene_flow_preds": scene_flow_predictions,
            "scene_flow_preds_raw": raw_scene_flow_predictions,
            # Common name for visualisation/evaluation of induced 2D motion.
            "flow_preds": induced_flow_predictions,
            "scene_flow_low": scene_flow_lows,
            "data_scene_flow_low": data_scene_flow_lows,
            "induced_flow_low": induced_flow_lows,
            "coupling_residual_low": coupling_lows,
            "delta_low": delta_lows,
            "data_delta_lows": data_delta_lows,
            "analytic_delta_lows": data_delta_lows,
            "learned_data_delta_lows": [
                torch.zeros_like(value) for value in data_delta_lows
            ],
            "proximal_anchor_lows": proximal_anchor_lows,
            "appearance_validity_lows": appearance_validity_lows,
            "geometry_validity_lows": geometry_validity_lows,
            "data_confidence_lows": data_confidence_lows,
            "depth_residual_lows": depth_residual_lows,
            "feature_residual_lows": feature_residual_lows,
            "match_proposal_lows": match_proposal_lows,
            "match_precision_lows": match_precision_lows,
            "lm_inverse_diagonal_lows": inverse_diagonal_lows,
            "lm_inverse_trace_lows": inverse_trace_lows,
            "lm_condition_lows": condition_lows,
            "beta_values": beta_values,
            "lm_damping_values": damping_values,
            "lambda_values": lambda_values,
            "prediction_scales": prediction_scales,
            "final_upsample_mask_logits": final_mask_logits,
            "scene_flow_final": final_scene_flow,
            "induced_flow_final": final_projection.induced_flow,
            "target_points_final": final_projection.target_points,
            "target_depth_predicted_final": final_projection.target_points[
                :, 2:3
            ],
            "target_depth_warped_final": final_warped_depth,
            "depth_residual_final": final_depth_residual,
            "projection_valid_final": final_geometry_valid,
            "global_init_scene_flow": initial,
            "global_init_valid": lifted_valid,
            "global_init_flow_xy": global_match["flow_xy"],
            "global_init_confidence": global_match["confidence"],
            "global_init_entropy": global_match["entropy"],
            "global_init_margin": global_match["margin"],
            "solver": "hqs_lm_sf",
            "scene_flow_convention": "source_camera_metric_before_T21",
        }

    def param_count(self) -> Dict[str, int]:
        def count(module: nn.Module) -> int:
            return sum(parameter.numel() for parameter in module.parameters())

        feature_count = (
            count(self.feature_encoder.stages)
            + count(self.feature_encoder.match_projections)
        )
        context_count = (
            count(self.feature_encoder.context_projections)
            + count(self.depth_context)
            + count(self.context_fusion)
        )
        stage_count = count(self.cells)
        return {
            "feature_encoder": feature_count,
            "context_encoder": context_count,
            "cells": stage_count,
            "final_upsampler": count(self.final_upsampler),
            "stages": stage_count,
            "total": count(self),
            "total_trainable": sum(
                parameter.numel()
                for parameter in self.parameters()
                if parameter.requires_grad
            ),
        }


__all__ = ["HQSLMSceneFlow"]
