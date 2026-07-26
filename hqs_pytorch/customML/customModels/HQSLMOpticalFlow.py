"""HQS-LM-OF: transformer-enhanced proximal LM optical flow.

This model is an isolated successor to ``HQSCore``.  It retains the
coarse-to-fine HQS state and source-conditioned proximal branch, but replaces
the scalar OFCE update and learned data residual with a robust multi-channel
Levenberg-Marquardt solve.  Transformer-enhanced matching features improve
long-range correspondence discrimination.  At each HQS iteration,
flow-conditioned correlation attention decodes a bounded vector observation,
matchability and full ``2 x 2`` precision inside the analytic solve.

At every iteration:

1. the current flow attends over the complete indexed correlation tensor;
2. attention is decoded into a learned match observation and full precision;
3. target features are warped and linearised at the data state ``w``;
4. a damped ``2 x 2`` normal equation fuses feature, match and HQS evidence;
5. a source-only proximal updates ``z``.

The learned vector is a probabilistic observation inside the normal equation.
There is no learned vector addition after the data solve.
"""
from __future__ import annotations

import math
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.correlation import LocalCorrBlock
from models.hqs_core_components import (
    AllPairsCorrelation,
    HQSCorePyramidEncoder,
    spatial_gradients,
)
from models.hqs_lm_components import (
    LMState,
    SourceGuidedFieldUpsampler,
)
from models.hqs_lm_learned_measurement import (
    HQSOpticalLearnedMeasurementCell,
)
from models.warp import resize_flow

from .sota_addons import TransformerFeatureEnhancer


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
            f"hqs_lm_of.{key} must contain {len(default)} entries, got {value}"
        )
    return result


class HQSLMOpticalFlow(nn.Module):
    """Four-scale HQS-LM optical-flow estimator."""

    scale_order: Tuple[int, ...] = (16, 8, 4, 2)

    def __init__(self, cfg) -> None:
        super().__init__()
        model_cfg = _cfg_get(cfg, "hqs_lm_of", cfg)
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

        iterations = _list(
            model_cfg, "iterations", (2, 4, 2, 2), int
        )
        if any(value < 1 for value in iterations):
            raise ValueError(
                f"All HQS-LM-OF iteration counts must be positive: {iterations}"
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
                    "max_data_delta",
                    (2.0, 2.0, 1.5, 1.0),
                    float,
                ),
            )
        )
        self.max_prox_delta = dict(
            zip(
                self.scale_order,
                _list(
                    model_cfg,
                    "max_prox_delta",
                    (0.5, 0.5, 0.5, 0.25),
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
        all_pairs_levels = _list(
            model_cfg, "all_pairs_levels", (1, 4), int
        )
        self.all_pairs_levels = {
            16: all_pairs_levels[0],
            8: all_pairs_levels[1],
        }

        self.feature_encoder = HQSCorePyramidEncoder(
            feature_channels=feature_channels,
            match_channels=match_channels,
            context_channels=context_channels,
            blocks_per_scale=blocks_per_scale,
            groups=groups,
        )
        match_by_scale = dict(zip((2, 4, 8, 16), match_channels))
        transformer_depth = dict(
            zip(
                self.scale_order,
                _list(
                    model_cfg,
                    "feature_transformer_depth",
                    (1, 1, 1, 0),
                    int,
                ),
            )
        )
        transformer_heads = dict(
            zip(
                self.scale_order,
                _list(
                    model_cfg,
                    "feature_transformer_heads",
                    (4, 4, 4, 4),
                    int,
                ),
            )
        )
        transformer_max_tokens = dict(
            zip(
                self.scale_order,
                _list(
                    model_cfg,
                    "feature_transformer_max_tokens",
                    (4096, 2048, 1024, 1024),
                    int,
                ),
            )
        )
        transformer_mlp_ratio = float(
            _cfg_get(model_cfg, "feature_transformer_mlp_ratio", 2.0)
        )
        transformer_dropout = float(
            _cfg_get(model_cfg, "feature_transformer_dropout", 0.0)
        )
        initial_blend = float(
            _cfg_get(
                model_cfg,
                "feature_transformer_initial_blend",
                0.10,
            )
        )
        if not 0.0 < initial_blend < 1.0:
            raise ValueError(
                "feature_transformer_initial_blend must be in (0, 1)"
            )
        self.match_feature_transformers = nn.ModuleDict()
        self.match_feature_transformer_logits = nn.ParameterDict()
        blend_logit = math.log(initial_blend / (1.0 - initial_blend))
        for scale in self.scale_order:
            depth = transformer_depth[scale]
            if depth < 0:
                raise ValueError(
                    "feature_transformer_depth entries must be nonnegative"
                )
            if depth == 0:
                continue
            channels = match_by_scale[scale]
            heads = transformer_heads[scale]
            if channels % heads != 0:
                raise ValueError(
                    f"1/{scale} matching channels ({channels}) must be "
                    f"divisible by transformer heads ({heads})"
                )
            self.match_feature_transformers[str(scale)] = (
                TransformerFeatureEnhancer(
                    feature_dim=channels,
                    num_heads=heads,
                    depth=depth,
                    mlp_ratio=transformer_mlp_ratio,
                    dropout=transformer_dropout,
                    max_tokens=transformer_max_tokens[scale],
                )
            )
            self.match_feature_transformer_logits[str(scale)] = nn.Parameter(
                torch.tensor(blend_logit, dtype=torch.float32)
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

        context_by_scale = dict(zip((2, 4, 8, 16), context_channels))
        raw_correlation_channels = {
            16: self.all_pairs_levels[16]
            * (2 * self.correlation_radii[16] + 1) ** 2,
            8: self.all_pairs_levels[8]
            * (2 * self.correlation_radii[8] + 1) ** 2,
            4: (2 * self.correlation_radii[4] + 1) ** 2,
            2: (2 * self.correlation_radii[2] + 1) ** 2,
        }
        proposal_embedding_channels = dict(
            zip(
                self.scale_order,
                _list(
                    model_cfg,
                    "proposal_embedding_channels",
                    (64, 64, 64, 64),
                    int,
                ),
            )
        )
        proposal_hidden_channels = dict(
            zip(
                self.scale_order,
                _list(
                    model_cfg,
                    "proposal_hidden_channels",
                    (96, 96, 64, 64),
                    int,
                ),
            )
        )
        maximum_proposal_delta = dict(
            zip(
                self.scale_order,
                _list(
                    model_cfg,
                    "max_learned_proposal_delta",
                    (2.0, 2.0, 1.5, 1.0),
                    float,
                ),
            )
        )
        correlation_attention_channels = dict(
            zip(
                self.scale_order,
                _list(
                    model_cfg,
                    "correlation_attention_channels",
                    (32, 32, 32, 32),
                    int,
                ),
            )
        )
        correlation_attention_heads = dict(
            zip(
                self.scale_order,
                _list(
                    model_cfg,
                    "correlation_attention_heads",
                    (4, 4, 4, 4),
                    int,
                ),
            )
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
            edge_alpha=float(_cfg_get(model_cfg, "edge_alpha", 10.0)),
            minimum_data_anchor=float(
                _cfg_get(model_cfg, "minimum_data_anchor", 0.05)
            ),
            initial_validity=float(
                _cfg_get(model_cfg, "initial_validity", 0.90)
            ),
            precision_minimum=float(
                _cfg_get(model_cfg, "learned_precision_minimum", 0.0)
            ),
            precision_maximum=float(
                _cfg_get(model_cfg, "learned_precision_maximum", 1.00)
            ),
            precision_correlation_limit=float(
                _cfg_get(
                    model_cfg,
                    "learned_precision_correlation_limit",
                    0.95,
                )
            ),
            initial_precision=float(
                _cfg_get(model_cfg, "learned_precision_initial", 0.20)
            ),
            initial_matchability=float(
                _cfg_get(model_cfg, "initial_matchability", 0.90)
            ),
            analytic_confidence_floor=float(
                _cfg_get(model_cfg, "analytic_confidence_floor", 0.02)
            ),
            correlation_attention_temperature=float(
                _cfg_get(
                    model_cfg,
                    "correlation_attention_temperature",
                    1.0,
                )
            ),
            charbonnier_epsilon=float(
                _cfg_get(model_cfg, "charbonnier_epsilon", 0.03)
            ),
            charbonnier_alpha=float(
                _cfg_get(model_cfg, "charbonnier_alpha", 0.45)
            ),
        )
        self.cells = nn.ModuleDict(
            {
                str(scale): HQSOpticalLearnedMeasurementCell(
                    correlation_channels=raw_correlation_channels[scale],
                    context_channels=context_by_scale[scale],
                    max_iterations=self.iterations_by_scale[scale],
                    radius=self.correlation_radii[scale],
                    match_temperature=self.match_temperatures[scale],
                    maximum_proposal_delta=maximum_proposal_delta[scale],
                    proposal_embedding_channels=(
                        proposal_embedding_channels[scale]
                    ),
                    proposal_hidden_channels=(
                        proposal_hidden_channels[scale]
                    ),
                    correlation_attention_channels=(
                        correlation_attention_channels[scale]
                    ),
                    correlation_attention_heads=(
                        correlation_attention_heads[scale]
                    ),
                    **common,
                )
                for scale in self.scale_order
            }
        )
        self.final_upsampler = SourceGuidedFieldUpsampler(
            field_channels=2,
            context_channels=context_by_scale[2],
            hidden_channels=int(
                _cfg_get(model_cfg, "upsample_hidden_dim", 64)
            ),
            groups=groups,
            rate=2,
            scale_vectors=True,
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
        self.allow_external_validity_inputs = bool(
            _cfg_get(model_cfg, "allow_external_validity_inputs", False)
        )
        self.solver_name = "hqs_lm_of"

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

    def _normalise(self, image: torch.Tensor) -> torch.Tensor:
        return (image - self.image_mean.to(image)) / self.image_std.to(image)

    @staticmethod
    def _to_yx(flow_xy: torch.Tensor) -> torch.Tensor:
        return torch.stack((flow_xy[:, 1], flow_xy[:, 0]), dim=1)

    def _raw_correlation(
        self,
        scale: int,
        flow_w: torch.Tensor,
        source_features: torch.Tensor,
        target_features: torch.Tensor,
        all_pairs: Optional[AllPairsCorrelation],
    ) -> torch.Tensor:
        if scale in (16, 8):
            if all_pairs is None:
                raise RuntimeError(f"Missing all-pairs correlation at 1/{scale}")
            return all_pairs.lookup(flow_w)
        return self.local_correlations[str(scale)](
            source_features, target_features, flow_w
        )

    def _enhance_matching_features(
        self,
        source_features: Dict[int, torch.Tensor],
        target_features: Dict[int, torch.Tensor],
    ) -> Tuple[Dict[int, torch.Tensor], Dict[int, torch.Tensor]]:
        """Apply gated self/cross-attention on the matching branch only.

        The proximal context pyramid is deliberately not passed through this
        cross-image transformer.  Target-conditioned feature enhancement
        therefore terminates at correspondence and feature data consistency.
        """
        enhanced_source = dict(source_features)
        enhanced_target = dict(target_features)
        for scale in self.scale_order:
            key = str(scale)
            if key not in self.match_feature_transformers:
                continue
            source_value, target_value = self.match_feature_transformers[key](
                source_features[scale],
                target_features[scale],
            )
            blend = torch.sigmoid(
                self.match_feature_transformer_logits[key]
            ).to(
                device=source_value.device,
                dtype=source_value.dtype,
            )
            enhanced_source[scale] = source_features[scale] + blend * (
                source_value - source_features[scale]
            )
            enhanced_target[scale] = target_features[scale] + blend * (
                target_value - target_features[scale]
            )
        return enhanced_source, enhanced_target

    def forward(
        self,
        image1: torch.Tensor,
        image2: torch.Tensor,
        iters: Optional[int] = None,
        flow_init: Optional[torch.Tensor] = None,
        source_valid: Optional[torch.Tensor] = None,
        target_valid: Optional[torch.Tensor] = None,
    ) -> Dict[str, object]:
        if image1.ndim != 4 or image1.shape != image2.shape:
            raise ValueError(
                "HQS-LM-OF expects equal [B,3,H,W] images, got "
                f"{tuple(image1.shape)} and {tuple(image2.shape)}"
            )
        if image1.shape[1] != 3:
            raise ValueError("HQS-LM-OF expects RGB inputs")
        if iters is not None and int(iters) != self.num_hqs_iterations:
            raise ValueError(
                "Set hqs_lm_of.iterations rather than forward(iters=...); "
                f"configured total={self.num_hqs_iterations}"
            )
        if (source_valid is not None or target_valid is not None) and not (
            self.allow_external_validity_inputs
        ):
            raise ValueError(
                "Ground-truth validity is loss-only for HQS-LM-OF"
            )

        image1 = image1.float()
        image2 = image2.float()
        _, _, full_height, full_width = image1.shape

        source_backbone = self.feature_encoder.backbone(
            self._normalise(image1)
        )
        target_backbone = self.feature_encoder.backbone(
            self._normalise(image2)
        )
        source_match_raw = self.feature_encoder.project_matching(
            source_backbone
        )
        target_match_raw = self.feature_encoder.project_matching(
            target_backbone
        )
        source_match_raw, target_match_raw = (
            self._enhance_matching_features(
                source_match_raw,
                target_match_raw,
            )
        )
        source_match = {
            scale: F.normalize(value, dim=1, eps=1e-6)
            for scale, value in source_match_raw.items()
        }
        target_match = {
            scale: F.normalize(value, dim=1, eps=1e-6)
            for scale, value in target_match_raw.items()
        }
        # This is the only learned prior/context pyramid.
        source_context = self.feature_encoder.project_context(source_backbone)

        source_gray_full = self._gray(image1)
        source_guidance: Dict[int, torch.Tensor] = {}
        target_gradients: Dict[
            int, Tuple[torch.Tensor, torch.Tensor]
        ] = {}
        for scale in self.scale_order:
            size = source_match[scale].shape[-2:]
            source_guidance[scale] = F.interpolate(
                source_gray_full,
                size=size,
                mode="bilinear",
                align_corners=False,
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
        if flow_init is not None:
            if flow_init.ndim != 4 or flow_init.shape[1] != 2:
                raise ValueError(
                    f"flow_init must be [B,2,H,W], got {tuple(flow_init.shape)}"
                )
            initial = resize_flow(
                flow_init.to(device=image1.device, dtype=image1.dtype),
                source_match[16].shape[-2:],
            )
        else:
            initial = global_match["flow_xy"]
            if self.global_confidence_gated:
                floor = min(max(self.global_confidence_floor, 0.0), 1.0)
                gate = floor + (1.0 - floor) * global_match["confidence"]
                initial = gate * initial

        state = LMState(w=initial, z=initial)
        flow_predictions: List[torch.Tensor] = []
        flow_lows: List[torch.Tensor] = []
        data_flow_lows: List[torch.Tensor] = []
        coupling_lows: List[torch.Tensor] = []
        delta_lows: List[torch.Tensor] = []
        data_delta_lows: List[torch.Tensor] = []
        learned_proposal_delta_lows: List[torch.Tensor] = []
        proposal_offset_lows: List[torch.Tensor] = []
        analytic_match_proposal_lows: List[torch.Tensor] = []
        matchability_lows: List[torch.Tensor] = []
        proximal_anchor_lows: List[torch.Tensor] = []
        reliability_lows: List[torch.Tensor] = []
        confidence_lows: List[torch.Tensor] = []
        match_proposal_lows: List[torch.Tensor] = []
        match_precision_lows: List[torch.Tensor] = []
        match_confidence_lows: List[torch.Tensor] = []
        match_entropy_lows: List[torch.Tensor] = []
        correlation_attention_entropy_lows: List[torch.Tensor] = []
        correlation_attention_peak_lows: List[torch.Tensor] = []
        feature_residual_lows: List[torch.Tensor] = []
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
                    w=resize_flow(state.w, target_size),
                    z=resize_flow(state.z, target_size),
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
            proposal_hidden = None
            if hasattr(cell, "initialise_proposal_hidden"):
                proposal_hidden = cell.initialise_proposal_hidden(
                    source_context[scale]
                )
            for iteration in range(self.iterations_by_scale[scale]):
                previous_w = state.w
                previous_z = state.z
                correlation = self._raw_correlation(
                    scale,
                    state.w,
                    source_match[scale],
                    target_match[scale],
                    all_pairs,
                )
                grad_x, grad_y = target_gradients[scale]
                cell_inputs = dict(
                    source_features=source_match[scale],
                    target_features=target_match[scale],
                    target_grad_x=grad_x,
                    target_grad_y=grad_y,
                    correlation=correlation,
                    source_context=source_context[scale],
                    source_guidance=source_guidance[scale],
                    state=state,
                    iteration=iteration,
                    jacobi_sweeps=self.jacobi_sweeps[scale],
                    max_data_delta=self.max_data_delta[scale],
                    max_prox_delta=self.max_prox_delta[scale],
                )
                if hasattr(cell, "initialise_proposal_hidden"):
                    cell_inputs["proposal_hidden"] = proposal_hidden
                update = cell(**cell_inputs)
                if update.proposal_hidden is not None:
                    proposal_hidden = update.proposal_hidden
                state = update.state

                flow_lows.append(state.z)
                data_flow_lows.append(state.w)
                coupling_lows.append(state.w - state.z)
                delta_lows.append(state.z - previous_z)
                data_delta_lows.append(update.data_delta)
                learned_proposal_delta_lows.append(
                    update.learned_proposal_delta
                    if update.learned_proposal_delta is not None
                    else torch.zeros_like(update.data_delta)
                )
                proposal_offset_lows.append(
                    update.proposal_offset
                    if update.proposal_offset is not None
                    else update.match_proposal - previous_w
                )
                analytic_match_proposal_lows.append(
                    update.analytic_match_proposal
                    if update.analytic_match_proposal is not None
                    else update.match_proposal
                )
                matchability_lows.append(
                    update.matchability
                    if update.matchability is not None
                    else update.match_confidence
                )
                proximal_anchor_lows.append(update.proximal_anchor)
                reliability_lows.append(update.appearance_validity)
                confidence_lows.append(update.data_confidence)
                match_proposal_lows.append(update.match_proposal)
                match_precision_lows.append(update.match_precision)
                match_confidence_lows.append(update.match_confidence)
                match_entropy_lows.append(update.match_entropy)
                correlation_attention_entropy_lows.append(
                    update.correlation_attention_entropy
                    if update.correlation_attention_entropy is not None
                    else torch.zeros_like(update.match_entropy)
                )
                correlation_attention_peak_lows.append(
                    update.correlation_attention_peak
                    if update.correlation_attention_peak is not None
                    else torch.zeros_like(update.match_confidence)
                )
                feature_residual_lows.append(update.feature_residual)
                inverse_trace_lows.append(update.inverse_trace)
                condition_lows.append(update.condition)
                beta_values.append(update.beta)
                damping_values.append(update.damping)
                lambda_values.append(update.regularisation)
                prediction_scales.append(scale)
                flow_predictions.append(
                    resize_flow(
                        state.z, (full_height, full_width)
                    )
                )

        raw_flow_predictions = list(flow_predictions)
        final_flow, final_mask_logits = self.final_upsampler(
            state.z,
            source_context[2],
            source_guidance[2],
        )
        if (
            final_flow.shape[-2] >= full_height
            and final_flow.shape[-1] >= full_width
        ):
            final_flow = final_flow[..., :full_height, :full_width]
        if final_flow.shape[-2:] != (full_height, full_width):
            final_flow = resize_flow(
                final_flow, (full_height, full_width)
            )
        flow_predictions[-1] = final_flow

        zero_data_residuals = [
            torch.zeros_like(delta) for delta in data_delta_lows
        ]
        return {
            "flow_preds": flow_predictions,
            "flow_preds_raw": raw_flow_predictions,
            "flow_low": flow_lows,
            "flow_low_raw": list(flow_lows),
            "data_flow_low": data_flow_lows,
            "aux_low": data_flow_lows,
            "delta_low": delta_lows,
            "coupling_residual_low": coupling_lows,
            "delta_match_low": data_delta_lows,
            "delta_prior_low": [-value for value in coupling_lows],
            "data_delta_lows": data_delta_lows,
            "analytic_delta_lows": data_delta_lows,
            # Explicitly zero: HQS-LM-OF has no learned vector data bypass.
            "learned_data_delta_lows": zero_data_residuals,
            # Learned target-conditioned vectors are correspondence
            # observations inside the normal equation, never post-solve
            # additions to the data state.
            "learned_proposal_delta_lows": learned_proposal_delta_lows,
            "proposal_offset_lows": proposal_offset_lows,
            "analytic_match_proposal_lows": (
                analytic_match_proposal_lows
            ),
            "matchability_lows": matchability_lows,
            "proximal_anchor_lows": proximal_anchor_lows,
            # Visibility supervision must act on the learned appearance gate,
            # not on max(appearance, fixed correlation confidence), otherwise
            # a confident false match can block the BCE gradient.
            "data_reliability_low": reliability_lows[-1],
            "data_reliability_lows": reliability_lows,
            "core_reliability_lows": reliability_lows,
            "appearance_validity_lows": reliability_lows,
            "data_valid_lows": confidence_lows,
            "core_validity_lows": reliability_lows,
            "data_weight_lows": confidence_lows,
            "match_proposal_lows": match_proposal_lows,
            "match_precision_lows": match_precision_lows,
            "match_confidence_lows": match_confidence_lows,
            "match_entropy_lows": match_entropy_lows,
            "correlation_attention_entropy_lows": (
                correlation_attention_entropy_lows
            ),
            "correlation_attention_peak_lows": (
                correlation_attention_peak_lows
            ),
            "feature_residual_lows": feature_residual_lows,
            "lm_inverse_trace_lows": inverse_trace_lows,
            "lm_condition_lows": condition_lows,
            "beta_values": beta_values,
            "lm_damping_values": damping_values,
            "lambda_values": lambda_values,
            "prediction_scales": prediction_scales,
            "final_upsample_mask_logits": final_mask_logits,
            "flow_final_raw": raw_flow_predictions[-1],
            "flow_final_refined": final_flow,
            "global_init_flow_xy": initial,
            "global_init_confidence": global_match["confidence"],
            "global_init_entropy": global_match["entropy"],
            "global_init_margin": global_match["margin"],
            "gmflow_init_flow_yx": self._to_yx(initial),
            "gmflow_init_conf": global_match["confidence"],
            "gmflow_init_entropy": global_match["entropy"],
            "gmflow_init_margin": global_match["margin"],
            "occupancy_masks": [],
            "reliability_states": [],
            "reliability_geometries": [],
            "reliability_residuals": [],
            "factorised_data_gates": [],
            "band_split": None,
            "feature_transformer_blends": {
                key: torch.sigmoid(value)
                for key, value in self.match_feature_transformer_logits.items()
            },
            "solver": self.solver_name,
        }

    def param_count(self) -> Dict[str, int]:
        def count(module: nn.Module) -> int:
            return sum(parameter.numel() for parameter in module.parameters())

        context_count = count(self.feature_encoder.context_projections)
        feature_count = (
            count(self.feature_encoder.stages)
            + count(self.feature_encoder.match_projections)
            + count(self.match_feature_transformers)
            + count(self.match_feature_transformer_logits)
        )
        stage_count = count(self.cells)
        return {
            "feature_encoder": feature_count,
            "feature_transformer": count(
                self.match_feature_transformers
            )
            + count(self.match_feature_transformer_logits),
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


__all__ = ["HQSLMOpticalFlow"]
