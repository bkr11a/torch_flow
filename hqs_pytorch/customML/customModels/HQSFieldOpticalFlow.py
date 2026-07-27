"""HQS-Field-OF: probabilistic correspondence and field inference.

The model preserves the thesis-level half-quadratic-splitting hypothesis while
changing the observation model to match the actual information available in
modern optical-flow systems:

* global/local correlation supplies a multi-modal likelihood over motion;
* learned modules calibrate hypotheses, matchability and full precision;
* a closed-form quadratic-majorisation step updates the HQS data variable;
* a source-conditioned graph quadratic completes the motion field.

There is no feature-constancy Gauss-Newton term in the canonical path and no
target-conditioned vector is added after the analytic data solve.  The
regularisation affinities are functions of source-frame features only.
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
)
from models.hqs_field_components import (
    CorrelationMixture,
    HQSCorrelationFieldCell,
    correlation_mixture_from_global,
)
from models.hqs_lm_components import LMState, SourceGuidedFieldUpsampler
from models.warp import backward_warp, flow_in_bounds_mask, resize_flow

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
            f"hqs_field_of.{key} must contain {len(default)} entries, "
            f"got {value}"
        )
    return result


class HQSFieldOpticalFlow(nn.Module):
    """Four-scale probabilistic field HQS optical-flow estimator."""

    scale_order: Tuple[int, ...] = (16, 8, 4, 2)

    def __init__(self, cfg) -> None:
        super().__init__()
        model_cfg = _cfg_get(cfg, "hqs_field_of", cfg)
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
                f"All HQS-Field iteration counts must be positive: "
                f"{iterations}"
            )
        self.iterations_by_scale = dict(zip(self.scale_order, iterations))
        self.num_hqs_iterations = sum(iterations)
        self.graph_sweeps = dict(
            zip(
                self.scale_order,
                _list(model_cfg, "graph_sweeps", (3, 3, 2, 2), int),
            )
        )
        self.max_data_delta = dict(
            zip(
                self.scale_order,
                _list(
                    model_cfg,
                    "max_data_delta",
                    (4.0, 3.0, 2.0, 1.0),
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
        self.num_hypotheses = dict(
            zip(
                self.scale_order,
                _list(
                    model_cfg, "num_hypotheses", (4, 4, 4, 4), int
                ),
            )
        )
        self.mixture_temperatures = dict(
            zip(
                self.scale_order,
                _list(
                    model_cfg,
                    "mixture_temperatures",
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
        context_by_scale = dict(zip((2, 4, 8, 16), context_channels))
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
        initial_blend = float(
            _cfg_get(
                model_cfg, "feature_transformer_initial_blend", 0.10
            )
        )
        if not 0.0 < initial_blend < 1.0:
            raise ValueError(
                "feature_transformer_initial_blend must be in (0,1)"
            )
        blend_logit = math.log(initial_blend / (1.0 - initial_blend))
        self.match_feature_transformers = nn.ModuleDict()
        self.match_feature_transformer_logits = nn.ParameterDict()
        for scale in self.scale_order:
            depth = transformer_depth[scale]
            if depth == 0:
                continue
            channels = match_by_scale[scale]
            heads = transformer_heads[scale]
            if channels % heads != 0:
                raise ValueError(
                    f"1/{scale} match channels must be divisible by heads"
                )
            self.match_feature_transformers[str(scale)] = (
                TransformerFeatureEnhancer(
                    feature_dim=channels,
                    num_heads=heads,
                    depth=depth,
                    mlp_ratio=float(
                        _cfg_get(
                            model_cfg,
                            "feature_transformer_mlp_ratio",
                            2.0,
                        )
                    ),
                    dropout=float(
                        _cfg_get(
                            model_cfg,
                            "feature_transformer_dropout",
                            0.0,
                        )
                    ),
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
        attention_channels = dict(
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
        attention_heads = dict(
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
        graph_embedding_channels = dict(
            zip(
                self.scale_order,
                _list(
                    model_cfg,
                    "graph_embedding_channels",
                    (32, 32, 32, 32),
                    int,
                ),
            )
        )
        maximum_hypothesis_delta = dict(
            zip(
                self.scale_order,
                _list(
                    model_cfg,
                    "max_hypothesis_delta",
                    (2.0, 2.0, 1.5, 1.0),
                    float,
                ),
            )
        )
        graph_dilations_cfg = _cfg_get(
            model_cfg,
            "graph_dilations",
            [[1, 2, 4], [1, 2, 4], [1, 2], [1, 2]],
        )
        if len(graph_dilations_cfg) != 4:
            raise ValueError("graph_dilations must contain four lists")
        graph_dilations = {
            scale: tuple(int(value) for value in values)
            for scale, values in zip(
                self.scale_order, graph_dilations_cfg
            )
        }
        common = dict(
            groups=groups,
            beta_initial=float(_cfg_get(model_cfg, "beta_initial", 0.10)),
            beta_minimum=float(_cfg_get(model_cfg, "beta_minimum", 0.01)),
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
            proximal_inertia_initial=float(
                _cfg_get(
                    model_cfg, "proximal_inertia_initial", 0.05
                )
            ),
            proximal_inertia_minimum=float(
                _cfg_get(
                    model_cfg, "proximal_inertia_minimum", 0.001
                )
            ),
            maximum_logit_adjustment=float(
                _cfg_get(model_cfg, "maximum_logit_adjustment", 2.0)
            ),
            precision_minimum=float(
                _cfg_get(model_cfg, "precision_minimum", 0.0)
            ),
            precision_maximum=float(
                _cfg_get(model_cfg, "precision_maximum", 1.0)
            ),
            precision_correlation_limit=float(
                _cfg_get(
                    model_cfg, "precision_correlation_limit", 0.95
                )
            ),
            initial_precision=float(
                _cfg_get(model_cfg, "initial_precision", 0.20)
            ),
            initial_matchability=float(
                _cfg_get(model_cfg, "initial_matchability", 0.90)
            ),
            confidence_floor=float(
                _cfg_get(model_cfg, "confidence_floor", 0.02)
            ),
            cycle_support_floor=float(
                _cfg_get(model_cfg, "cycle_support_floor", 0.02)
            ),
            graph_feature_temperature=float(
                _cfg_get(
                    model_cfg, "graph_feature_temperature", 0.25
                )
            ),
            graph_edge_alpha=float(
                _cfg_get(model_cfg, "graph_edge_alpha", 10.0)
            ),
            graph_anchor_floor=float(
                _cfg_get(model_cfg, "graph_anchor_floor", 0.05)
            ),
            correlation_attention_temperature=float(
                _cfg_get(
                    model_cfg,
                    "correlation_attention_temperature",
                    1.0,
                )
            ),
            maximum_attention_correlation_scale=float(
                _cfg_get(
                    model_cfg,
                    "maximum_attention_correlation_scale",
                    8.0,
                )
            ),
            responsibility_reference=str(
                _cfg_get(
                    model_cfg, "responsibility_reference", "proximal"
                )
            ),
        )
        self.cells = nn.ModuleDict(
            {
                str(scale): HQSCorrelationFieldCell(
                    correlation_channels=raw_correlation_channels[scale],
                    context_channels=context_by_scale[scale],
                    max_iterations=self.iterations_by_scale[scale],
                    radius=self.correlation_radii[scale],
                    num_hypotheses=self.num_hypotheses[scale],
                    mixture_temperature=self.mixture_temperatures[scale],
                    proposal_embedding_channels=(
                        proposal_embedding_channels[scale]
                    ),
                    proposal_hidden_channels=(
                        proposal_hidden_channels[scale]
                    ),
                    correlation_attention_channels=(
                        attention_channels[scale]
                    ),
                    correlation_attention_heads=attention_heads[scale],
                    graph_embedding_channels=(
                        graph_embedding_channels[scale]
                    ),
                    graph_dilations=graph_dilations[scale],
                    maximum_hypothesis_delta=(
                        maximum_hypothesis_delta[scale]
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
        self.cycle_sigma = float(
            _cfg_get(model_cfg, "cycle_sigma", 1.5)
        )
        self.cycle_confidence_floor = float(
            _cfg_get(model_cfg, "cycle_confidence_floor", 0.02)
        )
        self.cycle_scales = {
            int(value)
            for value in _cfg_get(model_cfg, "cycle_scales", [16, 8])
        }
        if not self.cycle_scales.issubset({16, 8}):
            raise ValueError("cycle_scales may contain only 16 and 8")
        self.allow_external_validity_inputs = bool(
            _cfg_get(model_cfg, "allow_external_validity_inputs", False)
        )
        self.solver_name = "hqs_field_of"
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

    def _encode_pair_features(
        self,
        image1: torch.Tensor,
        image2: torch.Tensor,
    ) -> Tuple[
        Dict[int, torch.Tensor],
        Dict[int, torch.Tensor],
        Dict[int, torch.Tensor],
        Dict[int, torch.Tensor],
    ]:
        """Encode a frame pair, allowing structured wrappers to reuse it."""
        cached = getattr(self, "_active_feature_cache", None)
        if cached is not None:
            return cached
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
                source_match_raw, target_match_raw
            )
        )
        return (
            source_backbone,
            target_backbone,
            source_match_raw,
            target_match_raw,
        )

    @staticmethod
    def _to_yx(flow_xy: torch.Tensor) -> torch.Tensor:
        return torch.stack((flow_xy[:, 1], flow_xy[:, 0]), dim=1)

    def _enhance_matching_features(
        self,
        source_features: Dict[int, torch.Tensor],
        target_features: Dict[int, torch.Tensor],
    ) -> Tuple[Dict[int, torch.Tensor], Dict[int, torch.Tensor]]:
        enhanced_source = dict(source_features)
        enhanced_target = dict(target_features)
        for scale in self.scale_order:
            key = str(scale)
            if key not in self.match_feature_transformers:
                continue
            with torch.autocast(
                device_type=source_features[scale].device.type,
                enabled=False,
            ):
                source_base = source_features[scale].float()
                target_base = target_features[scale].float()
                source_value, target_value = (
                    self.match_feature_transformers[key](
                        source_base, target_base
                    )
                )
                blend = torch.sigmoid(
                    self.match_feature_transformer_logits[key]
                ).to(source_value)
                enhanced_source[scale] = source_base + blend * (
                    source_value - source_base
                )
                enhanced_target[scale] = target_base + blend * (
                    target_value - target_base
                )
        return enhanced_source, enhanced_target

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
                raise RuntimeError(f"Missing all-pairs tensor at 1/{scale}")
            return all_pairs.lookup(flow_w)
        return self.local_correlations[str(scale)](
            source_features, target_features, flow_w
        )

    def _global_mixture_and_cycle(
        self,
        all_pairs: AllPairsCorrelation,
        *,
        scale: int,
        reference: torch.Tensor,
    ) -> Tuple[CorrelationMixture, torch.Tensor, torch.Tensor]:
        forward = all_pairs.global_topk_match(
            num_hypotheses=self.num_hypotheses[scale],
            temperature=self.global_temperature,
            query_chunk_size=self.global_query_chunk_size,
            reverse=False,
        )
        analytic = correlation_mixture_from_global(
            forward, reference=reference
        )
        forward_map = analytic.proposals[:, 0]
        if scale not in self.cycle_scales:
            cycle = torch.ones_like(analytic.in_bounds)
            reverse_map = torch.zeros_like(forward_map)
            return analytic, cycle, reverse_map

        reverse = all_pairs.global_topk_match(
            num_hypotheses=self.num_hypotheses[scale],
            temperature=self.global_temperature,
            query_chunk_size=self.global_query_chunk_size,
            reverse=True,
        )
        reverse_map = reverse["hypotheses"][:, 0].to(reference)
        reverse_confidence = reverse["confidence"].to(reference)
        cycle_modes = []
        for index in range(analytic.proposals.shape[1]):
            candidate = analytic.proposals[:, index]
            warped_reverse = backward_warp(
                reverse_map, candidate, padding_mode="zeros"
            )
            warped_reverse_confidence = backward_warp(
                reverse_confidence,
                candidate,
                padding_mode="zeros",
            )
            cycle_error = torch.sqrt(
                (candidate + warped_reverse).square().sum(
                    dim=1, keepdim=True
                )
                + 1e-8
            )
            confidence = self.cycle_confidence_floor + (
                1.0 - self.cycle_confidence_floor
            ) * (
                analytic.confidence
                * warped_reverse_confidence.clamp(0.0, 1.0)
            )
            cycle_modes.append(
                (
                    torch.exp(
                        -cycle_error / max(self.cycle_sigma, 1e-4)
                    )
                    * flow_in_bounds_mask(candidate)
                    * confidence
                ).clamp(0.0, 1.0)
            )
        cycle = torch.stack(cycle_modes, dim=1)
        return analytic, cycle, reverse_map

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
                "HQS-Field-OF expects equal [B,3,H,W] inputs"
            )
        if image1.shape[1] != 3:
            raise ValueError("HQS-Field-OF expects RGB inputs")
        if iters is not None and int(iters) != self.num_hqs_iterations:
            raise ValueError(
                "Set hqs_field_of.iterations rather than forward(iters=...)"
            )
        if (source_valid is not None or target_valid is not None) and not (
            self.allow_external_validity_inputs
        ):
            raise ValueError(
                "Ground-truth visibility is loss-only for HQS-Field-OF"
            )

        image1 = image1.float()
        image2 = image2.float()
        _, _, full_height, full_width = image1.shape
        (
            source_backbone,
            target_backbone,
            source_match_raw,
            target_match_raw,
        ) = self._encode_pair_features(
            image1,
            image2,
        )
        source_match = {
            scale: F.normalize(value.float(), dim=1, eps=1e-6)
            for scale, value in source_match_raw.items()
        }
        target_match = {
            scale: F.normalize(value.float(), dim=1, eps=1e-6)
            for scale, value in target_match_raw.items()
        }
        # This context is extracted only from I1 and is the sole source of
        # graph affinities used by the field proximal.
        source_context = self.feature_encoder.project_context(source_backbone)
        source_gray = self._gray(image1)
        source_guidance = {
            scale: F.interpolate(
                source_gray,
                size=source_match[scale].shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            for scale in self.scale_order
        }

        all_pairs = AllPairsCorrelation(
            source_match[16],
            target_match[16],
            num_levels=self.all_pairs_levels[16],
            radius=self.correlation_radii[16],
        )
        global_mixture, cycle_support, reverse_map = (
            self._global_mixture_and_cycle(
                all_pairs,
                scale=16,
                reference=source_match[16],
            )
        )
        if flow_init is None:
            # MAP initialisation avoids the under-motion produced by averaging
            # spatially separated global modes.
            initial = global_mixture.proposals[:, 0]
        else:
            initial = resize_flow(
                flow_init.to(device=image1.device, dtype=image1.dtype),
                source_match[16].shape[-2:],
            )
        state = LMState(w=initial, z=initial)

        flow_predictions: List[torch.Tensor] = []
        flow_lows: List[torch.Tensor] = []
        data_flow_lows: List[torch.Tensor] = []
        delta_lows: List[torch.Tensor] = []
        coupling_lows: List[torch.Tensor] = []
        data_delta_lows: List[torch.Tensor] = []
        posterior_proposals: List[torch.Tensor] = []
        collapsed_precisions: List[torch.Tensor] = []
        hypothesis_proposals: List[torch.Tensor] = []
        hypothesis_logits: List[torch.Tensor] = []
        hypothesis_probabilities: List[torch.Tensor] = []
        hypothesis_responsibilities: List[torch.Tensor] = []
        hypothesis_precisions: List[torch.Tensor] = []
        learned_hypothesis_deltas: List[torch.Tensor] = []
        analytic_hypotheses: List[torch.Tensor] = []
        matchabilities: List[torch.Tensor] = []
        measurement_supports: List[torch.Tensor] = []
        cycle_supports: List[torch.Tensor] = []
        responsibility_entropies: List[torch.Tensor] = []
        inverse_traces: List[torch.Tensor] = []
        conditions: List[torch.Tensor] = []
        attention_entropies: List[torch.Tensor] = []
        attention_peaks: List[torch.Tensor] = []
        beta_values: List[torch.Tensor] = []
        damping_values: List[torch.Tensor] = []
        lambda_values: List[torch.Tensor] = []
        inertia_values: List[torch.Tensor] = []
        prediction_scales: List[int] = []
        global_initial = initial
        reverse_flow_by_scale: Dict[int, torch.Tensor] = {16: reverse_map}
        global_mixture_by_scale: Dict[int, CorrelationMixture] = {
            16: global_mixture
        }
        global_cycle_modes_by_scale: Dict[int, torch.Tensor] = {
            16: cycle_support
        }
        cycle_by_scale: Dict[int, torch.Tensor] = {
            16: cycle_support.max(dim=1).values
        }

        for scale_index, scale in enumerate(self.scale_order):
            if scale_index > 0:
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
                    global_scale_mixture, cycle_support, reverse_map = (
                        self._global_mixture_and_cycle(
                            all_pairs,
                            scale=8,
                            reference=source_match[8],
                        )
                    )
                    global_mixture_by_scale[8] = global_scale_mixture
                    global_cycle_modes_by_scale[8] = cycle_support
                    cycle_by_scale[8] = cycle_support.max(dim=1).values
                    reverse_flow_by_scale[8] = reverse_map
                else:
                    all_pairs = None
                    parent = 8 if 8 in cycle_by_scale else 16
                    cycle_by_scale[scale] = F.interpolate(
                        cycle_by_scale[parent],
                        size=target_size,
                        mode="bilinear",
                        align_corners=False,
                    ).clamp(0.0, 1.0)

            cell = self.cells[str(scale)]
            proposal_hidden = cell.initialise_proposal_hidden(
                source_context[scale]
            )
            graph_embedding = cell.prepare_source_graph(
                source_context[scale]
            )
            for iteration in range(self.iterations_by_scale[scale]):
                previous_z = state.z
                correlation = self._raw_correlation(
                    scale,
                    state.w,
                    source_match[scale],
                    target_match[scale],
                    all_pairs,
                )
                analytic_override = None
                if iteration == 0 and scale in global_mixture_by_scale:
                    analytic_override = global_mixture_by_scale[scale]
                cycle_input = (
                    global_cycle_modes_by_scale[scale]
                    if analytic_override is not None
                    else cycle_by_scale[scale]
                )
                update = cell(
                    correlation=correlation,
                    source_context=source_context[scale],
                    source_guidance=source_guidance[scale],
                    source_graph_embedding=graph_embedding,
                    cycle_support=cycle_input,
                    state=state,
                    iteration=iteration,
                    graph_sweeps=self.graph_sweeps[scale],
                    max_data_delta=self.max_data_delta[scale],
                    proposal_hidden=proposal_hidden,
                    analytic_mixture=analytic_override,
                )
                proposal_hidden = update.proposal_hidden
                state = update.state
                flow_lows.append(state.z)
                data_flow_lows.append(state.w)
                delta_lows.append(state.z - previous_z)
                coupling_lows.append(state.w - state.z)
                data_delta_lows.append(update.data_delta)
                posterior_proposals.append(update.posterior_proposal)
                collapsed_precisions.append(update.collapsed_precision)
                hypothesis_proposals.append(update.hypotheses)
                hypothesis_logits.append(update.hypothesis_logits)
                hypothesis_probabilities.append(
                    update.hypothesis_probabilities
                )
                hypothesis_responsibilities.append(
                    update.hypothesis_responsibilities
                )
                hypothesis_precisions.append(update.hypothesis_precision)
                learned_hypothesis_deltas.append(
                    update.learned_hypothesis_deltas
                )
                analytic_hypotheses.append(update.analytic_hypotheses)
                matchabilities.append(update.matchability)
                measurement_supports.append(update.measurement_support)
                cycle_supports.append(update.cycle_support)
                responsibility_entropies.append(
                    update.responsibility_entropy
                )
                inverse_traces.append(update.inverse_trace)
                conditions.append(update.condition)
                attention_entropies.append(
                    update.correlation_attention_entropy
                )
                attention_peaks.append(
                    update.correlation_attention_peak
                )
                beta_values.append(update.beta)
                damping_values.append(update.damping)
                lambda_values.append(update.regularisation)
                inertia_values.append(update.proximal_inertia)
                prediction_scales.append(scale)
                flow_predictions.append(
                    resize_flow(state.z, (full_height, full_width))
                )

        raw_flow_predictions = list(flow_predictions)
        final_flow, final_mask_logits = self.final_upsampler(
            state.z, source_context[2], source_guidance[2]
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
        zero_data_bypass = [
            torch.zeros_like(value) for value in data_delta_lows
        ]
        reverse_finest = reverse_flow_by_scale.get(
            8, reverse_flow_by_scale[16]
        )
        reverse_full = resize_flow(
            reverse_finest, (full_height, full_width)
        )
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
            "learned_data_delta_lows": zero_data_bypass,
            "match_proposal_lows": posterior_proposals,
            "match_precision_lows": collapsed_precisions,
            "matchability_lows": matchabilities,
            "data_reliability_lows": measurement_supports,
            "core_reliability_lows": measurement_supports,
            "core_validity_lows": measurement_supports,
            "data_valid_lows": measurement_supports,
            "data_weight_lows": measurement_supports,
            "hypothesis_proposal_lows": hypothesis_proposals,
            "hypothesis_logits_lows": hypothesis_logits,
            "hypothesis_probability_lows": hypothesis_probabilities,
            "hypothesis_responsibility_lows": (
                hypothesis_responsibilities
            ),
            "hypothesis_precision_lows": hypothesis_precisions,
            "learned_hypothesis_delta_lows": (
                learned_hypothesis_deltas
            ),
            "analytic_hypothesis_lows": analytic_hypotheses,
            "measurement_support_lows": measurement_supports,
            "cycle_support_lows": cycle_supports,
            "responsibility_entropy_lows": responsibility_entropies,
            "lm_inverse_trace_lows": inverse_traces,
            "lm_condition_lows": conditions,
            "correlation_attention_entropy_lows": attention_entropies,
            "correlation_attention_peak_lows": attention_peaks,
            "beta_values": beta_values,
            "lm_damping_values": damping_values,
            "lambda_values": lambda_values,
            "proximal_inertia_values": inertia_values,
            "prediction_scales": prediction_scales,
            "final_upsample_mask_logits": final_mask_logits,
            "flow_final_raw": raw_flow_predictions[-1],
            "flow_final_refined": final_flow,
            "global_init_flow_xy": global_initial,
            "global_init_confidence": global_mixture.confidence,
            "global_init_entropy": global_mixture.entropy,
            "global_init_margin": global_mixture.margin,
            "gmflow_init_flow_yx": self._to_yx(global_initial),
            "gmflow_init_conf": global_mixture.confidence,
            "gmflow_init_entropy": global_mixture.entropy,
            "gmflow_init_margin": global_mixture.margin,
            "gmflow_reverse_flow_yx": self._to_yx(reverse_full),
            "occupancy_masks": [],
            "reliability_states": [],
            "reliability_geometries": [],
            "reliability_residuals": [],
            "factorised_data_gates": [],
            "band_split": None,
            "feature_transformer_blends": {
                key: torch.sigmoid(value)
                for key, value in (
                    self.match_feature_transformer_logits.items()
                )
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


__all__ = ["HQSFieldOpticalFlow"]
