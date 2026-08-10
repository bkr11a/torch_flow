"""Compact multi-scale operator-structured HQS optical-flow architecture.

HQSCore is an isolated experimental architecture.  It does not branch through
``HQSFlowModelTFPort`` and is selected only with ``model.model_type:
hqs_core``.  Its recurrent computation is explicitly organised as

    q, w, h -> warp -> linearised data update -> proximal update -> q', w', h'

at four scales (1/16, 1/8, 1/4 and 1/2).  The data and proximal operators use
separate interfaces so target-derived evidence cannot bypass the HQS split.
"""
from __future__ import annotations

from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.correlation import LocalCorrBlock
from models.hqs_core_components import (
    AllPairsCorrelation,
    CorrelationAdapter,
    HQSCorePyramidEncoder,
    HQSState,
    SharedValidityHead,
    SourceGuidedConvexUpsampler,
    StructuredHQSCell,
    spatial_gradients,
)
from models.hqs_deep_match_components import (
    HQSCoreDeepMatchingPyramid,
    decode_cycle_consistent_topk,
)
from models.warp import resize_flow


def _cfg_get(cfg, key: str, default):
    if cfg is None:
        return default
    if isinstance(cfg, Mapping):
        return cfg.get(key, default)
    try:
        return cfg.get(key, default)
    except Exception:
        return getattr(cfg, key, default)


def _int_list(cfg, key: str, default: Sequence[int]) -> Tuple[int, ...]:
    value = _cfg_get(cfg, key, default)
    result = tuple(int(v) for v in value)
    if len(result) != len(default):
        raise ValueError(
            f"hqs_core.{key} must contain {len(default)} entries, got {value}"
        )
    return result


def _float_list(
    cfg, key: str, default: Sequence[float]
) -> Tuple[float, ...]:
    value = _cfg_get(cfg, key, default)
    result = tuple(float(v) for v in value)
    if len(result) != len(default):
        raise ValueError(
            f"hqs_core.{key} must contain {len(default)} entries, got {value}"
        )
    return result


class HQSCore(nn.Module):
    """Four-scale structured-operator optical-flow solver.

    Resolution order and default recurrence budget are:

    - 1/16: two coarse-prior iterations;
    - 1/8: four moderate/coarse refinement iterations;
    - 1/4: two fine/small-object iterations;
    - 1/2: two final detail iterations.

    Coarse scales share one HQS cell and fine scales share another.  Scale
    changes resize the physically interpreted flow state and reset the data
    hidden state from the new source-only context.
    """

    scale_order: Tuple[int, ...] = (16, 8, 4, 2)

    def __init__(self, cfg) -> None:
        super().__init__()
        core_cfg = _cfg_get(cfg, "hqs_core", cfg)
        self.cfg = core_cfg

        # Encoder lists are fine-to-coarse: [1/2, 1/4, 1/8, 1/16].
        feature_channels = _int_list(
            core_cfg, "feature_channels", (32, 64, 96, 128)
        )
        match_channels = _int_list(
            core_cfg, "match_channels", (64, 96, 128, 128)
        )
        context_channels = _int_list(
            core_cfg, "context_channels", (64, 64, 96, 96)
        )
        blocks_per_scale = _int_list(
            core_cfg, "blocks_per_scale", (1, 2, 2, 2)
        )
        groups = int(_cfg_get(core_cfg, "groups", 8))

        # Iteration/operator lists are coarse-to-fine: [1/16,1/8,1/4,1/2].
        iterations = _int_list(core_cfg, "iterations", (2, 4, 2, 2))
        if any(v < 0 for v in iterations):
            raise ValueError(f"All HQSCore iteration counts must be non-negative: {iterations}")
        if sum(iterations) == 0:
            raise ValueError(f"HQSCore must have at least one iteration: {iterations}")
        self.iterations_by_scale = dict(zip(self.scale_order, iterations))
        self.num_hqs_iterations = sum(iterations)
        self.jacobi_sweeps = dict(
            zip(
                self.scale_order,
                _int_list(core_cfg, "jacobi_sweeps", (2, 2, 1, 1)),
            )
        )
        self.max_data_delta = dict(
            zip(
                self.scale_order,
                _float_list(
                    core_cfg, "max_data_delta", (2.0, 2.0, 1.5, 1.0)
                ),
            )
        )
        self.max_prox_delta = dict(
            zip(
                self.scale_order,
                _float_list(
                    core_cfg, "max_prox_delta", (0.5, 0.5, 0.5, 0.25)
                ),
            )
        )
        correlation_radii = dict(
            zip(
                self.scale_order,
                _int_list(core_cfg, "correlation_radii", (4, 4, 4, 3)),
            )
        )
        all_pairs_levels = dict(
            zip(
                (16, 8),
                _int_list(core_cfg, "all_pairs_levels", (1, 4)),
            )
        )
        self.correlation_radii = correlation_radii
        self.all_pairs_levels = all_pairs_levels
        local_corr_channel_chunk = int(
            _cfg_get(core_cfg, "local_corr_channel_chunk", 4)
        )
        local_corr_checkpoint = bool(
            _cfg_get(core_cfg, "local_corr_checkpoint", True)
        )
        self.correlation_embedding_dim = int(
            _cfg_get(core_cfg, "correlation_embedding_dim", 64)
        )

        self.matching_pyramid = str(
            _cfg_get(core_cfg, "matching_pyramid", "standard")
        ).lower()
        if self.matching_pyramid in {
            "deep_bidirectional",
            "deep_match",
            "strong",
        }:
            self.feature_encoder = HQSCoreDeepMatchingPyramid(
                feature_channels=feature_channels,
                match_channels=match_channels,
                context_channels=context_channels,
                blocks_per_scale=blocks_per_scale,
                groups=groups,
                fusion_depth=int(
                    _cfg_get(core_cfg, "matching_fusion_depth", 2)
                ),
                transformer_depth=_int_list(
                    core_cfg,
                    "matching_transformer_depth",
                    (0, 0, 2, 4),
                ),
                transformer_heads=_int_list(
                    core_cfg,
                    "matching_transformer_heads",
                    (4, 4, 8, 8),
                ),
                transformer_windows=_int_list(
                    core_cfg,
                    "matching_transformer_windows",
                    (0, 0, 8, 0),
                ),
                transformer_maximum_global_tokens=int(
                    _cfg_get(
                        core_cfg,
                        "matching_transformer_maximum_global_tokens",
                        4096,
                    )
                ),
                transformer_fallback_window=int(
                    _cfg_get(
                        core_cfg,
                        "matching_transformer_fallback_window",
                        16,
                    )
                ),
                transformer_mlp_ratio=float(
                    _cfg_get(
                        core_cfg,
                        "matching_transformer_mlp_ratio",
                        4.0,
                    )
                ),
                transformer_initial_blend=float(
                    _cfg_get(
                        core_cfg,
                        "matching_transformer_initial_blend",
                        0.25,
                    )
                ),
                transformer_gradient_checkpointing=bool(
                    _cfg_get(
                        core_cfg,
                        "matching_transformer_gradient_checkpointing",
                        True,
                    )
                ),
            )
        elif self.matching_pyramid == "standard":
            self.feature_encoder = HQSCorePyramidEncoder(
                feature_channels=feature_channels,
                match_channels=match_channels,
                context_channels=context_channels,
                blocks_per_scale=blocks_per_scale,
                groups=groups,
            )
        else:
            raise ValueError(
                "hqs_core.matching_pyramid must be standard or "
                f"deep_bidirectional, got {self.matching_pyramid!r}"
            )
        # Used only when Trainer's optional matcher warm-up is enabled.
        self.global_matcher_parameter_prefixes = ("feature_encoder.",)

        raw_corr_channels = {
            16: all_pairs_levels[16] * (2 * correlation_radii[16] + 1) ** 2,
            8: all_pairs_levels[8] * (2 * correlation_radii[8] + 1) ** 2,
            4: (2 * correlation_radii[4] + 1) ** 2,
            2: (2 * correlation_radii[2] + 1) ** 2,
        }
        self.correlation_adapters = nn.ModuleDict(
            {
                str(scale): CorrelationAdapter(
                    raw_corr_channels[scale],
                    self.correlation_embedding_dim,
                    groups=groups,
                )
                for scale in self.scale_order
            }
        )
        self.local_correlations = nn.ModuleDict(
            {
                "4": LocalCorrBlock(
                    radius=correlation_radii[4],
                    channel_chunk_size=local_corr_channel_chunk,
                    checkpoint_chunks=local_corr_checkpoint,
                ),
                "2": LocalCorrBlock(
                    radius=correlation_radii[2],
                    channel_chunk_size=local_corr_channel_chunk,
                    checkpoint_chunks=local_corr_checkpoint,
                ),
            }
        )

        # Context arrays are indexed fine-to-coarse.
        context_by_scale = dict(zip((2, 4, 8, 16), context_channels))
        if context_by_scale[16] != context_by_scale[8]:
            raise ValueError(
                "1/16 and 1/8 context dimensions must match for coarse-cell sharing"
            )
        if context_by_scale[4] != context_by_scale[2]:
            raise ValueError(
                "1/4 and 1/2 context dimensions must match for fine-cell sharing"
            )

        common_cell_args = dict(
            correlation_channels=self.correlation_embedding_dim,
            prior_hidden_channels=int(
                _cfg_get(core_cfg, "prior_hidden_channels", 64)
            ),
            groups=groups,
            beta_initial=float(_cfg_get(core_cfg, "beta_initial", 0.10)),
            beta_minimum=float(_cfg_get(core_cfg, "beta_minimum", 0.01)),
            lambda_initial=float(
                _cfg_get(core_cfg, "lambda_initial", 0.08)
            ),
            lambda_minimum=float(
                _cfg_get(core_cfg, "lambda_minimum", 0.001)
            ),
            edge_alpha=float(_cfg_get(core_cfg, "edge_alpha", 10.0)),
            analytic_weight=float(
                _cfg_get(core_cfg, "analytic_weight", 1.0)
            ),
            learned_data_weight=float(
                _cfg_get(core_cfg, "learned_data_weight", 1.0)
            ),
            analytic_validity_mode=str(
                _cfg_get(core_cfg, "analytic_validity_mode", "post_gate")
            ),
        )
        self.coarse_cell = StructuredHQSCell(
            context_channels=context_by_scale[16],
            hidden_channels=int(_cfg_get(core_cfg, "coarse_hidden_dim", 96)),
            max_iterations=max(iterations[0], iterations[1]),
            **common_cell_args,
        )
        self.fine_cell = StructuredHQSCell(
            context_channels=context_by_scale[4],
            hidden_channels=int(_cfg_get(core_cfg, "fine_hidden_dim", 64)),
            max_iterations=max(iterations[2], iterations[3]),
            **common_cell_args,
        )
        self.validity_head = SharedValidityHead(
            correlation_channels=self.correlation_embedding_dim,
            hidden_channels=int(
                _cfg_get(core_cfg, "validity_hidden_dim", 32)
            ),
            initial_reliability=float(
                _cfg_get(core_cfg, "initial_reliability", 0.90)
            ),
        )
        self.final_upsampler = SourceGuidedConvexUpsampler(
            context_channels=context_by_scale[2],
            hidden_channels=int(
                _cfg_get(core_cfg, "upsample_hidden_dim", 64)
            ),
            groups=groups,
            rate=2,
        )

        self.global_temperature = float(
            _cfg_get(core_cfg, "global_temperature", 0.07)
        )
        self.global_query_chunk_size = int(
            _cfg_get(core_cfg, "global_query_chunk_size", 512)
        )
        self.global_match_scale = int(
            _cfg_get(core_cfg, "global_match_scale", 16)
        )
        if self.global_match_scale not in (16, 8):
            raise ValueError(
                "hqs_core.global_match_scale must be 16 or 8, got "
                f"{self.global_match_scale}"
            )
        self.global_confidence_gated = bool(
            _cfg_get(core_cfg, "global_confidence_gated", True)
        )
        self.global_confidence_floor = float(
            _cfg_get(core_cfg, "global_confidence_floor", 0.05)
        )
        self.global_decoder = str(
            _cfg_get(core_cfg, "global_decoder", "soft_expectation")
        ).lower()
        if self.global_decoder not in {
            "soft_expectation",
            "multimodal_cycle",
        }:
            raise ValueError(
                "hqs_core.global_decoder must be soft_expectation or "
                f"multimodal_cycle, got {self.global_decoder!r}"
            )
        self.global_num_hypotheses = int(
            _cfg_get(core_cfg, "global_num_hypotheses", 4)
        )
        self.global_local_expectation_radius = int(
            _cfg_get(core_cfg, "global_local_expectation_radius", 2)
        )
        self.global_nms_radius = int(
            _cfg_get(core_cfg, "global_nms_radius", 3)
        )
        self.global_cycle_sigma = float(
            _cfg_get(core_cfg, "global_cycle_sigma", 1.5)
        )
        self.global_cycle_score_weight = float(
            _cfg_get(core_cfg, "global_cycle_score_weight", 1.0)
        )
        self.global_cycle_confidence_floor = float(
            _cfg_get(core_cfg, "global_cycle_confidence_floor", 0.02)
        )
        self.global_minimum_initialisation_confidence = float(
            _cfg_get(
                core_cfg,
                "global_minimum_initialisation_confidence",
                0.12,
            )
        )
        self.global_gate_temperature = float(
            _cfg_get(core_cfg, "global_gate_temperature", 0.05)
        )
        self.allow_external_validity_inputs = bool(
            _cfg_get(core_cfg, "allow_external_validity_inputs", False)
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
        weights = image.new_tensor((0.2989, 0.5870, 0.1140)).view(1, 3, 1, 1)
        return (image * weights).sum(dim=1, keepdim=True)

    def _normalise(self, image: torch.Tensor) -> torch.Tensor:
        return (image - self.image_mean.to(image)) / self.image_std.to(image)

    @staticmethod
    def _to_yx(flow_xy: torch.Tensor) -> torch.Tensor:
        return torch.stack((flow_xy[:, 1], flow_xy[:, 0]), dim=1)

    def _cell_for_scale(self, scale: int) -> StructuredHQSCell:
        return self.coarse_cell if scale in (16, 8) else self.fine_cell

    def _correlation_features(
        self,
        scale: int,
        flow_q: torch.Tensor,
        match1: torch.Tensor,
        match2: torch.Tensor,
        all_pairs: Optional[AllPairsCorrelation],
    ) -> torch.Tensor:
        if scale in (16, 8):
            if all_pairs is None:
                raise RuntimeError(f"Missing all-pairs correlation at 1/{scale}")
            raw = all_pairs.lookup(flow_q)
        else:
            raw = self.local_correlations[str(scale)](
                match1, match2, flow_q
            )
        return self.correlation_adapters[str(scale)](raw)

    def _encode_matching_pair(
        self,
        source_backbone: Dict[int, torch.Tensor],
        target_backbone: Dict[int, torch.Tensor],
    ) -> Tuple[Dict[int, torch.Tensor], Dict[int, torch.Tensor]]:
        source_match = self.feature_encoder.project_matching(
            source_backbone
        )
        target_match = self.feature_encoder.project_matching(
            target_backbone
        )
        if isinstance(
            self.feature_encoder,
            HQSCoreDeepMatchingPyramid,
        ):
            source_match, target_match = (
                self.feature_encoder.enhance_pair(
                    source_match,
                    target_match,
                )
            )
        return source_match, target_match

    def _decode_global_match(
        self,
        correlation: AllPairsCorrelation,
    ) -> Dict[str, object]:
        if self.global_decoder == "soft_expectation":
            soft = correlation.global_soft_match(
                temperature=self.global_temperature,
                query_chunk_size=self.global_query_chunk_size,
            )
            ones = torch.ones_like(soft["confidence"])
            zeros = torch.zeros_like(soft["confidence"])
            return {
                "initial_flow": soft["flow_xy"],
                "selected_flow": soft["flow_xy"],
                "selected_index": torch.zeros_like(
                    soft["confidence"],
                    dtype=torch.long,
                ),
                "selected_probability": ones,
                "selected_confidence": soft["confidence"],
                "selected_cycle_support": ones,
                "selected_cycle_error": zeros,
                "acceptance": ones,
                "hard_acceptance": ones,
                "soft_acceptance": ones,
                "cycle_support": ones.unsqueeze(1),
                "cycle_error": zeros.unsqueeze(1),
                "reverse_flow": torch.zeros_like(soft["flow_xy"]),
                "reverse_confidence": zeros,
                "forward": {
                    "hypotheses": soft["flow_xy"].unsqueeze(1),
                    "map_hypotheses": soft["flow_xy"].unsqueeze(1),
                    "probabilities": ones,
                    "confidence": soft["confidence"],
                    "entropy": soft["entropy"],
                    "margin": soft["margin"],
                    "peak": soft["confidence"],
                    "retained_mass": ones,
                },
                "reverse": None,
            }

        forward = correlation.global_multimodal_match(
            num_hypotheses=self.global_num_hypotheses,
            temperature=self.global_temperature,
            query_chunk_size=self.global_query_chunk_size,
            local_expectation_radius=(
                self.global_local_expectation_radius
            ),
            nms_radius=self.global_nms_radius,
            reverse=False,
        )
        reverse = correlation.global_multimodal_match(
            num_hypotheses=self.global_num_hypotheses,
            temperature=self.global_temperature,
            query_chunk_size=self.global_query_chunk_size,
            local_expectation_radius=(
                self.global_local_expectation_radius
            ),
            nms_radius=self.global_nms_radius,
            reverse=True,
        )
        decoded: Dict[str, object] = decode_cycle_consistent_topk(
            forward,
            reverse,
            cycle_sigma=self.global_cycle_sigma,
            cycle_score_weight=self.global_cycle_score_weight,
            cycle_confidence_floor=self.global_cycle_confidence_floor,
            minimum_initialisation_confidence=(
                self.global_minimum_initialisation_confidence
            ),
            gate_temperature=self.global_gate_temperature,
        )
        decoded["forward"] = forward
        decoded["reverse"] = reverse
        return decoded

    def forward(
        self,
        image1: torch.Tensor,
        image2: torch.Tensor,
        iters: Optional[int] = None,
        flow_init: Optional[torch.Tensor] = None,
        source_valid: Optional[torch.Tensor] = None,
        target_valid: Optional[torch.Tensor] = None,
    ) -> Dict[str, object]:
        if image1.shape != image2.shape or image1.ndim != 4:
            raise ValueError(
                "HQSCore expects equal image tensors [B,3,H,W], got "
                f"{tuple(image1.shape)} and {tuple(image2.shape)}"
            )
        if image1.shape[1] != 3:
            raise ValueError(f"HQSCore expects RGB images, got {image1.shape[1]} channels")
        if iters is not None and int(iters) != self.num_hqs_iterations:
            raise ValueError(
                "HQSCore uses a per-scale iteration plan. Set hqs_core.iterations "
                f"instead of forward(iters=...); configured total={self.num_hqs_iterations}."
            )
        if (source_valid is not None or target_valid is not None) and not (
            self.allow_external_validity_inputs
        ):
            raise ValueError(
                "External/ground-truth validity must not enter HQSCore.forward(). "
                "Use those masks only as loss targets."
            )

        image1 = image1.float()
        image2 = image2.float()
        batch, _, full_h, full_w = image1.shape
        normalised1 = self._normalise(image1)
        normalised2 = self._normalise(image2)

        source_backbone = self.feature_encoder.backbone(normalised1)
        target_backbone = self.feature_encoder.backbone(normalised2)
        source_match, target_match = self._encode_matching_pair(
            source_backbone,
            target_backbone,
        )
        # This is the only context/prior feature pyramid in the model and it
        # is computed exclusively from the source image.
        source_context = self.feature_encoder.project_context(source_backbone)

        gray1_full = self._gray(image1)
        gray2_full = self._gray(image2)
        source_gray: Dict[int, torch.Tensor] = {}
        target_gray: Dict[int, torch.Tensor] = {}
        target_gradients: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}
        for scale in self.scale_order:
            size = source_match[scale].shape[-2:]
            source_gray[scale] = F.interpolate(
                gray1_full, size=size, mode="bilinear", align_corners=False
            )
            target_gray[scale] = F.interpolate(
                gray2_full, size=size, mode="bilinear", align_corners=False
            )
            target_gradients[scale] = spatial_gradients(target_gray[scale])

        # The recurrent solver uses all-pairs lookup at both coarse scales.
        # Global matching may be decoded at 1/16 (legacy) or 1/8.  Construct
        # the selected global volume once and reuse it when the solver reaches
        # that scale.
        all_pairs_by_scale: Dict[int, AllPairsCorrelation] = {
            16: AllPairsCorrelation(
                source_match[16],
                target_match[16],
                num_levels=self.all_pairs_levels[16],
                radius=self.correlation_radii[16],
            )
        }
        if self.global_match_scale == 8:
            all_pairs_by_scale[8] = AllPairsCorrelation(
                source_match[8],
                target_match[8],
                num_levels=self.all_pairs_levels[8],
                radius=self.correlation_radii[8],
            )
        global_correlation = all_pairs_by_scale[self.global_match_scale]
        global_match = self._decode_global_match(global_correlation)
        matcher_candidate = global_match["selected_flow"]
        if not isinstance(matcher_candidate, torch.Tensor):
            raise RuntimeError("Global decoder returned an invalid candidate")
        if flow_init is not None:
            if flow_init.ndim != 4 or flow_init.shape[1] != 2:
                raise ValueError(
                    f"flow_init must be [B,2,H,W] in (x,y) order, got {tuple(flow_init.shape)}"
                )
            q_initial = resize_flow(
                flow_init.to(device=image1.device, dtype=image1.dtype),
                source_match[16].shape[-2:],
            )
        else:
            decoded_initial = global_match["initial_flow"]
            if not isinstance(decoded_initial, torch.Tensor):
                raise RuntimeError("Global decoder returned an invalid flow")
            admitted_initial = decoded_initial
            if (
                self.global_decoder == "soft_expectation"
                and self.global_confidence_gated
            ):
                floor = min(max(self.global_confidence_floor, 0.0), 1.0)
                selected_confidence = global_match[
                    "selected_confidence"
                ]
                if not isinstance(selected_confidence, torch.Tensor):
                    raise RuntimeError(
                        "Global decoder returned invalid confidence"
                    )
                gate = floor + (1.0 - floor) * selected_confidence
                admitted_initial = gate * admitted_initial
            # The global decoder returns flow in pixels of its native matching
            # grid.  The first HQS state is always 1/16, so a 1/8 candidate
            # must be spatially resized and have its vector components scaled.
            q_initial = resize_flow(
                admitted_initial,
                source_match[16].shape[-2:],
            )

        state: Optional[HQSState] = None
        flow_predictions: List[torch.Tensor] = []
        flow_lows: List[torch.Tensor] = []
        data_flow_lows: List[torch.Tensor] = []
        coupling_lows: List[torch.Tensor] = []
        delta_lows: List[torch.Tensor] = []
        hidden_states: List[torch.Tensor] = []
        reliability_lows: List[torch.Tensor] = []
        validity_lows: List[torch.Tensor] = []
        analytic_delta_lows: List[torch.Tensor] = []
        learned_delta_lows: List[torch.Tensor] = []
        data_delta_lows: List[torch.Tensor] = []
        proximal_anchor_lows: List[torch.Tensor] = []
        beta_values: List[torch.Tensor] = []
        lambda_values: List[torch.Tensor] = []
        prediction_scales: List[int] = []

        for scale_index, scale in enumerate(self.scale_order):
            cell = self._cell_for_scale(scale)
            if scale_index == 0:
                state = cell.initialise_state(source_context[scale], q_initial)
                all_pairs: Optional[AllPairsCorrelation] = (
                    all_pairs_by_scale[16]
                )
            else:
                assert state is not None
                transitioned_q = resize_flow(
                    state.q, source_match[scale].shape[-2:]
                )
                # HQS scale transition: q is the transferred solution, w is
                # reset to q, and hidden is reset from source context.
                state = cell.initialise_state(
                    source_context[scale], transitioned_q
                )
                if scale == 8:
                    if 8 not in all_pairs_by_scale:
                        all_pairs_by_scale[8] = AllPairsCorrelation(
                            source_match[8],
                            target_match[8],
                            num_levels=self.all_pairs_levels[8],
                            radius=self.correlation_radii[8],
                        )
                    all_pairs = all_pairs_by_scale[8]
                else:
                    all_pairs = None

            for iteration in range(self.iterations_by_scale[scale]):
                assert state is not None
                q_previous = state.q
                corr_features = self._correlation_features(
                    scale,
                    state.q,
                    source_match[scale],
                    target_match[scale],
                    all_pairs,
                )
                grad_x, grad_y = target_gradients[scale]
                update = cell(
                    source_gray=source_gray[scale],
                    target_gray=target_gray[scale],
                    target_grad_x=grad_x,
                    target_grad_y=grad_y,
                    correlation=corr_features,
                    source_context=source_context[scale],
                    state=state,
                    validity_head=self.validity_head,
                    iteration=iteration,
                    jacobi_sweeps=self.jacobi_sweeps[scale],
                    max_data_delta=self.max_data_delta[scale],
                    max_prox_delta=self.max_prox_delta[scale],
                )
                state = update.state

                flow_lows.append(state.q)
                data_flow_lows.append(state.w)
                coupling_lows.append(state.w - state.q)
                delta_lows.append(state.q - q_previous)
                hidden_states.append(state.hidden)
                reliability_lows.append(update.reliability)
                validity_lows.append(update.validity)
                analytic_delta_lows.append(update.analytic_delta)
                learned_delta_lows.append(update.learned_delta)
                data_delta_lows.append(update.data_delta)
                proximal_anchor_lows.append(update.proximal_anchor)
                beta_values.append(update.beta)
                lambda_values.append(update.regularisation)
                prediction_scales.append(scale)
                flow_predictions.append(
                    resize_flow(state.q, (full_h, full_w))
                )

        assert state is not None
        raw_flow_predictions = list(flow_predictions)
        final_flow, final_mask_logits = self.final_upsampler(
            state.q, source_context[2], source_gray[2]
        )
        # Odd unpadded inputs can produce one extra row/column after stride-2
        # encoding.  Training/evaluation pad to multiples of eight, but keep
        # direct model use shape-safe as well.
        if final_flow.shape[-2] >= full_h and final_flow.shape[-1] >= full_w:
            final_flow = final_flow[..., :full_h, :full_w]
        if final_flow.shape[-2:] != (full_h, full_w):
            final_flow = resize_flow(final_flow, (full_h, full_w))
        flow_predictions[-1] = final_flow

        actual_global_init = q_initial
        forward_match = global_match["forward"]
        if not isinstance(forward_match, dict):
            raise RuntimeError("Global decoder omitted forward diagnostics")
        global_confidence = global_match["selected_confidence"]
        reverse_flow = global_match["reverse_flow"]
        if not isinstance(global_confidence, torch.Tensor) or not isinstance(
            reverse_flow,
            torch.Tensor,
        ):
            raise RuntimeError("Global decoder diagnostics are invalid")
        reverse_full = resize_flow(
            reverse_flow,
            (full_h, full_w),
        )
        # Supervise the raw correspondence candidate, never the confidence-
        # gated/rejected solver state.  This keeps candidate accuracy and
        # solver admission as separate learning problems.
        matcher_supervision_flow = matcher_candidate
        interaction_blends: Dict[int, torch.Tensor] = {}
        if isinstance(
            self.feature_encoder,
            HQSCoreDeepMatchingPyramid,
        ):
            interaction_blends = (
                self.feature_encoder.interaction_blends()
            )
        return {
            "flow_preds": flow_predictions,
            "flow_preds_raw": raw_flow_predictions,
            "flow_low": flow_lows,
            "flow_low_raw": list(flow_lows),
            "data_flow_low": data_flow_lows,
            # Legacy diagnostics use aux_low for the other split variable.
            "aux_low": data_flow_lows,
            "delta_low": delta_lows,
            "coupling_residual_low": coupling_lows,
            # Evaluation aliases: data proposal and total proximal correction.
            "delta_match_low": data_delta_lows,
            "delta_prior_low": [-residual for residual in coupling_lows],
            "hidden_states": hidden_states,
            "data_reliability_low": reliability_lows[-1],
            "data_reliability_lows": reliability_lows,
            "core_reliability_lows": reliability_lows,
            "data_valid_lows": validity_lows,
            "core_validity_lows": validity_lows,
            "data_weight_lows": validity_lows,
            "analytic_delta_lows": analytic_delta_lows,
            "learned_data_delta_lows": learned_delta_lows,
            "data_delta_lows": data_delta_lows,
            "proximal_anchor_lows": proximal_anchor_lows,
            "beta_values": beta_values,
            "lambda_values": lambda_values,
            "prediction_scales": prediction_scales,
            "final_upsample_mask_logits": final_mask_logits,
            "flow_final_raw": raw_flow_predictions[-1],
            "flow_final_refined": final_flow,
            "global_init_flow_xy": actual_global_init,
            "global_init_candidate_flow_xy": matcher_candidate,
            "global_init_confidence": global_confidence,
            "global_init_entropy": forward_match["entropy"],
            "global_init_margin": forward_match["margin"],
            "global_init_acceptance": global_match["hard_acceptance"],
            "global_init_soft_acceptance": global_match[
                "soft_acceptance"
            ],
            "global_init_selected_index": global_match[
                "selected_index"
            ],
            "global_init_cycle_support": global_match[
                "selected_cycle_support"
            ],
            "global_init_cycle_error": global_match[
                "selected_cycle_error"
            ],
            "global_topk_flow_xy": forward_match["hypotheses"],
            "global_topk_map_flow_xy": forward_match[
                "map_hypotheses"
            ],
            "global_topk_probabilities": forward_match[
                "probabilities"
            ],
            "global_topk_cycle_support": global_match[
                "cycle_support"
            ],
            "global_topk_cycle_error": global_match["cycle_error"],
            "global_reverse_flow_xy": reverse_flow,
            "global_reverse_confidence": global_match[
                "reverse_confidence"
            ],
            "matching_transformer_blends": interaction_blends,
            "matching_pyramid": self.matching_pyramid,
            "global_decoder": self.global_decoder,
            "global_match_scale": self.global_match_scale,
            "global_solver_init_scale": 16,
            # The matcher candidate, rather than a rejected zero solver state,
            # receives direct global correspondence supervision.
            "gmflow_init_flow_yx": self._to_yx(
                matcher_supervision_flow
            ),
            "gmflow_init_conf": global_confidence,
            "gmflow_init_entropy": forward_match["entropy"],
            "gmflow_init_margin": forward_match["margin"],
            "gmflow_reverse_flow_yx": (
                self._to_yx(reverse_full)
                if self.global_decoder == "multimodal_cycle"
                else None
            ),
            "occupancy_masks": [],
            "reliability_states": [],
            "reliability_geometries": [],
            "reliability_residuals": [],
            "factorised_data_gates": [],
            "band_split": None,
        }

    def param_count(self) -> Dict[str, int]:
        """Repository-compatible parameter accounting."""

        def count(module: nn.Module) -> int:
            return sum(parameter.numel() for parameter in module.parameters())

        context_count = count(self.feature_encoder.context_projections)
        feature_count = count(self.feature_encoder) - context_count
        stage_count = (
            count(self.coarse_cell)
            + count(self.fine_cell)
            + count(self.validity_head)
            + count(self.correlation_adapters)
        )
        return {
            "feature_encoder": feature_count,
            "context_encoder": context_count,
            "correlation_adapters": count(self.correlation_adapters),
            "coarse_cell": count(self.coarse_cell),
            "fine_cell": count(self.fine_cell),
            "validity_head": count(self.validity_head),
            "final_upsampler": count(self.final_upsampler),
            "stages": stage_count,
            "total": count(self),
            "total_trainable": sum(
                parameter.numel()
                for parameter in self.parameters()
                if parameter.requires_grad
            ),
        }


__all__ = ["HQSCore"]
