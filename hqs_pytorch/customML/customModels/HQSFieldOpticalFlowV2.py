"""HQS-Field-OFv2: transport-conditioned probabilistic field inference.

Unlike ``HQS-OTOF``, optimal transport is not used only to seed the flow.
The coarse solver treats the transport plan as a latent measurement variable:

``global transport -> analytic data solve -> source semantic field proximal``.

The first transport step is independent of the current flow.  At later 1/8
iterations a gradually strengthened motion-consistency term reconditions the
global plan, preserving a route for re-acquisition without allowing the
initial state to dominate immediately.  Fine 1/4 and 1/2 stages retain the
local multi-hypothesis analytic data cells and source-only graph proximal from
HQS-Field-OF.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.hqs_lm_components import LMState
from models.hqs_ot_components import (
    BlockwiseDustbinSinkhorn,
    TransportFieldCell,
    TransportMeasurement,
    resize_transport_measurement,
)
from models.warp import resize_flow

from .HQSFieldOpticalFlow import _cfg_get, _list
from .HQSOTOpticalFlow import HQSOTOpticalFlow


class HQSFieldOpticalFlowV2(HQSOTOpticalFlow):
    """Four-scale HQS solver with 1/8 initial and repeated 1/8 transport."""

    required_transport_scale = 8
    global_matcher_parameter_prefixes = (
        *HQSOTOpticalFlow.global_matcher_parameter_prefixes,
        "retransport_matcher.",
    )

    def __init__(self, cfg) -> None:
        model_cfg = _cfg_get(cfg, "hqs_field_of_v2", cfg)
        super().__init__({"hqs_otof": model_cfg})
        self.cfg = model_cfg
        self.solver_name = "hqs_field_of_v2"
        self.retransport_scale = int(
            _cfg_get(model_cfg, "retransport_scale", 8)
        )
        if self.retransport_scale != 8:
            raise ValueError(
                "HQS-Field-OFv2 requires retransport_scale=8 so the initial "
                "1/4 plan is preserved while repeated OT remains tractable"
            )
        self.retransport_matcher = BlockwiseDustbinSinkhorn(
            temperature=float(
                _cfg_get(
                    model_cfg,
                    "retransport_temperature",
                    _cfg_get(model_cfg, "ot_temperature", 0.07),
                )
            ),
            sinkhorn_iterations=int(
                _cfg_get(model_cfg, "retransport_sinkhorn_iterations", 6)
            ),
            query_chunk_size=int(
                _cfg_get(model_cfg, "retransport_query_chunk_size", 256)
            ),
            local_expectation_radius=int(
                _cfg_get(
                    model_cfg,
                    "retransport_local_expectation_radius",
                    2,
                )
            ),
            num_hypotheses=int(
                _cfg_get(model_cfg, "retransport_num_hypotheses", 8)
            ),
            initial_dustbin_score=float(
                _cfg_get(model_cfg, "ot_initial_dustbin_score", 1.0)
            ),
            state_sigma=float(
                _cfg_get(model_cfg, "retransport_state_sigma", 3.0)
            ),
            state_cost_clip=float(
                _cfg_get(model_cfg, "retransport_state_cost_clip", 50.0)
            ),
            covariance_floor=float(
                _cfg_get(model_cfg, "ot_covariance_floor", 0.05)
            ),
            precision_maximum=float(
                _cfg_get(model_cfg, "ot_precision_maximum", 4.0)
            ),
            maximum_tokens=int(
                _cfg_get(model_cfg, "retransport_maximum_tokens", 5000)
            ),
            gradient_checkpointing=bool(
                _cfg_get(model_cfg, "ot_gradient_checkpointing", True)
            ),
            cache_all_pairs_scores=bool(
                _cfg_get(
                    model_cfg,
                    "retransport_cache_all_pairs_scores",
                    True,
                )
            ),
            half_precision_score_cache=bool(
                _cfg_get(
                    model_cfg,
                    "retransport_half_precision_score_cache",
                    True,
                )
            ),
        )
        # Both transport problems share the learned unmatched-state score.
        self.retransport_matcher.dustbin_score = (
            self.transport_matcher.dustbin_score
        )

        context_channels = _list(
            model_cfg, "context_channels", (64, 64, 96, 96), int
        )
        context_by_scale = dict(zip((2, 4, 8, 16), context_channels))
        # The coarse top-K field cells are replaced, not retained as unused
        # capacity. Fine local cells remain identical to HQS-Field-OF.
        del self.cells["16"]
        del self.cells["8"]

        graph_neighbours_cfg = _cfg_get(
            model_cfg, "semantic_graph_neighbours", [8, 8]
        )
        graph_max_tokens_cfg = _cfg_get(
            model_cfg, "semantic_graph_maximum_tokens", [2048, 4096]
        )
        if len(graph_neighbours_cfg) != 2:
            raise ValueError(
                "semantic_graph_neighbours must contain [1/16,1/8]"
            )
        if len(graph_max_tokens_cfg) != 2:
            raise ValueError(
                "semantic_graph_maximum_tokens must contain [1/16,1/8]"
            )
        neighbours = {
            scale: int(value)
            for scale, value in zip((16, 8), graph_neighbours_cfg)
        }
        maximum_tokens = {
            scale: int(value)
            for scale, value in zip((16, 8), graph_max_tokens_cfg)
        }
        transport_common = dict(
            groups=int(_cfg_get(model_cfg, "groups", 8)),
            calibrator_hidden_channels=int(
                _cfg_get(
                    model_cfg,
                    "transport_calibrator_hidden_channels",
                    64,
                )
            ),
            graph_embedding_channels=int(
                _cfg_get(
                    model_cfg, "semantic_graph_embedding_channels", 32
                )
            ),
            graph_feature_temperature=float(
                _cfg_get(
                    model_cfg, "semantic_graph_feature_temperature", 0.10
                )
            ),
            graph_anchor_floor=float(
                _cfg_get(model_cfg, "graph_anchor_floor", 0.05)
            ),
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
            maximum_log_precision_adjustment=float(
                _cfg_get(
                    model_cfg,
                    "maximum_transport_log_precision_adjustment",
                    2.0,
                )
            ),
            maximum_observability_logit_adjustment=float(
                _cfg_get(
                    model_cfg,
                    "maximum_transport_observability_logit_adjustment",
                    2.0,
                )
            ),
        )
        self.transport_cells = nn.ModuleDict(
            {
                str(scale): TransportFieldCell(
                    context_channels=context_by_scale[scale],
                    max_iterations=self.iterations_by_scale[scale],
                    graph_neighbours=neighbours[scale],
                    graph_maximum_tokens=maximum_tokens[scale],
                    **transport_common,
                )
                for scale in (16, 8)
            }
        )
        weights = tuple(
            float(value)
            for value in _cfg_get(
                model_cfg,
                "retransport_state_weights",
                [0.0, 0.20, 0.45, 0.70],
            )
        )
        expected = self.iterations_by_scale[8]
        if len(weights) != expected:
            raise ValueError(
                "retransport_state_weights must match the 1/8 iteration "
                f"count ({expected}), got {weights}"
            )
        if weights[0] != 0.0 or any(value < 0 for value in weights):
            raise ValueError(
                "The first retransport weight must be zero and all weights "
                "must be non-negative"
            )
        self.retransport_state_weights = weights

    def _transport_from_features(
        self,
        source_match: Dict[int, torch.Tensor],
        target_match: Dict[int, torch.Tensor],
        *,
        state_flow: Optional[torch.Tensor] = None,
        state_weight: float = 0.0,
    ) -> tuple[
        TransportMeasurement,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        source_transport, source_weights = self.transport_fusion(
            source_match, target_scale=self.transport_scale
        )
        target_transport, target_weights = self.transport_fusion(
            target_match, target_scale=self.transport_scale
        )
        if state_flow is not None:
            state_flow = resize_flow(
                state_flow, source_transport.shape[-2:]
            )
        measurement = self.transport_matcher(
            source_transport,
            target_transport,
            state_flow=state_flow,
            state_weight=state_weight,
        )
        return (
            measurement,
            source_transport,
            target_transport,
            source_weights,
            target_weights,
        )

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
                "HQS-Field-OFv2 expects equal [B,3,H,W] inputs"
            )
        if image1.shape[1] != 3:
            raise ValueError("HQS-Field-OFv2 expects RGB inputs")
        if iters is not None and int(iters) != self.num_hqs_iterations:
            raise ValueError(
                "Set hqs_field_of_v2.iterations rather than "
                "forward(iters=...)"
            )
        if flow_init is not None:
            raise ValueError(
                "HQS-Field-OFv2 initialises from its latent transport plan"
            )
        if (source_valid is not None or target_valid is not None) and not (
            self.allow_external_validity_inputs
        ):
            raise ValueError(
                "Ground-truth visibility is loss-only for HQS-Field-OFv2"
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
                source_match_raw, target_match_raw
            )
        )
        (
            initial_transport,
            _source_initial_transport,
            _target_initial_transport,
            source_fusion_weights,
            target_fusion_weights,
        ) = self._transport_from_features(
            source_match_raw,
            target_match_raw,
            state_flow=None,
            state_weight=0.0,
        )
        # The same hierarchical fusion weights are reused at 1/8, but all
        # pairwise scores and transport duals are recomputed on that grid.
        source_retransport, source_retransport_weights = (
            self.transport_fusion(
                source_match_raw,
                target_scale=self.retransport_scale,
            )
        )
        target_retransport, target_retransport_weights = (
            self.transport_fusion(
                target_match_raw,
                target_scale=self.retransport_scale,
            )
        )
        source_match = {
            scale: F.normalize(value.float(), dim=1, eps=1e-6)
            for scale, value in source_match_raw.items()
        }
        target_match = {
            scale: F.normalize(value.float(), dim=1, eps=1e-6)
            for scale, value in target_match_raw.items()
        }
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

        initial = resize_flow(
            initial_transport.flow, source_match[16].shape[-2:]
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
        transport_history: List[TransportMeasurement] = []
        transport_state_weights: List[float] = []

        latest_transport = initial_transport
        for scale_index, scale in enumerate(self.scale_order):
            target_size = source_match[scale].shape[-2:]
            if scale_index > 0:
                state = LMState(
                    w=resize_flow(state.w, target_size),
                    z=resize_flow(state.z, target_size),
                )

            if scale in (16, 8):
                cell = self.transport_cells[str(scale)]
                graph_embedding = cell.prepare_source_graph(
                    source_context[scale]
                )
                for iteration in range(self.iterations_by_scale[scale]):
                    previous_z = state.z
                    state_weight = 0.0
                    if scale == 8:
                        state_weight = self.retransport_state_weights[
                            iteration
                        ]
                        latest_transport = self.retransport_matcher(
                            source_retransport,
                            target_retransport,
                            state_flow=(
                                None
                                if state_weight == 0.0
                                else resize_flow(
                                    state.z,
                                    source_retransport.shape[-2:],
                                )
                            ),
                            state_weight=state_weight,
                        )
                    measurement = resize_transport_measurement(
                        latest_transport, target_size
                    )
                    update = cell(
                        measurement=measurement,
                        source_context=source_context[scale],
                        source_graph_embedding=graph_embedding,
                        state=state,
                        iteration=iteration,
                        graph_sweeps=self.graph_sweeps[scale],
                        max_data_delta=self.max_data_delta[scale],
                    )
                    state = update.state
                    probabilities = measurement.topk_probabilities
                    responsibilities = probabilities / probabilities.sum(
                        dim=1, keepdim=True
                    ).clamp_min(1e-8)
                    modes = measurement.topk_flows.shape[1]
                    mode_precision = update.precision.unsqueeze(1).expand(
                        -1, modes, -1, -1, -1
                    )

                    flow_lows.append(state.z)
                    data_flow_lows.append(state.w)
                    delta_lows.append(state.z - previous_z)
                    coupling_lows.append(state.w - state.z)
                    data_delta_lows.append(update.data_delta)
                    posterior_proposals.append(update.proposal)
                    collapsed_precisions.append(update.precision)
                    hypothesis_proposals.append(measurement.topk_flows)
                    hypothesis_logits.append(measurement.topk_logits)
                    hypothesis_probabilities.append(probabilities)
                    hypothesis_responsibilities.append(responsibilities)
                    hypothesis_precisions.append(mode_precision)
                    learned_hypothesis_deltas.append(
                        torch.zeros_like(measurement.topk_flows)
                    )
                    analytic_hypotheses.append(measurement.topk_flows)
                    matchabilities.append(
                        (
                            measurement.confidence
                            * measurement.observability
                        ).clamp(0.0, 1.0)
                    )
                    measurement_supports.append(
                        update.measurement_support
                    )
                    cycle_supports.append(measurement.observability)
                    responsibility_entropies.append(measurement.entropy)
                    inverse_traces.append(update.inverse_trace)
                    conditions.append(update.condition)
                    attention_entropies.append(measurement.entropy)
                    attention_peaks.append(measurement.peak_mass)
                    beta_values.append(update.beta)
                    damping_values.append(update.damping)
                    lambda_values.append(update.regularisation)
                    inertia_values.append(update.proximal_inertia)
                    prediction_scales.append(scale)
                    transport_history.append(measurement)
                    transport_state_weights.append(float(state_weight))
                    flow_predictions.append(
                        resize_flow(
                            state.z, (full_height, full_width)
                        )
                    )
                continue

            cell = self.cells[str(scale)]
            proposal_hidden = cell.initialise_proposal_hidden(
                source_context[scale]
            )
            graph_embedding = cell.prepare_source_graph(
                source_context[scale]
            )
            cycle_input = F.interpolate(
                latest_transport.observability,
                size=target_size,
                mode="bilinear",
                align_corners=False,
            ).clamp(0.0, 1.0)
            for iteration in range(self.iterations_by_scale[scale]):
                previous_z = state.z
                correlation = self._raw_correlation(
                    scale,
                    state.w,
                    source_match[scale],
                    target_match[scale],
                    all_pairs=None,
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
                    analytic_mixture=None,
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
        initial_full = resize_flow(
            initial_transport.flow, (full_height, full_width)
        )
        initial_margin = initial_transport.peak_mass
        if initial_transport.topk_probabilities.shape[1] > 1:
            initial_margin = (
                initial_transport.topk_probabilities[:, 0:1]
                - initial_transport.topk_probabilities[:, 1:2]
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
            "global_init_flow_xy": initial_transport.flow,
            "global_init_confidence": initial_transport.confidence,
            "global_init_entropy": initial_transport.entropy,
            "global_init_margin": initial_margin,
            "gmflow_init_flow_yx": self._to_yx(
                initial_transport.flow
            ),
            "gmflow_init_conf": initial_transport.confidence,
            "gmflow_init_entropy": initial_transport.entropy,
            "gmflow_init_margin": initial_margin,
            "ot_init_flow_xy": initial_transport.flow,
            "ot_init_flow_full_xy": initial_full,
            "ot_confidence": initial_transport.confidence,
            "ot_observability": initial_transport.observability,
            "ot_dustbin_probability": (
                initial_transport.dustbin_probability
            ),
            "ot_entropy": initial_transport.entropy,
            "ot_peak_mass": initial_transport.peak_mass,
            "ot_retained_mass": initial_transport.retained_mass,
            "ot_precision": initial_transport.precision,
            "ot_hypothesis_flows": initial_transport.topk_flows,
            "ot_hypothesis_logits": initial_transport.topk_logits,
            "ot_hypothesis_probabilities": (
                initial_transport.topk_probabilities
            ),
            "ot_retransport_flows": [
                value.flow for value in transport_history
            ],
            "ot_retransport_confidences": [
                value.confidence for value in transport_history
            ],
            "ot_retransport_observabilities": [
                value.observability for value in transport_history
            ],
            "ot_retransport_entropies": [
                value.entropy for value in transport_history
            ],
            "ot_retransport_dustbin_probabilities": [
                value.dustbin_probability for value in transport_history
            ],
            "ot_retransport_state_weights": transport_state_weights,
            "ot_source_fusion_weights": source_fusion_weights,
            "ot_target_fusion_weights": target_fusion_weights,
            "ot_retransport_source_fusion_weights": (
                source_retransport_weights
            ),
            "ot_retransport_target_fusion_weights": (
                target_retransport_weights
            ),
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
        result = super().param_count()
        result["retransport_matcher"] = sum(
            parameter.numel()
            for parameter in self.retransport_matcher.parameters()
        )
        result["transport_cells"] = sum(
            parameter.numel()
            for parameter in self.transport_cells.parameters()
        )
        result["total"] = sum(
            parameter.numel() for parameter in self.parameters()
        )
        result["total_trainable"] = sum(
            parameter.numel()
            for parameter in self.parameters()
            if parameter.requires_grad
        )
        return result


__all__ = ["HQSFieldOpticalFlowV2"]
