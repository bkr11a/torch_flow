"""HQS-OTOF: FlowIt-style OT initialisation followed by HQS field inference.

The measurement front end implements the first five published FlowIt stages:

1. hierarchical transformer-enhanced feature construction;
2. all-pairs correlation on the 1/4 grid;
3. entropy-regularised Sinkhorn transport with an unmatched dustbin;
4. local expectation around the most probable match;
5. confidence and observability derived from transported mass.

The FlowIt residual GRU is deliberately not used.  Its output is instead the
initial measurement state for the existing HQS correlation-data and
source-conditioned graph-field iterations.
"""
from __future__ import annotations

from typing import Dict, Mapping, Optional

import torch

from models.hqs_ot_components import (
    BlockwiseDustbinSinkhorn,
    HierarchicalTransportFusion,
    TransportMeasurement,
    resize_transport_measurement,
    transport_to_correlation_mixture,
)
from models.warp import resize_flow

from .HQSFieldOpticalFlow import (
    HQSFieldOpticalFlow,
    _cfg_get,
    _list,
)


class HQSOTOpticalFlow(HQSFieldOpticalFlow):
    """Static 1/4 OT measurement followed by the ten-step HQS-Field solver."""

    required_transport_scale = 4
    global_matcher_parameter_prefixes = (
        "feature_encoder.stages.",
        "feature_encoder.match_projections.",
        "match_feature_transformers.",
        "match_feature_transformer_logits.",
        "transport_fusion.",
        "transport_matcher.",
    )

    def __init__(self, cfg) -> None:
        model_cfg = _cfg_get(cfg, "hqs_otof", cfg)
        # Reuse the exact HQS-Field cell, scale and upsampling implementation.
        # The wrapper prevents the base constructor from looking for a second
        # configuration key.
        super().__init__({"hqs_field_of": model_cfg})
        self.cfg = model_cfg
        self.solver_name = "hqs_otof"

        match_channels = _list(
            model_cfg, "match_channels", (64, 96, 128, 128), int
        )
        channels_by_scale = dict(zip((2, 4, 8, 16), match_channels))
        transport_scale = int(_cfg_get(model_cfg, "ot_scale", 4))
        required_scale = int(self.required_transport_scale)
        if transport_scale != required_scale:
            raise ValueError(
                f"{self.__class__.__name__} requires ot_scale="
                f"{required_scale}, got {transport_scale}"
            )
        self.transport_scale = transport_scale
        transport_channels = int(
            _cfg_get(model_cfg, "ot_feature_channels", 128)
        )
        fusion_scales = tuple(
            int(value)
            for value in _cfg_get(model_cfg, "ot_fusion_scales", [4, 8, 16])
        )
        self.transport_fusion = HierarchicalTransportFusion(
            channels_by_scale,
            scales=fusion_scales,
            output_channels=transport_channels,
            groups=int(_cfg_get(model_cfg, "groups", 8)),
        )
        self.transport_matcher = BlockwiseDustbinSinkhorn(
            temperature=float(
                _cfg_get(model_cfg, "ot_temperature", 0.07)
            ),
            sinkhorn_iterations=int(
                _cfg_get(model_cfg, "ot_sinkhorn_iterations", 8)
            ),
            query_chunk_size=int(
                _cfg_get(model_cfg, "ot_query_chunk_size", 128)
            ),
            local_expectation_radius=int(
                _cfg_get(model_cfg, "ot_local_expectation_radius", 2)
            ),
            num_hypotheses=int(
                _cfg_get(model_cfg, "ot_num_hypotheses", 4)
            ),
            initial_dustbin_score=float(
                _cfg_get(model_cfg, "ot_initial_dustbin_score", 1.0)
            ),
            state_sigma=float(
                _cfg_get(model_cfg, "ot_state_sigma", 3.0)
            ),
            state_cost_clip=float(
                _cfg_get(model_cfg, "ot_state_cost_clip", 50.0)
            ),
            covariance_floor=float(
                _cfg_get(model_cfg, "ot_covariance_floor", 0.05)
            ),
            precision_maximum=float(
                _cfg_get(model_cfg, "ot_precision_maximum", 4.0)
            ),
            maximum_tokens=int(
                _cfg_get(model_cfg, "ot_maximum_tokens", 20000)
            ),
            gradient_checkpointing=bool(
                _cfg_get(model_cfg, "ot_gradient_checkpointing", True)
            ),
            cache_all_pairs_scores=bool(
                _cfg_get(model_cfg, "ot_cache_all_pairs_scores", True)
            ),
            half_precision_score_cache=bool(
                _cfg_get(
                    model_cfg,
                    "ot_half_precision_score_cache",
                    True,
                )
            ),
        )
        self._active_transport_measurement: Optional[
            TransportMeasurement
        ] = None
        self._active_feature_cache = None

    def _global_mixture_and_cycle(
        self,
        all_pairs,
        *,
        scale: int,
        reference: torch.Tensor,
    ):
        """Use static OT modes/support for the global HQS measurement steps."""
        if self._active_transport_measurement is None:
            return super()._global_mixture_and_cycle(
                all_pairs, scale=scale, reference=reference
            )
        measurement = resize_transport_measurement(
            self._active_transport_measurement,
            reference.shape[-2:],
        )
        analytic = transport_to_correlation_mixture(
            measurement,
            size=reference.shape[-2:],
            num_hypotheses=self.num_hypotheses[scale],
        )
        cycle = (
            analytic.in_bounds
            * measurement.observability.unsqueeze(1)
        ).clamp(0.0, 1.0)
        reverse = torch.zeros_like(measurement.flow)
        return analytic, cycle, reverse

    def _transport_frontend(
        self,
        image1: torch.Tensor,
        image2: torch.Tensor,
        *,
        state_flow: Optional[torch.Tensor] = None,
        state_weight: float = 0.0,
    ) -> tuple[TransportMeasurement, Dict[str, torch.Tensor]]:
        (
            source_backbone,
            target_backbone,
            source_match,
            target_match,
        ) = self._encode_pair_features(
            image1.float(),
            image2.float(),
        )
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
        self._active_feature_cache = (
            source_backbone,
            target_backbone,
            source_match,
            target_match,
        )
        return measurement, {
            "source_fusion_weights": source_weights,
            "target_fusion_weights": target_weights,
        }

    def forward(
        self,
        image1: torch.Tensor,
        image2: torch.Tensor,
        iters: Optional[int] = None,
        flow_init: Optional[torch.Tensor] = None,
        source_valid: Optional[torch.Tensor] = None,
        target_valid: Optional[torch.Tensor] = None,
    ) -> Dict[str, object]:
        if flow_init is not None:
            raise ValueError(
                "HQS-OTOF defines its initial state through 1/4 optimal "
                "transport; external flow_init would invalidate that model."
            )
        measurement, fusion = self._transport_frontend(image1, image2)
        full_initial = resize_flow(
            measurement.flow, image1.shape[-2:]
        )
        self._active_transport_measurement = measurement
        try:
            output = super().forward(
                image1,
                image2,
                iters=iters,
                flow_init=full_initial,
                source_valid=source_valid,
                target_valid=target_valid,
            )
        finally:
            self._active_transport_measurement = None
            self._active_feature_cache = None
        output.update(
            {
                "ot_init_flow_xy": measurement.flow,
                "ot_init_flow_full_xy": full_initial,
                "ot_confidence": measurement.confidence,
                "ot_observability": measurement.observability,
                "ot_dustbin_probability": (
                    measurement.dustbin_probability
                ),
                "ot_entropy": measurement.entropy,
                "ot_peak_mass": measurement.peak_mass,
                "ot_retained_mass": measurement.retained_mass,
                "ot_precision": measurement.precision,
                "ot_hypothesis_flows": measurement.topk_flows,
                "ot_hypothesis_logits": measurement.topk_logits,
                "ot_hypothesis_probabilities": (
                    measurement.topk_probabilities
                ),
                "ot_source_fusion_weights": (
                    fusion["source_fusion_weights"]
                ),
                "ot_target_fusion_weights": (
                    fusion["target_fusion_weights"]
                ),
                # Reuse the existing initialisation loss and reporting keys.
                "global_init_flow_xy": measurement.flow,
                "global_init_confidence": measurement.confidence,
                "global_init_entropy": measurement.entropy,
                "gmflow_init_flow_yx": self._to_yx(measurement.flow),
                "gmflow_init_conf": measurement.confidence,
                "gmflow_init_entropy": measurement.entropy,
                "solver": self.solver_name,
            }
        )
        return output

    def param_count(self) -> Dict[str, int]:
        result = super().param_count()
        result["ot_frontend"] = sum(
            parameter.numel()
            for module in (self.transport_fusion, self.transport_matcher)
            for parameter in module.parameters()
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


__all__ = ["HQSOTOpticalFlow"]
