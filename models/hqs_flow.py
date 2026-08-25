"""Compatibility shim for the repository's primary HQS model.

This module provides both the original multi-stage HQSFlowModel and the
TensorFlow faithful port (HQSFlowModelTFPort) as the primary option.

To use the TF port (recommended), set cfg.model_type = "tfport" in configs.
Otherwise defaults to the original multi-stage model.
"""
from __future__ import annotations

from hqs_pytorch.customML.customModels.HQSFlowModel import HQSFlowModel
from hqs_pytorch.customML.customModels.HQSFlowModelTFPort import HQSFlowModelTFPort


HQSFlow = HQSFlowModel


def _cfg_get(cfg, key: str, default):
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    try:
        return cfg.get(key, default)
    except Exception:
        return getattr(cfg, key, default)


def build_model(cfg, model_type: str = "tfport"):
    """
    Build an HQS optical flow model.
    
    Args:
        cfg: Configuration object with cfg.model sub-config
        model_type: "tfport" (default, faithful TensorFlow recreation),
                    "original" (multi-stage unrolled), ``hqs_core``,
                    ``hqs_core_recurrent_control`` (equal-capacity generic
                    recurrent control for the OF-A ablation), ``hqs_lm_of``
                    (learned probabilistic correspondence and
                    transformer attention fused inside LM), ``hqs_field_of``
                    (multi-hypothesis correlation data term with a
                    source-conditioned graph-field proximal), ``hqs_otof``
                    (FlowIt-style 1/4 OT initialisation followed by HQS),
                    ``hqs_field_of_v2`` (repeated transport-conditioned HQS),
                    or ``hqs_lm_sf`` (calibrated RGB-D scene-flow prototype).
    
    Returns:
        Instantiated HQS model
    """
    # Allow override from either top-level config or model section.
    model_type = _cfg_get(cfg, "model_type", model_type)
    model_type = _cfg_get(_cfg_get(cfg, "model", None), "model_type", model_type)
    model_type = _cfg_get(_cfg_get(cfg, "model", None), "implementation", model_type)
    
    if model_type in {"tfport", "tf_faithful", "tf_port"}:
        return HQSFlowModelTFPort(cfg.model)
    if model_type in {"original", "legacy"}:
        return HQSFlowModel(cfg.model)
    if model_type in {"hqs_core", "hqscore", "core"}:
        from hqs_pytorch.customML.customModels.HQSCore import HQSCore

        return HQSCore(cfg.model)
    if model_type in {
        "hqs_core_recurrent_control",
        "hqs_recurrent_control",
        "recurrent_control",
    }:
        from hqs_pytorch.customML.customModels.HQSCoreRecurrentControl import (
            HQSCoreRecurrentControl,
        )

        return HQSCoreRecurrentControl(cfg.model)
    if model_type in {"hqs_lm_of", "hqslm_of", "hqs-lm-of"}:
        from hqs_pytorch.customML.customModels.HQSLMOpticalFlow import (
            HQSLMOpticalFlow,
        )

        return HQSLMOpticalFlow(cfg.model)
    if model_type in {
        "hqs_field_of",
        "hqsfield_of",
        "hqs-field-of",
        "hqs_field",
    }:
        from hqs_pytorch.customML.customModels.HQSFieldOpticalFlow import (
            HQSFieldOpticalFlow,
        )

        return HQSFieldOpticalFlow(cfg.model)
    if model_type in {"hqs_otof", "hqs-otof", "otof"}:
        from hqs_pytorch.customML.customModels.HQSOTOpticalFlow import (
            HQSOTOpticalFlow,
        )

        return HQSOTOpticalFlow(cfg.model)
    if model_type in {
        "hqs_field_of_v2",
        "hqs-field-of-v2",
        "hqsfield_of_v2",
        "hqs_field_v2",
    }:
        from hqs_pytorch.customML.customModels.HQSFieldOpticalFlowV2 import (
            HQSFieldOpticalFlowV2,
        )

        return HQSFieldOpticalFlowV2(cfg.model)
    if model_type in {"hqs_lm_sf", "hqslm_sf", "hqs-lm-sf"}:
        from hqs_pytorch.customML.customModels.HQSLMSceneFlow import (
            HQSLMSceneFlow,
        )

        return HQSLMSceneFlow(cfg.model)

    raise ValueError(
        f"Unknown model_type: {model_type}. "
        "Use one of: tfport | tf_faithful | tf_port | original | legacy | "
        "hqs_core | hqscore | core | hqs_core_recurrent_control | "
        "hqs_lm_of | hqs_field_of | "
        "hqs_otof | hqs_field_of_v2 | hqs_lm_sf."
    )
