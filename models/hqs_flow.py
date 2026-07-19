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
                    "original" (multi-stage unrolled), or ``hqs_core``
                    (compact four-scale operator-structured model).
    
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

    raise ValueError(
        f"Unknown model_type: {model_type}. "
        "Use one of: tfport | tf_faithful | tf_port | original | legacy | "
        "hqs_core | hqscore | core."
    )
