"""Compatibility shim for the repository's primary HQS model.

The authoritative implementation now lives in hqs_pytorch so train.py uses the
model from that folder. This module remains as a stable import surface for the
rest of the repository.
"""
from __future__ import annotations

from hqs_pytorch.customML.customModels.HQSFlowModel import HQSFlowModel


HQSFlow = HQSFlowModel


def build_model(cfg) -> HQSFlow:
    return HQSFlow(cfg.model)
