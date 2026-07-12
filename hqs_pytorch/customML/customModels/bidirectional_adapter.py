"""Shared-weight bidirectional adapter for an existing optical-flow model.

This adapter is intentionally conservative: it invokes the same wrapped model
for I1->I2 and I2->I1. It does not duplicate parameters. It is immediately
usable for geometry/loss development before encoder-feature reuse is introduced.

For final efficiency, move bidirectional execution inside HQSFlowModelTFPort so
the two image encodings are computed once.
"""
from __future__ import annotations

from typing import Any

import torch.nn as nn
from torch import Tensor


class SharedWeightBidirectionalAdapter(nn.Module):
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(
        self,
        image1: Tensor,
        image2: Tensor,
        *args: Any,
        **kwargs: Any,
    ) -> dict[str, Any]:
        forward_output = self.model(image1, image2, *args, **kwargs)
        reverse_output = self.model(image2, image1, *args, **kwargs)
        return {
            "forward": forward_output,
            "reverse": reverse_output,
        }
