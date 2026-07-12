"""Factorised reliability state and prediction head for HQSFlowModelTFPort."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor


def _group_count(channels: int, maximum: int = 8) -> int:
    groups = min(maximum, channels)
    while channels % groups != 0 and groups > 1:
        groups -= 1
    return groups


class ConvGNAct(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(_group_count(out_channels), out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)


@dataclass
class ReliabilityState:
    """Unthresholded reliability predictions at one HQS iteration."""

    visibility_logits: Tensor
    match_logits: Tensor
    log_sigma_data: Tensor
    boundary_logits: Tensor
    hidden: Optional[Tensor] = None

    @property
    def p_visible(self) -> Tensor:
        return torch.sigmoid(self.visibility_logits)

    @property
    def p_match(self) -> Tensor:
        return torch.sigmoid(self.match_logits)

    @property
    def sigma_data(self) -> Tensor:
        return self.log_sigma_data.clamp(-7.0, 5.0).exp()

    @property
    def p_boundary(self) -> Tensor:
        return torch.sigmoid(self.boundary_logits)

    def data_gate(self, in_bounds: Tensor, minimum: float = 0.0) -> Tensor:
        gate = in_bounds * self.p_visible * self.p_match
        if minimum > 0:
            gate = minimum + (1.0 - minimum) * gate
        return gate

    def detached_dict(self) -> dict[str, Tensor]:
        return {
            "p_visible": self.p_visible.detach(),
            "p_match": self.p_match.detach(),
            "sigma_data": self.sigma_data.detach(),
            "p_boundary": self.p_boundary.detach(),
            "visibility_logits": self.visibility_logits.detach(),
            "match_logits": self.match_logits.detach(),
            "log_sigma_data": self.log_sigma_data.detach(),
            "boundary_logits": self.boundary_logits.detach(),
        }


class FactorisedReliabilityHead(nn.Module):
    """Predict visibility, matchability, data scale, and motion boundaries.

    Default input channels correspond to:
      |photometric residual| : 1
      image gradient         : 1
      in-bounds validity     : 1
      legacy confidence      : 1
      ||w-q||                : 1
      ||w||                  : 1
      FB confidence          : 1
      hole score             : 1
      collision score        : 1
      correlation entropy    : 1
      correlation margin     : 1
    Total: 11 channels.

    The head starts conservative but non-collapsed:
      p_visible ~= 0.88, p_match ~= 0.88, sigma ~= 1, p_boundary ~= 0.12.
    """

    def __init__(
        self,
        in_channels: int = 11,
        hidden_channels: int = 48,
        recurrent_channels: int = 0,
    ) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        self.recurrent_channels = int(recurrent_channels)
        total_in = self.in_channels + self.recurrent_channels

        self.trunk = nn.Sequential(
            ConvGNAct(total_in, hidden_channels),
            ConvGNAct(hidden_channels, hidden_channels),
            ConvGNAct(hidden_channels, hidden_channels),
        )
        self.visibility_head = nn.Conv2d(hidden_channels, 1, 3, padding=1)
        self.match_head = nn.Conv2d(hidden_channels, 1, 3, padding=1)
        self.scale_head = nn.Conv2d(hidden_channels, 1, 3, padding=1)
        self.boundary_head = nn.Conv2d(hidden_channels, 1, 3, padding=1)
        self.hidden_head = (
            nn.Conv2d(hidden_channels, recurrent_channels, 3, padding=1)
            if recurrent_channels > 0
            else None
        )
        self._initialise()

    def _initialise(self) -> None:
        for head in (
            self.visibility_head,
            self.match_head,
            self.scale_head,
            self.boundary_head,
        ):
            nn.init.zeros_(head.weight)
        nn.init.constant_(self.visibility_head.bias, 2.0)
        nn.init.constant_(self.match_head.bias, 2.0)
        nn.init.zeros_(self.scale_head.bias)
        nn.init.constant_(self.boundary_head.bias, -2.0)
        if self.hidden_head is not None:
            nn.init.zeros_(self.hidden_head.weight)
            nn.init.zeros_(self.hidden_head.bias)

    def forward(
        self,
        features: Tensor,
        previous_hidden: Optional[Tensor] = None,
    ) -> ReliabilityState:
        if features.ndim != 4 or features.shape[1] != self.in_channels:
            raise ValueError(
                f"features must be [B,{self.in_channels},H,W], "
                f"got {tuple(features.shape)}"
            )
        if self.recurrent_channels:
            if previous_hidden is None:
                previous_hidden = features.new_zeros(
                    features.shape[0],
                    self.recurrent_channels,
                    features.shape[2],
                    features.shape[3],
                )
            x = torch.cat((features, previous_hidden), dim=1)
        else:
            x = features
        h = self.trunk(x)
        next_hidden = torch.tanh(self.hidden_head(h)) if self.hidden_head is not None else None
        return ReliabilityState(
            visibility_logits=self.visibility_head(h),
            match_logits=self.match_head(h),
            log_sigma_data=self.scale_head(h),
            boundary_logits=self.boundary_head(h),
            hidden=next_hidden,
        )


def assemble_reliability_features(
    *,
    photometric_residual: Tensor,
    image_gradient: Tensor,
    in_bounds: Tensor,
    legacy_confidence: Tensor,
    hqs_residual_yx: Tensor,
    flow_yx: Tensor,
    fb_confidence: Tensor,
    hole_score: Tensor,
    collision_score: Tensor,
    corr_entropy: Tensor,
    corr_margin: Tensor,
    detach_inputs: bool = False,
) -> Tensor:
    """Assemble the canonical 11-channel reliability tensor."""
    hqs_norm = hqs_residual_yx.square().sum(1, keepdim=True).add(1e-8).sqrt()
    flow_norm = flow_yx.square().sum(1, keepdim=True).add(1e-8).sqrt()
    tensors = [
        photometric_residual.abs(),
        image_gradient,
        in_bounds,
        legacy_confidence,
        hqs_norm,
        flow_norm,
        fb_confidence,
        hole_score,
        collision_score,
        corr_entropy,
        corr_margin,
    ]
    spatial = tensors[0].shape[-2:]
    for tensor in tensors:
        if tensor.ndim != 4 or tensor.shape[1] != 1 or tensor.shape[-2:] != spatial:
            raise ValueError(
                "Every scalar reliability feature must have shape [B,1,H,W]."
            )
    features = torch.cat(tensors, dim=1)
    return features.detach() if detach_inputs else features
