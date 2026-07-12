"""Drop-in reliability-aware correction and proximal blocks.

These blocks preserve causal separation:
* DataCorrectionNet may consume matching/target-derived features.
* OcclusionPriorNet accepts source/context and predicted state only.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from .factorised_reliability import ReliabilityState


def _gn(channels: int) -> nn.GroupNorm:
    groups = min(8, channels)
    while channels % groups and groups > 1:
        groups -= 1
    return nn.GroupNorm(groups, channels)


class ResidualConvBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            _gn(channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            _gn(channels),
        )
        self.activation = nn.SiLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        return self.activation(x + self.net(x))


class DataCorrectionNet(nn.Module):
    """Correlation/data-driven residual correction."""

    def __init__(
        self,
        corr_channels: int,
        context_channels: int,
        hidden_channels: int = 128,
        max_delta: float = 4.0,
    ) -> None:
        super().__init__()
        self.max_delta = float(max_delta)
        in_channels = corr_channels + context_channels + 2 + 2 + 4
        self.input = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 3, padding=1, bias=False),
            _gn(hidden_channels),
            nn.SiLU(inplace=True),
        )
        self.body = nn.Sequential(
            ResidualConvBlock(hidden_channels),
            ResidualConvBlock(hidden_channels),
        )
        self.output = nn.Conv2d(hidden_channels, 2, 3, padding=1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(
        self,
        corr_features: Tensor,
        source_context: Tensor,
        flow_yx: Tensor,
        hqs_residual_yx: Tensor,
        reliability: ReliabilityState,
    ) -> Tensor:
        rel = torch.cat(
            (
                reliability.p_visible,
                reliability.p_match,
                reliability.sigma_data.log(),
                reliability.p_boundary,
            ),
            dim=1,
        )
        x = torch.cat(
            (corr_features, source_context, flow_yx, hqs_residual_yx, rel), dim=1
        )
        return self.max_delta * torch.tanh(self.output(self.body(self.input(x))))


class OcclusionPriorNet(nn.Module):
    """Source-conditioned correction with no target/correlation input."""

    def __init__(
        self,
        context_channels: int,
        hidden_channels: int = 128,
        max_delta: float = 4.0,
    ) -> None:
        super().__init__()
        self.max_delta = float(max_delta)
        # flow 2 + aux 2 + residual 2 + source context + rel 4 + geometry 3
        in_channels = 6 + context_channels + 4 + 3
        self.input = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 3, padding=1, bias=False),
            _gn(hidden_channels),
            nn.SiLU(inplace=True),
        )
        self.body = nn.Sequential(
            ResidualConvBlock(hidden_channels),
            ResidualConvBlock(hidden_channels),
        )
        self.output = nn.Conv2d(hidden_channels, 2, 3, padding=1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(
        self,
        source_context: Tensor,
        flow_yx: Tensor,
        aux_yx: Tensor,
        reliability: ReliabilityState,
        *,
        fb_confidence: Tensor,
        hole_score: Tensor,
        collision_score: Tensor,
    ) -> Tensor:
        residual = flow_yx - aux_yx
        rel = torch.cat(
            (
                reliability.p_visible,
                reliability.p_match,
                reliability.sigma_data.log(),
                reliability.p_boundary,
            ),
            dim=1,
        )
        geometry = torch.cat((fb_confidence, hole_score, collision_score), dim=1)
        x = torch.cat(
            (flow_yx, aux_yx, residual, source_context, rel, geometry), dim=1
        )
        return self.max_delta * torch.tanh(self.output(self.body(self.input(x))))


class ReliabilityAwareFusion(nn.Module):
    """Continuous blend of data and source-prior corrections."""

    def __init__(self, minimum_data_gate: float = 0.0) -> None:
        super().__init__()
        if not 0.0 <= minimum_data_gate < 1.0:
            raise ValueError("minimum_data_gate must be in [0,1).")
        self.minimum_data_gate = float(minimum_data_gate)

    def forward(
        self,
        flow_yx: Tensor,
        delta_data_yx: Tensor,
        delta_prior_yx: Tensor,
        reliability: ReliabilityState,
        in_bounds: Tensor,
    ) -> tuple[Tensor, Tensor]:
        gate = reliability.data_gate(in_bounds, self.minimum_data_gate)
        new_flow = flow_yx + gate * delta_data_yx + (1.0 - gate) * delta_prior_yx
        return new_flow, gate


class BoundaryAwareProximalNet(nn.Module):
    """Bounded learned proximal operator.

    This block accepts only source context and predicted states. Boundary
    probability suppresses the coupling correction near discontinuities.
    """

    def __init__(
        self,
        context_channels: int,
        hidden_channels: int = 96,
        max_delta: float = 1.0,
    ) -> None:
        super().__init__()
        self.max_delta = float(max_delta)
        # flow 2 + aux 2 + residual 2 + context + reliability 4
        in_channels = 6 + context_channels + 4
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 3, padding=1, bias=False),
            _gn(hidden_channels),
            nn.SiLU(inplace=True),
            ResidualConvBlock(hidden_channels),
            ResidualConvBlock(hidden_channels),
        )
        self.delta_head = nn.Conv2d(hidden_channels, 2, 3, padding=1)
        self.mix_head = nn.Conv2d(hidden_channels, 1, 3, padding=1)
        nn.init.zeros_(self.delta_head.weight)
        nn.init.zeros_(self.delta_head.bias)
        nn.init.zeros_(self.mix_head.weight)
        nn.init.zeros_(self.mix_head.bias)

    def forward(
        self,
        flow_yx: Tensor,
        previous_aux_yx: Tensor,
        source_context: Tensor,
        reliability: ReliabilityState,
    ) -> tuple[Tensor, Tensor]:
        residual = flow_yx - previous_aux_yx
        rel = torch.cat(
            (
                reliability.p_visible,
                reliability.p_match,
                reliability.sigma_data.log(),
                reliability.p_boundary,
            ),
            dim=1,
        )
        h = self.net(
            torch.cat((flow_yx, previous_aux_yx, residual, source_context, rel), dim=1)
        )
        delta = self.max_delta * torch.tanh(self.delta_head(h))
        # Weak coupling across predicted motion boundaries.
        mix = torch.sigmoid(self.mix_head(h)) * (1.0 - reliability.p_boundary)
        proposal = flow_yx + delta
        aux = mix * proposal + (1.0 - mix) * previous_aux_yx
        return aux, mix
