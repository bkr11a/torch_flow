"""Feature and context encoders for HQSFlow.

Two encoder variants are provided:
  BasicEncoder  – ResNet-style, used for full-resolution models.
  SmallEncoder  – Lightweight variant for ablation / fast experiments.

Both expose a ``feature_pyramid`` method that returns a list of feature maps
at successively halved spatial resolutions (strides 2, 4, 8).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional


# ---------------------------------------------------------------------------
# Normalisation helpers
# ---------------------------------------------------------------------------

def _make_norm(norm: str, num_channels: int) -> nn.Module:
    if norm == "batch":
        return nn.BatchNorm2d(num_channels)
    if norm == "instance":
        return nn.InstanceNorm2d(num_channels)
    if norm == "group":
        return nn.GroupNorm(min(32, num_channels), num_channels)
    if norm == "none":
        return nn.Identity()
    raise ValueError(f"Unknown norm type: {norm!r}")


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class ResidualBlock(nn.Module):
    """Standard pre-activation residual block with configurable normalisation."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        norm: str = "batch",
        stride: int = 1,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1,
                               stride=stride, bias=(norm == "none"))
        self.norm1 = _make_norm(norm, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1,
                               bias=(norm == "none"))
        self.norm2 = _make_norm(norm, out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.downsample: Optional[nn.Module] = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                _make_norm(norm, out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x if self.downsample is None else self.downsample(x)
        out = self.relu(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        return self.relu(out + identity)


class BottleneckBlock(nn.Module):
    """1×1 – 3×3 – 1×1 bottleneck residual block."""

    expansion: int = 4

    def __init__(
        self,
        in_channels: int,
        mid_channels: int,
        norm: str = "batch",
        stride: int = 1,
    ) -> None:
        super().__init__()
        out_channels = mid_channels * self.expansion
        bias = norm == "none"
        self.conv1 = nn.Conv2d(in_channels, mid_channels, 1, bias=bias)
        self.norm1 = _make_norm(norm, mid_channels)
        self.conv2 = nn.Conv2d(mid_channels, mid_channels, 3, padding=1,
                               stride=stride, bias=bias)
        self.norm2 = _make_norm(norm, mid_channels)
        self.conv3 = nn.Conv2d(mid_channels, out_channels, 1, bias=bias)
        self.norm3 = _make_norm(norm, out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.downsample: Optional[nn.Module] = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                _make_norm(norm, out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x if self.downsample is None else self.downsample(x)
        out = self.relu(self.norm1(self.conv1(x)))
        out = self.relu(self.norm2(self.conv2(out)))
        out = self.norm3(self.conv3(out))
        return self.relu(out + identity)


# ---------------------------------------------------------------------------
# BasicEncoder  (stride-8 output, optional FPN)
# ---------------------------------------------------------------------------

class BasicEncoder(nn.Module):
    """
    ResNet-style encoder.  Produces a *single* feature map at 1/8 scale by
    default (output_stride=8), or a multi-scale FPN pyramid.

    Args:
        in_channels:    Input image channels (3 for RGB).
        output_dim:     Number of output feature channels at the coarsest level.
        norm:           Normalisation layer type ("batch" | "instance" | "group" | "none").
        output_stride:  Spatial downsampling factor (8 or 4).
        dropout:        Stochastic depth / dropout probability in the stem.
    """

    def __init__(
        self,
        in_channels: int = 3,
        output_dim: int = 256,
        norm: str = "batch",
        output_stride: int = 8,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        assert output_stride in (4, 8), "output_stride must be 4 or 8"

        self.output_dim = output_dim
        self.output_stride = output_stride
        self.norm_type = norm

        # Stem
        self.conv1 = nn.Conv2d(in_channels, 64, 7, stride=2, padding=3, bias=False)
        self.norm1 = _make_norm(norm, 64)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout2d(p=dropout) if dropout > 0 else nn.Identity()

        # Residual stages
        self.layer1 = self._make_layer(64,  64,  num_blocks=2, stride=1, norm=norm)
        self.layer2 = self._make_layer(64,  128, num_blocks=3, stride=2, norm=norm)

        if output_stride == 8:
            self.layer3 = self._make_layer(128, output_dim, num_blocks=3, stride=2, norm=norm)
        else:
            # output_stride == 4 → no further downsampling
            self.layer3 = self._make_layer(128, output_dim, num_blocks=3, stride=1, norm=norm)

        # Weight initialisation
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out",
                                        nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    @staticmethod
    def _make_layer(
        in_ch: int, out_ch: int, num_blocks: int, stride: int, norm: str
    ) -> nn.Sequential:
        layers = [ResidualBlock(in_ch, out_ch, norm=norm, stride=stride)]
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_ch, out_ch, norm=norm))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dropout(self.relu(self.norm1(self.conv1(x))))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x


# ---------------------------------------------------------------------------
# SmallEncoder  (lightweight variant for ablations)
# ---------------------------------------------------------------------------

class SmallEncoder(nn.Module):
    """
    Slimmed-down encoder for low-compute ablations.

    3 residual blocks with 32/64/128 channels, output at 1/8 resolution.
    """

    def __init__(
        self,
        in_channels: int = 3,
        output_dim: int = 128,
        norm: str = "instance",
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.output_dim = output_dim
        self.output_stride = 8

        self.conv1 = nn.Conv2d(in_channels, 32, 7, stride=2, padding=3, bias=False)
        self.norm1 = _make_norm(norm, 32)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout2d(p=dropout) if dropout > 0 else nn.Identity()

        self.layer1 = nn.Sequential(ResidualBlock(32,  32, norm=norm, stride=1))
        self.layer2 = nn.Sequential(ResidualBlock(32,  64, norm=norm, stride=2))
        self.layer3 = nn.Sequential(ResidualBlock(64, output_dim, norm=norm, stride=2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dropout(self.relu(self.norm1(self.conv1(x))))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x


# ---------------------------------------------------------------------------
# Feature Pyramid Network (FPN) head for multi-scale correlation
# ---------------------------------------------------------------------------

class FPNHead(nn.Module):
    """
    Lightweight FPN that fuses encoder feature maps at different scales into a
    fixed-channel pyramid.  Use in conjunction with a multi-stage encoder that
    exposes intermediate activations.
    """

    def __init__(self, in_channels_list: List[int], out_channels: int = 128) -> None:
        super().__init__()
        self.laterals = nn.ModuleList(
            [nn.Conv2d(c, out_channels, 1) for c in in_channels_list]
        )
        self.outputs = nn.ModuleList(
            [nn.Conv2d(out_channels, out_channels, 3, padding=1)
             for _ in in_channels_list]
        )

    def forward(self, features: List[torch.Tensor]) -> List[torch.Tensor]:
        # top-down pathway
        laterals = [l(f) for l, f in zip(self.laterals, features)]
        for i in range(len(laterals) - 2, -1, -1):
            laterals[i] = laterals[i] + F.interpolate(
                laterals[i + 1], size=laterals[i].shape[-2:],
                mode="bilinear", align_corners=True
            )
        return [o(l) for o, l in zip(self.outputs, laterals)]


# ---------------------------------------------------------------------------
# Factory helper
# ---------------------------------------------------------------------------

def build_encoder(cfg) -> nn.Module:
    """
    Build the appropriate encoder from an OmegaConf config node.

    Expected cfg fields:
        type        : "basic" | "small"
        output_dim  : int
        norm        : str
        dropout     : float
        output_stride: int  (BasicEncoder only)
    """
    etype = cfg.get("type", "basic")
    if etype == "basic":
        return BasicEncoder(
            output_dim=cfg.output_dim,
            norm=cfg.get("norm", "batch"),
            output_stride=cfg.get("output_stride", 8),
            dropout=cfg.get("dropout", 0.0),
        )
    if etype == "small":
        return SmallEncoder(
            output_dim=cfg.output_dim,
            norm=cfg.get("norm", "instance"),
            dropout=cfg.get("dropout", 0.0),
        )
    raise ValueError(f"Unknown encoder type: {etype!r}")
