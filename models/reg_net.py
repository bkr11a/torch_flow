"""Regularisation subproblem network (learned proximal operator) for HQSFlow.

In each unrolled HQS stage the regularisation subproblem is:

    v^{k+1} = argmin_v  lambda * phi(v)  +  mu^k/2 * ||u^{k+1} - v||^2

Its closed-form is the proximal operator of (lambda/mu^k) * phi:

    v^{k+1} = prox_{sigma^2 * phi}( u^{k+1} )    where sigma = sqrt(lambda/mu^k)

We replace this with a *learned* proximal operator (CNN-based flow denoiser)
that is noise-level–conditioned on sigma, following the half-quadratic /
plug-and-play prior paradigm.

Two architectures are available:
  DnCNNProxNet   – DnCNN-style residual CNN (fast, good for ablations).
  UNetProxNet    – U-Net–based proximal operator (stronger, default).

Both are conditioned on the noise level sigma via FiLM (Feature-wise Linear
Modulation) layers so that the same network can adapt its denoising strength
at each unrolled stage.
"""
from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


# ---------------------------------------------------------------------------
# FiLM conditioning
# ---------------------------------------------------------------------------

class FiLMLayer(nn.Module):
    """
    Feature-wise Linear Modulation: scales and shifts feature maps based on
    a conditioning scalar (noise level sigma).

        out = gamma(sigma) * x + beta(sigma)

    where gamma and beta are small MLPs.
    """

    def __init__(self, channels: int, cond_dim: int = 64) -> None:
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(1, cond_dim), nn.SiLU(),
            nn.Linear(cond_dim, channels * 2),
        )

    def forward(self, x: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:     (B, C, H, W)
            sigma: (B,) noise-level scalar per sample.
        """
        params = self.fc(sigma.view(-1, 1))           # (B, 2C)
        gamma, beta = params.chunk(2, dim=-1)          # each (B, C)
        gamma = gamma.view(-1, x.shape[1], 1, 1) + 1  # init near identity
        beta  = beta.view(-1, x.shape[1], 1, 1)
        return gamma * x + beta


# ---------------------------------------------------------------------------
# DnCNN-style proximal network
# ---------------------------------------------------------------------------

class DnCNNProxNet(nn.Module):
    """
    Lightweight residual CNN proximal operator inspired by DnCNN.

    Structure: Conv → [Conv-BN-ReLU × depth] → Conv
    The network predicts a residual r and returns u - r (denoised flow).

    Noise-level conditioning is applied after the first and last interior
    blocks via FiLM layers.

    Args:
        in_channels:  2 (optical flow u and v components).
        num_features: Feature channels inside the network.
        depth:        Number of intermediate Conv-BN-ReLU blocks.
    """

    def __init__(
        self,
        in_channels: int = 2,
        num_features: int = 64,
        depth: int = 6,
    ) -> None:
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, num_features, 3, padding=1, bias=True),
            nn.ReLU(inplace=True),
        )
        blocks = []
        for _ in range(depth):
            blocks += [
                nn.Conv2d(num_features, num_features, 3, padding=1, bias=False),
                nn.BatchNorm2d(num_features),
                nn.ReLU(inplace=True),
            ]
        self.body = nn.Sequential(*blocks)
        self.tail = nn.Conv2d(num_features, in_channels, 3, padding=1, bias=True)

        self.film1 = FiLMLayer(num_features)
        self.film2 = FiLMLayer(num_features)

        # Zero-init tail so network starts as identity
        nn.init.zeros_(self.tail.weight)
        nn.init.zeros_(self.tail.bias)

    def forward(self, u: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        """
        Args:
            u:     (B, 2, H, W) input flow to denoise.
            sigma: (B,) per-sample noise level = sqrt(lambda/mu).
        Returns:
            v^{k+1} (B, 2, H, W) regularised flow.
        """
        feat = self.head(u)
        n = len(self.body) // 3  # number of [conv-bn-relu] triplets
        mid = n // 2
        for i in range(n):
            feat = self.body[3 * i](feat)    # conv
            feat = self.body[3 * i + 1](feat)  # bn
            feat = self.body[3 * i + 2](feat)  # relu
            if i == 0:
                feat = self.film1(feat, sigma)
            if i == mid:
                feat = self.film2(feat, sigma)
        residual = self.tail(feat)
        return u - residual


# ---------------------------------------------------------------------------
# U-Net proximal network (stronger default)
# ---------------------------------------------------------------------------

class UNetConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.InstanceNorm2d(out_ch, affine=True),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.InstanceNorm2d(out_ch, affine=True),
            nn.LeakyReLU(0.1, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class UNetProxNet(nn.Module):
    """
    U-Net–based proximal operator for the HQS regularisation subproblem.

    Encoder path:   [64, 128, 256] channels at strides [1, 2, 4].
    Decoder path:   symmetric with skip connections + FiLM conditioning.

    Args:
        in_channels:  2 (flow u, v).
        base_ch:      Channel count at first encoder level (default 64).
        depth:        Number of encoder/decoder levels (default 3).
    """

    def __init__(
        self,
        in_channels: int = 2,
        base_ch: int = 64,
        depth: int = 3,
    ) -> None:
        super().__init__()
        self.depth = depth

        # Encoder
        self.enc_blocks = nn.ModuleList()
        self.downs = nn.ModuleList()
        ch = in_channels
        enc_channels = []
        for i in range(depth):
            out_ch = base_ch * (2 ** i)
            self.enc_blocks.append(UNetConvBlock(ch, out_ch))
            enc_channels.append(out_ch)
            ch = out_ch
            if i < depth - 1:
                self.downs.append(nn.Conv2d(ch, ch, 3, stride=2, padding=1, bias=False))
            else:
                self.downs.append(nn.Identity())

        # Bottleneck FiLM
        self.bottleneck_film = FiLMLayer(enc_channels[-1])

        # Decoder
        self.ups = nn.ModuleList()
        self.dec_blocks = nn.ModuleList()
        self.dec_films = nn.ModuleList()
        for i in range(depth - 2, -1, -1):
            skip_ch = enc_channels[i]
            in_ch = enc_channels[i + 1]
            self.ups.append(
                nn.ConvTranspose2d(in_ch, skip_ch, kernel_size=2, stride=2)
            )
            self.dec_blocks.append(UNetConvBlock(skip_ch * 2, skip_ch))
            self.dec_films.append(FiLMLayer(skip_ch))

        self.tail = nn.Conv2d(enc_channels[0], in_channels, 1, bias=True)
        nn.init.zeros_(self.tail.weight)
        nn.init.zeros_(self.tail.bias)

    def forward(self, u: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        """
        Args:
            u:     (B, 2, H, W) flow to regularise.
            sigma: (B,) noise level sqrt(lambda/mu).
        Returns:
            (B, 2, H, W) regularised flow v^{k+1}.
        """
        skips = []
        x = u
        for i, (enc, down) in enumerate(zip(self.enc_blocks, self.downs)):
            x = enc(x)
            skips.append(x)
            if i < self.depth - 1:
                x = down(x)

        # Bottleneck
        x = self.bottleneck_film(x, sigma)

        # Decoder
        for up, dec, film, skip in zip(
            self.ups, self.dec_blocks, self.dec_films, reversed(skips[:-1])
        ):
            x = up(x)
            # Handle potential size mismatch from odd dimensions
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(x, size=skip.shape[-2:],
                                  mode="bilinear", align_corners=True)
            x = dec(torch.cat([x, skip], dim=1))
            x = film(x, sigma)

        residual = self.tail(x)
        return u - residual


# ---------------------------------------------------------------------------
# Identity proximal (ablation: no regularisation)
# ---------------------------------------------------------------------------

class IdentityProxNet(nn.Module):
    """Pass-through proximal operator for ablating the regulariser."""

    def forward(self, u: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        return u


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_prox_net(cfg) -> nn.Module:
    """
    Build a proximal operator network from config.

    Expected cfg fields:
        type         : "unet" | "dncnn" | "none"
        num_features : int  (DnCNN)
        depth        : int
        base_ch      : int  (UNet)
    """
    ptype = cfg.get("type", "unet")
    if ptype == "none":
        return IdentityProxNet()
    if ptype == "dncnn":
        return DnCNNProxNet(
            num_features=cfg.get("num_features", 64),
            depth=cfg.get("depth", 6),
        )
    if ptype == "unet":
        return UNetProxNet(
            base_ch=cfg.get("base_ch", 64),
            depth=cfg.get("depth", 3),
        )
    raise ValueError(f"Unknown proximal network type: {ptype!r}")
