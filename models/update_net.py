"""Data-subproblem solver network for HQS optical flow.

In each unrolled HQS stage k the data subproblem is:

    u^{k+1} = argmin_u  rho(I1, warp(I2, u))  +  mu^k/2 * ||u - v^k||^2

We replace the argmin with a learned ConvGRU update network that takes:
  - indexed correlation features  (B, corr_dim, H, W)
  - context features              (B, ctx_dim,  H, W)
  - current flow u^k              (B, 2, H, W)
  - auxiliary variable v^k        (B, 2, H, W)
  - penalty weight mu^k           (B, 1, H, W)  [broadcast scalar]
  - hidden state h^k              (B, hidden_dim, H, W)

and outputs:
  - flow update Δu                (B, 2, H, W)   →  u^{k+1} = u^k + Δu
  - new hidden state h^{k+1}
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


# ---------------------------------------------------------------------------
# Separable convolution helper
# ---------------------------------------------------------------------------

class SepConv(nn.Module):
    """Depth-wise separable convolution block."""

    def __init__(self, in_ch: int, out_ch: int, kernel: int = 3) -> None:
        super().__init__()
        pad = kernel // 2
        self.dw = nn.Conv2d(in_ch, in_ch, kernel, padding=pad, groups=in_ch, bias=False)
        self.pw = nn.Conv2d(in_ch, out_ch, 1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pw(self.dw(x))


# ---------------------------------------------------------------------------
# ConvGRU cell
# ---------------------------------------------------------------------------

class ConvGRUCell(nn.Module):
    """
    Convolutional GRU cell adapted for recurrent optical flow estimation.

    The reset and update gates operate on spatial feature maps rather than
    flat vectors.

    Args:
        input_dim:  Number of input channels (correlation + context + flow + ...).
        hidden_dim: Number of hidden state channels.
        kernel_size: Convolution kernel size inside gates (default 3).
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        kernel_size: int = 3,
    ) -> None:
        super().__init__()
        pad = kernel_size // 2

        # Reset gate
        self.reset_gate = nn.Conv2d(
            input_dim + hidden_dim, hidden_dim, kernel_size,
            padding=pad, bias=True
        )
        # Update gate
        self.update_gate = nn.Conv2d(
            input_dim + hidden_dim, hidden_dim, kernel_size,
            padding=pad, bias=True
        )
        # Candidate hidden state
        self.out_gate = nn.Conv2d(
            input_dim + hidden_dim, hidden_dim, kernel_size,
            padding=pad, bias=True
        )

    def forward(
        self,
        x: torch.Tensor,
        h: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """
        Args:
            x: Input features (B, input_dim, H, W).
            h: Previous hidden state (B, hidden_dim, H, W) or None (zeros).

        Returns:
            New hidden state (B, hidden_dim, H, W).
        """
        if h is None:
            h = torch.zeros(x.shape[0], self.reset_gate.out_channels,
                            x.shape[2], x.shape[3], device=x.device, dtype=x.dtype)

        xh = torch.cat([x, h], dim=1)
        r = torch.sigmoid(self.reset_gate(xh))
        z = torch.sigmoid(self.update_gate(xh))
        xrh = torch.cat([x, r * h], dim=1)
        n = torch.tanh(self.out_gate(xrh))
        return (1.0 - z) * h + z * n


# ---------------------------------------------------------------------------
# Input projection (correlation + context + flow → inp_dim)
# ---------------------------------------------------------------------------

class InputProjection(nn.Module):
    """
    Projects heterogeneous inputs (correlation features, context, flow
    residual u – v, penalty mu) to a common embedding dimension before
    feeding the GRU.
    """

    def __init__(
        self,
        corr_channels: int,
        context_channels: int,
        hidden_dim: int,
    ) -> None:
        super().__init__()
        # flow channels: 2 (u^k) + 2 (v^k) + 1 (mu, broadcast) = 5
        flow_ch = 5
        total_in = corr_channels + context_channels + flow_ch

        self.proj = nn.Sequential(
            nn.Conv2d(total_in, hidden_dim * 2, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim * 2, hidden_dim, 3, padding=1, bias=True),
            nn.ReLU(inplace=True),
        )

    def forward(
        self,
        corr_feat: torch.Tensor,  # (B, corr_ch, H, W)
        ctx: torch.Tensor,         # (B, ctx_ch, H, W)
        flow_u: torch.Tensor,      # (B, 2, H, W)
        flow_v: torch.Tensor,      # (B, 2, H, W)
        mu: torch.Tensor,          # (B,) or scalar
    ) -> torch.Tensor:
        B, _, H, W = corr_feat.shape
        # Broadcast mu to spatial map
        if not isinstance(mu, torch.Tensor):
            mu_map = torch.full((B, 1, H, W), mu,
                                device=corr_feat.device, dtype=corr_feat.dtype)
        else:
            mu = mu.view(B, 1, 1, 1)
            mu_map = mu.expand(B, 1, H, W)

        x = torch.cat([corr_feat, ctx, flow_u, flow_v, mu_map], dim=1)
        return self.proj(x)


# ---------------------------------------------------------------------------
# Flow head
# ---------------------------------------------------------------------------

class FlowHead(nn.Module):
    """Predicts a 2-channel Δu from hidden state."""

    def __init__(self, hidden_dim: int, head_dim: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(hidden_dim, head_dim, 3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(head_dim, 2, 3, padding=1, bias=True),
        )
        # Initialise final layer near zero for stable start
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.net(h)


# ---------------------------------------------------------------------------
# Up-sampling mask head (ConvexUpsample, RAFT-style)
# ---------------------------------------------------------------------------

class UpsampleMaskHead(nn.Module):
    """
    Predicts a (B, 9 * scale², H, W) mask for convex upsampling.
    Enables full-resolution flow from low-res hidden state.
    """

    def __init__(self, hidden_dim: int, scale: int = 8) -> None:
        super().__init__()
        self.scale = scale
        self.net = nn.Sequential(
            nn.Conv2d(hidden_dim, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, scale * scale * 9, 1),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.net(h)


# ---------------------------------------------------------------------------
# Full DataUpdateNet
# ---------------------------------------------------------------------------

class DataUpdateNet(nn.Module):
    """
    Learned solver for the HQS data subproblem.

    Wraps InputProjection + ConvGRUCell + FlowHead + (optional) mask head.

    Args:
        corr_channels:   Channels of the indexed correlation feature.
        context_channels: Channels of the context encoder output.
        hidden_dim:      GRU hidden state channels.
        head_dim:        Inner dimension of the flow prediction MLP.
        with_upsample_mask: If True, also predict a convex upsampling mask.
        upsample_scale:  Spatial upsampling factor (default 8 for stride-8 encoders).
    """

    def __init__(
        self,
        corr_channels: int,
        context_channels: int,
        hidden_dim: int = 128,
        head_dim: int = 256,
        with_upsample_mask: bool = True,
        upsample_scale: int = 8,
    ) -> None:
        super().__init__()

        self.input_proj = InputProjection(corr_channels, context_channels, hidden_dim)
        self.gru = ConvGRUCell(hidden_dim, hidden_dim)
        self.flow_head = FlowHead(hidden_dim, head_dim)

        self.up_head: Optional[UpsampleMaskHead] = None
        if with_upsample_mask:
            self.up_head = UpsampleMaskHead(hidden_dim, scale=upsample_scale)

    def forward(
        self,
        corr_feat: torch.Tensor,  # (B, corr_ch, H, W)
        context: torch.Tensor,    # (B, ctx_ch, H, W)
        flow_u: torch.Tensor,     # (B, 2, H, W)  current u^k
        flow_v: torch.Tensor,     # (B, 2, H, W)  current v^k
        mu: torch.Tensor,         # (B,)           penalty weight at stage k
        hidden: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Returns:
            delta_u  – (B, 2, H, W) flow update.
            hidden   – new hidden state.
            up_mask  – (B, 9*scale², H, W) or None.
        """
        inp = self.input_proj(corr_feat, context, flow_u, flow_v, mu)
        hidden = self.gru(inp, hidden)
        delta_u = self.flow_head(hidden)
        up_mask = self.up_head(hidden) if self.up_head is not None else None
        return delta_u, hidden, up_mask
