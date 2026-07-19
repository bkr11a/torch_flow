"""All-pairs and local correlation (cost volume) modules.

CorrBlock        – RAFT-style all-pairs 4D correlation with multi-scale
                   pooling, indexed by current flow estimate.
LocalCorrBlock   – Traditional local window correlation (cheaper, useful
                   for ablation studies of correlation type).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

from .warp import coords_grid


# ---------------------------------------------------------------------------
# All-pairs correlation
# ---------------------------------------------------------------------------

class CorrBlock(nn.Module):
    """
    Builds the full H×W × H×W correlation volume between two feature maps,
    then indexes it by the current flow to produce a local neighbourhood
    feature vector at each pixel.

    The volume is pooled at multiple scales (``num_levels`` levels, each
    halved by average pooling) to capture both fine and coarse matches.

    Args:
        num_levels: Number of pyramid levels (default 4).
        radius:     Look-around radius in each spatial direction (default 4).
                    Feature vector length = num_levels × (2r+1)².
    """

    def __init__(self, num_levels: int = 4, radius: int = 4) -> None:
        super().__init__()
        self.num_levels = num_levels
        self.radius = radius
        # Correlation pyramid – populated in initialise()
        self._corr_pyramid: List[torch.Tensor] = []

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def out_channels(self) -> int:
        """Dimensionality of the indexed correlation feature vector."""
        return self.num_levels * (2 * self.radius + 1) ** 2

    # ------------------------------------------------------------------
    # Build / index
    # ------------------------------------------------------------------

    def initialise(self, fmap1: torch.Tensor, fmap2: torch.Tensor) -> None:
        """
        Compute and store the multi-scale correlation pyramid.

        Args:
            fmap1, fmap2: (B, C, H, W) L2-normalised feature maps.
        """
        B, C, H, W = fmap1.shape
        # Flatten spatial dims for batched matrix multiplication
        f1 = fmap1.view(B, C, H * W).permute(0, 2, 1)   # (B, HW, C)
        f2 = fmap2.view(B, C, H * W)                     # (B, C, HW)
        corr = torch.bmm(f1, f2) / (C ** 0.5)            # (B, HW, HW)
        corr = corr.view(B * H * W, 1, H, W)

        self._corr_pyramid = [corr]
        for _ in range(1, self.num_levels):
            corr = F.avg_pool2d(corr, kernel_size=2, stride=2)
            self._corr_pyramid.append(corr)

    def index(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Index the correlation pyramid at sub-pixel positions *coords*.

        Args:
            coords: (B, 2, H, W) absolute pixel coordinates (x, y).

        Returns:
            Correlation features (B, num_levels × (2r+1)², H, W).
        """
        r = self.radius
        B, _, H, W = coords.shape

        # Build neighbourhood delta grid once
        dx = torch.linspace(-r, r, 2 * r + 1, device=coords.device)
        dy = torch.linspace(-r, r, 2 * r + 1, device=coords.device)
        delta_y, delta_x = torch.meshgrid(dy, dx, indexing="ij")
        # delta: (1, 1, 2r+1, 2r+1, 2)
        delta = torch.stack([delta_x, delta_y], dim=-1).view(1, 1, 2 * r + 1, 2 * r + 1, 2)

        out_pyramid = []
        for level, corr in enumerate(self._corr_pyramid):
            lvl_h, lvl_w = corr.shape[-2], corr.shape[-1]

            # coords scaled to this level: (B, H, W, 2) → (B, H*W, 1, 1, 2)
            centroid = coords.permute(0, 2, 3, 1) / (2 ** level)   # (B, H, W, 2)
            centroid = centroid.reshape(B, H * W, 1, 1, 2)

            # Absolute sampling positions in pyramid-level pixel space: (B, HW, 2r+1, 2r+1, 2)
            coords_lvl = centroid + delta   # broadcast over HW

            # Normalise to [-1, 1]
            coords_norm = coords_lvl.clone()
            coords_norm[..., 0] = coords_lvl[..., 0] / (lvl_w - 1) * 2.0 - 1.0
            coords_norm[..., 1] = coords_lvl[..., 1] / (lvl_h - 1) * 2.0 - 1.0

            # Reshape for grid_sample: need (N, C, H_in, W_in) + grid (N, H_out, W_out, 2)
            # corr: (B*H*W, 1, lvl_h, lvl_w)  (built during initialise)
            # grid: (B*H*W, 2r+1, 2r+1, 2)
            grid = coords_norm.reshape(B * H * W, 2 * r + 1, 2 * r + 1, 2)

            sampled = F.grid_sample(
                corr, grid,
                mode="bilinear", padding_mode="border", align_corners=True
            )  # (B*H*W, 1, 2r+1, 2r+1)
            sampled = sampled.view(B, H * W, (2 * r + 1) ** 2)
            out_pyramid.append(sampled)

        out = torch.cat(out_pyramid, dim=-1)  # (B, HW, num_levels*(2r+1)^2)
        return out.permute(0, 2, 1).view(B, -1, H, W)

    def forward(self, fmap1: torch.Tensor, fmap2: torch.Tensor,
                flow: torch.Tensor) -> torch.Tensor:
        """
        Build pyramid (or reuse) and return indexed correlations.

        For use in a recurrent loop where fmap1/fmap2 are fixed, call
        ``initialise`` once and then ``index`` repeatedly.  Calling
        ``forward`` re-builds each time.
        """
        self.initialise(fmap1, fmap2)
        B, _, H, W = flow.shape
        coords0 = coords_grid(B, H, W, device=flow.device)
        return self.index(coords0 + flow)


# ---------------------------------------------------------------------------
# Local correlation (cheaper alternative)
# ---------------------------------------------------------------------------

class LocalCorrBlock(nn.Module):
    """
    Traditional local correlation: for each pixel in fmap1, compute dot
    products with neighbours in a (2r+1)×(2r+1) window in fmap2 offset by
    the current flow estimate.

    Args:
        radius: Neighbourhood radius (default 4).
        max_displacement: Passed as alias for radius when loading older ckpts.
    """

    def __init__(
        self,
        radius: int = 4,
        channel_chunk_size: int = 0,
        checkpoint_chunks: bool = False,
    ) -> None:
        super().__init__()
        self.radius = radius
        self.channel_chunk_size = int(channel_chunk_size)
        self.checkpoint_chunks = bool(checkpoint_chunks)
        self._pad = nn.ZeroPad2d(radius)

    @property
    def out_channels(self) -> int:
        return (2 * self.radius + 1) ** 2

    def forward(self, fmap1: torch.Tensor, fmap2: torch.Tensor,
                flow: torch.Tensor) -> torch.Tensor:
        """
        Args:
            fmap1, fmap2: (B, C, H, W) feature maps.
            flow:         (B, 2, H, W) current flow (integer-rounded for sampling).

        Returns:
            (B, (2r+1)², H, W) correlation features.
        """
        r = self.radius
        B, C, H, W = fmap1.shape

        # Warp fmap2 by current flow to align with fmap1
        fmap2_warped = self._warp_fmap2(fmap2, flow)

        if self.channel_chunk_size > 0:
            return self._chunked_correlation(fmap1, fmap2_warped)

        fmap2_pad = self._pad(fmap2_warped)
        corr_list = []
        for dy in range(2 * r + 1):
            for dx in range(2 * r + 1):
                neighbour = fmap2_pad[:, :, dy:dy + H, dx:dx + W]
                dot = (fmap1 * neighbour).sum(dim=1, keepdim=True) / (C ** 0.5)
                corr_list.append(dot)

        return torch.cat(corr_list, dim=1)  # (B, (2r+1)², H, W)

    def _chunked_correlation(
        self,
        fmap1: torch.Tensor,
        fmap2_warped: torch.Tensor,
    ) -> torch.Tensor:
        """Equivalent local correlation with bounded temporary memory.

        ``unfold`` over all channels at 1/2 resolution can be prohibitively
        large.  Accumulating channel chunks retains the exact dot product while
        trading a small number of kernels for a controlled peak allocation.
        """
        r = self.radius
        kernel = 2 * r + 1
        b, channels, h, w = fmap1.shape
        chunk_size = min(max(self.channel_chunk_size, 1), channels)
        correlation: torch.Tensor | None = None
        for start in range(0, channels, chunk_size):
            stop = min(start + chunk_size, channels)
            source_chunk = fmap1[:, start:stop]
            target_chunk = fmap2_warped[:, start:stop]
            if self.checkpoint_chunks and torch.is_grad_enabled() and (
                source_chunk.requires_grad or target_chunk.requires_grad
            ):
                from torch.utils.checkpoint import checkpoint

                partial = checkpoint(
                    self._correlate_channel_chunk,
                    source_chunk,
                    target_chunk,
                    r,
                    use_reentrant=False,
                )
            else:
                partial = self._correlate_channel_chunk(
                    source_chunk, target_chunk, r
                )
            correlation = (
                partial if correlation is None else correlation + partial
            )
        assert correlation is not None
        return correlation / (channels ** 0.5)

    @staticmethod
    def _correlate_channel_chunk(
        source: torch.Tensor,
        target: torch.Tensor,
        radius: int,
    ) -> torch.Tensor:
        kernel = 2 * int(radius) + 1
        b, channels, h, w = source.shape
        patches = F.unfold(
            target,
            kernel_size=kernel,
            padding=int(radius),
        ).view(b, channels, kernel * kernel, h, w)
        return (source.unsqueeze(2) * patches).sum(dim=1)

    @staticmethod
    def _warp_fmap2(fmap2: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
        from .warp import backward_warp
        return backward_warp(fmap2, flow)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_corr_block(cfg) -> nn.Module:
    ctype = cfg.get("type", "all_pairs")
    if ctype == "all_pairs":
        return CorrBlock(
            num_levels=cfg.get("num_levels", 4),
            radius=cfg.get("radius", 4),
        )
    if ctype == "local":
        return LocalCorrBlock(
            radius=cfg.get("radius", 4),
            channel_chunk_size=cfg.get("channel_chunk_size", 0),
            checkpoint_chunks=cfg.get("checkpoint_chunks", False),
        )
    raise ValueError(f"Unknown correlation type: {ctype!r}")
