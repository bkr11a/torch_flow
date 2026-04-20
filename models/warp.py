"""Differentiable flow warping utilities.

backward_warp  – backward (inverse) warp: sample from img2 at locations
                 defined by img1 + flow.  This is the standard operation
                 for optical flow.

forward_warp   – forward (splatting) warp used when needed for photometric
                 consistency checks.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def backward_warp(
    img: torch.Tensor,
    flow: torch.Tensor,
    padding_mode: str = "border",
    align_corners: bool = True,
) -> torch.Tensor:
    """
    Backward warp *img* using *flow*.

    Args:
        img:          (B, C, H, W) tensor to be warped (typically I_2).
        flow:         (B, 2, H, W) optical flow in pixels (u=horizontal, v=vertical).
        padding_mode: "zeros" | "border" | "reflection"
        align_corners: passed to grid_sample.

    Returns:
        Warped image (B, C, H, W).
    """
    B, _, H, W = img.shape
    device = img.device
    dtype = img.dtype

    # Build normalised grid [-1, 1]
    grid_y, grid_x = torch.meshgrid(
        torch.arange(H, device=device, dtype=dtype),
        torch.arange(W, device=device, dtype=dtype),
        indexing="ij",
    )
    base_grid = torch.stack([grid_x, grid_y], dim=0).unsqueeze(0)  # (1,2,H,W)

    # Pixel positions after warping
    sample_grid = base_grid + flow  # (B,2,H,W)

    # Normalise to [-1, 1]
    norm_x = 2.0 * sample_grid[:, 0] / (W - 1) - 1.0
    norm_y = 2.0 * sample_grid[:, 1] / (H - 1) - 1.0
    grid_norm = torch.stack([norm_x, norm_y], dim=-1)  # (B,H,W,2)

    return F.grid_sample(
        img, grid_norm,
        mode="bilinear",
        padding_mode=padding_mode,
        align_corners=align_corners,
    )


def upsample_flow(flow: torch.Tensor, scale_factor: int = 8) -> torch.Tensor:
    """Upsample flow and rescale magnitude accordingly."""
    return F.interpolate(
        flow * scale_factor,
        scale_factor=scale_factor,
        mode="bilinear",
        align_corners=True,
    )


def coords_grid(batch: int, height: int, width: int,
                device: torch.device) -> torch.Tensor:
    """Return (B, 2, H, W) integer coordinate grid (x, y)."""
    ys = torch.arange(height, device=device, dtype=torch.float32)
    xs = torch.arange(width,  device=device, dtype=torch.float32)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    grid = torch.stack([grid_x, grid_y], dim=0).unsqueeze(0)  # (1,2,H,W)
    return grid.expand(batch, -1, -1, -1)
