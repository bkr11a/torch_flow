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


def resize_flow(
    flow: torch.Tensor,
    size: tuple[int, int],
    *,
    align_corners: bool = True,
) -> torch.Tensor:
    """Resize an ``(x, y)`` flow field while preserving pixel displacement.

    A flow component is measured in pixels of its current grid.  Spatial
    interpolation alone is therefore insufficient: horizontal and vertical
    components must also be scaled by the corresponding grid-size ratios.
    This helper handles non-power-of-two and odd-sized transitions safely.
    """
    if flow.ndim != 4 or flow.shape[1] != 2:
        raise ValueError(f"Expected flow [B,2,H,W], got {tuple(flow.shape)}")

    in_h, in_w = flow.shape[-2:]
    out_h, out_w = int(size[0]), int(size[1])
    if out_h <= 0 or out_w <= 0:
        raise ValueError(f"Invalid output size: {(out_h, out_w)}")
    if (in_h, in_w) == (out_h, out_w):
        return flow

    result = F.interpolate(
        flow,
        size=(out_h, out_w),
        mode="bilinear",
        align_corners=align_corners,
    )
    scale = result.new_tensor(
        [float(out_w) / float(in_w), float(out_h) / float(in_h)]
    ).view(1, 2, 1, 1)
    return result * scale


def flow_in_bounds_mask(flow: torch.Tensor) -> torch.Tensor:
    """Return a hard source-grid mask for valid backward-warp coordinates.

    Args:
        flow: ``(B,2,H,W)`` flow in repository ``(x,y)`` convention.

    Returns:
        Float mask ``(B,1,H,W)`` containing exactly zero or one.
    """
    if flow.ndim != 4 or flow.shape[1] != 2:
        raise ValueError(f"Expected flow [B,2,H,W], got {tuple(flow.shape)}")
    b, _, h, w = flow.shape
    grid = coords_grid(b, h, w, device=flow.device).to(dtype=flow.dtype)
    sample = grid + flow
    valid_x = (sample[:, 0:1] >= 0.0) & (sample[:, 0:1] <= float(w - 1))
    valid_y = (sample[:, 1:2] >= 0.0) & (sample[:, 1:2] <= float(h - 1))
    return (valid_x & valid_y).to(dtype=flow.dtype)


def convex_upsample(
    flow: torch.Tensor,
    mask_logits: torch.Tensor,
    rate: int,
) -> torch.Tensor:
    """RAFT-style learned convex upsampling for an ``(x,y)`` flow field.

    ``mask_logits`` must contain ``9 * rate**2`` channels.  The softmax is
    taken over a 3x3 neighbourhood, so every output vector is a convex
    combination of nearby low-resolution vectors.  Multiplication by
    ``rate`` converts displacement units to the finer grid.
    """
    if flow.ndim != 4 or flow.shape[1] != 2:
        raise ValueError(f"Expected flow [B,2,H,W], got {tuple(flow.shape)}")
    if rate < 1:
        raise ValueError(f"rate must be positive, got {rate}")

    b, channels, h, w = flow.shape
    expected = 9 * rate * rate
    if mask_logits.shape != (b, expected, h, w):
        raise ValueError(
            "Invalid convex-upsample mask shape: "
            f"expected {(b, expected, h, w)}, got {tuple(mask_logits.shape)}"
        )

    mask = mask_logits.view(b, 1, 9, rate, rate, h, w)
    mask = torch.softmax(mask, dim=2)
    neighbourhood = F.unfold(rate * flow, kernel_size=3, padding=1)
    neighbourhood = neighbourhood.view(b, channels, 9, 1, 1, h, w)
    upsampled = torch.sum(mask * neighbourhood, dim=2)
    return upsampled.permute(0, 1, 4, 2, 5, 3).reshape(
        b, channels, h * rate, w * rate
    )


def coords_grid(batch: int, height: int, width: int,
                device: torch.device) -> torch.Tensor:
    """Return (B, 2, H, W) integer coordinate grid (x, y)."""
    ys = torch.arange(height, device=device, dtype=torch.float32)
    xs = torch.arange(width,  device=device, dtype=torch.float32)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    grid = torch.stack([grid_x, grid_y], dim=0).unsqueeze(0)  # (1,2,H,W)
    return grid.expand(batch, -1, -1, -1)
