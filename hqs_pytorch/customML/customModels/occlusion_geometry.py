"""Inference-time geometry for occlusion/disocclusion-aware optical flow.

Conventions
-----------
All flow tensors use the TensorFlow-port convention [dy, dx] and shape
(B, 2, H, W). No ground-truth tensor is accepted by this module.

The forward splat uses bilinear accumulation through scatter_add_. Gradients
flow through the bilinear weights. Integer neighbour selection is piecewise
constant, as in standard differentiable splatting implementations.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor


@dataclass
class FlowGeometry:
    """Predicted geometric evidence at one resolution."""

    in_bounds: Tensor
    fb_error: Tensor
    fb_error_normalised: Tensor
    fb_confidence: Tensor
    occupancy: Tensor
    hole_score: Tensor
    collision_score: Tensor

    def detached_dict(self) -> dict[str, Tensor]:
        return {
            "in_bounds": self.in_bounds.detach(),
            "fb_error": self.fb_error.detach(),
            "fb_error_normalised": self.fb_error_normalised.detach(),
            "fb_confidence": self.fb_confidence.detach(),
            "occupancy": self.occupancy.detach(),
            "hole_score": self.hole_score.detach(),
            "collision_score": self.collision_score.detach(),
        }


def _check_flow_yx(flow_yx: Tensor) -> None:
    if flow_yx.ndim != 4 or flow_yx.shape[1] != 2:
        raise ValueError(
            f"Expected flow [B,2,H,W] in [dy,dx] order; got {tuple(flow_yx.shape)}"
        )
    if not flow_yx.is_floating_point():
        raise TypeError("Flow must be floating point.")


def coords_grid_yx(
    batch: int,
    height: int,
    width: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    """Return absolute [y,x] coordinates with shape [B,2,H,W]."""
    y, x = torch.meshgrid(
        torch.arange(height, device=device, dtype=dtype),
        torch.arange(width, device=device, dtype=dtype),
        indexing="ij",
    )
    return torch.stack((y, x), dim=0).unsqueeze(0).expand(batch, -1, -1, -1)


def flow_in_bounds_mask(flow_yx: Tensor, margin: float = 0.0) -> Tensor:
    """Geometric sampling validity computed only from predicted flow."""
    _check_flow_yx(flow_yx)
    b, _, h, w = flow_yx.shape
    q = coords_grid_yx(b, h, w, device=flow_yx.device, dtype=flow_yx.dtype) + flow_yx
    y, x = q[:, 0:1], q[:, 1:2]
    return (
        (y >= margin)
        & (y <= (h - 1 - margin))
        & (x >= margin)
        & (x <= (w - 1 - margin))
    ).to(flow_yx.dtype)


def backward_warp_yx(
    source: Tensor,
    flow_yx: Tensor,
    *,
    padding_mode: str = "zeros",
    align_corners: bool = True,
) -> Tensor:
    """Sample ``source`` at x + flow(x), with flow channels [dy, dx]."""
    _check_flow_yx(flow_yx)
    if source.ndim != 4:
        raise ValueError(f"Expected source [B,C,H,W], got {tuple(source.shape)}")
    b, _, h, w = source.shape
    if flow_yx.shape[0] != b or flow_yx.shape[-2:] != (h, w):
        raise ValueError("source and flow must have the same batch/spatial dimensions.")

    q = coords_grid_yx(b, h, w, device=source.device, dtype=source.dtype) + flow_yx
    y, x = q[:, 0], q[:, 1]
    gy = torch.zeros_like(y) if h == 1 else 2.0 * y / (h - 1) - 1.0
    gx = torch.zeros_like(x) if w == 1 else 2.0 * x / (w - 1) - 1.0
    grid = torch.stack((gx, gy), dim=-1)
    return F.grid_sample(
        source,
        grid,
        mode="bilinear",
        padding_mode=padding_mode,
        align_corners=align_corners,
    )


def compose_forward_backward_yx(flow_ab: Tensor, flow_ba: Tensor) -> Tensor:
    """Return w_ab(x) + w_ba(x + w_ab(x))."""
    _check_flow_yx(flow_ab)
    _check_flow_yx(flow_ba)
    if flow_ab.shape != flow_ba.shape:
        raise ValueError("Forward and reverse flows must have identical shapes.")
    return flow_ab + backward_warp_yx(flow_ba, flow_ab)


def forward_backward_error_yx(
    flow_ab: Tensor,
    flow_ba: Tensor,
    *,
    alpha: float = 0.01,
    beta: float = 0.5,
    eps: float = 1e-6,
) -> tuple[Tensor, Tensor, Tensor]:
    """Return absolute error, normalised error, and soft consistency confidence.

    The normaliser follows the common scale-aware consistency structure:
        alpha * (||w_ab||^2 + ||warp(w_ba)||^2) + beta.
    Confidence is exp(-normalised_error), not a hard threshold.
    """
    warped_ba = backward_warp_yx(flow_ba, flow_ab)
    residual = flow_ab + warped_ba
    error_sq = residual.square().sum(dim=1, keepdim=True)
    scale = alpha * (
        flow_ab.square().sum(dim=1, keepdim=True)
        + warped_ba.square().sum(dim=1, keepdim=True)
    ) + beta
    normalised = error_sq / scale.clamp_min(eps)
    return error_sq.add(eps).sqrt(), normalised, torch.exp(-normalised)


def _scatter_add_flat(
    output: Tensor,
    flat_index: Tensor,
    values: Tensor,
) -> None:
    output.scatter_add_(2, flat_index, values)


def forward_splat_occupancy_yx(
    flow_yx: Tensor,
    *,
    source_weight: Optional[Tensor] = None,
    eps: float = 1e-6,
) -> Tensor:
    """Bilinearly splat source support into the target frame.

    Returns accumulated target occupancy [B,1,H,W]. Values near zero identify
    target holes/disocclusions; values above one indicate many-to-one support.

    ``source_weight`` may be a predicted source confidence [B,1,H,W]. It must
    never be a ground-truth visibility tensor when used inside model.forward().
    """
    _check_flow_yx(flow_yx)
    b, _, h, w = flow_yx.shape
    if source_weight is None:
        source_weight = torch.ones(
            b, 1, h, w, device=flow_yx.device, dtype=flow_yx.dtype
        )
    if source_weight.shape != (b, 1, h, w):
        raise ValueError(
            f"source_weight must be [B,1,H,W], got {tuple(source_weight.shape)}"
        )

    base = coords_grid_yx(b, h, w, device=flow_yx.device, dtype=flow_yx.dtype)
    target = base + flow_yx
    ty, tx = target[:, 0], target[:, 1]

    y0 = torch.floor(ty)
    x0 = torch.floor(tx)
    y1 = y0 + 1.0
    x1 = x0 + 1.0

    wy1 = ty - y0
    wx1 = tx - x0
    wy0 = 1.0 - wy1
    wx0 = 1.0 - wx1

    output = torch.zeros(b, 1, h * w, device=flow_yx.device, dtype=flow_yx.dtype)
    src = source_weight[:, 0]

    for yy, xx, weight in (
        (y0, x0, wy0 * wx0),
        (y0, x1, wy0 * wx1),
        (y1, x0, wy1 * wx0),
        (y1, x1, wy1 * wx1),
    ):
        valid = (yy >= 0) & (yy <= h - 1) & (xx >= 0) & (xx <= w - 1)
        yi = yy.clamp(0, h - 1).long()
        xi = xx.clamp(0, w - 1).long()
        index = (yi * w + xi).reshape(b, 1, h * w)
        contribution = (src * weight * valid.to(weight.dtype)).reshape(b, 1, h * w)
        _scatter_add_flat(output, index, contribution)

    return output.reshape(b, 1, h, w).clamp_min(0.0)


def occupancy_scores(
    occupancy: Tensor,
    *,
    hole_threshold: float = 0.25,
    collision_threshold: float = 1.0,
    temperature: float = 0.1,
) -> tuple[Tensor, Tensor]:
    """Return smooth hole and collision scores in [0,1]."""
    if occupancy.ndim != 4 or occupancy.shape[1] != 1:
        raise ValueError("occupancy must have shape [B,1,H,W].")
    if temperature <= 0:
        raise ValueError("temperature must be positive.")
    hole = torch.sigmoid((hole_threshold - occupancy) / temperature)
    collision = torch.sigmoid((occupancy - collision_threshold) / temperature)
    return hole, collision


def compute_flow_geometry_yx(
    flow_ab: Tensor,
    flow_ba: Tensor,
    *,
    fb_alpha: float = 0.01,
    fb_beta: float = 0.5,
    hole_threshold: float = 0.25,
    collision_threshold: float = 1.0,
    occupancy_temperature: float = 0.1,
    detach_reverse_for_geometry: bool = False,
) -> FlowGeometry:
    """Compute all inference-time geometric evidence for direction a -> b."""
    reverse = flow_ba.detach() if detach_reverse_for_geometry else flow_ba
    fb_error, fb_normalised, fb_conf = forward_backward_error_yx(
        flow_ab, reverse, alpha=fb_alpha, beta=fb_beta
    )
    occupancy = forward_splat_occupancy_yx(flow_ab)
    hole, collision = occupancy_scores(
        occupancy,
        hole_threshold=hole_threshold,
        collision_threshold=collision_threshold,
        temperature=occupancy_temperature,
    )
    return FlowGeometry(
        in_bounds=flow_in_bounds_mask(flow_ab),
        fb_error=fb_error,
        fb_error_normalised=fb_normalised,
        fb_confidence=fb_conf,
        occupancy=occupancy,
        hole_score=hole,
        collision_score=collision,
    )
