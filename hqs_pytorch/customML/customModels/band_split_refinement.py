"""Physics-preserving half-resolution and band-split flow refinement.

The module deliberately separates three operations:

1. a bounded analytic OFCE data update, evaluated only where the predicted
   data gate says that the observation operator is applicable;
2. a source-conditioned learned prior, which never receives target image or
   correlation features; and
3. a full-resolution learned residual projected by the fixed high-pass
   operator H = I - U D so it cannot rewrite the coarse solution.

All internal flow tensors use [dy, dx].
"""
from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .occlusion_geometry import backward_warp_yx, flow_in_bounds_mask


def _groups(channels: int) -> int:
    groups = min(8, channels)
    while groups > 1 and channels % groups:
        groups -= 1
    return groups


def _conv_block(in_channels: int, out_channels: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
        nn.GroupNorm(_groups(out_channels), out_channels),
        nn.SiLU(inplace=True),
    )


def resize_flow_yx(flow_yx: Tensor, size: tuple[int, int]) -> Tensor:
    """Resize [dy,dx] flow while preserving displacement units."""
    in_h, in_w = flow_yx.shape[-2:]
    out_h, out_w = size
    if (in_h, in_w) == (out_h, out_w):
        return flow_yx
    result = F.interpolate(
        flow_yx, size=size, mode="bilinear", align_corners=True
    )
    scale = flow_yx.new_tensor(
        [float(out_h) / float(in_h), float(out_w) / float(in_w)]
    ).view(1, 2, 1, 1)
    return result * scale


def central_gradient_x(value: Tensor) -> Tensor:
    gradient = 0.5 * (value[..., 2:] - value[..., :-2])
    return F.pad(gradient, (1, 1, 0, 0), mode="replicate")


def central_gradient_y(value: Tensor) -> Tensor:
    gradient = 0.5 * (value[..., 2:, :] - value[..., :-2, :])
    return F.pad(gradient, (0, 0, 1, 1), mode="replicate")


def edge_magnitude(image: Tensor) -> Tensor:
    gx = central_gradient_x(image)
    gy = central_gradient_y(image)
    return torch.sqrt(gx.square() + gy.square() + 1e-6).mean(1, keepdim=True)


def fixed_high_pass(value: Tensor, factor: int = 2) -> Tensor:
    """Apply the exact block-null-space projection ``H = I - U D``.

    ``D`` is non-overlapping average pooling and ``U`` repeats each pooled
    value back over the same block.  Consequently ``D(Hx)=0`` (including odd
    image sizes via a smaller final block), unlike a bilinear reconstruction
    for which ``D U`` is not the identity.
    """
    if factor < 2:
        raise ValueError("factor must be at least two.")
    low = F.avg_pool2d(
        value,
        kernel_size=factor,
        stride=factor,
        ceil_mode=True,
        count_include_pad=False,
    )
    reconstruction = low.repeat_interleave(factor, dim=-2).repeat_interleave(
        factor, dim=-1
    )
    reconstruction = reconstruction[..., : value.shape[-2], : value.shape[-1]]
    return value - reconstruction


class SourcePriorUpdate(nn.Module):
    """Bounded source-only correction for the data-term null space."""

    def __init__(
        self,
        context_channels: int,
        hidden_channels: int,
        max_delta: float,
    ) -> None:
        super().__init__()
        self.max_delta = float(max_delta)
        # source context + flow + previous auxiliary + source edge
        self.net = nn.Sequential(
            _conv_block(context_channels + 2 + 2 + 1, hidden_channels),
            _conv_block(hidden_channels, hidden_channels),
            _conv_block(hidden_channels, hidden_channels),
            nn.Conv2d(hidden_channels, 2, 3, padding=1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(
        self,
        source_context: Tensor,
        flow_yx: Tensor,
        previous_aux_yx: Tensor,
        source_edge: Tensor,
    ) -> Tensor:
        raw = self.net(
            torch.cat(
                (source_context, flow_yx, previous_aux_yx, source_edge), dim=1
            )
        )
        return self.max_delta * torch.tanh(raw / max(self.max_delta, 1e-6))


class SourceProximalUpdate(nn.Module):
    """Near-identity source-only proximal update at half resolution."""

    def __init__(
        self,
        context_channels: int,
        hidden_channels: int,
        max_delta: float,
    ) -> None:
        super().__init__()
        self.max_delta = float(max_delta)
        # source context + flow + previous auxiliary + coupling residual +
        # source edge + predicted boundary probability
        self.net = nn.Sequential(
            _conv_block(context_channels + 2 + 2 + 2 + 1 + 1, hidden_channels),
            _conv_block(hidden_channels, hidden_channels),
            nn.Conv2d(hidden_channels, 2, 3, padding=1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(
        self,
        flow_yx: Tensor,
        previous_aux_yx: Tensor,
        source_context: Tensor,
        source_edge: Tensor,
        boundary_probability: Tensor,
    ) -> Tensor:
        residual = flow_yx - previous_aux_yx
        raw = self.net(
            torch.cat(
                (
                    source_context,
                    flow_yx,
                    previous_aux_yx,
                    residual,
                    source_edge,
                    boundary_probability,
                ),
                dim=1,
            )
        )
        # Boundary probability permits independent corrections at motion
        # discontinuities instead of coupling across them.
        permission = 0.5 + 0.5 * boundary_probability
        delta = permission * self.max_delta * torch.tanh(
            raw / max(self.max_delta, 1e-6)
        )
        return flow_yx + delta


class BandSplitHQSRefiner(nn.Module):
    """Half-resolution analytic refinement plus high-pass full-res residual."""

    def __init__(
        self,
        context_channels: int,
        hidden_channels: int = 96,
        half_iterations: int = 2,
        data_beta: float = 0.10,
        half_step: float = 0.50,
        max_data_delta: float = 1.50,
        max_prior_delta: float = 1.00,
        max_prox_delta: float = 0.50,
        max_full_detail: float = 4.0,
        photometric_tau: float = 0.10,
        high_pass_factor: int = 2,
    ) -> None:
        super().__init__()
        self.half_iterations = int(half_iterations)
        self.data_beta = float(data_beta)
        self.half_step = float(half_step)
        self.max_data_delta = float(max_data_delta)
        self.max_full_detail = float(max_full_detail)
        self.photometric_tau = float(photometric_tau)
        self.high_pass_factor = int(high_pass_factor)

        self.source_prior = SourcePriorUpdate(
            context_channels, hidden_channels, max_prior_delta
        )
        self.source_prox = SourceProximalUpdate(
            context_channels, hidden_channels, max_prox_delta
        )

        # The data detail head may see a warped target residual, but only its
        # output is admitted through the predicted data gate.
        self.data_detail_head = nn.Sequential(
            _conv_block(context_channels + 2 + 1 + 1 + 1, hidden_channels),
            _conv_block(hidden_channels, hidden_channels),
            nn.Conv2d(hidden_channels, 2, 3, padding=1),
        )
        # The prior detail head is source-only.
        self.prior_detail_head = nn.Sequential(
            _conv_block(context_channels + 2 + 1, hidden_channels),
            _conv_block(hidden_channels, hidden_channels),
            nn.Conv2d(hidden_channels, 2, 3, padding=1),
        )
        for head in (self.data_detail_head[-1], self.prior_detail_head[-1]):
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    @staticmethod
    def _gray(image: Tensor) -> Tensor:
        if image.shape[1] == 1:
            return image
        if image.shape[1] != 3:
            return image.mean(1, keepdim=True)
        coefficients = image.new_tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1)
        return (image * coefficients).sum(1, keepdim=True)

    @staticmethod
    def _resize_scalar(value: Tensor, size: tuple[int, int]) -> Tensor:
        if value.shape[-2:] == size:
            return value
        return F.interpolate(value, size=size, mode="bilinear", align_corners=False)

    def _analytic_data_delta(
        self,
        image1: Tensor,
        image2: Tensor,
        flow_yx: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        image2_warped = backward_warp_yx(image2, flow_yx)
        ix = 0.5 * (
            central_gradient_x(image1)
            + backward_warp_yx(central_gradient_x(image2), flow_yx)
        )
        iy = 0.5 * (
            central_gradient_y(image1)
            + backward_warp_yx(central_gradient_y(image2), flow_yx)
        )
        temporal = image2_warped - image1
        denominator = ix.square() + iy.square() + self.data_beta
        du = -ix * temporal / denominator
        dv = -iy * temporal / denominator
        delta = torch.cat((dv, du), dim=1)
        delta = self.max_data_delta * torch.tanh(
            delta / max(self.max_data_delta, 1e-6)
        )
        return delta, temporal, image2_warped, denominator

    def forward(
        self,
        *,
        source_context: Tensor,
        source_image: Tensor,
        target_image: Tensor,
        initial_flow_yx: Tensor,
        data_gate: Optional[Tensor] = None,
        boundary_probability: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        half_size = source_context.shape[-2:]
        source_half = F.interpolate(
            self._gray(source_image), size=half_size, mode="bilinear", align_corners=True
        )
        target_half = F.interpolate(
            self._gray(target_image), size=half_size, mode="bilinear", align_corners=True
        )
        source_edge_half = edge_magnitude(source_half)
        flow_half = resize_flow_yx(initial_flow_yx, half_size)
        aux_half = flow_half.clone()

        if data_gate is None:
            base_gate = torch.ones_like(source_edge_half)
        else:
            base_gate = self._resize_scalar(data_gate, half_size).clamp(0.0, 1.0)
        if boundary_probability is None:
            boundary_half = source_edge_half / source_edge_half.amax(
                dim=(-2, -1), keepdim=True
            ).clamp_min(1e-6)
        else:
            boundary_half = self._resize_scalar(
                boundary_probability, half_size
            ).clamp(0.0, 1.0)

        temporal = torch.zeros_like(source_edge_half)
        gate_half = base_gate
        for _ in range(self.half_iterations):
            delta_data, temporal, _, _ = self._analytic_data_delta(
                source_half, target_half, flow_half
            )
            in_bounds = flow_in_bounds_mask(flow_half)
            photo_confidence = torch.exp(
                -temporal.abs() / max(self.photometric_tau, 1e-6)
            )
            gate_half = (base_gate * in_bounds * photo_confidence).clamp(0.0, 1.0)
            delta_prior = self.source_prior(
                source_context, flow_half, aux_half, source_edge_half
            )
            flow_half = flow_half + self.half_step * (
                gate_half * delta_data + (1.0 - gate_half) * delta_prior
            )
            aux_half = self.source_prox(
                flow_half,
                aux_half,
                source_context,
                source_edge_half,
                boundary_half,
            )

        full_size = source_image.shape[-2:]
        flow_full = resize_flow_yx(aux_half, full_size)
        context_full = F.interpolate(
            source_context, size=full_size, mode="bilinear", align_corners=False
        )
        source_gray = self._gray(source_image)
        target_gray = self._gray(target_image)
        target_warped = backward_warp_yx(target_gray, flow_full)
        temporal_full = target_warped - source_gray
        edge_full = edge_magnitude(source_gray)
        gate_full = self._resize_scalar(gate_half, full_size).clamp(0.0, 1.0)
        boundary_full = self._resize_scalar(boundary_half, full_size).clamp(0.0, 1.0)

        data_raw = self.data_detail_head(
            torch.cat(
                (context_full, flow_full, temporal_full, edge_full, gate_full), dim=1
            )
        )
        prior_raw = self.prior_detail_head(
            torch.cat((context_full, flow_full, edge_full), dim=1)
        )
        raw_detail = gate_full * data_raw + (1.0 - gate_full) * prior_raw
        bounded_detail = self.max_full_detail * torch.tanh(
            raw_detail / max(self.max_full_detail, 1e-6)
        )
        boundary_permission = 0.5 + 0.5 * boundary_full
        # Projection must be the final operation on the residual.  A nonlinear
        # clamp or spatial gate applied after H would move it back out of the
        # null space and allow the detail head to rewrite pooled coarse flow.
        # H can double an element-wise bound (value minus block mean), hence
        # the factor 0.5 keeps |detail| <= max_full_detail.
        high_detail = 0.5 * fixed_high_pass(
            boundary_permission * bounded_detail,
            factor=self.high_pass_factor,
        )
        final_flow = flow_full + high_detail

        return {
            "flow_yx": final_flow,
            "flow_half_yx": flow_half,
            "aux_half_yx": aux_half,
            "data_gate_half": gate_half,
            "boundary_half": boundary_half,
            "detail_delta_yx": high_detail,
            "photometric_residual_half": temporal,
        }
