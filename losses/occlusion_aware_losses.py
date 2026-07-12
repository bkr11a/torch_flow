"""Losses for factorised reliability and bidirectional HQS optical flow."""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor

try:
    from hqs_pytorch.customML.customModels.factorised_reliability import ReliabilityState
    from hqs_pytorch.customML.customModels.occlusion_geometry import (
        backward_warp_yx,
        flow_in_bounds_mask,
    )
except ImportError as exc:  # pragma: no cover
    raise ImportError("Run from the torch_flow repository root.") from exc


def _masked_mean(value: Tensor, mask: Tensor, eps: float = 1e-6) -> Tensor:
    mask = mask.to(value.dtype)
    return (value * mask).sum() / mask.sum().clamp_min(eps)


def robust_l1(residual: Tensor, epsilon: float = 1e-3) -> Tensor:
    return torch.sqrt(residual.square() + epsilon * epsilon)


def visible_outlier_mixture_nll(
    residual: Tensor,
    reliability: ReliabilityState,
    *,
    valid: Tensor | None = None,
    outlier_scale: float = 20.0,
    minimum_probability: float = 1e-8,
) -> Tensor:
    """Mixture NLL that cannot be minimised by setting visibility to zero.

    The visible component is Laplace with learned scale. The outlier component
    is a broad Laplace. The residual may contain one or multiple channels.
    """
    sigma = reliability.sigma_data.clamp_min(1e-4)
    if sigma.shape[1] == 1 and residual.shape[1] != 1:
        sigma = sigma.expand(-1, residual.shape[1], -1, -1)
    pi = reliability.p_visible * reliability.p_match
    if pi.shape[1] == 1 and residual.shape[1] != 1:
        pi = pi.expand(-1, residual.shape[1], -1, -1)

    visible_log_prob = -residual.abs() / sigma - torch.log(2.0 * sigma)
    outlier_sigma = residual.new_tensor(float(outlier_scale))
    outlier_log_prob = -residual.abs() / outlier_sigma - torch.log(
        2.0 * outlier_sigma
    )

    log_pi = torch.log(pi.clamp_min(minimum_probability))
    log_one_minus_pi = torch.log((1.0 - pi).clamp_min(minimum_probability))
    log_prob = torch.logsumexp(
        torch.stack(
            (log_pi + visible_log_prob, log_one_minus_pi + outlier_log_prob),
            dim=0,
        ),
        dim=0,
    )
    nll = -log_prob.mean(dim=1, keepdim=True)
    if valid is None:
        return nll.mean()
    return _masked_mean(nll, valid)


def visibility_supervision_loss(
    reliability: ReliabilityState,
    visibility_target: Tensor,
    *,
    valid: Tensor | None = None,
    positive_weight: float | None = None,
) -> Tensor:
    """BCE target loss. The target is loss-only and never model input."""
    target = visibility_target.to(reliability.visibility_logits.dtype)
    pos_weight = (
        target.new_tensor(float(positive_weight))
        if positive_weight is not None
        else None
    )
    loss = F.binary_cross_entropy_with_logits(
        reliability.visibility_logits,
        target,
        reduction="none",
        pos_weight=pos_weight,
    )
    if valid is None:
        return loss.mean()
    return _masked_mean(loss, valid)


def matchability_supervision_loss(
    reliability: ReliabilityState,
    match_target: Tensor,
    *,
    valid: Tensor | None = None,
) -> Tensor:
    loss = F.binary_cross_entropy_with_logits(
        reliability.match_logits,
        match_target.to(reliability.match_logits.dtype),
        reduction="none",
    )
    if valid is None:
        return loss.mean()
    return _masked_mean(loss, valid)


def forward_backward_cycle_loss(
    flow_ab: Tensor,
    flow_ba: Tensor,
    *,
    visibility: Tensor | None = None,
    detach_reverse: bool = False,
    epsilon: float = 1e-3,
) -> Tensor:
    reverse = flow_ba.detach() if detach_reverse else flow_ba
    residual = flow_ab + backward_warp_yx(reverse, flow_ab)
    per_pixel = robust_l1(residual, epsilon=epsilon).mean(dim=1, keepdim=True)
    valid = flow_in_bounds_mask(flow_ab)
    if visibility is not None:
        valid = valid * visibility
    return _masked_mean(per_pixel, valid)


def reliability_temporal_loss(
    current: ReliabilityState,
    previous: ReliabilityState,
    *,
    boundary_discount: bool = True,
) -> Tensor:
    weight = 1.0 - current.p_boundary if boundary_discount else 1.0
    return (
        weight * (current.p_visible - previous.p_visible).abs()
    ).mean() + (
        weight * (current.p_match - previous.p_match).abs()
    ).mean()


def visibility_prior_loss(
    reliability: ReliabilityState,
    *,
    expected_visible_fraction: float,
    tolerance: float = 0.05,
) -> Tensor:
    """Soft anti-collapse prior for early training only."""
    mean = reliability.p_visible.mean()
    lower = expected_visible_fraction - tolerance
    upper = expected_visible_fraction + tolerance
    return F.relu(lower - mean).square() + F.relu(mean - upper).square()


@dataclass
class OcclusionAwareLossWeights:
    mixture: float = 0.02
    visibility: float = 0.01
    matchability: float = 0.005
    cycle: float = 0.01
    temporal: float = 0.001
    visibility_prior: float = 0.001
