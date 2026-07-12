"""Low-memory correlation statistics used by factorised reliability."""
from __future__ import annotations

import torch
from torch import Tensor


def correlation_statistics(
    correlation: Tensor,
    *,
    candidate_dim: int = 1,
    temperature: float = 1.0,
    eps: float = 1e-8,
) -> tuple[Tensor, Tensor, Tensor]:
    """Return normalised entropy, peak probability, and top-1/top-2 margin.

    ``correlation`` may be [B,N,H,W] or any tensor in which ``candidate_dim``
    indexes candidate matches. Returned tensors retain singleton candidate_dim.
    """
    if temperature <= 0:
        raise ValueError("temperature must be positive.")
    n = correlation.shape[candidate_dim]
    if n < 2:
        raise ValueError("At least two candidate correlations are required.")

    probabilities = torch.softmax(correlation / temperature, dim=candidate_dim)
    log_probabilities = torch.log(probabilities.clamp_min(eps))
    entropy = -(probabilities * log_probabilities).sum(
        dim=candidate_dim, keepdim=True
    )
    entropy = entropy / torch.log(
        torch.tensor(float(n), device=correlation.device, dtype=correlation.dtype)
    )
    top2 = probabilities.topk(k=2, dim=candidate_dim).values
    peak = top2.select(candidate_dim, 0).unsqueeze(candidate_dim)
    second = top2.select(candidate_dim, 1).unsqueeze(candidate_dim)
    margin = peak - second
    return entropy, peak, margin
