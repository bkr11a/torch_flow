"""Single unrolled HQS stage.

One stage corresponds to one iteration of the alternating minimisation:

    u^{k+1} = DataNet( corr(f1, f2, u^k), ctx, u^k, v^k, mu^k )
    v^{k+1} = ProxNet( u^{k+1}, sigma^k )    sigma^k = sqrt(lambda^k / mu^k)

where mu^k and log_lambda^k are *learnable per-stage scalars* so the network
can adapt the penalty schedule end-to-end.
"""
from __future__ import annotations

import math
import torch
import torch.nn as nn
from typing import Optional, Tuple

from .update_net import DataUpdateNet
from .reg_net import build_prox_net, IdentityProxNet
from .correlation import CorrBlock, LocalCorrBlock


PenaltyTuple = Tuple[torch.Tensor, torch.Tensor]   # (mu, sigma)


class HQSStage(nn.Module):
    """
    One unrolled HQS iteration.

    Args:
        update_net:  Shared or per-stage DataUpdateNet.
        prox_net:    Shared or per-stage proximal operator.
        share_weights: If True, this stage does NOT own update_net/prox_net
                       (they are shared from the parent model).  If False,
                       the stage owns its own copies (independent weights).
        init_mu_log: Initial value for the learnable log-mu scalar (default 0).
        init_lambda_log: Initial value for the learnable log-lambda scalar.
    """

    def __init__(
        self,
        update_net: DataUpdateNet,
        prox_net: nn.Module,
        share_weights: bool = True,
        init_mu_log: float = 0.0,
        init_lambda_log: float = 0.0,
    ) -> None:
        super().__init__()

        # The *caller* passes network references; sharing is controlled there.
        self.update_net = update_net
        self.prox_net = prox_net

        # Learnable per-stage penalty schedule parameters
        # mu^k = exp(log_mu)   (positive by construction)
        self.log_mu     = nn.Parameter(torch.tensor(init_mu_log))
        # sigma^k = exp(log_sigma) = sqrt(lambda^k / mu^k)
        self.log_sigma  = nn.Parameter(torch.tensor(init_lambda_log))

    def get_penalty(self, batch_size: int, device: torch.device) -> PenaltyTuple:
        """Return (mu, sigma) broadcast to (B,)."""
        mu    = self.log_mu.exp().expand(batch_size).to(device)
        sigma = self.log_sigma.exp().expand(batch_size).to(device)
        return mu, sigma

    def forward(
        self,
        fmap1: torch.Tensor,          # (B, C, H, W)  fixed feature map
        fmap2: torch.Tensor,          # (B, C, H, W)  fixed feature map
        corr_block,                   # CorrBlock or LocalCorrBlock (already initialised)
        context: torch.Tensor,        # (B, ctx_ch, H, W)
        flow_u: torch.Tensor,         # (B, 2, H, W)   u^k
        flow_v: torch.Tensor,         # (B, 2, H, W)   v^k
        hidden: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        One HQS iteration.

        Returns:
            u_new    – (B, 2, H, W)  u^{k+1}
            v_new    – (B, 2, H, W)  v^{k+1}
            hidden   – updated GRU hidden state
            up_mask  – convex upsample mask (B, 9*scale², H, W) or None
        """
        B = flow_u.shape[0]
        device = flow_u.device

        mu, sigma = self.get_penalty(B, device)

        # ---- Data subproblem ------------------------------------------------
        # Index correlation volume at current flow estimate
        if isinstance(corr_block, CorrBlock):
            corr_feat = corr_block.index(
                # coords_grid + flow
                _coords_plus_flow(flow_u)
            )
        else:  # LocalCorrBlock
            corr_feat = corr_block(fmap1, fmap2, flow_u)

        delta_u, hidden, up_mask = self.update_net(
            corr_feat, context, flow_u, flow_v, mu, hidden
        )
        u_new = flow_u + delta_u

        # ---- Regularisation subproblem (proximal step) ----------------------
        if isinstance(self.prox_net, IdentityProxNet):
            v_new = u_new
        else:
            v_new = self.prox_net(u_new, sigma)

        return u_new, v_new, hidden, up_mask


def _coords_plus_flow(flow: torch.Tensor) -> torch.Tensor:
    """Return absolute sampling coordinates: base grid + flow."""
    from .warp import coords_grid
    B, _, H, W = flow.shape
    return coords_grid(B, H, W, device=flow.device) + flow
