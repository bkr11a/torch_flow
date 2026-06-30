# hqs_pytorch/customML/customModels/pgma.py

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class PhysicsGatedMatchingAttention(nn.Module):
    """
    PGMA: Physics-Gated Matching Attention.

    Produces:
      global_flow_yx: [B, 2, H, W]
      global_conf:    [B, 1, H, W]
      gate_logits:    [B, 3, H, W] for [HQS, global_match, proximal_pull]
    """

    def __init__(
        self,
        feature_dim: int,
        gate_hidden: int = 64,
        temperature: float = 0.07,
        topk: int = 64,
        use_topk: bool = True,
    ):
        super().__init__()
        self.temperature = temperature
        self.topk = topk
        self.use_topk = use_topk

        self.q_proj = nn.Conv2d(feature_dim, feature_dim, 1)
        self.k_proj = nn.Conv2d(feature_dim, feature_dim, 1)

        # Gate inputs:
        # hqs_delta_norm      1
        # global_delta_norm   1
        # prox_delta_norm     1
        # hqs_resid_norm      1
        # local_conf          1
        # global_conf         1
        # validity            1
        # flow_norm           1
        # iteration scalar    1
        # total               9
        self.gate = nn.Sequential(
            nn.Conv2d(9, gate_hidden, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(gate_hidden, gate_hidden, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(gate_hidden, 3, 3, padding=1),
        )

        # Start close to the existing behaviour: mostly HQS/match branch,
        # small global attention, small proximal pull.
        nn.init.zeros_(self.gate[-1].weight)
        with torch.no_grad():
            self.gate[-1].bias[:] = torch.tensor([1.5, -1.0, -1.0])

    @staticmethod
    def coords_grid_yx(batch: int, height: int, width: int, device, dtype):
        ys, xs = torch.meshgrid(
            torch.arange(height, device=device, dtype=dtype),
            torch.arange(width, device=device, dtype=dtype),
            indexing="ij",
        )
        coords = torch.stack([ys, xs], dim=0)  # [2, H, W]
        return coords.unsqueeze(0).repeat(batch, 1, 1, 1)

    def global_match(self, f1: torch.Tensor, f2: torch.Tensor):
        """
        Dense global matching via feature cross-attention.

        f1, f2: [B, C, H, W]
        returns:
          flow_yx: [B, 2, H, W]
          conf:    [B, 1, H, W]
        """
        b, c, h, w = f1.shape
        n = h * w

        q = self.q_proj(f1)
        k = self.k_proj(f2)

        q = F.normalize(q.flatten(2).transpose(1, 2), dim=-1)  # [B, N, C]
        k = F.normalize(k.flatten(2), dim=1)                   # [B, C, N]

        logits = torch.bmm(q, k) / self.temperature             # [B, N, N]

        coords = self.coords_grid_yx(b, h, w, f1.device, f1.dtype)
        coords_flat = coords.flatten(2).transpose(1, 2)         # [B, N, 2]

        if self.use_topk and self.topk < n:
            vals, idx = torch.topk(logits, k=self.topk, dim=-1)  # [B, N, K]
            prob = F.softmax(vals, dim=-1)

            coords_exp = coords_flat[:, None, :, :].expand(b, n, n, 2)
            idx_exp = idx.unsqueeze(-1).expand(-1, -1, -1, 2)
            coords_k = torch.gather(coords_exp, 2, idx_exp)      # [B, N, K, 2]

            match_coords = (prob.unsqueeze(-1) * coords_k).sum(dim=2)
            conf = prob.max(dim=-1, keepdim=True).values
        else:
            prob = F.softmax(logits, dim=-1)
            match_coords = torch.bmm(prob, coords_flat)
            conf = prob.max(dim=-1, keepdim=True).values

        src_coords = coords_flat
        flow = match_coords - src_coords                         # [B, N, 2]
        flow = flow.transpose(1, 2).view(b, 2, h, w)              # [B, 2, H, W]
        conf = conf.transpose(1, 2).view(b, 1, h, w)

        return flow, conf

    def forward(
        self,
        f1: torch.Tensor,
        f2: torch.Tensor,
        flow_yx: torch.Tensor,
        delta_hqs: torch.Tensor,
        aux_yx: torch.Tensor,
        hqs_resid: torch.Tensor,
        local_conf: torch.Tensor,
        validity: torch.Tensor,
        iter_frac: torch.Tensor,
    ):
        global_flow_yx, global_conf = self.global_match(f1, f2)

        # Bring global proposal into current residual-update form.
        delta_global = global_flow_yx - flow_yx
        delta_prox = aux_yx - flow_yx

        def norm1(x):
            return x.abs().mean(dim=1, keepdim=True)

        if iter_frac.shape[-2:] != flow_yx.shape[-2:]:
            iter_frac = F.interpolate(iter_frac, size=flow_yx.shape[-2:], mode="nearest")

        gate_in = torch.cat(
            [
                norm1(delta_hqs),
                norm1(delta_global),
                norm1(delta_prox),
                norm1(hqs_resid),
                local_conf,
                global_conf,
                validity,
                norm1(flow_yx),
                iter_frac,
            ],
            dim=1,
        )

        gate_logits = self.gate(gate_in)
        gates = F.softmax(gate_logits, dim=1)

        alpha_hqs = gates[:, 0:1]
        beta_global = gates[:, 1:2]
        gamma_prox = gates[:, 2:3]

        delta = (
            alpha_hqs * delta_hqs
            + beta_global * delta_global
            + gamma_prox * delta_prox
        )

        return {
            "delta": delta,
            "global_flow_yx": global_flow_yx,
            "delta_global": delta_global,
            "global_conf": global_conf,
            "gates": gates,
        }