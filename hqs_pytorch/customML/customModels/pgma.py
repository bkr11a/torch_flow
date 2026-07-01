# hqs_pytorch/customML/customModels/pgma.py

from __future__ import annotations

import math
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class PhysicsGatedMatchingAttention(nn.Module):
    """
    PGMA: Physics-Gated Matching Attention.

    Provides:
      1. GMFlow-style global correspondence proposal:
             w_global(x) = E_y[y - x | x]

      2. Optional hard / straight-through matching to avoid soft ghost averages.

      3. Physics gate over:
             delta_hqs      : local HQS / recurrent update
             delta_global   : global matching correction
             delta_prox     : proximal pull q_k - w_k

    Flow convention:
      internal flow_yx = [dy, dx].
    """

    def __init__(
        self,
        feature_dim: int,
        gate_hidden: int = 64,
        temperature: float = 0.07,
        topk: int = 64,
        use_topk: bool = True,
        match_mode: str = "soft",  # "soft", "hard", or "straight_through"
        use_feature_enhancer: bool = True,
        gate_init_hqs: float = 1.5,
        gate_init_global: float = -1.0,
        gate_init_prox: float = -1.0,
    ):
        super().__init__()

        if match_mode not in {"soft", "hard", "straight_through"}:
            raise ValueError(
                f"match_mode must be 'soft', 'hard', or 'straight_through', got {match_mode}"
            )

        self.temperature = float(temperature)
        self.topk = int(topk)
        self.use_topk = bool(use_topk)
        self.match_mode = match_mode

        if use_feature_enhancer:
            self.feature_enhancer = nn.Sequential(
                nn.Conv2d(feature_dim, feature_dim, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(feature_dim, feature_dim, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
            )
        else:
            self.feature_enhancer = nn.Identity()

        self.q_proj = nn.Conv2d(feature_dim, feature_dim, kernel_size=1, bias=False)
        self.k_proj = nn.Conv2d(feature_dim, feature_dim, kernel_size=1, bias=False)

        # Gate inputs:
        # |delta_hqs|       1
        # |delta_global|    1
        # |delta_prox|      1
        # |hqs_resid|       1
        # local_conf        1
        # global_conf       1
        # global_entropy    1
        # global_margin     1
        # validity          1
        # |flow_yx|         1
        # iter_frac         1
        # total             11
        self.gate = nn.Sequential(
            nn.Conv2d(11, gate_hidden, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(gate_hidden, gate_hidden, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(gate_hidden, 3, kernel_size=3, padding=1),
        )

        # Conservative default: preserve existing HQS behaviour initially.
        nn.init.zeros_(self.gate[-1].weight)
        with torch.no_grad():
            self.gate[-1].bias[:] = torch.tensor(
                [gate_init_hqs, gate_init_global, gate_init_prox],
                dtype=self.gate[-1].bias.dtype,
                device=self.gate[-1].bias.device,
            )

    @staticmethod
    def coords_grid_yx(batch: int, height: int, width: int, device, dtype):
        ys, xs = torch.meshgrid(
            torch.arange(height, device=device, dtype=dtype),
            torch.arange(width, device=device, dtype=dtype),
            indexing="ij",
        )
        coords = torch.stack([ys, xs], dim=0)  # [2, H, W]
        return coords.unsqueeze(0).expand(batch, -1, -1, -1).contiguous()

    @staticmethod
    def _norm1(x: torch.Tensor) -> torch.Tensor:
        return x.abs().mean(dim=1, keepdim=True)

    def global_match(
        self,
        f1: torch.Tensor,
        f2: torch.Tensor,
        target_valid: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Dense global matching via feature cross-attention.

        Args:
          f1, f2:
              [B, C, H, W] features at the same resolution.
          target_valid:
              Optional [B, 1, H, W] target-frame validity mask.
              Invalid target pixels are excluded from the softmax.

        Returns:
          dict containing:
            flow_yx       [B, 2, H, W]
            hard_flow_yx  [B, 2, H, W]
            conf          [B, 1, H, W]
            entropy       [B, 1, H, W]
            margin        [B, 1, H, W]
            top1_index    [B, 1, H, W]
        """
        if f1.shape != f2.shape:
            raise ValueError(f"f1 and f2 must have identical shape, got {f1.shape}, {f2.shape}")

        b, c, h, w = f1.shape
        n = h * w

        f1e = self.feature_enhancer(f1)
        f2e = self.feature_enhancer(f2)

        q = self.q_proj(f1e)
        k = self.k_proj(f2e)

        q = F.normalize(q.flatten(2).transpose(1, 2), dim=-1)  # [B, N, C]
        k = F.normalize(k.flatten(2), dim=1)                   # [B, C, N]

        logits = torch.bmm(q, k) / max(self.temperature, 1e-6)  # [B, N, N]

        if target_valid is not None:
            if target_valid.shape[-2:] != (h, w):
                target_valid = F.interpolate(
                    target_valid.float(), size=(h, w), mode="nearest"
                )
            valid_flat = target_valid.flatten(2).bool()         # [B, 1, N]
            logits = logits.masked_fill(~valid_flat, -1e4)

        coords = self.coords_grid_yx(b, h, w, f1.device, f1.dtype)
        coords_flat = coords.flatten(2).transpose(1, 2)         # [B, N, 2]

        # Hard argmax correspondence.
        hard_idx = logits.argmax(dim=-1)                        # [B, N]
        hard_coords = torch.gather(
            coords_flat,
            dim=1,
            index=hard_idx.unsqueeze(-1).expand(-1, -1, 2),
        )
        hard_flow = hard_coords - coords_flat                   # [B, N, 2]
        hard_flow_yx = hard_flow.transpose(1, 2).view(b, 2, h, w)

        if self.use_topk and self.topk > 0 and self.topk < n:
            vals, idx = torch.topk(logits, k=self.topk, dim=-1)  # [B, N, K]
            prob = F.softmax(vals, dim=-1)

            coords_all = coords_flat[:, None, :, :].expand(b, n, n, 2)
            coords_k = torch.gather(
                coords_all,
                dim=2,
                index=idx.unsqueeze(-1).expand(-1, -1, -1, 2),
            )

            soft_coords = (prob.unsqueeze(-1) * coords_k).sum(dim=2)

            conf = prob.max(dim=-1, keepdim=True).values

            if self.topk >= 2:
                top2 = torch.topk(prob, k=2, dim=-1).values
                margin = top2[:, :, 0:1] - top2[:, :, 1:2]
            else:
                margin = conf

            entropy = -(prob * (prob + 1e-8).log()).sum(dim=-1, keepdim=True)
            entropy = entropy / math.log(float(self.topk))

            # Top-1 index in the full target grid.
            top_pos = prob.argmax(dim=-1, keepdim=True)
            top1_index = torch.gather(idx, dim=-1, index=top_pos)
        else:
            prob = F.softmax(logits, dim=-1)
            soft_coords = torch.bmm(prob, coords_flat)

            top2 = torch.topk(prob, k=2, dim=-1)
            conf = top2.values[:, :, 0:1]
            margin = top2.values[:, :, 0:1] - top2.values[:, :, 1:2]
            top1_index = top2.indices[:, :, 0:1]

            entropy = -(prob * (prob + 1e-8).log()).sum(dim=-1, keepdim=True)
            entropy = entropy / math.log(float(n))

        if self.match_mode == "soft":
            match_coords = soft_coords
        elif self.match_mode == "hard":
            match_coords = hard_coords
        else:
            # Forward pass uses hard correspondence, gradient follows soft path.
            match_coords = soft_coords + (hard_coords - soft_coords).detach()

        flow = match_coords - coords_flat
        flow_yx = flow.transpose(1, 2).view(b, 2, h, w)

        return {
            "flow_yx": flow_yx,
            "hard_flow_yx": hard_flow_yx,
            "conf": conf.transpose(1, 2).view(b, 1, h, w),
            "entropy": entropy.transpose(1, 2).view(b, 1, h, w),
            "margin": margin.transpose(1, 2).view(b, 1, h, w),
            "top1_index": top1_index.transpose(1, 2).view(b, 1, h, w),
        }

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
        target_valid: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        gm = self.global_match(f1, f2, target_valid=target_valid)

        global_flow_yx = gm["flow_yx"]
        global_conf = gm["conf"]
        global_entropy = gm["entropy"]
        global_margin = gm["margin"]

        delta_global = global_flow_yx - flow_yx
        delta_prox = aux_yx - flow_yx

        if iter_frac.shape[-2:] != flow_yx.shape[-2:]:
            iter_frac = F.interpolate(iter_frac, size=flow_yx.shape[-2:], mode="nearest")

        gate_in = torch.cat(
            [
                self._norm1(delta_hqs),
                self._norm1(delta_global),
                self._norm1(delta_prox),
                self._norm1(hqs_resid),
                local_conf,
                global_conf,
                global_entropy,
                global_margin,
                validity,
                self._norm1(flow_yx),
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
            "hard_global_flow_yx": gm["hard_flow_yx"],
            "delta_global": delta_global,
            "global_conf": global_conf,
            "global_entropy": global_entropy,
            "global_margin": global_margin,
            "global_top1_index": gm["top1_index"],
            "gates": gates,
            "alpha_hqs": alpha_hqs,
            "beta_global": beta_global,
            "gamma_prox": gamma_prox,
        }