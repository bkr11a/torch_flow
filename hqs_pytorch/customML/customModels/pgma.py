# hqs_pytorch/customML/customModels/pgma.py

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

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
        query_chunk_size: int = 512,
        position_scale: float = 0.05,
        confidence_floor: float = 0.05,
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
        self.query_chunk_size = max(1, int(query_chunk_size))
        self.position_scale = float(position_scale)
        self.confidence_floor = float(confidence_floor)
        if not 0.0 <= self.confidence_floor < 1.0:
            raise ValueError("confidence_floor must be in [0, 1).")

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

    @staticmethod
    def _position_encoding(
        batch: int,
        channels: int,
        height: int,
        width: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Fixed 2-D sine/cosine encoding with shape ``[B,C,H,W]``."""
        bands = max(1, math.ceil(channels / 4))
        y = torch.linspace(-1.0, 1.0, height, device=device, dtype=dtype)
        x = torch.linspace(-1.0, 1.0, width, device=device, dtype=dtype)
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        frequency = torch.linspace(1.0, 8.0, bands, device=device, dtype=dtype)
        frequency = frequency.view(bands, 1, 1) * math.pi
        pos = torch.cat(
            (
                torch.sin(frequency * yy),
                torch.cos(frequency * yy),
                torch.sin(frequency * xx),
                torch.cos(frequency * xx),
            ),
            dim=0,
        )[:channels]
        return pos.unsqueeze(0).expand(batch, -1, -1, -1)

    def _match_direction(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        coords_flat: torch.Tensor,
        target_valid: Optional[torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Full-distribution matching, evaluated in query chunks.

        Chunking bounds peak memory without truncating the probability
        distribution.  Entropy, peak probability and margin therefore remain
        calibrated against every target candidate rather than a renormalised
        top-k subset.
        """
        b, n, _ = query.shape
        valid_flat: Optional[torch.Tensor] = None
        if target_valid is not None:
            valid_flat = target_valid.flatten(1).bool()
            # A completely empty target mask must not create a uniform softmax
            # over identical -inf-style logits.
            empty = ~valid_flat.any(dim=1)
            if empty.any():
                valid_flat = valid_flat.clone()
                valid_flat[empty] = True

        soft_coords_parts = []
        hard_coords_parts = []
        peak_parts = []
        entropy_parts = []
        margin_parts = []
        index_parts = []
        key_t = key.transpose(1, 2)
        entropy_denominator = math.log(float(max(n, 2)))

        for start in range(0, n, self.query_chunk_size):
            stop = min(n, start + self.query_chunk_size)
            logits = torch.bmm(query[:, start:stop], key_t)
            logits = logits / max(self.temperature, 1e-6)
            if valid_flat is not None:
                logits = logits.masked_fill(~valid_flat[:, None, :], -1e4)

            probability = torch.softmax(logits, dim=-1)
            soft_coords_parts.append(torch.bmm(probability, coords_flat))

            top2 = torch.topk(probability, k=min(2, n), dim=-1)
            top1_index = top2.indices[..., 0]
            hard_coords_parts.append(
                torch.gather(
                    coords_flat,
                    dim=1,
                    index=top1_index.unsqueeze(-1).expand(-1, -1, 2),
                )
            )
            peak = top2.values[..., 0:1]
            if n > 1:
                margin = peak - top2.values[..., 1:2]
            else:
                margin = peak
            entropy = -(
                probability * torch.log(probability.clamp_min(1e-8))
            ).sum(dim=-1, keepdim=True) / entropy_denominator

            peak_parts.append(peak)
            margin_parts.append(margin)
            entropy_parts.append(entropy.clamp(0.0, 1.0))
            index_parts.append(top1_index)

        soft_coords = torch.cat(soft_coords_parts, dim=1)
        hard_coords = torch.cat(hard_coords_parts, dim=1)
        peak = torch.cat(peak_parts, dim=1)
        entropy = torch.cat(entropy_parts, dim=1)
        margin = torch.cat(margin_parts, dim=1)
        top1_index = torch.cat(index_parts, dim=1)

        if self.match_mode == "soft":
            selected_coords = soft_coords
        elif self.match_mode == "hard":
            selected_coords = hard_coords
        else:
            selected_coords = soft_coords + (hard_coords - soft_coords).detach()

        return {
            "flow": selected_coords - coords_flat,
            "hard_flow": hard_coords - coords_flat,
            "peak": peak,
            "entropy": entropy,
            "margin": margin,
            "top1_index": top1_index,
        }

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

        position = self._position_encoding(
            b, c, h, w, device=f1.device, dtype=f1.dtype
        )
        q1 = self.q_proj(f1e) + self.position_scale * position
        k2 = self.k_proj(f2e) + self.position_scale * position
        q2 = self.q_proj(f2e) + self.position_scale * position
        k1 = self.k_proj(f1e) + self.position_scale * position

        q1 = F.normalize(q1.flatten(2).transpose(1, 2), dim=-1)
        k2 = F.normalize(k2.flatten(2).transpose(1, 2), dim=-1)
        q2 = F.normalize(q2.flatten(2).transpose(1, 2), dim=-1)
        k1 = F.normalize(k1.flatten(2).transpose(1, 2), dim=-1)

        if target_valid is not None and target_valid.shape[-2:] != (h, w):
            target_valid = F.interpolate(
                target_valid.float(), size=(h, w), mode="nearest"
            )

        coords = self.coords_grid_yx(b, h, w, f1.device, f1.dtype)
        coords_flat = coords.flatten(2).transpose(1, 2)         # [B, N, 2]

        forward = self._match_direction(q1, k2, coords_flat, target_valid)
        reverse = self._match_direction(q2, k1, coords_flat, None)

        query_index = torch.arange(n, device=f1.device).view(1, n).expand(b, -1)
        reverse_at_forward = torch.gather(
            reverse["top1_index"], dim=1, index=forward["top1_index"]
        )
        mutual = (reverse_at_forward == query_index).to(f1.dtype).unsqueeze(-1)
        forward_conf = torch.sqrt(
            forward["peak"].clamp_min(0.0)
            * (1.0 - forward["entropy"]).clamp_min(0.0)
        )
        forward_conf = forward_conf * (
            self.confidence_floor + (1.0 - self.confidence_floor) * mutual
        )

        forward_at_reverse = torch.gather(
            forward["top1_index"], dim=1, index=reverse["top1_index"]
        )
        reverse_mutual = (forward_at_reverse == query_index).to(f1.dtype).unsqueeze(-1)
        reverse_conf = torch.sqrt(
            reverse["peak"].clamp_min(0.0)
            * (1.0 - reverse["entropy"]).clamp_min(0.0)
        )
        reverse_conf = reverse_conf * (
            self.confidence_floor
            + (1.0 - self.confidence_floor) * reverse_mutual
        )

        def chw(value: torch.Tensor, channels: int) -> torch.Tensor:
            return value.transpose(1, 2).reshape(b, channels, h, w)

        return {
            "flow_yx": chw(forward["flow"], 2),
            "hard_flow_yx": chw(forward["hard_flow"], 2),
            "conf": chw(forward_conf, 1),
            "entropy": chw(forward["entropy"], 1),
            "margin": chw(forward["margin"], 1),
            "mutual": chw(mutual, 1),
            "top1_index": chw(forward["top1_index"].unsqueeze(-1), 1),
            "reverse_flow_yx": chw(reverse["flow"], 2),
            "reverse_hard_flow_yx": chw(reverse["hard_flow"], 2),
            "reverse_conf": chw(reverse_conf, 1),
            "reverse_entropy": chw(reverse["entropy"], 1),
            "reverse_margin": chw(reverse["margin"], 1),
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
        global_match_result: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        gm = (
            global_match_result
            if global_match_result is not None
            else self.global_match(f1, f2, target_valid=target_valid)
        )

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
