# hqs_pytorch/customML/customModels/sota_addons.py

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------------------------------------------------------
# Utility functions
# -----------------------------------------------------------------------------

def coords_grid_yx(batch: int, height: int, width: int, device, dtype):
    ys, xs = torch.meshgrid(
        torch.arange(height, device=device, dtype=dtype),
        torch.arange(width, device=device, dtype=dtype),
        indexing="ij",
    )
    coords = torch.stack([ys, xs], dim=0)
    return coords.unsqueeze(0).expand(batch, -1, -1, -1).contiguous()


def resize_flow_yx(flow_yx: torch.Tensor, size_hw: Tuple[int, int]) -> torch.Tensor:
    """Resize [dy, dx] flow and rescale vector magnitudes."""
    h0, w0 = flow_yx.shape[-2:]
    h1, w1 = size_hw
    out = F.interpolate(flow_yx, size=size_hw, mode="bilinear", align_corners=True)
    out = out.clone()
    out[:, 0] *= float(h1) / float(h0)
    out[:, 1] *= float(w1) / float(w0)
    return out


def image_edge_magnitude(image: torch.Tensor, size_hw: Optional[Tuple[int, int]] = None) -> torch.Tensor:
    """
    Sobel-like edge magnitude from image.

    Args:
        image: [B, 3, H, W] or [B, 1, H, W]
    Returns:
        edge: [B, 1, h, w]
    """
    if image.shape[1] > 1:
        gray = image.mean(dim=1, keepdim=True)
    else:
        gray = image

    if size_hw is not None and gray.shape[-2:] != size_hw:
        gray = F.interpolate(gray, size=size_hw, mode="bilinear", align_corners=True)

    dx = gray[..., :, 1:] - gray[..., :, :-1]
    dy = gray[..., 1:, :] - gray[..., :-1, :]

    dx = F.pad(dx, (0, 1, 0, 0))
    dy = F.pad(dy, (0, 0, 0, 1))

    edge = torch.sqrt(dx.square() + dy.square() + 1e-8)
    return edge


def flow_gradients(flow: torch.Tensor):
    """Returns finite differences dx, dy for flow [B,2,H,W]."""
    dx = flow[..., :, 1:] - flow[..., :, :-1]
    dy = flow[..., 1:, :] - flow[..., :-1, :]
    dx = F.pad(dx, (0, 1, 0, 0))
    dy = F.pad(dy, (0, 0, 0, 1))
    return dx, dy


def flow_boundary_mask(flow_gt: torch.Tensor, threshold: float = 1.0) -> torch.Tensor:
    """
    Boundary mask from GT flow gradients.

    Args:
        flow_gt: [B,2,H,W], usually [dx,dy].
    Returns:
        [B,1,H,W] float mask.
    """
    dx, dy = flow_gradients(flow_gt)
    mag = torch.sqrt(dx.square().sum(dim=1, keepdim=True) + dy.square().sum(dim=1, keepdim=True) + 1e-8)
    return (mag > threshold).float()


# -----------------------------------------------------------------------------
# 1. Direct initial flow regression head
# -----------------------------------------------------------------------------

class DirectInitialFlowHead(nn.Module):
    """
    Direct initial flow regression.

    This is SEA-RAFT-like in spirit: do not require the recurrent update alone
    to discover all large displacement. Produce an initial flow proposal directly
    from feature-pair evidence.

    Convention:
        output flow_yx = [dy, dx].
    """

    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int = 128,
        use_global_stats: bool = True,
        max_init_flow: float = 32.0,
    ):
        super().__init__()
        self.use_global_stats = bool(use_global_stats)
        self.max_init_flow = float(max_init_flow)

        in_ch = feature_dim * 4
        if use_global_stats:
            in_ch += feature_dim * 2

        self.net = nn.Sequential(
            nn.Conv2d(in_ch, hidden_dim, 3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_dim, 2, 3, padding=1),
        )

        # Start close to zero displacement.
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(
        self,
        f1: torch.Tensor,
        f2: torch.Tensor,
        gm_flow_yx: Optional[torch.Tensor] = None,
        gm_conf: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if gm_flow_yx is None:
            gm_flow_yx = torch.zeros(f1.shape[0], 2, f1.shape[-2], f1.shape[-1], device=f1.device, dtype=f1.dtype)

        if gm_conf is None:
            gm_conf = torch.ones(f1.shape[0], 1, f1.shape[-2], f1.shape[-1], device=f1.device, dtype=f1.dtype)

        x = [f1, f2, f1 - f2, f1 * f2]

        if self.use_global_stats:
            g1 = f1.mean(dim=(2, 3), keepdim=True).expand_as(f1)
            g2 = f2.mean(dim=(2, 3), keepdim=True).expand_as(f2)
            x.extend([g1, g2])

        x = torch.cat(x, dim=1)

        delta = self.net(x)
        delta = self.max_init_flow * torch.tanh(delta / max(self.max_init_flow, 1e-6))

        # Confidence-weighted blend: direct head corrects the global proposal,
        # rather than replacing it blindly.
        return gm_flow_yx + gm_conf.clamp(0, 1) * delta


# -----------------------------------------------------------------------------
# 2. Transformer-enhanced matching features
# -----------------------------------------------------------------------------

class TransformerFeatureEnhancer(nn.Module):
    """
    Lightweight self/cross attention feature enhancement before global matching.

    This is intentionally modest. Full transformer cost-volume architectures are
    expensive; this module improves long-range feature discrimination while
    remaining drop-in.
    """

    def __init__(
        self,
        feature_dim: int,
        num_heads: int = 4,
        depth: int = 1,
        mlp_ratio: float = 2.0,
        dropout: float = 0.0,
        max_tokens: int = 4096,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.max_tokens = int(max_tokens)

        self.self_attn_1 = nn.ModuleList()
        self.self_attn_2 = nn.ModuleList()
        self.cross_attn_12 = nn.ModuleList()
        self.cross_attn_21 = nn.ModuleList()
        self.ffn_1 = nn.ModuleList()
        self.ffn_2 = nn.ModuleList()
        self.norms = nn.ModuleList()

        hidden = int(feature_dim * mlp_ratio)

        for _ in range(depth):
            self.self_attn_1.append(nn.MultiheadAttention(feature_dim, num_heads, dropout=dropout, batch_first=True))
            self.self_attn_2.append(nn.MultiheadAttention(feature_dim, num_heads, dropout=dropout, batch_first=True))
            self.cross_attn_12.append(nn.MultiheadAttention(feature_dim, num_heads, dropout=dropout, batch_first=True))
            self.cross_attn_21.append(nn.MultiheadAttention(feature_dim, num_heads, dropout=dropout, batch_first=True))

            self.ffn_1.append(nn.Sequential(
                nn.LayerNorm(feature_dim),
                nn.Linear(feature_dim, hidden),
                nn.GELU(),
                nn.Linear(hidden, feature_dim),
            ))
            self.ffn_2.append(nn.Sequential(
                nn.LayerNorm(feature_dim),
                nn.Linear(feature_dim, hidden),
                nn.GELU(),
                nn.Linear(hidden, feature_dim),
            ))

            self.norms.append(nn.ModuleList([
                nn.LayerNorm(feature_dim),
                nn.LayerNorm(feature_dim),
                nn.LayerNorm(feature_dim),
                nn.LayerNorm(feature_dim),
            ]))

    def _flatten(self, x):
        b, c, h, w = x.shape
        return x.flatten(2).transpose(1, 2), (b, c, h, w)

    def _unflatten(self, x, shape):
        b, c, h, w = shape
        return x.transpose(1, 2).view(b, c, h, w)

    def forward(self, f1: torch.Tensor, f2: torch.Tensor):
        b, c, h, w = f1.shape
        n = h * w

        # Safety fallback: if resolution is too large, downsample for attention
        # and then upsample enhanced residual.
        if n > self.max_tokens:
            scale = math.sqrt(self.max_tokens / float(n))
            h2 = max(4, int(h * scale))
            w2 = max(4, int(w * scale))
            f1_small = F.interpolate(f1, size=(h2, w2), mode="bilinear", align_corners=True)
            f2_small = F.interpolate(f2, size=(h2, w2), mode="bilinear", align_corners=True)
            e1, e2 = self.forward(f1_small, f2_small)
            e1 = F.interpolate(e1, size=(h, w), mode="bilinear", align_corners=True)
            e2 = F.interpolate(e2, size=(h, w), mode="bilinear", align_corners=True)
            return f1 + e1, f2 + e2

        x1, shape = self._flatten(f1)
        x2, _ = self._flatten(f2)

        for i in range(len(self.self_attn_1)):
            n11, n22, n12, n21 = self.norms[i]

            y1, _ = self.self_attn_1[i](n11(x1), n11(x1), n11(x1), need_weights=False)
            y2, _ = self.self_attn_2[i](n22(x2), n22(x2), n22(x2), need_weights=False)
            x1 = x1 + y1
            x2 = x2 + y2

            c12, _ = self.cross_attn_12[i](n12(x1), n12(x2), n12(x2), need_weights=False)
            c21, _ = self.cross_attn_21[i](n21(x2), n21(x1), n21(x1), need_weights=False)
            x1 = x1 + c12
            x2 = x2 + c21

            x1 = x1 + self.ffn_1[i](x1)
            x2 = x2 + self.ffn_2[i](x2)

        return self._unflatten(x1, shape), self._unflatten(x2, shape)


# -----------------------------------------------------------------------------
# 3. Multi-scale global-to-local matching
# -----------------------------------------------------------------------------

class LocalFlowRefiner(nn.Module):
    """
    Local soft matching around a current flow estimate.

    This is not a recurrent GRU. It refines a global proposal by searching in a
    small window around the current target coordinate.
    """

    def __init__(
        self,
        feature_dim: int,
        radius: int = 4,
        temperature: float = 0.05,
    ):
        super().__init__()
        self.radius = int(radius)
        self.temperature = float(temperature)
        self.q_proj = nn.Conv2d(feature_dim, feature_dim, 1, bias=False)
        self.k_proj = nn.Conv2d(feature_dim, feature_dim, 1, bias=False)

    def forward(self, f1: torch.Tensor, f2: torch.Tensor, current_flow_yx: torch.Tensor):
        b, c, h, w = f1.shape
        r = self.radius

        q = F.normalize(self.q_proj(f1), dim=1)
        k = F.normalize(self.k_proj(f2), dim=1)

        coords = coords_grid_yx(b, h, w, f1.device, f1.dtype)
        tgt = coords + current_flow_yx

        offsets = []
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                offsets.append((dy, dx))

        logits = []
        offset_tensors = []

        for dy, dx in offsets:
            sample = tgt.clone()
            sample[:, 0] += dy
            sample[:, 1] += dx

            # grid_sample expects [x,y] normalised coordinates.
            gx = 2.0 * (sample[:, 1] / max(w - 1, 1)) - 1.0
            gy = 2.0 * (sample[:, 0] / max(h - 1, 1)) - 1.0
            grid = torch.stack([gx, gy], dim=-1)

            k_s = F.grid_sample(k, grid, mode="bilinear", padding_mode="zeros", align_corners=True)
            score = (q * k_s).sum(dim=1, keepdim=True) / max(self.temperature, 1e-6)
            logits.append(score)

            off = torch.tensor([dy, dx], device=f1.device, dtype=f1.dtype).view(1, 2, 1, 1)
            offset_tensors.append(off.expand(b, -1, h, w))

        logits = torch.cat(logits, dim=1)  # [B, WN, H, W]
        prob = F.softmax(logits, dim=1)

        offsets_yx = torch.stack(offset_tensors, dim=1)  # [B, WN, 2, H, W]
        delta = (prob.unsqueeze(2) * offsets_yx).sum(dim=1)  # [B,2,H,W]

        conf = prob.max(dim=1, keepdim=True).values
        return current_flow_yx + delta, conf


class MultiScaleGlobalLocalInitializer(nn.Module):
    """
    Coarse global matching followed by finer local refinement.

    Expected feature keys:
        level2: coarse, e.g. 1/8
        level1: finer, e.g. 1/4

    This module can sit before the HQS loop.
    """

    def __init__(
        self,
        pgma_module: nn.Module,
        feature_dim: int,
        local_radius: int = 4,
        local_temperature: float = 0.05,
        use_level1: bool = True,
    ):
        super().__init__()
        self.pgma = pgma_module
        self.use_level1 = bool(use_level1)
        self.local_refiner = LocalFlowRefiner(feature_dim, radius=local_radius, temperature=local_temperature)

    def forward(
        self,
        feats1: Dict[str, torch.Tensor],
        feats2: Dict[str, torch.Tensor],
        target_valid: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        f1_l2 = feats1["level2"]
        f2_l2 = feats2["level2"]

        gm = self.pgma.global_match(f1_l2, f2_l2, target_valid=target_valid)
        flow_l2 = gm["flow_yx"]
        conf_l2 = gm["conf"]

        out = {
            "flow_l2_yx": flow_l2,
            "conf_l2": conf_l2,
            "entropy_l2": gm["entropy"],
            "margin_l2": gm["margin"],
        }

        if self.use_level1 and "level1" in feats1 and "level1" in feats2:
            f1_l1 = feats1["level1"]
            f2_l1 = feats2["level1"]

            flow_l1 = resize_flow_yx(flow_l2, f1_l1.shape[-2:])
            flow_l1, conf_l1 = self.local_refiner(f1_l1, f2_l1, flow_l1)

            out["flow_l1_yx"] = flow_l1
            out["conf_l1"] = conf_l1

        return out


# -----------------------------------------------------------------------------
# 4. Boundary-aware, confidence-aware learned proximal
# -----------------------------------------------------------------------------

class BoundaryConfidenceProximal(nn.Module):
    """
    Learned HQS proximal q-step.

    q_{k+1} = P_theta(w_{k+1}, q_k, features, validity, residual, confidence)

    This is designed to avoid over-smoothing across boundaries and to improve
    occlusion-region inpainting.
    """

    def __init__(
        self,
        context_dim: int,
        hidden_dim: int = 128,
        max_delta: float = 8.0,
    ):
        super().__init__()
        self.max_delta = float(max_delta)

        # Inputs:
        # flow_yx 2
        # prev_aux_yx 2
        # flow - aux 2
        # validity 1
        # image edge 1
        # hqs residual norm 1
        # global confidence 1
        # global entropy 1
        # context features context_dim
        in_ch = context_dim + 11

        self.net = nn.Sequential(
            nn.Conv2d(in_ch, hidden_dim, 3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_dim, 3, 3, padding=1),
        )

        # Output channels:
        # delta_aux_yx: 2
        # anchor_gate_logit: 1
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(
        self,
        flow_yx: torch.Tensor,
        prev_aux_yx: torch.Tensor,
        context_feat: torch.Tensor,
        validity: torch.Tensor,
        image_edge: Optional[torch.Tensor] = None,
        hqs_resid: Optional[torch.Tensor] = None,
        global_conf: Optional[torch.Tensor] = None,
        global_entropy: Optional[torch.Tensor] = None,
    ):
        h, w = flow_yx.shape[-2:]

        if context_feat.shape[-2:] != (h, w):
            context_feat = F.interpolate(context_feat, size=(h, w), mode="bilinear", align_corners=True)

        def ensure_1(x, default):
            if x is None:
                x = default
            if x.shape[-2:] != (h, w):
                x = F.interpolate(x.float(), size=(h, w), mode="bilinear", align_corners=True)
            return x

        zeros1 = flow_yx.new_zeros(flow_yx.shape[0], 1, h, w)
        ones1 = flow_yx.new_ones(flow_yx.shape[0], 1, h, w)

        validity = ensure_1(validity, ones1)
        image_edge = ensure_1(image_edge, zeros1)
        global_conf = ensure_1(global_conf, zeros1)
        global_entropy = ensure_1(global_entropy, ones1)

        if hqs_resid is None:
            hqs_norm = (flow_yx - prev_aux_yx).abs().mean(dim=1, keepdim=True)
        else:
            if hqs_resid.shape[-2:] != (h, w):
                hqs_resid = F.interpolate(hqs_resid, size=(h, w), mode="bilinear", align_corners=True)
            hqs_norm = hqs_resid.abs().mean(dim=1, keepdim=True)

        x = torch.cat(
            [
                flow_yx,
                prev_aux_yx,
                flow_yx - prev_aux_yx,
                validity,
                image_edge,
                hqs_norm,
                global_conf,
                global_entropy,
                context_feat,
            ],
            dim=1,
        )

        raw = self.net(x)
        delta = raw[:, 0:2]
        anchor_gate = torch.sigmoid(raw[:, 2:3])

        delta = self.max_delta * torch.tanh(delta / max(self.max_delta, 1e-6))
        proposed = prev_aux_yx + delta

        # Boundary-aware anchoring:
        # high validity/confidence -> stay closer to current flow;
        # high entropy/low validity -> allow prior/inpainting.
        confidence_anchor = (validity * global_conf * (1.0 - global_entropy.clamp(0, 1))).clamp(0, 1)
        anchor = torch.maximum(anchor_gate, confidence_anchor)

        aux = anchor * flow_yx + (1.0 - anchor) * proposed
        return aux, {"anchor": anchor, "delta_aux": delta}


# -----------------------------------------------------------------------------
# 5. Mixture Laplace uncertainty head and loss
# -----------------------------------------------------------------------------

class FlowUncertaintyHead(nn.Module):
    """
    Predicts mixture weights and log-scales for a Laplace mixture likelihood.

    This does not predict the flow itself. It predicts uncertainty for a given
    predicted flow.
    """

    def __init__(
        self,
        context_dim: int,
        hidden_dim: int = 96,
        num_mixtures: int = 2,
        min_log_scale: float = -5.0,
        max_log_scale: float = 3.0,
    ):
        super().__init__()
        self.num_mixtures = int(num_mixtures)
        self.min_log_scale = float(min_log_scale)
        self.max_log_scale = float(max_log_scale)

        in_ch = context_dim + 3  # context + flow magnitude + image edge + validity
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, hidden_dim, 3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_dim, 2 * num_mixtures, 3, padding=1),
        )

    def forward(
        self,
        context_feat: torch.Tensor,
        flow_pred: torch.Tensor,
        validity: Optional[torch.Tensor] = None,
        image_edge: Optional[torch.Tensor] = None,
    ):
        h, w = flow_pred.shape[-2:]

        if context_feat.shape[-2:] != (h, w):
            context_feat = F.interpolate(context_feat, size=(h, w), mode="bilinear", align_corners=True)

        flow_mag = flow_pred.pow(2).sum(dim=1, keepdim=True).sqrt()

        if validity is None:
            validity = flow_pred.new_ones(flow_pred.shape[0], 1, h, w)
        elif validity.shape[-2:] != (h, w):
            validity = F.interpolate(validity.float(), size=(h, w), mode="nearest")

        if image_edge is None:
            image_edge = flow_pred.new_zeros(flow_pred.shape[0], 1, h, w)
        elif image_edge.shape[-2:] != (h, w):
            image_edge = F.interpolate(image_edge.float(), size=(h, w), mode="bilinear", align_corners=True)

        x = torch.cat([context_feat, flow_mag, image_edge, validity], dim=1)
        raw = self.net(x)

        logits = raw[:, : self.num_mixtures]
        log_scales = raw[:, self.num_mixtures :]
        log_scales = log_scales.clamp(self.min_log_scale, self.max_log_scale)

        return {"mix_logits": logits, "log_scales": log_scales}


def mixture_laplace_nll(
    flow_pred: torch.Tensor,
    flow_gt: torch.Tensor,
    valid: torch.Tensor,
    mix_logits: torch.Tensor,
    log_scales: torch.Tensor,
    max_flow: float = 400.0,
):
    """
    Mixture Laplace negative log-likelihood.

    flow_pred, flow_gt: [B,2,H,W]
    mix_logits, log_scales: [B,K,H,W]
    valid: [B,H,W] or [B,1,H,W]
    """
    if valid.dim() == 3:
        valid = valid.unsqueeze(1)

    if flow_gt.shape[-2:] != flow_pred.shape[-2:]:
        h0, w0 = flow_gt.shape[-2:]
        h1, w1 = flow_pred.shape[-2:]
        flow_gt = F.interpolate(flow_gt, size=flow_pred.shape[-2:], mode="bilinear", align_corners=True)
        flow_gt = flow_gt.clone()
        flow_gt[:, 0] *= float(w1) / float(w0)
        flow_gt[:, 1] *= float(h1) / float(h0)
        valid = F.interpolate(valid.float(), size=flow_pred.shape[-2:], mode="nearest")

    mag = flow_gt.pow(2).sum(dim=1, keepdim=True).sqrt()
    mask = (valid > 0.5) & (mag < max_flow)

    residual = (flow_pred - flow_gt).abs().sum(dim=1, keepdim=True)  # [B,1,H,W]
    residual = residual.expand_as(log_scales)

    log_pi = F.log_softmax(mix_logits, dim=1)
    # Laplace NLL for 2D residual approximated with channel-summed L1.
    log_prob = log_pi - log_scales - residual / torch.exp(log_scales).clamp_min(1e-6)
    nll = -torch.logsumexp(log_prob, dim=1, keepdim=True)

    if mask.any():
        return nll[mask].mean()
    return nll.mean()


# -----------------------------------------------------------------------------
# 6. High-speed, boundary, occlusion-aware auxiliary losses
# -----------------------------------------------------------------------------

class RobustAuxiliaryFlowLosses(nn.Module):
    """
    Auxiliary losses for the failure cases:
      - high-speed motion
      - motion boundaries / thin structures
      - occluded or unmatched regions
      - PGMA global proposal supervision
    """

    def __init__(
        self,
        speed_thresh: float = 40.0,
        boundary_thresh: float = 1.0,
        high_speed_weight: float = 0.05,
        boundary_weight: float = 0.02,
        occlusion_weight: float = 0.03,
        gradient_weight: float = 0.02,
        pgma_init_weight: float = 0.05,
        pgma_iter_weight: float = 0.02,
        max_flow: float = 400.0,
    ):
        super().__init__()
        self.speed_thresh = float(speed_thresh)
        self.boundary_thresh = float(boundary_thresh)
        self.high_speed_weight = float(high_speed_weight)
        self.boundary_weight = float(boundary_weight)
        self.occlusion_weight = float(occlusion_weight)
        self.gradient_weight = float(gradient_weight)
        self.pgma_init_weight = float(pgma_init_weight)
        self.pgma_iter_weight = float(pgma_iter_weight)
        self.max_flow = float(max_flow)

    @staticmethod
    def _resize_flow_xy(flow_gt, size_hw):
        h0, w0 = flow_gt.shape[-2:]
        h1, w1 = size_hw
        out = F.interpolate(flow_gt, size=size_hw, mode="bilinear", align_corners=True)
        out = out.clone()
        out[:, 0] *= float(w1) / float(w0)
        out[:, 1] *= float(h1) / float(h0)
        return out

    @staticmethod
    def _masked_mean(x, mask):
        if mask.dim() == 3:
            mask = mask.unsqueeze(1)
        mask = mask.float()
        return (x * mask).sum() / mask.sum().clamp_min(1.0)

    def _endpoint(self, pred, gt):
        return (pred - gt).pow(2).sum(dim=1, keepdim=True).sqrt()

    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        flow_preds,
        flow_gt: torch.Tensor,
        valid: torch.Tensor,
        occlusion: Optional[torch.Tensor] = None,
    ):
        """
        Args:
            outputs: model output dict.
            flow_preds: list of full-resolution [B,2,H,W] predictions, [dx,dy].
            flow_gt: [B,2,H,W], [dx,dy].
            valid: [B,H,W] or [B,1,H,W].
            occlusion: optional [B,H,W] or [B,1,H,W], 1 means occluded/unmatched.
        """
        if valid.dim() == 3:
            valid1 = valid.unsqueeze(1).float()
        else:
            valid1 = valid.float()

        mag = flow_gt.pow(2).sum(dim=1, keepdim=True).sqrt()
        valid_flow = valid1 * (mag < self.max_flow).float()

        final = flow_preds[-1]
        epe = self._endpoint(final, flow_gt)

        total = flow_gt.new_zeros(())
        logs = {}

        # High-speed loss.
        speed_mask = valid_flow * (mag > self.speed_thresh).float()
        loss_speed = self._masked_mean(epe, speed_mask)
        total = total + self.high_speed_weight * loss_speed
        logs["loss_high_speed"] = loss_speed.detach()

        # Boundary / thin-structure loss.
        bmask = flow_boundary_mask(flow_gt, threshold=self.boundary_thresh) * valid_flow
        loss_boundary = self._masked_mean(epe, bmask)
        total = total + self.boundary_weight * loss_boundary
        logs["loss_boundary"] = loss_boundary.detach()

        # Flow-gradient loss.
        pred_dx, pred_dy = flow_gradients(final)
        gt_dx, gt_dy = flow_gradients(flow_gt)
        grad_err = (pred_dx - gt_dx).abs().mean(dim=1, keepdim=True) + (pred_dy - gt_dy).abs().mean(dim=1, keepdim=True)
        loss_grad = self._masked_mean(grad_err, valid_flow)
        total = total + self.gradient_weight * loss_grad
        logs["loss_flow_gradient"] = loss_grad.detach()

        # Occlusion / unmatched loss.
        if occlusion is not None:
            if occlusion.dim() == 3:
                occ = occlusion.unsqueeze(1).float()
            else:
                occ = occlusion.float()
            occ = F.interpolate(occ, size=flow_gt.shape[-2:], mode="nearest")
            occ_mask = valid_flow * occ
            loss_occ = self._masked_mean(epe, occ_mask)
            total = total + self.occlusion_weight * loss_occ
            logs["loss_occlusion"] = loss_occ.detach()

        # GMFlow-style init supervision.
        if "gmflow_init_flow_yx" in outputs:
            flow_yx = outputs["gmflow_init_flow_yx"]
            flow_xy = torch.stack([flow_yx[:, 1], flow_yx[:, 0]], dim=1)

            gt_low = self._resize_flow_xy(flow_gt, flow_xy.shape[-2:])
            valid_low = F.interpolate(valid1, size=flow_xy.shape[-2:], mode="nearest")

            loss_pgma_init = self._masked_mean(self._endpoint(flow_xy, gt_low), valid_low)
            total = total + self.pgma_init_weight * loss_pgma_init
            logs["loss_pgma_init"] = loss_pgma_init.detach()

        # Per-iteration PGMA global proposal supervision.
        pgma_flows = outputs.get("pgma_global_flow_lows", [])
        if len(pgma_flows) > 0:
            accum = flow_gt.new_zeros(())
            for flow_xy in pgma_flows:
                gt_low = self._resize_flow_xy(flow_gt, flow_xy.shape[-2:])
                valid_low = F.interpolate(valid1, size=flow_xy.shape[-2:], mode="nearest")
                accum = accum + self._masked_mean(self._endpoint(flow_xy, gt_low), valid_low)

            loss_pgma_iter = accum / len(pgma_flows)
            total = total + self.pgma_iter_weight * loss_pgma_iter
            logs["loss_pgma_iter"] = loss_pgma_iter.detach()

        logs["loss_aux_total"] = total.detach()
        return total, logs