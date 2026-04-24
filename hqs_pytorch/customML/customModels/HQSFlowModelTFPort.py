"""Faithful PyTorch recreation of the TensorFlow HQSFlowModel.

This implementation keeps the repository training contract:
- forward() returns a dict with flow_preds / flow_low / hidden_states
- cfg.model controls construction
- param_count() reports component totals

Core behavior (faithful to provided TF model):
- Single-scale HQS loop at 1/8 resolution
- All-pairs correlation pyramid sampled around current coordinates
- Data step + proximal step + residual refinement per iteration
- Final convex upsampling to full image resolution
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class GroupNorm2D(nn.Module):
    def __init__(self, channels: int, groups: int = 8, eps: float = 1e-5) -> None:
        super().__init__()
        g = min(groups, channels)
        while channels % g != 0 and g > 1:
            g -= 1
        self.norm = nn.GroupNorm(g, channels, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x)


class ConvGNAct(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        groups: int = 8,
        activation: Optional[str] = "silu",
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=kernel_size // 2,
            bias=False,
        )
        self.norm = GroupNorm2D(out_channels, groups=groups)
        if activation == "silu":
            self.act = nn.SiLU(inplace=True)
        elif activation == "relu":
            self.act = nn.ReLU(inplace=True)
        elif activation == "gelu":
            self.act = nn.GELU()
        elif activation is None:
            self.act = nn.Identity()
        else:
            raise ValueError(f"Unsupported activation: {activation}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.conv(x)))


class ResidualBlockGN(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        groups: int = 8,
        dropout_rate: float = 0.0,
    ) -> None:
        super().__init__()
        self.conv1 = ConvGNAct(
            in_channels, out_channels, kernel_size=3, stride=stride, groups=groups, activation="silu"
        )
        self.conv2 = ConvGNAct(
            out_channels, out_channels, kernel_size=3, stride=1, groups=groups, activation=None
        )
        self.dropout = nn.Dropout2d(dropout_rate) if dropout_rate > 0.0 else nn.Identity()
        self.proj = nn.Identity()
        if stride != 1 or in_channels != out_channels:
            self.proj = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                GroupNorm2D(out_channels, groups=groups),
            )
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.dropout(self.conv2(self.conv1(x)))
        return self.act(y + self.proj(x))


class OpticalFlowFeatureEncoderTF(nn.Module):
    def __init__(
        self,
        base_channels: int = 16,
        channel_multiplier: Tuple[int, ...] = (1, 2, 3),
        blocks_per_stage: Tuple[int, ...] = (2, 2, 2),
        groups: int = 8,
        dropout_rate: float = 0.0,
        output_projection_dim: int = 48,
    ) -> None:
        super().__init__()
        self.output_projection_dim = output_projection_dim
        self.stem = ConvGNAct(3, base_channels, kernel_size=7, stride=2, groups=groups, activation="silu")

        self.stages = nn.ModuleList()
        self.projections = nn.ModuleList()
        in_ch = base_channels
        for stage_idx, (mult, n_blocks) in enumerate(zip(channel_multiplier, blocks_per_stage)):
            out_ch = base_channels * mult
            stride = 1 if stage_idx == 0 else 2
            blocks: List[nn.Module] = [
                ResidualBlockGN(in_ch, out_ch, stride=stride, groups=groups, dropout_rate=dropout_rate)
            ]
            for _ in range(1, n_blocks):
                blocks.append(ResidualBlockGN(out_ch, out_ch, stride=1, groups=groups, dropout_rate=dropout_rate))
            self.stages.append(nn.Sequential(*blocks))
            self.projections.append(nn.Conv2d(out_ch, output_projection_dim, kernel_size=1))
            in_ch = out_ch

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        y = self.stem(x)
        feats: Dict[str, torch.Tensor] = {}
        for stage_idx, stage in enumerate(self.stages):
            y = stage(y)
            feats[f"level{stage_idx}"] = self.projections[stage_idx](y)
        return feats


class OpticalFlowContextEncoderTF(nn.Module):
    def __init__(
        self,
        base_channels: int = 16,
        channel_multiplier: Tuple[int, ...] = (1, 2, 3),
        blocks_per_stage: Tuple[int, ...] = (2, 2, 2),
        groups: int = 8,
        context_dim: int = 64,
        hidden_dim: int = 32,
        dropout_rate: float = 0.0,
    ) -> None:
        super().__init__()
        self.context_dim = context_dim
        self.hidden_dim = hidden_dim

        self.stem = ConvGNAct(3, base_channels, kernel_size=7, stride=2, groups=groups, activation="silu")
        self.stages = nn.ModuleList()
        in_ch = base_channels
        for stage_idx, (mult, n_blocks) in enumerate(zip(channel_multiplier, blocks_per_stage)):
            out_ch = base_channels * mult
            stride = 1 if stage_idx == 0 else 2
            blocks: List[nn.Module] = [
                ResidualBlockGN(in_ch, out_ch, stride=stride, groups=groups, dropout_rate=dropout_rate)
            ]
            for _ in range(1, n_blocks):
                blocks.append(ResidualBlockGN(out_ch, out_ch, stride=1, groups=groups, dropout_rate=dropout_rate))
            self.stages.append(nn.Sequential(*blocks))
            in_ch = out_ch

        self.context_head = nn.Conv2d(in_ch, context_dim, kernel_size=1)
        self.hidden_head = nn.Conv2d(in_ch, hidden_dim, kernel_size=1)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        y = self.stem(x)
        for stage in self.stages:
            y = stage(y)
        return {
            "context_feat": self.context_head(y),
            "hidden_state": self.hidden_head(y),
        }


def _cfg_get(cfg, key: str, default):
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    try:
        return cfg.get(key, default)
    except Exception:
        return getattr(cfg, key, default)


class HQSFlowModelTFPort(nn.Module):
    """TensorFlow-faithful HQS optical flow model in PyTorch."""

    def __init__(self, cfg) -> None:
        super().__init__()
        self.cfg = cfg

        tfp = _cfg_get(cfg, "tf_port", {})
        self.num_hqs_iterations = int(_cfg_get(tfp, "num_hqs_iterations", _cfg_get(cfg, "num_stages", 10)))
        self.upsample_scale = int(_cfg_get(cfg, "upsample_scale", 8))

        corr_cfg = _cfg_get(cfg, "corr", {})
        self.max_displacement = int(_cfg_get(tfp, "max_displacement", _cfg_get(corr_cfg, "radius", 4)))
        self.num_corr_levels = int(_cfg_get(tfp, "num_corr_levels", _cfg_get(corr_cfg, "num_levels", 4)))

        base_channels = int(_cfg_get(tfp, "base_channels", 16))
        channel_multiplier = tuple(_cfg_get(tfp, "channel_multiplier", (1, 2, 3)))
        blocks_per_stage = tuple(_cfg_get(tfp, "blocks_per_stage", (2, 2, 2)))
        groups = int(_cfg_get(tfp, "groups", 8))
        dropout_rate = float(_cfg_get(tfp, "dropout_rate", 0.0))
        feature_dim = int(_cfg_get(tfp, "feature_dim", 48))
        context_dim = int(_cfg_get(tfp, "context_dim", 64))
        hidden_dim = int(_cfg_get(tfp, "hidden_dim", 32))

        self.feature_encoder = OpticalFlowFeatureEncoderTF(
            base_channels=base_channels,
            channel_multiplier=channel_multiplier,
            blocks_per_stage=blocks_per_stage,
            groups=groups,
            dropout_rate=dropout_rate,
            output_projection_dim=feature_dim,
        )
        self.context_encoder = OpticalFlowContextEncoderTF(
            base_channels=base_channels,
            channel_multiplier=channel_multiplier,
            blocks_per_stage=blocks_per_stage,
            groups=groups,
            context_dim=context_dim,
            hidden_dim=hidden_dim,
            dropout_rate=dropout_rate,
        )

        # Learned penalties per HQS iteration.
        beta_init = torch.tensor([0.05, 0.08, 0.12, 0.16], dtype=torch.float32)
        lam_init = torch.tensor([0.03, 0.04, 0.05, 0.06], dtype=torch.float32)
        beta = beta_init.repeat((self.num_hqs_iterations + 3) // 4)[: self.num_hqs_iterations]
        lam = lam_init.repeat((self.num_hqs_iterations + 3) // 4)[: self.num_hqs_iterations]
        self.hqs_beta = nn.Parameter(beta)
        self.hqs_lambda = nn.Parameter(lam)

        corr_channels = self.num_corr_levels * (2 * self.max_displacement + 1) ** 2
        self.allpairs_corr_decoder = nn.Sequential(
            nn.Conv2d(corr_channels, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.allpairs_init_head = nn.Conv2d(64 + 2, 2, kernel_size=3, padding=1)
        self.allpairs_conf_head = nn.Sequential(nn.Conv2d(64, 1, kernel_size=3, padding=1), nn.Sigmoid())

        self.refinement_network = nn.Conv2d(2 + 2 + feature_dim + context_dim, 2, kernel_size=3, padding=1)
        self.final_refinement_network = nn.Sequential(
            nn.Conv2d(2 + context_dim + hidden_dim, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 2, kernel_size=3, padding=1),
        )
        self.final_mask_logits = nn.Conv2d(2 + context_dim + hidden_dim, 8 * 8 * 9, kernel_size=3, padding=1)

    @staticmethod
    def _normalise(img: torch.Tensor) -> torch.Tensor:
        if img.max() > 2.0:
            img = img / 255.0
        mean = img.new_tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = img.new_tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        return (img - mean) / std

    @staticmethod
    def l2_normalize_features(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        return x / (x.pow(2).sum(dim=1, keepdim=True).add(eps).sqrt())

    @staticmethod
    def _coords_grid(batch: int, h: int, w: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        ys, xs = torch.meshgrid(
            torch.arange(h, device=device, dtype=dtype),
            torch.arange(w, device=device, dtype=dtype),
            indexing="ij",
        )
        # [y, x] convention to match TF port internals.
        grid = torch.stack([ys, xs], dim=-1)  # [H, W, 2]
        return grid.unsqueeze(0).expand(batch, -1, -1, -1).contiguous()

    @staticmethod
    def _warp_yx(x: torch.Tensor, flow_yx: torch.Tensor) -> torch.Tensor:
        """Warp x with flow in [dy, dx] channel order."""
        b, _, h, w = x.shape
        ys, xs = torch.meshgrid(
            torch.arange(h, device=x.device, dtype=x.dtype),
            torch.arange(w, device=x.device, dtype=x.dtype),
            indexing="ij",
        )
        ys = ys.view(1, h, w).expand(b, -1, -1)
        xs = xs.view(1, h, w).expand(b, -1, -1)

        qy = ys + flow_yx[:, 0]
        qx = xs + flow_yx[:, 1]

        if h > 1:
            gy = 2.0 * qy / (h - 1) - 1.0
        else:
            gy = torch.zeros_like(qy)
        if w > 1:
            gx = 2.0 * qx / (w - 1) - 1.0
        else:
            gx = torch.zeros_like(qx)

        grid = torch.stack([gx, gy], dim=-1)
        return F.grid_sample(x, grid, mode="bilinear", padding_mode="border", align_corners=True)

    def build_all_pairs_correlation(self, f1: torch.Tensor, f2: torch.Tensor) -> torch.Tensor:
        """Return [B, H, W, H, W] all-pairs correlation."""
        f1 = self.l2_normalize_features(f1)
        f2 = self.l2_normalize_features(f2)
        b, c, h, w = f1.shape
        f1_flat = f1.view(b, c, h * w).transpose(1, 2)  # [B, HW, C]
        f2_flat = f2.view(b, c, h * w).transpose(1, 2)  # [B, HW, C]
        corr = torch.bmm(f1_flat, f2_flat.transpose(1, 2)) / math.sqrt(float(c))
        return corr.view(b, h, w, h, w)

    def build_corr_pyramid_from_all_pairs(self, corr: torch.Tensor, num_levels: Optional[int] = None) -> List[torch.Tensor]:
        """Build correlation pyramid as [B*H1*W1, 1, H2_l, W2_l] per level."""
        if num_levels is None:
            num_levels = self.num_corr_levels

        b, h1, w1, h2, w2 = corr.shape
        corr_lvl = corr.view(b * h1 * w1, 1, h2, w2)
        pyramid: List[torch.Tensor] = [corr_lvl]
        for _ in range(num_levels - 1):
            corr_lvl = F.avg_pool2d(corr_lvl, kernel_size=2, stride=2, ceil_mode=True)
            pyramid.append(corr_lvl)
        return pyramid

    def sample_all_pairs_corr_pyramid(self, corr_pyramid: List[torch.Tensor], coords_yx: torch.Tensor, radius: int) -> torch.Tensor:
        """Sample pyramid around coords using bilinear lookup.

        corr_pyramid: each level [B*H*W, 1, H_l, W_l]
        coords_yx: [B, H, W, 2] at finest scale, [y, x]
        returns: [B, H, W, L*(2r+1)^2]
        """
        b, h, w, _ = coords_yx.shape
        n = b * h * w
        k = (2 * radius + 1) ** 2

        dy, dx = torch.meshgrid(
            torch.arange(-radius, radius + 1, device=coords_yx.device, dtype=coords_yx.dtype),
            torch.arange(-radius, radius + 1, device=coords_yx.device, dtype=coords_yx.dtype),
            indexing="ij",
        )
        delta = torch.stack([dy, dx], dim=-1).view(1, 1, 1, k, 2)

        patches: List[torch.Tensor] = []
        for lvl, corr in enumerate(corr_pyramid):
            _, _, hl, wl = corr.shape
            coords_lvl = coords_yx / (2 ** lvl)
            coords_win = coords_lvl.unsqueeze(3) + delta  # [B,H,W,K,2]
            coords_win = coords_win.view(n, k, 2)

            y = coords_win[..., 0]
            x = coords_win[..., 1]
            if hl > 1:
                gy = 2.0 * y / (hl - 1) - 1.0
            else:
                gy = torch.zeros_like(y)
            if wl > 1:
                gx = 2.0 * x / (wl - 1) - 1.0
            else:
                gx = torch.zeros_like(x)

            grid = torch.stack([gx, gy], dim=-1).view(n, 1, k, 2)
            samp = F.grid_sample(corr, grid, mode="bilinear", padding_mode="border", align_corners=True)
            samp = samp.view(b, h, w, k)
            patches.append(samp)

        return torch.cat(patches, dim=-1)

    def build_all_pairs_init_from_lookup(self, corr_features: torch.Tensor, flow_yx: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # corr_features: [B,H,W,C] -> [B,C,H,W]
        x = corr_features.permute(0, 3, 1, 2).contiguous()
        x = self.allpairs_corr_decoder(x)
        delta_init = self.allpairs_init_head(torch.cat([flow_yx, x], dim=1))
        confidence = self.allpairs_conf_head(x)
        return delta_init, confidence, x

    @staticmethod
    def central_grad_x(x: torch.Tensor) -> torch.Tensor:
        gx = 0.5 * (x[:, :, :, 2:] - x[:, :, :, :-2])
        return F.pad(gx, (1, 1, 0, 0), mode="replicate")

    @staticmethod
    def central_grad_y(x: torch.Tensor) -> torch.Tensor:
        gy = 0.5 * (x[:, :, 2:, :] - x[:, :, :-2, :])
        return F.pad(gy, (0, 0, 1, 1), mode="replicate")

    def compute_ofce_derivatives(self, i1: torch.Tensor, i2: torch.Tensor, flow_yx: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        i2_warp = self._warp_yx(i2, flow_yx)
        gx1 = self.central_grad_x(i1)
        gx2 = self.central_grad_x(i2_warp)
        gy1 = self.central_grad_y(i1)
        gy2 = self.central_grad_y(i2_warp)

        ix = 0.5 * (gx1 + gx2)
        iy = 0.5 * (gy1 + gy2)
        it = i2_warp - i1

        if ix.shape[1] > 1:
            ix = ix.mean(dim=1, keepdim=True)
            iy = iy.mean(dim=1, keepdim=True)
            it = it.mean(dim=1, keepdim=True)
        return ix, iy, it, i2_warp

    @staticmethod
    def hqs_data_step(
        ix: torch.Tensor,
        iy: torch.Tensor,
        it: torch.Tensor,
        aux_yx: torch.Tensor,
        confidence: torch.Tensor,
        flow0_yx: torch.Tensor,
        beta: torch.Tensor,
        robust: bool = True,
        eps: float = 1e-3,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Keep same channel semantics as TF: flow_yx = [v, u] = [dy, dx].
        p = aux_yx[:, 0:1]
        q = aux_yx[:, 1:2]
        v0 = flow0_yx[:, 0:1]
        u0 = flow0_yx[:, 1:2]

        c = confidence.clamp(0.05, 1.0)
        if robust:
            r = ix * u0 + iy * v0 + it
            wd = 1.0 / (r.square() + eps ** 2).sqrt()
        else:
            wd = torch.ones_like(ix)

        w = c * wd
        a11 = w * ix * ix + beta
        a12 = w * ix * iy
        a22 = w * iy * iy + beta
        b1 = -w * ix * it + beta * p
        b2 = -w * iy * it + beta * q

        det = a11 * a22 - a12 * a12
        u = (a22 * b1 - a12 * b2) / (det + 1e-6)
        v = (-a12 * b1 + a11 * b2) / (det + 1e-6)
        flow_yx = torch.cat([v, u], dim=1)
        return flow_yx, wd, det

    @staticmethod
    def edge_weights(image: torch.Tensor, alpha: float = 10.0) -> Tuple[torch.Tensor, torch.Tensor]:
        dx = image[:, :, :, 1:] - image[:, :, :, :-1]
        dy = image[:, :, 1:, :] - image[:, :, :-1, :]
        wx = torch.exp(-alpha * dx.abs().mean(dim=1, keepdim=True))
        wy = torch.exp(-alpha * dy.abs().mean(dim=1, keepdim=True))
        return wx, wy

    @staticmethod
    def jacobi_prox_step(
        x: torch.Tensor,
        rhs: torch.Tensor,
        beta: torch.Tensor,
        lam: torch.Tensor,
        wx: torch.Tensor,
        wy: torch.Tensor,
    ) -> torch.Tensor:
        x_up = F.pad(x[:, :, :-1, :], (0, 0, 1, 0), mode="replicate")
        x_down = F.pad(x[:, :, 1:, :], (0, 0, 0, 1), mode="replicate")
        x_left = F.pad(x[:, :, :, :-1], (1, 0, 0, 0), mode="replicate")
        x_right = F.pad(x[:, :, :, 1:], (0, 1, 0, 0), mode="replicate")

        wx_l = F.pad(wx, (1, 0, 0, 0), mode="replicate")
        wx_r = F.pad(wx, (0, 1, 0, 0), mode="replicate")
        wy_u = F.pad(wy, (0, 0, 1, 0), mode="replicate")
        wy_d = F.pad(wy, (0, 0, 0, 1), mode="replicate")

        numer = beta * rhs + lam * (wy_u * x_up + wy_d * x_down + wx_l * x_left + wx_r * x_right)
        denom = beta + lam * (wy_u + wy_d + wx_l + wx_r)
        return numer / (denom + 1e-6)

    def hqs_prox_step(
        self,
        flow_yx: torch.Tensor,
        image: torch.Tensor,
        beta: torch.Tensor,
        lam: torch.Tensor,
        num_iter: int = 1,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        p = flow_yx[:, 0:1].clone()
        q = flow_yx[:, 1:2].clone()
        wx, wy = self.edge_weights(image)
        for _ in range(num_iter):
            p = self.jacobi_prox_step(p, flow_yx[:, 0:1], beta, lam, wx, wy)
            q = self.jacobi_prox_step(q, flow_yx[:, 1:2], beta, lam, wx, wy)
        return torch.cat([p, q], dim=1), wx, wy

    @staticmethod
    def convex_upsample(flow_lr: torch.Tensor, mask_logits: torch.Tensor, rate: int = 8) -> torch.Tensor:
        """RAFT-style convex upsampling, matching TensorFlow logic."""
        b, _, h, w = flow_lr.shape
        mask = mask_logits.view(b, 1, 9, rate, rate, h, w)
        mask = torch.softmax(mask, dim=2)

        up_flow = F.unfold(flow_lr, kernel_size=3, padding=1)
        up_flow = up_flow.view(b, 2, 9, 1, 1, h, w)
        up_flow = (mask * up_flow).sum(dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3).reshape(b, 2, rate * h, rate * w)
        return up_flow

    def _resize_flow_yx(self, flow: torch.Tensor, out_h: int, out_w: int) -> torch.Tensor:
        in_h, in_w = flow.shape[-2:]
        if in_h == out_h and in_w == out_w:
            return flow
        resized = F.interpolate(flow, size=(out_h, out_w), mode="bilinear", align_corners=True)
        sy = float(out_h) / float(in_h)
        sx = float(out_w) / float(in_w)
        scale = flow.new_tensor([sy, sx]).view(1, 2, 1, 1)
        return resized * scale

    def forward(
        self,
        image1: torch.Tensor,
        image2: torch.Tensor,
        iters: Optional[int] = None,
        flow_init: Optional[torch.Tensor] = None,
    ) -> Dict[str, List[torch.Tensor]]:
        iters = iters or self.num_hqs_iterations
        if iters > self.num_hqs_iterations:
            raise ValueError(
                f"Requested {iters} iterations, but model initialized with {self.num_hqs_iterations}."
            )

        i1 = self._normalise(image1)
        i2 = self._normalise(image2)

        feat1 = self.feature_encoder(i1)
        feat2 = self.feature_encoder(i2)
        f1 = feat1["level2"]
        f2 = feat2["level2"]

        ctx = self.context_encoder(i1)
        context_feat = ctx["context_feat"]
        hidden_state = ctx["hidden_state"]

        b, _, h, w = f1.shape
        i1_lvl = F.interpolate(i1, size=(h, w), mode="bilinear", align_corners=True)
        i2_lvl = F.interpolate(i2, size=(h, w), mode="bilinear", align_corners=True)
        context_feat_lvl = F.interpolate(context_feat, size=(h, w), mode="nearest")
        hidden_lvl = F.interpolate(hidden_state, size=(h, w), mode="nearest")

        if flow_init is None:
            flow_yx = torch.zeros(b, 2, h, w, device=f1.device, dtype=f1.dtype)
        else:
            # Input flow to this model is expected in [dx, dy] output convention, convert to [dy, dx].
            flow_yx = torch.stack([flow_init[:, 1], flow_init[:, 0]], dim=1)
            flow_yx = self._resize_flow_yx(flow_yx, h, w)

        aux_yx = torch.zeros_like(flow_yx)
        flow_preds: List[torch.Tensor] = []
        flow_lows: List[torch.Tensor] = []
        hidden_states: List[torch.Tensor] = []

        corr_allpairs = self.build_all_pairs_correlation(f1, f2)
        corr_pyr = self.build_corr_pyramid_from_all_pairs(corr_allpairs, num_levels=self.num_corr_levels)
        coords1 = self._coords_grid(b, h, w, device=f1.device, dtype=f1.dtype)

        coords_query = coords1 + flow_yx.permute(0, 2, 3, 1)
        corr_feat = self.sample_all_pairs_corr_pyramid(corr_pyr, coords_query, radius=self.max_displacement)
        delta_init, confidence, _ = self.build_all_pairs_init_from_lookup(corr_feat, flow_yx)
        flow_yx = flow_yx + delta_init
        aux_yx = flow_yx.clone()

        for k in range(iters):
            beta = F.softplus(self.hqs_beta[k]) + 1e-4
            lam = F.softplus(self.hqs_lambda[k]) + 1e-4

            coords_query = coords1 + flow_yx.permute(0, 2, 3, 1)
            corr_feat_k = self.sample_all_pairs_corr_pyramid(corr_pyr, coords_query, radius=self.max_displacement)
            _, confidence_k, _ = self.build_all_pairs_init_from_lookup(corr_feat_k, flow_yx)

            ix, iy, it, _ = self.compute_ofce_derivatives(i1_lvl, i2_lvl, flow_yx)
            flow_yx, _, _ = self.hqs_data_step(ix, iy, it, aux_yx, confidence_k, flow_yx, beta)
            aux_yx, _, _ = self.hqs_prox_step(flow_yx, i1_lvl, beta, lam, num_iter=1)

            # Match TF step refinement input: [flow, aux, F1, context_feat]
            refine_in = torch.cat([flow_yx, aux_yx, f1, context_feat_lvl], dim=1)
            delta_refine = self.refinement_network(refine_in)
            flow_yx = flow_yx + 0.1 * delta_refine

            # Save per-stage outputs for sequence supervision.
            flow_low_xy = torch.stack([flow_yx[:, 1], flow_yx[:, 0]], dim=1)
            flow_stage_up_yx = self._resize_flow_yx(flow_yx, image1.shape[-2], image1.shape[-1])
            flow_stage_up_xy = torch.stack([flow_stage_up_yx[:, 1], flow_stage_up_yx[:, 0]], dim=1)

            flow_lows.append(flow_low_xy)
            flow_preds.append(flow_stage_up_xy)
            hidden_states.append(hidden_lvl)

        final_in = torch.cat([flow_yx, context_feat_lvl, hidden_lvl], dim=1)
        final_mask_logits = self.final_mask_logits(final_in)
        flow_yx = flow_yx + 0.1 * self.final_refinement_network(final_in)
        flow_up_yx = self.convex_upsample(flow_yx, final_mask_logits, rate=self.upsample_scale)

        # If spatial mismatch due odd dimensions, resize with proper vector scaling.
        if flow_up_yx.shape[-2:] != image1.shape[-2:]:
            flow_up_yx = self._resize_flow_yx(flow_up_yx, image1.shape[-2], image1.shape[-1])

        # Convert internal [dy, dx] to repository output convention [dx, dy].
        flow_up_xy = torch.stack([flow_up_yx[:, 1], flow_up_yx[:, 0]], dim=1)
        flow_low_xy_final = torch.stack([flow_yx[:, 1], flow_yx[:, 0]], dim=1)

        # Keep prediction-list length == iters while making the final element
        # the best TF-faithful refined output.
        if flow_preds:
            flow_preds[-1] = flow_up_xy
            flow_lows[-1] = flow_low_xy_final
        else:
            flow_preds = [flow_up_xy]
            flow_lows = [flow_low_xy_final]
            hidden_states = [hidden_lvl]

        return {
            "flow_preds": flow_preds,
            "flow_low": flow_lows,
            "hidden_states": hidden_states,
        }

    def param_count(self) -> Dict[str, int]:
        def count(module: nn.Module) -> int:
            return sum(p.numel() for p in module.parameters())

        return {
            "feature_encoder": count(self.feature_encoder),
            "context_encoder": count(self.context_encoder),
            "stages": count(self.refinement_network) + count(self.final_refinement_network),
            "total": count(self),
        }
