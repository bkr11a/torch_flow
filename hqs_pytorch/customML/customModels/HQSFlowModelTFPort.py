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


def _inv_sigmoid(y: float) -> float:
    y = min(max(y, 1e-4), 1.0 - 1e-4)
    return math.log(y / (1.0 - y))

# ---------------------------------------------------------------------------
# Motion Encoder:
# ---------------------------------------------------------------------------

class MotionEncoder(nn.Module):
    """
    Encodes sampled correlation features, the current flow estimate, and the
    HQS coupling residual (flow_yx - aux_yx) into a compact motion feature
    consumed by the ConvGRU update unit.

    The HQS residual is a physics-informed cue: when it is large the data
    step and the prox step disagree, signalling that the update unit should
    apply a stronger correction.

    Inputs (all at encoder resolution, typically 1/8):
        corr_feat : (B, corr_channels, H, W)  sampled correlation pyramid
        flow_yx   : (B, 2, H, W)              current flow [dy, dx]
        hqs_resid : (B, 2, H, W)              flow_yx - aux_yx
    Output:
        (B, motion_dim, H, W)
    """

    def __init__(self, corr_channels: int, motion_dim: int = 128) -> None:
        super().__init__()
        self.corr_proj = nn.Sequential(
            nn.Conv2d(corr_channels, 256, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 192, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.flow_proj = nn.Sequential(
            nn.Conv2d(5, 64, kernel_size=7, padding=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.fusion = nn.Sequential(
            nn.Conv2d(192 + 64, motion_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.motion_dim = motion_dim

    def forward(
        self,
        corr_feat: torch.Tensor,   # (B, corr_channels, H, W)
        flow_yx: torch.Tensor,     # (B, 2, H, W)
        hqs_resid: torch.Tensor,   # (B, 2, H, W)  flow_yx - aux_yx
        confidence: torch.Tensor,       # (B, 1, H, W)  from iter_conf_head
    ) -> torch.Tensor:
        c = self.corr_proj(corr_feat)
        f = self.flow_proj(torch.cat([flow_yx, hqs_resid, confidence], dim=1))
        return self.fusion(torch.cat([c, f], dim=1))

# ---------------------------------------------------------------------------
# PriorMotionEncoder
# ---------------------------------------------------------------------------
class PriorMotionEncoder(nn.Module):
    """
    Source-context / prior motion encoder.

    This branch deliberately receives no correlation features.
    It can update flow using:
      - current flow
      - HQS residual w - q
      - data reliability/confidence
      - photometric residual
      - warp validity

    The intention is to provide an inpainting / regularising update path
    for regions where target-frame matching evidence is unreliable.
    """

    def __init__(self, motion_dim: int = 128) -> None:
        super().__init__()

        # Inputs:
        #   flow_yx      : 2
        #   hqs_resid    : 2
        #   confidence   : 1
        #   it           : 1
        #   valid        : 1
        # total = 7
        self.prior_proj = nn.Sequential(
            nn.Conv2d(7, 64, kernel_size=7, padding=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 96, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(96, motion_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.motion_dim = motion_dim

    def forward(
        self,
        flow_yx: torch.Tensor,
        hqs_resid: torch.Tensor,
        confidence: torch.Tensor,
        it: torch.Tensor,
        valid: torch.Tensor,
    ) -> torch.Tensor:
        x = torch.cat([flow_yx, hqs_resid, confidence, it, valid], dim=1)
        return self.prior_proj(x)

# ---------------------------------------------------------------------------
# Reliability Gate
# ---------------------------------------------------------------------------
class ReliabilityGate(nn.Module):
    """
    Predicts alpha in [alpha_min, alpha_max].

    alpha close to 1:
        trust correlation/match branch.

    alpha close to 0:
        trust source-context/prior branch.

    This gate deliberately does not consume raw correlation features.
    It uses reliability summaries instead.
    """

    def __init__(
        self,
        alpha_min: float = 0.05,
        alpha_max: float = 0.95,
        init_alpha: float = 0.75,
    ) -> None:
        super().__init__()

        self.alpha_min = float(alpha_min)
        self.alpha_max = float(alpha_max)

        # Inputs:
        #   flow_yx    : 2
        #   hqs_resid  : 2
        #   confidence : 1
        #   abs(it)    : 1
        #   valid      : 1
        # total = 7
        self.net = nn.Sequential(
            nn.Conv2d(7, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=3, padding=1),
        )

        # Initialise gate to mostly trust the match branch at the start.
        # This avoids destroying convergence at early training.
        p = (init_alpha - alpha_min) / max(alpha_max - alpha_min, 1e-6)
        p = min(max(p, 1e-4), 1.0 - 1e-4)
        bias = math.log(p / (1.0 - p))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.constant_(self.net[-1].bias, bias)

    def forward(
        self,
        flow_yx: torch.Tensor,
        hqs_resid: torch.Tensor,
        confidence: torch.Tensor,
        it: torch.Tensor,
        valid: torch.Tensor,
    ) -> torch.Tensor:
        x = torch.cat([flow_yx, hqs_resid, confidence, it.abs(), valid], dim=1)
        raw = torch.sigmoid(self.net(x))
        return self.alpha_min + (self.alpha_max - self.alpha_min) * raw

# ---------------------------------------------------------------------------
# Data Reliability Head
# ---------------------------------------------------------------------------
class DataReliabilityHead(nn.Module):
    """
    Predicts a soft visibility/reliability mask for the OFCE data term.

    Output:
        m_data in [m_min, 1]
        high  -> trust warped I2 residual
        low   -> suppress warped I2 residual and rely on prox/prior update
    """

    def __init__(self, in_channels: int = 6, hidden: int = 32, m_min: float = 0.02):
        super().__init__()
        self.m_min = float(m_min)

        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, 1, kernel_size=3, padding=1),
        )

        # Start close to trusting the data term, so training does not collapse.
        nn.init.zeros_(self.net[-1].weight)
        nn.init.constant_(self.net[-1].bias, 2.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raw = torch.sigmoid(self.net(x))
        return self.m_min + (1.0 - self.m_min) * raw

# ---------------------------------------------------------------------------
# ConvGRU Cell
# ---------------------------------------------------------------------------

class ConvGRUCell(nn.Module):
    """
    Single-step convolutional GRU cell.

    Args:
        input_dim  : channels of input x
        hidden_dim : channels of hidden state h
    """

    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.conv_z = nn.Conv2d(input_dim + hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.conv_r = nn.Conv2d(input_dim + hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.conv_h = nn.Conv2d(input_dim + hidden_dim, hidden_dim, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        xh = torch.cat([x, h], dim=1)
        z = torch.sigmoid(self.conv_z(xh))
        r = torch.sigmoid(self.conv_r(xh))
        h_cand = torch.tanh(self.conv_h(torch.cat([x, r * h], dim=1)))
        return (1.0 - z) * h + z * h_cand

# ---------------------------------------------------------------------------
# Learned Proximal Operator
# ---------------------------------------------------------------------------

class LearnedHQSProx(nn.Module):
    def __init__(
        self,
        context_dim: int,
        hidden_dim: int = 96,
        prox_scale: float = 1.0,
    ):
        super().__init__()
        self.prox_scale = prox_scale

        # Inputs:
        # flow_yx       : 2
        # prev_aux_yx   : 2
        # flow-prev_aux : 2
        # validity      : 1
        # source context: context_dim
        in_ch = 2 + 2 + 2 + 1 + context_dim

        self.encoder = nn.Sequential(
            nn.Conv2d(in_ch, hidden_dim, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.anchor_head = nn.Conv2d(hidden_dim, 2, 3, padding=1)
        self.prior_head = nn.Conv2d(hidden_dim, 2, 3, padding=1)

        self.mix_head = nn.Sequential(
            nn.Conv2d(hidden_dim, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, 3, padding=1),
            nn.Sigmoid(),
        )

        # Start as near-identity prox.
        nn.init.zeros_(self.anchor_head.weight)
        nn.init.zeros_(self.anchor_head.bias)
        nn.init.zeros_(self.prior_head.weight)
        nn.init.zeros_(self.prior_head.bias)

    def forward(
        self,
        flow_yx: torch.Tensor,
        prev_aux_yx: torch.Tensor,
        context_feat: torch.Tensor,
        validity: torch.Tensor,
    ):
        hqs_resid = flow_yx - prev_aux_yx

        x = torch.cat(
            [
                flow_yx,
                prev_aux_yx,
                hqs_resid,
                validity,
                context_feat,
            ],
            dim=1,
        )

        h = self.encoder(x)

        delta_anchor = self.prox_scale * torch.tanh(self.anchor_head(h))
        delta_prior = self.prox_scale * torch.tanh(self.prior_head(h))

        q_anchor = flow_yx + delta_anchor
        q_prior = prev_aux_yx + delta_prior

        learned_mix = self.mix_head(h)

        # Bias mixture toward geometric/learned validity.
        # High validity -> anchor to flow.
        # Low validity -> prior/inpaint from previous aux/source context.
        mix = 0.5 * validity + 0.5 * learned_mix

        aux_yx = mix * q_anchor + (1.0 - mix) * q_prior

        return aux_yx, mix

# ---------------------------------------------------------------------------
# HQS Update Unit  (ConvGRU + flow head + mask head)
# ---------------------------------------------------------------------------


class HQSUpdateUnit(nn.Module):
    """
    Per-iteration update unit that maintains a hidden state across HQS
    iterations, predicts a residual flow correction delta, and produces
    convex-upsampling mask logits for full-resolution output.

    Maintaining hidden state allows the unit to distinguish odd (data-step)
    from even (prox-step) iterations, eliminating the even/odd oscillation
    produced by a stateless conv applied to alternating HQS states.

    Args:
        motion_dim    : channels from MotionEncoder
        context_dim   : channels from context encoder (concatenated with motion)
        hidden_dim    : GRU hidden state channels
        upsample_scale: spatial upsampling factor (determines mask output channels)
    """

    def __init__(
        self,
        motion_dim: int,
        context_dim: int,
        hidden_dim: int,
        upsample_scale: int = 8,
    ) -> None:
        super().__init__()
        input_dim = motion_dim + context_dim
        self.gru = ConvGRUCell(input_dim, hidden_dim)
        self.flow_head = nn.Sequential(
            nn.Conv2d(hidden_dim, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 2, kernel_size=3, padding=1),
        )
        nn.init.zeros_(self.flow_head[-1].weight)
        nn.init.zeros_(self.flow_head[-1].bias)
        # Image-guided upsampling: mask head fuses GRU hidden state with raw
        # context features so the upsampling can attend directly to image
        # structure (edges, boundaries) rather than only to the recurrent state.
        self.mask_head = nn.Sequential(
            nn.Conv2d(hidden_dim + context_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, upsample_scale * upsample_scale * 9, kernel_size=3, padding=1),
        )
        self._context_dim = context_dim

    def forward(
        self,
        motion_feat: torch.Tensor,
        context_feat: torch.Tensor,
        net: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = torch.cat([motion_feat, context_feat], dim=1)
        net = self.gru(x, net)
        delta = self.flow_head(net)
        # Fuse GRU state with context features for image-aware upsampling weights.
        mask_logits = self.mask_head(torch.cat([net, context_feat], dim=1))
        return delta, mask_logits, net


class HQSFlowModelTFPort(nn.Module):
    """TensorFlow-faithful HQS optical flow model in PyTorch."""

    def __init__(self, cfg) -> None:
        super().__init__()
        self.cfg = cfg

        mb = _cfg_get(cfg, "model_backbone", {})
        legacy_num_stages = _cfg_get(cfg, "num_stages", None)
        self.num_hqs_iterations = int(
            _cfg_get(mb, "num_hqs_iterations", legacy_num_stages if legacy_num_stages is not None else 10)
        )
        if legacy_num_stages is not None and int(legacy_num_stages) != self.num_hqs_iterations:
            raise ValueError(
                "model.num_stages and model.model_backbone.num_hqs_iterations disagree. "
                "Use model.model_backbone.num_hqs_iterations as the authoritative setting."
            )
        self.upsample_scale = int(_cfg_get(cfg, "upsample_scale", 8))

        corr_cfg = _cfg_get(cfg, "corr", {})
        self.max_displacement = int(_cfg_get(mb, "max_displacement", _cfg_get(corr_cfg, "radius", 4)))
        self.num_corr_levels = int(_cfg_get(mb, "num_corr_levels", _cfg_get(corr_cfg, "num_levels", 4)))
        self.prox_jacobi_iters = int(_cfg_get(mb, "prox_jacobi_iters", 3))
        self.refine_level1_iters = int(_cfg_get(mb, "refine_level1_iters", 0))

        base_channels = int(_cfg_get(mb, "base_channels", 32))
        channel_multiplier = tuple(_cfg_get(mb, "channel_multiplier", (1, 2, 4)))
        blocks_per_stage = tuple(_cfg_get(mb, "blocks_per_stage", (2, 2, 2)))
        groups = int(_cfg_get(mb, "groups", 8))
        dropout_rate = float(_cfg_get(mb, "dropout_rate", 0.0))
        feature_dim = int(_cfg_get(mb, "feature_dim", 128))
        context_dim = int(_cfg_get(mb, "context_dim", 128))
        hidden_dim = int(_cfg_get(mb, "hidden_dim", 64))       # context encoder seed dim
        gru_hidden_dim = int(_cfg_get(mb, "gru_hidden_dim", 128))  # GRU state channels
        motion_dim = int(_cfg_get(mb, "motion_dim", 128))          # motion encoder output
        self._gru_hidden_dim = gru_hidden_dim  # stored for param_count

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

        # Bounded monotone HQS schedules reduce late-iteration instability while
        # preserving the unrolled physics-inspired structure.
        self.beta_min = float(_cfg_get(mb, "beta_min", 0.05))
        self.beta_max = float(_cfg_get(mb, "beta_max", 0.50))
        self.lambda_ratio_min = float(_cfg_get(mb, "lambda_ratio_min", 0.25))
        self.lambda_ratio_max = float(_cfg_get(mb, "lambda_ratio_max", 0.45))
        self.correction_gate_min = float(_cfg_get(mb, "correction_gate_min", 0.05))
        self.correction_gate_max = float(_cfg_get(mb, "correction_gate_max", 0.50))
        self.correction_gate_nonincreasing = bool(
            _cfg_get(mb, "correction_gate_nonincreasing", True)
        )
        self.freeze_correction_last_n = int(_cfg_get(mb, "freeze_correction_last_n", 1))

        self.hqs_beta_steps = nn.Parameter(torch.zeros(self.num_hqs_iterations))
        self.hqs_lambda_ratio_steps = nn.Parameter(torch.zeros(self.num_hqs_iterations))
        self.delta_gate_logits = nn.Parameter(
            torch.full((self.num_hqs_iterations,), _inv_sigmoid(0.40))
        )

        # ── Level-1 pyramid schedule parameters (2.1 Multi-Scale HQS) ────────
        # Dedicated learnable penalty and gate schedules for the level1 (1/4-scale)
        # refinement loop, independent of the main loop's schedule.
        self.pyramid_l1_iters = int(_cfg_get(mb, "pyramid_l1_iters", 0))
        if self.pyramid_l1_iters > 0:
            self.hqs_beta_steps_l1 = nn.Parameter(torch.zeros(self.pyramid_l1_iters))
            self.hqs_lambda_ratio_steps_l1 = nn.Parameter(torch.zeros(self.pyramid_l1_iters))
            self.delta_gate_logits_l1 = nn.Parameter(
                torch.full((self.pyramid_l1_iters,), _inv_sigmoid(0.40))
            )

        corr_channels = self.num_corr_levels * (2 * self.max_displacement + 1) ** 2

        # Initialization decoder: called once before the loop at zero-flow.
        # Not reused inside the loop to avoid distribution mismatch.
        self.allpairs_corr_decoder = nn.Sequential(
            nn.Conv2d(corr_channels, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.allpairs_init_head = nn.Conv2d(64 + 2, 2, kernel_size=3, padding=1)
        nn.init.zeros_(self.allpairs_init_head.weight)
        nn.init.zeros_(self.allpairs_init_head.bias)
        self.allpairs_conf_head = nn.Sequential(
            nn.Conv2d(64, 1, kernel_size=3, padding=1), nn.Sigmoid()
        )

        # Per-iteration confidence head: trained on localized correlation
        # features sampled around the current flow estimate, not at zero-offset.
        self.iter_conf_head = nn.Sequential(
            nn.Conv2d(corr_channels+2+1+1+1+1, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )

        # Project context encoder seed → GRU hidden state dimension.
        self.hidden_proj = nn.Conv2d(hidden_dim, gru_hidden_dim, kernel_size=1)

        # ---------------------------------------------------------------------
        # Split update option:
        #   match branch: sees correlation
        #   prior branch: does not see correlation
        # ---------------------------------------------------------------------
        self.use_split_delta_update = bool(
            _cfg_get(mb, "use_split_delta_update", True)
        )

        self.detach_reliability_inputs = bool(
            _cfg_get(mb, "detach_reliability_inputs", True)
        )

        self.reliability_alpha_min = float(
            _cfg_get(mb, "reliability_alpha_min", 0.05)
        )
        self.reliability_alpha_max = float(
            _cfg_get(mb, "reliability_alpha_max", 0.95)
        )
        self.reliability_init_alpha = float(
            _cfg_get(mb, "reliability_init_alpha", 0.75)
        )

        # Match branch: correlation + flow + HQS residual + confidence.
        self.motion_encoder = MotionEncoder(corr_channels, motion_dim)

        # Existing update unit becomes the match update unit.
        self.update_unit = HQSUpdateUnit(
            motion_dim, context_dim, gru_hidden_dim, self.upsample_scale
        )

        if self.use_split_delta_update:
            # Prior branch: no correlation features.
            self.prior_motion_encoder = PriorMotionEncoder(motion_dim)

            self.prior_update_unit = HQSUpdateUnit(
                motion_dim, context_dim, gru_hidden_dim, self.upsample_scale
            )

            self.reliability_gate = ReliabilityGate(
                alpha_min=self.reliability_alpha_min,
                alpha_max=self.reliability_alpha_max,
                init_alpha=self.reliability_init_alpha,
            )

        # Final post-loop refinement (uses context encoder hidden output directly).
        self.final_refinement_network = nn.Sequential(
            nn.Conv2d(2 + context_dim + hidden_dim, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 2, kernel_size=3, padding=1),
        )
        nn.init.zeros_(self.final_refinement_network[-1].weight)
        nn.init.zeros_(self.final_refinement_network[-1].bias)
        self.final_mask_logits = nn.Conv2d(
            2 + context_dim + hidden_dim,
            self.upsample_scale * self.upsample_scale * 9,
            kernel_size=3,
            padding=1,
        )

        self.transition_upsample_rate = 2
        self.level1_transition_mask = nn.Sequential(
            nn.Conv2d(2 + feature_dim + feature_dim, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, self.transition_upsample_rate * self.transition_upsample_rate * 9, kernel_size=3, padding=1),
        )

        self.use_data_reliability_mask = bool(
            _cfg_get(mb, "use_data_reliability_mask", True)
        )

        self.detach_data_reliability_inputs = bool(
            _cfg_get(mb, "detach_data_reliability_inputs", True)
        )

        self.data_reliability_m_min = float(
            _cfg_get(mb, "data_reliability_m_min", 0.02)
        )

        self.data_reliability_hidden_dim = int(
            _cfg_get(mb, "data_reliability_hidden_dim", 32)
        )

        # Inputs:
        # abs(it)      : 1
        # grad_mag     : 1
        # valid        : 1
        # confidence   : 1
        # ||w-q||      : 1
        # ||flow||     : 1
        # total = 6
        self.data_reliability_head = DataReliabilityHead(
            in_channels=6,
            hidden=self.data_reliability_hidden_dim,
            m_min=self.data_reliability_m_min,
        )

        self.use_learned_prox_net = bool(
            _cfg_get(mb, "use_learned_proximal", False)
        )

        if self.use_learned_prox_net:
            self.learned_prox = LearnedHQSProx(
                context_dim=context_dim,
                hidden_dim=int(_cfg_get(mb, "prox_hidden_dim", 96)),
                prox_scale=float(_cfg_get(mb, "prox_scale_factor", 1.0)),
            )

        else:
            self.learned_prox = None

        # Masked OFCE warp: option to mask the warping function with a validity mask to prevent corrupting features with out-of-bounds or invalid pixel within the warps.
        self.use_masked_ofce_warp = bool(
            _cfg_get(mb, "use_masked_ofce_warp", True)
        )

        # Use self occupancy mask in the OFCE warp to prevent warping from or to self-occluded pixels, which can produce large errors and instability. This is a relaxation of the masked OFCE warp, where only out-of-bounds warps are masked.
        self.use_self_occupancy_mask = bool(
            _cfg_get(mb, "use_self_occupancy_mask", True)
        )

        # Settings for the photometric confidence settings
        self.use_photometric_confidence = bool(
            _cfg_get(mb, "use_photometric_confidence", True)
        )
        self.photometric_confidence_tau = float(
            _cfg_get(mb, "photometric_confidence_tau", 0.25)
        )

        # Settings for inputs to the OFCE. Use either the photo for ofce and/or greyscale (since OFCE is an illumination model).
        self.use_photo_for_ofce = bool(
            _cfg_get(mb, "use_photo_for_ofce", True)
        )

        self.ofce_greyscale = bool(
            _cfg_get(mb, "ofce_greyscale", True)
        )

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
        return F.grid_sample(x, grid, mode="bilinear", padding_mode="zeros", align_corners=True)

    @staticmethod
    def _valid_warp_mask(flow_yx: torch.Tensor) -> torch.Tensor:
        """Warping near image boundaries produces invalid values. This mask identifies valid pixels."""
        b, _, h, w = flow_yx.shape
        ys, xs = torch.meshgrid(
            torch.arange(h, device=flow_yx.device, dtype=flow_yx.dtype),
            torch.arange(w, device=flow_yx.device, dtype=flow_yx.dtype),
            indexing="ij",
        )
        qy = ys.view(1, h, w) + flow_yx[:, 0]
        qx = xs.view(1, h, w) + flow_yx[:, 1]

        valid = (qx >= 0) & (qx <= w - 1) & (qy >= 0) & (qy <= h - 1)
        return valid.float().unsqueeze(1)  # [B, 1, H, W]

    @staticmethod
    def _photo_tensor(img: torch.Tensor) -> torch.Tensor:
        """
        Convert input image to [0, 1] photometric tensor.

        This is intentionally different from the ImageNet Normalisation.
        Feature encoders and correlation are trained on ImageNet-normalised features, but the OFCE is an illumination model that operates on raw photometric values, so we provide a separate normalisation here. This ensures that the residual magnitudes remain interpretable and consistent with the assumptions of the OFCE, even as the feature encoder architecture and training evolve.
        """
        if img.max() > 2.0:
            img = img / 255.0
        return img.clamp(0.0, 1.0)

    @staticmethod
    def _to_greyscale(img: torch.Tensor) -> torch.Tensor:
        """Convert RGB image to grayscale using standard luminance formula."""
        if img.shape[1] == 1:
            return img  # Already grayscale
        
        coeffs = torch.tensor([0.299, 0.587, 0.114], device=img.device, dtype=img.dtype).view(1, 3, 1, 1)
        gray = (img * coeffs).sum(dim=1, keepdim=True)
        return gray

    @staticmethod
    def _resize_mask(
        mask: torch.Tensor,
        out_h: int,
        out_w: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Optional[torch.Tensor]:
        """
        Resize a validity / confidence mask to a pyramid level.

        Accepts:
            - None
            - Tensor of shape [B, H, W]
            - Tensor of shape [B, 1, H, W]
            - Tensor of shape [B, C, H, W], which is averaged across channels to produce a single-channel mask.

        Returns:
            - None if input was None
            - Tensor of shape [B, 1, out_h, out_w] in [0, 1] if input was a mask
        """
        if mask is None:
            return None

        if mask.ndim == 3:
            mask = mask.unsqueeze(1)  # [B, 1, H, W]
        elif mask.ndim != 4:
            raise ValueError(f"mask must have shape [B, H, W] or [B, C, H, W], got {mask.shape}")

        mask = mask.to(device=device, dtype=dtype)

        if mask.ndim == 4 and mask.shape[1] > 1:
            mask = mask.mean(dim=1, keepdim=True)  # Average across channels

        # Handle uint8 masks with values in [0, 255].
        if mask.max() > 2.0:
            mask = mask / 255.0

        mask = mask.clamp(0.0, 1.0)

        mask = F.interpolate(
            mask, 
            size=(out_h, out_w), 
            mode="bilinear", 
            align_corners=True
            )

        return mask

    def masked_warp_yx(
            self,
            x: torch.Tensor,
            flow_yx: torch.Tensor,
            target_valid: Optional[torch.Tensor] = None,
            eps: float = 1e-6,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Validity aware backward warp.

        For forward flow w_(1->2), this samples x2 at x + w(x).

        If target_valid is provided, it must live in image-2 coordinates.
        The function warps both:
            target_valid
            target_valid * x (to produce a masked feature map where invalid pixels are zeroed out)
        
        and returns an alpha-normalised warped image:

            warp(target_valid*x) / (warp(target_valid) + eps)

        This prevents masked target pixels from being blended into valid pixels by the warping operation.

        Returns:
            x_warp        : warped x with shape [B, C, H, W]
            target_warp_valid : warped target_valid with shape [B, 1, H, W]
            bounds_valid    : validity mask for warps that go out of bounds, shape [B, 1, H, W]
        """
        b, _, h, w = x.shape

        bounds_valid = self._valid_warp_mask(flow_yx).to(device=x.device, dtype=x.dtype)

        if target_valid is None:
            x_warp = self._warp_yx(x, flow_yx)
            return x_warp, bounds_valid, bounds_valid

        target_valid = self._resize_mask(
            target_valid,
            out_h=h,
            out_w=w,
            device=x.device,
            dtype=x.dtype,
        )

        target_warp_valid = self._warp_yx(target_valid, flow_yx)
        target_warp_valid = (target_warp_valid * bounds_valid).clamp(0.0, 1.0)  # Ensure out-of-bounds warps are invalid.

        numerator = self._warp_yx(x * target_valid, flow_yx)
        denominator = target_warp_valid.clamp_min(eps)
        x_warp = numerator / denominator
        x_warp = torch.where(
            target_warp_valid > eps,
            x_warp,
            torch.zeros_like(x_warp)
        )

        return x_warp, target_warp_valid, bounds_valid

    @staticmethod
    def build_target_valid_from_flow_yx(
        flow_yx: torch.Tensor,
        source_valid: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Build a target-frame coverage mask by forward splatting source validity through the forward flow.
        
        USE THIS WITH GROUND TRUTH FLOW during training / debugging.

        flow_yx: [B, 2, H, W] forward flow from image 1 to image 2 in [dy, dx] order.
        source_valid: [B, 1, H, W] or None. 1 means the frame-1 pixel is valid/matchable.

        Returns:
            target_valid: [B, 1, H, W], in image-2 coordinates. 1 means some valid source pixel maps to this target location.
        """
        if source_valid is None:
            source_valid = torch.ones_like(flow_yx[:, :1])  # [B, 1, H, W]
        elif source_valid.ndim == 3:
            source_valid = source_valid.unsqueeze(1)  # [B, 1, H, W]

        source_valid = source_valid.to(device=flow_yx.device, dtype=flow_yx.dtype).clamp(0.0, 1.0)

        target_valid = HQSFlowModelTFPort.forward_splat(
            src=source_valid,
            flow_yx=flow_yx,
            normalize=False,
        )

        return target_valid.clamp(0.0, 1.0)

    @staticmethod
    def forward_splat(
        src: torch.Tensor,
        flow_yx: torch.Tensor,
        normalize: bool = False,
        eps: float = 1e-6,
    ) -> torch.Tensor:
        """
        Differentiable forward splatting with bilinear weights.

        Args:
            src:
                Tensor to splat from source/image-1 coordinates.
                Shape: (B, C, H, W)

            flow_yx:
                Forward flow from image 1 to image 2.
                Channel order: [dy, dx].
                Shape: (B, 2, H, W)

            normalize:
                If False:
                    return accumulated splat values.
                    This is what you want for occupancy splatting.

                If True:
                    return splatted values divided by splatted weights.
                    This is useful if splatting images/features.

            eps:
                Numerical stability constant.

        Returns:
            out:
                Forward-splatted tensor in target/image-2 coordinates.
                Shape: (B, C, H, W)
        """

        if src.ndim != 4:
            raise ValueError(f"src must have shape (B, C, H, W), got {src.shape}")

        if flow_yx.ndim != 4 or flow_yx.shape[1] != 2:
            raise ValueError(f"flow_yx must have shape (B, 2, H, W), got {flow_yx.shape}")

        B, C, H, W = src.shape

        if flow_yx.shape[0] != B or flow_yx.shape[2:] != (H, W):
            raise ValueError(
                f"src and flow_yx spatial sizes must match. "
                f"src={src.shape}, flow_yx={flow_yx.shape}"
            )

        device = src.device
        dtype = src.dtype

        # Source pixel coordinates.
        yy, xx = torch.meshgrid(
            torch.arange(H, device=device, dtype=dtype),
            torch.arange(W, device=device, dtype=dtype),
            indexing="ij",
        )

        yy = yy[None, None, :, :]  # (1, 1, H, W)
        xx = xx[None, None, :, :]  # (1, 1, H, W)

        # Target coordinates induced by forward flow.
        y_t = yy + flow_yx[:, 0:1]
        x_t = xx + flow_yx[:, 1:2]

        # Four bilinear neighbours.
        y0 = torch.floor(y_t)
        x0 = torch.floor(x_t)
        y1 = y0 + 1.0
        x1 = x0 + 1.0

        # Bilinear weights.
        wy1 = y_t - y0
        wx1 = x_t - x0
        wy0 = 1.0 - wy1
        wx0 = 1.0 - wx1

        y0_long = y0.long()
        x0_long = x0.long()
        y1_long = y1.long()
        x1_long = x1.long()

        out = src.new_zeros(B, C, H * W)

        if normalize:
            weight_accum = src.new_zeros(B, 1, H * W)
        else:
            weight_accum = None

        def splat_to(y_idx, x_idx, weight):
            nonlocal out, weight_accum

            valid = (
                (y_idx >= 0)
                & (y_idx < H)
                & (x_idx >= 0)
                & (x_idx < W)
            )

            # Flatten target index.
            linear_idx = y_idx.clamp(0, H - 1) * W + x_idx.clamp(0, W - 1)
            linear_idx = linear_idx.view(B, 1, H * W)

            valid_f = valid.to(dtype).view(B, 1, H * W)
            weight_f = weight.view(B, 1, H * W) * valid_f

            src_weighted = src.view(B, C, H * W) * weight_f

            out.scatter_add_(
                dim=2,
                index=linear_idx.expand(B, C, H * W),
                src=src_weighted,
            )

            if normalize:
                weight_accum.scatter_add_(
                    dim=2,
                    index=linear_idx,
                    src=weight_f,
                )

        splat_to(y0_long, x0_long, wy0 * wx0)
        splat_to(y0_long, x1_long, wy0 * wx1)
        splat_to(y1_long, x0_long, wy1 * wx0)
        splat_to(y1_long, x1_long, wy1 * wx1)

        if normalize:
            out = out / weight_accum.clamp_min(eps)

        return out.view(B, C, H, W)

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

    def compute_ofce_derivatives(
            self,
            i1: torch.Tensor,
            i2: torch.Tensor,
            flow_yx: torch.Tensor,
            source_valid: Optional[torch.Tensor] = None,
            target_valid: Optional[torch.Tensor] = None,
        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        OFCE derivative computation with validity-aware warping.

        For forward flow w_(1->2), we evaluate the OFCE residual at i2(x + w(x)) - i1(x).o

        Returns: 
            ix  : dI/dx at the evaluation point, shape [B, 1, H, W]
            iy  : dI/dy at the evaluation point, shape [B, 1, H, W]
            it  : dI/dt at the evaluation point, shape [B, 1, H, W]
            i2_warp : the warped i2 at the evaluation point, shape [B, C, H, W]
            data_valid : validity mask (in image 1 coords) for the OFCE data term at the evaluation point, shape [B, 1, H, W]

        Important details:
            - The OFCE residual is only valid at pixels where the warped i2 samples from valid target pixels. We compute a validity mask for the OFCE data term that identifies these pixels, which can be used to weight the data term in the HQS update to prevent corrupting the flow with target imaage occlusions, disocclusions and other invalid pixels.
        """
        b, _, h, w = i1.shape

        gx1 = self.central_grad_x(i1)
        gy1 = self.central_grad_y(i1)

        gx2_native = self.central_grad_x(i2)
        gy2_native = self.central_grad_y(i2)

        if self.use_masked_ofce_warp:
            i2_warp, target_warp_valid, bounds_valid = self.masked_warp_yx(
                i2,
                flow_yx,
                target_valid=target_valid if self.use_self_occupancy_mask else None
            )
            gx2_warp, gx2_valid, _ = self.masked_warp_yx(
                gx2_native,
                flow_yx,
                target_valid=target_valid if self.use_self_occupancy_mask else None
            )
            gy2_warp, gy2_valid, _ = self.masked_warp_yx(
                gy2_native,
                flow_yx,
                target_valid=target_valid if self.use_self_occupancy_mask else None
            )
        else:
            i2_warp = self._warp_yx(i2, flow_yx)
            gx2_warp = self._warp_yx(gx2_native, flow_yx)
            gy2_warp = self._warp_yx(gy2_native, flow_yx)
            bounds_valid = self._valid_warp_mask(flow_yx).to(device=i1.device, dtype=i1.dtype)
            target_warp_valid = bounds_valid  # If not using masked warp, we can only guarantee validity based on bounds.

        source_valid = self._resize_mask(
            source_valid,
            out_h=h,
            out_w=w,
            device=i1.device,
            dtype=i1.dtype,
        )

        if source_valid is None:
            source_valid = torch.ones_like(bounds_valid)

        data_valid = (bounds_valid * source_valid * target_warp_valid * gx2_valid * gy2_valid).clamp(0.0, 1.0)

        ix = 0.5 * (gx1 + gx2_warp)
        iy = 0.5 * (gy1 + gy2_warp)
        it = i2_warp - i1

        if ix.shape[1] > 1:
            ix = ix.mean(dim=1, keepdim=True)
            iy = iy.mean(dim=1, keepdim=True)
            it = it.mean(dim=1, keepdim=True)

        return ix, iy, it, i2_warp, data_valid

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
        q = aux_yx[:, 0:1]
        p = aux_yx[:, 1:2]
        v0 = flow0_yx[:, 0:1]
        u0 = flow0_yx[:, 1:2]

        du_prior = p - u0
        dv_prior = q - v0

        # ----- OLD CONFIDENCE CLAMP ----- #
        # c = confidence.clamp(0.05, 1.0)
        # ------------------------------ #
        
        # New clamping strategy to ensure that invalid pixels don't influence the update causing instabilty.
        c = confidence.clamp(0.00, 1.0)

        if robust:
            # Since we've already warped i2 with the current flow estimate, the residual r is just the photometric error it.
            # r = ix * u0 + iy * v0 + it
            r = it
            wd = 1.0 / (r.square() + eps ** 2).sqrt()
        else:
            wd = torch.ones_like(ix)

        # There is a risk that low-confidence pixels with high residuals could produce extreme weights that destabilize training. Clamping the max weight can mitigate this.
        wd = wd.clamp(max=10.0)  # Cap max weight to prevent extreme updates from outliers.

        w = c * wd
        a11 = w * ix * ix + beta
        a12 = w * ix * iy
        a22 = w * iy * iy + beta
        b1 = -w * ix * it + beta * du_prior
        b2 = -w * iy * it + beta * dv_prior

        det = a11 * a22 - a12 * a12
        du = (a22 * b1 - a12 * b2) / (det + 1e-6)
        dv = (-a12 * b1 + a11 * b2) / (det + 1e-6)
        
        u = u0 + du
        v = v0 + dv

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
        validity_mask: torch.Tensor,
    ) -> torch.Tensor:
        x_up = F.pad(x[:, :, :-1, :], (0, 0, 1, 0), mode="replicate")
        x_down = F.pad(x[:, :, 1:, :], (0, 0, 0, 1), mode="replicate")
        x_left = F.pad(x[:, :, :, :-1], (1, 0, 0, 0), mode="replicate")
        x_right = F.pad(x[:, :, :, 1:], (0, 1, 0, 0), mode="replicate")

        wx_l = F.pad(wx, (1, 0, 0, 0), mode="replicate")
        wx_r = F.pad(wx, (0, 1, 0, 0), mode="replicate")
        wy_u = F.pad(wy, (0, 0, 1, 0), mode="replicate")
        wy_d = F.pad(wy, (0, 0, 0, 1), mode="replicate")

        data_weight = validity_mask * beta
        smooth_weight = lam * (wy_u + wy_d + wx_l + wx_r)

        numer = data_weight * rhs + lam * (wy_u * x_up + wy_d * x_down + wx_l * x_left + wx_r * x_right)
        denom = data_weight + smooth_weight
        return numer / (denom + 1e-6)

    def hqs_prox_step(
        self,
        flow_yx: torch.Tensor,
        image: torch.Tensor,
        beta: torch.Tensor,
        lam: torch.Tensor,
        validity_mask: torch.Tensor,
        num_iter: int = 1,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        q = flow_yx[:, 0:1].clone()
        p = flow_yx[:, 1:2].clone()
        wx, wy = self.edge_weights(image)
        for _ in range(num_iter):
            q = self.jacobi_prox_step(q, flow_yx[:, 0:1], beta, lam, wx, wy, validity_mask=validity_mask)
            p = self.jacobi_prox_step(p, flow_yx[:, 1:2], beta, lam, wx, wy, validity_mask=validity_mask)
        return torch.cat([q, p], dim=1), wx, wy

    @staticmethod
    def _monotone_schedule(raw_steps: torch.Tensor, lower: float, upper: float) -> torch.Tensor:
        steps = F.softplus(raw_steps) + 1e-4
        cumulative = steps.cumsum(dim=0)
        normalized = cumulative / cumulative[-1].clamp_min(1e-6)
        return lower + (upper - lower) * normalized

    def _compute_hqs_penalties_from(
        self,
        beta_steps: torch.Tensor,
        lambda_ratio_steps: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generic HQS penalty computation, reused across pyramid scales."""
        beta = self._monotone_schedule(beta_steps, self.beta_min, self.beta_max)
        lam_ratio = self._monotone_schedule(
            lambda_ratio_steps,
            self.lambda_ratio_min,
            self.lambda_ratio_max,
        )
        return beta, beta * lam_ratio

    def _compute_correction_gates_from(self, gate_logits: torch.Tensor) -> torch.Tensor:
        """Generic correction gate schedule, reused across pyramid scales.
        Freeze logic is applied by the caller in the forward pass."""
        if not self.correction_gate_nonincreasing:
            gates = torch.sigmoid(gate_logits)
            return self.correction_gate_min + (self.correction_gate_max - self.correction_gate_min) * gates
        steps = F.softplus(gate_logits) + 1e-4
        cumulative = steps.cumsum(dim=0)
        normalized = cumulative / cumulative[-1].clamp_min(1e-6)
        return self.correction_gate_max - (
            self.correction_gate_max - self.correction_gate_min
        ) * normalized

    def _hqs_penalties(self) -> Tuple[torch.Tensor, torch.Tensor]:
        return self._compute_hqs_penalties_from(self.hqs_beta_steps, self.hqs_lambda_ratio_steps)

    def _correction_gates(self) -> torch.Tensor:
        return self._compute_correction_gates_from(self.delta_gate_logits)

    @staticmethod
    def convex_upsample(flow_lr: torch.Tensor, mask_logits: torch.Tensor, rate: int = 8) -> torch.Tensor:
        """RAFT-style convex upsampling, matching TensorFlow logic."""
        b, _, h, w = flow_lr.shape
        mask = mask_logits.view(b, 1, 9, rate, rate, h, w)
        mask = torch.softmax(mask, dim=2)

        up_flow = F.unfold(rate * flow_lr, kernel_size=3, padding=1)
        up_flow = up_flow.view(b, 2, 9, 1, 1, h, w)
        up_flow = (mask * up_flow).sum(dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3).reshape(b, 2, rate * h, rate * w)
        return up_flow

    def convex_transition_upsample_yx(
        self,
        flow_yx: torch.Tensor,
        f1_l2: torch.Tensor,
        f1_l1: torch.Tensor,
        f2_l1: torch.Tensor,
    ) -> torch.Tensor:
        # Build low-res conditioning features at level-2 resolution.
        f1_l1_down = F.interpolate(
            f1_l1,
            size=flow_yx.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        f2_l1_down = F.interpolate(
            f2_l1,
            size=flow_yx.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        mask_in = torch.cat([flow_yx, f1_l1_down, f2_l1_down], dim=1)
        mask_logits = self.level1_transition_mask(mask_in)

        flow_up = self.convex_upsample(flow_yx, mask_logits, rate=self.transition_upsample_rate)

        # Safely resize if shapes are not exactly 2x due to odd dimensions.
        if flow_up.shape[-2:] != f1_l1.shape[-2:]:
            flow_up = self._resize_flow_yx(flow_up, f1_l1.shape[-2], f1_l1.shape[-1])

        return flow_up

    def _resize_flow_yx(self, flow: torch.Tensor, out_h: int, out_w: int) -> torch.Tensor:
        in_h, in_w = flow.shape[-2:]
        if in_h == out_h and in_w == out_w:
            return flow
        resized = F.interpolate(flow, size=(out_h, out_w), mode="bilinear", align_corners=True)
        sy = float(out_h) / float(in_h)
        sx = float(out_w) / float(in_w)
        scale = flow.new_tensor([sy, sx]).view(1, 2, 1, 1)
        return resized * scale

    def photometric_confidence(self, it: torch.Tensor, tau: float = 0.25) -> torch.Tensor:
        """
        Soft mask confidence based from photometric residual, used to emulate occlusions.
        """
        return torch.exp(-it.abs() / tau).clamp(0.0, 1.0)

    def forward(
        self,
        image1: torch.Tensor,
        image2: torch.Tensor,
        iters: Optional[int] = None,
        flow_init: Optional[torch.Tensor] = None,
        source_valid: Optional[torch.Tensor] = None,
        target_valid: Optional[torch.Tensor] = None,
    ) -> Dict[str, List[torch.Tensor]]:
        iters = iters or self.num_hqs_iterations
        if iters > self.num_hqs_iterations:
            raise ValueError(
                f"Requested {iters} iterations, but model initialized with {self.num_hqs_iterations}."
            )

        i1 = self._normalise(image1)
        i2 = self._normalise(image2)

        # Photometric tensors for OFCE/data term
        # Remember these are deliberately NOT normalised according to ImageNet stats.
        if self.use_photo_for_ofce:
            i1_photo = self._photo_tensor(image1)
            i2_photo = self._photo_tensor(image2)
        else:
            i1_photo = i1
            i2_photo = i2

        # If we are using the luminance model enabled in the config, then we need to convert to greyscale
        if self.ofce_greyscale:
            i1_photo = self._to_greyscale(i1_photo)
            i2_photo = self._to_greyscale(i2_photo)

        feat1 = self.feature_encoder(i1)
        feat2 = self.feature_encoder(i2)
        f1 = feat1["level2"]
        f2 = feat2["level2"]

        ctx = self.context_encoder(i1)
        context_feat = ctx["context_feat"]
        hidden_state = ctx["hidden_state"]

        b, _, h, w = f1.shape
        # Update the OFCE tensors to the same resolution as the correlation features for the GRU updates.
        i1_lvl = F.interpolate(i1_photo, size=(h, w), mode="bilinear", align_corners=True)
        i2_lvl = F.interpolate(i2_photo, size=(h, w), mode="bilinear", align_corners=True)

        source_valid_lvl = self._resize_mask(
            source_valid,
            out_h=h,
            out_w=w,
            device=i1.device,
            dtype=i1.dtype,
        ) if source_valid is not None else None

        target_valid_lvl = self._resize_mask(
            target_valid,
            out_h=h,
            out_w=w,
            device=i1.device,
            dtype=i1.dtype,
        ) if target_valid is not None else None

        context_feat_lvl = F.interpolate(context_feat, size=(h, w), mode="bilinear", align_corners=False)
        hidden_lvl = F.interpolate(hidden_state, size=(h, w), mode="bilinear", align_corners=False)

        # Seed GRU hidden state from the context encoder's hidden output.
        net = torch.tanh(self.hidden_proj(hidden_lvl))  # (B, gru_hidden_dim, H, W)

        if flow_init is None:
            flow_yx = torch.zeros(b, 2, h, w, device=f1.device, dtype=f1.dtype)
        else:
            # Input flow in [dx, dy] repository convention → internal [dy, dx].
            flow_yx = torch.stack([flow_init[:, 1], flow_init[:, 0]], dim=1)
            flow_yx = self._resize_flow_yx(flow_yx, h, w)

        aux_yx = torch.zeros_like(flow_yx)

        corr_allpairs = self.build_all_pairs_correlation(f1, f2)
        corr_pyr = self.build_corr_pyramid_from_all_pairs(corr_allpairs, num_levels=self.num_corr_levels)
        coords1 = self._coords_grid(b, h, w, device=f1.device, dtype=f1.dtype)

        # Initialization: dense matching at current (zero) flow position.
        coords_query = coords1 + flow_yx.permute(0, 2, 3, 1)
        corr_feat_0 = self.sample_all_pairs_corr_pyramid(corr_pyr, coords_query, radius=self.max_displacement)
        corr_feat_0_chw = corr_feat_0.permute(0, 3, 1, 2).contiguous()
        x0 = self.allpairs_corr_decoder(corr_feat_0_chw)
        delta_init = self.allpairs_init_head(torch.cat([flow_yx, x0], dim=1))
        flow_yx = flow_yx + delta_init
        flow_yx = flow_yx
        aux_yx = flow_yx.clone()

        flow_preds: List[torch.Tensor] = []
        flow_lows: List[torch.Tensor] = []
        hidden_states: List[torch.Tensor] = []
        aux_lows: List[torch.Tensor] = []
        delta_lows: List[torch.Tensor] = []
        alpha_lows: List[torch.Tensor] = []
        delta_match_lows: List[torch.Tensor] = []
        delta_prior_lows: List[torch.Tensor] = []
        coupling_residual_lows: List[torch.Tensor] = []
        occupancy_masks: List[torch.Tensor] = []
        data_valid_lows: List[torch.Tensor] = []
        data_weight_lows: List[torch.Tensor] = []
        data_reliability_lows: List[torch.Tensor] = []
        matchability_lows: List[torch.Tensor] = []

        beta_schedule, lambda_schedule = self._hqs_penalties()
        correction_gates = self._correction_gates()
        if self.freeze_correction_last_n > 0:
            freeze_n = min(self.freeze_correction_last_n, iters)
            correction_gates = correction_gates.clone()
            correction_gates[iters - freeze_n:iters] = 0.0

        # Initialise prior hidden state from the same source-context seed.
        net_prior = net.detach().clone() if self.detach_reliability_inputs else net.clone()

        for k in range(iters):
            beta = beta_schedule[k]
            lam = lambda_schedule[k]

            # Sample correlation pyramid at current flow position.
            coords_query = coords1 + flow_yx.permute(0, 2, 3, 1)
            corr_feat_k = self.sample_all_pairs_corr_pyramid(corr_pyr, coords_query, radius=self.max_displacement)
            corr_feat_k_chw = corr_feat_k.permute(0, 3, 1, 2).contiguous()

            ######### # HQS u-step: closed-form OFCE data minimisation.
            ######### ix, iy, it, i2_warp = self.compute_ofce_derivatives(i1_lvl, i2_lvl, flow_yx)
            ######### # Adjust our confidence based on the validity of the warp to prevent out-of-bounds pixels from destabilizing training.
            ######### valid_k = self._valid_warp_mask(flow_yx)

            ######### # Compute the per iteration forward splat to build occupancy masks.
            ######### ones = torch.ones_like(flow_yx[:, :1, :, :])
            ######### occupancy = self.forward_splat(ones, flow_yx, normalize=False)
            ######### occupancy_mask = occupancy.clamp(0.0, 1.0)  # Optional: clamp to prevent extreme values from destabilizing training.

            ######### # Perform backwards warping of the occupancy mask to realign to image-1 coordinates.
            ######### occupancy_mask_warped = self._warp_yx(occupancy_mask, flow_yx)
            ######### occupancy_mask_warped = occupancy_mask_warped.clamp(0.0, 1.0)  # Optional: clamp to prevent extreme values from destabilizing training.
            ######### occupancy_masks.append(occupancy_mask_warped.detach())

            ######### # Per-iteration confidence from dedicated head (not the init decoder).
            ######### conf_in = torch.cat([corr_feat_k_chw, flow_yx, ix, iy, it, valid_k], dim=1)

            ######### confidence_k = self.iter_conf_head(conf_in)

            ######### if self.use_photometric_confidence:
                ######### photometric_conf_k = self.photometric_confidence(it, tau=self.photometric_confidence_tau)
            ######### else:
                ######### photometric_conf_k = torch.ones_like(it)

            ######### confidence_k = confidence_k * valid_k * photometric_conf_k

            ######### if self.use_data_reliability_mask:
                ######### grad_mag = torch.sqrt(ix * ix + iy * iy + 1e-6)
                ######### hqs_mag = torch.norm(flow_yx - aux_yx, dim=1, keepdim=True)
                ######### flow_mag = torch.norm(flow_yx, dim=1, keepdim=True)

                ######### rel_inputs = torch.cat(
                    ######### [
                        ######### it.abs(),
                        ######### grad_mag,
                        ######### valid_k,
                        ######### confidence_k,
                        ######### hqs_mag,
                        ######### flow_mag,
                    ######### ],
                    ######### dim=1,
                ######### )

                ######### # Detach the inputs if you want the reliability head to learn
                ######### # from these diagnostics without pushing gradients back through
                ######### # the diagnostic construction itself.
                ######### if self.detach_data_reliability_inputs:
                    ######### rel_inputs = rel_inputs.detach()

                ######### # IMPORTANT:
                ######### # Do NOT detach data_reliability if you want the head to learn.
                ######### data_reliability = self.data_reliability_head(rel_inputs)

                ######### data_weight = valid_k * data_reliability
            ######### else:
                ######### data_reliability = valid_k
                ######### data_weight = valid_k

            ######### # Mask off the data_weight using the occupancy mask
            ######### data_weight = occupancy_mask_warped * data_weight

            ######### # Mask the OFCE coefficients before they enter the analytic data update.
            ######### ix_data = ix 
            ######### iy_data = iy 
            ######### it_data = it

            ######### confidence_k = confidence_k * data_weight.detach()

            ######### flow_yx, _, _ = self.hqs_data_step(ix_data, iy_data, it_data, aux_yx, confidence_k, flow_yx, beta)

            # ------------------------------------------------------------
            # Leakage Safe OFCE Construction (Hopefully!)
            # ------------------------------------------------------------

            ix, iy, it, i2_warp, data_valid_k = self.compute_ofce_derivatives(
                i1_lvl,
                i2_lvl,
                flow_yx,
                source_valid=source_valid_lvl,
                target_valid=target_valid_lvl,
            )

            # Optional (config-controlled - occupancy masks)
            if self.use_self_occupancy_mask:
                ones = torch.ones_like(flow_yx[:, :1])
                occupancy = self.forward_splat(ones, flow_yx, normalize=False)
                occupancy_mask = occupancy.clamp(0.0, 1.0)
                occupancy_mask_warped = self._warp_yx(occupancy_mask, flow_yx).clamp(0.0, 1.0)
            else:
                occupancy_mask_warped = torch.ones_like(data_valid_k)

            data_valid_k = (data_valid_k * occupancy_mask_warped).clamp(0.0, 1.0)
            occupancy_masks.append(occupancy_mask_warped.detach())

            # Safe versions for learned branches
            # Do NOT use these inside the weighted least squares data step unless you intentially want m^2/m^3 weighting.
            # Use raw ix/iy/it + confidence for the analytic data step.

            ix_safe = ix * data_valid_k
            iy_safe = iy * data_valid_k
            it_safe = it * data_valid_k

            # --------------------------------------------------------------
            # Matchability confidence
            # --------------------------------------------------------------

            # Print the size of each component for debugging
            conf_in = torch.cat([
                corr_feat_k_chw,
                flow_yx,
                ix_safe,
                iy_safe, 
                it_safe,
                data_valid_k,
            ], dim=1)

            confidence_raw = self.iter_conf_head(conf_in)

            if self.use_photometric_confidence:
                photometric_conf_k = self.photometric_confidence(it_safe, tau=self.photometric_confidence_tau)
                photometric_conf_k = photometric_conf_k * data_valid_k  # Mask photometric confidence by data validity to prevent high confidence at invalid pixels.
                confidence_raw = confidence_raw * photometric_conf_k

            matchability_lows.append(confidence_raw.detach())

            confidence_masked = confidence_raw * data_valid_k

            # ------------------------------------------------------------
            # Data reliability weighting
            # ------------------------------------------------------------
            if self.use_data_reliability_mask:
                grad_mag = torch.sqrt(ix_safe * ix_safe + iy_safe * iy_safe + 1e-6)
                hqs_mag = torch.norm(flow_yx - aux_yx, dim=1, keepdim=True)
                flow_mag = torch.norm(flow_yx, dim=1, keepdim=True)

                rel_inputs = torch.cat(
                    [
                        it_safe.abs(),
                        grad_mag,
                        data_valid_k,
                        confidence_masked,
                        hqs_mag,
                        flow_mag,
                    ],
                    dim=1,
                )

                if self.detach_data_reliability_inputs:
                    rel_inputs = rel_inputs.detach()

                data_reliability = self.data_reliability_head(rel_inputs)
            else:                
                data_reliability = torch.ones_like(data_valid_k)

            data_weight = (data_valid_k * data_reliability).clamp(0.0, 1.0)

            confidence_k = (confidence_raw * data_weight).clamp(0.0, 1.0)

            data_valid_lows.append(data_valid_k.detach())
            data_weight_lows.append(data_weight.detach())
            data_reliability_lows.append(data_reliability.detach())

            # --------------------------------------------
            # Analytic HQS data step
            # --------------------------------------------
            flow_yx, _, _ = self.hqs_data_step(
                ix,
                iy,
                it,
                aux_yx,
                confidence_k,
                flow_yx,
                beta,
            )

            # ------------------------------------------------------------

            if self.use_learned_prox_net:
                # Use the learned prox network to predict the smoothed flow directly, rather than running the Jacobi iterations.
                aux_yx, _ = self.learned_prox(flow_yx=flow_yx, prev_aux_yx=aux_yx, context_feat=f1, validity=data_weight)
            else:
                # otherwise HQS v-step: Jacobi TV-proximal smoothing.
                aux_yx, _, _ = self.hqs_prox_step(
                    flow_yx, i1_lvl, beta, lam, validity_mask=data_weight, num_iter=self.prox_jacobi_iters
                )

            hqs_resid = flow_yx - aux_yx

            # ------------------------------------------------------------
            # Match branch: receives correlation.
            # ------------------------------------------------------------
            motion_confidence_k = confidence_k
            if self.detach_reliability_inputs:
                motion_confidence_k = motion_confidence_k.detach()

            motion_feat_match = self.motion_encoder(
                corr_feat_k_chw,
                flow_yx,
                hqs_resid,
                motion_confidence_k,
            )

            delta_match, mask_logits, net = self.update_unit(
                motion_feat_match,
                context_feat_lvl,
                net,
            )

            if self.use_split_delta_update:
                # --------------------------------------------------------
                # Prior branch: receives no correlation.
                # It has its own recurrent hidden state.
                # --------------------------------------------------------

                prior_confidence_k = confidence_k
                prior_it_k = it_safe
                prior_valid_k = data_valid_k

                if self.detach_reliability_inputs:
                    prior_confidence_k = prior_confidence_k.detach()
                    prior_it_k = prior_it_k.detach()
                    prior_valid_k = prior_valid_k.detach()

                motion_feat_prior = self.prior_motion_encoder(
                    flow_yx,
                    hqs_resid,
                    prior_confidence_k,
                    prior_it_k,
                    prior_valid_k,
                )

                delta_prior, _, net_prior = self.prior_update_unit(
                    motion_feat_prior,
                    context_feat_lvl,
                    net_prior,
                )

                gate_flow = flow_yx
                gate_resid = hqs_resid
                gate_conf = confidence_k
                gate_it = it_safe
                gate_valid = data_valid_k

                if self.detach_reliability_inputs:
                    gate_flow = gate_flow.detach()
                    gate_resid = gate_resid.detach()
                    gate_conf = gate_conf.detach()
                    gate_it = gate_it.detach()
                    gate_valid = gate_valid.detach()

                alpha = self.reliability_gate(
                    gate_flow,
                    gate_resid,
                    gate_conf,
                    gate_it,
                    gate_valid,
                )

                alpha = data_valid_k.detach()
                delta = alpha * delta_match + (1.0 - alpha) * delta_prior
            else:
                alpha = torch.ones_like(confidence_k)
                delta_prior = torch.zeros_like(delta_match)
                delta = delta_match

            alpha_lows.append(alpha.detach())
            delta_match_lows.append(torch.stack([delta_match[:, 1], delta_match[:, 0]], dim=1).detach())

            if self.use_split_delta_update:
                delta_prior_lows.append(torch.stack([delta_prior[:, 1], delta_prior[:, 0]], dim=1).detach())
            else:
                delta_prior_lows.append(torch.zeros_like(delta_match_lows[-1]))

            flow_yx = flow_yx + correction_gates[k] * delta

            if self.use_learned_prox_net:
                # Use the learned prox network to predict the smoothed flow directly, rather than running the Jacobi iterations.
                aux_yx, _ = self.learned_prox(flow_yx=flow_yx, prev_aux_yx=aux_yx, context_feat=f1, validity=data_weight)
            else:
                aux_yx, _, _ = self.hqs_prox_step(
                    flow_yx, i1_lvl, beta, lam, validity_mask=data_weight, num_iter=self.prox_jacobi_iters
                )

            # Per-iteration full-resolution prediction via convex upsampling.
            flow_up_yx = self.convex_upsample(flow_yx, mask_logits, rate=self.upsample_scale)
            if flow_up_yx.shape[-2:] != image1.shape[-2:]:
                flow_up_yx = self._resize_flow_yx(flow_up_yx, image1.shape[-2], image1.shape[-1])
            flow_up_xy = torch.stack([flow_up_yx[:, 1], flow_up_yx[:, 0]], dim=1)

            flow_lows.append(torch.stack([flow_yx[:, 1], flow_yx[:, 0]], dim=1))
            flow_preds.append(flow_up_xy)
            hidden_states.append(net)
            aux_lows.append(torch.stack([aux_yx[:, 1], aux_yx[:, 0]], dim=1))
            delta_lows.append(torch.stack([delta[:, 1], delta[:, 0]], dim=1))
            coupling_residual_lows.append(
                torch.stack([(flow_yx - aux_yx)[:, 1], (flow_yx - aux_yx)[:, 0]], dim=1)
            )

        if self.pyramid_l1_iters > 0 and "level1" in feat1 and "level1" in feat2:
            # ── Feature-pyramid multi-scale refinement (1.2 + 2.1) ──────────
            # Runs a dedicated HQS loop at level1 (1/4-scale) features using
            # its own learnable penalty and gate schedules.  Each iteration
            # appends a full-resolution prediction to flow_preds so the sequence
            # loss supervises convergence at this finer scale.  The refined flow
            # is transferred back to level2 so the post-loop network starts from
            # the best available state.
            f1_l1 = feat1["level1"]
            f2_l1 = feat2["level1"]
            _, _, h1, w1 = f1_l1.shape

            # ------ OLD Bilinear resize-based transfer (Option A) ------
            # flow_l1 = self._resize_flow_yx(flow_yx, h1, w1)
            # aux_l1 = self._resize_flow_yx(aux_yx, h1, w1)
            # ------ New convex upsample-based transfer (Option B) -----
            flow_l1 = self.convex_transition_upsample_yx(flow_yx, f1, f1_l1, f2_l1)
            aux_l1 = self.convex_transition_upsample_yx(aux_yx, f1, f1_l1, f2_l1)

            i1_l1 = F.interpolate(i1_photo, size=(h1, w1), mode="bilinear", align_corners=True)
            i2_l1 = F.interpolate(i2_photo, size=(h1, w1), mode="bilinear", align_corners=True)

            source_valid_l1 = self._resize_mask(
                source_valid,
                out_h=h1,
                out_w=w1,
                device=i1_photo.device,
                dtype=i1_photo.dtype,
            ) if source_valid is not None else None

            target_valid_l1 = self._resize_mask(
                target_valid,
                out_h=h1,
                out_w=w1,
                device=i1_photo.device,
                dtype=i1_photo.dtype,
            ) if target_valid is not None else None

            context_l1 = F.interpolate(context_feat, size=(h1, w1), mode="bilinear", align_corners=False)
            net_l1 = F.interpolate(net, size=(h1, w1), mode="bilinear", align_corners=False)

            corr_l1 = self.build_all_pairs_correlation(f1_l1, f2_l1)
            corr_pyr_l1 = self.build_corr_pyramid_from_all_pairs(
                corr_l1, num_levels=self.num_corr_levels
            )
            coords_l1 = self._coords_grid(b, h1, w1, device=f1.device, dtype=f1.dtype)

            # Dedicated schedules — independent of the main loop's trajectory.
            beta_l1_sched, lam_l1_sched = self._compute_hqs_penalties_from(
                self.hqs_beta_steps_l1, self.hqs_lambda_ratio_steps_l1
            )
            gates_l1 = self._compute_correction_gates_from(self.delta_gate_logits_l1)
            freeze_n_l1 = min(self.freeze_correction_last_n, self.pyramid_l1_iters)
            if freeze_n_l1 > 0:
                gates_l1 = gates_l1.clone()
                gates_l1[self.pyramid_l1_iters - freeze_n_l1:] = 0.0

            net_l1_prior = net_l1.clone()

            for k_l1 in range(self.pyramid_l1_iters):
                beta = beta_l1_sched[k_l1]
                lam = lam_l1_sched[k_l1]

                coords_query_l1 = coords_l1 + flow_l1.permute(0, 2, 3, 1)
                corr_feat_l1 = self.sample_all_pairs_corr_pyramid(
                    corr_pyr_l1, coords_query_l1, radius=self.max_displacement
                )
                corr_feat_l1_chw = corr_feat_l1.permute(0, 3, 1, 2).contiguous()

                ######### ix, iy, it, _ = self.compute_ofce_derivatives(i1_l1, i2_l1, flow_l1)
                
                ######### valid_l1 = self._valid_warp_mask(flow_l1)

                ######### # Compute the per iteration forward splat to build occupancy masks.
                ######### ones = torch.ones_like(flow_l1[:, :1, :, :])
                ######### occupancy = self.forward_splat(ones, flow_l1, normalize=False)
                ######### occupancy_mask = occupancy.clamp(0.0, 1.0)  # Optional: clamp to prevent extreme values from destabilizing training.

                ######### # Perform backwards warping of the occupancy mask to realign to image-1 coordinates.
                ######### occupancy_mask_warped = self._warp_yx(occupancy_mask, flow_l1)
                ######### occupancy_mask_warped = occupancy_mask_warped.clamp(0.0, 1.0)  # Optional: clamp to prevent extreme values from destabilizing training.
                ######### occupancy_masks.append(occupancy_mask_warped.detach())

                ######### if self.use_photometric_confidence:
                    ######### photometric_conf_l1 = self.photometric_confidence(it, tau=self.photometric_confidence_tau)
                ######### else:
                    ######### photometric_conf_l1 = torch.ones_like(it)

                ######### conf_in = torch.cat([corr_feat_l1_chw, flow_l1, ix, iy, it, valid_l1], dim=1)
                ######### confidence_l1 = self.iter_conf_head(conf_in)
                ######### confidence_l1 = confidence_l1 * valid_l1 * photometric_conf_l1

                ######### if self.use_data_reliability_mask:
                    ######### grad_mag = torch.sqrt(ix * ix + iy * iy + 1e-6)
                    ######### hqs_mag = torch.norm(flow_l1 - aux_l1, dim=1, keepdim=True)
                    ######### flow_mag = torch.norm(flow_l1, dim=1, keepdim=True)

                    ######### rel_inputs = torch.cat(
                        ######### [
                            ######### it.abs(),
                            ######### grad_mag,
                            ######### valid_l1,
                            ######### confidence_l1,
                            ######### hqs_mag,
                            ######### flow_mag,
                        ######### ],
                        ######### dim=1,
                    ######### )

                    ######### # Detach the inputs if you want the reliability head to learn
                    ######### # from these diagnostics without pushing gradients back through
                    ######### # the diagnostic construction itself.
                    ######### if self.detach_data_reliability_inputs:
                        ######### rel_inputs = rel_inputs.detach()

                    ######### # IMPORTANT:
                    ######### # Do NOT detach data_reliability if you want the head to learn.
                    ######### data_reliability = self.data_reliability_head(rel_inputs)

                    ######### data_weight = valid_l1 * data_reliability
                ######### else:
                    ######### data_reliability = valid_l1
                    ######### data_weight = valid_l1

                ######### # Mask off the data_weight using the occupancy mask
                ######### data_weight = occupancy_mask_warped * data_weight

                ######### # Mask the OFCE coefficients before they enter the analytic data update.
                ######### ix_data = ix 
                ######### iy_data = iy
                ######### it_data = it

                ######### confidence_l1 = confidence_l1 * data_weight.detach()

                ######### flow_l1, _, _ = self.hqs_data_step(
                    ######### ix_data, iy_data, it_data, aux_l1, confidence_l1, flow_l1, beta
                ######### )

                # ------------------------------------------------------------
                # Leakage Safe OFCE Construction (Hopefully!) at level 1
                # ------------------------------------------------------------
                ix, iy, it, i2_warp, data_valid_l1 = self.compute_ofce_derivatives(
                    i1_l1,
                    i2_l1,
                    flow_l1,
                    source_valid=source_valid_l1,
                    target_valid=target_valid_l1,
                )

                if self.use_self_occupancy_mask:
                    ones = torch.ones_like(flow_l1[:, :1])
                    occupancy = self.forward_splat(ones, flow_l1, normalize=False)
                    occupancy_mask = occupancy.clamp(0.0, 1.0)
                    occupancy_mask_warped = self._warp_yx(occupancy_mask, flow_l1).clamp(0.0, 1.0)
                else:
                    occupancy_mask_warped = torch.ones_like(data_valid_l1)

                data_valid_l1 = (data_valid_l1 * occupancy_mask_warped).clamp(0.0, 1.0)
                occupancy_masks.append(occupancy_mask_warped.detach())

                ix_safe = ix * data_valid_l1
                iy_safe = iy * data_valid_l1
                it_safe = it * data_valid_l1

                conf_in = torch.cat([
                    corr_feat_l1_chw,
                    flow_l1,
                    ix_safe,
                    iy_safe,
                    it_safe,
                    data_valid_l1,
                ], dim=1)

                confidence_raw_l1 = self.iter_conf_head(conf_in)

                if self.use_photometric_confidence:
                    photometric_conf_l1 = self.photometric_confidence(it_safe, tau=self.photometric_confidence_tau)
                    photometric_conf_l1 = photometric_conf_l1 * data_valid_l1  # Mask photometric confidence by data validity to prevent high confidence at invalid pixels.
                    confidence_raw_l1 = confidence_raw_l1 * photometric_conf_l1

                matchability_lows.append(confidence_raw_l1.detach())
                confidence_masked_l1 = confidence_raw_l1 * data_valid_l1

                if self.use_data_reliability_mask:
                    grad_mag = torch.sqrt(ix_safe * ix_safe + iy_safe * iy_safe + 1e-6)
                    hqs_mag = torch.norm(flow_l1 - aux_l1, dim=1, keepdim=True)
                    flow_mag = torch.norm(flow_l1, dim=1, keepdim=True)

                    rel_inputs = torch.cat(
                        [
                            it_safe.abs(),
                            grad_mag,
                            data_valid_l1,
                            confidence_masked_l1,
                            hqs_mag,
                            flow_mag,
                        ],
                        dim=1,
                    )

                    if self.detach_data_reliability_inputs:
                        rel_inputs = rel_inputs.detach()

                    data_reliability_l1 = self.data_reliability_head(rel_inputs)
                else:
                    data_reliability_l1 = torch.ones_like(data_valid_l1)

                data_weight = (data_valid_l1 * data_reliability_l1).clamp(0.0, 1.0)
                confidence_l1 = (confidence_raw_l1 * data_weight).clamp(0.0, 1.0)

                data_valid_lows.append(data_valid_l1.detach())
                data_weight_lows.append(data_weight.detach())
                data_reliability_lows.append(data_reliability_l1.detach())

                flow_l1, _, _ = self.hqs_data_step(
                    ix,
                    iy,
                    it,
                    aux_l1,
                    confidence_l1,
                    flow_l1,
                    beta,
                )

                # ------------------------------------------------------------

                if self.use_learned_prox_net:
                    # Use the learned prox network to predict the smoothed flow directly, rather than running the Jacobi iterations.
                    aux_l1, _ = self.learned_prox(
                        flow_yx=flow_l1, prev_aux_yx=aux_l1, context_feat=f1_l1, validity=data_weight
                    )
                else:
                    aux_l1, _, _ = self.hqs_prox_step(
                        flow_l1, i1_l1, beta, lam, validity_mask=data_weight, num_iter=self.prox_jacobi_iters
                    )

                ############ OLD UPDATE
                ############ hqs_resid_l1 = flow_l1 - aux_l1
                ############ motion_l1 = self.motion_encoder(corr_feat_l1_chw, flow_l1, hqs_resid_l1, confidence_l1)
                ############ delta_l1, _, net_l1 = self.update_unit(motion_l1, context_l1, net_l1)
                ############ flow_l1 = flow_l1 + gates_l1[k_l1] * delta_l1
                

                hqs_resid_l1 = flow_l1 - aux_l1

                motion_confidence_l1 = confidence_l1
                if self.detach_reliability_inputs:
                    motion_confidence_l1 = motion_confidence_l1.detach()

                motion_l1_match = self.motion_encoder(
                    corr_feat_l1_chw,
                    flow_l1,
                    hqs_resid_l1,
                    motion_confidence_l1,
                )

                delta_l1_match, _, net_l1 = self.update_unit(
                    motion_l1_match,
                    context_l1,
                    net_l1,
                )

                if self.use_split_delta_update:
                    prior_confidence_l1 = confidence_l1
                    prior_it_l1 = it_safe
                    prior_valid_l1 = data_valid_l1

                    if self.detach_reliability_inputs:
                        prior_confidence_l1 = prior_confidence_l1.detach()
                        prior_it_l1 = prior_it_l1.detach()
                        prior_valid_l1 = prior_valid_l1.detach()

                    motion_l1_prior = self.prior_motion_encoder(
                        flow_l1,
                        hqs_resid_l1,
                        prior_confidence_l1,
                        prior_it_l1,
                        prior_valid_l1,
                    )

                    delta_l1_prior, _, net_l1_prior = self.prior_update_unit(
                        motion_l1_prior,
                        context_l1,
                        net_l1_prior,
                    )

                    gate_flow_l1 = flow_l1
                    gate_resid_l1 = hqs_resid_l1
                    gate_conf_l1 = confidence_l1
                    gate_it_l1 = it_safe
                    gate_valid_l1 = data_valid_l1

                    if self.detach_reliability_inputs:
                        gate_flow_l1 = gate_flow_l1.detach()
                        gate_resid_l1 = gate_resid_l1.detach()
                        gate_conf_l1 = gate_conf_l1.detach()
                        gate_it_l1 = gate_it_l1.detach()
                        gate_valid_l1 = gate_valid_l1.detach()

                    alpha_l1 = self.reliability_gate(
                        gate_flow_l1,
                        gate_resid_l1,
                        gate_conf_l1,
                        gate_it_l1,
                        gate_valid_l1,
                    )

                    alpha_l1 = data_valid_l1.detach()
                    delta_l1 = alpha_l1 * delta_l1_match + (1.0 - alpha_l1) * delta_l1_prior
                else:
                    alpha_l1 = torch.ones_like(confidence_l1)
                    delta_l1_prior = torch.zeros_like(delta_l1_match)
                    delta_l1 = delta_l1_match

                flow_l1 = flow_l1 + gates_l1[k_l1] * delta_l1

                if self.use_learned_prox_net:
                    # Use the learned prox network to predict the smoothed flow directly, rather than running the Jacobi iterations.
                    aux_l1, _ = self.learned_prox(
                        flow_yx=flow_l1, prev_aux_yx=aux_l1, context_feat=f1_l1, validity=data_weight
                    )
                else:
                    aux_l1, _, _ = self.hqs_prox_step(
                        flow_l1, i1_l1, beta, lam, validity_mask=data_weight, num_iter=self.prox_jacobi_iters
                    )

                # Full-res prediction from 1/4-scale flow via bilinear upsampling.
                # Higher quality than main-loop predictions (finer features),
                # and supervised by the sequence loss at each step.
                flow_up_l1 = self._resize_flow_yx(flow_l1, image1.shape[-2], image1.shape[-1])
                flow_up_l1_xy = torch.stack([flow_up_l1[:, 1], flow_up_l1[:, 0]], dim=1)
                flow_preds.append(flow_up_l1_xy)
                flow_lows.append(torch.stack([flow_l1[:, 1], flow_l1[:, 0]], dim=1))
                hidden_states.append(net_l1)
                aux_lows.append(torch.stack([aux_l1[:, 1], aux_l1[:, 0]], dim=1))
                delta_lows.append(torch.stack([delta_l1[:, 1], delta_l1[:, 0]], dim=1))
                coupling_residual_lows.append(
                    torch.stack([(flow_l1 - aux_l1)[:, 1], (flow_l1 - aux_l1)[:, 0]], dim=1)
                )

            # Transfer refined flow back to level2 for post-loop refinement.
            flow_yx = self._resize_flow_yx(flow_l1, h, w)
            aux_yx = self._resize_flow_yx(aux_l1, h, w)
            net = F.interpolate(net_l1, size=(h, w), mode="bilinear", align_corners=False)

        raw_flow_preds = list(flow_preds)
        raw_flow_lows = list(flow_lows)

        # Post-loop refinement using the fixed context-encoder hidden output
        # (stable reference independent of GRU trajectory).
        final_in = torch.cat([flow_yx, context_feat_lvl, hidden_lvl], dim=1)
        flow_yx = flow_yx + 0.1 * self.final_refinement_network(final_in)
        final_mask = self.final_mask_logits(final_in)
        flow_up_final_yx = self.convex_upsample(flow_yx, final_mask, rate=self.upsample_scale)
        if flow_up_final_yx.shape[-2:] != image1.shape[-2:]:
            flow_up_final_yx = self._resize_flow_yx(flow_up_final_yx, image1.shape[-2], image1.shape[-1])
        flow_up_xy = torch.stack([flow_up_final_yx[:, 1], flow_up_final_yx[:, 0]], dim=1)

        if flow_preds:
            flow_preds[-1] = flow_up_xy
            flow_lows[-1] = torch.stack([flow_yx[:, 1], flow_yx[:, 0]], dim=1)
            if aux_lows:
                aux_lows[-1] = torch.stack([aux_yx[:, 1], aux_yx[:, 0]], dim=1)
            if coupling_residual_lows:
                coupling_residual_lows[-1] = torch.stack(
                    [(flow_yx - aux_yx)[:, 1], (flow_yx - aux_yx)[:, 0]], dim=1
                )
        else:
            flow_preds = [flow_up_xy]
            flow_lows = [torch.stack([flow_yx[:, 1], flow_yx[:, 0]], dim=1)]
            hidden_states = [net]
            aux_lows = [torch.stack([aux_yx[:, 1], aux_yx[:, 0]], dim=1)]
            delta_lows = [torch.zeros_like(flow_lows[0])]
            coupling_residual_lows = [
                torch.stack([(flow_yx - aux_yx)[:, 1], (flow_yx - aux_yx)[:, 0]], dim=1)
            ]

        return {
            "flow_preds": flow_preds,
            "flow_preds_raw": raw_flow_preds,
            "flow_low": flow_lows,
            "flow_low_raw": raw_flow_lows,
            "aux_low": aux_lows,
            "delta_low": delta_lows,
            "coupling_residual_low": coupling_residual_lows,
            "hidden_states": hidden_states,
            "flow_final_raw": raw_flow_preds[-1] if raw_flow_preds else flow_up_xy,
            "flow_final_refined": flow_up_xy,
            "alpha_low": alpha_lows,
            "delta_match_low": delta_match_lows,
            "delta_prior_low": delta_prior_lows,
            "data_reliability_low": data_reliability.detach() if self.use_data_reliability_mask else None,
            "occupancy_masks": occupancy_masks,
            "data_valid_lows": data_valid_lows,
            "data_weight_lows": data_weight_lows,
            "data_reliability_lows": data_reliability_lows,
            "matchability_lows": matchability_lows,
        }

    def param_count(self) -> Dict[str, int]:
        def count(module: nn.Module) -> int:
            return sum(p.numel() for p in module.parameters())

        def count_trainable(module: nn.Module) -> int:
            return sum(p.numel() for p in module.parameters() if p.requires_grad)

        base = {
            "feature_encoder": count(self.feature_encoder),
            "context_encoder": count(self.context_encoder),
            "motion_encoder": count(self.motion_encoder),
            "update_unit": count(self.update_unit),
            "init_decoder": (
                count(self.allpairs_corr_decoder)
                + count(self.allpairs_init_head)
                + count(self.allpairs_conf_head)
            ),
            "iter_conf_head": count(self.iter_conf_head),
            "final_refinement": (
                count(self.final_refinement_network) + count(self.final_mask_logits)
            ),
            "data_reliability_head": count(self.data_reliability_head),
            "hidden_proj": count(self.hidden_proj),
            "level1_transition_mask": count(self.level1_transition_mask),
            "stages": count(self.update_unit),  # alias kept for trainer logging compat
            "total": count(self),
            "total_trainable": count_trainable(self),
        }

        # Optional modules added by feature flags.
        base["prior_motion_encoder"] = count(self.prior_motion_encoder) if hasattr(self, "prior_motion_encoder") else 0
        base["prior_update_unit"] = count(self.prior_update_unit) if hasattr(self, "prior_update_unit") else 0
        base["reliability_gate"] = count(self.reliability_gate) if hasattr(self, "reliability_gate") else 0
        base["learned_prox"] = count(self.learned_prox) if self.learned_prox is not None else 0

        # Per-component trainable counts for logging parity.
        base["feature_encoder_trainable"] = count_trainable(self.feature_encoder)
        base["context_encoder_trainable"] = count_trainable(self.context_encoder)
        base["motion_encoder_trainable"] = count_trainable(self.motion_encoder)
        base["update_unit_trainable"] = count_trainable(self.update_unit)
        base["iter_conf_head_trainable"] = count_trainable(self.iter_conf_head)
        base["final_refinement_trainable"] = count_trainable(self.final_refinement_network) + count_trainable(self.final_mask_logits)
        base["prior_motion_encoder_trainable"] = count_trainable(self.prior_motion_encoder) if hasattr(self, "prior_motion_encoder") else 0
        base["prior_update_unit_trainable"] = count_trainable(self.prior_update_unit) if hasattr(self, "prior_update_unit") else 0
        base["reliability_gate_trainable"] = count_trainable(self.reliability_gate) if hasattr(self, "reliability_gate") else 0
        base["data_reliability_head_trainable"] = count_trainable(self.data_reliability_head)
        base["hidden_proj_trainable"] = count_trainable(self.hidden_proj)
        base["level1_transition_mask_trainable"] = count_trainable(self.level1_transition_mask)
        base["learned_prox_trainable"] = count_trainable(self.learned_prox) if self.learned_prox is not None else 0
        base["total_non_trainable"] = base["total"] - base["total_trainable"]

        return base
