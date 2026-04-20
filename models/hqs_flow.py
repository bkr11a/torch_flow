"""HQSFlow – top-level optical flow model.

Assembles:
  1. Feature pyramid encoder   (shared between both images)
  2. Context encoder           (applied to image 1 only)
  3. Correlation block         (all-pairs or local)
  4. K unrolled HQS stages     (configurable, default 12)
  5. Convex upsampling head    (×8 to original resolution)

Weight-sharing across stages is configurable:
  share_all        – all stages share one DataUpdateNet and one ProxNet
                     (like RAFT's recurrent design, fewest parameters).
  share_none       – every stage has its own independent weights
                     (maximum capacity, great for ablations).
  share_update     – shared DataUpdateNet, independent per-stage ProxNet.
  share_prox       – independent per-stage DataUpdateNet, shared ProxNet.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple

from .encoders import build_encoder, BasicEncoder, SmallEncoder
from .correlation import build_corr_block, CorrBlock
from .update_net import DataUpdateNet
from .reg_net import build_prox_net
from .hqs_stage import HQSStage
from .warp import upsample_flow


# ---------------------------------------------------------------------------
# Convex upsampling (learnable, RAFT-style)
# ---------------------------------------------------------------------------

def convex_upsample(
    flow: torch.Tensor,
    mask: torch.Tensor,
    scale: int = 8,
) -> torch.Tensor:
    """
    Upsample *flow* using a soft convex combination of neighbouring pixels,
    predicted by the update network.

    Args:
        flow: (B, 2, H, W) low-resolution flow.
        mask: (B, 9*scale², H, W) unnormalised logits.
        scale: upsampling factor (must match encoder output_stride).

    Returns:
        (B, 2, scale*H, scale*W) high-resolution flow.
    """
    B, _, H, W = flow.shape
    mask = mask.view(B, 1, 9, scale, scale, H, W)
    mask = torch.softmax(mask, dim=2)

    # Unfold flow into 3×3 neighbourhood patches
    up_flow = F.unfold(scale * flow, kernel_size=3, padding=1)
    up_flow = up_flow.view(B, 2, 9, 1, 1, H, W)

    up_flow = (mask * up_flow).sum(dim=2)      # (B, 2, scale, scale, H, W)
    return up_flow.permute(0, 1, 4, 2, 5, 3).reshape(
        B, 2, scale * H, scale * W
    )


# ---------------------------------------------------------------------------
# HQSFlow
# ---------------------------------------------------------------------------

class HQSFlow(nn.Module):
    """
    Unrolled Learned Half-Quadratic Splitting network for optical flow.

    Args:
        cfg: OmegaConf DictConfig (or plain dict) with model hyperparameters.

    Key cfg fields:
        num_stages          : int  – number of unrolled HQS iterations (default 12)
        weight_sharing      : str  – "share_all" | "share_none" | "share_update" | "share_prox"
        encoder.type        : "basic" | "small"
        encoder.output_dim  : int
        encoder.norm        : str
        context.output_dim  : int
        context.norm        : str
        corr.type           : "all_pairs" | "local"
        corr.num_levels     : int
        corr.radius         : int
        update.hidden_dim   : int
        update.head_dim     : int
        prox.type           : "unet" | "dncnn" | "none"
        upsample_scale      : int  (default 8)
        dropout             : float
    """

    def __init__(self, cfg) -> None:
        super().__init__()
        self.cfg = cfg

        # ------------------------------------------------------------------ #
        # Feature encoder (applied to both images, weights shared)
        # ------------------------------------------------------------------ #
        self.feature_encoder: nn.Module = build_encoder(cfg.encoder)
        feat_dim: int = cfg.encoder.output_dim

        # ------------------------------------------------------------------ #
        # Context encoder (applied to image 1 only)
        # ------------------------------------------------------------------ #
        self.context_encoder: nn.Module = build_encoder(cfg.context)
        ctx_dim: int = cfg.context.output_dim

        # ------------------------------------------------------------------ #
        # Correlation block
        # ------------------------------------------------------------------ #
        self.corr_block = build_corr_block(cfg.corr)
        corr_dim: int = self.corr_block.out_channels

        # ------------------------------------------------------------------ #
        # Build networks (shared or per-stage)
        # ------------------------------------------------------------------ #
        self.num_stages: int = cfg.num_stages
        share: str = cfg.get("weight_sharing", "share_all")

        upsample_scale: int = cfg.get("upsample_scale", 8)
        hidden_dim: int = cfg.update_net.hidden_dim
        head_dim:   int = cfg.update_net.get("head_dim", 256)

        def _make_update_net() -> DataUpdateNet:
            return DataUpdateNet(
                corr_channels=corr_dim,
                context_channels=ctx_dim,
                hidden_dim=hidden_dim,
                head_dim=head_dim,
                with_upsample_mask=True,
                upsample_scale=upsample_scale,
            )

        def _make_prox_net() -> nn.Module:
            return build_prox_net(cfg.prox)

        if share == "share_all":
            shared_update = _make_update_net()
            shared_prox   = _make_prox_net()
            self.stages = nn.ModuleList([
                HQSStage(shared_update, shared_prox, share_weights=True)
                for _ in range(self.num_stages)
            ])
        elif share == "share_none":
            self.stages = nn.ModuleList([
                HQSStage(_make_update_net(), _make_prox_net(), share_weights=False)
                for _ in range(self.num_stages)
            ])
        elif share == "share_update":
            shared_update = _make_update_net()
            self.stages = nn.ModuleList([
                HQSStage(shared_update, _make_prox_net(), share_weights=False)
                for _ in range(self.num_stages)
            ])
        elif share == "share_prox":
            shared_prox = _make_prox_net()
            self.stages = nn.ModuleList([
                HQSStage(_make_update_net(), shared_prox, share_weights=False)
                for _ in range(self.num_stages)
            ])
        else:
            raise ValueError(f"Unknown weight_sharing mode: {share!r}")

        self.upsample_scale = upsample_scale

        # Context features are projected into initial GRU hidden state & v
        self.ctx_to_hidden = nn.Conv2d(ctx_dim, hidden_dim, 1)
        self.ctx_to_v      = nn.Conv2d(ctx_dim, 2, 1)
        nn.init.zeros_(self.ctx_to_v.weight)
        nn.init.zeros_(self.ctx_to_v.bias)

    # ---------------------------------------------------------------------- #
    # Forward
    # ---------------------------------------------------------------------- #

    def forward(
        self,
        image1: torch.Tensor,
        image2: torch.Tensor,
        iters: Optional[int] = None,
        flow_init: Optional[torch.Tensor] = None,
    ) -> Dict[str, List[torch.Tensor]]:
        """
        Args:
            image1, image2: (B, 3, H, W) images, pixel values in [0, 1] or normalised.
            iters:          Number of HQS stages to run (default: all stages).
            flow_init:      Optional initial flow estimate (B, 2, H//scale, W//scale).

        Returns:
            Dictionary with keys:
              "flow_preds"    – list of (B, 2, H, W) full-res flow predictions,
                                one per stage (for sequence loss).
              "flow_low"      – list of (B, 2, H/scale, W/scale) low-res flows.
              "hidden_states" – list of GRU hidden states (useful for analysis).
        """
        iters = iters or self.num_stages
        assert iters <= self.num_stages, (
            f"Requested {iters} but model has only {self.num_stages} stages."
        )

        # ---- Feature extraction -------------------------------------------
        # Normalise images to [-1, 1] for the encoder
        img1 = self._normalise(image1)
        img2 = self._normalise(image2)

        fmap1 = self.feature_encoder(img1)
        fmap2 = self.feature_encoder(img2)

        # L2-normalise features for stable correlation
        fmap1 = F.normalize(fmap1, p=2, dim=1)
        fmap2 = F.normalize(fmap2, p=2, dim=1)

        context = self.context_encoder(img1)

        # ---- Initialise correlation pyramid ---------------------------------
        if isinstance(self.corr_block, CorrBlock):
            self.corr_block.initialise(fmap1, fmap2)

        # ---- Initialise flow and hidden state -------------------------------
        B, _, H, W = fmap1.shape
        device = fmap1.device

        if flow_init is not None:
            # Downsample to feature resolution if needed
            if flow_init.shape[-2:] != (H, W):
                flow_init = F.interpolate(
                    flow_init / self.upsample_scale,
                    size=(H, W), mode="bilinear", align_corners=True,
                )
            u = flow_init.clone()
        else:
            u = torch.zeros(B, 2, H, W, device=device, dtype=fmap1.dtype)

        # Initialise v from context (gives a data-dependent prior)
        v = torch.tanh(self.ctx_to_v(context))

        # Initialise GRU hidden state from context features
        hidden = torch.tanh(self.ctx_to_hidden(context))

        # ---- Unrolled HQS iterations ----------------------------------------
        flow_preds: List[torch.Tensor] = []
        flow_lows:  List[torch.Tensor] = []
        hidden_states: List[torch.Tensor] = []

        for k in range(iters):
            stage = self.stages[k]
            u, v, hidden, up_mask = stage(
                fmap1, fmap2, self.corr_block, context, u, v, hidden
            )
            flow_lows.append(u)
            hidden_states.append(hidden)

            # Upsample to full resolution
            if up_mask is not None:
                flow_up = convex_upsample(u, up_mask, scale=self.upsample_scale)
            else:
                flow_up = upsample_flow(u, scale_factor=self.upsample_scale)

            flow_preds.append(flow_up)

        return {
            "flow_preds":    flow_preds,    # for sequence loss
            "flow_low":      flow_lows,
            "hidden_states": hidden_states,
        }

    # ---------------------------------------------------------------------- #
    # Helpers
    # ---------------------------------------------------------------------- #

    @staticmethod
    def _normalise(img: torch.Tensor) -> torch.Tensor:
        """Map [0, 255] or [0, 1] images to approximately zero-mean, unit-std."""
        if img.max() > 2.0:               # assume [0, 255]
            img = img / 255.0
        mean = img.new_tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std  = img.new_tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        return (img - mean) / std

    def param_count(self) -> Dict[str, int]:
        """Return parameter counts by component for reporting."""
        def _count(m: nn.Module) -> int:
            return sum(p.numel() for p in m.parameters())

        return {
            "feature_encoder": _count(self.feature_encoder),
            "context_encoder": _count(self.context_encoder),
            "stages":          _count(self.stages),
            "total":           _count(self),
        }


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_model(cfg) -> HQSFlow:
    return HQSFlow(cfg.model)
