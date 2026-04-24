"""Primary configurable HQS optical-flow model.

This implementation now lives under hqs_pytorch and is the authoritative
model used by train.py / evaluate.py via models.build_model.

It preserves the repository's existing training contract:
  - configuration is read from cfg.model
  - forward() returns a dict with flow_preds / flow_low / hidden_states
  - param_count() reports component totals for trainer logging

The implementation reuses the stable solver components already present in the
repository while relocating the main model definition into hqs_pytorch, which
is now the primary solution for this codebase.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional

from models.encoders import build_encoder
from models.correlation import build_corr_block, CorrBlock
from models.update_net import DataUpdateNet
from models.reg_net import build_prox_net
from models.hqs_stage import HQSStage
from models.warp import upsample_flow


def convex_upsample(
    flow: torch.Tensor,
    mask: torch.Tensor,
    scale: int = 8,
) -> torch.Tensor:
    """RAFT-style convex upsampling from feature resolution to image space."""
    batch_size, _, height, width = flow.shape
    mask = mask.view(batch_size, 1, 9, scale, scale, height, width)
    mask = torch.softmax(mask, dim=2)

    up_flow = F.unfold(scale * flow, kernel_size=3, padding=1)
    up_flow = up_flow.view(batch_size, 2, 9, 1, 1, height, width)
    up_flow = (mask * up_flow).sum(dim=2)

    return up_flow.permute(0, 1, 4, 2, 5, 3).reshape(
        batch_size, 2, scale * height, scale * width
    )


class HQSFlowModel(nn.Module):
    """Configurable unrolled HQS optical-flow model backed by cfg.model."""

    def __init__(self, cfg) -> None:
        super().__init__()
        self.cfg = cfg

        self.feature_encoder = build_encoder(cfg.encoder)
        self.context_encoder = build_encoder(cfg.context)
        self.corr_block = build_corr_block(cfg.corr)

        feat_dim = cfg.encoder.output_dim
        ctx_dim = cfg.context.output_dim
        corr_dim = self.corr_block.out_channels

        self.num_stages = cfg.num_stages
        self.upsample_scale = cfg.get("upsample_scale", 8)

        hidden_dim = cfg.update_net.hidden_dim
        head_dim = cfg.update_net.get("head_dim", 256)
        share = cfg.get("weight_sharing", "share_all")

        def make_update_net() -> DataUpdateNet:
            return DataUpdateNet(
                corr_channels=corr_dim,
                context_channels=ctx_dim,
                hidden_dim=hidden_dim,
                head_dim=head_dim,
                with_upsample_mask=True,
                upsample_scale=self.upsample_scale,
            )

        def make_prox_net() -> nn.Module:
            return build_prox_net(cfg.prox)

        if share == "share_all":
            shared_update = make_update_net()
            shared_prox = make_prox_net()
            self.stages = nn.ModuleList([
                HQSStage(shared_update, shared_prox, share_weights=True)
                for _ in range(self.num_stages)
            ])
        elif share == "share_none":
            self.stages = nn.ModuleList([
                HQSStage(make_update_net(), make_prox_net(), share_weights=False)
                for _ in range(self.num_stages)
            ])
        elif share == "share_update":
            shared_update = make_update_net()
            self.stages = nn.ModuleList([
                HQSStage(shared_update, make_prox_net(), share_weights=False)
                for _ in range(self.num_stages)
            ])
        elif share == "share_prox":
            shared_prox = make_prox_net()
            self.stages = nn.ModuleList([
                HQSStage(make_update_net(), shared_prox, share_weights=False)
                for _ in range(self.num_stages)
            ])
        else:
            raise ValueError(f"Unknown weight_sharing mode: {share!r}")

        self.ctx_to_hidden = nn.Conv2d(ctx_dim, hidden_dim, 1)
        self.ctx_to_v = nn.Conv2d(ctx_dim, 2, 1)
        nn.init.zeros_(self.ctx_to_v.weight)
        nn.init.zeros_(self.ctx_to_v.bias)

        self._feature_dim = feat_dim

    def forward(
        self,
        image1: torch.Tensor,
        image2: torch.Tensor,
        iters: Optional[int] = None,
        flow_init: Optional[torch.Tensor] = None,
    ) -> Dict[str, List[torch.Tensor]]:
        iters = iters or self.num_stages
        if iters > self.num_stages:
            raise ValueError(
                f"Requested {iters} HQS stages, but model was built with {self.num_stages}."
            )

        img1 = self._normalise(image1)
        img2 = self._normalise(image2)

        fmap1 = self.feature_encoder(img1)
        fmap2 = self.feature_encoder(img2)
        fmap1 = F.normalize(fmap1, p=2, dim=1)
        fmap2 = F.normalize(fmap2, p=2, dim=1)
        context = self.context_encoder(img1)

        if isinstance(self.corr_block, CorrBlock):
            self.corr_block.initialise(fmap1, fmap2)

        batch_size, _, height, width = fmap1.shape
        if flow_init is not None:
            if flow_init.shape[-2:] != (height, width):
                flow_init = F.interpolate(
                    flow_init / self.upsample_scale,
                    size=(height, width),
                    mode="bilinear",
                    align_corners=True,
                )
            flow_u = flow_init.clone()
        else:
            flow_u = torch.zeros(
                batch_size, 2, height, width,
                device=fmap1.device,
                dtype=fmap1.dtype,
            )

        flow_v = torch.tanh(self.ctx_to_v(context))
        hidden = torch.tanh(self.ctx_to_hidden(context))

        flow_preds: List[torch.Tensor] = []
        flow_lows: List[torch.Tensor] = []
        hidden_states: List[torch.Tensor] = []

        for stage_idx in range(iters):
            flow_u, flow_v, hidden, up_mask = self.stages[stage_idx](
                fmap1, fmap2, self.corr_block, context, flow_u, flow_v, hidden
            )
            flow_lows.append(flow_u)
            hidden_states.append(hidden)

            if up_mask is not None:
                flow_up = convex_upsample(flow_u, up_mask, scale=self.upsample_scale)
            else:
                flow_up = upsample_flow(flow_u, scale_factor=self.upsample_scale)
            flow_preds.append(flow_up)

        return {
            "flow_preds": flow_preds,
            "flow_low": flow_lows,
            "hidden_states": hidden_states,
        }

    @staticmethod
    def _normalise(img: torch.Tensor) -> torch.Tensor:
        if img.max() > 2.0:
            img = img / 255.0
        mean = img.new_tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = img.new_tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        return (img - mean) / std

    def param_count(self) -> Dict[str, int]:
        def count(module: nn.Module) -> int:
            return sum(param.numel() for param in module.parameters())

        return {
            "feature_encoder": count(self.feature_encoder),
            "context_encoder": count(self.context_encoder),
            "stages": count(self.stages),
            "total": count(self),
        }
