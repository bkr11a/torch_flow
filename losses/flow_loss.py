"""Optical flow losses for HQSFlow.

SequenceLoss   – Weighted sum of L1/Charbonnier losses over all HQS stages.
                 Later stages receive higher weight (γ-decay from the end).
PhotometricLoss – Unsupervised photometric consistency (for self-supervised experiments).
SmoothnessLoss  – First/second-order spatial smoothness regulariser (auxiliary).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional

from hqs_pytorch.customML.customModels.band_split_refinement import fixed_high_pass
from hqs_pytorch.customML.customModels.factorised_reliability import ReliabilityState
from hqs_pytorch.customML.customModels.occlusion_geometry import (
    backward_warp_yx,
    flow_in_bounds_mask,
)
from .occlusion_aware_losses import (
    forward_backward_cycle_loss,
    matchability_supervision_loss,
    reliability_temporal_loss,
    visibility_prior_loss,
    visibility_supervision_loss,
    visible_outlier_mixture_nll,
)


# ---------------------------------------------------------------------------
# Charbonnier (robust L1) loss
# ---------------------------------------------------------------------------

def charbonnier(x: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    return (x.pow(2) + eps ** 2).sqrt()


# ---------------------------------------------------------------------------
# Sequence loss (main supervised loss)
# ---------------------------------------------------------------------------

class SequenceLoss(nn.Module):
    """
    Compute a weighted sum of per-stage endpoint errors.

    Two weighting modes:

    ``geometric`` (default):
        Loss = Σ_k  γ^{K-k-1}  ·  mean_valid( charbonnier(||u^k - u_gt||) )
        where K is the total number of stages, γ∈(0,1) favours later stages.

    ``learnable``:
        Per-stage weights are learned via a softmax over ``stage_logits``.  A
        uniform-floor mixture prevents the trivial solution of collapsing all
        weight onto a single stage (which would short-circuit early-stage
        training)::

            w_k = (1 - min_frac) * softmax(logits)[k] + min_frac * (1/K)

        This guarantees every stage receives at least ``stage_weight_min_frac``
        of its fair-share weight regardless of what gradient descent does.

    Args:
        gamma:   Geometric decay (default 0.85, only used in geometric mode).
        max_flow: Ignore pixels with GT flow magnitude above this threshold.
        loss_fn: "charbonnier" | "l1" | "l2"
        stage_weight_mode: "geometric" | "learnable"
        num_stages: Number of stages to initialise learnable logits for.
                    Extra runtime stages are handled by zero-padding.
        stage_weight_min_frac: Minimum fractional weight floor (learnable mode).
    """

    def __init__(
        self,
        gamma: float = 0.85,
        max_flow: float = 400.0,
        loss_fn: str = "charbonnier",
        stage_weight_mode: str = "geometric",
        num_stages: int = 12,
        stage_weight_min_frac: float = 0.15,
    ) -> None:
        super().__init__()
        self.gamma    = gamma
        self.max_flow = max_flow
        self.loss_fn  = loss_fn
        self.stage_weight_mode = stage_weight_mode
        self.stage_weight_min_frac = float(stage_weight_min_frac)
        if stage_weight_mode == "learnable":
            # Initialised to zero so softmax starts near-uniform; training will
            # adapt the distribution.  The minimum floor prevents collapse.
            self.stage_logits = nn.Parameter(torch.zeros(num_stages))

    def _point_loss(self, pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        diff = pred - gt
        if self.loss_fn == "charbonnier":
            return charbonnier(diff).mean(dim=1)  # (B, H, W)
        if self.loss_fn == "l1":
            return diff.abs().mean(dim=1)
        if self.loss_fn == "l2":
            return (diff ** 2).mean(dim=1).sqrt()
        raise ValueError(self.loss_fn)

    def get_stage_weights(self, K: int, device: torch.device) -> torch.Tensor:
        """Return per-stage loss weights of shape (K,), normalised to sum to 1.

        Geometric mode reproduces the classic \u03b3^{K-k-1} schedule.
        Learnable mode mixes softmax weights with a uniform floor so no stage
        can be completely ignored during training.
        """
        if self.stage_weight_mode == "learnable":
            n_init = self.stage_logits.shape[0]
            if K <= n_init:
                logits = self.stage_logits[:K]
            else:
                # Zero-pad for any extra stages beyond the initialised count.
                pad = self.stage_logits.new_zeros(K - n_init)
                logits = torch.cat([self.stage_logits, pad])
            logits = logits.to(device)
            soft = torch.softmax(logits, dim=0)
            uniform = torch.ones(K, device=device, dtype=soft.dtype) / K
            weights = (1.0 - self.stage_weight_min_frac) * soft + self.stage_weight_min_frac * uniform
            return weights / weights.sum()
        else:
            weights = torch.tensor(
                [self.gamma ** (K - k - 1) for k in range(K)],
                dtype=torch.float32,
                device=device,
            )
            return weights / weights.sum()

    def forward(
        self,
        flow_preds: List[torch.Tensor],   # list of (B, 2, H, W), full-res
        flow_gt: torch.Tensor,            # (B, 2, H, W)
        valid: torch.Tensor,              # (B, H, W)  0/1
    ) -> Dict[str, torch.Tensor]:
        """
        Returns dict with keys:
            "loss"   – total weighted loss (scalar)
            "epe"    – endpoint error at the *last* stage (for logging)
        """
        K = len(flow_preds)
        total_loss = flow_gt.new_zeros(())

        # Exclude GT-invalid and too-large-flow pixels
        mag = (flow_gt ** 2).sum(dim=1).sqrt()   # (B, H, W)
        valid_mask = valid.bool() & (mag < self.max_flow)

        stage_weights = self.get_stage_weights(K, flow_gt.device)

        for k, pred in enumerate(flow_preds):
            weight = stage_weights[k]

            # Upsample GT to match prediction resolution (should already match)
            gt_k = flow_gt
            if pred.shape[-2:] != flow_gt.shape[-2:]:
                scale_h = pred.shape[-2] / flow_gt.shape[-2]
                scale_w = pred.shape[-1] / flow_gt.shape[-1]
                gt_k = F.interpolate(flow_gt, size=pred.shape[-2:],
                                     mode="bilinear", align_corners=True)
                gt_k[:, 0] *= scale_w
                gt_k[:, 1] *= scale_h
                vm = F.interpolate(
                    valid_mask.float().unsqueeze(1),
                    size=pred.shape[-2:], mode="nearest"
                ).squeeze(1).bool()
            else:
                vm = valid_mask

            loss_k = self._point_loss(pred, gt_k)    # (B, H, W)
            if vm.any():
                total_loss = total_loss + weight * loss_k[vm].mean()

        # EPE at the final stage
        epe = (flow_preds[-1] - flow_gt).pow(2).sum(dim=1).sqrt()
        epe_mean = epe[valid_mask].mean() if valid_mask.any() else epe.mean()

        return {"loss": total_loss, "epe": epe_mean}


# ---------------------------------------------------------------------------
# Photometric loss (self-supervised auxiliary)
# ---------------------------------------------------------------------------

class PhotometricLoss(nn.Module):
    """
    SSIM + L1 photometric consistency loss.
    Useful for semi-supervised or self-supervised training.
    """

    def __init__(self, alpha: float = 0.85) -> None:
        super().__init__()
        self.alpha = alpha

    def forward(
        self,
        image1: torch.Tensor,   # (B, 3, H, W)
        image2: torch.Tensor,   # (B, 3, H, W)
        flow: torch.Tensor,     # (B, 2, H, W)
    ) -> torch.Tensor:
        from models.warp import backward_warp
        warped = backward_warp(image2, flow)
        l1 = (image1 - warped).abs().mean(dim=1, keepdim=True)
        ssim_val = self._ssim(image1, warped)
        return (self.alpha * ssim_val + (1 - self.alpha) * l1).mean()

    @staticmethod
    def _ssim(x: torch.Tensor, y: torch.Tensor,
              window_size: int = 11) -> torch.Tensor:
        """Simplified single-scale SSIM map."""
        C1, C2 = 0.01 ** 2, 0.03 ** 2
        mu_x = F.avg_pool2d(x, window_size, 1, window_size // 2)
        mu_y = F.avg_pool2d(y, window_size, 1, window_size // 2)
        mu_x2, mu_y2 = mu_x ** 2, mu_y ** 2
        mu_xy = mu_x * mu_y
        sig_x  = F.avg_pool2d(x * x, window_size, 1, window_size // 2) - mu_x2
        sig_y  = F.avg_pool2d(y * y, window_size, 1, window_size // 2) - mu_y2
        sig_xy = F.avg_pool2d(x * y, window_size, 1, window_size // 2) - mu_xy
        num = (2 * mu_xy + C1) * (2 * sig_xy + C2)
        den = (mu_x2 + mu_y2 + C1) * (sig_x + sig_y + C2)
        return (1.0 - num / den.clamp_min(1e-8)) / 2.0


# ---------------------------------------------------------------------------
# Smoothness loss (auxiliary regularisation during training)
# ---------------------------------------------------------------------------

class SmoothnessLoss(nn.Module):
    """First-order edge-aware spatial smoothness penalty."""

    def __init__(self, order: int = 1, edge_weight: float = 150.0) -> None:
        super().__init__()
        assert order in (1, 2)
        self.order       = order
        self.edge_weight = edge_weight

    def forward(
        self,
        flow: torch.Tensor,   # (B, 2, H, W)
        image: torch.Tensor,  # (B, 3, H, W)  for edge weighting
    ) -> torch.Tensor:
        if self.order == 1:
            return self._first_order(flow, image)
        return self._second_order(flow, image)

    def _first_order(self, flow, image):
        dx_flow = (flow[:, :, :, 1:] - flow[:, :, :, :-1]).abs().sum(dim=1)
        dy_flow = (flow[:, :, 1:] - flow[:, :, :-1, :]).abs().sum(dim=1)
        dx_img  = (image[:, :, :, 1:] - image[:, :, :, :-1]).abs().mean(dim=1)
        dy_img  = (image[:, :, 1:] - image[:, :, :-1, :]).abs().mean(dim=1)
        wx = torch.exp(-self.edge_weight * dx_img)
        wy = torch.exp(-self.edge_weight * dy_img)
        return (wx * dx_flow).mean() + (wy * dy_flow).mean()

    def _second_order(self, flow, image):
        dxx = (flow[:, :, :, 2:] - 2 * flow[:, :, :, 1:-1] + flow[:, :, :, :-2]
               ).abs().sum(dim=1)
        dyy = (flow[:, :, 2:] - 2 * flow[:, :, 1:-1] + flow[:, :, :-2, :]
               ).abs().sum(dim=1)
        dx2_img = (image[:, :, :, 2:] - 2 * image[:, :, :, 1:-1] + image[:, :, :, :-2]
                   ).abs().mean(dim=1)
        dy2_img = (image[:, :, 2:] - 2 * image[:, :, 1:-1] + image[:, :, :-2, :]
                   ).abs().mean(dim=1)
        wx = torch.exp(-self.edge_weight * dx2_img)
        wy = torch.exp(-self.edge_weight * dy2_img)
        return (wx * dxx).mean() + (wy * dyy).mean()


# ---------------------------------------------------------------------------
# OFCE Loss (Optical Flow Constraint Equation)
# ---------------------------------------------------------------------------

class OFCELoss(nn.Module):
    """
    Optical Flow Constraint Equation loss.
    
    Enforces the brightness constancy assumption:
        I_t + ∇I · u = 0
    
    This encourages flow to satisfy the fundamental optical flow equation.
    """

    def __init__(self, weight: float = 1.0) -> None:
        super().__init__()
        self.weight = weight

    def forward(
        self,
        image1: torch.Tensor,   # (B, 3, H, W) or (B, 1, H, W)
        image2: torch.Tensor,   # (B, 3, H, W) or (B, 1, H, W)
        flow: torch.Tensor,     # (B, 2, H, W)
    ) -> torch.Tensor:
        """
        Compute OFCE loss.
        
        Args:
            image1: Reference image
            image2: Target image  
            flow: Predicted optical flow [dx, dy]
            
        Returns:
            Scalar loss value
        """
        # Compute temporal gradient I_t = I2 - I1
        # Average across channels
        if image1.shape[1] == 3:
            i1 = image1.mean(dim=1, keepdim=True)
            i2 = image2.mean(dim=1, keepdim=True)
        else:
            i1 = image1
            i2 = image2
        
        i_t = i2 - i1  # (B, 1, H, W)

        # Compute spatial gradients ∇I = [I_x, I_y]
        kernel_x = torch.tensor(
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
            dtype=image1.dtype, device=image1.device
        ).view(1, 1, 3, 3) / 8.0
        
        kernel_y = torch.tensor(
            [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
            dtype=image1.dtype, device=image1.device
        ).view(1, 1, 3, 3) / 8.0

        i_x = F.conv2d(i1, kernel_x, padding=1)  # (B, 1, H, W)
        i_y = F.conv2d(i1, kernel_y, padding=1)  # (B, 1, H, W)

        # Extract flow components
        u = flow[:, 0:1, :, :]  # dx (B, 1, H, W)
        v = flow[:, 1:2, :, :]  # dy (B, 1, H, W)

        # Compute OFCE residual: I_t + I_x * u + I_y * v
        ofce_residual = i_t + i_x * u + i_y * v  # (B, 1, H, W)

        # Return mean squared residual
        return self.weight * (ofce_residual ** 2).mean()


# ---------------------------------------------------------------------------
# Combined loss
# ---------------------------------------------------------------------------

class HQSFlowLoss(nn.Module):
    """
    Master loss: SequenceLoss + optional auxiliary terms.

    cfg fields:
        gamma:             float (default 0.85)
        max_flow:          float (default 400)
        loss_fn:           str   (default "charbonnier")
        smooth_weight:     float (default 0.0)
        inter_smooth_weight: float (default 0.0)
        boundary_weight:   float (default 0.0)
        boundary_thresh:   float (default 0.03)
        raw_final_weight:  float (default 0.0)
        iter_damp_weight:  float (default 0.0)
        iter_damp_start_frac: float (default 0.5)
        photo_weight:      float (default 0.0)
        ofce_weight:       float (default 0.0)
    """

    def __init__(self, cfg) -> None:
        super().__init__()
        self.seq_loss = SequenceLoss(
            gamma=cfg.get("gamma", 0.85),
            max_flow=cfg.get("max_flow", 400.0),
            loss_fn=cfg.get("loss_fn", "charbonnier"),
            stage_weight_mode=cfg.get("stage_weight_mode", "geometric"),
            num_stages=cfg.get("num_stages", 12),
            stage_weight_min_frac=cfg.get("stage_weight_min_frac", 0.15),
        )
        self.smooth_weight = cfg.get("smooth_weight", 0.0)
        self.inter_smooth_weight = cfg.get("inter_smooth_weight", 0.0)
        self.boundary_weight = cfg.get("boundary_weight", 0.0)
        self.boundary_thresh = cfg.get("boundary_thresh", 0.03)
        self.raw_final_weight = cfg.get("raw_final_weight", 0.0)
        self.iter_damp_weight = cfg.get("iter_damp_weight", 0.0)
        self.iter_damp_start_frac = cfg.get("iter_damp_start_frac", 0.5)
        self.photo_weight  = cfg.get("photo_weight",  0.0)
        self.ofce_weight   = cfg.get("ofce_weight",   0.0)
        self.detail_weight = float(cfg.get("detail_weight", 0.0))
        self.detail_start_step = int(cfg.get("detail_start_step", 0))
        self.detail_ramp_steps = int(cfg.get("detail_ramp_steps", 0))
        self.detail_boundary_threshold = float(
            cfg.get("detail_boundary_threshold", 1.0)
        )
        self.detail_boundary_boost = float(
            cfg.get("detail_boundary_boost", 2.0)
        )
        self.detail_high_pass_factor = int(
            cfg.get("detail_high_pass_factor", 2)
        )
        self.global_init_weight = float(cfg.get("global_init_weight", 0.0))
        self.global_confidence_weight = float(
            cfg.get("global_confidence_weight", 0.0)
        )
        self.global_confidence_tau = float(
            cfg.get("global_confidence_tau", 1.0)
        )
        self.global_large_motion_threshold = float(
            cfg.get("global_large_motion_threshold", 40.0)
        )
        self.global_large_motion_boost = float(
            cfg.get("global_large_motion_boost", 2.0)
        )
        # Optional supervision for HQSCore's single learned reliability head.
        # Ground-truth masks remain loss targets and never enter model.forward.
        self.core_visibility_weight = float(
            cfg.get("core_visibility_weight", 0.0)
        )
        self.core_visibility_last_n = int(
            cfg.get("core_visibility_last_n", 4)
        )
        # Optional direct training signal for a learned correspondence
        # measurement. The proposal remains an internal data observation; this
        # loss never feeds ground truth into model.forward().
        self.proposal_supervision_weight = float(
            cfg.get("proposal_supervision_weight", 0.0)
        )
        self.proposal_matchability_weight = float(
            cfg.get("proposal_matchability_weight", 0.0)
        )
        self.proposal_confidence_tau = float(
            cfg.get("proposal_confidence_tau", 1.0)
        )
        self.proposal_supervision_last_n = int(
            cfg.get("proposal_supervision_last_n", 0)
        )

        occ_cfg = cfg.get("occlusion_aware", {})
        self.occlusion_aware_enabled = bool(occ_cfg.get("enabled", False))
        self.occ_mixture_weight = float(occ_cfg.get("mixture_weight", 0.02))
        self.occ_visibility_weight = float(
            occ_cfg.get("visibility_weight", 0.01)
        )
        self.occ_matchability_weight = float(
            occ_cfg.get("matchability_weight", 0.005)
        )
        self.occ_cycle_weight = float(occ_cfg.get("cycle_weight", 0.01))
        self.occ_temporal_weight = float(occ_cfg.get("temporal_weight", 0.001))
        self.occ_visibility_prior_weight = float(
            occ_cfg.get("visibility_prior_weight", 0.001)
        )
        self.occ_boundary_weight = float(
            occ_cfg.get("boundary_weight", 0.005)
        )
        self.occ_outlier_scale = float(occ_cfg.get("outlier_scale", 20.0))
        self.occ_detach_reverse_for_cycle = bool(
            occ_cfg.get("detach_reverse_for_cycle", True)
        )
        self.occ_expected_visible_fraction = float(
            occ_cfg.get("expected_visible_fraction", 0.85)
        )
        self.occ_visibility_prior_tolerance = float(
            occ_cfg.get("visibility_prior_tolerance", 0.10)
        )
        self.register_buffer(
            "_training_step", torch.zeros((), dtype=torch.long), persistent=False
        )

        # Always instantiate auxiliary terms so we can report their values
        # even when their weights are zero.
        self.smooth_loss = SmoothnessLoss()
        self.photo_loss = PhotometricLoss()
        self.ofce_loss = OFCELoss()

    def set_step(self, step: int) -> None:
        self._training_step.fill_(int(step))

    def _detail_weight_at_step(self) -> float:
        step = int(self._training_step.item())
        if step < self.detail_start_step:
            return 0.0
        if self.detail_ramp_steps <= 0:
            return self.detail_weight
        fraction = min(
            1.0,
            float(step - self.detail_start_step + 1)
            / float(self.detail_ramp_steps),
        )
        return self.detail_weight * fraction

    @staticmethod
    def _flow_grad(flow: torch.Tensor):
        dx = flow[:, :, :, 1:] - flow[:, :, :, :-1]
        dy = flow[:, :, 1:, :] - flow[:, :, :-1, :]
        return dx, dy

    @staticmethod
    def _mask4(
        mask: Optional[torch.Tensor],
        reference: torch.Tensor,
        *,
        default: float,
    ) -> torch.Tensor:
        b, _, h, w = reference.shape
        if mask is None:
            return reference.new_full((b, 1, h, w), float(default))
        if mask.ndim == 3:
            mask = mask.unsqueeze(1)
        if mask.ndim != 4:
            raise ValueError(f"Expected mask [B,H,W] or [B,1,H,W], got {mask.shape}")
        mask = mask.to(device=reference.device, dtype=reference.dtype)
        if mask.shape[1] != 1:
            mask = mask.mean(dim=1, keepdim=True)
        if mask.shape[-2:] != (h, w):
            mask = F.interpolate(mask, size=(h, w), mode="nearest")
        return mask.clamp(0.0, 1.0)

    @staticmethod
    def _resize_flow_gt(
        flow_gt_xy: torch.Tensor, size: tuple[int, int]
    ) -> torch.Tensor:
        in_h, in_w = flow_gt_xy.shape[-2:]
        out_h, out_w = size
        if (in_h, in_w) == (out_h, out_w):
            return flow_gt_xy
        result = F.interpolate(
            flow_gt_xy, size=size, mode="bilinear", align_corners=True
        )
        result = result.clone()
        result[:, 0] *= float(out_w) / float(in_w)
        result[:, 1] *= float(out_h) / float(in_h)
        return result

    def _motion_boundary_map(self, flow_gt: torch.Tensor) -> torch.Tensor:
        dx, dy = self._flow_grad(flow_gt)
        dx = F.pad(dx, (0, 1, 0, 0))
        dy = F.pad(dy, (0, 0, 0, 1))
        magnitude = torch.sqrt(
            dx.square().sum(1, keepdim=True)
            + dy.square().sum(1, keepdim=True)
            + 1e-9
        )
        return (magnitude > self.detail_boundary_threshold).to(flow_gt.dtype)

    def _detail_loss_and_metrics(
        self,
        prediction: torch.Tensor,
        flow_gt: torch.Tensor,
        valid: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        high_prediction = fixed_high_pass(
            prediction, factor=self.detail_high_pass_factor
        )
        high_gt = fixed_high_pass(flow_gt, factor=self.detail_high_pass_factor)
        point = charbonnier(high_prediction - high_gt).mean(1, keepdim=True)
        valid4 = self._mask4(valid, flow_gt, default=1.0)
        boundary = self._motion_boundary_map(flow_gt)
        weight = valid4 * (1.0 + self.detail_boundary_boost * boundary)
        detail = (point * weight).sum() / weight.sum().clamp_min(1.0)

        with torch.no_grad():
            recovery = (
                (high_prediction.abs() * valid4).sum()
                / (high_gt.abs() * valid4).sum().clamp_min(1e-6)
            )
            pred_flat = (high_prediction * valid4).flatten(1)
            gt_flat = (high_gt * valid4).flatten(1)
            alignment = (
                (pred_flat * gt_flat).sum(1)
                / (
                    pred_flat.norm(dim=1) * gt_flat.norm(dim=1)
                ).clamp_min(1e-6)
            ).mean()
        return detail, recovery, alignment

    def _visibility_targets(
        self,
        flow_gt_xy: torch.Tensor,
        valid: torch.Tensor,
        occlusion: Optional[torch.Tensor],
        invalid: Optional[torch.Tensor],
        synthetic_occlusion: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        valid4 = self._mask4(valid, flow_gt_xy, default=1.0)
        invalid4 = self._mask4(invalid, flow_gt_xy, default=0.0)
        occlusion4 = self._mask4(occlusion, flow_gt_xy, default=0.0)
        flow_gt_yx = torch.stack(
            (flow_gt_xy[:, 1], flow_gt_xy[:, 0]), dim=1
        )
        in_bounds = flow_in_bounds_mask(flow_gt_yx)

        erased_at_correspondence = torch.zeros_like(occlusion4)
        if synthetic_occlusion is not None:
            erased_target = self._mask4(
                synthetic_occlusion, flow_gt_xy, default=0.0
            )
            # Synthetic erase is recorded in target coordinates.  Ground-truth
            # flow is used here only to construct a loss target in source
            # coordinates; it never enters model.forward().
            erased_at_correspondence = backward_warp_yx(
                erased_target, flow_gt_yx
            ).clamp(0.0, 1.0)

        reliable_gt = valid4 * (1.0 - invalid4)
        visibility = (
            (1.0 - occlusion4)
            * (1.0 - erased_at_correspondence)
            * in_bounds
        ).clamp(0.0, 1.0)
        return visibility, reliable_gt

    def _core_visibility_loss(
        self,
        reliability_lows: List[torch.Tensor],
        flow_gt: torch.Tensor,
        valid: torch.Tensor,
        occlusion: Optional[torch.Tensor],
        invalid: Optional[torch.Tensor],
        synthetic_occlusion: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Supervise HQSCore reliability without leaking labels to inference."""
        visibility, reliable = self._visibility_targets(
            flow_gt,
            valid,
            occlusion,
            invalid,
            synthetic_occlusion,
        )
        selected = reliability_lows
        if self.core_visibility_last_n > 0:
            selected = selected[-self.core_visibility_last_n :]

        terms: List[torch.Tensor] = []
        for probability in selected:
            if not isinstance(probability, torch.Tensor):
                continue
            target = F.interpolate(
                visibility,
                size=probability.shape[-2:],
                mode="nearest",
            ).float()
            weight = F.interpolate(
                reliable,
                size=probability.shape[-2:],
                mode="nearest",
            ).float()
            # Evaluate BCE in float32: 1-1e-5 rounds to exactly one in fp16.
            probability = probability.float().clamp(1e-5, 1.0 - 1e-5)
            point = -(
                target * probability.log()
                + (1.0 - target) * (1.0 - probability).log()
            )
            terms.append(
                (point * weight).sum() / weight.sum().clamp_min(1.0)
            )
        if not terms:
            return flow_gt.new_zeros(())
        return torch.stack(terms).mean()

    def _proposal_measurement_loss(
        self,
        proposals: List[torch.Tensor],
        matchabilities: List[torch.Tensor],
        flow_gt: torch.Tensor,
        valid: torch.Tensor,
        occlusion: Optional[torch.Tensor],
        invalid: Optional[torch.Tensor],
        synthetic_occlusion: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Supervise correspondence observations only where they are defined.

        Matchability targets combine geometric visibility with the detached
        accuracy of the current proposal. This does not force a textureless
        but visible pixel to claim high correspondence confidence.
        """
        visibility, reliable = self._visibility_targets(
            flow_gt,
            valid,
            occlusion,
            invalid,
            synthetic_occlusion,
        )
        pairs = list(zip(proposals, matchabilities))
        if self.proposal_supervision_last_n > 0:
            pairs = pairs[-self.proposal_supervision_last_n :]

        proposal_terms: List[torch.Tensor] = []
        matchability_terms: List[torch.Tensor] = []
        for proposal, matchability in pairs:
            if not isinstance(proposal, torch.Tensor):
                continue
            gt_low = self._resize_flow_gt(flow_gt, proposal.shape[-2:])
            visible_low = F.interpolate(
                visibility, size=proposal.shape[-2:], mode="nearest"
            )
            reliable_low = F.interpolate(
                reliable, size=proposal.shape[-2:], mode="nearest"
            )
            proposal_weight = visible_low * reliable_low
            proposal_point = charbonnier(proposal - gt_low).mean(
                dim=1, keepdim=True
            )
            proposal_terms.append(
                (proposal_point * proposal_weight).sum()
                / proposal_weight.sum().clamp_min(1.0)
            )

            if not isinstance(matchability, torch.Tensor):
                continue
            if matchability.shape[-2:] != proposal.shape[-2:]:
                matchability = F.interpolate(
                    matchability,
                    size=proposal.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
            proposal_epe = (proposal - gt_low).square().sum(
                dim=1, keepdim=True
            ).sqrt()
            matchability_target = visible_low * torch.exp(
                -proposal_epe.detach()
                / max(self.proposal_confidence_tau, 1e-6)
            )
            probability = matchability.float().clamp(1e-5, 1.0 - 1e-5)
            target = matchability_target.float()
            point = -(
                target * probability.log()
                + (1.0 - target) * (1.0 - probability).log()
            )
            matchability_terms.append(
                (point * reliable_low.float()).sum()
                / reliable_low.sum().clamp_min(1.0)
            )

        zero = flow_gt.new_zeros(())
        proposal_loss = (
            torch.stack(proposal_terms).mean() if proposal_terms else zero
        )
        matchability_loss = (
            torch.stack(matchability_terms).mean()
            if matchability_terms
            else zero
        )
        return proposal_loss, matchability_loss

    def _factorised_reliability_loss(
        self,
        model_outputs: Dict[str, object],
        flow_gt: torch.Tensor,
        valid: torch.Tensor,
        occlusion: Optional[torch.Tensor],
        invalid: Optional[torch.Tensor],
        synthetic_occlusion: Optional[torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        states = model_outputs.get("reliability_states", [])
        low_predictions = model_outputs.get("flow_low", [])
        if not isinstance(states, list) or not states:
            return {}

        visibility, reliable_gt = self._visibility_targets(
            flow_gt,
            valid,
            occlusion,
            invalid,
            synthetic_occlusion,
        )
        mixture_terms = []
        visibility_terms = []
        match_terms = []
        boundary_terms = []
        prior_terms = []
        temporal_terms = []
        boundary_full = self._motion_boundary_map(flow_gt) * reliable_gt

        previous: Optional[ReliabilityState] = None
        for index, state in enumerate(states):
            if not isinstance(state, ReliabilityState):
                continue
            size = state.visibility_logits.shape[-2:]
            visibility_low = F.interpolate(
                visibility, size=size, mode="nearest"
            )
            reliable_low = F.interpolate(
                reliable_gt, size=size, mode="nearest"
            )
            visibility_terms.append(
                visibility_supervision_loss(
                    state, visibility_low, valid=reliable_low
                )
            )
            match_terms.append(
                matchability_supervision_loss(
                    state, visibility_low, valid=reliable_low
                )
            )
            # Motion boundaries are loss-only GT geometry.  Max pooling keeps
            # one-pixel structures alive when constructing a lower-resolution
            # target for the predicted regularisation boundary factor.
            boundary_low = F.adaptive_max_pool2d(boundary_full, size)
            positives = (boundary_low * reliable_low).sum()
            negatives = ((1.0 - boundary_low) * reliable_low).sum()
            positive_weight = (negatives / positives.clamp_min(1.0)).clamp(
                1.0, 20.0
            )
            boundary_point = F.binary_cross_entropy_with_logits(
                state.boundary_logits,
                boundary_low,
                reduction="none",
                pos_weight=positive_weight,
            )
            boundary_terms.append(
                (boundary_point * reliable_low).sum()
                / reliable_low.sum().clamp_min(1.0)
            )
            prior_terms.append(
                visibility_prior_loss(
                    state,
                    expected_visible_fraction=self.occ_expected_visible_fraction,
                    tolerance=self.occ_visibility_prior_tolerance,
                )
            )
            if previous is not None and previous.p_visible.shape == state.p_visible.shape:
                temporal_terms.append(reliability_temporal_loss(state, previous))
            previous = state

            if isinstance(low_predictions, list) and index < len(low_predictions):
                prediction_low = low_predictions[index]
                gt_low = self._resize_flow_gt(flow_gt, size)
                mixture_terms.append(
                    visible_outlier_mixture_nll(
                        prediction_low - gt_low,
                        state,
                        valid=reliable_low,
                        outlier_scale=self.occ_outlier_scale,
                    )
                )

        def mean_or_zero(values: List[torch.Tensor]) -> torch.Tensor:
            if not values:
                return flow_gt.new_zeros(())
            return torch.stack(values).mean()

        result = {
            "occ_mixture": mean_or_zero(mixture_terms),
            "occ_visibility": mean_or_zero(visibility_terms),
            "occ_matchability": mean_or_zero(match_terms),
            "occ_boundary": mean_or_zero(boundary_terms),
            "occ_temporal": mean_or_zero(temporal_terms),
            "occ_visibility_prior": mean_or_zero(prior_terms),
        }

        flow_ab = model_outputs.get("gmflow_init_flow_yx")
        flow_ba = model_outputs.get("gmflow_reverse_flow_yx")
        if isinstance(flow_ab, torch.Tensor) and isinstance(flow_ba, torch.Tensor):
            result["occ_cycle"] = forward_backward_cycle_loss(
                flow_ab,
                flow_ba,
                detach_reverse=self.occ_detach_reverse_for_cycle,
            )
        else:
            result["occ_cycle"] = flow_gt.new_zeros(())
        return result

    def _boundary_loss(
        self,
        pred: torch.Tensor,
        gt: torch.Tensor,
        valid: torch.Tensor,
    ) -> torch.Tensor:
        pdx, pdy = self._flow_grad(pred)
        gdx, gdy = self._flow_grad(gt)

        gmag_x = (gdx.pow(2).sum(dim=1) + 1e-9).sqrt()
        gmag_y = (gdy.pow(2).sum(dim=1) + 1e-9).sqrt()

        vx = valid[:, :, 1:] & valid[:, :, :-1]
        vy = valid[:, 1:, :] & valid[:, :-1, :]
        bx = vx & (gmag_x > self.boundary_thresh)
        by = vy & (gmag_y > self.boundary_thresh)

        loss = gt.new_zeros(())
        count = 0

        if bx.any():
            loss = loss + (pdx - gdx).abs().mean(dim=1)[bx].mean()
            count += 1
        if by.any():
            loss = loss + (pdy - gdy).abs().mean(dim=1)[by].mean()
            count += 1

        if count == 0:
            return loss
        return loss / float(count)

    def _iteration_damping(
        self,
        flow_preds: List[torch.Tensor],
        low_predictions: Optional[List[torch.Tensor]] = None,
    ) -> torch.Tensor:
        if len(flow_preds) < 2:
            return flow_preds[0].new_zeros(())

        start_idx = int(len(flow_preds) * self.iter_damp_start_frac)
        start_idx = max(1, min(start_idx, len(flow_preds) - 1))
        terms = []
        for i in range(start_idx, len(flow_preds)):
            # A coarse-to-fine transition is expected to make a substantial
            # correction.  Do not penalise it as recurrent instability.
            if (
                low_predictions is not None
                and i < len(low_predictions)
                and low_predictions[i].shape[-2:]
                != low_predictions[i - 1].shape[-2:]
            ):
                continue
            terms.append((flow_preds[i] - flow_preds[i - 1]).abs().mean())
        if not terms:
            return flow_preds[0].new_zeros(())
        return torch.stack(terms).mean()

    def forward(
        self,
        flow_preds: List[torch.Tensor],
        flow_gt: torch.Tensor,
        valid: torch.Tensor,
        image1: Optional[torch.Tensor] = None,
        image2: Optional[torch.Tensor] = None,
        model_outputs: Optional[Dict[str, object]] = None,
        occlusion: Optional[torch.Tensor] = None,
        invalid: Optional[torch.Tensor] = None,
        synthetic_occlusion: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        out = self.seq_loss(flow_preds, flow_gt, valid)
        total = out["loss"]
        pred_final = flow_preds[-1]
        valid_mask = valid.bool()

        raw_preds = None
        if isinstance(model_outputs, dict):
            candidate = model_outputs.get("flow_preds_raw", None)
            if isinstance(candidate, list) and len(candidate) > 0:
                raw_preds = candidate

        if self.raw_final_weight > 0 and raw_preds is not None:
            raw_final_dict = self.seq_loss([raw_preds[-1]], flow_gt, valid)
            out["raw_final"] = raw_final_dict["loss"]
            total = total + self.raw_final_weight * raw_final_dict["loss"]

        if self.iter_damp_weight > 0:
            damp_source = raw_preds if raw_preds is not None else flow_preds
            low_predictions = None
            if isinstance(model_outputs, dict):
                candidate_low = model_outputs.get("flow_low", None)
                if isinstance(candidate_low, list):
                    low_predictions = candidate_low
            damp = self._iteration_damping(damp_source, low_predictions)
            out["iter_damp"] = damp
            total = total + self.iter_damp_weight * damp

        if self.boundary_weight > 0:
            bnd = self._boundary_loss(pred_final, flow_gt, valid_mask)
            out["boundary"] = bnd
            total = total + self.boundary_weight * bnd

        if image1 is not None:
            if self.smooth_weight > 0:
                s = self.smooth_loss(pred_final, image1)
            else:
                with torch.no_grad():
                    s = self.smooth_loss(pred_final.detach(), image1)
            out["smooth"] = s
            if self.smooth_weight > 0:
                total = total + self.smooth_weight * s

            if self.inter_smooth_weight > 0 and len(flow_preds) > 1:
                inter_terms = []
                for pred in flow_preds[:-1]:
                    inter_terms.append(self.smooth_loss(pred, image1))
                inter_smooth = torch.stack(inter_terms).mean()
                out["smooth_inter"] = inter_smooth
                total = total + self.inter_smooth_weight * inter_smooth

        if image1 is not None and image2 is not None:
            if self.photo_weight > 0:
                p = self.photo_loss(image1, image2, pred_final)
            else:
                with torch.no_grad():
                    p = self.photo_loss(image1, image2, pred_final.detach())
            out["photo"] = p
            if self.photo_weight > 0:
                total = total + self.photo_weight * p

        if image1 is not None and image2 is not None:
            if self.ofce_weight > 0:
                o = self.ofce_loss(image1, image2, pred_final)
            else:
                with torch.no_grad():
                    o = self.ofce_loss(image1, image2, pred_final.detach())
            out["ofce"] = o
            if self.ofce_weight > 0:
                total = total + self.ofce_weight * o

        detail_weight = self._detail_weight_at_step()
        if detail_weight > 0:
            detail, recovery, alignment = self._detail_loss_and_metrics(
                pred_final, flow_gt, valid
            )
        else:
            with torch.no_grad():
                detail, recovery, alignment = self._detail_loss_and_metrics(
                    pred_final.detach(), flow_gt, valid
                )
        out["detail"] = detail
        out["hf_recovery"] = recovery
        out["hf_alignment"] = alignment
        if detail_weight > 0:
            total = total + detail_weight * detail
        out["detail_weight"] = total.new_tensor(detail_weight)

        if isinstance(model_outputs, dict):
            initial_yx = model_outputs.get("gmflow_init_flow_yx")
            if isinstance(initial_yx, torch.Tensor):
                initial_xy = torch.stack(
                    (initial_yx[:, 1], initial_yx[:, 0]), dim=1
                )
                gt_initial = self._resize_flow_gt(
                    flow_gt, initial_xy.shape[-2:]
                )
                # A matching/data proposal must not be trained to explain
                # pixels without an observation in frame 2.  Those pixels are
                # supervised through the final source-conditioned solution.
                visible_full, reliable_full = self._visibility_targets(
                    flow_gt,
                    valid,
                    occlusion,
                    invalid,
                    synthetic_occlusion,
                )
                valid_initial = F.interpolate(
                    visible_full * reliable_full,
                    size=initial_xy.shape[-2:],
                    mode="nearest",
                )
                magnitude_full = flow_gt.square().sum(1, keepdim=True).sqrt()
                large_motion = F.interpolate(
                    (
                        magnitude_full >= self.global_large_motion_threshold
                    ).to(flow_gt.dtype),
                    size=initial_xy.shape[-2:],
                    mode="nearest",
                )
                initial_weight = valid_initial * (
                    1.0 + self.global_large_motion_boost * large_motion
                )
                init_point = charbonnier(initial_xy - gt_initial).mean(
                    dim=1, keepdim=True
                )
                init_loss = (
                    (init_point * initial_weight).sum()
                    / initial_weight.sum().clamp_min(1.0)
                )
                out["global_init"] = init_loss
                if self.global_init_weight > 0:
                    total = total + self.global_init_weight * init_loss

                confidence = model_outputs.get("gmflow_init_conf")
                if isinstance(confidence, torch.Tensor):
                    if confidence.shape[-2:] != initial_xy.shape[-2:]:
                        confidence = F.interpolate(
                            confidence,
                            size=initial_xy.shape[-2:],
                            mode="bilinear",
                            align_corners=False,
                        )
                    init_epe = (initial_xy - gt_initial).square().sum(
                        1, keepdim=True
                    ).sqrt()
                    confidence_target = torch.exp(
                        -init_epe.detach()
                        / max(self.global_confidence_tau, 1e-6)
                    )
                    eps = 1e-6
                    confidence_point = -(
                        confidence_target * torch.log(confidence + eps)
                        + (1.0 - confidence_target)
                        * torch.log(1.0 - confidence + eps)
                    )
                    confidence_loss = (
                        (confidence_point * valid_initial).sum()
                        / valid_initial.sum().clamp_min(1.0)
                    )
                    out["global_confidence"] = confidence_loss
                    if self.global_confidence_weight > 0:
                        total = (
                            total
                            + self.global_confidence_weight * confidence_loss
                        )

            core_reliability = model_outputs.get("core_reliability_lows")
            if isinstance(core_reliability, list) and core_reliability:
                core_visibility = self._core_visibility_loss(
                    core_reliability,
                    flow_gt,
                    valid,
                    occlusion,
                    invalid,
                    synthetic_occlusion,
                )
                out["core_visibility"] = core_visibility
                if self.core_visibility_weight > 0:
                    total = (
                        total
                        + self.core_visibility_weight * core_visibility
                    )

            proposals = model_outputs.get("match_proposal_lows")
            matchabilities = model_outputs.get("matchability_lows")
            if (
                isinstance(proposals, list)
                and proposals
                and isinstance(matchabilities, list)
                and len(proposals) == len(matchabilities)
                and (
                    self.proposal_supervision_weight > 0
                    or self.proposal_matchability_weight > 0
                )
            ):
                proposal_loss, matchability_loss = (
                    self._proposal_measurement_loss(
                        proposals,
                        matchabilities,
                        flow_gt,
                        valid,
                        occlusion,
                        invalid,
                        synthetic_occlusion,
                    )
                )
                out["proposal_supervision"] = proposal_loss
                out["proposal_matchability"] = matchability_loss
                total = (
                    total
                    + self.proposal_supervision_weight * proposal_loss
                    + self.proposal_matchability_weight * matchability_loss
                )

        if self.occlusion_aware_enabled and isinstance(model_outputs, dict):
            occ_terms = self._factorised_reliability_loss(
                model_outputs,
                flow_gt,
                valid,
                occlusion,
                invalid,
                synthetic_occlusion,
            )
            out.update(occ_terms)
            total = total + self.occ_mixture_weight * occ_terms.get(
                "occ_mixture", total.new_zeros(())
            )
            total = total + self.occ_visibility_weight * occ_terms.get(
                "occ_visibility", total.new_zeros(())
            )
            total = total + self.occ_matchability_weight * occ_terms.get(
                "occ_matchability", total.new_zeros(())
            )
            total = total + self.occ_boundary_weight * occ_terms.get(
                "occ_boundary", total.new_zeros(())
            )
            total = total + self.occ_cycle_weight * occ_terms.get(
                "occ_cycle", total.new_zeros(())
            )
            total = total + self.occ_temporal_weight * occ_terms.get(
                "occ_temporal", total.new_zeros(())
            )
            total = total + self.occ_visibility_prior_weight * occ_terms.get(
                "occ_visibility_prior", total.new_zeros(())
            )

        out["loss"] = total
        return out
