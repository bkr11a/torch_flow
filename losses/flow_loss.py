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

    Loss = Σ_k  γ^{K-k-1}  ·  mean_valid( charbonnier(||u^k - u_gt||) )

    where K is the total number of stages, γ∈(0,1) favours later stages.

    Args:
        gamma:   Geometric decay (default 0.85).
        max_flow: Ignore pixels with GT flow magnitude above this threshold
                  (default 400 px, consistent with RAFT).
        loss_fn: "charbonnier" | "l1" | "l2"
    """

    def __init__(
        self,
        gamma: float = 0.85,
        max_flow: float = 400.0,
        loss_fn: str = "charbonnier",
    ) -> None:
        super().__init__()
        self.gamma    = gamma
        self.max_flow = max_flow
        self.loss_fn  = loss_fn

    def _point_loss(self, pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        diff = pred - gt
        if self.loss_fn == "charbonnier":
            return charbonnier(diff).mean(dim=1)  # (B, H, W)
        if self.loss_fn == "l1":
            return diff.abs().mean(dim=1)
        if self.loss_fn == "l2":
            return (diff ** 2).mean(dim=1).sqrt()
        raise ValueError(self.loss_fn)

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

        for k, pred in enumerate(flow_preds):
            weight = self.gamma ** (K - k - 1)

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
        photo_weight:      float (default 0.0)
        ofce_weight:       float (default 0.0)
    """

    def __init__(self, cfg) -> None:
        super().__init__()
        self.seq_loss = SequenceLoss(
            gamma=cfg.get("gamma", 0.85),
            max_flow=cfg.get("max_flow", 400.0),
            loss_fn=cfg.get("loss_fn", "charbonnier"),
        )
        self.smooth_weight = cfg.get("smooth_weight", 0.0)
        self.photo_weight  = cfg.get("photo_weight",  0.0)
        self.ofce_weight   = cfg.get("ofce_weight",   0.0)

        # Always instantiate auxiliary terms so we can report their values
        # even when their weights are zero.
        self.smooth_loss = SmoothnessLoss()
        self.photo_loss = PhotometricLoss()
        self.ofce_loss = OFCELoss()

    def forward(
        self,
        flow_preds: List[torch.Tensor],
        flow_gt: torch.Tensor,
        valid: torch.Tensor,
        image1: Optional[torch.Tensor] = None,
        image2: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        out = self.seq_loss(flow_preds, flow_gt, valid)
        total = out["loss"]
        pred_final = flow_preds[-1]

        if image1 is not None:
            if self.smooth_weight > 0:
                s = self.smooth_loss(pred_final, image1)
            else:
                with torch.no_grad():
                    s = self.smooth_loss(pred_final.detach(), image1)
            out["smooth"] = s
            if self.smooth_weight > 0:
                total = total + self.smooth_weight * s

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

        out["loss"] = total
        return out
