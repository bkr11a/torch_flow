"""
Physics-based optical flow loss.

Enforces physical constraints on optical flow predictions.
"""

__author__ = "Brad Rice"
__version__ = "1.0.0"

import torch
import torch.nn as nn
import torch.nn.functional as F


class OpticalFlowPhysicsLoss(nn.Module):
    \"\"\"
    Physics-based optical flow loss.
    
    Combines OFCE, smoothness, and other physical constraints.
    \"\"\"
    
    def __init__(
        self,
        ofce_weight: float = 1.0,
        smoothness_weight: float = 0.1,
        reduction: str = 'mean'
    ):
        \"\"\"
        Initialize physics loss.
        
        Args:
            ofce_weight: Weight for brightness constancy term
            smoothness_weight: Weight for smoothness term
            reduction: 'mean', 'sum', or 'none'
        \"\"\"
        super().__init__()
        self.ofce_weight = ofce_weight
        self.smoothness_weight = smoothness_weight
        self.reduction = reduction
    
    def _compute_smoothness(self, flow: torch.Tensor) -> torch.Tensor:
        \"\"\"
        Compute flow smoothness regularization.
        
        Args:
            flow: Flow field [B, 2, H, W]
            
        Returns:
            Smoothness loss
        \"\"\"
        # Compute flow gradients
        dy = torch.abs(flow[:, :, 1:, :] - flow[:, :, :-1, :])
        dx = torch.abs(flow[:, :, :, 1:] - flow[:, :, :, :-1])
        
        # Smoothness is L1 norm of gradients
        smoothness = torch.mean(dy) + torch.mean(dx)
        return smoothness
    
    def forward(
        self,
        I1: torch.Tensor,
        I2: torch.Tensor,
        flow: torch.Tensor,
        flow_gt: torch.Tensor = None,
        mask: torch.Tensor = None
    ) -> torch.Tensor:
        \"\"\"
        Compute physics-based loss.
        
        Args:
            I1: Reference image [B, C, H, W]
            I2: Target image [B, C, H, W]
            flow: Predicted flow [B, 2, H, W]
            flow_gt: Ground truth flow [B, 2, H, W] (optional)
            mask: Valid pixel mask [B, 1, H, W]
            
        Returns:
            Combined loss value
        \"\"\"
        batch_size, channels, height, width = I1.shape
        
        # Create sampling grid for warping
        coords_y = torch.linspace(-1, 1, height, device=I1.device)
        coords_x = torch.linspace(-1, 1, width, device=I1.device)
        grid_y, grid_x = torch.meshgrid(coords_y, coords_x, indexing='ij')
        grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0)  # [1, H, W, 2]
        grid = grid.expand(batch_size, -1, -1, -1)
        
        # Normalize flow
        flow_normalized = torch.zeros_like(flow)
        flow_normalized[:, 0, :, :] = flow[:, 1, :, :] * (2.0 / width)  # dx
        flow_normalized[:, 1, :, :] = flow[:, 0, :, :] * (2.0 / height)  # dy
        flow_normalized = flow_normalized.permute(0, 2, 3, 1)  # [B, H, W, 2]
        
        # Warp
        sampling_grid = grid + flow_normalized
        I2_warped = F.grid_sample(
            I2, sampling_grid,
            mode='bilinear',
            padding_mode='zeros',
            align_corners=True
        )
        
        # OFCE term
        ofce = torch.abs(I1 - I2_warped)
        if channels > 1:
            ofce = torch.mean(ofce, dim=1, keepdim=True)
        
        if mask is not None:
            ofce = ofce * mask
        
        ofce_loss = torch.mean(ofce)
        
        # Smoothness term
        smooth_loss = self._compute_smoothness(flow)
        
        # Total loss
        loss = self.ofce_weight * ofce_loss + self.smoothness_weight * smooth_loss
        
        return loss
