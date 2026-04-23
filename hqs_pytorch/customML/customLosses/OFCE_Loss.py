"""
Optical Flow Constraint Equation (OFCE) loss for optical flow.

Physics-based loss enforcing the brightness constancy constraint.
"""

__author__ = "Brad Rice"
__version__ = "1.0.0"

import torch
import torch.nn as nn
import torch.nn.functional as F


class OFCE_Loss(nn.Module):
    \"\"\"
    Optical Flow Constraint Equation loss.
    
    Enforces brightness constancy: I1(x) - I2(x + flow) ≈ 0
    Computes residual after warping I2 with predicted flow.
    \"\"\"
    
    def __init__(self, reduction: str = 'mean'):
        \"\"\"
        Initialize OFCE loss.
        
        Args:
            reduction: 'mean', 'sum', or 'none'
        \"\"\"
        super().__init__()
        self.reduction = reduction
    
    def forward(
        self,
        I1: torch.Tensor,
        I2: torch.Tensor,
        flow: torch.Tensor,
        mask: torch.Tensor = None
    ) -> torch.Tensor:
        \"\"\"
        Compute OFCE loss.
        
        Args:
            I1: Reference image [B, C, H, W]
            I2: Target image [B, C, H, W]
            flow: Optical flow [B, 2, H, W] (dy, dx)
            mask: Valid pixel mask [B, 1, H, W]
            
        Returns:
            Loss value
        \"\"\"
        batch_size, channels, height, width = I1.shape
        
        # Create sampling grid
        coords_y = torch.linspace(-1, 1, height, device=I1.device)
        coords_x = torch.linspace(-1, 1, width, device=I1.device)
        grid_y, grid_x = torch.meshgrid(coords_y, coords_x, indexing='ij')
        grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0)  # [1, H, W, 2]
        grid = grid.expand(batch_size, -1, -1, -1)
        
        # Normalize flow to grid coordinates
        flow_normalized = torch.zeros_like(flow)
        flow_normalized[:, 0, :, :] = flow[:, 1, :, :] * (2.0 / width)  # dx
        flow_normalized[:, 1, :, :] = flow[:, 0, :, :] * (2.0 / height)  # dy
        flow_normalized = flow_normalized.permute(0, 2, 3, 1)  # [B, H, W, 2]
        
        # Warped sampling grid
        sampling_grid = grid + flow_normalized
        
        # Warp I2 to I1 space
        I2_warped = F.grid_sample(
            I2, sampling_grid,
            mode='bilinear',
            padding_mode='zeros',
            align_corners=True
        )
        
        # Brightness constancy residual
        residual = I1 - I2_warped  # [B, C, H, W]
        
        # FIX Issue #9: Average across channels instead of using first only
        if channels > 1:
            residual = torch.mean(residual, dim=1, keepdim=True)  # [B, 1, H, W]
        
        loss = torch.abs(residual)
        
        if mask is not None:
            loss = loss * mask
            count = torch.sum(mask, dim=[1, 2, 3], keepdim=True)
            count = torch.clamp(count, min=1.0)
            loss = torch.sum(loss, dim=[1, 2, 3], keepdim=True) / count
        else:
            if self.reduction == 'mean':
                loss = torch.mean(loss)
            elif self.reduction == 'sum':
                loss = torch.sum(loss)
        
        return loss
