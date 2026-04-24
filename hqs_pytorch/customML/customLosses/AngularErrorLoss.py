"""
Angular error loss for optical flow.

Measures angular difference between flow vectors.
"""

__author__ = "Brad Rice"
__version__ = "1.0.0"

import torch
import torch.nn as nn


class AngularErrorLoss(nn.Module):
    """
    Angular error between flow vectors.
    
    Measures angle between predicted and ground truth flow directions.
    Invariant to flow magnitude scaling.
    """
    
    def __init__(self, reduction: str = 'mean'):
        """
        Initialize angular error loss.
        
        Args:
            reduction: 'mean', 'sum', or 'none'
        """
        super().__init__()
        self.reduction = reduction
    
    def forward(
        self,
        flow_pred: torch.Tensor,
        flow_gt: torch.Tensor,
        mask: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Compute angular error loss.
        
        Args:
            flow_pred: Predicted flow [B, 2, H, W]
            flow_gt: Ground truth flow [B, 2, H, W]
            mask: Valid pixel mask [B, 1, H, W]
            
        Returns:
            Angular error in degrees
        """
        # Add small epsilon to avoid division by zero
        eps = 1e-8
        
        # Normalize flows to unit vectors
        norm_pred = torch.norm(flow_pred, p=2, dim=1, keepdim=True) + eps
        norm_gt = torch.norm(flow_gt, p=2, dim=1, keepdim=True) + eps
        
        flow_pred_norm = flow_pred / norm_pred
        flow_gt_norm = flow_gt / norm_gt
        
        # Dot product of normalized vectors
        dot_product = torch.sum(flow_pred_norm * flow_gt_norm, dim=1, keepdim=True)
        dot_product = torch.clamp(dot_product, -1.0, 1.0)
        
        # Angular error in radians then convert to degrees
        angle_rad = torch.acos(dot_product)
        angle_deg = angle_rad * 180.0 / torch.tensor(3.14159265359, device=flow_pred.device)
        
        if mask is not None:
            angle_deg = angle_deg * mask
            count = torch.sum(mask, dim=[1, 2, 3], keepdim=True)
            count = torch.clamp(count, min=1.0)
            loss = torch.sum(angle_deg, dim=[1, 2, 3], keepdim=True) / count
        else:
            if self.reduction == 'mean':
                loss = torch.mean(angle_deg)
            elif self.reduction == 'sum':
                loss = torch.sum(angle_deg)
            else:
                loss = angle_deg
        
        return loss
