"""
Average End-Point Error (AEPE) loss for optical flow.

Standard metric for optical flow evaluation.
"""

__author__ = "Brad Rice"
__version__ = "1.0.0"

import torch
import torch.nn as nn


class AEPE_Loss(nn.Module):
    """
    Average End-Point Error loss.
    
    Computes mean Euclidean distance between predicted and ground truth flow.
    Handles invalid/occlusion masks.
    """
    
    def __init__(self, reduction: str = 'mean'):
        """
        Initialize AEPE loss.
        
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
        Compute AEPE loss.
        
        Args:
            flow_pred: Predicted flow [B, 2, H, W]
            flow_gt: Ground truth flow [B, 2, H, W]
            mask: Valid pixel mask [B, 1, H, W] (optional)
            
        Returns:
            Loss value (scalar or per-pixel)
        """
        # Compute end-point error
        epe = torch.norm(flow_pred - flow_gt, p=2, dim=1, keepdim=True)  # [B, 1, H, W]
        
        if mask is not None:
            epe = epe * mask
            count = torch.sum(mask, dim=[1, 2, 3], keepdim=True)
            count = torch.clamp(count, min=1.0)
            loss = torch.sum(epe, dim=[1, 2, 3], keepdim=True) / count
        else:
            if self.reduction == 'mean':
                loss = torch.mean(epe)
            elif self.reduction == 'sum':
                loss = torch.sum(epe)
            else:
                loss = epe
        
        return loss
