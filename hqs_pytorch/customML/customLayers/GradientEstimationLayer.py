"""
Gradient estimation layer for PyTorch.

Computes spatial gradients using Sobel filters.

FIX FOR CRITICAL ISSUE #2:
- Original TF version returned tuple (x_grad_x, x_grad_y)
- PyTorch version correctly stacks gradients into [B, C, H, 2] output
"""

__author__ = "Brad Rice"
__version__ = "1.0.0"

import torch
import torch.nn as nn
import torch.nn.functional as F


class GradientEstimationLayer(nn.Module):
    """
    Estimates image gradients using Sobel filters.
    
    Computes spatial gradients (∂I/∂x and ∂I/∂y) using Sobel operators.
    Outputs stacked gradients instead of tuple for consistency.
    """
    
    def __init__(self):
        """Initialize gradient estimation layer."""
        super().__init__()
        
        # Define Sobel kernels (3x3)
        # Note: These are applied with DepthwiseConv2D equivalent in PyTorch
        sobel_x = torch.tensor(
            [[1, 0, -1], [2, 0, -2], [1, 0, -1]], 
            dtype=torch.float32
        ).reshape(1, 1, 3, 3)
        
        sobel_y = torch.tensor(
            [[1, 2, 1], [0, 0, 0], [-1, -2, -1]], 
            dtype=torch.float32
        ).reshape(1, 1, 3, 3)
        
        # Register as buffers (not trainable, but part of state_dict)
        self.register_buffer('sobel_x_kernel', sobel_x)
        self.register_buffer('sobel_y_kernel', sobel_y)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute image gradients.
        
        Args:
            x: Input tensor of shape [B, C, H, W]
            
        Returns:
            Stacked gradients of shape [B, C, H, 2]
            where [:, :, :, 0] = ∂I/∂x and [:, :, :, 1] = ∂I/∂y
        """
        batch_size, channels, height, width = x.shape
        
        # Tile kernels for all channels (depthwise convolution)
        sobel_x = self.sobel_x_kernel.repeat(channels, 1, 1, 1)
        sobel_y = self.sobel_y_kernel.repeat(channels, 1, 1, 1)
        
        # Apply depthwise convolution
        # groups=channels makes it depthwise (one filter per channel)
        grad_x = F.conv2d(x, sobel_x, padding=1, groups=channels)  # [B, C, H, W]
        grad_y = F.conv2d(x, sobel_y, padding=1, groups=channels)  # [B, C, H, W]
        
        # Stack gradients: [B, C, H, 2]
        # FIX: Return stacked tensor, not tuple
        gradients = torch.stack([grad_x, grad_y], dim=-1)  # [B, C, H, W, 2]
        
        # Reshape to [B, C, H, W, 2] for consistency
        # If averaging across channels is desired, do it here:
        # gradients = torch.stack([grad_x.mean(dim=1, keepdim=True), 
        #                          grad_y.mean(dim=1, keepdim=True)], dim=-1)
        
        return gradients
