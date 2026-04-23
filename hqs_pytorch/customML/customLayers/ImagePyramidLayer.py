"""
Image pyramid layer for PyTorch.

Creates multi-scale pyramid of images via downsampling.
"""

__author__ = "Brad Rice"
__version__ = "1.0.0"

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


class ImagePyramidLayer(nn.Module):
    """
    Creates image pyramid by recursive downsampling.
    
    Generates multiple scales of an image for hierarchical processing.
    """
    
    def __init__(self, num_levels: int = 3):
        """
        Initialize pyramid layer.
        
        Args:
            num_levels: Number of pyramid levels
        """
        super().__init__()
        self.num_levels = num_levels
    
    def forward(self, image: torch.Tensor) -> List[torch.Tensor]:
        \"\"\"
        Create image pyramid.
        
        Args:
            image: Input image [B, C, H, W]
            
        Returns:
            List of images at different scales
        \"\"\"
        pyramid = [image]
        
        for level in range(1, self.num_levels):
            # FIX Issue #6: Use torch operations instead of int() on tensors
            downsampled = F.avg_pool2d(pyramid[-1], kernel_size=2, stride=2)
            pyramid.append(downsampled)
        
        return pyramid
