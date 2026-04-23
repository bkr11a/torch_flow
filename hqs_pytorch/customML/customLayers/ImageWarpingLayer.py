"""
Image warping layer for PyTorch.

Performs bilinear interpolation warping of images based on optical flow fields.
"""

__author__ = "Brad Rice"
__version__ = "1.0.0"

import torch
import torch.nn as nn
import torch.nn.functional as F


class ImageWarpingLayer(nn.Module):
    """
    Warps images using optical flow via bilinear interpolation.
    
    Given an image and a flow field, warps the image by displacing each pixel
    according to the flow. Uses bilinear interpolation for sub-pixel accuracy.
    
    Flow format: [B, H, W, 2] where [:,:,:,0] = dy (vertical) and [:,:,:,1] = dx (horizontal)
    """
    
    def __init__(self):
        """Initialize image warping layer."""
        super().__init__()
    
    def forward(
        self, 
        image: torch.Tensor, 
        flow: torch.Tensor,
        padding_mode: str = 'zeros'
    ) -> torch.Tensor:
        """
        Warp image according to flow field.
        
        Args:
            image: Image tensor of shape [B, C, H, W]
            flow: Optical flow of shape [B, H, W, 2]
                  Format: [dy, dx] (vertical, horizontal displacements)
            padding_mode: How to handle out-of-bounds ('zeros', 'border')
            
        Returns:
            Warped image of shape [B, C, H, W]
        """
        batch_size, channels, height, width = image.shape
        
        # Convert flow from [B, H, W, 2] to [B, 2, H, W] for grid_sample
        # Reorder from [dy, dx] to [dx, dy] for grid_sample
        flow = flow.permute(0, 3, 1, 2)  # [B, 2, H, W]
        
        # Create coordinate grid
        # grid_sample expects normalized coordinates in [-1, 1]
        coords_y = torch.linspace(-1, 1, height, device=image.device)
        coords_x = torch.linspace(-1, 1, width, device=image.device)
        grid_y, grid_x = torch.meshgrid(coords_y, coords_x, indexing='ij')  # [H, W]
        
        # Stack into [H, W, 2] format expected by grid_sample
        grid = torch.stack([grid_x, grid_y], dim=-1)  # [H, W, 2]
        grid = grid.unsqueeze(0).expand(batch_size, -1, -1, -1)  # [B, H, W, 2]
        
        # Normalize flow to match coordinate grid scale
        # Flow is in pixel units, convert to normalized coordinates
        flow_normalized = torch.zeros_like(flow)
        flow_normalized[:, 0, :, :] = flow[:, 1, :, :] * (2.0 / width)  # dx
        flow_normalized[:, 1, :, :] = flow[:, 0, :, :] * (2.0 / height)  # dy
        
        # Reorder to [B, H, W, 2] for grid_sample
        flow_normalized = flow_normalized.permute(0, 2, 3, 1)  # [B, H, W, 2]
        
        # Add flow to grid for warped coordinates
        sampling_grid = grid + flow_normalized
        
        # Warp using grid_sample
        warped = F.grid_sample(
            image,
            sampling_grid,
            mode='bilinear',
            padding_mode=padding_mode,
            align_corners=True
        )
        
        return warped
