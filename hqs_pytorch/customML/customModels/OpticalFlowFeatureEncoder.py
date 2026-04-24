"""
Optical flow feature encoder for PyTorch.

ResNet-based architecture for extracting multi-scale features from images.
"""

__author__ = "Brad Rice"
__version__ = "1.0.0"

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


class OpticalFlowFeatureEncoder(nn.Module):
    """
    Multi-scale feature encoder for optical flow.
    
    Uses ResNet-inspired blocks to extract hierarchical features
    from images at multiple scales.
    """
    
    def __init__(
        self,
        base_channels: int = 16,
        channel_multiplier: tuple = (1, 2, 3),
        blocks_per_stage: tuple = (2, 2, 2),
        groups: int = 8,
        output_projection_dim: int = 48
    ):
        """
        Initialize feature encoder.
        
        Args:
            base_channels: Base number of channels
            channel_multiplier: Channel multipliers for each stage
            blocks_per_stage: Number of blocks per stage
            groups: Number of groups for grouped convolutions
            output_projection_dim: Output projection dimension
        """
        super().__init__()
        self.base_channels = base_channels
        self.channel_multiplier = channel_multiplier
        self.output_projection_dim = output_projection_dim
        
        # Initial convolution
        self.conv1 = nn.Conv2d(3, base_channels, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(base_channels)
        
        # Residual stages
        channels = [base_channels * m for m in channel_multiplier]
        self.stages = nn.ModuleList()
        
        in_channels = base_channels
        for i, (out_channels, num_blocks) in enumerate(zip(channels, blocks_per_stage)):
            stage = self._make_stage(
                in_channels, out_channels, num_blocks,
                stride=2, groups=groups
            )
            self.stages.append(stage)
            in_channels = out_channels
        
        # Output projection
        self.output_projection = nn.Conv2d(
            channels[-1], output_projection_dim, kernel_size=1
        )
    
    def _make_stage(
        self,
        in_channels: int,
        out_channels: int,
        num_blocks: int,
        stride: int = 1,
        groups: int = 8
    ) -> nn.Sequential:
        """
        Create a residual stage.
        
        Args:
            in_channels: Input channels
            out_channels: Output channels
            num_blocks: Number of residual blocks
            stride: Stride for first block
            groups: Number of groups for grouped convolutions
            
        Returns:
            Sequential module of residual blocks
        """
        blocks = []
        blocks.append(
            ResidualBlock(
                in_channels, out_channels,
                stride=stride, groups=groups
            )
        )
        for _ in range(1, num_blocks):
            blocks.append(
                ResidualBlock(
                    out_channels, out_channels,
                    stride=1, groups=groups
                )
            )
        return nn.Sequential(*blocks)
    
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Extract multi-scale features.
        
        Args:
            x: Input image [B, 3, H, W]
            
        Returns:
            (final_features, feature_pyramid)
        """
        x = F.relu(self.bn1(self.conv1(x)))  # [B, base_channels, H/2, W/2]
        
        pyramid = []
        for stage in self.stages:
            x = stage(x)
            pyramid.append(x)
        
        # Project final features
        x = self.output_projection(x)
        
        return x, pyramid


class ResidualBlock(nn.Module):
    """
    Residual block for feature extraction.
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        groups: int = 8
    ):
        """
        Initialize residual block.
        
        Args:
            in_channels: Input channels
            out_channels: Output channels
            stride: Stride for first convolution
            groups: Number of groups for grouped conv
        """
        super().__init__()
        
        # First convolution
        self.conv1 = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=3, stride=stride, padding=1, groups=groups
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        # Second convolution
        self.conv2 = nn.Conv2d(
            out_channels, out_channels,
            kernel_size=3, stride=1, padding=1, groups=groups
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Shortcut
        self.shortcut = nn.Identity()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through residual block.
        """
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out
