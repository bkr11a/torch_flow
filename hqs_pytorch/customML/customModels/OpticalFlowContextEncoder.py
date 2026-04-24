"""
Optical flow context encoder for PyTorch.

UNET-inspired architecture for extracting context features.
"""

__author__ = "Brad Rice"
__version__ = "1.0.0"

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


class OpticalFlowContextEncoder(nn.Module):
    """
    Context encoder for optical flow using UNET architecture.
    
    Extracts contextual information with skip connections
    and produces context features and hidden states.
    """
    
    def __init__(
        self,
        base_channels: int = 16,
        context_dim: int = 64
    ):
        """
        Initialize context encoder.
        
        Args:
            base_channels: Base number of channels
            context_dim: Output context dimension
        """
        super().__init__()
        self.base_channels = base_channels
        self.context_dim = context_dim
        
        # Encoder
        self.enc1 = DoubleConv(3, base_channels, base_channels)
        self.pool1 = nn.MaxPool2d(2, 2)
        
        self.enc2 = DoubleConv(base_channels, base_channels*2, base_channels*2)
        self.pool2 = nn.MaxPool2d(2, 2)
        
        self.enc3 = DoubleConv(base_channels*2, base_channels*4, base_channels*4)
        self.pool3 = nn.MaxPool2d(2, 2)
        
        # Bottleneck
        self.bottleneck = DoubleConv(base_channels*4, base_channels*8, base_channels*8)
        
        # Decoder
        self.upconv3 = nn.ConvTranspose2d(base_channels*8, base_channels*4, 2, stride=2)
        self.dec3 = DoubleConv(base_channels*8, base_channels*4, base_channels*4)
        
        self.upconv2 = nn.ConvTranspose2d(base_channels*4, base_channels*2, 2, stride=2)
        self.dec2 = DoubleConv(base_channels*4, base_channels*2, base_channels*2)
        
        self.upconv1 = nn.ConvTranspose2d(base_channels*2, base_channels, 2, stride=2)
        self.dec1 = DoubleConv(base_channels*2, base_channels, base_channels)
        
        # Output heads
        self.context_head = nn.Conv2d(base_channels, context_dim, kernel_size=1)
        self.hidden_head = nn.Conv2d(base_channels, context_dim, kernel_size=1)
    
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Extract context features and hidden state.
        
        Args:
            x: Input tensor [B, 3, H, W]
            
        Returns:
            (context_features, hidden_state) each [B, context_dim, H, W]
        """
        # Encoder
        enc1 = self.enc1(x)  # [B, base, H, W]
        enc2 = self.enc2(self.pool1(enc1))  # [B, base*2, H/2, W/2]
        enc3 = self.enc3(self.pool2(enc2))  # [B, base*4, H/4, W/4]
        
        # Bottleneck
        bottleneck = self.bottleneck(self.pool3(enc3))  # [B, base*8, H/8, W/8]
        
        # Decoder
        dec3 = self.upconv3(bottleneck)  # [B, base*4, H/4, W/4]
        dec3 = torch.cat([dec3, enc3], dim=1)
        dec3 = self.dec3(dec3)
        
        dec2 = self.upconv2(dec3)  # [B, base*2, H/2, W/2]
        dec2 = torch.cat([dec2, enc2], dim=1)
        dec2 = self.dec2(dec2)
        
        dec1 = self.upconv1(dec2)  # [B, base, H, W]
        dec1 = torch.cat([dec1, enc1], dim=1)
        dec1 = self.dec1(dec1)
        
        # Output heads
        context = self.context_head(dec1)
        hidden = self.hidden_head(dec1)
        
        return context, hidden


class DoubleConv(nn.Module):
    """
    Double convolution block for UNET.
    """
    
    def __init__(
        self,
        in_channels: int,
        mid_channels: int,
        out_channels: int
    ):
        """
        Initialize double conv block.
        """
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, mid_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.conv2 = nn.Conv2d(mid_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        """
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        return x
