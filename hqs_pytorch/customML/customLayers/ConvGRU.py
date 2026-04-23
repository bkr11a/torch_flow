"""
Convolutional GRU layer for PyTorch.

Implements gated recurrent unit with convolutional operations.
"""

__author__ = "Brad Rice"
__version__ = "1.0.0"

import torch
import torch.nn as nn


class ConvGRU(nn.Module):
    \"\"\"
    Convolutional GRU cell for sequential processing.
    
    Combines GRU recurrence with spatial convolutions.
    \"\"\"
    
    def __init__(self, in_channels: int, hidden_channels: int, kernel_size: int = 3):
        \"\"\"
        Initialize ConvGRU.
        
        Args:
            in_channels: Number of input channels
            hidden_channels: Number of hidden state channels
            kernel_size: Convolutional kernel size
        \"\"\"
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        
        # Reset gate
        self.conv_reset = nn.Conv2d(
            in_channels + hidden_channels,
            hidden_channels,
            kernel_size,
            padding=self.padding,
            bias=True
        )
        
        # Update gate
        self.conv_update = nn.Conv2d(
            in_channels + hidden_channels,
            hidden_channels,
            kernel_size,
            padding=self.padding,
            bias=True
        )
        
        # Candidate hidden state
        self.conv_candidate = nn.Conv2d(
            in_channels + hidden_channels,
            hidden_channels,
            kernel_size,
            padding=self.padding,
            bias=True
        )
    
    def forward(
        self,
        x: torch.Tensor,
        h: torch.Tensor = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        \"\"\"
        Forward pass through ConvGRU cell.
        
        Args:
            x: Input tensor [B, C, H, W]
            h: Previous hidden state [B, H_hidden, H, W]
            
        Returns:
            (y, h_new) where y is output and h_new is new hidden state
        \"\"\"
        batch_size, channels, height, width = x.shape
        
        # Initialize hidden state if not provided
        if h is None:
            h = torch.zeros(
                batch_size,
                self.hidden_channels,
                height,
                width,
                device=x.device,
                dtype=x.dtype
            )
        
        # Concatenate input and hidden state
        combined = torch.cat([x, h], dim=1)
        
        # Reset gate
        reset = torch.sigmoid(self.conv_reset(combined))
        
        # Update gate
        update = torch.sigmoid(self.conv_update(combined))
        
        # Candidate hidden state
        combined_reset = torch.cat([x, reset * h], dim=1)
        candidate = torch.tanh(self.conv_candidate(combined_reset))
        
        # Compute new hidden state
        h_new = (1 - update) * h + update * candidate
        
        return h_new, h_new
