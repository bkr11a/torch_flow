"""
Input padding layer for PyTorch.

Pads images to ensure dimensions are divisible by a specified factor.
"""

__author__ = "Brad Rice"
__version__ = "1.0.0"

import torch
import torch.nn as nn


class InputPadderPyTorch(nn.Module):
    """
    Pads input images to ensure divisibility by a factor.
    
    PyTorch port of InputPadderTF. Automatically calculates required padding
    to make dimensions divisible by a specified factor (default 8).
    
    This is important for models with multiple stride-2 downsampling layers.
    """
    
    def __init__(self, factor: int = 8, padding_mode: str = 'constant'):
        """
        Initialize the input padder.
        
        Args:
            factor: Dimensions must be divisible by this factor (default 8)
            padding_mode: Mode for padding ('constant', 'reflect', 'replicate')
        """
        super().__init__()
        self.factor = factor
        self.padding_mode = padding_mode
        self._h_pad = 0
        self._w_pad = 0
    
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int]]:
        """
        Pad input tensor if necessary.
        
        Args:
            x: Input tensor of shape [B, C, H, W]
            
        Returns:
            padded_x: Padded tensor
            pad_info: Tuple of (h_pad, w_pad) for later unpadding
        """
        batch_size, channels, height, width = x.shape
        
        # Calculate required padding
        h_pad = (self.factor - (height % self.factor)) % self.factor
        w_pad = (self.factor - (width % self.factor)) % self.factor
        
        if h_pad == 0 and w_pad == 0:
            return x, (0, 0)
        
        # Apply padding: (left, right, top, bottom)
        # torch.nn.functional.pad expects (left, right, top, bottom)
        if self.padding_mode == 'constant':
            x_padded = torch.nn.functional.pad(
                x, (0, w_pad, 0, h_pad), mode='constant', value=0
            )
        elif self.padding_mode == 'reflect':
            x_padded = torch.nn.functional.pad(
                x, (0, w_pad, 0, h_pad), mode='reflect'
            )
        elif self.padding_mode == 'replicate':
            x_padded = torch.nn.functional.pad(
                x, (0, w_pad, 0, h_pad), mode='replicate'
            )
        else:
            raise ValueError(f"Unknown padding mode: {self.padding_mode}")
        
        return x_padded, (h_pad, w_pad)
    
    def unpad(self, x: torch.Tensor, pad_info: tuple[int, int]) -> torch.Tensor:
        """
        Remove padding from a padded tensor.
        
        Args:
            x: Padded tensor
            pad_info: Tuple of (h_pad, w_pad) from forward pass
            
        Returns:
            Unpadded tensor
        """
        h_pad, w_pad = pad_info
        
        if h_pad == 0 and w_pad == 0:
            return x
        
        if h_pad > 0 and w_pad > 0:
            return x[:, :, :-h_pad, :-w_pad]
        elif h_pad > 0:
            return x[:, :, :-h_pad, :]
        else:
            return x[:, :, :, :-w_pad]
