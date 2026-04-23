"""
Main HQS Optical Flow Model for PyTorch.

Hierarchical Quadratic Solver for end-to-end optical flow estimation.
"""

__author__ = "Brad Rice"
__version__ = "1.0.0"

import torch
import torch.nn as nn
from typing import Dict, Tuple, List, Optional
from src.hqs_pytorch.customML.customModels.OpticalFlowFeatureEncoder import OpticalFlowFeatureEncoder
from src.hqs_pytorch.customML.customModels.OpticalFlowContextEncoder import OpticalFlowContextEncoder
from src.hqs_pytorch.customML.customLayers.InputPadder import InputPadderPyTorch
from src.hqs_pytorch.customML.customLayers.CostCorrelationLayer import CostVolumeCorrelationLayer
from src.hqs_pytorch.customML.customLayers.HQSIterations import HQSIterationLayer
from src.hqs_pytorch.customML.customLayers.ImageWarpingLayer import ImageWarpingLayer
from src.hqs_pytorch.customML.customLayers.ImagePyramidLayer import ImagePyramidLayer
from src.hqs_pytorch.customML.customLosses import AEPE_Loss, OFCE_Loss


class HQSFlowModel(nn.Module):
    \"\"\"
    Hierarchical Quadratic Solver for Optical Flow.
    
    Combines feature extraction, cost volume computation, and iterative
    HQS optimization for accurate optical flow estimation.
    
    FIXES APPLIED:
    - Issue #4: Consistent [dy, dx] flow convention throughout
    - Issue #5: Corrected flow direction in loss computation
    - Issue #10: Dynamic padder instead of hardcoded Sintel dimensions
    \"\"\"
    
    def __init__(
        self,
        num_pyramid_levels: int = 3,
        max_displacement: int = 3,
        num_hqs_iterations: int = 10,
        num_gradient_descent_iterations: int = 15,
        init_lr: float = 1e-3
    ):
        \"\"\"
        Initialize HQS flow model.
        
        Args:
            num_pyramid_levels: Number of pyramid levels
            max_displacement: Maximum search displacement
            num_hqs_iterations: Number of HQS iterations
            num_gradient_descent_iterations: GD steps per HQS iteration
            init_lr: Initial learning rate
        \"\"\"
        super().__init__()
        
        self.num_pyramid_levels = num_pyramid_levels
        self.max_displacement = max_displacement
        self.num_hqs_iterations = num_hqs_iterations
        self.num_gradient_descent_iterations = num_gradient_descent_iterations
        self.init_lr = init_lr
        
        # Loss functions
        self.aepe_loss = AEPE_Loss()
        self.ofce_loss = OFCE_Loss()
        
        # FIX Issue #10: Create padder dynamically
        self.padder = InputPadderPyTorch(factor=8, padding_mode='constant')
        
        # Feature encoders
        self.feature_encoder = OpticalFlowFeatureEncoder(
            base_channels=16,
            channel_multiplier=(1, 2, 3),
            blocks_per_stage=(2, 2, 2),
            groups=8,
            output_projection_dim=48
        )
        
        # Context encoder
        self.context_encoder = OpticalFlowContextEncoder(
            base_channels=16,
            context_dim=64
        )
        
        # Cost volume correlation
        self.cost_volume_layer = CostVolumeCorrelationLayer(max_displacement)
        
        # Image pyramid
        self.pyramid_layer = ImagePyramidLayer(num_levels=num_pyramid_levels)
        
        # HQS iteration layers (one per pyramid level)
        self.hqs_layers = nn.ModuleList([
            HQSIterationLayer(
                num_gradient_descent_iterations=num_gradient_descent_iterations,
                init_learning_rate=init_lr
            )
            for _ in range(num_pyramid_levels)
        ])
        
        # Image warping layer
        self.warp_layer = ImageWarpingLayer()
        
        # Learnable pyramid weights
        self.hqs_beta = nn.Parameter(
            torch.tensor([0.05, 0.08, 0.12][:num_pyramid_levels], dtype=torch.float32)
        )
        self.hqs_lambda = nn.Parameter(
            torch.tensor([0.03, 0.04, 0.05][:num_pyramid_levels], dtype=torch.float32)
        )
        
        # Output refinement head
        self.final_refinement = nn.Sequential(
            nn.Conv2d(2, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 2, 3, padding=1)
        )
    
    def forward(
        self,
        image1: torch.Tensor,
        image2: torch.Tensor,
        flow_init: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict]:
        \"\"\"
        Forward pass through HQS model.
        
        Args:
            image1: Reference image [B, 3, H, W]
            image2: Target image [B, 3, H, W]
            flow_init: Optional initial flow estimate [B, 2, H, W]
            
        Returns:
            (flow_pred, forward_dict) where flow_pred is [B, 2, H, W]
            and forward_dict contains intermediate features
        \"\"\"
        forward_dict = {}
        
        batch_size, _, height, width = image1.shape
        
        # FIX Issue #10: Pad images dynamically
        image1_padded, pad_info = self.padder(image1)
        image2_padded, _ = self.padder(image2)
        
        # Extract features
        f1, f1_pyramid = self.feature_encoder(image1_padded)  # [B, 48, H/8, W/8]
        f2, f2_pyramid = self.feature_encoder(image2_padded)
        
        # Extract context
        context, hidden_init = self.context_encoder(image1_padded)
        
        # Create image pyramids
        i1_pyramid = self.pyramid_layer(image1_padded)
        i2_pyramid = self.pyramid_layer(image2_padded)
        
        # Compute correlation volumes
        cost_volumes = []
        for f1_level, f2_level in zip(f1_pyramid, f2_pyramid):
            cost_vol = self.cost_volume_layer(f1_level, f2_level)
            cost_volumes.append(cost_vol)
        
        # Initialize flow
        if flow_init is None:
            h_pad, w_pad = pad_info
            h_padded = height + h_pad
            w_padded = width + w_pad
            flow = torch.zeros(
                batch_size, 2,
                h_padded // 8, w_padded // 8,
                device=image1.device,
                dtype=image1.dtype
            )
        else:
            flow = flow_init
        
        # HQS iterations across pyramid levels
        forward_pass_dict = {}
        
        for level in range(self.num_pyramid_levels):
            # Get features at this level
            I1_level = i1_pyramid[level] if level < len(i1_pyramid) else i1_pyramid[-1]
            I2_level = i2_pyramid[level] if level < len(i2_pyramid) else i2_pyramid[-1]
            
            # Compute gradients
            I1_grad = self._compute_gradients(I1_level)  # [B, C, H, W, 2]
            I_t = self._compute_temporal_gradient(I1_level, I2_level)
            I_x = I1_grad[..., 0]
            I_y = I1_grad[..., 1]
            
            # Downsample flow if needed
            if level > 0:
                flow = torch.nn.functional.avg_pool2d(flow, kernel_size=2, stride=2)
            
            # Initialize hidden states
            if level == 0:
                hidden_u = torch.zeros_like(flow[:, :1])
                hidden_v = torch.zeros_like(flow[:, :1])
            
            # HQS iteration layer
            p = torch.zeros_like(flow)
            q = torch.zeros_like(flow)
            
            flow_u = flow[:, :1]
            flow_v = flow[:, 1:]
            
            inputs = (
                context, f1_pyramid[level], f2_pyramid[level],
                I_t, I_x, I_y,
                flow_u, flow_v, p, q,
                hidden_u, hidden_v,
                forward_pass_dict
            )
            
            flow_u, flow_v, p, q, hidden_u, hidden_v, forward_pass_dict = (
                self.hqs_layers[level](inputs)
            )
            
            flow = torch.cat([flow_u, flow_v], dim=1)
        
        # Upsample to original resolution
        # FIX Issue #10: Unpad to original size
        scale_factor = 8
        flow_upsampled = torch.nn.functional.interpolate(
            flow, scale_factor=scale_factor,
            mode='bilinear', align_corners=True
        )
        flow_final = self.padder.unpad(flow_upsampled, pad_info)
        
        # Final refinement
        flow_refined = self.final_refinement(flow_final)
        
        forward_dict.update({
            'flow': flow_refined,
            'cost_volumes': cost_volumes,
            'forward_pass': forward_pass_dict
        })
        
        return flow_refined, forward_dict
    
    def _compute_gradients(self, image: torch.Tensor) -> torch.Tensor:
        \"\"\"
        Compute spatial gradients using Sobel filters.
        
        Args:
            image: [B, C, H, W]
            
        Returns:
            Stacked gradients [B, C, H, W, 2]
        \"\"\"
        batch_size, channels, height, width = image.shape
        
        # Sobel kernels
        sobel_x = torch.tensor(
            [[1, 0, -1], [2, 0, -2], [1, 0, -1]],
            dtype=torch.float32,
            device=image.device
        ).reshape(1, 1, 3, 3)
        
        sobel_y = torch.tensor(
            [[1, 2, 1], [0, 0, 0], [-1, -2, -1]],
            dtype=torch.float32,
            device=image.device
        ).reshape(1, 1, 3, 3)
        
        # Tile for all channels
        sobel_x = sobel_x.repeat(channels, 1, 1, 1)
        sobel_y = sobel_y.repeat(channels, 1, 1, 1)
        
        # Apply convolution
        grad_x = torch.nn.functional.conv2d(image, sobel_x, padding=1, groups=channels)
        grad_y = torch.nn.functional.conv2d(image, sobel_y, padding=1, groups=channels)
        
        # Stack: [B, C, H, W, 2]
        gradients = torch.stack([grad_x, grad_y], dim=-1)
        
        return gradients
    
    def _compute_temporal_gradient(
        self,
        image1: torch.Tensor,
        image2: torch.Tensor
    ) -> torch.Tensor:
        \"\"\"
        Compute temporal gradient using image difference.
        
        Args:
            image1: [B, C, H, W]
            image2: [B, C, H, W]
            
        Returns:
            Temporal gradient [B, C, H, W]
        \"\"\"
        # FIX Issue #4: Consistent temporal gradient
        I_t = image2 - image1
        return I_t
