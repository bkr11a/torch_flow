"""
HQS Iterations layer for PyTorch.

Implements hierarchical quadratic solver iterations with attention integration.
"""

__author__ = "Brad Rice"
__version__ = "1.0.0"

import torch
import torch.nn as nn
from typing import List
from src.hqs_pytorch.customML.customLayers.GradientDescentLayer import GradientDescentLayer


class HQSIterationLayer(nn.Module):
    """
    HQS iteration layer with attention integration.
    
    Performs multiple gradient descent steps for optical flow refinement
    with attention mechanism integration.
    
    FIX Issue #13: Added shape validation before concatenation
    """
    
    def __init__(self, num_gradient_descent_iterations: int = 10, init_learning_rate: float = 1e-5):
        \"\"\"
        Initialize HQS iteration layer.
        
        Args:
            num_gradient_descent_iterations: Number of GD steps per iteration
            init_learning_rate: Initial learning rate
        \"\"\"
        super().__init__()
        self.num_iterations = num_gradient_descent_iterations
        self.learning_rate = init_learning_rate
        
        # Create gradient descent layers for each iteration
        self.gradient_descent_layers = nn.ModuleList([
            GradientDescentLayer(learning_rate=init_learning_rate)
            for _ in range(num_gradient_descent_iterations)
        ])
        
        # Attention integration layers
        self.skipA = nn.Conv2d(1, 1, 3, padding=1)
        self.skipB = nn.Conv2d(1, 1, 3, padding=1, activation=F.relu)
        self.skipC = nn.Conv2d(1, 1, 3, padding=1)
        self.skipD = nn.Conv2d(1, 1, 3, padding=1)
        self.skipE = nn.Conv2d(1, 1, 3, padding=1)
        
        self.attnconvA = nn.Conv2d(32, 32, 3, padding=1)
        self.attnconvB = nn.Conv2d(32, 32, 3, padding=1)
        
        self.conv_h_u = nn.Conv2d(16, 16, 3, padding=1)
        self.conv_h_v = nn.Conv2d(16, 16, 3, padding=1)
    
    def _integrate_attention(
        self,
        attention: torch.Tensor,
        features: torch.Tensor
    ) -> torch.Tensor:
        \"\"\"
        Integrate attention mechanism with features.
        
        FIX Issue #13: Validate shapes before concatenation
        
        Args:
            attention: Attention map [B, C_a, H, W]
            features: Feature map [B, C_f, H, W]
            
        Returns:
            Attention-integrated features [B, C', H, W]
        \"\"\"
        # Validate shapes match
        assert attention.shape[2:] == features.shape[2:], \\
            f\"Spatial dims mismatch: attention {attention.shape[2:]}, features {features.shape[2:]}\"
        
        features_skip = self.skipC(features)
        attention = self.attnconvB(attention)
        attention = self.attnconvA(attention)
        attention = self.skipD(attention)
        
        # Concatenate attention and features
        combined = torch.cat([features, attention], dim=1)
        integrated = torch.nn.functional.relu(combined)
        integrated = self.skipE(integrated)
        
        # Add skip connection
        integrated = integrated + features_skip
        
        return integrated
    
    def forward(
        self,
        inputs: tuple
    ) -> tuple:
        \"\"\"
        Forward pass through HQS iterations.
        
        Args:
            inputs: Tuple of (attention, F1, F2, I_t, I_x, I_y, u_prev, v_prev, p_prev, q_prev, hidden_u, hidden_v, forward_pass_dict)
            
        Returns:
            (u_next, v_next, p_next, q_next, hidden_u, hidden_v, forward_pass_dict)
        \"\"\"
        (
            attention, F1, F2, I_t, I_x, I_y,
            u_prev, v_prev, p_prev, q_prev,
            hidden_u, hidden_v, forward_pass_dict
        ) = inputs
        
        # Integrate attention with feature maps
        I_t = self._integrate_attention(attention, I_t)
        I_x = self._integrate_attention(attention, I_x)
        I_y = self._integrate_attention(attention, I_y)
        
        forward_pass_dict.update({
            'integrated_I_t': I_t,
            'integrated_I_x': I_x,
            'integrated_I_y': I_y
        })
        
        # Perform HQS iterations using gradient descent
        u_next = u_prev
        v_next = v_prev
        p_next = p_prev
        q_next = q_prev
        
        for i, gd_layer in enumerate(self.gradient_descent_layers):
            u_next, v_next, p_next, q_next, hidden_u, hidden_v, forward_pass_dict = gd_layer(
                attention, I_t, I_x, I_y,
                u_next, v_next, p_next, q_next,
                hidden_u, hidden_v,
                forward_pass_dict
            )
            
            # Store iteration results
            forward_pass_dict.update({
                f'iteration_{i}': {
                    'u': u_next,
                    'v': v_next,
                    'p': p_next,
                    'q': q_next
                }
            })
        
        # Update hidden states with final flow
        hidden_u = torch.tanh(self.conv_h_u(torch.cat([u_next, F1, attention, hidden_u], dim=1)))
        hidden_v = torch.tanh(self.conv_h_v(torch.cat([v_next, F1, attention, hidden_v], dim=1)))
        
        forward_pass_dict.update({
            'hidden_u': hidden_u,
            'hidden_v': hidden_v
        })
        
        return u_next, v_next, p_next, q_next, hidden_u, hidden_v, forward_pass_dict
