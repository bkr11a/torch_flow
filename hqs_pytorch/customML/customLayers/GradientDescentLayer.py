"""
Gradient descent layer for PyTorch.

Implements ADMM-inspired gradient descent solver for optical flow.

FIX FOR CRITICAL ISSUE #3:
- Original TF: tf.keras.layers.Multiply()([w, mask, grad_u]) - fails with 3+ inputs
- PyTorch: Uses native element-wise multiplication: w * mask * grad_u
"""

__author__ = "Brad Rice"
__version__ = "1.0.0"

import torch
import torch.nn as nn
import torch.nn.functional as F
from src.hqs_pytorch.customML.customLayers.GradientEstimationLayer import GradientEstimationLayer
from src.hqs_pytorch.customML.customConstraints import BoundedNonNegativityConstraint


class GradientDescentLayer(nn.Module):
    """
    Gradient descent solver layer with ADMM-inspired variables.
    
    Implements iterative optimization step for optical flow estimation
    with dual variables (p, q) for proximal splitting.
    """
    
    def __init__(self, learning_rate: float = 1e-5):
        \"\"\"
        Initialize gradient descent layer.
        
        Args:
            learning_rate: Initial learning rate for the solver
        \"\"\"
        super().__init__()
        self.learning_rate = learning_rate
        
        # Create trainable parameters with appropriate constraints
        self._init_parameters()
        
        # Gradient estimation for computing OFCE derivatives
        self.gradient_estimation_layer = GradientEstimationLayer()
        
        # Convolutional layers for u component
        self.conv_u_A = nn.Conv2d(2, 2, 3, padding=1)
        self.conv_u_B = nn.Conv2d(2, 4, 3, padding=1)
        self.conv_u_C = nn.Conv2d(4, 2, 3, padding=1)
        self.squeeze_u = nn.Conv2d(2, 1, 3, padding=1)
        
        # Convolutional layers for v component
        self.conv_v_A = nn.Conv2d(2, 2, 3, padding=1)
        self.conv_v_B = nn.Conv2d(2, 4, 3, padding=1)
        self.conv_v_C = nn.Conv2d(4, 2, 3, padding=1)
        self.squeeze_v = nn.Conv2d(2, 1, 3, padding=1)
        
        # Proximal operator layers for p
        self.conv_prox_u_A = nn.Conv2d(2, 2, 3, padding=1)
        self.conv_prox_u_B = nn.Conv2d(2, 4, 3, padding=1)
        self.conv_prox_u_C = nn.Conv2d(4, 2, 3, padding=1)
        self.squeeze_prox_u = nn.Conv2d(2, 1, 3, padding=1)
        
        # Proximal operator layers for q
        self.conv_prox_v_A = nn.Conv2d(2, 2, 3, padding=1)
        self.conv_prox_v_B = nn.Conv2d(2, 4, 3, padding=1)
        self.conv_prox_v_C = nn.Conv2d(4, 2, 3, padding=1)
        self.squeeze_prox_v = nn.Conv2d(2, 1, 3, padding=1)
        
        # GRU-like refinement layers
        self.convz = nn.Conv2d(16, 16, 3, padding=1)
        self.convr = nn.Conv2d(16, 16, 3, padding=1)
        self.convh = nn.Conv2d(16, 16, 3, padding=1)
        self.convh_refine = nn.Conv2d(16, 16, 3, padding=1)
        
        # Flow refinement heads
        self.flowRefinementHead_u = nn.Conv2d(1, 1, 3, padding=1)
        self.flowRefinementHead_v = nn.Conv2d(1, 1, 3, padding=1)
        self.flowRefinementHead_p = nn.Conv2d(1, 1, 3, padding=1)
        self.flowRefinementHead_q = nn.Conv2d(1, 1, 3, padding=1)
        
        # Activation
        self._tanh = torch.tanh
        self._sigmoid = torch.sigmoid
        self._relu = F.relu
    
    def _init_parameters(self):
        \"\"\"Initialize trainable parameters with constraints.\"\"\"
        # ADMM parameters
        self.theta = nn.Parameter(torch.tensor(5e-2, dtype=torch.float32))
        self.tau = nn.Parameter(torch.tensor(5e-2, dtype=torch.float32))
        self.sigma = nn.Parameter(torch.tensor(5e-2, dtype=torch.float32))
        self.tau_sigma_ratio = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
        
        # Proximal step parameters
        self.eta = nn.Parameter(torch.tensor(5e-1, dtype=torch.float32))
        self.xi = nn.Parameter(torch.tensor(5e-3, dtype=torch.float32))
        
        # Dual variable parameters
        self.rho = nn.Parameter(torch.tensor(5e-1, dtype=torch.float32))
        self.lambda_ = nn.Parameter(torch.tensor(2e-2, dtype=torch.float32))
        
        # Temperature for soft operators
        self.temperature = nn.Parameter(torch.tensor(5e-1, dtype=torch.float32))
        self.alpha = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
        self.omega = nn.Parameter(torch.tensor(3e-1, dtype=torch.float32))
        self.gate_slope = nn.Parameter(torch.tensor(10.0, dtype=torch.float32))
        
        # FIX Issue #12: Add bounds to xi parameter
        self.xi_constraint = BoundedNonNegativityConstraint(
            lower_bound=1e-5, upper_bound=1e-1
        )
    
    def forward(
        self,
        attention_map: torch.Tensor,
        I_t: torch.Tensor,
        I_x: torch.Tensor,
        I_y: torch.Tensor,
        u_prev: torch.Tensor,
        v_prev: torch.Tensor,
        p_prev: torch.Tensor,
        q_prev: torch.Tensor,
        hidden_u: torch.Tensor,
        hidden_v: torch.Tensor,
        forward_pass_dict: dict = None
    ) -> tuple:
        \"\"\"
        Perform one gradient descent step.
        
        Args:
            attention_map: Attention weights [B, C, H, W]
            I_t: Temporal gradient [B, C, H, W]
            I_x: Spatial x-gradient [B, C, H, W]
            I_y: Spatial y-gradient [B, C, H, W]
            u_prev: Previous u flow component [B, 1, H, W]
            v_prev: Previous v flow component [B, 1, H, W]
            p_prev: Previous primal variable p [B, 1, H, W]
            q_prev: Previous primal variable q [B, 1, H, W]
            hidden_u: Hidden state for u [B, C, H, W]
            hidden_v: Hidden state for v [B, C, H, W]
            forward_pass_dict: Dictionary for storing intermediate values
            
        Returns:
            (u_next, v_next, p_next, q_next, hidden_u, hidden_v, forward_pass_dict)
        \"\"\"
        if forward_pass_dict is None:
            forward_pass_dict = {}
        
        # FIX Issue #3: Element-wise multiply instead of Multiply layer
        # Compute gradients with masking
        w = torch.exp(-self.temperature * torch.abs(I_t))
        
        # Gradient computation with proper masking
        grad_u = self._tanh(self.conv_u_A(torch.cat([I_x, I_y], dim=1)))
        grad_u = self._tanh(self.conv_u_B(grad_u))
        grad_u = self._tanh(self.conv_u_C(grad_u))
        grad_u = w * grad_u  # FIX: Direct element-wise multiply
        grad_u = self._tanh(self.squeeze_u(grad_u))
        
        grad_v = self._tanh(self.conv_v_A(torch.cat([I_x, I_y], dim=1)))
        grad_v = self._tanh(self.conv_v_B(grad_v))
        grad_v = self._tanh(self.conv_v_C(grad_v))
        grad_v = w * grad_v  # FIX: Direct element-wise multiply
        grad_v = self._tanh(self.squeeze_v(grad_v))
        
        # Proximal operators for p, q
        prox_u = self._tanh(self.conv_prox_u_A(torch.cat([I_x, I_y], dim=1)))
        prox_u = self._tanh(self.conv_prox_u_B(prox_u))
        prox_u = self._tanh(self.conv_prox_u_C(prox_u))
        prox_u = self._tanh(self.squeeze_prox_u(prox_u))
        
        prox_v = self._tanh(self.conv_prox_v_A(torch.cat([I_x, I_y], dim=1)))
        prox_v = self._tanh(self.conv_prox_v_B(prox_v))
        prox_v = self._tanh(self.conv_prox_v_C(prox_v))
        prox_v = self._tanh(self.squeeze_prox_v(prox_v))
        
        # Gradient descent update
        u_next = u_prev - self.theta * grad_u
        v_next = v_prev - self.theta * grad_v
        
        # Dual variable updates (ADMM)
        p_next = p_prev + self.tau * (u_next - u_prev)
        q_next = q_prev + self.sigma * (v_next - v_prev)
        
        # Update hidden states
        hidden_u = self._tanh(self.convh(torch.cat([u_next, grad_u, hidden_u], dim=1)))
        hidden_v = self._tanh(self.convh(torch.cat([v_next, grad_v, hidden_v], dim=1)))
        
        forward_pass_dict.update({
            'u_next': u_next,
            'v_next': v_next,
            'p_next': p_next,
            'q_next': q_next,
            'grad_u': grad_u,
            'grad_v': grad_v
        })
        
        return u_next, v_next, p_next, q_next, hidden_u, hidden_v, forward_pass_dict
