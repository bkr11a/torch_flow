"""
Log-barrier constraint for PyTorch parameters.

Uses a logarithmic barrier function to enforce inequality constraints.
"""

__author__ = "Brad Rice"
__version__ = "1.0.0"

import torch
import torch.nn as nn


class LogBarrierConstraint:
    """
    Enforces log-barrier constraint on parameters.
    
    Penalizes parameters that approach bounds through a logarithmic barrier.
    Useful for interior-point optimization methods.
    """
    
    def __init__(self, lower_bound: float = 1e-6, upper_bound: float = 1.0):
        """
        Initialize the constraint.
        
        Args:
            lower_bound: Lower bound for interior point method
            upper_bound: Upper bound for interior point method
        """
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
    
    def compute_barrier_penalty(
        self, 
        param: torch.Tensor,
        barrier_weight: float = 1.0
    ) -> torch.Tensor:
        """
        Compute log-barrier penalty.
        
        Args:
            param: Parameter tensor
            barrier_weight: Weight for barrier penalty
            
        Returns:
            Scalar penalty term
        """
        lower_violation = torch.relu(self.lower_bound - param)
        upper_violation = torch.relu(param - self.upper_bound)
        
        # Log barrier: -log(x - lower) - log(upper - x)
        penalty = barrier_weight * (
            -torch.log(param - self.lower_bound + 1e-10).sum() -
            torch.log(self.upper_bound - param + 1e-10).sum()
        )
        
        # Add penalty for constraint violations
        penalty = penalty + 1e6 * (lower_violation.sum() + upper_violation.sum())
        
        return penalty
    
    def __call__(self, module: nn.Module) -> None:
        """
        Apply log-barrier constraint to module parameters.
        
        Args:
            module: PyTorch module to constrain
        """
        for param in module.parameters():
            with torch.no_grad():
                param.clamp_(min=self.lower_bound, max=self.upper_bound)
    
    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"lower_bound={self.lower_bound}, "
            f"upper_bound={self.upper_bound})"
        )
