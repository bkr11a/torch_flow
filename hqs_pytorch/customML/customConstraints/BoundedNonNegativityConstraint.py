"""
Bounded non-negativity constraint for PyTorch parameters.

Constrains parameters to remain within a specified range [lower_bound, upper_bound].
"""

__author__ = "Brad Rice"
__version__ = "1.0.0"

import torch
import torch.nn as nn


class BoundedNonNegativityConstraint:
    """
    Enforces bounded non-negativity constraint on parameters.
    
    Clips parameter values to be within [lower_bound, upper_bound] range.
    """
    
    def __init__(self, lower_bound: float = 0.0, upper_bound: float = 1.0):
        """
        Initialize the constraint.
        
        Args:
            lower_bound: Minimum allowed parameter value
            upper_bound: Maximum allowed parameter value
        """
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        
        if lower_bound > upper_bound:
            raise ValueError(
                f"lower_bound ({lower_bound}) must be <= upper_bound ({upper_bound})"
            )
    
    def __call__(self, module: nn.Module) -> None:
        """
        Apply bounded non-negativity constraint to module parameters.
        
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
