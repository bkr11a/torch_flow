"""
Non-negativity constraint for PyTorch parameters.

This constraint ensures that parameters remain non-negative during training
by projecting them back to the non-negative domain after updates.
"""

__author__ = "Brad Rice"
__version__ = "1.0.0"

import torch
import torch.nn as nn


class NonNegativityConstraint:
    """
    Enforces non-negativity constraint on parameters.
    
    This constraint clips parameter values to be >= 0 after each optimization step.
    """
    
    def __call__(self, module: nn.Module) -> None:
        """
        Apply non-negativity constraint to module parameters.
        
        Args:
            module: PyTorch module to constrain
        """
        for param in module.parameters():
            with torch.no_grad():
                param.clamp_(min=0.0)
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"
