"""
Custom loss functions for PyTorch HQS model.
"""

from .AEPE_Loss import AEPE_Loss
from .OFCE_Loss import OFCE_Loss
from .AngularErrorLoss import AngularErrorLoss
from .OpticalFlowPhysicsLoss import OpticalFlowPhysicsLoss

__all__ = [
    'AEPE_Loss',
    'OFCE_Loss',
    'AngularErrorLoss',
    'OpticalFlowPhysicsLoss',
]
