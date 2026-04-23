"""
PyTorch HQS Model - Main package initialization.
"""

__version__ = "1.0.0"
__author__ = "Brad Rice"

from src.hqs_pytorch.customML.customConstraints import (
    NonNegativityConstraint,
    BoundedNonNegativityConstraint,
    LogBarrierConstraint,
)

from src.hqs_pytorch.customML.customLayers import (
    InputPadderPyTorch,
    GradientEstimationLayer,
    ImageWarpingLayer,
    CostVolumeCorrelationLayer,
    HQSIterationLayer,
    GradientDescentLayer,
    ImagePyramidLayer,
    ConvGRU,
)

from src.hqs_pytorch.customML.customLosses import (
    AEPE_Loss,
    OFCE_Loss,
    AngularErrorLoss,
    OpticalFlowPhysicsLoss,
)

from src.hqs_pytorch.customML.customModels import (
    OpticalFlowFeatureEncoder,
    OpticalFlowContextEncoder,
    HQSFlowModel,
)

from src.hqs_pytorch.customML.customSchedulers.OneCycleLR import OneCycleLR

__all__ = [
    # Constraints
    'NonNegativityConstraint',
    'BoundedNonNegativityConstraint',
    'LogBarrierConstraint',
    # Layers
    'InputPadderPyTorch',
    'GradientEstimationLayer',
    'ImageWarpingLayer',
    'CostVolumeCorrelationLayer',
    'HQSIterationLayer',
    'GradientDescentLayer',
    'ImagePyramidLayer',
    'ConvGRU',
    # Loss Functions
    'AEPE_Loss',
    'OFCE_Loss',
    'AngularErrorLoss',
    'OpticalFlowPhysicsLoss',
    # Models
    'OpticalFlowFeatureEncoder',
    'OpticalFlowContextEncoder',
    'HQSFlowModel',
    # Schedulers
    'OneCycleLR',
]
