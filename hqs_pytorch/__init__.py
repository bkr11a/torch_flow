"""Public exports for the hqs_pytorch package.

The primary training model for this repository is now HQSFlowModel from this
package. Exports are intentionally limited to stable, import-safe symbols.
"""

from importlib import import_module

__version__ = "1.1.0"
__author__ = "Brad Rice"

from .customML.customLosses import AEPE_Loss, OFCE_Loss, AngularErrorLoss, OpticalFlowPhysicsLoss
from .customML.customModels import OpticalFlowFeatureEncoder, OpticalFlowContextEncoder

__all__ = [
    "AEPE_Loss",
    "OFCE_Loss",
    "AngularErrorLoss",
    "OpticalFlowPhysicsLoss",
    "HQSFlowModel",
    "HQSLMOpticalFlow",
    "HQSFieldOpticalFlow",
    "HQSOTOpticalFlow",
    "HQSFieldOpticalFlowV2",
    "HQSLMSceneFlow",
    "OpticalFlowFeatureEncoder",
    "OpticalFlowContextEncoder",
]


def __getattr__(name: str):
    """Load the model only when explicitly requested.

    Geometry, reliability and loss modules can then be imported without
    recursively constructing ``models.hqs_flow`` while HQSFlowModel is only
    partially initialized.
    """
    module_by_name = {
        "HQSFlowModel": ".customML.customModels.HQSFlowModel",
        "HQSLMOpticalFlow": ".customML.customModels.HQSLMOpticalFlow",
        "HQSFieldOpticalFlow": (
            ".customML.customModels.HQSFieldOpticalFlow"
        ),
        "HQSOTOpticalFlow": (
            ".customML.customModels.HQSOTOpticalFlow"
        ),
        "HQSFieldOpticalFlowV2": (
            ".customML.customModels.HQSFieldOpticalFlowV2"
        ),
        "HQSLMSceneFlow": ".customML.customModels.HQSLMSceneFlow",
    }
    if name not in module_by_name:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module_by_name[name], __name__), name)
    globals()[name] = value
    return value
