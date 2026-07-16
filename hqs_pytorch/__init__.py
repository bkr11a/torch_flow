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
    "OpticalFlowFeatureEncoder",
    "OpticalFlowContextEncoder",
]


def __getattr__(name: str):
    """Load the model only when explicitly requested.

    Geometry, reliability and loss modules can then be imported without
    recursively constructing ``models.hqs_flow`` while HQSFlowModel is only
    partially initialized.
    """
    if name != "HQSFlowModel":
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = import_module(
        ".customML.customModels.HQSFlowModel", __name__
    ).HQSFlowModel
    globals()[name] = value
    return value
