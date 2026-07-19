"""
Custom models for PyTorch HQS optical flow.
"""

from importlib import import_module

from .OpticalFlowFeatureEncoder import OpticalFlowFeatureEncoder
from .OpticalFlowContextEncoder import OpticalFlowContextEncoder

__all__ = [
    'OpticalFlowFeatureEncoder',
    'OpticalFlowContextEncoder',
    'HQSFlowModel',
    'HQSFlowModelTFPort',
    'HQSCore',
]


def __getattr__(name: str):
    if name not in {"HQSFlowModel", "HQSFlowModelTFPort", "HQSCore"}:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(f".{name}", __name__)
    value = getattr(module, name)
    globals()[name] = value
    return value
