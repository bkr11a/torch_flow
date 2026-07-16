"""models/__init__.py – public API for the models package."""
from importlib import import_module

from .encoders import BasicEncoder, SmallEncoder, build_encoder
from .correlation import CorrBlock, LocalCorrBlock, build_corr_block
from .update_net import DataUpdateNet, ConvGRUCell
from .reg_net import UNetProxNet, DnCNNProxNet, IdentityProxNet, build_prox_net
from .hqs_stage import HQSStage
from .warp import backward_warp, upsample_flow, coords_grid

__all__ = [
    "HQSFlow", "build_model",
    "BasicEncoder", "SmallEncoder", "build_encoder",
    "CorrBlock", "LocalCorrBlock", "build_corr_block",
    "DataUpdateNet", "ConvGRUCell",
    "UNetProxNet", "DnCNNProxNet", "IdentityProxNet", "build_prox_net",
    "HQSStage",
    "backward_warp", "upsample_flow", "coords_grid",
]


def __getattr__(name: str):
    # HQSFlowModel imports leaf modules such as models.encoders.  Defer the
    # compatibility shim so those imports cannot re-enter HQSFlowModel while it
    # is still being defined.
    if name not in {"HQSFlow", "build_model"}:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(".hqs_flow", __name__)
    value = getattr(module, name)
    globals()[name] = value
    return value
