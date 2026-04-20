"""Flow visualisation using the Middlebury colour wheel."""
from __future__ import annotations

import numpy as np
import torch


def flow_to_color(
    flow: "np.ndarray | torch.Tensor",
    max_flow: float = 0.0,
) -> np.ndarray:
    """
    Encode a 2-channel optical flow as an RGB colour image using the
    Middlebury colour scheme.

    Args:
        flow:     (H, W, 2) numpy array or (2, H, W) torch tensor.
        max_flow: Normalise by this value; 0 = auto (use max magnitude).

    Returns:
        (H, W, 3) uint8 RGB image.
    """
    if isinstance(flow, torch.Tensor):
        flow = flow.detach().cpu()
        if flow.dim() == 3 and flow.shape[0] == 2:
            flow = flow.permute(1, 2, 0).numpy()
        else:
            flow = flow.numpy()

    u = flow[:, :, 0]
    v = flow[:, :, 1]
    valid = np.isfinite(u) & np.isfinite(v)

    mag = np.sqrt(u ** 2 + v ** 2)
    if max_flow <= 0.0:
        max_flow = max(mag[valid].max() if valid.any() else 1.0, 1.0)

    u_norm = u / max_flow
    v_norm = v / max_flow

    return _make_color_wheel_image(u_norm, v_norm)


# ---------------------------------------------------------------------------
# Colour wheel
# ---------------------------------------------------------------------------

def _make_colorwheel() -> np.ndarray:
    RY, YG, GC, CB, BM, MR = 15, 6, 4, 11, 13, 6
    ncols = RY + YG + GC + CB + BM + MR
    cw = np.zeros((ncols, 3), dtype=np.float32)
    col = 0
    # RY
    cw[col:col+RY, 0] = 255
    cw[col:col+RY, 1] = np.floor(255 * np.arange(RY) / RY)
    col += RY
    # YG
    cw[col:col+YG, 0] = 255 - np.floor(255 * np.arange(YG) / YG)
    cw[col:col+YG, 1] = 255
    col += YG
    # GC
    cw[col:col+GC, 1] = 255
    cw[col:col+GC, 2] = np.floor(255 * np.arange(GC) / GC)
    col += GC
    # CB
    cw[col:col+CB, 1] = 255 - np.floor(255 * np.arange(CB) / CB)
    cw[col:col+CB, 2] = 255
    col += CB
    # BM
    cw[col:col+BM, 2] = 255
    cw[col:col+BM, 0] = np.floor(255 * np.arange(BM) / BM)
    col += BM
    # MR
    cw[col:col+MR, 2] = 255 - np.floor(255 * np.arange(MR) / MR)
    cw[col:col+MR, 0] = 255
    return cw


_COLORWHEEL = _make_colorwheel()


def _make_color_wheel_image(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    ncols = _COLORWHEEL.shape[0]
    angle = np.arctan2(-v, -u) / np.pi           # [-1, 1]
    fk    = (angle + 1) / 2 * (ncols - 1)
    k0    = np.floor(fk).astype(int) % ncols
    k1    = (k0 + 1) % ncols
    f     = (fk - np.floor(fk))[..., None]        # fractional part

    col = (1 - f) * _COLORWHEEL[k0] + f * _COLORWHEEL[k1]

    mag = np.sqrt(u ** 2 + v ** 2)
    mag = np.clip(mag, 0, 1)[..., None]

    # Whiten by magnitude
    col = 255 - mag * (255 - col)
    col = np.clip(col, 0, 255).astype(np.uint8)
    return col
