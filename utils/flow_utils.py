"""Flow I/O and helper utilities."""
from __future__ import annotations

import io
import struct
import numpy as np
import torch
import torch.nn.functional as F


def read_flow(path: str) -> np.ndarray:
    """Read .flo or .pfm flow file → (H, W, 2) float32 numpy array."""
    if path.endswith(".flo"):
        return _read_flo(path)
    if path.endswith(".pfm"):
        return _read_pfm(path)
    raise ValueError(f"Unknown flow file format: {path}")


def write_flow(path: str, flow: np.ndarray) -> None:
    """Write .flo file from (H, W, 2) float32 array."""
    H, W, _ = flow.shape
    with open(path, "wb") as f:
        f.write(struct.pack("f", 202021.25))
        f.write(struct.pack("ii", W, H))
        f.write(flow.astype(np.float32).tobytes())


def _read_flo(path: str) -> np.ndarray:
    with open(path, "rb") as f:
        magic = struct.unpack("f", f.read(4))[0]
        assert abs(magic - 202021.25) < 1e-3, f"Bad .flo magic: {magic}"
        W, H = struct.unpack("ii", f.read(8))
        data = np.frombuffer(f.read(H * W * 8), dtype=np.float32)
    return data.reshape(H, W, 2)


def _read_pfm(path: str) -> np.ndarray:
    with open(path, "rb") as f:
        header = f.readline().decode("utf-8").rstrip()
        channels = 3 if header == "PF" else 1
        W, H = map(int, f.readline().decode("utf-8").rstrip().split())
        scale = float(f.readline().decode("utf-8").rstrip())
        data = np.frombuffer(f.read(), dtype=np.float32)
    data = data.reshape(H, W, channels)
    if scale < 0:
        data = data[::-1].copy()
    return data[:, :, :2]


def flow_to_tensor(flow: np.ndarray) -> torch.Tensor:
    """(H, W, 2) → (2, H, W) float32 tensor."""
    return torch.from_numpy(flow.transpose(2, 0, 1).copy()).float()


class InputPadder:
    """Pad BCHW tensors so H and W are divisible by a given stride."""

    def __init__(self, shape, divisor: int = 8) -> None:
        if len(shape) < 4:
            raise ValueError(f"Expected BCHW shape, got {shape}")

        h, w = shape[-2], shape[-1]
        pad_h = (divisor - (h % divisor)) % divisor
        pad_w = (divisor - (w % divisor)) % divisor

        self._top = pad_h // 2
        self._bottom = pad_h - self._top
        self._left = pad_w // 2
        self._right = pad_w - self._left
        self._h = h
        self._w = w

    def pad(self, *inputs: torch.Tensor):
        """Replicate-pad one or more BCHW tensors."""
        padded = [
            F.pad(x, (self._left, self._right, self._top, self._bottom), mode="replicate")
            for x in inputs
        ]
        if len(padded) == 1:
            return padded[0]
        return padded

    def unpad(self, x: torch.Tensor) -> torch.Tensor:
        """Crop tensor back to the original unpadded spatial size."""
        return x[..., self._top:self._top + self._h, self._left:self._left + self._w]
