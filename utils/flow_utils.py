"""Flow I/O and helper utilities."""
from __future__ import annotations

import io
import struct
import numpy as np
import torch


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
