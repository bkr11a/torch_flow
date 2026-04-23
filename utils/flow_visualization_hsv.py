"""HSV-based optical flow visualization.

Converts optical flow to HSV color space for intuitive visualization:
- Hue: flow direction (angle)
- Saturation: flow magnitude (speed)
- Value: constant (brightness)
"""
from __future__ import annotations

import torch
import numpy as np
import cv2
from typing import Optional, Tuple


def flow_to_hsv(
    flow: torch.Tensor | np.ndarray,
    max_magnitude: Optional[float] = None,
    unknown_threshold: float = 1e6,
) -> np.ndarray:
    """
    Convert optical flow to HSV visualization.

    Args:
        flow: (H, W, 2) flow array [dx, dy] or torch tensor (2, H, W).
        max_magnitude: Maximum flow magnitude for saturation scaling.
                       If None, computed from data.
        unknown_threshold: Treat flow magnitudes > this as invalid (will be gray).

    Returns:
        (H, W, 3) uint8 RGB image.
    """
    # Convert tensor to numpy if needed
    if isinstance(flow, torch.Tensor):
        if flow.dim() == 3 and flow.shape[0] == 2:
            # (2, H, W) -> (H, W, 2)
            flow = flow.permute(1, 2, 0).detach().cpu().numpy()
        else:
            flow = flow.detach().cpu().numpy()

    flow = flow.astype(np.float32)
    H, W = flow.shape[:2]

    # Extract components
    dx = flow[..., 0]
    dy = flow[..., 1]

    # Compute magnitude and angle
    mag = np.sqrt(dx**2 + dy**2)
    angle = np.arctan2(dy, dx)

    # Create HSV image
    hsv = np.zeros((H, W, 3), dtype=np.uint8)

    # Hue: direction (0-180 in OpenCV)
    # Map angle from [-pi, pi] to [0, 180]
    hue = ((angle + np.pi) / (2 * np.pi) * 180).astype(np.uint8)
    hsv[..., 0] = hue

    # Saturation: magnitude (0-255)
    # Normalize by max magnitude
    if max_magnitude is None:
        # Use 95th percentile to avoid outliers
        valid_mag = mag[mag < unknown_threshold]
        if len(valid_mag) > 0:
            max_magnitude = np.percentile(valid_mag, 95)
        else:
            max_magnitude = 1.0

    saturation = (mag / max_magnitude * 255).clip(0, 255).astype(np.uint8)
    hsv[..., 1] = saturation

    # Value: constant brightness
    hsv[..., 2] = 255

    # Mark unknown/invalid flow as gray
    unknown_mask = mag > unknown_threshold
    hsv[unknown_mask] = [0, 0, 128]  # Gray in HSV

    # Convert HSV to RGB
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    return rgb


def flow_to_hsv_batch(
    flow_batch: torch.Tensor | np.ndarray,
    max_magnitude: Optional[float] = None,
) -> list[np.ndarray]:
    """
    Convert batch of optical flows to HSV visualizations.

    Args:
        flow_batch: (B, 2, H, W) tensor or (B, H, W, 2) array.
        max_magnitude: Optional global maximum for consistent scaling.

    Returns:
        List of (H, W, 3) uint8 RGB images.
    """
    if isinstance(flow_batch, torch.Tensor):
        if flow_batch.dim() == 4 and flow_batch.shape[1] == 2:
            # (B, 2, H, W) -> list of (H, W, 2)
            flow_batch = flow_batch.permute(0, 2, 3, 1).detach().cpu().numpy()

    results = []
    for b in range(flow_batch.shape[0]):
        rgb = flow_to_hsv(flow_batch[b], max_magnitude=max_magnitude)
        results.append(rgb)
    return results


def create_flow_colorwheel() -> np.ndarray:
    """
    Create a reference colorwheel showing all flow directions and magnitudes.

    Returns:
        (H, W, 3) uint8 image with colorwheel.
    """
    size = 400
    center = (size // 2, size // 2)
    radius = size // 2 - 20

    # Create circular colorwheel
    hsv = np.zeros((size, size, 3), dtype=np.uint8)

    for y in range(size):
        for x in range(size):
            dy = y - center[1]
            dx = x - center[0]
            dist = np.sqrt(dx**2 + dy**2)

            if dist <= radius:
                angle = np.arctan2(dy, dx)
                # Hue from angle
                hue = ((angle + np.pi) / (2 * np.pi) * 180).astype(np.uint8)
                # Saturation from distance
                sat = int((dist / radius) * 255)

                hsv[y, x, 0] = hue
                hsv[y, x, 1] = sat
                hsv[y, x, 2] = 255

    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

    # Add text annotations
    cv2.circle(rgb, center, radius, (0, 0, 0), 2)
    cv2.putText(rgb, "0px", (center[0] + 10, center[1] + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    cv2.putText(rgb, f"{radius}px", (center[0] + 50, center[1]),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    return rgb


if __name__ == "__main__":
    # Example usage
    import matplotlib.pyplot as plt

    # Generate synthetic flow
    H, W = 256, 256
    x, y = np.meshgrid(np.linspace(-1, 1, W), np.linspace(-1, 1, H))
    dx = np.sin(x * np.pi) * 10
    dy = np.cos(y * np.pi) * 10
    flow = np.stack([dx, dy], axis=-1)

    # Visualize
    rgb = flow_to_hsv(flow)
    wheel = create_flow_colorwheel()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].imshow(rgb)
    axes[0].set_title("Flow Visualization (HSV)")
    axes[0].axis("off")

    axes[1].imshow(wheel)
    axes[1].set_title("Flow Colorwheel")
    axes[1].axis("off")

    plt.tight_layout()
    plt.savefig("flow_hsv_visualization.png", dpi=150, bbox_inches="tight")
    print("Saved flow_hsv_visualization.png")
