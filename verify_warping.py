#!/usr/bin/env python3
"""Verify _warp_yx correctness with a controlled synthetic translation example.

Scenario:
- Image size: 10x10, grayscale replicated to 3 channels.
- Image 1 contains a 3x3 square (value 255) with top-left at (x=2, y=1).
- Image 2 moves this square right by 1 and down by 2.
- Forward flow (image1 -> image2) is [dy=2, dx=1] on square pixels.

We then warp image2 back onto image1 using HQSFlowModelTFPort._warp_yx and
compare against both:
1) a manually sampled expected warp (same sampling rule), and
2) the original image1 in relevant regions.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
from typing import Tuple

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import torch


def _load_hqs_tf_port_class():
    """Load HQSFlowModelTFPort without triggering package-level circular imports."""
    this_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(
        this_dir,
        "hqs_pytorch",
        "customML",
        "customModels",
        "HQSFlowModelTFPort.py",
    )
    spec = importlib.util.spec_from_file_location("HQSFlowModelTFPort_module", model_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module spec from {model_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.HQSFlowModelTFPort


HQSFlowModelTFPort = _load_hqs_tf_port_class()


def build_synthetic_pair(
    h: int = 10,
    w: int = 10,
    square_size: int = 3,
    square_top_left_xy: Tuple[int, int] = (3, 2),
    shift_dx: int = 1,
    shift_dy: int = 2,
    square_value: float = 255.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create synthetic image pair and forward flow in [dy, dx].

    Returns:
        image1: (1, 1, H, W)
        image2: (1, 1, H, W)
        flow_yx: (1, 2, H, W)
        square_mask: (1, 1, H, W) where image1 square exists
    """
    x0, y0 = square_top_left_xy

    image1 = torch.zeros((1, 1, h, w), dtype=torch.float32)
    image2 = torch.zeros((1, 1, h, w), dtype=torch.float32)
    mask = torch.zeros((1, 1, h, w), dtype=torch.float32)

    image1[:, :, y0 : y0 + square_size, x0 : x0 + square_size] = square_value
    mask[:, :, y0 : y0 + square_size, x0 : x0 + square_size] = 1.0

    x1 = x0 + shift_dx
    y1 = y0 + shift_dy
    image2[:, :, y1 : y1 + square_size, x1 : x1 + square_size] = square_value

    flow_yx = torch.zeros((1, 2, h, w), dtype=torch.float32)
    flow_yx[:, 0] = shift_dy * mask[:, 0]
    flow_yx[:, 1] = shift_dx * mask[:, 0]

    return image1, image2, flow_yx, mask


def manual_expected_warp(image2: torch.Tensor, flow_yx: torch.Tensor) -> torch.Tensor:
    """Manual reference for integer-flow sampling: out(y,x)=image2(y+dy, x+dx)."""
    b, c, h, w = image2.shape
    expected = torch.zeros_like(image2)

    for bi in range(b):
        for yi in range(h):
            for xi in range(w):
                dy = int(flow_yx[bi, 0, yi, xi].item())
                dx = int(flow_yx[bi, 1, yi, xi].item())
                sy = yi + dy
                sx = xi + dx
                if 0 <= sy < h and 0 <= sx < w:
                    expected[bi, :, yi, xi] = image2[bi, :, sy, sx]
    return expected


def compute_metrics(
    flow_yx: torch.Tensor,
    image1: torch.Tensor,
    warped: torch.Tensor,
    expected: torch.Tensor,
    square_mask: torch.Tensor,
) -> dict:
    """Compute quantitative checks for warp correctness."""
    abs_err_expected = (warped - expected).abs()
    abs_err_img1 = (warped - image1).abs()
    photometric_residual = image1 - warped
    abs_photometric_residual = photometric_residual.abs()

    valid_warp_mask = HQSFlowModelTFPort._valid_warp_mask(flow_yx).bool()
    valid_mask = square_mask.bool() & valid_warp_mask.bool()
    masked_photometric_residual = abs_photometric_residual[valid_mask]

    metrics = {
        "mae_vs_manual_expected": abs_err_expected.mean().item(),
        "maxae_vs_manual_expected": abs_err_expected.max().item(),
        "rmse_vs_manual_expected": torch.sqrt(((warped - expected) ** 2).mean()).item(),
        "mae_vs_image1_full": abs_err_img1.mean().item(),
        "mae_photometric_residual_full": abs_photometric_residual.mean().item(),
        "maxae_photometric_residual_full": abs_photometric_residual.max().item(),
        "mae_photometric_residual_masked": masked_photometric_residual.mean().item(),
        "maxae_photometric_residual_masked": masked_photometric_residual.max().item(),
        "valid_pixel_count": int(valid_mask.sum().item()),
    }

    square_pixels = square_mask.bool()
    if square_pixels.any():
        square_err = abs_err_img1[square_pixels]
        metrics["mae_vs_image1_on_square"] = square_err.mean().item()
        metrics["maxae_vs_image1_on_square"] = square_err.max().item()
        square_photo = abs_photometric_residual[square_pixels]
        metrics["mae_photometric_residual_on_square"] = square_photo.mean().item()
        metrics["maxae_photometric_residual_on_square"] = square_photo.max().item()

    return metrics


def flow_to_vis_components(flow_yx: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
    """Return dy/dx arrays for plotting (H, W)."""
    dy = flow_yx[0, 0].detach().cpu().numpy()
    dx = flow_yx[0, 1].detach().cpu().numpy()
    return dy, dx


def flow_yx_to_hsv_rgb(flow_yx: torch.Tensor) -> np.ndarray:
    """Convert [dy, dx] flow tensor to HSV-based RGB visualization."""
    dy, dx = flow_to_vis_components(flow_yx)

    mag = np.sqrt(dx ** 2 + dy ** 2)
    angle = np.arctan2(dy, dx)  # [-pi, pi]

    hue = (angle + np.pi) / (2.0 * np.pi)  # [0, 1]
    max_mag = max(float(np.percentile(mag, 99)), 1e-6)
    sat = np.clip(mag / max_mag, 0.0, 1.0)
    val = np.ones_like(sat)

    hsv = np.stack([hue, sat, val], axis=-1).astype(np.float32)
    rgb = mcolors.hsv_to_rgb(hsv)
    return (rgb * 255).astype(np.uint8)


def visualize_all(
    image1: torch.Tensor,
    image2: torch.Tensor,
    flow_yx: torch.Tensor,
    square_mask: torch.Tensor,
    warped: torch.Tensor,
    expected: torch.Tensor,
    out_path: str,
) -> None:
    """Create a 2x7 verification figure."""
    i1 = image1[0, 0].detach().cpu().numpy()
    i2 = image2[0, 0].detach().cpu().numpy()
    w2 = warped[0, 0].detach().cpu().numpy()
    exp = expected[0, 0].detach().cpu().numpy()
    err = np.abs(w2 - exp)
    photo_residual = i1 - w2
    abs_photo_residual = np.abs(photo_residual)

    valid_warp_mask = HQSFlowModelTFPort._valid_warp_mask(flow_yx).bool()
    masked_valid = (square_mask.bool() & valid_warp_mask)[0, 0].cpu().numpy()
    masked_residual = np.where(masked_valid, photo_residual, np.nan)
    abs_masked_residual = np.where(masked_valid, abs_photo_residual, np.nan)

    dy, dx = flow_to_vis_components(flow_yx)
    flow_hsv = flow_yx_to_hsv_rgb(flow_yx)

    yy, xx = np.mgrid[0 : i1.shape[0], 0 : i1.shape[1]]

    fig, axes = plt.subplots(2, 7, figsize=(21, 8))

    axes[0, 0].imshow(i1, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
    axes[0, 0].set_title("Image 1 (reference)")

    axes[0, 1].imshow(i2, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
    axes[0, 1].set_title("Image 2 (shifted)")

    axes[0, 2].imshow(i1, cmap="gray", vmin=0, vmax=255, interpolation="nearest", alpha=0.3)
    axes[0, 2].quiver(xx, yy, dx, dy, color="tab:red", angles="xy", scale_units="xy", scale=1)
    axes[0, 2].set_title("Forward flow (dx,dy)")

    axes[0, 3].imshow(exp, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
    axes[0, 3].set_title("Manual expected warp")

    axes[0, 4].imshow(flow_hsv, interpolation="nearest")
    axes[0, 4].set_title("Flow HSV")

    axes[0, 5].imshow(abs_photo_residual, cmap="magma", interpolation="nearest")
    axes[0, 5].set_title("|Photometric residual|")
    plt.colorbar(axes[0, 5].images[0], ax=axes[0, 5], fraction=0.046, pad=0.04)

    masked_limit = max(float(np.nanmax(np.abs(masked_residual))), 1.0)
    im_masked_abs = axes[0, 6].imshow(abs_masked_residual, cmap="magma", interpolation="nearest")
    axes[0, 6].set_title("|Masked photometric residual|")
    plt.colorbar(im_masked_abs, ax=axes[0, 6], fraction=0.046, pad=0.04)

    im_dy = axes[1, 0].imshow(dy, cmap="coolwarm", interpolation="nearest")
    axes[1, 0].set_title("Flow dy")
    plt.colorbar(im_dy, ax=axes[1, 0], fraction=0.046, pad=0.04)

    im_dx = axes[1, 1].imshow(dx, cmap="coolwarm", interpolation="nearest")
    axes[1, 1].set_title("Flow dx")
    plt.colorbar(im_dx, ax=axes[1, 1], fraction=0.046, pad=0.04)

    axes[1, 2].imshow(w2, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
    axes[1, 2].contour(err, levels=[1e-6], colors="yellow", linewidths=1)
    axes[1, 2].set_title("Warped image2 -> image1")

    axes[1, 3].imshow(err, cmap="hot", interpolation="nearest")
    axes[1, 3].set_title("Absolute error vs manual expected")
    plt.colorbar(axes[1, 3].images[0], ax=axes[1, 3], fraction=0.046, pad=0.04)

    residual_limit = max(float(np.max(np.abs(photo_residual))), 1.0)
    im_residual = axes[1, 4].imshow(
        photo_residual,
        cmap="coolwarm",
        vmin=-residual_limit,
        vmax=residual_limit,
        interpolation="nearest",
    )
    axes[1, 4].set_title("Photometric residual (I1 - warp(I2))")
    plt.colorbar(im_residual, ax=axes[1, 4], fraction=0.046, pad=0.04)

    im_masked_residual = axes[1, 5].imshow(
        masked_residual,
        cmap="coolwarm",
        vmin=-masked_limit,
        vmax=masked_limit,
        interpolation="nearest",
    )
    axes[1, 5].set_title("Masked photometric residual")
    plt.colorbar(im_masked_residual, ax=axes[1, 5], fraction=0.046, pad=0.04)

    # Keep the bottom-right slot for a compact HSV legend-like note.
    axes[1, 6].axis("off")
    axes[1, 6].text(
        0.05,
        0.85,
        "HSV flow key:\nHue = direction\nSaturation = magnitude\nValue = constant",
        fontsize=10,
        va="top",
    )

    for ax in axes.ravel():
        ax.set_xticks(range(i1.shape[1]))
        ax.set_yticks(range(i1.shape[0]))
        ax.set_xlim(-0.5, i1.shape[1] - 0.5)
        ax.set_ylim(i1.shape[0] - 0.5, -0.5)
        ax.grid(color="white", linestyle="-", linewidth=0.3, alpha=0.4)

    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify _warp_yx with synthetic translation data")
    parser.add_argument("--out_dir", type=str, default="verify_warping_out", help="Directory for outputs")
    # Add starting point and shift parameters if desired for more flexible testing
    parser.add_argument("--square_top_left_x", type=int, default=3, help="Square top-left x coordinate in image1")
    parser.add_argument("--square_top_left_y", type=int, default=2, help="Square top-left y coordinate in image1")
    parser.add_argument("--shift_dx", type=int, default=1, help="Horizontal shift (dx) applied to square in image2")
    parser.add_argument("--shift_dy", type=int, default=2, help="Vertical shift (dy) applied to square in image2")
    parser.add_argument("--square_value", type=float, default=255.0, help="Pixel value for the square region")
    parser.add_argument("--square_size", type=int, default=3, help="Size of the square region (square_size x square_size)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    image1, image2, flow_yx, square_mask = build_synthetic_pair(
        square_top_left_x=args.square_top_left_x,
        square_top_left_y=args.square_top_left_y,
        shift_dx=args.shift_dx,
        shift_dy=args.shift_dy,
        square_value=args.square_value,
        square_size=args.square_size,
    )

    with torch.no_grad():
        warped = HQSFlowModelTFPort._warp_yx(image2, flow_yx)

    expected = manual_expected_warp(image2, flow_yx)
    metrics = compute_metrics(flow_yx=flow_yx, image1=image1, warped=warped, expected=expected, square_mask=square_mask)
    photometric_residual = image1 - warped

    fig_path = os.path.join(args.out_dir, "warping_verification.png")
    visualize_all(
        image1=image1,
        image2=image2,
        flow_yx=flow_yx,
        square_mask=square_mask,
        warped=warped,
        expected=expected,
        out_path=fig_path,
    )

    np.save(os.path.join(args.out_dir, "image1.npy"), image1[0, 0].numpy())
    np.save(os.path.join(args.out_dir, "image2.npy"), image2[0, 0].numpy())
    np.save(os.path.join(args.out_dir, "flow_dy_dx.npy"), flow_yx[0].numpy())
    np.save(os.path.join(args.out_dir, "warped.npy"), warped[0, 0].numpy())
    np.save(os.path.join(args.out_dir, "expected.npy"), expected[0, 0].numpy())
    np.save(os.path.join(args.out_dir, "photometric_residual.npy"), photometric_residual[0, 0].numpy())


    print("=== verify_warping results ===")
    print("Synthetic setup:")
    print("  image size        : 10x10")
    print(f"  square top-left   : (x={args.square_top_left_x}, y={args.square_top_left_y})")
    print(f"  square size       : {args.square_size}x{args.square_size}")
    print(f"  square value      : {args.square_value}")
    print(f"  shift (dx, dy)    : ({args.shift_dx}, {args.shift_dy})")
    print(f"  forward flow (yx) : ({args.shift_dy}, {args.shift_dx}) on image1 square")
    print()
    print("Metrics:")
    for k, v in metrics.items():
        print(f"  {k:28s}: {v:.6f}")
    print()
    print(f"Saved figure: {fig_path}")


if __name__ == "__main__":
    main()
