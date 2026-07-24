"""Calibrated projective geometry for HQS-LM-SF.

Scene flow is represented as a metric three-dimensional displacement in the
source-camera coordinate system.  A point is first displaced by ``S`` and is
then mapped into the target camera by ``T_21``:

    X_2_hat = R_21 (X_1 + S) + t_21.

This convention separates object/scene motion from temporal camera motion and
matches the revised scene-flow inverse formulation.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn.functional as F

from .hqs_core_components import spatial_gradients
from .warp import backward_warp, coords_grid, flow_in_bounds_mask


def identity_transform(
    batch: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    return torch.eye(4, device=device, dtype=dtype).unsqueeze(0).expand(
        int(batch), -1, -1
    )


def validate_intrinsics(intrinsics: torch.Tensor, batch: int) -> None:
    if intrinsics.ndim != 3 or intrinsics.shape != (batch, 3, 3):
        raise ValueError(
            f"intrinsics must be [B,3,3], got {tuple(intrinsics.shape)}"
        )
    if not torch.isfinite(intrinsics).all():
        raise ValueError("intrinsics contain non-finite values")


def validate_transform(transform_21: torch.Tensor, batch: int) -> None:
    if transform_21.ndim != 3 or transform_21.shape != (batch, 4, 4):
        raise ValueError(
            f"transform_21 must be [B,4,4], got {tuple(transform_21.shape)}"
        )
    if not torch.isfinite(transform_21).all():
        raise ValueError("transform_21 contains non-finite values")


def scale_intrinsics(
    intrinsics: torch.Tensor,
    *,
    source_size: Tuple[int, int],
    target_size: Tuple[int, int],
) -> torch.Tensor:
    """Scale camera intrinsics from one image grid to another."""
    source_height, source_width = map(int, source_size)
    target_height, target_width = map(int, target_size)
    if min(source_height, source_width, target_height, target_width) <= 0:
        raise ValueError("Image sizes must be positive")
    scale_x = float(target_width) / float(source_width)
    scale_y = float(target_height) / float(source_height)
    result = intrinsics.clone()
    result[:, 0, :] = result[:, 0, :] * scale_x
    result[:, 1, :] = result[:, 1, :] * scale_y
    result[:, 2, :] = intrinsics[:, 2, :]
    return result


def backproject_depth(
    depth: torch.Tensor,
    intrinsics: torch.Tensor,
    pixels_xy: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Back-project depth into a camera-frame point map."""
    if depth.ndim != 4 or depth.shape[1] != 1:
        raise ValueError(f"depth must be [B,1,H,W], got {tuple(depth.shape)}")
    batch, _, height, width = depth.shape
    validate_intrinsics(intrinsics, batch)
    if pixels_xy is None:
        pixels_xy = coords_grid(batch, height, width, depth.device).to(
            dtype=depth.dtype
        )
    if pixels_xy.shape != (batch, 2, height, width):
        raise ValueError(
            "pixels_xy must align with depth: expected "
            f"{(batch, 2, height, width)}, got {tuple(pixels_xy.shape)}"
        )

    homogeneous = torch.cat(
        (pixels_xy, torch.ones_like(depth)), dim=1
    ).flatten(2)
    inverse = torch.linalg.inv(intrinsics.float())
    rays = torch.bmm(inverse, homogeneous.float()).reshape(
        batch, 3, height, width
    )
    return (rays * depth.float()).to(dtype=depth.dtype)


def transform_points(
    points: torch.Tensor,
    transform_21: torch.Tensor,
) -> torch.Tensor:
    """Map a source-camera point map into the target camera."""
    if points.ndim != 4 or points.shape[1] != 3:
        raise ValueError(
            f"points must be [B,3,H,W], got {tuple(points.shape)}"
        )
    batch = points.shape[0]
    validate_transform(transform_21, batch)
    rotation = transform_21[:, :3, :3].to(dtype=points.dtype)
    translation = transform_21[:, :3, 3].to(dtype=points.dtype)
    transformed = torch.einsum(
        "bij,bjhw->bihw", rotation, points
    )
    return transformed + translation[:, :, None, None]


def project_points(
    points: torch.Tensor,
    intrinsics: torch.Tensor,
    *,
    epsilon: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Project a target-camera point map to pixels and return positive-depth validity."""
    if points.ndim != 4 or points.shape[1] != 3:
        raise ValueError(
            f"points must be [B,3,H,W], got {tuple(points.shape)}"
        )
    batch = points.shape[0]
    validate_intrinsics(intrinsics, batch)
    projected = torch.einsum(
        "bij,bjhw->bihw", intrinsics.to(dtype=points.dtype), points
    )
    depth = points[:, 2:3]
    safe = projected[:, 2:3].clamp_min(float(epsilon))
    pixels = projected[:, :2] / safe
    valid = (depth > float(epsilon)).to(dtype=points.dtype)
    return pixels, valid


@dataclass
class SceneProjection:
    source_points: torch.Tensor
    target_points: torch.Tensor
    target_pixels: torch.Tensor
    induced_flow: torch.Tensor
    valid: torch.Tensor


def project_scene_flow(
    scene_flow: torch.Tensor,
    depth1: torch.Tensor,
    intrinsics: torch.Tensor,
    transform_21: torch.Tensor,
) -> SceneProjection:
    """Apply scene displacement and project it into the target image."""
    if scene_flow.ndim != 4 or scene_flow.shape[1] != 3:
        raise ValueError(
            f"scene_flow must be [B,3,H,W], got {tuple(scene_flow.shape)}"
        )
    if depth1.shape[0] != scene_flow.shape[0] or depth1.shape[-2:] != (
        scene_flow.shape[-2:]
    ):
        raise ValueError("depth1 and scene_flow grids must align")
    batch, _, height, width = scene_flow.shape
    source_points = backproject_depth(depth1, intrinsics)
    target_points = transform_points(
        source_points + scene_flow, transform_21
    )
    target_pixels, positive_depth = project_points(
        target_points, intrinsics
    )
    source_pixels = coords_grid(
        batch, height, width, scene_flow.device
    ).to(dtype=scene_flow.dtype)
    induced_flow = target_pixels - source_pixels
    valid = (
        positive_depth
        * (depth1 > 0.0).to(dtype=scene_flow.dtype)
        * flow_in_bounds_mask(induced_flow)
    )
    return SceneProjection(
        source_points=source_points,
        target_points=target_points,
        target_pixels=target_pixels,
        induced_flow=induced_flow,
        valid=valid,
    )


def projective_scene_jacobian(
    target_points: torch.Tensor,
    intrinsics: torch.Tensor,
    transform_21: torch.Tensor,
    *,
    epsilon: float = 1e-6,
) -> torch.Tensor:
    """Return ``d(u,v)/dS`` with shape ``[B,2,3,H,W]``."""
    if target_points.ndim != 4 or target_points.shape[1] != 3:
        raise ValueError("target_points must be [B,3,H,W]")
    batch = target_points.shape[0]
    validate_intrinsics(intrinsics, batch)
    validate_transform(transform_21, batch)

    x = target_points[:, 0:1].float()
    y = target_points[:, 1:2].float()
    z = target_points[:, 2:3].float().clamp_min(float(epsilon))
    fx = intrinsics[:, 0, 0].float().view(batch, 1, 1, 1)
    fy = intrinsics[:, 1, 1].float().view(batch, 1, 1, 1)
    rotation = transform_21[:, :3, :3].float()

    du_dx = fx / z
    du_dz = -fx * x / z.square()
    dv_dy = fy / z
    dv_dz = -fy * y / z.square()

    rows = []
    for component in range(3):
        du = (
            du_dx * rotation[:, 0, component].view(batch, 1, 1, 1)
            + du_dz
            * rotation[:, 2, component].view(batch, 1, 1, 1)
        )
        dv = (
            dv_dy * rotation[:, 1, component].view(batch, 1, 1, 1)
            + dv_dz
            * rotation[:, 2, component].view(batch, 1, 1, 1)
        )
        rows.append(torch.cat((du, dv), dim=1))
    # rows[j] is [B,2,H,W]; stack the scene-flow derivative dimension.
    return torch.stack(rows, dim=2).to(dtype=target_points.dtype)


def feature_scene_jacobian(
    warped_target_grad_x: torch.Tensor,
    warped_target_grad_y: torch.Tensor,
    projective_jacobian: torch.Tensor,
) -> torch.Tensor:
    """Chain feature spatial derivatives with the projection Jacobian."""
    if warped_target_grad_x.shape != warped_target_grad_y.shape:
        raise ValueError("Target feature gradient shapes differ")
    if projective_jacobian.ndim != 5 or projective_jacobian.shape[1:3] != (
        2,
        3,
    ):
        raise ValueError(
            "projective_jacobian must be [B,2,3,H,W], got "
            f"{tuple(projective_jacobian.shape)}"
        )
    return (
        warped_target_grad_x.unsqueeze(2)
        * projective_jacobian[:, 0:1]
        + warped_target_grad_y.unsqueeze(2)
        * projective_jacobian[:, 1:2]
    )


@dataclass
class DepthLinearisation:
    residual: torch.Tensor
    jacobian: torch.Tensor
    valid: torch.Tensor
    warped_depth: torch.Tensor


def linearise_target_depth(
    depth2: torch.Tensor,
    projection: SceneProjection,
    projective_jacobian: torch.Tensor,
    transform_21: torch.Tensor,
) -> DepthLinearisation:
    """Linearise the target-depth consistency residual with respect to scene flow."""
    if depth2.ndim != 4 or depth2.shape[1] != 1:
        raise ValueError(f"depth2 must be [B,1,H,W], got {tuple(depth2.shape)}")
    grad_x, grad_y = spatial_gradients(depth2)
    warped_depth = backward_warp(
        depth2, projection.induced_flow, padding_mode="border"
    )
    warped_grad_x = backward_warp(
        grad_x, projection.induced_flow, padding_mode="border"
    )
    warped_grad_y = backward_warp(
        grad_y, projection.induced_flow, padding_mode="border"
    )
    warped_valid = backward_warp(
        (depth2 > 0.0).to(dtype=depth2.dtype),
        projection.induced_flow,
        padding_mode="zeros",
    )
    valid = (
        projection.valid
        * (warped_valid > 0.999).to(dtype=depth2.dtype)
    )

    batch = depth2.shape[0]
    rotation_z = transform_21[:, 2, :3].to(dtype=depth2.dtype).view(
        batch, 1, 3, 1, 1
    )
    jacobian = (
        warped_grad_x.unsqueeze(2) * projective_jacobian[:, 0:1]
        + warped_grad_y.unsqueeze(2) * projective_jacobian[:, 1:2]
        - rotation_z
    )
    residual = warped_depth - projection.target_points[:, 2:3]
    return DepthLinearisation(
        residual=residual,
        jacobian=jacobian,
        valid=valid,
        warped_depth=warped_depth,
    )


@dataclass
class LiftedSceneFlow:
    scene_flow: torch.Tensor
    valid: torch.Tensor
    target_depth: torch.Tensor


def lift_flow_to_scene_flow(
    flow_xy: torch.Tensor,
    depth1: torch.Tensor,
    depth2: torch.Tensor,
    intrinsics: torch.Tensor,
    transform_21: torch.Tensor,
) -> LiftedSceneFlow:
    """Lift a 2D correspondence and two depths to metric scene flow."""
    if flow_xy.ndim != 4 or flow_xy.shape[1] != 2:
        raise ValueError("flow_xy must be [B,2,H,W]")
    batch, _, height, width = flow_xy.shape
    source_pixels = coords_grid(
        batch, height, width, flow_xy.device
    ).to(dtype=flow_xy.dtype)
    target_pixels = source_pixels + flow_xy
    target_depth = backward_warp(
        depth2, flow_xy, padding_mode="border"
    )
    target_depth_valid = backward_warp(
        (depth2 > 0.0).to(dtype=depth2.dtype),
        flow_xy,
        padding_mode="zeros",
    )
    source_points = backproject_depth(depth1, intrinsics)
    target_points = backproject_depth(
        target_depth, intrinsics, pixels_xy=target_pixels
    )

    rotation = transform_21[:, :3, :3].to(dtype=flow_xy.dtype)
    translation = transform_21[:, :3, 3].to(dtype=flow_xy.dtype)
    target_minus_translation = (
        target_points - translation[:, :, None, None]
    )
    pre_transform = torch.einsum(
        "bji,bjhw->bihw", rotation, target_minus_translation
    )
    scene_flow = pre_transform - source_points
    valid = (
        (depth1 > 0.0).to(dtype=flow_xy.dtype)
        * (target_depth_valid > 0.999).to(dtype=flow_xy.dtype)
        * flow_in_bounds_mask(flow_xy)
    )
    return LiftedSceneFlow(
        scene_flow=scene_flow * valid,
        valid=valid,
        target_depth=target_depth,
    )


def resize_metric_field(
    field: torch.Tensor,
    size: Tuple[int, int],
) -> torch.Tensor:
    """Resize a metric vector field without changing component magnitudes."""
    if field.ndim != 4:
        raise ValueError(f"field must be [B,C,H,W], got {tuple(field.shape)}")
    if field.shape[-2:] == tuple(size):
        return field
    return F.interpolate(
        field, size=size, mode="bilinear", align_corners=True
    )


__all__ = [
    "DepthLinearisation",
    "LiftedSceneFlow",
    "SceneProjection",
    "backproject_depth",
    "feature_scene_jacobian",
    "identity_transform",
    "lift_flow_to_scene_flow",
    "linearise_target_depth",
    "project_points",
    "project_scene_flow",
    "projective_scene_jacobian",
    "resize_metric_field",
    "scale_intrinsics",
    "transform_points",
    "validate_intrinsics",
    "validate_transform",
]
