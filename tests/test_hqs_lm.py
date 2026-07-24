from __future__ import annotations

import inspect

import torch
from omegaconf import OmegaConf

from models import build_model
from models.hqs_lm_components import (
    SourceOnlyMotionProximal,
    local_correlation_measurement,
    solve_optical_lm_increment,
)
from models.hqs_lm_scene_components import solve_scene_lm_increment
from models.scene_flow_geometry import (
    backproject_depth,
    identity_transform,
    lift_flow_to_scene_flow,
    project_points,
    project_scene_flow,
    projective_scene_jacobian,
)


def _scalar_map(value: float, height: int = 1, width: int = 1):
    return torch.full((1, 1, height, width), float(value))


def test_optical_lm_solves_rank_two_feature_system():
    residual = torch.tensor([[[[1.0]], [[2.0]]]])
    jacobian_x = torch.tensor([[[[1.0]], [[0.0]]]])
    jacobian_y = torch.tensor([[[[0.0]], [[1.0]]]])
    flow = torch.zeros(1, 2, 1, 1)
    result = solve_optical_lm_increment(
        residual=residual,
        jacobian_x=jacobian_x,
        jacobian_y=jacobian_y,
        appearance_validity=_scalar_map(1.0),
        flow_w=flow,
        flow_z=flow,
        match_proposal=flow,
        match_precision=_scalar_map(0.0),
        beta_map=_scalar_map(0.0),
        damping_map=_scalar_map(0.0),
        charbonnier_alpha=1.0,
    )
    assert torch.allclose(
        result.delta.flatten(), torch.tensor([-1.0, -2.0]), atol=1e-5
    )


def test_optical_lm_rank_one_feature_system_is_regularised_by_hqs():
    flow = torch.zeros(1, 2, 1, 1)
    result = solve_optical_lm_increment(
        residual=torch.ones(1, 1, 1, 1),
        jacobian_x=torch.ones(1, 1, 1, 1),
        jacobian_y=torch.zeros(1, 1, 1, 1),
        appearance_validity=_scalar_map(1.0),
        flow_w=flow,
        flow_z=flow,
        match_proposal=flow,
        match_precision=_scalar_map(0.0),
        beta_map=_scalar_map(1.0),
        damping_map=_scalar_map(0.0),
        charbonnier_alpha=1.0,
    )
    assert torch.allclose(result.delta[:, 0], _scalar_map(-0.5)[:, 0])
    assert torch.equal(
        result.delta[:, 1], torch.zeros_like(result.delta[:, 1])
    )


def test_optical_lm_fuses_match_as_precision_weighted_observation():
    flow = torch.zeros(1, 2, 1, 1)
    proposal = torch.tensor([[[[2.0]], [[-2.0]]]])
    result = solve_optical_lm_increment(
        residual=torch.zeros(1, 1, 1, 1),
        jacobian_x=torch.zeros(1, 1, 1, 1),
        jacobian_y=torch.zeros(1, 1, 1, 1),
        appearance_validity=_scalar_map(0.0),
        flow_w=flow,
        flow_z=flow,
        match_proposal=proposal,
        match_precision=_scalar_map(1.0),
        beta_map=_scalar_map(1.0),
        damping_map=_scalar_map(0.0),
        charbonnier_alpha=1.0,
    )
    assert torch.allclose(
        result.delta.flatten(), torch.tensor([1.0, -1.0]), atol=1e-6
    )


def test_optical_lm_invalid_data_reduces_to_damped_consensus():
    flow_w = torch.tensor([[[[2.0]], [[0.0]]]])
    flow_z = torch.zeros_like(flow_w)
    result = solve_optical_lm_increment(
        residual=torch.ones(1, 1, 1, 1),
        jacobian_x=torch.ones(1, 1, 1, 1),
        jacobian_y=torch.ones(1, 1, 1, 1),
        appearance_validity=_scalar_map(0.0),
        flow_w=flow_w,
        flow_z=flow_z,
        match_proposal=flow_w,
        match_precision=_scalar_map(0.0),
        beta_map=_scalar_map(1.0),
        damping_map=_scalar_map(1.0),
        charbonnier_alpha=1.0,
    )
    assert torch.allclose(
        result.delta.flatten(), torch.tensor([-1.0, 0.0]), atol=1e-6
    )


def test_local_correlation_measurement_decodes_zero_offset():
    radius = 1
    correlation = torch.full((1, 9, 2, 3), -10.0)
    correlation[:, 4] = 10.0
    current = torch.randn(1, 2, 2, 3)
    measurement = local_correlation_measurement(
        correlation, current, radius=radius, temperature=0.1
    )
    assert torch.allclose(measurement.proposal, current, atol=1e-5)
    assert measurement.confidence.min() > 0.99


def test_source_only_proximal_interface_rejects_target_evidence_by_design():
    parameters = set(
        inspect.signature(SourceOnlyMotionProximal.forward).parameters
    )
    forbidden = {
        "target",
        "target_image",
        "target_features",
        "correlation",
        "photometric_residual",
        "depth2",
    }
    assert parameters.isdisjoint(forbidden)


def _camera(height: int, width: int):
    return torch.tensor(
        [
            [
                [50.0, 0.0, (width - 1) / 2.0],
                [0.0, 50.0, (height - 1) / 2.0],
                [0.0, 0.0, 1.0],
            ]
        ]
    )


def test_backprojection_and_projection_are_inverse_on_valid_depth():
    height, width = 5, 7
    depth = torch.full((1, 1, height, width), 10.0)
    intrinsics = _camera(height, width)
    points = backproject_depth(depth, intrinsics)
    pixels, valid = project_points(points, intrinsics)
    y, x = torch.meshgrid(
        torch.arange(height), torch.arange(width), indexing="ij"
    )
    expected = torch.stack((x, y), dim=0).float().unsqueeze(0)
    assert torch.allclose(pixels, expected, atol=1e-5)
    assert torch.equal(valid, torch.ones_like(valid))


def test_static_scene_identity_camera_induces_zero_flow():
    height, width = 5, 7
    depth = torch.full((1, 1, height, width), 10.0)
    scene = torch.zeros(1, 3, height, width)
    intrinsics = _camera(height, width)
    transform = identity_transform(
        1, device=depth.device, dtype=depth.dtype
    )
    projection = project_scene_flow(
        scene, depth, intrinsics, transform
    )
    assert projection.induced_flow.abs().max() < 1e-5


def test_camera_translation_is_not_conflated_with_scene_displacement():
    height, width = 5, 11
    depth = torch.full((1, 1, height, width), 10.0)
    scene = torch.zeros(1, 3, height, width)
    intrinsics = _camera(height, width)
    transform = torch.eye(4).unsqueeze(0)
    transform[:, 0, 3] = 1.0
    projection = project_scene_flow(
        scene, depth, intrinsics, transform
    )
    assert torch.allclose(
        projection.induced_flow[:, 0],
        torch.full((1, height, width), 5.0),
        atol=1e-5,
    )
    assert projection.induced_flow[:, 1].abs().max() < 1e-5


def test_lifting_projected_static_scene_recovers_zero_scene_flow():
    height, width = 5, 11
    depth = torch.full((1, 1, height, width), 10.0)
    intrinsics = _camera(height, width)
    transform = torch.eye(4).unsqueeze(0)
    transform[:, 0, 3] = 0.2
    projection = project_scene_flow(
        torch.zeros(1, 3, height, width),
        depth,
        intrinsics,
        transform,
    )
    lifted = lift_flow_to_scene_flow(
        projection.induced_flow,
        depth,
        depth,
        intrinsics,
        transform,
    )
    valid = lifted.valid.expand_as(lifted.scene_flow).bool()
    assert lifted.scene_flow[valid].abs().max() < 1e-4


def test_projective_scene_jacobian_matches_finite_difference():
    height, width = 3, 5
    depth = torch.full((1, 1, height, width), 10.0)
    intrinsics = _camera(height, width)
    transform = torch.eye(4).unsqueeze(0)
    scene = torch.zeros(1, 3, height, width)
    base = project_scene_flow(scene, depth, intrinsics, transform)
    jacobian = projective_scene_jacobian(
        base.target_points, intrinsics, transform
    )
    epsilon = 1e-3
    for component in range(3):
        perturbation = torch.zeros_like(scene)
        perturbation[:, component] = epsilon
        changed = project_scene_flow(
            perturbation, depth, intrinsics, transform
        )
        finite_difference = (
            changed.induced_flow - base.induced_flow
        ) / epsilon
        assert torch.allclose(
            finite_difference,
            jacobian[:, :, component],
            atol=2e-2,
            rtol=2e-2,
        )


def test_scene_lm_solves_full_rank_feature_system():
    appearance_residual = torch.tensor(
        [[[[1.0]], [[2.0]], [[3.0]]]]
    )
    appearance_jacobian = torch.zeros(1, 3, 3, 1, 1)
    appearance_jacobian[0, 0, 0] = 1.0
    appearance_jacobian[0, 1, 1] = 1.0
    appearance_jacobian[0, 2, 2] = 1.0
    scene = torch.zeros(1, 3, 1, 1)
    result = solve_scene_lm_increment(
        appearance_residual=appearance_residual,
        appearance_jacobian=appearance_jacobian,
        appearance_validity=_scalar_map(1.0),
        geometry_residual=torch.zeros(1, 1, 1, 1),
        geometry_jacobian=torch.zeros(1, 1, 3, 1, 1),
        geometry_validity=_scalar_map(0.0),
        induced_flow=torch.zeros(1, 2, 1, 1),
        projective_jacobian=torch.zeros(1, 2, 3, 1, 1),
        match_proposal=torch.zeros(1, 2, 1, 1),
        match_precision=_scalar_map(0.0),
        scene_flow_w=scene,
        scene_flow_z=scene,
        beta_map=_scalar_map(0.0),
        damping_map=_scalar_map(0.0),
        charbonnier_alpha=1.0,
    )
    assert torch.allclose(
        result.delta.flatten(),
        torch.tensor([-1.0, -2.0, -3.0]),
        atol=2e-5,
    )


def _small_model_cfg(model_type: str):
    key = "hqs_lm_of" if model_type == "hqs_lm_of" else "hqs_lm_sf"
    model_cfg = {
        "feature_channels": [16, 24, 32, 32],
        "match_channels": [16, 24, 32, 32],
        "context_channels": [16, 16, 24, 24],
        "blocks_per_scale": [1, 1, 1, 1],
        "groups": 8,
        "iterations": [1, 1, 1, 1],
        "jacobi_sweeps": [1, 1, 1, 1],
        "correlation_radii": [1, 1, 1, 1],
        "all_pairs_levels": [1, 2],
        "match_temperatures": [0.1, 0.1, 0.1, 0.1],
        "prior_hidden_channels": 16,
        "reliability_hidden_channels": 8,
        "upsample_hidden_dim": 16,
        "local_corr_channel_chunk": 0,
        "local_corr_checkpoint": False,
    }
    return OmegaConf.create(
        {"model": {"model_type": model_type, key: model_cfg}}
    )


def test_hqs_lm_of_factory_and_forward_smoke():
    model = build_model(_small_model_cfg("hqs_lm_of"))
    image1 = torch.rand(1, 3, 32, 32)
    image2 = torch.rand_like(image1)
    output = model(image1, image2)
    assert output["flow_preds"][-1].shape == (1, 2, 32, 32)
    assert torch.isfinite(output["flow_preds"][-1]).all()
    assert len(output["flow_preds"]) == 4
    assert all(
        torch.count_nonzero(value) == 0
        for value in output["learned_data_delta_lows"]
    )


def test_hqs_lm_sf_factory_and_forward_smoke():
    model = build_model(_small_model_cfg("hqs_lm_sf"))
    image1 = torch.rand(1, 3, 32, 32)
    image2 = torch.rand_like(image1)
    depth1 = torch.full((1, 1, 32, 32), 10.0)
    depth2 = depth1.clone()
    intrinsics = _camera(32, 32)
    transform = torch.eye(4).unsqueeze(0)
    output = model(
        image1, image2, depth1, depth2, intrinsics, transform
    )
    assert output["scene_flow_final"].shape == (1, 3, 32, 32)
    assert output["induced_flow_final"].shape == (1, 2, 32, 32)
    assert torch.isfinite(output["scene_flow_final"]).all()
    assert len(output["scene_flow_preds"]) == 4
