import torch

from hqs_pytorch.customML.customModels.occlusion_geometry import (
    backward_warp_yx,
    compute_flow_geometry_yx,
    flow_in_bounds_mask,
    forward_splat_occupancy_yx,
)


def test_identity_geometry():
    flow = torch.zeros(2, 2, 8, 10, requires_grad=True)
    geometry = compute_flow_geometry_yx(flow, flow)
    assert torch.allclose(geometry.fb_error, torch.full_like(geometry.fb_error, 1e-3), atol=1e-6)
    assert torch.all(geometry.in_bounds == 1)
    assert torch.allclose(geometry.occupancy, torch.ones_like(geometry.occupancy))
    loss = geometry.occupancy.mean() + geometry.fb_confidence.mean()
    loss.backward()
    assert flow.grad is not None
    assert torch.isfinite(flow.grad).all()


def test_constant_translation_has_target_hole():
    flow = torch.zeros(1, 2, 6, 8)
    flow[:, 1] = 1.0  # dx = +1
    occupancy = forward_splat_occupancy_yx(flow)
    assert torch.allclose(occupancy[:, :, :, 0], torch.zeros_like(occupancy[:, :, :, 0]))
    assert torch.all(occupancy[:, :, :, 1:] > 0.99)


def test_in_bounds_right_border():
    flow = torch.zeros(1, 2, 4, 5)
    flow[:, 1] = 1.0
    valid = flow_in_bounds_mask(flow)
    assert torch.all(valid[:, :, :, :-1] == 1)
    assert torch.all(valid[:, :, :, -1] == 0)


def test_backward_warp_yx_channel_order():
    image = torch.arange(12.0).reshape(1, 1, 3, 4)
    flow = torch.zeros(1, 2, 3, 4)
    flow[:, 1] = 1.0  # sample one pixel right
    warped = backward_warp_yx(image, flow)
    assert torch.allclose(warped[0, 0, :, :-1], image[0, 0, :, 1:])
