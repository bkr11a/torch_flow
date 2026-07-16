import random

import numpy as np
import torch

from data.augmentation import RandomErase
from hqs_pytorch.customML.customModels.band_split_refinement import (
    BandSplitHQSRefiner,
    fixed_high_pass,
)
from hqs_pytorch.customML.customModels.factorised_reliability import (
    ReliabilityState,
)
from hqs_pytorch.customML.customModels.pgma import PhysicsGatedMatchingAttention
from losses.flow_loss import HQSFlowLoss


def test_fixed_high_pass_annihilates_constant_flow():
    flow = torch.full((2, 2, 12, 16), 3.0)
    assert torch.allclose(fixed_high_pass(flow), torch.zeros_like(flow))


def test_fixed_high_pass_has_zero_block_average_for_odd_shapes():
    flow = torch.randn(2, 2, 13, 17)
    detail = fixed_high_pass(flow, factor=2)
    pooled = torch.nn.functional.avg_pool2d(
        detail,
        kernel_size=2,
        stride=2,
        ceil_mode=True,
        count_include_pad=False,
    )
    assert torch.allclose(pooled, torch.zeros_like(pooled), atol=1e-6)


def test_band_split_identity_for_identical_images_and_zero_flow():
    module = BandSplitHQSRefiner(
        context_channels=8,
        hidden_channels=16,
        half_iterations=1,
    )
    image = torch.rand(1, 3, 16, 20)
    context = torch.rand(1, 8, 8, 10)
    initial = torch.zeros(1, 2, 4, 5)
    result = module(
        source_context=context,
        source_image=image,
        target_image=image,
        initial_flow_yx=initial,
    )
    assert result["flow_yx"].shape == (1, 2, 16, 20)
    assert torch.allclose(result["flow_yx"], torch.zeros_like(result["flow_yx"]), atol=1e-5)


def test_global_matcher_returns_calibrated_bidirectional_fields():
    matcher = PhysicsGatedMatchingAttention(
        feature_dim=4,
        temperature=0.01,
        use_topk=False,
        match_mode="soft",
        use_feature_enhancer=False,
        query_chunk_size=2,
        position_scale=0.0,
    )
    with torch.no_grad():
        matcher.q_proj.weight.zero_()
        matcher.k_proj.weight.zero_()
        for channel in range(4):
            matcher.q_proj.weight[channel, channel, 0, 0] = 1.0
            matcher.k_proj.weight[channel, channel, 0, 0] = 1.0
    feature = torch.eye(4).reshape(1, 4, 2, 2)
    result = matcher.global_match(feature, feature)
    assert result["flow_yx"].shape == (1, 2, 2, 2)
    assert result["reverse_flow_yx"].shape == (1, 2, 2, 2)
    assert torch.all(result["mutual"] == 1)
    assert torch.allclose(result["flow_yx"], torch.zeros_like(result["flow_yx"]), atol=1e-3)


def test_random_erase_records_loss_only_target_mask():
    random.seed(4)
    sample = {
        "image1": np.zeros((32, 40, 3), dtype=np.float32),
        "image2": np.ones((32, 40, 3), dtype=np.float32),
        "flow": np.zeros((32, 40, 2), dtype=np.float32),
        "valid": np.ones((32, 40), dtype=bool),
        "occlusion": None,
        "invalid": None,
        "synthetic_occlusion": None,
    }
    output = RandomErase(prob=1.0, num_patches=1)(sample)
    assert output["synthetic_occlusion"].dtype == np.bool_
    assert output["synthetic_occlusion"].any()


def test_global_data_proposal_is_not_supervised_in_occlusion():
    criterion = HQSFlowLoss({"global_init_weight": 1.0})
    flow_gt = torch.zeros(1, 2, 8, 10)
    final = torch.zeros_like(flow_gt)
    proposal_yx = torch.zeros_like(flow_gt)
    proposal_yx[..., 5:] = 100.0
    valid = torch.ones(1, 8, 10)
    occlusion = torch.zeros_like(valid)
    occlusion[..., 5:] = 1.0
    result = criterion(
        [final],
        flow_gt,
        valid,
        model_outputs={"gmflow_init_flow_yx": proposal_yx},
        occlusion=occlusion,
    )
    assert result["global_init"] < 0.01


def test_predicted_boundary_factor_receives_gt_boundary_gradient():
    criterion = HQSFlowLoss(
        {"occlusion_aware": {"enabled": True, "boundary_weight": 1.0}}
    )
    flow_gt = torch.zeros(1, 2, 8, 10)
    flow_gt[:, 0, :, 5:] = 10.0
    valid = torch.ones(1, 8, 10)
    logits = torch.zeros(1, 1, 4, 5, requires_grad=True)
    state = ReliabilityState(
        visibility_logits=torch.full_like(logits, 2.0, requires_grad=True),
        match_logits=torch.full_like(logits, 2.0, requires_grad=True),
        log_sigma_data=torch.zeros_like(logits, requires_grad=True),
        boundary_logits=logits,
    )
    terms = criterion._factorised_reliability_loss(
        {"reliability_states": [state], "flow_low": [torch.zeros(1, 2, 4, 5)]},
        flow_gt,
        valid,
        occlusion=None,
        invalid=None,
        synthetic_occlusion=None,
    )
    terms["occ_boundary"].backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()
