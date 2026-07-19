from __future__ import annotations

import inspect
from pathlib import Path
from types import SimpleNamespace

import torch
from omegaconf import OmegaConf

from hqs_pytorch.customML.customModels.HQSCore import HQSCore
from losses import HQSFlowLoss
from models import build_model
from models.correlation import LocalCorrBlock
from models.hqs_core_components import (
    AllPairsCorrelation,
    HQSState,
    SharedValidityHead,
    SourceConditionedProxResidual,
    StructuredHQSCell,
    spatial_gradients,
)
from models.warp import resize_flow


def _tiny_core_config(iterations=(1, 1, 1, 1)):
    return {
        "feature_channels": [16, 24, 32, 32],
        "match_channels": [16, 24, 32, 32],
        "context_channels": [16, 16, 24, 24],
        "blocks_per_scale": [0, 0, 0, 0],
        "groups": 4,
        "iterations": list(iterations),
        "jacobi_sweeps": [1, 1, 1, 1],
        "correlation_radii": [1, 1, 1, 1],
        "all_pairs_levels": [1, 2],
        "correlation_embedding_dim": 8,
        "coarse_hidden_dim": 12,
        "fine_hidden_dim": 8,
        "prior_hidden_channels": 8,
        "validity_hidden_dim": 8,
        "upsample_hidden_dim": 8,
        "global_query_chunk_size": 8,
    }


def test_hqs_core_overlay_is_a_complete_ten_stage_switch():
    root = Path(__file__).resolve().parents[1]
    cfg = OmegaConf.merge(
        OmegaConf.load(root / "configs/default.yaml"),
        OmegaConf.load(root / "configs/dropins/06_hqs_core.yaml"),
    )
    assert cfg.model.model_type == "hqs_core"
    assert list(cfg.model.hqs_core.iterations) == [2, 4, 2, 2]
    assert cfg.loss.num_stages == 10
    assert cfg.loss.occlusion_aware.enabled is False


def test_hqs_core_is_selected_only_by_model_type_switch():
    cfg = SimpleNamespace(
        model={
            "model_type": "hqs_core",
            "hqs_core": _tiny_core_config(),
        }
    )
    model = build_model(cfg)
    assert isinstance(model, HQSCore)
    assert model.num_hqs_iterations == 4


def test_hqs_core_four_scale_forward_contract_and_gradient():
    model = HQSCore({"hqs_core": _tiny_core_config()})
    image1 = torch.rand(1, 3, 32, 48)
    image2 = torch.rand(1, 3, 32, 48)
    output = model(image1, image2)

    assert output["prediction_scales"] == [16, 8, 4, 2]
    assert len(output["flow_preds"]) == 4
    assert len(output["flow_low"]) == 4
    assert output["flow_preds"][-1].shape == (1, 2, 32, 48)
    assert output["final_upsample_mask_logits"].shape[1] == 36
    assert all(torch.isfinite(flow).all() for flow in output["flow_preds"])
    assert all(
        ((mask >= 0.0) & (mask <= 1.0)).all()
        for mask in output["core_validity_lows"]
    )

    loss = output["flow_preds"][-1].abs().mean()
    loss.backward()
    encoder_grads = [
        parameter.grad
        for parameter in model.feature_encoder.parameters()
        if parameter.requires_grad and parameter.grad is not None
    ]
    assert encoder_grads
    assert all(torch.isfinite(grad).all() for grad in encoder_grads)


def test_canonical_iteration_plan_has_ten_predictions():
    model = HQSCore(
        {"hqs_core": _tiny_core_config(iterations=(2, 4, 2, 2))}
    )
    assert model.num_hqs_iterations == 10
    assert model.iterations_by_scale == {16: 2, 8: 4, 4: 2, 2: 2}


def test_out_of_bounds_data_update_is_exactly_gated():
    cell = StructuredHQSCell(
        correlation_channels=8,
        context_channels=8,
        hidden_channels=8,
        max_iterations=1,
        prior_hidden_channels=8,
        groups=4,
    )
    validity_head = SharedValidityHead(
        correlation_channels=8, hidden_channels=8
    )
    source = torch.rand(1, 1, 8, 8)
    target = torch.rand(1, 1, 8, 8)
    grad_x, grad_y = spatial_gradients(target)
    q = torch.full((1, 2, 8, 8), 100.0)
    context = torch.rand(1, 8, 8, 8)
    state = HQSState(
        w=torch.randn_like(q),
        q=q,
        hidden=cell.data_operator.initialise_hidden(context),
    )
    output = cell(
        source_gray=source,
        target_gray=target,
        target_grad_x=grad_x,
        target_grad_y=grad_y,
        correlation=torch.rand(1, 8, 8, 8),
        source_context=context,
        state=state,
        validity_head=validity_head,
        iteration=0,
        jacobi_sweeps=1,
        max_data_delta=2.0,
        max_prox_delta=0.5,
    )
    assert torch.count_nonzero(output.validity) == 0
    assert torch.equal(output.state.w, q)


def test_proximal_interface_has_no_target_or_correlation_argument():
    parameters = set(
        inspect.signature(SourceConditionedProxResidual.forward).parameters
    )
    forbidden = {
        "target",
        "target_image",
        "target_features",
        "correlation",
        "photometric_residual",
    }
    assert parameters.isdisjoint(forbidden)


def test_all_pairs_global_match_and_lookup_are_finite_on_tiny_grids():
    # Identical one-hot spatial descriptors have a near-identity global match.
    fmap = torch.eye(4).reshape(1, 4, 2, 2)
    corr = AllPairsCorrelation(fmap, fmap, num_levels=4, radius=1)
    match = corr.global_soft_match(temperature=0.01, query_chunk_size=2)
    assert match["flow_xy"].abs().max() < 1e-3
    assert match["confidence"].min() > 0.9
    lookup = corr.lookup(torch.zeros(1, 2, 2, 2))
    assert lookup.shape == (1, 4 * 9, 2, 2)
    assert torch.isfinite(lookup).all()


def test_resize_flow_rescales_xy_components_for_odd_shapes():
    flow = torch.ones(1, 2, 3, 5)
    resized = resize_flow(flow, (7, 11))
    assert resized.shape == (1, 2, 7, 11)
    assert torch.allclose(resized[:, 0], torch.full((1, 7, 11), 11 / 5))
    assert torch.allclose(resized[:, 1], torch.full((1, 7, 11), 7 / 3))


def test_chunked_local_correlation_matches_reference_path():
    torch.manual_seed(7)
    fmap1 = torch.randn(1, 5, 7, 9)
    fmap2 = torch.randn_like(fmap1)
    flow = 0.25 * torch.randn(1, 2, 7, 9)
    reference = LocalCorrBlock(radius=2, channel_chunk_size=0)(
        fmap1, fmap2, flow
    )
    chunked = LocalCorrBlock(radius=2, channel_chunk_size=2)(
        fmap1, fmap2, flow
    )
    assert torch.allclose(chunked, reference, atol=1e-5, rtol=1e-5)


def test_core_reliability_is_supervised_only_through_the_loss_interface():
    criterion = HQSFlowLoss(
        {
            "num_stages": 1,
            "stage_weight_mode": "geometric",
            "core_visibility_weight": 0.1,
            "core_visibility_last_n": 1,
            "occlusion_aware": {"enabled": False},
        }
    )
    prediction = torch.zeros(1, 2, 8, 10, requires_grad=True)
    flow_gt = torch.zeros_like(prediction)
    valid = torch.ones(1, 8, 10)
    occlusion = torch.zeros_like(valid)
    occlusion[..., 3:5] = 1.0
    reliability = torch.full(
        (1, 1, 4, 5), 0.8, requires_grad=True
    )
    output = criterion(
        [prediction],
        flow_gt,
        valid,
        model_outputs={"core_reliability_lows": [reliability]},
        occlusion=occlusion,
    )
    assert "core_visibility" in output
    assert torch.isfinite(output["loss"])
    output["loss"].backward()
    assert reliability.grad is not None
    assert torch.isfinite(reliability.grad).all()
