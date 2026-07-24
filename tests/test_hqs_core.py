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
    Linearisation,
    SharedValidityHead,
    SourceConditionedProxResidual,
    StructuredHQSCell,
    spatial_gradients,
    weighted_analytic_data_delta,
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


def test_weighted_analytic_overlay_selects_corrected_operator():
    root = Path(__file__).resolve().parents[1]
    cfg = OmegaConf.merge(
        OmegaConf.load(root / "configs/default.yaml"),
        OmegaConf.load(
            root / "configs/dropins/07_hqs_core_weighted_analytic.yaml"
        ),
    )
    assert cfg.model.model_type == "hqs_core"
    assert cfg.model.hqs_core.analytic_validity_mode == "weighted_solve"
    assert list(cfg.model.hqs_core.iterations) == [2, 4, 2, 2]
    assert cfg.loss.num_stages == 10


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


def test_weighted_analytic_delta_matches_batched_linear_solve():
    torch.manual_seed(11)
    batch, height, width = 3, 4, 5
    dtype = torch.float64
    grad_x = torch.randn(batch, 1, height, width, dtype=dtype)
    grad_y = torch.randn(batch, 1, height, width, dtype=dtype)
    residual = torch.randn(batch, 1, height, width, dtype=dtype)
    validity = torch.rand(batch, 1, height, width, dtype=dtype)
    beta_map = 0.01 + torch.rand(
        batch, 1, height, width, dtype=dtype
    )
    linearisation = Linearisation(
        residual=residual,
        grad_x=grad_x,
        grad_y=grad_y,
        in_bounds=torch.ones_like(validity),
    )

    actual = weighted_analytic_data_delta(
        linearisation,
        validity,
        beta_map,
    )

    gradient = torch.cat((grad_x, grad_y), dim=1)
    gradient_vectors = gradient.permute(0, 2, 3, 1).unsqueeze(-1)
    identity = torch.eye(dtype=dtype).view(1, 1, 1, 2, 2)
    system = (
        beta_map.permute(0, 2, 3, 1).unsqueeze(-1) * identity
        + validity.permute(0, 2, 3, 1).unsqueeze(-1)
        * gradient_vectors
        * gradient_vectors.transpose(-1, -2)
    )
    right_hand_side = (
        -validity.permute(0, 2, 3, 1).unsqueeze(-1)
        * residual.permute(0, 2, 3, 1).unsqueeze(-1)
        * gradient_vectors
    )
    expected = torch.linalg.solve(system, right_hand_side)
    expected = expected.squeeze(-1).permute(0, 3, 1, 2)

    assert torch.allclose(actual, expected, atol=1e-10, rtol=1e-10)


def test_weighted_analytic_delta_has_correct_soft_validity_limits():
    linearisation = Linearisation(
        residual=torch.ones(1, 1, 1, 1),
        grad_x=torch.ones(1, 1, 1, 1),
        grad_y=torch.zeros(1, 1, 1, 1),
        in_bounds=torch.ones(1, 1, 1, 1),
    )
    beta_map = torch.ones(1, 1, 1, 1)

    invalid = weighted_analytic_data_delta(
        linearisation,
        torch.zeros(1, 1, 1, 1),
        beta_map,
    )
    soft = weighted_analytic_data_delta(
        linearisation,
        torch.full((1, 1, 1, 1), 0.5),
        beta_map,
    )
    valid = weighted_analytic_data_delta(
        linearisation,
        torch.ones(1, 1, 1, 1),
        beta_map,
    )

    assert torch.equal(invalid, torch.zeros_like(invalid))
    assert torch.allclose(soft[:, 0], torch.full((1, 1, 1), -1.0 / 3.0))
    assert torch.allclose(valid[:, 0], torch.full((1, 1, 1), -0.5))
    assert torch.equal(soft[:, 1], torch.zeros_like(soft[:, 1]))


def test_weighted_solve_rejects_unknown_validity_mode():
    try:
        StructuredHQSCell(
            correlation_channels=8,
            context_channels=8,
            hidden_channels=8,
            max_iterations=1,
            analytic_validity_mode="unknown",
        )
    except ValueError as error:
        assert "analytic_validity_mode" in str(error)
    else:
        raise AssertionError("Unknown analytic validity mode was accepted")


def test_weighted_solve_preserves_exact_zero_validity_invariant():
    cell = StructuredHQSCell(
        correlation_channels=8,
        context_channels=8,
        hidden_channels=8,
        max_iterations=1,
        prior_hidden_channels=8,
        groups=4,
        analytic_validity_mode="weighted_solve",
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
    assert torch.equal(
        output.analytic_delta,
        torch.zeros_like(output.analytic_delta),
    )
    assert torch.equal(output.data_delta, torch.zeros_like(output.data_delta))
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
