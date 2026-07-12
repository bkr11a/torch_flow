import torch

from hqs_pytorch.customML.customModels.factorised_reliability import ReliabilityState
from losses.occlusion_aware_losses import visible_outlier_mixture_nll


def _state(logit):
    x = torch.full((1, 1, 4, 4), float(logit), requires_grad=True)
    return ReliabilityState(
        visibility_logits=x,
        match_logits=torch.full_like(x, 2.0, requires_grad=True),
        log_sigma_data=torch.zeros_like(x, requires_grad=True),
        boundary_logits=torch.full_like(x, -2.0, requires_grad=True),
    )


def test_all_occluded_is_not_free():
    residual = torch.full((1, 1, 4, 4), 0.2)
    low = visible_outlier_mixture_nll(residual, _state(-20.0))
    high = visible_outlier_mixture_nll(residual, _state(5.0))
    assert torch.isfinite(low)
    assert torch.isfinite(high)
    assert low.item() > 0.0


def test_extreme_logits_have_finite_gradients():
    residual = torch.randn(1, 2, 4, 4)
    state = _state(-30.0)
    loss = visible_outlier_mixture_nll(residual, state)
    loss.backward()
    assert torch.isfinite(state.visibility_logits.grad).all()
