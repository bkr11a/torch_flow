import torch

from hqs_pytorch.customML.customModels.factorised_reliability import (
    FactorisedReliabilityHead,
)
from hqs_pytorch.customML.customModels.reliability_aware_blocks import (
    OcclusionPriorNet,
)


def test_reliability_shapes_and_gradients():
    head = FactorisedReliabilityHead(in_channels=11, hidden_channels=16)
    x = torch.randn(2, 11, 8, 9, requires_grad=True)
    state = head(x)
    assert state.p_visible.shape == (2, 1, 8, 9)
    assert state.p_match.shape == (2, 1, 8, 9)
    assert state.sigma_data.shape == (2, 1, 8, 9)
    assert state.p_boundary.shape == (2, 1, 8, 9)
    state.p_visible.mean().backward()
    assert x.grad is not None


def test_prior_has_no_target_argument():
    prior = OcclusionPriorNet(context_channels=8, hidden_channels=16)
    signature_names = prior.forward.__code__.co_varnames[: prior.forward.__code__.co_argcount]
    forbidden = {"target", "warped_target", "corr", "image2", "fmap2"}
    assert forbidden.isdisjoint(signature_names)
