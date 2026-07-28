"""Repository-level smoke test for the HQSCore DeepMatch drop-in.

Run from the repository root:

    python smoketest_hqs_core_deep_match.py

The script constructs the complete overlay, executes the four-scale forward
pass, checks the correspondence diagnostics, and verifies a backward pass.
"""
from __future__ import annotations

from pathlib import Path

import torch
from omegaconf import OmegaConf

from models import build_model


def main() -> None:
    root = Path(__file__).resolve().parent
    cfg = OmegaConf.merge(
        OmegaConf.load(root / "configs/default.yaml"),
        OmegaConf.load(
            root / "configs/dropins/13_hqs_core_deep_match.yaml"
        ),
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg).to(device)
    model.train()

    # Multiples of 16 exercise the intended training path without padding
    # ambiguity.  Keep the smoke input small enough for CPU validation.
    image1 = torch.rand(1, 3, 64, 96, device=device)
    image2 = torch.rand_like(image1)
    output = model(image1, image2)

    assert len(output["flow_preds"]) == 10
    assert output["prediction_scales"] == [
        16,
        16,
        8,
        8,
        8,
        8,
        4,
        4,
        2,
        2,
    ]
    assert output["flow_preds"][-1].shape == (1, 2, 64, 96)
    assert output["global_topk_flow_xy"].shape == (1, 4, 2, 4, 6)
    assert output["global_topk_cycle_support"].shape == (
        1,
        4,
        1,
        4,
        6,
    )
    assert output["gmflow_reverse_flow_yx"].shape == (1, 2, 64, 96)
    assert all(
        torch.isfinite(value).all()
        for value in output["flow_preds"]
    )

    loss = output["flow_preds"][-1].square().mean()
    loss.backward()
    gradients = [
        parameter.grad
        for parameter in model.feature_encoder.pair_interactions.parameters()
        if parameter.requires_grad and parameter.grad is not None
    ]
    assert gradients
    assert all(torch.isfinite(value).all() for value in gradients)

    counts = model.param_count()
    print(
        "HQSCore DeepMatch smoke test passed "
        f"on {device.type}; parameters={counts['total']:,}; "
        f"trainable={counts['total_trainable']:,}."
    )


if __name__ == "__main__":
    main()
