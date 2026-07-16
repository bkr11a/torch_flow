"""Build and one-step gradient smoke test for the spectral/occlusion drop-ins."""
from __future__ import annotations

import argparse

import torch
from omegaconf import OmegaConf

from losses import HQSFlowLoss
from models import build_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--override", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--height", type=int, default=128)
    parser.add_argument("--width", type=int, default=256)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = OmegaConf.merge(
        OmegaConf.load(args.config), OmegaConf.load(args.override)
    )
    device = torch.device(args.device)
    model = build_model(cfg).to(device).train()
    criterion = HQSFlowLoss(cfg.loss).to(device)

    image1 = torch.rand(1, 3, args.height, args.width, device=device)
    image2 = torch.rand_like(image1)
    flow_gt = torch.randn(1, 2, args.height, args.width, device=device)
    valid = torch.ones(1, args.height, args.width, device=device)
    occlusion = torch.zeros_like(valid)
    occlusion[..., args.width // 3 : args.width // 2] = 1.0
    synthetic = torch.zeros_like(valid)
    synthetic[..., args.width // 2 : 2 * args.width // 3] = 1.0

    output = model(image1, image2)
    criterion.set_step(10_000)
    losses = criterion(
        output["flow_preds"],
        flow_gt,
        valid,
        image1,
        image2,
        model_outputs=output,
        occlusion=occlusion,
        synthetic_occlusion=synthetic,
    )
    if not torch.isfinite(losses["loss"]):
        raise RuntimeError(f"Non-finite loss: {losses['loss']}")
    losses["loss"].backward()

    finite_gradients = 0
    for parameter in model.parameters():
        if parameter.grad is not None:
            if not torch.isfinite(parameter.grad).all():
                raise RuntimeError("A model gradient is non-finite.")
            finite_gradients += 1
    if finite_gradients == 0:
        raise RuntimeError("No model gradients were produced.")

    prediction_count = len(output["flow_preds"])
    loss_value = float(losses["loss"].detach())
    del output, losses

    external = torch.ones(1, 1, args.height, args.width, device=device)
    try:
        with torch.no_grad():
            model(image1, image2, source_valid=external)
    except ValueError:
        pass
    else:
        if not cfg.model.model_backbone.get("allow_external_validity_inputs", False):
            raise RuntimeError("External validity was not rejected.")

    print(
        "drop-in smoke test passed:",
        f"predictions={prediction_count}",
        f"loss={loss_value:.6f}",
        f"gradient_tensors={finite_gradients}",
    )


if __name__ == "__main__":
    main()
