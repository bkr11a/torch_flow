#!/usr/bin/env python3
"""Acceptance gate for the frozen official-GMFlow HQSCore drop-in."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from omegaconf import OmegaConf, open_dict

from hqs_pytorch.customML.customModels.HQSCore import HQSCore
from losses import HQSFlowLoss


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("pretrained/gmflow_things-e9887eda.pth"),
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--height", type=int, default=64)
    parser.add_argument("--width", type=int, default=96)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if args.height % 16 or args.width % 16:
        raise ValueError("Preflight height and width must be divisible by 16")

    root = Path(__file__).resolve().parent
    cfg = OmegaConf.merge(
        OmegaConf.load(root / "configs/default.yaml"),
        OmegaConf.load(
            root / "configs/dropins/16_hqs_core_pretrained_gmflow.yaml"
        ),
    )
    with open_dict(cfg):
        cfg.model.hqs_core.gmflow_pretrained_checkpoint = str(
            args.checkpoint.expanduser().resolve()
        )

    model = HQSCore(cfg.model).to(device).train()
    criterion = HQSFlowLoss(cfg.loss).to(device)
    report = model.gmflow_pretrained_report
    if report is None:
        raise RuntimeError("Official GMFlow checkpoint was not loaded")

    torch.manual_seed(1701)
    source = torch.rand(1, 3, args.height, args.width, device=device)
    target = torch.rand_like(source)
    output = model(source, target)
    ground_truth = torch.zeros(
        1,
        2,
        args.height,
        args.width,
        device=device,
    )
    valid = torch.ones(1, args.height, args.width, device=device)
    losses = criterion(
        output["flow_preds"],
        ground_truth,
        valid,
        source,
        target,
        model_outputs=output,
    )
    losses["loss"].backward()

    matcher_parameters = list(model.gmflow_matcher.parameters())
    solver_gradients = [
        parameter.grad
        for name, parameter in model.named_parameters()
        if not name.startswith("gmflow_matcher.")
        and parameter.grad is not None
    ]
    checks = {
        "strict_checkpoint_loaded": report.loaded_tensor_count > 0,
        "released_graph_enabled": bool(
            model.gmflow_matcher.released_weights_compatible
        ),
        "matcher_frozen": all(
            not parameter.requires_grad for parameter in matcher_parameters
        ),
        "matcher_has_no_gradients": all(
            parameter.grad is None for parameter in matcher_parameters
        ),
        "solver_has_finite_gradients": bool(solver_gradients)
        and all(torch.isfinite(value).all() for value in solver_gradients),
        "raw_routing": model.global_propagation_routing == "raw",
        "propagation_disabled": not model.global_use_flow_propagation,
        "fb_data_gate_disabled": not model.global_fb_data_gate,
        "finite_raw_candidate": bool(
            torch.isfinite(output["global_init_candidate_flow_xy"]).all()
        ),
        "finite_final_prediction": bool(
            torch.isfinite(output["flow_preds"][-1]).all()
        ),
        "ten_hqs_predictions": len(output["flow_preds"]) == 10,
    }
    result = {
        "passed": all(checks.values()),
        "checks": checks,
        "checkpoint": report.path,
        "checkpoint_load": report.summary(),
        "parameters": model.param_count(),
        "loss": float(losses["loss"].detach().cpu()),
    }
    print(json.dumps(result, indent=2))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
