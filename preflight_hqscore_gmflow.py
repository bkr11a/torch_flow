#!/usr/bin/env python3
"""Pre-curriculum acceptance gate for the HQSCore GMFlow drop-in."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from omegaconf import OmegaConf

from hqs_pytorch.customML.customModels.HQSCore import HQSCore
from losses import HQSFlowLoss
from models.hqs_gmflow_components import gmflow_forward_backward_consistency


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--height", type=int, default=32)
    parser.add_argument("--width", type=int, default=48)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    root = Path(__file__).resolve().parent
    cfg = OmegaConf.merge(
        OmegaConf.load(root / "configs/default.yaml"),
        OmegaConf.load(root / "configs/dropins/15_hqs_core_gmflow.yaml"),
    )
    model = HQSCore(cfg.model).to(device).train()
    criterion = HQSFlowLoss(cfg.loss).to(device)

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
    valid = torch.ones(
        1,
        args.height,
        args.width,
        device=device,
    )
    losses = criterion(
        output["flow_preds"],
        ground_truth,
        valid,
        source,
        target,
        output,
    )
    losses["loss"].backward()

    matcher_gradients = [
        parameter.grad
        for parameter in model.gmflow_matcher.parameters()
        if parameter.requires_grad and parameter.grad is not None
    ]
    forward = torch.zeros(1, 2, 5, 7, device=device)
    reverse = torch.zeros_like(forward)
    forward[:, 0] = 1.0
    reverse[:, 0] = -1.0
    synthetic = gmflow_forward_backward_consistency(
        forward,
        reverse,
        feature_scale=1,
        softness_full_resolution=0.05,
    )

    checks = {
        "ten_hqs_predictions": len(output["flow_preds"]) == 10,
        "native_one_eighth_candidate": (
            tuple(output["global_init_candidate_flow_xy"].shape[-2:])
            == (args.height // 8, args.width // 8)
        ),
        "bidirectional_decode": output["gmflow_reverse_flow_yx"] is not None,
        "finite_forward_backward_reliability": bool(
            torch.isfinite(output["global_fb_reliability"]).all()
        ),
        "bounded_forward_backward_reliability": bool(
            (
                (output["global_fb_reliability"] >= 0.0)
                & (output["global_fb_reliability"] <= 1.0)
            ).all()
        ),
        "raw_matcher_loss_present": "global_init" in losses,
        "propagated_matcher_loss_present": "global_propagated" in losses,
        "forward_backward_visibility_loss_present": (
            "global_fb_visibility" in losses
        ),
        "finite_matcher_gradients": bool(matcher_gradients)
        and all(torch.isfinite(gradient).all() for gradient in matcher_gradients),
        "synthetic_consistent_interior": bool(
            synthetic["reliability"][..., :-1].mean() > 0.99
        ),
        "synthetic_boundary_occlusion": bool(
            (synthetic["occlusion"][..., -1] == 1).all()
        ),
        "proximal_not_warmup_parameter": all(
            prefix == "gmflow_matcher."
            for prefix in model.global_matcher_parameter_prefixes
        ),
    }
    report = {
        "passed": all(checks.values()),
        "checks": checks,
        "parameters": model.param_count(),
        "loss": float(losses["loss"].detach().cpu()),
        "global_init_loss": float(losses["global_init"].detach().cpu()),
        "global_propagated_loss": float(
            losses["global_propagated"].detach().cpu()
        ),
    }
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
