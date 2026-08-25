#!/usr/bin/env python3
"""Instantiate A0/A1 and verify total/trainable parameter matching.

This checker is intentionally runnable as either::

    python tools/check_recurrent_capacity.py ...
    python -m tools.check_recurrent_capacity ...

For capacity auditing, pretrained GMFlow *values* are irrelevant. If the
configured checkpoint is not present on the current host, the checker builds
GMFlow without loading a checkpoint and then freezes it before parameter
counting, reproducing the trainable/non-trainable partition required by the
ablation without requiring the weight file merely to count parameters.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Tuple

# Script execution places tools/ on sys.path. Add the repository root so the
# repository's top-level ``models`` package is importable.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from omegaconf import DictConfig, OmegaConf, open_dict

from models import build_model


def counts(model) -> Tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return int(total), int(trainable)


def _resolved_checkpoint_path(value) -> Path | None:
    if value in (None, "", "null"):
        return None
    text = str(value)
    # OmegaConf interpolation has normally resolved by attribute access, but
    # keep the common ${oc.env:VAR,default} case robust for diagnostics.
    if text.startswith("${oc.env:") and text.endswith("}"):
        body = text[len("${oc.env:") : -1]
        var, _, default = body.partition(",")
        text = os.environ.get(var.strip(), default.strip())
    path = Path(text).expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path


def _prepare_for_capacity_only(cfg: DictConfig, label: str) -> bool:
    """Avoid requiring the GMFlow weight file solely for parameter counting.

    Returns True when the matcher must be frozen manually after construction.
    """
    core = cfg.get("model", {}).get("hqs_core", None)
    if core is None or str(core.get("matching_pyramid", "")).lower() != "gmflow":
        return False
    if not bool(core.get("gmflow_freeze_pretrained", False)):
        return False

    checkpoint = _resolved_checkpoint_path(core.get("gmflow_pretrained_checkpoint", None))
    if checkpoint is not None and checkpoint.is_file():
        return False

    print(
        f"NOTE [{label}]: GMFlow checkpoint is not present on this host; "
        "building without loading weights and freezing GMFlow for capacity counting."
    )
    if checkpoint is not None:
        print(f"  configured checkpoint: {checkpoint}")

    with open_dict(core):
        core.gmflow_pretrained_checkpoint = None
        core.gmflow_freeze_pretrained = False
    return True


def _build_capacity_model(cfg: DictConfig, label: str):
    freeze_matcher_after_build = _prepare_for_capacity_only(cfg, label)
    model = build_model(cfg)
    if freeze_matcher_after_build:
        matcher = getattr(model, "gmflow_matcher", None)
        if matcher is None:
            raise RuntimeError(
                f"{label}: expected GMFlow matcher after capacity-only construction"
            )
        if hasattr(matcher, "set_trainable"):
            matcher.set_trainable(False)
        else:
            for parameter in matcher.parameters():
                parameter.requires_grad_(False)
    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-config", default="configs/default.yaml")
    parser.add_argument("--a0", default="configs/dropins/16_hqs_core_pretrained_gmflow.yaml")
    parser.add_argument("--a1", default="configs/dropins/17_hqs_core_recurrent_control.yaml")
    parser.add_argument("--tolerance", type=float, default=0.01)
    args = parser.parse_args()

    base = OmegaConf.load(args.base_config)
    a0_cfg = OmegaConf.merge(base, OmegaConf.load(args.a0))
    a1_cfg = OmegaConf.merge(base, OmegaConf.load(args.a1))

    a0 = _build_capacity_model(a0_cfg, "A0")
    a1 = _build_capacity_model(a1_cfg, "A1")

    a0_total, a0_train = counts(a0)
    a1_total, a1_train = counts(a1)
    relative = abs(a1_train - a0_train) / float(max(a0_train, 1))

    print(f"A0 total/trainable: {a0_total:,} / {a0_train:,}")
    print(f"A1 total/trainable: {a1_total:,} / {a1_train:,}")
    print(f"Trainable parameter mismatch: {100.0 * relative:.4f}%")
    report = getattr(a1, "capacity_match_report", None)
    if report is not None:
        print("Cell-level capacity match:")
        print(report)
    if relative > args.tolerance:
        raise SystemExit(
            f"FAILED: {100.0 * relative:.4f}% > {100.0 * args.tolerance:.4f}% tolerance"
        )
    print("PASS")


if __name__ == "__main__":
    main()
