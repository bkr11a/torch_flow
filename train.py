#!/usr/bin/env python3
from __future__ import annotations

# Allow MPS to fall back to CPU for ops not yet implemented (e.g. grid_sampler_2d_backward).
import os; os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import argparse
import logging
import sys

from omegaconf import OmegaConf, DictConfig, open_dict
from dotenv import load_dotenv


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train HQSFlow optical flow network",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    p.add_argument(
        "--config", "-c", required=True,
        help="Path to a YAML config file (e.g. configs/default.yaml).",
    )
    p.add_argument(
        "--override", "-O", default=None,
        help="Optional additional YAML file whose settings override --config.",
    )
    p.add_argument(
        "overrides", nargs="*",
        help="Dot-notation key=value overrides, e.g. model.model_backbone.num_hqs_iterations=8",
    )
    return p.parse_args()


def setup_logging(log_dir: str, run_name: str) -> None:
    log_file = os.path.join(log_dir, f"{run_name}.log")
    os.makedirs(os.path.dirname(log_file) or ".", exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file),
        ],
    )


def main() -> None:
    # Load local environment variables (e.g. MLflow TLS/cert settings).
    load_dotenv(override=False)

    args = parse_args()

    # Load base config
    cfg: DictConfig = OmegaConf.load(args.config)

    # Merge override file if provided
    if args.override:
        override_cfg = OmegaConf.load(args.override)
        cfg = OmegaConf.merge(cfg, override_cfg)

    # Apply CLI key=value overrides
    if args.overrides:
        cli_cfg = OmegaConf.from_dotlist(args.overrides)
        cfg = OmegaConf.merge(cfg, cli_cfg)

    # Persist launch metadata so Trainer can archive the exact input YAML(s).
    with open_dict(cfg):
        cfg.launch = {
            "config_path": os.path.abspath(args.config),
            "override_path": os.path.abspath(args.override) if args.override else None,
            "cli_overrides": list(args.overrides),
        }

    setup_logging(cfg.get("log_dir", "logs"), cfg.get("run_name", "hqs_flow"))
    logger = logging.getLogger(__name__)
    logger.info("Config:\n" + OmegaConf.to_yaml(cfg))

    from engine import Trainer
    trainer = Trainer(cfg)
    trainer.train()


if __name__ == "__main__":
    main()
