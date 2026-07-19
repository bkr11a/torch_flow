#!/usr/bin/env python3
from __future__ import annotations
import argparse, logging, os, sys
from typing import Any, Dict, Optional
from dotenv import load_dotenv
from omegaconf import DictConfig, OmegaConf, open_dict

RESERVED_STAGE_KEYS = {
    "id", "description", "enabled", "checkpoint_from_previous",
    "checkpoint_tag", "optional",
}

def parse_args():
    p = argparse.ArgumentParser(description="Run staged HQSFlow curriculum")
    p.add_argument("--config", "-c", required=True)
    p.add_argument(
        "--override", "-O", default=None,
        help="Optional model/config overlay merged on top of --config.",
    )
    p.add_argument("--curriculum", required=True)
    p.add_argument("--start-stage", default=None)
    p.add_argument("--stop-after-stage", default=None)
    p.add_argument("--run-optional", action="store_true")
    p.add_argument(
        "--base-run-name",
        default=None,
        help=(
            "Optional grouping name for all curriculum stages. "
            "Example: --base-run-name hqs_flow_curriculum_test"
        ),
    )
    p.add_argument("overrides", nargs="*")
    return p.parse_args()

def pop_global_run_name(cli_cfg: Optional[DictConfig]) -> Optional[str]:
    """
    Treat a CLI override like:

        run_name=hqs_flow_curriculum_test

    as a curriculum base/group name, not as a replacement for every stage
    run_name.

    Without this, all stages write into the same TensorBoard/checkpoint path.
    """
    if cli_cfg is None:
        return None

    if "run_name" not in cli_cfg:
        return None

    run_name = str(cli_cfg.run_name)

    with open_dict(cli_cfg):
        del cli_cfg["run_name"]

    return run_name

def setup_logging(log_dir: str, run_name: str) -> None:
    os.makedirs(log_dir, exist_ok=True)

    # run_name may now be nested, e.g.
    #   hqs_flow_curriculum_test/curr_u01_chairs_baseline
    log_file = os.path.join(log_dir, f"{run_name}.log")
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file),
        ],
        force=True,
    )

def stage_to_override(stage: DictConfig) -> DictConfig:
    d = OmegaConf.to_container(stage, resolve=True)
    assert isinstance(d, dict)
    return OmegaConf.create({k: v for k, v in d.items() if k not in RESERVED_STAGE_KEYS})

def checkpoint_path(cfg: DictConfig, tag: str) -> str:
    return os.path.join(cfg.get("checkpoint_dir", "checkpoints"), cfg.get("run_name", "hqs_flow"), f"{tag}.pth")

def pick_checkpoint(cfg: DictConfig, tag: str = "best") -> Optional[str]:
    p = checkpoint_path(cfg, tag)
    if os.path.isfile(p):
        return p
    p = checkpoint_path(cfg, "last")
    if os.path.isfile(p):
        return p
    return None

def main():
    load_dotenv(override=False)
    args = parse_args()
    base_cfg = OmegaConf.load(args.config)
    model_override_cfg = (
        OmegaConf.load(args.override) if args.override else None
    )
    curr_cfg = OmegaConf.load(args.curriculum)
    curr = curr_cfg.get("curriculum", None)
    if curr is None or not curr.get("enabled", False):
        raise RuntimeError("Curriculum YAML must contain curriculum.enabled: true")
    stages = curr.get("stages", [])
    if not stages:
        raise RuntimeError("Curriculum YAML has no stages")

    global_cli_cfg = OmegaConf.from_dotlist(args.overrides) if args.overrides else None
    
    # Backward-compatible convenience:
    # If the user passes run_name=foo on the CLI, treat it as a curriculum
    # grouping name, not as a stage run_name override.
    cli_base_run_name = pop_global_run_name(global_cli_cfg)
    
    from engine import Trainer

    previous_checkpoint = None
    started = args.start_stage is None

    for idx, stage in enumerate(stages):
        stage_id = str(stage.get("id", f"stage_{idx:02d}"))
        if not started:
            if stage_id == args.start_stage:
                started = True
            else:
                continue
        if not stage.get("enabled", True):
            print(f"Skipping disabled stage: {stage_id}")
            continue
        if stage.get("optional", False) and not args.run_optional:
            print(f"Skipping optional stage: {stage_id}")
            continue

        stage_run_name = str(stage.get("run_name", stage_id))

        cfg = OmegaConf.merge(base_cfg, stage_to_override(stage))
        # A named architecture/config overlay is an explicit global switch and
        # therefore takes precedence over legacy per-stage model/loss fields.
        if model_override_cfg is not None:
            cfg = OmegaConf.merge(cfg, model_override_cfg)
        if global_cli_cfg is not None:
            cfg = OmegaConf.merge(cfg, global_cli_cfg)

        base_run_name = (
            args.base_run_name
            or cli_base_run_name
            or curr.get("base_run_name", None)
        )

        if base_run_name:
            with open_dict(cfg):
                # This gives nested TensorBoard/checkpoint dirs:
                #
                #   logs/<base_run_name>/<stage_run_name>
                #   checkpoints/<base_run_name>/<stage_run_name>
                #
                # because Trainer writes to:
                #
                #   os.path.join(log_dir, run_name)
                #   os.path.join(checkpoint_dir, run_name)
                cfg.run_name = f"{base_run_name}/{stage_run_name}"

                if not hasattr(cfg, "mlflow") or cfg.mlflow is None:
                    cfg.mlflow = {}

                if not hasattr(cfg.mlflow, "tags") or cfg.mlflow.tags is None:
                    cfg.mlflow.tags = {}

                cfg.mlflow.tags["curriculum_base_run_name"] = str(base_run_name)
                cfg.mlflow.tags["curriculum_stage_run_name"] = str(stage_run_name)

        if bool(stage.get("checkpoint_from_previous", idx > 0)) and previous_checkpoint:
            with open_dict(cfg):
                cfg.training.checkpoint = previous_checkpoint
                cfg.training.resume_mode = "weights_only"

        with open_dict(cfg):
            cfg.curriculum = {
                "enabled": True,
                "stage_id": stage_id,
                "stage_index": idx,
                "curriculum_path": os.path.abspath(args.curriculum),
                "base_config_path": os.path.abspath(args.config),
            }
            cfg.launch = {
                "config_path": os.path.abspath(args.config),
                "override_path": (
                    os.path.abspath(args.override) if args.override else None
                ),
                "curriculum_path": os.path.abspath(args.curriculum),
                "cli_overrides": list(args.overrides),
            }

        setup_logging(cfg.get("log_dir", "logs"), cfg.get("run_name", stage_id))
        logger = logging.getLogger(__name__)
        logger.info("Curriculum stage %s/%s: %s", idx + 1, len(stages), stage_id)
        logger.info("Config:\n%s", OmegaConf.to_yaml(cfg))

        Trainer(cfg).train()

        previous_checkpoint = pick_checkpoint(cfg, str(stage.get("checkpoint_tag", "best")))
        if previous_checkpoint is None:
            raise RuntimeError(f"Stage {stage_id} finished but no best/last checkpoint was found")
        logger.info("Stage %s checkpoint for next stage: %s", stage_id, previous_checkpoint)

        if args.stop_after_stage is not None and stage_id == args.stop_after_stage:
            break

if __name__ == "__main__":
    main()
