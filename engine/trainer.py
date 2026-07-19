"""Training engine for HQSFlow.

Handles:
  - Auto device detection (CUDA → MPS → CPU)
  - Mixed-precision training (torch.amp)
  - OneCycleLR scheduler
  - tqdm progress bars: outer epoch bar + inner batch bar (leave=False)
  - Best-model tracking (saves whenever val EPE improves)
  - Final save of both best and last checkpoints with full training history
  - TensorBoard + optional W&B logging
  - Graceful validation when val_data is absent
"""
from __future__ import annotations

import json
import os
import shutil
import time
import logging
import math
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from omegaconf import OmegaConf

try:
    from torch.utils.tensorboard import SummaryWriter
    _TB_AVAILABLE = True
except ImportError:
    _TB_AVAILABLE = False

try:
    import mlflow
    import mlflow.pytorch
    from mlflow.tracking import MlflowClient
    _MLFLOW_AVAILABLE = True
except Exception:
    _MLFLOW_AVAILABLE = False

from models import build_model
from losses import HQSFlowLoss
from data import build_dataset, build_dataloader
from utils import compute_metrics, aggregate_metrics, flow_to_color, InputPadder
from hqs_pytorch.customML.customModels.HQSFlowModelTFPort import HQSFlowModelTFPort

logger = logging.getLogger(__name__)


def _flatten_dict(d, parent_key: str = "", sep: str = ".") -> Dict[str, object]:
    items: Dict[str, object] = {}
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else str(k)
        if isinstance(v, dict):
            items.update(_flatten_dict(v, new_key, sep=sep))
        else:
            items[new_key] = v
    return items


def _sanitize_metric_dict(d: Dict[str, object]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for k, v in d.items():
        try:
            fv = float(v)
            if math.isnan(fv) or math.isinf(fv):
                continue
            out[k] = fv
        except Exception:
            continue
    return out


# ---------------------------------------------------------------------------
# Running mean accumulator
# ---------------------------------------------------------------------------

class RunningMean:
    """Lightweight online mean — tracks sum and count, never stores history."""
    def __init__(self) -> None:
        self._sums: Dict[str, float] = {}
        self._counts: Dict[str, int] = {}

    def update(self, d: Dict[str, float]) -> None:
        for k, v in d.items():
            if isinstance(v, (float, np.floating)) and math.isnan(v):
                continue
            self._sums[k]   = self._sums.get(k, 0.0) + v
            self._counts[k] = self._counts.get(k, 0)  + 1

    def mean(self) -> Dict[str, float]:
        return {k: self._sums[k] / self._counts[k]
                for k in self._sums if self._counts[k] > 0}

    def reset(self) -> None:
        self._sums.clear()
        self._counts.clear()


# ---------------------------------------------------------------------------
# Device helper
# ---------------------------------------------------------------------------

def get_device(preferred: Optional[str] = None) -> torch.device:
    """Auto-detect the best available device, or honour *preferred*.

    Note: Some ops (e.g. grid_sampler backward) are not yet supported on MPS.
    Set PYTORCH_ENABLE_MPS_FALLBACK=1 in your environment to fall back to CPU
    for those ops, or train on CPU with --device cpu.
    """
    if preferred:
        return torch.device(preferred)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def amp_enabled(device: torch.device) -> bool:
    """AMP is only useful on CUDA (MPS has native float16 issues on some ops)."""
    return device.type == "cuda"


# ---------------------------------------------------------------------------
# Training history record
# ---------------------------------------------------------------------------

@dataclass
class TrainingHistory:
    """Persisted alongside model weights for full reproducibility."""
    train_loss:  List[Tuple[int, float]] = field(default_factory=list)
    train_epe:   List[Tuple[int, float]] = field(default_factory=list)
    val_metrics: List[Dict]              = field(default_factory=list)
    best_epe:    float                   = math.inf
    best_step:   int                     = 0
    run_name:    str                     = ""
    total_steps: int                     = 0

    def to_json(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def from_json(cls, path: str) -> "TrainingHistory":
        with open(path) as f:
            return cls(**json.load(f))


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def save_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: optim.Optimizer,
    scaler,
    scheduler,
    step: int,
    history: "TrainingHistory",
    cfg,
    tag: str = "",
) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(
        {
            "step":      step,
            "tag":       tag,
            "model":     model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler":    scaler.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "history":   asdict(history),
            "cfg":       cfg,
        },
        path,
    )
    logger.info(f"Checkpoint [{tag}] saved → {path}")


def load_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: Optional[optim.Optimizer] = None,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
    scheduler=None,
    load_optimizer: bool = True,
    load_scaler: bool = True,
    load_scheduler: bool = True,
    strict: bool = True,
    device: Optional[torch.device] = None,
) -> Tuple[int, Optional["TrainingHistory"]]:
    ckpt = torch.load(path, map_location=device or "cpu", weights_only=False)
    missing, unexpected = model.load_state_dict(ckpt["model"], strict=strict)
    if missing:
        logger.warning(f"Missing keys ({len(missing)}): {missing[:5]}...")
    if unexpected:
        logger.warning(f"Unexpected keys ({len(unexpected)}): {unexpected[:5]}...")
    step = ckpt.get("step", 0)
    if load_optimizer and optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    if load_scaler and scaler is not None and "scaler" in ckpt:
        scaler.load_state_dict(ckpt["scaler"])
    if load_scheduler and scheduler is not None and ckpt.get("scheduler") is not None:
        scheduler.load_state_dict(ckpt["scheduler"])
    history = None
    if "history" in ckpt:
        try:
            history = TrainingHistory(**ckpt["history"])
        except Exception:
            pass
    logger.info(f"Loaded checkpoint [{ckpt.get('tag', '')}] from {path} at step {step}")
    return step, history


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class Trainer:
    """
    Full training loop for HQSFlow.

    Features
    --------
    - Auto device detection (CUDA → MPS → CPU)
    - Mixed-precision (AMP) on CUDA; falls back to fp32 on MPS/CPU
    - OneCycleLR with configurable warmup
    - tqdm epoch bar (outer) + batch bar (inner, leave=False)
    - Best-model saving whenever val EPE improves
    - Final save of both ``best.pth`` and ``last.pth`` with full training
      history (JSON sidecar + embedded in checkpoint)
    - TensorBoard + optional W&B logging
    """

    def __init__(self, cfg) -> None:
        self.cfg = cfg

        # ── Device ──────────────────────────────────────────────────────────
        self.device = get_device(cfg.get("device", None))
        logger.info(f"Device: {self.device}  (AMP={'enabled' if amp_enabled(self.device) else 'disabled'})")

        # ── Model ────────────────────────────────────────────────────────────
        self.model = build_model(cfg).to(self.device)
        counts = self.model.param_count()
        logger.info(
            f"Parameters: total={counts['total']:,}  "
            f"encoder={counts['feature_encoder']:,}  "
            f"stages={counts['stages']:,}"
        )

        # ── Loss ─────────────────────────────────────────────────────────────
        self.criterion = HQSFlowLoss(cfg.loss).to(self.device)

        # ── Optimiser ────────────────────────────────────────────────────────
        # Include learnable loss parameters (e.g. adaptive stage weights from
        # SequenceLoss when stage_weight_mode="learnable").
        _opt_params = list(self.model.parameters()) + list(self.criterion.parameters())
        self.optimizer = optim.AdamW(
            _opt_params,
            lr=cfg.training.lr,
            weight_decay=cfg.training.get("weight_decay", 1e-4),
            eps=1e-8,
        )

        # ── OneCycleLR ───────────────────────────────────────────────────────
        train_steps = cfg.training.num_steps
        self.scheduler = optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=cfg.training.lr,
            total_steps=train_steps + 1,
            pct_start=cfg.training.get("warmup_pct", 0.05),
            cycle_momentum=False,
            anneal_strategy="cos",
            div_factor=cfg.training.get("div_factor", 25.0),
            final_div_factor=cfg.training.get("final_div_factor", 1e4),
        )

        # ── Mixed precision ───────────────────────────────────────────────────
        self._use_amp = amp_enabled(self.device)
        self.scaler = torch.amp.GradScaler("cuda", enabled=self._use_amp) if self._use_amp else torch.amp.GradScaler("cpu", enabled=False)

        # ── Data ─────────────────────────────────────────────────────────────
        train_data = build_dataset(cfg.data, split="train")
        self.train_loader = build_dataloader(train_data, cfg.data, split="train")
        logger.info(f"Training samples: {len(train_data):,}")

        self._has_val = hasattr(cfg, "val_data") and cfg.val_data is not None
        if self._has_val:
            val_data = build_dataset(cfg.val_data, split="val")
            self.val_loader = build_dataloader(val_data, cfg.val_data, split="val")
            logger.info(f"Validation samples: {len(val_data):,}")

        # ── Logging ───────────────────────────────────────────────────────────
        self.run_name = cfg.get("run_name", "hqs_flow")
        self.log_dir  = cfg.get("log_dir", "logs")
        self.ckpt_dir = cfg.get("checkpoint_dir", "checkpoints")
        self.run_dir = os.path.join(self.ckpt_dir, self.run_name)
        self.config_dir = os.path.join(self.run_dir, "config")
        self._archive_run_config()

        if _TB_AVAILABLE:
            self.writer: Optional[SummaryWriter] = SummaryWriter(
                os.path.join(self.log_dir, self.run_name)
            )
        else:
            self.writer = None

        self.use_wandb = cfg.get("use_wandb", False)
        if self.use_wandb:
            import wandb
            wandb.init(
                project=cfg.get("wandb_project", "hqs_flow"),
                name=self.run_name,
                config=dict(cfg),
            )

        # ── MLflow ───────────────────────────────────────────────────────────
        self.use_mlflow = bool(cfg.get("mlflow", {}).get("enabled", False))
        self.mlflow_run = None
        self.mlflow_run_id: Optional[str] = None
        self.mlflow_client: Optional["MlflowClient"] = None
        self.best_model_uri: Optional[str] = None
        self._warned_missing_occ_masks = False
        if self.use_mlflow:
            self._init_mlflow()

        # ── State ─────────────────────────────────────────────────────────────
        self.global_step = 0
        self._global_matcher_warmup_steps = int(
            cfg.training.get("global_matcher_warmup_steps", 0)
        )
        self._freeze_backbone_during_global_warmup = bool(
            cfg.training.get("freeze_backbone_during_global_warmup", False)
        )
        self._global_warmup_active: Optional[bool] = None
        self.history = TrainingHistory(
            run_name=self.run_name,
            total_steps=train_steps,
        )

        if cfg.training.get("checkpoint"):
            resume_mode = str(cfg.training.get("resume_mode", "full")).lower()
            if resume_mode not in {"full", "weights_only"}:
                raise ValueError(
                    f"Unknown training.resume_mode={resume_mode!r}. "
                    "Expected 'full' or 'weights_only'."
                )

            if resume_mode == "weights_only":
                load_checkpoint(
                    cfg.training.checkpoint,
                    self.model,
                    optimizer=self.optimizer,
                    scaler=self.scaler,
                    scheduler=self.scheduler,
                    load_optimizer=False,
                    load_scaler=False,
                    load_scheduler=False,
                    strict=cfg.training.get("strict", True),
                    device=self.device,
                )
                self.global_step = 0
                logger.info(
                    "Warm-started model weights only from checkpoint; "
                    "optimizer/scheduler/step/history were reset."
                )
            else:
                step, loaded_history = load_checkpoint(
                    cfg.training.checkpoint,
                    self.model,
                    self.optimizer,
                    self.scaler,
                    self.scheduler,
                    strict=cfg.training.get("strict", True),
                    device=self.device,
                )
                self.global_step = step
                if loaded_history is not None:
                    self.history = loaded_history
                    logger.info(f"Resumed history (best EPE={self.history.best_epe:.4f} "
                                 f"@ step {self.history.best_step})")

    # ─────────────────────────────────────────────────────────────────────── #
    # Main training loop
    # ─────────────────────────────────────────────────────────────────────── #

    def train(self) -> None:
        cfg        = self.cfg
        num_steps  = cfg.training.num_steps
        log_every  = cfg.training.get("log_every", 100)
        val_every  = cfg.training.get("val_every", 5000)
        save_every = cfg.training.get("save_every", 10000)
        steps_left = num_steps - self.global_step

        try:
            if steps_left <= 0:
                logger.info("Already at target step count — nothing to do.")
                self._mlflow_end_run("FINISHED")
                return

            self.model.train()
            loader_iter     = iter(self._infinite_loader(self.train_loader))
            steps_per_epoch = len(self.train_loader)

            # ── Epoch-level progress bar ─────────────────────────────────────
            total_epochs = math.ceil(steps_left / steps_per_epoch)
            epoch_bar = tqdm(
                total=total_epochs,
                desc="Epochs",
                unit="ep",
                position=0,
                leave=True,
                ncols=300,
            )

            epoch_idx    = 0
            steps_in_ep  = 0
            batch_bar: Optional[tqdm] = None
            t_epoch      = time.time()
            running      = RunningMean()

            while self.global_step < num_steps:
                validated_this_step = False
                # Create batch bar at the start of each epoch
                if steps_in_ep == 0:
                    batch_bar = tqdm(
                        total=min(steps_per_epoch, num_steps - self.global_step),
                        desc=f"  Epoch {epoch_idx + 1}",
                        unit="batch",
                        position=1,
                        leave=False,
                        ncols=300,
                    )
                    running.reset()
                    t_epoch = time.time()

                batch      = next(loader_iter)
                loss_dict  = self._train_step(batch)

                running.update(loss_dict)

                self.global_step += 1
                steps_in_ep += 1

                # ── Update batch bar with running means ──────────────────────
                lr = self.optimizer.param_groups[0]["lr"]
                if batch_bar is not None:
                    rm = running.mean()

                    def _fmt(k, fmt=".3f"):
                        v = rm.get(k, float("nan"))
                        return "nan" if math.isnan(v) else format(v, fmt)

                    def _fmt_cur(k, fmt=".3f"):
                        v = loss_dict.get(k, float("nan"))
                        return "nan" if math.isnan(v) else format(v, fmt)

                    postfix: Dict[str, str] = {
                        "curr_loss": f"{loss_dict['loss']:.4f}",
                        "curr_epe_all": _fmt_cur("epe_all"),
                        "loss":    f"{rm['loss']:.4f}",
                        "epe_m":   _fmt("epe_matched"),
                        "epe_u":   _fmt("epe_unmatched"),
                        "epe_all": _fmt("epe_all"),
                        "f1":      _fmt("f1"),
                        "s0_10":   _fmt("s0_10"),
                        "s10_40":  _fmt("s10_40"),
                        "s40+":    _fmt("s40_plus"),
                        "d0":      _fmt("d0"),
                        "d0_10":   _fmt("d0_10"),
                        "d10_60":  _fmt("d10_60"),
                        "d60_140": _fmt("d60_140"),
                        "d140+":   _fmt("d140_plus"),
                        "hf_r":    _fmt("hf_recovery"),
                        "hf_a":    _fmt("hf_alignment"),
                        "b_epe":   _fmt("boundary_epe"),
                        "lr":      f"{lr:.2e}",
                    }
                    postfix["smooth"] = _fmt("smooth", ".4f")
                    postfix["photo"] = _fmt("photo", ".4f")
                    postfix["ofce"] = _fmt("ofce", ".4f")
                    batch_bar.set_postfix(**postfix)
                    batch_bar.update(1)

                # ── Log scalars ──────────────────────────────────────────────
                if self.global_step % log_every == 0:
                    self.history.train_loss.append((self.global_step, loss_dict["loss"]))
                    self.history.train_epe.append((self.global_step, loss_dict["epe"]))
                    self._log_scalars(
                        {k: v for k, v in loss_dict.items()},
                        prefix="train",
                    )
                    self._log_scalars({"lr": lr}, prefix="train")

                # ── Validation ───────────────────────────────────────────────
                if self.global_step % val_every == 0:
                    val_metrics = self._validate()
                    if val_metrics:
                        epe = val_metrics.get("epe", math.inf)
                        is_best = epe < self.history.best_epe
                        if is_best:
                            self.history.best_epe  = epe
                            self.history.best_step = self.global_step
                            self._save("best")
                            self._mlflow_log_best_model_and_register()

                        # Surface val metrics in the epoch bar immediately
                        def _fmtv(k, fmt=".3f"):
                            v = val_metrics.get(k, float("nan"))
                            return "nan" if math.isnan(v) else format(v, fmt)

                        tag = " ★" if is_best else ""
                        logger.info(
                            f"✓ Val EPE={epe:.4f}  F1={_fmtv('f1')}%"
                            f"  epe_m={_fmtv('epe_matched')}"
                            f"  epe_u={_fmtv('epe_unmatched')}"
                            f"  epe_all={_fmtv('epe_all')}"
                            f"  s0_10={_fmtv('s0_10')}  s10_40={_fmtv('s10_40')}"
                            f"  s40+={_fmtv('s40_plus')}"
                            f"  d0={_fmtv('d0')}  d0_10={_fmtv('d0_10')}"
                            f"  d10_60={_fmtv('d10_60')}"
                            f"  d60_140={_fmtv('d60_140')}"
                            f"  d140+={_fmtv('d140_plus')}{tag}"
                        )
                        epoch_bar.set_postfix(
                            val_epe=f"{epe:.4f}",
                            best=f"{self.history.best_epe:.4f}",
                        )
                        validated_this_step = True

                # ── Periodic checkpoint ──────────────────────────────────────
                if self.global_step % save_every == 0:
                    self._save(f"step_{self.global_step:07d}")

                # ── Epoch boundary ───────────────────────────────────────────
                if steps_in_ep >= steps_per_epoch:
                    if batch_bar is not None:
                        batch_bar.close()
                        batch_bar = None
                    elapsed = time.time() - t_epoch
                    rm = running.mean()

                    def _fmt(k, fmt=".3f"):
                        v = rm.get(k, float("nan"))
                        return "nan" if math.isnan(v) else format(v, fmt)

                    epoch_bar.set_postfix(
                        loss=f"{rm['loss']:.4f}",
                        epe_m=_fmt("epe_matched"),
                        epe_u=_fmt("epe_unmatched"),
                        epe_all=_fmt("epe_all"),
                        f1=_fmt("f1"),
                        s0_10=_fmt("s0_10"),
                        s10_40=_fmt("s10_40"),
                        **{"s40+": _fmt("s40_plus")},
                        d0=_fmt("d0"),
                        d0_10=_fmt("d0_10"),
                        d10_60=_fmt("d10_60"),
                        d60_140=_fmt("d60_140"),
                        **{"d140+": _fmt("d140_plus")},
                        smooth=_fmt("smooth", ".4f"),
                        photo=_fmt("photo", ".4f"),
                        ofce=_fmt("ofce", ".4f"),
                        best_epe=f"{self.history.best_epe:.4f}",
                        t=f"{elapsed:.0f}s",
                    )
                    epoch_bar.update(1)

                    # Always run validation at epoch end (once per step maximum).
                    if self._has_val and not validated_this_step:
                        val_metrics = self._validate()
                        if val_metrics:
                            epe = val_metrics.get("epe", math.inf)
                            is_best = epe < self.history.best_epe
                            if is_best:
                                self.history.best_epe = epe
                                self.history.best_step = self.global_step
                                self._save("best")
                                self._mlflow_log_best_model_and_register()

                            def _fmtv(k, fmt=".3f"):
                                v = val_metrics.get(k, float("nan"))
                                return "nan" if math.isnan(v) else format(v, fmt)

                            tag = " ★" if is_best else ""
                            logger.info(
                                f"✓ [Epoch-end] Val EPE={epe:.4f}  F1={_fmtv('f1')}%"
                                f"  epe_m={_fmtv('epe_matched')}"
                                f"  epe_u={_fmtv('epe_unmatched')}"
                                f"  epe_all={_fmtv('epe_all')}"
                                f"  s0_10={_fmtv('s0_10')}  s10_40={_fmtv('s10_40')}"
                                f"  s40+={_fmtv('s40_plus')}"
                                f"  d0={_fmtv('d0')}  d0_10={_fmtv('d0_10')}"
                                f"  d10_60={_fmtv('d10_60')}"
                                f"  d60_140={_fmtv('d60_140')}"
                                f"  d140+={_fmtv('d140_plus')}{tag}"
                            )
                            epoch_bar.set_postfix(
                                val_epe=f"{epe:.4f}",
                                best=f"{self.history.best_epe:.4f}",
                            )

                    epoch_idx += 1
                    steps_in_ep = 0

            # Close any open batch bar at end of training
            if batch_bar is not None:
                batch_bar.close()
            epoch_bar.close()

            # ── Final saves ──────────────────────────────────────────────────
            self._save("last")
            history_path = os.path.join(self.ckpt_dir, self.run_name, "history.json")
            self.history.to_json(history_path)
            logger.info(f"Training history → {history_path}")
            logger.info(
                f"Training complete. "
                f"Best EPE={self.history.best_epe:.4f} @ step {self.history.best_step}."
            )
            self._log_final_mlflow_artifacts(history_path)
            self._mlflow_end_run("FINISHED")
        except Exception:
            self._mlflow_end_run("FAILED")
            raise
        finally:
            if self.writer:
                self.writer.close()
            if self.use_wandb:
                import wandb
                wandb.finish()

    # ────────────────────────────────────────────────────────────────────── #
    # Build Oracle masks for debugging target image leakage problem
    # ────────────────────────────────────────────────────────────────────── #

    def build_oracle_source_target_masks(
            self,
            flow_gt_xy: torch.Tensor,
            valid: Optional[torch.Tensor] = None,
            occlusion: Optional[torch.Tensor] = None,
            invalid: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Build source_valid and target_valid masks based on ground truth flow, for debugging leakage of target image information into the model.

        flow_gt_xy: (B, 2, H, W) ground truth flow in pixel units with (x, y) order
        valid: (B, H, W) or (B, 1, H, W) optional mask of valid flow pixels (1 for valid, 0 for invalid)
        occlusion: (B, H, W) or (B, 1, H, W) optional mask of occluded pixels (1 for occluded/unmatachable from frame 1 to frame 2, 0 for non-occluded)
        invalid: (B, H, W) or (B, 1, H, W) optional mask of invalid pixels (1 for invalid, 0 for valid)

        Returns:
            source_valid: (B, 1, H, W) boolean mask of pixels in source image that have valid, non-occluded correspondences in target image
            target_valid: (B, 1, H, W) boolean mask of pixels in target image that are valid targets for the model to match to (i.e. not occluded and not invalid)
        """
        if flow_gt_xy.ndim != 4 or flow_gt_xy.shape[1] != 2:
            raise ValueError(f"Expected flow_gt_xy shape (B, 2, H, W), got {flow_gt_xy.shape}")

        device = flow_gt_xy.device
        dtype = flow_gt_xy.dtype
        b, _, h, w = flow_gt_xy.shape

        def prep_mask(mask, default_value):
            if mask is None:
                return torch.full((b, 1, h, w), default_value, dtype=dtype, device=device)
            if mask.ndim == 3:
                mask = mask.unsqueeze(1)
            mask = mask.to(dtype=dtype, device=device)
            if mask.max() > 1.5:  # Assume binary mask is in {0, 255} format
                mask = mask / 255.0
            if mask.shape[-2:] != (h, w):
                mask = F.interpolate(mask, size=(h, w), mode="nearest")
            if mask.shape[1] != 1:
                mask = mask.mean(dim=1, keepdim=True)
            return mask.clamp(0.0, 1.0)

        valid = prep_mask(valid, default_value=1.0)
        occlusion = prep_mask(occlusion, default_value=0.0)
        invalid = prep_mask(invalid, default_value=0.0)

        # Convert [dx, dy] flow to our internal [dy, dx]
        flow_gt_yx = torch.stack([flow_gt_xy[:, 1], flow_gt_xy[:, 0]], dim=1)

        bounds = HQSFlowModelTFPort._valid_warp_mask(flow_gt_yx).to(device=device, dtype=dtype)

        # Source-valid mask in frame-1 coordinates
        source_valid = valid * (1.0 - occlusion) * (1.0 - invalid) * bounds
        source_valid = source_valid.clamp(0.0, 1.0)

        # Target-valid mask in frame-2 coordinates
        # Pixels in image2 not reached by any valid source pixel are newly visible
        target_valid = HQSFlowModelTFPort.forward_splat(source_valid, flow_gt_yx, normalize=False).clamp(0.0, 1.0)

        return source_valid, target_valid

    # ─────────────────────────────────────────────────────────────────────── #
    # Single training step
    # ─────────────────────────────────────────────────────────────────────── #

    def _configure_global_matcher_warmup(self) -> None:
        """Optionally pretrain the global proposal before joint optimisation."""
        active = (
            self._freeze_backbone_during_global_warmup
            and self.global_step < self._global_matcher_warmup_steps
        )
        if active == self._global_warmup_active:
            return
        for name, parameter in self.model.named_parameters():
            parameter.requires_grad_(not active or name.startswith("pgma."))
        self._global_warmup_active = active
        if active:
            logger.info(
                "Global-matcher warmup active: only model.pgma parameters "
                f"train until step {self._global_matcher_warmup_steps}."
            )
        else:
            logger.info("Joint model optimisation active.")

    def _train_step(self, batch: Dict) -> Dict[str, float]:
        self._configure_global_matcher_warmup()
        self.optimizer.zero_grad(set_to_none=True)

        img1  = batch["image1"].to(self.device, non_blocking=True)
        img2  = batch["image2"].to(self.device, non_blocking=True)
        flow  = batch["flow"].to(self.device, non_blocking=True)
        valid = batch["valid"].to(self.device, non_blocking=True)
        occ_batch = batch.get("occlusion")
        inv_batch = batch.get("invalid")
        synthetic_occ_batch = batch.get("synthetic_occlusion")
        if occ_batch is not None:
            occ_batch = occ_batch.to(self.device, non_blocking=True)
        if inv_batch is not None:
            inv_batch = inv_batch.to(self.device, non_blocking=True)
        if synthetic_occ_batch is not None:
            synthetic_occ_batch = synthetic_occ_batch.to(
                self.device, non_blocking=True
            )

        with torch.autocast(device_type=self.device.type, enabled=self._use_amp):
            # Ground-truth and augmentation masks are loss targets only.
            out = self.model(img1, img2)
            if hasattr(self.criterion, "set_step"):
                self.criterion.set_step(self.global_step)
            loss_dict = self.criterion(
                out["flow_preds"],
                flow,
                valid,
                img1,
                img2,
                model_outputs=out,
                occlusion=occ_batch,
                invalid=inv_batch,
                synthetic_occlusion=synthetic_occ_batch,
            )

        self.scaler.scale(loss_dict["loss"]).backward()
        self.scaler.unscale_(self.optimizer)
        nn.utils.clip_grad_norm_(
            self.model.parameters(),
            self.cfg.training.get("grad_clip", 1.0),
        )

        # With AMP, GradScaler may skip optimizer.step() on overflow. In that
        # case scheduler.step() must also be skipped to keep call order valid.
        scale_before = self.scaler.get_scale()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        scale_after = self.scaler.get_scale()
        if scale_after >= scale_before:
            self.scheduler.step()

        scalar_dict = {k: v.item() if isinstance(v, torch.Tensor) else v
                       for k, v in loss_dict.items()}

        # ── Per-batch speed + F1 metrics on final prediction ────────────────
        with torch.no_grad():
            pred_final = out["flow_preds"][-1].detach()
            # Average metrics over batch items
            batch_metrics: Dict[str, list] = {
                "epe": [], "epe_matched": [], "epe_unmatched": [], "epe_all": [],
                "f1": [], "s0_10": [], "s10_40": [], "s40_plus": [],
                "d0": [], "d0_10": [], "d10_60": [], "d60_140": [], "d140_plus": [],
                "hf_recovery": [], "hf_alignment": [], "boundary_epe": [],
            }
            for b in range(pred_final.shape[0]):
                occ = None
                inv = None
                if occ_batch is not None:
                    occ = occ_batch[b].to(self.device)
                if inv_batch is not None:
                    inv = inv_batch[b].to(self.device)
                m = compute_metrics(
                    pred_final[b], flow[b], valid[b], occ_mask=occ, invalid_mask=inv
                )
                for k in batch_metrics:
                    v = m.get(k, float("nan"))
                    if not math.isnan(v):
                        batch_metrics[k].append(v)
            for k, vals in batch_metrics.items():
                scalar_dict[k] = float(np.mean(vals)) if vals else float("nan")

        return scalar_dict

    # ─────────────────────────────────────────────────────────────────────── #
    # Validation
    # ─────────────────────────────────────────────────────────────────────── #

    def _get_validation_crop_size(self):
        """
        Return deterministic validation crop size.

        For training-time checkpoint selection, we do not want to run RAFT-style
        all-pairs correlation on native 1080p/2K validation frames. This crop is
        for validation during training only, not for final benchmark reporting.

        Uses:
            val_data.eval_crop_size if present
            else val_data.crop_size if present

        Disable with:
            val_data.eval_center_crop: false
        """
        if not hasattr(self.cfg, "val_data") or self.cfg.val_data is None:
            return None

        if self.cfg.val_data.get("eval_center_crop", True) is False:
            return None

        crop_size = self.cfg.val_data.get(
            "eval_crop_size",
            self.cfg.val_data.get("crop_size", None),
        )

        if crop_size is None:
            return None

        if len(crop_size) != 2:
            raise ValueError(f"Expected validation crop_size [H, W], got {crop_size}")

        return int(crop_size[0]), int(crop_size[1])

    @staticmethod
    def _crop_mask_like(mask, y0: int, x0: int, crop_h: int, crop_w: int):
        if mask is None:
            return None

        if mask.ndim == 3:
            return mask[:, y0:y0 + crop_h, x0:x0 + crop_w]

        if mask.ndim == 4:
            return mask[:, :, y0:y0 + crop_h, x0:x0 + crop_w]

        raise ValueError(f"Expected mask shape [B,H,W] or [B,1,H,W], got {mask.shape}")

    def _maybe_center_crop_validation_batch(
        self,
        img1,
        img2,
        flow,
        valid,
        occ_batch,
        inv_batch,
    ):
        crop_size = self._get_validation_crop_size()

        if crop_size is None:
            return img1, img2, flow, valid, occ_batch, inv_batch

        crop_h, crop_w = crop_size
        _, _, h, w = img1.shape

        crop_h = min(crop_h, h)
        crop_w = min(crop_w, w)

        if crop_h == h and crop_w == w:
            return img1, img2, flow, valid, occ_batch, inv_batch

        y0 = (h - crop_h) // 2
        x0 = (w - crop_w) // 2

        img1 = img1[:, :, y0:y0 + crop_h, x0:x0 + crop_w]
        img2 = img2[:, :, y0:y0 + crop_h, x0:x0 + crop_w]
        flow = flow[:, :, y0:y0 + crop_h, x0:x0 + crop_w]

        if valid.ndim == 3:
            valid = valid[:, y0:y0 + crop_h, x0:x0 + crop_w]
        elif valid.ndim == 4:
            valid = valid[:, :, y0:y0 + crop_h, x0:x0 + crop_w]
        else:
            raise ValueError(f"Expected valid shape [B,H,W] or [B,1,H,W], got {valid.shape}")

        occ_batch = self._crop_mask_like(occ_batch, y0, x0, crop_h, crop_w)
        inv_batch = self._crop_mask_like(inv_batch, y0, x0, crop_h, crop_w)

        return img1, img2, flow, valid, occ_batch, inv_batch

    def _validate(self) -> Optional[Dict[str, float]]:
        if not self._has_val:
            return None

        self.model.eval()
        results: List[Dict] = []

        with torch.no_grad():
            for batch in tqdm(
                self.val_loader,
                desc="  Validating",
                unit="batch",
                position=1,
                leave=False,
                ncols=300,
            ):
                img1  = batch["image1"].to(self.device, non_blocking=True)
                img2  = batch["image2"].to(self.device, non_blocking=True)
                flow  = batch["flow"].to(self.device, non_blocking=True)
                valid = batch["valid"].to(self.device, non_blocking=True)
                # occlusion mask is optional — None for datasets without masks
                occ_batch = batch.get("occlusion")  # (B, H, W) tensor or None
                inv_batch = batch.get("invalid")

                if occ_batch is not None:
                    occ_batch = occ_batch.to(self.device, non_blocking=True)

                if inv_batch is not None:
                    inv_batch = inv_batch.to(self.device, non_blocking=True)

                # Training-time validation crop.
                # This prevents all-pairs correlation from exploding on native Spring/HD1K/KITTI
                # validation frames.
                img1, img2, flow, valid, occ_batch, inv_batch = self._maybe_center_crop_validation_batch(
                    img1=img1,
                    img2=img2,
                    flow=flow,
                    valid=valid,
                    occ_batch=occ_batch,
                    inv_batch=inv_batch,
                )

                if occ_batch is None and not self._warned_missing_occ_masks:
                    logger.warning(
                        "Validation batches do not include occlusion masks; "
                        "distance-to-occlusion metrics (d0/d0_10/d10_60/d60_140/d140_plus) "
                        "will be NaN or absent."
                    )
                    self._warned_missing_occ_masks = True

                padder = InputPadder(img1.shape, divisor=8)
                img1, img2 = padder.pad(img1, img2)

                with torch.autocast(device_type=self.device.type, enabled=self._use_amp):
                    out = self.model(img1, img2)
                pred = padder.unpad(out["flow_preds"][-1])

                for b in range(img1.shape[0]):
                    occ = None
                    inv = None
                    if occ_batch is not None:
                        occ = occ_batch[b].to(self.device)
                    if inv_batch is not None:
                        inv = inv_batch[b].to(self.device)
                    results.append(
                        compute_metrics(
                            pred[b], flow[b], valid[b], occ_mask=occ, invalid_mask=inv
                        )
                    )

        agg = aggregate_metrics(results)
        logger.info(
            f"  [Val step {self.global_step}] "
            + "  ".join(f"{k}={v:.4f}" for k, v in agg.items()
                        if not math.isnan(v))
        )
        self.history.val_metrics.append({"step": self.global_step, **agg})
        self._log_scalars(agg, prefix="val")
        self.model.train()
        return agg

    # ─────────────────────────────────────────────────────────────────────── #
    # Helpers
    # ─────────────────────────────────────────────────────────────────────── #

    def _save(self, tag: str) -> None:
        """Save a checkpoint under ``<ckpt_dir>/<run_name>/<tag>.pth``."""
        path = os.path.join(self.ckpt_dir, self.run_name, f"{tag}.pth")
        save_checkpoint(
            path,
            self.model, self.optimizer, self.scaler, self.scheduler,
            self.global_step, self.history, self.cfg, tag=tag,
        )
        self._mlflow_log_checkpoint(path, tag)

    def _archive_run_config(self) -> None:
        """Save launch/resolved configs alongside checkpoints for this run."""
        os.makedirs(self.config_dir, exist_ok=True)

        resolved_path = os.path.join(self.config_dir, "resolved_config.yaml")
        with open(resolved_path, "w") as f:
            f.write(OmegaConf.to_yaml(self.cfg, resolve=True))

        launch_cfg = self.cfg.get("launch", {}) if hasattr(self.cfg, "get") else {}
        launch_config_path = launch_cfg.get("config_path") if isinstance(launch_cfg, dict) else None
        launch_override_path = launch_cfg.get("override_path") if isinstance(launch_cfg, dict) else None
        launch_curriculum_path = launch_cfg.get("curriculum_path") if isinstance(launch_cfg, dict) else None
        launch_cli_overrides = launch_cfg.get("cli_overrides") if isinstance(launch_cfg, dict) else None

        if launch_config_path and os.path.isfile(launch_config_path):
            shutil.copy2(launch_config_path, os.path.join(self.config_dir, "launch_config.yaml"))
        if launch_override_path and os.path.isfile(launch_override_path):
            shutil.copy2(launch_override_path, os.path.join(self.config_dir, "override_config.yaml"))
        if launch_curriculum_path and os.path.isfile(launch_curriculum_path):
            shutil.copy2(
                launch_curriculum_path,
                os.path.join(self.config_dir, "curriculum_config.yaml"),
            )
        if launch_cli_overrides:
            cli_path = os.path.join(self.config_dir, "cli_overrides.txt")
            with open(cli_path, "w") as f:
                for item in launch_cli_overrides:
                    f.write(f"{item}\n")

    def _log_scalars(self, d: Dict, prefix: str) -> None:
        if self.writer:
            for k, v in d.items():
                self.writer.add_scalar(f"{prefix}/{k}", v, self.global_step)
        if self.use_wandb:
            import wandb
            wandb.log({f"{prefix}/{k}": v for k, v in d.items()},
                      step=self.global_step)
        if self.use_mlflow and self.mlflow_run_id:
            metrics = _sanitize_metric_dict({f"{prefix}/{k}": v for k, v in d.items()})
            if metrics:
                self._mlflow_safe(
                    f"log_metrics_{prefix}",
                    lambda: mlflow.log_metrics(metrics, step=self.global_step),
                )

    def _mlflow_safe(self, fn_name: str, fn):
        try:
            return fn()
        except Exception as exc:
            strict = bool(self.cfg.get("mlflow", {}).get("strict", False))
            logger.exception(f"MLflow error during {fn_name}: {exc}")
            if strict:
                raise
            return None

    def _init_mlflow(self) -> None:
        if not _MLFLOW_AVAILABLE:
            raise RuntimeError("mlflow is not installed. Add mlflow to requirements.")

        mcfg = self.cfg.get("mlflow", {})

        # Backward-compatible alias: allow user-provided CART path to feed
        # MLflow's standard CERT environment variable.
        tracking_cert_path = os.getenv("MLFLOW_TRACKING_SERVER_CERT_PATH")
        tracking_cart_path = os.getenv("MLFLOW_TRACKING_SERVER_CART_PATH")
        if tracking_cart_path and not tracking_cert_path:
            os.environ["MLFLOW_TRACKING_SERVER_CERT_PATH"] = tracking_cart_path

        tracking_uri = mcfg.get("tracking_uri", None)
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)

        if mcfg.get("insecure_tls", False):
            os.environ["MLFLOW_TRACKING_INSECURE_TLS"] = "true"
        else:
            os.environ.pop("MLFLOW_TRACKING_INSECURE_TLS", None)

        exp_name = mcfg.get("experiment_name", "torch_flow")
        self._mlflow_safe("set_experiment", lambda: mlflow.set_experiment(exp_name))

        tags = {
            "run_name": self.run_name,
            "device": str(self.device),
        }
        extra_tags = mcfg.get("tags", {})
        if isinstance(extra_tags, dict):
            tags.update({str(k): str(v) for k, v in extra_tags.items()})

        self.mlflow_run = self._mlflow_safe(
            "start_run",
            lambda: mlflow.start_run(run_name=self.run_name, tags=tags),
        )
        if self.mlflow_run is None:
            return

        self.mlflow_run_id = self.mlflow_run.info.run_id
        self.mlflow_client = MlflowClient()

        cfg_dict = OmegaConf.to_container(self.cfg, resolve=True)
        flat_cfg = _flatten_dict(cfg_dict)
        params = {k: str(v)[:500] for k, v in flat_cfg.items()}

        def _log_params_chunked() -> None:
            keys = list(params.keys())
            chunk_size = 100
            for i in range(0, len(keys), chunk_size):
                chunk = {k: params[k] for k in keys[i:i + chunk_size]}
                mlflow.log_params(chunk)

        self._mlflow_safe("log_params", _log_params_chunked)

        cfg_path = os.path.join(self.config_dir, "resolved_config.yaml")
        self._mlflow_safe(
            "log_resolved_config",
            lambda: mlflow.log_artifact(cfg_path, artifact_path="config"),
        )

        counts = self.model.param_count()
        self._mlflow_safe(
            "log_param_counts",
            lambda: mlflow.log_params(
                {
                    "model.feature_encoder_params": counts["feature_encoder"],
                    "model.context_encoder_params": counts["context_encoder"],
                    "model.stages_params": counts["stages"],
                    "model.total_params": counts["total"],
                }
            ),
        )

    def _mlflow_log_checkpoint(self, path: str, tag: str) -> None:
        if not (self.use_mlflow and self.mlflow_run_id):
            return
        if not bool(self.cfg.get("mlflow", {}).get("log_checkpoints", True)):
            return
        self._mlflow_safe(
            f"log_checkpoint_{tag}",
            lambda: mlflow.log_artifact(path, artifact_path="checkpoints"),
        )

    def _mlflow_log_best_model_and_register(self) -> None:
        if not (self.use_mlflow and self.mlflow_run_id):
            return

        artifact_path = f"best_step_{self.global_step:07d}"
        model_info = self._mlflow_safe(
            "log_best_model",
            lambda: mlflow.pytorch.log_model(
                pytorch_model=self.model,
                artifact_path=artifact_path,
            ),
        )
        if model_info is None:
            return

        self.best_model_uri = model_info.model_uri

        mcfg = self.cfg.get("mlflow", {})
        if not bool(mcfg.get("register_best", True)):
            return
        model_name = mcfg.get("registered_model_name", None)
        if not model_name:
            return

        mv = self._mlflow_safe(
            "register_best_model",
            lambda: mlflow.register_model(model_uri=model_info.model_uri, name=model_name),
        )
        if mv is None:
            return

        alias = mcfg.get("best_alias", "best")
        if self.mlflow_client is not None and alias:
            self._mlflow_safe(
                "set_best_alias",
                lambda: self.mlflow_client.set_registered_model_alias(model_name, alias, mv.version),
            )

    def _log_final_mlflow_artifacts(self, history_path: str) -> None:
        if not (self.use_mlflow and self.mlflow_run_id):
            return

        self._mlflow_safe(
            "log_history_artifact",
            lambda: mlflow.log_artifact(history_path, artifact_path="history"),
        )

        if bool(self.cfg.get("mlflow", {}).get("log_tensorboard", True)):
            tb_dir = os.path.join(self.log_dir, self.run_name)
            if os.path.isdir(tb_dir):
                self._mlflow_safe(
                    "log_tensorboard_artifacts",
                    lambda: mlflow.log_artifacts(tb_dir, artifact_path="tensorboard"),
                )

        self._mlflow_safe(
            "log_summary_metrics",
            lambda: mlflow.log_metrics(
                {
                    "summary/best_epe": float(self.history.best_epe),
                    "summary/best_step": float(self.history.best_step),
                    "summary/final_step": float(self.global_step),
                },
                step=self.global_step,
            ),
        )

    def _mlflow_end_run(self, status: str) -> None:
        if not (self.use_mlflow and self.mlflow_run_id):
            return
        self._mlflow_safe("end_run", lambda: mlflow.end_run(status=status))
        self.mlflow_run_id = None
        self.mlflow_run = None

    @staticmethod
    def _infinite_loader(loader: DataLoader):
        while True:
            yield from loader
