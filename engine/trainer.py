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
import time
import logging
import math
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

try:
    from torch.utils.tensorboard import SummaryWriter
    _TB_AVAILABLE = True
except ImportError:
    _TB_AVAILABLE = False

from models import build_model
from losses import HQSFlowLoss
from data import build_dataset, build_dataloader
from utils import compute_metrics, aggregate_metrics, flow_to_color, InputPadder

logger = logging.getLogger(__name__)


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
    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scaler is not None and "scaler" in ckpt:
        scaler.load_state_dict(ckpt["scaler"])
    if scheduler is not None and ckpt.get("scheduler") is not None:
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
        self.optimizer = optim.AdamW(
            self.model.parameters(),
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

        # ── State ─────────────────────────────────────────────────────────────
        self.global_step = 0
        self.history = TrainingHistory(
            run_name=self.run_name,
            total_steps=train_steps,
        )

        if cfg.training.get("checkpoint"):
            step, loaded_history = load_checkpoint(
                cfg.training.checkpoint,
                self.model, self.optimizer, self.scaler, self.scheduler,
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

        if steps_left <= 0:
            logger.info("Already at target step count — nothing to do.")
            return

        self.model.train()
        loader_iter     = iter(self._infinite_loader(self.train_loader))
        steps_per_epoch = len(self.train_loader)

        # ── Epoch-level progress bar ─────────────────────────────────────────
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

            t_batch    = time.time()
            batch      = next(loader_iter)
            loss_dict  = self._train_step(batch)

            running.update(loss_dict)

            self.global_step += 1
            steps_in_ep += 1

            # ── Update batch bar with running means ──────────────────────────
            lr = self.optimizer.param_groups[0]["lr"]
            if batch_bar is not None:
                rm = running.mean()
                def _fmt(k, fmt=".3f"):
                    v = rm.get(k, float("nan"))
                    return "nan" if math.isnan(v) else format(v, fmt)
                postfix: Dict[str, str] = {
                    "loss":    f"{rm['loss']:.4f}",
                    "epe":     f"{rm['epe']:.3f}",
                    "f1":      _fmt("f1"),
                    "s0_10":   _fmt("s0_10"),
                    "s10_40":  _fmt("s10_40"),
                    "s40+":    _fmt("s40_plus"),
                    "lr":      f"{lr:.2e}",
                }
                # Add auxiliary losses if present
                if "smooth" in rm:
                    postfix["smooth"] = _fmt("smooth", ".4f")
                if "photo" in rm:
                    postfix["photo"] = _fmt("photo", ".4f")
                if "ofce" in rm:
                    postfix["ofce"] = _fmt("ofce", ".4f")
                batch_bar.set_postfix(**postfix)
                batch_bar.update(1)

            # ── Log scalars ──────────────────────────────────────────────────
            if self.global_step % log_every == 0:
                self.history.train_loss.append((self.global_step, loss_dict["loss"]))
                self.history.train_epe.append( (self.global_step, loss_dict["epe"]))
                self._log_scalars(
                    {k: v for k, v in loss_dict.items()},
                    prefix="train",
                )
                self._log_scalars({"lr": lr}, prefix="train")

            # ── Validation ───────────────────────────────────────────────────
            if self.global_step % val_every == 0:
                val_metrics = self._validate()
                if val_metrics:
                    epe = val_metrics.get("epe", math.inf)
                    is_best = epe < self.history.best_epe
                    if is_best:
                        self.history.best_epe  = epe
                        self.history.best_step = self.global_step
                        self._save("best")
                    # Surface val metrics in the epoch bar immediately
                    def _fmtv(k, fmt=".3f"):
                        v = val_metrics.get(k, float("nan"))
                        return "nan" if math.isnan(v) else format(v, fmt)
                    tag = " ★" if is_best else ""
                    logger.info(
                        f"✓ Val EPE={epe:.4f}  F1={_fmtv('f1')}%"
                        f"  s0_10={_fmtv('s0_10')}  s10_40={_fmtv('s10_40')}"
                        f"  s40+={_fmtv('s40_plus')}{tag}"
                    )
                    epoch_bar.set_postfix(
                        val_epe=f"{epe:.4f}",
                        best=f"{self.history.best_epe:.4f}",
                    )

            # ── Periodic checkpoint ──────────────────────────────────────────
            if self.global_step % save_every == 0:
                self._save(f"step_{self.global_step:07d}")

            # ── Epoch boundary ───────────────────────────────────────────────
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
                    epe=f"{rm['epe']:.3f}",
                    f1=_fmt("f1"),
                    s0_10=_fmt("s0_10"),
                    s10_40=_fmt("s10_40"),
                    **{"s40+": _fmt("s40_plus")},
                    best_epe=f"{self.history.best_epe:.4f}",
                    t=f"{elapsed:.0f}s",
                )
                epoch_bar.update(1)
                epoch_idx   += 1
                steps_in_ep  = 0

        # Close any open batch bar at end of training
        if batch_bar is not None:
            batch_bar.close()
        epoch_bar.close()

        # ── Final saves ──────────────────────────────────────────────────────
        self._save("last")
        history_path = os.path.join(self.ckpt_dir, self.run_name, "history.json")
        self.history.to_json(history_path)
        logger.info(f"Training history → {history_path}")
        logger.info(
            f"Training complete. "
            f"Best EPE={self.history.best_epe:.4f} @ step {self.history.best_step}."
        )

        if self.writer:
            self.writer.close()
        if self.use_wandb:
            import wandb
            wandb.finish()

    # ─────────────────────────────────────────────────────────────────────── #
    # Single training step
    # ─────────────────────────────────────────────────────────────────────── #

    def _train_step(self, batch: Dict) -> Dict[str, float]:
        self.optimizer.zero_grad(set_to_none=True)

        img1  = batch["image1"].to(self.device, non_blocking=True)
        img2  = batch["image2"].to(self.device, non_blocking=True)
        flow  = batch["flow"].to(self.device, non_blocking=True)
        valid = batch["valid"].to(self.device, non_blocking=True)

        with torch.autocast(device_type=self.device.type, enabled=self._use_amp):
            out       = self.model(img1, img2)
            loss_dict = self.criterion(out["flow_preds"], flow, valid, img1, img2)

        self.scaler.scale(loss_dict["loss"]).backward()
        self.scaler.unscale_(self.optimizer)
        nn.utils.clip_grad_norm_(
            self.model.parameters(),
            self.cfg.training.get("grad_clip", 1.0),
        )
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.scheduler.step()

        scalar_dict = {k: v.item() if isinstance(v, torch.Tensor) else v
                       for k, v in loss_dict.items()}

        # ── Per-batch speed + F1 metrics on final prediction ────────────────
        with torch.no_grad():
            pred_final = out["flow_preds"][-1].detach()
            # Average metrics over batch items
            batch_metrics: Dict[str, list] = {
                "f1": [], "s0_10": [], "s10_40": [], "s40_plus": []
            }
            for b in range(pred_final.shape[0]):
                m = compute_metrics(pred_final[b], flow[b], valid[b])
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

                padder = InputPadder(img1.shape, divisor=8)
                img1, img2 = padder.pad(img1, img2)

                with torch.autocast(device_type=self.device.type, enabled=self._use_amp):
                    out  = self.model(img1, img2)
                pred = padder.unpad(out["flow_preds"][-1])

                for b in range(img1.shape[0]):
                    occ = None
                    if occ_batch is not None:
                        occ = occ_batch[b].to(self.device)
                    results.append(
                        compute_metrics(pred[b], flow[b], valid[b], occ_mask=occ)
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

    def _log_scalars(self, d: Dict, prefix: str) -> None:
        if self.writer:
            for k, v in d.items():
                self.writer.add_scalar(f"{prefix}/{k}", v, self.global_step)
        if self.use_wandb:
            import wandb
            wandb.log({f"{prefix}/{k}": v for k, v in d.items()},
                      step=self.global_step)

    @staticmethod
    def _infinite_loader(loader: DataLoader):
        while True:
            yield from loader

