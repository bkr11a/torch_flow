"""Training loss for the HQS-LM-SF prototype."""
from __future__ import annotations

from typing import Dict, Mapping, Optional

import torch
import torch.nn as nn


def _cfg_get(cfg, key: str, default):
    if cfg is None:
        return default
    if isinstance(cfg, Mapping):
        return cfg.get(key, default)
    try:
        return cfg.get(key, default)
    except Exception:
        return getattr(cfg, key, default)


def _mask(valid: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    if valid.ndim == 3:
        valid = valid.unsqueeze(1)
    if valid.ndim != 4 or valid.shape[1] != 1:
        raise ValueError(
            f"valid must be [B,H,W] or [B,1,H,W], got {tuple(valid.shape)}"
        )
    if valid.shape[-2:] != reference.shape[-2:]:
        raise ValueError("valid and prediction grids differ")
    return valid.to(device=reference.device, dtype=reference.dtype)


def _masked_mean(value: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    return (value * valid).sum() / valid.sum().clamp_min(1.0)


class HQSSceneFlowLoss(nn.Module):
    """Sequence supervision plus optional induced-flow and geometry terms."""

    def __init__(self, cfg=None) -> None:
        super().__init__()
        self.gamma = float(_cfg_get(cfg, "gamma", 0.85))
        self.induced_flow_weight = float(
            _cfg_get(cfg, "induced_flow_weight", 0.10)
        )
        self.geometry_weight = float(
            _cfg_get(cfg, "geometry_weight", 0.05)
        )
        self.coupling_weight = float(
            _cfg_get(cfg, "coupling_weight", 0.0)
        )
        self.maximum_scene_epe = float(
            _cfg_get(cfg, "maximum_scene_epe", 400.0)
        )

    def forward(
        self,
        prediction: Dict[str, object],
        scene_flow_gt: torch.Tensor,
        valid: torch.Tensor,
        *,
        flow_gt: Optional[torch.Tensor] = None,
        flow_valid: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        if scene_flow_gt.ndim != 4 or scene_flow_gt.shape[1] != 3:
            raise ValueError("scene_flow_gt must be [B,3,H,W]")
        scene_predictions = prediction.get("scene_flow_preds")
        if not isinstance(scene_predictions, list) or not scene_predictions:
            raise ValueError("prediction must contain non-empty scene_flow_preds")

        scene_valid = _mask(valid, scene_flow_gt)
        finite = torch.isfinite(scene_flow_gt).all(
            dim=1, keepdim=True
        ).to(dtype=scene_flow_gt.dtype)
        magnitude_valid = (
            torch.linalg.vector_norm(scene_flow_gt, dim=1, keepdim=True)
            < self.maximum_scene_epe
        ).to(dtype=scene_flow_gt.dtype)
        scene_valid = scene_valid * finite * magnitude_valid

        count = len(scene_predictions)
        scene_loss = scene_flow_gt.new_zeros(())
        for index, estimate in enumerate(scene_predictions):
            if estimate.shape != scene_flow_gt.shape:
                raise ValueError(
                    "All scene-flow predictions must be full resolution; got "
                    f"{tuple(estimate.shape)} vs {tuple(scene_flow_gt.shape)}"
                )
            weight = self.gamma ** (count - index - 1)
            endpoint = torch.linalg.vector_norm(
                estimate - scene_flow_gt, dim=1, keepdim=True
            )
            scene_loss = scene_loss + weight * _masked_mean(
                endpoint, scene_valid
            )

        flow_loss = scene_loss.new_zeros(())
        induced_predictions = prediction.get("flow_preds")
        if flow_gt is not None:
            if flow_gt.ndim != 4 or flow_gt.shape[1] != 2:
                raise ValueError("flow_gt must be [B,2,H,W]")
            if not isinstance(induced_predictions, list) or not induced_predictions:
                raise ValueError("prediction must contain flow_preds")
            current_flow_valid = _mask(
                flow_valid if flow_valid is not None else valid,
                flow_gt,
            )
            flow_count = len(induced_predictions)
            for index, estimate in enumerate(induced_predictions):
                weight = self.gamma ** (flow_count - index - 1)
                endpoint = torch.linalg.vector_norm(
                    estimate - flow_gt, dim=1, keepdim=True
                )
                flow_loss = flow_loss + weight * _masked_mean(
                    endpoint, current_flow_valid
                )

        geometry_loss = scene_loss.new_zeros(())
        depth_residual = prediction.get("depth_residual_final")
        projection_valid = prediction.get("projection_valid_final")
        if isinstance(depth_residual, torch.Tensor):
            if not isinstance(projection_valid, torch.Tensor):
                projection_valid = torch.ones_like(depth_residual)
            geometry_loss = _masked_mean(
                torch.sqrt(depth_residual.square() + 1e-4),
                projection_valid.to(dtype=depth_residual.dtype),
            )

        coupling_loss = scene_loss.new_zeros(())
        coupling = prediction.get("coupling_residual_low")
        if self.coupling_weight > 0.0 and isinstance(coupling, list):
            if coupling:
                coupling_loss = torch.stack(
                    [
                        torch.linalg.vector_norm(
                            value, dim=1
                        ).mean()
                        for value in coupling
                    ]
                ).mean()

        total = (
            scene_loss
            + self.induced_flow_weight * flow_loss
            + self.geometry_weight * geometry_loss
            + self.coupling_weight * coupling_loss
        )
        final_scene_epe = _masked_mean(
            torch.linalg.vector_norm(
                scene_predictions[-1] - scene_flow_gt,
                dim=1,
                keepdim=True,
            ),
            scene_valid,
        )
        result = {
            "loss": total,
            "scene_sequence_loss": scene_loss,
            "scene_epe": final_scene_epe,
            "induced_flow_loss": flow_loss,
            "geometry_loss": geometry_loss,
            "coupling_loss": coupling_loss,
        }
        if flow_gt is not None and isinstance(induced_predictions, list):
            flow_mask = _mask(
                flow_valid if flow_valid is not None else valid,
                flow_gt,
            )
            result["flow_epe"] = _masked_mean(
                torch.linalg.vector_norm(
                    induced_predictions[-1] - flow_gt,
                    dim=1,
                    keepdim=True,
                ),
                flow_mask,
            )
        return result


__all__ = ["HQSSceneFlowLoss"]
