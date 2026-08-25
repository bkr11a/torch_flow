"""Equal-capacity unconstrained recurrent control for the HQSCore OF-A ablation.

This subclasses HQSCore specifically so that the feature pyramid, matching front
end, global initialisation, multiscale recurrence schedule, scale-transition
logic and final convex upsampler remain identical. Only ``coarse_cell`` and
``fine_cell`` are replaced.
"""
from __future__ import annotations

import logging
from typing import Mapping

from hqs_pytorch.customML.customModels.HQSCore import HQSCore, _cfg_get, _int_list
from models.hqs_recurrent_control_components import (
    build_capacity_matched_recurrent_cell,
    count_trainable_parameters,
)

logger = logging.getLogger(__name__)


class HQSCoreRecurrentControl(HQSCore):
    """OF-A1/A2 recurrent control with matched active trainable capacity."""

    def __init__(self, cfg) -> None:
        super().__init__(cfg)
        core_cfg = _cfg_get(cfg, "hqs_core", cfg)
        control_cfg = _cfg_get(core_cfg, "recurrent_control", {})

        # Capture target solver budgets *before* replacing the structured cells.
        coarse_target = count_trainable_parameters(self.coarse_cell)
        fine_target = count_trainable_parameters(self.fine_cell)

        context_channels = _int_list(
            core_cfg, "context_channels", (64, 64, 96, 96)
        )
        context_by_scale = dict(zip((2, 4, 8, 16), context_channels))
        iterations = _int_list(core_cfg, "iterations", (2, 4, 2, 2))
        groups = int(_cfg_get(core_cfg, "groups", 8))

        common = dict(
            correlation_channels=self.correlation_embedding_dim,
            groups=groups,
            hidden_min=int(_cfg_get(control_cfg, "hidden_min", 48)),
            hidden_max=int(_cfg_get(control_cfg, "hidden_max", 192)),
            hidden_step=int(_cfg_get(control_cfg, "hidden_step", 8)),
            residual_blocks_min=int(
                _cfg_get(control_cfg, "residual_blocks_min", 0)
            ),
            residual_blocks_max=int(
                _cfg_get(control_cfg, "residual_blocks_max", 6)
            ),
            adapter_max_channels=int(
                _cfg_get(control_cfg, "adapter_max_channels", 2048)
            ),
            tolerance=float(_cfg_get(control_cfg, "capacity_tolerance", 0.01)),
        )
        self.coarse_cell, coarse_report = build_capacity_matched_recurrent_cell(
            target_parameters=coarse_target,
            context_channels=context_by_scale[16],
            max_iterations=max(iterations[0], iterations[1]),
            **common,
        )
        self.fine_cell, fine_report = build_capacity_matched_recurrent_cell(
            target_parameters=fine_target,
            context_channels=context_by_scale[4],
            max_iterations=max(iterations[2], iterations[3]),
            **common,
        )

        self.capacity_match_report = {
            "coarse": coarse_report.as_dict(),
            "fine": fine_report.as_dict(),
            "semantic_note": (
                "HQSState.q is a compatibility alias equal to w; it is not an "
                "independent split variable in the recurrent control."
            ),
        }
        logger.info("OF-A recurrent-control capacity match: %s", self.capacity_match_report)


__all__ = ["HQSCoreRecurrentControl"]
