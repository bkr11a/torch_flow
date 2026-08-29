"""OF-B operator ablations for HQSCore.

The parent HQSCore supplies the identical feature pyramid, frozen GMFlow front
end, correlation construction, global initialisation, scale schedule, recurrent
loop and final upsampler used by A0.  Only the structured recurrent cells are
replaced by interface-compatible cells with explicitly declared operator
interventions.
"""
from __future__ import annotations

import logging

from hqs_pytorch.customML.customModels.HQSCore import HQSCore, _cfg_get, _int_list
from models.hqs_ablation_components import AblatedStructuredHQSCell
from models.hqs_recurrent_control_components import count_trainable_parameters

logger = logging.getLogger(__name__)


class HQSCoreOperatorAblation(HQSCore):
    """HQSCore with one controlled OF-B analytic/learned operator intervention."""

    def __init__(self, cfg) -> None:
        super().__init__(cfg)
        core_cfg = _cfg_get(cfg, "hqs_core", cfg)
        ab_cfg = _cfg_get(core_cfg, "operator_ablation", {})

        reference_coarse_cell = self.coarse_cell
        reference_fine_cell = self.fine_cell
        coarse_target = count_trainable_parameters(reference_coarse_cell)
        fine_target = count_trainable_parameters(reference_fine_cell)

        context_channels = _int_list(
            core_cfg, "context_channels", (64, 64, 96, 96)
        )
        context_by_scale = dict(zip((2, 4, 8, 16), context_channels))
        iterations = _int_list(core_cfg, "iterations", (2, 4, 2, 2))
        groups = int(_cfg_get(core_cfg, "groups", 8))

        common = dict(
            correlation_channels=self.correlation_embedding_dim,
            prior_hidden_channels=int(
                _cfg_get(core_cfg, "prior_hidden_channels", 64)
            ),
            groups=groups,
            beta_initial=float(_cfg_get(core_cfg, "beta_initial", 0.10)),
            beta_minimum=float(_cfg_get(core_cfg, "beta_minimum", 0.01)),
            lambda_initial=float(_cfg_get(core_cfg, "lambda_initial", 0.08)),
            lambda_minimum=float(_cfg_get(core_cfg, "lambda_minimum", 0.001)),
            edge_alpha=float(_cfg_get(core_cfg, "edge_alpha", 10.0)),
            analytic_weight=float(_cfg_get(core_cfg, "analytic_weight", 1.0)),
            learned_data_weight=float(
                _cfg_get(core_cfg, "learned_data_weight", 1.0)
            ),
            analytic_validity_mode=str(
                _cfg_get(core_cfg, "analytic_validity_mode", "post_gate")
            ),
            use_analytic_data=bool(
                _cfg_get(ab_cfg, "use_analytic_data", True)
            ),
            use_learned_data=bool(
                _cfg_get(ab_cfg, "use_learned_data", True)
            ),
            analytic_input_to_learned=bool(
                _cfg_get(ab_cfg, "analytic_input_to_learned", True)
            ),
            use_analytic_prox=bool(
                _cfg_get(ab_cfg, "use_analytic_prox", True)
            ),
            use_learned_prox=bool(
                _cfg_get(ab_cfg, "use_learned_prox", True)
            ),
        )

        new_coarse_cell = AblatedStructuredHQSCell(
            context_channels=context_by_scale[16],
            hidden_channels=int(_cfg_get(core_cfg, "coarse_hidden_dim", 96)),
            max_iterations=max(iterations[0], iterations[1]),
            **common,
        )
        new_fine_cell = AblatedStructuredHQSCell(
            context_channels=context_by_scale[4],
            hidden_channels=int(_cfg_get(core_cfg, "fine_hidden_dim", 64)),
            max_iterations=max(iterations[2], iterations[3]),
            **common,
        )

        # All parameterised submodules are shape-identical to A0.  Copy the
        # just-created reference initialisation so a fixed random seed gives
        # the same starting weights; the only intervention is the forward-path
        # operator switch.
        new_coarse_cell.load_state_dict(reference_coarse_cell.state_dict(), strict=True)
        new_fine_cell.load_state_dict(reference_fine_cell.state_dict(), strict=True)
        self.coarse_cell = new_coarse_cell
        self.fine_cell = new_fine_cell

        coarse_new = count_trainable_parameters(self.coarse_cell)
        fine_new = count_trainable_parameters(self.fine_cell)
        if coarse_new != coarse_target or fine_new != fine_target:
            raise RuntimeError(
                "OF-B cells must retain A0 nominal capacity exactly: "
                f"coarse {coarse_new}/{coarse_target}, "
                f"fine {fine_new}/{fine_target}"
            )

        self.operator_ablation_report = {
            "coarse_parameters": coarse_new,
            "fine_parameters": fine_new,
            "use_analytic_data": common["use_analytic_data"],
            "use_learned_data": common["use_learned_data"],
            "analytic_input_to_learned": common["analytic_input_to_learned"],
            "use_analytic_prox": common["use_analytic_prox"],
            "use_learned_prox": common["use_learned_prox"],
        }
        logger.info("OF-B operator ablation: %s", self.operator_ablation_report)


__all__ = ["HQSCoreOperatorAblation"]
