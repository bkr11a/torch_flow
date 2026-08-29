"""OF-A4 single-state operator-composed HQSCore control.

The reference front end, global initialisation, recurrent scale schedule and
final upsampler are inherited unchanged from HQSCore.  The only architectural
intervention is removal of the persistent HQS ``w/q`` split: each iteration
propagates one flow state through data and regularisation operators in sequence.
"""
from __future__ import annotations

import logging

from hqs_pytorch.customML.customModels.HQSCore import HQSCore, _cfg_get, _int_list
from models.hqs_ablation_components import build_capacity_matched_single_state_cell
from models.hqs_recurrent_control_components import count_trainable_parameters

logger = logging.getLogger(__name__)


class HQSCoreSingleState(HQSCore):
    """A4: one persistent flow state with A0-matched active recurrent capacity."""

    def __init__(self, cfg) -> None:
        super().__init__(cfg)
        core_cfg = _cfg_get(cfg, "hqs_core", cfg)
        single_cfg = _cfg_get(core_cfg, "single_state", {})

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
            adapter_max_channels=int(
                _cfg_get(single_cfg, "adapter_max_channels", 128)
            ),
            tolerance=float(
                _cfg_get(single_cfg, "capacity_tolerance", 0.01)
            ),
        )

        self.coarse_cell, coarse_report = build_capacity_matched_single_state_cell(
            target_parameters=coarse_target,
            context_channels=context_by_scale[16],
            hidden_channels=int(_cfg_get(core_cfg, "coarse_hidden_dim", 96)),
            max_iterations=max(iterations[0], iterations[1]),
            **common,
        )
        self.fine_cell, fine_report = build_capacity_matched_single_state_cell(
            target_parameters=fine_target,
            context_channels=context_by_scale[4],
            hidden_channels=int(_cfg_get(core_cfg, "fine_hidden_dim", 64)),
            max_iterations=max(iterations[2], iterations[3]),
            **common,
        )

        self.single_state_analytic_validity_mode = str(
            _cfg_get(core_cfg, "analytic_validity_mode", "post_gate")
        ).lower()
        self.capacity_match_report = {
            "coarse": coarse_report.as_dict(),
            "fine": fine_report.as_dict(),
            "semantic_note": (
                "One persistent flow state is propagated. HQSState.w and q are "
                "compatibility aliases at iteration boundaries; coupling residual "
                "diagnostics are not applicable to A4."
            ),
        }
        logger.info("OF-A4 single-state capacity match: %s", self.capacity_match_report)

    def forward(self, *args, **kwargs):
        out = super().forward(*args, **kwargs)

        # Parent HQSCore stores state.w as its data-proposal diagnostic.  A4
        # cannot do that without reintroducing a persistent second state, so
        # reconstruct the temporary data proposal from the previous single
        # flow, the declared data increment and its validity gate.
        flow_lows = out.get("flow_low", [])
        delta_lows = out.get("delta_low", [])
        data_deltas = out.get("data_delta_lows", [])
        validities = out.get("core_validity_lows", [])
        if (
            len(flow_lows) == len(delta_lows) == len(data_deltas)
            and len(flow_lows) == len(validities)
        ):
            data_flows = []
            prior_deltas = []
            for final_flow, total_delta, data_delta, validity in zip(
                flow_lows, delta_lows, data_deltas, validities
            ):
                previous_flow = final_flow - total_delta
                if self.single_state_analytic_validity_mode == "weighted_solve":
                    data_flow = previous_flow + data_delta
                else:
                    data_flow = previous_flow + validity * data_delta
                data_flows.append(data_flow)
                prior_deltas.append(final_flow - data_flow)
            out["data_flow_low"] = data_flows
            # Compatibility alias only: this is a transient data proposal, not
            # an auxiliary HQS state.
            out["aux_low"] = data_flows
            out["delta_prior_low"] = prior_deltas

        out["coupling_residual_low"] = [
            flow.new_zeros(flow.shape) for flow in flow_lows
        ]
        out["state_semantics"] = "single_persistent_flow"
        out["coupling_diagnostic_applicable"] = False
        return out


__all__ = ["HQSCoreSingleState"]
