"""Strong matching-pyramid and correspondence-decoding components.

These operators upgrade only the measurement front end of :class:`HQSCore`.
The source-conditioned context pyramid, analytic data update, proximal
operator and recurrent schedule remain unchanged.

The matching path has three stages:

1. a deeper shared Siamese convolutional pyramid;
2. bidirectional fine/coarse feature fusion;
3. symmetric self/cross-frame attention at the coarse matching scales.

The global decoder retains spatially distinct modes, refines each mode with a
local expectation, evaluates the modes against a reverse correspondence map,
and selects the best cycle-consistent candidate.  Confidence controls whether
the selected measurement is admitted as the initial state; it never scales a
valid displacement to an incorrect smaller displacement.
"""
from __future__ import annotations

import math
from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .hqs_core_components import (
    ConvGNAct,
    HQSCorePyramidEncoder,
    ResidualGNBlock,
)
from .warp import backward_warp, flow_in_bounds_mask


def _logit(probability: float) -> float:
    probability = min(max(float(probability), 1e-5), 1.0 - 1e-5)
    return math.log(probability / (1.0 - probability))


class SymmetricCrossFrameBlock(nn.Module):
    """One weight-shared self/cross-attention block for an image pair.

    Sharing the attention and feed-forward weights between the two directions
    preserves the Siamese symmetry of the matching representation.  Both
    cross-attention updates are calculated from the same pre-update states so
    the result is independent of evaluation order.
    """

    def __init__(
        self,
        channels: int,
        *,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        channels = int(channels)
        num_heads = int(num_heads)
        if channels % num_heads != 0:
            raise ValueError(
                f"channels={channels} must be divisible by heads={num_heads}"
            )
        hidden = max(int(round(channels * float(mlp_ratio))), channels)
        self.self_norm = nn.LayerNorm(channels)
        self.self_attention = nn.MultiheadAttention(
            channels,
            num_heads,
            dropout=float(dropout),
            batch_first=True,
        )
        self.cross_query_norm = nn.LayerNorm(channels)
        self.cross_key_norm = nn.LayerNorm(channels)
        self.cross_attention = nn.MultiheadAttention(
            channels,
            num_heads,
            dropout=float(dropout),
            batch_first=True,
        )
        self.feed_forward_norm = nn.LayerNorm(channels)
        self.feed_forward = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(hidden, channels),
        )

    def forward(
        self,
        source: torch.Tensor,
        target: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        source_self = self.self_norm(source)
        target_self = self.self_norm(target)
        source_delta, _ = self.self_attention(
            source_self,
            source_self,
            source_self,
            need_weights=False,
        )
        target_delta, _ = self.self_attention(
            target_self,
            target_self,
            target_self,
            need_weights=False,
        )
        source = source + source_delta
        target = target + target_delta

        source_query = self.cross_query_norm(source)
        target_query = self.cross_query_norm(target)
        source_key = self.cross_key_norm(source)
        target_key = self.cross_key_norm(target)
        source_cross, _ = self.cross_attention(
            source_query,
            target_key,
            target_key,
            need_weights=False,
        )
        target_cross, _ = self.cross_attention(
            target_query,
            source_key,
            source_key,
            need_weights=False,
        )
        source = source + source_cross
        target = target + target_cross
        source = source + self.feed_forward(
            self.feed_forward_norm(source)
        )
        target = target + self.feed_forward(
            self.feed_forward_norm(target)
        )
        return source, target


class PairFeatureInteraction(nn.Module):
    """Native-grid symmetric feature interaction.

    ``window_size=0`` uses global attention while the token count is below
    ``maximum_global_tokens``.  Larger grids fall back to native-resolution
    shifted windows instead of spatial pooling.  Positive alternating shifts
    connect neighbouring windows without deleting thin structures.
    """

    def __init__(
        self,
        channels: int,
        *,
        depth: int,
        num_heads: int,
        window_size: int = 0,
        fallback_window_size: int = 16,
        maximum_global_tokens: int = 4096,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        shifted_windows: bool = True,
        gradient_checkpointing: bool = True,
    ) -> None:
        super().__init__()
        self.channels = int(channels)
        self.window_size = max(int(window_size), 0)
        self.fallback_window_size = max(int(fallback_window_size), 1)
        self.maximum_global_tokens = max(int(maximum_global_tokens), 1)
        self.shifted_windows = bool(shifted_windows)
        self.gradient_checkpointing = bool(gradient_checkpointing)
        self.position_source = nn.Conv2d(
            self.channels,
            self.channels,
            kernel_size=3,
            padding=1,
            groups=self.channels,
            bias=True,
        )
        self.blocks = nn.ModuleList(
            [
                SymmetricCrossFrameBlock(
                    self.channels,
                    num_heads=int(num_heads),
                    mlp_ratio=float(mlp_ratio),
                    dropout=float(dropout),
                )
                for _ in range(max(int(depth), 0))
            ]
        )

    @staticmethod
    def _partition_windows(
        value: torch.Tensor,
        window_size: int,
        shift: int,
    ) -> Tuple[torch.Tensor, Tuple[int, int, int, int, int, int]]:
        batch, channels, height, width = value.shape
        # Offset the window grid with explicit top/left padding.  Unlike a
        # cyclic roll, this cannot create false adjacency between opposite
        # image borders.
        pad_top = int(shift)
        pad_left = int(shift)
        padded_content_h = height + pad_top
        padded_content_w = width + pad_left
        pad_bottom = (
            window_size - padded_content_h % window_size
        ) % window_size
        pad_right = (
            window_size - padded_content_w % window_size
        ) % window_size
        if pad_top or pad_left or pad_bottom or pad_right:
            value = F.pad(
                value,
                (pad_left, pad_right, pad_top, pad_bottom),
                mode="replicate",
            )
        padded_h, padded_w = value.shape[-2:]
        tokens = value.permute(0, 2, 3, 1).reshape(
            batch,
            padded_h // window_size,
            window_size,
            padded_w // window_size,
            window_size,
            channels,
        )
        tokens = tokens.permute(0, 1, 3, 2, 4, 5).reshape(
            -1,
            window_size * window_size,
            channels,
        )
        metadata = (
            batch,
            height,
            width,
            padded_h,
            padded_w,
            int(shift),
        )
        return tokens, metadata

    @staticmethod
    def _reverse_windows(
        tokens: torch.Tensor,
        metadata: Tuple[int, int, int, int, int, int],
        window_size: int,
        shift: int,
    ) -> torch.Tensor:
        (
            batch,
            height,
            width,
            padded_h,
            padded_w,
            metadata_shift,
        ) = metadata
        if metadata_shift != int(shift):
            raise RuntimeError("Window shift metadata mismatch")
        channels = tokens.shape[-1]
        value = tokens.reshape(
            batch,
            padded_h // window_size,
            padded_w // window_size,
            window_size,
            window_size,
            channels,
        )
        value = value.permute(0, 1, 3, 2, 4, 5).reshape(
            batch,
            padded_h,
            padded_w,
            channels,
        )
        value = value.permute(0, 3, 1, 2)
        return value[
            ...,
            metadata_shift : metadata_shift + height,
            metadata_shift : metadata_shift + width,
        ]

    def _run_block(
        self,
        block: SymmetricCrossFrameBlock,
        source: torch.Tensor,
        target: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        use_checkpoint = (
            self.training
            and self.gradient_checkpointing
            and torch.is_grad_enabled()
            and (source.requires_grad or target.requires_grad)
        )
        if use_checkpoint:
            return checkpoint(
                block,
                source,
                target,
                use_reentrant=False,
                preserve_rng_state=False,
            )
        return block(source, target)

    def forward(
        self,
        source: torch.Tensor,
        target: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if source.shape != target.shape or source.ndim != 4:
            raise ValueError(
                "PairFeatureInteraction expects equal [B,C,H,W] tensors, "
                f"got {tuple(source.shape)} and {tuple(target.shape)}"
            )
        if source.shape[1] != self.channels:
            raise ValueError(
                f"Expected {self.channels} channels, got {source.shape[1]}"
            )
        source = source + self.position_source(source)
        target = target + self.position_source(target)
        _, _, height, width = source.shape
        use_global = (
            self.window_size == 0
            and height * width <= self.maximum_global_tokens
        )
        window_size = (
            0
            if use_global
            else (
                self.window_size
                if self.window_size > 0
                else self.fallback_window_size
            )
        )

        for index, block in enumerate(self.blocks):
            if window_size == 0:
                shape = source.shape
                source_tokens = source.flatten(2).transpose(1, 2)
                target_tokens = target.flatten(2).transpose(1, 2)
                source_tokens, target_tokens = self._run_block(
                    block,
                    source_tokens,
                    target_tokens,
                )
                source = source_tokens.transpose(1, 2).reshape(shape)
                target = target_tokens.transpose(1, 2).reshape(shape)
                continue

            shift = (
                window_size // 2
                if self.shifted_windows and index % 2 == 1
                else 0
            )
            source_tokens, metadata = self._partition_windows(
                source,
                window_size,
                shift,
            )
            target_tokens, target_metadata = self._partition_windows(
                target,
                window_size,
                shift,
            )
            if metadata != target_metadata:
                raise RuntimeError("Source/target window metadata mismatch")
            source_tokens, target_tokens = self._run_block(
                block,
                source_tokens,
                target_tokens,
            )
            source = self._reverse_windows(
                source_tokens,
                metadata,
                window_size,
                shift,
            )
            target = self._reverse_windows(
                target_tokens,
                metadata,
                window_size,
                shift,
            )
        return source, target


class _FusionRefiner(nn.Module):
    """Residual refinement after one cross-scale fusion."""

    def __init__(
        self,
        channels: int,
        *,
        depth: int,
        groups: int,
    ) -> None:
        super().__init__()
        self.blocks = nn.Sequential(
            *[
                ResidualGNBlock(channels, groups=groups)
                for _ in range(max(int(depth), 1))
            ]
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.blocks(value)


class HQSCoreDeepMatchingPyramid(HQSCorePyramidEncoder):
    """Deeper bidirectional FPN with pairwise matching interaction.

    The class preserves the original ``backbone``, ``project_matching`` and
    ``project_context`` interface.  Context projections are inherited and are
    never exposed to target-frame or cross-attention features.
    """

    def __init__(
        self,
        feature_channels: Sequence[int] = (48, 96, 144, 192),
        match_channels: Sequence[int] = (96, 128, 160, 192),
        context_channels: Sequence[int] = (64, 64, 96, 96),
        blocks_per_scale: Sequence[int] = (2, 3, 4, 4),
        groups: int = 8,
        *,
        fusion_depth: int = 2,
        transformer_depth: Sequence[int] = (0, 0, 2, 4),
        transformer_heads: Sequence[int] = (4, 4, 8, 8),
        transformer_windows: Sequence[int] = (0, 0, 8, 0),
        transformer_maximum_global_tokens: int = 4096,
        transformer_fallback_window: int = 16,
        transformer_mlp_ratio: float = 4.0,
        transformer_initial_blend: float = 0.25,
        transformer_gradient_checkpointing: bool = True,
    ) -> None:
        super().__init__(
            feature_channels=feature_channels,
            match_channels=match_channels,
            context_channels=context_channels,
            blocks_per_scale=blocks_per_scale,
            groups=groups,
        )
        for name, values in {
            "transformer_depth": transformer_depth,
            "transformer_heads": transformer_heads,
            "transformer_windows": transformer_windows,
        }.items():
            if len(values) != 4:
                raise ValueError(
                    f"{name} must contain four fine-to-coarse entries"
                )
        match_by_scale = dict(zip(self.scales, self.match_channels))

        self.top_down_projections = nn.ModuleDict()
        self.top_down_refiners = nn.ModuleDict()
        for fine, coarse in ((8, 16), (4, 8), (2, 4)):
            self.top_down_projections[str(fine)] = ConvGNAct(
                match_by_scale[coarse],
                match_by_scale[fine],
                kernel_size=1,
                groups=groups,
            )
            self.top_down_refiners[str(fine)] = _FusionRefiner(
                match_by_scale[fine],
                depth=fusion_depth,
                groups=groups,
            )

        self.bottom_up_projections = nn.ModuleDict()
        self.bottom_up_refiners = nn.ModuleDict()
        for coarse, fine in ((4, 2), (8, 4), (16, 8)):
            self.bottom_up_projections[str(coarse)] = ConvGNAct(
                match_by_scale[fine],
                match_by_scale[coarse],
                kernel_size=1,
                groups=groups,
            )
            self.bottom_up_refiners[str(coarse)] = _FusionRefiner(
                match_by_scale[coarse],
                depth=fusion_depth,
                groups=groups,
            )

        self.pair_interactions = nn.ModuleDict()
        self.pair_interaction_logits = nn.ParameterDict()
        initial_logit = _logit(transformer_initial_blend)
        for scale, depth, heads, window in zip(
            self.scales,
            transformer_depth,
            transformer_heads,
            transformer_windows,
        ):
            if int(depth) <= 0:
                continue
            self.pair_interactions[str(scale)] = PairFeatureInteraction(
                match_by_scale[scale],
                depth=int(depth),
                num_heads=int(heads),
                window_size=int(window),
                fallback_window_size=int(transformer_fallback_window),
                maximum_global_tokens=int(
                    transformer_maximum_global_tokens
                ),
                mlp_ratio=float(transformer_mlp_ratio),
                shifted_windows=True,
                gradient_checkpointing=bool(
                    transformer_gradient_checkpointing
                ),
            )
            self.pair_interaction_logits[str(scale)] = nn.Parameter(
                torch.tensor(initial_logit, dtype=torch.float32)
            )

        self._initialise_new_modules()

    def _initialise_new_modules(self) -> None:
        # The base constructor has already initialised inherited modules.
        for collection in (
            self.top_down_projections,
            self.top_down_refiners,
            self.bottom_up_projections,
            self.bottom_up_refiners,
        ):
            for module in collection.modules():
                if isinstance(module, nn.Conv2d):
                    nn.init.kaiming_normal_(
                        module.weight,
                        mode="fan_out",
                        nonlinearity="relu",
                    )
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
                elif isinstance(module, nn.GroupNorm):
                    nn.init.ones_(module.weight)
                    nn.init.zeros_(module.bias)

    def project_matching(
        self,
        features: Dict[int, torch.Tensor],
    ) -> Dict[int, torch.Tensor]:
        raw = super().project_matching(features)

        top_down: Dict[int, torch.Tensor] = {16: raw[16]}
        for fine, coarse in ((8, 16), (4, 8), (2, 4)):
            coarse_value = self.top_down_projections[str(fine)](
                top_down[coarse]
            )
            coarse_value = F.interpolate(
                coarse_value,
                size=raw[fine].shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            top_down[fine] = self.top_down_refiners[str(fine)](
                (raw[fine] + coarse_value) / math.sqrt(2.0)
            )

        fused: Dict[int, torch.Tensor] = {2: top_down[2]}
        for coarse, fine in ((4, 2), (8, 4), (16, 8)):
            fine_value = F.interpolate(
                fused[fine],
                size=top_down[coarse].shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            fine_value = self.bottom_up_projections[str(coarse)](
                fine_value
            )
            fused[coarse] = self.bottom_up_refiners[str(coarse)](
                (top_down[coarse] + fine_value) / math.sqrt(2.0)
            )
        return fused

    def enhance_pair(
        self,
        source: Dict[int, torch.Tensor],
        target: Dict[int, torch.Tensor],
    ) -> Tuple[Dict[int, torch.Tensor], Dict[int, torch.Tensor]]:
        source_dtypes = {
            scale: value.dtype for scale, value in source.items()
        }
        target_dtypes = {
            scale: value.dtype for scale, value in target.items()
        }
        enhanced_source = dict(source)
        enhanced_target = dict(target)
        for scale in self.scales:
            key = str(scale)
            if key not in self.pair_interactions:
                continue
            # Attention and softmax are intentionally evaluated in float32.
            with torch.autocast(
                device_type=source[scale].device.type,
                enabled=False,
            ):
                source_base = source[scale].float()
                target_base = target[scale].float()
                source_value, target_value = self.pair_interactions[key](
                    source_base,
                    target_base,
                )
                blend = torch.sigmoid(
                    self.pair_interaction_logits[key]
                ).to(source_value)
                enhanced_source[scale] = source_base + blend * (
                    source_value - source_base
                )
                enhanced_target[scale] = target_base + blend * (
                    target_value - target_base
                )

        # Unit-direction descriptors with the original sqrt(C) activation
        # scale.  AllPairsCorrelation re-normalises these, while LocalCorrBlock
        # retains a well-conditioned cosine-like dot-product magnitude.
        for scale in self.scales:
            scale_factor = math.sqrt(float(enhanced_source[scale].shape[1]))
            enhanced_source[scale] = (
                F.normalize(
                    enhanced_source[scale].float(),
                    dim=1,
                    eps=1e-6,
                )
                * scale_factor
            ).to(dtype=source_dtypes[scale])
            enhanced_target[scale] = (
                F.normalize(
                    enhanced_target[scale].float(),
                    dim=1,
                    eps=1e-6,
                )
                * scale_factor
            ).to(dtype=target_dtypes[scale])
        return enhanced_source, enhanced_target

    def interaction_blends(self) -> Dict[int, torch.Tensor]:
        return {
            int(key): torch.sigmoid(value)
            for key, value in self.pair_interaction_logits.items()
        }


def _gather_mode(
    value: torch.Tensor,
    index: torch.Tensor,
) -> torch.Tensor:
    """Gather one mode from ``[B,K,C,H,W]`` using ``[B,1,H,W]`` indices."""
    if value.ndim != 5:
        raise ValueError(f"Expected [B,K,C,H,W], got {tuple(value.shape)}")
    gather_index = index.unsqueeze(2).expand(
        -1,
        -1,
        value.shape[2],
        -1,
        -1,
    )
    return value.gather(1, gather_index).squeeze(1)


def decode_cycle_consistent_topk(
    forward: Dict[str, torch.Tensor],
    reverse: Dict[str, torch.Tensor],
    *,
    cycle_sigma: float = 1.5,
    cycle_score_weight: float = 1.0,
    cycle_confidence_floor: float = 0.02,
    minimum_initialisation_confidence: float = 0.12,
    gate_temperature: float = 0.05,
) -> Dict[str, torch.Tensor]:
    """Rerank forward modes with reverse-cycle evidence and route the result.

    The returned ``initial_flow`` is either the complete selected displacement
    or zero in the forward pass.  A straight-through soft gate supplies useful
    confidence gradients during training without shrinking the physical flow.
    """

    hypotheses = forward["hypotheses"]
    probabilities = forward["probabilities"]
    if hypotheses.ndim != 5 or hypotheses.shape[2] != 2:
        raise ValueError(
            "forward hypotheses must be [B,K,2,H,W], got "
            f"{tuple(hypotheses.shape)}"
        )
    if probabilities.shape != (
        hypotheses.shape[0],
        hypotheses.shape[1],
        hypotheses.shape[3],
        hypotheses.shape[4],
    ):
        raise ValueError("forward probability/hypothesis shape mismatch")
    calculation_hypotheses = hypotheses.float()
    reverse_map = reverse["hypotheses"][:, 0].float()
    reverse_confidence = reverse["confidence"].float()
    cycle_sigma = max(float(cycle_sigma), 1e-4)
    confidence_floor = min(
        max(float(cycle_confidence_floor), 0.0),
        1.0,
    )

    cycle_errors: List[torch.Tensor] = []
    cycle_supports: List[torch.Tensor] = []
    for index in range(hypotheses.shape[1]):
        candidate = calculation_hypotheses[:, index]
        warped_reverse = backward_warp(
            reverse_map,
            candidate,
            padding_mode="zeros",
        )
        warped_reverse_confidence = backward_warp(
            reverse_confidence,
            candidate,
            padding_mode="zeros",
        ).clamp(0.0, 1.0)
        error = torch.sqrt(
            (candidate + warped_reverse).square().sum(
                dim=1,
                keepdim=True,
            )
            + 1e-8
        )
        confidence = torch.sqrt(
            (
                forward["confidence"].float()
                * warped_reverse_confidence
            ).clamp_min(0.0)
        )
        confidence = confidence_floor + (
            1.0 - confidence_floor
        ) * confidence
        support = (
            torch.exp(-0.5 * (error / cycle_sigma).square())
            * flow_in_bounds_mask(candidate)
            * confidence
        ).clamp(0.0, 1.0)
        cycle_errors.append(error)
        cycle_supports.append(support)

    cycle_error = torch.stack(cycle_errors, dim=1)
    cycle_support = torch.stack(cycle_supports, dim=1)
    score = probabilities.float().clamp_min(1e-9).log()
    score = score + float(cycle_score_weight) * (
        cycle_support.squeeze(2).float().clamp_min(1e-9).log()
    )
    selected_index = score.argmax(dim=1, keepdim=True)
    selected_flow = _gather_mode(hypotheses, selected_index)
    selected_cycle = _gather_mode(cycle_support, selected_index)
    selected_cycle_error = _gather_mode(cycle_error, selected_index)
    selected_probability = probabilities.gather(
        1,
        selected_index,
    )
    selected_confidence = torch.sqrt(
        (
            forward["confidence"].to(selected_cycle)
            * selected_cycle
        ).clamp_min(0.0)
    ).clamp(0.0, 1.0)

    threshold = min(
        max(float(minimum_initialisation_confidence), 0.0),
        1.0,
    )
    temperature = max(float(gate_temperature), 1e-4)
    soft_acceptance = torch.sigmoid(
        (selected_confidence - threshold) / temperature
    )
    hard_acceptance = (
        selected_confidence >= threshold
    ).to(selected_confidence.dtype)
    acceptance = (
        hard_acceptance
        + soft_acceptance
        - soft_acceptance.detach()
    )
    initial_flow = acceptance.to(selected_flow.dtype) * selected_flow

    return {
        "initial_flow": initial_flow,
        "selected_flow": selected_flow,
        "selected_index": selected_index,
        "selected_probability": selected_probability,
        "selected_confidence": selected_confidence,
        "selected_cycle_support": selected_cycle,
        "selected_cycle_error": selected_cycle_error,
        "acceptance": acceptance,
        "hard_acceptance": hard_acceptance,
        "soft_acceptance": soft_acceptance,
        "cycle_support": cycle_support,
        "cycle_error": cycle_error,
        "reverse_flow": reverse_map.to(hypotheses.dtype),
        "reverse_confidence": reverse_confidence,
    }


__all__ = [
    "HQSCoreDeepMatchingPyramid",
    "PairFeatureInteraction",
    "SymmetricCrossFrameBlock",
    "decode_cycle_consistent_topk",
]
