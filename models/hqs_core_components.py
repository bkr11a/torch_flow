"""Reusable operators for the compact, operator-structured HQSCore model.

The classes in this module deliberately expose the inverse-problem structure:

``warp -> linearise -> gate -> data update -> analytic prox -> learned prox``.

Target-image and correlation tensors terminate at the data operator.  The
proximal operator receives only the split state and source-image features.
This separation is an architectural invariant rather than a training-time
convention.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .warp import (
    backward_warp,
    convex_upsample,
    coords_grid,
    flow_in_bounds_mask,
)


def _groups(channels: int, requested: int) -> int:
    """Largest valid GroupNorm group count no greater than ``requested``."""
    for groups in range(min(int(requested), int(channels)), 0, -1):
        if channels % groups == 0:
            return groups
    return 1


class ConvGNAct(nn.Sequential):
    """Convolution, GroupNorm and SiLU used throughout HQSCore."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        *,
        stride: int = 1,
        groups: int = 8,
        activate: bool = True,
    ) -> None:
        padding = kernel_size // 2
        layers: List[nn.Module] = [
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride=stride,
                padding=padding,
                bias=False,
            ),
            nn.GroupNorm(_groups(out_channels, groups), out_channels),
        ]
        if activate:
            layers.append(nn.SiLU(inplace=True))
        super().__init__(*layers)


class ResidualGNBlock(nn.Module):
    """Small GroupNorm residual block with no change in resolution."""

    def __init__(self, channels: int, groups: int = 8) -> None:
        super().__init__()
        self.conv1 = ConvGNAct(channels, channels, groups=groups)
        self.conv2 = ConvGNAct(
            channels, channels, groups=groups, activate=False
        )
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.conv2(self.conv1(x)))


class HQSCorePyramidEncoder(nn.Module):
    """Shared Siamese feature pyramid at 1/2, 1/4, 1/8 and 1/16.

    The convolutional backbone is shared between both frames.  Matching
    projections may be applied to both frames; context projections are
    intended only for the source frame.
    """

    scales: Tuple[int, ...] = (2, 4, 8, 16)

    def __init__(
        self,
        feature_channels: Sequence[int] = (32, 64, 96, 128),
        match_channels: Sequence[int] = (64, 96, 128, 128),
        context_channels: Sequence[int] = (64, 64, 96, 96),
        blocks_per_scale: Sequence[int] = (1, 2, 2, 2),
        groups: int = 8,
    ) -> None:
        super().__init__()
        for name, values in {
            "feature_channels": feature_channels,
            "match_channels": match_channels,
            "context_channels": context_channels,
            "blocks_per_scale": blocks_per_scale,
        }.items():
            if len(values) != 4:
                raise ValueError(f"{name} must contain four entries, got {values}")

        self.feature_channels = tuple(int(v) for v in feature_channels)
        self.match_channels = tuple(int(v) for v in match_channels)
        self.context_channels = tuple(int(v) for v in context_channels)

        stages: List[nn.Module] = []
        in_channels = 3
        for channels, blocks in zip(self.feature_channels, blocks_per_scale):
            layers: List[nn.Module] = [
                ConvGNAct(
                    in_channels,
                    channels,
                    stride=2,
                    groups=groups,
                )
            ]
            layers.extend(
                ResidualGNBlock(channels, groups=groups)
                for _ in range(int(blocks))
            )
            stages.append(nn.Sequential(*layers))
            in_channels = channels
        self.stages = nn.ModuleList(stages)

        self.match_projections = nn.ModuleDict()
        self.context_projections = nn.ModuleDict()
        for scale, in_ch, match_ch, context_ch in zip(
            self.scales,
            self.feature_channels,
            self.match_channels,
            self.context_channels,
        ):
            self.match_projections[str(scale)] = ConvGNAct(
                in_ch, match_ch, kernel_size=1, groups=groups
            )
            self.context_projections[str(scale)] = ConvGNAct(
                in_ch, context_ch, kernel_size=3, groups=groups
            )

        self._initialise()

    def _initialise(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(
                    module.weight, mode="fan_out", nonlinearity="relu"
                )
            elif isinstance(module, nn.GroupNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def backbone(self, image: torch.Tensor) -> Dict[int, torch.Tensor]:
        features: Dict[int, torch.Tensor] = {}
        x = image
        for scale, stage in zip(self.scales, self.stages):
            x = stage(x)
            features[scale] = x
        return features

    def project_matching(
        self, features: Dict[int, torch.Tensor]
    ) -> Dict[int, torch.Tensor]:
        return {
            scale: self.match_projections[str(scale)](features[scale])
            for scale in self.scales
        }

    def project_context(
        self, source_features: Dict[int, torch.Tensor]
    ) -> Dict[int, torch.Tensor]:
        return {
            scale: self.context_projections[str(scale)](
                source_features[scale]
            )
            for scale in self.scales
        }


class AllPairsCorrelation:
    """Reusable all-pairs correlation with global match and RAFT lookup.

    It computes the expensive all-pairs matrix once.  The same matrix supports
    both a chunked global soft correspondence at 1/16 and recurrent local
    indexing at 1/16 or 1/8.
    """

    def __init__(
        self,
        fmap1: torch.Tensor,
        fmap2: torch.Tensor,
        *,
        num_levels: int,
        radius: int,
    ) -> None:
        if fmap1.shape != fmap2.shape:
            raise ValueError(
                "All-pairs matching requires equal feature shapes, got "
                f"{tuple(fmap1.shape)} and {tuple(fmap2.shape)}"
            )
        if num_levels < 1 or radius < 0:
            raise ValueError("num_levels must be >=1 and radius must be >=0")

        self.batch, self.channels, self.height, self.width = fmap1.shape
        self.num_levels = int(num_levels)
        self.radius = int(radius)

        f1 = F.normalize(fmap1.float(), dim=1, eps=1e-6)
        f2 = F.normalize(fmap2.float(), dim=1, eps=1e-6)
        f1_flat = f1.flatten(2).transpose(1, 2)
        f2_flat = f2.flatten(2)
        base = torch.bmm(f1_flat, f2_flat)
        # Preserve the feature dtype for the recurrent lookup.  The global
        # softmax below is explicitly evaluated in float32.
        self.base_correlation = base.to(dtype=fmap1.dtype)

        corr = self.base_correlation.reshape(
            self.batch * self.height * self.width,
            1,
            self.height,
            self.width,
        )
        self.pyramid: List[torch.Tensor] = [corr]
        # Per-level coordinate divisors.  Tracking each axis independently
        # keeps tiny smoke-test feature maps valid.
        self.level_scales: List[Tuple[int, int]] = [(1, 1)]
        scale_y = scale_x = 1
        for _ in range(1, self.num_levels):
            kernel_y = 2 if corr.shape[-2] > 1 else 1
            kernel_x = 2 if corr.shape[-1] > 1 else 1
            corr = F.avg_pool2d(
                corr,
                kernel_size=(kernel_y, kernel_x),
                stride=(kernel_y, kernel_x),
            )
            scale_y *= kernel_y
            scale_x *= kernel_x
            self.pyramid.append(corr)
            self.level_scales.append((scale_y, scale_x))

    @property
    def out_channels(self) -> int:
        return self.num_levels * (2 * self.radius + 1) ** 2

    @staticmethod
    def _normalise_grid(
        x: torch.Tensor,
        y: torch.Tensor,
        height: int,
        width: int,
    ) -> torch.Tensor:
        if width > 1:
            x = 2.0 * x / float(width - 1) - 1.0
        else:
            x = torch.zeros_like(x)
        if height > 1:
            y = 2.0 * y / float(height - 1) - 1.0
        else:
            y = torch.zeros_like(y)
        return torch.stack((x, y), dim=-1)

    def lookup(self, flow_xy: torch.Tensor) -> torch.Tensor:
        """Index a local correlation neighbourhood around ``grid + flow``."""
        b, _, h, w = flow_xy.shape
        if (b, h, w) != (self.batch, self.height, self.width):
            raise ValueError(
                "Flow/correlation grid mismatch: "
                f"flow={tuple(flow_xy.shape)}, corr={(self.batch, h, w)}"
            )

        coords = coords_grid(b, h, w, flow_xy.device).to(flow_xy.dtype)
        coords = (coords + flow_xy).permute(0, 2, 3, 1)
        r = self.radius
        delta_y, delta_x = torch.meshgrid(
            torch.arange(-r, r + 1, device=flow_xy.device, dtype=flow_xy.dtype),
            torch.arange(-r, r + 1, device=flow_xy.device, dtype=flow_xy.dtype),
            indexing="ij",
        )
        delta = torch.stack((delta_x, delta_y), dim=-1).view(
            1, 1, 2 * r + 1, 2 * r + 1, 2
        )

        outputs: List[torch.Tensor] = []
        for corr, (scale_y, scale_x) in zip(
            self.pyramid, self.level_scales
        ):
            centroid = coords.clone()
            centroid[..., 0] = centroid[..., 0] / float(scale_x)
            centroid[..., 1] = centroid[..., 1] / float(scale_y)
            centroid = centroid.reshape(b, h * w, 1, 1, 2)
            sample = centroid + delta
            grid = self._normalise_grid(
                sample[..., 0],
                sample[..., 1],
                corr.shape[-2],
                corr.shape[-1],
            ).reshape(b * h * w, 2 * r + 1, 2 * r + 1, 2)
            grid = grid.to(dtype=corr.dtype)
            values = F.grid_sample(
                corr,
                grid,
                mode="bilinear",
                padding_mode="border",
                align_corners=True,
            )
            outputs.append(values.reshape(b, h * w, -1))

        result = torch.cat(outputs, dim=-1)
        return result.transpose(1, 2).reshape(b, self.out_channels, h, w)

    def global_multimodal_match(
        self,
        *,
        num_hypotheses: int = 4,
        temperature: float = 0.07,
        query_chunk_size: int = 512,
        local_expectation_radius: int = 2,
        nms_radius: int = 3,
        reverse: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Decode distinct global modes with local sub-pixel expectations.

        A full soft expectation can fall between spatially separated modes,
        while a raw ``topk`` commonly spends all hypotheses on adjacent pixels
        from the same peak.  This decoder instead:

        1. selects one peak at a time;
        2. suppresses its spatial neighbourhood;
        3. evaluates a soft expectation only inside the selected mode.

        The exact peak displacement is returned separately as
        ``map_hypotheses``.  ``hypotheses`` contains the locally averaged
        sub-pixel displacement used by the cycle-consistent decoder.
        """
        hypotheses = int(num_hypotheses)
        temperature = max(float(temperature), 1e-4)
        chunk = max(int(query_chunk_size), 1)
        local_radius = max(int(local_expectation_radius), 0)
        suppression_radius = max(int(nms_radius), local_radius)
        matrix = (
            self.base_correlation.transpose(1, 2)
            if bool(reverse)
            else self.base_correlation
        )
        batch, queries, keys = matrix.shape
        if not 1 <= hypotheses <= keys:
            raise ValueError(
                f"num_hypotheses must be in [1,{keys}], got {hypotheses}"
            )

        coordinates = coords_grid(
            1,
            self.height,
            self.width,
            matrix.device,
        ).to(torch.float32)
        coordinates = coordinates.flatten(2).transpose(1, 2).squeeze(0)
        log_keys = math.log(max(keys, 2))

        local_y, local_x = torch.meshgrid(
            torch.arange(
                -local_radius,
                local_radius + 1,
                device=matrix.device,
            ),
            torch.arange(
                -local_radius,
                local_radius + 1,
                device=matrix.device,
            ),
            indexing="ij",
        )
        local_x = local_x.reshape(1, 1, -1)
        local_y = local_y.reshape(1, 1, -1)
        suppress_y, suppress_x = torch.meshgrid(
            torch.arange(
                -suppression_radius,
                suppression_radius + 1,
                device=matrix.device,
            ),
            torch.arange(
                -suppression_radius,
                suppression_radius + 1,
                device=matrix.device,
            ),
            indexing="ij",
        )
        suppress_x = suppress_x.reshape(1, 1, -1)
        suppress_y = suppress_y.reshape(1, 1, -1)

        hypothesis_chunks: List[torch.Tensor] = []
        map_chunks: List[torch.Tensor] = []
        probability_chunks: List[torch.Tensor] = []
        entropy_chunks: List[torch.Tensor] = []
        margin_chunks: List[torch.Tensor] = []
        peak_chunks: List[torch.Tensor] = []
        retained_chunks: List[torch.Tensor] = []

        for start in range(0, queries, chunk):
            stop = min(start + chunk, queries)
            scaled = matrix[:, start:stop].float() / temperature
            probabilities = torch.softmax(scaled, dim=-1)
            working = scaled.clone()
            negative = torch.finfo(working.dtype).min

            selected_flows: List[torch.Tensor] = []
            selected_maps: List[torch.Tensor] = []
            selected_masses: List[torch.Tensor] = []
            fallback_indices = scaled.topk(
                k=hypotheses,
                dim=-1,
            ).indices
            query_coordinates = coordinates[start:stop].view(
                1,
                stop - start,
                2,
            )

            for mode_index in range(hypotheses):
                peak_indices = working.argmax(dim=-1)
                peak_scores = working.gather(
                    -1,
                    peak_indices.unsqueeze(-1),
                ).squeeze(-1)
                exhausted = peak_scores <= negative / 2.0
                peak_indices = torch.where(
                    exhausted,
                    fallback_indices[..., mode_index],
                    peak_indices,
                )
                peak_x = peak_indices.remainder(self.width)
                peak_y = torch.div(
                    peak_indices,
                    self.width,
                    rounding_mode="floor",
                )

                neighbour_x = peak_x.unsqueeze(-1) + local_x
                neighbour_y = peak_y.unsqueeze(-1) + local_y
                neighbour_valid = (
                    (neighbour_x >= 0)
                    & (neighbour_x < self.width)
                    & (neighbour_y >= 0)
                    & (neighbour_y < self.height)
                )
                neighbour_indices = (
                    neighbour_y.clamp(0, self.height - 1) * self.width
                    + neighbour_x.clamp(0, self.width - 1)
                ).long()
                local_logits = scaled.gather(-1, neighbour_indices)
                local_logits = local_logits.masked_fill(
                    ~neighbour_valid,
                    negative,
                )
                local_weights = torch.softmax(local_logits, dim=-1)
                local_coordinates = torch.stack(
                    (neighbour_x, neighbour_y),
                    dim=-1,
                ).to(local_weights.dtype)
                expected_coordinate = (
                    local_weights.unsqueeze(-1) * local_coordinates
                ).sum(dim=-2)
                map_coordinate = coordinates[peak_indices]
                selected_flows.append(
                    expected_coordinate - query_coordinates
                )
                selected_maps.append(
                    map_coordinate - query_coordinates
                )
                local_mass = (
                    probabilities.gather(-1, neighbour_indices)
                    * neighbour_valid.to(probabilities.dtype)
                ).sum(dim=-1)
                selected_masses.append(local_mass)

                suppress_neighbour_x = (
                    peak_x.unsqueeze(-1) + suppress_x
                )
                suppress_neighbour_y = (
                    peak_y.unsqueeze(-1) + suppress_y
                )
                suppress_valid = (
                    (suppress_neighbour_x >= 0)
                    & (suppress_neighbour_x < self.width)
                    & (suppress_neighbour_y >= 0)
                    & (suppress_neighbour_y < self.height)
                )
                suppress_indices = (
                    suppress_neighbour_y.clamp(0, self.height - 1)
                    * self.width
                    + suppress_neighbour_x.clamp(0, self.width - 1)
                ).long()
                # Invalid padded offsets are redirected to the already
                # selected peak, avoiding accidental suppression of index 0.
                suppress_indices = torch.where(
                    suppress_valid,
                    suppress_indices,
                    peak_indices.unsqueeze(-1),
                )
                working.scatter_(
                    -1,
                    suppress_indices,
                    negative,
                )

            flows = torch.stack(selected_flows, dim=2)
            maps = torch.stack(selected_maps, dim=2)
            mode_masses = torch.stack(selected_masses, dim=2).clamp(
                0.0,
                1.0,
            )
            entropy = -(
                probabilities * probabilities.clamp_min(1e-9).log()
            ).sum(dim=-1) / log_keys
            peak = probabilities.max(dim=-1).values
            if hypotheses > 1:
                margin = mode_masses[..., 0] - mode_masses[..., 1]
            else:
                margin = mode_masses[..., 0]
            retained = mode_masses.sum(dim=-1).clamp(0.0, 1.0)

            hypothesis_chunks.append(flows)
            map_chunks.append(maps)
            probability_chunks.append(mode_masses)
            entropy_chunks.append(entropy)
            margin_chunks.append(margin)
            peak_chunks.append(peak)
            retained_chunks.append(retained)

        flow = torch.cat(hypothesis_chunks, dim=1)
        flow = flow.permute(0, 2, 3, 1).reshape(
            batch,
            hypotheses,
            2,
            self.height,
            self.width,
        )
        map_flow = torch.cat(map_chunks, dim=1)
        map_flow = map_flow.permute(0, 2, 3, 1).reshape(
            batch,
            hypotheses,
            2,
            self.height,
            self.width,
        )
        mode_probabilities = torch.cat(probability_chunks, dim=1)
        mode_probabilities = mode_probabilities.permute(0, 2, 1).reshape(
            batch,
            hypotheses,
            self.height,
            self.width,
        )

        def scalar_map(chunks: List[torch.Tensor]) -> torch.Tensor:
            return torch.cat(chunks, dim=1).reshape(
                batch,
                1,
                self.height,
                self.width,
            )

        entropy = scalar_map(entropy_chunks).clamp(0.0, 1.0)
        margin = scalar_map(margin_chunks)
        peak = scalar_map(peak_chunks)
        retained = scalar_map(retained_chunks)
        margin_ratio = margin / peak.clamp_min(1e-6)
        confidence = (
            0.35 * (1.0 - entropy)
            + 0.35 * margin_ratio.clamp(0.0, 1.0)
            + 0.30 * retained
        ).clamp(0.0, 1.0)
        dtype = self.base_correlation.dtype
        return {
            "hypotheses": flow.to(dtype=dtype),
            "map_hypotheses": map_flow.to(dtype=dtype),
            "logits": mode_probabilities.clamp_min(1e-9).log().to(
                dtype=dtype
            ),
            "probabilities": mode_probabilities.to(dtype=dtype),
            "confidence": confidence.to(dtype=dtype),
            "entropy": entropy.to(dtype=dtype),
            "margin": margin.to(dtype=dtype),
            "peak": peak.to(dtype=dtype),
            "retained_mass": retained.to(dtype=dtype),
        }

    def global_topk_match(
        self,
        *,
        num_hypotheses: int = 4,
        temperature: float = 0.07,
        query_chunk_size: int = 512,
        reverse: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Return global MAP correspondence modes without soft averaging.

        ``reverse=False`` decodes source-to-target hypotheses.  ``reverse=True``
        decodes target-to-source hypotheses from the transpose of the same
        all-pairs matrix, so bidirectional cycle support requires no duplicate
        correlation volume.
        """
        hypotheses = int(num_hypotheses)
        temperature = max(float(temperature), 1e-4)
        chunk = max(int(query_chunk_size), 1)
        matrix = (
            self.base_correlation.transpose(1, 2)
            if bool(reverse)
            else self.base_correlation
        )
        b, n, m = matrix.shape
        if not 1 <= hypotheses <= m:
            raise ValueError(
                f"num_hypotheses must be in [1,{m}], got {hypotheses}"
            )
        key_coords = coords_grid(
            1, self.height, self.width, matrix.device
        ).to(torch.float32)
        key_coords = key_coords.flatten(2).transpose(1, 2).squeeze(0)
        query_coords = key_coords
        log_m = math.log(max(m, 2))

        flow_chunks: List[torch.Tensor] = []
        logit_chunks: List[torch.Tensor] = []
        probability_chunks: List[torch.Tensor] = []
        confidence_chunks: List[torch.Tensor] = []
        entropy_chunks: List[torch.Tensor] = []
        margin_chunks: List[torch.Tensor] = []
        peak_chunks: List[torch.Tensor] = []
        retained_chunks: List[torch.Tensor] = []

        for start in range(0, n, chunk):
            stop = min(start + chunk, n)
            scaled = matrix[:, start:stop].float() / temperature
            probabilities = torch.softmax(scaled, dim=-1)
            top_probabilities, top_indices = probabilities.topk(
                k=hypotheses, dim=-1
            )
            top_logits = torch.gather(
                scaled, dim=-1, index=top_indices
            )
            selected_coords = key_coords[top_indices]
            query = query_coords[start:stop].view(
                1, stop - start, 1, 2
            )
            flow_chunks.append(selected_coords - query)
            logit_chunks.append(top_logits)
            probability_chunks.append(top_probabilities)

            entropy = -(
                probabilities * probabilities.clamp_min(1e-9).log()
            ).sum(dim=-1) / log_m
            peak = top_probabilities[..., 0]
            if m > 1:
                two = probabilities.topk(k=2, dim=-1).values
                margin = two[..., 0] - two[..., 1]
            else:
                margin = peak
            retained = top_probabilities.sum(dim=-1)
            confidence = (
                0.35 * (1.0 - entropy)
                + 0.35 * (margin / peak.clamp_min(1e-6))
                + 0.30 * retained
            ).clamp(0.0, 1.0)
            confidence_chunks.append(confidence)
            entropy_chunks.append(entropy)
            margin_chunks.append(margin)
            peak_chunks.append(peak)
            retained_chunks.append(retained)

        flow = torch.cat(flow_chunks, dim=1)
        flow = flow.permute(0, 2, 3, 1).reshape(
            b, hypotheses, 2, self.height, self.width
        )
        logits = torch.cat(logit_chunks, dim=1)
        logits = logits.permute(0, 2, 1).reshape(
            b, hypotheses, self.height, self.width
        )
        top_probabilities = torch.cat(probability_chunks, dim=1)
        top_probabilities = top_probabilities.permute(0, 2, 1).reshape(
            b, hypotheses, self.height, self.width
        )

        def scalar_map(chunks: List[torch.Tensor]) -> torch.Tensor:
            return torch.cat(chunks, dim=1).reshape(
                b, 1, self.height, self.width
            )

        dtype = self.base_correlation.dtype
        return {
            "hypotheses": flow.to(dtype=dtype),
            "logits": logits.to(dtype=dtype),
            "probabilities": top_probabilities.to(dtype=dtype),
            "confidence": scalar_map(confidence_chunks).to(dtype=dtype),
            "entropy": scalar_map(entropy_chunks).clamp(
                0.0, 1.0
            ).to(dtype=dtype),
            "margin": scalar_map(margin_chunks).to(dtype=dtype),
            "peak": scalar_map(peak_chunks).to(dtype=dtype),
            "retained_mass": scalar_map(retained_chunks).to(dtype=dtype),
        }

    def global_soft_match(
        self,
        *,
        temperature: float = 0.07,
        query_chunk_size: int = 512,
    ) -> Dict[str, torch.Tensor]:
        """Return expected global displacement and confidence at this grid."""
        temperature = max(float(temperature), 1e-4)
        chunk = max(int(query_chunk_size), 1)
        b, n, m = self.base_correlation.shape
        target_coords = coords_grid(
            1, self.height, self.width, self.base_correlation.device
        ).to(torch.float32)
        target_coords = target_coords.flatten(2).transpose(1, 2).squeeze(0)
        source_coords = target_coords

        expected_chunks: List[torch.Tensor] = []
        confidence_chunks: List[torch.Tensor] = []
        entropy_chunks: List[torch.Tensor] = []
        margin_chunks: List[torch.Tensor] = []
        log_m = math.log(max(m, 2))

        for start in range(0, n, chunk):
            stop = min(start + chunk, n)
            logits = self.base_correlation[:, start:stop].float()
            probabilities = torch.softmax(logits / temperature, dim=-1)
            expected_chunks.append(torch.matmul(probabilities, target_coords))

            entropy = -(
                probabilities * probabilities.clamp_min(1e-9).log()
            ).sum(dim=-1) / log_m
            if m > 1:
                top2 = probabilities.topk(k=2, dim=-1).values
                peak = top2[..., 0]
                margin = top2[..., 0] - top2[..., 1]
            else:
                peak = probabilities[..., 0]
                margin = peak
            margin_ratio = margin / peak.clamp_min(1e-6)
            confidence = 0.5 * (1.0 - entropy) + 0.5 * margin_ratio
            confidence_chunks.append(confidence.clamp(0.0, 1.0))
            entropy_chunks.append(entropy.clamp(0.0, 1.0))
            margin_chunks.append(margin)

        expected = torch.cat(expected_chunks, dim=1)
        source = source_coords.view(1, n, 2)
        flow = (expected - source).transpose(1, 2).reshape(
            b, 2, self.height, self.width
        )
        confidence = torch.cat(confidence_chunks, dim=1).reshape(
            b, 1, self.height, self.width
        )
        entropy = torch.cat(entropy_chunks, dim=1).reshape(
            b, 1, self.height, self.width
        )
        margin = torch.cat(margin_chunks, dim=1).reshape(
            b, 1, self.height, self.width
        )
        dtype = self.base_correlation.dtype
        return {
            "flow_xy": flow.to(dtype=dtype),
            "confidence": confidence.to(dtype=dtype),
            "entropy": entropy.to(dtype=dtype),
            "margin": margin.to(dtype=dtype),
        }


class CorrelationAdapter(nn.Module):
    """Project scale-dependent correlation vectors to a common embedding."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int = 64,
        groups: int = 8,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            ConvGNAct(
                in_channels, out_channels, kernel_size=1, groups=groups
            ),
            ConvGNAct(
                out_channels, out_channels, kernel_size=3, groups=groups
            ),
        )

    def forward(self, correlation: torch.Tensor) -> torch.Tensor:
        return self.net(correlation)


def spatial_gradients(image: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Centred image derivatives ``(Ix, Iy)`` with replicated boundaries."""
    if image.ndim != 4:
        raise ValueError(f"Expected image [B,C,H,W], got {tuple(image.shape)}")
    pad_x = F.pad(image, (1, 1, 0, 0), mode="replicate")
    pad_y = F.pad(image, (0, 0, 1, 1), mode="replicate")
    grad_x = 0.5 * (pad_x[..., 2:] - pad_x[..., :-2])
    grad_y = 0.5 * (pad_y[..., 2:, :] - pad_y[..., :-2, :])
    return grad_x, grad_y


def edge_magnitude(image: torch.Tensor) -> torch.Tensor:
    grad_x, grad_y = spatial_gradients(image)
    return torch.sqrt(grad_x.square() + grad_y.square() + 1e-8)


@dataclass
class Linearisation:
    residual: torch.Tensor
    grad_x: torch.Tensor
    grad_y: torch.Tensor
    in_bounds: torch.Tensor


class WarpLinearisation(nn.Module):
    """Fixed differentiable warp and first-order image linearisation."""

    def forward(
        self,
        source: torch.Tensor,
        target: torch.Tensor,
        target_grad_x: torch.Tensor,
        target_grad_y: torch.Tensor,
        flow_xy: torch.Tensor,
    ) -> Linearisation:
        warped_target = backward_warp(target, flow_xy, padding_mode="border")
        warped_grad_x = backward_warp(
            target_grad_x, flow_xy, padding_mode="border"
        )
        warped_grad_y = backward_warp(
            target_grad_y, flow_xy, padding_mode="border"
        )
        return Linearisation(
            residual=warped_target - source,
            grad_x=warped_grad_x,
            grad_y=warped_grad_y,
            in_bounds=flow_in_bounds_mask(flow_xy),
        )


class SharedValidityHead(nn.Module):
    """Small non-recurrent reliability estimator shared by all scales."""

    def __init__(
        self,
        correlation_channels: int = 64,
        hidden_channels: int = 32,
        initial_reliability: float = 0.90,
    ) -> None:
        super().__init__()
        # Scalars: |residual|, |gradient|, |w-q| and |q|/grid.
        in_channels = correlation_channels + 4
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_channels, 1, 1),
        )
        final = self.net[-1]
        nn.init.zeros_(final.weight)
        probability = min(max(float(initial_reliability), 1e-4), 1.0 - 1e-4)
        nn.init.constant_(final.bias, math.log(probability / (1.0 - probability)))

    def forward(
        self,
        correlation: torch.Tensor,
        linearisation: Linearisation,
        flow_w: torch.Tensor,
        flow_q: torch.Tensor,
    ) -> torch.Tensor:
        gradient_norm = torch.sqrt(
            linearisation.grad_x.square()
            + linearisation.grad_y.square()
            + 1e-8
        )
        coupling_norm = torch.sqrt(
            (flow_w - flow_q).square().sum(dim=1, keepdim=True) + 1e-8
        )
        flow_norm = torch.sqrt(
            flow_q.square().sum(dim=1, keepdim=True) + 1e-8
        ) / float(max(flow_q.shape[-2:]))
        inputs = torch.cat(
            (
                correlation,
                linearisation.residual.abs(),
                gradient_norm,
                coupling_norm,
                flow_norm,
            ),
            dim=1,
        )
        return torch.sigmoid(self.net(inputs))


class SeparableConv2d(nn.Module):
    """Depthwise-separable spatial convolution."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            3,
            padding=1,
            groups=in_channels,
            bias=False,
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pointwise(self.depthwise(x))


class SeparableConvGRU(nn.Module):
    """Compact spatial GRU used only by the data-consistency operator."""

    def __init__(self, input_channels: int, hidden_channels: int) -> None:
        super().__init__()
        total = input_channels + hidden_channels
        self.hidden_channels = hidden_channels
        self.reset = SeparableConv2d(total, hidden_channels)
        self.update = SeparableConv2d(total, hidden_channels)
        self.candidate = SeparableConv2d(total, hidden_channels)

    def forward(self, x: torch.Tensor, hidden: torch.Tensor) -> torch.Tensor:
        combined = torch.cat((x, hidden), dim=1)
        reset = torch.sigmoid(self.reset(combined))
        update = torch.sigmoid(self.update(combined))
        candidate = torch.tanh(
            self.candidate(torch.cat((x, reset * hidden), dim=1))
        )
        return (1.0 - update) * hidden + update * candidate


class RecurrentDataOperator(nn.Module):
    """Learned residual for the image-pair data subproblem."""

    def __init__(
        self,
        correlation_channels: int,
        context_channels: int,
        hidden_channels: int,
        groups: int = 8,
    ) -> None:
        super().__init__()
        # corr, source context, q, w-q, residual, Ix/Iy, analytic delta,
        # beta and validity.
        in_channels = (
            correlation_channels + context_channels + 2 + 2 + 1 + 2 + 2 + 1 + 1
        )
        self.input_projection = ConvGNAct(
            in_channels, hidden_channels, groups=groups
        )
        self.hidden_initialiser = nn.Conv2d(
            context_channels, hidden_channels, 3, padding=1
        )
        self.gru = SeparableConvGRU(hidden_channels, hidden_channels)
        self.flow_head = nn.Sequential(
            ConvGNAct(
                hidden_channels, hidden_channels, groups=groups
            ),
            nn.Conv2d(hidden_channels, 2, 3, padding=1),
        )
        nn.init.zeros_(self.flow_head[-1].weight)
        nn.init.zeros_(self.flow_head[-1].bias)

    def initialise_hidden(self, source_context: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.hidden_initialiser(source_context))

    def forward(
        self,
        correlation: torch.Tensor,
        source_context: torch.Tensor,
        flow_w: torch.Tensor,
        flow_q: torch.Tensor,
        linearisation: Linearisation,
        analytic_delta: torch.Tensor,
        beta_map: torch.Tensor,
        validity: torch.Tensor,
        hidden: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        inputs = torch.cat(
            (
                correlation,
                source_context,
                flow_q,
                flow_w - flow_q,
                linearisation.residual,
                linearisation.grad_x,
                linearisation.grad_y,
                analytic_delta,
                beta_map,
                validity,
            ),
            dim=1,
        )
        projected = self.input_projection(inputs)
        candidate = self.gru(projected, hidden)
        # Target-derived recurrent state is gated by exactly the same validity
        # used by the data correction.  Invalid locations retain their
        # source-context initial state.
        hidden_gate = validity.to(dtype=candidate.dtype)
        hidden_next = hidden + hidden_gate * (candidate - hidden)
        return self.flow_head(hidden_next), hidden_next


class PositiveSchedule(nn.Module):
    """Short positive parameter schedule shared across recurrent uses."""

    def __init__(
        self,
        length: int,
        initial_value: float,
        minimum: float,
    ) -> None:
        super().__init__()
        self.minimum = float(minimum)
        target = max(float(initial_value) - self.minimum, 1e-6)
        raw = math.log(math.expm1(target))
        self.raw = nn.Parameter(torch.full((int(length),), raw))

    def forward(self, index: int) -> torch.Tensor:
        index = min(max(int(index), 0), self.raw.numel() - 1)
        return F.softplus(self.raw[index]) + self.minimum


class EdgeAwareJacobiProx(nn.Module):
    """Analytic edge-aware quadratic proximal anchor."""

    def __init__(self, edge_alpha: float = 10.0) -> None:
        super().__init__()
        self.edge_alpha = float(edge_alpha)

    def forward(
        self,
        flow_w: torch.Tensor,
        source_gray: torch.Tensor,
        beta: torch.Tensor,
        regularisation: torch.Tensor,
        sweeps: int,
    ) -> torch.Tensor:
        if int(sweeps) <= 0:
            return flow_w

        right_weight = torch.zeros_like(source_gray)
        right_weight[..., :-1] = torch.exp(
            -self.edge_alpha
            * (source_gray[..., 1:] - source_gray[..., :-1]).abs()
        )
        left_weight = torch.zeros_like(source_gray)
        left_weight[..., 1:] = right_weight[..., :-1]

        down_weight = torch.zeros_like(source_gray)
        down_weight[..., :-1, :] = torch.exp(
            -self.edge_alpha
            * (source_gray[..., 1:, :] - source_gray[..., :-1, :]).abs()
        )
        up_weight = torch.zeros_like(source_gray)
        up_weight[..., 1:, :] = down_weight[..., :-1, :]

        weight_sum = right_weight + left_weight + down_weight + up_weight
        denominator = beta + regularisation * weight_sum
        q = flow_w
        for _ in range(int(sweeps)):
            right = F.pad(q[..., 1:], (0, 1, 0, 0))
            left = F.pad(q[..., :-1], (1, 0, 0, 0))
            down = F.pad(q[..., 1:, :], (0, 0, 0, 1))
            up = F.pad(q[..., :-1, :], (0, 0, 1, 0))
            neighbour_sum = (
                right_weight * right
                + left_weight * left
                + down_weight * down
                + up_weight * up
            )
            q = (
                beta * flow_w + regularisation * neighbour_sum
            ) / denominator.clamp_min(1e-6)
        return q


class SourceConditionedProxResidual(nn.Module):
    """Bounded learned residual for the proximal subproblem.

    There is intentionally no target-image or correlation argument.  The
    learned prior is conditioned on the split state, source context and source
    edge structure only.
    """

    def __init__(
        self,
        context_channels: int,
        hidden_channels: int,
        groups: int = 8,
    ) -> None:
        super().__init__()
        # q_bar, w-q_bar, q_previous, context, source edge, beta, lambda.
        in_channels = 2 + 2 + 2 + context_channels + 1 + 1 + 1
        self.net = nn.Sequential(
            ConvGNAct(in_channels, hidden_channels, groups=groups),
            ConvGNAct(hidden_channels, hidden_channels, groups=groups),
            nn.Conv2d(hidden_channels, 2, 3, padding=1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(
        self,
        q_bar: torch.Tensor,
        flow_w: torch.Tensor,
        q_previous: torch.Tensor,
        source_context: torch.Tensor,
        source_edge: torch.Tensor,
        beta_map: torch.Tensor,
        lambda_map: torch.Tensor,
        max_delta: float,
    ) -> torch.Tensor:
        inputs = torch.cat(
            (
                q_bar,
                flow_w - q_bar,
                q_previous,
                source_context,
                source_edge,
                beta_map,
                lambda_map,
            ),
            dim=1,
        )
        residual = float(max_delta) * torch.tanh(self.net(inputs))
        return q_bar + residual


def _bounded_vector(vector: torch.Tensor, maximum: float) -> torch.Tensor:
    maximum = max(float(maximum), 1e-6)
    norm = torch.sqrt(vector.square().sum(dim=1, keepdim=True) + 1e-12)
    ratio = norm / maximum
    scale = torch.where(
        ratio > 1e-4,
        torch.tanh(ratio) / ratio,
        1.0 - ratio.square() / 3.0,
    )
    return vector * scale


@dataclass
class HQSState:
    """Interpretable recurrent state for one resolution scale."""

    w: torch.Tensor
    q: torch.Tensor
    hidden: torch.Tensor


@dataclass
class HQSIterationOutput:
    state: HQSState
    reliability: torch.Tensor
    validity: torch.Tensor
    analytic_delta: torch.Tensor
    learned_delta: torch.Tensor
    data_delta: torch.Tensor
    proximal_anchor: torch.Tensor
    beta: torch.Tensor
    regularisation: torch.Tensor


def weighted_analytic_data_delta(
    linearisation: Linearisation,
    validity: torch.Tensor,
    beta_map: torch.Tensor,
) -> torch.Tensor:
    """Solve the confidence-weighted linearised data subproblem.

    For each pixel this returns the unique minimiser of

        0.5 * m * (r + g^T delta)^2 + 0.5 * beta * ||delta||_2^2,

    where ``m`` is ``validity``, ``r`` is the warped photometric residual,
    and ``g = (Ix, Iy)`` is the warped target-image gradient.  The returned
    increment is

        delta = -m * g * r / (beta + m * ||g||_2^2).

    ``beta_map`` is strictly positive under ``PositiveSchedule``.  The small
    denominator clamp is retained for mixed-precision numerical safety.
    """
    weight = validity.to(
        device=linearisation.residual.device,
        dtype=linearisation.residual.dtype,
    ).clamp(0.0, 1.0)
    beta_map = beta_map.to(
        device=linearisation.residual.device,
        dtype=linearisation.residual.dtype,
    )
    gradient_norm_squared = (
        linearisation.grad_x.square() + linearisation.grad_y.square()
    )
    denominator = (
        beta_map + weight * gradient_norm_squared
    ).clamp_min(1e-6)
    step = -weight * linearisation.residual / denominator
    return torch.cat(
        (
            linearisation.grad_x * step,
            linearisation.grad_y * step,
        ),
        dim=1,
    )


class StructuredHQSCell(nn.Module):
    """One reusable ``prox o data o validity o warp`` HQS iteration."""

    def __init__(
        self,
        *,
        correlation_channels: int,
        context_channels: int,
        hidden_channels: int,
        max_iterations: int,
        prior_hidden_channels: int = 64,
        groups: int = 8,
        beta_initial: float = 0.10,
        beta_minimum: float = 0.01,
        lambda_initial: float = 0.08,
        lambda_minimum: float = 0.001,
        edge_alpha: float = 10.0,
        analytic_weight: float = 1.0,
        learned_data_weight: float = 1.0,
        analytic_validity_mode: str = "post_gate",
    ) -> None:
        super().__init__()
        self.analytic_weight = float(analytic_weight)
        self.learned_data_weight = float(learned_data_weight)
        self.analytic_validity_mode = str(analytic_validity_mode).lower()
        if self.analytic_validity_mode not in {
            "post_gate",
            "weighted_solve",
        }:
            raise ValueError(
                "analytic_validity_mode must be 'post_gate' or "
                f"'weighted_solve', got {analytic_validity_mode!r}"
            )
        self.warp_linearisation = WarpLinearisation()
        self.data_operator = RecurrentDataOperator(
            correlation_channels,
            context_channels,
            hidden_channels,
            groups=groups,
        )
        self.analytic_prox = EdgeAwareJacobiProx(edge_alpha=edge_alpha)
        self.learned_prox = SourceConditionedProxResidual(
            context_channels,
            prior_hidden_channels,
            groups=groups,
        )
        self.beta_schedule = PositiveSchedule(
            max_iterations, beta_initial, beta_minimum
        )
        self.lambda_schedule = PositiveSchedule(
            max_iterations, lambda_initial, lambda_minimum
        )

    def initialise_state(
        self,
        source_context: torch.Tensor,
        flow_q: torch.Tensor,
    ) -> HQSState:
        hidden = self.data_operator.initialise_hidden(source_context)
        return HQSState(w=flow_q, q=flow_q, hidden=hidden)

    def forward(
        self,
        *,
        source_gray: torch.Tensor,
        target_gray: torch.Tensor,
        target_grad_x: torch.Tensor,
        target_grad_y: torch.Tensor,
        correlation: torch.Tensor,
        source_context: torch.Tensor,
        state: HQSState,
        validity_head: SharedValidityHead,
        iteration: int,
        jacobi_sweeps: int,
        max_data_delta: float,
        max_prox_delta: float,
    ) -> HQSIterationOutput:
        beta = self.beta_schedule(iteration).to(
            device=state.q.device, dtype=state.q.dtype
        )
        regularisation = self.lambda_schedule(iteration).to(
            device=state.q.device, dtype=state.q.dtype
        )
        beta_map = beta.view(1, 1, 1, 1).expand(
            state.q.shape[0], 1, *state.q.shape[-2:]
        )
        lambda_map = regularisation.view(1, 1, 1, 1).expand_as(beta_map)

        linearisation = self.warp_linearisation(
            source_gray,
            target_gray,
            target_grad_x,
            target_grad_y,
            state.q,
        )
        reliability = validity_head(
            correlation, linearisation, state.w, state.q
        )
        validity = linearisation.in_bounds * reliability
        if self.analytic_validity_mode == "weighted_solve":
            analytic_delta = weighted_analytic_data_delta(
                linearisation,
                validity,
                beta_map,
            )
        else:
            denominator = (
                linearisation.grad_x.square()
                + linearisation.grad_y.square()
                + beta_map
            ).clamp_min(1e-6)
            analytic_delta = torch.cat(
                (
                    -linearisation.grad_x
                    * linearisation.residual
                    / denominator,
                    -linearisation.grad_y
                    * linearisation.residual
                    / denominator,
                ),
                dim=1,
            )
        # This is a stability constraint on the exact analytic anchor, not
        # part of the unconstrained quadratic solve.
        analytic_delta = _bounded_vector(analytic_delta, max_data_delta)

        learned_delta, hidden_next = self.data_operator(
            correlation,
            source_context,
            state.w,
            state.q,
            linearisation,
            analytic_delta,
            beta_map,
            validity,
            state.hidden,
        )
        learned_proposal = (
            float(max_data_delta) * torch.tanh(learned_delta)
        )
        if self.analytic_validity_mode == "weighted_solve":
            # The analytic term already contains validity in both numerator
            # and denominator.  Gate only the learned residual here; applying
            # validity to the combined proposal would double-weight the
            # analytic correction.
            proposal = (
                self.analytic_weight * analytic_delta
                + self.learned_data_weight * validity * learned_proposal
            )
            data_delta = _bounded_vector(proposal, max_data_delta)
            flow_w_next = state.q + data_delta
        else:
            # Legacy HQSCore behaviour retained for a controlled ablation and
            # compatibility with checkpoints trained using 06_hqs_core.yaml.
            proposal = (
                self.analytic_weight * analytic_delta
                + self.learned_data_weight * learned_proposal
            )
            data_delta = _bounded_vector(proposal, max_data_delta)
            flow_w_next = state.q + validity * data_delta

        # Exact causal invariant in both modes:
        # validity == 0 implies flow_w_next == state.q.

        proximal_anchor = self.analytic_prox(
            flow_w_next,
            source_gray,
            beta,
            regularisation,
            sweeps=jacobi_sweeps,
        )
        q_next = self.learned_prox(
            proximal_anchor,
            flow_w_next,
            state.q,
            source_context,
            edge_magnitude(source_gray),
            beta_map,
            lambda_map,
            max_delta=max_prox_delta,
        )
        return HQSIterationOutput(
            state=HQSState(w=flow_w_next, q=q_next, hidden=hidden_next),
            reliability=reliability,
            validity=validity,
            analytic_delta=analytic_delta,
            learned_delta=learned_delta,
            data_delta=data_delta,
            proximal_anchor=proximal_anchor,
            beta=beta,
            regularisation=regularisation,
        )


class SourceGuidedConvexUpsampler(nn.Module):
    """Source-conditioned 1/2-to-full convex flow upsampler."""

    def __init__(
        self,
        context_channels: int,
        hidden_channels: int = 64,
        groups: int = 8,
        rate: int = 2,
    ) -> None:
        super().__init__()
        self.rate = int(rate)
        self.mask_head = nn.Sequential(
            ConvGNAct(
                context_channels + 2 + 1,
                hidden_channels,
                groups=groups,
            ),
            ConvGNAct(
                hidden_channels, hidden_channels, groups=groups
            ),
            nn.Conv2d(hidden_channels, 9 * self.rate * self.rate, 1),
        )
        # Uniform convex interpolation at initialisation.
        nn.init.zeros_(self.mask_head[-1].weight)
        nn.init.zeros_(self.mask_head[-1].bias)

    def forward(
        self,
        flow_half: torch.Tensor,
        source_context: torch.Tensor,
        source_gray: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        mask_logits = self.mask_head(
            torch.cat(
                (source_context, flow_half, edge_magnitude(source_gray)), dim=1
            )
        )
        return (
            convex_upsample(flow_half, mask_logits, rate=self.rate),
            mask_logits,
        )


__all__ = [
    "AllPairsCorrelation",
    "CorrelationAdapter",
    "EdgeAwareJacobiProx",
    "HQSCorePyramidEncoder",
    "HQSIterationOutput",
    "HQSState",
    "SharedValidityHead",
    "SourceConditionedProxResidual",
    "SourceGuidedConvexUpsampler",
    "StructuredHQSCell",
    "WarpLinearisation",
    "edge_magnitude",
    "spatial_gradients",
    "weighted_analytic_data_delta",
]
