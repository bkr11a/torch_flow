"""GMFlow-style correspondence operators for :class:`HQSCore`.

This module is a clean PyTorch implementation of the architecture described in
Xu et al., *GMFlow: Learning Optical Flow via Global Matching*, CVPR 2022.  It
reproduces the relevant operator sequence rather than importing the reference
repository:

``1/8 CNN features -> sine position -> six self/cross blocks -> matching``.

The source-conditioned HQS context and proximal paths remain outside this
module.  Forward/backward consistency is returned as measurement geometry; it
must not be passed to the source-only proximal operator.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Dict, Mapping, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .warp import backward_warp, flow_in_bounds_mask


class GMFlowResidualBlock(nn.Module):
    """Two-convolution residual block used by the GMFlow feature encoder."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int,
        *,
        released_weights_compatible: bool = False,
    ) -> None:
        super().__init__()
        self.released_weights_compatible = bool(
            released_weights_compatible
        )
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=int(stride),
            padding=1,
            bias=False,
        )
        self.norm1 = nn.InstanceNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            bias=False,
        )
        self.norm2 = nn.InstanceNorm2d(out_channels)
        self.downsample: Optional[nn.Module]
        if int(stride) != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=int(stride),
                ),
                nn.InstanceNorm2d(out_channels),
            )
        else:
            self.downsample = None

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        residual = value if self.downsample is None else self.downsample(value)
        value = F.relu(self.norm1(self.conv1(value)), inplace=True)
        value = self.norm2(self.conv2(value))
        # The released GMFlow backbone applies ReLU both before and after the
        # residual addition.  Keep the historical local implementation as the
        # default so existing scratch-trained checkpoints retain their exact
        # graph, while enabling the released graph for official weights.
        if self.released_weights_compatible:
            value = F.relu(value, inplace=True)
        return F.relu(residual + value, inplace=True)


class GMFlowMatchingBackbone(nn.Module):
    """Dedicated single-scale GMFlow CNN producing 128-D features at 1/8."""

    def __init__(
        self,
        output_channels: int = 128,
        *,
        released_weights_compatible: bool = False,
    ) -> None:
        super().__init__()
        self.output_channels = int(output_channels)
        self.released_weights_compatible = bool(
            released_weights_compatible
        )
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.stage2 = nn.Sequential(
            GMFlowResidualBlock(
                64,
                64,
                stride=1,
                released_weights_compatible=self.released_weights_compatible,
            ),
            GMFlowResidualBlock(
                64,
                64,
                stride=1,
                released_weights_compatible=self.released_weights_compatible,
            ),
        )
        self.stage4 = nn.Sequential(
            GMFlowResidualBlock(
                64,
                96,
                stride=2,
                released_weights_compatible=self.released_weights_compatible,
            ),
            GMFlowResidualBlock(
                96,
                96,
                stride=1,
                released_weights_compatible=self.released_weights_compatible,
            ),
        )
        self.stage8 = nn.Sequential(
            GMFlowResidualBlock(
                96,
                128,
                stride=2,
                released_weights_compatible=self.released_weights_compatible,
            ),
            GMFlowResidualBlock(
                128,
                128,
                stride=1,
                released_weights_compatible=self.released_weights_compatible,
            ),
        )
        self.projection = nn.Conv2d(128, self.output_channels, kernel_size=1)
        self._initialise()

    def _initialise(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(
                    module.weight,
                    mode="fan_out",
                    nonlinearity="relu",
                )
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        value = self.stem(image)
        value = self.stage2(value)
        value = self.stage4(value)
        value = self.stage8(value)
        return self.projection(value)


def _split_feature(value: torch.Tensor, num_splits: int) -> torch.Tensor:
    batch, channels, height, width = value.shape
    splits = int(num_splits)
    if height % splits != 0 or width % splits != 0:
        raise ValueError(
            f"Feature grid {(height, width)} is not divisible by {splits}"
        )
    value = value.view(
        batch,
        channels,
        splits,
        height // splits,
        splits,
        width // splits,
    )
    return value.permute(0, 2, 4, 1, 3, 5).reshape(
        batch * splits * splits,
        channels,
        height // splits,
        width // splits,
    )


def _merge_feature(value: torch.Tensor, num_splits: int) -> torch.Tensor:
    split_batch, channels, height, width = value.shape
    splits = int(num_splits)
    batch = split_batch // (splits * splits)
    value = value.view(batch, splits, splits, channels, height, width)
    return value.permute(0, 3, 1, 4, 2, 5).contiguous().view(
        batch,
        channels,
        splits * height,
        splits * width,
    )


def _sine_position_embedding(
    reference: torch.Tensor,
    channels: int,
    temperature: float = 10000.0,
) -> torch.Tensor:
    """Return the normalised two-dimensional sine encoding used by GMFlow."""
    batch, _, height, width = reference.shape
    if int(channels) % 4 != 0:
        raise ValueError("GMFlow positional channels must be divisible by four")
    half = int(channels) // 2
    mask = torch.ones(
        batch,
        height,
        width,
        device=reference.device,
        dtype=torch.float32,
    )
    y_embed = mask.cumsum(1)
    x_embed = mask.cumsum(2)
    scale = 2.0 * math.pi
    y_embed = y_embed / (y_embed[:, -1:, :] + 1e-6) * scale
    x_embed = x_embed / (x_embed[:, :, -1:] + 1e-6) * scale
    dim = torch.arange(half, device=reference.device, dtype=torch.float32)
    dim = float(temperature) ** (2.0 * torch.div(dim, 2, rounding_mode="floor") / half)

    pos_x = x_embed.unsqueeze(-1) / dim
    pos_y = y_embed.unsqueeze(-1) / dim
    pos_x = torch.stack(
        (pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()),
        dim=-1,
    ).flatten(3)
    pos_y = torch.stack(
        (pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()),
        dim=-1,
    ).flatten(3)
    position = torch.cat((pos_y, pos_x), dim=-1).permute(0, 3, 1, 2)
    return position.to(dtype=reference.dtype)


def _add_split_position(
    source: torch.Tensor,
    target: torch.Tensor,
    num_splits: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    splits = int(num_splits)
    if splits > 1:
        source_windows = _split_feature(source, splits)
        target_windows = _split_feature(target, splits)
        position = _sine_position_embedding(
            source_windows,
            source_windows.shape[1],
        )
        return (
            _merge_feature(source_windows + position, splits),
            _merge_feature(target_windows + position, splits),
        )
    position = _sine_position_embedding(source, source.shape[1])
    return source + position, target + position


def _shifted_window_mask(
    height: int,
    width: int,
    num_splits: int,
    device: torch.device,
) -> Optional[torch.Tensor]:
    splits = int(num_splits)
    if splits <= 1:
        return None
    window_h = height // splits
    window_w = width // splits
    shift_h = window_h // 2
    shift_w = window_w // 2
    if shift_h == 0 or shift_w == 0:
        return None

    regions = torch.zeros((1, height, width, 1), device=device)
    h_slices = (
        slice(0, -window_h),
        slice(-window_h, -shift_h),
        slice(-shift_h, None),
    )
    w_slices = (
        slice(0, -window_w),
        slice(-window_w, -shift_w),
        slice(-shift_w, None),
    )
    index = 0
    for h_slice in h_slices:
        for w_slice in w_slices:
            regions[:, h_slice, w_slice, :] = index
            index += 1
    windows = regions.view(
        1,
        splits,
        window_h,
        splits,
        window_w,
        1,
    ).permute(0, 1, 3, 2, 4, 5)
    windows = windows.reshape(splits * splits, window_h * window_w)
    mask = windows.unsqueeze(1) - windows.unsqueeze(2)
    return mask.masked_fill(mask != 0, -100.0).masked_fill(mask == 0, 0.0)


def _single_head_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    height: int,
    width: int,
    num_splits: int,
    shifted: bool,
    shifted_mask: Optional[torch.Tensor],
) -> torch.Tensor:
    batch, tokens, channels = query.shape
    if tokens != height * width:
        raise ValueError("Transformer token count does not match its feature grid")
    splits = int(num_splits)
    if splits <= 1:
        scores = torch.matmul(
            query.float(),
            key.float().transpose(1, 2),
        ) / math.sqrt(channels)
        probabilities = torch.softmax(scores, dim=-1)
        return torch.matmul(probabilities, value.float()).to(query.dtype)

    window_h = height // splits
    window_w = width // splits
    shift_h = window_h // 2 if shifted else 0
    shift_w = window_w // 2 if shifted else 0

    def window_tokens(tensor: torch.Tensor) -> torch.Tensor:
        tensor = tensor.view(batch, height, width, channels)
        if shift_h or shift_w:
            tensor = torch.roll(
                tensor,
                shifts=(-shift_h, -shift_w),
                dims=(1, 2),
            )
        return tensor.view(
            batch,
            splits,
            window_h,
            splits,
            window_w,
            channels,
        ).permute(0, 1, 3, 2, 4, 5).reshape(
            batch * splits * splits,
            window_h * window_w,
            channels,
        )

    query_windows = window_tokens(query)
    key_windows = window_tokens(key)
    value_windows = window_tokens(value)
    scores = torch.matmul(
        query_windows.float(),
        key_windows.float().transpose(1, 2),
    ) / math.sqrt(channels)
    if shifted and shifted_mask is not None:
        scores = scores + shifted_mask.repeat(batch, 1, 1)
    probabilities = torch.softmax(scores, dim=-1)
    output = torch.matmul(probabilities, value_windows.float()).to(query.dtype)
    output = output.view(
        batch,
        splits,
        splits,
        window_h,
        window_w,
        channels,
    ).permute(0, 1, 3, 2, 4, 5).reshape(
        batch,
        height,
        width,
        channels,
    )
    if shift_h or shift_w:
        output = torch.roll(output, shifts=(shift_h, shift_w), dims=(1, 2))
    return output.reshape(batch, tokens, channels)


class GMFlowTransformerLayer(nn.Module):
    """GMFlow single-head attention layer with optional concatenative FFN."""

    def __init__(
        self,
        channels: int,
        *,
        with_ffn: bool,
        ffn_expansion: int,
        shifted: bool,
    ) -> None:
        super().__init__()
        self.channels = int(channels)
        self.with_ffn = bool(with_ffn)
        self.shifted = bool(shifted)
        self.query = nn.Linear(channels, channels, bias=False)
        self.key = nn.Linear(channels, channels, bias=False)
        self.value = nn.Linear(channels, channels, bias=False)
        self.merge = nn.Linear(channels, channels, bias=False)
        self.norm1 = nn.LayerNorm(channels)
        if self.with_ffn:
            concatenated = 2 * channels
            self.ffn = nn.Sequential(
                nn.Linear(
                    concatenated,
                    concatenated * int(ffn_expansion),
                    bias=False,
                ),
                nn.GELU(),
                nn.Linear(
                    concatenated * int(ffn_expansion),
                    channels,
                    bias=False,
                ),
            )
            self.norm2 = nn.LayerNorm(channels)

    def forward(
        self,
        source: torch.Tensor,
        target: torch.Tensor,
        *,
        height: int,
        width: int,
        num_splits: int,
        shifted_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        message = _single_head_attention(
            self.query(source),
            self.key(target),
            self.value(target),
            height=height,
            width=width,
            num_splits=num_splits,
            shifted=self.shifted,
            shifted_mask=shifted_mask,
        )
        message = self.norm1(self.merge(message))
        if self.with_ffn:
            message = self.norm2(self.ffn(torch.cat((source, message), dim=-1)))
        return source + message


class GMFlowTransformerBlock(nn.Module):
    """One self-attention followed by cross-attention/FFN block."""

    def __init__(
        self,
        channels: int,
        *,
        ffn_expansion: int,
        shifted: bool,
    ) -> None:
        super().__init__()
        self.self_attention = GMFlowTransformerLayer(
            channels,
            with_ffn=False,
            ffn_expansion=ffn_expansion,
            shifted=shifted,
        )
        self.cross_attention = GMFlowTransformerLayer(
            channels,
            with_ffn=True,
            ffn_expansion=ffn_expansion,
            shifted=shifted,
        )

    def forward(
        self,
        source: torch.Tensor,
        target: torch.Tensor,
        *,
        height: int,
        width: int,
        num_splits: int,
        shifted_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        source = self.self_attention(
            source,
            source,
            height=height,
            width=width,
            num_splits=num_splits,
            shifted_mask=shifted_mask,
        )
        return self.cross_attention(
            source,
            target,
            height=height,
            width=width,
            num_splits=num_splits,
            shifted_mask=shifted_mask,
        )


class GMFlowFeatureTransformer(nn.Module):
    """Weight-shared symmetric GMFlow feature transformer."""

    def __init__(
        self,
        channels: int = 128,
        depth: int = 6,
        ffn_expansion: int = 4,
        num_splits: int = 2,
        gradient_checkpointing: bool = True,
    ) -> None:
        super().__init__()
        self.channels = int(channels)
        self.num_splits = max(int(num_splits), 1)
        self.gradient_checkpointing = bool(gradient_checkpointing)
        self.blocks = nn.ModuleList(
            GMFlowTransformerBlock(
                self.channels,
                ffn_expansion=int(ffn_expansion),
                shifted=(index % 2 == 1),
            )
            for index in range(int(depth))
        )
        for parameter in self.parameters():
            if parameter.ndim > 1:
                nn.init.xavier_uniform_(parameter)

    def forward(
        self,
        source: torch.Tensor,
        target: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if source.shape != target.shape:
            raise ValueError("GMFlow pair features must have identical shapes")
        if source.shape[1] != self.channels:
            raise ValueError(
                f"Expected {self.channels} feature channels, got {source.shape[1]}"
            )
        original_h, original_w = source.shape[-2:]
        pad_h = (-original_h) % self.num_splits
        pad_w = (-original_w) % self.num_splits
        if pad_h or pad_w:
            source = F.pad(source, (0, pad_w, 0, pad_h), mode="replicate")
            target = F.pad(target, (0, pad_w, 0, pad_h), mode="replicate")
        height, width = source.shape[-2:]
        source, target = _add_split_position(
            source,
            target,
            self.num_splits,
        )
        source_tokens = source.flatten(2).transpose(1, 2)
        target_tokens = target.flatten(2).transpose(1, 2)
        paired_source = torch.cat((source_tokens, target_tokens), dim=0)
        paired_target = torch.cat((target_tokens, source_tokens), dim=0)
        shifted_mask = _shifted_window_mask(
            height,
            width,
            self.num_splits,
            source.device,
        )

        for block in self.blocks:
            def apply_block(
                value: torch.Tensor,
                other: torch.Tensor,
                module: GMFlowTransformerBlock = block,
            ) -> torch.Tensor:
                return module(
                    value,
                    other,
                    height=height,
                    width=width,
                    num_splits=self.num_splits,
                    shifted_mask=shifted_mask,
                )

            if (
                self.gradient_checkpointing
                and self.training
                and paired_source.requires_grad
            ):
                paired_source = checkpoint(
                    apply_block,
                    paired_source,
                    paired_target,
                    use_reentrant=False,
                )
            else:
                paired_source = apply_block(paired_source, paired_target)
            halves = paired_source.chunk(2, dim=0)
            paired_target = torch.cat((halves[1], halves[0]), dim=0)

        source_tokens, target_tokens = paired_source.chunk(2, dim=0)
        batch = source_tokens.shape[0]
        source = source_tokens.view(
            batch,
            height,
            width,
            self.channels,
        ).permute(0, 3, 1, 2).contiguous()
        target = target_tokens.view(
            batch,
            height,
            width,
            self.channels,
        ).permute(0, 3, 1, 2).contiguous()
        return (
            source[..., :original_h, :original_w],
            target[..., :original_h, :original_w],
        )


class GMFlowFeatureFlowAttention(nn.Module):
    """Global source self-attention that propagates a decoded flow field."""

    def __init__(self, channels: int = 128, query_chunk_size: int = 512) -> None:
        super().__init__()
        self.channels = int(channels)
        self.query_chunk_size = max(int(query_chunk_size), 1)
        self.query = nn.Linear(self.channels, self.channels)
        self.key = nn.Linear(self.channels, self.channels)
        for parameter in self.parameters():
            if parameter.ndim > 1:
                nn.init.xavier_uniform_(parameter)

    def forward(
        self,
        feature: torch.Tensor,
        flow_xy: torch.Tensor,
    ) -> torch.Tensor:
        if feature.shape[0] != flow_xy.shape[0] or feature.shape[-2:] != flow_xy.shape[-2:]:
            raise ValueError("Feature-flow propagation requires aligned tensors")
        batch, channels, height, width = feature.shape
        tokens = height * width
        feature_tokens = feature.flatten(2).transpose(1, 2)
        query = self.query(feature_tokens)
        # This ordering matches the released GMFlow propagation module.
        key = self.key(query)
        value = flow_xy.flatten(2).transpose(1, 2)
        outputs = []
        for start in range(0, tokens, self.query_chunk_size):
            stop = min(start + self.query_chunk_size, tokens)
            scores = torch.matmul(
                query[:, start:stop].float(),
                key.float().transpose(1, 2),
            ) / math.sqrt(channels)
            probability = torch.softmax(scores, dim=-1)
            outputs.append(torch.matmul(probability, value.float()))
        propagated = torch.cat(outputs, dim=1).to(flow_xy.dtype)
        return propagated.transpose(1, 2).reshape(batch, 2, height, width)


@dataclass(frozen=True)
class GMFlowCheckpointLoadReport:
    """Auditable result of importing an official single-scale checkpoint."""

    path: str
    loaded_tensor_count: int
    loaded_parameter_count: int
    ignored_checkpoint_keys: Tuple[str, ...]

    def summary(self) -> str:
        ignored = len(self.ignored_checkpoint_keys)
        return (
            f"loaded {self.loaded_tensor_count} tensors / "
            f"{self.loaded_parameter_count:,} values; "
            f"ignored {ignored} official full-model tensors"
        )


def _strip_data_parallel_prefix(key: str) -> str:
    return key[7:] if key.startswith("module.") else key


def _official_gmflow_key_to_local(key: str) -> Optional[str]:
    """Translate one released GMFlow state-dict key to this front end."""

    key = _strip_data_parallel_prefix(str(key))
    if key.startswith("backbone."):
        suffix = key[len("backbone.") :]
        backbone_prefixes = {
            "conv1.": "stem.0.",
            "layer1.": "stage2.",
            "layer2.": "stage4.",
            "layer3.": "stage8.",
            "conv2.": "projection.",
        }
        for official, local in backbone_prefixes.items():
            if suffix.startswith(official):
                return "backbone." + local + suffix[len(official) :]
        return None

    if key.startswith("transformer.layers."):
        suffix = key[len("transformer.layers.") :]
        suffix = suffix.replace(".self_attn.", ".self_attention.")
        suffix = suffix.replace(
            ".cross_attn_ffn.",
            ".cross_attention.",
        )
        suffix = suffix.replace(".q_proj.", ".query.")
        suffix = suffix.replace(".k_proj.", ".key.")
        suffix = suffix.replace(".v_proj.", ".value.")
        suffix = suffix.replace(".mlp.", ".ffn.")
        return "transformer.blocks." + suffix

    if key.startswith("feature_flow_attn."):
        suffix = key[len("feature_flow_attn.") :]
        suffix = suffix.replace("q_proj.", "query.")
        suffix = suffix.replace("k_proj.", "key.")
        return "flow_propagation." + suffix

    return None


def _official_checkpoint_state_dict(
    path: str | Path,
) -> Mapping[str, torch.Tensor]:
    checkpoint_path = Path(path).expanduser()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            "Official GMFlow checkpoint not found: "
            f"{checkpoint_path}. Run tools/download_gmflow_pretrained.py "
            "or set hqs_core.gmflow_pretrained_checkpoint to the downloaded "
            "gmflow_things-e9887eda.pth file."
        )
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )
    if not isinstance(checkpoint, Mapping):
        raise TypeError(
            f"GMFlow checkpoint must contain a mapping, got {type(checkpoint)!r}"
        )
    state = checkpoint.get("model", checkpoint.get("state_dict", checkpoint))
    if not isinstance(state, Mapping):
        raise TypeError("GMFlow checkpoint 'model' entry is not a state dictionary")
    non_tensors = [
        key for key, value in state.items() if not torch.is_tensor(value)
    ]
    if non_tensors:
        preview = ", ".join(str(key) for key in non_tensors[:5])
        raise TypeError(
            "GMFlow model state contains non-tensor entries: " + preview
        )
    return state


class GMFlowMatchingFrontEnd(nn.Module):
    """Dedicated GMFlow backbone, Transformer and flow-propagation module."""

    def __init__(
        self,
        channels: int = 128,
        transformer_depth: int = 6,
        ffn_expansion: int = 4,
        attention_splits: int = 2,
        gradient_checkpointing: bool = True,
        propagation_query_chunk_size: int = 512,
        released_weights_compatible: bool = False,
    ) -> None:
        super().__init__()
        self.channels = int(channels)
        self.released_weights_compatible = bool(
            released_weights_compatible
        )
        self.backbone = GMFlowMatchingBackbone(
            self.channels,
            released_weights_compatible=self.released_weights_compatible,
        )
        self.transformer = GMFlowFeatureTransformer(
            channels=self.channels,
            depth=int(transformer_depth),
            ffn_expansion=int(ffn_expansion),
            num_splits=int(attention_splits),
            gradient_checkpointing=bool(gradient_checkpointing),
        )
        self.flow_propagation = GMFlowFeatureFlowAttention(
            channels=self.channels,
            query_chunk_size=int(propagation_query_chunk_size),
        )

    def load_official_checkpoint(
        self,
        path: str | Path,
        *,
        strict: bool = True,
    ) -> GMFlowCheckpointLoadReport:
        """Load the released GMFlow backbone/Transformer/propagation weights.

        The official checkpoint also contains a convex full-resolution
        upsampler.  HQSCore deliberately does not import it because its own
        source-conditioned upsampler remains part of the inverse solver.
        Everything in the correspondence front end is required in strict
        mode; silently partial loading is never permitted.
        """

        if not self.released_weights_compatible:
            raise RuntimeError(
                "Official GMFlow weights require "
                "released_weights_compatible=True so the backbone graph "
                "matches the released implementation."
            )
        if not strict:
            raise ValueError(
                "Non-strict official GMFlow imports are disabled. The "
                "correspondence front end must load completely."
            )

        official_state = _official_checkpoint_state_dict(path)
        expected_state = self.state_dict()
        converted: Dict[str, torch.Tensor] = {}
        ignored = []
        unsupported_checkpoint = []

        for original_key, value in official_state.items():
            stripped = _strip_data_parallel_prefix(str(original_key))
            local_key = _official_gmflow_key_to_local(stripped)
            if local_key is None:
                if stripped.startswith("upsampler."):
                    ignored.append(str(original_key))
                else:
                    unsupported_checkpoint.append(str(original_key))
                continue
            if local_key in converted:
                raise RuntimeError(
                    f"Multiple official GMFlow keys map to {local_key!r}"
                )
            converted[local_key] = value

        missing = sorted(set(expected_state) - set(converted))
        unexpected = sorted(set(converted) - set(expected_state))
        shape_mismatches = sorted(
            key
            for key in set(expected_state).intersection(converted)
            if tuple(expected_state[key].shape) != tuple(converted[key].shape)
        )
        if missing or unexpected or shape_mismatches or unsupported_checkpoint:
            details = []
            if missing:
                details.append("missing=" + ", ".join(missing[:8]))
            if unexpected:
                details.append("unexpected=" + ", ".join(unexpected[:8]))
            if shape_mismatches:
                shape_details = [
                    f"{key}: expected {tuple(expected_state[key].shape)}, "
                    f"got {tuple(converted[key].shape)}"
                    for key in shape_mismatches[:8]
                ]
                details.append("shape_mismatch=" + "; ".join(shape_details))
            if unsupported_checkpoint:
                details.append(
                    "unsupported_official_keys="
                    + ", ".join(unsupported_checkpoint[:8])
                )
            raise RuntimeError(
                "Official GMFlow checkpoint is not a strict single-scale "
                "front-end match: " + " | ".join(details)
            )

        loadable = {
            key: value
            for key, value in converted.items()
            if key in expected_state
            and tuple(value.shape) == tuple(expected_state[key].shape)
        }
        self.load_state_dict(loadable, strict=True)

        loaded_parameter_count = sum(value.numel() for value in loadable.values())
        return GMFlowCheckpointLoadReport(
            path=str(Path(path).expanduser()),
            loaded_tensor_count=len(loadable),
            loaded_parameter_count=loaded_parameter_count,
            ignored_checkpoint_keys=tuple(sorted(ignored)),
        )

    def set_trainable(self, trainable: bool) -> None:
        for parameter in self.parameters():
            parameter.requires_grad_(bool(trainable))

    def forward(
        self,
        source_image: torch.Tensor,
        target_image: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if source_image.shape != target_image.shape:
            raise ValueError("GMFlow matching images must have identical shapes")
        joined = torch.cat((source_image, target_image), dim=0)
        features = self.backbone(joined)
        source, target = features.chunk(2, dim=0)
        return self.transformer(source, target)

    def propagate(
        self,
        feature: torch.Tensor,
        flow_xy: torch.Tensor,
    ) -> torch.Tensor:
        return self.flow_propagation(feature, flow_xy)


def gmflow_forward_backward_consistency(
    forward_xy: torch.Tensor,
    backward_xy: torch.Tensor,
    *,
    feature_scale: int,
    alpha: float = 0.01,
    beta_full_resolution: float = 0.5,
    softness_full_resolution: float = 0.1,
    threshold_mode: str = "gmflow",
) -> Dict[str, torch.Tensor]:
    """Compute GMFlow/UnFlow-style consistency and a differentiable gate.

    The published ``beta`` threshold is specified in full-resolution pixels.
    These flows live on the feature grid, so the additive threshold and soft
    transition are divided by ``feature_scale`` before comparison.
    """
    if forward_xy.shape != backward_xy.shape:
        raise ValueError("Forward and backward flow must have identical shapes")
    warped_backward = backward_warp(
        backward_xy,
        forward_xy,
        padding_mode="zeros",
    )
    residual = forward_xy + warped_backward
    error = residual.square().sum(dim=1, keepdim=True).add(1e-12).sqrt()
    mode = str(threshold_mode).lower()
    if mode == "gmflow":
        magnitude = (
            forward_xy.square().sum(dim=1, keepdim=True).add(1e-12).sqrt()
            + backward_xy.square().sum(dim=1, keepdim=True).add(1e-12).sqrt()
        )
    elif mode == "aligned":
        magnitude = (
            forward_xy.square().sum(dim=1, keepdim=True).add(1e-12).sqrt()
            + warped_backward.square().sum(dim=1, keepdim=True).add(1e-12).sqrt()
        )
    else:
        raise ValueError("threshold_mode must be 'gmflow' or 'aligned'")
    scale = max(int(feature_scale), 1)
    threshold = float(alpha) * magnitude + float(beta_full_resolution) / scale
    softness = max(float(softness_full_resolution) / scale, 1e-4)
    in_bounds = flow_in_bounds_mask(forward_xy)
    reliability = torch.sigmoid((threshold - error) / softness) * in_bounds
    occlusion = ((error > threshold) | (in_bounds < 0.5)).to(forward_xy.dtype)
    return {
        "warped_backward_xy": warped_backward,
        "residual_xy": residual,
        "error": error,
        "threshold": threshold,
        "in_bounds": in_bounds,
        "reliability": reliability.clamp(0.0, 1.0),
        "occlusion": occlusion,
    }


__all__ = [
    "GMFlowCheckpointLoadReport",
    "GMFlowMatchingBackbone",
    "GMFlowFeatureTransformer",
    "GMFlowFeatureFlowAttention",
    "GMFlowMatchingFrontEnd",
    "gmflow_forward_backward_consistency",
]
