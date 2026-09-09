"""Dilated depthwise temporal mixing for PulseFormer feature tokens."""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


def _validate_temporal_shape(x: torch.Tensor, channels: int) -> None:
    if x.ndim != 3:
        raise ValueError(f"Expected [B, C, T], got {tuple(x.shape)}")
    if x.shape[1] != channels:
        raise ValueError(
            f"Expected {channels} temporal channels, got {x.shape[1]}"
        )


def _apply_group_norm(
    norm: nn.GroupNorm,
    x: torch.Tensor,
    causal: bool,
) -> torch.Tensor:
    """Apply GroupNorm without future leakage in causal mode.

    Standard GroupNorm on ``[B, C, T]`` includes the complete temporal axis in
    its statistics. Causal mode instead normalizes each token independently by
    reshaping it to ``[B*T, C, 1]`` while retaining the requested GroupNorm(1,C).
    """

    if not causal:
        return norm(x)
    batch, channels, tokens = x.shape
    tokenwise = x.transpose(1, 2).reshape(batch * tokens, channels, 1)
    tokenwise = norm(tokenwise)
    return tokenwise.reshape(batch, tokens, channels).transpose(1, 2)


class DilatedDepthwiseTemporalBlock(nn.Module):
    """One shape-preserving residual depthwise temporal convolution."""

    def __init__(
        self,
        channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float,
        causal: bool,
        layer_scale_init: float,
    ) -> None:
        super().__init__()
        if channels <= 0:
            raise ValueError(f"channels must be positive, got {channels}")
        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError(
                f"kernel_size must be a positive odd integer, got {kernel_size}"
            )
        if dilation <= 0:
            raise ValueError(f"dilation must be positive, got {dilation}")
        if not 0.0 <= dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {dropout}")

        self.channels = int(channels)
        self.kernel_size = int(kernel_size)
        self.dilation = int(dilation)
        self.causal = bool(causal)
        self.left_padding = self.dilation * (self.kernel_size - 1)
        symmetric_padding = self.left_padding // 2

        self.norm = nn.GroupNorm(num_groups=1, num_channels=self.channels)
        self.depthwise_conv = nn.Conv1d(
            in_channels=self.channels,
            out_channels=self.channels,
            kernel_size=self.kernel_size,
            dilation=self.dilation,
            groups=self.channels,
            padding=0 if self.causal else symmetric_padding,
            bias=True,
        )
        self.activation = nn.SiLU()
        self.dropout = nn.Dropout(dropout)
        self.layer_scale = nn.Parameter(
            torch.full((1, self.channels, 1), float(layer_scale_init))
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _validate_temporal_shape(x, self.channels)
        residual = x
        update = _apply_group_norm(self.norm, x, causal=self.causal)
        if self.causal:
            update = F.pad(update, (self.left_padding, 0))
        update = self.depthwise_conv(update)
        update = self.activation(update)
        update = self.dropout(update)
        if update.shape != residual.shape:
            raise RuntimeError(
                "Depthwise block changed the tensor shape: "
                f"input={tuple(residual.shape)}, output={tuple(update.shape)}"
            )
        return residual + self.layer_scale * update


class PointwiseGLU(nn.Module):
    """Token-preserving channel mixing with a pointwise gated linear unit."""

    def __init__(self, channels: int, causal: bool = False) -> None:
        super().__init__()
        self.channels = int(channels)
        self.causal = bool(causal)
        self.norm = nn.GroupNorm(num_groups=1, num_channels=self.channels)
        self.glu_in = nn.Conv1d(self.channels, 2 * self.channels, kernel_size=1)
        self.glu_out = nn.Conv1d(self.channels, self.channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _validate_temporal_shape(x, self.channels)
        x = _apply_group_norm(self.norm, x, causal=self.causal)
        value, gate = self.glu_in(x).chunk(2, dim=1)
        return self.glu_out(value * torch.sigmoid(gate))


class DilatedDepthwiseTCN(nn.Module):
    """Residual DW-TCN adapter that preserves ``[B, C, T]`` exactly."""

    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        dilations: tuple[int, ...] = (1, 2, 4, 8),
        dropout: float = 0.1,
        causal: bool = False,
        layer_scale_init: float = 1.0e-3,
    ) -> None:
        super().__init__()
        if not dilations:
            raise ValueError("dilations must contain at least one value")
        self.channels = int(channels)
        self.kernel_size = int(kernel_size)
        self.dilations = tuple(int(dilation) for dilation in dilations)
        self.causal = bool(causal)

        self.blocks = nn.ModuleList(
            [
                DilatedDepthwiseTemporalBlock(
                    channels=self.channels,
                    kernel_size=self.kernel_size,
                    dilation=dilation,
                    dropout=dropout,
                    causal=self.causal,
                    layer_scale_init=layer_scale_init,
                )
                for dilation in self.dilations
            ]
        )
        self.pointwise_glu = PointwiseGLU(
            channels=self.channels,
            causal=self.causal,
        )
        self.output_projection = nn.Conv1d(
            self.channels,
            self.channels,
            kernel_size=1,
        )
        nn.init.zeros_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)

    @property
    def receptive_field(self) -> int:
        return 1 + (self.kernel_size - 1) * sum(self.dilations)

    def coverage_ratio(self, temporal_tokens: int) -> float:
        if temporal_tokens <= 0:
            raise ValueError(
                f"temporal_tokens must be positive, got {temporal_tokens}"
            )
        return self.receptive_field / temporal_tokens

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _validate_temporal_shape(x, self.channels)
        residual = x
        for block in self.blocks:
            x = block(x)
        x = self.pointwise_glu(x)
        delta = self.output_projection(x)
        if delta.shape != residual.shape:
            raise RuntimeError(
                "DW-TCN changed the tensor shape: "
                f"input={tuple(residual.shape)}, output={tuple(delta.shape)}"
            )
        return residual + delta


def temporal_receptive_field(
    kernel_size: int,
    dilations: Sequence[int],
) -> int:
    """Return the receptive field for one convolution at every dilation."""

    if kernel_size <= 0:
        raise ValueError(f"kernel_size must be positive, got {kernel_size}")
    dilation_values = tuple(int(value) for value in dilations)
    if not dilation_values or any(value <= 0 for value in dilation_values):
        raise ValueError(f"dilations must be positive, got {dilation_values}")
    return 1 + (kernel_size - 1) * sum(dilation_values)
