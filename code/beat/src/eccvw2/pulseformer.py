"""BEAT adapters for a separately obtained, pinned egoPPG PulseFormer.

No upstream architecture implementation is redistributed here. The original
constructor and forward method are loaded from the user's egoPPG checkout;
BEAT registers its own modules at the original bottleneck or pooled output.
"""

from __future__ import annotations

import hashlib
import importlib.util
import os
from pathlib import Path
from types import ModuleType

import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet18_Weights


UPSTREAM_REPOSITORY = "https://github.com/eth-siplab/egoPPG"
UPSTREAM_REVISION = "5bb8437a13bfa4ee8ced89126dec66db4c40f4c0"
UPSTREAM_MODEL_SHA256 = "c0e1913cfa98ad94acb1c1be87ef50d3499487f31a53b74f19218314e1a2ddd1"
_UPSTREAM_MODULES: dict[Path, ModuleType] = {}


def upstream_root() -> Path:
    """Resolve the user-owned checkout, without downloading source or weights."""
    default = Path(__file__).resolve().parents[2] / "third_party" / "egoPPG"
    return Path(os.environ.get("BEAT_EGOPPG_ROOT", str(default))).expanduser().resolve()


def _load_upstream() -> ModuleType:
    path = upstream_root() / "ml" / "models" / "PulseFormer.py"
    if not path.is_file():
        raise FileNotFoundError(
            "Obtain the official egoPPG checkout at revision "
            f"{UPSTREAM_REVISION} and set BEAT_EGOPPG_ROOT to its root. "
            f"Missing model source: {path}"
        )
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != UPSTREAM_MODEL_SHA256:
        raise RuntimeError(
            "The egoPPG model source differs from the pinned paper dependency. "
            "Use a clean checkout of revision "
            f"{UPSTREAM_REVISION}; expected SHA256={UPSTREAM_MODEL_SHA256}, "
            f"received={digest}."
        )
    if path not in _UPSTREAM_MODULES:
        spec = importlib.util.spec_from_file_location("_beat_external_pulseformer", path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot import the external PulseFormer from {path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _UPSTREAM_MODULES[path] = module
    return _UPSTREAM_MODULES[path]


class PulseFormer(nn.Module):
    """Checkpoint-key-compatible adapter around the original external model."""

    temporal_insertion_point = "original"

    def __init__(self, frames=128, h=48, w=128, num_embeddings=141, temporal_mixer=None):
        super().__init__()
        external_class = _load_upstream().PulseFormer
        external = external_class(frames=frames, h=h, w=w, num_embeddings=num_embeddings)
        # Keep the original registration order and names. This consumes exactly
        # the same random draws as the frozen experiment's backbone constructor.
        for name, module in external._modules.items():
            self.add_module(name, module)
        for name, parameter in external._parameters.items():
            self.register_parameter(name, parameter)
        for name, buffer in external._buffers.items():
            self.register_buffer(name, buffer, name not in external._non_persistent_buffers_set)
        self.frames = int(frames)
        self.temporal_mixer = temporal_mixer if temporal_mixer is not None else nn.Identity()
        self._external_forward = external_class.forward

    def forward(self, x, imu):
        if x.ndim != 5:
            raise ValueError(f"Expected video [B,C,T,H,W], got {tuple(x.shape)}")
        output = self._external_forward(self, x, imu)
        if output.shape != (x.shape[0], x.shape[2]):
            raise RuntimeError(f"Unexpected waveform output shape: {tuple(output.shape)}")
        return output


class PulseFormerBottleneckDWTCN(PulseFormer):
    """Apply the shared BEAT temporal adapter to each bottleneck location."""

    temporal_insertion_point = "encoder_bottleneck"

    def __init__(self, *args, temporal_mixer, **kwargs):
        super().__init__(*args, temporal_mixer=temporal_mixer, **kwargs)
        if isinstance(self.temporal_mixer, nn.Identity):
            raise ValueError("The bottleneck variant requires a temporal adapter")
        channels = getattr(self.temporal_mixer, "channels", None)
        if channels != self.ConvBlock9[0].out_channels or channels != self.upsample[0].in_channels:
            raise ValueError("Adapter channels must match the encoder/decoder bottleneck")
        self.ConvBlock9.register_forward_hook(self._adapt_bottleneck)

    @staticmethod
    def reshape_bottleneck_tokens(features):
        if features.ndim != 5:
            raise RuntimeError(f"Expected [B,C,T,H,W], got {tuple(features.shape)}")
        batch, channels, length, height, width = features.shape
        return features.permute(0, 3, 4, 1, 2).contiguous().view(
            batch * height * width, channels, length
        )

    @staticmethod
    def restore_bottleneck_features(tokens, feature_shape):
        batch, channels, length, height, width = feature_shape
        if tuple(tokens.shape) != (batch * height * width, channels, length):
            raise RuntimeError("The adapter must preserve [B*H*W,C,T]")
        return tokens.reshape(batch, height, width, channels, length).permute(
            0, 3, 4, 1, 2
        ).contiguous()

    def forward_bottleneck_mixer(self, features):
        tokens = self.reshape_bottleneck_tokens(features)
        return self.restore_bottleneck_features(self.temporal_mixer(tokens), features.shape)

    def _adapt_bottleneck(self, module, inputs, output):
        return self.forward_bottleneck_mixer(output)


class PulseFormerPostPoolDWTCN(PulseFormer):
    """Paper placement control: adapter after decoder and spatial pooling."""

    temporal_insertion_point = "post_pool"

    def __init__(self, *args, temporal_mixer=None, **kwargs):
        super().__init__(*args, temporal_mixer=temporal_mixer, **kwargs)
        if isinstance(self.temporal_mixer, nn.Identity):
            raise ValueError("The post-pooling variant requires a temporal adapter")
        self.poolspa.register_forward_hook(self._adapt_pooled)

    def _adapt_pooled(self, module, inputs, output):
        if output.ndim != 5 or output.shape[-2:] != (1, 1):
            raise RuntimeError("Expected pooled features [B,C,T,1,1]")
        tokens = output.mean(dim=(-1, -2))
        mixed = self.temporal_mixer(tokens)
        if mixed.shape != tokens.shape:
            raise RuntimeError("The adapter must preserve [B,C,T]")
        return mixed.unsqueeze(-1).unsqueeze(-1)


class PulseFormerCoarseToFineDWTCN(PulseFormer):
    """Import-compatible name for an exploratory variant not in this release."""

    def __init__(self, *args, **kwargs):
        raise ValueError("Coarse-to-fine is not part of the BEAT paper release")


PulseFormerOriginal = PulseFormer
