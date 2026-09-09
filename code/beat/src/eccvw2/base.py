"""Reproducible BEAT construction and audits for user-supplied local weights.

The backbone is loaded from the separately obtained, pinned egoPPG dependency.
This release includes no learned parameter files.
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
from unittest.mock import patch

import torch
import torch.nn as nn

from . import pulseformer as pulseformer_module
from .pulseformer import (
    PulseFormerBottleneckDWTCN,
    PulseFormerCoarseToFineDWTCN,
    PulseFormerOriginal,
    PulseFormerPostPoolDWTCN,
)
from .temporal import DilatedDepthwiseTCN


SPATIAL_PREFIXES = tuple(
    [f"ConvBlock{index}." for index in range(1, 10)]
    + [f"SpatialAttentionGate{index}." for index in range(1, 5)]
)
MITA_PREFIXES = ("cross_attention.",)
DECODER_PREFIXES = ("upsample.", "upsample2.")
HEAD_PREFIXES = ("ConvBlock10.",)
NEW_TCN_PREFIXES = ("temporal_mixer.", "fine_temporal_mixer.")

# The audited source model has no replaceable temporal module. This allowlist
# is intentionally explicit: a broad "attention" rule would incorrectly drop
# the retained MITA cross_attention weights.
REMOVED_TEMPORAL_PREFIXES: tuple[str, ...] = ()


@dataclass(frozen=True)
class CheckpointAudit:
    checkpoint_path: str
    checkpoint_sha256: str
    source_key_count: int
    loaded_spatial_keys: tuple[str, ...]
    loaded_mita_keys: tuple[str, ...]
    loaded_decoder_keys: tuple[str, ...]
    loaded_head_keys: tuple[str, ...]
    loaded_other_keys: tuple[str, ...]
    removed_temporal_keys: tuple[str, ...]
    new_tcn_keys: tuple[str, ...]
    missing_keys: tuple[str, ...]
    unexpected_keys: tuple[str, ...]
    shape_mismatch_keys: tuple[str, ...]
    shape_mismatch_details: tuple[str, ...]

    @property
    def loaded_key_count(self) -> int:
        return sum(
            len(keys)
            for keys in (
                self.loaded_spatial_keys,
                self.loaded_mita_keys,
                self.loaded_decoder_keys,
                self.loaded_head_keys,
                self.loaded_other_keys,
            )
        )

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["loaded_key_count"] = self.loaded_key_count
        payload["loaded_base_keys"] = self.loaded_key_count
        payload["missing_temporal_mixer_keys"] = list(self.new_tcn_keys)
        payload["base_checkpoint_strict_status"] = "pass"
        return payload


@dataclass(frozen=True)
class FreshInitializationAudit:
    architecture: str
    initialization_seed: int
    imagenet_weights: str
    imagenet_cache_path: str
    imagenet_cache_sha256: str
    canonical_key_count: int
    target_key_count: int
    common_key_count: int
    new_temporal_mixer_keys: tuple[str, ...]
    common_state_sha256: str
    exact_copy_status: bool
    full_model_checkpoint_loaded: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def sha256_file(path: str | Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_state(state: dict[str, torch.Tensor], keys: tuple[str, ...]) -> str:
    digest = hashlib.sha256()
    for key in keys:
        tensor = state[key].detach().cpu().contiguous()
        metadata = f"{key}\0{tensor.dtype}\0{tuple(tensor.shape)}\0".encode("utf-8")
        raw = tensor.numpy().tobytes(order="C")
        digest.update(len(metadata).to_bytes(8, "big"))
        digest.update(metadata)
        digest.update(len(raw).to_bytes(8, "big"))
        digest.update(raw)
    return digest.hexdigest()


def _resolve_resnet18_imagenet1k_v1_cache(
    explicit_path: str | Path | None,
) -> Path:
    weights = pulseformer_module.ResNet18_Weights.IMAGENET1K_V1
    if explicit_path is None:
        filename = Path(urlparse(weights.url).path).name
        path = Path(torch.hub.get_dir()) / "checkpoints" / filename
    else:
        path = Path(explicit_path)
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(
            "ResNet18 IMAGENET1K_V1 is not present in the local torch cache; "
            f"refusing a network download: {path}"
        )
    return path


def _load_local_resnet18_imagenet1k_v1(
    constructor: Any,
    cache_path: Path,
) -> nn.Module:
    model = constructor(weights=None)
    state = torch.load(cache_path, map_location="cpu", weights_only=True)
    model.load_state_dict(_extract_state_dict(state), strict=True)
    return model


def _extract_state_dict(checkpoint: Any) -> dict[str, torch.Tensor]:
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        checkpoint = checkpoint["model_state_dict"]
    if not isinstance(checkpoint, dict):
        raise TypeError("Checkpoint must be a state_dict or contain model_state_dict")
    if not all(isinstance(key, str) for key in checkpoint):
        raise TypeError("Checkpoint state_dict keys must be strings")
    if not all(torch.is_tensor(value) for value in checkpoint.values()):
        raise TypeError("Checkpoint state_dict values must be tensors")
    return checkpoint


def _matches_prefix(key: str, prefixes: tuple[str, ...]) -> bool:
    return key.startswith(prefixes) if prefixes else False


def _classify_loaded_keys(
    keys: set[str],
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    spatial = tuple(sorted(key for key in keys if _matches_prefix(key, SPATIAL_PREFIXES)))
    mita = tuple(sorted(key for key in keys if _matches_prefix(key, MITA_PREFIXES)))
    decoder = tuple(sorted(key for key in keys if _matches_prefix(key, DECODER_PREFIXES)))
    head = tuple(sorted(key for key in keys if _matches_prefix(key, HEAD_PREFIXES)))
    classified = set(spatial) | set(mita) | set(decoder) | set(head)
    other = tuple(sorted(keys - classified))
    return spatial, mita, decoder, head, other


def validate_temporal_mixer_config(config: dict[str, Any]) -> None:
    mixer_type = str(config.get("type", "")).lower()
    if mixer_type not in {"original", "dilated_depthwise_tcn"}:
        raise ValueError(f"Unsupported temporal mixer type: {mixer_type!r}")
    insertion_point = str(
        config.get(
            "insertion_point",
            "original" if mixer_type == "original" else "post_pool",
        )
    ).lower()
    apply_per_spatial_location = bool(
        config.get("apply_per_spatial_location", False)
    )
    if mixer_type == "original":
        if insertion_point not in {"none", "original"}:
            raise ValueError(
                "Original temporal path requires insertion_point=original or none"
            )
        if apply_per_spatial_location:
            raise ValueError(
                "Original temporal path cannot apply a per-spatial-location mixer"
            )
        return

    if "insertion_point" not in config:
        raise ValueError("DW-TCN config requires insertion_point")
    if "apply_per_spatial_location" not in config:
        raise ValueError("DW-TCN config requires apply_per_spatial_location")
    if insertion_point not in {"encoder_bottleneck", "post_pool"}:
        raise ValueError(
            f"Unsupported DW-TCN insertion point: {insertion_point!r}"
        )
    expected_spatial_application = insertion_point == "encoder_bottleneck"
    if apply_per_spatial_location != expected_spatial_application:
        raise ValueError(
            "apply_per_spatial_location must be true only for "
            f"encoder_bottleneck, got insertion_point={insertion_point!r}, "
            f"apply_per_spatial_location={apply_per_spatial_location}"
        )

    required_values = {
        "normalization": "group_norm",
        "activation": "silu",
    }
    for key, expected in required_values.items():
        if config.get(key) != expected:
            raise ValueError(f"{key} must be {expected!r}, got {config.get(key)!r}")
    if config.get("layer_scale", {}).get("enabled") is not True:
        raise ValueError("layer_scale.enabled must be true")
    pointwise = config.get("pointwise_glu", {})
    if pointwise.get("enabled") is not True or pointwise.get("expansion") != 2:
        raise ValueError("pointwise_glu must be enabled with expansion=2")
    residual = config.get("residual_output", {})
    if residual.get("enabled") is not True or residual.get("zero_init") is not True:
        raise ValueError("residual_output must be enabled with zero_init=true")


def build_temporal_mixer(
    config: dict[str, Any],
    inferred_channels: int,
) -> nn.Module:
    validate_temporal_mixer_config(config)
    mixer_type = str(config["type"]).lower()
    configured_channels = config.get("channels")
    if configured_channels is not None and int(configured_channels) != inferred_channels:
        raise ValueError(
            "Configured temporal channels do not match the spatial backbone: "
            f"configured={configured_channels}, inferred={inferred_channels}"
        )
    if mixer_type == "original":
        return nn.Identity()
    return DilatedDepthwiseTCN(
        channels=inferred_channels,
        kernel_size=int(config.get("kernel_size", 3)),
        dilations=tuple(int(value) for value in config.get("dilations", (1, 2, 4, 8))),
        dropout=float(config.get("dropout", 0.1)),
        causal=bool(config.get("causal", False)),
        layer_scale_init=float(config["layer_scale"].get("init", 1.0e-3)),
    )


def _dual_resolution_fine_config(
    model_config: dict[str, Any],
    coarse_config: dict[str, Any],
) -> dict[str, Any] | None:
    """Validate and return the optional full-resolution DW-TCN config."""

    fine_config = model_config.get("fine_temporal_mixer")
    if fine_config is None:
        return None
    if not isinstance(fine_config, dict):
        raise TypeError("model.fine_temporal_mixer must be a mapping")
    validate_temporal_mixer_config(fine_config)
    coarse_type = str(coarse_config.get("type", "")).lower()
    coarse_insertion = str(coarse_config.get("insertion_point", "")).lower()
    if (
        coarse_type != "dilated_depthwise_tcn"
        or coarse_insertion != "encoder_bottleneck"
    ):
        raise ValueError(
            "Dual-resolution architecture requires the primary temporal_mixer "
            "at encoder_bottleneck"
        )
    fine_type = str(fine_config.get("type", "")).lower()
    fine_insertion = str(fine_config.get("insertion_point", "")).lower()
    if fine_type != "dilated_depthwise_tcn" or fine_insertion != "post_pool":
        raise ValueError(
            "fine_temporal_mixer must be a dilated_depthwise_tcn at post_pool"
        )
    return fine_config


def build_pulseformer(
    model_config: dict[str, Any],
    frames: int = 128,
) -> nn.Module:
    """Build the local model without a constructor-time ImageNet download."""

    original_resnet18 = pulseformer_module.models.resnet18

    def build_uninitialized_resnet18(*args: Any, **kwargs: Any) -> nn.Module:
        del args, kwargs
        return original_resnet18(weights=None)

    mixer_config = model_config["temporal_mixer"]
    validate_temporal_mixer_config(mixer_config)
    fine_mixer_config = _dual_resolution_fine_config(model_config, mixer_config)
    model_type = str(mixer_config["type"]).lower()
    insertion_point = str(
        mixer_config.get(
            "insertion_point",
            "original" if model_type == "original" else "post_pool",
        )
    ).lower()
    if fine_mixer_config is not None:
        model_class = PulseFormerCoarseToFineDWTCN
        expected_model_name = "pulseformer_coarse_to_fine_dwtcn"
    elif model_type == "original":
        model_class = PulseFormerOriginal
        expected_model_name = "pulseformer_original"
    elif insertion_point == "encoder_bottleneck":
        model_class = PulseFormerBottleneckDWTCN
        expected_model_name = "pulseformer_bottleneck_dwtcn"
    else:
        model_class = PulseFormerPostPoolDWTCN
        expected_model_name = "pulseformer_post_pool_dwtcn"
    configured_model_name = model_config.get("name")
    if configured_model_name is None:
        raise ValueError("model.name is required for unambiguous architecture selection")
    if configured_model_name != expected_model_name:
        raise ValueError(
            "model.name does not match temporal mixer selection: "
            f"configured={configured_model_name!r}, expected={expected_model_name!r}"
        )

    with patch.object(
        pulseformer_module.models,
        "resnet18",
        side_effect=build_uninitialized_resnet18,
    ):
        channel_probe = PulseFormerOriginal(
            frames=frames,
            temporal_mixer=nn.Identity(),
        )
        if insertion_point == "encoder_bottleneck":
            channel_projection = channel_probe.ConvBlock9[0]
            expected_projection_type = nn.Conv3d
        else:
            channel_projection = channel_probe.upsample2[0]
            expected_projection_type = nn.ConvTranspose3d
        if not isinstance(channel_projection, expected_projection_type):
            raise TypeError(
                "Could not infer temporal mixer channels from the selected "
                f"insertion point: got {type(channel_projection).__name__}"
            )
        inferred_channels = int(channel_projection.out_channels)
        mixer = build_temporal_mixer(
            mixer_config,
            inferred_channels=inferred_channels,
        )
        if fine_mixer_config is not None:
            fine_projection = channel_probe.upsample2[0]
            if not isinstance(fine_projection, nn.ConvTranspose3d):
                raise TypeError(
                    "Could not infer fine temporal channels from upsample2[0]: "
                    f"got {type(fine_projection).__name__}"
                )
            fine_mixer = build_temporal_mixer(
                fine_mixer_config,
                inferred_channels=int(fine_projection.out_channels),
            )
            del channel_probe
            model = model_class(
                frames=frames,
                temporal_mixer=mixer,
                fine_temporal_mixer=fine_mixer,
            )
        elif model_type == "original":
            model = channel_probe
            model.temporal_mixer = mixer
        else:
            del channel_probe
            model = model_class(frames=frames, temporal_mixer=mixer)
    return model


def build_fresh_paper_model(
    model_config: dict[str, Any],
    frames: int = 128,
    initialization_seed: int = 0,
    imagenet_cache_path: str | Path | None = None,
) -> tuple[nn.Module, FreshInitializationAudit]:
    """Build fresh Original/Bottleneck with one bit-identical shared base."""
    mixer_config = model_config["temporal_mixer"]
    validate_temporal_mixer_config(mixer_config)
    fine_mixer_config = _dual_resolution_fine_config(model_config, mixer_config)
    model_type = str(mixer_config["type"]).lower()
    insertion_point = str(mixer_config.get("insertion_point", "original")).lower()
    if fine_mixer_config is not None:
        architecture = "pulseformer_coarse_to_fine_dwtcn"
    elif model_type == "original":
        architecture = "pulseformer_original"
    elif (
        model_type == "dilated_depthwise_tcn"
        and insertion_point == "encoder_bottleneck"
    ):
        architecture = "pulseformer_bottleneck_dwtcn"
    else:
        raise ValueError(
            "Fresh paper builder supports only Original or Bottleneck DW-TCN"
        )
    if model_config.get("name") != architecture:
        raise ValueError(
            f"model.name mismatch: configured={model_config.get('name')!r}, "
            f"expected={architecture!r}"
        )

    cache_path = _resolve_resnet18_imagenet1k_v1_cache(imagenet_cache_path)
    weights = pulseformer_module.ResNet18_Weights.IMAGENET1K_V1
    original_resnet18 = pulseformer_module.models.resnet18

    def cached_resnet18(*args: Any, **kwargs: Any) -> nn.Module:
        requested = kwargs.get("weights", args[0] if args else None)
        if requested != weights:
            raise ValueError(
                f"Fresh paper initialization requires {weights}, got {requested}"
            )
        return _load_local_resnet18_imagenet1k_v1(
            original_resnet18,
            cache_path,
        )

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(initialization_seed))
        with patch.object(
            pulseformer_module.models,
            "resnet18",
            side_effect=cached_resnet18,
        ):
            canonical = PulseFormerOriginal(
                frames=frames,
                temporal_mixer=nn.Identity(),
            )
            if architecture == "pulseformer_original":
                target = canonical
            else:
                projection = canonical.ConvBlock9[0]
                if not isinstance(projection, nn.Conv3d):
                    raise TypeError("ConvBlock9[0] must be Conv3d")
                mixer = build_temporal_mixer(
                    mixer_config,
                    int(projection.out_channels),
                )
                if architecture == "pulseformer_coarse_to_fine_dwtcn":
                    fine_projection = canonical.upsample2[0]
                    if not isinstance(fine_projection, nn.ConvTranspose3d):
                        raise TypeError("upsample2[0] must be ConvTranspose3d")
                    fine_mixer = build_temporal_mixer(
                        fine_mixer_config,
                        int(fine_projection.out_channels),
                    )
                    target = PulseFormerCoarseToFineDWTCN(
                        frames=frames,
                        temporal_mixer=mixer,
                        fine_temporal_mixer=fine_mixer,
                    )
                else:
                    target = PulseFormerBottleneckDWTCN(
                        frames=frames,
                        temporal_mixer=mixer,
                    )

    source_state = canonical.state_dict()
    target_state = target.state_dict()
    source_keys, target_keys = set(source_state), set(target_state)
    source_only = source_keys - target_keys
    target_only = target_keys - source_keys
    bad_new = {
        key for key in target_only if not _matches_prefix(key, NEW_TCN_PREFIXES)
    }
    common = tuple(sorted(source_keys & target_keys))
    incompatible = tuple(
        key
        for key in common
        if source_state[key].shape != target_state[key].shape
        or source_state[key].dtype != target_state[key].dtype
    )
    if source_only or bad_new or incompatible:
        raise RuntimeError(
            "Fresh initialization architecture audit failed: "
            f"source_only={tuple(sorted(source_only))}, "
            f"bad_target_only={tuple(sorted(bad_new))}, "
            f"incompatible={incompatible}"
        )
    if target is not canonical:
        result = target.load_state_dict(
            {key: source_state[key] for key in common},
            strict=False,
        )
        if set(result.missing_keys) != target_only or result.unexpected_keys:
            raise RuntimeError(
                "Fresh common-state copy diverged: "
                f"missing={result.missing_keys}, unexpected={result.unexpected_keys}"
            )
    copied_state = target.state_dict()
    exact = all(
        torch.equal(source_state[key], copied_state[key]) for key in common
    )
    source_hash = _sha256_state(source_state, common)
    target_hash = _sha256_state(copied_state, common)
    if not exact or source_hash != target_hash:
        raise RuntimeError("Fresh common-state bit-exact copy verification failed")
    audit = FreshInitializationAudit(
        architecture=architecture,
        initialization_seed=int(initialization_seed),
        imagenet_weights=str(weights),
        imagenet_cache_path=str(cache_path),
        imagenet_cache_sha256=sha256_file(cache_path),
        canonical_key_count=len(source_state),
        target_key_count=len(copied_state),
        common_key_count=len(common),
        new_temporal_mixer_keys=tuple(sorted(target_only)),
        common_state_sha256=source_hash,
        exact_copy_status=True,
    )
    return target, audit


def audited_load_checkpoint(
    model: nn.Module,
    checkpoint_path: str | Path,
) -> CheckpointAudit:
    """Load every compatible preserved key and reject all silent drift."""

    checkpoint_path = Path(checkpoint_path).resolve()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    source_state = _extract_state_dict(checkpoint)
    target_state = model.state_dict()

    compatible: dict[str, torch.Tensor] = {}
    removed_temporal: list[str] = []
    unexpected: list[str] = []
    shape_mismatch_names: set[str] = set()
    shape_mismatch_details: list[str] = []
    for key, value in source_state.items():
        if key not in target_state:
            if _matches_prefix(key, REMOVED_TEMPORAL_PREFIXES):
                removed_temporal.append(key)
            else:
                unexpected.append(key)
            continue
        if target_state[key].shape != value.shape:
            shape_mismatch_names.add(key)
            shape_mismatch_details.append(
                f"{key}: source={tuple(value.shape)}, target={tuple(target_state[key].shape)}"
            )
            continue
        compatible[key] = value

    target_missing = set(target_state) - set(compatible)
    expected_new_tcn = {
        key for key in target_state if _matches_prefix(key, NEW_TCN_PREFIXES)
    }
    source_tcn_keys = {
        key for key in source_state if _matches_prefix(key, NEW_TCN_PREFIXES)
    }
    if source_tcn_keys:
        raise RuntimeError(
            "Audited base checkpoint loading rejects temporal_mixer keys; "
            "use an architecture-tagged strict reload for a trained model"
        )
    allowed_new_tcn = target_missing & expected_new_tcn
    new_tcn = tuple(sorted(allowed_new_tcn))
    missing = tuple(
        sorted(target_missing - allowed_new_tcn - shape_mismatch_names)
    )
    unexpected_tuple = tuple(sorted(unexpected))
    shape_mismatch_tuple = tuple(sorted(shape_mismatch_names))
    shape_mismatch_details_tuple = tuple(sorted(shape_mismatch_details))

    if missing or unexpected_tuple or shape_mismatch_tuple:
        raise RuntimeError(
            "Checkpoint compatibility audit failed: "
            f"missing={missing}, unexpected={unexpected_tuple}, "
            f"shape_mismatch={shape_mismatch_details_tuple}"
        )

    incompatible = model.load_state_dict(compatible, strict=False)
    if set(incompatible.missing_keys) != set(new_tcn) or incompatible.unexpected_keys:
        raise RuntimeError(
            "Checkpoint load result diverged from the pre-load audit: "
            f"missing={incompatible.missing_keys}, unexpected={incompatible.unexpected_keys}"
        )

    spatial, mita, decoder, head, other = _classify_loaded_keys(set(compatible))
    audit = CheckpointAudit(
        checkpoint_path=str(checkpoint_path),
        checkpoint_sha256=sha256_file(checkpoint_path),
        source_key_count=len(source_state),
        loaded_spatial_keys=spatial,
        loaded_mita_keys=mita,
        loaded_decoder_keys=decoder,
        loaded_head_keys=head,
        loaded_other_keys=other,
        removed_temporal_keys=tuple(sorted(removed_temporal)),
        new_tcn_keys=new_tcn,
        missing_keys=missing,
        unexpected_keys=unexpected_tuple,
        shape_mismatch_keys=shape_mismatch_tuple,
        shape_mismatch_details=shape_mismatch_details_tuple,
    )
    if audit.loaded_key_count != len(source_state) - len(removed_temporal):
        raise RuntimeError(f"Checkpoint key accounting failed: {audit}")
    return audit


def parameter_audit(model: nn.Module) -> dict[str, Any]:
    coarse_temporal_parameters = sum(
        parameter.numel() for parameter in model.temporal_mixer.parameters()
    )
    fine_temporal_module = getattr(model, "fine_temporal_mixer", None)
    fine_temporal_parameters = (
        sum(parameter.numel() for parameter in fine_temporal_module.parameters())
        if isinstance(fine_temporal_module, nn.Module)
        else 0
    )
    temporal_parameters = coarse_temporal_parameters + fine_temporal_parameters
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameters = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    original_temporal_parameters = 0
    reduction_ratio = (
        (original_temporal_parameters - temporal_parameters)
        / original_temporal_parameters
        if original_temporal_parameters
        else None
    )
    return {
        "original_temporal_mixer_parameters": original_temporal_parameters,
        "dwtcn_parameters": temporal_parameters,
        "coarse_dwtcn_parameters": coarse_temporal_parameters,
        "fine_dwtcn_parameters": fine_temporal_parameters,
        "total_model_parameters": total_parameters,
        "trainable_parameters": trainable_parameters,
        "parameter_reduction_ratio": reduction_ratio,
        "parameter_delta_vs_original": temporal_parameters,
        "parameter_increase_percent_vs_original": (
            100.0 * temporal_parameters / (total_parameters - temporal_parameters)
            if temporal_parameters
            else 0.0
        ),
    }
