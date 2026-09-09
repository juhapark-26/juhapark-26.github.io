"""Train and evaluate PulseFormer-Original or PulseFormer-DWTCN."""

from __future__ import annotations

import argparse
import json
import os
import re
import random
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from .base import audited_load_checkpoint, build_pulseformer, parameter_audit
from .data import build_fold_split, build_loader, build_participant_records, load_yaml
from .evaluation import collect_model_outputs, evaluate_waveform_store, validation_main_loss
from .losses import baseline_z_normalize
from .temporal import DilatedDepthwiseTCN


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
TRAINABLE_SCOPES = {
    "frozen",
    "full",
    "temporal_mixer_only",
    "temporal_mixer_and_late_backbone",
}
LATE_BACKBONE_PARAMETER_PREFIXES = (
    "ConvBlock8.",
    "ConvBlock9.",
    "upsample.",
    "upsample2.",
    "ConvBlock10.",
)
LATE_BACKBONE_MODULE_NAMES = (
    "ConvBlock8",
    "ConvBlock9",
    "upsample",
    "upsample2",
    "ConvBlock10",
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _json_ready(value: Any) -> Any:
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value


def write_json(path: Path, payload: Any) -> None:
    path = path.resolve()
    if not path.is_relative_to(WORKSPACE_ROOT):
        raise ValueError(f"Refusing to write outside {WORKSPACE_ROOT}: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(_json_ready(payload), handle, indent=2, sort_keys=True)
        handle.write("\n")


def resolve_base_checkpoint(config: dict[str, Any], fold: int) -> Path:
    best_epochs = config["base"]["best_epochs"]
    epoch = best_epochs.get(fold, best_epochs.get(str(fold)))
    if epoch is None:
        raise KeyError(f"No best epoch configured for fold {fold}")
    checkpoint = config["base"]["checkpoint_template"].format(
        fold=fold,
        epoch=epoch,
    )
    path = Path(checkpoint)
    if not path.is_file():
        raise FileNotFoundError(f"Base checkpoint does not exist: {path}")
    return path


def _validate_output_root(config: dict[str, Any]) -> Path:
    output_root = Path(config["paths"]["output_root"]).resolve()
    if not output_root.is_relative_to(WORKSPACE_ROOT):
        raise ValueError(
            f"output_root must stay inside {WORKSPACE_ROOT}, got {output_root}"
        )
    return output_root


def _resolve_run_root(output_root: Path, run_id: str | None) -> Path:
    if run_id is None:
        return output_root
    if not RUN_ID_PATTERN.fullmatch(run_id):
        raise ValueError(f"Invalid run ID: {run_id!r}")
    run_root = (output_root / "runs" / run_id).resolve()
    if not run_root.is_relative_to(output_root):
        raise ValueError(f"Run output escapes output_root: {run_root}")
    return run_root


def build_experiment_model(
    config: dict[str, Any],
    fold: int,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    model = build_pulseformer(
        model_config=config["model"],
        frames=int(config["data"]["chunk_length"]),
    )
    checkpoint_path = resolve_base_checkpoint(config, fold)
    checkpoint_audit = audited_load_checkpoint(model, checkpoint_path)
    audit = {
        "checkpoint": checkpoint_audit.to_dict(),
        "parameters": parameter_audit(model),
        "model_class": type(model).__name__,
    }
    return model, audit


def probe_architecture(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> dict[str, Any]:
    """Measure the real data-path shapes with forward hooks."""

    observed: dict[str, list[int]] = {}
    handles: list[Any] = []

    def capture(name: str):
        def hook(
            module: torch.nn.Module,
            inputs: tuple[torch.Tensor, ...],
            output: torch.Tensor,
        ) -> None:
            del module, inputs
            observed[name] = list(output.shape)

        return hook

    def capture_input(name: str):
        def hook(
            module: torch.nn.Module,
            inputs: tuple[torch.Tensor, ...],
        ) -> None:
            del module
            observed[name] = list(inputs[0].shape)

        return hook

    for name in ("ConvBlock9", "upsample", "upsample2", "poolspa"):
        handles.append(getattr(model, name).register_forward_hook(capture(name)))
    handles.append(
        model.upsample.register_forward_pre_hook(capture_input("decoder_input"))
    )
    if isinstance(model.temporal_mixer, DilatedDepthwiseTCN):
        handles.append(
            model.temporal_mixer.register_forward_hook(capture("temporal_mixer"))
        )
        handles.append(
            model.temporal_mixer.register_forward_pre_hook(
                capture_input("temporal_mixer_input")
            )
        )
    fine_temporal_mixer = getattr(model, "fine_temporal_mixer", None)
    if isinstance(fine_temporal_mixer, DilatedDepthwiseTCN):
        handles.append(
            fine_temporal_mixer.register_forward_hook(
                capture("fine_temporal_mixer")
            )
        )
        handles.append(
            fine_temporal_mixer.register_forward_pre_hook(
                capture_input("fine_temporal_mixer_input")
            )
        )

    video, imu, _, _, _ = next(iter(loader))
    video = video[:1].to(device)
    imu = imu[:1].to(device)
    model.eval()
    try:
        with torch.inference_mode():
            output = model(video, imu)
    finally:
        for handle in handles:
            handle.remove()

    bottleneck = observed["ConvBlock9"]
    insertion_point = model.temporal_insertion_point
    audit: dict[str, Any] = {
        "input_shape": list(video.shape),
        "input_frame_length": video.shape[2],
        "temporal_insertion_point": insertion_point,
        "spatial_encoder_bottleneck_shape": bottleneck,
        "decoder_input_shape": observed["decoder_input"],
        "upsample_shape": observed["upsample"],
        "last_prepool_feature_shape": observed["upsample2"],
        "existing_poolspa_shape": observed["poolspa"],
        "output_shape": list(output.shape),
    }
    if isinstance(model.temporal_mixer, DilatedDepthwiseTCN):
        mixer_input = observed["temporal_mixer_input"]
        mixer_output = observed["temporal_mixer"]
        if mixer_output != mixer_input:
            raise RuntimeError(
                "Temporal mixer did not preserve [N,C,T] shape"
            )
        if insertion_point == "encoder_bottleneck":
            expected_mixer_shape = [
                bottleneck[0] * bottleneck[3] * bottleneck[4],
                bottleneck[1],
                bottleneck[2],
            ]
            if mixer_input != expected_mixer_shape:
                raise RuntimeError(
                    "Per-location bottleneck token shape mismatch: "
                    f"expected={expected_mixer_shape}, actual={mixer_input}"
                )
            if observed["decoder_input"] != bottleneck:
                raise RuntimeError(
                    "Bottleneck feature shape was not restored before the decoder"
                )
        elif insertion_point == "post_pool":
            expected_mixer_shape = [
                observed["poolspa"][0],
                observed["poolspa"][1],
                observed["poolspa"][2],
            ]
            if mixer_input != expected_mixer_shape:
                raise RuntimeError(
                    "Post-pool token shape mismatch: "
                    f"expected={expected_mixer_shape}, actual={mixer_input}"
                )
        else:
            raise RuntimeError(f"Unexpected temporal insertion point: {insertion_point}")
        temporal_tokens = mixer_input[2]
        audit.update(
            {
                "temporal_mixer_input_shape": mixer_input,
                "temporal_mixer_shape": mixer_output,
                "temporal_token_length": temporal_tokens,
                "spatial_locations_per_sample": mixer_input[0] // video.shape[0],
                "kernel_size": model.temporal_mixer.kernel_size,
                "dilations": list(model.temporal_mixer.dilations),
                "receptive_field": model.temporal_mixer.receptive_field,
                "coverage_ratio": model.temporal_mixer.coverage_ratio(temporal_tokens),
                "causal": model.temporal_mixer.causal,
                "full_model_causal": False,
            }
        )
    else:
        audit.update(
            {
                "temporal_mixer_shape": None,
                "temporal_mixer_input_shape": None,
                "temporal_token_length": None,
                "receptive_field": None,
                "coverage_ratio": None,
            }
        )
    if isinstance(fine_temporal_mixer, DilatedDepthwiseTCN):
        fine_input = observed["fine_temporal_mixer_input"]
        fine_output = observed["fine_temporal_mixer"]
        if fine_output != fine_input:
            raise RuntimeError(
                "Fine temporal mixer did not preserve [B,C,T] shape"
            )
        expected_fine_shape = [
            observed["poolspa"][0],
            observed["poolspa"][1],
            observed["poolspa"][2],
        ]
        if fine_input != expected_fine_shape:
            raise RuntimeError(
                "Full-resolution fine token shape mismatch: "
                f"expected={expected_fine_shape}, actual={fine_input}"
            )
        fine_tokens = fine_input[2]
        audit.update(
            {
                "fine_temporal_insertion_point": getattr(
                    model,
                    "fine_temporal_insertion_point",
                    None,
                ),
                "fine_temporal_mixer_input_shape": fine_input,
                "fine_temporal_mixer_shape": fine_output,
                "fine_temporal_token_length": fine_tokens,
                "fine_kernel_size": fine_temporal_mixer.kernel_size,
                "fine_dilations": list(fine_temporal_mixer.dilations),
                "fine_receptive_field": fine_temporal_mixer.receptive_field,
                "fine_coverage_ratio": fine_temporal_mixer.coverage_ratio(
                    fine_tokens
                ),
                "fine_causal": fine_temporal_mixer.causal,
            }
        )
    else:
        audit.update(
            {
                "fine_temporal_insertion_point": None,
                "fine_temporal_mixer_input_shape": None,
                "fine_temporal_mixer_shape": None,
                "fine_temporal_token_length": None,
                "fine_receptive_field": None,
                "fine_coverage_ratio": None,
            }
        )
    return audit


def _trainable_scope(config: dict[str, Any]) -> str:
    max_steps = int(config["training"]["max_steps"])
    default_scope = "frozen" if max_steps == 0 else "full"
    scope = str(config["training"].get("trainable_scope", default_scope)).lower()
    if scope not in TRAINABLE_SCOPES:
        raise ValueError(
            f"Unsupported trainable_scope={scope!r}; expected {sorted(TRAINABLE_SCOPES)}"
        )
    if max_steps == 0 and scope != "frozen":
        raise ValueError("max_steps=0 requires trainable_scope=frozen")
    if max_steps > 0 and scope == "frozen":
        raise ValueError("A frozen experiment must use max_steps=0")
    return scope


def _configure_trainable_parameters(
    model: torch.nn.Module,
    scope: str,
) -> tuple[str, ...]:
    named_parameters = tuple(model.named_parameters())
    if scope == "full":
        selected_names = {name for name, _ in named_parameters}
    elif scope == "frozen":
        selected_names = set()
    elif scope == "temporal_mixer_only":
        if not isinstance(model.temporal_mixer, DilatedDepthwiseTCN):
            raise ValueError(
                "temporal_mixer_only requires a DilatedDepthwiseTCN model"
            )
        selected_names = {
            name for name, _ in named_parameters if name.startswith("temporal_mixer.")
        }
    elif scope == "temporal_mixer_and_late_backbone":
        if (
            not isinstance(model.temporal_mixer, DilatedDepthwiseTCN)
            or getattr(model, "temporal_insertion_point", None)
            != "encoder_bottleneck"
        ):
            raise ValueError(
                "temporal_mixer_and_late_backbone requires the bottleneck DW-TCN"
            )
        selected_names = {
            name
            for name, _ in named_parameters
            if name.startswith("temporal_mixer.")
            or name.startswith(LATE_BACKBONE_PARAMETER_PREFIXES)
        }
    else:
        raise AssertionError(f"Unhandled trainable scope: {scope}")

    if scope not in {"frozen", "full"} and not selected_names:
        raise RuntimeError(f"No parameters selected for trainable_scope={scope}")
    for name, parameter in named_parameters:
        parameter.requires_grad = name in selected_names
    actual_names = {
        name for name, parameter in named_parameters if parameter.requires_grad
    }
    if actual_names != selected_names:
        raise RuntimeError(
            "Trainable parameter selection diverged from its allowlist: "
            f"missing={sorted(selected_names - actual_names)}, "
            f"unexpected={sorted(actual_names - selected_names)}"
        )
    return tuple(sorted(actual_names))


def _set_training_mode(model: torch.nn.Module, scope: str) -> None:
    if scope == "full":
        model.train()
        return
    model.eval()
    if scope == "temporal_mixer_only":
        model.temporal_mixer.train()
    elif scope == "temporal_mixer_and_late_backbone":
        model.temporal_mixer.train()
        for module_name in LATE_BACKBONE_MODULE_NAMES:
            getattr(model, module_name).train()
    elif scope != "frozen":
        raise AssertionError(f"Unhandled trainable scope: {scope}")


def _build_optimizer(
    model: torch.nn.Module,
    config: dict[str, Any],
) -> torch.optim.Optimizer | None:
    scope = _trainable_scope(config)
    trainable_names = _configure_trainable_parameters(model, scope)
    if scope == "frozen":
        return None
    optimizer_config = config["training"]["optimizer"]
    optimizer_name = str(optimizer_config.get("name", "AdamW")).lower()
    if optimizer_name != "adamw":
        raise ValueError(f"Only AdamW is supported, got {optimizer_name!r}")

    named_parameters = dict(model.named_parameters())
    weight_decay = float(optimizer_config["weight_decay"])
    parameter_groups: list[dict[str, Any]] = []
    if scope == "full":
        parameter_groups.append(
            {
                "name": "full_model",
                "params": [named_parameters[name] for name in trainable_names],
                "lr": float(
                    optimizer_config.get(
                        "backbone_learning_rate",
                        optimizer_config["learning_rate"],
                    )
                ),
            }
        )
    elif scope == "temporal_mixer_only":
        parameter_groups.append(
            {
                "name": "temporal_mixer",
                "params": [named_parameters[name] for name in trainable_names],
                "lr": float(
                    optimizer_config.get(
                        "temporal_mixer_learning_rate",
                        optimizer_config["learning_rate"],
                    )
                ),
            }
        )
    elif scope == "temporal_mixer_and_late_backbone":
        temporal_names = tuple(
            name for name in trainable_names if name.startswith("temporal_mixer.")
        )
        late_names = tuple(
            name
            for name in trainable_names
            if name.startswith(LATE_BACKBONE_PARAMETER_PREFIXES)
        )
        parameter_groups.extend(
            [
                {
                    "name": "temporal_mixer",
                    "params": [named_parameters[name] for name in temporal_names],
                    "lr": float(optimizer_config["temporal_mixer_learning_rate"]),
                },
                {
                    "name": "late_backbone",
                    "params": [named_parameters[name] for name in late_names],
                    "lr": float(optimizer_config["backbone_learning_rate"]),
                },
            ]
        )

    grouped_parameters = [
        parameter for group in parameter_groups for parameter in group["params"]
    ]
    grouped_ids = [id(parameter) for parameter in grouped_parameters]
    expected_ids = {
        id(parameter) for parameter in model.parameters() if parameter.requires_grad
    }
    if len(grouped_ids) != len(set(grouped_ids)):
        raise RuntimeError("An optimizer parameter appears in multiple groups")
    if set(grouped_ids) != expected_ids:
        raise RuntimeError("Optimizer groups do not exactly cover trainable parameters")
    if any(not group["params"] for group in parameter_groups):
        raise RuntimeError("Optimizer parameter groups must not be empty")

    return torch.optim.AdamW(
        parameter_groups,
        weight_decay=weight_decay,
    )


def _training_policy_audit(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
    config: dict[str, Any],
) -> dict[str, Any]:
    scope = _trainable_scope(config)
    trainable = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]
    groups: list[dict[str, Any]] = []
    if optimizer is not None:
        for index, group in enumerate(optimizer.param_groups):
            groups.append(
                {
                    "name": group.get("name", f"group_{index}"),
                    "learning_rate": float(group["lr"]),
                    "parameter_tensors": len(group["params"]),
                    "parameters": sum(
                        parameter.numel() for parameter in group["params"]
                    ),
                }
            )
    return {
        "trainable_scope": scope,
        "trainable_parameter_names": [name for name, _ in trainable],
        "trainable_parameter_tensors": len(trainable),
        "trainable_parameters": sum(parameter.numel() for _, parameter in trainable),
        "frozen_parameters": sum(
            parameter.numel()
            for parameter in model.parameters()
            if not parameter.requires_grad
        ),
        "optimizer_groups": groups,
    }


class _TemporalResidualMonitor:
    def __init__(self, model: torch.nn.Module) -> None:
        self.latest: dict[str, float | None] = {
            "base_feature_norm": None,
            "delta_feature_norm": None,
            "delta_to_base_ratio": None,
        }
        self.handle = None
        if isinstance(model.temporal_mixer, DilatedDepthwiseTCN):
            self.handle = model.temporal_mixer.register_forward_hook(self._capture)

    def _capture(
        self,
        module: torch.nn.Module,
        inputs: tuple[torch.Tensor, ...],
        output: torch.Tensor,
    ) -> None:
        del module
        base = inputs[0].detach().float()
        delta = output.detach().float() - base
        base_norm = float(torch.linalg.vector_norm(base))
        delta_norm = float(torch.linalg.vector_norm(delta))
        self.latest = {
            "base_feature_norm": base_norm,
            "delta_feature_norm": delta_norm,
            "delta_to_base_ratio": delta_norm / (base_norm + 1.0e-8),
        }

    def close(self) -> None:
        if self.handle is not None:
            self.handle.remove()


def _gradient_norm(parameter: torch.nn.Parameter) -> float | None:
    if parameter.grad is None:
        return None
    return float(torch.linalg.vector_norm(parameter.grad.detach().float()))


def _temporal_training_record(model: torch.nn.Module) -> dict[str, Any]:
    if not isinstance(model.temporal_mixer, DilatedDepthwiseTCN):
        return {
            "output_projection_weight_norm": None,
            "output_projection_gradient_norm": None,
            "depthwise_gradient_norm_per_block": [],
            "glu_in_gradient_norm": None,
            "glu_out_gradient_norm": None,
            "layer_scale_mean_per_block": [],
        }
    mixer = model.temporal_mixer
    return {
        "output_projection_weight_norm": float(
            torch.linalg.vector_norm(mixer.output_projection.weight.detach().float())
        ),
        "output_projection_gradient_norm": _gradient_norm(
            mixer.output_projection.weight
        ),
        "depthwise_gradient_norm_per_block": [
            _gradient_norm(block.depthwise_conv.weight) for block in mixer.blocks
        ],
        "glu_in_gradient_norm": _gradient_norm(
            mixer.pointwise_glu.glu_in.weight
        ),
        "glu_out_gradient_norm": _gradient_norm(
            mixer.pointwise_glu.glu_out.weight
        ),
        "layer_scale_mean_per_block": [
            float(block.layer_scale.detach().float().mean()) for block in mixer.blocks
        ],
    }


def _gradient_audit(model: torch.nn.Module) -> dict[str, float | bool]:
    gradients = [
        parameter.grad.detach()
        for parameter in model.parameters()
        if parameter.grad is not None
    ]
    if not gradients:
        return {"gradient_finite": False, "gradient_norm": 0.0}
    finite = all(bool(torch.isfinite(gradient).all()) for gradient in gradients)
    norm = float(torch.sqrt(sum(gradient.float().pow(2).sum() for gradient in gradients)))
    return {"gradient_finite": finite, "gradient_norm": norm}


def _train(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer | None,
    config: dict[str, Any],
    device: torch.device,
    log_path: Path,
) -> dict[str, Any]:
    max_steps = int(config["training"]["max_steps"])
    if optimizer is None or max_steps == 0:
        return {"global_step": 0, "elapsed_seconds": 0.0, "last_loss": None}

    amp_enabled = bool(config["training"]["amp"] and device.type == "cuda")
    amp_dtype_name = str(config["training"].get("amp_dtype", "float16")).lower()
    amp_dtype_by_name = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    if amp_dtype_name not in amp_dtype_by_name:
        raise ValueError(
            f"Unsupported amp_dtype={amp_dtype_name!r}; "
            f"expected {sorted(amp_dtype_by_name)}"
        )
    amp_dtype = amp_dtype_by_name[amp_dtype_name]
    if (
        amp_enabled
        and amp_dtype is torch.bfloat16
        and not torch.cuda.is_bf16_supported()
    ):
        raise RuntimeError("Configured bfloat16 AMP is not supported by this GPU")
    scaler = torch.amp.GradScaler(
        device.type,
        enabled=amp_enabled and amp_dtype is torch.float16,
    )
    gradient_clip = float(config["training"].get("gradient_clip_norm", 0.0))
    log_every = int(config["training"].get("log_every_steps", 25))
    iterator = iter(loader)
    started = time.time()
    last_record: dict[str, Any] = {}
    scope = _trainable_scope(config)
    trainable_parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    residual_monitor = _TemporalResidualMonitor(model)
    log_path = log_path.resolve()
    if not log_path.is_relative_to(WORKSPACE_ROOT):
        raise ValueError(f"Refusing to write outside {WORKSPACE_ROOT}: {log_path}")
    log_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        with log_path.open("w", encoding="utf-8") as log_handle:
            for step in range(max_steps):
                try:
                    video, imu, target, _, _ = next(iterator)
                except StopIteration:
                    iterator = iter(loader)
                    video, imu, target, _, _ = next(iterator)

                _set_training_mode(model, scope)
                video = video.to(device, non_blocking=True)
                imu = imu.to(device, non_blocking=True)
                target = target.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(
                    device_type=device.type,
                    dtype=amp_dtype,
                    enabled=amp_enabled,
                ):
                    prediction = model(video, imu)
                    loss = F.mse_loss(
                        baseline_z_normalize(prediction),
                        baseline_z_normalize(target),
                    )

                if not torch.isfinite(loss):
                    raise FloatingPointError(
                        f"Non-finite loss at step {step}: {float(loss)}"
                    )
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                gradient_record = _gradient_audit(model)
                if not gradient_record["gradient_finite"]:
                    raise FloatingPointError(f"Non-finite gradient at step {step}")
                temporal_record = _temporal_training_record(model)
                if gradient_clip > 0:
                    torch.nn.utils.clip_grad_norm_(
                        trainable_parameters,
                        gradient_clip,
                    )
                scaler.step(optimizer)
                scaler.update()

                waveform_mse = float(loss.detach())
                learning_rates = {
                    str(group.get("name", f"group_{index}")): float(group["lr"])
                    for index, group in enumerate(optimizer.param_groups)
                }
                record = {
                    "step": step + 1,
                    "global_step": step + 1,
                    "total_loss": waveform_mse,
                    "waveform_mse": waveform_mse,
                    "learning_rate": float(optimizer.param_groups[0]["lr"]),
                    "learning_rates": learning_rates,
                    "amp_enabled": amp_enabled,
                    "amp_dtype": amp_dtype_name if amp_enabled else None,
                    "elapsed_seconds": time.time() - started,
                    **gradient_record,
                    **residual_monitor.latest,
                    **temporal_record,
                }
                ratio = record["delta_to_base_ratio"]
                record["delta_ratio_at_or_above_one"] = bool(
                    ratio is not None and ratio >= 1.0
                )
                last_record = record
                log_handle.write(
                    json.dumps(_json_ready(record), sort_keys=True) + "\n"
                )
                if (
                    step == 0
                    or (step + 1) % log_every == 0
                    or step + 1 == max_steps
                ):
                    log_handle.flush()
                    print(
                        f"step={step + 1}/{max_steps} "
                        f"loss={record['total_loss']:.6f} "
                        f"elapsed={record['elapsed_seconds']:.1f}s",
                        flush=True,
                    )
    finally:
        residual_monitor.close()

    return {
        "global_step": max_steps,
        "elapsed_seconds": time.time() - started,
        "last_loss": last_record,
    }


def run_fold(
    config_path: str | Path,
    fold: int,
    device_name: str,
    force: bool = False,
    run_id: str | None = None,
) -> Path:
    fold_started = datetime.now().astimezone()
    config_path = Path(config_path).resolve()
    config = load_yaml(config_path)
    output_root = _validate_output_root(config)
    run_root = _resolve_run_root(output_root, run_id)
    if run_id is not None:
        config["experiment"]["run_id"] = run_id
    set_seed(int(config["experiment"]["seed"]))
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but CUDA is unavailable")

    output_dir = (
        run_root / config["experiment"]["output_name"] / f"fold_{fold}"
    ).resolve()
    if not output_dir.is_relative_to(run_root):
        raise ValueError(
            f"Resolved experiment output escapes run root: {output_dir}"
        )
    if output_dir.exists() and force:
        shutil.rmtree(output_dir)
    if (output_dir / "metrics.json").is_file() and not force:
        print(f"Completed output already exists, skipping: {output_dir}")
        return output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(config_path, output_dir / "config_source.yaml")
    with (output_dir / "config_resolved.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)
    launch_command = os.environ.get("ECCVW_LAUNCH_COMMAND")
    if launch_command:
        (output_dir / "command.txt").write_text(launch_command + "\n", encoding="utf-8")
    run_metadata = {
        "run_id": run_id,
        "status": "running",
        "started_at": fold_started.isoformat(),
        "completed_at": None,
        "fold": fold,
        "seed": int(config["experiment"]["seed"]),
        "device_requested": device_name,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    write_json(output_dir / "run_metadata.json", run_metadata)

    records_by_participant = build_participant_records(
        data_dir=config["paths"]["preprocessed_data"],
        preprocessing_config_path=config["paths"]["preprocessing_config"],
        tasks_to_use=config["data"]["tasks_to_use"],
        chunk_length=int(config["data"]["chunk_length"]),
        downsampling=int(config["data"]["downsampling"]),
    )
    split = build_fold_split(
        records_by_participant,
        fold=fold,
        seed=int(config["experiment"]["seed"]),
        number_of_folds=int(config["data"]["number_of_folds"]),
    )
    write_json(output_dir / "split_audit.json", split.summary())

    train_loader = build_loader(
        split.train_records,
        batch_size=int(config["training"]["batch_size"]),
        augment=True,
        shuffle=True,
        seed=int(config["experiment"]["seed"]),
        num_workers=int(config["training"]["num_workers"]),
    )
    valid_loader = build_loader(
        split.valid_records,
        batch_size=int(config["evaluation"]["batch_size"]),
        augment=False,
        shuffle=False,
        seed=int(config["experiment"]["seed"]),
        num_workers=int(config["evaluation"]["num_workers"]),
    )
    test_loader = build_loader(
        split.test_records,
        batch_size=int(config["evaluation"]["batch_size"]),
        augment=False,
        shuffle=False,
        seed=int(config["experiment"]["seed"]),
        num_workers=int(config["evaluation"]["num_workers"]),
    )

    model, model_audit = build_experiment_model(config, fold)
    model = model.to(device)
    model_audit["architecture"] = probe_architecture(model, valid_loader, device)
    optimizer = _build_optimizer(model, config)
    model_audit["parameters_after_training_policy"] = parameter_audit(model)
    model_audit["training_policy"] = _training_policy_audit(
        model,
        optimizer,
        config,
    )
    write_json(output_dir / "model_audit.json", model_audit)
    train_summary = _train(
        model,
        train_loader,
        optimizer,
        config,
        device,
        output_dir / "train_log.jsonl",
    )

    validation_loss = validation_main_loss(model, valid_loader, device)
    checkpoint_payload = {
        "model_state_dict": model.state_dict(),
        "base_checkpoint_path": model_audit["checkpoint"]["checkpoint_path"],
        "base_checkpoint_sha256": model_audit["checkpoint"]["checkpoint_sha256"],
        "config": config,
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "global_step": train_summary["global_step"],
        "validation_main_waveform_mse": validation_loss,
    }
    if optimizer is not None:
        torch.save(checkpoint_payload, output_dir / "checkpoint.pt")

    waveform_store, diagnostics = collect_model_outputs(model, test_loader, device)
    metrics = evaluate_waveform_store(
        waveform_store,
        source_root=config["paths"]["source_root"],
        fs=int(config["evaluation"]["fs"]),
        window_seconds=int(config["evaluation"]["window_seconds"]),
        hr_method=config["evaluation"]["hr_method"],
    )
    fold_completed = datetime.now().astimezone()
    summary = {
        "run": {
            **run_metadata,
            "status": "completed",
            "completed_at": fold_completed.isoformat(),
            "elapsed_seconds": (fold_completed - fold_started).total_seconds(),
        },
        "experiment": config["experiment"],
        "fold": fold,
        "validation_main_waveform_mse": validation_loss,
        "training": train_summary,
        "metrics": metrics,
        "diagnostics": diagnostics,
        "model_audit": model_audit,
        "split": split.summary(),
    }
    write_json(output_dir / "metrics.json", summary)
    write_json(output_dir / "run_metadata.json", summary["run"])
    print(json.dumps(_json_ready(summary), indent=2, sort_keys=True), flush=True)
    return output_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--fold", required=True, type=int)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--run-id")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_fold(
        args.config,
        args.fold,
        args.device,
        force=args.force,
        run_id=args.run_id,
    )


if __name__ == "__main__":
    main()
