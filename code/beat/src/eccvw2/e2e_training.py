"""Paper-protocol end-to-end training for fresh PulseFormer models.

This module is deliberately separate from :mod:`eccvw2.training`: the latter
implements the historical fixed-step checkpoint experiments, whereas this
module runs every batch of 100 complete epochs and selects the best validation
epoch before testing.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import yaml
from tqdm import tqdm

from . import base as base_module
from .base import parameter_audit
from .data import (
    build_cross_clip_loader,
    build_fold_split,
    build_loader,
    build_participant_records,
    load_yaml,
)
from .evaluation import (
    collect_model_outputs,
    evaluate_waveform_store,
    validation_loss_breakdown,
)
from .losses import (
    LOSS_COMPONENT_KEYS,
    TOML_DIAGNOSTIC_KEYS,
    MotionDisentangledMultiDomainLoss,
    TOMLNumericalError,
    build_training_loss,
    loss_config_sha256,
    resolve_training_loss_config,
)
from .training import (
    WORKSPACE_ROOT,
    _gradient_audit,
    _json_ready,
    _resolve_run_root,
    _validate_output_root,
    probe_architecture,
    set_seed,
    write_json,
)


ValidationFunction = Callable[
    [torch.nn.Module, torch.utils.data.DataLoader, torch.device], float
]


def _validate_paper_protocol(config: dict[str, Any]) -> None:
    """Fail closed when a run drifts from the requested paper protocol."""

    training = config["training"]
    validation = config["validation"]
    evaluation = config["evaluation"]
    initialization = config.get("initialization", {})
    optimizer = training["optimizer"]
    scheduler = training["scheduler"]

    expected = {
        "training.epochs": (int(training["epochs"]), 100),
        "training.trainable_scope": (
            str(training.get("trainable_scope", "full")).lower(),
            "full",
        ),
        "training.batch_size": (int(training["batch_size"]), 4),
        "training.optimizer.name": (str(optimizer["name"]).lower(), "adam"),
        "training.optimizer.learning_rate": (
            float(optimizer["learning_rate"]),
            9.0e-4,
        ),
        "training.scheduler.name": (
            str(scheduler["name"]).lower(),
            "onecyclelr",
        ),
        "training.scheduler.max_lr": (float(scheduler["max_lr"]), 9.0e-4),
        "validation.batch_size": (int(validation["batch_size"]), 4),
        "evaluation.batch_size": (int(evaluation["batch_size"]), 1),
        "initialization.mode": (
            str(initialization.get("mode", "fresh_paper")).lower(),
            "fresh_paper",
        ),
    }
    drift = [
        f"{name}: expected {wanted!r}, got {actual!r}"
        for name, (actual, wanted) in expected.items()
        if actual != wanted
    ]
    if drift:
        raise ValueError("Paper protocol configuration drift:\n" + "\n".join(drift))
    if int(config["data"]["number_of_folds"]) != 5:
        raise ValueError("Paper protocol requires exactly five folds")
    if bool(training.get("amp", False)):
        raise ValueError("Paper protocol uses full-precision training (amp=false)")
    selection_metric = str(
        validation.get("selection_metric", "waveform")
    ).lower()
    if selection_metric != "waveform":
        raise ValueError(
            "For fair loss ablations, validation.selection_metric must be waveform"
        )
    resolved_loss = resolve_training_loss_config(
        config.get("loss"),
        expected_length=int(config["data"]["chunk_length"]),
        evaluation_fs=float(evaluation["fs"]),
    )
    cross_clip = training.get("cross_clip", {"enabled": False})
    if not isinstance(cross_clip, dict):
        raise TypeError("training.cross_clip must be a mapping")
    peak_config = resolved_loss.get("peak_consistency")
    if peak_config is not None and peak_config["enabled"]:
        if not bool(cross_clip.get("enabled", False)):
            raise ValueError(
                "Enabled peak consistency requires training.cross_clip.enabled=true"
            )
        if int(cross_clip.get("group_size", -1)) != int(
            peak_config["group_size"]
        ):
            raise ValueError(
                "training.cross_clip.group_size must match loss peak group_size"
            )
        if int(training["batch_size"]) != int(peak_config["group_size"]):
            raise ValueError(
                "Paper batch_size must equal the cross-clip group_size"
            )
        cross_band = [float(value) for value in cross_clip.get("band_hz", [])]
        if cross_band != peak_config["band_hz"]:
            raise ValueError(
                "training.cross_clip.band_hz must match peak loss band_hz"
            )


    ccpd_config = resolved_loss.get("ccpd")
    if ccpd_config is not None:
        if not bool(cross_clip.get("enabled", False)):
            raise ValueError(
                "CCPD experiments require training.cross_clip.enabled=true"
            )
        if int(cross_clip.get("group_size", -1)) != int(
            ccpd_config["group_size"]
        ):
            raise ValueError(
                "training.cross_clip.group_size must match loss CCPD group_size"
            )
        if int(training["batch_size"]) != int(ccpd_config["group_size"]):
            raise ValueError(
                "Paper batch_size must equal the CCPD group_size"
            )
        cross_band = [
            float(value) for value in cross_clip.get("band_hz", [])
        ]
        if cross_band != ccpd_config["band_hz"]:
            raise ValueError(
                "training.cross_clip.band_hz must match CCPD band_hz"
            )
        tail_config = cross_clip.get("tail_balanced_sampling", {})
        if not isinstance(tail_config, dict):
            raise TypeError(
                "training.cross_clip.tail_balanced_sampling must be a mapping"
            )
        if bool(tail_config.get("enabled", True)):
            raise ValueError(
                "Clean CCPD ablations require tail-balanced sampling disabled"
            )


def _build_optimizer_and_scheduler(
    model: torch.nn.Module,
    training_config: dict[str, Any],
    steps_per_epoch: int,
) -> tuple[torch.optim.Adam, torch.optim.lr_scheduler.OneCycleLR]:
    """Build the original Adam + per-batch OneCycleLR policy."""

    if steps_per_epoch <= 0:
        raise ValueError("The training loader must contain at least one batch")
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    trainable = list(model.parameters())
    if not trainable or not all(parameter.requires_grad for parameter in trainable):
        raise RuntimeError("End-to-end training requires every parameter to be trainable")

    optimizer_config = training_config["optimizer"]
    scheduler_config = training_config["scheduler"]
    if str(optimizer_config["name"]).lower() != "adam":
        raise ValueError("Paper protocol requires Adam")
    if str(scheduler_config["name"]).lower() != "onecyclelr":
        raise ValueError("Paper protocol requires OneCycleLR")

    optimizer = torch.optim.Adam(
        trainable,
        lr=float(optimizer_config["learning_rate"]),
    )
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=float(scheduler_config["max_lr"]),
        epochs=int(training_config["epochs"]),
        steps_per_epoch=steps_per_epoch,
    )
    return optimizer, scheduler


def _capture_rng_state(
    loader: torch.utils.data.DataLoader | None = None,
) -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }
    generator = getattr(loader, "generator", None)
    state["train_loader_generator"] = (
        generator.get_state() if generator is not None else None
    )
    return state


def _restore_rng_state(
    state: dict[str, Any],
    loader: torch.utils.data.DataLoader | None = None,
) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if state.get("cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([item.cpu() for item in state["cuda"]])
    generator = getattr(loader, "generator", None)
    loader_state = state.get("train_loader_generator")
    if generator is not None and loader_state is not None:
        generator.set_state(loader_state.cpu())


def _atomic_torch_save(payload: dict[str, Any], path: Path) -> None:
    path = path.resolve()
    if not path.is_relative_to(WORKSPACE_ROOT):
        raise ValueError(f"Refusing to write outside {WORKSPACE_ROOT}: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _append_numerical_event(output_dir: Path, payload: dict[str, Any]) -> None:
    """Persist a numerical failure before aborting the current optimizer step."""

    record = {
        "timestamp": datetime.now().astimezone().isoformat(),
        **payload,
    }
    event_path = output_dir / "numerical_events.jsonl"
    with event_path.open("a", encoding="utf-8") as event_handle:
        event_handle.write(json.dumps(_json_ready(record), sort_keys=True) + "\n")
        event_handle.flush()
        os.fsync(event_handle.fileno())


def _toml_diagnostics_to_python(
    diagnostics: dict[str, torch.Tensor],
) -> dict[str, int | float]:
    if not diagnostics:
        return {}
    stacked = torch.stack(
        [
            diagnostics[key].detach().to(dtype=torch.float64)
            for key in TOML_DIAGNOSTIC_KEYS
        ]
    )
    values = stacked.cpu().tolist()
    return {
        key: int(value) if key.endswith("_count") else float(value)
        for key, value in zip(TOML_DIAGNOSTIC_KEYS, values)
    }


def _checkpoint_payload(
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.OneCycleLR,
    epoch: int,
    global_step: int,
    best_epoch: int,
    best_validation_loss: float,
    validation_loss: float,
    validation_components: dict[str, float],
    criterion: MotionDisentangledMultiDomainLoss,
    config: dict[str, Any],
    train_loader: torch.utils.data.DataLoader,
) -> dict[str, Any]:
    return {
        "format_version": 2,
        "protocol": "paper_e2e_100epoch",
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "epoch": epoch,
        "global_step": global_step,
        "best_epoch": best_epoch,
        "best_validation_loss": best_validation_loss,
        "validation_loss": validation_loss,
        "selection_metric": "waveform",
        "selection_value": validation_loss,
        "validation_loss_components": validation_components,
        "loss_implementation_version": criterion.resolved_config[
            "implementation_version"
        ],
        "resolved_loss_config": criterion.resolved_config,
        "loss_config_sha256": criterion.config_sha256,
        "rng_state": _capture_rng_state(train_loader),
        "config": config,
    }


def _verify_checkpoint_loss_contract(
    payload: dict[str, Any],
    criterion: MotionDisentangledMultiDomainLoss,
    path: Path,
) -> None:
    format_version = int(payload.get("format_version", 1))
    if format_version < 2:
        if criterion.has_auxiliary_terms:
            raise RuntimeError(
                "A legacy checkpoint has no loss contract and cannot resume an "
                f"auxiliary-loss run: {path}"
            )
        return
    saved_config = payload.get("resolved_loss_config")
    saved_sha256 = payload.get("loss_config_sha256")
    if not isinstance(saved_config, dict) or not isinstance(saved_sha256, str):
        raise RuntimeError(f"Checkpoint loss contract is incomplete: {path}")
    if loss_config_sha256(saved_config) != saved_sha256:
        raise RuntimeError(f"Checkpoint loss contract hash is corrupt: {path}")
    if saved_sha256 != criterion.config_sha256:
        raise RuntimeError(
            "Resume checkpoint loss configuration differs from the current "
            f"config: {path}"
        )
    if str(payload.get("selection_metric", "")).lower() != "waveform":
        raise RuntimeError(
            f"Checkpoint selection metric is not waveform: {path}"
        )


def _load_resume_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.OneCycleLR,
    train_loader: torch.utils.data.DataLoader,
    expected_epochs: int,
    expected_steps_per_epoch: int,
    expected_criterion: MotionDisentangledMultiDomainLoss | None = None,
    expected_cross_clip_config: dict[str, Any] | None = None,
) -> tuple[int, int, int, float]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("protocol") != "paper_e2e_100epoch":
        raise RuntimeError(f"Not a paper-protocol checkpoint: {path}")
    saved_training = payload["config"]["training"]
    if int(saved_training["epochs"]) != expected_epochs:
        raise RuntimeError("Resume checkpoint epoch budget differs from the config")
    expected_total_steps = expected_epochs * expected_steps_per_epoch
    scheduler_state = payload["scheduler_state_dict"]
    if int(scheduler_state["total_steps"]) != expected_total_steps:
        raise RuntimeError("Resume checkpoint loader length differs from this run")
    if expected_cross_clip_config is not None:
        saved_cross_clip_config = saved_training.get(
            "cross_clip", {"enabled": False}
        )
        if saved_cross_clip_config != expected_cross_clip_config:
            raise RuntimeError(
                "Resume checkpoint cross-clip sampling configuration differs "
                "from the current config"
            )
    if expected_criterion is not None:
        _verify_checkpoint_loss_contract(payload, expected_criterion, path)

    model.load_state_dict(payload["model_state_dict"], strict=True)
    optimizer.load_state_dict(payload["optimizer_state_dict"])
    scheduler.load_state_dict(scheduler_state)
    _restore_rng_state(payload["rng_state"], train_loader)
    global_step = int(payload["global_step"])
    if int(scheduler.last_epoch) != global_step:
        raise RuntimeError(
            "Scheduler/global-step mismatch in resume checkpoint: "
            f"scheduler={scheduler.last_epoch}, global_step={global_step}"
        )
    return (
        int(payload["epoch"]),
        global_step,
        int(payload["best_epoch"]),
        float(payload["best_validation_loss"]),
    )


def _train_epochs(
    *,
    model: torch.nn.Module,
    train_loader: torch.utils.data.DataLoader,
    valid_loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.OneCycleLR,
    device: torch.device,
    total_epochs: int,
    output_dir: Path,
    config: dict[str, Any],
    start_epoch: int = 0,
    global_step: int = 0,
    best_epoch: int = -1,
    best_validation_loss: float = float("inf"),
    criterion: MotionDisentangledMultiDomainLoss | None = None,
    validation_function: ValidationFunction | None = None,
) -> dict[str, Any]:
    """Run complete epochs, validating and checkpointing after each one."""

    if start_epoch < 0 or start_epoch > total_epochs:
        raise ValueError(f"Invalid resume epoch {start_epoch}/{total_epochs}")
    log_path = output_dir / "train_log.jsonl"
    log_mode = "a" if start_epoch else "w"
    started = time.time()
    if criterion is None:
        data_config = config.get("data", {})
        evaluation_config = config.get("evaluation", {})
        criterion = build_training_loss(
            config.get("loss"),
            expected_length=data_config.get("chunk_length"),
            evaluation_fs=evaluation_config.get("fs"),
        )
    criterion = criterion.to(device)
    steps_per_epoch = len(train_loader)

    def toml_weight_for_step(step: int) -> tuple[float, dict[str, Any]]:
        toml_config = criterion.resolved_config["toml"]
        target_weight = float(toml_config["weight"])
        schedule = toml_config.get("schedule")
        if schedule is None:
            return target_weight, {
                "mode": "constant",
                "target_weight": target_weight,
                "warmup_steps": 0,
                "ramp_steps": 0,
            }
        warmup_steps = int(schedule["warmup_epochs"]) * steps_per_epoch
        ramp_steps = int(schedule["ramp_epochs"]) * steps_per_epoch
        if step < warmup_steps:
            effective_weight = 0.0
        elif step < warmup_steps + ramp_steps:
            completed_ramp_steps = step - warmup_steps + 1
            effective_weight = target_weight * (
                completed_ramp_steps / ramp_steps
            )
        else:
            effective_weight = target_weight
        return effective_weight, {
            "mode": schedule["mode"],
            "target_weight": target_weight,
            "warmup_steps": warmup_steps,
            "ramp_steps": ramp_steps,
        }


    def peak_weight_for_step(step: int) -> tuple[float, dict[str, Any]]:
        peak_config = criterion.resolved_config.get("peak_consistency")
        if peak_config is None or not peak_config["enabled"]:
            return 0.0, {
                "mode": "disabled",
                "target_weight": 0.0,
                "warmup_steps": 0,
                "ramp_steps": 0,
            }
        target_weight = float(peak_config["weight"])
        schedule = peak_config["schedule"]
        warmup_steps = int(schedule["warmup_epochs"]) * steps_per_epoch
        ramp_steps = int(schedule["ramp_epochs"]) * steps_per_epoch
        if step < warmup_steps:
            effective_weight = 0.0
        elif step < warmup_steps + ramp_steps:
            completed_ramp_steps = step - warmup_steps + 1
            effective_weight = target_weight * (
                completed_ramp_steps / ramp_steps
            )
        else:
            effective_weight = target_weight
        return effective_weight, {
            "mode": schedule["mode"],
            "target_weight": target_weight,
            "warmup_steps": warmup_steps,
            "ramp_steps": ramp_steps,
        }


    def ccpd_weight_for_step(step: int) -> tuple[float, dict[str, Any]]:
        ccpd_config = criterion.resolved_config.get("ccpd")
        if ccpd_config is None or not ccpd_config["enabled"]:
            return 0.0, {
                "mode": "disabled",
                "target_weight": 0.0,
                "warmup_steps": 0,
                "ramp_steps": 0,
            }
        target_weight = float(ccpd_config["weight"])
        schedule = ccpd_config["schedule"]
        warmup_steps = int(schedule["warmup_epochs"]) * steps_per_epoch
        ramp_steps = int(schedule["ramp_epochs"]) * steps_per_epoch
        if step < warmup_steps:
            effective_weight = 0.0
        elif step < warmup_steps + ramp_steps:
            completed_ramp_steps = step - warmup_steps + 1
            effective_weight = target_weight * (
                completed_ramp_steps / ramp_steps
            )
        else:
            effective_weight = target_weight
        return effective_weight, {
            "mode": schedule["mode"],
            "target_weight": target_weight,
            "warmup_steps": warmup_steps,
            "ramp_steps": ramp_steps,
        }


    def cpv_weight_for_step(step: int) -> tuple[float, dict[str, Any]]:
        cpv_config = criterion.resolved_config.get("cpv")
        if cpv_config is None or not cpv_config["enabled"]:
            return 0.0, {
                "mode": "disabled",
                "target_weight": 0.0,
                "warmup_steps": 0,
                "ramp_steps": 0,
            }
        target_weight = float(cpv_config["weight"])
        schedule = cpv_config["schedule"]
        warmup_steps = int(schedule["warmup_epochs"]) * steps_per_epoch
        ramp_steps = int(schedule["ramp_epochs"]) * steps_per_epoch
        if step < warmup_steps:
            effective_weight = 0.0
        elif step < warmup_steps + ramp_steps:
            completed_ramp_steps = step - warmup_steps + 1
            effective_weight = target_weight * (
                completed_ramp_steps / ramp_steps
            )
        else:
            effective_weight = target_weight
        return effective_weight, {
            "mode": schedule["mode"],
            "target_weight": target_weight,
            "warmup_steps": warmup_steps,
            "ramp_steps": ramp_steps,
        }


    with log_path.open(log_mode, encoding="utf-8") as log_handle:
        for epoch_index in range(start_epoch, total_epochs):
            model.train()
            batch_components: dict[str, list[float]] = {
                key: [] for key in LOSS_COMPONENT_KEYS
            }
            gradient_norms: list[float] = []
            toml_weights: list[float] = []
            peak_weights: list[float] = []
            ccpd_weights: list[float] = []
            cpv_weights: list[float] = []
            nonfinite_loss_count = 0
            nonfinite_gradient_count = 0
            toml_count_totals = {
                key: 0
                for key in TOML_DIAGNOSTIC_KEYS
                if key.endswith("_count")
            }
            toml_coefficient_norm_weighted_sum = 0.0
            toml_diagnostic_sample_count = 0
            toml_coefficient_norm_max = 0.0
            epoch_started = time.time()
            train_progress = tqdm(
                train_loader,
                desc=f"Train epoch {epoch_index + 1}/{total_epochs}",
                dynamic_ncols=True,
                disable=not sys.stderr.isatty(),
            )
            for video, imu, target, _, _ in train_progress:
                video = video.to(device, non_blocking=True)
                imu = imu.to(device, non_blocking=True)
                target = target.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                prediction = model(video, imu)
                effective_toml_weight, toml_schedule = toml_weight_for_step(
                    global_step
                )
                effective_peak_weight, peak_schedule = peak_weight_for_step(
                    global_step
                )
                effective_ccpd_weight, ccpd_schedule = ccpd_weight_for_step(
                    global_step
                )
                effective_cpv_weight, cpv_schedule = cpv_weight_for_step(
                    global_step
                )
                try:
                    terms = criterion(
                        prediction,
                        target,
                        imu,
                        toml_weight_override=effective_toml_weight,
                        peak_weight_override=effective_peak_weight,
                        ccpd_weight_override=effective_ccpd_weight,
                        cpv_weight_override=effective_cpv_weight,
                    )
                except TOMLNumericalError as error:
                    diagnostics = _toml_diagnostics_to_python(
                        error.diagnostics
                    )
                    _append_numerical_event(
                        output_dir,
                        {
                            "event": "toml_numerical_failure",
                            "epoch": epoch_index + 1,
                            "global_step": global_step + 1,
                            "message": str(error),
                            "diagnostics": diagnostics,
                            "nonfinite_loss_count": 1,
                        },
                    )
                    raise
                diagnostics = _toml_diagnostics_to_python(
                    criterion.last_toml_diagnostics
                )
                if diagnostics:
                    for key in toml_count_totals:
                        toml_count_totals[key] += int(diagnostics[key])
                    diagnostic_batch_size = int(prediction.shape[0])
                    toml_coefficient_norm_weighted_sum += (
                        float(diagnostics["coefficient_norm_mean"])
                        * diagnostic_batch_size
                    )
                    toml_diagnostic_sample_count += diagnostic_batch_size
                    toml_coefficient_norm_max = max(
                        toml_coefficient_norm_max,
                        float(diagnostics["coefficient_norm_max"]),
                    )
                non_finite = [
                    key
                    for key in LOSS_COMPONENT_KEYS
                    if not bool(torch.isfinite(terms[key]))
                ]
                if non_finite:
                    nonfinite_loss_count += 1
                    _append_numerical_event(
                        output_dir,
                        {
                            "event": "nonfinite_loss",
                            "epoch": epoch_index + 1,
                            "global_step": global_step + 1,
                            "components": non_finite,
                            "nonfinite_loss_count": nonfinite_loss_count,
                        },
                    )
                    raise FloatingPointError(
                        f"Non-finite loss components {non_finite} at "
                        f"epoch={epoch_index + 1}, step={global_step + 1}"
                    )
                loss = terms["total_loss"]
                loss.backward()
                gradient_record = _gradient_audit(model)
                gradient_norm = float(gradient_record["gradient_norm"])
                if (
                    not bool(gradient_record["gradient_finite"])
                    or not np.isfinite(gradient_norm)
                ):
                    nonfinite_gradient_count += 1
                    _append_numerical_event(
                        output_dir,
                        {
                            "event": "nonfinite_gradient",
                            "epoch": epoch_index + 1,
                            "global_step": global_step + 1,
                            "gradient_norm": gradient_norm,
                            "nonfinite_gradient_count": (
                                nonfinite_gradient_count
                            ),
                        },
                    )
                    raise FloatingPointError(
                        "Non-finite gradient before optimizer step at "
                        f"epoch={epoch_index + 1}, step={global_step + 1}"
                    )
                gradient_norms.append(gradient_norm)
                toml_weights.append(effective_toml_weight)
                peak_weights.append(effective_peak_weight)
                ccpd_weights.append(effective_ccpd_weight)
                cpv_weights.append(effective_cpv_weight)
                optimizer.step()
                scheduler.step()
                global_step += 1
                batch_values = {
                    key: float(terms[key].detach())
                    for key in LOSS_COMPONENT_KEYS
                }
                for key, value in batch_values.items():
                    batch_components[key].append(value)
                progress_values = {
                    "total": f"{batch_values['total_loss']:.4f}",
                    "wave": f"{batch_values['waveform']:.4f}",
                    "lr": f"{optimizer.param_groups[0]['lr']:.2e}",
                }
                ccpd_config = criterion.resolved_config.get("ccpd")
                if ccpd_config is not None and ccpd_config["enabled"]:
                    progress_values["ccpd"] = f"{batch_values['ccpd']:.4f}"
                    progress_values["w_ccpd"] = (
                        f"{batch_values['weighted_ccpd']:.4f}"
                    )
                dpd_config = criterion.resolved_config.get("dpd")
                if dpd_config is not None and dpd_config["enabled"]:
                    progress_values["dpd"] = f"{batch_values['dpd']:.4f}"
                    progress_values["w_dpd"] = (
                        f"{batch_values['weighted_dpd']:.4f}"
                    )
                cpv_config = criterion.resolved_config.get("cpv")
                if cpv_config is not None and cpv_config["enabled"]:
                    progress_values["cpv"] = f"{batch_values['cpv']:.4f}"
                    progress_values["w_cpv"] = (
                        f"{batch_values['weighted_cpv']:.4f}"
                    )
                train_progress.set_postfix(progress_values)

            if validation_function is None:
                try:
                    validation_components = validation_loss_breakdown(
                        model,
                        valid_loader,
                        device,
                        criterion,
                    )
                except TOMLNumericalError as error:
                    _append_numerical_event(
                        output_dir,
                        {
                            "event": "toml_validation_numerical_failure",
                            "epoch": epoch_index + 1,
                            "global_step": global_step,
                            "message": str(error),
                            "diagnostics": _toml_diagnostics_to_python(
                                error.diagnostics
                            ),
                        },
                    )
                    raise
            else:
                validation_loss_value = float(
                    validation_function(model, valid_loader, device)
                )
                validation_components = {
                    key: 0.0 for key in LOSS_COMPONENT_KEYS
                }
                validation_components["total_loss"] = validation_loss_value
                validation_components["waveform"] = validation_loss_value
            validation_loss = float(validation_components["waveform"])
            if not np.isfinite(validation_loss):
                raise FloatingPointError(
                    f"Non-finite validation loss at epoch={epoch_index + 1}"
                )
            epoch = epoch_index + 1
            improved = validation_loss < best_validation_loss
            if improved:
                best_validation_loss = validation_loss
                best_epoch = epoch

            payload = _checkpoint_payload(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                global_step=global_step,
                best_epoch=best_epoch,
                best_validation_loss=best_validation_loss,
                validation_loss=validation_loss,
                validation_components=validation_components,
                criterion=criterion,
                config=config,
                train_loader=train_loader,
            )
            if improved:
                _atomic_torch_save(payload, output_dir / "best_checkpoint.pt")
            _atomic_torch_save(payload, output_dir / "last_checkpoint.pt")

            record = {
                "epoch": epoch,
                "epochs": total_epochs,
                "global_step": global_step,
                "train_total_loss": float(
                    np.mean(batch_components["total_loss"])
                ),
                "train_total_loss_max": float(
                    np.max(batch_components["total_loss"])
                ),
                "train_waveform_mse": float(
                    np.mean(batch_components["waveform"])
                ),
                "train_spectral_js": float(
                    np.mean(batch_components["spectral"])
                ),
                "train_toml": float(np.mean(batch_components["toml"])),
                "train_toml_max": float(np.max(batch_components["toml"])),

                "train_peak_consistency": float(
                    np.mean(batch_components["peak_consistency"])
                ),
                "train_peak_map": float(
                    np.mean(batch_components["peak_map"])
                ),
                "train_peak_count": float(
                    np.mean(batch_components["peak_count"])
                ),
                "train_peak_timing": float(
                    np.mean(batch_components["peak_timing"])
                ),
                "train_predicted_soft_peak_count": float(
                    np.mean(batch_components["predicted_soft_peak_count"])
                ),
                "train_target_soft_peak_count": float(
                    np.mean(batch_components["target_soft_peak_count"])
                ),
                "train_boundary_continuity": float(
                    np.mean(batch_components["boundary_continuity"])
                ),
                "train_weighted_peak_consistency": float(
                    np.mean(batch_components["weighted_peak_consistency"])
                ),
                "train_weighted_peak_consistency_max": float(
                    np.max(batch_components["weighted_peak_consistency"])
                ),
                "train_weighted_peak_fraction": float(
                    np.mean(batch_components["weighted_peak_consistency"])
                    / max(
                        np.mean(batch_components["total_loss"]),
                        np.finfo(np.float64).tiny,
                    )
                ),
                "train_ccpd": float(
                    np.mean(batch_components["ccpd"])
                ),
                "train_ccpd_max": float(
                    np.max(batch_components["ccpd"])
                ),
                "train_ccpd_forward": float(
                    np.mean(batch_components["ccpd_forward"])
                ),
                "train_ccpd_backward": float(
                    np.mean(batch_components["ccpd_backward"])
                ),
                "train_predicted_soft_event_count": float(
                    np.mean(batch_components["predicted_soft_event_count"])
                ),
                "train_target_soft_event_count": float(
                    np.mean(batch_components["target_soft_event_count"])
                ),
                "train_soft_event_count_error": float(
                    np.mean(batch_components["soft_event_count_error"])
                ),
                "train_weighted_ccpd": float(
                    np.mean(batch_components["weighted_ccpd"])
                ),
                "train_weighted_ccpd_max": float(
                    np.max(batch_components["weighted_ccpd"])
                ),
                "train_weighted_ccpd_fraction": float(
                    np.mean(batch_components["weighted_ccpd"])
                    / max(
                        np.mean(batch_components["total_loss"]),
                        np.finfo(np.float64).tiny,
                    )
                ),
                "train_dpd": float(np.mean(batch_components["dpd"])),
                "train_dpd_max": float(np.max(batch_components["dpd"])),
                "train_weighted_dpd": float(
                    np.mean(batch_components["weighted_dpd"])
                ),
                "train_weighted_dpd_max": float(
                    np.max(batch_components["weighted_dpd"])
                ),
                "train_weighted_dpd_fraction": float(
                    np.mean(batch_components["weighted_dpd"])
                    / max(
                        np.mean(batch_components["total_loss"]),
                        np.finfo(np.float64).tiny,
                    )
                ),
                "train_cpv": float(np.mean(batch_components["cpv"])),
                "train_cpv_max": float(np.max(batch_components["cpv"])),
                "train_weighted_cpv": float(
                    np.mean(batch_components["weighted_cpv"])
                ),
                "train_weighted_cpv_max": float(
                    np.max(batch_components["weighted_cpv"])
                ),
                "train_weighted_cpv_fraction": float(
                    np.mean(batch_components["weighted_cpv"])
                    / max(
                        np.mean(batch_components["total_loss"]),
                        np.finfo(np.float64).tiny,
                    )
                ),
                "train_weighted_spectral": float(
                    np.mean(batch_components["weighted_spectral"])
                ),
                "train_weighted_toml": float(
                    np.mean(batch_components["weighted_toml"])
                ),
                "train_weighted_toml_max": float(
                    np.max(batch_components["weighted_toml"])
                ),
                "train_weighted_toml_fraction": float(
                    np.mean(batch_components["weighted_toml"])
                    / max(
                        np.mean(batch_components["total_loss"]),
                        np.finfo(np.float64).tiny,
                    )
                ),
                "train_toml_weight_mean": float(np.mean(toml_weights)),
                "train_toml_weight_min": float(np.min(toml_weights)),
                "train_toml_weight_max": float(np.max(toml_weights)),
                "train_toml_weight_last": float(toml_weights[-1]),
                "toml_target_weight": float(
                    toml_schedule["target_weight"]
                ),
                "toml_schedule_mode": toml_schedule["mode"],
                "toml_warmup_steps": int(toml_schedule["warmup_steps"]),
                "toml_ramp_steps": int(toml_schedule["ramp_steps"]),

                "train_peak_weight_mean": float(np.mean(peak_weights)),
                "train_peak_weight_min": float(np.min(peak_weights)),
                "train_peak_weight_max": float(np.max(peak_weights)),
                "train_peak_weight_last": float(peak_weights[-1]),
                "peak_target_weight": float(peak_schedule["target_weight"]),
                "peak_schedule_mode": peak_schedule["mode"],
                "peak_warmup_steps": int(peak_schedule["warmup_steps"]),
                "peak_ramp_steps": int(peak_schedule["ramp_steps"]),

                "train_ccpd_weight_mean": float(np.mean(ccpd_weights)),
                "train_ccpd_weight_min": float(np.min(ccpd_weights)),
                "train_ccpd_weight_max": float(np.max(ccpd_weights)),
                "train_ccpd_weight_last": float(ccpd_weights[-1]),
                "ccpd_target_weight": float(ccpd_schedule["target_weight"]),
                "ccpd_schedule_mode": ccpd_schedule["mode"],
                "ccpd_warmup_steps": int(ccpd_schedule["warmup_steps"]),
                "ccpd_ramp_steps": int(ccpd_schedule["ramp_steps"]),
                "train_cpv_weight_mean": float(np.mean(cpv_weights)),
                "train_cpv_weight_min": float(np.min(cpv_weights)),
                "train_cpv_weight_max": float(np.max(cpv_weights)),
                "train_cpv_weight_last": float(cpv_weights[-1]),
                "cpv_target_weight": float(cpv_schedule["target_weight"]),
                "cpv_schedule_mode": cpv_schedule["mode"],
                "cpv_warmup_steps": int(cpv_schedule["warmup_steps"]),
                "cpv_ramp_steps": int(cpv_schedule["ramp_steps"]),
                # Validation keeps the complete official clip set, so train-
                # only grouped peak and CCPD terms are disabled. CPV uses its
                # configured full weight; selection remains based solely on
                # the original waveform MSE.
                "validation_toml_weight": float(
                    criterion.resolved_config["toml"]["weight"]
                ),
                "validation_peak_weight": 0.0,
                "validation_ccpd_weight": 0.0,
                "validation_cpv_weight": float(
                    criterion.resolved_config.get("cpv", {}).get("weight", 0.0)
                ),
                "train_grad_norm": float(np.mean(gradient_norms)),
                "train_grad_norm_max": float(np.max(gradient_norms)),
                "nonfinite_loss_count": nonfinite_loss_count,
                "nonfinite_gradient_count": nonfinite_gradient_count,
                **toml_count_totals,
                "toml_coefficient_norm_mean": (
                    toml_coefficient_norm_weighted_sum
                    / toml_diagnostic_sample_count
                    if toml_diagnostic_sample_count
                    else 0.0
                ),
                "toml_coefficient_norm_max": toml_coefficient_norm_max,
                "validation_total_loss": validation_components["total_loss"],
                "validation_main_waveform_mse": validation_loss,
                "validation_spectral_js": validation_components["spectral"],
                "validation_toml": validation_components["toml"],
                "validation_weighted_spectral": validation_components[
                    "weighted_spectral"
                ],
                "validation_weighted_toml": validation_components[
                    "weighted_toml"
                ],

                "validation_peak_consistency": validation_components[
                    "peak_consistency"
                ],
                "validation_peak_map": validation_components["peak_map"],
                "validation_peak_count": validation_components["peak_count"],
                "validation_peak_timing": validation_components[
                    "peak_timing"
                ],
                "validation_predicted_soft_peak_count": validation_components[
                    "predicted_soft_peak_count"
                ],
                "validation_target_soft_peak_count": validation_components[
                    "target_soft_peak_count"
                ],
                "validation_boundary_continuity": validation_components[
                    "boundary_continuity"
                ],
                "validation_weighted_peak_consistency": validation_components[
                    "weighted_peak_consistency"
                ],
                "validation_ccpd": validation_components["ccpd"],
                "validation_ccpd_forward": validation_components[
                    "ccpd_forward"
                ],
                "validation_ccpd_backward": validation_components[
                    "ccpd_backward"
                ],
                "validation_predicted_soft_event_count": validation_components[
                    "predicted_soft_event_count"
                ],
                "validation_target_soft_event_count": validation_components[
                    "target_soft_event_count"
                ],
                "validation_soft_event_count_error": validation_components[
                    "soft_event_count_error"
                ],
                "validation_weighted_ccpd": validation_components[
                    "weighted_ccpd"
                ],
                "validation_dpd": validation_components["dpd"],
                "validation_weighted_dpd": validation_components[
                    "weighted_dpd"
                ],
                "dpd_transform": (
                    criterion.resolved_config.get("dpd", {}).get("transform")
                ),
                "validation_cpv": validation_components["cpv"],
                "validation_weighted_cpv": validation_components[
                    "weighted_cpv"
                ],
                "selection_metric": "waveform",
                "selection_value": validation_loss,
                "best_epoch": best_epoch,
                "best_validation_main_waveform_mse": best_validation_loss,
                "best_selection_value": best_validation_loss,
                "best_updated": improved,
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                "epoch_elapsed_seconds": time.time() - epoch_started,
                "elapsed_seconds": time.time() - started,
            }
            log_handle.write(json.dumps(_json_ready(record), sort_keys=True) + "\n")
            log_handle.flush()
            print(
                f"epoch={epoch}/{total_epochs} "
                f"train_total={record['train_total_loss']:.6f} "
                f"train_wave={record['train_waveform_mse']:.6f} "
                f"valid_wave={validation_loss:.6f} "
                f"best_epoch={best_epoch}",
                flush=True,
            )

    if best_epoch < 1 or not (output_dir / "best_checkpoint.pt").is_file():
        raise RuntimeError("Training completed without a best checkpoint")
    return {
        "epochs_completed": total_epochs,
        "global_step": global_step,
        "best_epoch": best_epoch,
        "best_validation_main_waveform_mse": best_validation_loss,
        "selection_metric": "waveform",
        "loss": criterion.resolved_config,
        "loss_config_sha256": criterion.config_sha256,
        "elapsed_seconds_this_process": time.time() - started,
    }


def _strict_load_best(
    path: Path,
    model: torch.nn.Module,
    expected_criterion: MotionDisentangledMultiDomainLoss | None = None,
) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("protocol") != "paper_e2e_100epoch":
        raise RuntimeError(f"Not a paper-protocol best checkpoint: {path}")
    if expected_criterion is not None:
        _verify_checkpoint_loss_contract(payload, expected_criterion, path)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    return payload


def _model_initialization(
    config: dict[str, Any],
) -> tuple[torch.nn.Module, dict[str, Any]]:
    builder = getattr(base_module, "build_fresh_paper_model", None)
    if builder is None:
        raise ImportError(
            "eccvw2.base.build_fresh_paper_model is required for fresh paper training"
        )
    initialization_config = config.get("initialization", {})
    initialization_seed = int(
        initialization_config.get(
            "seed", config["experiment"]["seed"]
        )
    )
    imagenet_cache_path = initialization_config.get("imagenet_cache_path")
    model, audit = builder(
        model_config=config["model"],
        frames=int(config["data"]["chunk_length"]),
        initialization_seed=initialization_seed,
        imagenet_cache_path=imagenet_cache_path,
    )
    if hasattr(audit, "to_dict"):
        audit = audit.to_dict()
    if not isinstance(audit, dict):
        raise TypeError("Fresh-model initialization audit must be a dict-like payload")
    return model, audit


def run_fold(
    config_path: str | Path,
    fold: int,
    device_name: str,
    *,
    run_id: str | None = None,
    resume: bool = False,
    force: bool = False,
) -> Path:
    """Train one fold and test only its strictly loaded best epoch."""

    if resume and force:
        raise ValueError("--resume and --force are mutually exclusive")
    config_path = Path(config_path).resolve()
    config = load_yaml(config_path)
    os.environ.setdefault("BEAT_EGOPPG_ROOT", config["paths"]["source_root"])
    if run_id is None:
        run_id = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S_%f")
    _validate_paper_protocol(config)
    criterion = build_training_loss(
        config.get("loss"),
        expected_length=int(config["data"]["chunk_length"]),
        evaluation_fs=float(config["evaluation"]["fs"]),
    )
    output_root = _validate_output_root(config)
    run_root = _resolve_run_root(output_root, run_id)
    if run_id is not None:
        config["experiment"]["run_id"] = run_id
    fold_count = int(config["data"]["number_of_folds"])
    if fold < 0 or fold >= fold_count:
        raise ValueError(f"fold must be in [0, {fold_count}), got {fold}")

    output_dir = (
        run_root / config["experiment"]["output_name"] / f"fold_{fold}"
    ).resolve()
    if not output_dir.is_relative_to(run_root):
        raise ValueError(f"Resolved experiment output escapes run root: {output_dir}")
    if output_dir.exists() and force:
        shutil.rmtree(output_dir)
    if (output_dir / "metrics.json").is_file() and not force:
        print(f"Completed output already exists, skipping: {output_dir}")
        return output_dir
    last_checkpoint = output_dir / "last_checkpoint.pt"
    if resume and not last_checkpoint.is_file():
        raise FileNotFoundError(f"No resumable checkpoint found: {last_checkpoint}")
    if output_dir.exists() and any(output_dir.iterdir()) and not resume and not force:
        raise FileExistsError(
            f"Partial fold output exists; use --resume or --force: {output_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    fold_started = datetime.now().astimezone()
    seed = int(config["experiment"]["seed"])
    set_seed(seed)
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but CUDA is unavailable")

    if not resume:
        shutil.copy2(config_path, output_dir / "config_source.yaml")
        with (output_dir / "config_resolved.yaml").open(
            "w", encoding="utf-8"
        ) as handle:
            yaml.safe_dump(config, handle, sort_keys=False)
    launch_command = os.environ.get("ECCVW_LAUNCH_COMMAND")
    if launch_command and not resume:
        (output_dir / "command.txt").write_text(
            launch_command + "\n", encoding="utf-8"
        )
    run_metadata = {
        "run_id": run_id,
        "status": "running",
        "started_at": fold_started.isoformat(),
        "completed_at": None,
        "fold": fold,
        "seed": seed,
        "resumed": resume,
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
        seed=seed,
        number_of_folds=fold_count,
    )
    write_json(output_dir / "split_audit.json", split.summary())

    cross_clip_audit: dict[str, Any] | None = None
    cross_clip_config = config["training"].get(
        "cross_clip", {"enabled": False}
    )
    if bool(cross_clip_config.get("enabled", False)):
        train_loader, cross_clip_audit = build_cross_clip_loader(
            split.train_records,
            batch_size=int(config["training"]["batch_size"]),
            augment=True,
            seed=seed,
            num_workers=int(config["training"]["num_workers"]),
            cross_clip_config=cross_clip_config,
            sampling_rate_hz=float(config["evaluation"]["fs"]),
        )
        write_json(
            output_dir / "cross_clip_sampling_audit.json",
            cross_clip_audit,
        )
    else:
        train_loader = build_loader(
            split.train_records,
            batch_size=int(config["training"]["batch_size"]),
            augment=True,
            shuffle=True,
            seed=seed,
            num_workers=int(config["training"]["num_workers"]),
        )
    valid_loader = build_loader(
        split.valid_records,
        batch_size=int(config["validation"]["batch_size"]),
        augment=False,
        shuffle=False,
        seed=seed,
        num_workers=int(config["validation"]["num_workers"]),
    )
    test_loader = build_loader(
        split.test_records,
        batch_size=int(config["evaluation"]["batch_size"]),
        augment=False,
        shuffle=False,
        seed=seed,
        num_workers=int(config["evaluation"]["num_workers"]),
    )

    model, initialization_audit = _model_initialization(config)
    model = model.to(device)
    optimizer, scheduler = _build_optimizer_and_scheduler(
        model,
        config["training"],
        steps_per_epoch=len(train_loader),
    )
    start_epoch = 0
    global_step = 0
    best_epoch = -1
    best_validation_loss = float("inf")
    if resume:
        (
            start_epoch,
            global_step,
            best_epoch,
            best_validation_loss,
        ) = _load_resume_checkpoint(
            last_checkpoint,
            model,
            optimizer,
            scheduler,
            train_loader,
            expected_epochs=int(config["training"]["epochs"]),
            expected_steps_per_epoch=len(train_loader),
            expected_criterion=criterion,
            expected_cross_clip_config=config["training"].get(
                "cross_clip", {"enabled": False}
            ),
        )

    model_audit = {
        "model_class": type(model).__name__,
        "initialization": initialization_audit,
        "parameters": parameter_audit(model),
        "all_parameters_trainable": all(
            parameter.requires_grad for parameter in model.parameters()
        ),
        "trainable_parameters": sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        ),
        "architecture": probe_architecture(model, valid_loader, device),
        "training_loss": criterion.resolved_config,
        "training_loss_config_sha256": criterion.config_sha256,
        "validation_selection_metric": "waveform",
    }
    if not model_audit["all_parameters_trainable"]:
        raise RuntimeError("Fresh model is not configured for full end-to-end training")
    write_json(output_dir / "model_audit.json", model_audit)

    train_summary = _train_epochs(
        model=model,
        train_loader=train_loader,
        valid_loader=valid_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        total_epochs=int(config["training"]["epochs"]),
        output_dir=output_dir,
        config=config,
        start_epoch=start_epoch,
        global_step=global_step,
        best_epoch=best_epoch,
        best_validation_loss=best_validation_loss,
        criterion=criterion,
    )

    best_payload = _strict_load_best(
        output_dir / "best_checkpoint.pt",
        model,
        expected_criterion=criterion,
    )
    selected_epoch = int(best_payload["epoch"])
    if selected_epoch != int(best_payload["best_epoch"]):
        raise RuntimeError("Best-checkpoint epoch metadata is inconsistent")
    # A plain state dictionary supports safe weights_only=True evaluation.
    # Optimizer and RNG state remain in the separate local training checkpoint.
    _atomic_torch_save(model.state_dict(), output_dir / "best_model_weights.pt")
    waveform_store, diagnostics = collect_model_outputs(model, test_loader, device)
    metrics = evaluate_waveform_store(
        waveform_store,
        source_root=config["paths"]["source_root"],
        fs=int(config["evaluation"]["fs"]),
        window_seconds=int(config["evaluation"]["window_seconds"]),
        hr_method=config["evaluation"]["hr_method"],
    )

    fold_completed = datetime.now().astimezone()
    completed_run = {
        **run_metadata,
        "status": "completed",
        "completed_at": fold_completed.isoformat(),
        "elapsed_seconds_this_process": (
            fold_completed - fold_started
        ).total_seconds(),
    }
    summary = {
        "run": completed_run,
        "experiment": config["experiment"],
        "fold": fold,
        "selected_best_epoch": selected_epoch,
        "loss": criterion.resolved_config,
        "loss_config_sha256": criterion.config_sha256,
        "validation_selection_metric": "waveform",
        "validation_main_waveform_mse": float(
            best_payload["best_validation_loss"]
        ),
        "training": train_summary,
        "metrics": metrics,
        "diagnostics": diagnostics,
        "model_audit": model_audit,
        "split": split.summary(),
        "cross_clip_sampling": cross_clip_audit,
    }
    write_json(output_dir / "metrics.json", summary)
    write_json(output_dir / "run_metadata.json", completed_run)
    print(json.dumps(_json_ready(summary), indent=2, sort_keys=True), flush=True)
    return output_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--fold", required=True, type=int)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--run-id")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_fold(
        args.config,
        args.fold,
        args.device,
        run_id=args.run_id,
        resume=args.resume,
        force=args.force,
    )


if __name__ == "__main__":
    main()
