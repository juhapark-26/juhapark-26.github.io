"""Official egoPPG waveform and heart-rate evaluation."""

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.stats import pearsonr
from tqdm import tqdm

from .losses import (
    LOSS_COMPONENT_KEYS,
    MotionDisentangledMultiDomainLoss,
    baseline_z_normalize,
)


WaveformStore = dict[str, dict[str, dict[int, np.ndarray]]]


def _official_metric_function(source_root: str | Path):
    source_root = str(Path(source_root).resolve())
    if source_root not in sys.path:
        sys.path.insert(0, source_root)
    from evaluation.post_process import calculate_metric_per_video_ppg

    return calculate_metric_per_video_ppg


def _new_store() -> WaveformStore:
    return defaultdict(lambda: defaultdict(dict))


def _safe_mean(values: Any) -> float:
    array = np.asarray(values, dtype=np.float64)
    return float(np.nanmean(array)) if array.size else float("nan")


def collect_model_outputs(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> tuple[WaveformStore, dict[str, float | bool]]:
    store = _new_store()
    model.eval()
    all_finite = True
    output_count = 0
    with torch.inference_mode():
        test_progress = tqdm(
            loader,
            desc="Test",
            dynamic_ncols=True,
            disable=not sys.stderr.isatty(),
        )
        for video, imu, target, participants, chunk_indices in test_progress:
            video = video.to(device, non_blocking=True)
            imu = imu.to(device, non_blocking=True)
            prediction = model(video, imu)
            target = target.to(device, non_blocking=True)
            all_finite = all_finite and bool(torch.isfinite(prediction).all())
            output_count += prediction.shape[0]
            test_progress.set_postfix(clips=output_count)

            waveforms = {"final": prediction, "target": target}
            for sample_index, (participant, chunk_index) in enumerate(
                zip(participants, chunk_indices.tolist())
            ):
                for signal_name, waveform in waveforms.items():
                    store[signal_name][participant][int(chunk_index)] = (
                        waveform[sample_index].detach().cpu().float().numpy()
                    )
    return store, {"forward_finite": all_finite, "evaluated_clips": output_count}


def _concatenate_participant(chunks: dict[int, np.ndarray]) -> np.ndarray:
    return np.concatenate([chunks[index] for index in sorted(chunks)])


def evaluate_waveform_store(
    store: WaveformStore,
    source_root: str | Path,
    fs: int = 30,
    window_seconds: int = 60,
    hr_method: str = "Peak_Detection",
) -> dict[str, dict[str, float]]:
    official_metric = _official_metric_function(source_root)
    target_by_participant = store["target"]
    all_metrics: dict[str, dict[str, float]] = {}
    for signal_name, signal_by_participant in store.items():
        if signal_name == "target":
            continue
        gt_hr_all: list[float] = []
        pred_hr_all: list[float] = []
        participant_pearsons: list[float] = []
        waveform_predictions: list[np.ndarray] = []
        waveform_targets: list[np.ndarray] = []
        window_size = fs * window_seconds

        for participant in sorted(signal_by_participant):
            prediction = _concatenate_participant(signal_by_participant[participant])
            target = _concatenate_participant(target_by_participant[participant])
            participant_gt: list[float] = []
            participant_pred: list[float] = []
            for start in range(0, len(prediction), window_size):
                pred_window = prediction[start : start + window_size]
                target_window = target[start : start + window_size]
                if len(pred_window) < window_size:
                    continue
                gt_hr, pred_hr, _, _, _ = official_metric(
                    pred_window,
                    target_window,
                    hr_method,
                    True,
                    fs,
                )
                gt_hr_all.append(float(gt_hr))
                pred_hr_all.append(float(pred_hr))
                participant_gt.append(float(gt_hr))
                participant_pred.append(float(pred_hr))

            participant_gt_array = np.asarray(participant_gt)
            participant_pred_array = np.asarray(participant_pred)
            valid_hr = np.isfinite(participant_gt_array) & np.isfinite(participant_pred_array)
            if valid_hr.sum() > 1:
                participant_pearsons.append(
                    float(
                        pearsonr(
                            participant_gt_array[valid_hr],
                            participant_pred_array[valid_hr],
                        ).statistic
                    )
                )

            pred_z = (prediction - prediction.mean()) / max(prediction.std(), 1.0e-8)
            target_z = (target - target.mean()) / max(target.std(), 1.0e-8)
            waveform_predictions.append(pred_z)
            waveform_targets.append(target_z)

        gt_hr = np.asarray(gt_hr_all)
        pred_hr = np.asarray(pred_hr_all)
        valid = np.isfinite(gt_hr) & np.isfinite(pred_hr)
        gt_hr = gt_hr[valid]
        pred_hr = pred_hr[valid]
        waveform_prediction = np.concatenate(waveform_predictions)
        waveform_target = np.concatenate(waveform_targets)
        all_metrics[signal_name] = {
            "hr_mae_bpm": float(np.mean(np.abs(pred_hr - gt_hr))),
            "hr_rmse_bpm": float(np.sqrt(np.mean((pred_hr - gt_hr) ** 2))),
            "hr_mape_percent": float(np.mean(np.abs((pred_hr - gt_hr) / gt_hr)) * 100),
            "hr_pearson": _safe_mean(participant_pearsons),
            "waveform_mae_z": float(np.mean(np.abs(waveform_prediction - waveform_target))),
            "waveform_rmse_z": float(
                np.sqrt(np.mean((waveform_prediction - waveform_target) ** 2))
            ),
            "waveform_pearson": float(
                pearsonr(waveform_target, waveform_prediction).statistic
            ),
            "number_of_hr_windows": int(len(gt_hr)),
            "number_of_participants": int(len(signal_by_participant)),
        }
    return all_metrics


def validation_main_loss(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> float:
    losses: list[float] = []
    model.eval()
    with torch.inference_mode():
        validation_progress = tqdm(
            loader,
            desc="Validation",
            dynamic_ncols=True,
            disable=not sys.stderr.isatty(),
        )
        for video, imu, target, _, _ in validation_progress:
            prediction = model(
                video.to(device, non_blocking=True),
                imu.to(device, non_blocking=True),
            )
            target = target.to(device, non_blocking=True)
            loss = torch.nn.functional.mse_loss(
                baseline_z_normalize(prediction),
                baseline_z_normalize(target),
            )
            batch_loss = float(loss)
            losses.append(batch_loss)
            validation_progress.set_postfix(loss=f"{batch_loss:.4f}")
    return float(np.mean(losses))


def validation_loss_breakdown(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    criterion: MotionDisentangledMultiDomainLoss,
) -> dict[str, float]:
    """Return validation components while preserving equal batch averaging."""

    component_values = {key: [] for key in LOSS_COMPONENT_KEYS}
    model.eval()
    with torch.inference_mode():
        validation_progress = tqdm(
            loader,
            desc="Validation",
            dynamic_ncols=True,
            disable=not sys.stderr.isatty(),
        )
        for video, imu, target, _, _ in validation_progress:
            video = video.to(device, non_blocking=True)
            imu = imu.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            prediction = model(video, imu)
            # Validation batches retain the complete official clip set and
            # are not guaranteed to be consecutive groups. Keep checkpoint
            # selection exactly on the original waveform metric and disable
            # the train-only cross-clip term here.
            terms = criterion(
                prediction,
                target,
                imu,
                peak_weight_override=0.0,
                ccpd_weight_override=0.0,
            )
            for key in LOSS_COMPONENT_KEYS:
                value = float(terms[key])
                if not np.isfinite(value):
                    raise FloatingPointError(
                        f"Non-finite validation loss component {key}"
                    )
                component_values[key].append(value)
            validation_progress.set_postfix(
                total=f"{component_values['total_loss'][-1]:.4f}",
                wave=f"{component_values['waveform'][-1]:.4f}",
            )
    if not component_values["total_loss"]:
        raise ValueError("Validation loader must contain at least one batch")
    return {
        key: float(np.mean(values))
        for key, values in component_values.items()
    }
