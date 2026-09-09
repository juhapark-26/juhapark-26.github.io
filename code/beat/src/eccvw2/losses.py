"""Waveform and motion-disentangled multi-domain training losses."""

from __future__ import annotations

import hashlib
import json
import math
from contextlib import nullcontext
from copy import deepcopy
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from scipy.signal import butter, filtfilt
from torch import nn


LOSS_IMPLEMENTATION_VERSION = "mdm_v1"
LATEST_LOSS_IMPLEMENTATION_VERSION = "mdm_v2"
PEAK_LOSS_IMPLEMENTATION_VERSION = "peak_v1"
CCPD_LOSS_IMPLEMENTATION_VERSION = "ccpd_v1"
DPD_LOSS_IMPLEMENTATION_VERSION = "dpd_v1"
CPV_LOSS_IMPLEMENTATION_VERSION = "cpv_v1"
SUPPORTED_LOSS_IMPLEMENTATION_VERSIONS = frozenset(
    {
        LOSS_IMPLEMENTATION_VERSION,
        LATEST_LOSS_IMPLEMENTATION_VERSION,
        PEAK_LOSS_IMPLEMENTATION_VERSION,
        CCPD_LOSS_IMPLEMENTATION_VERSION,
        DPD_LOSS_IMPLEMENTATION_VERSION,
        CPV_LOSS_IMPLEMENTATION_VERSION,
    }
)
TOML_DIAGNOSTIC_KEYS = (
    "nonfinite_input_count",
    "nonfinite_system_count",
    "solve_failure_count",
    "nonfinite_coefficient_count",
    "nonfinite_output_count",
    "coefficient_norm_mean",
    "coefficient_norm_max",
)


class TOMLNumericalError(FloatingPointError):
    """TOML numerical failure carrying detached diagnostic scalars."""

    def __init__(
        self,
        message: str,
        diagnostics: dict[str, torch.Tensor],
    ) -> None:
        super().__init__(message)
        self.diagnostics = {
            key: value.detach() for key, value in diagnostics.items()
        }


LOSS_COMPONENT_KEYS = (
    "total_loss",
    "waveform",
    "spectral",
    "toml",
    "weighted_spectral",
    "weighted_toml",
    "peak_consistency",
    "peak_map",
    "peak_count",
    "peak_timing",
    "predicted_soft_peak_count",
    "target_soft_peak_count",
    "boundary_continuity",
    "weighted_peak_consistency",
    "ccpd",
    "ccpd_forward",
    "ccpd_backward",
    "predicted_soft_event_count",
    "target_soft_event_count",
    "soft_event_count_error",
    "weighted_ccpd",
    "dpd",
    "weighted_dpd",
    "cpv",
    "weighted_cpv",
)


def baseline_z_normalize(x: torch.Tensor, epsilon: float = 1.0e-8) -> torch.Tensor:
    """Match the original trainer's global batch z-normalization convention."""

    return (x - torch.mean(x)) / torch.std(x).clamp_min(epsilon)


def _validate_waveforms(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> tuple[int, int]:
    if prediction.ndim != 2 or target.ndim != 2:
        raise ValueError(
            "prediction and target must both have shape [batch, time], got "
            f"{tuple(prediction.shape)} and {tuple(target.shape)}"
        )
    if prediction.shape != target.shape:
        raise ValueError(
            "prediction and target shapes differ: "
            f"{tuple(prediction.shape)} versus {tuple(target.shape)}"
        )
    if prediction.device != target.device:
        raise ValueError("prediction and target must be on the same device")
    if not prediction.is_floating_point() or not target.is_floating_point():
        raise TypeError("prediction and target must be floating-point tensors")
    if prediction.shape[0] < 1 or prediction.shape[1] < 2:
        raise ValueError("waveforms require batch >= 1 and time >= 2")
    return int(prediction.shape[0]), int(prediction.shape[1])


def batch_global_waveform_mse(
    prediction: torch.Tensor,
    target: torch.Tensor,
    epsilon: float = 1.0e-8,
) -> torch.Tensor:
    """Compute the original batch-global B by T z-normalized MSE."""

    _validate_waveforms(prediction, target)
    if not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("waveform epsilon must be finite and positive")
    return F.mse_loss(
        baseline_z_normalize(prediction, epsilon),
        baseline_z_normalize(target, epsilon),
    )


def samplewise_z_normalize(
    waveform: torch.Tensor,
    epsilon: float = 1.0e-8,
) -> torch.Tensor:
    """Population-standardize each waveform independently over time."""

    if waveform.ndim != 2 or not waveform.is_floating_point():
        raise ValueError("waveform must be a floating tensor with shape [batch, time]")
    if waveform.shape[0] < 1 or waveform.shape[1] < 2:
        raise ValueError("waveforms require batch >= 1 and time >= 2")
    if not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("normalization epsilon must be finite and positive")
    centered = waveform - waveform.mean(dim=-1, keepdim=True)
    scale = centered.square().mean(dim=-1, keepdim=True).sqrt().clamp_min(epsilon)
    return centered / scale


def zero_padded_lag_dictionary(
    motion: torch.Tensor,
    max_lag_samples: int,
) -> torch.Tensor:
    """Build D[t, lag + K] = motion[t - lag] without circular wraparound."""

    if motion.ndim != 2 or not motion.is_floating_point():
        raise ValueError("motion must be a floating tensor with shape [batch, time]")
    if motion.shape[0] < 1 or motion.shape[1] < 2:
        raise ValueError("motion requires batch >= 1 and time >= 2")
    if isinstance(max_lag_samples, bool) or int(max_lag_samples) != max_lag_samples:
        raise ValueError("max_lag_samples must be an integer")
    max_lag_samples = int(max_lag_samples)
    if max_lag_samples < 0:
        raise ValueError("max_lag_samples must be non-negative")
    if 2 * max_lag_samples + 1 > motion.shape[1]:
        raise ValueError("the lag dictionary width 2K+1 must not exceed time")

    time = int(motion.shape[1])
    columns: list[torch.Tensor] = []
    for lag in range(-max_lag_samples, max_lag_samples + 1):
        if lag < 0:
            offset = -lag
            column = F.pad(motion[:, offset:], (0, offset))
        elif lag > 0:
            column = F.pad(motion[:, : time - lag], (lag, 0))
        else:
            column = motion
        columns.append(column)
    return torch.stack(columns, dim=-1)


def _autocast_disabled(tensor: torch.Tensor):
    if tensor.device.type in {"cpu", "cuda"}:
        return torch.autocast(device_type=tensor.device.type, enabled=False)
    return nullcontext()


def _stable_work_dtype(tensor: torch.Tensor) -> torch.dtype:
    if tensor.dtype in {torch.float16, torch.bfloat16}:
        return torch.float32
    return tensor.dtype


def samplewise_z_normalize_stable(
    waveform: torch.Tensor,
    variance_epsilon: float = 1.0e-8,
) -> torch.Tensor:
    """Smooth per-sample standardization for numerically stable TOML."""

    if waveform.ndim != 2 or not waveform.is_floating_point():
        raise ValueError(
            "waveform must be a floating tensor with shape [batch, time]"
        )
    if waveform.shape[0] < 1 or waveform.shape[1] < 2:
        raise ValueError("waveforms require batch >= 1 and time >= 2")
    if not math.isfinite(variance_epsilon) or variance_epsilon <= 0:
        raise ValueError("variance_epsilon must be finite and positive")
    work_dtype = _stable_work_dtype(waveform)
    with _autocast_disabled(waveform):
        work = waveform.to(dtype=work_dtype)
        centered = work - work.mean(dim=-1, keepdim=True)
        variance = centered.square().mean(dim=-1, keepdim=True)
        return centered * torch.rsqrt(variance + float(variance_epsilon))


def _validate_spectral_parameters(
    *,
    time: int,
    sampling_rate_hz: float,
    band_hz: tuple[float, float] | list[float],
    n_fft: int,
    temperature: float,
    power_epsilon: float,
) -> tuple[float, float]:
    if not math.isfinite(sampling_rate_hz) or sampling_rate_hz <= 0:
        raise ValueError("sampling_rate_hz must be finite and positive")
    if len(band_hz) != 2:
        raise ValueError("band_hz must contain exactly [low, high]")
    low, high = float(band_hz[0]), float(band_hz[1])
    if not all(math.isfinite(value) for value in (low, high)):
        raise ValueError("band_hz values must be finite")
    if low <= 0 or high <= low or high > sampling_rate_hz / 2:
        raise ValueError(
            "band_hz must satisfy 0 < low < high <= Nyquist frequency"
        )
    if isinstance(n_fft, bool) or int(n_fft) != n_fft:
        raise ValueError("n_fft must be an integer")
    n_fft = int(n_fft)
    if n_fft < time:
        raise ValueError("n_fft must be at least the waveform length")
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("spectral temperature must be finite and positive")
    if not math.isfinite(power_epsilon) or power_epsilon <= 0:
        raise ValueError("spectral power_epsilon must be finite and positive")
    selected_bins = sum(
        low <= index * sampling_rate_hz / n_fft <= high
        for index in range(n_fft // 2 + 1)
    )
    if selected_bins < 2:
        raise ValueError("physiological band must contain at least two FFT bins")
    return low, high


def physiological_band_spectral_js(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    sampling_rate_hz: float,
    band_hz: tuple[float, float] | list[float],
    n_fft: int,
    hann_periodic: bool = True,
    temperature: float = 1.0,
    power_epsilon: float = 1.0e-8,
) -> torch.Tensor:
    """Align per-sample physiological-band log-power distributions with JSD."""

    _, time = _validate_waveforms(prediction, target)
    low, high = _validate_spectral_parameters(
        time=time,
        sampling_rate_hz=float(sampling_rate_hz),
        band_hz=band_hz,
        n_fft=n_fft,
        temperature=float(temperature),
        power_epsilon=float(power_epsilon),
    )
    n_fft = int(n_fft)
    work_dtype = _stable_work_dtype(prediction)
    with _autocast_disabled(prediction):
        prediction_work = prediction.to(dtype=work_dtype)
        target_work = target.to(dtype=work_dtype)
        window = torch.hann_window(
            time,
            periodic=bool(hann_periodic),
            dtype=work_dtype,
            device=prediction.device,
        )
        prediction_fft = torch.fft.rfft(
            (prediction_work - prediction_work.mean(dim=-1, keepdim=True)) * window,
            n=n_fft,
            dim=-1,
        )
        target_fft = torch.fft.rfft(
            (target_work - target_work.mean(dim=-1, keepdim=True)) * window,
            n=n_fft,
            dim=-1,
        )
        frequencies = torch.fft.rfftfreq(
            n_fft,
            d=1.0 / float(sampling_rate_hz),
            device=prediction.device,
        )
        band_mask = (frequencies >= low) & (frequencies <= high)
        prediction_log_power = torch.log(
            prediction_fft.abs().square()[..., band_mask] + float(power_epsilon)
        )
        target_log_power = torch.log(
            target_fft.abs().square()[..., band_mask] + float(power_epsilon)
        )
        prediction_log_probability = torch.log_softmax(
            prediction_log_power / float(temperature),
            dim=-1,
        )
        target_log_probability = torch.log_softmax(
            target_log_power / float(temperature),
            dim=-1,
        )
        prediction_probability = prediction_log_probability.exp()
        target_probability = target_log_probability.exp()
        mixture_probability = 0.5 * (
            prediction_probability + target_probability
        )
        mixture_log_probability = torch.log(
            mixture_probability.clamp_min(float(power_epsilon))
        )
        js = 0.5 * (
            prediction_probability
            * (prediction_log_probability - mixture_log_probability)
        ).sum(dim=-1)
        js = js + 0.5 * (
            target_probability
            * (target_log_probability - mixture_log_probability)
        ).sum(dim=-1)
        return js.mean().clamp_min(0.0)



def official_filtfilt_matrix(
    sequence_length: int,
    *,
    sampling_rate_hz: float,
    band_hz: tuple[float, float] | list[float],
    filter_order: int,
) -> torch.Tensor:
    """Build the exact fixed linear map used before official peak detection."""

    sequence_length = int(sequence_length)
    filter_order = int(filter_order)
    low, high = float(band_hz[0]), float(band_hz[1])
    if sequence_length < 2 or filter_order < 1:
        raise ValueError("Filter length and order must be positive")
    if not (0.0 < low < high <= float(sampling_rate_hz) / 2.0):
        raise ValueError("Peak-loss band must lie inside Nyquist")
    coefficients_b, coefficients_a = butter(
        filter_order,
        [low / float(sampling_rate_hz) * 2.0, high / float(sampling_rate_hz) * 2.0],
        btype="bandpass",
    )
    identity = np.eye(sequence_length, dtype=np.float64)
    matrix = filtfilt(
        coefficients_b,
        coefficients_a,
        identity,
        axis=-1,
    )
    if not np.isfinite(matrix).all():
        raise FloatingPointError("Official filtfilt matrix contains non-finite values")
    return torch.from_numpy(np.ascontiguousarray(matrix))


def _linear_detrend_batch(waveform: torch.Tensor) -> torch.Tensor:
    """Differentiable affine detrending over each long sequence."""

    centered = waveform - waveform.mean(dim=-1, keepdim=True)
    time = torch.linspace(
        -1.0,
        1.0,
        waveform.shape[-1],
        dtype=waveform.dtype,
        device=waveform.device,
    )
    denominator = time.square().sum().clamp_min(torch.finfo(waveform.dtype).eps)
    slope = (centered * time).sum(dim=-1, keepdim=True) / denominator
    return centered - slope * time


def derivative_pulse_dual_domain_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    filter_matrix: torch.Tensor,
    edge_crop_samples: int,
    transform: str,
    epsilon: float = 1.0e-8,
) -> torch.Tensor:
    """Compare fixed physiological-band representations of derivative PPG.

    ``reconstructed_pulse`` applies the inverse-difference path used to
    recover a pulse representation: cumulative sum followed by affine
    detrending. ``bandpass_derivative_control`` omits only the cumulative sum
    and is the matched control for testing whether pulse reconstruction, rather
    than generic band-limited waveform supervision, explains an improvement.
    Both paths use the same fixed zero-phase filter, crop, normalization, and
    weight in the composite objective.
    """

    _, time = _validate_waveforms(prediction, target)
    transform = str(transform).lower()
    if transform not in {
        "reconstructed_pulse",
        "bandpass_derivative_control",
    }:
        raise ValueError(f"Unsupported DPD transform: {transform!r}")
    if not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("DPD epsilon must be finite and positive")
    crop = int(edge_crop_samples)
    if crop < 0 or 2 * crop + 2 > time:
        raise ValueError("DPD edge crop leaves fewer than two samples")

    work_dtype = _stable_work_dtype(prediction)
    with _autocast_disabled(prediction):
        prediction_work = prediction.to(dtype=work_dtype)
        target_work = target.to(dtype=work_dtype)
        if transform == "reconstructed_pulse":
            prediction_state = -_linear_detrend_batch(
                torch.cumsum(prediction_work, dim=-1)
            )
            target_state = -_linear_detrend_batch(
                torch.cumsum(target_work, dim=-1)
            )
        else:
            prediction_state = -_linear_detrend_batch(prediction_work)
            target_state = -_linear_detrend_batch(target_work)

        matrix = filter_matrix.to(
            device=prediction.device,
            dtype=work_dtype,
        )
        expected_shape = (time, time)
        if tuple(matrix.shape) != expected_shape:
            raise ValueError(
                "DPD filter matrix shape does not match the clip length: "
                f"expected={expected_shape}, actual={tuple(matrix.shape)}"
            )
        prediction_state = prediction_state @ matrix
        target_state = target_state @ matrix
        if crop:
            prediction_state = prediction_state[:, crop:-crop]
            target_state = target_state[:, crop:-crop]
        return batch_global_waveform_mse(
            prediction_state,
            target_state,
            epsilon=float(epsilon),
        )


def _fft_analytic_signal(waveform: torch.Tensor) -> torch.Tensor:
    """Return the analytic signal using the standard FFT Hilbert multiplier."""

    if waveform.ndim != 2 or not waveform.is_floating_point():
        raise ValueError(
            "waveform must be a floating tensor with shape [batch, time]"
        )
    time = int(waveform.shape[-1])
    if time < 2:
        raise ValueError("analytic waveforms require at least two samples")

    spectrum = torch.fft.fft(waveform, dim=-1)
    multiplier = torch.zeros(
        time,
        dtype=waveform.dtype,
        device=waveform.device,
    )
    multiplier[0] = 1.0
    if time % 2 == 0:
        multiplier[1 : time // 2] = 2.0
        multiplier[time // 2] = 1.0
    else:
        multiplier[1 : (time + 1) // 2] = 2.0
    return torch.fft.ifft(spectrum * multiplier, dim=-1)


def cardiac_phase_velocity_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    filter_matrix: torch.Tensor,
    edge_crop_samples: int,
    lags: tuple[int, ...] | list[int],
    transform: str = "reconstructed_pulse",
    analytic_transform: str = "fft_hilbert",
    analytic_epsilon: float = 1.0e-2,
    increment_epsilon: float = 1.0e-2,
    target_scale_epsilon: float = 1.0e-8,
    envelope_weighting: str = "target_geometric_mean",
    envelope_weight_cap: float = 2.0,
) -> torch.Tensor:
    """Match local cardiac phase increments in reconstructed pulse space.

    The model predicts a standardized first derivative of PPG. This loss first
    follows the official inverse-difference evaluation path, then constructs an
    analytic pulse signal. Complex phase increments reduce sensitivity to
    global phase, amplitude, and polarity while retaining local cardiac-cycle
    velocity; finite-window and stabilizer effects make this approximate.
    Target-envelope confidence suppresses phase supervision where reference
    phase is poorly defined.

    Prediction normalization deliberately uses the detached target RMS. A
    sample-wise prediction standardization would amplify near-constant outputs
    and can create very large gradients before a pulse waveform has emerged.
    """

    _, time = _validate_waveforms(prediction, target)
    if str(transform).lower() != "reconstructed_pulse":
        raise ValueError("CPV transform must be reconstructed_pulse")
    if str(analytic_transform).lower() != "fft_hilbert":
        raise ValueError("CPV analytic_transform must be fft_hilbert")
    if str(envelope_weighting).lower() != "target_geometric_mean":
        raise ValueError(
            "CPV envelope_weighting must be target_geometric_mean"
        )

    epsilon_values = {
        "analytic_epsilon": analytic_epsilon,
        "increment_epsilon": increment_epsilon,
        "target_scale_epsilon": target_scale_epsilon,
        "envelope_weight_cap": envelope_weight_cap,
    }
    for name, value in epsilon_values.items():
        if not math.isfinite(float(value)) or float(value) <= 0:
            raise ValueError(f"CPV {name} must be finite and positive")

    if not isinstance(lags, (tuple, list)) or not lags:
        raise ValueError("CPV lags must be a non-empty list or tuple")
    resolved_lags: list[int] = []
    for lag in lags:
        if isinstance(lag, bool) or int(lag) != lag or int(lag) < 1:
            raise ValueError("CPV lags must contain positive integers")
        resolved_lags.append(int(lag))
    if resolved_lags != sorted(set(resolved_lags)):
        raise ValueError("CPV lags must be strictly increasing and unique")

    crop = int(edge_crop_samples)
    if (
        isinstance(edge_crop_samples, bool)
        or crop != edge_crop_samples
        or crop < 0
    ):
        raise ValueError("CPV edge_crop_samples must be a non-negative integer")
    remaining_time = time - 2 * crop
    if remaining_time <= max(resolved_lags):
        raise ValueError("CPV edge crop leaves too few samples for its lags")

    work_dtype = _stable_work_dtype(prediction)
    with _autocast_disabled(prediction):
        prediction_work = prediction.to(dtype=work_dtype)
        target_work = target.to(dtype=work_dtype)
        prediction_pulse = -_linear_detrend_batch(
            torch.cumsum(prediction_work, dim=-1)
        )
        target_pulse = -_linear_detrend_batch(
            torch.cumsum(target_work, dim=-1)
        )

        matrix = filter_matrix.to(
            device=prediction.device,
            dtype=work_dtype,
        )
        expected_shape = (time, time)
        if tuple(matrix.shape) != expected_shape:
            raise ValueError(
                "CPV filter matrix shape does not match the clip length: "
                f"expected={expected_shape}, actual={tuple(matrix.shape)}"
            )
        prediction_pulse = prediction_pulse @ matrix
        target_pulse = target_pulse @ matrix

        prediction_analytic = _fft_analytic_signal(prediction_pulse)
        target_analytic = _fft_analytic_signal(target_pulse)
        if crop:
            prediction_analytic = prediction_analytic[:, crop:-crop]
            target_analytic = target_analytic[:, crop:-crop]

        target_rms = torch.sqrt(
            target_analytic.abs().square().mean(dim=-1, keepdim=True)
            + float(target_scale_epsilon)
        ).detach()
        prediction_analytic = prediction_analytic / target_rms
        target_analytic = target_analytic / target_rms

        prediction_phasor = prediction_analytic / torch.sqrt(
            prediction_analytic.abs().square() + float(analytic_epsilon)
        )
        target_phasor = target_analytic / torch.sqrt(
            target_analytic.abs().square() + float(analytic_epsilon)
        )
        target_envelope = target_analytic.abs().detach()

        lag_losses: list[torch.Tensor] = []
        for lag in resolved_lags:
            prediction_increment = (
                prediction_phasor[:, lag:]
                * prediction_phasor[:, :-lag].conj()
            )
            target_increment = (
                target_phasor[:, lag:] * target_phasor[:, :-lag].conj()
            )
            prediction_velocity = prediction_increment / torch.sqrt(
                prediction_increment.abs().square()
                + float(increment_epsilon)
            )
            target_velocity = target_increment / torch.sqrt(
                target_increment.abs().square() + float(increment_epsilon)
            )

            confidence = torch.sqrt(
                target_envelope[:, lag:]
                * target_envelope[:, :-lag]
                + float(target_scale_epsilon)
            )
            confidence = confidence / (
                confidence.mean(dim=-1, keepdim=True)
                + float(target_scale_epsilon)
            )
            confidence = confidence.clamp(max=float(envelope_weight_cap))
            circular_error = 0.5 * (
                prediction_velocity - target_velocity
            ).abs().square()
            per_sample = (
                (confidence * circular_error).sum(dim=-1)
                / (
                    confidence.sum(dim=-1)
                    + float(target_scale_epsilon)
                )
            )
            lag_losses.append(per_sample.mean())

        return torch.stack(lag_losses).mean().clamp_min(0.0)



def _soft_peak_score(
    waveform: torch.Tensor,
    *,
    temperature: float,
    variance_epsilon: float,
    minimum_scale: torch.Tensor | None = None,
) -> torch.Tensor:
    """Differentiable local-maximum confidence with a zero flat baseline."""

    centered = waveform - waveform.mean(dim=-1, keepdim=True)
    scale = torch.sqrt(
        centered.square().mean(dim=-1, keepdim=True)
        + float(variance_epsilon)
    )
    if minimum_scale is not None:
        if minimum_scale.shape != scale.shape:
            raise ValueError("minimum_scale must have shape [batch, 1]")
        scale = torch.maximum(scale, minimum_scale.detach())
    normalized = centered / scale
    center = normalized[:, 1:-1]
    rising = center - normalized[:, :-2]
    falling = center - normalized[:, 2:]
    raw = (
        torch.sigmoid(rising / float(temperature))
        * torch.sigmoid(falling / float(temperature))
    )
    return F.relu((raw - 0.25) / 0.75)


def _soft_cardiac_phase_events(
    waveform: torch.Tensor,
    *,
    temperature: float,
    variance_epsilon: float,
    minimum_scale: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return differentiable upward cardiac-phase event evidence.

    A local-maximum detector is ambiguous when a sampled sinusoid has a
    two-sample plateau. Positive variation of a soft zero-crossing state is
    plateau-safe and contributes approximately one unit of event mass per
    cardiac cycle while remaining differentiable.
    """

    centered = waveform - waveform.mean(dim=-1, keepdim=True)
    scale = torch.sqrt(
        centered.square().mean(dim=-1, keepdim=True)
        + float(variance_epsilon)
    )
    if minimum_scale is not None:
        if minimum_scale.shape != scale.shape:
            raise ValueError("minimum_scale must have shape [batch, 1]")
        # A detached target-scale floor prevents near-constant predictions
        # from creating gradients proportional to 1/sqrt(epsilon).
        scale = torch.maximum(scale, minimum_scale.detach())
    normalized = centered / scale
    phase_state = torch.sigmoid(normalized / float(temperature))
    return F.relu(phase_state[:, 1:] - phase_state[:, :-1])


def bidirectional_counting_process_distance(
    prediction_events: torch.Tensor,
    target_events: torch.Tensor,
    *,
    charbonnier_epsilon: float,
) -> dict[str, torch.Tensor]:
    """Compare pulse-event streams through forward and reverse counts.

    A shifted event creates a cumulative-count discrepancy over the interval
    between the two events. Missing or additional events leave a persistent
    discrepancy. Averaging both time directions removes dependence on whether
    an unmatched event occurs early or late in the sequence.
    """

    _validate_waveforms(prediction_events, target_events)
    epsilon = float(charbonnier_epsilon)
    if not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("charbonnier_epsilon must be finite and positive")
    event_error = prediction_events - target_events
    forward_error = torch.cumsum(event_error, dim=-1)
    backward_error = torch.flip(
        torch.cumsum(torch.flip(event_error, dims=(-1,)), dim=-1),
        dims=(-1,),
    )

    def robust_penalty(value: torch.Tensor) -> torch.Tensor:
        return torch.sqrt(value.square() + epsilon * epsilon) - epsilon

    forward = robust_penalty(forward_error).mean()
    backward = robust_penalty(backward_error).mean()
    return {
        "ccpd": 0.5 * (forward + backward),
        "ccpd_forward": forward,
        "ccpd_backward": backward,
    }


def long_horizon_ccpd(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    group_size: int,
    filter_matrix: torch.Tensor,
    edge_crop_samples: int,
    event_temperature: float,
    variance_epsilon: float,
    charbonnier_epsilon: float,
) -> dict[str, torch.Tensor]:
    """Compute long-horizon CCPD over consecutive independent clips.

    The model still processes every clip independently. Only the derivative
    outputs are concatenated here, reconstructed to PPG, detrended, filtered
    with the fixed official band-pass operator, and converted to soft cardiac
    phase events before the counting-process comparison.
    """

    batch, clip_length = _validate_waveforms(prediction, target)
    group_size = int(group_size)
    if group_size < 2 or batch % group_size != 0:
        raise ValueError(
            "CCPD batch must contain complete consecutive groups: "
            f"batch={batch}, group_size={group_size}"
        )
    sequence_count = batch // group_size
    work_dtype = _stable_work_dtype(prediction)
    with _autocast_disabled(prediction):
        prediction_work = prediction.to(dtype=work_dtype).reshape(
            sequence_count, group_size * clip_length
        )
        target_work = target.to(dtype=work_dtype).reshape(
            sequence_count, group_size * clip_length
        )
        prediction_ppg = -_linear_detrend_batch(
            torch.cumsum(prediction_work, dim=-1)
        )
        target_ppg = -_linear_detrend_batch(
            torch.cumsum(target_work, dim=-1)
        )
        matrix = filter_matrix.to(
            device=prediction.device,
            dtype=work_dtype,
        )
        expected_shape = (
            group_size * clip_length,
            group_size * clip_length,
        )
        if tuple(matrix.shape) != expected_shape:
            raise ValueError(
                "CCPD filter matrix shape does not match the long sequence: "
                f"expected={expected_shape}, actual={tuple(matrix.shape)}"
            )
        prediction_ppg = prediction_ppg @ matrix
        target_ppg = target_ppg @ matrix
        crop = int(edge_crop_samples)
        if crop < 0 or 2 * crop + 2 >= prediction_ppg.shape[-1]:
            raise ValueError("CCPD edge crop leaves fewer than two samples")
        if crop:
            prediction_ppg = prediction_ppg[:, crop:-crop]
            target_ppg = target_ppg[:, crop:-crop]

        target_scale = torch.sqrt(
            (
                target_ppg - target_ppg.mean(dim=-1, keepdim=True)
            ).square().mean(dim=-1, keepdim=True)
            + float(variance_epsilon)
        )
        prediction_events = _soft_cardiac_phase_events(
            prediction_ppg,
            temperature=float(event_temperature),
            variance_epsilon=float(variance_epsilon),
            minimum_scale=target_scale,
        )
        target_events = _soft_cardiac_phase_events(
            target_ppg,
            temperature=float(event_temperature),
            variance_epsilon=float(variance_epsilon),
        )
        terms = bidirectional_counting_process_distance(
            prediction_events,
            target_events,
            charbonnier_epsilon=float(charbonnier_epsilon),
        )
        prediction_count = prediction_events.sum(dim=-1)
        target_count = target_events.sum(dim=-1)
        return {
            **terms,
            "predicted_soft_event_count": prediction_count.mean(),
            "target_soft_event_count": target_count.mean(),
            "soft_event_count_error": (
                prediction_count - target_count
            ).abs().mean(),
        }


def cross_clip_peak_consistency(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    group_size: int,
    filter_matrix: torch.Tensor,
    edge_crop_samples: int,
    peak_temperature: float,
    smooth_l1_beta: float,
    peak_map_weight: float,
    peak_count_weight: float,
    peak_timing_weight: float,
    seam_weight: float,
    variance_epsilon: float,
) -> dict[str, torch.Tensor]:
    """Align long-context soft peak events and the three clip seams.

    The inputs are independent derivative-waveform predictions with shape
    [batch, time]. A gap-safe loader guarantees that every group_size rows are
    consecutive clips from one participant and one raw task recording.
    """

    batch, clip_length = _validate_waveforms(prediction, target)
    group_size = int(group_size)
    if batch % group_size != 0:
        raise ValueError(
            "Cross-clip batch must contain complete consecutive groups: "
            f"batch={batch}, group_size={group_size}"
        )
    sequence_count = batch // group_size
    work_dtype = _stable_work_dtype(prediction)
    with _autocast_disabled(prediction):
        prediction_work = prediction.to(dtype=work_dtype)
        target_work = target.to(dtype=work_dtype)
        prediction_sequence = prediction_work.reshape(
            sequence_count, group_size * clip_length
        )
        target_sequence = target_work.reshape(
            sequence_count, group_size * clip_length
        )

        prediction_ppg = -_linear_detrend_batch(
            torch.cumsum(prediction_sequence, dim=-1)
        )
        target_ppg = -_linear_detrend_batch(
            torch.cumsum(target_sequence, dim=-1)
        )
        matrix = filter_matrix.to(
            device=prediction.device,
            dtype=work_dtype,
        )
        expected_matrix_shape = (
            group_size * clip_length,
            group_size * clip_length,
        )
        if tuple(matrix.shape) != expected_matrix_shape:
            raise ValueError(
                "Peak filter matrix shape does not match the cross-clip sequence: "
                f"expected={expected_matrix_shape}, actual={tuple(matrix.shape)}"
            )
        prediction_ppg = prediction_ppg @ matrix
        target_ppg = target_ppg @ matrix

        crop = int(edge_crop_samples)
        if crop > 0:
            prediction_ppg = prediction_ppg[:, crop:-crop]
            target_ppg = target_ppg[:, crop:-crop]
        target_peak_scale = torch.sqrt(
            (
                target_ppg - target_ppg.mean(dim=-1, keepdim=True)
            ).square().mean(dim=-1, keepdim=True)
            + float(variance_epsilon)
        )
        prediction_peak = _soft_peak_score(
            prediction_ppg,
            temperature=float(peak_temperature),
            variance_epsilon=float(variance_epsilon),
            minimum_scale=target_peak_scale,
        )
        target_peak = _soft_peak_score(
            target_ppg,
            temperature=float(peak_temperature),
            variance_epsilon=float(variance_epsilon),
        )

        peak_map = F.smooth_l1_loss(
            prediction_peak,
            target_peak,
            beta=float(smooth_l1_beta),
        )
        prediction_count = prediction_peak.sum(dim=-1)
        target_count = target_peak.sum(dim=-1)
        target_count_floor = target_count.detach().clamp_min(1.0)
        relative_count_error = (
            prediction_count - target_count
        ) / target_count_floor
        peak_count = F.smooth_l1_loss(
            relative_count_error,
            torch.zeros_like(relative_count_error),
            beta=float(smooth_l1_beta),
        )

        prediction_cdf_count = torch.maximum(
            prediction_count,
            target_count.detach(),
        ).clamp_min(1.0)
        target_cdf_count = target_count.clamp_min(1.0)
        prediction_cdf = torch.cumsum(prediction_peak, dim=-1) / (
            prediction_cdf_count.unsqueeze(-1)
        )
        target_cdf = torch.cumsum(target_peak, dim=-1) / (
            target_cdf_count.unsqueeze(-1)
        )
        peak_timing = F.smooth_l1_loss(
            prediction_cdf,
            target_cdf,
            beta=float(smooth_l1_beta),
        )

        prediction_centered = prediction_work - prediction_work.mean(
            dim=-1, keepdim=True
        )
        target_centered = target_work - target_work.mean(
            dim=-1, keepdim=True
        )
        target_scale = torch.sqrt(
            target_centered.square().mean(dim=-1, keepdim=True)
            + float(variance_epsilon)
        )
        prediction_scale = torch.sqrt(
            prediction_centered.square().mean(dim=-1, keepdim=True)
            + float(variance_epsilon)
        )
        # The detached target scale is a floor only in the near-constant
        # regime, preventing 1/sqrt(epsilon) seam gradients. Normal predictions
        # retain their own sample-wise scale invariance.
        prediction_scale = torch.maximum(
            prediction_scale, target_scale.detach()
        )
        prediction_clip_z = (prediction_centered / prediction_scale).reshape(
            sequence_count, group_size, clip_length
        )
        target_clip_z = (target_centered / target_scale).reshape(
            sequence_count, group_size, clip_length
        )
        prediction_jump = (
            prediction_clip_z[:, 1:, 0] - prediction_clip_z[:, :-1, -1]
        )
        target_jump = target_clip_z[:, 1:, 0] - target_clip_z[:, :-1, -1]
        boundary_continuity = F.smooth_l1_loss(
            prediction_jump,
            target_jump,
            beta=float(smooth_l1_beta),
        )

        peak_consistency = (
            float(peak_map_weight) * peak_map
            + float(peak_count_weight) * peak_count
            + float(peak_timing_weight) * peak_timing
            + float(seam_weight) * boundary_continuity
        )
        return {
            "peak_consistency": peak_consistency,
            "peak_map": peak_map,
            "peak_count": peak_count,
            "peak_timing": peak_timing,
            "predicted_soft_peak_count": prediction_count.mean(),
            "target_soft_peak_count": target_count.mean(),
            "boundary_continuity": boundary_continuity,
        }



def _empty_toml_diagnostics(
    reference: torch.Tensor,
) -> dict[str, torch.Tensor]:
    count = torch.zeros((), dtype=torch.int64, device=reference.device)
    value = reference.detach().new_zeros(())
    return {
        "nonfinite_input_count": count.clone(),
        "nonfinite_system_count": count.clone(),
        "solve_failure_count": count.clone(),
        "nonfinite_coefficient_count": count.clone(),
        "nonfinite_output_count": count.clone(),
        "coefficient_norm_mean": value.clone(),
        "coefficient_norm_max": value.clone(),
    }


def _sample_nonfinite_count(*tensors: torch.Tensor) -> torch.Tensor:
    batch = int(tensors[0].shape[0])
    finite = torch.ones(
        batch,
        dtype=torch.bool,
        device=tensors[0].device,
    )
    for tensor in tensors:
        finite = finite & torch.isfinite(tensor).reshape(batch, -1).all(dim=-1)
    return (~finite).sum(dtype=torch.int64)


def _toml_extended(
    prediction: torch.Tensor,
    target: torch.Tensor,
    motion: torch.Tensor,
    *,
    max_lag_samples: int,
    ridge: float,
    epsilon: float,
    standardize_motion: bool,
    sample_weights: torch.Tensor | None,
    normalization_variance_epsilon: float | None,
    projection_epsilon: float | None,
    center_motion: bool,
    return_diagnostics: bool,
) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
    batch = int(prediction.shape[0])
    work_dtype = _stable_work_dtype(prediction)
    with _autocast_disabled(prediction):
        prediction_work = prediction.to(dtype=work_dtype)
        target_work = target.to(dtype=work_dtype)
        motion_work = motion.to(dtype=work_dtype)
        diagnostics = _empty_toml_diagnostics(prediction_work)
        diagnostics["nonfinite_input_count"] = _sample_nonfinite_count(
            prediction_work,
            target_work,
            motion_work,
        )
        if bool(diagnostics["nonfinite_input_count"] > 0):
            raise TOMLNumericalError(
                "TOML received non-finite input",
                diagnostics,
            )

        if normalization_variance_epsilon is None:
            prediction_z = samplewise_z_normalize(prediction_work, epsilon)
            target_z = samplewise_z_normalize(target_work, epsilon)
        else:
            prediction_z = samplewise_z_normalize_stable(
                prediction_work,
                normalization_variance_epsilon,
            )
            target_z = samplewise_z_normalize_stable(
                target_work,
                normalization_variance_epsilon,
            )
        error = prediction_z - target_z

        if standardize_motion:
            if normalization_variance_epsilon is None:
                motion_work = samplewise_z_normalize(motion_work, epsilon)
            else:
                motion_work = samplewise_z_normalize_stable(
                    motion_work,
                    normalization_variance_epsilon,
                )
        if center_motion:
            motion_work = motion_work - motion_work.mean(dim=-1, keepdim=True)
        dictionary = zero_padded_lag_dictionary(
            motion_work,
            max_lag_samples,
        )

        target_energy = target_z.square().sum(dim=-1, keepdim=True)
        if projection_epsilon is None:
            projection_denominator = target_energy + float(epsilon)
        else:
            projection_denominator = target_energy.clamp_min(
                float(projection_epsilon)
            )
        projection_coefficients = torch.einsum(
            "bt,btp->bp",
            target_z,
            dictionary,
        ) / projection_denominator
        dictionary_perpendicular = (
            dictionary
            - target_z.unsqueeze(-1) * projection_coefficients.unsqueeze(1)
        )

        transposed = dictionary_perpendicular.transpose(-2, -1)
        width = int(dictionary_perpendicular.shape[-1])
        identity = torch.eye(
            width,
            dtype=work_dtype,
            device=prediction.device,
        ).expand(batch, -1, -1)
        gram = transposed @ dictionary_perpendicular + float(ridge) * identity
        right_hand_side = transposed @ error.unsqueeze(-1)
        diagnostics["nonfinite_system_count"] = _sample_nonfinite_count(
            dictionary_perpendicular,
            gram,
            right_hand_side,
        )
        if bool(diagnostics["nonfinite_system_count"] > 0):
            raise TOMLNumericalError(
                "TOML linear system is non-finite",
                diagnostics,
            )

        try:
            cholesky, info = torch.linalg.cholesky_ex(
                gram,
                check_errors=False,
            )
        except RuntimeError as error:
            diagnostics["solve_failure_count"] = torch.full(
                (),
                batch,
                dtype=torch.int64,
                device=prediction.device,
            )
            raise TOMLNumericalError(
                "TOML ridge Cholesky call failed",
                diagnostics,
            ) from error
        diagnostics["solve_failure_count"] = (info != 0).sum(
            dtype=torch.int64
        )
        if bool(diagnostics["solve_failure_count"] > 0):
            raise TOMLNumericalError(
                "TOML ridge Cholesky factorization failed",
                diagnostics,
            )
        try:
            coefficients = torch.cholesky_solve(
                right_hand_side,
                cholesky,
            )
        except RuntimeError as error:
            diagnostics["solve_failure_count"] = torch.full(
                (),
                batch,
                dtype=torch.int64,
                device=prediction.device,
            )
            raise TOMLNumericalError(
                "TOML ridge solve failed",
                diagnostics,
            ) from error
        diagnostics["nonfinite_coefficient_count"] = _sample_nonfinite_count(
            coefficients
        )
        if bool(diagnostics["nonfinite_coefficient_count"] > 0):
            raise TOMLNumericalError(
                "TOML coefficients are non-finite",
                diagnostics,
            )
        coefficient_norms = torch.linalg.vector_norm(
            coefficients.detach().squeeze(-1),
            ord=2,
            dim=-1,
        )
        diagnostics["coefficient_norm_mean"] = coefficient_norms.mean()
        diagnostics["coefficient_norm_max"] = coefficient_norms.max()

        motion_residual = (
            dictionary_perpendicular @ coefficients
        ).squeeze(-1)
        energy = motion_residual.square().mean(dim=-1)
        diagnostics["nonfinite_output_count"] = (
            ~torch.isfinite(energy)
        ).sum(dtype=torch.int64)
        if bool(diagnostics["nonfinite_output_count"] > 0):
            raise TOMLNumericalError(
                "TOML motion residual is non-finite",
                diagnostics,
            )
        weights = (
            torch.ones_like(energy)
            if sample_weights is None
            else sample_weights.to(dtype=work_dtype)
        )
        denominator = weights.sum().clamp_min(torch.finfo(work_dtype).tiny)
        loss = (weights * energy).sum() / denominator
        if not bool(torch.isfinite(loss)):
            diagnostics["nonfinite_output_count"] = (
                diagnostics["nonfinite_output_count"] + 1
            )
            raise TOMLNumericalError(
                "TOML loss is non-finite",
                diagnostics,
            )
        detached = {
            key: value.detach() for key, value in diagnostics.items()
        }
        return (loss, detached) if return_diagnostics else loss


def target_orthogonal_motion_leakage(
    prediction: torch.Tensor,
    target: torch.Tensor,
    motion: torch.Tensor,
    *,
    max_lag_samples: int,
    ridge: float,
    epsilon: float = 1.0e-8,
    standardize_motion: bool = False,
    sample_weights: torch.Tensor | None = None,
    normalization_variance_epsilon: float | None = None,
    projection_epsilon: float | None = None,
    center_motion: bool = False,
    return_diagnostics: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Penalize prediction error explained by target-orthogonal motion."""

    batch, time = _validate_waveforms(prediction, target)
    if motion.ndim != 2 or tuple(motion.shape) != (batch, time):
        raise ValueError(
            f"motion must have shape {(batch, time)}, got {tuple(motion.shape)}"
        )
    if motion.device != prediction.device:
        raise ValueError("motion and prediction must be on the same device")
    if not motion.is_floating_point():
        raise TypeError("motion must be floating point")
    if not math.isfinite(ridge) or ridge <= 0:
        raise ValueError("TOML ridge must be finite and positive")
    if not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("TOML epsilon must be finite and positive")
    if sample_weights is not None:
        if sample_weights.ndim != 1 or sample_weights.shape[0] != batch:
            raise ValueError("sample_weights must have shape [batch]")
        if sample_weights.device != prediction.device:
            raise ValueError("sample_weights and prediction must share a device")
        if not sample_weights.is_floating_point():
            raise TypeError("sample_weights must be floating point")
        if not bool(torch.isfinite(sample_weights).all()):
            raise ValueError("sample_weights must be finite")
        if bool((sample_weights < 0).any()):
            raise ValueError("sample_weights must be non-negative")
    if normalization_variance_epsilon is not None:
        if (
            not math.isfinite(normalization_variance_epsilon)
            or normalization_variance_epsilon <= 0
        ):
            raise ValueError(
                "normalization_variance_epsilon must be finite and positive"
            )
    if projection_epsilon is not None:
        if not math.isfinite(projection_epsilon) or projection_epsilon <= 0:
            raise ValueError("projection_epsilon must be finite and positive")
    if not isinstance(center_motion, bool):
        raise TypeError("center_motion must be boolean")
    if not isinstance(return_diagnostics, bool):
        raise TypeError("return_diagnostics must be boolean")

    extended = bool(
        normalization_variance_epsilon is not None
        or projection_epsilon is not None
        or center_motion
        or return_diagnostics
    )
    if extended:
        return _toml_extended(
            prediction,
            target,
            motion,
            max_lag_samples=max_lag_samples,
            ridge=ridge,
            epsilon=epsilon,
            standardize_motion=standardize_motion,
            sample_weights=sample_weights,
            normalization_variance_epsilon=normalization_variance_epsilon,
            projection_epsilon=projection_epsilon,
            center_motion=center_motion,
            return_diagnostics=return_diagnostics,
        )

    work_dtype = _stable_work_dtype(prediction)
    with _autocast_disabled(prediction):
        prediction_work = prediction.to(dtype=work_dtype)
        target_work = target.to(dtype=work_dtype)
        motion_work = motion.to(dtype=work_dtype)
        prediction_z = samplewise_z_normalize(prediction_work, epsilon)
        target_z = samplewise_z_normalize(target_work, epsilon)
        error = prediction_z - target_z
        if standardize_motion:
            motion_work = samplewise_z_normalize(motion_work, epsilon)
        dictionary = zero_padded_lag_dictionary(
            motion_work,
            max_lag_samples=max_lag_samples,
        )
        projection_coefficients = torch.einsum(
            "bt,btp->bp", target_z, dictionary
        ) / (target_z.square().sum(dim=-1, keepdim=True) + float(epsilon))
        dictionary_perpendicular = dictionary - target_z.unsqueeze(
            -1
        ) * projection_coefficients.unsqueeze(1)

        transposed = dictionary_perpendicular.transpose(-2, -1)
        dictionary_width = int(dictionary_perpendicular.shape[-1])
        identity = torch.eye(
            dictionary_width,
            dtype=work_dtype,
            device=prediction.device,
        ).expand(batch, -1, -1)
        gram = transposed @ dictionary_perpendicular + float(ridge) * identity
        right_hand_side = transposed @ error.unsqueeze(-1)
        coefficients = torch.linalg.solve(gram, right_hand_side)
        motion_residual = (
            dictionary_perpendicular @ coefficients
        ).squeeze(-1)
        energy = motion_residual.square().mean(dim=-1)
        if sample_weights is None:
            weights = torch.ones_like(energy)
        else:
            weights = sample_weights.to(dtype=work_dtype)
        return (weights * energy).sum() / (
            weights.sum() + float(epsilon)
        )


def _check_allowed_keys(
    payload: dict[str, Any],
    allowed: set[str],
    location: str,
) -> None:
    unknown = sorted(set(payload) - allowed)
    if unknown:
        raise ValueError(f"Unknown {location} keys: {unknown}")


def _finite_nonnegative(value: Any, name: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise ValueError(f"{name} must be finite and non-negative")
    return result


def _finite_positive(value: Any, name: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return result


def _strict_int(value: Any, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or int(value) != value:
        raise ValueError(f"{name} must be an integer")
    result = int(value)
    if result < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return result


def _loss_submapping(config: dict[str, Any] | None) -> dict[str, Any] | None:
    if config is None:
        return None
    if not isinstance(config, dict):
        raise TypeError("loss config must be a mapping")
    if "loss" in config and any(
        key in config for key in ("experiment", "paths", "data", "model", "training")
    ):
        nested = config["loss"]
        if not isinstance(nested, dict):
            raise TypeError("top-level loss must be a mapping")
        return nested
    return config


def resolve_training_loss_config(
    loss_config: dict[str, Any] | None,
    *,
    expected_length: int | None = None,
    evaluation_fs: float | None = None,
) -> dict[str, Any]:
    """Validate and canonicalize the configurable training loss contract."""

    raw_input = _loss_submapping(loss_config)
    raw = deepcopy(raw_input) if raw_input is not None else {}
    _check_allowed_keys(
        raw,
        {
            "name",
            "implementation_version",
            "waveform",
            "spectral",
            "toml",
            "peak_consistency",
            "ccpd",
            "dpd",
            "cpv",
        },
        "loss",
    )
    name = str(raw.get("name", "baseline_z_mse")).lower()
    if name not in {
        "baseline_z_mse",
        "motion_disentangled_multi_domain",
        "waveform_spectral_peak_consistency",
        "waveform_long_horizon_ccpd",
        "derivative_pulse_dual_domain",
        "cardiac_phase_velocity",
    }:
        raise ValueError(f"Unsupported loss.name: {name!r}")
    version = str(
        raw.get("implementation_version", LOSS_IMPLEMENTATION_VERSION)
    )
    if version not in SUPPORTED_LOSS_IMPLEMENTATION_VERSIONS:
        raise ValueError(
            f"Unsupported loss implementation_version: {version!r}"
        )

    waveform_raw = raw.get("waveform", {})
    if not isinstance(waveform_raw, dict):
        raise TypeError("loss.waveform must be a mapping")
    _check_allowed_keys(
        waveform_raw,
        {"normalization", "epsilon"},
        "loss.waveform",
    )
    waveform_normalization = str(
        waveform_raw.get("normalization", "batch_global_z")
    ).lower()
    if waveform_normalization != "batch_global_z":
        raise ValueError("loss.waveform.normalization must be batch_global_z")
    waveform = {
        "normalization": waveform_normalization,
        "epsilon": _finite_positive(
            waveform_raw.get("epsilon", 1.0e-8),
            "loss.waveform.epsilon",
        ),
    }

    default_fs = float(evaluation_fs) if evaluation_fs is not None else 30.0
    default_n_fft = int(expected_length) if expected_length is not None else 128
    spectral_raw = raw.get("spectral", {})
    if not isinstance(spectral_raw, dict):
        raise TypeError("loss.spectral must be a mapping")
    _check_allowed_keys(
        spectral_raw,
        {
            "enabled",
            "weight",
            "sampling_rate_hz",
            "band_hz",
            "n_fft",
            "hann_periodic",
            "temperature",
            "power_epsilon",
        },
        "loss.spectral",
    )
    spectral_enabled = bool(spectral_raw.get("enabled", False))
    spectral_weight = _finite_nonnegative(
        spectral_raw.get("weight", 0.0),
        "loss.spectral.weight",
    )
    spectral = {
        "enabled": spectral_enabled,
        "weight": spectral_weight,
        "sampling_rate_hz": _finite_positive(
            spectral_raw.get("sampling_rate_hz", default_fs),
            "loss.spectral.sampling_rate_hz",
        ),
        "band_hz": [
            float(value)
            for value in spectral_raw.get("band_hz", [0.7, 2.8])
        ],
        "n_fft": _strict_int(
            spectral_raw.get("n_fft", default_n_fft),
            "loss.spectral.n_fft",
            minimum=2,
        ),
        "hann_periodic": bool(spectral_raw.get("hann_periodic", True)),
        "temperature": _finite_positive(
            spectral_raw.get("temperature", 1.0),
            "loss.spectral.temperature",
        ),
        "power_epsilon": _finite_positive(
            spectral_raw.get("power_epsilon", 1.0e-8),
            "loss.spectral.power_epsilon",
        ),
    }
    if spectral_enabled and spectral_weight <= 0:
        raise ValueError("enabled spectral loss requires a positive weight")
    if not spectral_enabled and spectral_weight != 0:
        raise ValueError("disabled spectral loss requires weight=0")
    if spectral_enabled:
        if evaluation_fs is not None and not math.isclose(
            spectral["sampling_rate_hz"],
            float(evaluation_fs),
            rel_tol=0.0,
            abs_tol=1.0e-12,
        ):
            raise ValueError(
                "loss.spectral.sampling_rate_hz must match evaluation.fs"
            )
        validation_time = (
            int(expected_length)
            if expected_length is not None
            else spectral["n_fft"]
        )
        _validate_spectral_parameters(
            time=validation_time,
            sampling_rate_hz=spectral["sampling_rate_hz"],
            band_hz=spectral["band_hz"],
            n_fft=spectral["n_fft"],
            temperature=spectral["temperature"],
            power_epsilon=spectral["power_epsilon"],
        )

    toml_raw = raw.get("toml", {})
    if not isinstance(toml_raw, dict):
        raise TypeError("loss.toml must be a mapping")
    common_toml_keys = {
        "enabled",
        "weight",
        "max_lag_samples",
        "ridge",
        "lag_padding",
        "standardize_motion",
        "motion_gate",
    }
    if version in {
        LOSS_IMPLEMENTATION_VERSION,
        PEAK_LOSS_IMPLEMENTATION_VERSION,
        CCPD_LOSS_IMPLEMENTATION_VERSION,
        DPD_LOSS_IMPLEMENTATION_VERSION,
        CPV_LOSS_IMPLEMENTATION_VERSION,
    }:
        allowed_toml_keys = common_toml_keys | {"epsilon"}
    else:
        allowed_toml_keys = common_toml_keys | {
            "normalization_variance_epsilon",
            "projection_epsilon",
            "center_motion",
            "schedule",
        }
    _check_allowed_keys(toml_raw, allowed_toml_keys, "loss.toml")
    toml_enabled = bool(toml_raw.get("enabled", False))
    toml_weight = _finite_nonnegative(
        toml_raw.get("weight", 0.0),
        "loss.toml.weight",
    )
    if version in {
        LOSS_IMPLEMENTATION_VERSION,
        PEAK_LOSS_IMPLEMENTATION_VERSION,
        CCPD_LOSS_IMPLEMENTATION_VERSION,
        DPD_LOSS_IMPLEMENTATION_VERSION,
        CPV_LOSS_IMPLEMENTATION_VERSION,
    }:
        toml = {
            "enabled": toml_enabled,
            "weight": toml_weight,
            "max_lag_samples": _strict_int(
                toml_raw.get("max_lag_samples", 8),
                "loss.toml.max_lag_samples",
            ),
            "ridge": _finite_positive(
                toml_raw.get("ridge", 1.0e-3),
                "loss.toml.ridge",
            ),
            "epsilon": _finite_positive(
                toml_raw.get("epsilon", 1.0e-8),
                "loss.toml.epsilon",
            ),
            "lag_padding": str(
                toml_raw.get("lag_padding", "zero")
            ).lower(),
            "standardize_motion": bool(
                toml_raw.get("standardize_motion", False)
            ),
            "motion_gate": str(
                toml_raw.get("motion_gate", "uniform")
            ).lower(),
        }
    else:
        toml = {
            "enabled": toml_enabled,
            "weight": toml_weight,
            "max_lag_samples": _strict_int(
                toml_raw.get("max_lag_samples", 8),
                "loss.toml.max_lag_samples",
            ),
            "ridge": _finite_positive(
                toml_raw.get("ridge", 1.0e-3),
                "loss.toml.ridge",
            ),
            "normalization_variance_epsilon": _finite_positive(
                toml_raw.get("normalization_variance_epsilon", 1.0e-8),
                "loss.toml.normalization_variance_epsilon",
            ),
            "projection_epsilon": _finite_positive(
                toml_raw.get("projection_epsilon", 1.0e-8),
                "loss.toml.projection_epsilon",
            ),
            "center_motion": bool(toml_raw.get("center_motion", True)),
            "lag_padding": str(
                toml_raw.get("lag_padding", "zero")
            ).lower(),
            "standardize_motion": bool(
                toml_raw.get("standardize_motion", False)
            ),
            "motion_gate": str(
                toml_raw.get("motion_gate", "uniform")
            ).lower(),
        }
        if "schedule" in toml_raw:
            schedule_raw = toml_raw["schedule"]
            if not isinstance(schedule_raw, dict):
                raise TypeError("loss.toml.schedule must be a mapping")
            _check_allowed_keys(
                schedule_raw,
                {"mode", "warmup_epochs", "ramp_epochs"},
                "loss.toml.schedule",
            )
            schedule_mode = str(
                schedule_raw.get("mode", "delayed_linear_ramp")
            ).lower()
            if schedule_mode != "delayed_linear_ramp":
                raise ValueError(
                    "loss.toml.schedule.mode must be delayed_linear_ramp"
                )
            warmup_epochs = _strict_int(
                schedule_raw.get("warmup_epochs", 0),
                "loss.toml.schedule.warmup_epochs",
            )
            ramp_epochs = _strict_int(
                schedule_raw.get("ramp_epochs", 1),
                "loss.toml.schedule.ramp_epochs",
            )
            if warmup_epochs < 0:
                raise ValueError(
                    "loss.toml.schedule.warmup_epochs must be non-negative"
                )
            if ramp_epochs <= 0:
                raise ValueError(
                    "loss.toml.schedule.ramp_epochs must be positive"
                )
            toml["schedule"] = {
                "mode": schedule_mode,
                "warmup_epochs": warmup_epochs,
                "ramp_epochs": ramp_epochs,
            }
    if toml_enabled and toml_weight <= 0:
        raise ValueError("enabled TOML requires a positive weight")
    if not toml_enabled and toml_weight != 0:
        raise ValueError("disabled TOML requires weight=0")
    if toml["lag_padding"] != "zero":
        raise ValueError("loss.toml.lag_padding must be zero")
    if toml["motion_gate"] != "uniform":
        raise ValueError(
            "Only uniform TOML motion_gate is defined; g_b is fixed to one"
        )
    if (
        version == LATEST_LOSS_IMPLEMENTATION_VERSION
        and toml_enabled
        and not toml["center_motion"]
    ):
        raise ValueError("mdm_v2 TOML requires loss.toml.center_motion=true")
    if (
        version == LATEST_LOSS_IMPLEMENTATION_VERSION
        and toml_enabled
        and toml["standardize_motion"]
    ):
        raise ValueError(
            "mdm_v2 TOML requires loss.toml.standardize_motion=false"
        )
    if toml_enabled and expected_length is not None:
        if 2 * toml["max_lag_samples"] + 1 > int(expected_length):
            raise ValueError("enabled TOML requires 2K+1 <= chunk length")


    dpd: dict[str, Any] | None = None
    if "dpd" in raw:
        dpd_raw = raw["dpd"]
        if not isinstance(dpd_raw, dict):
            raise TypeError("loss.dpd must be a mapping")
        _check_allowed_keys(
            dpd_raw,
            {
                "enabled",
                "weight",
                "transform",
                "sampling_rate_hz",
                "band_hz",
                "filter_order",
                "filter_mode",
                "edge_crop_samples",
                "normalization",
                "epsilon",
            },
            "loss.dpd",
        )
        dpd_enabled = bool(dpd_raw.get("enabled", False))
        dpd_weight = _finite_nonnegative(
            dpd_raw.get("weight", 0.0),
            "loss.dpd.weight",
        )
        dpd_filter_order = _strict_int(
            dpd_raw.get("filter_order", 4),
            "loss.dpd.filter_order",
            minimum=1,
        )
        default_dpd_crop = 3 * (2 * dpd_filter_order + 1)
        dpd = {
            "enabled": dpd_enabled,
            "weight": dpd_weight,
            "transform": str(
                dpd_raw.get("transform", "reconstructed_pulse")
            ).lower(),
            "sampling_rate_hz": _finite_positive(
                dpd_raw.get("sampling_rate_hz", default_fs),
                "loss.dpd.sampling_rate_hz",
            ),
            "band_hz": [
                float(value)
                for value in dpd_raw.get("band_hz", [0.7, 2.8])
            ],
            "filter_order": dpd_filter_order,
            "filter_mode": str(
                dpd_raw.get("filter_mode", "official_filtfilt_matrix")
            ).lower(),
            "edge_crop_samples": _strict_int(
                dpd_raw.get("edge_crop_samples", default_dpd_crop),
                "loss.dpd.edge_crop_samples",
            ),
            "normalization": str(
                dpd_raw.get("normalization", "batch_global_z")
            ).lower(),
            "epsilon": _finite_positive(
                dpd_raw.get("epsilon", 1.0e-8),
                "loss.dpd.epsilon",
            ),
        }
        if dpd["transform"] not in {
            "reconstructed_pulse",
            "bandpass_derivative_control",
        }:
            raise ValueError(
                "loss.dpd.transform must be reconstructed_pulse or "
                "bandpass_derivative_control"
            )
        if len(dpd["band_hz"]) != 2:
            raise ValueError("loss.dpd.band_hz must contain [low, high]")
        dpd_low, dpd_high = dpd["band_hz"]
        if not (
            0.0
            < dpd_low
            < dpd_high
            <= dpd["sampling_rate_hz"] / 2.0
        ):
            raise ValueError("loss.dpd.band_hz must lie inside Nyquist")
        if dpd["filter_mode"] != "official_filtfilt_matrix":
            raise ValueError(
                "loss.dpd.filter_mode must be official_filtfilt_matrix"
            )
        if dpd["normalization"] != "batch_global_z":
            raise ValueError(
                "loss.dpd.normalization must be batch_global_z"
            )
        if evaluation_fs is not None and not math.isclose(
            dpd["sampling_rate_hz"],
            float(evaluation_fs),
            rel_tol=0.0,
            abs_tol=1.0e-12,
        ):
            raise ValueError(
                "loss.dpd.sampling_rate_hz must match evaluation.fs"
            )
        if dpd_enabled and dpd_weight <= 0:
            raise ValueError("Enabled DPD requires a positive weight")
        if not dpd_enabled and dpd_weight != 0:
            raise ValueError("Disabled DPD requires weight=0")
        if expected_length is None:
            if dpd_enabled:
                raise ValueError("Enabled DPD requires expected clip length")
            dpd_sequence_length = default_n_fft
        else:
            dpd_sequence_length = int(expected_length)
        dpd_filter_pad_length = 3 * (2 * dpd_filter_order + 1)
        if dpd_sequence_length <= dpd_filter_pad_length:
            raise ValueError(
                "DPD sequence is too short for official filtfilt padding"
            )
        if 2 * dpd["edge_crop_samples"] + 2 > dpd_sequence_length:
            raise ValueError("DPD edge crop leaves fewer than two samples")
        dpd["sequence_length"] = dpd_sequence_length


    cpv: dict[str, Any] | None = None
    if "cpv" in raw:
        cpv_raw = raw["cpv"]
        if not isinstance(cpv_raw, dict):
            raise TypeError("loss.cpv must be a mapping")
        _check_allowed_keys(
            cpv_raw,
            {
                "enabled",
                "weight",
                "transform",
                "sampling_rate_hz",
                "band_hz",
                "filter_order",
                "filter_mode",
                "edge_crop_samples",
                "lags",
                "analytic_transform",
                "analytic_epsilon",
                "increment_epsilon",
                "target_scale_epsilon",
                "envelope_weighting",
                "envelope_weight_cap",
                "schedule",
            },
            "loss.cpv",
        )
        cpv_enabled = bool(cpv_raw.get("enabled", False))
        cpv_weight = _finite_nonnegative(
            cpv_raw.get("weight", 0.0),
            "loss.cpv.weight",
        )
        cpv_filter_order = _strict_int(
            cpv_raw.get("filter_order", 4),
            "loss.cpv.filter_order",
            minimum=1,
        )
        default_cpv_crop = 3 * (2 * cpv_filter_order + 1)
        cpv_lags_raw = cpv_raw.get("lags", [1, 2, 4])
        if not isinstance(cpv_lags_raw, (list, tuple)) or not cpv_lags_raw:
            raise ValueError("loss.cpv.lags must be a non-empty list")
        cpv_lags = [
            _strict_int(value, "loss.cpv.lags", minimum=1)
            for value in cpv_lags_raw
        ]
        if cpv_lags != sorted(set(cpv_lags)):
            raise ValueError(
                "loss.cpv.lags must be strictly increasing and unique"
            )

        cpv_schedule_raw = cpv_raw.get(
            "schedule",
            {
                "mode": "delayed_linear_ramp",
                "warmup_epochs": 5,
                "ramp_epochs": 10,
            },
        )
        if not isinstance(cpv_schedule_raw, dict):
            raise TypeError("loss.cpv.schedule must be a mapping")
        _check_allowed_keys(
            cpv_schedule_raw,
            {"mode", "warmup_epochs", "ramp_epochs"},
            "loss.cpv.schedule",
        )
        cpv_schedule_mode = str(
            cpv_schedule_raw.get("mode", "delayed_linear_ramp")
        ).lower()
        if cpv_schedule_mode != "delayed_linear_ramp":
            raise ValueError(
                "loss.cpv.schedule.mode must be delayed_linear_ramp"
            )

        cpv = {
            "enabled": cpv_enabled,
            "weight": cpv_weight,
            "transform": str(
                cpv_raw.get("transform", "reconstructed_pulse")
            ).lower(),
            "sampling_rate_hz": _finite_positive(
                cpv_raw.get("sampling_rate_hz", default_fs),
                "loss.cpv.sampling_rate_hz",
            ),
            "band_hz": [
                float(value)
                for value in cpv_raw.get("band_hz", [0.7, 2.8])
            ],
            "filter_order": cpv_filter_order,
            "filter_mode": str(
                cpv_raw.get("filter_mode", "official_filtfilt_matrix")
            ).lower(),
            "edge_crop_samples": _strict_int(
                cpv_raw.get("edge_crop_samples", default_cpv_crop),
                "loss.cpv.edge_crop_samples",
            ),
            "lags": cpv_lags,
            "analytic_transform": str(
                cpv_raw.get("analytic_transform", "fft_hilbert")
            ).lower(),
            "analytic_epsilon": _finite_positive(
                cpv_raw.get("analytic_epsilon", 1.0e-2),
                "loss.cpv.analytic_epsilon",
            ),
            "increment_epsilon": _finite_positive(
                cpv_raw.get("increment_epsilon", 1.0e-2),
                "loss.cpv.increment_epsilon",
            ),
            "target_scale_epsilon": _finite_positive(
                cpv_raw.get("target_scale_epsilon", 1.0e-8),
                "loss.cpv.target_scale_epsilon",
            ),
            "envelope_weighting": str(
                cpv_raw.get(
                    "envelope_weighting",
                    "target_geometric_mean",
                )
            ).lower(),
            "envelope_weight_cap": _finite_positive(
                cpv_raw.get("envelope_weight_cap", 2.0),
                "loss.cpv.envelope_weight_cap",
            ),
            "schedule": {
                "mode": cpv_schedule_mode,
                "warmup_epochs": _strict_int(
                    cpv_schedule_raw.get("warmup_epochs", 5),
                    "loss.cpv.schedule.warmup_epochs",
                ),
                "ramp_epochs": _strict_int(
                    cpv_schedule_raw.get("ramp_epochs", 10),
                    "loss.cpv.schedule.ramp_epochs",
                    minimum=1,
                ),
            },
        }
        if cpv["transform"] != "reconstructed_pulse":
            raise ValueError(
                "loss.cpv.transform must be reconstructed_pulse"
            )
        if cpv["filter_mode"] != "official_filtfilt_matrix":
            raise ValueError(
                "loss.cpv.filter_mode must be official_filtfilt_matrix"
            )
        if cpv["analytic_transform"] != "fft_hilbert":
            raise ValueError(
                "loss.cpv.analytic_transform must be fft_hilbert"
            )
        if cpv["envelope_weighting"] != "target_geometric_mean":
            raise ValueError(
                "loss.cpv.envelope_weighting must be "
                "target_geometric_mean"
            )
        if len(cpv["band_hz"]) != 2 or not all(
            math.isfinite(value) for value in cpv["band_hz"]
        ):
            raise ValueError(
                "loss.cpv.band_hz must contain two finite values"
            )
        cpv_low, cpv_high = cpv["band_hz"]
        if not (
            0.0
            < cpv_low
            < cpv_high
            <= cpv["sampling_rate_hz"] / 2.0
        ):
            raise ValueError("loss.cpv.band_hz must lie inside Nyquist")
        if evaluation_fs is not None and not math.isclose(
            cpv["sampling_rate_hz"],
            float(evaluation_fs),
            rel_tol=0.0,
            abs_tol=1.0e-12,
        ):
            raise ValueError(
                "loss.cpv.sampling_rate_hz must match evaluation.fs"
            )
        if cpv_enabled and cpv_weight <= 0:
            raise ValueError("Enabled CPV requires a positive weight")
        if not cpv_enabled and cpv_weight != 0:
            raise ValueError("Disabled CPV requires weight=0")
        if expected_length is None:
            if cpv_enabled:
                raise ValueError("Enabled CPV requires expected clip length")
            cpv_sequence_length = default_n_fft
        else:
            cpv_sequence_length = int(expected_length)
        cpv_filter_pad_length = 3 * (2 * cpv_filter_order + 1)
        if cpv_sequence_length <= cpv_filter_pad_length:
            raise ValueError(
                "CPV sequence is too short for official filtfilt padding"
            )
        cpv_remaining_length = (
            cpv_sequence_length - 2 * cpv["edge_crop_samples"]
        )
        if cpv_remaining_length <= max(cpv["lags"]):
            raise ValueError(
                "CPV edge crop leaves too few samples for its lags"
            )
        cpv["sequence_length"] = cpv_sequence_length



    peak: dict[str, Any] | None = None
    if "peak_consistency" in raw:
        peak_raw = raw["peak_consistency"]
        if not isinstance(peak_raw, dict):
            raise TypeError("loss.peak_consistency must be a mapping")
        _check_allowed_keys(
            peak_raw,
            {
                "enabled",
                "weight",
                "group_size",
                "sampling_rate_hz",
                "band_hz",
                "filter_order",
                "filter_mode",
                "edge_crop_samples",
                "normalization_variance_epsilon",
                "peak_temperature",
                "smooth_l1_beta",
                "peak_map_weight",
                "peak_count_weight",
                "peak_timing_weight",
                "seam_weight",
                "schedule",
            },
            "loss.peak_consistency",
        )
        peak_enabled = bool(peak_raw.get("enabled", False))
        peak_weight = _finite_nonnegative(
            peak_raw.get("weight", 0.0),
            "loss.peak_consistency.weight",
        )
        group_size = _strict_int(
            peak_raw.get("group_size", 4),
            "loss.peak_consistency.group_size",
            minimum=2,
        )
        filter_order = _strict_int(
            peak_raw.get("filter_order", 4),
            "loss.peak_consistency.filter_order",
            minimum=1,
        )
        default_edge_crop = 3 * (2 * filter_order + 1)
        component_weights = {
            "peak_map_weight": _finite_nonnegative(
                peak_raw.get("peak_map_weight", 1.0),
                "loss.peak_consistency.peak_map_weight",
            ),
            "peak_count_weight": _finite_nonnegative(
                peak_raw.get("peak_count_weight", 0.5),
                "loss.peak_consistency.peak_count_weight",
            ),
            "peak_timing_weight": _finite_nonnegative(
                peak_raw.get("peak_timing_weight", 0.25),
                "loss.peak_consistency.peak_timing_weight",
            ),
            "seam_weight": _finite_nonnegative(
                peak_raw.get("seam_weight", 0.1),
                "loss.peak_consistency.seam_weight",
            ),
        }
        if not any(value > 0 for value in component_weights.values()):
            raise ValueError(
                "Enabled peak consistency requires a positive component weight"
            )
        peak = {
            "enabled": peak_enabled,
            "weight": peak_weight,
            "group_size": group_size,
            "sampling_rate_hz": _finite_positive(
                peak_raw.get("sampling_rate_hz", default_fs),
                "loss.peak_consistency.sampling_rate_hz",
            ),
            "band_hz": [
                float(value)
                for value in peak_raw.get("band_hz", [0.7, 2.8])
            ],
            "filter_order": filter_order,
            "filter_mode": str(
                peak_raw.get("filter_mode", "official_filtfilt_matrix")
            ).lower(),
            "edge_crop_samples": _strict_int(
                peak_raw.get("edge_crop_samples", default_edge_crop),
                "loss.peak_consistency.edge_crop_samples",
            ),
            "normalization_variance_epsilon": _finite_positive(
                peak_raw.get("normalization_variance_epsilon", 1.0e-6),
                "loss.peak_consistency.normalization_variance_epsilon",
            ),
            "peak_temperature": _finite_positive(
                peak_raw.get("peak_temperature", 0.02),
                "loss.peak_consistency.peak_temperature",
            ),
            "smooth_l1_beta": _finite_positive(
                peak_raw.get("smooth_l1_beta", 0.1),
                "loss.peak_consistency.smooth_l1_beta",
            ),
            **component_weights,
        }
        if len(peak["band_hz"]) != 2:
            raise ValueError(
                "loss.peak_consistency.band_hz must contain [low, high]"
            )
        peak_low, peak_high = peak["band_hz"]
        if not (
            0.0
            < peak_low
            < peak_high
            <= peak["sampling_rate_hz"] / 2.0
        ):
            raise ValueError(
                "loss.peak_consistency.band_hz must lie inside Nyquist"
            )
        if peak["filter_mode"] != "official_filtfilt_matrix":
            raise ValueError(
                "loss.peak_consistency.filter_mode must be "
                "official_filtfilt_matrix"
            )
        if evaluation_fs is not None and not math.isclose(
            peak["sampling_rate_hz"],
            float(evaluation_fs),
            rel_tol=0.0,
            abs_tol=1.0e-12,
        ):
            raise ValueError(
                "loss.peak_consistency.sampling_rate_hz must match evaluation.fs"
            )
        if peak_enabled and peak_weight <= 0:
            raise ValueError(
                "Enabled peak consistency requires a positive weight"
            )
        if not peak_enabled and peak_weight != 0:
            raise ValueError(
                "Disabled peak consistency requires weight=0"
            )
        if expected_length is None:
            if peak_enabled:
                raise ValueError(
                    "Enabled peak consistency requires expected clip length"
                )
            sequence_length = group_size * default_n_fft
        else:
            sequence_length = group_size * int(expected_length)
        filter_pad_length = 3 * (2 * filter_order + 1)
        if sequence_length <= filter_pad_length:
            raise ValueError(
                "Cross-clip sequence is too short for official filtfilt padding"
            )
        if 2 * peak["edge_crop_samples"] + 3 >= sequence_length:
            raise ValueError(
                "Peak edge crop leaves fewer than three samples"
            )
        peak["sequence_length"] = sequence_length
        schedule_raw = peak_raw.get(
            "schedule",
            {
                "mode": "delayed_linear_ramp",
                "warmup_epochs": 5,
                "ramp_epochs": 10,
            },
        )
        if not isinstance(schedule_raw, dict):
            raise TypeError("loss.peak_consistency.schedule must be a mapping")
        _check_allowed_keys(
            schedule_raw,
            {"mode", "warmup_epochs", "ramp_epochs"},
            "loss.peak_consistency.schedule",
        )
        schedule_mode = str(
            schedule_raw.get("mode", "delayed_linear_ramp")
        ).lower()
        if schedule_mode != "delayed_linear_ramp":
            raise ValueError(
                "loss.peak_consistency.schedule.mode must be "
                "delayed_linear_ramp"
            )
        peak["schedule"] = {
            "mode": schedule_mode,
            "warmup_epochs": _strict_int(
                schedule_raw.get("warmup_epochs", 5),
                "loss.peak_consistency.schedule.warmup_epochs",
            ),
            "ramp_epochs": _strict_int(
                schedule_raw.get("ramp_epochs", 10),
                "loss.peak_consistency.schedule.ramp_epochs",
                minimum=1,
            ),
        }

    ccpd: dict[str, Any] | None = None
    if "ccpd" in raw:
        ccpd_raw = raw["ccpd"]
        if not isinstance(ccpd_raw, dict):
            raise TypeError("loss.ccpd must be a mapping")
        _check_allowed_keys(
            ccpd_raw,
            {
                "enabled",
                "weight",
                "group_size",
                "sampling_rate_hz",
                "band_hz",
                "filter_order",
                "filter_mode",
                "edge_crop_samples",
                "normalization_variance_epsilon",
                "event_temperature",
                "charbonnier_epsilon",
                "schedule",
            },
            "loss.ccpd",
        )
        ccpd_enabled = bool(ccpd_raw.get("enabled", False))
        ccpd_weight = _finite_nonnegative(
            ccpd_raw.get("weight", 0.0),
            "loss.ccpd.weight",
        )
        ccpd_group_size = _strict_int(
            ccpd_raw.get("group_size", 4),
            "loss.ccpd.group_size",
            minimum=2,
        )
        ccpd_filter_order = _strict_int(
            ccpd_raw.get("filter_order", 4),
            "loss.ccpd.filter_order",
            minimum=1,
        )
        default_ccpd_crop = 3 * (2 * ccpd_filter_order + 1)
        ccpd = {
            "enabled": ccpd_enabled,
            "weight": ccpd_weight,
            "group_size": ccpd_group_size,
            "sampling_rate_hz": _finite_positive(
                ccpd_raw.get("sampling_rate_hz", default_fs),
                "loss.ccpd.sampling_rate_hz",
            ),
            "band_hz": [
                float(value)
                for value in ccpd_raw.get("band_hz", [0.7, 2.8])
            ],
            "filter_order": ccpd_filter_order,
            "filter_mode": str(
                ccpd_raw.get("filter_mode", "official_filtfilt_matrix")
            ).lower(),
            "edge_crop_samples": _strict_int(
                ccpd_raw.get("edge_crop_samples", default_ccpd_crop),
                "loss.ccpd.edge_crop_samples",
            ),
            "normalization_variance_epsilon": _finite_positive(
                ccpd_raw.get("normalization_variance_epsilon", 1.0e-6),
                "loss.ccpd.normalization_variance_epsilon",
            ),
            "event_temperature": _finite_positive(
                ccpd_raw.get("event_temperature", 0.1),
                "loss.ccpd.event_temperature",
            ),
            "charbonnier_epsilon": _finite_positive(
                ccpd_raw.get("charbonnier_epsilon", 1.0e-3),
                "loss.ccpd.charbonnier_epsilon",
            ),
        }
        if len(ccpd["band_hz"]) != 2:
            raise ValueError("loss.ccpd.band_hz must contain [low, high]")
        ccpd_low, ccpd_high = ccpd["band_hz"]
        if not (
            0.0
            < ccpd_low
            < ccpd_high
            <= ccpd["sampling_rate_hz"] / 2.0
        ):
            raise ValueError("loss.ccpd.band_hz must lie inside Nyquist")
        if ccpd["filter_mode"] != "official_filtfilt_matrix":
            raise ValueError(
                "loss.ccpd.filter_mode must be official_filtfilt_matrix"
            )
        if evaluation_fs is not None and not math.isclose(
            ccpd["sampling_rate_hz"],
            float(evaluation_fs),
            rel_tol=0.0,
            abs_tol=1.0e-12,
        ):
            raise ValueError(
                "loss.ccpd.sampling_rate_hz must match evaluation.fs"
            )
        if ccpd_enabled and ccpd_weight <= 0:
            raise ValueError("Enabled CCPD requires a positive weight")
        if not ccpd_enabled and ccpd_weight != 0:
            raise ValueError("Disabled CCPD requires weight=0")
        if expected_length is None:
            if ccpd_enabled:
                raise ValueError(
                    "Enabled CCPD requires expected clip length"
                )
            ccpd_sequence_length = ccpd_group_size * default_n_fft
        else:
            ccpd_sequence_length = (
                ccpd_group_size * int(expected_length)
            )
        ccpd_filter_pad_length = 3 * (2 * ccpd_filter_order + 1)
        if ccpd_sequence_length <= ccpd_filter_pad_length:
            raise ValueError(
                "CCPD sequence is too short for official filtfilt padding"
            )
        if (
            2 * ccpd["edge_crop_samples"] + 2
            >= ccpd_sequence_length
        ):
            raise ValueError(
                "CCPD edge crop leaves fewer than two samples"
            )
        ccpd["sequence_length"] = ccpd_sequence_length
        ccpd_schedule_raw = ccpd_raw.get(
            "schedule",
            {
                "mode": "delayed_linear_ramp",
                "warmup_epochs": 5,
                "ramp_epochs": 10,
            },
        )
        if not isinstance(ccpd_schedule_raw, dict):
            raise TypeError("loss.ccpd.schedule must be a mapping")
        _check_allowed_keys(
            ccpd_schedule_raw,
            {"mode", "warmup_epochs", "ramp_epochs"},
            "loss.ccpd.schedule",
        )
        ccpd_schedule_mode = str(
            ccpd_schedule_raw.get("mode", "delayed_linear_ramp")
        ).lower()
        if ccpd_schedule_mode != "delayed_linear_ramp":
            raise ValueError(
                "loss.ccpd.schedule.mode must be delayed_linear_ramp"
            )
        ccpd["schedule"] = {
            "mode": ccpd_schedule_mode,
            "warmup_epochs": _strict_int(
                ccpd_schedule_raw.get("warmup_epochs", 5),
                "loss.ccpd.schedule.warmup_epochs",
            ),
            "ramp_epochs": _strict_int(
                ccpd_schedule_raw.get("ramp_epochs", 10),
                "loss.ccpd.schedule.ramp_epochs",
                minimum=1,
            ),
        }

    peak_enabled = bool(peak is not None and peak["enabled"])
    if peak_enabled:
        if name != "waveform_spectral_peak_consistency":
            raise ValueError(
                "Enabled peak consistency requires "
                "loss.name=waveform_spectral_peak_consistency"
            )
        if version != PEAK_LOSS_IMPLEMENTATION_VERSION:
            raise ValueError(
                "Enabled peak consistency requires implementation_version=peak_v1"
            )
        if toml_enabled:
            raise ValueError("peak_v1 is TOML-free and requires TOML disabled")
        if not spectral_enabled:
            raise ValueError(
                "peak_v1 requires the established waveform + spectral baseline"
            )
    elif (
        name == "waveform_spectral_peak_consistency"
        or version == PEAK_LOSS_IMPLEMENTATION_VERSION
    ):
        raise ValueError(
            "peak_v1 name/version requires enabled loss.peak_consistency"
        )


    ccpd_enabled = bool(ccpd is not None and ccpd["enabled"])
    ccpd_name_selected = name == "waveform_long_horizon_ccpd"
    ccpd_version_selected = version == CCPD_LOSS_IMPLEMENTATION_VERSION
    if ccpd_name_selected != ccpd_version_selected:
        raise ValueError(
            "Long-horizon CCPD requires both "
            "loss.name=waveform_long_horizon_ccpd and "
            "implementation_version=ccpd_v1"
        )
    if ccpd_name_selected:
        if ccpd is None:
            raise ValueError(
                "ccpd_v1 requires a loss.ccpd mapping"
            )
        if spectral_enabled or toml_enabled or peak_enabled:
            raise ValueError(
                "ccpd_v1 isolates Wave + CCPD and requires spectral, "
                "TOML, and legacy peak consistency disabled"
            )
    elif ccpd is not None:
        raise ValueError(
            "loss.ccpd is reserved for waveform_long_horizon_ccpd/ccpd_v1"
        )

    dpd_enabled = bool(dpd is not None and dpd["enabled"])
    dpd_name_selected = name == "derivative_pulse_dual_domain"
    dpd_version_selected = version == DPD_LOSS_IMPLEMENTATION_VERSION
    if dpd_name_selected != dpd_version_selected:
        raise ValueError(
            "Derivative-pulse dual-domain loss requires both "
            "loss.name=derivative_pulse_dual_domain and "
            "implementation_version=dpd_v1"
        )
    if dpd_name_selected:
        if dpd is None or not dpd_enabled:
            raise ValueError(
                "dpd_v1 requires an enabled loss.dpd mapping"
            )
        if toml_enabled or peak_enabled or ccpd_enabled:
            raise ValueError(
                "dpd_v1 requires TOML, peak consistency, and CCPD disabled"
            )
    elif dpd is not None:
        raise ValueError(
            "loss.dpd is reserved for derivative_pulse_dual_domain/dpd_v1"
        )

    cpv_enabled = bool(cpv is not None and cpv["enabled"])
    cpv_name_selected = name == "cardiac_phase_velocity"
    cpv_version_selected = version == CPV_LOSS_IMPLEMENTATION_VERSION
    if cpv_name_selected != cpv_version_selected:
        raise ValueError(
            "Cardiac phase-velocity loss requires both "
            "loss.name=cardiac_phase_velocity and "
            "implementation_version=cpv_v1"
        )
    if cpv_name_selected:
        if cpv is None or not cpv_enabled:
            raise ValueError(
                "cpv_v1 requires an enabled loss.cpv mapping"
            )
        if toml_enabled or peak_enabled or ccpd_enabled or dpd_enabled:
            raise ValueError(
                "cpv_v1 requires TOML, peak consistency, CCPD, and DPD "
                "disabled"
            )
    elif cpv is not None:
        raise ValueError(
            "loss.cpv is reserved for cardiac_phase_velocity/cpv_v1"
        )


    if name == "baseline_z_mse" and (
        spectral_enabled
        or toml_enabled
        or peak_enabled
        or ccpd_enabled
        or dpd_enabled
        or cpv_enabled
    ):
        raise ValueError("baseline_z_mse cannot enable auxiliary loss terms")

    resolved = {
        "name": name,
        "implementation_version": version,
        "waveform": waveform,
        "spectral": spectral,
        "toml": toml,
    }
    if peak is not None:
        resolved["peak_consistency"] = peak
    if ccpd is not None:
        resolved["ccpd"] = ccpd
    if dpd is not None:
        resolved["dpd"] = dpd
    if cpv is not None:
        resolved["cpv"] = cpv
    return resolved


def loss_config_sha256(resolved_config: dict[str, Any]) -> str:
    """Return a stable fingerprint for checkpoint/resume compatibility."""

    canonical = json.dumps(
        resolved_config,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class MotionDisentangledMultiDomainLoss(nn.Module):
    """Configurable waveform, spectral, and TOML composite objective."""

    def __init__(self, resolved_config: dict[str, Any]) -> None:
        super().__init__()
        self.resolved_config = deepcopy(resolved_config)
        self.config_sha256 = loss_config_sha256(self.resolved_config)
        self.last_toml_diagnostics: dict[str, torch.Tensor] = {}
        peak_config = self.resolved_config.get("peak_consistency")
        if peak_config is not None and peak_config["enabled"]:
            peak_filter_matrix = official_filtfilt_matrix(
                peak_config["sequence_length"],
                sampling_rate_hz=peak_config["sampling_rate_hz"],
                band_hz=peak_config["band_hz"],
                filter_order=peak_config["filter_order"],
            )
        else:
            peak_filter_matrix = torch.empty(0, dtype=torch.float64)
        self.register_buffer(
            "peak_filter_matrix",
            peak_filter_matrix,
            persistent=False,
        )

        ccpd_config = self.resolved_config.get("ccpd")
        if ccpd_config is not None and ccpd_config["enabled"]:
            ccpd_filter_matrix = official_filtfilt_matrix(
                ccpd_config["sequence_length"],
                sampling_rate_hz=ccpd_config["sampling_rate_hz"],
                band_hz=ccpd_config["band_hz"],
                filter_order=ccpd_config["filter_order"],
            ).to(dtype=torch.float32)
        else:
            ccpd_filter_matrix = torch.empty(0, dtype=torch.float32)
        self.register_buffer(
            "ccpd_filter_matrix",
            ccpd_filter_matrix,
            persistent=False,
        )

        dpd_config = self.resolved_config.get("dpd")
        if dpd_config is not None and dpd_config["enabled"]:
            dpd_filter_matrix = official_filtfilt_matrix(
                dpd_config["sequence_length"],
                sampling_rate_hz=dpd_config["sampling_rate_hz"],
                band_hz=dpd_config["band_hz"],
                filter_order=dpd_config["filter_order"],
            ).to(dtype=torch.float32)
        else:
            dpd_filter_matrix = torch.empty(0, dtype=torch.float32)
        self.register_buffer(
            "dpd_filter_matrix",
            dpd_filter_matrix,
            persistent=False,
        )

        cpv_config = self.resolved_config.get("cpv")
        if cpv_config is not None and cpv_config["enabled"]:
            cpv_filter_matrix = official_filtfilt_matrix(
                cpv_config["sequence_length"],
                sampling_rate_hz=cpv_config["sampling_rate_hz"],
                band_hz=cpv_config["band_hz"],
                filter_order=cpv_config["filter_order"],
            ).to(dtype=torch.float32)
        else:
            cpv_filter_matrix = torch.empty(0, dtype=torch.float32)
        self.register_buffer(
            "cpv_filter_matrix",
            cpv_filter_matrix,
            persistent=False,
        )


    @property
    def has_auxiliary_terms(self) -> bool:
        peak_config = self.resolved_config.get("peak_consistency")
        ccpd_config = self.resolved_config.get("ccpd")
        dpd_config = self.resolved_config.get("dpd")
        cpv_config = self.resolved_config.get("cpv")
        return bool(
            self.resolved_config["spectral"]["enabled"]
            or self.resolved_config["toml"]["enabled"]
            or (peak_config is not None and peak_config["enabled"])
            or (ccpd_config is not None and ccpd_config["enabled"])
            or (dpd_config is not None and dpd_config["enabled"])
            or (cpv_config is not None and cpv_config["enabled"])
        )

    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        motion: torch.Tensor,
        sample_weights: torch.Tensor | None = None,
        toml_weight_override: float | None = None,
        peak_weight_override: float | None = None,
        ccpd_weight_override: float | None = None,
        cpv_weight_override: float | None = None,
    ) -> dict[str, torch.Tensor]:
        self.last_toml_diagnostics = {}
        waveform_config = self.resolved_config["waveform"]
        waveform = batch_global_waveform_mse(
            prediction,
            target,
            epsilon=waveform_config["epsilon"],
        )
        zero = waveform.detach().new_zeros(())

        spectral_config = self.resolved_config["spectral"]
        if spectral_config["enabled"] and spectral_config["weight"] > 0:
            spectral = physiological_band_spectral_js(
                prediction,
                target,
                sampling_rate_hz=spectral_config["sampling_rate_hz"],
                band_hz=spectral_config["band_hz"],
                n_fft=spectral_config["n_fft"],
                hann_periodic=spectral_config["hann_periodic"],
                temperature=spectral_config["temperature"],
                power_epsilon=spectral_config["power_epsilon"],
            )
            weighted_spectral = spectral * spectral_config["weight"]
        else:
            spectral = zero
            weighted_spectral = zero

        dpd_config = self.resolved_config.get("dpd")
        if dpd_config is not None and dpd_config["enabled"]:
            dpd = derivative_pulse_dual_domain_loss(
                prediction,
                target,
                filter_matrix=self.dpd_filter_matrix,
                edge_crop_samples=dpd_config["edge_crop_samples"],
                transform=dpd_config["transform"],
                epsilon=dpd_config["epsilon"],
            )
            weighted_dpd = dpd * dpd_config["weight"]
        else:
            dpd = zero
            weighted_dpd = zero

        cpv_config = self.resolved_config.get("cpv")
        cpv_weight = (
            float(cpv_config["weight"]) if cpv_config is not None else 0.0
        )
        if cpv_weight_override is not None:
            cpv_weight = _finite_nonnegative(
                cpv_weight_override,
                "cpv_weight_override",
            )
            configured_cpv_weight = (
                float(cpv_config["weight"]) if cpv_config is not None else 0.0
            )
            if cpv_weight > configured_cpv_weight:
                raise ValueError(
                    "cpv_weight_override cannot exceed configured CPV weight"
                )
        if (
            cpv_config is not None
            and cpv_config["enabled"]
            and cpv_weight > 0
        ):
            cpv = cardiac_phase_velocity_loss(
                prediction,
                target,
                filter_matrix=self.cpv_filter_matrix,
                edge_crop_samples=cpv_config["edge_crop_samples"],
                lags=cpv_config["lags"],
                transform=cpv_config["transform"],
                analytic_transform=cpv_config["analytic_transform"],
                analytic_epsilon=cpv_config["analytic_epsilon"],
                increment_epsilon=cpv_config["increment_epsilon"],
                target_scale_epsilon=cpv_config["target_scale_epsilon"],
                envelope_weighting=cpv_config["envelope_weighting"],
                envelope_weight_cap=cpv_config["envelope_weight_cap"],
            )
            weighted_cpv = cpv * cpv_weight
        else:
            cpv = zero
            weighted_cpv = zero


        toml_config = self.resolved_config["toml"]
        toml_weight = float(toml_config["weight"])
        if toml_weight_override is not None:
            toml_weight = _finite_nonnegative(
                toml_weight_override,
                "toml_weight_override",
            )
            if toml_weight > float(toml_config["weight"]):
                raise ValueError(
                    "toml_weight_override cannot exceed configured TOML weight"
                )
        if toml_config["enabled"] and toml_weight > 0:
            if (
                self.resolved_config["implementation_version"]
                == LATEST_LOSS_IMPLEMENTATION_VERSION
            ):
                try:
                    toml_result = target_orthogonal_motion_leakage(
                        prediction,
                        target,
                        motion,
                        max_lag_samples=toml_config["max_lag_samples"],
                        ridge=toml_config["ridge"],
                        standardize_motion=toml_config[
                            "standardize_motion"
                        ],
                        sample_weights=sample_weights,
                        normalization_variance_epsilon=toml_config[
                            "normalization_variance_epsilon"
                        ],
                        projection_epsilon=toml_config[
                            "projection_epsilon"
                        ],
                        center_motion=toml_config["center_motion"],
                        return_diagnostics=True,
                    )
                except TOMLNumericalError as error:
                    self.last_toml_diagnostics = error.diagnostics
                    raise
                if not isinstance(toml_result, tuple):
                    raise RuntimeError(
                        "mdm_v2 TOML did not return diagnostics"
                    )
                toml, diagnostics = toml_result
                self.last_toml_diagnostics = diagnostics
            else:
                toml = target_orthogonal_motion_leakage(
                    prediction,
                    target,
                    motion,
                    max_lag_samples=toml_config["max_lag_samples"],
                    ridge=toml_config["ridge"],
                    epsilon=toml_config["epsilon"],
                    standardize_motion=toml_config["standardize_motion"],
                    sample_weights=sample_weights,
                )
            weighted_toml = toml * toml_weight
        else:
            toml = zero
            weighted_toml = zero


        peak_config = self.resolved_config.get("peak_consistency")
        peak_weight = (
            float(peak_config["weight"]) if peak_config is not None else 0.0
        )
        if peak_weight_override is not None:
            peak_weight = _finite_nonnegative(
                peak_weight_override,
                "peak_weight_override",
            )
            configured_peak_weight = (
                float(peak_config["weight"]) if peak_config is not None else 0.0
            )
            if peak_weight > configured_peak_weight:
                raise ValueError(
                    "peak_weight_override cannot exceed configured peak weight"
                )
        if (
            peak_config is not None
            and peak_config["enabled"]
            and peak_weight > 0
        ):
            peak_terms = cross_clip_peak_consistency(
                prediction,
                target,
                group_size=peak_config["group_size"],
                filter_matrix=self.peak_filter_matrix,
                edge_crop_samples=peak_config["edge_crop_samples"],
                peak_temperature=peak_config["peak_temperature"],
                smooth_l1_beta=peak_config["smooth_l1_beta"],
                peak_map_weight=peak_config["peak_map_weight"],
                peak_count_weight=peak_config["peak_count_weight"],
                peak_timing_weight=peak_config["peak_timing_weight"],
                seam_weight=peak_config["seam_weight"],
                variance_epsilon=peak_config[
                    "normalization_variance_epsilon"
                ],
            )
            peak_consistency = peak_terms["peak_consistency"]
            peak_map = peak_terms["peak_map"]
            peak_count = peak_terms["peak_count"]
            peak_timing = peak_terms["peak_timing"]
            predicted_soft_peak_count = peak_terms["predicted_soft_peak_count"]
            target_soft_peak_count = peak_terms["target_soft_peak_count"]
            boundary_continuity = peak_terms["boundary_continuity"]
            weighted_peak_consistency = peak_consistency * peak_weight
        else:
            peak_consistency = zero
            peak_map = zero
            peak_count = zero
            peak_timing = zero
            predicted_soft_peak_count = zero
            target_soft_peak_count = zero
            boundary_continuity = zero
            weighted_peak_consistency = zero


        ccpd_config = self.resolved_config.get("ccpd")
        ccpd_weight = (
            float(ccpd_config["weight"]) if ccpd_config is not None else 0.0
        )
        if ccpd_weight_override is not None:
            ccpd_weight = _finite_nonnegative(
                ccpd_weight_override,
                "ccpd_weight_override",
            )
            configured_ccpd_weight = (
                float(ccpd_config["weight"])
                if ccpd_config is not None
                else 0.0
            )
            if ccpd_weight > configured_ccpd_weight:
                raise ValueError(
                    "ccpd_weight_override cannot exceed configured CCPD weight"
                )
        if (
            ccpd_config is not None
            and ccpd_config["enabled"]
            and ccpd_weight > 0
        ):
            ccpd_terms = long_horizon_ccpd(
                prediction,
                target,
                group_size=ccpd_config["group_size"],
                filter_matrix=self.ccpd_filter_matrix,
                edge_crop_samples=ccpd_config["edge_crop_samples"],
                event_temperature=ccpd_config["event_temperature"],
                variance_epsilon=ccpd_config[
                    "normalization_variance_epsilon"
                ],
                charbonnier_epsilon=ccpd_config["charbonnier_epsilon"],
            )
            ccpd = ccpd_terms["ccpd"]
            ccpd_forward = ccpd_terms["ccpd_forward"]
            ccpd_backward = ccpd_terms["ccpd_backward"]
            predicted_soft_event_count = ccpd_terms[
                "predicted_soft_event_count"
            ]
            target_soft_event_count = ccpd_terms[
                "target_soft_event_count"
            ]
            soft_event_count_error = ccpd_terms[
                "soft_event_count_error"
            ]
            weighted_ccpd = ccpd * ccpd_weight
        else:
            ccpd = zero
            ccpd_forward = zero
            ccpd_backward = zero
            predicted_soft_event_count = zero
            target_soft_event_count = zero
            soft_event_count_error = zero
            weighted_ccpd = zero


        total_loss = waveform
        if spectral_config["enabled"] and spectral_config["weight"] > 0:
            total_loss = total_loss + weighted_spectral
        if toml_config["enabled"] and toml_weight > 0:
            total_loss = total_loss + weighted_toml
        if (
            peak_config is not None
            and peak_config["enabled"]
            and peak_weight > 0
        ):
            total_loss = total_loss + weighted_peak_consistency
        if (
            ccpd_config is not None
            and ccpd_config["enabled"]
            and ccpd_weight > 0
        ):
            total_loss = total_loss + weighted_ccpd
        if (
            dpd_config is not None
            and dpd_config["enabled"]
            and dpd_config["weight"] > 0
        ):
            total_loss = total_loss + weighted_dpd
        if (
            cpv_config is not None
            and cpv_config["enabled"]
            and cpv_weight > 0
        ):
            total_loss = total_loss + weighted_cpv
        return {
            "total_loss": total_loss,
            "waveform": waveform,
            "spectral": spectral,
            "toml": toml,
            "weighted_spectral": weighted_spectral,
            "weighted_toml": weighted_toml,
            "peak_consistency": peak_consistency,
            "peak_map": peak_map,
            "peak_count": peak_count,
            "peak_timing": peak_timing,
            "predicted_soft_peak_count": predicted_soft_peak_count,
            "target_soft_peak_count": target_soft_peak_count,
            "boundary_continuity": boundary_continuity,
            "weighted_peak_consistency": weighted_peak_consistency,
            "ccpd": ccpd,
            "ccpd_forward": ccpd_forward,
            "ccpd_backward": ccpd_backward,
            "predicted_soft_event_count": predicted_soft_event_count,
            "target_soft_event_count": target_soft_event_count,
            "soft_event_count_error": soft_event_count_error,
            "weighted_ccpd": weighted_ccpd,
            "dpd": dpd,
            "weighted_dpd": weighted_dpd,
            "cpv": cpv,
            "weighted_cpv": weighted_cpv,
        }


def build_training_loss(
    loss_config: dict[str, Any] | None,
    *,
    expected_length: int | None = None,
    evaluation_fs: float | None = None,
) -> MotionDisentangledMultiDomainLoss:
    """Create the shared criterion used by training and validation."""

    resolved = resolve_training_loss_config(
        loss_config,
        expected_length=expected_length,
        evaluation_fs=evaluation_fs,
    )
    return MotionDisentangledMultiDomainLoss(resolved)
