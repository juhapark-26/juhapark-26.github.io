# Implementation contract

The paper calls the auxiliary objective **Multi-Lag Pulse-Phase Progression Loss
(MPPL)**. The frozen implementation calls it `cpv` /
`cardiac_phase_velocity_loss`; these are the same objective, not two losses.

## Architecture

The external PulseFormer backbone, spatial attention, IMU-conditioned MITA,
temporal decoder and waveform head remain intact. The model is trained end to
end. BEAT does **not** replace MITA or a Transformer block.

At the encoder bottleneck, a tensor `[B,64,32,3,8]` is rearranged to
`[B*24,64,32]`. One shared DW-TCN processes each spatial trajectory, then the
tensor is restored before entering the unchanged decoder. The public integration
uses a forward hook at `ConvBlock9`; no upstream backbone body is redistributed.

Each of four residual stages is:

```
x + layer_scale * Dropout(SiLU(DWConv1d(GroupNorm(x))))
```

The stages use kernel size 3, dilations `[1,2,4,8]`, depthwise groups=64, biases,
symmetric **zero** padding, one-group GroupNorm, dropout 0.1, and channel-wise
layer scales initialized to 0.001. Non-causal GroupNorm aggregates over channels
and time within one spatial trajectory. The 31-token receptive field is the
nominal convolutional receptive field, not a claim of strict locality for that
normalization operation.

After the stages: GroupNorm → pointwise Conv `64→128` → split value/gate →
`value * sigmoid(gate)` → pointwise Conv `64→64`. A further zero-initialized
pointwise output projection and the outer residual make the complete adapter an
exact identity at initialization. The adapter adds **18,560** parameters.

## Supervision

```
L = L_wave + 0.1 * L_spec + lambda(epoch,step) * L_MPPL
```

- `L_wave`: MSE after global `B*T` z-normalization, using `torch.std` with its
  default correction=1 and `std.clamp_min(1e-8)`. This preserves the base trainer's
  convention; it is not a newly invented waveform loss or sample-wise whitening.
- `L_spec`: independently mean-center the differential prediction and target;
  periodic Hann window; rFFT length 128 at 30 Hz; power `abs(FFT)^2`; logarithm with
  epsilon 1e-8; temperature 1 softmax over the included bins; Jensen–Shannon
  divergence, averaged over the batch. The requested inclusive 0.7–2.8 Hz band
  selects **bins 3–11, 0.703125–2.578125 Hz**. No zero padding or exact-HR-bin
  classification is used.
- MPPL: reconstruct each 128-sample pulse as
  `-linear_detrend(cumsum(differential_signal))`, identically for prediction and
  target. Apply the linear operator equivalent to SciPy's fourth-order
  Butterworth band-pass 0.7–2.8 Hz with forward/backward `filtfilt` and default odd
  extension. The filter's coefficient arrays have length 9, so default padlen is 27.
  Compute the FFT Hilbert analytic signal at the full length 128, without extra
  FFT padding, then remove **27 samples at each end** (74 remain). Cropping is an
  explicit edge-handling choice, not a guarantee that all filter transients vanish.
- Both analytic signals are scaled by the detached target RMS, with 1e-8 inside
  the square root. Analytic phasors and their lagged complex products are each
  normalized using a square-root squared-magnitude stabilizer of 1e-2. Lags are
  `[1,2,4]`. The error is half the squared complex chord distance.
- Target-envelope weights are the geometric mean of lag endpoints, mean-normalized
  **per sample and lag**, then capped at 2. The weighted temporal mean is computed
  with an epsilon 1e-8 denominator, then averaged across samples and lags.

The final MPPL coefficient ramps from 0 to 0.5 after 5 warmup epochs over 10 epochs;
the implementation schedules it at optimizer-step granularity. Validation uses
the configured full coefficient for its auxiliary report, but **waveform loss
alone selects the checkpoint**. MPPL is training-only; inference requires no GT.

The stable MPPL has finite forward/backward behavior for constant and near-constant
outputs. At an exactly constant prediction its own gradient may be zero: the
waveform term provides the restoration signal. The release does not claim MPPL
alone prevents collapsed predictions.

## Data, training and evaluation

The external official preprocessing supplies synchronized NIR video, motion and
differential standardized PPG. Inputs are 128 frames at 30 Hz, one channel, 48×128.
Tasks are video, office, kitchen, dancing, bike and walking. Task boundaries and
quality exclusions are read from the external official configuration, not
republished as participant metadata here.

All model parameters are trainable; 100 epochs; batch 4; FP32 (AMP off); Adam with
learning rate 9e-4 and default betas/epsilon; OneCycleLR maximum 9e-4 stepped each
batch; validation batch 4; test batch 1. Fresh initialization uses publicly
obtainable torchvision ResNet18 ImageNet1K V1 initialization, not a pretrained
BEAT or full PulseFormer checkpoint. A fixed canonical backbone is copied to
each architecture variant to retain the reference initialization procedure.

For HR metrics the saved clip outputs are ordered/concatenated by participant.
The external official evaluator applies identical inverse-difference and filtering
to prediction and GT and estimates HR by peak intervals in nonoverlapping 60 s
windows; incomplete tails are discarded. MAE/RMSE/MAPE/Pearson r are HR metrics,
not the training waveform or spectral loss. Preserve this distinction when
reporting results, and do not treat a retained reference snapshot as a new
multi-run average.
