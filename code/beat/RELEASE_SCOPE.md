# Source-only release boundary

This directory contains the BEAT extension, reproducible settings, command-line
integration and synthetic tests. It does not contain experiment artifacts.

Included:

- Shared location-wise bottleneck DW-TCN and MPPL numerical implementation.
- Waveform and spectral controls, final-model configuration and ablation settings.
- Training and official-evaluation integration using an external egoPPG checkout.
- Source-only tests, documentation and packaging metadata.

Not included:

- BEAT/PulseFormer checkpoints, learned parameters, optimizer/scheduler state.
- ImageNet weights; the user obtains public torchvision weights independently.
- Raw/preprocessed recordings, participant metadata, prediction arrays or logs.
- Research environments, credentials, private paths, unpublished manuscripts.
- The source bodies of the upstream egoPPG backbone or preprocessing/evaluator.

`cpv` and `cardiac_phase_velocity` are legacy identifiers for the paper's MPPL.
The implementation retains some unused compatibility helpers so the original
training/validation bookkeeping is preserved. TOML, DPD and cross-clip losses are
not part of the BEAT configuration and are not advertised as paper contributions.

Hyperparameters (learning rate, epochs, loss coefficients, etc.) are intentionally
included: they are needed to reproduce the method and are not trained weights.
Training will create private local outputs; keep those ignored directories out of
subsequent commits. The ignore file is defense in depth, not a substitute for
reviewing the exact files being published.

This release changes dependency loading and machine-specific paths, not the
paper's temporal adapter or MPPL equations. It does not claim that a new full
training run has been completed as part of release validation.
