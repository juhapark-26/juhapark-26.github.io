# BEAT

**Bottleneck-Enhanced Adaptive Temporal Learning for Heart Rate Estimation from
Egocentric NIR Eye Videos** — Juha Park and Sang Jun Lee.

[Project page](https://juhapark-26.github.io/projects/beat/) ·
[Implementation contract](METHOD.md) · [Release scope](RELEASE_SCOPE.md)

This is a **source-only** release of the paper's location-wise bottleneck DW-TCN
and Multi-Lag Pulse-Phase Progression Loss (MPPL), with end-to-end training and
official egoPPG evaluation integration. No trained weights, checkpoints, optimizer
state, recordings, participant metadata or private experiment logs are included.

The PulseFormer backbone and MITA are retained. BEAT adds an 18,560-parameter
adapter and MPPL; it does not replace a Transformer or MITA. In source/configs,
`cpv` is the legacy identifier for **MPPL**, not an additional objective.

## Quick start

Use Python 3.11 and an editable installation from this directory. The verified
environment uses PyTorch 2.11.0, torchvision 0.26.0 and CUDA 12.8. Choose the
appropriate PyTorch wheel for your own GPU/driver; the example below uses CUDA 12.8.
CPU is sufficient for synthetic tests, but full training requires a suitable GPU.

```bash
git clone https://github.com/juhapark-26/juhapark-26.github.io.git
cd juhapark-26.github.io/code/beat
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -e '.[test]'
```

### 1. Obtain the official dependency separately

Review the [official egoPPG repository](https://github.com/eth-siplab/egoPPG) and
its applicable source/data terms. Its source bodies are **not** redistributed here.
Keep a clean checkout of the pinned revision (the model loader checks its hash):

```bash
mkdir -p third_party
git clone https://github.com/eth-siplab/egoPPG.git third_party/egoPPG
git -C third_party/egoPPG checkout 5bb8437a13bfa4ee8ced89126dec66db4c40f4c0
export BEAT_EGOPPG_ROOT="$PWD/third_party/egoPPG"
```

Do not point this variable at a locally modified PulseFormer implementation.
Additional official preprocessing dependencies may be required for raw recordings;
follow the upstream installation instructions for those tools.

### 2. Prepare data

Obtain egoPPG-DB from its authors under the dataset's access agreement. No dataset
download, participant-level records or dataset archive is provided by this release.
Already preprocessed official clips can be used directly:

```bash
# Replace the placeholder with the directory containing the preprocessed clips.
export BEAT_DATA_ROOT="/path/to/preprocessed/clips"
```

Otherwise use the wrapper below. It reads task metadata from the external official
configuration and invokes its preprocessing with the paper's clip settings. It
does not alter the upstream checkout or publish metadata:

```bash
python scripts/preprocess.py --raw-data /path/to/egoPPG-DB/RawData --output-dir data/preprocessed
```

After preprocessing, set `BEAT_DATA_ROOT` to the generated clip directory, not the
raw-data directory. Video uses differential standardization, clip length 128,
resolution 48×128 and 30 Hz; tasks/quality exclusions come from the pinned upstream
configuration. See [METHOD.md](METHOD.md) for the exact supervision convention.

### 3. Obtain public initialization weights locally

The reference initialization is fresh BEAT/PulseFormer training with public
ResNet18 ImageNet1K V1 initialization. It does not load a trained BEAT model.
Download those weights independently into torchvision's normal cache:

```bash
python -c 'from torchvision.models import ResNet18_Weights; ResNet18_Weights.IMAGENET1K_V1.get_state_dict(progress=True, check_hash=True)'
```

The training builder checks that this cache exists. No weights are stored in this
Git repository, and core tests use random initialization without downloading any.

### 4. Train end to end

Run from `code/beat`. The launcher automatically creates a timestamped run folder
under ignored `outputs/`, logs loss components, selects by validation waveform
loss, and evaluates the selected checkpoint:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/train.py --config configs/beat.yaml --fold 0 --device cuda:0
```

`CUDA_VISIBLE_DEVICES` selects the physical GPU; the worker uses its visible
`cuda:0`. The reference configuration uses 100 epochs, batch 4, Adam 9e-4,
OneCycleLR 9e-4 and FP32. The seed is in `experiment.seed` / `initialization.seed`;
keep them synchronized. `--fold` chooses a participant-disjoint partition.
Select these explicitly for your intended run rather than combining results from
different protocols. The paper runner enforces 100 epochs, batch 4 and the defined
five-partition protocol; changing those YAML settings alone is intentionally rejected.

To run all defined partitions sequentially with one seed configuration:

```bash
for fold in 0 1 2 3 4; do
  CUDA_VISIBLE_DEVICES=0 python scripts/train.py --config configs/beat.yaml --fold "$fold" --device cuda:0 || break
done
```

The launcher also accepts `--run-id` and `--resume`; use the same run ID/config
when resuming your own trusted checkpoint. Full training checkpoints can contain
Python/NumPy RNG state: never load an untrusted external checkpoint.
`BEAT_OUTPUT_ROOT` may select another ignored directory **inside this worktree**.

### 5. Evaluate locally trained weights

The training command exports `best_model_weights.pt`, a plain state dictionary.
The separate evaluator loads that local artifact with `weights_only=True`:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/evaluate.py \
  --config configs/beat.yaml --fold 0 --device cuda:0 \
  --checkpoint /path/to/your/best_model_weights.pt
```

Prediction and GT use the same official post-processing and peak-interval HR
estimator on 60 s windows. The output includes HR MAE, RMSE, MAPE and Pearson r.
It is not an FFT peak-bin classifier. The local weights and evaluation outputs
remain private and are ignored by Git.

## Ablations

All component controls use the same training/evaluation pipeline. In this table,
the adapter is the bottleneck DW-TCN, and spectral loss has weight 0.1 when enabled.

| Configuration | Adapter | Waveform | Spectral | MPPL |
|---|:---:|:---:|:---:|:---:|
| `configs/backbone_control.yaml` | — | ✓ | ✓ | — |
| `configs/adapter_only.yaml` | ✓ | ✓ | ✓ | — |
| `configs/mppl_only.yaml` | — | ✓ | ✓ | ✓ |
| `configs/beat.yaml` | ✓ | ✓ | ✓ | ✓ |
| `configs/waveform_only.yaml` | ✓ | ✓ | — | — |
| `configs/waveform_mppl.yaml` | ✓ | ✓ | — | ✓ |

`dilation_1111.yaml`, `dilation_1222.yaml` and `dilation_1244.yaml` vary the four
dilations with the final loss. The final model uses `[1,2,4,8]`.
Change only `--config` in the training command to run a control.
“Backbone control” is not the published PulseFormer result: it shares our
waveform-plus-spectral training objective. The project page distinguishes published
reference numbers from locally controlled comparisons. No new full-training results
are claimed by this code release.

## Tests (no dataset, no checkpoints, no GPU)

```bash
CUDA_VISIBLE_DEVICES='' python -m unittest discover -s tests -v
```

The core adapter/loss tests run without egoPPG installed. The external-model
integration test runs when `BEAT_EGOPPG_ROOT` points to the pinned checkout; it
performs one synthetic forward/backward/optimizer step, with no weight download.
With the full pinned checkout, the suite also verifies the official HR evaluator
on synthetic signals. All 15 tests passed during release validation.

The release was also checked against the frozen reference implementation for
bit-identical initialized states and forward outputs, including a nonzero adapter
projection. These tests establish implementation fidelity, not full-training
performance or cross-hardware bitwise reproducibility.

## Files and terms

`src/eccvw2/temporal.py` contains the adapter; `losses.py` contains MPPL and its
waveform/spectral components; `pulseformer.py` attaches the adapter to the external
backbone. `data.py`, `e2e_training.py` and `evaluation.py` contain the corresponding
data, optimization and evaluation integration.

See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) and [LICENSE.md](LICENSE.md).
The website template's license does not license the research code or datasets.
This is research software, not a validated medical device.

## Citation

```bibtex
@inproceedings{park2026beat,
  title = {BEAT: Bottleneck-Enhanced Adaptive Temporal Learning for Heart Rate Estimation from Egocentric NIR Eye Videos},
  author = {Park, Juha and Lee, Sang Jun},
  booktitle = {ECCV Workshop on Wearable AI},
  year = {2026}
}
```

Please also cite [egoPPG](https://github.com/eth-siplab/egoPPG) when using its
backbone, evaluation protocol or dataset.
