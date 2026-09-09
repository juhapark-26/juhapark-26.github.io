# Third-party dependencies and attribution

## egoPPG / PulseFormer

- Repository: https://github.com/eth-siplab/egoPPG
- Required revision: `5bb8437a13bfa4ee8ced89126dec66db4c40f4c0`
- External model: `ml/models/PulseFormer.py`
- Model SHA256: `c0e1913cfa98ad94acb1c1be87ef50d3499487f31a53b74f19218314e1a2ddd1`
- Authors: Björn Braun, Rayan Armani, Manuel Meier, Max Moebus and Christian Holz.

The official backbone, preprocessing and evaluation source bodies are not bundled.
They are loaded from a checkout independently obtained by the user. At the pinned
revision, no repository-root license is provided and the model header contains
both a research/non-commercial statement and an MIT label. This release does not
resolve that ambiguity or purport to grant rights to that external source.

PulseFormer's source credits PhysNet; egoPPG also credits the rPPG-Toolbox for its
code structure. Consult those projects and retain their notices when separately
using their code. No blanket BEAT license should be applied to these dependencies.

## Other software and initialization

PyTorch, torchvision, NumPy, SciPy, scikit-learn, PyYAML, tqdm, NeuroKit2,
Matplotlib, pandas and OpenCV are installed separately and retain their own terms.
Torchvision ResNet18 ImageNet1K V1 initialization is obtained independently by the
user; its parameter tensors are not distributed in this release.

## Data

egoPPG-DB is obtained directly from its authors under their data transfer/use
agreement. This release contains no recordings, biosignals or participant-level
metadata and does not grant dataset access or redistribution rights.

## Website

The project page is adapted from the Nerfies template under CC BY-SA 4.0, as noted
on the page. That template attribution is separate from this source-code directory.
