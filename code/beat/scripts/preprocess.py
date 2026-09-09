#!/usr/bin/env python3
"""Call the separately obtained egoPPG preprocessor using BEAT input settings."""

from __future__ import annotations

import argparse
import importlib
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLBACKEND", "Agg")


def _process(job: tuple[str, str, dict, str]) -> str:
    source_root, participant, config, output_dir = job
    sys.path.insert(0, source_root)
    official = importlib.import_module("preprocessing.preprocessing_egoppg")
    official.preprocess_videos(config, participant, output_dir)
    official.preprocess_timeseries(config, participant, output_dir)
    import matplotlib.pyplot as plt
    plt.close("all")
    return participant


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", default=os.environ.get("BEAT_EGOPPG_ROOT", str(ROOT / "third_party/egoPPG")))
    parser.add_argument("--raw-data", required=True, help="egoPPG-DB RawData folder obtained from its authors")
    parser.add_argument("--output-dir", default=os.environ.get("BEAT_DATA_ROOT", str(ROOT / "data/preprocessed")))
    parser.add_argument("--config", help="External official preprocessing YAML; defaults to the upstream checkout")
    parser.add_argument("--participants", nargs="+", help="Optional official participant IDs")
    parser.add_argument("--workers", type=int, default=1, help="Memory-intensive; start with one worker")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be at least one")

    import yaml
    source_root = Path(args.source_root).expanduser().resolve()
    config_path = Path(args.config).expanduser().resolve() if args.config else source_root / "configs/preprocessing/config_preprocessing_egoppg.yml"
    if not (source_root / "preprocessing/preprocessing_egoppg.py").is_file():
        raise FileNotFoundError(f"Official egoPPG preprocessing source is missing: {source_root}")
    with config_path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    raw_root = Path(args.raw_data).expanduser().resolve()
    if not raw_root.is_dir():
        raise FileNotFoundError(f"Raw data directory is missing: {raw_root}")
    output_dir = Path(args.output_dir).expanduser().resolve()
    config.update({"original_data_path": str(raw_root), "preprocessed_data_path": str(output_dir),
                   "h": 48, "w": 128, "clip_length": 128,
                   "video_types": ["DiffStandardized"], "label_type": "DiffStandardized",
                   "downsampling": 1, "upsampling": 1})
    if int(config["fs_all"]["et"]) != 30:
        raise ValueError("The released BEAT protocol requires 30 Hz eye data")
    participants = args.participants or sorted(config["task_times"])
    for participant in participants:
        if participant not in config["task_times"]:
            raise ValueError(f"Participant is not present in the external official config: {participant}")
        if any(output_dir.glob(f"{participant}_*.npy")):
            raise FileExistsError(f"Existing clips found for {participant}; use a new output folder")
    output_dir.mkdir(parents=True, exist_ok=True)
    jobs = [(str(source_root), participant, config, str(output_dir)) for participant in participants]
    if args.workers == 1:
        for job in jobs:
            print(f"Completed participant {_process(job)}", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            for participant in executor.map(_process, jobs):
                print(f"Completed participant {participant}", flush=True)
    print(f"BEAT_DATA_ROOT={output_dir}")


if __name__ == "__main__":
    main()
