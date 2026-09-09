#!/usr/bin/env python3
"""Evaluate a locally trained checkpoint with the official egoPPG HR protocol."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("MPLBACKEND", "Agg")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True, help="Your locally trained best_model_weights.pt")
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--run-id")
    args = parser.parse_args()

    import torch
    from eccvw2.data import build_fold_split, build_loader, build_participant_records, load_yaml

    config = load_yaml(args.config)
    os.environ.setdefault("BEAT_EGOPPG_ROOT", config["paths"]["source_root"])
    from eccvw2.base import build_pulseformer
    from eccvw2.evaluation import collect_model_outputs, evaluate_waveform_store
    from eccvw2.training import _resolve_run_root, _validate_output_root, write_json

    records = build_participant_records(
        data_dir=config["paths"]["preprocessed_data"],
        preprocessing_config_path=config["paths"]["preprocessing_config"],
        tasks_to_use=config["data"]["tasks_to_use"],
        chunk_length=int(config["data"]["chunk_length"]),
        downsampling=int(config["data"]["downsampling"]),
    )
    split = build_fold_split(records, args.fold, seed=int(config["experiment"]["seed"]),
                             number_of_folds=int(config["data"]["number_of_folds"]))
    loader = build_loader(
        split.test_records, batch_size=int(config["evaluation"]["batch_size"]),
        augment=False, shuffle=False, seed=int(config["experiment"]["seed"]),
        num_workers=int(config["evaluation"]["num_workers"]),
    )
    device = torch.device(args.device)
    model = build_pulseformer(config["model"], frames=int(config["data"]["chunk_length"]))
    payload = torch.load(Path(args.checkpoint).expanduser(), map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise TypeError("Checkpoint must contain a state-dictionary mapping")
    model.load_state_dict(payload.get("model_state_dict", payload), strict=True)
    model.to(device)
    store, diagnostics = collect_model_outputs(model, loader, device)
    metrics = evaluate_waveform_store(
        store, source_root=config["paths"]["source_root"], fs=int(config["evaluation"]["fs"]),
        window_seconds=int(config["evaluation"]["window_seconds"]),
        hr_method=config["evaluation"]["hr_method"],
    )
    run_id = args.run_id or datetime.now().astimezone().strftime("%Y%m%d_%H%M%S_%f")
    output = _resolve_run_root(_validate_output_root(config), run_id) / "evaluation" / f"fold_{args.fold}.json"
    if output.exists():
        raise FileExistsError(f"Evaluation output already exists: {output}")
    report = {"metrics": metrics, "diagnostics": diagnostics,
              "fold": args.fold, "seed": int(config["experiment"]["seed"]),
              "selected_epoch": payload.get("epoch")}
    write_json(output, report)
    print(json.dumps(report, indent=2))
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
