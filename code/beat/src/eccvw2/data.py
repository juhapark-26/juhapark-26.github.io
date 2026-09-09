"""egoPPG data manifests and loaders without modifying the source repository."""

from __future__ import annotations

import os
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

import numpy as np
import torch
import yaml
from sklearn.model_selection import KFold, train_test_split
from scipy.signal import butter, detrend, filtfilt, find_peaks
from torch.utils.data import DataLoader, Dataset, Sampler
from torchvision.transforms import v2


PARTICIPANTS = tuple(f"{index:03d}" for index in range(1, 26))
CHUNK_PATTERN = re.compile(r"_input_et(?P<chunk>\d+)\.npy$")


@dataclass(frozen=True)
class ClipRecord:
    video_path: Path
    imu_path: Path
    label_path: Path
    participant: str
    chunk_index: int
    # Preserve the original recording segment (for example ``walking_1``).
    # Cross-clip batches must never bridge two tasks just because their global
    # chunk indices happen to be consecutive.
    task: str


@dataclass(frozen=True)
class FoldSplit:
    fold: int
    train_participants: tuple[str, ...]
    valid_participants: tuple[str, ...]
    test_participants: tuple[str, ...]
    train_records: tuple[ClipRecord, ...]
    valid_records: tuple[ClipRecord, ...]
    test_records: tuple[ClipRecord, ...]

    def summary(self) -> dict:
        return {
            "fold": self.fold,
            "train_participants": list(self.train_participants),
            "valid_participants": list(self.valid_participants),
            "test_participants": list(self.test_participants),
            "train_clips": len(self.train_records),
            "valid_clips": len(self.valid_records),
            "test_clips": len(self.test_records),
        }


def load_yaml(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise TypeError("Configuration must contain a YAML mapping")
    if "paths" not in config:
        return config
    root = Path(__file__).resolve().parents[2]

    def resolve(value: str | Path) -> str:
        candidate = Path(os.path.expandvars(str(value))).expanduser()
        return str((candidate if candidate.is_absolute() else root / candidate).resolve())

    paths = config["paths"]
    paths["source_root"] = resolve(os.environ.get(
        "BEAT_EGOPPG_ROOT", paths.get("source_root") or "third_party/egoPPG"
    ))
    paths["preprocessed_data"] = resolve(os.environ.get(
        "BEAT_DATA_ROOT", paths.get("preprocessed_data") or "data/preprocessed"
    ))
    paths["output_root"] = resolve(os.environ.get(
        "BEAT_OUTPUT_ROOT", paths.get("output_root") or "outputs"
    ))
    paths["preprocessing_config"] = resolve(
        os.environ.get("BEAT_PREPROCESSING_CONFIG")
        or paths.get("preprocessing_config")
        or Path(paths["source_root"]) / "configs/preprocessing/config_preprocessing_egoppg.yml"
    )
    initialization = config.get("initialization", {})
    if initialization.get("imagenet_cache_path"):
        initialization["imagenet_cache_path"] = resolve(initialization["imagenet_cache_path"])
    return config


def _chunk_index(path: Path) -> int:
    match = CHUNK_PATTERN.search(path.name)
    if match is None:
        raise ValueError(f"Unexpected eye clip filename: {path.name}")
    return int(match.group("chunk"))


def _task_chunk_metadata(
    participant: str,
    preprocessing_config: dict,
    tasks_to_use: Sequence[str],
    chunk_length: int,
    downsampling: int,
) -> list[tuple[str, bool]]:
    metadata: list[tuple[str, bool]] = []
    exclusion_list = preprocessing_config["exclusion_list"]
    for raw_task, (start, end) in preprocessing_config["task_times"][participant].items():
        task = "walking" if raw_task in {"walking_1", "walking_2", "walking_3"} else raw_task
        number_of_chunks = (((end - start) // downsampling + 1) // chunk_length)
        include_task = task in tasks_to_use and participant not in exclusion_list[task]
        metadata.extend([(raw_task, include_task)] * number_of_chunks)
    return metadata


def _task_keep_mask(
    participant: str,
    preprocessing_config: dict,
    tasks_to_use: Sequence[str],
    chunk_length: int,
    downsampling: int,
) -> list[bool]:
    """Compatibility helper returning only the task inclusion flags."""

    return [
        keep
        for _, keep in _task_chunk_metadata(
            participant, preprocessing_config, tasks_to_use, chunk_length, downsampling
        )
    ]


def build_participant_records(
    data_dir: str | Path,
    preprocessing_config_path: str | Path,
    participants: Sequence[str] = PARTICIPANTS,
    tasks_to_use: Sequence[str] = ("video", "office", "kitchen", "dancing", "bike", "walking"),
    chunk_length: int = 128,
    downsampling: int = 1,
) -> dict[str, tuple[ClipRecord, ...]]:
    data_dir = Path(data_dir)
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Preprocessed data directory does not exist: {data_dir}")
    preprocessing_config = load_yaml(preprocessing_config_path)

    records_by_participant: dict[str, tuple[ClipRecord, ...]] = {}
    for participant in participants:
        videos = sorted(data_dir.glob(f"{participant}_input_et*.npy"), key=_chunk_index)
        if not videos:
            raise FileNotFoundError(f"No eye clips found for participant {participant}")
        task_metadata = _task_chunk_metadata(
            participant,
            preprocessing_config,
            tasks_to_use,
            chunk_length,
            downsampling,
        )
        if len(task_metadata) != len(videos):
            raise RuntimeError(
                f"Task metadata/file mismatch for {participant}: "
                f"mask={len(task_metadata)}, clips={len(videos)}"
            )

        participant_records: list[ClipRecord] = []
        for video_path, (raw_task, keep) in zip(videos, task_metadata):
            if not keep:
                continue
            chunk_index = _chunk_index(video_path)
            imu_path = Path(str(video_path).replace("_input_et", "_input_imu_right"))
            label_path = Path(str(video_path).replace("_input_et", "_label_ppg_nose"))
            missing = [path for path in (imu_path, label_path) if not path.is_file()]
            if missing:
                raise FileNotFoundError(f"Missing companion files for {video_path}: {missing}")
            participant_records.append(
                ClipRecord(
                    video_path=video_path,
                    imu_path=imu_path,
                    label_path=label_path,
                    participant=participant,
                    chunk_index=chunk_index,
                    task=raw_task,
                )
            )
        records_by_participant[participant] = tuple(participant_records)
    return records_by_participant


def _flatten_records(
    records_by_participant: dict[str, tuple[ClipRecord, ...]],
    participants: Iterable[str],
) -> tuple[ClipRecord, ...]:
    return tuple(
        record
        for participant in participants
        for record in records_by_participant[participant]
    )


def build_fold_split(
    records_by_participant: dict[str, tuple[ClipRecord, ...]],
    fold: int,
    seed: int = 0,
    number_of_folds: int = 5,
) -> FoldSplit:
    if fold < 0 or fold >= number_of_folds:
        raise ValueError(f"fold must be in [0, {number_of_folds}), got {fold}")
    participants = np.asarray(PARTICIPANTS)
    splitter = KFold(n_splits=number_of_folds)
    selected: tuple[np.ndarray, np.ndarray] | None = None
    for fold_index, indices in enumerate(splitter.split(participants)):
        if fold_index == fold:
            selected = indices
            break
    if selected is None:
        raise RuntimeError(f"Could not build fold {fold}")

    train_valid_indices, test_indices = selected
    train_indices, valid_indices = train_test_split(
        train_valid_indices,
        random_state=seed,
        test_size=2,
    )
    train_indices.sort()
    valid_indices.sort()
    test_indices.sort()

    train_participants = tuple(participants[train_indices].tolist())
    valid_participants = tuple(participants[valid_indices].tolist())
    test_participants = tuple(participants[test_indices].tolist())
    return FoldSplit(
        fold=fold,
        train_participants=train_participants,
        valid_participants=valid_participants,
        test_participants=test_participants,
        train_records=_flatten_records(records_by_participant, train_participants),
        valid_records=_flatten_records(records_by_participant, valid_participants),
        test_records=_flatten_records(records_by_participant, test_participants),
    )


class EgoPPGClipDataset(Dataset):
    def __init__(
        self,
        records: Sequence[ClipRecord],
        augment: bool,
        height: int = 48,
        width: int = 128,
    ) -> None:
        self.records = tuple(records)
        self.augment = bool(augment)
        if self.augment:
            self.transform = v2.Compose(
                [
                    v2.RandomHorizontalFlip(p=0.5),
                    v2.RandomVerticalFlip(p=0.5),
                    v2.RandomCrop(size=(height // 2, width)),
                    v2.Pad((0, height // 4)),
                    v2.RandomRotation(degrees=(-20, 20)),
                    v2.ToDtype(torch.float32, scale=False),
                ]
            )
        else:
            self.transform = v2.ToDtype(torch.float32, scale=False)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, str, int]:
        record = self.records[index]
        video = torch.from_numpy(np.load(record.video_path)).float()
        video = self.transform(video)
        imu = torch.from_numpy(np.asarray(np.load(record.imu_path), dtype=np.float32))
        label = torch.from_numpy(np.asarray(np.load(record.label_path), dtype=np.float32))
        return video, imu, label, record.participant, record.chunk_index


def seed_worker(worker_id: int) -> None:
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def build_loader(
    records: Sequence[ClipRecord],
    batch_size: int,
    augment: bool,
    shuffle: bool,
    seed: int,
    num_workers: int,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    dataset = EgoPPGClipDataset(records=records, augment=augment)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        worker_init_fn=seed_worker,
        generator=generator,
    )


@dataclass(frozen=True)
class ConsecutiveClipGroup:
    """Indices of one participant/task-contiguous long-context training unit."""

    indices: tuple[int, ...]
    participant: str
    task: str
    first_chunk_index: int
    estimated_hr_bpm: float
    hr_bin: int


def _split_consecutive_runs(
    indexed_records: Sequence[tuple[int, ClipRecord]],
) -> list[list[tuple[int, ClipRecord]]]:
    runs: list[list[tuple[int, ClipRecord]]] = []
    current: list[tuple[int, ClipRecord]] = []
    previous: ClipRecord | None = None
    for indexed_record in indexed_records:
        record = indexed_record[1]
        continues = (
            previous is not None
            and record.participant == previous.participant
            and record.task == previous.task
            and record.chunk_index == previous.chunk_index + 1
        )
        if current and not continues:
            runs.append(current)
            current = []
        current.append(indexed_record)
        previous = record
    if current:
        runs.append(current)
    return runs


def _estimate_group_hr_bpm(
    records: Sequence[ClipRecord],
    sampling_rate_hz: float,
    band_hz: tuple[float, float],
) -> float:
    """Compute the official label-only peak HR for train stratification.

    This exactly follows cumsum, detrend, inversion, Butterworth bandpass,
    filtfilt, and unconstrained peak spacing from the official evaluation.
    It is never used as a model target or reported evaluation result.
    """

    derivative = np.concatenate(
        [
            np.asarray(np.load(record.label_path), dtype=np.float64).reshape(-1)
            for record in records
        ]
    )
    if derivative.size < 4 or not np.isfinite(derivative).all():
        raise ValueError("Cross-clip target labels must be finite and non-empty")
    waveform = detrend(np.cumsum(derivative))
    waveform = np.max(waveform) - waveform
    low, high = band_hz
    coefficients_b, coefficients_a = butter(
        4,
        [
            low / float(sampling_rate_hz) * 2.0,
            high / float(sampling_rate_hz) * 2.0,
        ],
        btype="bandpass",
    )
    filtered = filtfilt(coefficients_b, coefficients_a, waveform)
    peaks, _ = find_peaks(filtered)
    if peaks.size < 2:
        raise ValueError(
            "Official peak HR requires at least two peaks in each group label"
        )
    hr_bpm = 60.0 * float(sampling_rate_hz) / float(np.mean(np.diff(peaks)))
    if not np.isfinite(hr_bpm):
        raise FloatingPointError("Official group-label peak HR is non-finite")
    return hr_bpm


def build_consecutive_clip_groups(
    records: Sequence[ClipRecord],
    *,
    group_size: int,
    group_stride: int,
    sampling_rate_hz: float,
    band_hz: Sequence[float],
    hr_bin_edges_bpm: Sequence[float],
) -> tuple[tuple[ConsecutiveClipGroup, ...], dict[str, Any]]:
    """Build fixed-size groups without crossing participant, task, or gaps."""

    group_size = int(group_size)
    group_stride = int(group_stride)
    if group_size < 2:
        raise ValueError("cross_clip.group_size must be at least two")
    if group_stride < 1:
        raise ValueError("cross_clip.group_stride must be positive")
    if len(band_hz) != 2:
        raise ValueError("cross_clip.band_hz must contain [low, high]")
    low, high = (float(band_hz[0]), float(band_hz[1]))
    if not (0.0 < low < high <= float(sampling_rate_hz) / 2.0):
        raise ValueError("cross_clip.band_hz must lie inside Nyquist")
    edges = tuple(float(value) for value in hr_bin_edges_bpm)
    if tuple(sorted(edges)) != edges or len(set(edges)) != len(edges):
        raise ValueError("cross_clip HR bin edges must be strictly increasing")

    ordered = sorted(
        enumerate(records),
        key=lambda item: (
            item[1].participant,
            item[1].task,
            item[1].chunk_index,
        ),
    )
    groups: list[ConsecutiveClipGroup] = []
    covered_indices: set[int] = set()
    run_count = 0
    end_anchored_group_count = 0
    for run in _split_consecutive_runs(ordered):
        run_count += 1
        starts = list(
            range(0, len(run) - group_size + 1, group_stride)
        )
        if starts:
            end_start = len(run) - group_size
            if starts[-1] != end_start:
                starts.append(end_start)
                end_anchored_group_count += 1
        for start in starts:
            members = run[start : start + group_size]
            member_indices = tuple(index for index, _ in members)
            member_records = tuple(record for _, record in members)
            estimated_hr = _estimate_group_hr_bpm(
                member_records,
                sampling_rate_hz=float(sampling_rate_hz),
                band_hz=(low, high),
            )
            hr_bin = int(np.digitize(estimated_hr, edges, right=False))
            groups.append(
                ConsecutiveClipGroup(
                    indices=member_indices,
                    participant=member_records[0].participant,
                    task=member_records[0].task,
                    first_chunk_index=member_records[0].chunk_index,
                    estimated_hr_bpm=estimated_hr,
                    hr_bin=hr_bin,
                )
            )
            covered_indices.update(member_indices)
    if not groups:
        raise ValueError("No valid consecutive cross-clip groups could be built")

    bin_count = len(edges) + 1
    counts = np.bincount(
        [group.hr_bin for group in groups], minlength=bin_count
    )
    hrs = np.asarray(
        [group.estimated_hr_bpm for group in groups], dtype=np.float64
    )
    audit = {
        "group_size": group_size,
        "group_stride": group_stride,
        "number_of_groups": len(groups),
        "number_of_candidate_groups": len(groups),
        "end_anchored_group_count": end_anchored_group_count,
        "number_of_runs": run_count,
        "input_clip_count": len(records),
        "unique_grouped_clip_count": len(covered_indices),
        "ungrouped_clip_count": len(records) - len(covered_indices),
        "hr_bin_edges_bpm": list(edges),
        "hr_bin_group_counts": counts.tolist(),
        "estimated_hr_bpm": {
            "minimum": float(hrs.min()),
            "median": float(np.median(hrs)),
            "maximum": float(hrs.max()),
        },
        "participant_task_gap_safe": True,
        "target_source": "training_split_ground_truth_only",
        "hr_estimator": "official_peak_detection_label_only",
    }
    return tuple(groups), audit


class EgoPPGConsecutiveGroupDataset(Dataset):
    """Load one consecutive group and share one spatial augmentation draw."""

    def __init__(
        self,
        records: Sequence[ClipRecord],
        groups: Sequence[ConsecutiveClipGroup],
        augment: bool,
    ) -> None:
        self.records = tuple(records)
        self.groups = tuple(groups)
        # torchvision v2 samples random geometry once per call, so applying
        # this transform to [group,C,T,H,W] preserves cross-clip consistency.
        self.transform = EgoPPGClipDataset(
            records=records, augment=augment
        ).transform

    def __len__(self) -> int:
        return len(self.groups)

    def __getitem__(self, index: int):
        group = self.groups[index]
        member_records = [self.records[item] for item in group.indices]
        video = torch.stack(
            [
                torch.from_numpy(np.load(record.video_path)).float()
                for record in member_records
            ]
        )
        video = self.transform(video)
        imu = torch.stack(
            [
                torch.from_numpy(
                    np.asarray(np.load(record.imu_path), dtype=np.float32)
                )
                for record in member_records
            ]
        )
        label = torch.stack(
            [
                torch.from_numpy(
                    np.asarray(np.load(record.label_path), dtype=np.float32)
                )
                for record in member_records
            ]
        )
        participants = tuple(record.participant for record in member_records)
        chunk_indices = torch.tensor(
            [record.chunk_index for record in member_records],
            dtype=torch.int64,
        )
        return video, imu, label, participants, chunk_indices


def _unwrap_single_group(batch):
    if len(batch) != 1:
        raise RuntimeError("Cross-clip loader requires one group per batch")
    return batch[0]


class TailBalancedGroupSampler(Sampler[int]):
    """Sample consecutive groups with capped inverse-frequency HR weights."""

    def __init__(
        self,
        groups: Sequence[ConsecutiveClipGroup],
        *,
        generator: torch.Generator,
        enabled: bool,
        max_weight_ratio: float,
        num_samples: int | None = None,
    ) -> None:
        if not groups:
            raise ValueError("groups must not be empty")
        if not np.isfinite(max_weight_ratio) or max_weight_ratio < 1.0:
            raise ValueError(
                "max_weight_ratio must be finite and at least one"
            )
        self.groups = tuple(groups)
        self.generator = generator
        self.enabled = bool(enabled)
        self.num_samples = (
            len(self.groups) if num_samples is None else int(num_samples)
        )
        if self.num_samples < 1:
            raise ValueError("num_samples must be positive")
        bins = np.asarray([group.hr_bin for group in groups], dtype=np.int64)
        counts = np.bincount(bins)
        if self.enabled:
            inverse = np.asarray(
                [1.0 / counts[bin_index] for bin_index in bins],
                dtype=np.float64,
            )
            minimum = float(inverse.min())
            weights = np.minimum(
                inverse, minimum * float(max_weight_ratio)
            )
        else:
            weights = np.ones(len(groups), dtype=np.float64)
        self.weights = torch.as_tensor(weights, dtype=torch.float64)

    def __len__(self) -> int:
        return self.num_samples

    def __iter__(self) -> Iterator[int]:
        if self.enabled:
            selected = torch.multinomial(
                self.weights,
                num_samples=self.num_samples,
                replacement=True,
                generator=self.generator,
            ).tolist()
        elif self.num_samples <= len(self.groups):
            selected = torch.randperm(
                len(self.groups), generator=self.generator
            )[: self.num_samples].tolist()
        else:
            selected = torch.multinomial(
                self.weights,
                num_samples=self.num_samples,
                replacement=True,
                generator=self.generator,
            ).tolist()
        for group_index in selected:
            yield int(group_index)


def build_cross_clip_loader(
    records: Sequence[ClipRecord],
    *,
    batch_size: int,
    augment: bool,
    seed: int,
    num_workers: int,
    cross_clip_config: dict[str, Any],
    sampling_rate_hz: float,
) -> tuple[DataLoader, dict[str, Any]]:
    """Return a train loader whose every batch is one valid clip sequence."""

    if not isinstance(cross_clip_config, dict):
        raise TypeError("training.cross_clip must be a mapping")
    allowed = {
        "enabled",
        "group_size",
        "group_stride",
        "band_hz",
        "tail_balanced_sampling",
    }
    unknown = sorted(set(cross_clip_config) - allowed)
    if unknown:
        raise ValueError(f"Unknown training.cross_clip keys: {unknown}")
    if not bool(cross_clip_config.get("enabled", False)):
        raise ValueError(
            "build_cross_clip_loader requires cross_clip.enabled=true"
        )
    group_size = int(cross_clip_config.get("group_size", 4))
    if int(batch_size) != group_size:
        raise ValueError(
            "Cross-clip group_size must equal the paper-protocol batch_size: "
            f"group_size={group_size}, batch_size={batch_size}"
        )
    tail = cross_clip_config.get("tail_balanced_sampling", {})
    if not isinstance(tail, dict):
        raise TypeError("tail_balanced_sampling must be a mapping")
    tail_allowed = {"enabled", "hr_bin_edges_bpm", "max_weight_ratio"}
    tail_unknown = sorted(set(tail) - tail_allowed)
    if tail_unknown:
        raise ValueError(
            f"Unknown tail_balanced_sampling keys: {tail_unknown}"
        )
    groups, audit = build_consecutive_clip_groups(
        records,
        group_size=group_size,
        group_stride=int(cross_clip_config.get("group_stride", group_size)),
        sampling_rate_hz=float(sampling_rate_hz),
        band_hz=cross_clip_config.get("band_hz", [0.7, 2.8]),
        hr_bin_edges_bpm=tail.get(
            "hr_bin_edges_bpm", [65.0, 80.0, 95.0, 110.0]
        ),
    )
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    samples_per_epoch = (len(records) + group_size - 1) // group_size
    sampler = TailBalancedGroupSampler(
        groups,
        generator=generator,
        enabled=bool(tail.get("enabled", True)),
        max_weight_ratio=float(tail.get("max_weight_ratio", 3.0)),
        num_samples=samples_per_epoch,
    )
    dataset = EgoPPGConsecutiveGroupDataset(
        records=records, groups=groups, augment=augment
    )
    loader = DataLoader(
        dataset,
        sampler=sampler,
        batch_size=1,
        collate_fn=_unwrap_single_group,
        num_workers=num_workers,
        pin_memory=True,
        # Restart workers each epoch so their augmentation RNG is reconstructed
        # from the checkpointed loader generator on resume. Persistent worker
        # RNG state is not serializable through the standard DataLoader API.
        persistent_workers=False,
        worker_init_fn=seed_worker,
        generator=generator,
    )
    weights = sampler.weights.numpy()
    probabilities = weights / weights.sum()
    bin_count = len(audit["hr_bin_edges_bpm"]) + 1
    probability_by_bin = np.bincount(
        [group.hr_bin for group in groups],
        weights=probabilities,
        minlength=bin_count,
    )
    audit.update(
        {
            "tail_balanced_sampling_enabled": sampler.enabled,
            "max_weight_ratio": float(tail.get("max_weight_ratio", 3.0)),
            "sampling_probability_by_hr_bin": probability_by_bin.tolist(),
            "batches_per_epoch": len(sampler),
            "clips_drawn_per_epoch": len(sampler) * group_size,
            "standard_loader_batches_per_epoch": samples_per_epoch,
            "replacement": sampler.enabled or len(sampler) > len(groups),
        }
    )
    return loader, audit
