"""Hash-bound development window stores built from approved segment arrays."""

from __future__ import annotations

import json
import shutil
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from numpy.lib.format import open_memmap
from numpy.typing import NDArray

from .protocol import ProtocolConfiguration, load_protocol
from .provenance import sha256_canonical_json, sha256_file
from .signal_preprocessing import (
    SignalPreprocessingConfiguration,
    load_signal_preprocessing_configuration,
    preprocess_accelerometer_segment,
)
from .windowing import iter_window_bounds


FILTER_TRANSIENT_SAMPLES = 250
METADATA_DTYPE = np.dtype(
    [
        ("participant_id", "U4"),
        ("recording_id", "U64"),
        ("main_label", "U8"),
        ("sub_activity_label", "U32"),
        ("device_stream_id", "U12"),
        ("sensor_side", "U5"),
        ("segment_relative_name", "U180"),
        ("start_sample", "<i8"),
        ("end_sample_exclusive", "<i8"),
        ("preprocessing_status", "U64"),
        ("quality_status", "U4"),
    ]
)


@dataclass(frozen=True)
class DevelopmentWindowStore:
    """Memory-mapped windows, labels, and aligned immutable metadata."""

    windows: NDArray[np.float32]
    labels: NDArray[np.int64]
    metadata: NDArray[Any]
    index: Mapping[str, Any]


def _load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object in {path}")
    return value


def _validated_payload(value: Mapping[str, Any], *, field: str = "payload_sha256") -> None:
    recorded = str(value[field])
    payload = dict(value)
    payload.pop(field)
    if sha256_canonical_json(payload) != recorded:
        raise ValueError(f"{field} does not match the canonical payload")


def _window_count(sample_count: int, protocol: ProtocolConfiguration) -> int:
    usable = sample_count - FILTER_TRANSIENT_SAMPLES
    if usable < protocol.window.expected_samples:
        return 0
    return (
        (usable - protocol.window.expected_samples) // protocol.window.step_samples
    ) + 1


def _reference_counts(
    candidate_manifest: Mapping[str, Any],
    participants: set[str],
) -> dict[str, int]:
    return {
        str(entry["participant_id"]): int(entry["window_count"])
        for entry in candidate_manifest["feature_matrices"]["matrices"]
        if str(entry["participant_id"]) in participants
    }


def build_development_window_store(
    *,
    approved_manifest_path: Path,
    candidate_manifest_path: Path,
    segment_root: Path,
    protocol_path: Path,
    preprocessing_configuration_path: Path,
    output_directory: Path,
) -> dict[str, Any]:
    """Build an exclusive development-only memmap store from approved segments."""
    approved_manifest_path = approved_manifest_path.resolve()
    candidate_manifest_path = candidate_manifest_path.resolve()
    segment_root = segment_root.resolve()
    protocol_path = protocol_path.resolve()
    preprocessing_configuration_path = preprocessing_configuration_path.resolve()
    output_directory = output_directory.resolve()
    partial_directory = output_directory.with_name(f"{output_directory.name}.partial")
    if output_directory.exists() or partial_directory.exists():
        raise FileExistsError("Window-store output and partial paths must not exist")

    protocol = load_protocol(protocol_path)
    preprocessing = load_signal_preprocessing_configuration(
        preprocessing_configuration_path
    )
    approved = _load_object(approved_manifest_path)
    candidate = _load_object(candidate_manifest_path)
    _validated_payload(approved)
    if approved.get("status") != "author_approved_development_dataset":
        raise ValueError("Development segment manifest is not author approved")
    if approved.get("scientific_training_authorized") is not True:
        raise PermissionError("Development segment manifest does not authorize training")
    if approved.get("holdout_accessed") is not False:
        raise PermissionError("Development window generation cannot follow hold-out access")
    if not protocol.training_authorized or protocol.holdout_access_authorized:
        raise PermissionError("Protocol scope does not authorize development-only generation")
    if not preprocessing.authoritative or not preprocessing.controls_dataset_generation:
        raise PermissionError("Preprocessing configuration does not control generation")
    if protocol.segmentation_manifest != approved_manifest_path:
        raise ValueError("Protocol and approved segment manifest paths differ")
    if protocol.signal_preprocessing_configuration != preprocessing_configuration_path:
        raise ValueError("Protocol and preprocessing configuration paths differ")

    development = set(protocol.development_participants)
    if set(map(str, approved["development_participants"])) != development:
        raise ValueError("Approved manifest and protocol development cohorts differ")
    entries = [
        entry
        for entry in approved["segments"]
        if entry["quality_status"] == "GOOD"
    ]
    if {str(entry["participant_id"]) for entry in entries} != development:
        raise ValueError("Every development participant must have retained segments")

    calculated_counts: Counter[str] = Counter()
    class_counts: Counter[str] = Counter()
    total_windows = 0
    for entry in entries:
        participant = str(entry["participant_id"])
        count = _window_count(int(entry["shape"][0]), protocol)
        calculated_counts[participant] += count
        class_counts[str(entry["main_label"])] += count
        total_windows += count
    references = _reference_counts(candidate, development)
    if dict(sorted(calculated_counts.items())) != dict(sorted(references.items())):
        raise ValueError(
            "Approved segment-derived window counts disagree with the recovered feature matrices"
        )
    if total_windows <= 0:
        raise ValueError("Approved development segments produce no windows")

    partial_directory.mkdir(parents=True)
    windows_path = partial_directory / "windows.npy"
    labels_path = partial_directory / "labels.npy"
    metadata_path = partial_directory / "metadata.npy"
    windows = open_memmap(
        windows_path,
        mode="w+",
        dtype=np.float32,
        shape=(total_windows, protocol.window.expected_samples, protocol.accelerometer_channels),
    )
    labels = open_memmap(labels_path, mode="w+", dtype=np.int64, shape=(total_windows,))
    metadata = open_memmap(
        metadata_path,
        mode="w+",
        dtype=METADATA_DTYPE,
        shape=(total_windows,),
    )
    label_to_index = {label: index for index, label in enumerate(protocol.main_labels)}
    output_index = 0
    try:
        for entry in sorted(entries, key=lambda item: str(item["relative_name"])):
            segment_path = segment_root / str(entry["relative_name"])
            if not segment_path.is_file():
                raise FileNotFoundError(f"Approved segment is missing: {segment_path.name}")
            if segment_path.stat().st_size != int(entry["size_bytes"]):
                raise ValueError(f"Approved segment size changed: {segment_path.name}")
            if sha256_file(segment_path) != entry["sha256"]:
                raise ValueError(f"Approved segment hash changed: {segment_path.name}")
            segment = np.load(segment_path, mmap_mode="r", allow_pickle=False)
            if list(segment.shape) != list(entry["shape"]) or str(segment.dtype) != entry["dtype"]:
                raise ValueError(f"Approved segment schema changed: {segment_path.name}")
            processed = preprocess_accelerometer_segment(segment[:, 1:4], preprocessing)
            usable = processed.dynamic_acceleration_m_s2[FILTER_TRANSIENT_SAMPLES:]
            for start, end in iter_window_bounds(
                usable.shape[0],
                window_size_samples=protocol.window.expected_samples,
                step_size_samples=protocol.window.step_samples,
            ):
                windows[output_index] = usable[start:end]
                labels[output_index] = label_to_index[str(entry["main_label"])]
                metadata[output_index] = (
                    entry["participant_id"],
                    entry["recording_id"],
                    entry["main_label"],
                    entry["sub_activity_label"],
                    entry["device_stream_id"],
                    entry["sensor_side"],
                    entry["relative_name"],
                    start + FILTER_TRANSIENT_SAMPLES,
                    end + FILTER_TRANSIENT_SAMPLES,
                    f"{preprocessing.name};discard_initial={FILTER_TRANSIENT_SAMPLES}",
                    "GOOD",
                )
                output_index += 1
        if output_index != total_windows:
            raise RuntimeError(
                f"Generated window count changed during writing: {output_index} != {total_windows}"
            )
        windows.flush()
        labels.flush()
        metadata.flush()

        files = {
            "windows": {
                "relative_name": "windows.npy",
                "sha256": sha256_file(windows_path),
                "size_bytes": windows_path.stat().st_size,
            },
            "labels": {
                "relative_name": "labels.npy",
                "sha256": sha256_file(labels_path),
                "size_bytes": labels_path.stat().st_size,
            },
            "metadata": {
                "relative_name": "metadata.npy",
                "sha256": sha256_file(metadata_path),
                "size_bytes": metadata_path.stat().st_size,
            },
        }
        index = {
            "schema_version": 1,
            "status": "approved_development_window_store",
            "scientific_training_authorized": True,
            "authorization_scope": "development_selection_only",
            "holdout_accessed": False,
            "development_participants": sorted(development),
            "sealed_holdout_participants": list(protocol.holdout_participants),
            "class_labels": list(protocol.main_labels),
            "window_shape": [
                total_windows,
                protocol.window.expected_samples,
                protocol.accelerometer_channels,
            ],
            "window_dtype": "float32",
            "label_dtype": "int64",
            "metadata_dtype": METADATA_DTYPE.descr,
            "window_count": total_windows,
            "window_counts_per_participant": dict(sorted(calculated_counts.items())),
            "window_counts_per_class": dict(sorted(class_counts.items())),
            "retained_segment_count": len(entries),
            "filter_transient_samples_discarded": FILTER_TRANSIENT_SAMPLES,
            "source_identity": {
                "approved_manifest_sha256": sha256_file(approved_manifest_path),
                "approved_manifest_payload_sha256": approved["payload_sha256"],
                "candidate_manifest_sha256": sha256_file(candidate_manifest_path),
                "protocol_sha256": sha256_file(protocol_path),
                "preprocessing_configuration_sha256": sha256_file(
                    preprocessing_configuration_path
                ),
            },
            "files": files,
        }
        index["payload_sha256"] = sha256_canonical_json(index)
        (partial_directory / "index.json").write_text(
            json.dumps(index, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        partial_directory.rename(output_directory)
        return index
    except BaseException:
        del windows, labels, metadata
        shutil.rmtree(partial_directory, ignore_errors=True)
        raise


def load_development_window_store(
    directory: Path,
    *,
    verify_file_hashes: bool = False,
) -> DevelopmentWindowStore:
    """Load and validate an existing development-only store as memory maps."""
    directory = directory.resolve()
    index = _load_object(directory / "index.json")
    _validated_payload(index)
    if index.get("status") != "approved_development_window_store":
        raise ValueError("Window store has an unsupported status")
    if index.get("holdout_accessed") is not False:
        raise PermissionError("Development store records hold-out access")
    paths = {
        name: directory / str(entry["relative_name"])
        for name, entry in index["files"].items()
    }
    for name, path in paths.items():
        entry = index["files"][name]
        if path.stat().st_size != int(entry["size_bytes"]):
            raise ValueError(f"Window-store file size changed: {name}")
        if verify_file_hashes and sha256_file(path) != entry["sha256"]:
            raise ValueError(f"Window-store file hash changed: {name}")
    windows = np.load(paths["windows"], mmap_mode="r", allow_pickle=False)
    labels = np.load(paths["labels"], mmap_mode="r", allow_pickle=False)
    metadata = np.load(paths["metadata"], mmap_mode="r", allow_pickle=False)
    expected_shape = tuple(map(int, index["window_shape"]))
    if windows.shape != expected_shape or labels.shape != (expected_shape[0],):
        raise ValueError("Window-store arrays disagree with the index")
    if metadata.shape != (expected_shape[0],) or metadata.dtype != METADATA_DTYPE:
        raise ValueError("Window-store metadata disagree with the index")
    return DevelopmentWindowStore(windows, labels, metadata, index)
