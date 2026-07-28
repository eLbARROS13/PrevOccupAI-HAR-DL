"""Single-use, claim-gated external evaluation for frozen HAR models.

This module separates governance from data access.  The generic executor writes
the irreversible access ledger before it invokes any hold-out loader.  The data
helpers are consequently suitable only as callbacks reached after a successful
claim; they do not contain an alternate authorization path.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
from numpy.lib.format import open_memmap
from numpy.typing import NDArray

from .approved_dataset import parse_segment_filename
from .feature_store import (
    MAIN_LABELS,
    SUBACTIVITY_LABELS,
    SUBACTIVITY_MAIN_LABEL,
)
from .holdout import (
    HoldoutEvaluationPolicy,
    claim_holdout_access,
)
from .preprocessing import TrainOnlyChannelStandardizer
from .protocol import ProtocolConfiguration
from .provenance import sha256_canonical_json, sha256_file
from .signal_preprocessing import (
    SignalPreprocessingConfiguration,
    preprocess_accelerometer_segment,
)
from .window_store import (
    FILTER_TRANSIENT_SAMPLES,
    METADATA_DTYPE,
    DevelopmentWindowStore,
)
from .windowing import WindowMetadata, iter_window_bounds


EvaluationCallback = Callable[[Path, Mapping[str, Any]], Mapping[str, Any]]


def _utc_now() -> str:
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def _json_object(path: Path | str) -> dict[str, Any]:
    decoded = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(decoded, dict):
        raise TypeError(f"Expected a JSON object in {path}")
    return decoded


def _write_json_exclusive(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, allow_nan=False, indent=2, sort_keys=True)
        stream.write("\n")


def _assert_claim(
    claim_record: Mapping[str, Any],
    protocol: ProtocolConfiguration,
) -> None:
    if claim_record.get("state") != "access_claimed_before_data_read":
        raise PermissionError("Hold-out data helpers require a completed access claim")
    if int(claim_record.get("access_count", -1)) != 1:
        raise PermissionError("Hold-out data helpers require the single-use claim")
    if tuple(sorted(map(str, claim_record.get("holdout_participants", ())))) != tuple(
        sorted(protocol.holdout_participants)
    ):
        raise PermissionError("Claim and protocol hold-out cohorts differ")


def execute_claim_gated_holdout(
    *,
    protocol: ProtocolConfiguration,
    policy: HoldoutEvaluationPolicy,
    protocol_configuration_path: Path | str,
    model_freeze_manifest_path: Path | str,
    statistical_analysis_plan_path: Path | str,
    ledger_path: Path | str,
    accessed_at_utc: str,
    final_output_directory: Path | str,
    failure_record_path: Path | str,
    evaluator: EvaluationCallback,
) -> dict[str, Any]:
    """Claim access first, run one callback, and publish its output atomically.

    Any exception after the claim removes the partial result directory but does
    not remove or rewrite the access ledger.  A concise failure record is then
    written beside the intended result.  Existing outputs or failure records
    block before access is claimed.
    """
    final_directory = Path(final_output_directory).resolve()
    failure_path = Path(failure_record_path).resolve()
    ledger = Path(ledger_path).resolve()
    partial_directory = final_directory.with_name(
        f".{final_directory.name}.partial-{os.getpid()}"
    )
    for path, description in (
        (final_directory, "final hold-out result"),
        (partial_directory, "partial hold-out result"),
        (failure_path, "hold-out failure record"),
    ):
        if path.exists():
            raise FileExistsError(f"Existing {description} blocks execution: {path}")

    claim = claim_holdout_access(
        protocol=protocol,
        policy=policy,
        protocol_configuration_path=protocol_configuration_path,
        model_freeze_manifest_path=model_freeze_manifest_path,
        statistical_analysis_plan_path=statistical_analysis_plan_path,
        ledger_path=ledger,
        accessed_at_utc=accessed_at_utc,
    )
    if not ledger.is_file():
        raise RuntimeError("Access claim returned without a durable ledger")

    try:
        partial_directory.mkdir(parents=True, exist_ok=False)
        callback_result = evaluator(partial_directory, claim)
        if not isinstance(callback_result, Mapping):
            raise TypeError("Hold-out evaluator must return a mapping")
        completion = {
            "schema_version": 1,
            "status": "final_external_evaluation_completed",
            "authorization_id": claim["authorization_id"],
            "accessed_at_utc": claim["accessed_at_utc"],
            "holdout_participants": list(claim["holdout_participants"]),
            "access_ledger_sha256": sha256_file(ledger),
            "callback_result": dict(callback_result),
            "callback_result_payload_sha256": sha256_canonical_json(
                dict(callback_result)
            ),
        }
        _write_json_exclusive(partial_directory / "execution_record.json", completion)
        final_directory.parent.mkdir(parents=True, exist_ok=True)
        partial_directory.rename(final_directory)
        return completion
    except BaseException as error:
        shutil.rmtree(partial_directory, ignore_errors=True)
        failure = {
            "schema_version": 1,
            "status": "final_external_evaluation_failed_after_access_claim",
            "authorization_id": claim["authorization_id"],
            "accessed_at_utc": claim["accessed_at_utc"],
            "failure_recorded_at_utc": _utc_now(),
            "access_ledger_sha256": sha256_file(ledger),
            "exception_type": type(error).__name__,
            "exception_message": str(error),
            "retry_permitted": False,
        }
        _write_json_exclusive(failure_path, failure)
        raise


@dataclass(frozen=True)
class HoldoutFeatureMatrix:
    """Authoritative historical feature rows in their native per-file order."""

    features: NDArray[np.float64]
    labels: tuple[str, ...]
    subactivity_labels: tuple[str, ...]
    participant_ids: tuple[str, ...]
    feature_names: tuple[str, ...]
    manifest: Mapping[str, Any]


def _window_count(sample_count: int, protocol: ProtocolConfiguration) -> int:
    usable = sample_count - FILTER_TRANSIENT_SAMPLES
    if usable < protocol.window.expected_samples:
        return 0
    return (
        (usable - protocol.window.expected_samples) // protocol.window.step_samples
    ) + 1


def _recording_id(relative_name: str) -> str:
    return hashlib.sha256(relative_name.encode("utf-8")).hexdigest()


def build_holdout_window_store_after_claim(
    *,
    segment_root: Path | str,
    candidate_manifest_path: Path | str,
    preprocessing: SignalPreprocessingConfiguration,
    protocol: ProtocolConfiguration,
    claim_record: Mapping[str, Any],
    output_directory: Path | str,
) -> DevelopmentWindowStore:
    """Build hash-verified hold-out windows only after the access claim exists."""
    _assert_claim(claim_record, protocol)
    segment_directory = Path(segment_root).resolve()
    candidate_path = Path(candidate_manifest_path).resolve()
    output = Path(output_directory).resolve()
    if output.exists():
        raise FileExistsError(f"Hold-out window store already exists: {output}")
    if not preprocessing.authoritative or not preprocessing.controls_dataset_generation:
        raise PermissionError("Only the authoritative physical preprocessing is allowed")
    if preprocessing.sampling_rate_hz != protocol.sampling_rate_hz:
        raise ValueError("Preprocessing and protocol sampling rates differ")

    candidate = _json_object(candidate_path)
    segment_section = candidate.get("segments")
    feature_section = candidate.get("feature_matrices")
    if not isinstance(segment_section, Mapping) or not isinstance(
        feature_section, Mapping
    ):
        raise TypeError("Candidate manifest lacks segment or feature sections")
    if segment_section.get("checksums_calculated") is not True:
        raise ValueError("Candidate segment identities are not checksummed")
    arrays = segment_section.get("arrays")
    matrices = feature_section.get("matrices")
    if not isinstance(arrays, list) or not isinstance(matrices, list):
        raise TypeError("Candidate segment and feature entries must be arrays")
    holdout = set(protocol.holdout_participants)
    entries = [
        entry
        for entry in arrays
        if isinstance(entry, Mapping)
        and str(entry.get("participant_id")) in holdout
        and str(entry.get("quality_status")) == "GOOD"
    ]
    if {str(entry["participant_id"]) for entry in entries} != holdout:
        raise ValueError("Every hold-out participant must have retained segments")

    participant_counts: Counter[str] = Counter()
    class_counts: Counter[str] = Counter()
    total_windows = 0
    for entry in entries:
        parsed = parse_segment_filename(str(entry["relative_name"]))
        if (
            parsed.participant_id != str(entry["participant_id"])
            or parsed.quality_status != str(entry["quality_status"])
        ):
            raise ValueError("Candidate segment metadata and filename disagree")
        count = _window_count(int(entry["shape"][0]), protocol)
        participant_counts[parsed.participant_id] += count
        class_counts[parsed.main_label] += count
        total_windows += count
    reference_counts = {
        str(entry["participant_id"]): int(entry["window_count"])
        for entry in matrices
        if isinstance(entry, Mapping)
        and str(entry.get("participant_id")) in holdout
    }
    if dict(sorted(participant_counts.items())) != dict(sorted(reference_counts.items())):
        raise ValueError("Hold-out segment and historical feature window counts differ")
    if total_windows <= 0:
        raise ValueError("Hold-out segments produce no complete windows")

    output.mkdir(parents=True, exist_ok=False)
    windows_path = output / "windows.npy"
    labels_path = output / "labels.npy"
    metadata_path = output / "metadata.npy"
    windows = open_memmap(
        windows_path,
        mode="w+",
        dtype=np.float32,
        shape=(
            total_windows,
            protocol.window.expected_samples,
            protocol.accelerometer_channels,
        ),
    )
    labels = open_memmap(
        labels_path,
        mode="w+",
        dtype=np.int64,
        shape=(total_windows,),
    )
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
            relative_name = str(entry["relative_name"])
            parsed = parse_segment_filename(relative_name)
            segment_path = segment_directory / relative_name
            if not segment_path.is_file():
                raise FileNotFoundError(f"Hold-out segment is missing: {relative_name}")
            if segment_path.stat().st_size != int(entry["size_bytes"]):
                raise ValueError(f"Hold-out segment size changed: {relative_name}")
            if sha256_file(segment_path) != str(entry["sha256"]):
                raise ValueError(f"Hold-out segment hash changed: {relative_name}")
            segment = np.load(segment_path, mmap_mode="r", allow_pickle=False)
            if list(segment.shape) != list(entry["shape"]) or str(segment.dtype) != str(
                entry["dtype"]
            ):
                raise ValueError(f"Hold-out segment schema changed: {relative_name}")
            processed = preprocess_accelerometer_segment(segment[:, 1:4], preprocessing)
            usable = processed.dynamic_acceleration_m_s2[FILTER_TRANSIENT_SAMPLES:]
            recording_id = _recording_id(relative_name)
            for start, end in iter_window_bounds(
                usable.shape[0],
                window_size_samples=protocol.window.expected_samples,
                step_size_samples=protocol.window.step_samples,
            ):
                windows[output_index] = usable[start:end]
                labels[output_index] = label_to_index[parsed.main_label]
                metadata[output_index] = (
                    parsed.participant_id,
                    recording_id,
                    parsed.main_label,
                    parsed.sub_activity_label,
                    parsed.file_token,
                    "unk",
                    relative_name,
                    start + FILTER_TRANSIENT_SAMPLES,
                    end + FILTER_TRANSIENT_SAMPLES,
                    f"{preprocessing.name};discard_initial={FILTER_TRANSIENT_SAMPLES}",
                    "GOOD",
                )
                output_index += 1
        if output_index != total_windows:
            raise RuntimeError("Generated hold-out window count changed during writing")
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
        index: dict[str, Any] = {
            "schema_version": 1,
            "status": "authorized_external_holdout_window_store",
            "scientific_training_authorized": False,
            "authorization_scope": "single_final_external_evaluation",
            "holdout_accessed": True,
            "authorization_id": claim_record["authorization_id"],
            "holdout_participants": list(protocol.holdout_participants),
            "development_participants_loaded": [],
            "class_labels": list(protocol.main_labels),
            "window_shape": list(windows.shape),
            "window_dtype": str(windows.dtype),
            "label_dtype": str(labels.dtype),
            "metadata_dtype": METADATA_DTYPE.descr,
            "window_count": total_windows,
            "window_counts_per_participant": dict(sorted(participant_counts.items())),
            "window_counts_per_class": dict(sorted(class_counts.items())),
            "retained_segment_count": len(entries),
            "filter_transient_samples_discarded": FILTER_TRANSIENT_SAMPLES,
            "sensor_side_provenance": (
                "unknown_for_holdout; file-token stream identity retained; models are "
                "pooled and side-agnostic"
            ),
            "recording_id_provenance": "sha256_of_segment_relative_name",
            "candidate_manifest_sha256": sha256_file(candidate_path),
            "files": files,
        }
        index["payload_sha256"] = sha256_canonical_json(index)
        _write_json_exclusive(output / "index.json", index)
    except BaseException:
        del windows, labels, metadata
        shutil.rmtree(output, ignore_errors=True)
        raise
    return load_holdout_window_store(output, verify_file_hashes=True)


def load_holdout_window_store(
    directory: Path | str,
    *,
    verify_file_hashes: bool = False,
) -> DevelopmentWindowStore:
    """Reload an authorized hold-out store without offering an access bypass."""
    root = Path(directory).resolve()
    index = _json_object(root / "index.json")
    payload = dict(index)
    recorded_payload_sha256 = str(payload.pop("payload_sha256"))
    if sha256_canonical_json(payload) != recorded_payload_sha256:
        raise ValueError("Hold-out window-store payload digest changed")
    if (
        index.get("status") != "authorized_external_holdout_window_store"
        or index.get("holdout_accessed") is not True
        or index.get("development_participants_loaded") != []
    ):
        raise PermissionError("Window store is not an authorized hold-out artifact")
    paths = {
        name: root / str(entry["relative_name"])
        for name, entry in index["files"].items()
    }
    for name, path in paths.items():
        entry = index["files"][name]
        if path.stat().st_size != int(entry["size_bytes"]):
            raise ValueError(f"Hold-out window-store file size changed: {name}")
        if verify_file_hashes and sha256_file(path) != str(entry["sha256"]):
            raise ValueError(f"Hold-out window-store file hash changed: {name}")
    windows = np.load(paths["windows"], mmap_mode="r", allow_pickle=False)
    labels = np.load(paths["labels"], mmap_mode="r", allow_pickle=False)
    metadata = np.load(paths["metadata"], mmap_mode="r", allow_pickle=False)
    expected_shape = tuple(map(int, index["window_shape"]))
    if windows.shape != expected_shape or labels.shape != (expected_shape[0],):
        raise ValueError("Hold-out window arrays disagree with their index")
    if metadata.shape != (expected_shape[0],) or metadata.dtype != METADATA_DTYPE:
        raise ValueError("Hold-out metadata disagree with their index")
    return DevelopmentWindowStore(windows, labels, metadata, index)


def load_holdout_feature_matrix_after_claim(
    *,
    feature_root: Path | str,
    candidate_manifest_path: Path | str,
    protocol: ProtocolConfiguration,
    claim_record: Mapping[str, Any],
    manifest_output_path: Path | str,
) -> HoldoutFeatureMatrix:
    """Load hash-bound hold-out TSFEL matrices in native historical row order."""
    _assert_claim(claim_record, protocol)
    root = Path(feature_root).resolve()
    candidate_path = Path(candidate_manifest_path).resolve()
    candidate = _json_object(candidate_path)
    section = candidate.get("feature_matrices")
    if not isinstance(section, Mapping) or section.get("checksums_calculated") is not True:
        raise ValueError("Candidate feature identities are incomplete")
    feature_names = tuple(map(str, section.get("feature_columns", ())))
    if len(feature_names) != 45 or len(set(feature_names)) != 45:
        raise ValueError("Hold-out feature names are incomplete or duplicated")
    values = section.get("matrices")
    if not isinstance(values, list):
        raise TypeError("Candidate feature matrices must be an array")
    entries = {
        str(entry["participant_id"]): entry
        for entry in values
        if isinstance(entry, Mapping)
    }
    holdout = tuple(protocol.holdout_participants)
    if set(holdout) - set(entries):
        raise ValueError("Candidate manifest lacks hold-out feature matrices")

    feature_arrays: list[NDArray[np.float64]] = []
    labels: list[str] = []
    subactivities: list[str] = []
    participants: list[str] = []
    files: list[dict[str, Any]] = []
    for participant in holdout:
        entry = entries[participant]
        path = root / str(entry["relative_name"])
        if path.name != f"{participant}.npy":
            raise ValueError("Hold-out feature filename and participant disagree")
        if path.stat().st_size != int(entry["size_bytes"]):
            raise ValueError("Hold-out feature matrix size changed")
        digest = sha256_file(path)
        if digest != str(entry["sha256"]):
            raise ValueError("Hold-out feature matrix hash changed")
        matrix = np.load(path, allow_pickle=False)
        if (
            matrix.shape != tuple(map(int, entry["shape"]))
            or matrix.dtype != np.float64
            or matrix.shape[1] != 47
            or not np.isfinite(matrix).all()
        ):
            raise ValueError("Hold-out feature matrix schema or values are invalid")
        main_values = matrix[:, 45]
        sub_values = matrix[:, 46]
        if not np.equal(main_values, np.rint(main_values)).all() or not np.equal(
            sub_values, np.rint(sub_values)
        ).all():
            raise ValueError("Hold-out feature labels must be integral")
        main_codes = main_values.astype(np.int64)
        sub_codes = sub_values.astype(np.int64)
        if set(main_codes.tolist()) - set(MAIN_LABELS) or set(
            sub_codes.tolist()
        ) - set(SUBACTIVITY_LABELS):
            raise ValueError("Hold-out feature matrix contains unknown labels")
        if any(
            SUBACTIVITY_MAIN_LABEL[int(sub)] != int(main)
            for main, sub in zip(main_codes, sub_codes, strict=True)
        ):
            raise ValueError("Hold-out feature main and sub-activity labels disagree")
        feature_arrays.append(np.asarray(matrix[:, :45], dtype=np.float64))
        labels.extend(MAIN_LABELS[int(code)] for code in main_codes)
        subactivities.extend(SUBACTIVITY_LABELS[int(code)] for code in sub_codes)
        participants.extend([participant] * matrix.shape[0])
        files.append(
            {
                "participant_id": participant,
                "relative_name": path.name,
                "shape": list(matrix.shape),
                "dtype": str(matrix.dtype),
                "size_bytes": path.stat().st_size,
                "sha256": digest,
            }
        )
    combined = np.ascontiguousarray(np.vstack(feature_arrays), dtype=np.float64)
    payload: dict[str, Any] = {
        "authorization_scope": "single_final_external_evaluation",
        "authorization_id": claim_record["authorization_id"],
        "holdout_values_loaded": True,
        "development_values_loaded": False,
        "holdout_participants": list(holdout),
        "candidate_manifest_sha256": sha256_file(candidate_path),
        "feature_names": list(feature_names),
        "files": files,
        "row_count": int(combined.shape[0]),
        "column_count": int(combined.shape[1]),
        "class_counts": dict(sorted(Counter(labels).items())),
        "row_order_provenance": "native_authoritative_feature_matrix_order",
        "raw_window_row_alignment_claimed": False,
    }
    manifest = {
        "schema_version": 1,
        "status": "authorized_external_holdout_feature_matrix",
        **payload,
        "payload_sha256": sha256_canonical_json(payload),
    }
    _write_json_exclusive(Path(manifest_output_path), manifest)
    return HoldoutFeatureMatrix(
        features=combined,
        labels=tuple(labels),
        subactivity_labels=tuple(subactivities),
        participant_ids=tuple(participants),
        feature_names=feature_names,
        manifest=manifest,
    )


def metadata_records(store: DevelopmentWindowStore) -> tuple[WindowMetadata, ...]:
    """Decode path-free hold-out metadata for ordered temporal diagnostics."""
    return tuple(
        WindowMetadata(
            subject_id=str(row["participant_id"]),
            recording_id=str(row["recording_id"]),
            main_label=str(row["main_label"]),
            sub_activity_label=str(row["sub_activity_label"]),
            sensor_stream_id=str(row["device_stream_id"]),
            sensor_side=str(row["sensor_side"]),
            start_sample=int(row["start_sample"]),
            end_sample_exclusive=int(row["end_sample_exclusive"]),
            preprocessing_status=str(row["preprocessing_status"]),
            quality_status=str(row["quality_status"]),
        )
        for row in store.metadata
    )


def load_final_standardizer(
    path: Path | str,
    *,
    development_participants: Sequence[str],
    expected_payload_sha256: str,
) -> tuple[TrainOnlyChannelStandardizer, int]:
    """Restore fixed development-only channel statistics for inference."""
    decoded = _json_object(path)
    state = decoded.get("state")
    if (
        decoded.get("status") != "all_development_train_only_preprocessing"
        or decoded.get("holdout_accessed") is not False
        or not isinstance(state, Mapping)
    ):
        raise ValueError("Final preprocessing artifact has an invalid scope")
    if str(decoded.get("state_payload_sha256")) != expected_payload_sha256 or (
        sha256_canonical_json(state) != expected_payload_sha256
    ):
        raise ValueError("Final preprocessing payload digest changed")
    standardizer = TrainOnlyChannelStandardizer.for_subjects(development_participants)
    standardizer.mean_ = np.asarray(state.get("mean"), dtype=np.float64)
    standardizer.scale_ = np.asarray(state.get("scale"), dtype=np.float64)
    if (
        standardizer.mean_.shape != (3,)
        or standardizer.scale_.shape != (3,)
        or not np.isfinite(standardizer.mean_).all()
        or not np.isfinite(standardizer.scale_).all()
        or np.any(standardizer.scale_ <= 0)
    ):
        raise ValueError("Final preprocessing statistics are invalid")
    stride = int(state.get("sample_stride", -1))
    if stride <= 0:
        raise ValueError("Final preprocessing artifact lacks a valid sample stride")
    return standardizer, stride


def verify_modality_count_alignment(
    raw_store: DevelopmentWindowStore,
    features: HoldoutFeatureMatrix,
    *,
    class_labels: Sequence[str],
    holdout_participants: Sequence[str],
) -> dict[str, Any]:
    """Verify counts while explicitly refusing an unsupported rowwise pairing."""
    labels = tuple(map(str, class_labels))
    participants = tuple(map(str, holdout_participants))
    raw_labels = tuple(str(value) for value in raw_store.metadata["main_label"])
    raw_participants = tuple(str(value) for value in raw_store.metadata["participant_id"])
    if set(raw_participants) != set(participants) or set(features.participant_ids) != set(
        participants
    ):
        raise ValueError("Raw and feature modalities do not cover the frozen hold-out cohort")
    raw_counts = Counter(zip(raw_participants, raw_labels, strict=True))
    feature_counts = Counter(
        zip(features.participant_ids, features.labels, strict=True)
    )
    if raw_counts != feature_counts:
        raise ValueError("Raw and feature participant/class window counts differ")
    if set(raw_labels) - set(labels) or set(features.labels) - set(labels):
        raise ValueError("A hold-out modality contains labels outside the frozen classes")
    nested = {
        participant: {
            label: int(raw_counts[(participant, label)]) for label in labels
        }
        for participant in participants
    }
    return {
        "status": "participant_class_counts_match",
        "participant_class_counts": nested,
        "raw_window_count": len(raw_labels),
        "feature_row_count": len(features.labels),
        "raw_feature_rowwise_alignment_claimed": False,
        "paired_comparison_unit": "participant",
        "rf_temporal_diagnostics": (
            "not_computed_exact_feature_row_sequence_provenance_unavailable"
        ),
    }


__all__ = [
    "HoldoutFeatureMatrix",
    "build_holdout_window_store_after_claim",
    "execute_claim_gated_holdout",
    "load_final_standardizer",
    "load_holdout_feature_matrix_after_claim",
    "load_holdout_window_store",
    "metadata_records",
    "verify_modality_count_alignment",
]
