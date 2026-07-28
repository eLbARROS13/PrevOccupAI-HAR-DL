"""Audit a candidate processed muscleBAN dataset without authorizing its use."""

from __future__ import annotations

import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np


PARTICIPANT_PATTERN = re.compile(r"^(P[0-9]{3})(?:_|[.])")
SEGMENT_PATTERN = re.compile(
    r"^(?P<base>P[0-9]{3}_.+)_GlobalSegment(?P<global>[0-9]+)"
    r"(?:_LocalSegment(?P<local>[0-9]+))?(?P<failed>_failed)?[.]npy$"
)
EXPECTED_AXES = ("xAcc", "yAcc", "zAcc")


def _sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Return the SHA-256 digest of a file without loading it into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as file_handle:
        while chunk := file_handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _content_digest(entries: Iterable[dict[str, Any]]) -> str:
    """Hash stable JSON identities for a collection of audited files."""
    digest = hashlib.sha256()
    for entry in sorted(entries, key=lambda item: str(item["relative_name"])):
        identity = {
            "relative_name": entry["relative_name"],
            "size_bytes": entry["size_bytes"],
            "sha256": entry["sha256"],
        }
        digest.update(
            json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
        )
        digest.update(b"\n")
    return digest.hexdigest()


def _participant_from_name(name: str) -> str:
    """Extract a pseudonymous participant identifier from an artefact name."""
    match = PARTICIPANT_PATTERN.match(name)
    if match is None:
        raise ValueError(f"Cannot extract participant identifier from {name!r}")
    return match.group(1)


def _load_qa_summary(path: Path) -> tuple[dict[str, Any], dict[tuple[str, str], str]]:
    """Audit the per-axis QA table and aggregate it to evaluated segments."""
    with path.open(newline="", encoding="utf-8-sig") as file_handle:
        rows = list(csv.DictReader(file_handle))
    required_columns = {
        "original_file",
        "segment_name",
        "segment_duration_sec",
        "axis",
        "status",
    }
    if not rows:
        raise ValueError("QA summary is empty")
    missing_columns = required_columns - set(rows[0])
    if missing_columns:
        raise ValueError(f"QA summary lacks columns: {sorted(missing_columns)}")

    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[(row["original_file"], row["segment_name"])].append(row)

    aggregate_status: dict[tuple[str, str], str] = {}
    total_duration_seconds = 0.0
    rejected_duration_seconds = 0.0
    participants: Counter[str] = Counter()
    for key, segment_rows in grouped.items():
        rows_by_axis: dict[str, list[dict[str, str]]] = defaultdict(list)
        for row in segment_rows:
            rows_by_axis[row["axis"]].append(row)
        axes = tuple(sorted(rows_by_axis))
        if axes != tuple(sorted(EXPECTED_AXES)):
            raise ValueError(f"QA segment {key!r} has unexpected axes: {axes}")
        for axis, duplicate_rows in rows_by_axis.items():
            if any(row != duplicate_rows[0] for row in duplicate_rows[1:]):
                raise ValueError(
                    f"QA segment {key!r}, axis {axis!r} has conflicting duplicate rows"
                )
        unique_axis_rows = [rows_by_axis[axis][0] for axis in EXPECTED_AXES]
        statuses = {row["status"] for row in unique_axis_rows}
        if not statuses <= {"GOOD", "BAD"}:
            raise ValueError(f"QA segment {key!r} has unexpected statuses: {statuses}")
        status = "BAD" if "BAD" in statuses else "GOOD"
        aggregate_status[key] = status
        duration_seconds = float(unique_axis_rows[0]["segment_duration_sec"])
        total_duration_seconds += duration_seconds
        if status == "BAD":
            rejected_duration_seconds += duration_seconds
        participants[_participant_from_name(key[0])] += 1

    status_counts = Counter(aggregate_status.values())
    return (
        {
            "relative_name": path.name,
            "size_bytes": path.stat().st_size,
            "sha256": _sha256_file(path),
            "row_count": len(rows),
            "deduplicated_axis_row_count": len(grouped) * len(EXPECTED_AXES),
            "identical_duplicate_row_count": len(rows) - len(grouped) * len(EXPECTED_AXES),
            "evaluated_segment_count": len(grouped),
            "retained_segment_count": status_counts["GOOD"],
            "rejected_segment_count": status_counts["BAD"],
            "total_duration_seconds": total_duration_seconds,
            "total_duration_hours": total_duration_seconds / 3600.0,
            "rejected_duration_seconds": rejected_duration_seconds,
            "rejected_duration_minutes": rejected_duration_seconds / 60.0,
            "axes": list(EXPECTED_AXES),
            "participants": sorted(participants),
            "evaluated_segments_per_participant": dict(sorted(participants.items())),
        },
        aggregate_status,
    )


def _match_segment_to_qa(
    filename: str,
    qa_status: dict[tuple[str, str], str],
) -> str:
    """Return the QA status corresponding to a generated segment filename."""
    match = SEGMENT_PATTERN.fullmatch(filename)
    if match is None:
        raise ValueError(f"Unexpected candidate segment filename: {filename!r}")
    original_file = f"{match.group('base')}.csv"
    local_index = match.group("local")
    if local_index is None:
        key = (original_file, "complete_file")
        if key not in qa_status:
            raise ValueError(f"No complete-file QA row corresponds to {filename!r}")
        return qa_status[key]

    prefix = f"global_{match.group('global')}_local_{local_index}_samples_"
    matches = [
        status
        for (qa_file, segment_name), status in qa_status.items()
        if qa_file == original_file and segment_name.startswith(prefix)
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Expected one QA row for {filename!r} with prefix {prefix!r}; found {len(matches)}"
        )
    return matches[0]


def _audit_segments(
    root: Path,
    qa_status: dict[tuple[str, str], str],
    *,
    calculate_checksums: bool,
) -> dict[str, Any]:
    """Audit generated segment arrays and bind each array to its QA status."""
    paths = sorted(root.glob("P*.npy"))
    if not paths:
        raise ValueError(f"No candidate segment arrays found in {root}")

    entries: list[dict[str, Any]] = []
    counts_by_participant: Counter[str] = Counter()
    status_counts: Counter[str] = Counter()
    total_size_bytes = 0
    for path in paths:
        match = SEGMENT_PATTERN.fullmatch(path.name)
        if match is None:
            raise ValueError(f"Unexpected candidate segment filename: {path.name!r}")
        array = np.load(path, mmap_mode="r", allow_pickle=False)
        if array.ndim != 2 or array.shape[1] != 4:
            raise ValueError(f"Segment {path.name!r} must have shape (samples, 4); got {array.shape}")
        qa_segment_status = _match_segment_to_qa(path.name, qa_status)
        filename_status = "BAD" if match.group("failed") else "GOOD"
        if filename_status != qa_segment_status:
            raise ValueError(
                f"Filename and QA status disagree for {path.name!r}: "
                f"{filename_status} != {qa_segment_status}"
            )
        participant = _participant_from_name(path.name)
        size_bytes = path.stat().st_size
        entry = {
            "relative_name": path.name,
            "participant_id": participant,
            "quality_status": qa_segment_status,
            "shape": list(array.shape),
            "dtype": str(array.dtype),
            "size_bytes": size_bytes,
            "sha256": _sha256_file(path) if calculate_checksums else None,
        }
        entries.append(entry)
        counts_by_participant[participant] += 1
        status_counts[qa_segment_status] += 1
        total_size_bytes += size_bytes

    content_sha256 = _content_digest(entries) if calculate_checksums else None
    return {
        "root_label": "candidate_quality_controlled_segment_arrays",
        "array_count": len(entries),
        "retained_array_count": status_counts["GOOD"],
        "rejected_array_count": status_counts["BAD"],
        "participants": sorted(counts_by_participant),
        "arrays_per_participant": dict(sorted(counts_by_participant.items())),
        "total_size_bytes": total_size_bytes,
        "checksums_calculated": calculate_checksums,
        "content_sha256": content_sha256,
        "arrays": entries,
    }


def _audit_feature_matrices(
    root: Path,
    *,
    calculate_checksums: bool,
) -> dict[str, Any]:
    """Audit participant feature matrices and their feature/label schema."""
    metadata_path = root / "class_instances.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    columns = list(metadata.get("feature_cols", []))
    if len(columns) < 3 or columns[-2:] != ["main_label", "sub_label"]:
        raise ValueError("Feature metadata must end with main_label and sub_label")

    entries: list[dict[str, Any]] = []
    total_windows = 0
    for path in sorted(root.glob("P*.npy")):
        participant = _participant_from_name(path.name)
        array = np.load(path, mmap_mode="r", allow_pickle=False)
        if array.ndim != 2 or array.shape[1] != len(columns):
            raise ValueError(
                f"Feature matrix {path.name!r} has shape {array.shape}; expected (*, {len(columns)})"
            )
        labels = np.asarray(array[:, -2:])
        if not np.isfinite(labels).all() or not np.equal(labels, np.round(labels)).all():
            raise ValueError(f"Feature matrix {path.name!r} has invalid labels")
        size_bytes = path.stat().st_size
        entries.append(
            {
                "relative_name": path.name,
                "participant_id": participant,
                "shape": list(array.shape),
                "dtype": str(array.dtype),
                "window_count": int(array.shape[0]),
                "size_bytes": size_bytes,
                "sha256": _sha256_file(path) if calculate_checksums else None,
            }
        )
        total_windows += int(array.shape[0])
    if not entries:
        raise ValueError(f"No participant feature matrices found in {root}")

    metadata_entry = {
        "relative_name": metadata_path.name,
        "size_bytes": metadata_path.stat().st_size,
        "sha256": _sha256_file(metadata_path) if calculate_checksums else None,
    }
    digest_entries = [*entries, metadata_entry]
    return {
        "root_label": "candidate_tsfel_feature_matrices_no_window_scaling",
        "participant_count": len(entries),
        "participants": sorted(entry["participant_id"] for entry in entries),
        "candidate_feature_count": len(columns) - 2,
        "column_count_including_labels": len(columns),
        "feature_columns": columns[:-2],
        "label_columns": columns[-2:],
        "total_window_count": total_windows,
        "checksums_calculated": calculate_checksums,
        "content_sha256": _content_digest(digest_entries) if calculate_checksums else None,
        "metadata": metadata_entry,
        "matrices": entries,
    }


def build_candidate_snapshot_manifest(
    segment_root: Path,
    qa_summary_path: Path,
    feature_root: Path,
    *,
    calculate_checksums: bool = True,
) -> dict[str, Any]:
    """Build a comparison-ready manifest for a non-authoritative dataset snapshot."""
    segment_root = segment_root.resolve()
    qa_summary_path = qa_summary_path.resolve()
    feature_root = feature_root.resolve()
    if not segment_root.is_dir():
        raise FileNotFoundError(f"Segment root does not exist: {segment_root}")
    if not qa_summary_path.is_file():
        raise FileNotFoundError(f"QA summary does not exist: {qa_summary_path}")
    if not feature_root.is_dir():
        raise FileNotFoundError(f"Feature root does not exist: {feature_root}")

    qa_summary, qa_status = _load_qa_summary(qa_summary_path)
    segments = _audit_segments(
        segment_root,
        qa_status,
        calculate_checksums=calculate_checksums,
    )
    features = _audit_feature_matrices(
        feature_root,
        calculate_checksums=calculate_checksums,
    )
    participants = sorted(set(segments["participants"]) | set(features["participants"]))
    published_alignment = {
        "participant_count_is_20": len(participants) == 20,
        "includes_P019_and_P020": {"P019", "P020"} <= set(participants),
        "evaluated_segment_count_is_696": qa_summary["evaluated_segment_count"] == 696,
        "retained_segment_count_is_667": qa_summary["retained_segment_count"] == 667,
        "rejected_segment_count_is_29": qa_summary["rejected_segment_count"] == 29,
        "total_duration_rounds_to_51_2_hours": round(qa_summary["total_duration_hours"], 1)
        == 51.2,
        "rejected_duration_rounds_to_37_6_minutes": round(
            qa_summary["rejected_duration_minutes"], 1
        )
        == 37.6,
        "qa_and_array_counts_agree": (
            qa_summary["evaluated_segment_count"] == segments["array_count"]
            and qa_summary["retained_segment_count"] == segments["retained_array_count"]
            and qa_summary["rejected_segment_count"] == segments["rejected_array_count"]
        ),
    }

    source_entries = [
        {
            "relative_name": f"qa/{qa_summary['relative_name']}",
            "size_bytes": qa_summary["size_bytes"],
            "sha256": qa_summary["sha256"] if calculate_checksums else None,
        },
        *[
            {**entry, "relative_name": f"segments/{entry['relative_name']}"}
            for entry in segments["arrays"]
        ],
        {
            **features["metadata"],
            "relative_name": f"features/{features['metadata']['relative_name']}",
        },
        *[
            {**entry, "relative_name": f"features/{entry['relative_name']}"}
            for entry in features["matrices"]
        ],
    ]
    return {
        "schema_version": 1,
        "status": "candidate_snapshot_not_authoritative",
        "scientific_training_authorized": False,
        "holdout_accessed": False,
        "interpretation": (
            "Numerical agreement with the conference report is strong lineage evidence, "
            "but does not establish that this is the final approved dataset version."
        ),
        "participants": participants,
        "quality_summary": qa_summary,
        "segments": segments,
        "feature_matrices": features,
        "published_alignment": published_alignment,
        "all_published_alignment_checks_pass": all(published_alignment.values()),
        "checksums_calculated": calculate_checksums,
        "snapshot_content_sha256": (
            _content_digest(source_entries) if calculate_checksums else None
        ),
    }
