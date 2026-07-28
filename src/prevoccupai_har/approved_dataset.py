"""Approved processed-segment provenance for development-only HAR execution.

The approved analysis source uses QA-filtered NumPy segment arrays.  Raw
OpenSignals recordings are consulted only to restore the physical device and
sensor-side link that was lost when the historical segmenter emitted generic
``file01``/``file02`` identifiers.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .provenance import sha256_canonical_json, sha256_file


SEGMENT_FILENAME_PATTERN = re.compile(
    r"^(?P<participant>P[0-9]{3})_"
    r"(?P<activity>[a-z]+[0-9]?)_"
    r"(?P<file_token>file[0-9]{2})_"
    r"(?P<label>.+?)_GlobalSegment[0-9]+"
    r"(?:_LocalSegment[0-9]+)?(?P<failed>_failed)?[.]npy$"
)
RAW_MBAN_PATTERN = re.compile(
    r"^opensignals_(?P<device>[0-9A-F]{12})_.+[.]txt$"
)
ACCELERATION_OF_GRAVITY_M_S2 = 9.80665
ADC_BITS = 16
ACCELEROMETER_RANGE_G = 8


@dataclass(frozen=True, order=True)
class SegmentGroupKey:
    """Historical file token within one participant/activity recording."""

    participant_id: str
    activity_token: str
    file_token: str


@dataclass(frozen=True)
class ParsedSegmentName:
    """Scientific labels and source tokens parsed from a segment filename."""

    participant_id: str
    activity_token: str
    file_token: str
    raw_sub_activity_label: str
    main_label: str
    sub_activity_label: str
    quality_status: str

    @property
    def group_key(self) -> SegmentGroupKey:
        return SegmentGroupKey(
            self.participant_id,
            self.activity_token,
            self.file_token,
        )


def _normalise_sub_activity(raw_label: str) -> tuple[str, str]:
    """Map recovered filename labels to the frozen three-class vocabulary."""
    if raw_label == "sitting":
        return "sitting", "sitting_desk_work"
    if raw_label.startswith("stand_still_"):
        return "standing", "standing_still"
    if raw_label == "stand_conversing":
        return "standing", "standing_conversing"
    if raw_label == "drink_coffee":
        return "standing", "cabinets_coffee_tea"
    if raw_label == "moving_objects":
        return "standing", "cabinets_shelf_organization"
    if raw_label in {"walk_slow", "walk_medium", "walk_fast"}:
        return "walking", f"walking_{raw_label.removeprefix('walk_')}"
    if re.fullmatch(r"stairs_up_[1-4]", raw_label):
        return "walking", "stairs_up"
    if re.fullmatch(r"stairs_down_[1-4]", raw_label):
        return "walking", "stairs_down"
    raise ValueError(f"Unsupported segment sub-activity label: {raw_label!r}")


def parse_segment_filename(filename: str) -> ParsedSegmentName:
    """Parse one approved segment filename without reading participant data."""
    match = SEGMENT_FILENAME_PATTERN.fullmatch(filename)
    if match is None:
        raise ValueError(f"Unexpected approved segment filename: {filename!r}")
    main_label, sub_activity_label = _normalise_sub_activity(match.group("label"))
    return ParsedSegmentName(
        participant_id=match.group("participant"),
        activity_token=match.group("activity"),
        file_token=match.group("file_token"),
        raw_sub_activity_label=match.group("label"),
        main_label=main_label,
        sub_activity_label=sub_activity_label,
        quality_status="BAD" if match.group("failed") else "GOOD",
    )


def normalise_raw_activity_directory(directory_name: str) -> str:
    """Convert raw directory suffixes to historical segment activity tokens."""
    match = re.fullmatch(r"(?P<base>[a-z]+)(?:_(?P<repeat>[1-9][0-9]*))?", directory_name)
    if match is None:
        raise ValueError(f"Unexpected raw activity directory: {directory_name!r}")
    repeat = match.group("repeat")
    if repeat in (None, "1"):
        return match.group("base")
    return f"{match.group('base')}{repeat}"


def _adc_from_acceleration(value_m_s2: float) -> int:
    """Invert the historical 16-bit, plus/minus-8g conversion exactly."""
    midpoint = 2**ADC_BITS / 2
    full_span_g = 2 * ACCELEROMETER_RANGE_G
    adc_value = round(
        value_m_s2
        / ACCELERATION_OF_GRAVITY_M_S2
        * (2**ADC_BITS)
        / full_span_g
        + midpoint
    )
    reconstructed = (
        (adc_value - midpoint)
        * (full_span_g / (2**ADC_BITS))
        * ACCELERATION_OF_GRAVITY_M_S2
    )
    if not math.isclose(reconstructed, value_m_s2, rel_tol=0.0, abs_tol=5e-10):
        raise ValueError("Segment acceleration is not an exact historical ADC conversion")
    return adc_value


RawSignature = tuple[int, int, int, int]


def _signature_from_segment_row(row: np.ndarray) -> RawSignature:
    if row.shape != (4,) or not np.isfinite(row).all():
        raise ValueError("Segment signature rows must contain four finite values")
    nseq = int(round(float(row[0])))
    if not math.isclose(float(row[0]), nseq, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError("Segment nSeq values must be integral")
    return (
        nseq,
        _adc_from_acceleration(float(row[1])),
        _adc_from_acceleration(float(row[2])),
        _adc_from_acceleration(float(row[3])),
    )


def _sample_group_signatures(
    paths: Sequence[Path],
    *,
    maximum_signatures: int = 12,
) -> tuple[RawSignature, ...]:
    """Select deterministic signatures across a group's exported segments."""
    signatures: list[RawSignature] = []
    seen: set[RawSignature] = set()
    for path in sorted(paths):
        values = np.load(path, mmap_mode="r", allow_pickle=False)
        if values.ndim != 2 or values.shape[1] != 4 or values.shape[0] == 0:
            raise ValueError(f"Invalid approved segment shape for {path.name}: {values.shape}")
        indices = sorted(
            {
                0,
                values.shape[0] // 4,
                values.shape[0] // 2,
                (3 * values.shape[0]) // 4,
                values.shape[0] - 1,
            }
        )
        for index in indices:
            signature = _signature_from_segment_row(np.asarray(values[index]))
            if signature not in seen:
                signatures.append(signature)
                seen.add(signature)
            if len(signatures) >= maximum_signatures:
                return tuple(signatures)
    if len(signatures) < 3:
        raise ValueError("At least three unique raw signatures are required per file token")
    return tuple(signatures)


def _activity_directory(raw_root: Path, key: SegmentGroupKey) -> Path:
    participant_root = raw_root / key.participant_id
    matches = [
        path
        for path in participant_root.iterdir()
        if path.is_dir() and normalise_raw_activity_directory(path.name) == key.activity_token
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Expected one raw directory for {key}; found {[path.name for path in matches]}"
        )
    return matches[0]


def _parse_raw_signature(line: bytes) -> RawSignature | None:
    if not line or line.startswith(b"#"):
        return None
    fields = line.rstrip(b"\r\n").split(b"\t")
    if len(fields) < 6:
        return None
    try:
        return tuple(int(fields[index]) for index in (0, 3, 4, 5))  # type: ignore[return-value]
    except ValueError:
        return None


def _scan_raw_recording(
    path: Path,
    signature_lookup: Mapping[RawSignature, tuple[SegmentGroupKey, ...]],
) -> tuple[dict[SegmentGroupKey, set[RawSignature]], str, int]:
    """Scan a raw recording once, matching signatures while hashing its bytes."""
    matches: dict[SegmentGroupKey, set[RawSignature]] = defaultdict(set)
    digest = hashlib.sha256()
    sample_count = 0
    with path.open("rb") as stream:
        for line in stream:
            digest.update(line)
            signature = _parse_raw_signature(line)
            if signature is None:
                continue
            sample_count += 1
            for key in signature_lookup.get(signature, ()):
                matches[key].add(signature)
    return matches, digest.hexdigest(), sample_count


def recover_development_device_provenance(
    segment_root: Path,
    raw_root: Path,
    development_participants: Sequence[str],
    device_to_side: Mapping[str, str],
) -> dict[str, Any]:
    """Recover development file-token mappings through exact signal signatures."""
    segment_root = segment_root.resolve()
    raw_root = raw_root.resolve()
    development = tuple(sorted(map(str, development_participants)))
    holdout_values_loaded = False
    grouped_paths: dict[SegmentGroupKey, list[Path]] = defaultdict(list)
    for path in sorted(segment_root.glob("P*.npy")):
        parsed = parse_segment_filename(path.name)
        if parsed.participant_id in development:
            grouped_paths[parsed.group_key].append(path)

    observed = {key.participant_id for key in grouped_paths}
    if observed != set(development):
        raise ValueError(
            f"Development segment cohort mismatch: observed={sorted(observed)}, expected={list(development)}"
        )

    signatures_by_group = {
        key: _sample_group_signatures(paths)
        for key, paths in sorted(grouped_paths.items())
    }
    groups_by_directory: dict[Path, list[SegmentGroupKey]] = defaultdict(list)
    for key in signatures_by_group:
        groups_by_directory[_activity_directory(raw_root, key)].append(key)

    raw_records: dict[Path, dict[str, Any]] = {}
    matched_signatures: dict[SegmentGroupKey, dict[Path, set[RawSignature]]] = defaultdict(dict)
    for directory, keys in sorted(groups_by_directory.items(), key=lambda item: str(item[0])):
        lookup_values: dict[RawSignature, list[SegmentGroupKey]] = defaultdict(list)
        for key in keys:
            for signature in signatures_by_group[key]:
                lookup_values[signature].append(key)
        signature_lookup = {
            signature: tuple(group_keys)
            for signature, group_keys in lookup_values.items()
        }
        raw_paths = sorted(
            path
            for path in directory.iterdir()
            if path.is_file() and RAW_MBAN_PATTERN.fullmatch(path.name)
        )
        if not raw_paths:
            raise ValueError(f"No raw mBAN files found in {directory}")
        for raw_path in raw_paths:
            matches, content_sha256, sample_count = _scan_raw_recording(
                raw_path,
                signature_lookup,
            )
            raw_match = RAW_MBAN_PATTERN.fullmatch(raw_path.name)
            assert raw_match is not None
            device_id = raw_match.group("device")
            if device_id not in device_to_side:
                raise ValueError(f"No approved side mapping for device {device_id}")
            raw_records[raw_path] = {
                "raw_recording_name_sha256": hashlib.sha256(
                    str(raw_path.relative_to(raw_root)).encode("utf-8")
                ).hexdigest(),
                "raw_content_sha256": content_sha256,
                "raw_size_bytes": raw_path.stat().st_size,
                "raw_sample_count": sample_count,
                "device_stream_id": device_id,
                "sensor_side": device_to_side[device_id],
            }
            for key, values in matches.items():
                matched_signatures[key][raw_path] = values

    mappings: list[dict[str, Any]] = []
    for key, signatures in sorted(signatures_by_group.items()):
        scores = sorted(
            (
                (len(values), raw_path)
                for raw_path, values in matched_signatures.get(key, {}).items()
            ),
            key=lambda item: (-item[0], str(item[1])),
        )
        if not scores:
            raise ValueError(f"No raw recording matched signatures for {key}")
        best_count, best_path = scores[0]
        second_count = scores[1][0] if len(scores) > 1 else 0
        required_count = max(3, math.ceil(0.75 * len(signatures)))
        if best_count < required_count or best_count == second_count:
            raise ValueError(
                f"Ambiguous raw mapping for {key}: best={best_count}, second={second_count}, "
                f"required={required_count}, signatures={len(signatures)}"
            )
        raw_record = raw_records[best_path]
        mappings.append(
            {
                "participant_id": key.participant_id,
                "activity_token": key.activity_token,
                "file_token": key.file_token,
                "raw_recording_name_sha256": raw_record["raw_recording_name_sha256"],
                "raw_content_sha256": raw_record["raw_content_sha256"],
                "raw_size_bytes": raw_record["raw_size_bytes"],
                "raw_sample_count": raw_record["raw_sample_count"],
                "device_stream_id": raw_record["device_stream_id"],
                "sensor_side": raw_record["sensor_side"],
                "selected_signature_count": len(signatures),
                "matched_signature_count": best_count,
                "second_best_match_count": second_count,
                "evidence": "exact_nseq_and_inverse_adc_signature_match",
            }
        )

    payload = {
        "schema_version": 1,
        "status": "validated_development_device_provenance",
        "authoritative": True,
        "controls_development_dataset_generation": True,
        "holdout_values_loaded": holdout_values_loaded,
        "development_participants": list(development),
        "group_count": len(mappings),
        "all_groups_resolved": len(mappings) == len(grouped_paths),
        "mappings": mappings,
    }
    payload["payload_sha256"] = sha256_canonical_json(payload)
    return payload


def _load_json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object in {path}")
    return value


def build_approved_development_manifest(
    candidate_manifest_path: Path,
    approval_path: Path,
    device_provenance: Mapping[str, Any],
) -> dict[str, Any]:
    """Bind author approval, frozen segment hashes, QA, labels, and side mappings."""
    candidate = _load_json_object(candidate_manifest_path)
    approval = _load_json_object(approval_path)
    expected_snapshot_sha = str(approval["approved_snapshot_content_sha256"])
    if candidate.get("snapshot_content_sha256") != expected_snapshot_sha:
        raise ValueError("Approved and observed snapshot content digests differ")
    if candidate.get("all_published_alignment_checks_pass") is not True:
        raise ValueError("Approved snapshot does not reproduce the published aggregate checks")
    if approval.get("scientific_training_authorized_development") is not True:
        raise PermissionError("Approval does not authorize development training")
    if approval.get("holdout_access_authorized") is not False:
        raise ValueError("Development approval must leave hold-out access disabled")
    if device_provenance.get("authoritative") is not True:
        raise ValueError("Device provenance is not authoritative")
    if device_provenance.get("holdout_values_loaded") is not False:
        raise PermissionError("Development provenance must not load hold-out values")

    development = tuple(sorted(map(str, approval["development_participants"])))
    holdout = tuple(sorted(map(str, approval["holdout_participants"])))
    if set(development) & set(holdout):
        raise ValueError("Approved development and hold-out cohorts overlap")
    mapping_by_group = {
        SegmentGroupKey(
            str(entry["participant_id"]),
            str(entry["activity_token"]),
            str(entry["file_token"]),
        ): entry
        for entry in device_provenance["mappings"]
    }

    segment_entries: list[dict[str, Any]] = []
    for source_entry in candidate["segments"]["arrays"]:
        parsed = parse_segment_filename(str(source_entry["relative_name"]))
        if parsed.participant_id not in development:
            continue
        mapping = mapping_by_group.get(parsed.group_key)
        if mapping is None:
            raise ValueError(f"Missing device provenance for {parsed.group_key}")
        if parsed.quality_status != source_entry["quality_status"]:
            raise ValueError(f"Filename/manifest QA mismatch for {source_entry['relative_name']}")
        segment_entries.append(
            {
                "relative_name": source_entry["relative_name"],
                "participant_id": parsed.participant_id,
                "activity_token": parsed.activity_token,
                "file_token": parsed.file_token,
                "main_label": parsed.main_label,
                "sub_activity_label": parsed.sub_activity_label,
                "raw_sub_activity_label": parsed.raw_sub_activity_label,
                "quality_status": parsed.quality_status,
                "shape": source_entry["shape"],
                "dtype": source_entry["dtype"],
                "size_bytes": source_entry["size_bytes"],
                "sha256": source_entry["sha256"],
                "recording_id": hashlib.sha256(
                    str(source_entry["relative_name"]).encode("utf-8")
                ).hexdigest(),
                "device_stream_id": mapping["device_stream_id"],
                "sensor_side": mapping["sensor_side"],
                "side_mapping_evidence": mapping["evidence"],
            }
        )

    status_counts: dict[str, int] = defaultdict(int)
    for entry in segment_entries:
        status_counts[str(entry["quality_status"])] += 1
    source_identity = {
        "candidate_manifest_sha256": sha256_file(candidate_manifest_path),
        "approved_snapshot_content_sha256": expected_snapshot_sha,
        "approved_segment_content_sha256": candidate["segments"]["content_sha256"],
        "approved_qa_sha256": candidate["quality_summary"]["sha256"],
        "device_provenance_payload_sha256": device_provenance["payload_sha256"],
    }
    payload = {
        "schema_version": 1,
        "status": "author_approved_development_dataset",
        "authoritative": True,
        "controls_dataset_generation": True,
        "scientific_training_authorized": True,
        "authorization_scope": "development_selection_only",
        "holdout_accessed": False,
        "holdout_access_authorized": False,
        "approval_record": {
            "approved_on": approval["approved_on"],
            "approval_basis": approval["approval_basis"],
            "approval_sha256": sha256_file(approval_path),
        },
        "development_participants": list(development),
        "sealed_holdout_participants": list(holdout),
        "source_identity": source_identity,
        "quality_summary": candidate["quality_summary"],
        "development_segment_count": len(segment_entries),
        "retained_development_segment_count": status_counts.get("GOOD", 0),
        "rejected_development_segment_count": status_counts.get("BAD", 0),
        "device_provenance_group_count": device_provenance["group_count"],
        "segments": segment_entries,
        "notes": list(map(str, approval.get("notes", []))),
    }
    payload["payload_sha256"] = sha256_canonical_json(payload)
    return payload


def write_json_exclusive(path: Path, payload: Mapping[str, Any]) -> None:
    """Write deterministic JSON without replacing existing evidence."""
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite existing manifest: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
