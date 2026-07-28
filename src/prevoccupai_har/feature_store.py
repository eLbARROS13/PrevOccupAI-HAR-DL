"""Hold-out-sealed loader for approved development TSFEL matrices."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np
from numpy.typing import NDArray

from .protocol import ProtocolConfiguration
from .provenance import sha256_canonical_json, sha256_file


MAIN_LABELS = {0: "sitting", 1: "standing", 2: "walking"}
SUBACTIVITY_LABELS = {
    0: "sitting_desk_work",
    3: "standing_still",
    4: "standing_conversing",
    5: "cabinets_coffee_tea",
    6: "cabinets_folders",
    7: "walking_slow",
    8: "walking_medium",
    9: "walking_fast",
    10: "stairs_up",
    11: "stairs_down",
}
SUBACTIVITY_MAIN_LABEL = {
    0: 0,
    3: 1,
    4: 1,
    5: 1,
    6: 1,
    7: 2,
    8: 2,
    9: 2,
    10: 2,
    11: 2,
}


@dataclass(frozen=True)
class DevelopmentFeatureMatrix:
    """Aligned in-memory features and pseudonymous development metadata."""

    features: NDArray[np.float64]
    labels: tuple[str, ...]
    subactivity_labels: tuple[str, ...]
    participant_ids: tuple[str, ...]
    feature_names: tuple[str, ...]
    manifest: Mapping[str, object]


def _object(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object in {path}")
    return value


def load_approved_development_feature_matrix(
    *,
    feature_root: Path,
    candidate_manifest_path: Path,
    window_store_index_path: Path,
    protocol: ProtocolConfiguration,
) -> DevelopmentFeatureMatrix:
    """Load only development participant arrays and verify their frozen identities."""
    feature_root = feature_root.resolve()
    candidate_manifest_path = candidate_manifest_path.resolve()
    window_store_index_path = window_store_index_path.resolve()
    candidate = _object(candidate_manifest_path)
    window_store = _object(window_store_index_path)
    feature_section = candidate.get("feature_matrices")
    if not isinstance(feature_section, dict):
        raise TypeError("Candidate manifest feature_matrices must be an object")
    if feature_section.get("checksums_calculated") is not True:
        raise ValueError("Feature-matrix source manifest lacks checksums")
    if int(feature_section.get("candidate_feature_count", -1)) != 45:
        raise ValueError("Approved RF input requires 45 candidate features")
    feature_names = tuple(map(str, feature_section.get("feature_columns", ())))
    if len(feature_names) != 45 or len(set(feature_names)) != 45:
        raise ValueError("Candidate feature names are incomplete or duplicated")
    entries_value = feature_section.get("matrices")
    if not isinstance(entries_value, list):
        raise TypeError("Candidate feature matrices must be an array")
    entries = {
        str(entry["participant_id"]): entry
        for entry in entries_value
        if isinstance(entry, dict)
    }
    development = tuple(protocol.development_participants)
    holdout = frozenset(protocol.holdout_participants)
    if set(development) & holdout:
        raise ValueError("Protocol development and hold-out cohorts overlap")
    if set(development) - set(entries):
        raise ValueError("Candidate manifest lacks development feature matrices")
    expected_counts = {
        str(key): int(value)
        for key, value in dict(window_store["window_counts_per_participant"]).items()
    }
    if set(expected_counts) != set(development):
        raise ValueError("Window-store counts do not exactly describe development participants")

    feature_arrays: list[NDArray[np.float64]] = []
    labels: list[str] = []
    subactivities: list[str] = []
    participant_ids: list[str] = []
    file_records: list[dict[str, object]] = []
    for participant in development:
        if participant in holdout:
            raise PermissionError("Hold-out feature values cannot be loaded")
        entry = entries[participant]
        path = feature_root / str(entry["relative_name"])
        if path.name != f"{participant}.npy":
            raise ValueError("Feature filename and participant identity disagree")
        if path.stat().st_size != int(entry["size_bytes"]):
            raise ValueError("Feature matrix size differs from the frozen candidate")
        digest = sha256_file(path)
        if digest != str(entry["sha256"]):
            raise ValueError("Feature matrix hash differs from the frozen candidate")
        values = np.load(path, allow_pickle=False)
        expected_shape = tuple(map(int, entry["shape"]))
        if values.shape != expected_shape or values.dtype != np.float64:
            raise ValueError("Feature matrix shape or dtype differs from the frozen candidate")
        if values.shape[1] != 47 or not np.isfinite(values).all():
            raise ValueError("Feature matrix must contain 45 finite features and two labels")
        if values.shape[0] != expected_counts[participant]:
            raise ValueError("Feature and approved raw-window counts disagree")
        main_values = values[:, 45]
        sub_values = values[:, 46]
        if not np.equal(main_values, np.rint(main_values)).all() or not np.equal(
            sub_values, np.rint(sub_values)
        ).all():
            raise ValueError("Feature label columns must contain exact integer codes")
        main_codes = main_values.astype(np.int64)
        sub_codes = sub_values.astype(np.int64)
        if set(main_codes.tolist()) - set(MAIN_LABELS) or set(sub_codes.tolist()) - set(
            SUBACTIVITY_LABELS
        ):
            raise ValueError("Feature matrix contains an unknown activity label")
        if any(
            SUBACTIVITY_MAIN_LABEL[int(sub)] != int(main)
            for main, sub in zip(main_codes, sub_codes, strict=True)
        ):
            raise ValueError("Main and sub-activity feature labels disagree")
        feature_arrays.append(np.asarray(values[:, :45], dtype=np.float64))
        labels.extend(MAIN_LABELS[int(code)] for code in main_codes)
        subactivities.extend(SUBACTIVITY_LABELS[int(code)] for code in sub_codes)
        participant_ids.extend([participant] * values.shape[0])
        file_records.append(
            {
                "participant_id": participant,
                "relative_name": path.name,
                "shape": list(values.shape),
                "dtype": str(values.dtype),
                "size_bytes": path.stat().st_size,
                "sha256": digest,
            }
        )
    features = np.ascontiguousarray(np.vstack(feature_arrays), dtype=np.float64)
    if set(participant_ids) != set(development) or set(participant_ids) & holdout:
        raise PermissionError("Loaded feature cohort violates the sealed hold-out")
    class_counts = Counter(labels)
    payload: dict[str, object] = {
        "authorization_scope": "development_selection_only",
        "scientific_training_authorized": True,
        "holdout_values_loaded": False,
        "sealed_holdout_participants": sorted(holdout),
        "development_participants": list(development),
        "candidate_manifest_sha256": sha256_file(candidate_manifest_path),
        "window_store_index_sha256": sha256_file(window_store_index_path),
        "feature_names": list(feature_names),
        "files": file_records,
        "row_count": int(features.shape[0]),
        "column_count": int(features.shape[1]),
        "class_counts": dict(sorted(class_counts.items())),
    }
    manifest = {
        "schema_version": 1,
        "status": "approved_development_feature_matrix",
        **payload,
        "payload_sha256": sha256_canonical_json(payload),
    }
    return DevelopmentFeatureMatrix(
        features=features,
        labels=tuple(labels),
        subactivity_labels=tuple(subactivities),
        participant_ids=tuple(participant_ids),
        feature_names=feature_names,
        manifest=manifest,
    )


def write_development_feature_manifest(
    path: Path | str,
    dataset: DevelopmentFeatureMatrix,
) -> None:
    """Write the path-free development feature manifest exclusively."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(dataset.manifest, stream, allow_nan=False, indent=2, sort_keys=True)
        stream.write("\n")
