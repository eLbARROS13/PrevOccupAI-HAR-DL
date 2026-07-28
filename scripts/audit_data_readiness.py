#!/usr/bin/env python3
"""Audit whether a raw muscleBAN snapshot is ready for leakage-safe modeling."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Inspect participant, activity, and muscleBAN header coverage. The audit "
            "is read-only and fails closed when cohort or provenance requirements are missing."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/mban_protocol.json"),
        help="Protocol configuration JSON.",
    )
    parser.add_argument(
        "--raw-root",
        type=Path,
        help="Override the raw-data root specified by the protocol configuration.",
    )
    parser.add_argument("--output", type=Path, help="Optional output JSON path.")
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    """Load a JSON object and reject other top-level types."""
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object in {path}")
    return value


def read_opensignals_header(path: Path) -> dict[str, Any]:
    """Read the JSON header from an OpenSignals text file without loading samples."""
    with path.open("r", encoding="utf-8", errors="replace") as stream:
        first_line = stream.readline()
        second_line = stream.readline()
    if not first_line.startswith("# OpenSignals") or not second_line.startswith("# "):
        raise ValueError("unrecognised OpenSignals header")
    header = json.loads(second_line[2:])
    if not isinstance(header, dict) or not header:
        raise ValueError("empty OpenSignals device header")
    device_metadata = next(iter(header.values()))
    if not isinstance(device_metadata, dict):
        raise ValueError("invalid OpenSignals device metadata")
    return device_metadata


def activity_base(directory_name: str, required_bases: set[str]) -> str | None:
    """Map repeated activity directories such as walking_2 to their base activity."""
    for base in required_bases:
        if directory_name == base or re.fullmatch(rf"{re.escape(base)}_[0-9]+", directory_name):
            return base
    return None


def audit_snapshot(config_path: Path, raw_root_override: Path | None) -> dict[str, Any]:
    """Audit the raw snapshot using only directory names and recording headers."""
    config = load_json(config_path)
    raw_root = raw_root_override or Path(str(config["raw_data_root"]))
    raw_root = raw_root.resolve()
    if not raw_root.is_dir():
        raise FileNotFoundError(f"Raw-data root does not exist: {raw_root}")

    participant_pattern = re.compile(str(config["participant_id_pattern"]))
    muscleban_pattern = re.compile(str(config["muscleban_filename_pattern"]))
    development = set(map(str, config["development_participants"]))
    holdout = set(map(str, config["holdout_participants"]))
    expected_participants = development | holdout
    required_bases = set(map(str, config["required_activity_directory_bases"]))

    if development & holdout:
        raise ValueError(f"Participant split overlaps: {sorted(development & holdout)}")

    observed_participants = {
        path.name
        for path in raw_root.iterdir()
        if path.is_dir() and participant_pattern.fullmatch(path.name)
    }
    activity_coverage: dict[str, list[str]] = {}
    muscleban_files: list[Path] = []
    directories_without_two_mban_files: list[str] = []

    for participant in sorted(observed_participants):
        participant_path = raw_root / participant
        observed_bases: set[str] = set()
        for activity_directory in sorted(path for path in participant_path.iterdir() if path.is_dir()):
            base = activity_base(activity_directory.name, required_bases)
            if base is None:
                continue
            observed_bases.add(base)
            directory_mban_files = sorted(
                path
                for path in activity_directory.iterdir()
                if path.is_file() and muscleban_pattern.fullmatch(path.name)
            )
            muscleban_files.extend(directory_mban_files)
            if len(directory_mban_files) != 2:
                directories_without_two_mban_files.append(
                    f"{participant}/{activity_directory.name}: {len(directory_mban_files)}"
                )
        activity_coverage[participant] = sorted(observed_bases)

    sampling_rates: Counter[int | str] = Counter()
    accelerometer_channel_counts: Counter[int | str] = Counter()
    device_types: Counter[str] = Counter()
    device_identifiers: Counter[str] = Counter()
    invalid_header_count = 0
    for path in muscleban_files:
        try:
            metadata = read_opensignals_header(path)
            sampling_rates[metadata.get("sampling rate", "missing")] += 1
            sensors = metadata.get("sensor", [])
            accelerometer_channel_counts[
                sensors.count("gACC") if isinstance(sensors, list) else "missing"
            ] += 1
            device_types[str(metadata.get("device", "missing"))] += 1
            device_name = str(metadata.get("device name", "missing")).replace(":", "")
            device_identifiers[device_name] += 1
        except (OSError, ValueError, json.JSONDecodeError):
            invalid_header_count += 1

    participants_missing_activities = {
        participant: sorted(required_bases - set(coverage))
        for participant, coverage in activity_coverage.items()
        if required_bases - set(coverage)
    }
    missing_expected = sorted(expected_participants - observed_participants)
    unexpected = sorted(observed_participants - expected_participants)
    expected_sampling_rate = int(config["muscleban_sampling_rate_hz"])
    expected_accelerometer_channels = int(config["accelerometer_channels"])

    project_root = config_path.resolve().parent.parent

    def load_governance_configuration(key: str) -> tuple[dict[str, Any] | None, str | None]:
        """Load a configured governance JSON relative to the project root."""
        raw_path = config.get(key)
        if raw_path is None:
            return None, f"{key} is not configured"
        path = (project_root / str(raw_path)).resolve()
        try:
            return load_json(path), None
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as error:
            return None, f"{key} could not be loaded: {error}"

    signal_preprocessing, signal_preprocessing_error = load_governance_configuration(
        "signal_preprocessing_configuration"
    )
    segmentation_contract, segmentation_contract_error = load_governance_configuration(
        "segmentation_contract_configuration"
    )
    quality_manifest, quality_manifest_error = load_governance_configuration(
        "quality_assessment_manifest"
    )
    segment_manifest, segment_manifest_error = load_governance_configuration(
        "segmentation_manifest"
    )
    device_mapping, device_mapping_error = load_governance_configuration(
        "device_to_side_mapping"
    )

    development_manifest_valid = (
        segment_manifest is not None
        and segment_manifest.get("status") == "author_approved_development_dataset"
        and segment_manifest.get("authoritative") is True
        and segment_manifest.get("controls_dataset_generation") is True
        and segment_manifest.get("scientific_training_authorized") is True
        and segment_manifest.get("authorization_scope") == "development_selection_only"
        and segment_manifest.get("holdout_accessed") is False
        and set(map(str, segment_manifest.get("development_participants", []))) == development
        and set(map(str, segment_manifest.get("sealed_holdout_participants", []))) == holdout
    )
    quality_manifest_valid = (
        quality_manifest is not None
        and quality_manifest.get("authoritative") is True
        and quality_manifest.get("controls_dataset_generation") is True
        and quality_manifest.get("holdout_accessed") is False
        and quality_manifest.get("quality_summary", {}).get("evaluated_segment_count") == 696
        and quality_manifest.get("quality_summary", {}).get("retained_segment_count") == 667
        and quality_manifest.get("quality_summary", {}).get("rejected_segment_count") == 29
    )
    configured_device_mapping = (
        device_mapping.get("device_to_side", {}) if device_mapping is not None else {}
    )
    device_mapping_valid = (
        device_mapping is not None
        and device_mapping.get("authoritative") is True
        and set(configured_device_mapping) == set(device_identifiers)
        and set(map(str, configured_device_mapping.values())) == {"left", "right"}
    )

    blockers: list[str] = []
    warnings: list[str] = []
    if missing_expected:
        blockers.append(f"missing expected participants: {', '.join(missing_expected)}")
    if participants_missing_activities:
        message = "one or more participants lack complete raw activity-directory coverage"
        (warnings if development_manifest_valid else blockers).append(message)
    if directories_without_two_mban_files:
        message = "one or more raw activity directories contain repeated muscleBAN recordings"
        (warnings if development_manifest_valid else blockers).append(message)
    if invalid_header_count:
        blockers.append("one or more muscleBAN headers could not be parsed")
    if set(sampling_rates) != {expected_sampling_rate}:
        blockers.append("observed muscleBAN sampling rates do not match the configured value")
    if set(accelerometer_channel_counts) != {expected_accelerometer_channels}:
        blockers.append("observed accelerometer channel counts do not match the configured value")
    if quality_manifest_error is not None:
        blockers.append(quality_manifest_error)
    elif not quality_manifest_valid:
        blockers.append("quality-assessment manifest is not an approved exact snapshot")
    if segment_manifest_error is not None:
        blockers.append(segment_manifest_error)
    elif not development_manifest_valid:
        blockers.append("development segmentation manifest is not authoritative")
    if device_mapping_error is not None:
        blockers.append(device_mapping_error)
    elif not device_mapping_valid:
        blockers.append("device-to-side mapping does not cover the observed devices")
    if signal_preprocessing_error is not None:
        blockers.append(signal_preprocessing_error)
    elif not (
        signal_preprocessing.get("authoritative") is True
        and signal_preprocessing.get("controls_dataset_generation") is True
    ):
        blockers.append(
            "signal-preprocessing configuration is not authoritative for "
            "dataset generation"
        )
    if segmentation_contract_error is not None:
        blockers.append(segmentation_contract_error)
    elif not (
        segmentation_contract.get("authoritative") is True
        and segmentation_contract.get("controls_dataset_generation") is True
    ):
        blockers.append(
            "segmentation contract is not authoritative for dataset generation"
        )
    if config.get("training_authorized") is not True:
        blockers.append("protocol configuration does not authorize training")

    return {
        "schema_version": 1,
        "config_path": str(config_path.resolve()),
        "raw_root": str(raw_root),
        "development_participants": sorted(development),
        "holdout_participants": sorted(holdout),
        "observed_participants": sorted(observed_participants),
        "missing_expected_participants": missing_expected,
        "unexpected_participants": unexpected,
        "activity_coverage": activity_coverage,
        "participants_missing_activities": participants_missing_activities,
        "muscleban_file_count": len(muscleban_files),
        "activity_directories_without_exactly_two_muscleban_files": directories_without_two_mban_files,
        "sampling_rate_counts": {str(key): value for key, value in sorted(sampling_rates.items(), key=lambda item: str(item[0]))},
        "accelerometer_channel_count_distribution": {
            str(key): value
            for key, value in sorted(accelerometer_channel_counts.items(), key=lambda item: str(item[0]))
        },
        "device_type_counts": dict(sorted(device_types.items())),
        "invalid_header_count": invalid_header_count,
        "governance_configuration_status": {
            "quality_assessment": {
                "configured": quality_manifest is not None,
                "authoritative": quality_manifest_valid,
            },
            "approved_development_segments": {
                "configured": segment_manifest is not None,
                "authoritative": development_manifest_valid,
                "holdout_accessed": (
                    segment_manifest.get("holdout_accessed")
                    if segment_manifest is not None
                    else None
                ),
            },
            "device_to_side": {
                "configured": device_mapping is not None,
                "authoritative": device_mapping_valid,
                "observed_device_count": len(device_identifiers),
            },
            "signal_preprocessing": {
                "configured": signal_preprocessing is not None,
                "authoritative": (
                    signal_preprocessing.get("authoritative") is True
                    if signal_preprocessing is not None
                    else False
                ),
                "controls_dataset_generation": (
                    signal_preprocessing.get("controls_dataset_generation") is True
                    if signal_preprocessing is not None
                    else False
                ),
            },
            "segmentation": {
                "configured": segmentation_contract is not None,
                "authoritative": (
                    segmentation_contract.get("authoritative") is True
                    if segmentation_contract is not None
                    else False
                ),
                "controls_dataset_generation": (
                    segmentation_contract.get("controls_dataset_generation") is True
                    if segmentation_contract is not None
                    else False
                ),
            },
        },
        "train_ready": not blockers,
        "blockers": blockers,
        "warnings": warnings,
        "privacy_note": (
            "The report intentionally omits device identifiers, recording timestamps, and sample data."
        ),
    }


def main() -> None:
    """Run the readiness audit and write JSON if requested."""
    args = parse_args()
    result = audit_snapshot(args.config, args.raw_root)
    rendered = json.dumps(result, indent=2, sort_keys=True)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    raise SystemExit(0 if result["train_ready"] else 2)


if __name__ == "__main__":
    main()
