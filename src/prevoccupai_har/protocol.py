"""Validated protocol configuration for muscleBAN HAR experiments."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


@dataclass(frozen=True)
class WindowSpecification:
    """Sliding-window parameters expressed in samples and physical time."""

    duration_seconds: float
    overlap_fraction: float
    expected_samples: int

    def validate(self, sampling_rate_hz: int) -> None:
        """Reject internally inconsistent or unusable window definitions."""
        if self.duration_seconds <= 0:
            raise ValueError("Window duration must be positive")
        if not 0 <= self.overlap_fraction < 1:
            raise ValueError("Window overlap must be in [0, 1)")
        if self.expected_samples <= 0:
            raise ValueError("Expected window samples must be positive")
        calculated_samples = round(self.duration_seconds * sampling_rate_hz)
        if calculated_samples != self.expected_samples:
            raise ValueError(
                "Window duration, sampling rate, and expected sample count disagree: "
                f"{calculated_samples} != {self.expected_samples}"
            )
        if self.step_samples <= 0:
            raise ValueError("Window overlap produces a non-positive step")

    @property
    def step_samples(self) -> int:
        """Return the integer stride between consecutive windows."""
        return round(self.expected_samples * (1 - self.overlap_fraction))


@dataclass(frozen=True)
class ProtocolConfiguration:
    """Scientific and governance constraints loaded from the protocol JSON."""

    schema_version: int
    dataset_name: str
    source_status: str
    raw_data_root: Path
    participant_id_pattern: str
    development_participants: tuple[str, ...]
    holdout_participants: tuple[str, ...]
    required_activity_directory_bases: tuple[str, ...]
    main_labels: tuple[str, ...]
    muscleban_filename_pattern: str
    sampling_rate_hz: int
    accelerometer_channels: int
    window: WindowSpecification
    quality_assessment_manifest: Path | None
    segmentation_manifest: Path | None
    device_to_side_mapping: Path | None
    signal_preprocessing_configuration: Path | None
    segmentation_contract_configuration: Path | None
    training_authorized: bool
    training_authorization_scope: str
    holdout_access_authorized: bool
    training_blockers: tuple[str, ...]

    @property
    def all_expected_participants(self) -> tuple[str, ...]:
        """Return the complete, unique expected cohort in lexical order."""
        return tuple(sorted(set(self.development_participants) | set(self.holdout_participants)))

    def validate(self) -> None:
        """Validate split, label, sampling, and governance invariants."""
        if self.schema_version != 1:
            raise ValueError(f"Unsupported protocol schema version: {self.schema_version}")
        if self.sampling_rate_hz <= 0:
            raise ValueError("Sampling rate must be positive")
        if self.accelerometer_channels != 3:
            raise ValueError("The current primary protocol requires tri-axial ACC")
        if len(set(self.main_labels)) != len(self.main_labels) or not self.main_labels:
            raise ValueError("Main labels must be non-empty and unique")
        if len(set(self.required_activity_directory_bases)) != len(
            self.required_activity_directory_bases
        ):
            raise ValueError("Activity-directory bases must be unique")

        development = set(self.development_participants)
        holdout = set(self.holdout_participants)
        overlap = development & holdout
        if overlap:
            raise ValueError(f"Development and hold-out participants overlap: {sorted(overlap)}")
        if len(development) != len(self.development_participants):
            raise ValueError("Development participants contain duplicates")
        if len(holdout) != len(self.holdout_participants):
            raise ValueError("Hold-out participants contain duplicates")

        participant_pattern = re.compile(self.participant_id_pattern)
        invalid_participants = [
            participant
            for participant in self.all_expected_participants
            if participant_pattern.fullmatch(participant) is None
        ]
        if invalid_participants:
            raise ValueError(f"Invalid participant identifiers: {invalid_participants}")
        re.compile(self.muscleban_filename_pattern)
        self.window.validate(self.sampling_rate_hz)

        if self.training_authorized:
            if self.training_authorization_scope != "development_selection_only":
                raise ValueError(
                    "Scientific training authorization must be development_selection_only"
                )
            if self.holdout_access_authorized:
                raise ValueError(
                    "The development protocol cannot authorize external hold-out access"
                )
            required_governance_paths = {
                "quality_assessment_manifest": self.quality_assessment_manifest,
                "segmentation_manifest": self.segmentation_manifest,
                "device_to_side_mapping": self.device_to_side_mapping,
                "signal_preprocessing_configuration": self.signal_preprocessing_configuration,
                "segmentation_contract_configuration": self.segmentation_contract_configuration,
            }
            missing_paths = [
                name for name, path in required_governance_paths.items() if path is None
            ]
            if missing_paths:
                raise ValueError(
                    "Training cannot be authorized without governance paths: "
                    f"{missing_paths}"
                )
            if self.training_blockers:
                raise ValueError(
                    "Training cannot be authorized while configured blockers remain"
                )

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any], *, base_path: Path = Path(".")) -> "ProtocolConfiguration":
        """Construct and validate a protocol from a decoded JSON mapping."""
        window_value = value.get("window")
        if not isinstance(window_value, Mapping):
            raise TypeError("Protocol field 'window' must be an object")

        def optional_path(key: str) -> Path | None:
            raw_value = value.get(key)
            if raw_value is None:
                return None
            return (base_path / str(raw_value)).resolve()

        protocol = cls(
            schema_version=int(value["schema_version"]),
            dataset_name=str(value["dataset_name"]),
            source_status=str(value["source_status"]),
            raw_data_root=(base_path / str(value["raw_data_root"])).resolve(),
            participant_id_pattern=str(value["participant_id_pattern"]),
            development_participants=tuple(map(str, value["development_participants"])),
            holdout_participants=tuple(map(str, value["holdout_participants"])),
            required_activity_directory_bases=tuple(
                map(str, value["required_activity_directory_bases"])
            ),
            main_labels=tuple(map(str, value["main_labels"])),
            muscleban_filename_pattern=str(value["muscleban_filename_pattern"]),
            sampling_rate_hz=int(value["muscleban_sampling_rate_hz"]),
            accelerometer_channels=int(value["accelerometer_channels"]),
            window=WindowSpecification(
                duration_seconds=float(window_value["duration_seconds"]),
                overlap_fraction=float(window_value["overlap_fraction"]),
                expected_samples=int(window_value["expected_samples"]),
            ),
            quality_assessment_manifest=optional_path("quality_assessment_manifest"),
            segmentation_manifest=optional_path("segmentation_manifest"),
            device_to_side_mapping=optional_path("device_to_side_mapping"),
            signal_preprocessing_configuration=optional_path(
                "signal_preprocessing_configuration"
            ),
            segmentation_contract_configuration=optional_path(
                "segmentation_contract_configuration"
            ),
            training_authorized=value.get("training_authorized") is True,
            training_authorization_scope=str(
                value.get("training_authorization_scope", "none")
            ),
            holdout_access_authorized=value.get("holdout_access_authorized") is True,
            training_blockers=tuple(map(str, value.get("training_blockers", []))),
        )
        protocol.validate()
        return protocol


def load_protocol(path: Path | str) -> ProtocolConfiguration:
    """Load a validated protocol JSON file."""
    protocol_path = Path(path).resolve()
    decoded = json.loads(protocol_path.read_text(encoding="utf-8"))
    if not isinstance(decoded, dict):
        raise TypeError(f"Expected a JSON object in {protocol_path}")
    return ProtocolConfiguration.from_mapping(decoded, base_path=protocol_path.parent.parent)
