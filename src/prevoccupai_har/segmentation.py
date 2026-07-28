"""Governed interval semantics for muscleBAN activity segmentation.

This module deliberately does not detect boundaries from participant signals.
It validates externally supplied half-open intervals, assigns only protocol-
supported labels, and applies the camera-ready edge-cropping rules.  The
current non-authoritative contract can be exercised for synthetic validation
but refuses scientific dataset generation.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any, Mapping, Sequence


class SegmentationPurpose(StrEnum):
    """Declared use of a segmentation-contract operation."""

    VALIDATION = "validation"
    SCIENTIFIC_DATASET = "scientific_dataset"


@dataclass(frozen=True)
class ActivitySegmentationProtocol:
    """Ordered sub-activity labels and edge crop for one recording type."""

    activity: str
    main_label: str
    ordered_subactivities: tuple[str, ...]
    edge_crop_seconds: float

    def __post_init__(self) -> None:
        """Reject empty or nonsensical protocol fields."""
        if not self.activity:
            raise ValueError("activity must not be empty")
        if self.main_label not in {"sitting", "standing", "walking"}:
            raise ValueError("main_label must be sitting, standing, or walking")
        if not self.ordered_subactivities or any(
            not label for label in self.ordered_subactivities
        ):
            raise ValueError("ordered_subactivities must contain non-empty labels")
        if self.edge_crop_seconds < 0:
            raise ValueError("edge_crop_seconds must be non-negative")


@dataclass(frozen=True)
class SegmentationContractConfiguration:
    """Versioned protocol and governance state for segment intervals."""

    schema_version: int
    name: str
    authoritative: bool
    controls_dataset_generation: bool
    automatic_boundary_detection_authorized: bool
    source: Mapping[str, Any]
    notes: tuple[str, ...]
    boundary_evidence: Mapping[str, Any]
    sampling_rate_hz: int
    interval_convention: str
    activity_protocols: Mapping[str, ActivitySegmentationProtocol]

    def __post_init__(self) -> None:
        """Validate governance, sampling, and activity-protocol invariants."""
        if self.schema_version != 1:
            raise ValueError(
                f"unsupported segmentation schema version: {self.schema_version}"
            )
        if self.controls_dataset_generation and not self.authoritative:
            raise ValueError(
                "a non-authoritative segmentation contract cannot control "
                "dataset generation"
            )
        if self.automatic_boundary_detection_authorized and not self.authoritative:
            raise ValueError(
                "automatic boundary detection cannot be authorized by a "
                "non-authoritative contract"
            )
        if self.sampling_rate_hz <= 0:
            raise ValueError("sampling_rate_hz must be positive")
        if self.interval_convention != "half_open":
            raise ValueError("only half_open segment intervals are supported")
        expected_activities = {"walking", "stairs", "cabinets", "standing", "sitting"}
        if set(self.activity_protocols) != expected_activities:
            raise ValueError(
                "activity protocols must cover walking, stairs, cabinets, "
                "standing, and sitting exactly"
            )
        for activity, protocol in self.activity_protocols.items():
            if protocol.activity != activity:
                raise ValueError("activity protocol key and activity field disagree")


@dataclass(frozen=True)
class RawSegmentInterval:
    """One externally supplied half-open interval in source-sample coordinates."""

    start_sample: int
    stop_sample: int

    def __post_init__(self) -> None:
        """Require non-negative integer bounds with positive duration."""
        if isinstance(self.start_sample, bool) or not isinstance(self.start_sample, int):
            raise TypeError("start_sample must be an integer")
        if isinstance(self.stop_sample, bool) or not isinstance(self.stop_sample, int):
            raise TypeError("stop_sample must be an integer")
        if self.start_sample < 0:
            raise ValueError("start_sample must be non-negative")
        if self.stop_sample <= self.start_sample:
            raise ValueError("stop_sample must be greater than start_sample")

    @property
    def sample_count(self) -> int:
        """Return interval length under half-open semantics."""
        return self.stop_sample - self.start_sample


@dataclass(frozen=True)
class GovernedSegmentInterval:
    """Protocol-labelled raw and retained bounds for one sub-activity bout."""

    activity: str
    main_label: str
    subactivity_label: str
    ordinal: int
    raw_start_sample: int
    raw_stop_sample: int
    retained_start_sample: int
    retained_stop_sample: int
    boundary_manifest_id: str
    configuration_name: str

    @property
    def raw_sample_count(self) -> int:
        """Return the source interval length before transition cropping."""
        return self.raw_stop_sample - self.raw_start_sample

    @property
    def retained_sample_count(self) -> int:
        """Return the interval length retained after edge cropping."""
        return self.retained_stop_sample - self.retained_start_sample


def load_segmentation_contract_configuration(
    path: Path | str,
) -> SegmentationContractConfiguration:
    """Load and validate a versioned segmentation contract from JSON."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("segmentation contract must be a JSON object")
    parameters = payload.get("parameters")
    activity_values = payload.get("activity_protocols")
    if not isinstance(parameters, dict) or not isinstance(activity_values, dict):
        raise TypeError(
            "segmentation contract must contain parameters and activity_protocols objects"
        )

    protocols: dict[str, ActivitySegmentationProtocol] = {}
    for activity, value in activity_values.items():
        if not isinstance(value, dict):
            raise TypeError(f"activity protocol {activity!r} must be an object")
        protocols[str(activity)] = ActivitySegmentationProtocol(
            activity=str(activity),
            main_label=str(value["main_label"]),
            ordered_subactivities=tuple(
                str(label) for label in value["ordered_subactivities"]
            ),
            edge_crop_seconds=float(value["edge_crop_seconds"]),
        )

    return SegmentationContractConfiguration(
        schema_version=int(payload["schema_version"]),
        name=str(payload["name"]),
        authoritative=bool(payload["authoritative"]),
        controls_dataset_generation=bool(payload["controls_dataset_generation"]),
        automatic_boundary_detection_authorized=bool(
            payload["automatic_boundary_detection_authorized"]
        ),
        source=dict(payload["source"]),
        notes=tuple(str(note) for note in payload.get("notes", ())),
        boundary_evidence=dict(payload["boundary_evidence"]),
        sampling_rate_hz=int(parameters["sampling_rate_hz"]),
        interval_convention=str(parameters["interval_convention"]),
        activity_protocols=protocols,
    )


def build_governed_segment_intervals(
    raw_intervals: Sequence[RawSegmentInterval],
    *,
    activity: str,
    recording_sample_count: int,
    boundary_manifest_id: str,
    purpose: SegmentationPurpose,
    configuration: SegmentationContractConfiguration,
) -> tuple[GovernedSegmentInterval, ...]:
    """Validate, label, and crop externally supplied source intervals.

    No signal-derived boundary detector is invoked.  Scientific dataset use
    fails closed unless the selected configuration explicitly controls dataset
    generation.
    """
    if not isinstance(purpose, SegmentationPurpose):
        raise TypeError("purpose must be a SegmentationPurpose value")
    if purpose is SegmentationPurpose.SCIENTIFIC_DATASET and not (
        configuration.authoritative and configuration.controls_dataset_generation
    ):
        raise PermissionError(
            "the selected segmentation contract cannot control scientific "
            "dataset generation"
        )
    if isinstance(recording_sample_count, bool) or not isinstance(
        recording_sample_count, int
    ):
        raise TypeError("recording_sample_count must be an integer")
    if recording_sample_count <= 0:
        raise ValueError("recording_sample_count must be positive")
    if not isinstance(boundary_manifest_id, str):
        raise TypeError("boundary_manifest_id must be a string")
    if not boundary_manifest_id.strip():
        raise ValueError("boundary_manifest_id must not be empty")

    protocol = configuration.activity_protocols.get(activity)
    if protocol is None:
        raise ValueError(f"unsupported activity: {activity!r}")
    if len(raw_intervals) != len(protocol.ordered_subactivities):
        raise ValueError(
            f"activity {activity!r} requires "
            f"{len(protocol.ordered_subactivities)} intervals, "
            f"received {len(raw_intervals)}"
        )

    crop_samples = round(
        protocol.edge_crop_seconds * configuration.sampling_rate_hz
    )
    governed: list[GovernedSegmentInterval] = []
    previous_stop = 0
    for ordinal, (interval, subactivity) in enumerate(
        zip(raw_intervals, protocol.ordered_subactivities, strict=True),
        start=1,
    ):
        if not isinstance(interval, RawSegmentInterval):
            raise TypeError("raw_intervals must contain RawSegmentInterval values")
        if interval.stop_sample > recording_sample_count:
            raise ValueError("segment interval exceeds the recording length")
        if interval.start_sample < previous_stop:
            raise ValueError("segment intervals must be ordered and non-overlapping")
        if interval.sample_count <= 2 * crop_samples:
            raise ValueError(
                "segment is too short for the required symmetric edge crop"
            )
        retained_start = interval.start_sample + crop_samples
        retained_stop = interval.stop_sample - crop_samples
        governed.append(
            GovernedSegmentInterval(
                activity=activity,
                main_label=protocol.main_label,
                subactivity_label=subactivity,
                ordinal=ordinal,
                raw_start_sample=interval.start_sample,
                raw_stop_sample=interval.stop_sample,
                retained_start_sample=retained_start,
                retained_stop_sample=retained_stop,
                boundary_manifest_id=boundary_manifest_id,
                configuration_name=configuration.name,
            )
        )
        previous_stop = interval.stop_sample

    return tuple(governed)
