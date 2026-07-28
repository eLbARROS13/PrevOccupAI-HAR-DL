from dataclasses import replace
from pathlib import Path

import pytest

from prevoccupai_har.segmentation import (
    RawSegmentInterval,
    SegmentationContractConfiguration,
    SegmentationPurpose,
    build_governed_segment_intervals,
    load_segmentation_contract_configuration,
)


CONFIGURATION_PATH = Path("configs/mban_segmentation_contract.json")


@pytest.fixture
def configuration() -> SegmentationContractConfiguration:
    return load_segmentation_contract_configuration(CONFIGURATION_PATH)


def _build(
    intervals: list[RawSegmentInterval],
    activity: str,
    configuration: SegmentationContractConfiguration,
):
    return build_governed_segment_intervals(
        intervals,
        activity=activity,
        recording_sample_count=2_000_000,
        boundary_manifest_id="synthetic-boundaries-v1",
        purpose=SegmentationPurpose.VALIDATION,
        configuration=configuration,
    )


def test_configuration_is_approved_and_covers_ten_unique_subactivities(
    configuration: SegmentationContractConfiguration,
) -> None:
    assert configuration.authoritative is True
    assert configuration.controls_dataset_generation is True
    assert configuration.automatic_boundary_detection_authorized is False
    assert configuration.interval_convention == "half_open"
    unique_labels = {
        label
        for protocol in configuration.activity_protocols.values()
        for label in protocol.ordered_subactivities
    }
    assert len(unique_labels) == 10


def test_non_authoritative_contract_cannot_control_generation(
    configuration: SegmentationContractConfiguration,
) -> None:
    non_authoritative = replace(
        configuration,
        authoritative=False,
        controls_dataset_generation=False,
    )
    with pytest.raises(ValueError, match="cannot control dataset generation"):
        replace(non_authoritative, controls_dataset_generation=True)


def test_non_authoritative_contract_cannot_authorize_automatic_detection(
    configuration: SegmentationContractConfiguration,
) -> None:
    non_authoritative = replace(
        configuration,
        authoritative=False,
        controls_dataset_generation=False,
    )
    with pytest.raises(ValueError, match="automatic boundary detection"):
        replace(non_authoritative, automatic_boundary_detection_authorized=True)


def test_walking_intervals_receive_ordered_labels_and_five_second_crops(
    configuration: SegmentationContractConfiguration,
) -> None:
    intervals = [
        RawSegmentInterval(10_000, 310_000),
        RawSegmentInterval(320_000, 620_000),
        RawSegmentInterval(630_000, 930_000),
    ]

    governed = _build(intervals, "walking", configuration)

    assert [segment.subactivity_label for segment in governed] == [
        "walking_slow",
        "walking_medium",
        "walking_fast",
    ]
    assert all(segment.main_label == "walking" for segment in governed)
    assert governed[0].retained_start_sample == 15_000
    assert governed[0].retained_stop_sample == 305_000
    assert governed[0].retained_sample_count == 290_000


def test_stair_bouts_alternate_between_two_subactivity_labels(
    configuration: SegmentationContractConfiguration,
) -> None:
    intervals = [
        RawSegmentInterval(0, 75_000),
        RawSegmentInterval(80_000, 155_000),
        RawSegmentInterval(160_000, 235_000),
        RawSegmentInterval(240_000, 315_000),
    ]

    governed = _build(intervals, "stairs", configuration)

    assert [segment.subactivity_label for segment in governed] == [
        "stairs_up",
        "stairs_down",
        "stairs_up",
        "stairs_down",
    ]
    assert len({segment.subactivity_label for segment in governed}) == 2
    assert all(segment.main_label == "walking" for segment in governed)


def test_cabinet_and_standing_recordings_map_to_standing_main_label(
    configuration: SegmentationContractConfiguration,
) -> None:
    intervals = [RawSegmentInterval(0, 450_000), RawSegmentInterval(460_000, 910_000)]

    cabinets = _build(intervals, "cabinets", configuration)
    standing = _build(intervals, "standing", configuration)

    assert [segment.main_label for segment in cabinets] == ["standing", "standing"]
    assert [segment.subactivity_label for segment in standing] == [
        "standing_still",
        "standing_conversing",
    ]


def test_sitting_uses_thirty_second_edge_crop(
    configuration: SegmentationContractConfiguration,
) -> None:
    governed = _build(
        [RawSegmentInterval(100_000, 1_000_000)],
        "sitting",
        configuration,
    )

    assert governed[0].retained_start_sample == 130_000
    assert governed[0].retained_stop_sample == 970_000
    assert governed[0].retained_sample_count == 840_000


@pytest.mark.parametrize(
    ("activity", "count"),
    [("walking", 2), ("stairs", 8), ("cabinets", 3), ("standing", 3)],
)
def test_unsupported_segment_counts_fail_instead_of_receiving_generic_labels(
    configuration: SegmentationContractConfiguration,
    activity: str,
    count: int,
) -> None:
    intervals = [
        RawSegmentInterval(index * 100_000, (index + 1) * 100_000)
        for index in range(count)
    ]

    with pytest.raises(ValueError, match="requires"):
        _build(intervals, activity, configuration)


def test_intervals_must_be_ordered_and_non_overlapping(
    configuration: SegmentationContractConfiguration,
) -> None:
    intervals = [
        RawSegmentInterval(0, 300_000),
        RawSegmentInterval(250_000, 550_000),
        RawSegmentInterval(600_000, 900_000),
    ]

    with pytest.raises(ValueError, match="ordered and non-overlapping"):
        _build(intervals, "walking", configuration)


def test_interval_must_not_exceed_recording_length(
    configuration: SegmentationContractConfiguration,
) -> None:
    with pytest.raises(ValueError, match="exceeds"):
        build_governed_segment_intervals(
            [RawSegmentInterval(0, 2_100_000)],
            activity="sitting",
            recording_sample_count=2_000_000,
            boundary_manifest_id="synthetic-boundaries-v1",
            purpose=SegmentationPurpose.VALIDATION,
            configuration=configuration,
        )


def test_short_segments_fail_instead_of_silently_skipping_crop(
    configuration: SegmentationContractConfiguration,
) -> None:
    intervals = [
        RawSegmentInterval(0, 10_000),
        RawSegmentInterval(20_000, 30_000),
    ]

    with pytest.raises(ValueError, match="too short"):
        _build(intervals, "standing", configuration)


def test_scientific_dataset_purpose_accepts_externally_supplied_approved_boundaries(
    configuration: SegmentationContractConfiguration,
) -> None:
    governed = build_governed_segment_intervals(
        [RawSegmentInterval(0, 900_000)],
        activity="sitting",
        recording_sample_count=900_000,
        boundary_manifest_id="approved-boundaries-v1",
        purpose=SegmentationPurpose.SCIENTIFIC_DATASET,
        configuration=configuration,
    )

    assert len(governed) == 1
    assert governed[0].configuration_name == configuration.name


def test_boundary_manifest_identifier_is_required(
    configuration: SegmentationContractConfiguration,
) -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        build_governed_segment_intervals(
            [RawSegmentInterval(0, 900_000)],
            activity="sitting",
            recording_sample_count=900_000,
            boundary_manifest_id="   ",
            purpose=SegmentationPurpose.VALIDATION,
            configuration=configuration,
        )


def test_purpose_must_not_be_an_unvalidated_string(
    configuration: SegmentationContractConfiguration,
) -> None:
    with pytest.raises(TypeError, match="SegmentationPurpose"):
        build_governed_segment_intervals(
            [RawSegmentInterval(0, 900_000)],
            activity="sitting",
            recording_sample_count=900_000,
            boundary_manifest_id="synthetic-boundaries-v1",
            purpose="validation",  # type: ignore[arg-type]
            configuration=configuration,
        )


def test_boundary_manifest_identifier_must_be_a_string(
    configuration: SegmentationContractConfiguration,
) -> None:
    with pytest.raises(TypeError, match="must be a string"):
        build_governed_segment_intervals(
            [RawSegmentInterval(0, 900_000)],
            activity="sitting",
            recording_sample_count=900_000,
            boundary_manifest_id=1,  # type: ignore[arg-type]
            purpose=SegmentationPurpose.VALIDATION,
            configuration=configuration,
        )


def test_raw_interval_uses_exact_half_open_sample_count() -> None:
    interval = RawSegmentInterval(100, 150)

    assert interval.sample_count == 50


@pytest.mark.parametrize(
    ("start", "stop", "error"),
    [(0.0, 10, TypeError), (0, 10.0, TypeError), (-1, 10, ValueError), (10, 10, ValueError)],
)
def test_raw_interval_rejects_invalid_bounds(
    start: object,
    stop: object,
    error: type[Exception],
) -> None:
    with pytest.raises(error):
        RawSegmentInterval(start, stop)  # type: ignore[arg-type]
