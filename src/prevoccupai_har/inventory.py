"""Privacy-preserving inventory of raw OpenSignals muscleBAN recordings."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterator

from .protocol import ProtocolConfiguration


@dataclass(frozen=True)
class RecordingManifestEntry:
    """A recording identity that omits raw filenames, timestamps, and device IDs."""

    participant_id: str
    activity_directory: str
    activity_base: str
    recording_index: int
    file_size_bytes: int
    sampling_rate_hz: int | None
    accelerometer_channels: int | None
    device_type: str | None
    sha256: str | None
    header_valid: bool

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-safe record."""
        return asdict(self)


def _read_device_metadata(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8", errors="replace") as stream:
        first_line = stream.readline()
        second_line = stream.readline()
    if not first_line.startswith("# OpenSignals") or not second_line.startswith("# "):
        raise ValueError("Unrecognised OpenSignals header")
    decoded = json.loads(second_line[2:])
    if not isinstance(decoded, dict) or len(decoded) != 1:
        raise ValueError("Expected exactly one OpenSignals device header")
    metadata = next(iter(decoded.values()))
    if not isinstance(metadata, dict):
        raise ValueError("Invalid OpenSignals device metadata")
    return metadata


def _sha256(path: Path, chunk_size: int = 4 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _activity_base(directory_name: str, expected_bases: tuple[str, ...]) -> str | None:
    for base in expected_bases:
        if directory_name == base or re.fullmatch(rf"{re.escape(base)}_[0-9]+", directory_name):
            return base
    return None


def iter_recordings(
    protocol: ProtocolConfiguration,
    *,
    raw_root: Path | None = None,
) -> Iterator[tuple[Path, str, str, str, int]]:
    """Yield private paths plus their privacy-safe logical coordinates."""
    root = (raw_root or protocol.raw_data_root).resolve()
    participant_pattern = re.compile(protocol.participant_id_pattern)
    filename_pattern = re.compile(protocol.muscleban_filename_pattern)
    for participant_path in sorted(root.iterdir()):
        if not participant_path.is_dir() or participant_pattern.fullmatch(participant_path.name) is None:
            continue
        for activity_path in sorted(path for path in participant_path.iterdir() if path.is_dir()):
            base = _activity_base(
                activity_path.name,
                protocol.required_activity_directory_bases,
            )
            if base is None:
                continue
            files = sorted(
                path
                for path in activity_path.iterdir()
                if path.is_file() and filename_pattern.fullmatch(path.name) is not None
            )
            for recording_index, path in enumerate(files, start=1):
                yield path, participant_path.name, activity_path.name, base, recording_index


def build_recording_manifest(
    protocol: ProtocolConfiguration,
    *,
    raw_root: Path | None = None,
    calculate_checksums: bool = True,
) -> dict[str, object]:
    """Build a reproducible snapshot manifest without exposing raw path metadata."""
    root = (raw_root or protocol.raw_data_root).resolve()
    entries: list[RecordingManifestEntry] = []
    for path, participant, activity_directory, base, recording_index in iter_recordings(
        protocol,
        raw_root=root,
    ):
        metadata: dict[str, Any] = {}
        header_valid = True
        try:
            metadata = _read_device_metadata(path)
        except (OSError, ValueError, json.JSONDecodeError):
            header_valid = False
        sensors = metadata.get("sensor")
        entries.append(
            RecordingManifestEntry(
                participant_id=participant,
                activity_directory=activity_directory,
                activity_base=base,
                recording_index=recording_index,
                file_size_bytes=path.stat().st_size,
                sampling_rate_hz=(
                    int(metadata["sampling rate"])
                    if header_valid and "sampling rate" in metadata
                    else None
                ),
                accelerometer_channels=(
                    sensors.count("gACC") if isinstance(sensors, list) else None
                ),
                device_type=(str(metadata["device"]) if header_valid and "device" in metadata else None),
                sha256=_sha256(path) if calculate_checksums else None,
                header_valid=header_valid,
            )
        )

    digest = hashlib.sha256()
    for entry in entries:
        digest.update(json.dumps(entry.as_dict(), sort_keys=True).encode("utf-8"))
        digest.update(b"\n")
    observed = sorted({entry.participant_id for entry in entries})
    expected = set(protocol.all_expected_participants)
    return {
        "schema_version": 1,
        "dataset_name": protocol.dataset_name,
        "source_status": protocol.source_status,
        "recording_count": len(entries),
        "total_file_size_bytes": sum(entry.file_size_bytes for entry in entries),
        "checksums_calculated": calculate_checksums,
        "manifest_content_sha256": digest.hexdigest(),
        "observed_participants": observed,
        "missing_expected_participants": sorted(expected - set(observed)),
        "recordings": [entry.as_dict() for entry in entries],
        "privacy_note": (
            "Raw filenames, acquisition timestamps, filesystem paths, and device identifiers "
            "are intentionally omitted. Recording indices are local to each activity directory."
        ),
    }

