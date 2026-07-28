"""Tests for immutable final-model-freeze manifests."""

from dataclasses import replace
from pathlib import Path

import pytest

from prevoccupai_har.final_manifest import (
    EXPECTED_SEEDS,
    FinalModelFreezeManifest,
    FrozenDlRefitReference,
    FrozenRfReference,
    verify_final_model_freeze_files,
    write_final_model_freeze_manifest,
)
from prevoccupai_har.provenance import sha256_canonical_json, sha256_file


def _freeze(tmp_path: Path) -> FinalModelFreezeManifest:
    dl_references = []
    for seed in EXPECTED_SEEDS:
        directory = tmp_path / f"seed_{seed}"
        directory.mkdir()
        record = directory / "refit_record.json"
        preprocessing = directory / "preprocessing_state.json"
        state = directory / "model_state.npz"
        record.write_text(f"record-{seed}\n", encoding="utf-8")
        preprocessing.write_text(f"preprocessing-{seed}\n", encoding="utf-8")
        state.write_bytes(f"model-{seed}\n".encode())
        dl_references.append(
            FrozenDlRefitReference(
                random_seed=seed,
                fixed_epoch_count=2,
                refit_record=record.relative_to(tmp_path).as_posix(),
                refit_record_sha256=sha256_file(record),
                preprocessing_state=preprocessing.relative_to(tmp_path).as_posix(),
                preprocessing_state_sha256=sha256_file(preprocessing),
                preprocessing_state_payload_sha256="a" * 64,
                model_state=state.relative_to(tmp_path).as_posix(),
                model_state_file_sha256=sha256_file(state),
                model_state_payload_sha256="b" * 64,
            )
        )
    rf_directory = tmp_path / "rf"
    rf_directory.mkdir()
    rf_record = rf_directory / "refit_record.json"
    rf_pipeline = rf_directory / "fitted_pipeline.joblib"
    rf_record.write_text("rf-record\n", encoding="utf-8")
    rf_pipeline.write_bytes(b"rf-pipeline\n")
    manifest = FinalModelFreezeManifest(
        schema_version=1,
        manifest_id="synthetic-final-freeze",
        created_at_utc="2026-07-17T12:00:00Z",
        purpose="final_model_freeze",
        scientific_result=False,
        holdout_accessed=False,
        selected_candidate_id="synthetic-cnn",
        class_labels=("sitting", "standing", "walking"),
        development_participants=("P003",),
        holdout_participants=("P001",),
        final_stage_source_revision=f"tree-sha256:{'c' * 64}",
        historical_source_revisions=(
            f"tree-sha256:{'d' * 64}",
            f"tree-sha256:{'e' * 64}",
        ),
        ensemble_method="arithmetic_mean_of_five_softmax_probability_vectors",
        primary_prediction="argmax_of_mean_probability",
        dl_refits=tuple(dl_references),
        random_forest=FrozenRfReference(
            refit_record=rf_record.relative_to(tmp_path).as_posix(),
            refit_record_sha256=sha256_file(rf_record),
            fitted_pipeline=rf_pipeline.relative_to(tmp_path).as_posix(),
            fitted_pipeline_sha256=sha256_file(rf_pipeline),
            selected_feature_names=("feature_1",),
            hyperparameters={
                "criterion": "entropy",
                "max_depth": 20,
                "n_estimators": 1000,
            },
        ),
        analysis_settings={
            "probability_calibration": "none",
            "temporal_smoothing": "none",
            "calibration_bin_count": 15,
            "expected_step_size_samples": 2500,
            "short_run_max_windows": 2,
        },
        input_hashes={"synthetic_input_sha256": "f" * 64},
        manifest_payload_sha256="0" * 64,
    )
    return replace(
        manifest,
        manifest_payload_sha256=sha256_canonical_json(manifest._payload()),
    )


def test_final_manifest_detects_payload_change(tmp_path: Path) -> None:
    manifest = _freeze(tmp_path)
    manifest.validate()

    with pytest.raises(ValueError, match="payload digest"):
        replace(manifest, selected_candidate_id="changed-candidate").validate()


def test_final_manifest_verifies_and_detects_artifact_tampering(
    tmp_path: Path,
) -> None:
    manifest = _freeze(tmp_path)
    manifest_path = tmp_path / "final_model_freeze_manifest.json"
    write_final_model_freeze_manifest(manifest_path, manifest)

    assert verify_final_model_freeze_files(manifest_path) == manifest
    tampered = tmp_path / manifest.dl_refits[0].model_state
    tampered.write_bytes(b"tampered\n")
    with pytest.raises(ValueError, match="Frozen DL artifact hash changed"):
        verify_final_model_freeze_files(manifest_path)
