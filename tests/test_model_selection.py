"""Tests for frozen repeated-seed/fold planning and selection bundles."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from prevoccupai_har.analysis_records import (  # noqa: E402
    build_prediction_analysis_record,
    write_prediction_analysis_record,
)
from prevoccupai_har.model_selection import (  # noqa: E402
    ConservativeSelectionRule,
    DevelopmentSelectionPlan,
    SelectionCandidate,
    SelectionFold,
    build_development_selection_plan,
    build_model_selection_bundle,
    load_development_selection_plan,
    load_model_selection_bundle,
    write_development_selection_plan,
    write_model_selection_bundle,
)
from prevoccupai_har.prediction_artifacts import (  # noqa: E402
    PredictionArtifactRecord,
    PredictionWindowRecord,
    write_prediction_artifact_record,
)
from prevoccupai_har.provenance import (  # noqa: E402
    sha256_canonical_json,
    sha256_file,
)
from prevoccupai_har.results import (  # noqa: E402
    TrainingResultRecord,
    write_training_result_record,
)
from prevoccupai_har.selection_reporting import (  # noqa: E402
    generate_selection_report_package,
)


ROOT = Path(__file__).resolve().parents[1]
CREATED_AT = "2026-07-15T20:00:00Z"


def test_checked_in_selection_plan_is_complete_nonpredictive_and_immutable(
    tmp_path: Path,
) -> None:
    plan = build_development_selection_plan(
        configuration_path=ROOT / "configs/development_model_selection.json",
        created_at_utc=CREATED_AT,
    )
    output = tmp_path / "plan.json"
    write_development_selection_plan(output, plan)
    loaded = load_development_selection_plan(output)

    assert loaded.expected_run_count == 50
    assert len(loaded.folds) == 5
    assert loaded.random_seeds == (1103, 2207, 3301, 4409, 5519)
    assert {candidate.experiment_id for candidate in loaded.candidates} == {
        "cnn_1d_100hz_single_stream_v1",
        "tcn_1d_100hz_single_stream_v1",
    }
    assert {candidate.trainable_parameter_count for candidate in loaded.candidates} == {
        15_987,
        2_307,
    }
    assert loaded.training_authorized is True
    assert loaded.scientific_result is False
    assert loaded.holdout_accessed is False
    with pytest.raises(FileExistsError):
        write_development_selection_plan(output, plan)


def _synthetic_plan(path: Path) -> DevelopmentSelectionPlan:
    candidates = (
        SelectionCandidate(
            experiment_id="cnn_1d_single_stream_v1",
            role="reference",
            model_configuration_sha256=sha256_file(ROOT / "configs/cnn_1d.json"),
            complexity_profile_sha256=sha256_file(
                ROOT / "artifacts/software_validation/cnn_1d_complexity_profile_cpu.json"
            ),
            trainable_parameter_count=15_987,
        ),
        SelectionCandidate(
            experiment_id="tcn_1d_single_stream_v1",
            role="challenger",
            model_configuration_sha256=sha256_file(ROOT / "configs/tcn_1d.json"),
            complexity_profile_sha256=sha256_file(
                ROOT / "artifacts/software_validation/tcn_1d_complexity_profile_cpu.json"
            ),
            trainable_parameter_count=49_731,
        ),
    )
    folds = (
        SelectionFold(
            fold_index=0,
            training_subjects=("SYNTHETIC_DEVELOPMENT_B",),
            validation_subjects=("SYNTHETIC_DEVELOPMENT_A",),
            holdout_subjects=("SYNTHETIC_HOLDOUT_NOT_LOADED",),
        ),
        SelectionFold(
            fold_index=1,
            training_subjects=("SYNTHETIC_DEVELOPMENT_A",),
            validation_subjects=("SYNTHETIC_DEVELOPMENT_B",),
            holdout_subjects=("SYNTHETIC_HOLDOUT_NOT_LOADED",),
        ),
    )
    settings = {
        "probability_transform": {
            "method": "softmax",
            "temperature": 1.0,
            "fitted": False,
        },
        "calibration_bin_count": 5,
        "expected_step_size_samples": 2500,
        "short_run_max_windows": 1,
    }
    rule = ConservativeSelectionRule(
        primary_metric="participant_macro_f1",
        supporting_metric="participant_balanced_accuracy",
        challenger_minimum_primary_gain=0.01,
        challenger_maximum_supporting_loss=0.005,
        minimum_nonnegative_seed_fraction=0.6,
    )
    plan = DevelopmentSelectionPlan(
        schema_version=1,
        plan_id="synthetic-selection-contract-v1",
        created_at_utc=CREATED_AT,
        purpose="synthetic_validation",
        scientific_result=False,
        holdout_accessed=False,
        training_authorized=False,
        selection_configuration_sha256="9" * 64,
        protocol_configuration_sha256="a" * 64,
        split_manifest_sha256="b" * 64,
        window_store_index_sha256="f" * 64,
        statistical_analysis_plan_sha256="c" * 64,
        class_labels=("sitting", "standing", "walking"),
        random_seeds=(1103,),
        candidates=candidates,
        folds=folds,
        analysis_settings=settings,
        selection_rule=rule,
        expected_run_count=4,
        plan_payload_sha256="0" * 64,
    )
    plan = replace(plan, plan_payload_sha256=sha256_canonical_json(plan._payload()))
    write_development_selection_plan(path, plan)
    return plan


def _training_record(
    *,
    candidate: SelectionCandidate,
    fold: SelectionFold,
    run_id: str,
) -> TrainingResultRecord:
    validation_score = 1.0 if candidate.role == "challenger" else 0.55
    return TrainingResultRecord(
        schema_version=1,
        run_id=run_id,
        created_at_utc=CREATED_AT,
        experiment_id=candidate.experiment_id,
        purpose="synthetic_validation",
        scientific_result=False,
        holdout_accessed=False,
        source_revision="unversioned_workspace_software_test",
        model_configuration_sha256=candidate.model_configuration_sha256,
        learned_preprocessing_sha256="d" * 64,
        protocol_configuration_sha256=None,
        data_provenance=None,
        random_seed=1103,
        training_subjects=fold.training_subjects,
        validation_subjects=fold.validation_subjects,
        trainable_parameter_count=candidate.trainable_parameter_count,
        best_epoch=2,
        stopped_early=False,
        history=(
            {
                "epoch": 1,
                "training_loss": 1.1,
                "validation_loss": 1.0,
                "validation_macro_f1": validation_score - 0.05,
                "validation_balanced_accuracy": validation_score - 0.05,
            },
            {
                "epoch": 2,
                "training_loss": 0.9,
                "validation_loss": 0.8,
                "validation_macro_f1": validation_score,
                "validation_balanced_accuracy": validation_score,
            },
        ),
        software_versions={"numpy": "synthetic", "torch": "synthetic"},
    )


def _prediction_record(
    *,
    candidate: SelectionCandidate,
    fold: SelectionFold,
    training_path: Path,
    run_id: str,
) -> PredictionArtifactRecord:
    rows: list[PredictionWindowRecord] = []
    for index, label in enumerate(("sitting", "standing", "walking")):
        predicted = label
        if candidate.role == "reference" and label == "sitting":
            predicted = "standing"
        logits_by_label = {
            "sitting": (4.0, 1.0, 0.0),
            "standing": (1.0, 4.0, 0.0),
            "walking": (0.0, 1.0, 4.0),
        }
        participant = fold.validation_subjects[0]
        rows.append(
            PredictionWindowRecord(
                window_index=index,
                participant_id=participant,
                recording_key_sha256=sha256_canonical_json(
                    {"participant": participant, "label": label}
                ),
                sensor_stream_key_sha256=sha256_canonical_json(
                    {"participant": participant, "label": label, "stream": 1}
                ),
                main_label=label,
                sub_activity_label=f"synthetic_{label}",
                sensor_side="synthetic",
                start_sample=0,
                end_sample_exclusive=5000,
                preprocessing_status="synthetic",
                quality_status="synthetic",
                predicted_label=predicted,
                logits=logits_by_label[predicted],
            )
        )
    return PredictionArtifactRecord(
        schema_version=1,
        run_id=run_id,
        created_at_utc=CREATED_AT,
        experiment_id=candidate.experiment_id,
        purpose="synthetic_validation",
        scientific_result=False,
        holdout_accessed=False,
        source_revision="unversioned_workspace_software_test",
        training_run_id=json.loads(training_path.read_text(encoding="utf-8"))["run_id"],
        training_result_sha256=sha256_file(training_path),
        model_configuration_sha256=candidate.model_configuration_sha256,
        learned_preprocessing_sha256="d" * 64,
        model_state_sha256=("e" if candidate.role == "reference" else "f") * 64,
        class_labels=("sitting", "standing", "walking"),
        validation_subjects=fold.validation_subjects,
        logit_dtype="float32",
        window_count=len(rows),
        prediction_payload_sha256=sha256_canonical_json(
            [row.as_dict() for row in rows]
        ),
        windows=tuple(rows),
    )


def _synthetic_artifact_grid(tmp_path: Path) -> tuple[Path, Path]:
    plan_path = tmp_path / "plan.json"
    plan = _synthetic_plan(plan_path)
    runs: list[dict[str, object]] = []
    for candidate in plan.candidates:
        for fold in plan.folds:
            stem = f"{candidate.role}-fold-{fold.fold_index}-seed-1103"
            training_path = tmp_path / f"{stem}-training.json"
            prediction_path = tmp_path / f"{stem}-predictions.json"
            analysis_path = tmp_path / f"{stem}-analysis.json"
            write_training_result_record(
                training_path,
                _training_record(
                    candidate=candidate,
                    fold=fold,
                    run_id=f"{stem}-training",
                ),
            )
            write_prediction_artifact_record(
                prediction_path,
                _prediction_record(
                    candidate=candidate,
                    fold=fold,
                    training_path=training_path,
                    run_id=f"{stem}-predictions",
                ),
            )
            analysis = build_prediction_analysis_record(
                analysis_id=f"{stem}-analysis",
                created_at_utc=CREATED_AT,
                prediction_artifact_path=prediction_path,
                calibration_bin_count=5,
                expected_step_size_samples=2500,
                short_run_max_windows=1,
            )
            write_prediction_analysis_record(analysis_path, analysis)
            runs.append(
                {
                    "candidate_id": candidate.experiment_id,
                    "fold_index": fold.fold_index,
                    "random_seed": 1103,
                    "training_result": str(training_path),
                    "prediction_artifact": str(prediction_path),
                    "analysis_record": str(analysis_path),
                }
            )
    manifest_path = tmp_path / "runs.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "purpose": "synthetic_validation",
                "selection_plan_sha256": sha256_file(plan_path),
                "runs": runs,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return plan_path, manifest_path


def test_selection_bundle_binds_grid_summarizes_curves_and_applies_rule(
    tmp_path: Path,
) -> None:
    plan_path, manifest_path = _synthetic_artifact_grid(tmp_path)
    output_path = tmp_path / "bundle.json"
    bundle = build_model_selection_bundle(
        bundle_id="synthetic-selection-bundle-v1",
        created_at_utc=CREATED_AT,
        selection_plan_path=plan_path,
        run_manifest_path=manifest_path,
    )
    write_model_selection_bundle(output_path, bundle)
    loaded = load_model_selection_bundle(output_path)

    assert loaded.run_count == 4
    assert loaded.decision["status"] == "synthetic_contract_validation_only"
    assert loaded.decision["selected_candidate_id"] == "tcn_1d_single_stream_v1"
    assert loaded.decision["primary_gain"] > 0.01
    assert loaded.decision["nonnegative_seed_fraction"] == 1.0
    assert loaded.scientific_result is False
    assert loaded.holdout_accessed is False
    for summary in loaded.candidate_summaries.values():
        assert summary["run_count"] == 2
        assert len(summary["learning_curve"]) == 2
        assert summary["learning_curve"][1]["contributing_run_count"] == 2
    with pytest.raises(FileExistsError):
        write_model_selection_bundle(output_path, bundle)

    first_report = generate_selection_report_package(
        selection_bundle_path=output_path,
        output_directory=tmp_path / "first-report",
    )
    second_report = generate_selection_report_package(
        selection_bundle_path=output_path,
        output_directory=tmp_path / "second-report",
    )
    assert first_report.outputs == second_report.outputs
    assert first_report.scientific_result is False
    assert first_report.holdout_accessed is False
    assert set(first_report.outputs) == {
        "paired_seed_macro_f1",
        "learning_curves",
        "complexity_performance",
        "selection_summary",
        "learning_curve_summary",
    }
    report_manifest = (tmp_path / "first-report/REPORT_MANIFEST.json").read_text()
    assert "SYNTHETIC_DEVELOPMENT_A" not in report_manifest
    assert "SYNTHETIC_DEVELOPMENT_B" not in report_manifest
    for role, output_record in first_report.outputs.items():
        output_file = tmp_path / "first-report" / output_record["filename"]
        assert output_record["sha256"] == sha256_file(output_file)
        if role in {
            "paired_seed_macro_f1",
            "learning_curves",
            "complexity_performance",
        }:
            assert output_file.read_bytes().startswith(b"%PDF-")
    with pytest.raises(FileExistsError):
        generate_selection_report_package(
            selection_bundle_path=output_path,
            output_directory=tmp_path / "first-report",
        )


def test_checked_in_closed_plan_cannot_build_a_scientific_bundle(
    tmp_path: Path,
) -> None:
    current = build_development_selection_plan(
        configuration_path=ROOT / "configs/development_model_selection.json",
        created_at_utc=CREATED_AT,
    )
    closed = replace(current, training_authorized=False, plan_payload_sha256="0" * 64)
    closed = replace(
        closed,
        plan_payload_sha256=sha256_canonical_json(closed._payload()),
    )
    plan_path = tmp_path / "closed-plan.json"
    write_development_selection_plan(plan_path, closed)
    manifest_path = tmp_path / "runs.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "purpose": "development_selection",
                "selection_plan_sha256": sha256_file(plan_path),
                "runs": [],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(PermissionError, match="closed training gate"):
        build_model_selection_bundle(
            bundle_id="prohibited-scientific-bundle-v1",
            created_at_utc=CREATED_AT,
            selection_plan_path=plan_path,
            run_manifest_path=manifest_path,
        )


def test_selection_bundle_rejects_cross_candidate_artifact_substitution(
    tmp_path: Path,
) -> None:
    plan_path, manifest_path = _synthetic_artifact_grid(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["runs"][0]["training_result"] = manifest["runs"][2]["training_result"]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="experiment identifier|configuration"):
        build_model_selection_bundle(
            bundle_id="tampered-selection-bundle-v1",
            created_at_utc=CREATED_AT,
            selection_plan_path=plan_path,
            run_manifest_path=manifest_path,
        )
