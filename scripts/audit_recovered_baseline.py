#!/usr/bin/env python3
"""Audit recovered first-party HAR result archives without deserializing models."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import subprocess
from pathlib import Path
from typing import Any
from zipfile import ZipFile


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARCHIVE_PATH = "tmp_har_windows_package/Other_Model_Results.zip"
RF_RESULT_MEMBER = (
    "Other_Model_Results/Results/ML/5000_w_size/num_classes_3/"
    "Random Forest_f25_wNorm-none.csv"
)
SUMMARY_MEMBER = "Other_Model_Results/HAR_results/model_analysis_w5000.csv"
ARCHIVE_CONFUSION_MEMBER = (
    "Other_Model_Results/HAR/production_models/5000_w_size/"
    "ConfusionMatrix_5000_w_size.png"
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo",
        type=Path,
        default=PROJECT_ROOT / "PrevOccupAI_mBAN_HAR",
    )
    parser.add_argument("--ref", default="upstream/HAR_focused")
    parser.add_argument(
        "--published-confusion-image",
        type=Path,
        default=PROJECT_ROOT / "mBAN_Article" / "5000_w_size" / "ConfusionMatrix_5000_w_size.png",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "baseline" / "recovered_branch_audit.json",
    )
    return parser.parse_args()


def git_output(repo: Path, *arguments: str, text: bool = False) -> bytes | str:
    """Run a read-only Git command and return its checked output."""
    result = subprocess.run(
        ["git", "-C", str(repo), *arguments],
        check=True,
        capture_output=True,
        text=text,
    )
    return result.stdout


def parse_csv(data: bytes, *, delimiter: str) -> list[dict[str, str]]:
    """Parse a small UTF-8 CSV member."""
    return list(csv.DictReader(io.StringIO(data.decode("utf-8-sig")), delimiter=delimiter))


def index_or_none(text: str, fragment: str) -> int | None:
    """Return the first source offset of a fragment, if present."""
    index = text.find(fragment)
    return index if index >= 0 else None


def main() -> None:
    """Produce a safe evidence record for source, CSV, and image artifacts."""
    args = parse_args()
    commit = str(git_output(args.repo, "rev-parse", args.ref, text=True)).strip()
    archive = bytes(git_output(args.repo, "show", f"{args.ref}:{ARCHIVE_PATH}"))
    model_selection_source = str(
        git_output(args.repo, "show", f"{args.ref}:HAR/model_selection.py", text=True)
    )
    loader_source = str(git_output(args.repo, "show", f"{args.ref}:HAR/load.py", text=True))
    segmenter_source = str(
        git_output(args.repo, "show", f"{args.ref}:raw_data_processor/mban_data_segmenter.py", text=True)
    )
    config_source = str(
        git_output(
            args.repo,
            "show",
            f"{args.ref}:tmp_har_windows_package/har_training_windows/har_config.py",
            text=True,
        )
    )

    with ZipFile(io.BytesIO(archive)) as bundle:
        member_names = bundle.namelist()
        rf_rows = parse_csv(bundle.read(RF_RESULT_MEMBER), delimiter=";")
        summary_rows = parse_csv(bundle.read(SUMMARY_MEMBER), delimiter=",")
        archive_confusion = bundle.read(ARCHIVE_CONFUSION_MEMBER)

    rf_summary = next(
        row
        for row in summary_rows
        if row["estimator_name"] == "Random Forest" and row["norm_type"] == "none"
    )
    cv_means = {float(row["estimator_avg_acc"]) for row in rf_rows}
    cv_stds = {float(row["estimator_std_acc"]) for row in rf_rows}
    if len(cv_means) != 1 or len(cv_stds) != 1:
        raise ValueError("Recovered RF fold rows do not agree on the aggregate CV statistics")

    load_call = index_or_none(model_selection_source, "load_features(")
    split_call = index_or_none(model_selection_source, "next(splitter.split(")
    selection_call = index_or_none(model_selection_source, "select_k_best_features(")
    nested_evaluation_call = index_or_none(model_selection_source, "info_df = nested_cross_val(")
    published_confusion = args.published_confusion_image.read_bytes()
    result: dict[str, Any] = {
        "schema_version": 1,
        "source_repository": "https://github.com/eLbARROS13/PrevOccupAI_mBAN_only.git",
        "source_ref": args.ref,
        "source_commit": commit,
        "archive_path": ARCHIVE_PATH,
        "archive_sha256": hashlib.sha256(archive).hexdigest(),
        "safety": {
            "joblib_member_count": sum(name.endswith(".joblib") for name in member_names),
            "serialized_models_deserialized": False,
            "inspection_scope": "CSV text, source text, member names, and image hashes only",
        },
        "rf_25_features_no_normalization": {
            "outer_fold_accuracy_percent": [float(row["test_accuracy"]) for row in rf_rows],
            "nested_cv_accuracy_percent_mean": next(iter(cv_means)),
            "nested_cv_accuracy_percentage_point_sd": next(iter(cv_stds)),
            "outer_fold_count": len(rf_rows),
            "candidate_hyperparameters": [
                {
                    "criterion": row["criterion"],
                    "max_depth": int(row["max_depth"]),
                    "n_estimators": int(row["n_estimators"]),
                }
                for row in rf_rows
            ],
        },
        "archive_wide_rf_none_summary": {
            "pooled_outer_fold_accuracy_percent_mean": float(rf_summary["mean"]),
            "pooled_outer_fold_accuracy_percentage_point_sd": float(rf_summary["std"]),
            "row_count": int(rf_summary["count"]),
            "interpretation": (
                "This pools fold rows across multiple feature counts and is not the "
                "25-feature model estimate."
            ),
        },
        "confusion_image_provenance": {
            "recovered_archive_image_sha256": hashlib.sha256(archive_confusion).hexdigest(),
            "local_camera_ready_image_sha256": hashlib.sha256(published_confusion).hexdigest(),
            "images_are_identical": archive_confusion == published_confusion,
            "local_camera_ready_reported_accuracy_percent": 86.58,
            "recovered_archive_visually_transcribed_accuracy_percent": 85.15,
        },
        "implementation_order_flags_requiring_review": {
            "balanced_feature_loading_precedes_holdout_split": (
                load_call is not None and split_call is not None and load_call < split_call
            ),
            "feature_selection_precedes_nested_model_evaluation": (
                selection_call is not None
                and nested_evaluation_call is not None
                and selection_call < nested_evaluation_call
            ),
            "balancing_uses_all_subject_class_minima": (
                "class_instances_df.min(axis=0)" in loader_source
            ),
            "segmenter_subject_loop_is_hard_coded_to_p002": (
                'for subject_id in tqdm(["P002"]' in segmenter_source
            ),
            "packaged_production_defaults_use_20_features": (
                'HAR_NUM_FEATURES_RETAIN", "20"' in config_source
            ),
            "packaged_production_defaults_use_minmax": (
                'HAR_BEST_NORMALIZATION", "minmax"' in config_source
            ),
        },
        "interpretation": (
            "The recovered branch contains the exact published RF nested-CV estimate, but its "
            "hold-out image differs from the local camera-ready result. It is therefore evidence "
            "for method reconstruction, not a complete published-result provenance chain."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
