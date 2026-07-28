#!/usr/bin/env python3
"""Derive reproducible metrics from a JSON confusion-matrix record."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from prevoccupai_har.evaluation import metrics_from_confusion_matrix  # noqa: E402


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input",
        type=Path,
        nargs="?",
        default=PROJECT_ROOT / "artifacts" / "baseline" / "published_rf_confusion_matrix.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "baseline" / "published_rf_metrics.json",
    )
    return parser.parse_args()


def main() -> None:
    """Validate the record, derive metrics, and retain source provenance."""
    args = parse_args()
    source = json.loads(args.input.read_text(encoding="utf-8"))
    metrics = metrics_from_confusion_matrix(
        np.asarray(source["confusion_matrix"], dtype=np.int64),
        source["class_order"],
    )
    reported_accuracy = float(source["reported_accuracy_percent"]) / 100
    if not np.isclose(metrics["accuracy"], reported_accuracy, atol=0.00005):
        raise ValueError(
            "Derived and reported accuracies disagree beyond the two-decimal reporting tolerance"
        )
    output = {
        "schema_version": 1,
        "source_record": str(args.input.relative_to(PROJECT_ROOT)),
        "source_image_sha256": source["source_image_sha256"],
        "metrics": metrics,
        "interpretation": "quoted_prior_work_result_not_independently_reproduced",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

