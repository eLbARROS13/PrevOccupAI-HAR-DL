#!/usr/bin/env python3
"""Generate the configuration-derived CNN/TCN methods figure."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from prevoccupai_har.architecture_figure import generate_model_architecture_figure


def parse_args() -> argparse.Namespace:
    """Parse configuration and exclusive-output paths."""
    parser = argparse.ArgumentParser(
        description=(
            "Generate a deterministic vector schematic from the validated compact-CNN "
            "and residual-TCN configurations."
        )
    )
    parser.add_argument("--cnn-config", type=Path, required=True)
    parser.add_argument("--tcn-config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    """Generate the figure without participant data or performance values."""
    args = parse_args()
    record = generate_model_architecture_figure(
        cnn_configuration_path=args.cnn_config,
        tcn_configuration_path=args.tcn_config,
        output_directory=args.output_dir,
    )
    print(json.dumps(record.as_dict(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
