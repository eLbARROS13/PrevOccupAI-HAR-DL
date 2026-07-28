# PrevOccupAI-HAR-DL

Leakage-safe analysis code for subject-disjoint Human Activity Recognition
using bilateral, upper-trapezius muscleBAN accelerometry from the PrevOccupAI+
study.

This repository is the code release supporting the journal analysis developed
from the PrevOccupAI and muscleBAN HAR research programme. It contains the
Python implementation used for the compact 1D CNN, compact residual TCN,
fold-local Random Forest comparator, participant-aware evaluation, calibration
diagnostics, temporal diagnostics, provenance checks, and single-use holdout
guardrails.

## Scientific scope

The frozen study used:

- tri-axial muscleBAN accelerometry;
- 5 s windows with 50% overlap;
- participant-disjoint development folds;
- a fixed four-participant external holdout;
- train-fold-only normalization, balancing, feature selection, and tuning;
- five fixed random seeds for CNN and TCN development;
- participant-level macro F1 and balanced accuracy as the primary development
  criteria; and
- one authorized external evaluation after source and models were frozen.

The public repository deliberately contains no raw or processed participant
signals, feature matrices, window-level labels or predictions, device
identifiers, fitted model weights, serialized estimators, or access ledger.
Exact refitting therefore requires separately governed access to the study
data.

## Installation

Python 3.11 or newer is required. Python 3.12 was used for the frozen study.

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -e '.[test,dl,figures,classical]'
pytest -q
```

TSFEL reconstruction checks require the historical feature environment:

```bash
python3 -m pip install -r requirements/tsfel_reconstruction_exact.txt
python3 -m pip install -e . --no-deps
pytest -q tests/test_feature_reconstruction.py
```

Synthetic smoke tests do not load participant data:

```bash
python3 scripts/run_synthetic_cnn_smoke.py --output /tmp/cnn-smoke.json
python3 scripts/run_synthetic_model_smoke.py \
  --config configs/tcn_1d.json \
  --output /tmp/tcn-smoke.json
python3 scripts/run_synthetic_rf_smoke.py --output /tmp/rf-smoke.json
```

## Repository structure

- `src/prevoccupai_har/`: data contracts, models, training, evaluation, and
  provenance logic.
- `configs/`: model and method configurations. Hardware identifiers are
  replaced by explicit examples in the public release.
- `scripts/`: experiment and synthetic-validation entry points.
- `tests/`: synthetic and contract-level regression tests.
- `artifacts/software_validation/`: participant-data-free model complexity
  records.
- `docs/`: statistical plan, provenance, and public-release boundaries.

The public protocol retains the study cohort and method structure but points to
a redacted example device mapping. The holdout policy is intentionally disabled.
Publishing this code does not authorize renewed access to the consumed study
holdout.

## Foundations and attribution

This is a clean, leakage-safe implementation for the journal analysis; it does
not import the earlier pipelines at runtime. Its methodological and data
foundations were developed through the wider PrevOccupAI collaboration.
Phillip Probst contributed substantially to the original PrevOccupAI repository
and the foundations on which this work builds.

Relevant first-party repositories are:

- [PrevOccupAI_mBAN_HAR](https://github.com/novabiosignals/PrevOccupAI_mBAN_HAR)
  — prior classical muscleBAN HAR pipeline;
- [PrevOccupAI_mBAN_QA](https://github.com/novabiosignals/PrevOccupAI_mBAN_QA)
  — prior muscleBAN quality-assessment implementation; and
- [PrevOccupAI_HAR_JA](https://github.com/p-probst/PrevOccupAI_HAR_JA)
  — related smartphone HAR work.

Study authors are Gonçalo Barros, Sara Santos, Phillip Probst, and Hugo Gamboa.
See `CITATION.cff` and `NOTICE.md` for citation and provenance details.

## Licence and reuse

This initial public release is source-visible for scientific transparency.
Because the earlier first-party repositories do not currently declare a
machine-readable licence and formal reuse terms have not yet been agreed by all
contributors, no open-source licence is granted by this repository. Contact the
study authors before redistributing or adapting the code.
