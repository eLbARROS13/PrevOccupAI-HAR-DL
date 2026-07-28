# Software-Validation Artifacts

This directory contains data-independent execution artifacts. They must not be interpreted as participant-level, scientific, or deployment results.

`cnn_1d_complexity_profile_cpu.json` was generated from the checked-in compact-CNN configuration with a batch of one all-zero tensor, shape `1 x 3 x 5000`. It records architecture size and a hardware-specific forward-pass timing distribution. The run loaded no participant recording, derived window, label, prediction, or external hold-out data. Its `source_revision` is explicitly `unversioned_workspace_software_test` because the clean workspace root is not currently a Git repository.

`tcn_1d_complexity_profile_cpu.json` applies the same generated-zero, batch-one, one-thread protocol to the checked-in compact residual-TCN configuration. It is bound to that configuration by SHA-256 and remains a software-cost measurement rather than a predictive result.

`development_selection_plan.json` is a non-predictive, hash-bound 50-slot plan covering both candidates, five shared subject-disjoint folds, and five shared random seeds. It records the currently closed training gate and binds the model configurations, complexity profiles, protocol, split manifest, statistical plan, analysis settings, and conservative selection thresholds. It contains no participant prediction or model-selection outcome.

The synthetic Random Forest smoke record is intentionally not checked into this directory because its timestamped execution artifact is reproducible from `configs/rf_baseline_synthetic_smoke.json` and `scripts/run_synthetic_rf_smoke.py`. It uses generated feature rows only and records `scientific_result: false` and `holdout_accessed: false`. The authoritative 45-feature comparator configuration is `configs/rf_baseline.json`; neither configuration authorizes participant execution.

The latency values are provisional. Final manuscript values require frozen models, declared deployment hardware, controlled thread and power settings, repeated measurements, and a documented decision on whether preprocessing, data transfer, and batching are included.
