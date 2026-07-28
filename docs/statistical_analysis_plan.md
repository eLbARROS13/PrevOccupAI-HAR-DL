# Statistical Analysis Plan

Status: final predeclared specification; to be hash-bound with the frozen-model manifest before the single hold-out execution

Last reviewed: 2026-07-17

## Decision question

The primary comparison asks whether one validation-selected deep-learning model differs from a faithfully reproduced Random Forest baseline under the same participants, retained windows, labels, QA decisions, and fixed hold-out distribution. If that equivalence cannot be established, the RF comparison will be descriptive and contextual rather than inferential.

## Analysis populations

- Development cohort: P003--P015, P017, P019, and P020.
- Fixed hold-out cohort: P001, P002, P016, and P018.
- Validation assignments must be participant-disjoint and wholly contained in the development cohort.
- Left and right sensor samples from one participant remain in the same partition.

The author-approved processed snapshot is complete for P001--P020. Development-model selection has been completed without loading hold-out values. The selected DL configuration is the compact CNN, and its final prediction is the arithmetic mean of the five fixed-seed softmax probability vectors followed by argmax. For each seed, the final all-development epoch count is the exact median of the five validation-selected development-fold epochs: 17, 23, 31, 17, and 21 epochs for seeds 1103, 2207, 3301, 4409, and 5519, respectively. No early stopping or validation partition is used during these final refits.

The Random Forest comparator uses the authoritative historical 45-feature TSFEL matrices. Its grouped inner loop recomputes participant/sub-activity trimming from each inner training fold, leaves validation rows at their observed distribution, and fits variance, correlation, and ANOVA selection before each classifier fit. The frozen final tuple is entropy, 1,000 trees, and maximum depth 20; the final pipeline selects 25 features after fitting all transformations on development data only. The all-development refit uses random seed 42.

## Implemented software contract

The evaluation code now builds fixed-label confusion matrices from aligned retained predictions and reports accuracy, balanced accuracy, macro and weighted F1, class-wise precision/recall/F1, and the same metrics separately for each participant. Unknown labels, misaligned prediction vectors, and missing participant identifiers are rejected. Development prediction artefacts retain ordered logits and path-free metadata, bind them to the immutable training record and exact model state, and have no external-hold-out purpose. One immutable derived-analysis record binds the SHA-256 digest of the exact prediction file and regenerates classification, uncalibrated probability-quality, and unsmoothed temporal summaries under declared settings; aggregate outputs will be taken from this record rather than entered manually.

Participant-grouped percentile bootstrap intervals, exact paired sign-flip enumeration, and Holm adjustment are implemented and deterministic under a declared seed. The bootstrap is explicitly labelled `highly_unstable_descriptive_interval` when fewer than five participants are available. These are software implementations of this plan, not permission to analyse the hold-out cohort and not evidence that the methods are confirmatory.

`configs/holdout_evaluation_policy.json` remains disabled until the five CNN refits, RF refit, strict model reloads, immutable final-stage source manifest, and frozen-model manifest have all passed verification. Authorization must bind the protocol configuration, frozen-model manifest, and this statistical analysis plan by SHA-256; match the four protocol hold-out participants; and provide one authorization identifier. The access ledger is written exclusively before any hold-out array or feature value is read. An existing ledger blocks reuse even if evaluation fails after the claim; failures require explicit human adjudication and cannot trigger an automatic retry.

## Endpoints and estimands

Co-primary descriptive endpoints are macro F1-score and balanced accuracy. Accuracy is retained for continuity with the conference paper. Secondary endpoints are weighted F1, class-wise precision/recall/F1, participant-level metrics, and computational cost.

Probability-quality endpoints are exploratory: negative log-likelihood, multiclass Brier score, and top-label expected calibration error with 15 equal-width bins. Non-empty bin counts, mean confidence, accuracy, and absolute gaps will be retained. No probability calibration is fitted or applied; uncalibrated softmax probabilities and native RF probabilities are final.

Temporal-consistency endpoints are exploratory for the CNN only: predicted transition rate, transition-event disagreement rate, excess predicted runs, runs of no more than two windows, and median predicted run length. No temporal smoothing is applied. CNN sequences are delimited by participant, hashed recording identity, historical file-token stream, and exact 2,500-sample steps. The authoritative RF matrices preserve participant and activity labels but not an exact link from every feature row to the reconstructed raw-window sequence. RF temporal diagnostics are therefore not computed, and no raw-to-feature rowwise pairing is claimed. Both probability and temporal families are window- or sequence-level diagnostics and do not increase the participant-level inferential sample size.

For a comparable final model pair, the primary comparative estimands are:

1. the mean participant-level difference in macro F1, selected DL minus reproduced RF; and
2. the mean participant-level difference in balanced accuracy, selected DL minus reproduced RF.

Window-level confusion counts describe errors but are not treated as independent inferential observations.

## Model selection

Architecture, preprocessing, augmentation, regularisation, and stopping choices use development participants only. The current comparison is frozen in `configs/development_model_selection.json` and its hash-bound non-predictive plan. For each candidate and seed, participant-level metrics are collected across shared out-of-fold predictions so every development participant contributes once. Candidate performance is the mean of the five seed-level participant means; the population standard deviation across seed means describes optimization sensitivity.

The residual TCN is selected over the compact CNN only if its mean participant macro F1 gain is at least 0.01, its mean participant balanced-accuracy difference is at least -0.005, and at least 60% of paired seed-level macro-F1 differences are non-negative. Otherwise the smaller CNN remains the conservative fallback. This asymmetric rule was frozen before participant execution because the TCN is larger and slower under the software profile. The TCN did not satisfy the promotion rule, so the compact CNN is selected. The hold-out set is evaluated only after the selected configuration and preprocessing are frozen.

Calibration and temporal smoothing were not promoted into the final model. The primary CNN output is the uncalibrated, unsmoothed five-seed mean probability vector; the primary RF output is its uncalibrated native class-probability vector. Hold-out labels cannot be used to fit or tune any transform.

## Repeated seeds and folds

Each candidate configuration used the ordered seeds 1103, 2207, 3301, 4409, and 5519 across the same five participant-grouped development folds, giving 50 completed training runs. Predictions, not only aggregate metrics, are retained. Seed-level variability describes optimisation sensitivity; it does not increase the number of independent participants. Participant-grouped development folds are shared across models to preserve pairing. Learning curves are averaged by candidate and epoch with the contributing-run count retained because later epochs otherwise become survivor averages after early stopping.

## Hold-out uncertainty and tests

All four hold-out participants will be reported individually. CNN and RF predictions are evaluated in their respective authoritative row orders. Exact participant-by-class window counts must agree before any model comparison is reported; the paired analysis unit is the participant, not the row. With only four independent test participants, conventional asymptotic intervals and hypothesis tests are weakly supported. An exact paired sign-flip/randomisation test has only 16 possible assignments and a minimum attainable two-sided p-value of 0.125, so it cannot supply conventional 0.05-level evidence. The exact test is retained as a descriptive diagnostic for each co-primary endpoint. Participant-grouped percentile bootstrap intervals use 10,000 resamples, 95% coverage, and seed 1103; they are labelled highly unstable and interpreted descriptively.

The final emphasis is therefore the effect estimate, all participant-level differences, development-cohort stability, and the compatibility of the direction across participants. Exact p-values are not interpreted as confirmatory evidence.

## Multiplicity

The paired descriptive family contains one selected DL-versus-RF comparison on two co-primary endpoints. Holm adjustment is applied to the two exact sign-flip p-values. Other model, class, side, preprocessing, probability-quality, temporal, seed-sensitivity, and sub-activity analyses are exploratory and are labelled accordingly.

## Missingness and exclusions

Exclusions follow the frozen QA manifest before predictions are inspected. No participant or window is removed because of model error. Prerequisite failures block before access is claimed. Any software or data failure after the access ledger is written consumes the single authorization, removes partial result artifacts, writes a failure record, and permits no automatic rerun.

## Practical significance

Interpretation will jointly consider participant consistency, class-specific changes (especially sitting and standing), uncertainty, parameter count, model size, and inference cost. A nominally positive mean difference will not be called superior if it is unstable across participants or depends on a methodologically non-equivalent RF comparison.

## Frozen final choices

- Development validation design and participant-grouped fold assignments: frozen in the split manifest and selection plan.
- Selected DL model: compact CNN under the predeclared conservative CNN-versus-TCN rule.
- DL seeds and aggregation: seeds 1103, 2207, 3301, 4409, and 5519; arithmetic mean of uncalibrated softmax probabilities; argmax primary prediction.
- Final DL epoch counts: per-seed median of the five selected development-fold epochs, fixed before all-development refitting.
- RF input and model: authoritative 45-feature matrices; frozen entropy/1,000-tree/depth-20 tuple; 25 selected features; seed 42.
- Hold-out access: one access claim written before value loading; no automatic retry after a claimed failure.
- Co-primary paired estimands: participant-level macro-F1 and balanced-accuracy differences, CNN minus RF.
- Uncertainty and multiplicity: 10,000 participant bootstrap resamples at 95% with seed 1103; exact paired sign-flip diagnostics; Holm adjustment across two endpoints.
- Calibration: none fitted or applied; 15-bin probability-quality diagnostics only.
- Temporal analysis: no smoothing; CNN unsmoothed diagnostics with 2,500-sample adjacency and a two-window short-run threshold; RF temporal analysis unavailable because exact feature-row sequence provenance is absent.
