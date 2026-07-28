# Public release scope

## Exact scientific source

The Python modules under `src/prevoccupai_har/` and experiment entry points
under `scripts/` were copied byte-for-byte from the frozen final scientific
source used for the journal analysis.

The model, preprocessing, segmentation, QA-reconstruction, and Random Forest
method configurations are retained. Two governance adaptations are deliberate:

1. hardware identifiers were replaced by synthetic example identifiers; and
2. the public holdout policy is disabled and contains no scientific
   authorization hashes or access ledger.

The test harness was adapted only where an internal artifact or a separate
legacy checkout had previously been required. These adaptations do not modify
the released analysis implementation.

## Excluded material

The public repository excludes:

- raw and processed participant signals;
- participant-level feature matrices and window stores;
- window-level metadata, labels, logits, and predictions;
- real device identifiers;
- fitted CNN, TCN, and Random Forest objects;
- final-access authorization records and the consumed holdout ledger;
- local filesystem paths; and
- the manuscript source and internal scientific-audit workspace.

## Reproducibility boundary

The repository supports software validation, method inspection, configuration
inspection, and synthetic execution. Reproducing the participant results also
requires governed study data and the non-public integrity records that bind
those data to the frozen analysis. The external holdout was already consumed
once and must not be re-evaluated to refresh or adapt the reported results.

