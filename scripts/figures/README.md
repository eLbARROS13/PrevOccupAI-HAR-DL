# Figure-generation Scripts

`generate_model_architecture_figure.py` creates the manuscript methods figure directly from the validated compact-CNN and residual-TCN configurations. It verifies their shared input, class vocabulary, and protocol binding; computes the exact parameter counts from the model implementations; and writes a path-free checksum manifest. It reads no participant data or performance value.

```bash
python3 scripts/figures/generate_model_architecture_figure.py \
  --cnn-config configs/cnn_1d.json \
  --tcn-config configs/tcn_1d.json \
  --output-dir /path/to/new-method-figure-directory
```

The output directory must not already exist. Regenerate the figure whenever either model configuration or implementation changes; do not edit the PDF manually.

`generate_prediction_figures.py` creates vector-PDF confusion-matrix, participant-macro-F1, and calibration-reliability figures from one immutable derived-analysis record. It writes an identifier-free manifest binding the exact analysis, prediction, payload, and output-file digests. Participant tick labels are ordinal; no raw path or filename is retained. The output directory must not already exist.

Install the optional figure dependency and run:

```bash
python3 -m pip install -e '.[dl,figures]'
python3 scripts/figures/generate_prediction_figures.py \
  --analysis /path/to/frozen-analysis.json \
  --output-dir /path/to/new-figure-directory
```

Only figures generated from authorised scientific records may be copied into the manuscript's `figures/` directory. Synthetic outputs are software validation and must remain outside the manuscript.

`generate_selection_reports.py` creates paired-seed, learning-curve, and complexity--performance vector PDFs plus candidate-level and epoch-level CSV files from one validated model-selection bundle. Its identifier-free manifest binds the exact bundle, bundle payload, selection plan, and every output digest. Run it only after the complete bundle exists:

```bash
python3 scripts/figures/generate_selection_reports.py \
  --bundle /path/to/frozen-selection-bundle.json \
  --output-dir /path/to/new-selection-report-directory
```

Synthetic selection reports validate the reporting software only and must not be copied into the manuscript.
