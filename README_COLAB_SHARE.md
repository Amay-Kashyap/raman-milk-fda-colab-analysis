# Colab-Shareable Raman Analysis Folder

This folder contains the corrected, real-data Raman workflow in a shareable form.

## Notebooks

Run these in order in Google Colab:

1. `notebooks/01_corrected_fda_consensus_pipeline.ipynb`
2. `notebooks/02_corrected_biochemical_analysis_pipeline.ipynb`
3. `notebooks/03_authenticity_validation_pipeline.ipynb`

Each notebook is Colab-ready. The notebooks clone the public repository for the
existing workbook data and then recreate the corrected helper pipeline inside
the Colab runtime, so they do not depend on a private Windows path.

For notebook 3, Colab will prompt you to upload the raw milk `Sample 1-6` text
files. If only Samples 2-6 are uploaded, the notebook reports Sample 1 as
missing and continues without fabricating data.

## Helper

- `helper/run_corrected_fda_and_validation.py`

The same helper code is embedded in each notebook for shareability. The helper
copy is included here for inspection and reuse.

## Important Data

- `important_data/VALIDATION_README.md`: concise statistical summary and notes.
- `important_data/corrected_pipeline/`: corrected preprocessing, FDA consensus,
  thresholds, NMF, wavelet, and Wasserstein audit outputs.
- `important_data/validation/`: bacterial-vs-control validation and uploaded
  raw milk sample scoring outputs.
- `important_data/important_diagrams_and_statistics.pdf`: compact report with
  key plots and tables, if present.

## Headline Results From This Run

- Controls: 45 spectra from structured repo workbooks.
- Bacterial validation spectra: 9.
- Uploaded raw spectra found locally: 5; Sample 1 was missing.
- Bacterial-vs-control combined authenticity AUROC: 0.982716.
- Best individual metric: weighted_l2, AUROC 1.0.
- Raw unadulterated samples above the score threshold: 20%.

No synthetic spectra were used.
