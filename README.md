# Raman Milk FDA Colab Analysis

Corrected, real-data Raman spectroscopy workflow for milk authenticity analysis.

This repository contains three Google Colab-ready notebooks plus the important
outputs from the corrected run. No synthetic spectra are used.

## Run In Colab

## Corrected Workflow

- Structured controls: 45 spectra from repository workbooks.
- Preprocessing: Savitzky-Golay smoothing, ALS baseline correction, negative
  clipping, L2 normalization.
- FDA representation: cubic B-splines with `splrep/splev`, `k=3`, `s=1e-4`.
- Consensus outputs: arithmetic mean, spline mean, robust functional median,
  and Wasserstein peak locator.
- Biochemical analysis: variance weighting, distance thresholds, NMF, wavelet
  decomposition, and Wasserstein low-regularization audit.
- Validation: bacterial spectra from the experimental workbook plus uploaded
  raw milk files.

## Important Outputs

- `important_data/VALIDATION_README.md`
- `important_data/corrected_pipeline/`
- `important_data/validation/`
- `important_data/important_diagrams_and_statistics.pdf`

## Headline Results From The Local Run

- Controls: 45.
- Bacterial validation spectra: 9.
- Uploaded raw spectra found locally: 5; Sample 1 was missing.
- Bacterial-vs-control combined authenticity AUROC: 0.982716.
- Best individual metric: `weighted_l2`, AUROC 1.0.
- Raw unadulterated samples above the score threshold: 20%.

