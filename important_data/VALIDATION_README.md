# Corrected FDA Raman Validation Notes

## What Was Fixed

- Preprocessing now starts from the existing structured tabular datasets:
  `src/combined_raman_spectra.xlsx` plus the explicit non-bacterial D2 control
  in `src/combined_raman_spectra (Experimental).xlsx`. The previous generated
  pipeline mixed raw-file loading and normalization-first behavior; this pass
  applies the documented sequence: Savitzky-Golay smoothing, ALS baseline
  correction, negative clipping, and L2 normalization.
- Each preprocessed spectrum is fitted with a cubic B-spline (`splrep`, `k=3`,
  `s=1e-4`) before functional evaluation on the common 400-3500 cm-1 grid.
  This is the FDA step: it treats spectra as functions and makes the spline
  mean distinct from the discrete interpolated arithmetic mean.
- The Wasserstein curve is reframed as a shared dominant-peak locator. Low-reg
  Sinkhorn trials are recorded in
  `corrected_pipeline/wasserstein_sinkhorn_regularization_trials.csv`; the
  stable fallback used for peak identification is `exact_quantile_fallback`.
- Notebook 2-style NMF, wavelet, variance weights, and thresholds were rerun
  against the corrected functional control matrix.
- Notebook 3-style validation was run against the 9 bacterial spectra in the
  structured experimental workbook and the uploaded raw unadulterated samples.

## Region Definitions and Citations

- Fingerprint region: this implementation uses 400-1800 cm-1 because the
  reference peak table includes milk-relevant protein, lipid, and carbohydrate
  bands throughout that span. Literature boundaries vary: biomedical Raman work
  commonly uses 400-1800 cm-1, while other summaries use roughly 600-1800 or
  800-1800 cm-1.
- CH-stretch region: this implementation uses 2800-3000 cm-1, within the
  broader high-wavenumber region often described as 2800-3800 cm-1. The upper
  bound is narrowed to the instrument range and the supplied milk table, where
  fatty-acid peaks at 2865 and 2902 cm-1 are central.
- Sources consulted:
  - https://pmc.ncbi.nlm.nih.gov/articles/PMC2715834/
  - https://pmc.ncbi.nlm.nih.gov/articles/PMC10052158/
  - https://physicsopenlab.org/2022/01/11/raman-spectroscopy-of-organic-and-inorganic-molecules/
  - https://pubs.acs.org/doi/10.1021/acs.analchem.5c05031

## Corrected Thresholds

| distance_metric | mean | std | 90th_percentile | 95th_percentile | 99th_percentile | chosen_threshold |
| --- | --- | --- | --- | --- | --- | --- |
| l2 | 0.399553 | 0.241383 | 0.704252 | 0.763464 | 1.26045 | 0.763464 |
| weighted_l2 | 0.194326 | 0.101935 | 0.325129 | 0.359814 | 0.535014 | 0.359814 |
| auc | 5.52886 | 3.37414 | 8.78063 | 10.4576 | 14.4122 | 10.4576 |
| wasserstein_1d | 127.044 | 65.8004 | 213.899 | 241.535 | 298.397 | 241.535 |

## Validation Summary

| sample_group | n | mean_authenticity_score | percent_above_threshold | auroc_combined_score | best_individual_metric | best_individual_metric_auroc |
| --- | --- | --- | --- | --- | --- | --- |
| bacterial | 9 | 1.55153 | 100 | 0.982716 | weighted_l2 | 1 |
| control | 45 | 0.529524 | 4.44444 |  |  |  |
| raw_unadulterated | 5 | 0.809939 | 20 |  |  |  |

## Bacterial-vs-Control AUROC by Metric

| metric | auroc_bacterial_vs_control |
| --- | --- |
| weighted_l2 | 1 |
| authenticity_score | 0.982716 |
| l2 | 0.977778 |
| wasserstein_1d | 0.896296 |
| auc | 0.711111 |

## Uploaded Raw Sample Limitation

The user described six unadulterated raw milk samples, but only five files were
present in Downloads: Samples 2-6. Missing file(s): Sample 1 532nm 50% 600g 50xL 10s 10times 400-3500cm-1 200hole.txt. Because the
uploaded raw set is all labelled unadulterated and contains no adulterated
counter-class, AUROC is not computed for that set; raw scores are reported.
