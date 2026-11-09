# Changelog

## v1.0.1 (2026-11-08)

Corrections made during the final pre-submission check of the manuscript.

### Analysis
- **WGS ALT labels.** Five PCAWG-mapped TCGA donors have no WGS ALT probability (TCGA-BT-A3PH, TCGA-21-1078,
  TCGA-24-1552, TCGA-HC-7075, TCGA-MO-A47R). `data_build.py` previously coded them ALT-negative
  (`NaN > 0.5` is `False`), and `validate.external()` kept them. They are now unlabelled for every ALT
  analysis, matching the phenotype-trained model and the ALT-probability threshold analysis (n=697).
  `results_v3/validation_external.csv` was recomputed for the ALT tasks (TERT rows unchanged):
  - fixed glioma/sarcoma ALT signature: pooled AUROC 0.792 (82 donors, 31 ALT; previously 83);
    eligible within-cancer AUROC 0.504 in SARC (n=33; previously 0.519, n=34);
  - fixed pan-cancer ALT signature: pooled 0.692, within-cancer 0.536 (n=697; previously 0.542, n=702).
  The recomputation used scikit-learn 1.6.1; in that environment the unmodified code reproduced every
  released ALT row exactly. See `post_collection_corrections` in `results_v3/collection_manifest.json`.

### Figures and tables (`make_figures.py`)
- The TERT task is labelled "TERT expression proxy" instead of "Telomerase activation".
- `GBM (panel)` (histogram gradient boosting) is displayed as `HGB (panel)`.
- Figure 1 redrawn as a vector study-design diagram; survival forest plot lists estimates in a separate
  aligned column; Figure S4 titles no longer overlap; Figure S5 axis label no longer clipped.
- Supplementary Table S2 reports the computed cross-validated cancer-type-only within-cancer metrics
  (previously printed as "0.5 (by design)"). Paired-comparison p values below the bootstrap resolution
  are shown as "<0.002" (1,000 resamples) instead of "<0.001".
- Figures are written at 600 dpi.

### Metadata
- `CITATION.cff` version aligned with the release tag; `requirements.txt` pins the versions used for the
  released analysis; `METHODS_AUDIT.md` wording updated.

## v1.0.0 (2026-10-02)

Original manuscript release (Zenodo 10.5281/zenodo.23104719).
