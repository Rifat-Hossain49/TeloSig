# TeloSig

**Confounder-aware transcriptomic inference of telomere-maintenance mechanisms**

This pipeline asks what tumor RNA-seq contributes to ALT and telomerase inference after controlling label
circularity, tumor type, molecular subtype and copy-number-driven expression. Classical models run on CPU and
the nested MLP candidates use a GPU when available. This repository releases the corrected version 3 analysis:
compact outputs are in `results_v3/`, locked models are in `data/frozen_v3/`, and rendered figures are in
`figures_v3/`. Historical version 2 artifacts are intentionally excluded from this release.

## Analysis design

| Task | Target | Main cohort | Expression genes excluded |
|---|---|---|---|
| `alt` | ATRX/DAXX-altered and TERT-negative proxy | LGG, GBM, SARC | TERT |
| `alt_noATRX` | same proxy, ablation | LGG, GBM, SARC | TERT, ATRX, DAXX |
| `alt_pan` | same proxy | 31 TCGA cancer types | TERT |
| `alt_pheno` | WGS telomere-sequence ALT phenotype | PCAWG-TCGA donors | none |
| `tel` | detectable TERT expression | 31 TCGA cancer types | TERT |
| `tel_no5p15`, `tel_no5p` | telomerase cis-region sensitivities | 31 TCGA cancer types | TERT plus indicated region |
| `tl` | log tumor/normal TelSeq ratio | WGS/LPS samples | none |

The benchmark includes cancer/subtype references, correctly named published/gene-set comparators, a curated
63-gene panel, a compact in-fold signature, and whole-transcriptome linear models. Classical baselines include
L2 logistic regression, elastic-net logistic regression and a linear SVM; panel-only models also include random
forest, histogram gradient boosting and MLPs.

Safeguards:

- Every tunable decision is repeated inside each outer training set; signature L1 strength and MLP settings are
  never reused across outer folds.
- PCAWG donors are removed before filtering, tuning, selection, scaling, fitting and threshold selection for the
  held-out locked model.
- Locked scores always use TCGA-training means/SDs. Target-cohort re-standardization is forbidden.
- Primary performance is sample-weighted within-cancer AUROC from repeated 5-fold outer CV. Pooled metrics are
  secondary.
- The curated panel is compared with 1,000 random panels under the same fixed LR.
- Outer-fold selection frequency is descriptive; a separate 100-subsample analysis evaluates gene-list
  reproducibility.
- CNV-only, label-locus-excluded CNV, chromosome-region ablation, fold-wise CNV residualization and expression +
  CNV models make karyotype a central analysis.

## Output isolation

Never merge v3 into the historical paths. Set these environment variables for local post-processing:

```powershell
$env:TMM_RESULTS_DIR = 'results_v3'
$env:TMM_FROZEN_DIR = 'data/frozen_v3'
$env:TMM_FIGURES_DIR = 'figures_v3'
```

`config.py` resolves relative paths below the repository root. The v3 collector also refuses the historical
v2 destinations.

## Full v3 run

The benchmark is sharded to remain below Kaggle's per-kernel runtime limit:

```powershell
python push_to_kaggle.py --dataset-only
python push_to_kaggle.py --kernels-only
# Wait for the primary, auxiliary, sensitivity and pan-cancer shards to complete.
python push_to_kaggle.py --kernels-only --only tmm-v3-confounding
python push_to_kaggle.py --kernels-only --only tmm-v3-validate
python push_to_kaggle.py --fetch --only tmm-v3-alt tmm-v3-altaux tmm-v3-tel tmm-v3-telsens tmm-v3-altpan tmm-v3-altpan-sig tmm-v3-confounding tmm-v3-validate
python collect_v3.py
$env:TMM_RESULTS_DIR = 'results_v3'
$env:TMM_FROZEN_DIR = 'data/frozen_v3'
$env:TMM_FIGURES_DIR = 'figures_v3'
python audit_v3.py
python make_figures.py
```

The validation kernel depends on the six benchmark shards. The confounding kernel is independent but uses
the same input dataset. See `REPRODUCIBILITY.md` for the release audit and a clean-room verification workflow.

## Local stages

```powershell
python data_build.py
python data_build.py --extra
python data_build.py --transcriptome
python data_build.py --cnv
python data_build.py --clinical
python run_experiments.py
python signature.py
python data_build.py --wu2025
python confounding.py
python validate.py
python analyze.py
python make_figures.py
```

These commands are documented for reproducibility, not as a request to run them without compute quota.

## Key files

| File | Purpose |
|---|---|
| `config.py` | sources, tasks, paths, seeds and prespecified settings |
| `data_build.py` | labels, expression, CNV, clinical and external data assembly |
| `models.py` | fold-local preprocessing and model implementations |
| `run_experiments.py` | panel/transcriptome benchmark and random-panel null |
| `signature.py` | nested signature selection, subsampling reproducibility and locked artifacts |
| `confounding.py` | central karyotype/CNV experiment |
| `validate.py` | WGS labels, technical concordance, Wu 2025/legacy cell lines, survival and enrichment |
| `analyze.py` | bootstrap summaries, paired differences and equivalence tests |
| `collect_v3.py` | non-destructive result collection with hashes |
| `audit_v3.py` | read-only acceptance checks for a completed run |
| `METHODS_AUDIT.md` | reviewer concern-to-implementation traceability |

## Data provenance

Expression and annotations come from TCGA Toil, PanCanAtlas and PCAWG datasets distributed through UCSC Xena;
gene-level CNV is the TCGA Pan-Cancer GISTIC2 whole-genome-microarray matrix matched to the same
TCGA cohort, and tumor purity is the open PanCanAtlas ABSOLUTE call. Other inputs are the Barthel, Sieverling and EXTEND supplementary data, the Wu 2025 qTRAP/C-circle
atlas, the fixed 2022 Cell Model Passports RNA-seq archive, PCAWG identifier maps and CCLE expression. Exact
source identifiers are fixed in `config.py`.

## Citation

If you use TeloSig, cite the associated manuscript and the software metadata in `CITATION.cff`.

## License

The code is released under the [MIT License](LICENSE). Input data remain subject to the terms of their original
public sources.
