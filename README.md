# TeloSig

**Inferring telomere maintenance mechanisms from tumor transcriptomes**

Can routine RNA-seq reveal *how* a tumor maintains its telomeres — telomerase or ALT — beyond what its
cancer type already tells us? And what is the smallest gene set that does it?

## Study design

| Task | Label (source) | Cohort | Genes removed from inputs |
|---|---|---|---|
| `alt` (primary) | ALT-like = ATRX/DAXX-altered and TERT-negative (Barthel et al. 2017) | LGG, GBM, SARC (n=872) | TERT |
| `alt_noATRX` | same | same | TERT, ATRX, DAXX |
| `alt_pan` | same | 31 cancer types (n=6,788) | TERT |
| `alt_pheno` | ALT by WGS telomere-sequence features (Sieverling et al. 2020) | PCAWG-TCGA donors (n=697) | none |
| `tel` (primary) | TERT-expressing vs not (Barthel et al. 2017) | 31 cancer types (n=6,788) | TERT |
| `tl` (secondary) | log tumor/normal telomere-length ratio (TelSeq) | WGS/LPS samples (n=1,698) | none |

Feature sets compared on identical folds:

* **References** — cancer type only; cancer type + glioma molecular subtype.
* **Published methods** — Barthel et al. telomerase score and genes; EXTEND score (uses TERT/TERC) and
  EXTEND genes without TERT/TERC.
* **Curated telomere panel** (63 genes) — LR, random forest, gradient boosting, MLP (default and tuned).
* **Whole transcriptome** — L2 LR on all 19,254 protein-coding genes.
* **Compact signature** — genes selected *inside each training fold* (L1 and univariate ranking, K = 10–200);
  a locked 60-gene signature with weights and scaling is released for reuse.

## Evaluation principles

* Primary metric: **within-cancer AUROC** — a model that only recognises tissue of origin scores 0.5.
* 5-fold × 5 repeated CV, stratified bootstrap CIs, paired differences, TOST equivalence (±0.02).
* Curated panel vs 100 random gene sets; ALT inside single glioma molecular subtypes.
* Validation beyond CV: held-out donors with whole-genome labels; the same tumors quantified by an independent
  pipeline (PCAWG); cell lines with measured telomerase activity; Cox models of outcome; pathway enrichment.

## Running

Kaggle (CPU is enough):
```
python push_to_kaggle.py                                          # data + jobs J1-J3 (benchmark, signatures)
python push_to_kaggle.py --kernels-only --only tmm-v2-validate    # J4 after J1-J3 complete
python push_to_kaggle.py --fetch                                  # download outputs
python make_figures.py                                            # figures + LaTeX tables locally
```

Locally, stage by stage:
```
pip install -r requirements.txt
python data_build.py                  # 1  labels, panel expression, random pool, survival
python data_build.py --extra          # 1b EXTEND scores, cell lines (assays + transcriptomes)
python data_build.py --transcriptome  # 1c whole protein-coding transcriptome (TCGA Toil + PCAWG), ~1.5 GB download
python run_experiments.py             # 2  benchmark (add --quick for a smoke test)
python signature.py                   # 2b compact signatures + locked signatures
python validate.py                    # 3  external, platform, cell lines, survival, enrichment
python analyze.py                     # 3b CIs, paired comparisons, TOST, subtype control, size curve
python make_figures.py                # 4  figures and results/tables.tex
```

## Files

| File | Role |
|---|---|
| `config.py` | paths, sources, gene panel, task definitions, signature and model settings |
| `data_build.py` | downloads and harmonises all sources |
| `models.py` | logistic regression, random forest, gradient boosting, MLP, published-score wrapper |
| `evaluate.py` | folds, within-cancer metrics, bootstrap inference, TOST, permutation importance |
| `run_experiments.py` | cross-validated benchmark of feature sets and models |
| `signature.py` | in-fold signature selection, size curve, stability, locked signatures |
| `validate.py` | whole-genome labels, quantification robustness, cell lines, survival, enrichment |
| `analyze.py` | merges all predictions and computes the reported statistics |
| `make_figures.py` | figures and LaTeX tables |
| `push_to_kaggle.py` | private Kaggle dataset + CPU jobs |
| `data/frozen/*_signature_locked*.json` | released signatures (genes, weights, scaling) |

Graph neural networks on the STRING telomere network were evaluated in an earlier version of this project
(GraphTelNet; notebook kept in `legacy/`) and added nothing over graph-free models, so they are no longer part
of the pipeline.

## Data sources

* Barthel FP *et al.* *Nat Genet* 2017;49:349–357. · Sieverling L *et al.* *Nat Commun* 2020;11:733.
* Noureen N *et al.* (EXTEND) *Nat Commun* 2021;12:139. · Vivian J *et al.* (Toil) *Nat Biotechnol* 2017.
* Liu J *et al.* (TCGA-CDR) *Cell* 2018. · Ghandi M *et al.* (CCLE) *Nature* 2019. · PCAWG, *Nature* 2020.
