# Methods audit: reviewer concerns mapped to accepted v3

| Reviewer concern | v3 implementation and accepted evidence | Status |
|---|---|---|
| Hyperparameters reused across outer folds | Signature L1 strength and MLP search are selected independently within every outer training set. | Passed |
| PCAWG leakage through preselection | Donors are removed before filtering, tuning, selection, scaling, fitting and threshold selection. | Passed |
| External cohort re-standardisation | Locked scoring uses immutable TCGA means/SDs; target-cohort scaling is forbidden. | Passed |
| TERT/TERC handling in EXTEND | Published EXTEND is evaluated only on independent activity; the TCGA gene-set LR removes TERT and retains TERC. | Passed |
| Weak classical baselines | Elastic-net LR and linear SVM are included for panel and transcriptome feature sets. | Passed |
| 5p15 reconstructs the TERT locus | Both 5p15 and whole-5p telomerase sensitivities completed; transcriptome AUROC remained 0.808. | Passed |
| Karyotype drives the signal | Matched CNV-only, label-locus-excluded CNV, region exclusions, fold-wise residualisation and joint models completed. | Passed |
| PCAWG is not independent patients | Manuscript calls this held-out label/assay validation in overlapping TCGA donors. | Passed |
| Pooled WGS results obscure transfer | Overall and eligible per-cancer estimates are reported; within-cancer ALT transfer was 0.519–0.542. | Passed |
| ALT threshold dependence | WGS probability cutoffs 0.25, 0.50 and 0.75 are reported. | Passed |
| Proxy-label mismatch | A direct WGS-phenotype task completed; transcriptome within-cancer AUROC was 0.716. | Passed |
| Cell-line validation underpowered | Wu 2025 provides 947 matched expression/activity lines and 914 unambiguous ALT/TEL lines. | Passed with disclosed limitation |
| Missing external signature genes | 55–58/60 genes are observed; absent terms use locked training means and coverage is reported. | Passed with disclosed limitation |
| Survival confounding | Primary and exploratory purity/proliferation/stemness/stage/grade Cox models report complete cases and FDR. | Passed; no FDR-significant association |
| RNA pipeline comparison overclaimed | Described as same-donor technical concordance only. | Passed |
| Equivalence margin unexplained | ±0.02 was prespecified; the ALT compact signature did not meet equivalence. | Passed |
| “Stability selection” misnomer | Outer-fold frequency is descriptive; reproducibility uses 100 independent 80% subsamples. | Passed |
| Too few random panels | 1,000 matched fixed-LR random panels completed per primary task. | Passed |
| Multiplicity unclear | Primary contrasts are named; survival and exploratory families report adjusted results. | Passed |
| Locked model not deployable | JSON artifacts contain full coefficients, preprocessing parameters, formula and threshold. | Passed |
| v2/v3 result mixing | Versioned collector, hashes, explicit v3 paths and read-only audit prevent mixing. | Passed |

The remaining scientific limitation cannot be repaired computationally: no independent patient cohort currently
provides matched RNA-seq and direct C-circle/APB or equivalent phenotypic ALT measurements at adequate scale.
