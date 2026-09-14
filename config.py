"""Central configuration for the telomere-maintenance-mechanism (TMM) pipeline.

Every stage imports from here so that paths, gene panels, seeds and model settings
are defined exactly once.
"""
import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths. On Kaggle everything lives under /kaggle/working; locally next to this file.
# ---------------------------------------------------------------------------
ROOT = Path(os.environ.get("TMM_ROOT", Path(__file__).resolve().parent))
RAW = ROOT / "data" / "raw"            # downloaded source files (never edited)
PROC = ROOT / "data" / "processed"     # analysis-ready matrices
FROZEN = ROOT / "data" / "frozen"      # small files committed to git (locked signatures, gene panel)
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
for _p in (RAW, PROC, FROZEN, RESULTS, FIGURES):
    _p.mkdir(parents=True, exist_ok=True)

SEED = 42

# ---------------------------------------------------------------------------
# External resources
# ---------------------------------------------------------------------------
XENA_TOIL = "https://toil.xenahubs.net"
XENA_PANCAN = "https://pancanatlas.xenahubs.net"
TOIL_EXPR = "tcga_RSEM_gene_tpm"                 # stored as log2(TPM + 0.001): do NOT log again
PANCAN_SUBTYPES = "TCGASubtype.20170308.tsv"
PANCAN_GENE_LIST_DS = "EB++AdjustPANCAN_IlluminaHiSeq_RNASeqV2.geneExp.xena"  # only used to list gene symbols

_NATURE = "https://media.springernature.com/original/springer-static/esm/art%3A10.1038%2F"
BARTHEL_ST1 = _NATURE + "ng.3781/MediaObjects/41588_2017_BFng3781_MOESM72_ESM.xlsx"
BARTHEL_ST4 = _NATURE + "ng.3781/MediaObjects/41588_2017_BFng3781_MOESM75_ESM.xlsx"
SIEVERLING_SD1 = _NATURE + "s41467-019-13824-9/MediaObjects/41467_2019_13824_MOESM3_ESM.xlsx"
_PCAWG_OPS = "https://raw.githubusercontent.com/ICGC-TCGA-PanCancer/pcawg-operations/develop/lists/"
PCAWG_SAMPLE_SHEET = _PCAWG_OPS + "sample_sheet/pcawg_sample_sheet.2016-10-18.tsv"   # ICGC specimen -> TCGA UUIDs
PCAWG_UUID2BARCODE = _PCAWG_OPS + "pc_annotation-tcga_uuid2barcode.tsv"            # TCGA UUID -> barcode
XENA_PUBLIC = "https://ucscpublic.xenahubs.net"
CCLE_EXPR = "ccle/CCLE_DepMap_18Q2_RNAseq_RPKM_20180502"      # log2(RPKM+1) per cell line
_EXTEND = _NATURE + "s41467-020-20474-9/MediaObjects/41467_2020_20474_MOESM"
EXTEND_SIGNATURE = _EXTEND + "4_ESM.xlsx"    # Supplementary Data 1: the 13-gene signature
EXTEND_TCGA_SCORES = _EXTEND + "7_ESM.xlsx"  # Supplementary Data 4: published EXTEND score per TCGA sample
EXTEND_SOURCE_FIG1 = _EXTEND + "11_ESM.xlsx" # Source data Fig. 1: cell-line telomerase assays
# Whole transcriptome: the Toil TPM matrix is downloaded once and reduced to protein-coding genes.
TOIL_MATRIX = "https://toil-xena-hub.s3.us-east-1.amazonaws.com/download/tcga_RSEM_gene_tpm.gz"   # ~740 MB
TOIL_PROBEMAP = "https://toil-xena-hub.s3.us-east-1.amazonaws.com/download/probeMap%2Fgencode.v23.annotation.gene.probemap"
HGNC_COMPLETE = "https://storage.googleapis.com/public-download-files/hgnc/tsv/tsv/hgnc_complete_set.txt"
# Clinical endpoints (TCGA Pan-Cancer Clinical Data Resource, Liu et al. 2018) and PCAWG expression
PANCAN_SURVIVAL = "Survival_SupplementalTable_S1_20171025_xena_sp"
XENA_PCAWG = "https://pcawg.xenahubs.net"
PCAWG_EXPR = "tophat_star_fpkm_uq.v2_aliquot_gl.sp.log"      # independent quantification of the same tumors

# ---------------------------------------------------------------------------
# Curated telomere-maintenance gene panel (HGNC symbols). Grouped by function.
# ---------------------------------------------------------------------------
PANEL = {
    "telomerase":  ["TERT", "TERC", "DKC1", "NHP2", "NOP10", "GAR1", "NAF1", "WRAP53",
                    "RUVBL1", "RUVBL2", "PARN", "SHQ1"],
    "shelterin":   ["TERF1", "TERF2", "TINF2", "TERF2IP", "ACD", "POT1"],
    "CST":         ["CTC1", "STN1", "TEN1"],
    "replication": ["RTEL1", "WRN", "BLM", "DCLRE1B", "PIF1", "RECQL4"],
    "ALT/HR":      ["ATRX", "DAXX", "H3-3A", "SMARCAL1", "FANCM", "SLX4", "MUS81", "BRCA1",
                    "BRCA2", "RAD51", "RAD52", "POLD3", "TOP3A", "RMI1", "RMI2", "PML", "SP100",
                    "NBN", "MRE11", "RAD50", "FANCD2", "SMC5", "SMC6", "NSMCE2"],
    "regulators":  ["MYC", "MAX", "GABPA", "GABPB1", "SP1", "TNKS", "TNKS2", "PINX1", "HNRNPA1"],
    "SUMO/stress": ["G3BP1", "ZNF451", "SENP3"],
}
PANEL_GENES = [g for gs in PANEL.values() for g in gs]

# Current HGNC symbol -> symbols tried in the GENCODE v23 annotation used by the Toil hub.
GENE_ALIASES = {
    "STN1": ["OBFC1"], "TEN1": ["C17orf106"], "H3-3A": ["H3F3A"], "MRE11": ["MRE11A"],
    # Barthel et al. telomerase signature genes listed under retired symbols (verified via HGNC REST)
    "C14orf106": ["MIS18BP1"], "DCC1": ["DSCC1"], "KAL1": ["ANOS1"], "TXNDC1": ["TMX1"],
    "C1orf38": ["THEMIS2"], "C1orf108": ["AKIRIN1"],
}

# ---------------------------------------------------------------------------
# Tasks
# ---------------------------------------------------------------------------
ALT_COHORT = ["LGG", "GBM", "SARC"]          # ALT-prevalent cancers (gliomas and sarcomas)
MIN_PER_CLASS_WITHIN = 10                    # a cancer type enters within-cancer summaries if both classes >= this
TL_RELIABLE_LIBS = ["WGS", "LPS"]            # TL estimates from exome (WXS) are too noisy (replicate rho ~0.3)

# Genes removed from model inputs per task, to keep labels non-circular.
EXCLUDE = {
    "alt": ["TERT"],                         # ALT-like label requires absence of TERT expression
    "alt_noATRX": ["TERT", "ATRX", "DAXX"],  # ablation: can the rest of the network detect ALT?
    "tel": ["TERT"],                         # telomerase label is defined from TERT expression
    "tl": [],                                # TL is measured from DNA sequencing: no overlap with inputs
    "alt_pheno": [],                         # WGS telomere-sequence label is independent of expression
}

# Sensitivity analysis: telomerase task without genes near TERT (5p15.33). Genes whose HGNC cytoband
# starts with the prefix are removed in addition to C.EXCLUDE (e.g. 5p gain co-amplifies TERT neighbours).
REGION_EXCLUDE = {"tel_no5p15": "5p15", "tel_no5p": "5p"}

# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
N_FOLDS = 5
N_REPEATS = 5
N_BOOT = 1000
N_RANDOM_SETS = 100
RANDOM_POOL_SIZE = 3000

# ---------------------------------------------------------------------------
# Compact signature (selected inside each training fold from the whole transcriptome)
# ---------------------------------------------------------------------------
SIG_K = 60                                   # headline size: the size of the curated panel
SIG_K_GRID = [10, 20, 30, 60, 100, 200]      # size-performance curve
SIG_C_GRID = [0.05, 0.15, 0.5]               # L1 strengths; chosen inside the first training fold
SIG_STABLE = 0.8                             # a gene is "stable" if selected in >= 80% of folds

# ---------------------------------------------------------------------------
# Neural models
# ---------------------------------------------------------------------------
NN = dict(hidden=32, dropout=0.2, flat_hidden=128,
          lr=1e-3, weight_decay=1e-4, batch_size=128, max_epochs=300, patience=25, val_frac=0.15)

# Random search over MLP hyperparameters on an inner validation split of the first training fold.
NN_SEARCH = dict(
    n_iter=8,
    space=dict(hidden=[16, 32, 64], flat_hidden=[64, 128, 256], dropout=[0.1, 0.2, 0.4],
               lr=[3e-4, 1e-3, 3e-3], weight_decay=[1e-5, 1e-4, 1e-3]),
)

# Equivalence margin for TOST on within-cancer AUROC: differences smaller than this are
# treated as practically irrelevant when claiming that two models perform equivalently.
EQUIV_MARGIN = 0.02
