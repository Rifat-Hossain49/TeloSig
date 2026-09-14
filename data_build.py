"""Stage 1: download every source dataset and build analysis-ready tables.

Outputs (data/processed):
  labels.csv        one row per TCGA tumor with Barthel et al. (2017) TMM / telomere-length annotations
  expr_panel.csv.gz log2(TPM+0.001) expression for the telomere panel + Barthel signature genes
  expr_random.csv.gz expression for a random gene pool (null distribution of gene sets)
  external_pcawg.csv TCGA donors with independent WGS-based TMM calls (Sieverling et al. 2020), if resolvable
Outputs (data/frozen):
  (graph resources were removed: graph models added nothing in the benchmark)
"""
import io
import json
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import requests

import config as C

warnings.filterwarnings("ignore", category=FutureWarning)
import xenaPython as xena  # noqa: E402


# ---------------------------------------------------------------------------
# Download helpers
# ---------------------------------------------------------------------------
def download(url, dest, retries=5, timeout=600):
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    for attempt in range(1, retries + 1):
        try:
            with requests.get(url, stream=True, timeout=timeout, headers={"User-Agent": "Mozilla/5.0"}) as r:
                r.raise_for_status()
                tmp = dest.with_suffix(dest.suffix + ".part")
                with open(tmp, "wb") as f:
                    for chunk in r.iter_content(1 << 20):
                        f.write(chunk)
                tmp.rename(dest)
            return dest
        except Exception as e:  # network hiccups are common on public hubs
            print(f"  download failed ({attempt}/{retries}): {e}")
            time.sleep(5 * attempt)
    raise RuntimeError(f"Could not download {url}")


# ---------------------------------------------------------------------------
# Labels
# ---------------------------------------------------------------------------
def build_labels():
    st1 = download(C.BARTHEL_ST1, C.RAW / "barthel_ST1.xlsx")
    p = pd.read_excel(st1, sheet_name="Paired Set")
    grp = p["TERTexpr-ATRX/DAXXaltgrp"]
    lab = pd.DataFrame({
        "sample": p["SampleID"],
        "patient": p["PatientID"],
        "cancer": p["Disease"].astype(str),
        "lib": p["LibraryType"],
        "tl_ratio": p["TLratio"],                       # log(tumor TL / matched normal TL), DNA-derived
        "extended": p["isExtended"].astype(bool),       # samples with RNA-based TMM annotation
        "tmm_group": grp,
        "y_alt": np.where(grp.isna(), np.nan, (grp == "ATRX/DAXX alt").astype(float)),
        "y_tel": np.where(grp.isna(), np.nan, (grp == "TERT expr").astype(float)),
        "sig_score": p["TelomeraseSignatureScore"],     # Barthel et al. published telomerase signature score
        "atrx_daxx_status": p["ATRXDAXXstatus"],
        "age": pd.to_numeric(p["Age"], errors="coerce"),
        "sex": p["Gender"],
    })
    assert lab["sample"].is_unique and lab["patient"].is_unique, "expected one tumor per patient"
    return lab


def barthel_signature_genes():
    st4 = download(C.BARTHEL_ST4, C.RAW / "barthel_ST4.xlsx")
    return pd.read_excel(st4, sheet_name="Signature Genes (Sup fig 7)", header=None)[0].astype(str).str.strip().tolist()


def glioma_subtypes(samples):
    """Integrated glioma subtypes (IDH/1p19q) from the PanCanAtlas subtype table."""
    field = "Subtype_Selected"
    vals = xena.dataset_fetch(C.XENA_PANCAN, C.PANCAN_SUBTYPES, samples, [field])[0]
    codes = xena.field_codes(C.XENA_PANCAN, C.PANCAN_SUBTYPES, [field])[0]["code"].split("\t")
    return pd.Series([codes[int(v)] if isinstance(v, (int, float)) and v == v else np.nan for v in vals], index=samples)


# ---------------------------------------------------------------------------
# Expression
# ---------------------------------------------------------------------------
def _fetch_one(genes, ch, retries=4, hub=None, dataset=None):
    """One Xena request: genes x sample chunk -> {gene: values}. Missing genes come back as NaN."""
    for attempt in range(retries):
        try:
            got = {}
            for e in xena.dataset_gene_probe_avg(hub or C.XENA_TOIL, dataset or C.TOIL_EXPR, ch, genes):
                sc = e.get("scores") or []
                v = np.asarray(pd.to_numeric(pd.Series(sc[0] if len(sc) else []), errors="coerce"), dtype=float)
                got[e["gene"]] = v if len(v) == len(ch) else np.full(len(ch), np.nan)
            return {g: got.get(g, np.full(len(ch), np.nan)) for g in genes}
        except Exception as e:
            if attempt == retries - 1:
                raise
            time.sleep(5 * (attempt + 1))


def _fetch_block(genes, samples, g_chunk=10, s_chunk=2000, workers=8, verbose=False, hub=None, dataset=None):
    """The hub answers slowly per request but handles concurrent requests well, so fan out."""
    jobs = [(genes[i:i + g_chunk], j) for i in range(0, len(genes), g_chunk) for j in range(0, len(samples), s_chunk)]
    cols = {g: [None] * len(range(0, len(samples), s_chunk)) for g in genes}
    done = 0
    with ThreadPoolExecutor(workers) as ex:
        futures = {ex.submit(_fetch_one, gs, samples[j:j + s_chunk], hub=hub, dataset=dataset): (gs, j // s_chunk)
                   for gs, j in jobs}
        for fut in as_completed(futures):
            gs, k = futures[fut]
            for g, v in fut.result().items():
                cols[g][k] = v
            done += 1
            if verbose and done % 20 == 0:
                print(f"  expression requests: {done}/{len(jobs)}", flush=True)
    return pd.DataFrame({g: np.concatenate(v) for g, v in cols.items()}, index=samples)


def hgnc_aliases(genes, verbose=True):
    """Ask HGNC for previous/alias symbols of genes the hub does not know.

    The Toil compendium is annotated with GENCODE v23 (2015), so genes renamed since then
    (histone clusters, SHLD1, H2AX, ...) must be requested under their older symbols.
    """
    def one(g):
        try:                                   # short timeout: the service occasionally stalls
            r = requests.get(f"https://rest.genenames.org/fetch/symbol/{g}",
                             headers={"Accept": "application/json"}, timeout=(5, 10)).json()
            docs = r.get("response", {}).get("docs", [])
            return g, [a for a in (docs[0].get("prev_symbol", []) + docs[0].get("alias_symbol", [])) if a != g] if docs else []
        except Exception:
            return g, []
    out = {}
    with ThreadPoolExecutor(8) as ex:
        for g, alts in ex.map(one, genes):
            if alts:
                out[g] = alts
    if verbose:
        print(f"  HGNC lookups for {len(genes)} unknown symbols -> {sum(bool(v) for v in out.values())} with older names")
    return out


def fetch_expression(genes, samples, verbose=True, cache=None):
    """Fetch toil expression; retired/renamed symbols are retried via config.GENE_ALIASES and HGNC.
    Values stay on the hub's log2(TPM+0.001) scale. A cache file supplies genes already fetched."""
    cached = None
    if cache is not None and cache.exists():
        old = pd.read_csv(cache, index_col=0)
        if set(samples) <= set(old.index):
            cached = old.loc[samples]
            genes = [g for g in genes if g not in cached.columns]
            print(f"  cache {cache.name}: {cached.shape[1]} genes reused, {len(genes)} to fetch")
            if not genes:
                return cached, {}, []
    expr = _fetch_block(genes, samples, verbose=verbose)
    empty = [g for g in genes if expr[g].isna().all()]
    used = {}
    lookup = dict(C.GENE_ALIASES)
    for g, alts in hgnc_aliases([g for g in empty if g not in lookup], verbose=verbose).items():
        lookup.setdefault(g, alts)
    candidates = sorted({a for g in empty for a in lookup.get(g, [])})
    if candidates:                          # one batched round instead of one request per alias
        alt = _fetch_block(candidates, samples, verbose=verbose)
        for g in empty:
            for alias in lookup.get(g, []):
                if alias in alt and alt[alias].notna().any():
                    expr[g] = alt[alias].values
                    used[g] = alias
                    break
    missing = [g for g in genes if expr[g].isna().all()]
    expr = expr.drop(columns=missing)
    if cached is not None:                  # keep genes fetched by earlier runs
        expr = pd.concat([cached, expr], axis=1)
    if verbose:
        print(f"\n  aliases used: {len(used)} | still missing ({len(missing)}): {missing}")
    return expr, used, missing


# ---------------------------------------------------------------------------
# Graph
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# External orthogonal labels: PCAWG (Sieverling et al. 2020) on TCGA donors
# ---------------------------------------------------------------------------
def build_external():
    s = pd.read_excel(download(C.SIEVERLING_SD1, C.RAW / "sieverling_SD1.xlsx"))
    sheet = pd.read_csv(download(C.PCAWG_SAMPLE_SHEET, C.RAW / "pcawg_sample_sheet.tsv"), sep="\t")
    u2b = pd.read_csv(download(C.PCAWG_UUID2BARCODE, C.RAW / "pcawg_tcga_uuid2barcode.tsv"), sep="\t")
    barcode = dict(zip(u2b.iloc[:, 2].str.lower(), u2b.iloc[:, 3]))
    m = sheet[sheet.icgc_specimen_id.isin(s.icgc_specimen_id) & sheet.dcc_project_code.str.endswith("-US")]
    m = m.drop_duplicates("icgc_specimen_id").merge(s, on="icgc_specimen_id")
    project = m.dcc_project_code.str.replace("-US", "", regex=False)
    ext = pd.DataFrame({
        "patient": m.submitter_donor_id.str.lower().map(barcode),
        "sample": m.submitter_specimen_id.str.lower().map(barcode).str[:15],
        "cancer": project.replace({"COAD": "CRC", "READ": "CRC"}),   # Barthel et al. pool COAD/READ as CRC
        "icgc_specimen": m["icgc_specimen_id"],
        "histology": m["histology_abbreviation"], "alt_probability": m["ALT_probability"],
        "y_alt_wgs": (m["ALT_probability"] > 0.5).astype(int),
        "tmm_mut": m["TMM_associated_mut_summary"],               # TERT_mod / ATRX_DAXX_trunc / Other
        "y_tert_mod": (m["TMM_associated_mut_summary"] == "TERT_mod").astype(int),
    }).dropna(subset=["patient"]).drop_duplicates("patient")
    return ext


# ---------------------------------------------------------------------------
def main():
    t0 = time.time()
    print("[1/6] Barthel et al. labels")
    lab = build_labels()
    toil = set(xena.dataset_samples(C.XENA_TOIL, C.TOIL_EXPR, None))
    lab = lab[lab["sample"].isin(toil)].reset_index(drop=True)
    print(f"  tumors with labels and Toil RNA-seq: {len(lab)} ({lab.cancer.nunique()} cancer types)")

    print("[2/6] external PCAWG labels (Sieverling et al. 2020) on TCGA donors")
    ext = build_external()
    samples = lab["sample"].tolist()
    if ext is not None:
        ext = ext[ext["sample"].isin(toil)].copy()
        ext["in_barthel"] = ext["patient"].isin(lab["patient"])
        ext.to_csv(C.PROC / "external_pcawg.csv", index=False)
        samples += sorted(set(ext["sample"]) - set(samples))
        print(f"  TCGA donors with WGS-based TMM calls and RNA: {len(ext)} (ALT-like: {ext.y_alt_wgs.sum()}, "
              f"TERT-altered: {ext.y_tert_mod.sum()}); in Barthel cohort: {ext.in_barthel.sum()}")

    print("[3/6] glioma subtypes")
    lab["subtype"] = glioma_subtypes(lab["sample"].tolist()).values

    print("[4/6] panel + signature expression")
    sig = barthel_signature_genes()
    genes = list(dict.fromkeys(C.PANEL_GENES + sig))
    expr, used, missing = fetch_expression(genes, samples, cache=C.PROC / "expr_panel.csv.gz")
    if not used and (C.PROC / "gene_sets.json").exists():        # cached run: keep the recorded aliases
        used = json.load(open(C.PROC / "gene_sets.json"))["aliases_used"]
    assert not set(C.PANEL_GENES) & set(missing), f"panel genes missing: {set(C.PANEL_GENES) & set(missing)}"
    assert np.nanmin(expr.values) > -10.5 and np.nanmax(expr.values) < 25, "unexpected expression scale"
    expr.to_csv(C.PROC / "expr_panel.csv.gz")
    json.dump({"signature": [g for g in sig if g in expr], "signature_missing": [g for g in sig if g not in expr],
               "aliases_used": used}, open(C.PROC / "gene_sets.json", "w"), indent=1)

    print("[5/6] random gene pool for the null distribution")
    rng = np.random.default_rng(C.SEED)
    all_symbols = xena.dataset_field(C.XENA_PANCAN, C.PANCAN_GENE_LIST_DS)
    pool = [g for g in all_symbols if g not in set(genes) and not g.startswith(("LOC", "?"))]
    pool = sorted(rng.choice(pool, size=int(C.RANDOM_POOL_SIZE * 1.1), replace=False).tolist())
    rexpr, _, _ = fetch_expression(pool, lab["sample"].tolist(), cache=C.PROC / "expr_random.csv.gz")
    rexpr = rexpr.loc[:, rexpr.notna().mean() > 0.99]
    rexpr = rexpr.loc[:, rexpr.std() > 0.1].iloc[:, :C.RANDOM_POOL_SIZE]
    rexpr.to_csv(C.PROC / "expr_random.csv.gz")
    print(f"  random pool: {rexpr.shape[1]} genes")

    print("[6/6] clinical endpoints (TCGA Pan-Cancer Clinical Data Resource)")
    surv = survival_table(lab["sample"].tolist())
    print(f"  survival: OS for {surv['OS.time'].notna().sum()} tumors, PFI for {surv['PFI.time'].notna().sum()}")

    lab.to_csv(C.PROC / "labels.csv", index=False)
    print(f"done in {(time.time() - t0) / 60:.1f} min")




# ---------------------------------------------------------------------------
# Published baselines (EXTEND) and cell-line assay validation
# ---------------------------------------------------------------------------
def extend_resources(samples):
    """Published EXTEND telomerase scores per TCGA sample and the 13-gene signature.

    Note: the signature contains TERT and TERC, so the score cannot be compared fairly with a
    label defined by TERT expression; we therefore also evaluate the signature without them.
    """
    sig = pd.read_excel(download(C.EXTEND_SIGNATURE, C.RAW / "extend_signature.xlsx"), header=2)
    genes = sig["Gene"].dropna().astype(str).str.strip().tolist()[:13]
    sc = pd.read_excel(download(C.EXTEND_TCGA_SCORES, C.RAW / "extend_tcga_scores.xlsx"), header=3)
    sc = sc.dropna(subset=["SampleID"])
    sc["sample"] = sc["SampleID"].str.replace(".", "-", regex=False).str[:15]
    sc = sc.drop_duplicates("sample").set_index("sample")["EXTEND Scores"]
    out = pd.Series(sc.reindex(samples).values, index=samples, name="extend_score")
    out.to_csv(C.PROC / "extend_scores.csv")
    print(f"  EXTEND: {len(genes)}-gene signature {genes}; scores for {out.notna().sum()}/{len(samples)} tumors")
    return genes


def ccle_validation(genes):
    """Cell lines with measured telomerase activity (Noureen et al. 2021 source data) + CCLE expression."""
    src = download(C.EXTEND_SOURCE_FIG1, C.RAW / "extend_source_fig1.xlsx")
    s = pd.read_excel(src, sheet_name="Source Data Fig.1", header=None)
    rows = []
    b = s.iloc[2:, 6:11].copy()
    b.columns = ["SampleID", "cellline", "activity", "TERT", "extend_score"]
    b = b[b.SampleID.notna() & (b.SampleID != "SampleID")]
    panel_name = None
    for _, r in b.iterrows():
        txt = str(r.SampleID)
        if txt.startswith("Source Data"):
            panel_name = "lung" if "Lung" in txt else "bladder"
            continue
        if panel_name is None:
            panel_name = "bladder"
        rows.append({"sample": txt, "panel": panel_name, "activity": r.activity,
                     "TERT_paper": r.TERT, "extend_score": r.extend_score})
    assay = pd.DataFrame(rows).dropna(subset=["activity"])
    available = set(xena.dataset_samples(C.XENA_PUBLIC, C.CCLE_EXPR, None))
    assay = assay[assay["sample"].isin(available)]
    expr = _fetch_block(genes, assay["sample"].tolist(), verbose=False, hub=C.XENA_PUBLIC, dataset=C.CCLE_EXPR)
    expr = np.log2(expr.astype(float) + 1e-3)     # CCLE hub stores linear RPKM; TCGA hub stores log2 values
    expr.to_csv(C.PROC / "ccle_expr.csv.gz")
    assay.to_csv(C.PROC / "ccle_assay.csv", index=False)
    print(f"  CCLE: {len(assay)} cell lines with measured telomerase activity "
          f"({assay.panel.value_counts().to_dict()}), {expr.shape[1]} genes fetched")
    return assay


def main_extra():
    """Stage 1b: resources for the extended benchmark (published baselines, cell lines)."""
    t0 = time.time()
    lab = pd.read_csv(C.PROC / "labels.csv")
    samples = lab["sample"].tolist()
    panel_expr = pd.read_csv(C.PROC / "expr_panel.csv.gz", index_col=0)
    print("[1/2] EXTEND published baseline")
    extend_genes = extend_resources(samples)
    print("[2/2] cell lines with telomerase assays (panel + EXTEND genes, then all protein-coding genes)")
    assay = ccle_validation(sorted(set(C.PANEL_GENES) | set(extend_genes)))
    ccle_transcriptome(assay["sample"].tolist())
    sets = json.load(open(C.PROC / "gene_sets.json"))
    sets["extend_signature"] = extend_genes
    sets.pop("expanded_network", None)
    json.dump(sets, open(C.PROC / "gene_sets.json", "w"), indent=1)
    print(f"done in {(time.time() - t0) / 60:.1f} min")


# ---------------------------------------------------------------------------
# Clinical endpoints and whole-transcriptome matrices
# ---------------------------------------------------------------------------
SURVIVAL_FIELDS = ["OS", "OS.time", "PFI", "PFI.time", "DSS", "DSS.time",
                   "age_at_initial_pathologic_diagnosis", "gender", "histological_grade"]


def survival_table(samples):
    """Curated endpoints of the TCGA Pan-Cancer Clinical Data Resource (Liu et al. 2018)."""
    vals = xena.dataset_fetch(C.XENA_PANCAN, C.PANCAN_SURVIVAL, samples, SURVIVAL_FIELDS)
    codes = {c["name"]: (c["code"].split("\t") if c.get("code") else None)
             for c in xena.field_codes(C.XENA_PANCAN, C.PANCAN_SURVIVAL, SURVIVAL_FIELDS)}
    out = pd.DataFrame(index=samples)
    for f, col in zip(SURVIVAL_FIELDS, vals):
        cd = codes.get(f)
        out[f] = [cd[int(v)] if cd and isinstance(v, (int, float)) and v == v else
                  (np.nan if v in ("NaN", None) else v) for v in col]
    for f in ["OS", "OS.time", "PFI", "PFI.time", "DSS", "DSS.time", "age_at_initial_pathologic_diagnosis"]:
        out[f] = pd.to_numeric(out[f], errors="coerce")
    out.index.name = "sample"
    out.to_csv(C.PROC / "survival.csv")
    return out


def protein_coding_map():
    """Versioned-or-not Ensembl gene id -> HGNC symbol, restricted to protein-coding genes."""
    h = pd.read_csv(download(C.HGNC_COMPLETE, C.RAW / "hgnc_complete_set.txt"), sep="\t",
                    usecols=["symbol", "locus_group", "ensembl_gene_id"], low_memory=False)
    h = h[(h.locus_group == "protein-coding gene") & h.ensembl_gene_id.notna()]
    return dict(zip(h.ensembl_gene_id, h.symbol))


def _matrix_to_protein_coding(path, keep_samples, ens2sym, compression):
    """Stream a genes x samples matrix, keep protein-coding rows and the requested columns."""
    header = pd.read_csv(path, sep="\t", nrows=0, compression=compression).columns.tolist()
    cols = [header[0]] + [c for c in header[1:] if c in set(keep_samples)]
    blocks = []
    for chunk in pd.read_csv(path, sep="\t", usecols=cols, chunksize=4000, compression=compression,
                             low_memory=False):
        ids = chunk.iloc[:, 0].astype(str).str.split(".").str[0]
        sym = ids.map(ens2sym)
        chunk = chunk.loc[sym.notna()].copy()
        chunk.index = sym[sym.notna()].values
        blocks.append(chunk.iloc[:, 1:].astype(np.float32))
    m = pd.concat(blocks)
    m = m.groupby(level=0).mean()                      # a few symbols map to several Ensembl ids
    return m.T                                         # samples x genes


def save_matrix(df, name):
    np.save(C.PROC / f"{name}.npy", df.values.astype(np.float32))
    (C.PROC / f"{name}_genes.txt").write_text("\n".join(df.columns))
    (C.PROC / f"{name}_samples.txt").write_text("\n".join(df.index))
    print(f"  saved {name}: {df.shape[0]} samples x {df.shape[1]} protein-coding genes")


def load_matrix(name):
    X = np.load(C.PROC / f"{name}.npy", mmap_mode="r")
    genes = (C.PROC / f"{name}_genes.txt").read_text().split("\n")
    samples = (C.PROC / f"{name}_samples.txt").read_text().split("\n")
    return pd.DataFrame(np.asarray(X), index=samples, columns=genes)


def main_transcriptome(pcawg=True):
    """Stage 1c: whole protein-coding transcriptome for TCGA tumors (Toil) and PCAWG re-quantification."""
    t0 = time.time()
    ens2sym = protein_coding_map()
    lab = pd.read_csv(C.PROC / "labels.csv")
    ext = pd.read_csv(C.PROC / "external_pcawg.csv") if (C.PROC / "external_pcawg.csv").exists() else None
    samples = set(lab["sample"]) | (set(ext["sample"]) if ext is not None else set())

    print("[1/2] TCGA Toil TPM matrix (log2(TPM+0.001))")
    toil = download(C.TOIL_MATRIX, C.RAW / "tcga_RSEM_gene_tpm.gz", timeout=3600)
    save_matrix(_matrix_to_protein_coding(toil, samples, ens2sym, "gzip"), "transcriptome")

    if pcawg and ext is not None:
        print("[2/2] PCAWG re-quantification of the same tumors (TopHat2/STAR, FPKM-UQ, log2)")
        url = f"{C.XENA_PCAWG}/download/{C.PCAWG_EXPR}"
        pc = download(url, C.RAW / "pcawg_fpkm_uq.log.tsv", timeout=3600)
        m = _matrix_to_protein_coding(pc, set(ext["icgc_specimen"]), ens2sym, None)
        spec2sample = dict(zip(ext["icgc_specimen"], ext["sample"]))
        m.index = [spec2sample[i] for i in m.index]
        m = m[~m.index.duplicated()]
        save_matrix(m, "pcawg_transcriptome")
    print(f"done in {(time.time() - t0) / 60:.1f} min")


def ccle_transcriptome(cell_lines):
    """All protein-coding genes for the assay cell lines (only 24 lines, so the API is fast enough)."""
    h = pd.read_csv(download(C.HGNC_COMPLETE, C.RAW / "hgnc_complete_set.txt"), sep="\t",
                    usecols=["symbol", "locus_group"], low_memory=False)
    genes = sorted(h.loc[h.locus_group == "protein-coding gene", "symbol"].dropna().unique())
    expr = _fetch_block(genes, list(cell_lines), g_chunk=50, verbose=False, hub=C.XENA_PUBLIC, dataset=C.CCLE_EXPR)
    expr = expr.loc[:, expr.notna().any()]
    expr = np.log2(expr.astype(float) + 1e-3)
    expr.to_csv(C.PROC / "ccle_transcriptome.csv.gz")
    print(f"  CCLE transcriptome: {expr.shape[0]} cell lines x {expr.shape[1]} protein-coding genes")


if __name__ == "__main__":
    import sys
    if "--extra" in sys.argv:
        main_extra()
    elif "--transcriptome-tcga" in sys.argv:
        main_transcriptome(pcawg=False)
    elif "--transcriptome" in sys.argv:
        main_transcriptome()
    else:
        main()
