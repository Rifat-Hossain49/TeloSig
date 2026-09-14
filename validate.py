"""Stage 3: validation beyond cross-validation.

  external   TCGA donors re-profiled by PCAWG whole-genome sequencing, held out of training; labels are
             independent of the training labels (ALT by telomere-sequence features; TERT DNA alterations)
  platform   the same tumors quantified by an independent RNA-seq pipeline (PCAWG TopHat2/STAR FPKM-UQ vs
             TCGA Toil RSEM TPM): agreement of signature scores and discrimination on both quantifications
  celllines  cancer cell lines with measured telomerase enzymatic activity (TRAP / direct assay)
  survival   Cox models of signature scores within cancer types, adjusted for age, sex (and glioma subtype)
  enrichment g:Profiler over-representation of the stable signature genes (whole transcriptome background)

Signature scores used for survival are cross-validated (out-of-fold, averaged over repeats), so no tumor is
scored by a model that saw it. Usage: python validate.py [--steps external,platform,celllines,survival,enrichment]
"""
import argparse
import json

import numpy as np
import pandas as pd
import requests

import config as C
import evaluate as E
from models import SkModel, transfer
from run_experiments import Select, excluded, load, load_transcriptome, task_frame
from signature import score_locked


def _locked(task, heldout=False):
    p = C.FROZEN / f"{task}_signature_locked{'_heldout' if heldout else ''}.json"
    return json.load(open(p)) if p.exists() else None


def _auc_row(name, target, score, cancer, **extra):
    cancers = E.eligible_cancers(target, cancer, "clf")
    m = E.metrics(target, score, cancer, cancers, "clf")
    lo, hi = _boot_auc(target, score)
    return {"model": name, "n": len(target), "n_pos": int(np.sum(target)), "AUROC": m["AUROC"],
            "AUROC_lo": lo, "AUROC_hi": hi, "within_AUROC": m["within_AUROC"], **extra}


def _boot_auc(y, s, n=2000, seed=C.SEED):
    rng = np.random.default_rng(seed)
    y, s = np.asarray(y), np.asarray(s)
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    vals = []
    for _ in range(n):
        i = np.r_[rng.choice(pos, len(pos)), rng.choice(neg, len(neg))]
        vals.append(E.auc(y[i], s[i]))
    return tuple(np.nanpercentile(vals, [2.5, 97.5]))


# ---------------------------------------------------------------------------
def external(lab, expr, ext):
    """Held-out PCAWG donors scored against whole-genome labels."""
    rows = []
    T_all, _ = load_transcriptome(sorted(set(lab["sample"]) | set(ext["sample"])))
    for task, target_col, subset in [("alt", "y_alt_wgs", C.ALT_COHORT), ("alt_pan", "y_alt_wgs", None),
                                     ("tel", "y_tert_mod", None)]:
        d, X, cov, y = task_frame(task, lab, expr, exclude_patients=set(ext.patient))
        e = ext[ext["sample"].isin(expr.index) & ext["sample"].isin(T_all.index)].copy()
        if subset:
            e = e[e.cancer.isin(subset)]
        target = e[target_col].values
        if len(e) < 20 or target.sum() < 5:
            continue
        excl = excluded(task)
        genes = [g for g in C.PANEL_GENES if g in X.columns and g not in excl]
        Xe = expr.loc[e["sample"]].reset_index(drop=True)
        cove = pd.DataFrame({"cancer": e.cancer.values, "subtype": e.cancer.values})
        label = {"y_alt_wgs": "ALT by WGS telomere-sequence features", "y_tert_mod": "TERT DNA alteration (WGS)"}[target_col]
        for name, mk in [("Cancer type only", lambda: SkModel("cov_only", "clf", use_genes=False)),
                         ("LR (panel)", lambda: SkModel("lr", "clf")), ("GBM (panel)", lambda: SkModel("gbm", "clf"))]:
            s = Select(mk(), genes, ["cancer"]).fit(X, cov, y).predict(Xe, cove)
            rows.append(_auc_row(name, target, s, e.cancer.values, task=task, label_source=label))
        locked = _locked(task, heldout=True)
        if locked:
            s = score_locked(locked, T_all.loc[e["sample"]])
            rows.append(_auc_row(f"Signature (K={locked['k']}, locked)", target, s, e.cancer.values, task=task, label_source=label))
        if task in ("alt", "alt_pan"):       # how well does the training label itself agree with WGS?
            rows.append(_auc_row("Barthel genotype label", target,
                                 lab.set_index("patient").reindex(e.patient)["y_alt"].fillna(0).values,
                                 e.cancer.values, task=task, label_source=label))
        print(pd.DataFrame(rows)[lambda r: r.task == task][["model", "n", "n_pos", "AUROC", "within_AUROC"]]
              .round(3).to_string(index=False), flush=True)
    out = pd.DataFrame(rows)
    out.to_csv(C.RESULTS / "validation_external.csv", index=False)
    return out


def platform(lab, expr, ext):
    """Does the signature give the same answer when the same tumors are quantified by another pipeline?"""
    if not (C.PROC / "pcawg_transcriptome.npy").exists():
        print("    PCAWG re-quantification not built (data_build.py --transcriptome); skipped")
        return None
    from data_build import load_matrix
    P = load_matrix("pcawg_transcriptome")
    T, _ = load_transcriptome(P.index.tolist())
    rows, per_sample = [], []
    for task in ["alt", "tel"]:
        locked = _locked(task)
        if not locked:
            continue
        shared = [g for g in locked["genes"] if g in P.columns]
        if len(shared) < len(locked["genes"]):
            print(f"    {task}: {len(locked['genes']) - len(shared)} signature genes absent from PCAWG matrix")
        sub = dict(locked, genes=shared, weights=[w for g, w in zip(locked["genes"], locked["weights"]) if g in shared],
                   mean=[m for g, m in zip(locked["genes"], locked["mean"]) if g in shared],
                   sd=[v for g, v in zip(locked["genes"], locked["sd"]) if g in shared])
        d, _, cov, y = task_frame(task, lab, expr)
        dd = d.set_index("sample")
        common = [s for s in P.index if s in dd.index]
        s_toil = score_locked(sub, T.loc[common], restandardise=True)
        s_pcawg = score_locked(sub, P.loc[common], restandardise=True)
        yy = dd.loc[common, "y_alt" if task.startswith("alt") else "y_tel"].values
        canc = dd.loc[common, "cancer"].values
        rho = pd.Series(s_toil).corr(pd.Series(s_pcawg), method="spearman")
        per_sample.append(pd.DataFrame({"task": task, "sample": common, "cancer": canc, "label": yy,
                                        "score_toil": s_toil, "score_pcawg": s_pcawg}))
        for q, s in [("TCGA Toil (RSEM TPM)", s_toil), ("PCAWG (TopHat2/STAR FPKM-UQ)", s_pcawg)]:
            rows.append(_auc_row(f"Signature, {q}", yy, s, canc, task=task, score_spearman_between_pipelines=rho))
        print(f"    {task}: n={len(common)}, Spearman between pipelines = {rho:.3f}; "
              f"AUROC Toil {rows[-2]['AUROC']:.3f} vs PCAWG {rows[-1]['AUROC']:.3f}", flush=True)
    out = pd.DataFrame(rows)
    out.to_csv(C.RESULTS / "validation_platform.csv", index=False)
    if per_sample:
        pd.concat(per_sample).to_csv(C.RESULTS / "validation_platform_scores.csv", index=False)
    return out


def celllines(lab, expr, sets):
    """Measured telomerase activity in cancer cell lines vs published EXTEND, TERT alone and our models."""
    ca, ct = C.PROC / "ccle_assay.csv", C.PROC / "ccle_transcriptome.csv.gz"
    if not (ca.exists() and ct.exists()):
        print("    cell-line resources missing; skipped")
        return None
    assay = pd.read_csv(ca)
    cexpr = pd.read_csv(ct, index_col=0)
    assay = assay[assay["sample"].isin(cexpr.index)]
    d, X, cov, y = task_frame("tel", lab, expr)
    shared = [g for g in C.PANEL_GENES if g in X.columns and g in cexpr.columns]
    trained = {}
    for vname, genes in {"panel without TERT": [g for g in shared if g != "TERT"], "panel with TERT": shared}.items():
        trained[f"LR ({vname})"] = (Select(SkModel("lr", "clf"), genes, []).fit(X, cov, y), genes)
    locked = _locked("tel")
    rng = np.random.default_rng(C.SEED)

    def spearman_ci(a, b, n_boot=2000):
        a, b = np.asarray(a, float), np.asarray(b, float)
        r = float(pd.Series(a).corr(pd.Series(b), method="spearman"))
        bs = [float(pd.Series(a[i]).corr(pd.Series(b[i]), method="spearman"))
              for i in (rng.integers(0, len(a), len(a)) for _ in range(n_boot))]
        return r, *np.nanpercentile(bs, [2.5, 97.5])

    rows = []
    for panel, grp in assay.groupby("panel"):
        act = grp["activity"].astype(float).values
        Xc = cexpr.loc[grp["sample"]]
        preds = {}
        for name, (m, genes) in trained.items():
            transfer(m, Xc[genes])
            preds[name] = m.predict(Xc[genes], pd.DataFrame(index=range(len(grp))))
        if locked:
            present = [g for g in locked["genes"] if g in Xc.columns]
            sub = dict(locked, genes=present,
                       weights=[w for g, w in zip(locked["genes"], locked["weights"]) if g in present])
            preds[f"Signature (K={locked['k']}, no TERT)"] = score_locked(sub, Xc, restandardise=True)
        preds["EXTEND score (published; uses TERT/TERC)"] = grp["extend_score"].astype(float).values
        preds["TERT expression alone"] = grp["TERT_paper"].astype(float).values
        for name, sc in preds.items():
            r, lo, hi = spearman_ci(sc, act)
            rows.append({"panel": panel, "n": len(grp), "model": name, "spearman": r, "lo": lo, "hi": hi})
    out = pd.DataFrame(rows)
    out.to_csv(C.RESULTS / "validation_celllines.csv", index=False)
    print(out.pivot(index="model", columns="panel", values="spearman").round(3).to_string(), flush=True)
    return out


def survival(lab):
    """Cox proportional hazards of cross-validated signature scores, within cancer types."""
    from statsmodels.duration.hazard_regression import PHReg
    sp = C.PROC / "survival.csv"
    if not sp.exists():
        print("    survival table missing; skipped")
        return None
    surv = pd.read_csv(sp).set_index("sample")
    rows = []
    for task, groups in [("alt", [("LGG", ["LGG"]), ("GBM", ["GBM"]), ("SARC", ["SARC"])]), ("tel", [("pan-cancer", None)])]:
        f = C.RESULTS / f"{task}_sig_oof.npz"
        z = C.RESULTS / f"{task}_oof.npz"
        if not (f.exists() and z.exists()):
            continue
        meta = np.load(z, allow_pickle=True)
        score = pd.Series(np.nanmean(np.load(f)["oof"], 0), index=meta["sample"])
        df = lab.set_index("sample").loc[score.index, ["cancer", "subtype", "age", "sex"]].join(surv, how="left")
        df["score_sd"] = (score - score.mean()) / score.std()
        for gname, cancers in groups:
            for endpoint in ["OS", "PFI"]:
                sub = df if cancers is None else df[df.cancer.isin(cancers)]
                sub = sub.dropna(subset=[endpoint, f"{endpoint}.time", "age"])
                sub = sub[sub[f"{endpoint}.time"] > 0]
                if sub[endpoint].sum() < 15:
                    continue
                for adj, covs in [("unadjusted", []), ("age + sex", ["age", "male"]),
                                  ("age + sex + glioma subtype", ["age", "male", "subtype"])]:
                    if "subtype" in covs and gname not in ("LGG", "GBM"):
                        continue
                    design = pd.DataFrame({"score_sd": sub.score_sd.values, "age": sub.age.values,
                                           "male": (sub.sex.astype(str).str.lower() == "male").astype(float).values})
                    if "subtype" in covs:
                        st = sub.subtype.astype(str)
                        counts = st.value_counts()
                        st = st.where(st.map(counts) >= 15, "other")       # rare subtypes make Cox unstable
                        if st.nunique() < 2:
                            continue
                        dummies = pd.get_dummies(st, drop_first=True).astype(float)
                        design = pd.concat([design, dummies.reset_index(drop=True)], axis=1)
                    cols = ["score_sd"] + [c for c in covs if c in ("age", "male")] + \
                           ([c for c in design.columns if c not in ("score_sd", "age", "male")] if "subtype" in covs else [])
                    strata = sub.cancer.values if cancers is None else None
                    try:
                        res = PHReg(sub[f"{endpoint}.time"].values, design[cols].values, status=sub[endpoint].values,
                                    strata=strata).fit(disp=False)
                        b, se = res.params[0], res.bse[0]
                        if not (np.isfinite(b) and np.isfinite(se)):
                            raise ValueError("non-finite estimate")
                    except Exception as ex:
                        print(f"    Cox not estimable for {task}/{gname}/{endpoint}/{adj}: {ex}")
                        continue
                    rows.append({"task": task, "group": gname, "endpoint": endpoint, "adjustment": adj,
                                 "n": len(sub), "events": int(sub[endpoint].sum()), "HR_per_SD": np.exp(b),
                                 "HR_lo": np.exp(b - 1.96 * se), "HR_hi": np.exp(b + 1.96 * se), "p": res.pvalues[0]})
    out = pd.DataFrame(rows)
    out.to_csv(C.RESULTS / "validation_survival.csv", index=False)
    if len(out):
        print(out[["task", "group", "endpoint", "adjustment", "n", "events", "HR_per_SD", "HR_lo", "HR_hi", "p"]]
              .round(3).to_string(index=False), flush=True)
    return out


def enrichment():
    """Over-representation (g:Profiler) of stable signature genes against the candidate-gene background."""
    rows = []
    for task in ["alt", "tel"]:
        f = C.RESULTS / f"{task}_signature.json"
        if not f.exists():
            continue
        info = json.load(open(f))
        genes = info["stable_genes"] or info["locked_genes"]
        bg = None
        if (C.PROC / "transcriptome_genes.txt").exists():
            bg = (C.PROC / "transcriptome_genes.txt").read_text().split("\n")
        body = {"organism": "hsapiens", "query": genes, "sources": ["GO:BP", "REAC", "KEGG", "HP"],
                "user_threshold": 0.05, "significance_threshold_method": "g_SCS", "no_evidences": True}
        if bg:
            body.update(domain_scope="custom", background=bg)
        try:
            r = requests.post("https://biit.cs.ut.ee/gprofiler/api/gost/profile/", json=body, timeout=180)
            r.raise_for_status()
            res = r.json().get("result", [])
        except Exception as ex:
            print(f"    g:Profiler request failed for {task}: {ex}")
            continue
        for t in res:
            rows.append({"task": task, "source": t["source"], "term": t["name"], "p_adj": t["p_value"],
                         "intersection": t["intersection_size"], "term_size": t["term_size"]})
        print(f"    {task}: {len(genes)} genes -> {len(res)} enriched terms", flush=True)
    out = pd.DataFrame(rows)
    out.to_csv(C.RESULTS / "validation_enrichment.csv", index=False)
    return out


def chromosome_arms():
    """Are signature genes concentrated on chromosome arms (e.g. glioma 1p/19q), i.e. copy-number driven?"""
    from scipy.stats import hypergeom
    from statsmodels.stats.multitest import multipletests
    hg = C.RAW / "hgnc_complete_set.txt"
    if not hg.exists():
        from data_build import download
        download(C.HGNC_COMPLETE, hg)
    h = pd.read_csv(hg, sep="\t", usecols=["symbol", "location"], low_memory=False).dropna()
    parts = h.location.str.extract(r"^(\d+|X|Y)([pq])")
    h["arm"] = parts[0] + parts[1]
    arm = dict(zip(h.symbol, h.arm))
    genes_file = C.PROC / "transcriptome_genes.txt"
    bg_genes = genes_file.read_text().split("\n") if genes_file.exists() else list(arm)
    bg = pd.Series([arm.get(g) for g in bg_genes]).dropna()
    rows = []
    for task in ["alt", "alt_pan", "tel"]:
        for kind, f in [("locked", C.FROZEN / f"{task}_signature_locked.json"),
                        ("stable", C.RESULTS / f"{task}_signature.json")]:
            if not f.exists():
                continue
            info = json.load(open(f))
            genes = info["genes"] if kind == "locked" else info["stable_genes"]
            arms = pd.Series([arm.get(g) for g in genes]).dropna()
            if arms.empty:
                continue
            for a, k in arms.value_counts().items():
                K = int((bg == a).sum())
                rows.append({"task": task, "gene_set": kind, "arm": a, "genes_on_arm": int(k), "set_size": len(arms),
                             "background_on_arm": K, "fold_enrichment": (k / len(arms)) / (K / len(bg)),
                             "p": hypergeom.sf(k - 1, len(bg), K, len(arms)),
                             "genes": ",".join(g for g in genes if arm.get(g) == a)})
    out = pd.DataFrame(rows)
    if len(out):
        out["p_adj"] = out.groupby(["task", "gene_set"]).p.transform(lambda x: multipletests(x, method="fdr_bh")[1])
        out = out.sort_values(["task", "gene_set", "p"])
        show = out[out.p_adj < 0.1]
        print(show[["task", "gene_set", "arm", "genes_on_arm", "set_size", "fold_enrichment", "p_adj", "genes"]]
              .to_string(index=False) if len(show) else "    no arm enriched at FDR < 0.1", flush=True)
    out.to_csv(C.RESULTS / "validation_chromosome_arms.csv", index=False)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", default="external,platform,celllines,survival,enrichment,arms")
    a = ap.parse_args()
    lab, expr, sets, ext = load()
    for step in a.steps.split(","):
        print(f"\n=== validation: {step}", flush=True)
        if step == "external" and ext is not None:
            external(lab, expr, ext)
        elif step == "platform" and ext is not None:
            platform(lab, expr, ext)
        elif step == "celllines":
            celllines(lab, expr, sets)
        elif step == "survival":
            survival(lab)
        elif step == "enrichment":
            enrichment()
        elif step == "arms":
            chromosome_arms()


if __name__ == "__main__":
    main()
