"""Stage 3: validation beyond cross-validation.

  external   TCGA donors re-profiled by PCAWG whole-genome sequencing, held out of training; labels are
             independent of the training labels (ALT by telomere-sequence features; TERT DNA alterations)
  platform   the same tumors quantified by an independent RNA-seq pipeline (PCAWG TopHat2/STAR FPKM-UQ vs
             TCGA Toil RSEM TPM): agreement of signature scores and discrimination on both quantifications
  celllines  legacy cancer-cell-line panels with measured telomerase enzymatic activity
  wu2025     976-line panel with qTRAP telomerase activity, C-circles and fixed RNA-seq
  survival   Cox models of signature scores within cancer types, adjusted for age, sex (and glioma subtype)
  enrichment g:Profiler over-representation of the stable signature genes (whole transcriptome background)

Signature scores used for survival are cross-validated (out-of-fold, averaged over repeats), so no tumor is
scored by a model that saw it. Usage: python validate.py [--steps external,platform,celllines,wu2025,survival]
"""
import argparse
import json

import numpy as np
import pandas as pd
import requests

import config as C
import evaluate as E
from models import SkModel
from run_experiments import Select, excluded, load, load_transcriptome, task_frame
from signature import score_locked


def _locked(task, heldout=False):
    p = C.FROZEN / f"{task}_signature_locked{'_heldout' if heldout else ''}.json"
    return json.load(open(p)) if p.exists() else None


def _auc_row(name, target, score, cancer, threshold=None, **extra):
    cancers = E.eligible_cancers(target, cancer, "clf")
    m = E.metrics(target, score, cancer, cancers, "clf")
    lo, hi = _boot_auc(target, score)
    row = {"model": name, "n": len(target), "n_pos": int(np.sum(target)), "AUROC": m["AUROC"],
           "AUROC_lo": lo, "AUROC_hi": hi, "within_AUROC": m["within_AUROC"], **extra}
    if np.nanmin(score) >= 0 and np.nanmax(score) <= 1:
        from sklearn.metrics import brier_score_loss
        row["Brier"] = brier_score_loss(target, score)
    if threshold is not None:
        called, target = np.asarray(score) >= threshold, np.asarray(target).astype(bool)
        row.update(threshold=threshold,
                   sensitivity=float(called[target].mean()), specificity=float((~called[~target]).mean()),
                   PPV=float(target[called].mean()) if called.any() else np.nan,
                   NPV=float((~target[~called]).mean()) if (~called).any() else np.nan)
    return row


def _subset_locked(locked, genes):
    keep = [i for i, g in enumerate(locked["genes"]) if g in set(genes)]
    return dict(locked, genes=[locked["genes"][i] for i in keep],
                weights=[locked["weights"][i] for i in keep], mean=[locked["mean"][i] for i in keep],
                sd=[locked["sd"][i] for i in keep])


def _boot_auc(y, s, n=2000, seed=C.SEED):
    rng = np.random.default_rng(seed)
    y, s = np.asarray(y), np.asarray(s)
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    vals = []
    for _ in range(n):
        i = np.r_[rng.choice(pos, len(pos)), rng.choice(neg, len(neg))]
        vals.append(E.auc(y[i], s[i]))
    return tuple(np.nanpercentile(vals, [2.5, 97.5]))


def _boot_spearman(a, b, n=2000, seed=C.SEED):
    """Spearman estimate and percentile interval after dropping non-finite pairs."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    keep = np.isfinite(a) & np.isfinite(b)
    a, b = a[keep], b[keep]
    if len(a) < 3:
        return np.nan, np.nan, np.nan, len(a)
    estimate = float(pd.Series(a).corr(pd.Series(b), method="spearman"))
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(n):
        idx = rng.integers(0, len(a), len(a))
        values.append(float(pd.Series(a[idx]).corr(pd.Series(b[idx]), method="spearman")))
    lo, hi = np.nanpercentile(values, [2.5, 97.5])
    return estimate, float(lo), float(hi), len(a)


def _correlation_row(source, task, outcome, name, score, target, scope="all", **extra):
    estimate, lo, hi, n = _boot_spearman(score, target)
    return {"source": source, "task": task, "outcome": outcome, "model": name, "scope": scope,
            "metric": "Spearman", "estimate": estimate, "lo": lo, "hi": hi, "n": n, **extra}


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
        e = e.dropna(subset=[target_col])
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
            s = score_locked(locked, T_all.loc[e["sample"]], probability=True)
            rows.append(_auc_row(f"Signature (K={locked['k']}, locked)", target, s, e.cancer.values,
                                 threshold=locked.get("deployment", {}).get("threshold"),
                                 task=task, label_source=label, scope="all"))
            for cancer_name in sorted(e.cancer.unique()):
                q = e.cancer.values == cancer_name
                if min(target[q].sum(), (~target[q].astype(bool)).sum()) >= 5:
                    rows.append(_auc_row(f"Signature (K={locked['k']}, locked)", target[q], s[q],
                                         e.cancer.values[q],
                                         threshold=locked.get("deployment", {}).get("threshold"),
                                         task=task, label_source=label, scope=f"cancer:{cancer_name}"))
        if task in ("alt", "alt_pan"):       # how well does the training label itself agree with WGS?
            rows.append(_auc_row("Barthel genotype label", target,
                                 lab.set_index("patient").reindex(e.patient)["y_alt"].fillna(0).values,
                                 e.cancer.values, task=task, label_source=label))
        print(pd.DataFrame(rows)[lambda r: r.task == task][["model", "n", "n_pos", "AUROC", "within_AUROC"]]
              .round(3).to_string(index=False), flush=True)
    out = pd.DataFrame(rows)
    out.to_csv(C.RESULTS / "validation_external.csv", index=False)
    _alt_threshold_sensitivity(T_all, ext)
    return out


def _alt_threshold_sensitivity(T_all, ext):
    """Sensitivity of the locked phenotype transfer result to the WGS ALT-probability cut-off."""
    locked = _locked("alt_pan", heldout=True)
    if not locked:
        return None
    e = ext[ext["sample"].isin(T_all.index) & ext.alt_probability.notna()].copy()
    score = score_locked(locked, T_all.loc[e["sample"]], probability=True)
    rows = []
    for threshold in (0.25, 0.5, 0.75):
        target = (e.alt_probability.values > threshold).astype(int)
        for scope, keep in [("all", np.ones(len(e), bool))] + [
                (f"cancer:{c}", e.cancer.values == c) for c in sorted(e.cancer.unique())]:
            if keep.sum() >= 20 and min(target[keep].sum(), (1 - target[keep]).sum()) >= 5:
                rows.append(_auc_row("ALT signature locked", target[keep], score[keep], e.cancer.values[keep],
                                     threshold=locked.get("deployment", {}).get("threshold"),
                                     alt_probability_threshold=threshold, scope=scope))
    out = pd.DataFrame(rows)
    out.to_csv(C.RESULTS / "validation_alt_probability_thresholds.csv", index=False)
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
        # Use the signature for which these PCAWG donors were excluded before every training decision.
        locked = _locked(task, heldout=True)
        if not locked:
            continue
        shared = [g for g in locked["genes"] if g in P.columns]
        if len(shared) < len(locked["genes"]):
            print(f"    {task}: {len(locked['genes']) - len(shared)} signature genes absent from PCAWG matrix")
        sub = _subset_locked(locked, shared)
        d, _, cov, y = task_frame(task, lab, expr)
        dd = d.set_index("sample")
        common = [s for s in P.index if s in dd.index]
        s_toil = score_locked(sub, T.loc[common])
        s_pcawg = score_locked(sub, P.loc[common])
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

    rows = []
    for panel, grp in assay.groupby("panel"):
        act = grp["activity"].astype(float).values
        Xc = cexpr.loc[grp["sample"]]
        preds = {}
        for name, (m, genes) in trained.items():
            # The TCGA training scaler remains immutable.  Target-cohort standardisation would make the
            # model cohort-dependent and is therefore deliberately not performed.
            preds[name] = m.predict(Xc[genes], pd.DataFrame(index=range(len(grp))))
        if locked:
            present = [g for g in locked["genes"] if g in Xc.columns]
            sub = _subset_locked(locked, present)
            preds[f"Signature (K={locked['k']}, no TERT)"] = score_locked(sub, Xc)
        preds["EXTEND score (published; uses TERT/TERC)"] = grp["extend_score"].astype(float).values
        preds["TERT expression alone"] = grp["TERT_paper"].astype(float).values
        for name, sc in preds.items():
            r, lo, hi, _ = _boot_spearman(sc, act)
            rows.append({"panel": panel, "n": len(grp), "model": name, "spearman": r, "lo": lo, "hi": hi})
    out = pd.DataFrame(rows)
    out.to_csv(C.RESULTS / "validation_celllines.csv", index=False)
    print(out.pivot(index="model", columns="panel", values="spearman").round(3).to_string(), flush=True)
    return out


def wu2025_celllines():
    """Validate locked TCGA signatures in the independent 976-cell-line activity atlas.

    Primary ALT discrimination is restricted to unambiguous telomerase/ALT-labelled lines. Continuous C-circle
    and qTRAP measurements retain every line with a finite assay value. No model is refitted and the stored TCGA
    means, standard deviations, intercepts and operating thresholds remain immutable.
    """
    assay_path, expr_path = C.PROC / "wu2025_assay.csv", C.PROC / "wu2025_expr.csv.gz"
    if not (assay_path.exists() and expr_path.exists()):
        print("    Wu 2025 resources missing (data_build.py --wu2025); skipped")
        return None
    assay = pd.read_csv(assay_path)
    X = pd.read_csv(expr_path, index_col=0)
    assay["model_id"] = assay["model_id"].astype(str)
    X.index = X.index.astype(str)
    assay = assay[assay.model_id.isin(X.index)].drop_duplicates("model_id").copy()
    X = X.loc[assay.model_id]
    lineage_col = "Tissue_of_Origin" if "Tissue_of_Origin" in assay else "Cancer_Category"
    lineage = assay[lineage_col].fillna("unknown").astype(str).values

    predictions, metadata = {}, {}
    for task in ("alt", "alt_pan", "tel"):
        locked = _locked(task)
        if not locked:
            continue
        present = [g for g in locked["genes"] if g in X.columns]
        missing = [g for g in locked["genes"] if g not in X.columns]
        if not present:
            continue
        # A fixed external RNA release need not contain every locked feature.  Missing transcripts are assigned
        # their stored TCGA training means, hence zero standardised contribution; no target values or outcomes
        # enter this operation.  This is algebraically identical to the historical subset score, but makes the
        # deployment contract and coverage limitation explicit.
        predictions[task] = score_locked(locked, X, probability=True, missing="training_mean")
        metadata[task] = {"signature_genes_present": len(present),
                          "signature_genes_expected": len(locked["genes"]),
                          "signature_genes_imputed": len(missing),
                          "missing_gene_policy": "locked TCGA training mean",
                          "missing_genes": ";".join(missing),
                          "deployment_threshold": locked.get("deployment", {}).get("threshold")}

    rows = []
    tmm = assay["TMM"].fillna("").astype(str).str.upper().str.strip()
    is_alt = tmm.str.startswith("ALT")
    is_tel = tmm.str.startswith("TEL")
    unambiguous = (is_alt | is_tel).values
    if unambiguous.sum() >= 20 and is_alt[unambiguous].sum() >= 5:
        y_alt = is_alt[unambiguous].astype(int).values
        for task in ("alt", "alt_pan"):
            if task not in predictions:
                continue
            info = metadata[task]
            score = predictions[task][unambiguous]
            row = _auc_row(f"Locked {task} signature", y_alt, score, lineage[unambiguous],
                           threshold=info["deployment_threshold"], source="Wu 2025", task="alt",
                           outcome="unambiguous TMM category: ALT vs telomerase", scope="all",
                           analysis_role="primary", signature_genes_present=info["signature_genes_present"],
                           signature_genes_expected=info["signature_genes_expected"])
            row.update(metric="AUROC", estimate=row["AUROC"], lo=row["AUROC_lo"], hi=row["AUROC_hi"])
            rows.append(row)
            for group in sorted(set(lineage[unambiguous])):
                keep = unambiguous & (lineage == group)
                y_group = is_alt[keep].astype(int).values
                if keep.sum() >= 20 and min(y_group.sum(), len(y_group) - y_group.sum()) >= 5:
                    scoped = _auc_row(f"Locked {task} signature", y_group, predictions[task][keep], lineage[keep],
                                      threshold=info["deployment_threshold"], source="Wu 2025", task="alt",
                                      outcome="unambiguous TMM category: ALT vs telomerase",
                                      scope=f"lineage:{group}", analysis_role="exploratory",
                                      signature_genes_present=info["signature_genes_present"],
                                      signature_genes_expected=info["signature_genes_expected"])
                    scoped.update(metric="AUROC", estimate=scoped["AUROC"],
                                  lo=scoped["AUROC_lo"], hi=scoped["AUROC_hi"])
                    rows.append(scoped)

    c_circle_col = "C-Circle_%DOS16_Log2(n+1)"
    c_circle = (pd.to_numeric(assay[c_circle_col], errors="coerce").values if c_circle_col in assay else
                np.log2(pd.to_numeric(assay["C-Circle_%DOS16"], errors="coerce").values + 1))
    for task in ("alt", "alt_pan"):
        if task in predictions:
            rows.append(_correlation_row("Wu 2025", "alt", "C-circle %DOS16, log2(n+1)",
                                         f"Locked {task} signature", predictions[task], c_circle,
                                         analysis_role="secondary", **metadata[task]))

    activity_raw = pd.to_numeric(assay["Telomerase_Activity_%MCF7"], errors="coerce").values
    activity_log_col = "Telomerase_Activity_%MCF7_Log2(n+1)"
    activity_log = (pd.to_numeric(assay[activity_log_col], errors="coerce").values
                    if activity_log_col in assay else np.log2(activity_raw + 1))
    tel_predictions = {}
    if "tel" in predictions:
        tel_predictions["Locked telomerase signature"] = predictions["tel"]
    for gene in ("TERT", "TERC"):
        if gene in X:
            tel_predictions[f"{gene} expression alone"] = X[gene].values
    for name, score in tel_predictions.items():
        extra = metadata["tel"] if name.startswith("Locked") else {
            "signature_genes_present": 1, "signature_genes_expected": 1, "signature_genes_imputed": 0,
            "missing_gene_policy": "not applicable", "missing_genes": "", "deployment_threshold": np.nan}
        rows.append(_correlation_row("Wu 2025", "tel", "qTRAP activity %MCF7, log2(n+1)", name, score,
                                     activity_log, analysis_role="primary", **extra))

    if "tel" in predictions:
        finite = np.isfinite(activity_raw)
        for assay_threshold, role in ((2.0, "primary"), (10.0, "sensitivity")):
            target = (activity_raw[finite] >= assay_threshold).astype(int)
            if min(target.sum(), len(target) - target.sum()) < 5:
                continue
            info = metadata["tel"]
            row = _auc_row("Locked telomerase signature", target, predictions["tel"][finite], lineage[finite],
                           threshold=info["deployment_threshold"], source="Wu 2025", task="tel",
                           outcome=f"qTRAP activity >= {assay_threshold:g}% MCF7", scope="all",
                           analysis_role=role, signature_genes_present=info["signature_genes_present"],
                           signature_genes_expected=info["signature_genes_expected"])
            row.update(metric="AUROC", estimate=row["AUROC"], lo=row["AUROC_lo"], hi=row["AUROC_hi"])
            rows.append(row)

    out = pd.DataFrame(rows)
    out.to_csv(C.RESULTS / "validation_wu2025_celllines.csv", index=False)
    if len(out):
        print(out[["task", "outcome", "model", "scope", "n", "metric", "estimate", "lo", "hi"]]
              .round(3).to_string(index=False), flush=True)
    return out


def survival(lab):
    """Cox models with prespecified primary and exploratory confounder adjustments.

    Sensitivities use MKI67 expression, PanCanAtlas RNAss and ABSOLUTE tumor purity. Stage/grade are included
    where recorded; complete-case counts are reported for every model.
    """
    from statsmodels.duration.hazard_regression import PHReg
    from statsmodels.stats.multitest import multipletests
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
        T, _ = load_transcriptome(score.index.tolist())
        if "MKI67" in T.columns:
            mk = pd.to_numeric(T.reindex(score.index)["MKI67"], errors="coerce")
            df["MKI67_sd"] = (mk - mk.mean()) / mk.std()
        else:
            df["MKI67_sd"] = np.nan
        if "RNAss" in df:
            rn = pd.to_numeric(df["RNAss"], errors="coerce")
            df["RNAss_sd"] = (rn - rn.mean()) / rn.std()
        else:
            df["RNAss_sd"] = np.nan
        if "purity" in df:
            purity = pd.to_numeric(df["purity"], errors="coerce")
            df["purity_sd"] = (purity - purity.mean()) / purity.std()
        else:
            df["purity_sd"] = np.nan
        df["stage"] = df.get("ajcc_pathologic_tumor_stage", pd.Series(index=df.index, dtype=object))
        df["stage"] = df["stage"].fillna(df.get("clinical_stage", pd.Series(index=df.index, dtype=object)))
        df["grade"] = df.get("histological_grade", pd.Series(index=df.index, dtype=object))
        for gname, cancers in groups:
            for endpoint in ["OS", "PFI"]:
                cohort = df if cancers is None else df[df.cancer.isin(cancers)]
                adjustments = [
                    ("unadjusted", [], [], "exploratory"),
                    ("age + sex", ["age", "male"], [], "primary"),
                    ("age + sex + stage + grade", ["age", "male"], ["stage", "grade"], "exploratory"),
                    ("age + sex + purity", ["age", "male", "purity_sd"], [], "exploratory"),
                    ("age + sex + MKI67 + RNAss", ["age", "male", "MKI67_sd", "RNAss_sd"], [], "exploratory"),
                    ("age + sex + stage + grade + purity + MKI67 + RNAss",
                     ["age", "male", "purity_sd", "MKI67_sd", "RNAss_sd"],
                     ["stage", "grade"], "exploratory"),
                    ("age + sex + glioma subtype", ["age", "male"], ["subtype"], "primary"),
                ]
                for adj, numeric, categorical, role in adjustments:
                    if "subtype" in categorical and gname not in ("LGG", "GBM"):
                        continue
                    needed = [endpoint, f"{endpoint}.time"] + [n for n in numeric if n != "male"] + categorical
                    if "male" in numeric:
                        needed.append("sex")
                    sub = cohort.dropna(subset=needed).copy()
                    sub = sub[sub[f"{endpoint}.time"] > 0]
                    if len(sub) < 40 or sub[endpoint].sum() < 15:
                        continue
                    design = pd.DataFrame({"score_sd": sub.score_sd.values,
                                           "age": sub.age.values,
                                           "male": (sub.sex.astype(str).str.lower() == "male").astype(float).values,
                                           "purity_sd": sub.purity_sd.values,
                                           "MKI67_sd": sub.MKI67_sd.values,
                                           "RNAss_sd": sub.RNAss_sd.values})
                    dummy_cols = []
                    for cat in categorical:
                        values = sub[cat].astype(str)
                        counts = values.value_counts()
                        values = values.where(values.map(counts) >= 15, "other")
                        if values.nunique() < 2:
                            continue
                        dummies = pd.get_dummies(values, prefix=cat, drop_first=True).astype(float)
                        design = pd.concat([design, dummies.reset_index(drop=True)], axis=1)
                        dummy_cols.extend(dummies.columns)
                    cols = ["score_sd"] + numeric + dummy_cols
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
                                 "analysis_role": role,
                                 "n": len(sub), "events": int(sub[endpoint].sum()), "HR_per_SD": np.exp(b),
                                 "HR_lo": np.exp(b - 1.96 * se), "HR_hi": np.exp(b + 1.96 * se), "p": res.pvalues[0]})
    out = pd.DataFrame(rows)
    if len(out):
        out["p_fdr"] = multipletests(out.p, method="fdr_bh")[1]
        out["significant_fdr"] = out.p_fdr < 0.05
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
    ap.add_argument("--steps", default="external,platform,celllines,wu2025,survival,enrichment,arms")
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
        elif step == "wu2025":
            wu2025_celllines()
        elif step == "survival":
            survival(lab)
        elif step == "enrichment":
            enrichment()
        elif step == "arms":
            chromosome_arms()


if __name__ == "__main__":
    main()
