"""Stage 2: cross-validated benchmark of telomere-maintenance-mechanism (TMM) inference.

Axis of comparison is the *feature set*, not the architecture:
  references         cancer type only; cancer type + glioma molecular subtype
  published methods  Barthel et al. telomerase score; EXTEND (Noureen et al.) score and signature genes
  curated panel      63 telomere-maintenance genes, with LR / random forest / gradient boosting / MLP
  whole transcriptome all protein-coding genes, L2 logistic regression
The compact signature selected from the whole transcriptome is produced by signature.py on the same folds,
and analyze.py merges everything into one summary per task.

Usage:  python run_experiments.py                  # all tasks
        python run_experiments.py --tasks alt,tel  # subset
        python run_experiments.py --quick          # smoke test: 2 folds, 1 repeat, few epochs
"""
import argparse
import json
import platform
import time

import numpy as np
import pandas as pd
import torch

import config as C
import evaluate as E
from models import ScoreModel, SkModel, TorchModel, TunedTorchModel

TASK_INFO = {
    "alt":        ("clf", "ALT-like vs other, gliomas + sarcomas (TERT removed from inputs)"),
    "alt_noATRX": ("clf", "ALT-like, gliomas + sarcomas, without TERT/ATRX/DAXX"),
    "alt_pan":    ("clf", "ALT-like vs other, all cancer types (TERT removed)"),
    "alt_pheno":  ("clf", "ALT by whole-genome telomere-sequence features (Sieverling et al.)"),
    "tel":        ("clf", "TERT-expressing (telomerase-positive) vs non-expressing, all cancers (TERT removed)"),
    "tl":         ("reg", "log tumor/normal telomere-length ratio, WGS/LPS-derived samples only"),
    "tel_no5p15": ("clf", "Telomerase activation without TERT and without all 5p15 genes (TERT neighbourhood)"),
    "tel_no5p":   ("clf", "Telomerase activation without TERT and without all chromosome-5p genes"),
}
TRANSCRIPTOME_TASKS = {"alt", "alt_noATRX", "alt_pan", "alt_pheno", "tel", "tel_no5p15", "tel_no5p"}


class Select:
    """Restricts a model to given gene columns and covariate columns."""
    def __init__(self, model, genes, cov_cols):
        self.model, self.genes, self.cov_cols = model, genes, cov_cols

    def fit(self, X, cov, y):
        self.model.fit(X[self.genes], cov[self.cov_cols], y)
        return self

    def predict(self, X, cov):
        return self.model.predict(X[self.genes], cov[self.cov_cols])


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def load():
    lab = pd.read_csv(C.PROC / "labels.csv")
    expr = pd.read_csv(C.PROC / "expr_panel.csv.gz", index_col=0)
    sets = json.load(open(C.PROC / "gene_sets.json"))
    ext_path = C.PROC / "external_pcawg.csv"
    ext = pd.read_csv(ext_path) if ext_path.exists() else None
    sc_path = C.PROC / "extend_scores.csv"
    if sc_path.exists():                                    # published EXTEND score, used as a baseline column
        expr["extend_score"] = pd.read_csv(sc_path, index_col=0).reindex(expr.index).iloc[:, 0]
    return lab, expr, sets, ext


def load_transcriptome(samples):
    """Whole protein-coding transcriptome (stage 1c). Falls back to the random gene pool for local tests."""
    if (C.PROC / "transcriptome.npy").exists():
        from data_build import load_matrix
        return load_matrix("transcriptome").reindex(samples), "protein-coding transcriptome"
    pool = pd.read_csv(C.PROC / "expr_random.csv.gz", index_col=0).reindex(samples)
    return pool, "random gene pool (transcriptome matrix not built)"


def task_frame(task, lab, expr, ext=None, exclude_patients=()):
    if task == "alt_pheno":
        e = ext[ext["sample"].isin(expr.index)].dropna(subset=["alt_probability"]).copy()
        e = e[~e.patient.isin(exclude_patients)]
        X = expr.loc[e["sample"]].reset_index(drop=True)
        cov = pd.DataFrame({"cancer": e.cancer.values, "subtype": e.cancer.values})
        return e.reset_index(drop=True), X, cov, e.y_alt_wgs.values.astype(float)
    if task in ("alt", "alt_noATRX"):
        d = lab[lab.extended & lab.cancer.isin(C.ALT_COHORT)]
        y = d.y_alt
    elif task == "alt_pan":
        d, y = lab[lab.extended], lab.y_alt[lab.extended]
    elif task.startswith("tel"):
        d, y = lab[lab.extended], lab.y_tel[lab.extended]
    else:
        d = lab[lab.lib.isin(C.TL_RELIABLE_LIBS) & lab.tl_ratio.notna()]
        y = d.tl_ratio
    keep = ~d.patient.isin(exclude_patients)
    d, y = d[keep], y[keep]
    X = expr.loc[d["sample"]].reset_index(drop=True)
    X["sig_score"] = d["sig_score"].values
    cov = pd.DataFrame({"cancer": d.cancer.values,
                        "subtype": d.subtype.fillna(d.cancer).astype(str).values})
    return d.reset_index(drop=True), X, cov, y.values.astype(float)


def excluded(task):
    out = set(C.EXCLUDE.get(task, C.EXCLUDE.get(task.split("_")[0], [])))
    prefix = C.REGION_EXCLUDE.get(task)
    if prefix:                                   # remove every gene in the cytogenetic region
        hg = C.RAW / "hgnc_complete_set.txt"
        if not hg.exists():
            from data_build import download
            download(C.HGNC_COMPLETE, hg)
        h = pd.read_csv(hg, sep="\t", usecols=["symbol", "location"], low_memory=False).dropna()
        region = set(h.loc[h.location.str.startswith(prefix), "symbol"])
        out |= region
    return out


# ---------------------------------------------------------------------------
# Models per task
# ---------------------------------------------------------------------------
def model_specs(task, genes, sets, X, quick, tune=True, tune_iters=None):
    kind = TASK_INFO[task][0]
    nn = {"max_epochs": 15, "patience": 5} if quick else None
    base = ["cancer"]
    specs = {
        "Cancer type only": (lambda s: SkModel("cov_only", kind, s, use_genes=False), genes, base),
        "LR (panel)": (lambda s: SkModel("lr", kind, s), genes, base),
        "Elastic net (panel)": (lambda s: SkModel("elasticnet", kind, s), genes, base),
        "RF (panel)": (lambda s: SkModel("rf", kind, s), genes, base),
        "GBM (panel)": (lambda s: SkModel("gbm", kind, s), genes, base),
        "MLP (panel)": (lambda s: TorchModel("mlp", kind, seed=s, params=nn), genes, base),
    }
    if kind == "clf":
        specs["Linear SVM (panel)"] = (lambda s: SkModel("svm", kind, s), genes, base)
    if tune:
        specs["MLP tuned (panel)"] = (
            lambda s: TunedTorchModel(kind, seed=s, n_iter=tune_iters,
                                      overrides=({"max_epochs": 15, "patience": 5} if quick else None)),
            genes, base)
    if task == "alt":
        specs["Cancer + glioma subtype only"] = (lambda s: SkModel("cov_only", kind, s, use_genes=False), genes, ["subtype"])
        specs["LR (panel) + glioma subtype"] = (lambda s: SkModel("lr", kind, s), genes, ["subtype"])
    if task in ("alt_noATRX", "alt_pan", "alt_pheno") or task in C.REGION_EXCLUDE:
        for k in ["RF (panel)", "MLP (panel)", "MLP tuned (panel)"]:
            specs.pop(k, None)
    if task == "tel":
        sig = [g for g in sets.get("signature", []) if g in X.columns and g != "TERT"]
        specs["Barthel 2017 score (published)"] = (lambda s: ScoreModel("sig_score"), ["sig_score"], base)
        specs["LR (Barthel 2017 genes)"] = (lambda s: SkModel("lr", kind, s), sig, base)
        # The published EXTEND score is evaluated only against independent enzymatic activity.  On this
        # TERT-expression-defined target it would be circular.  Retain TERC here: only TERT defines the label.
        ext_genes = [g for g in sets.get("extend_signature", []) if g in X.columns and g != "TERT"]
        if ext_genes:
            specs["LR (EXTEND gene set, TERT removed)"] = (lambda s: SkModel("lr", kind, s), ext_genes, base)
    return specs


# ---------------------------------------------------------------------------
# Task runner
# ---------------------------------------------------------------------------
def run_task(task, lab, expr, sets, ext, args):
    kind, desc = TASK_INFO[task]
    d, X, cov, y = task_frame(task, lab, expr, ext=ext)
    excl = excluded(task)
    genes = [g for g in C.PANEL_GENES if g in X.columns and g not in excl]
    print(f"\n=== {task}: {desc}\n    n={len(y)}" + (f", positives={int(y.sum())}" if kind == "clf" else "") +
          f", panel genes={len(genes)}, cancers={cov.cancer.nunique()}, excluded genes={len(excl)}", flush=True)
    folds = E.make_folds(y, cov.cancer, kind, args.folds, args.repeats)
    specs = model_specs(task, genes, sets, X, args.quick,
                        tune=(task in ("alt", "tel") and not args.no_tune), tune_iters=args.tune_iters)
    if args.models:
        specs = {k: v for k, v in specs.items() if k in args.models or k == "Cancer type only"}

    oof, importance, timing = {}, [], {}
    cancers = E.eligible_cancers(y, cov.cancer.values, kind)
    for name, (factory, g, cov_cols) in specs.items():
        t0 = time.time()

        def after(m, rep, k, te, name=name, g=g):
            if rep == 0 and kind == "clf" and name == "LR (panel)" and not args.quick:
                importance.append(E.permutation_importance(m, X.iloc[te][g], cov.iloc[te], y[te]).rename(f"fold{k}"))

        oof[name] = E.run_cv(lambda s, f=factory, g=g, c=cov_cols: Select(f(s), g, c), X, cov, y, folds, after)
        timing[name] = time.time() - t0
        m0 = E.metrics(y, np.nanmean(oof[name], 0), cov.cancer.values, cancers, kind)
        key = ("AUROC", "within_AUROC") if kind == "clf" else ("Spearman", "within_Spearman")
        print(f"    {name:44s} {key[0]}={m0[key[0]]:.3f}  {key[1]}={m0[key[1]]:.3f}  ({timing[name]:.0f}s)", flush=True)

    np.savez_compressed(C.RESULTS / f"{task}_oof.npz", y=y, cancer=cov.cancer.values, sample=d["sample"].values,
                        subtype=cov.subtype.values, **{k.replace(" ", "_"): v for k, v in oof.items()})
    pd.Series(timing, name="seconds").to_csv(C.RESULTS / f"{task}_timing.csv")
    if importance:
        pd.concat(importance, axis=1).to_csv(C.RESULTS / f"{task}_importance.csv")

    if task in TRANSCRIPTOME_TASKS and not args.no_transcriptome:
        transcriptome_model(task, d, cov, y, folds, excl, args)
    if task in ("alt", "tel") and args.random_sets:
        random_null(task, d, X, cov, y, folds, genes, oof["LR (panel)"], excl, args)


def transcriptome_model(task, d, cov, y, folds, excl, args):
    """Strong linear baselines on every protein-coding gene, all using the same outer folds."""
    T, source = load_transcriptome(d["sample"].tolist())
    T = T.reset_index(drop=True)
    ok = T.notna().all().values & (T.std().values > 0)
    genes = [g for g, keep in zip(T.columns, ok) if keep and g not in excl]
    kindm = "lr" if args.quick or len(genes) < 5000 else "lr_wide"
    methods = [("LR (whole transcriptome)", kindm)]
    # The full model suite is a primary-task comparison.  Secondary ablations retain the same
    # transcriptome-wide LR on identical folds without multiplying several very expensive 19k-gene
    # inner searches; panel elastic net/SVM remain in their task OOF files.
    if task in ("alt", "tel"):
        methods += [("Elastic net (whole transcriptome)", "elasticnet_wide"),
                    ("Linear SVM (whole transcriptome)", "svm")]
    saved = {}
    for name, model_kind in methods:
        t0 = time.time()
        oof = E.run_cv(lambda s, mk=model_kind: Select(SkModel(mk, "clf", s), genes, ["cancer"]),
                       T, cov, y, folds)
        saved[name] = oof
        m = E.metrics(y, np.nanmean(oof, 0), cov.cancer.values,
                      E.eligible_cancers(y, cov.cancer.values, "clf"), "clf")
        print(f"    {name:44s} AUROC={m['AUROC']:.3f}  within_AUROC={m['within_AUROC']:.3f}  "
              f"({time.time() - t0:.0f}s; {len(genes)} genes from {source})", flush=True)
    np.savez_compressed(C.RESULTS / f"{task}_tw_oof.npz", n_genes=len(genes), source=source,
                        **{name.replace(" ", "_"): arr for name, arr in saved.items()})


def random_null(task, d, X, cov, y, folds, genes, oof_panel, excl, args):
    """Is the curated panel better than random gene sets of the same size (same model, same folds)?"""
    T, source = load_transcriptome(d["sample"].tolist())
    T = T.reset_index(drop=True)
    ok = T.notna().all().values & (T.std().values > 0)
    panel = set(C.PANEL_GENES)
    pool = [g for g, keep in zip(T.columns, ok) if keep and g not in excl and g not in panel]
    rng = np.random.default_rng(C.SEED)
    folds0 = [f for f in folds if f[0] == 0]
    cancers = E.eligible_cancers(y, cov.cancer.values, "clf")

    def score(s):
        m = E.metrics(y, s, cov.cancer.values, cancers, "clf")
        return m["AUROC"], m["within_AUROC"]

    # A fixed, prespecified LR penalty is used for both the curated and random panels.  This avoids
    # thousands of redundant inner searches while keeping the feature-set comparison exactly matched.
    # The curated panel includes the noncoding telomerase RNA TERC, whereas the random pool is
    # intentionally protein-coding.  Score the real panel from the already aligned panel matrix and
    # random sets from the whole-transcriptome matrix; both use exactly the same LR and folds.
    panel_fixed = E.run_cv(lambda s: Select(SkModel("lr_fixed", "clf", s), genes, ["cancer"]),
                           X, cov, y, folds0)
    rows = [("telomere panel", *score(panel_fixed[0]))]
    for i in range(args.random_sets):
        rg = rng.choice(pool, len(genes), replace=False).tolist()
        o = E.run_cv(lambda s: Select(SkModel("lr_fixed", "clf", s), rg, ["cancer"]), T, cov, y, folds0)
        rows.append((f"random_{i}", *score(o[0])))
    res = pd.DataFrame(rows, columns=["gene_set", "AUROC", "within_AUROC"])
    res.to_csv(C.RESULTS / f"{task}_random_null.csv", index=False)
    r = res.iloc[1:]
    p = (1 + (r.within_AUROC >= res.within_AUROC.iloc[0]).sum()) / (1 + len(r))
    print(f"    panel within_AUROC={res.within_AUROC.iloc[0]:.3f} vs {len(r)} random sets "
          f"(median {r.within_AUROC.median():.3f}); empirical p={p:.3f} [{source}]", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default="alt,alt_noATRX,alt_pan,alt_pheno,tel,tl")
    ap.add_argument("--models", default=None, help="comma-separated subset of model names")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--no-tune", action="store_true", dest="no_tune")
    ap.add_argument("--no-transcriptome", action="store_true", dest="no_transcriptome")
    args = ap.parse_args()
    args.models = args.models.split(",") if args.models else None
    args.folds, args.repeats = (2, 1) if args.quick else (C.N_FOLDS, C.N_REPEATS)
    args.random_sets = 3 if args.quick else C.N_RANDOM_SETS
    args.tune_iters = 2 if args.quick else C.NN_SEARCH["n_iter"]

    lab, expr, sets, ext = load()
    import sklearn
    info = {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
            "scikit-learn": sklearn.__version__, "torch": torch.__version__, "machine": platform.machine(),
            "cpus": __import__("os").cpu_count(), "args": vars(args), "started": time.strftime("%Y-%m-%d %H:%M:%S")}
    print(json.dumps(info), flush=True)
    t0 = time.time()
    for t in args.tasks.split(","):
        run_task(t, lab, expr, sets, ext, args)
    info["minutes"] = (time.time() - t0) / 60
    json.dump(info, open(C.RESULTS / f"run_info_{args.tasks.replace(',', '_')}.json", "w"), indent=1)
    print(f"\nAll done in {info['minutes']:.1f} min. Results in {C.RESULTS}")


if __name__ == "__main__":
    main()
