"""Stage 2b: compact expression signatures selected from the whole transcriptome.

Question: can a panel-sized gene list keep the performance of a whole-transcriptome model?

For every training fold of the same 5x5 cross-validation used by run_experiments.py:
  1. rank all candidate genes on the training tumors only, with two independent methods
       l1          |coefficient| of an L1-penalised logistic regression (multivariable)
       univariate  |correlation| with the label after centring genes and label within cancer type
  2. for each size K in config.SIG_K_GRID, refit an L2 logistic regression (+ cancer type) on the top K
     genes and predict the held-out tumors.
Gene selection therefore never sees test tumors.  The L1 strength is tuned again inside every outer
training set; no choice made for one outer fold is reused in another.

Outputs (results/):
  {task}_sig_oof.npz            headline signature (L1, K = config.SIG_K) -> merged by analyze.py
  {task}_sig_curve_oof.npz      every method x K, for the size-performance curve
  {task}_signature_genes.csv    selection frequency per gene across folds (stability)
  {task}_signature.json         summary: chosen C, stable genes, Jaccard stability, metrics per K
Outputs (data/frozen/):
  {task}_signature_locked.json          genes, weights and scaling fitted on all tumors (for reuse)
  {task}_signature_locked_heldout.json  same, fitted without the PCAWG donors (for external validation)

Usage: python signature.py [--tasks alt,tel] [--quick]
"""
import argparse
import itertools
import json
import time

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegressionCV, SGDClassifier
from sklearn.metrics import brier_score_loss, roc_curve
from sklearn.preprocessing import StandardScaler

import config as C
import evaluate as E
from models import SkModel
from run_experiments import Select, excluded, load, load_transcriptome, task_frame


# ---------------------------------------------------------------------------
# Gene ranking
# ---------------------------------------------------------------------------
def _standardise(X):
    sc = StandardScaler().fit(X)
    return np.nan_to_num(sc.transform(X)).astype(np.float32), sc


def rank_l1(Z, y, alpha, seed=C.SEED):
    # Stochastic L1 logistic regression has the same sparse multivariable objective used for ranking,
    # while remaining tractable for pan-cancer 7k x 19k matrices inside nested CV and 100 subsamples.
    m = SGDClassifier(loss="log_loss", penalty="l1", alpha=alpha, class_weight="balanced",
                      max_iter=3000, tol=1e-3, early_stopping=True, validation_fraction=0.15,
                      n_iter_no_change=10, random_state=seed).fit(Z, y)
    w = np.abs(m.coef_.ravel())
    return np.argsort(-w, kind="stable"), int((w > 0).sum())


def rank_univariate(Z, y, cancer):
    """|within-cancer correlation| between each gene and the label (vectorised over genes)."""
    Zc, yc = Z.copy(), y.astype(np.float32).copy()
    for c in np.unique(cancer):
        k = cancer == c
        Zc[k] -= Zc[k].mean(0)
        yc[k] -= yc[k].mean()
    num = Zc.T @ yc
    den = np.sqrt((Zc ** 2).sum(0) * (yc ** 2).sum()) + 1e-12
    return np.argsort(-np.abs(num / den), kind="stable")


def eligible_genes(T, genes, rows):
    """Label-free availability/variance filter estimated from the supplied training rows only."""
    x = T.iloc[rows][genes]
    ok = x.notna().all().values & (x.std().values > 0)
    return [g for g, keep in zip(genes, ok) if keep]


def choose_c(T, y, cancer, genes, rows, grid, k, seed=C.SEED):
    """Tune L1 strength using inner OOF predictions confined to one outer training set."""
    rows = np.asarray(rows)
    inner = E.make_folds(y[rows], cancer[rows], "clf", C.SIG_INNER_FOLDS, 1, seed)
    cov = pd.DataFrame({"cancer": cancer})
    best, best_score = grid[0], -np.inf
    for c in grid:
        pred = np.full(len(rows), np.nan)
        for _, fold, tr0, va0 in inner:
            tr, va = rows[tr0], rows[va0]
            fold_genes = eligible_genes(T, genes, tr)
            Z, _ = _standardise(T.iloc[tr][fold_genes].values)
            order, _ = rank_l1(Z, y[tr], c, seed + fold)
            top = [fold_genes[i] for i in order[:min(k, len(order))]]
            m = Select(SkModel("lr", "clf", seed + fold), top, ["cancer"])
            m.fit(T.iloc[tr], cov.iloc[tr], y[tr])
            pred[va0] = m.predict(T.iloc[va], cov.iloc[va])
        cancers = E.eligible_cancers(y[rows], cancer[rows], "clf")
        s = E.within(y[rows], pred, cancer[rows], cancers, "clf") if cancers else E.auc(y[rows], pred)
        if s > best_score:
            best, best_score = c, s
    return best


# ---------------------------------------------------------------------------
# Locked signature for reuse / external validation
# ---------------------------------------------------------------------------
def lock_signature(T, y, cancer, genes, alpha, k, path, note):
    """Select, preprocess, fit and set a deployment threshold using only the supplied training cohort."""
    Z, sc = _standardise(T[genes].values)
    order, _ = rank_l1(Z, y, alpha)
    top = [genes[i] for i in order[:k]]
    idx = [genes.index(g) for g in top]
    design = Z[:, idx]
    m = LogisticRegressionCV(Cs=np.logspace(-3, 1, 9), cv=3, scoring="roc_auc", class_weight="balanced",
                             max_iter=5000, random_state=C.SEED).fit(design, y)
    probability = m.predict_proba(design)[:, 1]
    fpr, tpr, thresholds = roc_curve(y, probability)
    threshold = float(thresholds[np.nanargmax(tpr - fpr)])
    called = probability >= threshold
    pos, neg = y == 1, y == 0
    deployment = {
        "threshold": threshold,
        "training_brier": float(brier_score_loss(y, probability)),
        "training_sensitivity": float(called[pos].mean()),
        "training_specificity": float((~called[neg]).mean()),
        "training_ppv": float(y[called].mean()) if called.any() else None,
        "training_npv": float((1 - y[~called]).mean()) if (~called).any() else None,
        "warning": "Training-cohort operating characteristics are descriptive, not external performance estimates."
    }
    locked = {"note": note, "n_tumors": int(len(y)), "n_positive": int(y.sum()),
              "l1_sgd_alpha": alpha, "k": k,
              "genes": top, "weights": m.coef_.ravel()[:len(top)].round(6).tolist(),
              "intercept": float(m.intercept_[0]),
              "mean": sc.mean_[idx].round(6).tolist(), "sd": np.sqrt(sc.var_[idx]).round(6).tolist(),
              "deployment": deployment,
              "usage": "logit = intercept + sum(weight * (log2 expression - training_mean) / training_sd); "
                       "probability = sigmoid(logit). Never re-estimate mean/SD on the target cohort."}
    with open(path, "w") as handle:
        json.dump(locked, handle, indent=1)
    return locked


def score_locked(locked, X, probability=False, missing="error"):
    """Apply immutable training-derived preprocessing; target-cohort normalisation is forbidden.

    ``missing="training_mean"`` is the deployment-safe policy for a target assay that does not measure every
    locked transcript.  Reindexing creates NaNs for absent features and the standardised contribution of each
    such feature is set to zero, exactly as if its immutable TCGA training mean had been supplied.  The default
    remains strict so an accidental schema mismatch cannot pass silently in ordinary scoring.
    """
    absent = [g for g in locked["genes"] if g not in X.columns]
    if absent and missing != "training_mean":
        raise KeyError(f"locked-model genes absent from target matrix: {absent}")
    if missing not in ("error", "training_mean"):
        raise ValueError("missing must be 'error' or 'training_mean'")
    G = X.reindex(columns=locked["genes"]).values.astype(float)
    mu, sd = np.array(locked["mean"]), np.array(locked["sd"]) + 1e-8
    logit = float(locked.get("intercept", 0.0)) + np.nan_to_num((G - mu) / sd) @ np.array(locked["weights"])
    return 1 / (1 + np.exp(-np.clip(logit, -40, 40))) if probability else logit


def subsample_selection_frequency(T, y, cancer, genes, c, k, n_subsamples, seed=C.SEED):
    """Dedicated stratified subsampling analysis, separate from overlapping outer-CV folds."""
    if n_subsamples <= 0:
        return pd.Series(dtype=float)
    rng = np.random.default_rng(seed)
    key = pd.Series(cancer.astype(str) + "|" + y.astype(int).astype(str))
    groups = [np.flatnonzero(key.values == v) for v in key.unique()]
    picked = []
    for b in range(n_subsamples):
        # Some WGS phenotype/cancer strata are singletons.  Preserve such rare strata instead of
        # requesting two observations without replacement; larger strata retain the prespecified 80%.
        rows = np.concatenate([
            rng.choice(g, min(len(g), max(1, int(np.floor(len(g) * C.SIG_STABILITY_FRACTION)))),
                       replace=False)
            for g in groups
        ])
        fold_genes = eligible_genes(T, genes, rows)
        Z, _ = _standardise(T.iloc[rows][fold_genes].values)
        order, _ = rank_l1(Z, y[rows], c, seed + b)
        picked.append([fold_genes[i] for i in order[:min(k, len(order))]])
    return pd.Series([g for selected in picked for g in selected]).value_counts() / n_subsamples


# ---------------------------------------------------------------------------
def derive(task, lab, expr, ext, args):
    d, _, cov, y = task_frame(task, lab, expr, ext=ext)
    T, source = load_transcriptome(d["sample"].tolist())
    T = T.reset_index(drop=True)
    excl = excluded(task)
    candidate_genes = [g for g in T.columns if g not in excl]
    genes = eligible_genes(T, candidate_genes, np.arange(len(T)))
    cancer = cov.cancer.values
    folds = E.make_folds(y, cov.cancer, "clf", args.folds, args.repeats)
    print(f"\n=== {task}: signatures from {len(genes)} candidate genes ({source}); n={len(y)}, "
          f"positives={int(y.sum())}", flush=True)
    n_rep = max(f[0] for f in folds) + 1
    methods = ["l1", "univariate"]
    oof = {(mth, k): np.full((n_rep, len(y)), np.nan) for mth in methods for k in C.SIG_K_GRID}
    picked = {mth: [] for mth in methods}
    chosen_cs = []
    t0 = time.time()
    for rep, fold, tr, te in folds:
        fold_genes = eligible_genes(T, candidate_genes, tr)
        c = choose_c(T, y, cancer, fold_genes, tr, C.SIG_C_GRID, C.SIG_K,
                     C.SEED + 100 * rep + fold)
        chosen_cs.append(c)
        Z, _ = _standardise(T.iloc[tr][fold_genes].values)
        orders = {"l1": rank_l1(Z, y[tr], c, C.SEED + 100 * rep + fold)[0],
                  "univariate": rank_univariate(Z, y[tr], cancer[tr])}
        for mth, order in orders.items():
            picked[mth].append([fold_genes[i] for i in order[:C.SIG_K]])
            for k in C.SIG_K_GRID:
                top = [fold_genes[i] for i in order[:k]]
                m = Select(SkModel("lr", "clf", C.SEED + 100 * rep + fold), top, ["cancer"])
                m.fit(T.iloc[tr], cov.iloc[tr], y[tr])
                oof[(mth, k)][rep, te] = m.predict(T.iloc[te], cov.iloc[te])
        print(f"    fold {rep}.{fold} done: nested L1 alpha={c} ({time.time() - t0:.0f}s)", flush=True)

    cancers = E.eligible_cancers(y, cancer, "clf")
    curve = []
    for (mth, k), o in oof.items():
        mm = E.metrics(y, np.nanmean(o, 0), cancer, cancers, "clf")
        curve.append({"method": mth, "k": k, "AUROC": mm["AUROC"], "within_AUROC": mm["within_AUROC"]})
    curve = pd.DataFrame(curve)
    print(curve.pivot(index="k", columns="method", values="within_AUROC").round(3).to_string(), flush=True)

    freq = pd.Series([g for gs in picked["l1"] for g in gs]).value_counts() / len(folds)
    stab = pd.DataFrame({"gene": freq.index, "outer_fold_selection_frequency": freq.values})
    stab["in_telomere_panel"] = stab.gene.isin(C.PANEL_GENES)
    stab["univariate_frequency"] = stab.gene.map(
        pd.Series([g for gs in picked["univariate"] for g in gs]).value_counts() / len(folds)).fillna(0)
    jac = [len(set(a) & set(b)) / len(set(a) | set(b)) for a, b in itertools.combinations(picked["l1"], 2)]

    np.savez_compressed(C.RESULTS / f"{task}_sig_oof.npz", oof=oof[("l1", C.SIG_K)])
    np.savez_compressed(C.RESULTS / f"{task}_sig_curve_oof.npz",
                        **{f"{mth}_K{k}": o for (mth, k), o in oof.items()})
    curve.to_csv(C.RESULTS / f"{task}_signature_curve.csv", index=False)

    all_rows = np.arange(len(y))
    c_all = choose_c(T, y, cancer, genes, all_rows, C.SIG_C_GRID, C.SIG_K, C.SEED + 9000)
    n_stability = args.stability_subsamples
    sf = subsample_selection_frequency(T, y, cancer, genes, c_all, C.SIG_K, n_stability)
    stab["subsample_selection_frequency"] = stab.gene.map(sf).fillna(0)
    for g, v in sf.items():
        if g not in set(stab.gene):
            stab.loc[len(stab)] = [g, 0.0, g in C.PANEL_GENES, 0.0, v]
    stab = stab.sort_values(["subsample_selection_frequency", "outer_fold_selection_frequency"], ascending=False)
    stab.to_csv(C.RESULTS / f"{task}_signature_genes.csv", index=False)
    stable = stab[stab.subsample_selection_frequency >= C.SIG_STABLE] if n_stability else stab.iloc[0:0]

    locked = lock_signature(T, y, cancer, genes, c_all, C.SIG_K, C.FROZEN / f"{task}_signature_locked.json",
                            f"{task}: fitted on all {len(y)} labelled tumors")
    if ext is not None and task in ("alt", "alt_pan", "tel"):
        keep = ~d.patient.isin(set(ext.patient)).values
        T_hold = T[keep].reset_index(drop=True)
        y_hold, cancer_hold = y[keep], cancer[keep]
        hold_genes = eligible_genes(T_hold, candidate_genes, np.arange(len(T_hold)))
        c_hold = choose_c(T_hold, y_hold, cancer_hold, hold_genes, np.arange(len(T_hold)),
                          C.SIG_C_GRID, C.SIG_K, C.SEED + 12000)
        lock_signature(T_hold, y_hold, cancer_hold, hold_genes, c_hold, C.SIG_K,
                       C.FROZEN / f"{task}_signature_locked_heldout.json",
                       f"{task}: PCAWG donors excluded before tuning, selection, scaling and fitting "
                       f"({int((~keep).sum())} donors held out)")
    summary = {"task": task, "candidates": len(genes), "source": source,
               "outer_fold_l1_sgd_alpha": chosen_cs, "locked_l1_sgd_alpha": c_all, "k": C.SIG_K,
               "n_folds": len(folds), "mean_jaccard_between_folds": float(np.mean(jac)),
               "stability_method": (f"{n_stability} stratified {C.SIG_STABILITY_FRACTION:.0%} subsamples"
                                    if n_stability else "not run for sensitivity-only task"),
               "stable_genes": stable.gene.tolist(), "locked_genes": locked["genes"],
               "curve": curve.to_dict(orient="records")}
    json.dump(summary, open(C.RESULTS / f"{task}_signature.json", "w"), indent=1)
    print(f"    stable genes (>= {C.SIG_STABLE:.0%} of folds): {len(stable)} "
          f" across dedicated subsamples ({int(stable.in_telomere_panel.sum())} from the telomere panel); "
          f"mean Jaccard between folds {np.mean(jac):.2f}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default="alt,tel")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--stability-subsamples", type=int, default=C.SIG_STABILITY_SUBSAMPLES)
    a = ap.parse_args()
    a.folds, a.repeats = (2, 1) if a.quick else (C.N_FOLDS, C.N_REPEATS)
    if a.quick:
        a.stability_subsamples = min(a.stability_subsamples, 3)
    lab, expr, sets, ext = load()
    for t in a.tasks.split(","):
        derive(t, lab, expr, ext, a)


if __name__ == "__main__":
    main()
