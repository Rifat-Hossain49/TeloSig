"""Stage 2b: compact expression signatures selected from the whole transcriptome.

Question: can a panel-sized gene list keep the performance of a whole-transcriptome model?

For every training fold of the same 5x5 cross-validation used by run_experiments.py:
  1. rank all candidate genes on the training tumors only, with two independent methods
       l1          |coefficient| of an L1-penalised logistic regression (multivariable)
       univariate  |correlation| with the label after centring genes and label within cancer type
  2. for each size K in config.SIG_K_GRID, refit an L2 logistic regression (+ cancer type) on the top K
     genes and predict the held-out tumors.
Gene selection therefore never sees test tumors. The L1 strength is picked on held-out data inside the
first training fold only (averaged over three splits).

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
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.model_selection import train_test_split
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


def rank_l1(Z, y, c, seed=C.SEED):
    m = LogisticRegression(penalty="l1", solver="liblinear", C=c, class_weight="balanced",
                           max_iter=3000, random_state=seed).fit(Z, y)
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


def choose_c(T, y, cancer, genes, folds, grid, k):
    tr_all = folds[0][2]
    best, best_score = grid[0], -np.inf
    for c in grid:
        scores = []
        for rep in range(3):
            tr, va = train_test_split(tr_all, test_size=0.25, random_state=C.SEED + rep, stratify=y[tr_all])
            Z, sc = _standardise(T.iloc[tr][genes].values)
            order, _ = rank_l1(Z, y[tr], c)
            top = [genes[i] for i in order[:k]]
            cov = pd.DataFrame({"cancer": cancer})
            m = Select(SkModel("lr", "clf"), top, ["cancer"]).fit(T.iloc[tr], cov.iloc[tr], y[tr])
            scores.append(E.auc(y[va], m.predict(T.iloc[va], cov.iloc[va])))
        s = float(np.mean(scores))
        print(f"    L1 strength C={c}: inner AUROC {s:.3f}", flush=True)
        if s > best_score:
            best, best_score = c, s
    return best


# ---------------------------------------------------------------------------
# Locked signature for reuse / external validation
# ---------------------------------------------------------------------------
def lock_signature(T, y, cancer, genes, c, k, path, note):
    """Select on all given tumors, then fit gene weights (with cancer type as covariate) and store scaling."""
    Z, sc = _standardise(T[genes].values)
    order, _ = rank_l1(Z, y, c)
    top = [genes[i] for i in order[:k]]
    idx = [genes.index(g) for g in top]
    onehot = pd.get_dummies(pd.Series(cancer)).values.astype(np.float32)
    design = np.hstack([Z[:, idx], onehot])
    m = LogisticRegressionCV(Cs=np.logspace(-3, 1, 9), cv=3, scoring="roc_auc", class_weight="balanced",
                             max_iter=5000, random_state=C.SEED).fit(design, y)
    locked = {"note": note, "n_tumors": int(len(y)), "n_positive": int(y.sum()), "l1_C": c, "k": k,
              "genes": top, "weights": m.coef_.ravel()[:len(top)].round(6).tolist(),
              "mean": sc.mean_[idx].round(6).tolist(), "sd": np.sqrt(sc.var_[idx]).round(6).tolist(),
              "usage": "score = sum(weight * (log2 expression - mean) / sd); re-estimate mean/sd within a new cohort "
                       "when its platform or units differ"}
    json.dump(locked, open(path, "w"), indent=1)
    return locked


def score_locked(locked, X, restandardise=False):
    G = X[locked["genes"]].values.astype(float)
    if restandardise:
        mu, sd = np.nanmean(G, 0), np.nanstd(G, 0) + 1e-8
    else:
        mu, sd = np.array(locked["mean"]), np.array(locked["sd"]) + 1e-8
    return np.nan_to_num((G - mu) / sd) @ np.array(locked["weights"])


# ---------------------------------------------------------------------------
def derive(task, lab, expr, ext, args):
    d, _, cov, y = task_frame(task, lab, expr, ext=ext)
    T, source = load_transcriptome(d["sample"].tolist())
    T = T.reset_index(drop=True)
    excl = excluded(task)
    ok = T.notna().all().values & (T.std().values > 0)
    genes = [g for g, keep in zip(T.columns, ok) if keep and g not in excl]
    cancer = cov.cancer.values
    folds = E.make_folds(y, cov.cancer, "clf", args.folds, args.repeats)
    print(f"\n=== {task}: signatures from {len(genes)} candidate genes ({source}); n={len(y)}, "
          f"positives={int(y.sum())}", flush=True)
    c = choose_c(T, y, cancer, genes, folds, C.SIG_C_GRID, C.SIG_K)
    print(f"    chosen C={c}", flush=True)

    n_rep = max(f[0] for f in folds) + 1
    methods = ["l1", "univariate"]
    oof = {(mth, k): np.full((n_rep, len(y)), np.nan) for mth in methods for k in C.SIG_K_GRID}
    picked = {mth: [] for mth in methods}
    t0 = time.time()
    for rep, fold, tr, te in folds:
        Z, _ = _standardise(T.iloc[tr][genes].values)
        orders = {"l1": rank_l1(Z, y[tr], c)[0], "univariate": rank_univariate(Z, y[tr], cancer[tr])}
        for mth, order in orders.items():
            picked[mth].append([genes[i] for i in order[:C.SIG_K]])
            for k in C.SIG_K_GRID:
                top = [genes[i] for i in order[:k]]
                m = Select(SkModel("lr", "clf", C.SEED + 100 * rep + fold), top, ["cancer"])
                m.fit(T.iloc[tr], cov.iloc[tr], y[tr])
                oof[(mth, k)][rep, te] = m.predict(T.iloc[te], cov.iloc[te])
        print(f"    fold {rep}.{fold} done ({time.time() - t0:.0f}s)", flush=True)

    cancers = E.eligible_cancers(y, cancer, "clf")
    curve = []
    for (mth, k), o in oof.items():
        mm = E.metrics(y, np.nanmean(o, 0), cancer, cancers, "clf")
        curve.append({"method": mth, "k": k, "AUROC": mm["AUROC"], "within_AUROC": mm["within_AUROC"]})
    curve = pd.DataFrame(curve)
    print(curve.pivot(index="k", columns="method", values="within_AUROC").round(3).to_string(), flush=True)

    freq = pd.Series([g for gs in picked["l1"] for g in gs]).value_counts() / len(folds)
    stab = pd.DataFrame({"gene": freq.index, "selection_frequency": freq.values})
    stab["in_telomere_panel"] = stab.gene.isin(C.PANEL_GENES)
    stab["univariate_frequency"] = stab.gene.map(
        pd.Series([g for gs in picked["univariate"] for g in gs]).value_counts() / len(folds)).fillna(0)
    stab.to_csv(C.RESULTS / f"{task}_signature_genes.csv", index=False)
    jac = [len(set(a) & set(b)) / len(set(a) | set(b)) for a, b in itertools.combinations(picked["l1"], 2)]
    stable = stab[stab.selection_frequency >= C.SIG_STABLE]

    np.savez_compressed(C.RESULTS / f"{task}_sig_oof.npz", oof=oof[("l1", C.SIG_K)])
    np.savez_compressed(C.RESULTS / f"{task}_sig_curve_oof.npz",
                        **{f"{mth}_K{k}": o for (mth, k), o in oof.items()})
    curve.to_csv(C.RESULTS / f"{task}_signature_curve.csv", index=False)

    locked = lock_signature(T, y, cancer, genes, c, C.SIG_K, C.FROZEN / f"{task}_signature_locked.json",
                            f"{task}: fitted on all {len(y)} labelled tumors")
    if ext is not None and task in ("alt", "alt_pan", "tel"):
        keep = ~d.patient.isin(set(ext.patient)).values
        lock_signature(T[keep].reset_index(drop=True), y[keep], cancer[keep], genes, c, C.SIG_K,
                       C.FROZEN / f"{task}_signature_locked_heldout.json",
                       f"{task}: fitted without the {int((~keep).sum())} PCAWG donors (external validation)")
    summary = {"task": task, "candidates": len(genes), "source": source, "l1_C": c, "k": C.SIG_K,
               "n_folds": len(folds), "mean_jaccard_between_folds": float(np.mean(jac)),
               "stable_genes": stable.gene.tolist(), "locked_genes": locked["genes"],
               "curve": curve.to_dict(orient="records")}
    json.dump(summary, open(C.RESULTS / f"{task}_signature.json", "w"), indent=1)
    print(f"    stable genes (>= {C.SIG_STABLE:.0%} of folds): {len(stable)} "
          f"({int(stable.in_telomere_panel.sum())} from the telomere panel); "
          f"mean Jaccard between folds {np.mean(jac):.2f}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default="alt,tel")
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    a.folds, a.repeats = (2, 1) if a.quick else (C.N_FOLDS, C.N_REPEATS)
    lab, expr, sets, ext = load()
    for t in a.tasks.split(","):
        derive(t, lab, expr, ext, a)


if __name__ == "__main__":
    main()
