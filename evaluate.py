"""Cross-validation, metrics and paired bootstrap inference.

Design:
* Every model in a task sees identical folds (repeated stratified K-fold; strata = label x cancer type),
  so model differences are paired.
* Out-of-fold (OOF) predictions are collected per repeat. Point estimates = mean over repeats.
* 95% CIs and p-values come from a stratified bootstrap over samples of the repeat-averaged OOF scores;
  differences between two models use the same resamples (paired bootstrap).
* "Within-cancer" metrics are computed inside each cancer type and averaged with sample-size weights, so
  a model cannot score well simply by recognising the tissue of origin.
"""
import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr
from sklearn.metrics import average_precision_score
from sklearn.model_selection import RepeatedStratifiedKFold

import config as C


# ---------------------------------------------------------------------------
# Folds
# ---------------------------------------------------------------------------
def make_folds(y, cancer, task, n_folds=C.N_FOLDS, n_repeats=C.N_REPEATS, seed=C.SEED):
    cancer = np.asarray(cancer).astype(str)
    key = cancer.copy() if task == "reg" else np.char.add(cancer, "|" + np.asarray(y).astype(int).astype(str))
    counts = pd.Series(key).value_counts()
    rare = np.isin(key, counts[counts < n_folds].index)
    if task == "clf":
        key[rare] = "rare|" + np.asarray(y).astype(int).astype(str)[rare]
    else:
        key[rare] = "rare"
    rskf = RepeatedStratifiedKFold(n_splits=n_folds, n_repeats=n_repeats, random_state=seed)
    return [(i // n_folds, i % n_folds, tr, te) for i, (tr, te) in enumerate(rskf.split(np.zeros(len(key)), key))]


def run_cv(factory, X, cov, y, folds, after_fit=None):
    """factory(seed) -> model. Returns OOF matrix (n_repeats, n).
    after_fit(model, rep, k, test_idx) is called after each fold (importance, gate values, logging)."""
    n_rep = max(f[0] for f in folds) + 1
    oof = np.full((n_rep, len(y)), np.nan)
    for rep, k, tr, te in folds:
        m = factory(C.SEED + 100 * rep + k).fit(X.iloc[tr], cov.iloc[tr], y[tr])
        oof[rep, te] = m.predict(X.iloc[te], cov.iloc[te])
        if after_fit:
            after_fit(m, rep, k, te)
    return oof


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def auc(y, s):
    """Mann-Whitney AUROC; nan if a class is absent."""
    y = np.asarray(y).astype(bool)
    n1, n0 = y.sum(), (~y).sum()
    if n1 == 0 or n0 == 0:
        return np.nan
    r = rankdata(s)
    return (r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def auprc(y, s):
    """Average precision with tied scores grouped into one threshold (needed for covariate-only models)."""
    y = np.asarray(y).astype(int)
    return float(average_precision_score(y, s)) if 0 < y.sum() < len(y) else np.nan


def sens_at_spec(y, s, spec=0.9):
    y = np.asarray(y).astype(bool)
    thr = np.quantile(np.asarray(s)[~y], spec)
    return float((np.asarray(s)[y] > thr).mean())


def eligible_cancers(y, cancer, task):
    cancer = np.asarray(cancer)
    out = []
    for c in np.unique(cancer):
        k = cancer == c
        if task == "clf":
            if min(y[k].sum(), (1 - y[k]).sum()) >= C.MIN_PER_CLASS_WITHIN:
                out.append(c)
        elif k.sum() >= 30:
            out.append(c)
    return out


def within(y, s, cancer, cancers, task):
    vals, w = [], []
    for c in cancers:
        k = cancer == c
        v = auc(y[k], s[k]) if task == "clf" else spearmanr(y[k], s[k]).correlation
        if np.isfinite(v):
            vals.append(v)
            w.append(k.sum())
    return float(np.average(vals, weights=w)) if vals else np.nan


def metrics(y, s, cancer, cancers, task):
    y, s, cancer = np.asarray(y), np.asarray(s), np.asarray(cancer)
    if task == "clf":
        m = {"AUROC": auc(y, s), "AUPRC": auprc(y, s), "Sens@Spec90": sens_at_spec(y, s),
             "within_AUROC": within(y, s, cancer, cancers, task)}
    else:
        m = {"Spearman": spearmanr(y, s).correlation,
             "R2": 1 - np.mean((y - s) ** 2) / np.var(y),
             "within_Spearman": within(y, s, cancer, cancers, task)}
    for c in cancers:
        k = cancer == c
        m[f"{'AUROC' if task == 'clf' else 'Spearman'}[{c}]"] = auc(y[k], s[k]) if task == "clf" else spearmanr(y[k], s[k]).correlation
    return m


def summarize(oof_by_model, y, cancer, task, reference, n_boot=C.N_BOOT, seed=C.SEED, pairs=None):
    """Point estimates, bootstrap 95% CIs, paired differences vs `reference` and between `pairs` of models.

    Note: within-cancer metrics of the covariate-only reference are biased below chance (a held-out fold with
    more positives gets a lower cross-validated group mean), so within-cancer results are judged against
    chance (0.5 AUROC / 0 rho) rather than against the reference; pooled metrics use the reference."""
    y, cancer = np.asarray(y), np.asarray(cancer)
    cancers = eligible_cancers(y, cancer, task)
    rng = np.random.default_rng(seed)
    strata = pd.Series(cancer.astype(str) + ("|" + y.astype(int).astype(str) if task == "clf" else ""))
    groups = [np.flatnonzero(strata.values == g) for g in strata.unique()]
    boots = [np.concatenate([rng.choice(g, len(g)) for g in groups]) for _ in range(n_boot)]
    keys = metric_keys(task)
    avg = {m: np.nanmean(o, axis=0) for m, o in oof_by_model.items()}
    boot_vals = {m: np.array([[metrics(y[b], avg[m][b], cancer[b], cancers, task)[k] for k in keys] for b in boots])
                 for m in oof_by_model}
    rows = []
    for m, o in oof_by_model.items():
        # point estimate on the same repeat-averaged predictions the bootstrap resamples; repeat SD separately
        row = {"model": m, **metrics(y, avg[m], cancer, cancers, task)}
        per_rep = [metrics(y, o[r], cancer, cancers, task) for r in range(o.shape[0])]
        for j, k in enumerate(keys):
            row[f"{k}_lo"], row[f"{k}_hi"] = np.nanpercentile(boot_vals[m][:, j], [2.5, 97.5])
            row[f"{k}_rep_sd"] = float(np.nanstd([p[k] for p in per_rep]))
            if reference in boot_vals and m != reference:
                row.update(_paired(boot_vals[m][:, j], boot_vals[reference][:, j], f"{k}_vs_ref"))
        rows.append(row)
    pair_rows = []
    for a, b in (pairs or []):
        if a in boot_vals and b in boot_vals:
            for j, k in enumerate(keys):
                r = _paired(boot_vals[a][:, j], boot_vals[b][:, j], "")
                pair_rows.append({"model_a": a, "model_b": b, "metric": k,
                                  "a": metrics(y, avg[a], cancer, cancers, task)[k],
                                  "b": metrics(y, avg[b], cancer, cancers, task)[k],
                                  "diff": r["d"], "lo": r["d_lo"], "hi": r["d_hi"], "p": r["p"],
                                  "lo90": r["d_lo90"], "hi90": r["d_hi90"], "equivalent": r["equivalent"]})
    return pd.DataFrame(rows), cancers, pd.DataFrame(pair_rows)


def metric_keys(task):
    return ["AUROC", "within_AUROC", "AUPRC"] if task == "clf" else ["Spearman", "within_Spearman", "R2"]


def _paired(a, b, suffix, margin=C.EQUIV_MARGIN):
    """Paired bootstrap difference a - b: mean, 95% CI, two-sided p, and a TOST equivalence verdict.

    Equivalence (two one-sided tests at alpha=0.05) holds when the 90% CI of the difference lies
    entirely inside +/- margin: the two models then perform equivalently, rather than merely
    failing to differ significantly.
    """
    d = a - b
    lo, hi = np.nanpercentile(d, [2.5, 97.5])
    lo90, hi90 = np.nanpercentile(d, [5, 95])
    p = float(min(1.0, 2 * min(np.nanmean(d <= 0), np.nanmean(d >= 0))))
    return {f"d{suffix}": float(np.nanmean(d)), f"d{suffix}_lo": lo, f"d{suffix}_hi": hi, f"p{suffix}": p,
            f"d{suffix}_lo90": lo90, f"d{suffix}_hi90": hi90,
            f"equivalent{suffix}": bool(lo90 > -margin and hi90 < margin)}


def permutation_importance(model, X, cov, y, n_perm=5, seed=C.SEED):
    """Mean drop in AUROC when one gene is permuted in held-out data."""
    rng = np.random.default_rng(seed)
    base = auc(y, model.predict(X, cov))
    out = {}
    for g in X.columns:
        drops = []
        for _ in range(n_perm):
            Xp = X.copy()
            Xp[g] = rng.permutation(Xp[g].values)
            drops.append(base - auc(y, model.predict(Xp, cov)))
        out[g] = float(np.mean(drops))
    return pd.Series(out)
