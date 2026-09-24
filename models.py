"""Model zoo with one interface: model.fit(X, cov, y) -> self ; model.predict(X, cov) -> scores.

X   : (n, genes) DataFrame of log2 expression (raw; each model standardises on its own training data)
cov : (n, k) DataFrame of categorical covariates (cancer type, optionally glioma subtype)
y   : binary labels (classification) or continuous values (regression)
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.ensemble import (HistGradientBoostingClassifier, HistGradientBoostingRegressor,
                              RandomForestClassifier, RandomForestRegressor)
from sklearn.linear_model import (ElasticNetCV, LogisticRegression, LogisticRegressionCV, RidgeCV,
                                  SGDClassifier)
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import LinearSVC

import config as C

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Shared preprocessing: z-score genes, one-hot covariates (fit on training data only)
# ---------------------------------------------------------------------------
class Prep:
    def fit(self, X, cov):
        self.genes = list(X.columns)
        self.sc = StandardScaler().fit(X.values)
        self.oh = OneHotEncoder(handle_unknown="ignore", sparse_output=False).fit(cov.astype(str).values) \
            if cov is not None and cov.shape[1] else None
        return self

    def genes_z(self, X):
        return np.nan_to_num(self.sc.transform(X[self.genes].values)).astype(np.float32)

    def cov_1h(self, cov, n):
        return self.oh.transform(cov.astype(str).values).astype(np.float32) if self.oh else np.zeros((n, 0), np.float32)

    def tabular(self, X, cov):
        return np.hstack([self.genes_z(X), self.cov_1h(cov, len(X))])


# ---------------------------------------------------------------------------
# Tabular (scikit-learn) models
# ---------------------------------------------------------------------------
class SkModel:
    """kind: lr | lr_fixed | lr_wide | elasticnet | elasticnet_wide | svm | rf | gbm | cov_only."""
    def __init__(self, kind, task, seed=C.SEED, use_genes=True):
        self.kind, self.task, self.seed, self.use_genes = kind, task, seed, use_genes

    def _make(self):
        s, clf = self.seed, self.task == "clf"
        if self.kind == "lr":
            return (LogisticRegressionCV(Cs=np.logspace(-3, 1, 9), cv=3, scoring="roc_auc", class_weight="balanced",
                                         max_iter=5000, random_state=s)
                    if clf else RidgeCV(alphas=np.logspace(-2, 4, 25)))
        if self.kind == "lr_fixed":
            return (LogisticRegression(C=1.0, class_weight="balanced", max_iter=5000, random_state=s)
                    if clf else RidgeCV(alphas=[1.0]))
        if self.kind == "lr_wide":   # whole transcriptome: coarser grid, looser tolerance, parallel inner folds
            return (LogisticRegressionCV(Cs=np.logspace(-4, -1, 4), cv=3, scoring="roc_auc", class_weight="balanced",
                                         max_iter=1000, tol=1e-3, n_jobs=-1, random_state=s)
                    if clf else RidgeCV(alphas=np.logspace(0, 5, 11)))
        if self.kind == "elasticnet":
            if clf:
                return LogisticRegressionCV(
                    Cs=np.logspace(-3, 0, 5), cv=3, scoring="roc_auc", class_weight="balanced",
                    penalty="elasticnet", solver="saga", l1_ratios=[0.1, 0.5, 0.9],
                    max_iter=3000, tol=1e-3, n_jobs=-1, random_state=s)
            return ElasticNetCV(l1_ratio=[0.1, 0.5, 0.9], alphas=np.logspace(-4, 1, 12), cv=3,
                                max_iter=5000, n_jobs=-1, random_state=s)
        if self.kind == "elasticnet_wide":
            if not clf:
                raise ValueError("wide elastic net is only configured for classification")
            # SAGA over 19k genes made one outer fold take hours.  This remains a genuinely nested
            # elastic-net logistic model, but uses the standard stochastic log-loss optimiser and tunes
            # both regularisation strength and mixing inside each outer training set.
            return GridSearchCV(
                SGDClassifier(loss="log_loss", penalty="elasticnet", class_weight="balanced",
                              max_iter=2000, tol=1e-3, early_stopping=True,
                              validation_fraction=0.15, n_iter_no_change=10,
                              random_state=s),
                # Three prespecified sparse-to-dense regimes keep the nested search feasible in
                # Kaggle's 12-hour kernel window while still selecting both penalty parameters.
                [{"alpha": [1e-5], "l1_ratio": [0.85]},
                 {"alpha": [1e-4], "l1_ratio": [0.5]},
                 {"alpha": [1e-3], "l1_ratio": [0.15]}],
                scoring="roc_auc", cv=3, n_jobs=-1)
        if self.kind == "svm":
            if not clf:
                raise ValueError("linear SVM is only configured for classification")
            return GridSearchCV(
                LinearSVC(class_weight="balanced", dual="auto", max_iter=10000, random_state=s),
                {"C": np.logspace(-3, 1, 5)}, scoring="roc_auc", cv=3, n_jobs=-1)
        if self.kind == "rf":
            return (RandomForestClassifier(500, min_samples_leaf=3, max_features="sqrt",
                                           class_weight="balanced_subsample", n_jobs=-1, random_state=s)
                    if clf else RandomForestRegressor(500, min_samples_leaf=5, max_features="sqrt", n_jobs=-1, random_state=s))
        if self.kind == "gbm":
            kw = dict(max_iter=400, learning_rate=0.04, max_leaf_nodes=15, min_samples_leaf=20,
                      l2_regularization=1.0, early_stopping=True, validation_fraction=0.15,
                      n_iter_no_change=30, random_state=s)
            return HistGradientBoostingClassifier(class_weight="balanced", **kw) if clf else HistGradientBoostingRegressor(**kw)
        if self.kind == "cov_only":
            return LogisticRegression(C=1.0, max_iter=5000) if clf else RidgeCV(alphas=np.logspace(-3, 3, 13))
        raise ValueError(self.kind)

    def _design(self, X, cov):
        return self.prep.tabular(X, cov) if self.use_genes else self.prep.cov_1h(cov, len(X))

    def fit(self, X, cov, y):
        self.prep = Prep().fit(X, cov)
        self.m = self._make().fit(self._design(X, cov), y)
        return self

    def predict(self, X, cov):
        Z = self._design(X, cov)
        if self.task != "clf":
            return self.m.predict(Z)
        if hasattr(self.m, "predict_proba"):
            return self.m.predict_proba(Z)[:, 1]
        return self.m.decision_function(Z)


def transfer(model, X_target):
    """Deprecated guard against target-cohort normalisation.

    A locked model must retain preprocessing estimated from its training cohort.  Re-fitting a scaler on
    target samples makes predictions cohort-dependent and leaks the target distribution into deployment.
    """
    raise RuntimeError("target-cohort re-standardisation is forbidden; use the fitted training scaler")


class ScoreModel:
    """A fixed, published score used as-is (no training), e.g. Barthel et al.'s telomerase signature."""
    def __init__(self, column):
        self.column = column

    def fit(self, X, cov, y):
        return self

    def predict(self, X, cov):
        s = X[self.column].astype(float)
        return s.fillna(s.median()).values   # a few tumors lack the published score


# ---------------------------------------------------------------------------
# Multilayer perceptron
# ---------------------------------------------------------------------------
class MLP(nn.Module):
    def __init__(self, n_genes, n_cov, p):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(n_genes, p["flat_hidden"]), nn.ReLU(), nn.Dropout(p["dropout"]),
                                  nn.Linear(p["flat_hidden"], p["hidden"]), nn.ReLU())
        self.head = nn.Sequential(nn.Linear(p["hidden"] + n_cov, p["hidden"]), nn.ReLU(), nn.Dropout(p["dropout"]),
                                  nn.Linear(p["hidden"], 1))

    def forward(self, xg, xc):
        return self.head(torch.cat([self.body(xg), xc], 1)).squeeze(1)


def sample_nn_params(rng):
    return {k: (v[rng.integers(len(v))] if not isinstance(v[0], float) else float(v[rng.integers(len(v))]))
            for k, v in C.NN_SEARCH["space"].items()}


class TorchModel:
    """MLP trained with AdamW, class-weighted loss and early stopping on an inner validation split."""
    def __init__(self, arch="mlp", task="clf", seed=C.SEED, params=None):
        assert arch == "mlp", "only the MLP remains in the pipeline"
        self.task, self.seed = task, seed
        self.p = dict(C.NN, **(params or {}))

    def _tensors(self, X, cov):
        return (torch.tensor(self.prep.genes_z(X), device=DEVICE),
                torch.tensor(self.prep.cov_1h(cov, len(X)), device=DEVICE))

    def fit(self, X, cov, y):
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        np.random.seed(self.seed)
        y = np.asarray(y, dtype=np.float32)
        self.prep = Prep().fit(X, cov)
        if self.task == "reg":
            self.y_mu, self.y_sd = float(y.mean()), float(y.std() + 1e-8)
            y = (y - self.y_mu) / self.y_sd
        strat = y if self.task == "clf" else None
        tr, va = train_test_split(np.arange(len(y)), test_size=self.p["val_frac"], stratify=strat, random_state=self.seed)
        xg, xc = self._tensors(X, cov)
        yt = torch.tensor(y, device=DEVICE)
        self.net = MLP(xg.shape[1], xc.shape[1], self.p).to(DEVICE)
        opt = torch.optim.AdamW(self.net.parameters(), lr=self.p["lr"], weight_decay=self.p["weight_decay"])
        if self.task == "clf":
            pw = torch.tensor((1 - y[tr].mean()) / max(y[tr].mean(), 1e-6), device=DEVICE)
            loss_fn = lambda out, t: F.binary_cross_entropy_with_logits(out, t, pos_weight=pw)
        else:
            loss_fn = F.mse_loss
        best, best_state, bad = np.inf, None, 0
        bs = self.p["batch_size"]
        for epoch in range(self.p["max_epochs"]):
            self.net.train()
            perm = np.random.permutation(tr)
            for i in range(0, len(perm), bs):
                b = torch.tensor(perm[i:i + bs], device=DEVICE)
                if len(b) < 2:
                    continue
                opt.zero_grad()
                loss_fn(self.net(xg[b], xc[b]), yt[b]).backward()
                opt.step()
            self.net.eval()
            with torch.no_grad():
                vl = float(loss_fn(self.net(xg[va], xc[va]), yt[va]))
            if vl < best - 1e-4:
                best, bad = vl, 0
                best_state = {k: v.detach().clone() for k, v in self.net.state_dict().items()}
            else:
                bad += 1
                if bad >= self.p["patience"]:
                    break
        self.net.load_state_dict(best_state)
        self.epochs_run = epoch + 1
        return self

    @torch.no_grad()
    def predict(self, X, cov):
        self.net.eval()
        xg, xc = self._tensors(X, cov)
        out = torch.cat([self.net(xg[i:i + 4096], xc[i:i + 4096]) for i in range(0, len(xg), 4096)]).cpu().numpy()
        return 1 / (1 + np.exp(-out)) if self.task == "clf" else out * self.y_sd + self.y_mu


def _nested_holdout(y, cov, seed, test_size=0.25):
    """A label-by-cancer stratified split wholly inside one outer training set."""
    y = np.asarray(y)
    is_binary = set(np.unique(y)) <= {0, 1}
    if cov is not None and len(cov.columns):
        label = y.astype(int).astype(str)
        strata = cov.iloc[:, 0].astype(str).values + "|" + label if is_binary else cov.iloc[:, 0].astype(str).values
        counts = pd.Series(strata).value_counts()
        # A 25% holdout needs at least four observations per retained stratum. Pool smaller cancer strata by
        # label, then fall back to label-only stratification if the pooled split is still infeasible.
        minimum = max(2, int(np.ceil(1 / test_size)))
        strata = np.where(pd.Series(strata).map(counts).values >= minimum, strata,
                          np.char.add("rare|", label) if is_binary else "rare")
    else:
        strata = y.astype(int) if is_binary else None
    try:
        return train_test_split(np.arange(len(y)), test_size=test_size, stratify=strata, random_state=seed)
    except ValueError:
        fallback = y.astype(int) if is_binary and np.min(np.bincount(y.astype(int))) >= 2 else None
        return train_test_split(np.arange(len(y)), test_size=test_size, stratify=fallback, random_state=seed)


def _validation_score(y, score, cov, task):
    if task != "clf":
        return -float(np.mean((np.asarray(y) - np.asarray(score)) ** 2))
    y, score = np.asarray(y), np.asarray(score)
    vals, weights = [], []
    if cov is not None and len(cov.columns):
        cancer = cov.iloc[:, 0].astype(str).values
        for c in np.unique(cancer):
            k = cancer == c
            if len(np.unique(y[k])) == 2:
                vals.append(roc_auc_score(y[k], score[k]))
                weights.append(k.sum())
    return float(np.average(vals, weights=weights)) if vals else float(roc_auc_score(y, score))


class TunedTorchModel:
    """MLP whose random search is repeated wholly inside each outer training set.

    ``fit`` is called separately by the outer-CV runner.  Candidate selection therefore cannot be reused
    across outer folds or repeats, and the final model is refitted on the complete outer training set.
    """
    def __init__(self, task="clf", seed=C.SEED, n_iter=None, overrides=None):
        self.task, self.seed = task, seed
        self.n_iter = n_iter or C.NN_SEARCH["n_iter"]
        self.overrides = overrides or {}

    def fit(self, X, cov, y):
        y = np.asarray(y)
        tr, va = _nested_holdout(y, cov, self.seed)
        rng = np.random.default_rng(self.seed)
        best, best_score = None, -np.inf
        for i in range(self.n_iter):
            params = dict(sample_nn_params(rng), **self.overrides)
            candidate = TorchModel("mlp", self.task, seed=self.seed + i, params=params)
            candidate.fit(X.iloc[tr], cov.iloc[tr], y[tr])
            score = candidate.predict(X.iloc[va], cov.iloc[va])
            value = _validation_score(y[va], score, cov.iloc[va], self.task)
            if value > best_score:
                best, best_score = params, value
        self.best_params_, self.inner_score_ = best, float(best_score)
        self.model = TorchModel("mlp", self.task, seed=self.seed, params=best).fit(X, cov, y)
        return self

    def predict(self, X, cov):
        return self.model.predict(X, cov)
