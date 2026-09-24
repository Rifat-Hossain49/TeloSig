"""Karyotype-confounding analysis for transcriptomic TMM inference.

This stage asks how much of the apparent expression signal is gene dosage:

* clinical structure only;
* gene-level copy number only;
* expression only;
* expression after removing prespecified suspect chromosome regions;
* expression residualized gene-by-gene for matched copy number;
* expression plus copy number.

All residualization parameters, missing-value replacements, preprocessing and model fits are estimated
inside each outer training fold.  Every model uses the same matched tumors and outer folds.
"""
import argparse
import json
import re
import time

import numpy as np
import pandas as pd

import config as C
import evaluate as E
from data_build import load_matrix
from models import SkModel
from run_experiments import Select, excluded, load, load_transcriptome, task_frame


def gene_regions():
    h = pd.read_csv(C.RAW / "hgnc_complete_set.txt", sep="\t", usecols=["symbol", "location"],
                    low_memory=False).dropna()
    return dict(zip(h.symbol, h.location.astype(str)))


class ResidualizedExpressionModel:
    """Remove the training-estimated linear CNV dosage component from each expression gene."""
    def __init__(self, genes, seed):
        self.genes, self.seed = list(genes), seed

    def _residuals(self, expression, cnv):
        x = expression[self.genes].values.astype(np.float32)
        c = cnv[self.genes].values.astype(np.float32)
        c = np.where(np.isfinite(c), c, self.cnv_mean_)
        expected = self.expr_mean_ + (c - self.cnv_mean_) * self.slope_
        return pd.DataFrame(np.nan_to_num(x - expected), columns=self.genes, index=expression.index)

    def fit(self, expression, cnv, cov, y):
        x = expression[self.genes].values.astype(np.float32)
        c = cnv[self.genes].values.astype(np.float32)
        self.cnv_mean_ = np.nanmean(c, axis=0)
        c = np.where(np.isfinite(c), c, self.cnv_mean_)
        self.expr_mean_ = np.nanmean(x, axis=0)
        xc, cc = x - self.expr_mean_, c - self.cnv_mean_
        self.slope_ = np.nansum(xc * cc, axis=0) / (np.sum(cc * cc, axis=0) + 1e-8)
        residual = self._residuals(expression, cnv)
        self.model = SkModel("lr_wide", "clf", self.seed).fit(residual, cov, y)
        return self

    def predict(self, expression, cnv, cov):
        return self.model.predict(self._residuals(expression, cnv), cov)


class JointModel:
    def __init__(self, genes, seed):
        self.genes, self.seed = list(genes), seed

    def _design(self, expression, cnv):
        a = expression[self.genes].copy()
        b = cnv[self.genes].copy()
        a.columns = [f"expr::{g}" for g in self.genes]
        b.columns = [f"cnv::{g}" for g in self.genes]
        return pd.concat([a, b], axis=1)

    def fit(self, expression, cnv, cov, y):
        self.model = SkModel("lr_wide", "clf", self.seed).fit(self._design(expression, cnv), cov, y)
        return self

    def predict(self, expression, cnv, cov):
        return self.model.predict(self._design(expression, cnv), cov)


def run_multimodal(factory, expression, cnv, cov, y, folds):
    n_repeats = max(f[0] for f in folds) + 1
    out = np.full((n_repeats, len(y)), np.nan)
    for repeat, fold, train, test in folds:
        model = factory(C.SEED + 100 * repeat + fold)
        model.fit(expression.iloc[train], cnv.iloc[train], cov.iloc[train], y[train])
        out[repeat, test] = model.predict(expression.iloc[test], cnv.iloc[test], cov.iloc[test])
    return out


def matched_task(task, quick=False):
    lab, panel, _, ext = load()
    d, _, cov, y = task_frame(task, lab, panel, ext=ext)
    expression, source = load_transcriptome(d["sample"].tolist())
    cnv = load_matrix("cnv")
    keep = np.array([s in cnv.index for s in d["sample"]])
    d, cov, y = d.loc[keep].reset_index(drop=True), cov.loc[keep].reset_index(drop=True), y[keep]
    expression = expression.loc[d["sample"]].reset_index(drop=True)
    cnv = cnv.loc[d["sample"]].reset_index(drop=True)
    # Expression respects the task's anti-circularity exclusions. CNV deliberately starts from the broader
    # matched universe so the diagnostic "CNV only" model can be contrasted with a prespecified version that
    # removes label-associated loci. Residualised and joint models use the common, non-circular expression set.
    expr_genes = [g for g in expression.columns if g in cnv.columns and g not in excluded(task)]
    cnv_genes = [g for g in cnv.columns if g in expression.columns]
    cnv_ok = np.isfinite(cnv[cnv_genes].values).any(axis=0)
    cnv_genes = [g for g, ok in zip(cnv_genes, cnv_ok) if ok]
    common = [g for g in expr_genes if g in set(cnv_genes)]
    if quick:
        common = common[:min(500, len(common))]
        cnv_genes = list(dict.fromkeys(common + [g for g in ("ATRX", "DAXX", "TERT") if g in cnv_genes]))
    return d, expression[common], cnv[cnv_genes], cov, y, common, cnv_genes, source


def run_task(task, args):
    d, expression, cnv, cov, y, genes, cnv_genes, source = matched_task(task, args.quick)
    folds = E.make_folds(y, cov.cancer, "clf", args.folds, args.repeats)
    location = gene_regions()
    region_sets = C.CONFOUNDING_REGION_SETS[task]
    label_loci = {"alt": {"ATRX", "DAXX", "TERT"}, "tel": {"TERT"}}[task]
    cnv_no_label = [g for g in cnv_genes if g not in label_loci]
    base_cov = ["cancer"]
    oof = {}
    t0 = time.time()
    oof["Cancer type only"] = E.run_cv(
        lambda s: Select(SkModel("cov_only", "clf", s, use_genes=False), genes, base_cov),
        expression, cov, y, folds)
    if task == "alt":
        oof["Cancer + molecular subtype only"] = E.run_cv(
            lambda s: Select(SkModel("cov_only", "clf", s, use_genes=False), genes, ["subtype"]),
            expression, cov, y, folds)
    oof["Expression only"] = E.run_cv(
        lambda s: Select(SkModel("lr_wide", "clf", s), genes, base_cov), expression, cov, y, folds)
    removed_counts = {}
    for region_name, prefixes in region_sets.items():
        no_region = [g for g in genes if not any(location.get(g, "").startswith(p) for p in prefixes)]
        removed_counts[region_name] = len(genes) - len(no_region)
        oof[f"Expression excluding {region_name}"] = E.run_cv(
            lambda s, selected=no_region: Select(SkModel("lr_wide", "clf", s), selected, base_cov),
            expression, cov, y, folds)
    oof["CNV only"] = E.run_cv(
        lambda s: Select(SkModel("lr_wide", "clf", s), cnv_genes, base_cov), cnv, cov, y, folds)
    oof["CNV only, label loci excluded"] = E.run_cv(
        lambda s: Select(SkModel("lr_wide", "clf", s), cnv_no_label, base_cov), cnv, cov, y, folds)
    oof["Expression residualized for CNV"] = run_multimodal(
        lambda s: ResidualizedExpressionModel(genes, s), expression, cnv, cov, y, folds)
    oof["Expression + CNV"] = run_multimodal(
        lambda s: JointModel(genes, s), expression, cnv, cov, y, folds)

    safe = {re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_"): values for name, values in oof.items()}
    np.savez_compressed(C.RESULTS / f"{task}_confounding_oof.npz", y=y, cancer=cov.cancer.values,
                        sample=d["sample"].values, **safe)
    pairs = [(name, "Expression only") for name in oof if name not in ("Cancer type only", "Expression only")]
    summary, _, comparisons = E.summarize(oof, y, cov.cancer.values, "clf", "Cancer type only",
                                           n_boot=args.boot, pairs=pairs)
    summary.insert(0, "task", task)
    comparisons.insert(0, "task", task)
    summary.to_csv(C.RESULTS / f"{task}_confounding_summary.csv", index=False)
    comparisons.to_csv(C.RESULTS / f"{task}_confounding_pairs.csv", index=False)
    metadata = {"task": task, "n": len(y), "n_positive": int(y.sum()), "n_expression_genes": len(genes),
                "n_cnv_genes": len(cnv_genes),
                "expression_source": source, "region_sets": region_sets,
                "n_region_genes_removed": removed_counts,
                "outer_folds": args.folds, "outer_repeats": args.repeats,
                "residualization": "gene-wise slope/intercept fit on each outer training fold only",
                "minutes": (time.time() - t0) / 60}
    json.dump(metadata, open(C.RESULTS / f"{task}_confounding_run.json", "w"), indent=1)
    print(summary[["model", "within_AUROC", "within_AUROC_lo", "within_AUROC_hi"]]
          .round(3).to_string(index=False), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default="alt,tel")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--boot", type=int, default=C.N_BOOT)
    args = ap.parse_args()
    args.folds, args.repeats = (2, 1) if args.quick else (C.N_FOLDS, C.N_REPEATS)
    if args.quick:
        args.boot = min(args.boot, 50)
    for task in args.tasks.split(","):
        run_task(task, args)


if __name__ == "__main__":
    main()
