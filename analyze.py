"""Stage 2c: merge predictions from all stages and compute the statistics reported in the paper.

Inputs (results/): {task}_oof.npz (run_experiments.py), {task}_tw_oof.npz (whole transcriptome),
{task}_sig_oof.npz and {task}_sig_curve_oof.npz (signature.py). All share identical folds.

Outputs: {task}_summary.csv (point estimates + 95% CI), {task}_pairs.csv (paired differences, TOST),
alt_within_subtype.csv (glioma molecular subtype control), {task}_signature_curve_ci.csv (size curve).
Usage: python analyze.py [--boot 1000]
"""
import argparse

import numpy as np
import pandas as pd

import config as C
import evaluate as E

KIND = {"alt": "clf", "alt_noATRX": "clf", "alt_pan": "clf", "alt_pheno": "clf", "tel": "clf", "tl": "reg",
        "tel_no5p15": "clf", "tel_no5p": "clf"}
ALL_GENES = " [all genes]"                     # suffix for reference models merged from the full telomerase task
SIG = f"LR (signature, K={C.SIG_K})"
TW = "LR (whole transcriptome)"
META = {"y", "cancer", "sample", "subtype"}

PAIRS = [
    # the headline: compact signature vs curated panel vs whole transcriptome
    (SIG, "LR (panel)"), (SIG, TW), (TW, "LR (panel)"),
    (SIG, "RF (panel)"), (SIG, "GBM (panel)"), (SIG, "MLP tuned (panel)"),
    (SIG, "Elastic net (panel)"), (SIG, "Linear SVM (panel)"),
    # does model flexibility help on the panel?
    ("RF (panel)", "LR (panel)"), ("GBM (panel)", "LR (panel)"), ("MLP (panel)", "LR (panel)"),
    ("MLP tuned (panel)", "MLP (panel)"),
    ("Elastic net (panel)", "LR (panel)"), ("Linear SVM (panel)", "LR (panel)"),
    ("Elastic net (whole transcriptome)", TW), ("Linear SVM (whole transcriptome)", TW),
    # beyond glioma molecular subtype
    ("LR (panel) + glioma subtype", "Cancer + glioma subtype only"), (SIG, "Cancer + glioma subtype only"),
    # published telomerase methods
    (SIG, "Barthel 2017 score (published)"), (SIG, "LR (EXTEND gene set, TERT removed)"),
    (TW, "LR (EXTEND gene set, TERT removed)"), ("LR (panel)", "LR (Barthel 2017 genes)"),
    # TERT-neighbourhood sensitivity: same model with vs without the excluded region
    (TW, TW + ALL_GENES), (SIG, SIG + ALL_GENES),
]


def load_oof(path):
    z = np.load(path, allow_pickle=True)
    oof = {k.replace("_", " "): z[k] for k in z.files if k not in META}
    task = path.name[:-len("_oof.npz")]
    f = path.parent / f"{task}_sig_oof.npz"       # model produced by signature.py
    if f.exists():
        arr = np.load(f, allow_pickle=True)["oof"]
        if arr.shape[1] == len(z["y"]):
            oof[SIG] = arr
    f = path.parent / f"{task}_tw_oof.npz"        # one or more transcriptome-wide baselines
    if f.exists():
        wide = np.load(f, allow_pickle=True)
        if "oof" in wide.files:                   # compatibility with pre-nested historical outputs
            if wide["oof"].shape[1] == len(z["y"]):
                oof[TW] = wide["oof"]
        for key in wide.files:
            if key in {"n_genes", "source", "oof"}:
                continue
            arr = wide[key]
            if getattr(arr, "ndim", 0) == 2 and arr.shape[1] == len(z["y"]):
                oof[key.replace("_", " ")] = arr
    subtype = z["subtype"].astype(str) if "subtype" in z.files else None
    return oof, z["y"], z["cancer"].astype(str), subtype


def analyze_task(task, oof, y, cancer, n_boot=C.N_BOOT):
    kind = KIND[task]
    summary, _, pairs = E.summarize(oof, y, cancer, kind, reference="Cancer type only", n_boot=n_boot, pairs=PAIRS)
    summary.insert(1, "task", task)
    summary["n"], summary["n_pos"] = len(y), (int(np.sum(y)) if kind == "clf" else np.nan)
    timing = C.RESULTS / f"{task}_timing.csv"
    if timing.exists():
        summary["seconds"] = summary.model.map(pd.read_csv(timing, index_col=0)["seconds"])
    summary.to_csv(C.RESULTS / f"{task}_summary.csv", index=False)
    if len(pairs):
        pairs.insert(0, "task", task)
        pairs.to_csv(C.RESULTS / f"{task}_pairs.csv", index=False)
    return summary, pairs


def within_subtype(oof, y, subtype):
    """Glioma control: discrimination inside each molecular subtype with enough tumors of both classes."""
    rows = []
    for sub in np.unique(subtype):
        k = subtype == sub
        if min(y[k].sum(), (1 - y[k]).sum()) >= C.MIN_PER_CLASS_WITHIN:
            rows += [{"subtype": sub, "n": int(k.sum()), "n_pos": int(y[k].sum()), "model": name,
                      "AUROC": E.auc(y[k], np.nanmean(o, 0)[k])} for name, o in oof.items()]
    out = pd.DataFrame(rows)
    out.to_csv(C.RESULTS / "alt_within_subtype.csv", index=False)
    return out


def signature_curve(task, y, cancer, n_boot):
    f = C.RESULTS / f"{task}_sig_curve_oof.npz"
    if not f.exists():
        return None
    z = np.load(f)
    oof = {k: z[k] for k in z.files}
    s, _, _ = E.summarize(oof, y, cancer, "clf", reference=None, n_boot=n_boot)
    s["method"] = s.model.str.rsplit("_K", n=1).str[0]
    s["k"] = s.model.str.rsplit("_K", n=1).str[1].astype(int)
    s.to_csv(C.RESULTS / f"{task}_signature_curve_ci.csv", index=False)
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot", type=int, default=C.N_BOOT)
    ap.add_argument("--tasks", default=None)
    a = ap.parse_args()
    wanted = a.tasks.split(",") if a.tasks else None
    for f in sorted(C.RESULTS.glob("*_oof.npz")):
        task = f.name[:-len("_oof.npz")]
        if task not in KIND or (wanted and task not in wanted):
            continue                                  # e.g. alt_sig / alt_tw are merged into their parent task
        oof, y, cancer, subtype = load_oof(f)
        if task in C.REGION_EXCLUDE and (C.RESULTS / "tel_oof.npz").exists():
            ref, y_ref, _, _ = load_oof(C.RESULTS / "tel_oof.npz")
            assert np.array_equal(y_ref, y), "sensitivity task must share the telomerase cohort order"
            for m in (TW, SIG):
                if m in ref:
                    oof[m + ALL_GENES] = ref[m]
        s, p = analyze_task(task, oof, y, cancer, a.boot)
        k = E.metric_keys(KIND[task])
        print(f"\n== {task} (n={len(y)})")
        print(s[["model", k[0], k[1], f"{k[1]}_lo", f"{k[1]}_hi"]].round(3).to_string(index=False))
        if len(p):
            print(p[p.metric == k[1]].drop(columns=["task", "metric"]).round(3).to_string(index=False))
        if task == "alt" and subtype is not None:
            ws = within_subtype(oof, y, subtype)
            if len(ws):
                print(ws.pivot(index="model", columns="subtype", values="AUROC").round(3).to_string())
        if task in ("alt", "alt_pan", "tel") or task in C.REGION_EXCLUDE:
            curve = signature_curve(task, y, cancer, a.boot)
            if curve is not None:
                print(curve.pivot(index="k", columns="method", values="within_AUROC").round(3).to_string())


if __name__ == "__main__":
    main()
