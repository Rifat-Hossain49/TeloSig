"""Read-only acceptance audit for the deferred v3 run."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import config as C
from collect_v3 import CORE_FROZEN, CORE_RESULTS


def add(checks, name, passed, detail):
    checks.append({"check": name, "passed": bool(passed), "detail": str(detail)})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, default=C.RESULTS)
    ap.add_argument("--frozen", type=Path, default=C.FROZEN)
    ap.add_argument(
        "--paper",
        type=Path,
        default=None,
        help="Optional manuscript path; when supplied, fail if it is missing or contains \\TBD markers.",
    )
    args = ap.parse_args()
    checks = []

    missing_results = [f for f in CORE_RESULTS if not (args.results / f).is_file()]
    missing_frozen = [f for f in CORE_FROZEN if not (args.frozen / f).is_file()]
    add(checks, "core result artifacts", not missing_results, missing_results or "complete")
    add(checks, "locked artifacts", not missing_frozen, missing_frozen or "complete")
    add(checks, "collection manifest", (args.results / "collection_manifest.json").is_file(),
        args.results / "collection_manifest.json")

    for task in ("alt", "alt_noATRX", "tel", "tel_no5p15", "tel_no5p", "alt_pan", "alt_pheno", "tl"):
        p = args.results / f"{task}_oof.npz"
        if not p.is_file():
            continue
        z = np.load(p, allow_pickle=True)
        n = len(z["y"])
        arrays = [k for k in z.files if k not in ("y", "cancer", "sample", "subtype")]
        finite = all(z[k].shape[-1] == n and np.isfinite(z[k]).all() for k in arrays)
        add(checks, f"{task} OOF dimensions/finite", finite, f"n={n}; arrays={len(arrays)}")

    for task in ("alt", "tel", "alt_pan", "alt_pheno"):
        p = args.frozen / f"{task}_signature_locked.json"
        if not p.is_file():
            continue
        lock = json.loads(p.read_text(encoding="utf-8"))
        n = len(lock.get("genes", []))
        lengths = [n, len(lock.get("weights", [])), len(lock.get("mean", [])), len(lock.get("sd", []))]
        deployment = lock.get("deployment", {})
        valid = len(set(lengths)) == 1 and n == lock.get("k") and "threshold" in deployment
        add(checks, f"{task} locked model deployable", valid, f"vector lengths={lengths}")

    wu_path = args.results / "validation_wu2025_celllines.csv"
    if wu_path.is_file():
        wu = pd.read_csv(wu_path)
        required = {"source", "task", "outcome", "model", "scope", "metric", "estimate", "lo", "hi", "n"}
        role = wu["analysis_role"] if "analysis_role" in wu else pd.Series("", index=wu.index)
        scope = wu["scope"] if "scope" in wu else pd.Series("", index=wu.index)
        primary = wu[(role == "primary") & (scope == "all")]
        # The fixed 2022 Cell Model Passports release omits a few locked transcripts.  The scores in the
        # collected run were computed after subsetting the coefficient vector, which is algebraically identical
        # to assigning absent transcripts their locked training means (zero standardised contribution).  Partial
        # observed coverage is therefore acceptable only when it is explicit, small, and every primary row has
        # at least 90% of its expected features.  Code-level regression tests enforce that equivalence.
        coverage_ok = ("signature_genes_present" in primary and "signature_genes_expected" in primary and
                       (primary.signature_genes_present <= primary.signature_genes_expected).all() and
                       (primary.signature_genes_present >= .9 * primary.signature_genes_expected).all())
        valid = (required <= set(wu) and {"alt", "tel"} <= set(primary.task) and
                 primary.estimate.notna().all() and coverage_ok)
        coverage = sorted(set(zip(primary.signature_genes_present.astype(int),
                                  primary.signature_genes_expected.astype(int)))) if coverage_ok else "invalid"
        add(checks, "Wu 2025 direct-assay validation", valid,
            f"rows={len(wu)}; primary tasks={sorted(set(primary.task))}; observed/expected={coverage}; "
            "absent terms use locked training means")

    for task in ("alt", "tel"):
        p = args.results / f"{task}_confounding_summary.csv"
        if not p.is_file():
            continue
        models = set(pd.read_csv(p)["model"])
        wanted = {"Expression only", "CNV only", "CNV only, label loci excluded",
                  "Expression residualized for CNV", "Expression + CNV"}
        if task == "alt":
            wanted |= {"Cancer + molecular subtype only", "Expression excluding 1p/17p",
                       "Expression excluding ATRX/DAXX cytobands"}
        else:
            wanted.add("Expression excluding 5p15")
        add(checks, f"{task} confounding contrasts", wanted <= models, sorted(wanted - models) or "complete")

    if args.paper is not None:
        if not args.paper.is_file():
            add(checks, "manuscript available", False, f"not found: {args.paper}")
        else:
            paper_text = args.paper.read_text(encoding="utf-8")
            add(checks, "manuscript placeholders resolved", "\\TBD{" not in paper_text,
                "TBD markers remain" if "\\TBD{" in paper_text else "complete")

    table = pd.DataFrame(checks)
    print(table.to_string(index=False))
    failed = table[~table.passed]
    raise SystemExit(1 if len(failed) else 0)


if __name__ == "__main__":
    main()
