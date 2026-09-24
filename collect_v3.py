"""Collect fetched v3 Kaggle outputs without overwriting the historical v2 results.

Run only after ``push_to_kaggle.py --fetch --only ...`` has downloaded every v3 job.
The collector refuses conflicting files unless ``--replace`` is explicitly supplied and writes a
SHA-256 manifest so manuscript inputs remain traceable.
"""
import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import config as C


JOBS = ("tmm-v3-alt", "tmm-v3-altaux", "tmm-v3-tel", "tmm-v3-telsens",
        "tmm-v3-altpan", "tmm-v3-altpan-sig", "tmm-v3-confounding", "tmm-v3-validate")
CORE_RESULTS = (
    "alt_oof.npz", "alt_tw_oof.npz", "alt_sig_oof.npz", "alt_signature.json",
    "alt_signature_genes.csv", "alt_signature_curve.csv", "alt_random_null.csv",
    "alt_noATRX_oof.npz", "alt_noATRX_tw_oof.npz",
    "tel_oof.npz", "tel_tw_oof.npz", "tel_sig_oof.npz", "tel_signature.json",
    "tel_signature_genes.csv", "tel_signature_curve.csv", "tel_random_null.csv",
    "alt_pan_oof.npz", "alt_pan_tw_oof.npz", "alt_pan_sig_oof.npz", "alt_pan_signature.json",
    "alt_pan_signature_genes.csv", "alt_pan_signature_curve.csv",
    "alt_pheno_oof.npz", "alt_pheno_tw_oof.npz", "alt_pheno_sig_oof.npz", "alt_pheno_signature.json",
    "alt_pheno_signature_genes.csv", "alt_pheno_signature_curve.csv",
    "tel_no5p15_oof.npz", "tel_no5p15_tw_oof.npz",
    "tel_no5p_oof.npz", "tel_no5p_tw_oof.npz",
    "tl_oof.npz",
    "alt_confounding_oof.npz", "alt_confounding_summary.csv", "alt_confounding_pairs.csv",
    "tel_confounding_oof.npz", "tel_confounding_summary.csv", "tel_confounding_pairs.csv",
    "validation_external.csv", "validation_alt_probability_thresholds.csv",
    "validation_platform.csv", "validation_celllines.csv",
    "validation_wu2025_celllines.csv",
    "validation_survival.csv", "validation_chromosome_arms.csv",
    "alt_summary.csv", "alt_noATRX_summary.csv", "alt_pheno_summary.csv",
    "tel_summary.csv", "alt_pan_summary.csv", "tl_summary.csv",
    "tel_no5p15_summary.csv", "tel_no5p_summary.csv",
)
CORE_FROZEN = (
    "alt_signature_locked.json", "alt_signature_locked_heldout.json",
    "tel_signature_locked.json", "tel_signature_locked_heldout.json",
    "alt_pan_signature_locked.json", "alt_pan_signature_locked_heldout.json",
    "alt_pheno_signature_locked.json",
)


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def output_dir(job_root, relative):
    direct = job_root / "tmm" / relative
    if direct.is_dir():
        return direct
    candidates = sorted(p for p in job_root.rglob(Path(relative).name)
                        if p.is_dir() and p.parent.name == "tmm")
    if len(candidates) != 1:
        raise FileNotFoundError(f"expected one {relative} directory below {job_root}; found {candidates}")
    return candidates[0]


def copy_outputs(source, destination, job, replace, manifest):
    destination.mkdir(parents=True, exist_ok=True)
    for src in sorted(source.iterdir()):
        if not src.is_file():
            continue
        digest = sha256(src)
        dst = destination / src.name
        if dst.exists():
            if sha256(dst) == digest:
                continue
            if not replace:
                raise FileExistsError(f"conflicting output {dst}; inspect it or rerun with --replace")
        shutil.copy2(src, dst)
        manifest.append({"job": job, "source": str(src), "destination": str(dst), "sha256": digest})


def require_files(root, names, kind):
    missing = [name for name in names if not (root / name).is_file()]
    if missing:
        raise FileNotFoundError(f"missing required {kind} artifacts: {', '.join(missing)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fetched", type=Path, default=C.ROOT / "kaggle_outputs")
    ap.add_argument("--results", type=Path, default=C.ROOT / "results_v3")
    ap.add_argument("--frozen", type=Path, default=C.ROOT / "data" / "frozen_v3")
    ap.add_argument("--replace", action="store_true", help="replace a conflicting collected file")
    args = ap.parse_args()
    args.results, args.frozen = args.results.resolve(), args.frozen.resolve()
    if args.results == (C.ROOT / "results").resolve() or args.frozen == (C.ROOT / "data" / "frozen").resolve():
        raise SystemExit("refusing to collect v3 into historical v2 paths")

    manifest = []
    for job in JOBS:
        job_root = args.fetched / job
        if not job_root.is_dir():
            raise FileNotFoundError(f"fetch output missing for {job}: {job_root}")
        copy_outputs(output_dir(job_root, "results"), args.results, job, args.replace, manifest)
        try:
            frozen = output_dir(job_root, "data/frozen")
        except FileNotFoundError:
            frozen = None
        if frozen:
            copy_outputs(frozen, args.frozen, job, args.replace, manifest)

    require_files(args.results, CORE_RESULTS, "result")
    require_files(args.frozen, CORE_FROZEN, "locked-signature")
    record = {"schema": 1, "collected_utc": datetime.now(timezone.utc).isoformat(),
              "jobs": list(JOBS), "results": str(args.results), "frozen": str(args.frozen),
              "files_copied": manifest}
    (args.results / "collection_manifest.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(f"collected {len(manifest)} files; manifest: {args.results / 'collection_manifest.json'}")


if __name__ == "__main__":
    main()
