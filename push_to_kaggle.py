"""Upload code + processed data as a private Kaggle dataset and launch private CPU kernels.

Requires KAGGLE_API_TOKEN (or ~/.kaggle credentials) in the environment.
Usage:  python push_to_kaggle.py                      # create/update dataset, push the benchmark jobs (J1-J3)
        python push_to_kaggle.py --only tmm-v2-validate --kernels-only   # push the validation job (after J1-J3)
        python push_to_kaggle.py --status | --fetch
"""
import argparse
import json
import shutil
import subprocess
import sys
import time

import config as C

USER = "rifathosain"
DATASET = f"{USER}/tmm-telomere-pipeline"
# slug -> (shell steps run inside /kaggle/working/tmm, kernel sources whose outputs are merged first)
JOBS = {
    "tmm-v2-alt": (["python -u data_build.py --transcriptome-tcga",
                    "python -u run_experiments.py --tasks alt,alt_noATRX,alt_pheno,tl",
                    "python -u signature.py --tasks alt"], []),
    "tmm-v2-tel": (["python -u data_build.py --transcriptome-tcga",
                    "python -u run_experiments.py --tasks tel",
                    "python -u signature.py --tasks tel"], []),
    "tmm-v2-altpan": (["python -u data_build.py --transcriptome-tcga",
                       "python -u run_experiments.py --tasks alt_pan"], []),
    "tmm-v2-validate": (["python -u data_build.py --transcriptome",
                         "python -u signature.py --tasks alt_pan",
                         "python -u validate.py",
                         "python -u analyze.py"],
                        [f"{USER}/tmm-v2-alt", f"{USER}/tmm-v2-tel", f"{USER}/tmm-v2-altpan"]),
}
JOBS.update({
    "tmm-v2-tel-no5p15": (["python -u data_build.py --transcriptome-tcga",
                           "python -u run_experiments.py --tasks tel_no5p15",
                           "python -u signature.py --tasks tel_no5p15"], []),
    "tmm-v2-tel-no5p": (["python -u data_build.py --transcriptome-tcga",
                         "python -u run_experiments.py --tasks tel_no5p",
                         "python -u signature.py --tasks tel_no5p"], []),
})
DEFAULT_JOBS = ["tmm-v2-alt", "tmm-v2-tel", "tmm-v2-altpan"]
CODE = ["config.py", "data_build.py", "models.py", "evaluate.py", "analyze.py", "run_experiments.py",
        "signature.py", "validate.py", "make_figures.py"]
SKIP_DATA = {"expr_expanded.csv.gz", "ccle_expr.csv.gz"}          # obsolete / superseded inputs
STAGE = C.ROOT / "kaggle_stage"
OUT = C.ROOT / "kaggle_outputs"

RUNNER = '''\
import glob, gzip, os, shutil, subprocess, sys
STEPS = {steps}
root = "/kaggle/working/tmm"
for d in ("data/processed", "data/frozen", "data/raw", "results"):
    os.makedirs(os.path.join(root, d), exist_ok=True)
src = os.path.dirname(glob.glob("/kaggle/input/**/labels.csv", recursive=True)[0])
print("dataset files:", sorted(os.listdir(src)), flush=True)
GZ = {{"expr_panel.csv", "expr_random.csv", "ccle_transcriptome.csv"}}   # Kaggle decompresses uploaded .gz
for f in os.listdir(src):
    p = os.path.join(src, f)
    if f.endswith(".py"):
        shutil.copy(p, root)
    elif f in GZ:
        with open(p, "rb") as fi, gzip.open(os.path.join(root, "data", "processed", f + ".gz"), "wb") as fo:
            shutil.copyfileobj(fi, fo)
    else:
        shutil.copy(p, os.path.join(root, "data", "processed"))
# outputs of earlier jobs (results + locked signatures)
for f in glob.glob("/kaggle/input/**/tmm/results/*", recursive=True):
    shutil.copy(f, os.path.join(root, "results"))
for f in glob.glob("/kaggle/input/**/tmm/data/frozen/*", recursive=True):
    shutil.copy(f, os.path.join(root, "data", "frozen"))
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "git+https://github.com/ucscXena/xenaPython"], check=True)
env = dict(os.environ, TMM_ROOT=root, PYTHONWARNINGS="ignore")
try:
    for step in STEPS:
        print("\\n########", step, flush=True)
        subprocess.run(step.split(), cwd=root, env=env, check=True)
finally:                                                  # keep the kernel output small
    shutil.rmtree(os.path.join(root, "data", "processed"), ignore_errors=True)
    shutil.rmtree(os.path.join(root, "data", "raw"), ignore_errors=True)
'''


def kaggle(*args, check=True):
    cmd = [sys.executable, "-m", "kaggle", *args]
    print("$ kaggle", " ".join(args))
    r = subprocess.run(cmd, capture_output=True, text=True)
    out = "\n".join(l for l in (r.stdout + r.stderr).splitlines() if "%|" not in l)
    print(out.strip())
    if check and r.returncode:
        raise SystemExit(r.returncode)
    return r


def push_dataset():
    d = STAGE / "dataset"
    shutil.rmtree(d, ignore_errors=True)
    d.mkdir(parents=True)
    for f in CODE:
        shutil.copy(C.ROOT / f, d)
    for f in C.PROC.iterdir():
        if f.is_file() and f.name not in SKIP_DATA and not f.name.endswith(".npy") and not f.name.endswith(".txt"):
            shutil.copy(f, d)
    json.dump({"title": "TMM telomere pipeline", "id": DATASET, "licenses": [{"name": "CC0-1.0"}]},
              open(d / "dataset-metadata.json", "w"), indent=1)
    exists = kaggle("datasets", "status", DATASET, check=False).returncode == 0
    if exists:
        kaggle("datasets", "version", "-p", str(d), "-m", time.strftime("update %Y-%m-%d %H:%M"))
    else:
        kaggle("datasets", "create", "-p", str(d))   # private unless --public is given
    time.sleep(60)                                    # status reports "ready" before a new version is processed
    for _ in range(90):
        if "ready" in kaggle("datasets", "status", DATASET, check=False).stdout.lower():
            return
        time.sleep(10)
    raise SystemExit("dataset did not become ready in 15 minutes")


def push_kernels(only=None):
    for slug in (only or DEFAULT_JOBS):
        steps, sources = JOBS[slug]
        k = STAGE / slug
        shutil.rmtree(k, ignore_errors=True)
        k.mkdir(parents=True)
        (k / "run.py").write_text(RUNNER.format(steps=json.dumps(steps)))
        json.dump({"id": f"{USER}/{slug}", "title": slug, "code_file": "run.py", "language": "python",
                   "kernel_type": "script", "is_private": True, "enable_gpu": False, "enable_internet": True,
                   "dataset_sources": [DATASET], "competition_sources": [], "kernel_sources": sources},
                  open(k / "kernel-metadata.json", "w"), indent=1)
        kaggle("kernels", "push", "-p", str(k))


def status(only=None):
    for slug in (only or JOBS):
        kaggle("kernels", "status", f"{USER}/{slug}", check=False)


def fetch(only=None):
    for slug in (only or JOBS):
        dest = OUT / slug
        dest.mkdir(parents=True, exist_ok=True)
        kaggle("kernels", "output", f"{USER}/{slug}", "-p", str(dest), check=False)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--fetch", action="store_true")
    ap.add_argument("--kernels-only", action="store_true")
    ap.add_argument("--only", nargs="*", help="restrict to these kernel slugs")
    ap.add_argument("--dataset-only", action="store_true", dest="dataset_only", help="upload data/code, push no kernels")
    a = ap.parse_args()
    if a.status:
        status(a.only)
    elif a.fetch:
        fetch(a.only)
    else:
        if not a.kernels_only:
            push_dataset()
        if not a.dataset_only:
            push_kernels(a.only)
