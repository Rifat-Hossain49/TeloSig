# Reproducing and auditing TeloSig v3

The repository contains the corrected v3 source code, compact collected outputs, locked model artifacts, and
rendered figures. Raw and processed omics matrices are intentionally excluded because they are large and can be
rebuilt from the public sources fixed in `config.py`.

## Environment

Use a recent 64-bit Python environment and install the declared dependencies:

```bash
python -m pip install -r requirements.txt
```

GPU acceleration is optional for the nested MLP candidates. Classical models and release audits run on CPU.

## Verify the released code and artifacts

From the repository root:

```bash
python -m pytest tests -q
python audit_v3.py --results results_v3 --frozen data/frozen_v3
```

To audit a local manuscript for unresolved `\\TBD{...}` markers as well, append
`--paper path/to/main.tex` to the second command.

The collector records SHA-256 hashes in `results_v3/collection_manifest.json`. Paths in the public manifest are
repository-relative; the hashes are unchanged from the accepted run.

## Rebuild figures from the released results

PowerShell:

```powershell
$env:TMM_RESULTS_DIR = 'results_v3'
$env:TMM_FROZEN_DIR = 'data/frozen_v3'
$env:TMM_FIGURES_DIR = 'figures_v3'
python analyze.py
python make_figures.py
```

On POSIX shells, set the same three variables with `export`.

## Re-run the full analysis

The complete workflow, including data acquisition, local stages, Kaggle sharding, and collection, is documented
in `README.md`. Never commit `.env`, raw or processed datasets, Kaggle staging directories, or downloaded kernel
outputs. The v3 collector refuses the historical `results/` and `data/frozen/` destinations to prevent accidental
mixing of analysis versions.
