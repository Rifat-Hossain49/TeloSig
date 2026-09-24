"""Regression tests for the leakage controls that define the v3 study design."""
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import signature  # noqa: E402
import evaluate  # noqa: E402
import run_experiments  # noqa: E402
from models import SkModel, _nested_holdout, transfer  # noqa: E402


class LeakageGuardTests(unittest.TestCase):
    def test_locked_score_is_independent_of_target_cohort_composition(self):
        locked = {"genes": ["G"], "weights": [2.0], "intercept": -0.5,
                  "mean": [10.0], "sd": [2.0]}
        one = pd.DataFrame({"G": [12.0]}, index=["patient"])
        cohort = pd.DataFrame({"G": [12.0, -100.0, 500.0]}, index=["patient", "a", "b"])
        self.assertAlmostEqual(signature.score_locked(locked, one)[0],
                               signature.score_locked(locked, cohort)[0])

    def test_missing_locked_gene_uses_training_mean_only_when_explicit(self):
        locked = {"genes": ["A", "B"], "weights": [2.0, -3.0], "intercept": 0.25,
                  "mean": [10.0, 20.0], "sd": [2.0, 4.0]}
        partial = pd.DataFrame({"A": [12.0]})
        complete_at_mean = pd.DataFrame({"A": [12.0], "B": [20.0]})
        with self.assertRaises(KeyError):
            signature.score_locked(locked, partial)
        self.assertAlmostEqual(
            signature.score_locked(locked, partial, missing="training_mean")[0],
            signature.score_locked(locked, complete_at_mean)[0],
        )

    def test_target_scaler_refit_is_forbidden(self):
        X = pd.DataFrame({"G": [0.0, 0.5, 1.0, 2.0, 2.5, 3.0]})
        cov = pd.DataFrame(index=X.index)
        model = SkModel("lr", "clf").fit(X, cov, np.array([0, 0, 0, 1, 1, 1]))
        with self.assertRaisesRegex(RuntimeError, "forbidden"):
            transfer(model, pd.DataFrame({"G": [100.0]}))

    def test_inner_signature_tuning_does_not_read_outer_test_rows(self):
        rng = np.random.default_rng(7)
        n = 72
        y = np.tile([0, 1], n // 2)
        cancer = np.repeat(["A", "B", "C"], n // 3)
        T = pd.DataFrame(rng.normal(size=(n, 8)), columns=[f"G{i}" for i in range(8)])
        T["G0"] += 2 * y
        outer_train = np.arange(54)
        first = signature.choose_c(T, y, cancer, list(T), outer_train, [1e-4, 1e-3], 3, seed=9)
        changed = T.copy()
        changed.iloc[54:] = rng.normal(1e6, 1e4, size=changed.iloc[54:].shape)
        y_changed = y.copy()
        y_changed[54:] = 1 - y_changed[54:]
        second = signature.choose_c(changed, y_changed, cancer, list(T), outer_train, [1e-4, 1e-3], 3, seed=9)
        self.assertEqual(first, second)

    def test_locked_artifact_contains_deployment_parameters(self):
        rng = np.random.default_rng(11)
        X = pd.DataFrame(rng.normal(size=(60, 5)), columns=list("ABCDE"))
        y = (X.A + X.B > 0).astype(int).values
        cancer = np.repeat(["A", "B"], 30)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "locked.json"
            signature.lock_signature(X, y, cancer, list(X), 1e-4, 3, path, "test")
            saved = json.loads(path.read_text())
        for key in ("genes", "weights", "mean", "sd", "intercept", "deployment"):
            self.assertIn(key, saved)
        self.assertIn("threshold", saved["deployment"])

    def test_nested_holdout_pools_singleton_cancer_label_strata(self):
        y = np.tile([0, 1], 20)
        cov = pd.DataFrame({"cancer": ["singleton"] + ["A"] * 19 + ["B"] * 20})
        train, valid = _nested_holdout(y, cov, seed=17)
        self.assertFalse(set(train) & set(valid))
        self.assertEqual(set(np.unique(y[train])), {0, 1})
        self.assertEqual(set(np.unique(y[valid])), {0, 1})

    def test_random_panel_null_keeps_non_protein_coding_panel_gene(self):
        """TERC belongs to the curated panel but not the protein-coding random-gene matrix."""
        rng = np.random.default_rng(23)
        n = 40
        y = np.tile([0, 1], n // 2)
        cov = pd.DataFrame({"cancer": np.repeat(["A", "B"], n // 2)})
        panel = pd.DataFrame({"TERC": rng.normal(size=n), "G0": rng.normal(size=n)})
        transcriptome = pd.DataFrame(rng.normal(size=(n, 5)), columns=["G0", "R1", "R2", "R3", "R4"])
        labels = pd.DataFrame({"sample": [f"S{i}" for i in range(n)]})
        folds = evaluate.make_folds(y, cov.cancer, "clf", n_folds=2, n_repeats=1)
        with tempfile.TemporaryDirectory() as tmp, \
             patch.object(run_experiments.C, "RESULTS", Path(tmp)), \
             patch.object(run_experiments, "load_transcriptome", return_value=(transcriptome, "test")):
            run_experiments.random_null("alt", labels, panel, cov, y, folds, ["TERC", "G0"], None, set(),
                                        SimpleNamespace(random_sets=1))
            result = pd.read_csv(Path(tmp) / "alt_random_null.csv")
        self.assertEqual(result.gene_set.tolist(), ["telomere panel", "random_0"])

    def test_stability_subsampling_preserves_singleton_strata(self):
        rng = np.random.default_rng(29)
        X = pd.DataFrame(rng.normal(size=(41, 6)), columns=[f"G{i}" for i in range(6)])
        y = np.tile([0, 1], 21)[:41]
        cancer = np.array(["singleton"] + ["A"] * 20 + ["B"] * 20)
        freq = signature.subsample_selection_frequency(X, y, cancer, list(X), 1e-4, 3, 2)
        self.assertGreater(len(freq), 0)
        self.assertTrue(np.isfinite(freq).all())


if __name__ == "__main__":
    unittest.main()
