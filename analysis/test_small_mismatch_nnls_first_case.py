"""Small numerical/KKT and cache tests; no production assets or GPU."""
import copy
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

import run_small_mismatch_nnls_first_case as runner


class InlinePool:
    def __init__(self, *, initializer, initargs, **_): initializer(*initargs)
    def __enter__(self): return self
    def __exit__(self, *args): return False
    def map(self, fn, values, **_): return map(fn, values)


class FirstCaseTests(unittest.TestCase):
    def test_kkt_accepts_optimum_and_rejects_wrong_nonnegative_fit(self):
        A = np.array([[1., 0.], [0., 1.], [1., 1.]])
        x = np.array([.5, 2.]); b = A @ x
        self.assertEqual(runner.kkt_check(A, b, x, np)["max_bound_ratio"], 0.)
        with self.assertRaisesRegex(RuntimeError, "KKT"):
            runner.kkt_check(A, b, np.zeros(2), np)
        with self.assertRaisesRegex(RuntimeError, "NEGATIVE"):
            runner.kkt_check(A, b, np.array([-.5, 2.]), np)
        # Scaling changes absolute errors but preserves the acceptance of an exact fit.
        for scale in (1e-9, 1e9): runner.kkt_check(A, scale * b, scale * x, np)

    def test_nnls_checkpoint_resume_uses_no_new_fit(self):
        import run_nnls_solver_baseline as baseline
        A = np.eye(391); mask = np.array([[True, False, True]])
        B = np.zeros((391, 1, 3), dtype=np.float32); B[3, 0, 0] = .25; B[30, 0, 2] = .75
        with tempfile.TemporaryDirectory(prefix="nnls-first-case-test-") as td:
            args = SimpleNamespace(output=Path(td)); design = {"fingerprint": "tiny-independent-system"}
            with patch.object(runner, "ProcessPoolExecutor", InlinePool):
                X, fitted, diagnostics = runner.fit_nnls(args, design, A, B, mask, np, baseline)
                np.testing.assert_array_equal(X, B); np.testing.assert_array_equal(fitted, B)
                self.assertEqual(diagnostics["max_bound_ratio"], 0.)
                with patch.object(baseline, "solve_pixel", side_effect=AssertionError("CACHE_MUST_NOT_REFIT")):
                    replay, replay_fit, replay_diagnostics = runner.fit_nnls(args, design, A, B, mask, np, baseline)
                np.testing.assert_array_equal(replay, X); np.testing.assert_array_equal(replay_fit, fitted)
                self.assertEqual(replay_diagnostics, diagnostics)
                with self.assertRaisesRegex(RuntimeError, "BINDING_CHANGED"):
                    runner.fit_nnls(args, {"fingerprint": "changed"}, A, B, mask, np, baseline)

    def test_accounting_keeps_solver_misses_and_filter_losses_separate(self):
        rows = [dict(lipid_name=n, raw_solver_reported=reported, molecular_truth=truth, reportable_truth=truth)
                for n, reported, truth in (("true1", True, True), ("true2", True, True),
                                           ("missed", False, True), ("false", True, False))]
        original = copy.deepcopy(rows)
        result = runner.accounting(rows, {"true1", "true2", "missed"}, {"true1", "true2", "missed"}, [rows[0]])
        self.assertEqual((result["raw_solver_TP"], result["raw_solver_FP"], result["raw_solver_FN"]), (2, 1, 1))
        self.assertEqual((result["filtered_TP"], result["filtered_FP"], result["filtered_FN"]), (1, 0, 2))
        self.assertEqual(result["filter_induced_true_loss"], 1)
        self.assertEqual(result["TP_retention"], .5); self.assertEqual(result["all_truth_recall"], 1 / 3)
        self.assertEqual(result["FDP"], 0.); self.assertEqual(rows, original)
        empty = runner.accounting(rows, {"true1", "true2", "missed"}, {"true1", "true2", "missed"}, [])
        self.assertIsNone(empty["FDP"]); self.assertEqual(empty["TP_retention"], 0.)


if __name__ == "__main__":
    unittest.main()
