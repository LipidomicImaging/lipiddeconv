"""Tiny analytic and checkpoint tests; never load project experiment arrays."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from analysis import refit_screened_nnls as refit


class ScreenedRefitTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def fit(self, A, foreground, indices, name="case", **kwargs):
        B = np.zeros((A.shape[0], 1, foreground.shape[1] + 1))
        B[:, 0, :-1] = foreground
        mask = np.array([[True] * foreground.shape[1] + [False]])
        return refit.fit_screened(A, B, mask, indices, self.root / name,
                                  workers=kwargs.pop("workers", 1), block_size=1, **kwargs)

    def test_orthogonal_exact_indices_and_zero_pixel(self):
        A = np.eye(3)
        result = self.fit(A, np.array([[2., 0.], [0., 0.], [4., 0.]]), [2, 0])
        np.testing.assert_array_equal(result["means"], [1., 0., 2.])
        with np.load(result["arrays_path"]) as archive:
            np.testing.assert_array_equal(archive["X_hat"][:, 0, :], [[2, 0, 0], [0, 0, 0], [4, 0, 0]])
            np.testing.assert_array_equal(archive["B_hat"], archive["X_hat"])
            self.assertEqual(archive["X_hat"].dtype, np.float32)
        record = json.loads((self.root / "case/nnls_blocks/block_000000_000001.json").read_text())
        self.assertEqual(record["retained_indices"], [2, 0])
        self.assertTrue(record["all_pixels_checked_before_float32"])

    def test_spawn_worker_and_empty_selection(self):
        result = self.fit(np.eye(2), np.array([[1.], [2.]]), [1], workers=2, name="spawn")
        np.testing.assert_array_equal(result["means"], [0., 2.])
        empty = self.fit(np.eye(2), np.array([[1.], [2.]]), [], name="empty")
        np.testing.assert_array_equal(empty["means"], [0., 0.])
        with np.load(empty["arrays_path"]) as archive:
            self.assertTrue((archive["B_hat"] == 0).all())

    def test_deleted_true_column_can_transfer_signal_to_wrong_column(self):
        A = np.array([[1., .8], [0., .6]])
        observation = A[:, :1] * 2
        full = self.fit(A, observation, [0, 1], name="full")
        reduced = self.fit(A, observation, [1], name="reduced")
        np.testing.assert_allclose(full["means"], [2, 0], atol=1e-12)
        np.testing.assert_allclose(reduced["means"], [0, 1.6], rtol=1e-7)
        records = refit.report_identities(reduced["means"], ["true", "false"], ["false"])
        self.assertEqual([r["lipid_name"] for r in records], ["false"])

    def test_alias_gate_precedes_sum_and_selected_membership(self):
        means = [.0006, .0006, .001, .002, .0009, .003, 10.]
        names = ["below", "below", "equal", "active", "active", "active", "excluded"]
        records = refit.report_identities(means, names, ["below", "equal", "active"])
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["lipid_name"], "active")
        self.assertAlmostEqual(records[0]["X_hat"], .005)
        self.assertEqual(records[0]["candidate_indices"], [3, 4, 5])
        self.assertEqual(records[0]["reported_candidate_indices"], [3, 5])

    def test_completed_cache_does_not_resolve(self):
        args = (np.eye(2), np.array([[1., 3.], [2., 4.]]), [1, 0])
        original = self.fit(*args, source_binding={"input": "fixed"})
        with patch.object(refit.baseline, "solve_pixel", side_effect=AssertionError("unexpected solve")):
            cached = self.fit(*args, source_binding={"input": "fixed"})
        self.assertTrue(cached["resumed"])
        self.assertEqual(original["fingerprint"], cached["fingerprint"])
        np.testing.assert_array_equal(original["means"], cached["means"])
        with self.assertRaisesRegex(RuntimeError, "IMMUTABLE_RECORD_CHANGED"):
            self.fit(*args, source_binding={"input": "changed"})
        with self.assertRaisesRegex(RuntimeError, "IMMUTABLE_RECORD_CHANGED"):
            self.fit(np.eye(2), args[1], [0, 1], source_binding={"input": "fixed"})

    def test_interrupted_run_resumes_only_missing_block(self):
        original_solver = refit.baseline.solve_pixel
        calls = []
        def interrupted(b):
            calls.append(b.copy())
            if len(calls) == 2:
                raise RuntimeError("simulated interruption")
            return original_solver(b)
        args = (np.eye(2), np.array([[1., 3.], [2., 4.]]), [0, 1])
        with patch.object(refit.baseline, "solve_pixel", side_effect=interrupted):
            with self.assertRaisesRegex(RuntimeError, "simulated interruption"):
                self.fit(*args)
        with patch.object(refit.baseline, "solve_pixel", wraps=original_solver) as solver:
            result = self.fit(*args)
        self.assertEqual(solver.call_count, 1)
        self.assertTrue(result["resumed"])
        np.testing.assert_array_equal(result["means"], [2, 3])

    def test_cached_hash_change_and_binding_change_rejected(self):
        args = (np.eye(2), np.array([[1.], [2.]]), [0, 1])
        self.fit(*args)
        block = self.root / "case/nnls_blocks/block_000000_000001.npz"
        with block.open("ab") as stream:
            stream.write(b"tampered")
        with self.assertRaisesRegex(RuntimeError, "NNLS_BLOCK_HASH_CHANGED"):
            self.fit(*args)
        self.fit(*args, name="binding")
        record_path = self.root / "binding/nnls_blocks/block_000000_000001.json"
        record = json.loads(record_path.read_text())
        record["fingerprint"] = "wrong"
        record_path.write_text(json.dumps(record))
        with self.assertRaisesRegex(RuntimeError, "NNLS_BLOCK_BINDING_CHANGED"):
            self.fit(*args, name="binding")

    def test_invalid_indices_and_background_rejected(self):
        A, B, mask = np.eye(2), np.ones((2, 1, 1)), np.ones((1, 1), dtype=bool)
        for indices in ([0, 0], [-1], [2], [1.], [True], [[0]]):
            with self.subTest(indices=indices):
                with self.assertRaises(RuntimeError):
                    refit.fit_screened(A, B, mask, indices, self.root / "invalid", workers=1)
        with self.assertRaisesRegex(RuntimeError, "NONZERO_BACKGROUND"):
            refit.fit_screened(A, np.ones((2, 1, 2)), np.array([[True, False]]), [0],
                               self.root / "background", workers=1)


if __name__ == "__main__":
    unittest.main()
