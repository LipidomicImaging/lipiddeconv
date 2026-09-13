"""Analytic implementation tests; no classifier is trained here."""
import copy
import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np

SOURCE = Path(__file__).resolve().parents[1] / "analysis/physical_block_monotone_score.py"
SPEC = importlib.util.spec_from_file_location("monotone_score_under_test", SOURCE)
score = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(score)


def model():
    return dict(features=score.FEATURES, base=dict(mean=[-2., -8.], scale=[.5, 5.]),
                auxiliary=dict(mean=[.01, .5], scale=[.02, .3]),
                coef=[.7, -.2, .3, .4, .5], intercept=.1)


def row(name="lipid", truth=True, reported=True):
    return dict(lipid_name=name, X_hat=.01, rho_zero=0. if reported else None,
                raw_solver_reported=reported, molecular_truth=truth)


def feature(name="lipid"):
    return dict(molecular_name=name, predictive_gain=.003, positive_block_fraction=.2)


class TestMonotoneScore(unittest.TestCase):
    def test_objective_gradient_finite_difference(self):
        x = np.array([[1., 1., -2., .3, .2], [-.4, .16, .2, -.3, -.2], [.2, .04, .5, 1., 2.]])
        y = np.array([1., 0., 1.])
        theta = np.array([.2, -.4, .1, .3, .5, -.1])
        value, gradient = score.objective_and_gradient(theta, x, y)
        self.assertTrue(np.isfinite(value))
        numeric = []
        for j in range(6):
            step = np.zeros(6)
            step[j] = 1e-6
            numeric.append((score.objective_and_gradient(theta+step, x, y)[0]
                            - score.objective_and_gradient(theta-step, x, y)[0])/2e-6)
        np.testing.assert_allclose(gradient, numeric, atol=1e-8, rtol=1e-7)

    def test_sum_loss_no_intercept_regularization(self):
        theta = np.array([.1, -.2, .3, .4, .5, .8])
        value, gradient = score.objective_and_gradient(theta, np.zeros((2, 5)), [0., 1.])
        expected = 2*np.logaddexp(0, .8)-.8+.5*np.sum(theta[:5]**2)
        self.assertAlmostEqual(value, expected)
        np.testing.assert_allclose(gradient[:5], theta[:5])
        self.assertAlmostEqual(gradient[5], 2/(1+np.exp(-.8))-1)

    def test_rho_monotonic_including_zero_and_floor(self):
        values = []
        for rho in (0., 1e-30, 1e-24, 1e-20, 1e-10, 1e-3, 1.):
            sample = row()
            sample["rho_zero"] = rho
            values.append(score.score_rows([sample], [feature()], model())[0])
        self.assertEqual(values[0], values[1])
        self.assertEqual(values[1], values[2])
        self.assertTrue(all(a <= b for a, b in zip(values, values[1:])))
        self.assertLess(values[2], values[-1])

    def test_evidence_monotonic_and_negative_coef_rejected(self):
        for key, vals in (("predictive_gain", [-10., -1., 0., .1, 2.]),
                          ("positive_block_fraction", [0., .1, .5, 1.])):
            results = []
            for value in vals:
                f = feature()
                f[key] = value
                results.append(score.score_rows([row()], [f], model())[0])
            self.assertTrue(all(a <= b for a, b in zip(results, results[1:])))
        bad = model()
        bad["coef"][2] = -.0001
        with self.assertRaisesRegex(ValueError, "NONMONOTONE"):
            score.score_rows([row()], [feature()], bad)

    def test_prediction_and_review_ignore_all_labels_and_unreported_rho(self):
        rows = [row(), row("missing", reported=False)]
        features = [feature(), feature("missing")]
        expected = score.score_rows(rows, features, model())
        altered = copy.deepcopy(rows)
        for r in altered:
            del r["molecular_truth"]
            r["reportable_truth"] = object()
            r["X_true"] = object()
        actual = score.score_rows(altered, features, model())
        self.assertEqual(actual, expected)
        self.assertIsNone(actual[1])
        self.assertEqual(score.review_predictions(altered, features, model(), actual)["status"], "PASS")
        self.assertEqual(score.select_names(altered, actual, 0.), ["lipid"])

    def test_95_percent_alltruth_complete_ties(self):
        rows = [row(str(j)) for j in range(20)] + [row("false", False)]
        values = [.9]*18+[.8, .7, .8]
        result = score.choose_thresholds(rows, values)
        self.assertEqual(result["pool_threshold"], .8)
        self.assertEqual(result["required_pool_truth_count"], 19)
        self.assertEqual(result["train_pool"]["TP"], 19)
        self.assertEqual(result["train_pool"]["FP"], 1)
        self.assertEqual(result["fdp5_threshold"], .7)
        self.assertEqual(result["fdp1_threshold"], .9)

    def test_missing_true_identities_cannot_disappear_from_pool_denominator(self):
        rows = [row("observed"), row("miss1", reported=False), row("miss2", reported=False)]
        result = score.choose_thresholds(rows, [.9, None, None])
        self.assertIsNone(result["pool_threshold"])
        self.assertEqual(result["required_pool_truth_count"], 3)
        self.assertEqual(result["train_pool"]["FN"], 3)
        self.assertEqual(result["train_pool"]["raw_solver_misses"], 2)

    def test_no_legal_fdp_cut_is_none_and_empty(self):
        rows = [row("true"), row("false", False)]
        result = score.choose_thresholds(rows, [.5, .5])
        self.assertIsNone(result["fdp5_threshold"])
        self.assertIsNone(result["fdp1_threshold"])
        self.assertEqual(result["train_fdp5"]["retained"], 0)
        self.assertEqual(score.select_names(rows, [.5, .5], None), [])

    def test_projected_gradient_boundary(self):
        theta = [1., 2., 0., .1, 0., 1.]
        gradient = [1., 2., 5., 4., -2., 6.]
        self.assertEqual(score.projected_gradient(theta, gradient), [1., 2., 0., 4., -2., 6.])

    def test_scalar_probability_replay_detects_corruption(self):
        rows, features = [row()], [feature()]
        values = score.score_rows(rows, features, model())
        self.assertLessEqual(score.review_predictions(rows, features, model(), values)["max_probability_error"], 1e-12)
        with self.assertRaisesRegex(ValueError, "PROBABILITY_REPLAY"):
            score.review_predictions(rows, features, model(), [values[0]+.01])

    def test_portable_seal_hash_and_immutable_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in score.ARTIFACTS:
                score.write(root/name, model() if name == "model.json" else {})
            score.write(root/"model_seal.json", dict(status="ONE_DEV_MODEL_FROZEN",
                        artifact_hashes={name: score.sha(root/name) for name in score.ARTIFACTS}))
            self.assertEqual(score.validate_model_directory(root)["model"], model())
            with self.assertRaisesRegex(ValueError, "IMMUTABLE_OUTPUT_CHANGED"):
                score.write(root/"model.json", {})
            (root/"thresholds.json").write_text("[]", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "MODEL_ARTIFACT_CHANGED"):
                score.validate_model_directory(root)


if __name__ == "__main__":
    unittest.main()
