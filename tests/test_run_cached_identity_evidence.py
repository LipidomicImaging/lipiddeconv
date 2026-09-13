"""Mathematical and inference checks without executing any model fit."""
import copy
import importlib.util
import math
from pathlib import Path
import unittest
from unittest import mock

import numpy as np

SOURCE = Path(__file__).resolve().parents[1] / "analysis/run_cached_identity_evidence.py"
SPEC = importlib.util.spec_from_file_location("cached_identity_runner_under_test", SOURCE)
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


class ObservableRow(dict):
    """Inference must not access labels, even if they are present in a record."""

    def __getitem__(self, key):
        if "truth" in key or key == "X_true":
            raise AssertionError("Prediction attempted to read a truth label")
        return super().__getitem__(key)

    def get(self, key, default=None):
        if "truth" in key or key == "X_true":
            raise AssertionError("Prediction attempted to read a truth label")
        return super().get(key, default)


def reference():
    return dict(features=["a", "a_squared", "r", "s", "c"],
                base=dict(mean=[-2., -8.], scale=[.5, 5.]),
                auxiliary=dict(mean=[.1, .5], scale=[.05, .25]),
                coef=[1., -.3, 0., .2, .1], intercept=.7)


def inputs():
    rows = [ObservableRow(lipid_name="a", raw_solver_reported=True,
                          X_hat=.01, rho_zero=0., molecular_truth=True),
            ObservableRow(lipid_name="b", raw_solver_reported=True,
                          X_hat=.02, rho_zero=1e-6, molecular_truth=False),
            ObservableRow(lipid_name="unreported", raw_solver_reported=False,
                          X_hat=None, rho_zero=None, molecular_truth=True)]
    old = [dict(molecular_name=name, predictive_gain=.1, positive_block_fraction=.5)
           for name in ("a", "b", "unreported")]
    new = [dict(molecular_name=name, trimmed_gain=t, direct_gain=d,
                direct_per_signal=q, stability=k, molecular_truth=object())
           for name, t, d, q, k in (("a", 2., 5., .2, .3),
                                    ("b", 4., 5., .4, .5),
                                    ("unreported", 1e9, -1e9, 100., 1.))]
    return rows, old, new


class TestCachedIdentityRunner(unittest.TestCase):
    def test_gradient_for_all_three_arm_dimensions(self):
        for n in (6, 7, 9):
            with self.subTest(features=n):
                x = np.arange(4*n, dtype=float).reshape(4, n)/13.-.8
                theta = np.linspace(-.3, .5, n+1)
                y = np.array([0., 1., 0., 1.])
                value, gradient = runner.objective(theta, x, y)
                numerical = []
                for j in range(n+1):
                    delta = np.zeros(n+1)
                    delta[j] = 1e-6
                    numerical.append((runner.objective(theta+delta, x, y)[0]
                                      - runner.objective(theta-delta, x, y)[0])/2e-6)
                self.assertTrue(math.isfinite(value))
                np.testing.assert_allclose(gradient, numerical, atol=2e-8, rtol=2e-7)

    def test_summed_loss_and_unpenalized_intercept(self):
        x = np.zeros((3, 7))
        y = np.array([1., 0., 1.])
        theta = np.r_[np.arange(7)/10., .6]
        value, gradient = runner.objective(theta, x, y)
        self.assertAlmostEqual(value, 3*np.logaddexp(0., .6)-1.2+.5*np.dot(theta[:-1], theta[:-1]))
        np.testing.assert_allclose(gradient[:-1], theta[:-1])
        self.assertAlmostEqual(gradient[-1], 3/(1+math.exp(-.6))-2)

    def test_dev_normalization_excludes_unreported_and_uses_population_std(self):
        rows, old, new = inputs()
        matrix, mean, scale = runner.new_matrix(rows, old, new, reference(),
                                                added=["trimmed_gain", "direct_gain"])
        self.assertEqual(matrix.shape, (2, 7))
        np.testing.assert_array_equal(mean, [3., 5.])
        np.testing.assert_array_equal(scale, [1., 1.])
        np.testing.assert_array_equal(matrix[:, -2:], [[-1., 0.], [1., 0.]])

    def test_inference_uses_frozen_norm_and_never_truth_or_optimizer(self):
        rows, old, new = inputs()
        model = dict(added_features=["trimmed_gain"], mean=[10.], scale=[2.],
                     coef=[0.]*5+[1.], intercept=0.)
        with mock.patch.object(runner, "minimize", side_effect=AssertionError("inference trained")):
            actual = runner.predictions(rows, old, new, reference(), model)
        self.assertAlmostEqual(actual[0], 1/(1+math.exp(4.)))
        self.assertAlmostEqual(actual[1], 1/(1+math.exp(3.)))
        self.assertIsNone(actual[2])
        self.assertEqual(model["mean"], [10.])

    def test_unreported_observables_and_labels_do_not_change_reported_scores(self):
        rows, old, new = inputs()
        for added in runner.ARMS.values():
            with self.subTest(features=added):
                model = dict(added_features=added, mean=[0.]*len(added), scale=[1.]*len(added),
                             coef=[.2, -.1, .0, .3, .1]+[.1]*len(added), intercept=.4)
                before = runner.predictions(rows, old, new, reference(), model)
                changed_rows = copy.deepcopy(rows)
                changed_rows[-1]["rho_zero"] = object()
                changed_rows[-1]["X_hat"] = object()
                for row in changed_rows:
                    row["molecular_truth"] = object()
                changed_features = copy.deepcopy(new)
                for key in added:
                    changed_features[-1][key] = object()
                after = runner.predictions(changed_rows, old, changed_features, reference(), model)
                self.assertEqual(before, after)

    def test_no_reported_rows_returns_none_for_every_arm(self):
        rows, old, new = inputs()
        for row in rows:
            row["raw_solver_reported"] = False
            row["rho_zero"] = None
            row["X_hat"] = None
        for added in runner.ARMS.values():
            with self.subTest(features=added):
                model = dict(added_features=added, mean=[0.]*len(added), scale=[1.]*len(added),
                             coef=[0.]*(5+len(added)), intercept=0.)
                self.assertEqual(runner.predictions(rows, old, new, reference(), model), [None]*3)


if __name__ == "__main__":
    unittest.main()
