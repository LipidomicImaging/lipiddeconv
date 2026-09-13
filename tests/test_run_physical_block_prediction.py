"""Tiny arithmetic tests for the frozen diagnostic runner; no case arrays."""
from copy import deepcopy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from analysis import run_physical_block_prediction as runner


def row(name, truth, reported=True):
    return dict(lipid_name=name, molecular_truth=truth, reportable_truth=truth,
                raw_solver_reported=reported)


class PhysicalBlockRunnerTests(unittest.TestCase):
    def test_unreported_null_rho_is_not_scored_or_filled_with_zero(self):
        rows = [dict(row('a', True), X_hat=.02, rho_zero=.001),
                dict(row('b', False, False), X_hat=0., rho_zero=None)]
        features = [dict(molecular_name=name, predictive_gain=.1, positive_block_fraction=.5)
                    for name in ('a', 'b')]
        model = dict(base=dict(mean=[0., 0.], scale=[1., 1.]),
                     auxiliary=dict(mean=[0., 0.], scale=[1., 1.]), coef=[0.] * 8, intercept=0.)
        self.assertEqual(runner.score_rows(rows, features, model), [.5, None])

    def test_pool_95_percent_uses_ceiling_and_keeps_complete_score_ties(self):
        rows = [row(f"t{i}", True) for i in range(21)] + [row("f", False)]
        scores = {f"t{i}": 2. - .01 * i for i in range(19)}
        scores.update(t19=.5, t20=.1, f=.5)
        result = runner.choose_thresholds(rows, [scores[r["lipid_name"]] for r in rows])
        self.assertEqual(result["pool_threshold"], .5)
        self.assertEqual(result["train_pool"]["TP"], 20)
        self.assertEqual(result["train_pool"]["FP"], 1)
        self.assertEqual(result["train_final"]["TP"], 19)
        self.assertEqual(result["train_final"]["FP"], 0)

    def test_no_legal_one_percent_cutoff_means_empty_final(self):
        rows = [row("f", False)] + [row(f"t{i}", True) for i in range(3)]
        scores = dict(f=4., t0=3., t1=2., t2=1.)
        result = runner.choose_thresholds(rows, [scores[r["lipid_name"]] for r in rows])
        self.assertIsNone(result["final_raw_threshold"])
        self.assertIsNone(result["final_effective_threshold"])
        self.assertEqual(result["train_final"]["TP"], 0)
        self.assertEqual(result["train_final"]["FP"], 0)
        self.assertEqual(result["train_final"]["FN"], 3)

    def test_final_effective_cutoff_respects_screened_pool(self):
        rows = [row(f"t{i}", True) for i in range(21)]
        scores = {f"t{i}": float(21 - i) for i in range(21)}
        result = runner.choose_thresholds(rows, [scores[r["lipid_name"]] for r in rows])
        self.assertEqual(result["pool_threshold"], 2.)
        self.assertEqual(result["final_raw_threshold"], 1.)
        self.assertEqual(result["final_effective_threshold"], 2.)
        self.assertEqual(result["train_pool"]["TP"], 20)
        self.assertEqual(result["train_final"]["TP"], 20)

    def test_counts_keep_unreported_truths_in_all_truth_denominator(self):
        rows = [row("a", True), row("b", True), row("unreported", True, False), row("f", False)]
        counted = runner.counts(rows, ["a"])
        self.assertEqual(counted["TP"], 1)
        self.assertEqual(counted["FP"], 0)
        self.assertEqual(counted["FN"], 2)
        self.assertAlmostEqual(counted["all_truth_recall"], 1 / 3)
        self.assertEqual(counted["all_truth_count"], 3)
        self.assertEqual(counted["raw_solver_misses"], 1)

    def test_raw_solver_misses_can_make_95_percent_pool_unattainable(self):
        rows = [row("a", True), row("b", True), row("unreported", True, False), row("f", False)]
        # Even a high score cannot make an unreported identity eligible.
        selected = runner.choose_thresholds(rows, [2., 1., 100., 0.])
        self.assertIsNone(selected["pool_threshold"])
        self.assertEqual(selected["final_raw_threshold"], 1.)
        self.assertIsNone(selected["final_effective_threshold"])
        self.assertEqual(selected["train_pool"]["TP"], 0)
        self.assertEqual(selected["train_pool"]["FN"], 3)
        self.assertEqual(selected["train_pool"]["all_truth_count"], 3)
        self.assertEqual(selected["train_pool"]["raw_solver_misses"], 1)
        self.assertEqual(selected["train_final"]["retained"], 0)

    @staticmethod
    def feature_fixture():
        A = np.array([[1., 1.], [1., 1.], [0., 1.]])
        b = np.array([1., 2., 3.])
        names = ["a", "b"]
        manifest = dict(blocks=[dict(block_index=i, channel_indices=[i]) for i in range(3)])
        differences = np.array([[1., -1.], [-.5, 2.], [9., 3.]])
        arrays = []
        for i in range(3):
            arrays.append(dict(deleted_names=np.array(names), full_held_loss=np.array(1.),
                               deleted_held_loss=1. + differences[i],
                               molecular_train_supported=np.array([True, True]),
                               molecular_held_supported=np.array([i < 2, True])))
        return A, b, names, manifest, arrays

    def test_predictive_gain_uses_all_blocks_positive_fraction_only_supported(self):
        A, b, names, manifest, arrays = self.feature_fixture()
        result = {r["molecular_name"]: r for r in runner.features_from_blocks(A, b, names, manifest, arrays)}
        self.assertAlmostEqual(result["a"]["predictive_gain"], 9.5 / 14)
        self.assertEqual(result["a"]["supported_block_count"], 2)
        self.assertAlmostEqual(result["a"]["positive_block_fraction"], .5)
        self.assertAlmostEqual(result["b"]["predictive_gain"], 4 / 14)
        self.assertEqual(result["b"]["supported_block_count"], 3)
        self.assertAlmostEqual(result["b"]["positive_block_fraction"], 2 / 3)

    def test_predictive_features_are_invariant_to_common_intensity_scaling(self):
        A, b, names, manifest, arrays = self.feature_fixture()
        original = runner.features_from_blocks(A, b, names, manifest, arrays)
        for scale in (.1, 7.):
            scaled_arrays = deepcopy(arrays)
            for block in scaled_arrays:
                block["full_held_loss"] *= scale ** 2
                block["deleted_held_loss"] *= scale ** 2
            scaled = runner.features_from_blocks(A, b * scale, names, manifest, scaled_arrays)
            self.assertEqual(len(original), len(scaled))
            for left, right in zip(original, scaled):
                self.assertEqual(left["molecular_name"], right["molecular_name"])
                self.assertAlmostEqual(left["predictive_gain"], right["predictive_gain"], places=12)
                self.assertEqual(left["positive_block_fraction"], right["positive_block_fraction"])
                self.assertEqual(left["supported_block_count"], right["supported_block_count"])


if __name__ == "__main__":
    unittest.main()
