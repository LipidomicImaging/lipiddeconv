"""Analytic cache checks; no optimizer or experimental input is used."""
from copy import deepcopy
import math
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from analysis.review_physical_block_prediction import independent_blocks, review_block, review_block_manifest, review_model_predictions


def cache(A, b, held, full, deleted, deleted_names):
    train = np.setdiff1d(np.arange(A.shape[0]), held)
    full, deleted = np.array(full, dtype=np.float64), np.array(deleted, dtype=np.float64)
    return dict(full_x=full, deleted_x=deleted, deleted_names=np.array(deleted_names),
                full_train_loss=np.sum((A[train] @ full - b[train]) ** 2),
                full_held_loss=np.sum((A[held] @ full - b[held]) ** 2),
                deleted_train_loss=np.sum((deleted @ A[train].T - b[train]) ** 2, axis=1),
                deleted_held_loss=np.sum((deleted @ A[held].T - b[held]) ** 2, axis=1))


class PhysicalBlockReviewTests(unittest.TestCase):
    def test_orthogonal_loss_and_prediction_checks(self):
        A = np.array([[1., 0.], [0., 1.], [1., 2.]])
        b = np.array([2., 3., 9.])
        arrays = cache(A, b, [2], [2., 3.], [[0., 3.], [2., 0.]], ["a", "b"])
        arrays["full_held_prediction"] = np.array([8.])
        result = review_block(A, b, ["a", "b"], [2], arrays)
        self.assertEqual(result["full_held_loss"], 1.)
        self.assertEqual(result["deleted_held_loss"], [9., 49.])
        self.assertEqual(result["held_loss_difference"], [8., 48.])
        arrays["full_held_prediction"] = np.array([9.])
        with self.assertRaisesRegex(RuntimeError, "SAVED_VALUE_MISMATCH"):
            review_block(A, b, ["a", "b"], [2], arrays)

    def test_all_aliases_deleted_and_empty_allowed_design(self):
        A = np.array([[1., 1.], [1., 2.]])
        b = np.array([2., 2.])
        arrays = cache(A, b, [1], [2., 0.], [[0., 0.]], ["same"])
        result = review_block(A, b, ["same", "same"], [1], arrays)
        self.assertEqual(result["aliases"], [[0, 1]])
        self.assertTrue(result["deleted_KKT"][0]["empty_design"])
        arrays = cache(A, b, [1], [2., 0.], [[0., 2.]], ["same"])
        with self.assertRaisesRegex(RuntimeError, "DELETED_ALIAS_NONZERO"):
            review_block(A, b, ["same", "same"], [1], arrays)

    def test_training_unsupported_column_must_be_zero(self):
        A = np.eye(2)
        b = np.array([2., 7.])
        arrays = cache(A, b, [1], [2., 0.], [[2., 0.]], ["held_only"])
        result = review_block(A, b, ["train", "held_only"], [1], arrays)
        self.assertEqual(result["unsupported_columns"], [1])
        self.assertEqual(result["group_train_supported"], [False])
        self.assertEqual(result["group_held_supported"], [True])
        arrays = cache(A, b, [1], [2., 7.], [[2., 0.]], ["held_only"])
        with self.assertRaisesRegex(RuntimeError, "UNSUPPORTED_COLUMN_NONZERO"):
            review_block(A, b, ["train", "held_only"], [1], arrays)

    def test_kkt_failure_even_when_saved_losses_match(self):
        A = np.array([[1., 0.], [0., 1.], [1., 2.]])
        b = np.array([2., 3., 9.])
        arrays = cache(A, b, [2], [1., 3.], [[0., 3.]], ["a"])
        with self.assertRaisesRegex(RuntimeError, "KKT_FAILED"):
            review_block(A, b, ["a", "b"], [2], arrays)

    def test_negative_nonfinite_wrong_dtype_and_loss_rejected(self):
        A = np.array([[1.], [1.]])
        b = np.array([2., 2.])
        original = cache(A, b, [1], [2.], [[0.]], ["a"])
        for value in (np.array([-1.]), np.array([np.nan]), np.array([2.], dtype=np.float32)):
            arrays = deepcopy(original)
            arrays["full_x"] = value
            with self.assertRaisesRegex(RuntimeError, "INVALID_FLOAT64_COEFFICIENTS"):
                review_block(A, b, ["a"], [1], arrays)
        arrays = deepcopy(original)
        arrays["deleted_held_loss"] = np.array([0.])
        with self.assertRaisesRegex(RuntimeError, "SAVED_VALUE_MISMATCH"):
            review_block(A, b, ["a"], [1], arrays)

    def test_held_and_deleted_membership_rejected(self):
        A = np.array([[1.], [1.]])
        b = np.array([2., 2.])
        arrays = cache(A, b, [1], [2.], [[0.]], ["a"])
        for held in ([1, 1], [2], [1.], [], [0, 1]):
            with self.assertRaises(RuntimeError):
                review_block(A, b, ["a"], held, arrays)
        arrays["deleted_names"] = np.array(["unknown"])
        with self.assertRaisesRegex(RuntimeError, "DELETED_NAME_MEMBERSHIP"):
            review_block(A, b, ["a"], [1], arrays)

    def test_zero_spectrum_and_support_flags(self):
        A = np.eye(2)
        b = np.zeros(2)
        arrays = cache(A, b, [1], [0., 0.], [[0., 0.]], ["a"])
        arrays.update(candidate_train_supported=np.array([True, False]),
                      candidate_held_supported=np.array([False, True]),
                      molecular_train_supported=np.array([True]),
                      molecular_held_supported=np.array([False]),
                      full_kkt=np.zeros(3), deleted_kkt=np.zeros((1, 3)))
        result = review_block(A, b, ["a", "b"], [1], arrays)
        self.assertEqual(result["full_KKT"]["max_bound_ratio"], 0.)
        self.assertEqual(result["full_held_loss"], 0.)
        arrays["candidate_train_supported"] = np.array([True, True])
        with self.assertRaisesRegex(RuntimeError, "SUPPORT_FLAG_MISMATCH"):
            review_block(A, b, ["a", "b"], [1], arrays)

    def test_independent_dfs_merges_transitive_shared_envelopes(self):
        C = np.array([[1., 0., 0.], [1., 1., 0.], [0., 1., 0.], [0., 0., 2.], [0., 0., 1.]])
        groups = independent_blocks(C)
        self.assertEqual(groups, [dict(channel_indices=[0, 1, 2], component_indices=[0, 1]),
                                  dict(channel_indices=[3, 4], component_indices=[2])])
        owner = np.array([0, 1, 1])
        A = np.column_stack([C[:, 0], C[:, 1] + C[:, 2]])
        blocks = [dict(block_index=0, **groups[0], candidate_indices=[0, 1], molecular_indices=[0]),
                  dict(block_index=1, **groups[1], candidate_indices=[1], molecular_indices=[0])]
        result = review_block_manifest(A, C, owner, ["same", "same"], blocks)
        self.assertEqual(result["block_sizes"], [3, 2])
        wrong = deepcopy(blocks)
        wrong[0]["component_indices"] = [0]
        with self.assertRaisesRegex(RuntimeError, "BLOCK_MEMBERSHIP_MISMATCH"):
            review_block_manifest(A, C, owner, ["same", "same"], wrong)
        wrong = deepcopy(blocks)
        wrong[0]["channel_indices"] = [0, 1]
        with self.assertRaisesRegex(RuntimeError, "BLOCK_CLOSURE_OR_INDEX_MISMATCH"):
            review_block_manifest(A, C, owner, ["same", "same"], wrong)

    @staticmethod
    def training_fixture(missing_truth=True, second_false=False):
        rows = [dict(lipid_name="a", raw_solver_reported=True, X_hat=1., rho_zero=1., molecular_truth=True),
                dict(lipid_name="b", raw_solver_reported=True, X_hat=100., rho_zero=.01, molecular_truth=not second_false),
                dict(lipid_name="unreported", raw_solver_reported=False, X_hat=0., rho_zero=None, molecular_truth=missing_truth)]
        features = [dict(molecular_name="a", predictive_gain=.2, positive_block_fraction=.5),
                    dict(molecular_name="b", predictive_gain=.8, positive_block_fraction=1.),
                    dict(molecular_name="unreported", predictive_gain=100., positive_block_fraction=0.)]
        model = dict(features=["predictive_gain", "positive_block_fraction"],
                     base=dict(mean=[1., -1.], scale=[1., 1.]),
                     auxiliary=dict(mean=[.5, .75], scale=[.3, .25], constant_features=[]),
                     coef=[0.] * 8, intercept=0., training_names=["a", "b"])
        saved = [dict(lipid_name="a", new_score=.5), dict(lipid_name="b", new_score=.5),
                 dict(lipid_name="unreported", new_score=None)]
        return rows, features, model, saved

    def test_scalar_model_replay_all_eight_terms_and_zero_rho_without_check_truth(self):
        rows = [dict(lipid_name="x", raw_solver_reported=True, X_hat=100., rho_zero=.01),
                dict(lipid_name="zero", raw_solver_reported=True, X_hat=10., rho_zero=0.),
                dict(lipid_name="u", raw_solver_reported=False, X_hat=0., rho_zero=None)]
        features = [dict(molecular_name="x", predictive_gain=-.1, positive_block_fraction=.9),
                    dict(molecular_name="zero", predictive_gain=0., positive_block_fraction=1.),
                    dict(molecular_name="u", predictive_gain=100., positive_block_fraction=0.)]
        coef = [.1, -.05, .02, .03, -.001, .1, .2, -.1]
        model = dict(features=["predictive_gain", "positive_block_fraction"],
                     base=dict(mean=[0., 0.], scale=[1., 1.]), auxiliary=dict(mean=[0., 0.], scale=[1., 1.]),
                     coef=coef, intercept=.2)
        phis = ([2., -2., 4., -4., 4., 0., -.1, .9], [1., -24., 1., -24., 576., 1., 0., 1.])
        values = [1 / (1 + math.exp(-(sum(a * c for a, c in zip(phi, coef)) + .2))) for phi in phis]
        saved = [dict(lipid_name="x", new_score=values[0]), dict(lipid_name="zero", new_score=values[1]),
                 dict(lipid_name="u", new_score=None)]
        result = review_model_predictions(rows, features, model, {"deliberately_not_checked": True}, saved)
        self.assertLess(result["maxprob_error"], 1e-15)
        saved[0]["new_score"] += .01
        with self.assertRaisesRegex(RuntimeError, "MODEL_PROBABILITY_MISMATCH"):
            review_model_predictions(rows, features, model, None, saved)

    def test_train_only_normalization_and_missing_truth_empty_pool(self):
        rows, features, model, saved = self.training_fixture()
        empty = dict(TP=0, FP=0, FN=3, FDP=None, all_truth_recall=0., TP_retention=0.,
                     all_truth_count=3, raw_solver_misses=1, retained=0)
        thresholds = dict(pool_threshold=None, final_raw_threshold=.5, final_effective_threshold=None,
                          train_pool=empty, train_final=empty)
        self.assertTrue(review_model_predictions(rows, features, model, thresholds, saved,
                                                is_train=True)["thresholds_independently_verified"])
        wrong = deepcopy(model)
        wrong["auxiliary"]["mean"][0] = (.2 + .8 + 100.) / 3
        with self.assertRaisesRegex(RuntimeError, "TRAIN_ONLY_NORMALIZATION_MISMATCH"):
            review_model_predictions(rows, features, wrong, thresholds, saved, is_train=True)
        wrong_thresholds = deepcopy(thresholds)
        wrong_thresholds["pool_threshold"] = .5
        with self.assertRaisesRegex(RuntimeError, "THRESHOLD_REPLAY_MISMATCH"):
            review_model_predictions(rows, features, model, wrong_thresholds, saved, is_train=True)
        saved[2]["new_score"] = .5
        with self.assertRaisesRegex(RuntimeError, "UNREPORTED_CLASSIFIER_SCORE"):
            review_model_predictions(rows, features, model, thresholds, saved)

    def test_train_complete_ties_fail_one_percent_instead_of_splitting(self):
        rows, features, model, saved = self.training_fixture(missing_truth=False, second_false=True)
        pool = dict(TP=1, FP=1, FN=0, FDP=.5, all_truth_recall=1., TP_retention=1.,
                    all_truth_count=1, raw_solver_misses=0, retained=2)
        empty = dict(TP=0, FP=0, FN=1, FDP=None, all_truth_recall=0., TP_retention=0.,
                     all_truth_count=1, raw_solver_misses=0, retained=0)
        thresholds = dict(pool_threshold=.5, final_raw_threshold=None, final_effective_threshold=None,
                          train_pool=pool, train_final=empty)
        result = review_model_predictions(rows, features, model, thresholds, saved, is_train=True)
        self.assertEqual(result["status"], "PASS")
        wrong = deepcopy(thresholds)
        wrong["final_raw_threshold"] = .5
        with self.assertRaisesRegex(RuntimeError, "THRESHOLD_REPLAY_MISMATCH"):
            review_model_predictions(rows, features, model, wrong, saved, is_train=True)

    def test_train_successful_pool_and_final_and_accounting_rejection(self):
        rows, features, model, saved = self.training_fixture(missing_truth=False)
        metrics = dict(TP=2, FP=0, FN=0, FDP=0., all_truth_recall=1., TP_retention=1.,
                       all_truth_count=2, raw_solver_misses=0, retained=2)
        thresholds = dict(pool_threshold=.5, final_raw_threshold=.5, final_effective_threshold=.5,
                          train_pool=metrics, train_final=deepcopy(metrics))
        result = review_model_predictions(rows, features, model, thresholds, saved, is_train=True)
        self.assertEqual(result["reported_count"], 2)
        thresholds["train_final"]["TP"] = 1
        with self.assertRaisesRegex(RuntimeError, "THRESHOLD_ACCOUNTING_MISMATCH"):
            review_model_predictions(rows, features, model, thresholds, saved, is_train=True)


if __name__ == "__main__":
    unittest.main()
