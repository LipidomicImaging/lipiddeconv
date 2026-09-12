"""Analytic checks only; no production case, solver training or EVAL access."""
import copy
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
import os
import pickle
import unittest
from unittest.mock import patch

import numpy as np

try:
    from . import physical_identity_confidence_core as core
except ImportError:
    import physical_identity_confidence_core as core


def pattern(pattern_id="MODEL_FORWARD", pair=(30., 35.), multipliers=(2., .5)):
    return dict(pattern_id=pattern_id, donor_identity=[pattern_id, "[M-H]-", "TEST"],
                source_files=[pattern_id + "_low", pattern_id + "_high"],
                ce_pair=list(pair), channels=["f1", "f2"],
                centered_log_change=np.log(multipliers).tolist(), multiplier=list(multipliers))


def tiny_bank(duplicate=False, extra_patterns=(), model_ids=("MODEL_FORWARD",)):
    if duplicate:
        A = np.array([[1., 1., 0.], [1., 1., 1.], [0., 0., 1.]])
        fractions = np.array([[1., 0., 1., 0.], [0., 1., 0., 1.], [0., 0., 0., 0.]])
        names = ["TRUE", "TRUE", "FALSE"]
        mapped = [0, 1]
    else:
        A = np.array([[1., 0.], [1., 1.], [0., 1.]])
        fractions = np.array([[1., 0.], [0., 1.], [0., 0.]])
        names = ["TRUE", "FALSE"]
        mapped = [0]
    patterns = [pattern(), *extra_patterns]
    mappings = [dict(candidate_index=j, lipid_name=names[j], rule="TEST",
                     channels=["f1", "f2"], component_indices=[2*j, 2*j+1],
                     allowed_pattern_ids=[p["pattern_id"] for p in patterns]) for j in mapped]
    return core.build_model_bank(A, fractions, patterns, mappings, model_ids,
                                 [dict(lipid_name=name) for name in names], list(range(len(names))))


def _spawn_delete(bank, name):
    return bank.score_delete(name)


class PhysicalIdentityConfidenceTests(unittest.TestCase):
    def test_true_deletion_false_candidate_and_physical_witness(self):
        bank = tiny_bank()
        b = np.array([2., .5, 0.])
        full, proof = bank.score_full(b)
        self.assertEqual(full["status"], "BOUNDS_VALID", full)
        self.assertLess(full["upper"], 1e-7)
        self.assertTrue(bank.verify_full(b, full, proof)["physical_witness_verified"])
        selected = next(row for row in full["lp_records"] if row["key"] == full["selected_physical_witness"])
        self.assertEqual(bank.atoms[selected["atom_indices"][0]]["pattern_id"], "MODEL_FORWARD")
        self.assertEqual(selected["ce_pair"], [30., 35.])
        false, false_proof = bank.score_delete("FALSE")
        true, true_proof = bank.score_delete(["TRUE"])
        self.assertEqual(false["status"], "BOUNDS_VALID", false)
        self.assertEqual(true["status"], "BOUNDS_VALID", true)
        self.assertLess(false["upper"], 1e-7)
        self.assertGreater(true["lower"], .79)
        self.assertEqual(false["lp_records"][0]["reuse"], "FULL_RELAXATION_EXACT_ZERO_MASS")
        self.assertEqual(bank.verify_delete(b, "TRUE", true, true_proof)["status"], "PASS")
        self.assertEqual(bank.verify_delete(b, "FALSE", false, false_proof)["status"], "PASS")
        self.assertEqual(core.classify(full, true, .01), "RETAINED")
        self.assertEqual(core.classify(full, false, .01), "RELAXATION_REPLACEABLE")

    def test_all_same_identity_aliases_deleted_and_share_pattern(self):
        bank = tiny_bank(duplicate=True)
        b = np.array([2., .5, 0.])
        full, proof = bank.score_full(b)
        self.assertEqual(bank.verify_full(b, full, proof)["status"], "PASS")
        for row in full["lp_records"]:
            if row["model_kind"] == "PHYSICAL_WITNESS":
                choices = [bank.atoms[i]["pattern_id"] for i in row["atom_indices"][:2]]
                self.assertEqual(choices[0], choices[1])
        deleted, arrays = bank.score_delete("TRUE")
        self.assertEqual(set(deleted["removed_atoms"]),
                         {a["atom_index"] for a in bank.atoms if a["lipid_name"] == "TRUE"})
        self.assertEqual(len(deleted["removed_atoms"]), 4)
        self.assertGreater(deleted["lower"], .79)
        self.assertEqual(bank.verify_delete(b, "TRUE", deleted, arrays)["status"], "PASS")
        selected = list(range(len(bank.kept)))
        selected[0] = bank._candidate_atoms[0]["MODEL_FORWARD"]
        with self.assertRaisesRegex(ValueError, "SAME_IDENTITY"):
            bank._validate_physical_selection(selected, (30., 35.))

    def test_model_bank_does_not_include_cal_or_eval_endpoints(self):
        extras = [pattern("CAL", multipliers=(3., 1/3)), pattern("EVAL", multipliers=(4., .25))]
        bank = tiny_bank(extra_patterns=extras)
        self.assertEqual(set(bank.patterns), {"MODEL_FORWARD"})
        self.assertEqual({a["pattern_id"] for a in bank.atoms}, {"MODEL_FORWARD", core.NOMINAL})
        self.assertEqual(bank.fingerprint, tiny_bank().fingerprint)
        changed = copy.deepcopy(extras)
        changed[0]["multiplier"] = [10., .1]
        self.assertEqual(bank.fingerprint, tiny_bank(extra_patterns=changed).fingerprint)
        with self.assertRaisesRegex(ValueError, "MODEL_PATTERN_MEMBERSHIP"):
            tiny_bank(model_ids=("UNKNOWN",))

    def test_only_forward_direction_and_common_pair(self):
        second = pattern("MODEL_OTHER_PAIR", pair=(35., 40.), multipliers=(.5, 2.))
        bank = tiny_bank(extra_patterns=[second], model_ids=("MODEL_FORWARD", "MODEL_OTHER_PAIR"))
        self.assertEqual(bank.ce_pairs, ((30., 35.), (35., 40.)))
        # A forward endpoint cannot be used under a different pair label.
        with self.assertRaisesRegex(ValueError, "CE_DIRECTIONS"):
            bank._validate_physical_selection([bank._candidate_atoms[0]["MODEL_FORWARD"], 1], (35., 40.))
        reverse = pattern("REVERSE", pair=(35., 30.))
        with self.assertRaisesRegex(ValueError, "FORWARD_CE_ENDPOINT"):
            tiny_bank(extra_patterns=[reverse], model_ids=("REVERSE",))

    def test_relaxed_zero_residual_is_not_physical_acceptance(self):
        bank = tiny_bank()
        # The mean of nominal and endpoint is generally not an allowed endpoint.
        b = bank.A[:, 0] + bank.matrix[:, bank._candidate_atoms[0]["MODEL_FORWARD"]]
        full, proof = bank.score_full(b)
        self.assertEqual(full["status"], "BOUNDS_VALID", full)
        self.assertLess(full["relaxation_upper"], 1e-7)
        self.assertGreater(full["upper"], .1)
        self.assertGreater(full["upper"] - full["lower"], core.GAMMA_NUM)
        self.assertEqual(bank.verify_full(b, full, proof)["status"], "PASS")
        deleted, _ = bank.score_delete("TRUE")
        self.assertEqual(core.classify(full, deleted, .01), "THRESHOLD_UNRESOLVED")

    def test_numerical_and_membership_tampering_rejected(self):
        bank = tiny_bank()
        b = np.array([2., .5, 0.])
        full, proof = bank.score_full(b)
        broken = {key: value.copy() for key, value in proof.items()}
        broken["all_relaxed__dual"][0] = 1.
        with self.assertRaises(AssertionError):
            bank.verify_full(b, full, broken)
        broken_record = copy.deepcopy(full)
        broken_record["upper"] = 0.
        with self.assertRaisesRegex(ValueError, "BOUND_SELECTION_CHANGED"):
            bank.verify_full(b, broken_record, proof)
        deleted, arrays = bank.score_delete("TRUE")
        broken_record = copy.deepcopy(deleted)
        broken_record["removed_atoms"] = broken_record["removed_atoms"][:-1]
        with self.assertRaisesRegex(ValueError, "DELETION_MEMBERSHIP_CHANGED"):
            bank.verify_delete(b, "TRUE", broken_record, arrays)
        with self.assertRaisesRegex(ValueError, "PROOF_HEADER_MISMATCH"):
            bank.verify_delete(np.array([1., 1., 0.]), "TRUE", deleted, arrays)

    def test_lp_failure_abstains(self):
        bank = tiny_bank()
        with patch.object(core._problem_class(), "solve", side_effect=RuntimeError("TEST_NUMERICAL_FAILURE")):
            full, proof = bank.score_full(np.array([2., .5, 0.]))
        self.assertEqual(full["status"], "NUMERICALLY_UNRESOLVED")
        self.assertIsNone(full["upper"])
        self.assertEqual(bank.verify_full(np.array([2., .5, 0.]), full, proof)["status"], "ABSTENTION_ONLY")
        deleted, arrays = bank.score_delete("TRUE")
        self.assertEqual(deleted["status"], "NUMERICALLY_UNRESOLVED")
        self.assertEqual(core.classify(full, deleted, .01), "NUMERICALLY_UNRESOLVED")
        self.assertEqual(bank.verify_delete(np.array([2., .5, 0.]), "TRUE", deleted, arrays)["status"], "ABSTENTION_ONLY")

    def test_resume_and_pickle_without_repeating_full_lp(self):
        bank = tiny_bank()
        b = np.array([2., .5, 0.])
        full, proof = bank.score_full(b)
        resumed = tiny_bank()
        with patch.object(core._problem_class(), "solve", side_effect=RuntimeError("NO_REPEAT_FULL_FIT")):
            self.assertEqual(resumed.restore_full(b, full, proof)["status"], "PASS")
            cached_delete, cached_arrays = resumed.score_delete("FALSE")
            self.assertEqual(cached_delete["status"], "BOUNDS_VALID")
        cloned = pickle.loads(pickle.dumps(resumed))
        self.assertIsNone(cloned._global_problem)
        deleted, arrays = cloned.score_delete("TRUE")
        self.assertEqual(deleted["status"], "BOUNDS_VALID")
        self.assertEqual(cloned.verify_delete(b, "TRUE", deleted, arrays)["status"], "PASS")
        self.assertEqual(cloned.verify_delete(b, "FALSE", cached_delete, cached_arrays)["status"], "PASS")

    def test_spawn_worker_reuses_cached_full_proof(self):
        bank = tiny_bank()
        b = np.array([2., .5, 0.])
        bank.score_full(b)
        with ProcessPoolExecutor(max_workers=2, mp_context=get_context("spawn")) as executor:
            jobs = [executor.submit(_spawn_delete, bank, name) for name in ("TRUE", "FALSE")]
            results = [job.result(timeout=60) for job in jobs]
        for name, (score, arrays) in zip(("TRUE", "FALSE"), results):
            self.assertEqual(score["status"], "BOUNDS_VALID", score)
            self.assertEqual(bank.verify_delete(b, name, score, arrays)["status"], "PASS")
        self.assertEqual(results[1][0]["lp_records"][0]["reuse"], "FULL_RELAXATION_EXACT_ZERO_MASS")

    def test_import_preserves_gpu_environment(self):
        keys = ("CUDA_VISIBLE_DEVICES", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
        previous = {key: os.environ.get(key) for key in keys}
        try:
            for key in keys:
                os.environ[key] = "7"
            core._problem_class.cache_clear()
            core._problem_class()
            self.assertEqual({key: os.environ[key] for key in keys}, {key: "7" for key in keys})
        finally:
            for key, value in previous.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value

    def test_threshold_guard_and_invalid_observation(self):
        good_full = dict(status="BOUNDS_VALID", lower=0., upper=.005)
        boundary = dict(status="BOUNDS_VALID", lower=.01 + core.GAMMA_NUM, upper=.02)
        self.assertEqual(core.classify(good_full, boundary, .01), "THRESHOLD_UNRESOLVED")
        bank = tiny_bank()
        for b in (np.zeros(3), np.array([1., -1., 1.]), np.array([np.nan, 1., 1.])):
            with self.assertRaisesRegex(ValueError, "FOREGROUND_SPECTRUM"):
                bank.score_full(b)
        with self.assertRaisesRegex(ValueError, "FREEZE_OBSERVATION"):
            bank.score_delete("TRUE")


if __name__ == "__main__":
    unittest.main()
