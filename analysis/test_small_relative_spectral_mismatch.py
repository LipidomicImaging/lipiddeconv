"""Tiny physical-envelope and reproducibility checks; no production assets."""
import copy
import json
import unittest

import numpy as np

from small_relative_spectral_mismatch import build_target_library


def fixture():
    # Two isotope/profile envelopes per candidate; candidate 1 aliases candidate 0.
    fragment = np.array([2., 1., 0., 0.])
    precursor = np.array([0., 0., 3., 1.5])
    components = np.column_stack([fragment, precursor, 2 * fragment, 2 * precursor,
                                  1.5 * fragment, .8 * precursor])
    metadata = []
    for owner, identity in enumerate(("lipid A", "lipid A", "lipid B")):
        for kind in ("fragment", "precursor"):
            metadata.append(dict(candidate_index=owner, kind=kind,
                                 physical_ion_key=dict(identity=identity, adduct="[M-H]-", CE=35, ion=kind)))
    return components[:, ::2] + components[:, 1::2], components, metadata


class SmallMismatchTests(unittest.TestCase):
    def test_zero_noise_identity_and_no_input_mutation(self):
        A, C, M = fixture(); originals = A.copy(), C.copy(), copy.deepcopy(M)
        target, audit = build_target_library(A, C, M, seed=7, relative_sd=0.)
        np.testing.assert_array_equal(target, A)
        self.assertIsNot(target, A)
        np.testing.assert_array_equal(A, originals[0]); np.testing.assert_array_equal(C, originals[1])
        self.assertEqual(M, originals[2]); self.assertEqual(audit["normalization_scales"], [1., 1., 1.])

    def test_envelopes_precursor_and_normalization_coupling(self):
        A, C, M = fixture(); target, audit = build_target_library(A, C, M, seed=11)
        self.assertEqual(audit["precursor_component_count"], 3)
        for owner in range(3):
            rows = [r for r in audit["sampled_factors"] if r["candidate_index"] == owner]
            pre = np.zeros(4)
            for item in rows:
                component = 2 * owner + (item["kind"] == "precursor")
                factor = item["sampled_multiplier"]
                self.assertGreater(factor, 0.); self.assertNotEqual(factor, 1.)
                pre += C[:, component] * factor
                active = C[:, component] > 0
                np.testing.assert_allclose((C[:, component] * factor)[active] / C[active, component], factor)
                np.testing.assert_allclose(target[active, owner] / C[active, component], item["effective_multiplier_after_normalization"])
            np.testing.assert_allclose(target[:, owner], pre * audit["normalization_scales"][owner])
        np.testing.assert_allclose(np.linalg.norm(target, axis=0), np.linalg.norm(A, axis=0), rtol=1e-14)
        self.assertFalse(np.allclose(audit["normalization_scales"], 1.))
        np.testing.assert_array_equal(target > 0, A > 0)

    def test_replay_and_component_order_invariance(self):
        A, C, M = fixture(); target, audit = build_target_library(A, C, M, seed=19)
        replay, binding = build_target_library(A, C, M, seed=19)
        np.testing.assert_array_equal(replay, target); self.assertEqual(binding, audit)
        order = [5, 0, 4, 1, 3, 2]
        reordered, other = build_target_library(A, C[:, order], [M[i] for i in order], seed=19)
        np.testing.assert_array_equal(reordered, target)
        self.assertEqual(other["sampled_factors"], audit["sampled_factors"])
        different, _ = build_target_library(A, C, M, seed=20)
        self.assertFalse(np.array_equal(target, different))

    def test_candidate_subset_permutation_and_alias_invariance(self):
        A, C, M = fixture(); target, audit = build_target_library(A, C, M, seed=23)
        order = [2, 0]; indices = [4, 5, 0, 1]; metadata = copy.deepcopy([M[i] for i in indices])
        for n, item in enumerate(metadata): item["candidate_index"] = n // 2
        subset, other = build_target_library(A[:, order], C[:, indices], metadata, seed=23)
        np.testing.assert_array_equal(subset, target[:, order])
        rows = audit["sampled_factors"]
        for kind in ("fragment", "precursor"):
            alias_factors = [r["sampled_multiplier"] for r in rows if r["candidate_index"] in (0, 1) and r["kind"] == kind]
            self.assertEqual(alias_factors[0], alias_factors[1])
        np.testing.assert_allclose(target[:, 1], 2 * target[:, 0])
        self.assertEqual(len({r["sampled_multiplier"] for r in rows}), 4)

    def test_incomplete_decomposition_including_tiny_peak_fails(self):
        A, C, M = fixture()
        with self.assertRaisesRegex(ValueError, "incomplete"):
            build_target_library(A, C[:, 1:], M[1:], seed=1)
        tiny = A.copy(); tiny[2, 0] += 1e-3
        with self.assertRaisesRegex(ValueError, "incomplete"):
            build_target_library(tiny, C, M, seed=1)
        sparse_A = np.array([[1.], [1e-20]])
        sparse_C = np.array([[1.], [0.]])
        with self.assertRaisesRegex(ValueError, "incomplete"):
            build_target_library(sparse_A, sparse_C, [M[0]], seed=1)

    def test_reject_duplicate_nonfinite_zero_negative_and_unsupported(self):
        A, C, M = fixture()
        duplicate = copy.deepcopy(M); duplicate[1] = copy.deepcopy(duplicate[0])
        with self.assertRaisesRegex(ValueError, "duplicate"):
            build_target_library(A, C, duplicate, seed=1)
        for bad in (np.nan, -1.):
            changed = C.copy(); changed[0, 0] = bad
            with self.assertRaises(ValueError): build_target_library(A, changed, M, seed=1)
        changed = C.copy(); changed[:, 0] = 0.
        with self.assertRaisesRegex(ValueError, "zero"): build_target_library(A, changed, M, seed=1)
        changed = copy.deepcopy(M); changed[0]["kind"] = "unknown"
        with self.assertRaisesRegex(ValueError, "unsupported"): build_target_library(A, C, changed, seed=1)
        with self.assertRaises(ValueError): build_target_library(A.astype(int), C, M, seed=1)
        with self.assertRaises(ValueError): build_target_library(A, C, M, seed=True)
        with self.assertRaises(ValueError): build_target_library(A, C, M, seed=1, relative_sd=-.05)

    def test_fixed_condition_key_schema_and_json_audit(self):
        A, C, M = fixture(); integer, audit = build_target_library(A, C, M, seed=31)
        other = copy.deepcopy(M)
        for item in other: item["physical_ion_key"]["CE"] = 35.
        floating, _ = build_target_library(A, C, other, seed=31)
        np.testing.assert_array_equal(integer, floating)
        json.dumps(audit, allow_nan=False)
        self.assertEqual(audit["multiplier_CV_before_normalization"], .05)
        self.assertAlmostEqual(audit["lognormal_tau"], np.sqrt(np.log1p(.05**2)))
        self.assertEqual(audit["candidate_coverage_count"], 3)
        self.assertEqual(audit["distinct_physical_ion_count"], 4)
        other[0]["physical_ion_key"]["candidate_index"] = 0
        with self.assertRaises(ValueError): build_target_library(A, C, other, seed=31)

    def test_candidate_context_and_library_CE_must_agree(self):
        A, C, M = fixture()
        for field, value in (("identity", "another lipid"), ("adduct", "[M+H]+"), ("CE", 40)):
            with self.subTest(field=field):
                changed = copy.deepcopy(M); changed[1]["physical_ion_key"][field] = value
                with self.assertRaisesRegex(ValueError, "candidate components disagree"):
                    build_target_library(A, C, changed, seed=41)
        changed = copy.deepcopy(M)
        for item in changed[4:]: item["physical_ion_key"]["CE"] = 40
        with self.assertRaisesRegex(ValueError, "one fixed CE"):
            build_target_library(A, C, changed, seed=41)

    def test_float32_support_norm_and_seeded_replay(self):
        A, C, M = fixture(); A = A.astype(np.float32); C = C.astype(np.float32)
        target, audit = build_target_library(A, C, M, seed=43)
        replay, replay_audit = build_target_library(A, C, M, seed=43)
        self.assertEqual(target.dtype, np.dtype("float32"))
        np.testing.assert_array_equal(target, replay)
        self.assertEqual(audit, replay_audit)
        np.testing.assert_array_equal(target > 0, A > 0)
        np.testing.assert_allclose(np.linalg.norm(target.astype(float), axis=0),
                                   np.linalg.norm(A.astype(float), axis=0), rtol=8 * np.finfo(np.float32).eps)
        self.assertEqual(audit["fixed_CE"], 35.)
        self.assertTrue(audit["candidate_identity_adduct_CE_consistent"])


if __name__ == "__main__":
    unittest.main()
