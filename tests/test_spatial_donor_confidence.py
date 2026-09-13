"""Analytic and independent optimizer checks for spatial donor descriptors."""

import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.optimize import nnls

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis.spatial_donor_confidence import compute_spatial_donor_confidence


def records(x, names):
    result = compute_spatial_donor_confidence(x, names)
    json.dumps(result, allow_nan=False)
    return {r["molecular_name"]: r for r in result["records"]}


def test_proportional_weak_copy():
    r = records([[1, 2, 3, 4], [.05, .1, .15, .2]], ["source", "copy"])["copy"]
    assert r["u1"] < 1e-28 and r["u2"] < 1e-28
    assert r["best_u1_donor_names"] == ["source"]
    assert r["spatial_donor_log10_u1"] == -12


def test_two_donor_mixture_and_unique_component():
    x = np.array([[3, 0, 0], [0, 3, 0], [.2, .2, 0], [.2, .2, .2]])
    r = records(x, ["a", "b", "mixture", "unique"])
    assert r["mixture"]["u1"] == pytest.approx(1 / 3)
    assert r["mixture"]["u2"] < 1e-28
    assert set(r["mixture"]["best_u2_donor_names"]) == {"a", "b"}
    assert r["unique"]["u1"] == pytest.approx(2 / 3)
    assert r["unique"]["u2"] == pytest.approx(1 / 3)


def test_all_aliases_aggregate_and_same_name_cannot_donate():
    x = np.array([[.4, 0, 0], [0, .4, 0], [2, 2, 0], [.1, .1, .1]])
    r = records(x, ["a", "a", "b", "c"])
    assert len(r) == 3
    assert r["a"]["candidate_indices"] == [0, 1]
    assert r["a"]["mean"] == pytest.approx(.8 / 3)
    assert r["a"]["u1"] < 1e-28
    assert r["a"]["best_u1_donor_names"] == ["b"]
    combined = records([[.4, .4, 0], [2, 2, 0], [.1, .1, .1]], ["a", "b", "c"])
    for name in r:
        assert r[name]["u1"] == pytest.approx(combined[name]["u1"])
        assert r[name]["u2"] == pytest.approx(combined[name]["u2"])


def test_no_stronger_donor_and_equal_means():
    r = records([[1, 0], [0, 1]], ["a", "b"])
    for row in r.values():
        assert row["stronger_donor_count"] == 0
        assert row["u1"] == row["u2"] == 1
        assert row["best_u2_donor_names"] == []


def test_common_scaling_and_spatial_reshape_invariance():
    rng = np.random.default_rng(13)
    x = rng.random((8, 20)) * np.arange(1, 9)[:, None]
    names = [str(i) for i in range(8)]
    reference = records(x, names)
    for scale in (1e-100, .013, 1e100):
        changed = records((x * scale).reshape(8, 4, 5), names)
        for name in names:
            for field in ("u1", "u2"):
                assert changed[name][field] == pytest.approx(reference[name][field], abs=2e-13)
            assert changed[name]["mean"] == pytest.approx(reference[name]["mean"] * scale)


def test_near_collinear_fallback_and_exact_degeneracy():
    for perturbation in (1e-5, 1e-8, 1e-12, 0):
        a = np.array([1., 1., 1., 0.])
        b = a + perturbation * np.array([1., -1., 0., 0.])
        target = .1 * a + .15 * b
        target[3] = .01
        result = compute_spatial_donor_confidence(np.array([a, b, target]), ["a", "b", "target"])
        row = result["records"][2]
        expected = .01 ** 2 / np.dot(target, target)
        assert row["u2"] == pytest.approx(expected, abs=1e-13)
        assert row["near_collinear_pair_count"] == 1
        assert row["degenerate_pair_count"] == (1 if perturbation == 0 else 0)
        assert row["u2"] <= row["u1"]
        json.dumps(result, allow_nan=False)


def test_positive_pair_constraints_include_boundary_solution():
    # The unconstrained two-column fit has a negative coefficient; the correct
    # nonnegative result is the single donor boundary.
    row = records([[2., 2., 0.], [0., 2., 2.], [.1, 0., 0.]], ["a", "b", "t"])["t"]
    assert row["u1"] == pytest.approx(.5)
    assert row["u2"] == pytest.approx(.5)
    assert row["best_u2_donor_names"] == ["a"]


def test_random_full_alias_inputs_against_independent_scipy_enumeration():
    rng = np.random.default_rng(620)
    for _ in range(5):
        x = rng.random((12, 31)) * rng.uniform(.01, 3, (12, 1))
        x[rng.random(x.shape) < .4] = 0
        names = ["a", "b", "a", "c", "d", "e", "f", "f", "g", "h", "i", "j"]
        result = compute_spatial_donor_confidence(x, names)
        groups = list(dict.fromkeys(names))
        spatial = np.array([x[np.array(names) == name].sum(axis=0) for name in groups])
        for g, row in enumerate(result["records"]):
            donors = [h for h in range(len(groups)) if spatial[h].mean() > spatial[g].mean()]
            best = {1: 1., 2: 1.}
            target_norm_sq = np.dot(spatial[g], spatial[g])
            for size in (1, 2):
                for donor_group in itertools.combinations(donors, size):
                    _, residual = nnls(spatial[list(donor_group)].T, spatial[g])
                    value = residual ** 2 / target_norm_sq
                    best[size] = min(best[size], value)
            best[2] = min(best[1], best[2])
            assert row["u1"] == pytest.approx(best[1], abs=2e-12)
            assert row["u2"] == pytest.approx(best[2], abs=2e-12)


def test_zero_maps_are_explicit_finite_placeholders():
    r = records([[0, 0], [1, 1]], ["zero", "positive"])
    assert r["zero"]["spatial_nonzero"] is False
    assert r["zero"]["u1"] == r["zero"]["u2"] == 1
    assert r["zero"]["best_u2_donor_names"] == []
    records(np.zeros((3, 5)), ["a", "a", "b"])


def test_bounded_direction_cache_with_many_near_collinear_pairs():
    rng = np.random.default_rng(791)
    donors = 1 + 1e-5 * rng.random((14, 7))
    targets = np.array([.12 * donors[i] + .18 * donors[i + 1] for i in range(3)])
    x = np.concatenate([donors, targets])
    result = compute_spatial_donor_confidence(x, [str(i) for i in range(len(x))])
    assert result["numerical_diagnostics"]["near_collinear_distinct_pair_count"] > 64
    for row in result["records"][-3:]:
        assert row["u2"] < 1e-20
        assert 0 <= row["u2"] <= row["u1"] <= 1
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("x,names", [
    ([[1, -1]], ["a"]), ([[1, np.nan]], ["a"]), ([[1, np.inf]], ["a"]),
    (np.zeros((1, 0)), ["a"]), ([[1, 2]], ["a", "b"]), ([[1, 2]], [""]),
])
def test_invalid_input_rejected(x, names):
    with pytest.raises(ValueError):
        compute_spatial_donor_confidence(x, names)
