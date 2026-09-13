"""Small analytic tests only; no production-case fitting."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis import physical_block_prediction as module


def test_complete_envelope_shared_channel_transitive_closure():
    components = np.zeros((7, 4))
    components[[0, 2], 0] = [1, 1e-300]
    components[[2, 4], 1] = 1
    components[[4, 5], 2] = 1
    components[[1, 3], 3] = 1
    manifest = module.build_blocks(components, [0, 1, 2, 0], ["a", "b", "a"])
    assert [b["channel_indices"] for b in manifest["blocks"]] == [[0, 2, 4, 5], [1, 3], [6]]
    assert manifest["blocks"][0]["component_indices"] == [0, 1, 2]
    assert manifest["aliases"] == [[0, 2], [1]]
    assert manifest["candidate_block_indices"] == [[0, 1], [0], [0]]
    assert manifest["molecular_block_indices"] == [[0, 1], [0]]
    assert manifest["uncovered_channel_indices"] == [6]
    assert manifest["statistical_independence_claim"] is False
    json.dumps(manifest, allow_nan=False)


def test_empty_physical_support_is_explicit():
    manifest = module.build_blocks(np.array([[0., 1.], [0., 0.]]), [0, 1], ["a", "b"])
    assert manifest["empty_component_indices"] == [0]
    assert manifest["component_to_block"] == [-1, 0]
    assert manifest["candidate_block_indices"] == [[], [0]]


def test_delete_all_aliases_and_preserve_original_indices():
    A = np.array([[1., 1., 0.], [0., 0., 1.], [2., 2., 1.]])
    result = module.solve_block(A, [3., 2., 8.], ["g", "g", "h"], [2])
    arrays = result["arrays"]
    assert arrays["deleted_names"].tolist() == ["g", "h"]
    assert arrays["full_x"].shape == (3,)
    assert arrays["deleted_x"].shape == (2, 3)
    assert arrays["full_x"][:2].sum() == pytest.approx(3)
    assert arrays["deleted_x"][0, :2].tolist() == [0, 0]
    assert arrays["deleted_x"][1, 2] == 0
    assert arrays["full_held_loss"] == pytest.approx(0)
    assert arrays["deleted_held_loss"] == pytest.approx([36, 4])
    assert arrays["deleted_train_loss"] == pytest.approx([9, 4])
    assert result["diagnostics"]["aliases"] == [[0, 1], [2]]
    json.dumps(result["diagnostics"], allow_nan=False)


def test_equal_strength_two_columns_and_one_combination_are_indistinguishable():
    A = np.array([[1., 0., 1.], [0., 1., 1.], [1., 1., 2.]])
    b = A @ np.array([1., 1., 0.])
    result = module.solve_block(A, b, ["first", "second", "combination"], [2])
    arrays = result["arrays"]
    assert arrays["full_train_loss"] == pytest.approx(0)
    assert arrays["deleted_train_loss"] == pytest.approx(np.zeros(3))
    assert arrays["deleted_held_loss"] == pytest.approx(np.zeros(3))
    assert result["diagnostics"]["coefficient_or_identity_uniqueness_claim"] is False


def test_changing_only_held_observations_does_not_change_any_training_coefficients():
    A = np.array([[1., 1., 0.], [0., 0., 1.], [2., 2., 1.]])
    first = module.solve_block(A, [3., 2., 8.], ["g", "g", "h"], [2])["arrays"]
    second = module.solve_block(A, [3., 2., 900.], ["g", "g", "h"], [2])["arrays"]
    for field in ("full_x", "deleted_x", "full_train_loss", "deleted_train_loss"):
        assert np.array_equal(first[field], second[field])
    assert second["full_held_loss"] == pytest.approx((900 - 8) ** 2)


def test_no_training_support_columns_are_zero_not_inferred_absent():
    A = np.array([[1., 0.], [2., 0.], [0., 1.]])
    result = module.solve_block(A, [2., 4., 9.], ["trained", "held_only"], [2])
    arrays = result["arrays"]
    assert arrays["candidate_train_supported"].tolist() == [True, False]
    assert arrays["molecular_train_supported"].tolist() == [True, False]
    assert arrays["candidate_held_supported"].tolist() == [False, True]
    assert arrays["full_x"] == pytest.approx([2., 0.])
    assert np.all(arrays["deleted_x"][:, 1] == 0)
    assert result["diagnostics"]["unsupported_training_candidate_indices"] == [1]
    assert result["diagnostics"]["deleted"][0]["empty_supported_model"] is True
    assert result["diagnostics"]["numerical_solver_calls"] == 2
    assert arrays["deleted_held_loss"] == pytest.approx([81, 81])


def test_empty_allowed_deletion_and_zero_observation():
    A = np.array([[1., 1.], [1., 1.]])
    result = module.solve_block(A, [2., 2.], ["same", "same"], [1])
    assert result["arrays"]["deleted_x"].tolist() == [[0., 0.]]
    assert result["diagnostics"]["deleted"][0]["KKT_mode"] == "EMPTY_ALLOWED_MODEL_ANALYTIC"
    zero = module.solve_block(A, [0., 0.], ["same", "same"], [1])
    assert zero["arrays"]["full_x"].tolist() == [0., 0.]
    assert zero["diagnostics"]["KKT_max"]["max_bound_ratio"] == 0


def test_reuses_original_primitive_and_original_kkt_on_all_allowed_columns(monkeypatch):
    solve = module.baseline.solve_pixel
    check = module.engine.kkt_check
    observations, shapes = [], []
    def recorded_solve(b):
        observations.append(np.array(b))
        return solve(b)
    def recorded_kkt(A, b, x, np_module):
        shapes.append(A.shape)
        return check(A, b, x, np_module)
    monkeypatch.setattr(module.baseline, "solve_pixel", recorded_solve)
    monkeypatch.setattr(module.engine, "kkt_check", recorded_kkt)
    result = module.solve_block([[1., 0.], [2., 0.], [0., 1.]], [2., 4., 9.], ["a", "b"], [2])
    assert len(observations) == 2
    assert all(np.array_equal(v, [2., 4.]) for v in observations)
    assert shapes == [(2, 2), (2, 1), (2, 1)]
    assert result["diagnostics"]["KKT_max"]["max_bound_ratio"] <= 1
    for key, values in result["arrays"].items():
        if key != "deleted_names":
            assert np.isfinite(values).all()


def test_input_arrays_not_modified_and_losses_are_not_normalized():
    A = np.array([[1., 0.], [0., 1.], [1., 1.]])
    b = np.array([2., 3., 9.])
    old_A, old_b = A.copy(), b.copy()
    result = module.solve_block(A, b, ["a", "b"], [2])
    assert np.array_equal(A, old_A) and np.array_equal(b, old_b)
    assert result["arrays"]["full_held_loss"] == pytest.approx(16)
    assert result["arrays"]["deleted_held_loss"] == pytest.approx([36, 49])


@pytest.mark.parametrize("held", [[], [0, 0], [-1], [3], [0., 1.], [0, 1, 2]])
def test_invalid_held_indices_rejected(held):
    with pytest.raises(ValueError):
        module.solve_block(np.ones((3, 2)), np.ones(3), ["a", "b"], held)


def test_invalid_components_owner_and_nonfinite_inputs_rejected():
    with pytest.raises(ValueError):
        module.build_blocks([[1., -1.]], [0, 0], ["a"])
    with pytest.raises(ValueError):
        module.build_blocks([[1., 1.]], [0, 1], ["a"])
    with pytest.raises(ValueError):
        module.build_blocks([[1., 1.]], [0., 0.], ["a"])
    with pytest.raises(ValueError):
        module.solve_block([[1.], [1.]], [1., np.nan], ["a"], [1])
