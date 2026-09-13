"""Analytic coefficient-cache examples only; no optimizer or real case data."""
import copy
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis.cached_identity_evidence import extract


def manifest(A, names):
    molecular = list(dict.fromkeys(names))
    return dict(candidate_names=names, molecular_names=molecular,
                aliases=[[i for i, n in enumerate(names) if n == name] for name in molecular],
                blocks=[dict(block_index=i, channel_indices=[i]) for i in range(A.shape[0])])


def caches(A, b, design, full, deleted):
    result = []
    for block, x, dx in zip(design['blocks'], full, deleted):
        held = block['channel_indices']
        x, dx = np.asarray(x, dtype=np.float64), np.asarray(dx, dtype=np.float64)
        result.append(dict(full_x=x, deleted_x=dx,
            deleted_names=np.array(design['molecular_names'], dtype=str),
            full_held_loss=np.asarray(np.sum((b[held] - A[held] @ x) ** 2)),
            deleted_held_loss=np.sum((b[held, None] - A[held] @ dx.T) ** 2, axis=0)))
    return result


def single_case(coefficients=(2., 2., 2.), scale=1.):
    A = np.ones((3, 1))
    b = np.full(3, 2. * scale)
    design = manifest(A, ['one'])
    blocks = caches(A, b, design, [[t * scale] for t in coefficients], [[[0.]]] * 3)
    return A, b, design, blocks


def test_isolated_signal_has_positive_direct_and_trimmed_gain():
    result = extract(*single_case())
    row = result['rows'][0]
    np.testing.assert_allclose(result['arrays']['direct'][:, 0], 4.)
    np.testing.assert_allclose(result['arrays']['redistribution'], 0.)
    assert row['direct_gain'] == pytest.approx(1.)
    assert row['trimmed_gain'] == pytest.approx(2. / 3.)
    assert row['own_signal_energy'] == pytest.approx(12.)
    assert row['direct_per_signal'] == pytest.approx(np.arcsinh(1.))
    assert row['stability'] == pytest.approx(1.)
    assert row['direct_informative'] and row['stability_informative']


def test_perfect_other_identity_replacement_has_zero_direct():
    A, b = np.ones((3, 2)), np.full(3, 2.)
    design = manifest(A, ['one', 'substitute'])
    blocks = caches(A, b, design, [[2., 0.]] * 3, [[[0., 2.], [2., 0.]]] * 3)
    result = extract(A, b, design, blocks)
    np.testing.assert_allclose(result['arrays']['direct'], 0.)
    np.testing.assert_allclose(result['arrays']['delta'], 0.)
    assert result['rows'][0]['own_signal_energy'] > 0
    assert result['rows'][0]['direct_informative']


def test_no_own_held_prediction_can_still_have_redistribution_delta():
    A = np.array([[1., 1.], [0., 1.], [1., 1.]])
    b = np.array([2., 1., 2.])
    design = manifest(A, ['one', 'other'])
    blocks = caches(A, b, design, [[1., 1.]] * 3, [[[0., 2.], [2., 0.]]] * 3)
    result = extract(A, b, design, blocks)
    arrays = result['arrays']
    assert arrays['own_energy'][1, 0] == arrays['direct'][1, 0] == 0
    assert arrays['delta'][1, 0] == arrays['redistribution'][1, 0] == pytest.approx(1.)
    assert not arrays['support'][1, 0]
    assert result['rows'][0]['support_count'] == 2


def test_same_name_alias_split_preserves_every_evidence_quantity():
    A = np.array([[1., .2], [.5, 1.], [2., .5]])
    b = A @ np.array([2., 1.])
    design = manifest(A, ['one', 'other'])
    full = np.array([[2., 1.], [3., 1.], [1., 2.]])
    deleted = np.array([[[0., 2.], [3., 0.]]] * 3)
    original = extract(A, b, design, caches(A, b, design, full, deleted))
    expanded_A = A[:, [0, 0, 1]]
    expanded_design = manifest(expanded_A, ['one', 'one', 'other'])
    expanded_full = np.column_stack((.25 * full[:, 0], .75 * full[:, 0], full[:, 1]))
    expanded_deleted = np.stack((.25 * deleted[:, :, 0], .75 * deleted[:, :, 0], deleted[:, :, 1]), axis=2)
    expanded = extract(expanded_A, b, expanded_design,
                       caches(expanded_A, b, expanded_design, expanded_full, expanded_deleted))
    for key in original['arrays']:
        np.testing.assert_allclose(original['arrays'][key], expanded['arrays'][key], rtol=1e-13, atol=1e-14)
    for before, after in zip(original['rows'], expanded['rows']):
        for key in before:
            if key == 'molecular_name':
                assert before[key] == after[key]
            else:
                assert before[key] == pytest.approx(after[key], rel=1e-13, abs=1e-14)


def test_stability_is_scale_invariant_and_includes_supported_zeros():
    values = [extract(*single_case((1., 2., 3.), scale))['rows'][0]['stability']
              for scale in (1e-150, 1., 1e150)]
    np.testing.assert_allclose(values, 6. / 7., rtol=1e-14)
    row = extract(*single_case((1., 0., 0.)))['rows'][0]
    assert row['stability'] == pytest.approx(1. / 3.)
    assert row['stability_informative'] and row['support_count'] == 3


def test_zero_coefficient_and_unsupported_identity_have_explicit_flags():
    result = extract(*single_case((0., 0., 0.)))
    row = result['rows'][0]
    assert row['stability'] == row['direct_gain'] == row['direct_per_signal'] == 0
    assert not row['stability_informative'] and not row['direct_informative']
    assert row['support_count'] == 3
    A, b = np.zeros((3, 1)), np.ones(3)
    design = manifest(A, ['unsupported'])
    row = extract(A, b, design, caches(A, b, design, [[0.]] * 3, [[[0.]]] * 3))['rows'][0]
    assert row['support_count'] == 0 and not row['stability_informative']


def test_loss_roundoff_is_tolerated_but_original_delta_is_preserved():
    A, b, design, blocks = single_case()
    blocks[0]['deleted_held_loss'][0] += 5e-12
    blocks[0]['full_held_loss'] += 1e-16
    result = extract(A, b, design, blocks)
    assert result['arrays']['delta'][0, 0] == blocks[0]['deleted_held_loss'][0] - blocks[0]['full_held_loss']
    assert result['arrays']['redistribution'][0, 0] != 0
    blocks[0]['deleted_held_loss'][0] += .01
    with pytest.raises(ValueError, match='cached loss'):
        extract(A, b, design, blocks)


def test_mismatched_alias_deletion_names_and_nonfinite_cache_are_rejected():
    A, b, design, blocks = single_case()
    broken_design = copy.deepcopy(design)
    broken_design['aliases'] = [[]]
    with pytest.raises(ValueError, match='alias'):
        extract(A, b, broken_design, blocks)
    broken_blocks = copy.deepcopy(blocks)
    broken_blocks[0]['deleted_names'][0] = 'bad'
    with pytest.raises(ValueError, match='deleted molecular'):
        extract(A, b, design, broken_blocks)
    broken_blocks = copy.deepcopy(blocks)
    broken_blocks[0]['deleted_x'][0, 0] = 1.
    with pytest.raises(ValueError, match='alias remains'):
        extract(A, b, design, broken_blocks)
    broken_blocks = copy.deepcopy(blocks)
    broken_blocks[0]['full_x'][0] = np.nan
    with pytest.raises(ValueError, match='nonfinite'):
        extract(A, b, design, broken_blocks)
