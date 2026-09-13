"""Synthetic integration checks between new spatial evidence and the runner."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'analysis'))
import run_two_stage_spatial_donor as flow
from spatial_donor_confidence import compute_spatial_donor_confidence


def test_scorer_runner_interface_and_frozen_normalization():
    pack = compute_spatial_donor_confidence(np.array([[2., 0, 1], [.2, 0, .1], [0., 1., 0]]), ['a', 'b', 'c'])
    rows = [dict(lipid_name=n, X_hat=v, rho_zero=r, raw_solver_reported=True)
            for n, v, r in [('a', 1., .2), ('b', .1, 0.), ('c', .3, .1)]]
    base = dict(mean=[-1., -2.], scale=[1., 2.], coef=[.2, .3, 0., 0., 0., 0.], intercept=.1)
    x, norm = flow.matrix(rows, pack, base)
    assert x.shape == (3, 8) and np.isfinite(x).all()
    aug = dict(arm='AUGMENTED', base=base, auxiliary=norm, coef=[1., 0., 0., 0., 0., 0., .1, .2], intercept=0.)
    a = flow.probabilities(rows, pack, aug)
    # Scoring a single new item must reuse TRAIN scaling, not re-standardize it.
    b = flow.probabilities(rows[1:2], pack, aug)
    assert np.allclose(a[1], b[0])
    assert flow.probabilities(rows, pack, dict(arm='BASELINE', base=base)).shape == (3,)


def test_cutoff_never_splits_a_truth_false_tie():
    rows = [dict(molecular_truth=True) for _ in range(10)] + [dict(molecular_truth=False)]
    assert flow.development_cutoff(rows, [1.] * len(rows))['cutoff'] is None
    scores = [2.] * 9 + [1., 1.]
    result = flow.development_cutoff(rows, scores)
    assert result['TP'] == 9 and result['FP'] == 0 and result['cutoff'] == 2.


def test_intermediate_subset_does_not_inherit_low_fdp():
    rows = [dict(lipid_name=str(i), molecular_truth=i < 20, reportable_truth=i < 20, raw_solver_reported=True)
            for i in range(21)]
    whole = flow.metrics(rows, {str(i) for i in range(21)})
    subset = flow.metrics(rows, {'0', '20'})
    assert whole['FDP'] < .05 and subset['FDP'] == .5
    assert subset['all_truth_recall'] == .05 and subset['FN'] == 19
