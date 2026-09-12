"""Independent local accounting and array review; never fit or choose HOLD cutoffs."""
import argparse
import hashlib
import json
import math
from pathlib import Path


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def accounting(rows, cutoff):
    truth = {r['lipid_name'] for r in rows if r['molecular_truth']}
    reportable = {r['lipid_name'] for r in rows if r['reportable_truth']}
    raw = {r['lipid_name'] for r in rows if r['raw_solver_reported']}
    selected = {r['lipid_name'] for r in rows if r['raw_solver_reported'] and cutoff is not None and r['joint_score'] >= cutoff}
    tp, fp, raw_tp = len(selected & truth), len(selected - truth), len(raw & truth)
    return dict(raw_TP=raw_tp, raw_FP=len(raw-truth), raw_FN=len(truth-raw), TP=tp,
                FP=fp, FN=len(truth-selected), FDP=fp/len(selected) if selected else None,
                all_truth_recall=tp/len(truth), TP_retention=tp/raw_tp if raw_tp else 0,
                reportable_truth_recall=len(selected & reportable)/len(reportable),
                retained=len(selected), coverage=len(selected)/len(raw) if raw else 0,
                filter_induced_true_loss=raw_tp-tp, truth_count=len(truth),
                distinct_retained_identities=len(selected))


def meets_target(m):
    return bool(m['retained'] and m['FDP'] <= .01 and m['all_truth_recall'] >= .8 and m['TP_retention'] >= .8)


def scalar_prediction(row, model):
    a = (math.log10(max(row['X_hat'], 1e-12))-model['mean'][0])/model['scale'][0]
    b = (math.log10(max(row['rho_zero'], 1e-24))-model['mean'][1])/model['scale'][1]
    terms = (a, b, a*a, a*b, b*b, int(row['rho_zero'] == 0))
    logit = sum(c*x for c, x in zip(model['coef'], terms))+model['intercept']
    return (1/(1+math.exp(-logit))) if logit >= 0 else math.exp(logit)/(1+math.exp(logit))


def review(root, case):
    import numpy as np
    out = root/'cases'/case
    source = read(out/'input.json')['scientific']
    result, remote = read(out/'result.json'), read(out/'review.json')
    model = read(root/'model.json')
    assert result['case'] == remote['case'] == case
    assert remote['status'] == 'PASS' and result['normal_completion'] and result['all_outputs_finite']
    assert sha(out/'result.json') == remote['result_sha256']
    assert sha(root/'model.json') == result['model_sha256'] == remote['model_sha256']
    assert sha(root/'contract.json') == read(root/'contract_seal.json')['contract_sha256']
    missing = [name for name in result['artifacts'] if not (out/name).is_file()]
    assert not missing, missing
    for name, expected in result['artifacts'].items():
        assert sha(out/name) == expected, name
    with np.load(out/'prepared_arrays.npz', allow_pickle=False) as z:
        A, T, X, B, mask = [z[k] for k in ('A_solver', 'A_target', 'X_true', 'B', 'mask')]
    with np.load(out/'learned_arrays.npz', allow_pickle=False) as z:
        Xhat, Bhat = z['X_hat'], z['B_hat']
    assert A.shape == (1084, 391) and X.shape == Xhat.shape == (391, *mask.shape)
    assert B.shape == Bhat.shape == (1084, *mask.shape)
    assert all(np.isfinite(q).all() for q in (A, T, X, B, Xhat, Bhat))
    assert (Xhat >= 0).all() and all((q[:, ~mask] == 0).all() for q in (X, B, Xhat, Bhat))
    for spectra, abundance, observation in ((T, X, B), (A, Xhat, Bhat)):
        predicted = spectra.astype(float) @ abundance[:, mask].astype(float)
        assert np.allclose(predicted, observation[:, mask], rtol=8*np.finfo(np.float32).eps, atol=0)
    metadata = read(out/'metadata.json')
    names = metadata['lipid_name']
    truths = {names[j] for j in np.flatnonzero(np.any(X[:, mask] > 0, axis=1))}
    reportable = {names[j] for j in np.flatnonzero(X[:, mask].mean(axis=1) > .001)}
    assert truths == reportable == set(source['truth_names']) and len(truths) == 125
    candidates = read(out/'candidate_records.json')
    rows = read(out/'molecular_records.json')
    assert len(candidates) == 391 and len(rows) == len(set(names))
    assert {r['lipid_name'] for r in rows} == set(names)
    means = Xhat[:, mask].mean(axis=1, dtype=float)
    reported = set(np.flatnonzero(means > .001).tolist())
    b = B[:, mask].mean(axis=1, dtype=float)
    denominator = float(b @ b)+1e-12
    for i, r in enumerate(candidates):
        assert r['candidate_index'] == i and r['lipid_name'] == names[i]
        assert math.isclose(r['X_hat'], means[i], rel_tol=2e-12, abs_tol=1e-14)
        assert r['raw_solver_reported'] == (i in reported)
        if i in reported:
            value = read(out/f'rho/candidate_{i:04d}.json')['result']
            assert value == r['rho_details']
            rho = max(0., (value['q_deleted']-value['q_star'])/denominator)
            assert math.isclose(rho, r['rho_zero'], rel_tol=2e-12, abs_tol=1e-22)
    for r in rows:
        members = [i for i, name in enumerate(names) if name == r['lipid_name']]
        active = [i for i in members if i in reported]
        assert members == r['candidate_indices'] and active == r['reported_candidate_indices']
        assert r['raw_solver_reported'] == bool(active)
        assert r['molecular_truth'] == (r['lipid_name'] in truths)
        assert r['reportable_truth'] == (r['lipid_name'] in reportable)
        assert r['X_hat'] == sum(candidates[i]['X_hat'] for i in active)
        assert r['rho_zero'] == (max(candidates[i]['rho_zero'] for i in active) if active else None)
        if active:
            assert abs(scalar_prediction(r, model)-r['joint_score']) < 2e-12
        else:
            assert r['joint_score'] is None
    foreground = Xhat[:, mask]
    for start in range(0, mask.sum(), 250):
        stop = min(start+250, int(mask.sum()))
        stem = out/f'nnls_blocks/block_{start:06d}_{stop:06d}'
        item = read(stem.with_suffix('.json'))
        assert item['start'] == start and item['stop'] == stop and item['KKT']['max_bound_ratio'] <= 1
        assert item['all_pixels_checked_before_float32'] and item['array_sha256'] == sha(stem.with_suffix('.npz'))
        with np.load(stem.with_suffix('.npz'), allow_pickle=False) as z:
            assert np.array_equal(z['X_hat'], foreground[:, start:stop])
    raw = accounting(rows, 0.)
    assert raw == remote['raw']
    if source['role'] == 'CAL':
        # This uses only CAL labels. Ascending order finds the lowest valid tied-score cutoff.
        values = sorted({r['joint_score'] for r in rows if r['raw_solver_reported']})
        valid = [(t, accounting(rows, t)) for t in values if meets_target(accounting(rows, t))]
        best = valid[0] if valid else None
        assert bool(best) == remote['calibration_feasible']
        if best:
            assert best[0] == remote['feasible_point']['threshold']
            assert best[1] == remote['feasible_point']['pooled']
        threshold, metrics = best if best else (None, None)
    else:
        seal = read(root/'calibration_seal.json')
        assert result['threshold_seal_sha256'] == sha(root/'calibration_seal.json')
        assert sha(root/'threshold.json') == seal['threshold_sha256']
        threshold = read(root/'threshold.json')['threshold']
        metrics = accounting(rows, threshold)
        assert metrics == remote['metrics'] and meets_target(metrics) == remote['target_met']
        assert remote['no_HOLD_threshold_curve'] is True
    report = dict(status='PASS', case=case, source_result_sha256=sha(out/'result.json'),
                  remote_review_sha256=sha(out/'review.json'), all_result_artifact_hashes_verified=True,
                  all_arrays_preserved_locally=True, finite_forward_equations_and_block_membership=True,
                  cached_rho_and_model_formula_verified=True, independent_accounting=True,
                  raw=raw, threshold=threshold, selected_metrics=metrics,
                  target_met=bool(metrics and meets_target(metrics)), no_optimizer_rerun=True,
                  no_HOLD_threshold_search=True, role=source['role'])
    target_path = out/'local_independent_review.json'
    encoded = json.dumps(report, indent=2, allow_nan=False)+'\n'
    if target_path.exists():
        assert target_path.read_text(encoding='utf-8') == encoded
    else:
        target_path.write_text(encoded, encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--case', choices=('CAL1', 'CAL2', 'HOLD1', 'HOLD2'), required=True)
    args = parser.parse_args()
    review(args.root, args.case)
