"""Bounded observable-only auxiliary feature screen on cached NNLS results."""
import argparse
import importlib.util
import json
import math
import os
from pathlib import Path
from types import SimpleNamespace

for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('prior_joint', ROOT / 'analysis/run_small_mismatch_joint_confidence.py')
prior = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prior)
read, write, sha, group, metrics = prior.read, prior.write, prior.sha, prior.group, prior.metrics
FEATURES = ('library_separation', 'competition_log_ratio', 'global_mean_log_disagreement', 'log_effective_pixel_fraction')
VARIANTS = {
    'PLUS_LIBRARY_COMPETITION': FEATURES[:2],
    'PLUS_MEAN_CONSISTENCY': FEATURES[2:3],
    'PLUS_SPATIAL_CONCENTRATION': FEATURES[3:4],
    'ALL_AUXILIARY': FEATURES,
}
CODE = ('analysis/run_small_mismatch_auxiliary_confidence.py', 'analysis/run_small_mismatch_joint_confidence.py',
        'analysis/run_joint_confidence_cal_pilot.py', 'docs/SMALL_MISMATCH_AUXILIARY_CONFIDENCE.md')
CASE_FILES = ('design.json', 'result.json', 'summary.json', 'metadata.json', 'candidate_records.json',
              'molecular_records.json', 'prepared_arrays.npz', 'learned_arrays.npz')
BASE_FILES = ('prepared/contract.json', 'prepared/seal.json', 'fit/provenance.json', 'fit/out_of_group_records.json',
              'fit/models.json', 'review.json') + tuple(f'fit/fold_{f}_seal.json' for f in range(5))


def observable_features(A, X, names, global_coefficients, reported):
    """No truth, target library, class, calibration labels or test labels enter."""
    import numpy as np
    A, X = np.asarray(A, dtype=float), np.asarray(X, dtype=float)
    assert A.ndim == X.ndim == 2 and A.shape[1] == X.shape[0] == len(names)
    assert np.isfinite(A).all() and np.isfinite(X).all() and (A >= 0).all() and (X >= 0).all()
    norms = np.linalg.norm(A, axis=0)
    assert (norms > 0).all() and X.shape[1] > 0
    unit = A / norms
    cosine = np.clip(unit.T @ unit, 0, 1)
    mean = X.mean(axis=1, dtype=np.float64)
    output = {}
    for name in dict.fromkeys(names[i] for i in reported):
        indices = [i for i in reported if names[i] == name]
        others = [i for i, n in enumerate(names) if n != name]
        assert others
        weight = mean[indices] / mean[indices].sum()
        other_cosine = cosine[np.ix_(indices, others)]
        separation = float(weight @ np.sqrt(np.maximum(0, 1-other_cosine.max(axis=1)**2)))
        pressure = float(weight @ ((other_cosine**2) @ mean[others]))
        abundance = float(mean[indices].sum())
        global_x = float(sum(global_coefficients[i] for i in indices))
        assert abundance > 0 and math.isfinite(global_x) and global_x >= 0
        spatial = X[indices].sum(axis=0)
        effective_pixels = float(spatial.sum()**2 / np.dot(spatial, spatial))
        output[name] = dict(
            library_separation=separation,
            competition_log_ratio=math.log10(abundance/(pressure+1e-12)),
            global_mean_log_disagreement=abs(math.log10((global_x+1e-12)/(abundance+1e-12))),
            log_effective_pixel_fraction=math.log10(effective_pixels/X.shape[1]),
            details=dict(reported_alias_indices=indices, abundance=abundance, cosine_weighted_competitor_abundance=pressure,
                         cached_global_NNLS_coefficient=global_x, effective_pixels=effective_pixels, foreground_pixels=X.shape[1]))
    return output


def load_observables(case_dir):
    import numpy as np
    with np.load(case_dir / 'prepared_arrays.npz', allow_pickle=False) as z:
        A, mask = z['A_solver'].astype(float), z['mask'].copy()
    with np.load(case_dir / 'learned_arrays.npz', allow_pickle=False) as z:
        X = z['X_hat'][:, mask].astype(float)
    metadata = read(case_dir / 'metadata.json')
    candidates = read(case_dir / 'candidate_records.json')
    assert A.shape == (1084, 391) and X.shape == (391, 15837) and mask.dtype == bool
    assert len(candidates) == len(metadata['lipid_name']) == 391
    mean = X.mean(axis=1)
    for i, row in enumerate(candidates):
        assert row['candidate_index'] == i and row['lipid_name'] == metadata['lipid_name'][i]
        assert math.isclose(mean[i], row['X_hat'], rel_tol=1e-12, abs_tol=1e-14)
        assert row['raw_solver_reported'] == (mean[i] > .001)
    reported = [i for i, r in enumerate(candidates) if r['raw_solver_reported']]
    coefficients = {i: candidates[i]['rho_details']['x_star_candidate'] for i in reported}
    return A, X, metadata['lipid_name'], coefficients, reported


def source_bindings(a):
    _, _, _ = prior.sources(a.case_dir)
    result = read(a.case_dir / 'result.json')
    case = {n: sha(a.case_dir/n) for n in CASE_FILES}
    for name in CASE_FILES:
        if name != 'result.json':
            assert case[name] == result['artifact_hashes'][name], name
    baseline = {n: sha(a.baseline_dir/n) for n in BASE_FILES}
    assert read(a.baseline_dir/'review.json')['status'] == 'CACHED_REVIEW_PASS'
    c = read(a.baseline_dir/'prepared/contract.json')
    assert c['original_source_hashes'] == {n: case[n] for n in c['original_source_hashes']}
    p = read(a.baseline_dir/'fit/provenance.json')
    for name, h in p['files'].items():
        assert sha(a.baseline_dir/'fit'/name) == h
    assert sha(ROOT/'analysis/run_joint_confidence_cal_pilot.py') == c['engine_sha256']
    return dict(case=case, baseline=baseline, implementation={n: sha(ROOT/n) for n in CODE})


def prepare(a):
    bindings = source_bindings(a)
    a.output.mkdir(parents=True, exist_ok=False)
    features = observable_features(*load_observables(a.case_dir))
    rows = read(a.case_dir/'molecular_records.json')
    for r in rows:
        r['fold'] = group(r['lipid_name'])
        r['auxiliary'] = features.get(r['lipid_name'])
        assert (r['auxiliary'] is not None) == r['raw_solver_reported']
        if r['auxiliary']:
            assert r['auxiliary']['details']['reported_alias_indices'] == r['reported_candidate_indices']
            assert math.isclose(r['auxiliary']['details']['abundance'], r['X_hat'], rel_tol=1e-12)
    write(a.output/'prepared/records.json', rows)
    contract = dict(version='AUXILIARY_EXPLORATION_V1', bindings=bindings, features=FEATURES, variants=VARIANTS,
                    records_sha256=sha(a.output/'prepared/records.json'),
                    model='Same six baseline features plus TRAIN-standardized linear auxiliaries; unchanged L2 logistic C=1, lbfgs, max_iter=2000, tol=1e-8',
                    baseline='Reuse exact prior frozen models and predictions; no baseline refit',
                    split='Unchanged five identity hash folds; TEST=f,CAL=(f+1)%5,TRAIN=other three',
                    calibration='Separate CAL cutoff for each variant/fold: maximum tied-score retention at empirical FDP<=.01',
                    primary_variant='ALL_AUXILIARY', target='Nonempty, FDP<=.01 and all-truth recall and TP retention>=.80',
                    partial_extensions='Three predeclared exploratory ablations, all reported; cannot substitute for the primary variant',
                    exposure='Same fully exposed single case and folds; adaptive method development across user turns, not independent confirmation',
                    signal_only='A_solver,X_hat,mask,cached global NNLS coefficient; no A_target/X_true/B or truth in feature calculation',
                    limitations='Full-library cosine is pairwise geometry, not a cone certificate; Neff is concentration, not repeatability; global disagreement is not error truth',
                    repeated_variants_or_search=False, new_NNLS_or_rho=False, GPU=False, next_case=False,
                    runtime=dict(numpy='2.1.3', scipy='1.15.3', sklearn='1.6.1'))
    write(a.output/'prepared/contract.json', contract)
    write(a.output/'prepared/seal.json', dict(status='FROZEN_BEFORE_AUGMENTED_MODEL_FIT', contract_sha256=sha(a.output/'prepared/contract.json')))
    print('PREPARED', len(features), 'reported identities', sha(a.output/'prepared/contract.json'), flush=True)


def validate(a):
    c = read(a.output/'prepared/contract.json')
    assert sha(a.output/'prepared/contract.json') == read(a.output/'prepared/seal.json')['contract_sha256']
    assert c['bindings'] == source_bindings(a)
    assert sha(a.output/'prepared/records.json') == c['records_sha256']
    assert c['variants'] == {k: list(v) for k, v in VARIANTS.items()}
    rows = read(a.output/'prepared/records.json')
    original = {r['lipid_name']: r for r in read(a.case_dir/'molecular_records.json')}
    assert len(rows) == len(original) == len({r['lipid_name'] for r in rows})
    for r in rows:
        assert all(r[k] == v for k, v in original[r['lipid_name']].items())
        assert r['fold'] == group(r['lipid_name'])
    return rows, c


def base_matrix(rows, baseline):
    import numpy as np
    logs = np.array([[np.log10(max(r['X_hat'], 1e-12)), np.log10(max(r['rho_zero'], 1e-24))] for r in rows])
    z = (logs-np.array(baseline['mean']))/np.array(baseline['scale'])
    return np.column_stack([z[:, 0], z[:, 1], z[:, 0]**2, z[:, 0]*z[:, 1], z[:, 1]**2,
                            [float(r['rho_zero'] == 0) for r in rows]])


def design_matrix(rows, names, baseline, mean=None, scale=None):
    import numpy as np
    auxiliary = np.array([[r['auxiliary'][k] for k in names] for r in rows])
    if mean is None:
        mean, scale = auxiliary.mean(axis=0), np.maximum(auxiliary.std(axis=0), 1e-12)
    return np.column_stack([base_matrix(rows, baseline), (auxiliary-mean)/scale]), mean, scale


def cutoff(calibration, scores):
    buckets = {}
    for r, score in zip(calibration, scores):
        buckets.setdefault(float(score), []).append(r['molecular_truth'])
    tp = fp = 0
    best = None
    for score in sorted(buckets, reverse=True):
        tp += sum(buckets[score]); fp += len(buckets[score])-sum(buckets[score])
        if fp/(tp+fp) <= .01:
            best = score
    return best


def run(a):
    import numpy as np
    import scipy, sklearn, warnings
    from sklearn.linear_model import LogisticRegression
    from sklearn.exceptions import ConvergenceWarning
    rows, c = validate(a)
    assert dict(numpy=np.__version__, scipy=scipy.__version__, sklearn=sklearn.__version__) == c['runtime']
    fit = a.output/'fit'; fit.mkdir(exist_ok=False)
    old = {r['lipid_name']: r for r in read(a.baseline_dir/'fit/out_of_group_records.json')}
    predicted = {r['lipid_name']: dict(lipid_name=r['lipid_name'], fold=r['fold'],
                    scores={'BASELINE': old[r['lipid_name']]['joint_probability']},
                    retained={'BASELINE': old[r['lipid_name']]['joint_probability_0.01_retained']}) for r in rows}
    files = []
    for variant, names in VARIANTS.items():
        for f in range(5):
            cal = (f+1)%5
            train = [r for r in rows if r['raw_solver_reported'] and r['fold'] not in (f, cal)]
            calibration = [r for r in rows if r['raw_solver_reported'] and r['fold'] == cal]
            test = [r for r in rows if r['fold'] == f]
            baseline = read(a.baseline_dir/f'fit/fold_{f}_seal.json')['model']
            assert sorted(r['lipid_name'] for r in train) == baseline['train_groups']
            X, mean, scale = design_matrix(train, names, baseline)
            with warnings.catch_warnings():
                warnings.simplefilter('error', ConvergenceWarning)
                model = LogisticRegression(C=1., solver='lbfgs', max_iter=2000, tol=1e-8).fit(X, [r['molecular_truth'] for r in train])
            scores = model.predict_proba(design_matrix(calibration, names, baseline, mean, scale)[0])[:, 1]
            threshold = cutoff(calibration, scores)
            seal = dict(variant=variant, fold=f, auxiliary_features=names, cutoff=threshold,
                        train_names=sorted(r['lipid_name'] for r in train), cal_names=sorted(r['lipid_name'] for r in calibration),
                        test_names=sorted(r['lipid_name'] for r in test),
                        baseline_model_sha256=sha(a.baseline_dir/f'fit/fold_{f}_seal.json'),
                        auxiliary_mean=mean.tolist(), auxiliary_scale=scale.tolist(), coef=model.coef_[0].tolist(),
                        intercept=float(model.intercept_[0]), n_iter=model.n_iter_.tolist(),
                        calibration_scores={r['lipid_name']: float(p) for r, p in zip(calibration, scores)})
            name = f'{variant}/fold_{f}_seal.json'; write(fit/name, seal); files.append(name)
            # This fold's threshold and model are immutable before its test predictions.
            reported = [r for r in test if r['raw_solver_reported']]
            values = model.predict_proba(design_matrix(reported, names, baseline, mean, scale)[0])[:, 1]
            by_name = {r['lipid_name']: float(p) for r, p in zip(reported, values)}
            for r in test:
                score = by_name.get(r['lipid_name'])
                predicted[r['lipid_name']]['scores'][variant] = score
                predicted[r['lipid_name']]['retained'][variant] = bool(score is not None and threshold is not None and score >= threshold)
    write(fit/'predictions.json', list(predicted.values())); files.append('predictions.json')
    write(fit/'provenance.json', dict(status='COMPLETE_AWAITING_REVIEW', contract_sha256=sha(a.output/'prepared/contract.json'),
                                     fitted_models=20, baseline_refitted=False, files={n: sha(fit/n) for n in files}))
    print('TWENTY_FIXED_SMALL_MODELS_COMPLETE', flush=True)


def review_features(a, rows):
    """Second formula path: scalar cosine sums and E[s]^2/E[s^2], no fitting."""
    import numpy as np
    A, X, names, coeff, reported = load_observables(a.case_dir)
    means, norms = X.mean(axis=1), np.linalg.norm(A, axis=0)
    maximum = 0.
    for row in rows:
        if not row['raw_solver_reported']:
            assert row['auxiliary'] is None
            continue
        indices = row['reported_candidate_indices']
        others = np.array([i for i, n in enumerate(names) if n != row['lipid_name']])
        assert indices == [i for i in reported if names[i] == row['lipid_name']]
        abundance = sum(means[i] for i in indices)
        pressure = separation = 0.
        for i in indices:
            cs = np.clip((A[:, i] @ A[:, others])/(norms[i]*norms[others]), 0, 1)
            pressure += means[i]/abundance * float(np.dot(cs**2, means[others]))
            separation += means[i]/abundance * math.sqrt(max(0., 1-float(max(cs))**2))
        spatial = sum((X[i] for i in indices), np.zeros(X.shape[1]))
        fraction = float(spatial.mean()**2 / np.square(spatial).mean())
        expected = (separation, math.log10(abundance/(pressure+1e-12)),
                    abs(math.log10((sum(coeff[i] for i in indices)+1e-12)/(abundance+1e-12))), math.log10(fraction))
        for key, value in zip(FEATURES, expected):
            maximum = max(maximum, abs(value-row['auxiliary'][key]))
            # sqrt(1-cos^2) can have O(sqrt(eps)) error at identical columns.
            assert math.isclose(value, row['auxiliary'][key], rel_tol=1e-8, abs_tol=5e-8), (row['lipid_name'], key)
    return maximum


def review(a):
    import numpy as np
    rows, c = validate(a); fit = a.output/'fit'; p = read(fit/'provenance.json')
    assert p['contract_sha256'] == sha(a.output/'prepared/contract.json') and p['fitted_models'] == 20 and not p['baseline_refitted']
    expected_files = {'predictions.json'} | {f'{v}/fold_{f}_seal.json' for v in VARIANTS for f in range(5)}
    assert set(p['files']) == expected_files
    for name, h in p['files'].items():
        assert sha(fit/name) == h
    predictions = read(fit/'predictions.json'); idx = {r['lipid_name']: r for r in predictions}
    assert len(predictions) == len(idx) == len(rows) and set(idx) == {r['lipid_name'] for r in rows}
    old = {r['lipid_name']: r for r in read(a.baseline_dir/'fit/out_of_group_records.json')}
    for r in rows:
        got = idx[r['lipid_name']]
        assert got['fold'] == r['fold']
        assert set(got['scores']) == set(got['retained']) == {'BASELINE', *VARIANTS}
        assert got['scores']['BASELINE'] == old[r['lipid_name']]['joint_probability']
        assert got['retained']['BASELINE'] == old[r['lipid_name']]['joint_probability_0.01_retained']
    feature_error = review_features(a, rows)
    prediction_error = 0.
    for variant, names in VARIANTS.items():
        for f in range(5):
            seal = read(fit/f'{variant}/fold_{f}_seal.json')
            baseline = read(a.baseline_dir/f'fit/fold_{f}_seal.json')['model']
            assert seal['variant'] == variant and seal['fold'] == f and seal['auxiliary_features'] == list(names)
            assert seal['baseline_model_sha256'] == sha(a.baseline_dir/f'fit/fold_{f}_seal.json')
            train = [r for r in rows if r['raw_solver_reported'] and r['fold'] not in (f, (f+1)%5)]
            calibration = [r for r in rows if r['raw_solver_reported'] and r['fold'] == (f+1)%5]
            test = [r for r in rows if r['fold'] == f]
            sets = [set(r['lipid_name'] for r in rs) for rs in (train, calibration, test)]
            assert not sets[0]&sets[1] and not sets[0]&sets[2] and not sets[1]&sets[2]
            for key, ns in zip(('train_names', 'cal_names', 'test_names'), sets):
                assert seal[key] == sorted(ns)
            aux = np.array([[r['auxiliary'][k] for k in names] for r in train])
            assert np.allclose(aux.mean(axis=0), seal['auxiliary_mean'], rtol=1e-12, atol=1e-14)
            assert np.allclose(np.maximum(aux.std(axis=0),1e-12), seal['auxiliary_scale'], rtol=1e-12, atol=1e-14)
            assert all(0 < n < 2000 for n in seal['n_iter'])
            def independent_predict(r):
                z = [(math.log10(max(r[k], floor))-baseline['mean'][i])/baseline['scale'][i]
                     for i, (k, floor) in enumerate((('X_hat',1e-12),('rho_zero',1e-24)))]
                phi = [z[0],z[1],z[0]**2,z[0]*z[1],z[1]**2,float(r['rho_zero']==0)]
                phi += [(r['auxiliary'][k]-seal['auxiliary_mean'][i])/seal['auxiliary_scale'][i] for i,k in enumerate(names)]
                v = math.fsum(x*b for x,b in zip(phi,seal['coef']))+seal['intercept']
                return 1/(1+math.exp(-v)) if v >= 0 else math.exp(v)/(1+math.exp(v))
            assert set(seal['calibration_scores']) == sets[1]
            for r in calibration + [r for r in test if r['raw_solver_reported']]:
                score = seal['calibration_scores'][r['lipid_name']] if r in calibration else idx[r['lipid_name']]['scores'][variant]
                error = abs(independent_predict(r)-score); prediction_error = max(prediction_error,error)
                assert error <= 2e-12
            # Enumerate saved calibration ties independently; no optimizer or threshold search on TEST.
            best = None
            for value in sorted(set(seal['calibration_scores'].values()), reverse=True):
                selected = [r for r in calibration if seal['calibration_scores'][r['lipid_name']] >= value]
                if sum(not r['molecular_truth'] for r in selected)/len(selected) <= .01:
                    best = value
            assert best == seal['cutoff']
            for r in test:
                score = idx[r['lipid_name']]['scores'][variant]
                assert (score is not None) == r['raw_solver_reported']
                if score is not None:
                    assert math.isfinite(score) and 0 <= score <= 1
                assert idx[r['lipid_name']]['retained'][variant] == bool(r['raw_solver_reported'] and best is not None and score >= best)
    results = {v: metrics(rows, [r['lipid_name'] for r in rows if idx[r['lipid_name']]['retained'][v]]) for v in ('BASELINE', *VARIANTS)}
    assert results['BASELINE'] == read(a.baseline_dir/'review.json')['results_at_separate_CAL_1pct']['joint_probability']
    paired = {}
    baseline_names = {r['lipid_name'] for r in rows if idx[r['lipid_name']]['retained']['BASELINE']}
    truths = {r['lipid_name'] for r in rows if r['molecular_truth']}
    for v in VARIANTS:
        selected = {r['lipid_name'] for r in rows if idx[r['lipid_name']]['retained'][v]}
        paired[v] = dict(gained_TP=len((selected-baseline_names)&truths), lost_TP=len((baseline_names-selected)&truths),
                         gained_FP=len((selected-baseline_names)-truths), removed_FP=len((baseline_names-selected)-truths))
    target = {v: bool(r['retained'] and r['FDP'] <= .01 and r['all_truth_recall'] >= .8 and r['TP_retention'] >= .8) for v,r in results.items()}
    rr = [r for r in rows if r['raw_solver_reported']]
    diagnostics = {}
    for key in FEATURES:
        values = np.array([r['auxiliary'][key] for r in rr])
        diagnostics[key] = dict(by_truth={str(truth): np.quantile([r['auxiliary'][key] for r in rr if r['molecular_truth']==truth], [.1,.5,.9]).tolist() for truth in (True,False)},
                               pearson_with_log_X=float(np.corrcoef(values,[math.log10(max(r['X_hat'],1e-12)) for r in rr])[0,1]),
                               pearson_with_log_rho=float(np.corrcoef(values,[math.log10(max(r['rho_zero'],1e-24)) for r in rr])[0,1]))
    fold_results = {str(f): {v: metrics([r for r in rows if r['fold']==f], [r['lipid_name'] for r in rows if r['fold']==f and idx[r['lipid_name']]['retained'][v]]) for v in results} for f in range(5)}
    report = dict(status='CACHED_INDEPENDENT_REVIEW_PASS', results=results, paired_vs_baseline=paired, fold_results=fold_results,
                  point_target_met=target, primary_all_auxiliary_target_met=target['ALL_AUXILIARY'], diagnostics=diagnostics,
                  maximum_feature_reconstruction_error=feature_error, maximum_model_prediction_error=prediction_error,
                  no_models_refitted_in_review=True, frozen_baseline_exactly_reproduced=True,
                  same_exposed_case_development_only=True, independent_FDR_validation=False, outcome_driven_rerun=False,
                  contract_sha256=sha(a.output/'prepared/contract.json'), provenance_sha256=sha(fit/'provenance.json'))
    write(a.output/'review.json', report)
    print(json.dumps(dict(status=report['status'], results=results, target=target, paired=paired), indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare','run','review'))
    parser.add_argument('--case-dir', type=Path, required=True)
    parser.add_argument('--baseline-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    a = parser.parse_args()
    a.case_dir, a.baseline_dir, a.output = a.case_dir.resolve(), a.baseline_dir.resolve(), a.output.resolve()
    assert not a.output.is_relative_to(a.case_dir) and not a.output.is_relative_to(a.baseline_dir)
    {'prepare':prepare, 'run':run, 'review':review}[a.action](a)
