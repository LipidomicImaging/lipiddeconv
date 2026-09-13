"""Frozen, bounded development comparison of screening and reduced NNLS."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import warnings

for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = ''

ROOT = Path(__file__).resolve().parents[1]
TRAIN = ROOT / 'results/small_mismatch_nnls_first_case/case'
CHECK = ROOT / 'results/identity_confidence_joint_validation/cases/CAL1'
BASE_MODEL = ROOT / 'results/identity_confidence_joint_validation/model.json'
OUTPUT = ROOT / 'results/two_stage_spatial_donor_v1'
ARMS = ('BASELINE', 'AUGMENTED')
FEATURES = ('spatial_donor_log10_u1', 'spatial_donor_log10_u2')
CODE = (
    'analysis/run_two_stage_spatial_donor.py', 'analysis/spatial_donor_confidence.py',
    'analysis/refit_screened_nnls.py', 'tests/test_spatial_donor_confidence.py',
    'tests/test_refit_screened_nnls.py', 'tests/test_two_stage_spatial_donor.py',
    'analysis/run_nnls_solver_baseline.py',
    'analysis/run_small_mismatch_nnls_first_case.py',
    'analysis/run_identity_confidence_joint_validation.py',
    'docs/TWO_STAGE_SPATIAL_DONOR_DEVELOPMENT_V1.md',
)
SOURCE_FILES = ('prepared_arrays.npz', 'learned_arrays.npz', 'metadata.json',
                'molecular_records.json', 'candidate_records.json')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    path = Path(path)
    content = json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n'
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text(encoding='utf-8') != content:
            raise RuntimeError('IMMUTABLE_OUTPUT_CHANGED: ' + str(path))
        return
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(content)


def runtime():
    import numpy, scipy, sklearn
    return dict(python=sys.version, numpy=numpy.__version__, scipy=scipy.__version__,
                sklearn=sklearn.__version__, platform=sys.platform)


def sources():
    result = {}
    for label, directory in (('DEV_TRAIN', TRAIN), ('EXPOSED_DEV_CHECK', CHECK)):
        original = read(directory / 'result.json')
        expected = original.get('artifact_hashes', original.get('artifacts'))
        if not original['normal_completion'] or not original['all_outputs_finite']:
            raise RuntimeError('SOURCE_NOT_COMPLETE')
        hashes = {name: sha(directory / name) for name in SOURCE_FILES}
        if any(value != expected[name] for name, value in hashes.items()):
            raise RuntimeError('SOURCE_ARTIFACT_HASH_CHANGED')
        hashes['result.json'] = sha(directory / 'result.json')
        result[label] = dict(path=str(directory.relative_to(ROOT)), hashes=hashes)
    if sha(BASE_MODEL) != read(CHECK / 'result.json')['model_sha256']:
        raise RuntimeError('BASELINE_MODEL_HASH_CHANGED')
    result['baseline_model_sha256'] = sha(BASE_MODEL)
    return result


def freeze(out):
    if out.exists():
        raise RuntimeError('OUTPUT_ALREADY_EXISTS')
    binding = sources()
    model = read(BASE_MODEL)
    training = read(TRAIN / 'molecular_records.json')
    if sorted(r['lipid_name'] for r in training if r['raw_solver_reported']) != model['training_names']:
        raise RuntimeError('BASELINE_TRAIN_MEMBERSHIP_CHANGED')
    design = dict(version='TWO_STAGE_SPATIAL_DONOR_V1', sources=binding,
                  code={p: sha(ROOT / p) for p in CODE}, runtime=runtime(),
                  arms=list(ARMS), features=list(FEATURES),
                  candidate_pool_training_FDP=0.05, final_reporting_gate=0.001,
                  strict_endpoint=dict(FDP=0.01, recall=0.8),
                  extended_descriptive_endpoint=dict(FDP=0.05, recall=0.8),
                  fit_count=1, reduced_refit_count=2,
                  exposure='Both cases exposed; shared identities/templates; development only',
                  no_new_observation=True, no_HOLD=True, no_rho=True, no_GPU=True,
                  storage='All arrays and checkpoints retained locally; compact records in Git')
    write(out / 'design.json', design)
    write(out / 'design_seal.json', dict(status='FROZEN_BEFORE_NEW_FEATURES_AND_FIT',
                                       design_sha256=sha(out / 'design.json')))
    print('FROZEN', sha(out / 'design.json'), flush=True)


def validate(out):
    design = read(out / 'design.json')
    if sha(out / 'design.json') != read(out / 'design_seal.json')['design_sha256']:
        raise RuntimeError('DESIGN_CHANGED')
    if design['sources'] != sources() or design['runtime'] != runtime():
        raise RuntimeError('INPUT_OR_RUNTIME_CHANGED')
    if design['code'] != {p: sha(ROOT / p) for p in CODE}:
        raise RuntimeError('IMPLEMENTATION_CHANGED')
    return design


def observed(directory, with_spectra=False):
    import numpy as np
    names = read(directory / 'metadata.json')['lipid_name']
    with np.load(directory / 'prepared_arrays.npz', allow_pickle=False) as z:
        mask = z['mask'].copy()
        A = z['A_solver'].astype(np.float64) if with_spectra else None
        B = z['B'].copy() if with_spectra else None
    with np.load(directory / 'learned_arrays.npz', allow_pickle=False) as z:
        X = z['X_hat'][:, mask].astype(np.float64)
    if X.shape != (391, 15837) or len(names) != 391 or mask.dtype != bool:
        raise RuntimeError('SOURCE_DIMENSION_CHANGED')
    if not np.isfinite(X).all() or (X < 0).any():
        raise RuntimeError('INVALID_CACHED_X')
    for i, row in enumerate(read(directory / 'candidate_records.json')):
        mu = float(X[i].mean())
        if row['candidate_index'] != i or row['lipid_name'] != names[i]:
            raise RuntimeError('CANDIDATE_ORDER_CHANGED')
        if not math.isclose(mu, row['X_hat'], rel_tol=1e-12, abs_tol=1e-14):
            raise RuntimeError('CACHED_MEAN_MISMATCH')
        if row['raw_solver_reported'] != (mu > 0.001):
            raise RuntimeError('SOURCE_GATE_MISMATCH')
    return names, X, mask, A, B


def feature_pack(directory):
    from spatial_donor_confidence import compute_spatial_donor_confidence
    names, X, _, _, _ = observed(directory)
    return compute_spatial_donor_confidence(X, names)


def feature_map(pack):
    return {r['molecular_name']: r for r in pack['records']}


def matrix(rows, pack, base, auxiliary=None):
    import numpy as np
    import run_identity_confidence_joint_validation as prior
    original = prior.transform(rows, base)[0]
    values = np.array([[feature_map(pack)[r['lipid_name']][k] for k in FEATURES] for r in rows])
    if auxiliary is None:
        auxiliary = dict(mean=values.mean(axis=0).tolist(),
                         scale=np.maximum(values.std(axis=0), 1e-12).tolist())
    return np.column_stack((original, (values - auxiliary['mean']) / auxiliary['scale'])), auxiliary


def probabilities(rows, pack, model):
    import numpy as np
    from scipy.special import expit
    import run_identity_confidence_joint_validation as prior
    if model['arm'] == 'BASELINE':
        return np.array([prior.predict(r, model['base']) for r in rows], dtype=float)
    x, _ = matrix(rows, pack, model['base'], model['auxiliary'])
    return expit(x @ np.array(model['coef']) + model['intercept'])


def development_cutoff(rows, scores):
    buckets = {}
    for r, score in zip(rows, scores):
        buckets.setdefault(float(score), []).append(bool(r['molecular_truth']))
    tp = fp = 0
    best = None
    for score in sorted(buckets, reverse=True):
        tp += sum(buckets[score])
        fp += len(buckets[score]) - sum(buckets[score])
        if fp / (tp + fp) <= 0.05:
            item = dict(cutoff=score, TP=tp, FP=fp, FDP=fp / (tp + fp))
            if best is None or (tp, -fp, score) > (best['TP'], -best['FP'], best['cutoff']):
                best = item
    return best or dict(cutoff=None, TP=0, FP=0, FDP=None)


def prepare(out):
    validate(out)
    if (out / 'prepared_seal.json').exists():
        raise RuntimeError('PREPARATION_ALREADY_COMPLETE')
    from sklearn.linear_model import LogisticRegression
    from sklearn.exceptions import ConvergenceWarning
    rows = [r for r in read(TRAIN / 'molecular_records.json') if r['raw_solver_reported']]
    base = read(BASE_MODEL)
    reservation = out / 'prepared/training_reservation.json'
    if (out / 'model_seal.json').exists():
        seal = read(out / 'model_seal.json')
        if seal['design_sha256'] != sha(out / 'design.json') or any(sha(out / p) != h for p, h in seal['files'].items()):
            raise RuntimeError('SAVED_TRAINING_SEAL_CHANGED')
        models = {arm: read(out / f'prepared/{arm}_model.json') for arm in ARMS}
    else:
        if reservation.exists():
            raise RuntimeError('INCOMPLETE_TRAINING_PRESERVED_NO_AUTOMATIC_REFIT')
        train_pack = feature_pack(TRAIN)
        write(out / 'prepared/train_features.json', train_pack)
        x, auxiliary = matrix(rows, train_pack, base)
        write(reservation, dict(status='ONE_AUGMENTED_FIT_RESERVED',
              design_sha256=sha(out / 'design.json'), training_feature_sha256=sha(out / 'prepared/train_features.json')))
        with warnings.catch_warnings():
            warnings.simplefilter('error', ConvergenceWarning)
            estimator = LogisticRegression(C=1., solver='lbfgs', max_iter=2000, tol=1e-8).fit(
                x, [int(r['molecular_truth']) for r in rows])
        models = {
            'BASELINE': dict(arm='BASELINE', base=base),
            'AUGMENTED': dict(arm='AUGMENTED', base=base, auxiliary=auxiliary,
                              coef=estimator.coef_[0].tolist(), intercept=float(estimator.intercept_[0]),
                              n_iter=estimator.n_iter_.tolist()),
        }
        for arm, model in models.items():
            scores = probabilities(rows, train_pack, model)
            model['training_cutoff'] = development_cutoff(rows, scores)
            model['training_names'] = [r['lipid_name'] for r in rows]
            write(out / f'prepared/{arm}_model.json', model)
            write(out / f'prepared/{arm}_train_scores.json', [dict(lipid_name=r['lipid_name'], score=float(s)) for r, s in zip(rows, scores)])
        model_files = ['prepared/train_features.json', 'prepared/training_reservation.json']
        model_files += [f'prepared/{arm}_{suffix}.json' for arm in ARMS for suffix in ('model', 'train_scores')]
        write(out / 'model_seal.json', dict(design_sha256=sha(out / 'design.json'),
              files={p: sha(out / p) for p in model_files},
              status='FROZEN_BEFORE_DEV_CHECK_FEATURES_AND_SELECTION'))
    check_pack = feature_pack(CHECK)
    write(out / 'prepared/check_features.json', check_pack)
    # Copy only deployment-observable fields into the selector.
    fields = ('lipid_name', 'X_hat', 'rho_zero', 'raw_solver_reported', 'candidate_indices')
    check_rows = [{k: r[k] for k in fields} for r in read(CHECK / 'molecular_records.json') if r['raw_solver_reported']]
    for arm, model in models.items():
        scores = probabilities(check_rows, check_pack, model)
        cutoff = model['training_cutoff']['cutoff']
        selected = [r['lipid_name'] for r, s in zip(check_rows, scores) if cutoff is not None and s >= cutoff]
        names = read(CHECK / 'metadata.json')['lipid_name']
        indices = [i for i, n in enumerate(names) if n in set(selected)]
        selection = dict(arm=arm, model_sha256=sha(out / f'prepared/{arm}_model.json'), cutoff=cutoff,
                         selected_names=selected, retained_indices=indices,
                         scores=[dict(lipid_name=r['lipid_name'], score=float(s), selected=r['lipid_name'] in selected)
                                 for r, s in zip(check_rows, scores)],
                         selection_inputs='Original reported scores/nominal alias names/cached X_hat-derived features; no truth')
        write(out / f'prepared/{arm}_selection.json', selection)
    files = ['model_seal.json', 'prepared/train_features.json', 'prepared/check_features.json']
    files += ['prepared/training_reservation.json']
    files += [f'prepared/{arm}_{kind}.json' for arm in ARMS for kind in ('model', 'selection', 'train_scores')]
    write(out / 'prepared_seal.json', dict(status='FROZEN_BEFORE_REDUCED_REFITS',
          design_sha256=sha(out / 'design.json'), files={p: sha(out / p) for p in files}))
    print('PREPARED', {a: len(read(out / f'prepared/{a}_selection.json')['selected_names']) for a in ARMS}, flush=True)


def validate_prepared(out):
    validate(out)
    seal = read(out / 'prepared_seal.json')
    if seal['design_sha256'] != sha(out / 'design.json') or any(sha(out / p) != h for p, h in seal['files'].items()):
        raise RuntimeError('PREPARED_BINDING_CHANGED')


def review_prepared(out):
    """Reconstruct saved predictions with scalar arithmetic, without fitting."""
    validate_prepared(out)
    check_rows = {r['lipid_name']: r for r in read(CHECK / 'molecular_records.json')}
    features = feature_map(read(out / 'prepared/check_features.json'))
    maximum_error = 0.
    for arm in ARMS:
        model = read(out / f'prepared/{arm}_model.json')
        selection = read(out / f'prepared/{arm}_selection.json')
        if {r['lipid_name'] for r in selection['scores']} != {n for n, r in check_rows.items() if r['raw_solver_reported']}:
            raise RuntimeError('SCORE_MEMBERSHIP_CHANGED')
        for score in selection['scores']:
            r = check_rows[score['lipid_name']]
            base = model['base']
            z = [(math.log10(max(r[k], floor)) - base['mean'][i]) / base['scale'][i]
                 for i, (k, floor) in enumerate((('X_hat', 1e-12), ('rho_zero', 1e-24)))]
            phi = [z[0], z[1], z[0] ** 2, z[0] * z[1], z[1] ** 2, float(r['rho_zero'] == 0)]
            if arm == 'AUGMENTED':
                phi += [(features[r['lipid_name']][k] - model['auxiliary']['mean'][i]) / model['auxiliary']['scale'][i]
                        for i, k in enumerate(FEATURES)]
                coefficients, intercept = model['coef'], model['intercept']
            else:
                coefficients, intercept = base['coef'], base['intercept']
            logit = math.fsum(a * b for a, b in zip(phi, coefficients)) + intercept
            p = 1 / (1 + math.exp(-logit)) if logit >= 0 else math.exp(logit) / (1 + math.exp(logit))
            maximum_error = max(maximum_error, abs(p - score['score']))
            flag = selection['cutoff'] is not None and score['score'] >= selection['cutoff']
            if flag != score['selected'] or flag != (r['lipid_name'] in selection['selected_names']):
                raise RuntimeError('SELECTION_THRESHOLD_MISMATCH')
        if selection['cutoff'] != model['training_cutoff']['cutoff']:
            raise RuntimeError('CUTOFF_NOT_TRAIN_BOUND')
        train_rows = {r['lipid_name']: r for r in read(TRAIN / 'molecular_records.json') if r['raw_solver_reported']}
        train_scores = read(out / f'prepared/{arm}_train_scores.json')
        if set(train_rows) != {r['lipid_name'] for r in train_scores} or len(train_rows) != len(train_scores):
            raise RuntimeError('TRAIN_SCORE_MEMBERSHIP_CHANGED')
        from scipy.special import expit
        import numpy as np
        train_matrix = matrix(list(train_rows.values()), read(out / 'prepared/train_features.json'), model['base'],
                              model.get('auxiliary'))[0]
        if arm == 'BASELINE':
            reconstructed = expit(train_matrix[:, :6] @ model['base']['coef'] + model['base']['intercept'])
        else:
            reconstructed = expit(train_matrix @ model['coef'] + model['intercept'])
        saved = {r['lipid_name']: r['score'] for r in train_scores}
        maximum_error = max(maximum_error, max(abs(float(p)-saved[n]) for n, p in zip(train_rows, reconstructed)))
        eligible = []
        for threshold in {r['score'] for r in train_scores}:
            names_at_t = {r['lipid_name'] for r in train_scores if r['score'] >= threshold}
            true_at_t = sum(train_rows[n]['molecular_truth'] for n in names_at_t)
            false_at_t = len(names_at_t)-true_at_t
            if false_at_t <= .05 * len(names_at_t):
                eligible.append((true_at_t, -false_at_t, threshold))
        winner = max(eligible) if eligible else (0, 0, None)
        saved_cutoff = model['training_cutoff']
        if (saved_cutoff['TP'], -saved_cutoff['FP'], saved_cutoff['cutoff']) != winner:
            raise RuntimeError('TRAIN_TIE_PRESERVING_OPTIMAL_CUTOFF_MISMATCH')
    if maximum_error > 2e-12:
        raise RuntimeError('SAVED_MODEL_PREDICTION_MISMATCH')
    write(out / 'prepared_review.json', dict(status='CACHED_PREDICTION_AND_SELECTION_REVIEW_PASS',
          max_probability_error=maximum_error, optimization_performed=False,
          prepared_seal_sha256=sha(out / 'prepared_seal.json')))


def refit(out, arm):
    from refit_screened_nnls import fit_screened
    validate_prepared(out)
    review_prepared(out)
    selection = read(out / f'prepared/{arm}_selection.json')
    names, _, mask, A, B = observed(CHECK, with_spectra=True)
    expected = [i for i, n in enumerate(names) if n in set(selection['selected_names'])]
    if expected != selection['retained_indices']:
        raise RuntimeError('ALIAS_POOL_MISMATCH')
    result = fit_screened(A, B, mask, expected, out / 'refit' / arm, workers=4, block_size=250,
                         source_binding=dict(design_sha256=sha(out / 'design.json'),
                                             selection_sha256=sha(out / f'prepared/{arm}_selection.json'),
                                             source_prepared_sha256=sha(CHECK / 'prepared_arrays.npz')))
    write(out / f'{arm}_completion.json', dict(arm=arm,
          arrays_path=str(Path(result['arrays_path']).relative_to(out)),
          receipt_path=str(Path(result['receipt_path']).relative_to(out)),
          arrays_sha256=sha(result['arrays_path']), receipt_sha256=sha(result['receipt_path']),
          selection_sha256=sha(out / f'prepared/{arm}_selection.json')))
    print(arm, 'REFIT_COMPLETE', flush=True)


def metrics(rows, selected):
    chosen = set(selected)
    truths = {r['lipid_name'] for r in rows if r['molecular_truth']}
    reportable = {r['lipid_name'] for r in rows if r['reportable_truth']}
    raw = {r['lipid_name'] for r in rows if r['raw_solver_reported']}
    tp = len(chosen & truths); fp = len(chosen - truths)
    return dict(TP=tp, FP=fp, FN=len(truths)-tp, FDP=fp / len(chosen) if chosen else None,
                all_truth_recall=tp / len(truths), reportable_truth_recall=len(chosen & reportable) / len(reportable),
                TP_retention=tp / len(raw & truths), solver_misses=len(truths - raw),
                retained=len(chosen), all_truth_count=len(truths), raw_reported_truth_count=len(raw & truths))


def review(out, arm):
    import numpy as np
    validate_prepared(out)
    selection = read(out / f'prepared/{arm}_selection.json')
    completion = read(out / f'{arm}_completion.json')
    if completion['selection_sha256'] != sha(out / f'prepared/{arm}_selection.json'):
        raise RuntimeError('COMPLETION_SELECTION_CHANGED')
    for key in ('arrays', 'receipt'):
        if sha(out / completion[key + '_path']) != completion[key + '_sha256']:
            raise RuntimeError('REFIT_ARTIFACT_CHANGED')
    names, _, mask, A, B = observed(CHECK, with_spectra=True)
    with np.load(out / completion['arrays_path'], allow_pickle=False) as z:
        X = z['X_hat'].copy(); Bhat = z['B_hat'].copy()
    indices = selection['retained_indices']; selected = set(selection['selected_names'])
    receipt_path = out / completion['receipt_path']
    receipt = read(receipt_path)
    binding_path = receipt_path.parent / 'binding.json'
    binding = read(binding_path)
    if receipt['status'] != 'COMPLETE' or not receipt['all_outputs_finite'] or not receipt['all_pixels_checked_before_float32']:
        raise RuntimeError('INCOMPLETE_REFIT_RECEIPT')
    if sha(binding_path) != receipt['binding_sha256'] or binding['fingerprint'] != receipt['fingerprint']:
        raise RuntimeError('REFIT_SOURCE_BINDING_CHANGED')
    expected_binding = dict(design_sha256=sha(out / 'design.json'),
                            selection_sha256=sha(out / f'prepared/{arm}_selection.json'),
                            source_prepared_sha256=sha(CHECK / 'prepared_arrays.npz'))
    if binding['scientific']['source_binding'] != expected_binding or receipt['retained_indices'] != indices:
        raise RuntimeError('REFIT_SELECTION_BINDING_CHANGED')
    blocks = receipt_path.parent / 'nnls_blocks'
    expected_files = {f'block_{start:06d}_{min(start+250, int(mask.sum())):06d}.{ext}'
                      for start in range(0, int(mask.sum()), 250) for ext in ('json', 'npz')}
    if {p.name for p in blocks.iterdir()} != expected_files or set(receipt['block_hashes']) != expected_files:
        raise RuntimeError('BLOCK_MEMBERSHIP_MISMATCH')
    for name in sorted(expected_files):
        if sha(blocks / name) != receipt['block_hashes'][name]:
            raise RuntimeError('BLOCK_HASH_CHANGED')
        if name.endswith('.json'):
            block = read(blocks / name)
            if block['fingerprint'] != receipt['fingerprint'] or block['retained_indices'] != indices:
                raise RuntimeError('BLOCK_SOURCE_CHANGED')
            if not block['all_pixels_checked_before_float32'] or block['KKT']['max_bound_ratio'] > 1.:
                raise RuntimeError('BLOCK_KKT_INVALID')
            with np.load(blocks / name.replace('.json', '.npz'), allow_pickle=False) as z:
                if not np.array_equal(z['X_hat'], X[indices][:, mask][:, block['start']:block['stop']]):
                    raise RuntimeError('BLOCK_FINAL_ARRAY_MISMATCH')
    excluded = sorted(set(range(len(names))) - set(indices))
    if X.shape != (391, *mask.shape) or Bhat.shape != B.shape or X.dtype != np.float32:
        raise RuntimeError('OUTPUT_DIMENSION_CHANGED')
    if not np.isfinite(X).all() or not np.isfinite(Bhat).all() or (X < 0).any():
        raise RuntimeError('NONFINITE_RESULT')
    if np.any(X[excluded]) or np.any(X[:, ~mask]) or np.any(Bhat[:, ~mask]):
        raise RuntimeError('OUTPUT_MEMBERSHIP_OR_BACKGROUND_CHANGED')
    reconstruction = (A @ X[:, mask].astype(float)).astype(np.float32)
    if not np.allclose(reconstruction, Bhat[:, mask], rtol=2e-6, atol=1e-8):
        raise RuntimeError('RECONSTRUCTION_MISMATCH')
    means = X[:, mask].mean(axis=1, dtype=np.float64)
    # Independent evaluator follows candidate-first reporting without the refit helper.
    reported = {names[i] for i in range(len(names)) if means[i] > 0.001}
    if not reported <= selected:
        raise RuntimeError('FINAL_NOT_SUBSET_OF_POOL')
    original = read(CHECK / 'molecular_records.json')
    raw = {r['lipid_name'] for r in original if r['raw_solver_reported']}
    truths = {r['lipid_name'] for r in original if r['molecular_truth']}
    if not selected <= raw:
        raise RuntimeError('POOL_CONTAINS_UNREPORTED_IDENTITY')
    scores = {r['lipid_name']: r['score'] for r in selection['scores']}
    rows = []
    for r in original:
        n = r['lipid_name']; candidate_indices = [i for i, name in enumerate(names) if name == n]
        final_indices = [i for i in candidate_indices if means[i] > 0.001]
        rows.append(dict(lipid_name=n, molecular_truth=r['molecular_truth'], reportable_truth=r['reportable_truth'],
             raw_reported=r['raw_solver_reported'], raw_abundance=r['X_hat'], original_full_library_rho=r['rho_zero'],
             score=scores.get(n), screened=n in selected, final_reported=n in reported,
             candidate_indices=candidate_indices, final_reported_candidate_indices=final_indices,
             final_abundance=float(means[final_indices].sum()) if final_indices else 0.,
             screening_true_loss=n in ((raw & truths) - selected),
             refit_true_loss=n in ((selected & truths) - reported)))
    candidate_rows = [dict(candidate_index=i, lipid_name=n, selected=n in selected,
                           final_abundance=float(means[i]), final_reported=bool(means[i] > .001)) for i, n in enumerate(names)]
    result = dict(arm=arm, status='CACHED_REVIEW_PASS',
                  raw=metrics(original, raw), screened=metrics(original, selected), final=metrics(original, reported),
                  screening_true_losses=sorted((raw & truths) - selected),
                  additional_refit_true_losses=sorted((selected & truths) - reported),
                  refit_removed_false_identities=sorted((selected - truths) - reported),
                  refit_removed_true_identities=sorted((selected & truths) - reported),
                  final_endpoint_1pct_80=bool(reported and metrics(original, reported)['FDP'] <= .01 and metrics(original, reported)['all_truth_recall'] >= .8),
                  final_endpoint_5pct_80=bool(reported and metrics(original, reported)['FDP'] <= .05 and metrics(original, reported)['all_truth_recall'] >= .8),
                  relative_reconstruction_residual=float(np.linalg.norm(Bhat[:, mask].astype(float)-B[:, mask]) / np.linalg.norm(B[:, mask].astype(float))),
                  numerical_review='Finite arrays, original source hashes, all-index membership, original candidate gate and cached reconstruction; no optimization',
                  exposure='Development only; no independent FDR or real-MSI claim',
                  arrays_sha256=completion['arrays_sha256'], receipt_sha256=completion['receipt_sha256'])
    write(out / f'{arm}_molecular_records.json', rows)
    write(out / f'{arm}_candidate_records.json', candidate_rows)
    write(out / f'{arm}_review.json', result)
    print(arm, json.dumps({k: result[k] for k in ('screened', 'final')}, ensure_ascii=False), flush=True)


def summarize(out):
    validate_prepared(out)
    arms = {arm: read(out / f'{arm}_review.json') for arm in ARMS}
    final_sets = {arm: {r['lipid_name'] for r in read(out / f'{arm}_molecular_records.json') if r['final_reported']} for arm in ARMS}
    comparison = dict(status='DEVELOPMENT_COMPARISON_COMPLETE', arms=arms,
          augmented_final_added=sorted(final_sets['AUGMENTED']-final_sets['BASELINE']),
          augmented_final_removed=sorted(final_sets['BASELINE']-final_sets['AUGMENTED']),
          independent_validation=False, old_CAL1_NO_GO_unchanged=True, further_experiment_started=False)
    write(out / 'comparison.json', comparison)
    print('COMPARISON_COMPLETE', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('freeze', 'prepare', 'review_prepared', 'refit', 'review', 'summarize'))
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--arm', choices=ARMS)
    args = parser.parse_args()
    if args.command in ('refit', 'review'):
        if args.arm is None:
            parser.error('--arm is required')
        globals()[args.command](args.output.resolve(), args.arm)
    else:
        globals()[args.command](args.output.resolve())
