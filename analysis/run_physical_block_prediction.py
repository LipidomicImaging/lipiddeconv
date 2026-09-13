"""Frozen mean-spectrum physical-block prediction and downstream development."""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback
import warnings

for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
CODE = ('analysis/run_physical_block_prediction.py', 'analysis/physical_block_prediction.py',
        'analysis/review_physical_block_prediction.py', 'analysis/run_nnls_solver_baseline.py',
        'analysis/run_small_mismatch_nnls_first_case.py', 'analysis/run_identity_confidence_joint_validation.py',
        'analysis/refit_screened_nnls.py', 'tests/test_physical_block_prediction.py',
        'tests/test_review_physical_block_prediction.py', 'tests/test_run_physical_block_prediction.py',
        'docs/PHYSICAL_BLOCK_PREDICTION_EXECUTION_V1.md')
CASES = ('DEV_TRAIN', 'CHECK')
FEATURES = ('predictive_gain', 'positive_block_fraction')


def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def write(p, value):
    p = Path(p)
    data = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n'
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.exists():
        require(p.read_text(encoding='utf-8') == data, 'IMMUTABLE_OUTPUT_CHANGED:' + str(p))
        return
    with p.open('x', encoding='utf-8', newline='\n') as f:
        f.write(data)


def save_npz(p, **arrays):
    import numpy as np
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('xb') as f:
        np.savez_compressed(f, **arrays)


def runtime():
    import numpy, scipy, sklearn
    return dict(numpy=numpy.__version__, scipy=scipy.__version__, sklearn=sklearn.__version__)


def prepare(a):
    import numpy as np
    from physical_block_prediction import build_blocks
    from review_physical_block_prediction import review_block_manifest
    require(not a.output.exists(), 'OUTPUT_EXISTS')
    require(runtime() == dict(numpy='2.1.3', scipy='1.15.3', sklearn='1.6.1'), 'RUNTIME_CHANGED')
    expected = read(a.expected)
    for path, digest in expected['source_hashes'].items():
        require(sha(path) == digest, 'SOURCE_CHANGED:' + path)
    with np.load(expected['components'], allow_pickle=False) as z:
        components, owner, nominal = z['components'], z['owner'], z['A']
    names = read(Path(expected['cases']['DEV_TRAIN']) / 'metadata.json')['lipid_name']
    manifest = build_blocks(components, owner, names)
    require(manifest['block_count'] == 34 and len(names) == 391 and len(manifest['molecular_names']) == 377,
            'REAL_UNIVERSE_CHANGED')
    manifest_review = review_block_manifest(nominal, components, owner, names, manifest['blocks'])
    a.output.mkdir(parents=True)
    write(a.output / 'expected_sources.json', expected)
    write(a.output / 'block_manifest.json', manifest)
    write(a.output / 'block_manifest_review.json', manifest_review)
    aggregates = {}
    for label in CASES:
        directory = Path(expected['cases'][label])
        require(read(directory / 'metadata.json')['lipid_name'] == names, 'CANDIDATE_ORDER_CHANGED')
        with np.load(directory / 'prepared_arrays.npz', allow_pickle=False) as z:
            A, B, mask = z['A_solver'].astype(np.float64), z['B'], z['mask']
            require(A.shape == (1084, 391) and B.shape == (1084, 200, 90)
                    and mask.dtype == bool and int(mask.sum()) == 15837, 'ARRAY_SHAPE_CHANGED')
            require(np.array_equal(A, nominal.astype(np.float64)), 'NOMINAL_COMPONENT_LIBRARY_CHANGED')
            require(np.isfinite(B).all() and np.all(B[:, ~mask] == 0), 'INVALID_OBSERVATION')
            b = B[:, mask].mean(axis=1, dtype=np.float64)
        require(float(b @ b) > 0, 'ZERO_SIGNAL')
        save_npz(a.output / 'prepared' / (label + '.npz'), A=A, b=b)
        rows = read(directory / 'molecular_records.json')
        require([r['lipid_name'] for r in rows] == manifest['molecular_names'], 'MOLECULAR_ORDER_CHANGED')
        write(a.output / 'prepared' / (label + '_records.json'), rows)
        aggregates[label] = dict(input_sha256=sha(a.output / 'prepared' / (label + '.npz')),
                                 records_sha256=sha(a.output / 'prepared' / (label + '_records.json')),
                                 foreground_pixels=15837, b_squared_norm=float(b @ b))
    write(a.output / 'baseline_model.json', read(expected['baseline_model']))
    design = dict(version='PHYSICAL_BLOCK_PREDICTION_EXECUTION_V1', runtime=runtime(),
                  source_hashes=expected['source_hashes'], code={p: sha(ROOT / p) for p in CODE},
                  expected_sources_sha256=sha(a.output / 'expected_sources.json'),
                  cases=aggregates, blocks_sha256=sha(a.output / 'block_manifest.json'),
                  baseline_model_sha256=sha(a.output / 'baseline_model.json'), workers=4,
                  case_wall_seconds=14400, case_block_bytes_limit=2 * 1024 ** 3,
                  block_count=34, models_per_block=378, score_positive_epsilon_multiplier=128,
                  features=list(FEATURES), candidate_pool_recall=.95, final_training_FDP=.01,
                  exposure='Both cases exposed; original data blocks are not independent acquisitions',
                  final_rule='pool AND frozen effective final score AND refit original candidate gate',
                  no_new_observation=True, no_rho=True, no_GPU=True, no_deletion=True)
    write(a.output / 'design.json', design)
    write(a.output / 'design_seal.json', dict(status='FROZEN_BEFORE_REAL_SOLVES', design_sha256=sha(a.output / 'design.json')))
    print('PREPARED', sha(a.output / 'design.json'), flush=True)


def validate(out, remote=False):
    design = read(out / 'design.json')
    require(sha(out / 'design.json') == read(out / 'design_seal.json')['design_sha256'], 'DESIGN_CHANGED')
    require({p: sha(ROOT / p) for p in CODE} == design['code'], 'CODE_CHANGED')
    require(sha(out / 'expected_sources.json') == design['expected_sources_sha256'], 'SOURCE_MAPPING_CHANGED')
    require(sha(out / 'block_manifest.json') == design['blocks_sha256'], 'BLOCKS_CHANGED')
    require(sha(out / 'baseline_model.json') == design['baseline_model_sha256'], 'BASE_MODEL_CHANGED')
    for case, item in design['cases'].items():
        require(sha(out / 'prepared' / (case + '.npz')) == item['input_sha256'], 'AGGREGATE_CHANGED')
        require(sha(out / 'prepared' / (case + '_records.json')) == item['records_sha256'], 'RECORDS_CHANGED')
    if remote:
        require(runtime() == design['runtime'], 'RUNTIME_CHANGED')
        for path, digest in design['source_hashes'].items():
            require(sha(path) == digest, 'ORIGINAL_SOURCE_CHANGED:' + path)
    return design


def case_input(out, case):
    import numpy as np
    with np.load(out / 'prepared' / (case + '.npz'), allow_pickle=False) as z:
        A, b = z['A'], z['b']
    manifest = read(out / 'block_manifest.json')
    return A, b, manifest['candidate_names'], manifest


def _block_task(out_string, case, block_index):
    import numpy as np
    from threadpoolctl import threadpool_limits
    from physical_block_prediction import solve_block
    out = Path(out_string)
    A, b, names, manifest = case_input(out, case)
    stem = out / 'cases' / case / 'blocks' / f'block_{block_index:03d}'
    source_hash = sha(out / 'prepared' / (case + '.npz'))
    if stem.with_suffix('.json').exists():
        record = read(stem.with_suffix('.json'))
        require(record['source_sha256'] == source_hash and record['design_sha256'] == sha(out / 'design.json')
                and record['arrays_sha256'] == sha(stem.with_suffix('.npz'))
                and record['block_index'] == block_index, 'CACHED_BLOCK_CHANGED')
        return record
    require(not stem.with_suffix('.npz').exists(), 'PARTIAL_BLOCK_ARRAY_PRESERVED')
    start = time.monotonic()
    with threadpool_limits(limits=1):
        result = solve_block(A, b, names, manifest['blocks'][block_index]['channel_indices'])
    save_npz(stem.with_suffix('.npz'), **result['arrays'])
    record = dict(block_index=block_index, case=case, source_sha256=source_hash,
                  design_sha256=sha(out / 'design.json'), arrays_sha256=sha(stem.with_suffix('.npz')),
                  elapsed_seconds=time.monotonic() - start, **result['diagnostics'])
    write(stem.with_suffix('.json'), record)
    return record


def features_from_blocks(A, b, names, manifest, block_arrays):
    import numpy as np
    molecular = list(dict.fromkeys(names))
    require(len(block_arrays) == len(manifest['blocks']), 'BLOCK_COUNT_MISMATCH')
    total = np.zeros(len(molecular))
    positives = np.zeros(len(molecular), dtype=int)
    counts_supported = np.zeros(len(molecular), dtype=int)
    per_group = [[] for _ in molecular]
    signal = float(b @ b)
    require(signal > 0 and math.isfinite(signal), 'INVALID_SIGNAL')
    for block, arrays in zip(manifest['blocks'], block_arrays):
        require(arrays['deleted_names'].tolist() == molecular, 'DELETION_ORDER_CHANGED')
        held = np.array(block['channel_indices'], dtype=int)
        train = np.setdiff1d(np.arange(len(b)), held)
        full = float(arrays['full_held_loss'])
        deleted = np.asarray(arrays['deleted_held_loss'])
        delta = deleted - full
        tau = 128 * np.finfo(np.float64).eps * max(A.shape) * np.maximum(np.maximum(deleted, full), float(b[held] @ b[held]))
        total += delta
        for g, name in enumerate(molecular):
            columns = [j for j, n in enumerate(names) if n == name]
            supported = bool(np.any(A[np.ix_(held, columns)] > 0) and np.any(A[np.ix_(train, columns)] > 0))
            counts_supported[g] += supported
            positives[g] += supported and delta[g] > tau[g]
            per_group[g].append(dict(block_index=block['block_index'], delta=float(delta[g]),
                                      positive_tolerance=float(tau[g]), supported=supported,
                                      positive=bool(supported and delta[g] > tau[g])))
    return [dict(molecular_name=name, predictive_gain=float(total[g] / signal),
                 positive_block_fraction=float(positives[g] / counts_supported[g]) if counts_supported[g] else 0.,
                 supported_block_count=int(counts_supported[g]), positive_block_count=int(positives[g]),
                 unsupported=bool(counts_supported[g] == 0), blocks=per_group[g]) for g, name in enumerate(molecular)]


def run_case(a):
    import numpy as np
    design = validate(a.output, remote=True)
    require_saved(a.output, 'design')
    if a.case == 'CHECK':
        require_saved(a.output, 'DEV_TRAIN')
        require_saved(a.output, 'model')
    case_dir = a.output / 'cases' / a.case
    require(not (case_dir / 'failure.json').exists(), 'PREVIOUS_FAILURE_REQUIRES_TECHNICAL_REVIEW')
    if (case_dir / 'completion.json').exists():
        print('CASE_ALREADY_COMPLETE', a.case, flush=True)
        return
    start = time.monotonic()
    try:
        first = _block_task(str(a.output), a.case, 0)
        timing = case_dir / 'timing.json'
        if not timing.exists():
            write(timing, dict(first_block_seconds=first['elapsed_seconds'], retained=True,
                               rough_remaining_seconds=first['elapsed_seconds'] * 33 / design['workers'],
                               projection_is_not_guarantee=True))
        print('FIRST_BLOCK_RETAINED', a.case, first['elapsed_seconds'], flush=True)
        require(time.monotonic() - start <= design['case_wall_seconds'], 'WALL_BUDGET_EXCEEDED')
        pool = ProcessPoolExecutor(max_workers=design['workers'])
        try:
            futures = {pool.submit(_block_task, str(a.output), a.case, k): k for k in range(1, 34)}
            remaining = design['case_wall_seconds'] - (time.monotonic() - start)
            for future in as_completed(futures, timeout=max(remaining, .01)):
                record = future.result()
                print('BLOCK_COMPLETE', a.case, record['block_index'], round(record['elapsed_seconds'], 3), flush=True)
                require(time.monotonic() - start <= design['case_wall_seconds'], 'WALL_BUDGET_EXCEEDED')
                size = sum(p.stat().st_size for p in (case_dir / 'blocks').iterdir())
                require(size <= design['case_block_bytes_limit'], 'STORAGE_BUDGET_EXCEEDED')
        except BaseException:
            for future in futures:
                future.cancel()
            # Running processes must stop on numerical failure/budget, preserving partial files.
            for process in pool._processes.values():
                process.terminate()
            pool.shutdown(wait=True, cancel_futures=True)
            raise
        else:
            pool.shutdown(wait=True)
        A, b, names, manifest = case_input(a.output, a.case)
        all_arrays = []
        artifacts = {}
        for k in range(34):
            for suffix in ('.npz', '.json'):
                p = case_dir / 'blocks' / (f'block_{k:03d}' + suffix)
                artifacts[str(p.relative_to(case_dir)).replace('\\', '/')] = sha(p)
            with np.load(case_dir / 'blocks' / f'block_{k:03d}.npz', allow_pickle=False) as z:
                all_arrays.append({key: z[key] for key in z.files})
        write(case_dir / 'features.json', features_from_blocks(A, b, names, manifest, all_arrays))
        artifacts['features.json'] = sha(case_dir / 'features.json')
        artifacts['timing.json'] = sha(case_dir / 'timing.json')
        write(case_dir / 'completion.json', dict(status='COMPLETE', case=a.case, block_count=34,
              requested_models=12852, artifacts=artifacts, design_sha256=sha(a.output / 'design.json'),
              source_sha256=design['cases'][a.case]['input_sha256'], elapsed_seconds=time.monotonic()-start,
              feature_only=True, final_refit_not_yet_run=True, independent_FDR_claim=False))
        print('CASE_COMPLETE', a.case, flush=True)
    except Exception as exc:
        failure = dict(status='TECHNICAL_FAILURE', case=a.case, error=type(exc).__name__, message=str(exc),
                       traceback=traceback.format_exc(), no_retry=True)
        if not (case_dir / 'failure.json').exists():
            write(case_dir / 'failure.json', failure)
        raise


def review_case(a):
    import numpy as np
    from review_physical_block_prediction import review_block
    validate(a.output, remote=a.remote)
    case_dir = a.output / 'cases' / a.case
    completion = read(case_dir / 'completion.json')
    require(completion['status'] == 'COMPLETE' and completion['block_count'] == 34, 'INCOMPLETE_CASE')
    require(completion['design_sha256'] == sha(a.output / 'design.json'), 'CASE_DESIGN_CHANGED')
    for p, digest in completion['artifacts'].items():
        require(sha(case_dir / p) == digest, 'CASE_ARTIFACT_CHANGED:' + p)
    A, b, names, manifest = case_input(a.output, a.case)
    reviews, all_arrays = [], []
    for block in manifest['blocks']:
        k = block['block_index']
        record = read(case_dir / 'blocks' / f'block_{k:03d}.json')
        require(record['held_indices'] == block['channel_indices'] and record['molecular_names'] == manifest['molecular_names'],
                'BLOCK_MEMBERSHIP_CHANGED')
        require(record['source_sha256'] == completion['source_sha256'] and record['design_sha256'] == completion['design_sha256'],
                'BLOCK_BINDING_CHANGED')
        with np.load(case_dir / 'blocks' / f'block_{k:03d}.npz', allow_pickle=False) as z:
            arrays = {key: z[key] for key in z.files}
        require(arrays['deleted_names'].tolist() == manifest['molecular_names'], 'DELETED_NAMES_CHANGED')
        reviews.append(review_block(A, b, names, block['channel_indices'], arrays))
        all_arrays.append(arrays)
        print('REVIEW_BLOCK', a.case, k, flush=True)
    recomputed = features_from_blocks(A, b, names, manifest, all_arrays)
    saved = read(case_dir / 'features.json')
    require(recomputed == saved, 'FEATURES_CHANGED')
    output = case_dir / ('remote_review.json' if a.remote else 'local_review.json')
    write(output, dict(status='CACHE_REVIEW_PASS', case=a.case, completion_sha256=sha(case_dir / 'completion.json'),
                       all_molecular_count=len(saved), source_checks='original files' if a.remote else 'sealed aggregates and code',
                       block_reviews=reviews, no_optimizer_or_training=True))
    print('REVIEW_PASS', a.case, flush=True)


def ack(a):
    require(len(a.commit) == 40 and all(c in '0123456789abcdef' for c in a.commit), 'FULL_COMMIT_REQUIRED')
    if a.case in ('model', 'design'):
        artifact = a.output / (a.case + '_seal.json')
    else:
        artifact = a.output / 'cases' / a.case / 'remote_review.json'
        require(read(artifact)['status'] == 'CACHE_REVIEW_PASS', 'UNREVIEWED_CASE')
    extra = {}
    if a.case == 'model':
        require(read(a.output / 'remote_model_review.json')['status'] == 'INDEPENDENT_MODEL_REPLAY_PASS', 'MODEL_NOT_REVIEWED')
        extra = dict(model_review_sha256=sha(a.output / 'remote_model_review.json'))
    if a.case == 'CHECK':
        extra = dict(selection_sha256=sha(a.output / 'selection.json'), selection_seal_sha256=sha(a.output / 'selection_seal.json'))
    write(a.output / 'handoffs' / (a.case + '.json'), dict(status='USER_AUTHORIZED_GIT_SNAPSHOT_CONFIRMED',
          artifact_sha256=sha(artifact), commit=a.commit, case=a.case, **extra))


def require_saved(out, case):
    item = read(out / 'handoffs' / (case + '.json'))
    artifact = out / (case + '_seal.json') if case in ('model', 'design') else out / 'cases' / case / 'remote_review.json'
    require(item['artifact_sha256'] == sha(artifact) and len(item['commit']) == 40, 'HANDOFF_CHANGED')
    if case in CASES:
        review = read(artifact)
        case_dir = out / 'cases' / case
        require(review['completion_sha256'] == sha(case_dir / 'completion.json'), 'REVIEW_COMPLETION_CHANGED')
        completion = read(case_dir / 'completion.json')
        design = read(out / 'design.json')
        require(completion['source_sha256'] == design['cases'][case]['input_sha256']
                and completion['design_sha256'] == sha(out / 'design.json'), 'CASE_SOURCE_BINDING_CHANGED')
        expected = {f'blocks/block_{k:03d}{suffix}' for k in range(34) for suffix in ('.json', '.npz')} | {'features.json', 'timing.json'}
        require(set(completion['artifacts']) == expected, 'CASE_ARTIFACT_MEMBERSHIP_CHANGED')
        for path, digest in completion['artifacts'].items():
            require(sha(case_dir / path) == digest, 'REVIEWED_ARTIFACT_CHANGED:' + path)
    if case == 'model':
        validate_model(out)
        require(item['model_review_sha256'] == sha(out / 'remote_model_review.json'), 'MODEL_REVIEW_CHANGED')
    if case == 'CHECK':
        require(item['selection_sha256'] == sha(out / 'selection.json')
                and item['selection_seal_sha256'] == sha(out / 'selection_seal.json'), 'SAVED_SELECTION_CHANGED')


def validate_model(out):
    require_saved(out, 'DEV_TRAIN')
    seal = read(out / 'model_seal.json')
    for name, key in (('model.json', 'model_sha256'), ('thresholds.json', 'thresholds_sha256'),
                      ('train_scores.json', 'train_scores_sha256'), ('fit_reservation.json', 'reservation_sha256'),
                      ('design.json', 'design_sha256'), ('cases/DEV_TRAIN/features.json', 'features_sha256')):
        require(sha(out / name) == seal[key], 'MODEL_SEAL_CHANGED:' + name)
    return seal


def counts(rows, names):
    names = set(names)
    truth = {r['lipid_name'] for r in rows if r['molecular_truth']}
    raw_truth = {r['lipid_name'] for r in rows if r['molecular_truth'] and r['raw_solver_reported']}
    require(names <= {r['lipid_name'] for r in rows}, 'UNKNOWN_IDENTITY')
    tp, fp = len(truth & names), len(names - truth)
    return dict(TP=tp, FP=fp, FN=len(truth)-tp, FDP=fp/(tp+fp) if tp+fp else None,
                all_truth_recall=tp/len(truth), TP_retention=tp/len(raw_truth) if raw_truth else None,
                all_truth_count=len(truth), raw_solver_misses=len(truth-raw_truth), retained=len(names))


def choose_thresholds(rows, scores):
    require(len(rows) == len(scores), 'SCORE_LENGTH')
    cuts = sorted({float(s) for r, s in zip(rows, scores) if r['raw_solver_reported']}, reverse=True)
    pool = None
    final = None
    best = None
    for cut in cuts:
        selected = [r['lipid_name'] for r, s in zip(rows, scores) if r['raw_solver_reported'] and s >= cut]
        metric = counts(rows, selected)
        if pool is None and metric['all_truth_recall'] >= .95:
            pool = cut
        if metric['retained'] and metric['FDP'] <= .01:
            objective = (metric['TP'], -metric['FP'], cut)
            if best is None or objective > best:
                best, final = objective, cut
    effective = max(pool, final) if pool is not None and final is not None else None
    def selected(cut):
        return [r['lipid_name'] for r, s in zip(rows, scores) if cut is not None and r['raw_solver_reported'] and s >= cut]
    return dict(pool_threshold=pool, final_raw_threshold=final, final_effective_threshold=effective,
                train_pool=counts(rows, selected(pool)), train_final=counts(rows, selected(effective)),
                rule='TRAIN development only; complete ties; no independent FDR')


def matrix(rows, feature_rows, model, fit=False):
    import numpy as np
    import run_identity_confidence_joint_validation as prior
    by_name = {r['molecular_name']: r for r in feature_rows}
    base = prior.transform(rows, model['base'])[0]
    values = np.array([[by_name[r['lipid_name']][key] for key in FEATURES] for r in rows])
    if fit:
        model['auxiliary'] = dict(mean=values.mean(axis=0).tolist(), scale=np.maximum(values.std(axis=0), 1e-12).tolist(),
                                  constant_features=[FEATURES[j] for j in range(2) if values[:, j].std() < 1e-12])
    values = (values - model['auxiliary']['mean']) / model['auxiliary']['scale']
    return np.column_stack((base, values))


def score_rows(rows, features, model):
    import numpy as np
    from scipy.special import expit
    reported = [r for r in rows if r['raw_solver_reported']]
    values = expit(matrix(reported, features, model) @ np.array(model['coef']) + model['intercept']) if reported else []
    lookup = {r['lipid_name']: float(v) for r, v in zip(reported, values)}
    return [lookup.get(r['lipid_name']) for r in rows]


def fit_model(a):
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.exceptions import ConvergenceWarning
    validate(a.output, remote=True)
    require_saved(a.output, 'DEV_TRAIN')
    if (a.output / 'model_seal.json').exists():
        validate_model(a.output)
        print('MODEL_ALREADY_SEALED', flush=True)
        return
    require(not (a.output / 'fit_reservation.json').exists(), 'PRIOR_FIT_RESERVATION_NO_RETRAIN')
    rows = read(a.output / 'prepared/DEV_TRAIN_records.json')
    reported = [r for r in rows if r['raw_solver_reported']]
    require(len(reported) == 174 and sum(r['molecular_truth'] for r in reported) == 125, 'TRAIN_MEMBERSHIP_CHANGED')
    features = read(a.output / 'cases/DEV_TRAIN/features.json')
    model = dict(base=read(a.output / 'baseline_model.json'), features=list(FEATURES), probability_is_uncalibrated=True)
    x = matrix(reported, features, model, fit=True)
    write(a.output / 'fit_reservation.json', dict(design_sha256=sha(a.output / 'design.json'),
          features_sha256=sha(a.output / 'cases/DEV_TRAIN/features.json'), fit_count=1))
    with warnings.catch_warnings():
        warnings.simplefilter('error', ConvergenceWarning)
        lr = LogisticRegression(C=1., solver='lbfgs', max_iter=2000, tol=1e-8).fit(x, [r['molecular_truth'] for r in reported])
    model.update(coef=lr.coef_[0].tolist(), intercept=float(lr.intercept_[0]), n_iter=lr.n_iter_.tolist(),
                 training_names=[r['lipid_name'] for r in reported])
    scores = score_rows(reported, features, model)
    require(np.allclose(scores, lr.predict_proba(x)[:, 1], rtol=1e-12, atol=1e-15), 'PREDICTION_REPLAY_FAILED')
    all_scores = score_rows(rows, features, model)
    thresholds = choose_thresholds(rows, all_scores)
    write(a.output / 'model.json', model)
    write(a.output / 'thresholds.json', thresholds)
    write(a.output / 'train_scores.json', [dict(**r, new_score=s) for r, s in zip(rows, all_scores)])
    write(a.output / 'model_seal.json', dict(status='ONE_MODEL_FROZEN_BEFORE_CHECK_FEATURES',
          model_sha256=sha(a.output / 'model.json'), thresholds_sha256=sha(a.output / 'thresholds.json'),
          train_scores_sha256=sha(a.output / 'train_scores.json'), features_sha256=sha(a.output / 'cases/DEV_TRAIN/features.json'),
          reservation_sha256=sha(a.output / 'fit_reservation.json'), design_sha256=sha(a.output / 'design.json')))
    print('MODEL_FROZEN', json.dumps(thresholds), flush=True)


def review_model(a):
    from review_physical_block_prediction import review_model_predictions
    validate(a.output, remote=a.remote)
    validate_model(a.output)
    model, thresholds = read(a.output / 'model.json'), read(a.output / 'thresholds.json')
    reviews = {}
    reviews['DEV_TRAIN'] = review_model_predictions(
        read(a.output / 'prepared/DEV_TRAIN_records.json'), read(a.output / 'cases/DEV_TRAIN/features.json'),
        model, thresholds, read(a.output / 'train_scores.json'), is_train=True)
    if (a.output / 'selection.json').exists():
        reviews['CHECK'] = review_model_predictions(
            read(a.output / 'prepared/CHECK_records.json'), read(a.output / 'cases/CHECK/features.json'),
            model, thresholds, read(a.output / 'selection.json')['rows'], is_train=False)
    stem = 'model_and_check_review.json' if 'CHECK' in reviews else 'model_review.json'
    name = ('remote_' if a.remote else 'local_') + stem
    write(a.output / name, dict(status='INDEPENDENT_MODEL_REPLAY_PASS', reviews=reviews,
          model_seal_sha256=sha(a.output / 'model_seal.json')))
    print('MODEL_REVIEW_PASS', flush=True)


def check_selection(a):
    validate(a.output, remote=a.remote)
    validate_model(a.output)
    case_dir = a.output / 'cases/CHECK'
    require(read(case_dir / 'remote_review.json')['completion_sha256'] == sha(case_dir / 'completion.json'), 'CHECK_NOT_REVIEWED')
    require(sha(case_dir / 'features.json') == read(case_dir / 'completion.json')['artifacts']['features.json'], 'CHECK_FEATURES_CHANGED')
    model, thresholds = read(a.output / 'model.json'), read(a.output / 'thresholds.json')
    rows = read(a.output / 'prepared/CHECK_records.json')
    features = read(a.output / 'cases/CHECK/features.json')
    scores = score_rows(rows, features, model)
    pool_cut, final_cut = thresholds['pool_threshold'], thresholds['final_effective_threshold']
    pool = [r['lipid_name'] for r, s in zip(rows, scores) if pool_cut is not None and r['raw_solver_reported'] and s >= pool_cut]
    eligible = [r['lipid_name'] for r, s in zip(rows, scores) if final_cut is not None and r['lipid_name'] in pool and s >= final_cut]
    result = dict(status='CHECK_SELECTION_FROZEN', model_sha256=sha(a.output / 'model.json'), thresholds=thresholds,
                  pool_names=pool, final_score_eligible_names=eligible, pool=counts(rows, pool),
                  final_score_eligible=counts(rows, eligible),
                  rows=[dict(**r, new_score=s, in_pool=r['lipid_name'] in pool,
                             final_score_eligible=r['lipid_name'] in eligible) for r, s in zip(rows, scores)],
                  independent_FDR_claim=False)
    write(a.output / 'selection.json', result)
    write(a.output / 'selection_seal.json', dict(selection_sha256=sha(a.output / 'selection.json'),
          model_seal_sha256=sha(a.output / 'model_seal.json'), check_features_sha256=sha(case_dir / 'features.json')))
    print('CHECK_SELECTION', json.dumps({k: result[k] for k in ('pool', 'final_score_eligible')}), flush=True)


def final_decision(a):
    validate(a.output, remote=True)
    require_saved(a.output, 'CHECK')
    require_saved(a.output, 'model')
    check_selection(a)
    if (a.output / 'final_decision.json').exists():
        require(read(a.output / 'final_decision.json')['selection_sha256'] == sha(a.output / 'selection.json'), 'FINAL_SELECTION_CHANGED')
        print('FINAL_DECISION_ALREADY_COMPLETE', flush=True)
        return
    selection = read(a.output / 'selection.json')
    ceiling = selection['final_score_eligible']
    if ceiling['all_truth_recall'] < .8:
        write(a.output / 'final_decision.json', dict(status='FROZEN_FINAL_RULE_RECALL_FUTILITY',
              final_refit_executed=False, recall_upper_bound=ceiling['all_truth_recall'],
              maximum_TP=ceiling['TP'], all_truth_count=ceiling['all_truth_count'],
              reason='Final set must be a subset of immutable score-eligible names; refit cannot recover excluded identities',
              selection_sha256=sha(a.output / 'selection.json'), no_claim_of_measured_refit_failure=True,
              independent_FDR_claim=False))
        print('FINAL_RECALL_FUTILITY', ceiling['TP'], flush=True)
        return
    import numpy as np
    from refit_screened_nnls import fit_screened
    sources = read(a.output / 'expected_sources.json')
    names = read(a.output / 'block_manifest.json')['candidate_names']
    indices = [i for i, name in enumerate(names) if name in selection['pool_names']]
    with np.load(Path(sources['cases']['CHECK']) / 'prepared_arrays.npz', allow_pickle=False) as z:
        result = fit_screened(z['A_solver'], z['B'], z['mask'], indices, a.output / 'final_refit', workers=4,
                              source_binding=dict(selection_sha256=sha(a.output / 'selection.json'), design_sha256=sha(a.output / 'design.json')))
    means = result['means']
    rows = read(a.output / 'prepared/CHECK_records.json')
    refit_names = {names[i] for i, mean in enumerate(means) if mean > .001}
    final_names = refit_names & set(selection['final_score_eligible_names'])
    candidate_rows = [dict(candidate_index=i, lipid_name=name, final_abundance=float(means[i]),
                           refit_reported=bool(means[i] > .001), final_reported=bool(means[i] > .001 and name in final_names))
                      for i, name in enumerate(names)]
    molecule_rows = [dict(**row, pool=row['lipid_name'] in selection['pool_names'],
                          score_eligible=row['lipid_name'] in selection['final_score_eligible_names'],
                          refit_reported=row['lipid_name'] in refit_names, final_reported=row['lipid_name'] in final_names,
                          final_abundance=float(sum(means[i] for i, name in enumerate(names) if name == row['lipid_name'] and means[i] > .001))) for row in rows]
    write(a.output / 'final_candidate_records.json', candidate_rows)
    write(a.output / 'final_molecular_records.json', molecule_rows)
    write(a.output / 'final_decision.json', dict(status='REFIT_COMPLETE', final_refit_executed=True,
          pool=selection['pool'], score_eligible=ceiling, refit_reported=counts(rows, refit_names), final=counts(rows, final_names),
          final_names=sorted(final_names), arrays_sha256=sha(result['arrays_path']), receipt_sha256=sha(result['receipt_path']),
          selection_sha256=sha(a.output / 'selection.json'), independent_FDR_claim=False))
    print('FINAL_REFIT_COMPLETE', json.dumps(counts(rows, final_names)), flush=True)


def review_final(a):
    import numpy as np
    validate(a.output, remote=True)
    require_saved(a.output, 'CHECK')
    require_saved(a.output, 'model')
    review_model(a)
    decision, selection = read(a.output / 'final_decision.json'), read(a.output / 'selection.json')
    require(decision['selection_sha256'] == sha(a.output / 'selection.json'), 'FINAL_BINDING_CHANGED')
    rows = read(a.output / 'prepared/CHECK_records.json')
    eligible = set(selection['final_score_eligible_names'])
    ceiling = counts(rows, eligible)
    if not decision['final_refit_executed']:
        require(decision['status'] == 'FROZEN_FINAL_RULE_RECALL_FUTILITY' and ceiling['all_truth_recall'] < .8
                and decision['maximum_TP'] == ceiling['TP'] and decision['recall_upper_bound'] == ceiling['all_truth_recall'],
                'INVALID_RECALL_FUTILITY_PROOF')
        report = dict(status='CACHE_FINAL_REVIEW_PASS', kind='RECALL_UPPER_BOUND_ONLY',
                      maximum_TP=ceiling['TP'], recall_upper_bound=ceiling['all_truth_recall'],
                      no_claim_of_measured_refit_failure=True)
    else:
        directory = a.output / 'final_refit'
        receipt = read(directory / 'completion.json')
        require(sha(directory / 'completion.json') == decision['receipt_sha256']
                and sha(directory / 'learned_arrays.npz') == decision['arrays_sha256'] == receipt['arrays_sha256'], 'FINAL_ARRAY_BINDING_CHANGED')
        for name, digest in receipt['block_hashes'].items():
            require(sha(directory / 'nnls_blocks' / name) == digest, 'FINAL_BLOCK_CHANGED')
        with np.load(directory / 'learned_arrays.npz', allow_pickle=False) as z:
            X, Bhat = z['X_hat'], z['B_hat']
        source = Path(read(a.output / 'expected_sources.json')['cases']['CHECK'])
        with np.load(source / 'prepared_arrays.npz', allow_pickle=False) as z:
            A, B, mask = z['A_solver'], z['B'], z['mask']
        names = read(a.output / 'block_manifest.json')['candidate_names']
        require(X.shape == (391, 200, 90) and Bhat.shape == B.shape and np.isfinite(X).all()
                and np.isfinite(Bhat).all() and (X >= 0).all() and np.all(X[:, ~mask] == 0), 'INVALID_FINAL_ARRAYS')
        excluded = [i for i, name in enumerate(names) if name not in selection['pool_names']]
        require(np.all(X[excluded] == 0), 'EXCLUDED_CANDIDATE_NONZERO')
        means = X[:, mask].mean(axis=1, dtype=np.float64)
        reported = {names[i] for i, value in enumerate(means) if value > .001}
        final = reported & eligible
        require(counts(rows, final) == decision['final'] and sorted(final) == decision['final_names'], 'FINAL_COUNTS_CHANGED')
        cached = read(a.output / 'final_candidate_records.json')
        require(len(cached) == len(names), 'FINAL_CANDIDATE_MEMBERSHIP')
        for i, row in enumerate(cached):
            require(row['candidate_index'] == i and row['lipid_name'] == names[i]
                    and math.isclose(row['final_abundance'], means[i], rel_tol=1e-12, abs_tol=1e-15), 'FINAL_CANDIDATE_CHANGED')
        # Replay stored float32 reconstruction in bounded pixel chunks.
        max_error = 0.
        xf, bf = X[:, mask], Bhat[:, mask]
        for start in range(0, xf.shape[1], 250):
            predicted = (A.astype(np.float64) @ xf[:, start:start+250].astype(np.float64)).astype(np.float32)
            max_error = max(max_error, float(np.abs(predicted - bf[:, start:start+250]).max()))
        require(max_error <= 5e-7 * max(float(np.abs(Bhat).max()), 1e-12), 'FINAL_RECONSTRUCTION_CHANGED')
        report = dict(status='CACHE_FINAL_REVIEW_PASS', kind='ACTUAL_FULL_PIXEL_REFIT', final=counts(rows, final),
                      candidate_count=391, molecular_count=len(rows), maximum_reconstruction_difference=max_error)
    report['decision_sha256'] = sha(a.output / 'final_decision.json')
    write(a.output / 'final_review.json', report)
    print('FINAL_REVIEW_PASS', flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('prepare', 'run_case', 'review_case', 'ack', 'fit_model', 'review_model', 'check_selection', 'final_decision', 'review_final'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--expected', type=Path)
    parser.add_argument('--case', choices=(*CASES, 'model', 'design'))
    parser.add_argument('--commit')
    parser.add_argument('--remote', action='store_true')
    args = parser.parse_args()
    globals()[args.action](args)


if __name__ == '__main__':
    main()
