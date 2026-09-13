"""Reuse the prepared HOLD1 observation as NEW_COMPOSITION_CHECK development.

The stopped historical CAL/HOLD runner is never executed or modified. Prepare
before model fitting; bind_model records the separately saved model snapshot.
The two initial computational stages may run concurrently, each with four
workers. Reviews use only cached arrays and never invoke an optimizer.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import math
from pathlib import Path
import re
import shutil
import time
from types import SimpleNamespace

if __package__:
    from . import run_physical_block_prediction as physical
    from . import run_physical_block_pool_refit as util
    from . import run_small_mismatch_nnls_first_case as engine
else:
    import run_physical_block_prediction as physical
    import run_physical_block_pool_refit as util
    import run_small_mismatch_nnls_first_case as engine

ROOT = physical.ROOT
read, write, sha, require = physical.read, physical.write, physical.sha, physical.require
CASE = 'NEW_COMPOSITION_CHECK'
PROTOCOL = 'docs/PHYSICAL_BLOCK_SCORE_CORRECTION_V2.md'
CODE = tuple(dict.fromkeys((*physical.CODE, 'analysis/run_physical_block_pool_refit.py',
    'analysis/run_new_composition_development.py', 'analysis/physical_block_monotone_score.py',
    'src/rho_zero.py', PROTOCOL)))
MODEL_FILES = ('model.json', 'thresholds.json', 'train_scores.json', 'fit_record.json',
               'reference_model.json', 'reference_train_scores.json', 'fit_reservation.json',
               'optimizer_result.json', 'model_review.json', 'model_seal.json')
SOURCE_SHA = dict(prepared_arrays='48a517e0f77568f7811ff8c711ab0223174ca8d8ad7053d20c5f7b7c6f52fbe7',
                  input='c316810365fdbd05151fa4edbafdb0195c0a1a85916d066be2b9d9e00cb603e2')


def score_module():
    if __package__:
        from . import physical_block_monotone_score
    else:
        import physical_block_monotone_score
    return physical_block_monotone_score


def copy_once(source, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        require(sha(source) == sha(target), 'COPIED_INPUT_CHANGED:' + str(target))
    else:
        require(not target.with_suffix(target.suffix + '.tmp').exists(), 'PARTIAL_COPY_PRESERVED')
        temporary = target.with_suffix(target.suffix + '.tmp')
        shutil.copyfile(source, temporary)
        require(sha(source) == sha(temporary), 'COPY_HASH_FAILED')
        temporary.rename(target)


def full_commit(value):
    require(value is not None and re.fullmatch('[0-9a-f]{40}', value) is not None,
            'EXTERNALLY_VERIFIED_FULL_PUSHED_COMMIT_REQUIRED')
    return value


def source_arrays(path):
    import numpy as np
    with np.load(path, allow_pickle=False) as z:
        A, B, mask = z['A_solver'], z['B'], z['mask']
    require(A.shape == (1084, 391) and B.shape == (1084, 200, 90)
            and mask.dtype == bool and mask.shape == (200, 90) and mask.any(), 'INVALID_SOURCE_SHAPES')
    require(np.isfinite(A).all() and np.isfinite(B).all() and (A >= 0).all()
            and np.all(B[:, ~mask] == 0), 'INVALID_SOURCE_ARRAYS')
    return A, B, mask


def prepare(a):
    import numpy as np
    require(not a.output.exists(), 'EXISTING_CASE_PRESERVED')
    require(physical.runtime() == dict(numpy='2.1.3', scipy='1.15.3', sklearn='1.6.1'), 'RUNTIME_CHANGED')
    source = a.source
    require(sha(source / 'prepared_arrays.npz') == SOURCE_SHA['prepared_arrays']
            and sha(source / 'input.json') == SOURCE_SHA['input'], 'WRONG_PREPARED_COMPOSITION')
    require(not any((source / name).exists() for name in
                    ('result.json', 'learned_arrays.npz', 'nnls_complete.json')), 'SOURCE_ALREADY_SOLVED')
    old_root = source.parents[1]
    source_input = read(source / 'input.json')
    scientific = source_input['scientific']
    preparation = read(old_root / 'preparation_seal.json')
    require(scientific['id'] == 'HOLD1' and scientific['seed'] == 7501
            and scientific['full_library_candidates'] == 391
            and scientific['old_source_omissions_used'] is False
            and preparation['CAL_HOLD_truth_overlap'] == 0
            and preparation['cases']['HOLD1']['input_sha256'] == SOURCE_SHA['input']
            and preparation['cases']['HOLD1']['fingerprint'] == source_input['fingerprint']
            and preparation['cases']['HOLD1']['actual_perturbed_truth_count'] == 125,
            'ORIGINAL_PREPARATION_BINDING_CHANGED')
    require(hashlib.sha256(engine.canonical(scientific)).hexdigest() == source_input['fingerprint'],
            'SOURCE_SCIENTIFIC_FINGERPRINT_CHANGED')
    require(sha(old_root / 'contract.json') == read(old_root / 'contract_seal.json')['contract_sha256']
            == scientific['contract_sha256'] == preparation['contract_sha256'], 'SOURCE_CONTRACT_CHANGED')
    for name, digest in scientific['files'].items():
        require(sha(source / name) == digest, 'SOURCE_PREPARED_FILE_CHANGED:' + name)
    A, B, mask = source_arrays(source / 'prepared_arrays.npz')
    require(engine.ah(A) == scientific['A_solver_sha256'] and engine.ah(B) == scientific['B_sha256'],
            'SOURCE_ARRAY_HASH_CHANGED')
    metadata = read(source / 'metadata.json')
    parent_design = read(a.physical_parent / 'design.json')
    require(sha(a.physical_parent / 'design.json')
            == read(a.physical_parent / 'design_seal.json')['design_sha256'], 'PARENT_DESIGN_CHANGED')
    require(sha(a.physical_parent / 'block_manifest.json') == parent_design['blocks_sha256'], 'PARENT_BLOCKS_CHANGED')
    require(sha(a.physical_parent / 'prepared/CHECK.npz') == parent_design['cases']['CHECK']['input_sha256'],
            'PARENT_LIBRARY_SOURCE_CHANGED')
    with np.load(a.physical_parent / 'prepared/CHECK.npz', allow_pickle=False) as z:
        require(np.array_equal(A.astype(np.float64), z['A']), 'NOMINAL_LIBRARY_CHANGED')
    manifest = read(a.physical_parent / 'block_manifest.json')
    names = metadata['lipid_name']
    require(names == manifest['candidate_names'] and len(names) == 391
            and len(set(names)) == 377 and manifest['block_count'] == 34
            and len(scientific['truth_names']) == len(set(scientific['truth_names'])) == 125,
            'COMPOSITION_UNIVERSE_CHANGED')
    a.output.mkdir(parents=True)
    for name in ('input.json', 'metadata.json', 'perturbation_audit.json'):
        copy_once(source / name, a.output / 'prepared' / ('source_' + name))
    for name in ('contract.json', 'contract_seal.json', 'preparation_seal.json'):
        copy_once(old_root / name, a.output / 'prepared' / ('source_' + name))
    copy_once(a.physical_parent / 'block_manifest.json', a.output / 'prepared/block_manifest.json')
    b = B[:, mask].mean(axis=1, dtype=np.float64)
    physical.save_npz(a.output / 'prepared/portable.npz', A=A, b=b, mask=mask)
    prepared = {p.relative_to(a.output).as_posix(): sha(p) for p in sorted((a.output / 'prepared').iterdir())}
    original_files = {str(source / n): sha(source / n)
                      for n in ('input.json', 'metadata.json', 'perturbation_audit.json', 'prepared_arrays.npz')}
    original_files.update({str(old_root / n): sha(old_root / n)
                          for n in ('contract.json', 'contract_seal.json', 'preparation_seal.json')})
    original_files.update({str(a.physical_parent / n): sha(a.physical_parent / n)
                          for n in ('design.json', 'design_seal.json', 'block_manifest.json', 'prepared/CHECK.npz')})
    design = dict(version='PHYSICAL_BLOCK_SCORE_CORRECTION_V2', case=CASE,
        role='DEVELOPMENT; original HOLD eligibility retired', runtime=physical.runtime(),
        code={p: sha(ROOT / p) for p in CODE}, prepared=prepared, original_files=original_files,
        source_prepared_path=str(source / 'prepared_arrays.npz'), source_prepared_sha256=SOURCE_SHA['prepared_arrays'],
        source_arrays={key: util.array_binding(value) for key, value in zip(('A', 'B', 'mask'), (A, B, mask))},
        candidate_count=391, molecular_count=377, all_truth_count=125, foreground_pixels=int(mask.sum()),
        workers_per_stage=4, simultaneous_stages_max=2, total_workers_max=8, blas_threads=1,
        block_size=250, maxiter=3910, physical_block_count=34, physical_models=12852,
        wall_budget_seconds_per_stage=14400, budget_enforcement='external process-group timeout; preserve partials',
        candidate_report_gate=.001, molecular_report_gate='any alias; sum reported abundance; max reported rho',
        model_required_before_any_solve=True, independent_FDR_claim=False, no_observation_regeneration=True,
        scope='Different true composition and perturbation together; existing spatial maps and relative abundances')
    write(a.output / 'design.json', design)
    write(a.output / 'design_seal.json', dict(status='PREPARED_BEFORE_MODEL_FIT_AND_NEW_SOLVES',
                                            design_sha256=sha(a.output / 'design.json')))
    validate(a.output, remote=True)
    print('NEW_COMPOSITION_PREPARED', sha(a.output / 'design_seal.json'), flush=True)


def validate(output, remote=False, need_model=False):
    import numpy as np
    design = read(output / 'design.json')
    require(read(output / 'design_seal.json') == dict(status='PREPARED_BEFORE_MODEL_FIT_AND_NEW_SOLVES',
            design_sha256=sha(output / 'design.json')), 'DESIGN_SEAL_CHANGED')
    require(design['case'] == CASE and design['code'] == {p: sha(ROOT / p) for p in CODE}, 'CODE_CHANGED')
    require(set(design['prepared']) == {p.relative_to(output).as_posix() for p in (output / 'prepared').iterdir()},
            'PREPARED_MEMBERSHIP_CHANGED')
    for path, digest in design['prepared'].items():
        require(sha(output / path) == digest, 'PREPARED_FILE_CHANGED:' + path)
    with np.load(output / 'prepared/portable.npz', allow_pickle=False) as z:
        require(set(z.files) == {'A', 'b', 'mask'}, 'PORTABLE_ARCHIVE_CHANGED')
        A, b, mask = z['A'], z['b'], z['mask']
    require(util.array_binding(A) == design['source_arrays']['A']
            and util.array_binding(mask) == design['source_arrays']['mask']
            and b.shape == (1084,) and np.isfinite(b).all() and float(b @ b) > 0,
            'PORTABLE_ARRAY_CHANGED')
    require((design['workers_per_stage'], design['block_size'], design['maxiter'],
             design['physical_block_count'], design['candidate_report_gate']) == (4, 250, 3910, 34, .001),
            'SCIENTIFIC_SETTINGS_CHANGED')
    if remote:
        require(physical.runtime() == design['runtime'], 'RUNTIME_CHANGED')
        for path, digest in design['original_files'].items():
            require(sha(path) == digest, 'ORIGINAL_SOURCE_CHANGED:' + path)
    if need_model:
        binding = read(output / 'model_binding.json')
        require(binding['design_sha256'] == sha(output / 'design.json')
                and binding['design_seal_sha256'] == sha(output / 'design_seal.json')
                and full_commit(binding['pushed_model_commit']) == binding['pushed_model_commit']
                and binding['artifacts'] == {p: sha(output / 'model' / p) for p in MODEL_FILES},
                'MODEL_BINDING_CHANGED')
        score_module().validate_model_directory(output / 'model')
        require(sha(output / 'model/optimizer_result.json')
                == read(output / 'model/fit_record.json')['optimizer_result_sha256']
                and read(output / 'model/model_review.json')['status'] == 'PASS'
                and read(output / 'model/model_review.json')['model_seal_sha256']
                == sha(output / 'model/model_seal.json'), 'MODEL_REVIEW_OR_OPTIMIZER_RECEIPT_CHANGED')
    return design, A.astype(np.float64), b, mask


def bind_model(a):
    validate(a.output, remote=True)
    full_commit(a.commit)
    require(a.model_dir is not None, 'MODEL_DIRECTORY_REQUIRED')
    score_module().validate_model_directory(a.model_dir)
    require(not any((a.output / n / 'completion.json').exists() for n in ('nnls_rho', 'physical_features')),
            'MODEL_MUST_BE_BOUND_BEFORE_SOLVES')
    for name in MODEL_FILES:
        copy_once(a.model_dir / name, a.output / 'model' / name)
    write(a.output / 'model_binding.json', dict(status='PUSHED_MODEL_FIXED_BEFORE_NEW_COMPOSITION_SOLVES',
        design_sha256=sha(a.output / 'design.json'), design_seal_sha256=sha(a.output / 'design_seal.json'),
        pushed_model_commit=a.commit, artifacts={p: sha(a.output / 'model' / p) for p in MODEL_FILES}))
    validate(a.output, remote=True, need_model=True)
    print('MODEL_BOUND', sha(a.output / 'model_binding.json'), flush=True)


def stage_binding(output):
    return dict(design_sha256=sha(output / 'design.json'), model_binding_sha256=sha(output / 'model_binding.json'))


def stage_fingerprint(output, stage):
    return hashlib.sha256(engine.canonical(dict(stage=stage, **stage_binding(output)))).hexdigest()


def make_records(output, means, rho_scores):
    metadata = read(output / 'prepared/source_metadata.json')
    s = read(output / 'prepared/source_input.json')['scientific']
    truth, reportable = set(s['truth_names']), set(s['reportable_truth_names'])
    candidates = []
    for i, name in enumerate(metadata['lipid_name']):
        candidates.append(dict(candidate_index=i, candidate_id=metadata['candidate_id'][i],
            lipid_name=name, lipid_class=metadata['lipid_class'][i], X_hat=float(means[i]),
            candidate_truth=i in s['truth_indices'], molecular_truth=name in truth,
            reportable_truth=name in reportable, raw_solver_reported=i in rho_scores,
            rho_zero=rho_scores[i]['rho_zero'] if i in rho_scores else None,
            rho_details=rho_scores.get(i), case=CASE, split='EXPOSED_NEW_COMPOSITION_DEV'))
    molecular = []
    for name in dict.fromkeys(metadata['lipid_name']):
        aliases = [r for r in candidates if r['lipid_name'] == name]
        active = [r for r in aliases if r['raw_solver_reported']]
        molecular.append(dict(case=CASE, lipid_name=name, molecular_truth=name in truth,
            reportable_truth=name in reportable, actually_perturbed_truth=name in s['actually_perturbed_truth_names'],
            raw_solver_reported=bool(active), X_hat=sum(r['X_hat'] for r in active),
            rho_zero=max(r['rho_zero'] for r in active) if active else None,
            candidate_indices=[r['candidate_index'] for r in aliases],
            reported_candidate_indices=[r['candidate_index'] for r in active]))
    return candidates, molecular


def nnls_rho(a):
    import numpy as np
    import run_nnls_solver_baseline as baseline
    import rho_zero
    from threadpoolctl import threadpool_limits
    design, A, b, mask = validate(a.output, remote=True, need_model=True)
    directory = a.output / 'nnls_rho'
    if (directory / 'completion.json').exists():
        print('NNLS_RHO_ALREADY_COMPLETE_REVIEW_CACHE', flush=True)
        return
    directory.mkdir(exist_ok=True)
    started = time.monotonic()
    fingerprint = stage_fingerprint(a.output, 'nnls_rho')
    _, B, source_mask = source_arrays(design['source_prepared_path'])
    require(np.array_equal(source_mask, mask), 'SOURCE_MASK_CHANGED')
    done_path = directory / 'nnls_complete.json'
    if (directory / 'learned_arrays.npz').exists() or done_path.exists():
        require((directory / 'learned_arrays.npz').exists() and done_path.exists(), 'PARTIAL_NNLS_COMPLETION_PRESERVED')
        done = read(done_path)
        require(done['fingerprint'] == fingerprint and done['arrays_sha256'] == sha(directory / 'learned_arrays.npz'),
                'NNLS_COMPLETE_CACHE_CHANGED')
        with np.load(directory / 'learned_arrays.npz', allow_pickle=False) as z:
            X, Bhat = z['X_hat'], z['B_hat']
    else:
        engine.CASE = CASE
        with threadpool_limits(limits=1):
            X, Bhat, kkt = engine.fit_nnls(SimpleNamespace(output=directory), dict(fingerprint=fingerprint),
                                         A, B, mask, np, baseline)
        require(np.isfinite(X).all() and np.isfinite(Bhat).all(), 'NONFINITE_NNLS_OUTPUT')
        physical.save_npz(directory / 'learned_arrays.npz', X_hat=X, B_hat=Bhat)
        done = dict(status='NNLS_COMPLETE', fingerprint=fingerprint,
            arrays_sha256=sha(directory / 'learned_arrays.npz'), KKT=kkt,
            all_pixels_KKT_checked_before_float32=True, residual=util.residual_summary(B, Bhat, mask))
        write(done_path, done)
    require(X.shape == (391, *mask.shape) and X.dtype == np.float32 and np.isfinite(X).all() and (X >= 0).all(),
            'INVALID_NNLS_ARRAY')
    means = X[:, mask].mean(axis=1, dtype=np.float64)
    weights = rho_zero.identity_weights(A)
    reported = np.flatnonzero(means > .001).tolist()
    scores = {}
    for number, i in enumerate(reported, 1):
        path = directory / 'rho' / f'candidate_{i:04d}.json'
        if path.exists():
            record = read(path)
        else:
            with threadpool_limits(limits=1):
                values = rho_zero.rho_zero_from_weighted_case(A, b, i, weights=weights)
            record = dict(fingerprint=fingerprint, candidate_index=i, b_sha256=engine.ah(b),
                          arrays_sha256=done['arrays_sha256'], result=values)
            write(path, record)
        require(record['fingerprint'] == fingerprint and record['candidate_index'] == i
                and record['b_sha256'] == engine.ah(b) and record['arrays_sha256'] == done['arrays_sha256']
                and all(math.isfinite(v) for v in record['result'].values())
                and record['result']['rho_zero'] >= 0, 'INVALID_RHO_CACHE')
        scores[i] = record['result']
        if number % 25 == 0 or number == len(reported):
            print('RHO', CASE, number, len(reported), flush=True)
    (directory / 'rho').mkdir(exist_ok=True)
    candidates, molecular = make_records(a.output, means, scores)
    write(directory / 'candidate_records.json', candidates)
    write(directory / 'molecular_records.json', molecular)
    paths = [directory / n for n in ('learned_arrays.npz', 'nnls_complete.json', 'candidate_records.json', 'molecular_records.json')]
    paths += [p for name in ('nnls_blocks', 'rho') for p in sorted((directory / name).iterdir())]
    write(directory / 'completion.json', dict(status='COMPLETE', case=CASE, stage='nnls_rho',
        fingerprint=fingerprint, binding=stage_binding(a.output), foreground_pixels=int(mask.sum()),
        elapsed_seconds=time.monotonic() - started, original_rho_unchanged=True,
        artifacts={p.relative_to(directory).as_posix(): sha(p) for p in paths}, independent_FDR_claim=False))
    print('NNLS_RHO_COMPLETE', flush=True)


def block_task(output_string, index):
    import numpy as np
    from physical_block_prediction import solve_block
    from threadpoolctl import threadpool_limits
    output = Path(output_string)
    with np.load(output / 'prepared/portable.npz', allow_pickle=False) as z:
        A, b = z['A'].astype(np.float64), z['b']
    manifest = read(output / 'prepared/block_manifest.json')
    stem = output / 'physical_features/blocks' / f'block_{index:03d}'
    binding = stage_binding(output)
    if stem.with_suffix('.json').exists():
        record = read(stem.with_suffix('.json'))
        require(record['binding'] == binding and record['block_index'] == index
                and record['arrays_sha256'] == sha(stem.with_suffix('.npz')), 'PHYSICAL_BLOCK_CACHE_CHANGED')
        return record
    require(not stem.with_suffix('.npz').exists(), 'PARTIAL_PHYSICAL_BLOCK_PRESERVED')
    started = time.monotonic()
    with threadpool_limits(limits=1):
        result = solve_block(A, b, manifest['candidate_names'], manifest['blocks'][index]['channel_indices'])
    physical.save_npz(stem.with_suffix('.npz'), **result['arrays'])
    record = dict(block_index=index, binding=binding, arrays_sha256=sha(stem.with_suffix('.npz')),
                  elapsed_seconds=time.monotonic() - started, **result['diagnostics'])
    write(stem.with_suffix('.json'), record)
    return record


def physical_features(a):
    import numpy as np
    design, A, b, _ = validate(a.output, remote=True, need_model=True)
    directory = a.output / 'physical_features'
    if (directory / 'completion.json').exists():
        print('PHYSICAL_FEATURES_ALREADY_COMPLETE_REVIEW_CACHE', flush=True)
        return
    (directory / 'blocks').mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    pool = ProcessPoolExecutor(max_workers=4)
    futures = {pool.submit(block_task, str(a.output), i): i for i in range(34)}
    try:
        for future in as_completed(futures, timeout=design['wall_budget_seconds_per_stage']):
            record = future.result()
            print('PHYSICAL_BLOCK_COMPLETE', record['block_index'], round(record['elapsed_seconds'], 3), flush=True)
    except BaseException:
        for future in futures:
            future.cancel()
        for process in pool._processes.values():
            process.terminate()
        pool.shutdown(wait=True, cancel_futures=True)
        raise
    else:
        pool.shutdown(wait=True)
    manifest = read(a.output / 'prepared/block_manifest.json')
    arrays = []
    for i in range(34):
        with np.load(directory / 'blocks' / f'block_{i:03d}.npz', allow_pickle=False) as z:
            arrays.append({key: z[key] for key in z.files})
    write(directory / 'features.json', physical.features_from_blocks(A, b, manifest['candidate_names'], manifest, arrays))
    paths = [p for p in sorted((directory / 'blocks').iterdir())] + [directory / 'features.json']
    write(directory / 'completion.json', dict(status='COMPLETE', case=CASE, stage='physical_features',
        binding=stage_binding(a.output), block_count=34, requested_models=12852,
        elapsed_seconds=time.monotonic() - started,
        artifacts={p.relative_to(directory).as_posix(): sha(p) for p in paths}, independent_FDR_claim=False))
    print('PHYSICAL_FEATURES_COMPLETE', flush=True)


def check_completion(output, stage, expected_paths):
    directory = output / stage
    completion = read(directory / 'completion.json')
    require(completion['status'] == 'COMPLETE' and completion['case'] == CASE
            and completion['stage'] == stage and completion['binding'] == stage_binding(output)
            and set(completion['artifacts']) == set(expected_paths), 'STAGE_COMPLETION_CHANGED:' + stage)
    for path, digest in completion['artifacts'].items():
        require(sha(directory / path) == digest, 'STAGE_ARTIFACT_CHANGED:' + path)
    return completion


def audit_nnls_arrays(directory, A, mask, fingerprint, indices=None, helper_receipt=None):
    """Replay full stored arrays and every original float64 KKT receipt."""
    import numpy as np
    from threadpoolctl import threadpool_limits
    indices = list(range(A.shape[1])) if indices is None else indices
    with np.load(directory / 'learned_arrays.npz', allow_pickle=False) as z:
        require(set(z.files) == {'X_hat', 'B_hat'}, 'LEARNED_ARRAY_MEMBERSHIP_CHANGED')
        X, Bhat = z['X_hat'], z['B_hat']
    require(X.shape == (391, *mask.shape) and Bhat.shape == (1084, *mask.shape)
            and X.dtype == Bhat.dtype == np.float32 and np.isfinite(X).all()
            and np.isfinite(Bhat).all() and (X >= 0).all() and np.all(X[:, ~mask] == 0)
            and np.all(Bhat[:, ~mask] == 0), 'INVALID_LEARNED_ARRAYS')
    require(np.all(X[[i for i in range(391) if i not in set(indices)]] == 0), 'EXCLUDED_COLUMN_REENTRY')
    flat, fitted = X[:, mask], Bhat[:, mask]
    count = flat.shape[1]
    blocks = directory / 'nnls_blocks'
    expected_files = {f'block_{start:06d}_{min(start+250,count):06d}.{ext}'
                      for start in range(0, count, 250) for ext in ('json', 'npz')}
    require({p.name for p in blocks.iterdir()} == expected_files, 'NNLS_BLOCK_MEMBERSHIP_CHANGED')
    if helper_receipt is not None:
        require(set(helper_receipt['block_hashes']) == expected_files, 'REFIT_BLOCK_HASH_MEMBERSHIP_CHANGED')
    overall = dict(max_dual_violation=0., max_complementarity=0., max_bound_ratio=0.)
    max_error = 0.
    with threadpool_limits(limits=1):
        for start in range(0, count, 250):
            stop = min(start + 250, count)
            stem = f'block_{start:06d}_{stop:06d}'
            record = read(blocks / (stem + '.json'))
            require(record['fingerprint'] == fingerprint and record['start'] == start and record['stop'] == stop
                    and record['array_sha256'] == sha(blocks / (stem + '.npz'))
                    and record['all_pixels_checked_before_float32'] is True, 'NNLS_BLOCK_BINDING_CHANGED')
            if helper_receipt is not None:
                require(record['retained_indices'] == indices
                        and record['empty_design_vacuous_kkt'] == (len(indices) == 0), 'REFIT_INDICES_CHANGED')
                for ext in ('.json', '.npz'):
                    require(sha(blocks / (stem + ext)) == helper_receipt['block_hashes'][stem + ext],
                            'REFIT_BLOCK_HASH_CHANGED')
            checks = record['KKT']
            require(set(checks) == set(overall) and all(math.isfinite(v) and v >= 0 for v in checks.values())
                    and checks['max_bound_ratio'] <= 1., 'INVALID_ORIGINAL_KKT_RECEIPT')
            for key, value in checks.items():
                overall[key] = max(overall[key], value)
            with np.load(blocks / (stem + '.npz'), allow_pickle=False) as z:
                require(set(z.files) == {'X_hat'}, 'NNLS_BLOCK_ARRAY_KEYS_CHANGED')
                block = z['X_hat']
            require(block.dtype == np.float32 and block.shape == (len(indices), stop-start)
                    and np.array_equal(block, flat[indices, start:stop]), 'NNLS_BLOCK_FINAL_ARRAY_CHANGED')
            predicted = (A[:, indices] @ block.astype(np.float64)).astype(np.float32)
            max_error = max(max_error, float(np.max(np.abs(predicted.astype(np.float64)
                                                        - fitted[:, start:stop].astype(np.float64)))))
    bound = 5e-7 * max(float(np.max(np.abs(fitted))), 1e-12)
    require(max_error <= bound, 'CACHED_RECONSTRUCTION_CHANGED')
    return X, Bhat, dict(KKT=overall, max_reconstruction_error=max_error, reconstruction_bound=bound,
                        original_KKT_receipts_checked=True, KKT_not_recomputed_from_float32=True,
                        block_count=len(expected_files)//2, foreground_pixels=count)


def review_nnls(output, design, A, b, mask, remote):
    import numpy as np
    directory = output / 'nnls_rho'
    done = read(directory / 'nnls_complete.json')
    fingerprint = stage_fingerprint(output, 'nnls_rho')
    require(done['status'] == 'NNLS_COMPLETE' and done['fingerprint'] == fingerprint
            and done['arrays_sha256'] == sha(directory / 'learned_arrays.npz')
            and done['all_pixels_KKT_checked_before_float32'] is True, 'NNLS_COMPLETE_CHANGED')
    X, Bhat, detail = audit_nnls_arrays(directory, A, mask, fingerprint)
    require(done['KKT'] == detail['KKT'], 'NNLS_KKT_AGGREGATION_CHANGED')
    means = X[:, mask].mean(axis=1, dtype=np.float64)
    reported = np.flatnonzero(means > .001).tolist()
    require({p.name for p in (directory / 'rho').iterdir()} == {f'candidate_{i:04d}.json' for i in reported},
            'RHO_MEMBERSHIP_CHANGED')
    scores = {}
    for i in reported:
        record = read(directory / 'rho' / f'candidate_{i:04d}.json')
        require(record['fingerprint'] == fingerprint and record['candidate_index'] == i
                and record['b_sha256'] == engine.ah(b) and record['arrays_sha256'] == done['arrays_sha256'],
                'RHO_BINDING_CHANGED')
        values = record['result']
        require(all(math.isfinite(v) for v in values.values()), 'NONFINITE_RHO')
        expected_signal = max(float(b @ b) + 1e-12, 1e-12)
        require(math.isclose(values['signal_norm2_plus_epsilon'], expected_signal, rel_tol=1e-12, abs_tol=1e-15)
                and math.isclose(values['rho_zero'], max(0., (values['q_deleted'] - values['q_star']) / expected_signal),
                                 rel_tol=1e-12, abs_tol=1e-22), 'RHO_FORMULA_CHANGED')
        scores[i] = values
    candidates, rows = make_records(output, means, scores)
    require(util.close_json(read(directory / 'candidate_records.json'), candidates)
            and util.close_json(read(directory / 'molecular_records.json'), rows), 'RAW_RECORDS_CHANGED')
    paths = {'learned_arrays.npz', 'nnls_complete.json', 'candidate_records.json', 'molecular_records.json'}
    paths |= {p.relative_to(directory).as_posix() for sub in ('nnls_blocks', 'rho') for p in (directory / sub).iterdir()}
    completion = check_completion(output, 'nnls_rho', paths)
    require(completion['fingerprint'] == fingerprint and completion['foreground_pixels'] == int(mask.sum()),
            'RAW_STAGE_DIMENSIONS_CHANGED')
    if remote:
        _, B, _ = source_arrays(design['source_prepared_path'])
        require(util.close_json(done['residual'], util.residual_summary(B, Bhat, mask)), 'RAW_RESIDUAL_CHANGED')
    return dict(**detail, raw_counts=physical.counts(rows, [r['lipid_name'] for r in rows if r['raw_solver_reported']]),
                residual_checked_against_original_B=remote, rho_recomputed=False,
                completion_sha256=sha(directory / 'completion.json'))


def review_physical(output, A, b):
    import numpy as np
    from review_physical_block_prediction import review_block
    directory = output / 'physical_features'
    manifest = read(output / 'prepared/block_manifest.json')
    expected_files = {f'blocks/block_{i:03d}{ext}' for i in range(34) for ext in ('.npz', '.json')}
    completion = check_completion(output, 'physical_features', expected_files | {'features.json'})
    require(completion['block_count'] == 34 and completion['requested_models'] == 12852
            and {p.relative_to(directory).as_posix() for p in (directory / 'blocks').iterdir()} == expected_files,
            'PHYSICAL_BLOCK_MEMBERSHIP_CHANGED')
    arrays, audits = [], []
    for i, block in enumerate(manifest['blocks']):
        record = read(directory / 'blocks' / f'block_{i:03d}.json')
        require(record['block_index'] == i and record['binding'] == stage_binding(output)
                and record['arrays_sha256'] == sha(directory / 'blocks' / f'block_{i:03d}.npz'),
                'PHYSICAL_BLOCK_BINDING_CHANGED')
        with np.load(directory / 'blocks' / f'block_{i:03d}.npz', allow_pickle=False) as z:
            values = {key: z[key] for key in z.files}
        audits.append(review_block(A, b, manifest['candidate_names'], block['channel_indices'], values))
        arrays.append(values)
    expected = physical.features_from_blocks(A, b, manifest['candidate_names'], manifest, arrays)
    require(util.close_json(read(directory / 'features.json'), expected), 'PHYSICAL_FEATURE_REPLAY_CHANGED')
    return dict(completion_sha256=sha(directory / 'completion.json'), block_count=34,
                block_cache_reviews=audits, features_replayed_without_fitting=True,
                feature_comparison='exact membership/booleans; numeric rtol1e-12 atol1e-15; original S/C formula unchanged')


def review(a):
    design, A, b, mask = validate(a.output, remote=a.remote, need_model=a.stage != 'prepared')
    if a.stage == 'prepared':
        detail = dict(prepared_hashes_checked=True, source_hashes_checked=a.remote, inference_not_run=True)
    elif a.stage == 'nnls_rho':
        detail = review_nnls(a.output, design, A, b, mask, a.remote)
    elif a.stage == 'physical_features':
        detail = review_physical(a.output, A, b)
    elif a.stage == 'pool_refit':
        detail = review_refit(a.output, design, A, b, mask, a.remote)
    else:
        raise RuntimeError('EXPLICIT_REVIEW_STAGE_REQUIRED')
    result = dict(status='CACHE_REVIEW_PASS', case=CASE, stage=a.stage,
        design_sha256=sha(a.output / 'design.json'), remote_source_checked=a.remote,
        reviewer_runtime=physical.runtime(), independent_FDR_claim=False, **detail)
    if not a.remote and a.stage in ('nnls_rho', 'pool_refit'):
        remote_path = a.output / 'reviews' / (a.stage + '_remote.json')
        remote = read(remote_path)
        require(remote['status'] == 'CACHE_REVIEW_PASS' and remote['remote_source_checked'] is True
                and remote['design_sha256'] == result['design_sha256']
                and remote['completion_sha256'] == result['completion_sha256'], 'REMOTE_RESIDUAL_REVIEW_NOT_BOUND')
        result['remote_review_sha256'] = sha(remote_path)
        result['original_B_residual_local_status'] = 'bound remote review; not independently recomputed locally'
    write(a.output / 'reviews' / (a.stage + ('_remote.json' if a.remote else '_local.json')), result)
    print('CACHE_REVIEW_PASS', a.stage, flush=True)


def require_stage_review(output, stage):
    path = output / 'reviews' / (stage + '_remote.json')
    record = read(path)
    require(record['status'] == 'CACHE_REVIEW_PASS' and record['stage'] == stage
            and record['design_sha256'] == sha(output / 'design.json')
            and record['completion_sha256'] == sha(output / stage / 'completion.json'), 'UNREVIEWED_STAGE:' + stage)
    return sha(path)


def scored_selection(output):
    rows = read(output / 'nnls_rho/molecular_records.json')
    features = read(output / 'physical_features/features.json')
    model_info = score_module().validate_model_directory(output / 'model')
    thresholds = model_info['thresholds']
    scores = score_module().score_rows(rows, features, model_info['model'])
    require(len(scores) == len(rows), 'SCORE_MEMBERSHIP_CHANGED')
    def selected(values, cut):
        return [r['lipid_name'] for r, s in zip(rows, values)
                if cut is not None and r['raw_solver_reported'] and s is not None and s >= cut]
    selected_sets = {label: selected(scores, thresholds[key]) for label, key in
                     (('pool', 'pool_threshold'), ('fdp5', 'fdp5_threshold'), ('fdp1', 'fdp1_threshold'))}
    old_model = read(output / 'model/reference_model.json')
    old_scores = physical.score_rows(rows, features, old_model)
    old_thresholds = thresholds['old_model_thresholds']
    old_sets = {label: selected(old_scores, old_thresholds[key]) for label, key in
               (('pool', 'pool_threshold'), ('fdp5', 'fdp5_threshold'), ('fdp1', 'fdp1_threshold'))}
    return dict(status='FIXED_MODEL_SELECTION_COMPLETE', case=CASE, binding=stage_binding(output),
        source_records_sha256=sha(output / 'nnls_rho/molecular_records.json'),
        source_features_sha256=sha(output / 'physical_features/features.json'),
        thresholds=thresholds, pool_names=selected_sets['pool'], selected_names=selected_sets,
        counts={key: physical.counts(rows, value) for key, value in selected_sets.items()},
        old_model_selected_names=old_sets, old_model_counts={key: physical.counts(rows, value) for key, value in old_sets.items()},
        rows=[dict(r, corrected_score=s, old_model_score=old, in_pool=r['lipid_name'] in selected_sets['pool'])
              for r, s, old in zip(rows, scores, old_scores)], independent_FDR_claim=False)


def score(a):
    validate(a.output, remote=a.remote, need_model=True)
    for stage in ('nnls_rho', 'physical_features'):
        require_stage_review(a.output, stage)
    selection = scored_selection(a.output)
    write(a.output / 'selection.json', selection)
    write(a.output / 'selection_seal.json', dict(selection_sha256=sha(a.output / 'selection.json'),
                                               binding=stage_binding(a.output)))
    print('SELECTION_COMPLETE', selection['counts'], flush=True)


def ack(a):
    validate(a.output, remote=True, need_model=True)
    full_commit(a.commit)
    require(util.close_json(read(a.output / 'selection.json'), scored_selection(a.output)), 'SAVED_SELECTION_CHANGED')
    require(read(a.output / 'selection_seal.json') == dict(selection_sha256=sha(a.output / 'selection.json'),
            binding=stage_binding(a.output)), 'SELECTION_SEAL_CHANGED')
    reviews = {stage: require_stage_review(a.output, stage) for stage in ('nnls_rho', 'physical_features')}
    write(a.output / 'case_handoff.json', dict(status='REVIEWED_DOWNLOADED_PUSHED_CASE_BEFORE_REFIT',
        commit=a.commit, reviews=reviews, selection_sha256=sha(a.output / 'selection.json'),
        selection_seal_sha256=sha(a.output / 'selection_seal.json'), binding=stage_binding(a.output)))
    print('CASE_SAVED_HANDOFF', flush=True)


def require_handoff(output):
    handoff = read(output / 'case_handoff.json')
    require(full_commit(handoff['commit']) == handoff['commit'] and handoff['binding'] == stage_binding(output)
            and handoff['selection_sha256'] == sha(output / 'selection.json')
            and handoff['selection_seal_sha256'] == sha(output / 'selection_seal.json')
            and handoff['reviews'] == {s: require_stage_review(output, s) for s in ('nnls_rho', 'physical_features')},
            'CASE_HANDOFF_CHANGED')
    require(read(output / 'selection_seal.json') == dict(selection_sha256=sha(output / 'selection.json'),
                                                       binding=stage_binding(output)), 'SELECTION_SEAL_CHANGED')


def pool_account(rows, names, selected, means):
    accounting = util.account(rows, names, selected, [], means)
    for key in ('candidate_records', 'molecular_records'):
        for row in accounting[key]:
            row.pop('old_strict_score_eligible')
            row.pop('old_strict_intersection')
    for key in ('old_strict_score_eligible', 'old_strict_intersection'):
        accounting['counts'].pop(key)
    for key in ('old_strict_removed_truth', 'old_strict_removed_false'):
        accounting['transitions'].pop(key)
    return accounting


def refit_binding(output):
    return dict(**stage_binding(output), selection_sha256=sha(output / 'selection.json'),
                case_handoff_sha256=sha(output / 'case_handoff.json'))


def pool_refit(a):
    import numpy as np
    from refit_screened_nnls import fit_screened
    design, _, _, mask = validate(a.output, remote=True, need_model=True)
    require_handoff(a.output)
    selection = read(a.output / 'selection.json')
    require(util.close_json(selection, scored_selection(a.output)), 'SAVED_SELECTION_CHANGED')
    directory = a.output / 'pool_refit'
    directory.mkdir(exist_ok=True)
    if (directory / 'completion.json').exists():
        print('POOL_REFIT_ALREADY_COMPLETE_REVIEW_CACHE', flush=True)
        return
    if not selection['pool_names']:
        write(directory / 'completion.json', dict(status='NOT_RUN_EMPTY_OR_ABSENT_POOL', case=CASE,
            stage='pool_refit', binding=refit_binding(a.output), reason='Frozen corrected pool is absent/empty; no invented fit outcome',
            independent_FDR_claim=False))
        print('NO_REFIT_EMPTY_POOL', flush=True)
        return
    metadata = read(a.output / 'prepared/source_metadata.json')
    names = metadata['lipid_name']
    indices = [i for i, name in enumerate(names) if name in selection['pool_names']]
    A, B, original_mask = source_arrays(design['source_prepared_path'])
    require((mask == original_mask).all(), 'REFIT_MASK_CHANGED')
    started = time.monotonic()
    result = fit_screened(A, B, mask, indices, directory / 'fit', workers=4, block_size=250,
                          source_binding=refit_binding(a.output))
    elapsed = time.monotonic() - started
    rows = read(a.output / 'nnls_rho/molecular_records.json')
    accounting = pool_account(rows, names, selection['pool_names'], result['means'])
    write(directory / 'candidate_records.json', accounting.pop('candidate_records'))
    write(directory / 'molecular_records.json', accounting.pop('molecular_records'))
    with np.load(result['arrays_path'], allow_pickle=False) as z:
        residual = util.residual_summary(B, z['B_hat'], mask)
    write(directory / 'completion.json', dict(status='COMPLETE', case=CASE, stage='pool_refit',
        binding=refit_binding(a.output), **accounting, residual=residual, elapsed_seconds=elapsed,
        matching_cached_blocks_reused=bool(result['resumed']), elapsed_scope='current helper invocation',
        artifacts={p: sha(directory / p) for p in ('candidate_records.json', 'molecular_records.json',
            'fit/binding.json', 'fit/completion.json', 'fit/learned_arrays.npz')}, independent_FDR_claim=False))
    print('POOL_REFIT_COMPLETE', accounting['counts'], flush=True)


def review_refit(output, design, A, b, mask, remote):
    import numpy as np
    require_handoff(output)
    selection = read(output / 'selection.json')
    require(util.close_json(selection, scored_selection(output)), 'SELECTION_REPLAY_CHANGED')
    directory = output / 'pool_refit'
    completion = read(directory / 'completion.json')
    require(completion['case'] == CASE and completion['stage'] == 'pool_refit'
            and completion['binding'] == refit_binding(output), 'REFIT_COMPLETION_BINDING_CHANGED')
    if completion['status'] == 'NOT_RUN_EMPTY_OR_ABSENT_POOL':
        require(not selection['pool_names'] and not (directory / 'fit').exists(), 'INVALID_EMPTY_POOL_STOP')
        return dict(completion_sha256=sha(directory / 'completion.json'), no_fit_performed=True,
                    reason=completion['reason'], residual_checked_against_original_B=False)
    require(completion['status'] == 'COMPLETE', 'INCOMPLETE_REFIT')
    expected_paths = {'candidate_records.json', 'molecular_records.json', 'fit/binding.json',
                      'fit/completion.json', 'fit/learned_arrays.npz'}
    require(set(completion['artifacts']) == expected_paths, 'REFIT_ARTIFACT_MEMBERSHIP_CHANGED')
    for path, digest in completion['artifacts'].items():
        require(sha(directory / path) == digest, 'REFIT_ARTIFACT_CHANGED:' + path)
    names = read(output / 'prepared/source_metadata.json')['lipid_name']
    indices = [i for i, name in enumerate(names) if name in selection['pool_names']]
    fit = directory / 'fit'
    binding, receipt = read(fit / 'binding.json'), read(fit / 'completion.json')
    expected_scientific = dict(version=1, **design['source_arrays'], retained_indices=indices,
        source_binding=refit_binding(output), solver='existing scipy NNLS; float64 solve, float32 storage',
        maxiter=3910, workers=4, block_size=250, blas_threads=1,
        implementation={Path(p).name: sha(ROOT / p) for p in ('analysis/refit_screened_nnls.py',
            'analysis/run_nnls_solver_baseline.py', 'analysis/run_small_mismatch_nnls_first_case.py')},
        numpy=design['runtime']['numpy'], scipy=design['runtime']['scipy'])
    fingerprint = hashlib.sha256(util.canonical(expected_scientific)).hexdigest()
    require(binding == dict(fingerprint=fingerprint, scientific=expected_scientific), 'REFIT_SCIENTIFIC_BINDING_CHANGED')
    require(receipt['status'] == 'COMPLETE' and receipt['fingerprint'] == fingerprint
            and receipt['binding_sha256'] == sha(fit / 'binding.json')
            and receipt['arrays_sha256'] == sha(fit / 'learned_arrays.npz')
            and receipt['retained_indices'] == indices and receipt['candidate_count'] == 391
            and receipt['fitted_column_count'] == len(indices) and receipt['foreground_pixels'] == int(mask.sum())
            and receipt['all_outputs_finite'] is True and receipt['all_pixels_checked_before_float32'] is True,
            'REFIT_RECEIPT_CHANGED')
    require(read(fit / 'status.json') == dict(fingerprint=fingerprint, status='COMPLETE',
            pixels_done=int(mask.sum()), total_pixels=int(mask.sum()), completion_sha256=sha(fit / 'completion.json')),
            'REFIT_STATUS_CHANGED')
    X, Bhat, detail = audit_nnls_arrays(fit, A, mask, fingerprint, indices, receipt)
    require(receipt['KKT'] == detail['KKT'], 'REFIT_KKT_AGGREGATE_CHANGED')
    rows = read(output / 'nnls_rho/molecular_records.json')
    expected = pool_account(rows, names, selection['pool_names'], X[:, mask].mean(axis=1, dtype=np.float64))
    require(util.close_json(read(directory / 'candidate_records.json'), expected.pop('candidate_records'))
            and util.close_json(read(directory / 'molecular_records.json'), expected.pop('molecular_records')),
            'REFIT_RECORDS_CHANGED')
    require(all(util.close_json(completion[key], value) for key, value in expected.items()), 'REFIT_ACCOUNTING_CHANGED')
    if remote:
        _, B, _ = source_arrays(design['source_prepared_path'])
        require(util.close_json(completion['residual'], util.residual_summary(B, Bhat, mask)), 'REFIT_RESIDUAL_CHANGED')
    return dict(**detail, completion_sha256=sha(directory / 'completion.json'), counts=expected['counts'],
                residual_checked_against_original_B=remote, candidate_count=391, molecular_count=377)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'bind_model', 'nnls_rho', 'physical_features',
                                         'review', 'score', 'ack', 'pool_refit'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--source', type=Path,
                        default=Path('/root/identity_confidence_joint_validation/results/cases/HOLD1'))
    parser.add_argument('--physical-parent', type=Path, default=Path('/root/physical_block_prediction_v1/results'))
    parser.add_argument('--model-dir', type=Path)
    parser.add_argument('--commit', help='Externally verified full pushed model/case snapshot SHA')
    parser.add_argument('--stage', choices=('prepared', 'nnls_rho', 'physical_features', 'pool_refit'))
    parser.add_argument('--remote', action='store_true', help='Read original remote source hashes/B for cached review')
    args = parser.parse_args()
    if args.action == 'review':
        require(args.stage is not None, 'REVIEW_STAGE_REQUIRED')
    globals()[args.action](args)


if __name__ == '__main__':
    main()
