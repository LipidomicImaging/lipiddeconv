"""User-authorized CHECK pool refit; the old strict score is descriptive only.

Preparation and execution require the original remote source. Cached local
review uses the downloaded parent A, saved mask and refit arrays; it never
dereferences an original remote source path. No review path calls a solver.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import time

if __package__:
    from . import run_physical_block_prediction as parent_engine
else:
    import run_physical_block_prediction as parent_engine

ROOT = parent_engine.ROOT
read, sha, write, require = (parent_engine.read, parent_engine.sha,
                             parent_engine.write, parent_engine.require)
CODE = tuple(parent_engine.CODE) + (
    'analysis/run_physical_block_pool_refit.py',
    'docs/PHYSICAL_BLOCK_POOL_REFIT_USER_AMENDMENT_V1.md',)
PARENT_FILES = (
    'design.json', 'design_seal.json', 'expected_sources.json',
    'block_manifest.json', 'model.json', 'model_seal.json', 'thresholds.json',
    'selection.json', 'selection_seal.json', 'remote_model_review.json',
    'prepared/CHECK.npz', 'prepared/CHECK_records.json',
    'cases/CHECK/remote_review.json', 'cases/CHECK/completion.json',
    'cases/CHECK/features.json', 'handoffs/design.json',
    'handoffs/DEV_TRAIN.json', 'handoffs/model.json', 'handoffs/CHECK.json')
VERSION = 'PHYSICAL_BLOCK_POOL_REFIT_USER_AMENDMENT_V1'
GATE = .001


def array_binding(array):
    import numpy as np
    array = np.asarray(array)
    return dict(shape=list(array.shape), dtype=array.dtype.str,
                sha256=hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest())


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('utf-8')


def parent_inputs(parent, remote=False):
    """Validate already saved parent results without refitting or rescoring."""
    design = parent_engine.validate(parent, remote=remote)
    for case in ('design', 'model', 'CHECK'):
        parent_engine.require_saved(parent, case)
    require(read(parent / 'cases/CHECK/remote_review.json')['status'] == 'CACHE_REVIEW_PASS',
            'PARENT_CHECK_NOT_REVIEWED')
    selection = read(parent / 'selection.json')
    seal = read(parent / 'selection_seal.json')
    require(seal == dict(selection_sha256=sha(parent / 'selection.json'),
                        model_seal_sha256=sha(parent / 'model_seal.json'),
                        check_features_sha256=sha(parent / 'cases/CHECK/features.json')),
            'PARENT_SELECTION_SEAL_CHANGED')
    require(selection['status'] == 'CHECK_SELECTION_FROZEN'
            and selection['model_sha256'] == sha(parent / 'model.json')
            and selection['thresholds'] == read(parent / 'thresholds.json'),
            'PARENT_SELECTION_BINDING_CHANGED')
    rows = read(parent / 'prepared/CHECK_records.json')
    names = read(parent / 'block_manifest.json')['candidate_names']
    molecular_names = list(dict.fromkeys(names))
    require(len(names) == 391 and len(molecular_names) == 377
            and [r['lipid_name'] for r in rows] == molecular_names
            and sum(bool(r['molecular_truth']) for r in rows) == 125,
            'PARENT_UNIVERSE_CHANGED')
    pool, strict = selection['pool_names'], selection['final_score_eligible_names']
    require(pool == [n for n in molecular_names if n in set(pool)]
            and strict == [n for n in molecular_names if n in set(strict)]
            and set(strict) <= set(pool)
            and set(pool) <= {r['lipid_name'] for r in rows if r['raw_solver_reported']},
            'INVALID_FROZEN_MEMBERSHIP')
    require(selection['pool'] == parent_engine.counts(rows, pool)
            and selection['final_score_eligible'] == parent_engine.counts(rows, strict),
            'PARENT_SELECTION_COUNTS_CHANGED')
    require(len(selection['rows']) == len(rows), 'SELECTION_ROW_COUNT_CHANGED')
    for original, saved in zip(rows, selection['rows']):
        require(all(saved.get(k) == v for k, v in original.items())
                and saved['in_pool'] == (original['lipid_name'] in pool)
                and saved['final_score_eligible'] == (original['lipid_name'] in strict),
                'SELECTION_ROW_CHANGED')
    for row in rows:
        indices = [i for i, name in enumerate(names) if name == row['lipid_name']]
        reported = row['reported_candidate_indices']
        require(row['candidate_indices'] == indices and len(set(reported)) == len(reported)
                and set(reported) <= set(indices)
                and bool(reported) == bool(row['raw_solver_reported']), 'ALIAS_MAP_CHANGED')
    return design, selection, rows, names


def source_arrays(parent, source_path, expected_hash):
    import numpy as np
    require(sha(source_path) == expected_hash, 'ORIGINAL_PREPARED_SOURCE_CHANGED')
    with np.load(source_path, allow_pickle=False) as z:
        A, B, mask = z['A_solver'], z['B'], z['mask']
    require(A.shape == (1084, 391) and B.shape == (1084, 200, 90)
            and mask.shape == (200, 90) and mask.dtype == bool
            and int(mask.sum()) == 15837, 'SOURCE_SHAPE_CHANGED')
    require(np.isfinite(A).all() and np.isfinite(B).all()
            and np.all(B[:, ~mask] == 0), 'INVALID_SOURCE_ARRAYS')
    with np.load(parent / 'prepared/CHECK.npz', allow_pickle=False) as z:
        require(np.array_equal(A.astype(np.float64), z['A']), 'PARENT_NOMINAL_A_CHANGED')
        require(np.allclose(B[:, mask].mean(axis=1, dtype=np.float64), z['b'],
                            rtol=1e-13, atol=1e-15), 'PARENT_MEAN_OBSERVATION_CHANGED')
    return A, B, mask


def prepared_records(rows, names, pool, strict):
    lookup = {r['lipid_name']: r for r in rows}
    molecular = [dict(r, in_pool=r['lipid_name'] in pool,
                      old_strict_score_eligible=r['lipid_name'] in strict) for r in rows]
    candidates = [dict(candidate_index=i, lipid_name=name,
                       molecular_truth=bool(lookup[name]['molecular_truth']),
                       raw_solver_reported=i in lookup[name]['reported_candidate_indices'],
                       in_pool=name in pool, old_strict_score_eligible=name in strict)
                  for i, name in enumerate(names)]
    return molecular, candidates


def prepare(a):
    import numpy as np
    require(not a.output.exists(), 'AMENDMENT_OUTPUT_ALREADY_EXISTS')
    parent_design, selection, rows, names = parent_inputs(a.parent, remote=True)
    expected = read(a.parent / 'expected_sources.json')
    source = str(Path(expected['cases']['CHECK']) / 'prepared_arrays.npz')
    source_hash = parent_design['source_hashes'][source]
    A, B, mask = source_arrays(a.parent, source, source_hash)
    pool, strict = selection['pool_names'], selection['final_score_eligible_names']
    molecular, candidates = prepared_records(rows, names, set(pool), set(strict))
    a.output.mkdir(parents=True)
    parent_engine.save_npz(a.output / 'prepared/mask.npz', mask=mask)
    write(a.output / 'prepared/pool_molecular_records.json', molecular)
    write(a.output / 'prepared/pool_candidate_records.json', candidates)
    design = dict(version=VERSION, case='CHECK', runtime=parent_engine.runtime(),
                  code={p: sha(ROOT / p) for p in CODE},
                  parent_artifacts={p: sha(a.parent / p) for p in PARENT_FILES},
                  source_prepared_path=source, source_prepared_sha256=source_hash,
                  source_arrays=dict(A=array_binding(A), B=array_binding(B), mask=array_binding(mask)),
                  prepared={p: sha(a.output / p) for p in (
                      'prepared/mask.npz', 'prepared/pool_molecular_records.json',
                      'prepared/pool_candidate_records.json')},
                  candidate_names=names, molecular_names=[r['lipid_name'] for r in rows],
                  pool_names=pool, old_strict_score_eligible_names=strict,
                  retained_indices=[i for i, name in enumerate(names) if name in pool],
                  workers=4, block_size=250, maxiter=3910, blas_threads=1,
                  foreground_pixels=int(mask.sum()), candidate_count=391, molecular_count=377,
                  all_truth_count=125, gate=GATE, wall_budget_seconds=14400,
                  primary_rule='frozen pool AND refit candidate mean > 0.001; molecular any alias',
                  secondary_rule='primary AND old frozen strict final-score eligibility; descriptive only',
                  budget_enforcement='external process-group timeout; preserve partials',
                  no_candidate_reentry=True, no_new_classifier=True, no_rho=True,
                  independent_FDR_claim=False)
    write(a.output / 'design.json', design)
    write(a.output / 'design_seal.json', dict(status='FROZEN_BEFORE_POOL_REFIT',
          design_sha256=sha(a.output / 'design.json')))
    validate(a.parent, a.output, remote=True)
    print('AMENDMENT_PREPARED', sha(a.output / 'design_seal.json'), flush=True)


def validate(parent, output, remote=False):
    import numpy as np
    _, selection, rows, names = parent_inputs(parent, remote=remote)
    design = read(output / 'design.json')
    require(read(output / 'design_seal.json') == dict(status='FROZEN_BEFORE_POOL_REFIT',
            design_sha256=sha(output / 'design.json')), 'AMENDMENT_SEAL_CHANGED')
    require(design['version'] == VERSION and design['case'] == 'CHECK', 'WRONG_AMENDMENT')
    require(design['code'] == {p: sha(ROOT / p) for p in CODE}, 'AMENDMENT_CODE_CHANGED')
    require(design['parent_artifacts'] == {p: sha(parent / p) for p in PARENT_FILES},
            'AMENDMENT_PARENT_BINDING_CHANGED')
    for path, digest in design['prepared'].items():
        require(sha(output / path) == digest, 'AMENDMENT_PREPARED_CHANGED:' + path)
    pool, strict = selection['pool_names'], selection['final_score_eligible_names']
    require(design['candidate_names'] == names and design['molecular_names'] == [r['lipid_name'] for r in rows]
            and design['pool_names'] == pool and design['old_strict_score_eligible_names'] == strict
            and design['retained_indices'] == [i for i, name in enumerate(names) if name in pool],
            'AMENDMENT_MEMBERSHIP_CHANGED')
    molecular, candidates = prepared_records(rows, names, set(pool), set(strict))
    require(read(output / 'prepared/pool_molecular_records.json') == molecular
            and read(output / 'prepared/pool_candidate_records.json') == candidates,
            'AMENDMENT_RECORDS_CHANGED')
    require((design['workers'], design['block_size'], design['maxiter'], design['blas_threads'],
             design['gate'], design['foreground_pixels'], design['all_truth_count'])
            == (4, 250, 3910, 1, GATE, 15837, 125), 'AMENDMENT_RULE_CHANGED')
    with np.load(output / 'prepared/mask.npz', allow_pickle=False) as z:
        require(set(z.files) == {'mask'}, 'MASK_ARCHIVE_MEMBERSHIP_CHANGED')
        mask = z['mask']
    with np.load(parent / 'prepared/CHECK.npz', allow_pickle=False) as z:
        A = z['A'].astype(np.dtype(design['source_arrays']['A']['dtype']))
    require(array_binding(A) == design['source_arrays']['A']
            and array_binding(mask) == design['source_arrays']['mask'], 'PORTABLE_SOURCE_BINDING_CHANGED')
    if remote:
        require(parent_engine.runtime() == design['runtime'], 'AMENDMENT_RUNTIME_CHANGED')
        actual = source_arrays(parent, design['source_prepared_path'], design['source_prepared_sha256'])
        require({key: array_binding(value) for key, value in zip(('A', 'B', 'mask'), actual)}
                == design['source_arrays'], 'ORIGINAL_ARRAY_BINDINGS_CHANGED')
    return design, rows, A, mask


def source_binding(output):
    return dict(amendment_design_sha256=sha(output / 'design.json'),
                amendment_seal_sha256=sha(output / 'design_seal.json'),
                run_authorization_sha256=sha(output / 'run_authorization.json'))


def account(rows, names, pool, strict, means):
    """Pure accounting; truth is used only after the fixed fit/reporting gate."""
    import numpy as np
    means = np.asarray(means)
    require(means.shape == (len(names),) and np.isfinite(means).all() and (means >= 0).all(),
            'INVALID_REFIT_MEANS')
    pool, strict = set(pool), set(strict)
    require(strict <= pool, 'STRICT_OUTSIDE_POOL')
    require(all(means[i] == 0 for i, name in enumerate(names) if name not in pool), 'CANDIDATE_REENTRY')
    raw = {r['lipid_name'] for r in rows if r['raw_solver_reported']}
    truth = {r['lipid_name'] for r in rows if r['molecular_truth']}
    reported = {name for i, name in enumerate(names) if means[i] > GATE}
    secondary = reported & strict
    molecular, candidates = prepared_records(rows, names, pool, strict)
    for record in candidates:
        i, name = record['candidate_index'], record['lipid_name']
        record.update(refit_mean=float(means[i]), refit_reported=bool(means[i] > GATE),
                      old_strict_intersection=bool(means[i] > GATE and name in strict))
    for record in molecular:
        name = record['lipid_name']
        active = [i for i in record['candidate_indices'] if means[i] > GATE]
        record.update(refit_reported=name in reported, refit_reported_candidate_indices=active,
                      refit_abundance=sum(float(means[i]) for i in active),
                      old_strict_intersection=name in secondary)
    stages = dict(raw=raw, pool=pool, refit_primary=reported,
                  old_strict_score_eligible=strict, old_strict_intersection=secondary)
    outcomes = {key: parent_engine.counts(rows, value) for key, value in stages.items()}
    order = [r['lipid_name'] for r in rows]
    ordered = lambda values: [name for name in order if name in values]
    transitions = dict(raw_solver_missed_truth=ordered(truth - raw),
                       screening_removed_truth=ordered((raw - pool) & truth),
                       screening_removed_false=ordered((raw - pool) - truth),
                       refit_removed_truth=ordered((pool - reported) & truth),
                       refit_removed_false=ordered((pool - reported) - truth),
                       refit_gained_names=ordered(reported - pool),
                       old_strict_removed_truth=ordered((reported - secondary) & truth),
                       old_strict_removed_false=ordered((reported - secondary) - truth))
    primary = outcomes['refit_primary']
    flags = {label: bool(primary['FDP'] is not None and primary['FDP'] <= risk
                         and primary['all_truth_recall'] >= .8)
             for label, risk in (('primary_strict_1pct_recall80_pass', .01),
                                 ('primary_secondary_5pct_recall80_pass', .05))}
    return dict(candidate_records=candidates, molecular_records=molecular,
                counts=outcomes, transitions=transitions, endpoint_flags=flags)


def residual_summary(B, B_hat, mask, block_size=250):
    import numpy as np
    observed, fitted = B[:, mask], B_hat[:, mask]
    sse, signal = 0., 0.
    for start in range(0, observed.shape[1], block_size):
        stop = min(start + block_size, observed.shape[1])
        target = observed[:, start:stop].astype(np.float64)
        difference = target - fitted[:, start:stop].astype(np.float64)
        sse += float(np.sum(difference * difference, dtype=np.float64))
        signal += float(np.sum(target * target, dtype=np.float64))
    require(signal > 0 and math.isfinite(sse), 'INVALID_RESIDUAL')
    return dict(foreground_squared_residual=sse, foreground_observation_squared_norm=signal,
                foreground_relative_l2_residual=math.sqrt(sse / signal))


def run(a):
    import numpy as np
    if __package__:
        from .refit_screened_nnls import fit_screened
    else:
        from refit_screened_nnls import fit_screened
    require(a.commit is not None and re.fullmatch('[0-9a-f]{40}', a.commit) is not None,
            'EXTERNALLY_VERIFIED_FULL_COMMIT_REQUIRED')
    design, rows, _, mask = validate(a.parent, a.output, remote=True)
    write(a.output / 'run_authorization.json', dict(
        status='EXTERNALLY_VERIFIED_PUSHED_AMENDMENT_SNAPSHOT', commit=a.commit,
        design_sha256=sha(a.output / 'design.json'), seal_sha256=sha(a.output / 'design_seal.json'),
        parent_CHECK_handoff_sha256=sha(a.parent / 'handoffs/CHECK.json')))
    if (a.output / 'result.json').exists():
        audit_cache(a.parent, a.output, remote=True)
        print('MATCHED_COMPLETE_POOL_REFIT_CACHE', flush=True)
        return
    A, B, mask = source_arrays(a.parent, design['source_prepared_path'], design['source_prepared_sha256'])
    started = time.perf_counter()
    result = fit_screened(A, B, mask, design['retained_indices'], a.output / 'refit',
                          workers=4, block_size=250, source_binding=source_binding(a.output))
    elapsed = time.perf_counter() - started
    accounting = account(rows, design['candidate_names'], design['pool_names'],
                         design['old_strict_score_eligible_names'], result['means'])
    with np.load(result['arrays_path'], allow_pickle=False) as z:
        residual = residual_summary(B, z['B_hat'], mask)
    write(a.output / 'final_candidate_records.json', accounting.pop('candidate_records'))
    write(a.output / 'final_molecular_records.json', accounting.pop('molecular_records'))
    write(a.output / 'result.json', dict(status='POOL_REFIT_COMPLETE', case='CHECK',
          **accounting, residual=residual, solver_call_elapsed_seconds=elapsed,
          matching_cached_blocks_reused=bool(result['resumed']),
          elapsed_scope='this helper invocation; cached blocks may predate it',
          source_binding=source_binding(a.output), runtime=parent_engine.runtime(),
          artifacts={p: sha(a.output / p) for p in ('final_candidate_records.json',
              'final_molecular_records.json', 'refit/binding.json', 'refit/completion.json',
              'refit/learned_arrays.npz')},
          primary_rule=design['primary_rule'], secondary_rule=design['secondary_rule'],
          independent_FDR_claim=False, prospective_validation=False))
    print('POOL_REFIT_COMPLETE', json.dumps(accounting['counts']), flush=True)


def close_json(actual, expected):
    """Only numerical aggregation differences may use a declared tolerance."""
    if isinstance(expected, dict):
        return (isinstance(actual, dict) and set(actual) == set(expected)
                and all(close_json(actual[k], value) for k, value in expected.items()))
    if isinstance(expected, list):
        return (isinstance(actual, list) and len(actual) == len(expected)
                and all(close_json(x, y) for x, y in zip(actual, expected)))
    if isinstance(expected, float):
        return (isinstance(actual, (int, float)) and not isinstance(actual, bool)
                and math.isfinite(actual) and math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-15))
    return type(actual) is type(expected) and actual == expected


def audit_cache(parent, output, remote=False):
    """Audit arrays, every checkpoint and accounting. This function never fits."""
    import numpy as np
    from threadpoolctl import threadpool_limits
    design, rows, A, mask = validate(parent, output, remote=remote)
    authorization = read(output / 'run_authorization.json')
    require(re.fullmatch('[0-9a-f]{40}', authorization['commit']) is not None
            and authorization == dict(status='EXTERNALLY_VERIFIED_PUSHED_AMENDMENT_SNAPSHOT',
                commit=authorization['commit'], design_sha256=sha(output / 'design.json'),
                seal_sha256=sha(output / 'design_seal.json'),
                parent_CHECK_handoff_sha256=sha(parent / 'handoffs/CHECK.json')), 'RUN_AUTHORIZATION_CHANGED')
    result = read(output / 'result.json')
    require(result['status'] == 'POOL_REFIT_COMPLETE' and result['case'] == 'CHECK'
            and result['source_binding'] == source_binding(output)
            and result['runtime'] == design['runtime']
            and result['primary_rule'] == design['primary_rule']
            and result['secondary_rule'] == design['secondary_rule']
            and result['independent_FDR_claim'] is False
            and result['prospective_validation'] is False
            and math.isfinite(result['solver_call_elapsed_seconds'])
            and result['solver_call_elapsed_seconds'] >= 0, 'INVALID_RESULT_BINDING')
    expected_artifacts = {'final_candidate_records.json', 'final_molecular_records.json',
                          'refit/binding.json', 'refit/completion.json', 'refit/learned_arrays.npz'}
    require(set(result['artifacts']) == expected_artifacts, 'RESULT_ARTIFACT_MEMBERSHIP_CHANGED')
    for path, digest in result['artifacts'].items():
        require(sha(output / path) == digest, 'RESULT_ARTIFACT_CHANGED:' + path)
    directory = output / 'refit'
    binding, receipt = read(directory / 'binding.json'), read(directory / 'completion.json')
    scientific = binding['scientific']
    expected_scientific = dict(version=1, **design['source_arrays'],
        retained_indices=design['retained_indices'], source_binding=source_binding(output),
        solver='existing scipy NNLS; float64 solve, float32 storage', maxiter=3910,
        workers=4, block_size=250, blas_threads=1,
        implementation={Path(p).name: sha(ROOT / p) for p in (
            'analysis/refit_screened_nnls.py', 'analysis/run_nnls_solver_baseline.py',
            'analysis/run_small_mismatch_nnls_first_case.py')},
        numpy=design['runtime']['numpy'], scipy=design['runtime']['scipy'])
    require(scientific == expected_scientific
            and binding['fingerprint'] == hashlib.sha256(canonical(scientific)).hexdigest(),
            'REFIT_SOURCE_BINDING_CHANGED')
    indices, count = design['retained_indices'], int(mask.sum())
    require(receipt['status'] == 'COMPLETE' and receipt['fingerprint'] == binding['fingerprint']
            and receipt['binding_sha256'] == sha(directory / 'binding.json')
            and receipt['arrays_sha256'] == sha(directory / 'learned_arrays.npz')
            and receipt['retained_indices'] == indices and receipt['foreground_pixels'] == count
            and receipt['candidate_count'] == len(design['candidate_names'])
            and receipt['fitted_column_count'] == len(indices)
            and receipt['all_outputs_finite'] is True
            and receipt['all_pixels_checked_before_float32'] is True, 'REFIT_COMPLETION_CHANGED')
    status = read(directory / 'status.json')
    require(status == dict(fingerprint=binding['fingerprint'], status='COMPLETE', pixels_done=count,
                           total_pixels=count, completion_sha256=sha(directory / 'completion.json')),
            'INCOMPLETE_REFIT_STATUS')
    with np.load(directory / 'learned_arrays.npz', allow_pickle=False) as z:
        require(set(z.files) == {'X_hat', 'B_hat'}, 'FINAL_ARRAY_MEMBERSHIP_CHANGED')
        X, Bhat = z['X_hat'], z['B_hat']
    require(X.shape == (391, 200, 90) and Bhat.shape == (1084, 200, 90)
            and X.dtype == Bhat.dtype == np.float32 and np.isfinite(X).all()
            and np.isfinite(Bhat).all() and (X >= 0).all()
            and np.all(X[:, ~mask] == 0) and np.all(Bhat[:, ~mask] == 0), 'INVALID_FINAL_ARRAYS')
    excluded = [i for i in range(X.shape[0]) if i not in set(indices)]
    require(np.all(X[excluded] == 0), 'EXCLUDED_CANDIDATE_REENTRY')
    flat, fitted = X[:, mask], Bhat[:, mask]
    spans = [(start, min(start + 250, count)) for start in range(0, count, 250)]
    expected_blocks = {f'block_{start:06d}_{stop:06d}.{ext}'
                       for start, stop in spans for ext in ('npz', 'json')}
    blocks = directory / 'nnls_blocks'
    require(set(receipt['block_hashes']) == expected_blocks
            and {p.name for p in blocks.iterdir()} == expected_blocks, 'BLOCK_MEMBERSHIP_CHANGED')
    overall = dict(max_dual_violation=0., max_complementarity=0., max_bound_ratio=0.)
    max_error, max_prediction = 0., 0.
    with threadpool_limits(limits=1):
        for start, stop in spans:
            stem = f'block_{start:06d}_{stop:06d}'
            for suffix in ('.npz', '.json'):
                require(sha(blocks / (stem + suffix)) == receipt['block_hashes'][stem + suffix],
                        'CHECKPOINT_HASH_CHANGED:' + stem)
            record = read(blocks / (stem + '.json'))
            require(record['fingerprint'] == binding['fingerprint']
                    and record['start'] == start and record['stop'] == stop
                    and record['retained_indices'] == indices
                    and record['array_sha256'] == receipt['block_hashes'][stem + '.npz']
                    and record['all_pixels_checked_before_float32'] is True
                    and record['empty_design_vacuous_kkt'] == (len(indices) == 0), 'CHECKPOINT_BINDING_CHANGED')
            kkt = record['KKT']
            require(set(kkt) == set(overall) and all(math.isfinite(v) and v >= 0 for v in kkt.values())
                    and kkt['max_bound_ratio'] <= 1., 'INVALID_ORIGINAL_KKT_RECEIPT')
            for key, value in kkt.items():
                overall[key] = max(overall[key], value)
            with np.load(blocks / (stem + '.npz'), allow_pickle=False) as z:
                require(set(z.files) == {'X_hat'}, 'CHECKPOINT_ARRAY_MEMBERSHIP_CHANGED')
                block = z['X_hat']
            require(block.dtype == np.float32 and block.shape == (len(indices), stop - start)
                    and np.array_equal(block, flat[indices, start:stop]), 'CHECKPOINT_FINAL_ARRAY_CHANGED')
            predicted = (A[:, indices].astype(np.float64) @ block.astype(np.float64)).astype(np.float32)
            max_error = max(max_error, float(np.max(np.abs(predicted.astype(np.float64)
                                                        - fitted[:, start:stop].astype(np.float64)))))
            max_prediction = max(max_prediction, float(np.max(np.abs(fitted[:, start:stop]))))
    require(receipt['KKT'] == overall, 'AGGREGATE_KKT_RECEIPT_CHANGED')
    reconstruction_bound = 5e-7 * max(max_prediction, 1e-12)
    require(max_error <= reconstruction_bound, 'RECONSTRUCTION_CHANGED')
    expected = account(rows, design['candidate_names'], design['pool_names'],
                       design['old_strict_score_eligible_names'], flat.mean(axis=1, dtype=np.float64))
    require(close_json(read(output / 'final_candidate_records.json'), expected.pop('candidate_records')),
            'FINAL_CANDIDATE_ACCOUNTING_CHANGED')
    require(close_json(read(output / 'final_molecular_records.json'), expected.pop('molecular_records')),
            'FINAL_MOLECULAR_ACCOUNTING_CHANGED')
    require(all(close_json(result[key], value) for key, value in expected.items()), 'FINAL_OUTCOMES_CHANGED')
    residual_review = 'remote source observation independently recomputed'
    remote_review_hash = None
    if remote:
        _, B, source_mask = source_arrays(parent, design['source_prepared_path'], design['source_prepared_sha256'])
        require(np.array_equal(source_mask, mask)
                and close_json(result['residual'], residual_summary(B, Bhat, mask)), 'RESIDUAL_CHANGED')
    else:
        remote_review = read(output / 'remote_review.json')
        require(remote_review['status'] == 'POOL_REFIT_CACHE_REVIEW_PASS'
                and remote_review['remote_source_checked'] is True
                and remote_review['result_sha256'] == sha(output / 'result.json')
                and remote_review['design_sha256'] == sha(output / 'design.json')
                and remote_review['learned_arrays_sha256'] == sha(directory / 'learned_arrays.npz'),
                'REMOTE_RESIDUAL_REVIEW_NOT_BOUND')
        remote_review_hash = sha(output / 'remote_review.json')
        residual_review = 'not recomputed locally; bound to downloaded remote source review'
    return dict(status='POOL_REFIT_CACHE_REVIEW_PASS', case='CHECK', remote_source_checked=remote,
                result_sha256=sha(output / 'result.json'), design_sha256=sha(output / 'design.json'),
                learned_arrays_sha256=sha(directory / 'learned_arrays.npz'),
                completion_sha256=sha(directory / 'completion.json'),
                reviewed_blocks=len(spans), foreground_pixels=count, candidate_count=391, molecular_count=377,
                all_final_and_checkpoint_arrays_finite=True, all_alias_membership_checked=True,
                max_reconstruction_error=max_error, reconstruction_bound=reconstruction_bound,
                original_float64_KKT_receipts_checked=True, KKT_not_recomputed_from_float32=True,
                residual_review=residual_review, remote_review_sha256=remote_review_hash,
                count_and_gate_replay='exact flags and counts; numeric aggregates rtol1e-12 atol1e-15',
                reviewer_runtime=parent_engine.runtime(), independent_FDR_claim=False,
                counts=expected['counts'])


def review(a):
    result = audit_cache(a.parent, a.output, remote=a.remote)
    write(a.output / ('remote_review.json' if a.remote else 'local_review.json'), result)
    print(result['status'], json.dumps(result['counts']), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'run', 'review'))
    parser.add_argument('--parent', type=Path, default=Path('results/physical_block_prediction_v1'))
    parser.add_argument('--output', type=Path,
                        default=Path('results/physical_block_prediction_v1/pool_refit_user_amendment'))
    parser.add_argument('--commit', help='Full externally verified pushed amendment snapshot SHA; required for run')
    parser.add_argument('--remote', action='store_true', help='Review original remote source hashes and B residual')
    args = parser.parse_args()
    globals()[args.action](args)


if __name__ == '__main__':
    main()
