"""Frozen real CE29 identity competition audit; no production result is modified."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/real_ce29_competition_audit_v1'
PARENT = ROOT / 'results/real_ce29_joint_screening_v2'
NNLS = ROOT / 'results/real_ce29_nnls_joint_screening_v2'
DEV10 = ROOT / 'results/real_ce29_dev10_threshold'
sys.path[:0] = [str(ROOT / '.venv/real_ce29_packages'), str(ROOT / 'analysis')]
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[key] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = ''


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def git(*args):
    return subprocess.check_output(['git', '-c', 'safe.directory=' + ROOT.as_posix(), *args], cwd=ROOT, text=True, encoding='utf-8').strip()


def require(ok, msg):
    if not ok:
        raise RuntimeError(msg)


def inventory():
    """Bounded source locking only; no scientific metrics or solver execution."""
    require(not (OUT / 'audit_contract.json').exists(), 'CONTRACT_ALREADY_FROZEN')
    sources = []
    evidence_commit = git('rev-parse', 'HEAD')

    def add(path, expected, role, manifest):
        path = Path(path).resolve()
        actual = sha(path) if path.is_file() else None
        item = dict(path=str(path), bytes=path.stat().st_size if actual else None,
                    sha256=actual, expected_sha256=expected, scientific_role=role,
                    expected_hash_manifest=str(manifest), source_commit=None,
                    source_commit_status='EXTERNAL_ASSET_HASH_BOUND_BY_EVIDENCE_COMMIT',
                    evidence_commit=evidence_commit, match=actual == expected)
        try:
            rel = path.relative_to(ROOT).as_posix()
            commit = git('log', '-1', '--format=%H', '--', rel)
            if commit:
                item.update(source_commit=commit, source_commit_status='LAST_COMMIT_FOR_PATH')
        except ValueError:
            pass
        sources.append(item)

    p = read(PARENT / 'input.json')
    add(PARENT / 'input.json', read(PARENT / 'input_seal.json')['sha256'], 'CE29 production source binding', PARENT / 'input_seal.json')
    for role, item in p['source_files'].items():
        add(item['path'], item['sha256'], 'CE29 original ' + role, PARENT / 'input.json')
    for name, h in p['files'].items():
        add(PARENT / name, h, 'CE29 prepared input ' + name, PARENT / 'input.json')
    for name in ('analysis/run_nnls_solver_baseline.py', 'analysis/run_small_mismatch_nnls_first_case.py'):
        add(ROOT / name, p['code'][name], 'unchanged NNLS/KKT implementation', PARENT / 'input.json')
    for base in (PARENT, NNLS, DEV10):
        manifest = base / 'final_storage_manifest.json'
        data = read(manifest)
        for name, info in data['files'].items():
            # Exact manifest membership; never recursively discover alternate assets.
            if base == DEV10 or name in ('molecular_records.json', 'candidate_records.json', 'rho.json', 'physical_features.json', 'report.json', 'nnls/learned_arrays.npz', 'nnls/binding.json', 'nnls/completion.json') or name.startswith(('blocks/', 'nnls/nnls_blocks/')):
                add(base / name, info['sha256'], base.name + ' cached ' + name, manifest)
        add(manifest, sha(manifest), 'source manifest snapshot', 'locked during bounded inventory')
    component_provenance = ROOT / 'results/small_mismatch_physical_components/provenance.json'
    data = read(component_provenance)
    write(OUT / 'provenance.json', dict(status='SOURCE_LOCKING', evidence_commit=evidence_commit,
          sources=sources, component_provenance_structure=data,
          missing_sources=[], prohibited_replacement_of_mismatched_inputs=True))
    mismatches = [x for x in sources if not x['match']]
    if mismatches:
        write(OUT / 'validation_report.json', dict(status='STOP_SOURCE_PROVENANCE_MISMATCH', mismatches=mismatches))
        raise RuntimeError('STOP_SOURCE_PROVENANCE_MISMATCH')
    print(json.dumps(dict(status='BOUNDED_SOURCE_LOCK_PASS', files=len(sources), bytes=sum(x['bytes'] for x in sources)), ensure_ascii=False), flush=True)


def lock():
    """Extend the bounded registry using the existing explicit component/V58 manifests."""
    d = read(OUT / 'provenance.json')
    seen = {x['path'] for x in d['sources']}
    head = git('rev-parse', 'HEAD')

    def add(path, expected, role, manifest):
        p = Path(path).resolve()
        if str(p) in seen:
            return
        h = sha(p)
        require(h == expected, 'STOP_SOURCE_PROVENANCE_MISMATCH:' + str(p))
        try:
            commit = git('log', '-1', '--format=%H', '--', p.relative_to(ROOT).as_posix()) or None
        except ValueError:
            commit = None
        d['sources'].append(dict(path=str(p), bytes=p.stat().st_size, sha256=h,
            expected_sha256=expected, scientific_role=role, expected_hash_manifest=str(manifest),
            source_commit=commit, source_commit_status='LAST_COMMIT_FOR_PATH' if commit else 'EXTERNAL_ASSET_HASH_BOUND_BY_EVIDENCE_COMMIT',
            evidence_commit=head, match=True))
        seen.add(str(p))

    cp = ROOT / 'results/small_mismatch_physical_components/provenance.json'
    add(NNLS / 'input.json', read(NNLS / 'input_seal.json')['sha256'], 'original full-library NNLS source binding', NNLS / 'input_seal.json')
    c = read(cp)
    add(cp, '9b6c9b7f1eb578160565a5f911524ce3fc8bf316a81449b7606792508771e85f', 'physical fragment mapping provenance', 'previous frozen component provenance')
    for path, info in c['inputs'].items():
        if Path(path).name in ('candidate_metadata_final.jsonl', 'channel_formulas.jsonl'):
            add(path, info['sha256'], 'production candidate/channel labels', cp)
    for name in ('components.npz', 'component_metadata.json', 'metadata.json', 'component_audit.json'):
        add(cp.parent / name, c['outputs'][name]['sha256'], 'physical fragment mapping ' + name, cp)
    vp = ROOT / 'results/ce_uncertainty_identity_pilot'
    transfer = vp / 'transfer_manifests/partial_01.json'
    tm = read(transfer)
    add(transfer, sha(transfer), 'existing V58 compact cache transfer manifest', 'bounded inventory snapshot')
    bindings = []
    for r in (1, 2):
        rel = f'inputs/MILD__CAL_R{r}_K125'
        for name in ('pilot_input.json', 'scoring_inputs.npz', 'molecular_false_negative_records.csv', 'candidate_false_negative_records.csv'):
            add(vp / rel / name, tm['files'][rel + '/' + name], 'V58 frozen MILD CAL input ' + name, transfer)
        pilot = read(vp / rel / 'pilot_input.json')
        bindings.append(dict(case=f'MILD__CAL_R{r}_K125', A_array_sha256=pilot['A_sha256'],
            B_cube_array_sha256=pilot['B_cube_sha256'], original_artifacts=pilot['original_artifacts'],
            role='original frozen provenance only; unavailable full arrays are not claimed locally verified',
            scoring_inputs_sha256=pilot['scoring_inputs_sha256']))
    directory_members = {}
    for role, path in read(OUT / 'audit_contract.json')['source_paths'].items():
        resolved = (ROOT / path).resolve()
        if resolved.is_dir() and role in ('spectral_blocks', 'spatial_blocks'):
            members = [x['path'] for x in d['sources'] if Path(x['path']).parent == resolved]
            require(bool(members), 'UNREGISTERED_CACHE_MEMBERS:' + role)
            directory_members[str(resolved)] = members
        else:
            require(str(resolved) in seen, 'UNREGISTERED_CONTRACT_SOURCE:' + role + ':' + path)
    d.update(status='FROZEN_BEFORE_DIAGNOSTIC', component_provenance_structure=c,
             v58_original_bindings=bindings, directory_members=directory_members,
             missing_sources=['ISTA molecular deletion cache', 'V58 MILD pixel/block observations locally'],
             unavailable_source_commit_note='For external untracked arrays source_commit is null; the evidence commit and exact source manifest bind content. No fabricated commit.')
    write(OUT / 'provenance.json', d)
    print('SOURCE_REGISTRY_FROZEN', len(d['sources']), flush=True)


def array_sha(a):
    return hashlib.sha256(a.tobytes(order='C')).hexdigest()


def expected_diagnostic_ids():
    return ['CE29_GLOBAL'] + [f'CE29_BLOCK_{i:03d}' for i in range(64)] + ['V58_MILD_CAL_R1_K125', 'V58_MILD_CAL_R2_K125']


def validate_sources():
    d = read(OUT / 'provenance.json')
    bad = []
    for item in d['sources']:
        p = Path(item['path'])
        if not p.is_file() or p.stat().st_size != item['bytes'] or sha(p) != item['sha256'] or item['sha256'] != item['expected_sha256']:
            bad.append(str(p))
    if bad:
        write(OUT / 'validation_report.json', dict(status='STOP_SOURCE_PROVENANCE_MISMATCH', mismatches=bad))
        raise RuntimeError('STOP_SOURCE_PROVENANCE_MISMATCH')
    return d


def preflight():
    import numpy as np
    import scipy
    from threadpoolctl import threadpool_limits
    require(not (OUT / 'input_binding.json').exists(), 'INPUT_ALREADY_PREPARED')
    source = validate_sources()
    code = ['analysis/run_real_ce29_competition_audit.py', 'analysis/real_ce29_competition_geometry.py',
            'analysis/review_real_ce29_competition_audit.py', 'docs/REAL_CE29_COMPETITION_AUDIT_V1.md',
            'results/real_ce29_competition_audit_v1/audit_contract.json',
            'results/real_ce29_competition_audit_v1/provenance.json']
    ack = read(OUT / 'pre_execution_git.json')
    require(ack['local_commit'] == ack['remote_commit'], 'PRE_EXECUTION_PUSH_NOT_VERIFIED')
    require(not git('diff', ack['local_commit'], '--', *code), 'FROZEN_CODE_OR_CONTRACT_CHANGED')
    for name in code:
        require(bool(git('ls-tree', ack['local_commit'], '--', name)), 'CODE_OR_CONTRACT_NOT_IN_PUSHED_COMMIT:' + name)
    require(tuple(sys.version_info[:3]) == (3, 12, 14) and
            (np.__version__, scipy.__version__) == ('2.1.3', '1.15.3'), 'RUNTIME_MISMATCH')
    with threadpool_limits(limits=1):
        p = read(PARENT / 'input.json')['source_files']
        A = np.load(p['A']['path'], allow_pickle=False)
        raw = np.load(p['raw_library']['path'], allow_pickle=False)
        require(raw.shape == (1, 1084, 391), 'RAW_LIBRARY_AXIS_MISMATCH')
        raw = raw[0]
        meta = np.load(p['metadata']['path'], allow_pickle=True).item()
        names = np.asarray(meta['lipid_name'], dtype=str)
        B = np.load(p['B']['path'], allow_pickle=False)[0].transpose(2, 0, 1)
        mask = np.load(p['mask']['path'], allow_pickle=False)
        require(A.shape == (1084, 391) and B.shape == (1084, 200, 90) and mask.shape == (200, 90) and int(mask.sum()) == 15837, 'INPUT_SHAPE_MISMATCH')
        require(np.array_equal(raw > 0, A > 0), 'RAW_PRODUCTION_SUPPORT_MISMATCH')
        require(len(dict.fromkeys(names.tolist())) == 377, 'MOLECULAR_MAPPING_MISMATCH')
        candidate = read(PARENT / 'candidate_records_input.json')
        require([x['candidate_index'] for x in candidate] == list(range(391)), 'CANDIDATE_ORDER_MISMATCH')
        require([x['lipid_name'] for x in candidate] == names.tolist(), 'CANDIDATE_NAME_MISMATCH')
        require([x['candidate_id'] for x in candidate] == list(meta['candidate_id']), 'CANDIDATE_ID_MISMATCH')
        A = A.astype(np.float64)
        spectra = B[:, mask]
        means = [spectra.mean(axis=1, dtype=np.float64)]
        ids = ['CE29_GLOBAL']
        positions = np.flatnonzero(mask.ravel())
        original_blocks = read(NNLS / 'nnls/completion.json')['block_hashes']
        block_counts = []
        for i, start in enumerate(range(0, 15837, 250)):
            end = min(start + 250, 15837)
            filename = f'block_{start:06d}_{end:06d}.npz'
            require(filename in original_blocks, 'ORIGINAL_BLOCK_MEMBERSHIP_MISMATCH')
            means.append(spectra[:, start:end].mean(axis=1, dtype=np.float64))
            block_counts.append(end-start)
            ids.append(f'CE29_BLOCK_{i:03d}')
        require(np.allclose(np.average(means[1:], axis=0, weights=block_counts), means[0], rtol=2e-15, atol=2e-15), 'BLOCK_MEAN_BINDING_MISMATCH')
        with np.load(PARENT / 'prepared.npz', allow_pickle=False) as z:
            require(np.array_equal(z['A'], A) and np.array_equal(z['b'], means[0]), 'ORIGINAL_PREPARED_INPUT_MISMATCH')
        matrices = []
        for r in (1, 2):
            case = ROOT / f'results/ce_uncertainty_identity_pilot/inputs/MILD__CAL_R{r}_K125'
            pilot = read(case / 'pilot_input.json')
            with np.load(case / 'scoring_inputs.npz', allow_pickle=False) as z:
                require(np.array_equal(z['kept'], np.arange(391)), 'V58_UNIVERSE_CHANGED')
                require(pilot['names'] == names.tolist(), 'V58_CANDIDATE_ORDER_MISMATCH')
                require(array_sha(z['A'].astype(np.float32)) == pilot['A_sha256'], 'V58_A_BINDING_MISMATCH')
                matrices.append(z['A'].copy())
                means.append(z['b'].copy())
                ids.append(f'V58_MILD_CAL_R{r}_K125')
        require(np.array_equal(matrices[0], matrices[1]), 'V58_A_DIFFERS_ACROSS_FIXED_CASES')
        bs = np.stack(means)
        require(ids == expected_diagnostic_ids() and bs.shape == (67, 1084), 'DIAGNOSTIC_CONTEXT_MEMBERSHIP_MISMATCH')
        require(np.isfinite(A).all() and np.isfinite(bs).all() and (A >= 0).all(), 'NONFINITE_INPUT')
        np.savez_compressed(OUT / 'diagnostic_inputs.npz', A=A, A_v58=matrices[0], bs=bs, names=names,
                            foreground_positions=positions, block_pixel_counts=np.asarray(block_counts))
    binding = dict(status='PASS', diagnostic_ids=ids, shape=list(bs.shape), candidate_count=391,
        molecular_count=377, foreground_pixels=15837, block_count=64,
        diagnostic_inputs_sha256=sha(OUT / 'diagnostic_inputs.npz'),
        contract_sha256=sha(OUT / 'audit_contract.json'), provenance_sha256=sha(OUT / 'provenance.json'),
        pre_execution_commit=ack['local_commit'], code_sha256={n:sha(ROOT / n) for n in code},
        source_count=len(source['sources']), A_array_sha256=array_sha(A), A_v58_array_sha256=array_sha(matrices[0]),
        b_array_sha256={name:array_sha(bs[i]) for i, name in enumerate(ids)},
        runtime=dict(python=sys.version, numpy=np.__version__, scipy=scipy.__version__),
        no_truth_in_solver_inputs=True, all_candidate_columns_preserved=True,
        observation_normalization='unchanged; mean aggregation only', new_computation='foreground-order block-mean NNLS deletion diagnostic')
    write(OUT / 'input_binding.json', binding)
    print('PREFLIGHT_PASS_391_377_15837_64', flush=True)


def fit_diagnostic(task):
    import numpy as np
    from threadpoolctl import threadpool_limits
    import run_nnls_solver_baseline as baseline
    from run_small_mismatch_nnls_first_case import kkt_check
    name, A, b, names, binding = task
    out = OUT / 'diagnostics'
    out.mkdir(parents=True, exist_ok=True)
    require(not (out / (name + '.npz')).exists(), 'EXISTING_DIAGNOSTIC_PRESERVED:' + name)
    start = time.monotonic()
    molecular = list(dict.fromkeys(names.tolist()))
    groups = [np.flatnonzero(names == n) for n in molecular]
    fields = ('max_dual_violation', 'max_complementarity', 'max_bound_ratio')
    with threadpool_limits(limits=1):
        baseline.initialize_worker(A)
        full = baseline.solve_pixel(b)
        full_kkt = kkt_check(A, b, full, np)
        rfull = b - A @ full
        q_full = float(rfull @ rfull)
        deleted = np.zeros((len(groups), A.shape[1]), dtype=float)
        q_deleted = np.zeros(len(groups))
        kkts = np.zeros((len(groups), 3))
        analytic = np.zeros(len(groups), dtype=bool)
        for j, group in enumerate(groups):
            allowed = np.setdiff1d(np.arange(A.shape[1]), group)
            reduced = A[:, allowed]
            if np.all(full[group] == 0):
                x = full.copy()
                analytic[j] = True
            else:
                baseline.initialize_worker(reduced)
                x = np.zeros(A.shape[1])
                x[allowed] = baseline.solve_pixel(b)
            checks = kkt_check(reduced, b, x[allowed], np)
            require(np.all(x[group] == 0), 'DELETED_ALIAS_NONZERO')
            residual = b - A @ x
            q = float(residual @ residual)
            bound = 256*np.finfo(float).eps*max(A.shape)*max(1., q_full, float(b @ b))
            require(q >= q_full - bound, 'DELETE_OBJECTIVE_IMPROVEMENT_OUTSIDE_ROUNDOFF')
            deleted[j] = x
            q_deleted[j] = q
            kkts[j] = [checks[k] for k in fields]
        np.savez_compressed(out / (name + '.npz'), b=b, full_x=full, deleted_x=deleted,
                            q_full=np.asarray(q_full), q_deleted=q_deleted,
                            full_kkt=np.asarray([full_kkt[k] for k in fields]), deleted_kkt=kkts,
                            analytic_zero_deletion=analytic)
    receipt = dict(diagnostic_id=name, output_sha256=sha(out / (name + '.npz')),
        bytes=(out / (name + '.npz')).stat().st_size, A_array_sha256=array_sha(A),
        b_array_sha256=array_sha(b), contract_sha256=binding['contract_sha256'],
        provenance_sha256=binding['provenance_sha256'], input_sha256=binding['diagnostic_inputs_sha256'],
        input_binding_sha256=sha(OUT / 'input_binding.json'),
        groups=[x.tolist() for x in groups], names=molecular,
        solver_calls=1+int((~analytic).sum()), analytic_zero_deletion_count=int(analytic.sum()),
        full_candidate_count=391, deleted_candidate_counts=[len(g) for g in groups],
        KKT_max_ratio=float(max(full_kkt['max_bound_ratio'], kkts[:, 2].max())),
        q_objective='unnormalized unweighted squared L2 error', seconds=time.monotonic()-start,
        completion='PASS', new_diagnostic_not_production_result=True)
    write(out / (name + '.json'), receipt)
    return {k: receipt[k] for k in ('diagnostic_id', 'solver_calls', 'analytic_zero_deletion_count', 'seconds')}


def run():
    import numpy as np
    import scipy
    from concurrent.futures import ProcessPoolExecutor, as_completed
    binding = read(OUT / 'input_binding.json')
    require(binding['diagnostic_ids'] == expected_diagnostic_ids(), 'DIAGNOSTIC_CONTEXT_MEMBERSHIP_MISMATCH')
    require(binding['runtime'] == dict(python=sys.version, numpy=np.__version__, scipy=scipy.__version__), 'RUNTIME_MISMATCH')
    require(not (OUT / 'run_started.json').exists(), 'EXISTING_RUN_PRESERVED_NO_AUTOMATIC_RETRY')
    validate_sources()
    require(sha(OUT / 'audit_contract.json') == binding['contract_sha256'], 'CONTRACT_CHANGED')
    require(sha(OUT / 'provenance.json') == binding['provenance_sha256'], 'PROVENANCE_CHANGED')
    require(sha(OUT / 'diagnostic_inputs.npz') == binding['diagnostic_inputs_sha256'], 'INPUT_CHANGED')
    for name, h in binding['code_sha256'].items():
        require(sha(ROOT / name) == h, 'FROZEN_CODE_CHANGED:' + name)
    write(OUT / 'run_started.json', dict(pid=os.getpid(), started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        input_binding_sha256=sha(OUT / 'input_binding.json'), code_commit=binding['pre_execution_commit']))
    with np.load(OUT / 'diagnostic_inputs.npz', allow_pickle=False) as z:
        require(z['bs'].shape == (67, 1084) and z['A'].shape == z['A_v58'].shape == (1084, 391)
                and z['names'].shape == (391,) and len(dict.fromkeys(z['names'].tolist())) == 377, 'DIAGNOSTIC_SHAPE_MISMATCH')
        tasks = [(name, z['A'] if i < 65 else z['A_v58'], z['bs'][i], z['names'], binding)
                 for i, name in enumerate(binding['diagnostic_ids'])]
    completed = []
    started = time.monotonic()
    executor = ProcessPoolExecutor(max_workers=4)
    futures = []
    try:
        futures = [executor.submit(fit_diagnostic, task) for task in tasks]
        for future in as_completed(futures):
            result = future.result()
            completed.append(result)
            write(OUT / 'status.json', dict(completed=len(completed), planned=67, records=completed))
            print(json.dumps(result), flush=True)
        executor.shutdown(wait=True)
        require(len(completed) == 67 and sorted(x['diagnostic_id'] for x in completed) == sorted(expected_diagnostic_ids()), 'COMPLETION_MEMBERSHIP_MISMATCH')
        write(OUT / 'diagnostic_completion.json', dict(status='PASS', planned=67, completed=67,
              elapsed_seconds=time.monotonic()-started, records=completed,
              output_hashes={p.name:sha(p) for p in sorted((OUT / 'diagnostics').iterdir())}))
    except Exception as exc:
        write(OUT / 'failure.json', dict(status='STOP_INVALID_AUDIT', error=str(exc), completed=len(completed)))
        for future in futures:
            future.cancel()
        executor.shutdown(wait=True, cancel_futures=True)
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['inventory', 'lock', 'preflight', 'run', 'summarize'])
    args = parser.parse_args()
    if args.command == 'summarize':
        from real_ce29_competition_geometry import summarize
        validate_sources()
        binding = read(OUT / 'input_binding.json')
        require(read(OUT / 'diagnostic_completion.json')['status'] == 'PASS', 'DIAGNOSTICS_INCOMPLETE')
        for name, h in binding['code_sha256'].items():
            require(sha(ROOT / name) == h, 'FROZEN_CODE_CHANGED:' + name)
        summarize(ROOT, OUT)
    else:
        globals()[args.command]()


if __name__ == '__main__':
    main()
