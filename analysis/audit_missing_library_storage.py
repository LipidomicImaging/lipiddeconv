"""Cached six-case missing-library review and exact checkpoint retirement.

Never trains, reconstructs synthetic inputs, recomputes rho, or changes cutoffs.
Unlike V58 retirement, latest_model.pth is a required result artifact and stays.
"""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import tarfile
from datetime import datetime, timezone

os.environ['CUDA_VISIBLE_DEVICES'] = ''
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'

ROOT = Path('/root/autodl-tmp/lipiddeconv')
OUT = ROOT / 'results/missing_library_challenge_k125'
PARENT = ROOT / 'results/v57_spectral_spatial_identity_confidence_benchmark'
ASSETS = ROOT / 'results/computational_closure/missing_library'
JOB = Path('/root/v58_jobs/missing_library_review_20260911')
EXPECTED = {f'{arm}__HOLD_R{r}_K125' for arm in ('close_neighbor', 'relatively_isolated') for r in (1, 2, 3)}
RETIRE_NAMES = {f'checkpoint_epoch_{e}.pth' for e in (1000, 1500, 2000, 2500)}


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def fp(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat()


def inactive():
    state = read(Path('/root/v58_jobs/storage_recovery_20260911/status.json'))
    require(state['status'] == 'ALL_COMPUTE_COMPLETE_AWAITING_REVIEW' and state.get('active_dataset') is None, 'SUPERVISOR_ACTIVE')
    state = read(OUT / 'status.json')
    require(state['status'] == 'REQUESTED_CASES_COMPLETE' and set(state['cases']) == EXPECTED, 'CASES_NOT_COMPLETE')
    # Only a bounded process-table read, not a filesystem walk.
    commands = subprocess.check_output(['ps', '-eo', 'pid,args'], text=True).splitlines()
    active = [line for line in commands if any(s in line for s in (
        'run_missing_library_challenge.py', 'run_v58_spectral_library_mismatch_fdr_recalibration.py',
        'run_ce_uncertainty_identity_pilot.py'))]
    require(not active, 'ACTIVE_DEPENDENT_JOB: ' + repr(active))


def finite(value):
    import numpy as np
    import torch
    if isinstance(value, torch.Tensor):
        return bool(torch.isfinite(value).all())
    if isinstance(value, np.ndarray):
        return bool(np.isfinite(value).all())
    if isinstance(value, dict):
        return all(finite(v) for v in value.values())
    if isinstance(value, (tuple, list)):
        return all(finite(v) for v in value)
    return math.isfinite(value) if isinstance(value, float) else True


def accounting(units, truths, reportable, score=None, threshold=None, omitted=0):
    retained = units if score is None else [u for u in units if threshold is not None and u[score] >= threshold]
    raw_tp = sum(u['molecular_truth'] for u in units)
    tp = sum(u['molecular_truth'] for u in retained)
    n = len(retained)
    false = n - tp
    d = dict(N_retained=n, TP=tp, FP=false, FN=truths-tp,
             raw_solver_TP=raw_tp, raw_solver_FP=len(units)-raw_tp, raw_solver_FN=truths-raw_tp,
             filtered_TP=tp, filtered_FP=false, filtered_FN=truths-tp,
             structural_omission_FN=omitted, additional_raw_solver_FN=truths-raw_tp-omitted,
             filter_induced_true_loss=raw_tp-tp,
             filter_induced_true_loss_fraction=(raw_tp-tp)/raw_tp if raw_tp else None,
             true_positive_retention=tp/raw_tp if raw_tp else None, TP_retention=tp/raw_tp if raw_tp else None,
             FDR=false/n if n else None, precision=tp/n if n else None,
             all_truth_recall=tp/truths, reportable_truth_recall=sum(u['reportable_truth'] for u in retained)/reportable,
             coverage=n/len(units) if units else None,
             surviving_truth_recall=tp/(truths-omitted),
             distinct_retained_identities=len({u['lipid_name'] for u in retained}),
             distinct_retained_true_identities=len({u['lipid_name'] for u in retained if u['molecular_truth']}))
    return d


def units_from_candidates(rows):
    groups = {}
    for row in rows:
        if row['raw_solver_reported']:
            groups.setdefault(row['lipid_name'], []).append(row)
    return [dict(lipid_name=n, molecular_truth=rs[0]['molecular_truth'], reportable_truth=rs[0]['reportable_truth'],
                 X_hat=sum(r['X_hat'] for r in rs), rho_zero=max(r['rho_zero'] for r in rs)) for n, rs in groups.items()]


def modes(units, count, pack, omitted):
    result = {'raw': accounting(units, count, count, omitted=omitted)}
    for score, targets in pack.items():
        for target, threshold in targets.items():
            result[score+'_'+target] = accounting(units, count, count, score,
                                                threshold['threshold'] if threshold else None, omitted)
    return result


def prepare():
    import numpy as np
    import torch
    torch.set_num_threads(1)
    inactive()
    require(not JOB.exists(), 'REVIEW_DIRECTORY_EXISTS')
    JOB.mkdir(parents=True)
    protected = {}
    compact = {}

    def keep(path, expected=None, export=False):
        h = sha(path)
        require(expected is None or h == expected, 'HASH_CHANGED: ' + str(path))
        protected[str(path)] = h
        if export:
            require(path.stat().st_size < 10_000_000, 'UNEXPECTED_LARGE_COMPACT_FILE')
            rel = 'source/' + str(path.relative_to(ROOT))
            dest = JOB / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, dest)
            require(sha(dest) == h, 'COPY_HASH_MISMATCH')
            compact[rel] = h
        return h

    design = read(OUT/'execution_design.json')
    s = design['scientific']
    require(fp(s) == design['fingerprint'], 'DESIGN_FINGERPRINT')
    require(set(s['cases']) == EXPECTED, 'DESIGN_MEMBERSHIP')
    seal = read(OUT/'execution_freeze.json')
    require(seal['status'] == 'FROZEN_BEFORE_TRAINING' and seal['fingerprint'] == design['fingerprint'], 'BAD_FREEZE')
    for n in ('execution_design.json', 'validation.json'):
        keep(OUT/n, seal['design_sha256' if n.startswith('execution') else 'validation_sha256'], True)
    for n in ('execution_freeze.json', 'status.json'):
        keep(OUT/n, export=True)
    require(read(OUT/'validation.json')['status'] == 'PASS', 'PREFLIGHT_NOT_PASS')
    parent = read(PARENT/'design.json')
    require(fp(parent['scientific']) == parent['design_fingerprint'] == s['parent_fingerprint'], 'PARENT_DESIGN_CHANGED')
    # The large frozen parent design already has documented storage; keep its
    # hash and original file without duplicating it into this compact snapshot.
    keep(PARENT/'design.json')
    selection = read(ASSETS/'selection_manifest.json')
    content = dict(selection)
    require(fp({k:v for k,v in content.items() if k != 'selection_fingerprint'}) == s['selection_fingerprint'] == selection['selection_fingerprint'], 'SELECTION_CHANGED')
    require(selection['arms'] == s['arms'], 'ARM_CHANGED')
    keep(ASSETS/'selection_manifest.json', export=True)
    keep(ASSETS/'control_reconstruction_audit.json', s['control_audit_sha256'], True)
    control_audit = read(ASSETS/'control_reconstruction_audit.json')
    require(control_audit['status'] == 'PASS', 'CONTROL_AUDIT_NOT_PASS')
    require(control_audit['production_config'] == s['production_config'], 'CONFIG_BINDING')
    for name, expected in s['production_hashes'].items():
        path = ROOT/('analysis' if name.startswith('run_v') else 'src')/name
        keep(path, expected)
    require(sha(Path('/root/run_missing_library_challenge.py')) == s['script_sha256'], 'EXECUTED_RUNNER_CHANGED')
    protected['/root/run_missing_library_challenge.py'] = s['script_sha256']
    keep(PARENT/'global_frozen_thresholds.json', s['threshold_file_sha256'], True)
    require(read(PARENT/'global_frozen_thresholds.json') == s['thresholds'], 'THRESHOLDS_CHANGED')
    require(s['thresholds']['status'] == 'FROZEN_FROM_CAL_ONLY' and s['thresholds']['design_fingerprint'] == s['parent_fingerprint'], 'BAD_CAL_SEAL')
    for dataset, files in s['thresholds']['CAL_source_hashes'].items():
        require(dataset.startswith('CAL_'), 'NON_CAL_THRESHOLD_SOURCE')
        for n, h in files.items():
            keep(PARENT/'clean'/dataset/n, h, True)
    maskpath = Path('/root/autodl-tmp/decon-lipid/adapter_pipeline/outputs/v38_ce29_fragmentionfixed_profile_overlap_empiricalfwhm_globalq99_ista_ready/foreground_pixel_mask.npy')
    keep(maskpath, parent['scientific']['input_hashes']['foreground_mask'])
    mask = np.load(maskpath).astype(bool)
    controls = {}
    control_units = {}
    pack = s['thresholds']['global_thresholds']
    for item in control_audit['datasets']:
        dataset = item['dataset_id']
        base = PARENT/'clean'/dataset
        for n, h in s['controls'][dataset].items():
            keep(base/n, h, n.endswith('.json'))
        for ck in item['checkpoints']:
            keep(base/ck['file'], ck['sha256'])
        for n in ('report.json', 'candidate_false_negative_records.csv', 'reported_identity_records.csv'):
            keep(base/n, export=True)
        with (base/'candidate_false_negative_records.csv').open() as stream:
            rows = list(csv.DictReader(stream))
        for row in rows:
            for key in ('molecular_truth', 'reportable_truth', 'raw_solver_reported'):
                row[key] = row[key].lower() == 'true'
            row['X_hat'] = float(row['X_hat'])
            row['rho_zero'] = float(row['rho_zero']) if row['raw_solver_reported'] else None
        controls[dataset] = rows
        control_units[dataset] = units_from_candidates(rows)
        report = read(base/'report.json')['learned_raw_identity_performance']
        require({r['lipid_name'] for r in rows if r['molecular_truth']} == set(report['truth_lipid_names']), 'CONTROL_TRUTH_MEMBERSHIP')
        require({r['lipid_name'] for r in rows if r['reportable_truth']} == set(report['reportable_truth_lipid_names']), 'CONTROL_REPORTABLE_MEMBERSHIP')
        metric = accounting(control_units[dataset], 125, 125)
        for name, value in metric.items():
            if name in report['molecular_level']:
                require(value == report['molecular_level'][name], 'CONTROL_METRIC_MISMATCH: '+name)

    checks, all_units, deletions, neighbors = {}, {}, [], []
    require({p.name for p in (OUT/'runs').iterdir() if p.is_dir()} == EXPECTED, 'RESULT_MEMBERSHIP')
    for key in sorted(EXPECTED):
        loc = OUT/'runs'/key
        require(not loc.is_symlink() and not (loc/'failure.json').exists(), 'UNSAFE_OR_FAILED_CASE')
        result = read(loc/'result.json')
        entry = s['cases'][key]
        arm = s['arms'][entry['arm']]
        kept = arm['reduced_to_original']
        require(len(kept) == len(set(kept)) == 386 and set(kept) == set(range(391))-set(arm['removed_original_indices']), 'BAD_MAP')
        require(result['status'] == 'COMPLETE' and result['case_id'] == key and result['fingerprint'] == design['fingerprint'], 'BAD_RESULT')
        binding = dict(fingerprint=design['fingerprint'], case_id=key, B_sha256=entry['B_sha256'],
                       A_solver_sha256=entry['A_solver_sha256'], reduced_to_original=kept)
        require(read(loc/'runtime_contract.json') == binding, 'RUNTIME_BINDING')
        require(entry['truth_count'] == entry['reportable_truth_count'] == 125, 'TRUTH_DENOMINATOR')
        for n, h in result['artifact_hashes'].items():
            require(Path(n).name == n and n not in RETIRE_NAMES, 'RESULT_DEPENDS_ON_INTERMEDIATE')
            keep(loc/n, h, n.endswith('.json'))
        keep(loc/'result.json', export=True)
        keep(loc/'checkpoint_diagnostics.csv', export=True)
        solver = read(loc/'solver_run.json')
        require(solver['stop_reason'] in ('max_epochs', 'converged') and 0 < solver['stopped_epoch'] <= 3000, 'ABNORMAL_COMPLETION')
        require(finite(solver) and finite(read(loc/'training_history.json')), 'NONFINITE_HISTORY')
        require(solver['scheduler_T_max'] == s['production_config']['n_epochs'], 'SCHEDULER_CHANGED')
        with np.load(loc/'learned_arrays.npz') as arrays:
            require(np.array_equal(arrays['reduced_to_original'], kept), 'ARRAY_MAP')
            require(arrays['X_hat'].shape == (386, *mask.shape) and arrays['B_hat'].shape == (1084, *mask.shape), 'ARRAY_SHAPES')
            require(all(finite(arrays[n]) for n in arrays.files), 'NONFINITE_ARRAY')
            means = arrays['X_hat'][:,mask].mean(axis=1, dtype=np.float64)
        epoch = solver['stopped_epoch']
        final = loc/f'checkpoint_epoch_{epoch}.pth'
        if not final.exists():
            final = loc/'latest_model.pth'
        for path in set((final, loc/'latest_model.pth')):
            keep(path)
            checkpoint = torch.load(path, map_location='cpu', weights_only=False)
            require(checkpoint['dataset_id'] == 'MISSING_LIBRARY__'+key and checkpoint['epoch'] == epoch, 'FINAL_MODEL_BINDING')
            require(checkpoint['terminal_stop_reason'] == solver['stop_reason'] and finite(checkpoint), 'FINAL_MODEL_INVALID')
            require(checkpoint['history'] == read(loc/'training_history.json'), 'FINAL_HISTORY_CHANGED')
            del checkpoint
        rows = read(loc/'candidate_records.json')
        require([r['candidate_index'] for r in rows] == list(range(391)), 'CANDIDATE_MEMBERSHIP')
        baseline = controls[entry['base_dataset']]
        require(len(baseline) == 391, 'PARENT_CANDIDATE_MEMBERSHIP')
        for i, row in enumerate(rows):
            ref = baseline[i]
            require(all(row[n] == ref[n] for n in ('candidate_id', 'lipid_name', 'molecular_truth', 'reportable_truth')), 'PARENT_IDENTITY_CHANGED')
            j = row['reduced_index']
            require(row['omitted_from_solver'] == (i not in kept), 'OMISSION_FLAG')
            require(j == (kept.index(i) if i in kept else None), 'CANDIDATE_MAP')
            require(row['X_hat'] == (float(means[j]) if j is not None else 0.), 'ABUNDANCE_RECORD_CHANGED')
            require(row['raw_solver_reported'] == (j is not None and row['X_hat'] > .001), 'REPORT_GATE_CHANGED')
            if row['raw_solver_reported']:
                require(finite(row['rho_zero']), 'NONFINITE_RHO')
        require(sum(r['omitted_from_solver'] and r['molecular_truth'] for r in rows) == 5, 'OMITTED_TRUTH')
        units = units_from_candidates(rows)
        require(units == read(loc/'molecular_units.json'), 'MOLECULAR_AGGREGATION')
        reviewed = modes(units, 125, pack, 5)
        for mode, metric in reviewed.items():
            for n, value in metric.items():
                if n in result['metrics'][mode]:
                    require(value == result['metrics'][mode][n], f'METRIC_MISMATCH: {key}/{mode}/{n}')
        require(result['structural_omission_FN'] == 5 and result['additional_raw_solver_FN'] == reviewed['raw']['additional_raw_solver_FN'], 'FN_PARTITION')
        require(finite(result['reconstruction_residual']), 'NONFINITE_RESIDUAL')
        all_units[key] = units
        checks[key] = dict(status='PASS', normal_completion=True, all_outputs_losses_finite=True,
                           runtime_design_binding=True, result_artifact_hashes_valid=True, independent_accounting=reviewed,
                           stopped_epoch=epoch, stop_reason=solver['stop_reason'], final_model=str(final),
                           reconstruction_residual=result['reconstruction_residual'])
        for omitted in arm['omitted']:
            i = omitted['neighbor_candidate_index']
            row, ref = rows[i], baseline[i]
            require(not row['molecular_truth'] and not row['omitted_from_solver'], 'NEIGHBOR_NOT_SURVIVING_NONTRUTH')
            neighbors.append(dict(case_id=key, omitted_lipid=omitted['lipid_name'], neighbor_lipid=row['lipid_name'],
                                  control_X_hat=ref['X_hat'], omission_X_hat=row['X_hat'], delta_X_hat=row['X_hat']-ref['X_hat'],
                                  control_rho_zero=ref['rho_zero'], omission_rho_zero=row['rho_zero'],
                                  control_reported=ref['raw_solver_reported'], omission_reported=row['raw_solver_reported']))
        for name in sorted(RETIRE_NAMES):
            path = loc/name
            require(path.is_file() and not path.is_symlink() and path.resolve() == path and path.stat().st_nlink == 1, 'UNSAFE_RETIRE_PATH')
            require(str(path) not in protected and int(name.split('_')[2].split('.')[0]) < epoch, 'RETAINED_OR_NONINTERMEDIATE')
            deletions.append(dict(case_id=key, path=str(path), resolved=str(path.resolve()), symlink=False,
                                  sha256=sha(path), bytes=path.stat().st_size,
                                  reason='Completed reduced-library fit; intermediate reload is not required by result verifier, final aggregation, cached pilot, or resume. Diagnostics and terminal models remain.'))
        print(key, 'AUDIT_PASS', flush=True)
    for dataset, units in control_units.items():
        checks['full_library__'+dataset] = dict(independent_accounting=modes(units, 125, pack, 0), provenance='Frozen control-audit hashes reverified; no new fitting')
    summary = {}
    for arm in ('close_neighbor', 'relatively_isolated', 'full_library'):
        units = [u for key, values in (control_units.items() if arm == 'full_library' else all_units.items())
                 if arm == 'full_library' or key.startswith(arm+'__') for u in values]
        summary[arm] = modes(units, 375, pack, 0 if arm == 'full_library' else 15)
    write(JOB/'scientific_review.json', dict(status='COMPLETE', cases=checks, pooled_by_arm=summary,
          limitations=['Three mapping replicates are not biological replication; pooled FDP is descriptive, not population FDR control.',
                       'Missing-library CLEAN challenge; not simultaneous spectral mismatch plus omission.',
                       'Original CAL thresholds unchanged. No new threshold selection, fitting, synthetic reconstruction or rho computation.',
                       'Residuals are stored outcomes, not recomputed from B; historical control reuse evidence is retained.',
                       'No intermediate checkpoint reload analysis is planned; training diagnostics are retained.']))
    write(JOB/'neighbor_allocation_review.json', dict(records=neighbors, interpretation='Paired descriptive changes, not causal source decomposition'))
    write(JOB/'compact_hashes.json', compact)
    plan = dict(status='PREPARED_NOT_DELETED', created_utc=now(), fingerprint=design['fingerprint'], completed=sorted(EXPECTED),
                source_files=protected, deletions=deletions, planned_delete_bytes=sum(r['bytes'] for r in deletions),
                free_bytes_before=shutil.disk_usage(OUT).free, audit_script_sha256=sha(Path(__file__)),
                retained_contract='All result-bound files including latest_model.pth; final named checkpoint; all arrays, candidate/molecular records, diagnostics, CAL files and every full-library parent checkpoint remain.',
                limitation='Retired intermediate weights cannot be reloaded; final usable models and scalar training/checkpoint diagnostics remain.')
    write(JOB/'cleanup_plan.json', plan)
    write(JOB/'process_audit.json', dict(status='PASS', created_utc=now(), completed_cases=6, protected_file_count=len(protected),
          exact_membership=True, all_result_hashes_and_finite_outputs_checked=True, parent_CAL_sources_verified=True,
          full_library_control_hashes_verified=True, latest_and_terminal_models_retained=True, training_or_rho_reexecuted=False))
    archive = JOB/'snapshot.tar.gz'
    with tarfile.open(archive, 'w:gz') as tar:
        for path in sorted(JOB.rglob('*')):
            if path.is_file() and path != archive:
                tar.add(path, arcname=str(path.relative_to(JOB)))
    print(json.dumps(dict(archive=str(archive), archive_sha256=sha(archive), planned_delete_bytes=plan['planned_delete_bytes'], compact_files=len(compact))), flush=True)


def apply(args):
    require(len(args.git_commit or '') == 40 and len(args.plan_sha256 or '') == 64, 'PUSHED_COMMIT_AND_PLAN_REQUIRED')
    require(sha(JOB/'cleanup_plan.json') == args.plan_sha256, 'PLAN_CHANGED')
    require(not (JOB/'deletion_receipt.json').exists(), 'RECEIPT_ALREADY_EXISTS')
    plan = read(JOB/'cleanup_plan.json')
    require(plan['audit_script_sha256'] == sha(Path(__file__)), 'AUDITOR_CHANGED')
    require(read(JOB/'process_audit.json')['status'] == 'PASS', 'AUDIT_NOT_PASS')
    inactive()
    for path, expected in plan['source_files'].items():
        require(sha(Path(path)) == expected, 'PRESERVED_SOURCE_CHANGED: '+path)
    require(len(plan['deletions']) == 24, 'UNEXPECTED_DELETION_MEMBERSHIP')
    for row in plan['deletions']:
        path = Path(row['path'])
        require(row['case_id'] in EXPECTED and path.parent == OUT/'runs'/row['case_id'] and path.name in RETIRE_NAMES, 'OUTSIDE_WHITELIST')
        require(not path.is_symlink() and str(path.resolve()) == row['resolved'] == str(path), 'SYMLINK_OR_PATH_CHANGED')
        require(path.stat().st_nlink == 1 and path.stat().st_size == row['bytes'] and sha(path) == row['sha256'], 'DELETE_ARTIFACT_CHANGED')
        require(str(path) not in plan['source_files'], 'PRESERVED_ARTIFACT')
    receipt = dict(status='IN_PROGRESS', started_utc=now(), git_snapshot_commit=args.git_commit,
                   plan_sha256=args.plan_sha256, free_bytes_before=shutil.disk_usage(OUT).free, removed_files=[])
    write(JOB/'deletion_receipt.json', receipt)
    for row in plan['deletions']:
        Path(row['path']).unlink()
        receipt['removed_files'].append(row)
        write(JOB/'deletion_receipt.json', receipt)
    for path, expected in plan['source_files'].items():
        require(sha(Path(path)) == expected, 'POST_DELETE_PRESERVED_SOURCE_CHANGED: '+path)
    require(all(not Path(row['path']).exists() for row in plan['deletions']), 'DELETION_INCOMPLETE')
    receipt.update(status='COMPLETE', finished_utc=now(), freed_bytes=sum(r['bytes'] for r in receipt['removed_files']),
                   free_bytes_after=shutil.disk_usage(OUT).free, all_preserved_hashes_valid=True,
                   protected_file_count=len(plan['source_files']), result_verifier_dependencies_unchanged=True)
    write(JOB/'deletion_receipt.json', receipt)
    print(json.dumps({k:v for k,v in receipt.items() if k != 'removed_files'}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['prepare', 'apply'])
    parser.add_argument('--git-commit')
    parser.add_argument('--plan-sha256')
    args = parser.parse_args()
    prepare() if args.mode == 'prepare' else apply(args)
