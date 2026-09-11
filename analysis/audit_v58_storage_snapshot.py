"""Remote V58 completed-result snapshot and explicitly scoped checkpoint retirement.

Prepare verifies and exports results before removal. Apply requires the pushed
snapshot commit and exact plan hash. No scientific runner is modified.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tarfile
from datetime import datetime, timezone
from types import SimpleNamespace

ROOT = Path('/root/autodl-tmp/lipiddeconv')
OUT = ROOT/'results/v58_spectral_library_mismatch_fdr_recalibration_compact_recovery'
CACHE = Path('/root/v58_compact_runtime_cache')
JOB = Path('/root/v58_jobs/storage_review_20260911')
NAMES = ['latest_model.pth'] + [f'checkpoint_epoch_{e}.pth' for e in (1000,1500,2000,2500)]


def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda: f.read(8*1024**2), b''): h.update(block)
    return h.hexdigest()


def read(p): return json.loads(p.read_text())


def write(p, d):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(d, indent=2, allow_nan=False)+'\n')


def prepare():
    JOB.mkdir(exist_ok=False)
    sys.path.insert(0, str(ROOT/'analysis'))
    import run_v58_spectral_library_mismatch_fdr_recalibration as v
    v.dependencies()
    args = SimpleNamespace(output_dir=OUT)
    design = read(OUT/'design.json')
    audit = v.checked_audit(args, design)
    # Freeze both complete CAL sets before opening any HOLD outcome.
    for severity in v.SEVERITIES:
        rows, reports, hashes = v.read_completed(args, design, severity, 'CAL')
        v.freeze_local_thresholds(args, design, severity, rows, reports, hashes)
    frozen = v.hold_read_guard(args, design)
    clean = read(ROOT/'results/v57_spectral_spatial_identity_confidence_benchmark/global_frozen_thresholds.json')
    files = list(OUT.glob('*.json')) + list(OUT.glob('*.csv'))
    files += list(OUT.glob('*/local_*threshold*.json')) + list(OUT.glob('*/local_*freeze*.json'))
    completed, plan, sizes, retained = [], [], {}, {}
    rows_by_s = {s: [] for s in v.SEVERITIES}
    reports_by_s = {s: {} for s in v.SEVERITIES}
    for dataset in v.dataset_ids():
        severity, base = dataset.split('__'); loc = OUT/severity/base
        if not (loc/'report.json').exists(): continue
        r = v.verified_report(args, design, dataset, audit['datasets'][dataset])
        solver = read(loc/'solver_run.json')
        assert sha(loc/'learned_arrays.npz') == solver['arrays_sha256']
        for n, expected in solver['training_artifact_hashes'].items(): assert sha(loc/n) == expected
        completed.append(dataset)
        files += [f for f in loc.iterdir() if f.is_file() and f.suffix in ('.json','.csv')]
        for f in loc.iterdir():
            if f.is_file(): sizes[f.suffix] = sizes.get(f.suffix, 0) + f.stat().st_size
        if base.startswith('HOLD'):
            reports_by_s[severity][dataset] = r
            rows_by_s[severity] += v.normalize_records(v.v57.rows(loc/'molecular_false_negative_records.csv'))
        # Keep every sentinel-related case and the final epoch3000 model/arrays.
        if base in ('CAL_R1_K050', 'CAL_R1_K175') or not (loc/'checkpoint_epoch_3000.pth').is_file(): continue
        retained[dataset] = {'final_checkpoint_sha256': sha(loc/'checkpoint_epoch_3000.pth'),
                             'arrays_sha256': solver['arrays_sha256']}
        for name in NAMES:
            f = loc/name
            if not f.is_file(): continue
            target = f.resolve()
            assert target.is_relative_to(OUT.resolve()) or target.is_relative_to(CACHE.resolve())
            plan.append(dict(dataset=dataset, path=str(f), resolved=str(target),
                             bytes=f.stat().st_size, sha256=sha(f), symlink=f.is_symlink()))
    summary = {}
    for severity in v.SEVERITIES:
        rows, reports = rows_by_s[severity], reports_by_s[severity]
        summary[severity] = {'HOLD_completed':len(reports), 'HOLD_planned':15,
                            'complete_severity':len(reports)==15,
                            'raw':v.v56.metrics([r for r in rows if r['raw_solver_reported']],reports)}
        for name, pack in [('CLEAN_FIXED', {'rho_zero':clean['global_thresholds']['rho_zero']}),
                           ('LOCAL_CAL_RECALIBRATION', frozen[severity]['global_thresholds'])]:
            overall, by_k, by_r = v.protocol_metrics(severity,name,rows,reports,pack)
            summary[severity][name] = {'overall':overall,'by_K':by_k,'by_replicate':by_r}
    plan_data = dict(created_utc=datetime.now(timezone.utc).isoformat(), fingerprint=design['design_fingerprint'],
                     completed=completed, logical_completed_bytes_by_suffix=sizes, deletions=plan,
                     retained=retained, source_files={str(f.relative_to(OUT)):sha(f) for f in sorted(set(files))},
                     reason='Retire intermediate/redundant ordinary completed-case checkpoints after Git snapshot. Keep final epoch3000, learned arrays, all reports and all sentinel evidence.',
                     limitation='Retired intermediate checkpoints cannot be reloaded; their recorded hashes and diagnostics remain. Final learned model and outputs remain remote.')
    write(JOB/'cleanup_plan.json',plan_data)
    write(JOB/'partial_summary.json',dict(status='INTERIM_COMPLETED_CASE_SNAPSHOT',completed=len(completed),planned=60,results=summary))
    with tarfile.open(JOB/'snapshot.tar.gz','w:gz') as tar:
        for f in sorted(set(files)): tar.add(f,arcname=str(f.relative_to(OUT)))
        for name in ('cleanup_plan.json','partial_summary.json'):tar.add(JOB/name,arcname=name)
    print(json.dumps(dict(completed=len(completed),planned_delete_bytes=sum(r['bytes'] for r in plan),
                          files=len(set(files)),archive_sha256=sha(JOB/'snapshot.tar.gz'))),flush=True)


def apply(args):
    assert len(args.git_commit)==40 and len(args.plan_sha256)==64
    assert sha(JOB/'cleanup_plan.json')==args.plan_sha256
    plan=read(JOB/'cleanup_plan.json')
    assert not (JOB/'deletion_receipt.json').exists()
    state=read(Path('/root/v58_jobs/storage_recovery_20260911/status.json'))
    # Verify the entire deletion set and retained evidence before removing anything.
    for row in plan['deletions']:
        assert row['dataset'] != state.get('active_dataset')
        path=Path(row['path']);target=path.resolve()
        assert path.is_relative_to(OUT) and path.name in NAMES
        assert str(target)==row['resolved'] and (target.is_relative_to(OUT.resolve()) or target.is_relative_to(CACHE.resolve()))
        assert sha(path)==row['sha256']
    for dataset, keep in plan['retained'].items():
        severity,base=dataset.split('__');loc=OUT/severity/base
        assert sha(loc/'checkpoint_epoch_3000.pth')==keep['final_checkpoint_sha256']
        assert sha(loc/'learned_arrays.npz')==keep['arrays_sha256']
    for name,h in plan['source_files'].items():assert sha(OUT/name)==h
    receipt=dict(git_snapshot_commit=args.git_commit,plan_sha256=args.plan_sha256,deleted=[],started_utc=datetime.now(timezone.utc).isoformat())
    for row in plan['deletions']:
        path=Path(row['path'])
        if path.is_symlink():Path(row['resolved']).unlink()
        path.unlink()
        receipt['deleted'].append(row['path'])
        write(JOB/'deletion_receipt.json',receipt)
    for name,h in plan['source_files'].items():assert sha(OUT/name)==h
    receipt.update(status='COMPLETE',freed_bytes=sum(r['bytes'] for r in plan['deletions']),
                   data_disk_free=shutil.disk_usage(OUT).free,root_disk_free=shutil.disk_usage('/root').free,
                   finished_utc=datetime.now(timezone.utc).isoformat(),all_preserved_source_hashes_valid=True)
    write(JOB/'deletion_receipt.json',receipt)
    print(json.dumps({k:v for k,v in receipt.items() if k!='deleted'}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','apply'])
    p.add_argument('--git-commit');p.add_argument('--plan-sha256');args=p.parse_args()
    prepare() if args.mode=='prepare' else apply(args)
