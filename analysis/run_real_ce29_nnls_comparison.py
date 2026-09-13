"""Full-library real CE29 NNLS using existing solver and fixed cached score evidence."""
from __future__ import annotations
import argparse
import csv
import math
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'.venv/real_ce29_packages'), str(ROOT/'analysis'), str(ROOT/'src')]
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import numpy as np
from threadpoolctl import threadpool_limits
import physical_block_monotone_score as score
from physical_block_monotone_score import read, write, sha, require
from run_real_ce29_joint_screening import validate as validate_parent, runtime

PARENT = ROOT/'results/real_ce29_joint_screening_v2'
CODE = ('analysis/run_real_ce29_nnls_comparison.py', 'analysis/refit_screened_nnls.py',
        'analysis/run_nnls_solver_baseline.py', 'analysis/run_small_mismatch_nnls_first_case.py',
        'analysis/physical_block_monotone_score.py', 'analysis/run_real_ce29_joint_screening.py',
        'src/rho_zero.py', 'docs/REAL_CE29_NNLS_COMPARISON.md')
PARENT_FILES = ('input.json', 'input_seal.json', 'prepared.npz', 'physical_features.json',
                'model.json', 'thresholds.json', 'model_seal.json', 'rho.json',
                'candidate_records_input.json', 'molecular_records.json', 'report.json',
                'independent_final_review.json')


def arrays():
    d = read(PARENT/'input.json')
    with np.load(PARENT/'prepared.npz', allow_pickle=False) as z:
        A, b = z['A'], z['b']
    B = np.load(d['source_files']['B']['path'], allow_pickle=False)[0].transpose(2, 0, 1)
    mask = np.load(d['source_files']['mask']['path'], allow_pickle=False)
    require(np.array_equal(b, B[:, mask].mean(axis=1, dtype=float)), 'MEAN_OBSERVATION_CHANGED')
    return A, B, mask, b


def prepare(out):
    require(not (out/'input.json').exists(), 'INPUT_ALREADY_FROZEN')
    validate_parent(PARENT)
    require(read(PARENT/'independent_final_review.json')['status'] == 'PASS', 'PARENT_NOT_REVIEWED')
    A, B, mask, _ = arrays()
    require(A.shape == (1084, 391) and B.shape == (1084, 200, 90) and int(mask.sum()) == 15837,
            'INPUT_LAYOUT_CHANGED')
    d = dict(version='REAL_CE29_FULL_NNLS_FIXED_JOINT_SCORE', parent=str(PARENT),
             parent_files={n:sha(PARENT/n) for n in PARENT_FILES},
             code={n:sha(ROOT/n) for n in CODE}, runtime=runtime(),
             library_indices=list(range(391)), workers=4, block_size=250,
             reporting_gate=.001, physical_features_reused=True, model_refit=False,
             source_files=read(PARENT/'input.json')['source_files'], true_labels_available=False)
    write(out/'input.json', d)
    write(out/'input_seal.json', dict(status='FROZEN_BEFORE_NNLS', sha256=sha(out/'input.json')))
    print('PREPARED_ALL_391_COLUMNS_15837_PIXELS', flush=True)


def validate(out):
    d = read(out/'input.json')
    require(sha(out/'input.json') == read(out/'input_seal.json')['sha256'], 'INPUT_CHANGED')
    require(d['runtime'] == runtime(), 'RUNTIME_CHANGED')
    for n, h in d['parent_files'].items(): require(sha(PARENT/n) == h, 'PARENT_CHANGED:'+n)
    for n, h in d['code'].items(): require(sha(ROOT/n) == h, 'CODE_CHANGED:'+n)
    for item in d['source_files'].values(): require(sha(item['path']) == item['sha256'], 'SOURCE_CHANGED')
    return d


def records(means, rho):
    candidates = []
    for old in read(PARENT/'candidate_records_input.json'):
        j = old['candidate_index']; row = dict(old)
        row.update(X_hat=float(means[j]), raw_solver_reported=bool(means[j]>.001),
                   rho_zero=rho[str(j)]['rho_zero'] if str(j) in rho else None)
        candidates.append(row)
    rows = []
    features = read(PARENT/'physical_features.json'); lookup = {r['molecular_name']:r for r in features}
    for old in read(PARENT/'molecular_records.json'):
        ids = old['candidate_indices']; active = [j for j in ids if means[j]>.001]
        f = lookup[old['lipid_name']]
        rows.append(dict(lipid_name=old['lipid_name'], lipid_class=old['lipid_class'],
                         candidate_indices=ids, reported_candidate_indices=active,
                         X_hat=sum(float(means[j]) for j in active), raw_solver_reported=bool(active),
                         molecular_truth=None, rho_zero=max(rho[str(j)]['rho_zero'] for j in active) if active else None,
                         predictive_gain=f['predictive_gain'], positive_block_fraction=f['positive_block_fraction']))
    model = read(PARENT/'model.json'); cuts = read(PARENT/'thresholds.json')
    values = score.score_rows(rows, features, model)
    replay = score.review_predictions(rows, features, model, values)
    for row, value in zip(rows, values):
        row['joint_score'] = value
        for k in ('pool', 'fdp5', 'fdp1'):
            row['retained_by_'+k] = bool(row['raw_solver_reported'] and value >= cuts[k+'_threshold'])
    return candidates, rows, replay


def compare(rows):
    old = read(PARENT/'molecular_records.json'); cuts = read(PARENT/'thresholds.json')
    comparison = {}
    for k in ('raw', 'pool', 'fdp5', 'fdp1'):
        field = 'raw_solver_reported' if k == 'raw' else 'retained_by_'+k
        nnls = {r['lipid_name'] for r in rows if r[field]}
        ista = {r['lipid_name'] for r in old if r[field]}
        comparison[k] = dict(NNLS=len(nnls), ISTA=len(ista), common=len(nnls & ista),
                             NNLS_names=sorted(nnls), ISTA_names=sorted(ista),
                             NNLS_only=sorted(nnls-ista), ISTA_only=sorted(ista-nnls),
                             threshold=None if k == 'raw' else cuts[k+'_threshold'])
    return comparison


def residual(B, Bhat, mask):
    observed, fitted = B[:, mask], Bhat[:, mask]
    sse, signal = 0., 0.
    for start in range(0, int(mask.sum()), 250):
        y = observed[:, start:start+250].astype(float)
        delta = y - fitted[:, start:start+250].astype(float)
        sse += float(np.sum(delta*delta)); signal += float(np.sum(y*y))
    return dict(squared_error=sse, squared_observation=signal, relative_l2=math.sqrt(sse/signal))


def run(out):
    from refit_screened_nnls import fit_screened
    from rho_zero import rho_zero_from_weighted_case
    d = validate(out)
    require(not (out/'run_started.json').exists(), 'EXISTING_RUN_PRESERVED_NO_AUTOMATIC_RETRY')
    write(out/'run_started.json', dict(pid=os.getpid(), input_sha256=sha(out/'input.json'),
                                      started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())))
    start = time.monotonic()
    try:
        A, B, mask, b = arrays()
        print('FULL_LIBRARY_NNLS_START', flush=True)
        fitted = fit_screened(A, B, mask, d['library_indices'], out/'nnls', workers=4, block_size=250,
                              source_binding=dict(input_sha256=sha(out/'input.json')))
        means = fitted['means']; rho = read(PARENT/'rho.json'); added = []
        for j in np.flatnonzero(means>.001):
            if str(j) not in rho:
                rho[str(j)] = rho_zero_from_weighted_case(A, b, int(j)); added.append(int(j))
                require(all(math.isfinite(v) for v in rho[str(j)].values()), 'NONFINITE_RHO')
        write(out/'rho.json', rho)
        candidates, rows, replay = records(means, rho)
        write(out/'candidate_records.json', candidates); write(out/'molecular_records.json', rows)
        write(out/'score_review.json', replay)
        with np.load(fitted['arrays_path'], allow_pickle=False) as z: fit_residual = residual(B, z['B_hat'], mask)
        comparison = compare(rows)
        report = dict(status='COMPLETE_AWAITING_CACHE_REVIEW', comparison=comparison,
                      original_reported_candidates=sum(r['raw_solver_reported'] for r in candidates),
                      actual_FDR=None, TP=None, FP=None, FN=None, recall=None, truth_available=False,
                      elapsed_seconds=time.monotonic()-start, reconstruction=fit_residual,
                      input_sha256=sha(out/'input.json'), reused_physical_features_sha256=sha(PARENT/'physical_features.json'),
                      model_sha256=sha(PARENT/'model.json'), thresholds_sha256=sha(PARENT/'thresholds.json'),
                      cached_rho_count=len(rho)-len(added), newly_computed_rho_candidates=added,
                      no_training=True, all_391_columns=True, no_deletion=True)
        validate(out)
        write(out/'report.json', report)
        fields = ['lipid_name', 'lipid_class', 'raw_solver_reported', 'X_hat', 'rho_zero', 'predictive_gain',
                  'positive_block_fraction', 'joint_score', 'retained_by_pool', 'retained_by_fdp5', 'retained_by_fdp1']
        with (out/'molecular_screening.csv').open('x', encoding='utf-8-sig', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore'); writer.writeheader(); writer.writerows(rows)
        review(out)
        print('NNLS_COMPLETE', {k:v['NNLS'] for k,v in comparison.items()}, flush=True)
    except BaseException as exc:
        write(out/'failure.json', dict(error=repr(exc), partial_outputs_preserved=True))
        raise


def review(out):
    """Recompute accounting from stored arrays; never calls an optimizer."""
    d = validate(out); A, B, mask, _ = arrays(); directory = out/'nnls'
    binding = read(directory/'binding.json'); receipt = read(directory/'completion.json')
    require(receipt['status']=='COMPLETE' and receipt['retained_indices']==list(range(391)), 'INCOMPLETE_FULL_LIBRARY')
    require(receipt['binding_sha256']==sha(directory/'binding.json') and receipt['fingerprint']==binding['fingerprint'], 'NNLS_BINDING_CHANGED')
    require(binding['scientific']['source_binding']==dict(input_sha256=sha(out/'input.json')), 'NNLS_SOURCE_CHANGED')
    require(receipt['arrays_sha256']==sha(directory/'learned_arrays.npz'), 'ARRAY_HASH_CHANGED')
    with np.load(directory/'learned_arrays.npz', allow_pickle=False) as z: X, Bhat = z['X_hat'], z['B_hat']
    require(X.shape==(391,200,90) and Bhat.shape==B.shape and X.dtype==Bhat.dtype==np.float32,
            'ARRAY_LAYOUT_CHANGED')
    require(np.isfinite(X).all() and np.isfinite(Bhat).all() and (X>=0).all()
            and (X[:,~mask]==0).all() and (Bhat[:,~mask]==0).all(), 'INVALID_ARRAY_VALUES')
    flat, prediction = X[:,mask], Bhat[:,mask]
    count = int(mask.sum()); spans = [(s,min(s+250,count)) for s in range(0,count,250)]
    expected = {f'block_{s:06d}_{e:06d}.{ext}' for s,e in spans for ext in ('npz','json')}
    require(set(receipt['block_hashes'])==expected and {p.name for p in (directory/'nnls_blocks').iterdir()}==expected,
            'BLOCK_MEMBERSHIP_CHANGED')
    aggregate = dict(max_dual_violation=0., max_complementarity=0., max_bound_ratio=0.)
    max_error, scale = 0., 0.
    for s,e in spans:
        stem = f'block_{s:06d}_{e:06d}'; blockdir = directory/'nnls_blocks'
        for ext in ('npz','json'): require(sha(blockdir/f'{stem}.{ext}')==receipt['block_hashes'][f'{stem}.{ext}'], 'BLOCK_HASH_CHANGED')
        record = read(blockdir/f'{stem}.json')
        require(record['fingerprint']==binding['fingerprint'] and record['start']==s and record['stop']==e
                and record['retained_indices']==list(range(391)) and record['all_pixels_checked_before_float32'] is True
                and record['array_sha256']==receipt['block_hashes'][stem+'.npz'], 'BLOCK_BINDING_CHANGED')
        require(record['KKT']['max_bound_ratio']<=1 and all(math.isfinite(v) and v>=0 for v in record['KKT'].values()), 'INVALID_KKT')
        for k,v in record['KKT'].items(): aggregate[k]=max(aggregate[k],v)
        with np.load(blockdir/f'{stem}.npz', allow_pickle=False) as z: block=z['X_hat']
        require(np.array_equal(block,flat[:,s:e]), 'BLOCK_FINAL_MISMATCH')
        estimate=(A@block.astype(float)).astype(np.float32)
        max_error=max(max_error,float(np.abs(estimate-prediction[:,s:e]).max()))
        scale=max(scale,float(np.abs(prediction[:,s:e]).max()))
    require(aggregate==receipt['KKT'] and max_error<=5e-7*max(scale,1e-12), 'RECONSTRUCTION_OR_KKT_CHANGED')
    rows=read(out/'molecular_records.json'); candidates=read(out/'candidate_records.json'); rho=read(out/'rho.json')
    for k,v in read(PARENT/'rho.json').items(): require(rho[k]==v, 'CACHED_RHO_CHANGED')
    means=flat.mean(axis=1,dtype=float); names=[r['lipid_name'] for r in candidates]
    original_candidates=read(PARENT/'candidate_records_input.json')
    require(len(candidates)==391 and len(rows)==len({r['lipid_name'] for r in rows})==377, 'IDENTITY_MEMBERSHIP_CHANGED')
    for j,r in enumerate(candidates):
        require(r['candidate_index']==j and r['X_hat']==float(means[j]) and r['raw_solver_reported']==bool(means[j]>.001), 'CANDIDATE_ACCOUNTING_CHANGED')
        require(all(r[k]==original_candidates[j][k] for k in ('candidate_id','lipid_name','lipid_class','molecular_truth')), 'CANDIDATE_METADATA_CHANGED')
    features={f['molecular_name']:f for f in read(PARENT/'physical_features.json')}
    for row in rows:
        ids=[j for j,n in enumerate(names) if n==row['lipid_name']]; active=[j for j in ids if means[j]>.001]
        require(row['candidate_indices']==ids and row['reported_candidate_indices']==active
                and row['raw_solver_reported']==bool(active) and row['X_hat']==sum(float(means[j]) for j in active)
                and row['molecular_truth'] is None, 'MOLECULAR_ACCOUNTING_CHANGED')
        require(row['rho_zero']==(max(rho[str(j)]['rho_zero'] for j in active) if active else None), 'RHO_AGGREGATION_CHANGED')
        for k in ('predictive_gain','positive_block_fraction'): require(row[k]==features[row['lipid_name']][k], 'PHYSICAL_FEATURE_CHANGED')
    replay=score.review_predictions(rows,list(features.values()),read(PARENT/'model.json'),[r['joint_score'] for r in rows])
    cuts=read(PARENT/'thresholds.json')
    for row in rows:
        for k in ('pool','fdp5','fdp1'):
            require(row['retained_by_'+k]==bool(row['raw_solver_reported'] and row['joint_score']>=cuts[k+'_threshold']), 'THRESHOLD_FLAG_CHANGED')
    report=read(out/'report.json')
    require(report['comparison']==compare(rows) and report['reconstruction']==residual(B,Bhat,mask), 'RESULT_CHANGED')
    require(all(report[k] is None for k in ('actual_FDR','TP','FP','FN','recall')), 'REAL_TRUTH_UNKNOWN')
    require(read(directory/'status.json')['status']=='COMPLETE', 'INCOMPLETE_STATUS')
    with (out/'molecular_screening.csv').open(encoding='utf-8-sig',newline='') as f:
        table=list(csv.DictReader(f))
    require(len(table)==377 and [r['lipid_name'] for r in table]==[r['lipid_name'] for r in rows], 'CSV_MEMBERSHIP_CHANGED')
    for item,row in zip(table,rows):
        for k in ('raw_solver_reported','retained_by_pool','retained_by_fdp5','retained_by_fdp1'):
            require(item[k]==str(row[k]), 'CSV_FLAG_CHANGED')
    write(out/'independent_final_review.json', dict(status='PASS', report_sha256=sha(out/'report.json'),
          reviewed_blocks=len(spans), foreground_pixels=count, full_library_columns=391,
          finite_arrays=True, original_KKT_receipts_checked=True, block_and_reconstruction_replay=True,
          molecular_accounting_and_fixed_scores_checked=True, score_replay=replay, no_optimizer_called=True))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('action',choices=('prepare','run','review'))
    p.add_argument('--output',type=Path,default=ROOT/'results/real_ce29_nnls_joint_screening_v2')
    args=p.parse_args(); args.output.mkdir(parents=True,exist_ok=True)
    with threadpool_limits(limits=1): globals()[args.action](args.output.resolve())
