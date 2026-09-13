"""Apply the published fixed joint score to the existing real CE29 ISTA export."""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'.venv/real_ce29_packages'))
sys.path.insert(0, str(ROOT/'analysis'))
sys.path.insert(0, str(ROOT/'src'))
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
from physical_block_monotone_score import read, write, sha, require

PUBLISHED = '704e05fa35e39b15735fd5c600eaed9f8510db0f'
MODEL = ROOT/'results/physical_block_score_correction_v2/model'
COMPONENTS = ROOT/'results/small_mismatch_physical_components/components.npz'
MANIFEST = ROOT/'results/physical_block_prediction_v1/block_manifest.json'
CODE = ('analysis/run_real_ce29_joint_screening.py', 'analysis/physical_block_monotone_score.py',
        'analysis/physical_block_prediction.py', 'analysis/run_physical_block_prediction.py',
        'analysis/review_physical_block_prediction.py', 'analysis/run_nnls_solver_baseline.py',
        'analysis/run_small_mismatch_nnls_first_case.py', 'src/rho_zero.py',
        'docs/REAL_CE29_JOINT_SCREENING_V2.md')


def runtime():
    import numpy, scipy, threadpoolctl
    return dict(python=sys.version, numpy=numpy.__version__, scipy=scipy.__version__,
                threadpoolctl=threadpoolctl.__version__, platform=sys.platform)


def npz(path, **arrays):
    import numpy as np
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as f:
        np.savez_compressed(f, **arrays)


def prepare(out):
    import numpy as np
    from physical_block_monotone_score import validate_model_directory
    from physical_block_prediction import build_blocks
    from review_physical_block_prediction import review_block_manifest
    require(not (out/'input.json').exists(), 'INPUT_ALREADY_FROZEN')
    require(runtime()['numpy']=='2.1.3' and runtime()['scipy']=='1.15.3', 'RUNTIME_CHANGED')
    validated = validate_model_directory(MODEL)
    require(sha(MODEL/'model.json')=='a1fd3193d994cc22df1676ab6e5d76c5880f9acb6442065127efa3fe7280a8bb', 'PUBLISHED_MODEL_CHANGED')
    require(sha(MODEL/'thresholds.json')=='df44a646c0d6146259ece3c08dc3af5d61af1e4896825261f35d8b83745a79f4', 'PUBLISHED_THRESHOLDS_CHANGED')
    assets = ROOT.parent/'decon-lipid/adapter_pipeline/outputs/v38_ce29_fragmentionfixed_profile_overlap_empiricalfwhm_globalq99_ista_ready'
    export = ROOT.parent/'decon-lipid/results_758_v38_ce29_empiricalfwhm_globalq99_fixed_library_joint_earlystop'
    lock = read(ROOT/'results/v48_production_ista_lock/v48_production_ista_lock.json')
    source_paths = {'X':export/'X_abundance.npy', 'A':export/'A_calibrated.npy', 'B':assets/'B_cube.npy',
                    'raw_library':assets/'A_library.npy', 'mask':assets/'foreground_pixel_mask.npy',
                    'metadata':assets/'candidate_metadata_final.npy', 'axis':assets/'shared_mz_final.npy',
                    'export_report':export/'export_report.json', 'components':COMPONENTS, 'blocks':MANIFEST,
                    'model':MODEL/'model.json', 'thresholds':MODEL/'thresholds.json', 'model_seal':MODEL/'model_seal.json'}
    hashes = {k:sha(p) for k,p in source_paths.items()}
    for k, key in (('X','reference_X'), ('B','B_cube'), ('raw_library','A_library'), ('metadata','candidate_metadata'), ('axis','channel_axis')):
        require(hashes[k] == lock['hashes_sha256'][key], 'PRODUCTION_LOCK_CHANGED:'+k)
    A, X, stored, mask = (np.load(source_paths[k], allow_pickle=False) for k in ('A','X','B','mask'))
    require(stored.shape==(1,200,90,1084) and mask.dtype==bool, 'PRODUCTION_LAYOUT_CHANGED')
    B = stored[0].transpose(2,0,1)
    metadata = np.load(source_paths['metadata'], allow_pickle=True).item()
    names = [str(n) for n in metadata['lipid_name']]
    require(A.shape==(1084,391) and X.shape==(391,200,90) and int(mask.sum())==15837, 'PRODUCTION_SHAPE_CHANGED')
    require(all(np.isfinite(q).all() for q in (A,X,B)) and (X>=0).all() and (B[:,~mask]==0).all(), 'INVALID_PRODUCTION_ARRAY')
    with np.load(COMPONENTS, allow_pickle=False) as z:
        nominal, components, owner = z['A'], z['components'], z['owner']
    require(np.array_equal(A>0,nominal>0) and np.max(np.abs(A-nominal)) <= 4*np.finfo(np.float32).eps, 'LIBRARY_SUPPORT_OR_VALUES_DIFFER')
    manifest = read(MANIFEST)
    require(manifest == build_blocks(components, owner, names), 'BLOCK_OR_ALIAS_ORDER_CHANGED')
    block_review = review_block_manifest(A, components, owner, names, manifest['blocks'])
    means = X[:,mask].mean(axis=1,dtype=float)
    b = B[:,mask].mean(axis=1,dtype=float)
    candidates = [dict(candidate_index=j, candidate_id=str(metadata['candidate_id'][j]), lipid_name=names[j],
                       lipid_class=str(metadata['lipid_class'][j]), X_hat=float(means[j]),
                       raw_solver_reported=bool(means[j]>.001), molecular_truth=None) for j in range(391)]
    rows = []
    for name in manifest['molecular_names']:
        indices=[j for j,n in enumerate(names) if n==name]; active=[j for j in indices if means[j]>.001]
        rows.append(dict(lipid_name=name, lipid_class=candidates[indices[0]]['lipid_class'], candidate_indices=indices,
                         reported_candidate_indices=active, X_hat=sum(float(means[j]) for j in active),
                         raw_solver_reported=bool(active), rho_zero=None, molecular_truth=None))
    npz(out/'prepared.npz', A=A.astype(float), b=b, candidate_mean=means)
    write(out/'candidate_records_input.json', candidates); write(out/'molecular_records_input.json', rows)
    for n in ('model.json','thresholds.json','model_seal.json'):
        target=out/n
        require(not target.exists(), 'EXISTING_MODEL_COPY');shutil.copyfile(MODEL/n,target)
    write(out/'block_manifest.json', manifest);write(out/'input_block_review.json', block_review)
    binding = dict(version='REAL_CE29_JOINT_SCORE_V2_APPLICATION', published_commit=PUBLISHED,
                   source_files={k:dict(path=str(source_paths[k]),sha256=v) for k,v in hashes.items()},
                   files={n:sha(out/n) for n in ('prepared.npz','candidate_records_input.json','molecular_records_input.json',
                          'model.json','thresholds.json','model_seal.json','block_manifest.json','input_block_review.json')},
                   code={n:sha(ROOT/n) for n in CODE}, runtime=runtime(), workers=4,
                   original_reported_candidates=sum(r['raw_solver_reported'] for r in candidates),
                   original_reported_molecular_identities=sum(r['raw_solver_reported'] for r in rows),
                   max_export_vs_benchmark_A_difference=float(np.max(np.abs(A-nominal))),
                   A_choice='original exported A_calibrated, same support; no renormalization',
                   true_identity_labels_available=False, no_model_fit=True, no_ISTA_rerun=True,
                   no_pixelwise_NNLS=True, no_GPU=True, actual_real_FDR=None, no_data_upload=True)
    write(out/'input.json',binding);write(out/'input_seal.json',dict(sha256=sha(out/'input.json'),status='FROZEN_BEFORE_FEATURES'))
    print('PREPARED_REAL',binding['original_reported_molecular_identities'],flush=True)


def validate(out):
    d=read(out/'input.json');require(sha(out/'input.json')==read(out/'input_seal.json')['sha256'],'INPUT_CHANGED')
    require(runtime()==d['runtime'],'RUNTIME_CHANGED')
    for n,h in d['files'].items():require(sha(out/n)==h,'PREPARED_FILE_CHANGED:'+n)
    for n,h in d['code'].items():require(sha(ROOT/n)==h,'CODE_CHANGED:'+n)
    for k,item in d['source_files'].items():require(sha(item['path'])==item['sha256'],'SOURCE_CHANGED:'+k)
    return d


def task(out_string, index):
    import numpy as np
    from threadpoolctl import threadpool_limits
    from physical_block_prediction import solve_block
    from review_physical_block_prediction import review_block
    out=Path(out_string); manifest=read(out/'block_manifest.json')
    with np.load(out/'prepared.npz',allow_pickle=False) as z:A,b=z['A'],z['b']
    start=time.monotonic()
    with threadpool_limits(limits=1):
        if index == -1:
            import rho_zero
            values={}
            for row in read(out/'candidate_records_input.json'):
                if row['raw_solver_reported']:
                    j=row['candidate_index'];values[str(j)]=rho_zero.rho_zero_from_weighted_case(A,b,j)
                    require(all(np.isfinite(v) for v in values[str(j)].values()),'NONFINITE_RHO')
            write(out/'rho.json',values)
            return dict(task='rho',reported_candidate_count=len(values),elapsed_seconds=time.monotonic()-start)
        held=manifest['blocks'][index]['channel_indices']
        result=solve_block(A,b,manifest['candidate_names'],held)
        stem=out/'blocks'/f'block_{index:03d}'
        npz(stem.with_suffix('.npz'),**result['arrays'])
        write(stem.with_suffix('.json'),result['diagnostics'])
        checked=review_block(A,b,manifest['candidate_names'],held,result['arrays'])
        write(stem.with_name(stem.name+'_review.json'),checked)
    return dict(task=index,elapsed_seconds=time.monotonic()-start,review='PASS')


def finalize(out, elapsed):
    import numpy as np
    import physical_block_monotone_score as score
    from run_physical_block_prediction import features_from_blocks
    d=validate(out);manifest=read(out/'block_manifest.json')
    with np.load(out/'prepared.npz',allow_pickle=False) as z:A,b=z['A'],z['b']
    arrays=[]
    for k in range(34):
        stem=out/'blocks'/f'block_{k:03d}'
        require(read(stem.with_name(stem.name+'_review.json'))['status']=='PASS','BLOCK_REVIEW_FAILED')
        with np.load(stem.with_suffix('.npz'),allow_pickle=False) as z:arrays.append({n:z[n] for n in z.files})
    features=features_from_blocks(A,b,manifest['candidate_names'],manifest,arrays)
    rows=read(out/'molecular_records_input.json');rho=read(out/'rho.json')
    for row in rows:
        ids=row['reported_candidate_indices']
        row['rho_zero']=max(rho[str(j)]['rho_zero'] for j in ids) if ids else None
    model=read(out/'model.json');cuts=read(out/'thresholds.json')
    values=score.score_rows(rows,features,model)
    review=score.review_predictions(rows,features,model,values)
    require(all(row['molecular_truth'] is None for row in rows),'REAL_TRUTH_MUST_REMAIN_UNKNOWN')
    selected={key:score.select_names(rows,values,cuts[key+'_threshold']) for key in ('pool','fdp5','fdp1')}
    by_name={f['molecular_name']:f for f in features}
    for row,value in zip(rows,values):
        f=by_name[row['lipid_name']];row.update(joint_score=value,predictive_gain=f['predictive_gain'],positive_block_fraction=f['positive_block_fraction'])
        row.update({f'retained_by_{k}':row['lipid_name'] in names for k,names in selected.items()})
    # Independent set accounting and scalar-score replay were completed above.
    for key,names in selected.items():
        independent={row['lipid_name'] for row in rows if row['raw_solver_reported'] and row['joint_score']>=cuts[key+'_threshold']}
        require(independent==set(names),'SELECTION_MEMBERSHIP_DISAGREES')
    write(out/'physical_features.json',features);write(out/'molecular_records.json',rows)
    candidates=read(out/'candidate_records_input.json')
    for row in candidates:row['rho_zero']=rho[str(row['candidate_index'])]['rho_zero'] if row['raw_solver_reported'] else None
    write(out/'candidate_records.json',candidates);write(out/'score_review.json',review)
    csv_fields=['lipid_name','lipid_class','raw_solver_reported','X_hat','rho_zero','predictive_gain','positive_block_fraction','joint_score','retained_by_pool','retained_by_fdp5','retained_by_fdp1']
    with (out/'molecular_screening.csv').open('x',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=csv_fields,extrasaction='ignore');writer.writeheader();writer.writerows(rows)
    total=d['original_reported_molecular_identities']
    report=dict(status='COMPLETE_REVIEWED',published_commit=PUBLISHED,elapsed_seconds=elapsed,
                original_reported_molecular_identities=total,original_reported_candidates=d['original_reported_candidates'],
                thresholds={k:dict(threshold=cuts[k+'_threshold'],retained=len(v),removed=total-len(v),coverage=len(v)/total,names=v) for k,v in selected.items()},
                actual_FDR=None,TP=None,FP=None,FN=None,recall=None,truth_available=False,
                interpretation='Exploratory transfer of synthetic NNLS-trained fixed score to real production ISTA. DEV1/5 labels are threshold provenance, not real FDR guarantees.',
                input_sha256=sha(out/'input.json'),model_sha256=sha(out/'model.json'),thresholds_sha256=sha(out/'thresholds.json'),
                all_block_cached_reviews_pass=True,score_replay_pass=True,no_ISTA_or_training_rerun=True,no_deletion=True)
    write(out/'report.json',report)
    write(out/'artifact_hashes.json',{p.relative_to(out).as_posix():sha(p) for p in sorted(out.iterdir()) if p.is_file() and p.name not in ('artifact_hashes.json','.gitattributes')}|
          {p.relative_to(out).as_posix():sha(p) for p in sorted((out/'blocks').iterdir()) if p.is_file()})
    print(json.dumps({k:v for k,v in report.items() if k!='thresholds'}),flush=True)
    print('RETAINED_COUNTS',{k:len(v) for k,v in selected.items()},flush=True)


def run(out):
    d=validate(out);require(not (out/'run_started.json').exists(),'EXISTING_RUN_PRESERVED_NO_AUTOMATIC_RETRY')
    write(out/'run_started.json',dict(pid=os.getpid(),started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),input_sha256=sha(out/'input.json')))
    start=time.monotonic();pool=ProcessPoolExecutor(max_workers=d['workers']);futures=[]
    try:
        futures=[pool.submit(task,str(out),i) for i in [-1]+list(range(34))]
        for future in as_completed(futures,timeout=14400):
            item=future.result();write(out/'task_receipts'/f"{item['task']}.json",item)
            print('TASK_COMPLETE',item['task'],round(item['elapsed_seconds'],2),flush=True)
        pool.shutdown(wait=True);finalize(out,time.monotonic()-start)
    except BaseException as exc:
        for f in futures:f.cancel()
        for p in (pool._processes or {}).values():p.terminate()
        pool.shutdown(wait=True,cancel_futures=True)
        write(out/'failure.json',dict(error=repr(exc),outputs_preserved=True));raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=('prepare','run'))
    p.add_argument('--output',type=Path,default=ROOT/'results/real_ce29_joint_screening_v2');args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    globals()[args.action](args.output.resolve())
