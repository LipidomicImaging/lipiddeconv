"""One finite cached-feature comparison, DEV-only selection, and fixed refits."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import scipy
from scipy.optimize import minimize
from scipy.special import expit
import physical_block_monotone_score as score

CASES = ('DEV_TRAIN', 'CHECK', 'NEW_COMPOSITION')
ARMS = {'TRIM': ['trimmed_gain'], 'DIRECT': ['direct_gain', 'direct_per_signal'],
        'JOINT': ['trimmed_gain', 'direct_gain', 'direct_per_signal', 'stability']}
OUT = ROOT / 'results/cached_identity_evidence_v1'
V1 = 'results/physical_block_prediction_v1'
V2 = 'results/physical_block_score_correction_v2'
CODE = ('analysis/run_cached_identity_evidence.py', 'analysis/cached_identity_evidence.py',
        'analysis/review_cached_identity_evidence.py', 'analysis/physical_block_monotone_score.py',
        'analysis/refit_screened_nnls.py', 'analysis/run_nnls_solver_baseline.py',
        'analysis/run_small_mismatch_nnls_first_case.py', 'analysis/run_new_composition_development.py',
        'analysis/run_physical_block_prediction.py', 'analysis/run_physical_block_pool_refit.py',
        'analysis/physical_block_prediction.py', 'analysis/review_physical_block_prediction.py',
        'analysis/run_identity_confidence_joint_validation.py',
        'tests/test_cached_identity_evidence.py', 'tests/test_run_cached_identity_evidence.py',
        'docs/CACHED_IDENTITY_EVIDENCE_V1.md')
read, write, sha, require = score.read, score.write, score.sha, score.require


def config(case):
    if case in ('DEV_TRAIN', 'CHECK'):
        prefix = f'{V1}/cases/{case}'
        return dict(input=f'{V1}/prepared/{case}.npz', manifest=f'{V1}/block_manifest.json',
            records=f'{V1}/prepared/{case}_records.json', features=f'{prefix}/features.json',
            completion=f'{prefix}/completion.json', blocks=f'{prefix}/blocks',
            source=('results/small_mismatch_nnls_first_case/case/prepared_arrays.npz' if case == 'DEV_TRAIN'
                    else 'results/identity_confidence_joint_validation/cases/CAL1/prepared_arrays.npz'))
    prefix = f'{V2}/new_composition_case'
    return dict(input=f'{prefix}/prepared/portable.npz', manifest=f'{prefix}/prepared/block_manifest.json',
        records=f'{prefix}/nnls_rho/molecular_records.json', features=f'{prefix}/physical_features/features.json',
        completion=f'{prefix}/physical_features/completion.json', blocks=f'{prefix}/physical_features/blocks',
        source=f'{V2}/source_new_composition/prepared_arrays.npz')


def source_hashes(case):
    cfg = config(case)
    paths = [cfg[k] for k in ('input', 'manifest', 'records', 'features', 'completion')]
    paths += [f"{cfg['blocks']}/block_{i:03d}.{ext}" for i in range(34) for ext in ('npz', 'json')]
    return {p: sha(ROOT / p) for p in paths}


def runtime():
    return dict(numpy=np.__version__, scipy=scipy.__version__)


def prepare(args):
    require(not (args.output / 'design.json').exists(), 'PREPARATION_ALREADY_EXISTS')
    reference = read(ROOT / V2 / 'model/model.json')
    manifest = read(ROOT / V1 / 'block_manifest.json')
    for p,digest in read(ROOT/V2/'new_composition_case/design.json')['code'].items():
        require(sha(ROOT/p)==digest,'FROZEN_DEPENDENCY_CHANGED:'+p)
    sources = {}
    for case in CASES:
        cfg = config(case)
        require(read(ROOT/cfg['manifest']) == manifest, 'BLOCK_MANIFEST_DIFFERS')
        rows = read(ROOT/cfg['records'])
        require([r['lipid_name'] for r in rows] == manifest['molecular_names'], 'RECORD_ORDER')
        require(sum(r['molecular_truth'] for r in rows) == 125, 'TRUTH_DENOMINATOR')
        for i in range(34):
            stem = ROOT/cfg['blocks']/f'block_{i:03d}'
            record = read(stem.with_suffix('.json'))
            require(record['arrays_sha256'] == sha(stem.with_suffix('.npz')), 'OLD_BLOCK_HASH')
        sources[case] = dict(config=cfg, hashes=source_hashes(case), source_sha256=sha(ROOT/cfg['source']))
    original = read(ROOT/V1/'design.json')
    require(sources['DEV_TRAIN']['source_sha256'] == original['source_hashes']['/root/small_mismatch_nnls_first/results/prepared_arrays.npz'], 'DEV_ORIGINAL_SOURCE')
    require(sources['CHECK']['source_sha256'] == original['source_hashes']['/root/identity_confidence_joint_validation/results/cases/CAL1/prepared_arrays.npz'], 'CHECK_ORIGINAL_SOURCE')
    require(sources['NEW_COMPOSITION']['source_sha256'] == read(ROOT/V2/'new_composition_case/design.json')['source_prepared_sha256'], 'NEW_ORIGINAL_SOURCE')
    design = dict(version='CACHED_IDENTITY_EVIDENCE_V1', runtime=runtime(), sources=sources,
        code={p: sha(ROOT/p) for p in CODE}, arms=ARMS, options=score.OPTIONS,
        candidate_names=manifest['candidate_names'], molecular_names=manifest['molecular_names'],
        reference_sha256=sha(ROOT/V2/'model/model.json'), exposure='All three are exposed development data',
        refit_policy='winner fixed DEV5% set', independent_FDR_claim=False)
    write(args.output/'reference_model.json', reference)
    write(args.output/'reference_thresholds.json', read(ROOT/V2/'model/thresholds.json'))
    design['reference_thresholds_sha256'] = sha(args.output/'reference_thresholds.json')
    write(args.output/'design.json', design)
    write(args.output/'design_seal.json', dict(status='FROZEN_BEFORE_EXTRACTION_AND_FIT', design_sha256=sha(args.output/'design.json')))
    print('PREPARED', sha(args.output/'design.json'), flush=True)


def validate(out, case=None, source_check=True):
    design = read(out/'design.json')
    require(sha(out/'design.json') == read(out/'design_seal.json')['design_sha256'], 'DESIGN_CHANGED')
    for p, digest in design['code'].items():
        require(sha(ROOT/p) == digest, 'CODE_CHANGED:'+p)
    require(sha(out/'reference_model.json') == design['reference_sha256'], 'REFERENCE_CHANGED')
    require(sha(out/'reference_thresholds.json') == design['reference_thresholds_sha256'], 'REFERENCE_CUTS_CHANGED')
    if case and source_check:
        require(source_hashes(case) == design['sources'][case]['hashes'], 'SOURCE_CHANGED')
    return design


def features(args):
    from cached_identity_evidence import extract
    design = validate(args.output, args.case)
    cfg = config(args.case); directory = args.output/'cases'/args.case
    require(not (directory/'feature_completion.json').exists(), 'FEATURES_ALREADY_COMPLETE')
    with np.load(ROOT/cfg['input'], allow_pickle=False) as z:
        A, b = z['A'], z['b']
    manifest = read(ROOT/cfg['manifest']); blocks = []
    for i in range(34):
        with np.load(ROOT/cfg['blocks']/f'block_{i:03d}.npz', allow_pickle=False) as z:
            blocks.append({k: z[k] for k in z.files})
    result = extract(A, b, manifest, blocks)
    write(directory/'features.json', result['rows'])
    with (directory/'feature_arrays.npz').open('xb') as f:
        np.savez_compressed(f, **result['arrays'])
    write(directory/'feature_completion.json', dict(status='COMPLETE', case=args.case,
        source_hashes=design['sources'][args.case]['hashes'], design_sha256=sha(args.output/'design.json'),
        artifacts={p: sha(directory/p) for p in ('features.json', 'feature_arrays.npz')},
        no_new_solver_or_rho=True))
    print('FEATURES_COMPLETE', args.case, flush=True)


def case_data(out, case):
    design=validate(out, case)
    directory=out/'cases'/case
    receipt=read(directory/'feature_completion.json')
    require(receipt['status']=='COMPLETE' and receipt['case']==case and
        receipt['design_sha256']==sha(out/'design.json') and
        receipt['source_hashes']==design['sources'][case]['hashes'],'FEATURE_COMPLETION_BINDING')
    for p,digest in receipt['artifacts'].items(): require(sha(directory/p)==digest, 'FEATURE_ARTIFACT_CHANGED')
    review=read(directory/'independent_feature_review.json')
    require(review['status']=='PASS' and review['case']==case and
        review['design_sha256']==sha(out/'design.json') and
        review['feature_completion_sha256']==sha(directory/'feature_completion.json') and
        review['source_hashes']==receipt['source_hashes'] and
        review['feature_artifact_hashes']==receipt['artifacts'], 'FEATURE_REVIEW_REQUIRED')
    cfg=config(case)
    return read(ROOT/cfg['records']), read(ROOT/cfg['features']), read(directory/'features.json')


def new_matrix(rows, old_features, new_features, reference, model=None, added=None):
    reported=[r for r in rows if r['raw_solver_reported']]
    base=score.matrix(reported,old_features,reference)
    lookup={r['molecular_name']:r for r in new_features}
    fields=added if model is None else model['added_features']
    values=np.array([[lookup[r['lipid_name']][k] for k in fields] for r in reported],dtype=float).reshape(-1,len(fields))
    if model is None:
        means=values.mean(axis=0); scales=values.std(axis=0); scales[scales==0]=1
    else:
        means=np.array(model['mean']); scales=np.array(model['scale'])
    matrix=np.c_[base,(values-means)/scales]
    require(np.isfinite(matrix).all(),'NONFINITE_NEW_MATRIX')
    return matrix,means,scales


def objective(theta,x,y):
    w=theta[:-1]; logits=x@w+theta[-1]; residual=expit(logits)-y
    return float(np.logaddexp(0,logits).sum()-y@logits+.5*(w@w)), np.r_[x.T@residual+w,residual.sum()]


def predictions(rows, old_features, new_features, reference, model):
    if not any(r['raw_solver_reported'] for r in rows): return [None]*len(rows)
    x,_,_=new_matrix(rows,old_features,new_features,reference,model=model)
    p=iter(expit(x@np.array(model['coef'])+model['intercept']).tolist())
    return [next(p) if r['raw_solver_reported'] else None for r in rows]


def fit(args):
    validate(args.output, 'DEV_TRAIN')
    require(args.commit and len(args.commit)==40,'PUSHED_INPUT_COMMIT_REQUIRED')
    directory=args.output/'models'
    require(not (directory/'fit_reservation.json').exists(),'NO_SECOND_FIT')
    rows,old_features,new_features=case_data(args.output,'DEV_TRAIN')
    reference=read(args.output/'reference_model.json')
    write(directory/'fit_reservation.json',dict(input_commit=args.commit,design_sha256=sha(args.output/'design.json'),arms=ARMS,runtime=runtime()))
    results={}
    for arm,added in ARMS.items():
        x,means,scales=new_matrix(rows,old_features,new_features,reference,added=added)
        y=np.array([r['molecular_truth'] for r in rows if r['raw_solver_reported']],dtype=float)
        n=x.shape[1]; bounds=[(None,None)]*2+[(0,None)]*(n-2)+[(None,None)]
        start=time.monotonic()
        result=minimize(objective,np.zeros(n+1),args=(x,y),jac=True,bounds=bounds,method='L-BFGS-B',options=score.OPTIONS)
        value,gradient=objective(result.x,x,y); pg=gradient.copy()
        for j in range(2,n):
            if result.x[j]<=0 and pg[j]>0: pg[j]=0
        record=dict(success=bool(result.success),message=str(result.message),niter=int(result.nit),nfev=int(result.nfev),
            objective=value,gradient=gradient.tolist(),max_projected_gradient=float(abs(pg).max()),elapsed_seconds=time.monotonic()-start)
        write(directory/arm/'optimizer.json',record)
        require(result.success and np.isfinite(result.x).all(),'FIXED_FIT_FAILED:'+arm)
        model=dict(arm=arm,added_features=added,mean=means.tolist(),scale=scales.tolist(),coef=result.x[:-1].tolist(),
            intercept=float(result.x[-1]),reference_sha256=sha(args.output/'reference_model.json'),runtime=runtime())
        probabilities=predictions(rows,old_features,new_features,reference,model)
        cuts=score.choose_thresholds(rows,probabilities)
        write(directory/arm/'model.json',model); write(directory/arm/'thresholds.json',cuts)
        write(directory/arm/'train_scores.json',[dict(lipid_name=r['lipid_name'],score=p) for r,p in zip(rows,probabilities)])
        results[arm]=cuts
    def rank(arm):
        m=results[arm]['train_fdp5']
        return (m['TP'],-m['FP'],-len(ARMS[arm]),-list(ARMS).index(arm))
    winner=max(ARMS,key=rank)
    write(directory/'selection.json',dict(winner=winner,rule='DEV5% TP max, FP min, fewer new features, TRIM/DIRECT/JOINT order',
        all_dev_results=results,design_sha256=sha(args.output/'design.json'),independent_FDR_claim=False))
    paths=[p for p in directory.rglob('*.json') if p.name!='model_seal.json']
    write(directory/'model_seal.json',dict(status='FROZEN_BEFORE_CHECK_SCORING',winner=winner,
        artifacts={p.relative_to(directory).as_posix():sha(p) for p in sorted(paths)}))
    print(json.dumps(dict(winner=winner,DEV=results),ensure_ascii=False),flush=True)


def validate_models(out):
    seal=read(out/'models/model_seal.json')
    for p,digest in seal['artifacts'].items(): require(sha(out/'models'/p)==digest,'MODEL_CHANGED:'+p)
    return seal


def screen(args):
    require(args.case in ('CHECK','NEW_COMPOSITION'),'SCREEN_CHECK_ONLY')
    require(args.commit and len(args.commit)==40,'PUSHED_MODEL_COMMIT_REQUIRED')
    rows,old_features,new_features=case_data(args.output,args.case)
    seal=validate_models(args.output); reference=read(args.output/'reference_model.json')
    outputs={}
    for arm in ('BASELINE',*ARMS):
        if arm=='BASELINE':
            probabilities=score.score_rows(rows,old_features,reference); cuts=read(args.output/'reference_thresholds.json')
        else:
            probabilities=predictions(rows,old_features,new_features,reference,read(args.output/'models'/arm/'model.json'))
            cuts=read(args.output/'models'/arm/'thresholds.json')
        names={k:score.select_names(rows,probabilities,cuts[k+'_threshold']) for k in ('pool','fdp5','fdp1')}
        outputs[arm]=dict(counts={k:score.counts(rows,v) for k,v in names.items()},selected_names=names,scores=probabilities)
    write(args.output/'cases'/args.case/'screen.json',dict(status='COMPLETE',case=args.case,winner=seal['winner'],
        model_commit=args.commit,model_seal_sha256=sha(args.output/'models/model_seal.json'),
        design_sha256=sha(args.output/'design.json'),
        feature_completion_sha256=sha(args.output/'cases'/args.case/'feature_completion.json'),
        rows=rows,arms=outputs,refit_names=outputs[seal['winner']]['selected_names']['fdp5'],independent_FDR_claim=False))
    print(json.dumps(dict(case=args.case,winner=seal['winner'],counts={a:r['counts'] for a,r in outputs.items()})),flush=True)


def refit(args):
    import refit_screened_nnls as helper
    from run_new_composition_development import pool_account
    design=validate(args.output,source_check=False); validate_models(args.output)
    require(args.commit and len(args.commit)==40,'PUSHED_SCREEN_COMMIT_REQUIRED')
    screen=read(args.output/'cases'/args.case/'screen.json')
    require(args.screen_sha256 and sha(args.output/'cases'/args.case/'screen.json')==args.screen_sha256,
        'EXTERNALLY_SAVED_SCREEN_HASH_REQUIRED')
    seal=read(args.output/'models/model_seal.json')
    require(screen['status']=='COMPLETE' and screen['case']==args.case and
        screen['winner']==seal['winner'] and screen['design_sha256']==sha(args.output/'design.json') and
        screen['model_seal_sha256']==sha(args.output/'models/model_seal.json') and
        screen['feature_completion_sha256']==sha(args.output/'cases'/args.case/'feature_completion.json'),
        'SCREEN_BINDING_CHANGED')
    winner_cut=read(args.output/'models'/seal['winner']/'thresholds.json')['fdp5_threshold']
    require(screen['refit_names']==score.select_names(screen['rows'],screen['arms'][seal['winner']]['scores'],winner_cut),
        'WINNER_FIXED_DEV5_SET_CHANGED')
    require([r['lipid_name'] for r in screen['rows']]==design['molecular_names'],'SCREEN_RECORD_ORDER')
    source=args.source or ROOT/config(args.case)['source']
    require(sha(source)==design['sources'][args.case]['source_sha256'],'REFIT_SOURCE_CHANGED')
    directory=args.output/'cases'/args.case/'refit'
    require(not (directory/'completion.json').exists(),'REFIT_ALREADY_COMPLETE')
    selection=screen['refit_names']; names=design['candidate_names']
    binding=dict(design_sha256=sha(args.output/'design.json'),model_seal_sha256=sha(args.output/'models/model_seal.json'),
        screen_sha256=sha(args.output/'cases'/args.case/'screen.json'),source_sha256=sha(source),
        screen_pushed_commit=args.commit,policy='fixed DEV5% winner; all aliases; original gate .001',runtime=runtime())
    write(directory/'refit_binding.json',binding)
    if not selection:
        write(directory/'completion.json',dict(status='NOT_RUN_EMPTY_POOL',binding=binding,counts=score.counts(screen['rows'],[])))
        return
    with np.load(source,allow_pickle=False) as z: A,B,mask=z['A_solver'],z['B'],z['mask']
    indices=[i for i,name in enumerate(names) if name in set(selection)]
    helper.fit_screened(A,B,mask,indices,directory/'fit',workers=4,block_size=250,source_binding=binding)
    with np.load(directory/'fit/learned_arrays.npz',allow_pickle=False) as z: X,Bhat=z['X_hat'],z['B_hat']
    means=X[:,mask].mean(axis=1,dtype=np.float64)
    account=pool_account(screen['rows'],names,selection,means)
    write(directory/'candidate_records.json',account.pop('candidate_records'))
    write(directory/'molecular_records.json',account.pop('molecular_records'))
    residual=B[:,mask].astype(float)-Bhat[:,mask].astype(float)
    metrics=dict(squared_residual=float(np.sum(residual**2)),observation_squared_norm=float(np.sum(B[:,mask].astype(float)**2)))
    metrics['relative_l2']=(metrics['squared_residual']/metrics['observation_squared_norm'])**.5
    write(directory/'completion.json',dict(status='COMPLETE',binding=binding,**account,residual=metrics,
        artifacts={p:sha(directory/p) for p in ('candidate_records.json','molecular_records.json','fit/learned_arrays.npz','fit/completion.json','fit/binding.json')},independent_FDR_claim=False))
    print(json.dumps(dict(case=args.case,final=account['counts'])),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('prepare','features','fit','screen','refit'))
    parser.add_argument('--output',type=Path,default=OUT)
    parser.add_argument('--case',choices=CASES)
    parser.add_argument('--commit')
    parser.add_argument('--source',type=Path)
    parser.add_argument('--screen-sha256')
    args=parser.parse_args()
    if args.action in ('features','screen','refit'): require(args.case is not None,'CASE_REQUIRED')
    globals()[args.action](args)


if __name__=='__main__': main()
