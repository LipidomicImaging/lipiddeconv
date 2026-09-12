"""Shared fixed classifier, new full-library 5% CAL/HOLD, sequential handoffs."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
import warnings
from types import SimpleNamespace

for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[name] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import run_small_mismatch_nnls_first_case as engine

read, write, sha, ah, require = engine.read, engine.write, engine.sha, engine.ah, engine.require
CASES = (
    dict(id='CAL1', role='CAL', source='SUPPORTED_MISMATCH__CAL_R71_K125', seed=7401),
    dict(id='CAL2', role='CAL', source='SUPPORTED_MISMATCH__CAL_R72_K125', seed=7402),
    dict(id='HOLD1', role='HOLD', source='close_neighbor__HOLD_R71_K125', seed=7501),
    dict(id='HOLD2', role='HOLD', source='close_neighbor__HOLD_R72_K125', seed=7502),
)
CODE = ('analysis/run_identity_confidence_joint_validation.py', 'analysis/test_identity_confidence_joint_validation.py', 'analysis/run_small_mismatch_nnls_first_case.py',
        'analysis/run_nnls_solver_baseline.py', 'analysis/small_relative_spectral_mismatch.py', 'src/rho_zero.py',
        'docs/IDENTITY_CONFIDENCE_CONTINUOUS_PLAN.md')


def digest(value):
    return hashlib.sha256(engine.canonical(value)).hexdigest()


def runtime():
    import numpy, scipy, sklearn
    return dict(numpy=numpy.__version__,scipy=scipy.__version__,sklearn=sklearn.__version__)


def transform(rows, model=None):
    import numpy as np
    logs = np.array([[np.log10(max(r['X_hat'],1e-12)),np.log10(max(r['rho_zero'],1e-24))] for r in rows])
    mean, scale = (logs.mean(axis=0),np.maximum(logs.std(axis=0),1e-12)) if model is None else (np.array(model['mean']),np.array(model['scale']))
    z = (logs-mean)/scale
    return np.column_stack((z[:,0],z[:,1],z[:,0]**2,z[:,0]*z[:,1],z[:,1]**2,[float(r['rho_zero']==0) for r in rows])),mean,scale


def predict(row, model):
    if not row['raw_solver_reported']:
        return None
    z=[(math.log10(max(row[k],floor))-model['mean'][i])/model['scale'][i] for i,(k,floor) in enumerate((('X_hat',1e-12),('rho_zero',1e-24)))]
    phi=[z[0],z[1],z[0]**2,z[0]*z[1],z[1]**2,float(row['rho_zero']==0)]
    v=math.fsum(x*b for x,b in zip(phi,model['coef']))+model['intercept']
    return 1/(1+math.exp(-v)) if v>=0 else math.exp(v)/(1+math.exp(v))


def counts(rows, threshold):
    truths=sum(r['molecular_truth'] for r in rows); reportable=sum(r['reportable_truth'] for r in rows)
    raw=[r for r in rows if r['raw_solver_reported']]
    selected=[r for r in raw if threshold is not None and r['joint_score']>=threshold]
    tp=sum(r['molecular_truth'] for r in selected);fp=len(selected)-tp;rtp=sum(r['molecular_truth'] for r in raw)
    return dict(raw_TP=rtp,raw_FP=len(raw)-rtp,raw_FN=truths-rtp,TP=tp,FP=fp,FN=truths-tp,
                FDP=fp/len(selected) if selected else None,all_truth_recall=tp/truths,TP_retention=tp/rtp if rtp else 0,
                reportable_truth_recall=sum(r['reportable_truth'] for r in selected)/reportable if reportable else None,
                retained=len(selected),coverage=len(selected)/len(raw) if raw else 0,filter_induced_true_loss=rtp-tp,
                truth_count=truths,distinct_retained_identities=len({r['lipid_name'] for r in selected}))


def target(m):
    return bool(m['retained'] and m['FDP']<=.01 and m['all_truth_recall']>=.8 and m['TP_retention']>=.8)


def choose(groups):
    values=sorted({r['joint_score'] for rr in groups for r in rr if r['raw_solver_reported']},reverse=True)
    best=None;curve=[]
    for t in values:
        per=[counts(rr,t) for rr in groups];pooled=counts([r for rr in groups for r in rr],t)
        curve.append(dict(threshold=t,per_case=per,pooled=pooled))
        if target(pooled) and all(target(m) for m in per):
            best=curve[-1]
    return best,curve


def bindings(a):
    paths={str(a.development/n):sha(a.development/n) for n in ('result.json','molecular_records.json','design.json')}
    original=read(a.development/'result.json')
    require(original['normal_completion'] and original['all_outputs_finite'], 'DEVELOPMENT_INCOMPLETE')
    require(sha(a.development/'molecular_records.json')==original['artifact_hashes']['molecular_records.json'],'DEVELOPMENT_CHANGED')
    require(sha(a.development/'design.json')==original['artifact_hashes']['design.json'],'DEVELOPMENT_DESIGN_CHANGED')
    provenance=read(a.components/'provenance.json')
    require(provenance['status']=='COMPLETE','COMPONENTS_INCOMPLETE')
    for n,h in provenance['outputs'].items():
        require(sha(a.components/n)==h['sha256'],'COMPONENT_CHANGED:'+n)
        paths[str(a.components/n)]=h['sha256']
    paths[str(a.components/'provenance.json')]=sha(a.components/'provenance.json')
    for entry in CASES:
        p=a.source_root/entry['source'];info=read(p/'input.json')
        require(sha(p/'observation_truth.npz')==info['observation_file_sha256'],'SOURCE_CONTAINER_CHANGED')
        for n in ('input.json','observation_truth.npz'):
            paths[str(p/n)]=sha(p/n)
    return dict(source_files=paths,code={n:sha(ROOT/n) for n in CODE},runtime=runtime())


def freeze(a):
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.exceptions import ConvergenceWarning
    require(not (a.output/'contract.json').exists(),'EXISTING_CONTRACT_PRESERVED')
    b=bindings(a);require(b['runtime']==dict(numpy='2.1.3',scipy='1.15.3',sklearn='1.6.1'),'RUNTIME_CHANGED')
    rows=read(a.development/'molecular_records.json');reported=[r for r in rows if r['raw_solver_reported']]
    require(len(reported)==174 and sum(r['molecular_truth'] for r in reported)==125,'DEVELOPMENT_MEMBERSHIP_CHANGED')
    x,mean,scale=transform(reported)
    with warnings.catch_warnings():
        warnings.simplefilter('error',ConvergenceWarning)
        lr=LogisticRegression(C=1.,solver='lbfgs',max_iter=2000,tol=1e-8).fit(x,[r['molecular_truth'] for r in reported])
    model=dict(mean=mean.tolist(),scale=scale.tolist(),coef=lr.coef_[0].tolist(),intercept=float(lr.intercept_[0]),
               n_iter=lr.n_iter_.tolist(),training_names=sorted(r['lipid_name'] for r in reported),training_case=original_case(a),
               score_is_uncalibrated=True,features=['log_X','log_rho','log_X_squared','log_X_log_rho','log_rho_squared','rho_is_zero'])
    # Check inference formula against library implementation using development inputs only.
    require(np.allclose([predict(r,model) for r in reported],lr.predict_proba(x)[:,1],rtol=1e-12,atol=1e-15),'MODEL_FORMULA_DISAGREES')
    write(a.output/'model.json',model)
    c=dict(version='JOINT_NEW_CASE_VALIDATION_V1',cases=CASES,bindings=b,model_sha256=sha(a.output/'model.json'),
           generator='Original physical-ion lognormal mean1 CV.05, column L2, seed frozen per case, one dataset scalar',
           solver='Unchanged full391 pixel NNLS and KKT via first-case engine; global-mean original rho_zero; mean gate .001',
           training='One fixed six-feature L2 logistic model, C1; no auxiliary features or hyperparameter search',
           calibration='One common tied-score cutoff; maximize pooled retention subject to per-case and pooled 1%/80% target',
           execution='CAL1 feasibility before CAL2; both CAL and seals before HOLD; per-case review/download/push/ACK',
           scope='New perturbation observations, fixed library/spatial construction; disjoint CAL/HOLD true pools, not unseen candidate names or real MSI',
           independent_population_FDR_guarantee=False, GPU=False, no_deletion=True)
    write(a.output/'contract.json',c)
    write(a.output/'contract_seal.json',dict(status='MODEL_FROZEN_BEFORE_NEW_SOLVES',contract_sha256=sha(a.output/'contract.json')))
    print('MODEL_AND_CONTRACT_FROZEN',sha(a.output/'contract.json'),flush=True)


def original_case(a):
    return read(a.development/'result.json')['case']


def validate(a):
    c=read(a.output/'contract.json')
    require(sha(a.output/'contract.json')==read(a.output/'contract_seal.json')['contract_sha256'],'CONTRACT_CHANGED')
    require(c['bindings']==bindings(a),'SOURCES_OR_IMPLEMENTATION_CHANGED')
    require(sha(a.output/'model.json')==c['model_sha256'],'MODEL_CHANGED')
    require(c['cases']==list(CASES),'MEMBERSHIP_CHANGED')
    return c


def prepare(a):
    import numpy as np
    import small_relative_spectral_mismatch as mismatch
    c=validate(a)
    require(not (a.output/'preparation_seal.json').exists(),'PREPARED_INPUTS_PRESERVED')
    with np.load(a.components/'components.npz',allow_pickle=False) as z:
        A,C=z['A'].copy(),z['components'].copy()
    require(A.shape==(1084,391) and ah(A)==engine.A_SHA,'LIBRARY_CHANGED')
    metadata=read(a.components/'metadata.json');parts=read(a.components/'component_metadata.json')
    summaries={};seen={read(a.development/'design.json')['scientific']['A_target_sha256']}
    for entry in CASES:
        p=a.source_root/entry['source'];info=read(p/'input.json')
        with np.load(p/'observation_truth.npz',allow_pickle=False) as z:
            X,mask=z['X_true'].copy(),z['mask'].copy()
        require(ah(X)==info['X_true_sha256'] and X.shape==(391,*mask.shape) and mask.dtype==bool,'SOURCE_X_CHANGED')
        require(np.isfinite(X).all() and (X>=0).all() and (X[:,~mask]==0).all(),'INVALID_SOURCE_X')
        truth=np.flatnonzero(np.any(X[:,mask]>0,axis=1)).tolist()
        require(set(truth)==set(info['truth_indices']) and len(truth)==125,'SOURCE_TRUTH_CHANGED')
        target_A,audit=mismatch.build_target_library(A,C,parts,seed=entry['seed'],relative_sd=.05)
        require(ah(target_A) not in seen,'TARGET_NOT_FRESH');seen.add(ah(target_A))
        unscaled=target_A.astype(float)@X[:,mask].astype(float)
        scalar=engine.SIGNAL_TARGET/float(np.median(np.linalg.norm(unscaled,axis=0)))
        Xtrue=X.astype(float)*scalar;B=np.zeros((1084,*mask.shape),dtype=np.float32);B[:,mask]=(unscaled*scalar).astype(np.float32)
        names=sorted({metadata['lipid_name'][j] for j in truth})
        reportable=sorted({metadata['lipid_name'][j] for j in np.flatnonzero(Xtrue[:,mask].mean(axis=1)>.001)})
        perturbed=sorted({metadata['lipid_name'][j] for j in truth if audit['relative_column_changes'][j]>1e-10})
        require(len(names)==len(reportable)==125,'TRUTH_REPORTABILITY_CHANGED')
        achieved=float(np.median(np.linalg.norm(B[:,mask].astype(float),axis=0)))
        require(abs(achieved-engine.SIGNAL_TARGET)<=8*np.finfo(np.float32).eps*engine.SIGNAL_TARGET,'SIGNAL_SCALAR_FAILED')
        out=a.output/'cases'/entry['id'];out.mkdir(parents=True,exist_ok=False)
        engine.npz_once(out/'prepared_arrays.npz',A_solver=A,A_target=target_A,X_true=Xtrue,B=B,mask=mask)
        write(out/'metadata.json',metadata);write(out/'perturbation_audit.json',audit)
        content=dict(**entry,contract_sha256=sha(a.output/'contract.json'),source_input_sha256=sha(p/'input.json'),source_X_sha256=ah(X),
                     source_container_sha256=sha(p/'observation_truth.npz'),A_solver_sha256=ah(A),A_target_sha256=ah(target_A),
                     B_sha256=ah(B),X_true_sha256=ah(Xtrue),global_scalar=scalar,foreground_signal_norm=achieved,
                     truth_indices=truth,truth_names=names,reportable_truth_names=reportable,actually_perturbed_truth_names=perturbed,
                     full_library_candidates=391,old_source_omissions_used=False,
                     files={n:sha(out/n) for n in ('prepared_arrays.npz','metadata.json','perturbation_audit.json')})
        write(out/'input.json',dict(fingerprint=digest(content),scientific=content))
        summaries[entry['id']]=dict(input_sha256=sha(out/'input.json'),fingerprint=digest(content),actual_perturbed_truth_count=len(perturbed))
        print('PREPARED',entry['id'],'TRUTH',len(names),'ACTUALLY_PERTURBED',len(perturbed),flush=True)
    cal=set(read(a.output/'cases/CAL1/input.json')['scientific']['truth_names'])
    hold=set(read(a.output/'cases/HOLD1/input.json')['scientific']['truth_names'])
    require(not cal&hold,'CAL_HOLD_TRUTH_OVERLAP')
    write(a.output/'preparation_seal.json',dict(status='FOUR_INPUTS_FROZEN_BEFORE_NNLS',contract_sha256=sha(a.output/'contract.json'),cases=summaries,CAL_HOLD_truth_overlap=0))


def case_input(a):
    validate(a);out=a.output/'cases'/a.case;d=read(out/'input.json');s=d['scientific']
    require(digest(s)==d['fingerprint'],'CASE_FINGERPRINT_CHANGED')
    seal=read(a.output/'preparation_seal.json')
    require(sha(out/'input.json')==seal['cases'][a.case]['input_sha256'],'CASE_INPUT_CHANGED')
    for n,h in s['files'].items():require(sha(out/n)==h,'CASE_ARRAY_CHANGED:'+n)
    require(s['id']==a.case and s['contract_sha256']==sha(a.output/'contract.json'),'CASE_BINDING_CHANGED')
    return out,d,s


def require_ack(a, case):
    out=a.output/'cases'/case;ack=read(a.output/'handoffs'/f'{case}.json')
    require(ack['case']==case and ack['review_sha256']==sha(out/'review.json') and ack['result_sha256']==sha(out/'result.json'), 'REVIEW_ACK_CHANGED')
    require(ack['download_hashes_verified'] is True and len(ack['pushed_commit'])==40,'UNSAVED_CASE_NO_HANDOFF')


def gate(a):
    if a.case=='CAL2':
        require_ack(a,'CAL1');require(read(a.output/'cases/CAL1/review.json')['calibration_feasible'],'CAL1_FAILED_STOP_VERSION')
    if a.case.startswith('HOLD'):
        for name in ('CAL1','CAL2'):require_ack(a,name)
        seal=read(a.output/'calibration_seal.json')
        require(seal['status']=='SEALED_FOR_HOLD' and seal['threshold_sha256']==sha(a.output/'threshold.json'),'CALIBRATION_NOT_SEALED')
        t=read(a.output/'threshold.json')
        require(seal['model_sha256']==t['model_sha256']==sha(a.output/'model.json') and seal['contract_sha256']==sha(a.output/'contract.json'),'CALIBRATION_MODEL_CHANGED')
        require(t['CAL_results']=={n:sha(a.output/f'cases/{n}/result.json') for n in ('CAL1','CAL2')},'CALIBRATION_RESULTS_CHANGED')
    if a.case=='HOLD2':
        require_ack(a,'HOLD1');require(read(a.output/'cases/HOLD1/review.json')['target_met'],'HOLD1_FAILED_STOP_VERSION')


def make_records(s, metadata, means, scores, model):
    candidates=[];truth=set(s['truth_names']);reportable=set(s['reportable_truth_names'])
    for j,name in enumerate(metadata['lipid_name']):
        candidates.append(dict(candidate_index=j,candidate_id=metadata['candidate_id'][j],lipid_name=name,lipid_class=metadata['lipid_class'][j],
                               X_hat=float(means[j]),candidate_truth=j in s['truth_indices'],molecular_truth=name in truth,
                               reportable_truth=name in reportable,raw_solver_reported=j in scores,rho_zero=scores[j]['rho_zero'] if j in scores else None,
                               rho_details=scores.get(j),case=s['id'],split=s['role']))
    molecules=[]
    for name in dict.fromkeys(metadata['lipid_name']):
        rr=[r for r in candidates if r['lipid_name']==name];active=[r for r in rr if r['raw_solver_reported']]
        r=dict(case=s['id'],lipid_name=name,molecular_truth=name in truth,reportable_truth=name in reportable,
               actually_perturbed_truth=name in s['actually_perturbed_truth_names'],raw_solver_reported=bool(active),
               X_hat=sum(x['X_hat'] for x in active),rho_zero=max(x['rho_zero'] for x in active) if active else None,
               candidate_indices=[x['candidate_index'] for x in rr],reported_candidate_indices=[x['candidate_index'] for x in active])
        r['joint_score']=predict(r,model);molecules.append(r)
    return candidates,molecules


def solve(a):
    import numpy as np
    import run_nnls_solver_baseline as nnls_engine
    import rho_zero
    out,d,s=case_input(a);gate(a)
    require(not (out/'result.json').exists(),'COMPLETED_CASE_REQUIRES_REVIEW_NOT_RERUN')
    with np.load(out/'prepared_arrays.npz',allow_pickle=False) as z:
        A,B,mask=z['A_solver'].astype(float),z['B'].copy(),z['mask'].copy()
    require((B[:,~mask]==0).all(),'NONZERO_BACKGROUND')
    if (out/'nnls_complete.json').exists():
        done=read(out/'nnls_complete.json');require(sha(out/'learned_arrays.npz')==done['arrays_sha256'],'SAVED_NNLS_CHANGED')
        with np.load(out/'learned_arrays.npz',allow_pickle=False) as z:Xhat,Bhat=z['X_hat'].copy(),z['B_hat'].copy()
    else:
        # The reused engine's CASE global changes only progress text; numerical settings are untouched.
        engine.CASE=a.case
        Xhat,Bhat,kkt=engine.fit_nnls(SimpleNamespace(output=out),d,A,B,mask,np,nnls_engine)
        engine.npz_once(out/'learned_arrays.npz',X_hat=Xhat,B_hat=Bhat)
        done=dict(fingerprint=d['fingerprint'],arrays_sha256=sha(out/'learned_arrays.npz'),normal_completion=True,all_outputs_finite=True,
                  KKT=kkt,all_pixels_KKT_checked_before_float32=True,
                  relative_reconstruction_residual=float(np.linalg.norm(Bhat[:,mask].astype(float)-B[:,mask].astype(float))/np.linalg.norm(B[:,mask].astype(float))))
        write(out/'nnls_complete.json',done)
    require(done['fingerprint']==d['fingerprint'] and done['normal_completion'] and done['all_outputs_finite'] and done['KKT']['max_bound_ratio']<=1,'NNLS_INVALID')
    require(np.isfinite(Xhat).all() and np.isfinite(Bhat).all() and (Xhat>=0).all(),'NONFINITE_OR_NEGATIVE')
    means=Xhat[:,mask].mean(axis=1,dtype=float);b=B[:,mask].mean(axis=1,dtype=float);weights=rho_zero.identity_weights(A)
    scores={};reported=np.flatnonzero(means>.001).tolist()
    for number,j in enumerate(reported,1):
        p=out/'rho'/f'candidate_{j:04d}.json'
        if p.exists():
            item=read(p);require(item['fingerprint']==d['fingerprint'] and item['b_sha256']==ah(b) and item['arrays_sha256']==done['arrays_sha256'],'RHO_BINDING_CHANGED')
        else:
            values=rho_zero.rho_zero_from_weighted_case(A,b,j,weights=weights)
            item=dict(fingerprint=d['fingerprint'],candidate_index=j,b_sha256=ah(b),arrays_sha256=done['arrays_sha256'],result=values);write(p,item)
        require(all(math.isfinite(v) for v in item['result'].values()) and item['result']['rho_zero']>=0,'RHO_NONFINITE')
        scores[j]=item['result']
        if number%25==0 or number==len(reported):print('RHO',a.case,number,len(reported),flush=True)
    candidates,molecules=make_records(s,read(out/'metadata.json'),means,scores,read(a.output/'model.json'))
    write(out/'candidate_records.json',candidates);write(out/'molecular_records.json',molecules)
    outputs=['input.json','prepared_arrays.npz','metadata.json','perturbation_audit.json','learned_arrays.npz','nnls_complete.json','candidate_records.json','molecular_records.json']
    outputs += [p.relative_to(out).as_posix() for directory in ('nnls_blocks','rho') for p in sorted((out/directory).iterdir()) if p.is_file()]
    result=dict(case=a.case,fingerprint=d['fingerprint'],status='COMPLETE_AWAITING_REVIEW',normal_completion=True,all_outputs_finite=True,
                model_sha256=sha(a.output/'model.json'),artifacts={n:sha(out/n) for n in outputs})
    if s['role']=='HOLD':result['threshold_seal_sha256']=sha(a.output/'calibration_seal.json')
    write(out/'result.json',result);print('COMPLETE_AWAITING_REVIEW',a.case,flush=True)


def review(a):
    import numpy as np
    out,d,s=case_input(a);result=read(out/'result.json');done=read(out/'nnls_complete.json')
    require(result['case']==a.case and result['fingerprint']==d['fingerprint'] and result['model_sha256']==sha(a.output/'model.json'),'RESULT_BINDING_CHANGED')
    require(result['normal_completion'] and result['all_outputs_finite'] and done['normal_completion'] and done['all_outputs_finite'] and done['fingerprint']==d['fingerprint'],'COMPLETION_INVALID')
    require(done['arrays_sha256']==sha(out/'learned_arrays.npz'),'COMPLETION_ARRAY_CHANGED')
    for n,h in result['artifacts'].items():require(sha(out/n)==h,'ARTIFACT_CHANGED:'+n)
    with np.load(out/'prepared_arrays.npz',allow_pickle=False) as z:A,At,Xtrue,B,mask=(z[k].copy() for k in ('A_solver','A_target','X_true','B','mask'))
    require(ah(A)==s['A_solver_sha256'] and ah(At)==s['A_target_sha256'] and ah(Xtrue)==s['X_true_sha256'] and ah(B)==s['B_sha256'],'INPUT_ARRAY_HASH_CHANGED')
    metadata=read(out/'metadata.json')
    actual_truth=np.flatnonzero(np.any(Xtrue[:,mask]>0,axis=1)).tolist()
    require(actual_truth==s['truth_indices'] and len(actual_truth)==125 and s['A_solver_sha256']==engine.A_SHA,'TRUTH_OR_LIBRARY_CHANGED')
    require(sorted({metadata['lipid_name'][j] for j in actual_truth})==s['truth_names'],'MOLECULAR_TRUTH_CHANGED')
    require(sorted({metadata['lipid_name'][j] for j in np.flatnonzero(Xtrue[:,mask].mean(axis=1)>.001)})==s['reportable_truth_names'],'REPORTABLE_TRUTH_CHANGED')
    delta=np.linalg.norm(At.astype(float)-A.astype(float),axis=0)/np.linalg.norm(A.astype(float),axis=0)
    require(sorted({metadata['lipid_name'][j] for j in actual_truth if delta[j]>1e-10})==s['actually_perturbed_truth_names'],'PERTURBED_TRUTH_CHANGED')
    require(np.allclose(At.astype(float)@Xtrue[:,mask],B[:,mask],rtol=8*np.finfo(np.float32).eps,atol=0),'FORWARD_EQUATION_FAILED')
    with np.load(out/'learned_arrays.npz',allow_pickle=False) as z:Xhat,Bhat=z['X_hat'].copy(),z['B_hat'].copy()
    require(Xhat.shape==Xtrue.shape and Bhat.shape==B.shape and np.isfinite(Xhat).all() and np.isfinite(Bhat).all() and (Xhat>=0).all(),'LEARNED_ARRAYS_INVALID')
    require(np.allclose(A.astype(float)@Xhat[:,mask].astype(float),Bhat[:,mask],rtol=8*np.finfo(np.float32).eps,atol=0),'FITTED_EQUATION_FAILED')
    require((Xhat[:,~mask]==0).all() and (Bhat[:,~mask]==0).all(),'BACKGROUND_CHANGED')
    kkt=dict(max_dual_violation=0.,max_complementarity=0.,max_bound_ratio=0.)
    for start in range(0,int(mask.sum()),250):
        stop=min(start+250,int(mask.sum()));stem=f'nnls_blocks/block_{start:06d}_{stop:06d}';record=read(out/(stem+'.json'))
        require(record['fingerprint']==d['fingerprint'] and record['start']==start and record['stop']==stop and record['all_pixels_checked_before_float32'] and record['KKT']['max_bound_ratio']<=1,'BLOCK_INVALID')
        require(sha(out/(stem+'.npz'))==record['array_sha256'],'BLOCK_HASH_CHANGED')
        with np.load(out/(stem+'.npz'),allow_pickle=False) as z:require(np.array_equal(z['X_hat'],Xhat[:,mask][:,start:stop]),'BLOCK_MEMBERSHIP_CHANGED')
        for k in kkt:kkt[k]=max(kkt[k],record['KKT'][k])
    require(kkt==done['KKT'] and done['all_pixels_KKT_checked_before_float32'],'KKT_AGGREGATE_CHANGED')
    means=Xhat[:,mask].mean(axis=1,dtype=float);reported=np.flatnonzero(means>.001).tolist();b=B[:,mask].mean(axis=1,dtype=float)
    expected_files={'input.json','prepared_arrays.npz','metadata.json','perturbation_audit.json','learned_arrays.npz','nnls_complete.json','candidate_records.json','molecular_records.json'}
    expected_files |= {f'rho/candidate_{j:04d}.json' for j in reported}
    expected_files |= {f'nnls_blocks/block_{start:06d}_{min(start+250,int(mask.sum())):06d}.{ext}' for start in range(0,int(mask.sum()),250) for ext in ('json','npz')}
    require(set(result['artifacts'])==expected_files,'RESULT_ARTIFACT_MEMBERSHIP_CHANGED')
    candidates=read(out/'candidate_records.json');rows=read(out/'molecular_records.json');metadata=read(out/'metadata.json');model=read(a.output/'model.json')
    require(len(candidates)==391 and len(rows)==len(set(metadata['lipid_name']))==len({r['lipid_name'] for r in rows}),'RESULT_MEMBERSHIP_CHANGED')
    for i,r in enumerate(candidates):
        require(r['candidate_index']==i and r['lipid_name']==metadata['lipid_name'][i] and r['X_hat']==means[i] and r['raw_solver_reported']==(i in reported),'CANDIDATE_RECORD_CHANGED')
        if i in reported:
            item=read(out/f'rho/candidate_{i:04d}.json');v=item['result']
            require(item['b_sha256']==ah(b) and item['candidate_index']==i and v==r['rho_details'],'RHO_RECORD_CHANGED')
            expected=max(0.,(v['q_deleted']-v['q_star'])/(float(b@b)+1e-12))
            require(math.isclose(expected,r['rho_zero'],rel_tol=1e-12,abs_tol=1e-22),'RHO_FORMULA_CHANGED')
    for r in rows:
        indices=[i for i,n in enumerate(metadata['lipid_name']) if n==r['lipid_name']];active=[i for i in indices if i in reported]
        require(r['candidate_indices']==indices and r['reported_candidate_indices']==active and r['raw_solver_reported']==bool(active),'ALIAS_MEMBERSHIP_CHANGED')
        require(r['X_hat']==sum(candidates[i]['X_hat'] for i in active) and r['rho_zero']==(max(candidates[i]['rho_zero'] for i in active) if active else None),'MOLECULAR_AGGREGATION_CHANGED')
        require(r['molecular_truth']==(r['lipid_name'] in s['truth_names']) and r['reportable_truth']==(r['lipid_name'] in s['reportable_truth_names']),'TRUTH_LABEL_CHANGED')
        require(r['actually_perturbed_truth']==(r['lipid_name'] in s['actually_perturbed_truth_names']),'PERTURBED_LABEL_CHANGED')
        if active:
            z=(np.array([np.log10(max(r['X_hat'],1e-12)),np.log10(max(r['rho_zero'],1e-24))])-model['mean'])/model['scale']
            phi=np.array([z[0],z[1],z[0]**2,z[0]*z[1],z[1]**2,float(r['rho_zero']==0)]);value=float(phi@np.array(model['coef'])+model['intercept'])
            score=1/(1+math.exp(-value)) if value>=0 else math.exp(value)/(1+math.exp(value))
            require(abs(score-r['joint_score'])<=2e-12,'PREDICTION_CHANGED')
        else:require(r['joint_score'] is None,'UNREPORTED_SCORE')
    report=dict(case=a.case,status='PASS',process_valid=True,model_sha256=sha(a.output/'model.json'),result_sha256=sha(out/'result.json'),
                fingerprint=d['fingerprint'],all_artifacts_hash_verified=True,all_pixels_KKT_checked_before_float32=True,
                actual_perturbed_truth_count=len(s['actually_perturbed_truth_names']),no_optimization_rerun=True,
                raw=counts(rows,0.),source_old_omissions_used=False)
    if s['role']=='CAL':
        best,curve=choose([rows]);write(out/'calibration_curve.json',curve)
        report.update(calibration_feasible=best is not None,feasible_point=best,not_independent_validation=True)
    else:
        gate(a);require(result['threshold_seal_sha256']==sha(a.output/'calibration_seal.json'),'HOLD_THRESHOLD_CHANGED')
        t=read(a.output/'threshold.json')['threshold'];m=counts(rows,t)
        report.update(threshold=t,metrics=m,target_met=target(m),no_HOLD_threshold_curve=True)
    write(out/'review.json',report);print(json.dumps(report,indent=2),flush=True)


def calibrate(a):
    validate(a)
    for name in ('CAL1','CAL2'):
        require_ack(a,name);require(read(a.output/f'cases/{name}/review.json')['status']=='PASS','CAL_REVIEW_FAILED')
    groups=[read(a.output/f'cases/{name}/molecular_records.json') for name in ('CAL1','CAL2')]
    best,curve=choose(groups);write(a.output/'CAL_combined_curve.json',curve)
    if best is None:
        write(a.output/'version_outcome.json',dict(status='NO_FEASIBLE_SHARED_CAL_CUTOFF',HOLD_started=False,broader_goal_complete=False));print('NO_FEASIBLE_SHARED_CAL_CUTOFF');return
    write(a.output/'threshold.json',dict(**best,model_sha256=sha(a.output/'model.json'),CAL_results={n:sha(a.output/f'cases/{n}/result.json') for n in ('CAL1','CAL2')}))
    write(a.output/'calibration_seal.json',dict(status='SEALED_FOR_HOLD',threshold_sha256=sha(a.output/'threshold.json'),model_sha256=sha(a.output/'model.json'),contract_sha256=sha(a.output/'contract.json')))
    print('CALIBRATION_SEALED_FOR_HOLD',best['threshold'],flush=True)


def evaluate(a):
    validate(a)
    for name in ('HOLD1','HOLD2'):
        require_ack(a,name)
        result=read(a.output/f'cases/{name}/result.json')
        require(sha(a.output/f'cases/{name}/molecular_records.json')==result['artifacts']['molecular_records.json'],'HOLD_RECORDS_CHANGED')
    seal=read(a.output/'calibration_seal.json')
    require(seal['threshold_sha256']==sha(a.output/'threshold.json') and seal['model_sha256']==sha(a.output/'model.json'),'FINAL_THRESHOLD_CHANGED')
    reviews=[read(a.output/f'cases/{n}/review.json') for n in ('HOLD1','HOLD2')]
    t=read(a.output/'threshold.json')['threshold'];pooled=counts([r for n in ('HOLD1','HOLD2') for r in read(a.output/f'cases/{n}/molecular_records.json')],t)
    passed=all(r['status']=='PASS' and r['target_met'] for r in reviews) and target(pooled)
    write(a.output/'final_evaluation.json',dict(status='TARGET_MET_ON_TWO_FROZEN_HOLD_CASES' if passed else 'NO_GO',per_case=reviews,pooled=pooled,
          independent_new_perturbations=True,real_data_validation=False,scope='fixed391 library, existing spatial templates, disjoint CAL/HOLD truth pools, assumed CV5% physical-ion mismatch',
          uncertainty='Two case realizations do not reliably estimate a population FDR confidence bound; identities within a mixture are dependent. Report raw error counts, not a <=1% population guarantee.',
          threshold_seal_sha256=sha(a.output/'calibration_seal.json'),model_sha256=sha(a.output/'model.json')))
    print('FINAL_EVALUATION',passed,pooled,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=('freeze','prepare','solve','review','calibrate','evaluate'))
    p.add_argument('--development',type=Path,required=True);p.add_argument('--source-root',type=Path,required=True);p.add_argument('--components',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--case',choices=tuple(e['id'] for e in CASES))
    a=p.parse_args()
    for k in ('development','source_root','components','output'):setattr(a,k,getattr(a,k).resolve())
    require(all(not a.output.is_relative_to(p) for p in (a.development,a.source_root,a.components)),'OUTPUT_MUST_BE_SEPARATE')
    if a.action in ('solve','review'):require(a.case is not None,'CASE_REQUIRED')
    a.output.mkdir(parents=True,exist_ok=True)
    try:globals()[a.action](a)
    except BaseException as exc:
        write(a.output/f'failure_{time.time_ns()}.json',dict(action=a.action,case=a.case,error=repr(exc),no_automatic_retry=True))
        raise
