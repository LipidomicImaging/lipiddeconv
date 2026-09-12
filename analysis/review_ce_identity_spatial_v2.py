"""Independent cached evidence, LP proof and accounting checks for spatial V2."""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path

for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
G=1e-6
KINDS=('MILD','close_neighbor','relatively_isolated')


def read(p):return json.loads(p.read_text(encoding='utf-8'))


def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(8*1024**2),b''):h.update(block)
    return h.hexdigest()


def write(p,d):p.write_text(json.dumps(d,indent=2,allow_nan=False)+'\n',encoding='utf-8',newline='\n')


def hashes(root,manifest):
    for name,h in manifest.items():assert sha(root/name)==h,name


def state(r,e):
    if r.get('input_status')=='NO_SPATIAL_EVIDENCE':return 'NO_SPATIAL_EVIDENCE'
    if r.get('numerical_error'):return 'NUMERICALLY_UNRESOLVED'
    f,d=r['full'],r['deleted']
    if f['lower']>e+G:return 'FULL_MODEL_INCOMPATIBLE'
    if not f['upper']<=e-G:return 'THRESHOLD_UNRESOLVED'
    if d['lower']>e+G:return 'RETAINED'
    return 'REPLACEABLE' if d['upper']<=e-G else 'THRESHOLD_UNRESOLVED'


def accounting(cases,e):
    rows=[r for c in cases for r in c['records']];kept=[r for r in rows if state(r,e)=='RETAINED']
    truth=sum(c['truth_count'] for c in cases);reportable=sum(c['reportable_truth_count'] for c in cases)
    raw=sum(r['molecular_truth'] for r in rows);tp=sum(r['molecular_truth'] for r in kept);false=len(kept)-tp
    return dict(TP=tp,FP=false,FN=truth-tp,raw_solver_TP=raw,raw_solver_FP=len(rows)-raw,raw_solver_FN=truth-raw,
        filter_induced_true_loss=raw-tp,TP_retention=tp/raw if raw else None,
        all_truth_recall=tp/truth if truth else None,
        reportable_truth_recall=sum(r['reportable_truth'] for r in kept)/reportable if reportable else None,
        FDP=false/len(kept) if kept else 0.,retained_count=len(kept),distinct_identities=len({r['lipid_name'] for r in kept}),
        case_count=len(cases),status_counts=dict(Counter(state(r,e) for r in rows)))


def evidence_review(root,key,old):
    import numpy as np
    case=root/'cases'/key;done=read(case/'training_complete.json');hashes(case,done['files'])
    design=read(root/'design.json');info=design['scientific']['cases'][key]
    assert done['fingerprint']==design['fingerprint'] and read(case/'input.json')==info
    old_manifest=read(old/'output_manifest.json');assert sha(old/'output_manifest.json')==design['scientific']['source_binding']['old_input_output_manifest_sha256']
    rel='inputs/MILD__CAL_R1_K125/pilot_input.json';assert sha(old/rel)==old_manifest[rel]
    names=read(old/rel)['names'];kept=info['kept'];solver_names=[names[i] for i in kept]
    with np.load(case/'observation_truth.npz') as z:B=z['B'];mask=z['mask'];truth_array=z['X_true']
    assert hashlib.sha256(B.tobytes()).hexdigest()==info['B_sha256']
    assert hashlib.sha256(truth_array.tobytes()).hexdigest()==info['X_true_sha256']
    with np.load(case/'training/learned_arrays.npz') as z:X=z['X_hat'];Bhat=z['B_hat']
    assert X.shape==(len(kept),*mask.shape) and B.shape==Bhat.shape
    assert np.isfinite(B).all() and np.isfinite(X).all() and np.isfinite(Bhat).all() and (X>=0).all()
    run=read(case/'training/solver_run.json');history=read(case/'training/training_history.json')
    assert run['stop_reason'] in ('converged','max_epochs') and 1<=run['stopped_epoch']<=3000
    assert history['epoch'][-1]==run['stopped_epoch']
    for field in ('train_total','eval_total','eval_physical_loss','eval_raw_losses','automatic_loss_weights'):
        assert len(history[field])==len(history['epoch']) and np.isfinite(np.asarray(history[field],float)).all()
    with np.load(case/'evidence.npz') as z:ev={k:z[k] for k in z.files}
    assert ev['kept'].tolist()==kept and hashlib.sha256(ev['A'].astype(np.float32).tobytes()).hexdigest()==info['A_sha256']
    rows=read(case/'molecular_records.json');assert [r['lipid_name'] for r in rows]==list(dict.fromkeys(names))
    truth={names[i] for i in info['truth_indices']};reportable={names[i] for i in info['reportable_truth_indices']}
    xf=X[:,mask].astype(float);bf=B[:,mask].astype(float);means=xf.mean(axis=1)
    maxima=dict(weight_absolute_error=0.,local_absolute_error=0.)
    for i,r in enumerate(rows):
        columns=[j for j,n in enumerate(solver_names) if n==r['lipid_name']]
        assert r['evidence_index']==i and r['removed']==columns
        assert r['molecular_truth']==(r['lipid_name'] in truth) and r['reportable_truth']==(r['lipid_name'] in reportable)
        assert r['raw_solver_reported']==bool(columns and np.any(means[columns]>.001))
        w=xf[columns].sum(axis=0) if columns else np.zeros(xf.shape[1]);total=float(w.sum())
        assert total==ev['weight_sums'][i]
        if total:w/=total
        err=float(np.max(abs(w-ev['weights'][i])));maxima['weight_absolute_error']=max(maxima['weight_absolute_error'],err)
        local=bf@w;err=float(np.max(abs(local-ev['local'][i])));maxima['local_absolute_error']=max(maxima['local_absolute_error'],err)
    assert maxima['weight_absolute_error']==0 and maxima['local_absolute_error']<=1e-12
    np.testing.assert_allclose(ev['global_b'],bf.mean(axis=1),rtol=0,atol=1e-14)
    result=dict(status='PASS',case=key,artifact_hashes=len(done['files']),normal_finite_training=True,
        all_same_name_unthresholded_weights_reconstructed=True,reported_membership_and_all_truth_denominators_verified=True,
        all_local_observations_independently_reconstructed=True,**maxima,
        outcome_analysis='DEFERRED_UNTIL_BOTH_CALIBRATION_SEALS')
    write(case/'independent_evidence_review.json',result);print(json.dumps(result),flush=True)


def full_review(root):
    import numpy as np
    from review_ce_uncertainty_identity_pilot import proof_bounds
    manifest=read(root/'output_manifest.json');hashes(root,manifest)
    design=read(root/'design.json');assert sha(root/'design.json')==read(root/'design_seal.json')['design_sha256']
    assert design['scientific']['contract']['gamma_num']==G
    evseal=read(root/'evidence_seal.json');assert sha(root/'spatial_novelty.json')==evseal['novelty_sha256']
    assert read(root/'spatial_novelty.json')['status']=='PASS'
    for key,h in evseal['cases'].items():hashes(root/'cases'/key,h)
    with np.load(root/'uncertainty/component_fractions.npz') as z:fractions=z['fractions']
    components=read(root/'uncertainty/components.json');summary=read(root/'summary.json');proof_count=0
    stats={};reviewed_cases={};max_gap=0.
    for method in ('global','local'):
        seal=read(root/f'{method}_calibration_seal.json');assert sha(root/f'{method}_calibration_seal.json')==summary['calibration_seals'][method]
        all_cases=[]
        for entry in design['scientific']['membership']:
            key=entry['key'];source=root/'cases'/key;folder=root/'scores'/method/key;c=read(folder/'scores.json')
            assert c['binding']['evidence_sha256']==sha(source/'evidence.npz')
            assert c['binding']['records_sha256']==sha(source/'molecular_records.json')
            assert c['binding']['input_evidence_seal_sha256']==sha(root/'evidence_seal.json')
            original=read(source/'molecular_records.json');by={r['lipid_name']:r for r in original}
            assert c['truth_count']==sum(r['molecular_truth'] for r in original)==125
            assert c['reportable_truth_count']==sum(r['reportable_truth'] for r in original)==125
            assert {r['lipid_name'] for r in c['records']}=={r['lipid_name'] for r in original if r['raw_solver_reported']}
            assert c['key']==key and c['role']==entry['role'] and c['kind']==entry['kind']
            hashes(folder,c['proof_files'])
            with np.load(source/'evidence.npz') as z:ev={k:z[k] for k in z.files}
            for r in c['records']:
                for k,value in by[r['lipid_name']].items():assert r[k]==value
                if r.get('input_status')=='NO_SPATIAL_EVIDENCE':
                    assert method=='local' and ev['weight_sums'][r['evidence_index']]==0;continue
                if 'proof_file' not in r:
                    assert r.get('numerical_error');continue
                b=ev['local'][r['evidence_index']] if method=='local' else ev['global_b']
                with np.load(folder/r['proof_file']) as z:
                    for label,removed in [('full',[]),('deleted',r['removed'])]:
                        if label not in r:
                            assert r.get('numerical_error');continue
                        low,high=proof_bounds(ev['A'],fractions,components,ev['kept'],b,z[label+'_primal'],z[label+'_dual'],removed)
                        shortcut=label=='deleted' and r[label]['proof']=='FULL_ZERO_IDENTITY_FEASIBLE_WITNESS'
                        assert abs(high-r[label]['upper'])<=1e-9
                        assert r[label]['lower']<=low+1e-9 if shortcut else abs(low-r[label]['lower'])<=1e-9
                        gap=high-r[label]['lower'];assert gap<=1e-6+1e-9;max_gap=max(max_gap,gap);proof_count+=1
            classified=read(folder/'classified_records.json')
            assert len(classified)==len(c['records'])
            for r,s in zip(c['records'],classified):assert s==dict(r,status=state(r,seal['epsilon']))
            for k,value in accounting([c],seal['epsilon']).items():assert summary['methods'][method]['by_case'][key][k]==value
            all_cases.append(c);print('INDEPENDENT_V2_PROOF_PASS',method,key,flush=True)
        cal=[c for c in all_cases if c['role']=='CAL'];evaluation=[c for c in all_cases if c['role']=='EVAL']
        assert set(seal['CAL_source_hashes'])=={c['key'] for c in cal}
        for key,h in seal['CAL_source_hashes'].items():assert sha(root/'scores'/method/key/'scores.json')==h
        events={0.,1.+G}
        for c in cal:
            for r in c['records']:
                if r.get('numerical_error') or 'deleted' not in r:continue
                for t in (r['full']['upper']+G,r['deleted']['lower']-G):
                    if t>=0:events.update([float(t),float(np.nextafter(t,-np.inf)),float(np.nextafter(t,np.inf))])
        choices=[dict(epsilon=e,**accounting(cal,e)) for e in sorted(events) if e>=0]
        eligible=[c for c in choices if c['retained_count'] and c['FDP']<=.01]
        best=max(eligible,key=lambda c:(c['TP'],-c['FP'],c['epsilon'])) if eligible else dict(epsilon=1.+G)
        assert best['epsilon']==seal['epsilon']
        observed=accounting(evaluation,seal['epsilon']);bykind={k:accounting([c for c in evaluation if c['kind']==k],seal['epsilon']) for k in KINDS}
        for key,value in observed.items():assert summary['methods'][method]['aggregate'][key]==value
        for kind,values in bykind.items():
            for key,value in values.items():assert summary['methods'][method]['by_challenge'][kind][key]==value
        go=bool(observed['retained_count'] and observed['FDP']<=.01 and observed['TP_retention'] is not None and observed['TP_retention']>=.4 and
                all(v['retained_count'] and v['TP_retention'] is not None and v['TP_retention']>=.4 for v in bykind.values()))
        assert go==summary['methods'][method]['go']
        assert summary['methods'][method]['strong_success']==bool(go and observed['TP_retention']>=.6)
        stats[method]=dict(aggregate=observed,by_challenge=bykind,go=go)
    assert summary['main_go']==stats['local']['go']
    result=dict(status='PASS',compact_hash_count=len(manifest),proofs_checked=proof_count,max_proof_gap=max_gap,
        both_CAL_seals_recomputed_without_EVAL=True,all_statuses_and_denominators_verified=True,methods=stats,
        limitation='Developmental new spatial realizations; fixed mismatch library and repeated identities, no formal population-risk guarantee.')
    write(root/'independent_review.json',result);print('INDEPENDENT_V2_REVIEW_PASS',proof_count,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('output',type=Path);p.add_argument('--case')
    p.add_argument('--old-v1',type=Path,default=Path('/root/v58_jobs/ce_uncertainty_identity_pilot_run04'))
    a=p.parse_args()
    evidence_review(a.output,a.case,a.old_v1) if a.case else full_review(a.output)
