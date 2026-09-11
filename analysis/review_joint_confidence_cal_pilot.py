"""Independent grouped CAL accounting and saved-model prediction check."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())


def probability(r,m):
    z=[(math.log10(max(r['X_hat'],1e-12))-m['mean'][0])/m['scale'][0],
       (math.log10(max(r['rho_zero'],1e-24))-m['mean'][1])/m['scale'][1]]
    features=[*z,z[0]**2,z[0]*z[1],z[1]**2,float(r['rho_zero']==0)]
    x=m['intercept'][0]+sum(c*v for c,v in zip(m['coef'][0],features))
    return 1/(1+math.exp(-x)) if x>=0 else math.exp(x)/(1+math.exp(x))


def counts(rows,key):
    truth=sum(r['molecular_truth'] for r in rows);reportable=sum(r['reportable_truth'] for r in rows)
    raw=[r for r in rows if r['raw_solver_reported']];retained=[r for r in rows if r[key]]
    tp=sum(r['molecular_truth'] for r in retained);raw_tp=sum(r['molecular_truth'] for r in raw)
    return dict(TP=tp,FP=len(retained)-tp,FN=truth-tp,raw_solver_TP=raw_tp,raw_solver_FP=len(raw)-raw_tp,
      raw_solver_FN=truth-raw_tp,filter_induced_true_loss=raw_tp-tp,
      FDR=(len(retained)-tp)/len(retained) if retained else None,all_truth_recall=tp/truth,
      reportable_truth_recall=sum(r['reportable_truth'] for r in retained)/reportable,
      TP_retention=tp/raw_tp if raw_tp else None,coverage=len(retained)/len(raw) if raw else None)


def main(out):
    provenance=read(out/'provenance.json')
    for name,h in provenance['files'].items():assert sha(out/name)==h,name
    assert sha(out/'executed_runner.py')==provenance['script_sha256']
    models=read(out/'models.json');rows=read(out/'out_of_group_records.json');contract=read(out/'contract.json')
    source=Path(__file__).resolve().parents[1]/'results/v58_final_snapshot_20260911'
    originals={}
    for name,h in contract['sources'].items():
        assert sha(source/name)==h
        with (source/name).open(newline='') as f:
            for r in csv.DictReader(f):originals[r['dataset_id'],r['lipid_name']]=r
    assert len(rows)==len({(r['dataset_id'],r['lipid_name']) for r in rows})==len(originals)
    max_prediction_error=0.
    for m in models:
        a,b,c=map(set,(m['train_groups'],m['calibration_groups'],m['test_groups']))
        assert not a&b and not a&c and not b&c
        assert sha(out/f"fold_{m['fold']}_seal.json")==provenance['fold_seal_hashes'][m['fold']]
        calibration=[]
        for old in originals.values():
            if old['lipid_name'] in b and old['raw_solver_reported']=='True':
                r=dict(X_hat=float(old['X_hat']),rho_zero=float(old['rho_zero']),molecular_truth=old['molecular_truth']=='True')
                r['joint_probability']=probability(r,m);calibration.append(r)
        seal=read(out/f"fold_{m['fold']}_seal.json")
        for key,levels in seal['cutoffs'].items():
            for target,cut in levels.items():
                groups={}
                for r in calibration:groups.setdefault(r[key],[]).append(r)
                tp=fp=0;chosen=None
                for value in sorted(groups,reverse=True):
                    tp+=sum(r['molecular_truth'] for r in groups[value]);fp+=sum(not r['molecular_truth'] for r in groups[value])
                    if fp/(tp+fp)<=float(target):chosen=value
                assert (chosen is None)==(cut is None)
                if chosen is not None:assert math.isclose(chosen,cut,abs_tol=1e-12,rel_tol=1e-12)
    for r in rows:
        old=originals[r['dataset_id'],r['lipid_name']]
        for k in ('molecular_truth','reportable_truth','raw_solver_reported'):assert r[k]==(old[k]=='True')
        assert r['X_hat']==float(old['X_hat']) and r['severity']==old['severity']
        if r['rho_zero'] is not None:assert r['rho_zero']==float(old['rho_zero'])
        fold=int.from_bytes(hashlib.sha256(('joint_confidence_CAL_v1|'+r['lipid_name']).encode()).digest()[:8],'little')%5
        assert r['fold']==fold and r['lipid_name'] in models[fold]['test_groups']
        if r['raw_solver_reported']:
            delta=abs(probability(r,models[fold])-r['joint_probability']);max_prediction_error=max(max_prediction_error,delta)
            assert delta<1e-12
        seal=read(out/f'fold_{fold}_seal.json')
        for key,levels in seal['cutoffs'].items():
            for target,t in levels.items():
                assert r[f'{key}_{target}_retained']==bool(r['raw_solver_reported'] and t is not None and r[key]>=t)
    summary=dict(status='REVIEWED_GROUPED_CAL_SCREEN',source_hashes_and_membership_verified=True,
      identity_disjoint_per_rotation=True,max_saved_model_prediction_error=max_prediction_error,results={},per_fold={})
    for severity in ('MILD','MODERATE'):
        subset=[r for r in rows if r['severity']==severity];assert sum(r['molecular_truth'] for r in subset)==1750
        summary['results'][severity]={}
        for score in ('rho_zero','X_hat','joint_probability'):
            for target in ('0.05','0.01'):
                key=f'{score}_{target}_retained'
                summary['results'][severity][score+'_'+target]=counts(subset,key)
        summary['per_fold'][severity]={str(f):counts([r for r in subset if r['fold']==f],'joint_probability_0.05_retained') for f in range(5)}
    success=True
    for severity,values in summary['results'].items():
        joint=values['joint_probability_0.05'];best=max(values[s+'_0.05']['all_truth_recall'] for s in ('rho_zero','X_hat'))
        success &= joint['FDR'] is not None and joint['FDR']<=.05 and joint['all_truth_recall']>=best+.1-1e-12
    summary['decision']='PROMISING_REQUIRES_NEW_VALIDATION' if success else 'STOP_THIS_VERSION_PREDEFINED_CRITERIA_NOT_MET'
    summary['limitation']='Identity-disjoint internal CAL validation; shared cubes, prior CAL exposure and dependent folds remain. Not original HOLD or population FDR validation.'
    (out/'review.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('output',type=Path);main(p.parse_args().output)
