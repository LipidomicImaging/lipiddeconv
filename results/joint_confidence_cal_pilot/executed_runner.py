"""Grouped CAL-only train/calibrate/evaluate screen using rho and abundance."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import warnings
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''


def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')


def main(a):
    import numpy as np
    import sklearn
    from sklearn.linear_model import LogisticRegression
    from sklearn.exceptions import ConvergenceWarning
    a.output.mkdir(parents=True,exist_ok=False)
    contract=read(a.contract);write(a.output/'contract.json',contract)
    records=[]
    for name,h in contract['sources'].items():
        p=a.source/name;assert sha(p)==h,name
        with p.open(newline='') as f:rr=list(csv.DictReader(f))
        for r in rr:
            assert r['split']=='CAL'
            for k in ('raw_solver_reported','molecular_truth','reportable_truth'):r[k]=r[k]=='True'
            r['X_hat']=float(r['X_hat']);r['rho_zero']=float(r['rho_zero']) if r['rho_zero'] else None
            r['fold']=int.from_bytes(hashlib.sha256(('joint_confidence_CAL_v1|'+r['lipid_name']).encode()).digest()[:8],'little')%5
        records.extend(rr)
    assert len({(r['dataset_id'],r['lipid_name']) for r in records})==len(records)
    def features(rows,mean=None,scale=None):
        # Explicit observable-only whitelist: neither identities nor truth/severity enter features.
        x=np.array([[np.log10(max(r['X_hat'],1e-12)),np.log10(max(r['rho_zero'],1e-24))] for r in rows])
        if mean is None:mean=x.mean(axis=0);scale=x.std(axis=0);scale=np.maximum(scale,1e-12)
        z=(x-mean)/scale
        return np.column_stack([z[:,0],z[:,1],z[:,0]**2,z[:,0]*z[:,1],z[:,1]**2,[float(r['rho_zero']==0) for r in rows]]),mean,scale
    def threshold(rows,key,target):
        groups={}
        for r in rows:groups.setdefault(r[key],[]).append(r)
        tp=fp=0;best=None
        for value in sorted(groups,reverse=True):
            tp+=sum(r['molecular_truth'] for r in groups[value]);fp+=sum(not r['molecular_truth'] for r in groups[value])
            if fp/(tp+fp)<=target:best=value
        return best
    models=[];eval_rows=[];seals=[]
    for fold in range(5):
        cal=(fold+1)%5
        train=[r for r in records if r['fold'] not in (fold,cal) and r['raw_solver_reported']]
        calibration=[dict(r) for r in records if r['fold']==cal and r['raw_solver_reported']]
        test=[dict(r) for r in records if r['fold']==fold]
        groups=[{r['lipid_name'] for r in rs} for rs in (train,calibration,test)]
        assert not groups[0]&groups[1] and not groups[0]&groups[2] and not groups[1]&groups[2]
        X,mean,scale=features(train);y=np.array([r['molecular_truth'] for r in train])
        with warnings.catch_warnings():
            warnings.simplefilter('error',ConvergenceWarning)
            model=LogisticRegression(C=1.,solver='lbfgs',max_iter=2000,tol=1e-8).fit(X,y)
        model_record=dict(fold=fold,calibration_fold=cal,train_groups=sorted(groups[0]),calibration_groups=sorted(groups[1]),
                          test_groups=sorted(groups[2]),mean=mean.tolist(),scale=scale.tolist(),coef=model.coef_.tolist(),
                          intercept=model.intercept_.tolist(),n_iter=model.n_iter_.tolist())
        models.append(model_record)
        prediction=model.predict_proba(features(calibration,mean,scale)[0])[:,1]
        for r,p in zip(calibration,prediction):r['joint_probability']=float(p)
        cuts={key:{str(t):threshold(calibration,key,t) for t in (.05,.01)} for key in ('rho_zero','X_hat','joint_probability')}
        # Persist the cutoffs before computing/evaluating this fold's test predictions.
        seal=dict(fold=fold,cutoffs=cuts,model=model_record,selection='maximum retention with tied scores at pooled calibration empirical risk')
        write(a.output/f'fold_{fold}_seal.json',seal);seals.append(sha(a.output/f'fold_{fold}_seal.json'))
        reported=[r for r in test if r['raw_solver_reported']]
        prediction=model.predict_proba(features(reported,mean,scale)[0])[:,1]
        for r,p in zip(reported,prediction):r['joint_probability']=float(p)
        for r in test:
            rr={k:r[k] for k in ('dataset_id','severity','lipid_name','molecular_truth','reportable_truth','raw_solver_reported','X_hat','rho_zero','fold')}
            rr['joint_probability']=r.get('joint_probability')
            for key,levels in cuts.items():
                for t,value in levels.items():rr[f'{key}_{t}_retained']=bool(r['raw_solver_reported'] and value is not None and r[key]>=value)
            eval_rows.append(rr)
    write(a.output/'models.json',models)
    write(a.output/'out_of_group_records.json',eval_rows)
    write(a.output/'provenance.json',dict(status='COMPLETE_AWAITING_REVIEW',script_sha256=sha(Path(__file__)),
        sklearn=sklearn.__version__,numpy=np.__version__,original_HOLD_used=False,identity_disjoint_per_fold=True,
        feature_names=['standardized_log_X','standardized_log_rho','log_X_squared','log_X_log_rho','log_rho_squared','rho_is_zero'],
        fold_seal_hashes=seals,source_hashes=contract['sources'],
        files={p.name:sha(p) for p in a.output.iterdir() if p.is_file()}))
    print('COMPLETE_AWAITING_REVIEW',len(eval_rows))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--contract',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);main(p.parse_args())
