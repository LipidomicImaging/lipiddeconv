"""Cache-only real CE29 application review and final storage receipt."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'.venv/real_ce29_packages'))
import numpy as np


def read(p):
    return json.loads(p.read_text(encoding='utf-8'))


def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for v in iter(lambda:f.read(1024*1024),b''):h.update(v)
    return h.hexdigest()


def main(out):
    report=read(out/'report.json');original=read(out/'input.json')
    assert report['status']=='COMPLETE_REVIEWED' and not report['truth_available']
    for key in ('actual_FDR','TP','FP','FN','recall'):assert report[key] is None
    assert sha(out/'input.json')==report['input_sha256']==read(out/'input_seal.json')['sha256']
    assert sha(out/'model.json')==report['model_sha256']
    assert sha(out/'thresholds.json')==report['thresholds_sha256']
    for name,value in original['files'].items():assert sha(out/name)==value,name
    for item in original['source_files'].values():assert sha(Path(item['path']))==item['sha256']
    snapshot=read(out/'artifact_hashes.json');mutable={}
    for name,value in snapshot.items():
        actual=sha(out/name)
        if name=='run.log' and value!=actual:
            mutable[name]={'snapshot_sha256':value,'final_sha256':actual,'reason':'The runner prints its terminal receipt after writing the manifest; the process log is a mutable execution transcript, not a scientific input or numeric artifact.'}
        else:assert actual==value,name
    rows=read(out/'molecular_records.json');initial=read(out/'molecular_records_input.json');model=read(out/'model.json')
    manifest=read(out/'block_manifest.json');features=read(out/'physical_features.json');by_name={f['molecular_name']:f for f in features}
    assert len(rows)==len({r['lipid_name'] for r in rows})==377
    assert [r['lipid_name'] for r in rows]==manifest['molecular_names']
    assert all(r['molecular_truth'] is None for r in rows)
    assert sum(r['raw_solver_reported'] for r in rows)==original['original_reported_molecular_identities']==76
    with np.load(out/'prepared.npz',allow_pickle=False) as z:A,b=z['A'],z['b']
    blocks=[]
    for k in range(34):
        assert read(out/f'blocks/block_{k:03d}_review.json')['status']=='PASS'
        with np.load(out/f'blocks/block_{k:03d}.npz',allow_pickle=False) as z:blocks.append({n:z[n] for n in z.files})
    for g,(row,source) in enumerate(zip(rows,initial)):
        for key in ('lipid_name','X_hat','raw_solver_reported','reported_candidate_indices'):
            assert row[key]==source[key]
        feature=by_name[row['lipid_name']];delta=[];positive=0;count=0
        for block,arrays in zip(manifest['blocks'],blocks):
            full=float(arrays['full_held_loss']);deleted=float(arrays['deleted_held_loss'][g]);gain=deleted-full;delta.append(gain)
            held=block['channel_indices'];training=sorted(set(range(len(b)))-set(held))
            aliases=row['candidate_indices']
            supported=bool(np.any(A[np.ix_(held,aliases)]>0) and np.any(A[np.ix_(training,aliases)]>0))
            tau=128*np.finfo(float).eps*max(A.shape)*max(deleted,full,float(b[held]@b[held]))
            count+=supported;positive+=bool(supported and gain>tau)
        S=math.fsum(delta)/float(b@b);C=positive/count if count else 0.
        assert math.isclose(S,feature['predictive_gain'],rel_tol=1e-11,abs_tol=1e-14)
        assert C==feature['positive_block_fraction']==row['positive_block_fraction']
        if row['raw_solver_reported']:
            a=(math.log10(max(row['X_hat'],1e-12))-model['base']['mean'][0])/model['base']['scale'][0]
            r=(math.log10(max(row['rho_zero'],1e-24))-model['base']['mean'][1])/model['base']['scale'][1]
            s=(feature['predictive_gain']-model['auxiliary']['mean'][0])/model['auxiliary']['scale'][0]
            c=(C-model['auxiliary']['mean'][1])/model['auxiliary']['scale'][1]
            logit=math.fsum([model['intercept']]+[x*y for x,y in zip((a,a*a,r,s,c),model['coef'])])
            pred=1/(1+math.exp(-logit)) if logit>=0 else math.exp(logit)/(1+math.exp(logit))
            assert abs(pred-row['joint_score'])<=1e-12
        else:assert row['joint_score'] is None
    for key,entry in report['thresholds'].items():
        selected=[r['lipid_name'] for r in rows if r['raw_solver_reported'] and r['joint_score']>=entry['threshold']]
        assert selected==entry['names'] and len(selected)==entry['retained'] and entry['removed']==76-len(selected)
        assert all(r['retained_by_'+key]==(r['lipid_name'] in selected) for r in rows)
    with (out/'molecular_screening.csv').open(encoding='utf-8-sig',newline='') as f:csv_rows=list(csv.DictReader(f))
    assert len(csv_rows)==377 and [r['lipid_name'] for r in csv_rows]==[r['lipid_name'] for r in rows]
    review={'status':'PASS','source_hashes_and_scientific_artifacts_verified':True,'original_ISTA_abundances_preserved':True,
            'independent_features_scores_and_counts_verified':True,'actual_FDR_and_recall_remain_unknown':True,
            'mutable_log_receipt':mutable,'no_solver_or_training_rerun':True,'report_sha256':sha(out/'report.json')}
    (out/'independent_final_review.json').write_text(json.dumps(review,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'status':'PASS','reported':76,'retained':{k:v['retained'] for k,v in report['thresholds'].items()}}))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,default=ROOT/'results/real_ce29_joint_screening_v2');main(p.parse_args().output)
