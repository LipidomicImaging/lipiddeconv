"""Bounded CAL-only channel-prediction feasibility test; no GPU or target spectra in scoring."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
import time

for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''

CONTRACT=dict(version='cal_channel_prediction_v1',dataset='CAL_R1_K125',domains=['CLEAN','MILD'],
    folds=3,mz_block_width=10.,training_buffer_Da=2.,parent_group='all production parent channels together',
    support_peak_fraction_of_fragment_max=.01,min_train_fragment_peaks=2,min_test_fragment_peaks=2,min_supported_folds=2,
    score='sum held-out deletion loss minus full loss / sum held-out observed energy plus 1e-12; signed, no clipping',
    fit='float64 unweighted NNLS, all original391 columns; maxiter3910; remove all columns of one molecular identity',
    candidates='all original reported molecular identities; original gate unchanged',
    calibration='NONE: retrospective CAL curves and ranking only; no operational cutoff',
    no_new_GPU_runs=True,alternative_reference_GPU_pilot='PAUSED')


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def read(p):return json.loads(p.read_text())


def write(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')


def fold_ids(mz,parent):
    groups=['PARENT' if is_parent else f'FRAG_{int(value//10)}' for value,is_parent in zip(mz,parent)]
    ids=[int.from_bytes(hashlib.sha256(('cal_channel_prediction_v1/'+g).encode()).digest()[:8],'little')%3 for g in groups]
    return groups,ids


def main(args):
    import numpy as np
    import scipy
    from scipy.optimize import nnls
    sys.path.insert(0,str(args.root/'analysis'));sys.path.insert(0,str(args.root/'src'))
    import run_v58_spectral_library_mismatch_fdr_recalibration as v
    v.dependencies()
    args.output.mkdir(parents=True,exist_ok=False)
    write(args.output/'contract.json',CONTRACT)
    started=time.monotonic()
    va=v.parse_args(['--dataset','MILD__CAL_R1_K125','--output-dir',str(args.root/'results/v58_spectral_library_mismatch_fdr_recalibration_compact_recovery'),
                     '--v57-output',str(args.root/'results/v57_spectral_spatial_identity_confidence_benchmark'),'--asset-root',str(args.asset_root)])
    parent,_,_=v.parent_provenance(va);context=v.load_context(va,parent)
    targets,_,_=v.build_targets(context);design=read(va.output_dir/'design.json');audit=v.checked_audit(va,design)
    mild,row=v.mismatch_case(context,parent,targets,'MILD__CAL_R1_K125');expected=audit['datasets']['MILD__CAL_R1_K125']
    row=v.v57.canonical(row)
    for k in expected:
        if k in ('forward_mismatch_residual','sentinel_gate_residual'):assert abs(row[k]-expected[k])<=1e-12
        else:assert row[k]==expected[k],k
    clean=v.v57.construct_case(context,parent['scientific'],CONTRACT['dataset'])
    assert v.array_sha(clean['B_sim'])==read(va.v57_output/'clean'/CONTRACT['dataset']/'runtime_contract.json')['B_sha256']
    A=np.asarray(context['A_solver'],dtype=np.float64);mz=np.load(context['paths']['channel_axis']).reshape(-1)
    assert v.array_sha(context['A_solver'])==design['scientific']['target_library_hashes']['A_solver_sha256']
    groups,ids=fold_ids(mz,context['parent_mask']);ids=np.array(ids)
    splits=[]
    for fold in range(3):
        test=ids==fold
        nearby=np.min(np.abs(mz[:,None]-mz[test][None,:]),axis=1)<=CONTRACT['training_buffer_Da']
        train=(~test)&(~nearby)
        assert train.any() and test.any() and not (train&test).any()
        assert np.min(np.abs(mz[train,None]-mz[test][None,:]))>2.
        splits.append((train,test))
    write(args.output/'folds.json',dict(groups=groups,fold_ids=ids.tolist(),mz=mz.tolist(),
        train_indices=[np.flatnonzero(a).tolist() for a,b in splits],test_indices=[np.flatnonzero(b).tolist() for a,b in splits],
        note='Blocked/buffered channels reduce local leakage; spectral or spatial statistical independence is not guaranteed.'))
    # Fit sees train rows only. Neither reference target spectra nor truth labels are arguments.
    def fit(train_A,train_b):return nnls(train_A,train_b,maxiter=3910)[0]
    def predictive_losses(train_A,train_b,test_A,test_b,keep):
        full=fit(train_A,train_b);deleted=fit(train_A[:,keep],train_b)
        return full,deleted,float(np.sum((test_A@full-test_b)**2)),float(np.sum((test_A[:,keep]@deleted-test_b)**2))
    # Test observations can change scoring, but cannot alter coefficients fitted on train observations.
    tiny=np.array([[1.,0.],[0.,1.],[1.,1.]])
    first=predictive_losses(tiny[:2],np.array([1.,2.]),tiny[2:],np.array([3.]),np.array([True,False]))
    second=predictive_losses(tiny[:2],np.array([1.,2.]),tiny[2:],np.array([30.]),np.array([True,False]))
    assert np.array_equal(first[0],second[0]) and np.array_equal(first[1],second[1])
    write(args.output/'self_test.json',dict(test_data_not_used_for_fitting='PASS',train_test_disjoint_and_buffered='PASS',identity_deletion_all_members=True))
    source_files={};output_rows=[];fold_rows=[];domain_results={}
    for domain,case in [('CLEAN',clean),('MILD',mild)]:
        path=va.v57_output/'calibration_records.csv' if domain=='CLEAN' else va.output_dir/'MILD'/CONTRACT['dataset']/'molecular_false_negative_records.csv'
        source_files[str(path)]=sha(path)
        table=v.v57.rows(path)
        if domain=='CLEAN':table=[r for r in table if r['dataset_id']==CONTRACT['dataset']]
        truth_count=sum(r['molecular_truth'] for r in table);assert truth_count==125
        reported=[r for r in table if r['raw_solver_reported']]
        b=case['B_sim'][:,case['foreground_mask']].mean(axis=1,dtype=np.float64)
        fulls=[]
        for train,test in splits:
            x=fit(A[train],b[train]);fulls.append((x,float(np.sum((A[test]@x-b[test])**2))))
        for index,r in enumerate(reported):
            members=r['candidate_indices']
            if isinstance(members,str):members=json.loads(members)
            keep=~np.isin(np.arange(A.shape[1]),members);assert sum(~keep)==len(members)
            gains=[];supported=0
            spectrum=A[:,members].max(axis=1);fragment_max=float(spectrum[~context['parent_mask']].max())
            peaks=(spectrum>=.01*fragment_max)&(spectrum>0)&(~context['parent_mask'])
            for fold,(train,test) in enumerate(splits):
                t=time.monotonic();deleted=fit(A[train][:,keep],b[train]);full,full_loss=fulls[fold]
                loss=float(np.sum((A[test][:,keep]@deleted-b[test])**2));gain=loss-full_loss;gains.append(gain)
                train_peaks=int(sum(peaks&train));test_peaks=int(sum(peaks&test))
                has_support=train_peaks>=2 and test_peaks>=2;supported+=int(has_support)
                fold_rows.append(dict(domain=domain,lipid_name=r['lipid_name'],fold=fold,full_test_loss=full_loss,
                    deleted_test_loss=loss,signed_gain=gain,train_fragment_peaks=train_peaks,test_fragment_peaks=test_peaks,
                    supported=has_support,seconds=time.monotonic()-t))
            evaluable=supported>=2
            score=sum(gains)/(float(b@b)+1e-12)
            output_rows.append(dict(domain=domain,lipid_name=r['lipid_name'],molecular_truth=r['molecular_truth'],
                reportable_truth=r['reportable_truth'],X_hat=float(r['X_hat']),rho_zero=float(r['rho_zero']),
                predictive_gain=score if evaluable else None,unrestricted_predictive_gain=score,
                evaluable=evaluable,supported_folds=supported,positive_gain_folds=sum(g>0 for g in gains),
                candidate_indices=members,raw_solver_reported=True))
            write(args.output/'status.json',dict(status='RUNNING',domain=domain,completed=index+1,planned=len(reported)))
            if (index+1)%25==0:print(domain,index+1,'/',len(reported),flush=True)
        domain_results[domain]=dict(all_truth_count=truth_count,reportable_truth_count=sum(r['reportable_truth'] for r in table),
                                    raw_reported=len(reported),raw_TP=sum(r['molecular_truth'] for r in reported))
    for name,data in [('scores.csv',output_rows),('fold_predictions.csv',fold_rows)]:
        with (args.output/name).open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
    write(args.output/'provenance.json',dict(status='COMPLETE_AWAITING_REVIEW',contract_sha256=sha(args.output/'contract.json'),
        script_sha256=sha(Path(__file__)),sources=source_files,domains=domain_results,seconds=time.monotonic()-started,
        numpy=np.__version__,scipy=scipy.__version__,B_clean_sha256=v.array_sha(clean['B_sim']),B_mild_sha256=row['B_target_sha256'],
        no_HOLD_outcomes_parsed=True,no_training=True,truth_and_A_target_not_used_in_scoring=True,
        limitation='Candidate reporting gate was computed using all channels. This is conditional post-hoc scoring, not fully independent validation; no calibrated risk guarantee.',
        files={n:sha(args.output/n) for n in ('scores.csv','fold_predictions.csv','folds.json','self_test.json')}))
    write(args.output/'status.json',dict(status='COMPLETE_AWAITING_REVIEW',domains=domain_results))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--asset-root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);main(p.parse_args())
