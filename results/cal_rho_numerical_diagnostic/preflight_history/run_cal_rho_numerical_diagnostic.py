"""Eight prespecified CAL examples: numerical sensitivity and mismatch attribution.

This diagnostic deliberately recomputes bounded CPU fits, not production results.
"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
import time

for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def read(p):return json.loads(p.read_text())


def write(p,d):p.write_text(json.dumps(d,indent=2,allow_nan=False)+'\n')


def main(args):
    import numpy as np
    import scipy
    from scipy.optimize import nnls,lsq_linear
    sys.path.insert(0,str(args.root/'analysis'));sys.path.insert(0,str(args.root/'src'))
    import run_v58_spectral_library_mismatch_fdr_recalibration as v
    from rho_zero import EPSILON_Q
    v.dependencies()
    plan=read(args.selection);assert plan['dataset']=='CAL_R1_K125' and len(plan['selected'])<=8
    args.output.mkdir(parents=True,exist_ok=False)
    started=time.monotonic()
    va=v.parse_args(['--dataset','MILD__CAL_R1_K125','--output-dir',str(args.root/'results/v58_spectral_library_mismatch_fdr_recalibration_compact_recovery'),
                     '--v57-output',str(args.root/'results/v57_spectral_spatial_identity_confidence_benchmark'),
                     '--asset-root',str(args.asset_root)])
    parent,_,_=v.parent_provenance(va);context=v.load_context(va,parent)
    targets,_,_=v.build_targets(context);design=read(va.output_dir/'design.json')
    case,bound=v.checked_case(va,context,parent,targets,design,'MILD__CAL_R1_K125')
    clean=v.v57.construct_case(context,parent['scientific'],plan['dataset'])
    assert v.array_sha(clean['B_sim'])==read(va.v57_output/'clean'/plan['dataset']/'runtime_contract.json')['B_sha256']
    A=context['A_solver'].astype(np.float64);mask=case['foreground_mask']
    signals={'CLEAN':clean['B_sim'][:,mask].mean(axis=1,dtype=np.float64),'MILD':case['B_sim'][:,mask].mean(axis=1,dtype=np.float64)}
    mismatch=signals['MILD']-signals['CLEAN']
    n_top=int(np.ceil(len(mismatch)*plan['top_mismatch_channel_fraction']))
    top=set(np.argsort(np.abs(mismatch),kind='stable')[-n_top:].tolist())
    mz=np.load(context['paths']['channel_axis']).reshape(-1)
    methods=plan['methods'];fit_rows=[];channels=[];full={}

    def fit(matrix,b,method):
        start=time.monotonic()
        if method=='scipy_lsq_linear_bvls_tol_1e-12':
            result=lsq_linear(matrix,b,bounds=(0,np.inf),method='bvls',tol=1e-12,max_iter=plan['maxiter'])
            assert result.success,result.message
            x=result.x;residual=float(np.linalg.norm(matrix@x-b))
        else:
            order=np.arange(matrix.shape[1])
            if method=='scipy_nnls_reverse_order':order=order[::-1]
            ordered,residual=nnls(matrix[:,order],b,maxiter=plan['maxiter']);x=np.empty_like(ordered);x[order]=ordered
        residual_vector=matrix@x-b;g=matrix.T@residual_vector
        assert np.isfinite(x).all() and np.min(x)>=0 and np.isfinite(residual)
        diagnostics=dict(seconds=time.monotonic()-start,returned_residual=float(residual),
            direct_q=float(residual_vector@residual_vector),dual_violation=float(np.maximum(-g,0).max()),
            complementarity=float(np.max(np.abs(x*g))),gradient_scale=float(max(np.max(np.abs(matrix.T@b)),1e-30)))
        return x,residual_vector,diagnostics

    # Analytic zero and exactly identified deletion sanity checks, no production assets.
    tiny=np.array([[1.,0.],[0.,1.],[1.,1.]])
    for method in methods:
        x,_,_=fit(tiny,tiny@np.array([.5,2.]),method);assert np.allclose(x,[.5,2.],atol=1e-12)
        x,_,_=fit(tiny,np.zeros(3),method);assert np.all(x==0)
    for domain,b in signals.items():
        for method in methods:full[domain,method]=fit(A,b,method)
    write(args.output/'status.json',dict(status='RUNNING',selected=len(plan['selected']),completed=0))
    for idx,selected in enumerate(plan['selected']):
        j=selected['candidate_index'];keep=np.arange(A.shape[1])!=j
        for domain,b in signals.items():
            energy=float(b@b)+EPSILON_Q
            for method in methods:
                x,rf,fd=full[domain,method];xd,rd,dd=fit(A[:,keep],b,method)
                delta=dd['returned_residual']**2-fd['returned_residual']**2
                stable=float(np.sum((rd.astype(np.longdouble)-rf)*(rd.astype(np.longdouble)+rf),dtype=np.longdouble))
                correction=rd-rf
                cosine=float(correction@mismatch/max(np.linalg.norm(correction)*np.linalg.norm(mismatch),1e-300))
                gain=rd*rd-rf*rf
                top_gain=float(sum(gain[i] for i in top))
                positive_gain=float(np.maximum(gain,0).sum())
                row=dict(**selected,domain=domain,method=method,q_full=fd['returned_residual']**2,q_deleted=dd['returned_residual']**2,
                    delta_q_unclipped=delta,stable_delta_q_longdouble_accumulation=stable,rho=max(0.,delta/energy),
                    stable_rho_unclipped=stable/energy,signal_energy_plus_epsilon=energy,x_full_candidate=float(x[j]),
                    original_stored_rho=selected['clean_rho' if domain=='CLEAN' else 'mild_rho'],
                    full_dual_violation=fd['dual_violation'],deleted_dual_violation=dd['dual_violation'],
                    full_complementarity=fd['complementarity'],deleted_complementarity=dd['complementarity'],
                    full_gradient_scale=fd['gradient_scale'],deleted_gradient_scale=dd['gradient_scale'],
                    full_seconds=fd['seconds'],deleted_seconds=dd['seconds'],
                    correction_cosine_with_known_mismatch=cosine,top_mismatch_channels_signed_gain=top_gain,
                    positive_gain_fraction_on_top_mismatch_channels=float(sum(max(gain[i],0.) for i in top)/positive_gain) if positive_gain else None,
                    passes_original_cutoff=max(0.,delta/energy)>=plan['clean_threshold'],
                    passes_mild_cutoff=max(0.,delta/energy)>=plan['mild_threshold'])
                fit_rows.append(row)
                if method==methods[0]:
                    for i in range(len(b)):channels.append(dict(candidate_index=j,domain=domain,channel_index=i,mz=float(mz[i]),
                        known_mismatch=float(mismatch[i]),full_residual=float(rf[i]),deleted_residual=float(rd[i]),
                        deletion_loss_increase=float(gain[i]),in_top_mismatch_channels=i in top))
        write(args.output/'status.json',dict(status='RUNNING',selected=len(plan['selected']),completed=idx+1))
        print('DONE',idx+1,j,flush=True)
    for name,data in [('fit_diagnostics.csv',fit_rows),('channel_loss_decomposition.csv',channels)]:
        with (args.output/name).open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
    write(args.output/'provenance.json',dict(status='COMPLETE_AWAITING_REVIEW',selection_sha256=sha(args.selection),
        script_sha256=sha(Path(__file__)),numpy=np.__version__,scipy=scipy.__version__,A_solver_sha256=v.array_sha(context['A_solver']),
        B_clean_sha256=v.array_sha(clean['B_sim']),B_mild_sha256=bound['B_target_sha256'],
        clean_mean_sha256=v.array_sha(signals['CLEAN']),mild_mean_sha256=v.array_sha(signals['MILD']),
        fingerprint=design['design_fingerprint'],seconds=time.monotonic()-started,
        files={name:sha(args.output/name) for name in ('fit_diagnostics.csv','channel_loss_decomposition.csv')},
        no_HOLD_read=True,no_GPU_training=True,known_mismatch_is_not_measurement_noise=True,
        longdouble_is_residual_accumulation_only_not_high_precision_optimization=True))
    write(args.output/'status.json',dict(status='COMPLETE_AWAITING_REVIEW',selected=len(plan['selected']),completed=len(plan['selected'])))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--asset-root',type=Path,required=True)
    p.add_argument('--selection',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    main(p.parse_args())
