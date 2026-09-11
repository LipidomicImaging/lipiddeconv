"""Bounded secondary CAL test: equally flexible spectra per individual identity."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback

for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())
def write(p,v):p.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')


def references(A,parent,normalize):
    import numpy as np
    variants=[A.copy(),A.copy(),A.copy()]
    seeds=[]
    for j in range(A.shape[1]):
        seed=int.from_bytes(hashlib.sha256(f'reference_flexibility_CAL_v1/reference/{j}'.encode()).digest()[:8],'little')
        rng=np.random.default_rng(seed);nz=np.flatnonzero((~parent)&(A[:,j]>0))
        z=rng.standard_normal(len(nz));u=rng.random(len(nz));p=rng.uniform(-1,1)
        strongest=nz[np.argmax(A[nz,j])] if len(nz) else -1
        for k,(sigma,drop,dev) in enumerate(((.1,.05,.1),(.2,.1,.2)),1):
            col=A[:,j].copy();col[nz]=(col[nz].astype(np.float64)*np.exp(sigma*z)).astype(np.float32)
            col[nz[(u<drop)&(nz!=strongest)]]=0;col[parent]*=np.float32(1+dev*p)
            variants[k][:,j]=normalize(col[:,None])[:,0]
        seeds.append(seed)
    return np.concatenate(variants,axis=1),seeds


def main(args):
    import numpy as np
    import scipy
    from scipy.optimize import nnls
    sys.path.insert(0,str(args.root/'analysis'));sys.path.insert(0,str(args.root/'src'))
    import run_v58_spectral_library_mismatch_fdr_recalibration as v
    v.dependencies();args.output.mkdir(parents=True,exist_ok=False)
    contract=dict(version='reference_flexibility_CAL_v1',dataset='CAL_R1_K125',domain='MILD',
      primary_universe='previously completed MILD pixelwise NNLS reported molecular identities',
      references_per_candidate=3,reference_parameters=[[0,0,0],[.1,.05,.1],[.2,.1,.2]],
      reference_source='A_solver only, new independent seed namespace; no target spectra/latents/truth',
      deletion='all three variants of every column belonging to the individual molecular identity',
      fit='float64 scipy NNLS maxiter11730, shared full fit, no learned GPU training',
      score='max(0, deleted SSE-full SSE)/observed energy; values <=1e-12 set to zero for this NEW score only',
      numerical_checks='finite, nonnegative, normalized KKT <=1e-9, normalized negative gain no less than -1e-10',
      cost_gate='full fit <=60s and first8 index-ordered deletions median <=10s; otherwise STOP_COST',
      comparison='same-universe original rho and X_hat; CAL retrospective tied-score curves only',
      continuation='at empirical5% FDR improve all-truth recall by >=10pp over BOTH baselines; otherwise stop this version',
      limitations=['Known synthetic perturbation family, new reference draws; not measured real spectral uncertainty.',
                   'Single already examined CAL case; no independent FDR/generalization guarantee.',
                   'No inter-identity group rescue, no original scientific definition change.'])
    write(args.output/'contract.json',contract)
    va=v.parse_args(['--dataset','MILD__CAL_R1_K125','--output-dir',str(args.root/'results/v58_spectral_library_mismatch_fdr_recalibration_compact_recovery'),
                     '--v57-output',str(args.root/'results/v57_spectral_spatial_identity_confidence_benchmark'),'--asset-root',str(args.asset_root)])
    parent,_,_=v.parent_provenance(va);context=v.load_context(va,parent)
    targets,_,_=v.build_targets(context);design=read(va.output_dir/'design.json');audit=v.checked_audit(va,design)
    case,row=v.mismatch_case(context,parent,targets,'MILD__CAL_R1_K125');row=v.v57.canonical(row)
    for k,val in audit['datasets']['MILD__CAL_R1_K125'].items():
        if k in ('forward_mismatch_residual','sentinel_gate_residual'):assert abs(row[k]-val)<=1e-12
        else:assert row[k]==val,k
    A=context['A_solver'];D,seeds=references(A,context['parent_mask'],v.v54.v52.normalize_like_get_A_matrix)
    again,_=references(A,context['parent_mask'],v.v54.v52.normalize_like_get_A_matrix)
    assert np.array_equal(D,again) and np.array_equal(D[:,:391],A)
    assert not np.array_equal(D[:,391:782],targets['MILD'])
    assert np.isfinite(D).all() and (D>=0).all() and np.allclose(np.linalg.norm(D,axis=0),1,atol=2e-6)
    np.savez_compressed(args.output/'reference_dictionary.npz',D=D)
    write(args.output/'reference_manifest.json',dict(seeds=seeds,namespace='reference_flexibility_CAL_v1/reference',
          dictionary_sha256=v.array_sha(D),A_solver_sha256=v.array_sha(A),scientific_case_audit=row))
    source=args.nnls_output/'molecular_units.json';source_report=read(args.nnls_output/'result.json')
    source_provenance=read(args.nnls_output/'provenance.json')
    assert sha(source)==source_provenance['files']['molecular_units.json']
    assert read(args.nnls_output/'contract.json')['scientific_input_audit']==row
    records=read(source);assert sum(r['molecular_truth'] for r in records)==125
    names=[str(x) for x in context['metadata']['lipid_name']]
    records.sort(key=lambda r:names.index(r['lipid_name']))
    b=case['B_sim'][:,case['foreground_mask']].mean(axis=1,dtype=np.float64);energy=float(b@b)+1e-12
    def fit(M,y):
        started=time.monotonic();x,_=nnls(np.asarray(M,dtype=np.float64),y,maxiter=11730)
        residual=M@x-y;g=M.T@residual
        dual=float(np.maximum(-g,0).max())/max(float(np.linalg.norm(y)),1e-30)
        comp=float(np.abs(x*g).max())/(float(y@y)+1e-12)
        assert np.isfinite(x).all() and (x>=0).all() and dual<=1e-9 and comp<=1e-9
        return x,float(residual@residual),dict(seconds=time.monotonic()-started,dual=dual,complementarity=comp)
    # Reduction, membership deletion and zero-coefficient feasible-bound checks.
    tiny=np.array([[1.,0.],[0.,1.],[1.,1.]])
    tx,loss,_=fit(tiny,np.array([1.,0.,1.]));assert np.allclose(tx,[1,0]) and loss<1e-20
    bundled=np.concatenate([tiny,tiny],axis=1);keep=np.array([False,True,False,True])
    _,removed,_=fit(bundled[:,keep],np.array([1.,0.,1.]));assert removed>0
    base,base_loss,_=fit(A.astype(np.float64),b)
    singleton=next(r for r in records if names.count(r['lipid_name'])==1)
    j=names.index(singleton['lipid_name']);keep=np.arange(391)!=j
    _,bl,_=fit(A[:,keep].astype(np.float64),b)
    parity=max(0.,bl-base_loss)/energy
    assert abs(parity-singleton['rho_zero'])<=1e-12
    write(args.output/'self_test.json',dict(status='PASS',deterministic_dictionary=True,all_identity_variants_deleted=True,
          one_reference_singleton_absolute_difference=abs(parity-singleton['rho_zero']),tiny_zero_and_exact_fit=True))
    started=time.monotonic();D=D.astype(np.float64);full,full_loss,full_diag=fit(D,b)
    if full_diag['seconds']>60:
        write(args.output/'status.json',dict(status='STOP_COST',full=full_diag));return
    outputs=[];timings=[]
    for i,r in enumerate(records):
        members=[j for j,n in enumerate(names) if n==r['lipid_name']]
        keep=~np.isin(np.arange(D.shape[1])%391,members)
        assert (~keep).sum()==3*len(members)
        x,loss,diag=fit(D[:,keep],b);gain=(loss-full_loss)/energy
        assert gain>=-1e-10
        # Shared full fit must dominate the feasible solution obtained by padding the deletion fit.
        if np.all(full[~keep]==0):assert gain<=1e-10
        score=max(0.,gain);score=0. if score<=1e-12 else score
        outputs.append(dict(**r,flexible_identity_score=score,unfloored_signed_gain=gain,full_SSE=full_loss,deleted_SSE=loss,**diag))
        timings.append(diag['seconds'])
        write(args.output/'status.json',dict(status='RUNNING',completed=i+1,planned=len(records)))
        if i==7 and float(np.median(timings))>10:
            write(args.output/'partial_scores.json',outputs);write(args.output/'status.json',dict(status='STOP_COST',median_deletion_seconds=float(np.median(timings))));return
        if (i+1)%25==0:print(i+1,'/',len(records),flush=True)
    write(args.output/'scores.json',outputs)
    assert sha(source)==source_provenance['files']['molecular_units.json']
    write(args.output/'provenance.json',dict(status='COMPLETE_AWAITING_REVIEW',seconds=time.monotonic()-started,
       source=str(source),source_sha256=sha(source),script_sha256=sha(Path(__file__)),scipy=scipy.__version__,numpy=np.__version__,
       full_fit=full_diag,full_SSE=full_loss,base_SSE=base_loss,truth_count=source_report['all_truth_molecular_identity_count'],
       files={p.name:sha(p) for p in args.output.iterdir() if p.is_file() and p.name!='status.json'}))
    write(args.output/'status.json',dict(status='COMPLETE_AWAITING_REVIEW'))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--asset-root',type=Path,required=True)
    p.add_argument('--nnls-output',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    try:main(a)
    except BaseException:
        if a.output.exists():write(a.output/'failure.json',dict(error=traceback.format_exc()))
        raise
