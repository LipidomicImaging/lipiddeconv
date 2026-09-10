"""CPU pixelwise full-library NNLS baseline on six frozen V57 K125 cases."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor

for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ[key]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''

DATASETS=[f'{split}_R{r}_K125' for split in ('CAL','HOLD') for r in range(1,4)]
_A=None


def initialize_worker(A):
    global _A
    _A=A


def solve_pixel(b):
    from scipy.optimize import nnls
    return nnls(_A,b,maxiter=3910)[0]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    temporary.replace(path)


def main(args):
    sys.path.insert(0,str(args.root/'analysis'))
    import numpy as np
    import scipy
    from scipy.optimize import nnls
    import run_v57_spectral_spatial_identity_confidence_benchmark as v57
    v54,v56=v57.v54,v57.v56
    parent=args.root/'results/v57_spectral_spatial_identity_confidence_benchmark'
    design=read(parent/'design.json');s=design['scientific']
    assert v57.fingerprint(s)==design['design_fingerprint']
    assert v57.source_implementations()==s['implementation_hashes']
    contract=dict(parent_fingerprint=design['design_fingerprint'],datasets=DATASETS,
                  method='scipy.optimize.nnls per foreground pixel; all391 columns; unweighted L2; no spatial penalty or learned spectral calibration',
                  maxiter=3910,precision='float64 solve, float32 stored X_hat; float64 foreground mean for reporting',
                  cpu_workers=args.workers,blas_threads_per_worker=1,
                  reporting_gate=0.001,background='require exactly zero input, return zero solution',
                  rho='unchanged v54.rho_values, W=I, mean observed foreground spectrum',
                  calibration='per solver CAL R1-R3 K125 maximum retention at empirical FDR5/FDR1, then HOLD R1-R3 K125',
                  scipy_version=scipy.__version__,numpy_version=np.__version__,script_sha256=digest(Path(__file__)),
                  limitations=['Matched inputs and reporting, different optimization objectives from production ISTA.',
                               'CLEAN exact-library performance may saturate; no outcome-driven harder-condition expansion.',
                               'Three mappings and repeated identities are not independent biological samples.'])
    fp=v57.fingerprint(contract)
    path=args.output/'design.json'
    if path.exists(): assert read(path)==dict(fingerprint=fp,contract=contract),'DESIGN_CHANGED'
    else: write(path,dict(fingerprint=fp,contract=contract))
    context=v56.load_context(args.asset_root)
    assert not context.get('missing_dependencies'),context.get('missing_dependencies')
    assert context['validation']['hashes_sha256']==s['input_hashes']
    context['units']=s['matched_units']
    A=np.asarray(context['A_solver'],dtype=np.float64)
    assert A.shape==(1084,391)
    # Independent algebraic smoke: positive rescaling/zero pixels and exact recovery.
    test_A=np.array([[1.,0.],[0.,1.],[1.,1.]])
    for truth in (np.array([0.,0.]),np.array([.5,2.])):
        estimate,residual=nnls(test_A,test_A@truth,maxiter=20)
        assert np.allclose(estimate,truth,atol=1e-12) and residual<1e-12
    write(args.output/'self_test.json',dict(status='PASS',test='independent tiny exact and zero NNLS systems'))
    reports={};units={};completed=[]
    for dataset in DATASETS:
        if dataset.startswith('HOLD'):
            assert digest(args.output/'thresholds.json')==read(args.output/'threshold_seal.json')['thresholds_sha256']
        out=args.output/dataset
        case=v57.construct_case(context,s,dataset)
        b_hash=hashlib.sha256(case['B_sim'].tobytes()).hexdigest()
        assert b_hash==read(parent/'clean'/dataset/'runtime_contract.json')['B_sha256']
        mask=case['foreground_mask']; B=case['B_sim']
        assert np.all(B[:,~mask]==0),'NONZERO_BACKGROUND_REQUIRES_SOLVE'
        spectra=B[:,mask].astype(np.float64)
        if args.timing_only:
            sample=np.linspace(0,spectra.shape[1]-1,min(32,spectra.shape[1]),dtype=int)
            started=time.monotonic()
            with ProcessPoolExecutor(max_workers=args.workers,initializer=initialize_worker,initargs=(A,)) as executor:
                list(executor.map(solve_pixel,(spectra[:,i] for i in sample),chunksize=4))
            seconds=time.monotonic()-started
            write(args.output/'timing.json',dict(dataset=dataset,sampled_pixels=len(sample),total_pixels=spectra.shape[1],
                                                sample_seconds=seconds,estimated_seconds_per_case=seconds/len(sample)*spectra.shape[1],
                                                estimate_excludes_rho=True,includes_pool_startup=True,cpu_workers=args.workers,
                                                selection='evenly spaced flattened foreground indices',B_sha256=b_hash))
            print(read(args.output/'timing.json'),flush=True)
            return
        result_path=out/'result.json'
        if result_path.exists():
            report=read(result_path)
            assert report['fingerprint']==fp and report['B_sha256']==b_hash
            assert digest(out/'molecular_units.json')==report['units_sha256']
            assert digest(out/'learned_arrays.npz')==report['arrays_sha256']
        else:
            started=time.monotonic()
            x=np.zeros((391,spectra.shape[1]),dtype=np.float32)
            max_dual=0.;max_complementarity=0.
            executor=ProcessPoolExecutor(max_workers=args.workers,initializer=initialize_worker,initargs=(A,))
            try:
                solutions=executor.map(solve_pixel,(spectra[:,i] for i in range(spectra.shape[1])),chunksize=16)
                for i,solution in enumerate(solutions):
                    assert np.isfinite(solution).all() and (solution>=0).all()
                    if i%500==0:
                        gradient=A.T@(A@solution-spectra[:,i])
                        max_dual=max(max_dual,float(np.maximum(-gradient,0).max()))
                        max_complementarity=max(max_complementarity,float(np.abs(solution*gradient).max()))
                    x[:,i]=solution
                    if i%250==0:
                        write(args.output/'status.json',dict(status='RUNNING',stage='PIXEL_NNLS',dataset=dataset,
                                                            completed=completed,pixels_done=i+1,total_pixels=spectra.shape[1],elapsed_seconds=time.monotonic()-started))
            finally:
                executor.shutdown(wait=True,cancel_futures=True)
            X=np.zeros_like(case['X_true']);X[:,mask]=x
            B_hat=np.zeros_like(B);B_hat[:,mask]=(A@x.astype(np.float64)).astype(np.float32)
            write(args.output/'status.json',dict(status='RUNNING',stage='RHO',dataset=dataset,completed=completed))
            records,raw=v54.learned_records(dataset,case,dict(X_hat=X,B_hat=B_hat),context,1)
            report=dict(fingerprint=fp,B_sha256=b_hash,dataset_id=dataset,
                        all_truth_molecular_identity_count=case['all_truth_molecular_identity_count'],
                        reportable_truth_molecular_identity_count=case['reportable_truth_molecular_identity_count'],
                        learned_raw_identity_performance=raw,elapsed_seconds=time.monotonic()-started,
                        sampled_KKT_max_dual_violation=max_dual,sampled_KKT_max_complementarity=max_complementarity)
            us=v54.molecular_units(records,{dataset:report})
            write(out/'molecular_units.json',us)
            write(out/'reported_candidate_records.json',records)
            out.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(out/'learned_arrays.npz',X_hat=X,B_hat=B_hat)
            report.update(units_sha256=digest(out/'molecular_units.json'),arrays_sha256=digest(out/'learned_arrays.npz'))
            write(result_path,report)
        reports[dataset]=report;units[dataset]=read(out/'molecular_units.json');completed.append(dataset)
        print(dataset,'COMPLETE',flush=True)
        if dataset=='CAL_R3_K125':
            cal=[u for d in completed for u in units[d]]
            pack,curves=v56.threshold_pack(cal,reports)
            # ISTA comparator recalibrated on exactly the same CAL subset.
            ista_rows=v57.rows(parent/'calibration_records.csv')
            ista_cal=[r for r in ista_rows if r['dataset_id'] in completed and r['raw_solver_reported']]
            ista_pack,_=v56.threshold_pack(ista_cal,reports)
            freeze=dict(fingerprint=fp,NNLS=pack,ISTA=ista_pack,
                        CAL_result_hashes={d:digest(args.output/d/'result.json') for d in completed},
                        ISTA_CAL_records_sha256=digest(parent/'calibration_records.csv'))
            f=args.output/'thresholds.json'
            if f.exists(): assert read(f)==freeze,'THRESHOLDS_CHANGED'
            else: write(f,freeze)
            seal=dict(thresholds_sha256=digest(f))
            sealpath=args.output/'threshold_seal.json'
            if sealpath.exists(): assert read(sealpath)==seal,'THRESHOLD_SEAL_CHANGED'
            else: write(sealpath,seal)
        if dataset.startswith('HOLD'):
            assert digest(args.output/'thresholds.json')==read(args.output/'threshold_seal.json')['thresholds_sha256']
    hold_reports={d:r for d,r in reports.items() if d.startswith('HOLD')}
    hold=[u for d in hold_reports for u in units[d]]
    frozen=read(args.output/'thresholds.json')
    ista_hold=[r for r in v57.rows(parent/'heldout_records.csv') if r['dataset_id'] in hold_reports and r['raw_solver_reported']]
    summary={}
    for solver,us in [('NNLS',hold),('ISTA',ista_hold)]:
        summary[solver]={'raw':v56.metrics(us,hold_reports)}
        for score in ('rho_zero','X_hat'):
            for target in ('FDR5','FDR1'):
                entry=frozen[solver][score][target]
                summary[solver][f'{score}_{target}']=v56.metrics(us,hold_reports,score,entry['threshold'] if entry else None)
            v57.write_rows(args.output/f'{solver}_{score}_HOLD_descriptive_curve.csv',v56.calibration_curve(us,score,hold_reports))
    write(args.output/'comparison.json',dict(status='COMPLETE_AWAITING_REVIEW',fingerprint=fp,results=summary))
    write(args.output/'status.json',dict(status='COMPLETE_AWAITING_REVIEW',completed=completed))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--asset-root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--timing-only',action='store_true')
    p.add_argument('--workers',type=int,choices=(1,2,4),default=4)
    args=p.parse_args()
    try: main(args)
    except BaseException:
        write(args.output/'failure.json',dict(status='FAIL',error=traceback.format_exc()))
        raise
