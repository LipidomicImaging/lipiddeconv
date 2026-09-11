"""Operational recovery only: preserve V58 failure, resume, then run frozen omission fits."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

ROOT=Path('/root/autodl-tmp/lipiddeconv')
OUT=ROOT/'results/v58_spectral_library_mismatch_fdr_recalibration_compact_recovery'
JOB=Path('/root/v58_jobs/storage_recovery_20260911')
FP='74032334eb538114e76459a1bb82ef0187d38f48ab20e60124021cd638afd8f0'
MISSING=ROOT/'results/missing_library_challenge_k125'
IDS=[f'{s}__{split}_R{r}_K{k:03d}' for s in ('MILD','MODERATE') for split in ('CAL','HOLD') for r in range(1,6) for k in (50,125,175)]


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    os.chdir(ROOT)
    JOB.mkdir(parents=True,exist_ok=True)
    # Exclusively owned launch directory; repeated starts cannot replace evidence.
    lock=JOB/'launch.json'
    with lock.open('x') as f:json.dump({'pid':os.getpid(),'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()},f)
    original=Path('/root/v58_jobs/compact_recovery_status.json')
    shutil.copy2(original,JOB/'original_failure_status.json')
    shutil.copy2('/root/v58_jobs/compact_recovery_job.py',JOB/'original_supervisor.py')
    state={'status':'RUNNING','pid':os.getpid(),'completed':[],'planned_count':60,'original_failure_sha256':sha(original),'fingerprint':FP}
    def save():
        state['updated_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
        temp=JOB/'status.tmp';temp.write_text(json.dumps(state,indent=2)+'\n');temp.replace(JOB/'status.json')
    def wait_storage(required):
        while shutil.disk_usage(OUT).free < required:
            state['status']='WAITING_FOR_DATA_DISK_EXPANSION';state['free_bytes']=shutil.disk_usage(OUT).free;state['required_free_bytes']=required;save();time.sleep(1800)
        state['status']='RUNNING';save()
    cmd=[sys.executable,str(ROOT/'analysis/run_v58_spectral_library_mismatch_fdr_recalibration.py'),'--asset-root','/root/autodl-tmp/decon-lipid','--output-dir',str(OUT)]
    try:
        for dataset in IDS:
            severity,base=dataset.split('__');p=OUT/severity/base/'report.json'
            state['active_dataset']=dataset;save()
            if not p.exists():
                wait_storage(int(2.6*1024**3))
                with (JOB/(dataset+'.log')).open('x') as log:subprocess.run(cmd+['--dataset',dataset],stdout=log,stderr=subprocess.STDOUT,check=True)
            report=json.loads(p.read_text())
            assert report['status']=='COMPLETE' and report['dataset_id']==dataset and report['design_fingerprint']==FP
            state['completed'].append(dataset);save()
        state['active_dataset']=None;state['stage']='V58_AGGREGATE';save()
        with (JOB/'aggregate.log').open('x') as log:subprocess.run(cmd+['--aggregate'],stdout=log,stderr=subprocess.STDOUT,check=True)
        state['V58']='AGGREGATED_AWAITING_REVIEW';state['stage']='MISSING_LIBRARY_PREFLIGHT';save()
        wait_storage(4*1024**3)
        # Residual GPU use from unrelated tasks is not permission to compete.
        while subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():
            state['status']='WAITING_FOR_GPU_RELEASE';save();time.sleep(1800)
        require_files=[MISSING/'execution_design.json',MISSING/'execution_freeze.json']
        assert all(p.exists() for p in require_files)
        frozen=json.loads(require_files[0].read_text());assert frozen['fingerprint']=='b51137f3c6f12cf9bdf0e3e9f1c00c1fff53f9e4d62dc1c54de0b04a7910d9ef'
        assert sha(Path('/root/run_missing_library_challenge.py'))==frozen['scientific']['script_sha256']
        assert not (MISSING/'runs').exists(),'MISSING_LIBRARY_ALREADY_STARTED_REVIEW_REQUIRED'
        omission=[sys.executable,'/root/run_missing_library_challenge.py','run','--root',str(ROOT),'--asset-root','/root/autodl-tmp/decon-lipid','--assets',str(ROOT/'results/computational_closure/missing_library'),'--output',str(MISSING),'--device','cuda:0','--rho-workers','1']
        state['status']='RUNNING';state['stage']='MISSING_LIBRARY';save()
        with (JOB/'missing_library.log').open('x') as log:subprocess.run(omission,stdout=log,stderr=subprocess.STDOUT,check=True)
        state['status']='ALL_COMPUTE_COMPLETE_AWAITING_REVIEW';save()
    except BaseException:
        state['status']='FAILED';state['error']=traceback.format_exc();save();raise


if __name__=='__main__':main()
