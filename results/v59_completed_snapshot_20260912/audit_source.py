import csv,hashlib,json,math,os,sys,tarfile
from pathlib import Path
from types import SimpleNamespace
from datetime import datetime,timezone
import numpy as np
repo=Path('/root/autodl-tmp/lipiddeconv')
sys.path.insert(0,str(repo/'analysis'))
import run_v59_standardized_cross_library_formal as v
v.deps()
root=repo/'results/v59_oracle_float64_recovery/v59_standardized_cross_library_formal'
out=Path('/root/autodl-tmp/v59_jobs/completed_snapshot_20260912_verified')
out.mkdir(exist_ok=False)
args=SimpleNamespace(output_dir=root,inventory_dir=repo/'results/v59_standardized_cross_library_inventory')
d=v.load_design(args,frozen=True)
v.require_sentinels(args,d)
fp=d['design_fingerprint'];keys=[k for k in v.ids() if (v.directory(args,k)/'result_seal.json').exists()]
assert len(keys)==7,keys
files=set(root/p for p in ('design.json','design_freeze.json','design_audit.json','dataset_manifest.json','oracle_all_validity.json','sentinel_solver_validity.json','precision_recovery_equivalence.json'))
files.update(v.directory(args,k)/'oracle.json' for k in v.ids())
thresholds={}
for domain in v.DOMAINS:
 if (root/domain/'frozen_thresholds.json').exists():
  thresholds[domain]=v.require_threshold(args,d,domain)
  files.update(root/domain/p for p in ('frozen_thresholds.json','frozen_thresholds.seal.json','CAL_threshold_curves.json'))
def rows(p):
 with p.open(newline='',encoding='utf-8-sig') as f:return list(csv.DictReader(f))
def truth(x):
 assert x in ('True','False','true','false','1','0'),x
 return x in ('True','true','1')
reviews=[]
for key in keys:
 domain,base,split,rep=v.parts(key);c=v.directory(args,key)
 if split=='HOLD': assert domain in thresholds
 seal=v.read(c/'result_seal.json')
 assert seal==dict(design_fingerprint=fp,hashes=v.result_hashes(args,key))
 report=v.read(c/'report.json');run=v.read(c/'solver_run.json');runtime=v.read(c/'runtime_contract.json')
 assert report['status']=='COMPLETE' and report['dataset_id']==key and report['design_fingerprint']==fp
 assert run['dataset_id']==base and runtime['dataset_id']==base and run['design_fingerprint']==runtime['design_fingerprint']==fp
 assert runtime['B_sha256']==d['scientific']['datasets'][key]['B_sha256']
 diag=report['process_diagnostics'];assert diag['normal_completion'] and diag['all_outputs_losses_finite']
 assert run['training']['stop_reason'] in ('converged','max_epochs') and run['training']['stopped_epoch']<=3000
 assert v.digest(c/'learned_arrays.npz')==run['arrays_sha256']
 with np.load(c/'learned_arrays.npz',allow_pickle=False) as arrays:
  checked=[]
  for name in arrays.files:
   a=arrays[name]
   if a.dtype.kind in 'fciu': assert np.isfinite(a).all(),(key,name);checked.append(name)
  assert list(arrays['X_hat'].shape)==d['scientific']['datasets'][key]['X_shape']
  assert list(arrays['B_hat'].shape)==d['scientific']['datasets'][key]['B_shape']
 history=v.read(c/'training_history.json')
 for name in ('train_total','eval_total','eval_physical_loss','eval_raw_losses','automatic_loss_weights'):
  assert len(history[name]) and np.isfinite(np.asarray(history[name],dtype=float)).all()
 ms=rows(c/'molecular_false_negative_records.csv');cs=rows(c/'candidate_false_negative_records.csv')
 assert len({r['lipid_name'] for r in ms})==len(ms)
 assert len(cs)==d['scientific']['domains'][domain]['shape'][1]
 assert {r['lipid_name'] for r in cs}=={r['lipid_name'] for r in ms}
 assert sum(truth(r['molecular_truth']) for r in ms)==125
 reported=[r for r in ms if truth(r['raw_solver_reported'])]
 tp=sum(truth(r['molecular_truth']) for r in reported);fpc=len(reported)-tp;fn=125-tp
 raw=report['learned_raw_identity_performance']['molecular_level']
 assert (raw['TP'],raw['FP'],raw['FN'])==(tp,fpc,fn)
 filtered={}
 if domain in thresholds:
  for score,targets in thresholds[domain]['thresholds'].items():
   filtered[score]={}
   for target,cfg in targets.items():
    tau=cfg['threshold'];kept=[r for r in reported if float(r[score])>=tau]
    t=sum(truth(r['molecular_truth']) for r in kept);f=len(kept)-t
    filtered[score][target]=dict(TP=t,FP=f,FN=125-t,FDP=f/len(kept) if kept else None,TP_retention=t/tp if tp else None,all_truth_recall=t/125,raw_solver_FN=fn,filter_induced_true_loss=tp-t)
 source_artifacts=v.solver_evidence(c)
 files.update(p for p in c.iterdir() if p.is_file() and p.suffix in ('.json','.csv'))
 reviews.append(dict(dataset_id=key,status='CACHED_PROCESS_BINDING_HASH_AND_ACCOUNTING_PASS',raw_solver_TP=tp,raw_solver_FP=fpc,raw_solver_FN=fn,filtered=filtered,filter_review='SEALED_DOMAIN_CAL_THRESHOLDS' if domain in thresholds else 'DEFERRED_NO_DOMAIN_CAL_SEAL',array_fields_finite=checked,source_artifact_hashes=source_artifacts,retained_source_directory=str(c),completed_utc=datetime.fromtimestamp((c/'report.json').stat().st_mtime,timezone.utc).isoformat(),deferred=['Final all-18 aggregation','Independent reconstruction residual recomputation','Historical raw MSI asset re-audit']))
 print('REVIEWED',key,flush=True)
state=dict(observed_utc=datetime.now(timezone.utc).isoformat(),planned_cases=18,completed_cases=keys,active_case='D1__CAL_R2_K125',supervisor=v.read(Path('/root/autodl-tmp/v59_jobs/oracle_float64_recovery/status.json')),formal_pid=4983,rho_workers=1,all_results_complete=False,design_fingerprint=fp,reviews=reviews,cleanup_performed=False,large_artifacts_retained_on_source=True)
def write(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
write(out/'review.json',state)
manifest={p.relative_to(root).as_posix():v.digest(p) for p in sorted(files)}
write(out/'source_manifest.json',dict(source_root=str(root),files=manifest))
(out/'audit_source.py').write_bytes(Path(__file__).read_bytes())
archive=out.with_suffix('.tar.gz')
with tarfile.open(archive,'w:gz') as t:
 for p in sorted(files):t.add(p,arcname='source/'+p.relative_to(root).as_posix())
 for p in sorted(out.iterdir()):t.add(p,arcname=p.name)
print('ARCHIVE',flush=True);print(v.digest(archive),flush=True);print(archive.stat().st_size,flush=True)
