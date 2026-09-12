from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,os,shutil,tarfile
SOURCE=Path('/root/autodl-tmp/lipiddeconv/results/ce_identity_spatial_v2')
DEST=Path('/root/v2_retained_runtime_20260912')
WORK=Path('/root/v58_jobs/v2_storage_retirement_20260912')
EXPORT=Path('/root/v58_jobs/ce_identity_spatial_v2_exports')
KEYS=['MILD__CAL_R61_K125','close_neighbor__HOLD_R61_K125','relatively_isolated__HOLD_R61_K125','MILD__CAL_R62_K125','close_neighbor__HOLD_R62_K125','relatively_isolated__HOLD_R62_K125']
EXPECTED=['f0a71bd8cbdef94eab1383d4298e5bf3a77b6d6bffa7554623d3862756aeb90a','c6434b69de60f2b5cc6d9540b7e0c24f598467ec4be03deef37809629817dafe','fd9a90db23b017315bcdadc8f7c30a38a4a9b091de626cbad4a72c2508341042','855a24d65f6d92425c6ebdd1825f6d9f1b23db0de6ff138d25c8a0689788db40','2cc83e59327db979213b060902736caecd4fa3aabbd98a53878beb89490e55ab','633af5007bf270ea30a41ade9a0c6ae56242060f643f7159dd9ed3e27a734f0a']
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(p.read_text())
def write(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
assert not DEST.exists() and not WORK.exists()
assert SOURCE.resolve()==SOURCE and EXPORT.resolve()==EXPORT
WORK.mkdir();DEST.mkdir()
assert DEST.resolve()==DEST and DEST.stat().st_dev!=SOURCE.stat().st_dev
status=read(Path('/root/v58_jobs/ce_identity_spatial_v2_audit_status.json'))
assert set(status['cases'])==set(KEYS)
relocations=[];removals=[];protected={}
for key,archive_hash in zip(KEYS,EXPECTED):
 c=SOURCE/'cases'/key;done=read(c/'training_complete.json')
 assert done['status']=='NORMAL_FINITE_COMPLETE' and done['stopped_epoch']<=3000
 assert done['stop_reason'] in ('max_epochs','converged')
 assert read(c/'independent_evidence_review.json')['status']=='PASS'
 for name,wanted in done['files'].items():
  p=c/name;assert p.is_file() and sha(p)==wanted,(key,name)
  protected[str(p)]=wanted
  if p.suffix=='.pth':
   assert p.parent==c/'training' and not p.is_symlink() and p.resolve()==p
   dst=DEST/p.relative_to(SOURCE)
   relocations.append(dict(case=key,source=str(p),destination=str(dst),sha256=wanted,bytes=p.stat().st_size,action='COPY_VERIFY_THEN_ATOMIC_SYMLINK_REPLACE',reason='Completed training; all bound checkpoint bytes remain accessible at original paths for verification/resumption and future comparisons.'))
 for name in ('training_complete.json','independent_evidence_review.json'):
  protected[str(c/name)]=sha(c/name)
 a=EXPORT/(key+'.tar.gz');receipt=read(EXPORT/(key+'_receipt.json'))
 assert not a.is_symlink() and a.resolve()==a and sha(a)==archive_hash==receipt['archive_sha256']
 removals.append(dict(case=key,path=str(a),resolved=str(a.resolve()),sha256=archive_hash,bytes=a.stat().st_size,reason='Redundant per-case transport archive already downloaded byte-exactly; compact records pushed and complete source evidence remains. Receipt and transfer manifest retained.'))
for name in ('design.json','design_seal.json','evidence_seal.json','spatial_novelty.json'):
 protected[str(SOURCE/name)]=sha(SOURCE/name)
assert len(relocations)==36 and len(removals)==6
needed=sum(x['bytes'] for x in relocations)
assert shutil.disk_usage(DEST).free-needed>4*1024**3
for item in relocations:
 src=Path(item['source']);dst=Path(item['destination']);dst.parent.mkdir(parents=True,exist_ok=True)
 assert dst.parent.resolve().is_relative_to(DEST) and not dst.exists()
 shutil.copy2(src,dst)
 with dst.open('rb') as f:os.fsync(f.fileno())
 assert sha(dst)==sha(src)==item['sha256']
plan=dict(status='PREPARED_COPIES_VERIFIED_AWAITING_GIT_PUSH',observed_utc=datetime.now(timezone.utc).isoformat(),source_root=str(SOURCE),retained_storage_root=str(DEST),pushed_case_snapshot='3e3ad1879859496a40e75d8d80d703e3ccdb2283',relocations=relocations,remove_transport_archives=removals,protected_hashes=protected,data_disk_bytes_to_free=needed,root_redundant_archive_bytes_to_free=sum(x['bytes'] for x in removals),free_before_switch=dict(data=shutil.disk_usage(SOURCE)._asdict(),root=shutil.disk_usage(DEST)._asdict()),scientific_contract_changed=False,training_finished_all_six=True,LP_scoring_remains_active=True,no_source_removed_yet=True,performance_review_deferred_until_both_CAL_seals_and_EVAL_complete=True)
write(WORK/'plan.json',plan)
(WORK/'prepare_source.py').write_bytes(Path(__file__).read_bytes())
archive=WORK/'prepared.tar.gz'
with tarfile.open(archive,'w:gz') as t:
 for n in ('plan.json','prepare_source.py'):t.add(WORK/n,arcname=n)
print('PREPARED',len(relocations),len(removals),needed,flush=True)
print(sha(archive),flush=True)
