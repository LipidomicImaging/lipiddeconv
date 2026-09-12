from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,os,shutil,sys,tarfile
sys.path.insert(0,'/root/v58_jobs/ce_identity_spatial_v2_code')
import run_ce_identity_spatial_v2 as runner
r=Path('/root/v58_jobs/v2_storage_retirement_20260912');p=json.loads((r/'plan.json').read_text());receipt=json.loads((r/'receipt.json').read_text())
assert receipt['status']=='COMPLETE' and receipt['unique_checkpoint_bytes_deleted']==0
assert runner.sha(Path(runner.__file__))=='580612a7a207bfc76ab1af4db32b7ca7cd33151a845818d46b58014d8bbb8ab3'
root=Path(p['source_root']);keys=sorted({x['case'] for x in p['relocations']})
for key in keys:
 c=root/'cases'/key
 runner.verify_files(c,json.loads((c/'training_complete.json').read_text())['files'])
for path,wanted in p['protected_hashes'].items(): assert runner.sha(Path(path))==wanted,path
for x in p['relocations']:
 src=Path(x['source']);dst=Path(x['destination'])
 assert src.is_symlink() and src.resolve()==dst and runner.sha(dst)==x['sha256']
for x in p['remove_transport_archives']:
 assert not Path(x['path']).exists()
 for suffix in ('.json','_receipt.json'):assert Path(x['path']).with_name(x['case']+suffix).is_file()
live={}
for pid in (102539,103069):
 try:os.kill(pid,0);live[str(pid)]=True
 except ProcessLookupError:live[str(pid)]=False
status=dict(status='PASS',observed_utc=datetime.now(timezone.utc).isoformat(),original_runner_sha256=runner.sha(Path(runner.__file__)),original_runner_verify_files_cases=keys,original_bound_artifact_count=90,protected_hashes_verified=len(p['protected_hashes']),verified_checkpoint_links=36,confirmed_removed_archive_count=6,unique_checkpoint_bytes_deleted=0,live_processes=live,latest_scoring_log=Path('/root/v58_jobs/ce_identity_spatial_v2_run.log').read_text().splitlines()[-4:],free=dict(data=shutil.disk_usage(root)._asdict(),root=shutil.disk_usage('/root')._asdict()))
(r/'post_verification.json').write_text(json.dumps(status,indent=2)+'\n')
a=r/'completed.tar.gz'
with tarfile.open(a,'w:gz') as t:
 for name in ('receipt.json','post_verification.json','post_verify.py'):t.add(r/name,arcname=name)
print('POST_VERIFY_PASS',len(keys),live,flush=True)
print(runner.sha(a),flush=True)
print(json.dumps(status['free']),flush=True)
