from pathlib import Path
from datetime import datetime,timezone
import base64,hashlib,json,os,shutil,tarfile
WORK=Path('/root/v58_jobs/v2_storage_retirement_20260912/final_export')
ROOT=Path('/root/autodl-tmp/lipiddeconv/results/ce_identity_spatial_v2')
TARGET=Path('/root/v58_jobs/ce_identity_spatial_v2_exports/final.tar.gz')
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(p.read_text())
plan_bytes=base64.b64decode('ewogICJzdGF0dXMiOiAiUEVORElOR19GSU5BTF9SRVNVTFRfUFVTSCIsCiAgInBhdGgiOiAiL3Jvb3QvdjU4X2pvYnMvY2VfaWRlbnRpdHlfc3BhdGlhbF92Ml9leHBvcnRzL2ZpbmFsLnRhci5neiIsCiAgInJlc29sdmVkIjogIi9yb290L3Y1OF9qb2JzL2NlX2lkZW50aXR5X3NwYXRpYWxfdjJfZXhwb3J0cy9maW5hbC50YXIuZ3oiLAogICJzeW1saW5rIjogZmFsc2UsCiAgInNoYTI1NiI6ICJiYTI5YTBmMzE1NmE3ZmI0YzE0NzQyMmViZmM4NjFhMmJkMzkwYWI4NWJiNjJkMDI0NmQ5OGM4YWU3ZmIwNzg5IiwKICAiYnl0ZXMiOiAyNTcxMTMzNjcsCiAgInJlYXNvbiI6ICJSZWR1bmRhbnQgdHJhbnNwb3J0IGFyY2hpdmU7IGV2ZXJ5IGNvbXBhY3QvbnVtZXJpY2FsIG91dHB1dCBpcyBkb3dubG9hZGVkIHdpdGggbWF0Y2hpbmcgaGFzaGVzIGFuZCBjb21wbGV0ZSBzb3VyY2Ugb3V0cHV0cyByZW1haW4uIERlbGV0ZSBvbmx5IGFmdGVyIHRoaXMgZXhhY3QgZmluYWwgc25hcHNob3QgaXMgc3VjY2Vzc2Z1bGx5IHB1c2hlZC4iLAogICJsb2NhbF92ZXJpZmllZF9vdXRwdXRfZGlyZWN0b3J5IjogIkM6XFxVc2Vyc1xcbGRcXERvY3VtZW50c1xcQ2hhdEdQVFxcXHU4YmZlXHU5ODk4XFxsaXBpZGRlY29udlxccmVzdWx0c1xcY2VfaWRlbnRpdHlfc3BhdGlhbF92MiIsCiAgInNvdXJjZV9yb290IjogIi9yb290L2F1dG9kbC10bXAvbGlwaWRkZWNvbnYvcmVzdWx0cy9jZV9pZGVudGl0eV9zcGF0aWFsX3YyIiwKICAib3V0cHV0X21hbmlmZXN0X3NoYTI1NiI6ICIzOGYzNjk0YTA1MjRlYmFmMzI4YThkMDVlYTBhYmExN2NhM2MxMDVhOGVhYWVmNTJiNDJkYmUxNDQxZjJhNGI4IiwKICAiaW5kZXBlbmRlbnRfcmV2aWV3X3NoYTI1NiI6ICJmMmJiZGMwN2I0NDY4OTAwYTExOWYzNWQwMmY0YTgzMzQ1NTI4ODA5ZmViMzZiMWQxMGQzZmVhZmQzYTZiZjc2IiwKICAicHJvdGVjdGVkX3RyYWluaW5nX21hbmlmZXN0IjogIi9yb290L3Y1OF9qb2JzL3YyX3N0b3JhZ2VfcmV0aXJlbWVudF8yMDI2MDkxMi9wbGFuLmpzb24iLAogICJwcmVjb25kaXRpb25zIjogWwogICAgIk5vIFYyIG1haW4gb3IgcG9zdHByb2Nlc3NvciBpcyBhY3RpdmUiLAogICAgIkV4YWN0IGZpbmFsIHJlc3VsdHMgYW5kIHJldGlyZW1lbnQgcGxhbiBwdXNoZWQiLAogICAgIkFsbCBzb3VyY2Ugb3V0cHV0IGhhc2hlcyBhbmQgYm91bmQgdHJhaW5pbmcgZGVwZW5kZW5jaWVzIHZlcmlmaWVkIiwKICAgICJMb2NhbCBjb21wbGV0ZSBvdXRwdXQgYmFja3VwIHZlcmlmaWVkIgogIF0sCiAgInJldGlyZV91bmlxdWVfbW9kZWxfb3JfYXJyYXkiOiBmYWxzZQp9Cg==')
assert hashlib.sha256(plan_bytes).hexdigest()=='d06eeac2acd490c5a828da66e3656097c6a75097e1f9e1afa8785f50213316ba'
p=json.loads(plan_bytes)
assert p['path']==str(TARGET) and p['source_root']==str(ROOT)
assert TARGET.resolve()==TARGET and not TARGET.is_symlink()
assert TARGET.parent.resolve()==Path('/root/v58_jobs/ce_identity_spatial_v2_exports')
for pid in (102539,103069):
 try:os.kill(pid,0)
 except ProcessLookupError:continue
 raise RuntimeError('V2_PROCESS_STILL_ALIVE')
assert sha(ROOT/'output_manifest.json')==p['output_manifest_sha256']
assert sha(ROOT/'independent_review.json')==p['independent_review_sha256']
assert read(ROOT/'independent_review.json')['status']=='PASS'
manifest=read(ROOT/'output_manifest.json')
protected=read(Path(p['protected_training_manifest']))['protected_hashes']
def verify():
 for name,h in manifest.items():assert sha(ROOT/name)==h,name
 for name,h in protected.items():assert sha(Path(name))==h,name
verify()
assert sha(TARGET)==p['sha256'] and TARGET.stat().st_size==p['bytes']
WORK.mkdir(exist_ok=False)
(WORK/'plan.json').write_bytes(plan_bytes)
before=shutil.disk_usage('/root').free
TARGET.unlink()
assert not TARGET.exists()
verify()
receipt=dict(status='COMPLETE',observed_utc=datetime.now(timezone.utc).isoformat(),git_final_snapshot_commit='e99f98e37c0c6c5385556c05be6841fdfe5d98fb',plan_sha256=hashlib.sha256(plan_bytes).hexdigest(),removed=[dict(path=str(TARGET),resolved=str(TARGET),sha256=p['sha256'],bytes=p['bytes'],reason=p['reason'])],unique_scientific_artifacts_deleted=False,original_output_hashes_verified=len(manifest),protected_training_hashes_verified=len(protected),free_root_before=before,free_root_after=shutil.disk_usage('/root').free,local_backup_verified_before_push=True,source_outputs_retained=True)
(WORK/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
(WORK/'retire_source.py').write_bytes(Path(__file__).read_bytes())
a=WORK/'receipt.tar.gz'
with tarfile.open(a,'w:gz') as t:
 for n in ('plan.json','receipt.json','retire_source.py'):t.add(WORK/n,arcname=n)
print('FINAL_ARCHIVE_RETIRED_SOURCE_PRESERVED',len(manifest),len(protected),flush=True)
print(sha(a),flush=True)
