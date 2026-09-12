"""Bounded postprocessing for one V2 job: audit completed cases, export, then exit."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import time
import traceback
from review_ce_identity_spatial_v2 import read, sha, write


def export(root,exports,tag,key=None):
    archive=exports/(tag+'.tar.gz');manifest=exports/(tag+'.json')
    if archive.exists():
        receipt=read(exports/(tag+'_receipt.json'))
        assert sha(archive)==receipt['archive_sha256'];return receipt
    if key:
        c=root/'cases'/key
        files=[p for p in c.iterdir() if p.is_file() and p.name!='observation_truth.npz']
        files += [p for p in (c/'training').iterdir() if p.suffix in ('.json','.csv')]
        files += [root/'deployment/launch.json']
    else:
        files=[root/n for n in read(root/'output_manifest.json')]
        files += [root/'output_manifest.json',root/'independent_review.json',root/'findings.md']
    files=sorted(set(files));hashes={p.relative_to(root).as_posix():sha(p) for p in files}
    write(manifest,dict(tag=tag,case=key,files=hashes,
        retained_storage=str(root),retained_large_artifacts='Models, observations/truth and learned arrays remain on the source host; no deletion.'))
    with tarfile.open(archive,'w:gz') as t:
        for p in files:t.add(p,arcname=p.relative_to(root).as_posix())
        t.add(manifest,arcname='transfer_manifests/'+tag+'.json')
    result=dict(archive=str(archive),archive_sha256=sha(archive),bytes=archive.stat().st_size,files=len(files))
    write(exports/(tag+'_receipt.json'),result);return result


def findings(root):
    s=read(root/'summary.json');assert read(root/'independent_review.json')['status']=='PASS'
    lines=['# V2 paired spatial-evidence development pilot','',
        'Reviewed outcome: '+('PROCEED_TO_INDEPENDENT_VALIDATION_DESIGN' if s['main_go'] else 'STOP_CURRENT_VERSION'),'',
        'This is a development result from new spatial realizations of reused identities and a fixed target library; it is not independent chemical-error or population-FDR validation. U, gamma and both CAL-selected thresholds remain unchanged.','',
        '| Method / EVAL challenge | TP | FP | Raw TP | FDP | TP retention | All-truth recall |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for method in ('global','local'):
        m=s['methods'][method]
        for kind,v in [('aggregate',m['aggregate']),*m['by_challenge'].items()]:
            risk=f"{v['FDP']:.2%}" if v['retained_count'] else 'EMPTY'
            retention=f"{v['TP_retention']:.2%}" if v['TP_retention'] is not None else 'UNDEFINED'
            lines.append(f"| {method} / {kind} | {v['TP']} | {v['FP']} | {v['raw_solver_TP']} | {risk} | {retention} | {v['all_truth_recall']:.2%} |")
        e=read(root/f'{method}_calibration_seal.json')['epsilon']
        lines += ['',f'{method}: epsilon={e:.17g}; gamma=1e-6; GO={m["go"]}; strong_success={m["strong_success"]}.',
                  f'Raw solver FN={m["aggregate"]["raw_solver_FN"]}; additional true identities removed by screening={m["aggregate"]["filter_induced_true_loss"]}; final FN={m["aggregate"]["FN"]}.',
                  'EVAL status counts: '+json.dumps(m['aggregate']['status_counts'],sort_keys=True)+'.','']
    lines += ['Each challenge must have a nonempty retained set and at least40% TP retention, in addition to the aggregate1% FDP/40% retention requirements. Empty sets and numerical/threshold nonselections remain in the original denominators.',
              'Spatial novelty distributions, exact calibration sources, all candidate records and numerical witnesses are preserved. No result-driven changes, additional runs or cleanup are performed by this postprocessor.','']
    (root/'findings.md').write_text('\n'.join(lines),encoding='utf-8',newline='\n')


def main(a):
    a.exports.mkdir(parents=True,exist_ok=True);review=Path(__file__).with_name('review_ce_identity_spatial_v2.py')
    keys=[r['key'] for r in read(a.output/'design.json')['scientific']['membership']];done=set();started=time.monotonic()
    while time.monotonic()-started<8*3600:
        for key in keys:
            if key in done or not (a.output/'cases'/key/'training_complete.json').exists():continue
            subprocess.run([sys.executable,str(review),str(a.output),'--case',key,'--old-v1',str(a.old_v1)],check=True)
            receipt=export(a.output,a.exports,key,key);done.add(key)
            write(a.status,dict(status='COMPLETED_CASE_REVIEWED_EXPORTED',cases=sorted(done),latest=receipt))
        if (a.output/'output_manifest.json').exists():
            subprocess.run([sys.executable,str(review),str(a.output)],check=True)
            findings(a.output);receipt=export(a.output,a.exports,'final')
            write(a.status,dict(status='COMPLETE_REVIEWED_ARCHIVE_READY',cases=keys,final=receipt,local_git_sync_required=True));return
        if (a.output/'failure.json').exists():
            write(a.status,dict(status='MAIN_JOB_FAILED_NO_AUTOMATIC_RETRY',cases=sorted(done)));return
        try:os.kill(a.main_pid,0)
        except ProcessLookupError:
            write(a.status,dict(status='MAIN_PROCESS_ENDED_WITHOUT_COMPLETE_MANIFEST',cases=sorted(done)));return
        time.sleep(15)
    write(a.status,dict(status='POSTPROCESSOR_TIME_LIMIT_NO_ACTION_ON_MAIN_JOB',cases=sorted(done)))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--exports',type=Path,required=True);p.add_argument('--status',type=Path,required=True)
    p.add_argument('--main-pid',type=int,required=True)
    p.add_argument('--old-v1',type=Path,default=Path('/root/v58_jobs/ce_uncertainty_identity_pilot_run04'))
    a=p.parse_args()
    try:main(a)
    except BaseException:
        write(a.status,dict(status='POSTPROCESSOR_FAILED_NEEDS_REVIEW',error=traceback.format_exc()));raise
