"""Verify downloaded V58 snapshot and independently audit interim HOLD counts."""
import csv
import hashlib
import json
import math
import argparse
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/v58_completed_snapshot_20260911'


def read(p):return json.loads(p.read_text(encoding='utf-8'))


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    plan=read(OUT/'cleanup_plan.json');summary=read(OUT/'partial_summary.json')
    final=summary['status']=='FINAL_COMPLETE_SNAPSHOT'
    if final:
        expected={f'{s}__{split}_R{r}_K{k:03d}' for s in ('MILD','MODERATE') for split in ('CAL','HOLD') for r in range(1,6) for k in (50,125,175)}
        assert set(plan['completed'])==expected and summary['completed']==60
        audit=read(OUT/'process_audit.json')
        assert set(audit['cases'])==expected and all(r['normal_completion'] and r['finite'] for r in audit['cases'].values())
        aggregate=read(OUT/'report.json')
        for protocol in ('CLEAN_FIXED','LOCAL_CAL_RECALIBRATION'):
            expected_metrics=[m for s in ('MILD','MODERATE') for m in summary['results'][s][protocol]['overall']]
            assert aggregate[protocol]==expected_metrics,'FINAL_AGGREGATE_DISAGREES'
        assert len({r['resolved'] for r in plan['deletions']})==len(plan['deletions'])
        protected={r['final_model'] for r in audit['cases'].values()}
        assert not protected & {r['path'] for r in plan['deletions']}
    for name,h in plan['source_files'].items():assert sha(OUT/name)==h,name
    if final:
        for dataset in plan['completed']:
            severity,base=dataset.split('__');loc=OUT/severity/base
            with (loc/'candidate_false_negative_records.csv').open(newline='',encoding='utf-8') as f:candidates=list(csv.DictReader(f))
            with (loc/'molecular_false_negative_records.csv').open(newline='',encoding='utf-8') as f:molecules=list(csv.DictReader(f))
            assert len(candidates)==391 and len({r['candidate_index'] for r in candidates})==391
            assert len({r['lipid_name'] for r in molecules})==len(molecules)
            for r in candidates:
                assert r['dataset_id']==dataset and math.isfinite(float(r['X_hat']))
                assert (r['raw_solver_reported']=='True')==(float(r['X_hat'])>.001)
                if r['raw_solver_reported']=='True':assert math.isfinite(float(r['rho_zero']))
            assert {r['lipid_name'] for r in candidates if r['raw_solver_reported']=='True'}=={r['lipid_name'] for r in molecules if r['raw_solver_reported']=='True'}
            truth=sum(r['molecular_truth']=='True' for r in molecules)
            raw=[r for r in molecules if r['raw_solver_reported']=='True'];tp=sum(r['molecular_truth']=='True' for r in raw)
            metrics=read(loc/'report.json')['learned_raw_identity_performance']['molecular_level']
            assert truth==int(base[-3:]) and (tp,len(raw)-tp,truth-tp)==(metrics['TP'],metrics['FP'],metrics['FN'])
    notes=['V58 completed-case storage review — 2026-09-11','',
           f"Snapshot: {summary['completed']}/60 completed cases. "+('Complete final benchmark review.' if final else 'Interim review; incomplete MODERATE results are not final.'),'',
           'Both severity-specific cutoffs were frozen with the unchanged runner on all15 CAL datasets per severity before this review opened HOLD records. No threshold was tuned using HOLD.','',
           '| Severity | Protocol / score / target | HOLD cases | TP | FP | Solver FN | Filter loss | FDR | Recall |',
           '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    raw_notes=[]
    for s,block in summary['results'].items():
        rows=[]
        for d in plan['completed']:
            if not d.startswith(s+'__HOLD'):continue
            _,base=d.split('__')
            with (OUT/s/base/'molecular_false_negative_records.csv').open(encoding='utf-8',newline='') as f:rows+=list(csv.DictReader(f))
        raw=[r for r in rows if r['raw_solver_reported']=='True']
        truth=sum(r['molecular_truth']=='True' for r in rows)
        raw_tp=sum(r['molecular_truth']=='True' for r in raw)
        assert (raw_tp,len(raw)-raw_tp,truth-raw_tp)==(block['raw']['TP'],block['raw']['FP'],block['raw']['FN'])
        for protocol in ('CLEAN_FIXED','LOCAL_CAL_RECALIBRATION'):
            for m in block[protocol]['overall']:
                retained=[r for r in raw if m['threshold'] is not None and float(r[m['score']])>=m['threshold']]
                tp=sum(r['molecular_truth']=='True' for r in retained);fp=len(retained)-tp
                assert (tp,fp,truth-raw_tp,raw_tp-tp)==(m['TP'],m['FP'],m['raw_solver_FN'],m['filter_induced_true_loss'])
                assert math.isclose(m['all_truth_recall'],tp/truth,abs_tol=1e-14)
                fdr=f"{100*fp/len(retained):.3f}%" if retained else 'undefined (none retained)'
                notes.append(f"| {s} | {protocol}/{m['score']}/{m['target']} | {block['HOLD_completed']}/15 | {tp} | {fp} | {truth-raw_tp} | {raw_tp-tp} | {fdr} | {100*tp/truth:.2f}% |")
            if final:
                for level,key in [('by_K','K'),('by_replicate','replicate')]:
                    for m in block[protocol][level]:
                        subset=[r for r in rows if str(r[key])==str(m[key])]
                        raw_sub=[r for r in subset if r['raw_solver_reported']=='True']
                        retained=[r for r in raw_sub if m['threshold'] is not None and float(r[m['score']])>=m['threshold']]
                        tp=sum(r['molecular_truth']=='True' for r in retained)
                        ntruth=sum(r['molecular_truth']=='True' for r in subset)
                        assert (tp,len(retained)-tp,ntruth-tp)==(m['TP'],m['FP'],m['FN'])
        raw_notes.append(f"{s} raw: TP{raw_tp}, FP{len(raw)-raw_tp}, FN{truth-raw_tp}; truth contexts={truth}.")
    notes += [''] + raw_notes + ['',
        'The complete MILD HOLD subset shows that CLEAN rho thresholds do not transfer under this spectral mismatch: empirical FDR is37.16%. Local CAL FDR5 recalibration reduces HOLD FDR to3.90%, but recall falls to21.14% and1310 raw true positives are filtered out. The local FDR1 cutoff reaches1.89% HOLD FDR, exceeding its nominal target. This is a substantial reliability/retention limitation, not a successful universal threshold transfer. '+('MODERATE now has all15 HOLD cases and is reported in full above.' if final else 'MODERATE has only5/15 HOLD cases here and must remain provisional.')]
    bytes_by_fs={}
    for r in plan['deletions']:
        fs='root cache' if r['resolved'].startswith('/root/v58_compact_runtime_cache/') else 'data disk'
        bytes_by_fs[fs]=bytes_by_fs.get(fs,0)+r['bytes']
    notes += ['', 'Storage decision:', '',
              f"Planned retirement: {len(plan['deletions'])} intermediate/redundant checkpoint files, {sum(bytes_by_fs.values())/1024**3:.3f} GiB. Split: {bytes_by_fs} bytes.",
              'Retain final epoch3000 checkpoint, learned X_hat/B_hat arrays, all molecular/candidate records, reports, training history, diagnostics, runtime bindings, CAL threshold seals, design/oracle/adoption provenance, and every sentinel-related case. V57 parent controls and old draft sentinel artifacts are outside the deletion scope.',
              'The completed-result verifier and final aggregator consume retained report/record hashes. The cached learned-result path consumes retained arrays/history/diagnostics. Intermediate checkpoints are not required by these paths. The missing-library challenge references V57 controls, not these compact V58 intermediate checkpoints.',
              'Deleting intermediate checkpoints prevents future reloading of those intermediate epochs; hashes and diagnostics retain the historical record, but are not backups of model weights. Final trained models and final arrays remain available remotely. Original runtime storage manifests remain historical; cleanup_plan and deletion_receipt explicitly supersede existence claims for retired files.',
              'No active/incomplete case is eligible. Actual deletion is a separate step after Git push and a second complete hash verification. Refer to deletion_receipt.json for whether it has occurred.',
              '', 'Interpretation: compare empirical FDR jointly with recall and TP loss. Repeated mappings and nested K share identities and are not independent biological replicates. Partial severity results cannot establish final performance. No scientific definition, training setting, rho calculation or target library was changed.']
    (OUT/'analysis_record.md').write_text('\n'.join(notes)+'\n',encoding='utf-8')
    (OUT/'local_review.json').write_text(json.dumps(dict(status='PASS',source_files_verified=len(plan['source_files']),
        completed=summary['completed'],independent_HOLD_counts_verified=True,final_aggregate_matches=final,
        candidate_gate_and_per_K_replicate_checks=final,cleanup_plan_sha256=sha(OUT/'cleanup_plan.json'),
        planned_freed_bytes_by_filesystem=bytes_by_fs),indent=2)+'\n',encoding='utf-8')
    print('PASS',summary['completed'],bytes_by_fs)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=OUT);args=p.parse_args();OUT=args.output;main()
