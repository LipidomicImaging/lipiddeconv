"""Freeze a bounded, explicitly selected CAL-only diagnostic; no outcome tuning."""
import csv
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/cal_rho_numerical_diagnostic'
C=ROOT/'results/v57_spectral_spatial_identity_confidence_benchmark'
M=ROOT/'results/v58_completed_snapshot_20260911'


def read(p):return json.loads(p.read_text(encoding='utf-8'))


def rows(p):
    with p.open(encoding='utf-8',newline='') as f:return list(csv.DictReader(f))


def main():
    dataset='CAL_R1_K125'
    clean={r['lipid_name']:r for r in rows(C/'calibration_records.csv') if r['dataset_id']==dataset}
    mild=rows(M/'MILD'/dataset/'molecular_false_negative_records.csv')
    tc=read(C/'global_frozen_thresholds.json')['global_thresholds']['rho_zero']['FDR5']['threshold']
    tm=read(M/'MILD/local_frozen_thresholds.json')['global_thresholds']['rho_zero']['FDR5']['threshold']
    eligible=[]
    for r in mild:
        c=clean[r['lipid_name']];ids=json.loads(r['candidate_indices'])
        if len(ids)!=1 or c['raw_solver_reported']!='True' or r['raw_solver_reported']!='True':continue
        assert c['candidate_indices']==r['candidate_indices'] and c['molecular_truth']==r['molecular_truth']
        eligible.append(dict(candidate_index=ids[0],lipid_name=r['lipid_name'],truth=r['molecular_truth']=='True',
                             clean_rho=float(c['rho_zero']),mild_rho=float(r['rho_zero'])))
    rules=[('true_positive_to_zero',lambda r:r['truth'] and r['clean_rho']>0 and r['mild_rho']==0),
           ('false_gains_original_cutoff',lambda r:not r['truth'] and r['clean_rho']<=tc and r['mild_rho']>tc),
           ('true_survives_mild_cutoff',lambda r:r['truth'] and r['mild_rho']>=tm),
           ('false_passes_mild_cutoff',lambda r:not r['truth'] and r['mild_rho']>=tm)]
    selected=[];used=set();counts={}
    for name,rule in rules:
        pool=sorted([r for r in eligible if rule(r)],key=lambda r:r['candidate_index']);counts[name]=len(pool)
        for r in [r for r in pool if r['candidate_index'] not in used][:2]:
            selected.append(dict(r,stratum=name));used.add(r['candidate_index'])
    paths=[C/'calibration_records.csv',C/'global_frozen_thresholds.json',M/'MILD'/dataset/'molecular_false_negative_records.csv',M/'MILD/local_frozen_thresholds.json']
    plan=dict(status='FROZEN_CAL_DIAGNOSTIC_SELECTION',dataset=dataset,severity='MILD',selected=selected,
        eligible_counts=counts,selection_rule='At most two lowest unused candidate indices per listed stratum, in listed order; jointly reported singleton identities only.',
        clean_threshold=tc,mild_threshold=tm,methods=['scipy_nnls_original_order','scipy_nnls_reverse_order','scipy_lsq_linear_bvls_tol_1e-12'],
        maxiter=3910,top_mismatch_channel_fraction=.10,threads=1,device='CPU_ONLY',
        sources={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        limitations=['Purposive CAL examples, not prevalence estimates or HOLD validation.',
                     'Independent finite-precision solves assess sensitivity, not a rigorous arithmetic error bound.',
                     'Known CLEAN/MILD signal difference is diagnostic-only, not a noise estimate or an input to a deployed score.'])
    OUT.mkdir(exist_ok=True);dest=OUT/'selection.json'
    if dest.exists():assert read(dest)==plan,'SELECTION_ALREADY_FROZEN'
    else:dest.write_text(json.dumps(plan,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(plan,indent=2))


if __name__=='__main__':main()
