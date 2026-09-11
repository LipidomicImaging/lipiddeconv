"""Audit bounded CAL diagnostic outputs and summarize numerical sensitivity."""
import csv
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/cal_rho_numerical_diagnostic'


def read(p):return json.loads(p.read_text(encoding='utf-8'))


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    plan=read(OUT/'selection.json');provenance=read(OUT/'provenance.json')
    assert provenance['selection_sha256']==sha(OUT/'selection.json')
    assert provenance['script_sha256']==sha(OUT/'executed_runner.py')
    for name,h in provenance['files'].items():assert sha(OUT/name)==h,name
    with (OUT/'fit_diagnostics.csv').open(newline='',encoding='utf-8') as f:rows=list(csv.DictReader(f))
    assert len(rows)==len(plan['selected'])*2*len(plan['methods'])
    assert len({(r['candidate_index'],r['domain'],r['method']) for r in rows})==len(rows)
    results=[]
    for s in plan['selected']:
        for domain in ('CLEAN','MILD'):
            group=[r for r in rows if int(r['candidate_index'])==s['candidate_index'] and r['domain']==domain]
            assert {r['method'] for r in group}==set(plan['methods'])
            original=next(r for r in group if r['method']==plan['methods'][0])
            values=[float(r['rho']) for r in group];stored=s['clean_rho' if domain=='CLEAN' else 'mild_rho']
            results.append(dict(candidate_index=s['candidate_index'],lipid_name=s['lipid_name'],truth=s['truth'],stratum=s['stratum'],
                domain=domain,stored_rho=stored,recomputed_original_rho=float(original['rho']),
                original_absolute_difference=abs(float(original['rho'])-stored),min_rho=min(values),max_rho=max(values),
                method_range=max(values)-min(values),
                original_cutoff_agreement=len({r['passes_original_cutoff'] for r in group})==1,
                mild_cutoff_agreement=len({r['passes_mild_cutoff'] for r in group})==1,
                max_dual_violation=max(float(r[k]) for r in group for k in ('full_dual_violation','deleted_dual_violation')),
                max_complementarity=max(float(r[k]) for r in group for k in ('full_complementarity','deleted_complementarity')),
                original_x_full_candidate=float(original['x_full_candidate']),
                correction_cosine_with_known_mismatch=float(original['correction_cosine_with_known_mismatch']),
                positive_gain_fraction_on_top_mismatch_channels=original['positive_gain_fraction_on_top_mismatch_channels']))
    with (OUT/'review_summary.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(results[0]));w.writeheader();w.writerows(results)
    summary=dict(status='REVIEWED_NUMERICAL_SENSITIVITY_NOT_A_NEW_CALIBRATION',fit_count=len(rows),
                 source_hashes_valid=True,source_script_sha256=provenance['script_sha256'],
                 original_cutoff_sensitive_cases=sum(not r['original_cutoff_agreement'] for r in results),
                 mild_cutoff_sensitive_cases=sum(not r['mild_cutoff_agreement'] for r in results),
                 max_BVLS_negative_projection_norm=max(float(r[k]) for r in rows for k in ('full_negative_projection_norm','deleted_negative_projection_norm')),
                 cases=results)
    (OUT/'review_summary.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
