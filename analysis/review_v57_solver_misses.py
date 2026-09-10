"""Descriptive V57 HOLD miss review; no fitting or independent-row inference."""
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from analyze_v57_computational_closure import metrics, yes

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'results/v57_spectral_spatial_identity_confidence_benchmark/molecular_false_negative_records.csv'
OUT=ROOT/'results/computational_closure/v57/stratified_review'


def save(name,rows):
    with (OUT/name).open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    with SOURCE.open(newline='',encoding='utf-8') as f:
        rows=[r for r in csv.DictReader(f) if r['split']=='HOLD']
    grouped={};strata=[]
    for field in ('K','replicate','lipid_class'):
        groups=defaultdict(list)
        for r in rows:groups[r[field]].append(r)
        for label,values in sorted(groups.items()):
            for target in ('FDR5','FDR1'):
                strata.append(dict(dimension=field,group=label,target=target,
                                   **metrics(values,f'retained_by_global_{target}')))
    save('by_K_replicate_class.csv',strata)
    truth=[r for r in rows if yes(r['molecular_truth'])]
    groups=defaultdict(list)
    for r in truth:groups[r['lipid_name']].append(r)
    identities=[]
    for name,values in groups.items():
        missing=[r for r in values if not yes(r['raw_solver_reported'])]
        identities.append(dict(lipid_name=name,lipid_class=values[0]['lipid_class'],
                               truth_contexts=len(values),solver_miss_contexts=len(missing),
                               miss_fraction=len(missing)/len(values),
                               K_with_misses='|'.join(sorted({r['K'] for r in missing},key=int)),
                               replicates_with_misses='|'.join(sorted({r['replicate'] for r in missing})),
                               cone_isolation=values[0]['cone_isolation'],max_fragment_cosine=values[0]['max_fragment_cosine']))
    identities.sort(key=lambda r:(-r['solver_miss_contexts'],r['lipid_name']))
    save('identity_miss_recurrence.csv',identities)
    count=sum(r['solver_miss_contexts'] for r in identities)
    assert count==122
    summary=dict(input_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
                 truth_contexts=len(truth),distinct_truth_identities=len(identities),solver_miss_contexts=count,
                 identities_ever_missed=sum(r['solver_miss_contexts']>0 for r in identities),
                 top5_miss_count=sum(r['solver_miss_contexts'] for r in identities[:5]),top10=identities[:10],
                 K_summary=[r for r in strata if r['dimension']=='K' and r['target']=='FDR5'],
                 replicate_summary=[r for r in strata if r['dimension']=='replicate' and r['target']=='FDR5'],
                 limitations=['Context counts share identities and nested K; no independent-row p-values.',
                              'Class-specific FDR is descriptive, not a class-calibrated guarantee.',
                              'Miss association does not establish causal spectral ambiguity.'])
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
