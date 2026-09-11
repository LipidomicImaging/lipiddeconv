"""Read-only paired CLEAN/MILD accounting; no causal or recalibration claim."""
import csv
import hashlib
import json
import statistics
from collections import Counter,defaultdict
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
CLEAN=ROOT/'results/v57_spectral_spatial_identity_confidence_benchmark'
MILD=ROOT/'results/v58_completed_snapshot_20260911'
OUT=ROOT/'results/v58_clean_mild_paired_review'


def read(p):return json.loads(p.read_text(encoding='utf-8'))


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def rows(p):
    with p.open(encoding='utf-8',newline='') as f:return list(csv.DictReader(f))


def write_csv(name,data):
    with (OUT/name).open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)


def main():
    OUT.mkdir(exist_ok=True)
    plan=read(MILD/'cleanup_plan.json')
    for name,h in plan['source_files'].items():assert sha(MILD/name)==h
    # CLEAN threshold bytes are bound in the frozen V58 parent provenance.
    design=read(MILD/'design.json')
    parent_hashes=design['scientific']['V57_parent_file_hashes']
    parent_checks={}
    for name in ('global_frozen_thresholds.json','heldout_records.csv'):
        actual=sha(CLEAN/name)
        normalized=hashlib.sha256((CLEAN/name).read_bytes().replace(b'\r\n',b'\n')).hexdigest()
        assert actual==parent_hashes[name] or normalized==parent_hashes[name],name
        parent_checks[name]=dict(expected=parent_hashes[name],actual=actual,LF_normalized=normalized,
                                match='BYTE_EXACT' if actual==parent_hashes[name] else 'LF_NORMALIZED_ONLY')
    source=rows(CLEAN/'heldout_records.csv')
    lookup={(r['dataset_id'],r['lipid_name']):r for r in source}
    assert len(lookup)==len(source)
    ct=read(CLEAN/'global_frozen_thresholds.json')['global_thresholds']['rho_zero']['FDR5']['threshold']
    mt=read(MILD/'MILD/local_frozen_thresholds.json')['global_thresholds']['rho_zero']['FDR5']['threshold']
    records=[]
    for r in [r for f in sorted((MILD/'MILD').glob('HOLD*/molecular_false_negative_records.csv')) for r in rows(f)]:
        c=lookup[r['base_dataset_id'],r['lipid_name']]
        for field in ('molecular_truth','reportable_truth','K','replicate','candidate_indices'):
            assert c[field]==r[field],(r['dataset_id'],r['lipid_name'],field)
        cr=c['raw_solver_reported']=='True';mr=r['raw_solver_reported']=='True'
        cs=float(c['rho_zero']) if cr else None;ms=float(r['rho_zero']) if mr else None
        record=dict(dataset_id=r['base_dataset_id'],lipid_name=r['lipid_name'],K=r['K'],replicate=r['replicate'],
                    truth=r['molecular_truth']=='True',reportable_truth=r['reportable_truth']=='True',
                    clean_raw=cr,mild_raw=mr,clean_rho=cs,mild_rho=ms,
                    clean_X_hat=float(c['X_hat']),mild_X_hat=float(r['X_hat']),
                    clean_at_clean=cr and cs>=ct,clean_at_mild=cr and cs>=mt,
                    mild_at_clean=mr and ms>=ct,mild_at_mild=mr and ms>=mt,
                    score_direction='not_jointly_reported' if not(cr and mr) else 'up' if ms>cs else 'down' if ms<cs else 'equal',
                    cone_isolation=r['cone_isolation'],max_fragment_cosine=r['max_fragment_cosine'])
        records.append(record)
    assert len({r['dataset_id'] for r in records})==15
    truth=[r for r in records if r['truth']];assert len(truth)==1750
    transitions=Counter((r['clean_raw'],r['mild_raw']) for r in truth)
    metrics=[]
    for flag in ('clean_raw','clean_at_clean','clean_at_mild','mild_raw','mild_at_clean','mild_at_mild'):
        selected=[r for r in records if r[flag]];tp=sum(r['truth'] for r in selected);fp=len(selected)-tp
        metrics.append(dict(condition=flag,TP=tp,FP=fp,FN=len(truth)-tp,FDR=fp/len(selected) if selected else None,
                            recall=tp/len(truth)))
    changes=[]
    for label in (True,False):
        g=[r for r in records if r['truth']==label and r['clean_raw'] and r['mild_raw']]
        positive=[r for r in g if r['clean_rho']>0 and r['mild_rho']>0]
        changes.append(dict(truth=label,jointly_reported=len(g),up=sum(r['score_direction']=='up' for r in g),
                            median_clean_rho=statistics.median(r['clean_rho'] for r in g),
                            median_mild_rho=statistics.median(r['mild_rho'] for r in g),
                            down=sum(r['score_direction']=='down' for r in g),equal=sum(r['score_direction']=='equal' for r in g),
                            clean_positive_to_mild_zero=sum(r['clean_rho']>0 and r['mild_rho']==0 for r in g),
                            clean_zero_to_mild_positive=sum(r['clean_rho']==0 and r['mild_rho']>0 for r in g),
                            both_positive_count=len(positive),median_mild_over_clean_if_both_positive=statistics.median(r['mild_rho']/r['clean_rho'] for r in positive) if positive else None))
    previously_kept=[r for r in truth if r['clean_at_clean']]
    categories=Counter('new_solver_miss' if not r['mild_raw'] else 'lost_at_original_threshold' if not r['mild_at_clean'] else 'lost_only_at_recalibrated_threshold' if not r['mild_at_mild'] else 'retained' for r in previously_kept)
    assert sum(categories.values())==len(previously_kept)
    assert sum(transitions.values())==len(truth)
    by_k=[]
    for k in ('50','125','175'):
        g=[r for r in previously_kept if r['K']==k]
        for category,count in Counter('new_solver_miss' if not r['mild_raw'] else 'lost_at_original_threshold' if not r['mild_at_clean'] else 'lost_only_at_recalibrated_threshold' if not r['mild_at_mild'] else 'retained' for r in g).items():
            by_k.append(dict(K=k,category=category,count=count,clean_retained_truth=len(g)))
    write_csv('paired_molecular_records.csv',records);write_csv('crossed_threshold_metrics.csv',metrics)
    write_csv('joint_score_changes.csv',changes);write_csv('paired_loss_by_K.csv',by_k)
    summary=dict(status='PASS',clean_threshold=ct,mild_threshold=mt,truth_contexts=len(truth),
        raw_truth_transitions={str(k):v for k,v in transitions.items()},previously_clean_retained_categories=dict(categories),
        metrics=metrics,joint_score_changes=changes,clean_parent_hash_checks=parent_checks,
        limitations=['Crossed thresholds are descriptive accounting, not new calibrated operating points.',
                    'Score changes conditional on joint reporting omit newly reported/disappearing candidates.',
                    'Same identities/K/maps are matched; changing B also changes fitted solver output. This does not isolate a direct spectral effect on rho.',
                    'Nested K and mappings repeat identities; no independent-sample or causal interaction claim.'])
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
