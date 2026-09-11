"""Cached V58 ranking/loss diagnostics; HOLD envelopes are descriptive, not calibration."""
import csv
import hashlib
import json
import statistics
from collections import defaultdict
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'results/v58_completed_snapshot_20260911'
OUT=ROOT/'results/v58_mismatch_ranking_diagnosis'


def read(p):return json.loads(p.read_text(encoding='utf-8'))


def rows(p):
    with p.open(encoding='utf-8',newline='') as f:return list(csv.DictReader(f))


def write_csv(name,data):
    with (OUT/name).open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)


def ranking(data,score):
    raw=[r for r in data if r['raw_solver_reported']=='True']
    truth=sum(r['molecular_truth']=='True' for r in data)
    reportable=sum(r.get('reportable_truth')=='True' for r in data)
    positives=sum(r['molecular_truth']=='True' for r in raw);negatives=len(raw)-positives
    groups=defaultdict(list)
    for r in raw:groups[float(r[score])].append(r)
    tp=fp=reportable_tp=0;ap=auc_numerator=0.;curve=[]
    for value,g in sorted(groups.items(),reverse=True):
        p=sum(r['molecular_truth']=='True' for r in g);n=len(g)-p
        auc_numerator+=p*(negatives-fp-n+.5*n)
        tp+=p;fp+=n
        reportable_tp+=sum(r['molecular_truth']=='True' and r.get('reportable_truth')=='True' for r in g)
        ap+=(p/positives)*(tp/(tp+fp)) if positives else 0
        curve.append(dict(threshold=value,TP=tp,FP=fp,FDR=fp/(tp+fp),precision=tp/(tp+fp),
                          all_truth_recall=tp/truth,reportable_truth_recall=reportable_tp/reportable if reportable else None,
                          FN=truth-tp,TP_retention=tp/positives,coverage=(tp+fp)/len(raw)))
    summary=dict(raw_TP=positives,raw_FP=negatives,solver_FN=truth-positives,
                 AUROC=auc_numerator/(positives*negatives) if positives*negatives else None,
                 average_precision=ap,raw_precision=positives/len(raw),
                 zero_score_true=sum(r['molecular_truth']=='True' for r in groups.get(0.,[])),
                 zero_score_false=sum(r['molecular_truth']!='True' for r in groups.get(0.,[])))
    for level in (.01,.05):
        eligible=[r for r in curve if r['FDR']<=level]
        best=max(eligible,key=lambda r:(r['TP'],r['TP']+r['FP'])) if eligible else None
        summary[f'descriptive_HOLD_max_recall_at_FDR{int(level*100)}']=best['all_truth_recall'] if best else 0.
    return summary,curve


def main():
    OUT.mkdir(exist_ok=True)
    plan=read(SOURCE/'cleanup_plan.json')
    for name,h in plan['source_files'].items():assert hashlib.sha256((SOURCE/name).read_bytes()).hexdigest()==h
    manifest={(r['severity'],int(r['candidate_index'])):r for r in rows(SOURCE/'spectral_perturbation_manifest.csv')}
    all_rows=[]
    for dataset in plan['completed']:
        severity,base=dataset.split('__')
        if base.startswith('HOLD'):all_rows+=rows(SOURCE/severity/base/'molecular_false_negative_records.csv')
    rankings=[];curves=[];truth_records=[]
    for s in ('MILD','MODERATE'):
        data=[r for r in all_rows if r['severity']==s]
        threshold=read(SOURCE/s/'local_frozen_thresholds.json')['global_thresholds']['rho_zero']['FDR5']['threshold']
        for k in ('ALL','50','125','175'):
            subset=data if k=='ALL' else [r for r in data if r['K']==k]
            for score in ('rho_zero','X_hat'):
                summary,curve=ranking(subset,score)
                common=dict(severity=s,K=k,score=score,completed_cases=len({r['dataset_id'] for r in subset}),
                            complete_severity=s=='MILD')
                rankings.append({**common,**summary})
                curves += [{**common,**c} for c in curve]
        for r in data:
            if r['molecular_truth']!='True':continue
            indices=json.loads(r['candidate_indices']);assert len(indices)==1,'TRUTH_NOT_SINGLETON'
            m=manifest[s,indices[0]];raw=r['raw_solver_reported']=='True'
            kept=raw and float(r['rho_zero'])>=threshold
            state='solver_miss' if not raw else 'retained_true' if kept else 'filter_loss'
            truth_records.append(dict(severity=s,dataset_id=r['dataset_id'],lipid_name=r['lipid_name'],K=r['K'],
                replicate=r['replicate'],state=state,X_hat=float(r['X_hat']),rho_zero=float(r['rho_zero']) if raw else None,
                cone_isolation=float(r['cone_isolation']),max_fragment_cosine=float(r['max_fragment_cosine']),
                collective_gain=float(r['collective_gain']),abundance_multiplier=float(r['abundance_multiplier']),
                full_spectral_change=1-float(m['full_cosine_A_target_vs_A_solver']),
                fragment_spectral_change=1-float(m['fragment_cosine_A_target_vs_A_solver']),
                achieved_dropout_rate=float(m['achieved_dropout_rate'])))
    # One row per identity/severity avoids treating repeated contexts as independent lipids.
    groups=defaultdict(list)
    for r in truth_records:groups[r['severity'],r['lipid_name']].append(r)
    identities=[]
    features=['cone_isolation','max_fragment_cosine','collective_gain','full_spectral_change','fragment_spectral_change','achieved_dropout_rate']
    for (s,name),g in groups.items():
        raw=sum(r['state']!='solver_miss' for r in g);loss=sum(r['state']=='filter_loss' for r in g)
        identities.append(dict(severity=s,lipid_name=name,contexts=len(g),raw_TP_contexts=raw,
            solver_miss_contexts=len(g)-raw,filter_loss_contexts=loss,loss_fraction_of_raw_TP=loss/raw if raw else None,
            comparison_group='no_raw_TP' if not raw else 'any_filter_loss' if loss else 'never_filter_loss',
            **{f:statistics.median(r[f] for r in g) for f in features}))
    feature_summary=[]
    for s in ('MILD','MODERATE'):
        for group in ('any_filter_loss','never_filter_loss','no_raw_TP'):
            g=[r for r in identities if r['severity']==s and r['comparison_group']==group]
            for f in features:feature_summary.append(dict(severity=s,group=group,unique_identities=len(g),feature=f,
                median=statistics.median(r[f] for r in g) if g else None))
    write_csv('ranking_summary.csv',rankings);write_csv('descriptive_HOLD_curves.csv',curves)
    write_csv('truth_loss_contexts.csv',truth_records);write_csv('truth_identity_summary.csv',identities)
    write_csv('identity_feature_summary.csv',feature_summary)
    (OUT/'provenance.json').write_text(json.dumps(dict(status='PASS',source_files_verified=len(plan['source_files']),
        source_snapshot_commit='4f6eda6308698eb4949d0b0ec052b5cc606e4c49',source_cleanup_plan_sha256=hashlib.sha256((SOURCE/'cleanup_plan.json').read_bytes()).hexdigest(),
        limitations=['HOLD envelope uses labels retrospectively; it is not a deployable cutoff or independent validation.',
        'MODERATE incomplete and K/replicate-unbalanced; no severity effect estimate.',
        'Identity summaries are descriptive, not adjusted causal effects or enrichment p-values.',
        'No training, rho recomputation, threshold modification or extra asset scan.']),indent=2)+'\n')
    print(json.dumps([r for r in rankings if r['K']=='ALL'],indent=2))


if __name__=='__main__':main()
