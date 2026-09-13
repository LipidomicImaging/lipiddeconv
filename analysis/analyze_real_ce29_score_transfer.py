"""Descriptive cached-data diagnosis of fixed-score transfer; no fit or new cut."""
import csv
import json
import math
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'.venv/real_ce29_packages'),str(ROOT/'analysis')]
import numpy as np
from physical_block_monotone_score import read,write,sha,score_rows,review_predictions


def quantiles(values):
    a=np.asarray(values,dtype=float)
    return dict(zip(('min','p10','p25','median','p75','p90','max'),map(float,np.quantile(a,[0,.1,.25,.5,.75,.9,1])))) if len(a) else None


def main():
    out=ROOT/'results/real_ce29_score_transfer_diagnosis'
    assert not (out/'diagnosis.json').exists()
    real='results/real_ce29_joint_screening_v2'
    old='results/physical_block_prediction_v1'
    new='results/physical_block_score_correction_v2/new_composition_case'
    model=read(ROOT/real/'model.json');cuts=read(ROOT/real/'thresholds.json')
    assert model['coef'][2]==0
    sources={}; rows=[]; summary={}; replays={}
    def load(path):
        p=ROOT/path;sources[path]=sha(p);return read(p)
    load(real+'/model.json');load(real+'/thresholds.json')
    sets=[('SIM_DEV',old+'/prepared/DEV_TRAIN_records.json',old+'/cases/DEV_TRAIN/features.json',None),
          ('SIM_CHECK',old+'/prepared/CHECK_records.json',old+'/cases/CHECK/features.json',None),
          ('SIM_NEW',new+'/nnls_rho/molecular_records.json',new+'/physical_features/features.json',None),
          ('REAL_ISTA_FULL',real+'/molecular_records.json',real+'/physical_features.json',None),
          ('REAL_ISTA_4500','results/real_ce29_nnls_joint_screening_v2/partial_004500/molecular_records.json',real+'/physical_features.json','ISTA'),
          ('REAL_NNLS_4500','results/real_ce29_nnls_joint_screening_v2/partial_004500/molecular_records.json',real+'/physical_features.json','NNLS')]
    bounds=load('results/physical_block_score_correction_v2/source_abundance_comparison.json')
    truth_lo=bounds['old_CHECK']['true_mean_coefficient_quantiles']['min']
    truth_hi=bounds['new_composition']['true_mean_coefficient_quantiles']['max']
    for label,path,featpath,method in sets:
        rs=load(path); fs=load(featpath)
        if method:rs=[r for r in rs if r['method']==method]
        flookup={f['molecular_name']:f for f in fs}
        if method is None:
            values=score_rows(rs,fs,model);replays[label]=review_predictions(rs,fs,model,values)
        else:values=[r['joint_score'] for r in rs]
        local=[]
        for row,value in zip(rs,values):
            if not row['raw_solver_reported']:continue
            if label=='REAL_ISTA_FULL':assert math.isclose(value,row['joint_score'],abs_tol=1e-12)
            f=flookup[row['lipid_name']]
            a=(math.log10(row['X_hat'])-model['base']['mean'][0])/model['base']['scale'][0]
            s=(f['predictive_gain']-model['auxiliary']['mean'][0])/model['auxiliary']['scale'][0]
            c=(f['positive_block_fraction']-model['auxiliary']['mean'][1])/model['auxiliary']['scale'][1]
            term_a=model['coef'][0]*a+model['coef'][1]*a*a
            term_s=model['coef'][3]*s;term_c=model['coef'][4]*c
            logit=math.fsum((model['intercept'],term_a,term_s,term_c))
            replay=1/(1+math.exp(-logit)) if logit>=0 else math.exp(logit)/(1+math.exp(logit))
            assert abs(replay-value)<=1e-12
            item=dict(dataset=label,lipid_name=row['lipid_name'],truth=row['molecular_truth'],X_hat=row['X_hat'],rho=row['rho_zero'],
                      S=f['predictive_gain'],C=f['positive_block_fraction'],supported_blocks=f['supported_block_count'],
                      score=value,standardized_log_abundance=a,abundance_logit=term_a,S_logit=term_s,C_logit=term_c,logit=logit)
            for k in ('pool','fdp5','fdp1'):item[k]=value>=cuts[k+'_threshold']
            rows.append(item);local.append(item)
        totals={}
        for group,items in [('reported',local)]+([('true_reported',[r for r in local if r['truth'] is True]),
                                                ('false_reported',[r for r in local if r['truth'] is False])] if label.startswith('SIM') else []):
            totals[group]=dict(n=len(items),**{v:quantiles([r[v] for r in items]) for v in ('X_hat','S','C','score','abundance_logit','S_logit','C_logit')},
                S_nonpositive=sum(r['S']<=0 for r in items),C_zero=sum(r['C']==0 for r in items),
                below_simulated_truth_abundance_floor=sum(r['X_hat']<truth_lo for r in items),
                above_simulated_truth_abundance_ceiling=sum(r['X_hat']>truth_hi for r in items),
                legacy_rho_retained=sum(r['rho'] is not None and r['rho']>=.001 for r in items),
                rho_missing=sum(r['rho'] is None for r in items),
                selected={k:sum(r[k] for r in items) for k in ('pool','fdp5','fdp1')})
        summary[label]=totals
    cached_q={
      'SIM_DEV':load('results/small_mismatch_nnls_first_case/case/rho/candidate_0000.json')['result'],
      'SIM_NEW':load(new+'/nnls_rho/candidate_records.json')[0]['rho_details'],
      'REAL':next(iter(load(real+'/rho.json').values()))}
    residuals={k:dict(q_star=v['q_star'],observation_norm=math.sqrt(v['signal_norm2_plus_epsilon']-1e-12),
                     best_full_library_mean_relative_residual=math.sqrt(v['q_star']/(v['signal_norm2_plus_epsilon']-1e-12)))
               for k,v in cached_q.items()}
    for label,path in [('REAL',real+'/prepared.npz'),('SIM_DEV',old+'/prepared/DEV_TRAIN.npz'),('SIM_NEW',new+'/prepared/portable.npz')]:
        sources[path]=sha(ROOT/path)
        with np.load(ROOT/path,allow_pickle=False) as z:A,b=z['A'],z['b']
        zero=(A==0).all(axis=1)
        residuals[label].update(zero_library_channels=int(zero.sum()),
                               mean_signal_fraction_on_zero_library_channels=float(b[zero]@b[zero]/(b@b)),
                               top10_channel_energy_fraction=float(np.sort(b*b)[-10:].sum()/(b@b)))
    realrows={r['lipid_name']:r for r in rows if r['dataset']=='REAL_ISTA_FULL'}
    matched={}
    for label in ('SIM_DEV','SIM_NEW'):
        pairs=[(r,realrows[r['lipid_name']]) for r in rows if r['dataset']==label and r['truth'] is True and r['lipid_name'] in realrows]
        matched[label]=dict(n=len(pairs),real_to_simulated_estimated_abundance_ratio=quantiles([b['X_hat']/a['X_hat'] for a,b in pairs]),
                           real_S_below_simulated=sum(b['S']<a['S'] for a,b in pairs),
                           real_C_below_simulated=sum(b['C']<a['C'] for a,b in pairs),
                           names=[a['lipid_name'] for a,b in pairs],
                           limitation='Matched nominal identities; real identity presence remains unverified, not a paired truth experiment')
    peak_a=-model['coef'][0]/(2*model['coef'][1])
    diagnostic=dict(status='DESCRIPTIVE_CACHED_DIAGNOSIS',same_fixed_model=True,no_refit_or_new_threshold=True,
                    simulated_truth_abundance_range_reference=[truth_lo,truth_hi],
                    abundance_logit_peak_X=10**(model['base']['mean'][0]+model['base']['scale'][0]*peak_a),
                    summary=summary,best_mean_fit=residuals,nominal_identity_matched_description=matched,
                    strongest_real_identities=sorted(realrows.values(),key=lambda r:-r['X_hat'])[:12],
                    actual_real_FDR=None,actual_real_recall=None,source_hashes=sources,score_replays=replays,
                    interpretation_limit='Distributions and exact fixed-score arithmetic; mismatch cause and real false-negative labels not identified')
    write(out/'diagnosis.json',diagnostic);write(out/'score_decomposition.json',rows)
    with (out/'score_decomposition.csv').open('x',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    print(json.dumps({'summary':{k:{g:{v:r[v] for v in ('n','X_hat','S','C','S_nonpositive','below_simulated_truth_abundance_floor','above_simulated_truth_abundance_ceiling','legacy_rho_retained','selected')}
                                 for g,r in groups.items()} for k,groups in summary.items()},
                      'residuals':residuals,'peak_X':diagnostic['abundance_logit_peak_X'],
                      'strongest_real':diagnostic['strongest_real_identities'][:5],'matched':matched},ensure_ascii=False))


if __name__=='__main__':main()
