"""Cache-only user-requested partial NNLS/ISTA comparison on identical pixels."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import sys
from datetime import datetime, timezone

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'.venv/real_ce29_packages'),str(ROOT/'analysis')]
import numpy as np
from scipy.special import expit
from physical_block_monotone_score import read, sha, write, require, validate_model


def main(count):
    parent=ROOT/'results/real_ce29_joint_screening_v2'
    run=ROOT/'results/real_ce29_nnls_joint_screening_v2'
    out=run/f'partial_{count:06d}'
    require(not out.exists(), 'EXISTING_SNAPSHOT_PRESERVED')
    d=read(run/'input.json'); status=read(run/'nnls/status.json')
    require(sha(run/'input.json')==read(run/'input_seal.json')['sha256'], 'INPUT_SEAL_CHANGED')
    require(0<count<=status['pixels_done'] and count%250==0, 'INCOMPLETE_REQUESTED_PREFIX')
    for n,h in d['parent_files'].items(): require(sha(parent/n)==h, 'PARENT_CHANGED:'+n)
    binding=read(run/'nnls/binding.json'); require(binding['fingerprint']==status['fingerprint'], 'RUN_BINDING_CHANGED')
    require(binding['scientific']['retained_indices']==list(range(391)) and
            binding['scientific']['source_binding']==dict(input_sha256=sha(run/'input.json')), 'FULL_LIBRARY_SOURCE_CHANGED')
    sources=read(parent/'input.json')['source_files']
    for k in ('X','mask'): require(sha(sources[k]['path'])==sources[k]['sha256'], 'SOURCE_CHANGED:'+k)
    mask=np.load(sources['mask']['path'],allow_pickle=False)
    ista=np.load(sources['X']['path'],allow_pickle=False)[:,mask][:,:count]
    require(int(mask.sum())==15837 and ista.shape==(391,count), 'PIXEL_ORDER_CHANGED')
    X=np.zeros((391,count),dtype=np.float32); hashes={}; kkt_max=0.
    for start in range(0,count,250):
        stop=start+250; stem=f'block_{start:06d}_{stop:06d}'; blockdir=run/'nnls/nnls_blocks'
        record=read(blockdir/(stem+'.json')); arrayfile=blockdir/(stem+'.npz')
        require(record['fingerprint']==binding['fingerprint'] and record['start']==start and record['stop']==stop
                and record['retained_indices']==list(range(391)) and record['all_pixels_checked_before_float32'] is True
                and record['array_sha256']==sha(arrayfile), 'BLOCK_PROVENANCE_FAILED')
        require(all(math.isfinite(v) and v>=0 for v in record['KKT'].values()) and record['KKT']['max_bound_ratio']<=1,
                'BLOCK_KKT_FAILED')
        kkt_max=max(kkt_max,record['KKT']['max_bound_ratio'])
        with np.load(arrayfile,allow_pickle=False) as z: block=z['X_hat']
        require(block.shape==(391,250) and block.dtype==np.float32 and np.isfinite(block).all() and (block>=0).all(),
                'INVALID_BLOCK_ARRAY')
        X[:,start:stop]=block
        for ext in ('npz','json'):
            p=blockdir/(stem+'.'+ext); hashes[p.relative_to(ROOT).as_posix()]=sha(p)
    model=validate_model(read(parent/'model.json')); cuts=read(parent/'thresholds.json')
    require(model['coef'][2]==0., 'RHO_TERM_NOT_ZERO_REQUIRES_COMPLETE_RHO')
    features={r['molecular_name']:r for r in read(parent/'physical_features.json')}
    molecular=read(parent/'molecular_records_input.json'); candidate=read(parent/'candidate_records_input.json')
    means={'NNLS':X.mean(axis=1,dtype=float),'ISTA':ista.mean(axis=1,dtype=float)}
    rho=read(parent/'rho.json'); rows=[]; candidates=[]
    for method,m in means.items():
        for j,old in enumerate(candidate):
            candidates.append(dict(method=method,candidate_index=j,candidate_id=old['candidate_id'],lipid_name=old['lipid_name'],
                                   X_hat=float(m[j]),raw_solver_reported=bool(m[j]>.001),molecular_truth=None))
        for old in molecular:
            ids=old['candidate_indices']; active=[j for j in ids if m[j]>.001]; f=features[old['lipid_name']]
            row=dict(method=method,lipid_name=old['lipid_name'],lipid_class=old['lipid_class'],candidate_indices=ids,
                     reported_candidate_indices=active,X_hat=sum(float(m[j]) for j in active),raw_solver_reported=bool(active),
                     molecular_truth=None,predictive_gain=f['predictive_gain'],positive_block_fraction=f['positive_block_fraction'],
                     rho_zero=None,rho_status='UNREPORTED',joint_score=None)
            if active:
                if all(str(j) in rho for j in active):
                    row.update(rho_zero=max(rho[str(j)]['rho_zero'] for j in active),rho_status='CACHED_FULL_FOREGROUND')
                else: row['rho_status']='NOT_COMPUTED_ZERO_COEFFICIENT'
                a=(math.log10(max(row['X_hat'],1e-12))-model['base']['mean'][0])/model['base']['scale'][0]
                s=(f['predictive_gain']-model['auxiliary']['mean'][0])/model['auxiliary']['scale'][0]
                c=(f['positive_block_fraction']-model['auxiliary']['mean'][1])/model['auxiliary']['scale'][1]
                # Algebraically omit the exactly zero rho coefficient, never substitute a made-up rho value.
                terms=(a,a*a,s,c); coefs=[model['coef'][i] for i in (0,1,3,4)]
                logit=math.fsum([model['intercept']]+[v*w for v,w in zip(terms,coefs)])
                value=1/(1+math.exp(-logit)) if logit>=0 else math.exp(logit)/(1+math.exp(logit))
                other=float(expit(np.asarray(terms)@np.asarray(coefs)+model['intercept']))
                require(abs(value-other)<=1e-12,'INDEPENDENT_SCORE_REPLAY_FAILED')
                row['joint_score']=value
            for k in ('pool','fdp5','fdp1'):
                row['retained_by_'+k]=bool(active and row['joint_score']>=cuts[k+'_threshold'])
            rows.append(row)
    comparison={}
    for k in ('raw','pool','fdp5','fdp1'):
        field='raw_solver_reported' if k=='raw' else 'retained_by_'+k
        sets={method:{r['lipid_name'] for r in rows if r['method']==method and r[field]} for method in means}
        n,i=sets['NNLS'],sets['ISTA']
        comparison[k]=dict(NNLS=len(n),ISTA=len(i),common=len(n&i),NNLS_names=sorted(n),ISTA_names=sorted(i),
                           NNLS_only=sorted(n-i),ISTA_only=sorted(i-n),threshold=None if k=='raw' else cuts[k+'_threshold'])
    # A second alias aggregation and set check, independent of the stored molecular row assembly.
    for method,m in means.items():
        groups={}
        for item in candidates:
            if item['method']==method and item['raw_solver_reported']:
                groups.setdefault(item['lipid_name'],[]).append(item['X_hat'])
        actual={r['lipid_name']:r for r in rows if r['method']==method and r['raw_solver_reported']}
        require(set(actual)==set(groups) and all(math.isclose(actual[n]['X_hat'],math.fsum(v),rel_tol=1e-14) for n,v in groups.items()),
                'INDEPENDENT_ALIAS_ACCOUNTING_FAILED')
    out.mkdir()
    write(out/'molecular_records.json',rows);write(out/'candidate_records.json',candidates)
    for r in rows:
        if r['raw_solver_reported']:
            require(r['joint_score'] is not None and math.isfinite(r['joint_score']),'INVALID_SCORE')
    report=dict(status='REVIEWED_PARTIAL_SNAPSHOT',created_utc=datetime.now(timezone.utc).isoformat(),
                completed_pixels=count,total_foreground_pixels=15837,pixel_coverage=count/15837,
                pixel_selection='First completed contiguous foreground-flat prefix; same positions for both solvers, not a random sample',
                evidence_scope='Subset foreground-mean abundance with unchanged cached whole-foreground S/C; no subset-specific spectral feature refit',
                comparison=comparison,actual_FDR=None,TP=None,FP=None,FN=None,recall=None,truth_available=False,
                final=False,main_run_unchanged=True,model_sha256=sha(parent/'model.json'),thresholds_sha256=sha(parent/'thresholds.json'),
                physical_features_sha256=sha(parent/'physical_features.json'),input_sha256=sha(run/'input.json'),
                no_training_or_rho_or_solver_run=True,zero_rho_term_omitted_exactly=True,
                independently_replayed_scores_and_alias_accounting=True,blocks_reviewed=count//250,max_original_KKT_bound_ratio=kkt_max,
                block_hashes=hashes,script_sha256=sha(Path(__file__)))
    write(out/'report.json',report)
    fields=['method','lipid_name','lipid_class','raw_solver_reported','X_hat','rho_zero','rho_status','predictive_gain',
            'positive_block_fraction','joint_score','retained_by_pool','retained_by_fdp5','retained_by_fdp1']
    with (out/'molecular_screening.csv').open('x',encoding='utf-8-sig',newline='') as stream:
        w=csv.DictWriter(stream,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(rows)
    labels={'raw':'原始报告','pool':'固定候选池阈值','fdp5':'固定 DEV 5% 阈值','fdp1':'固定 DEV 1% 阈值'}
    lines=['# 真实 CE29：已完成像素的阶段性 NNLS 对照','',f'快照覆盖 {count:,} / 15,837 个前景像素（{count/15837:.1%}）。只读取已完整保存且 hash/KKT 通过的块；主任务继续运行。','',
           '| 筛选档位 | 同位置 ISTA | NNLS | 两者共同 |','|---|---:|---:|---:|']
    for k,v in comparison.items(): lines.append(f"| {labels[k]} | {v['ISTA']} | {v['NNLS']} | {v['common']} |")
    lines += ['', '两种方法的丰度都只取相同的已完成像素；继续使用全图已缓存的谱块 S/C、原模型和三档阈值。该区域是按处理顺序截取，不是随机代表性样本；这些数量是阶段性结果，不能直接外推到全图。', '',
              '真实身份真值未知，实际 FDR 和 recall 均未知。模型 rho 系数精确为0，评分时代数省略该项；未缓存的 rho 明确标记 NOT_COMPUTED，不虚构0，不启动新rho计算。', '',
              '独立标量/向量评分及别名汇总检查通过。未修改运行中的源代码、模型、阈值或记录，无数据删除。完整名单、分数及阈值标志见 molecular_screening.csv；每档具体交集和差集见 report.json。']
    (out/'analysis_record.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({'pixels':count,'comparison':{k:{q:v[q] for q in ('NNLS','ISTA','common')} for k,v in comparison.items()},
                      'output':str(out)},ensure_ascii=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--pixels',type=int,required=True)
    main(parser.parse_args().pixels)
