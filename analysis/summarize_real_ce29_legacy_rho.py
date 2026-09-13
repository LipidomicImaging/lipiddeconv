"""Apply the historical rho>=1e-3 gate to real CE29 full/partial cached reports."""
import csv
import json
import math
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'.venv/real_ce29_packages'),str(ROOT/'analysis'),str(ROOT/'src')]
from physical_block_monotone_score import read,write,sha,require
import numpy as np
from threadpoolctl import threadpool_limits
from rho_zero import rho_zero_from_weighted_case


def main():
    parent=ROOT/'results/real_ce29_joint_screening_v2'
    partial=ROOT/'results/real_ce29_nnls_joint_screening_v2/partial_004500'
    out=ROOT/'results/real_ce29_legacy_rho_threshold'
    require(not (out/'report.json').exists(),'EXISTING_RESULT_PRESERVED')
    original=read(parent/'input.json')
    require(sha(parent/'input.json')==read(parent/'input_seal.json')['sha256'],'INPUT_CHANGED')
    require(sha(parent/'prepared.npz')==original['files']['prepared.npz'],'PREPARED_CHANGED')
    expected=read(parent/'final_storage_manifest.json')['files']
    for name in ('molecular_records.json','rho.json'):
        require(sha(parent/name)==expected[name]['sha256'],'PARENT_RESULT_CHANGED')
    require(read(parent/'independent_final_review.json')['status']=='PASS','PARENT_REVIEW_FAILED')
    require(sha(partial/'molecular_records.json')==read(partial/'artifact_hashes.json')['molecular_records.json'],
            'PARTIAL_CHANGED')
    require(read(partial/'snapshot_review.json')['status']=='PASS','PARTIAL_REVIEW_FAILED')
    full=read(parent/'molecular_records.json'); rows=read(partial/'molecular_records.json')
    groups={'ISTA_FULL':full, 'ISTA_PARTIAL_4500':[r for r in rows if r['method']=='ISTA'],
            'NNLS_PARTIAL_4500':[r for r in rows if r['method']=='NNLS']}
    rho=read(parent/'rho.json')
    missing=sorted({j for rs in groups.values() for r in rs if r['raw_solver_reported']
                    for j in r['reported_candidate_indices'] if str(j) not in rho})
    binding=dict(rule='raw reported molecular identity AND max reported-alias rho_zero >= 0.001',
                 threshold=.001, comparator='>=', input_sha256=sha(parent/'input.json'),
                 parent_records_sha256=sha(parent/'molecular_records.json'),parent_rho_sha256=sha(parent/'rho.json'),
                 partial_records_sha256=sha(partial/'molecular_records.json'),partial_pixels=4500,
                 prepared_sha256=sha(parent/'prepared.npz'), source_rho_sha256=sha(ROOT/'src/rho_zero.py'),
                 script_sha256=sha(Path(__file__)),missing_candidate_indices=missing,
                 evidence_scope='unchanged whole-foreground observed mean spectrum for all comparisons',
                 legacy_rule_source='analysis/run_small_mismatch_nnls_first_case.py:319-323',
                 legacy_rule_source_sha256=sha(ROOT/'analysis/run_small_mismatch_nnls_first_case.py'),
                 thresholds_not_selected_from_real_data=True, no_main_run_mutation=True)
    write(out/'input.json',binding);write(out/'input_seal.json',dict(sha256=sha(out/'input.json')))
    with np.load(parent/'prepared.npz',allow_pickle=False) as z:A,b=z['A'],z['b']
    # Compute only missing measurements using the exact frozen function, outside the active run directory.
    for j in missing:
        p=out/'additional_rho'/f'candidate_{j:03d}.json'
        if p.exists():
            saved=read(p);require(saved['input_sha256']==sha(out/'input.json') and saved['candidate_index']==j,'CACHE_BINDING_CHANGED')
            value=saved['values']
        else:
            value=rho_zero_from_weighted_case(A,b,j)
            require(all(math.isfinite(v) for v in value.values()),'NONFINITE_RHO')
            write(p,dict(candidate_index=j,input_sha256=sha(out/'input.json'),values=value))
        rho[str(j)]=value
    output=[];summary={}
    for label,rs in groups.items():
        active=[r for r in rs if r['raw_solver_reported']]; retained=[]
        for row in active:
            values=[rho[str(j)] for j in row['reported_candidate_indices']]
            for v in values:
                independent=max(0.,(v['q_deleted']-v['q_star'])/v['signal_norm2_plus_epsilon'])
                require(math.isclose(independent,v['rho_zero'],rel_tol=1e-12,abs_tol=1e-18),'RHO_FORMULA_REPLAY_FAILED')
            r=max(v['rho_zero'] for v in values); keep=r>=.001
            if keep:retained.append(row['lipid_name'])
            output.append(dict(dataset=label,lipid_name=row['lipid_name'],lipid_class=row['lipid_class'],
                               X_hat=row['X_hat'],reported_candidate_indices=row['reported_candidate_indices'],
                               rho_zero=r,retained_by_legacy_rho=keep,joint_score=row['joint_score'],
                               retained_by_joint_pool=row['retained_by_pool'],retained_by_joint_dev5=row['retained_by_fdp5'],
                               retained_by_joint_dev1=row['retained_by_fdp1'],molecular_truth=None))
        legacy=set(retained)
        summary[label]=dict(raw=len(active),retained=len(legacy),removed=len(active)-len(legacy),names=sorted(legacy),
                            exact_threshold_ties=sum(r['dataset']==label and r['rho_zero']==.001 for r in output),
                            joint_comparison={k:dict(retained=sum(r['retained_by_'+k] for r in active),
                                                    common=sorted(legacy & {r['lipid_name'] for r in active if r['retained_by_'+k]}))
                                              for k in ('pool','fdp5','fdp1')})
    write(out/'molecular_records.json',output)
    with (out/'molecular_screening.csv').open('x',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(output[0]));w.writeheader();w.writerows(output)
    report=dict(status='REVIEWED_LEGACY_THRESHOLD_COMPARISON',rule=binding['rule'],summary=summary,
                actual_FDR=None,TP=None,FP=None,FN=None,recall=None,truth_available=False,
                partial_NNLS_not_final=True,new_rho_count=len(missing),cached_rho_count=len(rho)-len(missing),
                full_foreground_rho_used=True,no_main_run_mutation=True,input_sha256=sha(out/'input.json'))
    write(out/'report.json',report)
    lines=['# 真实 CE29：原固定 rho 阈值对照','', '使用历史规则 rho_zero >= 0.001，仅在原报告门槛通过的分子身份中筛选。阈值没有根据本次结果调整。','',
           '| 数据范围 | 原始报告 | rho 保留 | 剔除 |','|---|---:|---:|---:|']
    for k,v in summary.items():lines.append(f"| {k} | {v['raw']} | {v['retained']} | {v['removed']} |")
    lines += ['', 'ISTA_FULL 为全15,837个前景像素；两个 PARTIAL_4500 使用相同4500像素的报告名单和丰度。三者 rho 都基于相同全图前景平均观测谱，复用已有候选缓存，仅为缺失候选调用原rho函数；没有重算像素求解或改变正在运行的NNLS。', '',
              '真实 FDR/recall 未知。rho 固定阈值不是实际1% FDR保证；此结果也不替代联合模型结果。NNLS整图仍待完成。','', '完整 ISTA 通过旧 rho 门槛的身份：','']
    for r in sorted((r for r in output if r['dataset']=='ISTA_FULL' and r['retained_by_legacy_rho']),key=lambda r:-r['rho_zero']):
        lines.append(f"- {r['lipid_name']}：rho={r['rho_zero']:.9g}")
    (out/'analysis_record.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({k:{q:v[q] for q in ('raw','retained','removed','names')} for k,v in summary.items()},ensure_ascii=False))


if __name__=='__main__':
    with threadpool_limits(limits=1):main()
