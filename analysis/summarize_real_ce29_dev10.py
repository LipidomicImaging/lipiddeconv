"""Transfer the saved DEV<10% cutoff to reviewed real CE29 scores, cache only."""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path
import shutil

from physical_block_monotone_score import read, write, sha, require, scalar_predictions, select_names

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/real_ce29_dev10_threshold'
QUERY = ROOT / 'results/cached_identity_evidence_v1/user_FDP10_recall_query.json'
PARENTS = {
    'ISTA': ROOT / 'results/real_ce29_joint_screening_v2',
    'NNLS': ROOT / 'results/real_ce29_nnls_joint_screening_v2',
}


def main():
    require(not OUT.exists(), 'EXISTING_ANALYSIS_PRESERVED')
    query = read(QUERY)
    require(query['status'] == 'DESCRIPTIVE_QUERY_COMPLETE', 'QUERY_INCOMPLETE')
    require(query['strict_risk_rule'] == 'FP*10 < selected_count; strict below10%, full score ties',
            'QUERY_RULE_CHANGED')
    primary = query['DEV_rule']['threshold']
    require(primary == 0.5406530976316042, 'UNEXPECTED_DEV_CUTOFF')
    cuts = {'DEV10_TRANSFER': primary}
    source_paths = [Path(__file__), ROOT / 'analysis/physical_block_monotone_score.py', QUERY]
    for case, item in query['cases'].items():
        source = ROOT / 'results/cached_identity_evidence_v1/cases' / case / 'screen.json'
        require(sha(source) == item['source_sha256'], 'QUERY_SOURCE_CHANGED:' + case)
        source_paths.append(source)
        cuts[case + '_OUTCOME_GUIDED_SENSITIVITY'] = item['case_specific_descriptive_maximum']['threshold']
    model_path = PARENTS['ISTA'] / 'model.json'
    old_cuts_path = PARENTS['ISTA'] / 'thresholds.json'
    model, old_cuts = read(model_path), read(old_cuts_path)
    require(sha(model_path) == sha(ROOT / 'results/physical_block_score_correction_v2/model/model.json'),
            'MODEL_DIFFERS_FROM_SIMULATION')
    source_paths += [model_path, old_cuts_path]
    rows_by_method, score_errors = {}, {}
    for method, parent in PARENTS.items():
        manifest = read(parent / 'final_storage_manifest.json')
        review_path = parent / 'independent_final_review.json'
        require(read(review_path)['status'] == 'PASS', 'PARENT_NOT_REVIEWED:' + method)
        for filename in ('molecular_records.json', 'report.json'):
            require(sha(parent / filename) == manifest['files'][filename]['sha256'],
                    'PARENT_ARTIFACT_CHANGED:' + filename)
        source_paths += [parent / name for name in
                         ('molecular_records.json', 'report.json', 'final_storage_manifest.json',
                          'independent_final_review.json')]
        if method == 'NNLS':
            completion = read(parent / 'final_completion.json')
            require(completion['status'] == 'COMPLETE_REVIEWED' and completion['all_pixels'] == 15837
                    and completion['report_sha256'] == sha(parent / 'report.json')
                    and completion['review_sha256'] == sha(review_path), 'NNLS_COMPLETION_BINDING')
            source_paths.append(parent / 'final_completion.json')
        else:
            require(read(parent / 'report.json')['status'] == 'COMPLETE_REVIEWED', 'ISTA_COMPLETION')
        rows = read(parent / 'molecular_records.json')
        require(len(rows) == len({r['lipid_name'] for r in rows}) == 377, 'IDENTITY_MEMBERSHIP')
        require(all(r['molecular_truth'] is None for r in rows), 'REAL_TRUTH_MUST_STAY_UNKNOWN')
        require(sum(r['raw_solver_reported'] for r in rows) == {'ISTA': 76, 'NNLS': 78}[method],
                'RAW_MEMBERSHIP_CHANGED')
        features = [dict(molecular_name=r['lipid_name'], predictive_gain=r['predictive_gain'],
                         positive_block_fraction=r['positive_block_fraction']) for r in rows]
        expected = scalar_predictions(rows, features, model)
        require(all((s is None) == (r['joint_score'] is None) for s, r in zip(expected, rows)),
                'SCORE_NULL_MEMBERSHIP')
        error = max(abs(s - r['joint_score']) for s, r in zip(expected, rows) if s is not None)
        require(error <= 1e-12, 'INDEPENDENT_SCALAR_SCORE_REPLAY')
        score_errors[method] = error
        rows_by_method[method] = rows

    sources = {p.relative_to(ROOT).as_posix(): sha(p) for p in source_paths}
    source_model_sha = sha(model_path)
    summaries, records, comparisons = {}, [], {}
    for method, rows in rows_by_method.items():
        scores = [r['joint_score'] for r in rows]
        names_by_cut = {key: select_names(rows, scores, cut) for key, cut in cuts.items()}
        active = [r for r in rows if r['raw_solver_reported']]
        primary_set = set(names_by_cut['DEV10_TRANSFER'])
        old_names = {}
        for key in ('pool', 'fdp5', 'fdp1'):
            old_names[key] = set(select_names(rows, scores, old_cuts[key + '_threshold']))
            require(old_names[key] == {r['lipid_name'] for r in rows if r['retained_by_' + key]},
                    'OLD_SELECTION_CHANGED')
            require(old_names[key] <= primary_set, 'NESTING_CHANGED')
        summaries[method] = dict(
            raw=len(active), retained=len(primary_set), removed=len(active) - len(primary_set),
            coverage=len(primary_set) / len(active), names=sorted(primary_set),
            old_counts={k: len(v) for k, v in old_names.items()},
            added_vs_old={k: sorted(primary_set - v) for k, v in old_names.items()},
            exact_threshold_ties=sum(r['joint_score'] == primary for r in active),
            sensitivity={k: dict(threshold=cuts[k], retained=len(v), names=sorted(v))
                         for k, v in names_by_cut.items() if k != 'DEV10_TRANSFER'})
        for row in rows:
            rec = dict(method=method, **row)
            for key, names in names_by_cut.items():
                rec['retained_by_' + key] = row['lipid_name'] in names
            records.append(rec)
    ista, nnls = (set(summaries[m]['names']) for m in ('ISTA', 'NNLS'))
    comparisons = dict(common=sorted(ista & nnls), ISTA_only=sorted(ista - nnls),
                       NNLS_only=sorted(nnls - ista), union=sorted(ista | nnls))
    report = dict(status='COMPLETE_REVIEWED', foreground_pixels=15837, candidate_columns=391,
                  primary_rule='raw_solver_reported AND joint_score >= saved DEV10 threshold',
                  threshold=primary, threshold_source='DEV_rule in saved user_FDP10_recall_query.json',
                  threshold_selection_used_real_data=False, model_sha256=source_model_sha,
                  summary=summaries, comparison=comparisons,
                  actual_FDR=None, TP=None, FP=None, FN=None, recall=None, TP_retention=None,
                  truth_available=False, coverage_is_not_recall=True,
                  source_hashes=sources, no_solver_or_rho_or_model_fit=True,
                  sensitivity_is_outcome_guided_and_not_a_recommendation=True)
    OUT.mkdir(parents=True)
    shutil.copyfile(QUERY, OUT / 'source_threshold_query.json')
    write(OUT / 'report.json', report)
    write(OUT / 'molecular_records.json', records)
    fields = ['method', 'lipid_name', 'lipid_class', 'raw_solver_reported', 'X_hat', 'rho_zero',
              'predictive_gain', 'positive_block_fraction', 'joint_score', 'molecular_truth',
              'retained_by_pool', 'retained_by_fdp5', 'retained_by_fdp1']
    fields += ['retained_by_' + k for k in cuts]
    for filename, subset in (
        ('molecular_screening.csv', records),
        ('retained_identities.csv', [r for r in records if r['retained_by_DEV10_TRANSFER']]),
        ('added_vs_DEV5.csv', [r for r in records if r['retained_by_DEV10_TRANSFER'] and not r['retained_by_fdp5']]),
    ):
        with (OUT / filename).open('x', encoding='utf-8-sig', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(subset)
    lines = ['# 真实 CE29：模拟 DEV10 阈值迁移', '',
             f'主分析使用同一个已保存DEV阈值 joint_score >= {primary:.16g}，仅筛选原报告身份。',
             '这表示阈值在模拟DEV中按实际错误比例严格低于10%选出，不代表真实FDR<10%。真实TP/FP/FN、recall及TP retention未知。',
             '使用全部15837前景像素的最终ISTA/NNLS记录；没有新拟合、rho、模型训练或删除。', '',
             '| 方法 | 原报告 | DEV1 | DEV5 | 原候选池 | DEV10保留 | 占原报告比例（不是recall） |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for method, v in summaries.items():
        lines.append(f"| {method} | {v['raw']} | {v['old_counts']['fdp1']} | {v['old_counts']['fdp5']} | {v['old_counts']['pool']} | {v['retained']} | {v['coverage']:.2%} |")
    lines += ['', f'主阈值两方法共同保留{len(ista & nnls)}名，ISTA独有{len(ista - nnls)}名，NNLS独有{len(nnls - ista)}名。一致不等于独立身份证实。', '',
              '## 截图中逐组看真值后选出的阈值：仅作敏感性展示', '',
              '| 阈值来源 | 分数阈值 | ISTA保留 | NNLS保留 |', '|---|---:|---:|---:|']
    for key in cuts:
        if key != 'DEV10_TRANSFER':
            lines.append(f"| {key} | {cuts[key]:.10g} | {summaries['ISTA']['sensitivity'][key]['retained']} | {summaries['NNLS']['sensitivity'][key]['retained']} |")
    lines += ['', '这两个数值分别来自原组合/新组合的结果导向最佳截断，不能选择其中在真实数据上留下更多身份的一个，再声称真实错误率低于10%。它们未作为正式筛选规则或再次解卷积输入。', '',
              '## 相比固定DEV5档新增的身份', '']
    for method, v in summaries.items():
        lines += [f"{method}（{len(v['added_vs_old']['fdp5'])}名）：" + '；'.join(v['added_vs_old']['fdp5']), '']
    lines += ['完整身份/原分数/三档新增标记见 molecular_screening.csv；主阈值名单见 retained_identities.csv；阈值来源原文已复制并保存hash。', '']
    (OUT / 'analysis_record.md').write_text('\n'.join(lines), encoding='utf-8', newline='\n')
    require(sources == {p: sha(ROOT / p) for p in sources}, 'INPUT_CHANGED_DURING_ANALYSIS')
    write(OUT / 'cache_review.json', dict(status='PASS', source_hashes_unchanged=True,
        parent_reviews='PASS', independent_scalar_score_errors=score_errors,
        original_threshold_flags_unchanged=True, unknown_truth_preserved=True,
        no_new_optimization=True, output_records=len(records)))
    write(OUT / 'artifact_hashes.json', {p.name: sha(p) for p in sorted(OUT.iterdir()) if p.is_file()})
    print(json.dumps(dict(threshold=primary, summary=summaries, comparison=comparisons), ensure_ascii=False))


if __name__ == '__main__':
    main()
