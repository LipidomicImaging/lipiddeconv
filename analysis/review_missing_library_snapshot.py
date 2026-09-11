"""Independently count downloaded, cached missing-library molecular records."""
import argparse
import hashlib
import json
from pathlib import Path


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def review(out):
    hashes = read(out/'compact_hashes.json')
    for name, expected in hashes.items():
        assert hashlib.sha256((out/name).read_bytes()).hexdigest() == expected, name
    source = out/'source/results/missing_library_challenge_k125'
    design = read(source/'execution_design.json')
    seal = read(source/'execution_freeze.json')
    assert hashlib.sha256((source/'execution_design.json').read_bytes()).hexdigest() == seal['design_sha256']
    assert read(Path('results/missing_library_challenge_k125/execution_freeze.json')) == seal
    pack = design['scientific']['thresholds']['global_thresholds']
    remote = read(out/'scientific_review.json')
    checked = {}
    for case in sorted(design['scientific']['cases']):
        rows = read(source/'runs'/case/'candidate_records.json')
        units = read(source/'runs'/case/'molecular_units.json')
        truth = {r['lipid_name'] for r in rows if r['molecular_truth']}
        reportable = {r['lipid_name'] for r in rows if r['reportable_truth']}
        omitted = {r['lipid_name'] for r in rows if r['omitted_from_solver']}
        assert len(rows) == 391 and len(truth) == len(reportable) == 125
        assert len(omitted) == 5 and omitted <= truth
        reported = {r['lipid_name'] for r in rows if r['raw_solver_reported']}
        assert reported == {u['lipid_name'] for u in units} and len(reported) == len(units)
        raw_tp = len(reported & truth)
        selections = {'raw': reported}
        for score, targets in pack.items():
            for target, cut in targets.items():
                selections[score+'_'+target] = {u['lipid_name'] for u in units if cut is not None and u[score] >= cut['threshold']}
        metrics = {}
        for mode, selected in selections.items():
            tp, false, fn = len(selected & truth), len(selected-truth), len(truth-selected)
            data = dict(TP=tp, FP=false, FN=fn, raw_solver_TP=raw_tp,
                        raw_solver_FP=len(reported-truth), raw_solver_FN=len(truth-reported),
                        structural_omission_FN=5, additional_raw_solver_FN=len(truth-reported-omitted),
                        filter_induced_true_loss=len((reported & truth)-selected),
                        FDR=false/len(selected) if selected else None,
                        TP_retention=tp/raw_tp, all_truth_recall=tp/125,
                        reportable_truth_recall=len(selected & reportable)/125)
            assert data['FN'] == data['structural_omission_FN'] + data['additional_raw_solver_FN'] + data['filter_induced_true_loss']
            for k, value in data.items():
                assert remote['cases'][case]['independent_accounting'][mode][k] == value, (case, mode, k)
            metrics[mode] = data
        checked[case] = metrics
    for arm in ('close_neighbor', 'relatively_isolated'):
        cases = [key for key in checked if key.startswith(arm+'__')]
        assert len(cases) == 3
        for mode, values in remote['pooled_by_arm'][arm].items():
            for k in ('TP', 'FP', 'FN', 'raw_solver_TP', 'raw_solver_FP', 'raw_solver_FN',
                      'structural_omission_FN', 'additional_raw_solver_FN', 'filter_induced_true_loss'):
                assert sum(checked[key][mode][k] for key in cases) == values[k]
    plan = read(out/'cleanup_plan.json')
    assert len(plan['deletions']) == 24 and len({r['path'] for r in plan['deletions']}) == 24
    assert all(r['path'] not in plan['source_files'] for r in plan['deletions'])
    for case in checked:
        root = '/root/autodl-tmp/lipiddeconv/results/missing_library_challenge_k125/runs/'+case+'/'
        assert all(root+n in plan['source_files'] for n in ('latest_model.pth', 'checkpoint_epoch_3000.pth', 'learned_arrays.npz'))
    findings = [
        '# Missing-library final review — 2026-09-11', '',
        'Six reduced-library fits completed and passed process, provenance and independent accounting review. '
        'Each removes five true singleton identities from the solver library, retains the original 125-truth '
        'denominator and uses the original V57 CAL thresholds without recalibration. The three existing full-library '
        'controls and their checkpoint evidence remain intact.', '',
        'All values below are pooled over three mapping replicates per arm (375 truth contexts). FDR is the '
        'observed false-discovery proportion here; mapping replicates and repeated identities are not independent '
        'biological samples. Zero observed FP does not establish zero population risk.', '',
        '| Library arm | Frozen rule | TP | FP | FN | Observed FDR | TP retention | All-truth recall |',
        '|---|---|---:|---:|---:|---:|---:|---:|'
    ]
    for arm in ('full_library', 'close_neighbor', 'relatively_isolated'):
        for mode in ('raw', 'rho_zero_FDR5', 'rho_zero_FDR1', 'X_hat_FDR5', 'X_hat_FDR1'):
            m = remote['pooled_by_arm'][arm][mode]
            findings.append(f"| {arm} | {mode} | {m['TP']} | {m['FP']} | {m['FN']} | {100*m['FDR']:.2f}% | {100*m['TP_retention']:.2f}% | {100*m['all_truth_recall']:.2f}% |")
    findings += ['',
        'The CLEAN-calibrated rho FDR1 cutoff retains 361 true and zero false contexts with the full library, '
        'but 302/187 TP/FP after close-neighbor omission and 279/184 after relatively-isolated omission. '
        'Observed FDR rises to 38.24% and 39.74%. This transfer fails the original nominal 1% target. '
        'These are missing-library CLEAN observations, not a combined spectral-mismatch-plus-omission experiment.', '',
        'Raw FN accounting: close-neighbor arm has 25 FN = 15 obligatory omitted truths + 10 additional solver misses; '
        'the relatively-isolated arm has 28 = 15 + 13. The rho FDR1 filter then loses another 48 and 68 raw true positives, '
        'respectively. Omitted truths were never removed from recall denominators.', '',
        'The six stored foreground reconstruction residuals range from '
        f"{min(remote['cases'][k]['reconstruction_residual'] for k in checked):.6f} to "
        f"{max(remote['cases'][k]['reconstruction_residual'] for k in checked):.6f}. "
        'They are scientific outcomes, not exclusion gates. No fit, synthetic reconstruction or rho was rerun. '
        'The original residual values were retained rather than independently recomputed from B.', '',
        'Candidate and molecular records, exact frozen thresholds, CAL source records, all six runtime/result/history '
        'files, checkpoint diagnostics and paired neighbor-allocation records are in this snapshot. '
        'The relatively-isolated arm is only relatively less similar; one selected neighbor cosine is 0.966. '
        'Neighbor allocation differences are descriptive and do not identify a unique source of signal.', '',
        'Storage decision: retire only epochs 1000, 1500, 2000 and 2500 for these six completed reduced-library fits: '
        f"24 files, {plan['planned_delete_bytes']} bytes ({plan['planned_delete_bytes']/2**30:.3f} GiB). "
        'All 178 protected source hashes must still match immediately before and after deletion. '
        'The result-bound latest_model.pth, final epoch-3000 model, learned arrays and all parent-control '
        'checkpoints remain. Intermediate weights will no longer be reloadable; their scalar diagnostics remain. '
        'This plan requires a successful Git snapshot push before execution; consult deletion_receipt.json for '
        'whether execution actually completed.', '',
        'Review-tool preflight notes: an initial export refused to duplicate the large frozen parent design; '
        'a second attempt encountered the nested V57 molecular-level report schema. The tool was corrected, '
        'partial exports retained separately on the remote host, and all six cases subsequently passed. '
        'Neither issue changed any experiment or source artifact.', ''
    ]
    (out/'final_findings.md').write_text('\n'.join(findings), encoding='utf-8', newline='\n')
    result = dict(status='PASS', downloaded_source_hashes_verified=len(hashes), independent_case_count=len(checked),
                  exact_molecular_membership=True, all_truth_denominators_include_omissions=True,
                  original_local_execution_seal_matches=True, per_case_and_pooled_counts_match=True,
                  final_models_and_arrays_excluded_from_retirement=True,
                  cleanup_plan_sha256=hashlib.sha256((out/'cleanup_plan.json').read_bytes()).hexdigest(),
                  source_archive_sha256='61dfad13e0c753556896547db06b788e2ce6579a82bb958256dec2873dfd0edf')
    (out/'local_verification.json').write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8', newline='\n')
    # Preserve original CSV CRLF bytes for provenance; recognize CR as part of
    # the line ending while retaining ordinary whitespace checks.
    (out/'.gitattributes').write_text('* -text whitespace=blank-at-eol,blank-at-eof,space-before-tab,cr-at-eol\n', encoding='utf-8', newline='\n')
    print(json.dumps(result))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('snapshot', type=Path)
    review(parser.parse_args().snapshot)
