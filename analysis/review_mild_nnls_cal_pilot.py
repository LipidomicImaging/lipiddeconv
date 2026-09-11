"""Independent cached molecular accounting for the one-case MILD NNLS screen."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

from review_cal_channel_prediction_pilot import ranking


def read(p):
    return json.loads(p.read_text())


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main(root):
    provenance = read(root/'provenance.json')
    assert provenance['normal_completion'] and provenance['all_outputs_finite']
    for name, expected in provenance['files'].items():
        if name == 'learned_arrays.npz':
            receipt = read(root/'remote_array_recheck.json')
            assert receipt['sha256'] == expected and receipt['verified']
        else:
            assert sha(root/name) == expected, name
    contract = read(root/'contract.json')
    assert sha(root/'executed_runner.py') == contract['implementation_sha256']
    assert sha(root/'executed_baseline.py') == contract['baseline_sha256']
    candidate_rows = read(root/'all_candidate_abundances.json')
    assert len(candidate_rows) == 391 and len({r['candidate_index'] for r in candidate_rows}) == 391
    assert all(math.isfinite(r['X_hat']) and r['X_hat'] >= 0 for r in candidate_rows)
    truth = {r['lipid_name'] for r in candidate_rows if r['molecular_truth']}
    reportable = {r['lipid_name'] for r in candidate_rows if r['reportable_truth']}
    reported = {r['lipid_name'] for r in candidate_rows if r['X_hat'] > .001}
    assert len(truth) == len(reportable) == 125
    comparisons = read(root/'comparison_units.json')
    assert set(comparisons) == {'CLEAN_NNLS', 'MILD_NNLS', 'CLEAN_ISTA', 'MILD_ISTA'}
    assert reported == {r['lipid_name'] for r in comparisons['MILD_NNLS']}
    summary = dict(status='REVIEWED_CAL_ONLY', source_and_output_hashes_valid=True,
                   arrays='Retained and rehashed remotely; not placed in lightweight Git.', arms={})
    curves = []
    for arm, rows in comparisons.items():
        assert len({r['lipid_name'] for r in rows}) == len(rows)
        assert all(r['molecular_truth'] == (r['lipid_name'] in truth) for r in rows)
        tp = sum(r['molecular_truth'] for r in rows); fp = len(rows)-tp
        summary['arms'][arm] = dict(raw_TP=tp, raw_FP=fp, raw_FN=125-tp,
                                  raw_FDR=fp/len(rows) if rows else None, raw_recall=tp/125, scores={})
        for score in ('rho_zero', 'X_hat'):
            assert all(math.isfinite(r[score]) for r in rows)
            m, curve = ranking(rows, score, 125, tp, 125, len(rows))
            summary['arms'][arm]['scores'][score] = m
            curves += [dict(arm=arm, score=score, **c) for c in curve]
    shared = []
    for domain in ('CLEAN', 'MILD'):
        a = {r['lipid_name']: r for r in comparisons[domain+'_NNLS']}
        b = {r['lipid_name']: r for r in comparisons[domain+'_ISTA']}
        for name in sorted(a.keys() & b.keys()):
            shared.append(dict(domain=domain, lipid_name=name, molecular_truth=name in truth,
                               NNLS_rho=a[name]['rho_zero'], ISTA_rho=b[name]['rho_zero'],
                               absolute_difference=abs(a[name]['rho_zero']-b[name]['rho_zero'])))
    summary['shared_rho_max_difference'] = {d: max(r['absolute_difference'] for r in shared if r['domain'] == d) for d in ('CLEAN', 'MILD')}
    summary['interpretation_limit'] = 'Single CAL pair, no independent FDR guarantee; different fitting objectives. Rho depends on A/B and reported identity, not fitted abundance except candidate selection.'
    (root/'review.json').write_text(json.dumps(summary, indent=2, allow_nan=False)+'\n')
    for name, rows in [('threshold_curves.csv', curves), ('shared_rho_comparison.csv', shared)]:
        with (root/name).open('w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('root', type=Path)
    main(p.parse_args().root)
