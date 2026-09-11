"""Review cached CAL-only prediction scores; no scientific computation or fitting."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def ranking(rows, score, truth_count, raw_tp, reportable_count=None, raw_count=None):
    usable = [r for r in rows if r[score] is not None]
    usable.sort(key=lambda r: r[score], reverse=True)
    positives = sum(r['molecular_truth'] for r in usable)
    negatives = len(usable) - positives
    tp = fp = reportable_tp = 0
    ap = auc = 0.
    curve = []
    i = 0
    while i < len(usable):
        j = i + 1
        while j < len(usable) and usable[j][score] == usable[i][score]:
            j += 1
        dt = sum(r['molecular_truth'] for r in usable[i:j])
        df = j-i-dt
        auc += df*(tp + dt/2)
        tp += dt
        fp += df
        reportable_tp += sum(r.get('reportable_truth', False) for r in usable[i:j])
        ap += dt*tp/(tp+fp)
        curve.append(dict(threshold=usable[i][score], TP=tp, FP=fp,
                          FDR=fp/(tp+fp), precision=tp/(tp+fp),
                          all_truth_recall=tp/truth_count, TP_retention=tp/raw_tp,
                          reportable_truth_recall=reportable_tp/reportable_count if reportable_count else None,
                          coverage=(tp+fp)/raw_count if raw_count else None,
                          solver_FN=truth_count-raw_tp, post_solver_true_loss=raw_tp-tp))
        i = j
    best = {}
    for level in (.05, .01):
        allowed = [c for c in curve if c['FDR'] <= level]
        best[str(level)] = max(allowed, key=lambda c: (c['TP'], -c['FP'])) if allowed else dict(TP=0, FP=0, all_truth_recall=0.)
    return dict(usable=len(usable), usable_TP=positives, usable_FP=negatives,
                excluded_reported_TP=raw_tp-positives, all_truth_recall_ceiling=positives/truth_count,
                AUROC=auc/(positives*negatives) if positives*negatives else None,
                AP=ap/positives if positives else None, retrospective_best=best), curve


def main(root):
    provenance = json.loads((root/'provenance.json').read_text())
    for name, digest in provenance['files'].items():
        assert sha(root/name) == digest, name
    assert sha(root/'contract.json') == provenance['contract_sha256']
    assert sha(root/'executed_runner.py') == provenance['script_sha256']
    rows = list(csv.DictReader((root/'scores.csv').open(newline='')))
    for r in rows:
        for k in ('molecular_truth', 'reportable_truth', 'evaluable'):
            assert r[k] in ('True', 'False')
            r[k] = r[k] == 'True'
        for k in ('rho_zero', 'X_hat', 'predictive_gain'):
            r[k] = float(r[k]) if r[k] else None
            assert r[k] is None or math.isfinite(r[k])
        assert r['evaluable'] == (r['predictive_gain'] is not None)
    assert len({(r['domain'], r['lipid_name']) for r in rows}) == len(rows)
    folds = list(csv.DictReader((root/'fold_predictions.csv').open(newline='')))
    assert len(folds) == 3*len(rows)
    assert len({(r['domain'], r['lipid_name'], r['fold']) for r in folds}) == len(folds)
    for r in folds:
        for k in ('full_test_loss', 'deleted_test_loss', 'signed_gain'):
            assert math.isfinite(float(r[k]))
        assert math.isclose(float(r['signed_gain']), float(r['deleted_test_loss'])-float(r['full_test_loss']), abs_tol=1e-18)
    test = [dict(molecular_truth=True, x=1), dict(molecular_truth=False, x=1)]
    check, _ = ranking(test, 'x', 1, 1)
    assert check['AUROC'] == .5 and check['AP'] == .5
    result = dict(status='REVIEWED', source_files_verified=True,
                  limitation='One CAL pair; retrospective empirical FDR envelopes are not calibrated operating thresholds.', domains={})
    curves = []
    for domain, counts in provenance['domains'].items():
        subset = [r for r in rows if r['domain'] == domain]
        assert len(subset) == counts['raw_reported']
        assert sum(r['molecular_truth'] for r in subset) == counts['raw_TP']
        data = dict(counts=counts, solver_FN=counts['all_truth_count']-counts['raw_TP'], methods={})
        for universe in ('original_reported', 'common_evaluable'):
            selected = subset if universe == 'original_reported' else [r for r in subset if r['evaluable']]
            data['methods'][universe] = {}
            for score in ('rho_zero', 'X_hat', 'predictive_gain'):
                summary, curve = ranking(selected, score, counts['all_truth_count'], counts['raw_TP'], counts['reportable_truth_count'], counts['raw_reported'])
                data['methods'][universe][score] = summary
                curves += [dict(domain=domain, universe=universe, score=score, **c) for c in curve]
        result['domains'][domain] = data
    methods = result['domains']['MILD']['methods']['original_reported']
    gains = {s: m['retrospective_best']['0.05']['TP'] for s, m in methods.items()}
    result['screen_decision'] = 'PROMISING_REQUIRES_INDEPENDENT_VALIDATION' if gains['predictive_gain'] > max(gains['rho_zero'], gains['X_hat']) else 'STOP_STANDALONE_ROUTE_NO_5_PERCENT_RECALL_GAIN'
    (root/'review.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    with (root/'threshold_curves.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(curves[0])); w.writeheader(); w.writerows(curves)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('root', type=Path)
    main(parser.parse_args().root)
