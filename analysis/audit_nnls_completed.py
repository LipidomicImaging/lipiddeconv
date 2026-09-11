"""Audit existing NNLS/ISTA tables only; no solver or certificate imports."""
import csv
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/nnls_solver_baseline_k125_cpu4'
PARENT = ROOT / 'results/v57_spectral_spatial_identity_confidence_benchmark'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def metrics(raw, selected):
    tp = sum(r['molecular_truth'] for r in selected)
    fp = len(selected) - tp
    raw_tp = sum(r['molecular_truth'] for r in raw)
    return dict(TP=tp, FP=fp, FN=375-tp, raw_solver_TP=raw_tp,
                raw_solver_FP=len(raw)-raw_tp, raw_solver_FN=375-raw_tp,
                filtered_TP=tp, filtered_FP=fp, filtered_FN=375-tp,
                all_truth_recall=tp/375,
                reportable_truth_recall=sum(r['reportable_truth'] for r in selected)/375,
                FDR=fp/len(selected), precision=tp/len(selected),
                TP_retention=tp/raw_tp, coverage=len(selected)/len(raw),
                filter_induced_true_loss=raw_tp-tp,
                filter_induced_true_loss_fraction=(raw_tp-tp)/raw_tp)


def main():
    manifest = read(OUT/'download_manifest.json')
    for name, expected in manifest['sha256'].items():
        assert sha(OUT/name) == expected, name
    design = read(OUT/'design.json')
    canonical = json.dumps(design['contract'], sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False)
    assert hashlib.sha256(canonical.encode()).hexdigest() == design['fingerprint']
    ids = [f'{split}_R{r}_K125' for split in ('CAL', 'HOLD') for r in (1, 2, 3)]
    assert design['contract']['datasets'] == ids
    assert read(OUT/'status.json')['completed'] == ids
    assert {p.parent.name for p in OUT.glob('*/result.json')} == set(ids)
    frozen = read(OUT/'thresholds.json')
    assert sha(OUT/'thresholds.json') == read(OUT/'threshold_seal.json')['thresholds_sha256']
    assert sha(PARENT/'calibration_records.csv') == frozen['ISTA_CAL_records_sha256']
    units = {'CAL': [], 'HOLD': []}
    for dataset in ids:
        report = read(OUT/dataset/'result.json')
        assert report['fingerprint'] == design['fingerprint'] == frozen['fingerprint']
        assert report['dataset_id'] == dataset
        assert report['all_truth_molecular_identity_count'] == report['reportable_truth_molecular_identity_count'] == 125
        assert sha(OUT/dataset/'molecular_units.json') == report['units_sha256']
        rows = read(OUT/dataset/'molecular_units.json')
        assert len({r['lipid_name'] for r in rows}) == len(rows)
        assert all(r['dataset_id'] == dataset for r in rows)
        units[dataset.split('_')[0]].extend(rows)
        if dataset.startswith('CAL'):
            assert sha(OUT/dataset/'result.json') == frozen['CAL_result_hashes'][dataset]
    ista = {}
    for split, filename in [('CAL', 'calibration_records.csv'), ('HOLD', 'heldout_records.csv')]:
        with (PARENT/filename).open(encoding='utf-8', newline='') as f:
            rows = [r for r in csv.DictReader(f) if r['dataset_id'] in ids]
        assert sum(r['molecular_truth'] == 'True' for r in rows) == 375
        assert sum(r['reportable_truth'] == 'True' for r in rows) == 375
        ista[split] = [dict(r, molecular_truth=r['molecular_truth'] == 'True',
                           reportable_truth=r['reportable_truth'] == 'True',
                           X_hat=float(r['X_hat']), rho_zero=float(r['rho_zero']))
                       for r in rows if r['raw_solver_reported'] == 'True']
    comparison = read(OUT/'comparison.json')
    assert comparison['fingerprint'] == design['fingerprint']
    audited = {}
    for solver, groups in [('NNLS', units), ('ISTA', ista)]:
        hold = groups['HOLD']
        result = {'raw': metrics(hold, hold)}
        for score in ('rho_zero', 'X_hat'):
            # Verify the frozen maximum-retention CAL choice; never write a new cutoff.
            cal = groups['CAL']
            choices = []
            for threshold in sorted({r[score] for r in cal}):
                selected = [r for r in cal if r[score] >= threshold]
                fp = sum(not r['molecular_truth'] for r in selected)
                choices.append((threshold, len(selected), fp))
            for target, level in [('FDR5', .05), ('FDR1', .01)]:
                entry = frozen[solver][score][target]
                eligible = [(t, n, fp) for t, n, fp in choices if fp/n <= level]
                best = max(eligible, key=lambda x: x[1])
                assert (entry['threshold'], entry['N_retained'], entry['FP']) == best
                selected = [r for r in hold if r[score] >= entry['threshold']]
                result[f'{score}_{target}'] = metrics(hold, selected)
        for method, values in result.items():
            for name, value in values.items():
                assert math.isclose(value, comparison['results'][solver][method][name], rel_tol=1e-12, abs_tol=1e-15), (solver, method, name)
        audited[solver] = result
    audit = dict(status='PASS', verified_download_files=len(manifest['sha256']),
                 fingerprint=design['fingerprint'], CAL_only_threshold_choices_verified=True,
                 HOLD_metrics_independently_reproduced=audited,
                 ISTA_HOLD_source_sha256=sha(PARENT/'heldout_records.csv'),
                 remote_array_hash_check_evidence='independent_audit.json; arrays remain remote',
                 limitations=['No solver or rho recomputation.', 'Remote chronology uses file timestamps, not an immutable external time seal.',
                              'KKT checks sample every 500 foreground pixels, not every pixel.',
                              'Repeated identities/mappings are not independent biological samples.'])
    (OUT/'local_review.json').write_text(json.dumps(audit, indent=2)+'\n', encoding='utf-8')
    print('PASS: source hashes, six cases, CAL cutoffs, and NNLS/ISTA HOLD accounting')


if __name__ == '__main__':
    main()
