"""Read-only V57 downstream accounting; no scientific runner imports or training."""
import csv
import hashlib
import json
import statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'results/v57_spectral_spatial_identity_confidence_benchmark'
OUT = ROOT / 'results/computational_closure/v57'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def yes(value):
    assert value in ('True', 'False'), value
    return value == 'True'


def write_csv(name, rows):
    with (OUT / name).open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def ratio(a, b):
    return a / b if b else None


def metrics(rows, flag):
    truth = [r for r in rows if yes(r['molecular_truth'])]
    reportable = [r for r in truth if yes(r['reportable_truth'])]
    raw = [r for r in rows if yes(r['raw_solver_reported'])]
    retained = raw if flag is None else [r for r in rows if yes(r[flag])]
    raw_tp = sum(yes(r['molecular_truth']) for r in raw)
    tp = sum(yes(r['molecular_truth']) for r in retained)
    fp = len(retained) - tp
    assert all(yes(r['raw_solver_reported']) for r in retained)
    return dict(truth_count=len(truth), reportable_truth_count=len(reportable),
                raw_solver_TP=raw_tp, raw_solver_FP=len(raw)-raw_tp,
                raw_solver_FN=len(truth)-raw_tp, filtered_TP=tp, filtered_FP=fp,
                filtered_FN=len(truth)-tp, FDR=ratio(fp, len(retained)),
                all_truth_recall=ratio(tp, len(truth)),
                reportable_truth_recall=ratio(sum(yes(r['molecular_truth']) and yes(r['reportable_truth']) for r in retained), len(reportable)),
                TP_retention=ratio(tp, raw_tp), filter_induced_true_loss=raw_tp-tp,
                filter_induced_true_loss_fraction=ratio(raw_tp-tp, raw_tp),
                coverage=ratio(len(retained), len(raw)))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    frozen = read_json(SOURCE / 'global_frozen_thresholds.json')
    design = read_json(SOURCE / 'design.json')
    canonical = json.dumps(design['scientific'], sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False)
    assert hashlib.sha256(canonical.encode()).hexdigest() == design['design_fingerprint']
    assert frozen['design_fingerprint'] == design['design_fingerprint']
    assert frozen['status'] == 'FROZEN_FROM_CAL_ONLY'
    provenance = []
    for dataset, files in frozen['CAL_source_hashes'].items():
        assert dataset.startswith('CAL_')
        for name, expected in files.items():
            path = SOURCE / 'clean' / dataset / name
            actual = digest(path) if path.exists() else None
            lf_hash = hashlib.sha256(path.read_bytes().replace(b'\r\n', b'\n')).hexdigest() if path.exists() else None
            assert actual is None or actual == expected or lf_hash == expected, str(path)
            provenance.append(dict(path=str(path.relative_to(ROOT)), expected=expected, actual=actual,
                                   lf_normalized_sha256=lf_hash,
                                   status=('BYTE_MATCH' if actual == expected else 'LF_NORMALIZED_MATCH' if lf_hash == expected else 'NOT_AVAILABLE_LOCALLY')))
    with (SOURCE / 'molecular_false_negative_records.csv').open(encoding='utf-8', newline='') as f:
        rows = list(csv.DictReader(f))
    assert len({(r['dataset_id'], r['lipid_name']) for r in rows}) == len(rows)
    assert {r['dataset_id'] for r in rows} == {f'{s}_R{r}_K{k:03d}' for s in ('CAL','HOLD') for r in range(1,6) for k in (50,75,100,125,150,175)}
    checks = []
    for score in ('rho_zero', 'X_hat'):
        for target in ('FDR5', 'FDR1'):
            entry = frozen['global_thresholds'][score][target]
            flag = f'retained_by_{score}_global_{target}'
            for row in rows:
                expected = yes(row['raw_solver_reported']) and entry is not None and float(row[score]) >= entry['threshold']
                assert yes(row[flag]) == expected, (row['dataset_id'], row['lipid_name'], flag)
            cal = metrics([r for r in rows if r['split'] == 'CAL'], flag)
            assert cal['filtered_TP'] == entry['TP'] and cal['filtered_FP'] == entry['FP']
            checks.append(dict(score=score, target=target, CAL_counts_match=True, all_retention_flags_match=True))
    hold = [r for r in rows if r['split'] == 'HOLD']
    summaries = []
    groups = [('GLOBAL','ALL',hold)]
    groups += [('K',k,[r for r in hold if r['K'] == k]) for k in sorted({r['K'] for r in hold}, key=int)]
    groups += [('replicate',rep,[r for r in hold if r['replicate'] == rep]) for rep in sorted({r['replicate'] for r in hold})]
    for scope, value, subset in groups:
        for score, target, flag in [('raw','NONE',None)] + [(s,t,f'retained_by_{s}_global_{t}') for s in ('rho_zero','X_hat') for t in ('FDR5','FDR1')]:
            summaries.append(dict(scope=scope, value=value, score=score, target=target, **metrics(subset,flag)))
    write_csv('hold_accounting.csv', summaries)
    strata = defaultdict(list)
    for target in ('FDR5','FDR1'):
        for row in hold:
            if not yes(row['molecular_truth']):
                continue
            status = ('solver_miss' if not yes(row['raw_solver_reported']) else
                      'retained' if yes(row[f'retained_by_global_{target}']) else 'filter_loss')
            # Exact frozen multiplier strata avoid data-dependent abundance cut points.
            key = (target,row['K'],row['replicate'],row['lipid_class'],row['abundance_multiplier'],status)
            strata[key].append(row)
    descriptive = []
    for key, values in sorted(strata.items()):
        row = dict(zip(('target','K','replicate','lipid_class','abundance_multiplier','status'),key))
        row['n_identity_contexts'] = len(values)
        for field in ('cone_isolation','max_fragment_cosine','max_full_cosine','collective_gain','X_hat'):
            vals = [float(r[field]) for r in values]
            row[f'{field}_median'] = statistics.median(vals)
        descriptive.append(row)
    write_csv('truth_loss_geometry_strata.csv', descriptive)
    losses = []
    for target in ('FDR5','FDR1'):
        by_name = defaultdict(list)
        for row in hold:
            if yes(row['molecular_truth']):
                by_name[row['lipid_name']].append(row)
        for name, values in sorted(by_name.items()):
            raw = sum(yes(r['raw_solver_reported']) for r in values)
            retained = sum(yes(r[f'retained_by_global_{target}']) for r in values)
            losses.append(dict(target=target,lipid_name=name,truth_contexts=len(values),
                               raw_solver_miss_contexts=len(values)-raw,filter_loss_contexts=raw-retained,
                               retained_contexts=retained))
    write_csv('identity_loss_recurrence.csv',losses)
    audit = dict(status='SNAPSHOT_ACCOUNTING_VERIFIED_SOURCE_PROVENANCE_PARTIAL' if any(r['actual'] is None for r in provenance) else 'VERIFIED',
                 design_fingerprint=design['design_fingerprint'],CAL_source_checks=provenance,
                 consistency_checks=checks,rows=len(rows),
                 input_hashes={n:digest(SOURCE/n) for n in ('design.json','global_frozen_thresholds.json','molecular_false_negative_records.csv')},
                 script_sha256=digest(Path(__file__)),
                 limitations=['Missing per-case source records prevent full source-chain verification.',
                              'Repeated identities, K levels and mappings are not independent biological samples.',
                              'Geometry strata are descriptive; no adjusted enrichment claim or independent-row p-values.',
                              'No threshold selection or training performed.'])
    (OUT/'audit.json').write_text(json.dumps(audit,indent=2)+'\n',encoding='utf-8')
    global_rows = [r for r in summaries if r['scope']=='GLOBAL']
    (OUT/'summary.json').write_text(json.dumps(global_rows,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(audit_status=audit['status'], missing_source_files=sum(r['actual'] is None for r in provenance),global_results=global_rows),indent=2))


if __name__ == '__main__':
    main()
