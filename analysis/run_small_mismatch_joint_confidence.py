"""Reuse the fixed rho/abundance grouped classifier on one cached 5% case."""
import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
from types import SimpleNamespace

for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
ROOT = Path(__file__).resolve().parents[1]
ENGINE = ROOT / 'analysis/run_joint_confidence_cal_pilot.py'
PROTOCOL = ROOT / 'docs/SMALL_MISMATCH_JOINT_CONFIDENCE.md'
CASE = 'NNLS_FULL_LIBRARY_5PCT__CAL_R71_K125'
KEYS = ('rho_zero', 'X_hat', 'joint_probability')


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(value, indent=2, allow_nan=False) + '\n').encode()
    if path.exists():
        assert path.read_bytes() == data, 'EXISTING_RECORD_PRESERVED:' + str(path)
    else:
        with path.open('xb') as f:
            f.write(data)


def group(name):
    return int.from_bytes(hashlib.sha256(('joint_confidence_CAL_v1|' + name).encode()).digest()[:8], 'little') % 5


def sources(case_dir):
    d = read(case_dir / 'design.json'); result = read(case_dir / 'result.json')
    assert d['scientific']['contract']['case'] == result['case'] == CASE
    assert d['fingerprint'] == result['fingerprint']
    assert result['normal_completion'] and result['all_outputs_finite']
    for name in ('molecular_records.json', 'summary.json', 'design.json'):
        assert sha(case_dir / name) == result['artifact_hashes'][name]
    rows = read(case_dir / 'molecular_records.json')
    assert len({r['lipid_name'] for r in rows}) == len(rows)
    assert sum(r['molecular_truth'] for r in rows) == 125
    assert sum(r['raw_solver_reported'] for r in rows) == 174
    for r in rows:
        assert math.isfinite(r['X_hat']) and r['X_hat'] >= 0
        assert not r['raw_solver_reported'] or (r['rho_zero'] is not None and math.isfinite(r['rho_zero']) and r['rho_zero'] >= 0)
    hashes = {name: sha(case_dir / name) for name in ('molecular_records.json', 'summary.json', 'design.json', 'result.json')}
    return rows, hashes, d['fingerprint']


def prepare(a):
    rows, hashes, fingerprint = sources(a.case_dir)
    prep = a.output / 'prepared'
    prep.mkdir(parents=True, exist_ok=False)
    source = prep / 'input'; source.mkdir()
    fields = ['dataset_id', 'severity', 'split', 'lipid_name', 'molecular_truth', 'reportable_truth', 'raw_solver_reported', 'X_hat', 'rho_zero']
    with (source / 'molecules.csv').open('x', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fields, lineterminator='\n'); writer.writeheader()
        for r in rows:
            writer.writerow(dict(dataset_id=CASE, severity='FULL_LIBRARY_5PCT', split='CAL', **{k: r[k] for k in fields[3:]}))
    folds = []
    for fold in range(5):
        rr = [r for r in rows if group(r['lipid_name']) == fold]
        folds.append(dict(fold=fold, members=sorted(r['lipid_name'] for r in rr),
                          raw_TP=sum(r['raw_solver_reported'] and r['molecular_truth'] for r in rr),
                          raw_FP=sum(r['raw_solver_reported'] and not r['molecular_truth'] for r in rr),
                          truth=sum(r['molecular_truth'] for r in rr)))
    contract = dict(version='SMALL_MISMATCH_JOINT_EXISTING_ENGINE_V1', case=CASE,
                    sources={'molecules.csv': sha(source / 'molecules.csv')}, original_source_hashes=hashes,
                    original_case_fingerprint=fingerprint, engine_sha256=sha(ENGINE), wrapper_sha256=sha(Path(__file__)),
                    protocol_sha256=sha(PROTOCOL), numpy='2.1.3', scipy='1.15.3', sklearn='1.6.1',
                    features='TRAIN-standardized log10(max(X_hat,1e-12)), log10(max(rho,1e-24)); two squares, interaction, rho-zero indicator',
                    model='Unchanged existing L2 LogisticRegression C=1, lbfgs, max_iter=2000, tol=1e-8; no weights or search',
                    split='Existing identity SHA256 namespace joint_confidence_CAL_v1, five groups; TEST=f, CAL=(f+1)%5, TRAIN=other three',
                    folds=folds, thresholds='Existing tied-score CAL maximum retention at empirical targets .01 and .05; primary .01 only',
                    primary_screen='Pooled out-of-group joint FDP<=.01, nonempty, all-truth recall>=.80 and TP retention>=.80',
                    secondary_5percent='Preserved legacy engine output only, cannot rescue primary failure',
                    leakage_boundaries='Identity only for split; truth only TRAIN labels and CAL threshold; test labels only counting; no class/geometry/seed/target spectra features',
                    exposure='All current case outcomes previously exposed; within-case grouped development, not fresh or independent validation',
                    probability_interpretation='joint_probability is an uncalibrated ranking score, not a molecular correctness probability',
                    no_solver_or_rho_rerun=True, no_GPU=True, no_next_case=True, no_hyperparameter_or_feature_search=True)
    write(prep / 'contract.json', contract)
    write(prep / 'seal.json', dict(status='FROZEN_BEFORE_CLASSIFIER_FIT', contract_sha256=sha(prep / 'contract.json')))
    print('PREPARED_CLASSIFIER_NOT_RUN', sha(prep / 'contract.json'), flush=True)


def validate(a):
    prep = a.output / 'prepared'; c = read(prep / 'contract.json')
    assert sha(prep / 'contract.json') == read(prep / 'seal.json')['contract_sha256']
    assert sha(ENGINE) == c['engine_sha256'] and sha(Path(__file__)) == c['wrapper_sha256'] and sha(PROTOCOL) == c['protocol_sha256']
    rows, hashes, fingerprint = sources(a.case_dir)
    assert hashes == c['original_source_hashes'] and fingerprint == c['original_case_fingerprint']
    assert sha(prep / 'input/molecules.csv') == c['sources']['molecules.csv']
    return rows, c


def run(a):
    _, c = validate(a)
    import numpy, scipy, sklearn
    assert (numpy.__version__, scipy.__version__, sklearn.__version__) == (c['numpy'], c['scipy'], c['sklearn'])
    spec = importlib.util.spec_from_file_location('unchanged_joint_engine', ENGINE)
    engine = importlib.util.module_from_spec(spec); spec.loader.exec_module(engine)
    engine.main(SimpleNamespace(source=a.output / 'prepared/input', contract=a.output / 'prepared/contract.json', output=a.output / 'fit'))


def metrics(rows, selected):
    true = {r['lipid_name'] for r in rows if r['molecular_truth']}
    reportable = {r['lipid_name'] for r in rows if r['reportable_truth']}
    raw = {r['lipid_name'] for r in rows if r['raw_solver_reported']}
    names = set(selected); tp = len(names & true); fp = len(names - true); raw_tp = len(raw & true)
    return dict(raw_TP=raw_tp, raw_FP=len(raw-true), raw_FN=len(true-raw), TP=tp, FP=fp, FN=len(true-names),
                FDP=fp/len(names) if names else None, TP_retention=tp/raw_tp if raw_tp else None,
                all_truth_recall=tp/len(true) if true else None,
                reportable_truth_recall=len(names & reportable)/len(reportable) if reportable else None,
                filter_induced_true_loss=raw_tp-tp, retained=len(names), coverage=len(names)/len(raw) if raw else None)


def review(a):
    import numpy as np
    source_rows, c = validate(a); fit = a.output / 'fit'; provenance = read(fit / 'provenance.json')
    assert provenance['script_sha256'] == c['engine_sha256'] and provenance['source_hashes'] == c['sources']
    for name, expected in provenance['files'].items():
        assert sha(fit / name) == expected, name
    assert sha(fit / 'contract.json') == sha(a.output / 'prepared/contract.json')
    rows = read(fit / 'out_of_group_records.json'); indexed = {r['lipid_name']: r for r in rows}
    assert len(rows) == len(indexed) == len(source_rows) and set(indexed) == {r['lipid_name'] for r in source_rows}
    for original in source_rows:
        got = indexed[original['lipid_name']]
        for key in ('molecular_truth', 'reportable_truth', 'raw_solver_reported', 'X_hat', 'rho_zero'):
            assert got[key] == original[key]
        assert got['fold'] == group(got['lipid_name']) and got['dataset_id'] == CASE
    def predict(items, model):
        logs = np.array([[math.log10(max(r['X_hat'], 1e-12)), math.log10(max(r['rho_zero'], 1e-24))] for r in items])
        z = (logs-np.array(model['mean']))/np.array(model['scale'])
        phi = np.column_stack((z[:, 0], z[:, 1], z[:, 0]**2, z[:, 0]*z[:, 1], z[:, 1]**2, [float(r['rho_zero'] == 0) for r in items]))
        value = phi @ np.array(model['coef'])[0] + model['intercept'][0]
        return [1/(1+math.exp(-float(v))) if v >= 0 else math.exp(float(v))/(1+math.exp(float(v))) for v in value]
    fold_reports = []; max_prediction_error = 0.
    for fold in range(5):
        seal = read(fit / f'fold_{fold}_seal.json'); model = seal['model']; cal = (fold+1)%5
        assert sha(fit / f'fold_{fold}_seal.json') == provenance['fold_seal_hashes'][fold]
        train = [r for r in source_rows if group(r['lipid_name']) not in (fold, cal) and r['raw_solver_reported']]
        calibration = [dict(r) for r in source_rows if group(r['lipid_name']) == cal and r['raw_solver_reported']]
        test = [r for r in rows if r['fold'] == fold]
        sets = [set(r['lipid_name'] for r in rs) for rs in (train, calibration, test)]
        assert not sets[0]&sets[1] and not sets[0]&sets[2] and not sets[1]&sets[2]
        for key, names in zip(('train_groups', 'calibration_groups', 'test_groups'), sets):
            assert model[key] == sorted(names)
        logs = np.array([[np.log10(max(r['X_hat'],1e-12)),np.log10(max(r['rho_zero'],1e-24))] for r in train])
        assert np.allclose(logs.mean(axis=0), model['mean'], rtol=1e-12, atol=0)
        assert np.allclose(np.maximum(logs.std(axis=0),1e-12), model['scale'], rtol=1e-12, atol=0)
        for row, prediction in zip(calibration, predict(calibration, model)):
            row['joint_probability'] = prediction
        for key in KEYS:
            for level in (.01, .05):
                best = None
                for value in sorted({r[key] for r in calibration}, reverse=True):
                    retained = [r for r in calibration if r[key] >= value]
                    if sum(not r['molecular_truth'] for r in retained)/len(retained) <= level:
                        best = value
                actual = seal['cutoffs'][key][str(level)]
                assert (best is None and actual is None) or (best is not None and actual is not None and math.isclose(best, actual, rel_tol=1e-12, abs_tol=1e-15))
        reported_test = [r for r in test if r['raw_solver_reported']]
        for row, pred in zip(reported_test, predict(reported_test, model)):
            max_prediction_error = max(max_prediction_error, abs(row['joint_probability']-pred))
            assert math.isclose(row['joint_probability'], pred, rel_tol=1e-12, abs_tol=1e-15)
        for row in test:
            for key in KEYS:
                for level in (.01, .05):
                    cutoff = seal['cutoffs'][key][str(level)]
                    expected = bool(row['raw_solver_reported'] and cutoff is not None and row[key] >= cutoff)
                    assert row[f'{key}_{level}_retained'] is expected
        fold_reports.append(dict(fold=fold, results={key: metrics(test, [r['lipid_name'] for r in test if r[f'{key}_0.01_retained']]) for key in KEYS}))
    aggregate = {key: metrics(rows, [r['lipid_name'] for r in rows if r[f'{key}_0.01_retained']]) for key in KEYS}
    joint = aggregate['joint_probability']
    reached = bool(joint['retained'] and joint['FDP'] <= .01 and joint['all_truth_recall'] >= .8 and joint['TP_retention'] >= .8)
    report = dict(status='CACHED_REVIEW_PASS', results_at_separate_CAL_1pct=aggregate, fold_results=fold_reports,
                  requested_1pct_80pct_developmental_target_met=reached,
                  maximum_saved_model_prediction_error=max_prediction_error, source_membership_and_split_verified=True,
                  models_refit_during_review=False, probabilities_not_risk_calibrated=True,
                  fresh_independent_case_or_real_data_validation=False,
                  original_first_case_rho_GO_unchanged=True, no_automatic_next_run=True,
                  provenance_sha256=sha(fit / 'provenance.json'), contract_sha256=sha(a.output / 'prepared/contract.json'))
    write(a.output / 'review.json', report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'run', 'review'))
    parser.add_argument('--case-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.case_dir, args.output = args.case_dir.resolve(), args.output.resolve()
    assert not args.output.is_relative_to(args.case_dir) and args.output != args.case_dir
    {'prepare': prepare, 'run': run, 'review': review}[args.action](args)
