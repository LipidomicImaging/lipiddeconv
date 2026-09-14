"""Explicit, sealed correction of two units in the immutable independent reviewer.

The original reviewer, contract, producer, diagnostics and scientific tables stay
unchanged. Reuse every original check; common/specific support must be raw sums,
as already specified by the pre-outcome contract, rather than normalized sums.
"""
from pathlib import Path
import argparse
import hashlib
import json
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/real_ce29_competition_audit_v1'
sys.path.insert(0, str(ROOT / 'analysis'))
import review_real_ce29_competition_audit as original

EXPECTED_REVIEW = '79358679a42253dccb20936b45f73d28b133bf327588b83065af5573e26a9428'
EXPECTED_CONTRACT = '05f0cb8d3222c8f9cc062a84aa08646b29f070861b839e987b71a3c612e6c022'
BASE_SUPPORT = original.support_values


def corrected_support_values(A, b, x, groups, physical):
    values = BASE_SUPPORT(A, b, x, groups, physical)
    for j, row in enumerate(values):
        row['common_fragment_support'] = float(b[physical['common'][:, j]].sum())
        row['specific_fragment_support'] = float(b[physical['specific'][:, j]].sum())
    return values


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n', encoding='utf-8')


def algebra_check():
    np = original.np
    A = np.array([[1., 0.], [0., 1.], [1., 1.]])
    b = np.array([2., 3., 4.])
    physical = dict(fragments=A > 0,
        common=np.array([[False, False], [False, False], [True, True]]),
        specific=np.array([[True, False], [False, True], [False, False]]))
    values = corrected_support_values(A, b, np.array([1., 1.]), [[0], [1]], physical)
    assert [r['common_fragment_support'] for r in values] == [4., 4.]
    assert [r['specific_fragment_support'] for r in values] == [2., 3.]
    return dict(status='PASS', no_solver=True, common_raw_sums=[4.,4.], specific_raw_sums=[2.,3.])


def prepare():
    assert original.sha(ROOT / 'analysis/review_real_ce29_competition_audit.py') == EXPECTED_REVIEW
    assert original.sha(OUT / 'audit_contract.json') == EXPECTED_CONTRACT
    assert not (OUT / 'review_units_correction.json').exists(), 'CORRECTION_ALREADY_SEALED'
    failed = original.read(OUT / 'validation_report.json')
    assert failed['status'] == 'STOP_INVALID_AUDIT'
    assert failed['error'] == 'AuditFailure: TABLE_VALUE: SUPPORT/PA 16:0_24:0/common_fragment_support'
    (OUT / 'validation_report_initial.json').write_bytes((OUT / 'validation_report.json').read_bytes())
    fixed_outputs = list(sorted(OUT.glob('*.csv'))) + [OUT / name for name in
        ('competition_graph_summary.json', 'decoy_coverage_audit.json', 'decision_summary.json')]
    write(OUT / 'review_units_correction.json', dict(
        status='REVIEW_IMPLEMENTATION_CORRECTION_BEFORE_CACHE_REVIEW',
        original_review_sha256=EXPECTED_REVIEW, contract_sha256=EXPECTED_CONTRACT,
        corrected_entrypoint_sha256=original.sha(Path(__file__)),
        initial_failure_sha256=original.sha(OUT / 'validation_report_initial.json'),
        root_cause='Original independent reviewer divided common/specific b sums by total fragment b; frozen contract and producer both require raw sums.',
        changed_fields=['common_fragment_support', 'specific_fragment_support'],
        invariant='No scientific contract, producer, source, solver, threshold, diagnostic array or table change.',
        original_scientific_output_hashes={p.name:original.sha(p) for p in fixed_outputs},
        algebra_check=algebra_check(), solver_calls=0))
    print('REVIEW_UNIT_CORRECTION_PREPARED_NO_SCIENTIFIC_RESULT_CHANGE', flush=True)


def run():
    seal = original.read(OUT / 'review_units_correction.json')
    assert original.sha(Path(__file__)) == seal['corrected_entrypoint_sha256']
    assert original.sha(ROOT / 'analysis/review_real_ce29_competition_audit.py') == seal['original_review_sha256']
    assert original.sha(OUT / 'audit_contract.json') == seal['contract_sha256']
    assert original.sha(OUT / 'validation_report_initial.json') == seal['initial_failure_sha256']
    for name, digest in seal['original_scientific_output_hashes'].items():
        assert original.sha(OUT / name) == digest, 'SCIENTIFIC_OUTPUT_CHANGED:' + name
    ack = original.read(OUT / 'review_correction_git.json')
    assert ack['local_commit'] == ack['remote_commit']
    for name in ('analysis/review_real_ce29_competition_support_units.py',
                 'results/real_ce29_competition_audit_v1/review_units_correction.json',
                 'results/real_ce29_competition_audit_v1/validation_report_initial.json'):
        content = subprocess.check_output(['git','-c','safe.directory='+ROOT.as_posix(),
            'show',ack['local_commit']+':'+name],cwd=ROOT)
        assert hashlib.sha256(content).hexdigest() == original.sha(ROOT / name), 'UNPUSHED_CORRECTION:' + name
    algebra_check()
    original.support_values = corrected_support_values
    try:
        report = original.review(OUT)
    except Exception as exc:
        report = dict(status=getattr(exc, 'status', 'STOP_INVALID_AUDIT'),
                      error=f'{type(exc).__name__}: {exc}', solver_calls=0)
    report['review_units_correction'] = dict(original_failure_preserved=True,
        correction_sha256=original.sha(OUT / 'review_units_correction.json'),
        entrypoint_sha256=original.sha(Path(__file__)), pushed_commit=ack['local_commit'],
        scientific_outputs_unchanged=all(original.sha(OUT / n)==h for n,h in seal['original_scientific_output_hashes'].items()))
    write(OUT / 'validation_report.json', report)
    print(json.dumps(report, ensure_ascii=False, allow_nan=False), flush=True)
    if report['status'] != 'PASS':
        raise SystemExit(1)


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('command', choices=['prepare','run'])
    globals()[parser.parse_args().command]()
