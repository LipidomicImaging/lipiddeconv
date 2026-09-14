"""Preserve original review and repair its undefined zero-map cosine semantics."""
from pathlib import Path
import argparse
import json
import subprocess
import hashlib
import review_real_ce29_competition_support_units as units

original = units.original
ROOT, OUT = units.ROOT, units.OUT
BASE_SPATIAL = original.spatial_values


def corrected_spatial(a, b):
    row = BASE_SPATIAL(a, b)
    if original.np.linalg.norm(a) == 0 or original.np.linalg.norm(b) == 0:
        row['spatial_cosine'] = float('nan')
    return row


def prepare():
    require = original.require
    require(not (OUT/'review_zero_map_correction.json').exists(), 'CORRECTION_ALREADY_FROZEN')
    failed = original.read(OUT/'validation_report.json')
    require(failed['error'] == 'AuditFailure: TABLE_VALUE: SOLVER/PA 24:0_16:1/spatial_cosine_ista_nnls', 'UNEXPECTED_FAILURE')
    (OUT/'validation_report_units.json').write_bytes((OUT/'validation_report.json').read_bytes())
    units.write(OUT/'review_zero_map_correction.json', dict(
        reason='Frozen solver-agreement contract requires undefined zero-vector cosine to be NA. Producer does so; original independent reviewer returned zero.',
        entrypoint_sha256=original.sha(Path(__file__)),
        units_entrypoint_sha256=original.sha(ROOT/'analysis/review_real_ce29_competition_support_units.py'),
        units_binding_sha256=original.sha(OUT/'review_units_correction.json'),
        second_failure_sha256=original.sha(OUT/'validation_report_units.json'),
        changed_field='spatial_cosine only when at least one map has EXACT zero L2 norm',
        scientific_contract_and_outputs_unchanged=True, solver_calls=0))
    print('ZERO_MAP_REVIEW_CORRECTION_PREPARED', flush=True)


def run():
    seal=original.read(OUT/'review_zero_map_correction.json')
    old=original.read(OUT/'review_units_correction.json')
    assert original.sha(Path(__file__))==seal['entrypoint_sha256']
    assert original.sha(ROOT/'analysis/review_real_ce29_competition_support_units.py')==seal['units_entrypoint_sha256']
    assert original.sha(OUT/'review_units_correction.json')==seal['units_binding_sha256']
    assert original.sha(OUT/'validation_report_units.json')==seal['second_failure_sha256']
    for name,h in old['original_scientific_output_hashes'].items():
        assert original.sha(OUT/name)==h, 'SCIENTIFIC_OUTPUT_CHANGED:'+name
    ack=original.read(OUT/'review_zero_map_git.json')
    assert ack['local_commit']==ack['remote_commit']
    for name in ('analysis/review_real_ce29_competition_zero_maps.py',
                 'results/real_ce29_competition_audit_v1/review_zero_map_correction.json',
                 'results/real_ce29_competition_audit_v1/validation_report_units.json'):
        data=subprocess.check_output(['git','-c','safe.directory='+ROOT.as_posix(),'show',ack['local_commit']+':'+name],cwd=ROOT)
        assert hashlib.sha256(data).hexdigest()==original.sha(ROOT/name), 'UNPUSHED_CORRECTION:'+name
    units.algebra_check()
    assert original.np.isnan(corrected_spatial(original.np.zeros(3), original.np.ones(3))['spatial_cosine'])
    original.support_values=units.corrected_support_values
    original.spatial_values=corrected_spatial
    try:
        report=original.review(OUT)
    except Exception as exc:
        report=dict(status=getattr(exc,'status','STOP_INVALID_AUDIT'),error=f'{type(exc).__name__}: {exc}',solver_calls=0)
    report['review_corrections']=dict(initial_failure_sha256=old['initial_failure_sha256'],
        units_review_failure_sha256=seal['second_failure_sha256'],
        units_binding_sha256=seal['units_binding_sha256'], zero_map_binding_sha256=original.sha(OUT/'review_zero_map_correction.json'),
        review_commit=ack['local_commit'], original_script_preserved=True,
        scientific_outputs_unchanged=all(original.sha(OUT/n)==h for n,h in old['original_scientific_output_hashes'].items()))
    units.write(OUT/'validation_report.json',report)
    print(json.dumps(report,ensure_ascii=False,allow_nan=False),flush=True)
    if report['status']!='PASS': raise SystemExit(1)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('command', choices=['prepare','run'])
    globals()[parser.parse_args().command]()
