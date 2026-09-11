"""Cached-only independent review of finite-reference CAL development scores."""
import argparse
import hashlib
import json
import csv
import math
from pathlib import Path
from review_cal_channel_prediction_pilot import ranking


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())


def main(out):
    provenance=read(out/'provenance.json')
    for name,h in provenance['files'].items():assert sha(out/name)==h,name
    assert sha(out/'executed_runner.py')==provenance['script_sha256']
    rows=read(out/'scores.json');assert len(rows)==len({r['lipid_name'] for r in rows})==211
    source=Path(__file__).resolve().parents[1]/'results/mild_nnls_cal_pilot/molecular_units.json'
    assert sha(source)==provenance['source_sha256']
    original={r['lipid_name']:r for r in read(source)}
    assert set(original)=={r['lipid_name'] for r in rows}
    for r in rows:
        for k,v in original[r['lipid_name']].items():assert r[k]==v,(r['lipid_name'],k)
        for k in ('flexible_identity_score','unfloored_signed_gain','full_SSE','deleted_SSE','dual','complementarity'):
            assert math.isfinite(r[k])
        assert r['flexible_identity_score']>=0 and r['unfloored_signed_gain']>=-1e-10
    tp=sum(r['molecular_truth'] for r in rows);assert tp==125
    summary=dict(status='REVIEWED_CAL_ONLY',raw_TP=tp,raw_FP=len(rows)-tp,raw_FN=0,methods={},
                 sources_and_outputs_verified=True,limitation='Single exposed CAL case; model-family-matched references, no independent FDR guarantee.')
    curves=[]
    for score in ('rho_zero','X_hat','flexible_identity_score'):
        result,curve=ranking(rows,score,125,tp,125,len(rows));summary['methods'][score]=result
        curves.extend(dict(score=score,**r) for r in curve)
    recalls={s:m['retrospective_best']['0.05']['all_truth_recall'] for s,m in summary['methods'].items()}
    summary['decision']='PROMISING_REQUIRES_INDEPENDENT_VALIDATION' if recalls['flexible_identity_score']-max(recalls['rho_zero'],recalls['X_hat'])>=.1-1e-12 else 'STOP_THIS_VERSION_NO_PRESPECIFIED_RECALL_GAIN'
    summary['full_loss_ratio_to_fixed_library']=provenance['full_SSE']/provenance['base_SSE']
    (out/'review.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    with (out/'threshold_curves.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(curves[0]));w.writeheader();w.writerows(curves)
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('output',type=Path);main(p.parse_args().output)
