"""Independent cached-data accounting and primal/dual proof review for the CE pilot."""
import argparse
import hashlib
import json
import os
from pathlib import Path

for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'


def read(p):
    return json.loads(p.read_text(encoding='utf-8'))


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def metrics(cases, epsilon=None, score=None, threshold=None):
    truth = sum(c['truth_count'] for c in cases)
    reportable = sum(c['reportable_truth_count'] for c in cases)
    raw_tp = sum(r['molecular_truth'] for c in cases for r in c['records'])
    raw_n = sum(len(c['records']) for c in cases)
    selected = []
    abstain = 0
    for case in cases:
        compatible = epsilon is None or case['full']['upper'] <= epsilon
        abstain += not compatible
        selected.extend(r for r in case['records'] if compatible
                        and (epsilon is None or r['deleted']['lower'] > epsilon)
                        and (score is None or (threshold is not None and r[score] >= threshold)))
    tp = sum(r['molecular_truth'] for r in selected)
    false = len(selected)-tp
    return dict(TP=tp, FP=false, FN=truth-tp, raw_solver_TP=raw_tp,
                raw_solver_FP=raw_n-raw_tp, raw_solver_FN=truth-raw_tp,
                filter_induced_true_loss=raw_tp-tp, TP_retention=tp/raw_tp,
                all_truth_recall=tp/truth, reportable_truth_recall=sum(r['reportable_truth'] for r in selected)/reportable,
                FDR=false/len(selected) if selected else 0., retained_count=len(selected),
                distinct_identities=len({r['lipid_name'] for r in selected}),
                case_count=len(cases), abstained_cases=abstain, abstention_fraction=abstain/len(cases))


def proof_bounds(A, fractions, components, kept, b, primal, dual, removed):
    """Rebuild block equations directly, independently of the LP solver assembly."""
    import numpy as np
    inverse = {int(original):i for i, original in enumerate(kept)}
    rows = [i for i,c in enumerate(components) if c['candidate_index'] in inverse]
    owners = np.array([inverse[components[i]['candidate_index']] for i in rows], dtype=int)
    low = np.array([components[i]['lower'] for i in rows])
    high = np.array([components[i]['upper'] for i in rows])
    varying = A[:,owners]*fractions[:,rows]
    fixed = A.copy()
    for i,j in enumerate(owners):
        fixed[:,j] -= varying[:,i]
    assert fixed.min() >= -1e-10
    fixed = np.maximum(fixed,0)
    n,g = len(kept),len(rows)
    m = len(b)
    assert primal.shape == (n+g+m,) and dual.shape == (2*m+2*g,)
    assert np.isfinite(primal).all() and np.isfinite(dual).all()
    b = b/b.sum()
    x,w,e = primal[:n],primal[n:n+g],primal[n+g:]
    assert min(primal) >= 0 and max(dual) <= 0
    assert np.all(x[removed] == 0) and np.all(w[np.isin(owners,removed)] == 0)
    assert max(low*x[owners]-w,default=0) <= 1e-8
    assert max(w-high*x[owners],default=0) <= 1e-8
    residual = fixed@x+varying@w-b
    assert np.max(abs(residual)-e) <= 1e-8
    minimum_mass = fixed.sum(axis=0)
    for i,j in enumerate(owners):
        minimum_mass[j] += low[i]*varying[:,i].sum()
    xmax = 4/minimum_mass
    wmax = high*xmax[owners]
    xmax[removed] = 0
    wmax[np.isin(owners,removed)] = 0
    upper = np.r_[xmax,wmax,np.full(m,2.)]
    assert np.max(primal-upper) <= 1e-8
    y = dual.astype(np.longdouble)
    residual_dual = y[:m]-y[m:2*m]
    hi_dual,lo_dual = y[2*m:2*m+g],y[2*m+g:]
    reduced_x = -(fixed.astype(np.longdouble).T@residual_dual)
    for i,j in enumerate(owners):
        reduced_x[j] += high[i]*hi_dual[i]-low[i]*lo_dual[i]
    reduced_w = -(varying.astype(np.longdouble).T@residual_dual)-hi_dual+lo_dual
    reduced_e = 1+y[:m]+y[m:2*m]
    reduced = np.r_[reduced_x,reduced_w,reduced_e]
    lower = max(0.,float(b.astype(np.longdouble)@residual_dual
                        +(upper.astype(np.longdouble)*np.minimum(reduced,0)).sum())-1e-8)
    upper_value = float(abs(residual).sum())+1e-8
    return lower,upper_value


def review(root):
    import numpy as np
    manifest = read(root/'output_manifest.json')
    for name, expected in manifest.items():
        assert sha(root/name) == expected, name
    uncertainty = root/'uncertainty'
    source = read(uncertainty/'provenance.json')
    frozen_source = read(Path('results/ce133_uncertainty_v1_ready/provenance.json'))
    assert source == frozen_source
    for name,expected in source['outputs'].items():
        assert sha(uncertainty/name) == expected
    summary = read(root/'summary.json')
    seal = read(root/'calibration_seal.json')
    assert sha(root/'calibration_seal.json') == summary['calibration_seal_sha256']
    protocol = read(root/'protocol.json')
    assert protocol['continue_target'] == {'FDR':.01,'TP_retention':.4}
    assert protocol['script_sha256'] == sha(Path('analysis/run_ce_uncertainty_identity_pilot.py'))
    assert read(root/'self_test.json')['status'] == 'PASS'
    with np.load(uncertainty/'component_fractions.npz') as data:
        fractions = data['fractions']
    components = read(uncertainty/'components.json')
    expected = {f'{kind}__{split}_R{r}_K125' for kind,split in
                [('MILD','CAL'),('close_neighbor','HOLD'),('relatively_isolated','HOLD')] for r in (1,2)}
    assert {p.name for p in (root/'scores').iterdir() if p.is_dir()} == expected
    cases, proof_count, shortcuts = [],0,0
    proof_review = {}
    for key in sorted(expected):
        case = read(root/'scores'/key/'scores.json')
        inputs = root/'inputs'/key
        item = read(inputs/'pilot_input.json')
        assert case['dataset'] == key and case['truth_count'] == case['reportable_truth_count'] == 125
        assert case['role'] == ('DEVELOPMENT_CAL' if '_R1_' in key else 'DEVELOPMENT_EVAL')
        for name,expected_hash in item['original_artifacts'].items():
            assert sha(inputs/name) == expected_hash
        assert sha(inputs/'scoring_inputs.npz') == item['scoring_inputs_sha256']
        with np.load(inputs/'scoring_inputs.npz') as data:
            A,b,kept = data['A'],data['b'],data['kept']
        assert hashlib.sha256(A.astype(np.float32).tobytes()).hexdigest() == case['A_sha256']
        assert len(set(kept.tolist())) == len(kept) == len(case['names'])
        proofs_path = root/'scores'/key/'proofs.npz'
        assert sha(proofs_path) == case['proofs_sha256']
        proof_gaps = []
        with np.load(proofs_path) as proofs:
            lo,hi = proof_bounds(A,fractions,components,kept,b,proofs['full_primal'],proofs['full_dual'],[])
            assert abs(lo-case['full']['lower']) <= 1e-9 and abs(hi-case['full']['upper']) <= 1e-9
            assert hi-lo <= 1e-6
            proof_count += 1
            proof_gaps.append(hi-lo)
            assert len({r['lipid_name'] for r in case['records']}) == len(case['records'])
            for row in case['records']:
                removed = [i for i,n in enumerate(case['names']) if n == row['lipid_name']]
                assert removed == row['removed'] and removed
                index = row['proof_index']
                shortcut = row['deleted']['proof'] == 'FULL_ZERO_IDENTITY_FEASIBLE_WITNESS'
                px = proofs['full_primal' if shortcut else f'primal_{index}']
                dy = proofs['full_dual' if shortcut else f'dual_{index}']
                low,high = proof_bounds(A,fractions,components,kept,b,px,dy,removed)
                assert abs(high-row['deleted']['upper']) <= 1e-9
                assert row['deleted']['lower'] <= low+1e-9
                if not shortcut:
                    assert abs(low-row['deleted']['lower']) <= 1e-9
                assert high-row['deleted']['lower'] <= 1e-6+1e-9
                proof_gaps.append(high-row['deleted']['lower'])
                proof_count += 1
                shortcuts += shortcut
        original = {r['lipid_name']:r for r in item['records']}
        assert set(original) == {r['lipid_name'] for r in case['records']}
        for row in case['records']:
            for field in ('X_hat','rho_zero','molecular_truth','reportable_truth'):
                assert row[field] == original[row['lipid_name']][field]
        values = metrics([case],seal['epsilon'])
        for name,value in values.items():
            assert summary['by_case'][key][name] == value,(key,name)
        retained = read(root/'scores'/key/'retained_records.json')
        assert len(retained) == len(case['records'])
        assert all(r['retained'] == (case['full']['upper'] <= seal['epsilon'] and r['deleted']['lower'] > seal['epsilon']) for r in retained)
        proof_review[key] = dict(status='PASS',identities=len(case['records']),max_primal_dual_gap=max(proof_gaps),full_upper=case['full']['upper'],full_seconds=case['full']['seconds'])
        cases.append(case)
        print('PROOFS_AND_COUNTS_PASS',key,flush=True)
    cal = [c for c in cases if c['role']=='DEVELOPMENT_CAL']
    evaluation = [c for c in cases if c['role']=='DEVELOPMENT_EVAL']
    events = sorted({0.,1.,*(c['full']['upper'] for c in cal),*(r['deleted']['lower'] for c in cal for r in c['records'])})
    options = [dict(epsilon=e,**metrics(cal,e)) for e in events]
    eligible = [v for v in options if v['retained_count'] and v['FDR'] <= .01]
    chosen = max(eligible,key=lambda v:(v['TP'],-v['FP'],v['epsilon'])) if eligible else next(v for v in options if v['epsilon']==1)
    assert chosen['epsilon'] == seal['epsilon']
    for role,data in [('calibration',cal),('evaluation',evaluation)]:
        for name,value in metrics(data,seal['epsilon']).items():
            assert summary[role][name] == value
    for kind in ('MILD','close_neighbor','relatively_isolated'):
        for name,value in metrics([c for c in evaluation if c['kind']==kind],seal['epsilon']).items():
            assert summary['by_challenge'][kind][name] == value
    for score in ('rho_zero','X_hat'):
        options = [dict(threshold=t,**metrics(cal,score=score,threshold=t))
                   for t in sorted({r[score] for c in cal for r in c['records']})]
        eligible = [v for v in options if v['retained_count'] and v['FDR'] <= .01]
        best = max(eligible,key=lambda v:(v['TP'],-v['FP'],v['threshold'])) if eligible else {'threshold':None}
        assert best['threshold'] == seal['baseline'][score]['threshold']
        for name,value in metrics(evaluation,score=score,threshold=best['threshold']).items():
            assert summary['baselines'][score][name] == value
    e = metrics(evaluation,seal['epsilon'])
    assert summary['go'] == bool(e['retained_count'] and e['FDR'] <= .01 and e['TP_retention'] >= .4)
    result = dict(status='PASS',output_hash_count=len(manifest),cases=6,proofs_checked=proof_count,
                  zero_coefficient_witnesses=shortcuts,case_proofs=proof_review,
                  all_counts_and_abstention_denominators_verified=True,calibration_recomputed_from_R1_only=True,
                  baseline_calibration_recomputed_from_R1_only=True,
                  inference='Numerically checked global LP bounds for this specified uncertainty model; not an exact-arithmetic or population FDR guarantee.',
                  go=summary['go'])
    (root/'independent_review.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8',newline='\n')
    print(json.dumps({k:v for k,v in result.items() if k!='case_proofs'}))


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output',type=Path)
    review(parser.parse_args().output)
