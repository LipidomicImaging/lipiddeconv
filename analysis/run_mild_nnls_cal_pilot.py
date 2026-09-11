"""One paired CAL MILD pixelwise NNLS test, reusing existing CLEAN results."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback

for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = ''


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def read(p):
    return json.loads(p.read_text())


def write(p, value):
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix+'.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    tmp.replace(p)


def main(args):
    import numpy as np
    import scipy
    sys.path.insert(0, str(args.root/'analysis'))
    sys.path.insert(0, str(args.root/'src'))
    import run_nnls_solver_baseline as baseline
    import run_v58_spectral_library_mismatch_fdr_recalibration as v
    v.dependencies()
    args.output.mkdir(parents=True, exist_ok=False)
    dataset = 'CAL_R1_K125'
    va = v.parse_args(['--dataset', 'MILD__'+dataset,
                      '--output-dir', str(args.root/'results/v58_spectral_library_mismatch_fdr_recalibration_compact_recovery'),
                      '--v57-output', str(args.root/'results/v57_spectral_spatial_identity_confidence_benchmark'),
                      '--asset-root', str(args.asset_root)])
    parent, _, _ = v.parent_provenance(va)
    context = v.load_context(va, parent)
    targets, _, _ = v.build_targets(context)
    design = read(va.output_dir/'design.json')
    audit = v.checked_audit(va, design)
    case, row = v.mismatch_case(context, parent, targets, 'MILD__'+dataset)
    row = v.v57.canonical(row)
    for k, val in audit['datasets']['MILD__'+dataset].items():
        if k in ('forward_mismatch_residual', 'sentinel_gate_residual'):
            assert abs(row[k]-val) <= 1e-12
        else:
            assert row[k] == val, k
    A = np.asarray(context['A_solver'], dtype=np.float64)
    assert A.shape == (1084, 391)
    assert v.array_sha(context['A_solver']) == design['scientific']['target_library_hashes']['A_solver_sha256']
    clean_root = args.root/'results/nnls_solver_baseline_k125_cpu4'
    clean = clean_root/dataset
    clean_result = read(clean/'result.json')
    assert sha(clean/'learned_arrays.npz') == clean_result['arrays_sha256']
    assert sha(clean/'molecular_units.json') == clean_result['units_sha256']
    assert clean_result['fingerprint'] == read(clean_root/'design.json')['fingerprint']
    assert clean_result['B_sha256'] == read(va.v57_output/'clean'/dataset/'runtime_contract.json')['B_sha256']
    paths = [clean_root/'design.json', clean/'result.json', clean/'molecular_units.json',
             va.output_dir/'MILD'/dataset/'molecular_false_negative_records.csv', va.v57_output/'calibration_records.csv']
    sources = {str(p): sha(p) for p in paths}
    contract = dict(dataset=dataset, domain='MILD', cached_control='CLEAN NNLS same CAL case',
                    solver='existing baseline.solve_pixel: scipy NNLS all391 columns, maxiter3910',
                    precision='float64 solve, float32 stored maps; float64 foreground reporting mean',
                    workers=4, reporting_gate=.001, rho='unchanged v54.learned_records/rho_values',
                    validation='CAL-only descriptive raw TP/FP/FN and tied threshold envelopes; no deployed cutoff',
                    no_HOLD=True, no_GPU=True, no_target_spectra_in_solver=True,
                    implementation_sha256=sha(Path(__file__)), baseline_sha256=sha(Path(baseline.__file__)),
                    numpy=np.__version__, scipy=scipy.__version__, sources=sources,
                    scientific_input_audit=row, v58_fingerprint=design['design_fingerprint'])
    write(args.output/'contract.json', contract)
    mask = case['foreground_mask']; B = case['B_sim']
    assert np.all(B[:, ~mask] == 0)
    spectra = B[:, mask].astype(np.float64)
    x = np.zeros((391, spectra.shape[1]), dtype=np.float32)
    started = time.monotonic(); max_dual = max_comp = 0.
    with ProcessPoolExecutor(max_workers=4, initializer=baseline.initialize_worker, initargs=(A,)) as pool:
        for i, solution in enumerate(pool.map(baseline.solve_pixel, (spectra[:, j] for j in range(spectra.shape[1])), chunksize=16)):
            assert np.isfinite(solution).all() and (solution >= 0).all()
            if i % 500 == 0:
                g = A.T@(A@solution-spectra[:, i])
                max_dual = max(max_dual, float(np.maximum(-g, 0).max()))
                max_comp = max(max_comp, float(np.abs(g*solution).max()))
            x[:, i] = solution
            if i % 250 == 0:
                write(args.output/'status.json', dict(status='RUNNING', stage='PIXEL_NNLS', pixels_done=i+1,
                                                     total_pixels=spectra.shape[1], seconds=time.monotonic()-started))
                print('PIXELS', i+1, '/', spectra.shape[1], flush=True)
    X = np.zeros_like(case['X_true']); X[:, mask] = x
    B_hat = np.zeros_like(B); B_hat[:, mask] = (A@x.astype(np.float64)).astype(np.float32)
    assert np.isfinite(B_hat).all()
    np.savez_compressed(args.output/'learned_arrays.npz', X_hat=X, B_hat=B_hat)
    write(args.output/'status.json', dict(status='RUNNING', stage='RHO'))
    records, raw = v.v57.v54.learned_records(dataset, case, dict(X_hat=X, B_hat=B_hat), context, 1)
    report = dict(dataset_id=dataset, all_truth_molecular_identity_count=case['all_truth_molecular_identity_count'],
                  reportable_truth_molecular_identity_count=case['reportable_truth_molecular_identity_count'],
                  learned_raw_identity_performance=raw)
    units = v.v57.v54.molecular_units(records, {dataset: report})
    write(args.output/'molecular_units.json', units)
    write(args.output/'reported_candidate_records.json', records)
    write(args.output/'result.json', report)
    comparisons = {}
    for domain, solver, us in [('MILD', 'NNLS', units), ('CLEAN', 'NNLS', read(clean/'molecular_units.json'))]:
        comparisons[domain+'_'+solver] = us
    for domain, p in [('CLEAN', paths[-1]), ('MILD', paths[-2])]:
        comparisons[domain+'_ISTA'] = [r for r in v.v57.rows(p) if r.get('base_dataset_id', r['dataset_id']) == dataset and r['raw_solver_reported']]
        assert comparisons[domain+'_ISTA'], 'MISSING_PAIRED_ISTA_RECORDS'
    write(args.output/'comparison_units.json', comparisons)
    # All candidate records, including misses, are retained for later independent auditing.
    means = X[:, mask].mean(axis=1, dtype=np.float64)
    true_indices = set(case['active_indices']); reportable_indices = set(case['reportable_truth_indices'])
    metadata = context['metadata']
    truth_names = {str(metadata['lipid_name'][i]) for i in true_indices}
    reportable_names = {str(metadata['lipid_name'][i]) for i in reportable_indices}
    write(args.output/'all_candidate_abundances.json', [dict(candidate_index=i, lipid_name=str(metadata['lipid_name'][i]),
          X_hat=float(means[i]), candidate_truth=i in true_indices, molecular_truth=str(metadata['lipid_name'][i]) in truth_names,
          reportable_truth=str(metadata['lipid_name'][i]) in reportable_names, raw_solver_reported=bool(means[i] > .001)) for i in range(391)])
    for p, h in sources.items():
        assert sha(Path(p)) == h, p
    files = {p.name: sha(p) for p in args.output.iterdir() if p.is_file() and p.name != 'status.json'}
    write(args.output/'provenance.json', dict(status='COMPLETE_AWAITING_REVIEW', seconds=time.monotonic()-started,
          normal_completion=True, all_outputs_finite=True, sampled_KKT_max_dual=max_dual,
          sampled_KKT_max_complementarity=max_comp, source_hashes_unchanged=True, files=files,
          retained_arrays_path=str(args.output/'learned_arrays.npz')))
    write(args.output/'status.json', dict(status='COMPLETE_AWAITING_REVIEW'))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--asset-root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    try:
        main(a)
    except BaseException:
        # A failed attempt is evidence and cannot be silently overwritten by a retry.
        if a.output.exists():
            write(a.output/'failure.json', dict(error=traceback.format_exc()))
        raise
