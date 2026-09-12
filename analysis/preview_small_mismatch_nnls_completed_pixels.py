"""One cached-prefix preview; no fitting, rho calculation or final decisions."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import argparse
import hashlib
import json
from pathlib import Path
import tarfile
import numpy as np


def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def read(p):
    return json.loads(p.read_text(encoding='utf-8'))


def write(p, value):
    data = (json.dumps(value, indent=2, allow_nan=False) + '\n').encode()
    assert not p.exists() or p.read_bytes() == data
    p.write_bytes(data)


def main(args):
    out = args.output.resolve()
    out.mkdir(exist_ok=False)
    with tarfile.open(args.archive) as archive:
        for member in archive.getmembers():
            target = (out / member.name).resolve()
            assert member.isfile() and target.is_relative_to(out) and not target.exists()
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(archive.extractfile(member).read())
    snap = read(out / 'snapshot_manifest.json')
    assert digest(args.prepared / 'design.json') == snap['design_sha256']
    d = read(args.prepared / 'design.json'); s = d['scientific']
    for name, expected in s['prepared_file_hashes'].items():
        assert digest(args.prepared / name) == expected
    for name, record in snap['files'].items():
        p = out / name
        assert p.stat().st_size == record['bytes'] and digest(p) == record['sha256']
    count = snap['foreground_prefix_pixels']
    X = np.zeros((391, count), dtype=np.float32)
    max_kkt = 0.
    for start in range(0, count, 250):
        stem = out / f'nnls_blocks/block_{start:06d}_{start+250:06d}'
        r = read(stem.with_suffix('.json'))
        assert r['fingerprint'] == d['fingerprint'] and r['start'] == start and r['stop'] == start+250
        assert r['all_pixels_checked_before_float32'] and r['KKT']['max_bound_ratio'] <= 1
        max_kkt = max(max_kkt, r['KKT']['max_bound_ratio'])
        with np.load(stem.with_suffix('.npz'), allow_pickle=False) as z:
            block = z['X_hat']
            assert block.shape == (391, 250) and np.isfinite(block).all() and (block >= 0).all()
            X[:, start:start+250] = block
    with np.load(args.prepared / 'prepared_arrays.npz', allow_pickle=False) as z:
        A = z['A_solver'].astype(np.float64); mask = z['mask']
        B = z['B'][:, mask][:, :count].astype(np.float64)
        truth_X = z['X_true'][:, mask][:, :count]
    metadata = read(args.prepared / 'metadata.json')
    names = metadata['lipid_name']; truth = set(s['truth_names'])
    reportable = set(s['reportable_truth_names']); total = int(mask.sum())
    means = X.mean(axis=1, dtype=np.float64)
    contribution = X.sum(axis=1, dtype=np.float64) / total
    # Diagnostic lower bound only. Account conservatively for positive summation
    # rounding; the original complete-case reporting gate stays unchanged.
    eps = np.finfo(np.float64).eps
    guard = 8 * count * eps * np.maximum(contribution, .001)
    guaranteed = contribution - guard > .001
    reported = means > .001
    truth_means = truth_X.mean(axis=1, dtype=np.float64)
    candidates = [dict(candidate_index=j, candidate_id=metadata['candidate_id'][j],
                       lipid_name=names[j], lipid_class=metadata['lipid_class'][j],
                       molecular_truth=names[j] in truth, reportable_truth=names[j] in reportable,
                       completed_pixel_mean_X_hat=float(means[j]),
                       whole_foreground_contribution=float(contribution[j]),
                       completed_pixel_mean_X_true=float(truth_means[j]),
                       partial_mean_reported=bool(reported[j]),
                       already_above_whole_foreground_gate=bool(guaranteed[j])) for j in range(391)]
    molecules = []
    for name in dict.fromkeys(names):
        indices = [j for j, value in enumerate(names) if value == name]
        molecules.append(dict(lipid_name=name, candidate_indices=indices,
                              molecular_truth=name in truth, reportable_truth=name in reportable,
                              partial_mean_reported=bool(reported[indices].any()),
                              already_above_whole_foreground_gate=bool(guaranteed[indices].any()),
                              truth_present_in_completed_pixels=bool((truth_X[indices] > 0).any()),
                              completed_truth_reportable=bool((truth_means[indices] > .001).any())))
    def metrics(flag):
        selected = {r['lipid_name'] for r in molecules if r[flag]}
        tp = len(selected & truth); fp = len(selected - truth)
        return dict(TP=tp, FP=fp, truth_not_in_this_reported_set=len(truth)-tp,
                    all_truth_recall=tp/len(truth), reported_count=len(selected),
                    raw_FDP=fp/len(selected) if selected else None)
    fitted = A @ X.astype(np.float64)
    error = np.linalg.norm(fitted-B, axis=0); norm = np.linalg.norm(B, axis=0)
    pixel_relative = np.divide(error, norm, out=np.zeros_like(error), where=norm > 0)
    coords = np.argwhere(mask)[:count]
    report = dict(status='PARTIAL_PIXEL_PREVIEW_NOT_FINAL', case=s['contract']['case'],
                  design_fingerprint=d['fingerprint'], source_archive_sha256=digest(args.archive),
                  snapshot_created_utc=snap['created_utc'], completed_pixels=count, total_foreground_pixels=total,
                  completed_fraction=count/total, pixel_selection='First completed foreground pixels in spatial traversal order; not a random sample',
                  spatial_row_range=[int(coords[:, 0].min()), int(coords[:, 0].max())],
                  spatial_column_range=[int(coords[:, 1].min()), int(coords[:, 1].max())],
                  all_truth_count=len(truth), global_reportable_truth_count=len(reportable),
                  truths_present_in_completed_pixels=sum(r['molecular_truth'] and r['truth_present_in_completed_pixels'] for r in molecules),
                  truths_reportable_in_completed_pixels=sum(r['molecular_truth'] and r['completed_truth_reportable'] for r in molecules),
                  partial_pixel_mean_raw=metrics('partial_mean_reported'),
                  whole_foreground_guaranteed_reported_lower_bound=metrics('already_above_whole_foreground_gate'),
                  lower_bound_not_reported_is_not_a_final_FN=True,
                  reconstruction_relative_residual=float(np.linalg.norm(error)/np.linalg.norm(B)),
                  per_pixel_relative_residual_quantiles=dict(zip(['min','q25','median','q75','q95','max'],np.quantile(pixel_relative,[0,.25,.5,.75,.95,1]).tolist())),
                  nominal_truth_forward_relative_residual=float(np.linalg.norm(A@truth_X-B)/np.linalg.norm(B)),
                  cached_blocks_verified=count//250, saved_precast_KKT_max_bound_ratio=max_kkt,
                  all_cached_outputs_finite_nonnegative=True, no_fitting_or_rho_computation=True,
                  final_rho_filter_metrics='PENDING_COMPLETE_CASE', final_GO_NO_GO='NOT_EVALUATED',
                  original_denominators_and_threshold_rule_unchanged=True, main_job_modified=False)
    write(out/'candidate_preview.json', candidates); write(out/'molecular_preview.json', molecules)
    write(out/'preview.json', report)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--prepared', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    main(parser.parse_args())
