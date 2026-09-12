"""One complete-library 5% mismatch case: CPU NNLS and unchanged rho_zero.

Prepare immutable inputs first, then run; no GPU, extra case or automatic expansion.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_name] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

CASE = "NNLS_FULL_LIBRARY_5PCT__CAL_R71_K125"
SOURCE_CASE = "SUPPORTED_MISMATCH__CAL_R71_K125"
SIGNAL_TARGET = 0.6036783456802368
A_SHA = "b9e05e185ffa692966b86022f5487fcdabff92f330a0a389a6bb4889a175a447"
CONTRACT = dict(case=CASE, role="FIRST_CAL_DEVELOPMENT_ONLY", seed=7301, relative_sd=.05,
                solver="existing per-foreground-pixel scipy NNLS, full391, float64, maxiter3910; store X_hat float32",
                workers=4, blas_threads=1, report_gate=.001, rho_fixed_threshold=.001,
                rho="unchanged identity_weights/rho_zero_from_weighted_case on foreground mean B; full391 competition",
                identity="any reported same-name candidate; max reported alias rho and sum reported alias X_hat",
                threshold_selection="distinct rho ties; >= threshold; max TP at empirical FDP<=.01, then min FP, then highest threshold",
                go="nonempty chosen set, empirical FDP<=.01, TP retention>=.40; >=.60 strong",
                signal_target=SIGNAL_TARGET, X_scaling="ONE common dataset scalar; preserve complete cached spatial shape",
                KKT="all pixels before float32: dual and active gradient <=64*eps64*max(M,N)*abs(A).T@(abs(Ax)+abs(b))",
                KKT_claim="floating-point engineering validation, not an exact-arithmetic optimality theorem",
                no_GPU=True, no_other_cases=True, no_CE_endpoint_inputs=True, no_new_score=True,
                interpretation="One developmental CAL case; selected empirical FDP is not independent FDR validation")


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def read(p):
    return json.loads(p.read_text(encoding="utf-8"))


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def sha(p):
    h = hashlib.sha256()
    with p.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024**2), b""):
            h.update(block)
    return h.hexdigest()


def ah(value):
    return hashlib.sha256(value.tobytes(order="C")).hexdigest()


def write(p, value, *, replace=False):
    p.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(value, indent=2, allow_nan=False) + "\n").encode()
    if p.exists() and not replace:
        require(p.read_bytes() == data, "IMMUTABLE_RECORD_CHANGED:" + p.name)
        return
    temporary = p.with_name(p.name + ".tmp")
    require(not temporary.exists(), "PARTIAL_WRITE_PRESERVED:" + temporary.name)
    with temporary.open("xb") as f:
        f.write(data)
    temporary.replace(p)


def npz_once(p, **arrays):
    import numpy as np
    require(not p.exists(), "EXISTING_ARRAY_PRESERVED:" + p.name)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("xb") as f:
        np.savez_compressed(f, **arrays)


def modules():
    import numpy as np
    import scipy
    import run_nnls_solver_baseline as baseline
    import rho_zero as rho
    import small_relative_spectral_mismatch as mismatch
    files = {str(Path(p).resolve()): sha(Path(p)) for p in
             (__file__, baseline.__file__, rho.__file__, mismatch.__file__)}
    return np, baseline, rho, mismatch, dict(files=files, numpy=np.__version__, scipy=scipy.__version__, epsilon_q=rho.EPSILON_Q)


def source_inputs(args, np):
    source = args.cached_case
    require(source.name == SOURCE_CASE, "ONLY_THE_FROZEN_FIRST_SPATIAL_CASE_IS_ALLOWED")
    info, metadata = read(source / "input.json"), read(source / "metadata.json")
    require(info["key"] == SOURCE_CASE and len(info["truth_indices"]) == 125, "SOURCE_TRUTH_MEMBERSHIP_CHANGED")
    require(sha(source / "observation_truth.npz") == info["observation_file_sha256"], "CACHED_SPATIAL_CONTAINER_CHANGED")
    with np.load(source / "observation_truth.npz", allow_pickle=False) as z:
        X, mask = z["X_true"].copy(), z["mask"].copy()  # Never read old B or A_target.
    require(ah(X) == info["X_true_sha256"] and mask.dtype == np.bool_ and X.shape == (391, *mask.shape)
            and np.isfinite(X).all() and (X >= 0).all() and np.all(X[:, ~mask] == 0), "CACHED_SPATIAL_TRUTH_INVALID")
    provenance = read(args.component_assets / "provenance.json")
    require(provenance["status"] == "COMPLETE", "PHYSICAL_COMPONENT_BUILDER_NOT_COMPLETE")
    for name, record in provenance["outputs"].items():
        path = (args.component_assets / name).resolve()
        require(path.is_relative_to(args.component_assets) and sha(path) == record["sha256"]
                and path.stat().st_size == record["bytes"], "COMPONENT_ASSET_CHANGED:" + name)
    with np.load(args.component_assets / "components.npz", allow_pickle=False) as z:
        A, components, owner = z["A"].copy(), z["components"].copy(), z["owner"].copy()
    rows = read(args.component_assets / "component_metadata.json")
    catalog = read(args.component_assets / "metadata.json")
    require(A.shape == (1084, 391) and ah(A) == A_SHA == info["A_solver_sha256"], "NOMINAL_LIBRARY_CHANGED")
    require(owner.tolist() == [r["candidate_index"] for r in rows], "COMPONENT_OWNER_CHANGED")
    require(all(catalog[k] == metadata[k] for k in ("candidate_id", "lipid_name", "lipid_class")), "CANDIDATE_CATALOG_CHANGED")
    for row in rows:
        j = row["candidate_index"]; key = row["physical_ion_key"]
        require(key["identity"] == catalog["lipid_name"][j] and key["adduct"] == catalog["adduct"][j]
                and key["CE"] == catalog["collision_energy_ev"][j], "COMPONENT_IDENTITY_ADDUCT_CE_CHANGED")
    paths = [source / name for name in ("input.json", "metadata.json", "observation_truth.npz")]
    paths += [args.component_assets / name for name in (*provenance["outputs"], "provenance.json")]
    return A, components, rows, catalog, X, mask, info, {str(p): sha(p) for p in paths}


def prepare(args):
    np, _, _, mismatch, implementation = modules()
    if (args.output / "design.json").exists():
        validate(args, implementation); print("PREPARED_INPUTS_VERIFIED", flush=True); return
    require(not any(args.output.iterdir()), "NONEMPTY_UNSEALED_OUTPUT_PRESERVED")
    A, C, rows, metadata, X, mask, info, sources = source_inputs(args, np)
    target, audit = mismatch.build_target_library(A, C, rows, seed=7301, relative_sd=.05)
    fg_B = target.astype(np.float64) @ X[:, mask].astype(np.float64)
    before = float(np.median(np.linalg.norm(fg_B, axis=0)))
    require(math.isfinite(before) and before > 0, "INVALID_UNSCALED_SIGNAL")
    scalar = SIGNAL_TARGET / before
    X_target = X.astype(np.float64) * scalar
    B = np.zeros((1084, *mask.shape), dtype=np.float32)
    B[:, mask] = (fg_B * scalar).astype(np.float32)
    achieved = float(np.median(np.linalg.norm(B[:, mask].astype(np.float64), axis=0)))
    require(abs(achieved - SIGNAL_TARGET) <= 8 * np.finfo(np.float32).eps * SIGNAL_TARGET, "SIGNAL_SCALAR_FAILED")
    truth_indices = np.flatnonzero(np.any(X_target[:, mask] > 0, axis=1)).tolist()
    reportable_indices = np.flatnonzero(X_target[:, mask].mean(axis=1, dtype=np.float64) > .001).tolist()
    require(set(truth_indices) == set(info["truth_indices"]) and len(truth_indices) == 125, "TRUTH_CHANGED_BY_GLOBAL_SCALING")
    truth_names = sorted({metadata["lipid_name"][j] for j in truth_indices})
    reportable_names = sorted({metadata["lipid_name"][j] for j in reportable_indices})
    perturbed = [j for j in truth_indices if audit["relative_column_changes"][j] > 1e-10]
    require(len(truth_names) == 125, "MOLECULAR_TRUTH_COUNT_CHANGED")
    npz_once(args.output / "prepared_arrays.npz", A_solver=A, A_target=target, X_true=X_target, B=B, mask=mask)
    write(args.output / "metadata.json", metadata); write(args.output / "perturbation_audit.json", audit)
    scientific = dict(contract=CONTRACT, implementation=implementation, source_hashes=sources,
                      source_X_sha256=ah(X), target_X_sha256=ah(X_target), B_sha256=ah(B), A_solver_sha256=ah(A), A_target_sha256=ah(target),
                      global_dataset_scalar=scalar, foreground_median_B_norm=achieved,
                      truth_indices=truth_indices, truth_names=truth_names, reportable_truth_indices=reportable_indices,
                      reportable_truth_names=reportable_names, actual_perturbed_truth_indices=perturbed,
                      actual_perturbed_truth_names=sorted({metadata["lipid_name"][j] for j in perturbed}),
                      truth_count=len(truth_names), reportable_truth_count=len(reportable_names), actual_perturbed_truth_count=len(perturbed),
                      prepared_file_hashes={n: sha(args.output / n) for n in ("prepared_arrays.npz", "metadata.json", "perturbation_audit.json")})
    design = dict(fingerprint=hashlib.sha256(canonical(scientific)).hexdigest(), scientific=scientific)
    write(args.output / "design.json", design)
    write(args.output / "design_seal.json", dict(status="FROZEN_BEFORE_NNLS_AND_RHO", fingerprint=design["fingerprint"], design_sha256=sha(args.output / "design.json")))
    write(args.output / "status.json", dict(status="PREPARED_NOT_RUN", case=CASE))
    print("PREPARED_NOT_RUN", design["fingerprint"], flush=True)


def validate(args, implementation):
    d, seal = read(args.output / "design.json"), read(args.output / "design_seal.json")
    require(hashlib.sha256(canonical(d["scientific"])).hexdigest() == d["fingerprint"] == seal["fingerprint"]
            and sha(args.output / "design.json") == seal["design_sha256"], "DESIGN_CHANGED")
    s = d["scientific"]
    require(s["contract"] == CONTRACT and s["implementation"] == implementation, "RUN_CONTRACT_OR_IMPLEMENTATION_CHANGED")
    for path, expected in s["source_hashes"].items():
        require(sha(Path(path)) == expected, "SOURCE_CHANGED:" + path)
    for name, expected in s["prepared_file_hashes"].items():
        require(sha(args.output / name) == expected, "PREPARED_INPUT_CHANGED:" + name)
    return d


def kkt_check(A, b, x, np):
    require(np.isfinite(x).all() and (x >= 0).all(), "NNLS_NONFINITE_OR_NEGATIVE")
    fitted = A @ x; gradient = A.T @ (fitted - b)
    bound = 64 * np.finfo(np.float64).eps * max(A.shape) * (np.abs(A).T @ (np.abs(fitted) + np.abs(b)))
    violation = np.maximum(-gradient, 0.)
    violation[x > 0] = np.abs(gradient[x > 0])
    ratio = np.divide(violation, bound, out=np.zeros_like(bound), where=bound > 0)
    require(np.isfinite(gradient).all() and np.all(violation <= bound), "NNLS_KKT_CHECK_FAILED")
    return dict(max_dual_violation=float(np.maximum(-gradient, 0.).max()),
                max_complementarity=float(np.abs(x * gradient).max()), max_bound_ratio=float(ratio.max()))


def fit_nnls(args, d, A, B, mask, np, baseline):
    spectra = B[:, mask].astype(np.float64); count = spectra.shape[1]
    x = np.zeros((391, count), dtype=np.float32); overall = dict(max_dual_violation=0., max_complementarity=0., max_bound_ratio=0.)
    directory = args.output / "nnls_blocks"; directory.mkdir(exist_ok=True)
    with ProcessPoolExecutor(max_workers=4, initializer=baseline.initialize_worker, initargs=(A,)) as pool:
        for start in range(0, count, 250):
            stop = min(start + 250, count); stem = f"block_{start:06d}_{stop:06d}"
            array_file, record_file = directory / (stem + ".npz"), directory / (stem + ".json")
            if array_file.exists() or record_file.exists():
                require(array_file.exists() and record_file.exists(), "PARTIAL_NNLS_BLOCK_PRESERVED")
                record = read(record_file)
                require(record["fingerprint"] == d["fingerprint"] and record["start"] == start and record["stop"] == stop
                        and record["array_sha256"] == sha(array_file), "NNLS_BLOCK_BINDING_CHANGED")
                with np.load(array_file, allow_pickle=False) as z: block = z["X_hat"].copy()
            else:
                block = np.zeros((391, stop - start), dtype=np.float32); checks = dict(overall, **{k: 0. for k in overall})
                for offset, solution in enumerate(pool.map(baseline.solve_pixel, (spectra[:, j] for j in range(start, stop)), chunksize=16)):
                    result = kkt_check(A, spectra[:, start + offset], solution, np)
                    for k, value in result.items(): checks[k] = max(checks[k], value)
                    block[:, offset] = solution
                npz_once(array_file, X_hat=block)
                record = dict(fingerprint=d["fingerprint"], start=start, stop=stop, array_sha256=sha(array_file), KKT=checks,
                              all_pixels_checked_before_float32=True)
                write(record_file, record)
            require(block.shape == (391, stop - start) and np.isfinite(block).all() and (block >= 0).all()
                    and record["all_pixels_checked_before_float32"] is True and record["KKT"]["max_bound_ratio"] <= 1., "INVALID_NNLS_CHECKPOINT")
            x[:, start:stop] = block
            for k in overall: overall[k] = max(overall[k], record["KKT"][k])
            write(args.output / "status.json", dict(status="RUNNING", case=CASE, stage="NNLS", pixels_done=stop, total_pixels=count), replace=True)
            print("NNLS_PIXELS", stop, count, flush=True)
    X_hat = np.zeros((391, *mask.shape), dtype=np.float32); X_hat[:, mask] = x
    B_hat = np.zeros_like(B); B_hat[:, mask] = (A @ x.astype(np.float64)).astype(np.float32)
    return X_hat, B_hat, overall


def accounting(molecules, truth_names, reportable_names, selected):
    reported = [r for r in molecules if r["raw_solver_reported"]]
    raw_tp = sum(r["molecular_truth"] for r in reported); tp = sum(r["molecular_truth"] for r in selected)
    fp = len(selected) - tp; count = len(truth_names)
    return dict(raw_solver_TP=raw_tp, raw_solver_FP=len(reported)-raw_tp, raw_solver_FN=count-raw_tp,
                filtered_TP=tp, filtered_FP=fp, filtered_FN=count-tp, filter_induced_true_loss=raw_tp-tp,
                filter_induced_true_loss_fraction=(raw_tp-tp)/raw_tp if raw_tp else None,
                TP_retention=tp/raw_tp if raw_tp else None, all_truth_recall=tp/count if count else None,
                reportable_truth_recall=sum(r["reportable_truth"] for r in selected)/len(reportable_names) if reportable_names else None,
                FDP=fp/len(selected) if selected else None, precision=tp/len(selected) if selected else None,
                retained_count=len(selected), coverage=len(selected)/len(reported) if reported else None,
                distinct_identities=len({r["lipid_name"] for r in selected}))


def run(args):
    np, baseline, rho, _, implementation = modules(); d = validate(args, implementation); s = d["scientific"]
    if (args.output / "result.json").exists():
        result = read(args.output / "result.json")
        require(result["fingerprint"] == d["fingerprint"], "CACHED_RESULT_BINDING_CHANGED")
        for name, expected in result["artifact_hashes"].items(): require(sha(args.output / name) == expected, "CACHED_RESULT_CHANGED:" + name)
        print("COMPLETE_CACHED_RESULT_VERIFIED", flush=True); return
    with np.load(args.output / "prepared_arrays.npz", allow_pickle=False) as z:
        A, B, mask = z["A_solver"].astype(np.float64), z["B"].copy(), z["mask"].copy()  # Solver never receives target A or truth X.
    require(np.all(B[:, ~mask] == 0), "NONZERO_BACKGROUND_REQUIRES_SOLVE")
    learned_path, done_path = args.output / "learned_arrays.npz", args.output / "nnls_complete.json"
    if learned_path.exists() or done_path.exists():
        require(learned_path.exists() and done_path.exists(), "PARTIAL_NNLS_STAGE_PRESERVED")
        done = read(done_path)
        require(done["fingerprint"] == d["fingerprint"] and done["arrays_sha256"] == sha(learned_path), "CACHED_NNLS_CHANGED")
        with np.load(learned_path, allow_pickle=False) as z: X_hat, B_hat = z["X_hat"].copy(), z["B_hat"].copy()
    else:
        X_hat, B_hat, kkt = fit_nnls(args, d, A, B, mask, np, baseline)
        require(np.isfinite(X_hat).all() and np.isfinite(B_hat).all(), "NONFINITE_LEARNED_ARRAYS")
        npz_once(learned_path, X_hat=X_hat, B_hat=B_hat)
        done = dict(fingerprint=d["fingerprint"], arrays_sha256=sha(learned_path), normal_completion=True,
                    all_outputs_finite=True, KKT=kkt, all_pixels_KKT_checked_before_float32=True,
                    foreground_reconstruction_relative_residual=float(np.linalg.norm(B_hat[:, mask].astype(float)-B[:, mask]) / np.linalg.norm(B[:, mask])))
        write(done_path, done)
    require(done["normal_completion"] is True and done["all_outputs_finite"] is True
            and done["all_pixels_KKT_checked_before_float32"] is True
            and math.isfinite(done["foreground_reconstruction_relative_residual"])
            and all(math.isfinite(v) and v >= 0 for v in done["KKT"].values())
            and done["KKT"]["max_bound_ratio"] <= 1.
            and X_hat.shape == (391, *mask.shape) and B_hat.shape == B.shape
            and np.isfinite(X_hat).all() and np.isfinite(B_hat).all()
            and (X_hat >= 0).all() and (B_hat >= 0).all()
            and np.all(X_hat[:, ~mask] == 0) and np.all(B_hat[:, ~mask] == 0),
            "INVALID_NNLS_COMPLETION_OR_CACHED_ARRAYS")
    metadata = read(args.output / "metadata.json"); means = X_hat[:, mask].mean(axis=1, dtype=np.float64)
    reported = np.flatnonzero(means > .001).tolist(); b = B[:, mask].mean(axis=1, dtype=np.float64); weights = rho.identity_weights(A)
    rho_dir = args.output / "rho"; rho_dir.mkdir(exist_ok=True); scores = {}
    for number, j in enumerate(reported, 1):
        path = rho_dir / f"candidate_{j:04d}.json"
        if path.exists():
            item = read(path)
            require(item["fingerprint"] == d["fingerprint"] and item["candidate_index"] == j
                    and item["learned_arrays_sha256"] == done["arrays_sha256"] and item["b_sha256"] == ah(b), "CACHED_RHO_CHANGED")
        else:
            values = rho.rho_zero_from_weighted_case(A, b, j, weights=weights)
            require(all(math.isfinite(float(v)) for v in values.values()) and values["rho_zero"] >= 0, "NONFINITE_RHO")
            item = dict(fingerprint=d["fingerprint"], candidate_index=j, learned_arrays_sha256=done["arrays_sha256"],
                        b_sha256=ah(b), result=values)
            write(path, item)
        require(all(math.isfinite(float(v)) for v in item["result"].values())
                and item["result"]["rho_zero"] >= 0, "NONFINITE_CACHED_RHO")
        scores[j] = item["result"]
        if number % 25 == 0 or number == len(reported):
            write(args.output / "status.json", dict(status="RUNNING", case=CASE, stage="EXISTING_RHO", completed=number, total=len(reported)), replace=True)
            print("RHO_CANDIDATES", number, len(reported), flush=True)
    truth, reportable, perturbed = set(s["truth_names"]), set(s["reportable_truth_names"]), set(s["actual_perturbed_truth_names"])
    candidates = [dict(candidate_index=j, candidate_id=metadata["candidate_id"][j], lipid_name=metadata["lipid_name"][j],
                       lipid_class=metadata["lipid_class"][j], X_hat=float(means[j]), candidate_truth=j in s["truth_indices"],
                       molecular_truth=metadata["lipid_name"][j] in truth, reportable_truth=metadata["lipid_name"][j] in reportable,
                       actually_perturbed_truth=metadata["lipid_name"][j] in perturbed, raw_solver_reported=j in scores,
                       rho_zero=scores[j]["rho_zero"] if j in scores else None, rho_details=scores.get(j), K=125, replicate="R71", split="CAL") for j in range(391)]
    molecules = []
    for name in dict.fromkeys(metadata["lipid_name"]):
        rows = [r for r in candidates if r["lipid_name"] == name]; active = [r for r in rows if r["raw_solver_reported"]]
        molecules.append(dict(lipid_name=name, molecular_truth=name in truth, reportable_truth=name in reportable,
                              actually_perturbed_truth=name in perturbed, raw_solver_reported=bool(active),
                              X_hat=sum(r["X_hat"] for r in active), X_hat_all_candidates=sum(r["X_hat"] for r in rows),
                              rho_zero=max(r["rho_zero"] for r in active) if active else None,
                              candidate_indices=[r["candidate_index"] for r in rows], reported_candidate_indices=[r["candidate_index"] for r in active]))
    active = [r for r in molecules if r["raw_solver_reported"]]
    curve = [dict(threshold=t, **accounting(molecules, truth, reportable, [r for r in active if r["rho_zero"] >= t]))
             for t in sorted({r["rho_zero"] for r in active}, reverse=True)]
    eligible = [p for p in curve if p["retained_count"] and p["FDP"] <= .01]
    chosen = max(eligible, key=lambda p: (p["filtered_TP"], -p["filtered_FP"], p["threshold"])) if eligible else None
    selected = [r for r in active if chosen is not None and r["rho_zero"] >= chosen["threshold"]]
    baseline_metrics = accounting(molecules, truth, reportable, [r for r in active if r["rho_zero"] >= .001])
    metrics = accounting(molecules, truth, reportable, selected)
    go = bool(metrics["retained_count"] and metrics["FDP"] <= .01 and metrics["TP_retention"] is not None and metrics["TP_retention"] >= .4)
    for r in molecules:
        r["retained_by_rho_1e3"] = bool(r["raw_solver_reported"] and r["rho_zero"] >= .001)
        r["retained_by_first_CAL_FDP1"] = bool(r in selected)
    write(args.output / "candidate_records.json", candidates); write(args.output / "molecular_records.json", molecules)
    write(args.output / "threshold_curve.json", curve)
    write(args.output / "summary.json", dict(case=CASE, raw=accounting(molecules, truth, reportable, active),
                                            fixed_rho_1e3=baseline_metrics, selected_first_CAL=metrics, chosen_threshold=chosen,
                                            first_case_developmental_GO=go, strong_success=bool(go and metrics["TP_retention"] >= .6),
                                            independent_FDR_validated=False, independent_EVAL=False, no_additional_cases_launched=True))
    validate(args, implementation)
    files = [p for p in args.output.iterdir() if p.is_file() and p.name not in ("status.json", "result.json")]
    files += list(rho_dir.glob("candidate_*.json")) + list((args.output / "nnls_blocks").glob("block_*"))
    write(args.output / "result.json", dict(status="COMPLETE_AWAITING_REVIEW", case=CASE, fingerprint=d["fingerprint"],
                                           normal_completion=True, all_outputs_finite=True, learned_training_performed=False,
                                           artifact_hashes={p.relative_to(args.output).as_posix(): sha(p) for p in files if p.is_file()}))
    write(args.output / "status.json", dict(status="COMPLETE_AWAITING_REVIEW", case=CASE), replace=True)
    print("COMPLETE_AWAITING_REVIEW", CASE, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "run"))
    parser.add_argument("--component-assets", type=Path, required=True)
    parser.add_argument("--cached-case", type=Path, default=Path("/root/physical_identity_mainline_v1/results/cases") / SOURCE_CASE)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, choices=(4,), default=4)
    args = parser.parse_args()
    args.component_assets, args.cached_case, args.output = (p.resolve() for p in (args.component_assets, args.cached_case, args.output))
    require(not args.output.is_relative_to(args.cached_case) and not args.output.is_relative_to(args.component_assets), "OUTPUT_MUST_BE_SEPARATE")
    args.output.mkdir(parents=True, exist_ok=True)
    try:
        {"prepare": prepare, "run": run}[args.action](args)
    except BaseException as exc:
        failure = dict(status="FAILED_REQUIRES_REVIEW", case=CASE, action=args.action, error=repr(exc), traceback=traceback.format_exc())
        path = args.output / "failure.json"
        if path.exists(): path = args.output / f"failure_{time.time_ns()}.json"
        write(path, failure)
        write(args.output / "status.json", dict(status="FAILED_REQUIRES_REVIEW", case=CASE, action=args.action), replace=True)
        raise


if __name__ == "__main__":
    main()
