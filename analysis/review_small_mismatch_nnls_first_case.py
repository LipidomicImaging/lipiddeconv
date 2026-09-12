"""Independent cached review of the single 5% NNLS/rho development case.

No optimizer, rho implementation, runner or generator is imported or executed.
All source paths must be available, optionally through an explicit path map.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path

import numpy as np

CASE = "NNLS_FULL_LIBRARY_5PCT__CAL_R71_K125"
SIGNAL_TARGET = 0.6036783456802368
A_SHA = "b9e05e185ffa692966b86022f5487fcdabff92f330a0a389a6bb4889a175a447"


def require(ok, reason):
    if not ok:
        raise RuntimeError(reason)


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024**2), b""):
            h.update(block)
    return h.hexdigest()


def ah(array):
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def same(actual, expected, location="record"):
    """Compare independent counts exactly; allow only relative floating rounding."""
    if isinstance(expected, dict):
        require(isinstance(actual, dict) and set(actual) == set(expected), "KEYS_CHANGED:" + location)
        for key, value in expected.items():
            same(actual[key], value, location + "." + key)
    elif isinstance(expected, list):
        require(isinstance(actual, list) and len(actual) == len(expected), "LENGTH_CHANGED:" + location)
        for i, value in enumerate(expected):
            same(actual[i], value, location + f"[{i}]")
    elif isinstance(expected, float):
        require(isinstance(actual, (int, float)) and math.isfinite(float(actual))
                and math.isclose(float(actual), expected, rel_tol=1e-12, abs_tol=0), "VALUE_CHANGED:" + location)
    elif isinstance(expected, bool):
        require(actual is expected, "BOOLEAN_CHANGED:" + location)
    else:
        require(actual == expected, "VALUE_CHANGED:" + location)


def inside(root, relative):
    path = (root / relative).resolve()
    require(path.is_relative_to(root) and path.is_file(), "INVALID_ARTIFACT_PATH:" + relative)
    return path


def measures(all_truth, reportable_truth, raw_names, retained_names):
    raw_tp = len(raw_names & all_truth)
    tp, fp = len(retained_names & all_truth), len(retained_names - all_truth)
    total = len(all_truth)
    return dict(raw_solver_TP=raw_tp, raw_solver_FP=len(raw_names)-raw_tp, raw_solver_FN=total-raw_tp,
                filtered_TP=tp, filtered_FP=fp, filtered_FN=total-tp, filter_induced_true_loss=raw_tp-tp,
                filter_induced_true_loss_fraction=(raw_tp-tp)/raw_tp if raw_tp else None,
                TP_retention=tp/raw_tp if raw_tp else None, all_truth_recall=tp/total if total else None,
                reportable_truth_recall=len(retained_names & reportable_truth)/len(reportable_truth) if reportable_truth else None,
                FDP=fp/len(retained_names) if retained_names else None,
                precision=tp/len(retained_names) if retained_names else None,
                retained_count=len(retained_names), coverage=len(retained_names)/len(raw_names) if raw_names else None,
                distinct_identities=len(retained_names))


def _check_reconstruction(actual, expected, label):
    require(actual.shape == expected.shape and np.isfinite(actual).all() and np.isfinite(expected).all(),
            "INVALID_RECONSTRUCTION:" + label)
    # Forward products may be summed by different BLAS kernels. This only checks
    # the saved float32 reconstruction, not NNLS optimality or identity confidence.
    tolerance = 8 * np.finfo(np.float32).eps * np.maximum(abs(actual), abs(expected))
    require(np.all(abs(actual.astype(np.float64)-expected.astype(np.float64)) <= tolerance),
            "FORWARD_RECONSTRUCTION_CHANGED:" + label)
    return float(np.max(abs(actual.astype(np.float64)-expected.astype(np.float64))))


def review(output, path_map=None):
    output = Path(output).resolve()
    path_map = path_map or {}
    d, seal, result = (read(output / n) for n in ("design.json", "design_seal.json", "result.json"))
    s = d["scientific"]
    fingerprint = hashlib.sha256(canonical(s)).hexdigest()
    require(fingerprint == d["fingerprint"] == seal["fingerprint"] == result["fingerprint"], "DESIGN_FINGERPRINT_CHANGED")
    require(sha(output / "design.json") == seal["design_sha256"]
            and seal["status"] == "FROZEN_BEFORE_NNLS_AND_RHO", "DESIGN_SEAL_CHANGED")
    require(result["status"] == "COMPLETE_AWAITING_REVIEW" and result["case"] == CASE
            and result["normal_completion"] is True and result["all_outputs_finite"] is True
            and result["learned_training_performed"] is False, "CASE_NOT_NORMALLY_COMPLETE")
    contract = s["contract"]
    require(contract["case"] == CASE and contract["seed"] == 7301 and contract["relative_sd"] == .05
            and contract["report_gate"] == .001 and contract["rho_fixed_threshold"] == .001
            and contract["workers"] == 4 and contract["blas_threads"] == 1
            and contract["no_GPU"] is True and contract["no_other_cases"] is True
            and contract["no_new_score"] is True, "FROZEN_FIRST_CASE_CONTRACT_CHANGED")
    require(s["implementation"]["epsilon_q"] == 1e-12, "RHO_EPSILON_CHANGED")
    artifact_hashes = result["artifact_hashes"]
    required = {"prepared_arrays.npz", "metadata.json", "perturbation_audit.json", "design.json", "design_seal.json",
                "learned_arrays.npz", "nnls_complete.json", "candidate_records.json", "molecular_records.json",
                "threshold_curve.json", "summary.json"}
    require(required <= set(artifact_hashes), "MISSING_REQUIRED_RESULT_ARTIFACTS")
    for relative, expected in artifact_hashes.items():
        require(sha(inside(output, relative)) == expected, "RESULT_ARTIFACT_CHANGED:" + relative)
    for relative, expected in s["prepared_file_hashes"].items():
        require(sha(inside(output, relative)) == expected, "PREPARED_ARTIFACT_CHANGED:" + relative)
    verified_sources = {}
    for source, expected in {**s["source_hashes"], **s["implementation"]["files"]}.items():
        local = Path(path_map.get(source, source)).resolve()
        require(local.is_file() and sha(local) == expected, "SOURCE_NOT_AVAILABLE_OR_CHANGED:" + source)
        verified_sources[source] = dict(verified_path=str(local), sha256=expected)
    with np.load(output / "prepared_arrays.npz", allow_pickle=False) as z:
        require(set(z.files) == {"A_solver", "A_target", "X_true", "B", "mask"}, "PREPARED_ARRAY_MEMBERSHIP_CHANGED")
        A, target, truth_X, B, mask = (z[k].copy() for k in ("A_solver", "A_target", "X_true", "B", "mask"))
    require(A.shape == target.shape == (1084, 391) and mask.dtype == np.bool_ and mask.any()
            and truth_X.shape == (391, *mask.shape) and B.shape == (1084, *mask.shape), "PREPARED_ARRAY_SHAPE_CHANGED")
    for value in (A, target, truth_X, B):
        require(np.isfinite(value).all() and np.all(value >= 0), "INVALID_PREPARED_VALUES")
    require(np.all(B[:, ~mask] == 0) and np.all(truth_X[:, ~mask] == 0), "NONZERO_PREPARED_BACKGROUND")
    require(ah(A) == A_SHA == s["A_solver_sha256"] and ah(target) == s["A_target_sha256"]
            and ah(truth_X) == s["target_X_sha256"] and ah(B) == s["B_sha256"], "PREPARED_ARRAY_DIGEST_CHANGED")
    original_containers = [name for name in s["source_hashes"] if Path(name).name == "observation_truth.npz"]
    require(len(original_containers) == 1, "AMBIGUOUS_ORIGINAL_SPATIAL_SOURCE")
    with np.load(Path(verified_sources[original_containers[0]]["verified_path"]), allow_pickle=False) as z:
        original_X, original_mask = z["X_true"].copy(), z["mask"].copy()
    scalar = s["global_dataset_scalar"]
    require(math.isfinite(scalar) and scalar > 0 and ah(original_X) == s["source_X_sha256"]
            and np.array_equal(mask, original_mask)
            and np.array_equal(truth_X, original_X.astype(np.float64) * scalar), "GLOBAL_SCALAR_OR_SPATIAL_SHAPE_CHANGED")
    metadata = read(output / "metadata.json")
    names = metadata["lipid_name"]
    require(len(names) == 391 and len(metadata["candidate_id"]) == 391
            and len(set(metadata["candidate_id"])) == 391, "INVALID_CANDIDATE_CATALOG")
    true_indices = np.flatnonzero(np.any(truth_X[:, mask] > 0, axis=1)).tolist()
    reportable_indices = np.flatnonzero(truth_X[:, mask].mean(axis=1, dtype=np.float64) > .001).tolist()
    all_truth = {names[j] for j in true_indices}; reportable_truth = {names[j] for j in reportable_indices}
    changes = np.linalg.norm(target.astype(np.float64)-A.astype(np.float64), axis=0) / np.linalg.norm(A.astype(np.float64), axis=0)
    perturbed_indices = [j for j in true_indices if changes[j] > 1e-10]
    perturbed_truth = {names[j] for j in perturbed_indices}
    require(len(true_indices) == len(all_truth) == 125, "NOT_THE_FROZEN_125_TRUTH_CASE")
    for key, value in dict(truth_indices=true_indices, truth_names=sorted(all_truth),
                           reportable_truth_indices=reportable_indices, reportable_truth_names=sorted(reportable_truth),
                           truth_count=len(all_truth), reportable_truth_count=len(reportable_truth),
                           actual_perturbed_truth_indices=perturbed_indices,
                           actual_perturbed_truth_names=sorted(perturbed_truth), actual_perturbed_truth_count=len(perturbed_indices)).items():
        same(s[key], value, "scientific." + key)
    achieved = float(np.median(np.linalg.norm(B[:, mask].astype(np.float64), axis=0)))
    require(abs(achieved-SIGNAL_TARGET) <= 8*np.finfo(np.float32).eps*SIGNAL_TARGET, "SIGNAL_TARGET_CHANGED")
    same(s["foreground_median_B_norm"], achieved, "foreground_norm")
    forward_error = _check_reconstruction(B[:, mask], target.astype(np.float64) @ truth_X[:, mask], "B_target")
    done = read(output / "nnls_complete.json")
    require(done["fingerprint"] == fingerprint and done["arrays_sha256"] == sha(output / "learned_arrays.npz")
            and done["normal_completion"] is True and done["all_outputs_finite"] is True
            and done["all_pixels_KKT_checked_before_float32"] is True, "NNLS_COMPLETION_CHANGED")
    with np.load(output / "learned_arrays.npz", allow_pickle=False) as z:
        require(set(z.files) == {"X_hat", "B_hat"}, "LEARNED_ARRAY_MEMBERSHIP_CHANGED")
        X_hat, B_hat = z["X_hat"].copy(), z["B_hat"].copy()
    require(X_hat.dtype == np.float32 and X_hat.shape == truth_X.shape and B_hat.shape == B.shape
            and np.isfinite(X_hat).all() and np.isfinite(B_hat).all()
            and np.all(X_hat >= 0) and np.all(B_hat >= 0)
            and np.all(X_hat[:, ~mask] == 0) and np.all(B_hat[:, ~mask] == 0), "INVALID_LEARNED_ARRAYS")
    learned_error = _check_reconstruction(B_hat[:, mask], A.astype(np.float64) @ X_hat[:, mask].astype(np.float64), "B_hat")
    residual = float(np.linalg.norm(B_hat[:, mask].astype(float)-B[:, mask]) / np.linalg.norm(B[:, mask]))
    same(done["foreground_reconstruction_relative_residual"], residual, "reconstruction_residual")
    foreground_X = X_hat[:, mask]
    block_files, kkt = set(), dict(max_dual_violation=0., max_complementarity=0., max_bound_ratio=0.)
    for start in range(0, foreground_X.shape[1], 250):
        stop = min(start+250, foreground_X.shape[1]); stem = f"nnls_blocks/block_{start:06d}_{stop:06d}"
        block_record = read(output / (stem + ".json"))
        block_files.update((stem + ".npz", stem + ".json"))
        require(block_record["fingerprint"] == fingerprint and block_record["start"] == start and block_record["stop"] == stop
                and block_record["array_sha256"] == sha(output / (stem + ".npz"))
                and block_record["all_pixels_checked_before_float32"] is True, "NNLS_BLOCK_BINDING_CHANGED")
        with np.load(output / (stem + ".npz"), allow_pickle=False) as z:
            require(set(z.files) == {"X_hat"} and np.array_equal(z["X_hat"], foreground_X[:, start:stop]), "NNLS_BLOCK_ARRAY_CHANGED")
        require(set(block_record["KKT"]) == set(kkt) and all(math.isfinite(v) and v >= 0 for v in block_record["KKT"].values())
                and block_record["KKT"]["max_bound_ratio"] <= 1, "INVALID_REPORTED_KKT_CHECK")
        for key in kkt:
            kkt[key] = max(kkt[key], block_record["KKT"][key])
    require({p.relative_to(output).as_posix() for p in (output / "nnls_blocks").glob("block_*")} == block_files
            and block_files <= set(artifact_hashes), "NNLS_BLOCK_MEMBERSHIP_CHANGED")
    same(done["KKT"], kkt, "aggregated_KKT")
    means = X_hat[:, mask].mean(axis=1, dtype=np.float64)
    reported_indices = set(np.flatnonzero(means > .001).tolist())
    b = B[:, mask].mean(axis=1, dtype=np.float64)
    signal = max(float(b @ b)+1e-12, 1e-12)
    scores, cancellation = {}, []
    expected_rho_files = {f"rho/candidate_{j:04d}.json" for j in reported_indices}
    require({p.relative_to(output).as_posix() for p in (output / "rho").glob("candidate_*.json")} == expected_rho_files
            and expected_rho_files <= set(artifact_hashes), "RHO_MEMBERSHIP_CHANGED")
    for j in sorted(reported_indices):
        item = read(output / f"rho/candidate_{j:04d}.json")
        require(item["fingerprint"] == fingerprint and item["candidate_index"] == j
                and item["learned_arrays_sha256"] == done["arrays_sha256"] and item["b_sha256"] == ah(b), "RHO_BINDING_CHANGED")
        value = item["result"]
        require(all(isinstance(v, (int, float)) and math.isfinite(v) for v in value.values())
                and value["q_star"] >= 0 and value["q_deleted"] >= 0 and value["x_star_candidate"] >= 0, "INVALID_RHO_DETAILS")
        delta = value["q_deleted"]-value["q_star"]
        expected = dict(rho_zero=max(0., delta/signal), necessity_signal=delta/signal,
                        necessity_fit=delta/max(value["q_star"], 1e-12), q_star=value["q_star"],
                        q_deleted=value["q_deleted"], signal_norm2_plus_epsilon=signal, x_star_candidate=value["x_star_candidate"])
        same(value, expected, f"rho[{j}]")
        scores[j] = expected
        subtraction_scale = (math.ulp(value["q_star"])+math.ulp(value["q_deleted"]))/signal
        cancellation.append(dict(candidate_index=j, rho_zero=expected["rho_zero"], signed_delta_q=delta,
                                 normalized_stored_q_arithmetic_scale=subtraction_scale,
                                 positive_rho_to_arithmetic_scale=expected["rho_zero"]/subtraction_scale if subtraction_scale else None))
    candidates = read(output / "candidate_records.json")
    require(len(candidates) == 391 and [r["candidate_index"] for r in candidates] == list(range(391)), "CANDIDATE_RECORD_MEMBERSHIP_CHANGED")
    for j, row in enumerate(candidates):
        expected = dict(candidate_index=j, candidate_id=metadata["candidate_id"][j], lipid_name=names[j], lipid_class=metadata["lipid_class"][j],
                        X_hat=float(means[j]), candidate_truth=j in true_indices, molecular_truth=names[j] in all_truth,
                        reportable_truth=names[j] in reportable_truth, actually_perturbed_truth=names[j] in perturbed_truth,
                        raw_solver_reported=j in reported_indices, rho_zero=scores[j]["rho_zero"] if j in scores else None,
                        rho_details=scores.get(j), K=125, replicate="R71", split="CAL")
        same(row, expected, f"candidate[{j}]")
    by_name = defaultdict(list)
    for j, name in enumerate(names):
        by_name[name].append(j)
    molecular_scores = {name: max(scores[j]["rho_zero"] for j in indices if j in scores)
                        for name, indices in by_name.items() if any(j in scores for j in indices)}
    raw_names = set(molecular_scores)
    curve = []
    for threshold in sorted(set(molecular_scores.values()), reverse=True):
        retained = {name for name, score in molecular_scores.items() if score >= threshold}
        curve.append(dict(threshold=threshold, **measures(all_truth, reportable_truth, raw_names, retained)))
    same(read(output / "threshold_curve.json"), curve, "complete_tie_preserving_curve")
    eligible = [row for row in curve if row["retained_count"] > 0 and row["FDP"] <= .01]
    chosen = sorted(eligible, key=lambda r: (-r["filtered_TP"], r["filtered_FP"], -r["threshold"]))[0] if eligible else None
    selected = {name for name, score in molecular_scores.items() if chosen and score >= chosen["threshold"]}
    fixed = {name for name, score in molecular_scores.items() if score >= .001}
    metrics = measures(all_truth, reportable_truth, raw_names, selected)
    go = bool(selected and metrics["FDP"] <= .01 and metrics["TP_retention"] is not None and metrics["TP_retention"] >= .4)
    summary = dict(case=CASE, raw=measures(all_truth, reportable_truth, raw_names, raw_names),
                   fixed_rho_1e3=measures(all_truth, reportable_truth, raw_names, fixed), selected_first_CAL=metrics,
                   chosen_threshold=chosen, first_case_developmental_GO=go,
                   strong_success=bool(go and metrics["TP_retention"] >= .6), independent_FDR_validated=False,
                   independent_EVAL=False, no_additional_cases_launched=True)
    same(read(output / "summary.json"), summary, "summary")
    molecules = read(output / "molecular_records.json")
    require(len(molecules) == len(by_name) and {r["lipid_name"] for r in molecules} == set(by_name), "MOLECULAR_MEMBERSHIP_CHANGED")
    for row in molecules:
        name = row["lipid_name"]; indices = by_name[name]; active = [j for j in indices if j in scores]
        expected = dict(lipid_name=name, molecular_truth=name in all_truth, reportable_truth=name in reportable_truth,
                        actually_perturbed_truth=name in perturbed_truth, raw_solver_reported=bool(active),
                        X_hat=sum(float(means[j]) for j in active), X_hat_all_candidates=sum(float(means[j]) for j in indices),
                        rho_zero=molecular_scores.get(name), candidate_indices=indices, reported_candidate_indices=active,
                        retained_by_rho_1e3=name in fixed, retained_by_first_CAL_FDP1=name in selected)
        same(row, expected, "molecular." + name)
    selected_candidates = {j for name in selected for j in by_name[name] if j in scores}
    selected_diagnostics = [r for r in cancellation if r["candidate_index"] in selected_candidates and r["rho_zero"] > 0]
    receipt = dict(status="PASS", case=CASE, reviewer_sha256=sha(Path(__file__)), design_fingerprint=fingerprint,
                   result_sha256=sha(output / "result.json"), verified_artifact_hashes=artifact_hashes,
                   verified_sources=verified_sources, all_truth_count=len(all_truth), reportable_truth_count=len(reportable_truth),
                   actually_perturbed_truth_count=len(perturbed_truth), independent_summary=summary,
                   actually_perturbed_raw_TP=len(perturbed_truth & raw_names), actually_perturbed_filtered_TP=len(perturbed_truth & selected),
                   actually_perturbed_TP_retention=len(perturbed_truth & selected)/len(perturbed_truth & raw_names) if perturbed_truth & raw_names else None,
                   nominal_and_target_forward_max_absolute_errors=dict(target=forward_error, learned=learned_error),
                   stored_NNLS_block_membership_verified=True, stored_KKT_aggregation_verified=True,
                   pre_float32_KKT_independently_recomputed=False, rho_formula_and_membership_verified=True,
                   rho_optima_independently_resolved=False, truth_leakage_review="Scoring inputs reviewed in source before execution; reviewer uses truth only for cached accounting",
                   numerical_cancellation=dict(rule="Diagnostic only; thresholds and GO unchanged",
                       scale="(ulp(stored q_star)+ulp(stored q_deleted))/signal; arithmetic descriptor, not optimizer error bound",
                       negative_signed_delta_count=sum(r["signed_delta_q"] < 0 for r in cancellation),
                       minimum_selected_positive_rho_to_arithmetic_scale=min((r["positive_rho_to_arithmetic_scale"] for r in selected_diagnostics), default=None),
                       candidate_diagnostics=cancellation),
                   no_optimization_or_rho_rerun=True, independent_real_data_FDR_validated=False)
    state = "GO" if go else "NO-GO"
    chosen_text = "none" if chosen is None else format(chosen["threshold"], ".17g")
    analysis = (f"# One-case 5% mismatch NNLS/rho review\n\n"
                f"Independent cached process/accounting review: PASS. Developmental first-case screen: **{state}**.\n\n"
                f"Truth identities: {len(all_truth)}; reportable truths: {len(reportable_truth)}; actually perturbed truths: {len(perturbed_truth)}. "
                f"Raw solver TP/FP/FN: {metrics['raw_solver_TP']}/{metrics['raw_solver_FP']}/{metrics['raw_solver_FN']}. "
                f"Selected TP/FP/FN: {metrics['filtered_TP']}/{metrics['filtered_FP']}/{metrics['filtered_FN']}; "
                f"filter-induced true losses: {metrics['filter_induced_true_loss']}.\n\n"
                f"Selected threshold: {chosen_text}; empirical FDP: {metrics['FDP']}; TP retention: {metrics['TP_retention']}; "
                f"all-truth recall: {metrics['all_truth_recall']}. Complete tied-score curve and fixed rho=1e-3 result independently agree.\n\n"
                "This is one previously exposed spatial development case under a user-specified 5% model. "
                "The selected threshold is not independent calibration and does not establish real-data FDR control. "
                "No further case is authorized by a GO result.\n\n"
                "Stored rho formulas were recomputed from q_star/q_deleted without solving NNLS again. "
                "The receipt describes floating subtraction scales; these do not bound optimizer error or alter selection. "
                "Pre-cast KKT records and block membership were checked; unavailable pre-cast float64 solutions were not reconstructed.\n")
    return receipt, analysis


def write_once(path, data):
    if path.exists():
        require(path.read_bytes() == data, "EXISTING_REVIEW_PRESERVED:" + path.name)
    else:
        with path.open("xb") as f:
            f.write(data)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--path-map", type=Path, help="JSON mapping original source absolute paths to verified local copies")
    args = parser.parse_args()
    receipt, analysis = review(args.output, read(args.path_map) if args.path_map else None)
    write_once(args.output / "independent_review.json", (json.dumps(receipt, indent=2, allow_nan=False)+"\n").encode())
    write_once(args.output / "independent_analysis.md", analysis.encode())
    print("INDEPENDENT_CACHED_REVIEW_PASS", CASE, flush=True)


if __name__ == "__main__":
    main()
