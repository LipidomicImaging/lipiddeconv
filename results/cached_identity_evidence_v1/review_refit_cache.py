"""Cache-only reproducer for fixed screening and completed selected-library fits.

No production-module import, fitting, rho calculation or threshold search.
Default --stage refit writes cases/<case>/independent_refit_review.json;
--stage screen writes independent_screen_review.json after screen completion.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re

for _key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_key] = "1"

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = Path(__file__).resolve().parent
ARMS = {"TRIM": ["trimmed_gain"], "DIRECT": ["direct_gain", "direct_per_signal"],
        "JOINT": ["trimmed_gain", "direct_gain", "direct_per_signal", "stability"]}
KKT_KEYS = {"max_dual_violation", "max_complementarity", "max_bound_ratio"}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def close(actual, expected, label):
    if isinstance(expected, dict):
        require(isinstance(actual, dict) and set(expected) <= set(actual), label)
        for key, value in expected.items():
            close(actual[key], value, label + "." + key)
    elif isinstance(expected, list):
        require(isinstance(actual, list) and len(actual) == len(expected), label)
        for i, (left, right) in enumerate(zip(actual, expected)):
            close(left, right, label + "." + str(i))
    elif isinstance(expected, float):
        require(isinstance(actual, (int, float)) and math.isfinite(actual)
                and math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-15), label)
    else:
        require(actual == expected, label)


def hashes(base, values):
    for path, digest in values.items():
        resolved = (base / path).resolve()
        require(resolved.is_relative_to(base.resolve()) and sha(resolved) == digest, "HASH:" + str(path))


def counts(names, truth, raw):
    selected = set(names)
    tp, fp = len(selected & truth), len(selected - truth)
    return dict(TP=tp, FP=fp, FN=125-tp, retained=len(selected), FDP=fp/len(selected) if selected else None,
                all_truth_recall=tp/125, TP_retention=tp/len(raw & truth) if raw & truth else None,
                all_truth_count=125, raw_solver_misses=len(truth-raw))


def scalar_probability(row, old_feature, new_feature, reference, model=None):
    if not row["raw_solver_reported"]:
        return None
    require(row["X_hat"] >= 0 and row["rho_zero"] is not None and row["rho_zero"] >= 0, "SCORE_DOMAIN")
    a, r = [(value-mean)/scale for value, mean, scale in zip(
        (math.log10(max(row["X_hat"], 1e-12)), math.log10(max(row["rho_zero"], 1e-24))),
        reference["base"]["mean"], reference["base"]["scale"])]
    s, c = [(value-mean)/scale for value, mean, scale in zip(
        (old_feature["predictive_gain"], old_feature["positive_block_fraction"]),
        reference["auxiliary"]["mean"], reference["auxiliary"]["scale"])]
    values = [a, a*a, r, s, c]
    fitted = reference if model is None else model
    if model is not None:
        require(len(model["added_features"]) == len(model["mean"]) == len(model["scale"])
                and all(math.isfinite(v) and v > 0 for v in model["scale"]), "NEW_NORMALIZATION")
        values += [(new_feature[key]-mean)/scale for key, mean, scale in
                   zip(model["added_features"], model["mean"], model["scale"])]
    require(len(values) == len(fitted["coef"]) and all(v >= 0 for v in fitted["coef"][2:]), "MODEL_SHAPE_BOUNDS")
    logit = math.fsum([fitted["intercept"]] + [v*w for v, w in zip(values, fitted["coef"])])
    require(math.isfinite(logit), "NONFINITE_LOGIT")
    return 1/(1+math.exp(-logit)) if logit >= 0 else math.exp(logit)/(1+math.exp(logit))


def screen_review(output, case):
    design = read(output / "design.json")
    require(read(output/"design_seal.json")["design_sha256"] == sha(output/"design.json"), "DESIGN_SEAL")
    require(design["arms"] == ARMS and design["refit_policy"] == "winner fixed DEV5% set", "DESIGN_POLICY")
    hashes(ROOT, design["code"])
    hashes(ROOT, design["sources"][case]["hashes"])
    require(sha(output/"reference_model.json") == design["reference_sha256"]
            and sha(output/"reference_thresholds.json") == design["reference_thresholds_sha256"], "REFERENCE_BINDING")
    cfg = design["sources"][case]["config"]
    source = ROOT / cfg["source"]
    require(sha(source) == design["sources"][case]["source_sha256"], "FULL_SOURCE_HASH")
    names, order = design["candidate_names"], design["molecular_names"]
    require(len(names) == 391 and len(order) == 377 and order == list(dict.fromkeys(names)), "UNIVERSE")
    aliases = {name: [i for i, n in enumerate(names) if n == name] for name in order}
    rows = read(ROOT/cfg["records"])
    old_features, new_features = read(ROOT/cfg["features"]), read(output/"cases"/case/"features.json")
    require([r["lipid_name"] for r in rows] == order
            and [r["molecular_name"] for r in old_features] == order
            and [r["molecular_name"] for r in new_features] == order, "INPUT_RECORD_ORDER")
    truth = {r["lipid_name"] for r in rows if r["molecular_truth"]}
    raw = {r["lipid_name"] for r in rows if r["raw_solver_reported"]}
    require(len(truth) == 125, "ALL_TRUTH_DENOMINATOR")
    for row in rows:
        require(row["candidate_indices"] == aliases[row["lipid_name"]]
                and set(row["reported_candidate_indices"]) <= set(row["candidate_indices"])
                and row["raw_solver_reported"] == bool(row["reported_candidate_indices"]), "RAW_ALIAS_MEMBERSHIP")
    directory = output/"cases"/case
    feature_done = read(directory/"feature_completion.json")
    close(feature_done, dict(status="COMPLETE", case=case, design_sha256=sha(output/"design.json"),
          source_hashes=design["sources"][case]["hashes"]), "FEATURE_COMPLETION")
    require(set(feature_done["artifacts"]) == {"features.json", "feature_arrays.npz"}, "FEATURE_MEMBERSHIP")
    hashes(directory, feature_done["artifacts"])
    feature_review = read(directory/"independent_feature_review.json")
    close(feature_review, dict(status="PASS", case=case, design_sha256=sha(output/"design.json"),
          feature_completion_sha256=sha(directory/"feature_completion.json"), source_hashes=feature_done["source_hashes"],
          feature_artifact_hashes=feature_done["artifacts"]), "FEATURE_REVIEW_BINDING")
    models = output/"models"
    seal, choice = read(models/"model_seal.json"), read(models/"selection.json")
    require(seal["status"] == "FROZEN_BEFORE_CHECK_SCORING", "MODEL_SEAL_STATUS")
    expected_artifacts = {f"{arm}/{name}.json" for arm in ARMS
                          for name in ("model", "thresholds", "optimizer", "train_scores")}
    require(set(seal["artifacts"]) == expected_artifacts | {"selection.json", "fit_reservation.json"}, "MODEL_ARTIFACT_MEMBERSHIP")
    hashes(models, seal["artifacts"])
    cuts = {arm: read(models/arm/"thresholds.json") for arm in ARMS}
    require(choice["all_dev_results"] == cuts and choice["design_sha256"] == sha(output/"design.json"), "DEV_WINNER_BINDING")
    winner = max(ARMS, key=lambda arm: (cuts[arm]["train_fdp5"]["TP"], -cuts[arm]["train_fdp5"]["FP"],
                                       -len(ARMS[arm]), -list(ARMS).index(arm)))
    require(seal["winner"] == choice["winner"] == winner, "FIXED_DEV_WINNER")
    screen = read(directory/"screen.json")
    close(screen, dict(status="COMPLETE", case=case, winner=winner, design_sha256=sha(output/"design.json"),
          model_seal_sha256=sha(models/"model_seal.json"),
          feature_completion_sha256=sha(directory/"feature_completion.json")), "SCREEN_BINDING")
    require(re.fullmatch(r"[0-9a-f]{40}", screen["model_commit"]) and screen["rows"] == rows
            and set(screen["arms"]) == {"BASELINE", *ARMS}, "SCREEN_MEMBERSHIP")
    reference = read(output/"reference_model.json")
    require(reference["features"] == ["a", "a_squared", "r", "s", "c"], "REFERENCE_FEATURES")
    arm_reviews = {}
    for arm in ("BASELINE", *ARMS):
        model = None if arm == "BASELINE" else read(models/arm/"model.json")
        if model is not None:
            require(model["arm"] == arm and model["added_features"] == ARMS[arm]
                    and model["reference_sha256"] == design["reference_sha256"], "ARM_BINDING")
        thresholds = read(output/"reference_thresholds.json") if arm == "BASELINE" else cuts[arm]
        probabilities = [scalar_probability(r, old, new, reference, model) for r, old, new in zip(rows, old_features, new_features)]
        recorded = screen["arms"][arm]
        require(len(recorded["scores"]) == 377, "SCORE_LENGTH")
        errors = []
        for value, stored in zip(probabilities, recorded["scores"]):
            if value is None:
                require(stored is None, "UNREPORTED_SCORE")
            else:
                require(stored is not None and math.isfinite(stored), "NONFINITE_SCORE")
                errors.append(abs(value-stored))
        require(max(errors, default=0.) <= 1e-12, "SCALAR_PROBABILITY")
        sets = {endpoint: [name for name, value in zip(order, probabilities)
                          if value is not None and thresholds[endpoint+"_threshold"] is not None
                          and value >= thresholds[endpoint+"_threshold"]] for endpoint in ("pool", "fdp5", "fdp1")}
        require(recorded["selected_names"] == sets, "EXACT_FIXED_THRESHOLD_SETS:" + arm)
        metrics = {endpoint: counts(values, truth, raw) for endpoint, values in sets.items()}
        close(recorded["counts"], metrics, "SCREEN_COUNTS:" + arm)
        arm_reviews[arm] = dict(counts=metrics, selected_names=sets,
            probability_max_absolute_error=max(errors, default=0.),
            minimum_fixed_cut_margin=min((abs(value-thresholds[endpoint+"_threshold"])
                for value in probabilities if value is not None for endpoint in ("pool", "fdp5", "fdp1")
                if thresholds[endpoint+"_threshold"] is not None), default=None))
    selected = arm_reviews[winner]["selected_names"]["fdp5"]
    require(screen["refit_names"] == selected, "REFIT_WINNER_DEV5_SET")
    review = dict(status="PASS", case=case, stage="screen", winner=winner,
        winner_DEV5_cut=cuts[winner]["fdp5_threshold"], design_sha256=sha(output/"design.json"),
        model_seal_sha256=sha(models/"model_seal.json"), screen_sha256=sha(directory/"screen.json"),
        feature_completion_sha256=sha(directory/"feature_completion.json"),
        source_sha256=sha(source), raw_counts=counts(raw, truth, raw), arms=arm_reviews, refit_names=selected,
        raw_solver_missed_truth=[n for n in order if n in truth-raw],
        screening_removed_truth=[n for n in order if n in (raw-set(selected)) & truth])
    return review, design, source, rows, names, order, aliases, truth, raw, selected


def refit_review(output, case, context):
    screening, design, source, rows, names, order, aliases, truth, raw, selected = context
    directory = output/"cases"/case/"refit"
    done, binding = read(directory/"completion.json"), read(directory/"refit_binding.json")
    require(done["binding"] == binding, "REFIT_COMPLETION_BINDING")
    close(binding, dict(design_sha256=screening["design_sha256"], model_seal_sha256=screening["model_seal_sha256"],
          screen_sha256=screening["screen_sha256"], source_sha256=screening["source_sha256"],
          policy="fixed DEV5% winner; all aliases; original gate .001"), "REFIT_SOURCE_AND_SELECTION")
    require(re.fullmatch(r"[0-9a-f]{40}", binding["screen_pushed_commit"]), "PUSHED_SCREEN_COMMIT")
    if done["status"] == "NOT_RUN_EMPTY_POOL":
        require(not selected and not (directory/"fit").exists(), "EMPTY_POOL_STOP")
        close(done["counts"], counts([], truth, raw), "EMPTY_SELECTION_COUNTS")
        return dict(status="PASS", case=case, stage="refit", fit_performed=False,
                    reason="Frozen winner DEV5 set empty; no invented fitted endpoint", screen_review=screening,
                    completion_sha256=sha(directory/"completion.json"))
    require(done["status"] == "COMPLETE" and selected, "REFIT_NOT_COMPLETE")
    require(set(done["artifacts"]) == {"candidate_records.json", "molecular_records.json", "fit/learned_arrays.npz",
                                      "fit/completion.json", "fit/binding.json"}, "REFIT_ARTIFACT_MEMBERSHIP")
    hashes(directory, done["artifacts"])
    with np.load(source, allow_pickle=False) as z:
        A, B, mask = z["A_solver"], z["B"], z["mask"]
    require(A.shape == (1084, 391) and B.shape == (1084, 200, 90) and mask.shape == (200, 90)
            and mask.dtype == bool and int(mask.sum()) == 15837 and np.isfinite(A).all()
            and np.isfinite(B).all() and np.all(A >= 0) and np.all(B[:, ~mask] == 0), "FULL_SOURCE_ARRAYS")
    indices = [i for i, name in enumerate(names) if name in set(selected)]
    fit = directory/"fit"
    fit_binding, receipt = read(fit/"binding.json"), read(fit/"completion.json")
    implementation = {Path(p).name: design["code"][p] for p in (
        "analysis/refit_screened_nnls.py", "analysis/run_nnls_solver_baseline.py", "analysis/run_small_mismatch_nnls_first_case.py")}
    scientific = dict(version=1, retained_indices=indices, source_binding=binding,
        solver="existing scipy NNLS; float64 solve, float32 storage", maxiter=3910, workers=4,
        block_size=250, blas_threads=1, implementation=implementation,
        numpy=binding["runtime"]["numpy"], scipy=binding["runtime"]["scipy"])
    for key, value in (("A", A), ("B", B), ("mask", mask)):
        scientific[key] = dict(shape=list(value.shape), dtype=value.dtype.str,
                              sha256=hashlib.sha256(value.tobytes(order="C")).hexdigest())
    fingerprint = hashlib.sha256(canonical(scientific)).hexdigest()
    require(fit_binding == dict(fingerprint=fingerprint, scientific=scientific), "EXACT_REFIT_SCIENTIFIC_BINDING")
    close(receipt, dict(status="COMPLETE", fingerprint=fingerprint, binding_sha256=sha(fit/"binding.json"),
        arrays_sha256=sha(fit/"learned_arrays.npz"), retained_indices=indices, foreground_pixels=15837,
        candidate_count=391, fitted_column_count=len(indices), all_pixels_checked_before_float32=True,
        all_outputs_finite=True), "FIT_COMPLETION_RECEIPT")
    require(read(fit/"status.json") == dict(fingerprint=fingerprint, status="COMPLETE", pixels_done=15837,
        total_pixels=15837, completion_sha256=sha(fit/"completion.json")), "FIT_FINAL_STATUS")
    with np.load(fit/"learned_arrays.npz", allow_pickle=False) as z:
        require(set(z.files) == {"X_hat", "B_hat"}, "FINAL_ARRAY_MEMBERSHIP")
        X, Bhat = z["X_hat"], z["B_hat"]
    require(X.shape == (391, 200, 90) and Bhat.shape == B.shape and X.dtype == Bhat.dtype == np.float32
            and np.isfinite(X).all() and np.isfinite(Bhat).all() and np.all(X >= 0)
            and np.all(X[:, ~mask] == 0) and np.all(Bhat[:, ~mask] == 0), "FINAL_ARRAYS")
    require(np.all(X[sorted(set(range(391))-set(indices))] == 0), "EXCLUDED_ALIAS_REENTRY")
    blocks = fit/"nnls_blocks"
    expected_files = {f"block_{start:06d}_{min(start+250,15837):06d}.{ext}"
                      for start in range(0, 15837, 250) for ext in ("npz", "json")}
    require(set(receipt["block_hashes"]) == {p.name for p in blocks.iterdir()} == expected_files, "EXACT_64_BLOCK_MEMBERSHIP")
    hashes(blocks, receipt["block_hashes"])
    flat, fitted, observed = X[:, mask], Bhat[:, mask], B[:, mask]
    overall = {key: 0. for key in KKT_KEYS}
    max_reconstruction_error, sse, signal = 0., 0., 0.
    reduced = np.asarray(A[:, indices], dtype=np.float64)
    for start in range(0, 15837, 250):
        stop = min(start+250, 15837)
        stem = f"block_{start:06d}_{stop:06d}"
        record = read(blocks/(stem+".json"))
        close(record, dict(fingerprint=fingerprint, start=start, stop=stop, retained_indices=indices,
              array_sha256=sha(blocks/(stem+".npz")), all_pixels_checked_before_float32=True,
              empty_design_vacuous_kkt=False), "BLOCK_RECEIPT")
        require(set(record["KKT"]) == KKT_KEYS and all(math.isfinite(v) and v >= 0 for v in record["KKT"].values())
                and record["KKT"]["max_bound_ratio"] <= 1., "ORIGINAL_KKT_RECEIPT")
        for key, value in record["KKT"].items():
            overall[key] = max(overall[key], value)
        with np.load(blocks/(stem+".npz"), allow_pickle=False) as z:
            require(set(z.files) == {"X_hat"}, "BLOCK_ARRAY_KEYS")
            block = z["X_hat"]
        require(block.dtype == np.float32 and block.shape == (len(indices), stop-start)
                and np.array_equal(block, flat[indices, start:stop]), "EXACT_BLOCK_TO_FINAL_MEMBERSHIP")
        prediction = (reduced @ block.astype(np.float64)).astype(np.float32)
        max_reconstruction_error = max(max_reconstruction_error,
            float(np.max(np.abs(prediction.astype(np.float64)-fitted[:, start:stop].astype(np.float64)))))
        target = observed[:, start:stop].astype(np.float64)
        difference = target-fitted[:, start:stop].astype(np.float64)
        sse += float(np.sum(difference*difference)); signal += float(np.sum(target*target))
    reconstruction_bound = 5e-7*max(float(np.max(np.abs(fitted))), 1e-12)
    require(max_reconstruction_error <= reconstruction_bound and receipt["KKT"] == overall, "RECONSTRUCTION_OR_KKT_AGGREGATION")
    residual = dict(squared_residual=sse, observation_squared_norm=signal, relative_l2=math.sqrt(sse/signal))
    close(done["residual"], residual, "INDEPENDENT_ORIGINAL_B_RESIDUAL")
    means = flat.mean(axis=1, dtype=np.float64)
    candidates, molecules = read(directory/"candidate_records.json"), read(directory/"molecular_records.json")
    require(len(candidates) == 391 and [r["lipid_name"] for r in molecules] == order, "FINAL_RECORD_ORDER")
    raw_aliases = {i for row in rows for i in row["reported_candidate_indices"]}
    for i, row in enumerate(candidates):
        close(row, dict(candidate_index=i, lipid_name=names[i], molecular_truth=names[i] in truth,
            raw_solver_reported=i in raw_aliases, in_pool=names[i] in selected,
            refit_mean=float(means[i]), refit_reported=bool(means[i] > .001)), "FINAL_CANDIDATE")
    final = []
    for original, row in zip(rows, molecules):
        name = row["lipid_name"]
        active = [i for i in aliases[name] if means[i] > .001]
        close(row, original, "FINAL_SOURCE_RECORD")
        close(row, dict(in_pool=name in selected, refit_reported=bool(active),
            refit_reported_candidate_indices=active, refit_abundance=sum(float(means[i]) for i in active)), "FINAL_ALIAS_GATE")
        if active:
            final.append(name)
    require(set(final) <= set(selected), "FINAL_IDENTITY_REENTRY")
    metrics = {key: counts(values, truth, raw) for key, values in (("raw", raw), ("pool", selected), ("refit_primary", final))}
    close(done["counts"], metrics, "FINAL_TP_FP_FN")
    selected_set, final_set = set(selected), set(final)
    ordered = lambda values: [name for name in order if name in values]
    transitions = dict(raw_solver_missed_truth=ordered(truth-raw),
        screening_removed_truth=ordered((raw-selected_set)&truth), screening_removed_false=ordered((raw-selected_set)-truth),
        refit_removed_truth=ordered((selected_set-final_set)&truth), refit_removed_false=ordered((selected_set-final_set)-truth),
        refit_gained_names=[])
    require(done["transitions"] == transitions, "EXACT_STAGE_LOSSES")
    metric = metrics["refit_primary"]
    flags = {key: bool(final and metric["TP"] >= 100 and metric["FP"]*100 <= risk*len(final))
             for key, risk in (("primary_strict_1pct_recall80_pass", 1), ("primary_secondary_5pct_recall80_pass", 5))}
    require(done["endpoint_flags"] == flags, "ENDPOINT_FLAGS")
    return dict(status="PASS", case=case, stage="refit", fit_performed=True, screen_review=screening,
        counts=metrics, transitions=transitions, endpoint_flags=flags, final_names=final,
        final_true_names=ordered(final_set&truth), final_false_names=ordered(final_set-truth),
        original_B_residual=residual, original_KKT_receipts=overall, block_count=64,
        max_reconstruction_error=max_reconstruction_error, reconstruction_bound=reconstruction_bound,
        refit_runtime=binding["runtime"], screen_pushed_commit=binding["screen_pushed_commit"],
        completion_sha256=sha(directory/"completion.json"), refit_binding_sha256=sha(directory/"refit_binding.json"),
        arrays_sha256=sha(fit/"learned_arrays.npz"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=("CHECK", "NEW_COMPOSITION"), required=True)
    parser.add_argument("--stage", choices=("screen", "refit"), default="refit")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = args.output.resolve()
    context = screen_review(output, args.case)
    result = context[0] if args.stage == "screen" else refit_review(output, args.case, context)
    result.update(reviewer_sha256=sha(__file__), independent_FDR_claim=False,
        cached_only=True, solver_optimizer_rho_training_called=False, accounting_helper_imported=False,
        KKT_recomputed=False, scope="Independent scalar scores/fixed-cut sets, exact aliases/identity counts and stage losses; "
              "final arrays and every block, original float64 KKT receipts, original-B residual. "
              "All cases remain exposed development data; no threshold retuning.")
    destination = output/"cases"/args.case/("independent_"+args.stage+"_review.json")
    data = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+"\n"
    if destination.exists():
        require(destination.read_text(encoding="utf-8") == data, "IMMUTABLE_REVIEW_CHANGED")
    else:
        with destination.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(data)
    print("INDEPENDENT_CACHE_PASS", args.case, args.stage, str(destination))


if __name__ == "__main__":
    main()
