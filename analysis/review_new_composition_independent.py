"""Independent completed-cache review; NumPy/stdlib only, never fits or rho calls.

Run --stage selection after both initial stages and selection are downloaded;
run --stage refit after the actual pool refit. Missing completions fail closed.
The reviewer is intentionally outside the frozen production CODE membership.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path

for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_key] = "1"

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CASE = ROOT / "results/physical_block_score_correction_v2/new_composition_case"
SOURCE = CASE.parent / "source_new_composition/prepared_arrays.npz"
SOURCE_SHA = "48a517e0f77568f7811ff8c711ab0223174ca8d8ad7053d20c5f7b7c6f52fbe7"
MODEL_SEAL_SHA = "d600620cddd3027166ab7743b8fe2f48430211fac419c36e5280185de2da8fc2"
ROLE = "NEW_COMPOSITION_CHECK"


def require(ok, message):
    if not ok:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def check(actual, expected, label):
    """Exact discrete fields; floating aggregates allow only arithmetic roundoff."""
    if isinstance(expected, dict):
        require(isinstance(actual, dict) and set(expected) <= set(actual), label)
        for key, value in expected.items():
            check(actual[key], value, label + "." + key)
    elif isinstance(expected, list):
        require(isinstance(actual, list) and len(actual) == len(expected), label)
        for i, (left, right) in enumerate(zip(actual, expected)):
            check(left, right, label + "." + str(i))
    elif isinstance(expected, float):
        require(isinstance(actual, (int, float)) and math.isfinite(actual)
                and math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-15), label)
    else:
        require(actual == expected, label)


def hashes(directory, mapping, expected=None):
    if expected is not None:
        require(set(mapping) == set(expected), "ARTIFACT_MEMBERSHIP:" + str(directory))
    for name, digest in mapping.items():
        path = (directory / name).resolve()
        require(path.is_relative_to(directory.resolve()), "NONLOCAL_ARTIFACT")
        require(sha(path) == digest, "HASH_CHANGED:" + str(path))


def binding():
    return dict(design_sha256=sha(CASE / "design.json"),
                model_binding_sha256=sha(CASE / "model_binding.json"))


def prepare():
    design = read(CASE / "design.json")
    check(read(CASE / "design_seal.json"), dict(
        status="PREPARED_BEFORE_MODEL_FIT_AND_NEW_SOLVES",
        design_sha256=sha(CASE / "design.json")), "DESIGN_SEAL")
    require(design["case"] == ROLE and design["candidate_report_gate"] == .001, "CASE_OR_GATE")
    hashes(ROOT, design["code"])
    hashes(CASE, design["prepared"], {p.relative_to(CASE).as_posix()
                                    for p in (CASE / "prepared").iterdir()})
    require(sha(SOURCE) == SOURCE_SHA == design["source_prepared_sha256"], "SOURCE_HASH")
    model_binding = read(CASE / "model_binding.json")
    check(model_binding, dict(status="PUSHED_MODEL_FIXED_BEFORE_NEW_COMPOSITION_SOLVES",
          design_sha256=sha(CASE / "design.json"),
          design_seal_sha256=sha(CASE / "design_seal.json")), "MODEL_BINDING")
    hashes(CASE / "model", model_binding["artifacts"])
    require(sha(CASE / "model/model_seal.json") == MODEL_SEAL_SHA, "FIXED_MODEL_SEAL")
    seal = read(CASE / "model/model_seal.json")
    require(seal["status"] == "ONE_DEV_MODEL_FROZEN", "MODEL_NOT_FROZEN")
    hashes(CASE / "model", seal["artifact_hashes"])
    source_input = read(CASE / "prepared/source_input.json")
    scientific = source_input["scientific"]
    require(hashlib.sha256(canonical(scientific)).hexdigest() == source_input["fingerprint"],
            "SCIENTIFIC_FINGERPRINT")
    metadata = read(CASE / "prepared/source_metadata.json")
    names = metadata["lipid_name"]
    order = list(dict.fromkeys(names))
    aliases = {name: [i for i, value in enumerate(names) if value == name] for name in order}
    manifest = read(CASE / "prepared/block_manifest.json")
    require(len(names) == 391 and len(order) == 377 and manifest["candidate_names"] == names
            and manifest["molecular_names"] == order and manifest["block_count"] == 34, "UNIVERSE")
    with np.load(SOURCE, allow_pickle=False) as z:
        A, B, mask, truth_array = z["A_solver"], z["B"], z["mask"], z["X_true"]
    require(A.shape == (1084, 391) and B.shape == (1084, 200, 90)
            and mask.shape == (200, 90) and mask.dtype == bool and mask.sum() == 15837
            and truth_array.shape == (391, 200, 90), "SOURCE_SHAPES")
    require(np.isfinite(truth_array).all() and np.all(truth_array >= 0), "TRUTH_ARRAY")
    true_indices = np.flatnonzero(np.any(truth_array[:, mask] > 0, axis=1)).tolist()
    truth = {names[i] for i in true_indices}
    require(len(truth) == 125 and true_indices == scientific["truth_indices"]
            and truth == set(scientific["truth_names"]), "TRUTH_MEMBERSHIP")
    del truth_array
    old_path = ROOT / "results/physical_block_prediction_v1/prepared/CHECK_records.json"
    require(sha(old_path) == "7beffe182c65bd7792002b779f8f5b7ac75f1f6a9486b5f24d9db884ae262170",
            "OLD_TRUTH_SOURCE_CHANGED")
    old_truth = {r["lipid_name"] for r in read(old_path) if r["molecular_truth"]}
    require(len(old_truth) == 125 and not truth & old_truth, "OLD_TRUTH_OVERLAP")
    for key, value in (("A", A), ("B", B), ("mask", mask)):
        require(np.isfinite(value).all(), "NONFINITE_SOURCE:" + key)
        check(design["source_arrays"][key], dict(shape=list(value.shape), dtype=value.dtype.str,
              sha256=hashlib.sha256(value.tobytes(order="C")).hexdigest()), "SOURCE_ARRAY:" + key)
    require(np.all(A >= 0) and np.all(B[:, ~mask] == 0), "SOURCE_SUPPORT")
    with np.load(CASE / "prepared/portable.npz", allow_pickle=False) as z:
        require(np.array_equal(A, z["A"]) and np.array_equal(mask, z["mask"]), "PORTABLE_SOURCE")
        b = z["b"]
    mean_error = float(np.max(np.abs(B[:, mask].mean(axis=1, dtype=np.float64) - b)))
    require(mean_error <= 1e-13 * max(float(np.max(np.abs(b))), 1e-12), "PORTABLE_MEAN")
    return design, metadata, names, order, aliases, truth, scientific, A, B, mask, b, dict(
        source_sha256=SOURCE_SHA, model_seal_sha256=MODEL_SEAL_SHA,
        design_sha256=sha(CASE / "design.json"), old_truth_source_sha256=sha(old_path),
        candidate_count=391, molecular_count=377, true_names=sorted(truth), true_count=125,
        old_truth_overlap=0, foreground_pixels=15837, portable_mean_max_error=mean_error,
        frozen_code_hash_count=len(design["code"]), pushed_model_commit=model_binding["pushed_model_commit"])


def completion(stage, expected, stage_binding=None):
    directory = CASE / stage
    done = read(directory / "completion.json")
    check(done, dict(status="COMPLETE", case=ROLE, stage=stage,
          binding=binding() if stage_binding is None else stage_binding), "COMPLETION:" + stage)
    hashes(directory, done["artifacts"], expected)
    review_path = CASE / "reviews" / (stage + "_remote.json")
    check(read(review_path), dict(status="CACHE_REVIEW_PASS", case=ROLE, stage=stage,
          design_sha256=sha(CASE / "design.json"),
          completion_sha256=sha(directory / "completion.json")), "OFFICIAL_REVIEW:" + stage)
    return done


def arrays(path, B, mask, allowed):
    with np.load(path, allow_pickle=False) as z:
        require(set(z.files) == {"X_hat", "B_hat"}, "LEARNED_KEYS")
        X, fitted = z["X_hat"], z["B_hat"]
    require(X.shape == (391, 200, 90) and fitted.shape == B.shape
            and X.dtype == fitted.dtype == np.float32 and np.isfinite(X).all()
            and np.isfinite(fitted).all() and np.all(X >= 0), "LEARNED_ARRAYS")
    require(np.all(X[:, ~mask] == 0) and np.all(fitted[:, ~mask] == 0), "BACKGROUND")
    excluded = sorted(set(range(391)) - set(allowed))
    require(np.all(X[excluded] == 0), "EXCLUDED_ALIAS_REENTRY")
    means = X[:, mask].mean(axis=1, dtype=np.float64)
    observed, fitted = B[:, mask], fitted[:, mask]
    sse, signal = 0., 0.
    for start in range(0, int(mask.sum()), 250):
        target = observed[:, start:start+250].astype(np.float64)
        difference = target - fitted[:, start:start+250].astype(np.float64)
        sse += float(np.sum(difference*difference, dtype=np.float64))
        signal += float(np.sum(target*target, dtype=np.float64))
    require(signal > 0 and math.isfinite(sse), "RESIDUAL_NONFINITE")
    return means, dict(foreground_squared_residual=sse, foreground_observation_squared_norm=signal,
                       foreground_relative_l2_residual=math.sqrt(sse/signal))


def metrics(selected, truth, raw):
    selected, raw = set(selected), set(raw)
    tp, fp = len(selected & truth), len(selected - truth)
    return dict(TP=tp, FP=fp, FN=125-tp, retained=len(selected), FDP=fp/len(selected) if selected else None,
                all_truth_recall=tp/125, TP_retention=tp/len(raw & truth) if raw & truth else None,
                all_truth_count=125, raw_solver_misses=len(truth-raw))


def scalar_score(row, feature, model, old=False):
    if not row["raw_solver_reported"]:
        return None
    abundance, rho = row["X_hat"], row["rho_zero"]
    require(abundance >= 0 and rho is not None and rho >= 0, "SCORE_OBSERVABLE_DOMAIN")
    a, r = [(v-m)/scale for v, m, scale in zip(
        (math.log10(max(abundance, 1e-12)), math.log10(max(rho, 1e-24))),
        model["base"]["mean"], model["base"]["scale"])]
    s, c = [(v-m)/scale for v, m, scale in zip(
        (feature["predictive_gain"], feature["positive_block_fraction"]),
        model["auxiliary"]["mean"], model["auxiliary"]["scale"])]
    phi = [a, r, a*a, a*r, r*r, float(rho == 0), s, c] if old else [a, a*a, r, s, c]
    require(len(phi) == len(model["coef"]), "COEFFICIENT_SHAPE")
    logit = math.fsum([model["intercept"]] + [x*w for x, w in zip(phi, model["coef"])])
    require(math.isfinite(logit), "NONFINITE_LOGIT")
    return 1/(1+math.exp(-logit)) if logit >= 0 else math.exp(logit)/(1+math.exp(logit))


def audit_selection(context):
    design, metadata, names, order, aliases, truth, scientific, A, B, mask, b, source_review = context
    directory = CASE / "nnls_rho"
    # Refuse partial stages before inspecting their numerical outcomes.
    require(read(directory / "completion.json")["status"] == "COMPLETE", "RAW_NOT_COMPLETE")
    require(read(CASE / "physical_features/completion.json")["status"] == "COMPLETE", "FEATURES_NOT_COMPLETE")
    means, residual = arrays(directory / "learned_arrays.npz", B, mask, range(391))
    active = np.flatnonzero(means > .001).tolist()
    fingerprint = hashlib.sha256(canonical(dict(stage="nnls_rho", **binding()))).hexdigest()
    done = read(directory / "nnls_complete.json")
    check(done, dict(status="NNLS_COMPLETE", fingerprint=fingerprint,
          arrays_sha256=sha(directory / "learned_arrays.npz"), residual=residual), "RAW_RECEIPT")
    rho_files = {f"rho/candidate_{i:04d}.json" for i in active}
    block_files = {f"nnls_blocks/block_{start:06d}_{min(start+250,15837):06d}.{ext}"
                   for start in range(0, 15837, 250) for ext in ("json", "npz")}
    require({"rho/"+p.name for p in (directory/"rho").iterdir()} == rho_files, "RHO_MEMBERSHIP")
    require({"nnls_blocks/"+p.name for p in (directory/"nnls_blocks").iterdir()} == block_files, "RAW_BLOCK_MEMBERSHIP")
    completion("nnls_rho", rho_files | block_files | {
        "learned_arrays.npz", "nnls_complete.json", "candidate_records.json", "molecular_records.json"})
    candidates, rows = read(directory / "candidate_records.json"), read(directory / "molecular_records.json")
    require(len(candidates) == 391 and [r["lipid_name"] for r in rows] == order, "RAW_RECORD_ORDER")
    for i, record in enumerate(candidates):
        detail = None
        if i in active:
            cache = read(directory / "rho" / f"candidate_{i:04d}.json")
            check(cache, dict(fingerprint=fingerprint, candidate_index=i,
                  b_sha256=hashlib.sha256(b.tobytes(order="C")).hexdigest(),
                  arrays_sha256=done["arrays_sha256"]), "RHO_BINDING")
            detail = cache["result"]
            require(all(math.isfinite(v) for v in detail.values()) and detail["rho_zero"] >= 0, "RHO_FINITE")
        check(record, dict(candidate_index=i, candidate_id=metadata["candidate_id"][i], lipid_name=names[i],
              lipid_class=metadata["lipid_class"][i], reportable_truth=names[i] in scientific["reportable_truth_names"],
              candidate_truth=i in scientific["truth_indices"], molecular_truth=names[i] in truth,
              X_hat=float(means[i]), raw_solver_reported=i in active,
              rho_zero=detail["rho_zero"] if detail is not None else None, rho_details=detail), "RAW_CANDIDATE")
    raw = []
    for row in rows:
        name = row["lipid_name"]
        selected_aliases = [i for i in aliases[name] if i in active]
        if selected_aliases:
            raw.append(name)
        check(row, dict(molecular_truth=name in truth, candidate_indices=aliases[name],
              reportable_truth=name in scientific["reportable_truth_names"],
              actually_perturbed_truth=name in scientific["actually_perturbed_truth_names"],
              reported_candidate_indices=selected_aliases, raw_solver_reported=bool(selected_aliases),
              X_hat=sum(float(means[i]) for i in selected_aliases),
              rho_zero=max(candidates[i]["rho_zero"] for i in selected_aliases) if selected_aliases else None),
              "RAW_MOLECULAR")
    feature_paths = {f"blocks/block_{i:03d}.{ext}" for i in range(34) for ext in ("json", "npz")}
    require({"blocks/"+p.name for p in (CASE/"physical_features/blocks").iterdir()} == feature_paths, "PHYSICAL_FILES")
    physical_done = completion("physical_features", feature_paths | {"features.json"})
    check(physical_done, dict(block_count=34, requested_models=12852), "PHYSICAL_COUNTS")
    features = read(CASE / "physical_features/features.json")
    require([r["molecular_name"] for r in features] == order, "FEATURE_ORDER")
    require(all(math.isfinite(r["predictive_gain"]) and 0 <= r["positive_block_fraction"] <= 1
                and len(r["blocks"]) == 34 for r in features), "FEATURE_DOMAIN")
    selection = read(CASE / "selection.json")
    check(selection, dict(status="FIXED_MODEL_SELECTION_COMPLETE", case=ROLE, binding=binding(),
          source_records_sha256=sha(directory / "molecular_records.json"),
          source_features_sha256=sha(CASE / "physical_features/features.json")), "SELECTION_BINDING")
    check(read(CASE / "selection_seal.json"), dict(selection_sha256=sha(CASE / "selection.json"),
          binding=binding()), "SELECTION_SEAL")
    cuts = read(CASE / "model/thresholds.json")
    require(selection["thresholds"] == cuts and len(selection["rows"]) == 377, "FROZEN_CUTS")
    outcome, sets_by_model, max_errors = {}, {}, {}
    for label, filename in (("corrected", "model.json"), ("old", "reference_model.json")):
        model = read(CASE / "model" / filename)
        if label == "corrected":
            require(model["features"] == ["a", "a_squared", "r", "s", "c"]
                    and all(x >= 0 for x in model["coef"][2:]), "MONOTONE_MODEL")
        scores = [scalar_score(r, f, model, old=label == "old") for r, f in zip(rows, features)]
        saved_key = "corrected_score" if label == "corrected" else "old_model_score"
        errors = []
        for original, saved, value in zip(rows, selection["rows"], scores):
            check(saved, original, "SELECTION_SOURCE_ROW")
            if value is None:
                require(saved[saved_key] is None, "UNREPORTED_SCORE")
            else:
                errors.append(abs(value-saved[saved_key]))
        require(max(errors, default=0.) <= 1e-12, "SCALAR_SCORE_REPLAY")
        local_cuts = cuts if label == "corrected" else cuts["old_model_thresholds"]
        sets = {endpoint: [r["lipid_name"] for r, value in zip(rows, scores)
                          if value is not None and local_cuts[endpoint+"_threshold"] is not None
                          and value >= local_cuts[endpoint+"_threshold"]]
                for endpoint in ("pool", "fdp5", "fdp1")}
        require(sets == selection["selected_names" if label == "corrected" else "old_model_selected_names"],
                "EXACT_SCALAR_SELECTION:" + label)
        metric = {key: metrics(value, truth, raw) for key, value in sets.items()}
        check(selection["counts" if label == "corrected" else "old_model_counts"], metric, "SELECTION_COUNTS")
        outcome[label], sets_by_model[label], max_errors[label] = metric, sets, max(errors, default=0.)
    pool = sets_by_model["corrected"]["pool"]
    require(pool == selection["pool_names"], "POOL_NAMES")
    for row in selection["rows"]:
        require(row["in_pool"] == (row["lipid_name"] in pool), "POOL_FLAG")
    review = dict(status="PASS", stage="selection", source=source_review, raw=metrics(raw, truth, raw),
        corrected=outcome["corrected"], old_model=outcome["old"], selected_names=sets_by_model,
        solver_missed_truth=[n for n in order if n in truth and n not in raw],
        corrected_filter_removed_truth={key: [n for n in raw if n in truth and n not in values]
                                        for key, values in sets_by_model["corrected"].items()},
        original_B_residual=residual, scalar_probability_max_errors=max_errors,
        selection_sha256=sha(CASE / "selection.json"),
        completion_sha256={s: sha(CASE/s/"completion.json") for s in ("nnls_rho", "physical_features")})
    return review, rows, raw, pool


def audit_refit(context, selection_review, rows, raw, pool):
    design, metadata, names, order, aliases, truth, scientific, A, B, mask, b, source_review = context
    directory = CASE / "pool_refit"
    done = read(directory / "completion.json")
    handoff = read(CASE / "case_handoff.json")
    check(handoff, dict(status="REVIEWED_DOWNLOADED_PUSHED_CASE_BEFORE_REFIT", binding=binding(),
          selection_sha256=sha(CASE / "selection.json"),
          selection_seal_sha256=sha(CASE / "selection_seal.json")), "REFIT_HANDOFF")
    expected_binding = dict(**binding(), selection_sha256=sha(CASE / "selection.json"),
                            case_handoff_sha256=sha(CASE / "case_handoff.json"))
    check(done, dict(case=ROLE, stage="pool_refit", binding=expected_binding), "REFIT_BINDING")
    if done["status"] == "NOT_RUN_EMPTY_OR_ABSENT_POOL":
        require(not pool and not (directory / "fit").exists(), "EMPTY_POOL_STOP")
        return dict(status="PASS", stage="refit", fit_performed=False, reason=done["reason"],
                    selection_review=selection_review, completion_sha256=sha(directory/"completion.json"))
    completion("pool_refit", {"candidate_records.json", "molecular_records.json", "fit/binding.json",
               "fit/completion.json", "fit/learned_arrays.npz"}, expected_binding)
    fit = directory / "fit"
    receipt, fit_binding = read(fit / "completion.json"), read(fit / "binding.json")
    indices = [i for i, name in enumerate(names) if name in pool]
    require(hashlib.sha256(canonical(fit_binding["scientific"])).hexdigest() == fit_binding["fingerprint"],
            "REFIT_FINGERPRINT")
    check(fit_binding["scientific"], dict(**design["source_arrays"], retained_indices=indices,
          source_binding=expected_binding, maxiter=3910, block_size=250, workers=4), "REFIT_INPUTS")
    check(receipt, dict(status="COMPLETE", fingerprint=fit_binding["fingerprint"],
          binding_sha256=sha(fit/"binding.json"), arrays_sha256=sha(fit/"learned_arrays.npz"),
          retained_indices=indices, candidate_count=391, fitted_column_count=len(indices),
          foreground_pixels=15837, all_outputs_finite=True), "REFIT_RECEIPT")
    block_files = {f"block_{start:06d}_{min(start+250,15837):06d}.{ext}"
                   for start in range(0, 15837, 250) for ext in ("json", "npz")}
    require({p.name for p in (fit/"nnls_blocks").iterdir()} == block_files, "REFIT_BLOCK_MEMBERSHIP")
    hashes(fit / "nnls_blocks", receipt["block_hashes"], block_files)
    means, residual = arrays(fit / "learned_arrays.npz", B, mask, indices)
    check(done["residual"], residual, "REFIT_ORIGINAL_B_RESIDUAL")
    candidates, final_rows = read(directory / "candidate_records.json"), read(directory / "molecular_records.json")
    require(len(candidates) == 391 and [r["lipid_name"] for r in final_rows] == order, "REFIT_RECORD_ORDER")
    raw_aliases = {i for row in rows for i in row["reported_candidate_indices"]}
    for i, row in enumerate(candidates):
        check(row, dict(candidate_index=i, lipid_name=names[i], molecular_truth=names[i] in truth,
              raw_solver_reported=i in raw_aliases,
              in_pool=names[i] in pool, refit_mean=float(means[i]), refit_reported=bool(means[i] > .001)),
              "REFIT_CANDIDATE")
    final = []
    for original, row in zip(rows, final_rows):
        name = row["lipid_name"]
        active = [i for i in aliases[name] if means[i] > .001]
        check(row, original, "REFIT_SOURCE_ROW")
        check(row, dict(in_pool=name in pool, refit_reported=bool(active),
              refit_reported_candidate_indices=active, refit_abundance=sum(float(means[i]) for i in active)),
              "REFIT_MOLECULAR")
        if active:
            final.append(name)
    require(set(final) <= set(pool), "REFIT_IDENTITY_REENTRY")
    counts = {key: metrics(value, truth, raw) for key, value in (("raw", raw), ("pool", pool), ("refit_primary", final))}
    check(done["counts"], counts, "REFIT_COUNTS")
    transitions = dict(raw_solver_missed_truth=[n for n in order if n in truth and n not in raw],
        screening_removed_truth=[n for n in raw if n in truth and n not in pool],
        screening_removed_false=[n for n in raw if n not in truth and n not in pool],
        refit_removed_truth=[n for n in pool if n in truth and n not in final],
        refit_removed_false=[n for n in pool if n not in truth and n not in final], refit_gained_names=[])
    require(done["transitions"] == transitions, "REFIT_TRANSITIONS")
    final_count = counts["refit_primary"]
    flags = {key: bool(final and final_count["TP"] >= 100 and final_count["FP"]*100 <= risk*len(final))
             for key, risk in (("primary_strict_1pct_recall80_pass", 1),
                               ("primary_secondary_5pct_recall80_pass", 5))}
    require(done["endpoint_flags"] == flags, "REFIT_ENDPOINT_FLAGS")
    return dict(status="PASS", stage="refit", fit_performed=True, selection_review=selection_review,
                counts=counts, transitions=transitions, final_names=final,
                final_true_names=[n for n in final if n in truth], final_false_names=[n for n in final if n not in truth],
                endpoint_flags=flags, original_B_residual=residual,
                completion_sha256=sha(directory/"completion.json"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("selection", "refit"), required=True)
    args = parser.parse_args()
    context = prepare()
    result, rows, raw, pool = audit_selection(context)
    if args.stage == "refit":
        result = audit_refit(context, result, rows, raw, pool)
    result.update(reviewer_sha256=sha(__file__), no_fit_or_training_or_rho_call=True,
        independent_FDR_claim=False, role="Exposed different-composition development check",
        scope="Direct array means, any-alias reporting, cached rho aggregation, scalar scores and exact fixed-cut sets; "
              "original-B residuals and artifact hashes. No repeated block loss/KKT or A@X reconstruction; "
              "those are covered by bound official cache reviews.")
    output = CASE / ("independent_" + args.stage + "_review.json")
    data = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if output.exists():
        require(output.read_text(encoding="utf-8") == data, "IMMUTABLE_REVIEW_CHANGED")
    else:
        with output.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(data)
    print("INDEPENDENT_CACHE_PASS", args.stage, str(output))


if __name__ == "__main__":
    main()
