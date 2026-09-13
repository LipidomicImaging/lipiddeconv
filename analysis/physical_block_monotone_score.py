"""One fixed DEV-only monotone logistic correction; observable-only inference.

No deconvolution, rho calculation or physical feature extraction occurs here.
The three evidence coefficients are nonnegative. This is an uncalibrated
development score, not a correctness probability or a guarantee of FDR.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
import time

for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_key] = "1"

FEATURES = ["a", "a_squared", "r", "s", "c"]
OPTIONS = {"maxiter": 2000, "gtol": 1e-8, "ftol": 2.220446049250313e-9}
ARTIFACTS = ("model.json", "thresholds.json", "train_scores.json",
             "fit_record.json", "reference_model.json", "reference_train_scores.json",
             "fit_reservation.json")


def require(ok, message):
    if not ok:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write(path, value):
    path = Path(path)
    data = json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        require(path.read_text(encoding="utf-8") == data, "IMMUTABLE_OUTPUT_CHANGED:" + str(path))
    else:
        with path.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(data)


def _finite(value):
    value = float(value)
    require(math.isfinite(value), "NONFINITE_VALUE")
    return value


def _index(rows, key):
    result = {r[key]: r for r in rows}
    require(len(result) == len(rows), "DUPLICATE_MEMBERSHIP:" + key)
    return result


def validate_model(model):
    require(model["features"] == FEATURES, "FEATURE_SCHEMA")
    for key in ("base", "auxiliary"):
        require(len(model[key]["mean"]) == len(model[key]["scale"]) == 2, "NORM_SHAPE")
        for mean, scale in zip(model[key]["mean"], model[key]["scale"]):
            _finite(mean)
            require(_finite(scale) > 0, "NONPOSITIVE_SCALE")
    require(len(model["coef"]) == 5, "COEFFICIENT_SHAPE")
    for coef in model["coef"]:
        _finite(coef)
    require(all(coef >= 0 for coef in model["coef"][2:]), "NONMONOTONE_COEFFICIENT")
    _finite(model["intercept"])
    return model


def _observable_values(row, feature):
    abundance, rho = _finite(row["X_hat"]), _finite(row["rho_zero"])
    gain = _finite(feature["predictive_gain"])
    fraction = _finite(feature["positive_block_fraction"])
    require(abundance >= 0 and rho >= 0 and 0 <= fraction <= 1, "OBSERVABLE_DOMAIN")
    return math.log10(max(abundance, 1e-12)), math.log10(max(rho, 1e-24)), gain, fraction


def matrix(reported_rows, feature_rows, model):
    """Five columns for already reported rows; never reads truth fields."""
    import numpy as np
    validate_model(model)
    features = _index(feature_rows, "molecular_name")
    _index(reported_rows, "lipid_name")
    values = []
    for row in reported_rows:
        require(row["raw_solver_reported"], "UNREPORTED_MATRIX_ROW")
        x, rho, gain, fraction = _observable_values(row, features[row["lipid_name"]])
        a = (x - model["base"]["mean"][0]) / model["base"]["scale"][0]
        r = (rho - model["base"]["mean"][1]) / model["base"]["scale"][1]
        s = (gain - model["auxiliary"]["mean"][0]) / model["auxiliary"]["scale"][0]
        c = (fraction - model["auxiliary"]["mean"][1]) / model["auxiliary"]["scale"][1]
        values.append([a, a*a, r, s, c])
    result = np.asarray(values, dtype=np.float64).reshape(-1, 5)
    require(np.isfinite(result).all(), "NONFINITE_TRANSFORM")
    return result


def score_rows(rows, feature_rows, model):
    """Same-order float/None scores. Unreported rows need no valid rho or label."""
    import numpy as np
    from scipy.special import expit
    _index(rows, "lipid_name")
    features = _index(feature_rows, "molecular_name")
    require(set(features) == {r["lipid_name"] for r in rows}, "FEATURE_MEMBERSHIP")
    reported = [r for r in rows if r["raw_solver_reported"]]
    values = expit(matrix(reported, feature_rows, model) @ np.asarray(model["coef"]) + model["intercept"])
    lookup = {r["lipid_name"]: float(v) for r, v in zip(reported, values)}
    return [lookup.get(r["lipid_name"]) for r in rows]


def objective_and_gradient(theta, x, y):
    import numpy as np
    from scipy.special import expit
    theta, x, y = np.asarray(theta), np.asarray(x), np.asarray(y)
    require(theta.shape == (6,) and x.shape == (len(y), 5), "OBJECTIVE_SHAPE")
    logits = x @ theta[:5] + theta[5]
    residual = expit(logits) - y
    value = np.logaddexp(0., logits).sum() - y @ logits + .5 * (theta[:5] @ theta[:5])
    gradient = np.r_[x.T @ residual + theta[:5], residual.sum()]
    return float(value), gradient


def projected_gradient(theta, gradient):
    result = [float(v) for v in gradient]
    for j in (2, 3, 4):
        if theta[j] <= 0 and result[j] > 0:
            result[j] = 0.
    return result


def _check_scores(rows, scores):
    _index(rows, "lipid_name")
    require(len(rows) == len(scores), "SCORE_LENGTH")
    for row, score in zip(rows, scores):
        if row["raw_solver_reported"]:
            require(score is not None and 0 <= _finite(score) <= 1, "INVALID_REPORTED_SCORE")
        else:
            require(score is None, "UNREPORTED_SCORE_MUST_BE_NONE")


def select_names(rows, scores, threshold):
    _check_scores(rows, scores)
    if threshold is None:
        return []
    threshold = _finite(threshold)
    return [r["lipid_name"] for r, score in zip(rows, scores)
            if r["raw_solver_reported"] and score >= threshold]


def counts(rows, names):
    known = _index(rows, "lipid_name")
    selected = set(names)
    require(selected <= set(known), "UNKNOWN_IDENTITY")
    truth = {r["lipid_name"] for r in rows if r["molecular_truth"]}
    raw_truth = {r["lipid_name"] for r in rows if r["molecular_truth"] and r["raw_solver_reported"]}
    tp, fp = len(truth & selected), len(selected - truth)
    return dict(TP=tp, FP=fp, FN=len(truth)-tp, retained=len(selected),
                FDP=fp/len(selected) if selected else None,
                all_truth_recall=tp/len(truth) if truth else None,
                TP_retention=tp/len(raw_truth) if raw_truth else None,
                all_truth_count=len(truth), raw_solver_misses=len(truth-raw_truth))


def choose_thresholds(rows, scores):
    """Complete ties; all truths are denominator. Three separate DEV policies."""
    _check_scores(rows, scores)
    total_truth = counts(rows, [])['all_truth_count']
    require(total_truth > 0, "NO_TRAIN_TRUTH")
    cuts = sorted({float(s) for r, s in zip(rows, scores) if r["raw_solver_reported"]}, reverse=True)
    pool, best = None, {5: None, 1: None}
    for cut in cuts:
        metric = counts(rows, select_names(rows, scores, cut))
        if pool is None and metric["TP"] * 100 >= 95 * total_truth:
            pool = cut
        for limit in best:
            if metric["FP"] * 100 <= limit * metric["retained"]:
                candidate = (metric["TP"], -metric["FP"], cut)
                if best[limit] is None or candidate > best[limit]:
                    best[limit] = candidate
    result = {"pool_threshold": pool, "required_pool_truth_count": (95*total_truth+99)//100}
    for limit in best:
        result[f"fdp{limit}_threshold"] = best[limit][2] if best[limit] is not None else None
    for endpoint in ("pool", "fdp5", "fdp1"):
        result["train_"+endpoint] = counts(rows, select_names(rows, scores, result[endpoint+"_threshold"]))
    result["rule"] = "DEV only; complete ties; 95% all-truth pool and separate empirical FDP policies; no FDR guarantee"
    return result


def _scalar_prob(logit):
    if logit >= 0:
        return 1. / (1. + math.exp(-logit))
    value = math.exp(logit)
    return value / (1. + value)


def scalar_predictions(rows, feature_rows, model, old_model=False):
    """Independent scalar probability replay, also for the sealed eight-term reference."""
    features = _index(feature_rows, "molecular_name")
    result = []
    for row in rows:
        if not row["raw_solver_reported"]:
            result.append(None)
            continue
        x, rho, gain, fraction = _observable_values(row, features[row["lipid_name"]])
        a, r = [(v-m)/scale for v, m, scale in zip((x, rho), model["base"]["mean"], model["base"]["scale"])]
        s, c = [(v-m)/scale for v, m, scale in zip((gain, fraction), model["auxiliary"]["mean"], model["auxiliary"]["scale"])]
        phi = [a, r, a*a, a*r, r*r, float(row["rho_zero"] == 0), s, c] if old_model else [a, a*a, r, s, c]
        require(len(phi) == len(model["coef"]), "SCALAR_COEF_SHAPE")
        result.append(_scalar_prob(math.fsum([model["intercept"]] + [v*w for v, w in zip(phi, model["coef"])])))
    return result


def review_normalization(rows, features, model):
    by_name = _index(features, "molecular_name")
    values = [_observable_values(r, by_name[r["lipid_name"]]) for r in rows if r["raw_solver_reported"]]
    require(values, "EMPTY_TRAIN")
    for j, (key, k) in enumerate((("base", 0), ("base", 1), ("auxiliary", 0), ("auxiliary", 1))):
        mean = math.fsum(v[j] for v in values)/len(values)
        scale = max(math.sqrt(math.fsum((v[j]-mean)**2 for v in values)/len(values)), 1e-12)
        require(math.isclose(mean, model[key]["mean"][k], rel_tol=1e-12, abs_tol=1e-12), "DEV_MEAN_CHANGED")
        require(math.isclose(scale, model[key]["scale"][k], rel_tol=1e-12, abs_tol=1e-12), "DEV_SCALE_CHANGED")


def review_predictions(rows, features, model, saved_scores):
    validate_model(model)
    _check_scores(rows, saved_scores)
    require(set(_index(features, "molecular_name")) == set(_index(rows, "lipid_name")), "FEATURE_MEMBERSHIP")
    expected = scalar_predictions(rows, features, model)
    error = max((abs(a-b) for a, b in zip(expected, saved_scores) if a is not None), default=0.)
    require(error <= 1e-12, "PROBABILITY_REPLAY")
    return dict(status="PASS", max_probability_error=error, reported_count=sum(s is not None for s in saved_scores),
                truth_used=False, monotonicity="nondecreasing in rho/S/C; rho floor unchanged")


def validate_model_directory(model_dir):
    """Portable artifact/hash validation; does not open training source paths."""
    root = Path(model_dir)
    seal = read(root / "model_seal.json")
    require(seal["status"] == "ONE_DEV_MODEL_FROZEN", "MODEL_NOT_FROZEN")
    require(set(seal["artifact_hashes"]) == set(ARTIFACTS), "SEAL_MEMBERSHIP")
    for name, expected in seal["artifact_hashes"].items():
        require(sha(root / name) == expected, "MODEL_ARTIFACT_CHANGED:" + name)
    model = validate_model(read(root / "model.json"))
    return dict(model=model, thresholds=read(root / "thresholds.json"), seal=seal)


def _source_paths(args):
    return {key: Path(getattr(args, key)).resolve() for key in
            ("rows", "features", "reference_model", "reference_scores", "protocol")}


def fit(args):
    """One optimization only. An interrupted reservation prohibits retraining."""
    import numpy as np
    import scipy
    from scipy.optimize import minimize
    output = Path(args.output)
    sources = _source_paths(args)
    hashes = {key: sha(path) for key, path in sources.items()}
    require(re.fullmatch(r"[0-9a-f]{40}", args.snapshot_commit) is not None, "SNAPSHOT_COMMIT_REQUIRED")
    if (output / "model_seal.json").exists():
        validate_model_directory(output)
        saved = read(output / "fit_record.json")
        require(saved["source_hashes"] == hashes and saved["snapshot_commit"] == args.snapshot_commit,
                "EXISTING_MODEL_BINDING_MISMATCH")
        print("MODEL_ALREADY_FROZEN", flush=True)
        return
    require(not (output / "fit_reservation.json").exists(), "PRIOR_FIT_RESERVATION_NO_RETRAIN")
    rows, features = read(sources["rows"]), read(sources["features"])
    reference, reference_scores = read(sources["reference_model"]), read(sources["reference_scores"])
    reported = [r for r in rows if r["raw_solver_reported"]]
    require(len(rows) == 377 and len(reported) == 174 and sum(r["molecular_truth"] for r in rows) == 125
            and sum(r["molecular_truth"] for r in reported) == 125, "FIXED_DEV_MEMBERSHIP")
    require([r["lipid_name"] for r in reported] == reference["training_names"], "REFERENCE_TRAIN_MEMBERSHIP")
    require([r["lipid_name"] for r in rows] == [r["lipid_name"] for r in reference_scores], "REFERENCE_SCORE_ORDER")
    require(all({k: v for k, v in saved.items() if k != "new_score"} == row
                for row, saved in zip(rows, reference_scores)), "REFERENCE_DEV_ROWS_CHANGED")
    old_scores = [r["new_score"] for r in reference_scores]
    _check_scores(rows, old_scores)
    old_expected = scalar_predictions(rows, features, reference, old_model=True)
    require(max(abs(a-b) for a, b in zip(old_scores, old_expected) if a is not None) <= 1e-12,
            "REFERENCE_PROBABILITY_REPLAY")
    model = dict(features=FEATURES, base={k: copy.deepcopy(reference["base"][k]) for k in ("mean", "scale")},
                 auxiliary={k: copy.deepcopy(reference["auxiliary"][k]) for k in ("mean", "scale")},
                 coef=[0.]*5, intercept=0., training_names=[r["lipid_name"] for r in reported],
                 probability_is_uncalibrated=True, constrained_nonnegative=["r", "s", "c"],
                 normalization_source="sealed original DEV reference; population std and unchanged floors")
    review_normalization(rows, features, model)
    x = matrix(reported, features, model)
    y = np.asarray([float(r["molecular_truth"]) for r in reported])
    binding = dict(source_hashes=hashes, source_paths={k: str(v) for k, v in sources.items()},
                   implementation_sha256=sha(__file__), snapshot_commit=args.snapshot_commit,
                   fit_count=1, role="DEV_TRAIN", options=OPTIONS, initial_theta=[0.]*6,
                   objective="sum logloss + 0.5 * squared coefficient norm; intercept unpenalized; C=1",
                   runtime=dict(python=sys.version, numpy=np.__version__, scipy=scipy.__version__))
    write(output / "fit_reservation.json", binding)
    started = time.perf_counter()
    result = minimize(objective_and_gradient, np.zeros(6), args=(x, y), jac=True,
                      method="L-BFGS-B", bounds=[(None, None)]*2+[(0., None)]*3+[(None, None)], options=OPTIONS)
    elapsed = time.perf_counter()-started
    diagnostic = dict(success=bool(result.success), message=str(result.message), iterations=int(result.nit),
                      function_evaluations=int(result.nfev), elapsed_seconds=elapsed, objective=float(result.fun),
                      gradient=result.jac.tolist(), projected_gradient=projected_gradient(result.x, result.jac),
                      theta=result.x.tolist())
    diagnostic["max_projected_gradient"] = max(abs(v) for v in diagnostic["projected_gradient"])
    write(output / "optimizer_result.json", diagnostic)
    require(result.success and np.isfinite(result.x).all() and np.isfinite(result.jac).all()
            and math.isfinite(float(result.fun)), "ONE_FIT_FAILED_NO_RETRY")
    model.update(coef=result.x[:5].tolist(), intercept=float(result.x[5]))
    scores = score_rows(rows, features, model)
    prediction_review = review_predictions(rows, features, model, scores)
    thresholds = choose_thresholds(rows, scores)
    thresholds["old_model_thresholds"] = choose_thresholds(rows, old_scores)
    record = dict(binding, optimizer=diagnostic, prediction_review=prediction_review,
                  optimizer_result_sha256=sha(output / "optimizer_result.json"))
    for name, value in (("model.json", model), ("thresholds.json", thresholds),
                        ("train_scores.json", [dict(r, new_score=s) for r, s in zip(rows, scores)]),
                        ("reference_model.json", reference), ("reference_train_scores.json", reference_scores),
                        ("fit_record.json", record)):
        write(output / name, value)
    write(output / "model_seal.json", dict(status="ONE_DEV_MODEL_FROZEN", snapshot_commit=args.snapshot_commit,
          artifact_hashes={name: sha(output/name) for name in ARTIFACTS}))
    print("MODEL_FROZEN", json.dumps(thresholds, ensure_ascii=False), flush=True)


def review(args):
    """Cached arithmetic only; no call to any optimizer or training function."""
    import numpy as np
    root = Path(args.model_dir)
    validated = validate_model_directory(root)
    model, thresholds = validated["model"], validated["thresholds"]
    record = read(root / "fit_record.json")
    require(sha(__file__) == record["implementation_sha256"], "IMPLEMENTATION_CHANGED")
    sources = {key: Path(getattr(args, key, None) or record["source_paths"][key]) for key in record["source_paths"]}
    for key, path in sources.items():
        require(sha(path) == record["source_hashes"][key], "SOURCE_CHANGED:" + key)
    rows, features = read(sources["rows"]), read(sources["features"])
    saved = read(root / "train_scores.json")
    require([dict(r, new_score=s["new_score"]) for r, s in zip(rows, saved)] == saved, "TRAIN_ROWS_CHANGED")
    require(len(rows) == len(saved), "TRAIN_LENGTH")
    scores = [r["new_score"] for r in saved]
    report = review_predictions(rows, features, model, scores)
    review_normalization(rows, features, model)
    reference = read(sources["reference_model"])
    for key in ("base", "auxiliary"):
        require(model[key] == {k: reference[key][k] for k in ("mean", "scale")}, "REFERENCE_NORM_CHANGED")
    old_rows = read(sources["reference_scores"])
    require(old_rows == read(root / "reference_train_scores.json"), "REFERENCE_SCORES_CHANGED")
    expected_thresholds = choose_thresholds(rows, scores)
    expected_thresholds["old_model_thresholds"] = choose_thresholds(rows, [r["new_score"] for r in old_rows])
    require(thresholds == expected_thresholds, "THRESHOLDS_CHANGED")
    reported = [r for r in rows if r["raw_solver_reported"]]
    require([r["lipid_name"] for r in reported] == model["training_names"], "TRAINING_NAMES_CHANGED")
    theta = np.r_[model["coef"], model["intercept"]]
    value, gradient = objective_and_gradient(theta, matrix(reported, features, model),
                                             [float(r["molecular_truth"]) for r in reported])
    require(math.isclose(value, record["optimizer"]["objective"], abs_tol=1e-10, rel_tol=1e-12), "OBJECTIVE_REPLAY")
    require(np.allclose(gradient, record["optimizer"]["gradient"], atol=1e-10, rtol=1e-10), "GRADIENT_REPLAY")
    require(record["optimizer"]["success"], "OPTIMIZER_DID_NOT_SUCCEED")
    require(record["options"] == OPTIONS and record["fit_count"] == 1, "FIT_POLICY_CHANGED")
    require(sha(root / "optimizer_result.json") == record["optimizer_result_sha256"], "OPTIMIZER_RECEIPT_CHANGED")
    report.update(truth_used="DEV threshold/normalization review only", optimizer_called=False,
                  model_seal_sha256=sha(root / "model_seal.json"),
                  max_projected_gradient=max(abs(v) for v in projected_gradient(theta, gradient)),
                  train_pool=thresholds["train_pool"], train_fdp5=thresholds["train_fdp5"], train_fdp1=thresholds["train_fdp1"])
    write(args.output or root / "model_review.json", report)
    print(json.dumps(report, ensure_ascii=False), flush=True)


def score(args):
    root = Path(args.model_dir)
    validated = validate_model_directory(root)
    rows, features = read(args.rows), read(args.features)
    scores = score_rows(rows, features, validated["model"])
    report = review_predictions(rows, features, validated["model"], scores)
    selections = {key: select_names(rows, scores, validated["thresholds"][key+"_threshold"])
                  for key in ("pool", "fdp5", "fdp1")}
    result = dict(model_seal_sha256=sha(root / "model_seal.json"), rows_sha256=sha(args.rows),
                  features_sha256=sha(args.features), review=report, selections=selections,
                  rows=[dict(lipid_name=r["lipid_name"], raw_solver_reported=r["raw_solver_reported"], new_score=s)
                        for r, s in zip(rows, scores)])
    write(args.output, result)
    print("SCORED", json.dumps({key: len(value) for key, value in selections.items()}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    pfit = sub.add_parser("fit")
    for arg in ("rows", "features", "reference-model", "reference-scores", "protocol", "snapshot-commit", "output"):
        pfit.add_argument("--"+arg, required=True)
    pfit.set_defaults(func=fit)
    preview = sub.add_parser("review")
    preview.add_argument("--model-dir", required=True)
    preview.add_argument("--output")
    for arg in ("rows", "features", "reference-model", "reference-scores", "protocol"):
        preview.add_argument("--"+arg)
    preview.set_defaults(func=review)
    pscore = sub.add_parser("score")
    for arg in ("model-dir", "rows", "features", "output"):
        pscore.add_argument("--"+arg, required=True)
    pscore.set_defaults(func=score)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
