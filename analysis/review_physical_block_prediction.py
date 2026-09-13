"""Independent cache-only checks for physical-block prediction experiments.

No optimizer, training function or block-builder implementation is invoked.
Block checks do not access truth; only the explicit TRAIN model audit uses
supplied labels to verify frozen cutoffs. Losses are unnormalized squared L2.
KKT uses the original float64 bound.
"""
from __future__ import annotations

import math
import numpy as np


def _require(condition, message):
    if not condition:
        raise RuntimeError(message)


def _indices(value, count, label, nonempty=False):
    array = np.asarray(value)
    _require(array.ndim == 1 and (array.size == 0 or array.dtype.kind in "iu"), label + "_NOT_INTEGER_VECTOR")
    _require(not nonempty or array.size > 0, label + "_EMPTY")
    _require(np.all(array >= 0) and np.all(array < count), label + "_OUT_OF_RANGE")
    values = array.astype(np.int64)
    _require(len(set(values.tolist())) == values.size, label + "_DUPLICATE")
    return values


def _kkt(A, b, x):
    """Independent copy of the original engineering bound on allowed columns."""
    _require(x.dtype == np.float64 and np.isfinite(x).all() and (x >= 0).all(), "INVALID_FLOAT64_COEFFICIENTS")
    if A.shape[1] == 0:
        return dict(max_dual_violation=0., max_complementarity=0., max_bound_ratio=0., empty_design=True)
    fitted = A @ x
    gradient = A.T @ (fitted - b)
    bound = 64 * np.finfo(np.float64).eps * max(A.shape) * (np.abs(A).T @ (np.abs(fitted) + np.abs(b)))
    violation = np.maximum(-gradient, 0.)
    violation[x > 0] = np.abs(gradient[x > 0])
    _require(np.isfinite(gradient).all() and np.isfinite(bound).all() and np.all(violation <= bound), "KKT_FAILED")
    ratios = np.divide(violation, bound, out=np.zeros_like(bound), where=bound > 0)
    return dict(max_dual_violation=float(np.maximum(-gradient, 0.).max()),
                max_complementarity=float(np.abs(x * gradient).max()),
                max_bound_ratio=float(ratios.max()), empty_design=False)


def _check_saved(arrays, key, expected, signal_scale):
    _require(key in arrays, "MISSING_SAVED_ARRAY:" + key)
    saved = np.asarray(arrays[key])
    expected = np.asarray(expected)
    _require(saved.shape == expected.shape and np.isfinite(saved).all(), "INVALID_SAVED_ARRAY:" + key)
    # Numerical comparison only; this tolerance never sets a selection cutoff.
    _require(np.allclose(saved, expected, rtol=1e-10, atol=1e-15 * max(signal_scale, 1e-30)),
             "SAVED_VALUE_MISMATCH:" + key)


def review_block(A, b, names, held_indices, arrays):
    """Recompute a block's cached fits and losses, without invoking a solver.

    Required arrays: full_x[N], deleted_x[G,N], deleted_names[G], and scalar/
    G-vector full/deleted train/held losses. Candidate names and held indices
    come from the bound metadata. All same-name aliases must be deleted.
    """
    A, b = np.asarray(A, dtype=np.float64), np.asarray(b, dtype=np.float64)
    names = list(names)
    _require(A.ndim == 2 and b.shape == (A.shape[0],) and len(names) == A.shape[1], "INPUT_SHAPE_MISMATCH")
    _require(np.isfinite(A).all() and np.isfinite(b).all(), "NONFINITE_INPUT")
    _require(all(isinstance(name, str) for name in names), "INVALID_CANDIDATE_NAMES")
    held = _indices(held_indices, A.shape[0], "HELD_INDICES", nonempty=True)
    train = np.setdiff1d(np.arange(A.shape[0]), held)
    _require(train.size > 0, "EMPTY_TRAINING_CHANNELS")
    _require(all(key in arrays for key in ("full_x", "deleted_x", "deleted_names")), "MISSING_COEFFICIENT_ARRAYS")
    full, deleted = np.asarray(arrays["full_x"]), np.asarray(arrays["deleted_x"])
    deleted_names_array = np.asarray(arrays["deleted_names"])
    _require(deleted_names_array.ndim == 1 and deleted_names_array.dtype.kind in "US", "INVALID_DELETED_NAMES")
    deleted_names = deleted_names_array.tolist()
    _require(len(set(deleted_names)) == len(deleted_names) and set(deleted_names) <= set(names), "DELETED_NAME_MEMBERSHIP")
    _require(full.shape == (A.shape[1],) and deleted.shape == (len(deleted_names), A.shape[1]), "COEFFICIENT_SHAPE_MISMATCH")
    _require(full.dtype == deleted.dtype == np.float64 and np.isfinite(full).all() and
             np.isfinite(deleted).all() and (full >= 0).all() and (deleted >= 0).all(), "INVALID_FLOAT64_COEFFICIENTS")
    train_A, held_A = A[train], A[held]
    unsupported = ~np.any(train_A != 0, axis=0)
    _require((full[unsupported] == 0).all() and (deleted[:, unsupported] == 0).all(), "UNSUPPORTED_COLUMN_NONZERO")
    full_check = _kkt(train_A, b[train], full)
    full_prediction, full_train_prediction = held_A @ full, train_A @ full
    deleted_prediction, deleted_train_prediction = deleted @ held_A.T, deleted @ train_A.T
    full_train_loss = float(np.sum((full_train_prediction - b[train]) ** 2))
    full_held_loss = float(np.sum((full_prediction - b[held]) ** 2))
    deleted_train_loss = np.sum((deleted_train_prediction - b[train]) ** 2, axis=1)
    deleted_held_loss = np.sum((deleted_prediction - b[held]) ** 2, axis=1)
    signal_scale = float(np.sum(b ** 2))
    for key, expected in (("full_train_loss", full_train_loss), ("full_held_loss", full_held_loss),
                          ("deleted_train_loss", deleted_train_loss), ("deleted_held_loss", deleted_held_loss)):
        _check_saved(arrays, key, expected, signal_scale)
    for key, expected in (("full_held_prediction", full_prediction),
                          ("deleted_held_prediction", deleted_prediction)):
        if key in arrays:
            _check_saved(arrays, key, expected, np.sqrt(signal_scale))
    deleted_checks, group_train_supported, group_held_supported, aliases = [], [], [], []
    name_array = np.asarray(names)
    for j, name in enumerate(deleted_names):
        omitted = name_array == name
        _require((deleted[j, omitted] == 0).all(), "DELETED_ALIAS_NONZERO:" + name)
        deleted_checks.append(_kkt(train_A[:, ~omitted], b[train], deleted[j, ~omitted]))
        aliases.append(np.flatnonzero(omitted).tolist())
        group_train_supported.append(bool(np.any(train_A[:, omitted] != 0)))
        group_held_supported.append(bool(np.any(held_A[:, omitted] != 0)))
    for key, expected in (("candidate_train_supported", ~unsupported),
                          ("candidate_held_supported", np.any(held_A != 0, axis=0)),
                          ("molecular_train_supported", np.asarray(group_train_supported, dtype=bool)),
                          ("molecular_held_supported", np.asarray(group_held_supported, dtype=bool))):
        if key in arrays:
            saved = np.asarray(arrays[key])
            _require(saved.dtype == np.bool_ and np.array_equal(saved, expected), "SUPPORT_FLAG_MISMATCH:" + key)
    for key, shape in (("full_kkt", (3,)), ("deleted_kkt", (len(deleted_names), 3))):
        if key in arrays:
            saved = np.asarray(arrays[key])
            _require(saved.shape == shape and np.isfinite(saved).all() and (saved >= 0).all() and
                     np.all(saved[..., 2] <= 1.), "INVALID_SAVED_KKT:" + key)
            # Fresh checks above establish KKT directly. Dimensionless error
            # ratios need not match bitwise after independent BLAS operations.
    return dict(status="PASS", held_indices=held.tolist(), train_indices=train.tolist(),
                deleted_names=deleted_names, aliases=aliases, unsupported_columns=np.flatnonzero(unsupported).tolist(),
                group_train_supported=group_train_supported, group_held_supported=group_held_supported,
                full_train_loss=full_train_loss, full_held_loss=full_held_loss,
                deleted_train_loss=deleted_train_loss.tolist(), deleted_held_loss=deleted_held_loss.tolist(),
                held_loss_difference=(deleted_held_loss - full_held_loss).tolist(),
                full_KKT=full_check, deleted_KKT=deleted_checks)


def independent_blocks(components):
    """Find complete-envelope closures with a bipartite DFS, not union-find."""
    C = np.asarray(components)
    _require(C.ndim == 2 and np.isfinite(C).all() and (C >= 0).all(), "INVALID_COMPONENTS")
    rows_for_component = [np.flatnonzero(C[:, j] > 0).tolist() for j in range(C.shape[1])]
    _require(all(rows for rows in rows_for_component), "EMPTY_PHYSICAL_COMPONENT")
    components_for_row = [[] for _ in range(C.shape[0])]
    for j, rows in enumerate(rows_for_component):
        for row in rows:
            components_for_row[row].append(j)
    visited_rows, visited_components, groups = set(), set(), []
    for start in range(C.shape[0]):
        if start in visited_rows:
            continue
        stack, row_group, component_group = [start], [], []
        visited_rows.add(start)
        while stack:
            row = stack.pop()
            row_group.append(row)
            for j in components_for_row[row]:
                if j in visited_components:
                    continue
                visited_components.add(j)
                component_group.append(j)
                for neighbor in rows_for_component[j]:
                    if neighbor not in visited_rows:
                        visited_rows.add(neighbor)
                        stack.append(neighbor)
        groups.append(dict(channel_indices=sorted(row_group), component_indices=sorted(component_group)))
    return groups


def review_block_manifest(A, components, owner, names, blocks):
    """Check exact closures and every component/candidate/molecular membership."""
    A, C, owner = np.asarray(A), np.asarray(components), np.asarray(owner)
    names = list(names)
    _require(A.ndim == 2 and C.ndim == 2 and A.shape[0] == C.shape[0] and len(names) == A.shape[1], "MANIFEST_SHAPE_MISMATCH")
    _require(owner.shape == (C.shape[1],) and owner.dtype.kind in "iu" and
             np.all(owner >= 0) and np.all(owner < A.shape[1]), "INVALID_COMPONENT_OWNER")
    _require(np.isfinite(A).all() and (A >= 0).all(), "INVALID_NOMINAL_LIBRARY")
    expected = independent_blocks(C)
    _require(isinstance(blocks, (list, tuple)) and len(blocks) == len(expected), "BLOCK_COUNT_MISMATCH")
    by_channels = {tuple(group["channel_indices"]): group for group in expected}
    molecular_names = list(dict.fromkeys(names))
    molecular_index = {name: j for j, name in enumerate(molecular_names)}
    seen = set()
    for index, block in enumerate(blocks):
        channels = tuple(_indices(block["channel_indices"], A.shape[0], "BLOCK_CHANNELS", nonempty=True).tolist())
        _require(isinstance(block["block_index"], int) and not isinstance(block["block_index"], bool) and
                 block["block_index"] == index and channels in by_channels and channels not in seen,
                 "BLOCK_CLOSURE_OR_INDEX_MISMATCH")
        seen.add(channels)
        component_indices = by_channels[channels]["component_indices"]
        candidates = sorted(set(int(owner[j]) for j in component_indices))
        molecules = sorted({molecular_index[names[j]] for j in candidates})
        _require(block["component_indices"] == component_indices and block["candidate_indices"] == candidates and
                 block["molecular_indices"] == molecules, "BLOCK_MEMBERSHIP_MISMATCH")
        actual_A_candidates = np.flatnonzero(np.any(A[list(channels)] > 0, axis=0)).tolist()
        _require(actual_A_candidates == candidates, "COMPONENT_NOMINAL_SUPPORT_MISMATCH")
    return dict(status="PASS", block_count=len(expected), channel_count=A.shape[0],
                component_count=C.shape[1], candidate_count=A.shape[1], molecular_count=len(molecular_names),
                block_sizes=[len(block["channel_indices"]) for block in blocks])


def review_model_predictions(rows, feature_rows, model, thresholds, saved_scores, *, is_train=False):
    """Replay probabilities with scalar math; only TRAIN checks labels/cutoffs.

    Unreported identities have no classifier probability. Threshold enumeration
    uses validated saved probabilities so exact stored ties remain intact when
    independent scalar arithmetic differs in its last floating-point bits.
    """
    feature_keys = ("predictive_gain", "positive_block_fraction")
    names = [r["lipid_name"] for r in rows]
    _require(len(names) == len(set(names)) and len(rows) == len(saved_scores), "MODEL_ROW_MEMBERSHIP")
    feature_names = [r["molecular_name"] for r in feature_rows]
    _require(len(feature_names) == len(set(feature_names)) and set(feature_names) == set(names), "MODEL_FEATURE_MEMBERSHIP")
    by_name = {r["molecular_name"]: r for r in feature_rows}
    _require(tuple(model["features"]) == feature_keys, "MODEL_FEATURE_ORDER_CHANGED")
    base, auxiliary = model["base"], model["auxiliary"]
    for normalization in (base, auxiliary):
        _require(len(normalization["mean"]) == len(normalization["scale"]) == 2 and
                 all(math.isfinite(float(x)) for x in normalization["mean"]) and
                 all(math.isfinite(float(x)) and x > 0 for x in normalization["scale"]), "INVALID_MODEL_NORMALIZATION")
    coefficients, intercept = model["coef"], model["intercept"]
    _require(len(coefficients) == 8 and all(math.isfinite(float(x)) for x in coefficients) and
             math.isfinite(float(intercept)), "INVALID_MODEL_COEFFICIENTS")
    logs, auxiliaries, reported_names, validated_scores = [], [], [], []
    maximum_error = 0.
    for row, saved in zip(rows, saved_scores):
        name = row["lipid_name"]
        _require(saved["lipid_name"] == name and "new_score" in saved, "SAVED_SCORE_MEMBERSHIP")
        if "raw_solver_reported" in saved:
            _require(saved["raw_solver_reported"] == row["raw_solver_reported"], "SAVED_REPORT_MEMBERSHIP_CHANGED")
        if not row["raw_solver_reported"]:
            _require(saved["new_score"] is None, "UNREPORTED_CLASSIFIER_SCORE")
            validated_scores.append(None)
            continue
        observed = (row["X_hat"], row["rho_zero"])
        _require(all(value is not None and math.isfinite(float(value)) and value >= 0 for value in observed),
                 "INVALID_REPORTED_FEATURE")
        log_values = [math.log10(max(float(observed[0]), 1e-12)), math.log10(max(float(observed[1]), 1e-24))]
        extra = [float(by_name[name][key]) for key in feature_keys]
        _require(all(math.isfinite(value) for value in extra) and 0 <= extra[1] <= 1, "INVALID_AUXILIARY_FEATURE")
        z0, z1 = [(log_values[j] - base["mean"][j]) / base["scale"][j] for j in range(2)]
        phi = [z0, z1, z0 * z0, z0 * z1, z1 * z1, float(observed[1] == 0)]
        phi.extend((extra[j] - auxiliary["mean"][j]) / auxiliary["scale"][j] for j in range(2))
        logit = math.fsum(value * coefficient for value, coefficient in zip(phi, coefficients)) + intercept
        _require(math.isfinite(logit), "NONFINITE_REPLAY_LOGIT")
        probability = 1 / (1 + math.exp(-logit)) if logit >= 0 else math.exp(logit) / (1 + math.exp(logit))
        stored_probability = saved["new_score"]
        _require(stored_probability is not None and math.isfinite(float(stored_probability)) and
                 0 <= stored_probability <= 1, "INVALID_SAVED_PROBABILITY")
        error = abs(probability - stored_probability)
        _require(error <= 1e-12, "MODEL_PROBABILITY_MISMATCH")
        maximum_error = max(maximum_error, error)
        logs.append(log_values)
        auxiliaries.append(extra)
        reported_names.append(name)
        validated_scores.append(float(stored_probability))
    output = dict(status="PASS", is_train=bool(is_train), row_count=len(rows),
                  reported_count=len(reported_names), maxprob_error=maximum_error,
                  unreported_probability="None", no_optimizer_or_training=True)
    if not is_train:
        return output
    _require(reported_names and model["training_names"] == reported_names, "TRAINING_NAME_MEMBERSHIP")
    for label, values, normalization in (("BASE", logs, base), ("AUXILIARY", auxiliaries, auxiliary)):
        means = [math.fsum(value[j] for value in values) / len(values) for j in range(2)]
        std = [math.sqrt(math.fsum((value[j] - means[j]) ** 2 for value in values) / len(values)) for j in range(2)]
        scales = [max(value, 1e-12) for value in std]
        _require(all(math.isclose(means[j], normalization["mean"][j], rel_tol=1e-12, abs_tol=1e-18) and
                     math.isclose(scales[j], normalization["scale"][j], rel_tol=1e-12, abs_tol=1e-18)
                     for j in range(2)), "TRAIN_ONLY_NORMALIZATION_MISMATCH:" + label)
        if label == "AUXILIARY" and "constant_features" in normalization:
            _require(normalization["constant_features"] == [feature_keys[j] for j in range(2) if std[j] < 1e-12],
                     "CONSTANT_FEATURE_MEMBERSHIP")
    truth = {r["lipid_name"] for r in rows if r["molecular_truth"]}
    raw_truth = {r["lipid_name"] for r in rows if r["molecular_truth"] and r["raw_solver_reported"]}
    _require(truth, "EMPTY_TRAIN_TRUTH")
    score_groups = {}
    for row, probability in zip(rows, validated_scores):
        if row["raw_solver_reported"]:
            score_groups.setdefault(probability, []).append(row["lipid_name"])
    retained, pool, final, best = set(), None, None, None
    for cut in sorted(score_groups, reverse=True):
        retained.update(score_groups[cut])
        tp, fp = len(retained & truth), len(retained - truth)
        if pool is None and 20 * tp >= 19 * len(truth):
            pool = cut
        if retained and 100 * fp <= len(retained):
            objective = (tp, -fp, cut)
            if best is None or objective > best:
                best, final = objective, cut
    effective = max(pool, final) if pool is not None and final is not None else None
    for key, expected in (("pool_threshold", pool), ("final_raw_threshold", final),
                          ("final_effective_threshold", effective)):
        _require(key in thresholds and thresholds[key] == expected, "THRESHOLD_REPLAY_MISMATCH:" + key)
    for key, cut in (("train_pool", pool), ("train_final", effective)):
        selected = {row["lipid_name"] for row, probability in zip(rows, validated_scores)
                    if cut is not None and row["raw_solver_reported"] and probability >= cut}
        tp, fp = len(selected & truth), len(selected - truth)
        metrics = dict(TP=tp, FP=fp, FN=len(truth) - tp, FDP=fp / len(selected) if selected else None,
                       all_truth_recall=tp / len(truth), TP_retention=tp / len(raw_truth) if raw_truth else None,
                       all_truth_count=len(truth), raw_solver_misses=len(truth - raw_truth), retained=len(selected))
        _require(key in thresholds and all(field in thresholds[key] and thresholds[key][field] == value
                 for field, value in metrics.items()), "THRESHOLD_ACCOUNTING_MISMATCH:" + key)
    output["thresholds_independently_verified"] = True
    return output
