"""Observable spatial donor descriptors; no labels or spectral fitting.

The input contains *foreground pixels only*. All aliases of each molecular name
are summed, and every other name with a strictly larger mean is a possible donor.
U1/U2 are the relative squared residuals of the best nonnegative projection onto
at most one/two donor maps. They are descriptive features, not confidence bounds.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Sequence

import numpy as np


FEATURE_NAMES = ("spatial_donor_log10_u1", "spatial_donor_log10_u2")
FEATURE_FLOOR = 1e-12
# This switches numerical algorithms only; it does not select molecular evidence.
NEAR_COLLINEAR_GAP = 1e-8
DEGENERATE_DIRECTION_NORM = 64 * np.finfo(np.float64).eps
MAX_CACHED_PIXEL_DIRECTIONS = 64


def compute_spatial_donor_confidence(
    x_hat: np.ndarray, candidate_names: Sequence[str]
) -> dict:
    """Return JSON-serializable molecular records, features and numerical audit.

    ``x_hat`` is candidate by foreground-pixel (additional spatial axes are
    flattened). Names retain first-occurrence order. Zero maps have undefined
    relative residuals: U1=U2=1 is a finite placeholder, explicitly flagged by
    ``spatial_nonzero=False``; they must not be treated as evidence of uniqueness.

    Well-conditioned pairs use the exact two-variable NNLS active-set formula
    on a unit Gram matrix, including all single-donor boundaries. Near-collinear
    pairs use a twice-orthogonalized donor difference in the original pixel
    vectors. Numerically identical pairs retain the single-donor boundary. For
    exact unit vectors with cosine c>=0, the improvement obtainable by any
    nonnegative pair over its best single donor is at most (1-c)/2, equivalently
    ||u-v||^2/4. That pair-specific bound is recorded when skipping a numerically
    degenerate direction. The ordinary floating-point error diagnostic is a
    conservative roundoff estimate, not an interval-arithmetic certificate.
    """
    x = np.asarray(x_hat, dtype=np.float64)
    names = list(candidate_names)
    if x.ndim < 2 or x.shape[0] != len(names) or not len(names):
        raise ValueError("x_hat must have one nonempty row per candidate name")
    if any(not isinstance(name, str) or not name for name in names):
        raise ValueError("candidate_names must contain nonempty strings")
    x = x.reshape(len(names), -1)
    if x.shape[1] == 0 or not np.isfinite(x).all() or (x < 0).any():
        raise ValueError("foreground values must be finite, nonnegative and nonempty")

    molecular_names = list(dict.fromkeys(names))
    lookup = {name: i for i, name in enumerate(molecular_names)}
    alias_indices = [[] for _ in molecular_names]
    for candidate, name in enumerate(names):
        alias_indices[lookup[name]].append(candidate)
    # A common scale avoids overflow/underflow in the spatial Gram matrix and
    # preserves strict donor ordering. It is not an abundance/reporting gate.
    scale = float(x.max()) or 1.0
    spatial = np.zeros((len(molecular_names), x.shape[1]), dtype=np.float64)
    for group, indices in enumerate(alias_indices):
        spatial[group] = np.sum(x[indices] / scale, axis=0)
    means_scaled = spatial.mean(axis=1)
    norms_scaled = np.linalg.norm(spatial, axis=1)
    with np.errstate(over="ignore", invalid="ignore"):
        means = means_scaled * scale
        norms = norms_scaled * scale
    if not np.isfinite(means).all() or not np.isfinite(norms).all():
        raise ValueError("aggregated means/norms exceed float64 representation")
    nonzero = norms_scaled > 0
    unit = np.zeros_like(spatial)
    unit[nonzero] = spatial[nonzero] / norms_scaled[nonzero, None]
    gram_raw = unit @ unit.T
    gram = np.clip(gram_raw, 0.0, 1.0)
    diagonal_error = float(np.max(np.abs(np.diag(gram_raw)[nonzero] - 1))) if nonzero.any() else 0.0
    eps = np.finfo(np.float64).eps
    gamma_p = x.shape[1] * eps / max(1 - x.shape[1] * eps, eps)
    roundoff_estimate = float(32 * (gamma_p + diagonal_error + eps))
    basis_cache = {}
    # Keep scalar diagnostics for all encountered pairs, but at most 64 full
    # pixel directions. Near-collinearity must not produce a G^2-by-P cache.
    direction_cache = OrderedDict()
    records = []
    maximum_gram_direct_discrepancy = 0.0

    def direct_residual(target, indices, coefficients):
        residual = unit[target].copy()
        for donor, coefficient in zip(indices, coefficients):
            residual -= coefficient * unit[donor]
        return float(np.dot(residual, residual) / gram_raw[target, target])

    def stable_basis(h, k):
        key = (int(h), int(k))
        if key in direction_cache:
            direction_cache.move_to_end(key)
            return (*basis_cache[key][:2], direction_cache[key], basis_cache[key][2])
        if key not in basis_cache or basis_cache[key][1] > DEGENERATE_DIRECTION_NORM:
            u, v = unit[h], unit[k]
            difference = v - u
            projection = float(np.dot(u, difference) / np.dot(u, u))
            orthogonal = difference - projection * u
            correction = float(np.dot(u, orthogonal) / np.dot(u, u))
            orthogonal -= correction * u
            projection += correction
            norm = float(np.linalg.norm(orthogonal))
            # Stable difference-based bound, also when Gram cosine rounds to 1.
            improvement_bound = float(np.dot(difference, difference) / 4)
            direction = orthogonal / norm if norm > DEGENERATE_DIRECTION_NORM else None
            basis_cache[key] = (1 + projection, norm, improvement_bound)
            if direction is not None:
                direction_cache[key] = direction
                if len(direction_cache) > MAX_CACHED_PIXEL_DIRECTIONS:
                    direction_cache.popitem(last=False)
            return 1 + projection, norm, direction, improvement_bound
        parallel, norm, improvement_bound = basis_cache[key]
        return parallel, norm, None, improvement_bound

    for g, name in enumerate(molecular_names):
        donors = np.flatnonzero(means_scaled > means_scaled[g])
        donors = donors[donors != g]
        best_single, best_pair = [], []
        best_single_coefficients, best_pair_coefficients = [], []
        u1 = u2 = 1.0
        near_count = degenerate_count = 0
        degenerate_bound = 0.0
        pair_count = int(len(donors) * (len(donors) - 1) // 2)
        if nonzero[g] and len(donors):
            a = gram[g, donors]
            single_position = int(np.argmax(a))
            h = int(donors[single_position])
            coefficient = float(gram_raw[g, h] / gram_raw[h, h])
            u1 = direct_residual(g, [h], [coefficient])
            maximum_gram_direct_discrepancy = max(
                maximum_gram_direct_discrepancy, abs(u1 - (1 - a[single_position] ** 2))
            )
            best_single, best_single_coefficients = [h], [coefficient]
            u2, best_pair, best_pair_coefficients = u1, [h], [coefficient]
            left, right = np.triu_indices(len(donors), 1)
            c = gram[donors[left], donors[right]]
            near = (1 - c) <= NEAR_COLLINEAR_GAP
            near_count = int(near.sum())
            ordinary = ~near
            if ordinary.any():
                il, ir, cc = left[ordinary], right[ordinary], c[ordinary]
                aa, bb = a[il], a[ir]
                numerator_left, numerator_right = aa - cc * bb, bb - cc * aa
                interior = (numerator_left > 0) & (numerator_right > 0)
                if interior.any():
                    il, ir, cc, aa, bb = [z[interior] for z in (il, ir, cc, aa, bb)]
                    denominator = (1 - cc) * (1 + cc)
                    alphas = (aa - cc * bb) / denominator
                    betas = (bb - cc * aa) / denominator
                    # Sum/difference coordinates avoid subtracting two large
                    # nearly equal quadratic forms when evaluating the fit.
                    predicted = 1 - ((aa + bb) ** 2 / (2 * (1 + cc)) +
                                     (aa - bb) ** 2 / (2 * (1 - cc)))
                    winner = int(np.argmin(predicted))
                    pair = [int(donors[il[winner]]), int(donors[ir[winner]])]
                    coefficients = [float(alphas[winner]), float(betas[winner])]
                    value = direct_residual(g, pair, coefficients)
                    maximum_gram_direct_discrepancy = max(
                        maximum_gram_direct_discrepancy, abs(value - float(predicted[winner]))
                    )
                    if value < u2:
                        u2, best_pair, best_pair_coefficients = value, pair, coefficients
            for i, j in zip(left[near], right[near]):
                h, k = int(donors[i]), int(donors[j])
                parallel, orthogonal_norm, direction, bound = stable_basis(h, k)
                if direction is None:
                    degenerate_count += 1
                    degenerate_bound = max(degenerate_bound, bound)
                    continue
                beta = float(np.dot(unit[g], direction) / orthogonal_norm)
                alpha = float(gram_raw[g, h] / gram_raw[h, h] - beta * parallel)
                if alpha > 0 and beta > 0:
                    value = direct_residual(g, [h, k], [alpha, beta])
                    if value < u2:
                        u2, best_pair, best_pair_coefficients = value, [h, k], [alpha, beta]

        u1, u2 = float(np.clip(u1, 0, 1)), float(np.clip(min(u2, u1), 0, 1))
        record = {
            "molecular_name": name, "molecular_index": g,
            "candidate_indices": alias_indices[g], "spatial_nonzero": bool(nonzero[g]),
            "mean": float(means[g]), "spatial_l2_norm": float(norms[g]),
            "stronger_donor_count": int(len(donors)), "donor_pair_count": pair_count,
            "u1": u1, "u2": u2,
            FEATURE_NAMES[0]: float(np.log10(max(u1, FEATURE_FLOOR))),
            FEATURE_NAMES[1]: float(np.log10(max(u2, FEATURE_FLOOR))),
            "best_u1_donor_indices": best_single,
            "best_u1_donor_names": [molecular_names[i] for i in best_single],
            "best_u2_donor_indices": best_pair,
            "best_u2_donor_names": [molecular_names[i] for i in best_pair],
            "best_u1_unit_coefficients": best_single_coefficients,
            "best_u2_unit_coefficients": best_pair_coefficients,
            "near_collinear_pair_count": near_count,
            "degenerate_pair_count": degenerate_count,
            "maximum_degenerate_pair_improvement_bound": degenerate_bound,
        }
        records.append(record)

    return {
        "schema_version": 1, "feature_names": list(FEATURE_NAMES), "records": records,
        "numerical_diagnostics": {
            "candidate_count": len(names), "molecular_count": len(molecular_names),
            "foreground_pixel_count": int(x.shape[1]), "feature_floor": FEATURE_FLOOR,
            "near_collinear_gap": NEAR_COLLINEAR_GAP,
            "degenerate_direction_norm": DEGENERATE_DIRECTION_NORM,
            "maximum_cached_pixel_directions": MAX_CACHED_PIXEL_DIRECTIONS,
            "gram_diagonal_max_abs_error": diagonal_error,
            "gram_clipping_max_abs_change": float(np.max(np.abs(gram - gram_raw))),
            "roundoff_error_estimate_not_certificate": roundoff_estimate,
            "maximum_gram_direct_residual_discrepancy": maximum_gram_direct_discrepancy,
            "near_collinear_distinct_pair_count": len(basis_cache),
            "total_near_collinear_pair_contexts": sum(r["near_collinear_pair_count"] for r in records),
            "total_degenerate_pair_contexts": sum(r["degenerate_pair_count"] for r in records),
            "maximum_degenerate_pair_improvement_bound": max(
                r["maximum_degenerate_pair_improvement_bound"] for r in records
            ),
            "donor_names_and_indices_are_audit_only": True,
            "zero_map_policy": "U1=U2=1 finite placeholders; spatial_nonzero=False",
        },
    }
