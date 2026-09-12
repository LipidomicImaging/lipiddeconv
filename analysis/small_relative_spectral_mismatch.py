"""Positive, independent physical-ion intensity variation before column scaling.

``components[:, p]`` is a complete physical-ion envelope owned by one candidate.
Each metadata row has candidate_index, kind (fragment/precursor), and a
physical_ion_key dict with exactly identity/adduct/CE/ion. Identity, adduct and
ion are nonempty strings; CE is a finite nonnegative number describing the fixed
condition, not a reference to any CE data. An envelope's isotope/profile channels
share one multiplier. Identical keys across alias candidates share that factor.
All components of a candidate must agree on identity/adduct/CE, and the entire
target library must describe a single fixed CE condition.

The stated mean-one multiplier CV applies BEFORE the common per-column L2
normalization. That normalization couples final within-column intensities.
There is no file I/O, abundance rescaling, observation input or solver access.
"""
from __future__ import annotations

import hashlib
import json
import math
from numbers import Integral, Real

import numpy as np


NAMESPACE = "SMALL_RELATIVE_SPECTRAL_MISMATCH_V1"


def _require(ok, message):
    if not ok:
        raise ValueError(message)


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _key(value):
    _require(isinstance(value, dict) and set(value) == {"identity", "adduct", "CE", "ion"},
             "physical_ion_key requires exactly identity/adduct/CE/ion")
    _require(all(isinstance(value[k], str) and value[k].strip() for k in ("identity", "adduct", "ion")),
             "identity/adduct/ion must be nonempty strings")
    ce = value["CE"]
    _require(isinstance(ce, Real) and not isinstance(ce, (bool, np.bool_)) and math.isfinite(float(ce)) and ce >= 0,
             "CE must be a finite nonnegative fixed-condition number")
    return _canonical({**value, "CE": float(ce)})


def _sha(array):
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def build_target_library(A, components, component_metadata, *, seed, relative_sd=0.05):
    """Return (A_target, JSON-serializable audit) without changing any input.

    A and components must be nonnegative float32/float64 matrices. Components
    must reconstruct EVERY nominal column, including residual precursor signal.
    Reconstruction is checked per channel at eight input machine epsilons, with
    identical positive support (no global absolute tolerance hides small peaks).
    Seeds are derived from namespace/experiment seed/key, never candidate order.
    """
    for name, value in (("A", A), ("components", components)):
        _require(isinstance(value, np.ndarray) and value.ndim == 2 and value.dtype in (np.dtype("float32"), np.dtype("float64")),
                 name + " must be a float32/float64 matrix")
        _require(value.size > 0 and np.isfinite(value).all() and (value >= 0).all(), name + " must be finite nonnegative and nonempty")
    _require(components.shape[0] == A.shape[0] and isinstance(component_metadata, (list, tuple))
             and len(component_metadata) == components.shape[1], "component shape/metadata mismatch")
    _require(isinstance(seed, Integral) and not isinstance(seed, (bool, np.bool_)), "seed must be an integer")
    _require(isinstance(relative_sd, Real) and not isinstance(relative_sd, (bool, np.bool_))
             and math.isfinite(float(relative_sd)) and relative_sd >= 0, "relative_sd must be finite and nonnegative")
    c = float(relative_sd)
    try:
        tau = math.sqrt(math.log1p(c * c))
    except (OverflowError, ValueError):
        raise ValueError("relative_sd outside finite numerical range") from None
    _require(math.isfinite(tau), "relative_sd outside finite numerical range")
    a, envelopes = A.astype(np.float64), components.astype(np.float64)
    _require((np.max(a, axis=0) > 0).all() and (np.max(envelopes, axis=0) > 0).all(), "zero candidate column or component")
    rows, unique, kinds, candidate_context, ce_values = [], set(), {}, {}, set()
    for index, item in enumerate(component_metadata):
        _require(isinstance(item, dict) and {"candidate_index", "physical_ion_key", "kind"} <= set(item), "component metadata fields missing")
        owner = item["candidate_index"]
        _require(isinstance(owner, Integral) and not isinstance(owner, (bool, np.bool_)) and 0 <= owner < A.shape[1], "invalid component owner")
        _require(item["kind"] in ("fragment", "precursor"), "unsupported physical-ion kind")
        key = _key(item["physical_ion_key"])
        decoded = json.loads(key)
        context = (decoded["identity"], decoded["adduct"], decoded["CE"])
        _require(int(owner) not in candidate_context or candidate_context[int(owner)] == context,
                 "candidate components disagree on identity/adduct/CE")
        candidate_context[int(owner)] = context
        ce_values.add(decoded["CE"])
        _require((int(owner), key) not in unique, "duplicate physical-ion key within candidate")
        _require(key not in kinds or kinds[key] == item["kind"], "alias key has inconsistent kind")
        unique.add((int(owner), key)); kinds[key] = item["kind"]
        rows.append((int(owner), key, index, item["kind"]))
    _require(len(ce_values) == 1, "target library must have one fixed CE condition")
    rows.sort(key=lambda row: (row[0], row[1]))  # Stable summation under component reordering.
    reconstructed = np.zeros_like(a)
    for owner, _, index, _ in rows:
        reconstructed[:, owner] += envelopes[:, index]
    epsilon = max(np.finfo(A.dtype).eps, np.finfo(components.dtype).eps)
    error, scale = np.abs(reconstructed - a), np.maximum(reconstructed, a)
    _require(np.array_equal(reconstructed > 0, a > 0) and np.all(error <= 8 * epsilon * scale), "incomplete or inconsistent full component decomposition")
    factors, standardized_draws = {}, {}
    for key in sorted(kinds):
        payload = _canonical(dict(namespace=NAMESPACE, experiment_seed=int(seed), physical_ion_key=json.loads(key)))
        key_seed = int.from_bytes(hashlib.sha256(payload.encode()).digest(), "big")
        z = float(np.random.Generator(np.random.PCG64(key_seed)).standard_normal())
        factor = math.exp(tau * z - .5 * tau * tau)
        _require(math.isfinite(factor) and factor > 0, "nonfinite or underflowed sampled multiplier")
        factors[key], standardized_draws[key] = factor, z
    perturbed = np.zeros_like(a)
    for owner, key, index, _ in rows:
        perturbed[:, owner] += factors[key] * envelopes[:, index]
    nominal_norm, target_norm = np.linalg.norm(a, axis=0), np.linalg.norm(perturbed, axis=0)
    _require(np.isfinite(nominal_norm).all() and np.isfinite(target_norm).all() and (target_norm > 0).all(), "invalid spectral column norm")
    normalization = nominal_norm / target_norm
    target = (perturbed * normalization).astype(A.dtype) if c else A.copy()
    if not c:
        normalization = np.ones(A.shape[1], dtype=np.float64)
    _require(np.isfinite(target).all() and (target >= 0).all() and np.array_equal(target > 0, A > 0), "target support changed or nonfinite")
    actual_norm = np.linalg.norm(target.astype(np.float64), axis=0)
    norm_error = np.abs(actual_norm / nominal_norm - 1.)
    _require((norm_error <= 8 * np.finfo(A.dtype).eps).all(), "column norm preservation failed")
    audit = dict(namespace=NAMESPACE, experiment_seed=int(seed), fixed_CE=next(iter(ce_values)),
                 candidate_identity_adduct_CE_consistent=True, relative_sd_before_normalization=c, lognormal_tau=tau,
                 multiplier_mean_before_normalization=1., multiplier_CV_before_normalization=c,
                 independence_unit="Distinct physical_ion_key; identical-key aliases share one draw",
                 sampling="PCG64 seeded by SHA256(canonical(namespace, experiment_seed, physical_ion_key)); one standard normal per key",
                 fixed_over_pixels=True, isotope_profile_envelope_unchanged=True, dropout=False, new_support=False,
                 column_normalization="One common positive scalar per candidate; preserves original L2 norm and couples final intensities",
                 no_X_rescaling=True, no_observation_or_truth_input=True,
                 source_sha256=_sha(A), target_sha256=_sha(target), components_sha256=_sha(components),
                 source_shape=list(A.shape), source_dtype=str(A.dtype), component_count=len(rows),
                 candidate_coverage_count=len({r[0] for r in rows}), distinct_physical_ion_count=len(factors),
                 fragment_component_count=sum(r[3] == "fragment" for r in rows), precursor_component_count=sum(r[3] == "precursor" for r in rows),
                 decomposition_max_absolute_error=float(error.max()), decomposition_relative_tolerance=float(8 * epsilon),
                 normalization_scales=normalization.tolist(), column_norm_relative_errors=norm_error.tolist(),
                 relative_column_changes=(np.linalg.norm(target.astype(np.float64) - a, axis=0) / nominal_norm).tolist(),
                 sampled_factors=[dict(candidate_index=owner, physical_ion_key=json.loads(key), kind=kind,
                                       standard_normal_draw=standardized_draws[key], sampled_multiplier=factors[key],
                                       effective_multiplier_after_normalization=float(factors[key] * normalization[owner]))
                                  for owner, key, _, kind in rows])
    audit["fingerprint"] = hashlib.sha256(_canonical(audit).encode()).hexdigest()
    return target, audit
