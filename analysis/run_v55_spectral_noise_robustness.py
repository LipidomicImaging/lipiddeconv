#!/usr/bin/env python3
"""Run the v55 spectral-library mismatch and empirical-noise robustness study."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
from collections import defaultdict
from datetime import date, datetime, timezone
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
ANALYSIS = ROOT / "analysis"
for import_root in (ROOT, ANALYSIS):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

import run_v53_stage0_empirical_domain as stage0
import run_v54_complexity_calibration as v54


VERSION = "v55_spectral_noise_robustness"
OUTPUT_DEFAULT = ROOT / "results/v55_spectral_noise_robustness"
V54_OUTPUT = ROOT / "results/v54_complexity_calibration"
EXPERIMENT_LOG = ROOT / "results/EXPERIMENT_LOG.md"
REPORT_GATE = v54.REPORT_GATE
PARENT_MZ_RANGE = (748.0, 803.0)
CONDITION_ORDER = (
    "NOISE_MILD",
    "MISMATCH_MILD",
    "COMBINED_MILD",
    "COMBINED_MODERATE",
)
SUMMARY_CONDITION_ORDER = ("CLEAN", *CONDITION_ORDER)
LOG_BEGIN = "<!-- BEGIN v55_spectral_noise_robustness -->"
LOG_END = "<!-- END v55_spectral_noise_robustness -->"
SEPARATION_EPS = 1.0e-12


def _frozen_condition(**values: float | int) -> MappingProxyType:
    return MappingProxyType(values)


ERROR_CONDITIONS = MappingProxyType({
    "NOISE_MILD": _frozen_condition(
        noise_eta=0.05, fragment_log_sigma=0.0,
        fragment_dropout_rate=0.0, parent_scale_min=1.0,
        parent_scale_max=1.0, seed=5501,
    ),
    "MISMATCH_MILD": _frozen_condition(
        noise_eta=0.0, fragment_log_sigma=0.10,
        fragment_dropout_rate=0.05, parent_scale_min=0.9,
        parent_scale_max=1.1, seed=5502,
    ),
    "COMBINED_MILD": _frozen_condition(
        noise_eta=0.05, fragment_log_sigma=0.10,
        fragment_dropout_rate=0.05, parent_scale_min=0.9,
        parent_scale_max=1.1, seed=5503,
    ),
    "COMBINED_MODERATE": _frozen_condition(
        noise_eta=0.10, fragment_log_sigma=0.20,
        fragment_dropout_rate=0.10, parent_scale_min=0.8,
        parent_scale_max=1.2, seed=5504,
    ),
})
BASE_DATASETS = tuple(v54.DATASETS)
V55_DATASETS = tuple(
    f"V55__{condition}__{base_dataset_id}"
    for condition in CONDITION_ORDER
    for base_dataset_id in BASE_DATASETS
)


def conditions_payload() -> dict[str, dict[str, float | int]]:
    return {name: dict(ERROR_CONDITIONS[name]) for name in CONDITION_ORDER}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_uint32(condition_seed: int, candidate_or_scope: str | int, kind: str) -> int:
    text = f"{VERSION}|{condition_seed}|{candidate_or_scope}|{kind}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(text).digest()[:4], "little", signed=False)


def array_digest(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    return hashlib.sha256(contiguous.view(np.uint8)).hexdigest()


def json_cell(value: Any) -> str:
    return json.dumps(stage0.to_jsonable(value), sort_keys=True, ensure_ascii=False, allow_nan=False)


def _metadata_value(metadata: dict, key: str, index: int) -> str:
    return str(np.asarray(metadata[key]).reshape(-1)[index])


def load_observed_production_B(paths: dict[str, Path], expected_shape: tuple[int, ...]) -> np.ndarray:
    """Load the exact locked Stage0 production observation, never a reconstruction."""
    stored = np.load(paths["B_cube"])
    if stored.ndim != 4 or stored.shape[0] != 1:
        raise RuntimeError(f"INVALID_LOCKED_B_SHAPE: stored={stored.shape}")
    observed = np.moveaxis(np.asarray(stored[0], dtype=np.float32), -1, 0)
    if observed.shape != expected_shape:
        raise RuntimeError(
            f"INVALID_LOCKED_B_SHAPE: observed={observed.shape}, expected={expected_shape}"
        )
    if not np.isfinite(observed).all():
        raise RuntimeError("INVALID_LOCKED_B: non-finite values")
    return observed


def validate_mapping_contract(context: dict) -> list[dict]:
    rows = [
        {key: v54.parse_scalar(value) for key, value in row.items()}
        for row in v54.read_csv(V54_OUTPUT / "template_mappings.csv")
    ]
    if {str(row["dataset_id"]) for row in rows} != set(BASE_DATASETS):
        raise RuntimeError("V54_MAPPING_DATASET_SET_CHANGED")
    v54.validate_mapping_nesting(rows)
    cal_names = {
        str(row["synthetic_truth_lipid_name"])
        for row in rows if str(row["split"]) == "CAL"
    }
    hold_names = {
        str(row["synthetic_truth_lipid_name"])
        for row in rows if str(row["split"]) == "HOLD"
    }
    overlap = cal_names & hold_names
    if overlap:
        raise RuntimeError(f"CAL_HOLD_LIPID_NAME_OVERLAP: {sorted(overlap)[:5]}")
    return rows


def load_context(asset_root: Path) -> dict:
    context = v54.load_context(asset_root)
    v54_dependencies = [
        V54_OUTPUT / "design.json",
        V54_OUTPUT / "template_mappings.csv",
        V54_OUTPUT / "dataset_manifest.csv",
    ]
    missing_v54 = [str(path) for path in v54_dependencies if not path.exists()]
    context["missing_v54_design"] = missing_v54
    if context.get("missing_assets") or context.get("missing_results") or missing_v54:
        return context

    design = v54.validate_frozen_design(V54_OUTPUT, context)
    mappings = validate_mapping_contract(context)
    normalized_again = v54.v52.normalize_like_get_A_matrix(context["A_raw"])
    if not np.array_equal(normalized_again, context["A_solver"]):
        raise RuntimeError("A_SOLVER_RENORMALIZATION_MISMATCH")
    mz = np.asarray(np.load(context["paths"]["channel_axis"]), dtype=np.float64).reshape(-1)
    if mz.size != context["A_solver"].shape[0]:
        raise RuntimeError("CHANNEL_AXIS_SIZE_MISMATCH")
    parent_mask = (mz >= PARENT_MZ_RANGE[0]) & (mz <= PARENT_MZ_RANGE[1])
    if not parent_mask.any() or parent_mask.all():
        raise RuntimeError("INVALID_PARENT_FRAGMENT_PARTITION")
    expected_B_shape = (context["A_solver"].shape[0], *context["mask"].shape)
    observed = load_observed_production_B(context["paths"], expected_B_shape)
    context.update({
        "v54_design": design,
        "v54_mappings": mappings,
        "v54_mapping_sha256": sha256_file(V54_OUTPUT / "template_mappings.csv"),
        "mz": mz,
        "parent_mask": parent_mask,
        "fragment_mask": ~parent_mask,
        "B_real": observed,
    })
    return context


def v54_case(context: dict, base_dataset_id: str) -> dict:
    mapping = v54.dataset_mapping(V54_OUTPUT, base_dataset_id)
    case = v54.construct_dataset(context, mapping, base_dataset_id)
    frozen_x = case["X_true"]
    if frozen_x.dtype != np.float32 or not np.isfinite(frozen_x).all():
        raise RuntimeError(f"INVALID_V54_X_TRUE: {base_dataset_id}")
    clean_check = np.einsum(
        "mc,cyx->myx", context["A_solver"], frozen_x, optimize=True
    ).astype(np.float32, copy=False)
    if not np.array_equal(frozen_x, case["X_true"]):
        raise RuntimeError(f"V54_X_TRUE_NOT_EXACT: {base_dataset_id}")
    if not np.allclose(clean_check, case["B_sim"], rtol=2e-6, atol=1e-7):
        raise RuntimeError(f"V54_CLEAN_FORWARD_MISMATCH: {base_dataset_id}")

    expected = context["v54_design"].get("dataset_truth_complexity", {}).get(base_dataset_id)
    if expected:
        checks = {
            "global_truth_K": case["global_truth_K"],
            "reportable_truth_global_count": case["reportable_truth_molecular_identity_count"],
            "subthreshold_truth_global_count": case["subthreshold_truth_molecular_identity_count"],
            "K_true_pixel_foreground": case["K_true_pixel_foreground"],
            "global_signal_scalar": case["global_scale"],
        }
        for key, value in checks.items():
            if expected.get(key) != value:
                raise RuntimeError(f"V54_FROZEN_TRUTH_METADATA_CHANGED: {base_dataset_id}:{key}")
    report_path = V54_OUTPUT / base_dataset_id / "report.json"
    if report_path.exists():
        report = v54.read_json(report_path)
        report_checks = {
            "global_truth_K": case["global_truth_K"],
            "reportable_truth_global_count": case["reportable_truth_molecular_identity_count"],
            "subthreshold_truth_global_count": case["subthreshold_truth_molecular_identity_count"],
            "true_per_pixel_K_foreground": case["true_per_pixel_K_foreground"],
            "global_signal_scalar": case["global_scale"],
        }
        for key, value in report_checks.items():
            if key in report and report[key] != value:
                raise RuntimeError(f"V54_RUNTIME_TRUTH_METADATA_CHANGED: {base_dataset_id}:{key}")
    return case


def cosine(left: np.ndarray, right: np.ndarray) -> float | None:
    left64 = np.asarray(left, dtype=np.float64)
    right64 = np.asarray(right, dtype=np.float64)
    denominator = float(np.linalg.norm(left64) * np.linalg.norm(right64))
    if denominator == 0:
        return 1.0 if np.array_equal(left64, right64) else None
    return float(np.clip(np.dot(left64, right64) / denominator, -1.0, 1.0))


def perturb_candidate(
    context: dict, condition_name: str, candidate_index: int
) -> tuple[np.ndarray, dict]:
    cfg = ERROR_CONDITIONS[condition_name]
    original = np.asarray(context["A_solver"][:, candidate_index], dtype=np.float32)
    result = original.copy()
    parent = context["parent_mask"]
    fragment = context["fragment_mask"]
    fragment_nonzero = np.flatnonzero(fragment & (original != 0))
    distortion_seed = stable_uint32(int(cfg["seed"]), candidate_index, "fragment_distortion")
    dropout_seed = stable_uint32(int(cfg["seed"]), candidate_index, "fragment_dropout")
    parent_seed = stable_uint32(int(cfg["seed"]), candidate_index, "parent_multiplier")
    bundle_seed = stable_uint32(int(cfg["seed"]), candidate_index, "spectral_bundle")

    sigma = float(cfg["fragment_log_sigma"])
    dropout_rate = float(cfg["fragment_dropout_rate"])
    parent_min = float(cfg["parent_scale_min"])
    parent_max = float(cfg["parent_scale_max"])
    parent_multiplier = float(
        np.random.default_rng(parent_seed).uniform(parent_min, parent_max)
    )
    dropped_local = np.zeros(fragment_nonzero.size, dtype=bool)

    mismatch_active = sigma > 0 or dropout_rate > 0 or parent_min != 1 or parent_max != 1
    if mismatch_active:
        if fragment_nonzero.size:
            log_multipliers = np.random.default_rng(distortion_seed).normal(
                0.0, sigma, size=fragment_nonzero.size
            )
            distorted = (
                original[fragment_nonzero].astype(np.float64)
                * np.exp(log_multipliers)
            )
            result[fragment_nonzero] = distorted.astype(np.float32)
            if dropout_rate > 0:
                dropped_local = (
                    np.random.default_rng(dropout_seed).random(fragment_nonzero.size)
                    < dropout_rate
                )
                if dropped_local.all():
                    keep = int(np.random.default_rng(dropout_seed).integers(fragment_nonzero.size))
                    dropped_local[keep] = False
                result[fragment_nonzero[dropped_local]] = np.float32(0.0)
        result[parent] *= np.float32(parent_multiplier)
        before_norm = float(np.linalg.norm(result.astype(np.float32)))
        if not np.isfinite(before_norm) or before_norm <= 0:
            raise RuntimeError(f"PERTURBED_ZERO_COLUMN: {condition_name}:{candidate_index}")
        # Match production: cast to float32, then normalize with norm + 1e-8.
        result = v54.v52.normalize_like_get_A_matrix(result[:, None])[:, 0]
    else:
        before_norm = float(np.linalg.norm(result.astype(np.float64)))

    if np.any((original[parent] == 0) != (result[parent] == 0)):
        raise RuntimeError(f"PARENT_DROPOUT_OR_SUPPORT_CHANGE: {condition_name}:{candidate_index}")
    after_norm = float(np.linalg.norm(result.astype(np.float64)))
    if not np.isfinite(after_norm) or after_norm <= 0:
        raise RuntimeError(f"PERTURBED_ZERO_COLUMN: {condition_name}:{candidate_index}")
    if mismatch_active and not np.isclose(after_norm, 1.0, rtol=2e-6, atol=2e-7):
        raise RuntimeError(f"PERTURBED_COLUMN_NOT_L2_NORMALIZED: {condition_name}:{candidate_index}")
    dropped = int(dropped_local.sum())
    manifest = {
        "error_condition": condition_name,
        "candidate_index": candidate_index,
        "candidate_id": _metadata_value(context["metadata"], "candidate_id", candidate_index),
        "lipid_name": _metadata_value(context["metadata"], "lipid_name", candidate_index),
        "deterministic_seed": bundle_seed,
        "fragment_distortion_seed": distortion_seed,
        "fragment_dropout_seed": dropout_seed,
        "parent_multiplier_seed": parent_seed,
        "original_nonzero_fragment_count": int(fragment_nonzero.size),
        "dropped_fragment_count": dropped,
        "requested_dropout_fraction": dropout_rate,
        "achieved_dropout_fraction": v54.safe_ratio(dropped, fragment_nonzero.size),
        "fragment_log_sigma": sigma,
        "parent_multiplier": parent_multiplier,
        "cosine_full_Aperturbed_vs_Asolver": cosine(result, original),
        "cosine_parent_Aperturbed_vs_Asolver": cosine(result[parent], original[parent]),
        "cosine_fragment_Aperturbed_vs_Asolver": cosine(result[fragment], original[fragment]),
        "original_column_l2": float(np.linalg.norm(original.astype(np.float64))),
        "perturbed_column_l2_before_normalization": before_norm,
        "perturbed_column_l2_after_normalization": after_norm,
    }
    return result, manifest


def build_perturbed_library(
    context: dict, condition_name: str, active_indices: list[int]
) -> tuple[np.ndarray, list[dict]]:
    solver = context["A_solver"]
    perturbed = solver.copy()
    manifests = []
    active = sorted(set(map(int, active_indices)))
    for candidate_index in active:
        column, row = perturb_candidate(context, condition_name, candidate_index)
        perturbed[:, candidate_index] = column
        manifests.append(row)
    inactive = np.ones(solver.shape[1], dtype=bool)
    inactive[active] = False
    if not np.array_equal(perturbed[:, inactive], solver[:, inactive]):
        raise RuntimeError("A_PERTURBED_CHANGED_NONTRUTH_COLUMN")
    changed = set(np.flatnonzero(np.any(perturbed != solver, axis=0)).astype(int).tolist())
    if not changed.issubset(set(active)):
        raise RuntimeError("A_PERTURBED_CHANGE_OUTSIDE_ACTIVE_TRUTH")
    return perturbed, manifests


def empirical_background(context: dict) -> dict:
    background_mask = ~np.asarray(context["mask"], dtype=bool)
    background_count = int(background_mask.sum())
    if background_count <= 0:
        raise RuntimeError("NO_BACKGROUND_PIXELS_FOR_EMPIRICAL_NOISE")
    samples = context["B_real"][:, background_mask].astype(np.float32, copy=False)
    center = np.median(samples, axis=1).astype(np.float32)
    residual_spectra = (samples - center[:, None]).astype(np.float32, copy=False)
    if residual_spectra.shape != (context["A_solver"].shape[0], background_count):
        raise RuntimeError("EMPIRICAL_BACKGROUND_SHAPE_MISMATCH")
    if not np.isfinite(residual_spectra).all():
        raise RuntimeError("EMPIRICAL_BACKGROUND_NONFINITE")
    return {
        "center": center,
        "residual_spectra": residual_spectra,
        "background_pixel_count": background_count,
        "center_summary": v54.numeric_summary(center),
    }


def noise_seed(condition_name: str, split: str, replicate: str) -> int:
    cfg = ERROR_CONDITIONS[condition_name]
    return stable_uint32(int(cfg["seed"]), f"{split}|{replicate}", "whole_spectrum_bootstrap")


def sampled_noise_field(
    context: dict, background: dict, condition_name: str, split: str, replicate: str
) -> tuple[np.ndarray, dict]:
    seed = noise_seed(condition_name, split, replicate)
    pixel_count = int(np.prod(context["mask"].shape))
    sampled_indices = np.random.default_rng(seed).integers(
        0, background["background_pixel_count"], size=pixel_count, dtype=np.int64
    )
    field = background["residual_spectra"][:, sampled_indices].reshape(
        context["A_solver"].shape[0], *context["mask"].shape
    ).astype(np.float32, copy=False)
    return field, {
        "deterministic_noise_seed": seed,
        "sampled_background_index_sha256": array_digest(sampled_indices),
    }


def add_empirical_noise(
    signal: np.ndarray, raw_noise: np.ndarray, foreground_mask: np.ndarray,
    target_eta: float,
) -> tuple[np.ndarray, dict]:
    signal_median = float(np.median(np.linalg.norm(signal[:, foreground_mask], axis=0)))
    unscaled_noise_median = float(
        np.median(np.linalg.norm(raw_noise[:, foreground_mask], axis=0))
    )
    if not np.isfinite(signal_median) or signal_median <= 0:
        raise RuntimeError("INVALID_PERTURBED_SIGNAL_MEDIAN")
    if target_eta > 0 and (not np.isfinite(unscaled_noise_median) or unscaled_noise_median <= 0):
        raise RuntimeError("INVALID_EMPIRICAL_NOISE_MEDIAN")
    alpha = 0.0 if target_eta == 0 else target_eta * signal_median / unscaled_noise_median
    scaled_noise = np.asarray(np.float32(alpha) * raw_noise, dtype=np.float32)
    preclip_eta = v54.safe_ratio(
        float(np.median(np.linalg.norm(scaled_noise[:, foreground_mask], axis=0))),
        signal_median,
    )
    observed = np.maximum(signal + scaled_noise, np.float32(0.0)).astype(np.float32)
    postclip_eta = v54.safe_ratio(
        float(np.median(np.linalg.norm(
            observed[:, foreground_mask] - signal[:, foreground_mask], axis=0
        ))),
        signal_median,
    )
    return observed, {
        "target_eta": float(target_eta),
        "achieved_eta_preclip": preclip_eta,
        "achieved_eta_postclip": postclip_eta,
        "median_unscaled_noise_l2": unscaled_noise_median,
        "scaling_alpha": float(alpha),
        "perturbed_pre_noise_foreground_median_signal_norm": signal_median,
        "observed_post_noise_foreground_median_signal_norm": float(
            np.median(np.linalg.norm(observed[:, foreground_mask], axis=0))
        ),
    }


def spectral_summary(rows: list[dict]) -> dict:
    if not rows:
        return {}
    return {
        "active_truth_candidate_count": len(rows),
        "requested_fragment_log_sigma": float(rows[0]["fragment_log_sigma"]),
        "requested_fragment_dropout_fraction": float(rows[0]["requested_dropout_fraction"]),
        "achieved_fragment_dropout_fraction": v54.numeric_summary(
            [float(row["achieved_dropout_fraction"]) for row in rows]
        ),
        "parent_multiplier": v54.numeric_summary(
            [float(row["parent_multiplier"]) for row in rows]
        ),
        "cosine_full_Aperturbed_vs_Asolver": v54.numeric_summary(
            [float(row["cosine_full_Aperturbed_vs_Asolver"]) for row in rows
             if row["cosine_full_Aperturbed_vs_Asolver"] is not None]
        ),
        "cosine_parent_Aperturbed_vs_Asolver": v54.numeric_summary(
            [float(row["cosine_parent_Aperturbed_vs_Asolver"]) for row in rows
             if row["cosine_parent_Aperturbed_vs_Asolver"] is not None]
        ),
        "cosine_fragment_Aperturbed_vs_Asolver": v54.numeric_summary(
            [float(row["cosine_fragment_Aperturbed_vs_Asolver"]) for row in rows
             if row["cosine_fragment_Aperturbed_vs_Asolver"] is not None]
        ),
    }


def construct_v55_case(
    context: dict, background: dict, condition_name: str, base_dataset_id: str
) -> tuple[dict, dict]:
    base = v54_case(context, base_dataset_id)
    frozen_x = base["X_true"]
    frozen_x_digest = array_digest(frozen_x)
    perturbed_A, perturbation_rows = build_perturbed_library(
        context, condition_name, base["active_indices"]
    )
    signal = np.einsum("mc,cyx->myx", perturbed_A, frozen_x, optimize=True).astype(
        np.float32, copy=False
    )
    split, replicate, complexity = base_dataset_id.split("_")
    raw_noise, seed_info = sampled_noise_field(
        context, background, condition_name, split, replicate
    )
    observed, noise_info = add_empirical_noise(
        signal, raw_noise, base["foreground_mask"],
        float(ERROR_CONDITIONS[condition_name]["noise_eta"]),
    )
    if array_digest(frozen_x) != frozen_x_digest:
        raise RuntimeError(f"X_TRUE_MUTATED_DURING_V55_CONSTRUCTION: {base_dataset_id}")
    if not np.array_equal(context["A_solver"], v54.v52.normalize_like_get_A_matrix(context["A_raw"])):
        raise RuntimeError("A_SOLVER_MUTATED_DURING_V55_CONSTRUCTION")
    v55_id = f"V55__{condition_name}__{base_dataset_id}"
    case = {
        **base,
        "case_name": v55_id,
        "B_sim": observed,
    }
    diagnostics = {
        "v55_dataset_id": v55_id,
        "base_v54_dataset_id": base_dataset_id,
        "error_condition": condition_name,
        "split": split,
        "replicate": replicate,
        "complexity_condition": complexity,
        "clean_foreground_median_signal_norm": base["achieved_foreground_B_l2_p50"],
        **seed_info,
        **noise_info,
        "spectral_mismatch_summary": spectral_summary(perturbation_rows),
        "frozen_X_true_sha256": frozen_x_digest,
    }
    return case, diagnostics


def truth_candidate_indices(mappings: list[dict]) -> list[int]:
    return sorted({int(row["synthetic_truth_candidate_index"]) for row in mappings})


def build_prepare_artifacts(context: dict) -> tuple[dict, list[dict], list[dict], list[dict]]:
    background = empirical_background(context)
    truth_indices = truth_candidate_indices(context["v54_mappings"])
    spectral_rows = []
    spectral_fingerprints: dict[tuple[str, int], str] = {}
    spectral_metadata: dict[tuple[str, int], dict] = {}
    for condition_name in CONDITION_ORDER:
        for candidate_index in truth_indices:
            column, row = perturb_candidate(context, condition_name, candidate_index)
            fingerprint = array_digest(column)
            key = (condition_name, candidate_index)
            if key in spectral_fingerprints and spectral_fingerprints[key] != fingerprint:
                raise RuntimeError("NONDETERMINISTIC_SPECTRAL_PERTURBATION")
            spectral_fingerprints[key] = fingerprint
            spectral_metadata[key] = row
            spectral_rows.append(row)

    # Replay every mapped occurrence. This explicitly checks that sharing an
    # identity across BASE/HIGH or R1/R2 cannot alter its perturbation.
    for condition_name in CONDITION_ORDER:
        for mapping_row in context["v54_mappings"]:
            candidate_index = int(mapping_row["synthetic_truth_candidate_index"])
            replay_column, replay_row = perturb_candidate(
                context, condition_name, candidate_index
            )
            key = (condition_name, candidate_index)
            if (
                array_digest(replay_column) != spectral_fingerprints[key]
                or replay_row != spectral_metadata[key]
            ):
                raise RuntimeError(
                    "SHARED_IDENTITY_PERTURBATION_MISMATCH: "
                    f"{condition_name}:{candidate_index}"
                )

    dataset_rows = []
    pair_diagnostics: dict[tuple[str, str, str], dict[str, dict]] = defaultdict(dict)
    for condition_name in CONDITION_ORDER:
        for base_dataset_id in BASE_DATASETS:
            case, diagnostics = construct_v55_case(
                context, background, condition_name, base_dataset_id
            )
            complexity = diagnostics["complexity_condition"]
            pair_key = (condition_name, diagnostics["split"], diagnostics["replicate"])
            pair_diagnostics[pair_key][complexity] = diagnostics
            dataset_rows.append({
                "v55_dataset_id": diagnostics["v55_dataset_id"],
                "base_v54_dataset_id": base_dataset_id,
                "error_condition": condition_name,
                "split": diagnostics["split"],
                "replicate": diagnostics["replicate"],
                "complexity_condition": complexity,
                "global_truth_K": case["global_truth_K"],
                "reportable_truth_global_count": case["reportable_truth_molecular_identity_count"],
                "subthreshold_truth_global_count": case["subthreshold_truth_molecular_identity_count"],
                "global_X_true_scaling_scalar": case["global_scale"],
                "target_noise_eta": diagnostics["target_eta"],
                "achieved_noise_eta_preclip": diagnostics["achieved_eta_preclip"],
                "achieved_noise_eta_postclip": diagnostics["achieved_eta_postclip"],
                "checkpoint_dataset_id": diagnostics["v55_dataset_id"],
                "dataset_directory": f"{condition_name}/{base_dataset_id}",
                "status": "PENDING_TRAINING",
            })

    noise_rows = []
    for condition_name in CONDITION_ORDER:
        for split in ("CAL", "HOLD"):
            for replicate in ("R1", "R2"):
                paired = pair_diagnostics[(condition_name, split, replicate)]
                if set(paired) != {"BASE76", "HIGH129"}:
                    raise RuntimeError(
                        f"INCOMPLETE_BASE_HIGH_NOISE_PAIR: {condition_name}:{split}:{replicate}"
                    )
                base_info, high_info = paired["BASE76"], paired["HIGH129"]
                if base_info["deterministic_noise_seed"] != high_info["deterministic_noise_seed"]:
                    raise RuntimeError("PAIRED_NOISE_SEED_MISMATCH")
                if base_info["sampled_background_index_sha256"] != high_info["sampled_background_index_sha256"]:
                    raise RuntimeError("PAIRED_NOISE_FIELD_MISMATCH")
                if base_info["median_unscaled_noise_l2"] != high_info["median_unscaled_noise_l2"]:
                    raise RuntimeError("PAIRED_UNSCALED_NOISE_NORM_MISMATCH")
                by_complexity = {
                    key: {
                        "achieved_eta_preclip": paired[key]["achieved_eta_preclip"],
                        "achieved_eta_postclip": paired[key]["achieved_eta_postclip"],
                        "scaling_alpha": paired[key]["scaling_alpha"],
                    }
                    for key in ("BASE76", "HIGH129")
                }
                noise_rows.append({
                    "error_condition": condition_name,
                    "split": split,
                    "replicate": replicate,
                    "deterministic_noise_seed": base_info["deterministic_noise_seed"],
                    "sampled_background_index_sha256": base_info["sampled_background_index_sha256"],
                    "background_pixel_count": background["background_pixel_count"],
                    "target_eta": float(ERROR_CONDITIONS[condition_name]["noise_eta"]),
                    "achieved_eta_preclip": json_cell({k: v["achieved_eta_preclip"] for k, v in by_complexity.items()}),
                    "achieved_eta_postclip": json_cell({k: v["achieved_eta_postclip"] for k, v in by_complexity.items()}),
                    "background_center_summary": json_cell(background["center_summary"]),
                    "median_unscaled_noise_l2": base_info["median_unscaled_noise_l2"],
                    "scaling_alpha": json_cell({k: v["scaling_alpha"] for k, v in by_complexity.items()}),
                    "same_noise_field_used_for_BASE_and_HIGH": True,
                })

    design = {
        "script_version": VERSION,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "status": "FROZEN_BEFORE_TRAINING",
        "v54_provenance": {
            "directory": str(V54_OUTPUT),
            "design_status": context["v54_design"]["status"],
            "script_version": context["v54_design"]["script_version"],
            "template_mappings_sha256": context["v54_mapping_sha256"],
            "pool_split_seed": v54.POOL_SPLIT_SEED,
            "mapping_seeds": v54.MAPPING_SEEDS,
            "base_dataset_ids": list(BASE_DATASETS),
            "CAL_HOLD_lipid_name_overlap": 0,
        },
        "input_hashes_sha256": context["validation"]["hashes_sha256"],
        "observed_B_noise_source": {
            "path": str(context["paths"]["B_cube"]),
            "sha256": context["validation"]["hashes_sha256"].get("B_cube"),
            "definition": "exact locked production observed B; background median centered per channel",
            "bootstrap_unit": "one complete 1084-channel background residual spectrum",
            "spatial_correlation_preserved": False,
        },
        "parent_mz_range_inclusive": list(PARENT_MZ_RANGE),
        "error_conditions": conditions_payload(),
        "datasets": list(V55_DATASETS),
        "new_GPU_dataset_count": len(V55_DATASETS),
        "truth_contract": {
            "source": "v54.construct_dataset using frozen v54 template_mappings.csv",
            "X_true_identical_to_v54": True,
            "independent_X_true_rescaling_after_perturbation": False,
            "molecular_identity_unit": "lipid_name",
            "report_gate": REPORT_GATE,
        },
        "model_mismatch_contract": {
            "A_solver_used_for_training_inference_rho_zero_and_NNLS": True,
            "A_true_perturbed_used_only_for_synthesis": True,
            "nontruth_columns_unchanged": True,
        },
        "training": context["v54_design"]["training"],
        "rho_zero": {"implementation": "src/rho_zero.py", "weighting": "W=I"},
        "noise_manifest_pair_scalars": (
            "Noise indices are shared by BASE76/HIGH129. Because eta is defined relative "
            "to each dataset's B_signal, achieved eta and alpha are JSON maps by complexity."
        ),
        "limitations": [
            "Whole-spectrum empirical bootstrap preserves within-spectrum channel covariance but not spatial noise correlation.",
            "Mismatch models fragment relative intensity, missing fragments, and parent/fragment balance only.",
            "No m/z drift or jitter, peak broadening, novel fragments, unmodeled chemical interferents, full production residual, or biological-library incompleteness is modeled.",
        ],
        "validity_status": "PENDING_REVIEW",
        "final_deployment_threshold_frozen": False,
    }
    return design, dataset_rows, spectral_rows, noise_rows


def write_prepare_artifacts(
    output: Path, design: dict, dataset_rows: list[dict],
    spectral_rows: list[dict], noise_rows: list[dict],
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    stage0.atomic_write_json(output / "design.json", design)
    v54.write_csv(output / "dataset_manifest.csv", dataset_rows, list(dataset_rows[0]))
    v54.write_csv(
        output / "spectral_perturbation_manifest.csv", spectral_rows, list(spectral_rows[0])
    )
    v54.write_csv(output / "noise_manifest.csv", noise_rows, list(noise_rows[0]))


def validate_v55_design(output: Path, context: dict) -> dict:
    design = v54.read_json(output / "design.json")
    if design.get("script_version") != VERSION:
        raise RuntimeError("V55_DESIGN_VERSION_MISMATCH")
    if design.get("status") != "FROZEN_BEFORE_TRAINING":
        raise RuntimeError("V55_DESIGN_STATUS_MISMATCH")
    if design.get("datasets") != list(V55_DATASETS):
        raise RuntimeError("V55_DESIGN_DATASET_MISMATCH")
    if design.get("error_conditions") != conditions_payload():
        raise RuntimeError("V55_ERROR_CONDITIONS_CHANGED")
    provenance = design.get("v54_provenance", {})
    if provenance.get("template_mappings_sha256") != context["v54_mapping_sha256"]:
        raise RuntimeError("V54_FROZEN_MAPPING_CHANGED_AFTER_V55_PREPARATION")
    if design.get("input_hashes_sha256") != context["validation"]["hashes_sha256"]:
        raise RuntimeError("V55_LOCKED_ASSET_HASH_MISMATCH")
    return design


def solver_library_fit_diagnostic(case: dict, context: dict) -> dict:
    result = v54.reporting_gate_oracle(case, context["A_solver"], context["metadata"])
    result["status"] = "DIAGNOSTIC_ONLY"
    result["label"] = "solver_library_fit_diagnostic"
    result["training_gate"] = False
    result["A_solver_used"] = True
    result["nonzero_false_identity_interpretation"] = (
        "Expected possible consequence of intentional noise/model mismatch; not a training blocker."
    )
    return result


def run_dataset(
    args: argparse.Namespace, context: dict, condition_name: str, base_dataset_id: str
) -> dict:
    validate_v55_design(args.output_dir, context)
    dataset_dir = args.output_dir / condition_name / base_dataset_id
    existing_path = dataset_dir / "report.json"
    if existing_path.exists():
        existing = v54.read_json(existing_path)
        if existing.get("status") == "COMPLETE":
            return existing
    background = empirical_background(context)
    case, diagnostics = construct_v55_case(
        context, background, condition_name, base_dataset_id
    )
    fit_diagnostic = solver_library_fit_diagnostic(case, context)
    dataset_dir.mkdir(parents=True, exist_ok=True)
    base_report = {
        "dataset_id": base_dataset_id,
        "v55_dataset_id": diagnostics["v55_dataset_id"],
        "base_v54_dataset_id": base_dataset_id,
        "error_condition": condition_name,
        "split": diagnostics["split"],
        "replicate": diagnostics["replicate"],
        "complexity_condition": diagnostics["complexity_condition"],
        "global_truth_K": case["global_truth_K"],
        "global_truth_identity_count": case["global_truth_identity_count"],
        "all_truth_molecular_identity_count": case["all_truth_molecular_identity_count"],
        "reportable_truth_global_count": case["reportable_truth_molecular_identity_count"],
        "subthreshold_truth_global_count": case["subthreshold_truth_molecular_identity_count"],
        "reportable_truth_molecular_identity_count": case["reportable_truth_molecular_identity_count"],
        "subthreshold_truth_molecular_identity_count": case["subthreshold_truth_molecular_identity_count"],
        "true_per_pixel_K_foreground": case["true_per_pixel_K_foreground"],
        "K_true_pixel_foreground": case["K_true_pixel_foreground"],
        "V54_global_X_true_scaling_scalar": case["global_scale"],
        "global_signal_scalar": case["global_scale"],
        "clean_foreground_median_signal_norm": diagnostics["clean_foreground_median_signal_norm"],
        "perturbed_pre_noise_foreground_median_signal_norm": diagnostics["perturbed_pre_noise_foreground_median_signal_norm"],
        "observed_post_noise_foreground_median_signal_norm": diagnostics["observed_post_noise_foreground_median_signal_norm"],
        "target_noise_eta": diagnostics["target_eta"],
        "achieved_noise_eta": diagnostics["achieved_eta_postclip"],
        "achieved_noise_eta_preclip": diagnostics["achieved_eta_preclip"],
        "spectral_mismatch_summaries": diagnostics["spectral_mismatch_summary"],
        "deterministic_noise_seed": diagnostics["deterministic_noise_seed"],
        "sampled_background_index_sha256": diagnostics["sampled_background_index_sha256"],
        "solver_library_fit_diagnostic": fit_diagnostic,
        "learned_reported_count_definition": "foreground_mean(X_hat_j) > 1e-3",
        "A_solver_used_by_solver": True,
        "A_perturbed_used_only_for_synthesis": True,
        "X_true_used_in_training_or_stopping": False,
        "real_data_checkpoint_loaded": False,
    }
    # Deliberately omit X_true and A_perturbed from the training-facing case.
    training_case = {
        "case_name": diagnostics["v55_dataset_id"],
        "B_sim": case["B_sim"],
        "foreground_mask": case["foreground_mask"],
    }
    if set(training_case) != {"case_name", "B_sim", "foreground_mask"}:
        raise RuntimeError("TRAINING_CASE_CONTAINS_FORBIDDEN_TRUTH_OR_LIBRARY_DATA")
    learned = v54.train_resumable(training_case, context, dataset_dir)
    if learned["stop_reason"] == "nonfinite_loss":
        report = {
            **base_report,
            "status": "TRAINING_FAILED_NONFINITE_LOSS",
            "learned_training_performed": True,
            "training": {
                "final_epoch": learned["stopped_epoch"],
                "stop_reason": learned["stop_reason"],
                "real_data_checkpoint_loaded": False,
                "X_true_used_in_training_or_stopping": False,
            },
        }
        stage0.atomic_write_json(existing_path, report)
        return report

    records, raw_result = v54.learned_records(
        base_dataset_id, case, learned, context, args.rho_workers
    )
    for row in records:
        row["v55_dataset_id"] = diagnostics["v55_dataset_id"]
        row["error_condition"] = condition_name
    report = {
        **base_report,
        "status": "COMPLETE",
        "learned_training_performed": True,
        "learned_raw_identity_performance": raw_result,
        "training": {
            "final_epoch": learned["stopped_epoch"],
            "stop_reason": learned["stop_reason"],
            "wall_time_seconds_this_invocation": learned["wall_time_seconds_this_invocation"],
            "hard_epoch_cap": v54.MAX_EPOCH,
            "checkpoint_epochs": list(v54.CHECKPOINT_EPOCHS),
            "scheduler_T_max": learned["scheduler_T_max"],
            "resumed_from_runtime_checkpoint": learned["resumed_from_runtime_checkpoint"],
            "checkpoint_dataset_id": diagnostics["v55_dataset_id"],
            "real_data_checkpoint_loaded": False,
            "X_true_used_in_training_or_stopping": False,
            "final_physical_loss": learned["final_physical_loss"],
            "final_raw_losses": learned["final_raw_losses"],
            "channel_weighting": learned["channel_weighting"],
        },
        "rho_zero_weighting": "W=I",
        "rho_zero_observation": "mean foreground of the actual V55 B_obs",
        "rho_zero_library": "original frozen A_solver",
    }
    fields = [
        "dataset_id", "v55_dataset_id", "error_condition", "split", "replicate",
        "complexity_condition", "global_truth_K", "candidate_index", "candidate_id",
        "lipid_name", "X_hat", "candidate_truth", "molecular_truth",
        "same_lipid_alternative_candidate", "rho_zero",
    ]
    v54.write_csv(dataset_dir / "reported_identity_records.csv", records, fields)
    stage0.atomic_write_json(existing_path, report)
    return report


def load_condition_completed(
    output: Path, condition_name: str, dataset_names: tuple[str, ...]
) -> tuple[list[dict], dict[str, dict]]:
    records: list[dict] = []
    reports: dict[str, dict] = {}
    for base_dataset_id in dataset_names:
        directory = output / condition_name / base_dataset_id
        report = v54.read_json(directory / "report.json")
        if report.get("status") != "COMPLETE":
            raise RuntimeError(f"DATASET_NOT_COMPLETE: {condition_name}:{base_dataset_id}")
        if report.get("error_condition") != condition_name:
            raise RuntimeError(f"DATASET_CONDITION_MISMATCH: {condition_name}:{base_dataset_id}")
        reports[base_dataset_id] = report
        records.extend(
            {key: v54.parse_scalar(value) for key, value in row.items()}
            for row in v54.read_csv(directory / "reported_identity_records.csv")
        )
    return records, reports


def selected_dataset_ids(reports: dict[str, dict], scope: str) -> list[str]:
    ids = sorted(reports)
    if scope == "POOLED":
        return ids
    return [name for name in ids if reports[name]["complexity_condition"] == scope]


def raw_molecular_performance(
    units: list[dict], reports: dict[str, dict], scope: str = "POOLED"
) -> dict:
    dataset_ids = selected_dataset_ids(reports, scope)
    scoped_units = [row for row in units if row["dataset_id"] in dataset_ids]
    all_truth = sum(int(reports[name]["global_truth_identity_count"]) for name in dataset_ids)
    reportable_truth = sum(
        len(reports[name]["learned_raw_identity_performance"]["reportable_truth_lipid_names"])
        for name in dataset_ids
    )
    tp = sum(bool(row["molecular_truth"]) for row in scoped_units)
    fp = len(scoped_units) - tp
    reportable_tp = sum(
        bool(row["molecular_truth"]) and bool(row["reportable_truth"])
        for row in scoped_units
    )
    return {
        "molecular_TP": tp,
        "molecular_FP": fp,
        "molecular_FN": all_truth - tp,
        "precision": v54.safe_ratio(tp, tp + fp),
        "FDR": v54.safe_ratio(fp, tp + fp),
        "all_truth_recall": v54.safe_ratio(tp, all_truth),
        "reportable_truth_recall": v54.safe_ratio(reportable_tp, reportable_truth),
        "reported_molecular_identity_count": len(scoped_units),
    }


def required_distribution(values: list[float]) -> dict:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return {
            "n": 0, "min": None, "q10": None, "q25": None, "median": None,
            "q75": None, "q90": None, "max": None,
        }
    return {
        "n": int(array.size),
        "min": float(array.min()),
        "q10": float(np.quantile(array, 0.10)),
        "q25": float(np.quantile(array, 0.25)),
        "median": float(np.median(array)),
        "q75": float(np.quantile(array, 0.75)),
        "q90": float(np.quantile(array, 0.90)),
        "max": float(array.max()),
    }


def rho_distributions(units: list[dict]) -> dict:
    true_values = [float(row["rho_zero"]) for row in units if bool(row["molecular_truth"])]
    false_values = [float(row["rho_zero"]) for row in units if not bool(row["molecular_truth"])]
    true_summary = required_distribution(true_values)
    false_summary = required_distribution(false_values)
    separation = {"eps": SEPARATION_EPS, "descriptive_only": True}
    if true_values and false_values:
        separation.update({
            "log_median_separation": float(
                np.log10(np.median(true_values) + SEPARATION_EPS)
                - np.log10(np.median(false_values) + SEPARATION_EPS)
            ),
            "conservative_gap": float(
                np.log10(np.min(true_values) + SEPARATION_EPS)
                - np.log10(np.max(false_values) + SEPARATION_EPS)
            ),
            "numerical_zero_gaps_are_not_physical_quantities": True,
        })
    else:
        separation.update({"log_median_separation": None, "conservative_gap": None})
    return {"true": true_summary, "false": false_summary, "separation": separation}


def threshold_validation(
    thresholds: dict, hold_units: list[dict], hold_reports: dict[str, dict], score: str
) -> dict:
    result = {}
    for name, threshold in thresholds.items():
        if threshold is None:
            result[name] = {"achievable_on_CAL": False}
            continue
        value = float(threshold["threshold"])
        result[name] = {
            "achievable_on_CAL": True,
            "frozen_CAL_threshold": value,
            "HOLD_POOLED": v54.apply_threshold(
                hold_units, hold_reports, value, score, "POOLED"
            ),
            "HOLD_BASE76": v54.apply_threshold(
                hold_units, hold_reports, value, score, "BASE76"
            ),
            "HOLD_HIGH129": v54.apply_threshold(
                hold_units, hold_reports, value, score, "HIGH129"
            ),
        }
    return result


def analyze_condition(output: Path, condition_name: str) -> dict:
    cal_records, cal_reports = load_condition_completed(output, condition_name, v54.CAL_DATASETS)
    hold_records, hold_reports = load_condition_completed(output, condition_name, v54.HOLD_DATASETS)
    all_reports = {**cal_reports, **hold_reports}
    cal_units = v54.molecular_units(cal_records, cal_reports)
    hold_units = v54.molecular_units(hold_records, hold_reports)
    all_units = [*cal_units, *hold_units]
    rho_curves = {
        scope: v54.metric_curve(cal_units, cal_reports, "rho_zero", scope)
        for scope in ("POOLED", "BASE76", "HIGH129")
    }
    abundance_curves = {
        scope: v54.metric_curve(cal_units, cal_reports, "X_hat", scope)
        for scope in ("POOLED", "BASE76", "HIGH129")
    }
    rho_thresholds = {
        "tau_cal_FDR5": v54.calibration_threshold(rho_curves["POOLED"], 0.05),
        "tau_cal_FDR1": v54.calibration_threshold(rho_curves["POOLED"], 0.01),
    }
    abundance_thresholds = {
        "tau_cal_FDR5": v54.calibration_threshold(abundance_curves["POOLED"], 0.05),
        "tau_cal_FDR1": v54.calibration_threshold(abundance_curves["POOLED"], 0.01),
    }
    raw = {
        "ALL_POOLED": raw_molecular_performance(all_units, all_reports),
        "CAL_POOLED": raw_molecular_performance(cal_units, cal_reports),
        "HOLD_POOLED": raw_molecular_performance(hold_units, hold_reports),
        "HOLD_BASE76": raw_molecular_performance(hold_units, hold_reports, "BASE76"),
        "HOLD_HIGH129": raw_molecular_performance(hold_units, hold_reports, "HIGH129"),
    }
    rho_distribution = {
        "ALL_POOLED": rho_distributions(all_units),
        "CAL_POOLED": rho_distributions(cal_units),
        "HOLD_POOLED": rho_distributions(hold_units),
        "HOLD_BASE76": rho_distributions(
            [row for row in hold_units if row["complexity_condition"] == "BASE76"]
        ),
        "HOLD_HIGH129": rho_distributions(
            [row for row in hold_units if row["complexity_condition"] == "HIGH129"]
        ),
    }
    return {
        "error_condition": condition_name,
        "parameters": dict(ERROR_CONDITIONS[condition_name]),
        "raw_learned_identity_performance": raw,
        "rho_zero_distributions": rho_distribution,
        "rho_zero_calibration": {
            "thresholds": rho_thresholds,
            "descriptive_CAL_curves": rho_curves,
        },
        "condition_specific_HOLD_validation": threshold_validation(
            rho_thresholds, hold_units, hold_reports, "rho_zero"
        ),
        "abundance_only_baseline": {
            "calibration_thresholds": abundance_thresholds,
            "descriptive_CAL_curves": abundance_curves,
            "HOLD_validation": threshold_validation(
                abundance_thresholds, hold_units, hold_reports, "X_hat"
            ),
        },
        "BASE76_vs_HIGH129": {
            "raw_HOLD_BASE76": raw["HOLD_BASE76"],
            "raw_HOLD_HIGH129": raw["HOLD_HIGH129"],
        },
        "dataset_reports": all_reports,
        "_hold_units": hold_units,
        "_hold_reports": hold_reports,
    }


def extract_thresholds(payload: dict | None) -> dict[str, float | None]:
    def find_value(value: Any) -> float | None:
        if value is None:
            return None
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, dict):
            for key in ("threshold", "frozen_CAL_threshold", "tau"):
                if key in value and isinstance(value[key], (int, float)):
                    return float(value[key])
        return None

    payload = payload or {}
    for _ in range(4):
        nested = next((
            payload[key]
            for key in (
                "rho_zero_calibration", "rho_zero", "thresholds",
                "cal_frozen_thresholds",
            )
            if key in payload and isinstance(payload[key], dict)
        ), None)
        if nested is None:
            break
        payload = nested
    return {
        name: find_value(payload.get(name))
        for name in ("tau_cal_FDR5", "tau_cal_FDR1")
    }


def clean_threshold_source() -> dict:
    report_path = V54_OUTPUT / "report.json"
    frozen_path = V54_OUTPUT / "cal_frozen_thresholds.json"
    report_payload = v54.read_json(report_path) if report_path.exists() else None
    frozen_payload = v54.read_json(frozen_path) if frozen_path.exists() else None
    report_section = (
        report_payload.get("rho_zero_calibration")
        if isinstance(report_payload, dict) else None
    )
    report_thresholds = (
        extract_thresholds(report_section)
        if isinstance(report_section, dict)
        and set(("tau_cal_FDR5", "tau_cal_FDR1")).issubset(report_section)
        else None
    )
    frozen_thresholds = extract_thresholds(frozen_payload) if frozen_payload else None
    if report_thresholds and frozen_thresholds:
        for name in report_thresholds:
            left, right = report_thresholds[name], frozen_thresholds[name]
            if left is None and right is None:
                continue
            if left is None or right is None or not np.isclose(left, right, rtol=1e-10, atol=1e-12):
                raise RuntimeError(f"V54_CLEAN_THRESHOLD_SOURCE_DISAGREEMENT: {name}")
    if report_thresholds:
        return {"status": "AVAILABLE", "source": str(report_path), "thresholds": report_thresholds}
    if frozen_thresholds:
        return {"status": "AVAILABLE", "source": str(frozen_path), "thresholds": frozen_thresholds}
    return {"status": "PENDING_V54_THRESHOLDS", "source": None, "thresholds": {}}


def clean_reference() -> dict:
    report_path = V54_OUTPUT / "report.json"
    if not report_path.exists():
        return {"status": "PENDING_V54_AGGREGATE", "source": str(report_path)}
    aggregate_report = v54.read_json(report_path)
    try:
        cal_records, cal_reports = v54.load_completed(V54_OUTPUT, v54.CAL_DATASETS)
        hold_records, hold_reports = v54.load_completed(V54_OUTPUT, v54.HOLD_DATASETS)
    except RuntimeError as exc:
        return {
            "status": "PENDING_V54_HOLD_DATASETS",
            "source": str(report_path),
            "diagnostic": str(exc),
            "aggregate_report": aggregate_report,
        }
    cal_units = v54.molecular_units(cal_records, cal_reports)
    hold_units = v54.molecular_units(hold_records, hold_reports)
    reports = {**cal_reports, **hold_reports}
    units = [*cal_units, *hold_units]
    return {
        "status": "AVAILABLE",
        "source": str(report_path),
        "raw_learned_identity_performance": {
            "ALL_POOLED": raw_molecular_performance(units, reports),
            "CAL_POOLED": raw_molecular_performance(cal_units, cal_reports),
            "HOLD_POOLED": raw_molecular_performance(hold_units, hold_reports),
            "HOLD_BASE76": raw_molecular_performance(hold_units, hold_reports, "BASE76"),
            "HOLD_HIGH129": raw_molecular_performance(hold_units, hold_reports, "HIGH129"),
        },
        "rho_zero_distributions": {
            "ALL_POOLED": rho_distributions(units),
            "CAL_POOLED": rho_distributions(cal_units),
            "HOLD_POOLED": rho_distributions(hold_units),
            "HOLD_BASE76": rho_distributions(
                [row for row in hold_units if row["complexity_condition"] == "BASE76"]
            ),
            "HOLD_HIGH129": rho_distributions(
                [row for row in hold_units if row["complexity_condition"] == "HIGH129"]
            ),
        },
        "rho_zero_calibration": aggregate_report.get("rho_zero_calibration"),
        "condition_specific_HOLD_validation": aggregate_report.get("rho_heldout_validation"),
        "abundance_only_baseline": aggregate_report.get("abundance_only_baseline"),
    }


def apply_clean_transfer(condition_analysis: dict, source: dict) -> dict:
    if source["status"] != "AVAILABLE":
        return {"status": source["status"], "source": source.get("source")}
    hold_units = condition_analysis["_hold_units"]
    hold_reports = condition_analysis["_hold_reports"]
    result = {"status": "COMPLETE", "source": source["source"]}
    for name, value in source["thresholds"].items():
        if value is None:
            result[name] = {"available_from_V54_CLEAN": False}
            continue
        result[name] = {
            "available_from_V54_CLEAN": True,
            "V54_CLEAN_frozen_threshold": value,
            "HOLD_POOLED": v54.apply_threshold(
                hold_units, hold_reports, value, "rho_zero", "POOLED"
            ),
            "HOLD_BASE76": v54.apply_threshold(
                hold_units, hold_reports, value, "rho_zero", "BASE76"
            ),
            "HOLD_HIGH129": v54.apply_threshold(
                hold_units, hold_reports, value, "rho_zero", "HIGH129"
            ),
        }
    return result


def build_summary(report: dict) -> str:
    conditions = report["conditions"]
    return "\n".join([
        "# v55 Spectral and Noise Robustness", "",
        "## 1. Purpose", "",
        "Test whether rho_zero remains a reliable molecular-identity certificate when the observed spectrum contains empirical measurement noise and/or cannot be represented exactly by the solver library.", "",
        "## 2. V54 CLEAN reference", "",
        f"`{json.dumps(conditions['CLEAN'], ensure_ascii=False)}`", "",
        "## 3. Measurement-noise model", "",
        "The exact locked production B is loaded through the Stage0 lock. Per-channel background medians are removed, then whole 1084-channel background residual spectra are bootstrapped. BASE76/HIGH129 pairs share sampled spectra; scaling is dataset-specific because eta is defined relative to each B_signal. Spatial correlation is not claimed.", "",
        "## 4. Spectral-library mismatch model", "",
        "Only truth columns are perturbed. Originally nonzero fragments receive log-normal relative-intensity variation and deterministic dropout with at least one fragment retained; parents receive one candidate-level multiplier; the full column is then L2-normalized. A_perturbed is synthesis-only.", "",
        "## 5. Perturbation severity audit", "",
        f"`{json.dumps(report['perturbation_severity_audit'], ensure_ascii=False)}`", "",
        "## 6. Raw learned identity performance", "",
        f"`{json.dumps({name: value.get('raw_learned_identity_performance') for name, value in conditions.items()}, ensure_ascii=False)}`", "",
        "## 7. rho_zero true/false distributions", "",
        f"`{json.dumps({name: value.get('rho_zero_distributions') for name, value in conditions.items()}, ensure_ascii=False)}`", "",
        "## 8. Condition-specific CAL thresholds", "",
        f"`{json.dumps({name: conditions[name].get('rho_zero_calibration', {}).get('thresholds') for name in CONDITION_ORDER}, ensure_ascii=False)}`", "",
        "## 9. Condition-specific HOLD validation", "",
        f"`{json.dumps({name: conditions[name].get('condition_specific_HOLD_validation') for name in CONDITION_ORDER}, ensure_ascii=False)}`", "",
        "## 10. V54 CLEAN-threshold transfer", "",
        f"`{json.dumps(report['clean_threshold_transfer'], ensure_ascii=False)}`", "",
        "## 11. BASE76 vs HIGH129", "",
        f"`{json.dumps({name: conditions[name].get('BASE76_vs_HIGH129') for name in CONDITION_ORDER}, ensure_ascii=False)}`", "",
        "## 12. Abundance-only baseline", "",
        f"`{json.dumps({name: conditions[name].get('abundance_only_baseline') for name in CONDITION_ORDER}, ensure_ascii=False)}`", "",
        "## 13. Limitations", "",
        "- Whole-spectrum background bootstrap preserves channel covariance within a sampled spectrum, not spatial noise correlation.",
        "- Modeled mismatch covers fragment relative intensity, missing fragments, and parent/fragment balance.",
        "- It does not cover m/z drift or jitter, peak broadening, novel fragments, unmodeled chemical interferents, the full empirical production residual, or biological-library incompleteness.",
        "- Numerical-zero separation gaps are descriptive, not physical quantities.", "",
        "## 14. Validity status", "",
        "PENDING_REVIEW", "",
        "Final deployment threshold frozen: NO", "",
    ])


def update_experiment_log(report: dict, design: dict) -> None:
    condition_results = {
        name: {
            "rho_zero_calibration": report["conditions"][name]["rho_zero_calibration"]["thresholds"],
            "condition_specific_HOLD_validation": report["conditions"][name]["condition_specific_HOLD_validation"],
        }
        for name in CONDITION_ORDER
    }
    lines = [
        LOG_BEGIN,
        "## v55 Spectral and Noise Robustness", "",
        f"- Version: {VERSION}",
        f"- Date: {date.today().isoformat()}",
        f"- Exact condition parameters: {json.dumps(conditions_payload(), ensure_ascii=False)}",
        f"- Seeds: {json.dumps({name: ERROR_CONDITIONS[name]['seed'] for name in CONDITION_ORDER}, ensure_ascii=False)}; noise additionally keyed by split/replicate; spectral perturbations additionally keyed by candidate/type.",
        f"- V54 provenance: {json.dumps(design['v54_provenance'], ensure_ascii=False)}",
        f"- Locked hashes: {json.dumps(design['input_hashes_sha256'], ensure_ascii=False)}",
        f"- Number of new trained datasets: {len(V55_DATASETS)}",
        f"- Condition calibration and heldout results: {json.dumps(condition_results, ensure_ascii=False)}",
        f"- CLEAN-threshold transfer: {json.dumps(report['clean_threshold_transfer'], ensure_ascii=False)}",
        "- Limitations: whole-spectrum noise bootstrap does not preserve spatial correlation; mismatch omits m/z drift/jitter, broadening, novel fragments, unmodeled interferents, full empirical production residual, and biological-library incompleteness.",
        "- Validity status: PENDING_REVIEW",
        "- Final threshold frozen: NO", "", LOG_END,
    ]
    entry = "\n".join(lines)
    current = EXPERIMENT_LOG.read_text(encoding="utf-8") if EXPERIMENT_LOG.exists() else "# Experiment Log\n"
    if current.count(LOG_BEGIN) > 1 or current.count(LOG_END) > 1:
        raise RuntimeError("DUPLICATE_EXPERIMENT_LOG_MARKERS")
    start, end = current.find(LOG_BEGIN), current.find(LOG_END)
    if start >= 0 and end >= start:
        end += len(LOG_END)
        current = current[:start].rstrip() + "\n\n" + entry + current[end:]
    elif start >= 0 or end >= 0:
        raise RuntimeError("INVALID_EXPERIMENT_LOG_MARKERS")
    else:
        current = current.rstrip() + "\n\n" + entry + "\n"
    stage0.atomic_write_text(EXPERIMENT_LOG, current.rstrip() + "\n")


def aggregate(output: Path) -> dict:
    design = v54.read_json(output / "design.json")
    if design.get("script_version") != VERSION:
        raise RuntimeError("V55_DESIGN_VERSION_MISMATCH")
    analyses = {name: analyze_condition(output, name) for name in CONDITION_ORDER}
    threshold_source = clean_threshold_source()
    transfers = {
        name: apply_clean_transfer(analyses[name], threshold_source)
        for name in CONDITION_ORDER
    }
    for analysis in analyses.values():
        analysis.pop("_hold_units", None)
        analysis.pop("_hold_reports", None)
    clean = clean_reference()
    conditions = {"CLEAN": clean, **analyses}
    conditions = {name: conditions[name] for name in SUMMARY_CONDITION_ORDER}
    perturbation_audit = {
        name: {
            "parameters": dict(ERROR_CONDITIONS[name]),
            "dataset_spectral_summaries": {
                dataset_id: analyses[name]["dataset_reports"][dataset_id]["spectral_mismatch_summaries"]
                for dataset_id in BASE_DATASETS
            },
            "dataset_noise_eta": {
                dataset_id: {
                    "target": analyses[name]["dataset_reports"][dataset_id]["target_noise_eta"],
                    "preclip": analyses[name]["dataset_reports"][dataset_id]["achieved_noise_eta_preclip"],
                    "postclip": analyses[name]["dataset_reports"][dataset_id]["achieved_noise_eta"],
                }
                for dataset_id in BASE_DATASETS
            },
        }
        for name in CONDITION_ORDER
    }
    report = {
        "status": "PENDING_REVIEW",
        "script_version": VERSION,
        "condition_order": list(SUMMARY_CONDITION_ORDER),
        "conditions": conditions,
        "perturbation_severity_audit": perturbation_audit,
        "clean_threshold_transfer_source": threshold_source,
        "clean_threshold_transfer": transfers,
        "new_GPU_dataset_count": len(V55_DATASETS),
        "solver_library_contract": {
            "A_solver_used_by_solver": True,
            "A_perturbed_used_only_for_synthesis": True,
            "X_true_used_in_training_or_stopping": False,
            "real_data_checkpoint_loaded": False,
        },
        "validity_status": "PENDING_REVIEW",
        "decision": "PENDING_RESULT_REVIEW",
        "final_deployment_threshold_frozen": False,
    }
    stage0.atomic_write_json(output / "report.json", report)
    stage0.atomic_write_text(output / "summary.md", build_summary(report))
    update_experiment_log(report, design)
    return report


def selected_pairs(args: argparse.Namespace) -> list[tuple[str, str]]:
    conditions = [args.condition] if args.condition else list(CONDITION_ORDER)
    datasets = [args.dataset] if args.dataset else list(BASE_DATASETS)
    return [(condition, dataset) for condition in conditions for dataset in datasets]


def run_child(args: argparse.Namespace, condition_name: str, base_dataset_id: str, gpu: str) -> None:
    command = [
        sys.executable, str(Path(__file__).resolve()),
        "--asset-root", str(args.asset_root.resolve()),
        "--output-dir", str(args.output_dir.resolve()),
        "--condition", condition_name,
        "--dataset", base_dataset_id,
        "--rho-workers", str(args.rho_workers),
    ]
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = gpu
    completed = subprocess.run(command, env=environment, check=False)
    if completed.returncode != 0:
        raise RuntimeError(
            f"DATASET_WORKER_FAILED: {condition_name}:{base_dataset_id} on GPU {gpu}"
        )


def run_selected(args: argparse.Namespace, context: dict) -> list[dict]:
    pending = []
    for condition_name, base_dataset_id in selected_pairs(args):
        report_path = args.output_dir / condition_name / base_dataset_id / "report.json"
        if report_path.exists() and v54.read_json(report_path).get("status") == "COMPLETE":
            continue
        pending.append((condition_name, base_dataset_id))
    if args.parallel_gpus:
        from concurrent.futures import ThreadPoolExecutor

        gpus = [item.strip() for item in args.parallel_gpus.split(",") if item.strip()]
        if not gpus or len(gpus) != len(set(gpus)):
            raise RuntimeError("INVALID_PARALLEL_GPU_LIST")
        queues = {gpu: pending[position::len(gpus)] for position, gpu in enumerate(gpus)}
        with ThreadPoolExecutor(max_workers=len(gpus)) as executor:
            futures = [
                executor.submit(
                    lambda gpu=gpu: [
                        run_child(args, condition, dataset, gpu)
                        for condition, dataset in queues[gpu]
                    ]
                )
                for gpu in gpus
            ]
            for future in futures:
                future.result()
        return []
    return [run_dataset(args, context, condition, dataset) for condition, dataset in pending]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-root", type=Path, default=ROOT.parent / "decon-lipid")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DEFAULT)
    parser.add_argument("--prepare-design", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--condition", choices=CONDITION_ORDER)
    parser.add_argument("--dataset", choices=BASE_DATASETS)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--aggregate", action="store_true")
    parser.add_argument("--parallel-gpus")
    parser.add_argument("--rho-workers", type=int, default=1)
    args = parser.parse_args()
    action_count = sum((args.prepare_design, args.dry_run, args.aggregate))
    training_requested = bool(args.all or args.dataset)
    if action_count + int(training_requested) != 1:
        parser.error(
            "choose exactly one action: --prepare-design, --dry-run, --aggregate, "
            "--all [optionally --condition], or --condition ... --dataset ..."
        )
    if args.dataset and not args.condition:
        parser.error("--dataset requires --condition")
    if args.all and args.dataset:
        parser.error("--all and --dataset are mutually exclusive")
    if args.condition and not (args.all or args.dataset) and not args.dry_run:
        parser.error("--condition requires --all or --dataset")
    if args.parallel_gpus and not args.all:
        parser.error("--parallel-gpus is valid only with --all")
    if args.rho_workers < 1:
        parser.error("--rho-workers must be >= 1")
    return args


def main() -> None:
    args = parse_args()
    args.output_dir = args.output_dir.resolve()
    if args.aggregate:
        result = aggregate(args.output_dir)
        print(json.dumps(stage0.to_jsonable(result), ensure_ascii=False, indent=2, allow_nan=False))
        return

    context = load_context(args.asset_root.resolve())
    missing = {
        "missing_locked_assets": context.get("missing_assets", []),
        "missing_v51_v53_results": context.get("missing_results", []),
        "missing_v54_frozen_design": context.get("missing_v54_design", []),
    }
    if any(missing.values()):
        payload = {
            "status": "DRY_RUN_REMOTE_DEPENDENCIES_REQUIRED",
            **missing,
            "planned_conditions": conditions_payload(),
            "planned_base_datasets": list(BASE_DATASETS),
            "new_GPU_dataset_count": len(V55_DATASETS),
            "outputs_written": False,
            "GPU_training_performed": False,
        }
        if args.dry_run:
            print(json.dumps(payload, ensure_ascii=False, indent=2))
            return
        raise RuntimeError("REQUIRED_LOCKED_DEPENDENCIES_MISSING: " + json.dumps(payload))

    if args.dry_run:
        design, dataset_rows, spectral_rows, noise_rows = build_prepare_artifacts(context)
        print(json.dumps(stage0.to_jsonable({
            "status": "DRY_RUN_READY",
            "v54_design_status": context["v54_design"]["status"],
            "locked_hashes_validated": True,
            "conditions": design["error_conditions"],
            "new_GPU_dataset_count": len(V55_DATASETS),
            "dataset_manifest_rows": len(dataset_rows),
            "spectral_perturbation_manifest_rows": len(spectral_rows),
            "noise_manifest_rows": len(noise_rows),
            "outputs_written": False,
            "GPU_training_performed": False,
        }), ensure_ascii=False, indent=2, allow_nan=False))
        return

    if args.prepare_design:
        design, dataset_rows, spectral_rows, noise_rows = build_prepare_artifacts(context)
        write_prepare_artifacts(
            args.output_dir, design, dataset_rows, spectral_rows, noise_rows
        )
        print(json.dumps({
            "status": "DESIGN_PREPARED",
            "output_dir": str(args.output_dir),
            "new_GPU_dataset_count": len(V55_DATASETS),
            "GPU_training_performed": False,
        }, indent=2))
        return

    if not (args.output_dir / "design.json").exists():
        raise RuntimeError("DESIGN_NOT_PREPARED: run --prepare-design first")
    validate_v55_design(args.output_dir, context)
    reports = run_selected(args, context)
    print(json.dumps(stage0.to_jsonable({
        "status": "REQUESTED_DATASETS_COMPLETE",
        "completed_this_invocation": [report["v55_dataset_id"] for report in reports],
    }), ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
