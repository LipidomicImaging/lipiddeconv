#!/usr/bin/env python3
"""Run the v53 clean template-truth molecular-identity calibration pilot.

The locked production X_hat maps supply spatial/amplitude templates only.
Synthetic molecular truth is selected independently from frozen v51 library
geometry, fixed before optimization, and never enters training or stopping.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import sys
from collections import defaultdict
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import nnls


ROOT = Path(__file__).resolve().parents[1]
ANALYSIS = ROOT / "analysis"
for import_root in (ROOT, ANALYSIS):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

import run_v50_reoptimized_sanity as v50
import run_v52_mechanism_leakage as v52
import run_v53_stage0_empirical_domain as stage0
from src.rho_zero import identity_weights, rho_zero_from_weighted_case


VERSION = "v53_template_truth_pilot"
OUTPUT_DEFAULT = ROOT / "results/v53_template_truth_pilot"
STAGE0_DIR = ROOT / "results/v53_stage0_empirical_domain"
V51_DIR = ROOT / "results/v51_competition_structure"
EXPERIMENT_LOG = ROOT / "results/EXPERIMENT_LOG.md"
REPORT_GATE = 1.0e-3
SENSITIVITY_GATE = 1.0e-4
EXPECTED_TEMPLATE_COUNT = 76
EXPECTED_CANDIDATE_COUNT = 391
DIFFICULTY_STRATA = (
    "Q4_highest_competition",
    "Q3_moderate_high_competition",
    "Q2_moderate_low_competition",
    "Q1_lowest_competition",
)
PERMUTATION_MULTIPLIER = 17
PERMUTATION_OFFSET = 11
LOG_BEGIN = "<!-- BEGIN v53_template_truth_pilot -->"
LOG_END = "<!-- END v53_template_truth_pilot -->"


def read_json(path: Path) -> dict:
    if not path.exists():
        raise RuntimeError(f"MISSING_DEPENDENCY: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise RuntimeError(f"MISSING_DEPENDENCY: {path}")
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def finite_float(value: Any, name: str) -> float:
    number = float(value)
    if not np.isfinite(number):
        raise RuntimeError(f"INVALID_V51_GEOMETRY: non-finite {name}")
    return number


def safe_ratio(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def distribution(values: list[float] | np.ndarray) -> dict:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return {"count": 0, "min": None, "q25": None, "median": None,
                "q75": None, "max": None, "mean": None}
    return {
        "count": int(array.size),
        "min": float(array.min()),
        "q25": float(np.quantile(array, 0.25)),
        "median": float(np.median(array)),
        "q75": float(np.quantile(array, 0.75)),
        "max": float(array.max()),
        "mean": float(array.mean()),
    }


def rank_percentiles(values: np.ndarray) -> np.ndarray:
    """Average-tie empirical percentile ranks, increasing with value."""
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float64)
    start = 0
    while start < values.size:
        stop = start + 1
        while stop < values.size and values[order[stop]] == values[order[start]]:
            stop += 1
        average_rank = 0.5 * (start + stop - 1) + 1.0
        ranks[order[start:stop]] = average_rank / values.size
        start = stop
    return ranks


def required_result_paths() -> dict[str, Path]:
    return {
        "stage0_design": STAGE0_DIR / "design.json",
        "stage0_domain_summary": STAGE0_DIR / "domain_summary.json",
        "v51_candidate_geometry": V51_DIR / "candidate_geometry.csv",
        "v51_summary": V51_DIR / "summary.json",
    }


def validate_stage0_results(
    result_paths: dict[str, Path], validation: dict, paths: dict[str, Path]
) -> tuple[dict, dict]:
    design = read_json(result_paths["stage0_design"])
    domain = read_json(result_paths["stage0_domain_summary"])
    if design.get("script_version") != "v53_stage0_empirical_domain":
        raise RuntimeError("INVALID_STAGE0_RESULT: script_version")
    if domain.get("status") != "VALID_MAINLINE":
        raise RuntimeError("INVALID_STAGE0_RESULT: status is not VALID_MAINLINE")
    if domain.get("reported_active_count_is_not_ground_truth_K") is not True:
        raise RuntimeError("INVALID_STAGE0_RESULT: interpretation boundary missing")
    count = domain.get("reported_active_count", {}).get(
        "foreground_mean_candidate_count_gt_1e-3"
    )
    if int(count) != EXPECTED_TEMPLATE_COUNT:
        raise RuntimeError(
            f"INVALID_STAGE0_RESULT: expected {EXPECTED_TEMPLATE_COUNT} templates, got {count}"
        )
    stage_hashes = design.get("input_hashes_sha256", {})
    current_hashes = validation.get("hashes_sha256", {})
    for name in (
        "A_library", "B_cube", "candidate_metadata", "channel_axis",
        "production_X_hat", "foreground_mask",
    ):
        if stage_hashes.get(name) != current_hashes.get(name):
            raise RuntimeError(f"STAGE0_RESULT_INPUT_MISMATCH: {name}")
    stage_shapes = design.get("shapes", {})
    if tuple(stage_shapes.get("A_solver", ())) != (1084, EXPECTED_CANDIDATE_COUNT):
        raise RuntimeError("INVALID_STAGE0_RESULT: A_solver shape")
    if Path(paths["production_X_hat"]).name != "X_abundance.npy":
        raise RuntimeError("INVALID_STAGE0_RESULT: production X path resolution")
    return design, domain


def load_locked_arrays(
    paths: dict[str, Path]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict, np.ndarray]:
    A_raw, metadata = v50.load_library_and_metadata(paths)
    A_solver = v52.normalize_like_get_A_matrix(A_raw)
    X_real = np.load(paths["production_X_hat"]).astype(np.float32)
    if X_real.ndim == 4 and X_real.shape[0] == 1:
        X_real = X_real[0]
    mask = np.load(paths["foreground_mask"]).astype(bool)
    if X_real.shape != (EXPECTED_CANDIDATE_COUNT, *mask.shape):
        raise RuntimeError(
            f"DIMENSION_MISMATCH: X_hat {X_real.shape}, mask {mask.shape}"
        )
    if A_solver.shape[1] != X_real.shape[0]:
        raise RuntimeError("DIMENSION_MISMATCH: A_solver and X_hat")
    return A_raw, A_solver, X_real, metadata, mask


def empirical_template_bank(X_real: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    foreground_mean = X_real[:, mask].mean(axis=1, dtype=np.float64)
    indices = np.flatnonzero(foreground_mean > REPORT_GATE)
    if indices.size != EXPECTED_TEMPLATE_COUNT:
        raise RuntimeError(
            "TEMPLATE_COUNT_MISMATCH: foreground_mean(X_hat_j) > 1e-3 "
            f"selected {indices.size}, expected {EXPECTED_TEMPLATE_COUNT}"
        )
    return indices.astype(int), foreground_mean


def load_geometry_rows(path: Path, metadata: dict) -> list[dict]:
    raw = read_csv(path)
    required = {
        "candidate_index", "candidate_id", "lipid_name", "lipid_class",
        "cone_isolation", "max_parent_cosine", "max_fragment_cosine",
        "max_full_cosine",
    }
    if not raw or not required.issubset(raw[0]):
        raise RuntimeError("INVALID_V51_GEOMETRY: required columns missing")
    if len(raw) != EXPECTED_CANDIDATE_COUNT:
        raise RuntimeError(
            f"INVALID_V51_GEOMETRY: expected 391 rows, got {len(raw)}"
        )
    rows: list[dict] = []
    seen: set[int] = set()
    for item in raw:
        index = int(item["candidate_index"])
        if index in seen or not 0 <= index < EXPECTED_CANDIDATE_COUNT:
            raise RuntimeError("INVALID_V51_GEOMETRY: candidate ordering")
        seen.add(index)
        row = {
            "candidate_index": index,
            "candidate_id": str(item["candidate_id"]),
            "lipid_name": str(item["lipid_name"]),
            "lipid_class": str(item["lipid_class"]),
            "cone_isolation": finite_float(item["cone_isolation"], "cone_isolation"),
            "max_parent_cosine": finite_float(item["max_parent_cosine"], "max_parent_cosine"),
            "max_fragment_cosine": finite_float(item["max_fragment_cosine"], "max_fragment_cosine"),
            "max_full_cosine": finite_float(item["max_full_cosine"], "max_full_cosine"),
        }
        for name in ("candidate_id", "lipid_name", "lipid_class"):
            if row[name] != str(metadata[name][index]):
                raise RuntimeError(f"V51_METADATA_MISMATCH: {name} at {index}")
        rows.append(row)
    return sorted(rows, key=lambda row: row["candidate_index"])


def select_truth_identities(geometry_rows: list[dict]) -> tuple[list[dict], dict]:
    """Select 76 unique molecular identities without learned-result feedback.

    The score exists only to stratify this experiment design.  It is the mean
    empirical percentile of three v51 maximum-cosine measures and inverse
    cone isolation; it is not a confidence score or deployment threshold.
    """
    maxima = ("max_parent_cosine", "max_fragment_cosine", "max_full_cosine")
    for metric in maxima:
        ranks = rank_percentiles(np.array([row[metric] for row in geometry_rows]))
        for row, rank in zip(geometry_rows, ranks):
            row[f"_{metric}_rank"] = float(rank)
    inverse_cone = rank_percentiles(
        -np.array([row["cone_isolation"] for row in geometry_rows])
    )
    for row, rank in zip(geometry_rows, inverse_cone):
        row["_inverse_cone_rank"] = float(rank)
        row["geometry_design_score_only"] = float(np.mean([
            row["_max_parent_cosine_rank"],
            row["_max_fragment_cosine_rank"],
            row["_max_full_cosine_rank"],
            row["_inverse_cone_rank"],
        ]))

    # One representative per primary molecular identity.  If the library has
    # multiple candidate/adduct rows for a name, retain its most competitive
    # representative according to frozen geometry alone.
    ordered = sorted(
        geometry_rows,
        key=lambda row: (
            -row["geometry_design_score_only"],
            -row["max_full_cosine"],
            -row["max_fragment_cosine"],
            -row["max_parent_cosine"],
            row["cone_isolation"],
            row["candidate_index"],
        ),
    )
    representatives: list[dict] = []
    names: set[str] = set()
    for row in ordered:
        if row["lipid_name"] not in names:
            representatives.append(dict(row))
            names.add(row["lipid_name"])
    if len(representatives) < EXPECTED_TEMPLATE_COUNT:
        raise RuntimeError(
            f"INSUFFICIENT_UNIQUE_LIPID_NAMES: {len(representatives)}"
        )

    # Divide the unique-name representatives into four equal empirical
    # difficulty strata, then round-robin across stratum and lipid class.
    for stratum, positions in zip(
        DIFFICULTY_STRATA,
        np.array_split(np.arange(len(representatives)), len(DIFFICULTY_STRATA)),
    ):
        for position in positions:
            representatives[int(position)]["difficulty_stratum"] = stratum
    queues: dict[tuple[str, str], list[dict]] = defaultdict(list)
    classes = sorted({row["lipid_class"] for row in representatives})
    for row in representatives:
        queues[(row["difficulty_stratum"], row["lipid_class"])].append(row)
    for queue in queues.values():
        queue.sort(key=lambda row: (-row["geometry_design_score_only"], row["candidate_index"]))

    selected: list[dict] = []
    level = 0
    while len(selected) < EXPECTED_TEMPLATE_COUNT:
        added = 0
        for stratum in DIFFICULTY_STRATA:
            for lipid_class in classes:
                queue = queues[(stratum, lipid_class)]
                if level < len(queue):
                    selected.append(dict(queue[level]))
                    added += 1
                    if len(selected) == EXPECTED_TEMPLATE_COUNT:
                        break
            if len(selected) == EXPECTED_TEMPLATE_COUNT:
                break
        if added == 0:
            raise RuntimeError("TRUTH_SELECTION_FAILED: exhausted deterministic queues")
        level += 1

    if len({row["candidate_index"] for row in selected}) != EXPECTED_TEMPLATE_COUNT:
        raise RuntimeError("TRUTH_SELECTION_FAILED: duplicate candidate")
    if len({row["lipid_name"] for row in selected}) != EXPECTED_TEMPLATE_COUNT:
        raise RuntimeError("TRUTH_SELECTION_FAILED: duplicate lipid_name")
    audit = {
        "rule": (
            "Use v51 candidate geometry only. Choose the most competitive candidate "
            "per unique lipid_name by a design-only mean empirical percentile of "
            "max parent/fragment/full cosine and inverse cone isolation; split those "
            "representatives into four equal empirical difficulty strata; then "
            "round-robin deterministically across difficulty stratum and lipid_class."
        ),
        "score_is_confidence_threshold": False,
        "unique_library_lipid_names": len(representatives),
        "selected_count": len(selected),
        "selected_unique_lipid_names": len({row["lipid_name"] for row in selected}),
        "selected_class_counts": dict(sorted(
            (name, sum(row["lipid_class"] == name for row in selected))
            for name in {row["lipid_class"] for row in selected}
        )),
        "selected_stratum_counts": {
            name: sum(row["difficulty_stratum"] == name for row in selected)
            for name in DIFFICULTY_STRATA
        },
        "library_max_full_cosine": distribution(
            [row["max_full_cosine"] for row in geometry_rows]
        ),
        "selected_max_full_cosine": distribution(
            [row["max_full_cosine"] for row in selected]
        ),
        "library_cone_isolation": distribution(
            [row["cone_isolation"] for row in geometry_rows]
        ),
        "selected_cone_isolation": distribution(
            [row["cone_isolation"] for row in selected]
        ),
    }
    return selected, audit


def build_truth_mapping(
    template_indices: np.ndarray,
    foreground_mean: np.ndarray,
    selected: list[dict],
    metadata: dict,
) -> list[dict]:
    if math.gcd(PERMUTATION_MULTIPLIER, EXPECTED_TEMPLATE_COUNT) != 1:
        raise RuntimeError("INVALID_FIXED_PERMUTATION: multiplier is not coprime")
    permutation = [
        (PERMUTATION_MULTIPLIER * k + PERMUTATION_OFFSET)
        % EXPECTED_TEMPLATE_COUNT
        for k in range(EXPECTED_TEMPLATE_COUNT)
    ]
    if len(set(permutation)) != EXPECTED_TEMPLATE_COUNT:
        raise RuntimeError("INVALID_FIXED_PERMUTATION: not bijective")
    mapping = []
    for map_slot, (template_index, truth_position) in enumerate(
        zip(template_indices.tolist(), permutation)
    ):
        truth = selected[truth_position]
        mapping.append({
            "mapping_slot": map_slot,
            "template_original_candidate_index": int(template_index),
            "template_original_candidate_id_provenance_only": str(metadata["candidate_id"][template_index]),
            "template_original_lipid_name_discarded": str(metadata["lipid_name"][template_index]),
            "template_foreground_mean": float(foreground_mean[template_index]),
            "truth_selection_position": int(truth_position),
            "synthetic_truth_candidate_index": int(truth["candidate_index"]),
            "synthetic_truth_candidate_id": truth["candidate_id"],
            "synthetic_truth_lipid_name": truth["lipid_name"],
            "synthetic_truth_lipid_class": truth["lipid_class"],
            "difficulty_stratum": truth["difficulty_stratum"],
            "geometry_design_score_only": truth["geometry_design_score_only"],
            "cone_isolation": truth["cone_isolation"],
            "max_parent_cosine": truth["max_parent_cosine"],
            "max_fragment_cosine": truth["max_fragment_cosine"],
            "max_full_cosine": truth["max_full_cosine"],
        })
    if len({row["synthetic_truth_lipid_name"] for row in mapping}) != EXPECTED_TEMPLATE_COUNT:
        raise RuntimeError("TRUTH_MAPPING_FAILED: molecular truth is not unique")
    return mapping


def construct_clean_case(
    A_solver: np.ndarray,
    X_real: np.ndarray,
    mask: np.ndarray,
    mapping: list[dict],
) -> dict:
    X_true = np.zeros((A_solver.shape[1], *mask.shape), dtype=np.float32)
    for row in mapping:
        source = row["template_original_candidate_index"]
        target = row["synthetic_truth_candidate_index"]
        X_true[target] = X_real[source]
    B_sim = np.einsum("mc,cyx->myx", A_solver, X_true, optimize=True)
    B_check = (A_solver @ X_true.reshape(A_solver.shape[1], -1)).reshape(B_sim.shape)
    forward_max_abs = float(np.max(np.abs(B_sim - B_check)))
    forward_relative_l2 = float(
        np.linalg.norm(B_sim.astype(np.float64) - B_check.astype(np.float64))
        / max(np.linalg.norm(B_sim.astype(np.float64)), 1.0e-30)
    )
    if not np.allclose(B_sim, B_check, rtol=2.0e-6, atol=1.0e-7):
        raise RuntimeError("FORWARD_CONSISTENCY_FAILED")
    achieved = float(np.median(np.linalg.norm(B_sim[:, mask], axis=0)))
    return {
        "case_name": VERSION,
        "active_indices": [row["synthetic_truth_candidate_index"] for row in mapping],
        "X_true": X_true,
        "B_sim": B_sim.astype(np.float32, copy=False),
        "foreground_mask": mask,
        "global_scale": 1.0,
        "target_foreground_B_l2_p50": achieved,
        "achieved_foreground_B_l2_p50": achieved,
        "forward_consistency_max_abs": forward_max_abs,
        "forward_consistency_relative_l2": forward_relative_l2,
    }


def molecular_metrics(recovered: np.ndarray, truth_indices: list[int], metadata: dict, gate: float) -> dict:
    reported = np.flatnonzero(recovered > gate).tolist()
    truth_candidates = set(map(int, truth_indices))
    truth_names = {str(metadata["lipid_name"][j]) for j in truth_candidates}
    reported_names = {str(metadata["lipid_name"][j]) for j in reported}
    candidate_tp = len(set(reported) & truth_candidates)
    candidate_fp = len(set(reported) - truth_candidates)
    candidate_fn = len(truth_candidates - set(reported))
    molecular_tp = len(reported_names & truth_names)
    molecular_fp = len(reported_names - truth_names)
    molecular_fn = len(truth_names - reported_names)
    return {
        "gate": gate,
        "reported_candidate_indices": reported,
        "candidate_level": {
            "TP": candidate_tp, "FP": candidate_fp, "FN": candidate_fn,
            "precision": safe_ratio(candidate_tp, candidate_tp + candidate_fp),
            "recall": safe_ratio(candidate_tp, candidate_tp + candidate_fn),
            "FDR": safe_ratio(candidate_fp, candidate_tp + candidate_fp),
        },
        "molecular_level": {
            "TP": molecular_tp, "FP": molecular_fp, "FN": molecular_fn,
            "precision": safe_ratio(molecular_tp, molecular_tp + molecular_fp),
            "recall": safe_ratio(molecular_tp, molecular_tp + molecular_fn),
            "FDR": safe_ratio(molecular_fp, molecular_tp + molecular_fp),
        },
    }


def run_oracle(case: dict, A_solver: np.ndarray, metadata: dict) -> dict:
    base = v50.oracle_case(case, A_solver, metadata)
    mean_spectrum = case["B_sim"][:, case["foreground_mask"]].mean(axis=1).astype(np.float64)
    recovered, residual = nnls(A_solver.astype(np.float64), mean_spectrum,
                               maxiter=10 * A_solver.shape[1])
    metrics = molecular_metrics(recovered, case["active_indices"], metadata, REPORT_GATE)
    molecular = metrics["molecular_level"]
    clean = molecular["FP"] == 0 and molecular["FN"] == 0
    return {
        "status": "PASS" if clean else "SERIOUS_MOLECULAR_AMBIGUITY",
        "scope": "foreground-mean exact full-library NNLS clean-construction sanity check",
        "reporting_gate": REPORT_GATE,
        "reconstruction_relative_residual": float(
            residual / max(np.linalg.norm(mean_spectrum), 1.0e-30)
        ),
        "candidate_and_molecular_metrics": metrics,
        "candidate_level_v50_oracle": base,
        "truth_set_redesigned_from_oracle": False,
    }


def identity_records_and_report(
    case: dict,
    learned: dict,
    A_solver: np.ndarray,
    metadata: dict,
) -> tuple[list[dict], dict]:
    mask = case["foreground_mask"]
    xhat = learned["X_hat"][:, mask].mean(axis=1, dtype=np.float64)
    truth_candidates = set(map(int, case["active_indices"]))
    truth_names = {str(metadata["lipid_name"][j]) for j in truth_candidates}
    reported = np.flatnonzero(xhat > REPORT_GATE)
    mean_spectrum = case["B_sim"][:, mask].mean(axis=1).astype(np.float64)
    weights = identity_weights(A_solver)
    records: list[dict] = []
    for index in reported:
        index = int(index)
        name = str(metadata["lipid_name"][index])
        candidate_truth = index in truth_candidates
        molecular_truth = name in truth_names
        certificate = rho_zero_from_weighted_case(
            A_solver, mean_spectrum, index, weights=weights
        )
        records.append({
            "candidate_index": index,
            "candidate_id": str(metadata["candidate_id"][index]),
            "lipid_name": name,
            "X_hat": float(xhat[index]),
            "X_hat_foreground_mean": float(xhat[index]),
            "molecular_truth": bool(molecular_truth),
            "candidate_truth": bool(candidate_truth),
            "same_lipid_alternative_candidate": bool(molecular_truth and not candidate_truth),
            "rho_zero": float(certificate["rho_zero"]),
        })
    records.sort(key=lambda row: row["candidate_index"])

    metrics = molecular_metrics(xhat, case["active_indices"], metadata, REPORT_GATE)
    false_molecular = np.array(
        [str(metadata["lipid_name"][j]) not in truth_names for j in range(xhat.size)],
        dtype=bool,
    )
    reported_mask = xhat > REPORT_GATE
    total_mass = float(np.sum(xhat))
    reported_mass = float(np.sum(xhat[reported_mask]))
    residual = float(
        np.linalg.norm(learned["B_hat"][:, mask] - case["B_sim"][:, mask])
        / max(np.linalg.norm(case["B_sim"][:, mask]), 1.0e-30)
    )
    all_false_mass = float(np.sum(xhat[false_molecular]))
    reported_false_mass = float(np.sum(xhat[reported_mask & false_molecular]))
    report = {
        "status": "PENDING_REVIEW",
        "reporting_gate": REPORT_GATE,
        "truth_molecular_identity_count": len(truth_names),
        "reported_identity_count": len(records),
        "candidate_level": metrics["candidate_level"],
        "molecular_level": metrics["molecular_level"],
        "n_outputs_gt_1e-4": int(np.sum(xhat > SENSITIVITY_GATE)),
        "n_outputs_gt_1e-3": int(np.sum(reported_mask)),
        "reconstruction_relative_residual": residual,
        "all_output_false_molecular_identity_mass": all_false_mass,
        "all_output_false_molecular_identity_mass_fraction": safe_ratio(
            all_false_mass, total_mass
        ),
        "reported_only_false_molecular_identity_mass": reported_false_mass,
        "reported_only_false_molecular_identity_mass_fraction": safe_ratio(
            reported_false_mass, reported_mass
        ),
        "training": {
            "final_training_epoch": learned["stopped_epoch"],
            "stop_reason": learned["stop_reason"],
            "wall_time_seconds": learned["wall_time_seconds"],
            "final_physical_loss": learned["final_physical_loss"],
            "final_raw_losses": learned["final_raw_losses"],
            "channel_weighting": learned["channel_weighting"],
            "checkpoint_loaded": False,
            "X_true_used_in_training_or_stopping": False,
        },
    }
    return records, report


def threshold_curve(records: list[dict], truth_names: set[str]) -> list[dict]:
    if not records:
        return []
    thresholds = sorted({float(row["rho_zero"]) for row in records}, reverse=True)
    rows = []
    for threshold in thresholds:
        retained = [row for row in records if float(row["rho_zero"]) >= threshold]
        retained_names = {str(row["lipid_name"]) for row in retained}
        tp = len(retained_names & truth_names)
        fp = len(retained_names - truth_names)
        rows.append({
            "rho_zero_threshold": threshold,
            "n_retained": len(retained),
            "molecular_TP": tp,
            "molecular_FP": fp,
            "precision": safe_ratio(tp, tp + fp),
            "FDR": safe_ratio(fp, tp + fp),
            "recall": safe_ratio(tp, len(truth_names)),
            "coverage_fraction": safe_ratio(len(retained), len(records)),
        })
    return rows


def descriptive_operating_points(curve: list[dict], limit: int = 5) -> list[dict]:
    # Return Pareto-improving precision/recall descriptions only.  This does
    # not nominate or freeze a threshold.
    frontier = []
    for row in curve:
        dominated = any(
            other["precision"] >= row["precision"]
            and other["recall"] >= row["recall"]
            and (other["precision"] > row["precision"] or other["recall"] > row["recall"])
            for other in curve
        )
        if not dominated:
            frontier.append(row)
    frontier.sort(key=lambda row: (-row["precision"], -row["recall"], -row["rho_zero_threshold"]))
    return frontier[:limit]


def design_payload(
    args: argparse.Namespace,
    paths: dict[str, Path],
    validation: dict,
    stage0_design: dict,
    v51_summary: dict,
    mapping: list[dict],
    selection_audit: dict,
) -> dict:
    return {
        "script_version": VERSION,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "clean real-complexity synthetic molecular-identity rho_zero calibration pilot",
        "input_paths": {
            **{name: str(path) for name, path in paths.items()},
            **{name: str(path) for name, path in required_result_paths().items()},
        },
        "input_hashes_sha256": validation["hashes_sha256"],
        "stage0_script_version": stage0_design.get("script_version"),
        "v51_status": v51_summary.get("status"),
        "A_solver_contract": (
            "A_raw float32; column L2 normalization with denominator norm+1e-8, "
            "using v52.normalize_like_get_A_matrix"
        ),
        "template_bank": {
            "source": "locked real production X_hat",
            "selection": "foreground_mean(X_hat_j over locked foreground) > 1e-3",
            "count": len(mapping),
            "values_used_as": "full 2D spatial/amplitude templates only",
            "individual_map_rescaling": False,
            "original_lipid_identities_define_truth": False,
        },
        "truth_selection": selection_audit,
        "mapping_permutation": {
            "formula": "truth_position = (17 * mapping_slot + 11) mod 76",
            "multiplier": PERMUTATION_MULTIPLIER,
            "offset": PERMUTATION_OFFSET,
            "bijective": True,
        },
        "truth_candidate_indices": [row["synthetic_truth_candidate_index"] for row in mapping],
        "truth_lipid_names": [row["synthetic_truth_lipid_name"] for row in mapping],
        "synthetic_B": {
            "definition": "B_sim = A_solver @ X_true",
            "clean": True,
            "empirical_residual_added": False,
            "Gaussian_noise_added": False,
            "extra_model_mismatch_added": False,
        },
        "solver": {
            "implementation": "v50.train_fresh_case",
            "production_training_contract": v50.training_contract(),
            "fresh_network": True,
            "real_trained_checkpoint_loaded": False,
            "production_hyperparameters_changed": False,
            "X_true_used_in_training_or_stopping": False,
            "device": "cuda",
        },
        "rho_zero": {
            "implementation": "src/rho_zero.py:rho_zero_from_weighted_case",
            "weighting": "W=I",
            "case_vector": "foreground-mean clean B_sim spectrum",
            "scope": "every learned output reported above 1e-3",
            "final_threshold_frozen": False,
        },
        "cli_mode": "oracle-only" if args.oracle_only else "full",
        "validity_status": "PENDING_REVIEW",
        "next_step": "PENDING_RESULT_REVIEW",
    }


def build_summary_md(report: dict, curve: list[dict]) -> str:
    result = report.get("learned_result")
    if not result:
        return "\n".join([
            "# v53 Template-Truth Identity Calibration Pilot", "",
            "## Purpose", "", "Test molecular truth consistency before learned training.", "",
            "## Truth construction", "",
            "Real production abundance maps were reused only as spatial/amplitude templates; their original lipid identities were discarded.", "",
            "## Solver contract", "", "Oracle-only mode; no learned solver was run.", "",
            "## Results", "", f"Oracle status: {report['oracle']['status']}.", "",
            "## Interpretation", "", "Learned rho_zero enrichment is not assessed in oracle-only mode.", "",
            "## Limitations", "",
            "- Template maps originate from production X_hat and are not biological truth.",
            "- This first pilot is clean and contains no empirical residual.",
            "- Only one identity remapping is tested.",
            "- No final rho_zero threshold is calibrated here.", "",
            "## Validity status", "", "PENDING_REVIEW", "",
            "## Next step", "", "PENDING_RESULT_REVIEW", "",
        ])
    true_rho = report["rho_zero_distributions"]["true_reported_molecular_identities"]
    false_rho = report["rho_zero_distributions"]["false_reported_molecular_identities"]
    molecular = result["molecular_level"]
    points = report["best_descriptive_high_rho_operating_points"]
    point_lines = [
        f"- rho≥{row['rho_zero_threshold']:.8g}: retained={row['n_retained']}, "
        f"precision={row['precision']:.6g}, FDR={row['FDR']:.6g}, recall={row['recall']:.6g}."
        for row in points
    ] or ["- No reported candidates; no descriptive operating point exists."]
    return "\n".join([
        "# v53 Template-Truth Identity Calibration Pilot", "",
        "## Purpose", "",
        "Test whether rho_zero separates true from false molecular identities in one clean, real-complexity synthetic dataset.", "",
        "## Truth construction", "",
        "Real production abundance maps were reused only as spatial/amplitude templates; their original lipid identities were discarded.",
        f"The frozen remapping contains {result['truth_molecular_identity_count']} unique synthetic molecular identities.", "",
        "## Solver contract", "",
        "A completely fresh 12-layer production LipidENNet trajectory used the validated v50 workflow, unchanged loss, optimizer, scheduler, channel weighting, and early stopping. X_true was not used in training or stopping, and no real-data checkpoint was loaded.", "",
        "## Results", "",
        f"- Truth molecular identity count: {result['truth_molecular_identity_count']}.",
        f"- Reported identity count (>1e-3): {result['reported_identity_count']}.",
        f"- Molecular TP/FP/FN: {molecular['TP']}/{molecular['FP']}/{molecular['FN']}.",
        f"- Molecular precision/FDR/recall: {molecular['precision']:.8g}/{molecular['FDR']:.8g}/{molecular['recall']:.8g}.",
        f"- Reconstruction relative residual: {result['reconstruction_relative_residual']:.8g}.",
        f"- rho_zero, true reported molecular identities: {json.dumps(true_rho, ensure_ascii=False)}.",
        f"- rho_zero, false reported molecular identities: {json.dumps(false_rho, ensure_ascii=False)}.",
        "- Best descriptive high-rho operating points (not threshold selection):", *point_lines, "",
        "## Interpretation", "",
        report["interpretation"], "",
        "No final threshold is claimed.", "",
        "## Limitations", "",
        "- Template maps originate from production X_hat and are not biological truth.",
        "- This first pilot is clean and contains no empirical residual.",
        "- Only one identity remapping is tested.",
        "- No final rho_zero threshold is calibrated here.", "",
        "## Validity status", "", "PENDING_REVIEW", "",
        "## Next step", "", "PENDING_RESULT_REVIEW", "",
    ])


def experiment_log_entry(design: dict, report: dict) -> str:
    result = report["learned_result"]
    molecular = result["molecular_level"]
    hashes = design["input_hashes_sha256"]
    return "\n".join([
        LOG_BEGIN,
        "## v53 Template-Truth Identity Calibration Pilot", "",
        f"- Version: {VERSION}",
        f"- Date: {date.today().isoformat()}",
        "- Purpose: direct clean-pilot test of rho_zero separation for known molecular truth using real-complexity abundance-map templates.",
        "- Inputs/hashes: " + "; ".join(f"{name}={value}" for name, value in sorted(hashes.items())),
        "- Truth construction: 76 locked production X_hat maps above foreground mean 1e-3 used only as unscaled spatial/amplitude templates; original identities discarded; deterministically remapped to 76 unique lipid_name truths selected from frozen v51 geometry.",
        "- Experimental design: one clean B_sim=A_solver@X_true case; exact full-library NNLS oracle; fresh unchanged production LipidENNet; W=I rho_zero for every output above 1e-3.",
        f"- Key results: reported={result['reported_identity_count']}; molecular TP/FP/FN={molecular['TP']}/{molecular['FP']}/{molecular['FN']}; precision/FDR/recall={molecular['precision']:.8g}/{molecular['FDR']:.8g}/{molecular['recall']:.8g}; reconstruction relative residual={result['reconstruction_relative_residual']:.8g}.",
        "- Validity status: PENDING_REVIEW",
        "- Limitations: production-X_hat templates are not biological truth; clean case without empirical residual; one remapping only; no final rho_zero threshold calibrated.",
        "- Decision: PENDING_RESULT_REVIEW",
        "- Next step: PENDING_RESULT_REVIEW",
        "- Git commit: PENDING_USER_COMMIT", "",
        LOG_END,
    ])


def update_experiment_log(entry: str) -> None:
    current = EXPERIMENT_LOG.read_text(encoding="utf-8") if EXPERIMENT_LOG.exists() else "# Experiment Log\n"
    if LOG_BEGIN in current or LOG_END in current:
        start = current.find(LOG_BEGIN)
        end = current.find(LOG_END)
        if start < 0 or end < start:
            raise RuntimeError("INVALID_EXPERIMENT_LOG_MARKERS")
        end += len(LOG_END)
        updated = current[:start].rstrip() + "\n\n" + entry + current[end:]
    else:
        updated = current.rstrip() + "\n\n" + entry + "\n"
    stage0.atomic_write_text(EXPERIMENT_LOG, updated.rstrip() + "\n")


def prepare(args: argparse.Namespace) -> dict:
    lock = stage0.load_lock()
    paths = stage0.resolve_locked_paths(args.asset_root.resolve(), lock)
    missing_assets = stage0.missing_paths(paths)
    if missing_assets:
        raise RuntimeError("MISSING_LOCKED_PRODUCTION_ASSETS: " + "; ".join(missing_assets))
    validation, provenance = stage0.validate_shapes_and_hashes(paths, lock)
    A_raw, A_solver, X_real, metadata, mask = load_locked_arrays(paths)
    template_indices, foreground_mean = empirical_template_bank(X_real, mask)
    result_paths = required_result_paths()
    missing_results = [str(path) for path in result_paths.values() if not path.exists()]
    if missing_results:
        return {
            "paths": paths, "validation": validation, "provenance": provenance,
            "A_raw": A_raw, "A_solver": A_solver, "X_real": X_real, "metadata": metadata,
            "mask": mask, "template_indices": template_indices,
            "foreground_mean": foreground_mean, "missing_results": missing_results,
        }
    stage0_design, stage0_domain = validate_stage0_results(
        result_paths, validation, paths
    )
    v51_summary = read_json(result_paths["v51_summary"])
    if v51_summary.get("status") != "COMPLETE":
        raise RuntimeError("INVALID_V51_RESULT: status is not COMPLETE")
    if int(v51_summary.get("candidate_count", -1)) != EXPECTED_CANDIDATE_COUNT:
        raise RuntimeError("INVALID_V51_RESULT: candidate count")
    v51_hashes = v51_summary.get("asset_validation", {}).get("validated_hashes", {})
    for name in ("A_library", "candidate_metadata", "channel_axis"):
        if v51_hashes.get(name) != validation["hashes_sha256"].get(name):
            raise RuntimeError(f"V51_RESULT_INPUT_MISMATCH: {name}")
    geometry = load_geometry_rows(result_paths["v51_candidate_geometry"], metadata)
    selected, selection_audit = select_truth_identities(geometry)
    mapping = build_truth_mapping(template_indices, foreground_mean, selected, metadata)
    return {
        "paths": paths, "validation": validation, "provenance": provenance,
        "A_raw": A_raw, "A_solver": A_solver, "X_real": X_real, "metadata": metadata,
        "mask": mask, "template_indices": template_indices,
        "foreground_mean": foreground_mean, "missing_results": [],
        "stage0_design": stage0_design, "stage0_domain": stage0_domain,
        "v51_summary": v51_summary, "mapping": mapping,
        "selection_audit": selection_audit,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-root", type=Path, default=ROOT.parent / "decon-lipid")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DEFAULT)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--oracle-only", action="store_true")
    args = parser.parse_args()
    if args.dry_run and args.oracle_only:
        parser.error("--dry-run and --oracle-only are mutually exclusive")
    return args


def main() -> None:
    args = parse_args()
    args.output_dir = args.output_dir.resolve()
    prepared = prepare(args)
    if prepared["missing_results"]:
        payload = {
            "status": "DRY_RUN_BLOCKED_REQUIRED_RESULTS_MISSING",
            "empirical_template_count": int(len(prepared["template_indices"])),
            "unique_synthetic_molecular_identity_count": EXPECTED_TEMPLATE_COUNT,
            "missing_required_results": prepared["missing_results"],
            "required_action": "materialize the completed v53 Stage0 and v51 lightweight outputs; do not recompute geometry here",
            "outputs_written": False,
            "experiment_log_modified": False,
        }
        if args.dry_run:
            print(json.dumps(payload, ensure_ascii=False, indent=2))
            return
        raise RuntimeError(
            "MISSING_REQUIRED_RESULTS: " + "; ".join(prepared["missing_results"])
        )

    design = design_payload(
        args, prepared["paths"], prepared["validation"],
        prepared["stage0_design"], prepared["v51_summary"],
        prepared["mapping"], prepared["selection_audit"],
    )
    if args.dry_run:
        print(json.dumps(stage0.to_jsonable({
            "status": "DRY_RUN_READY",
            "design": design,
            "frozen_truth_mapping": prepared["mapping"],
            "empirical_template_count": len(prepared["mapping"]),
            "unique_synthetic_molecular_identity_count": len({
                row["synthetic_truth_lipid_name"] for row in prepared["mapping"]
            }),
            "training_performed": False,
            "outputs_written": False,
            "experiment_log_modified": False,
        }), ensure_ascii=False, indent=2, allow_nan=False))
        return

    case = construct_clean_case(
        prepared["A_solver"], prepared["X_real"], prepared["mask"], prepared["mapping"]
    )
    design["synthetic_B"]["forward_consistency_max_abs"] = case["forward_consistency_max_abs"]
    design["synthetic_B"]["forward_consistency_relative_l2"] = case["forward_consistency_relative_l2"]
    oracle = run_oracle(case, prepared["A_solver"], prepared["metadata"])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stage0.atomic_write_json(args.output_dir / "design.json", design)
    write_csv(args.output_dir / "truth_mapping.csv", prepared["mapping"], list(prepared["mapping"][0]))
    if oracle["status"] != "PASS":
        report = {
            "status": "STOPPED_ORACLE_MOLECULAR_AMBIGUITY",
            "oracle": oracle,
            "learned_training_performed": False,
            "validity_status": "PENDING_REVIEW",
            "next_step": "PENDING_RESULT_REVIEW",
        }
        stage0.atomic_write_json(args.output_dir / "report.json", report)
        stage0.atomic_write_text(args.output_dir / "summary.md", build_summary_md(report, []))
        print(json.dumps(stage0.to_jsonable(report), ensure_ascii=False, indent=2, allow_nan=False))
        return
    if args.oracle_only:
        report = {
            "status": "ORACLE_ONLY_PASS",
            "oracle": oracle,
            "learned_training_performed": False,
            "validity_status": "PENDING_REVIEW",
            "next_step": "PENDING_RESULT_REVIEW",
        }
        stage0.atomic_write_json(args.output_dir / "report.json", report)
        stage0.atomic_write_text(args.output_dir / "summary.md", build_summary_md(report, []))
        print(json.dumps(stage0.to_jsonable(report), ensure_ascii=False, indent=2, allow_nan=False))
        return

    runtime_dir = args.output_dir / ".runtime_training"
    learned = v50.train_fresh_case(
        case, prepared["A_raw"], prepared["metadata"], prepared["paths"],
        runtime_dir, "cuda",
    )
    history_source = runtime_dir / "training_history.json"
    if not history_source.exists():
        raise RuntimeError("TRAINING_OUTPUT_MISSING: training_history.json")
    shutil.copyfile(history_source, args.output_dir / "training_history.json")
    records, learned_report = identity_records_and_report(
        case, learned, prepared["A_solver"], prepared["metadata"]
    )
    truth_names = {row["synthetic_truth_lipid_name"] for row in prepared["mapping"]}
    curve = threshold_curve(records, truth_names)
    true_rho = [row["rho_zero"] for row in records if row["molecular_truth"]]
    false_rho = [row["rho_zero"] for row in records if not row["molecular_truth"]]
    true_median = distribution(true_rho)["median"]
    false_median = distribution(false_rho)["median"]
    if true_median is None or false_median is None:
        interpretation = (
            "This single pilot does not contain both true and false reported molecular identities, "
            "so rho_zero enrichment cannot yet be judged."
        )
    elif true_median > false_median:
        interpretation = (
            "In this clean pilot, higher rho_zero appears to enrich for correct molecular identities "
            "descriptively; replication is required before any threshold is considered."
        )
    else:
        interpretation = (
            "In this clean pilot, rho_zero does not show a positive median separation for correct "
            "molecular identities."
        )
    report = {
        "status": "PENDING_REVIEW",
        "oracle": oracle,
        "learned_result": learned_report,
        "rho_zero_distributions": {
            "true_reported_molecular_identities": distribution(true_rho),
            "false_reported_molecular_identities": distribution(false_rho),
        },
        "best_descriptive_high_rho_operating_points": descriptive_operating_points(curve),
        "interpretation": interpretation,
        "final_rho_zero_threshold_frozen": False,
        "validity_status": "PENDING_REVIEW",
        "next_step": "PENDING_RESULT_REVIEW",
    }
    identity_fields = [
        "candidate_index", "candidate_id", "lipid_name", "X_hat",
        "X_hat_foreground_mean", "molecular_truth", "candidate_truth",
        "same_lipid_alternative_candidate", "rho_zero",
    ]
    curve_fields = [
        "rho_zero_threshold", "n_retained", "molecular_TP", "molecular_FP",
        "precision", "FDR", "recall", "coverage_fraction",
    ]
    write_csv(args.output_dir / "reported_identity_records.csv", records, identity_fields)
    write_csv(args.output_dir / "rho_threshold_curve.csv", curve, curve_fields)
    stage0.atomic_write_json(args.output_dir / "report.json", report)
    stage0.atomic_write_text(args.output_dir / "summary.md", build_summary_md(report, curve))
    update_experiment_log(experiment_log_entry(design, report))
    # The reused trainer necessarily emits a checkpoint.  It is a runtime-only
    # intermediate and is removed after successful lightweight outputs exist.
    shutil.rmtree(runtime_dir)
    print(json.dumps(stage0.to_jsonable({
        "status": "PENDING_REVIEW",
        "output_dir": str(args.output_dir),
        "empirical_template_count": EXPECTED_TEMPLATE_COUNT,
        "unique_synthetic_molecular_identity_count": EXPECTED_TEMPLATE_COUNT,
        "reported_identity_count": len(records),
        "experiment_log": str(EXPERIMENT_LOG),
        "runtime_checkpoint_retained": False,
        "next_step": "PENDING_RESULT_REVIEW",
    }), ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
