#!/usr/bin/env python3
"""Run frozen K_true=1 mechanism-specific leakage tests with fresh ISTA fits.

The full 391-candidate solver library is used in every case.  Training is
strictly one case at a time and delegates to the validated v50 implementation.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
ANALYSIS = ROOT / "analysis"
if str(ANALYSIS) not in sys.path:
    sys.path.insert(0, str(ANALYSIS))

import run_v50_reoptimized_sanity as v50


OUTPUT_DEFAULT = ROOT / "results/v52_mechanism_leakage"
CONE_CACHE = ROOT / "results/v50_reoptimized_sanity_k1_k3/design.json"
PARENT_MZ_RANGE = (748.0, 803.0)
REPORT_GATE = 1.0e-3
SENSITIVITY_GATE = 1.0e-4

FROZEN_CASES = (
    {"case_name": "A_0_true", "mechanism": "A_control", "true_index": 0, "competitor_index": 132},
    {"case_name": "A_132_true", "mechanism": "A_control", "true_index": 132, "competitor_index": 0},
    {"case_name": "B_260_true", "mechanism": "B_parent_only_competition", "true_index": 260, "competitor_index": 266},
    {"case_name": "B_266_true", "mechanism": "B_parent_only_competition", "true_index": 266, "competitor_index": 260},
    {"case_name": "C1_177_true", "mechanism": "C1_same_class_cross_parent_fragment_competition", "true_index": 177, "competitor_index": 249},
    {"case_name": "C1_249_true", "mechanism": "C1_same_class_cross_parent_fragment_competition", "true_index": 249, "competitor_index": 177},
    {"case_name": "C2_248_true", "mechanism": "C2_cross_class_fragment_competition", "true_index": 248, "competitor_index": 287},
    {"case_name": "C2_287_true", "mechanism": "C2_cross_class_fragment_competition", "true_index": 287, "competitor_index": 248},
    {"case_name": "D_78_true", "mechanism": "D_joint_parent_fragment_competition", "true_index": 78, "competitor_index": 313},
    {"case_name": "D_313_true", "mechanism": "D_joint_parent_fragment_competition", "true_index": 313, "competitor_index": 78},
    {"case_name": "E_292_true", "mechanism": "E_collective_approximability", "true_index": 292, "competitor_index": None, "frozen_best_single_residual": 0.780578, "frozen_collective_factor": 4.216},
    {"case_name": "E_296_true", "mechanism": "E_collective_approximability", "true_index": 296, "competitor_index": None, "frozen_best_single_residual": 0.773605, "frozen_collective_factor": 4.336},
)
CASE_NAMES = tuple(row["case_name"] for row in FROZEN_CASES)


def normalize_like_get_A_matrix(A_raw: np.ndarray) -> np.ndarray:
    """Match utils.get_A_matrix: float32 followed by column L2 normalization."""
    import torch

    tensor = torch.as_tensor(A_raw, dtype=torch.float32, device="cpu")
    normalized = tensor / (
        torch.linalg.vector_norm(tensor, dim=0, keepdim=True) + 1.0e-8
    )
    return normalized.numpy().copy()


def cosine(left: np.ndarray, right: np.ndarray) -> float:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator <= 0:
        return 0.0
    return float(np.clip(np.dot(left, right) / denominator, -1.0, 1.0))


def parent_centroid(column: np.ndarray, mz: np.ndarray, mask: np.ndarray) -> float | None:
    weights = np.abs(column[mask])
    total = float(weights.sum())
    if total <= 0:
        return None
    return float(np.dot(mz[mask], weights) / total)


def load_cone_cache(candidate_count: int) -> tuple[np.ndarray, dict[int, list]]:
    if not CONE_CACHE.exists():
        raise RuntimeError(f"MISSING_DEPENDENCY: {CONE_CACHE}")
    design = json.loads(CONE_CACHE.read_text(encoding="utf-8"))
    rows = design.get("cone_isolation_ranking", [])
    if len(rows) != candidate_count:
        raise RuntimeError(
            "MISSING_DEPENDENCY: complete cached v50 cone-isolation values required"
        )
    scores = np.full(candidate_count, np.nan, dtype=np.float64)
    contributors = {}
    for row in rows:
        index = int(row["candidate_index"])
        if index < 0 or index >= candidate_count or np.isfinite(scores[index]):
            raise RuntimeError("INVALID_CACHED_DESIGN: cone-isolation ranking")
        scores[index] = float(row["cone_isolation"])
        cached = row.get("top_nonnegative_contributors")
        if cached:
            contributors[index] = cached
    if not np.all(np.isfinite(scores)):
        raise RuntimeError("INVALID_CACHED_DESIGN: incomplete cone-isolation map")
    return scores, contributors


def candidate_record(index: int, metadata: dict) -> dict:
    return {
        "candidate_index": int(index),
        "candidate_id": str(metadata["candidate_id"][index]),
        "lipid_name": str(metadata["lipid_name"][index]),
        "lipid_class": str(metadata["lipid_class"][index]),
    }


def build_design(
    A_solver: np.ndarray,
    mz: np.ndarray,
    metadata: dict,
    cone: np.ndarray,
    contributors: dict[int, list],
    validation: dict,
) -> dict:
    parent = (mz >= PARENT_MZ_RANGE[0]) & (mz <= PARENT_MZ_RANGE[1])
    fragment = ~parent
    cases = []
    for frozen in FROZEN_CASES:
        true_index = int(frozen["true_index"])
        competitor = frozen["competitor_index"]
        row = {
            **frozen,
            "K_true": 1,
            "true_candidate": candidate_record(true_index, metadata),
            "known_nnls_contributor_indices": contributors.get(true_index, []),
        }
        if competitor is not None:
            competitor = int(competitor)
            left, right = A_solver[:, true_index], A_solver[:, competitor]
            left_centroid = parent_centroid(left, mz, parent)
            right_centroid = parent_centroid(right, mz, parent)
            row.update({
                "competitor_candidate": candidate_record(competitor, metadata),
                "parent_cosine": cosine(left[parent], right[parent]),
                "fragment_cosine": cosine(left[fragment], right[fragment]),
                "full_cosine": cosine(left, right),
                "parent_centroid_separation_da": (
                    abs(left_centroid - right_centroid)
                    if left_centroid is not None and right_centroid is not None
                    else None
                ),
            })
        else:
            row.update({
                "cone_isolation": float(cone[true_index]),
                "cached_nnls_contributors_available": true_index in contributors,
            })
        cases.append(row)
    return {
        "status": "DESIGN_READY",
        "case_count": len(cases),
        "cases": cases,
        "A_solver_contract": {
            "source": "frozen A_library.npy",
            "normalization": "float32 column L2 normalization exactly matching utils.get_A_matrix: A / (||A_j||_2 + 1e-8)",
            "synthesis": "B_sim = A_solver @ X_true",
            "oracle": "exact scipy NNLS against A_solver",
            "training": "v50 train_fresh_case receives frozen raw A and internally constructs the same A_solver before network creation",
        },
        "identity_definitions": {
            "candidate_level": "candidate j is reported iff foreground-mean X_hat[j] > 1e-3; only the designated true candidate is candidate-level truth",
            "lipid_name_level": "map candidate-level reports to unique metadata lipid_name values; the designated candidate's lipid_name is molecular truth; masses are not summed before applying the reporting gate",
            "same_lipid_alternative": "reported non-true candidate with the same lipid_name; candidate/adduct ambiguity but not molecular false identity",
        },
        "synthetic_contract": {
            "K_true": 1,
            "residual": 0.0,
            "noise": 0.0,
            "spatial_template": "v50 foreground-masked real-B per-pixel channel-L2 norm normalized to foreground median 1",
            "target_median_foreground_B_l2": v50.TARGET_B_P50,
        },
        "training_contract": v50.training_contract(),
        "v50_reused_functions": [
            "asset_paths", "validate_assets", "load_library_and_metadata",
            "load_real_spatial_assets", "synthesize_case", "oracle_case",
            "train_fresh_case", "support_metrics", "atomic_write_json",
        ],
        "cone_cache": {
            "path": str(CONE_CACHE),
            "values_reused": int(len(cone)),
            "rho_zero_computed": False,
        },
        "asset_validation": validation,
    }


def metric_payload(reported: set, truth: set, gate: float) -> dict:
    tp = len(reported & truth)
    fp = len(reported - truth)
    fn = len(truth - reported)
    return {
        "gate": gate,
        "reported_support": sorted(reported),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": tp / (tp + fn) if tp + fn else 0.0,
    }


def identity_support(
    values: np.ndarray, true_index: int, metadata: dict, gate: float
) -> dict:
    names = np.asarray(metadata["lipid_name"], dtype=str)
    reported_candidates = set(np.flatnonzero(values > gate).astype(int).tolist())
    true_name = str(names[true_index])
    reported_names = {str(names[index]) for index in reported_candidates}
    same_lipid_alternatives = sorted(
        index for index in reported_candidates
        if index != true_index and str(names[index]) == true_name
    )
    return {
        "candidate_level": metric_payload(
            reported_candidates, {int(true_index)}, gate
        ),
        "lipid_name_level": metric_payload(reported_names, {true_name}, gate),
        "same_lipid_alternative_candidate_adduct_assignment_count": len(same_lipid_alternatives),
        "same_lipid_alternative_candidate_indices": same_lipid_alternatives,
    }


def oracle_report(case: dict, A_solver: np.ndarray, metadata: dict) -> dict:
    base = v50.oracle_case(case, A_solver, metadata)
    recovered = np.zeros(A_solver.shape[1], dtype=np.float64)
    for row in base["recovered_support_details"]:
        recovered[int(row["candidate_index"])] = float(row["abundance"])
    true_index = int(case["active_indices"][0])
    identity = identity_support(recovered, true_index, metadata, REPORT_GATE)
    molecular = identity["lipid_name_level"]
    clean = molecular["fp"] == 0 and molecular["fn"] == 0
    return {
        "case_name": case["case_name"],
        "status": "PASS" if clean else "ORACLE_IDENTITY_AMBIGUOUS",
        "true_candidate": candidate_record(true_index, metadata),
        "candidate_level_support": identity["candidate_level"],
        "lipid_name_level_support": molecular,
        "same_lipid_alternative_candidate_adduct_assignment_count": identity["same_lipid_alternative_candidate_adduct_assignment_count"],
        "same_lipid_alternative_candidate_indices": identity["same_lipid_alternative_candidate_indices"],
        "recovered_support_details": base["recovered_support_details"],
        "reconstruction_relative_residual": base["reconstruction_relative_residual"],
        "forward_consistency_max_abs": base["forward_consistency_max_abs"],
        "target_foreground_B_l2_p50": base["target_foreground_B_l2_p50"],
        "achieved_foreground_B_l2_p50": base["achieved_foreground_B_l2_p50"],
    }


def top_false_candidates(
    x_hat: np.ndarray, true_name: str, metadata: dict, limit: int = 10
) -> list[dict]:
    names = np.asarray(metadata["lipid_name"], dtype=str)
    false = np.flatnonzero(names != true_name)
    order = false[np.argsort(-x_hat[false], kind="stable")[:limit]]
    return [
        {
            **candidate_record(int(index), metadata),
            "X_hat_foreground_mean": float(x_hat[index]),
        }
        for index in order
    ]


def top_false_lipid_names(
    x_hat: np.ndarray, true_name: str, metadata: dict, limit: int = 10
) -> list[dict]:
    names = np.asarray(metadata["lipid_name"], dtype=str)
    rows = []
    for name in np.unique(names):
        if name == true_name:
            continue
        indices = np.flatnonzero(names == name)
        rows.append({
            "lipid_name": str(name),
            "summed_X_hat_foreground_mean": float(x_hat[indices].sum()),
            "candidate_indices": indices.astype(int).tolist(),
        })
    return sorted(
        rows,
        key=lambda row: (-row["summed_X_hat_foreground_mean"], row["lipid_name"]),
    )[:limit]


def contributor_indices(cached: list) -> list[int]:
    indices = []
    for item in cached:
        if isinstance(item, dict) and "candidate_index" in item:
            indices.append(int(item["candidate_index"]))
        elif isinstance(item, (int, np.integer)):
            indices.append(int(item))
    return sorted(set(indices))


def learned_report(
    frozen: dict, case: dict, learned: dict, metadata: dict, cached_contributors: list
) -> dict:
    mask = case["foreground_mask"]
    x_true = case["X_true"][:, mask].mean(axis=1).astype(np.float64)
    x_hat = learned["X_hat"][:, mask].mean(axis=1).astype(np.float64)
    true_index = int(frozen["true_index"])
    true_name = str(metadata["lipid_name"][true_index])
    names = np.asarray(metadata["lipid_name"], dtype=str)
    identity = identity_support(x_hat, true_index, metadata, REPORT_GATE)
    reported = x_hat > REPORT_GATE
    false_molecular = names != true_name
    total_mass = float(x_hat.sum())
    reported_mass = float(x_hat[reported].sum())
    rec_rel = float(
        np.linalg.norm(learned["B_hat"][:, mask] - case["B_sim"][:, mask])
        / max(np.linalg.norm(case["B_sim"][:, mask]), 1.0e-30)
    )
    true_relative_error = float(
        abs(x_hat[true_index] - x_true[true_index])
        / max(abs(x_true[true_index]), 1.0e-30)
    )
    report = {
        "status": "COMPLETE",
        "case_name": case["case_name"],
        "mechanism": frozen["mechanism"],
        "K_true": 1,
        "true_candidate_index": true_index,
        "true_lipid_name": true_name,
        "candidate_level_identity_at_1e-3": identity["candidate_level"],
        "lipid_name_level_identity_at_1e-3": identity["lipid_name_level"],
        "same_lipid_alternative_candidate_adduct_assignment_count": identity["same_lipid_alternative_candidate_adduct_assignment_count"],
        "same_lipid_alternative_candidate_indices": identity["same_lipid_alternative_candidate_indices"],
        "n_outputs_gt_1e-4": int(np.sum(x_hat > SENSITIVITY_GATE)),
        "n_outputs_gt_1e-3": int(np.sum(reported)),
        "all_output_false_molecular_identity_mass": float(x_hat[false_molecular].sum() / max(total_mass, 1.0e-30)),
        "reported_only_false_molecular_identity_mass": float(x_hat[reported & false_molecular].sum() / max(reported_mass, 1.0e-30)),
        "reconstruction_relative_residual": rec_rel,
        "true_abundance_relative_error": true_relative_error,
        "X_true_foreground_mean": float(x_true[true_index]),
        "X_hat_true": float(x_hat[true_index]),
        "top_10_false_molecular_candidates": top_false_candidates(x_hat, true_name, metadata),
        "top_10_false_molecular_lipid_names": top_false_lipid_names(x_hat, true_name, metadata),
        "forward_consistency_max_abs": case["forward_consistency_max_abs"],
        "target_foreground_B_l2_p50": case["target_foreground_B_l2_p50"],
        "achieved_foreground_B_l2_p50": case["achieved_foreground_B_l2_p50"],
        "final_training_epoch": learned["stopped_epoch"],
        "final_physical_loss": learned["final_physical_loss"],
        "final_raw_losses": dict(zip(v50.LOSS_NAMES, learned["final_raw_losses"])),
        "stop_reason": learned["stop_reason"],
        "wall_time_seconds": learned["wall_time_seconds"],
        "channel_weighting": learned["channel_weighting"],
        "checkpoint_loaded": False,
        "X_true_used_in_training": False,
        "A_solver_used_for_synthesis_oracle_training": True,
    }
    competitor = frozen["competitor_index"]
    if competitor is not None:
        competitor = int(competitor)
        report.update({
            "competitor_candidate_index": competitor,
            "competitor_lipid_name": str(metadata["lipid_name"][competitor]),
            "X_hat_competitor": float(x_hat[competitor]),
            "competitor_fraction_of_total_mass": float(x_hat[competitor] / max(total_mass, 1.0e-30)),
            "competitor_gt_1e-4": bool(x_hat[competitor] > SENSITIVITY_GATE),
            "competitor_gt_1e-3": bool(x_hat[competitor] > REPORT_GATE),
        })
    else:
        indices = contributor_indices(cached_contributors)
        report.update({
            "known_targeted_nnls_contributor_indices": indices,
            "summed_X_hat_mass_on_known_targeted_nnls_contributors": (
                float(x_hat[indices].sum()) if indices else None
            ),
            "targeted_nnls_contributor_status": (
                "RECOVERED_FROM_EXISTING_CACHE" if indices else "NOT_AVAILABLE_IN_EXISTING_CACHE"
            ),
        })
    return report


def prepare(args):
    paths = v50.asset_paths(args.asset_root.resolve())
    validation = v50.validate_assets(paths)
    A_raw, metadata = v50.load_library_and_metadata(paths)
    mz = np.asarray(np.load(paths["channel_axis"]), dtype=np.float64).reshape(-1)
    if mz.shape != (A_raw.shape[0],):
        raise RuntimeError(f"STAGE0_LOCK_MISMATCH: channel axis shape {mz.shape}")
    A_solver = normalize_like_get_A_matrix(A_raw)
    A_solver_repeat = normalize_like_get_A_matrix(A_raw)
    if not np.array_equal(A_solver, A_solver_repeat):
        raise RuntimeError("A_SOLVER_MISMATCH: normalization is not deterministic")
    cone, contributors = load_cone_cache(A_raw.shape[1])
    design = build_design(
        A_solver, mz, metadata, cone, contributors, validation
    )
    design["A_solver_contract"].update({
        "shape": list(A_solver.shape),
        "normalization_repeat_max_abs": float(np.max(np.abs(A_solver - A_solver_repeat))),
    })
    return paths, A_raw, A_solver, metadata, contributors, design


def frozen_case(case_name: str) -> dict:
    return next(row for row in FROZEN_CASES if row["case_name"] == case_name)


def synthesize(frozen: dict, A_solver: np.ndarray, B_real: np.ndarray, mask: np.ndarray) -> dict:
    return v50.synthesize_case(
        frozen["case_name"], [int(frozen["true_index"])], A_solver, B_real, mask
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--asset-root", type=Path, default=ROOT.parent / "decon-lipid"
    )
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DEFAULT)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--oracle-only", action="store_true")
    mode.add_argument("--case", choices=CASE_NAMES)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir = args.output_dir.resolve()
    paths, A_raw, A_solver, metadata, contributors, design = prepare(args)
    if args.dry_run:
        v50.print_summary({
            "status": "DRY_RUN_READY",
            "cases": design["cases"],
            "A_solver_contract": design["A_solver_contract"],
            "identity_definitions": design["identity_definitions"],
            "v50_reused_functions": design["v50_reused_functions"],
            "gpu_work_performed": False,
            "oracle_work_performed": False,
            "outputs_written": False,
        })
        return

    v50.atomic_write_json(args.output_dir / "design.json", design)
    B_real, mask = v50.load_real_spatial_assets(paths, A_solver)
    requested = FROZEN_CASES if args.oracle_only else (frozen_case(args.case),)
    cases = {}
    oracle_rows = []
    for frozen in requested:
        case = synthesize(frozen, A_solver, B_real, mask)
        cases[frozen["case_name"]] = case
        oracle_rows.append(oracle_report(case, A_solver, metadata))
    oracle_payload = {
        "status": (
            "PASS" if all(row["status"] == "PASS" for row in oracle_rows)
            else "ORACLE_IDENTITY_AMBIGUOUS"
        ),
        "cases": oracle_rows,
    }
    v50.atomic_write_json(args.output_dir / "oracle_report.json", oracle_payload)
    v50.print_summary(oracle_payload)
    if args.oracle_only:
        return

    oracle = oracle_rows[0]
    if oracle["status"] != "PASS":
        raise SystemExit("ORACLE_IDENTITY_AMBIGUOUS")
    frozen = requested[0]
    case = cases[frozen["case_name"]]
    case_output = args.output_dir / frozen["case_name"]
    learned = v50.train_fresh_case(
        case, A_raw, metadata, paths, case_output, args.device
    )
    report = learned_report(
        frozen, case, learned, metadata, contributors.get(frozen["true_index"], [])
    )
    v50.atomic_write_json(case_output / "report.json", report)
    v50.print_summary(report)


if __name__ == "__main__":
    main()
