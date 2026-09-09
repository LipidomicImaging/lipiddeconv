#!/usr/bin/env python3
"""Run the v54 matched CAL/HOLD complexity calibration experiment."""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
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
import run_v53_template_truth_pilot as v53
from src.rho_zero import identity_weights, rho_zero_from_weighted_case


VERSION = "v54_complexity_calibration"
OUTPUT_DEFAULT = ROOT / "results/v54_complexity_calibration"
EXPERIMENT_LOG = ROOT / "results/EXPERIMENT_LOG.md"
STAGE0_DIR = ROOT / "results/v53_stage0_empirical_domain"
V51_DIR = ROOT / "results/v51_competition_structure"
REPORT_GATE = 1.0e-3
SENSITIVITY_GATE = 1.0e-4
BASE_K = 76
HIGH_K = 129
WEAK_K = HIGH_K - BASE_K
TARGET_SIGNAL_P50 = 0.6036783456802368
POOL_SPLIT_SEED = 5400
MAPPING_SEEDS = {"R1": 5401, "R2": 5402}
MAX_EPOCH = 3000
CHECKPOINT_EPOCHS = (1000, 1500, 2000, 2500, 3000)
EXPECTED_A_SHAPE = (1084, 391)
CONTINUOUS_FEATURES = (
    "max_parent_cosine", "max_fragment_cosine", "max_full_cosine",
    "cone_isolation", "collective_gain",
)
BALANCE_FEATURES = (*CONTINUOUS_FEATURES, "d_single")
DATASETS = tuple(
    f"{split}_{replicate}_{condition}"
    for split in ("CAL", "HOLD")
    for replicate in ("R1", "R2")
    for condition in ("BASE76", "HIGH129")
)
CAL_DATASETS = tuple(name for name in DATASETS if name.startswith("CAL_"))
HOLD_DATASETS = tuple(name for name in DATASETS if name.startswith("HOLD_"))
LOG_BEGIN = "<!-- BEGIN v54_complexity_calibration -->"
LOG_END = "<!-- END v54_complexity_calibration -->"


def read_json(path: Path) -> dict:
    if not path.exists():
        raise RuntimeError(f"MISSING_DEPENDENCY: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise RuntimeError(f"MISSING_DEPENDENCY: {path}")
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def safe_ratio(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def numeric_summary(values: list[float] | np.ndarray) -> dict:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    quantiles = (0.01, 0.10, 0.20, 0.25, 0.50, 0.75, 0.80, 0.90, 0.95, 0.99)
    if values.size == 0:
        return {
            "count": 0, "min": None, **{f"q{int(q*100):02d}": None for q in quantiles},
            "max": None, "mean": None, "median": None, "std": None,
        }
    return {
        "count": int(values.size), "min": float(values.min()),
        **{f"q{int(q*100):02d}": float(np.quantile(values, q)) for q in quantiles},
        "max": float(values.max()), "mean": float(values.mean()),
        "median": float(np.median(values)),
        "std": float(values.std(ddof=0)),
    }


def rank_percentiles(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    result = np.empty(values.size, dtype=np.float64)
    start = 0
    while start < values.size:
        stop = start + 1
        while stop < values.size and values[order[stop]] == values[order[start]]:
            stop += 1
        result[order[start:stop]] = (0.5 * (start + stop - 1) + 1.0) / values.size
        start = stop
    return result


def required_results() -> dict[str, Path]:
    return {
        "stage0_design": STAGE0_DIR / "design.json",
        "stage0_domain_summary": STAGE0_DIR / "domain_summary.json",
        "v51_candidate_geometry": V51_DIR / "candidate_geometry.csv",
        "v51_summary": V51_DIR / "summary.json",
    }


def load_context(asset_root: Path) -> dict:
    lock = stage0.load_lock()
    paths = stage0.resolve_locked_paths(asset_root.resolve(), lock)
    missing_assets = stage0.missing_paths(paths)
    if missing_assets:
        return {"paths": paths, "missing_assets": missing_assets, "missing_results": []}
    validation, provenance = stage0.validate_shapes_and_hashes(paths, lock)
    A_raw, A_solver, X_real, metadata, mask = v53.load_locked_arrays(paths)
    means = X_real[:, mask].mean(axis=1, dtype=np.float64)
    base_templates = np.flatnonzero(means > REPORT_GATE).astype(int)
    high_templates = np.flatnonzero(means > SENSITIVITY_GATE).astype(int)
    if len(base_templates) != BASE_K or len(high_templates) != HIGH_K:
        raise RuntimeError(
            f"TEMPLATE_COUNT_MISMATCH: base={len(base_templates)}, high={len(high_templates)}"
        )
    names = np.asarray(metadata["lipid_name"], dtype=str)
    counts = Counter(names.tolist())
    singleton_indices = np.array(
        [index for index, name in enumerate(names) if counts[name] == 1], dtype=int
    )
    result_paths = required_results()
    missing_results = [str(path) for path in result_paths.values() if not path.exists()]
    context = {
        "paths": paths, "validation": validation, "provenance": provenance,
        "A_raw": A_raw, "A_solver": A_solver, "X_real": X_real,
        "metadata": metadata, "mask": mask, "foreground_means": means,
        "base_templates": base_templates, "high_templates": high_templates,
        "singleton_indices": singleton_indices, "missing_assets": [],
        "missing_results": missing_results,
    }
    if missing_results:
        return context
    stage_design, stage_domain = v53.validate_stage0_results(result_paths, validation, paths)
    v51_summary = read_json(result_paths["v51_summary"])
    if v51_summary.get("status") != "COMPLETE":
        raise RuntimeError("INVALID_V51_RESULT: status")
    v51_hashes = v51_summary.get("asset_validation", {}).get("validated_hashes", {})
    for name in ("A_library", "candidate_metadata", "channel_axis"):
        if v51_hashes.get(name) != validation["hashes_sha256"].get(name):
            raise RuntimeError(f"V51_RESULT_INPUT_MISMATCH: {name}")
    context.update({
        "stage0_design": stage_design, "stage0_domain": stage_domain,
        "v51_summary": v51_summary,
        "geometry": v53.load_geometry_rows(result_paths["v51_candidate_geometry"], metadata),
    })
    return context


def interference_features(context: dict) -> list[dict]:
    singleton = set(context["singleton_indices"].tolist())
    rows = []
    for source in context["geometry"]:
        if source["candidate_index"] not in singleton:
            continue
        full = float(np.clip(source["max_full_cosine"], -1.0, 1.0))
        d_single = math.sqrt(max(0.0, 1.0 - full * full))
        rows.append({
            "candidate_index": int(source["candidate_index"]),
            "candidate_id": source["candidate_id"],
            "lipid_name": source["lipid_name"],
            "lipid_class": source["lipid_class"],
            "max_parent_cosine": float(source["max_parent_cosine"]),
            "max_fragment_cosine": float(source["max_fragment_cosine"]),
            "max_full_cosine": full,
            "cone_isolation": float(source["cone_isolation"]),
            "d_single": d_single,
            "collective_gain": d_single - float(source["cone_isolation"]),
        })
    if len(rows) != len(singleton):
        raise RuntimeError("SINGLETON_GEOMETRY_MISMATCH")
    for feature in CONTINUOUS_FEATURES:
        ranks = rank_percentiles(np.array([row[feature] for row in rows]))
        for row, rank in zip(rows, ranks):
            row[f"rank_{feature}"] = float(rank)
    for row in rows:
        row["ambiguity_design_score"] = float(np.mean([
            row["rank_max_parent_cosine"], row["rank_max_fragment_cosine"],
            row["rank_max_full_cosine"], 1.0 - row["rank_cone_isolation"],
            row["rank_collective_gain"],
        ]))
    return sorted(rows, key=lambda row: row["candidate_index"])


def feature_distance(left: dict, right: dict) -> float:
    return float(np.linalg.norm([
        left[f"rank_{feature}"] - right[f"rank_{feature}"]
        for feature in CONTINUOUS_FEATURES
    ]))


def greedy_pairs(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    """Pair nearest rank-feature neighbors, within class before leftovers."""
    pairs: list[dict] = []
    leftovers: list[dict] = []

    def consume(group: list[dict], within_class: bool) -> list[dict]:
        remaining = sorted(group, key=lambda row: row["candidate_index"])
        while len(remaining) >= 2:
            best = min(
                (feature_distance(remaining[i], remaining[j]),
                 remaining[i]["candidate_index"], remaining[j]["candidate_index"], i, j)
                for i in range(len(remaining)) for j in range(i + 1, len(remaining))
            )
            distance, _, _, i, j = best
            left, right = remaining[i], remaining[j]
            pairs.append({
                "member_a_index": left["candidate_index"],
                "member_a_lipid_name": left["lipid_name"],
                "member_a_class": left["lipid_class"],
                "member_b_index": right["candidate_index"],
                "member_b_lipid_name": right["lipid_name"],
                "member_b_class": right["lipid_class"],
                "rank_feature_distance": distance,
                "matched_within_lipid_class": within_class,
                "pair_ambiguity_score": 0.5 * (
                    left["ambiguity_design_score"] + right["ambiguity_design_score"]
                ),
            })
            for index in sorted((i, j), reverse=True):
                remaining.pop(index)
        return remaining

    by_class: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_class[row["lipid_class"]].append(row)
    for lipid_class in sorted(by_class):
        leftovers.extend(consume(by_class[lipid_class], True))
    leftovers = consume(leftovers, False)
    pairs.sort(key=lambda row: (row["pair_ambiguity_score"], row["member_a_index"], row["member_b_index"]))
    for pair_id, row in enumerate(pairs):
        row["matched_pair_id"] = pair_id
    return pairs, leftovers


def spanning_positions(size: int, count: int) -> list[int]:
    if size < count:
        raise RuntimeError(f"INSUFFICIENT_MATCHED_PAIRS: {size} < {count}")
    if count == 1:
        return [size // 2]
    positions = [int(math.floor(k * (size - 1) / (count - 1))) for k in range(count)]
    if len(set(positions)) != count:
        raise RuntimeError("SPANNING_SELECTION_NOT_UNIQUE")
    return positions


def geometry_balance(pool_rows: list[dict], features_by_index: dict[int, dict]) -> dict:
    result = {}
    for split in ("CAL", "HOLD"):
        indices = [int(row[f"{split.lower()}_candidate_index"]) for row in pool_rows]
        result[split] = {
            feature: numeric_summary([features_by_index[index][feature] for index in indices])
            for feature in BALANCE_FEATURES
        }
        result[split]["lipid_class_counts"] = dict(sorted(Counter(
            features_by_index[index]["lipid_class"] for index in indices
        ).items()))
    return result


def build_design(context: dict) -> dict:
    features = interference_features(context)
    feature_by_index = {row["candidate_index"]: row for row in features}
    all_pairs, leftovers = greedy_pairs(features)
    selected_positions = spanning_positions(len(all_pairs), HIGH_K)
    selected_pairs = [dict(all_pairs[position]) for position in selected_positions]
    rng = np.random.default_rng(POOL_SPLIT_SEED)
    pool_rows = []
    for selected_order, pair in enumerate(selected_pairs):
        if int(rng.integers(0, 2)) == 0:
            cal_index, hold_index = pair["member_a_index"], pair["member_b_index"]
        else:
            cal_index, hold_index = pair["member_b_index"], pair["member_a_index"]
        pool_rows.append({
            "selected_pair_order": selected_order,
            "matched_pair_id": pair["matched_pair_id"],
            "pair_ambiguity_score": pair["pair_ambiguity_score"],
            "rank_feature_distance": pair["rank_feature_distance"],
            "matched_within_lipid_class": pair["matched_within_lipid_class"],
            "cal_candidate_index": cal_index,
            "cal_candidate_id": feature_by_index[cal_index]["candidate_id"],
            "cal_lipid_name": feature_by_index[cal_index]["lipid_name"],
            "cal_lipid_class": feature_by_index[cal_index]["lipid_class"],
            "hold_candidate_index": hold_index,
            "hold_candidate_id": feature_by_index[hold_index]["candidate_id"],
            "hold_lipid_name": feature_by_index[hold_index]["lipid_name"],
            "hold_lipid_class": feature_by_index[hold_index]["lipid_class"],
        })
    if set(row["cal_lipid_name"] for row in pool_rows) & set(row["hold_lipid_name"] for row in pool_rows):
        raise RuntimeError("CAL_HOLD_LIPID_NAME_OVERLAP")
    base_positions = spanning_positions(HIGH_K, BASE_K)
    base_pair_ids = {pool_rows[position]["matched_pair_id"] for position in base_positions}
    base_rows = [
        {"base_subset_order": order, **pool_rows[position]}
        for order, position in enumerate(base_positions)
    ]
    for pair in all_pairs:
        pair["selected_for_high129"] = pair["matched_pair_id"] in {
            row["matched_pair_id"] for row in pool_rows
        }
        pair["selected_for_base76"] = pair["matched_pair_id"] in base_pair_ids

    base_templates = context["base_templates"].tolist()
    weak_templates = sorted(set(context["high_templates"].tolist()) - set(base_templates))
    if len(weak_templates) != WEAK_K:
        raise RuntimeError("WEAK_TEMPLATE_COUNT_MISMATCH")
    pool_by_pair = {row["matched_pair_id"]: row for row in pool_rows}
    base_pairs = [row["matched_pair_id"] for row in base_rows]
    extra_pairs = [row["matched_pair_id"] for row in pool_rows if row["matched_pair_id"] not in base_pair_ids]
    mappings = []
    for replicate, seed in MAPPING_SEEDS.items():
        map_rng = np.random.default_rng(seed)
        shuffled_base = np.array(base_pairs, dtype=int)
        shuffled_extra = np.array(extra_pairs, dtype=int)
        map_rng.shuffle(shuffled_base)
        map_rng.shuffle(shuffled_extra)
        assignment = list(zip(base_templates, shuffled_base.tolist(), ["BASE"] * BASE_K))
        assignment += list(zip(weak_templates, shuffled_extra.tolist(), ["WEAK_EXTRA"] * WEAK_K))
        for split in ("CAL", "HOLD"):
            identity_key = f"{split.lower()}_candidate_index"
            name_key = f"{split.lower()}_lipid_name"
            id_key = f"{split.lower()}_candidate_id"
            for condition, limit in (("BASE76", BASE_K), ("HIGH129", HIGH_K)):
                dataset_id = f"{split}_{replicate}_{condition}"
                for slot, (template_index, pair_id, template_tier) in enumerate(assignment[:limit]):
                    identity = pool_by_pair[pair_id]
                    mappings.append({
                        "dataset_id": dataset_id, "split": split,
                        "replicate": replicate, "complexity_condition": condition,
                        "mapping_seed": seed, "mapping_slot": slot,
                        "template_tier": template_tier,
                        "template_original_candidate_index": template_index,
                        "template_foreground_mean": context["foreground_means"][template_index],
                        "matched_pair_id": pair_id,
                        "synthetic_truth_candidate_index": identity[identity_key],
                        "synthetic_truth_candidate_id": identity[id_key],
                        "synthetic_truth_lipid_name": identity[name_key],
                    })
    validate_mapping_nesting(mappings)
    template_rows = []
    for index in context["high_templates"]:
        template_rows.append({
            "template_original_candidate_index": int(index),
            "foreground_mean": float(context["foreground_means"][index]),
            "template_bank": "BASE" if index in set(context["base_templates"].tolist()) else "WEAK_EXTRA",
            "used_as_spatial_amplitude_only": True,
            "original_identity_discarded": True,
        })
    manifest = []
    for dataset_id in DATASETS:
        split, replicate, condition = dataset_id.split("_")
        manifest.append({
            "dataset_id": dataset_id, "split": split, "replicate": replicate,
            "complexity_condition": condition,
            "global_truth_K": BASE_K if condition == "BASE76" else HIGH_K,
            "mapping_seed": MAPPING_SEEDS[replicate],
            "clean_synthesis": True, "global_signal_target": TARGET_SIGNAL_P50,
            "dataset_directory": dataset_id,
        })
    balance = geometry_balance(pool_rows, feature_by_index)
    design = {
        "script_version": VERSION,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "status": "FROZEN_BEFORE_TRAINING",
        "input_hashes_sha256": context["validation"]["hashes_sha256"],
        "A_solver_contract": {
            "shape": list(context["A_solver"].shape),
            "definition": "float32 A_library column-L2 normalized with denominator norm+1e-8",
            "consistent_for_synthesis_oracle_training_rho_zero": True,
        },
        "singleton_identity_count": len(features),
        "same_lipid_name_partners_excluded": True,
        "full_391_candidate_library_is_interference_reference": True,
        "interference_features": list(BALANCE_FEATURES) + ["lipid_class"],
        "cone_isolation_rank_direction": "raw ascending percentile retained; lower raw cone_isolation means greater ambiguity",
        "matching": {
            "rank_features": list(CONTINUOUS_FEATURES),
            "algorithm": "deterministic greedy nearest-neighbor Euclidean matching within lipid_class, then across-class leftovers",
            "all_pair_count": len(all_pairs), "unpaired_leftover_count": len(leftovers),
            "selected_pair_count": HIGH_K,
            "selected_pair_rule": "129 evenly spaced positions after sorting all pairs by pair ambiguity score",
            "pool_split_seed": POOL_SPLIT_SEED,
            "CAL_HOLD_lipid_name_overlap": 0,
        },
        "base76_subset": {
            "count": BASE_K,
            "rule": "76 evenly spaced positions across the 129 selected pairs sorted by pair ambiguity score",
            "nested_in_high129": True,
        },
        "geometry_balance": balance,
        "template_banks": {
            "BASE76": "foreground_mean(real production X_hat_j) > 1e-3",
            "HIGH129": "foreground_mean(real production X_hat_j) > 1e-4; contains BASE76 plus 53 weak maps",
            "base_count": BASE_K, "high_count": HIGH_K, "weak_extra_count": WEAK_K,
            "individual_map_rescaling": False, "original_identities_discarded": True,
        },
        "mapping_seeds": MAPPING_SEEDS,
        "mapping_contract": "same pair permutation for CAL/HOLD; HIGH preserves every BASE assignment",
        "datasets": list(DATASETS),
        "synthesis": {
            "definition": "B_sim = A_solver @ (s * X_true_unscaled)",
            "clean": True, "empirical_residual": False, "Gaussian_noise": False,
            "target_foreground_median_signal_norm": TARGET_SIGNAL_P50,
            "one_global_scalar_per_dataset": True,
        },
        "training": {
            **v50.training_contract(), "epoch_ceiling": MAX_EPOCH,
            "checkpoint_loads": (
                "same-dataset v54 runtime resume checkpoints only; real-data and "
                "cross-dataset checkpoints forbidden"
            ),
            "scheduler_T_max": v50.CFG_DEFAULTS["n_epochs"],
            "runtime_checkpoint_epochs": list(CHECKPOINT_EPOCHS),
            "resumable_latest_model": True, "truth_dependent_stopping": False,
        },
        "rho_zero": {"implementation": "src/rho_zero.py", "weighting": "W=I"},
        "validity_status": "PENDING_REVIEW", "final_deployment_threshold_frozen": False,
    }
    return {
        "design": design, "features": features, "all_pairs": all_pairs,
        "pool_rows": pool_rows, "base_rows": base_rows,
        "template_rows": template_rows, "mappings": mappings, "manifest": manifest,
    }


def validate_mapping_nesting(mappings: list[dict]) -> None:
    by_dataset: dict[str, list[dict]] = defaultdict(list)
    for row in mappings:
        by_dataset[row["dataset_id"]].append(row)
    for split in ("CAL", "HOLD"):
        for replicate in ("R1", "R2"):
            base = by_dataset[f"{split}_{replicate}_BASE76"]
            high = by_dataset[f"{split}_{replicate}_HIGH129"]
            base_signature = [
                (row["template_original_candidate_index"], row["synthetic_truth_candidate_index"])
                for row in base
            ]
            high_signature = [
                (row["template_original_candidate_index"], row["synthetic_truth_candidate_index"])
                for row in high[:BASE_K]
            ]
            if base_signature != high_signature:
                raise RuntimeError(f"MAPPING_NESTING_FAILED: {split}_{replicate}")


def write_design(output: Path, bundle: dict) -> None:
    output.mkdir(parents=True, exist_ok=True)
    stage0.atomic_write_json(output / "design.json", bundle["design"])
    feature_fields = [
        "candidate_index", "candidate_id", "lipid_name", "lipid_class",
        *BALANCE_FEATURES, *[f"rank_{name}" for name in CONTINUOUS_FEATURES],
        "ambiguity_design_score",
    ]
    write_csv(output / "identity_interference_features.csv", bundle["features"], feature_fields)
    write_csv(output / "matched_identity_pairs.csv", bundle["all_pairs"], list(bundle["all_pairs"][0]))
    write_csv(output / "identity_pool_split.csv", bundle["pool_rows"], list(bundle["pool_rows"][0]))
    write_csv(output / "base76_pair_subset.csv", bundle["base_rows"], list(bundle["base_rows"][0]))
    write_csv(output / "template_bank.csv", bundle["template_rows"], list(bundle["template_rows"][0]))
    write_csv(output / "template_mappings.csv", bundle["mappings"], list(bundle["mappings"][0]))
    write_csv(output / "dataset_manifest.csv", bundle["manifest"], list(bundle["manifest"][0]))


def parse_scalar(value: str) -> Any:
    lowered = value.lower()
    if lowered in ("true", "false"):
        return lowered == "true"
    try:
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            return value


def dataset_mapping(output: Path, dataset_id: str) -> list[dict]:
    rows = [
        {key: parse_scalar(value) for key, value in row.items()}
        for row in read_csv(output / "template_mappings.csv")
        if row["dataset_id"] == dataset_id
    ]
    expected = BASE_K if dataset_id.endswith("BASE76") else HIGH_K
    if len(rows) != expected:
        raise RuntimeError(f"MAPPING_COUNT_MISMATCH: {dataset_id}={len(rows)}")
    rows.sort(key=lambda row: int(row["mapping_slot"]))
    return rows


def validate_frozen_design(output: Path, context: dict) -> dict:
    design = read_json(output / "design.json")
    if design.get("script_version") != VERSION:
        raise RuntimeError("FROZEN_DESIGN_VERSION_MISMATCH")
    if design.get("status") != "FROZEN_BEFORE_TRAINING":
        raise RuntimeError("FROZEN_DESIGN_STATUS_MISMATCH")
    if design.get("datasets") != list(DATASETS):
        raise RuntimeError("FROZEN_DESIGN_DATASET_MISMATCH")
    frozen_hashes = design.get("input_hashes_sha256", {})
    for name, digest in context["validation"]["hashes_sha256"].items():
        if frozen_hashes.get(name) != digest:
            raise RuntimeError(f"FROZEN_DESIGN_INPUT_MISMATCH: {name}")
    return design


def construct_dataset(context: dict, mapping: list[dict], dataset_id: str) -> dict:
    X_true = np.zeros((EXPECTED_A_SHAPE[1], *context["mask"].shape), dtype=np.float32)
    for row in mapping:
        X_true[int(row["synthetic_truth_candidate_index"])] = context["X_real"][
            int(row["template_original_candidate_index"])
        ]
    B_unscaled = np.einsum("mc,cyx->myx", context["A_solver"], X_true, optimize=True)
    unscaled_p50 = float(np.median(np.linalg.norm(B_unscaled[:, context["mask"]], axis=0)))
    if not np.isfinite(unscaled_p50) or unscaled_p50 <= 0:
        raise RuntimeError("INVALID_UNSCALED_SIGNAL")
    scale = TARGET_SIGNAL_P50 / unscaled_p50
    X_true *= np.float32(scale)
    B_sim = np.einsum("mc,cyx->myx", context["A_solver"], X_true, optimize=True)
    B_check = (context["A_solver"] @ X_true.reshape(EXPECTED_A_SHAPE[1], -1)).reshape(B_sim.shape)
    forward_error = float(np.max(np.abs(B_sim - B_check)))
    if not np.allclose(B_sim, B_check, rtol=2e-6, atol=1e-7):
        raise RuntimeError("FORWARD_CONSISTENCY_FAILED")
    achieved = float(np.median(np.linalg.norm(B_sim[:, context["mask"]], axis=0)))
    true_indices = [int(row["synthetic_truth_candidate_index"]) for row in mapping]
    foreground_truth_means = X_true[:, context["mask"]].mean(axis=1, dtype=np.float64)
    reportable = [index for index in true_indices if foreground_truth_means[index] > REPORT_GATE]
    K_pixel = np.sum(X_true[true_indices][:, context["mask"]] > REPORT_GATE, axis=0)
    return {
        "case_name": dataset_id, "active_indices": true_indices,
        "X_true": X_true, "B_sim": B_sim.astype(np.float32, copy=False),
        "foreground_mask": context["mask"], "global_scale": float(scale),
        "target_foreground_B_l2_p50": TARGET_SIGNAL_P50,
        "achieved_foreground_B_l2_p50": achieved,
        "forward_consistency_max_abs": forward_error,
        "global_truth_identity_count": len(true_indices),
        "reportable_truth_indices": reportable,
        "true_per_pixel_K_foreground": numeric_summary(K_pixel),
    }


def checkpoint_fields() -> list[str]:
    return [
        "epoch", "reconstruction_relative_residual", "foreground_mean_X",
        "n_outputs_gt_1e-4", "n_outputs_gt_1e-3", "support_gt_1e-3",
        "relative_X_change_from_previous_checkpoint",
    ]


def train_resumable(
    case: dict, context: dict, output: Path, device_name: str = "cuda:0"
) -> dict:
    """Reuse the production model/loss contract with a v54 capped driver."""
    import torch
    import torch.nn.functional as F
    from config_758 import Cfg
    from lipid_ista import LipidENNet
    from run_758_ista import build_channel_weights
    from utils import set_seed

    if device_name.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA_REQUIRED_FOR_TRAINING")
    device = torch.device(device_name)
    set_seed(Cfg.seed)
    output.mkdir(parents=True, exist_ok=True)
    A_init = torch.as_tensor(context["A_raw"], dtype=torch.float32, device=device)
    A_init = A_init / (torch.linalg.vector_norm(A_init, dim=0, keepdim=True) + 1e-8)
    B = torch.as_tensor(case["B_sim"][None], dtype=torch.float32, device=device)
    net = LipidENNet(A_init=A_init, K=Cfg.K_layers,
                     clamp_min=Cfg.calib_clamp_min, clamp_max=Cfg.calib_clamp_max).to(device)
    AutomaticWeightedLoss = v50.make_automatic_weighted_loss(torch)
    loss_balancer = AutomaticWeightedLoss(4).to(device)
    namespace = v50.production_weight_namespace(A_init, context["metadata"], torch, device)
    fragment_weights, weight_report = build_channel_weights(
        namespace, A_init, v50.runtime_cfg(context["paths"], Cfg)
    )
    optimizer = torch.optim.AdamW([
        {"params": net.parameters(), "lr": Cfg.lr_net},
        {"params": loss_balancer.parameters(), "lr": 1e-3},
    ], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Cfg.n_epochs, eta_min=1e-6
    )
    history = {
        "epoch": [], "train_total": [], "eval_total": [],
        "eval_physical_loss": [], "eval_raw_losses": [],
        "relative_eval_loss_change": [], "relative_abundance_change": [],
        "stable_checks": [], "automatic_loss_weights": [],
    }
    diagnostics: list[dict] = []
    previous_eval_loss = None
    previous_eval_x = None
    previous_checkpoint_x = None
    stable_checks = 0
    start_epoch = 1
    terminal_resume_epoch = None
    terminal_resume_reason = None
    latest = output / "latest_model.pth"
    resumed = False
    if latest.exists():
        try:
            saved = torch.load(latest, map_location=device, weights_only=False)
        except TypeError:
            saved = torch.load(latest, map_location=device)
        if saved.get("dataset_id") != case["case_name"]:
            raise RuntimeError("CHECKPOINT_DATASET_MISMATCH")
        net.load_state_dict(saved["net"])
        loss_balancer.load_state_dict(saved["loss_balancer"])
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        if saved.get("torch_rng_state") is not None:
            torch.set_rng_state(saved["torch_rng_state"].cpu())
        if torch.cuda.is_available() and saved.get("cuda_rng_states") is not None:
            torch.cuda.set_rng_state_all(saved["cuda_rng_states"])
        history = saved["history"]
        diagnostics = saved["diagnostics"]
        previous_eval_loss = saved["previous_eval_loss"]
        previous_eval_x = saved["previous_eval_x"].to(device) if saved["previous_eval_x"] is not None else None
        previous_checkpoint_x = saved["previous_checkpoint_x"]
        stable_checks = int(saved["stable_checks"])
        start_epoch = int(saved["epoch"]) + 1
        terminal_resume_reason = saved.get("terminal_stop_reason")
        if terminal_resume_reason is not None:
            terminal_resume_epoch = int(saved["epoch"])
            start_epoch = MAX_EPOCH + 1
        resumed = True

    def save_checkpoint(epoch: int, named: bool = False) -> None:
        payload = {
            "dataset_id": case["case_name"], "epoch": epoch,
            "net": net.state_dict(), "loss_balancer": loss_balancer.state_dict(),
            "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
            "history": history, "diagnostics": diagnostics,
            "previous_eval_loss": previous_eval_loss,
            "previous_eval_x": previous_eval_x.detach().cpu() if previous_eval_x is not None else None,
            "previous_checkpoint_x": previous_checkpoint_x,
            "stable_checks": stable_checks,
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_states": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "terminal_stop_reason": (
                "converged" if stable_checks >= Cfg.early_stop_patience
                else ("max_epochs" if epoch >= MAX_EPOCH else None)
            ),
            "X_true_used_in_training_or_stopping": False,
        }
        torch.save(payload, latest)
        if named:
            torch.save(payload, output / f"checkpoint_epoch_{epoch}.pth")

    stopped_epoch = terminal_resume_epoch if terminal_resume_epoch is not None else start_epoch - 1
    stop_reason = terminal_resume_reason or "max_epochs"
    started = time.time()
    for epoch in range(start_epoch, MAX_EPOCH + 1):
        net.train()
        net.calibrator.W.requires_grad = epoch >= Cfg.warmup_epochs
        optimizer.zero_grad(set_to_none=True)
        _, X_list, A_curr, _ = net(B)
        raw_losses, _ = v50.compute_advanced_losses(
            torch, F, X_list, B, A_curr, A_init, fragment_weights
        )
        total_loss, _ = loss_balancer(raw_losses)
        if not torch.isfinite(total_loss):
            stop_reason, stopped_epoch = "nonfinite_loss", epoch
            break
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), Cfg.grad_clip_norm)
        optimizer.step()
        scheduler.step()
        stopped_epoch = epoch
        if epoch % Cfg.early_stop_check_freq == 0 or epoch == 1:
            net.eval()
            with torch.no_grad():
                X_eval, X_layers, A_eval, _ = net(B)
                eval_raw, B_hat = v50.compute_advanced_losses(
                    torch, F, X_layers, B, A_eval, A_init, fragment_weights
                )
                eval_total_tensor, _ = loss_balancer(eval_raw)
                physical = float(eval_raw.sum().item())
                if previous_eval_loss is None:
                    loss_change = x_change = float("inf")
                else:
                    loss_change = abs(physical - previous_eval_loss) / max(abs(previous_eval_loss), 1e-12)
                    x_change = float(torch.linalg.vector_norm(X_eval - previous_eval_x) /
                                     torch.clamp(torch.linalg.vector_norm(previous_eval_x), min=1e-12))
                if (epoch >= Cfg.early_stop_min_epochs
                        and loss_change <= Cfg.early_stop_loss_rtol
                        and x_change <= Cfg.early_stop_x_rtol):
                    stable_checks += 1
                else:
                    stable_checks = 0
                history["epoch"].append(epoch)
                history["train_total"].append(float(total_loss.item()))
                history["eval_total"].append(float(eval_total_tensor.item()))
                history["eval_physical_loss"].append(physical)
                history["eval_raw_losses"].append([float(value) for value in eval_raw.detach().cpu()])
                history["relative_eval_loss_change"].append(v50.json_number(loss_change))
                history["relative_abundance_change"].append(v50.json_number(x_change))
                history["stable_checks"].append(stable_checks)
                weights = 0.5 * torch.exp(-torch.clamp(loss_balancer.params, -9.0, 10.0))
                history["automatic_loss_weights"].append([float(value) for value in weights.detach().cpu()])
                previous_eval_loss = physical
                previous_eval_x = X_eval.detach().clone()
                if epoch in CHECKPOINT_EPOCHS:
                    x_mean = X_eval[0, :, torch.as_tensor(case["foreground_mask"], device=device)].mean(dim=1).detach().cpu().numpy()
                    checkpoint_change = None if previous_checkpoint_x is None else float(
                        np.linalg.norm(x_mean - previous_checkpoint_x) /
                        max(float(np.linalg.norm(previous_checkpoint_x)), 1e-30)
                    )
                    support = np.flatnonzero(x_mean > REPORT_GATE).astype(int).tolist()
                    diagnostics.append({
                        "epoch": epoch,
                        "reconstruction_relative_residual": float(
                            torch.linalg.vector_norm(B_hat[0, :, torch.as_tensor(case["foreground_mask"], device=device)] -
                                                     B[0, :, torch.as_tensor(case["foreground_mask"], device=device)]) /
                            torch.clamp(torch.linalg.vector_norm(B[0, :, torch.as_tensor(case["foreground_mask"], device=device)]), min=1e-30)
                        ),
                        "foreground_mean_X": json.dumps([float(value) for value in x_mean]),
                        "n_outputs_gt_1e-4": int(np.sum(x_mean > SENSITIVITY_GATE)),
                        "n_outputs_gt_1e-3": len(support),
                        "support_gt_1e-3": json.dumps(support),
                        "relative_X_change_from_previous_checkpoint": checkpoint_change,
                    })
                    previous_checkpoint_x = x_mean.copy()
            stage0.atomic_write_json(output / "training_history.json", history)
            write_csv(output / "checkpoint_diagnostics.csv", diagnostics, checkpoint_fields())
            save_checkpoint(epoch, named=epoch in CHECKPOINT_EPOCHS)
            if stable_checks >= Cfg.early_stop_patience:
                stop_reason = "converged"
                break
    if stopped_epoch == MAX_EPOCH:
        stop_reason = "max_epochs"
    net.eval()
    with torch.no_grad():
        X_hat, X_layers, A_final, _ = net(B)
        final_raw, B_hat = v50.compute_advanced_losses(
            torch, F, X_layers, B, A_final, A_init, fragment_weights
        )
    return {
        "X_hat": X_hat[0].detach().cpu().numpy().astype(np.float32),
        "B_hat": B_hat[0].detach().cpu().numpy().astype(np.float32),
        "stopped_epoch": stopped_epoch, "stop_reason": stop_reason,
        "wall_time_seconds_this_invocation": time.time() - started,
        "final_physical_loss": float(final_raw.sum().item()),
        "final_raw_losses": [float(value) for value in final_raw.detach().cpu()],
        "channel_weighting": weight_report, "resumed_from_runtime_checkpoint": resumed,
        "scheduler_T_max": Cfg.n_epochs,
    }


_RHO_A: np.ndarray | None = None
_RHO_B: np.ndarray | None = None
_RHO_W: np.ndarray | None = None


def init_rho_worker(A: np.ndarray, b: np.ndarray, weights: np.ndarray) -> None:
    global _RHO_A, _RHO_B, _RHO_W
    _RHO_A, _RHO_B, _RHO_W = A, b, weights


def rho_worker(index: int) -> tuple[int, float]:
    if _RHO_A is None or _RHO_B is None or _RHO_W is None:
        raise RuntimeError("RHO_WORKER_NOT_INITIALIZED")
    result = rho_zero_from_weighted_case(_RHO_A, _RHO_B, index, weights=_RHO_W)
    return index, float(result["rho_zero"])


def rho_values(A: np.ndarray, b: np.ndarray, indices: list[int], workers: int) -> dict[int, float]:
    weights = identity_weights(A)
    if workers <= 1:
        return {
            index: float(rho_zero_from_weighted_case(A, b, index, weights=weights)["rho_zero"])
            for index in indices
        }
    with ProcessPoolExecutor(
        max_workers=workers, initializer=init_rho_worker,
        initargs=(A.astype(np.float64), b.astype(np.float64), weights),
    ) as executor:
        return dict(executor.map(rho_worker, indices))


def learned_records(
    dataset_id: str, case: dict, learned: dict, context: dict, workers: int
) -> tuple[list[dict], dict]:
    mask = case["foreground_mask"]
    xhat = learned["X_hat"][:, mask].mean(axis=1, dtype=np.float64)
    reported = np.flatnonzero(xhat > REPORT_GATE).astype(int).tolist()
    metadata = context["metadata"]
    truth_indices = set(case["active_indices"])
    truth_names = {str(metadata["lipid_name"][index]) for index in truth_indices}
    reportable_names = {str(metadata["lipid_name"][index]) for index in case["reportable_truth_indices"]}
    b = case["B_sim"][:, mask].mean(axis=1, dtype=np.float64)
    rho = rho_values(context["A_solver"], b, reported, workers)
    split, replicate, condition = dataset_id.split("_")
    records = []
    for index in reported:
        name = str(metadata["lipid_name"][index])
        candidate_truth = index in truth_indices
        molecular_truth = name in truth_names
        records.append({
            "dataset_id": dataset_id, "split": split, "replicate": replicate,
            "complexity_condition": condition,
            "global_truth_K": case["global_truth_identity_count"],
            "candidate_index": index, "candidate_id": str(metadata["candidate_id"][index]),
            "lipid_name": name, "X_hat": float(xhat[index]),
            "candidate_truth": candidate_truth, "molecular_truth": molecular_truth,
            "same_lipid_alternative_candidate": molecular_truth and not candidate_truth,
            "rho_zero": rho[index],
        })
    reported_names = {row["lipid_name"] for row in records}
    candidate_tp = len(set(reported) & truth_indices)
    candidate_fp = len(set(reported) - truth_indices)
    molecular_tp = len(reported_names & truth_names)
    molecular_fp = len(reported_names - truth_names)
    result = {
        "reported_candidate_count_gt_1e-3": len(records),
        "learned_reported_molecular_identity_count": len(reported_names),
        "candidate_level": {
            "TP": candidate_tp, "FP": candidate_fp, "FN": len(truth_indices) - candidate_tp,
            "precision": safe_ratio(candidate_tp, len(records)),
            "FDR": safe_ratio(candidate_fp, len(records)),
        },
        "molecular_level": {
            "TP": molecular_tp, "FP": molecular_fp, "FN": len(truth_names) - molecular_tp,
            "precision": safe_ratio(molecular_tp, molecular_tp + molecular_fp),
            "FDR": safe_ratio(molecular_fp, molecular_tp + molecular_fp),
            "all_truth_recall": safe_ratio(molecular_tp, len(truth_names)),
            "reportable_truth_recall": safe_ratio(len(reported_names & reportable_names), len(reportable_names)),
        },
        "n_outputs_gt_1e-4": int(np.sum(xhat > SENSITIVITY_GATE)),
        "n_outputs_gt_1e-3": len(records),
        "reconstruction_relative_residual": float(
            np.linalg.norm(learned["B_hat"][:, mask] - case["B_sim"][:, mask]) /
            max(np.linalg.norm(case["B_sim"][:, mask]), 1e-30)
        ),
        "truth_lipid_names": sorted(truth_names),
        "reportable_truth_lipid_names": sorted(reportable_names),
    }
    return records, result


def run_dataset(args: argparse.Namespace, context: dict, dataset_id: str, oracle_only: bool = False) -> dict:
    mapping = dataset_mapping(args.output_dir, dataset_id)
    case = construct_dataset(context, mapping, dataset_id)
    oracle = v53.run_oracle(case, context["A_solver"], context["metadata"])
    dataset_dir = args.output_dir / dataset_id
    dataset_dir.mkdir(parents=True, exist_ok=True)
    base_report = {
        "dataset_id": dataset_id, "split": dataset_id.split("_")[0],
        "replicate": dataset_id.split("_")[1],
        "complexity_condition": dataset_id.split("_")[2],
        "global_truth_identity_count": case["global_truth_identity_count"],
        "true_per_pixel_K_foreground": case["true_per_pixel_K_foreground"],
        "learned_reported_count_definition": "foreground_mean(X_hat_j) > 1e-3",
        "global_signal_scalar": case["global_scale"],
        "target_foreground_median_signal_norm": TARGET_SIGNAL_P50,
        "achieved_foreground_median_signal_norm": case["achieved_foreground_B_l2_p50"],
        "forward_consistency_max_abs": case["forward_consistency_max_abs"],
        "oracle": oracle, "identity_pool_or_mapping_redesigned_from_oracle": False,
    }
    if oracle["status"] != "PASS":
        report = {**base_report, "status": "ORACLE_MOLECULAR_AMBIGUITY",
                  "learned_training_performed": False}
        stage0.atomic_write_json(dataset_dir / "report.json", report)
        return report
    if oracle_only:
        report = {**base_report, "status": "ORACLE_ONLY_PASS", "learned_training_performed": False}
        stage0.atomic_write_json(dataset_dir / "report.json", report)
        return report
    learned = train_resumable(case, context, dataset_dir)
    if learned["stop_reason"] == "nonfinite_loss":
        report = {
            **base_report, "status": "TRAINING_FAILED_NONFINITE_LOSS",
            "learned_training_performed": True,
            "training": {"final_epoch": learned["stopped_epoch"],
                         "stop_reason": learned["stop_reason"]},
        }
        stage0.atomic_write_json(dataset_dir / "report.json", report)
        return report
    records, raw_result = learned_records(dataset_id, case, learned, context, args.rho_workers)
    report = {
        **base_report, "status": "COMPLETE", "learned_training_performed": True,
        "learned_raw_identity_performance": raw_result,
        "training": {
            "final_epoch": learned["stopped_epoch"], "stop_reason": learned["stop_reason"],
            "wall_time_seconds_this_invocation": learned["wall_time_seconds_this_invocation"],
            "hard_epoch_cap": MAX_EPOCH, "checkpoint_epochs": list(CHECKPOINT_EPOCHS),
            "scheduler_T_max": learned["scheduler_T_max"],
            "resumed_from_runtime_checkpoint": learned["resumed_from_runtime_checkpoint"],
            "real_data_checkpoint_loaded": False,
            "X_true_used_in_training_or_stopping": False,
            "final_physical_loss": learned["final_physical_loss"],
            "final_raw_losses": learned["final_raw_losses"],
            "channel_weighting": learned["channel_weighting"],
        },
        "rho_zero_weighting": "W=I",
    }
    fields = [
        "dataset_id", "split", "replicate", "complexity_condition", "global_truth_K",
        "candidate_index", "candidate_id", "lipid_name", "X_hat", "candidate_truth",
        "molecular_truth", "same_lipid_alternative_candidate", "rho_zero",
    ]
    write_csv(dataset_dir / "reported_identity_records.csv", records, fields)
    stage0.atomic_write_json(dataset_dir / "report.json", report)
    return report


def molecular_units(records: list[dict], reports: dict[str, dict]) -> list[dict]:
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in records:
        grouped[(str(row["dataset_id"]), str(row["lipid_name"]))].append(row)
    units = []
    for (dataset_id, name), rows in grouped.items():
        reportable = set(reports[dataset_id]["learned_raw_identity_performance"]["reportable_truth_lipid_names"])
        units.append({
            "dataset_id": dataset_id, "complexity_condition": rows[0]["complexity_condition"],
            "lipid_name": name, "molecular_truth": any(bool(row["molecular_truth"]) for row in rows),
            "reportable_truth": name in reportable,
            "rho_zero": max(float(row["rho_zero"]) for row in rows),
            "X_hat": sum(float(row["X_hat"]) for row in rows),
        })
    return units


def metric_curve(units: list[dict], reports: dict[str, dict], score: str, scope: str) -> list[dict]:
    scoped_units = [row for row in units if scope == "POOLED" or row["complexity_condition"] == scope]
    dataset_ids = sorted(
        name for name, report in reports.items()
        if scope == "POOLED" or report["complexity_condition"] == scope
    )
    all_truth_denominator = sum(reports[name]["global_truth_identity_count"] for name in dataset_ids)
    reportable_denominator = sum(len(reports[name]["learned_raw_identity_performance"]["reportable_truth_lipid_names"])
                                 for name in dataset_ids)
    thresholds = sorted({float(row[score]) for row in scoped_units}, reverse=True)
    curve = []
    for threshold in thresholds:
        retained = [row for row in scoped_units if float(row[score]) >= threshold]
        tp = sum(bool(row["molecular_truth"]) for row in retained)
        fp = len(retained) - tp
        reportable_tp = sum(bool(row["molecular_truth"]) and bool(row["reportable_truth"]) for row in retained)
        curve.append({
            "scope": scope, "score": score, "threshold": threshold,
            "n_retained": len(retained), "TP": tp, "FP": fp,
            "precision": safe_ratio(tp, tp + fp), "FDR": safe_ratio(fp, tp + fp),
            "all_truth_recall": safe_ratio(tp, all_truth_denominator),
            "reportable_truth_recall": safe_ratio(reportable_tp, reportable_denominator),
            "coverage": safe_ratio(len(retained), len(scoped_units)),
        })
    return curve


def calibration_threshold(curve: list[dict], target_fdr: float) -> dict | None:
    eligible = [row for row in curve if row["FDR"] <= target_fdr and row["n_retained"] > 0]
    if not eligible:
        return None
    chosen = max(eligible, key=lambda row: (row["n_retained"], -row["threshold"]))
    return {"target_FDR": target_fdr, "threshold": chosen["threshold"],
            "calibration_operating_point": chosen}


def apply_threshold(units: list[dict], reports: dict[str, dict], threshold: float, score: str, scope: str) -> dict:
    scoped = [row for row in units if scope == "POOLED" or row["complexity_condition"] == scope]
    retained = [row for row in scoped if float(row[score]) >= threshold]
    dataset_ids = sorted(
        name for name, report in reports.items()
        if scope == "POOLED" or report["complexity_condition"] == scope
    )
    all_den = sum(reports[name]["global_truth_identity_count"] for name in dataset_ids)
    reportable_den = sum(len(reports[name]["learned_raw_identity_performance"]["reportable_truth_lipid_names"])
                         for name in dataset_ids)
    tp = sum(bool(row["molecular_truth"]) for row in retained)
    fp = len(retained) - tp
    reportable_tp = sum(bool(row["molecular_truth"]) and bool(row["reportable_truth"]) for row in retained)
    return {
        "scope": scope, "threshold": threshold, "n_retained": len(retained),
        "TP": tp, "FP": fp, "precision": safe_ratio(tp, tp + fp),
        "FDR": safe_ratio(fp, tp + fp), "all_truth_recall": safe_ratio(tp, all_den),
        "reportable_truth_recall": safe_ratio(reportable_tp, reportable_den),
        "coverage": safe_ratio(len(retained), len(scoped)),
    }


def load_completed(output: Path, dataset_names: tuple[str, ...]) -> tuple[list[dict], dict[str, dict]]:
    records, reports = [], {}
    for dataset_id in dataset_names:
        report = read_json(output / dataset_id / "report.json")
        if report.get("status") != "COMPLETE":
            raise RuntimeError(f"DATASET_NOT_COMPLETE: {dataset_id}")
        reports[dataset_id] = report
        records.extend({key: parse_scalar(value) for key, value in row.items()}
                       for row in read_csv(output / dataset_id / "reported_identity_records.csv"))
    return records, reports


def build_summary(report: dict) -> str:
    rho = report["rho_zero_calibration"]
    held = report["rho_heldout_validation"]
    abundance = report["abundance_only_baseline"]
    return "\n".join([
        "# v54 Complexity Calibration", "", "## Purpose", "",
        "Calibrate extremely high-confidence molecular identity selection on CAL identities and test frozen thresholds on disjoint HOLD identities under two complexity conditions.", "",
        "## Complexity definition", "",
        "Global truth K is exactly 76 (BASE76) or 129 (HIGH129). True per-pixel K counts scaled truth maps above 1e-3 at each foreground pixel. Learned reported count independently counts foreground-mean X_hat above 1e-3.", "",
        "## Empirical template construction", "",
        "BASE76 uses intact production X_hat maps with foreground mean >1e-3. HIGH129 contains those same 76 maps plus 53 maps above 1e-4 and at most 1e-3. Original production identities were discarded; one dataset-wide scalar targets foreground median signal norm 0.6036783456802368.", "",
        "## Interference-aware identity selection", "",
        f"CAL/HOLD use 129 disjoint singleton-name matched pairs. Geometry balance: `{json.dumps(report['geometry_balance'], ensure_ascii=False)}`.", "",
        "## Mapping replicates", "",
        "R1 (seed 5401) and R2 (seed 5402) are different frozen map-to-identity permutations. CAL and HOLD share the pair permutation structure, and HIGH preserves all BASE assignments.", "",
        "## Oracle results", "", f"`{json.dumps(report['oracle_results'], ensure_ascii=False)}`", "",
        "## Learned raw molecular identity performance", "", f"`{json.dumps(report['learned_raw_results'], ensure_ascii=False)}`", "",
        "## rho_zero calibration", "", f"Calibration candidate thresholds only: `{json.dumps(rho, ensure_ascii=False)}`", "",
        "## Held-out validation", "", f"Frozen-threshold HOLD results: `{json.dumps(held, ensure_ascii=False)}`", "",
        "## BASE76 vs HIGH129", "",
        report["base76_vs_high129_interpretation"], "",
        "## Abundance-only baseline", "", f"`{json.dumps(abundance, ensure_ascii=False)}`", "",
        "## Limitations", "",
        "- Clean synthesis only.", "- Empirical templates originate from production X_hat.",
        "- No empirical residual is included yet.", "- Only two mapping replicates are used.",
        "- Training has a 3000 epoch hard cap.", "- True biological K remains unknown.", "",
        "## Validity status", "", "PENDING_REVIEW", "", "## Decision", "",
        "PENDING_RESULT_REVIEW", "", "Final deployment threshold frozen: NO", "",
    ])


def update_log(report: dict, design: dict) -> None:
    lines = [
        LOG_BEGIN, "## v54 Complexity Calibration", "",
        f"- Version: {VERSION}", f"- Date: {date.today().isoformat()}",
        "- Purpose: CAL/HOLD calibration of high-confidence rho_zero identity selection across global truth K 76 and 129.",
        f"- Input hashes: {json.dumps(design['input_hashes_sha256'], ensure_ascii=False)}",
        "- Global truth K conditions: BASE76=76; HIGH129=129; nested within split/replicate.",
        f"- True per-pixel K distributions: {json.dumps(report['true_per_pixel_K_distributions'], ensure_ascii=False)}",
        "- CAL/HOLD identity split: 129 disjoint singleton lipid_name matched pairs; seed 5400.",
        f"- Interference geometry balance: {json.dumps(design['geometry_balance'], ensure_ascii=False)}",
        "- Template mapping seeds: R1=5401; R2=5402.",
        f"- Global scaling factors: {json.dumps(report['global_scaling_factors'], ensure_ascii=False)}",
        f"- Oracle results: {json.dumps(report['oracle_results'], ensure_ascii=False)}",
        f"- Learned raw results: {json.dumps(report['learned_raw_results'], ensure_ascii=False)}",
        f"- rho calibration candidate thresholds: {json.dumps(report['rho_zero_calibration'], ensure_ascii=False)}",
        f"- Held-out validation: {json.dumps(report['rho_heldout_validation'], ensure_ascii=False)}",
        f"- Abundance-only baseline: {json.dumps(report['abundance_only_baseline'], ensure_ascii=False)}",
        f"- Training epochs/stop reasons: {json.dumps(report['training_epochs_and_stop_reasons'], ensure_ascii=False)}",
        "- Limitations: clean synthesis; production-X_hat templates; no empirical residual; two mapping replicates; 3000 epoch cap; biological K unknown.",
        "- Validity status: PENDING_REVIEW", "- Decision: PENDING_RESULT_REVIEW",
        "- Final threshold frozen: NO", "- Git commit: PENDING_USER_COMMIT", "", LOG_END,
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
    cal_records, cal_reports = load_completed(output, CAL_DATASETS)
    hold_records, hold_reports = load_completed(output, HOLD_DATASETS)
    all_reports = {**cal_reports, **hold_reports}
    cal_units = molecular_units(cal_records, cal_reports)
    hold_units = molecular_units(hold_records, hold_reports)
    rho_curve = sum((metric_curve(cal_units, cal_reports, "rho_zero", scope)
                     for scope in ("POOLED", "BASE76", "HIGH129")), [])
    abundance_curve = sum((metric_curve(cal_units, cal_reports, "X_hat", scope)
                           for scope in ("POOLED", "BASE76", "HIGH129")), [])
    pooled_rho = [row for row in rho_curve if row["scope"] == "POOLED"]
    pooled_abundance = [row for row in abundance_curve if row["scope"] == "POOLED"]
    rho_thresholds = {
        "tau_cal_FDR5": calibration_threshold(pooled_rho, 0.05),
        "tau_cal_FDR1": calibration_threshold(pooled_rho, 0.01),
    }
    abundance_thresholds = {
        "tau_cal_FDR5": calibration_threshold(pooled_abundance, 0.05),
        "tau_cal_FDR1": calibration_threshold(pooled_abundance, 0.01),
    }

    def validation(thresholds: dict, score: str) -> dict:
        result = {}
        for name, threshold in thresholds.items():
            if threshold is None:
                result[name] = {"achievable_on_CAL": False}
            else:
                value = float(threshold["threshold"])
                result[name] = {
                    "achievable_on_CAL": True, "frozen_CAL_threshold": value,
                    "HOLD_POOLED": apply_threshold(hold_units, hold_reports, value, score, "POOLED"),
                    "HOLD_BASE76": apply_threshold(hold_units, hold_reports, value, score, "BASE76"),
                    "HOLD_HIGH129": apply_threshold(hold_units, hold_reports, value, score, "HIGH129"),
                }
        return result

    rho_held = validation(rho_thresholds, "rho_zero")
    abundance_held = validation(abundance_thresholds, "X_hat")

    preferred = rho_held.get("tau_cal_FDR5", {})
    if not preferred.get("achievable_on_CAL"):
        preferred = rho_held.get("tau_cal_FDR1", {})
    if preferred.get("achievable_on_CAL"):
        base = preferred["HOLD_BASE76"]
        high = preferred["HOLD_HIGH129"]
        delta = float(high["FDR"] - base["FDR"])
        direction = "higher" if delta > 0 else ("lower" if delta < 0 else "equal")
        complexity_interpretation = (
            f"At the frozen CAL threshold {preferred['frozen_CAL_threshold']:.8g}, "
            f"HOLD molecular FDR was {base['FDR']:.8g} for BASE76 and "
            f"{high['FDR']:.8g} for HIGH129. Increasing global truth complexity "
            f"therefore produced {direction} observed FDR (HIGH minus BASE = "
            f"{delta:.8g}); this two-replicate comparison is descriptive and the "
            "threshold was not retuned by condition."
        )
    else:
        complexity_interpretation = (
            "Neither requested CAL FDR operating point was achievable, so no frozen "
            "rho_zero threshold was available for a BASE76-versus-HIGH129 HOLD comparison."
        )
    write_csv(output / "calibration_records.csv", cal_records, list(cal_records[0]))
    write_csv(output / "heldout_records.csv", hold_records, list(hold_records[0]))
    curve_fields = ["scope", "score", "threshold", "n_retained", "TP", "FP", "precision", "FDR",
                    "all_truth_recall", "reportable_truth_recall", "coverage"]
    write_csv(output / "rho_calibration_threshold_curve.csv", rho_curve, curve_fields)
    write_csv(output / "abundance_calibration_threshold_curve.csv", abundance_curve, curve_fields)
    stage0.atomic_write_json(output / "rho_heldout_validation.json", rho_held)
    stage0.atomic_write_json(output / "abundance_heldout_validation.json", abundance_held)
    design = read_json(output / "design.json")
    raw_results = {name: report["learned_raw_identity_performance"] for name, report in all_reports.items()}
    report = {
        "status": "PENDING_REVIEW", "script_version": VERSION,
        "global_truth_K_conditions": {"BASE76": BASE_K, "HIGH129": HIGH_K},
        "true_per_pixel_K_distributions": {name: item["true_per_pixel_K_foreground"] for name, item in all_reports.items()},
        "global_scaling_factors": {name: item["global_signal_scalar"] for name, item in all_reports.items()},
        "geometry_balance": design["geometry_balance"],
        "oracle_results": {name: item["oracle"] for name, item in all_reports.items()},
        "learned_raw_results": raw_results,
        "rho_zero_calibration": rho_thresholds,
        "rho_heldout_validation": rho_held,
        "abundance_only_baseline": {
            "calibration_thresholds": abundance_thresholds,
            "heldout_validation": abundance_held,
        },
        "training_epochs_and_stop_reasons": {
            name: {"final_epoch": item["training"]["final_epoch"], "stop_reason": item["training"]["stop_reason"]}
            for name, item in all_reports.items()
        },
        "base76_vs_high129_interpretation": complexity_interpretation,
        "validity_status": "PENDING_REVIEW", "decision": "PENDING_RESULT_REVIEW",
        "final_deployment_threshold_frozen": False,
    }
    stage0.atomic_write_json(output / "report.json", report)
    stage0.atomic_write_text(output / "summary.md", build_summary(report))
    update_log(report, design)
    return report


def run_child(args: argparse.Namespace, dataset_id: str, gpu: str) -> None:
    command = [
        sys.executable, str(Path(__file__).resolve()),
        "--asset-root", str(args.asset_root.resolve()),
        "--output-dir", str(args.output_dir.resolve()),
        "--dataset", dataset_id, "--rho-workers", str(args.rho_workers),
    ]
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = gpu
    completed = subprocess.run(command, env=environment, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"DATASET_WORKER_FAILED: {dataset_id} on GPU {gpu}")


def run_all(args: argparse.Namespace) -> None:
    pending = [name for name in DATASETS if not (
        (args.output_dir / name / "report.json").exists()
        and read_json(args.output_dir / name / "report.json").get("status") == "COMPLETE"
    )]
    if args.parallel_gpus:
        gpus = [item.strip() for item in args.parallel_gpus.split(",") if item.strip()]
        if len(gpus) != len(set(gpus)) or not gpus:
            raise RuntimeError("INVALID_PARALLEL_GPU_LIST")
        queues = {gpu: pending[position::len(gpus)] for position, gpu in enumerate(gpus)}
        with ThreadPoolExecutor(max_workers=len(gpus)) as executor:
            futures = [executor.submit(lambda gpu=gpu: [run_child(args, name, gpu) for name in queues[gpu]])
                       for gpu in gpus]
            for future in futures:
                future.result()
    else:
        for name in pending:
            run_child(args, name, "0")
    aggregate(args.output_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-root", type=Path, default=ROOT.parent / "decon-lipid")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DEFAULT)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare-design", action="store_true")
    mode.add_argument("--oracle-only", action="store_true")
    mode.add_argument("--dataset", choices=DATASETS)
    mode.add_argument("--all", action="store_true")
    mode.add_argument("--aggregate", action="store_true")
    mode.add_argument("--dry-run", action="store_true")
    parser.add_argument("--parallel-gpus")
    parser.add_argument("--rho-workers", type=int, default=1)
    args = parser.parse_args()
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
    context = load_context(args.asset_root)
    if context["missing_assets"] or context["missing_results"]:
        payload = {
            "status": "DRY_RUN_READY_REMOTE_ASSETS_REQUIRED",
            "missing_locked_assets": context["missing_assets"],
            "missing_v51_v53_results": context["missing_results"],
            "singleton_identity_count": len(context.get("singleton_indices", [])) or None,
            "base_template_count": len(context.get("base_templates", [])) or None,
            "high_template_count": len(context.get("high_templates", [])) or None,
            "planned_datasets": list(DATASETS), "outputs_written": False,
            "GPU_training_performed": False,
        }
        if args.dry_run:
            print(json.dumps(payload, ensure_ascii=False, indent=2))
            return
        raise RuntimeError("REQUIRED_ASSETS_OR_RESULTS_MISSING: " + json.dumps(payload))
    bundle = build_design(context)
    if args.dry_run:
        print(json.dumps(stage0.to_jsonable({
            "status": "DRY_RUN_READY", "singleton_identity_count": len(bundle["features"]),
            "interference_features": list(BALANCE_FEATURES) + ["lipid_class"],
            "matching": bundle["design"]["matching"],
            "base76_subset": bundle["design"]["base76_subset"],
            "template_banks": bundle["design"]["template_banks"],
            "mapping_seeds": MAPPING_SEEDS, "datasets": list(DATASETS),
            "global_signal_target": TARGET_SIGNAL_P50,
            "training_cap": MAX_EPOCH, "checkpoint_epochs": list(CHECKPOINT_EPOCHS),
            "outputs_written": False, "GPU_training_performed": False,
        }), ensure_ascii=False, indent=2, allow_nan=False))
        return
    if args.prepare_design or args.oracle_only or args.all:
        write_design(args.output_dir, bundle)
    elif args.dataset and not (args.output_dir / "design.json").exists():
        raise RuntimeError("DESIGN_NOT_PREPARED: run --prepare-design first")
    if args.prepare_design:
        print(json.dumps({"status": "DESIGN_PREPARED", "output_dir": str(args.output_dir)}, indent=2))
        return
    if args.oracle_only:
        reports = [run_dataset(args, context, name, oracle_only=True) for name in DATASETS]
        print(json.dumps(stage0.to_jsonable({"status": "ORACLE_ONLY_COMPLETE", "datasets": reports}),
                         ensure_ascii=False, indent=2, allow_nan=False))
        return
    if args.dataset:
        validate_frozen_design(args.output_dir, context)
        report = run_dataset(args, context, args.dataset)
        print(json.dumps(stage0.to_jsonable(report), ensure_ascii=False, indent=2, allow_nan=False))
        return
    run_all(args)


if __name__ == "__main__":
    main()
