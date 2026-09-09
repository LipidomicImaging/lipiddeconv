#!/usr/bin/env python3
"""V56 fixed-total-signal molecular-identity benchmark; training requires frozen design.

Mapping replicates are not biological replicates. Empirical CAL FDR thresholds
do not guarantee population FDR. No measurement noise is synthesized.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "analysis"))
import run_v54_complexity_calibration as v54

stage0 = v54.stage0
VERSION = "v56_paper_identity_fdr_benchmark"
OUTPUT_DEFAULT = ROOT / "results" / VERSION
K_LEVELS = (50, 75, 100, 125, 150, 175)
ROBUST_K = (50, 100, 150)
IDENTITY_SPLIT_SEED = 5600
MAPPING_SEEDS = {f"R{i}": 5600 + i for i in range(1, 6)}
FEATURES = (*v54.CONTINUOUS_FEATURES, "d_single")
TARGET_SIGNAL = 0.6036783456802368
GATE = 1e-3
TEMPLATE_RELATIVE_FLOOR = 1e-8
TEMPLATE_AMPLITUDE_RULE = (
    "Divide each selected complete production map by its positive foreground mean before identity assignment; "
    "then apply only the existing one dataset-wide scalar, with no subsequent per-identity scaling."
)
MISMATCH = {
    "MISMATCH_MILD": {"fragment_log_sigma": 0.10, "fragment_dropout": 0.05,
                      "parent_min": 0.9, "parent_max": 1.1},
    "MISMATCH_MODERATE": {"fragment_log_sigma": 0.20, "fragment_dropout": 0.10,
                          "parent_min": 0.8, "parent_max": 1.2},
}
TABLES = ("identity_geometry", "matched_identity_pairs", "identity_blocks",
          "template_bank", "template_blocks", "template_mappings", "dataset_manifest")
LIMITATIONS = [
    "Synthetic global K is controlled mixture complexity, not the unknown biological K of tissue.",
    "FIXED-TOTAL-SIGNAL COMPLEXITY STRESS TEST: as K increases, more components share the same total spectral signal budget; per-component abundance is not held fixed.",
    "CAL/HOLD is molecular-identity separation, not network train/test splitting.",
    "Mapping replicates repeat spatial/amplitude-to-identity assignments, not biological samples.",
    "FDR calibration is empirical and does not guarantee 1% or 5% FDR.",
    "Wilson intervals are descriptive binomial intervals; repeated identities, nested K and shared templates induce dependence and are not independent biological trials.",
    "Spectral mismatch excludes measurement noise: locked production B background is zeroed. No Gaussian, low-signal foreground or production residual noise fallback is permitted.",
    "Log gaps involving numerical-zero false rho are descriptive, not physical quantities.",
    "This is a controlled fixed-total-signal complexity benchmark using foreground-mean-normalized real spatial templates; it preserves spatial morphology but does not preserve the original between-lipid abundance distribution.",
]


def require(ok: bool, message: str) -> None:
    if not ok:
        raise RuntimeError(message)


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def payload_hash(value) -> str:
    return hashlib.sha256(json.dumps(stage0.to_jsonable(value), sort_keys=True,
                                     allow_nan=False).encode()).hexdigest()


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    stage0.atomic_write_json(path, value)


def write_rows(path: Path, rows: list[dict], fields=None) -> None:
    fields = fields or list(dict.fromkeys(key for row in rows for key in row))
    encoded = [{k: json.dumps(v, sort_keys=True) if isinstance(v, (list, dict)) else v
                for k, v in row.items()} for row in rows]
    v54.write_csv(path, encoded, fields)


def read_rows(path: Path) -> list[dict]:
    return [{k: v54.parse_scalar(v) for k, v in row.items()} for row in v54.read_csv(path)]


def dataset_ids(levels=K_LEVELS, split=None) -> list[str]:
    return [f"{s}_{r}_K{k:03d}" for s in ((split,) if split else ("CAL", "HOLD"))
            for r in MAPPING_SEEDS for k in levels]


def load_context(asset_root: Path) -> dict:
    # Reuse locked-array validation, without V54's historical 76/129 template gate.
    lock = stage0.load_lock()
    paths = stage0.resolve_locked_paths(asset_root.resolve(), lock)
    results = v54.required_results()
    missing = stage0.missing_paths(paths) + [str(p) for p in results.values() if not p.exists()]
    if missing:
        return {"missing_dependencies": missing}
    validation, provenance = stage0.validate_shapes_and_hashes(paths, lock)
    A_raw, A_solver, X_real, metadata, mask = v54.v53.load_locked_arrays(paths)
    require(A_solver.shape == (1084, 391), "FULL_LIBRARY_SHAPE_CHANGED")
    require(np.array_equal(A_solver, v54.v52.normalize_like_get_A_matrix(A_raw)),
            "PRODUCTION_NORMALIZATION_CHANGED")
    stage_design, domain = v54.v53.validate_stage0_results(results, validation, paths)
    summary = v54.read_json(results["v51_summary"])
    require(summary.get("status") == "COMPLETE", "V51_NOT_COMPLETE")
    for key in ("A_library", "candidate_metadata", "channel_axis"):
        require(summary["asset_validation"]["validated_hashes"].get(key) ==
                validation["hashes_sha256"].get(key), f"V51_HASH_MISMATCH: {key}")
    counts = Counter(str(n) for n in metadata["lipid_name"])
    singleton = np.array([i for i, n in enumerate(metadata["lipid_name"])
                          if counts[str(n)] == 1], dtype=int)
    require(len(singleton) >= 350, f"INSUFFICIENT_SINGLETONS: {len(singleton)}")
    require(np.isfinite(X_real).all() and (X_real >= 0).all() and mask.any(),
            "INVALID_PRODUCTION_TEMPLATES")
    return {"paths": paths, "validation": validation, "provenance": provenance,
            "A_raw": A_raw, "A_solver": A_solver, "X_real": X_real,
            "metadata": metadata, "mask": mask, "singleton_indices": singleton,
            "foreground_means": X_real[:, mask].mean(axis=1, dtype=np.float64),
            "stage0_design": stage_design, "stage0_domain": domain,
            "geometry": v54.v53.load_geometry_rows(results["v51_candidate_geometry"], metadata),
            "result_hashes": {k: digest(p) for k, p in results.items()}}


def balanced_blocks(rows: list[dict], score: str, class_key=None) -> list[dict]:
    """Exact five members per rank quintile per block; greedily balance class counts."""
    require(len(rows) == 175, "BLOCK_INPUT_COUNT")
    ordered = sorted(rows, key=lambda r: (r[score], r["item_id"]))
    class_counts = [Counter() for _ in range(7)]
    sizes = [0] * 7
    for q in range(5):
        chunk = ordered[q * 35:(q + 1) * 35]
        frequencies = Counter(r[class_key] for r in chunk) if class_key else Counter()
        if class_key:
            chunk = sorted(chunk, key=lambda r: (-frequencies[r[class_key]], r[class_key],
                                                  r[score], r["item_id"]))
        quota = [0] * 7
        for row in chunk:
            label = row[class_key] if class_key else "all"
            block = min((b for b in range(7) if quota[b] < 5),
                        key=lambda b: (class_counts[b][label], sizes[b], (b - q) % 7))
            row.update({"quintile": q + 1, "block": block + 1})
            quota[block] += 1
            sizes[block] += 1
            class_counts[block][label] += 1
    return sorted(rows, key=lambda r: (r["block"], r["quintile"], r["item_id"]))


def matched_pairs(features: list[dict]) -> list[dict]:
    by_class = defaultdict(list)
    for row in features:
        by_class[row["lipid_class"]].append(row)
    pairs, leftovers = [], []

    def consume(group):
        remaining = sorted(group, key=lambda r: r["candidate_index"])
        while len(remaining) > 1:
            _, i, j = min(
                (sum((a[f"rank_{f}"] - b[f"rank_{f}"]) ** 2 for f in FEATURES), i, j)
                for i, a in enumerate(remaining) for j, b in enumerate(remaining) if i < j)
            a, b = remaining[i], remaining[j]
            pairs.append({"a": a, "b": b,
                          "score": (a["ambiguity_design_score"] + b["ambiguity_design_score"]) / 2})
            remaining.pop(j)
            remaining.pop(i)
        return remaining

    for label in sorted(by_class):
        leftovers.extend(consume(by_class[label]))
    consume(leftovers)
    pairs.sort(key=lambda p: (p["score"], p["a"]["candidate_index"]))
    require(len(pairs) >= 175, "INSUFFICIENT_MATCHED_PAIRS")
    selected = [pairs[i] for i in v54.spanning_positions(len(pairs), 175)]
    rng = np.random.default_rng(IDENTITY_SPLIT_SEED)
    result = []
    for i, pair in enumerate(selected):
        a, b = pair["a"], pair["b"]
        if rng.integers(2):
            a, b = b, a
        result.append({"item_id": i + 1, "pair_id": i + 1,
                       "CAL_index": a["candidate_index"], "HOLD_index": b["candidate_index"],
                       "CAL_lipid_name": a["lipid_name"], "HOLD_lipid_name": b["lipid_name"],
                       "CAL_class": a["lipid_class"], "HOLD_class": b["lipid_class"],
                       "class_pair": "|".join(sorted((a["lipid_class"], b["lipid_class"]))),
                       "matched_within_class": a["lipid_class"] == b["lipid_class"],
                       "pair_ambiguity_score": pair["score"],
                       "rank_feature_distance": math.sqrt(sum(
                           (a[f"rank_{f}"] - b[f"rank_{f}"]) ** 2 for f in FEATURES))})
    return balanced_blocks(result, "pair_ambiguity_score", "class_pair")


def make_tables(context: dict) -> tuple[dict, dict]:
    features = v54.interference_features(context)
    # All six audited geometry features enter distance, with average-tie ranks.
    ranks = v54.rank_percentiles(np.array([r["d_single"] for r in features]))
    for row, rank in zip(features, ranks):
        row["rank_d_single"] = float(rank)
    pairs = matched_pairs(features)
    used = {p[f"{s}_index"] for p in pairs for s in ("CAL", "HOLD")}
    for row in features:
        row["selected_for_truth"] = row["candidate_index"] in used
    means = context["foreground_means"]
    floor = max(float(means.max()) * TEMPLATE_RELATIVE_FLOOR, float(np.finfo(np.float32).tiny))
    eligible = sorted(np.flatnonzero(np.isfinite(means) & (means > floor)).tolist(),
                      key=lambda i: (means[i], i))
    require(len(eligible) >= 175,
            f"INSUFFICIENT_NONDEGENERATE_TEMPLATES: {len(eligible)}; floor={floor}; do not retune")
    selected = [eligible[i] for i in v54.spanning_positions(len(eligible), 175)]
    templates = balanced_blocks([
        {"item_id": i, "original_candidate_index": i, "template_rank": rank + 1,
         "foreground_mean": float(means[i]),
         "occupancy_fraction": float(np.mean(context["X_real"][i, context["mask"]] > 0))}
        for rank, i in enumerate(selected)], "foreground_mean")
    mappings = []
    for replicate, seed in MAPPING_SEEDS.items():
        rng = np.random.default_rng(seed)
        for block in range(1, 8):
            block_pairs = [p for p in pairs if p["block"] == block]
            block_templates = [t for t in templates if t["block"] == block]
            for pair, pos in zip(block_pairs, rng.permutation(25)):
                template = block_templates[int(pos)]
                for split in ("CAL", "HOLD"):
                    mappings.append({"replicate": replicate, "mapping_seed": seed, "split": split,
                                     "block": block, "pair_id": pair["pair_id"],
                                     "synthetic_truth_candidate_index": pair[f"{split}_index"],
                                     "synthetic_truth_lipid_name": pair[f"{split}_lipid_name"],
                                     "template_original_candidate_index": template["original_candidate_index"]})
    manifest = []
    for condition, levels in [("CLEAN", K_LEVELS), *((c, ROBUST_K) for c in MISMATCH)]:
        for dataset in dataset_ids(levels):
            split, replicate, label = dataset.split("_")
            manifest.append({"condition": condition, "dataset_id": dataset, "split": split,
                             "replicate": replicate, "K": int(label[1:]),
                             "directory": str(dataset_directory(Path("."), condition, dataset)).replace("\\", "/"),
                             "status": "PLANNED", "global_signal_scalar": None})
    tables = {"identity_geometry": features, "matched_identity_pairs": pairs,
              "identity_blocks": [{k: p[k] for k in ("pair_id", "block", "quintile", "pair_ambiguity_score", "class_pair")} for p in pairs],
              "template_bank": templates,
              "template_blocks": [{k: t[k] for k in ("original_candidate_index", "block", "quintile", "foreground_mean")} for t in templates],
              "template_mappings": mappings, "dataset_manifest": manifest}
    details = {"singleton_identity_count": len(features),
               "unused_singleton_identities": [r for r in features if not r["selected_for_truth"]],
               "template_numerical_floor": floor, "template_relative_floor": TEMPLATE_RELATIVE_FLOOR,
               "eligible_template_count": len(eligible),
               "excluded_template_indices": [i for i in range(391) if i not in eligible],
               "unused_eligible_template_indices": [i for i in eligible if i not in selected]}
    return tables, details


def normalized_template_bank(context: dict, templates: list[dict]) -> tuple[np.ndarray, dict]:
    """Normalize full maps once; retain raw X_real for selection and provenance."""
    indices = tuple(sorted(t["original_candidate_index"] for t in templates))
    cached = context.get("normalized_template_bank")
    if cached is not None and cached[0] == indices:
        return cached[1], cached[2]
    bank = np.zeros_like(context["X_real"], dtype=np.float32)
    metadata = {}
    for i in indices:
        original = context["X_real"][i]
        mean = float(original[context["mask"]].mean(dtype=np.float64))
        require(np.isfinite(mean) and mean > 0, f"INVALID_TEMPLATE_FOREGROUND_MEAN: {i}")
        normalized = (original.astype(np.float64) / mean).astype(np.float32)
        require(np.isfinite(normalized).all(), f"NONFINITE_NORMALIZED_TEMPLATE: {i}")
        require(np.array_equal(original > 0, normalized > 0), f"TEMPLATE_OCCUPANCY_CHANGED: {i}")
        require(np.allclose(normalized.astype(np.float64) * mean, original, rtol=2e-6, atol=0),
                f"TEMPLATE_SPATIAL_SHAPE_CHANGED: {i}")
        normalized_mean = float(normalized[context["mask"]].mean(dtype=np.float64))
        require(np.isclose(normalized_mean, 1.0, rtol=2e-6), f"NORMALIZED_TEMPLATE_MEAN_NOT_ONE: {i}")
        bank[i] = normalized
        metadata[i] = {"template_amplitude_rule": TEMPLATE_AMPLITUDE_RULE,
                       "pre_normalization_foreground_mean": mean,
                       "normalized_foreground_mean": normalized_mean}
    context["normalized_template_bank"] = (indices, bank, metadata)
    return bank, metadata


def annotate_template_amplitudes(tables: dict, context: dict) -> None:
    _, metadata = normalized_template_bank(context, tables["template_bank"])
    for template in tables["template_bank"]:
        template.update(metadata[template["original_candidate_index"]])


def audit_tables(tables: dict, context: dict) -> dict:
    pairs, templates, mappings = (tables[k] for k in
                                 ("matched_identity_pairs", "template_bank", "template_mappings"))
    require(len(pairs) == 175 and len(templates) == 175, "MASTER_BANK_SIZE")
    names = {s: {p[f"{s}_lipid_name"] for p in pairs} for s in ("CAL", "HOLD")}
    require(len(names["CAL"]) == len(names["HOLD"]) == 175 and
            not names["CAL"] & names["HOLD"], "IDENTITY_SPLIT_LEAKAGE")
    singleton = set(context["singleton_indices"].tolist())
    for p in pairs:
        for s in names:
            i = p[f"{s}_index"]
            require(i in singleton and str(context["metadata"]["lipid_name"][i]) == p[f"{s}_lipid_name"],
                    "NON_SINGLETON_OR_INCORRECT_TRUTH_IDENTITY")
    require(len({t["original_candidate_index"] for t in templates}) == 175, "DUPLICATE_TEMPLATE")
    for rows in (pairs, templates):
        require(Counter(r["block"] for r in rows) == Counter({b: 25 for b in range(1, 8)}), "BLOCK_SIZE")
        require(Counter((r["block"], r["quintile"]) for r in rows) ==
                Counter({(b, q): 5 for b in range(1, 8) for q in range(1, 6)}), "QUINTILE_BALANCE")
    require(len(mappings) == 175 * 5 * 2, "MAPPING_COUNT")
    pmap = {p["pair_id"]: p for p in pairs}
    tmap = {t["original_candidate_index"]: t for t in templates}
    signatures = []
    for replicate in MAPPING_SEEDS:
        subset = [m for m in mappings if m["replicate"] == replicate]
        for split in names:
            rows = [m for m in subset if m["split"] == split]
            require(len(rows) == 175 and {m["synthetic_truth_lipid_name"] for m in rows} == names[split],
                    "CROSS_REPLICATE_MEMBERSHIP_CHANGED")
            require(len({m["template_original_candidate_index"] for m in rows}) == 175, "TEMPLATE_REUSED")
            previous = {}
            for k in K_LEVELS:
                current = {m["pair_id"]: m["template_original_candidate_index"] for m in rows if m["block"] <= k // 25}
                require(len(current) == k and all(current.get(p) == t for p, t in previous.items()), "K_NESTING_FAILED")
                previous = current
        for m in subset:
            p = pmap[m["pair_id"]]
            require(m["mapping_seed"] == MAPPING_SEEDS[replicate] and
                    m["synthetic_truth_candidate_index"] == p[f"{m['split']}_index"] and
                    m["block"] == p["block"] == tmap[m["template_original_candidate_index"]]["block"],
                    "MAPPING_OUTSIDE_FROZEN_BLOCK")
        for pair_id in pmap:
            rows = [m for m in subset if m["pair_id"] == pair_id]
            require(len(rows) == 2 and len({m["template_original_candidate_index"] for m in rows}) == 1,
                    "CAL_HOLD_TEMPLATE_MISMATCH")
        signatures.append(tuple(m["template_original_candidate_index"] for m in subset if m["split"] == "CAL"))
    require(len(set(signatures)) == 5, "MAPPING_REPLICATES_NOT_DISTINCT")
    geometry = {r["candidate_index"]: r for r in tables["identity_geometry"]}
    expected_manifest = {(condition, dataset) for condition, levels in
                         [("CLEAN", K_LEVELS), *((c, ROBUST_K) for c in MISMATCH)]
                         for dataset in dataset_ids(levels)}
    require(len(tables["dataset_manifest"]) == 120 and
            {(r["condition"], r["dataset_id"]) for r in tables["dataset_manifest"]} == expected_manifest,
            "DATASET_MANIFEST_COUNT_OR_MEMBERSHIP")
    require(tables["identity_blocks"] == [
        {k: p[k] for k in ("pair_id", "block", "quintile", "pair_ambiguity_score", "class_pair")}
        for p in pairs], "IDENTITY_BLOCK_TABLE_MISMATCH")
    require(tables["template_blocks"] == [
        {k: t[k] for k in ("original_candidate_index", "block", "quintile", "foreground_mean")}
        for t in templates], "TEMPLATE_BLOCK_TABLE_MISMATCH")
    for t in templates:
        require(np.isclose(t["foreground_mean"], context["foreground_means"][t["original_candidate_index"]],
                           rtol=1e-12, atol=0), "TEMPLATE_MEAN_CHANGED")
    _, amplitude_metadata = normalized_template_bank(context, templates)
    for t in templates:
        expected = amplitude_metadata[t["original_candidate_index"]]
        # Fresh dry-run tables may not yet have annotation; saved designs must agree.
        for key, value in expected.items():
            if key in t:
                require(t[key] == value, f"TEMPLATE_AMPLITUDE_METADATA_CHANGED: {key}")
    balance = {}
    for k in K_LEVELS:
        ps = [p for p in pairs if p["block"] <= k // 25]
        ts = [t for t in templates if t["block"] <= k // 25]
        balance[str(k)] = {
            "template_abundance": v54.numeric_summary([t["foreground_mean"] for t in ts]),
            "normalized_template_abundance": v54.numeric_summary([
                amplitude_metadata[t["original_candidate_index"]]["normalized_foreground_mean"] for t in ts]),
            "pair_ambiguity": v54.numeric_summary([p["pair_ambiguity_score"] for p in ps]),
            "identity_quintile_counts": dict(Counter(p["quintile"] for p in ps)),
            "template_quintile_counts": dict(Counter(t["quintile"] for t in ts)),
            "splits": {s: {"lipid_classes": dict(Counter(p[f"{s}_class"] for p in ps)),
                           "geometry": {f: v54.numeric_summary([geometry[p[f"{s}_index"]][f] for p in ps])
                                        for f in FEATURES}} for s in names}}
    return {"status": "PASS", "unique_singleton_truth_names": 350, "CAL_HOLD_overlap": 0,
            "CAL_HOLD_membership_fixed_across_replicates": True, "matched_pairs": 175,
            "identity_blocks": 7, "template_blocks": 7, "members_per_block": 25,
            "members_per_quintile_per_block": 5, "templates": 175,
            "exact_K_nesting": True, "CAL_HOLD_same_template": True,
            "replicates_change_only_within_block_mapping": True, "solver_candidates": 391,
            "training_input_keys": ["case_name", "B_sim", "foreground_mask"],
            "balance_by_K": balance, "class_balance_is_approximate": True,
            "template_amplitude_rule": TEMPLATE_AMPLITUDE_RULE,
            "template_amplitudes": amplitude_metadata,
            "complete_spatial_shape_and_occupancy_preserved": True,
            "dataset_truth_complexity": {
                f"{r['condition']}/{r['dataset_id']}": {
                    key: (json.loads(r[key]) if key == "K_true_pixel_foreground" and isinstance(r[key], str) else r[key])
                    for key in ("reportable_truth_global_count", "reportable_fraction", "K_true_pixel_foreground")}
                for r in tables["dataset_manifest"] if "reportable_fraction" in r},
            "available_singleton_ambiguity": v54.numeric_summary(
                [r["ambiguity_design_score"] for r in geometry.values()]),
            "selected_identity_ambiguity": v54.numeric_summary(
                [geometry[p[f"{s}_index"]]["ambiguity_design_score"] for p in pairs for s in names]),
            "unused_identity_ambiguity": v54.numeric_summary(
                [r["ambiguity_design_score"] for r in geometry.values() if not r["selected_for_truth"]]),
            "matching_distance": v54.numeric_summary([p["rank_feature_distance"] for p in pairs]),
            "cross_class_pairs": sum(not p["matched_within_class"] for p in pairs)}


def dataset_directory(output: Path, condition: str, dataset: str) -> Path:
    return output / "clean" / dataset if condition == "CLEAN" else output / "robustness" / condition / dataset


def construct_case(context: dict, tables: dict, dataset: str, condition="CLEAN") -> tuple[dict, list]:
    split, replicate, label = dataset.split("_")
    k = int(label[1:])
    require(dataset in dataset_ids(K_LEVELS if condition == "CLEAN" else ROBUST_K), "INVALID_DATASET")
    mapping = [m for m in tables["template_mappings"] if m["split"] == split and
               m["replicate"] == replicate and m["block"] <= k // 25]
    require(len(mapping) == k, "DATASET_MAPPING_COUNT")
    bank, _ = normalized_template_bank(context, tables["template_bank"])
    # V54 assigns these normalized templates and performs the unchanged single
    # dataset scalar. Raw production maps in the original context remain untouched.
    case = v54.construct_dataset({**context, "X_real": bank}, mapping, dataset)
    case.update({"template_amplitude_rule": TEMPLATE_AMPLITUDE_RULE,
                 "reportable_truth_global_count": case["reportable_truth_molecular_identity_count"],
                 "reportable_fraction": case["reportable_truth_molecular_identity_count"] / k})
    require(np.isclose(case["achieved_foreground_B_l2_p50"], TARGET_SIGNAL, rtol=2e-6), "SIGNAL_NORMALIZATION_FAILED")
    perturbations = []
    if condition != "CLEAN":
        cfg = MISMATCH[condition]
        A = context["A_solver"].copy()
        mz = np.asarray(np.load(context["paths"]["channel_axis"])).reshape(-1)
        parent = (mz >= 748.0) & (mz <= 803.0)
        require(parent.any() and not parent.all(), "INVALID_PARENT_FRAGMENT_PARTITION")
        for i in case["active_indices"]:
            original = A[:, i].copy()
            seed = int.from_bytes(hashlib.sha256(f"{VERSION}|{condition}|{i}".encode()).digest()[:8], "little")
            rng = np.random.default_rng(seed)
            fragment = np.flatnonzero((~parent) & (original != 0))
            A[fragment, i] *= np.exp(rng.normal(0, cfg["fragment_log_sigma"], len(fragment))).astype(np.float32)
            drop = rng.random(len(fragment)) < cfg["fragment_dropout"]
            if len(drop) and drop.all():
                drop[int(rng.integers(len(drop)))] = False
            A[fragment[drop], i] = 0
            multiplier = float(rng.uniform(cfg["parent_min"], cfg["parent_max"]))
            A[parent, i] *= np.float32(multiplier)
            A[:, i] = v54.v52.normalize_like_get_A_matrix(A[:, i:i + 1])[:, 0]
            require(np.isclose(np.linalg.norm(A[:, i]), 1, rtol=2e-6), "PERTURBED_NORM")
            perturbations.append({"candidate_index": i, "seed": seed, "parent_multiplier": multiplier,
                                  "fragment_count": len(fragment), "dropped_count": int(drop.sum()),
                                  "cosine": float(np.dot(original, A[:, i]) /
                                                  (np.linalg.norm(original) * np.linalg.norm(A[:, i])))})
        case["B_sim"] = np.einsum("mc,cyx->myx", A, case["X_true"], optimize=True).astype(np.float32)
        # Keep exactly the CLEAN X_true and its one global scalar: no mismatch rescaling.
    return case, perturbations


def implementation_hashes() -> dict:
    return {str(p.relative_to(ROOT)): digest(p) for p in
            (Path(__file__).resolve(), ROOT / "analysis/run_v54_complexity_calibration.py",
             ROOT / "src/rho_zero.py", ROOT / "src/config_758.py",
             ROOT / "src/lipid_ista.py", ROOT / "src/run_758_ista.py")}


def design_fingerprint(design: dict) -> str:
    payload = {k: v for k, v in design.items() if k not in ("design_fingerprint", "review_confirmed_utc")}
    payload["status"] = "DESIGN_PREPARED"
    return payload_hash(payload)


def load_design(output: Path, context: dict, frozen=False, amplitude_upgrade=False) -> tuple[dict, dict]:
    design = v54.read_json(output / "design.json")
    require(design["design_fingerprint"] == design_fingerprint(design), "DESIGN_CONTENT_CHANGED")
    require(design["script_version"] == VERSION, "DESIGN_VERSION_MISMATCH")
    require(design["status"] in ("DESIGN_PREPARED", "DESIGN_FROZEN_BEFORE_TRAINING"), "DESIGN_STATUS")
    if frozen:
        require(design["status"] == "DESIGN_FROZEN_BEFORE_TRAINING", "EXPLICIT_DESIGN_REVIEW_AND_FREEZE_REQUIRED")
    require(design["input_hashes"] == context["validation"]["hashes_sha256"] and
            design["result_hashes"] == context["result_hashes"], "FROZEN_INPUT_CHANGED")
    if amplitude_upgrade:
        require(design["status"] == "DESIGN_PREPARED" and
                design.get("template_amplitude_rule") != TEMPLATE_AMPLITUDE_RULE,
                "AMPLITUDE_UPGRADE_REQUIRES_OLD_UNFROZEN_DESIGN")
        own_path = str(Path(__file__).resolve().relative_to(ROOT))
        require({k: v for k, v in design["implementation_hashes"].items() if k != own_path} ==
                {k: v for k, v in implementation_hashes().items() if k != own_path},
                "NON_AMPLITUDE_IMPLEMENTATION_CHANGED")
    else:
        require(design["implementation_hashes"] == implementation_hashes(), "FROZEN_IMPLEMENTATION_CHANGED")
        require(design.get("template_amplitude_rule") == TEMPLATE_AMPLITUDE_RULE,
                "PREPARE_NORMALIZED_TEMPLATE_DESIGN_FIRST")
    for name, checksum in design["table_hashes"].items():
        require(digest(output / f"{name}.csv") == checksum, f"FROZEN_TABLE_CHANGED: {name}")
    tables = {name: read_rows(output / f"{name}.csv") for name in TABLES}
    if amplitude_upgrade:
        annotate_template_amplitudes(tables, context)
    audit_tables(tables, context)
    return design, tables


def prepare(output: Path, context: dict) -> dict:
    previous = None
    if (output / "design.json").exists():
        previous = v54.read_json(output / "design.json")
        if previous.get("template_amplitude_rule") == TEMPLATE_AMPLITUDE_RULE:
            design, _ = load_design(output, context)
            return {"status": "EXISTING_DESIGN_VALIDATED_NOT_REDRAWN", "design_status": design["status"]}
        require(previous.get("status") == "DESIGN_PREPARED", "CANNOT_REGENERATE_FROZEN_DESIGN")
        # Refuse to invalidate any oracle, training, checkpoint or aggregation result.
        for subdir in (output / "clean", output / "robustness"):
            require(not subdir.exists() or not any(p.is_file() for p in subdir.rglob("*")),
                    "AMPLITUDE_REGENERATION_REQUIRES_NO_RUNTIME_RESULTS")
        require(not (output / "report.json").exists() and not (output / "global_frozen_thresholds.json").exists(),
                "AMPLITUDE_REGENERATION_REQUIRES_NO_ANALYSIS_RESULTS")
        oracle_summary = output / "oracle_summary.csv"
        require(not oracle_summary.exists() or not read_rows(oracle_summary),
                "AMPLITUDE_REGENERATION_REQUIRES_NO_ORACLE_RESULTS")
        previous, tables = load_design(output, context, amplitude_upgrade=True)
        details = {key: previous[key] for key in (
            "singleton_identity_count", "unused_singleton_identities", "template_numerical_floor",
            "template_relative_floor", "eligible_template_count", "excluded_template_indices",
            "unused_eligible_template_indices")}
    else:
        tables, details = make_tables(context)
    annotate_template_amplitudes(tables, context)
    audit_tables(tables, context)
    # CPU-only synthesis records true complexity and signal scaling, not oracle/training.
    clean_summaries = {}
    for dataset in dataset_ids():
        case, _ = construct_case(context, tables, dataset)
        clean_summaries[dataset] = {"global_signal_scalar": case["global_scale"],
                                   "global_truth_K": case["global_truth_K"],
                                   "reportable_truth_global_count": case["reportable_truth_molecular_identity_count"],
                                   "reportable_fraction": case["reportable_fraction"],
                                   "K_true_pixel_foreground": case["K_true_pixel_foreground"],
                                   "achieved_clean_signal_p50": case["achieved_foreground_B_l2_p50"]}
    for row in tables["dataset_manifest"]:
        row.update(clean_summaries[row["dataset_id"]])
        row["template_amplitude_rule"] = TEMPLATE_AMPLITUDE_RULE
    audit = audit_tables(tables, context)
    for name, rows in tables.items():
        write_rows(output / f"{name}.csv", rows)
    write_rows(output / "oracle_summary.csv", [], ["condition", "dataset_id", "status", "FP", "FN"])
    design = {"script_version": VERSION, "status": "DESIGN_PREPARED",
              "created_utc": datetime.now(timezone.utc).isoformat(),
              "K_LEVELS": list(K_LEVELS), "ROBUST_K": list(ROBUST_K),
              "identity_split_seed": IDENTITY_SPLIT_SEED, "mapping_seeds": MAPPING_SEEDS,
              "planned_clean_runs": 60, "planned_additional_mismatch_runs": 60,
              "identity_unit": "lipid_name", "singleton_truth_only": True,
              "matching_rule": "Within-class nearest neighbors in six percentile-rank features, then cross-class leftovers; select 175 evenly spaced ambiguity-ranked pairs.",
              "ambiguity_score": "mean(rank parent cosine, rank fragment cosine, rank full cosine, 1-rank cone isolation, rank collective gain)",
              "derived_geometry_provenance": "V54 audited convention: d_single=sqrt(max(0,1-max_full_cosine^2)); collective_gain=d_single-cone_isolation; no geometry optimization rerun",
              "block_rule": "7 x 25; exactly five members from each rank quintile per block; greedy class balance",
              "template_rule": "175 evenly spaced raw abundance ranks above prespecified numerical floor; original identity discarded; original indices, ranks, blocks and mappings retained; foreground-mean normalization before assignment",
              "template_amplitude_rule": TEMPLATE_AMPLITUDE_RULE,
              "template_amplitudes": {str(t["original_candidate_index"]): {
                  key: t[key] for key in ("pre_normalization_foreground_mean", "normalized_foreground_mean")}
                  for t in tables["template_bank"]},
              "dataset_truth_complexity": clean_summaries,
              "amplitude_regeneration": {"previous_design_fingerprint": previous["design_fingerprint"] if previous else None,
                                         "existing_indices_blocks_mappings_reused": previous is not None},
              "signal_target": TARGET_SIGNAL, "reporting_gate": GATE,
              "robustness": MISMATCH, "robustness_clean_reference": "Existing V56 CLEAN runs; never retrain CLEAN for robustness",
              "mismatch_truth_contract": "Reuse exact CLEAN X_true and global scalar; original A_solver used for training, diagnostic NNLS and rho_zero",
              "measurement_noise_status": "NOISE_SOURCE_UNAVAILABLE",
              "training": {**v54.v50.training_contract(), "implementation": "v54.train_resumable",
                           "epoch_ceiling": 3000, "scheduler_T_max": 5000,
                           "checkpoint_epochs": list(v54.CHECKPOINT_EPOCHS),
                           "same_dataset_resume_only": True, "X_true_used_in_training_or_stopping": False},
              "rho_zero": {"implementation": "src/rho_zero.py", "weighting": "W=I"},
              "input_hashes": context["validation"]["hashes_sha256"],
              "result_hashes": context["result_hashes"], "implementation_hashes": implementation_hashes(),
              "table_hashes": {name: digest(output / f"{name}.csv") for name in TABLES},
              "audit": audit, **details, "limitations": LIMITATIONS,
              "validity_status": "PENDING_REVIEW", "final_deployment_threshold_frozen": False}
    design["design_fingerprint"] = design_fingerprint(design)
    write_json(output / "design.json", design)
    update_log(output, design)
    return design


def truth_report(case: dict) -> dict:
    return {k: case[k] for k in ("global_truth_K", "global_truth_identity_count",
                                "template_amplitude_rule", "reportable_truth_global_count", "reportable_fraction",
                                "all_truth_molecular_identity_count", "reportable_truth_molecular_identity_count",
                                "subthreshold_truth_molecular_identity_count", "K_true_pixel_foreground", "global_scale")}


def oracle_one(output: Path, context: dict, design: dict, tables: dict, dataset: str,
               condition="CLEAN") -> dict:
    path = dataset_directory(output, condition, dataset) / "oracle.json"
    if path.exists():
        cached = v54.read_json(path)
        require(cached["design_fingerprint"] == design["design_fingerprint"] and
                cached["condition"] == condition and cached["dataset_id"] == dataset,
                "CACHED_ORACLE_DESIGN_MISMATCH")
        return cached
    case, perturbations = construct_case(context, tables, dataset, condition)
    oracle = v54.reporting_gate_oracle(case, context["A_solver"], context["metadata"])
    record = {"condition": condition, "dataset_id": dataset, "design_fingerprint": design["design_fingerprint"],
              **truth_report(case), "oracle": oracle, "perturbations": perturbations,
              "status": oracle["status"] if condition == "CLEAN" else "DIAGNOSTIC_ONLY"}
    write_json(path, record)
    return record


def refresh_oracle_summary(output: Path) -> None:
    rows = []
    for condition, levels in [("CLEAN", K_LEVELS), *((c, ROBUST_K) for c in MISMATCH)]:
        for dataset in dataset_ids(levels):
            path = dataset_directory(output, condition, dataset) / "oracle.json"
            if not path.exists():
                continue
            record = v54.read_json(path)
            metrics = record["oracle"]["reportable_truth_metrics"]
            rows.append({"condition": condition, "dataset_id": dataset, "status": record["status"],
                         "FP": metrics["FP"], "FN": metrics["FN"],
                         "all_truth_metrics": record["oracle"]["all_truth_metrics"],
                         "reportable_truth_metrics": metrics})
    write_rows(output / "oracle_summary.csv", rows,
               ["condition", "dataset_id", "status", "FP", "FN", "all_truth_metrics", "reportable_truth_metrics"])


def preflight_oracles(output: Path, context: dict, design: dict, tables: dict,
                      datasets: list[str], summary=True) -> None:
    for dataset in datasets:
        record = oracle_one(output, context, design, tables, dataset)
        if summary:
            refresh_oracle_summary(output)
        require(record["status"] == "PASS", f"CLEAN_ORACLE_FAILED: {dataset}; saved failure; no identity redesign or GPU training")


def run_one(args, context: dict, design: dict, tables: dict, dataset: str, condition: str) -> dict:
    directory = dataset_directory(args.output_dir, condition, dataset)
    path = directory / "report.json"
    if path.exists():
        existing = v54.read_json(path)
        require(existing["design_fingerprint"] == design["design_fingerprint"], "CACHED_RESULT_DESIGN_MISMATCH")
        if existing["status"] == "COMPLETE":
            return existing
    oracle = oracle_one(args.output_dir, context, design, tables, dataset, condition)
    if condition == "CLEAN":
        require(oracle["status"] == "PASS", f"CLEAN_ORACLE_FAILED: {dataset}")
    case, perturbations = construct_case(context, tables, dataset, condition)
    runtime_contract = {"design_fingerprint": design["design_fingerprint"], "dataset_id": dataset,
                        "condition": condition, "B_sha256": hashlib.sha256(case["B_sim"].tobytes()).hexdigest()}
    contract_path = directory / "runtime_contract.json"
    if contract_path.exists():
        require(v54.read_json(contract_path) == runtime_contract, "CROSS_DATASET_OR_CHANGED_CHECKPOINT_INPUT")
    else:
        require(not list(directory.glob("*.pth")), "UNBOUND_RUNTIME_CHECKPOINT")
        write_json(contract_path, runtime_contract)
    training_case = {"case_name": f"V56__{condition}__{dataset}",
                     "B_sim": case["B_sim"], "foreground_mask": case["foreground_mask"]}
    from config_758 import Cfg
    expected_cfg = {**v54.v50.CFG_DEFAULTS, "K_layers": 12, "seed": 42,
                    "parent_channel_weight_multiplier": 1.0,
                    "calib_clamp_min": 0.5, "calib_clamp_max": 1.5}
    for key, value in expected_cfg.items():
        require(getattr(Cfg, key) == value, f"PRODUCTION_TRAINING_SETTING_CHANGED: {key}")
    require(v54.MAX_EPOCH == 3000 and tuple(v54.CHECKPOINT_EPOCHS) == (1000, 1500, 2000, 2500, 3000),
            "V54_TRAINING_DRIVER_CONTRACT_CHANGED")
    learned = v54.train_resumable(training_case, context, directory)
    report = {"dataset_id": dataset, "condition": condition,
              "design_fingerprint": design["design_fingerprint"], **truth_report(case),
              "oracle": oracle, "perturbations": perturbations,
              "training": {k: v for k, v in learned.items() if k not in ("X_hat", "B_hat")},
              "learned_training_performed": True, "X_true_used_in_training_or_stopping": False,
              "rho_zero_weighting": "W=I"}
    if learned["stop_reason"] == "nonfinite_loss":
        report["status"] = "TRAINING_FAILED_NONFINITE_LOSS"
        write_json(path, report)
        raise RuntimeError(f"TRAINING_FAILED_NONFINITE_LOSS: {dataset}")
    records, raw = v54.learned_records(dataset, case, learned, context, args.rho_workers)
    raw["molecular_level"].update(recall_accounting(
        raw["molecular_level"]["TP"], raw["molecular_level"]["FP"],
        raw["molecular_level"]["FN"], raw["molecular_level"]["TP"], raw["molecular_level"]["FP"]))
    preserve_candidate_records(directory, dataset, case, learned, records, context)
    for level in ("candidate_level", "molecular_level"):
        counts = raw[level]
        counts.update(wilson(counts["FP"], counts["TP"] + counts["FP"]))
        if counts["TP"] + counts["FP"] == 0:
            counts["precision"] = None
    fields = ["dataset_id", "split", "replicate", "complexity_condition", "global_truth_K",
              "candidate_index", "candidate_id", "lipid_name", "X_hat", "candidate_truth",
              "molecular_truth", "same_lipid_alternative_candidate", "rho_zero"]
    write_rows(directory / "reported_identity_records.csv", records, fields)
    report.update({"status": "COMPLETE", "learned_raw_identity_performance": raw})
    write_json(path, report)
    return report


def wilson(fp: int, n: int) -> dict:
    if n == 0:
        return {"FDR": None, "FDR_lower95": None, "FDR_upper95": None}
    z = 1.959963984540054
    p = fp / n
    denominator = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denominator
    radius = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return {"FDR": p, "FDR_lower95": max(0.0, center - radius), "FDR_upper95": min(1.0, center + radius)}


def recall_accounting(raw_tp: int, raw_fp: int, raw_fn: int, tp: int, fp: int) -> dict:
    loss = raw_tp - tp
    require(loss >= 0, "FILTER_CANNOT_ADD_TRUE_IDENTITIES")
    retention = tp / raw_tp if raw_tp else None
    return {"raw_solver_TP": raw_tp, "raw_solver_FP": raw_fp, "raw_solver_FN": raw_fn,
            "filtered_TP": tp, "filtered_FP": fp, "filtered_FN": raw_fn + loss,
            "true_positive_retention": retention, "TP_retention": retention,
            "filter_induced_true_loss": loss,
            "filter_induced_true_loss_fraction": loss / raw_tp if raw_tp else None}


def metrics(units: list[dict], reports: dict, score=None, threshold=None) -> dict:
    retained = units if score is None else ([] if threshold is None else [u for u in units if u[score] >= threshold])
    tp = sum(u["molecular_truth"] for u in retained)
    fp = len(retained) - tp
    truth = sum(r["all_truth_molecular_identity_count"] for r in reports.values())
    reportable = sum(r["reportable_truth_molecular_identity_count"] for r in reports.values())
    raw_tp = sum(u["molecular_truth"] for u in units)
    return {"N_retained": len(retained), "TP": tp, "FP": fp, "FN": truth - tp,
            **recall_accounting(raw_tp, len(units) - raw_tp, truth - raw_tp, tp, fp),
            "precision": tp / len(retained) if retained else None, **wilson(fp, len(retained)),
            "coverage": len(retained) / len(units) if units else None,
            "all_truth_recall": tp / truth if truth else None,
            "reportable_truth_recall": sum(u["reportable_truth"] for u in retained) / reportable if reportable else None}


def calibration_curve(units: list[dict], score: str, reports: dict) -> list[dict]:
    require(all(np.isfinite(u[score]) for u in units), f"NONFINITE_CAL_SCORE: {score}")
    # Keep tied molecular scores together. No HOLD information enters this function.
    groups = defaultdict(list)
    for unit in units:
        groups[unit[score]].append(unit)
    truth = sum(r["all_truth_molecular_identity_count"] for r in reports.values())
    reportable = sum(r["reportable_truth_molecular_identity_count"] for r in reports.values())
    raw_tp = sum(u["molecular_truth"] for u in units)
    rows, tp, fp, reportable_tp = [], 0, 0, 0
    for threshold in sorted(groups, reverse=True):
        group = groups[threshold]
        tp += sum(u["molecular_truth"] for u in group)
        fp += sum(not u["molecular_truth"] for u in group)
        reportable_tp += sum(u["reportable_truth"] for u in group)
        rows.append({"threshold": threshold, "N_retained": tp + fp, "TP": tp, "FP": fp,
                     "FN": truth - tp, **wilson(fp, tp + fp), "precision": tp / (tp + fp),
                     **recall_accounting(raw_tp, len(units) - raw_tp, truth - raw_tp, tp, fp),
                     "all_truth_recall": tp / truth if truth else None,
                     "reportable_truth_recall": reportable_tp / reportable if reportable else None,
                     "coverage": (tp + fp) / len(units) if units else None})
    return rows


def choose_threshold(curve: list[dict], target: float):
    eligible = [r for r in curve if r["FDR"] <= target]
    if not eligible:
        return None
    chosen = min(eligible, key=lambda r: (-r["N_retained"], r["threshold"]))
    # Preserve the frozen threshold payload and selection rule; new diagnostics live in curves.
    return {k: chosen[k] for k in ("threshold", "N_retained", "TP", "FP", "FDR", "precision")}


def load_completed(output: Path, condition: str, split: str, levels, fingerprint: str):
    reports, records, hashes = {}, [], {}
    for dataset in dataset_ids(levels, split):
        directory = dataset_directory(output, condition, dataset)
        report = v54.read_json(directory / "report.json")
        require(report["status"] == "COMPLETE" and report["design_fingerprint"] == fingerprint,
                f"INCOMPLETE_OR_STALE_DATASET: {condition}/{dataset}")
        require(report["dataset_id"] == dataset and report["condition"] == condition, "RESULT_ID_MISMATCH")
        reports[dataset] = report
        records.extend(read_rows(directory / "reported_identity_records.csv"))
        hashes[dataset] = {name: digest(directory / name) for name in ("report.json", "reported_identity_records.csv")}
    units = v54.molecular_units(records, reports)
    for unit in units:
        split_name, replicate, label = unit["dataset_id"].split("_")
        unit.update({"split": split_name, "replicate": replicate, "K": int(label[1:])})
        require(all(np.isfinite(unit[s]) for s in ("rho_zero", "X_hat")), "NONFINITE_MOLECULAR_SCORE")
    return units, reports, hashes


def subset(units, reports, k=None, replicate=None):
    ids = {d for d in reports if (k is None or int(d.split("_")[2][1:]) == k) and
           (replicate is None or d.split("_")[1] == replicate)}
    return [u for u in units if u["dataset_id"] in ids], {d: reports[d] for d in ids}


def threshold_pack(units, reports) -> tuple[dict, dict]:
    curves = {score: calibration_curve(units, score, reports) for score in ("rho_zero", "X_hat")}
    pack = {score: {f"FDR{int(target * 100)}": choose_threshold(curves[score], target)
                    for target in (0.05, 0.01)} for score in curves}
    return pack, curves


def validation(units, reports, pack, levels) -> tuple[dict, list, list]:
    overall, by_k, by_r = {}, [], []
    for score, targets in pack.items():
        for target, entry in targets.items():
            tau = entry["threshold"] if entry else None
            label = f"{score}_{target}"
            base = {"score": score, "target": target, "threshold": tau,
                    "threshold_available": entry is not None}
            overall[label] = {**base, **metrics(units, reports, score, tau)}
            for k in levels:
                us, rs = subset(units, reports, k=k)
                reps = {}
                for replicate in MAPPING_SEEDS:
                    ru, rr = subset(us, rs, replicate=replicate)
                    reps[replicate] = metrics(ru, rr, score, tau)
                values = [m["FDR"] for m in reps.values() if m["FDR"] is not None]
                stats = {"mean": float(np.mean(values)) if values else None,
                         "SD": float(np.std(values, ddof=1)) if len(values) > 1 else None,
                         "median": float(np.median(values)) if values else None,
                         "min": min(values) if values else None, "max": max(values) if values else None,
                         "n_defined": len(values)}
                by_k.append({**base, "K": k, **metrics(us, rs, score, tau),
                             "replicate_FDR_values": {r: m["FDR"] for r, m in reps.items()},
                             "replicate_metrics": reps, "replicate_FDR_summary": stats})
            for replicate in MAPPING_SEEDS:
                us, rs = subset(units, reports, replicate=replicate)
                by_r.append({**base, "replicate": replicate, **metrics(us, rs, score, tau)})
    return overall, by_k, by_r


def rho_distributions(units, split, levels):
    rows = []
    for k in (None, *levels):
        us = [u for u in units if k is None or u["K"] == k]
        values = {truth: np.array([u["rho_zero"] for u in us if u["molecular_truth"] == truth])
                  for truth in (True, False)}
        true, false = values[True], values[False]
        gaps = {"median_log10_gap": float(np.log10(np.median(true) + 1e-12) - np.log10(np.median(false) + 1e-12)) if len(true) and len(false) else None,
                "extreme_log10_gap": float(np.log10(min(true) + 1e-12) - np.log10(max(false) + 1e-12)) if len(true) and len(false) else None}
        for truth, x in values.items():
            rows.append({"split": split, "K": k if k else "GLOBAL", "truth": truth, "n": len(x),
                         "min": float(min(x)) if len(x) else None,
                         **{name: float(np.quantile(x, q)) if len(x) else None for name, q in
                            (("q10", .1), ("q25", .25), ("median", .5), ("q75", .75), ("q90", .9))},
                         "max": float(max(x)) if len(x) else None, **gaps,
                         "gap_interpretation": "descriptive_not_physical; eps=1e-12"})
    return rows


GEOMETRY_FIELDS = ("cone_isolation", "max_fragment_cosine", "max_full_cosine",
                   "collective_gain", "lipid_class")


def analysis_geometry(context: dict) -> dict:
    result = {}
    for source in context["geometry"]:
        row = {key: source[key] for key in GEOMETRY_FIELDS if key != "collective_gain"}
        full = float(np.clip(source["max_full_cosine"], -1, 1))
        row["collective_gain"] = math.sqrt(max(0.0, 1 - full * full)) - source["cone_isolation"]
        result[int(source["candidate_index"])] = row
    return result


def preserve_candidate_records(directory, dataset, case, learned, records, context):
    """Save all solver outputs; rho remains evaluated only for reported candidates."""
    geometry = analysis_geometry(context)
    reported = {r["candidate_index"]: r for r in records}
    xhat = learned["X_hat"][:, case["foreground_mask"]].mean(axis=1, dtype=np.float64)
    names = context["metadata"]["lipid_name"]
    truth_names = {str(names[i]) for i in case["active_indices"]}
    reportable_names = {str(names[i]) for i in case["reportable_truth_indices"]}
    split, replicate, label = dataset.split("_")
    rows = []
    for i, name in enumerate(names):
        rows.append({"dataset_id": dataset, "candidate_index": i, "lipid_name": str(name),
                     "X_hat": float(xhat[i]), "rho_zero": reported[i]["rho_zero"] if i in reported else None,
                     "rho_zero_status": "COMPUTED" if i in reported else "NOT_COMPUTED_NOT_REPORTED",
                     "raw_solver_reported": i in reported, "molecular_truth": str(name) in truth_names,
                     "reportable_truth": str(name) in reportable_names,
                     "K": int(label[1:]), "replicate": replicate, "split": split, **geometry[i]})
    write_rows(directory / "candidate_false_negative_records.csv", rows)


def export_false_negative_records(output, destination, condition, units, reports, pack, context):
    """Flags are the primary molecular decisions, repeated on constituent candidates.

    Keep both score rules in separate rows. Geometry of multi-candidate names is
    preserved as candidate-indexed values, never collapsed to a new group score.
    """
    geometry = analysis_geometry(context)
    by_unit = {(u["dataset_id"], u["lipid_name"]): u for u in units}
    candidate_rows, molecular_rows = [], []
    for dataset, report in reports.items():
        directory = dataset_directory(output, condition, dataset)
        preserved = directory / "candidate_false_negative_records.csv"
        if preserved.exists():
            candidates = read_rows(preserved)
            for row in candidates:
                if row["rho_zero"] == "":
                    row["rho_zero"] = None
        else:
            # Legacy cached results lack sub-gate abundance. Preserve missingness;
            # do not rerun inference or falsely replace missing abundances by zero.
            known = {r["candidate_index"]: r for r in read_rows(directory / "reported_identity_records.csv")}
            raw = report["learned_raw_identity_performance"]
            split, replicate, label = dataset.split("_")
            candidates = [{"dataset_id": dataset, "candidate_index": i, "lipid_name": str(name),
                           "rho_zero": known[i]["rho_zero"] if i in known else None,
                           "X_hat": known[i]["X_hat"] if i in known else None,
                           "rho_zero_status": "COMPUTED" if i in known else "NOT_COMPUTED_NOT_REPORTED",
                           "raw_solver_reported": i in known,
                           "molecular_truth": str(name) in raw["truth_lipid_names"],
                           "reportable_truth": str(name) in raw["reportable_truth_lipid_names"],
                           "K": int(label[1:]), "replicate": replicate, "split": split,
                           **geometry[i]} for i, name in enumerate(context["metadata"]["lipid_name"])]
        grouped = defaultdict(list)
        for row in candidates:
            grouped[row["lipid_name"]].append(row)
        for name, members in grouped.items():
            unit = by_unit.get((dataset, name))
            first = members[0]
            for score, targets in pack.items():
                flags = {}
                for target in ("FDR5", "FDR1"):
                    entry = targets[target]
                    flags[f"retained_by_global_{target}"] = bool(
                        unit is not None and entry is not None and unit[score] >= entry["threshold"])
                    flags[f"global_{target}_threshold_available"] = entry is not None
                    flags[f"solver_missed_truth_{target}"] = bool(first["molecular_truth"] and unit is None)
                    flags[f"filter_removed_truth_{target}"] = bool(
                        first["molecular_truth"] and unit is not None and not flags[f"retained_by_global_{target}"])
                common = {"condition": condition, "filter_score": score, **flags}
                for member in members:
                    candidate_rows.append({**member, **common,
                                           "retention_flag_unit": "dataset/lipid_name",
                                           "X_hat_available": member["X_hat"] is not None})
                molecular = {key: first[key] for key in
                             ("dataset_id", "lipid_name", "molecular_truth", "reportable_truth", "K", "replicate", "split")}
                molecular.update({**common, "raw_solver_reported": unit is not None,
                                  "candidate_indices": [m["candidate_index"] for m in members],
                                  "rho_zero": unit["rho_zero"] if unit else None,
                                  "rho_zero_status": "COMPUTED" if unit else "NOT_COMPUTED_NOT_REPORTED",
                                  "X_hat": unit["X_hat"] if unit else (
                                      sum(m["X_hat"] for m in members) if all(m["X_hat"] is not None for m in members) else None),
                                  "X_hat_definition": "sum reported candidates (primary score); for solver misses, sum all candidates if available"})
                for key in GEOMETRY_FIELDS:
                    molecular[key] = first[key] if len(members) == 1 else {
                        str(m["candidate_index"]): m[key] for m in members}
                molecular_rows.append(molecular)
    write_rows(destination / "candidate_false_negative_analysis.csv", candidate_rows)
    write_rows(destination / "molecular_false_negative_analysis.csv", molecular_rows)


def export_recall_curves(destination, condition, cal, cal_reports, hold, hold_reports, levels):
    rows = []
    for split, units, reports in (("CAL", cal, cal_reports), ("HOLD", hold, hold_reports)):
        for k in (None, *levels):
            us, rs = subset(units, reports, k=k)
            for score in ("rho_zero", "X_hat"):
                rows.extend({"condition": condition, "split": split, "K": k if k else "GLOBAL",
                             "score": score, "curve_role": "DESCRIPTIVE_NO_HOLD_THRESHOLD_SELECTION", **row}
                            for row in calibration_curve(us, score, rs))
    # Each CSV contains global/per-K curves for both scores and both splits.
    base = ["condition", "split", "K", "score", "curve_role", "threshold", "TP", "FP", "FDR",
            "FDR_lower95", "FDR_upper95", "precision"]
    write_rows(destination / "fdr_vs_recall_curves.csv", rows,
               base + ["all_truth_recall", "reportable_truth_recall", "TP_retention", "coverage",
                       "raw_solver_TP", "raw_solver_FP", "raw_solver_FN", "filtered_TP", "filtered_FP", "filtered_FN",
                       "true_positive_retention", "filter_induced_true_loss", "filter_induced_true_loss_fraction"])
    write_rows(destination / "fdr_vs_tp_retention_curves.csv", rows,
               base + ["TP_retention", "coverage", "all_truth_recall", "reportable_truth_recall",
                       "raw_solver_TP", "raw_solver_FP", "raw_solver_FN", "filtered_TP", "filtered_FP", "filtered_FN",
                       "true_positive_retention", "filter_induced_true_loss", "filter_induced_true_loss_fraction"])


def aggregate_condition(output: Path, design: dict, context: dict, condition="CLEAN") -> dict:
    levels = K_LEVELS if condition == "CLEAN" else ROBUST_K
    destination = output if condition == "CLEAN" else output / "robustness" / condition
    fingerprint = design["design_fingerprint"]
    cal, cal_reports, cal_hashes = load_completed(output, condition, "CAL", levels, fingerprint)
    pack, curves = threshold_pack(cal, cal_reports)
    secondary = {str(k): threshold_pack(*subset(cal, cal_reports, k=k))[0]["rho_zero"] for k in levels}
    frozen = {"status": "CAL_THRESHOLDS_FROZEN_BEFORE_HOLD", "condition": condition,
              "design_fingerprint": fingerprint, "CAL_source_hashes": cal_hashes,
              "global_thresholds": pack, "SECONDARY_K_SPECIFIC_CALIBRATION": secondary,
              "rule": "CAL only; maximum retained molecular units subject to empirical FDR <= target; ties choose lower threshold; score >= threshold",
              "molecular_score_rule": "per dataset/lipid_name: max rho_zero and summed X_hat among reported candidates",
              "GLOBAL_tau_CAL_FDR5": pack["rho_zero"]["FDR5"]["threshold"] if pack["rho_zero"]["FDR5"] else None,
              "GLOBAL_tau_CAL_FDR1": pack["rho_zero"]["FDR1"]["threshold"] if pack["rho_zero"]["FDR1"] else None}
    frozen_path = destination / "global_frozen_thresholds.json"
    if frozen_path.exists():
        require(v54.read_json(frozen_path) == frozen, "FROZEN_CAL_THRESHOLDS_OR_SOURCES_CHANGED; no HOLD-based recalibration allowed")
    else:
        write_json(frozen_path, frozen)
    for score, name in (("rho_zero", "rho"), ("X_hat", "abundance")):
        write_rows(destination / f"global_{name}_calibration_curve.csv", curves[score],
                   list(curves[score][0]) if curves[score] else ["threshold", "TP", "FP", "FDR", "precision",
                       "all_truth_recall", "reportable_truth_recall", "TP_retention", "coverage"])
    clean = None
    if condition != "CLEAN":
        clean = v54.read_json(output / "global_frozen_thresholds.json")
        require(clean["design_fingerprint"] == fingerprint and clean["condition"] == "CLEAN",
                "CLEAN_THRESHOLD_DESIGN_MISMATCH")
        clean_cal, clean_reports, clean_hashes = load_completed(output, "CLEAN", "CAL", K_LEVELS, fingerprint)
        require(clean["CAL_source_hashes"] == clean_hashes and
                clean["global_thresholds"] == threshold_pack(clean_cal, clean_reports)[0],
                "CLEAN_FROZEN_THRESHOLD_OR_CAL_SOURCE_CHANGED")
    # HOLD is intentionally loaded only after CAL thresholds are persisted.
    hold, hold_reports, _ = load_completed(output, condition, "HOLD", levels, fingerprint)
    export_recall_curves(destination, condition, cal, cal_reports, hold, hold_reports, levels)
    export_false_negative_records(output, destination, condition, cal + hold,
                                  {**cal_reports, **hold_reports}, pack, context)
    for dataset, dataset_report in {**cal_reports, **hold_reports}.items():
        us = [u for u in cal + hold if u["dataset_id"] == dataset]
        dataset_report["learned_raw_identity_performance"]["molecular_level"].update(
            metrics(us, {dataset: dataset_report}))
    overall, by_k, by_r = validation(hold, hold_reports, pack, levels)
    secondary_results = {}
    for k in levels:
        us, rs = subset(hold, hold_reports, k=k)
        secondary_results[str(k)] = validation(us, rs, {"rho_zero": secondary[str(k)]}, (k,))[0]
    transfer = None
    if condition != "CLEAN":
        transfer_overall, transfer_k, transfer_r = validation(
            hold, hold_reports, {"rho_zero": clean["global_thresholds"]["rho_zero"]}, levels)
        transfer = {"source": str(output / "global_frozen_thresholds.json"),
                    "recalibrated": False, "overall": transfer_overall, "by_K": transfer_k, "by_replicate": transfer_r}
    distributions = rho_distributions(cal, "CAL", levels) + rho_distributions(hold, "HOLD", levels)
    raw = {"CAL": metrics(cal, cal_reports), "HOLD": metrics(hold, hold_reports),
           "HOLD_by_K": {str(k): metrics(*subset(hold, hold_reports, k=k)) for k in levels},
           "HOLD_by_replicate": {r: metrics(*subset(hold, hold_reports, replicate=r)) for r in MAPPING_SEEDS}}
    # Every molecular unit contains at least one candidate strictly above GATE.
    # This retains the entire raw reported set and supplies the same CI/replicate summaries.
    raw_validation = validation(hold, hold_reports, {"X_hat": {"RAW_REPORTED": {"threshold": GATE}}}, levels)
    raw["HOLD_by_K_with_replicate_FDR_statistics"] = raw_validation[1]
    dataset_validation = {}
    for dataset, dataset_report in {**cal_reports, **hold_reports}.items():
        us = [u for u in cal + hold if u["dataset_id"] == dataset]
        dataset_validation[dataset] = {
            f"{score}_{target}": {"threshold_available": entry is not None,
                **metrics(us, {dataset: dataset_report}, score, entry["threshold"] if entry else None)}
            for score, targets in pack.items() for target, entry in targets.items()}
    statistical = {"method": "95% Wilson binomial false-proportion interval; z=1.959963984540054",
                   "zero_retained_policy": "FDR/precision/CI undefined (null), not zero",
                   "replicate_SD": "sample SD (ddof=1), separate from Wilson CI",
                   "dependence_limitation": LIMITATIONS[5], "heldout_by_K": by_k}
    report = {"status": "PENDING_REVIEW", "condition": condition, "design_fingerprint": fingerprint,
              "global_frozen_thresholds": frozen, "heldout_global_validation": overall,
              "heldout_by_K": by_k, "heldout_by_replicate": by_r,
              "SECONDARY_K_SPECIFIC_CALIBRATION": secondary_results,
              "clean_threshold_transfer_without_recalibration": transfer, "raw_molecular_metrics": raw,
              "dataset_raw_metrics": {d: r["learned_raw_identity_performance"] for d, r in {**cal_reports, **hold_reports}.items()},
              "dataset_global_filtered_metrics": dataset_validation,
              "statistical_summary": statistical, "limitations": LIMITATIONS,
              "recall_accounting": {
                  "definition": "filtered_FN = raw_solver_FN + filter_induced_true_loss; all mapped identities remain truth",
                  "zero_raw_TP_policy": "TP retention and filter-induced loss fraction are null",
                  "curve_files": ["fdr_vs_recall_curves.csv", "fdr_vs_tp_retention_curves.csv"],
                  "record_files": ["candidate_false_negative_analysis.csv", "molecular_false_negative_analysis.csv"],
                  "record_flags": "global molecular decision per filter_score; unavailable thresholds retain none",
                  "missing_scores": "Unreported candidates have no computed rho; legacy sub-gate X_hat may be null"},
              "final_deployment_threshold_frozen": False}
    write_json(destination / "heldout_global_validation.json", overall)
    write_rows(destination / "heldout_by_K.csv", by_k)
    write_rows(destination / "heldout_by_replicate.csv", by_r)
    write_rows(destination / "rho_distribution_summary.csv", distributions)
    write_json(destination / "statistical_summary.json", statistical)
    write_json(destination / "report.json", report)
    summary = [f"# V56 molecular-identity FDR benchmark: {condition}", "",
               "Status: PENDING_REVIEW. Final deployment threshold frozen: NO.", "",
               "Primary: one global CAL threshold per target, applied unchanged across HOLD K and mapping replicates.",
               "Secondary K-specific calibration is descriptive only.", "",
               "Solver misses (raw_solver_FN) and filtering losses (filter_induced_true_loss) are separate; their sum is filtered_FN. TP retention is relative to raw solver TP, whereas recall uses mapped truth denominators.", "",
               "## Global HOLD validation", "", "```json", json.dumps(overall, indent=2), "```", "",
               "## Interpretation boundaries", "", *[f"- {line}" for line in LIMITATIONS]]
    stage0.atomic_write_text(destination / "summary.md", "\n".join(summary) + "\n")
    update_log(output, design)
    return report


def update_log(output: Path, design: dict) -> None:
    path = ROOT / "results/EXPERIMENT_LOG.md"
    begin, end = f"<!-- BEGIN {VERSION} -->", f"<!-- END {VERSION} -->"
    current = path.read_text(encoding="utf-8") if path.exists() else "# Experiment Log\n"
    require(current.count(begin) <= 1 and current.count(end) <= 1, "DUPLICATE_LOG_MARKERS")
    payload = {"design_status": design["status"], "K_LEVELS": list(K_LEVELS),
               "identity_split_seed": IDENTITY_SPLIT_SEED, "mapping_seeds": MAPPING_SEEDS,
               "block_design": design["block_rule"], "template_design": design["template_rule"],
               "signal_normalization": {"target": TARGET_SIGNAL, "interpretation": LIMITATIONS[1]},
               "planned_runs": {"clean": 60, "additional_mismatch": 60}, "robustness": MISMATCH,
               "limitations": LIMITATIONS, "validity_status": "PENDING_REVIEW",
               "final_deployment_threshold_frozen": False}
    oracle_path = output / "oracle_summary.csv"
    payload["oracle_results"] = read_rows(oracle_path) if oracle_path.exists() else []
    payload["analyses"] = {}
    for condition in ("CLEAN", *MISMATCH):
        report_path = (output if condition == "CLEAN" else output / "robustness" / condition) / "report.json"
        if report_path.exists():
            report = v54.read_json(report_path)
            payload["analyses"][condition] = {key: report[key] for key in
                ("global_frozen_thresholds", "heldout_global_validation", "statistical_summary",
                 "clean_threshold_transfer_without_recalibration")}
    entry = f"{begin}\n## V56 paper identity FDR benchmark\n\n```json\n{json.dumps(stage0.to_jsonable(payload), indent=2)}\n```\n{end}"
    if begin in current and end in current:
        a, b = current.index(begin), current.index(end)
        require(a < b, "INVALID_LOG_MARKERS")
        current = current[:a] + entry + current[b + len(end):]
    else:
        require(begin not in current and end not in current, "INCOMPLETE_LOG_MARKERS")
        current = current.rstrip() + "\n\n" + entry + "\n"
    stage0.atomic_write_text(path, current)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-root", type=Path, default=ROOT.parent / "decon-lipid")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DEFAULT)
    action = parser.add_mutually_exclusive_group(required=True)
    for flag in ("prepare-design", "audit-design", "oracle-only", "freeze-design", "clean-all",
                 "aggregate-clean", "aggregate-robustness", "dry-run"):
        action.add_argument(f"--{flag}", action="store_true")
    action.add_argument("--dataset", choices=dataset_ids())
    action.add_argument("--robustness-condition", choices=tuple(MISMATCH))
    parser.add_argument("--condition", choices=("CLEAN", *MISMATCH), default="CLEAN",
                        help="Condition for --dataset or --oracle-only; defaults to CLEAN")
    parser.add_argument("--parallel-gpus", help="Comma-separated GPU IDs; omitted means single GPU serial")
    parser.add_argument("--rho-workers", type=int, default=1)
    args = parser.parse_args()
    if args.rho_workers < 1:
        parser.error("--rho-workers must be >= 1")
    if args.parallel_gpus and not (args.clean_all or args.robustness_condition):
        parser.error("--parallel-gpus requires --clean-all or --robustness-condition")
    if args.condition != "CLEAN" and not (args.dataset or args.oracle_only):
        parser.error("--condition applies only to --dataset or --oracle-only")
    if args.dataset and args.condition != "CLEAN" and args.dataset not in dataset_ids(ROBUST_K):
        parser.error("Mismatch datasets require K=50,100,150")
    args.output_dir = args.output_dir.resolve()
    return args


def main() -> None:
    args = parse_args()
    context = load_context(args.asset_root)
    if context.get("missing_dependencies"):
        payload = {"status": "LOCKED_DEPENDENCIES_REQUIRED", **context,
                   "planned_clean_runs": 60, "planned_additional_mismatch_runs": 60,
                   "GPU_training_performed": False}
        if args.dry_run:
            print(json.dumps(payload, indent=2))
            return
        raise RuntimeError(json.dumps(payload))
    if args.dry_run:
        if (args.output_dir / "design.json").exists():
            design, tables = load_design(args.output_dir, context)
            details = {"design_status": design["status"]}
        else:
            tables, details = make_tables(context)
        print(json.dumps(stage0.to_jsonable({"status": "DRY_RUN_READY", **details,
                         "audit": audit_tables(tables, context), "planned_clean_runs": 60,
                         "planned_additional_mismatch_runs": 60, "GPU_training_performed": False}), indent=2))
        return
    if args.prepare_design:
        print(json.dumps(stage0.to_jsonable(prepare(args.output_dir, context)), indent=2))
        return
    training = bool(args.dataset or args.clean_all or args.robustness_condition)
    design, tables = load_design(args.output_dir, context, frozen=training or args.aggregate_clean or args.aggregate_robustness)
    if args.audit_design:
        print(json.dumps(stage0.to_jsonable(audit_tables(tables, context)), indent=2))
        return
    if args.freeze_design:
        design["status"] = "DESIGN_FROZEN_BEFORE_TRAINING"
        design.setdefault("review_confirmed_utc", datetime.now(timezone.utc).isoformat())
        write_json(args.output_dir / "design.json", design)
        update_log(args.output_dir, design)
        print("DESIGN_FROZEN_BEFORE_TRAINING; explicit --freeze-design records review; no GPU training")
        return
    if args.aggregate_clean:
        aggregate_condition(args.output_dir, design, context)
        print("CLEAN_ANALYSIS_WRITTEN_PENDING_REVIEW")
        return
    if args.aggregate_robustness:
        for condition in MISMATCH:
            aggregate_condition(args.output_dir, design, context, condition)
        print("ROBUSTNESS_ANALYSIS_WRITTEN_PENDING_REVIEW")
        return
    condition = args.robustness_condition or args.condition
    datasets = [args.dataset] if args.dataset else dataset_ids(K_LEVELS if condition == "CLEAN" else ROBUST_K)
    if args.oracle_only:
        if condition == "CLEAN":
            preflight_oracles(args.output_dir, context, design, tables, datasets)
        else:
            for dataset in datasets:
                oracle_one(args.output_dir, context, design, tables, dataset, condition)
            refresh_oracle_summary(args.output_dir)
        update_log(args.output_dir, design)
        print("ORACLE_ONLY_COMPLETE; GPU training performed: NO")
        return
    # All requested clean oracles run before any GPU worker; failures stop the batch.
    preflight_oracles(args.output_dir, context, design, tables, datasets, summary=not args.dataset)
    if args.parallel_gpus:
        gpus = [g.strip() for g in args.parallel_gpus.split(",") if g.strip()]
        require(bool(gpus) and len(set(gpus)) == len(gpus), "INVALID_GPU_LIST")

        def worker(gpu, queue):
            for dataset in queue:
                command = [sys.executable, str(Path(__file__).resolve()), "--asset-root", str(args.asset_root.resolve()),
                           "--output-dir", str(args.output_dir), "--dataset", dataset, "--condition", condition,
                           "--rho-workers", str(args.rho_workers)]
                completed = subprocess.run(command, env={**os.environ, "CUDA_VISIBLE_DEVICES": gpu}, check=False)
                require(completed.returncode == 0, f"DATASET_WORKER_FAILED: {condition}/{dataset}")
        with ThreadPoolExecutor(max_workers=len(gpus)) as executor:
            futures = [executor.submit(worker, gpu, datasets[i::len(gpus)]) for i, gpu in enumerate(gpus)]
            for future in futures:
                future.result()
    else:
        for dataset in datasets:
            run_one(args, context, design, tables, dataset, condition)
    if not args.dataset:
        refresh_oracle_summary(args.output_dir)
        update_log(args.output_dir, design)
    print("REQUESTED_DATASETS_COMPLETE; run explicit aggregation after CAL/HOLD completion")


if __name__ == "__main__":
    main()
