#!/usr/bin/env python3
"""V57 CLEAN spectral x spatial identity confidence benchmark.

No production source is modified. Lifecycle: prepare -> audit -> freeze ->
60 exact oracles -> two sentinel runs -> CLEAN -> CAL freeze -> HOLD analysis.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "analysis"))
import run_v56_paper_identity_fdr_benchmark as v56
import run_v55_spectral_noise_robustness as v55

v54 = v56.v54
require = v56.require
VERSION = "v57_spectral_spatial_identity_confidence_benchmark"
DEFAULT_OUTPUT = ROOT / "results" / VERSION
K_LEVELS = (50, 75, 100, 125, 150, 175)
MAPPING_SEEDS = {f"R{i}": 5700 + i for i in range(1, 6)}
SENTINELS = ("CAL_R1_K050", "CAL_R1_K175")
ROLES = tuple(f"spectral_{s}_spatial_{p}" for s in ("hard", "easy") for p in ("high", "low"))
PAIR_METRICS = ("fragment_cosine", "fragment_support_jaccard", "full_cosine", "parent_cosine")
TAILS = ("p90_over_p50", "p95_over_p50", "p99_over_p50", "max_over_p50")
CONTRACT = {
    "design_seed": 5700, "mapping_seeds": MAPPING_SEEDS, "K_levels": K_LEVELS,
    "healthy_count": 97, "occupancy_min": 0.2782092568036876,
    "positive_q99_over_mean_max": 16.194432316884477, "max_over_mean_max": 35.33171664958769,
    "pair_weights": dict(zip(PAIR_METRICS, (.40, .15, .15, .10))), "cone_weight": .20,
    "pair_pool_fraction": .35, "pair_search_restarts": 20000,
    "assignment_search_restarts": 20000, "max_base_reuse": 2,
    "smooth_sigma_pixels": 2.0, "morphology_log_sigma": .05,
    "spatial_high_min": .95, "spatial_low_quantile": .25, "spatial_low_slack": .05,
    "abundance_clip": (.5, 2.0), "abundance_quantiles": tuple((i + .5) / 25 for i in range(25)),
    "signal_target": .6036783456802368, "report_gate": .001,
    "domain_envelope_multiplier": 2.0, "sentinel_max_residual": .10,
    "hard_epoch_cap": 3000, "checkpoint_epochs": (1000, 1500, 2000, 2500, 3000),
    "sentinel_requires_epoch_3000": False, "production_early_stop_unchanged": True,
    "measurement_noise": False, "mismatch_runs": 0, "clean_runs": 60,
    "pair_rank_population": "all C(175,2) unit pairs separately in CAL and HOLD; cone ranks over 175 candidates per split",
    "pair_pool_ties": "exact 105 of 300 per block, score then unit index",
    "spatial_field": "normal field over full image -> gaussian_filter(sigma=2, mode=reflect) -> foreground standardization -> background zero",
    "nontruth_role": "UNASSIGNED_FALSE_POSITIVE; no invented pair attribution or category FDR",
}
LIMITATIONS = [
    "K is controlled synthetic mixture complexity, not unknown biological tissue complexity.",
    "Fixed total spectral signal: components share a fixed budget as K rises.",
    "Healthy real maps preserve morphology; bounded abundance multipliers do not preserve extreme real abundance tails.",
    "CAL/HOLD separates molecular identities, not network training data. Mapping replicates are not biological replicates.",
    "Empirical FDR is not a population guarantee. Wilson intervals ignore dependence from nested identities and shared templates.",
    "Spatial 5% multiplicative perturbations are morphology changes, not measurement noise; CLEAN contains no additive noise.",
    "Only truth units have designed spectral/spatial roles. Category recovery is descriptive; FP and false rho have no valid role attribution and are reported separately.",
    "Production early stopping is retained; 3000 epochs is a hard cap only. A normally completed finite sentinel passes when its final foreground reconstruction relative residual is <=0.10, including early-stopped runs.",
]


def canonical(value):
    if isinstance(value, dict):
        require(len({str(k) for k in value}) == len(value), "JSON_KEY_COLLISION")
        return {str(k): canonical(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [canonical(v) for v in value]
    if isinstance(value, np.ndarray):
        return canonical(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    return value


def fingerprint(value):
    return hashlib.sha256(json.dumps(canonical(value), sort_keys=True, separators=(",", ":"),
                                    ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def write_json(path, value):
    v56.write_json(path, canonical(value))


def rows(path):
    return v56.read_rows(path)


def write_rows(path, values, fields=None):
    v56.write_rows(path, canonical(values), fields)


def ids(split=None):
    return [f"{s}_{r}_K{k:03d}" for s in ((split,) if split else ("CAL", "HOLD"))
            for r in MAPPING_SEEDS for k in K_LEVELS]


def seed(*parts):
    return int.from_bytes(hashlib.sha256("|".join(map(str, (VERSION, *parts))).encode()).digest()[:8], "little")


def load_inputs(args):
    required = [args.v56_output / name for name in ("design.json", "matched_identity_pairs.csv", "identity_blocks.csv")]
    required += [v54.V51_DIR / "pair_geometry.csv", args.v54_output / "design.json", args.v54_output / "template_mappings.csv"]
    required += [args.v54_output / d / "report.json" for d in v54.DATASETS]
    missing = [str(p) for p in required if not p.exists()]
    require(not missing, "MISSING_DEPENDENCIES: " + json.dumps(missing))
    context = v56.load_context(args.asset_root)
    require(not context.get("missing_dependencies"), "MISSING_DEPENDENCIES: " + json.dumps(context.get("missing_dependencies")))
    source = v54.read_json(args.v56_output / "design.json")
    require(source["input_hashes"] == context["validation"]["hashes_sha256"], "V56_ASSET_PROVENANCE_CHANGED")
    units = rows(args.v56_output / "matched_identity_pairs.csv")
    for name in ("matched_identity_pairs", "identity_blocks"):
        require(source["table_hashes"][name] == v56.digest(args.v56_output / f"{name}.csv"), "V56_IDENTITY_TABLE_CHANGED")
    require(rows(args.v56_output / "identity_blocks.csv") == [
        {k: u[k] for k in ("pair_id", "block", "quintile", "pair_ambiguity_score", "class_pair")} for u in units],
        "V56_BLOCK_MEMBERSHIP_MISMATCH")
    audit_units(units, context)
    require(len(context["singleton_indices"]) == 368, "SINGLETON_POOL_CHANGED")
    context["units"] = units
    context["pair_geometry"] = rows(v54.V51_DIR / "pair_geometry.csv")
    context["source_hashes"] = {str(p.resolve()): v56.digest(p) for p in required}
    context["source_v56_fingerprint"] = source["design_fingerprint"]
    return context


def audit_units(units, context):
    require(len(units) == 175 and len({u["pair_id"] for u in units}) == 175, "MATCHED_UNIT_COUNT")
    require(Counter(u["block"] for u in units) == Counter({b: 25 for b in range(1, 8)}), "BLOCK_COUNTS")
    names = {s: {u[f"{s}_lipid_name"] for u in units} for s in ("CAL", "HOLD")}
    require(len(names["CAL"]) == len(names["HOLD"]) == 175 and not names["CAL"] & names["HOLD"], "CAL_HOLD_LEAKAGE")
    singles = set(context["singleton_indices"].tolist())
    for u in units:
        for s in names:
            i = u[f"{s}_index"]
            require(i in singles and str(context["metadata"]["lipid_name"][i]) == u[f"{s}_lipid_name"], "NON_SINGLETON_IDENTITY")


def spectral_scores(context):
    units = sorted(context["units"], key=lambda u: u["pair_id"])
    geometries = {(int(r["i"]), int(r["j"])): r for r in context["pair_geometry"]}
    cone = {r["candidate_index"]: r["cone_isolation"] for r in context["geometry"]}
    combinations = list(itertools.combinations(units, 2))
    result = [{"unit_i": a["pair_id"], "unit_j": b["pair_id"],
               "block": a["block"] if a["block"] == b["block"] else None} for a, b in combinations]
    for split in ("CAL", "HOLD"):
        candidates = [u[f"{split}_index"] for u in units]
        cone_ranks = dict(zip(candidates, v54.rank_percentiles(np.array([cone[i] for i in candidates]))))
        values = []
        for a, b in combinations:
            i, j = sorted((a[f"{split}_index"], b[f"{split}_index"]))
            require((i, j) in geometries, f"V51_PAIR_MISSING: {i},{j}")
            values.append(geometries[i, j])
        ranks = {m: v54.rank_percentiles(np.array([r[m] for r in values], dtype=float)) for m in PAIR_METRICS}
        for pos, ((a, b), row) in enumerate(zip(combinations, result)):
            i, j = a[f"{split}_index"], b[f"{split}_index"]
            score = sum(CONTRACT["pair_weights"][m] * ranks[m][pos] for m in PAIR_METRICS)
            score += .20 * ((1 - cone_ranks[i]) + (1 - cone_ranks[j])) / 2
            row[f"spectral_score_{split}"] = float(score)
            row[f"geometry_{split}"] = {m: float(values[pos][m]) for m in PAIR_METRICS}
            row[f"geometry_{split}"].update({"cone_isolation_i": cone[i], "cone_isolation_j": cone[j]})
            row[f"ranks_{split}"] = {m: float(ranks[m][pos]) for m in PAIR_METRICS}
    for row in result:
        row["joint_hard_score"] = min(row["spectral_score_CAL"], row["spectral_score_HOLD"])
        row["joint_easy_score"] = max(row["spectral_score_CAL"], row["spectral_score_HOLD"])
    return result


def select_topology(scores):
    selected = []
    for block in range(1, 8):
        candidates = [r for r in scores if r["block"] == block]
        require(len(candidates) == 300, "BLOCK_PAIR_COUNT")
        hard = sorted(candidates, key=lambda r: (-r["joint_hard_score"], r["unit_i"], r["unit_j"]))[:105]
        easy = sorted(candidates, key=lambda r: (r["joint_easy_score"], r["unit_i"], r["unit_j"]))[:105]
        for r in candidates:
            r["hard_candidate_pool"] = r in hard
            r["easy_candidate_pool"] = r in easy
        rng = np.random.default_rng(5700 + block)
        best, best_key = None, None
        for _ in range(20000):
            used, proposal = set(), {"hard": [], "easy": []}
            # Randomize eight category slots so hard selection cannot always preempt easy.
            for kind in rng.permutation(["hard"] * 4 + ["easy"] * 4):
                options = [r for r in (hard if kind == "hard" else easy)
                           if r["unit_i"] not in used and r["unit_j"] not in used]
                if not options:
                    break
                row = options[int(rng.integers(len(options)))]
                proposal[kind].append(row)
                used.update((row["unit_i"], row["unit_j"]))
            if len(used) != 16:
                continue
            objective = np.mean([r["joint_hard_score"] for r in proposal["hard"]]) - np.mean([r["joint_easy_score"] for r in proposal["easy"]])
            lexical = tuple((kind, r["unit_i"], r["unit_j"]) for kind in ("hard", "easy")
                            for r in sorted(proposal[kind], key=lambda r: (r["unit_i"], r["unit_j"])))
            key = (-float(objective), lexical)
            if best_key is None or key < best_key:
                best, best_key = proposal, key
        require(best is not None, f"DESIGN_FAIL: block {block}, no disjoint 4 hard + 4 easy solution after 20000 restarts")
        for kind in ("hard", "easy"):
            four = sorted(best[kind], key=lambda r: (r["unit_i"], r["unit_j"]))
            score_key = "joint_hard_score" if kind == "hard" else "joint_easy_score"
            def balance_key(indices):
                high = np.mean([four[i][score_key] for i in indices])
                low = np.mean([four[i][score_key] for i in range(4) if i not in indices])
                return (abs(high - low), tuple((four[i]["unit_i"], four[i]["unit_j"]) for i in indices))
            high = min(itertools.combinations(range(4), 2), key=balance_key)
            for pos, row in enumerate(four):
                selected.append({**row, "pair_id": f"B{block}_U{row['unit_i']}_U{row['unit_j']}",
                                 "spectral_label": kind, "spatial_relation": "high" if pos in high else "low",
                                 "role": f"spectral_{kind}_spatial_{'high' if pos in high else 'low'}",
                                 "selection_objective": -best_key[0], "high_low_score_gap": balance_key(high)[0]})
    return selected


def healthy_pool(context):
    all_rows, bases = [], {}
    mask = context["mask"]
    for i, original in enumerate(context["X_real"]):
        x = original[mask].astype(np.float64)
        mean = float(x.mean())
        occupancy = float(np.mean(x > 0))
        positive = x[x > 0]
        q99 = float(np.quantile(positive, .99) / mean) if mean > 0 else None
        maximum = float(x.max() / mean) if mean > 0 else None
        healthy = bool(np.isfinite(x).all() and mean > 0 and occupancy >= CONTRACT["occupancy_min"] and
                       q99 <= CONTRACT["positive_q99_over_mean_max"] and maximum <= CONTRACT["max_over_mean_max"])
        all_rows.append({"candidate_index": i, "candidate_id": str(context["metadata"]["candidate_id"][i]),
                         "lipid_name": str(context["metadata"]["lipid_name"][i]), "raw_mean_fg": mean,
                         "occupancy": occupancy, "positive_q99_over_mean": q99,
                         "max_over_mean": maximum, "selected_healthy": healthy})
        if healthy:
            base = np.zeros(mask.shape, dtype=np.float32)
            base[mask] = (x / mean).astype(np.float32)
            bases[i] = base
    return all_rows, bases


def correlation(a, b, mask):
    x, y = a[mask].astype(np.float64), b[mask].astype(np.float64)
    x -= x.mean()
    y -= y.mean()
    denominator = np.linalg.norm(x) * np.linalg.norm(y)
    require(denominator > 0, "UNDEFINED_SPATIAL_CORRELATION")
    return float(np.clip(np.dot(x, y) / denominator, -1, 1))


def base_correlations(bases, mask):
    result = [{"i": i, "j": j, "correlation": correlation(bases[i], bases[j], mask)}
              for i, j in itertools.combinations(sorted(bases), 2)]
    q25 = float(np.quantile([r["correlation"] for r in result], .25))
    return result, q25


def perturb(base, mask, rng_seed):
    field = gaussian_filter(np.random.default_rng(rng_seed).normal(size=mask.shape), sigma=2.0, mode="reflect")
    std = float(field[mask].std())
    require(std > 0, "DEGENERATE_SPATIAL_FIELD")
    field = (field - field[mask].mean()) / std
    field[~mask] = 0
    result = base.astype(np.float64) * np.exp(.05 * field)
    result[~mask] = 0
    result /= result[mask].mean()
    result = result.astype(np.float32)
    require(np.isfinite(result).all() and np.array_equal(result > 0, base > 0), "SPATIAL_MORPHOLOGY_INVALID")
    return result


def abundance_design(context):
    means = context["foreground_means"]
    reference = means[means > .001]
    require(len(reference) > 0, "NO_REPORTABLE_ABUNDANCE_REFERENCE")
    reference = np.clip(reference / np.median(reference), .5, 2)
    quantiles = np.clip(np.quantile(reference, CONTRACT["abundance_quantiles"]), .5, 2)
    result = []
    for block in range(1, 8):
        units = sorted([u for u in context["units"] if u["block"] == block], key=lambda u: u["pair_id"])
        order = np.random.default_rng(5800 + block).permutation(25)
        result += [{"unit_id": u["pair_id"], "block": block, "abundance_multiplier": float(quantiles[int(p)]),
                    "quantile_index": int(p), "seed": 5800 + block} for u, p in zip(units, order)]
    return result, {"reportable_real_map_count": len(reference), "reference_abundance": reference.tolist(),
                    "block_quantile_multiset": quantiles.tolist()}


def spatial_assignments(context, topology, bases, correlations, q25):
    low_options = [(r["i"], r["j"]) for r in correlations if r["correlation"] <= q25]
    all_indices = sorted(bases)
    paired = {i for p in topology for i in (p["unit_i"], p["unit_j"])}
    singletons = sorted([u for u in context["units"] if u["pair_id"] not in paired], key=lambda u: u["pair_id"])
    low_pairs = [p for p in topology if p["spatial_relation"] == "low"]
    high_pairs = [p for p in topology if p["spatial_relation"] == "high"]
    output = []
    for replicate, rseed in MAPPING_SEEDS.items():
        rng = np.random.default_rng(rseed)
        chosen = None
        for _ in range(CONTRACT["assignment_search_restarts"]):
            counts, proposal = Counter(), []
            for pos in rng.permutation(len(low_pairs)):
                pair = low_pairs[int(pos)]
                options = [(i, j) for i, j in low_options if counts[i] < 2 and counts[j] < 2]
                if not options:
                    break
                i, j = options[int(rng.integers(len(options)))]
                if rng.integers(2):
                    i, j = j, i
                proposal.extend([(pair["unit_i"], i, pair["pair_id"] + "_i", pair),
                                 (pair["unit_j"], j, pair["pair_id"] + "_j", pair)])
                counts.update((i, j))
            else:
                other = [(p["unit_i"], p) for p in high_pairs] + [(u["pair_id"], None) for u in singletons]
                for pos in rng.permutation(len(other)):
                    unit, pair = other[int(pos)]
                    options = [i for i in all_indices if counts[i] < 2]
                    if not options:
                        break
                    i = options[int(rng.integers(len(options)))]
                    assignment = pair["pair_id"] if pair else f"singleton_U{unit}"
                    proposal.append((unit, i, assignment, pair))
                    if pair:
                        proposal.append((pair["unit_j"], i, assignment, pair))
                    counts[i] += 1
                else:
                    chosen = proposal
                    break
        require(chosen is not None, f"DESIGN_FAIL: template assignment max reuse=2 infeasible for {replicate}")
        # Correlation failures are hard failures, not grounds for reseeding or reselection.
        realized = {}
        for unit, base, assignment, pair in chosen:
            perturb_seed = seed(rseed, unit, "spatial")
            realized[unit] = perturb(bases[base], context["mask"], perturb_seed)
            output.append({"replicate": replicate, "unit_id": unit, "base_template_id": base,
                           "assignment_id": assignment, "perturbation_seed": perturb_seed,
                           "pair_id": pair["pair_id"] if pair else None,
                           "role": pair["role"] if pair else "singleton_control",
                           "block": next(u["block"] for u in context["units"] if u["pair_id"] == unit),
                           "map_sha256": hashlib.sha256(realized[unit].tobytes()).hexdigest()})
        for p in topology:
            corr = correlation(realized[p["unit_i"]], realized[p["unit_j"]], context["mask"])
            require(corr >= .95 if p["spatial_relation"] == "high" else corr <= q25 + .05,
                    f"DESIGN_FAIL: realized spatial correlation {replicate}/{p['pair_id']}={corr}")
            for row in output:
                if row["replicate"] == replicate and row["pair_id"] == p["pair_id"]:
                    row["realized_spatial_correlation"] = corr
        for row in output:
            row.setdefault("realized_spatial_correlation", None)
    return sorted(output, key=lambda r: (r["replicate"], r["unit_id"]))


def domain_stats(B, mask):
    norms = np.linalg.norm(B[:, mask].astype(np.float64), axis=0)
    p50 = float(np.median(norms))
    require(p50 > 0 and np.isfinite(norms).all(), "INVALID_B_DOMAIN")
    return {"p50": p50, **{f"p{q}_over_p50": float(np.quantile(norms, q / 100) / p50) for q in (90, 95, 99)},
            "max_over_p50": float(norms.max() / p50)}


def domain_reference(context, args):
    real = v55.load_observed_production_B(context["paths"], (1084, *context["mask"].shape))
    stats = {"REAL": domain_stats(real, context["mask"])}
    v54.validate_frozen_design(args.v54_output, context)
    for dataset in v54.DATASETS:
        report = v54.read_json(args.v54_output / dataset / "report.json")
        require(report["status"] == "COMPLETE", f"V54_REFERENCE_NOT_SUCCESSFUL: {dataset}")
        mapping = v54.dataset_mapping(args.v54_output, dataset)
        case = v54.construct_dataset(context, mapping, dataset)
        stats[dataset] = domain_stats(case["B_sim"], context["mask"])
    bounds = {m: 2 * max(s[m] for s in stats.values()) for m in TAILS}
    rejected = bounds["p99_over_p50"] < 23 or bounds["max_over_p50"] < 749
    require(rejected, "DOMAIN_REFERENCE_FAIL: envelope would not reject failed V56")
    return {"references": stats, "upper_bounds": bounds, "failed_V56_rejected": rejected,
            "failed_V56_observations_source": "user-provided preflight observations, not rerun"}


def construct_case(context, scientific, dataset):
    split, replicate, label = dataset.split("_")
    k = int(label[1:])
    require(dataset in ids(), "INVALID_DATASET_ID")
    if "bases" not in context:
        pool, bases = healthy_pool(context)
        require([r["candidate_index"] for r in pool if r["selected_healthy"]] == scientific["healthy_indices"], "HEALTHY_POOL_CHANGED")
        context["bases"] = bases
    assignments = {r["unit_id"]: r for r in scientific["spatial_assignments"] if r["replicate"] == replicate}
    abundance = {r["unit_id"]: r["abundance_multiplier"] for r in scientific["abundance_multipliers"]}
    bank = np.zeros((391, *context["mask"].shape), dtype=np.float32)
    mapping = []
    for position, unit in enumerate(scientific["matched_units"]):
        if unit["block"] > k // 25:
            continue
        a = assignments[unit["pair_id"]]
        spatial = perturb(context["bases"][a["base_template_id"]], context["mask"], a["perturbation_seed"])
        require(hashlib.sha256(spatial.tobytes()).hexdigest() == a["map_sha256"], "SPATIAL_REALIZATION_CHANGED")
        bank[position] = spatial * np.float32(abundance[unit["pair_id"]])
        mapping.append({"synthetic_truth_candidate_index": unit[f"{split}_index"],
                        "template_original_candidate_index": position})
    require(len(mapping) == k, "K_NESTING_FAILED")
    case = v54.construct_dataset({**context, "X_real": bank}, mapping, dataset)
    case["case_name"] = f"V57__{dataset}"
    return case


def source_implementations():
    files = [Path(__file__).resolve(), Path(v56.__file__), Path(v54.__file__), Path(v55.__file__),
             ROOT / "analysis/run_v50_reoptimized_sanity.py", ROOT / "src/rho_zero.py",
             ROOT / "src/config_758.py", ROOT / "src/lipid_ista.py", ROOT / "src/run_758_ista.py"]
    return {p.name: v56.digest(p) for p in files}


def prepare(args, context):
    path = args.output_dir / "design.json"
    require(not path.exists(), "DESIGN_EXISTS: never redraw V57; use --audit-design")
    pool, bases = healthy_pool(context)
    write_rows(args.output_dir / "healthy_spatial_template_pool.csv", pool)
    require(len(bases) == 97, f"DESIGN_FAIL: healthy template count={len(bases)}, expected=97; diagnostic CSV saved")
    context["bases"] = bases
    scores = spectral_scores(context)
    write_rows(args.output_dir / "candidate_pair_scores.csv", scores)
    topology = select_topology(scores)
    write_rows(args.output_dir / "candidate_pair_scores.csv", scores)
    write_rows(args.output_dir / "spectral_pair_design.csv", topology)
    correlations, q25 = base_correlations(bases, context["mask"])
    assignments = spatial_assignments(context, topology, bases, correlations, q25)
    abundances, abundance_reference = abundance_design(context)
    reference = domain_reference(context, args)
    scientific = canonical({"version": VERSION, "contract": CONTRACT, "matched_units": context["units"],
                  "source_v56_fingerprint": context["source_v56_fingerprint"], "source_hashes": context["source_hashes"],
                  "input_hashes": context["validation"]["hashes_sha256"], "v51_stage0_hashes": context["result_hashes"],
                  "implementation_hashes": source_implementations(), "candidate_pair_scores": scores,
                  "spectral_pair_design": topology, "healthy_pool": pool, "healthy_indices": sorted(bases),
                  "healthy_pair_correlations": correlations, "spatial_low_q25": q25,
                  "spatial_assignments": assignments, "abundance_multipliers": abundances,
                  "abundance_reference": abundance_reference, "domain_reference": reference})
    design = {"status": "DESIGN_PREPARED", "scientific": scientific, "design_fingerprint": fingerprint(scientific),
              "created_utc": datetime.now(timezone.utc).isoformat()}
    write_json(path, design)
    for name, values in (("matched_identity_units", context["units"]), ("candidate_pair_scores", scores),
                         ("spectral_pair_design", topology), ("spatial_assignments", assignments),
                         ("abundance_multipliers", abundances)):
        write_rows(args.output_dir / f"{name}.csv", values)
    write_json(args.output_dir / "domain_reference.json", reference)
    write_rows(args.output_dir / "dataset_manifest.csv", [{"dataset_id": d, "K": int(d.split("_")[2][1:]),
               "replicate": d.split("_")[1], "split": d.split("_")[0], "status": "PENDING_AUDIT"} for d in ids()])
    write_rows(args.output_dir / "oracle_summary.csv", [], ["dataset_id", "status", "candidate_FP", "candidate_FN", "molecular_FP", "molecular_FN"])
    v56.stage0.atomic_write_text(args.output_dir / "summary.md", "\n".join([
        "# V57 spectral–spatial identity confidence benchmark", "", "Status: DESIGN_PREPARED; audit and freeze pending.",
        "No oracle, sentinel or CLEAN training has been run by prepare-design.", "",
        *[f"- {line}" for line in LIMITATIONS], "", "Final deployment threshold frozen: NO.", ""]))
    print(json.dumps({"status": "DESIGN_PREPARED", "healthy_template_count": len(bases), "matched_units": 175,
                      "CAL_HOLD_overlap": 0, "block_sizes": dict(Counter(u["block"] for u in context["units"])),
                      "next": "--audit-design; no oracle or GPU run"}, indent=2))
    return design


def load_design(args, context, frozen=False):
    design = v54.read_json(args.output_dir / "design.json")
    require(not design["status"].startswith("INVALID") and design["status"] != "DESIGN_FAIL", "V57_INVALID_NO_REDESIGN_ALLOWED")
    s = design["scientific"]
    require(design["design_fingerprint"] == fingerprint(s), "DESIGN_CONTENT_CHANGED")
    require(s["contract"] == canonical(CONTRACT), "CONTRACT_CHANGED")
    require(s["implementation_hashes"] == source_implementations(), "IMPLEMENTATION_CHANGED")
    require(s["input_hashes"] == context["validation"]["hashes_sha256"] and
            s["v51_stage0_hashes"] == context["result_hashes"] and s["source_hashes"] == context["source_hashes"], "SOURCE_CHANGED")
    require(s["matched_units"] == canonical(context["units"]), "INHERITED_IDENTITY_CHANGED")
    require(not design["status"].startswith("INVALID") and design["status"] != "DESIGN_FAIL", "V57_INVALID_NO_REDESIGN_ALLOWED")
    if frozen:
        require(design["status"] == "DESIGN_FROZEN_BEFORE_TRAINING", "FREEZE_REQUIRED_BEFORE_ORACLE_OR_TRAINING")
    return design


def invalidate(args, design, status, diagnostic):
    design.update({"status": status, "diagnostic": diagnostic})
    write_json(args.output_dir / "design.json", design)


def audit(args, context, design):
    s = design["scientific"]
    audit_units(s["matched_units"], context)
    pool, bases = healthy_pool(context)
    require(canonical(pool) == s["healthy_pool"] and len(bases) == 97, "HEALTHY_POOL_CHANGED")
    correlations, q25 = base_correlations(bases, context["mask"])
    require(canonical(correlations) == s["healthy_pair_correlations"] and q25 == s["spatial_low_q25"], "BASE_CORRELATIONS_CHANGED")
    context["bases"] = bases
    abundance, ref = abundance_design(context)
    require(canonical(abundance) == s["abundance_multipliers"] and canonical(ref) == s["abundance_reference"], "ABUNDANCE_CHANGED")
    block_counts, high_corr, low_corr, reuse = {}, [], [], {}
    topology = s["spectral_pair_design"]
    saved_scores = {(r["unit_i"], r["unit_j"]): r for r in s["candidate_pair_scores"]}
    recomputed = spectral_scores(context)
    for row in recomputed:
        saved = saved_scores[row["unit_i"], row["unit_j"]]
        require(all(canonical(value) == saved[key] for key, value in row.items()), "FROZEN_SPECTRAL_SCORE_CHANGED")
    unit_blocks = {u["pair_id"]: u["block"] for u in s["matched_units"]}
    for block in range(1, 8):
        pairs = [p for p in topology if p["block"] == block]
        used = [u for p in pairs for u in (p["unit_i"], p["unit_j"])]
        require(len(used) == len(set(used)) == 16, "PAIR_TOPOLOGY_NOT_DISJOINT")
        counts = Counter(p["role"] for p in pairs)
        require(counts == Counter({role: 2 for role in ROLES}), "FACTORIAL_PAIR_BALANCE")
        for p in pairs:
            require(unit_blocks[p["unit_i"]] == unit_blocks[p["unit_j"]] == block, "PAIR_OUTSIDE_INHERITED_BLOCK")
            source = saved_scores[p["unit_i"], p["unit_j"]]
            require(source[f"{p['spectral_label']}_candidate_pool"], "PAIR_OUTSIDE_FIXED_35_PERCENT_POOL")
        for kind in ("hard", "easy"):
            four = sorted([p for p in pairs if p["spectral_label"] == kind], key=lambda p: (p["unit_i"], p["unit_j"]))
            metric = f"joint_{kind}_score"
            def split_gap(high):
                return (abs(np.mean([four[i][metric] for i in high]) - np.mean([four[i][metric] for i in range(4) if i not in high])),
                        tuple((four[i]["unit_i"], four[i]["unit_j"]) for i in high))
            expected_high = min(itertools.combinations(range(4), 2), key=split_gap)
            require(tuple(i for i in range(4) if four[i]["spatial_relation"] == "high") == expected_high,
                    "SPECTRAL_SEVERITY_SPATIAL_SPLIT_NOT_BALANCED")
        block_counts[block] = {**counts, "singleton_control": 9}
    for replicate in MAPPING_SEEDS:
        records = [r for r in s["spatial_assignments"] if r["replicate"] == replicate]
        require(len(records) == 175 and len({r["unit_id"] for r in records}) == 175, "SPATIAL_UNIT_COUNT")
        assignments = {}
        maps = {}
        for r in records:
            require(r["base_template_id"] in bases, "UNHEALTHY_ASSIGNMENT")
            assignments.setdefault(r["assignment_id"], r["base_template_id"])
            require(assignments[r["assignment_id"]] == r["base_template_id"], "ASSIGNMENT_BASE_CHANGED")
            maps[r["unit_id"]] = perturb(bases[r["base_template_id"]], context["mask"], r["perturbation_seed"])
        require(len(assignments) == 147, "BASE_ASSIGNMENT_COUNT")
        require({r["unit_id"] for r in records} == set(unit_blocks), "ASSIGNED_UNITS_CHANGED")
        for block in range(1, 8):
            block_records = [r for r in records if r["block"] == block]
            require(len(block_records) == 25 and len({r["assignment_id"] for r in block_records}) == 21,
                    "BLOCK_ASSIGNMENT_COUNT")
            require(Counter(r["role"] for r in block_records) == Counter({**{role: 4 for role in ROLES}, "singleton_control": 9}),
                    "BLOCK_SPATIAL_ROLE_COUNTS")
        for r in records:
            require(r["block"] == unit_blocks[r["unit_id"]] and r["perturbation_seed"] == seed(MAPPING_SEEDS[replicate], r["unit_id"], "spatial"),
                    "SPATIAL_SEED_OR_BLOCK_CHANGED")
            require(np.isclose(maps[r["unit_id"]][context["mask"]].mean(dtype=np.float64), 1, rtol=2e-6), "NORMALIZED_MAP_MEAN_CHANGED")
            require(hashlib.sha256(maps[r["unit_id"]].tobytes()).hexdigest() == r["map_sha256"], "SPATIAL_HASH_CHANGED")
        reuse[replicate] = max(Counter(assignments.values()).values())
        require(reuse[replicate] <= 2, "BASE_REUSE_EXCEEDED")
        lookup = {r["unit_id"]: r for r in records}
        for p in topology:
            a, b = lookup[p["unit_i"]], lookup[p["unit_j"]]
            corr = correlation(maps[p["unit_i"]], maps[p["unit_j"]], context["mask"])
            require(corr == a["realized_spatial_correlation"] == b["realized_spatial_correlation"], "REALIZED_CORRELATION_CHANGED")
            require(a["role"] == b["role"] == p["role"] and a["pair_id"] == b["pair_id"] == p["pair_id"], "SPATIAL_ROLE_CHANGED")
            if p["spatial_relation"] == "high":
                require(a["assignment_id"] == b["assignment_id"] and corr >= .95, "SPATIAL_HIGH_FAIL")
                high_corr.append(corr)
            else:
                base_corr = correlation(bases[a["base_template_id"]], bases[b["base_template_id"]], context["mask"])
                require(a["base_template_id"] != b["base_template_id"] and base_corr <= q25 and corr <= q25 + .05, "SPATIAL_LOW_FAIL")
                low_corr.append(corr)
    reference = domain_reference(context, args)
    require(canonical(reference) == s["domain_reference"], "DOMAIN_REFERENCE_CHANGED")
    domain_rows, failures = [], []
    for dataset in ids():
        case = construct_case(context, s, dataset)
        stats = domain_stats(case["B_sim"], context["mask"])
        k = int(dataset.split("_")[2][1:])
        passed = case["reportable_truth_molecular_identity_count"] == k and all(
            stats[m] <= reference["upper_bounds"][m] for m in TAILS)
        passed = passed and np.isclose(stats["p50"], CONTRACT["signal_target"], rtol=2e-6)
        row = {"dataset_id": dataset, "K": k, "replicate": dataset.split("_")[1], "split": dataset.split("_")[0],
               **stats, "global_scale": case["global_scale"], "truth_count": k,
               "reportable_truth_count": case["reportable_truth_molecular_identity_count"],
               "K_true_pixel_foreground": case["K_true_pixel_foreground"], "status": "PASS" if passed else "FAIL",
               **{f"upper_bound_{m}": v for m, v in reference["upper_bounds"].items()}}
        domain_rows.append(row)
        if not passed:
            failures.append(dataset)
    write_rows(args.output_dir / "synthetic_B_domain_audit.csv", domain_rows)
    write_rows(args.output_dir / "dataset_manifest.csv", domain_rows)
    result = {"status": "FAIL" if failures else "PASS", "design_fingerprint": design["design_fingerprint"],
              "healthy_template_count": len(bases), "expected_healthy_count": 97, "matched_unit_count": 175,
              "CAL_HOLD_overlap": 0, "block_sizes": dict(Counter(u["block"] for u in s["matched_units"])),
              "per_block_counts": block_counts, "high_realized_corr": v54.numeric_summary(high_corr),
              "low_base_q25": q25, "low_realized_corr": v54.numeric_summary(low_corr), "max_base_reuse_by_replicate": reuse,
              "exact_nested_blocks": True, "CAL_HOLD_same_spatial_and_abundance": True,
              "reference_domain": reference, "all_60_dataset_domain": domain_rows,
              "failures": failures, "review_confirmed": False}
    write_json(args.output_dir / "design_audit.json", result)
    print(json.dumps(canonical(result), indent=2))
    if failures:
        invalidate(args, design, "DESIGN_FAIL", failures)
        raise RuntimeError("DESIGN_FAIL: reportability or B-domain gate; no V57 redesign")
    return result


def freeze(args, context, design):
    require(design["status"] == "DESIGN_PREPARED", "DESIGN_ALREADY_FROZEN_OR_INVALID")
    checked = v54.read_json(args.output_dir / "design_audit.json")
    require(checked["design_fingerprint"] == design["design_fingerprint"] and checked["status"] == "PASS",
            "ALL_DESIGN_AUDITS_MUST_PASS_BEFORE_FREEZE")
    require(len(checked["all_60_dataset_domain"]) == 60 and
            {r["dataset_id"] for r in checked["all_60_dataset_domain"]} == set(ids()) and
            all(r["status"] == "PASS" and r["truth_count"] == r["reportable_truth_count"]
                for r in checked["all_60_dataset_domain"]), "INCOMPLETE_DOMAIN_AUDIT")
    design.update({"status": "DESIGN_FROZEN_BEFORE_TRAINING", "review_confirmed_utc": datetime.now(timezone.utc).isoformat(),
                   "frozen_audit_sha256": v56.digest(args.output_dir / "design_audit.json")})
    write_json(args.output_dir / "design.json", design)
    print("DESIGN_FROZEN_BEFORE_TRAINING; no oracle or GPU executed")


def require_frozen_audit(args, design):
    require(design["status"] == "DESIGN_FROZEN_BEFORE_TRAINING", "FROZEN_DESIGN_REQUIRED")
    require(design["frozen_audit_sha256"] == v56.digest(args.output_dir / "design_audit.json"), "FROZEN_AUDIT_CHANGED")


def case_with_gate(args, context, design, dataset):
    case = construct_case(context, design["scientific"], dataset)
    require(case["reportable_truth_molecular_identity_count"] == case["global_truth_K"], "REPORTABILITY_GATE_FAILED")
    stats = domain_stats(case["B_sim"], context["mask"])
    require(all(stats[m] <= design["scientific"]["domain_reference"]["upper_bounds"][m] for m in TAILS) and
            np.isclose(stats["p50"], CONTRACT["signal_target"], rtol=2e-6), "B_DOMAIN_GATE_FAILED")
    return case


def oracle_all(args, context, design):
    require_frozen_audit(args, design)
    summaries, hashes = [], {}
    for dataset in ids():
        directory = args.output_dir / "clean" / dataset
        path = directory / "oracle.json"
        if path.exists():
            report = v54.read_json(path)
            require(report["design_fingerprint"] == design["design_fingerprint"] and report["dataset_id"] == dataset,
                    "ORACLE_PROVENANCE_CHANGED")
        else:
            case = case_with_gate(args, context, design, dataset)
            try:
                oracle = v54.reporting_gate_oracle(case, context["A_solver"], context["metadata"])
            except Exception as exc:
                diagnostic = {"dataset_id": dataset, "design_fingerprint": design["design_fingerprint"],
                              "status": "INVALID_EXACT_IDENTIFIABILITY", "diagnostic": str(exc)}
                write_json(path, diagnostic)
                summaries.append({"dataset_id": dataset, "status": "INVALID_EXACT_IDENTIFIABILITY", "diagnostic": str(exc)})
                write_rows(args.output_dir / "oracle_summary.csv", summaries)
                invalidate(args, design, "INVALID_EXACT_IDENTIFIABILITY", diagnostic)
                raise
            candidate = oracle["candidate_level_all_truth_metrics"]
            molecular = oracle["all_truth_metrics"]
            passed = all(m["FP"] == m["FN"] == 0 for m in (candidate, molecular))
            report = {"dataset_id": dataset, "design_fingerprint": design["design_fingerprint"],
                      "status": "PASS" if passed else "INVALID_EXACT_IDENTIFIABILITY", "oracle": oracle,
                      "weighting": "W=I", "solver_library": "production A_solver, full 391 candidates"}
            write_json(path, report)
        candidate, molecular = report["oracle"]["candidate_level_all_truth_metrics"], report["oracle"]["all_truth_metrics"]
        passed = all(m["FP"] == m["FN"] == 0 for m in (candidate, molecular))
        summaries.append({"dataset_id": dataset, "status": "PASS" if passed else "INVALID_EXACT_IDENTIFIABILITY",
                          "candidate_FP": candidate["FP"], "candidate_FN": candidate["FN"],
                          "molecular_FP": molecular["FP"], "molecular_FN": molecular["FN"],
                          "reconstruction_relative_residual": report["oracle"]["reconstruction_relative_residual"]})
        write_rows(args.output_dir / "oracle_summary.csv", summaries)
        if not passed:
            invalidate(args, design, "INVALID_EXACT_IDENTIFIABILITY", {"dataset_id": dataset, "oracle": report})
            raise RuntimeError("INVALID_EXACT_IDENTIFIABILITY; no GPU or V57 redesign allowed")
        hashes[dataset] = v56.digest(path)
        print(f"Oracle {dataset}: PASS ({len(summaries)}/60)", flush=True)
    write_json(args.output_dir / "oracle_all_validity.json", {
        "status": "PASS", "count": 60, "design_fingerprint": design["design_fingerprint"],
        "oracle_sha256": hashes, "summary_sha256": v56.digest(args.output_dir / "oracle_summary.csv")})


def require_oracles(args, design):
    require_frozen_audit(args, design)
    gate = v54.read_json(args.output_dir / "oracle_all_validity.json")
    require(gate["status"] == "PASS" and gate["count"] == 60 and
            gate["design_fingerprint"] == design["design_fingerprint"] and set(gate["oracle_sha256"]) == set(ids()),
            "ALL_60_ORACLES_REQUIRED_BEFORE_TRAINING")
    require(gate["summary_sha256"] == v56.digest(args.output_dir / "oracle_summary.csv"), "ORACLE_SUMMARY_CHANGED")
    for dataset, checksum in gate["oracle_sha256"].items():
        require(checksum == v56.digest(args.output_dir / "clean" / dataset / "oracle.json"), "ORACLE_REPORT_CHANGED")


def train_or_resume(args, context, design, dataset, case):
    """Use the production driver verbatim; only observation storage is added."""
    from config_758 import Cfg
    for key, value in {**v54.v50.CFG_DEFAULTS, "parent_channel_weight_multiplier": 1.0}.items():
        require(getattr(Cfg, key) == value, f"PRODUCTION_CONFIG_CHANGED: {key}")
    require(v54.MAX_EPOCH == 3000 and list(v54.CHECKPOINT_EPOCHS) == list(CONTRACT["checkpoint_epochs"]),
            "TRAINING_CAP_OR_CHECKPOINTS_CHANGED")
    directory = args.output_dir / "clean" / dataset
    directory.mkdir(parents=True, exist_ok=True)
    bound = {"dataset_id": dataset, "design_fingerprint": design["design_fingerprint"],
             "B_sha256": hashlib.sha256(case["B_sim"].tobytes()).hexdigest()}
    binding = directory / "runtime_contract.json"
    if binding.exists():
        require(v54.read_json(binding) == bound, "CROSS_DATASET_CHECKPOINT_OR_CHANGED_INPUT")
    else:
        require(not list(directory.glob("*.pth")) and not (directory / "learned_arrays.npz").exists(), "UNBOUND_CHECKPOINT")
        write_json(binding, bound)
    arrays = directory / "learned_arrays.npz"
    result_path = directory / "solver_run.json"
    if arrays.exists() and result_path.exists():
        saved = v54.read_json(result_path)
        require(saved["design_fingerprint"] == design["design_fingerprint"] and
                saved["arrays_sha256"] == v56.digest(arrays), "CACHED_SOLVER_OUTPUT_CHANGED")
        with np.load(arrays) as data:
            return {**saved["training"], "X_hat": data["X_hat"].copy(), "B_hat": data["B_hat"].copy()}
    training_case = {"case_name": case["case_name"], "B_sim": case["B_sim"], "foreground_mask": case["foreground_mask"]}
    learned = v54.train_resumable(training_case, context, directory, device_name=args.device)
    training = {k: v for k, v in learned.items() if k not in ("X_hat", "B_hat")}
    temporary = directory / "learned_arrays.tmp.npz"
    np.savez_compressed(temporary, X_hat=learned["X_hat"], B_hat=learned["B_hat"])
    temporary.replace(arrays)
    write_json(result_path, {"dataset_id": dataset, "design_fingerprint": design["design_fingerprint"],
                            "arrays_sha256": v56.digest(arrays), "training": training,
                            "X_true_used_in_training_or_stopping": False})
    return learned


def reconstruction_residual(learned, case):
    mask = case["foreground_mask"]
    return float(np.linalg.norm(learned["B_hat"][:, mask] - case["B_sim"][:, mask]) /
                 max(float(np.linalg.norm(case["B_sim"][:, mask])), 1e-30))


def sentinel(args, context, design):
    require_oracles(args, design)
    path = args.output_dir / "sentinel_solver_validity.json"
    previous = v54.read_json(path) if path.exists() else None
    if previous:
        require(previous["design_fingerprint"] == design["design_fingerprint"], "SENTINEL_DESIGN_CHANGED")
        require(previous["status"] != "FAIL", "SENTINEL_NOT_PASS; no clean-all or outcome-driven redesign")
    results = []

    def save_gate(status):
        write_json(path, {"status": status, "design_fingerprint": design["design_fingerprint"], "datasets": results,
                          "validity_metric": "normal finite completion and final foreground reconstruction relative residual <=0.10",
                          "duration_requirement": "none; production early-stop retained; 3000 epochs is the hard cap only",
                          "identity_outcomes_used": False, "rho_computed": False})

    for dataset in SENTINELS:
        # Fail closed even if the process is killed before an exception can be saved.
        results.append({"dataset_id": dataset, "status": "FAIL", "epoch": None,
                        "final_residual": None, "failure_reason": "RUN_DID_NOT_COMPLETE"})
        save_gate("FAIL")
        try:
            case = case_with_gate(args, context, design, dataset)
            learned = train_or_resume(args, context, design, dataset, case)
            residual = reconstruction_residual(learned, case)
            diagnostics_path = args.output_dir / "clean" / dataset / "checkpoint_diagnostics.csv"
            diagnostics = rows(diagnostics_path) if diagnostics_path.exists() else []
            trajectory = [{"epoch": r["epoch"], "reconstruction_relative_residual": r["reconstruction_relative_residual"]}
                          for r in diagnostics]
            trajectory.append({"epoch": learned["stopped_epoch"],
                               "reconstruction_relative_residual": residual if np.isfinite(residual) else None})
            solver_finite = bool(np.isfinite(learned["X_hat"]).all() and np.isfinite(learned["B_hat"]).all() and
                                 np.isfinite(learned["final_physical_loss"]) and np.isfinite(learned["final_raw_losses"]).all())
            completed_normally = learned["stop_reason"] in ("converged", "max_epochs")
            residual_pass = bool(np.isfinite(residual) and residual <= CONTRACT["sentinel_max_residual"])
            status = "PASS" if completed_normally and solver_finite and residual_pass else "FAIL"
            results[-1] = {"dataset_id": dataset, "epoch": learned["stopped_epoch"], "stop_reason": learned["stop_reason"],
                           "reconstruction_residual_trajectory": trajectory, "final_residual": residual if np.isfinite(residual) else None,
                           "solver_finite": solver_finite, "completed_normally": completed_normally,
                           "residual_gate_pass": residual_pass, "status": status,
                           "solver_run_sha256": v56.digest(args.output_dir / "clean" / dataset / "solver_run.json")}
        except (Exception, KeyboardInterrupt) as exc:
            results[-1]["failure_reason"] = f"CRASH: {type(exc).__name__}: {exc}"
            save_gate("FAIL")
            invalidate(args, design, "INVALID_LEARNED_SOLVER_DOMAIN", results[-1])
            raise
        save_gate(status if status != "PASS" else ("PASS" if len(results) == 2 else "IN_PROGRESS"))
        if status != "PASS":
            invalidate(args, design, "INVALID_LEARNED_SOLVER_DOMAIN", results[-1])
            raise RuntimeError(f"SENTINEL_{status}: {dataset}; clean-all remains blocked")
    print("SENTINEL PASS 2/2; inspect sentinel_solver_validity.json before requesting CLEAN")


def require_sentinel(args, design):
    require_oracles(args, design)
    gate = v54.read_json(args.output_dir / "sentinel_solver_validity.json")
    require(gate["status"] == "PASS" and gate["design_fingerprint"] == design["design_fingerprint"] and
            {r["dataset_id"] for r in gate["datasets"]} == set(SENTINELS), "SENTINEL_2_OF_2_PASS_REQUIRED")
    for r in gate["datasets"]:
        require(r["status"] == "PASS" and r.get("completed_normally") and r.get("solver_finite") and
                r["final_residual"] is not None and np.isfinite(r["final_residual"]) and
                r["final_residual"] <= CONTRACT["sentinel_max_residual"] and
                r["solver_run_sha256"] == v56.digest(args.output_dir / "clean" / r["dataset_id"] / "solver_run.json"),
                "SENTINEL_PROVENANCE_OR_VALIDITY_CHANGED")


def mechanism_metadata(scientific, dataset, context):
    split, replicate, label = dataset.split("_")
    k = int(label[1:])
    topology = {u: p for p in scientific["spectral_pair_design"] for u in (p["unit_i"], p["unit_j"])}
    spatial = {r["unit_id"]: r for r in scientific["spatial_assignments"] if r["replicate"] == replicate}
    abundance = {r["unit_id"]: r["abundance_multiplier"] for r in scientific["abundance_multipliers"]}
    geometry = v56.analysis_geometry(context)
    meta = {}
    for i in range(391):
        meta[i] = {**geometry[i], "pair_id": None, "unit_id": None, "role": "UNASSIGNED_FALSE_POSITIVE",
                   "spectral_score_CAL": None, "spectral_score_HOLD": None, "joint_hard_score": None,
                   "joint_easy_score": None, **{m: None for m in PAIR_METRICS}, "base_template_id": None,
                   "realized_spatial_correlation": None, "abundance_multiplier": None,
                   "K": k, "replicate": replicate, "split": split}
    for unit in scientific["matched_units"]:
        if unit["block"] > k // 25:
            continue
        uid, i = unit["pair_id"], unit[f"{split}_index"]
        p, a = topology.get(uid), spatial[uid]
        meta[i].update({"unit_id": uid, "pair_id": p["pair_id"] if p else None,
                        "role": p["role"] if p else "singleton_control", "base_template_id": a["base_template_id"],
                        "realized_spatial_correlation": a["realized_spatial_correlation"], "abundance_multiplier": abundance[uid]})
        if p:
            meta[i].update({key: p[key] for key in ("spectral_score_CAL", "spectral_score_HOLD", "joint_hard_score", "joint_easy_score")})
            meta[i].update({m: p[f"geometry_{split}"][m] for m in PAIR_METRICS})
    return meta


def molecular_all_records(candidate_records, units):
    reported = {u["lipid_name"]: u for u in units}
    grouped = defaultdict(list)
    for row in candidate_records:
        grouped[row["lipid_name"]].append(row)
    result = []
    for name, members in sorted(grouped.items()):
        unit, first = reported.get(name), members[0]
        record = {k: v for k, v in first.items() if k not in ("candidate_index", "candidate_id")}
        record.update({"candidate_indices": [m["candidate_index"] for m in members],
                       "raw_solver_reported": unit is not None, "molecular_truth": any(m["molecular_truth"] for m in members),
                       "reportable_truth": any(m["reportable_truth"] for m in members),
                       "rho_zero": unit["rho_zero"] if unit else None,
                       "rho_zero_status": "COMPUTED" if unit else "NOT_COMPUTED",
                       "X_hat": unit["X_hat"] if unit else sum(m["X_hat"] for m in members),
                       "X_hat_definition": "sum reported candidates for primary score; all candidates for unreported descriptive abundance"})
        if len(members) > 1:
            for field in v56.GEOMETRY_FIELDS:
                record[field] = {str(m["candidate_index"]): m[field] for m in members}
        result.append(record)
    return result


def run_dataset(args, context, design, dataset):
    require_sentinel(args, design)
    directory = args.output_dir / "clean" / dataset
    report_path = directory / "report.json"
    if report_path.exists():
        existing = v54.read_json(report_path)
        require(existing["design_fingerprint"] == design["design_fingerprint"], "DATASET_REPORT_CHANGED")
        if existing["status"] == "COMPLETE":
            return existing
    case = case_with_gate(args, context, design, dataset)
    learned = train_or_resume(args, context, design, dataset, case)
    require(learned["stop_reason"] != "nonfinite_loss", f"NONFINITE_TRAINING: {dataset}")
    records, raw = v54.learned_records(dataset, case, learned, context, args.rho_workers)
    metadata = mechanism_metadata(design["scientific"], dataset, context)
    names = context["metadata"]["lipid_name"]
    for record in records:
        record.update(metadata[record["candidate_index"]])
        record["reportable_truth"] = record["lipid_name"] in raw["reportable_truth_lipid_names"]
    fields = ["dataset_id", "split", "replicate", "complexity_condition", "global_truth_K", "candidate_index",
              "candidate_id", "lipid_name", "X_hat", "candidate_truth", "molecular_truth",
              "same_lipid_alternative_candidate", "rho_zero", "reportable_truth", *metadata[0]]
    write_rows(directory / "reported_identity_records.csv", records, list(dict.fromkeys(fields)))
    xhat = learned["X_hat"][:, context["mask"]].mean(axis=1, dtype=np.float64)
    reported = {r["candidate_index"]: r for r in records}
    candidates = [{"dataset_id": dataset, "candidate_index": i, "candidate_id": str(context["metadata"]["candidate_id"][i]),
                   "lipid_name": str(name), "X_hat": float(xhat[i]), "rho_zero": reported[i]["rho_zero"] if i in reported else None,
                   "rho_zero_status": "COMPUTED" if i in reported else "NOT_COMPUTED", "raw_solver_reported": i in reported,
                   "candidate_truth": i in case["active_indices"], "molecular_truth": str(name) in raw["truth_lipid_names"],
                   "reportable_truth": str(name) in raw["reportable_truth_lipid_names"], **metadata[i]}
                  for i, name in enumerate(names)]
    report = {"status": "COMPLETE", "dataset_id": dataset, "condition": "CLEAN", "design_fingerprint": design["design_fingerprint"],
              "global_truth_K": case["global_truth_K"], "all_truth_molecular_identity_count": case["global_truth_K"],
              "reportable_truth_molecular_identity_count": case["reportable_truth_molecular_identity_count"],
              "global_scale": case["global_scale"], "K_true_pixel_foreground": case["K_true_pixel_foreground"],
              "learned_raw_identity_performance": raw,
              "training": {k: v for k, v in learned.items() if k not in ("X_hat", "B_hat")},
              "X_true_used_in_training_or_stopping": False, "rho_zero_weighting": "W=I"}
    units = v54.molecular_units(records, {dataset: report})
    raw["molecular_level"].update(v56.metrics(units, {dataset: report}))
    write_rows(directory / "candidate_false_negative_records.csv", candidates)
    write_rows(directory / "molecular_false_negative_records.csv", molecular_all_records(candidates, units))
    write_json(report_path, report)
    print(f"COMPLETE {dataset}", flush=True)
    return report


def decorate_records(records, pack):
    result = []
    for source in records:
        row = dict(source)
        for field in ("rho_zero", "pair_id", "base_template_id", "realized_spatial_correlation"):
            if row.get(field) == "":
                row[field] = None
        for score, targets in pack.items():
            for target, entry in targets.items():
                retained = bool(row["raw_solver_reported"] and entry is not None and row[score] is not None and row[score] >= entry["threshold"])
                row[f"retained_by_{score}_global_{target}"] = retained
                row[f"{score}_{target}_threshold_available"] = entry is not None
                if score == "rho_zero":
                    row[f"retained_by_global_{target}"] = retained
                    row[f"filter_induced_true_loss_{target}"] = bool(row["molecular_truth"] and row["raw_solver_reported"] and not retained)
        row["solver_false_negative"] = bool(row["molecular_truth"] and not row["raw_solver_reported"])
        result.append(row)
    return result


def mechanism_metrics(molecular, pack):
    output = []
    for split in ("CAL", "HOLD"):
        for k in (None, *K_LEVELS):
            scope = [r for r in molecular if r["split"] == split and (k is None or r["K"] == k)]
            for role in (*ROLES, "singleton_control", "UNASSIGNED_FALSE_POSITIVE"):
                category = [r for r in scope if r["role"] == role]
                truth_count = sum(r["molecular_truth"] for r in category)
                raw = [r for r in category if r["raw_solver_reported"]]
                raw_tp = sum(r["molecular_truth"] for r in raw)
                for target in ("FDR5", "FDR1"):
                    retained = [r for r in raw if r[f"retained_by_global_{target}"]]
                    tp = sum(r["molecular_truth"] for r in retained)
                    true_rho = [r["rho_zero"] for r in raw if r["molecular_truth"]]
                    false_rho = [r["rho_zero"] for r in raw if not r["molecular_truth"]]
                    row = {"analysis": "SECONDARY_NO_CATEGORY_CALIBRATION", "split": split, "K": k if k else "GLOBAL",
                           "role": role, "target": target, "global_rho_threshold": pack["rho_zero"][target],
                           **v56.recall_accounting(raw_tp, len(raw) - raw_tp, truth_count - raw_tp, tp, len(retained) - tp),
                           "all_truth_recall": tp / truth_count if truth_count else None,
                           "reportable_truth_recall": tp / truth_count if truth_count else None,
                           "median_true_rho": float(np.median(true_rho)) if true_rho else None,
                           "median_false_rho": float(np.median(false_rho)) if false_rho else None,
                           "FDR": None, "precision": None,
                           "FDR_scope": "NOT_APPLICABLE: truth roles do not assign false positives; see global FDR and unassigned FP counts"}
                    if role != "UNASSIGNED_FALSE_POSITIVE":
                        # Zero attributed FP is not evidence of zero category FDR.
                        row["raw_solver_FP"] = None
                        row["filtered_FP"] = None
                    output.append(row)
    return output


def aggregate(args, context, design):
    require_sentinel(args, design)
    output, fp = args.output_dir, design["design_fingerprint"]
    # Reuse V56 statistics only; no V56 designs, training or thresholds are loaded.
    cal, cal_reports, cal_hashes = v56.load_completed(output, "CLEAN", "CAL", K_LEVELS, fp)
    pack, curves = v56.threshold_pack(cal, cal_reports)
    frozen = {"status": "FROZEN_FROM_CAL_ONLY", "design_fingerprint": fp, "CAL_source_hashes": cal_hashes,
              "global_thresholds": pack,
              "rho_tau_FDR5": pack["rho_zero"]["FDR5"]["threshold"] if pack["rho_zero"]["FDR5"] else None,
              "rho_tau_FDR1": pack["rho_zero"]["FDR1"]["threshold"] if pack["rho_zero"]["FDR1"] else None,
              "selection": "maximum retained molecular units with empirical CAL FDR<=target; lower threshold breaks ties",
              "molecular_score": "max reported-candidate rho; sum reported-candidate X_hat; no group-rho"}
    path = output / "global_frozen_thresholds.json"
    if path.exists():
        require(v54.read_json(path) == canonical(frozen), "FROZEN_CAL_THRESHOLDS_CHANGED")
    else:
        write_json(path, frozen)
    # HOLD files may be read only after the threshold freeze above.
    hold, hold_reports, _ = v56.load_completed(output, "CLEAN", "HOLD", K_LEVELS, fp)
    overall, by_k, by_r = v56.validation(hold, hold_reports, pack, K_LEVELS)
    molecular, candidates = [], []
    for split in ("CAL", "HOLD"):
        split_molecular = []
        for dataset in ids(split):
            directory = output / "clean" / dataset
            ms = decorate_records(rows(directory / "molecular_false_negative_records.csv"), pack)
            lookup = {r["lipid_name"]: r for r in ms}
            cs = rows(directory / "candidate_false_negative_records.csv")
            for c in cs:
                if c["rho_zero"] == "":
                    c["rho_zero"] = None
                # Candidate records carry their candidate score and molecular retention decision.
                for key, value in lookup[c["lipid_name"]].items():
                    if key.startswith(("retained_by_", "filter_induced_true_loss_")):
                        c[key] = value
                c["retention_flag_unit"] = "dataset/lipid_name"
            candidates.extend(cs)
            molecular.extend(ms)
            split_molecular.extend(ms)
        write_rows(output / ("calibration_records.csv" if split == "CAL" else "heldout_records.csv"), split_molecular)
    write_rows(output / "candidate_false_negative_records.csv", candidates)
    write_rows(output / "molecular_false_negative_records.csv", molecular)
    mechanism = mechanism_metrics(molecular, pack)
    write_rows(output / "mechanism_stratified_metrics.csv", mechanism)
    for score, label in (("rho_zero", "rho"), ("X_hat", "xhat")):
        points = []
        for split, us, rs in (("CAL", cal, cal_reports), ("HOLD", hold, hold_reports)):
            for k in (None, *K_LEVELS):
                subset, subreports = v56.subset(us, rs, k=k)
                points += [{"split": split, "K": k if k else "GLOBAL", "score": score,
                            "role": "DESCRIPTIVE_CURVE_NOT_HOLD_CALIBRATION", **r}
                           for r in v56.calibration_curve(subset, score, subreports)]
        empty_fields = ["split", "K", "score", "threshold", "FDR", "all_truth_recall", "reportable_truth_recall", "TP_retention"]
        write_rows(output / f"fdr_recall_curve_{label}.csv", points, None if points else empty_fields)
        write_rows(output / f"fdr_tp_retention_curve_{label}.csv", points, None if points else empty_fields)
    secondary = {}
    for k in K_LEVELS:
        cu, cr = v56.subset(cal, cal_reports, k=k)
        hu, hr = v56.subset(hold, hold_reports, k=k)
        secondary[str(k)] = v56.validation(hu, hr, v56.threshold_pack(cu, cr)[0], (k,))[0]
    report = {"status": "PENDING_REVIEW", "design_fingerprint": fp, "primary_global_CAL_thresholds": frozen,
              "heldout_global": overall, "heldout_by_K": by_k, "heldout_by_replicate": by_r,
              "raw_CAL": v56.metrics(cal, cal_reports), "raw_HOLD": v56.metrics(hold, hold_reports),
              "SECONDARY_K_SPECIFIC_CALIBRATION": secondary,
              "mechanism_stratified_metrics": mechanism, "limitations": LIMITATIONS,
              "final_deployment_threshold_frozen": False}
    write_json(output / "report.json", report)
    write_rows(output / "heldout_by_K.csv", by_k)
    write_rows(output / "heldout_by_replicate.csv", by_r)
    text = ["# V57 spectral–spatial identity confidence benchmark", "", "Status: PENDING_REVIEW", "",
            "## Primary global HOLD validation", "", "```json", json.dumps(canonical(overall), indent=2), "```", "",
            "## Interpretation boundaries", "", *[f"- {s}" for s in LIMITATIONS], "",
            "Final deployment threshold frozen: NO."]
    v56.stage0.atomic_write_text(output / "summary.md", "\n".join(text) + "\n")
    print("V57 CLEAN analysis written; global thresholds frozen before HOLD; final deployment threshold not reviewed")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-root", type=Path, default=ROOT.parent / "decon-lipid")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--v54-output", type=Path, default=ROOT / "results/v54_complexity_calibration")
    parser.add_argument("--v56-output", type=Path, default=ROOT / "results/v56_paper_identity_fdr_benchmark")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--rho-workers", type=int, default=1)
    action = parser.add_mutually_exclusive_group(required=True)
    for name in ("prepare-design", "audit-design", "freeze-design", "oracle-all", "sentinel", "clean-all", "aggregate-clean"):
        action.add_argument(f"--{name}", action="store_true")
    action.add_argument("--dataset", choices=ids())
    args = parser.parse_args()
    if args.rho_workers < 1:
        parser.error("--rho-workers must be positive")
    args.output_dir = args.output_dir.resolve()
    args.v54_output = args.v54_output.resolve()
    args.v56_output = args.v56_output.resolve()
    return args


def main():
    args = parse_args()
    context = load_inputs(args)
    if args.prepare_design:
        try:
            prepare(args, context)
        except Exception as exc:
            # Persist failure, but never overwrite an existing design on a repeat command.
            path = args.output_dir / "design.json"
            if not path.exists():
                write_json(path, {"status": "DESIGN_FAIL", "diagnostic": str(exc), "version": VERSION})
            raise
        return
    design = load_design(args, context, frozen=not (args.audit_design or args.freeze_design))
    if args.audit_design:
        require(design["status"] == "DESIGN_PREPARED", "AUDIT_BEFORE_FREEZE_ONLY; frozen audit cannot be replaced")
        try:
            audit(args, context, design)
        except Exception as exc:
            invalidate(args, design, "DESIGN_FAIL", str(exc))
            raise
    elif args.freeze_design:
        freeze(args, context, design)
    elif args.oracle_all:
        oracle_all(args, context, design)
    elif args.sentinel:
        sentinel(args, context, design)
    elif args.aggregate_clean:
        aggregate(args, context, design)
    else:
        require_sentinel(args, design)
        for dataset in ([args.dataset] if args.dataset else ids()):
            run_dataset(args, context, design, dataset)


if __name__ == "__main__":
    main()
