#!/usr/bin/env python3
"""Audit competition geometry in the frozen 391-candidate observation matrix.

This is a CPU-only structural audit.  It reads the frozen Stage-0 assets and
reuses the complete v50 cone-isolation cache; it never recomputes cone NNLS.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

DATA_RELATIVE = Path(
    "adapter_pipeline/outputs/"
    "v38_ce29_fragmentionfixed_profile_overlap_empiricalfwhm_globalq99_ista_ready"
)
OUTPUT_DEFAULT = ROOT / "results/v51_competition_structure"
LOCK_PATH = ROOT / "results/v48_production_ista_lock/v48_production_ista_lock.json"
CONE_CACHE_PATH = ROOT / "results/v50_reoptimized_sanity_k1_k3/design.json"
EXPECTED_A_SHAPE = (1084, 391)
PARENT_MZ_RANGE = (748.0, 803.0)
SUPPORT_THRESHOLD = 1.0e-8
N_MECHANISM_EXAMPLES = 3


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    temporary.replace(path)


def asset_paths(asset_root: Path) -> dict[str, Path]:
    data = asset_root / DATA_RELATIVE
    return {
        "A_library": data / "A_library.npy",
        "channel_axis": data / "shared_mz_final.npy",
        "candidate_metadata": data / "candidate_metadata_final.npy",
    }


def validate_assets(paths: dict[str, Path]) -> dict:
    required = [*paths.values(), LOCK_PATH, CONE_CACHE_PATH]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise RuntimeError("MISSING_DEPENDENCY: " + "; ".join(missing))

    lock = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    expected = lock["hashes_sha256"]
    lock_names = {
        "A_library": "A_library",
        "channel_axis": "channel_axis",
        "candidate_metadata": "candidate_metadata",
    }
    actual = {}
    for asset_name, lock_name in lock_names.items():
        actual[asset_name] = sha256(paths[asset_name])
        if actual[asset_name] != expected[lock_name]:
            raise RuntimeError(f"STAGE0_LOCK_MISMATCH: {asset_name}")
    for relative, lock_name in {
        "src/config_758.py": "production_config",
        "src/run_758_ista.py": "production_main_script",
        "src/lipid_ista.py": "production_solver_implementation",
    }.items():
        if sha256(ROOT / relative) != expected[lock_name]:
            raise RuntimeError(f"STAGE0_LOCK_MISMATCH: {relative}")
    return {"lock_path": str(LOCK_PATH), "validated_hashes": actual}


def load_inputs(paths: dict[str, Path]) -> tuple[np.ndarray, np.ndarray, dict]:
    from config_758 import Cfg

    if tuple(map(float, Cfg.parent_channel_mz_range)) != PARENT_MZ_RANGE:
        raise RuntimeError(
            "STAGE0_LOCK_MISMATCH: frozen parent_channel_mz_range is not 748–803"
        )
    A = np.load(paths["A_library"]).astype(np.float64)
    if A.ndim == 3:
        A = A[0]
    mz = np.asarray(np.load(paths["channel_axis"]), dtype=np.float64).reshape(-1)
    metadata = np.load(paths["candidate_metadata"], allow_pickle=True).item()
    if A.shape != EXPECTED_A_SHAPE:
        raise RuntimeError(f"STAGE0_LOCK_MISMATCH: A shape {A.shape}")
    if mz.shape != (A.shape[0],):
        raise RuntimeError(f"STAGE0_LOCK_MISMATCH: channel axis shape {mz.shape}")
    for key in ("candidate_id", "lipid_name", "lipid_class"):
        if key not in metadata or len(metadata[key]) != A.shape[1]:
            raise RuntimeError(f"STAGE0_LOCK_MISMATCH: candidate metadata {key}")
    return A, mz, metadata


def load_cached_cone_isolation(candidate_count: int) -> tuple[np.ndarray, dict]:
    design = json.loads(CONE_CACHE_PATH.read_text(encoding="utf-8"))
    rows = design.get("cone_isolation_ranking", [])
    if len(rows) != candidate_count:
        raise RuntimeError(
            "MISSING_DEPENDENCY: v50 design.json lacks the complete frozen "
            "cone-isolation ranking; refusing to recompute cone NNLS"
        )
    scores = np.full(candidate_count, np.nan, dtype=np.float64)
    duplicate = False
    for row in rows:
        index = int(row["candidate_index"])
        if index < 0 or index >= candidate_count or np.isfinite(scores[index]):
            duplicate = True
            break
        scores[index] = float(row["cone_isolation"])
    if duplicate or not np.all(np.isfinite(scores)):
        raise RuntimeError("INVALID_CACHED_DESIGN: incomplete cone-isolation map")
    contributors = {
        int(row["candidate_index"]): row.get("top_nonnegative_contributors")
        for row in rows
        if row.get("top_nonnegative_contributors")
    }
    return scores, {
        "path": str(CONE_CACHE_PATH),
        "candidate_count": len(rows),
        "cone_nnls_recomputed": False,
        "cached_contributor_solutions": len(contributors),
        "contributors": contributors,
    }


def cosine_matrix(block: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    norms = np.linalg.norm(block, axis=0)
    denominator = norms[:, None] * norms[None, :]
    cosine = np.divide(
        block.T @ block,
        denominator,
        out=np.zeros((block.shape[1], block.shape[1]), dtype=np.float64),
        where=denominator > 0,
    )
    cosine = np.clip(cosine, -1.0, 1.0)
    return cosine, norms


def support_overlap_columns(
    support: np.ndarray, i: np.ndarray, j: np.ndarray, prefix: str
) -> dict[str, np.ndarray]:
    counts = support.sum(axis=0).astype(np.int64)
    support_int = support.astype(np.int32, copy=False)
    intersection_matrix = support_int.T @ support_int
    intersection = intersection_matrix[i, j]
    union = counts[i] + counts[j] - intersection
    jaccard = np.divide(
        intersection,
        union,
        out=np.zeros_like(intersection, dtype=np.float64),
        where=union > 0,
    )
    return {
        f"{prefix}_support_i": counts[i],
        f"{prefix}_support_j": counts[j],
        f"{prefix}_support_intersection": intersection,
        f"{prefix}_support_union": union,
        f"{prefix}_support_jaccard": jaccard,
    }


def geometry_tables(
    A: np.ndarray, mz: np.ndarray, metadata: dict, cone: np.ndarray
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    parent_mask = (mz >= PARENT_MZ_RANGE[0]) & (mz <= PARENT_MZ_RANGE[1])
    fragment_mask = ~parent_mask
    parent_cosine, parent_norm = cosine_matrix(A[parent_mask])
    fragment_cosine, fragment_norm = cosine_matrix(A[fragment_mask])
    full_cosine, full_norm = cosine_matrix(A)

    count = A.shape[1]
    i, j = np.triu_indices(count, k=1)
    pair = pd.DataFrame({
        "i": i,
        "j": j,
        "lipid_i": np.asarray(metadata["lipid_name"], dtype=str)[i],
        "lipid_j": np.asarray(metadata["lipid_name"], dtype=str)[j],
        "parent_cosine": parent_cosine[i, j],
        "fragment_cosine": fragment_cosine[i, j],
        "full_cosine": full_cosine[i, j],
    })
    parent_support = np.abs(A[parent_mask]) > SUPPORT_THRESHOLD
    fragment_support = np.abs(A[fragment_mask]) > SUPPORT_THRESHOLD
    full_support = np.abs(A) > SUPPORT_THRESHOLD
    for support, prefix in (
        (parent_support, "parent"),
        (fragment_support, "fragment"),
        (full_support, "full"),
    ):
        for name, values in support_overlap_columns(support, i, j, prefix).items():
            pair[name] = values

    def maxima(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        work = matrix.copy()
        np.fill_diagonal(work, -np.inf)
        partner = np.argmax(work, axis=1)
        return work[np.arange(count), partner], partner

    max_parent, argmax_parent = maxima(parent_cosine)
    max_fragment, argmax_fragment = maxima(fragment_cosine)
    max_full, argmax_full = maxima(full_cosine)
    candidate = pd.DataFrame({
        "candidate_index": np.arange(count),
        "candidate_id": np.asarray(metadata["candidate_id"], dtype=str),
        "lipid_name": np.asarray(metadata["lipid_name"], dtype=str),
        "lipid_class": np.asarray(metadata["lipid_class"], dtype=str),
        "cone_isolation": cone,
        "max_parent_cosine": max_parent,
        "max_fragment_cosine": max_fragment,
        "max_full_cosine": max_full,
        "argmax_parent_partner": argmax_parent,
        "argmax_fragment_partner": argmax_fragment,
        "argmax_full_partner": argmax_full,
        "parent_support_count": parent_support.sum(axis=0),
        "fragment_support_count": fragment_support.sum(axis=0),
        "full_support_count": full_support.sum(axis=0),
    })
    diagnostics = {
        "parent_channel_count": int(parent_mask.sum()),
        "fragment_channel_count": int(fragment_mask.sum()),
        "zero_parent_norm_candidates": int(np.sum(parent_norm == 0)),
        "zero_fragment_norm_candidates": int(np.sum(fragment_norm == 0)),
        "zero_full_norm_candidates": int(np.sum(full_norm == 0)),
        "zero_norm_cosine_convention": 0.0,
    }
    return candidate, pair, diagnostics


def distribution(values: pd.Series) -> dict:
    quantiles = values.quantile([0.0, 0.01, 0.10, 0.25, 0.50, 0.75, 0.90, 0.99, 1.0])
    return {
        "count": int(values.size),
        "mean": float(values.mean()),
        "quantiles": {f"q{int(q * 100):02d}": float(value) for q, value in quantiles.items()},
    }


def descriptive_pair_bins(pair: pd.DataFrame) -> tuple[dict, dict[str, pd.Series]]:
    p_low, p_high = pair["parent_cosine"].quantile([0.10, 0.90])
    f_low, f_high = pair["fragment_cosine"].quantile([0.10, 0.90])
    masks = {
        "low_parent_high_fragment": (pair.parent_cosine <= p_low) & (pair.fragment_cosine >= f_high),
        "high_parent_low_fragment": (pair.parent_cosine >= p_high) & (pair.fragment_cosine <= f_low),
        "high_both": (pair.parent_cosine >= p_high) & (pair.fragment_cosine >= f_high),
        "low_both": (pair.parent_cosine <= p_low) & (pair.fragment_cosine <= f_low),
    }
    thresholds = {
        "definition": "descriptive bottom/top deciles of all frozen-library pairs; not confidence thresholds",
        "parent_bottom_decile_max": float(p_low),
        "parent_top_decile_min": float(p_high),
        "fragment_bottom_decile_max": float(f_low),
        "fragment_top_decile_min": float(f_high),
    }
    return thresholds, masks


def pair_example_rows(
    pair: pd.DataFrame,
    masks: dict[str, pd.Series],
    cone: np.ndarray,
    metadata: dict,
) -> list[dict]:
    specifications = {
        "A_easy_control": ("low_both", ["parent_cosine", "fragment_cosine", "full_cosine", "i", "j"], [True] * 5),
        "B_parent_overlap_fragment_distinct": ("high_parent_low_fragment", ["parent_cosine", "fragment_cosine", "full_cosine", "i", "j"], [False, True, False, True, True]),
        "C_parent_distinct_fragment_overlap": ("low_parent_high_fragment", ["fragment_cosine", "parent_cosine", "full_cosine", "i", "j"], [False, True, False, True, True]),
        "D_parent_and_fragment_overlap": ("high_both", ["full_cosine", "parent_cosine", "fragment_cosine", "i", "j"], [False, False, False, True, True]),
    }
    rows = []
    names = np.asarray(metadata["lipid_name"], dtype=str)
    classes = np.asarray(metadata["lipid_class"], dtype=str)
    for mechanism, (bin_name, columns, ascending) in specifications.items():
        subset = pair.loc[masks[bin_name]].sort_values(
            columns, ascending=ascending, kind="mergesort"
        ).head(N_MECHANISM_EXAMPLES)
        for rank, row in enumerate(subset.itertuples(index=False), start=1):
            left, right = int(row.i), int(row.j)
            rows.append({
                "mechanism": mechanism,
                "example_rank": rank,
                "candidate_indices": json.dumps([left, right]),
                "lipid_names": json.dumps([names[left], names[right]], ensure_ascii=False),
                "lipid_classes": json.dumps([classes[left], classes[right]], ensure_ascii=False),
                "parent_cosine": float(row.parent_cosine),
                "fragment_cosine": float(row.fragment_cosine),
                "full_cosine": float(row.full_cosine),
                "cone_isolation_values": json.dumps([float(cone[left]), float(cone[right])]),
                "collective_target_index": "",
                "context_partner_indices": "",
                "top_nonnegative_combination_contributors": "",
                "selection_note": f"representative from descriptive pair bin {bin_name}",
            })
    return rows


def collective_example_rows(
    candidate: pd.DataFrame,
    pair: pd.DataFrame,
    cone_cache: dict,
) -> tuple[list[dict], dict]:
    cone_q10 = float(candidate.cone_isolation.quantile(0.10))
    max_full_q50 = float(candidate.max_full_cosine.quantile(0.50))
    eligible = candidate.loc[
        (candidate.cone_isolation <= cone_q10)
        & (candidate.max_full_cosine <= max_full_q50)
    ]
    rule = "cone bottom decile and candidate maximum full cosine at/below candidate median"
    if eligible.empty:
        cone_q25 = float(candidate.cone_isolation.quantile(0.25))
        max_full_q75 = float(candidate.max_full_cosine.quantile(0.75))
        eligible = candidate.loc[
            (candidate.cone_isolation <= cone_q25)
            & (candidate.max_full_cosine <= max_full_q75)
        ]
        rule = "fallback descriptive stratum: cone bottom quartile and maximum full cosine at/below candidate q75"
    selected = eligible.sort_values(
        ["cone_isolation", "max_full_cosine", "candidate_index"],
        kind="mergesort",
    ).head(N_MECHANISM_EXAMPLES)

    rows = []
    for rank, target in enumerate(selected.itertuples(index=False), start=1):
        index = int(target.candidate_index)
        incident = pair.loc[(pair.i == index) | (pair.j == index)].copy()
        incident["partner"] = np.where(incident.i == index, incident.j, incident.i)
        context = incident.sort_values(
            ["full_cosine", "partner"], ascending=[False, True], kind="mergesort"
        ).head(2)
        partners = context.partner.astype(int).tolist()
        context_parent = context.parent_cosine.astype(float).tolist()
        context_fragment = context.fragment_cosine.astype(float).tolist()
        context_full = context.full_cosine.astype(float).tolist()
        contributor_data = cone_cache["contributors"].get(index)
        rows.append({
            "mechanism": "E_collective_cone_ambiguity",
            "example_rank": rank,
            "candidate_indices": json.dumps([index, *partners]),
            "lipid_names": json.dumps(
                [target.lipid_name, *candidate.set_index("candidate_index").loc[partners, "lipid_name"].tolist()],
                ensure_ascii=False,
            ),
            "lipid_classes": json.dumps(
                [target.lipid_class, *candidate.set_index("candidate_index").loc[partners, "lipid_class"].tolist()],
                ensure_ascii=False,
            ),
            "parent_cosine": json.dumps(context_parent),
            "fragment_cosine": json.dumps(context_fragment),
            "full_cosine": json.dumps(context_full),
            "cone_isolation_values": json.dumps(
                [float(target.cone_isolation), *candidate.set_index("candidate_index").loc[partners, "cone_isolation"].astype(float).tolist()]
            ),
            "collective_target_index": index,
            "context_partner_indices": json.dumps(partners),
            "top_nonnegative_combination_contributors": (
                json.dumps(contributor_data, ensure_ascii=False) if contributor_data else ""
            ),
            "selection_note": (
                rule
                + "; context partners are the two largest pairwise full cosines and are not claimed as cone-NNLS contributors"
            ),
        })
    return rows, {
        "selection_rule": rule,
        "cone_bottom_decile_max": cone_q10,
        "candidate_max_full_cosine_median": max_full_q50,
        "selected_count": len(rows),
        "cached_contributors_available": cone_cache["cached_contributor_solutions"],
    }


def run_audit(
    A: np.ndarray,
    mz: np.ndarray,
    metadata: dict,
    cone: np.ndarray,
    cone_cache: dict,
    output_dir: Path,
    validation: dict,
) -> None:
    candidate, pair, diagnostics = geometry_tables(A, mz, metadata, cone)
    thresholds, masks = descriptive_pair_bins(pair)
    mechanism_rows = pair_example_rows(pair, masks, cone, metadata)
    collective_rows, collective_summary = collective_example_rows(
        candidate, pair, cone_cache
    )
    mechanism = pd.DataFrame(mechanism_rows + collective_rows)

    output_dir.mkdir(parents=True, exist_ok=True)
    candidate.to_csv(output_dir / "candidate_geometry.csv", index=False)
    pair.to_csv(output_dir / "pair_geometry.csv", index=False)
    mechanism.to_csv(output_dir / "mechanism_examples.csv", index=False)

    pair_count = len(pair)
    category_counts = {
        name: {
            "count": int(mask.sum()),
            "fraction": float(mask.mean()),
        }
        for name, mask in masks.items()
    }
    summary = {
        "status": "COMPLETE",
        "scientific_scope": "descriptive frozen-library competition geometry; no confidence thresholds",
        "candidate_count": int(A.shape[1]),
        "pair_count": pair_count,
        "parent_region_mz_inclusive": list(PARENT_MZ_RANGE),
        "fragment_region_definition": "all channels outside the inclusive parent region",
        "support_nonzero_threshold": SUPPORT_THRESHOLD,
        "support_threshold_provenance": "existing project convention: abs(A) > 1e-8",
        "cosine_distributions": {
            metric: distribution(pair[metric])
            for metric in ("parent_cosine", "fragment_cosine", "full_cosine")
        },
        "descriptive_percentile_bins": thresholds,
        "pair_bin_counts": category_counts,
        "collective_ambiguity_examples": collective_summary,
        "geometry_diagnostics": diagnostics,
        "cone_cache": {key: value for key, value in cone_cache.items() if key != "contributors"},
        "asset_validation": validation,
        "outputs": [
            "candidate_geometry.csv",
            "pair_geometry.csv",
            "mechanism_examples.csv",
            "summary.json",
        ],
    }
    atomic_write_json(output_dir / "summary.json", summary)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--asset-root",
        type=Path,
        default=ROOT.parent / "decon-lipid",
        help="Root containing the frozen v38 production assets",
    )
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DEFAULT)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate frozen inputs and cone cache without calculating pair geometry or writing outputs",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = asset_paths(args.asset_root.resolve())
    validation = validate_assets(paths)
    A, mz, metadata = load_inputs(paths)
    cone, cone_cache = load_cached_cone_isolation(A.shape[1])
    parent_mask = (mz >= PARENT_MZ_RANGE[0]) & (mz <= PARENT_MZ_RANGE[1])
    if args.dry_run:
        print(json.dumps({
            "status": "DRY_RUN_READY",
            "candidate_count": int(A.shape[1]),
            "channel_count": int(A.shape[0]),
            "expected_pair_count": int(A.shape[1] * (A.shape[1] - 1) // 2),
            "parent_region_mz_inclusive": list(PARENT_MZ_RANGE),
            "parent_channel_count": int(parent_mask.sum()),
            "fragment_channel_count": int((~parent_mask).sum()),
            "support_nonzero_threshold": SUPPORT_THRESHOLD,
            "cone_cache": {key: value for key, value in cone_cache.items() if key != "contributors"},
            "asset_validation": validation,
            "full_pair_audit_performed": False,
            "outputs_written": False,
        }, ensure_ascii=False, indent=2))
        return
    run_audit(
        A, mz, metadata, cone, cone_cache, args.output_dir.resolve(), validation
    )


if __name__ == "__main__":
    main()
