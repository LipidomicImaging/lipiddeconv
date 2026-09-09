#!/usr/bin/env python3
"""Characterize the empirical domain of locked production B and X_hat.

This CPU-only Stage 0 is descriptive.  Reported active counts are solver
outputs and must never be interpreted as ground-truth molecular K.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
ANALYSIS = ROOT / "analysis"
if str(ANALYSIS) not in sys.path:
    sys.path.insert(0, str(ANALYSIS))

import run_v50_reoptimized_sanity as v50
import run_v52_mechanism_leakage as v52


VERSION = "v53_stage0_empirical_domain"
OUTPUT_DEFAULT = ROOT / "results/v53_stage0_empirical_domain"
LOCK_PATH = ROOT / "results/v48_production_ista_lock/v48_production_ista_lock.json"
EXPERIMENT_LOG_PATH = ROOT / "results/EXPERIMENT_LOG.md"
PRODUCTION_X_FILENAME = "X_abundance.npy"
EXPECTED_A_SOLVER_SHAPE = (1084, 391)
PARENT_MZ_RANGE = (748.0, 803.0)
REPORT_GATE = 1.0e-3
SENSITIVITY_GATE = 1.0e-4
EPSILON = 1.0e-12
QUANTILES = (0.01, 0.05, 0.10, 0.20, 0.25, 0.35, 0.50, 0.65, 0.75, 0.80, 0.90, 0.95, 0.99)
REPRESENTATIVE_QUANTILES = (0.10, 0.25, 0.50, 0.75, 0.90)
RESIDUAL_BLOCK_PIXELS = 512
LOG_BEGIN = "<!-- BEGIN v53_stage0_empirical_domain -->"
LOG_END = "<!-- END v53_stage0_empirical_domain -->"


def to_jsonable(value: Any) -> Any:
    if isinstance(value, np.generic):
        return to_jsonable(value.item())
    if isinstance(value, np.ndarray):
        return to_jsonable(value.tolist())
    if isinstance(value, dict):
        return {key: to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def atomic_write_json(path: Path, payload: dict) -> None:
    atomic_write_text(
        path,
        json.dumps(
            to_jsonable(payload), ensure_ascii=False, indent=2, allow_nan=False
        ) + "\n",
    )


def load_lock() -> dict:
    if not LOCK_PATH.exists():
        raise RuntimeError(f"MISSING_DEPENDENCY: production lock not found: {LOCK_PATH}")
    lock = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    if lock.get("status") != "PASS_PRODUCTION_ISTA_LOCKED":
        raise RuntimeError("STAGE0_LOCK_MISMATCH: production lock status")
    return lock


def resolve_locked_paths(asset_root: Path, lock: dict) -> dict[str, Path]:
    canonical = lock["canonical_run"]
    paths = v50.asset_paths(asset_root)
    expected_relative = {
        "A_library": canonical["A_library"],
        "B_cube": canonical["B_input"],
        "candidate_metadata": canonical["candidate_metadata"],
        "channel_axis": canonical["channel_axis"],
    }
    for name, relative in expected_relative.items():
        expected_path = (asset_root / relative).resolve()
        if paths[name].resolve() != expected_path:
            raise RuntimeError(f"STAGE0_LOCK_MISMATCH: resolved {name} path")
    checkpoint = (asset_root / canonical["checkpoint"]).resolve()
    paths.update({
        "production_checkpoint": checkpoint,
        "production_X_hat": checkpoint.parent / PRODUCTION_X_FILENAME,
    })
    return paths


def missing_paths(paths: dict[str, Path]) -> list[str]:
    required = (
        "A_library", "B_cube", "foreground_mask", "candidate_metadata",
        "channel_axis", "production_X_hat",
    )
    return [str(paths[name]) for name in required if not paths[name].exists()]


def validate_shapes_and_hashes(
    paths: dict[str, Path], lock: dict
) -> tuple[dict, dict]:
    validation = v50.validate_assets({
        name: paths[name]
        for name in (
            "A_library", "B_cube", "foreground_mask", "candidate_metadata",
            "channel_axis",
        )
    })
    expected_hashes = lock["hashes_sha256"]
    X_hash = v50.sha256(paths["production_X_hat"])
    X_hash_role = next(
        (
            role for role in ("reference_X", "reproduced_X")
            if X_hash == expected_hashes.get(role)
        ),
        None,
    )
    if X_hash_role is None:
        raise RuntimeError(
            "STAGE0_LOCK_MISMATCH: production X_hat hash matches neither "
            "locked reference_X nor reproduced_X"
        )

    A_stored = np.load(paths["A_library"], mmap_mode="r")
    B_stored = np.load(paths["B_cube"], mmap_mode="r")
    X_stored = np.load(paths["production_X_hat"], mmap_mode="r")
    mask_stored = np.load(paths["foreground_mask"], mmap_mode="r")
    mz_stored = np.load(paths["channel_axis"], mmap_mode="r")
    locked_preprocessing = lock["production_input_and_preprocessing"]
    expected_B_stored = tuple(locked_preprocessing["B_stored_shape"])
    expected_X = tuple(lock["reproduction_gate"]["reference_X_shape"])
    if tuple(B_stored.shape) != expected_B_stored:
        raise RuntimeError(
            f"DIMENSION_MISMATCH: B stored {tuple(B_stored.shape)} != {expected_B_stored}"
        )
    if tuple(X_stored.shape) != expected_X:
        raise RuntimeError(
            f"DIMENSION_MISMATCH: X_hat {tuple(X_stored.shape)} != {expected_X}"
        )
    height, width = expected_X[1:]
    if tuple(mask_stored.shape) != (height, width):
        raise RuntimeError(
            f"DIMENSION_MISMATCH: foreground mask {tuple(mask_stored.shape)}"
        )
    if int(np.asarray(mask_stored, dtype=bool).sum()) != int(
        locked_preprocessing["B_normalization"]["foreground_pixel_count"]
    ):
        raise RuntimeError("FOREGROUND_MISMATCH: locked foreground pixel count")
    if int(np.asarray(mz_stored).reshape(-1).size) != EXPECTED_A_SOLVER_SHAPE[0]:
        raise RuntimeError("DIMENSION_MISMATCH: channel axis")

    A_raw, _ = v50.load_library_and_metadata(paths)
    A_solver = v52.normalize_like_get_A_matrix(A_raw)
    if tuple(A_solver.shape) != EXPECTED_A_SOLVER_SHAPE:
        raise RuntimeError(
            f"DIMENSION_MISMATCH: A_solver {tuple(A_solver.shape)}"
        )
    if A_solver.dtype != np.float32 or not np.isfinite(A_solver).all():
        raise RuntimeError("A_SOLVER_MISMATCH: expected finite float32 A_solver")

    available_hashes = {
        **validation["validated_hashes"],
        "production_X_hat": X_hash,
        "foreground_mask": v50.sha256(paths["foreground_mask"]),
    }
    checkpoint = paths["production_checkpoint"]
    if checkpoint.exists():
        checkpoint_hash = v50.sha256(checkpoint)
        if checkpoint_hash != expected_hashes["production_checkpoint"]:
            raise RuntimeError("STAGE0_LOCK_MISMATCH: production checkpoint hash")
        available_hashes["production_checkpoint"] = checkpoint_hash
    provenance = {
        "production_X_hash_role": X_hash_role,
        "production_X_path_resolution": (
            "canonical checkpoint parent plus X_abundance.npy, exactly as "
            "src/run_758_ista.py export_results"
        ),
        "foreground_runtime_path": str(paths["foreground_mask"]),
        "foreground_construction_source_from_lock": locked_preprocessing[
            "B_normalization"
        ]["foreground_mask_source"],
    }
    return {"hashes_sha256": available_hashes, **validation}, provenance


def distribution(values: np.ndarray, include_count: bool = True) -> dict:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    values = values[np.isfinite(values)]
    if values.size == 0:
        result = {
            "min": None, "max": None, "mean": None, "median": None,
            "std": None,
            "quantiles": {f"q{int(q * 100):02d}": None for q in QUANTILES},
        }
        if include_count:
            result["count"] = 0
        return result
    result = {
        "min": float(values.min()),
        "max": float(values.max()),
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "std": float(values.std(ddof=0)),
        "quantiles": {
            f"q{int(q * 100):02d}": float(np.quantile(values, q))
            for q in QUANTILES
        },
    }
    if include_count:
        result["count"] = int(values.size)
    return result


def flatten_distribution_rows(
    distributions: dict[str, dict]
) -> list[dict]:
    rows = []
    ordered = ("count", "min", "q01", "q05", "q10", "q20", "q25", "q35", "q50", "q65", "q75", "q80", "q90", "q95", "q99", "max", "mean", "median", "std")
    for name, summary in distributions.items():
        values = {**summary, **summary.get("quantiles", {})}
        for statistic in ordered:
            if statistic in values:
                rows.append({
                    "distribution": name,
                    "statistic": statistic,
                    "value": values[statistic],
                })
    return rows


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def residual_statistics(
    A_solver: np.ndarray,
    B_real: np.ndarray,
    X_hat: np.ndarray,
    mz: np.ndarray,
) -> dict:
    channels, height, width = B_real.shape
    pixels = height * width
    B_flat = B_real.reshape(channels, pixels)
    X_flat = X_hat.reshape(X_hat.shape[0], pixels)
    parent = (mz >= PARENT_MZ_RANGE[0]) & (mz <= PARENT_MZ_RANGE[1])
    fragment = ~parent

    signal_norm = np.empty(pixels, dtype=np.float64)
    residual_norm = np.empty(pixels, dtype=np.float64)
    parent_ratio = np.empty(pixels, dtype=np.float64)
    fragment_ratio = np.empty(pixels, dtype=np.float64)
    B_energy = residual_energy = 0.0
    parent_B_energy = parent_residual_energy = 0.0
    fragment_B_energy = fragment_residual_energy = 0.0

    for start in range(0, pixels, RESIDUAL_BLOCK_PIXELS):
        stop = min(start + RESIDUAL_BLOCK_PIXELS, pixels)
        B_block = np.asarray(B_flat[:, start:stop], dtype=np.float32)
        B_hat_block = A_solver @ np.asarray(X_flat[:, start:stop], dtype=np.float32)
        residual = B_block - B_hat_block
        B_parent, R_parent = B_block[parent], residual[parent]
        B_fragment, R_fragment = B_block[fragment], residual[fragment]

        signal_norm[start:stop] = np.linalg.norm(B_block, axis=0)
        residual_norm[start:stop] = np.linalg.norm(residual, axis=0)
        parent_ratio[start:stop] = np.linalg.norm(R_parent, axis=0) / (
            np.linalg.norm(B_parent, axis=0) + EPSILON
        )
        fragment_ratio[start:stop] = np.linalg.norm(R_fragment, axis=0) / (
            np.linalg.norm(B_fragment, axis=0) + EPSILON
        )
        B_energy += float(np.sum(B_block.astype(np.float64) ** 2))
        residual_energy += float(np.sum(residual.astype(np.float64) ** 2))
        parent_B_energy += float(np.sum(B_parent.astype(np.float64) ** 2))
        parent_residual_energy += float(np.sum(R_parent.astype(np.float64) ** 2))
        fragment_B_energy += float(np.sum(B_fragment.astype(np.float64) ** 2))
        fragment_residual_energy += float(np.sum(R_fragment.astype(np.float64) ** 2))

    return {
        "signal_norm": signal_norm,
        "residual_norm": residual_norm,
        "residual_to_signal": residual_norm / (signal_norm + EPSILON),
        "parent_residual_to_signal": parent_ratio,
        "fragment_residual_to_signal": fragment_ratio,
        "global_relative_residual": float(
            np.sqrt(residual_energy) / (np.sqrt(B_energy) + EPSILON)
        ),
        "global_parent_residual_relative_l2": float(
            np.sqrt(parent_residual_energy) / (np.sqrt(parent_B_energy) + EPSILON)
        ),
        "global_fragment_residual_relative_l2": float(
            np.sqrt(fragment_residual_energy) / (np.sqrt(fragment_B_energy) + EPSILON)
        ),
        "residual_energy_fraction_parent": float(
            parent_residual_energy / max(residual_energy, EPSILON)
        ),
        "residual_energy_fraction_fragment": float(
            fragment_residual_energy / max(residual_energy, EPSILON)
        ),
        "parent_channel_count": int(parent.sum()),
        "fragment_channel_count": int(fragment.sum()),
    }


def representative_pixels(
    mask: np.ndarray, residual: dict
) -> list[dict]:
    flat_foreground = np.flatnonzero(mask.reshape(-1))
    ratios = residual["residual_to_signal"][flat_foreground]
    width = mask.shape[1]
    rows = []
    for quantile in REPRESENTATIVE_QUANTILES:
        target = float(np.quantile(ratios, quantile))
        local_index = int(np.argmin(np.abs(ratios - target)))
        flat_index = int(flat_foreground[local_index])
        rows.append({
            "quantile_target": f"q{int(quantile * 100):02d}",
            "pixel_flat_index": flat_index,
            "y": flat_index // width,
            "x": flat_index % width,
            "signal_norm": float(residual["signal_norm"][flat_index]),
            "residual_norm": float(residual["residual_norm"][flat_index]),
            "residual_to_signal": float(residual["residual_to_signal"][flat_index]),
            "parent_residual_to_signal": float(residual["parent_residual_to_signal"][flat_index]),
            "fragment_residual_to_signal": float(residual["fragment_residual_to_signal"][flat_index]),
        })
    return rows


def format_value(value: Any) -> str:
    if value is None:
        return "unavailable"
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.8g}"
    return str(value)


def q(summary: dict, name: str) -> Any:
    return summary["quantiles"].get(name)


def build_summary_md(domain: dict, design: dict) -> str:
    active = domain["reported_active_count"]
    abundance = domain["production_Xhat_abundance_distribution"]
    residual = domain["production_residual_distribution"]
    hashes = design["input_hashes_sha256"]
    paths = design["input_paths"]
    lines = [
        "# v53 Stage 0 — Empirical Domain Characterization",
        "",
        "## Purpose",
        "",
        "Describe the reported-complexity, abundance, and residual domain of the locked real production deconvolution without asserting molecular truth.",
        "",
        "## Locked inputs and hashes",
        "",
    ]
    for name in ("A_library", "B_cube", "production_X_hat", "foreground_mask", "channel_axis"):
        lines.append(f"- {name}: `{paths[name]}` — `{hashes.get(name, 'unavailable')}`")
    lines += [
        "",
        "## Definitions",
        "",
        "`reported_active_count` is the number of production X_hat candidate outputs above a reporting gate at a foreground pixel. It is a reported-complexity proxy. The true molecular K is unknown; reported active-count is not ground-truth K.",
        "",
        "## Results",
        "",
    ]
    for gate_name in ("pixel_reported_active_count_gt_1e-3", "pixel_reported_active_count_gt_1e-4"):
        item = active[gate_name]
        lines.append(
            f"- {gate_name}: q20={format_value(q(item, 'q20'))}, q50={format_value(q(item, 'q50'))}, q80={format_value(q(item, 'q80'))}; range={format_value(item['min'])}–{format_value(item['max'])}."
        )
    candidate_abundance = abundance["candidate_foreground_mean_reported_gt_1e-3"]
    pixel_abundance = abundance["pixel_reported_abundance_gt_1e-3"]
    ratio = residual["foreground_residual_to_signal"]
    lines += [
        f"- Candidate-level foreground-mean abundance: q20={format_value(q(candidate_abundance, 'q20'))}, q50={format_value(q(candidate_abundance, 'q50'))}, q80={format_value(q(candidate_abundance, 'q80'))}.",
        f"- Pixel-level reported abundance: q20={format_value(q(pixel_abundance, 'q20'))}, q50={format_value(q(pixel_abundance, 'q50'))}, q80={format_value(q(pixel_abundance, 'q80'))}.",
        f"- Global relative residual: {format_value(residual['global_relative_residual'])}.",
        f"- Foreground residual-to-signal: q20={format_value(q(ratio, 'q20'))}, q50={format_value(q(ratio, 'q50'))}, q80={format_value(q(ratio, 'q80'))}.",
        f"- Parent vs fragment: global relative L2={format_value(residual['global_parent_residual_relative_l2'])} vs {format_value(residual['global_fragment_residual_relative_l2'])}; residual-energy fractions={format_value(residual['residual_energy_fraction_parent'])} vs {format_value(residual['residual_energy_fraction_fragment'])}.",
        "",
        "## Interpretation",
        "",
        "The measurements describe the empirical range produced by the locked solver on the real production dataset. They do not select synthetic complexity, abundance ratios, residual stress levels, or a rho_zero threshold.",
        "",
        "## Limitations",
        "",
        "1. X_hat-derived active counts are solver outputs, not true molecular counts.",
        "2. The X_hat abundance distribution is estimator-dependent.",
        "3. R_real contains experimental mismatch, library mismatch, and solver residual; it is not pure measurement noise.",
        "4. This Stage 0 contains no ground-truth identity validation.",
        "",
        "## Validity status",
        "",
        "VALID_MAINLINE",
        "",
        "Scope: empirical-domain characterization only.",
        "",
        "## Decision",
        "",
        "The next experiment may use these empirical distributions to freeze the synthetic calibration design, but no calibration threshold is determined here.",
        "",
    ]
    return "\n".join(lines)


def experiment_log_entry(domain: dict, design: dict) -> str:
    active = domain["reported_active_count"]["pixel_reported_active_count_gt_1e-3"]
    abundance = domain["production_Xhat_abundance_distribution"]["pixel_reported_abundance_gt_1e-3"]
    residual = domain["production_residual_distribution"]
    return "\n".join([
        LOG_BEGIN,
        "## v53 Stage 0 — Empirical Domain Characterization",
        "",
        f"Version: {VERSION}",
        "",
        f"Date: {design['timestamp_utc']}",
        "",
        "Purpose: Characterize the locked real production reported-complexity, abundance, and residual domain before ground-truth calibration design.",
        "",
        "Inputs / hashes:",
        *[f"- {name}: `{path}` — `{design['input_hashes_sha256'].get(name, 'unavailable')}`" for name, path in design["input_paths"].items()],
        "",
        "Experimental design: CPU-only descriptive audit of foreground production X_hat and residual R_real = B_real - A_solver @ X_hat; no training or ground truth.",
        "",
        f"Key results: reported-active-count q20/q50/q80 = {format_value(q(active, 'q20'))}/{format_value(q(active, 'q50'))}/{format_value(q(active, 'q80'))}; pixel abundance q20/q50/q80 = {format_value(q(abundance, 'q20'))}/{format_value(q(abundance, 'q50'))}/{format_value(q(abundance, 'q80'))}; global residual = {format_value(residual['global_relative_residual'])}.",
        "",
        "Conclusion: The empirical production domain was measured without interpreting reported active count as true K.",
        "",
        "Validity status: VALID_MAINLINE",
        "",
        "Limitations: X_hat counts and abundances are estimator-dependent; R_real mixes experimental, library, and solver mismatch; no identity truth is available.",
        "",
        "Decision: These distributions may inform a later frozen synthetic design; no calibration threshold is determined here.",
        "",
        "Next step: Freeze a separate ground-truth calibration design using this empirical-domain record.",
        "",
        "Git commit: PENDING_USER_COMMIT",
        LOG_END,
    ])


def update_experiment_log(entry: str) -> None:
    if EXPERIMENT_LOG_PATH.exists():
        current = EXPERIMENT_LOG_PATH.read_text(encoding="utf-8")
    else:
        current = "# Experiment Log\n"
    pattern = re.compile(
        re.escape(LOG_BEGIN) + r".*?" + re.escape(LOG_END), re.DOTALL
    )
    if pattern.search(current):
        updated = pattern.sub(entry, current, count=1)
    else:
        updated = current.rstrip() + "\n\n" + entry + "\n"
    atomic_write_text(EXPERIMENT_LOG_PATH, updated)


def full_audit(
    paths: dict[str, Path], lock: dict, validation: dict, provenance: dict,
    output_dir: Path,
) -> None:
    A_raw, _ = v50.load_library_and_metadata(paths)
    A_solver = v52.normalize_like_get_A_matrix(A_raw)
    B_stored = np.load(paths["B_cube"])
    X_hat = np.asarray(np.load(paths["production_X_hat"]), dtype=np.float32)
    mask = np.asarray(np.load(paths["foreground_mask"]), dtype=bool)
    mz = np.asarray(np.load(paths["channel_axis"]), dtype=np.float64).reshape(-1)

    expected_B = tuple(lock["production_input_and_preprocessing"]["B_stored_shape"])
    if tuple(B_stored.shape) != expected_B:
        raise RuntimeError("DIMENSION_MISMATCH: B changed after validation")
    B_real = np.moveaxis(np.asarray(B_stored[0], dtype=np.float32), -1, 0)
    if B_real.shape != (A_solver.shape[0], *X_hat.shape[1:]):
        raise RuntimeError(
            f"DIMENSION_MISMATCH: A={A_solver.shape}, B={B_real.shape}, X={X_hat.shape}"
        )

    X_foreground = X_hat[:, mask]
    active_1e3 = np.sum(X_foreground > REPORT_GATE, axis=0)
    active_1e4 = np.sum(X_foreground > SENSITIVITY_GATE, axis=0)
    xbar = X_foreground.mean(axis=1, dtype=np.float64)
    candidate_reported = xbar[xbar > REPORT_GATE]
    pixel_reported = X_foreground[X_foreground > REPORT_GATE].astype(np.float64)
    active_distributions = {
        "pixel_reported_active_count_gt_1e-3": distribution(active_1e3),
        "pixel_reported_active_count_gt_1e-4": distribution(active_1e4),
    }
    abundance_distributions = {
        "candidate_foreground_mean_reported_gt_1e-3": distribution(candidate_reported),
        "candidate_foreground_mean_reported_gt_1e-3_log10": distribution(np.log10(candidate_reported[candidate_reported > 0])),
        "pixel_reported_abundance_gt_1e-3": distribution(pixel_reported),
        "pixel_reported_abundance_gt_1e-3_log10": distribution(np.log10(pixel_reported[pixel_reported > 0])),
    }
    residual = residual_statistics(A_solver, B_real, X_hat, mz)
    foreground_flat = mask.reshape(-1)
    residual_distributions = {
        "foreground_signal_norm": distribution(residual["signal_norm"][foreground_flat]),
        "foreground_residual_norm": distribution(residual["residual_norm"][foreground_flat]),
        "foreground_residual_to_signal": distribution(residual["residual_to_signal"][foreground_flat]),
        "foreground_parent_residual_to_signal": distribution(residual["parent_residual_to_signal"][foreground_flat]),
        "foreground_fragment_residual_to_signal": distribution(residual["fragment_residual_to_signal"][foreground_flat]),
    }
    representatives = representative_pixels(mask, residual)

    foreground_count = int(mask.sum())
    total_pixels = int(mask.size)
    residual_summary = {
        key: value for key, value in residual.items()
        if not isinstance(value, np.ndarray)
    }
    residual_summary.update(residual_distributions)
    domain = {
        "status": "VALID_MAINLINE",
        "scope": "empirical-domain characterization only",
        "reported_active_count_is_not_ground_truth_K": True,
        "foreground": {
            "foreground_pixel_count": foreground_count,
            "total_pixel_count": total_pixels,
            "foreground_fraction": foreground_count / total_pixels,
        },
        "reported_active_count": {
            **active_distributions,
            "foreground_mean_candidate_count_gt_1e-3": int(np.sum(xbar > REPORT_GATE)),
            "foreground_mean_candidate_count_gt_1e-4": int(np.sum(xbar > SENSITIVITY_GATE)),
        },
        "production_Xhat_abundance_distribution": abundance_distributions,
        "production_residual_distribution": residual_summary,
        "representative_residual_pixels": representatives,
        "interpretation_boundaries": {
            "synthetic_K_selected": False,
            "synthetic_abundance_ratios_selected": False,
            "residual_stress_levels_selected": False,
            "rho_zero_threshold_selected": False,
        },
    }
    timestamp = datetime.now(timezone.utc).isoformat()
    design = {
        "purpose": "v53 Stage 0 empirical-domain characterization for the final molecular-identity confidence calibration mainline",
        "script_version": VERSION,
        "timestamp_utc": timestamp,
        "input_paths": {name: str(path) for name, path in paths.items()},
        "input_hashes_sha256": validation["hashes_sha256"],
        "input_provenance": provenance,
        "A_solver_normalization_contract": "A_raw float32; A_solver[:,j] = A_raw[:,j] / (||A_raw[:,j]||_2 + 1e-8), using v52.normalize_like_get_A_matrix",
        "shapes": {
            "A_solver": list(A_solver.shape),
            "B_real": list(B_real.shape),
            "X_hat": list(X_hat.shape),
            "foreground_mask": list(mask.shape),
        },
        "reporting_gates": {
            "primary": REPORT_GATE,
            "descriptive_sensitivity": SENSITIVITY_GATE,
        },
        "foreground_provenance": {
            "runtime_path": str(paths["foreground_mask"]),
            "locked_construction_source": provenance["foreground_construction_source_from_lock"],
            "reused_logic": "v50 asset path and mask; no new threshold",
            "foreground_pixel_count": foreground_count,
        },
        "parent_fragment_region_definition": {
            "parent_mz_inclusive": list(PARENT_MZ_RANGE),
            "fragment": "all remaining channels",
            "parent_channel_count": residual["parent_channel_count"],
            "fragment_channel_count": residual["fragment_channel_count"],
        },
        "spatial_indexing": {
            "compatible_with_y_x": True,
            "shape_y_x": list(mask.shape),
            "flat_index_order": "NumPy C order; y = flat_index // width, x = flat_index % width",
        },
        "explicit_statement": "reported active-count is not ground-truth K",
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output_dir / "design.json", design)
    atomic_write_json(output_dir / "domain_summary.json", domain)
    write_csv(
        output_dir / "active_count_quantiles.csv",
        flatten_distribution_rows(active_distributions),
        ["distribution", "statistic", "value"],
    )
    write_csv(
        output_dir / "abundance_quantiles.csv",
        flatten_distribution_rows(abundance_distributions),
        ["distribution", "statistic", "value"],
    )
    write_csv(
        output_dir / "residual_quantiles.csv",
        flatten_distribution_rows(residual_distributions),
        ["distribution", "statistic", "value"],
    )
    write_csv(
        output_dir / "representative_residual_pixels.csv",
        representatives,
        [
            "quantile_target", "pixel_flat_index", "y", "x", "signal_norm",
            "residual_norm", "residual_to_signal",
            "parent_residual_to_signal", "fragment_residual_to_signal",
        ],
    )
    summary_md = build_summary_md(domain, design)
    atomic_write_text(output_dir / "summary.md", summary_md)
    update_experiment_log(experiment_log_entry(domain, design))
    print(json.dumps({
        "status": "VALID_MAINLINE",
        "scope": "empirical-domain characterization only",
        "output_dir": str(output_dir),
        "reported_active_count_is_not_ground_truth_K": True,
        "experiment_log": str(EXPERIMENT_LOG_PATH),
    }, ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--asset-root", type=Path, default=ROOT.parent / "decon-lipid"
    )
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DEFAULT)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    asset_root = args.asset_root.resolve()
    lock = load_lock()
    paths = resolve_locked_paths(asset_root, lock)
    missing = missing_paths(paths)
    if missing:
        if args.dry_run:
            print(json.dumps({
                "status": "DRY_RUN_READY_REMOTE_ASSETS_REQUIRED",
                "asset_root": str(asset_root),
                "resolved_paths": {name: str(path) for name, path in paths.items()},
                "missing_required_paths": missing,
                "planned_calculations": [
                    "reported active-count distributions at 1e-3 and 1e-4",
                    "candidate-mean and pixel-level reported abundance distributions",
                    "global and per-pixel production residual distributions",
                    "parent/fragment residual structure",
                    "representative residual pixel indices",
                ],
                "reported_active_count_is_not_ground_truth_K": True,
                "heavy_array_calculation_performed": False,
                "outputs_written": False,
                "experiment_log_modified": False,
            }, ensure_ascii=False, indent=2))
            return
        raise RuntimeError(
            "MISSING_LOCKED_PRODUCTION_ASSETS: " + "; ".join(missing)
        )

    validation, provenance = validate_shapes_and_hashes(paths, lock)
    if args.dry_run:
        print(json.dumps({
            "status": "DRY_RUN_READY",
            "asset_root": str(asset_root),
            "resolved_paths": {name: str(path) for name, path in paths.items()},
            "validated_hashes_sha256": validation["hashes_sha256"],
            "provenance": provenance,
            "planned_calculations": [
                "reported active-count distributions at 1e-3 and 1e-4",
                "foreground-mean candidate abundance and pixel-level reported abundance",
                "log10 abundance distributions for positive reported values",
                "R_real = B_real - A_solver @ X_hat in bounded pixel blocks",
                "global, per-pixel, parent, and fragment residual summaries",
                "deterministic q10/q25/q50/q75/q90 residual pixel indices",
            ],
            "reported_active_count_is_not_ground_truth_K": True,
            "heavy_array_calculation_performed": False,
            "outputs_written": False,
            "experiment_log_modified": False,
        }, ensure_ascii=False, indent=2))
        return
    full_audit(paths, lock, validation, provenance, args.output_dir.resolve())


if __name__ == "__main__":
    main()
