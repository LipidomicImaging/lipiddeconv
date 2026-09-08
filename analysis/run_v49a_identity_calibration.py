#!/usr/bin/env python3
"""Prepare and, on the remote GPU host, execute the gated v49a calibration."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path, PurePosixPath

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
from rho_zero import identity_weights, rho_zero_from_weighted_case

K_VALUES = (13, 25, 55, 103)
RESIDUALS = (0.0, 0.1)
SELECTIONS = ("random", "hard_competition")
ABUNDANCES = ("lighter_tail", "central", "heavier_tail")
REPLICATES = (0, 1, 2, 3)
SEED_BASE = 20260908
PARITY_BOOTSTRAP_SEED = 20260909
PRIMARY_GATE = 1e-3
PARITY_PAIRS = (
    (7, 321, 1, "true_low"), (47, 355, 1, "true_low"),
    (4, 354, 1, "true_mid"), (10, 367, 1, "true_mid"),
    (29, 347, 1, "true_high"), (11, 352, 1, "true_high"),
    (0, 21, 0, "false_low"), (14, 271, 0, "false_low"),
    (28, 333, 0, "false_low"), (42, 189, 0, "false_low"),
    (4, 347, 0, "false_low"), (7, 166, 0, "false_high"),
)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def planned_cases(seed_base: int) -> list[dict]:
    rows = []
    case_id = 0
    for k in K_VALUES:
        for residual in RESIDUALS:
            for selection in SELECTIONS:
                for abundance in ABUNDANCES:
                    for replicate in REPLICATES:
                        rows.append({
                            "case_id": case_id, "K": k,
                            "residual_ratio": residual, "selection_mode": selection,
                            "abundance_mode": abundance, "replicate": replicate,
                            "split": "calibration" if replicate < 2 else "held_out_test",
                            "seed": seed_base + case_id,
                        })
                        case_id += 1
    return rows


def validate_plan(rows: list[dict]) -> None:
    assert len(rows) == 192
    assert sum(r["split"] == "calibration" for r in rows) == 96
    assert sum(r["split"] == "held_out_test" for r in rows) == 96
    assert len({r["case_id"] for r in rows}) == 192
    assert len({r["seed"] for r in rows}) == 192
    counts = {}
    for r in rows:
        key = (r["K"], r["residual_ratio"], r["selection_mode"], r["abundance_mode"])
        counts[key] = counts.get(key, 0) + 1
    assert set(counts.values()) == {4}


def parity_manifest() -> pd.DataFrame:
    cached = ROOT / "results/v48_pilot_48/v48_pilot_lipid_observations.csv"
    d = pd.read_csv(cached)
    rows = []
    for case_id, candidate_index, truth, label in PARITY_PAIRS:
        m = d[(d.case_id == case_id) & (d.candidate_index == candidate_index)]
        if len(m) != 1 or int(m.iloc[0].X_true > 0) != truth:
            raise RuntimeError("frozen parity pair no longer matches cached reference")
        rows.append({
            "pair_id": f"case_{case_id:03d}_candidate_{candidate_index:03d}",
            "case_id": case_id, "candidate_index": candidate_index,
            "truth_identity": truth, "rho_band": label,
            "cached_rho_zero": float(m.iloc[0].rho_zero),
        })
    return pd.DataFrame(rows)


def asset_paths(asset_root: Path) -> dict[str, PurePosixPath]:
    # The future runner is executed on Linux; keep dry-run output POSIX even
    # when this preparation check is invoked by a Windows interpreter.
    root = PurePosixPath(str(asset_root).replace("\\", "/"))
    data = root / "adapter_pipeline/outputs/v38_ce29_fragmentionfixed_profile_overlap_empiricalfwhm_globalq99_ista_ready"
    result = root / "results_758_v38_ce29_empiricalfwhm_globalq99_fixed_library_joint_earlystop"
    return {
        "A_library": data / "A_library.npy",
        "B_cube": data / "B_cube.npy",
        "candidate_metadata": data / "candidate_metadata_final.npy",
        "channel_axis": data / "shared_mz_final.npy",
        "foreground_mask": data / "foreground_pixel_mask.npy",
        "checkpoint": result / "latest_model.pth",
    }


def reconstruct_v48_certificate_b(
    manifest: dict, asset_root: Path, A: np.ndarray, empirical: np.ndarray
) -> np.ndarray:
    """Exact v48 attach_v47_certificates_v48_pilot.py reconstruction path."""
    truth_spec = (
        asset_root / "v48_pilot_48" / "cases" / manifest["case_name"] / "truth_spec.npz"
    )
    spec = np.load(truth_spec)
    selected = spec["selected_indices"].astype(int)
    true_values = spec["true_abundances"].astype(np.float32)
    clean = A[:, selected] @ true_values
    sources = spec["residual_source_indices"].astype(int)
    if float(manifest["residual_ratio"]) == 0.0:
        return clean
    sampled = empirical[:, sources].copy()
    sampled /= np.maximum(np.linalg.norm(sampled, axis=0, keepdims=True), 1e-15)
    sampled *= float(manifest["residual_ratio"]) * np.linalg.norm(clean)
    observed = np.maximum(clean[:, None] + sampled, 0.0)
    return observed.mean(axis=1)


def run_rho_zero_parity(asset_root: Path, output_dir: Path, pairs: pd.DataFrame) -> None:
    """Recompute the frozen 12-pair v48 certificate and record raw parity."""
    data = asset_root / "adapter_pipeline/outputs/v38_ce29_fragmentionfixed_profile_overlap_empiricalfwhm_globalq99_ista_ready"
    reference = asset_root / "results_758_v38_ce29_empiricalfwhm_globalq99_fixed_library_joint_earlystop"
    manifest_path = asset_root / "v48_pilot_48" / "v48_pilot_manifest.csv"
    required = [data / "A_library.npy", data / "B_cube.npy", data / "foreground_pixel_mask.npy", reference / "X_abundance.npy", manifest_path]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise RuntimeError("MISSING_V48_PARITY_ASSET: " + "; ".join(missing))
    manifest = pd.read_csv(manifest_path).set_index("case_id")
    A = np.load(data / "A_library.npy").astype(np.float64)
    A = A[0] if A.ndim == 3 else A
    B_real = np.load(data / "B_cube.npy").astype(np.float64)[0]
    mask = np.load(data / "foreground_pixel_mask.npy").astype(bool)
    X_prod = np.load(reference / "X_abundance.npy").astype(np.float64)
    measured = np.moveaxis(B_real, -1, 0).reshape(A.shape[0], -1)
    real_x = X_prod.reshape(A.shape[1], -1)
    empirical = measured[:, mask.reshape(-1)] - A @ real_x[:, mask.reshape(-1)]
    empirical = empirical[:, np.linalg.norm(empirical, axis=0) > 1e-12]
    weights = identity_weights(A)
    reconstructed: dict[int, np.ndarray] = {}
    results = []
    for pair in pairs.to_dict("records"):
        case_id = int(pair["case_id"])
        if case_id not in manifest.index:
            raise RuntimeError(f"MISSING_V48_CASE_MANIFEST: case_id={case_id}")
        if case_id not in reconstructed:
            reconstructed[case_id] = reconstruct_v48_certificate_b(
                manifest.loc[case_id].to_dict(), asset_root, A, empirical
            )
        recomputed = rho_zero_from_weighted_case(
            A, reconstructed[case_id], int(pair["candidate_index"]), weights
        )["rho_zero"]
        cached = float(pair["cached_rho_zero"])
        absolute = abs(float(recomputed) - cached)
        results.append({
            **pair,
            "recomputed_rho_zero": float(recomputed),
            "abs_diff": absolute,
            "relative_diff": absolute / max(abs(cached), 1e-30),
        })
    output_dir.mkdir(parents=True, exist_ok=True)
    result_frame = pd.DataFrame(results)
    result_frame.to_csv(output_dir / "rho_zero_parity_results.csv", index=False)
    status = {
        "status": "REVIEW_REQUIRED",
        "n_pairs": int(len(pairs)),
        "n_success": int(len(results)),
        "max_abs_diff": float(result_frame.abs_diff.max()),
        "median_abs_diff": float(result_frame.abs_diff.median()),
        "max_relative_diff": float(result_frame.relative_diff.max()),
        "median_relative_diff": float(result_frame.relative_diff.median()),
        "historical_weighting": "identity (W=I)",
        "historical_objective": "nonnegative NNLS q_star and leave-one-candidate-out nonnegative NNLS q_deleted; rho_zero=max(0,(q_deleted-q_star)/(b^T W^2 b + 1e-12))",
        "notes": [
            "Exact v48 reconstruction from run_v48_pilot_48.py and attach_v47_certificates_v48_pilot.py.",
            "No frozen numerical tolerance exists; raw differences require human review before PASS is recorded.",
        ],
    }
    (output_dir / "rho_zero_parity_status.json").write_text(
        json.dumps(status, indent=2), encoding="utf-8"
    )
    print(json.dumps(status, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--asset-root", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed-base", type=int, default=SEED_BASE)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--validate-rho-zero", action="store_true")
    args = ap.parse_args()
    rows = planned_cases(args.seed_base)
    validate_plan(rows)
    pairs = parity_manifest()
    if args.dry_run:
        paths = asset_paths(args.asset_root)
        print(json.dumps({
            "status": "DRY_RUN_OK", "planned_cases": len(rows),
            "calibration_cases": 96, "held_out_test_cases": 96,
            "factor_counts": {"K": 4, "residual": 2, "selection": 2, "abundance": 3, "replicates": 4},
            "seed_base": args.seed_base, "unique_case_ids": True, "unique_case_seeds": True,
            "parity_pairs": pairs.to_dict("records"),
            "expected_stage0_hashes": json.loads((ROOT / "results/v48_production_ista_lock/v48_production_ista_lock.json").read_text(encoding="utf-8"))["hashes_sha256"],
            "required_remote_asset_paths": {k: str(v) for k, v in paths.items()},
            "output_dir": str(args.output_dir),
        }, indent=2))
        return
    # Validation establishes the status file; it must not require one first.
    if args.validate_rho_zero:
        run_rho_zero_parity(args.asset_root, args.output_dir, pairs)
        return
    parity_status = args.output_dir / "rho_zero_parity_status.json"
    if not parity_status.exists():
        raise SystemExit("RHO_ZERO_PARITY_NOT_VALIDATED")
    status = json.loads(parity_status.read_text(encoding="utf-8"))
    if status.get("status") != "PASS":
        raise SystemExit("RHO_ZERO_PARITY_FAILED")
    raise SystemExit("V49A_SMOKE_RUNTIME_NOT_EXECUTED_IN_PREPARATION_MODE")


if __name__ == "__main__":
    main()
