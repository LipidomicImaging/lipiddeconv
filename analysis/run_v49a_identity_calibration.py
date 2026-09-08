#!/usr/bin/env python3
"""Prepare and, on the remote GPU host, execute the gated v49a calibration."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import time
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
CALIBRATION_BOOTSTRAP_SEED = 20260909
PRIMARY_GATE = 1e-3
SENSITIVITY_GATES = (1e-4, 1e-2)
ABUNDANCE_POWERS = {
    "lighter_tail": 0.75,
    "central": 1.0,
    "heavier_tail": 1.25,
}
N_BOOTSTRAP = 2000
HELD_OUT_BOOTSTRAP_SEED = 20260910
RISK_TARGET = 0.05
MIN_RETAINED_COVERAGE = 0.05
TARGET_B_NORM = 0.6036783456802368
ACTIVE_THRESHOLD = 1e-4
SMOKE_CASE_IDS = (0, 61, 130, 191)
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
        "production_abundance": result / "X_abundance.npy",
        "competition_groups": root / "profile_758_v47_30roi_data_only/competition_group_family.csv",
    }


def smoke_cases(rows: list[dict]) -> list[dict]:
    """Return the frozen four-case smoke subset in execution order."""
    by_id = {int(row["case_id"]): row for row in rows}
    selected = [by_id[case_id] for case_id in SMOKE_CASE_IDS]
    assert [row["K"] for row in selected] == list(K_VALUES)
    assert {row["split"] for row in selected} == {"calibration", "held_out_test"}
    assert {row["residual_ratio"] for row in selected} == set(RESIDUALS)
    assert {row["selection_mode"] for row in selected} == set(SELECTIONS)
    return selected


def atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def atomic_save_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, path)


def load_runtime_assets(asset_root: Path, device_name: str) -> dict:
    """Load the exact assets used by the historical v48 generator and ISTA."""
    import torch

    from lipid_ista import LipidENNet
    from run_v48_pilot_48 import load_csv
    from utils import get_A_matrix

    paths = {key: Path(str(value)) for key, value in asset_paths(asset_root).items()}
    required = list(paths.values())
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise RuntimeError("MISSING_DEPENDENCY: " + "; ".join(missing))
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("MISSING_DEPENDENCY: CUDA requested but unavailable")

    A = np.load(paths["A_library"]).astype(np.float32)
    A = A[0] if A.ndim == 3 else A
    B_real = np.load(paths["B_cube"]).astype(np.float32)[0]
    mask = np.load(paths["foreground_mask"]).astype(bool)
    X_prod = np.load(paths["production_abundance"]).astype(np.float32)
    metadata = np.load(paths["candidate_metadata"], allow_pickle=True).item()
    measured = np.moveaxis(B_real, -1, 0).reshape(A.shape[0], -1)
    real_x = X_prod.reshape(A.shape[1], -1)
    flat_mask = mask.reshape(-1)
    empirical = measured[:, flat_mask] - A @ real_x[:, flat_mask]
    empirical = empirical[:, np.linalg.norm(empirical, axis=0) > 1e-12]
    if empirical.shape[1] == 0:
        raise RuntimeError("MISSING_DEPENDENCY: no nonzero empirical residual columns")
    X_fg = X_prod[:, mask].T
    group_rows = [
        row for row in load_csv(paths["competition_groups"])
        if row["weight_mode"] == "identity"
        and abs(float(row["d_frag_cutoff"]) - 0.02) < 1e-12
    ]
    hard_pool = np.asarray(sorted({
        int(index)
        for row in group_rows
        for index in row["member_indices"].split(";")
    }), dtype=int)
    if hard_pool.size == 0:
        raise RuntimeError("MISSING_DEPENDENCY: empty frozen d_frag=0.02 hard pool")

    A_init = get_A_matrix(str(paths["A_library"]), device)
    if A_init is None:
        raise RuntimeError("MISSING_DEPENDENCY: production A tensor failed to load")
    net = LipidENNet(A_init=A_init, K=12, clamp_min=0.5, clamp_max=1.5).to(device)
    net.load_state_dict(torch.load(
        paths["checkpoint"], map_location=device, weights_only=True
    ))
    net.eval()
    return {
        "torch": torch, "device": device, "A": A, "mask": mask,
        "metadata": metadata, "empirical": empirical, "X_fg": X_fg,
        "hard_pool": hard_pool, "net": net,
        "all_indices": np.arange(A.shape[1], dtype=int),
    }


def generate_v49a_case(case: dict, runtime: dict) -> dict:
    """Generate one case through the frozen historical v48 simulation path."""
    A = runtime["A"]
    mask = runtime["mask"]
    X_fg = runtime["X_fg"]
    rng = np.random.default_rng(int(case["seed"]))
    k = int(case["K"])
    eligible = np.flatnonzero((X_fg > ACTIVE_THRESHOLD).sum(axis=1) >= k)
    if eligible.size == 0:
        raise RuntimeError(f"MISSING_DEPENDENCY: no empirical rank profile supports K={k}")
    if case["selection_mode"] == "random":
        selected = np.sort(rng.choice(runtime["all_indices"], size=k, replace=False))
    else:
        take_hard = min(k, runtime["hard_pool"].size)
        selected_hard = rng.choice(runtime["hard_pool"], size=take_hard, replace=False)
        if take_hard < k:
            remaining = np.setdiff1d(
                runtime["all_indices"], selected_hard, assume_unique=False
            )
            selected = np.sort(np.concatenate([
                selected_hard,
                rng.choice(remaining, size=k - take_hard, replace=False),
            ]))
        else:
            selected = np.sort(selected_hard)

    profile_index = int(rng.choice(eligible))
    weights = np.sort(
        X_fg[profile_index][X_fg[profile_index] > ACTIVE_THRESHOLD]
    )[::-1][:k].astype(np.float64)
    abundance_mode = str(case["abundance_mode"])
    weights = np.power(weights, ABUNDANCE_POWERS[abundance_mode])
    weights /= weights.sum()
    weights = weights[rng.permutation(k)]
    clean = A[:, selected] @ weights.astype(np.float32)
    scale = TARGET_B_NORM / max(float(np.linalg.norm(clean)), 1e-30)
    true_values = (weights * scale).astype(np.float32)
    clean = A[:, selected] @ true_values
    B_sim = np.zeros((A.shape[0], *mask.shape), dtype=np.float32)
    B_sim[:, mask] = clean[:, None]
    residual_sources = np.full(mask.sum(), -1, dtype=np.int32)
    if float(case["residual_ratio"]) > 0.0:
        residual_sources = rng.choice(
            runtime["empirical"].shape[1], size=mask.sum(), replace=True
        ).astype(np.int32)
        sampled = runtime["empirical"][:, residual_sources].copy()
        sampled /= np.maximum(np.linalg.norm(sampled, axis=0, keepdims=True), 1e-15)
        sampled *= (
            float(case["residual_ratio"])
            * np.linalg.norm(B_sim[:, mask], axis=0, keepdims=True)
        )
        B_sim[:, mask] = np.maximum(B_sim[:, mask] + sampled, 0.0)
    return {
        "selected": selected, "profile_index": profile_index,
        "true_values": true_values, "clean": clean, "B_sim": B_sim,
        "residual_sources": residual_sources, "scale": scale,
    }


def run_one_case(case: dict, runtime: dict, output_dir: Path) -> dict:
    """Execute and atomically save one resumable v49a case."""
    torch = runtime["torch"]
    case_name = (
        f"case_{int(case['case_id']):03d}_K{case['K']}_{case['selection_mode']}_"
        f"r{int(float(case['residual_ratio']) * 100):02d}_"
        f"{case['abundance_mode']}_rep{case['replicate']}"
    )
    case_dir = output_dir / "cases" / case_name
    completion_path = case_dir / "complete.json"
    result_path = case_dir / "case_result.npz"
    if completion_path.exists() and result_path.exists():
        completion = json.loads(completion_path.read_text(encoding="utf-8"))
        if completion.get("status") == "PASS" and completion.get("case") == case:
            print(json.dumps({"case_id": case["case_id"], "status": "RESUMED"}), flush=True)
            return completion

    started = time.perf_counter()
    generated = generate_v49a_case(case, runtime)
    B_tensor = torch.as_tensor(
        generated["B_sim"][None], dtype=torch.float32, device=runtime["device"]
    )
    if runtime["device"].type == "cuda":
        torch.cuda.synchronize()
    forward_started = time.perf_counter()
    with torch.no_grad():
        X_hat_t, _, A_cal_t, _ = runtime["net"](B_tensor)
        B_hat_t = torch.einsum("ml,blhw->bmhw", A_cal_t, X_hat_t)
    if runtime["device"].type == "cuda":
        torch.cuda.synchronize()
    forward_seconds = time.perf_counter() - forward_started
    X_hat = X_hat_t[0].cpu().numpy().astype(np.float32)
    B_hat = B_hat_t[0].cpu().numpy().astype(np.float32)
    xhat_mean = X_hat[:, runtime["mask"]].mean(axis=1).astype(np.float64)
    xtrue = np.zeros(runtime["A"].shape[1], dtype=np.float64)
    xtrue[generated["selected"]] = generated["true_values"]
    certificate_candidates = np.flatnonzero(
        xhat_mean > min(PRIMARY_GATE, *SENSITIVITY_GATES)
    )
    certificate_b = generated["B_sim"][:, runtime["mask"]].mean(axis=1)
    certificate_weights = identity_weights(runtime["A"])
    rho_zero = np.full(runtime["A"].shape[1], np.nan, dtype=np.float64)
    for candidate_index in certificate_candidates:
        rho_zero[candidate_index] = rho_zero_from_weighted_case(
            runtime["A"], certificate_b, int(candidate_index), certificate_weights
        )["rho_zero"]
    rec_rel = float(np.linalg.norm(
        B_hat[:, runtime["mask"]] - generated["B_sim"][:, runtime["mask"]]
    ) / max(np.linalg.norm(generated["B_sim"][:, runtime["mask"]]), 1e-30))
    total_hat = max(float(xhat_mean.sum()), 1e-30)
    false_mass = float(xhat_mean[xtrue == 0].sum() / total_hat)
    atomic_save_npz(
        result_path,
        candidate_index=np.arange(runtime["A"].shape[1], dtype=np.int32),
        X_true=xtrue, X_hat=xhat_mean, rho_zero=rho_zero,
        reported_primary=(xhat_mean > PRIMARY_GATE),
        selected_indices=generated["selected"],
        true_abundances=generated["true_values"],
        residual_source_indices=generated["residual_sources"],
    )
    completion = {
        "status": "PASS", "case": case, "case_name": case_name,
        "empirical_profile_pixel_index": generated["profile_index"],
        "n_reported_primary": int((xhat_mean > PRIMARY_GATE).sum()),
        "false_allocation_mass": false_mass,
        "reconstruction_relative_residual": rec_rel,
        "GPU_forward_seconds": forward_seconds,
        "case_wall_seconds": time.perf_counter() - started,
        "result_path": str(result_path),
    }
    atomic_write_json(completion_path, completion)
    print(json.dumps({
        "case_id": case["case_id"], "status": "PASS",
        "n_reported_primary": int((xhat_mean > PRIMARY_GATE).sum()),
    }), flush=True)
    return completion


def run_cases(rows: list[dict], asset_root: Path, output_dir: Path, device: str) -> None:
    runtime = load_runtime_assets(asset_root, device)
    output_dir.mkdir(parents=True, exist_ok=True)
    for case in rows:
        run_one_case(case, runtime, output_dir)


def result_case_name(case: dict) -> str:
    return (
        f"case_{int(case['case_id']):03d}_K{case['K']}_{case['selection_mode']}_"
        f"r{int(float(case['residual_ratio']) * 100):02d}_"
        f"{case['abundance_mode']}_rep{case['replicate']}"
    )


def load_case_frame(rows: list[dict], output_dir: Path) -> pd.DataFrame:
    """Load completed atomic case payloads into one candidate-level table."""
    frames = []
    for case in rows:
        case_dir = output_dir / "cases" / result_case_name(case)
        completion_path = case_dir / "complete.json"
        result_path = case_dir / "case_result.npz"
        if not completion_path.exists() or not result_path.exists():
            raise RuntimeError(f"INCOMPLETE_V49A_CASE: case_id={case['case_id']}")
        completion = json.loads(completion_path.read_text(encoding="utf-8"))
        if completion.get("status") != "PASS" or completion.get("case") != case:
            raise RuntimeError(f"INVALID_V49A_CASE_MARKER: case_id={case['case_id']}")
        with np.load(result_path) as result:
            frame = pd.DataFrame({
                "candidate_index": result["candidate_index"].astype(int),
                "X_true": result["X_true"].astype(float),
                "X_hat": result["X_hat"].astype(float),
                "rho_zero": result["rho_zero"].astype(float),
            })
        for key, value in case.items():
            frame[key] = value
        frames.append(frame)
    combined = pd.concat(frames, ignore_index=True)
    expected = len(rows) * int(frames[0].candidate_index.nunique())
    if len(combined) != expected:
        raise RuntimeError(f"INVALID_V49A_ROW_COUNT: {len(combined)} != {expected}")
    return combined


def metric_contributions(
    frame: pd.DataFrame, gate: float, thresholds: np.ndarray
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Aggregate pooled metric numerators/denominators by whole case."""
    case_ids = np.asarray(sorted(frame.case_id.unique()), dtype=int)
    thresholds = np.asarray(thresholds, dtype=float)
    shape = (len(case_ids), len(thresholds))
    values = {
        "retained": np.zeros(shape, dtype=float),
        "false_retained": np.zeros(shape, dtype=float),
        "retained_mass": np.zeros(shape, dtype=float),
        "false_retained_mass": np.zeros(shape, dtype=float),
        "true_retained": np.zeros(shape, dtype=float),
        "reported": np.zeros(len(case_ids), dtype=float),
        "true_total": np.zeros(len(case_ids), dtype=float),
    }
    for row_index, case_id in enumerate(case_ids):
        case = frame[frame.case_id == case_id]
        truth = case.X_true.to_numpy() > 0
        reported = case.X_hat.to_numpy() > gate
        if reported.any() and not np.isfinite(case.loc[reported, "rho_zero"]).all():
            raise RuntimeError(
                f"MISSING_RHO_ZERO_FOR_GATE: case_id={case_id}, gate={gate}"
            )
        values["reported"][row_index] = reported.sum()
        values["true_total"][row_index] = truth.sum()
        rho = case.loc[reported, "rho_zero"].to_numpy(dtype=float)
        order = np.argsort(rho)
        rho = rho[order]
        reported_truth = truth[reported][order]
        reported_mass = case.loc[reported, "X_hat"].to_numpy(dtype=float)[order]
        indices = np.searchsorted(rho, thresholds, side="left")
        prefix_false = np.concatenate([[0.0], np.cumsum(~reported_truth)])
        prefix_mass = np.concatenate([[0.0], np.cumsum(reported_mass)])
        prefix_false_mass = np.concatenate([
            [0.0], np.cumsum(reported_mass * (~reported_truth))
        ])
        prefix_true = np.concatenate([[0.0], np.cumsum(reported_truth)])
        values["retained"][row_index] = len(rho) - indices
        values["false_retained"][row_index] = prefix_false[-1] - prefix_false[indices]
        values["retained_mass"][row_index] = prefix_mass[-1] - prefix_mass[indices]
        values["false_retained_mass"][row_index] = (
            prefix_false_mass[-1] - prefix_false_mass[indices]
        )
        values["true_retained"][row_index] = prefix_true[-1] - prefix_true[indices]
    return case_ids, values


def safe_ratio(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    """Return pooled ratios; empty retained samples receive conservative risk 1."""
    return np.divide(
        numerator, denominator, out=np.ones_like(numerator, dtype=float),
        where=denominator > 0,
    )


def point_metrics(values: dict[str, np.ndarray], column: int) -> dict[str, float]:
    retained = float(values["retained"][:, column].sum())
    retained_mass = float(values["retained_mass"][:, column].sum())
    reported = float(values["reported"].sum())
    true_total = float(values["true_total"].sum())
    return {
        "candidate_false_identity_fraction": (
            float(values["false_retained"][:, column].sum()) / retained
            if retained > 0 else 1.0
        ),
        "mass_weighted_false_fraction": (
            float(values["false_retained_mass"][:, column].sum()) / retained_mass
            if retained_mass > 0 else 1.0
        ),
        "retained_candidate_coverage": retained / reported if reported > 0 else 0.0,
        "true_identity_recall": (
            float(values["true_retained"][:, column].sum()) / true_total
            if true_total > 0 else 0.0
        ),
        "n_reported": int(reported),
        "n_retained": int(retained),
    }


def bootstrap_upper_bounds(
    values: dict[str, np.ndarray], seed: int, n_bootstrap: int
) -> tuple[np.ndarray, np.ndarray]:
    """Whole-case bootstrap upper95 risks for every threshold column."""
    n_cases, n_thresholds = values["retained"].shape
    rng = np.random.default_rng(seed)
    sampled = rng.integers(0, n_cases, size=(n_bootstrap, n_cases))
    multiplicities = np.zeros((n_bootstrap, n_cases), dtype=np.int16)
    for row in range(n_bootstrap):
        multiplicities[row] = np.bincount(sampled[row], minlength=n_cases)
    candidate_upper = np.empty(n_thresholds, dtype=float)
    mass_upper = np.empty(n_thresholds, dtype=float)
    for start in range(0, n_thresholds, 256):
        stop = min(start + 256, n_thresholds)
        retained = multiplicities @ values["retained"][:, start:stop]
        false_retained = multiplicities @ values["false_retained"][:, start:stop]
        retained_mass = multiplicities @ values["retained_mass"][:, start:stop]
        false_mass = multiplicities @ values["false_retained_mass"][:, start:stop]
        candidate_values = safe_ratio(false_retained, retained)
        mass_values = safe_ratio(false_mass, retained_mass)
        candidate_upper[start:stop] = np.quantile(candidate_values, 0.95, axis=0)
        mass_upper[start:stop] = np.quantile(mass_values, 0.95, axis=0)
    return candidate_upper, mass_upper


def evaluate_thresholds(
    frame: pd.DataFrame, gate: float, thresholds: np.ndarray,
    bootstrap_seed: int,
) -> pd.DataFrame:
    _, values = metric_contributions(frame, gate, thresholds)
    candidate_upper, mass_upper = bootstrap_upper_bounds(
        values, bootstrap_seed, N_BOOTSTRAP
    )
    rows = []
    for column, threshold in enumerate(thresholds):
        row = {"tau": float(threshold), **point_metrics(values, column)}
        row["candidate_false_identity_upper95"] = float(candidate_upper[column])
        row["mass_weighted_false_upper95"] = float(mass_upper[column])
        rows.append(row)
    return pd.DataFrame(rows)


def atomic_write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def select_identity_threshold(calibration: pd.DataFrame) -> tuple[float, pd.DataFrame]:
    reported = calibration.X_hat > PRIMARY_GATE
    thresholds = np.unique(calibration.loc[reported, "rho_zero"].to_numpy(dtype=float))
    if thresholds.size == 0 or not np.isfinite(thresholds).all():
        raise RuntimeError("MISSING_CALIBRATION_RHO_ZERO")
    evaluated = evaluate_thresholds(
        calibration, PRIMARY_GATE, thresholds, CALIBRATION_BOOTSTRAP_SEED
    )
    valid = evaluated[
        (evaluated.candidate_false_identity_upper95 <= RISK_TARGET)
        & (evaluated.mass_weighted_false_upper95 <= RISK_TARGET)
        & (evaluated.retained_candidate_coverage >= MIN_RETAINED_COVERAGE)
    ]
    if valid.empty:
        raise RuntimeError("NO_VALID_V49A_IDENTITY_THRESHOLD")
    chosen = valid.sort_values(
        ["retained_candidate_coverage", "tau"], ascending=[False, True]
    ).iloc[0]
    evaluated["selected"] = evaluated.tau == float(chosen.tau)
    return float(chosen.tau), evaluated


def evaluate_single_threshold(
    frame: pd.DataFrame, gate: float, tau: float, bootstrap_seed: int
) -> dict[str, float]:
    evaluated = evaluate_thresholds(
        frame, gate, np.asarray([tau], dtype=float), bootstrap_seed
    )
    return evaluated.iloc[0].to_dict()


def held_out_strata(held_out: pd.DataFrame, tau: float) -> pd.DataFrame:
    rows = []
    for factor in ("K", "residual_ratio", "selection_mode", "abundance_mode"):
        for level, stratum in held_out.groupby(factor, sort=True):
            rows.append({
                "factor": factor, "level": level,
                **evaluate_single_threshold(
                    stratum, PRIMARY_GATE, tau, HELD_OUT_BOOTSTRAP_SEED
                ),
            })
    return pd.DataFrame(rows)


def final_decision(overall: dict, strata: pd.DataFrame) -> str:
    point_pass = (
        overall["candidate_false_identity_fraction"] <= RISK_TARGET
        and overall["mass_weighted_false_fraction"] <= RISK_TARGET
    )
    coverage_pass = overall["retained_candidate_coverage"] >= MIN_RETAINED_COVERAGE
    if not point_pass or not coverage_pass:
        return "FAIL"
    uncertainty_pass = (
        overall["candidate_false_identity_upper95"] <= RISK_TARGET
        and overall["mass_weighted_false_upper95"] <= RISK_TARGET
    )
    strata_pass = bool((
        (strata.candidate_false_identity_fraction <= RISK_TARGET)
        & (strata.mass_weighted_false_fraction <= RISK_TARGET)
    ).all())
    return "PASS" if uncertainty_pass and strata_pass else "CONDITIONAL"


def analyze_completed_run(rows: list[dict], output_dir: Path) -> None:
    frame = load_case_frame(rows, output_dir)
    calibration = frame[frame.replicate.isin((0, 1))].copy()
    held_out = frame[frame.replicate.isin((2, 3))].copy()
    if calibration.case_id.nunique() != 96 or held_out.case_id.nunique() != 96:
        raise RuntimeError("INVALID_V49A_SPLIT")
    tau, calibration_table = select_identity_threshold(calibration)
    overall = evaluate_single_threshold(
        held_out, PRIMARY_GATE, tau, HELD_OUT_BOOTSTRAP_SEED
    )
    strata = held_out_strata(held_out, tau)
    sensitivity_rows = []
    for gate in SENSITIVITY_GATES:
        sensitivity_rows.append({
            "reporting_gate": gate,
            **evaluate_single_threshold(
                held_out, gate, tau, HELD_OUT_BOOTSTRAP_SEED
            ),
        })
    sensitivity = pd.DataFrame(sensitivity_rows)
    decision = final_decision(overall, strata)
    atomic_write_csv(output_dir / "v49a_calibration_thresholds.csv", calibration_table)
    atomic_write_csv(output_dir / "v49a_held_out_strata.csv", strata)
    atomic_write_csv(output_dir / "v49a_sensitivity.csv", sensitivity)
    report = {
        "status": decision,
        "tau_identity": tau,
        "primary_reporting_gate": PRIMARY_GATE,
        "calibration_case_count": 96,
        "held_out_case_count": 96,
        "n_bootstrap": N_BOOTSTRAP,
        "calibration_bootstrap_seed": CALIBRATION_BOOTSTRAP_SEED,
        "held_out_bootstrap_seed": HELD_OUT_BOOTSTRAP_SEED,
        "held_out_overall": overall,
        "stress_strata": strata.to_dict("records"),
        "sensitivity": sensitivity.to_dict("records"),
        "decision_rules": {
            "risk_target": RISK_TARGET,
            "minimum_retained_candidate_coverage": MIN_RETAINED_COVERAGE,
            "stratum_exceeds_target": (
                "candidate false-identity point estimate > 0.05 or "
                "mass-weighted false point estimate > 0.05"
            ),
        },
    }
    atomic_write_json(output_dir / "v49a_identity_calibration_report.json", report)
    print(json.dumps({"status": decision, "tau_identity": tau}, indent=2), flush=True)


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
            "abundance_powers": ABUNDANCE_POWERS,
            "primary_reporting_gate": PRIMARY_GATE,
            "sensitivity_reporting_gates": list(SENSITIVITY_GATES),
            "calibration": {
                "replicates": [0, 1], "candidate_thresholds": "all unique observed rho_zero",
                "n_bootstrap": N_BOOTSTRAP, "bootstrap_seed": CALIBRATION_BOOTSTRAP_SEED,
                "risk_upper_quantile": 0.95, "risk_target": RISK_TARGET,
                "minimum_retained_candidate_coverage": MIN_RETAINED_COVERAGE,
                "selection": "maximum retained coverage; smallest tau breaks ties",
            },
            "held_out": {
                "replicates": [2, 3], "bootstrap_seed": HELD_OUT_BOOTSTRAP_SEED,
                "same_tau_identity": True,
            },
            "seed_base": args.seed_base, "unique_case_ids": True, "unique_case_seeds": True,
            "smoke_case_ids": list(SMOKE_CASE_IDS),
            "smoke_cases": smoke_cases(rows),
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
    selected_rows = smoke_cases(rows) if args.smoke else rows
    run_cases(selected_rows, args.asset_root, args.output_dir, args.device)
    if not args.smoke:
        analyze_completed_run(rows, args.output_dir)


if __name__ == "__main__":
    main()
