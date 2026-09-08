from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import time
from pathlib import Path

# Prevent process-level parallelism from nesting BLAS thread pools.
for _thread_variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_thread_variable, "1")

import numpy as np
from joblib import Parallel, delayed
from scipy.optimize import minimize, nnls


RHO_GRID = (0.0, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2)
CONE_CUTOFFS = (0.01, 0.02, 0.05, 0.10)


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_rows(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def auto_channel_weights(A: np.ndarray, classes: list[str]) -> np.ndarray:
    present = np.abs(A) > 1e-8
    counts = present.sum(axis=1)
    rarity = np.empty(A.shape[0], dtype=np.float64)
    rarity[counts <= 3] = 5.0
    rarity[(counts > 3) & (counts <= 10)] = 3.0
    rarity[(counts > 10) & (counts <= 30)] = 1.5
    rarity[counts > 30] = 0.5
    specificity = np.ones(A.shape[0], dtype=np.float64)
    labels = np.asarray(classes, dtype=str)
    for row in range(A.shape[0]):
        columns = np.flatnonzero(present[row])
        if not len(columns):
            continue
        _, class_counts = np.unique(labels[columns], return_counts=True)
        fraction = float(class_counts.max() / class_counts.sum())
        specificity[row] = 4.0 if fraction > 0.90 else (2.0 if fraction > 0.70 else 1.0)
    weights = rarity * specificity
    weights /= max(float(weights.mean()), 1e-15)
    weights = np.clip(weights, 0.2, 5.0)
    # This matches the v38 implementation: clip after the first mean normalization.
    return weights


def solve_nonnegative_lasso(A: np.ndarray, b: np.ndarray, penalty: float, iterations: int = 10000) -> np.ndarray:
    if penalty == 0:
        return nnls(A, b, maxiter=10 * A.shape[1])[0]
    gram = A.T @ A
    atb = A.T @ b
    vector = np.ones(A.shape[1], dtype=np.float64)
    vector /= np.linalg.norm(vector)
    for _ in range(100):
        vector = gram @ vector
        vector /= max(np.linalg.norm(vector), 1e-15)
    lipschitz = max(float(vector @ gram @ vector), 1e-15)
    step = 0.98 / lipschitz
    x = np.zeros(A.shape[1], dtype=np.float64)
    y = x.copy()
    momentum = 1.0
    for _ in range(iterations):
        x_next = np.maximum(y - step * (gram @ y - atb) - step * penalty, 0.0)
        next_momentum = 0.5 * (1.0 + math.sqrt(1.0 + 4.0 * momentum * momentum))
        y = x_next + ((momentum - 1.0) / next_momentum) * (x_next - x)
        if np.linalg.norm(x_next - x) <= 1e-10 * max(np.linalg.norm(x), 1.0):
            x = x_next
            break
        x = x_next
        momentum = next_momentum
    return x


def build_rois(B: np.ndarray, mask: np.ndarray, block: int, count: int) -> tuple[list[dict], list[np.ndarray]]:
    candidates: list[dict] = []
    spectra: list[np.ndarray] = []
    height, width, _ = B.shape
    for y0 in range(0, height, block):
        for x0 in range(0, width, block):
            y1, x1 = min(y0 + block, height), min(x0 + block, width)
            local_mask = mask[y0:y1, x0:x1]
            foreground = int(local_mask.sum())
            total = int(local_mask.size)
            if foreground < max(4, math.ceil(0.50 * total)):
                continue
            pixels = B[y0:y1, x0:x1][local_mask]
            spectrum = pixels.mean(axis=0).astype(np.float64)
            spectra.append(spectrum)
            candidates.append(
                {
                    "grid_id": len(candidates), "y0": y0, "y1": y1, "x0": x0, "x1": x1,
                    "center_y": 0.5 * (y0 + y1 - 1), "center_x": 0.5 * (x0 + x1 - 1),
                    "foreground_pixels": foreground, "block_pixels": total,
                    "foreground_fraction": foreground / total,
                    "tic": float(spectrum.sum()), "b_l2": float(np.linalg.norm(spectrum)),
                }
            )
    scores = np.asarray([row["b_l2"] for row in candidates])
    q1, q2 = np.quantile(scores, [1 / 3, 2 / 3])
    strata = np.where(scores <= q1, 0, np.where(scores <= q2, 1, 2))
    per_stratum = [count // 3] * 3
    for i in range(count % 3):
        per_stratum[2 - i] += 1
    selected: list[int] = []
    for stratum, wanted in enumerate(per_stratum):
        pool = np.flatnonzero(strata == stratum).tolist()
        if not pool:
            continue
        center_score = float(np.median(scores[pool]))
        first = min(pool, key=lambda i: abs(scores[i] - center_score))
        chosen = [first]
        while len(chosen) < min(wanted, len(pool)):
            remaining = [i for i in pool if i not in chosen]
            nxt = max(
                remaining,
                key=lambda i: min(
                    (candidates[i]["center_y"] - candidates[j]["center_y"]) ** 2
                    + (candidates[i]["center_x"] - candidates[j]["center_x"]) ** 2
                    for j in chosen
                ),
            )
            chosen.append(nxt)
        selected.extend(chosen)
    rows, out_spectra = [], []
    for roi_id, index in enumerate(selected[:count]):
        row = dict(candidates[index])
        row.update(
            {
                "roi_id": f"ROI_{roi_id:02d}",
                "signal_stratum": ("low", "mid", "high")[int(strata[index])],
                "stratum_q33_b_l2": float(q1), "stratum_q67_b_l2": float(q2),
                "selection_basis": "raw_B_block_mean_l2_then_spatial_farthest_sampling",
            }
        )
        rows.append(row)
        out_spectra.append(spectra[index])
    return rows, out_spectra


def cone_redundancy(
    A: np.ndarray, mz: np.ndarray, metadata: list[dict], weights: dict[str, np.ndarray]
) -> tuple[list[dict], dict[tuple[str, int], tuple[float, list[int]]]]:
    regions = {
        "full": np.ones(len(mz), dtype=bool),
        "fragment_lt650": mz < 650.0,
        "parent_related_650_803": (mz >= 650.0) & (mz <= 803.0),
    }
    rows: list[dict] = []
    compact: dict[tuple[str, int], tuple[float, list[int]]] = {}
    for weight_name, w in weights.items():
        for region_name, region in regions.items():
            Aw = A[region] * w[region, None]
            for j in range(A.shape[1]):
                keep = np.arange(A.shape[1]) != j
                target = Aw[:, j]
                denom = max(float(np.linalg.norm(target)), 1e-15)
                coefficients, residual = nnls(Aw[:, keep], target, maxiter=10 * A.shape[1])
                donors_all = np.flatnonzero(keep)
                order = np.argsort(coefficients)[::-1]
                donors = [int(donors_all[k]) for k in order[:8] if coefficients[k] > 1e-8]
                donor_coeffs = [float(coefficients[k]) for k in order[:8] if coefficients[k] > 1e-8]
                relative = float(residual / denom)
                rows.append(
                    {
                        "weight_mode": weight_name, "region": region_name, "candidate_index": j,
                        "candidate_id": metadata[j]["candidate_id"], "lipid_name": metadata[j]["lipid_name"],
                        "precursor_mz": metadata[j]["mz"], "relative_cone_residual": relative,
                        "top_donor_indices": ";".join(map(str, donors)),
                        "top_donor_lipids": " | ".join(metadata[k]["lipid_name"] for k in donors),
                        "top_donor_coefficients": ";".join(f"{v:.9g}" for v in donor_coeffs),
                    }
                )
                if region_name == "fragment_lt650":
                    compact[(weight_name, j)] = (relative, donors)
    return rows, compact


def connected_components(n: int, edges: list[tuple[int, int]]) -> list[list[int]]:
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for a, b in edges:
        union(a, b)
    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    return [members for members in groups.values() if len(members) > 1]


def build_group_family(
    metadata: list[dict], compact: dict[tuple[str, int], tuple[float, list[int]]]
) -> list[dict]:
    rows: list[dict] = []
    for weight_name in sorted({key[0] for key in compact}):
        for cutoff in CONE_CUTOFFS:
            edges: set[tuple[int, int]] = set()
            for j in range(len(metadata)):
                residual, donors = compact[(weight_name, j)]
                if residual < cutoff:
                    for donor in donors[:3]:
                        edges.add(tuple(sorted((j, donor))))
            for group_number, members in enumerate(connected_components(len(metadata), sorted(edges))):
                rows.append(
                    {
                        "weight_mode": weight_name, "d_frag_cutoff": cutoff,
                        "group_id": f"{weight_name}_d{cutoff:g}_{group_number:03d}",
                        "member_count": len(members), "member_indices": ";".join(map(str, members)),
                        "members": " | ".join(metadata[i]["lipid_name"] for i in members),
                    }
                )
    return rows


def profile_path_1d(
    A: np.ndarray, b: np.ndarray, j: int, x_star: np.ndarray, q_star: float,
    q_deleted: float, rhos: tuple[float, ...], epsilon_q: float, max_bisect: int,
) -> list[dict]:
    keep = np.arange(A.shape[1]) != j
    Ar = A[:, keep]
    aj = A[:, j]
    signal = float(b @ b) + epsilon_q
    cache: dict[float, float] = {}

    def g(t: float) -> float:
        key = round(float(t), 13)
        if key not in cache:
            _, residual = nnls(Ar, b - float(t) * aj, maxiter=10 * A.shape[1])
            cache[key] = float(residual * residual)
        return cache[key]

    center = max(float(x_star[j]), 0.0)
    cache[round(0.0, 13)] = q_deleted
    center_loss = g(center)
    requested = [q_star + rho * signal for rho in rhos]
    thresholds = [max(value, center_loss * (1.0 + 1e-10)) for value in requested]

    def simultaneous_inverse(
        indexed_levels: list[tuple[int, float]], lo: float, hi: float, increasing: bool
    ) -> dict[int, float]:
        roots: dict[int, float] = {}

        def recurse(items: list[tuple[int, float]], left: float, right: float, depth: int) -> None:
            if not items:
                return
            if depth >= max_bisect or right - left <= 1e-10 * max(1.0, abs(left), abs(right)):
                for index, _ in items:
                    # Increasing branch: left is feasible. Decreasing branch: right is feasible.
                    roots[index] = left if increasing else right
                return
            mid = 0.5 * (left + right)
            middle_loss = g(mid)
            left_items: list[tuple[int, float]] = []
            right_items: list[tuple[int, float]] = []
            for item in items:
                _, level = item
                if increasing:
                    (right_items if middle_loss <= level else left_items).append(item)
                else:
                    (left_items if middle_loss <= level else right_items).append(item)
            recurse(left_items, left, mid, depth + 1)
            recurse(right_items, mid, right, depth + 1)

        recurse(indexed_levels, lo, hi, 0)
        return roots

    lower = {index: 0.0 for index, threshold in enumerate(thresholds) if q_deleted <= threshold}
    lower_needed = [(index, threshold) for index, threshold in enumerate(thresholds) if q_deleted > threshold]
    if lower_needed and center > 0:
        lower.update(simultaneous_inverse(lower_needed, 0.0, center, increasing=False))

    upper_hi = max(center + 1e-8, 1.25 * center + 1e-6, np.linalg.norm(b) / max(np.linalg.norm(aj), 1e-15))
    maximum_threshold = max(thresholds)
    while g(upper_hi) <= maximum_threshold and upper_hi < 1e6:
        upper_hi *= 2.0
    upper = simultaneous_inverse(list(enumerate(thresholds)), center, upper_hi, increasing=True)

    rho_zero = max(0.0, (q_deleted - q_star) / signal)
    rows = []
    for index, rho in enumerate(rhos):
        lo, hi = lower[index], upper[index]
        rows.append(
            {
                "rho": rho, "lower": lo, "upper": hi,
                "requested_threshold": requested[index], "used_threshold": thresholds[index],
                "center_profile_loss": center_loss, "profile_evaluations_shared": len(cache),
                "lower_boundary_loss": g(lo), "upper_boundary_loss": g(hi),
                "rho_zero_exact": rho_zero,
                "lower_zero_by_necessity_prune": bool(q_deleted <= thresholds[index]),
            }
        )
    # len(cache) above was sampled before boundary diagnostics were requested; make it final and shared.
    for row in rows:
        row["profile_evaluations_shared"] = len(cache)
    return rows


def profile_candidate_rows(
    Aw: np.ndarray, bw: np.ndarray, metadata: list[dict], roi_id: str,
    weight_name: str, j: int, reasons: list[str], x_net_roi: np.ndarray,
    x_nnls: np.ndarray, x_lasso: np.ndarray, q_star: float, signal_norm2: float,
    epsilon_q: float, max_bisect: int,
) -> list[dict]:
    keep = np.arange(Aw.shape[1]) != j
    _, deleted_residual = nnls(Aw[:, keep], bw, maxiter=10 * Aw.shape[1])
    q_deleted = float(deleted_residual * deleted_residual)
    delta_q = max(0.0, q_deleted - q_star)
    necessity_signal = delta_q / max(signal_norm2, epsilon_q)
    necessity_fit = delta_q / max(q_star, epsilon_q)
    rows: list[dict] = []
    t0 = time.time()
    path = profile_path_1d(
        Aw, bw, j, x_nnls, q_star, q_deleted, RHO_GRID, epsilon_q, max_bisect
    )
    elapsed = time.time() - t0
    for audit in path:
        rho, lower, upper = audit.pop("rho"), audit.pop("lower"), audit.pop("upper")
        rows.append(
            {
                "roi_id": roi_id, "candidate_index": j,
                "candidate_id": metadata[j]["candidate_id"], "lipid_name": metadata[j]["lipid_name"],
                "lipid_class": metadata[j]["lipid_class"], "adduct": metadata[j]["adduct"],
                "precursor_mz": metadata[j]["mz"], "weight_mode": weight_name,
                "rho": rho, "c_mode": "none_data_only", "selection_reason": ";".join(reasons),
                "x_net": float(x_net_roi[j]), "x_ref_nnls": float(x_nnls[j]),
                "x_ref_lasso003": float(x_lasso[j]), "lower": lower, "upper": upper,
                "absolute_width": upper - lower,
                "relative_width_vs_nnls": (upper - lower) / max(float(x_nnls[j]), 1e-12),
                "lower_fraction_vs_nnls": lower / max(float(x_nnls[j]), 1e-12),
                "q_star": q_star, "q_deleted": q_deleted, "necessity_signal": necessity_signal,
                "necessity_fit": necessity_fit, **audit,
                "elapsed_seconds_profile_path": elapsed,
            }
        )
    return rows


def group_endpoints_slsqp(
    A: np.ndarray, b: np.ndarray, members: list[int], x0: np.ndarray,
    threshold: float, maximize: bool, maxiter: int,
) -> tuple[float, bool, int, float, str, np.ndarray]:
    member_mask = np.zeros(A.shape[1], dtype=np.float64)
    member_mask[members] = 1.0

    def objective(x: np.ndarray) -> float:
        value = float(member_mask @ x)
        return -value if maximize else value

    def objective_jac(_x: np.ndarray) -> np.ndarray:
        return -member_mask if maximize else member_mask

    def constraint(x: np.ndarray) -> float:
        residual = A @ x - b
        return threshold - float(residual @ residual)

    def constraint_jac(x: np.ndarray) -> np.ndarray:
        return -2.0 * A.T @ (A @ x - b)

    out = minimize(
        objective, x0, jac=objective_jac, method="SLSQP",
        bounds=[(0.0, None)] * A.shape[1],
        constraints=[{"type": "ineq", "fun": constraint, "jac": constraint_jac}],
        options={"ftol": 1e-10, "maxiter": maxiter, "disp": False},
    )
    value = float(member_mask @ out.x)
    violation = max(0.0, -constraint(out.x))
    return value, bool(out.success), int(out.nit), violation, str(out.message), out.x


def profile_group_path(
    Aw: np.ndarray, bw: np.ndarray, group: dict, roi_id: str, weight_name: str,
    x_nnls: np.ndarray, q_star: float, signal_norm2: float,
    epsilon_q: float, maxiter: int,
) -> list[dict]:
    members = list(map(int, group["member_indices"].split(";")))
    keep = np.ones(Aw.shape[1], dtype=bool)
    keep[members] = False
    _, deleted_residual = nnls(Aw[:, keep], bw, maxiter=10 * Aw.shape[1])
    q_deleted_group = float(deleted_residual * deleted_residual)
    rho_zero_group = max(0.0, (q_deleted_group - q_star) / max(signal_norm2 + epsilon_q, epsilon_q))
    reference_sum = float(x_nnls[members].sum())
    lower_start = x_nnls.copy()
    upper_start = x_nnls.copy()
    rows: list[dict] = []
    for rho in RHO_GRID:
        threshold = q_star + rho * (signal_norm2 + epsilon_q)
        if q_deleted_group <= threshold:
            low = (0.0, True, 0, 0.0, "exact leave-group-out prune", lower_start)
            pruned = True
        else:
            low = group_endpoints_slsqp(Aw, bw, members, lower_start, threshold, False, maxiter)
            lower_start = low[5]
            pruned = False
        high = group_endpoints_slsqp(Aw, bw, members, upper_start, threshold, True, maxiter)
        upper_start = high[5]
        rows.append(
            {
                "roi_id": roi_id, "weight_mode": weight_name, "rho": rho,
                "c_mode": "none_data_only", "group_id": group["group_id"],
                "d_frag_cutoff": group["d_frag_cutoff"], "member_count": len(members),
                "member_indices": group["member_indices"], "members": group["members"],
                "x_ref_group_sum": reference_sum, "lower": low[0], "upper": high[0],
                "absolute_width": high[0] - low[0],
                "relative_width_vs_nnls": (high[0] - low[0]) / max(reference_sum, 1e-12),
                "lower_success": low[1], "upper_success": high[1],
                "lower_iterations": low[2], "upper_iterations": high[2],
                "lower_primal_violation": low[3], "upper_primal_violation": high[3],
                "lower_message": low[4], "upper_message": high[4],
                "q_deleted_group": q_deleted_group, "rho_zero_group_exact": rho_zero_group,
                "lower_zero_by_group_necessity_prune": pruned,
            }
        )
    return rows


def profile_group_row_parallel(
    Aw: np.ndarray, bw: np.ndarray, group: dict, roi_id: str, weight_name: str,
    rho: float, x_nnls: np.ndarray, q_star: float, signal_norm2: float,
    epsilon_q: float, maxiter: int, q_deleted_group: float,
) -> dict:
    members = list(map(int, group["member_indices"].split(";")))
    threshold = q_star + rho * (signal_norm2 + epsilon_q)
    rho_zero_group = max(0.0, (q_deleted_group - q_star) / max(signal_norm2 + epsilon_q, epsilon_q))
    if q_deleted_group <= threshold:
        low = (0.0, True, 0, 0.0, "exact leave-group-out prune", x_nnls)
        pruned = True
    else:
        low = group_endpoints_slsqp(Aw, bw, members, x_nnls, threshold, False, maxiter)
        pruned = False
    high = group_endpoints_slsqp(Aw, bw, members, x_nnls, threshold, True, maxiter)
    reference_sum = float(x_nnls[members].sum())
    return {
        "roi_id": roi_id, "weight_mode": weight_name, "rho": rho,
        "c_mode": "none_data_only", "group_id": group["group_id"],
        "d_frag_cutoff": group["d_frag_cutoff"], "member_count": len(members),
        "member_indices": group["member_indices"], "members": group["members"],
        "x_ref_group_sum": reference_sum, "lower": low[0], "upper": high[0],
        "absolute_width": high[0] - low[0],
        "relative_width_vs_nnls": (high[0] - low[0]) / max(reference_sum, 1e-12),
        "lower_success": low[1], "upper_success": high[1],
        "lower_iterations": low[2], "upper_iterations": high[2],
        "lower_primal_violation": low[3], "upper_primal_violation": high[3],
        "lower_message": low[4], "upper_message": high[4],
        "q_deleted_group": q_deleted_group, "rho_zero_group_exact": rho_zero_group,
        "lower_zero_by_group_necessity_prune": pruned,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="v47 profile-path identifiability audit for 758 DIA-MSI")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--roi-count", type=int, default=30)
    parser.add_argument("--roi-start", type=int, default=0, help="Inclusive index in the deterministic ROI manifest")
    parser.add_argument("--roi-stop", type=int, default=None, help="Exclusive index in the deterministic ROI manifest")
    parser.add_argument("--block-size", type=int, default=10)
    parser.add_argument("--top-net", type=int, default=10)
    parser.add_argument("--top-nnls", type=int, default=10)
    parser.add_argument("--top-redundant", type=int, default=12)
    parser.add_argument("--unstable-count", type=int, default=12)
    parser.add_argument("--max-bisect", type=int, default=22)
    parser.add_argument("--max-groups-per-roi", type=int, default=8)
    parser.add_argument("--group-slsqp-maxiter", type=int, default=400)
    parser.add_argument("--weight-modes", default="identity,auto")
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()

    started = time.time()
    data = args.data_dir.resolve()
    result = args.result_dir.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)

    A_path, B_path = data / "A_library.npy", data / "B_cube.npy"
    meta_path, mz_path = data / "candidate_metadata_final.jsonl", data / "shared_mz_final.npy"
    A = np.load(A_path).astype(np.float64)
    A = A[0] if A.ndim == 3 else A
    B = np.load(B_path).astype(np.float64)
    B = B[0]
    if B.shape[-1] != A.shape[0]:
        B = np.moveaxis(B, 0, -1)
    mask = np.load(data / "foreground_pixel_mask.npy").astype(bool)
    mz = np.load(mz_path).astype(np.float64).reshape(-1)
    metadata = load_jsonl(meta_path)
    X_net = np.load(result / "X_abundance.npy").astype(np.float64)
    if X_net.shape[0] != A.shape[1]:
        raise RuntimeError(f"X/A mismatch: {X_net.shape} vs {A.shape}")

    requested_modes = [item.strip() for item in args.weight_modes.split(",") if item.strip()]
    all_weights = {
        "identity": np.ones(A.shape[0], dtype=np.float64),
        "auto": auto_channel_weights(A, [row["lipid_class"] for row in metadata]),
    }
    weights = {name: all_weights[name] for name in requested_modes}

    roi_rows, roi_spectra = build_rois(B, mask, args.block_size, args.roi_count)
    write_rows(output / "roi_manifest.csv", roi_rows)
    print(f"selected {len(roi_rows)} observation-defined ROIs", flush=True)

    cone_rows, compact = cone_redundancy(A, mz, metadata, weights)
    write_rows(output / "cone_redundancy.csv", cone_rows)
    group_family = build_group_family(metadata, compact)
    write_rows(output / "competition_group_family.csv", group_family)
    print(f"completed cone audit: {len(cone_rows)} rows, {len(group_family)} groups", flush=True)

    stability_path = result.parent / "stability_758_v38" / "audit" / "lipid_stability_by_precursor_mz.csv"
    unstable_indices: list[int] = []
    if stability_path.exists():
        with stability_path.open(encoding="utf-8-sig") as handle:
            stability = list(csv.DictReader(handle))
        stability.sort(key=lambda row: (row["stability_status"] == "稳定", -float(row["exact_total_cv"] or 0)))
        unstable_indices = [int(row["candidate_index"]) for row in stability[: args.unstable_count]]

    profile_rows: list[dict] = []
    group_rows: list[dict] = []
    solver_rows: list[dict] = []
    epsilon_q = 1e-15
    roi_stop = len(roi_rows) if args.roi_stop is None else min(args.roi_stop, len(roi_rows))
    work_items = list(zip(roi_rows, roi_spectra))[args.roi_start:roi_stop]
    for roi_number, (roi, b_raw) in enumerate(work_items, 1):
        y0, y1, x0, x1 = (int(roi[k]) for k in ("y0", "y1", "x0", "x1"))
        local_mask = mask[y0:y1, x0:x1]
        x_net_roi = X_net[:, y0:y1, x0:x1][:, local_mask].mean(axis=1)
        for weight_name, w in weights.items():
            Aw = A * w[:, None]
            bw = b_raw * w
            x_nnls, nnls_residual = nnls(Aw, bw, maxiter=10 * A.shape[1])
            x_lasso = solve_nonnegative_lasso(Aw, bw, 0.003)
            q_star = float(nnls_residual * nnls_residual)
            signal_norm2 = float(bw @ bw)
            top_net = np.argsort(x_net_roi)[::-1][: args.top_net]
            top_nnls = np.argsort(x_nnls)[::-1][: args.top_nnls]
            redundant = sorted(range(A.shape[1]), key=lambda j: compact[(weight_name, j)][0])[: args.top_redundant]
            selected = sorted(set(map(int, top_net)) | set(map(int, top_nnls)) | set(redundant) | set(unstable_indices))
            selection_reason: dict[int, list[str]] = {j: [] for j in selected}
            for label, values in (("top_net", top_net), ("top_nnls", top_nnls), ("high_cone_redundancy", redundant), ("unstable", unstable_indices)):
                for j in values:
                    if int(j) in selection_reason:
                        selection_reason[int(j)].append(label)

            net_fit = Aw @ x_net_roi - bw
            lasso_fit = Aw @ x_lasso - bw
            solver_rows.append(
                {
                    "roi_id": roi["roi_id"], "weight_mode": weight_name,
                    "q_star_nnls": q_star, "signal_norm2": signal_norm2,
                    "relative_residual_nnls": math.sqrt(q_star / max(signal_norm2, epsilon_q)),
                    "relative_residual_net": float(np.linalg.norm(net_fit) / max(np.linalg.norm(bw), 1e-15)),
                    "relative_residual_lasso003": float(np.linalg.norm(lasso_fit) / max(np.linalg.norm(bw), 1e-15)),
                    "net_nnls_relative_x_difference": float(np.linalg.norm(x_net_roi - x_nnls) / max(np.linalg.norm(x_nnls), 1e-15)),
                    "lasso_nnls_relative_x_difference": float(np.linalg.norm(x_lasso - x_nnls) / max(np.linalg.norm(x_nnls), 1e-15)),
                    "nnls_active": int((x_nnls > 1e-8).sum()), "net_active": int((x_net_roi > 1e-8).sum()),
                    "profile_candidate_count": len(selected),
                }
            )

            candidate_batches = Parallel(n_jobs=args.workers, backend="loky", verbose=0)(
                delayed(profile_candidate_rows)(
                    Aw, bw, metadata, roi["roi_id"], weight_name, j, selection_reason[j],
                    x_net_roi, x_nnls, x_lasso, q_star, signal_norm2, epsilon_q, args.max_bisect,
                )
                for j in selected
            )
            for batch in candidate_batches:
                profile_rows.extend(batch)
            print(f"{roi['roi_id']} {weight_name}: profiled {len(selected)} candidates", flush=True)

            # Group endpoint pilot: profile only groups that intersect selected columns, prioritizing small groups.
            eligible_groups = [
                row for row in group_family
                if row["weight_mode"] == weight_name
                and any(int(i) in selected for i in row["member_indices"].split(";"))
            ]
            eligible_groups.sort(key=lambda row: (int(row["member_count"]), float(row["d_frag_cutoff"])))
            seen_members: set[tuple[int, ...]] = set()
            chosen_groups = []
            for row in eligible_groups:
                member_tuple = tuple(map(int, row["member_indices"].split(";")))
                if member_tuple not in seen_members:
                    chosen_groups.append(row)
                    seen_members.add(member_tuple)
                if len(chosen_groups) >= args.max_groups_per_roi:
                    break
            group_deleted_losses: dict[str, float] = {}
            for group in chosen_groups:
                members = list(map(int, group["member_indices"].split(";")))
                keep = np.ones(A.shape[1], dtype=bool)
                keep[members] = False
                _, deleted_residual = nnls(Aw[:, keep], bw, maxiter=10 * A.shape[1])
                group_deleted_losses[group["group_id"]] = float(deleted_residual * deleted_residual)
            group_batches = Parallel(n_jobs=args.workers, backend="loky", verbose=0)(
                delayed(profile_group_row_parallel)(
                    Aw, bw, group, roi["roi_id"], weight_name, rho, x_nnls,
                    q_star, signal_norm2, epsilon_q, args.group_slsqp_maxiter,
                    group_deleted_losses[group["group_id"]],
                )
                for group in chosen_groups for rho in RHO_GRID
            )
            group_rows.extend(group_batches)

        write_rows(output / "per_lipid_profile_path.csv", profile_rows)
        write_rows(output / "per_group_profile_path.csv", group_rows)
        write_rows(output / "solver_audit.csv", solver_rows)
        print(f"completed ROI {roi_number}/{len(work_items)} ({roi['roi_id']})", flush=True)

    report = {
        "definition": "v47 data-only profile-path pilot; no model training and no formal calibrated epsilon",
        "rho_grid": list(RHO_GRID), "cone_cutoffs": list(CONE_CUTOFFS),
        "roi_manifest_count": len(roi_rows), "roi_start": args.roi_start, "roi_stop": roi_stop,
        "processed_roi_count": len(work_items), "profile_rows": len(profile_rows), "group_rows": len(group_rows),
        "weight_modes": requested_modes, "c_modes_completed": ["none_data_only"],
        "warning": "Intervals are near-optimal feasible intervals, not formal confidence intervals. Risk calibration is deferred to v48/v49.",
        "hashes": {"A": file_sha256(A_path), "B": file_sha256(B_path), "metadata": file_sha256(meta_path)},
        "elapsed_seconds": time.time() - started,
    }
    (output / "v47_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
