from __future__ import annotations

import csv
import json
import time
from pathlib import Path

import numpy as np
import torch

from lipid_ista import LipidENNet
from utils import get_A_matrix


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "adapter_pipeline/outputs/v38_ce29_fragmentionfixed_profile_overlap_empiricalfwhm_globalq99_ista_ready"
REFERENCE = ROOT / "results_758_v38_ce29_empiricalfwhm_globalq99_fixed_library_joint_earlystop"
V47 = ROOT / "profile_758_v47_30roi_data_only"
OUT = ROOT / "v48_pilot_48"
SEED = 20260907
K_VALUES = (13, 25, 55, 103)
RESIDUAL_RATIOS = (0.0, 0.10)
SELECTION_MODES = ("random", "hard_competition")
REPLICATES = 3
TARGET_B_NORM = 0.6036783456802368
ACTIVE_THRESHOLD = 1.0e-4


def load_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    case_dir = OUT / "cases"
    case_dir.mkdir(exist_ok=True)
    A = np.load(DATA / "A_library.npy").astype(np.float32)
    A = A[0] if A.ndim == 3 else A
    B_real = np.load(DATA / "B_cube.npy").astype(np.float32)[0]
    mask = np.load(DATA / "foreground_pixel_mask.npy").astype(bool)
    X_prod = np.load(REFERENCE / "X_abundance.npy").astype(np.float32)
    metadata = np.load(DATA / "candidate_metadata_final.npy", allow_pickle=True).item()
    measured = np.moveaxis(B_real, -1, 0).reshape(A.shape[0], -1)
    real_x = X_prod.reshape(A.shape[1], -1)
    empirical_residual = measured[:, mask.reshape(-1)] - A @ real_x[:, mask.reshape(-1)]
    empirical_residual = empirical_residual[:, np.linalg.norm(empirical_residual, axis=0) > 1e-12]
    X_fg = X_prod[:, mask].T

    group_rows = [
        row for row in load_csv(V47 / "competition_group_family.csv")
        if row["weight_mode"] == "identity" and abs(float(row["d_frag_cutoff"]) - 0.02) < 1e-12
    ]
    hard_pool = np.asarray(sorted({int(i) for row in group_rows for i in row["member_indices"].split(";")}), dtype=int)
    all_indices = np.arange(A.shape[1], dtype=int)

    device = torch.device("cuda")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    A_init = get_A_matrix(str(DATA / "A_library.npy"), device)
    net = LipidENNet(A_init=A_init, K=12, clamp_min=0.5, clamp_max=1.5).to(device)
    net.load_state_dict(torch.load(REFERENCE / "latest_model.pth", map_location=device, weights_only=True))
    net.eval()

    manifest_rows: list[dict] = []
    lipid_rows: list[dict] = []
    case_id = 0
    for k in K_VALUES:
        eligible_profiles = np.flatnonzero((X_fg > ACTIVE_THRESHOLD).sum(axis=1) >= k)
        if eligible_profiles.size == 0:
            raise RuntimeError(f"No frozen empirical rank profile supports K={k}")
        for residual_ratio in RESIDUAL_RATIOS:
            for selection_mode in SELECTION_MODES:
                for replicate in range(REPLICATES):
                    started = time.perf_counter()
                    rng = np.random.default_rng(SEED + case_id)
                    if selection_mode == "random":
                        selected = np.sort(rng.choice(all_indices, size=k, replace=False))
                    else:
                        take_hard = min(k, hard_pool.size)
                        selected_hard = rng.choice(hard_pool, size=take_hard, replace=False)
                        if take_hard < k:
                            remaining = np.setdiff1d(all_indices, selected_hard, assume_unique=False)
                            selected = np.sort(np.concatenate([selected_hard, rng.choice(remaining, size=k-take_hard, replace=False)]))
                        else:
                            selected = np.sort(selected_hard)

                    profile_index = int(rng.choice(eligible_profiles))
                    weights = np.sort(X_fg[profile_index][X_fg[profile_index] > ACTIVE_THRESHOLD])[::-1][:k].astype(np.float64)
                    weights /= weights.sum()
                    weights = weights[rng.permutation(k)]
                    clean_vector = A[:, selected] @ weights.astype(np.float32)
                    scale = TARGET_B_NORM / max(float(np.linalg.norm(clean_vector)), 1e-30)
                    true_values = (weights * scale).astype(np.float32)
                    clean_vector = A[:, selected] @ true_values
                    B_sim = np.zeros((A.shape[0], *mask.shape), dtype=np.float32)
                    B_sim[:, mask] = clean_vector[:, None]
                    residual_sources = np.full(mask.sum(), -1, dtype=np.int32)
                    if residual_ratio > 0:
                        residual_sources = rng.choice(empirical_residual.shape[1], size=mask.sum(), replace=True).astype(np.int32)
                        sampled = empirical_residual[:, residual_sources].copy()
                        sampled /= np.maximum(np.linalg.norm(sampled, axis=0, keepdims=True), 1e-15)
                        sampled *= residual_ratio * np.linalg.norm(B_sim[:, mask], axis=0, keepdims=True)
                        B_sim[:, mask] = np.maximum(B_sim[:, mask] + sampled, 0.0)

                    B_tensor = torch.as_tensor(B_sim[None], dtype=torch.float32, device=device)
                    torch.cuda.synchronize()
                    forward_start = time.perf_counter()
                    with torch.no_grad():
                        X_hat_t, _, A_cal_t, _ = net(B_tensor)
                        B_hat_t = torch.einsum("ml,blhw->bmhw", A_cal_t, X_hat_t)
                    torch.cuda.synchronize()
                    forward_seconds = time.perf_counter() - forward_start
                    X_hat = X_hat_t[0].cpu().numpy().astype(np.float32)
                    B_hat = B_hat_t[0].cpu().numpy().astype(np.float32)
                    xhat_mean = X_hat[:, mask].mean(axis=1).astype(np.float64)
                    xtrue_mean = np.zeros(A.shape[1], dtype=np.float64)
                    xtrue_mean[selected] = true_values
                    total_hat = max(float(xhat_mean.sum()), 1e-30)
                    true_mask = xtrue_mean > 0
                    false_mass = float(xhat_mean[~true_mask].sum() / total_hat)
                    rec_rel = float(np.linalg.norm(B_hat[:, mask] - B_sim[:, mask]) / max(np.linalg.norm(B_sim[:, mask]), 1e-30))
                    case_name = f"case_{case_id:03d}_K{k}_{selection_mode}_r{int(residual_ratio*100):02d}_rep{replicate}"
                    this_dir = case_dir / case_name
                    this_dir.mkdir(exist_ok=True)
                    np.save(this_dir / "X_hat.npy", X_hat)
                    np.savez_compressed(
                        this_dir / "truth_spec.npz",
                        selected_indices=selected,
                        true_abundances=true_values,
                        residual_source_indices=residual_sources,
                        scale_factor=np.asarray(scale),
                        seed=np.asarray(SEED + case_id),
                    )
                    for index in range(A.shape[1]):
                        truth = float(xtrue_mean[index])
                        estimate = float(xhat_mean[index])
                        lipid_rows.append({
                            "case_id": case_id,
                            "case_name": case_name,
                            "candidate_index": index,
                            "candidate_id": str(metadata["candidate_id"][index]),
                            "lipid_name": str(metadata["lipid_name"][index]),
                            "is_true": int(truth > 0),
                            "X_true": truth,
                            "X_hat": estimate,
                            "absolute_error": abs(estimate-truth),
                            "relative_error": abs(estimate-truth)/truth if truth > 0 else "",
                            "false_allocation_fraction": estimate/total_hat if truth == 0 else 0.0,
                            "case_false_allocation_mass": false_mass,
                            "reconstruction_relative_residual": rec_rel,
                        })
                    manifest_rows.append({
                        "case_id": case_id,
                        "case_name": case_name,
                        "K": k,
                        "residual_ratio": residual_ratio,
                        "selection_mode": selection_mode,
                        "replicate": replicate,
                        "seed": SEED + case_id,
                        "empirical_profile_pixel_index": profile_index,
                        "selected_indices": ";".join(map(str, selected)),
                        "target_B_norm": TARGET_B_NORM,
                        "clean_B_norm": float(np.linalg.norm(clean_vector)),
                        "false_allocation_mass": false_mass,
                        "reconstruction_relative_residual": rec_rel,
                        "GPU_forward_seconds": forward_seconds,
                        "case_wall_seconds": time.perf_counter()-started,
                        "X_hat_path": str((this_dir / "X_hat.npy").resolve()),
                        "truth_spec_path": str((this_dir / "truth_spec.npz").resolve()),
                    })
                    with (OUT / "v48_pilot_manifest.csv").open("w", encoding="utf-8-sig", newline="") as handle:
                        writer = csv.DictWriter(handle, fieldnames=list(manifest_rows[0]))
                        writer.writeheader(); writer.writerows(manifest_rows)
                    with (OUT / "v48_pilot_lipid_observations_partial.csv").open("w", encoding="utf-8-sig", newline="") as handle:
                        writer = csv.DictWriter(handle, fieldnames=list(lipid_rows[0]))
                        writer.writeheader(); writer.writerows(lipid_rows)
                    print(json.dumps({"completed": case_id+1, "case": case_name, "forward_s": forward_seconds, "false_mass": false_mass, "rec_rel": rec_rel}), flush=True)
                    case_id += 1

    report = {
        "status": "PASS_48_PRODUCTION_ISTA_CASES_COMPLETE_CERTIFICATES_PENDING",
        "case_count": len(manifest_rows),
        "K": list(K_VALUES),
        "residual_ratios": list(RESIDUAL_RATIOS),
        "selection_modes": list(SELECTION_MODES),
        "replicates": REPLICATES,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device),
        "certificate_features_added": False,
    }
    (OUT / "production_cases_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
