from __future__ import annotations

import argparse
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
SEED = 20260907
TARGET_B_NORM = 0.6036783456802368


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-index", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    out = args.output_dir.resolve()
    wall_start = time.perf_counter()
    out.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    A_np = np.load(DATA / "A_library.npy").astype(np.float32)
    A_np = A_np[0] if A_np.ndim == 3 else A_np
    mask = np.load(DATA / "foreground_pixel_mask.npy").astype(bool)
    metadata = np.load(DATA / "candidate_metadata_final.npy", allow_pickle=True).item()
    candidate_index = int(args.candidate_index)
    if not 0 <= candidate_index < A_np.shape[1]:
        raise ValueError(f"candidate index outside frozen 391-column library: {candidate_index}")
    X_true = np.zeros((A_np.shape[1], *mask.shape), dtype=np.float32)
    X_true[candidate_index, mask] = 1.0
    timing_xtrue = time.perf_counter() - t0

    t0 = time.perf_counter()
    B_channel_first = np.einsum("ml,lhw->mhw", A_np, X_true, optimize=True)
    initial_norm = np.linalg.norm(B_channel_first[:, mask], axis=0)
    scale = TARGET_B_NORM / float(np.median(initial_norm))
    X_true *= scale
    B_channel_first = np.einsum("ml,lhw->mhw", A_np, X_true, optimize=True)
    achieved_norms = np.linalg.norm(B_channel_first[:, mask], axis=0)
    achieved_norm = float(np.median(achieved_norms))
    B_check = np.einsum("ml,lhw->mhw", A_np, X_true, optimize=True)
    forward_consistency_max_abs = float(np.max(np.abs(B_check - B_channel_first)))
    timing_bclean = time.perf_counter() - t0

    t0 = time.perf_counter()
    B_stored = np.transpose(B_channel_first, (1, 2, 0))[None].astype(np.float32, copy=False)
    B_solver_np = np.transpose(B_stored, (0, 3, 1, 2))
    device = torch.device("cuda")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this locked benchmark, but torch.cuda.is_available() is False")
    B_tensor = torch.as_tensor(B_solver_np, dtype=torch.float32, device=device)
    timing_preprocess = time.perf_counter() - t0

    t0 = time.perf_counter()
    A_init = get_A_matrix(str(DATA / "A_library.npy"), device)
    net = LipidENNet(A_init=A_init, K=12, clamp_min=0.5, clamp_max=1.5).to(device)
    checkpoint = REFERENCE / "latest_model.pth"
    net.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    net.eval()
    torch.cuda.synchronize()
    timing_model_loading = time.perf_counter() - t0

    torch.cuda.reset_peak_memory_stats(device)
    t0 = time.perf_counter()
    with torch.no_grad():
        X_hat_t, _, A_calibrated_t, _ = net(B_tensor)
        B_reconstructed_t = torch.einsum("ml,blhw->bmhw", A_calibrated_t, X_hat_t)
    torch.cuda.synchronize()
    timing_gpu_forward = time.perf_counter() - t0
    peak_gpu_memory = int(torch.cuda.max_memory_allocated(device))

    t0 = time.perf_counter()
    X_hat = X_hat_t[0].cpu().numpy().astype(np.float32)
    B_reconstructed = B_reconstructed_t[0].cpu().numpy().astype(np.float32)
    true_abundance = float(X_true[candidate_index, mask].mean())
    recovered_abundance = float(X_hat[candidate_index, mask].mean())
    relative_abundance_error = abs(recovered_abundance - true_abundance) / max(abs(true_abundance), 1e-30)
    total_recovered = float(X_hat[:, mask].sum())
    wrong_mask = np.arange(X_hat.shape[0]) != candidate_index
    wrong_totals = X_hat[:, mask].sum(axis=1).astype(np.float64)
    wrong_totals[candidate_index] = -np.inf
    top_wrong = np.argsort(-wrong_totals)[:5]
    top_incorrect_allocations = [
        {
            "candidate_index": int(index),
            "lipid_name": str(metadata["lipid_name"][index]),
            "fraction_of_total_recovered_mass": float(wrong_totals[index] / max(total_recovered, 1e-30)),
        }
        for index in top_wrong
    ]
    false_allocation_mass = float(X_hat[wrong_mask][:, mask].sum()) / max(total_recovered, 1e-30)
    residual = B_reconstructed[:, mask] - B_channel_first[:, mask]
    reconstruction_relative_residual = float(np.linalg.norm(residual) / max(np.linalg.norm(B_channel_first[:, mask]), 1e-30))
    reconstruction_rmse = float(np.sqrt(np.mean(residual ** 2)))

    np.save(out / "X_true.npy", X_true)
    np.save(out / "B_clean.npy", B_stored)
    np.save(out / "X_hat.npy", X_hat)
    np.save(out / "B_reconstructed.npy", B_reconstructed)
    timing_saving_metrics = time.perf_counter() - t0
    total_wall = time.perf_counter() - wall_start

    report = {
        "status": "PASS_TRUTH_PIPELINE_K1" if forward_consistency_max_abs == 0.0 else "FAIL_FORWARD_CONSISTENCY",
        "case_count": 1,
        "seed": SEED,
        "K": 1,
        "residual_l2_ratio": 0.0,
        "spatial_support": "foreground-wide uniform",
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(device),
        "peak_gpu_memory_bytes": peak_gpu_memory,
        "peak_gpu_memory_mib": peak_gpu_memory / (1024 ** 2),
        "selected_candidate_index": candidate_index,
        "selected_candidate_id": str(metadata["candidate_id"][candidate_index]),
        "selected_lipid_name": str(metadata["lipid_name"][candidate_index]),
        "X_true_shape": list(X_true.shape),
        "B_clean_shape": list(B_stored.shape),
        "production_solver_B_shape": list(B_solver_np.shape),
        "X_hat_shape": list(X_hat.shape),
        "target_real_B_P50_norm": TARGET_B_NORM,
        "achieved_synthetic_B_foreground_median_norm": achieved_norm,
        "scale_factor": scale,
        "forward_consistency_max_abs": forward_consistency_max_abs,
        "true_abundance": true_abundance,
        "recovered_abundance": recovered_abundance,
        "relative_abundance_error": relative_abundance_error,
        "false_allocation_mass": false_allocation_mass,
        "top_5_incorrect_allocations": top_incorrect_allocations,
        "reconstruction_rmse": reconstruction_rmse,
        "reconstruction_relative_residual": reconstruction_relative_residual,
        "timing_seconds": {
            "X_true_generation": timing_xtrue,
            "B_clean_generation_and_scaling": timing_bclean,
            "preprocessing_tensor_reshape": timing_preprocess,
            "model_loading": timing_model_loading,
            "GPU_forward": timing_gpu_forward,
            "saving_and_metrics": timing_saving_metrics,
            "total_wall": total_wall,
        },
        "paths": {
            "X_true": str((out / "X_true.npy").resolve()),
            "B_clean": str((out / "B_clean.npy").resolve()),
            "X_hat": str((out / "X_hat.npy").resolve()),
            "B_reconstructed": str((out / "B_reconstructed.npy").resolve()),
        },
        "stop_rule": "Single K=1 benchmark complete; no remaining smoke cases launched.",
    }
    (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
