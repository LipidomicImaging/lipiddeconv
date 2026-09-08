#!/usr/bin/env python3
"""Run the deliberately easy v50 per-case reoptimized ISTA sanity tests.

This runner never loads a checkpoint.  Each learned case constructs a fresh
12-layer LipidENNet and optimizes it only against that case's synthetic B.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scipy.optimize import nnls


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

DATA_RELATIVE = Path(
    "adapter_pipeline/outputs/"
    "v38_ce29_fragmentionfixed_profile_overlap_empiricalfwhm_globalq99_ista_ready"
)
OUTPUT_DEFAULT = ROOT / "results/v50_reoptimized_sanity_k1_k3"
TARGET_B_P50 = 0.6036783456802368
REPORT_GATE = 1.0e-3
SENSITIVITY_GATE = 1.0e-4
EXPECTED_A_SHAPE = (1084, 391)
K3_POOL_FRACTION = 0.10
LOSS_NAMES = ("reconstruction", "ssim", "tv", "anchor")
CFG_DEFAULTS = {
    "seed": 42,
    "n_epochs": 5000,
    "lr_net": 1.0e-4,
    "grad_clip_norm": 1.0,
    "K_layers": 12,
    "warmup_epochs": 500,
    "calib_clamp_min": 0.5,
    "calib_clamp_max": 1.5,
    "early_stop_min_epochs": 800,
    "early_stop_check_freq": 25,
    "early_stop_patience": 12,
    "early_stop_loss_rtol": 2.0e-4,
    "early_stop_x_rtol": 5.0e-4,
    "loss_channel_weighting": "auto",
}


def atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    temporary.replace(path)


def json_number(value: float) -> float | None:
    value = float(value)
    return value if np.isfinite(value) else None


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def asset_paths(asset_root: Path) -> dict[str, Path]:
    data = asset_root / DATA_RELATIVE
    return {
        "A_library": data / "A_library.npy",
        "B_cube": data / "B_cube.npy",
        "foreground_mask": data / "foreground_pixel_mask.npy",
        "candidate_metadata": data / "candidate_metadata_final.npy",
        "channel_axis": data / "shared_mz_final.npy",
    }


def validate_assets(paths: dict[str, Path]) -> dict:
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise RuntimeError("MISSING_DEPENDENCY: " + "; ".join(missing))

    lock_path = ROOT / "results/v48_production_ista_lock/v48_production_ista_lock.json"
    if not lock_path.exists():
        raise RuntimeError(f"MISSING_DEPENDENCY: {lock_path}")
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    expected = lock["hashes_sha256"]
    lock_names = {
        "A_library": "A_library",
        "B_cube": "B_cube",
        "candidate_metadata": "candidate_metadata",
        "channel_axis": "channel_axis",
    }
    actual_hashes = {}
    for asset_name, hash_name in lock_names.items():
        actual_hashes[asset_name] = sha256(paths[asset_name])
        if actual_hashes[asset_name] != expected[hash_name]:
            raise RuntimeError(f"STAGE0_LOCK_MISMATCH: {asset_name}")
    for source_name, hash_name in {
        "src/run_758_ista.py": "production_main_script",
        "src/lipid_ista.py": "production_solver_implementation",
        "src/config_758.py": "production_config",
    }.items():
        path = ROOT / source_name
        if sha256(path) != expected[hash_name]:
            raise RuntimeError(f"STAGE0_LOCK_MISMATCH: {source_name}")
    return {"production_lock": str(lock_path), "validated_hashes": actual_hashes}


def load_library_and_metadata(paths: dict[str, Path]) -> tuple[np.ndarray, dict]:
    A = np.load(paths["A_library"]).astype(np.float64)
    if A.ndim == 3:
        A = A[0]
    metadata = np.load(paths["candidate_metadata"], allow_pickle=True).item()
    if A.shape != EXPECTED_A_SHAPE:
        raise RuntimeError(f"STAGE0_LOCK_MISMATCH: A shape {A.shape}")
    if len(metadata["candidate_id"]) != A.shape[1]:
        raise RuntimeError("STAGE0_LOCK_MISMATCH: candidate ordering")
    return A, metadata


def candidate_record(index: int, scores: np.ndarray, metadata: dict) -> dict:
    return {
        "candidate_index": int(index),
        "candidate_id": str(metadata["candidate_id"][index]),
        "lipid_name": str(metadata["lipid_name"][index]),
        "lipid_class": str(metadata["lipid_class"][index]),
        "cone_isolation": float(scores[index]),
    }


def compute_design(A: np.ndarray, metadata: dict, paths: dict[str, Path]) -> dict:
    scores = np.full(A.shape[1], np.nan, dtype=np.float64)
    for index in range(A.shape[1]):
        norm = np.linalg.norm(A[:, index])
        if not np.isfinite(norm) or norm <= 0:
            continue
        _, residual = nnls(np.delete(A, index, axis=1), A[:, index])
        scores[index] = residual / norm
    finite = np.flatnonzero(np.isfinite(scores))
    if finite.size < 3:
        raise RuntimeError("EASY_CASE_NOT_IDENTIFIABLE: fewer than three finite cone scores")
    ranking = sorted(finite.tolist(), key=lambda j: (-scores[j], j))
    k1_index = ranking[0]

    pool_size = max(3, int(math.ceil(K3_POOL_FRACTION * len(ranking))))
    pool = ranking[:pool_size]
    norms = np.linalg.norm(A, axis=0)
    cosine = (A.T @ A) / np.maximum(norms[:, None] * norms[None, :], 1.0e-30)

    def triplet_key(indices: tuple[int, int, int]) -> tuple:
        similarities = [cosine[a, b] for a, b in itertools.combinations(indices, 2)]
        return (
            max(similarities),
            float(np.mean(similarities)),
            -min(scores[list(indices)]),
            -float(np.sum(scores[list(indices)])),
            indices,
        )

    k3_indices = min(itertools.combinations(pool, 3), key=triplet_key)
    pairs = []
    for left, right in itertools.combinations(k3_indices, 2):
        pairs.append({
            "candidate_index_left": int(left),
            "candidate_index_right": int(right),
            "cosine_similarity": float(cosine[left, right]),
        })
    return {
        "status": "DESIGN_READY",
        "asset_paths": {key: str(value) for key, value in paths.items()},
        "candidate_count": int(A.shape[1]),
        "channel_count": int(A.shape[0]),
        "cone_isolation_definition": "min_{z>=0} ||A_-j z - A_j||_2 / ||A_j||_2 (scipy.optimize.nnls)",
        "ranking_rule": "descending finite cone-isolation score, then ascending candidate index",
        "k1_easy": [candidate_record(k1_index, scores, metadata)],
        "k3_selection_rule": {
            "pool": f"top ceil({K3_POOL_FRACTION} * n_finite) cone-isolated candidates",
            "pool_size": pool_size,
            "triplet_order": "minimize maximum pairwise cosine, then mean cosine, then maximize minimum and summed cone-isolation, then lexicographic indices",
        },
        "k3_easy": [candidate_record(j, scores, metadata) for j in k3_indices],
        "k3_pairwise_cosine": pairs,
        "synthetic_contract": {
            "spatial_template": "foreground-masked per-pixel channel L2 norm of frozen real B, normalized to foreground median 1",
            "k1_relative_abundance": [1],
            "k3_relative_abundance": [1, 1, 1],
            "shared_template_across_k3_members": True,
            "global_scale_target": {
                "statistic": "median foreground B_sim channel-L2 norm",
                "value": TARGET_B_P50,
            },
            "forward_model": "B_sim = A_library @ X_true",
            "residual": 0.0,
            "noise": 0.0,
        },
        "training_contract": training_contract(),
    }


def training_contract() -> dict:
    return {
        "network": "fresh lipid_ista.LipidENNet per case",
        "unfolded_ista_layers": 12,
        "checkpoint_loads": "forbidden; none",
        "training_inputs": ["frozen normalized A_library", "case B_sim"],
        "X_true_used_by_optimization": False,
        "losses": list(LOSS_NAMES),
        "loss_source": "exact production train (1).ipynb compute_advanced_losses semantics",
        "loss_balancer": "fresh exact production AutomaticWeightedLoss(num_losses=4)",
        "optimizer": {"name": "AdamW", "lr_net": CFG_DEFAULTS["lr_net"], "lr_loss_balancer": 1.0e-3, "weight_decay": 1.0e-4},
        "scheduler": {"name": "CosineAnnealingLR", "T_max": CFG_DEFAULTS["n_epochs"], "eta_min": 1.0e-6},
        "channel_weighting": CFG_DEFAULTS["loss_channel_weighting"],
        "gradient_clip_norm": CFG_DEFAULTS["grad_clip_norm"],
        "warmup_epochs": CFG_DEFAULTS["warmup_epochs"],
        "epoch_ceiling": CFG_DEFAULTS["n_epochs"],
        "early_stop": {
            "minimum_epochs": CFG_DEFAULTS["early_stop_min_epochs"],
            "check_frequency": CFG_DEFAULTS["early_stop_check_freq"],
            "patience_checks": CFG_DEFAULTS["early_stop_patience"],
            "physical_loss_rtol": CFG_DEFAULTS["early_stop_loss_rtol"],
            "abundance_rtol": CFG_DEFAULTS["early_stop_x_rtol"],
        },
    }


def load_real_spatial_assets(paths: dict[str, Path], A: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    B = np.load(paths["B_cube"]).astype(np.float32)
    if B.shape == (1, 200, 90, A.shape[0]):
        B = np.moveaxis(B[0], -1, 0)
    elif B.shape == (1, A.shape[0], 200, 90):
        B = B[0]
    else:
        raise RuntimeError(f"STAGE0_LOCK_MISMATCH: B shape {B.shape}")
    mask = np.load(paths["foreground_mask"]).astype(bool)
    if mask.shape != B.shape[1:]:
        raise RuntimeError(f"STAGE0_LOCK_MISMATCH: foreground mask shape {mask.shape}")
    return B, mask


def synthesize_case(
    case_name: str,
    active_indices: list[int],
    A: np.ndarray,
    B_real: np.ndarray,
    mask: np.ndarray,
) -> dict:
    spatial = np.linalg.norm(B_real.astype(np.float64), axis=0)
    spatial[~mask] = 0.0
    spatial_median = float(np.median(spatial[mask]))
    if not np.isfinite(spatial_median) or spatial_median <= 0:
        raise RuntimeError("MISSING_DEPENDENCY: invalid real-B foreground spatial template")
    spatial /= spatial_median

    active = np.asarray(active_indices, dtype=int)
    base_spectrum = A[:, active].sum(axis=1)
    unscaled_norm_median = float(np.median(np.linalg.norm(base_spectrum[:, None] * spatial[mask], axis=0)))
    global_scale = TARGET_B_P50 / unscaled_norm_median

    X_true = np.zeros((A.shape[1], *mask.shape), dtype=np.float32)
    X_true[active] = (global_scale * spatial).astype(np.float32)
    B_sim = np.einsum("ml,lhw->mhw", A.astype(np.float32), X_true, optimize=True)
    B_check = (A.astype(np.float32) @ X_true.reshape(A.shape[1], -1)).reshape(B_sim.shape)
    forward_error = float(np.max(np.abs(B_check - B_sim)))
    achieved = float(np.median(np.linalg.norm(B_sim[:, mask].astype(np.float64), axis=0)))
    return {
        "case_name": case_name,
        "active_indices": active.tolist(),
        "spatial_template": spatial.astype(np.float32),
        "global_scale": float(global_scale),
        "X_true": X_true,
        "B_sim": B_sim,
        "foreground_mask": mask,
        "forward_consistency_max_abs": forward_error,
        "target_foreground_B_l2_p50": TARGET_B_P50,
        "achieved_foreground_B_l2_p50": achieved,
    }


def support_metrics(recovered: np.ndarray, true_indices: list[int], gate: float) -> dict:
    reported = set(np.flatnonzero(recovered > gate).tolist())
    truth = set(map(int, true_indices))
    tp = len(reported & truth)
    fp = len(reported - truth)
    fn = len(truth - reported)
    return {
        "gate": gate,
        "recovered_support": sorted(reported),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": tp / (tp + fn) if tp + fn else 0.0,
    }


def oracle_case(case: dict, A: np.ndarray, metadata: dict) -> dict:
    mean_spectrum = case["B_sim"][:, case["foreground_mask"]].mean(axis=1).astype(np.float64)
    recovered, residual_norm = nnls(A, mean_spectrum)
    metrics = support_metrics(recovered, case["active_indices"], REPORT_GATE)
    denominator = max(float(np.linalg.norm(mean_spectrum)), 1.0e-30)
    clean = metrics["fp"] == 0 and metrics["fn"] == 0
    return {
        "case_name": case["case_name"],
        "status": "PASS" if clean else "EASY_CASE_NOT_IDENTIFIABLE",
        "active_indices": case["active_indices"],
        "global_scale": case["global_scale"],
        "target_foreground_B_l2_p50": case["target_foreground_B_l2_p50"],
        "achieved_foreground_B_l2_p50": case["achieved_foreground_B_l2_p50"],
        "forward_consistency_max_abs": case["forward_consistency_max_abs"],
        "reconstruction_relative_residual": float(residual_norm / denominator),
        **metrics,
        "recovered_support_details": [
            {
                "candidate_index": int(j),
                "candidate_id": str(metadata["candidate_id"][j]),
                "lipid_name": str(metadata["lipid_name"][j]),
                "abundance": float(recovered[j]),
            }
            for j in metrics["recovered_support"]
        ],
    }


# Exact numerical definitions used by the production notebook cell.
def create_window(torch, window_size: int, channels: int, device):
    gaussian = torch.tensor([
        math.exp(-((x - window_size // 2) ** 2) / float(2 * 1.5**2))
        for x in range(window_size)
    ])
    gaussian = gaussian / gaussian.sum()
    window_2d = gaussian.unsqueeze(1).mm(gaussian.unsqueeze(0)).float().unsqueeze(0).unsqueeze(0)
    return window_2d.expand(channels, 1, window_size, window_size).contiguous().to(device)


def ssim_loss(torch, F, image1, image2, window_size: int = 11):
    channels = image1.shape[1]
    window = create_window(torch, window_size, channels, image1.device)
    mu1 = F.conv2d(image1, window, padding=window_size // 2, groups=channels)
    mu2 = F.conv2d(image2, window, padding=window_size // 2, groups=channels)
    mu1_sq, mu2_sq, mu12 = mu1.pow(2), mu2.pow(2), mu1 * mu2
    sigma1_sq = F.conv2d(image1**2, window, padding=window_size // 2, groups=channels) - mu1_sq
    sigma2_sq = F.conv2d(image2**2, window, padding=window_size // 2, groups=channels) - mu2_sq
    sigma12 = F.conv2d(image1 * image2, window, padding=window_size // 2, groups=channels) - mu12
    c1, c2 = 0.01**2, 0.03**2
    ssim_map = ((2 * mu12 + c1) * (2 * sigma12 + c2)) / (
        (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)
    )
    return 1.0 - ssim_map.mean()


def compute_advanced_losses(torch, F, X_layers, B_obs, A_curr, A_init, fragment_weights=None):
    X_final = X_layers[-1]
    B_hat = torch.einsum("ml,blhw->bmhw", A_curr, X_final)
    residual = torch.abs(B_hat - B_obs)
    if fragment_weights is not None:
        residual = residual * fragment_weights.view(1, -1, 1, 1)
    l_rec = residual.mean()
    norm_factor = B_obs.max().clamp(min=1.0e-8)
    l_ssim = ssim_loss(torch, F, B_hat / norm_factor, B_obs / norm_factor)
    h_x, w_x = X_final.shape[2], X_final.shape[3]
    h_tv = (X_final[:, :, 1:, :] - X_final[:, :, : h_x - 1, :]).pow(2).mean()
    w_tv = (X_final[:, :, :, 1:] - X_final[:, :, :, : w_x - 1]).pow(2).mean()
    l_tv = h_tv + w_tv
    l_anchor = F.mse_loss(A_curr, A_init)
    return torch.stack([l_rec, l_ssim, l_tv, l_anchor]), B_hat


def make_automatic_weighted_loss(torch):
    class AutomaticWeightedLoss(torch.nn.Module):
        def __init__(self, num_losses: int):
            super().__init__()
            self.params = torch.nn.Parameter(torch.zeros(num_losses))

        def forward(self, loss_list):
            total_loss = 0.0
            weights = []
            for index, loss in enumerate(loss_list):
                s = torch.clamp(self.params[index], min=-9.0, max=10.0)
                precision = 0.5 * torch.exp(-s)
                total_loss += precision * loss + 0.5 * s
                weights.append(precision.item())
            return total_loss, weights

    return AutomaticWeightedLoss


def production_weight_namespace(A_init, metadata: dict, torch, device) -> dict:
    def build_fragment_weight_vector_from_A(A_tensor):
        A_np = A_tensor.detach().cpu().numpy()
        present = np.abs(A_np) > 1.0e-8
        counts = present.sum(axis=1)
        rarity = np.empty(A_np.shape[0], dtype=np.float32)
        rarity[counts <= 3] = 5.0
        rarity[(counts > 3) & (counts <= 10)] = 3.0
        rarity[(counts > 10) & (counts <= 30)] = 1.5
        rarity[counts > 30] = 0.5
        classes = np.asarray(metadata["lipid_class"], dtype=str)
        specificity = np.ones(A_np.shape[0], dtype=np.float32)
        for channel in range(A_np.shape[0]):
            columns = np.flatnonzero(present[channel])
            if columns.size == 0:
                continue
            _, class_counts = np.unique(classes[columns], return_counts=True)
            dominant_fraction = class_counts.max() / class_counts.sum()
            specificity[channel] = 4.0 if dominant_fraction > 0.90 else (2.0 if dominant_fraction > 0.70 else 1.0)
        raw = rarity * specificity
        weights = np.clip(raw / (raw.mean() + 1.0e-12), 0.2, 5.0).astype(np.float32)
        return torch.as_tensor(weights, device=device)

    return {"build_fragment_weight_vector_from_A": build_fragment_weight_vector_from_A}


def runtime_cfg(paths: dict[str, Path], Cfg):
    return SimpleNamespace(
        loss_channel_weighting=Cfg.loss_channel_weighting,
        meta_path=str(paths["candidate_metadata"]),
        mz_path=str(paths["channel_axis"]),
        parent_channel_mz_range=Cfg.parent_channel_mz_range,
        parent_channel_weight_multiplier=Cfg.parent_channel_weight_multiplier,
    )


def train_fresh_case(case: dict, A: np.ndarray, metadata: dict, paths: dict[str, Path], output: Path, device_name: str) -> dict:
    import torch
    import torch.nn.functional as F

    from config_758 import Cfg
    from lipid_ista import LipidENNet
    from run_758_ista import build_channel_weights
    from utils import set_seed

    if device_name.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("MISSING_DEPENDENCY: CUDA requested but unavailable")
    device = torch.device(device_name)
    set_seed(Cfg.seed)
    output.mkdir(parents=True, exist_ok=True)
    A_init = torch.as_tensor(A, dtype=torch.float32, device=device)
    A_init = A_init / (torch.linalg.vector_norm(A_init, dim=0, keepdim=True) + 1.0e-8)
    B = torch.as_tensor(case["B_sim"][None], dtype=torch.float32, device=device)

    # Fresh objects only.  There is deliberately no checkpoint/resume branch.
    net = LipidENNet(
        A_init=A_init,
        K=Cfg.K_layers,
        clamp_min=Cfg.calib_clamp_min,
        clamp_max=Cfg.calib_clamp_max,
    ).to(device)
    AutomaticWeightedLoss = make_automatic_weighted_loss(torch)
    loss_balancer = AutomaticWeightedLoss(num_losses=4).to(device)
    namespace = production_weight_namespace(A_init, metadata, torch, device)
    fragment_weights, channel_weight_report = build_channel_weights(namespace, A_init, runtime_cfg(paths, Cfg))
    optimizer = torch.optim.AdamW(
        [
            {"params": net.parameters(), "lr": Cfg.lr_net},
            {"params": loss_balancer.parameters(), "lr": 1.0e-3},
        ],
        weight_decay=1.0e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Cfg.n_epochs, eta_min=1.0e-6
    )

    history = {
        "epoch": [], "train_total": [], "eval_total": [],
        "eval_physical_loss": [], "eval_raw_losses": [],
        "relative_eval_loss_change": [], "relative_abundance_change": [],
        "stable_checks": [], "automatic_loss_weights": [],
    }
    previous_eval_loss = None
    previous_eval_x = None
    stable_checks = 0
    stop_reason = "max_epochs"
    stopped_epoch = Cfg.n_epochs
    started = time.time()

    for epoch in range(1, Cfg.n_epochs + 1):
        net.train()
        net.calibrator.W.requires_grad = epoch >= Cfg.warmup_epochs
        optimizer.zero_grad(set_to_none=True)
        _, X_list, A_curr, _ = net(B)
        raw_losses, _ = compute_advanced_losses(torch, F, X_list, B, A_curr, A_init, fragment_weights)
        total_loss, _ = loss_balancer(raw_losses)
        if not torch.isfinite(total_loss):
            stop_reason = "nonfinite_loss"
            stopped_epoch = epoch
            break
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), Cfg.grad_clip_norm)
        optimizer.step()
        scheduler.step()

        if epoch % Cfg.early_stop_check_freq == 0 or epoch == 1:
            net.eval()
            with torch.no_grad():
                X_eval, X_eval_list, A_eval, _ = net(B)
                eval_raw, _ = compute_advanced_losses(
                    torch, F, X_eval_list, B, A_eval, A_init, fragment_weights
                )
                eval_total_tensor, _ = loss_balancer(eval_raw)
                eval_total = float(eval_total_tensor.item())
                physical = float(eval_raw.sum().item())
                if previous_eval_loss is None:
                    loss_change = float("inf")
                    x_change = float("inf")
                else:
                    loss_change = abs(physical - previous_eval_loss) / max(abs(previous_eval_loss), 1.0e-12)
                    x_change = float(
                        torch.linalg.vector_norm(X_eval - previous_eval_x)
                        / torch.clamp(torch.linalg.vector_norm(previous_eval_x), min=1.0e-12)
                    )
                if (
                    epoch >= Cfg.early_stop_min_epochs
                    and loss_change <= Cfg.early_stop_loss_rtol
                    and x_change <= Cfg.early_stop_x_rtol
                ):
                    stable_checks += 1
                else:
                    stable_checks = 0
                history["epoch"].append(epoch)
                history["train_total"].append(float(total_loss.item()))
                history["eval_total"].append(eval_total)
                history["eval_physical_loss"].append(physical)
                history["eval_raw_losses"].append([float(value) for value in eval_raw.detach().cpu()])
                history["relative_eval_loss_change"].append(json_number(loss_change))
                history["relative_abundance_change"].append(json_number(x_change))
                history["stable_checks"].append(stable_checks)
                weights = 0.5 * torch.exp(-torch.clamp(loss_balancer.params, min=-9.0, max=10.0))
                history["automatic_loss_weights"].append([float(value) for value in weights.detach().cpu()])
                previous_eval_loss = physical
                previous_eval_x = X_eval.detach().clone()
            print(
                f"epoch={epoch} train={total_loss.item():.7g} eval_phys={physical:.7g} "
                f"eval_awl={eval_total:.7g} dloss={loss_change:.3g} dX={x_change:.3g} "
                f"stable={stable_checks}/{Cfg.early_stop_patience}", flush=True,
            )
            if stable_checks >= Cfg.early_stop_patience:
                stop_reason = "converged"
                stopped_epoch = epoch
                break

    torch.save(net.state_dict(), output / "latest_model.pth")
    atomic_write_json(output / "training_history.json", history)
    net.eval()
    with torch.no_grad():
        X_hat, X_layers, A_final, _ = net(B)
        final_raw, B_hat = compute_advanced_losses(
            torch, F, X_layers, B, A_final, A_init, fragment_weights
        )
    elapsed = time.time() - started
    X_hat_np = X_hat[0].detach().cpu().numpy().astype(np.float32)
    B_hat_np = B_hat[0].detach().cpu().numpy().astype(np.float32)
    return {
        "X_hat": X_hat_np,
        "B_hat": B_hat_np,
        "stopped_epoch": stopped_epoch,
        "stop_reason": stop_reason,
        "wall_time_seconds": elapsed,
        "final_physical_loss": float(final_raw.sum().item()),
        "final_raw_losses": [float(value) for value in final_raw.detach().cpu()],
        "channel_weighting": channel_weight_report,
    }


def learned_report(case: dict, learned: dict, metadata: dict) -> dict:
    mask = case["foreground_mask"]
    x_true_mean = case["X_true"][:, mask].mean(axis=1).astype(np.float64)
    x_hat_mean = learned["X_hat"][:, mask].mean(axis=1).astype(np.float64)
    np.save(learned["output"] / "X_true_foreground_mean.npy", x_true_mean.astype(np.float32))
    np.save(learned["output"] / "X_hat_foreground_mean.npy", x_hat_mean.astype(np.float32))
    identity = support_metrics(x_hat_mean, case["active_indices"], REPORT_GATE)
    true_mask = np.zeros(len(x_hat_mean), dtype=bool)
    true_mask[case["active_indices"]] = True
    total_mass = float(x_hat_mean.sum())
    reported_mask = x_hat_mean > REPORT_GATE
    reported_mass = float(x_hat_mean[reported_mask].sum())
    false_indices = np.flatnonzero(~true_mask)
    false_order = false_indices[np.argsort(-x_hat_mean[false_indices])[:10]]
    abundance_errors = []
    for index in case["active_indices"]:
        abundance_errors.append({
            "candidate_index": int(index),
            "candidate_id": str(metadata["candidate_id"][index]),
            "lipid_name": str(metadata["lipid_name"][index]),
            "X_true_foreground_mean": float(x_true_mean[index]),
            "X_hat_foreground_mean": float(x_hat_mean[index]),
            "relative_error": float(abs(x_hat_mean[index] - x_true_mean[index]) / max(abs(x_true_mean[index]), 1.0e-30)),
        })
    rec_rel = float(
        np.linalg.norm(learned["B_hat"][:, mask] - case["B_sim"][:, mask])
        / max(np.linalg.norm(case["B_sim"][:, mask]), 1.0e-30)
    )
    passed = identity["fp"] == 0 and identity["fn"] == 0
    return {
        "status": "PASS_LOW_COMPLEXITY_IDENTITY_SANITY" if passed else "FAIL_LOW_COMPLEXITY_IDENTITY_SANITY",
        "case_name": case["case_name"],
        "active_indices": case["active_indices"],
        "identity_at_1e-3": identity,
        "reconstruction_relative_residual": rec_rel,
        "abundance_relative_error": abundance_errors,
        "all_output_false_allocation_mass": float(x_hat_mean[~true_mask].sum() / max(total_mass, 1.0e-30)),
        "reported_only_false_allocation_mass": float(x_hat_mean[reported_mask & ~true_mask].sum() / max(reported_mass, 1.0e-30)),
        "n_outputs_gt_1e-4": int(np.sum(x_hat_mean > SENSITIVITY_GATE)),
        "n_outputs_gt_1e-3": int(np.sum(reported_mask)),
        "top_10_false_allocations": [
            {
                "candidate_index": int(index),
                "candidate_id": str(metadata["candidate_id"][index]),
                "lipid_name": str(metadata["lipid_name"][index]),
                "X_hat_foreground_mean": float(x_hat_mean[index]),
            }
            for index in false_order
        ],
        "forward_consistency_max_abs": case["forward_consistency_max_abs"],
        "spatial_template_construction": "foreground-masked real-B per-pixel channel L2 norm, normalized to foreground median 1",
        "global_scale": case["global_scale"],
        "target_foreground_B_l2_p50": case["target_foreground_B_l2_p50"],
        "achieved_foreground_B_l2_p50": case["achieved_foreground_B_l2_p50"],
        "final_training_epoch": learned["stopped_epoch"],
        "final_physical_loss": learned["final_physical_loss"],
        "final_raw_losses": dict(zip(LOSS_NAMES, learned["final_raw_losses"])),
        "stop_reason": learned["stop_reason"],
        "wall_time_seconds": learned["wall_time_seconds"],
        "channel_weighting": learned["channel_weighting"],
        "checkpoint_loaded": False,
        "X_true_used_in_training": False,
    }


def prepare(args):
    paths = asset_paths(args.asset_root.resolve())
    validation = validate_assets(paths)
    A, metadata = load_library_and_metadata(paths)
    design = compute_design(A, metadata, paths)
    design["asset_validation"] = validation
    atomic_write_json(args.output_dir / "design.json", design)
    return paths, A, metadata, design


def selected_indices(design: dict, case_name: str) -> list[int]:
    return [row["candidate_index"] for row in design[f"{case_name}_easy"]]


def print_summary(payload: dict) -> None:
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--asset-root", type=Path,
        default=ROOT.parent / "decon-lipid",
        help="Root containing the frozen v38 production assets",
    )
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DEFAULT)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--oracle-only", action="store_true")
    mode.add_argument("--case", choices=("k1", "k3"))
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir = args.output_dir.resolve()
    paths, A, metadata, design = prepare(args)
    if args.dry_run:
        print_summary({
            "status": "DRY_RUN_READY",
            "asset_paths": design["asset_paths"],
            "k1_easy": design["k1_easy"],
            "k3_easy": design["k3_easy"],
            "k3_pairwise_cosine": design["k3_pairwise_cosine"],
            "synthetic_contract": design["synthetic_contract"],
            "training_contract": design["training_contract"],
            "gpu_work_performed": False,
        })
        return

    B_real, mask = load_real_spatial_assets(paths, A)
    requested_cases = ("k1", "k3") if args.oracle_only else (args.case,)
    oracle_reports = []
    prepared_cases = {}
    for case_name in requested_cases:
        case = synthesize_case(
            case_name, selected_indices(design, case_name), A, B_real, mask
        )
        prepared_cases[case_name] = case
        oracle_reports.append(oracle_case(case, A, metadata))
    oracle_payload = {
        "status": (
            "PASS" if all(row["status"] == "PASS" for row in oracle_reports)
            else "EASY_CASE_NOT_IDENTIFIABLE"
        ),
        "cases": oracle_reports,
    }
    atomic_write_json(args.output_dir / "oracle_report.json", oracle_payload)
    print_summary(oracle_payload)
    if oracle_payload["status"] != "PASS":
        raise SystemExit("EASY_CASE_NOT_IDENTIFIABLE")
    if args.oracle_only:
        return

    case = prepared_cases[args.case]
    case_output = args.output_dir / f"{args.case}_easy"
    learned = train_fresh_case(case, A, metadata, paths, case_output, args.device)
    learned["output"] = case_output
    report = learned_report(case, learned, metadata)
    atomic_write_json(case_output / "report.json", report)
    print_summary(report)


if __name__ == "__main__":
    main()
