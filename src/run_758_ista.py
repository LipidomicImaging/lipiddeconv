from __future__ import annotations

import argparse
import csv
import json
import os
import time
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parent


def build_channel_weights(namespace: dict, A_init: torch.Tensor, cfg) -> tuple[torch.Tensor, dict]:
    weighting_mode = getattr(cfg, "loss_channel_weighting", "auto")
    weight_reference_path = os.environ.get("ISTA_WEIGHT_REFERENCE_A_PATH", "").strip()
    weight_reference_meta_path = os.environ.get(
        "ISTA_WEIGHT_REFERENCE_META_PATH", ""
    ).strip()
    weight_reference = A_init
    if weight_reference_path:
        reference_np = np.load(weight_reference_path)
        if reference_np.ndim == 3:
            reference_np = reference_np[0]
        if reference_np.shape[0] != A_init.shape[0]:
            raise RuntimeError(
                f"weight-reference channel mismatch: {reference_np.shape} vs {tuple(A_init.shape)}"
            )
        weight_reference = torch.as_tensor(
            reference_np, dtype=A_init.dtype, device=A_init.device
        )
    if weighting_mode == "none":
        weights = torch.ones(
            A_init.shape[0], dtype=A_init.dtype, device=A_init.device
        )
    elif weighting_mode == "auto":
        original_meta_path = getattr(cfg, "meta_path", None)
        if weight_reference_meta_path:
            cfg.meta_path = weight_reference_meta_path
        try:
            weights = namespace["build_fragment_weight_vector_from_A"](weight_reference)
        finally:
            if original_meta_path is not None:
                cfg.meta_path = original_meta_path
    else:
        raise ValueError(
            "ISTA_LOSS_CHANNEL_WEIGHTING must be either 'auto' or 'none'"
        )
    mz_axis = np.asarray(np.load(cfg.mz_path), dtype=float).reshape(-1)
    low, high = cfg.parent_channel_mz_range
    parent_mask_np = (mz_axis >= low) & (mz_axis <= high)
    if len(parent_mask_np) != len(weights):
        raise RuntimeError(
            f"channel-weight shape mismatch: mz={len(parent_mask_np)}, weights={len(weights)}"
        )
    multiplier = float(cfg.parent_channel_weight_multiplier)
    if multiplier <= 0:
        raise ValueError("ISTA_PARENT_WEIGHT_MULTIPLIER must be positive")
    if multiplier != 1.0:
        parent_mask = torch.as_tensor(parent_mask_np, device=weights.device)
        weights = weights.clone()
        weights[parent_mask] *= multiplier
        weights /= weights.mean().clamp_min(1.0e-12)
        weights = torch.clamp(weights, 0.2, 5.0)
        weights /= weights.mean().clamp_min(1.0e-12)
    parent_values = weights[parent_mask_np]
    fragment_values = weights[~parent_mask_np]
    report = {
        "mode": weighting_mode,
        "reference_A_path": weight_reference_path or "solver_A",
        "reference_A_shape": list(weight_reference.shape),
        "reference_meta_path": weight_reference_meta_path or "solver_metadata",
        "parent_weight_multiplier_before_renormalization": multiplier,
        "parent_mz_range": [low, high],
        "parent_channel_count": int(parent_mask_np.sum()),
        "fragment_channel_count": int((~parent_mask_np).sum()),
        "mean_parent_channel_weight": float(parent_values.mean().item()),
        "mean_fragment_channel_weight": float(fragment_values.mean().item()),
        "mean_all_channel_weight": float(weights.mean().item()),
    }
    return weights, report


def load_training_namespace() -> dict:
    """Load only the ISTA cell, excluding the notebook's NNLS/ElasticNet cell."""
    notebook_path = ROOT / "train (1).ipynb"
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    source = "".join(notebook["cells"][0]["source"])
    namespace = {"__name__": "ista_758_training_cell"}
    exec(compile(source, str(notebook_path) + "#cell0", "exec"), namespace)
    return namespace


def benchmark(namespace: dict, crop_height: int, crop_width: int) -> None:
    from config_758 import Cfg
    from lipid_ista import LipidENNet
    from utils import get_A_matrix

    device = Cfg.device
    A_init = get_A_matrix(Cfg.processed_A_path, device)
    B_np = np.load(Cfg.processed_B_path)
    if B_np.ndim == 4 and B_np.shape[-1] == A_init.shape[0]:
        B_np = np.transpose(B_np, (0, 3, 1, 2))
    B = torch.as_tensor(B_np).float().to(device)
    if crop_height > 0 and crop_width > 0:
        B = B[:, :, :crop_height, :crop_width]

    net = LipidENNet(
        A_init=A_init,
        K=Cfg.K_layers,
        clamp_min=Cfg.calib_clamp_min,
        clamp_max=Cfg.calib_clamp_max,
    ).to(device)
    fragment_weights, channel_weight_report = build_channel_weights(namespace, A_init, Cfg)
    optimizer = torch.optim.AdamW(net.parameters(), lr=Cfg.lr_net, weight_decay=1e-4)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    started = time.time()
    optimizer.zero_grad(set_to_none=True)
    X_final, X_list, A_curr, _ = net(B)
    losses, _ = namespace["compute_advanced_losses"](
        X_list, B, A_curr, A_init, fragment_weights=fragment_weights
    )
    total = losses.sum()
    total.backward()
    optimizer.step()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.time() - started

    report = {
        "device": str(device),
        "A_shape": list(A_init.shape),
        "B_shape": list(B.shape),
        "K_layers": Cfg.K_layers,
        "elapsed_seconds_one_step": elapsed,
        "losses": [float(value) for value in losses.detach().cpu()],
        "total_loss_unweighted": float(total.detach().cpu()),
        "peak_cuda_memory_gib": (
            torch.cuda.max_memory_allocated() / 1024**3 if torch.cuda.is_available() else None
        ),
        "output_finite": bool(torch.isfinite(X_final).all().item()),
        "channel_weighting": channel_weight_report,
    }
    output = Path(Cfg.out_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "benchmark.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


def train_early_stopping(namespace: dict) -> None:
    """Train the unfolded ISTA network with deterministic convergence checks.

    ``Cfg.K_layers`` is the number of ISTA steps represented by the network.
    Optimizer epochs only learn the step sizes, thresholds, attention and the
    bounded spectral calibration.  Therefore a large fixed epoch count is not
    itself a meaningful stopping rule.
    """
    from config_758 import Cfg
    from lipid_ista import LipidENNet
    from utils import get_A_matrix, set_seed

    set_seed(Cfg.seed)
    device = Cfg.device
    output = Path(Cfg.out_dir)
    output.mkdir(parents=True, exist_ok=True)

    A_init = get_A_matrix(Cfg.processed_A_path, device)
    B_np = np.load(Cfg.processed_B_path)
    if B_np.ndim == 4 and B_np.shape[-1] == A_init.shape[0]:
        B_np = np.transpose(B_np, (0, 3, 1, 2))
    B = torch.as_tensor(B_np).float().to(device)

    net = LipidENNet(
        A_init=A_init,
        K=Cfg.K_layers,
        clamp_min=Cfg.calib_clamp_min,
        clamp_max=Cfg.calib_clamp_max,
    ).to(device)
    resume_checkpoint = os.environ.get("ISTA_RESUME_CHECKPOINT", "")
    if resume_checkpoint:
        net.load_state_dict(
            torch.load(resume_checkpoint, map_location=device, weights_only=True)
        )

    loss_balancer = namespace["AutomaticWeightedLoss"](num_losses=4).to(device)
    fragment_weights, channel_weight_report = build_channel_weights(namespace, A_init, Cfg)
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

    history: dict[str, list] = {
        "epoch": [],
        "train_total": [],
        "eval_total": [],
        "eval_physical_loss": [],
        "eval_raw_losses": [],
        "relative_eval_loss_change": [],
        "relative_abundance_change": [],
        "stable_checks": [],
    }
    previous_eval_loss: float | None = None
    previous_eval_x: torch.Tensor | None = None
    stable_checks = 0
    stop_reason = "max_epochs"
    stopped_epoch = Cfg.n_epochs
    started = time.time()

    for epoch in range(1, Cfg.n_epochs + 1):
        net.train()
        net.calibrator.W.requires_grad = epoch >= Cfg.warmup_epochs
        optimizer.zero_grad(set_to_none=True)
        _, X_list, A_curr, _ = net(B)
        raw_losses, _ = namespace["compute_advanced_losses"](
            X_list, B, A_curr, A_init, fragment_weights=fragment_weights
        )
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
                eval_raw, _ = namespace["compute_advanced_losses"](
                    X_eval_list, B, A_eval, A_init,
                    fragment_weights=fragment_weights,
                )
                eval_total_tensor, _ = loss_balancer(eval_raw)
                eval_total = float(eval_total_tensor.item())
                # AutomaticWeightedLoss contains learned log-variance terms;
                # its scalar can keep drifting (and become negative) even
                # after the physical reconstruction has stabilized.  Use the
                # unweighted physical objective for convergence instead.
                eval_physical_loss = float(eval_raw.sum().item())
                if previous_eval_loss is None:
                    loss_change = float("inf")
                    x_change = float("inf")
                else:
                    loss_change = abs(eval_physical_loss - previous_eval_loss) / max(
                        abs(previous_eval_loss), 1.0e-12
                    )
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
                history["eval_physical_loss"].append(eval_physical_loss)
                history["eval_raw_losses"].append(
                    [float(value) for value in eval_raw.detach().cpu()]
                )
                history["relative_eval_loss_change"].append(loss_change)
                history["relative_abundance_change"].append(x_change)
                history["stable_checks"].append(stable_checks)

                previous_eval_loss = eval_physical_loss
                previous_eval_x = X_eval.detach().clone()

            print(
                f"epoch={epoch} train={total_loss.item():.7g} "
                f"eval_phys={eval_physical_loss:.7g} "
                f"eval_awl={eval_total:.7g} dloss={loss_change:.3g} "
                f"dX={x_change:.3g} stable={stable_checks}/"
                f"{Cfg.early_stop_patience}",
                flush=True,
            )
            if stable_checks >= Cfg.early_stop_patience:
                stop_reason = "converged"
                stopped_epoch = epoch
                break

        if epoch % Cfg.save_freq == 0:
            torch.save(net.state_dict(), output / "latest_model.pth")

    torch.save(net.state_dict(), output / "latest_model.pth")
    np.save(output / "training_history.npy", history, allow_pickle=True)
    serializable_history = {
        key: [None if isinstance(v, float) and not np.isfinite(v) else v for v in values]
        for key, values in history.items()
    }
    (output / "training_history.json").write_text(
        json.dumps(serializable_history, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    report = {
        "device": str(device),
        "A_shape": list(A_init.shape),
        "B_shape": list(B.shape),
        "K_layers_actual_ista_steps": Cfg.K_layers,
        "optimizer_epoch_ceiling": Cfg.n_epochs,
        "stopped_epoch": stopped_epoch,
        "stop_reason": stop_reason,
        "early_stop": {
            "min_epochs": Cfg.early_stop_min_epochs,
            "check_frequency": Cfg.early_stop_check_freq,
            "patience_checks": Cfg.early_stop_patience,
            "loss_relative_tolerance": Cfg.early_stop_loss_rtol,
            "abundance_relative_tolerance": Cfg.early_stop_x_rtol,
        },
        "elapsed_seconds": time.time() - started,
        "final_eval_total": history["eval_total"][-1] if history["eval_total"] else None,
        "final_eval_physical_loss": (
            history["eval_physical_loss"][-1]
            if history["eval_physical_loss"] else None
        ),
        "final_relative_loss_change": (
            serializable_history["relative_eval_loss_change"][-1]
            if history["relative_eval_loss_change"] else None
        ),
        "final_relative_abundance_change": (
            serializable_history["relative_abundance_change"][-1]
            if history["relative_abundance_change"] else None
        ),
        "channel_weighting": channel_weight_report,
    }
    (output / "training_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


def export_results() -> None:
    from config_758 import Cfg
    from lipid_ista import LipidENNet
    from utils import get_A_matrix

    device = Cfg.device
    output = Path(Cfg.out_dir)
    checkpoint = output / "latest_model.pth"
    if not checkpoint.exists():
        raise FileNotFoundError(f"Missing trained checkpoint: {checkpoint}")

    A_init = get_A_matrix(Cfg.processed_A_path, device)
    B_np = np.load(Cfg.processed_B_path)
    if B_np.ndim == 4 and B_np.shape[-1] == A_init.shape[0]:
        B_np = np.transpose(B_np, (0, 3, 1, 2))
    B = torch.as_tensor(B_np).float().to(device)
    net = LipidENNet(
        A_init=A_init,
        K=Cfg.K_layers,
        clamp_min=Cfg.calib_clamp_min,
        clamp_max=Cfg.calib_clamp_max,
    ).to(device)
    net.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    net.eval()
    with torch.no_grad():
        X, _, A_calibrated, _ = net(B)
        B_reconstructed = torch.einsum("ml,blhw->bmhw", A_calibrated, X)
        residual = B_reconstructed - B

    X_np = X[0].cpu().numpy().astype(np.float32)
    A_np = A_calibrated.cpu().numpy().astype(np.float32)
    B_rec_np = B_reconstructed[0].cpu().numpy().astype(np.float32)
    residual_np = residual[0].cpu().numpy().astype(np.float32)
    np.save(output / "X_abundance.npy", X_np)
    np.save(output / "A_calibrated.npy", A_np)
    np.save(output / "B_reconstructed.npy", B_rec_np)
    np.save(output / "B_residual.npy", residual_np)

    metadata = np.load(Cfg.meta_path, allow_pickle=True).item()
    total = X_np.sum(axis=(1, 2))
    mean = X_np.mean(axis=(1, 2))
    maximum = X_np.max(axis=(1, 2))
    active_fraction = (X_np > 1.0e-6).mean(axis=(1, 2))
    order = np.argsort(-total)
    with (output / "candidate_abundance_summary.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "rank",
                "candidate_index",
                "candidate_id",
                "lipid_name",
                "lipid_class",
                "isotope_model",
                "total_abundance",
                "mean_abundance",
                "max_abundance",
                "active_pixel_fraction",
            ]
        )
        for rank, index in enumerate(order, start=1):
            writer.writerow(
                [
                    rank,
                    int(index),
                    metadata["candidate_id"][index],
                    metadata["lipid_name"][index],
                    metadata["lipid_class"][index],
                    "joint_retained_isotope_ranks",
                    float(total[index]),
                    float(mean[index]),
                    float(maximum[index]),
                    float(active_fraction[index]),
                ]
            )
    report = {
        "checkpoint": str(checkpoint),
        "X_shape": list(X_np.shape),
        "A_calibrated_shape": list(A_np.shape),
        "B_reconstructed_shape": list(B_rec_np.shape),
        "reconstruction_rmse": float(np.sqrt(np.mean(residual_np**2))),
        "reconstruction_mae": float(np.mean(np.abs(residual_np))),
        "finite": bool(
            np.isfinite(X_np).all()
            and np.isfinite(A_np).all()
            and np.isfinite(B_rec_np).all()
        ),
    }
    (output / "export_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=("benchmark", "train", "train-earlystop", "export"),
        default="benchmark",
    )
    parser.add_argument("--crop-height", type=int, default=0)
    parser.add_argument("--crop-width", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    os.chdir(ROOT)
    args = parse_args()
    if args.mode == "benchmark":
        namespace = load_training_namespace()
        benchmark(namespace, args.crop_height, args.crop_width)
    elif args.mode == "train":
        namespace = load_training_namespace()
        namespace["train_single_sample"]()
    elif args.mode == "train-earlystop":
        namespace = load_training_namespace()
        train_early_stopping(namespace)
    else:
        export_results()


if __name__ == "__main__":
    main()
