import os
from pathlib import Path

import torch


class Cfg:
    project_dir = Path(__file__).resolve().parent
    default_data_dir = (
        project_dir / "adapter_pipeline" / "outputs"
        / "v27_ce30_mammalian_prior_canonical_diacyl_inference_peak_masked_ista_ready"
    )
    data_dir = Path(os.environ.get("ISTA_DATA_DIR", str(default_data_dir)))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed = int(os.environ.get("ISTA_SEED", "42"))

    ce = 758
    le = 3
    ri = 0
    Windows = (748, 798)

    processed_A_path = os.environ.get("ISTA_A_PATH", str(data_dir / "A_library.npy"))
    processed_B_path = os.environ.get("ISTA_B_PATH", str(data_dir / "B_cube.npy"))
    mz_path = str(data_dir / "shared_mz_final.npy")
    meta_path = os.environ.get(
        "ISTA_META_PATH", str(data_dir / "candidate_metadata_final.npy")
    )

    out_dir = os.environ.get(
        "ISTA_OUT_DIR",
        str(project_dir / "results_758_v27_ce30_mammalian_prior_joint_earlystop"),
    )

    # Optional region-balanced reconstruction-loss ablation. M0 candidates are
    # in 748-798; retained higher isotope ranks can extend to about 803.
    parent_channel_weight_multiplier = float(
        os.environ.get("ISTA_PARENT_WEIGHT_MULTIPLIER", "1.0")
    )
    loss_channel_weighting = os.environ.get(
        "ISTA_LOSS_CHANNEL_WEIGHTING", "auto"
    ).strip().lower()
    parent_channel_mz_range = (748.0, 803.0)

    # This is the optimizer epoch ceiling, not the number of unfolded ISTA
    # steps.  Actual ISTA depth is K_layers below.  Training now stops early
    # when both the objective and abundance maps have stabilized.
    n_epochs = int(os.environ.get("ISTA_EPOCHS", "5000"))
    batch_size = 1
    lr_net = 1e-4
    lr_weight = 5e-4
    grad_clip_norm = 1.0

    K_layers = int(os.environ.get("ISTA_K_LAYERS", "12"))
    full_image_shape = (200, 90)

    warmup_epochs = int(os.environ.get("ISTA_WARMUP_EPOCHS", "500"))
    calib_clamp_min = float(os.environ.get("ISTA_CALIB_CLAMP_MIN", "0.5"))
    calib_clamp_max = float(os.environ.get("ISTA_CALIB_CLAMP_MAX", "1.5"))

    vis_freq = int(os.environ.get("ISTA_VIS_FREQ", "100"))
    save_freq = int(os.environ.get("ISTA_SAVE_FREQ", "500"))

    early_stop_min_epochs = int(os.environ.get("ISTA_EARLY_MIN_EPOCHS", "800"))
    early_stop_check_freq = int(os.environ.get("ISTA_EARLY_CHECK_FREQ", "25"))
    early_stop_patience = int(os.environ.get("ISTA_EARLY_PATIENCE", "12"))
    early_stop_loss_rtol = float(os.environ.get("ISTA_EARLY_LOSS_RTOL", "2e-4"))
    early_stop_x_rtol = float(os.environ.get("ISTA_EARLY_X_RTOL", "5e-4"))
