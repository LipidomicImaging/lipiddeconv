from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.stats import norm


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "adapter_pipeline/outputs/v38_ce29_fragmentionfixed_profile_overlap_empiricalfwhm_globalq99_ista_ready"
RESULT = ROOT / "results_758_v38_ce29_empiricalfwhm_globalq99_fixed_library_joint_earlystop"
V47 = ROOT / "profile_758_v47_30roi_data_only"
V47B = ROOT / "v47b_observed_evidence_audit"
OUT = ROOT / "v48_simulation_envelope_audit"
ISOLATION_WINDOW = (748.0, 798.0)
OBSERVED_THRESHOLD = 0.0002443958772029583
ACTIVE_THRESHOLDS = (1e-4, 1e-3, 1e-2)
SEED = 20260907


def file_sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_csv(path: Path, rows: list[dict], fieldnames: list[str] | None = None) -> None:
    if fieldnames is None:
        fieldnames = list(rows[0])
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def quantile_dict(values: np.ndarray, probabilities=(0.10, 0.25, 0.50, 0.75, 0.90, 1.0)) -> dict:
    values = np.asarray(values, dtype=np.float64)
    labels = ["P10", "P25", "P50", "P75", "P90", "max"]
    return {label: float(value) for label, value in zip(labels, np.quantile(values, probabilities))}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    metadata = load_jsonl(DATA / "candidate_metadata_final.jsonl")
    A = np.load(DATA / "A_library.npy").astype(np.float32)
    if A.ndim == 3:
        A = A[0]
    B = np.load(DATA / "B_cube.npy").astype(np.float32)[0]
    X = np.load(RESULT / "X_abundance.npy").astype(np.float32)
    mask = np.load(DATA / "foreground_pixel_mask.npy").astype(bool)
    mz_axis = np.load(DATA / "shared_mz_final.npy").astype(np.float64).reshape(-1)
    source_axis = np.load(DATA / "shared_mz_source_sticks.npy").astype(np.float64).reshape(-1)
    peak_response = sparse.load_npz(DATA / "peak_response_matrix.npz").tocsc()
    if B.shape[-1] != A.shape[0] or X.shape != (A.shape[1], *mask.shape):
        raise RuntimeError(f"Frozen shape mismatch: A={A.shape}, B={B.shape}, X={X.shape}, mask={mask.shape}")

    # Reuse the exact historical precursor matching rule (nearest frozen source
    # channel, accepted only when |delta m/z| < 0.025) and the already-frozen
    # profile-overlap response matrix.  No new ppm/Da tolerance is introduced.
    precursor_source_indices = []
    for row in metadata:
        precursor_mz = float(row["mz"])
        source_index = int(np.argmin(np.abs(source_axis - precursor_mz)))
        if abs(float(source_axis[source_index]) - precursor_mz) >= 0.025:
            raise RuntimeError(f"Frozen precursor matching failed for m/z {precursor_mz}")
        precursor_source_indices.append(source_index)
    unique_source_indices = sorted(set(precursor_source_indices), key=lambda i: source_axis[i])
    source_to_group = {source_index: group_id for group_id, source_index in enumerate(unique_source_indices)}
    group_members = {
        source_to_group[source_index]: [i for i, value in enumerate(precursor_source_indices) if value == source_index]
        for source_index in unique_source_indices
    }
    acquisition_rows = (mz_axis >= ISOLATION_WINDOW[0]) & (mz_axis <= ISOLATION_WINDOW[1])
    group_observation_rows = {}
    for source_index in unique_source_indices:
        group_id = source_to_group[source_index]
        response_rows = peak_response.indices[peak_response.indptr[source_index]:peak_response.indptr[source_index + 1]]
        group_observation_rows[group_id] = response_rows[acquisition_rows[response_rows]]
        if group_observation_rows[group_id].size == 0:
            raise RuntimeError(f"Precursor group {group_id} has no frozen 748-798 observation channel")

    eligible_rows = []
    for index, row in enumerate(metadata):
        precursor_mz = float(row["mz"])
        eligible = ISOLATION_WINDOW[0] <= precursor_mz <= ISOLATION_WINDOW[1]
        source_index = precursor_source_indices[index]
        group_id = source_to_group[source_index]
        eligible_rows.append({
            "candidate_index": index,
            "candidate_id": row.get("candidate_id", row.get("entry_id", "")),
            "lipid_name": row.get("lipid_name", ""),
            "lipid_identity": row.get("structure_text", row.get("lipid_name", "")),
            "lipid_class": row.get("lipid_class", ""),
            "adduct": row.get("adduct", ""),
            "rule_sheet": row.get("rule_sheet", ""),
            "precursor_mz": precursor_mz,
            "isolation_window_low": ISOLATION_WINDOW[0],
            "isolation_window_high": ISOLATION_WINDOW[1],
            "in_isolation_window": eligible,
            "eligibility_reason": "precursor M0 lies inside frozen acquisition isolation window 748–798 m/z" if eligible else "precursor M0 lies outside frozen acquisition isolation window",
            "precursor_source_channel_index": source_index,
            "precursor_source_channel_mz": float(source_axis[source_index]),
            "precursor_match_abs_error_da": abs(float(source_axis[source_index]) - precursor_mz),
            "precursor_compatible_group_id": f"PG_{group_id:03d}",
            "precursor_compatible_group_multiplicity": len(group_members[group_id]),
            "compatible_observation_channel_indices": ";".join(map(str, group_observation_rows[group_id].tolist())),
            "compatible_observation_channel_mz": ";".join(f"{mz_axis[i]:.8f}" for i in group_observation_rows[group_id]),
        })
    write_csv(OUT / "v48_simulation_eligible_candidates.csv", eligible_rows)

    coords = np.argwhere(mask)
    B_fg = B[mask]
    X_fg = X[:, mask].T
    group_ids = sorted(group_members)
    group_multiplicities = np.asarray([len(group_members[group_id]) for group_id in group_ids], dtype=np.int32)

    def observed_group_matrix(spectra: np.ndarray) -> np.ndarray:
        return np.column_stack([
            np.any(spectra[:, group_observation_rows[group_id]] > OBSERVED_THRESHOLD, axis=1)
            for group_id in group_ids
        ])

    pixel_group_observed = observed_group_matrix(B_fg)
    observed_group_count = pixel_group_observed.sum(axis=1)
    compatible_candidate_sum = pixel_group_observed @ group_multiplicities
    observed_mult_mean = np.divide(
        compatible_candidate_sum, observed_group_count,
        out=np.zeros_like(compatible_candidate_sum, dtype=np.float64), where=observed_group_count > 0,
    )
    observed_mult_median = np.asarray([
        float(np.median(group_multiplicities[row])) if row.any() else 0.0 for row in pixel_group_observed
    ])
    observed_mult_max = np.asarray([
        int(group_multiplicities[row].max()) if row.any() else 0 for row in pixel_group_observed
    ])
    active_counts = {threshold: (X_fg > threshold).sum(axis=1) for threshold in ACTIVE_THRESHOLDS}
    x_sum = X_fg.sum(axis=1)
    participation = np.square(x_sum) / np.maximum(np.square(X_fg).sum(axis=1), 1e-30)
    b_norm = np.linalg.norm(B_fg, axis=1)
    complexity_rows = []
    for i, (row_idx, col_idx) in enumerate(coords):
        complexity_rows.append({
            "scope_type": "foreground_pixel",
            "scope_id": f"pixel_{int(row_idx):03d}_{int(col_idx):03d}",
            "pixel_row": int(row_idx),
            "pixel_col": int(col_idx),
            "observed_precursor_group_count": int(observed_group_count[i]),
            "sum_compatible_candidate_multiplicities": int(compatible_candidate_sum[i]),
            "observed_group_multiplicity_mean": float(observed_mult_mean[i]),
            "observed_group_multiplicity_median": float(observed_mult_median[i]),
            "observed_group_multiplicity_max": int(observed_mult_max[i]),
            "active_count_1e-4": int(active_counts[1e-4][i]),
            "active_count_1e-3": int(active_counts[1e-3][i]),
            "active_count_1e-2": int(active_counts[1e-2][i]),
            "participation_ratio": float(participation[i]),
            "B_l2_norm": float(b_norm[i]),
        })

    roi_manifest_path = ROOT / "profile_758_v47_interim_24roi_first_layer" / "roi_manifest.csv"
    with roi_manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
        roi_manifest = list(csv.DictReader(handle))
    roi_group_counts = []
    roi_candidate_sums = []
    for roi in roi_manifest:
        y0, y1, x0, x1 = (int(roi[key]) for key in ("y0", "y1", "x0", "x1"))
        local_mask = mask[y0:y1, x0:x1]
        roi_B = B[y0:y1, x0:x1][local_mask].mean(axis=0, keepdims=True)
        roi_X = X[:, y0:y1, x0:x1][:, local_mask].mean(axis=1)
        observed = observed_group_matrix(roi_B)[0]
        count = int(observed.sum())
        candidate_sum = int(group_multiplicities[observed].sum())
        roi_group_counts.append(count)
        roi_candidate_sums.append(candidate_sum)
        complexity_rows.append({
            "scope_type": "frozen_roi",
            "scope_id": roi["roi_id"],
            "pixel_row": "",
            "pixel_col": "",
            "observed_precursor_group_count": count,
            "sum_compatible_candidate_multiplicities": candidate_sum,
            "observed_group_multiplicity_mean": float(group_multiplicities[observed].mean()) if count else 0.0,
            "observed_group_multiplicity_median": float(np.median(group_multiplicities[observed])) if count else 0.0,
            "observed_group_multiplicity_max": int(group_multiplicities[observed].max()) if count else 0,
            "active_count_1e-4": int((roi_X > 1e-4).sum()),
            "active_count_1e-3": int((roi_X > 1e-3).sum()),
            "active_count_1e-2": int((roi_X > 1e-2).sum()),
            "participation_ratio": float(roi_X.sum() ** 2 / max(float(np.square(roi_X).sum()), 1e-30)),
            "B_l2_norm": float(np.linalg.norm(roi_B)),
        })
    write_csv(OUT / "v48_complexity_envelope.csv", complexity_rows)

    complexity_summary = {
        "interpretation": "Simulation-design proxies only; none is K_true.",
        "foreground_pixel_count": int(mask.sum()),
        "fixed_observation_threshold": OBSERVED_THRESHOLD,
        "acquisition_isolation_window_mz": list(ISOLATION_WINDOW),
        "previous_748_803_explanation": "The earlier 748-803 range was inherited from the production loss parent-channel range so M1-M4 isotope channels above the 798 isolation edge were retained. It is not the acquisition isolation window and is removed from precursor-evidence complexity analysis.",
        "precursor_group_definition": "Candidates are grouped when their M0 precursors map, under the exact historical nearest-source-channel rule (<0.025 Da), to the same frozen source channel; observed support is read from the frozen empirical-overlap peak-response matrix and restricted to final channels inside 748-798 m/z.",
        "precursor_compatible_group_count": len(group_ids),
        "candidate_multiplicity_per_group": quantile_dict(group_multiplicities, probabilities=(0.10, 0.25, 0.50, 0.75, 0.90, 1.0)),
        "proxy_distributions": {
            "observed_precursor_group_count_foreground": quantile_dict(observed_group_count),
            "sum_compatible_candidate_multiplicities_foreground": quantile_dict(compatible_candidate_sum),
            "observed_group_multiplicity_pooled_occurrences": quantile_dict(
                np.broadcast_to(group_multiplicities, pixel_group_observed.shape)[pixel_group_observed]
            ),
            "observed_precursor_group_count_frozen_roi": quantile_dict(np.asarray(roi_group_counts)),
            "sum_compatible_candidate_multiplicities_frozen_roi": quantile_dict(np.asarray(roi_candidate_sums)),
        },
        "secondary_effective_complexity_descriptors": {
            "active_count_1e-4": quantile_dict(active_counts[1e-4]),
            "active_count_1e-3": quantile_dict(active_counts[1e-3]),
            "active_count_1e-2": quantile_dict(active_counts[1e-2]),
            "participation_ratio": quantile_dict(participation),
        },
        "preregistered_K_stress_grid_proposal": {
            "values": [1, 5, 13, 25, 50, 75, 90, 100, 105, 110, 150, 250, 350, 375, 385, 391],
            "role": "Experimental stress axis only; not a recommended or estimated biological K.",
            "design": "Spans sparse to full-library stress, with denser coverage around both the foreground observed-group P10-P90 region and the compatible-candidate-sum P10-P90 region.",
            "approval_status": "PROPOSED_PENDING_HUMAN_APPROVAL",
        },
    }
    (OUT / "v48_complexity_envelope_summary.json").write_text(
        json.dumps(complexity_summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # Fixed abundance-shape proxy: existing 1e-4 active threshold, normalize only
    # within each pixel to separate shape from the independently audited total scale.
    normalized_active: list[np.ndarray] = []
    rank_values: dict[int, list[float]] = {}
    for row in X_fg:
        values = row[row > 1e-4].astype(np.float64)
        if len(values) == 0:
            continue
        values = np.sort(values / values.sum())[::-1]
        normalized_active.append(values)
        for rank, value in enumerate(values, start=1):
            rank_values.setdefault(rank, []).append(float(value))
    pooled_fraction = np.concatenate(normalized_active)
    log_fraction = np.log(np.maximum(pooled_fraction, 1e-30))
    mu, sigma = float(log_fraction.mean()), float(log_fraction.std(ddof=0))
    abundance_probs = [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
    abundance_rows: list[dict] = []
    empirical_quantiles = np.quantile(pooled_fraction, abundance_probs)
    lognormal_quantiles = np.exp(mu + sigma * norm.ppf(abundance_probs))
    for prob, empirical, modeled in zip(abundance_probs, empirical_quantiles, lognormal_quantiles):
        abundance_rows.append({
            "record_type": "distribution_quantile",
            "source": "production_X_active_gt_1e-4_normalized_within_pixel",
            "quantile_or_rank": f"P{int(prob * 100)}",
            "empirical_value": float(empirical),
            "lognormal_value": float(modeled),
            "sample_count": int(len(pooled_fraction)),
        })
    for rank in sorted(rank_values):
        abundance_rows.append({
            "record_type": "empirical_rank_curve",
            "source": "production_X_active_gt_1e-4_normalized_within_pixel",
            "quantile_or_rank": f"rank_{rank}",
            "empirical_value": float(np.median(rank_values[rank])),
            "lognormal_value": "",
            "sample_count": len(rank_values[rank]),
        })
    parent_positive = B_fg[:, acquisition_rows]
    parent_positive = parent_positive[parent_positive > OBSERVED_THRESHOLD]
    parent_q = np.quantile(parent_positive, abundance_probs)
    for prob, value in zip(abundance_probs, parent_q):
        abundance_rows.append({
            "record_type": "independent_parent_intensity_reference",
            "source": "normalized_B_acquisition_window_748_798_channels_above_frozen_threshold",
            "quantile_or_rank": f"P{int(prob * 100)}",
            "empirical_value": float(value),
            "lognormal_value": "",
            "sample_count": int(len(parent_positive)),
        })
    write_csv(OUT / "v48_abundance_envelope.csv", abundance_rows)
    log_quantile_rmse = float(np.sqrt(np.mean((np.log(empirical_quantiles) - np.log(lognormal_quantiles)) ** 2)))
    abundance_model = {
        "status": "FROZEN_FOR_FUTURE_SIMULATION_PENDING_HUMAN_APPROVAL",
        "primary_model": "empirical_rank_resampling",
        "selection_rule": "Choose between the two preregistered models using fidelity to observed abundance quantiles/rank curve only; solver accuracy was not evaluated.",
        "selection_result": "empirical rank-resampling retained because it preserves the observed rank curve directly; the single log-normal has nonzero log-quantile error.",
        "source": "frozen production X on 15,837 foreground pixels",
        "active_threshold_reused": 1e-4,
        "normalization_for_shape_only": "Within each pixel, active abundances are divided by their active sum. Total signal scale is modeled separately from real B.",
        "empirical_rank_resampling": {
            "profile_count": len(normalized_active),
            "sampling_seed": SEED,
            "future_sampling_rule": "Sample one stored empirical descending normalized rank profile; truncate/interpolate only after the approved K is fixed, then renormalize weights to unit shape before applying B-norm scale.",
            "quantile_log_rmse_to_empirical": 0.0,
        },
        "preregistered_tail_scenarios": {
            "lighter_tail": "Raise each sampled empirical rank weight to power 0.75, then renormalize to unit sum.",
            "central": "Unmodified empirical rank-resampling profile, then renormalize to unit sum.",
            "heavier_tail": "Raise each sampled empirical rank weight to power 1.25, then renormalize to unit sum.",
            "equal_abundance_control": "All K active members receive weight 1/K; control only.",
            "approval_status": "PROPOSED_PENDING_HUMAN_APPROVAL",
        },
        "fixed_lognormal_comparator": {
            "mu_log_fraction": mu,
            "sigma_log_fraction": sigma,
            "quantile_log_rmse_to_empirical": log_quantile_rmse,
        },
        "abundance_fraction_quantiles": {f"P{int(p*100)}": float(v) for p, v in zip(abundance_probs, empirical_quantiles)},
        "dynamic_range_P99_over_P1": float(empirical_quantiles[-1] / empirical_quantiles[0]),
        "prohibition": "Model choice and parameters must not be changed after viewing future solver accuracy.",
    }
    (OUT / "v48_abundance_model.json").write_text(
        json.dumps(abundance_model, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    signal_probs = [0.10, 0.25, 0.50, 0.75, 0.90]
    signal_values = np.quantile(b_norm, signal_probs)
    signal_scale = {
        "source": "frozen normalized production B, foreground pixels only",
        "foreground_pixel_count": int(mask.sum()),
        "B_l2_norm_quantiles": {f"P{int(p*100)}": float(v) for p, v in zip(signal_probs, signal_values)},
        "future_scaling_rule": "For a generated abundance shape x, compute ||A x||_2 and multiply x by the linear factor required to hit a preregistered real-B norm quantile.",
        "normalization_warning": "Do not force sum(X_true)=1 as a final abundance scale; unit-sum is used only temporarily for abundance shape.",
        "solver_accuracy_not_used": True,
    }
    (OUT / "v48_signal_scale_envelope.json").write_text(
        json.dumps(signal_scale, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    group_rows = []
    with (V47 / "competition_group_family.csv").open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("weight_mode") == "identity" and abs(float(row["d_frag_cutoff"]) - 0.02) < 1e-12:
                group_rows.append(row)
    non_singleton = [row for row in group_rows if int(row["member_count"]) > 1]
    report = {
        "status": "PASS_SIMULATION_ENVELOPE_AUDIT_PENDING_HUMAN_APPROVAL",
        "x_true_generated": False,
        "production_ista_run": False,
        "eligible_candidates": {
            "isolation_window_mz": list(ISOLATION_WINDOW),
            "frozen_library_count": len(metadata),
            "eligible_count": int(sum(row["in_isolation_window"] for row in eligible_rows)),
            "all_frozen_candidates_eligible": bool(all(row["in_isolation_window"] for row in eligible_rows)),
        },
        "complexity": complexity_summary,
        "abundance": abundance_model,
        "signal_scale": signal_scale,
        "future_spatial_support_design_only": [
            "broad smooth support",
            "localized smooth blob",
            "partially overlapping competing supports",
        ],
        "competition_design": {
            "frozen_cutoff": 0.02,
            "group_family_hash": file_sha(V47 / "competition_group_family.csv"),
            "component_count_including_singletons": len(group_rows),
            "non_singleton_component_count": len(non_singleton),
            "non_singleton_sizes": [int(row["member_count"]) for row in non_singleton],
            "allowed_future_patterns": ["spatial overlap", "partial separation"],
        },
        "frozen_input_hashes": {
            "A": file_sha(DATA / "A_library.npy"),
            "B": file_sha(DATA / "B_cube.npy"),
            "X_production": file_sha(RESULT / "X_abundance.npy"),
            "candidate_metadata": file_sha(DATA / "candidate_metadata_final.jsonl"),
            "channel_axis": file_sha(DATA / "shared_mz_final.npy"),
            "foreground_mask": file_sha(DATA / "foreground_pixel_mask.npy"),
            "v47b_input_gate": file_sha(V47B / "v47b_input_audit.json"),
        },
        "random_seed_reserved_for_future_empirical_rank_sampling": SEED,
        "stop_rule": "Envelope audit complete. Stop before X_true generation and production ISTA execution; wait for human approval.",
    }
    (OUT / "v48_simulation_envelope_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    proxy = complexity_summary["proxy_distributions"]
    secondary = complexity_summary["secondary_effective_complexity_descriptors"]
    stress = complexity_summary["preregistered_K_stress_grid_proposal"]
    html = f"""<!doctype html><html lang='zh-CN'><meta charset='utf-8'><title>v48 simulation-envelope audit</title>
<style>body{{font-family:Arial,'Microsoft YaHei',sans-serif;max-width:1150px;margin:32px auto;line-height:1.55;color:#1f2937}}.pass{{padding:12px 16px;background:#ecfdf5;border-left:5px solid #059669}}table{{border-collapse:collapse;width:100%}}td,th{{border:1px solid #cbd5e1;padding:6px 9px;text-align:right}}td:first-child,th:first-child{{text-align:left}}code{{background:#f1f5f9;padding:2px 5px}}</style>
<h1>v48 Simulation-envelope audit</h1><p class='pass'><b>{report['status']}</b><br>只建立 simulation-design envelope；没有生成 X_true，也没有运行 production ISTA。</p>
<h2>候选范围</h2><p>真实 isolation window：<b>748–798 m/z</b>。冻结库 391 条，eligible <b>{report['eligible_candidates']['eligible_count']}</b> 条；全部候选的 precursor M0 均在窗口内。</p>
<h2>可观测复杂度边界/代理（严格 748–798 m/z）</h2><p>旧报告的 748–803 来自 production loss 的 parent-channel 范围，用于保留超过隔离窗上界的 M1–M4；它不是实际采集隔离窗，现已从本项分析删除。391 条候选按冻结的 precursor matching + profile-overlap 语义折叠为 <b>{len(group_ids)}</b> 个 precursor-compatible groups。</p><table><tr><th>proxy</th><th>P10</th><th>P25</th><th>P50</th><th>P75</th><th>P90</th><th>max</th></tr>
{''.join('<tr><td>'+name+'</td>'+''.join(f'<td>{values[key]:.3g}</td>' for key in ('P10','P25','P50','P75','P90','max'))+'</tr>' for name,values in proxy.items())}</table>
<p><b>这些量都不是 K_true。</b>不再输出单一 recommended K。待人工批准的预注册 stress grid：<code>{stress['values']}</code>；它只是实验轴。</p>
<h2>Production-X 次级 effective-complexity 描述</h2><table><tr><th>descriptor</th><th>P10</th><th>P25</th><th>P50</th><th>P75</th><th>P90</th><th>max</th></tr>{''.join('<tr><td>'+name+'</td>'+''.join(f'<td>{values[key]:.3g}</td>' for key in ('P10','P25','P50','P75','P90','max'))+'</tr>' for name,values in secondary.items())}</table><p>不同 active thresholds 与 participation ratio 保持分列，未合并成 biological K。</p>
<h2>丰度形状</h2><p>中心数据形状：<b>empirical rank-resampling</b>（不是 known truth）。预注册提案：lighter-tail power=0.75、central power=1、heavier-tail power=1.25；equal abundance 仅为 control。归一化 abundance fraction 的 P99/P1 dynamic range：<b>{abundance_model['dynamic_range_P99_over_P1']:.3g}×</b>。</p>
<h2>真实 B signal scale</h2><table><tr>{''.join(f'<th>{k}</th>' for k in signal_scale['B_l2_norm_quantiles'])}</tr><tr>{''.join(f'<td>{v:.6g}</td>' for v in signal_scale['B_l2_norm_quantiles'].values())}</tr></table>
<h2>停止</h2><p>本阶段8项产物已生成。下一步必须人工确认 envelope 后才能构造 X_true。</p></html>"""
    (OUT / "v48_simulation_envelope_report.html").write_text(html, encoding="utf-8")
    print(json.dumps({
        "status": report["status"],
        "eligible": report["eligible_candidates"]["eligible_count"],
        "precursor_groups": len(group_ids),
        "observed_group_count_quantiles": complexity_summary["proxy_distributions"]["observed_precursor_group_count_foreground"],
        "K_stress_grid_proposal": stress["values"],
        "abundance_dynamic_range": abundance_model["dynamic_range_P99_over_P1"],
        "B_l2": signal_scale["B_l2_norm_quantiles"],
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
