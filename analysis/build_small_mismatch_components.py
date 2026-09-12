"""Recover complete physical-ion envelopes using cached production builders.

No spectrum fitting, MSI extraction, candidate selection or isotope matching.
The current production A remains authoritative. Exact original isotope mass
keys and the original observation-axis projection identify component ownership;
saved production response columns are reused. Float32 accumulation differences
are recorded before fractional ownership partitions the unchanged A.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter, defaultdict
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name] = "1"

ROOT = Path(__file__).resolve().parents[1]
PRODUCTION = "v38_ce29_fragmentionfixed_profile_overlap_empiricalfwhm_globalq99_ista_ready"
PREDICTION = "v36_linear_ce_response_msi29_parent_gradient_fixed_msiw5"
SOLVER_ARRAY_SHA = "b9e05e185ffa692966b86022f5487fcdabff92f330a0a389a6bb4889a175a447"


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def original_projection(source, np, sparse):
    """Load exact original pure projection functions, avoiding raw-MSI imports."""
    tree = ast.parse(source.read_text(encoding="utf-8"))
    names = {"empirical_fwhm", "enforce_observation_axis_resolution"}
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    require({node.name for node in nodes} == names, "ORIGINAL_PROJECTION_FUNCTIONS_MISSING")
    namespace = {"np": np, "sparse": sparse}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), "exec"), namespace)
    return namespace["enforce_observation_axis_resolution"]


def main(args):
    import numpy as np
    from scipy import sparse

    pipeline = args.asset_root / "adapter_pipeline"
    production, prediction = pipeline / "outputs" / PRODUCTION, pipeline / "outputs" / PREDICTION
    cached = args.isotope_cache
    required = [args.solver_library, args.pccl_workbook,
                pipeline / "build_isotope_library_v23.py", pipeline / "build_isotope_library_758.py",
                pipeline / "process_758_two_pass.py"]
    required += [prediction / n for n in ("metadata_ce29_full.jsonl", "channel_predictions_ce29.npy",
                 "channel_target_mz.npy", "channel_masks.npy", "channels.json")]
    optional_survival = prediction / "precursor_survival_ce29.npy"
    if optional_survival.exists():
        required.append(optional_survival)
    required += [cached / n for n in ("A_library_joint.npy", "shared_mz.npy", "candidate_metadata.jsonl",
                 "axis_exact_mass_to_cluster.npz", "channel_formulas.jsonl", "isotope_library_report.json")]
    required += [production / n for n in ("spectrum_source_STICK.npy", "shared_mz_source_sticks.npy",
                 "peak_response_matrix.npz", "candidate_metadata_final.jsonl", "two_pass_report.json",
                 "peak_channel_coefficient.npy", "A_library.npy")]
    missing = [str(p) for p in required if not p.is_file()]
    require(not missing, "REQUIRED_DIRECT_SOURCE_MISSING:" + json.dumps(missing))
    require(not args.output.exists(), "NEW_COMPONENT_OUTPUT_REQUIRED")
    sys.path.insert(0, str(pipeline))
    import build_isotope_library_v23 as isotope
    import build_isotope_library_758 as legacy

    A = np.load(args.solver_library, allow_pickle=False)
    require(A.shape == (1084, 391) and A.dtype == np.float32 and
            hashlib.sha256(A.tobytes(order="C")).hexdigest() == SOLVER_ARRAY_SHA, "FIXED_SOLVER_LIBRARY_CHANGED")
    production_rows = jsonl(production / "candidate_metadata_final.jsonl")
    source_rows = jsonl(prediction / "metadata_ce29_full.jsonl")
    index = {r["entry_id"]: i for i, r in enumerate(source_rows)}
    channels = read(prediction / "channels.json")
    predictions = np.load(prediction / "channel_predictions_ce29.npy", allow_pickle=False)
    targets = np.load(prediction / "channel_target_mz.npy", allow_pickle=False)
    masks = np.load(prediction / "channel_masks.npy", allow_pickle=False)
    formulas = {(r["entry_id"], r["channel"]): r for r in jsonl(cached / "channel_formulas.jsonl")
                if r["channel"] != "pccl_physical_fragment"}
    pc_rows, pc_peaks, _ = isotope.load_pccl(args.pccl_workbook, prediction, source_rows, 29, 748., 798., .002)
    pc_index = {r["entry_id"]: (r, peaks) for r, peaks in zip(pc_rows, pc_peaks)}
    iso_report = read(cached / "isotope_library_report.json")
    require(iso_report["collision_energy_ev"] == 29 and iso_report["max_isotope_rank"] == 4 and
            iso_report["window"] == [748., 798.], "CACHED_ISOTOPE_CONTRACT_CHANGED")
    nominal_peaks, owners, components_meta = [], [], []
    origin_counts = Counter()
    for candidate, final in enumerate(production_rows):
        entry = final["selected_source_entry_id"]
        if entry in index:
            src = index[entry]
            row, peaks = source_rows[src], []
            for ch_index, channel in enumerate(channels):
                if masks[src, ch_index] <= 0 or not np.isfinite(targets[src, ch_index]) or predictions[src, ch_index] <= 0:
                    continue
                saved = formulas[(entry, channel)]
                require(float(saved["target_mz"]) == float(targets[src, ch_index]), "SAVED_MONOISOTOPIC_MASS_CHANGED")
                composition = legacy.parse_formula(saved["isotope_ion_formula"])
                peaks.append((float(targets[src, ch_index]), float(predictions[src, ch_index]), composition))
            origin = "V36_CHANNEL_PREDICTION"
        else:
            require(entry in pc_index and final["rule_sheet"] == "PC+Cl", "UNACCOUNTED_PRODUCTION_SOURCE:" + entry)
            row, peaks = pc_index[entry]
            require(row["precursor_survival_transferred"] == final["precursor_survival_transferred"], "PC_CL_PARENT_SURVIVAL_CHANGED")
            origin = "PC_CL_ORIGINAL_WORKBOOK_PLUS_ORIGINAL_SURVIVAL_TRANSFER"
        origin_counts[origin] += 1
        # This is the exact original duplicate-formula merge on the already
        # selected source. It does not repeat candidate/sn-source selection.
        _, merged, _ = legacy.collapse_sn_permutations([dict(row)], [peaks])
        parent = isotope.ion_formula(row)
        parent_tuple = isotope.counts_tuple(parent)
        has_parent = 0
        for mz, intensity, composition in merged[0]:
            kind = "precursor" if isotope.counts_tuple(composition) == parent_tuple else "fragment"
            has_parent += kind == "precursor"
            formula = legacy.formula_text(composition)
            owners.append(candidate)
            nominal_peaks.append(((mz, intensity, composition), parent_tuple, isotope.precursor_weights(parent_tuple, 4)))
            components_meta.append(dict(candidate_index=candidate, candidate_id=final["candidate_id"],
                selected_source_entry_id=entry, kind=kind, physical_ion_key=dict(identity=final["lipid_name"],
                    adduct=final["adduct"], CE=29., ion=kind + ":" + formula),
                monoisotopic_mz=mz, monoisotopic_intensity=intensity, isotope_ion_formula=formula,
                source_route=origin, grouping="Original selected-source duplicate-formula merge; exact isotope mass states"))
        require(has_parent == 1, "CANDIDATE_REQUIRES_ONE_ORIGINAL_PRECURSOR_ENVELOPE:" + entry)
    require(origin_counts["V36_CHANNEL_PREDICTION"] == 366 and len(owners) == len(components_meta), "FULL_SOURCE_COVERAGE_CHANGED")
    iso_axis = np.load(cached / "shared_mz.npy", allow_pickle=False)
    with np.load(cached / "axis_exact_mass_to_cluster.npz", allow_pickle=False) as z:
        mass_to_cluster = dict(zip(z["exact_mz"].tolist(), z["cluster_index"].tolist()))
    # Each physical ion remains one complete envelope across M0-M4. Existing
    # formula probabilities, 1e-14 pruning and float32 projection are unchanged.
    intrinsic = np.zeros((len(iso_axis), len(owners)), dtype=np.float64)
    for rank in range(5):
        spectra = [isotope.conditional_spectrum([peak], parent, rank) for peak, parent, _ in nominal_peaks]
        projected = isotope.project_spectra_to_resolution_axis(spectra, mass_to_cluster, len(iso_axis))
        intrinsic += projected * np.array([weights[rank] for _, _, weights in nominal_peaks])[None, :]
    cached_matrix = np.load(cached / "A_library_joint.npy", allow_pickle=False)[0]
    cached_rows = jsonl(cached / "candidate_metadata.jsonl")
    by_id = {r["candidate_id"]: i for i, r in enumerate(cached_rows)}
    kept = [by_id[r["candidate_id"]] for r in production_rows]
    report = read(production / "two_pass_report.json")
    merge = report["observation_axis_resolution"]
    project = original_projection(pipeline / "process_758_two_pass.py", np, sparse)
    # The function's ranked-matrix argument applies the same linear projection
    # to our component columns. Merge decisions use only the cached full library.
    merged_axis, merged_nominal, component_projection, merge_report = project(iso_axis, cached_matrix,
        intrinsic[None], empirical_fwhm_intercept=merge["empirical_fwhm_intercept"],
        empirical_fwhm_slope=merge["empirical_fwhm_slope"], empirical_fwhm_fraction=merge["empirical_fwhm_fraction"])
    theoretical = np.any(merged_nominal[:, kept] > 0, axis=1)
    nominal_sticks = merged_nominal[theoretical][:, kept]
    original_sticks = np.load(production / "spectrum_source_STICK.npy", allow_pickle=False).T
    require(nominal_sticks.dtype == original_sticks.dtype and np.array_equal(nominal_sticks, original_sticks),
            "ORIGINAL_SOURCE_STICKS_NOT_EXACTLY_RECONSTRUCTED")
    projected_components = component_projection[0, theoretical].astype(np.float64)
    source_axis = np.load(production / "shared_mz_source_sticks.npy", allow_pickle=False).reshape(-1)
    require(source_axis.shape == merged_axis[theoretical].shape, "SOURCE_AXIS_SHAPE_CHANGED")
    response = sparse.load_npz(production / "peak_response_matrix.npz")
    require(response.shape == (1084, 1084) and np.all(response.data >= 0), "INVALID_SAVED_PRODUCTION_RESPONSE")
    coefficient = np.load(production / "peak_channel_coefficient.npy", allow_pickle=False).reshape(-1)
    require(np.array_equal(coefficient, np.ones(1084)), "UNSUPPORTED_PRODUCTION_CHANNEL_WEIGHTING")
    envelopes = np.asarray(response @ projected_components, dtype=np.float64)
    source_sums = np.zeros_like(original_sticks, dtype=np.float64)
    response_sums = np.zeros_like(A, dtype=np.float64)
    for i, owner in enumerate(owners):
        source_sums[:, owner] += projected_components[:, i]
        response_sums[:, owner] += envelopes[:, i]
    # A componentwise reordering of the existing float32 sums changes rounding.
    # No absolute tolerance may hide an unaccounted weak peak or new support.
    require(np.array_equal(source_sums > 0, original_sticks > 0), "PHYSICAL_COMPONENT_SOURCE_SUPPORT_CHANGED")
    relative = np.divide(abs(source_sums-original_sticks), np.maximum(source_sums, original_sticks),
                         out=np.zeros_like(source_sums), where=np.maximum(source_sums, original_sticks)>0)
    guard = 64 * np.finfo(np.float32).eps
    require(np.all(relative <= guard), "PHYSICAL_COMPONENT_SOURCE_RECONSTRUCTION_EXCEEDS_FLOAT32_GUARD")
    require(np.array_equal(response_sums > 0, A > 0), "PHYSICAL_RESPONSE_SUPPORT_DOES_NOT_COVER_FIXED_LIBRARY")
    # Exact per-channel ownership of unchanged production A; original response
    # and physical envelope contributions determine every fraction, including
    # unresolved/overlapping ions. There is no nearest-m/z isotope assignment.
    D = np.zeros_like(envelopes)
    for i, owner in enumerate(owners):
        D[:, i] = np.divide(envelopes[:, i], response_sums[:, owner], out=np.zeros(1084),
                            where=response_sums[:, owner]>0) * A[:, owner]
    total = np.zeros_like(A, dtype=np.float64)
    for i, owner in enumerate(owners):
        total[:, owner] += D[:, i]
    partition_relative = np.divide(abs(total-A), np.maximum(total, A), out=np.zeros_like(total), where=A>0)
    require(np.array_equal(total>0, A>0) and np.all(partition_relative <= 8*np.finfo(np.float64).eps),
            "FIXED_LIBRARY_COMPONENT_PARTITION_FAILED")
    require(np.all(np.max(D,axis=0)>0), "ZERO_RESPONSE_PHYSICAL_COMPONENT")
    raw = np.load(production / "A_library.npy", allow_pickle=False).reshape(A.shape)
    normalized = response_sums / np.linalg.norm(response_sums, axis=0)
    raw_comparison = np.divide(abs(normalized-raw), np.maximum(normalized,raw), out=np.zeros_like(normalized), where=raw>0)
    require(np.all(raw_comparison <= guard), "PHYSICAL_ENVELOPES_NOT_PRODUCTION_NORMALIZED_LIBRARY")
    args.output.mkdir(parents=True, exist_ok=False)
    np.save(args.output / "A_solver.npy", A)
    np.savez_compressed(args.output / "components.npz", A=A, components=D, owner=np.asarray(owners,dtype=np.int32))
    write(args.output / "component_metadata.json", components_meta)
    write(args.output / "metadata.json", {name:[row[name] for row in production_rows]
          for name in ("candidate_id", "lipid_name", "lipid_class", "adduct", "collision_energy_ev", "selected_source_entry_id")})
    audit = dict(status="FULL_PHYSICAL_ION_COMPONENTS_VERIFIED", candidate_count=391, component_count=len(owners),
        source_candidate_counts=dict(origin_counts), kind_counts=dict(Counter(r["kind"] for r in components_meta)),
        candidate_coverage_count=len(set(owners)), precursor_candidate_coverage_count=len({r["candidate_index"] for r in components_meta if r["kind"]=="precursor"}),
        source_sticks_bitwise_parity=True, source_sticks_max_componentwise_relative_rounding=float(relative.max()),
        source_float32_rounding_guard=float(guard), original_saved_axis_and_response_reused=True,
        recomputed_axis_center_max_difference=float(abs(source_axis-merged_axis[theoretical]).max()),
        axis_center_difference_not_used_to_modify_production=True,
        normalized_component_response_vs_production_max_relative=float(raw_comparison.max()),
        fixed_A_partition_max_absolute_error=float(abs(total-A).max()), fixed_A_partition_max_relative_error=float(partition_relative.max()),
        exact_positive_support=True, A_solver_array_sha256=SOLVER_ARRAY_SHA,
        component_array_sha256=hashlib.sha256(D.tobytes(order="C")).hexdigest(),
        physical_family_definition="Original selected-source formula-merged monoisotopic ion, its exact conditional isotope states M0-M4, original cluster projection and saved profile response",
        final_partition="Per-channel fractions of original physical contributions multiplied by unchanged production A; float32 accumulation/normalization rounding preserved in nominal A",
        no_peak_matching_heuristic=True, no_observation_selection_or_MSI_processing=True, no_new_spectra_fit=True,
        merge_report=merge_report)
    write(args.output / "component_audit.json", audit)
    required.append(Path(__file__))
    manifest = dict(status="COMPLETE", inputs={str(p.resolve()):dict(bytes=p.stat().st_size,sha256=sha(p)) for p in required},
                    outputs={p.name:dict(bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(args.output.iterdir()) if p.is_file()})
    write(args.output / "provenance.json", manifest)
    print(json.dumps(audit), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-root", type=Path, default=ROOT.parent / "decon-lipid")
    parser.add_argument("--isotope-cache", type=Path, default=ROOT / "results/v59_w0_reconstruction/isotope_rebuilt")
    parser.add_argument("--solver-library", type=Path, default=ROOT / "results/physical_identity_mainline_pilot/A_solver.npy")
    parser.add_argument("--pccl-workbook", type=Path, default=Path("D:/jupyter/GNN/database/PC_Cl_MSMS_Library.xlsx"))
    parser.add_argument("--output", type=Path, default=ROOT / "results/small_mismatch_physical_components")
    main(parser.parse_args())
