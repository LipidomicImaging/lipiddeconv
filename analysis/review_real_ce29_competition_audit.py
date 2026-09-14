"""Independent cached-array review of the frozen CE29 competition audit.

This module deliberately imports neither the audit runner nor its geometry
helpers and never invokes a solver.  Every diagnostic is reviewed, including
all allowed-column KKT conditions in all molecular deletions.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
import time

for _key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
             "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_key] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / ".venv/real_ce29_packages"))
import numpy as np

EPS = 1e-12
IDS = (["CE29_GLOBAL"] + [f"CE29_BLOCK_{i:03d}" for i in range(64)]
       + [f"V58_MILD_CAL_R{i}_K125" for i in (1, 2)])


class AuditFailure(RuntimeError):
    def __init__(self, message, status="STOP_INVALID_AUDIT"):
        super().__init__(message)
        self.status = status


def require(condition, message, source=False):
    if not condition:
        raise AuditFailure(message, "STOP_SOURCE_PROVENANCE_MISMATCH" if source
                           else "STOP_INVALID_AUDIT")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def array_sha(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def records(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def flag(value):
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    require(str(value).strip().lower() in {"true", "false", "1", "0"},
            f"INVALID_BOOLEAN: {value}")
    return str(value).strip().lower() in {"true", "1"}


def numeric(value):
    if value is None or str(value).strip().lower() in {"", "none", "null", "nan"}:
        return float("nan")
    return float(value)


def close(actual, expected, label, rtol=2e-9, atol=2e-12):
    a, b = np.asarray(actual, dtype=float), np.asarray(expected, dtype=float)
    require(a.shape == b.shape and np.allclose(a, b, rtol=rtol, atol=atol,
                                             equal_nan=True), label)


def groups_from_names(candidate_names):
    groups, names, lookup = [], [], {}
    for index, name in enumerate(candidate_names):
        if name not in lookup:
            lookup[name] = len(names)
            names.append(name)
            groups.append([])
        groups[lookup[name]].append(index)
    return names, groups


def molecular_sum(array, groups):
    return np.stack([np.asarray(array)[..., indices].sum(axis=-1, dtype=np.float64)
                     for indices in groups], axis=-1)


def representatives(A, groups):
    columns = A / np.maximum(np.linalg.norm(A, axis=0), np.finfo(float).tiny)
    reps = np.stack([columns[:, indices].mean(axis=1) for indices in groups], axis=1)
    reps /= np.maximum(np.linalg.norm(reps, axis=0), np.finfo(float).tiny)
    return reps


def cosine_rows(left, right):
    return (left @ right.T) / (np.linalg.norm(left, axis=1)[:, None]
                              * np.linalg.norm(right, axis=1)[None, :] + EPS)


def provenance_review(out, report):
    provenance = read(out / "provenance.json")
    require(isinstance(provenance.get("sources"), list) and provenance["sources"],
            "EMPTY_PROVENANCE", source=True)
    checked, total_bytes = {}, 0
    for item in provenance["sources"]:
        path = Path(item["path"])
        require(path.is_file(), f"SOURCE_MISSING: {path}", source=True)
        require(path.stat().st_size == item["bytes"], f"SOURCE_BYTES: {path}", source=True)
        actual = checked.setdefault(str(path), sha(path)) if str(path) not in checked else checked[str(path)]
        require(actual == item["sha256"] == item["expected_sha256"],
                f"SOURCE_HASH: {path}", source=True)
        total_bytes += item["bytes"]
    report["source_review"] = dict(entries=len(provenance["sources"]),
                                    unique_files=len(checked), bytes=total_bytes,
                                    all_path_bytes_sha256_expected_sha256_match=True)
    return provenance


def kkt_independent(A, b, X, groups=None):
    """Vectorized original engineering bound, with deleted columns excluded."""
    X = np.atleast_2d(X)
    fitted = X @ A.T
    gradient = (fitted - b) @ A
    bound = (64 * np.finfo(np.float64).eps * max(A.shape)
             * ((np.abs(fitted) + np.abs(b)) @ np.abs(A)))
    violation = np.maximum(-gradient, 0)
    active = X > 0
    violation[active] = np.abs(gradient[active])
    allowed = np.ones(X.shape, dtype=bool)
    if groups is not None:
        for row, indices in enumerate(groups):
            allowed[row, indices] = False
    require(np.isfinite(gradient).all() and np.isfinite(bound).all(), "NONFINITE_KKT")
    require(np.all(violation[allowed] <= bound[allowed]), "INDEPENDENT_KKT_FAILED")
    ratio = np.divide(violation, bound, out=np.zeros_like(bound), where=bound > 0)
    dual = np.maximum(-gradient, 0)
    complementarity = np.abs(X * gradient)
    for array in (ratio, dual, complementarity):
        array[~allowed] = 0
    receipt = np.column_stack((dual.max(axis=1), complementarity.max(axis=1),
                               ratio.max(axis=1)))
    return fitted, receipt


def input_review(out, contract, report):
    paths = contract["source_paths"]
    with np.load(out / "diagnostic_inputs.npz", allow_pickle=False) as archive:
        require(set(archive.files) == {"A", "A_v58", "bs", "names", "foreground_positions", "block_pixel_counts"},
                "DIAGNOSTIC_INPUT_MEMBERSHIP_OR_TRUTH_LEAK")
        A, Av, bs, candidate_names = (archive[key].copy()
                                     for key in ("A", "A_v58", "bs", "names"))
        saved_positions = archive["foreground_positions"].copy()
        saved_counts = archive["block_pixel_counts"].copy()
    candidate_names = candidate_names.astype(str).tolist()
    names, groups = groups_from_names(candidate_names)
    require(A.shape == Av.shape == (1084, 391) and bs.shape == (67, 1084),
            "INPUT_DIMENSIONS")
    require(A.dtype == Av.dtype == bs.dtype == np.float64, "INPUT_PRECISION")
    require(len(candidate_names) == 391 and len(names) == 377, "MOLECULAR_COUNT")
    require(np.isfinite(A).all() and np.isfinite(Av).all() and np.isfinite(bs).all()
            and (A >= 0).all() and (Av >= 0).all(), "INPUT_FINITE_NONNEGATIVE")
    metadata = [json.loads(line) for line in Path(paths["metadata_jsonl"]).read_text(
        encoding="utf-8-sig").splitlines() if line.strip()]
    require([str(row["lipid_name"]) for row in metadata] == candidate_names,
            "CANDIDATE_ORDER_NOT_EXACT")
    original_candidates=read(Path(paths["ce29_binding"]).parent/"candidate_records_input.json")
    require([row["candidate_index"] for row in original_candidates]==list(range(391))
            and [row["candidate_id"] for row in original_candidates]==[row["candidate_id"] for row in metadata],
            "ORIGINAL_CANDIDATE_IDS_OR_ORDER_CHANGED")
    parent = read(paths["ce29_binding"])
    raw = np.load(parent["source_files"]["A"]["path"], mmap_mode="r", allow_pickle=False)
    require(np.array_equal(A, raw.astype(np.float64)), "A_NOT_ORIGINAL_EXPORTED_ARRAY")
    del raw
    raw_library = np.load(parent["source_files"]["raw_library"]["path"], mmap_mode="r", allow_pickle=False)[0]
    require(np.array_equal(A > 0, raw_library > 0), "ORIGINAL_LIBRARY_SUPPORT_CHANGED")
    B = np.load(parent["source_files"]["B"]["path"], mmap_mode="r", allow_pickle=False)[0].transpose(2, 0, 1)
    mask = np.load(paths["mask"], allow_pickle=False).astype(bool)
    require(int(mask.sum()) == 15837, "FOREGROUND_COUNT_CHANGED")
    spectra = np.asarray(B[:, mask], dtype=np.float64)
    expected = np.stack([spectra.mean(axis=1)] + [spectra[:, i:min(i + 250, 15837)].mean(axis=1)
                                                for i in range(0, 15837, 250)])
    close(bs[:65], expected, "B_OR_BLOCK_MEMBERSHIP_CHANGED", rtol=1e-14, atol=1e-15)
    lengths = np.array([min(250, 15837 - start) for start in range(0, 15837, 250)])
    require(np.array_equal(saved_positions, np.flatnonzero(mask.ravel()))
            and np.array_equal(saved_counts, lengths), "FOREGROUND_ORDER_OR_BLOCK_COUNTS")
    old_blocks=read(Path(paths["nnls_binding"]).parent/"nnls/completion.json")["block_hashes"]
    for start in range(0,15837,250):
        require(f"block_{start:06d}_{min(start+250,15837):06d}.npz" in old_blocks,
                "ORIGINAL_NNLS_BLOCK_BOUNDARY_CHANGED")
    close((bs[1:65] * lengths[:, None]).sum(axis=0) / lengths.sum(), bs[0],
          "BLOCK_WEIGHTED_MEAN_MISMATCH", rtol=1e-14, atol=1e-15)
    for offset, replicate in enumerate((1, 2), 65):
        with np.load(paths[f"v58_scoring_R{replicate}"], allow_pickle=False) as archive:
            require(np.array_equal(Av, archive["A"].astype(float)), "V58_A_CHANGED")
            require(np.array_equal(bs[offset], archive["b"].astype(float)), "V58_B_CHANGED")
    report["input_review"] = dict(candidate_order_exact=True, molecular_grouping_exact=True,
        candidate_count=391, molecular_count=377, foreground_pixels=15837,
        original_A_exact=True, original_B_block_means_recomputed=True,
        block_count=64, block_unit="consecutive foreground-order pixels, 250 except last",
        full_mean_weighted_block_mean_consistent=True, V58_A_B_exact=True,
        no_truth_field_in_solver_input=True)
    return A, Av, bs, candidate_names, names, groups, mask, metadata


def diagnostic_review(out, identifier, A, b, names, groups, hashes):
    array_path, receipt_path = out / "diagnostics" / f"{identifier}.npz", out / "diagnostics" / f"{identifier}.json"
    receipt = read(receipt_path)
    require(receipt["diagnostic_id"]==identifier and receipt["completion"]=="PASS"
            and receipt["bytes"]==array_path.stat().st_size
            and receipt["full_candidate_count"]==391
            and receipt["deleted_candidate_counts"]==[len(group) for group in groups],
            f"DIAGNOSTIC_COMPLETION_UNIVERSE: {identifier}")
    require(receipt["groups"] == groups and receipt["names"] == names, f"GROUP_BINDING: {identifier}")
    for key, expected in (("output_sha256", sha(array_path)),
                          ("A_array_sha256", array_sha(A)), ("b_array_sha256", array_sha(b))):
        require(receipt[key] == expected, f"DIAGNOSTIC_BINDING_{key}: {identifier}")
    for key, expected in hashes.items():
        require(receipt[key] == expected, f"DIAGNOSTIC_BINDING_{key}: {identifier}")
    with np.load(array_path, allow_pickle=False) as archive:
        require(set(archive.files) == {"b", "full_x", "deleted_x", "q_full", "q_deleted",
                                      "full_kkt", "deleted_kkt", "analytic_zero_deletion"},
                f"DIAGNOSTIC_MEMBERSHIP: {identifier}")
        z = {key: archive[key].copy() for key in archive.files}
    x, xd = z["full_x"], z["deleted_x"]
    require(x.shape == (391,) and xd.shape == (377, 391) and x.dtype == xd.dtype == np.float64,
            f"DIAGNOSTIC_SHAPE_OR_PRECISION: {identifier}")
    require(np.array_equal(z["b"], b), f"FULL_DELETE_B_CHANGED: {identifier}")
    require(np.isfinite(x).all() and np.isfinite(xd).all() and (x >= 0).all() and (xd >= 0).all(),
            f"DIAGNOSTIC_NONFINITE_OR_NEGATIVE: {identifier}")
    for j, aliases in enumerate(groups):
        require(np.all(xd[j, aliases] == 0), f"DELETED_ALIAS_NONZERO: {identifier}/{j}")
    analytic = np.asarray(z["analytic_zero_deletion"])
    require(analytic.shape == (377,) and analytic.dtype == np.bool_, "ANALYTIC_FLAG_SHAPE")
    for j in np.flatnonzero(analytic):
        require(np.all(x[groups[j]] == 0) and np.array_equal(xd[j], x),
                f"INVALID_ANALYTIC_ZERO_DELETION: {identifier}/{j}")
    f, kkf = kkt_independent(A, b, x)
    fd, kkd = kkt_independent(A, b, xd, groups)
    require(z["full_kkt"].shape == (3,) and z["deleted_kkt"].shape == (377, 3), "KKT_RECEIPT_SHAPE")
    require(np.isfinite(z["full_kkt"]).all() and np.isfinite(z["deleted_kkt"]).all()
            and (z["full_kkt"] >= 0).all() and (z["deleted_kkt"] >= 0).all()
            and z["full_kkt"][2] <= 1 and (z["deleted_kkt"][:, 2] <= 1).all(), "KKT_RECEIPT_INVALID")
    # BLAS matrix/vector reductions have different last bits. Independently
    # enforce every original bound; do not require identical receipt ratios.
    # Preserve the stated full-vector residual evaluation; changing it to a
    # batched reduction can dominate roundoff-sized delta-residual directions.
    r = b - A @ x
    rd = b - fd
    q, qd = float(r @ r), np.einsum("ij,ij->i", rd, rd)
    close(z["q_full"], q, f"SSE_FULL: {identifier}")
    close(z["q_deleted"], qd, f"SSE_DELETE: {identifier}")
    require(np.all(qd - q >= -1e-10 * max(1., float(b @ b))), "DELETE_BETTER_THAN_FULL")
    mx, md = molecular_sum(x, groups), molecular_sum(xd, groups)
    gains = np.maximum(md - mx, 0)
    np.fill_diagonal(gains, 0)
    total = gains.sum(axis=1)
    tolerance = (64 * np.finfo(float).eps * max(A.shape)
                 * np.maximum(np.maximum(1., float(x.max())), xd.max(axis=1)))
    maxima = gains.max(axis=1)
    tops = np.argmax(gains, axis=1)
    unique = (np.sum(np.abs(gains - maxima[:, None]) <= tolerance[:, None], axis=1) == 1) & (maxima > tolerance)
    tops[~unique] = -1
    reps = representatives(A, groups)
    absorption = cosine_rows(rd, reps.T)
    delta_absorption = cosine_rows(rd - r, reps.T)
    removed = np.stack([A[:, indices] @ x[indices] for indices in groups])
    replacement = np.maximum(xd-x, 0) @ A.T
    product = np.einsum("ij,ij->i", removed, replacement)
    source_norm = np.linalg.norm(removed, axis=1)
    signal_cosine = product/(source_norm*np.linalg.norm(replacement, axis=1)+EPS)
    signal_projection = product/(source_norm**2+EPS)
    return dict(identifier=identifier, x=x, xd=xd, q=q, qd=qd, loss=qd-q,
                relative_loss=(qd-q)/(q+EPS), gains=gains, totals=total,
                residual=absorption, delta_residual=delta_absorption,
                tolerance=tolerance, eligible=mx > tolerance, top=tops,
                molecular_full=mx, analytic_count=int(analytic.sum()),
                signal_cosine=signal_cosine, signal_projection=signal_projection,
                residual_norm=np.linalg.norm(rd, axis=1), delta_norm=np.linalg.norm(rd-r, axis=1),
                max_kkt_ratio=max(float(kkf[:, 2].max()), float(kkd[:, 2].max())))


def indexed_identity_table(out, filename, names):
    rows = records(out / filename)
    require(len(rows) == len(names), f"IDENTITY_TABLE_COUNT: {filename}")
    by_name = {row["lipid_name"]: row for row in rows}
    require(len(by_name) == len(names) and set(by_name) == set(names),
            f"IDENTITY_TABLE_MEMBERSHIP: {filename}")
    for index, name in enumerate(names):
        require(int(by_name[name]["identity_id"]) == index, f"IDENTITY_TABLE_ORDER: {filename}")
    return by_name


def check_numbers(row, expected, label):
    for key, value in expected.items():
        require(key in row, f"MISSING_COLUMN: {label}/{key}")
        close(numeric(row[key]), value, f"TABLE_VALUE: {label}/{key}")


def parsed(value):
    return json.loads(value) if isinstance(value, str) else value


def summary(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return dict(n=0, median=np.nan, q25=np.nan, q75=np.nan, IQR=np.nan)
    q25, median, q75 = np.quantile(values, [.25, .5, .75])
    return dict(n=len(values), median=median, q25=q25, q75=q75, IQR=q75-q25,
                minimum=float(values.min()), maximum=float(values.max()))


def check_summary(actual, values, label):
    actual = parsed(actual)
    for key, expected in summary(values).items():
        require(key in actual, f"SUMMARY_MISSING: {label}/{key}")
        close(numeric(actual[key]), expected, f"SUMMARY_VALUE: {label}/{key}")


def pearson(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.size < 2 or np.ptp(a) == 0 or np.ptp(b) == 0:
        return np.nan
    a, b = a-a.mean(), b-b.mean()
    return float(a @ b / np.sqrt((a @ a)*(b @ b)))


def average_ranks(row):
    order = np.argsort(row, kind="stable")
    values = np.asarray(row)[order]
    starts = np.r_[0, np.flatnonzero(values[1:] != values[:-1])+1]
    stops = np.r_[starts[1:], len(values)]
    ranks = np.empty(len(values), dtype=float)
    for start, stop in zip(starts, stops):
        ranks[order[start:stop]] = (start+stop-1)/2+1
    return ranks


def rank_pairs(rows):
    if len(rows) < 2:
        return np.empty(0)
    ranks = np.stack([average_ranks(row) for row in rows])
    centered = ranks-ranks.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(centered, axis=1)
    numer = centered @ centered.T
    denom = norms[:, None]*norms[None, :]
    correlations = np.divide(numer, denom, out=np.full_like(numer, np.nan), where=denom>0)
    return correlations[np.triu_indices(len(rows), 1)]


def spatial_values(a, b):
    union = np.count_nonzero((a > 0) | (b > 0))
    return dict(spatial_cosine=float(a @ b/(np.linalg.norm(a)*np.linalg.norm(b)+EPS)),
                pearson_spatial=pearson(a, b),
                support_overlap=np.count_nonzero((a>0)&(b>0))/union if union else np.nan,
                weighted_overlap=np.minimum(a, b).sum()/(np.maximum(a, b).sum()+EPS))


def physical_support(paths, A, groups):
    with np.load(paths["components_npz"], allow_pickle=False) as archive:
        components, owners = archive["components"], archive["owner"]
    metadata = read(paths["component_metadata"])
    require(len(metadata) == components.shape[1] == len(owners), "PHYSICAL_COMPONENT_COUNT")
    candidate_fragments = np.zeros(A.shape, bool)
    for index, row in enumerate(metadata):
        owner = int(owners[index])
        require(owner == int(row["candidate_index"]), "PHYSICAL_COMPONENT_OWNER")
        if row["kind"] == "fragment":
            candidate_fragments[:, owner] |= (components[:, index] > 0) & (A[:, owner] > 0)
    fragments = np.stack([candidate_fragments[:, group].any(axis=1) for group in groups], axis=1)
    support = np.stack([(A[:, group] > 0).any(axis=1) for group in groups], axis=1)
    occupancy = support.sum(axis=1)
    reps = representatives(A, groups)
    weighted = reps/np.sqrt(np.maximum(occupancy, 1))[:, None]
    weighted /= np.maximum(np.linalg.norm(weighted, axis=0), np.finfo(float).tiny)
    full, reduced = reps.T @ reps, weighted.T @ weighted
    np.fill_diagonal(full, np.nan)
    np.fill_diagonal(reduced, np.nan)
    return dict(fragments=fragments, common=fragments & (occupancy[:, None] > 1),
                specific=fragments & (occupancy[:, None] == 1), full=full, reduced=reduced)


def support_values(A, b, x, groups, physical):
    contribution = np.stack([A[:, group] @ x[group] for group in groups], axis=1)
    attribution = contribution/(contribution.sum(axis=1)[:, None]+EPS)
    answer = []
    for j in range(len(groups)):
        frag, common, specific = (physical[key][:, j] for key in ("fragments", "common", "specific"))
        denominator = b[frag].sum()+EPS
        answer.append(dict(common_fragment_support=b[common].sum()/denominator,
            specific_fragment_support=b[specific].sum()/denominator,
            common_fragment_count=int(common.sum()), specific_fragment_count=int(specific.sum()),
            fragment_channel_count=int(frag.sum()),
            diagnostic_attribution_mean=float(attribution[frag,j].mean()) if frag.any() else np.nan,
            diagnostic_attribution_max=float(attribution[frag,j].max()) if frag.any() else np.nan,
            specific_support_attribution=float(attribution[specific,j].mean()) if specific.any() else np.nan,
            specific_predicted_signal=float(contribution[specific,j].sum())))
    return answer


def expected_deletion(item, j, names):
    gain, total = item["gains"][j], item["totals"][j]
    order = [int(k) for k in np.argsort(-gain, kind="stable")[:5] if gain[k] > 0]
    first = order[0] if order else None
    values = dict(q_full=item["q"], q_deleted=item["qd"][j],
        absolute_necessity_loss=item["loss"][j], relative_necessity_loss=item["relative_loss"][j],
        full_diagnostic_abundance=item["molecular_full"][j], redistribution_total=total,
        top_replacement_gain=gain[first] if first is not None else 0.,
        top_replacement_fraction=gain[first]/(total+EPS) if first is not None else 0.,
        numerical_edge_tolerance=item["tolerance"][j],
        removed_vs_positive_replacement_signal_cosine=item["signal_cosine"][j],
        removed_signal_projection_fraction=item["signal_projection"][j],
        deletion_residual_norm=item["residual_norm"][j], delta_residual_norm=item["delta_norm"][j],
        residual_absorption=item["residual"][j,first] if first is not None else np.nan,
        delta_residual_absorption=item["delta_residual"][j,first] if first is not None else np.nan)
    return values, order


def check_deletion_row(row, item, j, names, A):
    values, order = expected_deletion(item, j, names)
    check_numbers(row, values, f"DELETION/{item['identifier']}/{j}")
    require((row["top_replacement_identity"] or None) == (names[order[0]] if order else None), "TOP_REPLACEMENT_CHANGED")
    require(parsed(row["top5_replacement_identities"]) == [names[k] for k in order], "TOP5_IDENTITIES_CHANGED")
    close(parsed(row["top5_replacement_gains"]), item["gains"][j, order], "TOP5_GAINS_CHANGED")
    require(flag(row["source_eligible"]) == bool(item["eligible"][j]), "SOURCE_ELIGIBILITY_CHANGED")
    require((row["unique_top_replacement"] or None) == (names[item["top"][j]] if item["top"][j]>=0 else None), "UNIQUE_TOP_CHANGED")
    maximum = item["gains"][j].max()
    ties = np.flatnonzero((item["gains"][j] >= maximum-item["tolerance"][j])
                         & (item["gains"][j] > item["tolerance"][j]))
    require(parsed(row["top_tie_identities"]) == [names[k] for k in ties], "TOP_TIES_CHANGED")


def production_review(out, paths, names, groups, mask):
    master = indexed_identity_table(out, "identity_master_table.csv", names)
    source_rows = read(paths["current_records"])
    source = {(row["method"], row["lipid_name"]): row for row in source_rows}
    Xista = np.load(paths["ista_x"], mmap_mode="r", allow_pickle=False)
    with np.load(paths["nnls_arrays"], allow_pickle=False) as archive:
        require("X_hat" in archive.files, "NNLS_X_MISSING")
        Xnnls = archive["X_hat"]
    arrays = {"ISTA": np.asarray(Xista[:, mask], dtype=float),
              "NNLS": np.asarray(Xnnls[:, mask], dtype=float)}
    counts = {}
    for method, X in arrays.items():
        means = X.mean(axis=1, dtype=np.float64)
        gate = means > .001
        counts[method] = dict(reported=0, retained=0)
        for j, name in enumerate(names):
            row, saved = master[name], source[method, name]
            aliases = np.asarray(groups[j])
            reported = bool(gate[aliases].any())
            abundance = float(means[aliases[gate[aliases]]].sum())
            require(flag(saved["raw_solver_reported"]) == reported, "SOURCE_PRODUCTION_GATE_CHANGED")
            require(flag(row[f"{method.lower()}_reported"]) == reported, "MASTER_PRODUCTION_GATE_CHANGED")
            close(numeric(saved["X_hat"]), abundance, "SOURCE_REPORTED_ALIAS_ABUNDANCE")
            check_numbers(row, {f"{method.lower()}_reported_alias_mean_abundance": abundance,
                                f"{method.lower()}_mean_abundance": float(means[aliases].sum()),
                                f"{method.lower()}_total_abundance": float(X[aliases].sum())}, name)
            require(parsed(row["candidate_indices"]) == groups[j]
                    and int(row["candidate_count"]) == len(groups[j]), "MASTER_ALIAS_MAPPING")
            outcome = "retained" if flag(saved["retained_by_DEV10_TRANSFER"]) else "rejected" if reported else "unreported"
            require(row[f"{method.lower()}_current_confidence_outcome"] == outcome, "CURRENT_OUTCOME_CHANGED")
            if reported:
                require(flag(saved["retained_by_DEV10_TRANSFER"]) == (numeric(saved["joint_score"]) >= .5406530976316042),
                        "DEV10_THRESHOLD_CHANGED")
            counts[method]["reported"] += reported
            counts[method]["retained"] += flag(saved["retained_by_DEV10_TRANSFER"])
    require(counts == {"ISTA": {"reported": 76, "retained": 38},
                       "NNLS": {"reported": 78, "retained": 49}}, "ORIGINAL_COUNTS_CHANGED")
    return master, arrays, counts


def no_real_truth_fields(out):
    checked = []
    for path in sorted(out.glob("*.csv")):
        if path.name.startswith("v58_"):
            continue
        with path.open(encoding="utf-8-sig", newline="") as handle:
            fields = csv.DictReader(handle).fieldnames or []
        for field in fields:
            require(not re.search(r"(^|_)(TP|FP|FN|FDR|truth|true_identity|false_identity)($|_)",
                                  field, re.I), f"FORBIDDEN_CE29_TRUTH_FIELD: {path.name}/{field}")
        checked.append(path.name)
    return checked


def review(out):
    started = time.monotonic()
    report = dict(status="REVIEW_IN_PROGRESS", reviewer="independent_numpy_cached_arrays",
                  solver_calls=0, full_coverage=True, sampling_used=False)
    provenance_review(out, report)
    contract = read(out / "audit_contract.json")
    hashes = {"contract_sha256": sha(out / "audit_contract.json"),
              "provenance_sha256": sha(out / "provenance.json"),
              "input_sha256": sha(out / "diagnostic_inputs.npz"),
              "input_binding_sha256": sha(out / "input_binding.json")}
    binding, ack = read(out / "input_binding.json"), read(out / "pre_execution_git.json")
    require(binding["contract_sha256"] == hashes["contract_sha256"]
            and binding["provenance_sha256"] == hashes["provenance_sha256"]
            and binding["diagnostic_inputs_sha256"] == hashes["input_sha256"], "INPUT_BINDING_CHANGED")
    require(binding["diagnostic_ids"] == IDS and binding["pre_execution_commit"]
            == ack["local_commit"] == ack["remote_commit"], "PRE_EXECUTION_BINDING_INVALID")
    for name, expected_sha in binding["code_sha256"].items():
        require(sha(ROOT / name) == expected_sha, f"FROZEN_CODE_CHANGED: {name}")
    A, Av, bs, candidate_names, names, groups, mask, metadata = input_review(out, contract, report)
    expected = {f"{identifier}.{extension}" for identifier in IDS for extension in ("npz", "json")}
    require({path.name for path in (out / "diagnostics").iterdir()} == expected, "DIAGNOSTIC_EXACT_MEMBERSHIP")
    diagnostics = []
    for i, identifier in enumerate(IDS):
        diagnostics.append(diagnostic_review(out, identifier, A if i < 65 else Av, bs[i], names, groups, hashes))
    report["diagnostic_review"] = dict(datasets=67, full_models=67, molecular_deletions=67*377,
        deleted_all_aliases=True, no_candidate_pruning=True, full_delete_same_b=True,
        all_allowed_columns_KKT_checked=True, all_SSE_recomputed=True,
        analytic_zero_deletions=sum(item["analytic_count"] for item in diagnostics),
        maximum_independent_KKT_bound_ratio=max(item["max_kkt_ratio"] for item in diagnostics))
    master, spatial, counts = production_review(out, contract["source_paths"], names, groups, mask)
    report["production_counts"] = counts
    report["ce29_tables_without_truth_fields"] = no_real_truth_fields(out)
    # Table/graph checks below are independent from the numerical runner.
    table_review(out, contract, A, Av, names, groups, diagnostics, master, spatial, report)
    report.update(status="PASS", elapsed_seconds=round(time.monotonic()-started, 3),
                  contract_sha256=hashes["contract_sha256"], input_sha256=hashes["input_sha256"],
                  limitations=["KKT verifies numerical optimality, not unique identity attribution.",
                    "Block means are diagnostics, not pixelwise production refits or independent replicates.",
                    "Missing ISTA deletion evidence remains unavailable.",
                    "No real CE29 truth, FDR or recall is inferred."])
    return report


def anchor_review(saved, names, master, agreement, global_d, physical, specific):
    matrix=np.column_stack((agreement,global_d["relative_loss"],np.nanmax(physical["full"],axis=1),
                            [row["specific_support_attribution"] for row in specific]))
    both=np.array([flag(master[name]["ista_reported"]) and flag(master[name]["nnls_reported"]) for name in names])
    eligible=both&np.isfinite(matrix).all(axis=1)
    strong,weak=np.zeros(len(names),bool),np.zeros(len(names),bool)
    if eligible.any():
        q25,q75=np.quantile(matrix[eligible],[.25,.75],axis=0)
        strong=eligible&(matrix[:,0]>=q75[0])&(matrix[:,1]>=q75[1])&(matrix[:,2]<=q25[2])&(matrix[:,3]>=q75[3])
        weak=eligible&(matrix[:,0]<=q25[0])&(matrix[:,1]<=q25[1])&(matrix[:,2]>=q75[2])&(matrix[:,3]<=q25[3])
    else:
        q25=q75=np.full(4,np.nan)
    ties=strong&weak
    strong &= ~ties
    weak &= ~ties
    close([numeric(x) for x in saved["lower_quartiles"]],q25,"ANCHOR_LOWER_QUARTILES")
    close([numeric(x) for x in saved["upper_quartiles"]],q75,"ANCHOR_UPPER_QUARTILES")
    require(saved["complete_both_reported_count"]==int(eligible.sum())
            and saved["strong_anchor_count"]==int(strong.sum())
            and saved["weak_anchor_count"]==int(weak.sum())
            and saved["unresolved_tie_count"]==int(ties.sum()),"ANCHOR_COUNTS")
    for key,mask in (("strong_anchor_identities",strong),("weak_anchor_identities",weak),("unresolved_tie_identities",ties)):
        require(saved[key]==[names[j] for j in np.flatnonzero(mask)],f"ANCHOR_MEMBERSHIP: {key}")
    for index,key in enumerate(("agreement","necessity","ambiguity","specific_attribution")):
        check_summary(saved["anchor_metric_distributions"][key],matrix[eligible,index],f"ANCHOR_DISTRIBUTION/{key}")
    return strong,weak


def synthetic_review(out,paths,A,names,groups,diagnostics,physical):
    rows=records(out/"v58_identity_geometry.csv")
    lookup={(row["case_id"],row["lipid_name"]):row for row in rows}
    require(len(rows)==len(lookup)==754,"V58_EXACT_MEMBERSHIP")
    output={}
    for replicate,item in enumerate(diagnostics,1):
        sources={row["lipid_name"]:row for row in records(paths[f"v58_records_R{replicate}"])}
        require(set(sources)==set(names),"V58_SOURCE_IDENTITY_MEMBERSHIP")
        with np.load(out/"diagnostics"/f"{item['identifier']}.npz",allow_pickle=False) as z:
            supports=support_values(A,z["b"],item["x"],groups,physical)
        case_rows=[]
        for j,name in enumerate(names):
            row=lookup[item["identifier"],name]
            require(int(row["identity_id"])==j,"V58_IDENTITY_ORDER")
            check_deletion_row(row,item,j,names,A)
            source=sources[name]
            truth,reported=flag(source["molecular_truth"]),flag(source["raw_solver_reported"])
            label="TP" if truth and reported else "FP" if reported else "FN" if truth else "UNREPORTED_NONTRUTH"
            require(flag(row["synthetic_truth"])==truth and flag(row["production_reported"])==reported
                    and row["production_identity_status"]==label,"V58_TRUTH_ONLY_GROUPING")
            check_numbers(row,dict(production_X_hat=numeric(source["X_hat"]),production_rho=numeric(source["rho_zero"])),"V58_PRODUCTION")
            nearest=int(np.nanargmax(physical["full"][j]))
            reduced=int(np.nanargmax(physical["reduced"][j]))
            shared=physical["fragments"][:,j]&physical["fragments"][:,nearest]
            either=physical["fragments"][:,j]|physical["fragments"][:,nearest]
            extra=dict(nearest_neighbor_cosine_full=physical["full"][j,nearest],
                       nearest_neighbor_cosine_reduced=physical["reduced"][j,reduced],
                       shared_fragment_count=shared.sum(),shared_fragment_fraction=shared.sum()/either.sum() if either.any() else 0.)
            check_numbers(row,extra,"V58_GEOMETRY")
            check_numbers(row,supports[j],"V58_SPECIFIC_SUPPORT")
            require(row["nearest_neighbor_identity"]==names[nearest]
                    and row["nearest_neighbor_identity_reduced"]==names[reduced],"V58_NEAREST_NAMES")
            values,_=expected_deletion(item,j,names)
            values.update(extra,production_identity_status=label,lipid_name=name)
            case_rows.append(values)
        output[item["identifier"]]=case_rows
    return output


def effect_values(a,b):
    a=np.sort(np.asarray(a,float)); a=a[np.isfinite(a)]
    b=np.sort(np.asarray(b,float)); b=b[np.isfinite(b)]
    result={"ce29_"+key:value for key,value in summary(a).items()}
    result.update({"v58_"+key:value for key,value in summary(b).items()})
    if len(a) and len(b):
        points=np.union1d(a,b)
        ce=np.searchsorted(a,points,side="right")/len(a)
        sy=np.searchsorted(b,points,side="right")/len(b)
        wins=np.searchsorted(a,b,side="left").sum()
        losses=(len(a)-np.searchsorted(a,b,side="right")).sum()
        result.update(cliffs_delta=(wins-losses)/(len(a)*len(b)),
            wasserstein_distance=float(np.sum(np.abs(ce[:-1]-sy[:-1])*np.diff(points))),
            ks_statistic=float(np.max(np.abs(ce-sy))))
    else:
        result.update(cliffs_delta=np.nan,wasserstein_distance=np.nan,ks_statistic=np.nan)
    return result


def check_ecdf(saved,values,label):
    values=np.asarray(values,float); values=values[np.isfinite(values)]
    unique,counts=np.unique(values,return_counts=True)
    require(saved["n"]==len(values),f"ECDF_COUNT: {label}")
    close(saved["x"],unique,f"ECDF_X: {label}")
    close(saved["cumulative_probability"],np.cumsum(counts)/len(values) if len(values) else [],f"ECDF_PROBABILITY: {label}")


def comparison_review(out,names,master,ce,synthetic,strong,weak,stable):
    strata={"ISTA_REJECTED":np.array([master[name]["ista_current_confidence_outcome"]=="rejected" for name in names]),
        "NNLS_REJECTED":np.array([master[name]["nnls_current_confidence_outcome"]=="rejected" for name in names]),
        "ISTA_RETAINED":np.array([master[name]["ista_current_confidence_outcome"]=="retained" for name in names]),
        "NNLS_RETAINED":np.array([master[name]["nnls_current_confidence_outcome"]=="retained" for name in names]),
        "BOTH_REJECTED":np.array([master[name]["ista_current_confidence_outcome"]=="rejected" and master[name]["nnls_current_confidence_outcome"]=="rejected" for name in names]),
        "WEAK_CONSISTENCY":weak,"STRONG_CONSISTENCY":strong,"STRICT_STABLE_REPLACEMENT_SOURCE":stable.any(axis=1)}
    metrics=("nearest_neighbor_cosine_full","redistribution_total","residual_absorption","delta_residual_absorption",
             "shared_fragment_fraction","top_replacement_fraction","removed_vs_positive_replacement_signal_cosine")
    rows=records(out/"v58_vs_ce29_geometry_comparison.csv")
    lookup={(row["case_id"],row["synthetic_group"],row["ce29_group"],row["metric"]):row for row in rows}
    expected_keys={(case,label,stratum,metric) for case in synthetic for label in ("TP","FP")
                   for stratum in strata for metric in metrics}
    require(len(rows)==len(lookup) and set(lookup)==expected_keys,"COMPARISON_MEMBERSHIP")
    cdfs=read(out/"decoy_coverage_audit.json")["v58_comparison_ECDFs"]
    require(set(cdfs)=={"|".join(key) for key in expected_keys},"COMPARISON_ECDF_MEMBERSHIP")
    for key,row in lookup.items():
        case,label,stratum,metric=key
        a=[ce[j][metric] for j in np.flatnonzero(strata[stratum])]
        b=[item[metric] for item in synthetic[case] if item["production_identity_status"]==label]
        check_numbers(row,effect_values(a,b),f"COMPARISON/{key}")
        ecdf=cdfs["|".join(key)]
        check_ecdf(ecdf["ce29"],a,f"CE29/{key}")
        check_ecdf(ecdf["synthetic"],b,f"V58/{key}")


def maximal_cliques(adjacency):
    neighbors=[set(np.flatnonzero(row)) for row in adjacency]
    found=[]
    def visit(selected,candidates,excluded):
        if not candidates and not excluded:
            if len(selected)>=3:
                found.append(tuple(sorted(selected)))
            return
        pivot=max(candidates|excluded,key=lambda value:len(candidates&neighbors[value])) if candidates|excluded else None
        for vertex in sorted(candidates-(neighbors[pivot] if pivot is not None else set())):
            visit(selected|{vertex},candidates&neighbors[vertex],excluded&neighbors[vertex])
            candidates.remove(vertex)
            excluded.add(vertex)
    visit(set(),set(range(len(neighbors))),set())
    return sorted(found)


def table_review(out, contract, A, Av, names, groups, diagnostics, master, spatial, report):
    paths, n = contract["source_paths"], len(names)
    by_name = {name:j for j,name in enumerate(names)}
    global_d, blocks = diagnostics[0], diagnostics[1:65]
    eligibility = np.stack([item["eligible"] for item in blocks])
    gains = np.stack([item["gains"] for item in blocks])
    tolerances = np.stack([item["tolerance"] for item in blocks])
    tops = np.stack([item["top"] for item in blocks])
    numerical = gains > tolerances[:, :, None]
    presence = numerical & eligibility[:, :, None]
    eligible_counts=eligibility.sum(axis=0)
    persistent=(eligible_counts[:,None]>=2)&(presence.sum(axis=0)==eligible_counts[:,None])
    np.fill_diagonal(persistent,False)
    global_presence = global_d["gains"] > global_d["tolerance"][:, None]
    union = global_presence | numerical.any(axis=0)
    np.fill_diagonal(union, False)
    stable = np.zeros((n,n), dtype=bool)
    rank_values = []
    for j in range(n):
        eligible_tops = tops[eligibility[:,j],j]
        if len(eligible_tops)>=2 and eligible_tops[0]>=0 and np.all(eligible_tops==eligible_tops[0]):
            stable[j,eligible_tops[0]] = True
        rank_values.append(rank_pairs(gains[eligibility[:,j],j,:]))
    physical = physical_support(paths, A, groups)
    physical_v = physical_support(paths, Av, groups)
    maps = {method:np.stack([values[group].sum(axis=0) for group in groups])
            for method,values in spatial.items()}
    # Reconstruct B directly from each cache; b was already bound to raw B.
    with np.load(out/"diagnostics/CE29_GLOBAL.npz", allow_pickle=False) as z:
        b = z["b"].copy()
    support_i = support_values(A,b,spatial["ISTA"].mean(axis=1),groups,physical)
    support_n = support_values(A,b,spatial["NNLS"].mean(axis=1),groups,physical)
    support_global = support_values(A,b,global_d["x"],groups,physical)
    geometry = indexed_identity_table(out,"identity_spectral_geometry.csv",names)
    deletion = indexed_identity_table(out,"identity_deletion_metrics.csv",names)
    solver = indexed_identity_table(out,"identity_solver_agreement.csv",names)
    support_table = indexed_identity_table(out,"identity_specific_support.csv",names)
    classes = [master[name]["lipid_class"] for name in names]
    agreement = np.empty(n)
    expected_ce = []
    for j,name in enumerate(names):
        check_deletion_row(deletion[name],global_d,j,names,A)
        require(parsed(deletion[name]["eligible_blocks"]) == np.flatnonzero(eligibility[:,j]).tolist(), "BLOCK_DENOMINATOR")
        require(parsed(deletion[name]["unique_top_by_block"]) == [names[k] if k>=0 else None for k in tops[:,j]], "BLOCK_TOP_NAMES")
        check_summary(deletion[name]["replacement_rank_stability"],rank_values[j],f"RANK_STABILITY/{j}")
        order = np.argsort(-np.nan_to_num(physical["full"][j],nan=-np.inf),kind="stable")[:5]
        reduced_order = np.argsort(-np.nan_to_num(physical["reduced"][j],nan=-np.inf),kind="stable")[:5]
        nearest = int(order[0])
        grow = geometry[name]
        require(grow["nearest_neighbor_identity"]==names[nearest], "SPECTRAL_NEAREST_IDENTITY")
        require(parsed(grow["top5_neighbor_identities"])==[names[k] for k in order], "SPECTRAL_TOP5")
        close(parsed(grow["top5_cosines_full"]),physical["full"][j,order], "SPECTRAL_TOP5_COSINE")
        close(parsed(grow["top5_cosines_reduced"]),physical["reduced"][j,reduced_order], "SPECTRAL_TOP5_REDUCED_COSINE")
        require(grow["nearest_neighbor_identity_reduced"]==names[reduced_order[0]]
                and parsed(grow["top5_neighbor_identities_reduced"])==[names[k] for k in reduced_order], "REDUCED_NEIGHBOR_ORDER")
        shared = physical["fragments"][:,j]&physical["fragments"][:,nearest]
        shared_union = physical["fragments"][:,j]|physical["fragments"][:,nearest]
        same = np.array([cls==classes[j] for cls in classes]); same[j]=False
        cross = np.array([cls!=classes[j] for cls in classes])
        check_numbers(grow,dict(nearest_neighbor_cosine_full=physical["full"][j,nearest],
            nearest_neighbor_cosine_reduced=physical["reduced"][j,reduced_order[0]],
            full_neighbor_matched_cosine_reduced=physical["reduced"][j,nearest],
            same_class_nearest_cosine=np.nanmax(physical["full"][j,same]) if same.any() else np.nan,
            cross_class_nearest_cosine=np.nanmax(physical["full"][j,cross]) if cross.any() else np.nan,
            shared_fragment_count=shared.sum(),shared_fragment_fraction=shared.sum()/shared_union.sum() if shared_union.any() else 0),name)
        check_numbers(support_table[name],support_i[j],f"SUPPORT/{name}")
        check_numbers(support_table[name],dict(nnls_specific_support_attribution=support_n[j]["specific_support_attribution"]),f"SUPPORT_NNLS/{name}")
        check_numbers(support_table[name],dict(global_diagnostic_specific_support_attribution=support_global[j]["specific_support_attribution"]),f"SUPPORT_GLOBAL/{name}")
        require(not math.isfinite(numeric(support_table[name]["specific_fragment_observed_fraction"])), "UNVERIFIED_OBSERVATION_THRESHOLD_USED")
        spatial_pair=spatial_values(maps["ISTA"][j],maps["NNLS"][j])
        agreement[j]=spatial_pair["spatial_cosine"]
        im,nm=maps["ISTA"][j].mean(),maps["NNLS"][j].mean()
        check_numbers(solver[name],dict(ista_nnls_ratio=im/nm if nm>0 else np.nan,
            ista_nnls_log_ratio=np.log(im/nm) if im>0 and nm>0 else np.nan,
            pixelwise_abundance_corr=spatial_pair["pearson_spatial"],spatial_cosine_ista_nnls=agreement[j]),f"SOLVER/{name}")
        expected_values,_=expected_deletion(global_d,j,names)
        expected_values.update(nearest_neighbor_cosine_full=physical["full"][j,nearest],
            shared_fragment_fraction=shared.sum()/shared_union.sum() if shared_union.any() else 0.)
        expected_ce.append(expected_values)
    expected_edges={(names[j],names[k]) for j,k in zip(*np.nonzero(union))}
    edge_tables={}
    for filename in ("competition_edges.csv","competition_graph_edges.csv","block_edge_stability.csv","solver_edge_agreement.csv"):
        rows=records(out/filename)
        lookup={(row["source_identity"],row["replacement_identity"]):row for row in rows}
        require(len(lookup)==len(rows) and set(lookup)==expected_edges,f"EDGE_MEMBERSHIP: {filename}")
        edge_tables[filename]=lookup
    for pair,row in edge_tables["competition_edges.csv"].items():
        j,k=(by_name[name] for name in pair)
        use=eligibility[:,j]
        count=int(use.sum())
        g=float(global_d["gains"][j,k])
        numerator=int(presence[:,j,k].sum())
        top_count=int(np.sum((tops[:,j]==k)&use))
        fraction=numerator/count if count else np.nan
        consistency=top_count/count if count else np.nan
        da=np.array([item["delta_residual"][j,k] for item in blocks])[use]
        ra=np.array([item["residual"][j,k] for item in blocks])[use]
        dsum=summary(da)
        numbers=dict(replacement_gain=g,replacement_fraction=g/(global_d["totals"][j]+EPS),
            redistribution_total=global_d["totals"][j],spectral_cosine_full=physical["full"][j,k],
            spectral_cosine_reduced=physical["reduced"][j,k],residual_absorption=global_d["residual"][j,k],
            delta_residual_absorption=global_d["delta_residual"][j,k],edge_presence_fraction=fraction,
            top_replacement_consistency=consistency,eligible_block_count=count,
            absorption_median=dsum["median"],absorption_IQR=dsum["IQR"])
        shared=physical["fragments"][:,j]&physical["fragments"][:,k]
        shared_union=physical["fragments"][:,j]|physical["fragments"][:,k]
        numbers.update(shared_fragment_count=shared.sum(),shared_fragment_fraction=shared.sum()/shared_union.sum() if shared_union.any() else 0)
        sm=spatial_values(maps["NNLS"][j],maps["NNLS"][k])
        si=spatial_values(maps["ISTA"][j],maps["ISTA"][k])
        numbers.update(spatial_cosine=sm["spatial_cosine"],pearson_spatial=sm["pearson_spatial"],
            spatial_overlap=sm["support_overlap"],weighted_overlap=sm["weighted_overlap"],
            ista_spatial_cosine=si["spatial_cosine"],ista_spatial_overlap=si["support_overlap"])
        check_numbers(row,numbers,f"EDGE/{pair}")
        require(flag(row["stable_edge"])==bool(stable[j,k]), "EDGE_STABILITY")
        require(flag(row["presence_persistent_edge"])==bool(persistent[j,k]), "EDGE_PERSISTENT_PRESENCE")
        require(flag(row["nnls_edge_present"])==bool(global_presence[j,k]), "GLOBAL_EDGE_PRESENCE")
        require(row["ista_edge_present"]=="" and row["solver_edge_agreement"] not in {"PASS","True","SOLVER_DEPENDENT_EDGE"}, "FABRICATED_ISTA_EDGE")
        block=edge_tables["block_edge_stability.csv"][pair]
        check_numbers(block,dict(eligible_block_count=count,edge_presence_count=numerator,
            edge_presence_fraction=fraction,unique_top_count=top_count,top_replacement_consistency=consistency,
            absorption_median=dsum["median"],absorption_IQR=dsum["IQR"],
            residual_absorption_median=summary(ra)["median"],gain_median=summary(gains[use,j,k])["median"],
            gain_IQR=summary(gains[use,j,k])["IQR"]),f"BLOCK/{pair}")
        require(flag(block["stable_edge"])==bool(stable[j,k]), "BLOCK_STABILITY")
        require(flag(block["presence_persistent_edge"])==bool(persistent[j,k]), "BLOCK_PERSISTENT_PRESENCE")
        check_summary(block["replacement_rank_stability"],rank_values[j],f"EDGE_RANK/{pair}")
        graph=edge_tables["competition_graph_edges.csv"][pair]
        for key in ("replacement_gain","replacement_fraction","delta_residual_absorption","edge_presence_fraction","top_replacement_consistency"):
            close(numeric(graph[key]),numbers[key],f"GRAPH_EDGE/{pair}/{key}")
        require(flag(graph["stable_edge"])==bool(stable[j,k]), "GRAPH_STABILITY")
        require(flag(graph["presence_persistent_edge"])==bool(persistent[j,k]), "GRAPH_PERSISTENT_PRESENCE")
        other=edge_tables["solver_edge_agreement.csv"][pair]
        for key in ("ista_edge_present","same_top_replacement","top5_replacement_overlap","redistribution_rank_corr","edge_presence_both"):
            require(other[key]=="",f"FABRICATED_SOLVER_AGREEMENT: {key}")
    graph=read(out/"competition_graph_summary.json")
    stable_count=int(stable.sum())
    require(graph["node_count"]==377 and graph["edge_count"]==len(expected_edges)
            and graph["strict_stable_edge_count"]==stable_count,"GRAPH_COUNTS")
    close(numeric(graph["strict_stable_edge_fraction"]),stable_count/len(expected_edges) if expected_edges else np.nan,"GRAPH_STABLE_FRACTION")
    any_pairs=int(np.triu(stable|stable.T,1).sum())
    reciprocal_pairs=int(np.triu(stable&stable.T,1).sum())
    require(graph["stable_pair_count"]==any_pairs,"STABLE_PAIR_COUNT")
    require(graph["reciprocal_stable_pair_count"]==reciprocal_pairs,"RECIPROCAL_STABLE_PAIR_COUNT")
    reciprocal_persistent=persistent&persistent.T
    cliques=maximal_cliques(reciprocal_persistent)
    saved_cliques=sorted(tuple(sorted(by_name[name] for name in clique)) for clique in graph["maximal_cliques"])
    require(saved_cliques==cliques and graph["stable_clique_count"]==len(cliques)
            and graph["largest_clique_size"]==max(map(len,cliques),default=0),"PERSISTENT_CLIQUE_ENUMERATION")
    require(graph["presence_persistent_edge_count"]==int(persistent.sum())
            and graph["reciprocal_presence_persistent_pair_count"]==int(np.triu(reciprocal_persistent,1).sum()),"PERSISTENT_GRAPH_COUNTS")
    clique_members=set().union(*(set(clique) for clique in cliques)) if cliques else set()
    close(graph["competition_clique_coverage"],len(clique_members)/n,"CLIQUE_COVERAGE")
    expected_strong,expected_weak=anchor_review(graph["anchors"],names,master,agreement,global_d,physical,support_global)
    synthetic=synthetic_review(out,paths,Av,names,groups,diagnostics[-2:],physical_v)
    comparison_review(out,names,master,expected_ce,synthetic,expected_strong,expected_weak,stable)
    decision=read(out/"decision_summary.json")
    require(decision["Q1_stable_spectral_competition"] in ("PARTIAL","NO")
            and decision["Q3_competitive_decoy_assignment"] in ("PARTIALLY_SUPPORTED","NOT_SUPPORTED")
            and decision["Q4_next_route"]!="COMPETITIVE_DECOY_ASSIGNMENT", "UNSUPPORTED_CROSS_SOLVER_DECISION")
    coverage=read(out/"decoy_coverage_audit.json")
    require(coverage["solver_consistent_edge_fraction"] is None,"FABRICATED_SOLVER_COVERAGE")
    close(coverage["competition_clique_coverage"],len(clique_members)/n,"COVERAGE_CLIQUES")
    require(graph["spectral_fold_secondary"]["cache_count"]==34
            and graph["spectral_fold_secondary"]["not_spatial_stability"] is True
            and graph["spectral_fold_secondary"]["not_pooled_with_primary"] is True,"SPECTRAL_BLOCK_SCOPE")
    source_paths={str(Path(item["path"]).resolve()) for item in read(out/"provenance.json")["sources"]}
    for block in range(34):
        for extension in ("json","npz"):
            path=Path(paths["spectral_blocks"])/f"block_{block:03d}.{extension}"
            require(str(path.resolve()) in source_paths,f"SPECTRAL_CACHE_NOT_HASH_VERIFIED: {path}")
    report["table_review"]=dict(molecular_tables=5,CE29_global_molecular_metrics=377,
        all_edge_metrics_recomputed=len(expected_edges),all_block_edge_denominators_recomputed=True,
        all_pairwise_block_rank_statistics_recomputed=True,strict_stable_edges=stable_count,
        strict_stable_unordered_pairs=any_pairs,reciprocal_strict_stable_pairs=reciprocal_pairs,
        graph_membership_exact=True,all_V58_molecular_labels_checked=754,
        original_spectral_fold_cache_hashes_verified=34,
        source_data_outcomes_preserved=True,ISTA_deletion_evidence_not_fabricated=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "results/real_ce29_competition_audit_v1")
    args = parser.parse_args()
    try:
        result = review(args.output)
    except Exception as exc:
        result = dict(status=getattr(exc, "status", "STOP_INVALID_AUDIT"),
                      error=f"{type(exc).__name__}: {exc}", solver_calls=0)
    (args.output / "validation_report.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, allow_nan=False), flush=True)
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
