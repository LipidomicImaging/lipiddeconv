"""Frozen, descriptive CE29 replacement geometry from completed cached fits.

This module performs no fitting, selection, calibration, or external discovery.
Real observations never receive truth labels. Synthetic labels enter only the
last, descriptive comparison after the CE29 metrics have been computed.
"""
from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy import stats
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

EPS = 1e-12
MISSING_EDGE = "NOT_EVALUABLE_CACHE_ABSENT"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [clean(v) for v in value]
    if isinstance(value, np.generic):
        return clean(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def write_json(path, value):
    Path(path).write_text(json.dumps(clean(value), ensure_ascii=False, indent=2,
                                   allow_nan=False) + "\n", encoding="utf-8")


def write_csv(path, rows, columns=None):
    rows = list(rows)
    fields = columns or list(dict.fromkeys(k for row in rows for k in row))
    if not fields:
        fields = ["status"]
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            value = clean(row)
            writer.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict))
                             else v for k, v in value.items()})


def finite(values):
    x = np.asarray([np.nan if v is None else v for v in values], dtype=float)
    return x[np.isfinite(x)]


def distribution(values):
    x = np.sort(finite(values))
    if not len(x):
        return dict(n=0, median=None, q25=None, q75=None, IQR=None)
    a, b, c = np.quantile(x, [.25, .5, .75])
    return dict(n=len(x), median=b, q25=a, q75=c, IQR=c-a,
                minimum=x[0], maximum=x[-1])


def ecdf(values):
    x = np.sort(finite(values))
    if not len(x):
        return dict(n=0, x=[], cumulative_probability=[])
    u, counts = np.unique(x, return_counts=True)
    return dict(n=len(x), x=u, cumulative_probability=np.cumsum(counts)/len(x))


def correlation(a, b, rank=False):
    a, b = np.asarray(a, float), np.asarray(b, float)
    good = np.isfinite(a) & np.isfinite(b)
    a, b = a[good], b[good]
    if len(a) < 2 or np.ptp(a) == 0 or np.ptp(b) == 0:
        return np.nan
    return float(stats.spearmanr(a, b).statistic if rank else np.corrcoef(a, b)[0, 1])


def cosine(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if np.linalg.norm(a)==0 or np.linalg.norm(b)==0:
        return np.nan
    return float(a @ b / (np.linalg.norm(a)*np.linalg.norm(b)+EPS))


def unit_columns(a):
    n = np.linalg.norm(a, axis=0)
    return np.divide(a, n[None, :], out=np.zeros_like(a, dtype=float), where=n[None, :] > 0)


def alias_groups(names):
    molecular = list(dict.fromkeys(names))
    index = {n: j for j, n in enumerate(molecular)}
    groups = [[] for _ in molecular]
    for j, name in enumerate(names):
        groups[index[name]].append(j)
    return molecular, groups


def aggregate(values, groups):
    return np.stack([np.sum(values[..., g], axis=-1) for g in groups], axis=-1)


def spectral_arrays(A, groups, fragment_candidate_support):
    normal = unit_columns(A)
    rep = unit_columns(np.column_stack([normal[:, g].mean(axis=1) for g in groups]))
    support = np.column_stack([np.any(A[:, g] > 0, axis=1) for g in groups])
    occupancy = support.sum(axis=1)
    weights = 1 / np.sqrt(np.maximum(occupancy, 1))
    reduced = unit_columns(rep * weights[:, None])
    fragments = np.column_stack([np.any(fragment_candidate_support[:, g], axis=1)
                                 for g in groups])
    fragment_occupancy = fragments.sum(axis=1)
    common = fragments & (occupancy[:, None] > 1)
    specific = fragments & (occupancy[:, None] == 1)
    sim = rep.T @ rep
    sim_reduced = reduced.T @ reduced
    np.fill_diagonal(sim, np.nan)
    np.fill_diagonal(sim_reduced, np.nan)
    return dict(rep=rep, reduced=reduced, support=support, fragments=fragments,
                common=common, specific=specific, occupancy=occupancy,
                fragment_occupancy=fragment_occupancy, sim=sim, sim_reduced=sim_reduced)


def load_physical_mapping(paths, A, groups, metadata):
    with np.load(paths["components_npz"], allow_pickle=False) as z:
        components, owner = z["components"], z["owner"]
    cm = read(paths["component_metadata"])
    formula_rows = [json.loads(x) for x in Path(paths["channel_formulas"]).read_text(encoding="utf-8").splitlines() if x]
    lookup = defaultdict(list)
    for r in formula_rows:
        lookup[(r["entry_id"], r.get("isotope_ion_formula", r["formula"]))].append(r)
    fragment_support = np.zeros(A.shape, bool)
    labels = [set() for _ in metadata]
    fragment_count = np.zeros(A.shape[1], int)
    unmatched = []
    for i, r in enumerate(cm):
        j = int(owner[i])
        if j != r["candidate_index"] or r["candidate_id"] != metadata[j]["candidate_id"]:
            raise RuntimeError("STOP_INVALID_AUDIT:PHYSICAL_OWNER_MAPPING")
        if r["kind"] != "fragment":
            continue
        fragment_count[j] += 1
        fragment_support[:, j] |= (components[:, i] > 0) & (A[:, j] > 0)
        joined = lookup.get((r["selected_source_entry_id"], r["isotope_ion_formula"]), [])
        if not joined:
            unmatched.append(i)
        for original in joined:
            labels[j].add(original.get("fragment_mechanism", original["channel"]))
    molecular_labels = [sorted(set().union(*(labels[j] for j in group))) for group in groups]
    return fragment_support, molecular_labels, dict(fragment_component_count=int(fragment_count.sum()),
            mapped_fragment_components=int(fragment_count.sum())-len(unmatched),
            unmapped_component_indices=unmatched,
            chemical_rule_status="VERIFIED_SOURCE_FRAGMENT_LABELS" if not unmatched else "DIAGNOSTIC_RULE_NOT_VERIFIED",
            statistical_specific_support_is_chemical_diagnostic=False,
            parent_components_excluded_from_fragment_geometry=True)


def support_rows(A, b, candidate_mean, groups, spectral):
    contributions = np.column_stack([A[:, g] @ candidate_mean[g] for g in groups])
    total = contributions.sum(axis=1)
    attr = contributions / (total[:, None] + EPS)
    rows = []
    for j in range(len(groups)):
        common, specific, frag = (spectral[k][:, j] for k in ("common", "specific", "fragments"))
        specific_b = b[specific]
        weights = contributions[specific, j]
        rows.append(dict(common_fragment_support=float(b[common].sum()),
            specific_fragment_support=float(specific_b.sum()),
            common_fragment_count=int(common.sum()), specific_fragment_count=int(specific.sum()),
            fragment_channel_count=int(frag.sum()),
            specific_fragment_observed_fraction=None,
            observation_rule_status="FROZEN_CHANNEL_DETECTION_THRESHOLD_NOT_AVAILABLE",
            positive_B_fraction=float(np.mean(specific_b > 0)) if specific.any() else None,
            diagnostic_attribution_mean=float(attr[frag, j].mean()) if frag.any() else None,
            diagnostic_attribution_max=float(attr[frag, j].max()) if frag.any() else None,
            specific_support_attribution=float(attr[specific, j].mean()) if specific.any() else None,
            specific_support_B_weighted_attribution=float(specific_b @ attr[specific, j]/(specific_b.sum()+EPS)) if specific.any() else None,
            specific_predicted_signal=float(weights.sum()),
            attribution_scope="candidate spectral contributions; arithmetic channel mean; all aliases",
            support_unit="sum of unchanged b values on specified channels; no normalization",
            diagnostic_label="LIBRARY_SHARING_SUPPORT_NOT_VERIFIED_IDENTITY_DIAGNOSTIC"))
    return rows


def compute_deletion(A, b, cache, groups, representatives):
    """Array-only diagnostic; no fitting. Kept public for schema inspection."""
    full = np.asarray(cache["full_x"], float)
    deleted = np.asarray(cache["deleted_x"], float)
    molecular_full = aggregate(full, groups)
    molecular_deleted = aggregate(deleted, groups)
    gain = np.maximum(molecular_deleted - molecular_full[None, :], 0)
    np.fill_diagonal(gain, 0)
    tol = 64*np.finfo(np.float64).eps*max(A.shape)*np.maximum(
        max(1., float(full.max())), deleted.max(axis=1))
    eligible = molecular_full > tol
    top = np.full(len(groups), -1, int)
    top_ties = []
    for j, row in enumerate(gain):
        maximum = float(row.max())
        tied = np.flatnonzero(row >= maximum-tol[j]).tolist() if maximum > tol[j] else []
        top_ties.append(tied)
        if len(tied) == 1:
            top[j] = tied[0]
    rfull = b-A@full
    residuals = b[None, :]-deleted@A.T
    delta = residuals-rfull[None, :]
    rnorm = np.linalg.norm(residuals, axis=1)
    dnorm = np.linalg.norm(delta, axis=1)
    repnorm = np.linalg.norm(representatives, axis=0)
    absorption = residuals @ representatives / (rnorm[:, None]*repnorm[None, :]+EPS)
    delta_absorption = delta @ representatives / (dnorm[:, None]*repnorm[None, :]+EPS)
    removed = np.column_stack([A[:, g]@full[g] for g in groups]).T
    positive_replacement = np.maximum(deleted-full[None, :], 0)@A.T
    source_norm = np.linalg.norm(removed, axis=1)
    replnorm = np.linalg.norm(positive_replacement, axis=1)
    signal_cos = np.sum(removed*positive_replacement, axis=1)/(source_norm*replnorm+EPS)
    projection = np.sum(removed*positive_replacement, axis=1)/(source_norm**2+EPS)
    q = float(np.asarray(cache["q_full"]))
    qdel = np.asarray(cache["q_deleted"], float)
    return dict(full=full, deleted=deleted, molecular_full=molecular_full, gain=gain,
                tol=tol, eligible=eligible, top=top, top_ties=top_ties,
                total=gain.sum(axis=1), absorption=absorption, delta_absorption=delta_absorption,
                signal_cosine=signal_cos, signal_projection=projection,
                q_full=q, q_deleted=qdel, absolute=qdel-q, relative=(qdel-q)/(q+EPS),
                residual_norm=rnorm, delta_residual_norm=dnorm)


def diagnostic(path, A, groups, rep):
    with np.load(path, allow_pickle=False) as z:
        cache = {k: z[k] for k in z.files}
    for j, group in enumerate(groups):
        if np.any(cache["deleted_x"][j, group] != 0):
            raise RuntimeError("STOP_INVALID_AUDIT:DELETION_ALIAS_NONZERO")
    return compute_deletion(A, cache["b"], cache, groups, rep)


def top_indices(row, count=5):
    return [int(k) for k in np.argsort(-row, kind="stable")[:count] if row[k] > 0]


def deletion_row(j, name, d, names):
    top5 = top_indices(d["gain"][j])
    top = top5[0] if top5 else None
    total = d["total"][j]
    return dict(identity_id=j, lipid_name=name, q_full=d["q_full"], q_deleted=d["q_deleted"][j],
        absolute_necessity_loss=d["absolute"][j], relative_necessity_loss=d["relative"][j],
        full_diagnostic_abundance=d["molecular_full"][j], redistribution_total=total,
        top_replacement_identity=names[top] if top is not None else None,
        top_replacement_gain=d["gain"][j, top] if top is not None else 0.,
        top_replacement_fraction=d["gain"][j, top]/(total+EPS) if top is not None else 0.,
        top5_replacement_identities=[names[k] for k in top5],
        top5_replacement_gains=[d["gain"][j, k] for k in top5],
        unique_top_replacement=names[d["top"][j]] if d["top"][j] >= 0 else None,
        top_tie_identities=[names[k] for k in d["top_ties"][j]],
        source_eligible=d["eligible"][j], numerical_edge_tolerance=d["tol"][j],
        residual_absorption=d["absorption"][j, top] if top is not None else None,
        delta_residual_absorption=d["delta_absorption"][j, top] if top is not None else None,
        removed_vs_positive_replacement_signal_cosine=d["signal_cosine"][j],
        removed_signal_projection_fraction=d["signal_projection"][j],
        deletion_residual_norm=d["residual_norm"][j],
        delta_residual_norm=d["delta_residual_norm"][j],
        residual_direction_nonzero=bool(d["residual_norm"][j]>EPS),
        delta_residual_direction_nonzero=bool(d["delta_residual_norm"][j]>EPS),
        diagnostic_scope="global foreground mean NNLS; not mean pixelwise NNLS",
        absorption_interpretation="raw postfit residual direction, constrained by NNLS KKT; not absorbed signal fraction")


def spatial_metrics(a, b):
    positive_a, positive_b = a > 0, b > 0
    union = int(np.count_nonzero(positive_a | positive_b))
    return dict(spatial_cosine=cosine(a, b), pearson_spatial=correlation(a, b),
        support_overlap=float(np.count_nonzero(positive_a & positive_b))/union if union else None,
        weighted_overlap=float(np.minimum(a, b).sum())/(float(np.maximum(a, b).sum())+EPS),
        support_overlap_gate=0.)


def effect_comparison(a, b):
    a, b = finite(a), finite(b)
    result = {"ce29_"+k: v for k, v in distribution(a).items()}
    result.update({"v58_"+k: v for k, v in distribution(b).items()})
    if len(a) and len(b):
        # Cliff delta is P(synthetic > CE29)-P(synthetic < CE29), exact ties.
        sorted_b = np.sort(b)
        wins = np.searchsorted(sorted_b, a, side="left").sum()
        losses = (len(b)-np.searchsorted(sorted_b, a, side="right")).sum()
        result.update(cliffs_delta=float(losses-wins)/(len(a)*len(b)),
            wasserstein_distance=stats.wasserstein_distance(a, b),
            ks_statistic=stats.ks_2samp(a, b).statistic)
    else:
        result.update(cliffs_delta=None, wasserstein_distance=None, ks_statistic=None)
    return result


def bool_value(value):
    return value is True or str(value).strip().lower() in ("true", "1", "1.0")


def synthetic_records(path):
    if Path(path).suffix == ".csv":
        with Path(path).open(encoding="utf-8-sig", newline="") as f:
            rows = list(csv.DictReader(f))
    else:
        rows = read(path)
    return {r["lipid_name"]: r for r in rows}


def maximal_presence_cliques(adjacency):
    """Exact deterministic Bron--Kerbosch maximal cliques of size >= 3.

    The graph is reciprocal presence persistence, independent of unique-top
    stability. Integer bit sets keep intersections cheap without dependencies.
    No outcome-dependent truncation or transitive clique completion is used.
    """
    adjacency = np.asarray(adjacency, dtype=bool)
    if not np.array_equal(adjacency, adjacency.T) or np.diag(adjacency).any():
        raise RuntimeError("STOP_INVALID_AUDIT:CLIQUE_GRAPH_NOT_SIMPLE_UNDIRECTED")
    neighbors = []
    for row in adjacency:
        value = 0
        for j in np.flatnonzero(row):
            value |= 1 << int(j)
        neighbors.append(value)
    cliques = []

    def search(chosen, possible, excluded):
        if len(chosen) + possible.bit_count() < 3:
            return
        if not possible and not excluded:
            cliques.append(sorted(chosen))
            return
        pool = possible | excluded
        pivot = -1
        best = -1
        while pool:
            bit = pool & -pool
            j = bit.bit_length()-1
            score = (possible & neighbors[j]).bit_count()
            if score > best:
                pivot, best = j, score
            pool ^= bit
        candidates = possible & ~neighbors[pivot] if pivot >= 0 else possible
        while candidates:
            bit = candidates & -candidates
            j = bit.bit_length()-1
            search(chosen+[j], possible & neighbors[j], excluded & neighbors[j])
            possible &= ~bit
            excluded |= bit
            candidates ^= bit

    # Vertices with fewer than two neighbors cannot occur in any size>=3
    # clique. Removing them does not affect maximality for those cliques.
    possible = 0
    for j, row in enumerate(adjacency):
        if int(row.sum()) >= 2:
            possible |= 1 << j
    search([], possible, 0)
    return sorted(cliques)


def spectral_fold_secondary(paths, groups, names, spectral):
    """Reuse every original full/delete spectral-fold cache, separately."""
    folds, source_records = [], [[] for _ in names]
    receivers = Counter()
    for block in range(34):
        stem=paths["spectral_blocks"]/f"block_{block:03d}"
        binding=read(stem.with_suffix(".json"))
        with np.load(stem.with_suffix(".npz"),allow_pickle=False) as z:
            full=z["full_x"]; deleted=z["deleted_x"]
            if z["deleted_names"].tolist()!=names:
                raise RuntimeError("STOP_INVALID_AUDIT:SPECTRAL_CACHE_IDENTITY_ORDER")
            molecular=aggregate(full,groups)
            gain=np.maximum(aggregate(deleted,groups)-molecular[None,:],0)
            np.fill_diagonal(gain,0)
            total=gain.sum(axis=1)
            held_delta=z["deleted_held_loss"]-float(z["full_held_loss"])
            train_delta=z["deleted_train_loss"]-float(z["full_train_loss"])
            tolerance=64*np.finfo(float).eps*max(len(binding["train_indices"]),len(full))*np.maximum(max(1.,float(full.max())),deleted.max(axis=1))
            eligible=molecular>tolerance
            top=np.argmax(gain,axis=1)
            fractions=gain[np.arange(len(names)),top]/(total+EPS)
            for j in range(len(names)):
                maximum=gain[j,top[j]]
                tie_count=int(np.sum(gain[j]>=maximum-tolerance[j])) if maximum>tolerance[j] else 0
                unique=bool(eligible[j] and tie_count==1)
                if unique:
                    receivers[names[top[j]]]+=1
                source_records[j].append(dict(eligible=bool(eligible[j]),
                    unique_top=names[top[j]] if unique else None,
                    redistribution_total=float(total[j]), top_fraction=float(fractions[j]),
                    top_full_spectral_cosine=float(spectral["sim"][j,top[j]]) if maximum>0 else None,
                    held_loss_delta=float(held_delta[j]), training_loss_delta=float(train_delta[j])))
            folds.append(dict(block_index=block, held_channel_count=len(binding["held_indices"]),
                training_channel_count=len(binding["train_indices"]),eligible_source_count=int(eligible.sum()),
                numerical_edge_count=int(np.sum(gain>tolerance[:,None])),
                redistribution_total=distribution(total[eligible]),
                top_replacement_fraction=distribution(fractions[eligible]),
                held_loss_delta=distribution(held_delta),training_loss_delta=distribution(train_delta)))
    by_identity=[]
    for j,rows in enumerate(source_records):
        selected=[r for r in rows if r["eligible"]]
        tops=Counter(r["unique_top"] for r in selected if r["unique_top"] is not None)
        by_identity.append(dict(identity_id=j,lipid_name=names[j],eligible_spectral_fold_count=len(selected),
            unique_top_counts=dict(tops),
            top_replacement_consistency=max(tops.values())/len(selected) if selected and tops else 0. if selected else None,
            redistribution_total=distribution([r["redistribution_total"] for r in selected]),
            top_replacement_fraction=distribution([r["top_fraction"] for r in selected]),
            top_full_spectral_cosine=distribution([r["top_full_spectral_cosine"] for r in selected]),
            held_loss_delta=distribution([r["held_loss_delta"] for r in rows]),
            training_loss_delta=distribution([r["training_loss_delta"] for r in rows])))
    return dict(cache_count=34,scope="same whole-foreground mean b; leave-physical-channel-block-out NNLS",
        not_spatial_stability=True,not_pooled_with_primary=True,
        coefficient_redistribution_definition="positive molecular alias-sum difference after deletion",
        loss_scopes="training-channel SSE and held-channel prediction SSE explicitly separate",
        folds=folds,identity_summaries=by_identity,unique_top_receiver_counts=dict(receivers.most_common()))


def summarize(root: Path, out: Path):
    root, out = Path(root), Path(out)
    contract = read(out/"audit_contract.json")
    paths = {k: Path(v) for k, v in contract["source_paths"].items()}
    with np.load(out/"diagnostic_inputs.npz", allow_pickle=False) as z:
        A, A_v58, bs, candidate_names = z["A"], z["A_v58"], z["bs"], z["names"].tolist()
    names, groups = alias_groups(candidate_names)
    metadata = [json.loads(x) for x in paths["metadata_jsonl"].read_text(encoding="utf-8").splitlines() if x]
    if [r["lipid_name"] for r in metadata] != candidate_names:
        raise RuntimeError("STOP_INVALID_AUDIT:CANDIDATE_ORDER")
    n = len(names)
    if n != 377 or len(candidate_names) != 391 or bs.shape != (67, 1084):
        raise RuntimeError("STOP_INVALID_AUDIT:DIAGNOSTIC_SCOPE")
    fragments, labels, mapping = load_physical_mapping(paths, A, groups, metadata)
    spec = spectral_arrays(A, groups, fragments)
    spec_v58 = spectral_arrays(A_v58, groups, fragments & (A_v58 > 0))
    classes = [metadata[g[0]]["lipid_class"] for g in groups]
    adducts = [sorted({metadata[j]["adduct"] for j in g}) for g in groups]
    rules = [sorted({metadata[j]["rule_sheet"] for j in g}) for g in groups]
    precursors = [[float(metadata[j]["mz"]) for j in g] for g in groups]
    mask = np.load(paths["mask"], allow_pickle=False)
    ista = np.load(paths["ista_x"], allow_pickle=False)[:, mask].astype(float)
    with np.load(paths["nnls_arrays"], allow_pickle=False) as z:
        nnls = z["X_hat"][:, mask].astype(float)
    pixel_count = int(mask.sum())
    ista_m = np.stack([ista[g].sum(axis=0) for g in groups])
    nnls_m = np.stack([nnls[g].sum(axis=0) for g in groups])
    ista_mean, nnls_mean = ista.mean(axis=1), nnls.mean(axis=1)
    ista_mm, nnls_mm = ista_m.mean(axis=1), nnls_m.mean(axis=1)
    current = {(r["method"], r["lipid_name"]): r for r in read(paths["current_records"])}
    supported_ista = np.array([bool(current[("ISTA", name)]["raw_solver_reported"]) for name in names])
    supported_nnls = np.array([bool(current[("NNLS", name)]["raw_solver_reported"]) for name in names])
    support_ista = support_rows(A, bs[0], ista_mean, groups, spec)
    support_nnls = support_rows(A, bs[0], nnls_mean, groups, spec)
    diag_global = diagnostic(out/"diagnostics/CE29_GLOBAL.npz", A, groups, spec["rep"])
    support_diagnostic = support_rows(A, bs[0], diag_global["full"], groups, spec)
    # Retain only numerical summaries between cases, not copies of deleted arrays.
    blocks = []
    for i in range(64):
        d = diagnostic(out/"diagnostics"/f"CE29_BLOCK_{i:03d}.npz", A, groups, spec["rep"])
        d.pop("full"); d.pop("deleted")
        blocks.append(d)
    gain_stack = np.stack([d["gain"] for d in blocks])
    eligible = np.stack([d["eligible"] for d in blocks])
    top_stack = np.stack([d["top"] for d in blocks])
    tolerances = np.asarray([d["tol"] for d in blocks])
    presence = (gain_stack > tolerances[:, :, None]) & eligible[:, :, None]
    eligible_count = eligible.sum(axis=0)
    union = (diag_global["gain"] > diag_global["tol"][:, None]) | np.any(
        gain_stack > tolerances[:, :, None], axis=0)
    np.fill_diagonal(union, False)
    stable = np.zeros((n, n), bool)
    for j in range(n):
        k = top_stack[eligible[:, j], j]
        if len(k) >= 2 and np.all(k == k[0]) and k[0] >= 0:
            stable[j, k[0]] = True
    persistent_presence = (eligible_count[:, None] >= 2) & (
        presence.sum(axis=0) == eligible_count[:, None])
    np.fill_diagonal(persistent_presence, False)
    master, solver_rows, geometry, specific_rows, deletion = [], [], [], [], []
    agreement = []
    for j, name in enumerate(names):
        ci, cn = current[("ISTA", name)], current[("NNLS", name)]
        outcome_i = "retained" if ci["retained_by_DEV10_TRANSFER"] else "rejected" if ci["raw_solver_reported"] else "unreported"
        outcome_n = "retained" if cn["retained_by_DEV10_TRANSFER"] else "rejected" if cn["raw_solver_reported"] else "unreported"
        support = "reported_by_both" if supported_ista[j] and supported_nnls[j] else "ista_only" if supported_ista[j] else "nnls_only" if supported_nnls[j] else "neither"
        master.append(dict(identity_id=j, lipid_name=name, lipid_class=classes[j], candidate_count=len(groups[j]),
            candidate_indices=groups[j], adducts=adducts[j], rule_sheets=rules[j],
            ista_reported=supported_ista[j], nnls_reported=supported_nnls[j], support_agreement=support,
            ista_mean_abundance=ista_mm[j], nnls_mean_abundance=nnls_mm[j],
            ista_total_abundance=float(ista_m[j].sum()), nnls_total_abundance=float(nnls_m[j].sum()),
            abundance_definition="foreground mean/total of ALL same-name aliases; no normalization",
            ista_reported_alias_mean_abundance=ci["X_hat"], nnls_reported_alias_mean_abundance=cn["X_hat"],
            current_confidence_outcome=outcome_i+"/"+outcome_n,
            ista_current_confidence_outcome=outcome_i, nnls_current_confidence_outcome=outcome_n,
            rho_identity=ci["rho_zero"], nnls_rho_identity=cn["rho_zero"],
            current_joint_score=ci["joint_score"], nnls_current_joint_score=cn["joint_score"],
            current_threshold_status="SAVED_DEV10_TRANSFER", current_threshold=.5406530976316042,
            verified_fragment_labels=labels[j]))
        map_metrics = spatial_metrics(ista_m[j], nnls_m[j])
        agreement.append(map_metrics["spatial_cosine"])
        ratio = ista_mm[j]/nnls_mm[j] if nnls_mm[j] > 0 else None
        logratio = float(np.log(ista_mm[j]/nnls_mm[j])) if ista_mm[j] > 0 and nnls_mm[j] > 0 else None
        solver_rows.append(dict(identity_id=j, lipid_name=name, ista_nnls_ratio=ratio,
            ista_nnls_log_ratio=logratio, both_reported=support=="reported_by_both",
            ista_only=support=="ista_only", nnls_only=support=="nnls_only",
            pixelwise_abundance_corr=map_metrics["pearson_spatial"],
            spatial_cosine_ista_nnls=map_metrics["spatial_cosine"],
            status="SOLVER_SUPPORT_DISAGREEMENT" if support in ("ista_only", "nnls_only") else support,
            ratio_is_solver_correctness=False))
        order = np.argsort(-np.nan_to_num(spec["sim"][j], nan=-np.inf), kind="stable")[:5]
        reduced_order = np.argsort(-np.nan_to_num(spec["sim_reduced"][j], nan=-np.inf), kind="stable")[:5]
        nearest = int(order[0])
        shared = spec["fragments"][:, j] & spec["fragments"][:, nearest]
        frag_union = spec["fragments"][:, j] | spec["fragments"][:, nearest]
        same = np.array([c == classes[j] for c in classes]); same[j] = False
        cross = np.array([c != classes[j] for c in classes])
        geometry.append(dict(identity_id=j, lipid_name=name, nearest_neighbor_identity=names[nearest],
            nearest_neighbor_cosine_full=spec["sim"][j, nearest],
            nearest_neighbor_cosine_reduced=spec["sim_reduced"][j, reduced_order[0]],
            nearest_neighbor_identity_reduced=names[reduced_order[0]],
            full_neighbor_matched_cosine_reduced=spec["sim_reduced"][j, nearest],
            nearest_full_tie=bool(np.sum(spec["sim"][j]==spec["sim"][j,nearest])>1),
            nearest_reduced_tie=bool(np.sum(spec["sim_reduced"][j]==spec["sim_reduced"][j,reduced_order[0]])>1),
            top5_neighbor_identities=[names[k] for k in order],
            top5_cosines_full=spec["sim"][j, order], top5_cosines_reduced=spec["sim_reduced"][j, reduced_order],
            top5_neighbor_identities_reduced=[names[k] for k in reduced_order],
            same_class_nearest_cosine=float(np.nanmax(spec["sim"][j, same])) if same.any() else None,
            cross_class_nearest_cosine=float(np.nanmax(spec["sim"][j, cross])) if cross.any() else None,
            shared_fragment_count=int(shared.sum()), shared_fragment_fraction=float(shared.sum())/frag_union.sum() if frag_union.any() else 0.,
            sharing_unit="observed channels, physical fragment envelopes only",
            reduced_definition="full library molecular channel occupancy inverse square-root weight",
            verified_fragment_labels=labels[j]))
        specific_rows.append(dict(identity_id=j, lipid_name=name, **support_ista[j],
            nnls_specific_support_attribution=support_nnls[j]["specific_support_attribution"],
            global_diagnostic_specific_support_attribution=support_diagnostic[j]["specific_support_attribution"],
            verified_fragment_labels=labels[j]))
        row = deletion_row(j, name, diag_global, names)
        vectors = gain_stack[eligible[:,j],j,:]
        varied = np.ptp(vectors,axis=1)>0 if len(vectors) else np.zeros(0,bool)
        rank_corrs = []
        if int(varied.sum())>=2:
            ranked = stats.rankdata(vectors[varied],axis=1)
            centered = ranked-ranked.mean(axis=1,keepdims=True)
            norms = np.linalg.norm(centered,axis=1)
            correlations = (centered@centered.T)/(norms[:,None]*norms[None,:])
            rank_corrs = correlations[np.triu_indices(len(ranked),1)]
        row.update(eligible_block_count=int(eligible_count[j]),
            replacement_rank_stability=distribution(rank_corrs),
            unique_top_by_block=[names[k] if k >= 0 else None for k in top_stack[:, j]],
            eligible_blocks=np.flatnonzero(eligible[:, j]).tolist())
        deletion.append(row)
    intersection = int(np.sum(supported_ista & supported_nnls)); support_union = int(np.sum(supported_ista | supported_nnls))
    solver_summary = dict(molecular_identity_count=n, candidate_count=len(candidate_names), foreground_pixels=pixel_count,
        ista_reported=int(supported_ista.sum()), nnls_reported=int(supported_nnls.sum()),
        overlap_count=intersection, ista_only_count=int(np.sum(supported_ista & ~supported_nnls)),
        nnls_only_count=int(np.sum(supported_nnls & ~supported_ista)),
        neither_count=int(np.sum(~supported_ista & ~supported_nnls)), report_set_jaccard=intersection/support_union if support_union else None,
        abundance_pearson=correlation(ista_mm, nnls_mm), abundance_spearman=correlation(ista_mm, nnls_mm, rank=True),
        rank_correlation=correlation(ista_mm, nnls_mm, rank=True),
        abundance_scope="all 377 identities, foreground mean of all production aliases",
        edge_agreement_status=MISSING_EDGE)
    write_csv(out/"identity_master_table.csv", master)
    write_csv(out/"identity_solver_agreement.csv", solver_rows)
    write_csv(out/"identity_spectral_geometry.csv", geometry)
    write_csv(out/"identity_specific_support.csv", specific_rows)
    write_csv(out/"identity_deletion_metrics.csv", deletion)

    edge_rows, block_rows, solver_edges = [], [], []
    for j, k in zip(*np.nonzero(union)):
        use = eligible[:, j]
        count = int(eligible_count[j])
        fraction = float(presence[:, j, k].sum())/count if count else None
        consistency = float(np.sum((top_stack[:, j] == k) & use))/count if count else None
        da = [d["delta_absorption"][j, k] for b, d in enumerate(blocks) if use[b]]
        aa = [d["absorption"][j, k] for b, d in enumerate(blocks) if use[b]]
        dist = distribution(da)
        shared = spec["fragments"][:, j] & spec["fragments"][:, k]
        shared_union = spec["fragments"][:, j] | spec["fragments"][:, k]
        sm = spatial_metrics(nnls_m[j], nnls_m[k])
        si = spatial_metrics(ista_m[j], ista_m[k])
        gain = diag_global["gain"][j, k]
        row = dict(source_identity_id=int(j), replacement_identity_id=int(k),
            source_identity=names[j], replacement_identity=names[k],
            source_outcome=master[j]["current_confidence_outcome"],
            source_ista_abundance=ista_mm[j], source_nnls_abundance=nnls_mm[j],
            replacement_gain=gain, replacement_fraction=gain/(diag_global["total"][j]+EPS),
            redistribution_total=diag_global["total"][j],
            spectral_cosine_full=spec["sim"][j, k], spectral_cosine_reduced=spec["sim_reduced"][j, k],
            shared_fragment_count=int(shared.sum()), shared_fragment_fraction=float(shared.sum())/shared_union.sum() if shared_union.any() else 0.,
            source_specific_support=support_ista[j]["specific_support_attribution"],
            replacement_specific_support=support_ista[k]["specific_support_attribution"],
            residual_absorption=diag_global["absorption"][j, k],
            delta_residual_absorption=diag_global["delta_absorption"][j, k],
            spatial_cosine=sm["spatial_cosine"], pearson_spatial=sm["pearson_spatial"],
            spatial_overlap=sm["support_overlap"], weighted_overlap=sm["weighted_overlap"],
            ista_spatial_cosine=si["spatial_cosine"], ista_spatial_overlap=si["support_overlap"],
            spatial_overlap_pixel_gate=0.,
            edge_presence_fraction=fraction, top_replacement_consistency=consistency,
            eligible_block_count=count, stable_edge=bool(stable[j, k]),
            presence_persistent_edge=bool(persistent_presence[j,k]),
            ista_edge_present=None, nnls_edge_present=bool(gain > diag_global["tol"][j]),
            global_numerical_edge=bool(gain>diag_global["tol"][j]),
            any_block_numerical_edge=bool(np.any(gain_stack[:,j,k]>tolerances[:,j])),
            solver_edge_agreement=MISSING_EDGE,
            same_class=classes[j]==classes[k], same_adduct=bool(set(adducts[j]) & set(adducts[k])),
            same_rule_sheet=bool(set(rules[j]) & set(rules[k])),
            precursor_distance=min(abs(a-b) for a in precursors[j] for b in precursors[k]),
            absorption_median=dist["median"], absorption_IQR=dist["IQR"],
            absorption_stability_metric="delta_residual_absorption")
        edge_rows.append(row)
        block_rows.append(dict(source_identity=names[j], replacement_identity=names[k],
            eligible_block_count=count, edge_presence_count=int(presence[:, j, k].sum()),
            edge_presence_fraction=fraction, unique_top_count=int(np.sum((top_stack[:, j]==k)&use)),
            top_replacement_consistency=consistency, stable_edge=bool(stable[j,k]),
            presence_persistent_edge=bool(persistent_presence[j,k]),
            replacement_rank_stability=deletion[j]["replacement_rank_stability"],
            absorption_median=dist["median"], absorption_IQR=dist["IQR"],
            residual_absorption_median=distribution(aa)["median"],
            residual_nonzero_direction_count=sum(d["residual_norm"][j]>EPS for b,d in enumerate(blocks) if use[b]),
            delta_residual_nonzero_direction_count=sum(d["delta_residual_norm"][j]>EPS for b,d in enumerate(blocks) if use[b]),
            gain_median=distribution(gain_stack[use, j, k])["median"],
            gain_IQR=distribution(gain_stack[use, j, k])["IQR"],
            stability_unit="original 250-foreground-index processing blocks; last block 87 pixels"))
        solver_edges.append(dict(source_identity=names[j], replacement_identity=names[k],
            ista_edge_present=None, nnls_edge_present=bool(gain>diag_global["tol"][j]),
            same_top_replacement=None, top5_replacement_overlap=None,
            redistribution_rank_corr=None, edge_presence_both=None, solver_edge_agreement=MISSING_EDGE))
    write_csv(out/"competition_edges.csv", edge_rows)
    graph_fields = ["source_identity", "replacement_identity", "replacement_gain", "replacement_fraction",
                    "delta_residual_absorption", "edge_presence_fraction", "top_replacement_consistency", "stable_edge", "presence_persistent_edge"]
    write_csv(out/"competition_graph_edges.csv", [{k:r[k] for k in graph_fields} for r in edge_rows], graph_fields)
    write_csv(out/"block_edge_stability.csv", block_rows)
    write_csv(out/"solver_edge_agreement.csv", solver_edges,
              ["source_identity", "replacement_identity", "ista_edge_present", "nnls_edge_present", "same_top_replacement",
               "top5_replacement_overlap", "redistribution_rank_corr", "edge_presence_both", "solver_edge_agreement"])

    # Descriptive anchors use the predeclared simultaneous quartile intersection.
    anchor_matrix = np.column_stack([agreement, diag_global["relative"],
            [g["nearest_neighbor_cosine_full"] for g in geometry],
            [np.nan if s["specific_support_attribution"] is None else s["specific_support_attribution"] for s in support_diagnostic]])
    complete = np.all(np.isfinite(anchor_matrix), axis=1) & supported_ista & supported_nnls
    strong = np.zeros(n, bool); weak = np.zeros(n, bool)
    if complete.any():
        q25, q75 = np.quantile(anchor_matrix[complete], [.25,.75], axis=0)
        strong = complete & (anchor_matrix[:,0]>=q75[0]) & (anchor_matrix[:,1]>=q75[1]) & (anchor_matrix[:,2]<=q25[2]) & (anchor_matrix[:,3]>=q75[3])
        weak = complete & (anchor_matrix[:,0]<=q25[0]) & (anchor_matrix[:,1]<=q25[1]) & (anchor_matrix[:,2]>=q75[2]) & (anchor_matrix[:,3]<=q25[3])
    else:
        q25=q75=np.full(4,np.nan)
    anchor_ties = strong & weak
    strong &= ~anchor_ties
    weak &= ~anchor_ties
    anchor_summary = dict(complete_both_reported_count=int(complete.sum()), strong_anchor_count=int(strong.sum()),
        weak_anchor_count=int(weak.sum()), strong_anchor_identities=[names[j] for j in np.flatnonzero(strong)],
        unresolved_tie_count=int(anchor_ties.sum()), unresolved_tie_identities=[names[j] for j in np.flatnonzero(anchor_ties)],
        weak_anchor_identities=[names[j] for j in np.flatnonzero(weak)],
        anchor_metrics=["spatial_cosine_ista_nnls", "relative_necessity_loss", "nearest_neighbor_cosine_full", "specific_support_attribution"],
        lower_quartiles=q25, upper_quartiles=q75,
        status="DESCRIPTIVE_INTERNAL_GRADIENT" if strong.any() and weak.any() else "NO_CLEAR_INTERNAL_EVIDENCE_GRADIENT",
        anchor_metric_distributions={key:distribution(anchor_matrix[complete,k]) for k,key in enumerate(["agreement","necessity","ambiguity","specific_attribution"])},
        anchors_are_truth_labels=False)
    synthetic_rows, comparisons, comparison_cdfs = [], [], {}
    synthetic_cases = {}
    for r in (1,2):
        case = f"V58_MILD_CAL_R{r}_K125"
        data = diagnostic(out/"diagnostics"/(case+".npz"), A_v58, groups, spec_v58["rep"])
        specific = support_rows(A_v58, bs[64+r], data["full"], groups, spec_v58)
        records = synthetic_records(paths[f"v58_records_R{r}"])
        case_rows = []
        for j, name in enumerate(names):
            rec = records[name]
            truth = bool_value(rec["molecular_truth"])
            reported = bool_value(rec["raw_solver_reported"])
            label = "TP" if truth and reported else "FP" if reported else "FN" if truth else "UNREPORTED_NONTRUTH"
            base = deletion_row(j, name, data, names)
            nearest = int(np.nanargmax(spec_v58["sim"][j]))
            nearest_reduced = int(np.nanargmax(spec_v58["sim_reduced"][j]))
            shared = spec_v58["fragments"][:,j] & spec_v58["fragments"][:,nearest]
            shared_union = spec_v58["fragments"][:,j] | spec_v58["fragments"][:,nearest]
            row = dict(case_id=case, **base, synthetic_truth=truth, production_reported=reported,
                production_identity_status=label, production_X_hat=float(rec["X_hat"]),
                production_rho=float(rec["rho_zero"]) if rec.get("rho_zero") not in (None,"", "nan", "None") else None,
                nearest_neighbor_cosine_full=spec_v58["sim"][j,nearest],
                nearest_neighbor_cosine_reduced=spec_v58["sim_reduced"][j,nearest_reduced],
                nearest_neighbor_identity_reduced=names[nearest_reduced],
                nearest_neighbor_identity=names[nearest], shared_fragment_count=int(shared.sum()),
                shared_fragment_fraction=float(shared.sum())/shared_union.sum() if shared_union.any() else 0.,
                **specific[j])
            row["diagnostic_scope"]="frozen synthetic foreground mean NNLS; production labels only for comparison"
            case_rows.append(row); synthetic_rows.append(row)
        synthetic_cases[case]=case_rows
    metrics = ["nearest_neighbor_cosine_full", "redistribution_total", "residual_absorption", "delta_residual_absorption",
               "shared_fragment_fraction", "top_replacement_fraction", "removed_vs_positive_replacement_signal_cosine"]
    ce_values = []
    for j in range(n):
        ce_values.append(dict(**deletion[j], nearest_neighbor_cosine_full=geometry[j]["nearest_neighbor_cosine_full"],
                              shared_fragment_fraction=geometry[j]["shared_fragment_fraction"]))
    ce_groups = {"ISTA_REJECTED":np.array([r["ista_current_confidence_outcome"]=="rejected" for r in master]),
                 "NNLS_REJECTED":np.array([r["nnls_current_confidence_outcome"]=="rejected" for r in master]),
                 "ISTA_RETAINED":np.array([r["ista_current_confidence_outcome"]=="retained" for r in master]),
                 "NNLS_RETAINED":np.array([r["nnls_current_confidence_outcome"]=="retained" for r in master]),
                 "BOTH_REJECTED":np.array([r["ista_current_confidence_outcome"]=="rejected" and r["nnls_current_confidence_outcome"]=="rejected" for r in master]),
                 "WEAK_CONSISTENCY":weak, "STRONG_CONSISTENCY":strong,
                 "STRICT_STABLE_REPLACEMENT_SOURCE":stable.any(axis=1)}
    for case, synthetic in synthetic_cases.items():
        for label in ("TP", "FP"):
            sr = [r for r in synthetic if r["production_identity_status"]==label]
            for cname, choose in ce_groups.items():
                cr=[ce_values[j] for j in np.flatnonzero(choose)]
                for metric in metrics:
                    av=[r[metric] for r in cr]; bv=[r[metric] for r in sr]
                    comparisons.append(dict(case_id=case, synthetic_group=label, ce29_group=cname, metric=metric,
                        **effect_comparison(av,bv), independent_sample_claim=False,
                        comparison_unit="one molecular identity within one specimen/case"))
                    comparison_cdfs["|".join((case,label,cname,metric))]=dict(ce29=ecdf(av), synthetic=ecdf(bv))
    write_csv(out/"v58_identity_geometry.csv", synthetic_rows)
    write_csv(out/"v58_vs_ce29_geometry_comparison.csv", comparisons)

    gain = diag_global["gain"]
    proportions = np.divide(gain, gain.sum(axis=1,keepdims=True), out=np.zeros_like(gain), where=gain.sum(axis=1,keepdims=True)>0)
    entropies = -np.sum(np.where(proportions>0,proportions*np.log(np.maximum(proportions, np.finfo(float).tiny)),0),axis=1)
    entropies[gain.sum(axis=1)==0]=np.nan
    replacement_counts=(gain>0).sum(axis=1)
    normalized_entropy=np.divide(entropies, np.log(np.maximum(replacement_counts,1)),
        out=np.full(n,np.nan),where=replacement_counts>=2)
    reciprocal = stable & stable.T
    pairs = [(int(j),int(k)) for j,k in zip(*np.nonzero(np.triu(reciprocal,1)))]
    any_pairs = [(int(j),int(k)) for j,k in zip(*np.nonzero(np.triu(stable|stable.T,1)))]
    stable_edges = int(stable.sum())
    reciprocal_presence = persistent_presence & persistent_presence.T
    cliques = maximal_presence_cliques(reciprocal_presence)
    clique_members = {j for clique in cliques for j in clique}
    scc_count, scc_labels = connected_components(csr_matrix(union), directed=True, connection="strong")
    scc_sizes = np.bincount(scc_labels)
    receiver_counts = Counter(names[d["top"][j]] for d in blocks for j in range(n) if d["eligible"][j] and d["top"][j]>=0)
    concentration_total=sum(receiver_counts.values())
    same_stable=sum(classes[j]==classes[k] for j,k in zip(*np.nonzero(stable)))
    graph_summary=dict(node_count=n, edge_count=len(edge_rows), strict_stable_edge_count=stable_edges,
        strict_stable_edge_fraction=stable_edges/len(edge_rows) if edge_rows else None,
        stable_pair_count=len(any_pairs), stable_pairs=[[names[j],names[k]] for j,k in any_pairs],
        reciprocal_stable_pair_count=len(pairs), reciprocal_stable_pairs=[[names[j],names[k]] for j,k in pairs],
        presence_persistent_edge_count=int(persistent_presence.sum()),
        reciprocal_presence_persistent_pair_count=int(np.triu(reciprocal_presence,1).sum()),
        stable_clique_count=len(cliques), largest_clique_size=max(map(len,cliques),default=0),
        maximal_cliques=[[names[j] for j in clique] for clique in cliques],
        competition_clique_coverage=len(clique_members)/n,
        clique_definition="maximal size>=3 clique in reciprocal presence-persistent graph; presence in every eligible block with >=2 eligible blocks; no unique-top requirement",
        presence_persistent_definition="numerical edge present in every eligible original spatial block; source has >=2 eligible blocks",
        strongly_connected_component_count=int(scc_count), largest_strongly_connected_component_size=int(scc_sizes.max()),
        strongly_connected_component_sizes=sorted(scc_sizes.tolist(), reverse=True),
        all_positive_graph_SCC_is_not_stable_clique=True,
        edge_entropy=distribution(entropies), normalized_edge_entropy=distribution(normalized_entropy),
        top_replacement_concentration=distribution([r["top_replacement_fraction"] for r in deletion]),
        top5_replacement_concentration=distribution([sum(r["top5_replacement_gains"])/(r["redistribution_total"]+EPS) for r in deletion]),
        block_unique_top_receiver_counts=dict(receiver_counts.most_common()),
        top5_receiver_fraction=sum(v for _,v in receiver_counts.most_common(5))/(concentration_total+EPS),
        global_gain_top5_receiver_fraction=float(np.sort(gain.sum(axis=0))[-5:].sum())/(float(gain.sum())+EPS),
        stable_same_class_edge_fraction=same_stable/stable_edges if stable_edges else None,
        stable_cross_class_edge_fraction=(stable_edges-same_stable)/stable_edges if stable_edges else None,
        union_same_class_edge_fraction=sum(r["same_class"] for r in edge_rows)/len(edge_rows) if edge_rows else None,
        global_gain_weighted_same_class_fraction=sum(r["replacement_gain"] for r in edge_rows if r["same_class"])/(float(gain.sum())+EPS),
        solver_agreement=solver_summary, anchors=anchor_summary, physical_rule_mapping=mapping,
        spectral_fold_secondary=spectral_fold_secondary(paths,groups,names,spec),
        threshold_free_analysis=True, stable_definition="unique top in every eligible original spatial processing block, at least 2 eligible blocks")
    write_json(out/"competition_graph_summary.json", graph_summary)
    coverage=dict(nearest_cosine_CDF=ecdf([r["nearest_neighbor_cosine_full"] for r in geometry]),
        residual_absorption_CDF=ecdf([r["residual_absorption"] for r in edge_rows]),
        delta_residual_absorption_CDF=ecdf([r["delta_residual_absorption"] for r in edge_rows]),
        stable_replacement_edge_fraction=graph_summary["strict_stable_edge_fraction"],
        strict_stable_edge_count=stable_edges, edge_presence_fraction_CDF=ecdf([r["edge_presence_fraction"] for r in edge_rows]),
        top_replacement_consistency_CDF=ecdf([r["top_replacement_consistency"] for r in edge_rows]),
        stable_same_class_edge_fraction=graph_summary["stable_same_class_edge_fraction"],
        stable_cross_class_edge_fraction=graph_summary["stable_cross_class_edge_fraction"],
        competition_clique_coverage=len(clique_members)/n,
        competition_clique_definition=graph_summary["clique_definition"],
        presence_persistent_edge_count=int(persistent_presence.sum()),
        solver_consistent_edge_fraction=None, solver_consistency_status=MISSING_EDGE,
        source_removed_vs_positive_replacement_signal_cosine_CDF=ecdf(diag_global["signal_cosine"]),
        anchors=anchor_summary, v58_comparison_ECDFs=comparison_cdfs,
        direction_of_cliffs_delta="P(V58 > CE29)-P(V58 < CE29)",
        no_empirical_decoy_exchangeability_test=True,
        residual_absorption_caveat="An active NNLS replacement is KKT-orthogonal to postfit residual; small cosine cannot by itself reject replacement geometry.")
    write_json(out/"decoy_coverage_audit.json", coverage)
    decision=dict(Q1_stable_spectral_competition="PARTIAL" if edge_rows else "NO",
        Q2_CE29_vs_V58_known_FP_geometry="PARTIAL",
        Q3_competitive_decoy_assignment="PARTIALLY_SUPPORTED" if stable_edges else "NOT_SUPPORTED",
        Q4_next_route="SOLVER_STABILITY_FIRST" if stable_edges else "INSUFFICIENT_EVIDENCE",
        decision_status="COMPETITIVE_DECOY_GEOMETRY_PARTIAL_SOLVER_EVIDENCE_REQUIRED" if stable_edges else "COMPETITIVE_DECOY_NOT_SUPPORTED",
        evidence=dict(edge_count=len(edge_rows), strict_stable_edge_count=stable_edges,
            strict_stable_edge_fraction=graph_summary["strict_stable_edge_fraction"],
            support_jaccard=solver_summary["report_set_jaccard"],
            same_class_strict_stable_fraction=graph_summary["stable_same_class_edge_fraction"],
            top5_receiver_fraction=graph_summary["top5_receiver_fraction"],
            global_top_delta_residual_absorption=distribution([r["delta_residual_absorption"] for r in deletion]),
            global_removed_signal_cosine=distribution(diag_global["signal_cosine"])),
        reasons=["Cross-solver deletion edges cannot be evaluated: only unchanged NNLS diagnostic is available.",
                 "Two synthetic CAL specimens permit mechanism effect comparisons but do not establish equivalence or population generality.",
                 "Raw postfit residual cosine is constrained by KKT and is not an absorption fraction.",
                 "Current production retention supplies descriptive groups and was not changed."],
        limitations=["Real molecular identity correctness, error rates and recall remain unknown.",
                     "Original processing blocks follow foreground index order and are neither randomized nor independent specimens.",
                     "Mean-spectrum replacement need not reproduce every pixelwise replacement.",
                     "Library-exclusive channels do not establish chemically unique identity evidence.",
                     "No exchangeability, null calibration, target-decoy library, or new selector was constructed."],
        permitted_next_execution="NONE; audit ends after independent review and saved result")
    write_json(out/"decision_summary.json", decision)
    lines=["# Real CE29 identity competition / failure geometry audit", "",
      "This audit does not estimate real CE29 FDR and does not label real identities as true or false. It tests whether low-confidence identity assignments exhibit stable library-internal replacement geometry consistent with the known false-assignment mechanism observed under synthetic spectral mismatch.", "",
      f"Audited {n} molecular identities from 391 original columns and {pixel_count} foreground pixels. All same-name aliases participate in every deletion; original production results and DEV10 transfer outcomes remain unchanged.", "",
      f"ISTA reports {solver_summary['ista_reported']}; NNLS reports {solver_summary['nnls_reported']}; overlap {intersection}; support Jaccard {solver_summary['report_set_jaccard']:.6f}. All-alias mean abundance Pearson {solver_summary['abundance_pearson']:.6f}, Spearman {solver_summary['abundance_spearman']:.6f}.", "",
      f"The union contains {len(edge_rows)} directed positive-gain edges. {stable_edges} edges are unique top replacements in every eligible frozen processing block, with at least two eligible blocks. Stability uses 64 original 250-pixel blocks (last 87), not the 34 cached spectral folds. Reciprocal strict-stable pairs: {len(pairs)}.", "",
      f"Presence-persistent edges require numerical presence in every eligible block, with at least two eligible blocks, without requiring top replacement. Their reciprocal graph contains {len(cliques)} maximal cliques of size at least three; largest size {max(map(len,cliques),default=0)}, identity coverage {len(clique_members)/n:.6f}. This clique analysis is separate from strict unique-top stability.", "",
      f"Top five receiver identities account for {graph_summary['top5_receiver_fraction']:.6f} of eligible block unique-top events. Stable same-class fraction: {graph_summary['stable_same_class_edge_fraction']}. These distributions are descriptive; no arbitrary GO cutoff was selected.", "",
      "Necessity uses unweighted squared residual loss. Abundance redistribution is positive molecular coefficient change. Absorption columns contain raw postfit-residual and residual-change cosines; NNLS KKT can force these near zero. Removed-source versus positive-replacement signal cosine and projection are separately named, and do not imply identity correctness.", "",
      "Spectral representatives average unit-normalized alias spectra, then normalize. Reduced cosine weights each observed channel by inverse square root of full-library molecular occupancy. Fragment sharing uses only mapped physical fragment envelopes; common means shared by multiple molecular identities and specific means library-exclusive. These are library-sharing definitions, not independently verified chemical diagnostics. Original source fragment labels are retained separately.", "",
      f"Quartile-intersection descriptive anchors: strong {int(strong.sum())}, weak {int(weak.sum())}; {anchor_summary['status']}. No anchors served as labels. V58 MILD CAL R1/R2 comparisons retain one identity per case and report median, IQR, ECDF, Cliff delta, Wasserstein distance and KS statistic; no pooling independence claim or significance-driven conclusion.", "",
      f"Cross-solver edge agreement: {MISSING_EDGE}; support and spatial agreement cannot substitute for deleted-ISTA solutions. Decision: {decision['Q4_next_route']}. Q1 {decision['Q1_stable_spectral_competition']}, Q2 {decision['Q2_CE29_vs_V58_known_FP_geometry']}, Q3 {decision['Q3_competitive_decoy_assignment']}.", "",
      "The global and per-block means each have their own full-library NNLS full/deleted diagnostic. They are not pixelwise-fit coefficient averages. The existing 34 spectral-fold caches remain source-audited secondary assets and are not spatial evidence. New arrays remain outside lightweight Git; provenance binds all source files and the frozen contract.", "",
      "The audit does not identify which real molecules are present, measure real identity error or recall, establish decoy exchangeability, or authorize the next experiment. Independent review is required before these outputs are accepted."]
    (out/"analysis_record.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    return clean(dict(identity_count=n, edge_count=len(edge_rows), stable_edge_count=stable_edges,
                      solver_summary=solver_summary, decision=decision))
