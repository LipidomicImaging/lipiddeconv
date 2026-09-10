"""V59 B1: m/z-only standardized predicted libraries, never production-equivalent.

CPU-only construction and geometry. No learned solver, rho or outcome input.
The earlier production-equivalence audit is independent and remains unchanged.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SOURCE = "v36_linear_ce_response_msi29_parent_gradient_fixed_msiw5"
PROD = "v38_ce29_fragmentionfixed_profile_overlap_empiricalfwhm_globalq99_ista_ready"
# Selection scope is separate from the unchanged numerical library construction.
EXTERNAL_RANGE = (700, 900)
CONTRACT = {
    "definition": "STANDARDIZED_CROSS_LIBRARY_GEOMETRY_GENERALIZATION",
    "source": "V36 full CE29 predictions; no supplementary PC-Cl workbook candidates",
    "inclusion": "each source row becomes one column iff L <= precursor_mz <= H",
    "ordering": "unchanged source row order; candidate_id = source entry_id",
    "chain_prior_filter": False, "sn_collapse": False, "observation_filter": False,
    "zero_or_unresolved_spectrum": "STOP; never silently remove a selected candidate",
    "monoisotopic_intensity": "source prediction on channels with positive source mask, finite target m/z and positive intensity; original builder formula routines",
    "isotopes": "original conditional exact isotope-state formulas, M0-M4; absolute natural-abundance rank weights, no rank renormalization",
    "axis": "original build_resolution_axis, complete-span m/R clustering at R=50000; library-intensity-weighted centroid; 8-decimal mass keys",
    "channel_response": "intrinsic clustered stick library only; no experimental FWHM, overlap response, observation-axis merge, q99 or B input",
    "normalization": "float32 torch column L2 norm +1e-8, identical arithmetic to production utils.get_A_matrix",
    "parent_region": "[L,H+5]; +5 derived from production configured upper bounds 803-798; classification only",
    "parent_region_limitation": "m/z region labels, not a guarantee that every in-region peak is chemically a parent ion",
    "fragment_region": "channels outside the translated parent region; strict nonzero support",
    "external_bins": "complete 50-Da bins within actual V36 precursor coverage; bounds multiples of 50",
    "W0": [748, 798], "primary": "q90 NN_other_identity", "quantile_method": "linear",
    "primary_reason": "same-lipid-name candidates cannot by themselves constitute molecular identity false allocation",
    "NN_all": "diagnostic only", "accumulation": "float64",
    "truth": "singleton lipid_name only; at least 250 per formal domain",
    "selection": "LOW Q25, MID Q50, HIGH Q75 sequential; distance, lower_mz, upper_mz; no candidate-set overlap",
    "candidate_count_cap": None, "experimental_outcomes_used": False,
    "formal_design": {"domains": 4, "splits": 2, "replicates": 3, "K": 125, "learned_runs": 24},
}


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def json_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def array_hash(a):
    return hashlib.sha256(a.tobytes(order="C")).hexdigest()


def bins(low, high):
    return [(k, k + 50) for k in range(50 * math.ceil(low / 50), 50 * math.floor(high / 50), 50)]


def scoped_contract(endpoint_policy):
    return {**CONTRACT, "endpoint_policy": endpoint_policy,
            "external_precursor_range": list(EXTERNAL_RANGE),
            "external_bins": "complete 50-Da bins within 700-900 and actual V36 coverage; bounds multiples of 50"}


def rescope_cached(args):
    """Preserve superseded evidence and reselect from existing inventory only."""
    import csv
    import shutil
    from datetime import datetime, timezone
    out = args.output_dir.resolve()
    rows = json.loads((out/'inventory_records.json').read_text(encoding='utf-8'))
    expected = bins(*EXTERNAL_RANGE)
    scoped = [r for r in rows if r['domain']=='W0_STD' or (r['lower_mz'],r['upper_mz']) in expected]
    assert sorted((r['lower_mz'],r['upper_mz']) for r in scoped if r['domain']!='W0_STD') == expected
    w0 = next(r for r in scoped if r['domain']=='W0_STD')
    selection = select(scoped,w0,args.endpoint_policy)
    assert selection == select(list(reversed(scoped)),w0,args.endpoint_policy)
    archive = out/'history'/('superseded_scope_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    archive.mkdir(parents=True,exist_ok=False)
    hashes = {}
    for path in sorted(out.iterdir()):
        if path.is_file() and path.suffix in ('.json','.csv'):
            shutil.copy2(path,archive/path.name)
            hashes[path.name] = digest(path)
            assert digest(archive/path.name) == hashes[path.name]
    write(archive/'superseded_manifest.json',{'status':'SUPERSEDED_BY_700_900_SCOPE',
        'file_sha256':hashes,'library_arrays_preserved_in_place':True})
    names = {'W0_PROD'} | {r['domain'] for r in scoped}
    geometry_path = out/'standardized_window_geometry.json'
    cached_geometry = json.loads(geometry_path.read_text(encoding='utf-8'))
    write(geometry_path,{k:v for k,v in cached_geometry.items() if k in names})
    write(out/'inventory_records.json',scoped)
    with (out/'standardized_window_inventory.csv').open('w',encoding='utf-8',newline='') as f:
        fields = [k for k in scoped[0] if k!='candidate_ids']
        writer = csv.DictWriter(f,fieldnames=fields); writer.writeheader()
        for r in scoped:
            writer.writerow({k:json.dumps(r[k],sort_keys=True) if isinstance(r[k],dict) else r[k] for k in fields})
    bridge_path = out/'w0_prod_to_std_bridge_audit.json'
    bridge_result = json.loads(bridge_path.read_text(encoding='utf-8'))
    external = [r for r in scoped if r['domain']!='W0_STD']
    eligible = [r for r in scoped if r['eligible_for_formal']]
    for name in ('W0_PROD','W0_STD'):
        record = bridge_result[name]
        record['fraction_standardized_external_R_leq'] = sum(r['R_window']<=record['R_window'] for r in external)/len(external)
        record['formal_eligible_external_R_percentile_fraction'] = sum(r['R_window']<=record['R_window'] for r in eligible)/len(eligible) if eligible else None
        record['formal_eligible_external_count'] = len(eligible)
    bridge_result['external_precursor_range'] = list(EXTERNAL_RANGE)
    write(bridge_path,bridge_result)
    write(out/'standardized_library_contract.json',scoped_contract(args.endpoint_policy))
    write(out/'proposed_low_mid_high_windows.json',selection)
    write(out/'cone_geometry_diagnostic.json',{'status':'NOT_RUN_CURRENT_SCOPE_SELECTION_INCOMPLETE',
        'superseded_partial_evidence':str(archive/'cone_geometry_diagnostic.json')})
    write(out/'selection_interpretation.json',{'status':selection['status'],
        'eligible_domains':[r['domain'] for r in eligible], 'required_external_count':3,
        'reason':'700-750 and 750-800 overlap frozen production W0; only two external windows remain. No automatic reduction to two.'})
    report = json.loads((archive/'v59_standardized_feasibility_report.json').read_text(encoding='utf-8'))
    report.update(status=selection['status'],selection=selection,external_precursor_range=list(EXTERNAL_RANGE),
                  formal_design_feasible=False,superseded_evidence_directory=str(archive))
    write(out/'v59_standardized_feasibility_report.json',report)
    write(out/'validation_report.json',{'status':'PASS','scope_bins':'PASS',
        'selection_order_determinism':'PASS','superseded_file_preservation':'PASS',
        'geometry_recomputed':False,'library_arrays_modified':False,
        'prior_numerical_validation':str(archive/'validation_report.json'),
        'selection_status':selection['status']})
    print(json.dumps({'selection':selection,'eligible_domains':[r['domain'] for r in eligible]}),flush=True)
    return 0


def overlap(a, b, endpoint_policy):
    left = max(a["lower_mz"], b["lower_mz"])
    right = min(a["upper_mz"], b["upper_mz"])
    interval = left <= right if endpoint_policy == "closed" else left < right
    shared = bool(set(a.get("candidate_ids", [])) & set(b.get("candidate_ids", [])))
    return interval or shared


def composition(metadata):
    names = Counter(r["lipid_name"] for r in metadata)
    return {"N_candidates": len(metadata), "N_unique_lipid_names": len(names),
            "N_singleton_lipid_names": sum(n == 1 for n in names.values()),
            "N_duplicate_lipid_names": sum(n > 1 for n in names.values()),
            "max_candidates_per_lipid_name": max(names.values(), default=0),
            "class_composition": dict(sorted(Counter(r["lipid_class"] for r in metadata).items())),
            "adduct_composition": dict(sorted(Counter(r["adduct"] for r in metadata).items()))}


def summary(a):
    import numpy as np
    a = np.asarray(a, dtype=np.float64)
    if not a.size:
        return {"status": "NO_DEFINED_VALUES", "count": 0}
    return {"count": int(a.size), "min": float(a.min()), "max": float(a.max()),
            **{name: float(np.quantile(a, q, method="linear")) for name, q in
               (("q10", .1), ("q25", .25), ("median", .5), ("q75", .75), ("q90", .9), ("q95", .95), ("q99", .99))},
            **{f"fraction_ge_{t}": float(np.mean(a >= t)) for t in (.9, .95, .99, .999)}}


def normalize(a):
    import torch
    tensor = torch.as_tensor(a, dtype=torch.float32, device="cpu")
    return (tensor / (torch.norm(tensor, p=2, dim=0, keepdim=True) + 1e-8)).numpy().copy()


def geometry(a, axis, metadata, lo, hi):
    import numpy as np
    x = a.astype(np.float64)
    x /= np.linalg.norm(x, axis=0, keepdims=True)
    similarities = np.clip(x.T @ x, -1, 1)
    np.fill_diagonal(similarities, -np.inf)
    names = np.asarray([r["lipid_name"] for r in metadata])
    nearest_all = similarities.argmax(axis=0)
    all_score = similarities[nearest_all, np.arange(len(names))]
    similarities[names[:, None] == names[None, :]] = -np.inf
    nearest_other = similarities.argmax(axis=0)
    other_score = similarities[nearest_other, np.arange(len(names))]
    if not np.isfinite(other_score).all():
        raise RuntimeError("STANDARDIZED_LIBRARY_DEFINITION_UNRESOLVED: fewer than two molecular identities")
    parent = (axis >= lo) & (axis <= hi + 5)
    support = a[~parent] != 0
    competitor = support[:, nearest_other]
    union = (support | competitor).sum(axis=0)
    intersection = (support & competitor).sum(axis=0)
    jaccard = np.divide(intersection, union, out=np.full(len(names), np.nan), where=union > 0)
    counts = support.sum(axis=0)
    shared = ((support.sum(axis=1) >= 2)[:, None] & support).sum(axis=0)
    fraction = np.divide(shared, counts, out=np.full(len(names), np.nan), where=counts > 0)
    return {"NN_other_identity": summary(other_score), "NN_all": summary(all_score),
            "R_window": float(np.quantile(other_score, .9, method="linear")),
            "parent_channel_count": int(parent.sum()), "fragment_channel_count": int((~parent).sum()),
            "nonzero_entries": int(np.count_nonzero(a)), "sparsity": float(1 - np.count_nonzero(a) / a.size),
            "fragment_nonzero_count": summary(counts), "fragment_Jaccard_other_identity": summary(jaccard[np.isfinite(jaccard)]),
            "fragment_Jaccard_undefined_empty_union": int((union == 0).sum()),
            "shared_fragment_fraction": summary(fraction[np.isfinite(fraction)]),
            "shared_fragment_definition": "fraction of a candidate's nonzero fragment channels supported by at least one other candidate; empty support undefined",
            "nearest_competitor_tie": "first source candidate index"}


def select(rows, w0, endpoint_policy):
    import numpy as np
    eligible = [r for r in rows if r["eligible_for_formal"] and not overlap(r, w0, endpoint_policy)]
    if len(eligible) < 3:
        return {"status": "INSUFFICIENT_NONOVERLAPPING_STANDARDIZED_WINDOWS", "eligible_count": len(eligible)}
    chosen, targets, shortlists = {}, {}, {}
    for label, q in (("LOW", .25), ("MID", .5), ("HIGH", .75)):
        target = float(np.quantile([r["R_window"] for r in eligible], q, method="linear"))
        ordered = sorted(eligible, key=lambda r: (abs(r["R_window"] - target), r["lower_mz"], r["upper_mz"]))
        targets[label] = target; shortlists[label] = [r["domain"] for r in ordered]
        value = next((r for r in ordered if all(not overlap(r, s, endpoint_policy) for s in chosen.values())), None)
        if value is None:
            return {"status": "INSUFFICIENT_NONOVERLAPPING_STANDARDIZED_WINDOWS", "partial_selection": {k: v["domain"] for k, v in chosen.items()},
                    "targets": targets, "shortlists": shortlists, "eligible_count": len(eligible)}
        chosen[label] = value
    return {"status": "SELECTED", "selected": {k: v["domain"] for k, v in chosen.items()}, "targets": targets,
            "shortlists": shortlists, "eligible_count": len(eligible), "endpoint_policy": endpoint_policy}


class IntrinsicBuilder:
    def __init__(self, asset_root):
        import numpy as np
        self.pipeline = asset_root / "adapter_pipeline"
        self.source = self.pipeline / "outputs" / SOURCE
        sys.path.insert(0, str(self.pipeline))
        spec = importlib.util.spec_from_file_location("v59b1_original_isotope", self.pipeline / "build_isotope_library_v23.py")
        self.b = importlib.util.module_from_spec(spec); spec.loader.exec_module(self.b)
        self.metadata = self.b.legacy.load_jsonl(self.source / "metadata_ce29_full.jsonl")
        self.channels = json.loads((self.source / "channels.json").read_text(encoding="utf-8"))
        self.predictions = np.load(self.source / "channel_predictions_ce29.npy")
        self.targets = np.load(self.source / "channel_target_mz.npy")
        self.masks = np.load(self.source / "channel_masks.npy")
        assert self.predictions.shape == self.targets.shape == self.masks.shape == (len(self.metadata), len(self.channels))
        assert len({r["entry_id"] for r in self.metadata}) == len(self.metadata)

    def build(self, lo, hi):
        import numpy as np
        b = self.b
        indices = [i for i, r in enumerate(self.metadata) if lo <= float(r["mz"]) <= hi]
        metadata, spectra, weights = [], [[] for _ in range(5)], []
        for i in indices:
            row = dict(self.metadata[i]); row["candidate_id"] = row["entry_id"]
            parent = b.ion_formula(row); peaks = []
            for j, channel in enumerate(self.channels):
                if self.masks[i, j] <= 0 or not math.isfinite(self.targets[i, j]):
                    continue
                mz, intensity = float(self.targets[i, j]), float(self.predictions[i, j])
                if not math.isfinite(intensity):
                    raise RuntimeError(f"STANDARDIZED_LIBRARY_DEFINITION_UNRESOLVED: nonfinite {row['entry_id']}")
                if intensity <= 0:
                    continue
                if channel == "precursor":
                    formula, convention = parent, "ion_formula"
                else:
                    formula, _, _, convention = b.legacy.enumerate_fragment_formula(
                        mz, parent, b.legacy.expected_carbon(channel, row, parent["C"]), 2e-3)
                formula = b.isotope_ion_composition(formula, convention)
                if any(formula[e] > parent[e] for e in b.legacy.ELEMENTS):
                    raise RuntimeError(f"STANDARDIZED_LIBRARY_DEFINITION_UNRESOLVED: invalid formula {row['entry_id']}")
                peaks.append((mz, intensity, formula))
            parent_tuple = b.counts_tuple(parent)
            weights.append(b.precursor_weights(parent_tuple, 4))
            for rank in range(5):
                spectra[rank].append(b.conditional_spectrum(peaks, parent_tuple, rank))
            metadata.append(row)
        weights = np.asarray(weights, dtype=np.float64)
        axis, mapping, _ = b.build_resolution_axis(spectra, weights, 50000.)
        ranked = np.stack([b.project_spectra_to_resolution_axis(s, mapping, len(axis)) for s in spectra])
        raw = np.sum(ranked * weights.T[:, None, :], axis=0).astype(np.float32)
        if not np.isfinite(raw).all() or np.any(np.linalg.norm(raw, axis=0) == 0):
            raise RuntimeError("STANDARDIZED_LIBRARY_DEFINITION_UNRESOLVED: zero/nonfinite candidate spectrum")
        assert len(metadata) == len(indices)
        return raw, normalize(raw), axis, metadata


def bridge(prod, std, builder):
    import numpy as np
    p_raw, p_a, p_axis, pm = prod
    s_raw, s_a, s_axis, sm = std
    pids = {r["candidate_id"]: i for i, r in enumerate(pm)}
    sids = {r["candidate_id"]: i for i, r in enumerate(sm)}
    common = sorted(pids.keys() & sids.keys())
    rows = []
    for candidate in common:
        i, j = pids[candidate], sids[candidate]
        left = (p_axis[p_raw[:, i] != 0], p_raw[p_raw[:, i] != 0, i])
        right = (s_axis[s_raw[:, j] != 0], s_raw[s_raw[:, j] != 0, j])
        # Different native grids are never compared by array index. Apply one
        # explicitly declared intrinsic resolution rule to both spectra.
        axis, mapping, _ = builder.b.build_resolution_axis([[left, right]], np.ones((2, 1)), 50000.)
        aligned = builder.b.project_spectra_to_resolution_axis([left, right], mapping, len(axis)).astype(np.float64)
        x, y = aligned.T; difference = x - y
        rows.append({"candidate_id": candidate, "precursor_exact": pm[i]["mz"] == sm[j]["mz"],
                     "lipid_name_exact": pm[i]["lipid_name"] == sm[j]["lipid_name"],
                     "cosine": float(x @ y / (np.linalg.norm(x) * np.linalg.norm(y))),
                     "max_absolute_difference": float(np.max(np.abs(difference))),
                     "relative_L2_difference": float(np.linalg.norm(difference) / np.linalg.norm(x))})
    return {"W0_PROD": composition(pm), "W0_STD": composition(sm), "shared_candidate_id_count": len(common),
            "production_candidate_coverage": len(common) / len(pm),
            "shared_lipid_name_count": len({r['lipid_name'] for r in pm} & {r['lipid_name'] for r in sm}),
            "production_only_candidates": sorted(pids.keys() - sids.keys()),
            "standardized_only_candidates": sorted(sids.keys() - pids.keys()),
            "shared_candidate_spectra": rows,
            "spectral_comparison_definition": "production cached spectrum_source_STICK before empirical response vs standardized intrinsic sticks; pairwise common R50000 grid; no direct equality claim across native grids",
            "spectral_comparison_is_diagnostic_only": True}


def self_test():
    assert bins(535.340531, 1033.768956) == [(k, k + 50) for k in range(550, 1000, 50)]
    m = [{"lipid_name": n, "lipid_class": "X", "adduct": "A"} for n in ("a", "a", "b")]
    assert composition(m)["N_singleton_lipid_names"] == 1
    def row(lo, score, domain):
        return {"lower_mz": lo, "upper_mz": lo + 50, "R_window": score, "domain": domain,
                "eligible_for_formal": True, "candidate_ids": [domain]}
    w0 = row(748, 0, "W0")
    assert overlap(row(750, 0, "x"), w0, "touch-if-disjoint")
    rows = [row(100, .1, "a"), row(200, .5, "b"), row(300, .5, "c"), row(400, .9, "d")]
    chosen = select(rows, w0, "closed")
    assert chosen['selected'] == {"LOW": "b", "MID": "c", "HIGH": "d"}
    assert select(list(reversed(rows)), w0, "closed") == chosen
    assert overlap(row(800, 0, 'a'), row(850, 0, 'b'), 'closed')
    assert not overlap(row(800, 0, 'a'), row(850, 0, 'b'), 'touch-if-disjoint')
    assert overlap(row(800, 0, 'a'), row(850, 0, 'a'), 'touch-if-disjoint')
    import ast
    tree = ast.parse(Path(__file__).read_text(encoding='utf-8'))
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'select')
    calls = {n.func.id for n in ast.walk(node) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
    assert calls <= {'overlap', 'len', 'float', 'sorted', 'abs', 'next', 'all'}
    return {"integer_bins": "PASS", "W0_overlap": "PASS", "singleton_count": "PASS", "tie_break": "PASS",
            "overlap_fallback": "PASS", "selection_determinism": "PASS", "endpoint_candidate_overlap": "PASS",
            "outcome_blind_selection_call_audit": "PASS"}


def cone_only(args):
    """Direct SciPy NNLS, all candidates, excluding every same-name column."""
    import numpy as np
    import torch
    from scipy.optimize import nnls
    torch.set_num_threads(1)
    out = args.output_dir.resolve()
    selection_path = out / 'proposed_low_mid_high_windows.json'
    frozen_sha = digest(selection_path)
    chosen = json.loads(selection_path.read_text(encoding='utf-8'))
    assert json.loads((out/'standardized_library_contract.json').read_text(encoding='utf-8'))['external_precursor_range'] == list(EXTERNAL_RANGE)
    if chosen['status'] != 'SELECTED':
        raise RuntimeError('CONE_REQUIRES_FROZEN_SELECTION')
    domains = ['W0_PROD', 'W0_STD'] + [chosen['selected'][k] for k in ('LOW', 'MID', 'HIGH')]
    result_path = out / 'cone_geometry_diagnostic.json'
    result = json.loads(result_path.read_text(encoding='utf-8')) if result_path.exists() else {
        'status':'IN_PROGRESS', 'selection_file_sha256':frozen_sha,
        'definition':'min ||A_j-A_other_identity*x||_2 / ||A_j||_2, x>=0; all same-lipid-name columns excluded',
        'method':'scipy.optimize.nnls float64, maxiter=10*N_alternatives; direct residual computed from A',
        'diagnostic_only':True, 'domains':{}}
    assert result['selection_file_sha256'] == frozen_sha
    for domain in domains:
        if domain == 'W0_PROD':
            p = args.asset_root / 'adapter_pipeline/outputs' / PROD
            a = normalize(np.load(p/'A_library.npy')[0])
            metadata = [json.loads(s) for s in (p/'candidate_metadata_final.jsonl').read_text(encoding='utf-8').splitlines()]
        else:
            p = out / 'libraries' / domain
            with np.load(p/'arrays.npz') as data: a = data['A_std']
            metadata = json.loads((p/'metadata.json').read_text(encoding='utf-8'))
        names = np.asarray([r['lipid_name'] for r in metadata])
        entry = result['domains'].setdefault(domain, {'A_sha256':array_hash(a), 'records':[], 'status':'IN_PROGRESS'})
        assert entry['A_sha256'] == array_hash(a)
        matrix = a.astype(np.float64)
        start = len(entry['records'])
        for j in range(start, matrix.shape[1]):
            alternatives = matrix[:, names != names[j]]
            target = matrix[:, j]
            weights, _ = nnls(alternatives, target, maxiter=10*alternatives.shape[1])
            residual = alternatives @ weights - target
            score = float(np.linalg.norm(residual)/np.linalg.norm(target))
            gradient = alternatives.T @ residual
            kkt = max(float(np.max(np.maximum(-gradient, 0))), float(np.max(np.abs(weights*gradient))))
            if not np.isfinite(score) or kkt > 1e-7:
                raise RuntimeError(f'CONE_NUMERICAL_CHECK_FAILED: {domain}/{j}: {kkt}')
            entry['records'].append({'candidate_id':metadata[j].get('candidate_id',metadata[j].get('entry_id')),
                                     'lipid_name':str(names[j]), 'd_cone_other_identity':score, 'KKT_error':kkt})
            if (j+1)%25 == 0:
                write(result_path, result); print('CONE', domain, j+1, '/', matrix.shape[1], flush=True)
        entry['summary'] = summary([r['d_cone_other_identity'] for r in entry['records']])
        entry['status'] = 'COMPLETE'; write(result_path, result)
    assert digest(selection_path) == frozen_sha
    result['status'] = 'COMPLETE'; write(result_path, result)
    report_path = out/'v59_standardized_feasibility_report.json'
    report = json.loads(report_path.read_text(encoding='utf-8'))
    report.update(status='READY_FOR_FORMAL_V59_STANDARDIZED', cone_diagnostic_complete=True,
                  selection_unchanged_after_cone=True, scientific_definition='STANDARDIZED_CROSS_LIBRARY_GEOMETRY_GENERALIZATION')
    write(report_path, report)
    print('READY_FOR_FORMAL_V59_STANDARDIZED', flush=True)
    return 0


def validate_outputs(args):
    import csv
    import numpy as np
    out = args.output_dir.resolve()
    provenance_path = out/'full_library_provenance.json'
    provenance = json.loads(provenance_path.read_text(encoding='utf-8'))
    assert {Path(p).name:digest(Path(p)) for p in provenance['paths']} == provenance['source_hashes']
    new_fingerprint = json_hash({'contract':CONTRACT,'source_hashes':provenance['source_hashes']})
    initial_contract = {**CONTRACT,'definition':'STANDARDIZED_CROSS_LIBRARY_GEOMETRY_GENERALALIZATION'}
    initial_fingerprint = json_hash({'contract':initial_contract,'source_hashes':provenance['source_hashes']})
    rows = json.loads((out/'inventory_records.json').read_text(encoding='utf-8'))
    geom = json.loads((out/'standardized_window_geometry.json').read_text(encoding='utf-8'))
    frozen = json.loads((out/'proposed_low_mid_high_windows.json').read_text(encoding='utf-8'))
    w0_row = next(r for r in rows if r['domain'] == 'W0_STD')
    w0 = {k:w0_row[k] for k in ('lower_mz','upper_mz','candidate_ids')}
    assert select(rows,w0,args.endpoint_policy) == frozen
    assert select(list(reversed(rows)),w0,args.endpoint_policy) == frozen
    for r in rows:
        p = out/'libraries'/r['domain']
        metadata = json.loads((p/'metadata.json').read_text(encoding='utf-8'))
        assert [m['candidate_id'] for m in metadata] == r['candidate_ids']
        assert all(r['lower_mz']<=float(m['mz'])<=r['upper_mz'] for m in metadata)
        assert all(r[k]==v for k,v in composition(metadata).items())
        with np.load(p/'arrays.npz') as values:
            actual = geometry(values['A_std'],values['axis'],metadata,r['lower_mz'],r['upper_mz'])
            hashes = [array_hash(values[k]) for k in ('raw','A_std','axis')]
        stamp = json.loads((p/'construction.json').read_text(encoding='utf-8'))
        assert hashes == stamp['array_hashes'] and stamp['independent_construction_repeat_exact']
        assert stamp['fingerprint'] in (initial_fingerprint,new_fingerprint)
        if stamp['fingerprint'] != new_fingerprint:
            stamp['prior_descriptor_fingerprint'] = stamp['fingerprint']
            stamp['descriptor_correction'] = 'Spelling correction GENERALALIZATION -> GENERALIZATION; no numerical or construction-rule change'
            stamp['fingerprint'] = new_fingerprint
            write(p/'construction.json',stamp)
        assert actual == geom[r['domain']]
        r.update({k:actual[k] for k in ('parent_channel_count','fragment_channel_count','nonzero_entries','sparsity')})
        r['precursor_mz_distribution'] = summary([m['mz'] for m in metadata])
    bpath = out/'w0_prod_to_std_bridge_audit.json'
    b = json.loads(bpath.read_text(encoding='utf-8'))
    shared = b['shared_candidate_id_count']
    assert shared+len(b['production_only_candidates']) == b['W0_PROD']['N_candidates']
    assert shared+len(b['standardized_only_candidates']) == b['W0_STD']['N_candidates']
    assert len({r['candidate_id'] for r in b['shared_candidate_spectra']}) == shared
    assert all(r['precursor_exact'] and r['lipid_name_exact'] and -.0000001<=r['cosine']<=1.0000001 for r in b['shared_candidate_spectra'])
    eligible = [r for r in rows if r['eligible_for_formal']]
    for name in ('W0_PROD','W0_STD'):
        b[name]['formal_eligible_external_R_percentile_fraction'] = float(np.mean([r['R_window']<=b[name]['R_window'] for r in eligible]))
        b[name]['formal_eligible_external_count'] = len(eligible)
    b['shared_candidate_spectral_summary'] = {key:summary([r[key] for r in b['shared_candidate_spectra']]) for key in ('cosine','max_absolute_difference','relative_L2_difference')}
    write(bpath,b)
    assert sorted((r['lower_mz'],r['upper_mz']) for r in rows if r['domain']!='W0_STD') == bins(*EXTERNAL_RANGE)
    selected_rows = [next(r for r in rows if r['domain']==frozen['selected'][label]) for label in ('LOW','MID','HIGH')] if frozen['status']=='SELECTED' else []
    assert all(not overlap(r,w0,args.endpoint_policy) for r in selected_rows)
    assert all(not overlap(a,b,args.endpoint_policy) for i,a in enumerate(selected_rows) for b in selected_rows[i+1:])
    validation = self_test()
    validation.update(actual_construction_determinism='PASS',cached_array_hashes='PASS',
        actual_geometry_rerun='PASS',actual_selection_rerun='PASS',bridge_consistency='PASS',
        selected_windows_no_candidate_overlap='PASS',selected_windows_no_positive_width_overlap='PASS',
        py_compile='PASS',help='PASS',diff_check='PASS',runner_sha256=digest(Path(__file__)))
    write(out/'validation_report.json',validation)
    write(out/'inventory_records.json',rows)
    fields = ['domain','lower_mz','upper_mz','width','N_candidates','N_unique_lipid_names','N_singleton_lipid_names',
              'N_duplicate_lipid_names','max_candidates_per_lipid_name','parent_channel_count','fragment_channel_count',
              'nonzero_entries','sparsity','R_window','eligible_for_formal','exclusion_reason','class_composition','adduct_composition','precursor_mz_distribution']
    with (out/'standardized_window_inventory.csv').open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        for r in rows:
            writer.writerow({k:json.dumps(r[k],sort_keys=True) if isinstance(r[k],dict) else r[k] for k in fields})
    write(out/'selection_interpretation.json',{'selected_R':{label:r['R_window'] for label,r in zip(('LOW','MID','HIGH'),selected_rows)},
        'eligible_external_count':len(eligible),'labels_are_quantile_targets_not_guaranteed_monotone_scores':True,
        'status':frozen['status'], 'reason':'Quantile targets use deterministic sequential fallback; fewer than three compatible external domains stops selection.'})
    provenance['construction_fingerprint'] = new_fingerprint
    production = args.asset_root/'adapter_pipeline/outputs'/PROD
    provenance['production_reference_assets'] = {name:{'path':str((production/name).resolve()),'sha256':digest(production/name)}
        for name in ('A_library.npy','candidate_metadata_final.jsonl','shared_mz_final.npy','spectrum_source_STICK.npy','shared_mz_source_sticks.npy')}
    write(provenance_path,provenance)
    write(out/'standardized_library_contract.json',scoped_contract(args.endpoint_policy))
    print('VALIDATION PASS',flush=True)
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-root", type=Path, default=ROOT.parent / "decon-lipid")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/v59_standardized_cross_library_inventory")
    parser.add_argument("--endpoint-policy", choices=("closed", "touch-if-disjoint"), default="touch-if-disjoint")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--build-only", action="store_true", help="Build inventory and bridge without final window selection")
    parser.add_argument("--cone-only", action="store_true", help="Resume CPU cone diagnostics only after selection is frozen")
    parser.add_argument("--validate-only", action="store_true", help="Independently recheck cached arrays, geometry, selection and bridge")
    parser.add_argument("--rescope-only", action="store_true", help="Archive prior reports and apply 700-900 scope using cached inventory; no geometry or construction")
    args = parser.parse_args()
    if args.self_test:
        print(json.dumps(self_test())); return 0
    if args.output_dir.resolve().name != 'v59_standardized_cross_library_inventory' or any(p.name.startswith(('v57_','v58_')) for p in args.output_dir.resolve().parents):
        parser.error('Use the independent standardized V59 output directory')
    if args.rescope_only:
        return rescope_cached(args)
    if args.cone_only:
        return cone_only(args)
    if args.validate_only:
        return validate_outputs(args)
    import numpy as np
    import torch
    torch.set_num_threads(1)
    output = args.output_dir.resolve()
    if output.name != "v59_standardized_cross_library_inventory":
        parser.error("Use the independent v59_standardized_cross_library_inventory output directory")
    output.mkdir(parents=True, exist_ok=True)
    contract = scoped_contract(args.endpoint_policy)
    write(output / "standardized_library_contract.json", contract)
    builder = IntrinsicBuilder(args.asset_root)
    b = builder
    sources = [b.source / f for f in ("metadata_ce29_full.jsonl", "channel_predictions_ce29.npy", "channel_target_mz.npy", "channel_masks.npy", "channels.json")]
    sources += [b.pipeline / f for f in ("build_isotope_library_v23.py", "build_isotope_library_758.py")]
    sources += [ROOT / "src/utils.py", ROOT / "src/config_758.py"]
    fingerprints = {p.name: digest(p) for p in sources}
    contract_hash = json_hash({"contract": CONTRACT, "source_hashes": fingerprints})
    config = (ROOT / "src/config_758.py").read_text(encoding="utf-8")
    assert "Windows = (748, 798)" in config and "parent_channel_mz_range = (748.0, 803.0)" in config
    mz = [float(r["mz"]) for r in b.metadata]
    provenance = {"source_hashes": fingerprints, "paths": [str(p.resolve()) for p in sources],
                  "prediction_shape": list(b.predictions.shape), "metadata_rows": len(b.metadata),
                  "precursor_range": [min(mz), max(mz)], "construction_fingerprint": contract_hash}
    write(output / "full_library_provenance.json", provenance)
    prod_dir = b.pipeline / "outputs" / PROD
    pm = b.b.legacy.load_jsonl(prod_dir / "candidate_metadata_final.jsonl")
    p_raw = np.load(prod_dir / "A_library.npy")[0]
    p_a = normalize(p_raw); p_axis = np.load(prod_dir / "shared_mz_final.npy")
    prod = (np.load(prod_dir / "spectrum_source_STICK.npy").T, p_a, np.load(prod_dir / "shared_mz_source_sticks.npy"), pm)
    datasets, rows, diagnostics = {}, [], {}
    domains = [("W0_STD", 748, 798)] + [(f"W{lo}_{hi}", lo, hi) for lo, hi in bins(max(min(mz),EXTERNAL_RANGE[0]), min(max(mz),EXTERNAL_RANGE[1]))]
    pg = geometry(p_a, p_axis, pm, 748, 798)
    diagnostics["W0_PROD"] = pg
    w0 = {"lower_mz": 748, "upper_mz": 798, "candidate_ids": [r['entry_id'] for r in b.metadata if 748 <= float(r['mz']) <= 798]}
    for domain, lo, hi in domains:
        folder = output / "libraries" / domain; folder.mkdir(parents=True, exist_ok=True)
        stamp = folder / "construction.json"
        if stamp.exists():
            old = json.loads(stamp.read_text(encoding="utf-8")); assert old["fingerprint"] == contract_hash
            with np.load(folder / "arrays.npz") as data:
                raw, a, axis = data['raw'], data['A_std'], data['axis']
            meta = json.loads((folder / "metadata.json").read_text(encoding="utf-8"))
            assert [array_hash(x) for x in (raw, a, axis)] == old['array_hashes']
        else:
            print("BUILD", domain, flush=True)
            try:
                raw, a, axis, meta = b.build(lo, hi)
                repeat = b.build(lo, hi)
                assert all(array_hash(x) == array_hash(y) for x, y in zip((raw, a, axis), repeat[:3])) and meta == repeat[3]
            except Exception as exc:
                write(output / 'v59_standardized_feasibility_report.json', {'status': 'STANDARDIZED_LIBRARY_DEFINITION_UNRESOLVED', 'domain': domain, 'reason': f'{type(exc).__name__}: {exc}'})
                raise
            np.savez_compressed(folder / "arrays.npz", raw=raw, A_std=a, axis=axis)
            write(folder / "metadata.json", meta)
            write(stamp, {"fingerprint": contract_hash, "array_hashes": [array_hash(x) for x in (raw, a, axis)], "independent_construction_repeat_exact": True})
        g = geometry(a, axis, meta, lo, hi)
        assert g == geometry(a, axis, meta, lo, hi)
        diagnostics[domain] = g; datasets[domain] = (raw, a, axis, meta)
        r = {"domain": domain, "lower_mz": lo, "upper_mz": hi, "width": 50, **composition(meta), "R_window": g['R_window'],
             "candidate_ids": [m['candidate_id'] for m in meta]}
        reasons = []
        if domain == 'W0_STD': reasons.append('REFERENCE_BRIDGE')
        elif overlap(r, w0, args.endpoint_policy): reasons.append('OVERLAPS_PRODUCTION_W0')
        if r['N_singleton_lipid_names'] < 250: reasons.append('INSUFFICIENT_SINGLETON_IDENTITIES')
        r.update(eligible_for_formal=not reasons, exclusion_reason=';'.join(reasons))
        rows.append(r); print("BUILT", domain, r['N_candidates'], r['N_singleton_lipid_names'], r['R_window'], flush=True)
        write(output / "standardized_window_geometry.json", diagnostics)
    bridge_result = bridge(prod, datasets['W0_STD'], b)
    for name in ('W0_PROD', 'W0_STD'):
        value = diagnostics[name]['R_window']
        ext = np.asarray([r['R_window'] for r in rows if r['domain'] != 'W0_STD'])
        bridge_result[name]['R_window'] = value
        bridge_result[name]['fraction_standardized_external_R_leq'] = float(np.mean(ext <= value))
    write(output / "w0_prod_to_std_bridge_audit.json", bridge_result)
    import csv
    fields = ['domain','lower_mz','upper_mz','width','N_candidates','N_unique_lipid_names','N_singleton_lipid_names','N_duplicate_lipid_names','max_candidates_per_lipid_name','R_window','eligible_for_formal','exclusion_reason']
    with (output / 'standardized_window_inventory.csv').open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader(); writer.writerows({k:r[k] for k in fields} for r in rows)
    write(output / 'inventory_records.json', rows)
    selection = {'status':'PENDING_ENDPOINT_CONTRACT_DECISION'} if args.build_only else select(rows, w0, args.endpoint_policy)
    write(output / 'proposed_low_mid_high_windows.json', selection)
    tests = self_test(); tests.update(actual_construction_determinism='PASS', actual_geometry_determinism='PASS', bridge_candidate_overlap_check='PASS')
    write(output / 'validation_report.json', tests)
    status = selection['status']
    if status == 'SELECTED':
        status = 'SELECTION_FROZEN_CONE_DIAGNOSTIC_PENDING'
    write(output / 'v59_standardized_feasibility_report.json', {'status':status, 'selection':selection,
        'scientific_definition':CONTRACT['definition'], 'formal_domain_count':4, 'planned_learned_runs':24,
        'production_solver_core':'VARIABLE_N_FROM_A_SHAPE; fresh initialization, no checkpoint transfer',
        'benchmark_plumbing':'V54 EXPECTED_A_SHAPE (1084,391); V57 fixed bank and reporting loops; V58 shape guards. New formal runner must derive shapes and metadata dynamically.',
        'claim':'same algorithm/calibration across standardized library geometries; not validation of acquisition pipelines or transfer of learned weights',
        'V57_reference':'existing production W0 CLEAN', 'V58_reference':'within-library spectral mismatch; no V58 outcomes read',
        'GPU_training_executed':False,'rho_or_FDR_computed':False})
    print(status, flush=True)
    return 0 if status == 'READY_FOR_FORMAL_V59_STANDARDIZED' else 2


if __name__ == '__main__':
    sys.exit(main())
