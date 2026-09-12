"""Outcome-free donor grouping and finite physical endpoint construction.

These helpers do not read outcomes, fit a model, resample an infeasible split,
or alter abundance. The caller seals their returned bindings before training.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json


ROLES = ("MODEL", "CAL", "EVAL")
ROLE_SLOTS = ("MODEL", "MODEL", "MODEL", "CAL", "EVAL")
DONOR_NAMESPACE = "physical_contract_v1_donors"
TARGET_NAMESPACE = "physical_contract_v1_target"


def _require(condition, message):
    if not condition:
        raise RuntimeError(message)


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False)


def _fingerprint(value):
    return hashlib.sha256(_canonical(value).encode()).hexdigest()


def build_donor_split(patterns, identity_records):
    """Partition all-record connected donor groups; return a sealed-ready dict.

    Edges include every CE record in ``identity_records``, including identities
    and records without admitted endpoints. Only components containing admitted
    patterns receive a role, so irrelevant isolated records cannot move slots.
    """
    _require(patterns and identity_records, "EMPTY_DONOR_INPUT")
    ids = [p["pattern_id"] for p in patterns]
    _require(len(ids) == len(set(ids)), "DUPLICATE_PATTERN_ID")
    parents = {}
    records_by_identity = defaultdict(list)

    def find(node):
        parents.setdefault(node, node)
        if parents[node] != node:
            parents[node] = find(parents[node])
        return parents[node]

    def union(left, right):
        left, right = find(left), find(right)
        if left != right:
            parents[max(left, right)] = min(left, right)

    for item in identity_records:
        r = item["identity"]
        identity = (r["structure_text"], r["adduct"], r["rule_sheet"])
        node = ("identity", *identity)
        find(node)
        for row in item["records"]:
            _require(bool(row["source_file"]), "MISSING_SOURCE_FILENAME")
            union(node, ("file", row["source_file"]))
            records_by_identity[identity].append(row)

    members = defaultdict(list)
    for pattern in patterns:
        identity = tuple(pattern["donor_identity"])
        _require(identity in records_by_identity, "DONOR_IDENTITY_NOT_IN_ALL_RECORDS")
        known = {r["row_uid"]: r for r in records_by_identity[identity]}
        _require(pattern["source_low"] in known and pattern["source_high"] in known,
                 "DONOR_ENDPOINT_NOT_IN_ALL_RECORDS")
        files = {known[pattern[k]]["source_file"] for k in ("source_low", "source_high")}
        _require(files == set(pattern["source_files"]), "DONOR_SOURCE_FILES_CHANGED")
        members[find(("identity", *identity))].append(pattern)

    graph_members = defaultdict(list)
    for node in parents:
        graph_members[find(node)].append(node)
    groups = []
    for component, rows in members.items():
        pattern_ids = sorted(r["pattern_id"] for r in rows)
        group_id = hashlib.sha256((DONOR_NAMESPACE + "|" + _canonical(pattern_ids)).encode()).hexdigest()
        nodes = graph_members[component]
        all_identities = sorted([list(n[1:]) for n in nodes if n[0] == "identity"])
        all_files = sorted(n[1] for n in nodes if n[0] == "file")
        all_uids = sorted({r["row_uid"] for identity in all_identities
                           for r in records_by_identity[tuple(identity)]})
        groups.append(dict(group_id=group_id, pattern_ids=pattern_ids,
                           donor_identities=all_identities, source_files=all_files,
                           all_source_row_uids=all_uids))
    groups.sort(key=lambda g: g["group_id"])
    pattern_roles = {}
    for index, group in enumerate(groups):
        group["role"] = ROLE_SLOTS[index % len(ROLE_SLOTS)]
        pattern_roles.update({pid: group["role"] for pid in group["pattern_ids"]})
    for field in ("source_files", "all_source_row_uids", "donor_identities"):
        seen = {}
        for group in groups:
            for member in group[field]:
                key = _canonical(member)
                _require(key not in seen, "CROSS_GROUP_SOURCE_LEAKAGE:" + field)
                seen[key] = group["group_id"]

    support_sets = sorted({(p["donor_identity"][2], tuple(p["channels"])) for p in patterns})
    pairs = sorted({tuple(p["ce_pair"]) for p in patterns}, key=lambda p: (p[1] - p[0], p))
    counts = []
    eligible_pairs = []
    for pair in pairs:
        eligible = True
        for rule, channels in support_sets:
            row = dict(ce_pair=list(pair), rule=rule, channels=list(channels), roles={})
            for role in ROLES:
                rows = [p for p in patterns if tuple(p["ce_pair"]) == pair
                        and p["donor_identity"][2] == rule and tuple(p["channels"]) == channels
                        and pattern_roles[p["pattern_id"]] == role]
                row["roles"][role] = dict(pattern_count=len(rows), distinct_identity_count=len(
                    {tuple(p["donor_identity"]) for p in rows}))
                eligible = eligible and bool(rows)
            counts.append(row)
        if eligible:
            eligible_pairs.append(list(pair))
    _require(eligible_pairs, "NO_COMMON_CE_PAIR_AFTER_ALL_RECORD_GROUPING_NO_REDRAW")
    result = dict(status="FROZEN_BEFORE_GENERATION_AND_OUTCOMES", namespace=DONOR_NAMESPACE,
                  grouping="All identity CE records and transitive shared files, including non-admitted bridge records",
                  group_sort="sha256(namespace + '|' + canonical(sorted admitted pattern_ids))",
                  role_slots=list(ROLE_SLOTS), groups=groups, pattern_roles=pattern_roles,
                  role_group_counts=dict(Counter(g["role"] for g in groups)),
                  role_pattern_counts=dict(Counter(pattern_roles.values())),
                  chosen_ce_pair=eligible_pairs[0], eligible_ce_pairs=eligible_pairs,
                  ce_selection="Shortest CE span, then numeric low/high; require every support set in all three roles",
                  counts=counts, input_patterns_sha256=_fingerprint(patterns),
                  all_identity_records_sha256=_fingerprint(identity_records))
    result["fingerprint"] = _fingerprint(result)
    return result


def build_target_library(A, fractions, patterns, mapping, split, role, metadata):
    """Return ``(A_target, binding)`` using one complete held-out endpoint per identity.

    ``A`` is the complete 391-column nominal production library. ``fractions``
    has one row per mass channel and one column per original physical component.
    The caller handles the sole dataset abundance scalar after this operation.
    """
    import numpy as np

    _require(role in ("CAL", "EVAL"), "TARGET_ROLE_MUST_BE_CAL_OR_EVAL")
    _require(split["fingerprint"] == _fingerprint({k: v for k, v in split.items() if k != "fingerprint"}),
             "DONOR_SPLIT_CHANGED")
    _require(split["input_patterns_sha256"] == _fingerprint(patterns), "DONOR_PATTERNS_CHANGED")
    nominal = np.asarray(A)
    _require(nominal.ndim == 2 and nominal.shape[1] == 391 and nominal.dtype.kind == "f",
             "COMPLETE_PRODUCTION_LIBRARY_REQUIRED")
    _require(np.isfinite(nominal).all() and (nominal >= 0).all(), "INVALID_NOMINAL_LIBRARY")
    fractions = np.asarray(fractions, dtype=np.float64)
    _require(fractions.ndim == 2 and fractions.shape[0] == nominal.shape[0]
             and np.isfinite(fractions).all() and (fractions >= 0).all(), "INVALID_COMPONENT_FRACTIONS")
    _require(len(metadata["lipid_name"]) == 391, "METADATA_LENGTH_CHANGED")
    ids = [r["candidate_index"] for r in mapping]
    _require(len(ids) == len(set(ids)) and all(0 <= i < 391 for i in ids), "INVALID_CANDIDATE_MAPPING")
    patterns_by_id = {p["pattern_id"]: p for p in patterns}
    allowed_role_ids = {pid for pid, r in split["pattern_roles"].items() if r == role}
    model_ids = {pid for pid, r in split["pattern_roles"].items() if r == "MODEL"}
    _require(not allowed_role_ids.intersection(model_ids), "MODEL_TARGET_DONOR_LEAKAGE")
    source_sets = {r: {s for g in split["groups"] if g["role"] == r for s in g["source_files"]}
                   for r in ROLES}
    _require(not source_sets[role].intersection(source_sets["MODEL"]), "MODEL_TARGET_SOURCE_LEAKAGE")
    target = nominal.copy()
    assigned = {}
    assignments = []
    norm_errors = []
    for row in sorted(mapping, key=lambda r: r["candidate_index"]):
        i = row["candidate_index"]
        _require(str(metadata["lipid_name"][i]) == row["lipid_name"], "CANDIDATE_NAME_CHANGED")
        if "candidate_id" in metadata:
            _require(str(metadata["candidate_id"][i]) == row["candidate_id"], "CANDIDATE_ID_CHANGED")
        group = [row["rule"], row["lipid_name"], row["channels"]]
        canonical_group = _canonical(group)
        eligible = sorted(pid for pid in row["allowed_pattern_ids"] if pid in allowed_role_ids
                          and patterns_by_id[pid]["ce_pair"] == split["chosen_ce_pair"])
        _require(eligible, "NO_TARGET_ENDPOINT_FOR_CANDIDATE_NO_REDRAW")
        for pid in eligible:
            p = patterns_by_id[pid]
            _require(p["channels"] == row["channels"] and p["donor_identity"][2] == row["rule"],
                     "TARGET_RULE_OR_CHANNEL_SET_CHANGED")
        choice = int(hashlib.sha256((TARGET_NAMESPACE + "|" + role + "|" + canonical_group).encode()).hexdigest(), 16)
        pid = eligible[choice % len(eligible)]
        _require(canonical_group not in assigned or assigned[canonical_group] == pid,
                 "DUPLICATE_IDENTITY_ENDPOINT_CHANGED")
        assigned[canonical_group] = pid
        p = patterns_by_id[pid]
        a = nominal[:, i].astype(np.float64)
        component_indices = row["component_indices"]
        _require(len(component_indices) == len(row["channels"]) and len(set(component_indices)) == len(component_indices),
                 "COMPONENT_MEMBERSHIP_CHANGED")
        _require(all(0 <= index < fractions.shape[1] for index in component_indices),
                 "COMPONENT_INDEX_OUT_OF_RANGE")
        f = fractions[:, component_indices].T
        _require((f.sum(axis=0) <= 1. + 1e-12).all(), "COMPONENT_FRACTIONS_OVERLAP")
        D = f * a[None, :]
        multipliers = np.asarray(p["multiplier"], dtype=np.float64)
        _require(multipliers.shape == (len(component_indices),) and np.isfinite(multipliers).all()
                 and (multipliers > 0).all(), "INVALID_POSITIVE_ENDPOINT_MULTIPLIER")
        t = a - D.sum(axis=0) + (multipliers[:, None] * D).sum(axis=0)
        norm = float(np.linalg.norm(a))
        _require(norm > 0 and np.isfinite(t).all() and (t >= 0).all()
                 and float(np.linalg.norm(t)) > 0, "INVALID_TARGET_ENDPOINT")
        value = (norm * t / np.linalg.norm(t)).astype(nominal.dtype)
        _require(np.isfinite(value).all() and (value >= 0).all(), "NONFINITE_TARGET_LIBRARY")
        _require(np.array_equal(value > 0, nominal[:, i] > 0), "TARGET_SUPPORT_CHANGED")
        norm_error = abs(float(np.linalg.norm(value.astype(np.float64))) / norm - 1.)
        _require(norm_error <= 8 * np.finfo(nominal.dtype).eps, "TARGET_COLUMN_NORM_CHANGED")
        norm_errors.append(norm_error)
        target[:, i] = value
        assignments.append(dict(candidate_index=i, candidate_id=row["candidate_id"], lipid_name=row["lipid_name"],
                                rule=row["rule"], channels=row["channels"], pattern_id=pid,
                                donor_identity=p["donor_identity"], source_files=p["source_files"],
                                relative_l2_change=float(np.linalg.norm(value.astype(np.float64) - a) / norm),
                                relative_column_norm_error=norm_error))
    fixed = sorted(set(range(391)) - set(ids))
    _require(np.array_equal(target[:, fixed], nominal[:, fixed]), "UNSUPPORTED_CANDIDATE_CHANGED")
    _require(np.array_equal(target > 0, nominal > 0), "LIBRARY_SUPPORT_CHANGED")
    binding = dict(role=role, namespace=TARGET_NAMESPACE, donor_split_fingerprint=split["fingerprint"],
                   ce_pair=split["chosen_ce_pair"], assignments=assignments,
                   assignment_rule="sha256(namespace+'|'+role+'|'+canonical([rule,lipid_name,channels])) modulo sorted eligible pattern IDs",
                   A_solver_sha256=hashlib.sha256(nominal.tobytes()).hexdigest(),
                   A_target_sha256=hashlib.sha256(target.tobytes()).hexdigest(),
                   relative_frobenius_change=float(np.linalg.norm(target.astype(np.float64) - nominal) /
                                                  np.linalg.norm(nominal.astype(np.float64))),
                   max_relative_column_norm_error=max(norm_errors, default=0.),
                   supported_candidate_indices=sorted(ids), fixed_candidate_indices=fixed,
                   source_files_disjoint_from_MODEL=True, unchanged_support=True,
                   positive_finite_multipliers=True, original_column_norms_preserved=True,
                   same_endpoint_all_pixels_and_challenges=True, per_identity_X_scaling=False)
    binding["fingerprint"] = _fingerprint(binding)
    return target, binding
