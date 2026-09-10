#!/usr/bin/env python3
"""Final compact V58 spectral-library mismatch robustness and FDR recalibration.

Lifecycle: prepare -> audit -> freeze -> 60 target oracles -> four sentinels
-> mismatch-all -> aggregate (CAL -> immutable thresholds -> HOLD).
No measurement noise, source solver changes, or CLEAN retraining.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VERSION = "v58_spectral_library_mismatch_fdr_recalibration"
PERTURBATION_NAMESPACE = "v58_spectral_library_mismatch_fdr_recalibration"
RESULT_VARIANT = "v58_spectral_library_mismatch_fdr_recalibration_compact"
ADOPTED_SENTINEL_MODE = "ADOPTED_EQUIVALENT_SUPERSEDED_SENTINEL"
NATIVE_SENTINEL_MODE = "COMPACT_LEARNED_SENTINEL"
# Exact pre-adoption compact runner. Only the explicit adoption action can authorize
# this code-only transition; the frozen design and its fingerprint are never edited.
PRE_ADOPTION_COMPACT_RUNNER_SHA256 = "30a641cf9402735adf4896290093d9da07dcd257113e09a49ed776c54af9c3d2"
EQUIVALENCE_RTOL = 1e-12
EQUIVALENCE_ATOL = 1e-15
SUPERSEDED_DRAFT = {
    "superseded_draft_fingerprint": "d9df4ea9d9395573f47104572816b607e0728065f8171d5745dc5748337f1b9a",
    "superseded_draft_status": "FROZEN_AND_TARGET_ORACLE_PASS_BEFORE_LEARNED_TRAINING",
    "superseded_draft_disposition": "SUPERSEDED_BEFORE_LEARNED_TRAINING",
    "superseded_draft_mismatch_runs": 120,
    "superseded_draft_reason": "compute-efficiency redesign: six-point nested K ladder was redundant for the primary mismatch/FDR "
                               "recalibration question; no learned-solver mismatch training had begun.",
}
EXPECTED_DRAFT_LIBRARY_HASHES = {
    "A_solver_sha256": "b9e05e185ffa692966b86022f5487fcdabff92f330a0a389a6bb4889a175a447",
    "A_target_MILD_sha256": "b835ef8e531e12803c1a949b0e89fa35c32f0f82ea2dbfefa2fefbcc9dcfa24b",
    "A_target_MODERATE_sha256": "838af17725b3bdf28fd83668d86ce3856c4e8d6650b2e9785bf7c03cd5206ecd",
}
COMPACT_RATIONALE = (
    "V58 primary question is mismatch-domain identity-confidence calibration, not reconstruction-complexity mapping. "
    "V57 has already established the six-point K complexity curve. Because V57 K levels are nested, repeating six K "
    "levels in both mismatch severities produces strongly correlated repeated identity observations. K50/K125/K175 "
    "retain the low endpoint, intermediate state and high endpoint, while K175 preserves the full 175-identity "
    "CAL/HOLD molecular-identity coverage. Five spatial replicates are retained because they provide meaningful "
    "variation in spatial realization and solver behavior."
)
PARENT_VERSION = "v57_spectral_spatial_identity_confidence_benchmark"
PARENT_FINGERPRINT = "805c310f2c3778206e2ba4680c00fd6b15266806f3720d3dfaeeb4ad246071e9"
PARENT_COMMIT = "705c52c128dbd73496ba970f6b9a61874c3f292c"
EXPECTED_CLEAN_RHO = {"FDR5": 1.272651000158792e-20, "FDR1": 6.683949277609072e-20}
PARENT_K_LEVELS = (50, 75, 100, 125, 150, 175)
K_LEVELS = (50, 125, 175)
REPLICATES = tuple(f"R{i}" for i in range(1, 6))
SEVERITIES = {
    "MILD": {"fragment_log_sigma": .10, "fragment_dropout_rate": .05, "parent_deviation": .10},
    "MODERATE": {"fragment_log_sigma": .20, "fragment_dropout_rate": .10, "parent_deviation": .20},
}
SENTINELS = tuple(f"{s}__CAL_R1_K{k:03d}" for s in SEVERITIES for k in (50, 175))
SELECTION = "maximum retained molecular units with empirical CAL FDR<=alpha; lower threshold breaks ties; pooled K/R"
GATE_FORMULA = "min(0.50, max(0.10, 2.0 * forward_mismatch_residual + 0.05))"
CONTRACT = {
    "design_seed": 5800, "K_levels": K_LEVELS, "replicates": REPLICATES, "splits": ("CAL", "HOLD"),
    "severity_parameters": SEVERITIES, "measurement_noise": False, "mismatch_runs": 60, "clean_runs": 0,
    "result_variant": RESULT_VARIANT, "perturbation_namespace": PERTURBATION_NAMESPACE,
    "parent_K_levels": PARENT_K_LEVELS, "compact_rationale": COMPACT_RATIONALE,
    "expected_draft_library_hashes": EXPECTED_DRAFT_LIBRARY_HASHES, **SUPERSEDED_DRAFT,
    "parent_fingerprint": PARENT_FINGERPRINT, "parent_commit": PARENT_COMMIT,
    "X_true": "unmodified V57 frozen construction, including its original single source-domain global scalar",
    "nested_perturbation": "SHA256(V58|5800|candidate_index|kind), shared z/u/v; severity absent from seed",
    "protected_fragment": "strongest originally nonzero fragment; lowest channel index breaks ties; never dropped",
    "normalization": "V52 normalize_like_get_A_matrix: float32 column/(L2 norm+1e-8)",
    "target_domain": "all 391 candidates; one fixed full library per severity across all K/R/CAL/HOLD",
    "post_mismatch_rescaling": False, "A_target_used_by_solver_or_rho": False,
    "hard_epoch_cap": 3000, "checkpoint_epochs": (1000, 1500, 2000, 2500, 3000),
    "production_early_stop_unchanged": True, "sentinel_requires_epoch_3000": False,
    "sentinels": SENTINELS, "sentinel_gate": GATE_FORMULA,
    "solver_validity": "stop_reason in {converged,max_epochs}; all outputs/losses finite; sentinel full-cube residual<=gate",
    "sentinel_identity_outcomes_used": False, "oracle_max_relative_residual": 1e-5,
    "report_gate": .001, "rho_zero_weighting": "W=I", "identity_unit": "lipid_name",
    "molecular_score": "max reported-candidate rho; sum reported-candidate X_hat; no group-rho",
    "threshold_protocols": {"CLEAN_FIXED": "verified V57 thresholds, no adjustment",
                            "LOCAL_CAL_RECALIBRATION": SELECTION},
    "target_FDR": {"FDR1": .01, "FDR5": .05},
    "HOLD_read_guard": "both severity CAL threshold files and SHA seals written before any HOLD record read",
    "scientific_outcomes_affect_process_validity": False,
}
LIMITATIONS = [
    "Synthetic library mismatch has fixed, nested directions; it is not measured spectral error or measurement noise.",
    "V57 identities, spatial morphology, abundance and source-domain signal scaling are inherited unchanged.",
    "CAL/HOLD separates truth molecular identities; replicates and nested K are not independent biological samples.",
    "Because K is nested, the same molecular identity may contribute multiple dataset-level observations across K and "
    "replicates; these are correlated solver contexts, not independent molecular samples. Calibration retains the "
    "dataset x molecular-identity unit without deduplication.",
    "Empirical CAL FDR and descriptive Wilson intervals do not guarantee population FDR.",
    "Target NNLS checks reporting-gate identifiability; source NNLS is a mismatch diagnostic only.",
    "Truth roles cannot attribute false positives; category FP/FDR are not applicable.",
    "A failed scientific transfer/recalibration result is not process invalidity; deployment threshold is not frozen.",
]
PARENT_FILES = ("design.json", "design_audit.json", "global_frozen_thresholds.json", "report.json",
                "calibration_records.csv", "heldout_records.csv")


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def dependencies():
    # --help needs only the standard library, never assets or a GPU.
    global np, v57, v56, v55, v54
    import numpy as np
    sys.path.insert(0, str(ROOT / "analysis"))
    import run_v57_spectral_spatial_identity_confidence_benchmark as v57
    v56, v55, v54 = v57.v56, v57.v55, v57.v54


def base_ids(split=None):
    return [f"{s}_{r}_K{k:03d}" for s in ((split,) if split else ("CAL", "HOLD"))
            for r in REPLICATES for k in K_LEVELS]


def parent_base_ids(split=None):
    return [f"{s}_{r}_K{k:03d}" for s in ((split,) if split else ("CAL", "HOLD"))
            for r in REPLICATES for k in PARENT_K_LEVELS]


def guard_output(args):
    output = args.output_dir.resolve()
    old = (ROOT / "results" / VERSION).resolve()
    require(output != old and old not in output.parents and output not in old.parents,
            "REFUSE_TO_OVERWRITE_SUPERSEDED_FROZEN_DRAFT")
    existing = output / "design.json"
    if existing.is_file():
        require(load_json(existing).get("design_fingerprint") != SUPERSEDED_DRAFT["superseded_draft_fingerprint"],
                "REFUSE_TO_OVERWRITE_SUPERSEDED_FROZEN_DRAFT")


def dataset_ids():
    return [f"{s}__{d}" for s in SEVERITIES for d in base_ids()]


def parts(dataset):
    require(dataset in dataset_ids(), f"INVALID_DATASET_ID: {dataset}")
    severity, base = dataset.split("__")
    split, replicate, label = base.split("_")
    return severity, base, split, replicate, int(label[1:])


def directory(args, dataset):
    severity, base, *_ = parts(dataset)
    return args.output_dir / severity / base


def array_sha(value):
    # Same byte-level definition as V57 spatial realization hashes.
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def load_json(path):
    require(path.is_file(), f"MISSING_REQUIRED_FILE: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path, value):
    v57.write_json(path, value)


def write_rows(path, values, fields=None):
    v57.write_rows(path, values, fields or (None if values else ["status"]))


def implementation_hashes():
    return {**v57.source_implementations(), Path(__file__).name: v56.digest(Path(__file__)),
            Path(v54.v52.__file__).name: v56.digest(Path(v54.v52.__file__)),
            Path(v54.v53.__file__).name: v56.digest(Path(v54.v53.__file__)),
            Path(v54.stage0.__file__).name: v56.digest(Path(v54.stage0.__file__))}


def parent_provenance(args):
    """Pin actual file bytes to the specified committed parent, not supplied constants alone."""
    hashes = {}
    for name in PARENT_FILES:
        path = args.v57_output / name
        require(path.is_file(), f"MISSING_V57_PARENT_FILE: {path}")
        ref = f"{PARENT_COMMIT}:results/{PARENT_VERSION}/{name}"
        result = subprocess.run(["git", "-C", str(ROOT), "show", ref], capture_output=True)
        require(result.returncode == 0, f"V57_COMMITTED_PARENT_UNAVAILABLE: {ref}; {result.stderr.decode(errors='replace')}")
        expected = hashlib.sha256(result.stdout).hexdigest()
        hashes[name] = v56.digest(path)
        require(hashes[name] == expected, f"V57_PARENT_FILE_CHANGED_FROM_COMMIT: {name}")
    parent = load_json(args.v57_output / "design.json")
    require(parent["status"] == "DESIGN_FROZEN_BEFORE_TRAINING" and
            parent["design_fingerprint"] == v57.fingerprint(parent["scientific"]) == PARENT_FINGERPRINT,
            "V57_FINGERPRINT_CHANGED")
    require(parent["frozen_audit_sha256"] == hashes["design_audit.json"], "V57_FROZEN_AUDIT_CHANGED")
    audit = load_json(args.v57_output / "design_audit.json")
    require(audit["status"] == "PASS" and audit["design_fingerprint"] == PARENT_FINGERPRINT, "V57_AUDIT_NOT_PASS")
    thresholds = load_json(args.v57_output / "global_frozen_thresholds.json")
    require(thresholds["design_fingerprint"] == PARENT_FINGERPRINT and
            thresholds["status"] == "FROZEN_FROM_CAL_ONLY", "V57_THRESHOLD_PROVENANCE_CHANGED")
    require(set(thresholds["CAL_source_hashes"]) == set(parent_base_ids("CAL")), "V57_CAL_SOURCE_SET_CHANGED")
    for target, expected in EXPECTED_CLEAN_RHO.items():
        require(thresholds["global_thresholds"]["rho_zero"][target]["threshold"] == expected ==
                thresholds[f"rho_tau_{target}"], f"V57_THRESHOLD_VALUE_CHANGED: {target}")
    report = load_json(args.v57_output / "report.json")
    require(report["design_fingerprint"] == PARENT_FINGERPRINT and
            report["primary_global_CAL_thresholds"] == thresholds, "V57_REPORT_THRESHOLD_MISMATCH")
    return parent, thresholds, hashes


def load_context(args, parent):
    context = v56.load_context(args.asset_root)
    require(not context.get("missing_dependencies"), f"MISSING_PRODUCTION_ASSETS: {context.get('missing_dependencies')}")
    scientific = parent["scientific"]
    require(context["validation"]["hashes_sha256"] == scientific["input_hashes"] and
            context["result_hashes"] == scientific["v51_stage0_hashes"], "V57_PRODUCTION_ASSET_SHA_CHANGED")
    require(scientific["implementation_hashes"] == v57.source_implementations(), "V57_IMPLEMENTATION_CHANGED")
    require(scientific["contract"] == v57.canonical(v57.CONTRACT), "V57_CONTRACT_CHANGED")
    v57.audit_units(scientific["matched_units"], context)
    require(len(context["singleton_indices"]) == 368, "IDENTITY_POOL_CHANGED")
    mz = np.asarray(np.load(context["paths"]["channel_axis"]), dtype=np.float64).reshape(-1)
    require(mz.size == 1084 and np.isfinite(mz).all(), "CHANNEL_AXIS_CHANGED")
    parent_mask = (mz >= v55.PARENT_MZ_RANGE[0]) & (mz <= v55.PARENT_MZ_RANGE[1])
    require(parent_mask.any() and not parent_mask.all(), "INVALID_CHANNEL_PARTITION")
    context.update(parent_mask=parent_mask, fragment_mask=~parent_mask)
    validate_source(context)
    return context


def validate_source(context, expected=None):
    A = context["A_solver"]
    require(A.shape == (1084, 391) and np.isfinite(A).all() and (A >= 0).all(), "INVALID_A_SOLVER")
    require(np.array_equal(A, v54.v52.normalize_like_get_A_matrix(context["A_raw"])), "A_SOLVER_MUTATED")
    if expected is not None:
        require(array_sha(A) == expected, "A_SOLVER_SHA_CHANGED")


def latent_seed(index, kind):
    text = f"{PERTURBATION_NAMESPACE}|{CONTRACT['design_seed']}|{int(index)}|{kind}"
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "little")


def build_targets(context):
    """Target arrays stay outside the context passed to training and rho."""
    A, parent, fragment = context["A_solver"], context["parent_mask"], context["fragment_mask"]
    before = array_sha(A)
    targets = {s: A.copy() for s in SEVERITIES}
    manifest = []
    for j in range(A.shape[1]):
        original = A[:, j]
        nonzero = np.flatnonzero(fragment & (original > 0))
        protected = int(nonzero[np.argmax(original[nonzero])]) if len(nonzero) else None
        seeds = {kind: latent_seed(j, kind) for kind in ("fragment_intensity", "fragment_dropout", "parent")}
        z = np.random.default_rng(seeds["fragment_intensity"]).standard_normal(len(nonzero))
        u = np.random.default_rng(seeds["fragment_dropout"]).random(len(nonzero))
        v = float(np.random.default_rng(seeds["parent"]).uniform(-1, 1))
        latent = {"z_sha256": array_sha(z), "u_sha256": array_sha(u), "v": v, "seeds": seeds}
        realized = {}
        for severity, cfg in SEVERITIES.items():
            log_distortion = cfg["fragment_log_sigma"] * z
            dropped = (u < cfg["fragment_dropout_rate"]) & (nonzero != protected)
            multiplier = 1 + cfg["parent_deviation"] * v
            column = original.copy()
            column[nonzero] = (original[nonzero].astype(np.float64) * np.exp(log_distortion)).astype(np.float32)
            column[nonzero[dropped]] = 0
            column[parent] *= np.float32(multiplier)
            require(np.isfinite(column).all() and (column >= 0).all() and np.linalg.norm(column) > 0,
                    "INVALID_PRENORMALIZATION_TARGET")
            column = v54.v52.normalize_like_get_A_matrix(column[:, None])[:, 0]
            require(np.array_equal(column[parent] > 0, original[parent] > 0), "PARENT_SUPPORT_CHANGED")
            require(protected is None or column[protected] > 0, "PROTECTED_FRAGMENT_DROPPED")
            require(np.array_equal(column[nonzero] == 0, dropped), "UNREQUESTED_FRAGMENT_DROPOUT")
            require(not np.any(column[original == 0]), "TARGET_CREATED_SUPPORT")
            targets[severity][:, j] = column
            realized[severity] = (dropped, multiplier, log_distortion)
            manifest.append({"severity": severity, "candidate_index": j,
                "candidate_id": str(context["metadata"]["candidate_id"][j]),
                "lipid_name": str(context["metadata"]["lipid_name"][j]),
                "fragment_nonzero_count": len(nonzero), "protected_fragment_channel": protected,
                "requested_dropout_rate": cfg["fragment_dropout_rate"], "dropped_fragment_count": int(dropped.sum()),
                "dropped_fragment_channels": nonzero[dropped].tolist(),
                "achieved_dropout_rate": float(dropped.mean()) if len(dropped) else 0.0,
                "parent_multiplier": multiplier, "fragment_log_sigma": cfg["fragment_log_sigma"],
                "full_cosine_A_target_vs_A_solver": v55.cosine(column, original),
                "fragment_cosine_A_target_vs_A_solver": v55.cosine(column[fragment], original[fragment]),
                "parent_cosine_A_target_vs_A_solver": v55.cosine(column[parent], original[parent]),
                "source_column_l2": float(np.linalg.norm(original)), "target_column_l2": float(np.linalg.norm(column)),
                "latent_provenance": latent, "pre_dropout_log_distortion_sha256": array_sha(log_distortion)})
        mild, moderate = realized["MILD"], realized["MODERATE"]
        require(np.all(~mild[0] | moderate[0]), "DROPOUT_NESTING_BROKEN")
        require(math.isclose(moderate[1] - 1, 2 * (mild[1] - 1), abs_tol=5e-16), "PARENT_NESTING_BROKEN")
        require(np.array_equal(moderate[2], 2 * mild[2]), "LOG_DISTORTION_NESTING_BROKEN")
    require(before == array_sha(A), "A_SOLVER_MUTATED")
    for target in targets.values():
        require(target.shape == (1084, 391) and np.isfinite(target).all() and (target >= 0).all(), "INVALID_A_TARGET")
        require(np.allclose(np.linalg.norm(target, axis=0), 1, rtol=2e-6, atol=2e-7), "TARGET_NORMALIZATION_INVALID")
    hashes = {"A_solver_sha256": before, **{f"A_target_{s}_sha256": array_sha(t) for s, t in targets.items()}}
    return targets, manifest, hashes


def relative_residual(left, right, mask=None):
    if mask is not None:
        left, right = left[:, mask], right[:, mask]
    # float64 accumulation; bounded-memory cube conversion at a single diagnostic boundary.
    difference = np.asarray(left, dtype=np.float64) - right
    return float(np.linalg.norm(difference) / max(float(np.linalg.norm(np.asarray(right, dtype=np.float64))), 1e-30))


def collapse_gate(forward):
    require(math.isfinite(forward) and forward >= 0, "INVALID_FORWARD_MISMATCH_RESIDUAL")
    return min(.50, max(.10, 2.0 * forward + .05))


def mismatch_case(context, parent, targets, dataset):
    severity, base, split, replicate, k = parts(dataset)
    clean = v57.construct_case(context, parent["scientific"], base)
    x_sha = array_sha(clean["X_true"])
    B = np.einsum("mc,cyx->myx", targets[severity], clean["X_true"], optimize=True)
    require(np.isfinite(B).all() and (B >= 0).all(), "INVALID_B_TARGET")
    require(array_sha(clean["X_true"]) == x_sha, "X_TRUE_MUTATED")
    forward = relative_residual(clean["B_sim"], B, context["mask"])
    domain = v57.domain_stats(B, context["mask"])
    clean_domain = v57.domain_stats(clean["B_sim"], context["mask"])
    record = {"dataset_id": dataset, "base_dataset_id": base, "severity": severity, "split": split,
        "replicate": replicate, "K": k, "X_true_sha256": x_sha, "V57_clean_X_true_sha256": x_sha,
        "B_target_sha256": array_sha(B), "V57_clean_B_sha256": array_sha(clean["B_sim"]),
        "forward_mismatch_residual": forward, "sentinel_gate_residual": collapse_gate(forward),
        "B_domain": domain, "mismatch_B_p50_over_V57_clean_B_p50": domain["p50"] / clean_domain["p50"],
        "V57_global_scale": clean["global_scale"], "post_mismatch_rescaling": False,
        "truth_indices": clean["active_indices"], "reportable_truth_count": clean["reportable_truth_molecular_identity_count"]}
    case = {**clean, "case_name": f"V58__{dataset}", "B_sim": B}
    return case, record


def snapshot(args, context, parent_hashes, library_hashes):
    require(library_hashes == EXPECTED_DRAFT_LIBRARY_HASHES, "COMPACT_TARGET_LIBRARIES_CHANGED_FROM_DRAFT")
    return v57.canonical({"version": VERSION, "result_variant": RESULT_VARIANT, "contract": CONTRACT,
        "V57_parent_file_hashes": parent_hashes,
        "production_asset_hashes": context["validation"]["hashes_sha256"], "stage0_hashes": context["result_hashes"],
        "source_implementation_hashes": implementation_hashes(), "target_library_hashes": library_hashes,
        "candidate_order": [{"candidate_index": j, "candidate_id": str(context["metadata"]["candidate_id"][j]),
                             "lipid_name": str(context["metadata"]["lipid_name"][j])} for j in range(391)],
        "dataset_ids": dataset_ids(), "source_provenance": context["provenance"]})


def update_log(design, report=None):
    begin, end = f"<!-- BEGIN {RESULT_VARIANT} -->", f"<!-- END {RESULT_VARIANT} -->"
    path = ROOT / "results/EXPERIMENT_LOG.md"
    text = path.read_text(encoding="utf-8") if path.exists() else ""
    payload = report or {"status": "DESIGN_PREPARED; audit/oracle/sentinel/results pending"}
    section = "\n".join([begin, "", "## Final compact V58 spectral-library mismatch and FDR recalibration", "",
        "Final compact V58. Supersedes frozen 120-run pre-training draft for compute efficiency.",
        "The superseded-draft metadata records the pre-training redesign declaration. Any later adoption of draft sentinel "
        "training is recorded separately in sentinel_solver_validity.json, retaining its original provenance.",
        "Target perturbation libraries are byte-identical to the superseded draft.",
        json.dumps(SUPERSEDED_DRAFT, sort_keys=True), COMPACT_RATIONALE,
        "Scientific questions: CLEAN threshold transfer; target CAL recalibration; true/false rho separation and drift.",
        f"V57 parent: {PARENT_FINGERPRINT}; commit {PARENT_COMMIT}.",
        f"V58 design fingerprint: {design['design_fingerprint']}.",
        "MILD sigma/dropout/parent deviation=0.10/0.05/0.10; MODERATE=0.20/0.10/0.20.",
        "Shared candidate z/u/v, protected strongest fragment, nested dropout; fixed full 391-column target libraries.",
        "60 mismatch datasets; K50/K125/K175; V57 X_true unchanged; CLEAN runs=0; measurement noise=false; no rescaling.",
        f"Sentinel: four runs; normal finite completion and residual <= {GATE_FORMULA}; early-stop retained.",
        "```json", json.dumps(v57.canonical(payload), ensure_ascii=False, indent=2), "```",
        *[f"- {item}" for item in LIMITATIONS], "Final deployment threshold frozen: false.", "", end])
    require(text.count(begin) == text.count(end) and text.count(begin) <= 1, "DUPLICATE_OR_BROKEN_V58_LOG_MARKER")
    if begin in text:
        start, stop = text.index(begin), text.index(end) + len(end)
        text = text[:start] + section + text[stop:]
    else:
        text = text.rstrip() + "\n\n" + section + "\n"
    v56.stage0.atomic_write_text(path, text)


def prepare(args, scientific, manifest):
    guard_output(args)
    path = args.output_dir / "design.json"
    require(not path.exists(), "DESIGN_EXISTS: audit existing design; never silently regenerate")
    design = {"status": "DESIGN_PREPARED", "scientific": scientific, **SUPERSEDED_DRAFT,
              "design_fingerprint": v57.fingerprint(scientific)}
    write_rows(args.output_dir / "spectral_perturbation_manifest.csv", manifest)
    write_json(args.output_dir / "target_library_hashes.json", scientific["target_library_hashes"])
    write_json(path, design)
    update_log(design)
    print("DESIGN_PREPARED; audit/freeze/oracle/sentinel pending; no training")
    return design


def load_design(args, expected):
    guard_output(args)
    design = load_json(args.output_dir / "design.json")
    require(not design["status"].startswith("INVALID"), "V58_INVALID; no outcome-driven redesign")
    require(design["design_fingerprint"] == v57.fingerprint(design["scientific"]), "DESIGN_CONTENT_CHANGED")
    if design["scientific"] != expected:
        # A new evidence reader changes this runner's file hash, not any scientific
        # input. Permit precisely the pinned compact predecessor, never other code.
        runner = Path(__file__).name
        previous_sha = design["scientific"]["source_implementation_hashes"].get(runner)
        require(previous_sha == PRE_ADOPTION_COMPACT_RUNNER_SHA256, "UNSUPPORTED_FROZEN_RUNNER_UPGRADE")
        compatible = json.loads(json.dumps(expected))
        compatible["source_implementation_hashes"][runner] = previous_sha
        require(design["scientific"] == compatible, "DESIGN_PROVENANCE_OR_SCIENTIFIC_CONTRACT_CHANGED")
        if not getattr(args, "adopt_equivalent_sentinel", False):
            evidence = load_json(args.output_dir / "sentinel_solver_validity.json")
            require(evidence.get("evidence_mode") == ADOPTED_SENTINEL_MODE and
                    evidence.get("adoption_runner_sha256") == expected["source_implementation_hashes"][runner] and
                    evidence.get("design_fingerprint") == design["design_fingerprint"],
                    "EXPLICIT_EQUIVALENT_SENTINEL_ADOPTION_REQUIRED_FOR_RUNNER_UPGRADE")
    require(all(design.get(k) == v for k, v in SUPERSEDED_DRAFT.items()), "SUPERSEDED_DRAFT_PROVENANCE_CHANGED")
    require(load_json(args.output_dir / "target_library_hashes.json") == expected["target_library_hashes"], "TARGET_LIBRARY_HASHES_CHANGED")
    return design


def invalidate(args, design, reason):
    design.update(status="INVALID_PROCESS", failure_reason=reason)
    write_json(args.output_dir / "design.json", design)


def audit_design(args, context, parent, targets, design, manifest):
    _, again, hashes = build_targets(context)
    require(hashes == design["scientific"]["target_library_hashes"] and again == manifest, "NONDETERMINISTIC_A_TARGET")
    require(hashes == EXPECTED_DRAFT_LIBRARY_HASHES, "COMPACT_TARGET_LIBRARIES_CHANGED_FROM_DRAFT")
    # Compare parsed CSV cells using the shared JSON-cell encoding.
    persisted = v57.rows(args.output_dir / "spectral_perturbation_manifest.csv")
    expected = [{k: v54.parse_scalar(json.dumps(v, sort_keys=True) if isinstance(v, (dict, list)) else ("" if v is None else str(v)))
                 for k, v in row.items()} for row in v57.canonical(manifest)]
    require(persisted == expected, "PERTURBATION_MANIFEST_CHANGED")
    v57.audit_units(parent["scientific"]["matched_units"], context)
    records, truths = {}, {}
    for dataset in dataset_ids():
        case, row = mismatch_case(context, parent, targets, dataset)
        severity, base, split, replicate, k = parts(dataset)
        require(np.isfinite(case["X_true"]).all() and (case["X_true"] >= 0).all(), "INVALID_X_TRUE")
        require(len(case["active_indices"]) == k and row["reportable_truth_count"] == k, "V57_TRUTH_GATE_CHANGED")
        if base in truths:
            require(truths[base] == row["X_true_sha256"], "MILD_MODERATE_X_TRUE_SHA_CHANGED")
        truths[base] = row["X_true_sha256"]
        records[dataset] = row
        del case
        print(f"AUDIT {dataset}", flush=True)
    require(set(records) == set(dataset_ids()) and len(truths) == 30, "INCOMPLETE_60_DATASET_AUDIT")
    audit_selected_k_nesting(records)
    validate_source(context, hashes["A_solver_sha256"])
    result = {"status": "PASS", "design_fingerprint": design["design_fingerprint"], "datasets": records,
              "target_library_hashes": hashes, "manifest_sha256": v56.digest(args.output_dir / "spectral_perturbation_manifest.csv"),
              "latent_nesting_verified": True, "CAL_HOLD_leakage": False, "K_nesting_verified": True,
              "byte_identical_to_superseded_draft_libraries": hashes == EXPECTED_DRAFT_LIBRARY_HASHES,
              "selected_K_predecessors": {"50": None, "125": 50, "175": 125},
              "domain_diagnostics_used_to_select_cases": False, "scientific_outcomes_read": False}
    path = args.output_dir / "design_audit.json"
    if design["status"] == "DESIGN_FROZEN_BEFORE_TRAINING":
        require(load_json(path) == v57.canonical(result), "FROZEN_AUDIT_RECONSTRUCTION_CHANGED")
        require(v56.digest(path) == design["frozen_audit_sha256"], "FROZEN_AUDIT_CHANGED")
    else:
        write_json(path, result)
    print("DESIGN AUDIT PASS 60/60; draft library SHAs identical; signal tails are diagnostics only")


def audit_selected_k_nesting(records):
    require(set(records) == set(dataset_ids()), "INCOMPLETE_60_DATASET_NESTING_AUDIT")
    for severity in SEVERITIES:
        for split in ("CAL", "HOLD"):
            for replicate in REPLICATES:
                for previous_k, k in zip(K_LEVELS, K_LEVELS[1:]):
                    previous = records[f"{severity}__{split}_{replicate}_K{previous_k:03d}"]
                    current = records[f"{severity}__{split}_{replicate}_K{k:03d}"]
                    require(set(previous["truth_indices"]) < set(current["truth_indices"]), "K_NESTING_CHANGED")


def checked_audit(args, design, frozen=True):
    path = args.output_dir / "design_audit.json"
    audit = load_json(path)
    require(audit["status"] == "PASS" and audit["design_fingerprint"] == design["design_fingerprint"] and
            set(audit["datasets"]) == set(dataset_ids()), "DESIGN_AUDIT_60_PASS_REQUIRED")
    require(audit["target_library_hashes"] == EXPECTED_DRAFT_LIBRARY_HASHES, "COMPACT_TARGET_LIBRARIES_CHANGED_FROM_DRAFT")
    require(audit["manifest_sha256"] == v56.digest(args.output_dir / "spectral_perturbation_manifest.csv"), "MANIFEST_MUTATED")
    if frozen:
        require(design["status"] == "DESIGN_FROZEN_BEFORE_TRAINING" and
                design["frozen_audit_sha256"] == v56.digest(path), "FROZEN_DESIGN_AUDIT_REQUIRED")
    return audit


def freeze(args, design):
    require(design["status"] == "DESIGN_PREPARED", "DESIGN_ALREADY_FROZEN_OR_INVALID")
    checked_audit(args, design, frozen=False)
    design.update(status="DESIGN_FROZEN_BEFORE_TRAINING",
                  frozen_audit_sha256=v56.digest(args.output_dir / "design_audit.json"))
    write_json(args.output_dir / "design.json", design)
    print("DESIGN_FROZEN_BEFORE_TRAINING; no oracle or training executed")


def checked_case(args, context, parent, targets, design, dataset):
    audit = checked_audit(args, design)
    case, row = mismatch_case(context, parent, targets, dataset)
    require(v57.canonical(row) == audit["datasets"][dataset], f"DATASET_RECONSTRUCTION_CHANGED: {dataset}")
    hashes = design["scientific"]["target_library_hashes"]
    severity = parts(dataset)[0]
    validate_source(context, hashes["A_solver_sha256"])
    require(array_sha(targets[severity]) == hashes[f"A_target_{severity}_sha256"], "A_TARGET_MUTATED")
    return case, row


def binding(design, dataset, row):
    severity = parts(dataset)[0]
    libraries = design["scientific"]["target_library_hashes"]
    return {"severity": severity, "dataset_id": dataset, "design_fingerprint": design["design_fingerprint"],
            "V57_parent_fingerprint": PARENT_FINGERPRINT, "A_solver_sha256": libraries["A_solver_sha256"],
            "A_target_sha256": libraries[f"A_target_{severity}_sha256"],
            "B_target_sha256": row["B_target_sha256"], "X_true_sha256": row["X_true_sha256"]}


def source_diagnostic(case, context):
    # Reuse the exact NNLS/reporting definition, but never interpret its status as validity.
    allocation = v54.reporting_gate_oracle(case, context["A_solver"], context["metadata"])
    b = case["B_sim"][:, context["mask"]].mean(axis=1).astype(np.float64)
    coefficients, residual = v54.nnls(context["A_solver"].astype(np.float64), b, maxiter=3910)
    require(np.isfinite(coefficients).all() and math.isfinite(residual), "NONFINITE_SOURCE_DIAGNOSTIC")
    molecular = {}
    for j in allocation["reported_candidate_indices"]:
        name = str(context["metadata"]["lipid_name"][j])
        molecular[name] = molecular.get(name, 0.0) + float(coefficients[j])
    return {"status": "DIAGNOSTIC_ONLY", "used_for_validity": False, "source_NNLS": allocation,
            "source_NNLS_residual": allocation["reconstruction_relative_residual"],
            "candidate_allocation": coefficients.tolist(), "reported_molecular_allocation": molecular,
            "false_identity_count": allocation["all_truth_metrics"]["FP"]}


def target_oracle_pass(report):
    target = report["target_oracle"]
    metrics = (target["candidate_level_all_truth_metrics"], target["all_truth_metrics"],
               target["candidate_level_reportable_truth_metrics"], target["reportable_truth_metrics"])
    residuals = (target["reconstruction_relative_residual"], report["target_forward_reconstruction_relative_residual"])
    return all(m["FP"] == m["FN"] == 0 for m in metrics) and all(
        math.isfinite(r) and r <= CONTRACT["oracle_max_relative_residual"] for r in residuals)


def oracle_all(args, context, parent, targets, design):
    checked_audit(args, design)
    hashes = {}
    for dataset in dataset_ids():
        case, row = checked_case(args, context, parent, targets, design, dataset)
        path = args.output_dir / "oracle" / f"{dataset}.json"
        bound = binding(design, dataset, row)
        try:
            if path.exists():
                result = load_json(path)
                require(result["runtime_binding"] == bound, "TARGET_ORACLE_PROVENANCE_CHANGED")
            else:
                target = targets[parts(dataset)[0]]
                oracle = v54.reporting_gate_oracle(case, target, context["metadata"])
                forward = (target @ case["X_true"].reshape(391, -1)).reshape(case["B_sim"].shape)
                result = {"runtime_binding": bound, "target_oracle": oracle,
                    "target_forward_reconstruction_relative_residual": relative_residual(forward, case["B_sim"]),
                    "source_library_diagnostic": source_diagnostic(case, context),
                    "forward_mismatch_residual": row["forward_mismatch_residual"],
                    "target_oracle_library": "A_target; exact full-library foreground-mean NNLS, W=I"}
                result["status"] = "PASS" if target_oracle_pass(result) else "FAIL"
                write_json(path, result)
            require(result["status"] == "PASS" and target_oracle_pass(result), f"TARGET_ORACLE_IDENTITY_FAILED: {dataset}")
            hashes[dataset] = v56.digest(path)
        except Exception as exc:
            write_json(args.output_dir / "oracle_summary.json", {"status": "FAIL", "failed_dataset": dataset,
                       "design_fingerprint": design["design_fingerprint"], "reason": str(exc), "completed": hashes})
            invalidate(args, design, f"TARGET_ORACLE_INVALID: {dataset}: {exc}")
            raise
        del case
        print(f"TARGET_ORACLE PASS {dataset}", flush=True)
    write_json(args.output_dir / "oracle_summary.json", {"status": "PASS", "count": 60,
        "design_fingerprint": design["design_fingerprint"], "oracle_sha256": hashes,
        "source_library_diagnostic_used_for_validity": False})


def require_oracles(args, design):
    audit = checked_audit(args, design)
    summary = load_json(args.output_dir / "oracle_summary.json")
    require(summary["status"] == "PASS" and summary["count"] == 60 and
            summary["design_fingerprint"] == design["design_fingerprint"] and
            set(summary["oracle_sha256"]) == set(dataset_ids()), "TARGET_ORACLE_60_OF_60_PASS_REQUIRED")
    for dataset, expected in summary["oracle_sha256"].items():
        path = args.output_dir / "oracle" / f"{dataset}.json"
        require(v56.digest(path) == expected, "TARGET_ORACLE_CHANGED")
        result = load_json(path)
        require(result["runtime_binding"] == binding(design, dataset, audit["datasets"][dataset]) and
                result["status"] == "PASS" and target_oracle_pass(result), "TARGET_ORACLE_INVALID")
    return summary


def finite_tree(value):
    if isinstance(value, dict):
        return all(finite_tree(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return all(finite_tree(v) for v in value)
    if isinstance(value, (float, int, np.number)):
        return bool(np.isfinite(value))
    return True


def finite_loss_history(history):
    # Production uses None for the first undefined relative-change diagnostic.
    # Actual loss/weight series must never treat a serialized nonfinite as missing data.
    def numbers(value):
        if isinstance(value, (list, tuple)):
            return all(numbers(v) for v in value)
        return value is not None and isinstance(value, (int, float, np.number)) and bool(np.isfinite(value))

    fields = ("train_total", "eval_total", "eval_physical_loss", "eval_raw_losses", "automatic_loss_weights")
    epochs = history.get("epoch", [])
    return bool(epochs) and finite_tree(history) and all(
        field in history and len(history[field]) == len(epochs) and numbers(history[field]) for field in fields)


def solver_validity(learned):
    normal = learned["stop_reason"] in ("converged", "max_epochs")
    finite = bool(np.isfinite(learned["X_hat"]).all() and np.isfinite(learned["B_hat"]).all() and
                  learned["final_physical_loss"] is not None and np.isfinite(learned["final_physical_loss"]) and
                  len(learned["final_raw_losses"]) == 4 and
                  all(v is not None and np.isfinite(v) for v in learned["final_raw_losses"]))
    return normal, finite


def source_training_context(context):
    # Whitelist: no A_target, X_true, oracle, or truth-derived gate can cross this boundary.
    return {key: context[key] for key in ("A_raw", "A_solver", "metadata", "paths")}


def verify_solver_run(args, design, dataset, bound, require_arrays=False):
    output = directory(args, dataset)
    require(load_json(output / "runtime_contract.json") == bound, "RUNTIME_CONTRACT_CHANGED")
    saved = load_json(output / "solver_run.json")
    require(saved["runtime_binding"] == bound, "SOLVER_RUN_BINDING_CHANGED")
    for name, sha in saved["training_artifact_hashes"].items():
        require(v56.digest(output / name) == sha, f"TRAINING_ARTIFACT_CHANGED: {name}")
    if require_arrays:
        require(v56.digest(output / "learned_arrays.npz") == saved["arrays_sha256"], "LEARNED_ARRAYS_CHANGED")
    return saved


def train_or_resume(args, context, design, dataset, case, row):
    from config_758 import Cfg
    for key, value in {**v54.v50.CFG_DEFAULTS, "parent_channel_weight_multiplier": 1.0}.items():
        require(getattr(Cfg, key) == value, f"PRODUCTION_CONFIG_CHANGED: {key}")
    require(v54.MAX_EPOCH == CONTRACT["hard_epoch_cap"] and
            tuple(v54.CHECKPOINT_EPOCHS) == CONTRACT["checkpoint_epochs"], "PRODUCTION_TRAINING_CAP_CHANGED")
    bound, output = binding(design, dataset, row), directory(args, dataset)
    validate_source(context, bound["A_solver_sha256"])
    require(array_sha(case["B_sim"]) == bound["B_target_sha256"] and
            array_sha(case["X_true"]) == bound["X_true_sha256"], "RUNTIME_INPUT_CHANGED")
    output.mkdir(parents=True, exist_ok=True)
    path = output / "runtime_contract.json"
    if path.exists():
        require(load_json(path) == bound, "RUNTIME_CONTRACT_CHANGED_WHEN_RESUMING")
    else:
        require(not list(output.glob("*.pth")) and not (output / "learned_arrays.npz").exists() and
                not (output / "solver_run.json").exists(), "UNBOUND_CHECKPOINT_OR_LEARNED_ARRAYS")
        write_json(path, bound)
    if (output / "solver_run.json").exists():
        saved = verify_solver_run(args, design, dataset, bound, require_arrays=True)
        with np.load(output / "learned_arrays.npz") as data:
            learned = {**saved["training"], "X_hat": data["X_hat"].copy(), "B_hat": data["B_hat"].copy()}
    else:
        training_case = {key: case[key] for key in ("case_name", "B_sim", "foreground_mask")}
        learned = v54.train_resumable(training_case, source_training_context(context), output, device_name=args.device)
        # Do not normalize/reconstruct B_hat or alter production early stopping.
        normal, finite = solver_validity(learned)
        require(normal and finite, f"SOLVER_NONFINITE_OR_ABNORMAL: {dataset}: {learned['stop_reason']}")
        history = load_json(output / "training_history.json")
        require(finite_loss_history(history), "NONFINITE_TRAINING_HISTORY")
        temporary = output / "learned_arrays.tmp.npz"
        np.savez_compressed(temporary, X_hat=learned["X_hat"], B_hat=learned["B_hat"])
        temporary.replace(output / "learned_arrays.npz")
        training = {key: value for key, value in learned.items() if key not in ("X_hat", "B_hat")}
        # Production may converge before its first checkpoint; keep an explicit empty diagnostics table.
        diagnostics = output / "checkpoint_diagnostics.csv"
        if not diagnostics.exists():
            write_rows(diagnostics, [], v54.checkpoint_fields())
        write_json(output / "solver_run.json", {"runtime_binding": bound, "training": training,
            "arrays_sha256": v56.digest(output / "learned_arrays.npz"),
            "training_artifact_hashes": {name: v56.digest(output / name) for name in
                                         ("training_history.json", "checkpoint_diagnostics.csv")},
            "X_true_used_in_training_or_stopping": False, "A_target_used_by_solver": False})
    require(all(solver_validity(learned)) and finite_loss_history(load_json(output / "training_history.json")),
            "CACHED_SOLVER_NONFINITE_OR_ABNORMAL")
    require(learned["X_hat"].shape == case["X_true"].shape and learned["B_hat"].shape == case["B_sim"].shape,
            "SOLVER_OUTPUT_SHAPE_CHANGED")
    validate_source(context, bound["A_solver_sha256"])
    return learned


def sentinel(args, context, parent, targets, design):
    require_oracles(args, design)
    path = args.output_dir / "sentinel_solver_validity.json"
    if path.exists():
        previous = load_json(path)
        require(previous.get("evidence_mode") != ADOPTED_SENTINEL_MODE,
                "ADOPTED_SENTINEL_EVIDENCE_ALREADY_EXISTS; do not overwrite it with native evidence")
        require(previous["design_fingerprint"] == design["design_fingerprint"] and previous["status"] != "FAIL",
                "SENTINEL_FAILED_OR_CHANGED; no outcome-driven redesign")
    results = {}

    def save(status):
        write_json(path, {"status": status, "design_fingerprint": design["design_fingerprint"], "datasets": results,
            "evidence_mode": NATIVE_SENTINEL_MODE, "learned_training_reexecuted": True,
            "gate_formula": GATE_FORMULA, "gate_frozen_audit_sha256": design["frozen_audit_sha256"],
            "residual_scope": "full observed cube, all channels/pixels; no channel weighting",
            "early_stop_allowed": True, "hard_cap": 3000, "identity_outcomes_used": False,
            "rho_computed": False, "truth_used_in_training_or_stopping": False})

    for dataset in SENTINELS:
        results[dataset] = {"status": "IN_PROGRESS", "failure_reason": "RUN_NOT_COMPLETED"}
        save("IN_PROGRESS")
        try:
            case, row = checked_case(args, context, parent, targets, design, dataset)
            learned = train_or_resume(args, context, design, dataset, case, row)
            normal, finite = solver_validity(learned)
            residual = relative_residual(learned["B_hat"], case["B_sim"])
            gate = row["sentinel_gate_residual"]
            passed = normal and finite and math.isfinite(residual) and residual <= gate
            results[dataset] = {"status": "PASS" if passed else "FAIL", "runtime_binding": binding(design, dataset, row),
                "stop_reason": learned["stop_reason"], "epoch": learned["stopped_epoch"],
                "normal_completion": normal, "all_outputs_losses_finite": finite,
                "forward_mismatch_residual": row["forward_mismatch_residual"], "gate_residual": gate,
                "final_full_cube_reconstruction_relative_residual": residual if math.isfinite(residual) else None,
                "solver_run_sha256": v56.digest(directory(args, dataset) / "solver_run.json")}
            save("IN_PROGRESS" if passed else "FAIL")
            require(passed, f"SENTINEL_COLLAPSE_GATE_FAILED: {dataset}")
        except BaseException as exc:
            results[dataset].update(status="FAIL", failure_reason=f"{type(exc).__name__}: {exc}")
            save("FAIL")
            invalidate(args, design, f"SENTINEL_FAILED: {dataset}: {exc}")
            raise
        del case, learned
        print(f"SENTINEL PASS {dataset}", flush=True)
    save("PASS")


def require_sentinels(args, design, context=None, parent=None, targets=None):
    require_oracles(args, design)
    audit = checked_audit(args, design)
    gate = load_json(args.output_dir / "sentinel_solver_validity.json")
    if gate.get("evidence_mode") == ADOPTED_SENTINEL_MODE:
        context, parent, targets = adoption_inputs(args, design, context, parent, targets)
        verified = equivalent_sentinel_evidence(args, context, parent, targets, design,
                                                Path(gate["source_draft_directory"]))
        require(gate == v57.canonical(verified), "ADOPTED_SENTINEL_EVIDENCE_OR_SOURCE_ARTIFACTS_CHANGED")
        return gate
    require(gate.get("evidence_mode", NATIVE_SENTINEL_MODE) == NATIVE_SENTINEL_MODE, "UNKNOWN_SENTINEL_EVIDENCE_MODE")
    require(gate["status"] == "PASS" and gate["design_fingerprint"] == design["design_fingerprint"] and
            set(gate["datasets"]) == set(SENTINELS) and gate["gate_formula"] == GATE_FORMULA and
            gate["gate_frozen_audit_sha256"] == design["frozen_audit_sha256"], "SENTINEL_4_OF_4_PASS_REQUIRED")
    for dataset, result in gate["datasets"].items():
        row = audit["datasets"][dataset]
        bound = binding(design, dataset, row)
        residual = result["final_full_cube_reconstruction_relative_residual"]
        require(result["status"] == "PASS" and result["normal_completion"] and result["all_outputs_losses_finite"] and
                result["stop_reason"] in ("converged", "max_epochs") and residual is not None and math.isfinite(residual) and
                residual <= result["gate_residual"] == row["sentinel_gate_residual"] == collapse_gate(row["forward_mismatch_residual"]) and
                result["runtime_binding"] == bound, "SENTINEL_VALIDITY_CHANGED")
        require(v56.digest(directory(args, dataset) / "solver_run.json") == result["solver_run_sha256"], "SENTINEL_SOLVER_CHANGED")
        verify_solver_run(args, design, dataset, bound)
    return gate


def adoption_inputs(args, design, context=None, parent=None, targets=None):
    if context is None or parent is None or targets is None:
        parent, _, hashes = parent_provenance(args)
        require(hashes == design["scientific"]["V57_parent_file_hashes"], "ADOPTION_PARENT_CHANGED")
        context = load_context(args, parent)
        targets, _, _ = build_targets(context)
    return context, parent, targets


def equivalent_sentinel_evidence(args, context, parent, targets, design, source):
    """Read/reconstruct only; never train, copy artifacts, or rebind source outputs."""
    source = source.resolve()
    output = args.output_dir.resolve()
    require(source != output and source not in output.parents and output not in source.parents,
            "ADOPTION_SOURCE_OVERLAPS_COMPACT_OUTPUT")
    require(design["design_fingerprint"] == v57.fingerprint(design["scientific"]), "COMPACT_DESIGN_CHANGED")
    require(design["scientific"]["contract"] == v57.canonical(CONTRACT), "COMPACT_CONTRACT_CHANGED")
    require_oracles(args, design)
    files = ("design.json", "design_audit.json", "target_library_hashes.json", "sentinel_solver_validity.json")
    source_hashes = {name: v56.digest(source / name) for name in files}
    old = load_json(source / "design.json")
    old_fp = SUPERSEDED_DRAFT["superseded_draft_fingerprint"]
    require(old["status"] == "DESIGN_FROZEN_BEFORE_TRAINING" and
            old["design_fingerprint"] == v57.fingerprint(old["scientific"]) == old_fp, "ADOPTION_SOURCE_DESIGN_CHANGED")
    require(old["frozen_audit_sha256"] == source_hashes["design_audit.json"], "ADOPTION_SOURCE_FROZEN_AUDIT_CHANGED")
    old_audit = load_json(source / "design_audit.json")
    require(old_audit["status"] == "PASS" and old_audit["design_fingerprint"] == old_fp, "ADOPTION_SOURCE_AUDIT_INVALID")
    old_gate = load_json(source / "sentinel_solver_validity.json")
    require(old_gate["status"] == "PASS" and old_gate["design_fingerprint"] == old_fp and
            set(old_gate["datasets"]) == set(SENTINELS) and
            old_gate.get("evidence_mode", NATIVE_SENTINEL_MODE) != ADOPTED_SENTINEL_MODE,
            "ADOPTION_REQUIRES_EXACT_FOUR_NATIVE_DRAFT_SENTINELS_PASS")
    require(old_gate["gate_frozen_audit_sha256"] == old["frozen_audit_sha256"] and
            old_gate["gate_formula"] == GATE_FORMULA and old_gate["hard_cap"] == CONTRACT["hard_epoch_cap"] and
            old_gate["early_stop_allowed"] is True and old_gate["identity_outcomes_used"] is False and
            old_gate["rho_computed"] is False and old_gate["truth_used_in_training_or_stopping"] is False,
            "ADOPTION_SOURCE_GATE_CONTRACT_CHANGED")
    libraries = design["scientific"]["target_library_hashes"]
    require(libraries == old["scientific"]["target_library_hashes"] ==
            load_json(source / "target_library_hashes.json") == old_audit["target_library_hashes"] ==
            EXPECTED_DRAFT_LIBRARY_HASHES, "ADOPTION_LIBRARY_HASH_MISMATCH")
    # The known draft fingerprint anchors the original config and driver guards.
    # All scientific contract fields it contains must agree, apart from the K/count contraction.
    for key, value in old["scientific"]["contract"].items():
        if key not in ("K_levels", "mismatch_runs"):
            require(v57.canonical(CONTRACT.get(key)) == value, f"ADOPTION_CONTRACT_MISMATCH: {key}")
    for key in ("production_asset_hashes", "stage0_hashes", "candidate_order", "V57_parent_file_hashes"):
        require(old["scientific"][key] == design["scientific"][key], f"ADOPTION_INPUT_PROVENANCE_MISMATCH: {key}")
    current_hashes = implementation_hashes()
    runner = Path(__file__).name
    old_implementations = {k: h for k, h in old["scientific"]["source_implementation_hashes"].items() if k != runner}
    for implementations in (design["scientific"]["source_implementation_hashes"], current_hashes):
        require(old_implementations == {k: h for k, h in implementations.items() if k != runner},
                "ADOPTION_PRODUCTION_IMPLEMENTATION_MISMATCH")
    require(design["scientific"]["source_implementation_hashes"][runner] in
            (PRE_ADOPTION_COMPACT_RUNNER_SHA256, current_hashes[runner]), "ADOPTION_UNSUPPORTED_COMPACT_RUNNER")
    from config_758 import Cfg
    config = {**v54.v50.CFG_DEFAULTS, "parent_channel_weight_multiplier": 1.0}
    require(all(getattr(Cfg, k) == value for k, value in config.items()), "ADOPTION_PRODUCTION_CONFIG_CHANGED")
    require(v54.MAX_EPOCH == CONTRACT["hard_epoch_cap"] and tuple(v54.CHECKPOINT_EPOCHS) == CONTRACT["checkpoint_epochs"],
            "ADOPTION_TRAINING_DURATION_CONTRACT_CHANGED")
    datasets, checks, artifacts = {}, {}, {}
    for dataset in SENTINELS:
        case, row = checked_case(args, context, parent, targets, design, dataset)
        compact_binding = binding(design, dataset, row)
        original = old_gate["datasets"][dataset]
        old_binding = original["runtime_binding"]
        # Compare the common scientific fields; keep the two fingerprints separate.
        require(old_binding["design_fingerprint"] == old_fp and
                {k: v for k, v in old_binding.items() if k != "design_fingerprint"} ==
                {k: v for k, v in compact_binding.items() if k != "design_fingerprint"}, "ADOPTION_DATASET_INPUT_MISMATCH")
        require(old_binding == binding(old, dataset, old_audit["datasets"][dataset]), "ADOPTION_OLD_RUNTIME_AUDIT_MISMATCH")
        forward = original["forward_mismatch_residual"]
        require(math.isfinite(forward) and math.isclose(forward, row["forward_mismatch_residual"],
                rel_tol=EQUIVALENCE_RTOL, abs_tol=EQUIVALENCE_ATOL), "ADOPTION_FORWARD_MISMATCH_CHANGED")
        require(math.isclose(forward, old_audit["datasets"][dataset]["forward_mismatch_residual"],
                rel_tol=EQUIVALENCE_RTOL, abs_tol=EQUIVALENCE_ATOL), "ADOPTION_OLD_FORWARD_AUDIT_CHANGED")
        require(original["gate_residual"] == row["sentinel_gate_residual"] == collapse_gate(row["forward_mismatch_residual"]) ==
                collapse_gate(forward) == old_audit["datasets"][dataset]["sentinel_gate_residual"], "ADOPTION_GATE_RESIDUAL_CHANGED")
        severity, base, *_ = parts(dataset)
        location = source / severity / base
        require(v56.digest(location / "solver_run.json") == original["solver_run_sha256"], "ADOPTION_SOURCE_SOLVER_RUN_CHANGED")
        saved = load_json(location / "solver_run.json")
        require(saved["runtime_binding"] == load_json(location / "runtime_contract.json") == old_binding,
                "ADOPTION_SOURCE_RUNTIME_BINDING_CHANGED")
        require(saved["X_true_used_in_training_or_stopping"] is False and saved["A_target_used_by_solver"] is False,
                "ADOPTION_SOURCE_TRAINING_LEAKAGE")
        require(set(saved["training_artifact_hashes"]) == {"training_history.json", "checkpoint_diagnostics.csv"},
                "ADOPTION_INCOMPLETE_TRAINING_ARTIFACT_HASHES")
        for name, sha in saved["training_artifact_hashes"].items():
            require(v56.digest(location / name) == sha, f"ADOPTION_SOURCE_ARTIFACT_CHANGED: {dataset}/{name}")
        require(v56.digest(location / "learned_arrays.npz") == saved["arrays_sha256"], "ADOPTION_SOURCE_ARRAYS_CHANGED")
        require(finite_loss_history(load_json(location / "training_history.json")), "ADOPTION_SOURCE_NONFINITE_HISTORY")
        with np.load(location / "learned_arrays.npz", allow_pickle=False) as arrays:
            learned = {**saved["training"], "X_hat": arrays["X_hat"], "B_hat": arrays["B_hat"]}
            require(all(solver_validity(learned)) and learned["X_hat"].shape == case["X_true"].shape and
                    learned["B_hat"].shape == case["B_sim"].shape, "ADOPTION_SOURCE_SOLVER_NONFINITE_OR_ABNORMAL")
            residual = relative_residual(learned["B_hat"], case["B_sim"])
        recorded_residual = original["final_full_cube_reconstruction_relative_residual"]
        require(original["status"] == "PASS" and original["normal_completion"] is True and
                original["all_outputs_losses_finite"] is True and original["stop_reason"] == saved["training"]["stop_reason"] and
                original["epoch"] == saved["training"]["stopped_epoch"] and
                1 <= original["epoch"] <= CONTRACT["hard_epoch_cap"] and
                saved["training"]["scheduler_T_max"] == config["n_epochs"] and
                recorded_residual is not None and math.isfinite(recorded_residual) and
                math.isclose(residual, recorded_residual, rel_tol=EQUIVALENCE_RTOL, abs_tol=EQUIVALENCE_ATOL) and
                max(residual, recorded_residual) <= row["sentinel_gate_residual"], "ADOPTION_SOURCE_SENTINEL_VALIDITY_FAILED")
        checkpoint_names = {"latest_model.pth", *[f"checkpoint_epoch_{epoch}.pth" for epoch in CONTRACT["checkpoint_epochs"]
                                                if epoch <= original["epoch"]]}
        require({p.name for p in location.glob("*.pth")} == checkpoint_names, "ADOPTION_CHECKPOINT_SET_INCOMPLETE_OR_CHANGED")
        names = {*checkpoint_names, "runtime_contract.json", "solver_run.json", "learned_arrays.npz",
                 "training_history.json", "checkpoint_diagnostics.csv"}
        artifacts[dataset] = {name: v56.digest(location / name) for name in sorted(names)}
        require(artifacts[dataset]["solver_run.json"] == original["solver_run_sha256"] and
                artifacts[dataset]["learned_arrays.npz"] == saved["arrays_sha256"] and
                all(artifacts[dataset][name] == sha for name, sha in saved["training_artifact_hashes"].items()),
                "ADOPTION_SOURCE_ARTIFACT_CHANGED_DURING_READ")
        # Source evidence is embedded intact: its runtime_binding remains the old fingerprint.
        datasets[dataset] = original
        checks[dataset] = {"scientific_inputs_identical": True, "compact_runtime_binding": compact_binding,
            "source_runtime_binding": old_binding, "reconstructed_forward_mismatch_residual": row["forward_mismatch_residual"],
            "reconstructed_final_residual": residual, "compact_gate_residual": row["sentinel_gate_residual"],
            "normal_completion": True, "all_outputs_losses_finite": True, "artifact_hashes_verified": True}
        del case, learned
    require(all(v56.digest(source / name) == sha for name, sha in source_hashes.items()), "ADOPTION_SOURCE_CHANGED_DURING_READ")
    for dataset, file_hashes in artifacts.items():
        severity, base, *_ = parts(dataset)
        require(all(v56.digest(source / severity / base / name) == sha for name, sha in file_hashes.items()),
                "ADOPTION_SOURCE_ARTIFACT_CHANGED_DURING_READ")
    return {"status": "PASS", "design_fingerprint": design["design_fingerprint"], "datasets": datasets,
        "evidence_mode": ADOPTED_SENTINEL_MODE, "learned_training_reexecuted": False,
        "source_draft_directory": str(source), "source_draft_fingerprint": old_fp,
        "source_sentinel_file_sha256": source_hashes["sentinel_solver_validity.json"], "source_file_hashes": source_hashes,
        "source_solver_artifact_hashes": artifacts, "equivalence_checks": checks, "scientific_inputs_identical": True,
        "production_implementation_hashes": old_implementations, "production_training_config": config,
        "training_config_evidence": "identical frozen config/driver hashes and production config guards; scheduler_T_max verified",
        "checkpoint_hash_provenance": "weights first SHA-pinned at adoption; source runner already pinned history, diagnostics and arrays",
        "compact_frozen_runner_sha256": design["scientific"]["source_implementation_hashes"][runner],
        "adoption_runner_sha256": current_hashes[runner], "float_equivalence_tolerance": {"rtol": EQUIVALENCE_RTOL, "atol": EQUIVALENCE_ATOL},
        "gate_formula": GATE_FORMULA, "gate_frozen_audit_sha256": design["frozen_audit_sha256"],
        "early_stop_allowed": True, "hard_cap": CONTRACT["hard_epoch_cap"], "checkpoint_epochs": CONTRACT["checkpoint_epochs"],
        "identity_outcomes_used": False, "rho_computed": False, "truth_used_in_training_or_stopping": False,
        "ordinary_mismatch_results_adopted": False}


def adopt_equivalent_sentinel(args, context, parent, targets, design):
    guard_output(args)
    evidence = equivalent_sentinel_evidence(args, context, parent, targets, design, args.source_draft_dir)
    path = args.output_dir / "sentinel_solver_validity.json"
    if path.exists():
        require(load_json(path) == v57.canonical(evidence), "REFUSE_TO_REPLACE_EXISTING_SENTINEL_EVIDENCE")
    else:
        write_json(path, evidence)
    print("ADOPTED_EQUIVALENT_SUPERSEDED_SENTINEL PASS 4/4; source artifacts and runtime bindings unchanged; no training")


RECORD_FILES = ("reported_identity_records.csv", "candidate_false_negative_records.csv", "molecular_false_negative_records.csv")


def verified_report(args, design, dataset, row):
    output = directory(args, dataset)
    report = load_json(output / "report.json")
    require(report["status"] == "COMPLETE" and report["runtime_binding"] == binding(design, dataset, row), "DATASET_REPORT_PROVENANCE_CHANGED")
    for name, sha in report["artifact_hashes"].items():
        require(v56.digest(output / name) == sha, f"DATASET_ARTIFACT_CHANGED: {dataset}/{name}")
    require(set(report["artifact_hashes"]) == set((*RECORD_FILES, "solver_run.json", "runtime_contract.json")), "RESULT_ARTIFACT_SET_CHANGED")
    return report


def run_dataset(args, context, parent, targets, design, dataset):
    require_sentinels(args, design, context, parent, targets)
    case, row = checked_case(args, context, parent, targets, design, dataset)
    output = directory(args, dataset)
    if (output / "report.json").exists():
        return verified_report(args, design, dataset, row)
    try:
        learned = train_or_resume(args, context, design, dataset, case, row)
    except BaseException as exc:
        write_json(output / "solver_failure.json", {"status": "FAIL", "runtime_binding": binding(design, dataset, row),
                                                    "reason": f"{type(exc).__name__}: {exc}"})
        invalidate(args, design, f"SOLVER_PROCESS_FAILED: {dataset}: {exc}")
        raise
    severity, base, split, replicate, k = parts(dataset)
    # V54 parses the base ID; rewrite record IDs only after its frozen rho calculation.
    records, raw = v54.learned_records(base, case, learned, source_training_context(context), args.rho_workers)
    metadata = v57.mechanism_metadata(parent["scientific"], base, context)
    for record in records:
        require(math.isfinite(record["rho_zero"]) and record["rho_zero"] >= 0, "NONFINITE_RHO_OUTPUT")
        record.update(metadata[record["candidate_index"]])
        record.update(dataset_id=dataset, base_dataset_id=base, severity=severity, rho_zero_status="COMPUTED",
                      raw_solver_reported=True, reportable_truth=record["lipid_name"] in raw["reportable_truth_lipid_names"])
    reported = {r["candidate_index"]: r for r in records}
    xhat = learned["X_hat"][:, context["mask"]].mean(axis=1, dtype=np.float64)
    candidates = [{"dataset_id": dataset, "base_dataset_id": base, "severity": severity,
        "candidate_index": j, "candidate_id": str(context["metadata"]["candidate_id"][j]), "lipid_name": str(name),
        "X_hat": float(xhat[j]), "rho_zero": reported[j]["rho_zero"] if j in reported else None,
        "rho_zero_status": "COMPUTED" if j in reported else "NOT_COMPUTED", "raw_solver_reported": j in reported,
        "candidate_truth": j in case["active_indices"], "molecular_truth": str(name) in raw["truth_lipid_names"],
        "reportable_truth": str(name) in raw["reportable_truth_lipid_names"], **metadata[j]}
        for j, name in enumerate(context["metadata"]["lipid_name"])]
    oracle = load_json(args.output_dir / "oracle" / f"{dataset}.json")
    report = {"status": "COMPLETE", "dataset_id": dataset, "condition": severity, "severity": severity,
        "split": split, "replicate": replicate, "K": k, "design_fingerprint": design["design_fingerprint"],
        **binding(design, dataset, row), "runtime_binding": binding(design, dataset, row),
        "forward_mismatch_residual": row["forward_mismatch_residual"], "B_domain": row["B_domain"],
        "mismatch_B_p50_over_V57_clean_B_p50": row["mismatch_B_p50_over_V57_clean_B_p50"],
        "target_oracle_diagnostic": oracle["target_oracle"], "source_library_diagnostic": oracle["source_library_diagnostic"],
        "learned_solver_reconstruction_relative_residual": relative_residual(learned["B_hat"], case["B_sim"]),
        "global_truth_K": k, "all_truth_molecular_identity_count": case["all_truth_molecular_identity_count"],
        "reportable_truth_molecular_identity_count": case["reportable_truth_molecular_identity_count"],
        "learned_raw_identity_performance": raw, "rho_zero_weighting": "W=I",
        "training": {key: value for key, value in learned.items() if key not in ("X_hat", "B_hat")},
        "X_true_used_in_training_or_stopping": False, "A_target_used_by_solver": False,
        "A_target_used_by_rho": False, "post_mismatch_rescaling": False}
    units = v54.molecular_units(records, {dataset: report})
    raw["molecular_level"].update(v56.metrics(units, {dataset: report}))
    write_rows(output / RECORD_FILES[0], records, None if records else ["dataset_id", "lipid_name", "rho_zero", "X_hat"])
    write_rows(output / RECORD_FILES[1], candidates)
    write_rows(output / RECORD_FILES[2], v57.molecular_all_records(candidates, units))
    report["artifact_hashes"] = {name: v56.digest(output / name) for name in (*RECORD_FILES, "solver_run.json", "runtime_contract.json")}
    write_json(output / "report.json", report)
    print(f"COMPLETE {dataset}", flush=True)
    return report


def normalize_records(records):
    for record in records:
        if record.get("rho_zero") == "":
            record["rho_zero"] = None
        if record["raw_solver_reported"]:
            require(record["rho_zero"] is not None and math.isfinite(record["rho_zero"]) and
                    record["rho_zero"] >= 0 and math.isfinite(record["X_hat"]), "INVALID_REPORTED_SCORE")
        else:
            require(record["rho_zero"] is None and record["rho_zero_status"] == "NOT_COMPUTED",
                    "UNREPORTED_RHO_MUST_BE_NONE")
    return records


def threshold_path(args, severity):
    require(severity in SEVERITIES, "INVALID_SEVERITY")
    return args.output_dir / severity / "local_frozen_thresholds.json"


def require_local_freeze(args, design, severity):
    path = threshold_path(args, severity)
    seal = load_json(path.with_name("local_threshold_freeze.json"))
    require(seal["design_fingerprint"] == design["design_fingerprint"] and seal["severity"] == severity and
            v56.digest(path) == seal["threshold_file_sha256"], "LOCAL_THRESHOLD_HASH_CHANGED_OR_NOT_FROZEN")
    frozen = load_json(path)
    require(frozen["status"] == "FROZEN_FROM_CAL_ONLY" and frozen["design_fingerprint"] == design["design_fingerprint"] and
            frozen["severity"] == severity and frozen["selection"] == SELECTION and
            set(frozen["CAL_source_hashes"]) == {f"{severity}__{d}" for d in base_ids("CAL")}, "LOCAL_CAL_PROVENANCE_CHANGED")
    # Recheck immutable CAL bytes, never open HOLD records while calibrating.
    for dataset, files in frozen["CAL_source_hashes"].items():
        for name, sha in files.items():
            require(v56.digest(directory(args, dataset) / name) == sha, "LOCAL_CAL_SOURCE_CHANGED_AFTER_FREEZE")
    return frozen


def hold_read_guard(args, design):
    # Stronger than per-severity ordering: freeze both target domains before either HOLD read.
    return {severity: require_local_freeze(args, design, severity) for severity in SEVERITIES}


def read_completed(args, design, severity, split):
    require(split in ("CAL", "HOLD") and severity in SEVERITIES, "INVALID_RESULT_SCOPE")
    if split == "HOLD":
        hold_read_guard(args, design)
    audit = checked_audit(args, design)
    reports, hashes, records = {}, {}, []
    for base in base_ids(split):
        dataset = f"{severity}__{base}"
        report = verified_report(args, design, dataset, audit["datasets"][dataset])
        require(report["split"] == split and report["severity"] == severity and report["dataset_id"] == dataset, "RESULT_SCOPE_CHANGED")
        output = directory(args, dataset)
        rows = normalize_records(v57.rows(output / "molecular_false_negative_records.csv"))
        require(all(r["dataset_id"] == dataset and r["split"] == split and r["severity"] == severity for r in rows), "RECORD_SCOPE_CHANGED")
        require(len({r["lipid_name"] for r in rows}) == len(rows), "DUPLICATE_MOLECULAR_UNITS")
        require(sum(r["molecular_truth"] for r in rows) == report["all_truth_molecular_identity_count"], "TRUTH_RECORDS_MISSING")
        records.extend(rows)
        reports[dataset] = report
        hashes[dataset] = {name: v56.digest(output / name) for name in ("report.json", *RECORD_FILES)}
    return records, reports, hashes


def freeze_local_thresholds(args, design, severity, records, reports, hashes):
    require(all(r["split"] == "CAL" and r["severity"] == severity for r in records) and
            set(reports) == {f"{severity}__{d}" for d in base_ids("CAL")}, "ONLY_15_SAME_SEVERITY_CAL_DATASETS_ALLOWED")
    require(set(hashes) == set(reports) and all(set(files) == set(("report.json", *RECORD_FILES))
                                             for files in hashes.values()), "INCOMPLETE_CAL_SOURCE_HASHES")
    units = [r for r in records if r["raw_solver_reported"]]
    pack, curves = v56.threshold_pack(units, reports)
    frozen = v57.canonical({"status": "FROZEN_FROM_CAL_ONLY", "severity": severity,
        "design_fingerprint": design["design_fingerprint"], "CAL_source_hashes": hashes,
        "target_FDR": CONTRACT["target_FDR"], "global_thresholds": pack, "selection": SELECTION,
        "no_eligible_threshold": "None; explicitly reject all, report FDR undefined, scientific calibration failure only",
        "HOLD_read_before_freeze": False, "per_K_or_replicate_calibration": False})
    path = threshold_path(args, severity)
    if path.exists():
        require(load_json(path) == frozen, "FROZEN_LOCAL_CAL_THRESHOLDS_CHANGED")
    else:
        write_json(path, frozen)
    seal = {"design_fingerprint": design["design_fingerprint"], "severity": severity,
            "threshold_file_sha256": v56.digest(path)}
    seal_path = path.with_name("local_threshold_freeze.json")
    if seal_path.exists():
        require(load_json(seal_path) == seal, "LOCAL_THRESHOLD_SEAL_CHANGED")
    else:
        write_json(seal_path, seal)
    for score, label in (("rho_zero", "rho"), ("X_hat", "xhat")):
        write_rows(path.parent / f"calibration_curve_{label}.csv",
                   [{"severity": severity, "score": score, **r} for r in curves[score]],
                   None if curves[score] else ["severity", "score", "threshold", "N_retained", "TP", "FP", "FDR", "precision",
                       "all_truth_recall", "reportable_truth_recall", "TP_retention", "filter_induced_true_loss", "coverage"])
    return require_local_freeze(args, design, severity)


def subset(records, reports, k=None, replicate=None):
    selected = {d: r for d, r in reports.items() if (k is None or r["K"] == k) and
                (replicate is None or r["replicate"] == replicate)}
    return [r for r in records if r["dataset_id"] in selected], selected


def protocol_metrics(severity, protocol, records, reports, pack):
    units = [r for r in records if r["raw_solver_reported"]]
    overall, by_k, by_r = [], [], []
    for score, entries in pack.items():
        for target, entry in entries.items():
            threshold = entry["threshold"] if entry else None
            common = {"severity": severity, "protocol": protocol, "score": score, "target": target,
                "threshold": threshold, "threshold_available": entry is not None,
                "calibration_status": "AVAILABLE" if entry else "NO_ELIGIBLE_CAL_THRESHOLD",
                "primary_rho_result": score == "rho_zero"}
            overall.append({**common, **v56.metrics(units, reports, score, threshold)})
            for k in K_LEVELS:
                us, rs = subset(units, reports, k=k)
                by_k.append({**common, "K": k, **v56.metrics(us, rs, score, threshold)})
            for replicate in REPLICATES:
                us, rs = subset(units, reports, replicate=replicate)
                by_r.append({**common, "replicate": replicate, **v56.metrics(us, rs, score, threshold)})
    return overall, by_k, by_r


def separation_summary(records, domain, split, k=None):
    units = [r for r in records if r["raw_solver_reported"] and (k is None or r["K"] == k)]
    true = np.array([r["rho_zero"] for r in units if r["molecular_truth"]], dtype=float)
    false = np.array([r["rho_zero"] for r in units if not r["molecular_truth"]], dtype=float)
    row = {"domain": domain, "split": split, "K": k if k is not None else "GLOBAL", "true_n": len(true), "false_n": len(false)}
    for label, values, quantiles in (("true", true, (("q01", .01), ("q05", .05), ("median", .5), ("q95", .95))),
                                     ("false", false, (("median", .5), ("q95", .95), ("q99", .99)))):
        row.update({f"{label}_min": float(values.min()) if len(values) else None,
                    f"{label}_max": float(values.max()) if len(values) else None})
        row.update({f"{label}_{name}": float(np.quantile(values, q)) if len(values) else None for name, q in quantiles})
    row.update(false_exact_zero_count=int((false == 0).sum()),
               false_exact_zero_fraction=float((false == 0).mean()) if len(false) else None)
    low, high = row["true_q05"], row["false_q95"]
    if low is None or high is None:
        gap, definition = None, "UNDEFINED_MISSING_CLASS"
    elif low == 0 and high == 0:
        gap, definition = None, "UNDEFINED_BOTH_QUANTILES_ZERO"
    elif high == 0:
        gap, definition = "INF", "positive true q05 / exactly zero false q95"
    elif low == 0:
        gap, definition = "-INF", "exactly zero true q05 / positive false q95"
    else:
        gap, definition = math.log10(low) - math.log10(high), "log10(q05_true)-log10(q95_false); no epsilon"
    row.update(log_gap_q05_q95=gap, log_gap_definition=definition, AUROC_rho=None, AUPRC=None,
               AUPRC_definition="average precision; tied score groups retained together")
    if len(true) and len(false):
        from scipy.stats import rankdata
        ranks = rankdata(np.concatenate((true, false)), method="average")
        row["AUROC_rho"] = float((ranks[:len(true)].sum() - len(true) * (len(true) + 1) / 2) / (len(true) * len(false)))
    if len(true):
        grouped = {}
        for u in units:
            counts = grouped.setdefault(u["rho_zero"], [0, 0])
            counts[0] += int(u["molecular_truth"])
            counts[1] += 1
        tp, count, ap = 0, 0, 0.0
        for score in sorted(grouped, reverse=True):
            positives, size = grouped[score]
            tp, count = tp + positives, count + size
            ap += positives / len(true) * tp / count
        row["AUPRC"] = ap
    return row


def descriptive_curves(records, reports, domain, split):
    units = [r for r in records if r["raw_solver_reported"]]
    points = []
    for k in (None, *K_LEVELS):
        us, rs = subset(units, reports, k=k)
        positives = sum(r["molecular_truth"] for r in us)
        negatives = len(us) - positives
        for score in ("rho_zero", "X_hat"):
            points.extend({"domain": domain, "split": split, "K": k if k is not None else "GLOBAL", "score": score,
                "role": "DESCRIPTIVE_ONLY_NO_HOLD_THRESHOLD_SELECTION", **r,
                "ROC_TPR": r["TP"] / positives if positives else None,
                "ROC_FPR": r["FP"] / negatives if negatives else None}
                for r in v56.calibration_curve(us, score, rs))
    return points


def decorate_protocol(records, pack, severity, protocol):
    decorated = v57.decorate_records(records, pack)
    return [{**r, "severity": severity, "protocol": protocol} for r in decorated]


def export_threshold_records(args, severity, protocol, records, pack):
    decorated = decorate_protocol(records, pack, severity, protocol)
    candidates = []
    grouped = {}
    for row in decorated:
        grouped.setdefault(row["dataset_id"], {})[row["lipid_name"]] = row
    for dataset, lookup in grouped.items():
        # Only called after both local freezes; original raw artifacts remain immutable.
        source = normalize_records(v57.rows(directory(args, dataset) / "candidate_false_negative_records.csv"))
        enriched = []
        for row in source:
            unit = lookup[row["lipid_name"]]
            enriched.append({**row, "protocol": protocol, "retention_flag_unit": "dataset/lipid_name",
                "candidate_solver_false_negative": bool(row["candidate_truth"] and not row["raw_solver_reported"]),
                **{key: value for key, value in unit.items() if key.startswith(
                    ("retained_by_", "filter_induced_true_loss_", "solver_false_negative")) or key.endswith("_threshold_available")}})
        candidates.extend(enriched)
        write_rows(directory(args, dataset) / f"candidate_false_negative_{protocol}.csv", enriched)
        write_rows(directory(args, dataset) / f"molecular_false_negative_{protocol}.csv", list(lookup.values()))
    return decorated, candidates


def clean_reference(args):
    records, reports = [], {}
    for split, filename in (("CAL", "calibration_records.csv"), ("HOLD", "heldout_records.csv")):
        current = normalize_records(v57.rows(args.v57_output / filename))
        require({r["dataset_id"] for r in current} == set(parent_base_ids(split)) and all(r["split"] == split for r in current),
                "CLEAN_REFERENCE_SCOPE_CHANGED")
        require(all(r["K"] == int(r["dataset_id"].split("_")[2][1:]) for r in current), "CLEAN_REFERENCE_K_CHANGED")
        selected = [r for r in current if r["K"] in K_LEVELS]
        require({r["dataset_id"] for r in selected} == set(base_ids(split)), "COMPACT_CLEAN_REFERENCE_INCOMPLETE")
        records.extend(selected)
    for row in records:
        report = reports.setdefault(row["dataset_id"], {"K": row["K"], "replicate": row["replicate"],
            "all_truth_molecular_identity_count": 0, "reportable_truth_molecular_identity_count": 0})
        report["all_truth_molecular_identity_count"] += int(row["molecular_truth"])
        report["reportable_truth_molecular_identity_count"] += int(row["reportable_truth"])
    require(all(r["all_truth_molecular_identity_count"] == r["K"] for r in reports.values()), "CLEAN_REFERENCE_TRUTH_CHANGED")
    return records, reports


def compact_mechanism_metrics(records, pack):
    # V57 deliberately retains its full six-K contract. Filter only its returned analysis rows.
    return [row for row in v57.mechanism_metrics(records, pack) if row["K"] in ("GLOBAL", *K_LEVELS)]


def aggregate(args, design, clean_thresholds, context=None, parent=None, targets=None):
    sentinels = require_sentinels(args, design, context, parent, targets)
    calibrated = {}
    # Phase 1 is exclusively CAL. No HOLD records are even loaded into memory here.
    for severity in SEVERITIES:
        records, reports, hashes = read_completed(args, design, severity, "CAL")
        frozen = freeze_local_thresholds(args, design, severity, records, reports, hashes)
        calibrated[severity] = (records, reports, frozen)
    hold_read_guard(args, design)
    fixed_metrics, local_metrics, by_k, by_r, separation, mechanisms, curves = [], [], [], [], [], [], []
    calibration_records, heldout_records, molecular_records, candidate_records, dataset_summaries = [], [], [], [], []
    for severity in SEVERITIES:
        cal, cal_reports, frozen = calibrated[severity]
        hold, hold_reports, _ = read_completed(args, design, severity, "HOLD")
        protocols = {"CLEAN_FIXED": {"rho_zero": clean_thresholds["global_thresholds"]["rho_zero"]},
                     "LOCAL_CAL_RECALIBRATION": frozen["global_thresholds"]}
        for protocol, pack in protocols.items():
            overall, krows, rrows = protocol_metrics(severity, protocol, hold, hold_reports, pack)
            (fixed_metrics if protocol == "CLEAN_FIXED" else local_metrics).extend(overall)
            by_k.extend(krows)
            by_r.extend(rrows)
            combined, candidates = export_threshold_records(args, severity, protocol, cal + hold, pack)
            molecular_records.extend(combined)
            candidate_records.extend(candidates)
            calibration_records.extend(r for r in combined if r["split"] == "CAL")
            heldout_records.extend(r for r in combined if r["split"] == "HOLD")
            mechanisms.extend({"severity": severity, "protocol": protocol, **row,
                "truth_count": row["raw_solver_TP"] + row["raw_solver_FN"], "solver_FN": row["raw_solver_FN"],
                "category_FP": None, "category_FDR": None,
                "threshold_available": pack["rho_zero"][row["target"]] is not None}
                for row in compact_mechanism_metrics(combined, pack))
        for split, records, reports in (("CAL", cal, cal_reports), ("HOLD", hold, hold_reports)):
            separation.extend(separation_summary(records, severity, split, k) for k in (None, *K_LEVELS))
            curves.extend(descriptive_curves(records, reports, severity, split))
            dataset_summaries.extend({"dataset_id": dataset, "severity": severity, "split": split, "K": r["K"],
                "replicate": r["replicate"], "forward_mismatch_residual": r["forward_mismatch_residual"],
                "learned_solver_reconstruction_relative_residual": r["learned_solver_reconstruction_relative_residual"],
                "source_NNLS_residual": r["source_library_diagnostic"]["source_NNLS_residual"],
                "source_NNLS_false_identity_count": r["source_library_diagnostic"]["false_identity_count"],
                "X_true_sha256": r["X_true_sha256"], "B_target_sha256": r["B_target_sha256"],
                **r["B_domain"], "mismatch_B_p50_over_V57_clean_B_p50": r["mismatch_B_p50_over_V57_clean_B_p50"]}
                for dataset, r in reports.items())
    clean, clean_reports = clean_reference(args)
    for split in ("CAL", "HOLD", "ALL"):
        records = [r for r in clean if split == "ALL" or r["split"] == split]
        reports = {d: r for d, r in clean_reports.items() if split == "ALL" or d.startswith(split + "_")}
        separation.extend(separation_summary(records, "CLEAN_REFERENCE", split, k) for k in (None, *K_LEVELS))
        if split != "ALL":
            curves.extend(descriptive_curves(records, reports, "CLEAN_REFERENCE", split))
    for filename, values in (("clean_fixed_transfer_metrics.csv", fixed_metrics), ("local_recalibration_metrics.csv", local_metrics),
        ("heldout_by_K.csv", by_k), ("heldout_by_replicate.csv", by_r), ("rho_separation_summary.csv", separation),
        ("mechanism_stratified_metrics.csv", mechanisms), ("calibration_records.csv", calibration_records),
        ("heldout_records.csv", heldout_records), ("candidate_false_negative_records.csv", candidate_records),
        ("molecular_false_negative_records.csv", molecular_records), ("dataset_summary.csv", dataset_summaries),
        ("roc_fdr_recall_curves.csv", curves)):
        write_rows(args.output_dir / filename, values)
    for score, label in (("rho_zero", "rho"), ("X_hat", "xhat")):
        points = [r for r in curves if r["score"] == score]
        write_rows(args.output_dir / f"fdr_recall_curve_{label}.csv", points)
        write_rows(args.output_dir / f"fdr_tp_retention_curve_{label}.csv", points)
    report = {"status": "PENDING_REVIEW", "process_validity": "PASS", "design_fingerprint": design["design_fingerprint"],
        "V57_parent_fingerprint": PARENT_FINGERPRINT, "V57_parent_file_hashes": design["scientific"]["V57_parent_file_hashes"],
        "result_variant": RESULT_VARIANT, "compact_rationale": COMPACT_RATIONALE, **SUPERSEDED_DRAFT,
        "measurement_noise": False, "mismatch_runs": 60, "clean_runs": 0,
        "target_oracle": {"status": "PASS", "count": 60, "summary_sha256": v56.digest(args.output_dir / "oracle_summary.json")},
        "sentinel": {"status": sentinels["status"], "count": len(sentinels["datasets"]), "gate_formula": GATE_FORMULA,
                     "evidence_mode": sentinels.get("evidence_mode", NATIVE_SENTINEL_MODE),
                     "source_draft_fingerprint": sentinels.get("source_draft_fingerprint"),
                     "source_sentinel_file_sha256": sentinels.get("source_sentinel_file_sha256")},
        "CLEAN_FIXED": fixed_metrics, "LOCAL_CAL_RECALIBRATION": local_metrics,
        "rho_separation": [r for r in separation if r["K"] == "GLOBAL"],
        "local_threshold_file_hashes": {s: v56.digest(threshold_path(args, s)) for s in SEVERITIES},
        "scientific_outcomes_used_for_validity": False, "limitations": LIMITATIONS,
        "final_deployment_threshold_frozen": False}
    write_json(args.output_dir / "report.json", report)
    summary = ["# Final compact V58 spectral-library mismatch and FDR recalibration", "", "Status: PENDING_REVIEW", "",
        COMPACT_RATIONALE, "",
        "CLEAN_FIXED evaluates absolute threshold transfer; LOCAL_CAL_RECALIBRATION uses only same-domain CAL.",
        "Scientific FDR/retention failures remain results, not invalid experiments.", "",
        "```json", json.dumps(v57.canonical(report), ensure_ascii=False, indent=2), "```", "",
        "Final deployment threshold frozen: false."]
    v56.stage0.atomic_write_text(args.output_dir / "summary.md", "\n".join(summary) + "\n")
    update_log(design, report)
    print("PENDING_REVIEW; both local CAL thresholds frozen before HOLD; deployment threshold not frozen")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-root", type=Path, default=Path("/root/autodl-tmp/decon-lipid"))
    parser.add_argument("--v57-output", type=Path, default=ROOT / "results" / PARENT_VERSION)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results" / RESULT_VARIANT)
    parser.add_argument("--source-draft-dir", type=Path, default=ROOT / "results" / VERSION,
                        help="read-only source of the four equivalent draft sentinels")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--rho-workers", type=int, default=1)
    actions = parser.add_mutually_exclusive_group(required=True)
    for name in ("prepare-design", "audit-design", "freeze-design", "oracle-all", "sentinel",
                 "adopt-equivalent-sentinel", "mismatch-all", "aggregate"):
        actions.add_argument(f"--{name}", action="store_true")
    actions.add_argument("--dataset", choices=dataset_ids(), metavar="SEVERITY__SPLIT_Rn_Knnn",
                         help="one mismatch dataset, e.g. MILD__CAL_R1_K050; all lifecycle gates still required")
    return parser.parse_args(argv)


def main():
    args = parse_args()
    guard_output(args)
    dependencies()
    sys.path.insert(0, str(ROOT / "src"))
    require(args.rho_workers >= 1, "RHO_WORKERS_MUST_BE_POSITIVE")
    # A typo must never put new artifacts inside the immutable parent.
    # The repository/results directory may legitimately live under the asset root.
    output, parent_path = args.output_dir.resolve(), args.v57_output.resolve()
    require(output != parent_path and parent_path not in output.parents and output not in parent_path.parents,
            "V58_OUTPUT_OVERLAPS_V57_PARENT")
    parent, clean_thresholds, parent_hashes = parent_provenance(args)
    context = load_context(args, parent)
    targets, manifest, hashes = build_targets(context)
    scientific = snapshot(args, context, parent_hashes, hashes)
    if args.prepare_design:
        prepare(args, scientific, manifest)
        return
    design = load_design(args, scientific)
    if args.audit_design:
        try:
            audit_design(args, context, parent, targets, design, manifest)
        except Exception as exc:
            invalidate(args, design, f"DESIGN_AUDIT_FAILED: {exc}")
            raise
    elif args.freeze_design:
        freeze(args, design)
    elif args.oracle_all:
        oracle_all(args, context, parent, targets, design)
    elif args.sentinel:
        sentinel(args, context, parent, targets, design)
    elif args.adopt_equivalent_sentinel:
        adopt_equivalent_sentinel(args, context, parent, targets, design)
    elif args.aggregate:
        aggregate(args, design, clean_thresholds, context, parent, targets)
    else:
        require_sentinels(args, design, context, parent, targets)
        for dataset in ([args.dataset] if args.dataset else dataset_ids()):
            run_dataset(args, context, parent, targets, design, dataset)


if __name__ == "__main__":
    main()
