"""Independent cached per-case review and compact export; no model execution."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import tarfile

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_name] = "1"

PHYSICAL_FP = "3b2e75c563a0d23ac16db2ce87f1815ae0c3574fac59c3d8e89b794768697df7"
KINDS = ("SUPPORTED_MISMATCH", "close_neighbor", "relatively_isolated")
CODE_SNAPSHOT_FILES = {
    "run_physical_identity_mainline_pilot.py", "physical_identity_pilot_design.py",
    "physical_identity_confidence_core.py", "run_ce_uncertainty_identity_pilot.py",
    "run_missing_library_challenge.py",
}


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def fp(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def ah(value):
    return hashlib.sha256(value.tobytes()).hexdigest()


def write(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n",
                         encoding="utf-8", newline="\n")
    temporary.replace(path)


def safe_path(root, relative):
    require(isinstance(relative, str) and "\\" not in relative and ":" not in relative, "UNSAFE_MANIFEST_PATH")
    relative = PurePosixPath(relative)
    require(not relative.is_absolute() and ".." not in relative.parts, "UNSAFE_MANIFEST_PATH")
    return Path(root).joinpath(*relative.parts)


def verify_hashes(root, hashes):
    for name, expected in hashes.items():
        path = safe_path(root, name)
        require(path.is_file() and sha(path) == expected, "ARTIFACT_CHANGED:" + name)


def finite_tree(value):
    if isinstance(value, dict):
        return all(finite_tree(v) for v in value.values())
    if isinstance(value, list):
        return all(finite_tree(v) for v in value)
    return not isinstance(value, (int, float)) or math.isfinite(value)


def expected_membership():
    return [dict(key=f"{kind}__{pool}_{rep}_K125", kind=kind, base=f"{pool}_{rep}_K125",
                 role=role, replicate=rep)
            for rep, role in (("R71", "CAL"), ("R72", "EVAL"))
            for kind, pool in (("SUPPORTED_MISMATCH", "CAL"), ("close_neighbor", "HOLD"),
                               ("relatively_isolated", "HOLD"))]


def design_review(args):
    design = read(args.output / "design.json")
    seal = read(args.output / "design_seal.json")
    scientific = design["scientific"]
    require(design["fingerprint"] == fp(scientific) == seal["fingerprint"], "DESIGN_FINGERPRINT_CHANGED")
    require(sha(args.output / "design.json") == seal["design_sha256"], "DESIGN_BYTES_CHANGED")
    require(scientific["membership"] == expected_membership(), "CASE_MEMBERSHIP_CHANGED")
    require(set(scientific["cases"]) == {e["key"] for e in expected_membership()}, "CASE_SET_CHANGED")
    require(scientific["contract"]["physical_contract"] == PHYSICAL_FP, "PHYSICAL_BINDING_CHANGED")
    require(scientific["contract"]["K"] == 125 and scientific["contract"]["fits"] == 6
            and scientific["contract"]["gamma_num"] == 1e-6, "PILOT_CORE_CONTRACT_CHANGED")
    require(sha(args.output / "target_containment.json") == seal["target_containment_sha256"], "CONTAINMENT_BYTES_CHANGED")
    require(read(args.output / "target_containment.json")["status"] == "PASS", "TARGET_CONTAINMENT_NOT_PASS")
    binding = scientific["source_binding"]
    donor = read(args.output / "donor_split.json")
    require(donor == scientific["donor_split"] and fp(donor) == binding["donor_split_fingerprint"], "DONOR_BINDING_CHANGED")
    require(donor["fingerprint"] == fp({k: v for k, v in donor.items() if k != "fingerprint"}), "DONOR_CONTENT_CHANGED")
    require(sha(args.output / "model_patterns.json") == scientific["model_patterns_sha256"], "MODEL_PATTERN_FILE_CHANGED")
    require({p["pattern_id"] for p in read(args.output / "model_patterns.json")} == set(scientific["model_pattern_ids"]),
            "MODEL_PATTERN_EXPORT_MEMBERSHIP_CHANGED")
    require(set(scientific["model_pattern_ids"]) == {p for p, role in donor["pattern_roles"].items() if role == "MODEL"},
            "INFERENCE_ENDPOINT_MEMBERSHIP_CHANGED")
    files = {role: set() for role in ("MODEL", "CAL", "EVAL")}
    identities = {role: set() for role in files}
    for group in donor["groups"]:
        files[group["role"]].update(group["source_files"])
        identities[group["role"]].update(tuple(i) for i in group["donor_identities"])
    for left, right in (("MODEL", "CAL"), ("MODEL", "EVAL"), ("CAL", "EVAL")):
        require(not files[left] & files[right] and not identities[left] & identities[right], "DONOR_SOURCE_OR_IDENTITY_LEAKAGE")
    physical = args.snapshot / "results/physical_perturbation_contract_v1"
    source_seal = read(physical / "seal.json")
    require(fp(source_seal["content"]) == source_seal["fingerprint"] == PHYSICAL_FP, "PHYSICAL_SOURCE_SEAL_CHANGED")
    for row in source_seal["content"]["inputs"]:
        require(sha(safe_path(args.snapshot, row["path"])) == row["sha256"], "PHYSICAL_INPUT_CHANGED:" + row["path"])
    verify_hashes(physical, source_seal["content"]["outputs"])
    records_path = args.snapshot / "results/rho_mismatch_mechanism_ce_audit/ce_identity_records.json"
    require(sha(records_path) == binding["donor_records_sha256"], "DONOR_RECORD_SOURCE_CHANGED")
    parent_root = args.root / "results/v57_spectral_spatial_identity_confidence_benchmark"
    verify_hashes(parent_root, binding["parent_files"])
    require(read(parent_root / "design.json")["design_fingerprint"] == binding["parent_fingerprint"], "PARENT_DESIGN_CHANGED")
    omission_root = args.root / "results/missing_library_challenge_k125"
    require(sha(omission_root / "execution_design.json") == binding["omission_design_sha256"], "OMISSION_SOURCE_CHANGED")
    omission = read(omission_root / "execution_design.json")
    require(fp(omission["scientific"]) == omission["fingerprint"], "OMISSION_FINGERPRINT_CHANGED")
    require(read(omission_root / "execution_freeze.json")["design_sha256"] == sha(omission_root / "execution_design.json"),
            "OMISSION_FREEZE_CHANGED")
    source_paths = {}
    for name, expected in binding["implementation"].items():
        require(Path(name).name == name, "UNEXPECTED_IMPLEMENTATION_PATH")
        options = ([args.code_dir / name] if name in CODE_SNAPSHOT_FILES else
                   [args.root / "analysis" / name, args.root / "src" / name, args.root / name])
        path = next((p for p in options if p.is_file()), None)
        require(path is not None and sha(path) == expected, "IMPLEMENTATION_CHANGED:" + name)
        source_paths[name] = str(path.resolve())
    return design, omission, source_paths


def case_review(args):
    import numpy as np

    design, omission, source_paths = design_review(args)
    scientific = design["scientific"]
    entry = next((e for e in scientific["membership"] if e["key"] == args.case), None)
    require(entry is not None, "UNKNOWN_CASE")
    root = args.output / "cases" / args.case
    done = read(root / "training_complete.json")
    info = read(root / "input.json")
    require(info == scientific["cases"][args.case], "CASE_INPUT_BINDING_CHANGED")
    require(done["status"] == "NORMAL_FINITE_COMPLETE" and done["case"] == args.case
            and done["fingerprint"] == design["fingerprint"], "CASE_NOT_COMPLETE_OR_DIFFERENT_DESIGN")
    verify_hashes(root, done["files"])
    require(sha(root / "observation_truth.npz") == info["observation_file_sha256"], "OBSERVATION_ARCHIVE_CHANGED")
    runtime = read(root / "training/runtime_contract.json")
    require(runtime == dict(design_fingerprint=design["fingerprint"], case=args.case, input=info), "RUNTIME_BINDING_CHANGED")
    with np.load(root / "observation_truth.npz", allow_pickle=False) as z:
        B, X_true, mask = z["B"], z["X_true"], z["mask"]
    require(B.ndim == 3 and X_true.shape == (391, *B.shape[1:]) and mask.shape == B.shape[1:]
            and mask.dtype == np.bool_ and mask.any(), "OBSERVATION_OR_TRUTH_SHAPE_CHANGED")
    require(np.isfinite(B).all() and np.isfinite(X_true).all() and (X_true >= 0).all(), "INVALID_OBSERVATION_TRUTH")
    require(ah(B) == info["B_sha256"] and ah(X_true) == info["X_true_sha256"], "OBSERVATION_TRUTH_BYTES_CHANGED")
    target_path = args.output / f"A_target_{entry['role']}.npy"
    require(sha(target_path) == scientific["targets"][entry["role"]]["file_sha256"], "TARGET_FILE_CHANGED")
    target = np.load(target_path, allow_pickle=False)
    require(ah(target) == info["A_target_sha256"] == scientific["targets"][entry["role"]]["array_sha256"], "TARGET_ARRAY_CHANGED")
    require(target.shape == (B.shape[0], 391) and np.isfinite(target).all(), "TARGET_SHAPE_CHANGED")
    predicted = np.einsum("mc,cyx->myx", target, X_true, optimize=True)
    forward_error = float(np.max(np.abs(predicted - B)))
    require(np.allclose(predicted, B, rtol=2e-6, atol=1e-7), "CACHED_FORWARD_EQUATION_FAILED")
    del predicted
    nominal_path = args.output / "A_solver.npy"
    require(sha(nominal_path) == scientific["A_solver_file_sha256"], "NOMINAL_FILE_CHANGED")
    nominal = np.load(nominal_path, allow_pickle=False)
    require(ah(nominal) == scientific["source_binding"]["A_solver_sha256"], "NOMINAL_ARRAY_CHANGED")
    kept = info["kept"]
    require(len(set(kept)) == len(kept) and kept == sorted(kept), "INVALID_REDUCED_CANDIDATE_ORDER")
    if entry["kind"] == "SUPPORTED_MISMATCH":
        require(kept == list(range(391)), "FULL_LIBRARY_CHANGED")
    else:
        arm = omission["scientific"]["arms"][entry["kind"]]
        require(kept == arm["reduced_to_original"] and set(arm["removed_original_indices"]) <= set(info["truth_indices"]),
                "OMISSION_MEMBERSHIP_CHANGED")
    require(ah(nominal[:, kept]) == info["A_solver_sha256"], "FIXED_SOLVER_SUBSET_CHANGED")
    with np.load(root / "evidence.npz", allow_pickle=False) as z:
        evidence = {name: z[name] for name in z.files}
    require(evidence["kept"].tolist() == kept and np.array_equal(evidence["A"], nominal[:, kept].astype(float)),
            "INFERENCE_NOMINAL_A_CHANGED")
    global_b = B[:, mask].mean(axis=1, dtype=float)
    require(np.array_equal(evidence["global_b"], global_b), "FOREGROUND_GLOBAL_EVIDENCE_CHANGED")
    with np.load(root / "training/learned_arrays.npz", allow_pickle=False) as z:
        X, B_hat = z["X_hat"], z["B_hat"]
    require(X.shape == (len(kept), *mask.shape) and B_hat.shape == B.shape, "LEARNED_ARRAY_SHAPE_CHANGED")
    require(np.isfinite(X).all() and np.isfinite(B_hat).all() and (X >= 0).all(), "NONFINITE_OR_NEGATIVE_LEARNED_ARRAY")
    run = read(root / "training/solver_run.json")
    history = read(root / "training/training_history.json")
    epoch = run["stopped_epoch"]
    require(run["stop_reason"] in ("converged", "max_epochs") and 1 <= epoch <= 3000, "ABNORMAL_PRODUCTION_COMPLETION")
    require(run["stop_reason"] != "max_epochs" or epoch == 3000, "MAX_EPOCH_STOP_BEFORE_CAP")
    require(done["stopped_epoch"] == epoch and done["stop_reason"] == run["stop_reason"], "COMPLETION_RECORD_CHANGED")
    require(history["epoch"] and history["epoch"][-1] == epoch and finite_tree(history) and finite_tree(run), "NONFINITE_HISTORY_OR_FINAL_LOSS")
    for field in ("train_total", "eval_total", "eval_physical_loss", "eval_raw_losses", "automatic_loss_weights"):
        require(len(history[field]) == len(history["epoch"])
                and np.isfinite(np.asarray(history[field], dtype=float)).all(), "TRAINING_HISTORY_FIELD_INVALID:" + field)
    require(run["final_physical_loss"] is not None and math.isfinite(run["final_physical_loss"])
            and len(run["final_raw_losses"]) == 4 and np.isfinite(np.asarray(run["final_raw_losses"], dtype=float)).all(),
            "MISSING_OR_NONFINITE_FINAL_LOSS")
    require(done["training_uses_no_truth_or_target_library"] is True, "TRAINING_TRUTH_BOUNDARY_CHANGED")
    require("training/latest_model.pth" in done["files"], "FINAL_USABLE_MODEL_NOT_BOUND")
    for checkpoint_epoch in (1000, 1500, 2000, 2500, 3000):
        if checkpoint_epoch <= epoch:
            require(f"training/checkpoint_epoch_{checkpoint_epoch}.pth" in done["files"], "REQUIRED_CHECKPOINT_NOT_BOUND")
    metadata = read(root / "metadata.json")
    names = metadata["lipid_name"]
    require(len(names) == 391, "METADATA_LENGTH_CHANGED")
    candidate = read(root / "candidate_records.json")
    molecular = read(root / "molecular_records.json")
    require(len(candidate) == 391 and [r["candidate_index"] for r in candidate] == list(range(391)), "CANDIDATE_RECORD_MEMBERSHIP_CHANGED")
    active = np.flatnonzero(np.any(X_true != 0, axis=(1, 2))).tolist()
    require(set(active) == set(info["truth_indices"]) and len(active) == 125, "TRUTH_MEMBERSHIP_CHANGED")
    truth_means = X_true[:, mask].mean(axis=1, dtype=float)
    reportable_indices = {i for i in active if truth_means[i] > .001}
    require(reportable_indices == set(info["reportable_truth_indices"]), "REPORTABLE_TRUTH_CHANGED")
    mapped = read(args.snapshot / "results/physical_perturbation_contract_v1/candidate_mapping.json")
    changes = np.linalg.norm(target.astype(float) - nominal.astype(float), axis=0) / np.linalg.norm(nominal.astype(float), axis=0)
    supported_indices = set(active) & {r["candidate_index"] for r in mapped if changes[r["candidate_index"]] > 1e-10}
    require(supported_indices == set(info["supported_perturbed_truth_indices"]), "SUPPORTED_PERTURBED_TRUTH_CHANGED")
    truth_names = {names[i] for i in active}
    reportable_names = {names[i] for i in reportable_indices}
    supported_names = {names[i] for i in supported_indices}
    means = X[:, mask].mean(axis=1, dtype=float)
    inverse = {original: reduced for reduced, original in enumerate(kept)}
    expected_candidate = []
    for i, name in enumerate(names):
        local = inverse.get(i)
        value = float(means[local]) if local is not None else 0.
        expected = dict(candidate_index=i, reduced_index=local, lipid_name=name,
                        candidate_id=str(metadata["candidate_id"][i]), lipid_class=str(metadata["lipid_class"][i]),
                        X_hat=value, raw_solver_reported=bool(local is not None and value > .001),
                        molecular_truth=name in truth_names, reportable_truth=name in reportable_names,
                        supported_perturbed_truth=name in supported_names, omitted_from_solver=local is None,
                        K=125, replicate=entry["replicate"], split=entry["role"])
        require(candidate[i] == expected, "CANDIDATE_RECORD_CHANGED:" + str(i))
        expected_candidate.append(expected)
    expected_molecular = []
    for i, name in enumerate(dict.fromkeys(names)):
        rows = [r for r in expected_candidate if r["lipid_name"] == name]
        expected_molecular.append(dict(lipid_name=name, evidence_index=i, X_hat=sum(r["X_hat"] for r in rows),
            raw_solver_reported=any(r["raw_solver_reported"] for r in rows), molecular_truth=name in truth_names,
            reportable_truth=name in reportable_names, supported_perturbed_truth=name in supported_names,
            removed=[r["reduced_index"] for r in rows if r["reduced_index"] is not None]))
    require(molecular == expected_molecular, "MOLECULAR_MEMBERSHIP_OR_FLAGS_CHANGED")
    signal = float(np.median(np.linalg.norm(B[:, mask], axis=0)))
    require(abs(signal - info["achieved_foreground_signal_norm"]) <= 1e-12
            and abs(signal - 0.6036783456802368) <= 2e-6, "DATASET_SIGNAL_SCALAR_CHANGED")
    # No raw/filtered EVAL counts or confidence outcomes are evaluated or exported here.
    compact = {}
    retained = {}
    for name, digest in done["files"].items():
        path = safe_path(root, name)
        relative = path.relative_to(args.output).as_posix()
        big = path.suffix == ".pth" or path.name in ("learned_arrays.npz", "observation_truth.npz")
        if big:
            retained[relative] = dict(sha256=digest, bytes=path.stat().st_size,
                                      path=str(path.resolve()), storage="required_original_artifact; not deleted")
        else:
            compact[relative] = digest
    for name in ("design.json", "design_seal.json", "donor_split.json", "target_containment.json", "model_patterns.json"):
        compact[name] = sha(args.output / name)
    compact[f"cases/{args.case}/training_complete.json"] = sha(root / "training_complete.json")
    for name in ("A_solver.npy", "A_target_CAL.npy", "A_target_EVAL.npy"):
        path = args.output / name
        retained[name] = dict(sha256=sha(path), bytes=path.stat().st_size, path=str(path.resolve()),
                              storage="required_original_generative_library; not deleted")
    result = dict(status="PASS", case=args.case, role=entry["role"], design_fingerprint=design["fingerprint"],
        training_complete_sha256=sha(root / "training_complete.json"),
        source_implementation_paths=source_paths, bound_artifact_count=len(done["files"]),
        normal_completion=True, all_outputs_losses_finite=True, runtime_and_source_bindings_verified=True,
        unchanged_production_early_stop=True, hard_cap=3000, stopped_epoch=epoch, stop_reason=run["stop_reason"],
        original_forward_equation_verified=True, forward_max_absolute_error=forward_error,
        fixed_solver_library_verified=True, global_foreground_spectrum_reconstructed=True,
        raw_candidate_molecular_membership_and_truth_flags_verified=True,
        original_truth_reportable_supported_denominators_verified=True,
        outcome_analysis="DEFERRED: process review never computes or publishes TP/FP/retention; final review requires CAL seal",
        compact_files=compact, retained_artifacts=retained,
        cleanup_performed=False, unique_models_and_arrays_retained=True,
        reviewer_sha256=sha(Path(__file__)))
    write(root / "case_review.json", result)
    verify_hashes(args.output, compact)
    print(json.dumps(dict(status="PASS", case=args.case, bound_artifact_count=len(done["files"]),
                          stopped_epoch=epoch, outcome_analysis="DEFERRED")), flush=True)
    return result


def export_case(args, review):
    """Export exactly reviewed compact files plus the review; never mutate source artifacts."""
    args.export_dir.mkdir(parents=True, exist_ok=True)
    manifest = dict(review["compact_files"])
    review_name = f"cases/{args.case}/case_review.json"
    manifest[review_name] = sha(args.output / review_name)
    verify_hashes(args.output, manifest)
    archive = args.export_dir / (args.case + ".tar.gz")
    receipt_path = args.export_dir / (args.case + ".receipt.json")
    if archive.exists() or receipt_path.exists():
        require(archive.is_file() and receipt_path.is_file(), "INCOMPLETE_PREVIOUS_EXPORT_PRESERVED")
        receipt = read(receipt_path)
        require(receipt["files"] == manifest and sha(archive) == receipt["archive_sha256"], "EXISTING_EXPORT_CHANGED")
        return receipt
    temporary = archive.with_suffix(archive.suffix + ".tmp")
    require(not temporary.exists(), "PARTIAL_ARCHIVE_EXISTS_PRESERVED")
    with tarfile.open(temporary, "w:gz") as tar:
        for name in sorted(manifest):
            source = safe_path(args.output, name)
            require(not source.is_symlink(), "COMPACT_SYMLINK_NOT_EXPORTABLE")
            tar.add(source, arcname=name, recursive=False)
    with tarfile.open(temporary, "r:gz") as tar:
        members = tar.getmembers()
        require(len(members) == len(manifest) and {m.name for m in members} == set(manifest), "EXPORT_MEMBERSHIP_CHANGED")
        for member in members:
            require(member.isfile(), "EXPORT_NONREGULAR_MEMBER")
            handle = tar.extractfile(member)
            require(handle is not None and hashlib.sha256(handle.read()).hexdigest() == manifest[member.name],
                    "EXPORT_BYTES_CHANGED:" + member.name)
    temporary.replace(archive)
    receipt = dict(status="REVIEWED_COMPACT_EXPORT_READY_NOT_YET_DOWNLOADED_OR_PUSHED", case=args.case,
                   design_fingerprint=review["design_fingerprint"], files=manifest,
                   archive_path=str(archive.resolve()), archive_sha256=sha(archive), archive_bytes=archive.stat().st_size,
                   training_complete_sha256=review["training_complete_sha256"],
                   retained_artifacts=review["retained_artifacts"], no_source_artifacts_deleted=True)
    write(receipt_path, receipt)
    print(json.dumps(dict(status=receipt["status"], case=args.case, archive=str(archive))), flush=True)
    return receipt


def classification(record, epsilon):
    full, deletion = record["full"], record.get("deleted")
    if full.get("status") != "BOUNDS_VALID" or not deletion or deletion.get("status") != "BOUNDS_VALID":
        return "NUMERICALLY_UNRESOLVED"
    if full["lower"] > epsilon + 1e-6:
        return "FULL_MODEL_INCOMPATIBLE"
    if full["upper"] > epsilon - 1e-6:
        return "THRESHOLD_UNRESOLVED_FULL"
    if deletion["lower"] > epsilon + 1e-6:
        return "RETAINED"
    if deletion["upper"] <= epsilon - 1e-6:
        return "RELAXATION_REPLACEABLE"
    return "THRESHOLD_UNRESOLVED"


def independent_accounting(cases, epsilon, supported=False):
    records = [row for case in cases for row in case["records"]]
    selections = [row for row in records if classification(row, epsilon) == "RETAINED"]
    population_flag = "supported_perturbed_truth" if supported else "molecular_truth"
    population_count = "supported_perturbed_truth_count" if supported else "truth_count"
    truth = sum(case[population_count] for case in cases)
    raw_tp = sum(bool(row[population_flag]) for row in records)
    retained_tp = sum(bool(row[population_flag]) for row in selections)
    retained_fp = sum(not row["molecular_truth"] for row in selections)
    reportable = sum(case["reportable_truth_count"] for case in cases)
    require(0 <= retained_tp <= raw_tp <= truth, "INVALID_TRUE_POSITIVE_ACCOUNTING")
    return dict(raw_solver_TP=raw_tp, raw_solver_FP=None if supported else len(records) - raw_tp,
        raw_solver_FN=truth - raw_tp, filtered_TP=retained_tp, filtered_FP=None if supported else retained_fp,
        filtered_FN=truth - retained_tp, filter_induced_true_loss=raw_tp - retained_tp,
        filter_induced_true_loss_fraction=(raw_tp - retained_tp) / raw_tp if raw_tp else None,
        TP_retention=retained_tp / raw_tp if raw_tp else None, all_truth_recall=retained_tp / truth if truth else None,
        reportable_truth_recall=(sum(r["reportable_truth"] for r in selections) / reportable
                                 if not supported and reportable else None),
        FDP=retained_fp / len(selections) if not supported and selections else None,
        retained_count=retained_tp if supported else len(selections), truth_count=truth,
        distinct_identities=len({row["lipid_name"] for row in selections if not supported or row[population_flag]}),
        status_counts=dict(Counter(classification(row, epsilon) for row in records if not supported or row[population_flag])),
        risk_scope="NOT_APPLICABLE_TRUE_SUBSET" if supported else "all reported molecular identities")


def independent_calibration(cases):
    require(cases and all(case["role"] == "CAL" for case in cases), "CALIBRATION_REQUIRES_CAL_ONLY")
    events = {0., 1.000001}
    for case in cases:
        for row in case["records"]:
            if row["full"]["status"] != "BOUNDS_VALID" or row["deleted"]["status"] != "BOUNDS_VALID":
                continue
            for edge in (row["full"]["upper"] + 1e-6, row["deleted"]["lower"] - 1e-6):
                if edge >= 0:
                    events.update((float(edge), math.nextafter(float(edge), -math.inf),
                                   math.nextafter(float(edge), math.inf)))
    curve = [dict(epsilon=epsilon, **independent_accounting(cases, epsilon))
             for epsilon in sorted(events) if epsilon >= 0]
    feasible = [point for point in curve if point["retained_count"] and point["FDP"] <= .01]
    if feasible:
        chosen = max(feasible, key=lambda p: (p["filtered_TP"], -p["filtered_FP"], p["epsilon"]))
    else:
        chosen = dict(epsilon=1.000001, **independent_accounting(cases, 1.000001))
    return dict(status="CALIBRATED" if feasible else "EMPTY_CALIBRATION", epsilon=chosen["epsilon"],
                chosen=chosen, curve=curve)


def independent_go(cases, epsilon):
    aggregate = independent_accounting(cases, epsilon)
    supported = independent_accounting(cases, epsilon, True)
    by_challenge = {kind: dict(all_truth=independent_accounting([c for c in cases if c["kind"] == kind], epsilon),
                              supported_perturbed_truth=independent_accounting(
                                  [c for c in cases if c["kind"] == kind], epsilon, True)) for kind in KINDS}
    required_populations = [aggregate, supported]
    for result in by_challenge.values():
        required_populations.extend((result["all_truth"], result["supported_perturbed_truth"]))
    useful = all(p["retained_count"] > 0 and p["TP_retention"] is not None and p["TP_retention"] >= .4
                 for p in required_populations)
    go = bool(useful and aggregate["FDP"] is not None and aggregate["FDP"] <= .01)
    return dict(aggregate=aggregate, supported_perturbed_truth=supported, by_challenge=by_challenge,
                go=go, strong_success=bool(go and aggregate["TP_retention"] >= .6))


def final_review(args):
    """Recheck saved numerical bounds and all decisions without optimizing anything."""
    import numpy as np
    import sys
    sys.path.insert(0, str(args.code_dir))
    from physical_identity_confidence_core import build_model_bank

    design, _, _ = design_review(args)
    scientific = design["scientific"]
    # These reads precede any EVAL confidence or performance record access.
    seal_path = args.output / "calibration_seal.json"
    threshold = read(seal_path)
    access_path = args.output / "evaluation_access.json"
    access = read(access_path)
    require(access == dict(status="CAL_SEALED_BEFORE_EVAL_CONFIDENCE", design_fingerprint=design["fingerprint"],
                           calibration_seal_sha256=sha(seal_path)), "CAL_SEAL_OR_EVAL_ACCESS_CHANGED")
    require(threshold["design_fingerprint"] == design["fingerprint"] and threshold["physical_contract"] == PHYSICAL_FP,
            "CAL_THRESHOLD_BINDING_CHANGED")
    require(threshold["model_patterns_sha256"] == scientific["model_patterns_sha256"], "CAL_MODEL_BINDING_CHANGED")
    evidence_path = args.output / "evidence_seal.json"
    evidence_seal = read(evidence_path)
    require(evidence_seal["design_fingerprint"] == design["fingerprint"]
            and set(evidence_seal["cases"]) == set(scientific["cases"]), "EVIDENCE_SEAL_MEMBERSHIP_CHANGED")
    model_patterns = read(args.output / "model_patterns.json")
    mapping = read(args.snapshot / "results/physical_perturbation_contract_v1/candidate_mapping.json")
    with np.load(args.snapshot / "results/ce133_uncertainty_v1_ready/component_fractions.npz", allow_pickle=False) as z:
        fractions = z["fractions"].copy()
    compact, retained, numerical = {}, {}, {}
    all_cases, case_checks = [], []
    proof_count, max_gap = 0, 0.
    for entry in scientific["membership"]:
        key = entry["key"]
        source, scores_dir = args.output / "cases" / key, args.output / "scores" / key
        reviewed = read(source / "case_review.json")
        require(reviewed["status"] == "PASS" and reviewed["design_fingerprint"] == design["fingerprint"], "CASE_NOT_INDEPENDENTLY_REVIEWED")
        require(reviewed["training_complete_sha256"] == sha(source / "training_complete.json"), "REVIEWED_TRAINING_CHANGED")
        done = read(source / "training_complete.json")
        verify_hashes(source, done["files"])
        verify_hashes(source, evidence_seal["cases"][key])
        verify_hashes(args.output, reviewed["compact_files"])
        compact.update(reviewed["compact_files"])
        retained.update(reviewed["retained_artifacts"])
        compact[f"cases/{key}/case_review.json"] = sha(source / "case_review.json")
        ack_path = source / "handoff_ack.json"
        ack = read(ack_path)
        require(ack["status"] == "VERIFIED_DOWNLOADED_AND_PUSHED" and ack["case"] == key
                and ack["training_complete_sha256"] == sha(source / "training_complete.json"), "CASE_GIT_HANDOFF_CHANGED")
        compact[f"cases/{key}/handoff_ack.json"] = sha(ack_path)
        original = read(source / "molecular_records.json")
        metadata = [{"lipid_name": n} for n in read(source / "metadata.json")["lipid_name"]]
        with np.load(source / "evidence.npz", allow_pickle=False) as z:
            A, kept, b = z["A"].copy(), z["kept"].tolist(), z["global_b"].copy()
        bank = build_model_bank(A, fractions, model_patterns, mapping, scientific["model_pattern_ids"], metadata, kept)
        require(read(scores_dir / "bank_manifest.json") == bank.manifest, "BANK_MANIFEST_CHANGED")
        expected_binding = dict(design_fingerprint=design["fingerprint"], evidence_sha256=sha(source / "evidence.npz"),
            records_sha256=sha(source / "molecular_records.json"), bank_fingerprint=bank.fingerprint,
            evidence_seal_sha256=sha(evidence_path), evaluation_access_sha256=sha(access_path) if entry["role"] == "EVAL" else None)
        full_doc = read(scores_dir / "full.json")
        require(full_doc["binding"] == expected_binding and full_doc["proof_sha256"] == sha(scores_dir / "full_proof.npz"),
                "FULL_NUMERICAL_BINDING_CHANGED")
        with np.load(scores_dir / "full_proof.npz", allow_pickle=False) as z:
            full_proof = {k: z[k].copy() for k in z.files}
        full_result = bank.verify_full(b, full_doc["result"], full_proof)
        require(full_result["status"] in ("PASS", "ABSTENTION_ONLY"), "FULL_NUMERICAL_PROOF_INVALID")
        proof_count += full_result.get("lp_proofs", 0)
        max_gap = max(max_gap, full_result.get("max_numerical_gap", 0.))
        case = read(scores_dir / "scores.json")
        require(all(case[k] == value for k, value in entry.items()) and case["binding"] == expected_binding
                and case["full_sha256"] == sha(scores_dir / "full.json"), "SCORE_BINDING_CHANGED")
        require(case["truth_count"] == sum(r["molecular_truth"] for r in original)
                and case["reportable_truth_count"] == sum(r["reportable_truth"] for r in original)
                and case["supported_perturbed_truth_count"] == sum(r["supported_perturbed_truth"] for r in original),
                "SCORE_TRUTH_DENOMINATOR_CHANGED")
        reported = [r for r in original if r["raw_solver_reported"]]
        require([r["lipid_name"] for r in case["records"]] == [r["lipid_name"] for r in reported], "SCORED_REPORT_MEMBERSHIP_CHANGED")
        expected_proofs = {"full_proof.npz": full_doc["proof_sha256"]}
        for row, record in zip(reported, case["records"]):
            item_path = scores_dir / f"record_{row['evidence_index']:04d}.json"
            item = read(item_path)
            require(item["binding"] == expected_binding and item["input_record"] == row
                    and item["full_sha256"] == sha(scores_dir / "full.json"), "DELETION_RECORD_BINDING_CHANGED")
            require(item["proof_file"] == f"proof_{row['evidence_index']:04d}.npz"
                    and item["proof_files"] == {item["proof_file"]: sha(scores_dir / item["proof_file"])}, "DELETION_PROOF_FILE_CHANGED")
            with np.load(scores_dir / item["proof_file"], allow_pickle=False) as z:
                proof = {k: z[k].copy() for k in z.files}
            checked = bank.verify_delete(b, row["lipid_name"], item["deletion_result"], proof)
            require(checked["status"] in ("PASS", "ABSTENTION_ONLY"), "DELETION_NUMERICAL_PROOF_INVALID")
            proof_count += checked.get("lp_proofs", 0)
            max_gap = max(max_gap, checked.get("max_numerical_gap", 0.))
            short = lambda r: {k: r.get(k) for k in ("status", "lower", "upper", "numerical_error")}
            expected_record = dict(row, full=short(full_doc["result"]), deleted=short(item["deletion_result"]))
            require(record == item["record"] == expected_record, "CLASSIFICATION_INPUT_BOUNDS_CHANGED")
            expected_proofs.update(item["proof_files"])
            compact[item_path.relative_to(args.output).as_posix()] = sha(item_path)
        require(case["proof_files"] == expected_proofs, "NUMERICAL_PROOF_MEMBERSHIP_CHANGED")
        verify_hashes(scores_dir, expected_proofs)
        for name, digest in expected_proofs.items():
            path = scores_dir / name
            rel = path.relative_to(args.output).as_posix()
            numerical[rel] = digest
            retained[rel] = dict(sha256=digest, bytes=path.stat().st_size, path=str(path.resolve()),
                                 storage="retained numerical proof; included in final transport archive")
        for name in ("full.json", "scores.json", "bank_manifest.json", "classified_records.json"):
            compact[f"scores/{key}/{name}"] = sha(scores_dir / name)
        classified = read(scores_dir / "classified_records.json")
        require(classified == [dict(r, classification=classification(r, threshold["epsilon"])) for r in case["records"]],
                "SAVED_CLASSIFICATIONS_CHANGED")
        all_cases.append(case)
        case_checks.append(dict(case=key, status="PASS", full_proof_status=full_result["status"],
                                scored_identity_count=len(reported), all_same_name_deletions_verified=True))
        print("INDEPENDENT_PHYSICAL_PROOF_PASS", key, flush=True)
    calibration = [case for case in all_cases if case["role"] == "CAL"]
    evaluation = [case for case in all_cases if case["role"] == "EVAL"]
    rebuilt = independent_calibration(calibration)
    expected_cal_sources = {c["key"]: sha(args.output / "scores" / c["key"] / "scores.json") for c in calibration}
    require(threshold["CAL_source_hashes"] == expected_cal_sources, "CAL_SOURCE_MEMBERSHIP_CHANGED")
    require(all(threshold[name] == value for name, value in rebuilt.items()), "CAL_THRESHOLD_OR_CURVE_NOT_REPRODUCED")
    epsilon = rebuilt["epsilon"]
    expected_summary = independent_go(evaluation, epsilon)
    expected_summary.update(status="COMPLETE_AWAITING_INDEPENDENT_REVIEW", epsilon=epsilon,
        calibration=independent_accounting(calibration, epsilon), CAL_supported=independent_accounting(calibration, epsilon, True),
        by_case={c["key"]: dict(all_truth=independent_accounting([c], epsilon),
                               supported=independent_accounting([c], epsilon, True)) for c in all_cases},
        independent_real_experiment_validation=False)
    require(read(args.output / "summary.json") == expected_summary, "FINAL_ACCOUNTING_OR_GO_NOT_REPRODUCED")
    require(sha(seal_path) == access["calibration_seal_sha256"], "CAL_SEAL_CHANGED_DURING_REVIEW")
    for rel, record in retained.items():
        require(sha(safe_path(args.output, rel)) == record["sha256"], "RETAINED_ARTIFACT_CHANGED:" + rel)
    result = dict(status="PASS", design_fingerprint=design["fingerprint"], cases=case_checks,
        valid_LP_proof_count=proof_count, maximum_verified_LP_gap=max_gap,
        full_acceptance_requires_original_physical_witness=True, deletion_bounds_are_conservative_relaxation=True,
        all_same_name_deletions=True, CAL_only_threshold_curve_reproduced=True,
        EVAL_access_bound_to_prior_CAL_seal=True, all_classifications_accounting_and_GO_reproduced=True,
        empirical_result=expected_summary, formal_FDR_guarantee=False,
        risk_uncertainty="Small dependent identity-context development sample; empirical FDP is not a validated population FDR bound. No binomial independence assumption or formal confidence guarantee.",
        model_scope="CE-informed shared-fragment systematic mismatch only; no estimated censoring or pixel fluctuation",
        no_optimization_or_training_performed=True, no_source_artifacts_deleted=True,
        numerical_proof_files=numerical, reviewer_sha256=sha(Path(__file__)))
    write(args.output / "independent_final_review.json", result)
    for name in ("evidence_seal.json", "calibration_seal.json", "evaluation_access.json", "summary.json", "independent_final_review.json"):
        compact[name] = sha(args.output / name)
    write(args.output / "retained_large_artifacts.json", retained)
    compact["retained_large_artifacts.json"] = sha(args.output / "retained_large_artifacts.json")
    verify_hashes(args.output, compact)
    write(args.output / "output_manifest.json", compact)
    print(json.dumps(dict(status="PASS", cases=len(case_checks), valid_LP_proofs=proof_count,
                          maximum_LP_gap=max_gap, go=expected_summary["go"])), flush=True)
    return result


def export_final(args, review):
    args.export_dir.mkdir(parents=True, exist_ok=True)
    manifest = read(args.output / "output_manifest.json")
    manifest["output_manifest.json"] = sha(args.output / "output_manifest.json")
    manifest.update(review["numerical_proof_files"])
    verify_hashes(args.output, manifest)
    archive = args.export_dir / "final.tar.gz"
    receipt_path = args.export_dir / "final.receipt.json"
    if archive.exists() or receipt_path.exists():
        require(archive.is_file() and receipt_path.is_file(), "INCOMPLETE_FINAL_EXPORT_PRESERVED")
        receipt = read(receipt_path)
        require(receipt["files"] == manifest and sha(archive) == receipt["archive_sha256"], "FINAL_EXPORT_CHANGED")
        return receipt
    temporary = args.export_dir / "final.tar.gz.tmp"
    require(not temporary.exists(), "PARTIAL_FINAL_EXPORT_PRESERVED")
    with tarfile.open(temporary, "w:gz") as tar:
        for name in sorted(manifest):
            path = safe_path(args.output, name)
            require(not path.is_symlink(), "FINAL_EXPORT_SYMLINK_FORBIDDEN")
            tar.add(path, arcname=name, recursive=False)
    with tarfile.open(temporary, "r:gz") as tar:
        members = tar.getmembers()
        require(len(members) == len(manifest) and {m.name for m in members} == set(manifest), "FINAL_ARCHIVE_MEMBERSHIP_CHANGED")
        for member in members:
            require(member.isfile(), "FINAL_ARCHIVE_NONREGULAR_MEMBER")
            require(hashlib.sha256(tar.extractfile(member).read()).hexdigest() == manifest[member.name], "FINAL_ARCHIVE_HASH_MISMATCH")
    temporary.replace(archive)
    receipt = dict(status="FINAL_INDEPENDENTLY_REVIEWED_EXPORT_READY_NOT_YET_PUSHED", design_fingerprint=review["design_fingerprint"],
                   archive_path=str(archive.resolve()), archive_sha256=sha(archive), archive_bytes=archive.stat().st_size,
                   files=manifest, retained_artifacts=read(args.output / "retained_large_artifacts.json"),
                   all_numerical_proofs_included=True, unique_source_models_arrays_not_deleted=True)
    write(receipt_path, receipt)
    print(json.dumps(dict(status=receipt["status"], archive=str(archive))), flush=True)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--case")
    mode.add_argument("--final", action="store_true")
    parser.add_argument("--root", type=Path, default=Path("/root/autodl-tmp/lipiddeconv"))
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--code-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--export-dir", type=Path)
    args = parser.parse_args()
    if args.final:
        review = final_review(args)
        if args.export_dir is not None:
            export_final(args, review)
    else:
        review = case_review(args)
        if args.export_dir is not None:
            export_case(args, review)


if __name__ == "__main__":
    main()
