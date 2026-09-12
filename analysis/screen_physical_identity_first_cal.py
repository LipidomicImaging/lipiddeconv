"""User-directed first-CAL budget screen; separate from the frozen six-case pilot.

Reuses the frozen MODEL-only scoring, numerical proofs and calibration rule.
Does not train, touch formal CAL/EVAL outputs or declare an EVAL GO/NO-GO.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

KEY = "SUPPORTED_MISMATCH__CAL_R71_K125"
DESIGN = "a86cadbfcafe8abc9a12810c3ee71d673c5a1bae55c018824fb952e07bdac9e4"
PUSHED = "382ee2091773d93d027faecd93b27fde5f22e4a6"


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            h.update(block)
    return h.hexdigest()


def immutable(path, value):
    data = (json.dumps(value, indent=2, allow_nan=False) + "\n").encode()
    if path.exists():
        require(path.read_bytes() == data, "SCREEN_FILE_CHANGED:" + path.name)
    else:
        with path.open("xb") as stream:
            stream.write(data)


def prepare(args):
    require(not args.output.is_relative_to(args.source_output) and not args.source_output.is_relative_to(args.output)
            and not args.output.is_relative_to(args.snapshot), "SCREEN_OUTPUT_MUST_BE_SEPARATE")
    manifest = read(args.snapshot / "source_manifest.json")
    require(len(manifest["files"]) == 25, "FROZEN_SOURCE_MEMBERSHIP_CHANGED")
    for name, item in manifest["files"].items():
        path = (args.snapshot / name).resolve()
        require(path.is_relative_to(args.snapshot) and sha(path) == item["sha256"]
                and path.stat().st_size == item["bytes"], "FROZEN_SOURCE_CHANGED:" + name)
    design = read(args.source_output / "design.json")
    seal = read(args.source_output / "design_seal.json")
    canonical = json.dumps(design["scientific"], sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    require(hashlib.sha256(canonical).hexdigest() == design["fingerprint"] == seal["fingerprint"] == DESIGN
            and sha(args.source_output / "design.json") == seal["design_sha256"], "ORIGINAL_DESIGN_CHANGED")
    require(sha(args.source_output / "target_containment.json") == seal["target_containment_sha256"], "TARGET_CONTAINMENT_CHANGED")
    entry = next(x for x in design["scientific"]["membership"] if x["key"] == KEY)
    require(entry["role"] == "CAL" and len(design["scientific"]["membership"]) == 6, "ORIGINAL_ROLE_OR_MEMBERSHIP_CHANGED")
    source = args.source_output / "cases" / KEY
    done, review, ack = (read(source / name) for name in ("training_complete.json", "case_review.json", "handoff_ack.json"))
    require(done["status"] == "NORMAL_FINITE_COMPLETE" and done["case"] == KEY and done["fingerprint"] == DESIGN,
            "FIRST_FIT_NOT_NORMAL_FINITE_COMPLETE")
    require(review["status"] == "PASS" and review["case"] == KEY and review["design_fingerprint"] == DESIGN
            and review["normal_completion"] is True and review["all_outputs_losses_finite"] is True,
            "FIRST_CASE_REVIEW_NOT_PASS")
    require(ack["status"] == "VERIFIED_DOWNLOADED_AND_PUSHED" and ack["case"] == KEY and ack["pushed_commit"] == PUSHED
            and ack["training_complete_sha256"] == review["training_complete_sha256"] == sha(source / "training_complete.json"),
            "FIRST_CASE_HANDOFF_CHANGED")
    for name, value in review["compact_files"].items():
        path = (args.source_output / name).resolve()
        require(path.is_relative_to(args.source_output) and sha(path) == value, "REVIEWED_COMPACT_SOURCE_CHANGED:" + name)
    require(read(source / "input.json") == design["scientific"]["cases"][KEY], "FIRST_INPUT_CHANGED")
    names = ("input.json", "evidence.npz", "metadata.json", "molecular_records.json", "candidate_records.json")
    for name in names:
        require(sha(source / name) == done["files"][name], "FIRST_SCORE_SOURCE_CHANGED:" + name)
    bound = {name: sha(source / name) for name in (*names, "training_complete.json", "case_review.json", "handoff_ack.json")}
    args.output.mkdir(parents=True, exist_ok=True)
    protocol = dict(status="FIRST_CAL_ONLY_NOT_OFFICIAL_CAL", user_directed_budget_screen=True, case=KEY,
                    original_six_case_design_fingerprint=DESIGN, original_case_pushed_commit=PUSHED,
                    source_output=str(args.source_output), snapshot=str(args.snapshot), input_hashes=bound,
                    source_manifest_sha256=sha(args.snapshot / "source_manifest.json"), wrapper_sha256=sha(Path(__file__)),
                    scope="Original first CAL only; no other fit, formal CAL seal or EVAL access",
                    reuse="Unchanged frozen score_case, MODEL uncertainty, LP proofs, gamma and calibration",
                    stop_rule="Any numerical unresolved: TECHNICAL_UNRESOLVED; otherwise same-epsilon all/support 40% utility, then 1% risk",
                    scientific_interpretation="Budget screen only; not formal EVAL failure or a general impossibility result")
    immutable(args.output / "protocol.json", protocol)
    immutable(args.output / "evidence_seal.json", dict(status="FIRST_CAL_ONLY_NOT_OFFICIAL_CAL", design_fingerprint=DESIGN,
                                                      protocol_sha256=sha(args.output / "protocol.json"), cases={KEY: bound}))
    data = (args.source_output / "model_patterns.json").read_bytes()
    require(hashlib.sha256(data).hexdigest() == design["scientific"]["model_patterns_sha256"], "MODEL_PATTERNS_CHANGED")
    destination = args.output / "model_patterns.json"
    if destination.exists():
        require(destination.read_bytes() == data, "SCREEN_MODEL_CHANGED")
    else:
        destination.write_bytes(data)
    alias = args.output / "cases" / KEY
    alias.parent.mkdir(exist_ok=True)
    if alias.is_symlink():
        require(alias.resolve() == source.resolve(), "FIRST_CASE_ALIAS_CHANGED")
    else:
        require(not alias.exists(), "SCREEN_CASE_PATH_CONFLICT")
        alias.symlink_to(source, target_is_directory=True)
    return design, entry, bound


def verify_proofs(args, design, entry, run, core):
    import numpy as np
    source, scores = args.source_output / "cases" / KEY, args.output / "scores" / KEY
    with np.load(source / "evidence.npz", allow_pickle=False) as z:
        A, kept, b = z["A"].copy(), z["kept"].tolist(), z["global_b"].copy()
    with np.load(args.snapshot / "results/ce133_uncertainty_v1_ready/component_fractions.npz", allow_pickle=False) as z:
        fractions = z["fractions"].copy()
    bank = core.build_model_bank(A, fractions, read(args.output / "model_patterns.json"),
                                read(args.snapshot / "results/physical_perturbation_contract_v1/candidate_mapping.json"),
                                design["scientific"]["model_pattern_ids"],
                                [{"lipid_name": n} for n in read(source / "metadata.json")["lipid_name"]], kept)
    require(read(scores / "bank_manifest.json") == bank.manifest, "MODEL_BANK_CHANGED")
    binding = dict(design_fingerprint=DESIGN, evidence_sha256=sha(source / "evidence.npz"),
                   records_sha256=sha(source / "molecular_records.json"), bank_fingerprint=bank.fingerprint,
                   evidence_seal_sha256=sha(args.output / "evidence_seal.json"), evaluation_access_sha256=None)
    full = read(scores / "full.json")
    require(full["binding"] == binding and full["proof_sha256"] == sha(scores / "full_proof.npz"), "FULL_BINDING_CHANGED")
    def proof(path):
        with np.load(path, allow_pickle=False) as z:
            return {k: z[k].copy() for k in z.files}
    proofs = proof(scores / "full_proof.npz")
    checks = [bank.verify_full(b, full["result"], proofs)]
    bank.restore_full(b, full["result"], proofs)
    case = read(scores / "scores.json")
    original = read(source / "molecular_records.json")
    reported = [r for r in original if r["raw_solver_reported"]]
    require(all(case[k] == v for k, v in entry.items()) and case["binding"] == binding
            and case["full_sha256"] == sha(scores / "full.json"), "SCORE_BINDING_CHANGED")
    for count, flag in (("truth_count", "molecular_truth"), ("reportable_truth_count", "reportable_truth"),
                        ("supported_perturbed_truth_count", "supported_perturbed_truth")):
        require(case[count] == sum(r[flag] for r in original), "TRUTH_DENOMINATOR_CHANGED")
    require(len(case["records"]) == len(reported), "REPORTED_MEMBERSHIP_CHANGED")
    expected_files = {"full_proof.npz": full["proof_sha256"]}
    for row, record in zip(reported, case["records"]):
        item = read(scores / f"record_{row['evidence_index']:04d}.json")
        filename = f"proof_{row['evidence_index']:04d}.npz"
        require(item["binding"] == binding and item["input_record"] == row and item["full_sha256"] == sha(scores / "full.json")
                and item["proof_file"] == filename and item["proof_files"] == {filename: sha(scores / filename)}, "DELETION_BINDING_CHANGED")
        checks.append(bank.verify_delete(b, row["lipid_name"], item["deletion_result"], proof(scores / filename)))
        require(record == item["record"] == dict(row, full=run.short_bounds(full["result"]),
                                                deleted=run.short_bounds(item["deletion_result"])), "RECORD_BOUNDS_CHANGED")
        expected_files.update(item["proof_files"])
    require(case["proof_files"] == expected_files and all(c["status"] in ("PASS", "ABSTENTION_ONLY") for c in checks),
            "NUMERICAL_PROOF_INVALID")
    return case, dict(status="PASS", no_refit=True, verified_identity_count=len(reported),
                      lp_proofs=sum(c.get("lp_proofs", 0) for c in checks),
                      max_numerical_gap=max(c.get("max_numerical_gap", 0.) for c in checks),
                      abstention_only_proof_count=sum(c["status"] == "ABSTENTION_ONLY" for c in checks))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-output", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, choices=(1, 4), default=4)
    args = parser.parse_args()
    args.source_output, args.snapshot, args.output = (p.resolve() for p in (args.source_output, args.snapshot, args.output))
    design, entry, original_hashes = prepare(args)
    sys.path.insert(0, str(args.snapshot / "analysis"))
    import run_physical_identity_mainline_pilot as run
    import physical_identity_confidence_core as core
    import review_physical_identity_mainline_pilot as reviewer
    run.score_case(args, design, entry)
    case, proof_review = verify_proofs(args, design, entry, run, core)
    calibration = run.calibrated([case])
    require(calibration == reviewer.independent_calibration([case]), "INDEPENDENT_CALIBRATION_MISMATCH")
    curve = []
    for point in calibration["curve"]:
        supported = run.metrics([case], point["epsilon"], True)
        require(supported == reviewer.independent_accounting([case], point["epsilon"], True), "SUPPORTED_ACCOUNTING_MISMATCH")
        curve.append(dict(**point, supported=supported))
    raw_tp, raw_supported = curve[0]["raw_solver_TP"], curve[0]["supported"]["raw_solver_TP"]
    minimum_tp, minimum_supported = math.ceil(.4 * raw_tp), math.ceil(.4 * raw_supported)
    joint = [p for p in curve if p["filtered_TP"] >= minimum_tp and p["supported"]["filtered_TP"] >= minimum_supported
             and p["retained_count"] and p["supported"]["retained_count"]]
    risk_joint = [p for p in joint if p["FDP"] is not None and p["FDP"] <= .01]
    unresolved = sum(r["full"]["status"] != "BOUNDS_VALID" or r["deleted"]["status"] != "BOUNDS_VALID" for r in case["records"])
    status = ("TECHNICAL_UNRESOLVED" if unresolved or proof_review["abstention_only_proof_count"] else
              "EARLY_STOP_CAL_FUTILITY" if not joint else "EARLY_STOP_FIRST_CAL_RISK_UTILITY_SCREEN" if not risk_joint
              else "FIRST_CAL_SCREEN_PASS_NOT_EVAL_GO")
    best = max(curve, key=lambda p: (p["filtered_TP"], p["supported"]["filtered_TP"], -p["filtered_FP"], p["epsilon"]))
    chosen_supported = run.metrics([case], calibration["epsilon"], True)
    report = dict(status=status, scope="USER_DIRECTED_FIRST_CAL_BUDGET_SCREEN_NOT_OFFICIAL_CAL_OR_EVAL",
                  case=KEY, original_design_fingerprint=DESIGN, numerical_unresolved_identities=unresolved,
                  proof_review=proof_review, raw_counts={k: curve[0][k] for k in ("raw_solver_TP", "raw_solver_FP", "raw_solver_FN")},
                  raw_supported_solver_TP=raw_supported, minimum_TP=minimum_tp, minimum_supported_TP=minimum_supported,
                  maximum_TP_utility_ignoring_FDP=best,
                  maximum_supported_TP_ignoring_FDP=max(p["supported"]["filtered_TP"] for p in curve),
                  standalone_one_percent_choice=dict(status=calibration["status"], **calibration["chosen"], supported=chosen_supported),
                  same_epsilon_joint_utility_exists=bool(joint), same_epsilon_joint_one_percent_utility_exists=bool(risk_joint),
                  joint_utility_epsilon_count=len(joint), joint_one_percent_utility_epsilon_count=len(risk_joint),
                  original_source_unchanged=True, formal_CAL_seal_written=False, EVAL_read=False, additional_GPU_fits=0,
                  interpretation="Exploratory budget decision on the first CAL case; not independent risk control or formal EVAL NO-GO")
    for name, value in original_hashes.items():
        require(sha(args.source_output / "cases" / KEY / name) == value, "SOURCE_CHANGED_DURING_SCREEN")
    immutable(args.output / "threshold_curve.json", curve)
    immutable(args.output / "screening_report.json", report)
    files = list((args.output / "scores" / KEY).iterdir()) + [p for p in args.output.iterdir() if p.is_file() and p.name != "artifact_manifest.json"]
    manifest = {p.relative_to(args.output).as_posix(): dict(sha256=sha(p), bytes=p.stat().st_size) for p in files if p.is_file()}
    immutable(args.output / "artifact_manifest.json", dict(status="SCREEN_REVIEWED", files=manifest,
                                                          retained_source_case=str(args.source_output / "cases" / KEY), source_hashes=original_hashes))
    print(json.dumps(dict(status=status, case=KEY, output=str(args.output))), flush=True)


if __name__ == "__main__":
    main()
