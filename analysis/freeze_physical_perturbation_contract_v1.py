"""Seal existing whole CE change vectors and their supported mapping; no simulation."""
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/physical_perturbation_contract_v1"
OLD = ROOT / "results/ce133_uncertainty_v1_ready"
JOINT = ROOT / "results/ce133_joint_variation_audit"
DOC = ROOT / "docs/PHYSICAL_PERTURBATION_CONTRACT_V1.md"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2, ensure_ascii=False,
                                      allow_nan=False) + "\n", encoding="utf-8", newline="\n")


def main():
    old_prov, joint_prov = read(OLD / "provenance.json"), read(JOINT / "provenance.json")
    inputs = []
    for folder, prov, names in [
        (OLD, old_prov, ["contract.json", "components.json", "component_fractions.npz"]),
        (JOINT, joint_prov, ["observed_joint_changes.json", "exact_support_joint_statistics.json"]),
    ]:
        for name in names:
            path = folder / name
            assert sha(path) == prov["outputs"][name], path
            inputs.append(path)
    minimum = read(OLD / "contract.json")["minimum_distinct_identities"]
    assert minimum == 10
    strata = [r for r in read(JOINT / "exact_support_joint_statistics.json")
              if r["distinct_identity_count"] >= minimum]
    eligible = {(r["rule"], tuple(r["ce_pair"]), tuple(r["channels"])) for r in strata}
    patterns = []
    for row in read(JOINT / "observed_joint_changes.json"):
        key = (row["identity"][2], tuple(row["ce_pair"]), tuple(row["channels"]))
        if key not in eligible:
            continue
        pattern_key = [row["identity"], row["source_low"], row["source_high"], row["channels"]]
        pid = hashlib.sha256(json.dumps(pattern_key, separators=(",", ":"),
                                        ensure_ascii=True).encode()).hexdigest()
        multiplier = [math.exp(v) for v in row["centered_log_change"]]
        assert all(math.isfinite(v) and v > 0 for v in multiplier)
        assert abs(sum(row["centered_log_change"])) < 1e-12
        patterns.append(dict(pattern_id=pid, donor_identity=row["identity"],
            source_low=row["source_low"], source_high=row["source_high"],
            source_files=row["source_files"], ce_pair=row["ce_pair"], channels=row["channels"],
            centered_log_change=row["centered_log_change"], multiplier=multiplier))
    patterns.sort(key=lambda r: r["pattern_id"])
    assert len({r["pattern_id"] for r in patterns}) == len(patterns) == 79
    component_rows = read(OLD / "components.json")
    by_candidate = defaultdict(list)
    for index, row in enumerate(component_rows):
        by_candidate[row["candidate_index"]].append((index, row))
    mapping = []
    for candidate, rows in sorted(by_candidate.items()):
        channels = sorted(r["channel"] for _, r in rows)
        assert len(channels) == len(set(channels))
        rule = rows[0][1]["rule"]
        options = [r for r in patterns if r["donor_identity"][2] == rule and r["channels"] == channels]
        if not options:
            continue
        component_index = {r["channel"]: index for index, r in rows}
        mapping.append(dict(candidate_index=candidate, candidate_id=rows[0][1]["candidate_id"],
            lipid_name=rows[0][1]["lipid_name"], rule=rule, channels=channels,
            component_indices=[component_index[ch] for ch in channels],
            allowed_pattern_ids=[r["pattern_id"] for r in options]))
    varied = {r["candidate_index"] for r in mapping}
    assert len(varied) == 69 and all(r["rule"] == "PEO-H" for r in mapping)
    assert len(strata) == 5
    library_entry = next(r for r in old_prov["inputs"] if r["path"].endswith("A_library.npy"))
    contract = dict(
        contract_id="PHYSICAL_PERTURBATION_CONTRACT_V1", status="FROZEN_SUPPORTED_SCOPE",
        scope="CE-informed controlled systematic variation of positive shared-fragment intensities",
        nominal_library_asset=library_entry,
        supported_candidates=69, fixed_candidates=322, production_candidate_count=391,
        fixed_candidate_indices=sorted(set(range(391)) - varied),
        whole_pattern_count=79, exact_stratum_count=5,
        minimum_distinct_identities=10,
        minimum_interpretation="Inherited n reference declared as this contract's inclusion policy; not coverage guarantee",
        allowed_strata=[{k: r[k] for k in ("rule", "ce_pair", "channels", "distinct_identity_count")} for r in strata],
        systematic=dict(status="SUPPORTED_FINITE_JOINT_ENDPOINTS_WITH_TRANSFER_ASSUMPTION",
            shared_not_automatically_strong=True, independent_fragment_box=False,
            direction="observed low CE to high CE only", nominal_allowed=True,
            interpolation=False, extrapolation=False, covariance_sampling=False,
            arbitrary_severity_rescaling=False, new_support=False, dropout=False,
            one_donor_ce_pair_per_dataset=True, one_whole_vector_per_identity_fixed_over_pixels=True,
            duplicate_rule_identity_channel_set_shares_donor=True,
            isotope_profile_coupling="Reuse original physical component fractions",
            transform="t = a - sum(D_c) + sum(exp(delta_c)*D_c); a_target = norm2(a)*t/norm2(t)",
            nominal_transform="Return original a unchanged",
            fixed_part="Unperturbed before common column normalization; no absolute precursor invariance claim",
            abundance="No per-identity X_true scaling; retain production spatial recipe and one dataset scalar"),
        weak_peak_censoring=dict(status="NOT_ESTIMATED", simulated=False, estimated_zero=False),
        pixel_fluctuation=dict(status="NOT_SEPARATELY_IDENTIFIABLE", simulated=False,
            scope="Currently verified evidence only; not a general impossibility claim",
            smaller_than_systematic="NOT_ESTABLISHED", estimated_zero=False),
        measurement_noise=dict(status="NOT_ESTIMATED", simulated=False, gaussian_fallback=False),
        donor_isolation=dict(
            required_roles=["MODEL", "CAL", "EVAL"],
            grouping="All CE records for a donor identity plus transitively shared source files stay together",
            freeze_membership="During the single pilot preparation, before generation/scoring",
            inference_dictionary="MODEL donors only; no CAL/EVAL endpoints or target labels",
            full_bank_is_not_inference_dictionary=True,
            exact_target_in_inference="Representation check only, not unseen-mismatch GO",
            production_training_exposure="Present historically; developmental inference only"),
        confidence_boundary=dict(
            new_score=False, frozen_rho_changed=False, same_observation_full_delete=True,
            molecular_deletion=True,
            cone_relaxation="Not physical endpoint set; deletion lower bound may be conservative",
            full_acceptance="Requires feasible witness in physical single-endpoint/common-condition model",
            invalid_numerical_proof="ABSTAIN"),
        endpoint=dict(empirical_FDP_max=.01, TP_retention_min=.40, strong_success_retention=.60,
            report_uncertainty=True, retained_truth_stays_in_original_denominator=True,
            report_supported_perturbed_truth_separately=True,
            unperturbed_truth_cannot_establish_mismatch_robustness=True,
            missing_library_required=True),
        next_task="主线 identity robustness pilot", additional_prerequisite_audit=False,
        pixel_or_censoring_estimation_blocks_pilot=False,
        V2_closed=True, V2_1_allowed=False, new_architecture=False, new_K_ladder=False,
        new_spatial_variant=False, new_CE_benchmark=False, outcome_driven_uncertainty_changes=False)
    OUT.mkdir(parents=True, exist_ok=False)
    write("joint_patterns.json", patterns)
    write("candidate_mapping.json", mapping)
    write("contract.json", contract)
    inputs += [OLD / "provenance.json", JOINT / "provenance.json", DOC, Path(__file__)]
    content = dict(
        contract_id=contract["contract_id"],
        inputs=[dict(path=p.relative_to(ROOT).as_posix(), bytes=p.stat().st_size, sha256=sha(p)) for p in inputs],
        outputs={p.name: sha(p) for p in sorted(OUT.iterdir()) if p.is_file()},
        no_synthetic_data_generated=True, no_solver_or_outcomes_read=True)
    encoded = json.dumps(content, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    write("seal.json", dict(content=content, fingerprint=hashlib.sha256(encoded).hexdigest()))
    print(json.dumps(dict(status=contract["status"], patterns=len(patterns),
                         candidates=len(mapping), fingerprint=hashlib.sha256(encoded).hexdigest())))


if __name__ == "__main__":
    main()
