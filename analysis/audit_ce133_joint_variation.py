"""Describe existing CE joint changes; do not fit an uncertainty or identity model."""
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "results/ce133_uncertainty_v1_ready"
OUTPUT = ROOT / "results/ce133_joint_variation_audit"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(name, value):
    (OUTPUT / name).write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8", newline="\n")


def main():
    provenance = read(SOURCE / "provenance.json")
    names = ["paired_changes.json", "co_variation.json", "contract.json", "summary.json"]
    inputs = [SOURCE / name for name in names]
    for path in inputs:
        assert sha(path) == provenance["outputs"][path.name], path
    target_entry = next(r for r in provenance["inputs"] if r["path"].endswith(".csv"))
    target = Path(target_entry["path"])
    assert sha(target) == target_entry["sha256"]
    builder = ROOT / "analysis/build_ce133_uncertainty.py"
    assert sha(builder) == next(r["sha256"] for r in provenance["inputs"]
                                if r["path"].endswith("build_ce133_uncertainty.py"))
    inputs += [SOURCE / "provenance.json", target, builder,
               ROOT / "analysis/run_ce_uncertainty_identity_pilot.py",
               ROOT / "results/rho_mismatch_mechanism_ce_audit/ce_asset_audit.json",
               Path(__file__)]
    pairs = read(SOURCE / "paired_changes.json")
    wanted = {r[k] for r in pairs for k in ("source_low", "source_high")}
    rows = {}
    with target.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["row_uid"] in wanted:
                assert row["row_uid"] not in rows
                rows[row["row_uid"]] = row
    assert set(rows) == wanted

    strata, identities = defaultdict(list), defaultdict(list)
    records, strongest_changes = [], []
    parity_error = 0.
    for pair in pairs:
        channels = sorted(pair["centered_log_changes"])
        lo, hi = rows[pair["source_low"]], rows[pair["source_high"]]
        values = np.array([[float(row["label__" + ch]) for ch in channels] for row in (lo, hi)])
        assert np.all(np.isfinite(values)) and np.all(values > 0)
        change = np.log(values[1] / values[0])
        change -= change.mean()
        saved = np.array([pair["centered_log_changes"][ch] for ch in channels])
        parity_error = max(parity_error, float(np.max(np.abs(change - saved))))
        assert np.allclose(change, saved, rtol=0, atol=1e-12)
        strongest = np.flatnonzero(values[0] == values[0].max())
        strongest_changes.extend(change[strongest].tolist())
        key = (pair["identity"][2], pair["ce_low"], pair["ce_high"], tuple(channels))
        strata[key].append(pair)
        identities[tuple(pair["identity"])].append(pair)
        records.append(dict(
            identity=pair["identity"], source_low=pair["source_low"], source_high=pair["source_high"],
            ce_pair=[pair["ce_low"], pair["ce_high"]], channels=channels,
            source_files=[lo["source_file"], hi["source_file"]],
            shared_channel_labels=values.tolist(),
            relative_to_strongest_shared_channel=(values / values.max(axis=1)[:, None]).tolist(),
            centered_log_change=change.tolist(),
            lower_ce_strongest_shared_channels=[channels[i] for i in strongest],
            interpretation="Paired selected positive channels; not a strong/weak cutoff or a non-detection sample"))

    joint = []
    for key, part in sorted(strata.items()):
        rule, ce_low, ce_high, channels = key
        assert len({tuple(r["identity"]) for r in part}) == len(part)
        matrix = np.array([[r["centered_log_changes"][ch] for ch in channels] for r in part])
        centered = matrix - matrix.mean(axis=0)
        singular = np.linalg.svd(centered, compute_uv=False)
        energy = singular ** 2
        record = dict(rule=rule, ce_pair=[ce_low, ce_high], channels=list(channels),
            distinct_identity_count=len(part),
            identities=[r["identity"] for r in part],
            source_pairs=[[r["source_low"], r["source_high"]] for r in part],
            mean_centered_log_change=matrix.mean(axis=0).tolist(),
            forced_rank_upper_bound=min(len(part) - 1, len(channels) - 1),
            singular_values=singular.tolist(),
            empirical_variance_fraction=(energy / energy.sum()).tolist() if energy.sum() > 1e-24 else None,
            covariance=(centered.T @ centered / (len(part) - 1)).tolist() if len(part) > 1 else None,
            interpretation="Descriptive only: compositional closure, selected support, shared source files and small n; no physical rank/coverage claim")
        joint.append(record)

    triple_count = 0
    composition_errors = []
    for part in identities.values():
        by_ce = {(r["ce_low"], r["ce_high"]): r for r in part}
        if not all(k in by_ce for k in ((30., 35.), (35., 40.), (30., 40.))):
            continue
        triple_count += 1
        maps = [by_ce[k]["centered_log_changes"] for k in ((30., 35.), (35., 40.), (30., 40.))]
        common = sorted(set(maps[0]) & set(maps[1]) & set(maps[2]))
        if len(common) < 2:
            continue
        # Recenter on identical support only to check the algebraic dependence of CE pairs.
        a = np.array([[r[ch] for ch in common] for r in maps])
        a -= a.mean(axis=1)[:, None]
        composition_errors.append(float(np.max(abs(a[0] + a[1] - a[2]))))
    assert not composition_errors or max(composition_errors) < 1e-12
    contract = read(SOURCE / "contract.json")
    # This is the existing v1 sample-count reference, not a newly selected model gate.
    old_n = contract["minimum_distinct_identities"]
    old_corr = read(SOURCE / "co_variation.json")
    examples = [r for r in old_corr if r["rule"] == "PEO-H" and
                r["first"] == "sn2_fragment" and r["second"] == "sn2_fragment_co2"]
    summary = dict(
        status="BOUNDED_DESCRIPTIVE_AUDIT_COMPLETE_PHYSICAL_MODEL_NOT_FROZEN",
        source_pair_count=len(pairs), source_identity_count=len(identities),
        source_row_count=len(rows), exact_ce_rule_support_strata=len(joint),
        stratum_size_histogram={str(k): v for k, v in sorted(Counter(len(v) for v in strata.values()).items())},
        strata_at_existing_v1_n_reference=[{k: r[k] for k in
            ("rule", "ce_pair", "channels", "distinct_identity_count")} for r in joint if r["distinct_identity_count"] >= old_n],
        old_n_reference=old_n, source_centered_change_max_abs_error=parity_error,
        identities_with_all_three_eligible_pairs=triple_count,
        triples_with_common_support_at_least_two=len(composition_errors),
        ce_pair_composition_max_abs_error=max(composition_errors) if composition_errors else None,
        strongest_shared_lower_ce_change=dict(
            count_including_ties=len(strongest_changes), minimum=float(min(strongest_changes)),
            median=float(np.median(strongest_changes)), maximum=float(max(strongest_changes)),
            units="log intensity ratio after geometric-mean scaling over shared channels",
            limitation="Selected positive shared channels only; cannot estimate strong-peak disappearance or define censoring"),
        existing_descriptive_correlation_example=examples,
        conclusions=dict(
            independent_fragment_box_confirmed=True,
            isotope_and_profile_coupling_preserved=True,
            joint_co_variation_unused_by_v1=True,
            box_caused_v2_failure="NOT_VERIFIED",
            physical_low_dimensional_rank="NOT_VERIFIED",
            strong_fragment_definition_and_detection_stability="NOT_VERIFIED",
            weak_fragment_censoring="NOT_VERIFIED",
            pixel_variation_and_smaller_than_systematic="NOT_VERIFIED",
            independent_library_to_experiment_error_coverage="NOT_VERIFIED"),
        no_solver_or_simulation_run=True, no_uncertainty_model_fit=True,
        no_parameter_selection_or_v2_changes=True)
    OUTPUT.mkdir(parents=True, exist_ok=False)
    write("observed_joint_changes.json", records)
    write("exact_support_joint_statistics.json", joint)
    write("summary.json", summary)
    write("provenance.json", dict(
        inputs=[dict(path=str(p.resolve()), bytes=p.stat().st_size, sha256=sha(p)) for p in inputs],
        outputs={p.name: sha(p) for p in OUTPUT.iterdir() if p.is_file()}, numpy_version=np.__version__))
    print(json.dumps({k: summary[k] for k in ("status", "source_pair_count", "source_identity_count",
          "exact_ce_rule_support_strata", "source_centered_change_max_abs_error",
          "triples_with_common_support_at_least_two", "ce_pair_composition_max_abs_error")}))


if __name__ == "__main__":
    main()
