"""Bounded cached-data audit; no model import, fitting, scoring or raw processing."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
KEY = ("structure_text", "adduct", "rule_sheet")
CES = (30, 35, 40)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def source(path):
    return dict(path=str(path.resolve()), bytes=path.stat().st_size,
                sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                    encoding="utf-8", newline="\n")


def main(args):
    provenance_path = ROOT / "results/v59_standardized_cross_library_inventory/full_library_provenance.json"
    provenance = read_json(provenance_path)
    # Follow exactly one already documented output and its direct training-input reference.
    prediction_path = Path(next(p for p in provenance["paths"]
                                if Path(p).name == "channel_predictions_ce29.npy"))
    report_path = prediction_path.parent / "training_report.json"
    report = read_json(report_path)
    target_path = Path(report["target_csv"])
    trainer_path = prediction_path.parents[2] / "train_ce_response_adapter_v25.py"
    rows = read_csv(target_path)
    assert len(rows) == report["target_rows"]
    assert all(all(row[k] for k in KEY) for row in rows)
    assert all(float(row["ce"]) == float(row["collision_energy_ev"]) for row in rows)
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[k] for k in KEY)].append(row)
    assert len(groups) == report["target_unique_identities"]
    selected = [row for row in rows if float(row["ce"]) in CES]
    ce_sets = {ce: {tuple(row[k] for k in KEY) for row in selected if float(row["ce"]) == ce}
               for ce in CES}
    counts = Counter((*tuple(row[k] for k in KEY), float(row["ce"])) for row in selected)
    instruments = [key for key in rows[0] if any(token in key.lower()
                   for token in ("instrument", "vendor", "analyzer", "standard"))]
    identities = []
    for key, parts in sorted(groups.items()):
        parts = [row for row in parts if float(row["ce"]) in CES]
        if not parts:
            continue
        identities.append(dict(
            identity=dict(zip(KEY, key)),
            collision_energies_ev=sorted({float(row["ce"]) for row in parts}),
            records=[{field: row[field] for field in (
                "row_uid", "entry_id", "source_file", "collision_energy_ev", "replicate_id",
                "lipid_class", "source_family", "quality_label", "ambig_group_count",
                "mixture_order_in_file", "v18f_split", "ce_assignment_basis")}
                for row in sorted(parts, key=lambda row: float(row["ce"]))]))
    ce_audit = dict(
        status="MATCHED_MULTI_CE_ANNOTATED_RECORDS_VERIFIED_NOT_INDEPENDENT_VALIDATION",
        identity_key=list(KEY), exact_key_no_new_canonicalization=True,
        total_rows=len(rows), total_annotated_identities=len(groups),
        ce_row_counts=dict(Counter(row["collision_energy_ev"] for row in rows)),
        selected_ce_rows=len(selected), selected_ce_union_identities=len(set.union(*ce_sets.values())),
        per_ce_identity_counts={str(ce): len(keys) for ce, keys in ce_sets.items()},
        pairwise_identity_overlap={f"{a}_{b}": len(ce_sets[a] & ce_sets[b])
                                   for a, b in ((30, 35), (30, 40), (35, 40))},
        identities_at_all_three=len(set.intersection(*ce_sets.values())),
        identities_at_least_two=sum(len(item["collision_energies_ev"]) >= 2 for item in identities),
        same_identity_ce_cells_with_multiple_rows=sum(n > 1 for n in counts.values()),
        source_family_counts=dict(Counter(row["source_family"] for row in selected)),
        adduct_row_counts=dict(Counter(row["adduct"] for row in selected)),
        source_file_candidate_count_histogram=dict(Counter(Counter(row["source_file"] for row in selected).values())),
        instrument_or_standard_metadata_columns=instruments,
        production_adapter_training_exposure="YES: final models use all_rows; cached OOF predictions also exist",
        not_verified=[
            "Raw spectrum files, acquisition provenance and CE units beyond this table's eV labels",
            "Independent identity/adduct confirmation or authentic-standard ground truth",
            "Independent same-CE repeatability; this retained table has one row per identity/CE",
            "Full base-predictor and preprocessing train/test independence of cached OOF predictions",
            "Channel alignment, ambiguous peak allocation and unbiased library prediction-error coverage",
        ],
        limitations=[
            "These are selected rule-annotated DDA channel records, not 389 verified pure reference spectra.",
            "Multiple candidate records may share a source spectrum; shared peaks are not independent observations.",
            "Cross-energy records characterize condition dependence, not same-condition measurement noise.",
            "The production adapter has already trained on these rows: they cannot be called unseen final validation.",
            "Earlier EMPIRICAL_VARIABILITY_NOT_VERIFIED inventory was a limited search result, not proof of absent assets.",
        ])

    diagnostic = ROOT / "results/cal_rho_numerical_diagnostic"
    channel_path = diagnostic / "channel_loss_decomposition.csv"
    fit_path = diagnostic / "fit_diagnostics.csv"
    channel_rows = read_csv(channel_path)
    fit_rows = read_csv(fit_path)
    # known_mismatch is already cached as MILD minus CLEAN foreground-mean b.
    blocks = defaultdict(dict)
    for row in channel_rows:
        key = (row["candidate_index"], row["domain"])
        index = int(row["channel_index"])
        assert index not in blocks[key]
        blocks[key][index] = float(row["known_mismatch"])
    reference = blocks[("8", "CLEAN")]
    assert set(reference) == set(range(1084))
    assert all(block == reference for block in blocks.values())
    chosen = [row for row in fit_rows if row["method"] == "scipy_nnls_original_order"
              and row["domain"] == "CLEAN" and row["candidate_index"] in ("8", "10")]
    assert len(chosen) == 2
    energy = float(chosen[0]["signal_energy_plus_epsilon"]) - 1e-12
    assert all(float(row["signal_energy_plus_epsilon"]) - 1e-12 == energy for row in chosen)
    mismatch_norm2 = math.fsum(value * value for value in reference.values())
    mechanism = dict(
        status="CACHED_DIAGNOSTIC_SCALE_COMPARISON_NOT_CAUSAL_ATTRIBUTION",
        dataset="CAL_R1_K125", domain="1084-channel foreground-mean spectrum; same domain as rho",
        clean_signal_norm2=energy, cached_mild_minus_clean_norm2=mismatch_norm2,
        relative_mean_spectrum_change=math.sqrt(mismatch_norm2 / energy),
        selected_true_deletion_margins=[dict(
            candidate_index=int(row["candidate_index"]), lipid_name=row["lipid_name"],
            clean_rho=float(row["rho"]), clean_q_full=float(row["q_full"]),
            normalized_deleted_distance=math.sqrt(float(row["q_deleted"]) / energy),
            sqrt_rho=math.sqrt(float(row["rho"]))) for row in chosen],
        interpretation="q_full is almost zero, so sqrt(rho) approximates normalized deletion distance. "
                       "Total perturbation exceeds these two margins, but its direction matters; this is not a "
                       "robustness radius, a causal proof, or population-level prevalence.",
        toy_example=dict(database_true=[1, 1], database_absent=[0, 1], observed_clean=[1, 1],
                         observed_shifted_true=[1, 1.2], shifted_false_coefficient=0.2,
                         full_sse=0, delete_false_sse=0.02,
                         false_rho=0.02 / (2.44 + 1e-12),
                         status="ANALYTIC_ILLUSTRATION_NOT_EXPERIMENTAL_DATA"))
    inputs = [provenance_path, report_path, target_path, trainer_path, channel_path, fit_path,
              ROOT / "src/rho_zero.py", ROOT / "analysis/run_cal_rho_numerical_diagnostic.py",
              ROOT / "results/v58_clean_mild_paired_review/analysis_record.md",
              diagnostic / "analysis_record.md", Path(__file__)]
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / "ce_asset_audit.json", ce_audit)
    write_json(args.output / "ce_identity_records.json", identities)
    write_json(args.output / "cached_mechanism_scales.json", mechanism)
    write_json(args.output / "provenance.json", dict(
        inputs=[source(path) for path in inputs],
        scope="Known production provenance -> training report -> one directly referenced 2.2 MB CSV and trainer; "
              "two cached diagnostic tables. No recursive scan, model import, fit, rho or simulation.",
        output_hashes={name: source(args.output / name) for name in (
            "ce_asset_audit.json", "ce_identity_records.json", "cached_mechanism_scales.json")}))
    print(json.dumps(dict(status="PASS", ce_pairs=ce_audit["pairwise_identity_overlap"],
                          all_three=ce_audit["identities_at_all_three"],
                          relative_mean_spectrum_change=mechanism["relative_mean_spectrum_change"])))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "results/rho_mismatch_mechanism_ce_audit")
    main(parser.parse_args())
