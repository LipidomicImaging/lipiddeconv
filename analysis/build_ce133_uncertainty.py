"""One bounded CE133 estimate and physical fragment-component export; no fits."""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import itertools
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def read(p):
    return json.loads(p.read_text(encoding="utf-8"))


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def write(p, value):
    p.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                 encoding="utf-8", newline="\n")


def trusted(row, channel):
    if float(row.get("mask__" + channel) or 0) != 1:
        return False
    if float(row.get("label__" + channel) or 0) <= 0:
        return False
    peaks = json.loads(row.get("channel_peak_indices_json") or "{}").get(channel, [])
    ownership = json.loads(row.get("ownership_allocations_json") or "{}")
    return bool(peaks) and all(
        str(i) in ownership and abs(float(ownership[str(i)]["share"]) - 1) < 1e-12
        and ownership[str(i)]["channels"] == [channel] for i in peaks)


def main(args):
    audit_root = ROOT / "results/rho_mismatch_mechanism_ce_audit"
    provenance = read(audit_root / "provenance.json")
    target = Path(next(r["path"] for r in provenance["inputs"] if r["path"].endswith(".csv")
                       and "target_existing" in r["path"]))
    expected = next(r["sha256"] for r in provenance["inputs"] if r["path"] == str(target))
    assert sha(target) == expected
    full_prov = read(ROOT / "results/v59_standardized_cross_library_inventory/full_library_provenance.json")
    prediction = Path(next(p for p in full_prov["paths"] if p.endswith("channel_predictions_ce29.npy"))).parent
    prod = Path(full_prov["production_reference_assets"]["A_library.npy"]["path"]).parent
    channels = read(prediction / "channels.json")[1:]
    manifest = read(audit_root / "ce_identity_records.json")
    allowed_keys = {tuple(x["identity"][k] for k in ("structure_text", "adduct", "rule_sheet"))
                    for x in manifest if len(x["collision_energies_ev"]) >= 2}
    assert len(allowed_keys) == 133
    with target.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    groups = defaultdict(list)
    for row in rows:
        key = tuple(row[k] for k in ("structure_text", "adduct", "rule_sheet"))
        if key in allowed_keys and float(row["ce"]) in (30, 35, 40):
            groups[key].append(row)
    assert len(groups) == 133
    # Fix this estimator before inspecting new screening outcomes: no tuning grid.
    contract = dict(version="CE133_SHARED_FRAGMENT_BOX_V1", minimum_distinct_identities=10,
        radius="95th percentile of per-identity maximum absolute centered log ratio over available CE pairs",
        centering="subtract mean log ratio over >=2 trusted channels shared by both observations",
        trusted="positive active label; every assigned peak wholly owned and assigned to this one channel",
        directions="both CE directions included via a symmetric log envelope", allowed="existing fragment intensity only",
        fixed="precursor, zeros, unassigned/ambiguous components, and rule/channel strata with <10 identities",
        transfer="CE-conditioned annotated DDA variation transferred to predicted MSI library; developmental assumption",
        joint_model="independent component box outer envelope; observed co-variation reported, not falsely guaranteed",
        mapping="unambiguous predicted channel + nominal isotope0..4 inside existing profile bin; coupled peak response",
        no_measurement_noise_model=True, no_new_score_in_builder=True)
    deltas = []
    statuses = Counter()
    maxima = defaultdict(dict)
    for key, part in sorted(groups.items()):
        part.sort(key=lambda r: float(r["ce"]))
        for lo, hi in itertools.combinations(part, 2):
            common = []
            for ch in channels:
                active = [float(r.get("mask__" + ch) or 0) == 1 for r in (lo, hi)]
                positive = [float(r.get("label__" + ch) or 0) > 0 for r in (lo, hi)]
                statuses["both_active" if all(active) else "masked_or_inapplicable"] += 1
                if all(active) and positive[0] != positive[1]:
                    statuses["observed_presence_change_not_validated_dropout"] += 1
                if not all(trusted(r, ch) for r in (lo, hi)):
                    continue
                masses = [float(r["targetmz__" + ch]) for r in (lo, hi)]
                if not all(math.isfinite(x) for x in masses) or abs(masses[0] - masses[1]) > 1e-6:
                    statuses["unaligned_target_mass"] += 1
                    continue
                common.append(ch)
            if len(common) < 2:
                statuses["pair_insufficient_shared_trusted_channels"] += 1
                continue
            ratios = {ch: math.log(float(hi["label__" + ch]) / float(lo["label__" + ch])) for ch in common}
            center = sum(ratios.values()) / len(ratios)
            centered = {ch: val - center for ch, val in ratios.items()}
            for ch, val in centered.items():
                bucket = maxima[(key[2], ch)]
                bucket[key] = max(bucket.get(key, 0), abs(val))
            deltas.append(dict(identity=list(key), source_low=lo["row_uid"], source_high=hi["row_uid"],
                               ce_low=float(lo["ce"]), ce_high=float(hi["ce"]), centered_log_changes=centered))
    bounds = []
    for (rule, channel), observations in sorted(maxima.items()):
        radius = float(np.quantile(list(observations.values()), .95, method="linear"))
        supported = len(observations) >= 10
        bounds.append(dict(rule=rule, channel=channel, distinct_identities=len(observations),
                           observed_log_radius_q95=radius, enabled=supported,
                           lower=math.exp(-radius) if supported else 1.,
                           upper=math.exp(radius) if supported else 1.))
    by_key = {(r["rule"], r["channel"]): r for r in bounds if r["enabled"]}
    correlations = []
    for rule in sorted({r["identity"][2] for r in deltas}):
        for first, second in itertools.combinations(channels, 2):
            per_identity = defaultdict(list)
            for row in deltas:
                change = row["centered_log_changes"]
                if row["identity"][2] == rule and first in change and second in change:
                    per_identity[tuple(row["identity"])].append([change[first], change[second]])
            if len(per_identity) >= 10:
                a = np.array([np.mean(v, axis=0) for v in per_identity.values()])
                corr = float(np.corrcoef(a.T)[0, 1]) if np.all(np.std(a, axis=0) > 0) else None
                correlations.append(dict(rule=rule, first=first, second=second, n=len(a),
                                         correlation=corr, centered_compositional_correlation=True))

    # Reuse the original stick library and positive instrumental response; no new spectra synthesized.
    paths = [prod / n for n in ("spectrum_source_STICK.npy", "peak_response_matrix.npz",
             "candidate_metadata_final.jsonl", "profile_bin_left.npy", "profile_bin_right.npy",
             "shared_mz_source_sticks.npy", "A_library.npy")]
    paths += [prediction / n for n in ("metadata_ce29_full.jsonl", "channel_target_mz.npy", "channel_masks.npy", "channels.json")]
    metadata = [json.loads(line) for line in paths[2].read_text(encoding="utf-8").splitlines()]
    source_meta = [json.loads(line) for line in (prediction / "metadata_ce29_full.jsonl").read_text(encoding="utf-8").splitlines()]
    source_index = {r["entry_id"]: i for i, r in enumerate(source_meta)}
    all_channels = read(prediction / "channels.json")
    target_mz = np.load(prediction / "channel_target_mz.npy")
    masks = np.load(prediction / "channel_masks.npy")
    sticks = np.load(paths[0]).T.astype(float)
    packed = np.load(paths[1]); assert packed["format"].item() == b"csr"
    response = np.zeros(tuple(packed["shape"]), dtype=float)
    for i in range(response.shape[0]):
        k = slice(packed["indptr"][i], packed["indptr"][i + 1])
        response[i, packed["indices"][k]] = packed["data"][k]
    left, right = np.load(paths[3]).reshape(-1), np.load(paths[4]).reshape(-1)
    axis = np.load(paths[5]).reshape(-1)
    raw = np.load(paths[6]).reshape(1084, 391).astype(float)
    reconstructed = response @ sticks
    coefficient_path = prod / "peak_channel_coefficient.npy"
    coefficient = np.load(coefficient_path).reshape(-1)
    assert np.array_equal(coefficient, np.ones(1084)), "Unsupported production channel weighting"
    normalized_reconstruction = reconstructed / np.linalg.norm(reconstructed, axis=0)
    assert np.allclose(normalized_reconstruction, raw, rtol=2e-5, atol=1e-8), "Original normalized stick/response parity failed"
    paths += [coefficient_path, prod.parents[1] / "process_758_two_pass.py"]
    components, members = [], []
    for j, meta in enumerate(metadata):
        if meta["selected_source_entry_id"] not in source_index:
            statuses["production_candidate_outside_known_prediction_source_kept_fixed"] += 1
            continue
        src = source_index[meta["selected_source_entry_id"]]
        assignment = defaultdict(list)
        for i in np.flatnonzero(sticks[:, j] > 0):
            if 748 <= axis[i] <= 803:
                continue
            matches = []
            for c, ch in enumerate(all_channels):
                if ch == "precursor" or masks[src, c] <= 0 or not np.isfinite(target_mz[src, c]):
                    continue
                if any(left[i] <= target_mz[src, c] + rank * 1.003354835 <= right[i] for rank in range(5)):
                    matches.append(ch)
            if len(matches) == 1 and (meta["rule_sheet"], matches[0]) in by_key:
                assignment[matches[0]].append(i)
        for channel, indices in sorted(assignment.items()):
            partial = response[:, indices] @ sticks[indices, j]
            fraction = np.divide(partial, reconstructed[:, j], out=np.zeros(1084), where=reconstructed[:, j] > 0)
            assert np.all(fraction >= -1e-12) and np.all(fraction <= 1 + 1e-10)
            components.append(np.minimum(fraction, 1))
            members.append(dict(candidate_index=j, candidate_id=meta["candidate_id"],
                lipid_name=meta["lipid_name"], rule=meta["rule_sheet"], channel=channel,
                source_stick_indices=[int(i) for i in indices], **{k: by_key[(meta["rule_sheet"], channel)][k] for k in ("lower", "upper")}))
    assert components, "No supported mapped fragment components"
    fractions = np.stack(components, axis=1)
    sums = np.zeros_like(sticks)
    for i, member in enumerate(members):
        sums[:, member["candidate_index"]] += fractions[:, i]
    assert np.max(sums) <= 1 + 1e-10
    args.output.mkdir(parents=True, exist_ok=False)
    write(args.output / "contract.json", contract)
    write(args.output / "bounds.json", bounds)
    write(args.output / "paired_changes.json", deltas)
    write(args.output / "co_variation.json", correlations)
    write(args.output / "components.json", members)
    np.savez_compressed(args.output / "component_fractions.npz", fractions=fractions)
    write(args.output / "summary.json", dict(status="CE_BOUNDARY_READY_FOR_PILOT", matched_identities=133,
        eligible_pairs=len(deltas), identities_with_eligible_pair=len({tuple(r["identity"]) for r in deltas}),
        enabled_rule_channels=len(by_key), components=len(members),
        production_candidates_with_variation=len({r["candidate_index"] for r in members}),
        observation_status_counts=dict(statuses), stick_response_max_abs_difference=float(np.max(abs(normalized_reconstruction - raw))),
        physical_mapping="Candidate entry ID -> predicted fragment mass -> unambiguous source stick -> original response",
        restriction="No peak appearance/disappearance, m/z shifts, precursor variability or empirical measurement noise",
        limitation="Shared-fragment CE variation only; coverage of independent prediction error not established"))
    write(args.output / "provenance.json", dict(inputs=[dict(path=str(p.resolve()), bytes=p.stat().st_size, sha256=sha(p))
        for p in [target, audit_root / "ce_identity_records.json", Path(__file__), *paths]],
        outputs={p.name: sha(p) for p in args.output.iterdir() if p.is_file()}))
    print(json.dumps(read(args.output / "summary.json")))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "results/ce133_uncertainty_v1_ready")
    main(parser.parse_args())
