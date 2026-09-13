"""Independent, label-free audit of completed cached identity evidence.

Uses four squared losses in two deletion orders, not the extractor's direct
dot-product formula. No extractor, optimizer, rho or training module imports.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path

for _key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_key] = "1"

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "results/cached_identity_evidence_v1"
EPS = np.finfo(np.float64).eps
ARRAY_KEYS = {"direct", "redistribution", "delta", "own_energy", "fold_coefficient", "support"}
ROW_KEYS = {"molecular_name", "trimmed_gain", "direct_gain", "direct_per_signal", "stability",
            "own_signal_energy", "direct_informative", "stability_informative", "support_count"}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_config(case):
    parent = ROOT / "results/physical_block_prediction_v1"
    if case in ("DEV_TRAIN", "CHECK"):
        directory = parent / "cases" / case
        return dict(input=parent / "prepared" / (case + ".npz"),
                    manifest=parent / "block_manifest.json",
                    records=parent / "prepared" / (case + "_records.json"),
                    old_features=directory / "features.json", completion=directory / "completion.json",
                    blocks=directory / "blocks")
    require(case == "NEW_COMPOSITION", "UNKNOWN_CASE")
    parent = ROOT / "results/physical_block_score_correction_v2/new_composition_case"
    return dict(input=parent / "prepared/portable.npz", manifest=parent / "prepared/block_manifest.json",
                records=parent / "nnls_rho/molecular_records.json",
                old_features=parent / "physical_features/features.json",
                completion=parent / "physical_features/completion.json",
                blocks=parent / "physical_features/blocks")


def compare_array(actual, expected, bound, label):
    error = np.abs(actual - expected)
    limits = np.broadcast_to(bound, expected.shape)
    require(np.isfinite(actual).all() and np.isfinite(expected).all()
            and np.isfinite(limits).all() and (limits >= 0).all(), "NONFINITE:" + label)
    require(np.all(error <= limits), "NUMERICAL_REPLAY:" + label)
    ratios = np.divide(error, limits, out=np.zeros_like(error), where=limits > 0)
    return dict(max_absolute_error=float(error.max(initial=0.)),
                max_error_to_bound_ratio=float(ratios.max(initial=0.)),
                exact_different_cells=int(np.count_nonzero(error)))


def scalar_equal(actual, expected, label, abs_tol=1e-13):
    require(isinstance(actual, (int, float)) and not isinstance(actual, bool)
            and math.isfinite(actual) and math.isfinite(expected)
            and math.isclose(actual, expected, rel_tol=1e-11, abs_tol=abs_tol), "ROW_REPLAY:" + label)


def audit(case, output):
    directory = output / "cases" / case
    completion = read(directory / "feature_completion.json")
    require(completion["status"] == "COMPLETE" and completion["case"] == case, "NOT_COMPLETE")
    require(completion["design_sha256"] == sha(output / "design.json"), "DESIGN_BINDING")
    design = read(output / "design.json")
    require(isinstance(design["code"], dict) and design["code"], "MISSING_CODE_BINDING")
    for path, digest in design["code"].items():
        require(sha(ROOT / path) == digest, "CODE_CHANGED:" + path)
    require(set(completion["artifacts"]) == {"features.json", "feature_arrays.npz"}, "OUTPUT_MEMBERSHIP")
    for path, digest in completion["artifacts"].items():
        require(sha(directory / path) == digest, "OUTPUT_HASH_CHANGED:" + path)
    source = source_config(case)
    paths = [path for key, path in source.items() if key != "blocks"]
    paths += [source["blocks"] / f"block_{k:03d}.{ext}" for k in range(34) for ext in ("npz", "json")]
    expected_hashes = {path.relative_to(ROOT).as_posix(): sha(path) for path in paths}
    require(completion["source_hashes"] == expected_hashes, "EXACT_SOURCE_HASH_MEMBERSHIP")
    original_completion = read(source["completion"])
    require(original_completion["status"] == "COMPLETE", "ORIGINAL_CASE_NOT_COMPLETE")
    manifest = read(source["manifest"])
    names, molecular_names, aliases = (manifest[key] for key in ("candidate_names", "molecular_names", "aliases"))
    require(len(names) == 391 and len(molecular_names) == 377
            and molecular_names == list(dict.fromkeys(names)) and manifest["block_count"] == 34
            and len(manifest["blocks"]) == 34, "ORIGINAL_UNIVERSE")
    require(aliases == [[i for i, name in enumerate(names) if name == molecular]
                        for molecular in molecular_names], "ALL_ALIAS_MEMBERSHIP")
    with np.load(source["input"], allow_pickle=False) as archive:
        A, b = archive["A"].astype(np.float64), archive["b"].astype(np.float64)
    require(A.shape == (1084, 391) and b.shape == (1084,) and np.isfinite(A).all()
            and np.isfinite(b).all() and (A >= 0).all(), "OBSERVABLE_INPUT")
    bnorm2 = math.fsum(float(value)*float(value) for value in b)
    require(bnorm2 > 0, "ZERO_OBSERVATION")
    tau = 128 * EPS * max(A.shape) * bnorm2
    old = read(source["old_features"])
    require([row["molecular_name"] for row in old] == molecular_names, "OLD_FEATURE_MEMBERSHIP")
    rows = read(directory / "features.json")
    require(isinstance(rows, list) and [row["molecular_name"] for row in rows] == molecular_names
            and all(set(row) == ROW_KEYS for row in rows), "NEW_ROW_SCHEMA")
    with np.load(directory / "feature_arrays.npz", allow_pickle=False) as archive:
        require(set(archive.files) == ARRAY_KEYS, "NEW_ARRAY_SCHEMA")
        saved = {key: archive[key] for key in archive.files}
    for key, value in saved.items():
        require(value.shape == (34, 377) and value.dtype == (bool if key == "support" else np.float64)
                and np.isfinite(value).all(), "NEW_ARRAY_SHAPE_DTYPE_FINITE:" + key)
    require((saved["own_energy"] >= 0).all() and (saved["fold_coefficient"] >= 0).all(), "NEGATIVE_OWN_EVIDENCE")
    expected = {key: np.zeros((34, 377), dtype=bool if key == "support" else np.float64) for key in ARRAY_KEYS}
    direct_bound = np.zeros((34, 377)); energy_bound = np.zeros((34, 377)); coef_bound = np.zeros((34, 377))
    loss_replay = []
    for k, block in enumerate(manifest["blocks"]):
        require(block["block_index"] == k, "BLOCK_ORDER")
        held = np.asarray(block["channel_indices"], dtype=np.int64)
        train = np.setdiff1d(np.arange(len(A)), held)
        require(len(held) and len(train) and len(set(held.tolist())) == len(held), "HELD_MEMBERSHIP")
        meta = read(source["blocks"] / f"block_{k:03d}.json")
        npz_path = source["blocks"] / f"block_{k:03d}.npz"
        require(meta["arrays_sha256"] == sha(npz_path) and meta["held_indices"] == held.tolist()
                and meta["train_indices"] == train.tolist() and meta["candidate_names"] == names
                and meta["molecular_names"] == molecular_names and meta["aliases"] == aliases, "SOURCE_BLOCK_BINDING")
        if case in ("DEV_TRAIN", "CHECK"):
            require(meta["source_sha256"] == original_completion["source_sha256"] == sha(source["input"])
                    and meta["design_sha256"] == original_completion["design_sha256"], "V1_SOURCE_BINDING")
        else:
            require(meta["binding"] == original_completion["binding"], "V2_SOURCE_BINDING")
        with np.load(npz_path, allow_pickle=False) as archive:
            xf, xd = archive["full_x"], archive["deleted_x"]
            full_loss, deleted_loss = float(archive["full_held_loss"]), archive["deleted_held_loss"]
            require(archive["deleted_names"].tolist() == molecular_names, "DELETED_ORDER")
            require(xf.shape == (391,) and xd.shape == (377, 391) and xf.dtype == xd.dtype == np.float64
                    and np.isfinite(xf).all() and np.isfinite(xd).all() and (xf >= 0).all() and (xd >= 0).all()
                    and deleted_loss.shape == (377,) and math.isfinite(full_loss)
                    and np.isfinite(deleted_loss).all(), "SOURCE_COEFFICIENTS")
            candidate_train = np.any(A[train] > 0, axis=0)
            candidate_held = np.any(A[held] > 0, axis=0)
            require(np.array_equal(archive["candidate_train_supported"], candidate_train)
                    and np.array_equal(archive["candidate_held_supported"], candidate_held), "CANDIDATE_SUPPORT")
            support = np.asarray([candidate_train[group].any() and candidate_held[group].any() for group in aliases])
            require(np.array_equal(support, archive["molecular_train_supported"] & archive["molecular_held_supported"]),
                    "MOLECULAR_SUPPORT")
            require(np.all(xf[~candidate_train] == 0) and np.all(xd[:, ~candidate_train] == 0)
                    and all(np.all(xd[g, group] == 0) for g, group in enumerate(aliases)), "DELETED_OR_UNSUPPORTED_ALIAS")
        ah, bh = A[held], b[held]
        prediction_full = ah @ xf
        prediction_deleted = ah @ xd.T
        u = np.column_stack([np.sum(ah[:, group] * xf[group], axis=1) for group in aliases])
        # Independently assign the identity's loss change in both deletion orders.
        residual_full = bh - prediction_full
        residual_deleted = bh[:, None] - prediction_deleted
        loss_full = np.sum(residual_full**2)
        loss_deleted = np.sum(residual_deleted**2, axis=0)
        loss_full_without_identity = np.sum((residual_full[:, None] + u)**2, axis=0)
        loss_deleted_with_identity = np.sum((residual_deleted - u)**2, axis=0)
        expected["direct"][k] = .5 * ((loss_full_without_identity - loss_full)
                                     + (loss_deleted - loss_deleted_with_identity))
        expected["delta"][k] = deleted_loss - full_loss
        expected["redistribution"][k] = expected["delta"][k] - expected["direct"][k]
        expected["own_energy"][k] = np.sum(u*u, axis=0)
        expected["fold_coefficient"][k] = [math.fsum(float(xf[j]) for j in group) for group in aliases]
        expected["support"][k] = support
        # A conservative binary64 forward-error allowance for two predictions,
        # residuals and four losses. This is numerical compatibility, not an
        # interval certificate or a new scientific significance threshold.
        operations = 2 * (A.shape[1] + len(held)) + 32
        gamma = operations * EPS / (1 - operations * EPS)
        scale = np.sum(bh*bh) + np.sum(prediction_full**2) + np.sum(prediction_deleted**2, axis=0) + np.sum(u*u, axis=0)
        direct_bound[k] = 32 * gamma * scale
        energy_bound[k] = 32 * gamma * expected["own_energy"][k]
        coef_bound[k] = 32 * EPS * expected["fold_coefficient"][k]
        loss_replay.append(compare_array(deleted_loss - full_loss, loss_deleted - loss_full,
                                         direct_bound[k], "ORIGINAL_DELTA_BLOCK:" + str(k)))
        require(np.array_equal(saved["delta"][k], expected["delta"][k]), "EXACT_SAVED_DELTA")
        for g, row in enumerate(old):
            require(len(row["blocks"]) == 34 and row["blocks"][k]["block_index"] == k
                    and row["blocks"][k]["delta"] == expected["delta"][k, g]
                    and row["blocks"][k]["supported"] == bool(support[g]), "OLD_DELTA_SUPPORT")
        zero_signal = np.all(u == 0, axis=0)
        require(np.all(saved["direct"][k, zero_signal] == 0), "NO_OWN_SIGNAL_DIRECT_MUST_BE_ZERO")
    require(np.array_equal(saved["support"], expected["support"]), "EXACT_SUPPORT_FLAGS")
    require(np.array_equal(saved["redistribution"], saved["delta"] - saved["direct"]), "EXACT_DECOMPOSITION")
    checks = {key: compare_array(saved[key], expected[key], bound, key) for key, bound in (
        ("direct", direct_bound), ("redistribution", direct_bound),
        ("own_energy", energy_bound), ("fold_coefficient", coef_bound))}
    row_max_errors = {key: 0. for key in ("trimmed_gain", "direct_gain", "direct_per_signal", "stability", "own_signal_energy")}
    for g, row in enumerate(rows):
        delta = expected["delta"][:, g]
        # Direct cells have already passed the independent four-loss audit.
        # Aggregate their saved values to avoid amplifying four-loss cancellation
        # by a very small own-energy denominator; this also independently checks
        # the extractor's sums, normalization, asinh and support/stability logic.
        direct_sum = math.fsum(float(v) for v in saved["direct"][:, g])
        own_energy = math.fsum(float(v) for v in saved["own_energy"][:, g])
        t = expected["fold_coefficient"][:, g][expected["support"][:, g]]
        informative = len(t) >= 2 and bool(np.any(t > 0))
        if informative:
            scaled = t / float(np.max(t))
            mean = math.fsum(float(v) for v in scaled) / len(t)
            mean_square = math.fsum(float(v)*float(v) for v in scaled) / len(t)
            stability = mean*mean / mean_square
        else:
            stability = 0.
        values = dict(trimmed_gain=(math.fsum(float(v) for v in delta) - max(0., float(np.max(delta)))) / bnorm2,
                      direct_gain=direct_sum/bnorm2, direct_per_signal=math.asinh(direct_sum/(own_energy+tau)),
                      own_signal_energy=own_energy, stability=stability)
        require(type(row["support_count"]) is int and row["support_count"] == len(t)
                and type(row["direct_informative"]) is bool and row["direct_informative"] == (own_energy > 0)
                and type(row["stability_informative"]) is bool and row["stability_informative"] == informative,
                "EXACT_ROW_FLAGS:" + row["molecular_name"])
        require(row["direct_informative"] == bool(np.any(expected["own_energy"][:, g] > 0)), "INDEPENDENT_ENERGY_FLAG")
        if not informative:
            require(row["stability"] == 0, "UNINFORMATIVE_STABILITY_MUST_BE_ZERO")
        for key, value in values.items():
            scalar_equal(row[key], value, row["molecular_name"] + "." + key,
                         abs_tol=0. if key == "own_signal_energy" else 1e-13)
            row_max_errors[key] = max(row_max_errors[key], abs(row[key]-value))
    return dict(status="PASS", case=case, independent_formula="mean of the two identity-removal-order squared-loss differences",
        candidate_count=391, molecular_count=377, block_count=34, cell_count=34*377,
        source_hashes=expected_hashes, feature_completion_sha256=sha(directory/"feature_completion.json"),
        features_sha256=sha(directory/"features.json"), arrays_sha256=sha(directory/"feature_arrays.npz"),
        design_sha256=completion["design_sha256"], feature_artifact_hashes=completion["artifacts"],
        source_record_labels_read=False, truth_arrays_read=False, solver_or_rho_or_training_called=False,
        extractor_imported=False, KKT_recomputed=False, original_delta_exact=True, support_and_flags_exact=True,
        all_alias_membership_checked=True, bnorm2_independent_fsum=bnorm2, tau_independent=tau,
        array_numeric_compatibility=checks, row_aggregation_max_absolute_errors=row_max_errors,
        original_delta_loss_replay_max_absolute_error=max(r["max_absolute_error"] for r in loss_replay),
        numerical_scope="Binary64 compatibility, not bitwise direct equality: cell bounds 32*gamma_n*(held observation "
            "energy + full/delete prediction energies + own energy), n=2*(391+held_count)+32. "
            "Energy/coefficient checks use relative forward-error bounds; row aggregate rtol1e-11/atol1e-13 "
            "(own_signal_energy has no absolute tolerance; uninformative stability is exactly zero). "
            "No clipping, threshold changes or modification of saved features. Historical exact-tau failures remain unchanged.",
        independent_FDR_claim=False, reviewer_sha256=sha(__file__), numpy_version=np.__version__)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=("DEV_TRAIN", "CHECK", "NEW_COMPOSITION"), required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Root containing design.json and cases/")
    args = parser.parse_args()
    result = audit(args.case, args.output.resolve())
    path = args.output / "cases" / args.case / "independent_feature_review.json"
    data = json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    if path.exists():
        require(path.read_text(encoding="utf-8") == data, "IMMUTABLE_REVIEW_CHANGED")
    else:
        with path.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(data)
    print("INDEPENDENT_FEATURE_PASS", args.case, str(path))


if __name__ == "__main__":
    main()
