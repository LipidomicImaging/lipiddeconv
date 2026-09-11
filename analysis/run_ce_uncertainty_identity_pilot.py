"""Six cached-mixture development contexts; bounded convex identity deletion, no training."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time
import traceback

for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""


def read(p):
    return json.loads(p.read_text(encoding="utf-8"))


def sha(p):
    h = hashlib.sha256()
    with p.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024**2), b""):
            h.update(block)
    return h.hexdigest()


def write(p, value):
    p.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n")


def model(A, fractions, components, kept):
    import numpy as np
    inverse = {j: i for i, j in enumerate(kept)}
    selected = [i for i, r in enumerate(components) if r["candidate_index"] in inverse]
    owner = np.array([inverse[components[i]["candidate_index"]] for i in selected], dtype=int)
    low = np.array([components[i]["lower"] for i in selected])
    high = np.array([components[i]["upper"] for i in selected])
    D = fractions[:, selected] * A[:, owner]
    fixed = A.copy()
    for i, j in enumerate(owner):
        fixed[:, j] -= D[:, i]
    assert np.min(fixed) >= -1e-10
    fixed = np.maximum(fixed, 0)
    check = fixed.copy()
    for i, j in enumerate(owner):
        check[:, j] += D[:, i]
    assert np.max(abs(check - A)) <= 1e-10
    return fixed, D, owner, low, high


class Problem:
    def __init__(self, fixed, D, owner, low, high, b):
        import numpy as np
        from scipy import sparse as sp
        self.fixed, self.D, self.owner, self.low, self.high = fixed, D, owner, low, high
        self.b = b / np.sum(b)
        self.n, self.g, self.m = fixed.shape[1], D.shape[1], len(b)
        n, g, m = self.n, self.g, self.m
        M = sp.csr_matrix(np.column_stack([fixed, D]))
        link = sp.csr_matrix((np.ones(g), (np.arange(g), owner)), shape=(g, n))
        self.constraint = sp.vstack([
            sp.hstack([M, -sp.eye(m)]), sp.hstack([-M, -sp.eye(m)]),
            sp.hstack([-sp.diags(high) @ link, sp.eye(g), sp.csr_matrix((g, m))]),
            sp.hstack([sp.diags(low) @ link, -sp.eye(g), sp.csr_matrix((g, m))])], format="csr")
        self.rhs = np.r_[self.b, -self.b, np.zeros(2 * g)]
        self.objective = np.r_[np.zeros(n + g), np.ones(m)]
        lower_mass = fixed.sum(axis=0)
        for i, j in enumerate(owner):
            lower_mass[j] += low[i] * D[:, i].sum()
        assert np.all(lower_mass > 0)
        # Redundant finite bounds: zero prediction has objective1, hence any relevant optimum
        # has predicted L1 mass<=2. The factor4 adds slack; it is not an abundance prior.
        self.upper = np.r_[4 / lower_mass, high * (4 / lower_mass)[owner], np.full(m, 2.)]

    def solve(self, removed=()):
        import numpy as np
        from scipy.optimize import linprog
        upper = self.upper.copy()
        upper[list(removed)] = 0
        upper[self.n + np.flatnonzero(np.isin(self.owner, removed))] = 0
        started = time.monotonic()
        result = linprog(self.objective, A_ub=self.constraint, b_ub=self.rhs,
                         bounds=list(zip(np.zeros(len(upper)), upper)), method="highs",
                         options=dict(primal_feasibility_tolerance=1e-9, dual_feasibility_tolerance=1e-9,
                                      ipm_optimality_tolerance=1e-10, time_limit=60))
        assert result.success, result.message
        x = np.clip(result.x[:self.n], 0, upper[:self.n])
        w = np.clip(result.x[self.n:self.n+self.g], self.low*x[self.owner], self.high*x[self.owner])
        residual = self.fixed @ x + self.D @ w - self.b
        primal = np.r_[x, w, abs(residual)]
        dual = np.minimum(result.ineqlin.marginals, 0.)
        lb, ub = self.check_bounds(primal, dual, upper)
        assert ub - lb <= 1e-6, (lb, ub, result.fun)
        return dict(lower=lb, upper=ub, seconds=time.monotonic()-started,
                    objective=float(result.fun), proof="GLOBAL_LP_BOUNDS_NUMERICAL_GUARD_1e-8"), primal, dual

    def check_bounds(self, primal, dual, upper):
        import numpy as np
        assert np.isfinite(primal).all() and np.isfinite(dual).all()
        assert np.min(primal) >= 0 and np.max(primal-upper) <= 1e-8 and np.max(dual) <= 0
        assert np.max(self.constraint @ primal - self.rhs) <= 1e-8
        # Any nonpositive dual vector supplies a global lower bound for the bounded LP:
        # rhs@y + sum(upper * min(c - C.T@y,0)). No local-optimum inference is used.
        long_dual = dual.astype(np.longdouble)
        reduced = self.objective.astype(np.longdouble) - self.constraint.astype(np.longdouble).T @ long_dual
        raw_lb = float(self.rhs.astype(np.longdouble) @ long_dual
                       + np.sum(upper.astype(np.longdouble) * np.minimum(reduced, 0)))
        ub = float(np.sum(abs(self.fixed @ primal[:self.n] + self.D @ primal[self.n:self.n+self.g] - self.b))) + 1e-8
        lb = max(0., raw_lb - 1e-8)
        assert lb <= ub
        return lb, ub


def self_test():
    import numpy as np
    A = np.array([[1., 0.], [1., 1.]])
    fixed = np.array([[1., 0.], [0., 1.]])
    D = np.array([[0.], [1.]])
    problem = Problem(fixed, D, np.array([0]), np.array([.8]), np.array([1.2]), np.array([1., 1.2]))
    full, _, _ = problem.solve()
    false_deleted, _, _ = problem.solve([1])
    true_deleted, _, _ = problem.solve([0])
    assert full["upper"] < 1e-7 and false_deleted["upper"] < 1e-7
    assert true_deleted["lower"] > .45
    fixed_problem = Problem(A, np.zeros((2, 0)), np.array([], dtype=int), np.array([]), np.array([]), np.array([1., 1.2]))
    old, _, _ = fixed_problem.solve([1])
    assert abs(old["upper"] - 1e-8 - 1/11) < 1e-7
    return dict(status="PASS", exact_tiny_case=True, all_same_identity_components_removed=True,
                false_necessity_removed_by_allowed_true_variation=True, known_global_values_and_dual_bounds=True)


def counts(cases, eps=None, score=None, threshold=None):
    selected = []
    raw_tp = sum(sum(r["molecular_truth"] for r in c["records"]) for c in cases)
    raw_n = sum(len(c["records"]) for c in cases)
    truth = sum(c["truth_count"] for c in cases)
    reportable = sum(c["reportable_truth_count"] for c in cases)
    abstained = 0
    for case in cases:
        valid = eps is None or case["full"]["upper"] <= eps
        abstained += not valid
        for row in case["records"]:
            retain = valid
            if eps is not None:
                retain &= row["deleted"]["lower"] > eps
            if score:
                retain &= threshold is not None and row[score] >= threshold
            if retain:
                selected.append(row)
    tp = sum(r["molecular_truth"] for r in selected)
    fp = len(selected) - tp
    return dict(TP=tp, FP=fp, FN=truth-tp, filtered_TP=tp, filtered_FP=fp, filtered_FN=truth-tp,
        raw_solver_TP=raw_tp, raw_solver_FP=raw_n-raw_tp,
        raw_solver_FN=truth-raw_tp, filter_induced_true_loss=raw_tp-tp,
        filter_induced_true_loss_fraction=(raw_tp-tp)/raw_tp if raw_tp else None,
        true_positive_retention=tp/raw_tp if raw_tp else None,
        TP_retention=tp/raw_tp if raw_tp else None, all_truth_recall=tp/truth if truth else None,
        reportable_truth_recall=sum(r["reportable_truth"] for r in selected)/reportable if reportable else None,
        FDR=fp/len(selected) if selected else 0., distinct_identities=len({r["lipid_name"] for r in selected}),
        retained_count=len(selected), case_count=len(cases), abstained_cases=abstained,
        abstention_fraction=abstained/len(cases) if cases else None)


def calibrate(cases):
    # One scalar validity/deletion tolerance, chosen on development calibration contexts only.
    values = sorted({0., 1., *(c["full"]["upper"] for c in cases),
                     *(r["deleted"]["lower"] for c in cases for r in c["records"])})
    curve = [dict(epsilon=v, **counts(cases, eps=v)) for v in values]
    eligible = [r for r in curve if r["retained_count"] and r["FDR"] <= .01]
    chosen = max(eligible, key=lambda r: (r["TP"], -r["FP"], r["epsilon"])) if eligible else next(r for r in curve if r["epsilon"] == 1.)
    baseline = {}
    for score in ("rho_zero", "X_hat"):
        thresholds = sorted({r[score] for c in cases for r in c["records"]})
        options = [dict(threshold=t, **counts(cases, score=score, threshold=t)) for t in thresholds]
        allowed = [r for r in options if r["retained_count"] and r["FDR"] <= .01]
        baseline[score] = max(allowed, key=lambda r: (r["TP"], -r["FP"], r["threshold"])) if allowed else dict(threshold=None)
    return dict(epsilon=chosen["epsilon"], calibration=chosen, baseline=baseline,
                status="CALIBRATED" if eligible else "NO_NONEMPTY_CALIBRATION_SET", curve=curve)


def prepare(args):
    import numpy as np
    sys.path[:0] = [str(args.root/"analysis"), str(args.root/"src")]
    import run_v58_spectral_library_mismatch_fdr_recalibration as v
    v.dependencies()
    va = v.parse_args(["--output-dir", str(args.root/"results/v58_spectral_library_mismatch_fdr_recalibration_compact_recovery"),
                      "--v57-output", str(args.root/"results/v57_spectral_spatial_identity_confidence_benchmark"),
                      "--asset-root", str(args.asset_root), "--audit-design"])
    parent, _, _ = v.parent_provenance(va)
    context = v.load_context(va, parent)
    targets, _, _ = v.build_targets(context)
    design = read(va.output_dir/"design.json")
    audit = v.checked_audit(va, design)
    missing = args.root/"results/missing_library_challenge_k125"
    missing_design = read(missing/"execution_design.json")
    assert missing_design["fingerprint"] == "b51137f3c6f12cf9bdf0e3e9f1c00c1fff53f9e4d62dc1c54de0b04a7910d9ef"
    assert hashlib.sha256(json.dumps(missing_design["scientific"], sort_keys=True,
        separators=(",", ":"), allow_nan=False).encode()).hexdigest() == missing_design["fingerprint"]
    missing_seal = read(missing/"execution_freeze.json")
    assert missing_seal["design_sha256"] == sha(missing/"execution_design.json")
    assert missing_seal["validation_sha256"] == sha(missing/"validation.json")
    datasets = []
    for rep, role in ((1, "DEVELOPMENT_CAL"), (2, "DEVELOPMENT_EVAL")):
        mild_id = f"MILD__CAL_R{rep}_K125"
        mild, actual = v.mismatch_case(context, parent, targets, mild_id)
        actual = v.v57.canonical(actual)
        for k, expected in audit["datasets"][mild_id].items():
            if k in ("forward_mismatch_residual", "sentinel_gate_residual"):
                assert abs(actual[k]-expected) <= 1e-12
            else:
                assert actual[k] == expected, k
        clean = v.v57.construct_case(context, parent["scientific"], f"HOLD_R{rep}_K125")
        for kind in ("MILD", "close_neighbor", "relatively_isolated"):
            key = mild_id if kind == "MILD" else f"{kind}__HOLD_R{rep}_K125"
            loc = v.directory(va, mild_id) if kind == "MILD" else missing/"runs"/key
            dest = args.output/"inputs"/key
            dest.mkdir(parents=True)
            if kind == "MILD":
                report = v.verified_report(va, design, mild_id, actual)
                solver = read(loc/"solver_run.json")
                v.verify_solver_run(va, design, mild_id, v.binding(design, mild_id, actual))
                assert sha(loc/"learned_arrays.npz") == solver["arrays_sha256"]
                with np.load(loc/"learned_arrays.npz") as a:
                    normal, finite = v.solver_validity({**solver["training"], "X_hat":a["X_hat"], "B_hat":a["B_hat"]})
                assert normal and finite
                molecular_rows = v.normalize_records(v.v57.rows(loc/"molecular_false_negative_records.csv"))
                candidate_rows = v.normalize_records(v.v57.rows(loc/"candidate_false_negative_records.csv"))
                records = [r for r in molecular_rows if r["raw_solver_reported"]]
                kept = list(range(391)); case = mild
            else:
                report = read(loc/"result.json")
                assert report["status"] == "COMPLETE" and report["fingerprint"] == missing_design["fingerprint"]
                assert report["case_id"] == key
                for name, expected in report["artifact_hashes"].items():
                    assert sha(loc/name) == expected, (key, name)
                runtime = read(loc/"runtime_contract.json"); kept = runtime["reduced_to_original"]; case = clean
                assert runtime["case_id"] == key and runtime["fingerprint"] == missing_design["fingerprint"]
                entry = missing_design["scientific"]["cases"][key]
                arm = missing_design["scientific"]["arms"][kind]
                assert kept == arm["reduced_to_original"] and len(set(kept)) == 386
                assert runtime["B_sha256"] == entry["B_sha256"]
                assert runtime["A_solver_sha256"] == entry["A_solver_sha256"]
                assert runtime["B_sha256"] == v.array_sha(case["B_sim"])
                assert runtime["A_solver_sha256"] == v.array_sha(context["A_solver"][:,kept])
                solver = read(loc/"solver_run.json")
                assert solver["stop_reason"] in ("max_epochs", "converged")
                assert np.isfinite(solver["final_raw_losses"]).all() and np.isfinite(solver["final_physical_loss"]).all()
                with np.load(loc/"learned_arrays.npz") as a:
                    assert np.isfinite(a["X_hat"]).all() and np.isfinite(a["B_hat"]).all()
                    assert np.array_equal(a["reduced_to_original"], kept)
                records = read(loc/"molecular_units.json")
                candidate_rows = read(loc/"candidate_records.json")
            assert len({r["lipid_name"] for r in records}) == len(records)
            truth_names = {str(context["metadata"]["lipid_name"][i]) for i in case["active_indices"]}
            reportable_names = {str(context["metadata"]["lipid_name"][i]) for i in case["reportable_truth_indices"]}
            assert len(truth_names) == 125
            assert all(r["molecular_truth"] == (r["lipid_name"] in truth_names) for r in records)
            assert all(r["reportable_truth"] == (r["lipid_name"] in reportable_names) for r in records)
            expected_names = {r["lipid_name"] for r in candidate_rows if r["raw_solver_reported"]}
            assert {r["lipid_name"] for r in records} == expected_names
            assert all(r["raw_solver_reported"] == (r["X_hat"] > .001) for r in candidate_rows)
            copies = [p for p in loc.iterdir() if p.is_file() and p.suffix in (".json", ".csv")]
            for p in copies:
                shutil.copy2(p, dest/p.name)
            A = context["A_solver"][:,kept].astype(float)
            b = case["B_sim"][:,case["foreground_mask"]].mean(axis=1, dtype=np.float64)
            np.savez_compressed(dest/"scoring_inputs.npz", A=A, b=b, kept=kept)
            item = dict(dataset=key, kind=kind, role=role, truth_count=125, records=records,
                        reportable_truth_count=len(reportable_names),
                        names=[str(context["metadata"]["lipid_name"][i]) for i in kept],
                        source_dir=str(loc), input_dir=str(dest),
                        B_cube_sha256=v.array_sha(case["B_sim"]), A_sha256=v.array_sha(context["A_solver"][:,kept]),
                        original_artifacts={p.name:sha(p) for p in copies},
                        scoring_inputs_sha256=sha(dest/"scoring_inputs.npz"),
                        normal_completion=True, finite_arrays=True, runtime_binding_verified=True)
            write(dest/"pilot_input.json", item)
            datasets.append(item)
            print("INPUT_VERIFIED", key, len(records), flush=True)
    return datasets


def score_case(item, args, fractions, components):
    import numpy as np
    source = Path(item["input_dir"])
    assert sha(source/"scoring_inputs.npz") == item["scoring_inputs_sha256"]
    data = np.load(source/"scoring_inputs.npz")
    model_args = model(data["A"], fractions, components, data["kept"].tolist())
    problem = Problem(*model_args, data["b"])
    full, full_x, full_y = problem.solve()
    proofs = {"full_primal":full_x, "full_dual":full_y}
    if full["seconds"] > 30:
        raise RuntimeError("STOP_COST: full LP >30s")
    output = []
    for i, row in enumerate(sorted(item["records"], key=lambda r: item["names"].index(r["lipid_name"]))):
        removed = [j for j, n in enumerate(item["names"]) if n == row["lipid_name"]]
        if np.all(full_x[removed] == 0):
            deleted = dict(full, seconds=0., proof="FULL_ZERO_IDENTITY_FEASIBLE_WITNESS")
        else:
            deleted, px, dy = problem.solve(removed)
            proofs[f"primal_{i}"] = px; proofs[f"dual_{i}"] = dy
        assert deleted["upper"] >= full["lower"] - 1e-8
        output.append(dict(**row, deleted=deleted, removed=removed, proof_index=i))
        if i == 7 and np.median([r["deleted"]["seconds"] for r in output]) > 10:
            raise RuntimeError("STOP_COST: first8 median LP >10s")
        if (i+1) % 50 == 0:
            print(item["dataset"], i+1, "/", len(item["records"]), flush=True)
    dest = args.output/"scores"/item["dataset"]
    dest.mkdir(parents=True)
    np.savez_compressed(dest/"proofs.npz", **proofs)
    result = {k:v for k,v in item.items() if k != "records"}
    result.update(full=full, records=output, proofs_sha256=sha(dest/"proofs.npz"))
    write(dest/"scores.json", result)
    return result


def main(args):
    import numpy as np
    if args.action == "self-test":
        print(json.dumps(self_test())); return
    args.output.mkdir(parents=True, exist_ok=False)
    write(args.output/"self_test.json", self_test())
    uncertainty = read(args.uncertainty/"provenance.json")
    for name, expected in uncertainty["outputs"].items():
        assert sha(args.uncertainty/name) == expected
    shutil.copytree(args.uncertainty, args.output/"uncertainty")
    write(args.output/"protocol.json", dict(version="CE_UNCERTAINTY_IDENTITY_LP_V1", script_sha256=sha(Path(__file__)),
        uncertainty_manifest_sha256=sha(args.uncertainty/"provenance.json"), metric="minimum relative L1 mean-spectrum residual",
        CAL="MILD CAL_R1_K125 and both original missing-library HOLD_R1_K125 arms; developmental reuse",
        EVAL="same three contexts at R2; developmental reuse, not independent final HOLD",
        calibration="max CAL TP at observed FDP<=1%; ties fewer FP, then larger epsilon; empty ->epsilon1",
        decision="full upper<=epsilon and deleted lower>epsilon", continue_target=dict(FDR=.01, TP_retention=.4),
        strong_retention=.6, all_truth_denominators_preserved=True, no_GPU_training=True,
        bounds="finite redundant optimizer bounds, global LP weak duality and feasible witnesses, 1e-8 guards",
        formal_completion="independent CAL/HOLD FDR<=1% and TP retention>=40%;60% not required"))
    items = prepare(args)
    fractions = np.load(args.uncertainty/"component_fractions.npz")["fractions"]
    components = read(args.uncertainty/"components.json")
    calibrated = [score_case(i,args,fractions,components) for i in items if i["role"] == "DEVELOPMENT_CAL"]
    seal = calibrate(calibrated)
    write(args.output/"calibration_seal.json", seal)
    seal_hash = sha(args.output/"calibration_seal.json")
    print("CALIBRATION_SEALED", seal["epsilon"], seal["calibration"], flush=True)
    evaluated = [score_case(i,args,fractions,components) for i in items if i["role"] == "DEVELOPMENT_EVAL"]
    assert sha(args.output/"calibration_seal.json") == seal_hash
    summary = dict(status="COMPLETE_AWAITING_REVIEW", calibration_seal_sha256=seal_hash,
                   calibration=counts(calibrated,eps=seal["epsilon"]), evaluation=counts(evaluated,eps=seal["epsilon"]),
                   by_case={}, by_challenge={}, baselines={})
    for c in calibrated+evaluated:
        summary["by_case"][c["dataset"]] = counts([c],eps=seal["epsilon"])
        write(args.output/"scores"/c["dataset"]/"retained_records.json", [dict(r,
            retained=c["full"]["upper"]<=seal["epsilon"] and r["deleted"]["lower"]>seal["epsilon"])
            for r in c["records"]])
    for kind in ("MILD","close_neighbor","relatively_isolated"):
        summary["by_challenge"][kind] = counts([c for c in evaluated if c["kind"]==kind],eps=seal["epsilon"])
    for score, info in seal["baseline"].items():
        summary["baselines"][score] = counts(evaluated,score=score,threshold=info["threshold"])
    e = summary["evaluation"]
    summary["go"] = e["retained_count"]>0 and e["FDR"]<=.01 and e["TP_retention"]>=.4
    summary["strong_success"] = summary["go"] and e["TP_retention"]>=.6
    summary["inference"] = "Developmental proceed/stop decision; not formal FDR validation or impossibility proof"
    write(args.output/"summary.json",summary)
    files = {str(p.relative_to(args.output)):sha(p) for p in args.output.rglob("*") if p.is_file()}
    write(args.output/"output_manifest.json",files)
    print(json.dumps(summary),flush=True)


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=("self-test","run"))
    p.add_argument("--root",type=Path,default=Path("/root/autodl-tmp/lipiddeconv"))
    p.add_argument("--asset-root",type=Path,default=Path("/root/autodl-tmp/decon-lipid"))
    p.add_argument("--uncertainty",type=Path,default=Path("/root/v58_jobs/ce133_uncertainty_v1_ready"))
    p.add_argument("--output",type=Path,default=Path("/root/v58_jobs/ce_uncertainty_identity_pilot"))
    a=p.parse_args()
    try:
        main(a)
    except BaseException:
        if a.output.exists():
            write(a.output/"failure.json",dict(error=traceback.format_exc()))
        raise
