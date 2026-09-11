# Latest operational snapshot — 2026-09-11

V58 recovered from archive storage failure with original evidence preserved. Active supervisor: /root/v58_jobs/resume_v58_storage_handoff.py (PID60098); active status: /root/v58_jobs/storage_recovery_20260911/status.json. Code: analysis/resume_v58_storage_handoff.py. At01:49UTC37/60 complete, active MODERATE__CAL_R3_K125. Old /root/v58_jobs/compact_recovery_status.json remains FAILED and must not be mistaken for active state. Same output and scientific fingerprint. New results stay on data disk; no additional root-cache archiving. User reports20GB expansion; guest still reports50GiB total at snapshot. Space guards wait rather than delete. After60 results and aggregate, the supervisor validates the frozen missing-library fingerprint/code and GPU release before six omission fits.

V59: all18 oracles and all3 sentinels PASS; formal PID4983 is active, three D0 CAL reports complete. D0 HOLD_R1 training completed, CPU rho pending. GPU idle during this step is not release. Current snapshot: results/active_experiments/v59_formal_status_20260911.json.

NNLS: all6 complete and reviewed; results/nnls_solver_baseline_k125_cpu4/analysis_record.md. Original source status is preserved; local_review.json records review PASS. The dated snapshot text below remains historical.

# Active experiments: V58 and V59

Updated 2026-09-10. Status below is an observed snapshot, not a completion claim. No credentials are stored here. Read the named remote status files for later progress; do not restart training just to obtain a status update.

## Compact V58: spectral-library mismatch and FDR recalibration

Question: when the true spectral library differs from the production solver library, does rho_zero still rank identity reliability, does the original CLEAN cutoff transfer, and can mismatch-domain CAL recalibration restore useful HOLD FDR/recall?

- Production precursor range remains 748–798. Solver uses the frozen full391-candidate production library.
- Synthetic observations use a fixed perturbed full target library per severity. X_true, spatial construction and original global scaling remain inherited from V57; no post-mismatch rescaling.
- MILD: fragment log-sigma0.10, fragment dropout0.05, parent deviation0.10. MODERATE:0.20,0.10,0.20. No measurement noise or new m/z window.
- **60 datasets = 2 severities × K(50,125,175) × CAL/HOLD × R1–R5.** K levels are nested; these are correlated solver contexts, not60 independent biological samples.
- Primary comparison: V57 CLEAN_FIXED thresholds versus severity-local CAL recalibration, evaluated on identity-disjoint HOLD. Report FDR together with recall, retained identities and TP retention. No group-rho.
- Four old draft sentinels may supply process evidence only through the explicit equivalence/provenance adoption checks. Ordinary60-run results require compact-design training; no ordinary draft result inheritance.

Runtime:

- Host: `connect.westc.seetacloud.com`, SSH port55786.
- Job: `/root/v58_jobs/compact_recovery_job.py`; status: `/root/v58_jobs/compact_recovery_status.json`.
- Output: `/root/autodl-tmp/lipiddeconv/results/v58_spectral_library_mismatch_fdr_recalibration_compact_recovery/`.
- Design fingerprint: `74032334eb538114e76459a1bb82ef0187d38f48ab20e60124021cd638afd8f0`.
- Snapshot: RUNNING, **25/60 complete**, active `MILD__HOLD_R4_K125`; state updated2026-09-10T11:24:53Z.
- Preserve the old interrupted directory and adoption source artifacts. Large runtime artifacts partly reside in `/root/v58_compact_runtime_cache`; original result paths are links with a hash manifest. Do not delete either copy based on filename or apparent redundancy.

## V59: standardized cross-library recalibratability

Question: does the same algorithm and identity-confidence calibration framework remain useful when candidate-library composition changes across standardized precursor domains?

- **18 datasets = D0/D1/D2 × CAL/HOLD × R1–R3**, fixed K125, singleton-only active truth.
- D0=W0_STD748–798:896 candidates,758 singletons. D1=E1 800–850:1223 candidates,923 singletons. D2=E2 850–900:1133 candidates,761 singletons.
- Scope700–900 was fixed before V59 learned/rho/FDR outcomes.700–750 and750–800 overlap W0;900–950 is out of scope. E1/E2 are all eligible external windows, not LOW/HIGH quantile selections. No cone/outcome-based selection.
- CLEAN exact library: A_synthesis=A_solver=A_std, with the complete candidate library. No spectral mismatch, measurement noise or new tissue. Reuse V57 frozen spatial maps and abundance assignment without additional normalization.
- Domain-local CAL→sealed threshold→HOLD is primary; V57 CLEAN_FIXED cutoff transfer is secondary. No claim of acquisition validation, trained-checkpoint transfer, universal rho threshold or all-mass-range generalization.
- Three CAL_R1 sentinels are members of the18 datasets. PASS requires normal finite completion and final reconstruction residual<=0.10; early stopping allowed,3000 is a hard cap only. Identity metrics do not gate process validity.

Runtime:

- Host: `connect.cqa1.seetacloud.com`, SSH port48975; independent GPU from V58.
- Active job/status: `/root/autodl-tmp/v59_jobs/oracle_float64_recovery/run_job.py` and `status.json`.
- Active output: `/root/autodl-tmp/lipiddeconv/results/v59_oracle_float64_recovery/v59_standardized_cross_library_formal/`.
- Active fingerprint: `a8331e8ecf4cd51ed3bfb5a1e1310de46297f3470012f89ab1b9e0e1c4382fa3`.
- Snapshot: **all18 oracles PASS**, GPU sentinel stage running. D0 CAL_R1 reached epoch850, GPU95%, memory20406MiB. Stage began2026-09-10T11:26:56Z; sentinel PASS not yet established at this snapshot.
- Supervisor runs selftest→prepare→equivalence→audit→freeze→oracle→sentinel→formal→aggregate, stops on any failure, and survives SSH disconnection.

Precision recovery provenance:

- Original remote fingerprint `f0767f657af3aa80d5493edef5481de3934f606c61c65f65b6ae723bbf49127d` stopped at D1 CAL_R3:TP125/FP0/FN0, residual1.1030472619523216e-6 exceeded the pre-frozen1e-6 gate.
- Root cause: inherited oracle averaged float32 B in float32 before casting. On identical B, float64 accumulation gave residual7.827999147121255e-10, stillTP125/FP0/FN0.
- Corrected only the V59 oracle adapter, not V54/V57/V58 or production code. Tolerance remained1e-6. Original failed directory, logs, oracle records and source snapshot remain preserved.
- Recovery equivalence confirms all18 X_true/B hashes, libraries, identities, splits, spatial/abundance assignments and production training/rho are unchanged. Only oracle accumulation declaration and V59 implementation binding change. The new18 oracles were recomputed, never relabelled PASS.
- Local initial prepared fingerprint `7197add774baf570869275f38d70248cd0b5f6047bf481fb3ab3803cd8a65367` is historical. Windows/Linux forward products have distinct B byte hashes; source libraries and all18 X_true match. Do not replace a runtime fingerprint with the local one.

## Git and final results handoff

The user explicitly requires code, experimental results and analysis records to be retained in Git. A progress commit is not the final experiment deliverable.

After each experiment completes:

1. Read its supervisor state and all planned completion records; account for exactly60 V58 or18 V59 datasets. A failure remains a failure; do not silently omit it or tune the design.
2. Retrieve compact reports, CAL threshold files/seals, HOLD metrics, recall/FDR curves, candidate/molecular false-negative records, oracle/sentinel validity, design and implementation provenance, recovery/adoption records and runtime/storage manifests. Verify downloaded hashes.
3. Review the scientific results, including raw solver FN versus filter-induced loss and low/zero retention. Keep scientific failure separate from process failure. Document limitations and unresolved checks.
4. Commit and push the completed reports and analysis records to `LipidomicImaging/lipiddeconv`; update this register and CURRENT_STATE/NEXT_TASK with the final commit and artifact locations.
5. Preserve large checkpoints/arrays in their explicit remote/storage locations with checksums and a documented backup. Do not add them blindly to lightweight Git, and never delete them before dependency/backup verification.

Do not infer a live monitoring automation from this checklist. Neither experiment's final results have been committed by this progress update. Do not stage unrelated V52 edits or historical V48 output directories with these experiments.
