# CPU NNLS solver baseline

User authorized implementation and CPU execution on 2026-09-10. Six cases: frozen V57 K125, CAL/HOLD R1-R3. Reuse complete 391-column A_solver and exact B/X_true construction. Old V57 oracle is already full-library NNLS on the foreground mean; this baseline instead solves every foreground pixel. Truth never restricts the fit or determines stopping.

Objective: unweighted nonnegative least squares with scipy.optimize.nnls, float64, maxiter3910. No learned spectral calibration or spatial regularizer; objectives differ from production ISTA. Reporting remains foreground mean X_hat>0.001, with float32 stored maps and float64 mean. Background is returned zero only after verifying observed background is exactly zero. Production rho and individual molecular aggregation are reused. No GPU imports are used to allocate a device; CUDA_VISIBLE_DEVICES is empty.

Per-solver thresholds use the same three CAL cases, maximum retained identities at empirical FDR5/FDR1, with a hash seal written before HOLD execution/reading. ISTA comparison reuses frozen-parent aggregate records; no ISTA retraining. HOLD curves are descriptive, not threshold selection. Report raw and filtered FDR, recall, retention, FN and filter losses. Exact CLEAN NNLS may saturate; that is a valid outcome, not justification to redesign conditions. No universal solver-independence or real-data FDR guarantee follows from two solver families.

Remote runtime: connect.westc.seetacloud.com:55786, /root/nnls_solver_baseline_cpu4.py. Output /root/autodl-tmp/lipiddeconv/results/nnls_solver_baseline_k125_cpu4. Four CPU workers, one BLAS thread each, nice10; host container CPU quota16. No V58/V59 source edits. Progress is in status.json; failure.json records exceptions. Completed cases have hashed outputs; incomplete cases restart rather than pretending partial fits are complete.

Timing on32 evenly spaced foreground pixels (15837 total): single worker14.12s, four workers3.885s including pool startup. Four-worker extrapolation~32min per dataset,~3.2h for six pixel solves, excluding rho, I/O and contention. This is an estimate, not a completion deadline. Timing sampled no identity outcome. Earlier single-core timing/design preserved in the separate nnls_solver_baseline_k125 directory.

Validation before launch: py_compile and --help; independent small exact/zero NNLS systems; real32-pixel serial and parallel timing; parent source implementation and asset hashes; exact runtime B hash. Reduced-library challenge is separate and remains pending its adapter.
