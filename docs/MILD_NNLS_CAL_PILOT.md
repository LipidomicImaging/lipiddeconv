# MILD NNLS paired CAL pilot

Authorized 2026-09-11. One new full391-column pixelwise NNLS solve: frozen V58 MILD__CAL_R1_K125. Reuse previously audited CLEAN NNLS CAL_R1_K125 and the matching CLEAN/MILD production ISTA records. No HOLD outcomes, no new GPU fit, no target spectra in the solver. Reconstruct exact frozen observations using the existing builder and verify hashes; do not modify any V57/V58 scientific inputs.

Reuse run_nnls_solver_baseline.solve_pixel, float64 unweighted NNLS maxiter3910, four workers with one BLAS thread each. Store float32 maps and report foreground float64 mean>0.001. Reuse production rho and molecular aggregation unchanged. Different objectives from production ISTA preclude attribution to one architectural feature.

Primary question: does the same observed MILD case still produce substantial false molecular reports under NNLS? Report raw TP/FP/FN, recall, FDR, then per-score tied-threshold descriptive CAL envelopes at1% and5% empirical FDR. Compare rho and X_hat, with solver misses and filter-induced losses separated. Preserve every candidate abundance, reported candidate rho, molecular records and full arrays. No operational threshold is calibrated or applied to HOLD in this screen. A one-case comparison cannot establish independent generalization or FDR control.

Do not tune solver settings, reporting gates, cases or perturbations after observing outcomes. Do not expand automatically. If NNLS also reports many false identities, record that changing solver alone is insufficient in this case. If it improves, report the gain with recall and scope limitations before proposing further validation. Preserve all failed attempts; no checkpoint/data cleanup belongs to this pilot.
