# V59 oracle precision recovery

The first remote job stopped before GPU training at D1__CAL_R3_K125. It recovered TP125 / FP0 / FN0 but reported residual 1.1030472619523216e-6 against the pre-frozen 1e-6 tolerance.

The inherited V54 helper reduced float32 observations in float32 and only then cast the mean to float64. A read-only diagnostic on the identical frozen B showed that float64 mean accumulation gives NNLS residual 7.827999147121255e-10 with TP125 / FP0 / FN0. The mean-vector rounding difference was 2.0503746044039997e-6 relative.

Only the V59 oracle adapter was corrected: exactly cast the original float32 B values to float64 BEFORE foreground averaging. No B value, identity, K, assignment, amplitude, threshold, production model, training or rho implementation changed. The old failed directory, diagnostics, logs, design and runner snapshot are preserved. No gate file was changed to PASS.

Recovery output:
`/root/autodl-tmp/lipiddeconv/results/v59_oracle_float64_recovery/v59_standardized_cross_library_formal/`

Recovery job:
`/root/autodl-tmp/v59_jobs/oracle_float64_recovery/run_job.py`

Before freezing the new implementation binding, the supervisor requires all18 X_true/B hashes and all candidate/library/split/spatial/abundance definitions to equal the initial remote design exactly. Only the oracle accumulation declaration and V59 implementation hash may differ. It writes precision_recovery_equivalence.json and stops if any comparison fails. All18 oracles must be recomputed with the corrected accumulator before any sentinel starts.

The original local prepared design and its validation report refer to the pre-fix implementation and remain historical evidence. They must not be used to bypass the corrected remote recovery lifecycle. No commit/push was performed for this fix.
