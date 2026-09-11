# Paired MILD NNLS CAL screen

Completed 2026-09-11. One new pixelwise full391-column NNLS run on frozen MILD__CAL_R1_K125,15,837 foreground pixels, four CPU workers,1082 seconds. CLEAN NNLS and both ISTA comparisons are cached. No GPU training or HOLD outcome analysis. Protocol/runner committed as8f66b31 before completion. This is a descriptive CAL screen, not independent validation.

## Raw molecular results

| Domain | Solver | TP | FP | FN | Recall | FDR |
|---|---|---:|---:|---:|---:|---:|
| CLEAN | ISTA |123|102|2|98.4%|45.33%|
| CLEAN | NNLS |125|0|0|100%|0%|
| MILD | ISTA |122|109|3|97.6%|47.19%|
| MILD | NNLS |125|86|0|100%|40.76%|

NNLS recovers the three truths missed by ISTA, and reports23 fewer false identities in total. This is not simply a subset filter:65 false identities are shared,44 are ISTA-only and21 NNLS-only. NNLS therefore improves raw recovery in this case but does not remove the substantial false-allocation problem under mismatch. Its reconstruction relative residual is0.0484761; a small reconstruction error is not evidence that every reported identity is correct.

## Original scores after NNLS

Retrospective maximum all-truth recall over tied thresholds with empirical FDR<=5% on this same CAL case:

| Solver | Score | TP | FP | Recall | Original solver FN | Additional true loss |
|---|---|---:|---:|---:|---:|---:|
| ISTA | rho_zero |32|1|25.6%|3|90|
| NNLS | rho_zero |28|1|22.4%|0|97|
| ISTA | X_hat |45|2|36.0%|3|77|
| NNLS | X_hat |47|1|37.6%|0|78|

At1% empirical FDR, NNLS rho retains20 truths/0 false(16% recall), abundance45/0(36%). These CAL-label-selected envelopes are not deployed cutoffs or claimed FDR guarantees; the apparent1% gain needs independent evaluation. All125 truths are reportable, so all-truth and reportable-truth denominators coincide.

The rho score is exactly equal for every shared reported molecular identity between NNLS and ISTA, in both CLEAN and MILD. Its reported-universe AUROC differs because candidate selection differs (MILD NNLS0.6446, ISTA0.7340), not because NNLS changes the score for shared identities. Original rho depends on A, observed B and candidate deletion; learned abundance primarily determines which candidates are reported.

## Conclusion and limits

Changing to this unweighted NNLS objective alone is insufficient for low-FDR/high-recall annotation in the tested MILD case. Do not retune reporting gates or launch a larger sweep from this result. The remaining question is how to account for library error itself while preserving identity discrimination; this experiment does not establish any particular robust method or prove all strategies impossible. Different NNLS/production objectives and a single already examined CAL mapping preclude architecture-specific causal attribution or broad generalization.

## Audit and storage

Exact frozen A/B/X_truth audit fields match. Two derived residual scalars use only the already documented absolute1e-12 allowance for reduction roundoff. Existing CLEAN arrays and records were hashed before reuse; all input source hashes were unchanged at completion. All pixel solutions and stored predictions are finite/nonnegative where required. Sampled float64 KKT maximum dual violation3.43e-16, complementarity5.05e-17; these are diagnostics, not exhaustive numerical error bounds.

Local review verifies source/output/script hashes,391 unique candidate records, the125-truth universe, reporting gate membership, molecular truth labels, finite scores and separate solver/filter FN accounting. Download archive SHA256:6bed16fc0301ef20bddbed7d0dcec48b8aa94a8087ff5941274397ac12aebeea. Full learned_arrays.npz remains at /root/v58_jobs/mild_nnls_cal_pilot/learned_arrays.npz and was rehashed at export; its exact hash/size are in remote_array_recheck.json. This remote hash verification is not a local array backup. All lightweight outputs, immutable executed scripts, logs, and per-candidate abundances are retained in this directory. No data/checkpoints deleted.

Syntax compilation and Git diff whitespace checks pass. The initial launch guard found the baseline helper absent from the analysis directory before any computation; the existing helper was uploaded alongside the new runner, then the single computation started normally. No failed scientific run was overwritten.
