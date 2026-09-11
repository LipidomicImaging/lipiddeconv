NNLS six-case review completed on 2026-09-11.

On this CLEAN exact-library K125 benchmark, pixelwise NNLS recovers every reported truth without false positives. Applying its CAL-frozen rho cutoff does not improve correctness and removes three true records. This is a boundary of the proposed filter's benefit, not grounds to change the threshold or construct a harder replacement test.

The experiment uses the full391-candidate library and the same six V57 observations (CAL/HOLD R1–R3, K125). The reporting gate remains foreground-mean X_hat > 0.001. NNLS solves unweighted nonnegative least squares per foreground pixel; production ISTA includes different training/spatial objectives and learned spectral calibration. This is a matched-input solver baseline, not an isolated architecture ablation or evidence of performance on experimental MSI.

HOLD accounting (375 truth contexts, including repeated identities across three mappings):

| Solver / retained set | TP | FP | Total FN | Solver FN | Filter-induced loss | FDR | Truth recall |
|---|---:|---:|---:|---:|---:|---:|---:|
| NNLS raw | 375 | 0 | 0 | 0 | 0 | 0% | 100% |
| NNLS rho FDR5 | 372 | 0 | 3 | 0 | 3 | 0% | 99.20% |
| NNLS rho FDR1 | 372 | 0 | 3 | 0 | 3 | 0% | 99.20% |
| NNLS X_hat FDR5 | 375 | 0 | 0 | 0 | 0 | 0% | 100% |
| NNLS X_hat FDR1 | 375 | 0 | 0 | 0 | 0 | 0% | 100% |
| ISTA raw | 361 | 354 | 14 | 14 | 0 | 49.51% | 96.27% |
| ISTA rho FDR5 | 361 | 3 | 14 | 14 | 0 | 0.824% | 96.27% |
| ISTA rho FDR1 | 361 | 0 | 14 | 14 | 0 | 0% | 96.27% |
| ISTA X_hat FDR5 | 151 | 12 | 224 | 14 | 210 | 7.362% | 40.27% |
| ISTA X_hat FDR1 | 105 | 2 | 270 | 14 | 256 | 1.869% | 28.00% |

All truth is reportable in these six cases, so all-truth and reportable-truth recall coincide. NNLS rho TP retention is99.2%; ISTA rho retains100% of raw TP. Both solvers are calibrated separately on the same three CAL cases. In particular, the ISTA thresholds here are subset-calibrated comparators, not the original full V57 global thresholds. FDR5/FDR1 name the CAL targets, not a guarantee for HOLD: the ISTA abundance baseline exceeds both targets on HOLD.

The three removed NNLS records are **one identity**, PE O-18:1_20:3, repeated in HOLD R1/R2/R3. Their rho values are approximately5.983e-10, below the frozen3.0491092908005636e-9 cutoff; abundance is approximately0.00766–0.00792, above the reporting gate. CAL NNLS has no FP, and the frozen maximum-retention rule selects its lowest CAL score. A lower HOLD true score can therefore be rejected despite perfect raw recovery. Do not interpret three repeated records as three independent ambiguity events, infer enrichment from this single identity, or lower the cutoff after viewing HOLD.

This establishes that the unchanged rho calculation can be applied to NNLS outputs. It does **not** establish solver-independent improvement or a universal numeric threshold. For production ISTA, it removes many false allocations while preserving the raw TP in this subset; for CLEAN NNLS, the additional filter has a small recall cost and no FP benefit. Zero observed FP is not a population-level guarantee. Stored candidate-level binomial intervals do not account for repeated identities or shared synthetic inputs and must not be presented as biological confidence intervals.

Validation and provenance:

- All30 downloaded source files match the remote SHA256 manifest. Original reports/status remain byte-preserved as COMPLETE_AWAITING_REVIEW; `local_review.json` records the completed review separately.
- Six-case membership, design fingerprint, CAL report hashes, threshold seal, maximum-retention CAL choices, and all NNLS/ISTA HOLD count/recall/filter-loss metrics independently pass `analysis/audit_nnls_completed.py` without importing solvers or recomputing rho.
- Remote array hashes were checked against six result reports before download; arrays stay at the named remote output directory. Remote `independent_audit.json` records numerical checks and confirms HOLD result timestamps follow the threshold seal. Timestamps are corroboration, not an immutable external chronology seal.
- Sampled KKT dual violation is at most7.524e-16 and complementarity at most2.672e-16. Sampling is every500 foreground pixels, not exhaustive. Six pixelwise solves took approximately29.5–32.2minutes each, excluding subsequent rho work; no GPU was used.
- Four HOLD curves are preserved for descriptive FDR/recall/TP-retention analysis; they must not be used to choose a new operational cutoff. Molecular and candidate records, frozen thresholds, and per-case reports remain available for later analysis.

No scientific parameter was changed, no training or rho was rerun for this review, and no biological or wet-lab validation is claimed.
