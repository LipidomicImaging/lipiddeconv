# Missing-library final review — 2026-09-11

Six reduced-library fits completed and passed process, provenance and independent accounting review. Each removes five true singleton identities from the solver library, retains the original 125-truth denominator and uses the original V57 CAL thresholds without recalibration. The three existing full-library controls and their checkpoint evidence remain intact.

All values below are pooled over three mapping replicates per arm (375 truth contexts). FDR is the observed false-discovery proportion here; mapping replicates and repeated identities are not independent biological samples. Zero observed FP does not establish zero population risk.

| Library arm | Frozen rule | TP | FP | FN | Observed FDR | TP retention | All-truth recall |
|---|---|---:|---:|---:|---:|---:|---:|
| full_library | raw | 361 | 354 | 14 | 49.51% | 100.00% | 96.27% |
| full_library | rho_zero_FDR5 | 361 | 8 | 14 | 2.17% | 100.00% | 96.27% |
| full_library | rho_zero_FDR1 | 361 | 0 | 14 | 0.00% | 100.00% | 96.27% |
| full_library | X_hat_FDR5 | 119 | 3 | 256 | 2.46% | 32.96% | 31.73% |
| full_library | X_hat_FDR1 | 3 | 0 | 372 | 0.00% | 0.83% | 0.80% |
| close_neighbor | raw | 350 | 361 | 25 | 50.77% | 100.00% | 93.33% |
| close_neighbor | rho_zero_FDR5 | 305 | 214 | 70 | 41.23% | 87.14% | 81.33% |
| close_neighbor | rho_zero_FDR1 | 302 | 187 | 73 | 38.24% | 86.29% | 80.53% |
| close_neighbor | X_hat_FDR5 | 111 | 13 | 264 | 10.48% | 31.71% | 29.60% |
| close_neighbor | X_hat_FDR1 | 4 | 1 | 371 | 20.00% | 1.14% | 1.07% |
| relatively_isolated | raw | 347 | 363 | 28 | 51.13% | 100.00% | 92.53% |
| relatively_isolated | rho_zero_FDR5 | 281 | 193 | 94 | 40.72% | 80.98% | 74.93% |
| relatively_isolated | rho_zero_FDR1 | 279 | 184 | 96 | 39.74% | 80.40% | 74.40% |
| relatively_isolated | X_hat_FDR5 | 105 | 10 | 270 | 8.70% | 30.26% | 28.00% |
| relatively_isolated | X_hat_FDR1 | 6 | 0 | 369 | 0.00% | 1.73% | 1.60% |

The CLEAN-calibrated rho FDR1 cutoff retains 361 true and zero false contexts with the full library, but 302/187 TP/FP after close-neighbor omission and 279/184 after relatively-isolated omission. Observed FDR rises to 38.24% and 39.74%. This transfer fails the original nominal 1% target. These are missing-library CLEAN observations, not a combined spectral-mismatch-plus-omission experiment.

Raw FN accounting: close-neighbor arm has 25 FN = 15 obligatory omitted truths + 10 additional solver misses; the relatively-isolated arm has 28 = 15 + 13. The rho FDR1 filter then loses another 48 and 68 raw true positives, respectively. Omitted truths were never removed from recall denominators.

The six stored foreground reconstruction residuals range from 0.037371 to 0.047163. They are scientific outcomes, not exclusion gates. No fit, synthetic reconstruction or rho was rerun. The original residual values were retained rather than independently recomputed from B.

Candidate and molecular records, exact frozen thresholds, CAL source records, all six runtime/result/history files, checkpoint diagnostics and paired neighbor-allocation records are in this snapshot. The relatively-isolated arm is only relatively less similar; one selected neighbor cosine is 0.966. Neighbor allocation differences are descriptive and do not identify a unique source of signal.

Storage decision: retire only epochs 1000, 1500, 2000 and 2500 for these six completed reduced-library fits: 24 files, 949983592 bytes (0.885 GiB). All 178 protected source hashes must still match immediately before and after deletion. The result-bound latest_model.pth, final epoch-3000 model, learned arrays and all parent-control checkpoints remain. Intermediate weights will no longer be reloadable; their scalar diagnostics remain. This plan requires a successful Git snapshot push before execution; consult deletion_receipt.json for whether execution actually completed.

Review-tool preflight notes: an initial export refused to duplicate the large frozen parent design; a second attempt encountered the nested V57 molecular-level report schema. The tool was corrected, partial exports retained separately on the remote host, and all six cases subsequently passed. Neither issue changed any experiment or source artifact.
