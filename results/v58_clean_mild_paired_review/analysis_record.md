Paired CLEAN -> MILD review, 2026-09-11.

The original explanation must be narrowed: loss is not explained by a general decrease in true-identity rho. In1650 truth contexts reported by both solvers,921 scores increase and729 decrease. In1283 false-identity contexts reported by both,721 increase,238 decrease and324 tie. A plausible mechanism is that fixed-library competitors gain apparent necessity while absorbing mismatch, but these paired tables do not prove that mechanism or distinguish it from numerical effects in all near-zero scores.

We match by base dataset and molecular identity across the15 common HOLD cases (K50/125/175, R1-R5), verifying truth, reportable truth, candidate membership, K and replicate. The source V58 snapshot is complete for MILD. V57 heldout_records.csv matches its V58-bound parent hash byte-for-byte; the V57 threshold JSON matches only after LF normalization, explicitly recorded in summary.json. All412 V58 source hashes pass. No recalibration, solver rerun or rho recomputation occurred.

| Observations / cutoff | TP | FP | FN | Actual FDR | All-truth recall |
|---|---:|---:|---:|---:|---:|
| CLEAN raw | 1676 | 1528 | 74 | 47.69% | 95.77% |
| CLEAN / original CLEAN FDR5 | 1676 | 60 | 74 | 3.46% | 95.77% |
| MILD raw | 1680 | 1514 | 70 | 47.40% | 96.00% |
| MILD / original CLEAN FDR5 | 1412 | 835 | 338 | 37.16% | 80.69% |
| MILD / MILD CAL FDR5 | 370 | 15 | 1380 | 3.90% | 21.14% |

The original threshold is1.272651000158792e-20; the MILD CAL threshold is1.7543740552697457e-5. The original threshold is effectively close to the zero-score boundary, so enormous score ratios against tiny CLEAN scores must not be interpreted as corresponding magnitudes of biological evidence. Applying the MILD threshold to CLEAN would retain350 TP and0 FP (20% recall); this crossed cutoff is diagnostic only and is not a calibrated CLEAN operating point.

Raw solver recovery changes little in total:1650 true contexts remain reported,26 disappear,30 newly appear and44 remain missed. MILD's net raw TP gain of4 does not imply universal improvement because the identities of misses change. This matched subset differs from the full V57 six-K benchmark; do not compare its1750 truth denominator with V57's3375 aggregate as if the cohorts were identical.

Among the1676 truth contexts retained by the original CLEAN FDR5 rule:

- 26 become raw solver misses in MILD.
- 260 are still reported in MILD but fail even the original CLEAN cutoff; all260 move from positive CLEAN rho to exactly zero MILD rho.
- 1020 still pass the original cutoff on MILD but fail the higher MILD CAL cutoff.
- 370 remain retained under MILD recalibration.

This is an exact sequential accounting partition, not a causal mediation analysis. It depends on holding the original cutoff fixed first. Across all MILD truth (including newly raw-recovered identities), changing only its cutoff reduces retained TP1412 ->370, losing1042 true contexts and removing820 FP. At the original cutoff, FP rise60 ->835 across CLEAN/MILD even though raw FP totals are similar. Thus the key difficulty is the changed false-score support and true/false overlap, not merely loss of raw solver recall or uniformly declining true scores.

The prior identity-level association between ambiguity and filtering remains descriptive. CLEAN high recall is compatible with that association; it does not establish a causal ambiguity-by-mismatch interaction. Nested K, abundance changes, repeated spatial assignments, joint-reporting selection and fixed perturbation directions limit interpretation. Further mechanism tests should inspect residual differences and numerical optimality, and ask whether false competitors are explaining mismatch-specific signal. The existing pilot already requires such CPU diagnostics before new learned runs.

Outputs retain all matched molecular rows, raw-score changes including unavailable scores, crossed-threshold counts and per-K loss partition. Original V57/V58 results, thresholds and the frozen pilot protocol remain unchanged. No new GPU work or deletion was performed.
