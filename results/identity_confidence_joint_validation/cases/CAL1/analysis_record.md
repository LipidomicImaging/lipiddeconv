# CAL1: NO-GO under the frozen 1% / 80% contract

CAL1 finished at 2026-09-12 10:18:09 UTC (Beijing18:18:09), about22m29s after launch. The remote completion hook reviewed cached results successfully. The current user-requested check recovered316 files with matching hashes; local independent array, score, membership and accounting review also passed. No optimization was rerun. The remaining CAL2/HOLD1/HOLD2 solves were not started; there is no HOLD threshold seal or final validation claim.

| Operating point | True retained | False retained | Observed FDP | All-truth recall / TP retention | Filter-induced true losses |
|---|---:|---:|---:|---:|---:|
| Raw NNLS |125|51|28.98%|100%|0|
| Maximum recall with FDP<=1% |64|0|0%|51.2%|61|
| First threshold reaching80% recall |100|3|2.91%|80%|25|
| Lowest FDP anywhere with recall>=80% |117|3|2.50%|93.6%|8|

All125 truths are reportable and actually perturbed. Solver misses are0 at every point; all filtered false negatives are filter-induced. The listed curve points are permitted CAL diagnostics, not adopted deployment thresholds. Empty selections do not count as success, tied scores are not selectively split, and the denominator remains125.

The high-score false identity PG17:0_20:3 is ranked below only64 true identities. Two other false identities enter after85 and86 truths. Thus any cutoff that reaches80% recall already retains at least3 false identities; changing a single cutoff cannot meet1%/80%. The best lower-error point removes61 true identities. These names are accounting records only; no blacklist or post-hoc veto is used.

A bounded cached comparison with the old development case shows that this same shared model achieved at most99TP/1FP (79.2%) even in-sample. The earlier107TP/1FP oracle diagnostic used different fold-trained rankings and label-aware fold cutoffs; it was never evidence that this shared classifier would generalize. In CAL1, some false allocations acquire larger abundance and rho under a fresh5% perturbation; the shared model assigns them higher scores. The tested shared-model/calibration-only hypothesis is rejected. This does not establish that every identity-confidence strategy is impossible.

Preserved: all391 candidate records, all molecular records, score/rho details,125 truth labels, all threshold curve points, full NNLS outputs, all checkpoint blocks, source/model/input/result bindings and independent reviews. Large arrays remain both locally under this case directory and remotely at `/root/identity_confidence_joint_validation/results/cases/CAL1`; they are excluded from lightweight Git. No files were deleted.

This version is closed as NO_GO_CAL1. The broader target remains unresolved. The next scientific decision must follow this failure mechanism, preserve this negative result and distinguish development from a new prospective evaluation. Do not run the remaining three cases to try to rescue this version, tune its model/cutoff, or call these observed CAL rates a population FDR guarantee.
