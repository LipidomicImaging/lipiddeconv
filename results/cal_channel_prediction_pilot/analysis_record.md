# CAL held-out-channel prediction: stop standalone route

Completed 2026-09-11 on one CPU thread in 585 seconds. Fixed CAL_R1_K125 CLEAN/MILD pair; 456 reported identity contexts, 1,368 deletion fits plus six full fits. No GPU training, no HOLD outcome access, no frozen production changes. Review specification was pushed as 88df213 while computation was still running and before score outputs were inspected.

## Retrospective CAL screen

These are maximum all-truth recalls among tied-score thresholds with empirical FDR <=5% on the same CAL dataset. They are optimistic descriptive envelopes, not transferable thresholds or a guarantee of FDR control. Both domains contain125 true/reportable identities.

| Domain | Score | TP | FP | All-truth recall | Ranking AUROC |
|---|---|---:|---:|---:|---:|
| CLEAN | Original rho_zero |123|0|98.4%|1.000|
| CLEAN | X_hat |51|2|40.8%|0.874|
| CLEAN | Held-out predictive gain |103|0|82.4%|1.000|
| MILD | Original rho_zero |32|1|25.6%|0.734|
| MILD | X_hat |45|2|36.0%|0.883|
| MILD | Held-out predictive gain |0|0|0%|0.458|

For MILD predictive gain, no nonempty threshold-selected set meets5% empirical FDR. The empty set is represented by TP/FP=0 and recall=0; its FDR is undefined, not a successful zero-FDR result. At1%, maximum recall is22.4% for rho,10.4% for X_hat, and0% for prediction.

## Miss accounting and support exclusions

CLEAN original solver TP123/FP102/FN2. The prespecified fragment-support gate excludes20 reported truths and15 reported false identities, leaving103 TP/87 FP evaluable. Its all-truth recall ceiling is82.4% before any score threshold.

MILD original solver TP122/FP109/FN3. Support excludes20 reported truths and15 false identities, leaving102 TP/94 FP. Its ceiling is81.6%. At the5% screen, prediction retains no identities: three solver misses,20 support exclusions and102 further score-selection losses. These sources of FN remain separate.

The common-evaluable comparison also fails: MILD rho retains27 TP/1 FP (21.6% all-truth recall), abundance39 TP/2 FP (31.2%), prediction none. Thus the observed failure is not explained only by the support gate shrinking the available truth set.

## Decision and limits

STOP_STANDALONE_ROUTE_NO_5_PERCENT_RECALL_GAIN. Do not tune folds, support thresholds, score sign or numerical settings after this outcome. The separately proposed six-fit alternative-reference GPU pilot remains PAUSED; it was not executed.

This result rejects this particular foreground-mean, blocked-channel deletion prediction score as the next standalone confidence method. It does not prove all mismatch-robust annotation strategies impossible. A fixed-library training-channel fit is not guaranteed to predict held-out channels robustly under library error; that mechanism is a hypothesis, not established causality here. No experimental perturbed reference spectra were supplied to scoring. Whole-data candidate selection still precedes scoring, and channel grouping does not guarantee statistical independence. Previously analyzed CAL identities are not an independent validation cohort.

## Verification and retention

The executed runner reconstructed the exact frozen A/B/X_truth bindings; only two derived residual scalar comparisons allow absolute1e-12 for BLAS reduction differences. Remote CAL input file hashes were rechecked unchanged at export. Download archive SHA256: b6c80db77c855bd0a6cd139bd00459a315d8526715013037958a55bf360835e2. Local source/output/script hashes, identity/fold uniqueness, finite scores/losses, loss subtraction, and original TP/report membership pass. The test-observation isolation and buffered/disjoint-channel tests pass; ranking review passes perfect/reversed/tied-score checks. Syntax compilation and Git whitespace checks pass.

Raw identity scores, every fold loss/support record, channel folds, immutable executed script, log, contracts and provenance are retained here. No files or checkpoints were retired. Existing V58/V59 and the already frozen missing-library challenge retain priority; this pilot adds no GPU job.
