# Computational closure plan

Recorded 2026-09-10 following user authorization to proceed by priority.
Status: PREPARATION; no new GPU job launched or queued.

## Dependencies and bounded asset check

- V58 supervisor currently reports RUNNING, 29/60 completed, active MILD__HOLD_R5_K175; status timestamp 2026-09-10T12:12:52.526082+00:00.
- V59 supervisor currently reports RUNNING at sentinel, with selftest/prepare/audit/freeze/oracle completed. This stage marker is not proof of current GPU progress or sentinel PASS.
- Neither experiment must finish before planning the missing-library challenge or analyzing completed V57 outputs. New GPU work should use a released GPU after the existing job exits; no automatic launch has been installed.
- Final V58 and V59 comparisons each require their own complete result set and sealed CAL thresholds. Do not choose challenge cases or alter its design using their learned HOLD outcomes.
- Local V57 output directory contains candidate/molecular false-negative records, frozen thresholds, recall/retention curves, by-K/by-replicate tables and design assets. Header inspection confirms geometry, abundance, truth, reportable truth, raw reporting, solver FN and filter-loss fields. This is an availability/schema check, not a completed provenance or statistical audit.

## Priority 1: small missing-library challenge

Scientific question: can omitted true identities induce high-confidence false allocation to remaining candidates?

Reuse frozen V57 CLEAN observations and spatial/abundance construction. Do not rebuild or rescale B after removing solver columns. Preserve original truth and reportable-truth denominators, including identities absent from the reduced solver library. Retain a mapping from reduced columns to original candidate and molecular identifiers. A molecular omission must remove every candidate representation of that identity.

Prepare a bounded paired design with a complete-library reference and two equal-sized omission sets: truths with close surviving spectral neighbors and relatively isolated truths. Fix selection using source truth/library geometry only, with deterministic tie-breaking; do not use rho, solver recovery or new HOLD outcomes. Match or describe abundance and other selection imbalances. Exact case count, omission count, selection rule and asset hashes remain NOT_FROZEN pending the focused reuse audit; no new K/window/noise grid.

Reuse an existing reference result only if scientific inputs, production implementation/configuration and artifact provenance match. Reduced-library arms require actual solver execution, not copied reference estimates. Preserve production training, early stopping, reporting gate and individual rho definition. Evaluate frozen CLEAN thresholds as transfer tests; no omission-outcome recalibration in the primary analysis.

Report overall TP/FP/FN, all/reportable-truth recall, raw misses, filter-induced loss, retained false identities, and loss of still-covered truths. Report allocation to predesignated surviving spectral neighbors descriptively; do not claim that an individual false positive has a uniquely attributable omitted source. Separate structural unavoidable FN from additional solver misses. Missing-library residual inflation is a scientific endpoint: do not automatically use the complete-library residual gate to censor difficult omission cases. Audit normal completion and finite outputs separately, with the process contract fixed before execution.

## Priority 2: existing-output analysis

Start with completed V57 outputs after validating provenance and CAL seals. Reuse existing rho and X_hat curves; report FDR alongside recall, coverage and TP retention. Compare frozen CAL-selected rules on HOLD. Any HOLD threshold sweep is descriptive, never a new deployment threshold.

Analyze true identities retained versus filtered using cone isolation, fragment/full cosine and collective gain, accounting for abundance, K, class and repeated identity/mapping structure. Keep original solver misses separate. Do not treat all candidate rows or nested K observations as independent replicates. No group-rho or method retuning.

## Priority 3: completed V58/V59 handoff

After each completes, follow ACTIVE_EXPERIMENTS.md: account for every planned dataset, verify provenance, retrieve compact outputs, and review calibration transfer/recalibration plus false-negative costs. Show replicate/domain variation and limited replication; do not claim universal FDR control from pooled point estimates or candidate-level independence intervals.

Keep code, analysis definitions, compact reports and provenance in the Git handoff. Existing push approval remains unresolved; this plan does not claim any new remote Git update. Preserve all source results and interrupted-run evidence.
