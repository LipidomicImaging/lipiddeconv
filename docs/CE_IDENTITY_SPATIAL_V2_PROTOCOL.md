# CE identity spatial V2 — frozen development pilot

2026-09-12. User-authorized implementation. V1 run04 is closed unchanged. Primary scientific contrast: global foreground mean versus molecular-identity-conditioned spatial spectrum. This is a six-fit development pilot, not a new full-factorial benchmark or formal population-FDR claim.

## Sources, new cases and unchanged factors

Reuse the exact CE v1 outputs and LP implementation (SHA25663768f2d37a7cb90da4cb2dcdd0dbd51f5cc71f25d12f3ede5f0b5f3f67a35d9). Keep107/391 covered full-library candidates,17 strata,480 components, shared-fragment intensity bounds, molecular deletion and all-candidate competition. Reduced libraries inherit the applicable subset of the same component mapping. No new perturbation/noise distribution or original rho calculation.

Use the existing V57 spatial assignment and construction functions, original healthy real maps, spectral topology,175 matched identity pairs, block design, abundance multipliers and one global signal scalar. New realization namespace: R61 seed6201 for development CAL; R62 seed6202 for development EVAL. These do not modify R1–R5 or any original source/result. Generate each seed once; a failed construction or novelty gate stops, never reseeds.

Exactly six new production fits: MILD__CAL_R61_K125, close_neighbor__HOLD_R61_K125, relatively_isolated__HOLD_R61_K125, and the same three at R62. Original CAL/HOLD in these names denotes the inherited identity pool; V2 developmental CAL/EVAL denotes R61/R62. These are new spatial realizations of reused identities and the same fixed MILD target library, not independent chemical-error realizations. Missing-library observations remain CLEAN with the same five omitted truths in each arm. No simultaneous mismatch-plus-omission claim.

Production training/losses/channel weights/seed/scheduler/early-stop remain unchanged.3000 is a hard cap, not a required epoch count. No oracle or extra sentinel fits are added. Process acceptance requires normal completion and finite outputs/history; no truth, TP/FP/FDR/recall or reconstruction outcome controls training/stopping. Preserve all new checkpoints, final arrays and model files; no cleanup in this pilot.

## Frozen spatial evidence

For each solver-reported molecular lipid_name, sum ALL same-name production X_hat channels at every original foreground pixel. No thresholding, ROI, top-percentile, truth weighting or per-identity refit. Reporting itself remains the original any-candidate foreground mean>0.001 rule.

Normalize spatial weights by their sum; compute weighted B mean, then the same L1 spectral normalization used by v1. Freeze the weights and local spectrum before all full/delete LPs. Deleting an identity removes all its solver columns and associated uncertainty weights, never recomputes the spatial weights or observation. A zero weight sum gives NO_SPATIAL_EVIDENCE with no global fallback. The global control uses the same reported molecular universe and all-candidate solver dictionary.

Local evidence remains data-adaptive because X_hat comes from B. Validate the entire pipeline; no claim that this is a new independent measurement. Keep the spatial weights, sums, observations and source array hashes for independent reconstruction checks.

## Numerical decision and accounting

Gamma_num=1e-6, from the frozen global-bound gap tolerance. Retain only full_upper<=epsilon-gamma and deleted_lower>epsilon+gamma. LP objective, nonnegative component coupling, finite redundant bounds,1e-8 outward proof guards and<=1e-6 primal/dual gap are inherited. Identical HiGHS/default then IPM retry shares60 seconds; full LP cost cap30 seconds. Numerical failure is a nonselection, never evidence of identity necessity.

Mutually exclusive status priority: no spatial evidence; numerical unresolved; full model incompatible if full_lower>epsilon+gamma; otherwise full-model threshold unresolved unless full_upper<=epsilon-gamma; then retained if deleted_lower>epsilon+gamma, replaceable if deleted_upper<=epsilon-gamma, otherwise threshold unresolved. Output status vocabulary is exactly NO_SPATIAL_EVIDENCE / NUMERICALLY_UNRESOLVED / FULL_MODEL_INCOMPATIBLE / RETAINED / REPLACEABLE / THRESHOLD_UNRESOLVED. Failed/full-threshold cases cannot be rescued by a deletion result.

All nonselected truths remain in retention/recall denominators. Preserve125 truths per case, including omitted truths. Report raw solver TP/FP/FN separately from filter-induced loss and final FN. Empty sets are not risk-control successes. Report all statuses, distinct identities, all-truth and reportable recall.

## Input novelty before performance scoring

Construct all new solver outputs/evidence before scoring; EVAL access at this stage is input/provenance/novelty only. Never use EVAL deletion scores or performance to select cases, weights or thresholds. Compare new CAL vs EVAL and EACH new realization vs BOTH old run04 realizations, separately per challenge. Old source arrays are used only to reconstruct comparison inputs; no old v1 score, threshold or performance selection enters V2.

Record per-identity normalized local-spectrum L2 and relative L2 differences, normalized-weight overlap, L1 distance and Pearson correlation, plus raw/normalized global mean changes. Before outcomes, define substantive novelty as at least20 common reported nonzero-weight identities and at least50% having BOTH local relative L2>0.001 and normalized-weight L1>0.05, for every comparison. These are fixed input-novelty checks, not spatial weight selection thresholds. Failure stops without redrawing; all comparison records remain.

## Calibration, evaluation and stopping

Global and local share U, case membership, candidate universe, gamma and calibration rule. Each chooses its own epsilon using only the three R61 CAL cases: maximize TP subject to aggregate observed FDP<=1%; ties fewer FP then larger epsilon. Enumerate classification events with their adjacent floating-point values; no positive selection at an unresolved boundary. Empty calibration gives epsilon1+gamma and an empty selection. Both method seals must exist and their hashes be checked before any R62 LP/performance evaluation.

Local EVAL GO requires simultaneously aggregate FDP<=1%, aggregate TP retention>=40%, EACH of MILD/close-neighbor/relatively-isolated retention>=40%, and EACH retained set nonempty. No added per-arm FDP criterion beyond the user's explicit aggregate rule; nevertheless report each arm's FDP prominently. Aggregate retention>=60% is strong success only after all GO conditions pass. Global is evaluated with the identical rule as a paired control.

Stop after the reviewed pilot. A favorable result warrants designing independent validation, not declaring the main endpoint achieved. A failed result stops this version; do not tune U/epsilon/gamma/weights/denominators on exposed EVAL. Preserve source, numerical evidence, counts, calibration seals and reports; push reviewed compact records. Do not launch V59/V60, quantification or an open-set variant.

The bounded postprocessor analysis/audit_ce_identity_spatial_v2_job.py audits each completed training/evidence case, exports hashed compact archives, then independently checks the completed paired pilot and writes its descriptive report. It exits after completion/failure and uses no model calls while waiting. It never modifies scientific inputs, retries failed experiments or deletes files. Archive generation is not a Git push: this thread separately downloads, verifies and pushes each reviewed snapshot.
