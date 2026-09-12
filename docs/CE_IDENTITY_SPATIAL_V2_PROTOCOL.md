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

## Final report: method decision and spatial contribution are separate

User clarification2026-09-12, recorded during execution without changing any case, score, threshold, denominator, numerical guard, source binding or running code. Complete the existing CAL seals -> EVAL -> independent review -> decision sequence. Add no intermediate experiment or outcome-driven patch.

First report METHOD_GO or METHOD_NO_GO using exactly the existing local EVAL rule: aggregate FDP<=1%, aggregate TP retention>=40%, each of the three challenges TP retention>=40%, and each retained set nonempty. Aggregate retention>=60% remains strong success only after GO. Global performance is not an additional condition in the method GO formula. A reviewed method GO permits designing the next physical-mismatch validation; it does not establish real-experiment robustness.

Then report the paired spatial-contribution interpretation separately, applying the identical frozen GO rule to global:

| Local EVAL | Same-batch global EVAL | Interpretation for this pilot |
|---|---|---|
| GO | NO-GO | Strong paired evidence that spatial conditioning adds useful discrimination under this fixed benchmark. |
| NO-GO | NO-GO | V2 fails its method gate; partial numerical improvements cannot override NO-GO. |
| GO | GO | Feasibility signal for the identity-screening route, but crossing the gate cannot be attributed to spatial conditioning. Easier new cases remain an alternative explanation; do not assert that explanation is proven. |
| NO-GO | GO | Spatial conditioning fails where the global control succeeds; stop this spatial version and report the adverse paired result. |

Show each missing-library arm separately: raw solver TP, retained TP/FP, FDP, TP retention, all-truth recall, abstention/status counts and local-minus-global differences from the existing outputs. State whether each local arm recovered a nonzero subset and whether it reached40%; these are different achievements. V1's historical zero retention is context only because V2 uses different cases. Attribute spatial contribution using the same-batch global control, not a direct V1-to-V2 outcome comparison.

This reporting clarification supersedes any earlier wording that merged method GO with proof of spatial benefit. The physical model below remains a next-stage validation boundary. No new cases, physical perturbations, diagnostics for selection, retuning or changes to the current V2 implementation are authorized by this clarification.

## Physical interpretation and future-validation boundary

Documentation addendum authorized 2026-09-12, after V2 execution began. This records interpretation and future model scope only. The six cases, frozen design and source bindings, CE uncertainty v1, gamma, epsilon calibration rule, production GPU fits, full/delete LPs, reporting and accounting, and local/global evidence definitions remain unchanged. Keep the originally deployed protocol/source snapshots and their hashes; this addendum does not replace or rebind them. No new physical model or diagnostic is added to the running code.

### Fixed library, systematic mismatch and pixel variation

The library A_lib is fixed. The proposed interpretation for future validation distinguishes the experiment's mean spectrum from its pixel-specific spectrum:

```text
A_exp   = A_lib + Delta_A_sys
A_exp,p = A_exp + Delta_A_pixel,p
b_p     = (A_lib + Delta_A_sys + Delta_A_pixel,p) x_p + epsilon_p
```

Delta_A_sys denotes library-to-experiment systematic spectral mismatch. Delta_A_pixel,p denotes within-experiment spectral variation. Epsilon_p denotes measurement error beyond that spectral model. This notation separates mechanisms; it does not identify them from a single observation, license negative spectra, or specify arbitrary Gaussian noise. Weak-peak censoring must be modeled as an observation/detection mechanism if supported, not silently absorbed into an unrestricted additive error.

For the future physical model, initially allow relative intensity variation on established fragment support. Preserve high-intensity, empirically stable fragments rather than randomly deleting them. Permit weak-fragment censoring or occasional non-detection only where supported by S/N, acquisition thresholds or repeatability evidence; do not extend it to strong peaks. Do not introduce new high-intensity unknown fragments by default. Unmodeled fragmentation belongs to a separately assessed out-of-model mechanism. These are proposed scope constraints, not claims that every strong fragment is invariant across all experimental conditions.

Systematic mismatch cannot be assumed to vanish through spatial averaging. The proposal that within-experiment pixel variation is smaller, and that averaging partly reduces it, is a hypothesis to validate. Correlation across pixels, systematic detection bias and weights estimated from the same data can limit or prevent that reduction. No noise magnitude, independence, zero-mean property or averaging benefit is asserted by V2.

### What CE30/35/40 supports

CE30/35/40 informs condition-dependent systematic spectral-shape variation; it is not a pixel-noise measurement. The existing CE133-derived v1 envelope remains a limited shared-fragment intensity model, not a demonstrated bound on all library-to-experiment errors. CE variation alone establishes neither its coverage of prediction/instrument mismatch nor within-MSI pixel variance.

Future pixel variability and weak-peak detection behavior need separate evidence from MSI repeatability, independently justified relatively homogeneous regions, technical repeats or standards. Spatial abundance/composition differences must be distinguished from spectral fluctuations; a visually uniform region alone does not establish that distinction. Unsupported mechanisms and ranges remain NOT_VERIFIED and are not added to the allowed uncertainty merely to improve a result.

### What the current stress test can establish

The inherited MILD target is a controlled synthetic stress test, not the final implementation of the physical model above. Its builder, analysis/run_v58_spectral_library_mismatch_fdr_recalibration.py::build_targets, perturbs nonzero fragment intensities, protects the strongest fragment, randomly drops eligible other nonzero fragments and changes parent intensity before normalization. It creates no new support. Thus it is not literally dropout of every fragment; its eligible-fragment dropout is also not an empirically calibrated weak-peak censoring model. The fixed target is reused across pixels and does not implement a separate pixel-variation process.

Do not describe independent fragment dropout as the established physical mechanism of real MSI mismatch. Keep old settings and results as historical stress tests. V2 asks only whether identity-conditioned spatial evidence improves discrimination over global foreground mean with the inherited mismatch and uncertainty held fixed. Its missing-library arms remain CLEAN omission challenges; they are not combined realistic mismatch-plus-omission validation. Neither success nor failure directly validates or refutes performance under real experimental perturbations.

After the paired pilot is independently reviewed, apply the existing frozen method GO rule unchanged and separately report spatial contribution using the four outcomes above. Method GO motivates designing a separately frozen validation model combining systematic fragment-intensity mismatch, strong-peak preservation, evidence-based weak-peak censoring and separately estimated pixel variability; it does not itself establish spatial superiority. Method NO-GO stops this version without tuning on exposed EVAL. No new numerical definition of "clear improvement" is inserted into V2's sealed calibration or GO decision. Any later confirmatory comparison criterion must be specified prospectively. No future experiment starts automatically from this addendum.

### Optional diagnostic only: spatial weight concentration

For the same fixed molecular weights s_j on the original foreground, define:

```text
N_eff,j = (sum_p s_j(p))^2 / sum_p s_j(p)^2
```

It may be derived later from preserved completed evidence as a separate, provenance-linked diagnostic without rerunning training or changing evidence files. Zero total weight gives an undefined diagnostic and leaves NO_SPATIAL_EVIDENCE unchanged. This describes effective weight concentration, not a verified number of statistically independent pixels. It must never affect reporting, spatial weights, case membership, CAL/EVAL, epsilon, thresholding, retention/recall denominators or GO. No current diagnostic threshold or new filtering rule is introduced.
