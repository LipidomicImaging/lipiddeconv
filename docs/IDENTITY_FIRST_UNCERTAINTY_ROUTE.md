# Identity first: CE-informed uncertainty development route

2026-09-11. Status: OBJECTIVE_AND_UTILITY_TARGETS_FIXED; UNCERTAINTY_MODEL_AND_PILOT_NOT_YET_FROZEN.

The primary objective is a useful subset of molecular identities with very low false allocation under realistic spectral mismatch and missing-library challenges. Quantification follows identity validation. This route supersedes new benchmark expansion as the main development priority, without changing any frozen production definition or retrospectively applying new success criteria to completed pilots.

## Fixed user targets and accounting

- Risk target: FDR <= 0.01.
- Minimum continuation utility: TP retention >= 0.40.
- Strong utility success: TP retention >= 0.60, while meeting the same risk target.
- TP retention = filtered TP / raw solver TP in the same prespecified reporting universe. If raw TP is zero, report undefined rather than a success.
- All-truth recall = filtered TP / all molecular truth, including omitted library identities. Report solver misses separately from filter-induced true losses.
- Observed false-discovery proportion = FP / (TP + FP). This point estimate alone is not proof of population FDR control. Empty retained sets have zero discoveries and do not pass utility.
- Retain case-level outcomes, pooled counts, distinct retained identities, abstention/out-of-model rates and numerical unresolved states. Repeated identities, pixels and nested datasets are not independent replicates.
- An abstained case contributes zero retained TP and retains its original raw-TP/truth denominators. Never report a selected non-abstained subset as the complete efficacy result.

These numeric targets are fixed before the new pilot. Exact case membership, risk estimand, aggregation weights, challenge-specific pass rules and uncertainty interval construction remain to be frozen before scoring. Do not choose them using new pilot outcomes. In particular, safe whole-case rejection of missing-library observations can pass a safety check while failing utility; it is not evidence of simultaneous low risk and useful recovery. No silent exemption from the retention accounting is allowed.

## Stage 1: bounded CE133 data capability audit

Reuse results/rho_mismatch_mechanism_ce_audit and the one directly referenced target CSV. The preceding audit found133 exact-key identities at two or more of CE30/35/40 and66 at all three. Keys are structure_text/adduct/rule_sheet; do not silently merge identities or change production identity definitions.

Audit only the matched records and directly referenced processing/source evidence:

1. Align fragment/rule channels and applicable-channel masks across CE.
2. Separate observed peak presence/absence from disabled channels, ambiguity, detection censoring and unassigned peaks. Do not interpret every table zero as a disappeared physical peak.
3. Preserve shared-peak ownership, multi-candidate source spectra and dependence. Assess peak-intensity co-variation only where assignments and normalization support it.
4. Summarize supported variation by fragment/rule category and identity, including denominators, sparse strata and missing evidence.
5. Audit raw-source availability and existing OOF lineage through direct references only. No recursive historical search, raw MSI processing or retraining.

Output a data-supported proposal for allowed spectral variation U with provenance and explicit unsupported aspects. An honest DATA_INSUFFICIENT result is allowed; do not force an uncertainty model from inadequate evidence. No arbitrary sigma/dropout, new confidence score, threshold tuning or GPU work in this stage.

These are selected DDA rule annotations already used by the final production CE adapter, not independently verified pure standards or unseen final validation. Cross-CE variation does not itself quantify same-condition noise, instrument transfer or all prediction-to-MSI error. Cached OOF predictions require an upstream leakage audit before use. Do not treat production reconstruction residual as empirical measurement noise.

## Stage 2: freeze the uncertainty and identity decision contract

Let d_full be the minimum observed-data residual over the prespecified allowed library model and nonnegative abundances. Let d_-j be the corresponding minimum after deleting every allowed representation of molecular identity j. Both optimizations must admit all competing library candidates, not an oracle list of true identities or just the reported subset. Fix the data domain, normalization, residual metric, allowed joint spectral variations and epsilon_valid before pilot outcomes.

The intended sufficient decision is d_full <= epsilon_valid AND d_-j > epsilon_valid. epsilon_valid needs independent error-calibration evidence and a declared scope; it is not a post-hoc fit tolerance chosen to make cases pass. Penalized objective differences alone do not establish data compatibility.

- Full model proved incompatible: ABSTAIN / OUT_OF_MODEL; confirm no identities for that case.
- Full feasible fit, and a valid lower bound proves deleted-model residual exceeds epsilon_valid: identity is necessary conditional on the uncertainty/observation model.
- A feasible deleted-model explanation exists: identity is not certified.
- Numerical failure, missing convergence evidence or an unresolved optimization gap: ABSTAIN / NUMERICALLY_UNRESOLVED, not a proof of identity or out-of-model status.

Local optimizer failure to find an alternative explanation is not proof that no explanation exists. If certification bounds are unavailable, label any resulting output an empirical screening rule rather than a mathematical certificate. Calibration and independent assessment remain required in either case.

Missing-library contributions are not automatically covered by per-library-column U. Prespecify any unknown-component model and constrain it from evidence; unrestricted unknown signal would defeat identifiability. Some unknown spectra lie within the allowed known-library cone and are observationally indistinguishable. Passing sampled missing-library challenges does not guarantee universal unknown detection.

## Stage 3: bounded developmental pilot

Reuse existing known-truth MILD and available missing-library observations after required provenance/CAL checks. Freeze a small exact input/identity-evaluation manifest and CPU cost limit before new scores; clarify whole-mixture datasets versus candidate-deletion evaluations. Do not launch dozens of new learned fits.

Previously examined CAL/HOLD records are developmental evidence for this new route. They cannot become an unseen final test by relabeling a directory or making a new split after exposure. Compare frozen original rho and abundance baselines on the same reporting universe. Apply the fixed1% /40% /60% targets without outcome-driven adjustment. A favorable pilot justifies independent validation, not an immediate FDR guarantee.

Failure diagnosis must distinguish model undercoverage, overly permissive variation, unmodeled components, observational ambiguity, optimization failure and statistical uncertainty. Failure of one implementation is not a proof that every identity rule is impossible. Preserve negative results; do not repeatedly tune this exposed pilot until it passes.

## Stage 4: independent risk/utility validation

Freeze and audit model-development, calibration and evaluation separation across identities, source spectra/acquisitions and upstream predictors. Establish an actually defensible independent validation set; existing multi-CE training rows alone do not supply it. Report risk uncertainty using the declared sampling/dependence structure, alongside TP retention, all-truth recall, distinct identities and abstention. Do not equate a small observed FP count or a pooled1% proportion with a demonstrated FDR guarantee.

Independent calibration is a substantive part of risk control, not a name attached to a threshold; see [Bates et al., Distribution-Free, Risk-Controlling Prediction Sets](https://arxiv.org/abs/2101.02703). This source motivates statistical separation; it does not establish that its guarantees apply to our dependent, shifted lipid data.

## Stage 5: quantification, deferred

Only after the identity stage is assessed, design matched known-support, full-deconvolution and confusion-group quantification experiments. Known support is an oracle diagnostic, not a deployable method. Its contrast with full-library estimation measures the effect of supplying support under the chosen estimator; it is not automatically a pure causal estimate of false-allocation damage because feasible sets, conditioning, shrinkage and their interactions also change.

Prespecify confusion groups using library information, preserve the frozen production groups, and do not merge identities merely to rescue weak results. Group abundance reliability must be tested, not assumed. A sum of coefficients from normalized library columns is not automatically a sum of molecular concentrations. Declare whether the estimand is synthetic coefficient, reconstructed ion signal, relative response or response-calibrated molar amount. Absolute lipid quantification requires a defensible response/calibration model; see [Torta et al., concordant ceramide concentrations via authentic standards](https://www.nature.com/articles/s41467-024-52087-x).

## Execution priority and boundaries

Next authorized development task is the bounded CE133 capability audit; this document does not claim that audit, U, epsilon_valid, the pilot manifest or independent validation are already complete. New V59/V60 expansion and large quantification experiments are deferred. Existing experiments still require completion/provenance review and Git preservation under AGENTS.md. This planning update neither terminates remote jobs nor deletes their artifacts. The missing-library challenge remains directly relevant to the main objective. Production ISTA, rho_zero, reporting gate and all frozen V57/V58/V59 science remain unchanged.
