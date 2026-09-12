# Observable auxiliary confidence exploration — frozen V1

2026-09-12: the user explicitly authorizes exploring other auxiliary indicators. This is a new bounded development study after the completed rho/abundance screen; neither the original NNLS case nor any prior method result is changed. The full single-case outcomes and the prior five test folds have already been exposed. Prospective freezing here limits further choices within this turn; it does not undo adaptive reuse across turns or create an independent test.

## Inputs and four auxiliary quantities

Use only the hash-bound completed `NNLS_FULL_LIBRARY_5PCT__CAL_R71_K125` records, nominal `A_solver`, saved `X_hat`, foreground mask and already-computed global-mean NNLS coefficients `rho_details.x_star_candidate`. No new solver/certificate/rho calculation. The feature function receives no truth labels, lipid class, perturbed target library, true abundance, or observation B. Identity names are used only for alias aggregation, excluding same-name competitors and the unchanged fold assignment.

Let mu_i be the saved candidate mean abundance over foreground; R_g are the originally reported aliases of molecular identity g. All four quantities use that same frozen reporting membership. Let w_i=mu_i/sum_{h in R_g}mu_h and c_ik be cosine similarity between unit-normalized full nominal columns. All 391 candidates, including unreported candidates, participate as other-name competitors. In this nonnegative library clip numerical cosine errors to [0,1].

1. **Library separation**: sum_{i in R_g} w_i sqrt(max(0,1-max_{k: name(k)!=g}c_ik^2)). This extends the existing nearest-column geometric convention to reported molecular aliases. It measures pairwise full-channel separation, not fragment-specific evidence or separation from the complete competitor cone.
2. **Competition-adjusted abundance**: log10(mu_g/(P_g+1e-12)), where mu_g=sum_{i in R_g}mu_i and P_g=sum_{i in R_g}w_i sum_{k: name(k)!=g}c_ik^2 mu_k. The squared cosine weights all other-name candidates continuously; no top-K, competitor cutoff or outcome-selected neighborhood. Full nominal library norms are approximately one under the original convention.
3. **Global/mean coefficient disagreement**: abs(log10((sum_{i in R_g}x_global_i+1e-12)/(mu_g+1e-12))). The global fit is the original cached full-library NNLS on foreground mean B used for rho; the denominator averages the original pixelwise NNLS. This is disagreement between two existing summarizations, not a new solver or proof of identity error.
4. **Spatial concentration**: log10(N_eff,g/P), with s_g(p)=sum_{i in R_g}X_hat_i(p), N_eff=(sum_p s_g)^2/sum_p s_g^2, and P all original foreground pixels. No threshold, ROI or reweighting. This describes effective pixel coverage, not technical repeatability, perturbation stability or molecular correctness. A false allocation can have coherent spatial structure. V2's evidence/certificate/weights remain unchanged.

No feature is defined for unreported identities because they cannot be retained; all such truths remain in recall and solver-miss accounting. Preserve the exact molecular aliases and reportable-truth definition. Numeric floors are arithmetic guards, not physical uncertainty parameters. The independent feature checker uses equivalent scalar cosine sums and E[s]^2/E[s^2]; at exactly overlapping columns sqrt(1-cos^2) can differ by O(sqrt(machine epsilon)), so its arithmetic check allows 5e-8 absolute error. This does not change selection thresholds or any scientific gate.

## Fixed comparisons and calibration

Reuse the previous exact five TRAIN/CAL/TEST identity groups, baseline TRAIN mean/scale, original six-feature mapping and baseline model/predictions without refitting. Add only linear, TRAIN-standardized auxiliary features (standard deviation floor 1e-12). All augmented classifiers use the same L2 logistic regression C=1, lbfgs, max_iter=2000, tol=1e-8, no class weights and no search.

- Baseline: previously frozen rho + abundance + squares/interaction/zero-rho indicator.
- PLUS_LIBRARY_COMPETITION: baseline plus quantities 1 and 2.
- PLUS_MEAN_CONSISTENCY: baseline plus quantity 3.
- PLUS_SPATIAL_CONCENTRATION: baseline plus quantity 4.
- ALL_AUXILIARY: baseline plus all four quantities; this is the declared primary extension.

Fit four extensions across five folds (20 tiny CPU classifier fits); no additional variants or reruns. For TEST fold f use CAL=(f+1)%5 and other three for TRAIN. Choose each method's own largest tied-score CAL retained set at empirical FDP<=1%, or no identities if none qualify. Write model and threshold seals before corresponding TEST predictions. No post-hoc test cutoff adjustment, identity blacklist, winning-fold selection, feature removal, interaction search or pooled-probability recalibration.

Report all five methods with TP/FP/FN, raw solver misses, filter-induced losses, FDP, recall, TP retention, coverage and per-fold counts; pair each extension against the exact baseline to distinguish gained/lost TP and gained/removed FP. Report raw feature 10th/50th/90th percentiles by truth and correlation with log abundance/rho as descriptive diagnostics only. Primary point target remains nonempty retained set, FDP<=1%, recall>=80% and TP retention>=80% for ALL_AUXILIARY. Three partial extensions are exploratory ablations; their outcomes cannot replace a failed primary result. Even a point-target hit is not independent FDR validation, especially after exposed-case feature exploration.

## Execution, verification and stopping

Use `analysis/run_small_mismatch_auxiliary_confidence.py prepare/run/review`. Analytical unit checks cover orthogonal/identical columns, same-name alias exclusion, unreported competitors, effective-pixel arithmetic and calibration ties. Preparation freezes exact feature values, source/code/protocol hashes and all variants; push before model fitting. Runtime for fitting is the existing westc CPU environment: NumPy2.1.3, SciPy1.15.3, scikit-learn1.6.1. Source case lives at `/root/small_mismatch_nnls_first/results`; baseline at `/root/small_mismatch_joint_confidence`; new isolated root `/root/small_mismatch_auxiliary_confidence`.

A cached independent formula path verifies features, TRAIN standardization, predictions from saved coefficients, CAL thresholds, groups, every retention flag and metrics without fitting. Download compact records with hashes, repeat cached review locally, preserve all results in Git, then STOP. No GPU, NNLS/rho/certificate optimization, simulation, new physical assumption, new case, automatic monitoring or deletion. Do not treat this single-case feature development as a deployable classifier or calibrated annotation probability. Subsequent validation would require a separately frozen procedure and independent cases.
