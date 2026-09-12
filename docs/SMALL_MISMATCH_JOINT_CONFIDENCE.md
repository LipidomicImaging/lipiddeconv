# Fixed joint rho/abundance development screen

User request, 2026-09-12: develop other or combined confidence indicators to improve the 1% error / 80% recall tradeoff. This authorizes one fixed, cached-data CPU screen; the original NNLS experiment and its 1%/40% decision remain unchanged.

## Scope and prior exposure

Use only the completed `NNLS_FULL_LIBRARY_5PCT__CAL_R71_K125` molecular records, their result/design/summary bindings, and the existing `analysis/run_joint_confidence_cal_pilot.py`. This case has 125 reported truths and 49 reported false identities. Its complete outcomes and single-score curves are already exposed. Molecular grouping prevents the same identity from training and testing in one fold, but all folds still share one mixture, library, spatial realization and perturbation draw. This is exploratory within-case development, not fresh independent case validation, formal FDR control, or real-data validation.

Reuse the existing joint engine without changing it. Its earlier V58 stress-domain screen did not meet its target; that result remains in `results/joint_confidence_cal_pilot/analysis_record.md`. The present all-library 5% assumption is a different, user-requested domain. No retrospective rescue of the old experiment is claimed.

## One fixed model and comparison

Observable features are TRAIN-standardized log10 abundance (floor 1e-12) and log10 rho (floor 1e-24), their two squares and interaction, and the rho-equals-zero indicator. Fit the existing L2 logistic regression (C=1, lbfgs, max_iter=2000, tol=1e-8); no feature search, class weighting, hyperparameter search or seed search. The output named `joint_probability` is an uncalibrated ranking score, not an identity correctness probability.

Use the existing SHA256 identity grouping namespace `joint_confidence_CAL_v1|`, modulo five. For fold f, TEST=f, CAL=(f+1)%5, TRAIN=the other three groups. Report every identity once in the pooled out-of-group test. Only reported identities train or set thresholds; all truths, including any solver misses, stay in recall denominators. Identity names determine membership only; truth supplies TRAIN labels, CAL threshold selection and final TEST accounting. Neither identity/class/geometry nor perturbed target spectra enter the features.

Within each fold compare joint, rho alone and abundance alone under exactly the same split and CAL rule. Each selects its own cutoff: retain the largest tied-score CAL set with empirical FDP <=1%, or retain nothing if no nonempty set qualifies. Keep the existing engine's 5% secondary outputs for reproducibility; they cannot rescue the primary decision. Freeze each fold's model and cutoffs before its TEST predictions. No subsequent model refit on all data and no choice of the best fold.

Primary requested development target: pooled joint retained set nonempty, FDP <=1%, all-truth recall >=80%, and TP retention >=80%. With 125 raw true positives the last two measures coincide. Report TP/FP/FN, solver misses, filter-induced true losses, recall, retention, coverage, and each fold's outcomes for all three methods. The small CAL groups' zero-FP cutoffs do not guarantee <=1% error on unseen data. Do not compare pooled out-of-group results with the earlier in-sample best-cutoff points as if their evaluation conditions were identical.

## Execution and stopping

`analysis/run_small_mismatch_joint_confidence.py prepare` freezes exact source, implementation, protocol and derived-input hashes plus memberships before fitting. Push this preparation before execution. Runtime versions: NumPy 2.1.3, SciPy 1.15.3, scikit-learn 1.6.1. Run only the unchanged small CPU classifier on westc; no NNLS, rho, simulation, raw MSI, GPU or new case. A cached independent reviewer verifies input/output hashes, group exclusion, TRAIN normalization, predictions from saved coefficients, CAL cutoffs, retention flags and denominators without fitting again.

After this one screen, preserve compact results and review in Git and stop. A negative result does not prove all joint methods impossible; it also does not authorize trying more settings on these exposed folds. A positive result motivates separately frozen new-case validation, not automatic new computation. No files are deleted.

Multivariate postprocessing has precedent in proteomic identification, for example [Käll et al., Nature Methods 2007](https://www.nature.com/articles/nmeth1113). That precedent motivates checking complementary features; its peptide target-decoy validity does not transfer to lipid mixtures or establish risk control here.
