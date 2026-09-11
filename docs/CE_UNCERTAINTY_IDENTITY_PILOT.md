# CE-informed identity pilot v1 — prospective development contract

2026-09-11. User requested a short route to FDR<=1% and TP retention>=40%, followed by one independent formal validation.60% is strong success only. This is one new method and a six-context CPU pilot, not another benchmark expansion or an impossibility study. Original scientific definitions remain unchanged.

CE boundary is complete in results/ce133_uncertainty_v1_ready.133 exact identity/adduct/rule keys yielded244 usable cross-energy pairs from120 identities. Positive active channels must be wholly owned and unambiguously assigned at both energies, with consistent target mass and at least2 trusted shared channels. Center log intensity changes by their shared-channel mean to remove the common amplitude factor. For each rule/channel, take each identity's maximum absolute centered change, then its empirical95th percentile; allow symmetric reciprocal intensity multipliers only if at least10 distinct identities support that stratum. These estimator choices precede screening outcomes and are not a tuning grid.

17 supported strata map to480 physical fragment components in107 of391 production candidates. Data support PC+CH3COO, PEO-H and PS-H strata; applicability to the production candidate set is explicitly restricted by source ID and component mapping. Unsupported candidates/channels stay fixed. No new peaks, disappearance, mass shifts or precursor-component changes. No Gaussian/background noise surrogate. These are CE-derived developmental envelopes, not validated95% coverage of real MSI prediction error.

The original source stick library and original positive peak response decompose a candidate into fixed contribution plus uncertain fragment contributions. Each unambiguously assigned fragment's existing isotope sticks and profile response share one multiplier. Their contribution fractions sum to at most1; applying unit multipliers reproduces the input A. Box combinations are a conservative outer envelope: measured co-variation is reported, but unobserved combinations are not claimed experimentally validated. No original A, X_true or B is modified.

First implementation preflight corrected omitted original column normalization in a parity check, retained25 separately sourced candidates as fixed rather than inventing source spectra, and corrected NumPy integer JSON serialization. The partial export results/ce133_uncertainty_v1 is preserved; the completed input for the pilot is exclusively results/ce133_uncertainty_v1_ready. No screening result existed during these corrections.

## Convex compatibility calculation

Use foreground-mean b and original candidate competition, including unreported candidates. Minimize relative L1 residual over nonnegative abundance x and fragment contribution weights w with l*x<=w<=u*x. This is a linear program. It represents the specified component uncertainty exactly; it is not a reimplementation or modification of rho_zero. The L1 residual metric is fixed for this new pilot before scoring.

Deleting a molecular identity forces all its candidate x and associated w to zero. Compare full and deleted absolute residual bounds. The full feasible upper bound must be <=epsilon and the deleted global lower bound must be >epsilon for retention. Sparse HiGHS LPs return feasible witnesses and dual vectors; recompute weak-duality bounds with long-double accumulation and1e-8 outward guards. Bound gaps must be<=1e-6. All proof vectors are retained for review. A local failure never becomes identity evidence.

The redundant finite LP bounds follow from zero prediction having relative L1 residual1 and nonnegative predicted mass: relevant optimal predicted total mass<=2. x is bounded conservatively by4 divided by its minimum allowed column mass, w by its upper multiplier times that x bound. These computational bounds do not add a biological abundance prior. Cost gates: full LP<=30s; first8 deletion median<=10s; any individual LP time limit60s. No cost-driven scientific simplification after results.

## Six fixed cached contexts and operating rule

DEVELOPMENT_CAL: MILD__CAL_R1_K125, close_neighbor__HOLD_R1_K125, relatively_isolated__HOLD_R1_K125.

DEVELOPMENT_EVAL: MILD__CAL_R2_K125, close_neighbor__HOLD_R2_K125, relatively_isolated__HOLD_R2_K125.

All are previously generated, known-truth developmental data, including original HOLD contexts reused explicitly for this new development task. They are not unseen final HOLD. Source solver reporting gate remains0.001; no candidate removal based on truth. Raw TP denominators and all125 truths per context, including5 omitted identities in each missing-library context, remain in accounting. The shared source observations and repeated identities are not independent biological replicates. The missing-library challenges here are the existing clean-spectrum omissions; this pilot does not claim a combined mismatch-plus-omission test.

Choose a single epsilon on the three R1 calibration contexts to maximize retained TP subject to observed FDP<=1%; ties prefer fewer FP then larger epsilon. Candidate/full residual event values define all changes in selection. If no nonempty admissible set exists, use epsilon1, return no certified identities, and preserve the negative result. epsilon is a development-calibrated operating tolerance, not an estimated measurement-noise level. Save its seal before R2 scoring. No R2 outcome may change U, epsilon, cases or method. Calibrate secondary rho/X_hat baselines on the same R1 universe without changing their original scores.

Count R2 results over all three planned contexts, including whole-case abstentions with zero retained TP. Continue only if observed aggregate FDP<=1% AND TP retention>=40%, with a nonempty retained set;>=60% is strong success. Report each challenge separately so pooled performance cannot hide a failing domain; only supported domains could enter a later formal claim. CAL success alone is insufficient. A favorable R2 development result proceeds to independent design/validation, not immediate completion of the main objective.

Every original source must pass runtime/input binding, normal finite completion, result membership and listed artifact hashes before use. Preserve compact source records and numerical proofs, review cached outputs, then Git. Do not train, rerun original rho, change original thresholds, delete artifacts or start independent validation automatically from an unreviewed pilot result.

The formal endpoint remains independent CAL/HOLD FDR<=1% and TP retention>=40%, accompanied by recall, distinct identities, abstention and risk uncertainty. A negative pilot stops this version, not the scientific objective; it does not establish observational impossibility.

## Numerical execution note — before any calibration or evaluation outcome

The first input-only attempt stopped at a missing required V58 parser mode; the call was corrected without changing inputs. The next attempt verified all six input sets, then the HiGHS default algorithm returned an invalid status during the first MILD CAL deletion sequence, before any case or calibration result existed. Both stopped directories are preserved. A computational recovery uses the identical LP: try the original HiGHS algorithm, then HiGHS IPM only on solver non-success, sharing the original total60-second budget and the same tolerances. Every accepted result still needs the independent feasible/dual bounds; no failed solve becomes identity evidence. Save partial numerical evidence every25 identities and on failure. These are execution/recovery changes, not changes to U, the objective, thresholds, cases or risk/retention targets.
