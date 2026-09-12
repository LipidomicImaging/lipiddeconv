# Joint confidence development screen: improvement, strict target not met

The unchanged joint rho/abundance classifier retained 93 true and 1 false molecular identity: recall and TP retention 74.4%, observed FDP 1.063829787%. It improves on both single-score controls under the same molecular grouping and calibration, but **does not pass FDP <=1% and recall/retention >=80%**. Neither value is rounded into a pass. This one fixed screen is complete; no parameter, cutoff or feature retry is authorized by its result.

| Method | TP | FP | FN | Observed FDP | All-truth recall / TP retention |
|---|---:|---:|---:|---:|---:|
| Raw NNLS | 125 | 49 | 0 | 28.1609% | 100% |
| rho, separate CAL cutoff | 61 | 3 | 64 | 4.6875% | 48.8% |
| Abundance, separate CAL cutoff | 84 | 2 | 41 | 2.3256% | 67.2% |
| Fixed joint classifier, separate CAL cutoff | 93 | 1 | 32 | 1.0638% | 74.4% |

All thresholds target empirical CAL FDP <=1%, then apply unchanged to their corresponding test identities. The table aggregates the five out-of-group tests. These are not the earlier in-sample best-cutoff tradeoffs (60/0 for rho and 54/0 for abundance); the evaluation procedures differ. The new joint result neither changes the original experiment's 40% developmental GO nor establishes the user's higher 80% target.

Relative to the abundance control, joint filtering adds nine true identities, removes one false identity, and loses no previously retained true identity on this dataset. Relative to rho it retains 32 more true identities and two fewer false identities. The remaining joint false positive is `PE O-16:1_22:3`; naming it is accounting only and must not become a blacklist or a feature-selection cue for this exposed test. All 32 joint false negatives are filter-induced losses; raw NNLS misses zero truths. All 125 reportable truths are in the denominators and were actually perturbed in the source case. Every retained set and original molecular record is saved in `fit/out_of_group_records.json`.

## What was frozen and verified

Pre-execution preparation commit `4b72366f0b2308c84051ed089259b33357604ca8` was successfully pushed and verified before fitting. Contract SHA256: `b01e93557a98b5d45f3171180ccbd81bc4685a1ada3aa88e8c5016c3a2d9ccc1`. The fixed existing six-feature logistic engine ran once (five prescribed folds), with disjoint identity membership for TRAIN/CAL/TEST within each fold and separate pre-test cutoff seals. No simulation, NNLS, rho or GPU calculation was repeated. No whole-data refit or next case followed.

Remote cached review independently reconstructed TRAIN normalization, saved-coefficient predictions, tied-score CAL thresholds, test flags, membership and counts. Its maximum probability reconstruction discrepancy was 7.77e-16. All 12 exported files matched hashes locally; a second cached local review reproduced all method and fold counts, with maximum prediction discrepancy 5.55e-17. Review requires no model fitting. Syntax compilation, CLI help and Git whitespace checks passed. Original source artifacts remain unchanged; nothing was deleted.

Remote storage: `/root/small_mismatch_joint_confidence`; original source: `/root/small_mismatch_nnls_first/results`. Compact local records include preparation/input hashes, five model/cutoff seals, all molecular outputs, saved coefficients, provenance, execution receipt and both reviews. Transport archives stay outside lightweight Git.

## Scientific limits and stopping decision

This case's complete outcomes were already exposed before the new user request. Grouping limits direct label reuse in a fold, but all groups share the same library, mixture, perturbation draw and spatial realization. Across folds an identity may train another fold's model. This is within-case development, not fresh independent validation; the algorithm's `joint_probability` field is a ranking score, not a calibrated probability of correct annotation. Empirical zero-FP calibration on small groups did not ensure <=1% out-of-group FDP, and one observed FP in 94 retained identities is not a stable population FDR estimate.

The measured gain supports pursuing a combined strategy, while the strict 1%/80% objective remains unmet. Future claims require a separately frozen method and independent cases representative of deployment; adding more feature/model searches to these exposed folds cannot supply that evidence. This execution stops after result preservation in Git. No automatic deconvolution, new case, monitor or cleanup is scheduled.
