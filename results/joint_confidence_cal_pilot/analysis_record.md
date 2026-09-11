# Joint rho/abundance grouped CAL screen: criteria not met

Five identity-disjoint train/calibration/evaluation rotations;11310 molecular observations from30 CAL datasets. Original HOLD outcomes were not read. Fixed quadratic logistic model with only rho/abundance-derived features, no hyperparameter search. Each observation is evaluated once outside its identity training/calibration groups. Shared simulated cubes, previously examined CAL data and overlapping training folds still limit independence.

| Severity | Score | TP | FP | Observed FDR | All-truth recall | Solver FN | Filter loss |
|---|---|---:|---:|---:|---:|---:|---:|
| MILD | rho_zero | 320 | 14 | 4.192% | 18.29% | 53 | 1377 |
| MILD | X_hat | 570 | 37 | 6.096% | 32.57% | 53 | 1127 |
| MILD | joint_probability | 538 | 36 | 6.272% | 30.74% | 53 | 1159 |
| MODERATE | rho_zero | 360 | 52 | 12.621% | 20.57% | 56 | 1334 |
| MODERATE | X_hat | 538 | 44 | 7.560% | 30.74% | 56 | 1156 |
| MODERATE | joint_probability | 514 | 39 | 7.052% | 29.37% | 56 | 1180 |

Thresholds were chosen at nominal5% empirical risk on separate calibration identities, pooled across severities. These evaluation numbers are not retrospective best-threshold envelopes and are not directly interchangeable with the preceding single-CAL screens. Both joint-rule severity FDRs exceed5%; recall does not surpass the abundance baseline. The prespecified criterion of <=5% FDR plus10pp recall gain over both baselines in each severity fails. STOP_THIS_VERSION_PREDEFINED_CRITERIA_NOT_MET.

Per-fold results show heterogeneity (joint5% observed FDR reaches13.33% MILD and17.5% MODERATE in one fold). Do not pool away that variation or claim the predicted logistic probability is calibrated for real annotation. This failure does not prove all possible feature combinations impossible. It rules out the fixed tested model as the proposed solution. No post-outcome feature/hyperparameter changes or original HOLD retuning.

Independent review checks all30 source hashes, every source truth/reporting record, unique out-of-group membership, per-rotation group disjointness, all saved model probabilities (max difference2.22e-16), independently recomputed calibration thresholds, seals and retained flags. Full models, groups, observations, protocol and executed runner are retained. Protocol/code pushed as8617635 before execution. Archive SHA2568ecfc02783adc2b66957e9a55c347ef4ff45ba7f78fc9701de53d604e3278b11. No production solver training or rho rerun, no GPU, no cleanup. The secondary logistic models were fitted on CPU.
