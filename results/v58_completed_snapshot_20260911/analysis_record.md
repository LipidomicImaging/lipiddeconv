V58 completed-case storage review — 2026-09-11

Snapshot: 50/60 completed cases. This is an interim review; incomplete MODERATE results are not the final benchmark.

Both severity-specific cutoffs were frozen with the unchanged runner on all15 CAL datasets per severity before this review opened HOLD records. No threshold was tuned using HOLD.

| Severity | Protocol / score / target | HOLD cases | TP | FP | Solver FN | Filter loss | FDR | Recall |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| MILD | CLEAN_FIXED/rho_zero/FDR5 | 15/15 | 1412 | 835 | 70 | 268 | 37.161% | 80.69% |
| MILD | CLEAN_FIXED/rho_zero/FDR1 | 15/15 | 1412 | 835 | 70 | 268 | 37.161% | 80.69% |
| MILD | LOCAL_CAL_RECALIBRATION/rho_zero/FDR5 | 15/15 | 370 | 15 | 70 | 1310 | 3.896% | 21.14% |
| MILD | LOCAL_CAL_RECALIBRATION/rho_zero/FDR1 | 15/15 | 260 | 5 | 70 | 1420 | 1.887% | 14.86% |
| MILD | LOCAL_CAL_RECALIBRATION/X_hat/FDR5 | 15/15 | 520 | 29 | 70 | 1160 | 5.282% | 29.71% |
| MILD | LOCAL_CAL_RECALIBRATION/X_hat/FDR1 | 15/15 | 61 | 0 | 70 | 1619 | 0.000% | 3.49% |
| MODERATE | CLEAN_FIXED/rho_zero/FDR5 | 5/15 | 411 | 272 | 19 | 95 | 39.824% | 78.29% |
| MODERATE | CLEAN_FIXED/rho_zero/FDR1 | 5/15 | 411 | 272 | 19 | 95 | 39.824% | 78.29% |
| MODERATE | LOCAL_CAL_RECALIBRATION/rho_zero/FDR5 | 5/15 | 91 | 12 | 19 | 415 | 11.650% | 17.33% |
| MODERATE | LOCAL_CAL_RECALIBRATION/rho_zero/FDR1 | 5/15 | 52 | 2 | 19 | 454 | 3.704% | 9.90% |
| MODERATE | LOCAL_CAL_RECALIBRATION/X_hat/FDR5 | 5/15 | 121 | 6 | 19 | 385 | 4.724% | 23.05% |
| MODERATE | LOCAL_CAL_RECALIBRATION/X_hat/FDR1 | 5/15 | 21 | 0 | 19 | 485 | 0.000% | 4.00% |

MILD raw: TP1680, FP1514, FN70; truth contexts=1750.
MODERATE raw: TP506, FP490, FN19; truth contexts=525.

The complete MILD HOLD subset shows that CLEAN rho thresholds do not transfer under this spectral mismatch: empirical FDR is37.16%. Local CAL FDR5 recalibration reduces HOLD FDR to3.90%, but recall falls to21.14% and1310 raw true positives are filtered out. The local FDR1 cutoff reaches1.89% HOLD FDR, exceeding its nominal target. This is a substantial reliability/retention limitation, not a successful universal threshold transfer. MODERATE has only5/15 HOLD cases here and must remain provisional.

Storage decision:

Planned retirement: 230 intermediate/redundant checkpoint files, 8.588 GiB. Split: {'root cache': 6415061888, 'data disk': 2806588928} bytes.
Retain final epoch3000 checkpoint, learned X_hat/B_hat arrays, all molecular/candidate records, reports, training history, diagnostics, runtime bindings, CAL threshold seals, design/oracle/adoption provenance, and every sentinel-related case. V57 parent controls and old draft sentinel artifacts are outside the deletion scope.
The completed-result verifier and final aggregator consume retained report/record hashes. The cached learned-result path consumes retained arrays/history/diagnostics. Intermediate checkpoints are not required by these paths. The missing-library challenge references V57 controls, not these compact V58 intermediate checkpoints.
Deleting intermediate checkpoints prevents future reloading of those intermediate epochs; hashes and diagnostics retain the historical record, but are not backups of model weights. Final trained models and final arrays remain available remotely. Original runtime storage manifests remain historical; cleanup_plan and deletion_receipt explicitly supersede existence claims for retired files.
No active/incomplete case is eligible. Actual deletion is a separate step after Git push and a second complete hash verification. Refer to deletion_receipt.json for whether it has occurred.

Interpretation: compare empirical FDR jointly with recall and TP loss. Repeated mappings and nested K share identities and are not independent biological replicates. Partial severity results cannot establish final performance. No scientific definition, training setting, rho calculation or target library was changed.
