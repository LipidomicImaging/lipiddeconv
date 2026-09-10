# V57 computational closure accounting

Recomputed from the existing molecular records using analysis/analyze_v57_computational_closure.py. No solver or rho execution, threshold tuning, or source changes.

| HOLD rule | TP | FP | FN | FDR | All-truth recall | Filter-induced true loss |
|---|---:|---:|---:|---:|---:|---:|
| Raw solver | 3253 | 3212 | 122 | 49.683% | 96.385% | 0 |
| rho_zero CAL-FDR5 | 3253 | 112 | 122 | 3.328% | 96.385% | 0 |
| rho_zero CAL-FDR1 | 3253 | 8 | 122 | 0.245% | 96.385% | 0 |
| X_hat CAL-FDR5 | 1025 | 43 | 2350 | 4.026% | 30.370% | 2228 |
| X_hat CAL-FDR1 | 151 | 0 | 3224 | 0% | 4.474% | 3102 |

These are identity-by-dataset counts across nested K and repeated spatial mappings, not 3375 independent molecules or biological replicates. Both rho rules retain all 3253 solver true positives. The 122 remaining FN are solver misses. Consequently, enrichment of rho-filtered true identities is NOT_ESTIMABLE at these frozen V57 thresholds: there are no such events. Do not conflate solver misses with filter loss or claim a protective biological mechanism from an empty loss group. Geometry strata and per-identity recurrence are descriptive outputs for auditing and later comparisons.

X_hat thresholds were selected on CAL using the existing rule; these are comparable calibration targets, not matched realized HOLD FDR. The zero-FP X_hat FDR1 result has only 151 retained true contexts and very low recall; it is not evidence of superior useful recovery.

Validation: scientific design fingerprint recomputes exactly; all 60 expected dataset IDs present; molecular keys unique; all retention flags reproduce from frozen thresholds; CAL TP/FP counts match the frozen entries. Local CAL JSON source reports match after restoring LF line endings. Thirty per-case reported_identity_records.csv files are absent locally; a separate bounded remote check verified all 60 CAL source-file hashes. The molecular aggregate is byte-identical locally/remotely and the threshold file matches after LF normalization. See audit.json and remote_provenance_check.json. This does not certify all solver/checkpoint provenance or biological FDR guarantees.

V58/V59 results remain outside this analysis. No curves were reselected on HOLD. No independent-row significance test was used.
