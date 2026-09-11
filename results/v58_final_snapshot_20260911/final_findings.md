# Final V58 findings

All60 datasets and the complete30 HOLD datasets are present. Original global per-severity CAL thresholds were independently recomputed from all15 CAL cases and matched without rewriting. Original final aggregate agrees exactly with the audited metrics; per-K/per-replicate counts and all60 candidate reporting gates pass. Oracle/sentinel provenance, runtime bindings, arrays, training artifacts and finite completion checks pass. Scientific outcome weakness is not a process failure.

MILD: raw TP1680/FP1514/FN70 (96% recall). CLEAN-fixed rho threshold: FDR37.16%, recall80.69%. Local CAL rho FDR5: HOLD TP370/FP15, FDR3.896%, recall21.14%, filter-induced true loss1310. Local FDR1: FDR1.887%, recall14.86%, exceeding its nominal target.

MODERATE: raw TP1674/FP1473/FN76 (95.66% recall). CLEAN-fixed rho: FDR38.20%, recall74.06%. Local CAL rho FDR5: HOLD TP270/FP36, FDR11.765%, recall15.43%, filter-induced true loss1404. Local FDR1: FDR3.333%, recall8.29%. Both nominal rho risk targets fail on complete MODERATE HOLD.

Abundance local FDR5 baseline: MILD FDR5.282%/recall29.71%; MODERATE FDR3.561%/recall18.57%. Thus neither universal CLEAN-threshold transfer nor simultaneous high-recall/low-FDR performance is established. Report empirical risk and recall together. Nested K and repeated mappings share identities and are not independent biological replicates. No thresholds or perturbations were changed in response to these outcomes.

Cleanup proposal:50 remaining ordinary intermediate/redundant models from ten completed MODERATE HOLD datasets,2004697600 bytes (1.867GiB), all on data disk, no symlinks. Preserve final epoch3000 checkpoints, arrays, reports, candidate/molecular records, training histories/diagnostics, CAL seals and all oracle/sentinel/adoption/parent evidence. No V57/V59/missing-library assets are removed. Previous230-file retirement receipt remains in results/v58_completed_snapshot_20260911; historical storage manifests do not override these explicit receipts.

This snapshot must be pushed successfully before removal. Consult deletion_receipt.json for execution status. Original final report status PENDING_REVIEW is preserved as source evidence; local_review.json records the completed independent review. One audit CLI preflight failed before checks began; its log and script remain under /root/v58_jobs/final_review_20260911. The successful read-only retry uses final_review_20260911_verified.

Cleanup COMPLETED after successful snapshot push0997160. Exact50 planned files retired,2004697600 bytes freed; data free5675184128 bytes at receipt. All retained source/model/array checks and original post-retirement oracle/adopted-sentinel/CAL/result verification PASS for60 cases. See deletion_receipt.json and post_retirement_verification.json. Full scientific records and final models/arrays remain; no training/rho rerun.
