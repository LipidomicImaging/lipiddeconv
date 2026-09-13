# User amendment: actually refit the high-recall physical-prediction pool

2026-09-13. User steering: “新方案 第一次解卷积如果FDR=5 recall是多少 而且没关系如果 FDP=6.3 但是recall=95.2 第二次筛选后再解卷积呢”. This explicitly redirects the current task to evaluate the screened high-recall pool after a second deconvolution. Preserve the original execution contract, model, thresholds and results; this is a separately recorded downstream amendment, not an outcome-driven hidden edit to that contract.

## Questions and separation of evidence

1. Answer the user's 5% question using cached DEV TRAIN scores only. Report the maximum TP under observed FDP<=5%, complete score ties, then minimum FP and highest cutoff. Preserve all125 truths in recall. Record the selected names, score/model hashes and exact count arithmetic. This is retrospective development description, not an adopted new threshold, fresh validation or population FDR control.
2. On the current exposed CHECK, apply the already frozen high-recall pool threshold 0.5735000428231877 and unchanged model. After all34 physical blocks complete and pass cached reviews, compute/save its actual membership and counts. The reported DEV119/8 is not assumed to equal CHECK membership or counts.
3. Actually refit this CHECK pool on all original15837 foreground pixels, allowing every pool member to be reported under the unchanged candidate mean>0.001 / molecular any-alias gate. The original strict final-score threshold must not prevent this separate pool-only experiment from running or restrict its primary endpoint.

The user is asking whether the second deconvolution removes false identities while preserving true identities. No benefit is assumed: equal before/after membership is an admissible negative result. Coefficient increases after shrinking the library are not themselves evidence of identity correctness.

## Fixed inputs and implementation

Parent results are results/physical_block_prediction_v1, remote /root/physical_block_prediction_v1/results. Reuse the original exposed CHECK source /root/identity_confidence_joint_validation/results/cases/CAL1/prepared_arrays.npz, frozen model/threshold/selection, original391 candidate order and377 molecular names. Use all aliases of every selected name, unchanged A_solver/B/mask and original NNLS/KKT helper analysis/refit_screened_nnls.py. Four CPU workers, BLAS1, block size250, maxiter3910, float64 solves/float32 final storage; no new solver, rho, profile, simulation, observation, GPU or classifier fit.

New orchestration only: analysis/run_physical_block_pool_refit.py. It must bind its code, this amendment, the existing helpers, parent source/design/model/selection seals and exact candidate membership. Original physical-prediction code/protocol stays byte-unchanged. New results are only results/physical_block_prediction_v1/pool_refit_user_amendment. Save all391 candidate and377 molecular records, raw/pool/refit reported flags, source/design/threshold hashes, block/final arrays, actual runtime and completion/review records. Preserve all model and array files locally and remotely; no cleanup.

Only one actual CHECK pool refit is requested. A completed exactly matching refit cache may be reused; do not call the old final_decision path in addition and duplicate solving. No candidate reentry, additional confidence classifier or new post-refit rho is part of this bounded amendment. Consequently pool screening losses remain explicit; this experiment can show whether refitting helps the retained pool, but cannot recover a name absent from it.

## Required saving order

Complete the existing CHECK feature run without interruption or restart. Independently review the full cached case, original source/runtime/code bindings, exact34-block membership and all arrays. Download with matching hashes. Freeze the unchanged CHECK selection, replay its probabilities and counts, then commit/push this specific CHECK before acknowledging its handoff.

Prepare the new amendment input/selection/code seal only from the saved CHECK. Review the exact pool membership, all-alias mapping and input binding without solving. Commit/push this exact new seal and verify its pushed commit before the one refit. Refit accepts the externally verified full snapshot commit and records it. The 4-hour per-case budget applies; numerical failure or budget overrun preserves partials and stops this version without parameter changes.

After completion, independently replay cached array/block membership, hashes, finite/nonnegative values, original mean/alias gate, reconstruction and TP/FP/FN accounting. Review does not rerun optimization, model fitting or rho. Download all new final/block arrays and compact results, verify hashes, write the requested report, update CURRENT_STATE/NEXT_TASK, commit/push the specific final case and stop. Retain any technical discrepancy and its separately documented compatibility review rather than relabeling an exact-check failure as PASS.

## Endpoint and honest comparison

Primary endpoint is the actual pool-only post-refit reported set, with full125-truth recall and empirical FDP. Report raw first NNLS, screened pool and second NNLS side by side; enumerate screening-induced true losses, any additional refit true losses and refit-removed false identities. Report strict1%/80% and secondary5%/80% pass flags without changing either target or denominator.

Also retain the old strict final-score eligibility and its intersection with the new refit as a secondary descriptive result. That secondary set cannot overrule the primary pool-only experiment's completion. Do not claim it was the original contract's successful execution or change the old model/threshold seal. Do not apply any further post-refit cutoff to make outcomes pass.

Compare previous spatial-donor119/8 and120/8 CHECK workflows only as historical observed results, acknowledging their different screening rules. Neither this exposed CHECK nor the DEV5% query is independent FDR validation, and neither tests new relative-abundance families. A positive result motivates future prospective validation; a negative result is preserved without another automatic variant.
