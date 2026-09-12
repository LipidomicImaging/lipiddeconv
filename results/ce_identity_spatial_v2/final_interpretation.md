# V2 final decision: method NO-GO; paired spatial rescue not achieved

The six-case pilot is complete. Both CAL thresholds were sealed before EVAL. Independent review passes5689 output hashes,5572 full/deletion proofs, both CAL-only threshold reconstructions and all status/accounting checks. The largest checked proof gap is9.327438015766833e-7, within the unchanged1e-6 contract. All5692 final transfer files match their hashes locally; local reclassification and accounting reproduce all12 method-case outputs. This is a scientific NO-GO with valid process/evidence, not a crashed or numerically unresolved experiment.

## 1. Method decision

METHOD_NO_GO. Local EVAL has16 TP/0 FP, empirical FDP0%, TP retention16/350=4.57%, all-truth/reportable recall16/375=4.27%, and16 distinct retained identities. Aggregate utility and all three per-challenge40% retention requirements fail; close-neighbor also has an empty retained set. The1% risk/40% utility gate and60% strong-success definition are unchanged. Zero false selections among sixteen retained observations does not establish a population FDR guarantee.

| EVAL challenge | Raw solver TP | Global TP / FP | Global TP retention | Local TP / FP | Local TP retention | Local all-truth recall |
|---|---:|---:|---:|---:|---:|---:|
| MILD | 123 | 24 / 0 | 19.51% | 15 / 0 | 12.20% | 12.00% |
| Close-neighbor missing | 115 | 0 / 0 (empty) | 0% | 0 / 0 (empty) | 0% | 0% |
| Relatively-isolated missing | 112 | 0 / 0 (empty) | 0% | 1 / 0 | 0.89% | 0.80% |
| Aggregate | 350 | 24 / 0 | 6.86% | 16 / 0 | 4.57% | 4.27% |

Empty sets are displayed as empty, not successful risk control, even though the immutable machine-readable accounting uses FDP0 by its empty-set convention. Both nonempty aggregate sets have observed FDP0. Global aggregate all-truth recall is24/375=6.40%, with24 distinct retained identities.

Raw solver accounting is350 TP/334 FP/25 FN. Global screening adds326 true losses, giving351 final FN. Local screening adds334 true losses, giving359 final FN. All nonselections remain in the original denominators. The large loss occurs after the solver has reported most truths; it is not explained by raw solver FN alone.

## 2. Spatial contribution

The predeclared paired category is BOTH_NO_GO. Spatial conditioning did not restore a subset meeting the frozen objective. Local retains eight fewer true identity contexts overall than global: nine fewer in MILD, no change in close-neighbor missing, and one additional identity in relatively-isolated missing. Aggregate retention is2.29 percentage points lower. This does not establish that every possible spatial method is ineffective.

The missing-library test is decisive for this version. Close-neighbor remains0/115; relatively-isolated improves from the matched global0/112 to1/112. One recovered identity is nonzero recovery but is far below the40% requirement. V1's historical zero retention remains context only: V2 uses new cases, so spatial attribution comes from this same-batch control rather than V1-to-V2 outcome comparison.

Local EVAL status counts are16 RETAINED,577 REPLACEABLE and91 FULL_MODEL_INCOMPATIBLE; all91 incompatibilities occur among the230 reported MILD identity-local observations. Close-neighbor has227 REPLACEABLE; relatively-isolated has226 REPLACEABLE and one RETAINED. These are identity-conditioned decisions, not91 independent experiments. Global has24 RETAINED,659 REPLACEABLE and one THRESHOLD_UNRESOLVED. Neither method has a NUMERICALLY_UNRESOLVED EVAL selection status. Report these facts without changing U or epsilon in response.

Frozen epsilon: global0.02004338299709658; local0.020362849583791913; gamma1e-6. Their CAL results, source hashes and full curves are retained. The independent reviewer recomputed each threshold from CAL without EVAL.

## Scope and closure

Close this V2 version without new cases, threshold/U/gamma/weight/denominator patches or an automatic next-stage experiment. The method gate does not authorize progression into formal physical-mismatch validation. The broader high-confidence identity objective remains open; this result does not prove impossibility.

The existing synthetic MILD library and CLEAN omission challenges are not the proposed real physical mismatch model. Systematic fragment-intensity changes, strong-peak preservation, empirically supported weak-peak censoring and separately estimated pixel variation remain future validation boundaries. CE30/35/40 is not a measurement of pixel noise. None of those mechanisms was inserted into this V2.

All candidate/molecular records, per-identity statuses, calibration seals, novelty records, proofs, summaries and source bindings are preserved. The2793 numerical NPZ files (275593082 bytes) exist in both the local result directory and the remote source directory with verified hashes and exact locations in final_transfer_review.json. The six evidence arrays and2786 proof files remain outside lightweight Git; the small previously tracked uncertainty component_fractions.npz remains in Git. Source checkpoint/model bytes remain remotely accessible through the documented retained store. The raw source findings.md and independent_review.json remain byte-exact; this interpretation is a separate final report.

The redundant final transport archive may be deleted only after the exact final result snapshot and final_archive_retirement_plan.json are pushed, followed by complete source verification and a deletion receipt. No model, array, proof, calibration source or unique result is a retirement target. Both V2 processes have exited; westc GPU was0%/0MiB at04:25UTC2026-09-12. No new experiment is queued.
