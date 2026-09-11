\# LD Project — Codex Working Rules



This repository is the lightweight working repository for the LD lipid deconvolution project.



Default rule:



REUSE EXISTING CODE AND RESULTS FIRST.

DO NOT REBUILD EXISTING PIPELINES UNLESS A REQUIRED COMPONENT IS PROVEN MISSING.



\## Execution rules



1\. Always read:

&#x20;  - AGENTS.md

&#x20;  - docs/CURRENT\_STATE.md

&#x20;  - docs/NEXT\_TASK.md



2\. Use only the files explicitly required by CURRENT\_STATE / NEXT\_TASK.



3\. Do not reconstruct old v17-v48 project history.



4\. Do not scan the archived heavy repository unless NEXT\_TASK explicitly requires it.



5\. For simple statistics:

&#x20;  existing CSV/Parquet/JSON

&#x20;  → pandas/scipy

&#x20;  → requested outputs

&#x20;  → STOP.



6\. Do not rerun simulation, ISTA, profile, certificate, or raw MSI processing for ordinary downstream analysis.



7\. Frozen definitions must not be changed because results are weak:

&#x20;  - production ISTA

&#x20;  - rho\_zero

&#x20;  - necessity

&#x20;  - profile L/U

&#x20;  - rho=1e-3

&#x20;  - d\_frag=0.02 competition groups

&#x20;  - observation threshold



8\. Scientific priority:

&#x20;  1. identity correctness / false allocation

&#x20;  2. group-level rescue

&#x20;  3. quantitative abundance accuracy



9\. Never use:

&#x20;  bad result → tune parameters → rerun.



10\. When NEXT\_TASK is complete:

&#x20;   - write only requested outputs

&#x20;   - update CURRENT\_STATE.md

&#x20;   - mark NEXT\_TASK completed

&#x20;   - commit and push

&#x20;   - STOP.


## Per-case result retention and cleanup — user instruction 2026-09-11

For subsequent experiment execution, handle each completed dataset in this order:

1. Verify normal completion, finite outputs, design/runtime binding, exact result membership and artifact hashes. Use cached results only; do not rerun training or rho for review.
2. Preserve candidate/molecular records and independently audit TP/FP/FN, solver misses versus filter-induced losses, recall and retention where allowed by the frozen calibration contract. Before the required CAL seals exist, restrict review to process/provenance and permitted CAL results; defer HOLD outcome analysis. Never tune cutoffs using HOLD or relabel partial results as final.
3. Download compact outputs with matching hashes, record the per-case review and missing/deferred checks, then commit and successfully push to the authorized experiment-record branch. Verify the pushed commit includes this case. A previous batch commit is not evidence that a new case was saved.
4. Only after the above, retire explicitly listed intermediate/redundant checkpoints whose removal cannot break final aggregation, calibration, provenance verification, checkpoint comparisons, resumption dependencies or downstream experiments. If dependencies are unclear, retain the files. Do not reuse the V58 retirement list blindly for V59 or the missing-library runner.
5. Preserve final learned arrays, the final usable model (including valid production early-stopped runs), all candidate/molecular result records, reports, training history/diagnostics, design/runtime bindings, source and threshold hashes/seals, frozen calibration inputs and all sentinel/adoption/parent-control evidence. Hashes are not backups of model weights. Preserve any intermediate checkpoint still required by a verifier or planned analysis.
6. Before deletion, verify exact paths and symlink targets, per-file hashes, completion and exclusion of active cases; save the plan and the pushed snapshot commit. After deletion, verify preserved dependencies again and commit/push a deletion receipt listing removed paths, hashes, reason and freed bytes.

This is continuing authorization for narrowly audited per-case retirement, not blanket deletion of completed result directories. Failed audit/download/push/dependency checks prohibit deletion. No recursive sweep of historical assets. Large retained arrays/checkpoints stay in documented storage; do not put them indiscriminately into lightweight Git. Existing remote supervisors do not acquire automatic review/Git/deletion capability merely because this rule is recorded.
