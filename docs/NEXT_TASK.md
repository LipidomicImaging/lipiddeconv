# Active next task — 2026-09-11 handoff

COMPLETED: six-case NNLS result retrieval, independent provenance/CAL/HOLD audit and scientific analysis. V57 stratified review also complete.

RUNNING: V58 storage recovery (37/60 at01:49UTC), with frozen missing-library six-fit handoff automatically queued in remote supervisor /root/v58_jobs/storage_recovery_20260911/status.json. Do NOT start a duplicate missing-library job. User expanded data disk20GB, not yet visible in guest snapshot; supervisor waits if capacity becomes insufficient. Preserve original failure and all archives.

RUNNING: V59 formal serial CPU rho, three D0 CAL reports complete; not18/18. Next review V58/18-case V59 and six missing-library outputs as each actually completes; retrieve compact records with source hashes and commit/push. Use docs/ACTIVE_EXPERIMENTS.md for paths. Full task remains IN_PROGRESS; NNLS review is not completion of either GPU benchmark. Earlier instructions below describe prior states.

# Active next task — V58/V59 result handoff

Scheduled continuation authorized: first2026-09-11 02:00 Asia/Shanghai, then every30min, see docs/EXPERIMENT_MONITOR_HANDOFF.md. V57 stratified miss review complete. Next scheduled run checks NNLS numerical outputs and GPU experiment completion, launches frozen missing-library only on a genuinely released GPU, then reviews complete outputs and pushes Git. V59 GPU idle during serial rho is not a completion signal.

The user's two requested missing-library preparation items are COMPLETE: isolated execution adapter and CPU checks/frozen six-case manifest. Next run six omission fits on a released GPU using results/missing_library_challenge_k125/EXECUTION.md, without changing the frozen design. No omission GPU run or automatic queue has been started. Continue V58/V59/NNLS result handoffs independently.

User authorized CPU NNLS baseline. Runner and six-case protocol: analysis/run_nnls_solver_baseline.py and docs/NNLS_BASELINE_PROTOCOL.md. Four-worker timing passed (~32min/case extrapolation excluding rho); run independently of GPU jobs, then review both raw and filtered outcomes. Do not treat the old foreground-mean oracle as the new pixelwise baseline. Missing-library adapter remains a separate unfinished priority.

Control reconstruction/checkpoint audit now PASS for all three references. Next implement and validate isolated reduced-N context/metadata/output mapping, preserving original B and full truth denominators. Production channel weights are library-dependent: retain the formula and record resulting weights rather than silently fixing full-library weights. Then freeze the execution manifest; no new training before these checks.

Missing-library selection and initial three-control artifact audit now saved under results/computational_closure/missing_library. Next reconstruct the three parent B/X_true cases and verify runtime B hashes plus production configuration/checkpoint lineage before implementing/freezing the isolated reduced-N adapter. Do not relabel selection preparation as execution freeze. Git push is now explicitly authorized to the user's repository; preserve progress on codex/v58-v59-run-records.

Closure progress: V57 snapshot accounting and bounded remote CAL provenance check completed; see results/computational_closure/v57/analysis_record.md. Next resolve exact parent solver artifacts and produce the missing-library selection manifest under docs/MISSING_LIBRARY_CHALLENGE_PROTOCOL.md before any new training. Do not attempt rho true-loss enrichment on V57's empty filtered-truth group.

User-authorized closure preparation: see [COMPUTATIONAL_CLOSURE_PLAN.md](COMPUTATIONAL_CLOSURE_PLAN.md). Prepare the small missing-library challenge and audit existing V57 analysis inputs while V58/V59 continue; no new GPU job has been queued. The final result handoff below remains pending.

Status: REMOTE_EXPERIMENTS_RUNNING; FINAL_RESULTS_NOT_YET_COMPLETE.

Read [ACTIVE_EXPERIMENTS.md](ACTIVE_EXPERIMENTS.md) for exact scientific definitions, hosts, status/output paths, frozen fingerprints and recovery evidence. When asked for status, read those bounded status/log paths; do not rerun or redesign the experiments. After completion, verify the planned60/18 result sets, retrieve compact results and provenance, analyze them, then commit and push the results/analysis records as requested by the user. Do not mark this task complete from a progress snapshot. Do not stage unrelated V52 or historical V48 changes.

The earlier V48/V49 task text below is retained as historical context, not the active execution task.

\# NEXT TASK — v48 All-Reported-Candidate Identity Validation



Status: V49A_RHO_ZERO_PARITY_READY

The frozen historical rho_zero implementation has been recovered into
`src/rho_zero.py`. Numerical parity has NOT yet been executed.

First, manually run on the remote GPU host:

```text
python analysis/run_v49a_identity_calibration.py \
  --asset-root /root/autodl-tmp/decon-lipid \
  --output-dir results/v49a_identity_runtime \
  --device cuda \
  --validate-rho-zero
```

Only after parity reports PASS, manually run the fixed four-case smoke test:

```text
python analysis/run_v49a_identity_calibration.py \
  --asset-root /root/autodl-tmp/decon-lipid \
  --output-dir results/v49a_identity_runtime \
  --device cuda \
  --smoke
```

After human inspection of smoke output, rerun without `--smoke` for the full
192 cases. Do not auto-start the full run after smoke.



\## Goal



Using existing cached v48 results only, determine whether rho\_zero can identify a high-confidence subset among all candidates that would actually be reported.



\## Input



Primary cached table:



v48\_pilot\_lipid\_observations.csv



Expected:



48 × 391 = 18,768 rows



Use cached fields only.



\## Do not



Do NOT:



\- rerun simulation

\- regenerate X\_true

\- regenerate B\_sim

\- rerun production ISTA

\- rerun NNLS

\- rerun profile

\- rerun rho\_zero

\- rerun necessity

\- rerun cone

\- recompute groups

\- add features

\- train a model

\- optimize thresholds

\- start v49



\## Reporting gates



Use exactly:



\- X\_hat > 1e-4

\- X\_hat > 1e-3

\- X\_hat > 1e-2



For each gate:



truth\_identity = 1 if X\_true > 0

truth\_identity = 0 if X\_true == 0



\## Baseline statistics



For each gate report:



\- n\_reported

\- n\_true\_reported

\- n\_false\_reported

\- identity\_precision

\- false\_identity\_fraction

\- mass\_weighted\_false\_fraction

\- true\_identity\_recall



\## Primary certificate



Feature:



rho\_zero



Direction:



higher = more trustworthy



For each reporting gate, sort reported candidates by rho\_zero descending.



Evaluate fixed coverage:



\- 5%

\- 10%

\- 20%

\- 40%

\- 60%

\- 80%

\- 100%



At each coverage report:



\- n retained

\- identity precision

\- false identity fraction

\- mass-weighted false fraction

\- true identity recall



Do not select a best coverage.



\## Stratified analysis



For rho\_zero top10% and top20% repeat within:



K:

13 / 25 / 55 / 103



residual:

0 / 0.1



selection:

random / hard\_competition



Report:



\- n

\- identity precision

\- false identity fraction

\- mass-weighted false fraction

\- true identity recall



\## Secondary comparison



Pooled AUROC only for:



\- necessity\_signal

\- profile\_relative\_width

\- global\_fragment\_cone\_residual



Fixed directions:



\- necessity\_signal: higher trustworthy

\- profile\_relative\_width: lower trustworthy

\- global cone: higher trustworthy



Do not combine features.



\## Outputs



Generate only:



\- v48\_identity\_allreported\_summary.csv

\- v48\_identity\_allreported\_riskcoverage.csv

\- v48\_identity\_allreported\_feature\_summary.csv

\- v48\_identity\_allreported\_report.html

\- v48\_identity\_allreported\_report.json



\## Efficiency



Expected workflow:



read cached table

→ filter

→ groupby/statistics

→ output

→ STOP



No optimization.

No GPU.

No raw MSI.

No historical project scan.



\## Completion



When complete:



1\. update CURRENT\_STATE.md

2\. mark this task COMPLETED

3\. commit and push

4\. STOP



Do not start v49.

