# MILD NNLS CAL pilot RUNNING — 2026-09-11

User authorized one paired CAL_R1_K125 MILD pixelwise NNLS run, reusing existing CLEAN NNLS and both original ISTA records. Protocol: docs/MILD_NNLS_CAL_PILOT.md. Running on westc at /root/v58_jobs/mild_nnls_cal_pilot with four nice10 CPU workers; exact frozen input audit and CLEAN cached file checks pass, first foreground pixels completed. No GPU or HOLD outcomes. Next finish run, download/hash/review raw TP/FP/FN plus original rho/abundance CAL curves, preserve arrays remotely and compact evidence in Git. Do not retune or automatically expand. Earlier channel-prediction route remains stopped, six-fit alternative-reference GPU proposal paused.

# CAL channel-prediction pilot COMPLETED — 2026-09-11

Bounded CAL_R1_K125 CLEAN/MILD CPU pilot and review complete in results/cal_channel_prediction_pilot. 456 reported identity contexts, 1368 deletion fits, 585 seconds, no GPU/HOLD outcomes. Original frozen A/B/X_truth bindings, output/source/script hashes, finite losses, membership and fold tests pass. At retrospective empirical5% FDR on MILD: rho32 TP/1 FP (25.6% all-truth recall), X_hat45/2 (36%), predictive gain no nonempty admissible set (0% recall); predictive AUROC0.458. Twenty raw true identities lack prescribed fragment support; the common-evaluable comparison also fails. CLEAN rho retains123 truths versus prediction103 after support exclusion.

Decision: STOP this standalone channel-prediction route; no outcome-driven retuning. The previous six-fit alternative-reference GPU pilot remains PAUSED, original protocol preserved. This selected CAL screen does not establish independent FDR control or impossibility of other strategies. See analysis_record.md for separate solver/support/score misses and limitations.

Next active work is existing V58/V59 completion and frozen missing-library handoff, with per-case provenance/download/Git before any safe checkpoint retirement. At05:52:55UTC V58 still57/60, active MODERATE__HOLD_R5_K050, data free4.26GiB. Do not duplicate queued jobs; storage guard remains active. No deletion during this pilot. Earlier task status paragraphs below are historical.

# CAL channel-prediction pilot RUNNING — 2026-09-11

User-directed bounded CPU pilot is now executing on V58 host, PID67733, /root/v58_jobs/cal_channel_prediction_pilot. Only CAL_R1_K125 CLEAN/MILD, original reported identities, three blocked/buffered channel folds. Scoring uses database A and observed B; no target spectra or truth enters fitting. Review protocol: docs/CAL_CHANNEL_PREDICTION_PILOT.md. Existing six-fit alternative-reference GPU pilot is PAUSED, superseding older next-action notes below; its original frozen artifact is preserved.

Next: finish this CPU run, verify/download outputs, compare rho/X_hat/predictive score with all-truth denominators and support exclusions, record a stop/continue decision and push. No threshold/fold tuning after results. Existing production jobs retain priority. At05:46UTC V58 completed57/60, active MODERATE__HOLD_R5_K050; final aggregate/missing-library execution remain pending. No cleanup in this pilot.

# CAL numerical/mechanism diagnostic completed — 2026-09-11

COMPLETED:8 prespecified CAL identities,48 CPU deletion comparisons using original/reversed NNLS and independent BVLS. results/cal_rho_numerical_diagnostic includes raw channel decomposition, source hashes, failures and review. All16 original-order stored scores reproduce exactly. Two near-zero true-score cases cross the original1.27e-20 cutoff across methods; three substantial false MILD scores (~8.5e-6 to2.3e-5) are stable to~1.8e-18. Thus numerical sensitivity exists but does not explain all high false scores. Channel-overlap evidence is not false-specific and does not prove mismatch absorption. No HOLD outcomes parsed, no GPU training, no original runner/threshold edits. Thread-reduction roundoff and tiny BVLS feasibility projection are explicitly documented; all scientific arrays match frozen hashes.

Next: bounded alternative-reference CPU implementation/parity/zero-coefficient deletion/cost tests under docs/MISMATCH_CONFIDENCE_PILOT.md, followed by actual asset freeze if feasible. Existing V58/V59/missing-library jobs keep priority. Do not substitute this selected CAL diagnostic for independent performance validation or infer population numerical-error frequency.

# Paired CLEAN/MILD review completed — 2026-09-11

User-requested matched identity/K/replicate review COMPLETE in results/v58_clean_mild_paired_review. Correction: true rho does not generally decline;921/1650 jointly reported true contexts increase,729 decrease. Raw TP1676 ->1680; original cutoff FP60 ->835. Of1676 CLEAN-retained truths,26 become solver misses,260 fail the original cutoff (rho becomes zero),1020 fail only the higher MILD CAL cutoff,370 remain. This is sequential accounting, not a causal interaction test. Original threshold1.27e-20 versus MILD1.75e-5; near-zero score ratios are not meaningful biological effect sizes. Parent CSV exact-hash and threshold JSON LF-normalized checks recorded. No frozen method/threshold/pilot change, new GPU execution or cleanup. Next remains bounded CPU implementation/numerical checks under the existing pilot protocol; current benchmark completion has priority.

# Mismatch-confidence diagnosis — 2026-09-11

COMPLETED: cached V58 ranking and true-loss diagnosis, results/v58_mismatch_ranking_diagnosis. MILD rho AUROC0.7635; retrospective global-threshold max recall at empirical5% FDR28.86%, abundance31.54%. This is HOLD-label-dependent diagnosis, not new calibration. Thus threshold migration alone does not explain poor recall. Identity-level descriptive losses align with lower cone isolation/higher nearest-fragment cosine, not larger median perturbation; no causal/enrichment claim. All412 source hashes pass; tied-score ranking checks pass.

COMPLETED: bounded research protocol frozen in docs/MISMATCH_CONFIDENCE_PILOT.md (hash in pilot_protocol_freeze.json). Six future K125 fits, three scoring arms, new phase-specific perturbation draws; alternative-reference individual identity score is a hypothesis. NEXT: implement CPU-only score/parity/leakage/numerical/cost checks, then freeze actual assets before any new GPU work. Existing V58/V59/missing-library jobs retain priority. No new GPU run or modification of existing rho/training/thresholds. Previously exposed HOLD identities are not an unseen-identity test; new pilot only tests new perturbation draws conditional on this pool.

# Standing per-case handoff rule — 2026-09-11

User now authorizes future per-case review -> compact result download/hash verification -> successful Git push -> dependency-safe intermediate checkpoint retirement -> verification and pushed receipt. Apply AGENTS.md "Per-case result retention and cleanup". This supersedes the earlier single-plan-only authorization, while keeping final arrays/models, all result/calibration/provenance records and verifier/parent/sentinel dependencies protected. HOLD outcome review must wait for the required CAL seals. Audit and push failures mean retain files. The currently running remote supervisor has not been changed to perform automatic per-case review/Git/deletion; this is the required workflow for subsequent execution and handoffs.

# V58 storage cleanup completed — 2026-09-11

Completed requested50-case analysis/Git/cleanup handoff. Snapshot pushed as4f6eda6308698eb4949d0b0ec052b5cc606e4c49 BEFORE deletion.230 intermediate/redundant checkpoints retired with per-file hashes;8.588GiB freed (root5.974GiB, data2.614GiB). Final epoch3000 models, learned arrays, all result/provenance tables and all sentinel-related cases remain. Remote deletion receipt and all retained source hashes verified; see results/v58_completed_snapshot_20260911/deletion_receipt.json. At04:32UTC V58 still50/60, active MODERATE__HOLD_R2_K175, no restart; data free6.2GiB/root8.1GiB. Final60-case aggregation and missing-library handoff remain pending. Keep existing storage guards; no further files authorized for automatic deletion beyond this exact completed plan.

# V58 storage handoff — 2026-09-11

Completed:50-case snapshot download, source-hash validation and independent interim HOLD accounting. Next push snapshot, execute exactly cleanup_plan.json for ordinary completed-case intermediate checkpoints, verify retained source hashes, and commit deletion receipt. Continue existing V58 supervisor; do not delete final models, learned arrays, parent controls, sentinel evidence or active cases. Final60-case review remains pending.

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

