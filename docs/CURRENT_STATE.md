# rho mismatch explanation and matched CE asset audit COMPLETED — 2026-09-11

User-confirmed CE30/35/40 data located through the known production report's direct target_csv reference. results/rho_mismatch_mechanism_ce_audit contains the source hashes, exact identity/adduct/rule keys and record manifest:389 selected DDA annotated rows,133 identities at >=2 energies,66 at all3; pair counts30/35=107,30/40=81,35/40=77. These rows already trained the final production CE adapter. They are developmental evidence, not independent final validation or verified pure standard spectra; same-CE repeats and full upstream independence remain unverified. This supersedes the earlier limited inventory's uncertainty about asset availability, while preserving that historical record.

Mechanism clarified from frozen rho: under ideal CLEAN b=A*x_true, deleting an absent identity leaves the exact truth fit feasible, forcing false rho=0; mismatch removes this guarantee. Large false scores are numerically reproducible; true rho does not generally decline. Cached same-domain CAL mean-spectrum change5.677% exceeds two selected clean deletion margins0.02766%/0.01654%, but direction and individual causal attribution remain unresolved. No fit/rho/training, threshold change, broad archive search or deletion.

Broader confidence goal OPEN. Next bounded prerequisite: audit channel alignment, ambiguous/shared peak assignments and direct raw-source provenance for these133 matched identities; inspect existing OOF lineage before freezing an evidence-based variation model and independent validation. No further tuning of the stopped CAL screens or automatic GPU expansion. Existing V59 and missing-library job audits remain separate pending work; no fresh remote status claimed this turn.

# Two bounded confidence-method development screens COMPLETED — 2026-09-11

No validated solution has yet achieved the requested low-FDR/high-recall goal. Finite3-reference identity-deletion test: MILD CAL recall20.8% at empirical5% FDR versus rho22.4%/abundance37.6%, despite77.32% SSE reduction; stop that version. Grouped joint rho/abundance model: identity-disjoint internal CAL evaluation gives MILD FDR6.27%/recall30.74%, MODERATE FDR7.05%/recall29.37%, no gain over abundance and fails5% target; stop that version. Results in results/reference_flexibility_cal_pilot and results/joint_confidence_cal_pilot. Both code/protocols were pushed before outcomes; all source/model/seal/accounting checks pass. No original method/threshold edits, no GPU method expansion, no deletions.

The broader scientific objective remains OPEN. Do not keep modifying features/reference counts on these exposed CAL outcomes until something passes. Next prerequisite is evidence-based spectral-variation calibration and a new independent validation design. Existing full-library provenance identifies CE29 predicted spectra, not verified matched experimental multi-condition spectra; see upstream_uncertainty_inventory.json. Do not infer missing assets globally or scan archives: only known paths/direct references may be audited after a scoped task. The six-GPU alternative-reference protocol remains paused.

Existing missing-library GPU job was verified RUNNING/MISSING_LIBRARY at06:51UTC; V59 unchanged. Their audits remain separate pending tasks.

# Reference-flexibility screen completed; joint-evidence CAL screen next — 2026-09-11

Finite3-reference/full1173-column individual-deletion CAL screen completed in results/reference_flexibility_cal_pilot. At5% empirical CAL FDR new recall20.8%, original rho22.4%, abundance37.6%; full SSE fell77.32% without identification gain. Stop this version; no parameter/variant tuning, no GPU expansion. Full dictionary, scores, checks and analysis retained.

User's broader confidence-method request continues with one fixed cached-data joint rho/abundance classifier screen under docs/JOINT_CONFIDENCE_CAL_PILOT.md. Five identity groups separate TRAIN/CALIBRATION/EVALUATION within each rotation; no geometry/identity/severity features, no original HOLD records, no new solver/rho/GPU. Fixed logistic quadratic model, no hyperparameter search; criteria frozen before execution. Existing missing-library GPU job is running independently.

# Finite reference-flexibility CPU development RUNNING — 2026-09-11

User requested an implemented response to mismatch. Bounded new CAL-only test under docs/REFERENCE_FLEXIBILITY_CAL_PILOT.md: original plus2 independently generated reference variants per candidate; full1173-column competition and deletion of every same-identity variant. No inter-identity group rescue or A_target in scoring. Uses MILD CAL_R1_K125 NNLS reporting universe211/125truths. Original production pipelines remain frozen. Six-GPU prior protocol stays paused. New score must beat both same-universe rho/X_hat by10pp recall at empirical5% FDR before further validation; do not tune after outcome.

Remote /root/v58_jobs/reference_flexibility_cal_pilot running after input/parity/tiny/deletion checks; next complete cost/numerical gates, download and independently review, commit/push. Scoring uses a known synthetic perturbation family, not measured real uncertainty. Sources motivating explicit library adjustment are linked in protocol; this finite dictionary is not a DANSER/PLMM reproduction.

At06:51UTC the existing frozen missing-library supervisor is RUNNING/MISSING_LIBRARY after V58 cleanup restored space. No duplicate GPU job. No further deletion authorized by this new scoring experiment.

# Final V58 audit and scoped retirement COMPLETED — 2026-09-11

Final60-case snapshot independently audited and successfully pushed as09971606550e904e0dca8aec75bf3fec5a79a84b BEFORE deletion. Exactly50 remaining ordinary intermediate/redundant checkpoints retired from ten completed MODERATE HOLD cases;2004697600 bytes (1.867GiB) freed on data disk. Receipt data free5675184128 bytes (~5.29GiB). Final models/arrays, all candidate/molecular records, CAL inputs/seals, reports/history/diagnostics and all oracle/sentinel/adoption/parent evidence remain. No V57/V59/missing-library files removed. All814 source hashes and retained model/array checks pass; original oracle/sentinel/CAL/result verification passes again after deletion. Receipts under results/v58_final_snapshot_20260911.

Final scientific limitation: MILD local rho FDR5 yields HOLD FDR3.896%/recall21.14%; MODERATE FDR11.765%/recall15.43%, failing nominal5%. No outcome-driven changes. This requested final audit/cleanup task is COMPLETE.

Next existing-job action: the frozen missing-library supervisor should resume at its next30-minute storage poll (about06:50UTC/14:50China) now that4GiB headroom is restored; last observed status still waiting, not yet a training-start claim. Do not create a duplicate job or bypass checks. V59 not changed or freshly audited during this task.

# Final V58 audit COMPLETE; cleanup pending pushed snapshot — 2026-09-11

All60 cases and final aggregate audited in results/v58_final_snapshot_20260911. All814 downloaded source hashes, candidate reporting gates, raw molecular counts, per-K/per-replicate outcomes, original CAL seals/threshold recomputation and exact final aggregate comparison pass. Remote finite arrays/completion, training/runtime binding and oracle/adopted sentinel checks pass. No training/rho rerun or scientific edits.

Complete HOLD rho local FDR5: MILD FDR3.896%/recall21.14%; MODERATE FDR11.765%/recall15.43%. MODERATE does not meet nominal5% FDR. See final_findings.md for raw, fixed-threshold and abundance results. Next successfully push this snapshot, then apply only its50-file/1.867GiB checkpoint retirement plan and reverify/push receipt. Final arrays/models and all scientific/provenance evidence remain. Missing-library supervisor currently waits for4GiB free space; do not bypass it or launch a duplicate.

# Paired MILD NNLS CAL screen COMPLETED — 2026-09-11

One new CAL_R1_K125 MILD pixelwise full391 NNLS run completed and independently reviewed in results/mild_nnls_cal_pilot (1082s CPU, no GPU/HOLD). Raw MILD NNLS TP125/FP86/FN0 versus ISTA122/109/3; cached CLEAN NNLS125/0/0. False identities overlap65, with21 NNLS-only and44 ISTA-only. Thus changing solver improves raw recovery but does not resolve mismatch false allocation. At retrospective CAL empirical5% FDR, NNLS rho recall22.4%, abundance37.6%; ISTA25.6%/36%. Shared identity rho values are exactly equal. No independent FDR guarantee or outcome-driven parameter change.

All source/output/script hashes, finite solutions, molecular/gate/truth membership and independent miss accounting pass. Full arrays remain remotely with export hash/size; compact data and analysis are preserved in Git. No cleanup. Pilot task COMPLETE; do not automatically expand. Prior channel-prediction route stopped and alternative-reference six-GPU proposal remains paused.

Existing experiment update06:30:57UTC: V58 has60/60 and reached MISSING_LIBRARY_PREFLIGHT, but supervisor is WAITING_FOR_DATA_DISK_EXPANSION. Missing-library has NOT started. Next existing-job action is final V58 audit/download/Git and narrowly dependency-checked checkpoint retirement if safe, to restore the required4GiB headroom; never bypass the guard or start a duplicate job. V59 status was not freshly checked in this CPU task. Earlier status below is historical.

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

# V58 storage review — 2026-09-11

50/60 completed-case snapshot audited under results/v58_completed_snapshot_20260911. Both severity CAL freezes use the original runner and full15 CAL cases before HOLD review. MILD HOLD15/15: CLEAN_FIXED rho FDR37.16%, recall80.69%; local FDR5 rho FDR3.90%, recall21.14%, filter loss1310. MODERATE HOLD5/15 is interim. No outcome-driven changes. Planned retirement of230 intermediate/redundant checkpoints frees8.588GiB across root/data, retains final models/arrays/all records and all sentinel cases; deletion occurs only after snapshot Git push, then hash-verified receipt.

# Active state update — 2026-09-11

NNLS six-case baseline and independent result review COMPLETE. See results/nnls_solver_baseline_k125_cpu4/analysis_record.md and local_review.json: raw NNLS HOLD TP375/FP0/FN0; rho FDR5/FDR1 TP372/FP0/FN3 (one identity repeated across three mappings). Same-subset ISTA raw TP361/FP354/FN14; rho FDR5 FP3, FDR1 FP0, no extra true loss. All30 source downloads, CAL seals and independent HOLD accounting pass. No universal filter-benefit claim.

V58 old supervisor stopped on storage archive failure after36 complete reports (old state counted35). Preserved original failure; replacement operational supervisor PID60098 now running unchanged scientific fingerprint. At01:49UTC completed37/60, active MODERATE__CAL_R3_K125. New status: /root/v58_jobs/storage_recovery_20260911/status.json. Remaining outputs stay on data disk; no old assets deleted. User reports adding20GB, but observed data capacity remains50GiB, free~7.67GiB; expansion visibility NOT_VERIFIED. Supervisor waits safely for space and automatically runs the frozen six missing-library fits after60 V58 plus aggregate and GPU release.

V59 formal PID4983 remains active in CPU postprocessing; three D0 CAL reports complete. All sentinels passed, but18 formal results are NOT complete. Do not launch a conflicting GPU job from zero instantaneous utilization. Snapshot files under results/active_experiments. No active desktop heartbeat is established by this update; remote recovery handoff runs independently of desktop sleep. Earlier dated progress below is historical.

# Active state update — 2026-09-10

Monitoring schedule corrected per user: first check2026-09-11 02:00 China time, then every30min; automation id automation. V59 all sentinels PASS; formal serial CPU postprocessing leaves GPU temporarily idle, not released (13:35UTC snapshot). Earlier V59 completion estimate omitted rho and is withdrawn. NNLS first case10501/15837 pixels at13:32UTC, no reported failure. V57 stratified review complete:122 miss contexts across30 identities; top5 account for48; no rho filter-induced true loss. See docs/EXPERIMENT_MONITOR_HANDOFF.md and results/computational_closure/v57/stratified_review.

Missing-library adaptation and CPU verification COMPLETE; six-fit execution frozen as b51137f3c6f12cf9bdf0e3e9f1c00c1fff53f9e4d62dc1c54de0b04a7910d9ef. All six input/386-candidate forward checks, omitted-truth FN, filter-loss, metadata and cache rejection tests PASS. Separate reload check reproduces fingerprint. See results/missing_library_challenge_k125/EXECUTION.md. READY_FOR_GPU_EXECUTION, not running or automatically queued.

CPU NNLS baseline now actually running on V58 host: supervisor PID41814 with four nice10 CPU workers, no GPU allocation. Six frozen V57 K125 CAL/HOLD R1-R3 cases; first case CAL_R1_K125 at PIXEL_NNLS. Runner hash matches committed8f21a4e. See results/active_experiments/nnls_cpu4_launch.json and docs/NNLS_BASELINE_PROTOCOL.md. Automatically seals CAL thresholds before HOLD, then compares to existing ISTA outputs. No completion claim yet.

V59 live check at 2026-09-10T12:55Z: no stall. D0/D1 sentinels PASS (final residual0.0417713/0.0525857); D2 actively training at epoch2150 with fresh history and100% GPU utilization. Supervisor stage timestamp is coarse, not an epoch heartbeat. Snapshot: results/active_experiments/v59_status_20260910T1255Z.json. No process restarted.

Three missing-library reference controls passed CPU reconstruction and production/checkpoint audit: original B bindings match byte-for-byte, all production implementation/config checks pass, 15 named checkpoints and final histories agree. See results/computational_closure/missing_library/control_reconstruction_audit.json. X_true hashes are frozen-design reconstructions, not independent historical seals. Reduced-N adapter remains pending; no GPU work launched.

Missing-library selection prepared: five close-neighbor truths and five same-class/exact-abundance matched controls, same IDs for HOLD_R1–R3 K125. Selection fingerprint bc5c0efd7fa07b0ecfd206ce37b189bb63a05d28d90df1dfdbebb1c874c0501a. Three control array/binding audits pass; execution freeze still awaits B/X_true reconstruction and implementation/checkpoint provenance. No new training. Git updates authorized and previous progress pushed to codex/v58-v59-run-records (321e684).

V57 closure accounting completed in results/computational_closure/v57: frozen rho FDR5/FDR1 retain all3253 raw HOLD TP, with112/8 FP and122 solver FN. Zero filter-induced true losses: ambiguity enrichment of that empty group is not estimable. Remote60 CAL source hashes pass; local molecular aggregate is byte-identical. Missing-library protocol is bounded to three K125 mappings and two five-identity omission arms (six new fits if controls verify), with exact asset manifest still pending; see docs/MISSING_LIBRARY_CHALLENGE_PROTOCOL.md. No new GPU job launched.

Computational closure preparation has started; see [COMPUTATIONAL_CLOSURE_PLAN.md](COMPUTATIONAL_CLOSURE_PLAN.md). Latest bounded status read: V58 29/60 complete; V59 supervisor remains at sentinel after all18 oracle checks. Local V57 analysis tables and required headers are present; provenance/statistical review remains pending. No new training launched.

V58 compact (60 datasets) and V59 standardized cross-library (18 datasets) are executing on separate remote instances. Authoritative current task context and runtime paths: [ACTIVE_EXPERIMENTS.md](ACTIVE_EXPERIMENTS.md). Final result review/commit/push remains pending. The V48/V49 notes below are historical context, not instructions to restart those experiments.

\# LD Project — Current State



Last updated: 2026-09-08



\## Current scientific objective



Build a reliability certificate for DIA-MSI lipid deconvolution.



Given:



\- A = frozen MS/MS library

\- B = acquired DIA-MSI data

\- X\_hat = deconvolution output



classify results into:



\- Tier 1: high-confidence individual lipid identity + reliable quantification

\- Tier 2: individual identity supported, quantitative uncertainty remains

\- Tier 3: individual member unresolved, but competition-group result is reliable

\- Tier 4: unresolved / reject



Priority:



1\. identity correctness

2\. group rescue

3\. quantitative accuracy



\## Frozen production solver



Status:



PASS\_PRODUCTION\_ISTA\_LOCKED



Canonical solver:



\- run\_758\_ista.py --mode export

\- lipid\_ista.LipidENNet.forward

\- 12 unfolded ISTA layers

\- production full spatial input: 1 × 1084 × 200 × 90

\- output: 391 × 200 × 90

\- no auto channel weighting applied to A/B during production inference



Do not retune or replace the production solver.



\## Acquisition / library



Isolation window:



748–798 m/z



Frozen production candidates:



391



All 391 are eligible in the isolation window.



Precursor-compatible groups:



110



K is an experimental stress axis, not biological K\_true.



\## v48 truth pilot



48 production-ISTA truth cases completed.



Pilot K:



13 / 25 / 55 / 103



Residual:



0 / 0.1



Selection:



random / hard\_competition



Replicates:



3



Main cached table:



v48\_pilot\_lipid\_observations.csv



Expected size:



48 × 391 = 18,768 rows



Do not rerun ISTA/X\_true/B\_sim for ordinary analysis.



\## Identity-first findings



Raw production ISTA contains substantial false allocation.



Therefore raw nonzero X\_hat cannot be treated directly as confident lipid identification.



Primary certificate candidate:



rho\_zero



Interpretation:



how much normalized fit degradation is required before a candidate can be forced to zero.



Higher rho\_zero = stronger data necessity / more trustworthy identity.



Mass90 validation:



\- status: CONDITIONAL\_IDENTITY\_SIGNAL

\- rho\_zero identity AUROC ≈ 0.816

\- top 5% coverage: identity precision 1.000, false mass 0

\- top 10% coverage: identity precision ≈ 0.9977, false mass ≈ 0.00017

\- top 20% coverage: identity precision ≈ 0.923, false mass ≈ 0.044



Interpretation:



rho\_zero provides strong identity-risk ranking at narrow high-confidence coverage,

but performance degrades at broader coverage and higher complexity.



\## Other features



necessity\_signal:

very similar to rho\_zero; do not treat as fully independent evidence.



profile\_relative\_width:

AUROC ≈ 0.854, but contains strong near-zero-reference denominator effect.

Secondary / diagnostic only.



global\_fragment\_cone\_residual:

AUROC ≈ 0.466.

Current negative comparator.



\## Group rescue



Frozen competition cutoff:



d\_frag = 0.02



Mass90 false allocation:



\- within-group ≈ 15.8%

\- outside-group ≈ 84.2%



Group rescue is useful for some member swaps,

but does not explain most false allocations.



\## Current methodological gap



Mass90 validation covers most false allocation mass,

but not every candidate that would actually be reported.



All-reported-candidate validation completed using existing abundance gates.



Historical reporting gates:



\- X\_hat > 1e-4

\- X\_hat > 1e-3

\- X\_hat > 1e-2



These are existing gates and must not be retuned based on current results.



\## Current stage



v48 identity validation complete

status = CONDITIONAL_READY_FOR_V49

frozen historical rho_zero implementation recovered

v49a code preparation complete

rho_zero parity NOT YET executed

no remote v49a cases executed


Cached input: `results/v48_pilot_48/v48_pilot_lipid_observations.csv` (18,768 rows)


Requested outputs written under `results/v48_identity_allreported/`.



No new simulation required.

No new certificate calculation required.

No v49 calibration yet.

