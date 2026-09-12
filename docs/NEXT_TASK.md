# 当前任务 ACTIVE：共享模型与新CAL/HOLD验证 — 2026-09-12

用户授权持续分析/探索直到1%错误与80%recall/retention目标。当前诊断：既有排序的事后oracle可达107TP/1FP、85.6%，但不能用其阈值；旧每折CAL仅25–41身份。依 docs/IDENTITY_CONFIDENCE_CONTINUOUS_PLAN.md 先冻结同一六特征共享模型与四个新5%case，CAL1优先，失败停止该版本后分析；通过才CAL2→seal→HOLD1/2。真值CAL/HOLD各125且不相交；使用既有完整空间图、不改变旧实验、不启GPU。允许的源为当前NNLS/联合结果、原physical mainline四个明确source的X/mask、既有完整components和必要solver/helper代码；新输出 results/identity_confidence_joint_validation。先实现/封存/推送，再CPU执行；逐case审查下载Git后继续。总目标保持active，不能用已曝光数据调参过线冒充完成。

# 当前任务 COMPLETED：有界辅助指标探索

用户授权的四辅助量、三类单加与全加入固定比较完成；没有方案改善既有93TP/1FP、74.4%联合baseline。主扩展92TP/2FP，73.6%recall、2.1277%FDP，未过1%/80%。保留全部特征/模型/名单/分组/审查记录于 results/small_mismatch_auxiliary_confidence；两端核验与25文件hash均PASS。最后推送紧凑结果后STOP；不根据失败增加变体、改正则/阈值/分组，不对已曝光case声称独立FDR。没有新解卷积/rho/GPU/病例/监控/清理。主线仍开放，当前仅表明这四个固定辅助量没有带来增益。

# 当前任务：固定四个辅助量的缓存开发探索 — 2026-09-12

用户最新授权探索其他辅助指标。仅按 docs/SMALL_MISMATCH_AUXILIARY_CONFIDENCE.md 使用已完成5% case的 nominal A_solver、X_hat、mask、candidate/molecular records、metadata、result/design/summary，以及已冻结联合模型的seals/predictions。新增代码 analysis/run_small_mismatch_auxiliary_confidence.py 与analytic test；新输出 results/small_mismatch_auxiliary_confidence。四个observable量分为谱库/竞争、global-mean一致性、空间集中度三类，固定三类单加及全部加入四扩展，原baseline不重训。冻结并推送后仅20个小CPU分类器，复核/Git后停止；全部旧fold已曝光，不宣称独立FDR、不改旧结果、不做NNLS/rho/GPU/新case/清理。

# 当前任务 COMPLETED：固定联合指标筛查

本次联合策略得到93TP/1FP，74.4%召回与1.06383% FDP；优于同拆分单独rho/丰度，但未满足严格1%/80%。报告 results/small_mismatch_joint_confidence/analysis_record.md；模型、全部molecular记录、CAL seals、hash及远程/本地缓存审查均保留。仅完成一次固定CPU分类器，未重新解卷积/rho/训练GPU，也未启动新case。现在提交推送明确列出的紧凑结果后STOP，不在已曝光test上增删特征/调参/改阈值/重试。主线独立验证与80%目标仍未完成；不因此宣称联合方法不可能。

# 当前任务：一次固定联合指标CPU开发筛查

用户最新联合指标请求取代下方已完成任务的“不运行事后联合模型”限制。范围仅 docs/SMALL_MISMATCH_JOINT_CONFIDENCE.md、analysis/run_small_mismatch_joint_confidence.py、原 analysis/run_joint_confidence_cal_pilot.py 和 results/small_mismatch_nnls_first_case/case 的 molecular_records/design/result/summary JSON；新结果写 results/small_mismatch_joint_confidence。冻结输入/模型/五组membership并推送后，用原主机已有依赖执行一次固定模型；缓存独立审查、下载hash、记录比较、Git后STOP。已曝光单case无独立FDR保证；不调参/换特征重试，不运行优化解卷积/rho/GPU/新case，不清理。

# 当前追加分析已完成 — 80%召回与错误率的现有曲线权衡

用户要求检查80%召回。已只读重计完整rho曲线并比较已有X_hat基线：rho80%对应14FP/12.28%FDP，X_hat80%对应3FP/2.91%；严格1%限制下二者最大召回48%/43.2%。保存 results/small_mismatch_nnls_first_case/recall80_tradeoff.json 和.md 后commit/push并STOP。不修改冻结阈值/原GO，不运行新实验或事后联合模型。后续80%方案及独立验证尚未冻结。

# 当前任务已完成 — 单组NNLS开发GO，保存后停止

用户授权的唯一首组NNLS_FULL_LIBRARY_5PCT__CAL_R71_K125已完整执行和审查：原始125TP/49FP；筛选60TP/0FP，48%retention，开发GO。报告与全部计数/curve/provenance位于 results/small_mismatch_nnls_first_case。333个文件下载验证通过，最终数组和全部源记录保留。数值汇总的跨平台float32差异已单列，身份计数和阈值曲线一致。

本轮工作完成，提交并推送明确列出的紧凑结果后STOP。没有新case、GPU或缺库实验排队，不恢复旧五组，也不删除数据。后续独立验证需要单独冻结设计；当前0FP/60与48%只是开发结果，不宣称独立FDR已控制。一次后续处理nnls-git已不存在，无需再监控。

# 当前状态 — 单组运行中，部分像素预览已完成

用户追加的10000像素缓存预览已完成，属于非最终原始NNLS结果。首组仍运行；不实时轮询、不改正在执行的代码或科学合同。一次后续处理 nnls-git 安排北京时间2026-09-12 16:45：结束后执行缓存独立审查、结果下载核对和紧凑Git保存；若未结束，只按剩余时间延后一次。最终阈值/GO尚未评估，不把部分预览作为完整实验结果。继续范围与下方唯一首组合同相同。

# 当前任务 — 仅一组完整库5%失配，先NNLS，失败即停止

LATEST USER REQUEST supersedes preparation-only scope below: execute exactly NNLS_FULL_LIBRARY_5PCT__CAL_R71_K125,seed7301,CPU NNLS+existingrho_zero, complete391candidate library andcachedR71spatial/relativeabundance,oneglobalsignal scalar. NoGPU, nooldfivefits, nomissing/HOLD/secondseed queue, no newconfidence score. Requirecompletephysicalcomponent reconstruction thenfreeze/push exactinputdesign before outcomes. Review fullcurve forsinglecaseFDP<=1% andTPretention>=40%; NO-GO ifnone, technicalunresolvedkeptseparate. Independentreview/download/hash/Git thenstop; evenGOdoesnotautolaunchnextcase. Details:docs/SMALL_RELATIVE_SPECTRAL_MISMATCH_5PCT.md.

User chose5%relativeSD for new independentfragment/precursor intensity mismatch around correctly matched fixed library. Complete only the reproducible perturbation helper, tiny invariant tests and docs/SMALL_RELATIVE_SPECTRAL_MISMATCH_5PCT.md; retain whole physicalion envelopes and per-experiment fixed spectra. No pixelnoise/dropout/newCEassumption. Positive mean1/exactCV=.05 lognormal multiplier is permitted by user'sGaussian-or-other wording; record pre-column-normalization semantics explicitly. Require complete nominal-library component reconstruction, never silently reuse partialCE coverage. Do not launch unsealed newGPU/scoring or resume oldfive runs. Save checks/protocol/code inGit, mark this preparation complete and stop; overall mainline remainsOPEN.

# Historical completed task — 首组 CAL 提前止损，剩余五组不启动

The user-directed firstCAL screen is COMPLETE: EARLY_STOP_CAL_FUTILITY.23TP/0FP,18.70% TP retention; supported perturbed truths2/26(7.69%).These are also the maximum counts over the unchanged threshold curve even without an FDP constraint. Raw solver123TP/109FP/FN2; no numerical unresolved result. See results/physical_identity_first_cal_feasibility/analysis_record.md and screening_report.json.479 files downloaded/hashPASS; save the reviewed compact result and independent accounting in Git, verify successful push, then STOP.

Do not resume the otherfive GPU fits, official CAL/EVAL scoring or local coordinator. Remote execution stop receipt confirms allfive not started and GPUidle. No scientific tuning, cleanup, new prerequisite audit or automatic next experiment. Broader mainline objective remainsOPEN; this is a resource-stop result on firstCAL, not independent EVAL failure or impossibility proof.

# Historical plan — 主线 identity robustness pilot

LATEST USER PRIORITY2026-09-12: pause the remainingfive GPU fits; firstCAL has completed/reviewed/downloaded/pushed382ee209. Follow docs/PHYSICAL_IDENTITY_FIRST_CAL_FEASIBILITY.md for a CPU-only firstCAL screen using unchanged U/full-deleteLP/gamma/accounting. Keep its evidence/threshold exploration separate from officialsix-case seals. Independently review, download/hash/Git the result; user-authorized earlystop is a developmental resource decision, not formal EVAL NO-GO. Secondcasehasnotstarted; do not automatically resume the coordinator before this decision.

Physical interpretation corrected by user: keep the best available library for the actual CE fixed; no CE-selection/between-CE-error project. Any later reality validation concerns same-CE residual mismatch. Existing joint-CE endpoint pilot continues unchanged as its declared controlled test, without claiming its range is the same-CE residual range.

User-requested future reasoning is recorded in docs/IDENTITY_CONFIDENCE_DECISION_PATH_20260912.md; no extra prerequisite or scientific edit is introduced. Keep executing the current six-case pilot. After its reviewed outcome, choose only the evidence-supported next branch; do not tune the exposed EVAL or equate a feasible deletion relaxation with a physical replacement proof.

RUNNING2026-09-12: first production fit launched06:27:58UTC on westc, PID115253, after design commitdbf304f was pushed/verified. Continue exact six-case per-fit handoff; do not repeat prepare or duplicate first launch. No outcomes yet.

PREPARED2026-09-12: six-case design a86cadbfcafe8abc9a12810c3ee71d673c5a1bae55c018824fb952e07bdac9e4 passes independent preparation and exact local15-file transfer review. Push the prepared design, then start SUPPORTED_MISMATCH__CAL_R71_K125 on westc from the frozen source snapshot. Do not repeat prepare. Complete each fit's review/download/push before writing its handoff ACK and starting the next. Source/CE uncertainty/denominators remain frozen; no EVAL confidence before the CAL seal. No new outcome is available yet.

ACTIVE2026-09-12: implemented under docs/PHYSICAL_IDENTITY_MAINLINE_PILOT_V1.md;22 CPU analytic/integration tests PASS. Sourcevalidation in results/physical_identity_mainline_pilot/implementation_validation.json. Prepare exactsixK125R71/R72cases in /root/physical_identity_mainline_v1/results from /source snapshot, independently checkpreparedbindings andpushdesign beforeGPU. Use existing fixedproductionA/training and target-basedONEglobaldataset scalar; bothmissingarmsnowjointwithsupportedmismatch. Aftereachfit: reviewer→compactdownload/hash→commit/push→handoffACK, thennextfit. Afterallsixhandoffs: oneCALseal→EVAL→independentproof/metricsreview→Git. Preserveboundmodels/arrays/checkpoints anddonorseparation. No newoutcomeexistsyet; do not markpilotcomplete untilreviewedresults. OldNOT_STARTED entriesbelow arehistorical.

The single physical prerequisite is COMPLETED: docs/PHYSICAL_PERTURBATION_CONTRACT_V1.md and results/physical_perturbation_contract_v1, fingerprint3b2e75c563a0d23ac16db2ce87f1815ae0c3574fac59c3d8e89b794768697df7. Finish this turn by pushing the verified contract, then stop the freeze task. The next task is this one mainline pilot, not another audit. Current pilot status: NOT_STARTED; no new GPU/scoring job exists from this contract freeze.

Implement the frozen finite joint-endpoint systematic mismatch using existing production component/spatial/solver infrastructure.79 observed whole vectors in five supported strata map to69 PEO-H candidates;322 others fixed. No independent box, extra severity scale, interpolation/dropout, new architecture/rho/K ladder/spatial variant or extra CE experiment. Censoring NOT_ESTIMATED and pixel fluctuation NOT_SEPARATELY_IDENTIFIABLE are explicit exclusions, not reasons to postpone this scoped pilot. Do not change V2 or reopen its outcomes for tuning.

Within this single pilot's preparation, prospectively seal exact fresh case membership/counts, identity-and-shared-source grouped MODEL/CAL/EVAL donors, inference implementation, original denominators and supported-perturbed-truth reporting. No CAL/EVAL generator endpoints in the inference dictionary; no target information passed to confidence. Preserve production training/early-stop, fixed A_solver and reporting semantics. Original physical-model feasible witness is required for full acceptance; any convex deletion relaxation is identified as a conservative lower bound. Evaluate missing-library and keep solver misses/filter-induced losses/abstention separate. Preserve1%/40% utility-risk targets with60% only strong success; no claim of realistic or unseen robustness from fixed-truth dilution or dictionary containment. Calibrate only on CAL, seal before EVAL, run once, independently review, save compact outputs and Git; no result-driven change to U or repeat on exposed EVAL.

Use only the frozen contract inputs and required existing production/pilot functions. Preparing cases/verification within this task is implementation, not permission to open another CE audit series. Apply the existing per-case review/download/push/dependency-safe retention contract to any newly executed fits. Future independent formal validation follows only a favorable reviewed pilot; weak/pixel extension awaits separate evidence later, without blocking the current supported-scope experiment.

# COMPLETED — bounded CE joint-variation check; V2 remains closed — 2026-09-12

Latest user priority: measure physically supported correlated fragment changes before a new identity method/open-set pilot. This turn checked the existing CE133 paired records, their directly referenced target table and original uncertainty/LP code only. Outputs: results/ce133_joint_variation_audit; reproducer: analysis/audit_ce133_joint_variation.py. No archive search, source-data edits, uncertainty fit, solver, training, rho, remote job or deletion.

The independent fragment box is confirmed but its causal role in V2 failure is unproven. Joint vectors are preserved; covariance/rank descriptions are conditional on exact CE/support and are not physical coverage or independent-replicate estimates. Strong/weak boundaries, censoring, pixel variation and independent prediction-error coverage remain NOT_VERIFIED. Do not generate a new sigma/dropout or claim a complete U_phys from this check. Full findings, supported next-model scope, missing data and margin normalization/cone caveats are in analysis_record.md. V2 stays NO-GO / BOTH_NO_GO with no V2.1 or spatial-weight iteration.

Commit/push only this bounded result, its reproducer and CURRENT_STATE/NEXT_TASK, then STOP. No automatic V3, identifiability pilot, raw MSI processing or additional audit loop is queued. A later small pilot requires a prospectively frozen physical model and fresh evaluation; do not reuse exposed V2 outcomes for model/threshold selection. Broader 1% risk / 40% retention objective remains open.

# COMPLETED — V2 pilot, independent review and result handoff — 2026-09-12

V2 has finished and independently passed5572 numerical proof checks plus source hashes, CAL-only seals and all accounting. Method NO-GO; paired result BOTH_NO_GO. Local aggregate TP retention4.57%, global6.86%; local challenge retentions12.20%/0%/0.89%, all below40%. Report: results/ce_identity_spatial_v2/final_interpretation.md. Main scientific objective remains open, but this version is closed without tuning or a new validation launch. No further intermediate task is authorized by the V2 contract.

Final scientific snapshot and single-archive retirement plan pushed/verified ase99f98e37c0c6c5385556c05be6841fdfe5d98fb. Exact final.tar.gz retired after local backup and source checks; all5689 output hashes/106 protected training hashes still pass. Receipt is in results/ce_identity_spatial_v2/final_archive_retirement. Preserve all local/remote numerical proof and evidence files, trained arrays, source results and /root/v2_retained_runtime_20260912. The postprocessor archive-ready status is now a historical export record; do not restart V2 or recreate retired archives. No active V2 process or GPU allocation remains. After pushing the closing receipt/documentation, STOP; no new task, tuning or physical-validation experiment is queued by this file.

# Historical V2 execution instructions

Both CAL seals verified; R62 EVAL active at03:36UTC2026-09-12. Proceed only through existing EVAL -> independent review -> final report/Git. The final report must separate unchanged local METHOD_GO/NO_GO from paired SPATIAL_CONTRIBUTION (local-only/both/neither/global-only GO), with each missing-library arm's recovery and40% gate explicit. V1 zero retention is historical context; same-batch global is the attribution control. No extra intermediate task or current-version patch. A method GO may motivate a new prospective physical-validation design without claiming spatial superiority when both controls pass. Latest user clarification supersedes older wording that conflated those conclusions; see docs/CE_IDENTITY_SPATIAL_V2_PROTOCOL.md.

Storage recovery COMPLETE after pushed planb846f40:36 checkpoint files now retained in /root/v2_retained_runtime_20260912 through original-path symlinks, all original bindings verified. Six already-downloaded transport archives retired; all six case snapshots remain local/in Git. Do not download those retired archives again or remove the protected retained store. Exact mapping and receipt: results/v2_storage_retirement_20260912. Data free~3.42GiB; final screening/export/review still pending. Before future experiment launches, define storage budget and safe retention dependencies alongside the artifact contract; do not again bind every disposable intermediate checkpoint and then assume it can be deleted during execution. Current frozen V2 code and scientific definitions remain unchanged.

Latest handoff2026-09-12 03:06UTC: all six V2 fits and independent spatial-evidence reviews complete; all six compact case records downloaded. Main job now scores CPU LPs; status.json still names the last training case, so also read the main log and bounded postprocessor status. Final CAL seals/EVAL/GO review is pending. Separate cqa1 V59 is7/18 complete and still running serial rho;91 compact source files plus seven case reviews recovered in results/v59_completed_snapshot_20260912. Continue incremental verified result handoff; do not treat either experiment as final or interrupt its live job for a progress check.

Follow docs/CE_IDENTITY_SPATIAL_V2_PROTOCOL.md and analysis/run_ce_identity_spatial_v2.py. User explicitly requested implementation/execution with frozen U v1 and gamma1e-6. Prepare six genuinely new spatial cases using the inherited V57 recipe and fixed new seeds6201/6202; retain original identity/abundance/solver/mismatch contracts. Execute fresh production X_hat fits; old run04 only input/debug/novelty references. Freeze input novelty before LP scoring; separate global/local CAL epsilon seals before either EVAL. GO requires aggregate FDP<=1%, aggregate and EACH challenge retention>=40%, each nonempty;60% aggregate only strong success. Preserve all nonselections in denominators.

Required outputs: frozen design/protocol/source bindings, evidence/novelty records, complete production artifacts, numerical proof files, independent review, raw/filter-loss/status metrics and paired global/local report. Reuse existing code and known paths only. Review/download/Git each completed case; no cleanup, rho rerun, additional benchmark or outcome-driven change. After reviewed success or failure, update state, commit/push and STOP this pilot; no automatic independent validation or open-set expansion.

RUNNING on westc: main PID102539, independent postprocessor PID103069. Read only/root/autodl-tmp/lipiddeconv/results/ce_identity_spatial_v2/status.json and/root/v58_jobs/ce_identity_spatial_v2_audit_status.json first. Verified archives appear in/root/v58_jobs/ce_identity_spatial_v2_exports with per-archive receipts. First MILD R61 and close-neighbor R61 snapshots downloaded under results/ce_identity_spatial_v2; local_transfer_review.json documents large evidence arrays deliberately excluded from Git. Download/verify new archives incrementally; do not overwrite different existing bytes, rerun completed fits, infer archive-ready means pushed, or delete retained arrays/models. When final archive is ready, verify all proofs/accounting/seals, review findings.md, then Git and mark this task complete.

# COMPLETED — CE boundary and six-context identity pilot — 2026-09-11

All six contexts finished; all numerical proofs, original source/output hashes, fixed calibration and accounting independently PASS. Results and analysis: results/ce_uncertainty_identity_pilot_run04/findings.md. R2 FDR2.22%, TP retention12.43%: STOP_CURRENT_VERSION under the frozen1%/40% rule. MILD alone35.48% retention; omission arms retain none. No result-driven threshold/U changes or automatic formal validation.

The main high-confidence identity endpoint remains OPEN. This pilot does not prove impossibility. Its normalized R1/R2 mean spectra are nearly identical, and the single false selection is within5.08e-12 of the threshold; neither is evidence of independent population-risk validation. Preserve the exact negative result and limitations. Finish the final Git push, then STOP this bounded task; no further audit, GPU run, benchmark expansion or cleanup is queued by this file.

# CE pilot: calibration complete; finish frozen evaluation — 2026-09-11

Run04 has three reviewed R1 calibration cases and sealed epsilon0.019717481319156117. Calibration44 TP/0 FP,12.36% retention: do not tune or change the model. Finish the three R2 development evaluation cases, independently validate all proofs/counts/seal, preserve each completed case in Git, and record this version's continue/stop outcome. Formal main endpoint remains OPEN. No GPU, cleanup, new benchmark or automatic independent validation.

# CE pilot: two calibration cases reviewed; strict numerical recovery — 2026-09-11

MILD CAL_R1 and close-neighbor omission R1 completed and independently passed232/245 full-plus-deletion proof checks. The recovered snapshot is results/ce_uncertainty_identity_pilot_run03; the earlier first-case snapshot remains separate. Four independent CPU workers preserve the original LP, tolerances,60s per-problem budget and CAL-before-EVAL ordering.365 saved identity results were revalidated, not silently rebound.

Relatively-isolated R1 full-model LP was rejected because its bound gap1.52e-6 exceeded the unchanged1e-6 limit, before any calibration or evaluation result. Retry of the identical problem now also covers failed numerical certification, using HiGHS IPM within the same budget. Preserve all run00-run03 records; reuse only independently valid completed evidence in a new run04 directory. No U/threshold/target change and no GPU training. Main endpoint remains independently validated FDR<=1% and retention>=40%; no pilot performance conclusion yet.

# CE identity pilot RUNNING; first case independently reviewed — 2026-09-11

Remote CPU run:/root/v58_jobs/ce_uncertainty_identity_pilot_run02, PID81846, code frozen as79b01682fe08ca3c501af9b4d09d3e99278fa0c0. Two earlier execution stops are preserved (parser mode, then invalid HiGHS status before first case completion). Same LP now retries HiGHS IPM only after non-success, sharing the original60s cap and unchanged tolerances; partial evidence saved every25 identities. No scientific model/target changes.

MILD__CAL_R1_K125 complete:84 compact source hashes and232 full/deleted numerical proofs independently verified, including89 zero-coefficient witnesses; one backend non-success recovered. Snapshot in results/ce_uncertainty_identity_pilot. Final selection/calibration/evaluation accounting deferred until sealed pilot completes. Continue remaining five contexts, preserve each completed case, then audit all counts and bounds and push. No GPU, original rho rerun, additional benchmark or cleanup. Formal independent FDR<=1%/retention>=40% endpoint remains OPEN.

# CE-informed identity pilot ready for execution — 2026-09-11

User explicitly authorized uploading the305KB CE boundary, pilot script and protocol to westc:55786 and running the CPU pilot. The same CE source table already exists there with matching SHA. Both local and remote analytical LP self-tests pass; local nominal-spectrum parity is5.55e-17, miss/abstention accounting tests pass, and all7 boundary output hashes match. Code and frozen development contract: docs/CE_UNCERTAINTY_IDENTITY_PILOT.md; inputs: results/ce133_uncertainty_v1_ready. Runtime/seal checks and explicit reportable-recall fields were completed before any data scoring.

Next execute exactly six cached contexts: R1 developmental calibration/R2 evaluation, each MILD plus close-neighbor and relatively-isolated missing-library arms. Keep FDR<=1% and TP retention>=40% continuation criteria;60% optional strong success. Save calibration seal before R2 scoring, review all numerical proofs and counts, download and push results. Original solver/rho/thresholds unchanged, no GPU training or new cleanup. This is developmental reuse of exposed data, not formal independent HOLD. The broader identity objective remains OPEN.

# Westc missing-library audit and scoped cleanup COMPLETED — 2026-09-11

Six completed omission fits and three cached full-library controls have been reviewed. All127 compact source files and the exact24-file retirement plan were successfully pushed as95591b05366097b2773e349c9098ee693af0f786 before deletion. The24 ordinary intermediate checkpoints were then retired;949983592 bytes (0.885GiB) freed. Data free space rose from3803459584 to4753518592 bytes (4.43GiB; df rounds to4.5G,92% used). GPU remains0%/0MiB. All178 protected remote hashes and127 local source hashes pass after deletion.

Final learned arrays, result-bound latest_model.pth, epoch3000 models, candidate/molecular records, histories/diagnostics, CAL sources/seals and all full-library parent checkpoints remain. The cleanup never touched V58/V59 or production assets. Receipt, exact removed paths/hashes and independent miss accounting are in results/missing_library_final_review_20260911. py_compile and diff-check pass. No training, synthetic reconstruction, rho or threshold recalibration was executed. The requested cleanup task is COMPLETE; push the receipt and stop. The separate CE uncertainty pilot remains unfinished and was not launched during this task.

# Missing-library final audit complete; scoped retirement pending push — 2026-09-11

All six reduced-library fits and three reused complete-library controls reviewed in results/missing_library_final_review_20260911. Remote runtime/design, normal completion, finite outputs/history/final models and source hashes pass; all127 downloaded compact source files match, independent local molecular counts match. Original V57 CAL thresholds remain fixed. rho FDR1 transfer yields observed FDR38.24%/39.74% in close-neighbor/relatively-isolated arms versus0/361 false contexts in the complete-library controls. This is CLEAN missing-library stress, not joint mismatch+omission; three mappings are not biological replication.

Westc data disk is93% used (3.6GiB available), GPU0%/0MiB. New six-fit output accounts for about1.77GiB. Pending plan: after successfully pushing this exact six-case snapshot, retire24 intermediate checkpoints (epochs1000/1500/2000/2500),949983592 bytes. Preserve all178 protected dependencies including latest_model.pth, final epoch3000 models, arrays and every parent-control checkpoint. Then verify retained hashes and push deletion receipt. No broad scan, source-result changes, training or rho rerun. CE uncertainty pilot remains separate pending work; no new job started.

# CE error boundary complete; direct identity pilot next — 2026-09-11

User endpoint: independently validated FDR<=1% AND TP retention>=40%;60% is optional strong success. Do not expand audits or seek an impossibility conclusion. CE133 boundary completed in results/ce133_uncertainty_v1_ready:120 usable identities/244 CE pairs,17 supported rule-channel strata,480 coupled physical fragment components in107 production candidates; unsupported parts fixed. This is a limited CE-informed envelope, not complete MSI error coverage. Partial serialization-preflight export retained separately.

Next execute the implemented CPU LP pilot under docs/CE_UNCERTAINTY_IDENTITY_PILOT.md: six cached contexts, R1 development calibration/R2 development evaluation, each MILD plus two missing-library arms. All candidates compete; full feasible residual bound and deleted global lower bound define selection. Seal the calibrated tolerance before R2 scoring. Fixed1%/40% targets, no new training/rho or outcome-driven changes. After scoring, independently review preserved proofs/counts/source hashes, download and push. Formal independent validation only after a reviewed favorable pilot.

Remote bounded status check confirmed all six existing missing-library computations complete (supervisor ALL_COMPUTE_COMPLETE_AWAITING_REVIEW); their final standalone scientific audit remains pending. No remote job terminated or data deleted. No new V59/V60 expansion or quantification work.

# Identity-first priority and utility targets recorded — 2026-09-11

Primary route: docs/IDENTITY_FIRST_UNCERTAINTY_ROUTE.md. User targets fixed for the NEW route: FDR<=1%, TP retention>=40% minimum continuation,>=60% strong utility; quantification deferred. Exact U, epsilon_valid, pilot membership/aggregation and independent risk assessment are NOT yet frozen. Abstention keeps its original denominators; safe missing-library rejection can still fail utility. Numerical optimizer failure is not proof of identity or model incompatibility. Do not apply new targets retrospectively to completed frozen pilots.

NEXT: bounded CE133 capability audit using results/rho_mismatch_mechanism_ce_audit, its directly referenced target CSV and required direct source/OOF lineage. Assess valid channel alignment, peak absence versus masks/censoring, ambiguous/shared peak assignments and supported co-variation; no new score, arbitrary perturbations, raw processing or training. Output a supported U proposal or explicit data insufficiency. Prior133/66 counts establish availability only, not completion of this deeper capability audit.

New V59/V60 expansion and large quantification experiments deferred. Existing job audits/Git preservation and the missing-library challenge remain relevant; no remote job stopped or artifact removed by this planning turn. Known-support/full-library/group quantification is future work and needs separate coefficient/ion-signal/molar estimands. Planning record complete; broader identity objective remains OPEN.

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
