# Experiment Log

<!-- BEGIN v53_stage0_empirical_domain -->
## v53 Stage 0 — Empirical Domain Characterization

Version: v53_stage0_empirical_domain

Date: 2026-09-09T02:21:21.576702+00:00

Purpose: Characterize the locked real production reported-complexity, abundance, and residual domain before ground-truth calibration design.

Inputs / hashes:
- A_library: `/root/autodl-tmp/decon-lipid/adapter_pipeline/outputs/v38_ce29_fragmentionfixed_profile_overlap_empiricalfwhm_globalq99_ista_ready/A_library.npy` — `d19e88f6d25bcd66359ea2fcc21f07962b75470a2a1f54ea73938d4662312e69`
- B_cube: `/root/autodl-tmp/decon-lipid/adapter_pipeline/outputs/v38_ce29_fragmentionfixed_profile_overlap_empiricalfwhm_globalq99_ista_ready/B_cube.npy` — `e716031ee95953a071f34beb386381eeb05cb4198096951b504a46ea6cab99e4`
- foreground_mask: `/root/autodl-tmp/decon-lipid/adapter_pipeline/outputs/v38_ce29_fragmentionfixed_profile_overlap_empiricalfwhm_globalq99_ista_ready/foreground_pixel_mask.npy` — `a93f3adb97afe0295690d6f943917b6f0771aeeb9fff7d21582024c8ffbb4293`
- candidate_metadata: `/root/autodl-tmp/decon-lipid/adapter_pipeline/outputs/v38_ce29_fragmentionfixed_profile_overlap_empiricalfwhm_globalq99_ista_ready/candidate_metadata_final.npy` — `c6567b2e852c0e97255efbffe501f0527f2c8ea4daedbb181bb0d25308f230b3`
- channel_axis: `/root/autodl-tmp/decon-lipid/adapter_pipeline/outputs/v38_ce29_fragmentionfixed_profile_overlap_empiricalfwhm_globalq99_ista_ready/shared_mz_final.npy` — `85d6bee7a986f5e9cd2c038b7c4b207235f4d3293e2e890b820247dc52d434bf`
- production_checkpoint: `/root/autodl-tmp/decon-lipid/results_758_v38_ce29_empiricalfwhm_globalq99_fixed_library_joint_earlystop/latest_model.pth` — `32bcf603195883b18e4237d430ece1e99e98d184daf83b97c277d088f614d3ef`
- production_X_hat: `/root/autodl-tmp/decon-lipid/results_758_v38_ce29_empiricalfwhm_globalq99_fixed_library_joint_earlystop/X_abundance.npy` — `698bd3ed72b57ffe7a6f2bd16cee580388a442a0dbf1bdaaa6e819b0a5c419a4`

Experimental design: CPU-only descriptive audit of foreground production X_hat and residual R_real = B_real - A_solver @ X_hat; no training or ground truth.

Key results: reported-active-count q20/q50/q80 = 47/55/63; pixel abundance q20/q50/q80 = 0.0022702336/0.0074840626/0.033241708; global residual = 0.24941739.

Conclusion: The empirical production domain was measured without interpreting reported active count as true K.

Validity status: VALID_MAINLINE

Limitations: X_hat counts and abundances are estimator-dependent; R_real mixes experimental, library, and solver mismatch; no identity truth is available.

Decision: These distributions may inform a later frozen synthetic design; no calibration threshold is determined here.

Next step: Freeze a separate ground-truth calibration design using this empirical-domain record.

Git commit: PENDING_USER_COMMIT
<!-- END v53_stage0_empirical_domain -->

<!-- BEGIN v53_template_truth_pilot -->
## v53_template_truth_pilot

**Version:** v53_template_truth_pilot

**Purpose:** First direct known-truth test of rho_zero as a molecular-identity confidence certificate using empirical spatial/abundance templates.

**Experimental design:** 76 production-derived abundance maps were identity-remapped to 76 unique known synthetic molecular identities. Clean B_sim = A_solver @ X_true. Exact NNLS oracle followed by fresh learned LipidENNet deconvolution and rho_zero evaluation.

**Key results:**
- Oracle: TP=76, FP=0, FN=0, precision=1.0, recall=1.0.
- Learned molecular: TP=51, FP=45, FN=25, precision=0.53125, FDR=0.46875, recall=0.6710526315789473.
- Learned reconstruction relative residual: 0.023598041385412216.
- False molecular rho_zero maximum: 6.030171939676753e-15.
- True reported rho_zero median: 5.943263004138303e-07.
- Descriptive rho threshold 4.167353732004888e-09 retained 51 TP and 0 FP (precision=1.0, FDR=0), but is NOT a frozen deployment threshold.
- Training reached max_epochs=5000; stop_reason=max_epochs.

**Conclusion:** rho_zero shows strong proof-of-concept separation of true versus false reported molecular identities in this single clean real-complexity pilot.

**Validity status:** VALID_MAINLINE — proof-of-concept / pilot evidence only; not valid for final threshold calibration.

**Limitations:** one identity remapping; clean synthesis; no empirical residual; templates derived from production X_hat; global truth count 76; per-pixel true K not yet summarized; training hit max_epochs.

**Decision:** Proceed to replicated calibration across complexity and empirical residual conditions. Final rho_zero threshold frozen: NO.

**Next step:** Replicated ground-truth calibration.

**Git commit:** PENDING_USER_COMMIT
<!-- END v53_template_truth_pilot -->

