# Active state update — 2026-09-10

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

