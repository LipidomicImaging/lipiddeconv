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



v48 all-reported-candidate identity validation COMPLETED


Cached input: `results/v48_pilot_48/v48_pilot_lipid_observations.csv` (18,768 rows)


Requested outputs written under `results/v48_identity_allreported/`.



No new simulation required.

No new certificate calculation required.

No v49 calibration yet.

