V58 mismatch ranking diagnosis, completed 2026-09-11.

This analysis uses the already reviewed50-case snapshot and recomputes no solver, spectrum or rho. All412 source hashes pass. MILD has the full15 HOLD cases; MODERATE has only5 and is unbalanced across K/replicate. All results here are developmental/descriptive after HOLD inspection, not new confirmatory evidence.

MILD pooled reported-candidate ranking:

| Score | AUROC | Average precision | Retrospective max recall at empirical FDR<=5% | At FDR<=1% |
|---|---:|---:|---:|---:|
| rho_zero | 0.7635 | 0.8026 | 28.86% | 14.29% |
| X_hat | 0.8588 | 0.8735 | 31.54% | 6.80% |

These envelopes select among thresholds on the observed HOLD labels, retaining whole ties. They are optimistic descriptive limits of one global threshold on each existing score for this finite snapshot, not operational cutoffs, population upper bounds, or permission to replace frozen CAL thresholds. AUROC/AP are conditional on passing the original reporting gate; recall uses all truth contexts, including solver misses. The existing CAL-frozen rho FDR5 result is21.14% recall at3.90% HOLD FDR: calibration adjustment alone cannot recover high recall with the same global ranking. Abundance is stronger over the overall ranking and at the5% envelope, whereas rho is stronger in the1% tail. Neither dominates at every risk level.

Within K50/125/175, rho's descriptive5% recall envelopes are52.00%,28.80%,21.71%; abundance's are54.40%,39.52%,33.49%. These are separate post hoc within-K thresholds, not a permitted recalibration protocol. K changes abundance under fixed-total-signal construction as well as mixture composition, so this is not a causal estimate of candidate complexity alone.

268 raw true records and679 false records have rho exactly zero in MILD. The current returned score cannot separate these tied records. A zero is the output of the fixed-library residual-difference calculation and clipping; it does not prove absence, intrinsic non-identifiability, or numerical correctness of every underlying optimization. A future diagnostic should inspect unclipped differences and solver optimality before assigning a mechanism.

At identity level, MILD has152 identities with at least one filter loss,22 with raw recovery and no filter loss, and1 never recovered by the raw solver. Median cone isolation is0.276 versus0.575 in the first two groups, and maximum fragment cosine is0.925 versus0.691. This is consistent with greater library ambiguity among identities experiencing loss. Median full-spectrum change (1-cosine) is0.00234 versus0.00258, and dropout fraction0.0445 versus0.0533: this comparison does not support the simple explanation that filtered identities merely received larger perturbations.

This is not an adjusted enrichment test. Identity groups have unequal numbers of nested-K contexts, abundance and spatial conditions may confound the comparison, and the same perturbation is reused across mappings. Grouping by identity avoids counting repeated records as independent identities but does not remove those confounders. Do not train a rule to reject ambiguous identities from this analysis: that would likely reinforce the recall loss.

MODERATE descriptive5% envelopes are13.33% for rho and26.48% for abundance; its partial/unbalanced subset cannot establish a severity effect. Complete V58 and V59 retain their original definitions and analyses.

Next research step: the separately frozen protocol in docs/MISMATCH_CONFIDENCE_PILOT.md tests individual-identity support with alternative spectra, using new perturbation draws and explicit missing-library limitations. It is a research hypothesis; improvement has not been demonstrated. No additional GPU run, new operational threshold or cleanup is performed by this diagnosis.
