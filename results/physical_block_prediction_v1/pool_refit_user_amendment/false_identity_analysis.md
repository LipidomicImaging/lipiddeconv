# Actual joint score and retained false identities

The current model expands four evidence types (original abundance, rho, physical predictive gain S and positive-block fraction C) into eight logistic features. Spatial donor features are not part of this version. Scores always use original full-library evidence; they are ranking scores, not calibrated correctness probabilities.

The eight inputs are z(log10 X), z(log10 rho), their two squares and cross-product, rho==0, z(S), z(C). Floors are X=1e-12 and rho=1e-24. Actual coefficients: [2.454114460564734, 0.054664316402764895, -1.1545314317607194, -0.019488418416100366, 0.3984592563929103, 0.05451779004870727, 0.4440128255243357, 0.46076422407859086]; intercept 2.301860450255234. Full normalization, per-false logit contributions and hashes are retained in false_identity_analysis.json.

| False identity | Initial X | Refit X | Ratio | Score | S | C |
|---|---:|---:|---:|---:|---:|---:|
| PG 17:0_20:3 | 0.01449401 | 0.01469005 | 1.014 | 0.976806 | 0.0025428972 | 0.6667 |
| PG 14:0_22:3 | 0.00423618 | 0.01249219 | 2.949 | 0.914429 | -3.769995e-06 | 0.0000 |
| PG 16:0_20:5 | 0.00501126 | 0.00637809 | 1.273 | 0.814188 | 0.0027361454 | 0.7778 |
| PE O-18:1_20:2 | 0.00284086 | 0.00602416 | 2.121 | 0.641051 | -3.2442667e-05 | 0.0000 |
| PG 18:2_20:2 | 0.00560808 | 0.00523634 | 0.934 | 0.687171 | 0.00062353772 | 0.2222 |
| PE 17:1_20:1 | 0.00291915 | 0.00426143 | 1.460 | 0.669979 | -0.00017064615 | 0.0000 |
| PE O-16:1_22:3 | 0.00500787 | 0.00369613 | 0.738 | 0.807782 | 0.0051371048 | 0.7500 |
| PG 17:1_20:4 | 0.00512493 | 0.00323028 | 0.630 | 0.772987 | 0.001756792 | 0.6000 |
| PE O-18:2_22:5 | 0.00899789 | 0.00196343 | 0.218 | 0.971108 | 0.0035014967 | 1.0000 |
| PA 20:1_22:6 | 0.00425122 | 0.00150097 | 0.353 | 0.674121 | 0.016327537 | 0.4286 |

Intensity means model coefficient averaged over the full foreground, not absolute detector counts or a composition percentage. All ten remain above the unchanged0.001 gate. Five increase and five decrease; none crosses out of the reported set. The two highest false coefficients exceed58 and51 of121 retained true coefficients, respectively.

Three retained false identities have rho=0, S<0 and C=0: PG14:0_22:3, PE17:1_20:1 and PEO-18:1_20:2 (see exact spaced names in the table). The unconstrained additive logistic model allows other terms, including the rho square, to offset this evidence. The other seven have positive S and C; PG17:0_20:3 and PEO-18:2_22:5 even pass the old strict score. Thus predictive contribution and high fitted abundance do not certify true identity in this case. No specific donor or causal interference mechanism is established by this cache analysis.

Raw125TP51FP becomes screened121TP10FP and refit121TP10FP. No additional refit true loss and no refit false removal occurred. The separate fixed DEV5% cutoff transfer gives CHECK117TP6FP, FDP4.87805%, recall93.6%; it is not this refit pool.
