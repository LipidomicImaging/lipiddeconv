# CHECK complete: frozen screening results, pool refit pending

The 34 physical-block units completed with **12,852 actual NNLS calls** (34 full and 12,818 all-alias molecular deletions). Original design/input snapshot: `df8844c02be8fcab5b4df0f652fa760722e3cdb7`; fixed model snapshot: `2ce3b343383888eb7b4691729938c1b2bf36c8be`. All 391 candidates and 377 molecular identities remain represented.

| Exposed CHECK stage | TP | FP | FN | Observed FDP | All-truth recall | Identities |
|---|---:|---:|---:|---:|---:|---:|
| Original full-library NNLS | 125 | 51 | 0 | 28.977273% | 100% | 176 |
| Frozen high-recall screening pool | 121 | 10 | 4 | 7.633588% | 96.8% | 131 |
| Original strict final-score eligibility, before refit | 83 | 2 | 42 | 2.352941% | 66.4% | 85 |
| Descriptive transfer of fixed DEV 5% query cutoff | 117 | 6 | 8 | 4.878049% | 93.6% | 123 |

Pool cutoff `0.5735000428231877` and strict cutoff `0.9424186933811763` are unchanged. The descriptive 5% query uses the existing DEV cutoff `0.7457128634105533` without optimizing CHECK. It is **not the refit pool**. The actual user-authorized second deconvolution will use the unchanged 131-name pool, including all its original aliases; the strict 83-TP eligibility does not cancel this separate amendment. No amended refit outcome exists at this handoff.

The frozen remote exact review passed. The local official review completed all 34 independent loss/KKT checks, then **failed** exact feature-object equality with `FEATURES_CHANGED`; its failure remains in `cases/CHECK/official_local_review_failure.json`. Independent diagnosis found only 5,639 per-block `positive_tolerance` scalar differences: maximum absolute 4.0389678347315804e-28, maximum relative 2.1774624502378105e-16. All 377 S/C inputs, 12,818 deltas, support/positive flags, ordering and rankings are exactly unchanged. Each difference satisfies the fixed-input float64 dot/product bound (largest observed/bound ratio 0.0311393); changed contexts are held-norm dominated. This separate numerical compatibility PASS does not relabel the official exact-check failure. Original remote features and scientific code were not modified.

Independent verification covers exact 34-block membership, all 70 completion-artifact hashes, all 77 downloaded-member hashes, model/threshold/selection seals and scalar probability replay (maximum error 2.2204460492503131e-16). Download receipt: `cases/CHECK/download_review.json`; archive SHA256 `8d2f0ee977b61934a6e890e3a0c0d3f812782240482fe33b5330b5f6d0b7e85d`. Every block array is retained locally and remotely. No repeated KKT/solver/model/test/rho work was performed by this audit.

Selection SHA256 `378dbb19302fdfeb3edfc81c672bd8877478d93dd74c44c918f0420a534ee746`; model seal SHA256 `dd132e777da5451aee5068450ca6b577f31df056ab975257b8640913047e3f3f`; threshold SHA256 `b31d1f2406ea606210d2076f84271fde0a2126aa2373498f8f37f40e8b8d3645`. The transfer query is bound to CHECK query SHA256 `7c230b434702a3aa70e19e2fde65d413796785d27530081433c78498a504c180` and DEV query SHA256 `662946e179328ae60d15279c769d9a5782edd444215de65c5f941829e4474bf4`. Full details are in `cases/CHECK/independent_summary_review.json`.

This is exposed development evidence, not independent FDR validation or a final refit result. Commit/push this specific CHECK and its discrepancy record before the separately frozen user-amendment pool refit. No automatic new threshold or model variant is adopted.
