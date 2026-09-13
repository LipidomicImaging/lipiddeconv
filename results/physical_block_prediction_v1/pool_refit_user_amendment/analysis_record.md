# Actual high-recall pool refit: identity membership did not improve

The user requested the actual second deconvolution despite the earlier strict-score recall ceiling. The saved CHECK pool and amendment were pushed as b6fdfea75ffa6c593fac90b2845121f354b910e0; independently verified exact inputs were pushed as 51568621c16bd8cf944c0d42a8cbbe4a834755c2 before execution. All131 selected columns and15837 foreground pixels were refit once using the existing NNLS/KKT helper. Actual helper runtime was156.08 seconds. No new classifier, rho, observation or threshold was introduced.

| Same exposed CHECK case | TP | FP | FN | FDP | All-truth recall |
|---|---:|---:|---:|---:|---:|
| Original full-library NNLS | 125 | 51 | 0 | 28.9773% | 100% |
| Frozen high-recall pool | 121 | 10 | 4 | 7.6336% | 96.8% |
| Actual second NNLS, pool-only primary endpoint | 121 | 10 | 4 | 7.6336% | 96.8% |
| Secondary intersection with the unchanged old strict score | 83 | 2 | 42 | 2.3529% | 66.4% |

The second NNLS removed neither true nor false identities. Screening had already excluded four truths: PE18:2_22:4, PE17:0_20:2, PG16:0_22:5 and PG20:2_18:3 (exact spaced names are preserved in the molecular records). The actual refit did not meet either the strict1%/80% endpoint or the secondary5%/80% endpoint. Foreground relative reconstruction residual is0.944803%; small residual does not certify identities.

The user's separate5% question has a different answer: cached DEV scores under FDP<=5% retain112TP/5FP, FDP4.2735%, recall89.6%. That exact DEV query cutoff transferred to CHECK retains117TP/6FP, FDP4.8780%, recall93.6%. This describes the original evidence's screening point; it was not substituted for the131-member refit pool and is not a measured refit of the123-member selection.

All ten false identities and actual model contributions are in false_identity_analysis.md/.json. Five false coefficients increase and five decrease, but all stay above the original0.001 reporting gate. PG14:0_22:3 increases2.949-fold. Three false identities have rho=0, S<0 and C=0 yet pass the additive polynomial classifier; seven have positive physical prediction evidence. The rho-square term permits low/zero rho to contribute positively. No particular competing donor or causal interference source is established by this descriptive analysis, and the uncalibrated logistic score is not a probability of correct identity.

The request to inspect completed pixels arrived as computation finished: all15837 cached pixels were already available. The resulting partial_015837/preview.json explicitly says ALL_PIXELS_CACHED_PRE_COMPLETION_REVIEW and was not relabelled as an earlier partial result. Its cache counts match the final reviewed result.

Remote and local cache reviews passed without solving again; all64 block arrays, final learned arrays, original source/runtime/code bindings,391 candidate and377 molecular records, original any-alias gate, reconstruction and final counts were verified. The independent direct-array identity recount passed. All144 remote archive members were downloaded with matching hashes, archive SHA2567102838b0f9e77745241fcec2c6d847d4937f5c7de1ea6932890ab9bf11869ba. All final and block arrays remain on both documented storage locations, excluded from lightweight Git. Earlier DEV/CHECK physical-feature exact local tau discrepancies and their separate compatibility reviews remain preserved; no scientific file was changed to hide them.

This bounded user-amended experiment is complete. It is exposed development evidence, not independent real-MSI FDR validation or a test of new relative-abundance families. Preserve this final case and analysis in Git, update project state and stop; no automatic next variant or cleanup.
