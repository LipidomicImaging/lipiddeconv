# Different true-composition screening completed

The corrected model was frozen and successfully pushed as 704e05fa35e39b15735fd5c600eaed9f8510db0f before the new case. The existing seed7501 full-library input has125 true molecular identities disjoint from the old CHECK truths,391 candidate columns,377 molecular names and15837 foreground pixels. Both source abundance ranges span approximately4-fold and median observation norm is0.60367834. This is a constructed physical-ion CV5% perturbation case, not real acquisition or a newly designed equal-intensity/strength-swap family. Different composition and perturbation change together. Its old unused HOLD role is retired; these outcomes belong to this development experiment only.

| Fixed DEV policy | Old model TP/FP | Old FDP / recall | Corrected TP/FP | Corrected FDP / recall |
|---|---:|---:|---:|---:|
| High-recall pool |120/9|6.9767% /96.0%|121/9|6.9231% /96.8%|
| DEV <=5% cut |118/7|5.6000% /94.4%|113/5|4.2373% /90.4%|
| DEV <=1% cut |86/1|1.1494% /68.8%|67/0|0% /53.6%|

Raw full-library NNLS reports124TP/56FP/1FN (FDP31.1111%, recall99.2%). PE O-18:1_20:3 is already missing at the original reporting gate. Corrected pool filtering removes three additional truths: PE O-18:2_20:3, PE O-18:2_22:4, PG16:0_22:6 (exact spaced names in JSON). All recalls retain the full125 denominator; pool TP retention is121/124=97.5806%, distinct from recall96.8%.

The nine corrected-pool false identities share no names with the old CHECK's ten false identities. Five have positiveS, four negativeS; PE O-18:2_20:2 hasrho0/S<0/C0 but still scores0.712673 above the frozen0.629929 pool cut. Eliminating the rho-square reward therefore does not eliminate compensation by abundance and intercept. Frozen DEV5% selection retains five false identities, including PE16:0_22:3 withS<0. No new cutoff or model change follows these observations.

NNLS/rho took1377.75s and34-block prediction1216.77s, running concurrently with fourCPU workers each. Remote and local completed-cache reviews passed; independent source/array/alias/score/threshold/count review passed without fitting or rho recomputation. The initial-case archive contains412files,100052918bytes, SHA256 20210a28398b11785f2167541c0810b360ce4afc06fac0f477ef4471b666a2eb. All411 manifest member hashes match; all candidate/molecular records, rho receipts,64NNLS blocks,34physical blocks and learned arrays are preserved locally and remotely. Nothing was deleted.

Next, save and push this exact completed case, then run the declared corrected130-name high-recall pool refit once. The118-name DEV5% selection is descriptive and is not a separate refit input. No refit result exists at this snapshot.
