# Finite-reference CAL prototype: stop this version

All211 NNLS-reported molecular identities evaluated,125 truths/86 false reports. Fixed3 references per candidate,1173 competing columns; reference randomness independent of observation draws. Removing an identity removes all of its reference variants. No target spectra/truth enters scoring, no original pipeline or rho change, no GPU training.

At retrospective CAL empirical5% FDR: original rho28 TP/1 FP (22.4% recall), abundance47/1 (37.6%), flexible-reference score26/1 (20.8%). At1%: new score6/0 (4.8% recall), rho20/0 (16%), abundance45/0 (36%). No solver misses in this reporting universe; new5% rule loses99 of125 raw true identities. Scores are tied as declared, no candidate exclusions.

Expanded dictionary lowers full-fit SSE to22.68% of the original (77.32% reduction), yet confidence separation does not improve. New AUROC0.6557 versus rho0.6446 and abundance0.8412; AP0.7653 versus0.7694/0.8934. Better reconstruction does not establish better identity discrimination.

Decision STOP_THIS_VERSION_NO_PRESPECIFIED_RECALL_GAIN. Do not add reference variants, change their distribution or tune cutoffs to salvage this version. The known perturbation family and already exposed single CAL case do not establish real-data robustness or independent FDR control. This rejects the tested finite3-reference scheme, not all models of library error.

Input hashes, exact case audit (derived residual tolerance1e-12 only), deterministic references, original single-reference singleton parity, complete same-identity deletion and finite/nonnegative/KKT checks pass. Local output/source/immutable-runner hashes and exact shared reporting universe pass. Original plus independent reference dictionary is retained here. Runtime365 seconds; no GPU or HOLD. Initial protocol/code pushed as707c1e2 while running and before outcomes inspected. Archive SHA2562e61b9c6983191b4b164bf01ff255424f267fd4c7b920c9419d4be04f3ed6c88. No deletion.
