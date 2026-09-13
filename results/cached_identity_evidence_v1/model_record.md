# Fixed cached-feature model comparison

The pre-extraction/fit snapshot74e6c05be9835e55c6a2506768ec1cbd62648872 was pushed and its remote hash verified. All three cases'34block cached feature extractions completed; independent four-loss direct-contribution, all-alias, support, delta, row and source/hash reviews pass. No new observations, NNLS, rho or physical-block fits were used for these features.14 analytic and model-interface tests passed before execution.

Exactly the three predeclared DEV fits were performed. TRIM, DIRECT and JOINT all produce the same DEV counts as the frozen baseline:95%-recall pool119TP9FP; DEV5%107TP5FP (FDP4.4643%, recall85.6%); DEV1%65TP0FP (recall52%). The tie rule chooses TRIM because it adds the fewest features. This is a mechanical selection, not evidence of improvement.

Every newly added feature coefficient is exactly0 under the prespecified nonnegative constraints. Their objective gradients at the boundary are positive, so increasing these coefficients would locally worsen the fixed training objective. Small changes in base coefficients reflect optimizer termination, not evidence from the new metrics. Maximum projected gradients are3.40e-5,1.01e-5,1.44e-4 respectively; all fixed L-BFGS-B runs stopped successfully on the predeclared objective-reduction criterion. No restart, bound relaxation or negative-weight retry follows these results.

This does not prove the raw features contain no information under any method. It does show that these proposed features have not added useful positive evidence within this frozen model. CHECK scoring has not occurred at this snapshot. Freeze/push the three models and DEV-only winner before transferring their cuts to the two exposed CHECK datasets. The planned actual refit input is the winner's DEV5% set, not the previous high-recall pool.

All feature arrays and model outputs are retained locally, with a734127-byte backup archive SHA256465c1f29e22e9d9b76c9acefa06b162ba77b774dd3216f52c22099d54bc47775; per-file hashes are in feature_model_backup.json. The source full arrays and prior102 block fits remain at their documented local/remote locations. No deletion occurred.
