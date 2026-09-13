# Monotone score correction, one DEV fit

The pre-execution snapshot 43030116a14fc49e7d2b2da81b088822d752740a was successfully pushed and its remote hash verified before the sole fit. The corrected five-term model removes rho-square, abundance*rho and the zero indicator; rho/S/C coefficients are constrained nonnegative. Original DEV normalization and input features are unchanged. One L-BFGS-B fit converged with maximum projected gradient 7.302e-6 (the fixed objective-reduction stop criterion was reached). Independent scalar probability, normalization, objective/gradient and threshold replay pass in both environments, without refitting.

Model seal SHA256: d600620cddd3027166ab7743b8fe2f48430211fac419c36e5280185de2da8fc2. The 13-file model/comparison archive SHA256 is 63065a11ecb43fa14f1e5e53cbeaf26550fbe3fa04ff42607bb633352ac15c90; all downloaded member hashes match.

Actual coefficients on [a,a²,r,s,c] are [2.29293497,-0.84590173,0,0.58866207,0.05809418], intercept2.43016350. The constrained fit puts rho exactly at its zero lower bound: rho is recorded and used by the old reference model, but contributes nothing to this corrected score. This is a result, not a further feature removal or retry.

| Case and fixed DEV policy | Old TP/FP | Corrected TP/FP | Corrected FDP | Corrected all-truth recall |
|---|---:|---:|---:|---:|
| DEV 95% recall pool |119/8|119/9|7.0313%|95.2%|
| DEV <=5% descriptive cut |112/5|107/5|4.4643%|85.6%|
| DEV <=1% descriptive cut |83/0|65/0|0%|52.0%|
| Exposed old CHECK, transferred pool |121/10|117/7|5.6452%|93.6%|
| Exposed old CHECK, transferred DEV5% cut |117/6|111/2|1.7699%|88.8%|
| Exposed old CHECK, transferred DEV1% cut |83/2|71/1|1.3889%|56.8%|

Corrected fixed cuts: pool0.6299285665027182, DEV5%0.8050711504567813, DEV1%0.958782322096608. Old and corrected models each use their own DEV-selected cut under identical policies; numerical cut values are not comparable.

The old CHECK pool loses exactly the three zero-rho/S<0/C0 false identities (PG14:0_22:3, PEO-18:1_20:2, PE17:1_20:1; exact spaced names in JSON) and four true identities: PA18:1_24:5, PA22:4_22:6, PE18:1_20:3, PG16:1_20:4. No identity is added. Seven positive-evidence false identities remain. This removes the structural rho reward but does not solve the recall/FDP tradeoff. These are first-screen descriptive development results; no corrected old-CHECK refit was performed. The new-composition inputs are prepared, but no learned outcomes exist yet. Freeze and push this model before its computation; do not select another model/cut from the new case outcomes.
