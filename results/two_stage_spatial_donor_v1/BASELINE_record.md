# BASELINE reduced-library refit: no qualitative membership change

The baseline model and candidate pool were pushed as `d899022` before this refit. All15,837 foreground pixels completed with the original per-pixel KKT check; max bound ratio0.00026510015980389624 (required<=1). Exactly127 selected columns were fitted and mapped back to the original391 positions. All arrays are finite, excluded rows/background are zero, all64 checkpoint blocks and their hashes match the final model arrays, and the original reporting gate/alias aggregation passed the cached review.

| Stage | TP | FP | FN | Observed FDP | All-truth recall |
|---|---:|---:|---:|---:|---:|
| Original full-library NNLS |125|51|0|28.9773%|100%|
| Baseline screening |119|8|6|6.2992%|95.2%|
| After reduced-library NNLS |119|8|6|6.2992%|95.2%|

The refit removes no additional true or false identity. All six lost truths are initial screening losses, and the original solver misses none. TRAIN's5% candidate-pool cutoff does not produce CHECK FDP<=5%. Neither final1%/80% nor5%/80% descriptive endpoint passes. This is a frozen development comparison on an exposed case, not independently calibrated FDR or evidence that all two-stage methods fail.

Final arrays and all blocks remain under `refit/BASELINE`. Final array SHA256: `e00699dbbcc11cd048614388dd08e77d8183a6dc85c3448347143f44b21fa71c`. Completion receipt SHA256: `229c966493c5ede5542388622ac2428fa8fe091750deb0fa5696cdd2e053a144`. All391 candidate and377 molecular records are preserved. The original full-library results, model and rho were not changed or recomputed; no files were deleted. Outputs are already local, so a remote download is not applicable.

After this exact snapshot is independently reviewed and successfully pushed, execute the predeclared AUGMENTED arm under its already sealed model/cutoff. It is the second half of this fixed paired comparison, not an outcome-driven adjustment of the baseline.
