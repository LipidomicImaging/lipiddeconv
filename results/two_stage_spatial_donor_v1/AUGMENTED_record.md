# AUGMENTED reduced-library refit: one additional true identity, false set unchanged

Both model/candidate-pool seals were pushed as `d899022` before either refit. The completed BASELINE result and its independent review were pushed as `f1dff72b811a8f7d386f24a0119ee08045a96781` before starting this predeclared second arm. No feature, model, threshold, source, solver or reporting parameter changed between arms.

All15,837 foreground pixels completed against the128 selected columns using the original per-pixel KKT check. Max bound ratio0.0003347977349724175 (required<=1). The complete391-index arrays are finite; excluded rows and background remain zero. All64 checkpoint blocks, source/selection bindings and final candidate-first reporting passed the cached review.

| Stage | TP | FP | FN | Observed FDP | All-truth recall |
|---|---:|---:|---:|---:|---:|
| Original full-library NNLS |125|51|0|28.9773%|100%|
| Augmented screening |120|8|5|6.25%|96%|
| After reduced-library NNLS |120|8|5|6.25%|96%|

Refitting does not remove any additional identity. Relative to the baseline final set, exactly one true identity, `PE O-18:1_22:6`, is added; no identity is removed, and the eight false identities are exactly the same. This name is an accounting record only, not a feature or blacklist entry. All five false negatives are screening losses; original solver misses and additional refit losses are zero.

Neither final1%/80% nor5%/80% descriptive endpoint passes. The two added spatial descriptors recover one truth in this fixed development test, but do not resolve the false allocations. The result does not establish that all spatial evidence or two-stage strategies fail. Both source cases were exposed and share identity/spatial context; no independent or real-MSI FDR claim is made.

Final arrays and all blocks remain in `refit/AUGMENTED`. Final array SHA256: `6da51bb323516b1f1ba99e4a8f5023bb9efe431358301582a0de17bef8ad2aee`; completion receipt SHA256: `fdbc9ddb12825d9bb9e98599e7f906eebee9056c632227310e9ab07e4dda5f60`. All391 candidate and377 molecular records are preserved. Outputs are local; no remote download is applicable. No original source or result was modified, and no arrays/checkpoints were deleted.
