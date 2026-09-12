# Auxiliary indicators: no improvement in the fixed development screen

The predeclared primary extension, adding all four observable auxiliary quantities, retained 92 true and 2 false identities: recall/TP retention 73.6%, observed FDP 2.12766%. The preserved rho/abundance baseline retains 93 true and 1 false: 74.4%, 1.06383%. None of the four predeclared extensions met the 1%/80% point target or improved both recall and false allocation relative to the baseline. This bounded exploration is complete; no outcome-driven second round was run.

| Model | TP | FP | FN | Observed FDP | Recall / TP retention |
|---|---:|---:|---:|---:|---:|
| Preserved rho + abundance baseline | 93 | 1 | 32 | 1.0638% | 74.4% |
| + library separation and competitor allocation | 91 | 2 | 34 | 2.1505% | 72.8% |
| + global/mean coefficient consistency | 93 | 1 | 32 | 1.0638% | 74.4% |
| + spatial concentration | 92 | 2 | 33 | 2.1277% | 73.6% |
| + all four auxiliaries (primary) | 92 | 2 | 33 | 2.1277% | 73.6% |

All methods use identical original identity folds and the same separate-CAL empirical 1% threshold rule. The baseline's original coefficients, thresholds and test predictions are reused exactly, without refitting. All 125 truths are reported by raw NNLS, so raw solver misses are zero and each FN above is a filter-induced loss. Raw false positives are 49 for all methods. All 125 reportable truths remain in recall denominators, and all were actually perturbed in the original full-library 5% case. No missing-library or fresh-case result is claimed.

## Paired changes, not just aggregate counts

Compared with the exact baseline:

- Library/competition adds 2 TP, loses 4 TP, adds 1 FP and removes no FP.
- Global/mean consistency yields the exact same retained set, not merely the same totals.
- Spatial concentration adds 4 TP, loses 5 TP, adds 1 FP and removes no FP.
- All auxiliaries adds 3 TP, loses 4 TP, adds 1 FP and removes no FP.

Spatial-only and all-auxiliary have equal totals but different true-identity memberships. Preserve every feature, test score, retained flag and fold seal; equal totals do not justify collapsing models or discarding records.

## What the auxiliary diagnostics show

Across 174 reported identities, competition-adjusted abundance correlates 0.810 with log abundance. Global/mean coefficient disagreement correlates -0.975 with log rho. These observations support substantial overlap with existing inputs; they do not establish that correlation caused the lack of improvement. Library separation and spatial concentration show broadly overlapping true/false distributions. A feature having different true/false medians by itself does not show incremental value in joint screening.

Feature definitions and their limits are frozen in `docs/SMALL_MISMATCH_AUXILIARY_CONFIDENCE.md`. Pairwise full-library cosine is not a full-cone identity certificate; global versus pixel-mean NNLS disagreement is not measured spectral error; effective pixel count is concentration, not replicate stability. False allocations can share strong spectra or smooth spatial structure. No metadata/class/identity label, true abundance or perturbed target library entered feature construction.

## Provenance and checks

- Preparation commit `834772420f79e220f71cf8186592ee03a7a964a6` was pushed and verified before any augmented classifier fitting.
- Contract SHA256: `3523ddf0c904a97b66dda0fcf86b11ffa9a9362b596a56c6c8d3c3b8975f9f37`.
- Four fixed extensions x five folds = 20 tiny CPU fits, each with unchanged regularization/optimizer settings and a model/cutoff seal before that fold's test predictions. No baseline refit, search, threshold revision or further variants.
- Five analytical tests passed locally and remotely; syntax compilation and CLI help passed. Cached independent review verified source hashes, alias membership, all candidate competition, features via a second formula, TRAIN-only normalization, coefficients/predictions, calibration ties and complete accounting.
- Maximum feature reconstruction difference 1.69e-15 on both hosts. Maximum prediction reconstruction differences 7.77e-16 remote and 2.22e-16 local. Exact baseline and all method/fold counts match.
- All 25 result files were downloaded with matching hashes. Remote root: `/root/small_mismatch_auxiliary_confidence`. Original case and baseline roots are preserved. No NNLS/rho/certificate/GPU work was repeated; no new case or monitoring started and no files were deleted.

The complete records are `prepared/records.json`, `fit/predictions.json`, the 20 `fit/*/fold_*_seal.json` files, `review.json`, `local_review.json`, provenance, execution and transfer receipts. Transport archives remain outside lightweight Git. Git whitespace checks are run before final result preservation.

## Interpretation and stopping

This is explicitly adaptive method development on one previously exposed case, including reusing the earlier test groups. Group separation prevents within-fold identity-label overlap but does not produce new independent evidence. Even a successful point target here would require new-case validation before claims of controlled FDR or deployment. Here the primary and all exploratory extensions failed the point target.

The conclusion is limited: these four fixed auxiliary quantities, used in this fixed logistic extension, did not improve this development screen. It does not prove all auxiliary indicators or confidence strategies impossible. Preserve the simpler existing baseline as the reference; do not promote any of these extensions or retune the exposed folds. Result review, Git preservation and STOP complete this user request. The broader 1% error / 80% recall objective remains open.
