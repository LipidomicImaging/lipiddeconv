# v53 Template-Truth Identity Calibration Pilot

## Purpose

Test whether rho_zero can distinguish correct versus false molecular identities after fresh learned-ISTA deconvolution of a clean, known-truth synthetic dataset with empirical spatial/abundance templates.

## Truth construction

76 abundance maps from the real production X_hat were reused only as empirical spatial/amplitude templates. Their original lipid identities were discarded and remapped to 76 unique known synthetic molecular identities.

The clean synthetic observation was generated as:

B_sim = A_solver @ X_true

No empirical residual or Gaussian noise was added.

## Oracle result

The exact full-library NNLS oracle recovered all 76 synthetic truth identities:

- TP: 76
- FP: 0
- FN: 0
- Precision: 1.0
- Recall: 1.0
- Reconstruction relative residual: 5.486875133180915e-07

Thus the clean construction is mathematically recoverable by the full frozen library.

## Fresh learned-solver result

At the foreground-mean reporting gate X_hat > 1e-3:

- Reported candidates: 103
- Molecular TP: 51
- Molecular FP: 45
- Molecular FN: 25
- Molecular precision: 0.53125
- Molecular FDR: 0.46875
- Molecular recall: 0.6710526315789473
- Reconstruction relative residual: 0.023598041385412216

Candidate-level:

- TP: 51
- FP: 52
- FN: 25
- Precision: 0.49514563106796117
- FDR: 0.5048543689320388

The difference between candidate-level and molecular-level FP reflects same-lipid alternative candidate/adduct assignments, which are not counted as false molecular identities.

## rho_zero result

For reported true molecular identities:

- n = 58 candidate records
- median rho_zero = 5.943263004138303e-07
- maximum rho_zero = 0.001576425977926672

For reported false molecular identities:

- n = 45
- median rho_zero = 9.473988848455898e-19
- maximum rho_zero = 6.030171939676753e-15

A descriptive pilot operating point was:

rho_zero >= 4.167353732004888e-09

which retained:

- n = 51
- Molecular TP: 51
- Molecular FP: 0
- Precision: 1.0
- FDR: 0.0
- Recall: 0.6710526315789473
- Coverage fraction: 0.49514563106796117

This threshold is descriptive for this pilot only and is NOT a frozen deployment threshold.

## Training

- Fresh LipidENNet
- No real-data checkpoint loaded
- X_true was not used in training or stopping
- Final epoch: 5000
- Stop reason: max_epochs
- Wall time: 889.5411412715912 s

The 5000-epoch endpoint should not be interpreted as evidence that 5000 epochs are necessary; the run reached the configured maximum rather than an identity-specific convergence criterion.

## Interpretation

1. The clean synthetic problem is exactly recoverable by NNLS.
2. The fresh learned solver nevertheless generates substantial molecular false identities.
3. In this pilot, rho_zero provides very strong separation between reported true and false molecular identities.
4. All 45 false molecular identities have rho_zero essentially at numerical zero.
5. A positive rho_zero threshold removes every molecular FP in this pilot while retaining all 51 true molecular identities recovered by the learned solver.
6. This provides strong proof-of-concept evidence for rho_zero as the primary molecular-identity certificate.

## Limitations

- Only one synthetic truth remapping was tested.
- The observation is clean B = A_solver @ X_true.
- No empirical residual/model mismatch was added.
- Spatial/amplitude templates originate from production X_hat and are not biological ground truth.
- Global synthetic truth identity count is 76.
- True per-pixel K has not yet been summarized for this experiment.
- Training reached max_epochs=5000.
- No final rho_zero deployment threshold is calibrated here.

## Validity status

VALID_MAINLINE

Scope: proof-of-concept / pilot evidence only. Not valid for freezing the final rho_zero threshold.

## Decision

Proceed to replicated ground-truth calibration across complexity and empirical residual conditions.

Final rho_zero threshold frozen: NO.
