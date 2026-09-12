# V59 partial result recovery — 2026-09-12

Seven of eighteen formal cases are complete: D0 CAL/HOLD R1–R3 and D1 CAL_R1. The live D1 CAL_R2 process is computing serial rho after training. This is a partial snapshot, not final cross-library validation. Previously only progress was in Git; this snapshot closes the handoff gap for these seven completed cases.

The original design, implementation, training contract, oracle/sentinel evidence, domain CAL seal and seven result seals were verified using the existing read-only runner checks. Cached learned arrays pass recorded SHA256, design shape and finiteness checks; training histories and normal completion pass. Runtime observation hashes match frozen dataset signatures. Independent CSV accounting reproduces each raw molecular TP/FP/FN. The91 compact source files match their downloaded hashes. Exact checks and retained model/checkpoint hashes are in review.json, source_manifest.json and local_transfer_review.json. The audit source is preserved; its initial attempt stopped before reviewing cases because the runner's explicit dependency initializer had not been called. The corrected standalone reader calls that initializer; no experiment code, output or gate was changed.

D0 has a verified CAL-only threshold seal. Applying its existing rho FDR1 threshold to its three HOLD cases gives:

| HOLD case | Raw TP | Raw FP | Retained TP | Retained FP | Solver FN | Filter-induced true loss |
|---|---:|---:|---:|---:|---:|---:|
| R1 | 121 | 183 | 121 | 0 | 4 | 0 |
| R2 | 120 | 195 | 120 | 0 | 5 | 0 |
| R3 | 120 | 202 | 120 | 0 | 5 | 0 |

Pooled D0 HOLD:361 TP/0 FP, observed FDP0%, TP retention100%, all-truth recall361/375=96.27%. These are CLEAN, reused-identity spatial contexts; zero observed false positives is not proof of zero population FDR, and D0 alone does not establish cross-library success. Thresholds are unchanged. D1 filtering is deferred until its own CAL seal exists. No HOLD-driven tuning or final18-case claim.

D0 report timing indicates roughly3–4h serial rho per case, and the first D1 formal interval was about8h. Instantaneously idle GPU does not mean the sequential job is finished. No reliable D2 timing or complete-experiment deadline is available from this snapshot.

Final18-case aggregation, independent reconstruction-residual recomputation and historical raw-asset re-audit are deferred. Final arrays/models/checkpoints remain at the explicit source directories in review.json; this compact transfer is not their backup. No data were deleted, no fit or rho was rerun, and no live process was interrupted.
