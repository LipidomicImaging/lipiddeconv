# Frozen missing-library execution

Status: READY_FOR_GPU_EXECUTION; no omission fit launched during preparation.

Fingerprint: b51137f3c6f12cf9bdf0e3e9f1c00c1fff53f9e4d62dc1c54de0b04a7910d9ef.

Six new fits: close_neighbor and relatively_isolated arms, each on HOLD_R1/R2/R3_K125. Three complete-library controls are reused under frozen artifact hashes. The old V57/V58/V59 directories and production implementations are unchanged.

CPU validation passed all six reduced-library forwards (1084 channels,386 candidates); candidate metadata/order and original/reduced maps agree. Reconstructed B and X_true match the audited parent cases, with unchanged global scalar. Production normalization and channel-weight construction pass. Synthetic accounting tests verify125 truths,120 recovered TP,one spurious FP,five omitted FN; filtering all reports correctly adds120 true losses. Cross-design and unbound checkpoint tests reject old caches. Separate check invocation reproduced the exact frozen fingerprint after reading the saved design/seal. Downloaded JSON artifacts match remote SHA256. JSON files here disable Git newline conversion to preserve seals.

Scope: CPU inference on a tiny patch is a shape/finite check, not a learned-process sentinel or scientific result. X_true provenance and checkpoint snapshot limitations from control_reconstruction_audit.json remain applicable. A high residual under omission is an outcome, not a failure gate. Thresholds are the original V57 CAL thresholds; no missing-library HOLD calibration.

Remote host: connect.westc.seetacloud.com:55786. These commands operate on already prepared remote files. Execute the run command only when the intended GPU is released. No automatic queue was created.

```bash
python /root/run_missing_library_challenge.py check \
  --root /root/autodl-tmp/lipiddeconv \
  --asset-root /root/autodl-tmp/decon-lipid \
  --assets /root/autodl-tmp/lipiddeconv/results/computational_closure/missing_library \
  --output /root/autodl-tmp/lipiddeconv/results/missing_library_challenge_k125

nohup python -u /root/run_missing_library_challenge.py run \
  --root /root/autodl-tmp/lipiddeconv \
  --asset-root /root/autodl-tmp/decon-lipid \
  --assets /root/autodl-tmp/lipiddeconv/results/computational_closure/missing_library \
  --output /root/autodl-tmp/lipiddeconv/results/missing_library_challenge_k125 \
  --device cuda:0 --rho-workers 1 \
  > /root/missing_library_challenge_k125.log 2>&1 < /dev/null &
```

The run command revalidates the manifest, binds each new output directory before training, runs the unchanged production driver and computes original-denominator individual identity metrics. It cannot load full-library control checkpoints into reduced fits. Completed outputs require matching artifact hashes; a recorded failure requires review. Large arrays/checkpoints stay remote and are not part of this Git payload.
