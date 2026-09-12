# V2 bounded storage recovery — 2026-09-12

The user requested immediate execution of the standing review/Git/cleanup policy. At03:14UTC westc data storage was96% used with about2.1GiB free, while root storage had about7.6GiB free. The six new V2 cases added about2.4GiB. All six completed training/evidence records are already preserved in pushed commit3e3ad1879859496a40e75d8d80d703e3ccdb2283. CAL/EVAL LP screening is still active; no final performance decision is available.

The V2 runner's training_complete.json binds every training file, including intermediate checkpoints. The independent case reviewer and resume/novelty checks require those exact bytes. Deleting unique checkpoints would violate existing dependencies. The implementation did not provide a per-case retirement path compatible with the standing cleanup policy; do not repair that by editing frozen manifests or weakening verification during the run.

This plan therefore has two different actions:

1. Preserve36 checkpoint/model files,1431394074 bytes, in /root/v2_retained_runtime_20260912. Each destination copy already matches the original recorded SHA256. After the plan is pushed, atomically replace each original pathname with a symlink to its verified copy. Original paths, all checkpoint bytes and all frozen hashes remain usable. This frees about1.33GiB on the data volume but is relocation, not deletion of unique scientific artifacts.
2. Delete six redundant remote per-case transport archives,182562594 bytes, after verifying their downloaded local copies and pushed case snapshots. Archive receipts/manifests and source evidence remain. This frees about174MiB on the root volume. These archives contain completed training/evidence snapshots, not the active LP outputs.

The exact path/hash/size lists are in plan.json; local_backup_verification.json records all six byte-identical downloaded archives and the pushed source commit. Protected source hashes cover the90 bound artifacts, twelve completion/review records and four design/input seals. No broad historical sweep, model/array deletion, active score-file mutation, threshold change or process interruption is authorized by this plan. No V57/V58/V59 or old CE result is a deletion target.

The transfer destinations are fully copied and verified before any original pathname is replaced. The application rechecks every planned source, destination, archive and protected hash before mutation; symlink replacement is atomic, so readers keep a valid path. Each action receives an incremental receipt. Final verification must confirm all original bound hashes, all36 destination links, deleted archive membership and actual volume free space. A failed check stops the remaining actions; a partial receipt must not be relabelled complete.

This plan is awaiting its own successful Git push. Preparation has removed nothing. See receipt.json for actual execution status after that prerequisite.
