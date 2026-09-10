# V57 HOLD stratified review

Read-only reanalysis of existing molecular records, whose SHA256 matches the earlier remote provenance audit. No solver/rho rerun or tuning.

122 solver-miss contexts involve30 of175 distinct truth identities. The five most frequently missed identities account for48/122 misses (39.3%). The ranking is by repeated-context count; identities appear in different numbers of nested-K contexts, so this is not a frequency-adjusted causal ranking. The accompanying table includes denominators and per-identity miss fractions.

By K50/75/100/125/150/175, solver FN counts are0/2/14/28/32/46, with truth-context denominators250/375/500/625/750/875. Recall is100%/99.47%/97.20%/95.52%/95.73%/94.74%. There is an overall complexity-associated decrease, not a strictly monotone decline or a causal isolation of K from fixed total-signal dilution.

Across R1-R5, FN counts26/20/26/24/26; recall spans96.15%-97.04%. FDR under frozen rho CAL-FDR5 spans2.25%-4.38% across replicates and1.57%-4.93% across K. Both frozen rho thresholds add zero true losses in every stratum. There is no filtered-true group on which to estimate ambiguity enrichment. Class-specific tables are descriptive uses of a global threshold, not class-specific calibration guarantees. Counts share identities, templates and nested K; no independent-row p-values were used.

Outputs: by_K_replicate_class.csv, identity_miss_recurrence.csv and summary.json. Further interpretation should compare V58/V59 outcomes when complete, without redesigning frozen conditions.
