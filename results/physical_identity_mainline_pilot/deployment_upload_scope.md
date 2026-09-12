# Mainline pilot upload scope

Destination: user-designated westc:55786, /root/physical_identity_mainline_v1/source.
Archive: 182138 bytes; SHA256 6981a478530f5905a6a3dbffc5b8363aea6ed8086d4f2821271b572faac385ff.
Source commit: 06914c6123e9b37bcd266f6067bacb0f32401898.

25 files, plus a generated transfer manifest inside the archive. No passwords, model weights, production X/B arrays, raw MSI, or full upstream spectral library are included.

| File | Bytes |
|---|---:|
| analysis/freeze_physical_perturbation_contract_v1.py | 8947 |
| analysis/physical_identity_confidence_core.py | 29076 |
| analysis/physical_identity_pilot_design.py | 13345 |
| analysis/review_physical_identity_mainline_pilot.py | 40633 |
| analysis/run_ce_uncertainty_identity_pilot.py | 29481 |
| analysis/run_missing_library_challenge.py | 17176 |
| analysis/run_physical_identity_mainline_pilot.py | 35446 |
| analysis/test_physical_identity_confidence_core.py | 11610 |
| analysis/test_physical_identity_mainline_pilot.py | 13984 |
| docs/PHYSICAL_IDENTITY_MAINLINE_PILOT_V1.md | 7644 |
| docs/PHYSICAL_PERTURBATION_CONTRACT_V1.md | 7520 |
| results/ce133_joint_variation_audit/exact_support_joint_statistics.json | 181409 |
| results/ce133_joint_variation_audit/observed_joint_changes.json | 325450 |
| results/ce133_joint_variation_audit/provenance.json | 2862 |
| results/ce133_uncertainty_v1_ready/component_fractions.npz | 15250 |
| results/ce133_uncertainty_v1_ready/components.json | 143625 |
| results/ce133_uncertainty_v1_ready/contract.json | 1055 |
| results/ce133_uncertainty_v1_ready/provenance.json | 5478 |
| results/physical_perturbation_contract_v1/.gitattributes | 72 |
| results/physical_perturbation_contract_v1/candidate_mapping.json | 238136 |
| results/physical_perturbation_contract_v1/contract.json | 8150 |
| results/physical_perturbation_contract_v1/joint_patterns.json | 59952 |
| results/physical_perturbation_contract_v1/seal.json | 2373 |
| results/physical_perturbation_contract_v1/verification.json | 620 |
| results/rho_mismatch_mechanism_ce_audit/ce_identity_records.json | 241705 |
