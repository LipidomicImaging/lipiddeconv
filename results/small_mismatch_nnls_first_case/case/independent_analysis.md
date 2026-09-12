# One-case 5% mismatch NNLS/rho review

Independent cached process/accounting review: PASS. Developmental first-case screen: **GO**.

Truth identities: 125; reportable truths: 125; actually perturbed truths: 125. Raw solver TP/FP/FN: 125/49/0. Selected TP/FP/FN: 60/0/65; filter-induced true losses: 65.

Selected threshold: 7.8551487512534912e-07; empirical FDP: 0.0; TP retention: 0.48; all-truth recall: 0.48. Complete tied-score curve and fixed rho=1e-3 result independently agree.

This is one previously exposed spatial development case under a user-specified 5% model. The selected threshold is not independent calibration and does not establish real-data FDR control. No further case is authorized by a GO result.

Stored rho formulas were recomputed from q_star/q_deleted without solving NNLS again. The receipt describes floating subtraction scales; these do not bound optimizer error or alter selection. Pre-cast KKT records and block membership were checked; unavailable pre-cast float64 solutions were not reconstructed.
