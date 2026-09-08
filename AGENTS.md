\# LD Project — Codex Working Rules



This repository is the lightweight working repository for the LD lipid deconvolution project.



Default rule:



REUSE EXISTING CODE AND RESULTS FIRST.

DO NOT REBUILD EXISTING PIPELINES UNLESS A REQUIRED COMPONENT IS PROVEN MISSING.



\## Execution rules



1\. Always read:

&#x20;  - AGENTS.md

&#x20;  - docs/CURRENT\_STATE.md

&#x20;  - docs/NEXT\_TASK.md



2\. Use only the files explicitly required by CURRENT\_STATE / NEXT\_TASK.



3\. Do not reconstruct old v17-v48 project history.



4\. Do not scan the archived heavy repository unless NEXT\_TASK explicitly requires it.



5\. For simple statistics:

&#x20;  existing CSV/Parquet/JSON

&#x20;  → pandas/scipy

&#x20;  → requested outputs

&#x20;  → STOP.



6\. Do not rerun simulation, ISTA, profile, certificate, or raw MSI processing for ordinary downstream analysis.



7\. Frozen definitions must not be changed because results are weak:

&#x20;  - production ISTA

&#x20;  - rho\_zero

&#x20;  - necessity

&#x20;  - profile L/U

&#x20;  - rho=1e-3

&#x20;  - d\_frag=0.02 competition groups

&#x20;  - observation threshold



8\. Scientific priority:

&#x20;  1. identity correctness / false allocation

&#x20;  2. group-level rescue

&#x20;  3. quantitative abundance accuracy



9\. Never use:

&#x20;  bad result → tune parameters → rerun.



10\. When NEXT\_TASK is complete:

&#x20;   - write only requested outputs

&#x20;   - update CURRENT\_STATE.md

&#x20;   - mark NEXT\_TASK completed

&#x20;   - commit and push

&#x20;   - STOP.

