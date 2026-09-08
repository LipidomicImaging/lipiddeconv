\# NEXT TASK — v48 All-Reported-Candidate Identity Validation



Status: READY



\## Goal



Using existing cached v48 results only, determine whether rho\_zero can identify a high-confidence subset among all candidates that would actually be reported.



\## Input



Primary cached table:



v48\_pilot\_lipid\_observations.csv



Expected:



48 × 391 = 18,768 rows



Use cached fields only.



\## Do not



Do NOT:



\- rerun simulation

\- regenerate X\_true

\- regenerate B\_sim

\- rerun production ISTA

\- rerun NNLS

\- rerun profile

\- rerun rho\_zero

\- rerun necessity

\- rerun cone

\- recompute groups

\- add features

\- train a model

\- optimize thresholds

\- start v49



\## Reporting gates



Use exactly:



\- X\_hat > 1e-4

\- X\_hat > 1e-3

\- X\_hat > 1e-2



For each gate:



truth\_identity = 1 if X\_true > 0

truth\_identity = 0 if X\_true == 0



\## Baseline statistics



For each gate report:



\- n\_reported

\- n\_true\_reported

\- n\_false\_reported

\- identity\_precision

\- false\_identity\_fraction

\- mass\_weighted\_false\_fraction

\- true\_identity\_recall



\## Primary certificate



Feature:



rho\_zero



Direction:



higher = more trustworthy



For each reporting gate, sort reported candidates by rho\_zero descending.



Evaluate fixed coverage:



\- 5%

\- 10%

\- 20%

\- 40%

\- 60%

\- 80%

\- 100%



At each coverage report:



\- n retained

\- identity precision

\- false identity fraction

\- mass-weighted false fraction

\- true identity recall



Do not select a best coverage.



\## Stratified analysis



For rho\_zero top10% and top20% repeat within:



K:

13 / 25 / 55 / 103



residual:

0 / 0.1



selection:

random / hard\_competition



Report:



\- n

\- identity precision

\- false identity fraction

\- mass-weighted false fraction

\- true identity recall



\## Secondary comparison



Pooled AUROC only for:



\- necessity\_signal

\- profile\_relative\_width

\- global\_fragment\_cone\_residual



Fixed directions:



\- necessity\_signal: higher trustworthy

\- profile\_relative\_width: lower trustworthy

\- global cone: higher trustworthy



Do not combine features.



\## Outputs



Generate only:



\- v48\_identity\_allreported\_summary.csv

\- v48\_identity\_allreported\_riskcoverage.csv

\- v48\_identity\_allreported\_feature\_summary.csv

\- v48\_identity\_allreported\_report.html

\- v48\_identity\_allreported\_report.json



\## Efficiency



Expected workflow:



read cached table

→ filter

→ groupby/statistics

→ output

→ STOP



No optimization.

No GPU.

No raw MSI.

No historical project scan.



\## Completion



When complete:



1\. update CURRENT\_STATE.md

2\. mark this task COMPLETED

3\. commit and push

4\. STOP



Do not start v49.

