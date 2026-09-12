# V2 paired spatial-evidence development pilot

Reviewed outcome: STOP_CURRENT_VERSION

This is a development result from new spatial realizations of reused identities and a fixed target library; it is not independent chemical-error or population-FDR validation. U, gamma and both CAL-selected thresholds remain unchanged.

| Method / EVAL challenge | TP | FP | Raw TP | FDP | TP retention | All-truth recall |
|---|---:|---:|---:|---:|---:|---:|
| global / aggregate | 24 | 0 | 350 | 0.00% | 6.86% | 6.40% |
| global / MILD | 24 | 0 | 123 | 0.00% | 19.51% | 19.20% |
| global / close_neighbor | 0 | 0 | 115 | EMPTY | 0.00% | 0.00% |
| global / relatively_isolated | 0 | 0 | 112 | EMPTY | 0.00% | 0.00% |

global: epsilon=0.020043382997096581; gamma=1e-6; GO=False; strong_success=False.
Raw solver FN=25; additional true identities removed by screening=326; final FN=351.
EVAL status counts: {"REPLACEABLE": 659, "RETAINED": 24, "THRESHOLD_UNRESOLVED": 1}.

| local / aggregate | 16 | 0 | 350 | 0.00% | 4.57% | 4.27% |
| local / MILD | 15 | 0 | 123 | 0.00% | 12.20% | 12.00% |
| local / close_neighbor | 0 | 0 | 115 | EMPTY | 0.00% | 0.00% |
| local / relatively_isolated | 1 | 0 | 112 | 0.00% | 0.89% | 0.80% |

local: epsilon=0.020362849583791913; gamma=1e-6; GO=False; strong_success=False.
Raw solver FN=25; additional true identities removed by screening=334; final FN=359.
EVAL status counts: {"FULL_MODEL_INCOMPATIBLE": 91, "REPLACEABLE": 577, "RETAINED": 16}.

Each challenge must have a nonempty retained set and at least40% TP retention, in addition to the aggregate1% FDP/40% retention requirements. Empty sets and numerical/threshold nonselections remain in the original denominators.
Spatial novelty distributions, exact calibration sources, all candidate records and numerical witnesses are preserved. No result-driven changes, additional runs or cleanup are performed by this postprocessor.
