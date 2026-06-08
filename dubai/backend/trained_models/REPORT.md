# Dubai collision severity model — report

Binary RandomForest predicting **minor vs severe** from incident type, hour, day-of-week, month, and location.

## Headline
- **ROC-AUC: 0.865**  ·  severe-class recall: **0.806**
- Accuracy 0.802 (majority-class baseline 0.893) — we trade accuracy for severe recall via `class_weight='balanced'`, so accuracy is *below* baseline by design. Lead with AUC/recall.
- Train/test: 294,876 / 73,720 (severe rate 10.7%)

## Per-class
| class | precision | recall | f1 |
|---|---|---|---|
| minor | 0.972 | 0.802 | 0.878 |
| severe | 0.328 | 0.806 | 0.466 |

## Confusion matrix (test)
| | pred minor | pred severe |
|---|---|---|
| actual minor | 52,739 | 13,061 |
| actual severe | 1,540 | 6,380 |

## Feature importances
- type_code: 0.73
- lng: 0.089
- lat: 0.08
- hour: 0.048
- month: 0.029
- day_of_week: 0.023

## Honest framing (viva)
- **Descriptive, not predictive-for-routing.** Incident type dominates importance, but type is unknown before a crash — so this explains *what makes crashes severe*, it does not score a future route. Geo+time alone is much weaker (AUC ~0.62).
- Non-circular: the severity tag is stripped from the type label before use, and the same type appears as both minor and severe.