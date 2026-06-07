# Dubai collision severity model — report

Binary RandomForest predicting **minor vs severe** from incident type, hour, day-of-week, month, and location.

## Headline
- **ROC-AUC: 0.867**  ·  severe-class recall: **0.815**
- Accuracy 0.8 (majority-class baseline 0.894) — we trade accuracy for severe recall via `class_weight='balanced'`, so accuracy is *below* baseline by design. Lead with AUC/recall.
- Train/test: 304,226 / 76,057 (severe rate 10.6%)

## Per-class
| class | precision | recall | f1 |
|---|---|---|---|
| minor | 0.973 | 0.798 | 0.877 |
| severe | 0.324 | 0.815 | 0.463 |

## Confusion matrix (test)
| | pred minor | pred severe |
|---|---|---|
| actual minor | 54,259 | 13,734 |
| actual severe | 1,491 | 6,573 |

## Feature importances
- type_code: 0.731
- lng: 0.088
- lat: 0.081
- hour: 0.049
- month: 0.029
- day_of_week: 0.022

## Honest framing (viva)
- **Descriptive, not predictive-for-routing.** Incident type dominates importance, but type is unknown before a crash — so this explains *what makes crashes severe*, it does not score a future route. Geo+time alone is much weaker (AUC ~0.62).
- Non-circular: the severity tag is stripped from the type label before use, and the same type appears as both minor and severe.