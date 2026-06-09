# Dubai collision severity model — report

Binary RandomForest predicting **minor vs severe** from incident type, hour, day-of-week, month, and location.

## Headline
- **ROC-AUC: 0.869**  ·  severe-class recall: **0.834**
- Accuracy 0.786 (majority-class baseline 0.879) — we trade accuracy for severe recall via `class_weight='balanced'`, so accuracy is *below* baseline by design. Lead with AUC/recall.
- Train/test: 306,476 / 76,619 (severe rate 12.1%)

## Per-class
| class | precision | recall | f1 |
|---|---|---|---|
| minor | 0.972 | 0.779 | 0.865 |
| severe | 0.342 | 0.834 | 0.485 |

## Confusion matrix (test)
| | pred minor | pred severe |
|---|---|---|
| actual minor | 52,491 | 14,860 |
| actual severe | 1,534 | 7,734 |

## Feature importances
- type_code: 0.744
- lng: 0.085
- lat: 0.076
- hour: 0.048
- month: 0.028
- day_of_week: 0.02

## Honest framing (viva)
- **Descriptive, not predictive-for-routing.** Incident type dominates importance, but type is unknown before a crash — so this explains *what makes crashes severe*, it does not score a future route. Geo+time alone is much weaker (AUC ~0.62).
- Non-circular: the severity tag is stripped from the type label before use, and the same type appears as both minor and severe.