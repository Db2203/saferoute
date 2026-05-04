# temporal random forest — evaluation

Trained on London STATS19 collisions (2020-2024). 80/20 stratified split, 98,860 training rows, 24,716 test rows.

Hyperparameters: `n_estimators=200`, `max_depth=15`, `min_samples_leaf=10`, `class_weight='balanced'`, `random_state=42`.

Baseline expected severity weight: **1.168** (used for multiplier normalization — context with predicted weight equal to this scores 1.0).

## per-class metrics

| class | label | precision | recall | f1 | support |
|-------|-------|-----------|--------|----|---------|
| 1 | Fatal | 0.013 | 0.117 | 0.023 | 120 |
| 2 | Serious | 0.175 | 0.559 | 0.267 | 3901 |
| 3 | Slight | 0.860 | 0.463 | 0.602 | 20695 |

**Overall accuracy: 0.477**

**Macro F1: 0.297**

**Weighted F1: 0.546**

## confusion matrix

Rows = true class, columns = predicted class. Classes are 1, 2, 3.

```
        pred=1   pred=2   pred=3
true=1:      14       67       39
true=2:     198     2181     1522
true=3:     893    10216     9586
```

## feature importances

- `hour`: 0.298
- `month`: 0.212
- `day_of_week`: 0.151
- `road_type`: 0.135
- `speed_limit`: 0.132
- `weather_conditions`: 0.073
