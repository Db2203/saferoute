"""Random Forest temporal severity predictor.

Given a (time, weather, road) context, predicts a probability distribution
over severity classes (1=fatal / 2=serious / 3=slight) and converts that
into a single "risk multiplier" — a float where 1.0 means baseline (the
average severity expectation across the training set), values >1 mean the
context is more dangerous than baseline, values <1 mean less.

Routing (Stage 14) multiplies each edge's static risk_score by this
multiplier so that, e.g., a 11pm Friday in the rain raises the apparent
risk of every road compared to a sunny Tuesday morning.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

# Inverted weights: STATS19 has 1=fatal as the lowest int but it's the
# highest severity, so the formula multiplies by 4 - severity.
SEVERITY_WEIGHT = {1: 3, 2: 2, 3: 1}

FEATURES = ["hour", "day_of_week", "month", "weather_conditions", "road_type", "speed_limit"]

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MODEL_PATH = REPO_ROOT / "backend" / "trained_models" / "temporal_rf.pkl"


@dataclass
class TemporalArtifact:
    model: RandomForestClassifier
    classes: np.ndarray
    baseline_expected_weight: float


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    # -1 stands for "data missing or unknown" in the STATS19 codes; using
    # it for nulls keeps the model honest about missingness.
    return df[FEATURES].fillna(-1).astype(int)


def train_temporal_model(
    df: pd.DataFrame,
    *,
    random_state: int = 42,
) -> tuple[TemporalArtifact, dict[str, Any]]:
    X = build_features(df)
    y = df["collision_severity"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=random_state
    )

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_leaf=10,
        class_weight="balanced",
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    # Baseline expected severity weight = sum_c P(class=c) × weight(c)
    class_p = y_train.value_counts(normalize=True).to_dict()
    baseline = sum(class_p.get(c, 0.0) * SEVERITY_WEIGHT[c] for c in (1, 2, 3))

    artifact = TemporalArtifact(model=model, classes=model.classes_, baseline_expected_weight=float(baseline))

    y_pred = model.predict(X_test)
    metrics = {
        "report": classification_report(y_test, y_pred, output_dict=True, zero_division=0),
        "confusion_matrix": confusion_matrix(y_test, y_pred, labels=[1, 2, 3]),
        "feature_importances": dict(zip(FEATURES, model.feature_importances_)),
        "baseline_expected_weight": float(baseline),
        "n_train": len(y_train),
        "n_test": len(y_test),
    }
    return artifact, metrics


def save_artifact(artifact: TemporalArtifact, path: Path | str | None = None) -> Path:
    p = Path(path) if path else DEFAULT_MODEL_PATH
    p.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, p)
    return p


def load_artifact(path: Path | str | None = None) -> TemporalArtifact:
    return joblib.load(Path(path) if path else DEFAULT_MODEL_PATH)


def predict_severity_distribution(
    artifact: TemporalArtifact,
    features: dict[str, int | float],
) -> dict[int, float]:
    X = pd.DataFrame([features])[FEATURES].fillna(-1).astype(int)
    probs = artifact.model.predict_proba(X)[0]
    return {int(c): float(p) for c, p in zip(artifact.classes, probs)}


def predict_risk_multiplier(
    artifact: TemporalArtifact,
    features: dict[str, int | float],
) -> float:
    dist = predict_severity_distribution(artifact, features)
    expected_weight = sum(dist.get(c, 0.0) * SEVERITY_WEIGHT[c] for c in (1, 2, 3))
    if artifact.baseline_expected_weight <= 0:
        return 1.0
    return expected_weight / artifact.baseline_expected_weight
