import numpy as np
import pandas as pd
import pytest

from app.models.temporal import (
    build_features,
    load_artifact,
    predict_risk_multiplier,
    predict_severity_distribution,
    save_artifact,
    train_temporal_model,
)


@pytest.fixture
def synthetic_df():
    rng = np.random.default_rng(42)
    n = 600
    return pd.DataFrame(
        {
            "hour": rng.integers(0, 24, n),
            "day_of_week": rng.integers(1, 8, n),
            "month": rng.integers(1, 13, n),
            "weather_conditions": rng.integers(1, 9, n),
            "road_type": rng.integers(1, 7, n),
            "speed_limit": rng.choice([20, 30, 40, 50, 60, 70], n),
            "collision_severity": rng.choice([1, 2, 3], n, p=[0.05, 0.2, 0.75]),
        }
    )


@pytest.fixture
def trained_artifact(synthetic_df):
    artifact, _ = train_temporal_model(synthetic_df)
    return artifact


def test_build_features_returns_only_expected_columns():
    df = pd.DataFrame(
        [{"hour": 8, "day_of_week": 2, "month": 5, "weather_conditions": 1, "road_type": 6, "speed_limit": 30, "extra": "ignore"}]
    )
    out = build_features(df)
    assert list(out.columns) == ["hour", "day_of_week", "month", "weather_conditions", "road_type", "speed_limit"]


def test_train_returns_baseline_and_metrics(synthetic_df):
    artifact, metrics = train_temporal_model(synthetic_df)
    assert set(int(c) for c in artifact.classes).issubset({1, 2, 3})
    assert 1.0 < artifact.baseline_expected_weight < 3.0  # always between weight bounds
    assert "report" in metrics and "confusion_matrix" in metrics
    assert metrics["confusion_matrix"].shape == (3, 3)


def test_predict_severity_distribution_sums_to_one(trained_artifact):
    features = {"hour": 8, "day_of_week": 2, "month": 5, "weather_conditions": 1, "road_type": 6, "speed_limit": 30}
    dist = predict_severity_distribution(trained_artifact, features)
    assert abs(sum(dist.values()) - 1.0) < 1e-6
    for c, p in dist.items():
        assert c in {1, 2, 3}
        assert 0.0 <= p <= 1.0


def test_predict_risk_multiplier_is_positive_float(trained_artifact):
    features = {"hour": 22, "day_of_week": 5, "month": 12, "weather_conditions": 2, "road_type": 6, "speed_limit": 60}
    mult = predict_risk_multiplier(trained_artifact, features)
    assert isinstance(mult, float)
    assert 0.3 < mult < 3.0  # generous range; should not blow up


def test_artifact_save_load_roundtrip(trained_artifact, tmp_path):
    path = tmp_path / "rf.pkl"
    save_artifact(trained_artifact, path)
    loaded = load_artifact(path)

    features = {"hour": 12, "day_of_week": 3, "month": 6, "weather_conditions": 1, "road_type": 6, "speed_limit": 30}
    before = predict_severity_distribution(trained_artifact, features)
    after = predict_severity_distribution(loaded, features)
    assert before.keys() == after.keys()
    for c in before:
        assert before[c] == pytest.approx(after[c], abs=1e-9)
    assert loaded.baseline_expected_weight == pytest.approx(trained_artifact.baseline_expected_weight)
