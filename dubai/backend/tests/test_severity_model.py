import pandas as pd

from app.models.severity_model import (
    FEATURES,
    make_features,
    predict_proba,
    train,
)


def _df():
    # type B is severe-heavy, type A is minor — a learnable signal
    rows = [{"incident_type": "A", "severity": "minor"} for _ in range(60)]
    rows += [
        {"incident_type": "B", "severity": "severe" if i < 40 else "minor"}
        for i in range(60)
    ]
    df = pd.DataFrame(rows)
    n = len(df)
    df["hour"] = [i % 24 for i in range(n)]
    df["day_of_week"] = [i % 7 for i in range(n)]
    df["month"] = [(i % 12) + 1 for i in range(n)]
    df["lat"] = 25.2
    df["lng"] = 55.3
    return df


def test_make_features_columns_and_unknown_type():
    codes = {"A": 0, "B": 1}
    df = pd.DataFrame(
        {
            "incident_type": ["A", "Z"],  # Z unknown -> -1
            "hour": [1, 2],
            "day_of_week": [0, 1],
            "month": [1, 2],
            "lat": [25.2, 25.2],
            "lng": [55.3, 55.3],
        }
    )
    X = make_features(df, codes)
    assert list(X.columns) == FEATURES
    assert X["type_code"].tolist() == [0, -1]


def test_train_returns_artifact_with_metrics():
    artifact, _ = train(_df())
    assert artifact.features == FEATURES
    assert set(artifact.type_codes) == {"A", "B"}
    assert "roc_auc" in artifact.metrics
    assert 0.0 <= artifact.metrics["roc_auc"] <= 1.0
    assert set(artifact.importances) == set(FEATURES)


def test_predict_proba_in_range():
    artifact, _ = train(_df())
    p = predict_proba(artifact, _df())
    assert len(p) == 120
    assert ((p >= 0) & (p <= 1)).all()


def test_save_load_roundtrip(tmp_path):
    from app.models.severity_model import load_artifact, save_artifact

    artifact, _ = train(_df())
    path = tmp_path / "severity.pkl"
    save_artifact(artifact, path)
    loaded = load_artifact(path)
    assert loaded.features == FEATURES
    out = predict_proba(loaded, _df())
    assert len(out) == 120 and ((out >= 0) & (out <= 1)).all()


def test_unknown_type_does_not_crash():
    artifact, _ = train(_df())
    df = pd.DataFrame(
        {
            "incident_type": ["a-type-never-seen-in-training"],
            "hour": [3],
            "day_of_week": [2],
            "month": [5],
            "lat": [25.2],
            "lng": [55.3],
        }
    )
    p = predict_proba(artifact, df)
    assert len(p) == 1 and 0.0 <= float(p.iloc[0]) <= 1.0


def test_empty_df_predict_returns_empty():
    artifact, _ = train(_df())
    empty = pd.DataFrame(
        columns=["incident_type", "hour", "day_of_week", "month", "lat", "lng"]
    )
    assert len(predict_proba(artifact, empty)) == 0


def test_training_is_reproducible():
    a1, _ = train(_df())
    a2, _ = train(_df())
    assert a1.metrics["roc_auc"] == a2.metrics["roc_auc"]
    assert a1.type_codes == a2.type_codes
    assert a1.importances == a2.importances
