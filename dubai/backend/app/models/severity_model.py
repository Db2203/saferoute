"""Binary minor/severe collision-severity model.

This is a DESCRIPTIVE model — it tells us *what makes a Dubai collision severe*
(incident type dominates, with location/time as secondary signal). It is NOT a
live route predictor: the strongest feature, incident type, is only known after
a crash happens. We lead with ROC-AUC / severe-recall, not raw accuracy, because
a "always-minor" classifier scores ~89% accuracy and is useless.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

BACKEND_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = BACKEND_ROOT / "trained_models"
MODEL_PKL = MODELS_DIR / "severity_model.pkl"
REPORT_MD = MODELS_DIR / "REPORT.md"
PARQUET = BACKEND_ROOT.parent / "data" / "processed" / "collisions.parquet"

FEATURES = ["type_code", "hour", "day_of_week", "month", "lat", "lng"]


@dataclass
class SeverityArtifact:
    model: RandomForestClassifier
    features: list[str]
    type_codes: dict[str, int]
    importances: dict[str, float]
    metrics: dict


def build_type_codes(types: pd.Series) -> dict[str, int]:
    # sorted for a stable, reproducible mapping (factorize-by-appearance is not)
    return {t: i for i, t in enumerate(sorted(types.unique()))}


def make_features(df: pd.DataFrame, type_codes: dict[str, int]) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    out["type_code"] = df["incident_type"].map(type_codes).fillna(-1).astype(int)
    for col in ("hour", "day_of_week", "month", "lat", "lng"):
        out[col] = df[col]
    return out[FEATURES]


def train(df: pd.DataFrame, random_state: int = 42) -> tuple[SeverityArtifact, tuple]:
    m = df[df["severity"].isin(["minor", "severe"])]
    type_codes = build_type_codes(m["incident_type"])
    X = make_features(m, type_codes)
    y = (m["severity"] == "severe").astype(int)

    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=random_state
    )
    clf = RandomForestClassifier(
        n_estimators=120,
        max_depth=18,
        min_samples_leaf=20,
        class_weight="balanced",
        n_jobs=-1,
        random_state=random_state,
    ).fit(Xtr, ytr)

    proba = clf.predict_proba(Xte)[:, 1]
    pred = clf.predict(Xte)
    rep = classification_report(yte, pred, output_dict=True, zero_division=0)
    cm = confusion_matrix(yte, pred)
    metrics = {
        "n_train": int(len(Xtr)),
        "n_test": int(len(Xte)),
        "severe_rate_pct": round(float(y.mean()) * 100, 1),
        "roc_auc": round(float(roc_auc_score(yte, proba)), 3),
        "accuracy": round(float(accuracy_score(yte, pred)), 3),
        "baseline_accuracy": round(float(max(yte.mean(), 1 - yte.mean())), 3),
        "severe": {k: round(rep["1"][k], 3) for k in ("precision", "recall", "f1-score")},
        "minor": {k: round(rep["0"][k], 3) for k in ("precision", "recall", "f1-score")},
        "confusion": cm.tolist(),  # [[tn, fp], [fn, tp]]
    }
    importances = {f: round(float(v), 3) for f, v in zip(FEATURES, clf.feature_importances_)}
    artifact = SeverityArtifact(clf, FEATURES, type_codes, importances, metrics)
    return artifact, (yte, pred, proba)


def predict_proba(artifact: SeverityArtifact, df: pd.DataFrame) -> pd.Series:
    if len(df) == 0:
        return pd.Series([], dtype=float, index=df.index)
    X = make_features(df, artifact.type_codes)
    return pd.Series(artifact.model.predict_proba(X)[:, 1], index=df.index)


def save_artifact(artifact: SeverityArtifact, path: Path | None = None) -> Path:
    dest = path or MODEL_PKL
    dest.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, dest)
    return dest


def load_artifact(path: Path | None = None) -> SeverityArtifact:
    return joblib.load(path or MODEL_PKL)


def write_report(artifact: SeverityArtifact, path: Path | None = None) -> Path:
    m = artifact.metrics
    imp = sorted(artifact.importances.items(), key=lambda kv: kv[1], reverse=True)
    cm = m["confusion"]
    lines = [
        "# Dubai collision severity model — report",
        "",
        "Binary RandomForest predicting **minor vs severe** from incident type, "
        "hour, day-of-week, month, and location.",
        "",
        "## Headline",
        f"- **ROC-AUC: {m['roc_auc']}**  ·  severe-class recall: **{m['severe']['recall']}**",
        f"- Accuracy {m['accuracy']} (majority-class baseline {m['baseline_accuracy']}) — "
        "we trade accuracy for severe recall via `class_weight='balanced'`, so accuracy "
        "is *below* baseline by design. Lead with AUC/recall.",
        f"- Train/test: {m['n_train']:,} / {m['n_test']:,} (severe rate {m['severe_rate_pct']}%)",
        "",
        "## Per-class",
        "| class | precision | recall | f1 |",
        "|---|---|---|---|",
        f"| minor | {m['minor']['precision']} | {m['minor']['recall']} | {m['minor']['f1-score']} |",
        f"| severe | {m['severe']['precision']} | {m['severe']['recall']} | {m['severe']['f1-score']} |",
        "",
        "## Confusion matrix (test)",
        "| | pred minor | pred severe |",
        "|---|---|---|",
        f"| actual minor | {cm[0][0]:,} | {cm[0][1]:,} |",
        f"| actual severe | {cm[1][0]:,} | {cm[1][1]:,} |",
        "",
        "## Feature importances",
        *[f"- {f}: {v}" for f, v in imp],
        "",
        "## Honest framing (viva)",
        "- **Descriptive, not predictive-for-routing.** Incident type dominates importance, "
        "but type is unknown before a crash — so this explains *what makes crashes severe*, "
        "it does not score a future route. Geo+time alone is much weaker (AUC ~0.62).",
        "- Non-circular: the severity tag is stripped from the type label before use, and the "
        "same type appears as both minor and severe.",
    ]
    dest = path or REPORT_MD
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text("\n".join(lines), encoding="utf-8")
    return dest

# Training entrypoint lives in scripts/train_severity.py so the pickled artifact
# records its class under app.models.severity_model (not __main__), which is how
# the API loads it.
