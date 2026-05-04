"""Train the Random Forest temporal severity model on London STATS19.

    cd backend
    .venv/Scripts/python -m scripts.train_temporal
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from app.models.temporal import save_artifact, train_temporal_model

REPO_ROOT = Path(__file__).resolve().parents[2]
PROCESSED = REPO_ROOT / "data" / "processed"
TRAINED = REPO_ROOT / "backend" / "trained_models"


def _write_report(metrics: dict[str, Any], path: Path) -> None:
    report = metrics["report"]
    cm = metrics["confusion_matrix"]
    fi = metrics["feature_importances"]
    label = {"1": "Fatal", "2": "Serious", "3": "Slight"}

    lines = [
        "# temporal random forest — evaluation\n",
        "\n",
        f"Trained on London STATS19 collisions (2020-2024). 80/20 stratified split, ",
        f"{metrics['n_train']:,} training rows, {metrics['n_test']:,} test rows.\n",
        f"\nHyperparameters: `n_estimators=200`, `max_depth=15`, `min_samples_leaf=10`, ",
        "`class_weight='balanced'`, `random_state=42`.\n",
        f"\nBaseline expected severity weight: **{metrics['baseline_expected_weight']:.3f}** ",
        "(used for multiplier normalization — context with predicted weight equal to this scores 1.0).\n",
        "\n## per-class metrics\n",
        "\n",
        "| class | label | precision | recall | f1 | support |\n",
        "|-------|-------|-----------|--------|----|---------|\n",
    ]
    for c in ("1", "2", "3"):
        if c in report:
            r = report[c]
            lines.append(
                f"| {c} | {label[c]} | {r['precision']:.3f} | {r['recall']:.3f} | "
                f"{r['f1-score']:.3f} | {int(r['support'])} |\n"
            )
    lines.append(f"\n**Overall accuracy: {report['accuracy']:.3f}**\n")
    lines.append(f"\n**Macro F1: {report['macro avg']['f1-score']:.3f}**\n")
    lines.append(f"\n**Weighted F1: {report['weighted avg']['f1-score']:.3f}**\n")

    lines.append("\n## confusion matrix\n\nRows = true class, columns = predicted class. Classes are 1, 2, 3.\n\n```\n")
    lines.append("        pred=1   pred=2   pred=3\n")
    for i, row in enumerate(cm):
        lines.append(f"true={i+1}: " + "  ".join(str(int(v)).rjust(7) for v in row) + "\n")
    lines.append("```\n")

    lines.append("\n## feature importances\n\n")
    for f, imp in sorted(fi.items(), key=lambda kv: kv[1], reverse=True):
        lines.append(f"- `{f}`: {imp:.3f}\n")

    path.write_text("".join(lines))


def main() -> None:
    started = time.perf_counter()
    df = pd.read_parquet(PROCESSED / "london_accidents.parquet")
    print(f"loaded {len(df):,} accidents from london parquet")

    artifact, metrics = train_temporal_model(df)
    elapsed = time.perf_counter() - started

    print(
        f"trained in {elapsed:.1f}s — "
        f"accuracy={metrics['report']['accuracy']:.3f}, "
        f"macro f1={metrics['report']['macro avg']['f1-score']:.3f}, "
        f"baseline weight={metrics['baseline_expected_weight']:.3f}"
    )
    print("per-class f1:")
    for c in ("1", "2", "3"):
        if c in metrics["report"]:
            r = metrics["report"][c]
            print(f"  class {c}: precision={r['precision']:.3f}, recall={r['recall']:.3f}, f1={r['f1-score']:.3f}")

    path = save_artifact(artifact)
    print(f"saved {path} ({path.stat().st_size / 1e6:.1f} MB)")

    report_path = TRAINED / "REPORT.md"
    _write_report(metrics, report_path)
    print(f"wrote {report_path}")


if __name__ == "__main__":
    main()
