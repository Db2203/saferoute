"""Compute risk score per road segment and persist to DB.

    cd backend
    .venv/Scripts/python -m scripts.build_risk_scores
"""
from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
from sqlalchemy import insert, text

from app.db.connection import engine
from app.db.models import RoadRiskScore
from app.models.graph import load_london_graph
from app.models.risk_scoring import score_road_segments

REPO_ROOT = Path(__file__).resolve().parents[2]
PROCESSED = REPO_ROOT / "data" / "processed"

CHUNK = 5000


def main() -> None:
    started = time.perf_counter()

    print("loading graph...")
    graph = load_london_graph()
    print(f"  graph: {graph.number_of_edges():,} edges, {graph.number_of_nodes():,} nodes")

    print("loading parquet...")
    accidents = pd.read_parquet(PROCESSED / "london_accidents.parquet")
    aadt = pd.read_parquet(PROCESSED / "london_aadt.parquet")
    print(f"  {len(accidents):,} accidents, {len(aadt):,} aadt points")

    print("snapping + scoring (this is the slow part)...")
    scores = score_road_segments(graph, accidents, aadt)
    fallback_pct = 100.0 * scores["aadt_is_fallback"].sum() / max(len(scores), 1)
    print(
        f"  {len(scores):,} scored edges; fallback rate: {fallback_pct:.1f}%; "
        f"risk min/median/max = "
        f"{scores['risk_score'].min():.2f} / {scores['risk_score'].median():.2f} / {scores['risk_score'].max():.2f}"
    )

    typed_records = [
        {
            "u": int(r.u),
            "v": int(r.v),
            "key": int(r.key),
            "accident_count": int(r.accident_count),
            "severity_sum": int(r.severity_sum),
            "aadt": float(r.aadt),
            "aadt_is_fallback": bool(r.aadt_is_fallback),
            "raw_score": float(r.raw_score),
            "risk_score": float(r.risk_score),
        }
        for r in scores.itertuples(index=False)
    ]

    print("writing to DB...")
    with engine.begin() as conn:
        conn.execute(text("TRUNCATE TABLE road_risk_scores"))
        for i in range(0, len(typed_records), CHUNK):
            conn.execute(insert(RoadRiskScore.__table__), typed_records[i : i + CHUNK])

    elapsed = time.perf_counter() - started
    print(f"done in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
