"""Cluster London collisions into hotspots and persist to DB.

    cd backend
    .venv/Scripts/python -m scripts.build_hotspots
"""
from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
from geoalchemy2.shape import from_shape
from shapely.geometry import Point
from sqlalchemy import insert, text

from app.db.connection import engine
from app.db.models import Hotspot
from app.models.clustering import cluster_hotspots, compute_cluster_centroids

REPO_ROOT = Path(__file__).resolve().parents[2]
PROCESSED = REPO_ROOT / "data" / "processed"


def main() -> None:
    started = time.perf_counter()
    df = pd.read_parquet(PROCESSED / "london_accidents.parquet")

    clustered = cluster_hotspots(df)
    centroids = compute_cluster_centroids(clustered)

    n_noise = int((clustered["cluster_id"] == -1).sum())
    print(f"clustered {len(df):,} accidents into {len(centroids):,} hotspots ({n_noise:,} noise)")

    records = [
        {
            "cluster_id": int(r.cluster_id),
            "centroid": from_shape(Point(r.longitude, r.latitude), srid=4326),
            "accident_count": int(r.accident_count),
            "avg_severity_weight": float(r.avg_severity_weight),
        }
        for r in centroids.itertuples(index=False)
    ]
    with engine.begin() as conn:
        conn.execute(text("TRUNCATE TABLE hotspots RESTART IDENTITY"))
        if records:
            conn.execute(insert(Hotspot.__table__), records)

    elapsed = time.perf_counter() - started
    print(f"done in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
