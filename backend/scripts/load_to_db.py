"""Bulk-load preprocessed Parquet into Postgres.

Truncates tables first; the Parquet at `data/processed/*.parquet` is the
source of truth. Run after `init_db` and after preprocessing has produced
the parquet files.

    cd backend
    .venv/Scripts/python -m scripts.load_to_db
"""
from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
from geoalchemy2.shape import from_shape
from shapely.geometry import Point
from sqlalchemy import insert, text

from app.db.connection import engine
from app.db.models import AADTPoint, Accident

REPO_ROOT = Path(__file__).resolve().parents[2]
PROCESSED = REPO_ROOT / "data" / "processed"

CHUNK = 5000


def _build_accident_records(df: pd.DataFrame) -> list[dict]:
    records: list[dict] = []
    for row in df.itertuples(index=False):
        records.append(
            {
                "collision_index": row.collision_index,
                "geom": from_shape(Point(row.longitude, row.latitude), srid=4326),
                "occurred_at": row.datetime,
                "hour": int(row.hour) if pd.notna(row.hour) else None,
                "day_of_week": int(row.day_of_week) if pd.notna(row.day_of_week) else None,
                "month": int(row.month) if pd.notna(row.month) else None,
                "severity": int(row.collision_severity),
                "severity_label": row.collision_severity_label,
                "weather": int(row.weather_conditions) if pd.notna(row.weather_conditions) else None,
                "weather_label": row.weather_conditions_label,
                "road_type": int(row.road_type) if pd.notna(row.road_type) else None,
                "speed_limit": int(row.speed_limit) if pd.notna(row.speed_limit) else None,
                "light_conditions": int(row.light_conditions) if pd.notna(row.light_conditions) else None,
                "road_surface_conditions": int(row.road_surface_conditions) if pd.notna(row.road_surface_conditions) else None,
                "urban_or_rural_area": int(row.urban_or_rural_area) if pd.notna(row.urban_or_rural_area) else None,
                "number_of_vehicles": int(row.number_of_vehicles) if pd.notna(row.number_of_vehicles) else None,
                "number_of_casualties": int(row.number_of_casualties) if pd.notna(row.number_of_casualties) else None,
            }
        )
    return records


def _build_aadt_records(df: pd.DataFrame) -> list[dict]:
    records: list[dict] = []
    for row in df.itertuples(index=False):
        records.append(
            {
                "count_point_id": int(row.count_point_id),
                "geom": from_shape(Point(row.longitude, row.latitude), srid=4326),
                "year": int(row.year) if pd.notna(row.year) else None,
                "road_name": row.road_name if pd.notna(row.road_name) else None,
                "road_category": row.road_category if pd.notna(row.road_category) else None,
                "road_type": row.road_type if pd.notna(row.road_type) else None,
                "local_authority_name": row.local_authority_name if pd.notna(row.local_authority_name) else None,
                "region_name": row.region_name if pd.notna(row.region_name) else None,
                "link_length_km": float(row.link_length_km) if pd.notna(row.link_length_km) else None,
                "all_motor_vehicles": int(row.all_motor_vehicles) if pd.notna(row.all_motor_vehicles) else None,
            }
        )
    return records


def _bulk_insert(table_obj, records: list[dict]) -> None:
    with engine.begin() as conn:
        for i in range(0, len(records), CHUNK):
            conn.execute(insert(table_obj), records[i : i + CHUNK])


def main() -> None:
    started = time.perf_counter()

    accidents_df = pd.read_parquet(PROCESSED / "london_accidents.parquet")
    aadt_df = pd.read_parquet(PROCESSED / "london_aadt.parquet")

    print(f"truncating + reloading {len(accidents_df):,} accidents and {len(aadt_df):,} aadt points")
    with engine.begin() as conn:
        conn.execute(text("TRUNCATE TABLE accidents, aadt_points RESTART IDENTITY"))

    _bulk_insert(Accident.__table__, _build_accident_records(accidents_df))
    print(f"  accidents loaded ({time.perf_counter() - started:.1f}s)")

    _bulk_insert(AADTPoint.__table__, _build_aadt_records(aadt_df))
    elapsed = time.perf_counter() - started

    with engine.connect() as conn:
        a = conn.execute(text("SELECT COUNT(*) FROM accidents")).scalar()
        b = conn.execute(text("SELECT COUNT(*) FROM aadt_points")).scalar()
    print(f"done in {elapsed:.1f}s — accidents={a:,}, aadt_points={b:,}")


if __name__ == "__main__":
    main()
