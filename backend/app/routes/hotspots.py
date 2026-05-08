"""GET /api/hotspots — accident hotspots within a bounding box.

Bounds are passed as `?bounds=south,west,north,east` (four comma-separated
floats in WGS84 / EPSG:4326). The query uses PostGIS `ST_MakeEnvelope` +
`ST_Intersects` against the GIST index on `hotspots.centroid`.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy import text
from sqlalchemy.engine import Engine

from app.db.connection import engine

router = APIRouter()


class Hotspot(BaseModel):
    cluster_id: int
    lat: float
    lng: float
    accident_count: int
    avg_severity_weight: float


class HotspotsResponse(BaseModel):
    hotspots: list[Hotspot]


def get_engine() -> Engine:
    return engine


def _parse_bounds(raw: str) -> tuple[float, float, float, float]:
    try:
        parts = [float(x) for x in raw.split(",")]
    except ValueError as e:
        raise HTTPException(status_code=400, detail="bounds must be four comma-separated floats") from e
    if len(parts) != 4:
        raise HTTPException(status_code=400, detail="bounds must be 'south,west,north,east'")
    south, west, north, east = parts
    if south >= north or west >= east:
        raise HTTPException(status_code=400, detail="bounds must have south < north and west < east")
    return south, west, north, east


_QUERY = text(
    """
    SELECT cluster_id,
           ST_Y(centroid::geometry) AS lat,
           ST_X(centroid::geometry) AS lng,
           accident_count,
           avg_severity_weight
    FROM hotspots
    WHERE ST_Intersects(
        centroid,
        ST_MakeEnvelope(:west, :south, :east, :north, 4326)
    )
    ORDER BY accident_count DESC
    """
)


@router.get("/api/hotspots", response_model=HotspotsResponse)
def get_hotspots(
    bounds: str = Query(..., description="south,west,north,east in WGS84"),
    db_engine: Engine = Depends(get_engine),
) -> HotspotsResponse:
    south, west, north, east = _parse_bounds(bounds)
    with db_engine.connect() as conn:
        rows = conn.execute(
            _QUERY, {"south": south, "west": west, "north": north, "east": east}
        ).fetchall()
    return HotspotsResponse(
        hotspots=[
            Hotspot(
                cluster_id=int(r.cluster_id),
                lat=float(r.lat),
                lng=float(r.lng),
                accident_count=int(r.accident_count),
                avg_severity_weight=float(r.avg_severity_weight),
            )
            for r in rows
        ]
    )
