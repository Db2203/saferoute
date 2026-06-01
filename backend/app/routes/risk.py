"""GET /api/risk and GET /api/temporal.

`/api/risk`     — risk score for one OSM road segment (u, v, key) at a given time.
`/api/temporal` — 24-hour risk-multiplier profile for a location (lat, lng).

Both reuse the Random Forest temporal multiplier from `app.models.temporal`
and the percentile-ranked risk scores loaded into `app.state` at startup. The
RF has no geographic features, so a location influences the result only through
the speed limit of its nearest road — a documented MVP simplification, the same
one `/api/route` makes.
"""
from __future__ import annotations

from datetime import datetime

import osmnx as ox
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel

from app.models.routing import EdgeKey
from app.models.temporal import TemporalArtifact, predict_risk_multiplier
from app.routes.routing import (
    _parse_when,
    get_graph,
    get_risk_scores,
    get_temporal_artifact,
)

router = APIRouter()

# Same defaults as /api/route: single carriageway is the dominant London road
# class, 30 mph the dominant limit (used when OSM has no maxspeed for a road).
_DEFAULT_ROAD_TYPE = 6
_DEFAULT_SPEED_LIMIT = 30


def _stats19_day_of_week(dt: datetime) -> int:
    # STATS19 encodes 1=Sun..7=Sat; Python isoweekday() is 1=Mon..7=Sun.
    # (iso % 7) + 1 rotates Mon(1)->2 ... Sun(7)->1.
    return (dt.isoweekday() % 7) + 1


def _parse_one_speed(value: object) -> int | None:
    text = str(value).lower().replace("mph", "").strip()
    try:
        return int(float(text))
    except ValueError:
        return None


def _edge_speed_limit(attrs: dict) -> int:
    """Best-effort OSM maxspeed → integer mph, falling back to the London default.

    OSM `maxspeed` is a string like "40 mph", sometimes a list when an edge
    merges several ways, sometimes absent.
    """
    raw = attrs.get("maxspeed")
    if raw is None:
        return _DEFAULT_SPEED_LIMIT
    candidates = raw if isinstance(raw, list) else [raw]
    for c in candidates:
        parsed = _parse_one_speed(c)
        if parsed is not None:
            return parsed
    return _DEFAULT_SPEED_LIMIT


class RiskContext(BaseModel):
    hour: int
    day_of_week: int
    month: int
    weather: int


class RiskResponse(BaseModel):
    u: int
    v: int
    key: int
    risk_score: float  # percentile-ranked 0-100 routing score
    speed_limit: int
    temporal_multiplier: float
    adjusted_risk: float  # risk_score * temporal_multiplier
    context: RiskContext


class HourlyRisk(BaseModel):
    hour: int
    temporal_multiplier: float


class TemporalResponse(BaseModel):
    lat: float
    lng: float
    matched_edge: list[int]  # [u, v, key] of the nearest road segment
    speed_limit: int
    day_of_week: int
    month: int
    weather: int
    profile: list[HourlyRisk]  # 24 entries, hour 0..23


@router.get("/api/risk", response_model=RiskResponse)
def get_risk(
    u: int = Query(..., description="OSM edge start node id"),
    v: int = Query(..., description="OSM edge end node id"),
    key: int = Query(0, description="OSM parallel-edge key (usually 0)"),
    when: str | None = Query(None, description="ISO-8601 datetime; defaults to now"),
    weather: int = Query(1, description="STATS19 weather code (1=Fine no high winds)"),
    graph=Depends(get_graph),
    risk_scores: dict[EdgeKey, float] = Depends(get_risk_scores),
    artifact: TemporalArtifact = Depends(get_temporal_artifact),
) -> RiskResponse:
    edge_key = (u, v, key)
    if edge_key not in risk_scores:
        raise HTTPException(404, "no risk score for that road segment (u, v, key)")

    attrs = graph.edges[u, v, key] if graph.has_edge(u, v, key) else {}
    speed_limit = _edge_speed_limit(attrs)

    when_dt = _parse_when(when)
    features = {
        "hour": when_dt.hour,
        "day_of_week": _stats19_day_of_week(when_dt),
        "month": when_dt.month,
        "weather_conditions": weather,
        "road_type": _DEFAULT_ROAD_TYPE,
        "speed_limit": speed_limit,
    }
    multiplier = float(predict_risk_multiplier(artifact, features))
    score = risk_scores[edge_key]

    return RiskResponse(
        u=u,
        v=v,
        key=key,
        risk_score=score,
        speed_limit=speed_limit,
        temporal_multiplier=multiplier,
        adjusted_risk=score * multiplier,
        context=RiskContext(
            hour=when_dt.hour,
            day_of_week=_stats19_day_of_week(when_dt),
            month=when_dt.month,
            weather=weather,
        ),
    )


@router.get("/api/temporal", response_model=TemporalResponse)
def get_temporal(
    lat: float = Query(..., description="latitude (WGS84)"),
    lng: float = Query(..., description="longitude (WGS84)"),
    when: str | None = Query(None, description="ISO-8601 datetime; date sets day/month, hour is swept"),
    weather: int = Query(1, description="STATS19 weather code (1=Fine no high winds)"),
    graph=Depends(get_graph),
    artifact: TemporalArtifact = Depends(get_temporal_artifact),
) -> TemporalResponse:
    if not (-90.0 <= lat <= 90.0) or not (-180.0 <= lng <= 180.0):
        raise HTTPException(400, "lat/lng out of range")

    when_dt = _parse_when(when)
    dow = _stats19_day_of_week(when_dt)

    # Snap to the nearest road segment so the location influences the profile
    # through that road's speed limit (the RF has no lat/lng feature).
    try:
        u, v, k = ox.distance.nearest_edges(graph, X=lng, Y=lat)
    except Exception as e:  # graph empty, or point not matchable
        raise HTTPException(503, "could not match location to the road network") from e

    attrs = graph.edges[u, v, k] if graph.has_edge(u, v, k) else {}
    speed_limit = _edge_speed_limit(attrs)

    profile: list[HourlyRisk] = []
    for hour in range(24):
        features = {
            "hour": hour,
            "day_of_week": dow,
            "month": when_dt.month,
            "weather_conditions": weather,
            "road_type": _DEFAULT_ROAD_TYPE,
            "speed_limit": speed_limit,
        }
        multiplier = float(predict_risk_multiplier(artifact, features))
        profile.append(HourlyRisk(hour=hour, temporal_multiplier=multiplier))

    return TemporalResponse(
        lat=lat,
        lng=lng,
        matched_edge=[int(u), int(v), int(k)],
        speed_limit=speed_limit,
        day_of_week=dow,
        month=when_dt.month,
        weather=weather,
        profile=profile,
    )
