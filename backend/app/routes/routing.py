"""GET /api/route — fastest + safest route between two points.

Query parameters:
  origin=lat,lng       required, WGS84
  dest=lat,lng         required, WGS84
  when=ISO_DATETIME    optional; defaults to now (UTC). Used for hour/dow/month features.
  weather=int          optional; STATS19 weather code (default 1 = Fine no high winds).

The endpoint computes a single temporal multiplier from the Random Forest
model in `app.models.temporal` and applies it uniformly to every edge during
both the fastest (alpha=1.0) and safest (alpha=0.3) route computations.
Per-edge road_type and speed_limit features fed to the RF use the dominant
London values (single carriageway, 30 mph) — a documented MVP simplification;
the dominant signal in the model is the time/weather features anyway.
"""
from __future__ import annotations

from datetime import datetime, timezone

import networkx as nx
from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import BaseModel

from app.models.routing import (
    EdgeKey,
    RouteResult,
    compute_fastest,
    compute_safest,
)
from app.models.temporal import TemporalArtifact, predict_risk_multiplier

router = APIRouter()

# Defaults baked into the multiplier feature vector when the user only gives
# a time. road_type=6 is "Single carriageway" (the dominant London class),
# speed_limit=30 is the dominant London limit.
_DEFAULT_ROAD_TYPE = 6
_DEFAULT_SPEED_LIMIT = 30


class RouteSegment(BaseModel):
    geometry: list[list[float]]  # [[lng, lat], ...] GeoJSON-friendly
    total_time_s: float
    total_distance_m: float
    total_risk: float
    edge_count: int


class RouteComparison(BaseModel):
    extra_time_s: float
    extra_distance_m: float
    risk_reduction: float
    risk_reduction_pct: float


class RouteContext(BaseModel):
    hour: int
    day_of_week: int
    month: int
    weather: int
    temporal_multiplier: float


class RouteResponse(BaseModel):
    fastest: RouteSegment
    safest: RouteSegment
    comparison: RouteComparison
    context: RouteContext


def get_graph(request: Request) -> nx.MultiDiGraph:
    graph = getattr(request.app.state, "graph", None)
    if graph is None:
        raise HTTPException(503, "graph not loaded; backend still warming up")
    return graph


def get_risk_scores(request: Request) -> dict[EdgeKey, float]:
    scores = getattr(request.app.state, "risk_scores", None)
    if scores is None:
        raise HTTPException(503, "risk scores not loaded; backend still warming up")
    return scores


def get_temporal_artifact(request: Request) -> TemporalArtifact:
    art = getattr(request.app.state, "temporal_artifact", None)
    if art is None:
        raise HTTPException(503, "temporal model not loaded; backend still warming up")
    return art


def _parse_coord(raw: str, label: str) -> tuple[float, float]:
    try:
        parts = [float(x) for x in raw.split(",")]
    except ValueError as e:
        raise HTTPException(400, f"{label} must be 'lat,lng' as two floats") from e
    if len(parts) != 2:
        raise HTTPException(400, f"{label} must be 'lat,lng' as two floats")
    lat, lng = parts
    if not (-90.0 <= lat <= 90.0) or not (-180.0 <= lng <= 180.0):
        raise HTTPException(400, f"{label} lat/lng out of range")
    return lat, lng


def _parse_when(raw: str | None) -> datetime:
    if raw is None:
        return datetime.now(tz=timezone.utc)
    try:
        return datetime.fromisoformat(raw)
    except ValueError as e:
        raise HTTPException(400, "when must be ISO-8601 datetime (e.g. 2026-05-09T08:30)") from e


def _segment(result: RouteResult) -> RouteSegment:
    return RouteSegment(
        geometry=[[lng, lat] for lng, lat in result.geometry],
        total_time_s=result.total_time_s,
        total_distance_m=result.total_distance_m,
        total_risk=result.total_risk,
        edge_count=len(result.edges),
    )


def _comparison(fastest: RouteResult, safest: RouteResult) -> RouteComparison:
    extra_time = safest.total_time_s - fastest.total_time_s
    extra_dist = safest.total_distance_m - fastest.total_distance_m
    risk_reduction = fastest.total_risk - safest.total_risk
    pct = (risk_reduction / fastest.total_risk * 100.0) if fastest.total_risk > 0 else 0.0
    return RouteComparison(
        extra_time_s=extra_time,
        extra_distance_m=extra_dist,
        risk_reduction=risk_reduction,
        risk_reduction_pct=pct,
    )


@router.get("/api/route", response_model=RouteResponse)
def get_route(
    origin: str = Query(..., description="lat,lng of starting point"),
    dest: str = Query(..., description="lat,lng of destination"),
    when: str | None = Query(None, description="ISO-8601 datetime; defaults to now"),
    weather: int = Query(1, description="STATS19 weather code (1=Fine no high winds)"),
    graph: nx.MultiDiGraph = Depends(get_graph),
    risk_scores: dict = Depends(get_risk_scores),
    artifact: TemporalArtifact = Depends(get_temporal_artifact),
) -> RouteResponse:
    o = _parse_coord(origin, "origin")
    d = _parse_coord(dest, "dest")
    when_dt = _parse_when(when)

    # STATS19 encodes day_of_week as 1=Sun..7=Sat (verified empirically against
    # the parquet). Python's isoweekday() is 1=Mon..7=Sun. The mapping
    # `(iso % 7) + 1` rotates by one with wrap-around: Mon(1)→2, Sun(7)→1.
    features = {
        "hour": when_dt.hour,
        "day_of_week": (when_dt.isoweekday() % 7) + 1,
        "month": when_dt.month,
        "weather_conditions": weather,
        "road_type": _DEFAULT_ROAD_TYPE,
        "speed_limit": _DEFAULT_SPEED_LIMIT,
    }
    multiplier = predict_risk_multiplier(artifact, features)

    try:
        fastest = compute_fastest(graph, o, d, risk_scores, temporal_multiplier=multiplier)
        safest = compute_safest(graph, o, d, risk_scores, temporal_multiplier=multiplier)
    except nx.NetworkXNoPath as e:
        raise HTTPException(404, "no route found between origin and destination") from e

    return RouteResponse(
        fastest=_segment(fastest),
        safest=_segment(safest),
        comparison=_comparison(fastest, safest),
        context=RouteContext(
            hour=features["hour"],
            day_of_week=features["day_of_week"],
            month=features["month"],
            weather=weather,
            temporal_multiplier=float(multiplier),
        ),
    )
