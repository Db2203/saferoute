from dataclasses import dataclass

import networkx as nx
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.models.routing import RouteResult
from app.routes.routing import (
    get_graph,
    get_risk_scores,
    get_temporal_artifact,
)


@dataclass
class _StubArtifact:
    multiplier_to_return: float = 1.0


def _stub_predict_multiplier(monkeypatch, value: float) -> None:
    monkeypatch.setattr("app.routes.routing.predict_risk_multiplier", lambda art, feats: value)


def _stub_routes(monkeypatch, fast: RouteResult, safe: RouteResult) -> None:
    monkeypatch.setattr(
        "app.routes.routing.compute_fastest",
        lambda graph, o, d, risk, *, temporal_multiplier=1.0: fast,
    )
    monkeypatch.setattr(
        "app.routes.routing.compute_safest",
        lambda graph, o, d, risk, *, temporal_multiplier=1.0: safe,
    )


@pytest.fixture
def client():
    app.dependency_overrides[get_graph] = lambda: nx.MultiDiGraph(crs="epsg:4326")
    app.dependency_overrides[get_risk_scores] = lambda: {}
    app.dependency_overrides[get_temporal_artifact] = lambda: _StubArtifact()
    c = TestClient(app)
    yield c
    app.dependency_overrides.clear()


def _route_result(*, time_s: float, risk: float, dist_m: float = 1000.0) -> RouteResult:
    return RouteResult(
        nodes=[1, 2, 3],
        edges=[(1, 2, 0), (2, 3, 0)],
        geometry=[(-0.10, 51.50), (-0.09, 51.50), (-0.08, 51.50)],
        total_time_s=time_s,
        total_risk=risk,
        total_distance_m=dist_m,
    )


def test_returns_both_routes_and_comparison(client, monkeypatch):
    fast = _route_result(time_s=600.0, risk=80.0, dist_m=5000.0)
    safe = _route_result(time_s=900.0, risk=20.0, dist_m=7000.0)
    _stub_predict_multiplier(monkeypatch, 1.2)
    _stub_routes(monkeypatch, fast, safe)

    resp = client.get(
        "/api/route",
        params={"origin": "51.5,-0.1", "dest": "51.5,-0.08", "when": "2026-05-09T08:30"},
    )
    assert resp.status_code == 200
    data = resp.json()

    assert data["fastest"]["total_time_s"] == 600.0
    assert data["safest"]["total_time_s"] == 900.0
    assert data["comparison"]["extra_time_s"] == 300.0
    assert data["comparison"]["risk_reduction"] == 60.0
    assert data["comparison"]["risk_reduction_pct"] == pytest.approx(75.0)
    assert data["context"]["hour"] == 8
    # 2026-05-09 is a Saturday; STATS19 code for Saturday is 7
    assert data["context"]["day_of_week"] == 7
    assert data["context"]["month"] == 5
    assert data["context"]["temporal_multiplier"] == 1.2


def test_geometry_is_lng_lat_array(client, monkeypatch):
    fast = _route_result(time_s=100.0, risk=10.0)
    safe = _route_result(time_s=120.0, risk=5.0)
    _stub_predict_multiplier(monkeypatch, 1.0)
    _stub_routes(monkeypatch, fast, safe)

    resp = client.get("/api/route", params={"origin": "51.5,-0.1", "dest": "51.5,-0.08"})
    assert resp.status_code == 200
    geom = resp.json()["fastest"]["geometry"]
    assert geom[0] == [-0.10, 51.50]
    assert geom[-1] == [-0.08, 51.50]


def test_no_path_returns_404(client, monkeypatch):
    def raise_no_path(*args, **kwargs):
        raise nx.NetworkXNoPath("no path")
    _stub_predict_multiplier(monkeypatch, 1.0)
    monkeypatch.setattr("app.routes.routing.compute_fastest", raise_no_path)
    monkeypatch.setattr("app.routes.routing.compute_safest", raise_no_path)

    resp = client.get("/api/route", params={"origin": "51.5,-0.1", "dest": "51.5,-0.08"})
    assert resp.status_code == 404


def test_bad_origin_format_returns_400(client):
    resp = client.get("/api/route", params={"origin": "garbage", "dest": "51.5,-0.08"})
    assert resp.status_code == 400


def test_origin_out_of_range_returns_400(client):
    resp = client.get("/api/route", params={"origin": "200.0,500.0", "dest": "51.5,-0.08"})
    assert resp.status_code == 400


def test_bad_when_format_returns_400(client):
    resp = client.get(
        "/api/route",
        params={"origin": "51.5,-0.1", "dest": "51.5,-0.08", "when": "tomorrow at 8"},
    )
    assert resp.status_code == 400


def test_returns_503_when_graph_not_loaded():
    # don't override get_graph — it'll try to access app.state which is unset
    app.dependency_overrides.pop(get_graph, None)
    app.dependency_overrides[get_risk_scores] = lambda: {}
    app.dependency_overrides[get_temporal_artifact] = lambda: _StubArtifact()
    c = TestClient(app)
    try:
        resp = c.get("/api/route", params={"origin": "51.5,-0.1", "dest": "51.5,-0.08"})
        assert resp.status_code == 503
    finally:
        app.dependency_overrides.clear()


def test_zero_risk_route_still_computes_pct(client, monkeypatch):
    fast = _route_result(time_s=100.0, risk=0.0)
    safe = _route_result(time_s=200.0, risk=0.0)
    _stub_predict_multiplier(monkeypatch, 1.0)
    _stub_routes(monkeypatch, fast, safe)

    resp = client.get("/api/route", params={"origin": "51.5,-0.1", "dest": "51.5,-0.08"})
    assert resp.status_code == 200
    assert resp.json()["comparison"]["risk_reduction_pct"] == 0.0
