import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.data import analytics
from app.main import app

client = TestClient(app)


def _df():
    # 8 rows: 4 pedestrian (3 severe), 4 collision (1 severe), spread over hours/dows/years
    return pd.DataFrame(
        {
            "collision_id": list(range(1, 9)),
            # rows 1,2,5 share a cell (3 pedestrian crashes -> a real blackspot cell)
            "lat": [25.20, 25.20, 25.30, 25.30, 25.20, 25.10, 25.40, 25.40],
            "lng": [55.30, 55.30, 55.40, 55.40, 55.30, 55.20, 55.50, 55.50],
            "datetime": pd.to_datetime(
                [
                    "2022-03-07 02:00",  # Mon
                    "2022-03-08 02:00",  # Tue
                    "2023-06-15 14:00",  # Thu
                    "2023-06-16 14:00",  # Fri
                    "2022-03-07 02:00",  # Mon
                    "2023-06-15 14:00",  # Thu
                    "2022-07-01 23:00",
                    "2023-07-01 23:00",
                ]
            ),
            "hour": [2, 2, 14, 14, 2, 14, 23, 23],
            "day_of_week": [0, 1, 3, 4, 0, 3, 4, 5],
            "month": [3, 3, 6, 6, 3, 6, 7, 7],
            "year": [2022, 2022, 2023, 2023, 2022, 2023, 2022, 2023],
            "incident_type": ["P", "P", "C", "C", "P", "P", "C", "C"],
            "incident_type_en": [
                "Pedestrian run-over",
                "Pedestrian run-over",
                "Two-vehicle collision",
                "Two-vehicle collision",
                "Pedestrian run-over",
                "Pedestrian run-over",
                "Two-vehicle collision",
                "Two-vehicle collision",
            ],
            "severity": [
                "severe",
                "severe",
                "severe",
                "minor",
                "severe",
                "minor",
                "minor",
                "minor",
            ],
        }
    )


def setup_function():
    app.state.df = None


# --- pure functions -------------------------------------------------------

def test_unfiltered_summary_counts_everything():
    out = analytics.compute(_df(), {})
    assert out["summary"]["total"] == 8
    assert out["summary"]["severe"] == 4

def test_breakdowns_have_full_axes():
    out = analytics.compute(_df(), {})
    assert len(out["by_hour"]) == 24  # zero-filled, stable axis
    assert len(out["by_dow"]) == 7
    assert len(out["by_month"]) == 12
    assert len(out["hour_dow"]) == 24 * 7

def test_type_filter_narrows():
    full = analytics.compute(_df(), {})["summary"]["total"]
    ped = analytics.compute(_df(), {"type": "Pedestrian run-over"})["summary"]["total"]
    assert ped == 4 and ped < full

def test_combined_filter():
    out = analytics.compute(_df(), {"type": "Pedestrian run-over", "year": 2022})
    assert out["summary"]["total"] == 3  # rows 1,2,5

def test_empty_slice_is_zero_not_error():
    out = analytics.compute(_df(), {"type": "Pedestrian run-over", "year": 1999})
    assert out["summary"]["total"] == 0
    assert out["summary"]["severe_rate_pct"] is None
    assert all(h["count"] == 0 for h in out["by_hour"])

def test_dow_name_filter():
    out = analytics.compute(_df(), {"dow": "Mon"})
    assert out["summary"]["total"] == 2  # rows 1 and 5

def test_bad_dow_raises():
    with pytest.raises(ValueError):
        analytics.compute(_df(), {"dow": "Funday"})


# --- API ------------------------------------------------------------------

def test_analytics_503_when_no_df():
    assert client.get("/api/analytics").status_code == 503

def test_analytics_returns_when_loaded():
    app.state.df = _df()
    r = client.get("/api/analytics")
    assert r.status_code == 200 and r.json()["summary"]["total"] == 8

def test_analytics_filter_via_query():
    app.state.df = _df()
    r = client.get("/api/analytics?type=Pedestrian run-over&year=2022")
    assert r.status_code == 200 and r.json()["summary"]["total"] == 3

def test_analytics_bad_dow_422():
    app.state.df = _df()
    assert client.get("/api/analytics?dow=Funday").status_code == 422

def test_blackspots_recomputes_when_filtered():
    app.state.df = _df()
    app.state.blackspots = {"type": "FeatureCollection", "features": []}  # precomputed is empty
    # filtered path must come from df, not the empty precomputed set
    r = client.get("/api/blackspots?type=Pedestrian run-over")
    assert r.status_code == 200
    assert len(r.json()["features"]) > 0


def test_blackspots_sparse_filter_returns_empty_not_500():
    # a filter where no grid cell reaches min_count must not crash (regression:
    # empty agg used to reindex to NaN -> "cannot convert float NaN to integer")
    app.state.df = _df()
    app.state.blackspots = {"type": "FeatureCollection", "features": []}
    # Two-vehicle rows are scattered 2-per-cell (< min_count 3)
    r = client.get("/api/blackspots?type=Two-vehicle collision")
    assert r.status_code == 200
    assert r.json()["features"] == []


def test_analytics_invalid_severity_and_type_422():
    app.state.df = _df()
    assert client.get("/api/analytics?severity=garbage").status_code == 422
    assert client.get("/api/analytics?type=NotARealType").status_code == 422
