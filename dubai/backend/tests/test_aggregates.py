import pandas as pd

from app.data.aggregates import (
    counts_by_dow,
    grid_blackspots,
    severe_rate_by_type,
    summary,
)


def _df():
    return pd.DataFrame(
        {
            "collision_id": [1, 2, 3, 4, 5, 6],
            "lat": [25.2000, 25.2001, 25.3000, 25.3000, 25.3000, 25.1000],
            "lng": [55.3000, 55.3001, 55.4000, 55.4000, 55.4000, 55.2000],
            "datetime": pd.to_datetime(["2020-01-06"] * 6),  # a Monday
            "hour": [1, 2, 3, 4, 5, 6],
            "day_of_week": [0, 1, 2, 3, 4, 5],
            "month": [1] * 6,
            "year": [2020] * 6,
            "incident_type": ["A", "A", "B", "B", "B", "A"],
            "incident_type_en": ["Atype", "Atype", "Btype", "Btype", "Btype", "Atype"],
            "severity": ["severe", "minor", "severe", "severe", "minor", "minor"],
        }
    )


def test_severe_rate_by_type_sorted():
    rows = severe_rate_by_type(_df(), min_n=1)  # grouped by English label
    assert [r["type_en"] for r in rows] == ["Btype", "Atype"]  # B 66.7% before A 33.3%
    assert rows[0]["severe_rate_pct"] == 66.7
    assert rows[1]["severe_rate_pct"] == 33.3


def test_summary():
    s = summary(_df())
    assert s["total"] == 6
    assert s["severe"] == 3 and s["minor"] == 3
    assert s["severe_rate_pct"] == 50.0


def test_grid_blackspots_cells_and_dominant():
    geo = grid_blackspots(_df(), cell=0.002, min_count=1)
    assert geo["type"] == "FeatureCollection"
    # rows 1&2 merge into one cell; rows 3-5 one cell; row 6 one cell -> 3 cells
    assert len(geo["features"]) == 3
    top = geo["features"][0]["properties"]  # sorted by wsum desc -> the 3-row cell
    assert top["count"] == 3 and top["dominant_type"] == "Btype"
    # GeoJSON coordinate order is [lng, lat]
    lng, lat = geo["features"][0]["geometry"]["coordinates"]
    assert 55.0 < lng < 56.0 and 24.0 < lat < 26.0


def test_counts_by_dow():
    rows = counts_by_dow(_df())
    by = {r["dow"]: r["count"] for r in rows}
    assert by["Mon"] == 1 and by["Sat"] == 1
