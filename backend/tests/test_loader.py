from pathlib import Path

import pytest

from app.data.loader import (
    load_aadt,
    load_casualties,
    load_stats19,
    load_vehicles,
)

FIXTURES = Path(__file__).parent / "fixtures"


def test_load_stats19_returns_collision_rows():
    df = load_stats19(FIXTURES / "stats19")
    assert len(df) == 5
    assert {"collision_index", "latitude", "longitude", "collision_severity"} <= set(df.columns)


def test_load_stats19_severity_in_expected_range():
    df = load_stats19(FIXTURES / "stats19")
    assert df["collision_severity"].isin([1, 2, 3]).all()


def test_load_vehicles_and_casualties_join_on_collision_index():
    collisions = load_stats19(FIXTURES / "stats19")
    vehicles = load_vehicles(FIXTURES / "stats19")
    casualties = load_casualties(FIXTURES / "stats19")
    assert vehicles["collision_index"].isin(collisions["collision_index"]).all()
    assert casualties["collision_index"].isin(collisions["collision_index"]).all()


def test_load_aadt_dedups_by_count_point_keeping_latest_year():
    df = load_aadt(FIXTURES / "aadt")
    # count point 1001 appears in 2021/2022/2023 — should be deduped to one row from 2023
    cp1001 = df[df["count_point_id"] == 1001]
    assert len(cp1001) == 1
    assert cp1001["year"].iloc[0] == 2023
    assert cp1001["all_motor_vehicles"].iloc[0] == 31200


def test_load_aadt_drops_rows_with_null_coords():
    df = load_aadt(FIXTURES / "aadt")
    # count point 9999 has null lat/lng, should be dropped
    assert 9999 not in df["count_point_id"].values
    assert df[["latitude", "longitude"]].notna().all().all()


def test_loader_raises_when_no_match(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_stats19(tmp_path)
