import pandas as pd
import pytest

from app.data.preprocessing import (
    LONDON_BBOX,
    build_lookup,
    filter_to_london,
    preprocess_aadt,
    preprocess_collisions,
)


@pytest.fixture
def code_df():
    return pd.DataFrame(
        [
            {"table": "collision", "field_name": "collision_severity", "code": 1, "label": "Fatal"},
            {"table": "collision", "field_name": "collision_severity", "code": 2, "label": "Serious"},
            {"table": "collision", "field_name": "collision_severity", "code": 3, "label": "Slight"},
            {"table": "collision", "field_name": "weather_conditions", "code": 1, "label": "Fine no high winds"},
            {"table": "collision", "field_name": "weather_conditions", "code": 2, "label": "Raining no high winds"},
            {"table": "collision", "field_name": "day_of_week", "code": 2, "label": "Monday"},
        ]
    )


def _collision_row(**overrides):
    row = {
        "collision_index": "X",
        "collision_severity": 3,
        "longitude": -0.127758,
        "latitude": 51.507351,
        "date": "15/03/2021",
        "time": "08:30",
        "day_of_week": 2,
        "weather_conditions": 1,
    }
    row.update(overrides)
    return row


def test_filter_to_london_drops_rows_outside_bbox():
    df = pd.DataFrame(
        [
            {"latitude": 51.5, "longitude": -0.1},  # London
            {"latitude": 53.48, "longitude": -2.24},  # Manchester
            {"latitude": 55.95, "longitude": -3.19},  # Edinburgh
        ]
    )
    out = filter_to_london(df)
    assert len(out) == 1
    assert out.iloc[0]["latitude"] == 51.5


def test_preprocess_drops_null_coords(code_df):
    df = pd.DataFrame(
        [
            _collision_row(collision_index="A"),
            _collision_row(collision_index="B", latitude=None),
            _collision_row(collision_index="C", longitude=None),
        ]
    )
    out = preprocess_collisions(df, code_df)
    assert list(out["collision_index"]) == ["A"]


def test_preprocess_parses_datetime_and_extracts_hour_month(code_df):
    df = pd.DataFrame([_collision_row(date="15/03/2021", time="08:30")])
    out = preprocess_collisions(df, code_df)
    assert out.iloc[0]["datetime"] == pd.Timestamp("2021-03-15 08:30:00")
    assert out.iloc[0]["hour"] == 8
    assert out.iloc[0]["month"] == 3


def test_preprocess_decodes_severity_and_weather(code_df):
    df = pd.DataFrame(
        [
            _collision_row(collision_severity=1, weather_conditions=2),
            _collision_row(collision_severity=3, weather_conditions=1),
        ]
    )
    out = preprocess_collisions(df, code_df)
    assert list(out["collision_severity_label"]) == ["Fatal", "Slight"]
    assert list(out["weather_conditions_label"]) == ["Raining no high winds", "Fine no high winds"]


def test_preprocess_unknown_code_decodes_to_nan(code_df):
    df = pd.DataFrame([_collision_row(weather_conditions=99)])
    out = preprocess_collisions(df, code_df)
    assert pd.isna(out.iloc[0]["weather_conditions_label"])


def test_preprocess_aadt_keeps_only_london():
    df = pd.DataFrame(
        [
            {"count_point_id": 1, "latitude": 51.5, "longitude": -0.1, "all_motor_vehicles": 1000},
            {"count_point_id": 2, "latitude": 53.48, "longitude": -2.24, "all_motor_vehicles": 800},
        ]
    )
    out = preprocess_aadt(df)
    assert list(out["count_point_id"]) == [1]


def test_build_lookup_filters_by_table_and_field(code_df):
    sev = build_lookup(code_df, "collision", "collision_severity")
    assert sev == {1: "Fatal", 2: "Serious", 3: "Slight"}
    weather = build_lookup(code_df, "collision", "weather_conditions")
    assert weather[2] == "Raining no high winds"


def test_london_bbox_is_sane():
    min_lat, min_lng, max_lat, max_lng = LONDON_BBOX
    assert min_lat < max_lat
    assert min_lng < max_lng
    # central London (Trafalgar Square) should be inside
    assert min_lat <= 51.5074 <= max_lat
    assert min_lng <= -0.1278 <= max_lng
