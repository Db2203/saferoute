import pandas as pd

from app.data.preprocessing import (
    clean,
    decode_severity,
    in_dubai_bbox,
    incident_type,
    is_collision,
)
from app.data.type_labels import label_en


def test_decode_severity():
    assert decode_severity("اصطدام بين مركبتين - بليغ") == "severe"
    assert decode_severity("صدم جدار - بسيط") == "minor"
    assert decode_severity("حادث اصطدام بين سيارتين- متوسط") == "moderate"
    assert decode_severity("تعطل مركبة على طريق عام") == "untagged"


def test_incident_type_strips_severity_suffix():
    assert incident_type("اصطدام بين مركبتين - بليغ") == "اصطدام بين مركبتين"
    assert incident_type("حادث صدم عمود- بسيط") == "حادث صدم عمود"
    # moderate suffix (with and without surrounding spaces) is stripped too
    assert incident_type("حادث اصطدام بين سيارتين- متوسط") == "حادث اصطدام بين سيارتين"
    assert incident_type("حادث دهس طفل-متوسط") == "حادث دهس طفل"


def test_is_collision():
    assert is_collision("اصطدام بين مركبتين")
    assert is_collision("صدم جدار")
    assert is_collision("دهس")
    assert is_collision("تدهور دراجة نارية")  # bare rollover (no حادث prefix)
    assert is_collision("حريق مركبة أثناء سيرها")  # vehicle fire
    assert not is_collision("تعطل مركبة على طريق عام")  # breakdown
    assert not is_collision("الوقوف خلف المركبات (دبل بارك)")  # double-parking


def test_in_dubai_bbox():
    lat = pd.Series([25.2, 99.0, 24.0])
    lng = pd.Series([55.3, 55.3, 55.3])
    assert list(in_dubai_bbox(lat, lng)) == [True, False, False]


def test_clean_swaps_coords_and_filters():
    raw = pd.DataFrame(
        {
            "acci_id": [1, 2, 3, 4],
            "acci_time": [
                "2024-05-01 14:30:00.000",  # kept
                "2024-05-01 02:00:00.000",  # breakdown -> dropped
                "2024-05-01 03:00:00.000",  # out of bbox -> dropped
                "not-a-date",               # bad time -> dropped
            ],
            "acci_name": [
                "اصطدام بين مركبتين - بليغ",
                "تعطل مركبة على طريق عام",
                "صدم جدار - بسيط",
                "صدم جدار - بسيط",
            ],
            "acci_x": [25.20, 25.20, 99.00, 25.20],  # row 3 latitude out of range
            "acci_y": [55.27, 55.27, 55.27, 55.27],
        }
    )
    out = clean(raw)
    assert list(out["collision_id"]) == [1]
    row = out.iloc[0]
    assert row["lat"] == 25.20 and row["lng"] == 55.27  # coords mapped correctly
    assert row["severity"] == "severe"
    assert row["incident_type"] == "اصطدام بين مركبتين"
    assert row["hour"] == 14


def test_clean_drops_exact_duplicates_but_keeps_distinct_locations():
    raw = pd.DataFrame(
        {
            "acci_id": [7, 7, 7],
            "acci_time": ["2024-05-01 10:00:00.000"] * 3,
            "acci_name": ["صدم عمود - بسيط"] * 3,
            "acci_x": [25.20, 25.20, 25.25],  # rows 1&2 identical, row 3 elsewhere
            "acci_y": [55.30, 55.30, 55.35],
        }
    )
    out = clean(raw)
    assert len(out) == 2  # one exact dup removed, distinct-location row kept


def test_label_en():
    assert label_en("صدم دراجة نارية") == "Hit motorcycle"
    assert label_en("nonexistent type") is None


def test_label_en_merges_vocabulary_migration():
    # the old حادث-prefixed phrasing and the new bare phrasing collapse to one
    assert label_en("صدم عمود") == label_en("حادث صدم عمود") == "Hit pole"
    assert label_en("اصطدام بين مركبتين") == label_en("حادث اصطدام بين سيارتين") == "Two-vehicle collision"
    # old victim-specific pedestrian types all merge with the new generic دهس
    for t in ("دهس", "حادث دهس رجل", "حادث دهس امراة", "حادث دهس طفل"):
        assert label_en(t) == "Pedestrian run-over"
    # bare new-era rollovers map alongside the old حادث-prefixed ones
    assert label_en("تدهور دراجة نارية") == label_en("حادث تدهور مركبة خفيفة") == "Vehicle rollover"
