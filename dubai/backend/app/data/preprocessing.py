from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from app.data.loader import DUBAI_ROOT, load_incidents
from app.data.type_labels import label_en

# Greater-Dubai bounding box: (min_lat, max_lat, min_lng, max_lng). Rows outside
# are bad geocodes / other emirates / (0,0) null-island junk.
DUBAI_BBOX = (24.7, 25.5, 54.8, 55.8)

SEVERE_AR = "بليغ"
MINOR_AR = "بسيط"
MODERATE_AR = "متوسط"
# Dubai Police tagged severity with three words pre-2021 (minor/moderate/severe)
# and effectively dropped "moderate" after. Strip whichever suffix is present so
# the incident type comes out clean (the old parser only knew minor/severe and
# left "- متوسط" stuck on ~3k types, fragmenting them).
_SEV_SUFFIX = re.compile(r"\s*-\s*(بسيط|بليغ|متوسط)\s*$")

# An incident counts as a collision if its type mentions one of these verbs.
# NOTE: rollovers are "تدهور" in this data ("انقلاب" never appears); the bare
# new-era forms (تدهور دراجة نارية, …) carry no other keyword, so without
# "تدهور" here ~15k rollovers/fires — many severe — were silently dropped.
# "حريق" keeps vehicle fires (the old-era حادث حريق… is already a category).
COLLISION_KW = ("صدم", "اصطدام", "دهس", "حادث", "تدهور", "حريق")
_COLLISION_RE = re.compile("|".join(COLLISION_KW))

PROCESSED = DUBAI_ROOT / "data" / "processed" / "collisions.parquet"


def decode_severity(name: str) -> str:
    s = name.strip()
    if s.endswith(SEVERE_AR):
        return "severe"
    if s.endswith(MINOR_AR):
        return "minor"
    if s.endswith(MODERATE_AR):
        return "moderate"
    return "untagged"


def incident_type(name: str) -> str:
    return _SEV_SUFFIX.sub("", name.strip()).strip()


def is_collision(base_type: str) -> bool:
    return bool(_COLLISION_RE.search(base_type))


def in_dubai_bbox(lat: pd.Series, lng: pd.Series) -> pd.Series:
    lo_lat, hi_lat, lo_lng, hi_lng = DUBAI_BBOX
    return lat.between(lo_lat, hi_lat) & lng.between(lo_lng, hi_lng)


def clean(raw: pd.DataFrame) -> pd.DataFrame:
    name = raw["acci_name"].astype(str)
    df = pd.DataFrame({"collision_id": raw["acci_id"]})
    # acci_x/acci_y are mislabeled in the source: acci_x is the LATITUDE,
    # acci_y is the LONGITUDE. Map them to the correct names here.
    df["lat"] = pd.to_numeric(raw["acci_x"], errors="coerce")
    df["lng"] = pd.to_numeric(raw["acci_y"], errors="coerce")
    # Year-first ISO timestamps — do NOT pass dayfirst, it corrupts them.
    dt = pd.to_datetime(raw["acci_time"], errors="coerce")
    df["datetime"] = dt
    df["hour"] = dt.dt.hour
    df["day_of_week"] = dt.dt.dayofweek
    df["month"] = dt.dt.month
    df["year"] = dt.dt.year
    df["incident_type"] = name.map(incident_type)
    df["incident_type_en"] = df["incident_type"].map(label_en)
    df["severity"] = name.map(decode_severity)

    keep = (
        df["incident_type"].map(is_collision)
        & in_dubai_bbox(df["lat"], df["lng"])
        & df["datetime"].notna()
    )
    # Drop exact-duplicate records (same id reported twice with identical
    # time/coords/type/severity). Rows sharing an id but differing in location
    # are kept — they're distinct geocoded points.
    return df[keep].drop_duplicates().reset_index(drop=True)


def run_pipeline(out: Path | None = None) -> pd.DataFrame:
    df = clean(load_incidents())
    dest = out or PROCESSED
    dest.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(dest, index=False)
    return df


if __name__ == "__main__":
    result = run_pipeline()
    print(f"wrote {len(result):,} collisions to {PROCESSED}")
    print(result["severity"].value_counts().to_dict())
