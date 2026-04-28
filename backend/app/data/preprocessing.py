"""Clean STATS19 collisions and DfT AADT counts down to a London-only Parquet.

Three transforms on collisions:
  - drop rows missing latitude/longitude
  - parse date + time into a single datetime column (plus hour/month for ML features)
  - decode integer codes (severity, weather, road_type, etc.) into human labels
    using the lookup table shipped in the STATS19 data guide xlsx

For AADT we just bbox-filter to London — the file is already coordinate-keyed.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RAW_DIR = REPO_ROOT / "data" / "raw"
DEFAULT_PROCESSED_DIR = REPO_ROOT / "data" / "processed"

# Approx GLA bbox; tighter than full Greater London but covers all of TfL's network.
LONDON_BBOX = (51.28, -0.51, 51.69, 0.33)  # min_lat, min_lng, max_lat, max_lng

DECODE_FIELDS = (
    "collision_severity",
    "weather_conditions",
    "road_type",
    "light_conditions",
    "road_surface_conditions",
    "urban_or_rural_area",
    "day_of_week",
)


def load_code_list(guide_path: Path | str) -> pd.DataFrame:
    df = pd.read_excel(guide_path, sheet_name="2024_code_list")
    df = df.dropna(subset=["code/format", "label"])
    df["code/format"] = pd.to_numeric(df["code/format"], errors="coerce")
    df = df.dropna(subset=["code/format"])
    df["code/format"] = df["code/format"].astype(int)
    return df.rename(columns={"field name": "field_name", "code/format": "code"})


def build_lookup(code_df: pd.DataFrame, table: str, field: str) -> dict[int, str]:
    sub = code_df[(code_df["table"] == table) & (code_df["field_name"] == field)]
    return dict(zip(sub["code"].astype(int), sub["label"]))


def filter_to_london(df: pd.DataFrame) -> pd.DataFrame:
    min_lat, min_lng, max_lat, max_lng = LONDON_BBOX
    return df[
        df["latitude"].between(min_lat, max_lat)
        & df["longitude"].between(min_lng, max_lng)
    ].reset_index(drop=True)


def preprocess_collisions(collisions: pd.DataFrame, code_df: pd.DataFrame) -> pd.DataFrame:
    df = collisions.dropna(subset=["latitude", "longitude"]).copy()
    df = filter_to_london(df)

    df["datetime"] = pd.to_datetime(
        df["date"].astype(str) + " " + df["time"].astype(str),
        format="%d/%m/%Y %H:%M",
        errors="coerce",
    )
    df["hour"] = df["datetime"].dt.hour.astype("Int64")
    df["month"] = df["datetime"].dt.month.astype("Int64")

    for field in DECODE_FIELDS:
        if field in df.columns:
            df[f"{field}_label"] = df[field].map(build_lookup(code_df, "collision", field))

    return df


def preprocess_aadt(aadt: pd.DataFrame) -> pd.DataFrame:
    return filter_to_london(aadt.dropna(subset=["latitude", "longitude"]))


def run_pipeline(
    raw_dir: Path | str | None = None,
    processed_dir: Path | str | None = None,
) -> tuple[Path, Path]:
    from app.data.loader import load_aadt, load_stats19

    raw = Path(raw_dir) if raw_dir else DEFAULT_RAW_DIR
    out = Path(processed_dir) if processed_dir else DEFAULT_PROCESSED_DIR
    out.mkdir(parents=True, exist_ok=True)

    code_df = load_code_list(raw / "stats19" / "data-guide-2024.xlsx")

    london_collisions = preprocess_collisions(load_stats19(raw / "stats19"), code_df)
    coll_path = out / "london_accidents.parquet"
    london_collisions.to_parquet(coll_path, index=False)

    london_aadt = preprocess_aadt(load_aadt(raw / "aadt"))
    aadt_path = out / "london_aadt.parquet"
    london_aadt.to_parquet(aadt_path, index=False)

    return coll_path, aadt_path


if __name__ == "__main__":
    coll_path, aadt_path = run_pipeline()
    print(f"wrote {coll_path}")
    print(f"wrote {aadt_path}")
