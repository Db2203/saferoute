from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from app.data.loader import DUBAI_ROOT

PROCESSED = DUBAI_ROOT / "data" / "processed"
PARQUET = PROCESSED / "collisions.parquet"
STATS_JSON = PROCESSED / "stats.json"
BLACKSPOTS_GEOJSON = PROCESSED / "blackspots.geojson"

DOW = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def _tagged(df: pd.DataFrame) -> pd.DataFrame:
    # severe-rate is only meaningful where a minor/severe tag exists
    return df[df["severity"] != "untagged"]


def severe_rate_by_type(df: pd.DataFrame, min_n: int = 2000) -> list[dict]:
    # group by the merged English label (the canonical category), matching the
    # live /api/analytics view — grouping by raw Arabic type would split each
    # category into its pre/post-2021 vocabulary halves.
    out = []
    for en, sub in _tagged(df).dropna(subset=["incident_type_en"]).groupby("incident_type_en"):
        if len(sub) < min_n:
            continue
        out.append(
            {
                "type_en": en,
                "n": int(len(sub)),
                "severe_rate_pct": round((sub["severity"] == "severe").mean() * 100, 1),
            }
        )
    return sorted(out, key=lambda r: r["severe_rate_pct"], reverse=True)


def severe_pct_by_hour(df: pd.DataFrame) -> list[dict]:
    g = _tagged(df).groupby("hour")["severity"]
    return [
        {"hour": int(h), "severe_rate_pct": round((s == "severe").mean() * 100, 1)}
        for h, s in g
    ]


def counts_by_hour(df: pd.DataFrame) -> list[dict]:
    return [{"hour": int(h), "count": int(n)} for h, n in df.groupby("hour").size().items()]


def counts_by_dow(df: pd.DataFrame) -> list[dict]:
    return [{"dow": DOW[int(d)], "count": int(n)} for d, n in df.groupby("day_of_week").size().items()]


def yearly(df: pd.DataFrame, year_from: int = 2019, year_to: int = 2025) -> list[dict]:
    out = []
    for y, sub in df[df["year"].between(year_from, year_to)].groupby("year"):
        tag = _tagged(sub)
        out.append(
            {
                "year": int(y),
                "count": int(len(sub)),
                "severe_rate_pct": round((tag["severity"] == "severe").mean() * 100, 1)
                if len(tag)
                else None,
            }
        )
    return out


def summary(df: pd.DataFrame) -> dict:
    severe = int((df["severity"] == "severe").sum())
    minor = int((df["severity"] == "minor").sum())
    moderate = int((df["severity"] == "moderate").sum())
    tagged = severe + minor + moderate  # everything with a known severity
    return {
        "total": int(len(df)),
        "severe": severe,
        "minor": minor,
        "moderate": moderate,
        "untagged": int((df["severity"] == "untagged").sum()),
        "severe_rate_pct": round(severe / tagged * 100, 1) if tagged else None,
        "date_from": str(df["datetime"].min().date()),
        "date_to": str(df["datetime"].max().date()),
        "n_types": int(df["incident_type_en"].nunique()),
    }


def grid_blackspots(
    df: pd.DataFrame, cell: float = 0.002, min_count: int = 50, max_cells: int | None = None
) -> dict:
    """Aggregate collisions into ~220m grid cells -> GeoJSON points.

    min_count=50 keeps genuine blackspots (~2,000 cells holding ~half of all
    collisions) rather than the long tail of low-count cells — performant to
    render and a meaningful "worst spots" set. `max_cells` caps to the worst
    cells (used by the filtered map path, where counts are far lower).
    """
    g = df.copy()
    g["clat"] = (g["lat"] / cell).round() * cell
    g["clng"] = (g["lng"] / cell).round() * cell
    agg = g.groupby(["clat", "clng"]).agg(
        count=("collision_id", "size"),
        severe=("severity", lambda s: int((s == "severe").sum())),
        lat=("lat", "mean"),
        lng=("lng", "mean"),
    )
    agg = agg[agg["count"] >= min_count].copy()
    if agg.empty:  # a sparse filter can leave no qualifying cell
        return {"type": "FeatureCollection", "features": []}
    if max_cells is not None:
        agg = agg.sort_values("count", ascending=False).head(max_cells)
    # dominant type only for the surviving cells (cheap; avoids scanning all)
    cells = g.set_index(["clat", "clng"]).index.isin(agg.index)
    dom = (
        g[cells]
        .dropna(subset=["incident_type_en"])
        .groupby(["clat", "clng"])["incident_type_en"]
        .agg(lambda s: s.value_counts().idxmax())
    )
    agg["dominant_type"] = dom.reindex(agg.index)

    features = [
        {
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [round(float(r["lng"]), 6), round(float(r["lat"]), 6)],
            },
            "properties": {
                "count": int(r["count"]),
                "severe": int(r["severe"]),
                "wsum": int(r["count"] + r["severe"]),
                "dominant_type": r["dominant_type"] if pd.notna(r["dominant_type"]) else None,
            },
        }
        for _, r in agg.iterrows()
    ]
    features.sort(key=lambda f: f["properties"]["wsum"], reverse=True)
    return {"type": "FeatureCollection", "features": features}


def build() -> tuple[dict, dict]:
    df = pd.read_parquet(PARQUET)
    geo = grid_blackspots(df)
    stats = {
        "summary": summary(df),
        "severe_rate_by_type": severe_rate_by_type(df),
        "severe_pct_by_hour": severe_pct_by_hour(df),
        "counts_by_hour": counts_by_hour(df),
        "counts_by_dow": counts_by_dow(df),
        "yearly": yearly(df),
    }
    stats["summary"]["n_blackspots"] = len(geo["features"])
    STATS_JSON.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    BLACKSPOTS_GEOJSON.write_text(json.dumps(geo, ensure_ascii=False), encoding="utf-8")
    return stats, geo


if __name__ == "__main__":
    stats, geo = build()
    print(f"summary: {stats['summary']}")
    print(f"types in chart: {len(stats['severe_rate_by_type'])}")
    print(f"blackspot cells: {len(geo['features'])}")
