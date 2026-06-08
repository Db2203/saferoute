"""On-the-fly analytics over the full collision DataFrame, computed under an
optional filter (type / year / hour / day-of-week / severity). Held in memory
at startup (~107 MB), a full rebuild of every breakdown takes ~36 ms, so the
frontend can cross-filter every panel from a single click.

Breakdowns mirror the precomputed stats.json keys plus by_month and the
hour x day-of-week matrix the heatmap needs. Every breakdown carries the
sample size n so the UI can flag thin slices instead of drawing noise."""
from __future__ import annotations

import pandas as pd

from app.data.aggregates import DOW, _tagged, grid_blackspots

FILTER_KEYS = ("type", "year", "hour", "dow", "severity")


def dow_index(dow) -> int:
    """Accept a weekday name ('Mon'..'Sun') or an int 0-6; raise on anything else."""
    if isinstance(dow, str) and dow in DOW:
        return DOW.index(dow)
    i = int(dow)
    if 0 <= i <= 6:
        return i
    raise ValueError(f"bad day-of-week: {dow!r}")


def apply_filters(df: pd.DataFrame, filters: dict) -> pd.DataFrame:
    m = pd.Series(True, index=df.index)
    if filters.get("type") is not None:
        m &= df["incident_type_en"] == filters["type"]
    if filters.get("year") is not None:
        m &= df["year"] == int(filters["year"])
    if filters.get("hour") is not None:
        m &= df["hour"] == int(filters["hour"])
    if filters.get("dow") is not None:
        m &= df["day_of_week"] == dow_index(filters["dow"])
    if filters.get("severity") is not None:
        m &= df["severity"] == filters["severity"]
    return df[m]


def _severe_rate(severity: pd.Series) -> float | None:
    tag = severity[severity != "untagged"]
    return round((tag == "severe").mean() * 100, 1) if len(tag) else None


def _summary(df: pd.DataFrame) -> dict:
    severe = int((df["severity"] == "severe").sum())
    minor = int((df["severity"] == "minor").sum())
    moderate = int((df["severity"] == "moderate").sum())
    tagged = severe + minor + moderate  # everything with a known severity
    out = {
        "total": int(len(df)),
        "severe": severe,
        "minor": minor,
        "moderate": moderate,
        "untagged": int((df["severity"] == "untagged").sum()),
        "severe_rate_pct": round(severe / tagged * 100, 1) if tagged else None,
        "date_from": None,
        "date_to": None,
    }
    if len(df):
        out["date_from"] = str(df["datetime"].min().date())
        out["date_to"] = str(df["datetime"].max().date())
    return out


def _by_type(df: pd.DataFrame, min_n: int = 30, top: int = 15) -> list[dict]:
    """Mapped types only, sorted by severe-rate (the 'what's dangerous' view).
    min_n keeps the rate stable; if a heavy filter leaves nothing above it,
    fall back to whatever is present so the panel never goes blank."""
    rows = []
    for typ_en, g in df.dropna(subset=["incident_type_en"]).groupby("incident_type_en"):
        rows.append(
            {"type_en": typ_en, "n": int(len(g)), "severe_rate_pct": _severe_rate(g["severity"])}
        )
    above = [r for r in rows if r["n"] >= min_n] or rows
    above.sort(key=lambda r: (r["severe_rate_pct"] is not None, r["severe_rate_pct"] or 0), reverse=True)
    return above[:top]


def _by_hour(df: pd.DataFrame) -> list[dict]:
    counts = df.groupby("hour").size()
    tag = _tagged(df)
    sev = (tag["severity"] == "severe").groupby(tag["hour"]).mean() * 100 if len(tag) else pd.Series(dtype=float)
    return [
        {
            "hour": h,
            "count": int(counts.get(h, 0)),
            "severe_rate_pct": round(float(sev[h]), 1) if h in sev.index else None,
        }
        for h in range(24)
    ]


def _by_dow(df: pd.DataFrame) -> list[dict]:
    counts = df.groupby("day_of_week").size()
    return [{"dow": DOW[d], "count": int(counts.get(d, 0))} for d in range(7)]


def _by_month(df: pd.DataFrame) -> list[dict]:
    counts = df.groupby("month").size()
    return [{"month": m, "count": int(counts.get(m, 0))} for m in range(1, 13)]


def _by_year(df: pd.DataFrame) -> list[dict]:
    out = []
    for y, sub in df.groupby("year"):
        out.append(
            {"year": int(y), "count": int(len(sub)), "severe_rate_pct": _severe_rate(sub["severity"])}
        )
    return out


def _hour_dow(df: pd.DataFrame) -> list[dict]:
    counts = df.groupby(["hour", "day_of_week"]).size()
    return [
        {"hour": h, "dow": DOW[d], "count": int(counts.get((h, d), 0))}
        for h in range(24)
        for d in range(7)
    ]


def compute(df: pd.DataFrame, filters: dict | None = None) -> dict:
    sub = apply_filters(df, filters or {})
    return {
        "filters": {k: (filters or {}).get(k) for k in FILTER_KEYS},
        "summary": _summary(sub),
        "by_type": _by_type(sub),
        "by_hour": _by_hour(sub),
        "by_dow": _by_dow(sub),
        "by_month": _by_month(sub),
        "by_year": _by_year(sub),
        "hour_dow": _hour_dow(sub),
    }


def filtered_grid(df: pd.DataFrame, filters: dict, max_cells: int = 400) -> dict:
    """Recompute the blackspot grid over a filtered subset, capped to the worst
    cells so the map stays readable (filtered counts are far lower than the
    precomputed >=50 threshold, so we take the top cells by weight instead)."""
    sub = apply_filters(df, filters)
    geo = grid_blackspots(sub, min_count=3)
    geo["features"] = geo["features"][:max_cells]
    return geo
