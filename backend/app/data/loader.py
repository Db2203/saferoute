"""STATS19 + AADT data loaders.

Reads the raw CSVs DfT publishes. We don't pre-join collision/vehicle/casualty
because the collision file already carries number_of_vehicles and
number_of_casualties aggregates — that's enough for risk scoring + clustering.
The other two are exposed via load_vehicles / load_casualties for analyses
that need per-vehicle or per-casualty rows.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RAW_DIR = REPO_ROOT / "data" / "raw"


def load_stats19(raw_dir: Path | str | None = None) -> pd.DataFrame:
    return _read_one(_resolve(raw_dir, "stats19"), "collision*.csv")


def load_vehicles(raw_dir: Path | str | None = None) -> pd.DataFrame:
    return _read_one(_resolve(raw_dir, "stats19"), "vehicle*.csv")


def load_casualties(raw_dir: Path | str | None = None) -> pd.DataFrame:
    return _read_one(_resolve(raw_dir, "stats19"), "casualty*.csv")


def load_aadt(raw_dir: Path | str | None = None) -> pd.DataFrame:
    df = _read_one(_resolve(raw_dir, "aadt"), "*aadf*.csv")
    df = df.sort_values("year").drop_duplicates("count_point_id", keep="last")
    df = df.dropna(subset=["latitude", "longitude"])
    return df.reset_index(drop=True)


def _resolve(raw_dir: Path | str | None, sub: str) -> Path:
    if raw_dir is None:
        return DEFAULT_RAW_DIR / sub
    p = Path(raw_dir)
    return p if p.name == sub else p / sub


def _read_one(directory: Path, pattern: str) -> pd.DataFrame:
    matches = sorted(directory.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"no file matching {pattern!r} in {directory}")
    if len(matches) > 1:
        raise ValueError(f"multiple files match {pattern!r} in {directory}: {[m.name for m in matches]}")
    return pd.read_csv(matches[0], low_memory=False)
