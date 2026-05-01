"""ORM models for the SafeRoute Postgres+PostGIS database.

Each row of `accidents` is one STATS19 collision; geom is the (lng, lat) point
in WGS84 (SRID 4326). `aadt_points` holds the deduped DfT traffic counts for
London. Hotspot/risk-score tables come in later stages.
"""
from __future__ import annotations

from datetime import datetime

from geoalchemy2 import Geometry
from sqlalchemy import DateTime, Float, Integer, String
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class Accident(Base):
    __tablename__ = "accidents"

    collision_index: Mapped[str] = mapped_column(String, primary_key=True)
    geom: Mapped[object] = mapped_column(Geometry("POINT", srid=4326), nullable=False)

    occurred_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    hour: Mapped[int | None] = mapped_column(Integer)
    day_of_week: Mapped[int | None] = mapped_column(Integer)
    month: Mapped[int | None] = mapped_column(Integer)

    severity: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    severity_label: Mapped[str | None] = mapped_column(String)

    weather: Mapped[int | None] = mapped_column(Integer)
    weather_label: Mapped[str | None] = mapped_column(String)
    road_type: Mapped[int | None] = mapped_column(Integer)
    speed_limit: Mapped[int | None] = mapped_column(Integer)
    light_conditions: Mapped[int | None] = mapped_column(Integer)
    road_surface_conditions: Mapped[int | None] = mapped_column(Integer)
    urban_or_rural_area: Mapped[int | None] = mapped_column(Integer)

    number_of_vehicles: Mapped[int | None] = mapped_column(Integer)
    number_of_casualties: Mapped[int | None] = mapped_column(Integer)


class AADTPoint(Base):
    __tablename__ = "aadt_points"

    count_point_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    geom: Mapped[object] = mapped_column(Geometry("POINT", srid=4326), nullable=False)

    year: Mapped[int | None] = mapped_column(Integer)
    road_name: Mapped[str | None] = mapped_column(String)
    road_category: Mapped[str | None] = mapped_column(String)
    road_type: Mapped[str | None] = mapped_column(String)
    local_authority_name: Mapped[str | None] = mapped_column(String)
    region_name: Mapped[str | None] = mapped_column(String)

    link_length_km: Mapped[float | None] = mapped_column(Float)
    all_motor_vehicles: Mapped[int | None] = mapped_column(Integer)
