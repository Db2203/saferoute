"""Create / refresh database schema.

Idempotent: enables PostGIS if missing, creates tables if missing. Existing
data isn't touched. To start fresh, drop the database in psql and re-run.

    cd backend
    .venv/Scripts/python -m scripts.init_db
"""
from __future__ import annotations

from sqlalchemy import text

from app.db.connection import engine
from app.db.models import Base


def main() -> None:
    with engine.begin() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS postgis"))

    Base.metadata.create_all(engine)

    with engine.connect() as conn:
        rows = conn.execute(
            text("SELECT tablename FROM pg_tables WHERE schemaname = 'public' ORDER BY tablename")
        ).fetchall()
    print("tables in public schema:")
    for (name,) in rows:
        print(f"  - {name}")


if __name__ == "__main__":
    main()
