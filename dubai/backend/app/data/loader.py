from pathlib import Path

import pandas as pd

DUBAI_ROOT = Path(__file__).resolve().parents[3]
RAW_CSV = DUBAI_ROOT / "data" / "raw" / "dp_traffic_incidents.csv"


def load_incidents(path: Path | None = None) -> pd.DataFrame:
    """Read the raw Dubai Pulse traffic-incidents snapshot.

    Columns: acci_id, acci_time, acci_name, acci_x, acci_y, load_timestamp.
    """
    return pd.read_csv(path or RAW_CSV, low_memory=False)
