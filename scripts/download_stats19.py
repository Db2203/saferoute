"""Download the UK STATS19 road safety dataset (collision/vehicle/casualty + data guide).

We pull the "last 5 years" combined CSVs since DfT no longer ships individual files
for 2019 on this page, and these cover 2019-latest in one go.

Usage:
    python scripts/download_stats19.py            # download missing files
    python scripts/download_stats19.py --force    # redownload everything
    python scripts/download_stats19.py --list     # print URLs and exit
"""
from __future__ import annotations

import argparse
from pathlib import Path

import requests
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent
DEST_DIR = REPO_ROOT / "data" / "raw" / "stats19"

BASE = "https://data.dft.gov.uk/road-accidents-safety-data"
GUIDE_BASE = "https://assets.publishing.service.gov.uk/media"

FILES: dict[str, str] = {
    "collision-last-5-years.csv": f"{BASE}/dft-road-casualty-statistics-collision-last-5-years.csv",
    "vehicle-last-5-years.csv": f"{BASE}/dft-road-casualty-statistics-vehicle-last-5-years.csv",
    "casualty-last-5-years.csv": f"{BASE}/dft-road-casualty-statistics-casualty-last-5-years.csv",
    "data-guide-2024.xlsx": f"{GUIDE_BASE}/691c6440e39a085bda43eed6/dft-road-casualty-statistics-road-safety-open-dataset-data-guide-2024.xlsx",
}


def download(url: str, dest: Path, *, force: bool = False) -> None:
    if dest.exists() and not force:
        print(f"skip {dest.name} ({dest.stat().st_size / 1e6:.1f} MB on disk)")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("Content-Length", 0))
        with open(dest, "wb") as f, tqdm(
            total=total or None, unit="B", unit_scale=True, desc=dest.name
        ) as pbar:
            for chunk in r.iter_content(chunk_size=64 * 1024):
                f.write(chunk)
                pbar.update(len(chunk))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="redownload even if file exists")
    parser.add_argument("--list", action="store_true", help="print URLs and exit")
    args = parser.parse_args()

    if args.list:
        for name, url in FILES.items():
            print(f"{url}\n  -> {DEST_DIR / name}")
        return

    for name, url in FILES.items():
        download(url, DEST_DIR / name, force=args.force)


if __name__ == "__main__":
    main()
