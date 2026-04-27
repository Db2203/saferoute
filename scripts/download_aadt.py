"""Download the UK DfT AADF (Annual Average Daily Flow) traffic count dataset.

This is GB-wide point-level traffic counts (~6M rows, 2000-latest). We'll filter
to the London bounding box during preprocessing.

Usage:
    python scripts/download_aadt.py              # download + extract if missing
    python scripts/download_aadt.py --force      # redownload + re-extract
    python scripts/download_aadt.py --list       # print URL and exit
    python scripts/download_aadt.py --no-extract # download only, skip unzip
"""
from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

import requests
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent
DEST_DIR = REPO_ROOT / "data" / "raw" / "aadt"
ZIP_URL = "https://storage.googleapis.com/dft-statistics/road-traffic/downloads/data-gov-uk/dft_traffic_counts_aadf.zip"
ZIP_NAME = "dft_traffic_counts_aadf.zip"


def download(url: str, dest: Path, *, force: bool = False) -> bool:
    if dest.exists() and not force:
        print(f"skip {dest.name} ({dest.stat().st_size / 1e6:.1f} MB on disk)")
        return False
    dest.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        total = int(r.headers.get("Content-Length", 0))
        with open(dest, "wb") as f, tqdm(
            total=total or None, unit="B", unit_scale=True, desc=dest.name
        ) as pbar:
            for chunk in r.iter_content(chunk_size=64 * 1024):
                f.write(chunk)
                pbar.update(len(chunk))
    return True


def extract(zip_path: Path) -> None:
    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
        print(f"extracting {len(names)} file(s) from {zip_path.name}")
        zf.extractall(zip_path.parent)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="redownload + re-extract")
    parser.add_argument("--list", action="store_true", help="print URL and exit")
    parser.add_argument("--no-extract", action="store_true", help="skip unzip step")
    args = parser.parse_args()

    zip_path = DEST_DIR / ZIP_NAME

    if args.list:
        print(f"{ZIP_URL}\n  -> {zip_path}")
        return

    fetched = download(ZIP_URL, zip_path, force=args.force)
    if not args.no_extract and (fetched or args.force):
        extract(zip_path)


if __name__ == "__main__":
    main()
