# scripts

One-off data downloaders. Run these from the repo root using the backend venv's Python.

## STATS19 (road safety / collision data)

```
backend/.venv/Scripts/python scripts/download_stats19.py
```

Pulls the "last 5 years" collision, vehicle, and casualty CSVs from data.dft.gov.uk into `data/raw/stats19/`, plus the column data guide (xlsx). Idempotent — already-downloaded files are skipped. Use `--force` to re-fetch, `--list` to preview URLs.

Combined size ~150-200 MB.

## DfT AADF (traffic counts)

```
backend/.venv/Scripts/python scripts/download_aadt.py
```

Pulls the GB-wide AADF zip from storage.googleapis.com into `data/raw/aadt/` and extracts it. Idempotent. Use `--force` to re-fetch + re-extract, `--no-extract` to skip the unzip, `--list` to preview the URL.

Zip is ~100-200 MB. Extracted CSV covers all GB count points 2000-latest; we filter to London during preprocessing (Stage 7).

## Notes

- `data/raw/` is gitignored. Files stay local.
- Both scripts are quiet on success and use tqdm progress bars during download.
- If a URL 404s, DfT may have moved it — fix the constant at the top of the script.
