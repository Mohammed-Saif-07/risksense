"""Ingest macro/rates series from FRED (no API key required).

Uses the public ``fredgraph.csv`` endpoint, which serves any FRED series as
CSV without authentication — keeping the pipeline zero-credential. Series
list lives in ``config/data.yaml``.

Output: data/raw/macro.parquet — columns: date, series_id, name, value

Usage: python data/ingest_macro.py
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from risksense.config import data_dir, load_config  # noqa: E402

# cosd pins the observation start date — without it fredgraph defaults some
# series (e.g. the BofA OAS family) to a recent window instead of full history.
FRED_CSV_URL = (
    "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}&cosd={start}"
)
REQUEST_TIMEOUT_S = 30


def fetch_series(series_id: str, name: str, start: str) -> pd.DataFrame:
    """Fetch one FRED series as a tidy frame (date, series_id, name, value)."""
    resp = requests.get(
        FRED_CSV_URL.format(series_id=series_id, start=start),
        timeout=REQUEST_TIMEOUT_S,
    )
    resp.raise_for_status()
    df = pd.read_csv(io.StringIO(resp.text))
    date_col, value_col = df.columns[0], df.columns[1]
    df[value_col] = pd.to_numeric(df[value_col], errors="coerce")  # '.' => NaN
    out = pd.DataFrame(
        {
            "date": pd.to_datetime(df[date_col]),
            "series_id": series_id,
            "name": name,
            "value": df[value_col],
        }
    ).dropna(subset=["value"])
    return out


def ingest(output: Path | None = None) -> Path:
    """Fetch all configured series and write ``data/raw/macro.parquet``."""
    cfg = load_config("data")
    series_map: dict[str, str] = cfg["macro"]["series"]
    start: str = cfg["prices"]["start_date"]  # align macro history with prices
    output = output or data_dir() / "raw" / "macro.parquet"

    frames = []
    for series_id, name in series_map.items():
        df = fetch_series(series_id, name, start)
        frames.append(df)
        print(f"  {series_id} ({name}): {len(df):,} observations")

    macro = pd.concat(frames, ignore_index=True).sort_values(["series_id", "date"])
    output.parent.mkdir(parents=True, exist_ok=True)
    macro.to_parquet(output, index=False)
    print(f"Wrote {output}: {macro['series_id'].nunique()} series, {len(macro):,} rows")
    return output


if __name__ == "__main__":
    ingest()
