"""Ingest daily adjusted prices for the S&P 500 universe.

Primary source: Yahoo Finance via ``yfinance`` (batched, retried).
Fallback: Stooq daily CSV endpoint per ticker for anything Yahoo misses.

The constituent list is scraped once from Wikipedia and cached to
``config/sp500_universe.csv`` so subsequent runs (and offline CI) never
depend on Wikipedia. Output is a single long-format parquet:

    data/raw/prices.parquet  — columns: date, ticker, adj_close, volume, source

Usage:
    python data/ingest_prices.py                 # full universe, full history
    python data/ingest_prices.py --tickers AAPL MSFT --start 2020-01-01
"""

from __future__ import annotations

import argparse
import io
import sys
import time
from datetime import date
from pathlib import Path

import pandas as pd
import requests

# Allow running as a script from the repo root without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from risksense.config import REPO_ROOT, data_dir, load_config  # noqa: E402

WIKI_SP500_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
STOOQ_URL = "https://stooq.com/q/d/l/?s={symbol}.us&d1={d1}&d2={d2}&i=d"
REQUEST_TIMEOUT_S = 30


def load_universe(cache_path: Path) -> list[str]:
    """Return S&P 500 tickers, scraping Wikipedia once and caching to CSV.

    Yahoo uses ``-`` where official tickers use ``.`` (BRK.B → BRK-B);
    the cache stores the Yahoo convention.
    """
    if cache_path.exists():
        return pd.read_csv(cache_path)["ticker"].tolist()

    resp = requests.get(
        WIKI_SP500_URL,
        headers={"User-Agent": "risksense-research/0.1"},
        timeout=REQUEST_TIMEOUT_S,
    )
    resp.raise_for_status()
    tables = pd.read_html(io.StringIO(resp.text))
    constituents = tables[0]
    tickers = (
        constituents["Symbol"].astype(str).str.replace(".", "-", regex=False).tolist()
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"ticker": sorted(tickers)}).to_csv(cache_path, index=False)
    print(f"Cached {len(tickers)} S&P 500 tickers to {cache_path}")
    return sorted(tickers)


def fetch_yahoo_batch(
    tickers: list[str], start: str, end: str
) -> pd.DataFrame:
    """Download one batch of tickers from Yahoo, long format.

    Uses ``auto_adjust=True`` so ``Close`` is split/dividend adjusted —
    total-return prices, which is what risk on a portfolio needs.
    """
    import yfinance as yf

    raw = yf.download(
        tickers,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
        group_by="ticker",
        threads=True,
    )
    frames: list[pd.DataFrame] = []
    if raw is None or raw.empty:
        return pd.DataFrame(columns=["date", "ticker", "adj_close", "volume", "source"])
    for t in tickers:
        try:
            sub = raw[t][["Close", "Volume"]].dropna(subset=["Close"])
        except KeyError:
            continue
        if sub.empty:
            continue
        frames.append(
            pd.DataFrame(
                {
                    "date": sub.index,
                    "ticker": t,
                    "adj_close": sub["Close"].to_numpy(dtype=float),
                    "volume": sub["Volume"].to_numpy(dtype=float),
                    "source": "yahoo",
                }
            )
        )
    if not frames:
        return pd.DataFrame(columns=["date", "ticker", "adj_close", "volume", "source"])
    return pd.concat(frames, ignore_index=True)


def fetch_stooq(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Fallback: fetch one ticker's daily history from Stooq (no API key).

    Stooq prices are split-adjusted but NOT dividend-adjusted; rows are
    tagged ``source='stooq'`` so the data-quality report can disclose the
    mixed adjustment basis (SR 11-7 data lineage).
    """
    symbol = ticker.replace("-", "").lower()
    url = STOOQ_URL.format(
        symbol=symbol, d1=start.replace("-", ""), d2=end.replace("-", "")
    )
    resp = requests.get(url, timeout=REQUEST_TIMEOUT_S)
    if resp.status_code != 200 or "Date" not in resp.text[:100]:
        return pd.DataFrame(columns=["date", "ticker", "adj_close", "volume", "source"])
    df = pd.read_csv(io.StringIO(resp.text), parse_dates=["Date"])
    if df.empty:
        return pd.DataFrame(columns=["date", "ticker", "adj_close", "volume", "source"])
    return pd.DataFrame(
        {
            "date": df["Date"],
            "ticker": ticker,
            "adj_close": df["Close"].astype(float),
            "volume": df.get("Volume", pd.Series(dtype=float)),
            "source": "stooq",
        }
    )


def ingest(
    tickers: list[str] | None = None,
    start: str | None = None,
    end: str | None = None,
    output: Path | None = None,
) -> Path:
    """Run the full ingestion and write ``data/raw/prices.parquet``."""
    cfg = load_config("data")["prices"]
    start = start or cfg["start_date"]
    end = end or cfg.get("end_date") or date.today().isoformat()
    if tickers is None:
        tickers = load_universe(REPO_ROOT / cfg["universe_cache"])
    output = output or data_dir() / "raw" / "prices.parquet"

    batch_size = int(cfg["batch_size"])
    max_retries = int(cfg["max_retries"])
    wait_s = float(cfg["retry_wait_seconds"])

    frames: list[pd.DataFrame] = []
    got: set[str] = set()
    for i in range(0, len(tickers), batch_size):
        batch = tickers[i : i + batch_size]
        for attempt in range(1, max_retries + 1):
            try:
                df = fetch_yahoo_batch(batch, start, end)
                break
            except Exception as exc:  # network / rate-limit — retry
                print(f"  batch {i // batch_size}: attempt {attempt} failed: {exc}")
                time.sleep(wait_s * attempt)
        else:
            df = pd.DataFrame(columns=["date", "ticker", "adj_close", "volume", "source"])
        frames.append(df)
        got |= set(df["ticker"].unique())
        print(
            f"  yahoo batch {i // batch_size + 1}/"
            f"{(len(tickers) + batch_size - 1) // batch_size}: "
            f"{df['ticker'].nunique()} tickers, {len(df)} rows"
        )

    # Retry stragglers one-by-one without threading: yfinance's shared
    # sqlite cache can raise 'database is locked' under threaded batches,
    # and a solo retry almost always recovers the ticker.
    missing = [t for t in tickers if t not in got]
    if missing:
        print(f"Yahoo per-ticker retry for {len(missing)}: {missing[:10]}...")
        for t in missing:
            try:
                df = fetch_yahoo_batch([t], start, end)
            except Exception:
                df = pd.DataFrame(columns=["date", "ticker", "adj_close", "volume", "source"])
            if not df.empty:
                frames.append(df)
                got.add(t)
            time.sleep(0.5)

    # Last resort: Stooq. Note Stooq intermittently gates its CSV endpoint
    # behind a JS browser check; when that happens this stage yields nothing
    # and the coverage warning below fires instead of failing silently.
    missing = [t for t in tickers if t not in got]
    if missing:
        print(f"Stooq fallback for {len(missing)} tickers: {missing[:10]}...")
        for t in missing:
            df = fetch_stooq(t, start, end)
            if not df.empty:
                frames.append(df)
                got.add(t)
            time.sleep(0.5)  # be polite to the free endpoint

    prices = pd.concat(frames, ignore_index=True)
    prices["date"] = pd.to_datetime(prices["date"]).dt.tz_localize(None).dt.normalize()
    prices = prices.sort_values(["ticker", "date"]).reset_index(drop=True)

    coverage = len(got) / max(len(tickers), 1)
    if coverage < float(cfg["min_coverage"]):
        print(
            f"WARNING: coverage {coverage:.0%} below configured minimum "
            f"{float(cfg['min_coverage']):.0%} — check network / rate limits."
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    prices.to_parquet(output, index=False)
    print(
        f"Wrote {output}: {prices['ticker'].nunique()} tickers, "
        f"{len(prices):,} rows, {prices['date'].min().date()} → "
        f"{prices['date'].max().date()}"
    )
    return output


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tickers", nargs="*", default=None, help="override universe")
    parser.add_argument("--start", default=None, help="YYYY-MM-DD")
    parser.add_argument("--end", default=None, help="YYYY-MM-DD")
    args = parser.parse_args()
    ingest(tickers=args.tickers, start=args.start, end=args.end)


if __name__ == "__main__":
    main()
