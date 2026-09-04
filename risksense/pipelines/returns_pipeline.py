"""PySpark ETL: raw prices → cleaned log returns → portfolio P&L.

Steps
-----
1. Read the long-format price parquet produced by ``data/ingest_prices.py``
   (columns: date, ticker, adj_close, volume, source).
2. Data-quality filters (SR 11-7 §data quality): drop non-positive prices,
   deduplicate (date, ticker), require a minimum history per ticker.
3. Daily log return per ticker via a lag window.
4. Winsorise |r| > ``max_abs_return`` (data errors, not market moves —
   threshold in ``config/model_params.yaml``, flagged not silently dropped).
5. Portfolio return: weight-averaged constituent returns (equal-weight,
   daily-rebalanced by default; weights configurable in
   ``config/portfolio.yaml``).
6. Write ``returns.parquet`` (per-ticker) and ``portfolio_returns.parquet``.

Run:  ``python -m risksense.pipelines.returns_pipeline``
"""

from __future__ import annotations

from pathlib import Path

from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as F

from risksense.config import data_dir, load_config
from risksense.pipelines.spark_session import get_spark


def clean_prices(prices: DataFrame, min_history_days: int) -> DataFrame:
    """Apply data-quality filters to the raw price frame.

    Drops non-positive prices, deduplicates (date, ticker) keeping the
    first source, and removes tickers with fewer than ``min_history_days``
    observations (too short to estimate a 250-day VaR window).
    """
    df = prices.filter(F.col("adj_close") > 0).dropDuplicates(["date", "ticker"])
    counts = df.groupBy("ticker").agg(F.count("*").alias("n_obs"))
    keep = counts.filter(F.col("n_obs") >= min_history_days).select("ticker")
    return df.join(keep, on="ticker", how="inner")


def compute_log_returns(prices: DataFrame, max_abs_return: float) -> DataFrame:
    """Daily log return per ticker, with data-error winsorisation.

    Returns with ``|r| > max_abs_return`` are clipped and flagged in the
    ``winsorized`` column rather than silently dropped, so the DQ report can
    surface them (SR 11-7 data-quality traceability).
    """
    w = Window.partitionBy("ticker").orderBy("date")
    df = prices.withColumn("prev_close", F.lag("adj_close").over(w))
    df = df.filter(F.col("prev_close").isNotNull())
    df = df.withColumn("raw_return", F.log(F.col("adj_close") / F.col("prev_close")))
    df = df.withColumn("winsorized", F.abs(F.col("raw_return")) > F.lit(max_abs_return))
    df = df.withColumn(
        "log_return",
        F.when(
            F.col("winsorized"),
            F.signum("raw_return") * F.lit(max_abs_return),
        ).otherwise(F.col("raw_return")),
    )
    return df.select("date", "ticker", "adj_close", "log_return", "winsorized")


def compute_portfolio_returns(
    returns: DataFrame, weights: dict[str, float] | None
) -> DataFrame:
    """Aggregate constituent returns to a single portfolio return series.

    ``weights=None`` means equal-weight across whatever tickers trade that
    day (daily rebalancing). Explicit weights are renormalised over the
    tickers present each day so missing data doesn't leak weight.
    """
    if weights:
        wdf = returns.sparkSession.createDataFrame(
            [(k, float(v)) for k, v in weights.items()], ["ticker", "w"]
        )
        df = returns.join(wdf, on="ticker", how="inner")
        agg = df.groupBy("date").agg(
            (F.sum(F.col("w") * F.col("log_return")) / F.sum("w")).alias(
                "portfolio_return"
            ),
            F.count("*").alias("n_constituents"),
        )
    else:
        agg = returns.groupBy("date").agg(
            F.avg("log_return").alias("portfolio_return"),
            F.count("*").alias("n_constituents"),
        )
    return agg.orderBy("date")


def run(
    prices_path: str | Path | None = None,
    output_dir: str | Path | None = None,
) -> Path:
    """Execute the full ETL and return the portfolio returns parquet path."""
    params = load_config("model_params")["data_quality"]
    portfolio_cfg = load_config("portfolio")

    d = data_dir()
    prices_path = Path(prices_path) if prices_path else d / "raw" / "prices.parquet"
    output_dir = Path(output_dir) if output_dir else d / "processed"
    output_dir.mkdir(parents=True, exist_ok=True)

    if not prices_path.exists():
        raise FileNotFoundError(
            f"{prices_path} not found — run `python data/ingest_prices.py` first."
        )

    spark = get_spark("risksense-returns")
    prices = spark.read.parquet(str(prices_path))

    cleaned = clean_prices(prices, int(params["min_history_days"]))
    returns = compute_log_returns(cleaned, float(params["max_abs_return"]))

    weights = portfolio_cfg.get("weights")  # None => equal-weight
    portfolio = compute_portfolio_returns(returns, weights)

    returns_out = output_dir / "returns.parquet"
    portfolio_out = output_dir / "portfolio_returns.parquet"
    returns.write.mode("overwrite").parquet(str(returns_out))
    portfolio.write.mode("overwrite").parquet(str(portfolio_out))

    n_days = portfolio.count()
    n_tickers = returns.select("ticker").distinct().count()
    print(
        f"Wrote {returns_out} ({n_tickers} tickers) and {portfolio_out} ({n_days} days)"
    )
    return portfolio_out


if __name__ == "__main__":
    run()
