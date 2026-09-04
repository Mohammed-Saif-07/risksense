"""Tests for the PySpark returns ETL (skipped when pyspark isn't installed)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pyspark = pytest.importorskip("pyspark")

from risksense.pipelines.returns_pipeline import (  # noqa: E402
    clean_prices,
    compute_log_returns,
    compute_portfolio_returns,
)
from risksense.pipelines.spark_session import get_spark  # noqa: E402


@pytest.fixture(scope="module")
def spark():
    s = get_spark("risksense-tests")
    yield s


@pytest.fixture(scope="module")
def price_frame(spark):
    """Two tickers, 10 days, deterministic prices; one bad row and one dupe."""
    dates = pd.bdate_range("2024-01-01", periods=10)
    rows = []
    for t, base in [("AAA", 100.0), ("BBB", 50.0)]:
        for i, d in enumerate(dates):
            rows.append((d.to_pydatetime(), t, base * (1.01 ** i), 1000.0, "yahoo"))
    rows.append((dates[3].to_pydatetime(), "AAA", -5.0, 0.0, "stooq"))  # bad price
    rows.append((dates[4].to_pydatetime(), "BBB", 50.0 * 1.01**4, 1000.0, "stooq"))  # dupe
    return spark.createDataFrame(
        rows, ["date", "ticker", "adj_close", "volume", "source"]
    )


class TestCleanPrices:
    def test_drops_nonpositive_and_duplicates(self, price_frame) -> None:
        out = clean_prices(price_frame, min_history_days=5)
        pdf = out.toPandas()
        assert (pdf["adj_close"] > 0).all()
        assert not pdf.duplicated(["date", "ticker"]).any()
        assert set(pdf["ticker"]) == {"AAA", "BBB"}

    def test_min_history_filter(self, price_frame) -> None:
        out = clean_prices(price_frame, min_history_days=11)
        assert out.count() == 0


class TestReturns:
    def test_log_return_values(self, price_frame) -> None:
        cleaned = clean_prices(price_frame, min_history_days=5)
        returns = compute_log_returns(cleaned, max_abs_return=0.60).toPandas()
        # Constant 1% growth => log return = ln(1.01) every day.
        aaa = returns[returns["ticker"] == "AAA"]["log_return"]
        assert np.allclose(aaa, np.log(1.01), atol=1e-12)
        assert len(aaa) == 9  # first day has no lag

    def test_winsorisation_flags_not_drops(self, spark) -> None:
        dates = pd.bdate_range("2024-01-01", periods=3)
        rows = [
            (dates[0].to_pydatetime(), "CCC", 100.0, 0.0, "yahoo"),
            (dates[1].to_pydatetime(), "CCC", 300.0, 0.0, "yahoo"),  # +110% log
            (dates[2].to_pydatetime(), "CCC", 300.0, 0.0, "yahoo"),
        ]
        df = spark.createDataFrame(rows, ["date", "ticker", "adj_close", "volume", "source"])
        out = compute_log_returns(df, max_abs_return=0.60).toPandas()
        flagged = out[out["winsorized"]]
        assert len(flagged) == 1
        assert flagged["log_return"].iloc[0] == pytest.approx(0.60)

    def test_equal_weight_portfolio_is_mean(self, price_frame) -> None:
        cleaned = clean_prices(price_frame, min_history_days=5)
        returns = compute_log_returns(cleaned, max_abs_return=0.60)
        port = compute_portfolio_returns(returns, weights=None).toPandas()
        assert np.allclose(port["portfolio_return"], np.log(1.01), atol=1e-12)
        assert (port["n_constituents"] == 2).all()

    def test_explicit_weights(self, price_frame) -> None:
        cleaned = clean_prices(price_frame, min_history_days=5)
        returns = compute_log_returns(cleaned, max_abs_return=0.60)
        port = compute_portfolio_returns(
            returns, weights={"AAA": 0.8, "BBB": 0.2}
        ).toPandas()
        # Both tickers return ln(1.01), so any weighting gives ln(1.01).
        assert np.allclose(port["portfolio_return"], np.log(1.01), atol=1e-12)
