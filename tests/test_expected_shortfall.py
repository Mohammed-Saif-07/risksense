"""Tests for ES utilities: Euler allocation and Acerbi-Székely Z2."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from risksense.var.expected_shortfall import es_backtest_z2, euler_es_allocation


@pytest.fixture(scope="module")
def wide_window() -> pd.DataFrame:
    rng = np.random.default_rng(21)
    n_days, n_assets = 250, 30
    cov = 0.0001 * (0.25 * np.ones((n_assets, n_assets)) + 0.75 * np.eye(n_assets))
    x = rng.multivariate_normal(np.zeros(n_assets), cov, size=n_days)
    dates = pd.bdate_range("2024-01-01", periods=n_days)
    return pd.DataFrame(x, index=dates, columns=[f"T{i}" for i in range(n_assets)])


class TestEulerAllocation:
    def test_contributions_sum_to_portfolio_es(self, wide_window: pd.DataFrame) -> None:
        from sklearn.covariance import LedoitWolf

        contrib = euler_es_allocation(wide_window, es_level=0.975)
        n = wide_window.shape[1]
        w = np.full(n, 1.0 / n)
        mu = wide_window.mean().to_numpy()
        cov = LedoitWolf().fit(wide_window.to_numpy()).covariance_
        sigma_p = np.sqrt(w @ cov @ w)
        z = stats.norm.ppf(0.975)
        total_es = -w @ mu + sigma_p * stats.norm.pdf(z) / 0.025
        assert contrib.sum() == pytest.approx(total_es, rel=1e-10)
        assert len(contrib) == n

    def test_sorted_descending(self, wide_window: pd.DataFrame) -> None:
        contrib = euler_es_allocation(wide_window)
        assert (contrib.diff().dropna() <= 1e-15).all()

    def test_too_few_tickers_raise(self, wide_window: pd.DataFrame) -> None:
        with pytest.raises(ValueError):
            euler_es_allocation(wide_window.iloc[:, :1])


class TestAcerbiSzekelyZ2:
    @staticmethod
    def _normal_setup(n: int, seed: int = 5) -> tuple[np.ndarray, float, float]:
        rng = np.random.default_rng(seed)
        sigma = 0.01
        returns = rng.normal(0.0, sigma, size=n)
        q = 0.975
        var975 = sigma * stats.norm.ppf(q)
        es975 = sigma * stats.norm.pdf(stats.norm.ppf(q)) / (1 - q)
        return returns, var975, es975

    def test_correct_forecasts_not_rejected(self) -> None:
        returns, var975, es975 = self._normal_setup(20_000)
        res = es_backtest_z2(
            returns, np.full_like(returns, var975), np.full_like(returns, es975)
        )
        assert abs(res.z2_stat) < 0.1
        assert not res.reject_h0

    def test_understated_es_rejected(self) -> None:
        returns, var975, es975 = self._normal_setup(20_000)
        res = es_backtest_z2(
            returns,
            np.full_like(returns, var975),
            np.full_like(returns, es975 * 0.5),  # promises half the true tail
        )
        assert res.z2_stat > 0.5
        assert res.reject_h0

    def test_nonpositive_es_raises(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            es_backtest_z2(np.zeros(10), np.full(10, 0.02), np.zeros(10))

    def test_misaligned_raises(self) -> None:
        with pytest.raises(ValueError, match="align"):
            es_backtest_z2(np.zeros(10), np.full(9, 0.02), np.full(10, 0.03))
