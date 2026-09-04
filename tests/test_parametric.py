"""Tests for the parametric VaR/ES engine (normal, Student-t, Ledoit-Wolf)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from risksense.var.base import VAR_RESULT_COLUMNS, validate_result_frame
from risksense.var.parametric import (
    fit_dof_from_kurtosis,
    ledoit_wolf_sigma_series,
    parametric_var_es,
)


class TestDofFit:
    def test_moment_inversion(self) -> None:
        # Excess kurtosis of t(10) is 6/(10-4) = 1.0 → nu = 10.
        k = pd.Series([1.0, 6.0, 0.5])
        nu = fit_dof_from_kurtosis(k, (3.0, 50.0))
        assert nu.tolist() == pytest.approx([10.0, 5.0, 16.0])

    def test_thin_tails_map_to_upper_bound(self) -> None:
        nu = fit_dof_from_kurtosis(pd.Series([-0.2, 0.0]), (3.0, 50.0))
        assert (nu == 50.0).all()

    def test_clipping(self) -> None:
        # Moment inversion approaches 4 from above as kurtosis grows, so a
        # lower bound above 4 binds for extreme kurtosis...
        assert fit_dof_from_kurtosis(pd.Series([1e9]), (4.5, 50.0)).iloc[0] == 4.5
        # ...and tiny positive kurtosis maps to the upper bound.
        assert fit_dof_from_kurtosis(pd.Series([1e-9]), (3.0, 50.0)).iloc[0] == 50.0


class TestNormalEngine:
    def test_matches_closed_form(self, normal_returns: pd.Series) -> None:
        out = parametric_var_es(normal_returns, distribution="normal", window=250)
        # Manual check on the first forecast date (naive sigma fallback).
        win = normal_returns.iloc[:250]
        z99 = stats.norm.ppf(0.99)
        expected = -win.mean() + win.std(ddof=1) * z99
        assert out["var"].iloc[0] == pytest.approx(expected, rel=1e-10)
        assert out["method"].iloc[0] == "parametric_normal_naive"

    def test_schema_and_coherence(self, normal_returns: pd.Series) -> None:
        out = parametric_var_es(normal_returns, distribution="normal")
        assert list(out.columns) == VAR_RESULT_COLUMNS
        validate_result_frame(out)
        # Normal: ES97.5 ≈ VaR99 up to a small gap (2.338σ vs 2.326σ).
        assert (out["es"] >= out["var_at_es_level"]).all()

    def test_exception_rate_near_nominal(self, normal_returns: pd.Series) -> None:
        from risksense.var.base import realized_exceptions

        out = parametric_var_es(normal_returns, distribution="normal")
        joined = realized_exceptions(out, normal_returns)
        assert 0.002 < joined["exception"].mean() < 0.025


class TestStudentTEngine:
    def test_es_matches_numerical_integration(self) -> None:
        # Fixed nu: closed-form t ES vs numeric tail integral of the quantile.
        nu, q = 6.0, 0.975
        tq = stats.t.ppf(q, nu)
        closed = stats.t.pdf(tq, nu) / (1 - q) * (nu + tq**2) / (nu - 1)
        from scipy import integrate

        numeric = integrate.quad(lambda u: stats.t.ppf(u, nu), q, 1.0, epsabs=1e-10)[
            0
        ] / (1 - q)
        assert closed == pytest.approx(numeric, rel=1e-6)

    def test_t_var_exceeds_normal_var(self, fat_tailed_returns: pd.Series) -> None:
        n = parametric_var_es(fat_tailed_returns, distribution="normal")
        t = parametric_var_es(fat_tailed_returns, distribution="t")
        # Same sigma, fatter tail at 99% → t VaR above normal VaR on average.
        assert t["var"].mean() > n["var"].mean()
        assert t["method"].iloc[0] == "parametric_t_naive"


class TestLedoitWolf:
    @pytest.fixture(scope="class")
    def wide(self, rng_class=None) -> pd.DataFrame:
        rng = np.random.default_rng(7)
        n_days, n_assets = 600, 40
        cov = 0.0001 * (0.3 * np.ones((n_assets, n_assets)) + 0.7 * np.eye(n_assets))
        x = rng.multivariate_normal(np.zeros(n_assets), cov, size=n_days)
        dates = pd.bdate_range("2020-01-01", periods=n_days)
        return pd.DataFrame(x, index=dates, columns=[f"T{i}" for i in range(n_assets)])

    def test_sigma_close_to_truth(self, wide: pd.DataFrame) -> None:
        sigma = ledoit_wolf_sigma_series(wide, window=250, refit_days=21)
        n = wide.shape[1]
        w = np.full(n, 1.0 / n)
        cov_true = 0.0001 * (0.3 * np.ones((n, n)) + 0.7 * np.eye(n))
        true_sigma = np.sqrt(w @ cov_true @ w)
        assert sigma.iloc[0] == pytest.approx(true_sigma, rel=0.15)
        assert len(sigma) == 600 - 250

    def test_refit_schedule_holds_sigma_constant(self, wide: pd.DataFrame) -> None:
        sigma = ledoit_wolf_sigma_series(wide, window=250, refit_days=21)
        # Within a refit block sigma is constant; across blocks it changes.
        assert sigma.iloc[0] == sigma.iloc[20]
        assert sigma.iloc[0] != sigma.iloc[21]

    def test_engine_uses_lw_sigma(self, wide: pd.DataFrame) -> None:
        port = wide.mean(axis=1)
        out = parametric_var_es(
            port, wide_returns=wide, distribution="normal", window=250
        )
        assert out["method"].iloc[0] == "parametric_normal"
        validate_result_frame(out)
