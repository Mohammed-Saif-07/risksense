"""Tests for the Monte Carlo VaR/ES engine (normal, t, GARCH-t)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from risksense.var.base import VAR_RESULT_COLUMNS, validate_result_frame
from risksense.var.monte_carlo import monte_carlo_var_es
from risksense.var.parametric import parametric_var_es


class TestNormalMC:
    def test_converges_to_parametric(self, normal_returns: pd.Series) -> None:
        # With the same (mu, sigma) estimates, MC-normal should agree with
        # closed-form parametric-normal up to simulation noise.
        mc = monte_carlo_var_es(
            normal_returns, distribution="normal", n_paths=50_000, seed=1
        )
        par = parametric_var_es(normal_returns, distribution="normal")
        rel_gap = ((mc["var"] - par["var"]).abs() / par["var"]).mean()
        assert rel_gap < 0.03

    def test_schema(self, normal_returns: pd.Series) -> None:
        out = monte_carlo_var_es(normal_returns, distribution="normal", n_paths=2_000)
        assert list(out.columns) == VAR_RESULT_COLUMNS
        validate_result_frame(out)
        assert (out["method"] == "monte_carlo_normal").all()

    def test_seed_reproducibility(self, normal_returns: pd.Series) -> None:
        a = monte_carlo_var_es(normal_returns, distribution="normal", n_paths=2_000, seed=9)
        b = monte_carlo_var_es(normal_returns, distribution="normal", n_paths=2_000, seed=9)
        pd.testing.assert_frame_equal(a, b)

    def test_unknown_distribution_raises(self, normal_returns: pd.Series) -> None:
        with pytest.raises(ValueError, match="distribution"):
            monte_carlo_var_es(normal_returns, distribution="cauchy")


class TestStudentTMC:
    def test_fatter_tail_than_normal(self, fat_tailed_returns: pd.Series) -> None:
        mc_n = monte_carlo_var_es(
            fat_tailed_returns, distribution="normal", n_paths=20_000, seed=2
        )
        mc_t = monte_carlo_var_es(
            fat_tailed_returns, distribution="t", n_paths=20_000, seed=2
        )
        assert mc_t["var"].mean() > mc_n["var"].mean()


class TestGarchMC:
    @pytest.fixture(scope="class")
    def garch_returns(self) -> pd.Series:
        # Simulate a GARCH(1,1)-t path so the engine sees real clustering.
        rng = np.random.default_rng(11)
        n = 1_400
        omega, alpha, beta, nu = 2e-6, 0.08, 0.90, 6.0
        sig2 = omega / (1 - alpha - beta)
        r = np.empty(n)
        for i in range(n):
            eta = rng.standard_t(nu) * np.sqrt((nu - 2) / nu)
            r[i] = np.sqrt(sig2) * eta
            sig2 = omega + alpha * r[i] ** 2 + beta * sig2
        dates = pd.bdate_range("2019-01-01", periods=n)
        return pd.Series(r, index=dates)

    def test_runs_and_reacts_to_shocks(self, garch_returns: pd.Series) -> None:
        out = monte_carlo_var_es(
            garch_returns,
            distribution="garch_t",
            n_paths=5_000,
            seed=3,
            garch_refit_days=63,
            garch_fit_window=1_000,
            garch_min_window=750,
        )
        validate_result_frame(out)
        assert (out["var"] > 0).all()
        assert (out["method"] == "monte_carlo_garch_t").all()
        # Volatility clustering: VaR the day after a large |return| should
        # exceed VaR the day after a calm day, on average.
        joined = out.set_index("date")["var"]
        abs_r = garch_returns.reindex(joined.index).shift(1).abs()
        hi = joined[abs_r > abs_r.quantile(0.95)].mean()
        lo = joined[abs_r < abs_r.quantile(0.50)].mean()
        assert hi > lo

    def test_sample_starts_after_min_window(self, garch_returns: pd.Series) -> None:
        out = monte_carlo_var_es(
            garch_returns,
            distribution="garch_t",
            n_paths=1_000,
            garch_refit_days=200,
            garch_min_window=750,
        )
        assert out["date"].iloc[0] == garch_returns.index[750]
