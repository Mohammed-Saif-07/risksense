"""Tests for the Historical Simulation VaR/ES engine."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from risksense.var.base import (
    VAR_RESULT_COLUMNS,
    realized_exceptions,
    validate_result_frame,
)
from risksense.var.historical import historical_var_es, hs_es, hs_var


class TestPointEstimators:
    def test_var_matches_known_quantile(self, rng: np.random.Generator) -> None:
        # For a large N(0,1) loss sample, the empirical 99% quantile should be
        # close to the theoretical 2.326.
        losses = rng.normal(size=200_000)
        assert hs_var(losses, 0.99) == pytest.approx(stats.norm.ppf(0.99), abs=0.03)

    def test_es_exceeds_var_same_level(self, rng: np.random.Generator) -> None:
        losses = rng.normal(size=50_000)
        assert hs_es(losses, 0.975) >= hs_var(losses, 0.975)

    def test_es_matches_normal_closed_form(self, rng: np.random.Generator) -> None:
        # Normal ES_q = phi(z_q) / (1 - q)  (McNeil-Frey-Embrechts 2015, ex. 2.14).
        losses = rng.normal(size=500_000)
        q = 0.975
        theoretical = stats.norm.pdf(stats.norm.ppf(q)) / (1 - q)
        assert hs_es(losses, q) == pytest.approx(theoretical, rel=0.02)

    def test_empty_losses_raise(self) -> None:
        with pytest.raises(ValueError):
            hs_var(np.array([]), 0.99)


class TestRollingEngine:
    def test_output_schema(self, normal_returns: pd.Series) -> None:
        out = historical_var_es(normal_returns, window=250)
        assert list(out.columns) == VAR_RESULT_COLUMNS
        validate_result_frame(out)
        assert (out["method"] == "historical").all()
        assert len(out) == len(normal_returns) - 250

    def test_no_lookahead(self, normal_returns: pd.Series) -> None:
        # The forecast for the first date must equal the quantile of the
        # window strictly before it.
        window = 250
        out = historical_var_es(normal_returns, window=window)
        first = out.iloc[0]
        manual = hs_var(-normal_returns.iloc[:window].to_numpy(), 0.99)
        assert first["var"] == pytest.approx(manual)
        assert first["date"] == normal_returns.index[window]

    def test_exception_rate_near_nominal(self, normal_returns: pd.Series) -> None:
        # Under i.i.d. normal returns, ~1% of days should breach 99% VaR.
        out = historical_var_es(normal_returns, window=250)
        joined = realized_exceptions(out, normal_returns)
        rate = joined["exception"].mean()
        # Binomial 99.9% band around 1% for ~1750 trials.
        assert 0.002 < rate < 0.025

    def test_var_positive_loss_convention(self, normal_returns: pd.Series) -> None:
        out = historical_var_es(normal_returns, window=250)
        assert (out["var"] > 0).all()
        assert (out["es"] > 0).all()

    def test_fat_tails_raise_es_ratio(
        self, normal_returns: pd.Series, fat_tailed_returns: pd.Series
    ) -> None:
        # ES/VaR at the same level grows with tail heaviness — a sanity check
        # that the engine actually sees the tails.
        n = historical_var_es(normal_returns, window=500, var_level=0.975)
        t = historical_var_es(fat_tailed_returns, window=500, var_level=0.975)
        assert (t["es"] / t["var"]).mean() > (n["es"] / n["var"]).mean()

    def test_too_short_series_raises(self, normal_returns: pd.Series) -> None:
        with pytest.raises(ValueError, match="window"):
            historical_var_es(normal_returns.iloc[:100], window=250)

    def test_multiday_horizon_rejected(self, normal_returns: pd.Series) -> None:
        with pytest.raises(NotImplementedError):
            historical_var_es(normal_returns, horizon_days=10)
