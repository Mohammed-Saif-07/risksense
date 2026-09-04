"""Tests for the stress testing suite (sensitivities, scenarios, reverse)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from risksense.stress.historical_scenarios import max_drawdown, replay_scenario
from risksense.stress.hypothetical import apply_scenario
from risksense.stress.reverse import reverse_stress
from risksense.stress.sensitivities import (
    FACTOR_KEYS,
    FactorSensitivities,
    estimate_sensitivities,
)

TRUE_BETAS = {
    "rates_level_bp": -2e-4,  # +100bp parallel → -2% equity
    "rates_slope_bp": 5e-5,
    "credit_ig_bp": -8e-4,  # +100bp IG OAS → -8%
}


@pytest.fixture(scope="module")
def synthetic_world() -> tuple[pd.Series, pd.DataFrame]:
    """Portfolio returns generated from KNOWN betas over synthetic factors."""
    rng = np.random.default_rng(31)
    n = 1500
    dates = pd.bdate_range("2019-06-01", periods=n)
    factors = pd.DataFrame(
        {
            "rates_level_bp": rng.normal(0, 4, n),
            "rates_slope_bp": rng.normal(0, 3, n),
            "credit_ig_bp": rng.normal(0, 2, n),
        },
        index=dates,
    )
    noise = rng.normal(0, 0.006, n)
    r = sum(TRUE_BETAS[k] * factors[k] for k in FACTOR_KEYS) + noise
    return pd.Series(r, index=dates, name="r_p"), factors


@pytest.fixture(scope="module")
def sens(synthetic_world) -> FactorSensitivities:
    returns, factors = synthetic_world
    return estimate_sensitivities(returns, factors)


class TestSensitivities:
    def test_recovers_true_betas(self, sens: FactorSensitivities) -> None:
        for k, true in TRUE_BETAS.items():
            assert sens.betas[k] == pytest.approx(true, abs=3 * sens.stderrs[k])

    def test_short_sample_refused(self, synthetic_world) -> None:
        returns, factors = synthetic_world
        with pytest.raises(ValueError, match="too short"):
            estimate_sensitivities(returns.iloc[:100], factors.iloc[:100])

    def test_serialisable(self, sens: FactorSensitivities) -> None:
        d = sens.to_dict()
        assert set(d["betas"]) == set(FACTOR_KEYS)
        assert d["n_obs"] == 1500


class TestHypothetical:
    def test_parallel_shift_maps_to_level_only(self, sens: FactorSensitivities) -> None:
        res = apply_scenario(
            "parallel_up_100",
            {"label": "+100bp", "curve_shift_bp": {"2y": 100, "10y": 100, "30y": 100}},
            sens,
            notional_usd=1_000_000,
        )
        assert res.contributions["rates_slope"] == pytest.approx(0.0)
        assert res.contributions["equity"] == 0.0
        assert res.contributions["rates_level"] == pytest.approx(
            -sens.betas["rates_level_bp"] * 100
        )

    def test_waterfall_sums_to_total(self, sens: FactorSensitivities) -> None:
        res = apply_scenario(
            "combo",
            {
                "equity_shock": -0.20,
                "ig_oas_bp": 150,
                "curve_shift_bp": {"2y": -50, "10y": 25, "30y": 75},
            },
            sens,
            notional_usd=2_000_000,
        )
        assert sum(res.contributions.values()) == pytest.approx(res.total_loss_frac)
        assert res.total_loss_usd == pytest.approx(res.total_loss_frac * 2_000_000)
        # Equity crash passes through 1:1.
        assert res.contributions["equity"] == pytest.approx(0.20)

    def test_double_count_guard(self, sens: FactorSensitivities) -> None:
        # With an explicit equity shock, macro betas must NOT add more loss
        # (they encode the equity move a macro shock brings — already given).
        res = apply_scenario(
            "combo",
            {"equity_shock": -0.35, "ig_oas_bp": 300},
            sens,
            notional_usd=1_000_000,
        )
        assert res.total_loss_frac == pytest.approx(0.35)
        assert res.contributions["credit_ig"] == 0.0
        assert res.suppressed == ["credit_ig"]

    def test_macro_only_scenario_uses_betas(self, sens: FactorSensitivities) -> None:
        res = apply_scenario(
            "ig_widening", {"ig_oas_bp": 100}, sens, notional_usd=1_000_000
        )
        assert res.suppressed == []
        assert res.contributions["credit_ig"] == pytest.approx(
            -sens.betas["credit_ig_bp"] * 100
        )

    def test_steepener_hits_slope(self, sens: FactorSensitivities) -> None:
        res = apply_scenario(
            "steepener",
            {"curve_shift_bp": {"2y": -50, "10y": 50, "30y": 75}},
            sens,
            notional_usd=1_000_000,
        )
        assert res.contributions["rates_slope"] == pytest.approx(
            -sens.betas["rates_slope_bp"] * 125
        )


class TestHistoricalReplay:
    def test_max_drawdown_known_path(self) -> None:
        # +10%, -20%, +5% in log terms: peak after day 1, trough after day 2.
        r = pd.Series([0.10, -0.20, 0.05])
        assert max_drawdown(r) == pytest.approx(1 - np.exp(-0.20))

    def test_replay_matches_window_sum(
        self, synthetic_world, sens: FactorSensitivities
    ) -> None:
        returns, factors = synthetic_world
        scenario = {"label": "test window", "start": "2020-03-02", "end": "2020-04-30"}
        res = replay_scenario("w", scenario, returns, factors, sens, 1_000_000)
        window = returns.loc["2020-03-02":"2020-04-30"]
        assert res.n_days == len(window)
        assert res.cumulative_loss_frac == pytest.approx(-np.expm1(window.sum()))
        assert sum(res.contributions.values()) == pytest.approx(
            res.cumulative_loss_frac
        )
        assert res.factors_missing == []
        assert res.worst_day_loss_frac == pytest.approx(-window.min())

    def test_missing_factor_history_flagged(
        self, synthetic_world, sens: FactorSensitivities
    ) -> None:
        returns, factors = synthetic_world
        gappy = factors.copy()
        gappy.loc[:"2021-01-01", "credit_ig_bp"] = np.nan  # no early credit data
        scenario = {"label": "early", "start": "2019-08-01", "end": "2019-10-31"}
        res = replay_scenario("e", scenario, returns, gappy, sens, 1_000_000)
        assert "credit_ig_bp" in res.factors_missing
        assert "credit_ig" not in res.contributions
        assert sum(res.contributions.values()) == pytest.approx(
            res.cumulative_loss_frac
        )

    def test_empty_window_raises(
        self, synthetic_world, sens: FactorSensitivities
    ) -> None:
        returns, factors = synthetic_world
        scenario = {"start": "1999-01-01", "end": "1999-02-01"}
        with pytest.raises(ValueError, match="observations"):
            replay_scenario("x", scenario, returns, factors, sens, 1.0)


class TestReverseStress:
    SCALES = {"equity": 0.01, "rates_level_bp": 10.0, "credit_ig_bp": 10.0}
    WIDE_BOUNDS = {
        "equity": [-1.0, 1.0],
        "rates_level_bp": [-1000.0, 1000.0],
        "credit_ig_bp": [-1000.0, 1000.0],
    }

    def test_matches_closed_form_when_unbounded(
        self, sens: FactorSensitivities
    ) -> None:
        factors = ["equity", "rates_level_bp", "credit_ig_bp"]
        res = reverse_stress(
            sens,
            target_loss_frac=0.15,
            factors=factors,
            scales=self.SCALES,
            bounds=self.WIDE_BOUNDS,
        )
        beta = np.array([1.0, sens.betas["rates_level_bp"], sens.betas["credit_ig_bp"]])
        s = np.array([0.01, 10.0, 10.0])
        expected = -0.15 * (s**2) * beta / np.sum((s * beta) ** 2)
        got = np.array([res.shocks[f] for f in factors])
        assert got == pytest.approx(expected, rel=1e-3)
        assert res.achieved_loss_frac == pytest.approx(0.15, rel=1e-3)
        assert res.binding_bounds == []

    def test_bounds_bind_and_target_still_met(self, sens: FactorSensitivities) -> None:
        bounds = {
            "equity": [-0.05, 0.0],  # equity capped at -5% → others must work
            "rates_level_bp": [-300.0, 300.0],
            "credit_ig_bp": [0.0, 500.0],
        }
        res = reverse_stress(
            sens,
            target_loss_frac=0.10,
            factors=["equity", "rates_level_bp", "credit_ig_bp"],
            scales=self.SCALES,
            bounds=bounds,
        )
        assert res.achieved_loss_frac >= 0.10 * (1 - 1e-4)
        assert "equity" in res.binding_bounds

    def test_infeasible_target_raises(self, sens: FactorSensitivities) -> None:
        bounds = {
            "equity": [-0.01, 0.0],
            "rates_level_bp": [-10.0, 10.0],
            "credit_ig_bp": [0.0, 10.0],
        }
        with pytest.raises(RuntimeError, match="infeasible"):
            reverse_stress(
                sens,
                target_loss_frac=0.50,
                factors=["equity", "rates_level_bp", "credit_ig_bp"],
                scales=self.SCALES,
                bounds=bounds,
            )
