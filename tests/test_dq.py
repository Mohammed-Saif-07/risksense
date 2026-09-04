"""Tests for the Engle-Manganelli (2004) Dynamic Quantile test."""

from __future__ import annotations

import numpy as np
import pytest

from risksense.backtesting.dq_test import dq_test


class TestDQ:
    def test_iid_correct_coverage_not_rejected(self, rng: np.random.Generator) -> None:
        n = 5000
        hits = rng.random(n) < 0.01
        var = np.full(n, 0.02)
        res = dq_test(hits, var, coverage=0.99, lags=4)
        assert not res.reject_h0
        assert res.n_obs == n - 4

    def test_clustered_hits_rejected(self) -> None:
        hits = np.zeros(2000, dtype=bool)
        hits[1000:1025] = True  # heavy clustering
        var = np.full(2000, 0.02)
        res = dq_test(hits, var, coverage=0.99, lags=4)
        assert res.reject_h0
        assert res.p_value < 1e-6

    def test_var_correlated_hits_rejected(self, rng: np.random.Generator) -> None:
        # Hits that occur predominantly when VaR is LOW (model too tight in
        # calm regimes) — invisible to Kupiec, caught by the VaR regressor.
        n = 4000
        var = np.where(np.arange(n) % 2 == 0, 0.01, 0.03)
        p = np.where(var == 0.01, 0.02, 0.0)
        hits = rng.random(n) < p
        res = dq_test(hits, var, coverage=0.99, lags=4)
        assert res.reject_h0

    def test_misaligned_inputs_raise(self) -> None:
        with pytest.raises(ValueError, match="align"):
            dq_test([True, False], np.array([0.02]))

    def test_too_short_raises(self) -> None:
        with pytest.raises(ValueError):
            dq_test([False] * 5, np.full(5, 0.02), lags=4)

    def test_serialisable(self, rng: np.random.Generator) -> None:
        hits = rng.random(1000) < 0.01
        d = dq_test(hits, np.full(1000, 0.02)).to_dict()
        assert {"dq_stat", "p_value", "lags"} <= set(d)
