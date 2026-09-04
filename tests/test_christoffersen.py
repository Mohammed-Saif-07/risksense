"""Tests for the Christoffersen (1998) independence / conditional coverage tests."""

from __future__ import annotations

import numpy as np
import pytest

from risksense.backtesting.christoffersen import christoffersen_test


class TestChristoffersen:
    def test_known_transition_counts(self) -> None:
        # Hand-built sequence: F F T T F T F F  (7 transitions)
        # pairs: FF, FT, TT, TF, FT, TF, FF -> n00=2, n01=2, n10=2, n11=1
        hits = np.array([0, 0, 1, 1, 0, 1, 0, 0], dtype=bool)
        res = christoffersen_test(hits, coverage=0.99)
        assert (res.n00, res.n01, res.n10, res.n11) == (2, 2, 2, 1)
        assert res.pi01 == pytest.approx(2 / 4)
        assert res.pi11 == pytest.approx(1 / 3)

    def test_known_lr_ind_value(self) -> None:
        hits = np.array([0, 0, 1, 1, 0, 1, 0, 0], dtype=bool)
        res = christoffersen_test(hits)
        n00, n01, n10, n11 = 2, 2, 2, 1
        pi = (n01 + n11) / 7
        pi01, pi11 = 2 / 4, 1 / 3
        ll0 = (n00 + n10) * np.log(1 - pi) + (n01 + n11) * np.log(pi)
        ll1 = (
            n00 * np.log(1 - pi01)
            + n01 * np.log(pi01)
            + n10 * np.log(1 - pi11)
            + n11 * np.log(pi11)
        )
        assert res.lr_ind == pytest.approx(-2 * (ll0 - ll1), rel=1e-12)
        assert res.lr_cc == pytest.approx(res.lr_uc + res.lr_ind, rel=1e-12)

    def test_clustered_exceptions_rejected(self, rng: np.random.Generator) -> None:
        # 20 exceptions all in one consecutive block of 2000 days: correct
        # count (1%) but grossly dependent → independence rejected.
        hits = np.zeros(2000, dtype=bool)
        hits[1000:1020] = True
        res = christoffersen_test(hits, coverage=0.99)
        assert res.reject_independence
        assert res.reject_conditional_coverage
        assert not res.lr_uc > 6  # count itself is fine

    def test_iid_exceptions_not_rejected(self, rng: np.random.Generator) -> None:
        hits = rng.random(5000) < 0.01
        res = christoffersen_test(hits, coverage=0.99)
        assert not res.reject_independence

    def test_zero_exceptions_vacuous_independence(self) -> None:
        res = christoffersen_test(np.zeros(500, dtype=bool), coverage=0.99)
        assert res.lr_ind == 0.0
        assert res.p_ind == 1.0

    def test_too_short_raises(self) -> None:
        with pytest.raises(ValueError):
            christoffersen_test([True])

    def test_serialisable(self) -> None:
        d = christoffersen_test(np.zeros(300, dtype=bool)).to_dict()
        assert {"lr_ind", "p_cc", "n01", "pi11"} <= set(d)
