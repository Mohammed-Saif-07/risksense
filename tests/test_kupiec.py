"""Tests for the Kupiec (1995) POF test."""

from __future__ import annotations

import numpy as np
import pytest

from risksense.backtesting.kupiec import kupiec_pof_test


class TestKupiec:
    def test_exact_coverage_not_rejected(self) -> None:
        # Exactly the expected number of exceptions: LR ≈ 0, p ≈ 1.
        res = kupiec_pof_test(10, n_obs=1000, coverage=0.99)
        assert res.lr_stat == pytest.approx(0.0, abs=1e-9)
        assert res.p_value == pytest.approx(1.0, abs=1e-6)
        assert not res.reject_h0

    def test_gross_undercoverage_rejected(self) -> None:
        # 40 exceptions in 1000 days at 99% is a broken model.
        res = kupiec_pof_test(40, n_obs=1000, coverage=0.99)
        assert res.reject_h0
        assert res.p_value < 1e-6

    def test_known_lr_value(self) -> None:
        # Hand-computed LR for x=15, T=1000, p=0.01:
        # LR = -2[ln((0.99^985)(0.01^15)) - ln((0.985^985)(0.015^15))]
        res = kupiec_pof_test(15, n_obs=1000, coverage=0.99)
        p, pi, t, x = 0.01, 0.015, 1000, 15
        expected = -2.0 * (
            ((t - x) * np.log(1 - p) + x * np.log(p))
            - ((t - x) * np.log(1 - pi) + x * np.log(pi))
        )
        assert res.lr_stat == pytest.approx(expected, rel=1e-12)

    def test_zero_exceptions_handled(self) -> None:
        res = kupiec_pof_test(0, n_obs=250, coverage=0.99)
        assert np.isfinite(res.lr_stat)
        assert res.n_exceptions == 0
        # LR = -2*T*ln(1-p) = 5.025 for x=0, T=250, p=0.01. The asymptotic
        # chi-squared p-value (0.025) rejects at 5% even though the exact
        # binomial P(X=0) ≈ 8% would not — the documented small-sample
        # weakness of the asymptotic POF test (docs/limitations.md #9).
        assert res.lr_stat == pytest.approx(-2 * 250 * np.log(0.99), rel=1e-12)
        assert res.reject_h0

    def test_zero_exceptions_long_sample_rejects(self) -> None:
        # 5000 days with zero exceptions => over-conservative model, rejected.
        res = kupiec_pof_test(0, n_obs=5000, coverage=0.99)
        assert res.reject_h0

    def test_boolean_sequence_input(self, rng: np.random.Generator) -> None:
        seq = rng.random(1000) < 0.01
        res = kupiec_pof_test(seq, coverage=0.99)
        assert res.n_exceptions == int(seq.sum())
        assert res.n_obs == 1000

    def test_count_without_n_obs_raises(self) -> None:
        with pytest.raises(ValueError, match="n_obs"):
            kupiec_pof_test(5)

    def test_invalid_coverage_raises(self) -> None:
        with pytest.raises(ValueError, match="coverage"):
            kupiec_pof_test(5, n_obs=100, coverage=1.5)

    def test_result_serialisable(self) -> None:
        d = kupiec_pof_test(8, n_obs=500, coverage=0.99).to_dict()
        assert d["n_exceptions"] == 8
        assert set(d) >= {"lr_stat", "p_value", "reject_h0", "coverage"}
