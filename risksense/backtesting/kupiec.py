"""Kupiec (1995) proportion-of-failures (POF) test.

Reference: Kupiec, P. (1995), "Techniques for Verifying the Accuracy of Risk
Measurement Models", *Journal of Derivatives* 3(2), 73-84. See also Jorion
(2007), ch. 6, and Christoffersen (1998) for the LR framework.

The POF test checks **unconditional coverage**: under a correctly calibrated
VaR at level ``q``, exceptions are Bernoulli with p = 1 - q, so the observed
exception count x over T days should be consistent with Binomial(T, p).

The likelihood-ratio statistic

    LR_pof = -2 ln[ (1-p)^(T-x) p^x ] + 2 ln[ (1-pi)^(T-x) pi^x ],
    pi = x / T

is asymptotically chi-squared with 1 degree of freedom under H0.

Regulatory mapping: Basel III backtesting of 99% one-day VaR over a 250-day
window; SR 11-7 outcomes analysis.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats


@dataclass(frozen=True)
class KupiecResult:
    """Outcome of a Kupiec POF test."""

    n_obs: int
    n_exceptions: int
    expected_exceptions: float
    coverage: float  # VaR level q (e.g. 0.99)
    exception_rate: float  # observed x / T
    lr_stat: float
    p_value: float
    reject_h0: bool  # True => model coverage rejected at ``alpha``
    alpha: float

    def to_dict(self) -> dict[str, Any]:
        """Serialisable form for reports and the dashboard."""
        return asdict(self)


def kupiec_pof_test(
    exceptions: pd.Series | np.ndarray | list[bool] | int,
    n_obs: int | None = None,
    coverage: float = 0.99,
    alpha: float = 0.05,
) -> KupiecResult:
    """Run the Kupiec proportion-of-failures test.

    Parameters
    ----------
    exceptions:
        Either a boolean sequence of daily exception indicators, or an
        integer exception count (in which case ``n_obs`` is required).
    n_obs:
        Number of backtest days T. Inferred from the sequence if omitted.
    coverage:
        VaR confidence level q; expected exception probability is 1 - q.
    alpha:
        Significance level for the H0 rejection flag.

    Returns
    -------
    KupiecResult
        LR statistic, p-value (chi-squared, 1 dof) and rejection flag.

    Notes
    -----
    Edge cases x = 0 and x = T are handled by dropping the vanishing log
    terms (the MLE likelihood contribution of an empty category is 1).
    With x = 0 the test can still reject for large T: observing *no*
    exceptions is evidence of an over-conservative model, which SR 11-7
    treats as a model weakness too (capital efficiency).
    """
    if isinstance(exceptions, (int, np.integer)):
        if n_obs is None:
            raise ValueError("n_obs is required when passing an exception count")
        x = int(exceptions)
        t = int(n_obs)
    else:
        arr = np.asarray(exceptions, dtype=bool)
        x = int(arr.sum())
        t = int(len(arr)) if n_obs is None else int(n_obs)

    if t <= 0:
        raise ValueError("n_obs must be positive")
    if not 0 <= x <= t:
        raise ValueError(f"Exception count {x} outside [0, {t}]")
    if not 0.0 < coverage < 1.0:
        raise ValueError("coverage must be in (0, 1)")

    p = 1.0 - coverage
    pi = x / t

    def log_lik(prob: float) -> float:
        ll = 0.0
        if t - x > 0:
            ll += (t - x) * np.log(1.0 - prob)
        if x > 0:
            ll += x * np.log(prob)
        return ll

    lr = -2.0 * (log_lik(p) - log_lik(pi))
    lr = max(lr, 0.0)  # guard tiny negative values from float rounding
    p_value = float(stats.chi2.sf(lr, df=1))

    return KupiecResult(
        n_obs=t,
        n_exceptions=x,
        expected_exceptions=t * p,
        coverage=coverage,
        exception_rate=pi,
        lr_stat=float(lr),
        p_value=p_value,
        reject_h0=bool(p_value < alpha),
        alpha=alpha,
    )
