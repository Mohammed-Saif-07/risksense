"""Engle-Manganelli (2004) Dynamic Quantile test.

Reference: Engle, R. & Manganelli, S. (2004), "CAViaR: Conditional
Autoregressive Value at Risk by Regression Quantiles", *Journal of Business
& Economic Statistics* 22(4), 367-381, §4.

The DQ test regresses the demeaned hit sequence

    Hit_t = I(loss_t > VaR_t) - (1 - q)

on a constant, L lagged hits, and the contemporaneous VaR forecast. Under a
correct model, Hit_t is unpredictable from anything in the information set
— all coefficients are zero — and the Wald-type statistic

    DQ = β̂' X'X β̂ / (p(1-p)),   p = 1 - q

is asymptotically χ²(L+2). Unlike Kupiec (count only) and Christoffersen
(one-day clustering only), DQ catches longer-range clustering and hits that
correlate with the VaR level itself (a model that is too tight exactly when
it predicts calm).

Regulatory mapping: SR 11-7 outcomes analysis / model performance
monitoring.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats


@dataclass(frozen=True)
class DQResult:
    """Outcome of the Dynamic Quantile test."""

    n_obs: int  # effective sample after losing `lags` startup rows
    n_exceptions: int
    coverage: float
    lags: int
    dq_stat: float
    p_value: float
    reject_h0: bool
    alpha: float

    def to_dict(self) -> dict[str, Any]:
        """Serialisable form for reports and the dashboard."""
        return asdict(self)


def dq_test(
    exceptions: pd.Series | np.ndarray | list[bool],
    var_forecasts: pd.Series | np.ndarray,
    coverage: float = 0.99,
    lags: int = 4,
    alpha: float = 0.05,
) -> DQResult:
    """Run the Engle-Manganelli Dynamic Quantile test.

    Parameters
    ----------
    exceptions:
        Boolean daily exception indicators, in date order.
    var_forecasts:
        The VaR forecasts (positive loss fractions) aligned 1:1 with
        ``exceptions`` — included as a regressor per Engle-Manganelli.
    coverage:
        VaR confidence level q.
    lags:
        Number of lagged hits in the regression (standard: 4).
    alpha:
        Significance level for the rejection flag.

    Returns
    -------
    DQResult
        DQ statistic with its χ²(lags+2) p-value.
    """
    hits_b = np.asarray(exceptions, dtype=bool)
    var_a = np.asarray(var_forecasts, dtype=float)
    if len(hits_b) != len(var_a):
        raise ValueError("exceptions and var_forecasts must align 1:1")
    if len(hits_b) <= lags + 2:
        raise ValueError(f"Need more than {lags + 2} observations")
    if lags < 1:
        raise ValueError("lags must be >= 1")

    p = 1.0 - coverage
    hit = hits_b.astype(float) - p  # demeaned hit sequence

    # Rows t = lags..T-1; columns: const, Hit_{t-1..t-lags}, VaR_t.
    y = hit[lags:]
    n = len(y)
    x = np.column_stack(
        [np.ones(n)] + [hit[lags - k : -k] for k in range(1, lags + 1)] + [var_a[lags:]]
    )

    xtx = x.T @ x
    beta = np.linalg.pinv(xtx) @ (x.T @ y)
    dq = float(beta @ xtx @ beta) / (p * (1.0 - p))
    dq = max(dq, 0.0)
    dof = x.shape[1]
    p_value = float(stats.chi2.sf(dq, df=dof))

    return DQResult(
        n_obs=n,
        n_exceptions=int(hits_b[lags:].sum()),
        coverage=coverage,
        lags=lags,
        dq_stat=dq,
        p_value=p_value,
        reject_h0=bool(p_value < alpha),
        alpha=alpha,
    )
