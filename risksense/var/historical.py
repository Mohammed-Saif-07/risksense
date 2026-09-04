"""Historical Simulation VaR and Expected Shortfall.

Methodology
-----------
Full-revaluation historical simulation (Jorion, *Value at Risk*, 3rd ed.,
2007, ch. 10): the loss distribution for day ``t`` is the empirical
distribution of the previous ``window`` daily portfolio returns
(``t-window`` … ``t-1``). No distributional assumption is made.

* VaR at level ``q`` is the empirical ``q``-quantile of losses.
* ES at level ``q_es`` is the mean loss **beyond** the ``q_es`` VaR
  (Acerbi & Tasche, 2002, "On the coherence of expected shortfall"),
  reported at 97.5% per FRTB (BCBS 2019) alongside 99% VaR (Basel III
  backtesting anchor).

Regulatory mapping: FRTB IMA (ES 97.5%), Basel III backtesting (VaR 99%),
SR 11-7 (documented assumptions: i.i.d. within window, equal weighting).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from risksense.var.base import VAR_RESULT_COLUMNS

METHOD_NAME = "historical"


def hs_var(losses: np.ndarray, level: float) -> float:
    """Empirical VaR: the ``level``-quantile of the loss distribution.

    Uses the 'higher' interpolation so the estimate is an actually observed
    loss and the coverage is conservative for small samples.
    """
    if len(losses) == 0:
        raise ValueError("Empty loss array")
    return float(np.quantile(losses, level, method="higher"))


def hs_es(losses: np.ndarray, level: float) -> float:
    """Empirical ES: mean of losses at or beyond the ``level``-quantile.

    Follows Acerbi & Tasche (2002); with the 'higher' quantile convention the
    tail set is never empty.
    """
    var_l = hs_var(losses, level)
    tail = losses[losses >= var_l]
    return float(tail.mean())


def historical_var_es(
    returns: pd.Series,
    window: int = 250,
    var_level: float = 0.99,
    es_level: float = 0.975,
    horizon_days: int = 1,
) -> pd.DataFrame:
    """Rolling one-day Historical Simulation VaR and ES.

    Parameters
    ----------
    returns:
        Daily portfolio log returns, indexed by trading date, oldest first.
    window:
        Estimation window in trading days (Basel III minimum: 250,
        i.e. one year of daily observations).
    var_level:
        VaR confidence level (Basel III backtesting anchor: 0.99).
    es_level:
        ES confidence level (FRTB standard: 0.975).
    horizon_days:
        Holding period in days. Only 1-day is estimated directly; longer
        horizons should be estimated with overlapping returns, not scaled
        (documented limitation — sqrt-of-time scaling understates fat-tailed
        risk; see docs/limitations.md).

    Returns
    -------
    DataFrame in the uniform engine schema (:data:`VAR_RESULT_COLUMNS`),
    one row per forecast date. The forecast for date ``t`` uses returns
    strictly prior to ``t`` (no look-ahead).
    """
    if horizon_days != 1:
        raise NotImplementedError(
            "Only 1-day horizon is supported; do not sqrt-scale (fat tails)."
        )
    r = returns.dropna().astype(float)
    if len(r) <= window:
        raise ValueError(
            f"Need more than window={window} observations, got {len(r)}"
        )
    if not r.index.is_monotonic_increasing:
        r = r.sort_index()

    losses = -r.to_numpy()
    dates = r.index

    # Strided rolling windows: row i holds losses for dates[i .. i+window-1],
    # forecasting date dates[i+window]. Vectorised — ~1500x faster than a
    # Python loop over 5,000 days.
    windows = np.lib.stride_tricks.sliding_window_view(losses, window)[:-1]
    var_arr = np.quantile(windows, var_level, axis=1, method="higher")
    es_cut = np.quantile(windows, es_level, axis=1, method="higher")
    tail_mask = windows >= es_cut[:, None]
    es_arr = np.where(tail_mask, windows, np.nan)
    es_arr = np.nanmean(es_arr, axis=1)

    out = pd.DataFrame(
        {
            "date": dates[window:],
            "method": METHOD_NAME,
            "var_level": var_level,
            "es_level": es_level,
            "horizon_days": horizon_days,
            "var": var_arr,
            "es": es_arr,
            "var_at_es_level": es_cut,
            "window_days": window,
        }
    )
    return out[VAR_RESULT_COLUMNS]
