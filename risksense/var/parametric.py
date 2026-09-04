"""Parametric (variance-covariance) VaR/ES with Ledoit-Wolf shrinkage.

Methodology
-----------
Rolling one-day parametric VaR (Jorion 2007, ch. 8) under two innovation
distributions, with the portfolio volatility taken from the full
constituent covariance matrix:

* **Volatility**: portfolio variance ``w' Σ w`` where Σ is the Ledoit-Wolf
  shrunk sample covariance of the ~500-asset universe (Ledoit & Wolf 2004,
  "A well-conditioned estimator for large-dimensional covariance matrices",
  *J. Multivariate Analysis* 88(2)), via ``sklearn.covariance.LedoitWolf``.
  With T=250 observations and N≈500 assets the raw sample covariance is
  singular; shrinkage toward a scaled identity restores conditioning.
  Σ is re-estimated every ``cov_refit_days`` (monthly by default — daily
  refits of a 500×500 estimator add noise, not information) and held
  constant in between; the mean is a daily rolling estimate.

* **Normal**: VaR_q = -μ + σ z_q ;  ES_q = -μ + σ φ(z_q)/(1-q).

* **Student-t**: degrees of freedom ν fitted per window by method of
  moments from excess kurtosis (ν = 4 + 6/κ, clipped to configured
  bounds), scale s = σ √((ν-2)/ν). Closed-form ES from McNeil, Frey &
  Embrechts (2015), *Quantitative Risk Management*, §2.3:
  ES_q = -μ + s · f_ν(t_q)/(1-q) · (ν + t_q²)/(ν-1).

No look-ahead: every estimate for forecast date *t* uses returns strictly
prior to *t*.

Regulatory mapping: FRTB (ES 97.5%), Basel III (VaR 99%), SR 11-7
benchmarking (independent challenger to historical simulation).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from risksense.var.base import VAR_RESULT_COLUMNS


def ledoit_wolf_sigma_series(
    wide_returns: pd.DataFrame,
    window: int,
    refit_days: int,
    weights: dict[str, float] | None = None,
) -> pd.Series:
    """Daily portfolio volatility from a Ledoit-Wolf shrunk covariance.

    Parameters
    ----------
    wide_returns:
        Date × ticker log-return matrix (NaN where a ticker didn't trade).
    window:
        Trailing estimation window in trading days.
    refit_days:
        Re-estimate Σ every this many days; σ is held constant in between
        (documented approximation — see module docstring).
    weights:
        Ticker → weight mapping; ``None`` = equal weight across the tickers
        with complete data in each estimation window.

    Returns
    -------
    pd.Series
        σ_p for each forecast date (index = dates from position ``window``
        onward), estimated from data strictly before that date.
    """
    from sklearn.covariance import LedoitWolf

    dates = wide_returns.index
    n = len(dates)
    if n <= window:
        raise ValueError(f"Need more than window={window} rows, got {n}")

    values = wide_returns.to_numpy()
    sigma = np.empty(n - window)
    current = np.nan
    for pos in range(window, n):
        if (pos - window) % refit_days == 0:
            win = values[pos - window : pos]
            complete = ~np.isnan(win).any(axis=0)
            sub = win[:, complete]
            if weights is None:
                w = np.full(sub.shape[1], 1.0 / sub.shape[1])
            else:
                tickers = wide_returns.columns[complete]
                raw = np.array([weights.get(t, 0.0) for t in tickers])
                if raw.sum() <= 0:
                    raise ValueError("No configured weights overlap the window")
                w = raw / raw.sum()
            cov = LedoitWolf().fit(sub).covariance_
            current = float(np.sqrt(w @ cov @ w))
        sigma[pos - window] = current
    return pd.Series(sigma, index=dates[window:], name="lw_sigma")


def fit_dof_from_kurtosis(
    excess_kurtosis: pd.Series, bounds: tuple[float, float]
) -> pd.Series:
    """Method-of-moments Student-t degrees of freedom from excess kurtosis.

    For a t distribution with ν > 4, excess kurtosis κ = 6/(ν-4), so
    ν = 4 + 6/κ. Non-positive κ (thinner than normal) maps to the upper
    bound (≈ normal); estimates are clipped to ``bounds`` for stability.
    """
    lo, hi = bounds
    nu = np.where(excess_kurtosis > 0, 4.0 + 6.0 / excess_kurtosis, hi)
    return pd.Series(np.clip(nu, lo, hi), index=excess_kurtosis.index, name="dof")


def parametric_var_es(
    returns: pd.Series,
    wide_returns: pd.DataFrame | None = None,
    window: int = 250,
    var_level: float = 0.99,
    es_level: float = 0.975,
    horizon_days: int = 1,
    distribution: str = "normal",
    cov_refit_days: int = 21,
    dof_bounds: tuple[float, float] = (3.0, 50.0),
    weights: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Rolling one-day parametric VaR and ES.

    Parameters
    ----------
    returns:
        Daily portfolio log returns (used for the rolling mean, and for
        kurtosis when ``distribution='t'``).
    wide_returns:
        Date × ticker return matrix for the Ledoit-Wolf portfolio
        volatility. If ``None``, σ falls back to the rolling standard
        deviation of the portfolio series (no shrinkage — flagged in the
        method name as ``*_naive``).
    window, var_level, es_level, horizon_days:
        As in :func:`risksense.var.historical.historical_var_es`.
    distribution:
        ``'normal'`` or ``'t'``.
    cov_refit_days:
        Ledoit-Wolf re-estimation frequency (trading days).
    dof_bounds:
        (min, max) clip for the fitted Student-t degrees of freedom.
    weights:
        Portfolio weights; ``None`` = equal weight.

    Returns
    -------
    DataFrame in the uniform engine schema; ``method`` is
    ``parametric_normal`` / ``parametric_t`` (``_naive`` suffix without
    Ledoit-Wolf).
    """
    if horizon_days != 1:
        raise NotImplementedError(
            "Only 1-day horizon is supported; do not sqrt-scale (fat tails)."
        )
    if distribution not in {"normal", "t"}:
        raise ValueError(f"Unknown distribution: {distribution!r}")

    r = returns.dropna().astype(float).sort_index()
    if len(r) <= window:
        raise ValueError(f"Need more than window={window} observations, got {len(r)}")

    # shift(1): the estimate dated t uses returns through t-1 (no look-ahead).
    mu = r.rolling(window).mean().shift(1).iloc[window:]
    forecast_dates = mu.index

    if wide_returns is not None:
        wide = wide_returns.loc[r.index]
        sigma = ledoit_wolf_sigma_series(wide, window, cov_refit_days, weights)
        sigma = sigma.loc[forecast_dates]
        method = f"parametric_{distribution}"
    else:
        sigma = r.rolling(window).std(ddof=1).shift(1).iloc[window:]
        method = f"parametric_{distribution}_naive"

    mu_a, sig_a = mu.to_numpy(), sigma.to_numpy()

    if distribution == "normal":
        z_v = stats.norm.ppf(var_level)
        z_e = stats.norm.ppf(es_level)
        var_arr = -mu_a + sig_a * z_v
        var_es_arr = -mu_a + sig_a * z_e
        es_arr = -mu_a + sig_a * stats.norm.pdf(z_e) / (1.0 - es_level)
    else:
        kurt = r.rolling(window).kurt().shift(1).iloc[window:]  # excess kurtosis
        nu = fit_dof_from_kurtosis(kurt, dof_bounds).to_numpy()
        scale = sig_a * np.sqrt((nu - 2.0) / nu)  # standardised-t scale
        t_v = stats.t.ppf(var_level, nu)
        t_e = stats.t.ppf(es_level, nu)
        var_arr = -mu_a + scale * t_v
        var_es_arr = -mu_a + scale * t_e
        es_arr = -mu_a + scale * (
            stats.t.pdf(t_e, nu) / (1.0 - es_level) * (nu + t_e**2) / (nu - 1.0)
        )

    out = pd.DataFrame(
        {
            "date": forecast_dates,
            "method": method,
            "var_level": var_level,
            "es_level": es_level,
            "horizon_days": horizon_days,
            "var": var_arr,
            "es": es_arr,
            "var_at_es_level": var_es_arr,
            "window_days": window,
        }
    )
    return out[VAR_RESULT_COLUMNS].reset_index(drop=True)
