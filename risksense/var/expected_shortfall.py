"""Expected Shortfall utilities (FRTB 97.5% standard).

Each engine already reports ES alongside VaR; this module adds the two
cross-cutting pieces:

* **Euler allocation** — decompose portfolio ES into per-constituent
  contributions that sum exactly to the total (Tasche 2008, "Capital
  allocation to business units and sub-portfolios: the Euler principle").
  Under a normal model, ESC_i = -w_i μ_i + w_i (Σw)_i / σ_p · φ(z_q)/(1-q).

* **Acerbi-Székely (2014) Z₂ backtest** — ES is not elicitable (Gneiting
  2011), so it cannot be backtested by counting like VaR. The Z₂ statistic

      Z₂ = Σ_t  L_t · I_t / (T · (1-q) · ES_t)  - 1

  has expectation 0 under a correct model; Z₂ > 0 means realised tail
  losses exceed what the ES forecasts promised. Significance here is
  approximated by an i.i.d. bootstrap over days — a simplification (the
  original prescribes simulation under the model's own distribution), and
  the i.i.d. assumption understates uncertainty under volatility
  clustering; both caveats are stated in docs/limitations.md.

References: Acerbi & Székely (2014), "Back-testing expected shortfall",
*Risk* 27(11); McNeil, Frey & Embrechts (2015) §2.3; BCBS (2019) FRTB MAR33.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats


@dataclass(frozen=True)
class ESBacktestResult:
    """Outcome of the Acerbi-Székely Z₂ ES backtest."""

    n_obs: int
    n_exceptions: int
    es_level: float
    z2_stat: float
    bootstrap_se: float
    p_value: float  # one-sided (H1: ES understated), bootstrap-normal approx
    reject_h0: bool
    alpha: float
    n_bootstrap: int

    def to_dict(self) -> dict[str, Any]:
        """Serialisable form for reports and the dashboard."""
        return asdict(self)


def euler_es_allocation(
    wide_returns: pd.DataFrame,
    es_level: float = 0.975,
    weights: dict[str, float] | None = None,
) -> pd.Series:
    """Per-constituent ES contributions under a normal model (Euler rule).

    Parameters
    ----------
    wide_returns:
        Date × ticker return window (e.g. the trailing 250 days); tickers
        with any missing value in the window are dropped.
    es_level:
        ES confidence level (FRTB: 0.975).
    weights:
        Ticker → weight; ``None`` = equal weight over complete tickers.

    Returns
    -------
    pd.Series
        ES contribution per ticker (positive loss fractions), summing to
        the portfolio ES under the same normal model. Uses the Ledoit-Wolf
        shrunk covariance for consistency with the parametric engine.
    """
    from sklearn.covariance import LedoitWolf

    complete = wide_returns.dropna(axis=1)
    if complete.shape[1] < 2:
        raise ValueError("Need at least 2 complete tickers in the window")
    tickers = complete.columns
    if weights is None:
        w = np.full(len(tickers), 1.0 / len(tickers))
    else:
        raw = np.array([weights.get(t, 0.0) for t in tickers])
        if raw.sum() <= 0:
            raise ValueError("No configured weights overlap the window")
        w = raw / raw.sum()

    mu = complete.mean().to_numpy()
    cov = LedoitWolf().fit(complete.to_numpy()).covariance_
    sigma_p = float(np.sqrt(w @ cov @ w))
    z = stats.norm.ppf(es_level)
    es_factor = stats.norm.pdf(z) / (1.0 - es_level)

    contrib = -w * mu + w * (cov @ w) / sigma_p * es_factor
    return pd.Series(contrib, index=tickers, name="es_contribution").sort_values(
        ascending=False
    )


def es_backtest_z2(
    returns: pd.Series | np.ndarray,
    var_forecasts: pd.Series | np.ndarray,
    es_forecasts: pd.Series | np.ndarray,
    es_level: float = 0.975,
    alpha: float = 0.05,
    n_bootstrap: int = 5_000,
    seed: int = 42,
) -> ESBacktestResult:
    """Acerbi-Székely Z₂ test of ES forecast adequacy.

    Parameters
    ----------
    returns:
        Realised returns aligned 1:1 with the forecasts.
    var_forecasts, es_forecasts:
        VaR and ES forecasts at ``es_level`` (positive loss fractions).
        Exceptions are taken against the *same-level* VaR per the original
        formulation.
    alpha, n_bootstrap, seed:
        Significance level, bootstrap resamples, RNG seed.

    Returns
    -------
    ESBacktestResult
        Z₂ with a bootstrap standard error and one-sided p-value.
    """
    losses = -np.asarray(returns, dtype=float)
    var_a = np.asarray(var_forecasts, dtype=float)
    es_a = np.asarray(es_forecasts, dtype=float)
    if not (len(losses) == len(var_a) == len(es_a)):
        raise ValueError("returns, var_forecasts, es_forecasts must align 1:1")
    t = len(losses)
    if t == 0:
        raise ValueError("Empty sample")
    if np.any(es_a <= 0):
        raise ValueError("ES forecasts must be positive loss fractions")

    hits = losses > var_a
    terms = np.where(hits, losses / es_a, 0.0)
    scale = t * (1.0 - es_level)
    z2 = float(terms.sum() / scale - 1.0)

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, t, size=(n_bootstrap, t))
    boot = terms[idx].sum(axis=1) / scale - 1.0
    se = float(boot.std(ddof=1))
    # One-sided: reject when Z2 significantly > 0 (tail losses exceed ES).
    p_value = float(stats.norm.sf(z2 / se)) if se > 0 else (0.0 if z2 > 0 else 1.0)

    return ESBacktestResult(
        n_obs=t,
        n_exceptions=int(hits.sum()),
        es_level=es_level,
        z2_stat=z2,
        bootstrap_se=se,
        p_value=p_value,
        reject_h0=bool(p_value < alpha),
        alpha=alpha,
        n_bootstrap=n_bootstrap,
    )
