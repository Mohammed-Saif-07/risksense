"""Monte Carlo VaR/ES: multivariate normal, multivariate t, GARCH(1,1)-t.

Methodology
-----------
10,000+ simulated one-day portfolio P&L paths per forecast date, seeded for
reproducibility (``numpy.random.default_rng``), VaR/ES read off the
empirical quantiles of the simulated loss distribution.

Data-generating processes:

a. **Multivariate normal** — asset covariance from the Ledoit-Wolf shrunk
   estimator (shared with the parametric engine). For a *linear* portfolio
   the projection ``w'X`` of X ~ N(μ, Σ) is exactly one-dimensional normal
   with variance ``w'Σw``, so paths are drawn from the projected
   distribution with the full 500-asset Σ entering through ``w'Σw``. This
   is exact, not an approximation, for the linear equal-weight portfolio
   (it would break for options — see docs/limitations.md).

b. **Multivariate Student-t** — same projection property holds (an affine
   combination of a multivariate t is univariate t with the same ν;
   McNeil-Frey-Embrechts 2015, prop. 6.28); ν fitted per window by method
   of moments from portfolio kurtosis.

c. **GARCH(1,1) with Student-t innovations** (Bollerslev 1986) on the
   portfolio series for volatility clustering: parameters re-fitted every
   ``refit_days`` on a trailing window via the ``arch`` package; between
   refits the conditional variance recursion
   ``σ²_t = ω + α ε²_{t-1} + β σ²_{t-1}`` updates daily, so the VaR
   forecast reacts to yesterday's shock even when parameters are stale.

Regulatory mapping: FRTB (ES 97.5%), SR 11-7 benchmarking (three challenger
specifications spanning thin tails, fat tails, and volatility clustering).
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd

from risksense.var.base import VAR_RESULT_COLUMNS
from risksense.var.parametric import fit_dof_from_kurtosis, ledoit_wolf_sigma_series

#: Simulate this many forecast dates per vectorised block (memory guard:
#: chunk × n_paths floats per block, ~20 MB at 250 × 10,000).
_CHUNK_DATES = 250


def _empirical_var_es(
    losses: np.ndarray, var_level: float, es_level: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Row-wise empirical VaR, ES, and es-level VaR from simulated paths."""
    var_arr = np.quantile(losses, var_level, axis=1)
    es_cut = np.quantile(losses, es_level, axis=1)
    tail = np.where(losses >= es_cut[:, None], losses, np.nan)
    return var_arr, np.nanmean(tail, axis=1), es_cut


def _simulate_location_scale(
    mu: np.ndarray,
    sigma: np.ndarray,
    draw: Callable[[np.random.Generator, tuple[int, int], int], np.ndarray],
    n_paths: int,
    var_level: float,
    es_level: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Chunked simulation of r = μ + σ·η with unit-variance innovations η."""
    n = len(mu)
    var_out = np.empty(n)
    es_out = np.empty(n)
    var_es_out = np.empty(n)
    for s in range(0, n, _CHUNK_DATES):
        e = min(s + _CHUNK_DATES, n)
        eta = draw(rng, (e - s, n_paths), s)
        losses = -(mu[s:e, None] + sigma[s:e, None] * eta)
        var_out[s:e], es_out[s:e], var_es_out[s:e] = _empirical_var_es(
            losses, var_level, es_level
        )
    return var_out, es_out, var_es_out


def _fit_garch_path(
    r: pd.Series,
    window: int,
    refit_days: int,
    fit_window: int,
    min_window: int,
    dof_floor: float,
) -> tuple[pd.Index, np.ndarray, np.ndarray, np.ndarray]:
    """Rolling-refit GARCH(1,1)-t conditional forecasts (no look-ahead).

    Returns forecast dates and per-date arrays (μ, σ_t+1|t, ν). Fits are in
    percent space (arch's recommended scaling) and converted back.
    """
    from arch import arch_model

    values = r.to_numpy() * 100.0  # percent scale for optimiser stability
    dates = r.index
    start = max(window, min_window)
    n = len(values)
    if n <= start:
        raise ValueError(f"Need more than {start} observations, got {n}")

    mu_out = np.empty(n - start)
    sig_out = np.empty(n - start)
    nu_out = np.empty(n - start)

    mu = omega = alpha = beta = nu = np.nan
    sigma2 = np.nan
    for pos in range(start, n):
        if (pos - start) % refit_days == 0:
            lo = max(0, pos - fit_window)
            res = arch_model(
                values[lo:pos], mean="Constant", vol="GARCH", p=1, q=1, dist="t"
            ).fit(disp="off", show_warning=False)
            mu = float(res.params["mu"])
            omega = float(res.params["omega"])
            alpha = float(res.params["alpha[1]"])
            beta = float(res.params["beta[1]"])
            nu = max(float(res.params["nu"]), dof_floor)
            # σ² for the day after the fit window, from the fitted state.
            last_eps = values[pos - 1] - mu
            last_sig2 = float(res.conditional_volatility[-1]) ** 2
            sigma2 = omega + alpha * last_eps**2 + beta * last_sig2
        else:
            eps = values[pos - 1] - mu
            sigma2 = omega + alpha * eps**2 + beta * sigma2
        mu_out[pos - start] = mu / 100.0
        sig_out[pos - start] = np.sqrt(sigma2) / 100.0
        nu_out[pos - start] = nu
    return dates[start:], mu_out, sig_out, nu_out


def monte_carlo_var_es(
    returns: pd.Series,
    wide_returns: pd.DataFrame | None = None,
    window: int = 250,
    var_level: float = 0.99,
    es_level: float = 0.975,
    horizon_days: int = 1,
    distribution: str = "normal",
    n_paths: int = 10_000,
    seed: int = 42,
    cov_refit_days: int = 21,
    dof_bounds: tuple[float, float] = (3.0, 50.0),
    garch_refit_days: int = 21,
    garch_fit_window: int = 1_000,
    garch_min_window: int = 750,
    weights: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Rolling one-day Monte Carlo VaR and ES.

    Parameters
    ----------
    returns:
        Daily portfolio log returns.
    wide_returns:
        Date × ticker matrix for the Ledoit-Wolf covariance (normal / t
        DGPs). Falls back to rolling portfolio std if ``None``.
    distribution:
        ``'normal'`` | ``'t'`` | ``'garch_t'``.
    n_paths, seed:
        Simulation size and RNG seed (config: ``monte_carlo``).
    garch_refit_days, garch_fit_window, garch_min_window:
        GARCH re-fit cadence, trailing fit window, and minimum history
        before the first forecast.
    Others as in :func:`risksense.var.parametric.parametric_var_es`.

    Returns
    -------
    DataFrame in the uniform engine schema; method =
    ``monte_carlo_<distribution>``. Note the GARCH sample starts later
    (needs ``garch_min_window`` days) — align on dates when benchmarking.
    """
    if horizon_days != 1:
        raise NotImplementedError(
            "Only 1-day horizon is supported; do not sqrt-scale (fat tails)."
        )
    if distribution not in {"normal", "t", "garch_t"}:
        raise ValueError(f"Unknown distribution: {distribution!r}")

    r = returns.dropna().astype(float).sort_index()
    rng = np.random.default_rng(seed)

    if distribution == "garch_t":
        forecast_dates, mu_a, sig_a, nu_a = _fit_garch_path(
            r,
            window,
            garch_refit_days,
            garch_fit_window,
            garch_min_window,
            dof_floor=dof_bounds[0],
        )
        used_window = garch_fit_window

        def draw(
            g: np.random.Generator, shape: tuple[int, int], offset: int
        ) -> np.ndarray:
            nu_blk = nu_a[offset : offset + shape[0], None]
            return g.standard_t(nu_blk, size=shape) * np.sqrt((nu_blk - 2.0) / nu_blk)

    else:
        if len(r) <= window:
            raise ValueError(
                f"Need more than window={window} observations, got {len(r)}"
            )
        mu = r.rolling(window).mean().shift(1).iloc[window:]
        forecast_dates = mu.index
        mu_a = mu.to_numpy()
        used_window = window
        if wide_returns is not None:
            sigma = ledoit_wolf_sigma_series(
                wide_returns.loc[r.index], window, cov_refit_days, weights
            ).loc[forecast_dates]
        else:
            sigma = r.rolling(window).std(ddof=1).shift(1).iloc[window:]
        sig_a = sigma.to_numpy()

        if distribution == "t":
            kurt = r.rolling(window).kurt().shift(1).iloc[window:]
            nu_a = fit_dof_from_kurtosis(kurt, dof_bounds).to_numpy()

            def draw(
                g: np.random.Generator, shape: tuple[int, int], offset: int
            ) -> np.ndarray:
                nu_blk = nu_a[offset : offset + shape[0], None]
                return g.standard_t(nu_blk, size=shape) * np.sqrt(
                    (nu_blk - 2.0) / nu_blk
                )

        else:

            def draw(
                g: np.random.Generator, shape: tuple[int, int], offset: int
            ) -> np.ndarray:
                return g.standard_normal(size=shape)

    var_arr, es_arr, var_es_arr = _simulate_location_scale(
        mu_a, sig_a, draw, n_paths, var_level, es_level, rng
    )

    out = pd.DataFrame(
        {
            "date": forecast_dates,
            "method": f"monte_carlo_{distribution}",
            "var_level": var_level,
            "es_level": es_level,
            "horizon_days": horizon_days,
            "var": var_arr,
            "es": es_arr,
            "var_at_es_level": var_es_arr,
            "window_days": used_window,
        }
    )
    return out[VAR_RESULT_COLUMNS].reset_index(drop=True)
