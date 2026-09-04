"""Uniform output schema shared by every VaR/ES engine.

Regulatory context
------------------
FRTB (BCBS, *Minimum capital requirements for market risk*, 2019) moves the
internal-models capital metric from 99% VaR to 97.5% Expected Shortfall;
Basel II.5/III backtesting remains anchored on 99% one-day VaR. RiskSense
therefore reports both metrics side-by-side from every engine.

Conventions
-----------
* Returns are daily log returns of the portfolio.
* ``var`` and ``es`` are reported as **positive loss fractions** of portfolio
  value (e.g. ``var = 0.024`` means a 2.4% one-day loss at the stated
  confidence level).
* A VaR *exception* (breach) on day ``t`` occurs when the realised loss
  ``-r_t`` exceeds the VaR forecast made **using information up to t-1**.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

#: Column order every engine's output DataFrame must follow.
VAR_RESULT_COLUMNS: list[str] = [
    "date",  # forecast date t (VaR applies to the return realised on t)
    "method",  # 'historical' | 'parametric_normal' | 'parametric_t' | 'monte_carlo'
    "var_level",  # e.g. 0.99
    "es_level",  # e.g. 0.975 (FRTB)
    "horizon_days",  # 1 for daily
    "var",  # positive loss fraction
    "es",  # positive loss fraction
    "var_at_es_level",  # VaR at es_level (e.g. 97.5%) — used by the
    #   Acerbi-Szekely ES backtest, whose exception indicator must be at the
    #   same confidence level as the ES forecast
    "window_days",  # estimation window length
]


@dataclass(frozen=True)
class VaRResult:
    """A single VaR/ES estimate (one method, one date)."""

    date: pd.Timestamp
    method: str
    var_level: float
    es_level: float
    horizon_days: int
    var: float
    es: float
    var_at_es_level: float
    window_days: int


def empty_result_frame() -> pd.DataFrame:
    """Return an empty DataFrame with the uniform VaR result schema."""
    return pd.DataFrame(columns=VAR_RESULT_COLUMNS)


def validate_result_frame(df: pd.DataFrame) -> None:
    """Raise ``ValueError`` if ``df`` violates the uniform schema.

    Used in tests and by the validation module to guarantee engines stay
    comparable (SR 11-7 benchmarking).
    """
    missing = [c for c in VAR_RESULT_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"VaR result frame missing columns: {missing}")
    if len(df) and (df["var"] < 0).any():
        raise ValueError("VaR must be reported as a positive loss fraction")
    if len(df):
        # ES_q >= VaR_q only holds at the same level; ES 97.5% vs VaR 99%
        # (the FRTB/Basel pairing) has no fixed ordering. Only enforce the
        # coherence bound where levels allow it.
        same_or_deeper = df["es_level"] >= df["var_level"]
        bad = same_or_deeper & (df["es"] + 1e-12 < df["var"])
        if bad.any():
            raise ValueError(
                f"ES < VaR on {int(bad.sum())} rows at es_level >= var_level "
                "— check engine tail logic"
            )
        # At the SAME level the coherence bound always applies:
        # ES(es_level) >= VaR(es_level).
        bad_same = df["es"] + 1e-12 < df["var_at_es_level"]
        if bad_same.any():
            raise ValueError(
                f"ES < VaR at es_level on {int(bad_same.sum())} rows "
                "— check engine tail logic"
            )


def realized_exceptions(
    result: pd.DataFrame, returns: pd.Series
) -> pd.DataFrame:
    """Join VaR forecasts with realised returns and flag exceptions.

    Parameters
    ----------
    result:
        Uniform-schema VaR frame (one method).
    returns:
        Daily portfolio log returns indexed by date.

    Returns
    -------
    DataFrame with added columns ``realized_return`` and ``exception``
    (bool: realised loss exceeded the VaR forecast).
    """
    out = result.copy()
    out["realized_return"] = out["date"].map(returns).astype(float)
    out = out.dropna(subset=["realized_return"])
    out["exception"] = -out["realized_return"] > out["var"]
    return out


def losses_from_returns(returns: np.ndarray) -> np.ndarray:
    """Convert a return array to a loss array (loss = -return)."""
    return -np.asarray(returns, dtype=float)
