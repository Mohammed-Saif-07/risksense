"""Portfolio factor sensitivities for stress testing.

The portfolio is long-only linear equity, so stress scenarios expressed in
macro terms (yield shifts, credit spread widening) reach it through
estimated sensitivities:

    r_p = α + β_lvl ΔLevel_bp + β_slp ΔSlope_bp
            + β_ig ΔIG_bp + β_hy ΔHY_bp + ε

* **Curve factors**: Level = mean(2y, 10y, 30y), Slope = 30y - 2y, in
  basis points — raw per-tenor changes are too collinear for stable
  per-tenor betas (documented limitation).
* **Credit factor**: daily change in ICE BofA IG OAS (bp). HY OAS is
  ingested but deliberately EXCLUDED from the regression: IG and HY
  changes are so collinear that including both flips the IG beta's sign
  (observed on the real sample) — the classic multicollinearity failure,
  kept out rather than explained away.
* **Equity**: the scenario's equity shock applies to the portfolio
  directly (the portfolio *is* the equity leg — unit pass-through by
  construction), so it is not a regression coefficient.

These are **marginal** betas: they answer "given only this macro move,
what equity move comes with it?". They must NOT be stacked on top of an
explicitly specified equity shock — that double-counts, because the
correlated equity move is exactly what the beta encodes. The hypothetical
module enforces this (its double-count guard).

Estimation is OLS with heteroskedasticity-robust (HC1) standard errors via
statsmodels, on the common sample where all factors are available. Note
the ICE BofA OAS series are license-capped to ~3 trailing years on FRED,
which caps the joint sample — stated in the output and in
docs/limitations.md.

Regulatory mapping: CCAR-style scenario translation; SR 11-7 (documented,
challengeable sensitivity estimates with standard errors).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from risksense.config import data_dir

#: Regression factor keys, in reporting order (betas are per basis point).
#: ``credit_hy_bp`` exists in :func:`load_factor_changes` output but is
#: excluded here — see the module docstring on IG/HY multicollinearity.
FACTOR_KEYS = ["rates_level_bp", "rates_slope_bp", "credit_ig_bp"]

#: FRED series names (from config/data.yaml) feeding each raw input.
_SERIES = {
    "y2": "2y_treasury_yield",
    "y10": "10y_treasury_yield",
    "y30": "30y_treasury_yield",
    "ig": "ig_credit_oas",
    "hy": "hy_credit_oas",
}


@dataclass(frozen=True)
class FactorSensitivities:
    """Estimated portfolio betas to the stress factors."""

    betas: dict[str, float]  # portfolio return per unit factor change
    stderrs: dict[str, float]  # HC1 robust standard errors
    alpha: float
    r_squared: float
    n_obs: int
    sample_start: str
    sample_end: str
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialisable form for reports and the dashboard."""
        return asdict(self)


def load_factor_changes(macro_path: Path | None = None) -> pd.DataFrame:
    """Build daily factor-change columns from the ingested FRED series.

    Returns a date-indexed frame with ``rates_level_bp``, ``rates_slope_bp``,
    ``credit_ig_bp``, ``credit_hy_bp`` (NaN where a series is unavailable —
    the credit columns only cover FRED's licensed OAS window).
    """
    macro_path = macro_path or data_dir() / "raw" / "macro.parquet"
    if not macro_path.exists():
        raise FileNotFoundError(
            f"{macro_path} not found — run `python data/ingest_macro.py` first."
        )
    macro = pd.read_parquet(macro_path)
    wide = macro.pivot_table(index="date", columns="name", values="value")
    wide.index = pd.to_datetime(wide.index)
    wide = wide.sort_index()

    missing = [v for v in _SERIES.values() if v not in wide.columns]
    if missing:
        raise ValueError(f"Macro parquet missing series: {missing}")

    # FRED yields/OAS are in percent; ×100 → basis points.
    level = wide[[_SERIES["y2"], _SERIES["y10"], _SERIES["y30"]]].mean(axis=1)
    slope = wide[_SERIES["y30"]] - wide[_SERIES["y2"]]
    out = pd.DataFrame(
        {
            "rates_level_bp": level.diff() * 100.0,
            "rates_slope_bp": slope.diff() * 100.0,
            "credit_ig_bp": wide[_SERIES["ig"]].diff() * 100.0,
            "credit_hy_bp": wide[_SERIES["hy"]].diff() * 100.0,
        }
    )
    return out


def estimate_sensitivities(
    portfolio_returns: pd.Series,
    factor_changes: pd.DataFrame,
    min_obs: int = 250,
) -> FactorSensitivities:
    """OLS betas of portfolio returns on daily factor changes.

    Parameters
    ----------
    portfolio_returns:
        Daily portfolio log returns, date-indexed.
    factor_changes:
        Output of :func:`load_factor_changes`.
    min_obs:
        Minimum joint sample size; below this the estimate is refused
        rather than silently unreliable.

    Returns
    -------
    FactorSensitivities
        Betas (return per bp), HC1 standard errors, R², sample bounds.
    """
    import statsmodels.api as sm

    joint = factor_changes[FACTOR_KEYS].copy()
    joint["r_p"] = portfolio_returns
    joint = joint.dropna()
    if len(joint) < min_obs:
        raise ValueError(
            f"Joint factor sample too short: {len(joint)} < {min_obs} "
            "(credit OAS history on FRED is license-capped — see limitations)"
        )

    x = sm.add_constant(joint[FACTOR_KEYS])
    model = sm.OLS(joint["r_p"], x).fit(cov_type="HC1")

    notes = [
        "Betas from the joint sample where all factors exist; the ICE BofA "
        "OAS series cap this to ~3 trailing years on FRED.",
        "Equity shocks bypass this regression: unit pass-through by "
        "construction for the all-equity portfolio.",
        "HY OAS excluded: collinear with IG (flips the IG beta's sign).",
        "Marginal betas — never stacked on an explicit equity shock "
        "(double-count guard in the hypothetical module).",
    ]
    return FactorSensitivities(
        betas={k: float(model.params[k]) for k in FACTOR_KEYS},
        stderrs={k: float(model.bse[k]) for k in FACTOR_KEYS},
        alpha=float(model.params["const"]),
        r_squared=float(model.rsquared),
        n_obs=int(model.nobs),
        sample_start=str(joint.index.min().date()),
        sample_end=str(joint.index.max().date()),
        notes=notes,
    )
