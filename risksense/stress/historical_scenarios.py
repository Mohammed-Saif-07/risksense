"""Historical scenario replay: 2008 GFC, COVID-19, SVB.

Replays the *realised* crisis windows against today's portfolio
construction. Because the portfolio is equal-weight and daily-rebalanced,
its return path over a window is exactly what today's rule would have
earned — so the replay uses the actual portfolio return series (no factor
approximation for the headline number). Windows live in
``config/scenarios.yaml``.

Per scenario the module reports:

* cumulative loss over the window (log returns summed, then ``expm1``),
* worst single day and maximum drawdown within the window,
* a factor **attribution waterfall**: realised factor moves over the window
  × estimated betas, with the unexplained remainder labelled
  ``equity_residual`` (for an equity portfolio the equity move *is* most of
  the story; the residual keeps the waterfall summing exactly to the
  total). Factors whose macro history does not cover the window (the FRED
  credit OAS series only reach back ~3 years) are dropped and named in
  ``factors_missing`` rather than silently zeroed.

Caveat stated everywhere this is shown: the universe is today's S&P 500
constituents — survivorship bias makes these replays *milder* than the
crises were (docs/limitations.md #1).

Regulatory mapping: CCAR historical scenarios; FRTB stressed-period logic.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from risksense.config import load_config
from risksense.stress.sensitivities import FACTOR_KEYS, FactorSensitivities

#: Maps sensitivity factor keys → waterfall component names.
_COMPONENT_OF = {
    "rates_level_bp": "rates_level",
    "rates_slope_bp": "rates_slope",
    "credit_ig_bp": "credit_ig",
}


@dataclass(frozen=True)
class HistoricalReplayResult:
    """One crisis window replayed against today's portfolio."""

    name: str
    label: str
    start: str
    end: str
    n_days: int
    cumulative_loss_frac: float  # positive = loss
    cumulative_loss_usd: float
    worst_day: str
    worst_day_loss_frac: float
    max_drawdown_frac: float
    contributions: dict[str, float]  # waterfall, sums to cumulative loss
    factors_missing: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialisable form for reports and the dashboard."""
        return asdict(self)


def max_drawdown(log_returns: pd.Series) -> float:
    """Maximum peak-to-trough drawdown (positive fraction) within a window."""
    wealth = np.exp(log_returns.cumsum())
    peak = wealth.cummax()
    return float((1.0 - wealth / peak).max())


def replay_scenario(
    name: str,
    scenario: dict[str, Any],
    portfolio_returns: pd.Series,
    factor_changes: pd.DataFrame,
    sens: FactorSensitivities,
    notional_usd: float,
) -> HistoricalReplayResult:
    """Replay one configured crisis window.

    Parameters
    ----------
    name, scenario:
        Key and config mapping (``label``, ``start``, ``end``).
    portfolio_returns:
        Full daily portfolio log-return series.
    factor_changes:
        Daily factor changes from
        :func:`risksense.stress.sensitivities.load_factor_changes`.
    sens:
        Estimated sensitivities for the attribution.
    notional_usd:
        Portfolio notional for dollar figures.
    """
    start, end = str(scenario["start"]), str(scenario["end"])
    window = portfolio_returns.loc[start:end]
    if len(window) < 5:
        raise ValueError(
            f"Scenario {name}: only {len(window)} portfolio observations in "
            f"[{start}, {end}] — was the price history ingested from 2006?"
        )

    cum_log = float(window.sum())
    cum_loss = -float(np.expm1(cum_log))  # positive = loss
    worst_idx = window.idxmin()

    # Attribution: realised factor move over the window × beta.
    contributions: dict[str, float] = {}
    missing: list[str] = []
    explained = 0.0
    fac_window = factor_changes.loc[start:end]
    for key in FACTOR_KEYS:
        moves = fac_window[key].dropna()
        # Require the factor to actually span the window, not a sliver of it.
        if len(moves) < 0.5 * len(window):
            missing.append(key)
            continue
        impact = -sens.betas[key] * float(moves.sum())  # loss share
        contributions[_COMPONENT_OF[key]] = impact
        explained += impact
    contributions["equity_residual"] = cum_loss - explained

    return HistoricalReplayResult(
        name=name,
        label=str(scenario.get("label", name)),
        start=start,
        end=end,
        n_days=int(len(window)),
        cumulative_loss_frac=cum_loss,
        cumulative_loss_usd=cum_loss * notional_usd,
        worst_day=str(pd.Timestamp(worst_idx).date()),
        worst_day_loss_frac=-float(window.min()),
        max_drawdown_frac=max_drawdown(window),
        contributions=contributions,
        factors_missing=missing,
    )


def run_all(
    portfolio_returns: pd.Series,
    factor_changes: pd.DataFrame,
    sens: FactorSensitivities,
) -> list[HistoricalReplayResult]:
    """Replay every window in ``config/scenarios.yaml``'s historical block."""
    scenarios: dict[str, Any] = load_config("scenarios")["historical"]
    notional = float(load_config("portfolio")["notional_usd"])
    return [
        replay_scenario(
            name, scenario, portfolio_returns, factor_changes, sens, notional
        )
        for name, scenario in scenarios.items()
    ]
