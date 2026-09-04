"""Basel traffic-light backtesting zones.

Reference: BCBS, *Supervisory framework for the use of "backtesting" in
conjunction with the internal models approach to market risk capital
requirements*, January 1996 (retained under Basel III).

Over a 250-day window at 99% one-day VaR:

* **Green** : 0-4 exceptions — no supervisory response.
* **Yellow**: 5-9 exceptions — capital multiplier add-on (0.40 → 0.85).
* **Red**   : 10+ exceptions — multiplier +1.00, presumption of model
  rejection.

The zone boundaries correspond to cumulative binomial probabilities: green
covers outcomes up to ~95% cumulative probability under a correct model,
yellow up to ~99.99%.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from scipy import stats

#: BCBS (1996) Table 2 multiplier add-ons for 250 observations at 99% VaR.
_YELLOW_ADDONS: dict[int, float] = {5: 0.40, 6: 0.50, 7: 0.65, 8: 0.75, 9: 0.85}

GREEN_MAX = 4
YELLOW_MAX = 9
BASEL_WINDOW = 250
BASEL_COVERAGE = 0.99


@dataclass(frozen=True)
class BaselZone:
    """Traffic-light classification of a backtest window."""

    n_obs: int
    n_exceptions: int
    zone: str  # 'green' | 'yellow' | 'red'
    multiplier_addon: float  # add-on to the base capital multiplier of 3.0
    cumulative_probability: float  # P(X <= x) under a correct 99% model

    def to_dict(self) -> dict[str, Any]:
        """Serialisable form for reports and the dashboard."""
        return asdict(self)


def basel_traffic_light(
    n_exceptions: int,
    n_obs: int = BASEL_WINDOW,
    coverage: float = BASEL_COVERAGE,
) -> BaselZone:
    """Classify an exception count into the Basel traffic-light zones.

    Parameters
    ----------
    n_exceptions:
        VaR exceptions observed in the window.
    n_obs:
        Backtest window length (Basel: 250 trading days). Zone boundaries
        (4/9) are the BCBS values calibrated for 250 days; for other window
        lengths the cumulative probability is still reported but boundaries
        keep the BCBS convention.
    coverage:
        VaR level (Basel: 0.99).

    Returns
    -------
    BaselZone
        Zone, capital multiplier add-on and binomial cumulative probability.
    """
    if n_exceptions < 0 or n_obs <= 0:
        raise ValueError("n_exceptions must be >= 0 and n_obs > 0")
    if n_exceptions > n_obs:
        raise ValueError("n_exceptions cannot exceed n_obs")

    if n_exceptions <= GREEN_MAX:
        zone, addon = "green", 0.0
    elif n_exceptions <= YELLOW_MAX:
        zone, addon = "yellow", _YELLOW_ADDONS[n_exceptions]
    else:
        zone, addon = "red", 1.0

    cum_prob = float(stats.binom.cdf(n_exceptions, n_obs, 1.0 - coverage))

    return BaselZone(
        n_obs=n_obs,
        n_exceptions=n_exceptions,
        zone=zone,
        multiplier_addon=addon,
        cumulative_probability=cum_prob,
    )
