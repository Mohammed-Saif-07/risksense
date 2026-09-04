"""Hypothetical (deterministic) stress scenarios.

CCAR-style instantaneous shocks defined entirely in ``config/scenarios.yaml``
(no scenario hard-coded here): parallel yield shifts (±100bp, ±200bp),
steepener, flattener, and combined equity-crash + credit-spread-widening
scenarios.

Scenario P&L for the linear equity portfolio, in two regimes:

* **Macro-only scenario** (no ``equity_shock``): the betas *translate* the
  macro move into an implied equity move —
  ``loss = -(β_lvl ΔLevel_bp + β_slp ΔSlope_bp + β_ig ΔIG_bp)``.

* **Equity-specified scenario** (``equity_shock`` present): the equity
  shock IS the portfolio impact (unit pass-through for an all-equity
  book), and the **double-count guard** suppresses the macro beta
  contributions. The betas are marginal — they encode the equity move
  that *comes with* a macro move — so stacking them on an explicitly
  given equity crash counts the same loss twice. (First run without the
  guard priced "equity -35% + IG +300bp" at a 98% loss; the guard is why
  it now prices at 35%.) Suppressed factors are reported with zero
  contribution and an ``embedded_in_equity`` note, so the waterfall stays
  honest rather than quietly dropping them.

Per-tenor curve shifts in the config are converted to the Level/Slope
basis (Level = mean of tenors, Slope = 30y - 2y). Every scenario returns a
factor-by-factor waterfall that sums exactly to the total (checked in
tests) — the attribution the dashboard and the SR 11-7 report render.

Regulatory mapping: CCAR scenario design; BCBS stress testing principles
(2018), Principle 3 (scenario translation must be documented and
defensible).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import pandas as pd

from risksense.config import load_config
from risksense.stress.sensitivities import FactorSensitivities

#: Waterfall component order for reporting.
COMPONENTS = ["equity", "rates_level", "rates_slope", "credit_ig"]


@dataclass(frozen=True)
class ScenarioResult:
    """One hypothetical scenario's loss and factor attribution."""

    name: str
    label: str
    total_loss_frac: float  # positive = loss, fraction of portfolio value
    total_loss_usd: float
    contributions: dict[str, float]  # per COMPONENTS key, sums to total
    suppressed: list[str]  # macro factors zeroed by the double-count guard

    def to_dict(self) -> dict[str, Any]:
        """Serialisable form for reports and the dashboard."""
        return asdict(self)


def _curve_to_level_slope(curve_shift_bp: dict[str, float]) -> tuple[float, float]:
    """Convert per-tenor shifts (bp) to the Level/Slope factor basis."""
    y2 = float(curve_shift_bp.get("2y", 0.0))
    y10 = float(curve_shift_bp.get("10y", 0.0))
    y30 = float(curve_shift_bp.get("30y", 0.0))
    return (y2 + y10 + y30) / 3.0, y30 - y2


def apply_scenario(
    name: str,
    scenario: dict[str, Any],
    sens: FactorSensitivities,
    notional_usd: float,
) -> ScenarioResult:
    """Price one config-defined scenario through the factor sensitivities.

    Parameters
    ----------
    name:
        Scenario key from ``config/scenarios.yaml``.
    scenario:
        Its config mapping — any of ``curve_shift_bp`` (per-tenor bp),
        ``equity_shock`` (fractional return), ``ig_oas_bp``, ``hy_oas_bp``.
    sens:
        Estimated factor sensitivities.
    notional_usd:
        Portfolio notional for the dollar figure.
    """
    level_bp, slope_bp = _curve_to_level_slope(scenario.get("curve_shift_bp", {}))
    has_equity = "equity_shock" in scenario
    equity = float(scenario.get("equity_shock", 0.0))
    ig_bp = float(scenario.get("ig_oas_bp", 0.0))

    b = sens.betas
    # Contribution = loss share (positive = loss), so negate return impacts.
    macro = {
        "rates_level": -b["rates_level_bp"] * level_bp,
        "rates_slope": -b["rates_slope_bp"] * slope_bp,
        "credit_ig": -b["credit_ig_bp"] * ig_bp,
    }
    suppressed: list[str] = []
    if has_equity:
        # Double-count guard: marginal betas already embed the equity move
        # that comes with a macro shock — with the equity shock given, the
        # macro legs are context, not additional P&L (module docstring).
        suppressed = [k for k, v in macro.items() if v != 0.0]
        macro = {k: 0.0 for k in macro}

    contributions = {"equity": -equity, **macro}
    total = sum(contributions.values())
    return ScenarioResult(
        name=name,
        label=str(scenario.get("label", name)),
        total_loss_frac=total,
        total_loss_usd=total * notional_usd,
        contributions=contributions,
        suppressed=suppressed,
    )


def run_all(sens: FactorSensitivities) -> list[ScenarioResult]:
    """Price every scenario in ``config/scenarios.yaml``'s hypothetical block."""
    scenarios: dict[str, Any] = load_config("scenarios")["hypothetical"]
    notional = float(load_config("portfolio")["notional_usd"])
    return [
        apply_scenario(name, scenario, sens, notional)
        for name, scenario in scenarios.items()
    ]


def results_frame(results: list[ScenarioResult]) -> pd.DataFrame:
    """Long-format frame (scenario × component) for the heatmap/waterfall."""
    rows = []
    for r in results:
        for component, value in r.contributions.items():
            rows.append(
                {
                    "scenario": r.name,
                    "label": r.label,
                    "component": component,
                    "contribution": value,
                    "total_loss_frac": r.total_loss_frac,
                }
            )
    return pd.DataFrame(rows)
