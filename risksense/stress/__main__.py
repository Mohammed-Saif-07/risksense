"""Stress testing driver: ``python -m risksense.stress``.

Chains sensitivities → hypothetical scenarios → historical replays →
reverse stress, writing ``data/processed/stress_results.json``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from risksense.cli import load_portfolio_returns
from risksense.config import data_dir
from risksense.stress import historical_scenarios, hypothetical
from risksense.stress.reverse import reverse_stress
from risksense.stress.sensitivities import (
    estimate_sensitivities,
    load_factor_changes,
)


def run(output_path: Path | None = None) -> dict[str, Any]:
    """Run the full stress suite and write the dashboard JSON."""
    returns = load_portfolio_returns()
    factors = load_factor_changes()
    sens = estimate_sensitivities(returns, factors)
    print(
        f"Sensitivities on {sens.n_obs} joint days "
        f"({sens.sample_start} → {sens.sample_end}), R²={sens.r_squared:.3f}"
    )
    for k, b in sens.betas.items():
        print(f"  {k:16s} beta={b:+.3e} (se {sens.stderrs[k]:.1e})")

    hypo = hypothetical.run_all(sens)
    for scenario in sorted(hypo, key=lambda s: -s.total_loss_frac):
        print(f"  hypo {scenario.label:34s} loss {scenario.total_loss_frac:+7.2%}")

    hist = historical_scenarios.run_all(returns, factors, sens)
    for replay in hist:
        print(
            f"  hist {replay.label:34s} loss {replay.cumulative_loss_frac:7.2%} "
            f"({replay.n_days}d, worst day {replay.worst_day_loss_frac:.2%}, "
            f"maxDD {replay.max_drawdown_frac:.2%})"
        )

    rev = reverse_stress(sens)
    print(
        f"  reverse: target {rev.target_loss_frac:.0%} via "
        + ", ".join(f"{k}={v:+.4g}" for k, v in rev.shocks.items())
    )

    out = {
        "sensitivities": sens.to_dict(),
        "hypothetical": [r.to_dict() for r in hypo],
        "historical": [r.to_dict() for r in hist],
        "reverse": rev.to_dict(),
    }
    output_path = output_path or data_dir() / "processed" / "stress_results.json"
    with output_path.open("w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    print(f"Wrote {output_path}")
    return out


if __name__ == "__main__":
    run()
