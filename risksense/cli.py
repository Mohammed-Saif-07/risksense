"""RiskSense command-line driver.

Chains the processed portfolio returns through the VaR engines and the
backtesting suite, writing dashboard-ready outputs:

    data/processed/var_results.parquet   — uniform-schema VaR/ES forecasts
                                            + realised returns + exceptions
    data/processed/backtest_summary.json — Kupiec + Basel traffic light

Usage:
    python -m risksense.cli            # run all implemented engines + backtests
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from risksense.backtesting.basel_traffic_light import basel_traffic_light
from risksense.backtesting.kupiec import kupiec_pof_test
from risksense.config import data_dir, load_config
from risksense.var.base import realized_exceptions, validate_result_frame
from risksense.var.historical import historical_var_es


def load_portfolio_returns(path: Path | None = None) -> pd.Series:
    """Load the portfolio return series produced by the PySpark pipeline."""
    path = path or data_dir() / "processed" / "portfolio_returns.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found — run "
            "`python -m risksense.pipelines.returns_pipeline` first."
        )
    df = pd.read_parquet(path).sort_values("date")
    return pd.Series(
        df["portfolio_return"].to_numpy(),
        index=pd.to_datetime(df["date"]),
        name="portfolio_return",
    )


def run(output_dir: Path | None = None) -> dict[str, object]:
    """Compute VaR/ES, flag exceptions, run backtests, write outputs."""
    cfg = load_config("model_params")
    var_cfg = cfg["var"]
    bt_cfg = cfg["backtesting"]

    returns = load_portfolio_returns()
    output_dir = output_dir or data_dir() / "processed"

    result = historical_var_es(
        returns,
        window=int(var_cfg["window_days"]),
        var_level=float(var_cfg["var_level"]),
        es_level=float(var_cfg["es_level"]),
        horizon_days=int(var_cfg["horizon_days"]),
    )
    validate_result_frame(result)
    result = realized_exceptions(result, returns)
    result.to_parquet(output_dir / "var_results.parquet", index=False)

    # Kupiec over the full backtest history; Basel traffic light over the
    # most recent 250-day regulatory window.
    exceptions = result["exception"].to_numpy()
    kupiec = kupiec_pof_test(
        exceptions,
        coverage=float(var_cfg["var_level"]),
        alpha=float(bt_cfg["alpha"]),
    )
    basel_window = int(bt_cfg["basel_window"])
    recent = exceptions[-basel_window:]
    basel = basel_traffic_light(
        int(recent.sum()), n_obs=len(recent), coverage=float(var_cfg["var_level"])
    )

    summary: dict[str, object] = {
        "method": "historical",
        "n_forecast_days": int(len(result)),
        "kupiec_full_sample": kupiec.to_dict(),
        "basel_traffic_light_last_250d": basel.to_dict(),
    }
    with (output_dir / "backtest_summary.json").open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    print(
        f"Historical VaR: {len(result):,} forecast days | "
        f"exceptions {kupiec.n_exceptions} vs expected "
        f"{kupiec.expected_exceptions:.1f} | Kupiec p={kupiec.p_value:.3f} | "
        f"Basel zone (last {len(recent)}d): {basel.zone.upper()}"
    )
    return summary


if __name__ == "__main__":
    run()
