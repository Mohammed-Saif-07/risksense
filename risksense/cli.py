"""RiskSense command-line driver.

Runs every implemented VaR/ES engine over the processed portfolio returns,
then the full backtesting suite per engine, writing dashboard-ready outputs:

    data/processed/var_results.parquet   — uniform-schema forecasts for ALL
                                            methods + realised returns +
                                            exception flags
    data/processed/backtest_summary.json — per-method Kupiec, Christoffersen,
                                            DQ, Acerbi-Szekely Z2 and Basel
                                            traffic light

Usage:
    python -m risksense.cli                       # all engines
    python -m risksense.cli --methods historical parametric_t
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import Any

import pandas as pd

from risksense.backtesting.basel_traffic_light import basel_traffic_light
from risksense.backtesting.christoffersen import christoffersen_test
from risksense.backtesting.dq_test import dq_test
from risksense.backtesting.kupiec import kupiec_pof_test
from risksense.config import data_dir, load_config
from risksense.var.base import realized_exceptions, validate_result_frame
from risksense.var.expected_shortfall import es_backtest_z2
from risksense.var.historical import historical_var_es
from risksense.var.monte_carlo import monte_carlo_var_es
from risksense.var.parametric import parametric_var_es


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


def load_wide_returns(path: Path | None = None) -> pd.DataFrame:
    """Load per-ticker returns as a date × ticker matrix (for Ledoit-Wolf)."""
    path = path or data_dir() / "processed" / "returns.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found — run "
            "`python -m risksense.pipelines.returns_pipeline` first."
        )
    df = pd.read_parquet(path)
    wide = df.pivot_table(index="date", columns="ticker", values="log_return")
    wide.index = pd.to_datetime(wide.index)
    return wide.sort_index()


def build_engines(
    returns: pd.Series, wide: pd.DataFrame, cfg: dict[str, Any]
) -> dict[str, Callable[[], pd.DataFrame]]:
    """Map method name → zero-arg callable computing its forecast frame."""
    var_cfg = cfg["var"]
    par_cfg = cfg["parametric"]
    mc_cfg = cfg["monte_carlo"]
    weights = load_config("portfolio").get("weights")

    # Annotated dict[str, Any]: the shared kwargs mix int and float, which
    # mypy would otherwise narrow to dict[str, float] and reject at every
    # ``**common`` call site.
    common: dict[str, Any] = dict(
        window=int(var_cfg["window_days"]),
        var_level=float(var_cfg["var_level"]),
        es_level=float(var_cfg["es_level"]),
        horizon_days=int(var_cfg["horizon_days"]),
    )
    lo, hi = par_cfg["t_dof_bounds"]
    dof_bounds: tuple[float, float] = (float(lo), float(hi))
    lw_wide = wide if bool(par_cfg["ledoit_wolf_shrinkage"]) else None
    garch = mc_cfg["garch"]

    # functools.partial rather than `lambda d=dist:` — the default-argument
    # trick for binding a loop variable is easy to get subtly wrong (late
    # binding) and mypy cannot check it against Callable[[], DataFrame].
    engines: dict[str, Callable[[], pd.DataFrame]] = {
        "historical": partial(historical_var_es, returns, **common),
    }
    for dist in ("normal", "t"):
        engines[f"parametric_{dist}"] = partial(
            parametric_var_es,
            returns,
            wide_returns=lw_wide,
            distribution=dist,
            cov_refit_days=int(par_cfg["cov_refit_days"]),
            dof_bounds=dof_bounds,
            weights=weights,
            **common,
        )
    for dist in mc_cfg["distributions"]:
        engines[f"monte_carlo_{dist}"] = partial(
            monte_carlo_var_es,
            returns,
            wide_returns=lw_wide,
            distribution=dist,
            n_paths=int(mc_cfg["n_paths"]),
            seed=int(mc_cfg["seed"]),
            cov_refit_days=int(par_cfg["cov_refit_days"]),
            dof_bounds=dof_bounds,
            garch_refit_days=int(garch["refit_days"]),
            garch_fit_window=int(garch["fit_window_days"]),
            garch_min_window=int(garch["min_window_days"]),
            weights=weights,
            **common,
        )
    return engines


def backtest_method(result: pd.DataFrame, cfg: dict[str, Any]) -> dict[str, Any]:
    """Run the full backtest suite on one method's exception-flagged frame."""
    var_cfg, bt_cfg = cfg["var"], cfg["backtesting"]
    coverage = float(var_cfg["var_level"])
    alpha = float(bt_cfg["alpha"])
    exceptions = result["exception"].to_numpy()

    kupiec = kupiec_pof_test(exceptions, coverage=coverage, alpha=alpha)
    christ = christoffersen_test(exceptions, coverage=coverage, alpha=alpha)
    dq = dq_test(
        exceptions,
        result["var"].to_numpy(),
        coverage=coverage,
        lags=int(bt_cfg["dq_lags"]),
        alpha=alpha,
    )
    # Z2's exception indicator must be at the same level as the ES forecast
    # (97.5%), hence var_at_es_level rather than the 99% VaR.
    es_bt = es_backtest_z2(
        result["realized_return"].to_numpy(),
        result["var_at_es_level"].to_numpy(),
        result["es"].to_numpy(),
        es_level=float(var_cfg["es_level"]),
        alpha=alpha,
        n_bootstrap=int(bt_cfg["es_bootstrap"]),
    )
    basel_window = int(bt_cfg["basel_window"])
    recent = exceptions[-basel_window:]
    basel = basel_traffic_light(int(recent.sum()), n_obs=len(recent), coverage=coverage)

    return {
        "n_forecast_days": int(len(result)),
        "first_date": str(result["date"].iloc[0].date()),
        "kupiec": kupiec.to_dict(),
        "christoffersen": christ.to_dict(),
        "dynamic_quantile": dq.to_dict(),
        "es_acerbi_szekely_z2": es_bt.to_dict(),
        "basel_traffic_light_last_250d": basel.to_dict(),
    }


def run(
    methods: list[str] | None = None, output_dir: Path | None = None
) -> dict[str, Any]:
    """Compute all engines, flag exceptions, run backtests, write outputs."""
    cfg = load_config("model_params")
    returns = load_portfolio_returns()
    wide = load_wide_returns()
    output_dir = output_dir or data_dir() / "processed"

    engines = build_engines(returns, wide, cfg)
    if methods:
        unknown = set(methods) - set(engines)
        if unknown:
            raise ValueError(f"Unknown methods {unknown}; have {sorted(engines)}")
        engines = {m: engines[m] for m in methods}

    frames: list[pd.DataFrame] = []
    summary: dict[str, Any] = {}
    for name, engine in engines.items():
        t0 = time.perf_counter()
        result = engine()
        validate_result_frame(result)
        result = realized_exceptions(result, returns)
        frames.append(result)
        summary[name] = backtest_method(result, cfg)
        k = summary[name]["kupiec"]
        z = summary[name]["basel_traffic_light_last_250d"]["zone"]
        print(
            f"{name:22s} {len(result):5d} days | exc {k['n_exceptions']:3d} "
            f"vs {k['expected_exceptions']:6.1f} | Kupiec p={k['p_value']:.3f} | "
            f"CC p={summary[name]['christoffersen']['p_cc']:.3f} | "
            f"DQ p={summary[name]['dynamic_quantile']['p_value']:.3f} | "
            f"{z.upper():6s} | {time.perf_counter() - t0:5.1f}s"
        )

    all_results = pd.concat(frames, ignore_index=True)
    all_results.to_parquet(output_dir / "var_results.parquet", index=False)
    with (output_dir / "backtest_summary.json").open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    print(f"Wrote {output_dir / 'var_results.parquet'} and backtest_summary.json")
    return summary


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--methods", nargs="*", default=None)
    args = parser.parse_args()
    run(methods=args.methods)


if __name__ == "__main__":
    main()
