"""Integration test for the CLI driver (multi-engine VaR + backtest chain)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from risksense import cli


@pytest.fixture()
def processed_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, rng: np.random.Generator
) -> Path:
    """Point the CLI at a temp data dir with synthetic portfolio + ticker returns."""
    processed = tmp_path / "processed"
    processed.mkdir(parents=True)
    (tmp_path / "raw").mkdir()

    n_days, n_tickers = 1200, 8
    dates = pd.bdate_range("2018-01-01", periods=n_days)
    tickers = [f"T{i}" for i in range(n_tickers)]
    asset_r = rng.normal(0.0003, 0.012, size=(n_days, n_tickers))

    long = pd.DataFrame(
        {
            "date": np.repeat(dates, n_tickers),
            "ticker": np.tile(tickers, n_days),
            "log_return": asset_r.ravel(),
            "winsorized": False,
        }
    )
    long.to_parquet(processed / "returns.parquet", index=False)

    pd.DataFrame(
        {
            "date": dates,
            "portfolio_return": asset_r.mean(axis=1),
            "n_constituents": n_tickers,
        }
    ).to_parquet(processed / "portfolio_returns.parquet", index=False)

    monkeypatch.setattr(cli, "data_dir", lambda: tmp_path)
    return processed


class TestCliRun:
    def test_end_to_end_outputs(self, processed_dir: Path) -> None:
        methods = ["historical", "parametric_normal", "parametric_t"]
        summary = cli.run(methods=methods)

        results = pd.read_parquet(processed_dir / "var_results.parquet")
        assert set(results["method"]) == set(methods)
        assert {"var", "es", "var_at_es_level", "exception"} <= set(results.columns)
        per_method = results.groupby("method").size()
        assert (per_method == 1200 - 250).all()

        with (processed_dir / "backtest_summary.json").open() as fh:
            on_disk = json.load(fh)
        assert on_disk == summary
        for m in methods:
            assert {
                "kupiec",
                "christoffersen",
                "dynamic_quantile",
                "es_acerbi_szekely_z2",
                "basel_traffic_light_last_250d",
            } <= set(on_disk[m])
            assert on_disk[m]["basel_traffic_light_last_250d"]["zone"] in {
                "green",
                "yellow",
                "red",
            }

    def test_unknown_method_raises(self, processed_dir: Path) -> None:
        with pytest.raises(ValueError, match="Unknown methods"):
            cli.run(methods=["quantum_var"])

    def test_missing_inputs_give_actionable_error(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(cli, "data_dir", lambda: tmp_path)
        (tmp_path / "processed").mkdir()
        with pytest.raises(FileNotFoundError, match="returns_pipeline"):
            cli.run()
