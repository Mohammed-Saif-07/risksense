"""Integration test for the CLI driver (VaR + backtest chain)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from risksense import cli


@pytest.fixture()
def processed_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, rng: np.random.Generator) -> Path:
    """Point the CLI at a temp data dir holding synthetic portfolio returns."""
    processed = tmp_path / "processed"
    processed.mkdir(parents=True)
    (tmp_path / "raw").mkdir()

    dates = pd.bdate_range("2018-01-01", periods=1200)
    df = pd.DataFrame(
        {
            "date": dates,
            "portfolio_return": rng.normal(0.0003, 0.01, size=len(dates)),
            "n_constituents": 500,
        }
    )
    df.to_parquet(processed / "portfolio_returns.parquet", index=False)
    monkeypatch.setattr(cli, "data_dir", lambda: tmp_path)
    return processed


class TestCliRun:
    def test_end_to_end_outputs(self, processed_dir: Path) -> None:
        summary = cli.run()

        results = pd.read_parquet(processed_dir / "var_results.parquet")
        assert {"var", "es", "exception", "realized_return"} <= set(results.columns)
        assert len(results) == 1200 - 250  # forecasts start after the window

        with (processed_dir / "backtest_summary.json").open() as fh:
            on_disk = json.load(fh)
        assert on_disk == summary
        assert on_disk["kupiec_full_sample"]["n_obs"] == len(results)
        assert on_disk["basel_traffic_light_last_250d"]["zone"] in {
            "green",
            "yellow",
            "red",
        }

    def test_missing_inputs_give_actionable_error(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(cli, "data_dir", lambda: tmp_path)
        (tmp_path / "processed").mkdir()
        with pytest.raises(FileNotFoundError, match="returns_pipeline"):
            cli.run()
