"""Tests for config loading and the shipped YAML files."""

from __future__ import annotations

import pytest

from risksense.config import load_config


class TestConfigs:
    def test_model_params_load_and_have_regulatory_anchors(self) -> None:
        cfg = load_config("model_params")
        assert cfg["var"]["var_level"] == 0.99  # Basel III
        assert cfg["var"]["es_level"] == 0.975  # FRTB
        assert cfg["var"]["window_days"] >= 250  # Basel III minimum
        assert cfg["backtesting"]["basel_window"] == 250

    def test_data_config_loads(self) -> None:
        cfg = load_config("data")
        assert "prices" in cfg and "macro" in cfg
        assert len(cfg["macro"]["series"]) >= 5

    def test_scenarios_config_has_required_scenarios(self) -> None:
        cfg = load_config("scenarios")
        assert set(cfg["historical"]) == {"gfc_2008", "covid_2020", "svb_2023"}
        assert len(cfg["hypothetical"]) >= 6  # spec minimum

    def test_missing_config_raises(self) -> None:
        with pytest.raises(FileNotFoundError):
            load_config("does_not_exist")
