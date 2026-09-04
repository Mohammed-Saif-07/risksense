"""Tests for the Basel traffic-light classification (BCBS 1996)."""

from __future__ import annotations

import pytest

from risksense.backtesting.basel_traffic_light import basel_traffic_light


class TestBaselZones:
    @pytest.mark.parametrize(
        "x,zone",
        [
            (0, "green"),
            (4, "green"),
            (5, "yellow"),
            (9, "yellow"),
            (10, "red"),
            (25, "red"),
        ],
    )
    def test_zone_boundaries(self, x: int, zone: str) -> None:
        assert basel_traffic_light(x).zone == zone

    @pytest.mark.parametrize(
        "x,addon", [(4, 0.0), (5, 0.40), (7, 0.65), (9, 0.85), (10, 1.0)]
    )
    def test_bcbs_multiplier_addons(self, x: int, addon: float) -> None:
        assert basel_traffic_light(x).multiplier_addon == pytest.approx(addon)

    def test_cumulative_probability_matches_binomial(self) -> None:
        # BCBS (1996): P(X <= 4 | T=250, p=0.01) ≈ 0.8922.
        res = basel_traffic_light(4)
        assert res.cumulative_probability == pytest.approx(0.8922, abs=0.001)

    def test_invalid_inputs_raise(self) -> None:
        with pytest.raises(ValueError):
            basel_traffic_light(-1)
        with pytest.raises(ValueError):
            basel_traffic_light(300, n_obs=250)

    def test_serialisable(self) -> None:
        d = basel_traffic_light(6).to_dict()
        assert d["zone"] == "yellow"
        assert d["multiplier_addon"] == pytest.approx(0.50)
