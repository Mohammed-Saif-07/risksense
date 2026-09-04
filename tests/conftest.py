"""Shared pytest fixtures.

Tests use synthetic draws from *known* distributions on purpose: known-answer
testing requires knowing the true quantile. The production pipeline itself
never touches synthetic data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

N_SYNTH_DAYS = 2_000
SEED = 12345


@pytest.fixture(scope="session")
def rng() -> np.random.Generator:
    """Seeded RNG so every test run is identical."""
    return np.random.default_rng(SEED)


@pytest.fixture(scope="session")
def normal_returns(rng: np.random.Generator) -> pd.Series:
    """Daily returns from N(0.0003, 0.01^2) with a business-day index."""
    dates = pd.bdate_range("2015-01-01", periods=N_SYNTH_DAYS)
    values = rng.normal(loc=0.0003, scale=0.01, size=N_SYNTH_DAYS)
    return pd.Series(values, index=dates, name="portfolio_return")


@pytest.fixture(scope="session")
def fat_tailed_returns(rng: np.random.Generator) -> pd.Series:
    """Student-t(4) returns scaled to ~1% daily vol — fat tails."""
    dates = pd.bdate_range("2015-01-01", periods=N_SYNTH_DAYS)
    dof = 4.0
    scale = 0.01 / np.sqrt(dof / (dof - 2.0))  # unit-variance t, then 1% vol
    values = rng.standard_t(dof, size=N_SYNTH_DAYS) * scale
    return pd.Series(values, index=dates, name="portfolio_return")
