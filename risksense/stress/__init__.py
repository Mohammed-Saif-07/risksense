"""Stress testing: historical replay, hypothetical scenarios, reverse stress.

Run everything with ``python -m risksense.stress`` after the returns
pipeline and macro ingestion; outputs land in
``data/processed/stress_results.json`` for the dashboard and reports.
"""

from risksense.stress.historical_scenarios import (  # noqa: F401
    HistoricalReplayResult,
    replay_scenario,
)
from risksense.stress.hypothetical import ScenarioResult, apply_scenario  # noqa: F401
from risksense.stress.reverse import ReverseStressResult, reverse_stress  # noqa: F401
from risksense.stress.sensitivities import (  # noqa: F401
    FactorSensitivities,
    estimate_sensitivities,
    load_factor_changes,
)
