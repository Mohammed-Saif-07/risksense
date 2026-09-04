"""Hypothetical stress scenarios (Week 3).

CCAR-style deterministic shocks, all defined in ``config/scenarios.yaml``:
parallel yield shifts (±100bp, ±200bp), steepener, flattener, and combined
equity-crash + credit-spread-widening scenarios. Sensitivities come from the
FRED macro series ingested by ``data/ingest_macro.py``.
"""

from __future__ import annotations


def apply_scenario(*args: object, **kwargs: object) -> None:
    """Placeholder — implemented in Week 3. See module docstring."""
    raise NotImplementedError("Hypothetical scenarios ship in Week 3.")
