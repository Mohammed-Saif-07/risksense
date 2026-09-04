"""Granger causality: Narrative Risk Score → VaR breaches (Week 4).

Tests whether monthly NRS changes Granger-cause VaR exception clustering
(Granger 1969), with lag-structure analysis via
``statsmodels.tsa.stattools.grangercausalitytests``. This is a validation
step, not a causal claim — see docs/limitations.md.
"""

from __future__ import annotations


def granger_nrs_breaches(*args: object, **kwargs: object) -> None:
    """Placeholder — implemented in Week 4. See module docstring."""
    raise NotImplementedError("Granger analysis ships in Week 4.")
