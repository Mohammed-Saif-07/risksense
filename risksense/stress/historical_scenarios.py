"""Historical scenario replay (Week 3).

Replays realised factor paths over crisis windows against today's portfolio
(CCAR-style historical scenarios; FRTB stressed ES calibration period logic):

* 2008 GFC:  2008-09-01 → 2009-03-31
* COVID-19:  2020-02-19 → 2020-04-07
* SVB/regional banks: 2023-03-08 → 2023-03-24

Windows are defined in ``config/scenarios.yaml`` — no dates hard-coded here.
Output: scenario P&L path + waterfall factor attribution.
"""

from __future__ import annotations


def replay_scenario(*args: object, **kwargs: object) -> None:
    """Placeholder — implemented in Week 3. See module docstring."""
    raise NotImplementedError("Historical scenario replay ships in Week 3.")
