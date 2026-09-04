"""Expected Shortfall utilities (FRTB 97.5% standard).

ES is already computed inside each engine (see
:func:`risksense.var.historical.hs_es`); this module adds cross-engine
helpers in Week 2:

* ES back-allocation to constituents (Euler allocation).
* ES backtest via the Acerbi-Szekely (2014) Z-statistics — ES is not
  elicitable (Gneiting 2011), so it is backtested jointly with VaR.

References: Acerbi & Tasche (2002); McNeil, Frey & Embrechts (2015), §2.3;
BCBS (2019) FRTB MAR33.
"""

from __future__ import annotations


def es_allocation(*args: object, **kwargs: object) -> None:
    """Placeholder — implemented in Week 2. See module docstring."""
    raise NotImplementedError("ES allocation ships in Week 2.")
