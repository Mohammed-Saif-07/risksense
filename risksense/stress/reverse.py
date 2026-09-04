"""Reverse stress testing (Week 3).

Gradient-based search for the smallest factor shock producing a target
portfolio loss (BCBS *Principles for sound stress testing practices*, 2018,
Principle 6; also required by EBA guidelines). Implemented as constrained
minimisation of shock magnitude subject to loss >= target, using
``scipy.optimize.minimize`` (SLSQP).
"""

from __future__ import annotations


def reverse_stress(*args: object, **kwargs: object) -> None:
    """Placeholder — implemented in Week 3. See module docstring."""
    raise NotImplementedError("Reverse stress testing ships in Week 3.")
