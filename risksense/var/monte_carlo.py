"""Monte Carlo VaR/ES.

Planned methodology (Week 2)
----------------------------
10,000+ simulated one-day P&L paths under three data-generating processes:

a. Multivariate normal (Cholesky of the Ledoit-Wolf shrunk covariance).
b. Multivariate Student-t (fat tails; dof fitted by MLE).
c. GARCH(1,1) with Student-t innovations for volatility clustering
   (Bollerslev 1986; ``arch`` package), portfolio-level.

Seeded RNG (``numpy.random.default_rng``) for reproducibility; path count
and seed in ``config/model_params.yaml``.

Output: uniform schema from :mod:`risksense.var.base`, method
``monte_carlo`` with a ``distribution`` tag.
"""

from __future__ import annotations


def monte_carlo_var_es(*args: object, **kwargs: object) -> None:
    """Placeholder — implemented in Week 2. See module docstring."""
    raise NotImplementedError("Monte Carlo VaR ships in Week 2.")
