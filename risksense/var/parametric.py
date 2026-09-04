"""Parametric (variance-covariance) VaR/ES.

Planned methodology (Week 2)
----------------------------
* Normal and Student-t innovations (Jorion 2007, ch. 8).
* Ledoit-Wolf covariance shrinkage for the ~500-asset universe (Ledoit &
  Wolf 2004, "A well-conditioned estimator for large-dimensional covariance
  matrices", *J. Multivariate Analysis*), via
  ``sklearn.covariance.LedoitWolf``.
* Closed-form ES for both distributions (McNeil, Frey & Embrechts,
  *Quantitative Risk Management*, 2015, §2.3).

Output: uniform schema from :mod:`risksense.var.base`, methods
``parametric_normal`` and ``parametric_t``.
"""

from __future__ import annotations


def parametric_var_es(*args: object, **kwargs: object) -> None:
    """Placeholder — implemented in Week 2. See module docstring."""
    raise NotImplementedError("Parametric VaR ships in Week 2.")
