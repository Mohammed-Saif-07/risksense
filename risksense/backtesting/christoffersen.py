"""Christoffersen (1998) independence and conditional coverage tests.

Reference: Christoffersen, P. (1998), "Evaluating Interval Forecasts",
*International Economic Review* 39(4), 841-862.

Kupiec's POF test only checks the *number* of exceptions; a model can pass
it while its exceptions cluster (all in one crisis week — exactly when it
matters). Christoffersen adds:

* **Independence (LR_ind)**: model the hit sequence as a first-order Markov
  chain with transition probabilities π01 = P(hit | no hit yesterday) and
  π11 = P(hit | hit yesterday). Under H0 (independence) π01 = π11.
  LR_ind = -2 [ln L(π̂) - ln L(π̂01, π̂11)] ~ χ²(1).

* **Conditional coverage (LR_cc)**: joint test,
  LR_cc = LR_uc + LR_ind ~ χ²(2), where LR_uc is Kupiec's statistic.

Regulatory mapping: Basel III backtesting / SR 11-7 outcomes analysis
(exception clustering is the standard supervisory follow-up question).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from risksense.backtesting.kupiec import kupiec_pof_test


@dataclass(frozen=True)
class ChristoffersenResult:
    """Outcome of the Christoffersen independence + conditional coverage tests."""

    n_obs: int
    n_exceptions: int
    coverage: float
    # Transition counts: n_ij = days with state i yesterday and j today.
    n00: int
    n01: int
    n10: int
    n11: int
    pi01: float  # P(exception today | none yesterday)
    pi11: float  # P(exception today | exception yesterday)
    lr_uc: float
    p_uc: float
    lr_ind: float
    p_ind: float
    lr_cc: float
    p_cc: float
    reject_independence: bool
    reject_conditional_coverage: bool
    alpha: float

    def to_dict(self) -> dict[str, Any]:
        """Serialisable form for reports and the dashboard."""
        return asdict(self)


def _bernoulli_loglik(successes: int, failures: int, p: float) -> float:
    """Log-likelihood of a Bernoulli sample; empty categories contribute 0."""
    ll = 0.0
    if failures > 0 and p < 1.0:
        ll += failures * np.log(1.0 - p)
    if successes > 0 and p > 0.0:
        ll += successes * np.log(p)
    return ll


def christoffersen_test(
    exceptions: pd.Series | np.ndarray | list[bool],
    coverage: float = 0.99,
    alpha: float = 0.05,
) -> ChristoffersenResult:
    """Run Christoffersen's independence and conditional coverage tests.

    Parameters
    ----------
    exceptions:
        Boolean daily exception indicators, in date order (consecutive
        trading days — the Markov structure is over adjacent observations).
    coverage:
        VaR confidence level q; expected exception probability is 1 - q.
    alpha:
        Significance level for rejection flags.

    Returns
    -------
    ChristoffersenResult
        Transition counts, LR_ind and joint LR_cc with χ² p-values.

    Notes
    -----
    With zero exceptions the independence test is vacuous: LR_ind = 0,
    p_ind = 1, and LR_cc reduces to Kupiec's statistic.
    """
    hits = np.asarray(exceptions, dtype=bool)
    t = len(hits)
    if t < 2:
        raise ValueError("Need at least 2 observations for transition counts")

    prev, curr = hits[:-1], hits[1:]
    n00 = int(np.sum(~prev & ~curr))
    n01 = int(np.sum(~prev & curr))
    n10 = int(np.sum(prev & ~curr))
    n11 = int(np.sum(prev & curr))

    pi01 = n01 / (n00 + n01) if (n00 + n01) > 0 else 0.0
    pi11 = n11 / (n10 + n11) if (n10 + n11) > 0 else 0.0
    pi = (n01 + n11) / (t - 1)

    ll_h0 = _bernoulli_loglik(n01 + n11, n00 + n10, pi)
    ll_h1 = _bernoulli_loglik(n01, n00, pi01) + _bernoulli_loglik(n11, n10, pi11)
    lr_ind = max(-2.0 * (ll_h0 - ll_h1), 0.0)
    p_ind = float(stats.chi2.sf(lr_ind, df=1))

    uc = kupiec_pof_test(hits, coverage=coverage, alpha=alpha)
    lr_cc = uc.lr_stat + lr_ind
    p_cc = float(stats.chi2.sf(lr_cc, df=2))

    return ChristoffersenResult(
        n_obs=t,
        n_exceptions=int(hits.sum()),
        coverage=coverage,
        n00=n00,
        n01=n01,
        n10=n10,
        n11=n11,
        pi01=pi01,
        pi11=pi11,
        lr_uc=uc.lr_stat,
        p_uc=uc.p_value,
        lr_ind=lr_ind,
        p_ind=p_ind,
        lr_cc=lr_cc,
        p_cc=p_cc,
        reject_independence=bool(p_ind < alpha),
        reject_conditional_coverage=bool(p_cc < alpha),
        alpha=alpha,
    )
