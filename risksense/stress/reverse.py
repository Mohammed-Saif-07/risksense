"""Reverse stress testing: find the smallest shock producing a target loss.

Reference: BCBS, *Stress testing principles* (2018), Principle 6 — banks
should identify the scenarios that would threaten them, not only price the
scenarios they can imagine. EBA/GL/2018/04 §
"reverse stress testing".

Formulation
-----------
Minimise the scaled shock magnitude subject to reaching the target loss:

    min_x  Σ_i (x_i / s_i)²      s.t.  loss(x) ≥ target,  bounds on x

where ``x`` = (equity shock, curve level shift bp, IG OAS widening bp),
``s`` = per-factor scales making magnitudes comparable (config:
``reverse.scales`` — e.g. a 1% equity move "costs" the same as a 10bp rate
move), and ``loss(x) = -(x_e + β_lvl x_lvl + β_ig x_ig)``.

Solved with SLSQP (``scipy.optimize.minimize``) as the spec's
gradient-based search. For this *linear* loss the unbounded problem has a
closed form (Lagrange: x*_i ∝ -s_i² β_i), which the test suite uses to
validate the optimiser; the optimiser earns its keep once bounds bind
(credit cannot tighten in a stress scenario, equity cannot fall more than
100%) and remains the scaffold for any nonlinear loss later.

Regulatory mapping: BCBS 2018 P6, CCAR severely-adverse design intuition.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from risksense.config import load_config
from risksense.stress.sensitivities import FactorSensitivities

#: Effective beta of each reverse-stress factor (equity = unit pass-through).
_EQUITY_KEY = "equity"


@dataclass(frozen=True)
class ReverseStressResult:
    """Smallest-shock combination reaching the target loss."""

    target_loss_frac: float
    achieved_loss_frac: float
    shocks: dict[str, float]  # factor → shock in its native unit
    scaled_magnitude: float  # sqrt of the minimised objective
    converged: bool
    n_iterations: int
    binding_bounds: list[str]

    def to_dict(self) -> dict[str, Any]:
        """Serialisable form for reports and the dashboard."""
        return asdict(self)


def _effective_betas(factors: list[str], sens: FactorSensitivities) -> np.ndarray:
    """Loss sensitivity per factor: loss = -(Σ beta_i · x_i)."""
    betas = []
    for f in factors:
        if f == _EQUITY_KEY:
            betas.append(1.0)  # unit pass-through, all-equity portfolio
        else:
            betas.append(sens.betas[f])
    return np.asarray(betas, dtype=float)


def reverse_stress(
    sens: FactorSensitivities,
    target_loss_frac: float | None = None,
    factors: list[str] | None = None,
    scales: dict[str, float] | None = None,
    bounds: dict[str, list[float]] | None = None,
) -> ReverseStressResult:
    """Search for the minimal shock combination producing the target loss.

    Parameters default to the ``reverse`` block of
    ``config/scenarios.yaml``.

    Returns
    -------
    ReverseStressResult
        Native-unit shocks (equity fraction, bp for rates/credit), the
        achieved loss, and which bounds bind at the optimum.

    Raises
    ------
    RuntimeError
        If the optimiser cannot reach the target within bounds (the target
        is infeasible for this factor set — itself a reportable finding).
    """
    from scipy.optimize import minimize

    cfg = load_config("scenarios")["reverse"]
    target = float(
        target_loss_frac
        if target_loss_frac is not None
        else cfg["target_loss_fraction"]
    )
    factors = list(factors if factors is not None else cfg["factors"])
    scales_map = dict(scales if scales is not None else cfg["scales"])
    bounds_map = dict(bounds if bounds is not None else cfg["bounds"])

    beta = _effective_betas(factors, sens)
    s = np.asarray([float(scales_map[f]) for f in factors])
    bnds = [tuple(float(v) for v in bounds_map[f]) for f in factors]

    def objective(x: np.ndarray) -> float:
        return float(np.sum((x / s) ** 2))

    def loss(x: np.ndarray) -> float:
        return float(-(beta @ x))

    # Warm start at the unbounded closed-form optimum, clipped into bounds.
    denom = float(np.sum((s * beta) ** 2))
    x0 = -target * (s**2) * beta / denom
    x0 = np.clip(x0, [b[0] for b in bnds], [b[1] for b in bnds])

    res = minimize(
        objective,
        x0,
        method="SLSQP",
        bounds=bnds,
        constraints=[{"type": "ineq", "fun": lambda x: loss(x) - target}],
        options={"maxiter": 200, "ftol": 1e-9},
    )
    achieved = loss(res.x)
    # Judge by feasibility, not res.success: when the warm start is already
    # optimal, SLSQP exits with success=False ("positive directional
    # derivative") while sitting exactly on the constraint.
    feasible = achieved >= target * (1 - 1e-4)
    if not feasible:
        raise RuntimeError(
            f"Reverse stress infeasible: best achievable loss "
            f"{achieved:.2%} < target {target:.2%} within bounds {bounds_map} "
            "— report this as a finding, or widen the factor set/bounds."
        )

    tol = 1e-9
    binding = [
        f
        for f, x, (lo, hi) in zip(factors, res.x, bnds, strict=True)
        if abs(x - lo) < tol or abs(x - hi) < tol
    ]
    return ReverseStressResult(
        target_loss_frac=target,
        achieved_loss_frac=achieved,
        shocks={f: float(x) for f, x in zip(factors, res.x, strict=True)},
        scaled_magnitude=float(np.sqrt(res.fun)),
        converged=bool(res.success or feasible),
        n_iterations=int(res.nit),
        binding_bounds=binding,
    )
