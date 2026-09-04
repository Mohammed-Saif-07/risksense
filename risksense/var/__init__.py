"""VaR / Expected Shortfall engines.

All engines emit the uniform result schema defined in
:mod:`risksense.var.base` so that methods can be compared head-to-head
(SR 11-7 benchmarking requirement).
"""

from risksense.var.base import VAR_RESULT_COLUMNS, VaRResult  # noqa: F401
from risksense.var.expected_shortfall import (  # noqa: F401
    es_backtest_z2,
    euler_es_allocation,
)
from risksense.var.historical import historical_var_es  # noqa: F401
from risksense.var.monte_carlo import monte_carlo_var_es  # noqa: F401
from risksense.var.parametric import parametric_var_es  # noqa: F401
