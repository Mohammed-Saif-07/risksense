# Model Validation Report — RiskSense VaR/ES Suite

*Structured after Federal Reserve SR 11-7 / OCC 2011-12. Generated content
lands in Month 4 via `risksense/validation/report.py`; this skeleton fixes the
section structure now so every module writes toward it.*

## 1. Model Purpose & Scope
*(Month 4)* One-day VaR 99% / ES 97.5% for a long-only S&P 500 equity
portfolio; intended use: learning/demonstration, not capital calculation.

## 2. Conceptual Soundness
*(Month 4)* Estimator choices and their theoretical basis, per engine.

## 3. Data Quality Assessment
*(Month 4, auto-generated)* Coverage, source mix (yahoo/stooq), winsorised
observations, excluded tickers.

## 4. Backtesting Results
*(Month 2+)* Kupiec, Christoffersen, DQ, Basel zones — per engine, per
sub-period (calm vs. crisis).

## 5. Benchmarking
*(Month 2+)* All four engines head-to-head on identical dates: exception
rates, ES tail-severity ratios, capital implications.

## 6. Sensitivity Analysis
*(Month 4)* Window length, confidence level, shrinkage intensity, MC path
count and seed stability.

## 7. Model Limitations & Assumptions
See [limitations.md](limitations.md) — maintained continuously, not
retrofitted.

## 8. Ongoing Monitoring Plan
*(Month 4)* Weekly re-run cadence, traffic-light escalation thresholds,
data-drift checks, NRS overlay as a leading indicator.
