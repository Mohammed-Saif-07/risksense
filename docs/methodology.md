# Methodology

Full derivations and estimator choices for every RiskSense module. Sections
are added as each weekly milestone ships; regulatory anchors are cited inline.

## 1. Data and returns (Week 1)

- **Universe**: current S&P 500 constituents (cached from Wikipedia to
  `config/sp500_universe.csv`). Known limitation: survivorship bias — see
  `docs/limitations.md`.
- **Prices**: Yahoo Finance adjusted closes (`auto_adjust=True`, so
  split- and dividend-adjusted = total-return prices); Stooq fallback rows
  are split-adjusted only and carry `source='stooq'` for the DQ report.
- **Returns**: daily log returns, `r_t = ln(P_t / P_{t-1})`. Log returns are
  time-additive and the natural input for the GARCH and parametric engines.
- **Data quality (SR 11-7)**: non-positive prices dropped; (date, ticker)
  deduplicated; tickers below 300 observations excluded; |r| > 0.60 treated
  as data error, winsorised and flagged (thresholds in
  `config/model_params.yaml`).
- **Portfolio**: equal-weight, daily-rebalanced across constituents present
  each day; portfolio log return approximated by the weighted mean of
  constituent log returns (exact for the arithmetic-return portfolio up to
  second-order terms; documented approximation).

## 2. Historical Simulation VaR/ES (Week 1)

For forecast day *t*, the loss distribution is the empirical distribution of
the previous 250 daily portfolio losses (Jorion 2007, ch. 10). No look-ahead:
the window ends at *t-1*.

- VaR_q = empirical q-quantile of losses ('higher' interpolation, so the
  estimate is an observed loss and small-sample coverage is conservative).
- ES_q = mean of losses ≥ VaR_q (Acerbi & Tasche 2002). Reported at 97.5%
  per FRTB (BCBS 2019, MAR33) alongside 99% VaR (Basel III backtesting
  anchor).

## 3. Backtesting (Week 1: Kupiec + traffic light; Week 2: full suite)

- **Kupiec (1995) POF**: LR test of unconditional coverage against
  Binomial(T, 1-q); chi-squared(1) asymptotics. Both under- *and*
  over-coverage reject — an over-conservative model wastes capital.
- **Basel traffic light (BCBS 1996)**: exceptions in the trailing 250 days →
  Green (0-4), Yellow (5-9, multiplier add-on 0.40-0.85), Red (10+, add-on
  1.00).
- Week 2: Christoffersen (1998) independence/conditional coverage;
  Engle-Manganelli (2004) Dynamic Quantile.

## 4-7. Parametric, Monte Carlo, stress testing, narrative overlay

Ship with Weeks 2-4; sections will be filled as the code lands.
