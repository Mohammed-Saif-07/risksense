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

## 3. Backtesting suite

- **Kupiec (1995) POF**: LR test of unconditional coverage against
  Binomial(T, 1-q); chi-squared(1) asymptotics. Both under- *and*
  over-coverage reject — an over-conservative model wastes capital.
- **Christoffersen (1998)**: exception clustering via a first-order Markov
  chain — LR_ind tests π01 = π11 (chi-squared 1), LR_cc = LR_uc + LR_ind
  (chi-squared 2) tests coverage and independence jointly.
- **Engle-Manganelli (2004) Dynamic Quantile**: Wald test that the demeaned
  hit sequence is unpredictable from a constant, 4 lagged hits and the VaR
  forecast itself; chi-squared(6). Catches longer-range clustering and
  hits correlated with the forecast level.
- **Acerbi-Székely (2014) Z₂** for ES: since ES is not elicitable
  (Gneiting 2011), the Z₂ statistic compares realised tail losses to the
  promised ES at the same 97.5% level; significance via an i.i.d. bootstrap
  (approximation — see limitations #11).
- **Basel traffic light (BCBS 1996)**: exceptions in the trailing 250 days →
  Green (0-4), Yellow (5-9, multiplier add-on 0.40-0.85), Red (10+, add-on
  1.00).

## 4. Parametric VaR/ES (Week 2)

Rolling variance-covariance VaR with the portfolio volatility from the full
constituent covariance: σ_p² = w'Σw, Σ the **Ledoit-Wolf (2004)** shrunk
estimator over ~500 assets (the raw 250-observation sample covariance is
singular at that dimension). Σ is re-estimated monthly (`cov_refit_days`);
the rolling mean updates daily. Two innovation models:

- **Normal**: VaR_q = -μ + σz_q, ES_q = -μ + σφ(z_q)/(1-q).
- **Student-t**: ν by method of moments from window excess kurtosis
  (ν = 4 + 6/κ, clipped to [3, 50]); closed-form ES per
  McNeil-Frey-Embrechts (2015) §2.3.

## 5. Monte Carlo VaR/ES (Week 2)

10,000 seeded paths per forecast date, VaR/ES from empirical quantiles of
simulated losses, under three DGPs:

- **MV normal / MV Student-t**: for a linear portfolio the projection
  w'X is exactly univariate (normal, or t with the same ν), so paths are
  drawn from the projected distribution with Σ entering through w'Σw —
  exact for this portfolio, would not survive optionality.
- **GARCH(1,1)-t** (Bollerslev 1986): fitted on the portfolio series with
  the `arch` package, parameters re-fit monthly on a trailing 1,000-day
  window; between refits the variance recursion
  σ²_t = ω + αε²_{t-1} + βσ²_{t-1} updates daily so forecasts react to
  yesterday's shock. First forecast requires 750 days of history.

Benchmarking result on the real 2006-2026 sample (SR 11-7 §5): exception
counts vs ~49.5 expected — parametric/MC normal ≈ 156, Student-t ≈ 130,
historical 77, GARCH-t 66. The ordering is the textbook one: constant-vol
normal understates tails worst; fat tails help; conditional volatility
helps most.

## 6-7. Stress testing, narrative overlay

Ship with Weeks 3-4; sections will be filled as the code lands.
