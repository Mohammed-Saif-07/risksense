# Model Limitations & Assumptions

Honest accounting of what RiskSense does *not* do, in SR 11-7 spirit: a model
without documented limitations is a model risk. Expanded as modules ship.

## Data

1. **Survivorship bias.** The universe is today's S&P 500 membership.
   Delisted/failed firms (Lehman, SVB as a constituent, etc.) are absent
   from history, biasing volatility and tail estimates downward.
2. **Mixed adjustment basis.** Yahoo rows are total-return adjusted; Stooq
   fallback rows are split-adjusted only. Rows are tagged by `source` and the
   share of Stooq rows is reported in the DQ assessment.
3. **Vendor restatements.** Yahoo adjusted prices can change retroactively;
   the pipeline is rerun-from-source, so results are reproducible only up to
   vendor data stability.

## Historical Simulation

4. **The window is the model.** HS assumes the last 250 days span tomorrow's
   distribution. Regime changes enter with a lag: the model is blind to a
   COVID-sized shock the day before it happens and over-conservative for a
   year afterward (ghost effects as large losses roll out of the window).
5. **Equal weighting within the window.** No decay (cf. BRW/FHS variants);
   a 249-day-old observation counts as much as yesterday's.
6. **One-day horizon only.** We deliberately do not sqrt-scale to 10-day
   VaR: scaling assumes i.i.d. normality and understates fat-tailed,
   autocorrelated risk (documented rather than silently applied).

## Portfolio construction

7. **Linear, long-only equity portfolio.** No optionality, no fixed income,
   no FX. Stress scenarios therefore map macro shocks through estimated
   sensitivities (betas), not full revaluation.
8. **Log-return aggregation approximation.** Portfolio log return is
   computed as the weighted mean of constituent log returns; exact
   arithmetic aggregation differs at second order (negligible at daily
   horizon, stated for completeness).

## Statistical tests

9. **Asymptotic p-values.** Kupiec/Christoffersen LR statistics use
   chi-squared asymptotics; with 250-day windows and 1% coverage, exact
   binomial small-sample behaviour matters (the Basel zones exist precisely
   because of this).
10. **Multiple testing.** Running several backtests across several engines
    inflates the family-wise false-rejection rate; the validation report
    interprets results jointly, not as isolated verdicts.

## Coming with Weeks 2-4

Correlation breakdown in crises (Ledoit-Wolf conditions the matrix, doesn't
make dependence stationary), GARCH parameter instability, FinBERT domain
shift on risk-factor language, and Granger ≠ causation for the NRS overlay.
