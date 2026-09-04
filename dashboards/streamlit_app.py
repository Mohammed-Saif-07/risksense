"""RiskSense Streamlit dashboard.

Pages: Portfolio Overview, VaR & Backtesting (per engine), Engine Comparison
(all six head-to-head), Stress Testing (replay, scenarios, reverse), and the
Month 4 narrative-risk placeholder.

Visual language lives in ``dashboards/theme.py`` so every chart renders as
one system. Run locally:  ``streamlit run dashboards/streamlit_app.py``.
Deploys as-is to Streamlit Community Cloud (free tier).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from dashboards import theme as T  # noqa: E402

PROCESSED = REPO_ROOT / "data" / "processed"

st.set_page_config(
    page_title="RiskSense — Market Risk Engine",
    layout="wide",
    page_icon="◱",
    initial_sidebar_state="expanded",
)
st.markdown(T.CSS, unsafe_allow_html=True)


# --------------------------------------------------------------------------
# Data loading
# --------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_outputs() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Load VaR pipeline outputs (portfolio returns, forecasts, backtests)."""
    portfolio = pd.read_parquet(PROCESSED / "portfolio_returns.parquet")
    var_results = pd.read_parquet(PROCESSED / "var_results.parquet")
    with (PROCESSED / "backtest_summary.json").open() as fh:
        summary = json.load(fh)
    portfolio = portfolio.sort_values("date")
    portfolio["date"] = pd.to_datetime(portfolio["date"])
    var_results["date"] = pd.to_datetime(var_results["date"])
    return portfolio, var_results, summary


@st.cache_data(show_spinner=False)
def load_stress() -> dict | None:
    """Load stress results; ``None`` if the stress suite hasn't been run."""
    path = PROCESSED / "stress_results.json"
    if not path.exists():
        return None
    with path.open() as fh:
        return json.load(fh)


def verdict(reject: bool, pass_text: str, fail_text: str) -> tuple[str, str]:
    """Map a hypothesis-test rejection flag to (label, status token)."""
    return (fail_text, "critical") if reject else (pass_text, "good")


# --------------------------------------------------------------------------
# Pages
# --------------------------------------------------------------------------
def page_overview(
    portfolio: pd.DataFrame, var_results: pd.DataFrame, summary: dict
) -> None:
    """Headline risk position and the portfolio's realised history."""
    hist = var_results[var_results["method"] == "historical"]
    latest = hist.iloc[-1]
    basel = summary["historical"]["basel_traffic_light_last_250d"]
    zone = basel["zone"]
    span = f"{portfolio['date'].min():%b %Y} – {portfolio['date'].max():%b %Y}"

    st.markdown(
        T.header(
            "Portfolio Overview",
            "RiskSense · Market Risk Engine",
            f"Equal-weight S&P 500 constituent portfolio, daily-rebalanced, "
            f"{len(portfolio):,} trading days ({span}). One-day risk measures "
            "on log returns: 99% VaR is the Basel III backtesting anchor, "
            "97.5% Expected Shortfall the FRTB standard. Headline figures use "
            "Historical Simulation; the comparison page runs all six engines "
            "side by side.",
        ),
        unsafe_allow_html=True,
    )

    st.markdown(
        T.kpi_row(
            [
                {
                    "label": "VaR 99% · 1 day",
                    "value": f"{latest['var']:.2%}",
                    "note": "Basel III anchor",
                },
                {
                    "label": "ES 97.5% · 1 day",
                    "value": f"{latest['es']:.2%}",
                    "note": "FRTB standard",
                },
                {
                    "label": "Constituents",
                    "value": f"{int(portfolio['n_constituents'].iloc[-1])}",
                    "note": "after data-quality filters",
                },
                {
                    "label": "Basel zone · 250d",
                    "value": zone.upper(),
                    "note": f"{basel['n_exceptions']} exceptions · add-on "
                    f"+{basel['multiplier_addon']:.2f}",
                    "status": T.ZONE_STATUS[zone],
                },
            ]
        ),
        unsafe_allow_html=True,
    )

    cum = (1.0 + portfolio.set_index("date")["portfolio_return"]).cumprod() - 1.0
    fig = go.Figure(
        go.Scatter(
            x=cum.index,
            y=cum.values,
            mode="lines",
            line={"color": T.SERIES[0], "width": 2},
            fill="tozeroy",
            fillcolor="rgba(57,135,229,0.08)",
            name="Cumulative return",
            hovertemplate="%{x|%d %b %Y}<br>%{y:.1%}<extra></extra>",
        )
    )
    # Single series: the title names it, so no legend box is needed.
    T.style_fig(fig, height=360, y_tickformat=".0%", legend=False)
    T.chart_card(
        "Cumulative portfolio return",
        "Growth of 1 unit, log returns compounded. The 2008 and 2020 "
        "drawdowns are the windows the stress page replays.",
        fig,
    )


def page_var_backtest(var_results: pd.DataFrame, summary: dict) -> None:
    """One engine at a time: forecast path, exceptions, and the test suite."""
    st.markdown(
        T.header(
            "VaR & Backtesting",
            "Basel III · SR 11-7 outcomes analysis",
            "Each engine's daily forecast against realised P&L, with the four "
            "statistical tests a supervisor would ask for: does the exception "
            "count match (Kupiec), do exceptions cluster (Christoffersen), are "
            "they predictable (Dynamic Quantile), and is the tail as deep as "
            "promised (Acerbi-Székely).",
        ),
        unsafe_allow_html=True,
    )

    methods = [m for m in T.METHOD_LABELS if m in set(var_results["method"])]
    method = st.selectbox(
        "Engine", methods, format_func=lambda m: T.METHOD_LABELS.get(m, m)
    )
    df = var_results[var_results["method"] == method]
    s = summary[method]
    k, c, d, e = (
        s["kupiec"],
        s["christoffersen"],
        s["dynamic_quantile"],
        s["es_acerbi_szekely_z2"],
    )
    basel = s["basel_traffic_light_last_250d"]

    st.markdown(
        T.kpi_row(
            [
                {
                    "label": "Exceptions",
                    "value": f"{k['n_exceptions']}",
                    "note": f"expected {k['expected_exceptions']:.1f}",
                },
                {
                    "label": "Exception rate",
                    "value": f"{k['exception_rate']:.2%}",
                    "note": "target 1.00%",
                },
                {
                    "label": "Forecast days",
                    "value": f"{s['n_forecast_days']:,}",
                    "note": f"from {s['first_date']}",
                },
                {
                    "label": "Basel zone · 250d",
                    "value": basel["zone"].upper(),
                    "note": f"{basel['n_exceptions']} exceptions",
                    "status": T.ZONE_STATUS[basel["zone"]],
                },
            ]
        ),
        unsafe_allow_html=True,
    )

    exc = df[df["exception"]]
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df["date"],
            y=df["realized_return"],
            mode="lines",
            name="Realised return",
            line={"color": T.REALIZED, "width": 1},
            hovertemplate="%{y:.2%}<extra>Realised</extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df["date"],
            y=-df["var"],
            mode="lines",
            name="VaR 99% bound",
            line={"color": T.METHOD_COLORS[method], "width": 2},
            hovertemplate="%{y:.2%}<extra>VaR 99%</extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df["date"],
            y=-df["es"],
            mode="lines",
            name="ES 97.5% (FRTB)",
            line={"color": T.METHOD_COLORS[method], "width": 1, "dash": "dot"},
            opacity=0.75,
            hovertemplate="%{y:.2%}<extra>ES 97.5%</extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=exc["date"],
            y=exc["realized_return"],
            mode="markers",
            name=f"Exception ({len(exc)})",
            marker={"color": T.LOSS, "size": 8, "symbol": "x", "line": {"width": 0}},
            hovertemplate="%{x|%d %b %Y}<br>loss %{y:.2%}<extra>Exception</extra>",
        )
    )
    T.style_fig(fig, height=430, y_tickformat=".0%")
    T.chart_card(
        f"Daily VaR vs realised P&L — {T.METHOD_LABELS[method]}",
        "An exception is a day whose realised loss broke the VaR forecast "
        "made the evening before.",
        fig,
    )

    left, right = st.columns([3, 2])
    with (
        left,
        T.card(
            "Statistical test suite",
            "p &lt; 0.05 rejects the null hypothesis named on each row.",
        ),
    ):
        v, sst = verdict(k["reject_h0"], "coverage OK", "coverage rejected")
        rows = T.test_row(
            "Kupiec POF · unconditional coverage",
            f"LR {k['lr_stat']:.2f} · p {k['p_value']:.4f}",
            v,
            sst,
        )
        v, sst = verdict(c["reject_independence"], "independent", "clustered")
        rows += T.test_row(
            "Christoffersen · independence",
            f"π01 {c['pi01']:.3f} · π11 {c['pi11']:.3f} · p {c['p_ind']:.4f}",
            v,
            sst,
        )
        v, sst = verdict(c["reject_conditional_coverage"], "joint OK", "joint rejected")
        rows += T.test_row(
            "Christoffersen · conditional coverage",
            f"LR_cc {c['lr_cc']:.2f} · p {c['p_cc']:.4f}",
            v,
            sst,
        )
        v, sst = verdict(d["reject_h0"], "unpredictable", "predictable")
        rows += T.test_row(
            f"Dynamic Quantile · {d['lags']} lags",
            f"DQ {d['dq_stat']:.2f} · p {d['p_value']:.4f}",
            v,
            sst,
        )
        v, sst = verdict(e["reject_h0"], "ES adequate", "ES understated")
        rows += T.test_row(
            "Acerbi-Székely Z₂ · ES adequacy",
            f"Z₂ {e['z2_stat']:+.3f} · p {e['p_value']:.4f}",
            v,
            sst,
        )
        st.markdown(rows, unsafe_allow_html=True)

    with (
        right,
        T.card(
            "Basel traffic light",
            "Exceptions in the trailing 250 days (BCBS 1996).",
        ),
    ):
        zone = basel["zone"]
        color = T.STATUS[T.ZONE_STATUS[zone]]
        st.markdown(
            f'<div style="background:{color}14;border:1px solid {color}44;'
            f'border-radius:10px;padding:1rem;text-align:center;margin:.6rem 0">'
            f'<div style="font-size:1.5rem;font-weight:650;color:{color}">'
            f"{zone.upper()}</div>"
            f'<div style="font-size:.8rem;color:{T.TEXT_SECONDARY};margin-top:.15rem">'
            f"{basel['n_exceptions']} exceptions in 250 days</div></div>"
            + T.test_row(
                "Capital multiplier add-on",
                f"+{basel['multiplier_addon']:.2f}",
                "no add-on" if basel["multiplier_addon"] == 0 else "add-on",
                "good" if basel["multiplier_addon"] == 0 else "warning",
            )
            + T.note(
                "Green 0-4 · Yellow 5-9 (add-on 0.40-0.85) · Red 10+ "
                "(add-on 1.00). Zones are cumulative binomial thresholds "
                "under a correct 99% model."
            ),
            unsafe_allow_html=True,
        )


def page_comparison(var_results: pd.DataFrame, summary: dict) -> None:
    """Six engines head-to-head — the SR 11-7 benchmarking view."""
    st.markdown(
        T.header(
            "Engine Comparison",
            "SR 11-7 §5 · Benchmarking",
            "The same portfolio, the same days, six independent models. A "
            "model is only as credible as the challengers it is measured "
            "against — this is the view that shows which assumptions actually "
            "pay for themselves.",
        ),
        unsafe_allow_html=True,
    )

    rows = []
    for method, s in summary.items():
        k = s["kupiec"]
        rows.append(
            {
                "method": method,
                "Engine": T.METHOD_LABELS.get(method, method),
                "Days": s["n_forecast_days"],
                "Exceptions": k["n_exceptions"],
                "Expected": round(k["expected_exceptions"], 1),
                "Rate": k["exception_rate"] * 100,
                "Kupiec p": k["p_value"],
                "CC p": s["christoffersen"]["p_cc"],
                "DQ p": s["dynamic_quantile"]["p_value"],
                "ES Z₂": s["es_acerbi_szekely_z2"]["z2_stat"],
                "Basel 250d": s["basel_traffic_light_last_250d"]["zone"].upper(),
            }
        )
    board = pd.DataFrame(rows)

    # Exception count against the 1% expectation. One measure, one axis; the
    # reference line carries the target rather than a second scale.
    order = board.sort_values("Exceptions")
    fig = go.Figure()
    for _, r in order.iterrows():
        fig.add_trace(
            go.Bar(
                x=[r["Exceptions"]],
                y=[T.METHOD_SHORT[r["method"]]],
                orientation="h",
                name=r["Engine"],
                marker={"color": T.METHOD_COLORS[r["method"]], "cornerradius": 4},
                width=0.6,
                showlegend=False,
                hovertemplate=(
                    f"{r['Engine']}<br>%{{x}} exceptions vs "
                    f"{r['Expected']} expected<extra></extra>"
                ),
            )
        )
    # GARCH needs 750 days of history, so its sample — and its expected count
    # — is shorter than the others'. The line marks the value the five
    # full-sample engines share; each bar's own expectation is in its hover.
    expected = float(board["Expected"].mode().iloc[0])
    fig.add_vline(
        x=expected,
        line={"color": T.TEXT_MUTED, "width": 1},
        annotation_text=f"expected {expected:.0f} at 1%",
        annotation_position="bottom right",
        annotation_font={"size": 11, "color": T.TEXT_MUTED},
    )
    T.style_fig(fig, height=320, legend=False, hovermode="closest")
    fig.update_layout(bargap=0.34)
    n_short = int((board["Expected"] != expected).sum())
    T.chart_card(
        "Exceptions over the full backtest",
        "Closer to the reference line is better calibrated. Fat tails "
        "(Student-t) and conditional volatility (GARCH) each cut the excess; "
        "the constant-volatility normal models miss worst."
        + (
            f" GARCH is measured over a shorter sample ({n_short} engine), so "
            "its own expectation is lower — hover any bar for its exact "
            "figure."
            if n_short
            else ""
        ),
        fig,
    )

    years = st.slider("Window (years shown)", 1, 20, 5)
    cutoff = var_results["date"].max() - pd.DateOffset(years=years)
    recent = var_results[var_results["date"] >= cutoff]

    fig2 = go.Figure()
    for method in T.METHOD_LABELS:
        sub = recent[recent["method"] == method]
        if sub.empty:
            continue
        fig2.add_trace(
            go.Scatter(
                x=sub["date"],
                y=sub["var"],
                mode="lines",
                name=T.METHOD_SHORT[method],
                line={"color": T.METHOD_COLORS[method], "width": 1.8},
                hovertemplate="%{y:.2%}<extra>" + T.METHOD_SHORT[method] + "</extra>",
            )
        )
    T.style_fig(fig2, height=420, y_tickformat=".0%")
    T.chart_card(
        "One-day 99% VaR by engine",
        "GARCH reacts within a day of a shock then decays; Historical "
        "Simulation steps up only after the loss enters its 250-day window "
        "and stays elevated for a year — the ghost effect.",
        fig2,
    )

    with T.card(
        "Backtest scoreboard",
        "The table view of every statistic on this page. p &lt; 0.05 rejects; "
        "Z₂ &gt; 0 means realised tail losses exceeded the promised Expected "
        "Shortfall.",
    ):
        st.dataframe(
            board.drop(columns=["method"]),
            use_container_width=True,
            hide_index=True,
            column_config={
                "Rate": st.column_config.NumberColumn("Rate %", format="%.2f"),
                "Kupiec p": st.column_config.NumberColumn(format="%.4f"),
                "CC p": st.column_config.NumberColumn(format="%.4f"),
                "DQ p": st.column_config.NumberColumn(format="%.4f"),
                "ES Z₂": st.column_config.NumberColumn(format="%.3f"),
            },
        )
        st.markdown(
            T.note(
                "Every engine fails Dynamic Quantile over the full 2006-2026 "
                "sample: no single-regime model kept exceptions unpredictable "
                "across two crises. Reported, not tuned away — see "
                "docs/limitations.md."
            ),
            unsafe_allow_html=True,
        )


def _replay_tab(portfolio: pd.DataFrame, stress: dict) -> None:
    """Crisis-window replays against today's portfolio construction."""
    hist = stress["historical"]
    st.markdown(
        T.kpi_row(
            [
                {
                    "label": h["label"].split(" / ")[0],
                    "value": f"−{h['cumulative_loss_frac']:.1%}",
                    "note": f"max drawdown {h['max_drawdown_frac']:.1%}",
                    "status": (
                        "critical" if h["cumulative_loss_frac"] > 0.2 else "warning"
                    ),
                }
                for h in hist
            ]
        ),
        unsafe_allow_html=True,
    )

    names = {h["label"]: h for h in hist}
    chosen = st.selectbox("Crisis window", list(names))
    h = names[chosen]
    window = portfolio.set_index("date")["portfolio_return"].loc[h["start"] : h["end"]]
    wealth = (1.0 + window).cumprod() - 1.0

    fig = go.Figure(
        go.Scatter(
            x=wealth.index,
            y=wealth.values,
            mode="lines",
            line={"color": T.LOSS, "width": 2},
            fill="tozeroy",
            fillcolor="rgba(208,59,59,0.10)",
            name="Cumulative return",
            hovertemplate="%{x|%d %b %Y}<br>%{y:.1%}<extra></extra>",
        )
    )
    trough = wealth.idxmin()
    fig.add_trace(
        go.Scatter(
            x=[trough],
            y=[wealth.min()],
            mode="markers+text",
            marker={
                "color": T.LOSS,
                "size": 9,
                "line": {"color": T.SURFACE, "width": 2},
            },
            text=[f"  trough {wealth.min():.1%}"],
            textposition="middle right",
            textfont={"size": 11, "color": T.TEXT_SECONDARY},
            hoverinfo="skip",
            showlegend=False,
        )
    )
    T.style_fig(fig, height=340, y_tickformat=".0%", legend=False)
    T.chart_card(
        f"{h['label']} — {h['start']} to {h['end']}",
        f"{h['n_days']} trading days · worst single day "
        f"−{h['worst_day_loss_frac']:.2%} on {h['worst_day']}",
        fig,
    )

    if h["factors_missing"]:
        st.markdown(
            T.note(
                "Factors without data over this window, excluded from the "
                "attribution rather than silently zeroed: "
                + ", ".join(h["factors_missing"])
            ),
            unsafe_allow_html=True,
        )
    st.markdown(
        T.note(
            "The universe is <em>today's</em> S&P 500 membership, so firms "
            "that failed are absent: these replays are milder than the crises "
            "were (survivorship bias, limitations.md #21)."
        ),
        unsafe_allow_html=True,
    )


def _scenario_tab(stress: dict) -> None:
    """Hypothetical scenario ranking, waterfall and factor heatmap."""
    hypo = stress["hypothetical"]
    ordered = sorted(hypo, key=lambda s: -s["total_loss_frac"])

    fig = go.Figure(
        go.Bar(
            x=[s["total_loss_frac"] for s in ordered],
            y=[s["label"] for s in ordered],
            orientation="h",
            marker={
                "color": [
                    T.LOSS if s["total_loss_frac"] > 0 else T.GAIN for s in ordered
                ],
                "cornerradius": 4,
            },
            width=0.62,
            hovertemplate="%{y}<br>%{x:.2%}<extra></extra>",
        )
    )
    fig.add_vline(x=0, line={"color": T.BASELINE, "width": 1})
    T.style_fig(fig, height=360, x_tickformat=".0%", legend=False, hovermode="closest")
    fig.update_yaxes(autorange="reversed")
    T.chart_card(
        "Scenario P&L",
        "Positive is a loss. Instantaneous shocks priced through the "
        "estimated factor betas.",
        fig,
    )

    names = {s["label"]: s for s in hypo}
    chosen = st.selectbox("Attribution for scenario", list(names))
    s = names[chosen]
    contrib = {k: v for k, v in s["contributions"].items() if v != 0.0}

    col_a, col_b = st.columns([3, 2])
    with (
        col_a,
        T.card(
            "Factor attribution",
            f"{s['label']} — components sum to the "
            f"{s['total_loss_frac']:.2%} total",
        ),
    ):
        if contrib:
            wf = go.Figure(
                go.Waterfall(
                    orientation="v",
                    measure=["relative"] * len(contrib) + ["total"],
                    x=[T.COMPONENT_LABELS.get(k, k) for k in contrib] + ["Total"],
                    y=list(contrib.values()) + [0],
                    increasing={"marker": {"color": T.LOSS}},
                    decreasing={"marker": {"color": T.GAIN}},
                    totals={"marker": {"color": T.SERIES[0]}},
                    connector={"line": {"color": T.BASELINE, "width": 1}},
                    hovertemplate="%{x}<br>%{y:.2%}<extra></extra>",
                )
            )
            T.style_fig(
                wf, height=300, y_tickformat=".0%", legend=False, hovermode="closest"
            )
            st.plotly_chart(wf, use_container_width=True, config=T.PLOTLY_CONFIG)
        else:
            st.markdown(T.note("No non-zero components."), unsafe_allow_html=True)

    with col_b, T.card("Components"):
        st.markdown(
            "".join(
                T.test_row(
                    T.COMPONENT_LABELS.get(k, k),
                    f"{v:+.2%}",
                    "loss" if v > 0 else ("gain" if v < 0 else "flat"),
                    "critical" if v > 0 else ("good" if v < 0 else "warning"),
                )
                for k, v in s["contributions"].items()
            ),
            unsafe_allow_html=True,
        )
        if s["suppressed"]:
            st.markdown(
                T.note(
                    "<strong>Double-count guard.</strong> "
                    + ", ".join(T.COMPONENT_LABELS.get(k, k) for k in s["suppressed"])
                    + " contribute zero here. The betas are <em>marginal</em> "
                    "— they already encode the equity move that accompanies a "
                    "macro shock — so adding them on top of an explicit equity "
                    "shock would count the same loss twice."
                ),
                unsafe_allow_html=True,
            )

    heat = pd.DataFrame(
        {
            s["label"]: {
                T.COMPONENT_LABELS.get(k, k): v for k, v in s["contributions"].items()
            }
            for s in hypo
        }
    )
    hm = go.Figure(
        go.Heatmap(
            z=heat.to_numpy(),
            x=list(heat.columns),
            y=list(heat.index),
            colorscale=[[0, T.SERIES[0]], [0.5, "#2a3040"], [1, T.LOSS]],
            zmid=0,
            xgap=2,
            ygap=2,
            colorbar={
                "tickformat": ".0%",
                "outlinewidth": 0,
                "tickfont": {"size": 11, "color": T.TEXT_MUTED},
                "thickness": 12,
            },
            hovertemplate="%{x}<br>%{y}: %{z:.2%}<extra></extra>",
        )
    )
    T.style_fig(hm, height=280, legend=False, hovermode="closest")
    hm.update_yaxes(showgrid=False)
    T.chart_card(
        "Factor contribution heatmap",
        "Loss share by scenario and factor. Blue is a gain, red a loss, "
        "neutral grey no contribution.",
        hm,
    )


def _reverse_tab(stress: dict) -> None:
    """Reverse stress result and the sensitivities behind every scenario."""
    rev = stress["reverse"]
    sens = stress["sensitivities"]
    units = {"equity": "", "rates_level_bp": "bp", "credit_ig_bp": "bp"}

    st.markdown(
        T.kpi_row(
            [
                {
                    "label": T.COMPONENT_LABELS.get(f.replace("_bp", ""), f),
                    "value": (
                        f"{v:.2%}" if f == "equity" else f"{v:+.1f} {units.get(f, '')}"
                    ),
                    "note": "shock required",
                }
                for f, v in rev["shocks"].items()
            ]
            + [
                {
                    "label": "Achieved loss",
                    "value": f"{rev['achieved_loss_frac']:.2%}",
                    "note": f"target {rev['target_loss_frac']:.0%}",
                    "status": "critical",
                },
            ]
        ),
        unsafe_allow_html=True,
    )
    st.markdown(
        T.note(
            "BCBS (2018) Principle 6: instead of pricing scenarios we already "
            "imagined, search for the <em>smallest</em> combined shock that "
            f"produces a {rev['target_loss_frac']:.0%} loss — minimising "
            "scaled shock magnitude subject to the loss constraint (SLSQP), "
            "bounded so equity can only fall and spreads only widen."
        ),
        unsafe_allow_html=True,
    )
    if rev["binding_bounds"]:
        st.markdown(
            T.note(
                "Bounds binding at the optimum: " + ", ".join(rev["binding_bounds"])
            ),
            unsafe_allow_html=True,
        )

    with T.card(
        "Estimated factor sensitivities",
        f"OLS with HC1 robust standard errors · {sens['n_obs']} joint days "
        f"({sens['sample_start']} to {sens['sample_end']}) · "
        f"R² {sens['r_squared']:.3f}",
    ):
        st.dataframe(
            pd.DataFrame(
                {
                    "Factor": [
                        T.COMPONENT_LABELS.get(k.replace("_bp", ""), k)
                        for k in sens["betas"]
                    ],
                    "Beta (return per bp)": [
                        f"{v:+.3e}" for v in sens["betas"].values()
                    ],
                    "Std error (HC1)": [f"{v:.1e}" for v in sens["stderrs"].values()],
                    "t-stat": [
                        f"{b / se:+.2f}"
                        for b, se in zip(
                            sens["betas"].values(),
                            sens["stderrs"].values(),
                            strict=True,
                        )
                    ],
                }
            ),
            use_container_width=True,
            hide_index=True,
        )
        st.markdown("".join(T.note(n) for n in sens["notes"]), unsafe_allow_html=True)


def page_stress(portfolio: pd.DataFrame, stress: dict) -> None:
    """Stress testing: replay, hypothetical scenarios, reverse search."""
    st.markdown(
        T.header(
            "Stress Testing",
            "CCAR-style scenarios · BCBS 2018 principles",
            "What VaR cannot tell you: what happens in the tail it never "
            "sampled. Crisis windows are replayed on the realised portfolio "
            "path; hypothetical shocks are priced through estimated factor "
            "betas; reverse stress asks the question backwards.",
        ),
        unsafe_allow_html=True,
    )
    tab_hist, tab_hypo, tab_rev = st.tabs(
        ["Historical replay", "Hypothetical scenarios", "Reverse stress"]
    )
    with tab_hist:
        _replay_tab(portfolio, stress)
    with tab_hypo:
        _scenario_tab(stress)
    with tab_rev:
        _reverse_tab(stress)


def page_narrative() -> None:
    """Roadmap page for the Month 4 milestone."""
    st.markdown(
        T.header(
            "Narrative Risk",
            "Month 4 · In progress",
            "FinBERT sentiment over 10-K Item 1A Risk Factors, aggregated to a "
            "monthly firm-level Narrative Risk Score, then tested for Granger "
            "causality against VaR exceptions.",
        ),
        unsafe_allow_html=True,
    )
    st.markdown(
        T.note(
            "Not yet shipped. The completed milestones — ingestion and PySpark "
            "ETL, six VaR/ES engines, the backtesting suite, and stress "
            "testing — are on the other pages."
        ),
        unsafe_allow_html=True,
    )


# --------------------------------------------------------------------------
# Router
# --------------------------------------------------------------------------
def sidebar() -> str:
    """Sidebar branding, navigation and provenance."""
    with st.sidebar:
        st.markdown(
            f'<div style="font-size:1.25rem;font-weight:650;'
            f'color:{T.TEXT_PRIMARY};letter-spacing:-0.01em">RiskSense</div>'
            f'<div style="font-size:.78rem;color:{T.TEXT_MUTED};line-height:1.5;'
            'margin:.25rem 0 1.1rem">Market risk engine — VaR &amp; Expected '
            "Shortfall, statistical backtesting, stress testing.</div>",
            unsafe_allow_html=True,
        )
        page = st.radio(
            "Navigation",
            [
                "Portfolio Overview",
                "VaR & Backtesting",
                "Engine Comparison",
                "Stress Testing",
                "Narrative Risk",
            ],
            label_visibility="collapsed",
        )
        st.markdown(
            f'<div style="border-top:1px solid {T.BORDER};margin-top:1.4rem;'
            'padding-top:.9rem">'
            '<div style="font-size:.68rem;font-weight:600;letter-spacing:.08em;'
            f'text-transform:uppercase;color:{T.TEXT_MUTED}">Frameworks</div>'
            f'<div style="font-size:.78rem;color:{T.TEXT_SECONDARY};'
            'line-height:1.7;margin-top:.4rem">FRTB · ES 97.5%<br>'
            "Basel III · VaR 99%<br>SR 11-7 · validation<br>"
            "CCAR · stress scenarios</div></div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div style="border-top:1px solid {T.BORDER};margin-top:1.1rem;'
            f"padding-top:.9rem;font-size:.72rem;color:{T.TEXT_MUTED};"
            'line-height:1.6">Personal research project on real S&amp;P 500 '
            "data. Not investment advice and not a production risk system; "
            "the limitations are documented in the repository.</div>",
            unsafe_allow_html=True,
        )
    return page


def main() -> None:
    """Load outputs and dispatch to the selected page."""
    page = sidebar()
    try:
        portfolio, var_results, summary = load_outputs()
    except FileNotFoundError:
        st.markdown(
            T.header(
                "Pipeline outputs not found",
                "Setup required",
                "Generate the data before launching the dashboard.",
            ),
            unsafe_allow_html=True,
        )
        st.code(
            "python data/ingest_prices.py\n"
            "python data/ingest_macro.py\n"
            "python -m risksense.pipelines.returns_pipeline\n"
            "python -m risksense.cli\n"
            "python -m risksense.stress",
            language="bash",
        )
        return

    if page == "Portfolio Overview":
        page_overview(portfolio, var_results, summary)
    elif page == "VaR & Backtesting":
        page_var_backtest(var_results, summary)
    elif page == "Engine Comparison":
        page_comparison(var_results, summary)
    elif page == "Stress Testing":
        stress = load_stress()
        if stress is None:
            st.markdown(
                T.header(
                    "Stress results not found",
                    "Setup required",
                    "Run the stress suite to populate this page.",
                ),
                unsafe_allow_html=True,
            )
            st.code(
                "python data/ingest_macro.py\npython -m risksense.stress",
                language="bash",
            )
        else:
            page_stress(portfolio, stress)
    else:
        page_narrative()


if __name__ == "__main__":
    main()
