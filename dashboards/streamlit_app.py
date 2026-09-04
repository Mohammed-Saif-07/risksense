"""RiskSense Streamlit dashboard.

Pages: Portfolio Overview, VaR & Backtesting (per method), VaR Method
Comparison (all engines head-to-head), Stress Testing (historical replay,
hypothetical waterfalls, reverse stress), plus the Week 4 placeholder.

Run locally:  streamlit run dashboards/streamlit_app.py
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

PROCESSED = REPO_ROOT / "data" / "processed"

ZONE_COLORS = {"green": "#2e7d32", "yellow": "#f9a825", "red": "#c62828"}
METHOD_COLORS = {
    "historical": "#1565c0",
    "parametric_normal": "#2e7d32",
    "parametric_t": "#00838f",
    "monte_carlo_normal": "#ef6c00",
    "monte_carlo_t": "#6a1b9a",
    "monte_carlo_garch_t": "#c62828",
}
METHOD_LABELS = {
    "historical": "Historical Simulation",
    "parametric_normal": "Parametric Normal (Ledoit-Wolf)",
    "parametric_t": "Parametric Student-t (Ledoit-Wolf)",
    "monte_carlo_normal": "Monte Carlo Normal",
    "monte_carlo_t": "Monte Carlo Student-t",
    "monte_carlo_garch_t": "Monte Carlo GARCH(1,1)-t",
}
COMPONENT_LABELS = {
    "equity": "Equity",
    "equity_residual": "Equity (residual)",
    "rates_level": "Rates level",
    "rates_slope": "Rates slope",
    "credit_ig": "Credit IG",
}

st.set_page_config(page_title="RiskSense", layout="wide", page_icon="📉")


@st.cache_data(show_spinner=False)
def load_outputs() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Load pipeline outputs; raise a friendly error if the pipeline hasn't run."""
    portfolio = pd.read_parquet(PROCESSED / "portfolio_returns.parquet").sort_values("date")
    var_results = pd.read_parquet(PROCESSED / "var_results.parquet")
    with (PROCESSED / "backtest_summary.json").open() as fh:
        summary = json.load(fh)
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


def zone_banner(basel: dict) -> None:
    """Render the Basel traffic-light banner."""
    color = ZONE_COLORS[basel["zone"]]
    st.markdown(
        f"<div style='background:{color};color:white;padding:12px;"
        f"border-radius:8px;font-size:20px;text-align:center'>"
        f"{basel['zone'].upper()} — {basel['n_exceptions']} exceptions"
        "</div>",
        unsafe_allow_html=True,
    )
    st.caption(
        f"Capital multiplier add-on: +{basel['multiplier_addon']:.2f} "
        "(BCBS 1996 supervisory framework)"
    )


def page_overview(portfolio: pd.DataFrame, var_results: pd.DataFrame, summary: dict) -> None:
    """Portfolio composition, cumulative P&L and headline risk metrics."""
    st.title("Portfolio Overview")
    st.caption(
        "Equal-weight S&P 500 constituent portfolio, daily-rebalanced. "
        "All metrics are one-day, log-return based. Headline figures use "
        "Historical Simulation; see the comparison page for all engines."
    )

    hist = var_results[var_results["method"] == "historical"]
    latest = hist.iloc[-1]
    basel = summary["historical"]["basel_traffic_light_last_250d"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("VaR 99% (1d)", f"{latest['var']:.2%}")
    c2.metric("ES 97.5% (1d, FRTB)", f"{latest['es']:.2%}")
    c3.metric("Constituents", f"{int(portfolio['n_constituents'].iloc[-1])}")
    c4.metric(
        "Basel zone (250d)",
        basel["zone"].upper(),
        f"{basel['n_exceptions']} exceptions",
        delta_color="off",
    )

    cum = (1.0 + portfolio.set_index("date")["portfolio_return"]).cumprod() - 1.0
    fig = go.Figure(
        go.Scatter(x=cum.index, y=cum.values, mode="lines", name="Cumulative return")
    )
    fig.update_layout(
        height=420, yaxis_tickformat=".0%", title="Cumulative portfolio return"
    )
    st.plotly_chart(fig, use_container_width=True)


def page_var_backtest(var_results: pd.DataFrame, summary: dict) -> None:
    """Per-method VaR vs realised returns + the full statistical test suite."""
    st.title("VaR & Backtesting")

    methods = [m for m in METHOD_LABELS if m in set(var_results["method"])]
    method = st.selectbox(
        "Engine", methods, format_func=lambda m: METHOD_LABELS.get(m, m)
    )
    df = var_results[var_results["method"] == method]
    s = summary[method]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df["date"], y=df["realized_return"], mode="lines",
            name="Realised return", line={"width": 1, "color": "#90a4ae"},
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df["date"], y=-df["var"], mode="lines",
            name="VaR 99% bound", line={"color": "#1565c0"},
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df["date"], y=-df["es"], mode="lines",
            name="ES 97.5% (FRTB)", line={"color": "#6a1b9a", "dash": "dot"},
        )
    )
    exc = df[df["exception"]]
    fig.add_trace(
        go.Scatter(
            x=exc["date"], y=exc["realized_return"], mode="markers",
            name=f"Exceptions ({len(exc)})",
            marker={"color": "#c62828", "size": 7, "symbol": "x"},
        )
    )
    fig.update_layout(
        height=480, yaxis_tickformat=".1%",
        title=f"Daily VaR vs realised P&L — {METHOD_LABELS.get(method, method)}",
    )
    st.plotly_chart(fig, use_container_width=True)

    k, c, d, e = s["kupiec"], s["christoffersen"], s["dynamic_quantile"], s["es_acerbi_szekely_z2"]
    col1, col2, col3 = st.columns(3)
    with col1:
        st.subheader("Kupiec POF")
        st.write(
            f"Exceptions **{k['n_exceptions']}** vs expected "
            f"**{k['expected_exceptions']:.1f}** ({k['n_obs']:,} days)"
        )
        st.write(f"LR = {k['lr_stat']:.2f}, p = {k['p_value']:.4f}")
        st.write("❌ reject coverage" if k["reject_h0"] else "✅ coverage not rejected")
    with col2:
        st.subheader("Christoffersen")
        st.write(
            f"π01 = {c['pi01']:.3f}, π11 = {c['pi11']:.3f} "
            f"(clustering if π11 ≫ π01)"
        )
        st.write(f"LR_ind p = {c['p_ind']:.4f} · LR_cc p = {c['p_cc']:.4f}")
        st.write(
            "❌ exceptions cluster" if c["reject_independence"] else "✅ independence not rejected"
        )
    with col3:
        st.subheader("Dynamic Quantile / ES")
        st.write(f"DQ({d['lags']}) = {d['dq_stat']:.2f}, p = {d['p_value']:.4f}")
        st.write(f"Acerbi-Székely Z₂ = {e['z2_stat']:.3f}, p = {e['p_value']:.4f}")
        st.write(
            "❌ hits predictable" if d["reject_h0"] else "✅ hits unpredictable"
        )

    st.subheader("Basel traffic light (last 250 days)")
    zone_banner(s["basel_traffic_light_last_250d"])


def page_comparison(var_results: pd.DataFrame, summary: dict) -> None:
    """All engines head-to-head: VaR paths + test scoreboard (SR 11-7 §5)."""
    st.title("VaR Method Comparison")
    st.caption(
        "SR 11-7 benchmarking: the same portfolio through six independent "
        "engines. GARCH starts later (needs 750 days of history)."
    )

    years = st.slider("Show last N years", 1, 20, 5)
    cutoff = var_results["date"].max() - pd.DateOffset(years=years)
    recent = var_results[var_results["date"] >= cutoff]

    fig = go.Figure()
    for method, sub in recent.groupby("method"):
        fig.add_trace(
            go.Scatter(
                x=sub["date"], y=sub["var"], mode="lines",
                name=METHOD_LABELS.get(str(method), str(method)),
                line={"color": METHOD_COLORS.get(str(method), "#607d8b"), "width": 1.4},
            )
        )
    fig.update_layout(
        height=480, yaxis_tickformat=".1%", title="One-day 99% VaR by engine"
    )
    st.plotly_chart(fig, use_container_width=True)

    rows = []
    for method, s in summary.items():
        k, c, d, e = s["kupiec"], s["christoffersen"], s["dynamic_quantile"], s["es_acerbi_szekely_z2"]
        b = s["basel_traffic_light_last_250d"]
        rows.append(
            {
                "Engine": METHOD_LABELS.get(method, method),
                "Days": s["n_forecast_days"],
                "Exceptions": k["n_exceptions"],
                "Expected": round(k["expected_exceptions"], 1),
                "Kupiec p": round(k["p_value"], 4),
                "CC p": round(c["p_cc"], 4),
                "DQ p": round(d["p_value"], 4),
                "ES Z2": round(e["z2_stat"], 3),
                "Basel (250d)": b["zone"].upper(),
            }
        )
    st.subheader("Backtest scoreboard")
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    st.caption(
        "p < 0.05 rejects the null (correct coverage / independence / "
        "unpredictable hits). Z₂ > 0 with small p: ES understated."
    )


def page_stress(portfolio: pd.DataFrame, stress: dict) -> None:
    """Historical replay, hypothetical waterfalls, factor heatmap, reverse stress."""
    st.title("Stress Testing")
    sens = stress["sensitivities"]
    st.caption(
        "CCAR-style scenarios translated through OLS factor betas "
        f"(HC1 robust, {sens['n_obs']} joint days, "
        f"{sens['sample_start']} → {sens['sample_end']}, R²={sens['r_squared']:.3f}). "
        "Historical replays use the realised portfolio path, not the betas."
    )

    tab_hist, tab_hypo, tab_rev = st.tabs(
        ["Historical replay", "Hypothetical scenarios", "Reverse stress"]
    )

    with tab_hist:
        hist = stress["historical"]
        cols = st.columns(len(hist))
        for col, h in zip(cols, hist, strict=True):
            col.metric(
                h["label"].split(" / ")[0],
                f"{h['cumulative_loss_frac']:.1%}",
                f"maxDD {h['max_drawdown_frac']:.1%}",
                delta_color="off",
            )

        names = {h["label"]: h for h in hist}
        chosen = st.selectbox("Crisis window", list(names))
        h = names[chosen]
        cum = (
            portfolio.set_index("date")["portfolio_return"].loc[h["start"]:h["end"]]
        )
        wealth = (1.0 + cum).cumprod() - 1.0
        fig = go.Figure(
            go.Scatter(x=wealth.index, y=wealth.values, mode="lines",
                       line={"color": "#c62828"}, name="Cumulative return")
        )
        fig.update_layout(
            height=340, yaxis_tickformat=".0%",
            title=f"{h['label']}: {h['start']} → {h['end']} ({h['n_days']} trading days)",
        )
        st.plotly_chart(fig, use_container_width=True)
        c1, c2, c3 = st.columns(3)
        c1.metric("Cumulative loss", f"{h['cumulative_loss_frac']:.2%}")
        c2.metric("Worst day", f"{h['worst_day_loss_frac']:.2%}", h["worst_day"],
                  delta_color="off")
        c3.metric("Max drawdown", f"{h['max_drawdown_frac']:.2%}")
        if h["factors_missing"]:
            st.warning(
                "Factors without data over this window (excluded from the "
                "attribution, not zeroed silently): "
                + ", ".join(h["factors_missing"])
            )
        st.info(
            "Universe is *today's* S&P 500 membership — survivorship bias makes "
            "these replays milder than the crises were (limitations.md #1)."
        )

    with tab_hypo:
        hypo = stress["hypothetical"]
        rows = [
            {
                "Scenario": s["label"],
                "Loss": s["total_loss_frac"],
                "Loss USD": s["total_loss_usd"],
            }
            for s in sorted(hypo, key=lambda s: -s["total_loss_frac"])
        ]
        df = pd.DataFrame(rows)
        fig = go.Figure(
            go.Bar(
                x=df["Loss"], y=df["Scenario"], orientation="h",
                marker_color=["#c62828" if v > 0 else "#2e7d32" for v in df["Loss"]],
            )
        )
        fig.update_layout(
            height=380, xaxis_tickformat=".0%",
            title="Scenario P&L (positive = loss)",
            yaxis={"autorange": "reversed"},
        )
        st.plotly_chart(fig, use_container_width=True)

        names = {s["label"]: s for s in hypo}
        chosen = st.selectbox("Waterfall for scenario", list(names))
        s = names[chosen]
        contrib = {k: v for k, v in s["contributions"].items() if v != 0.0}
        if contrib:
            labels = [COMPONENT_LABELS.get(k, k) for k in contrib] + ["Total"]
            wf = go.Figure(
                go.Waterfall(
                    orientation="v",
                    measure=["relative"] * len(contrib) + ["total"],
                    x=labels,
                    y=list(contrib.values()) + [0],
                    increasing={"marker": {"color": "#c62828"}},
                    decreasing={"marker": {"color": "#2e7d32"}},
                    totals={"marker": {"color": "#1565c0"}},
                )
            )
            wf.update_layout(height=340, yaxis_tickformat=".1%",
                             title=f"Factor attribution — {s['label']}")
            st.plotly_chart(wf, use_container_width=True)
        if s["suppressed"]:
            st.warning(
                "Double-count guard: "
                + ", ".join(COMPONENT_LABELS.get(k, k) for k in s["suppressed"])
                + " contribute 0 here. The betas are *marginal* — they encode "
                "the equity move that accompanies a macro shock — so adding "
                "them on top of an explicit equity shock would count the same "
                "loss twice."
            )

        heat = pd.DataFrame(
            {
                s["label"]: {
                    COMPONENT_LABELS.get(k, k): v for k, v in s["contributions"].items()
                }
                for s in hypo
            }
        )
        hm = go.Figure(
            go.Heatmap(
                z=heat.to_numpy(), x=list(heat.columns), y=list(heat.index),
                colorscale="RdBu_r", zmid=0, colorbar={"tickformat": ".0%"},
            )
        )
        hm.update_layout(height=300, title="Factor contribution heatmap (loss share)")
        st.plotly_chart(hm, use_container_width=True)

    with tab_rev:
        rev = stress["reverse"]
        st.subheader(
            f"Smallest shock reaching a {rev['target_loss_frac']:.0%} loss"
        )
        st.caption(
            "BCBS (2018) Principle 6: minimise scaled shock magnitude subject "
            "to loss ≥ target (SLSQP, bounded to plausible moves)."
        )
        cols = st.columns(len(rev["shocks"]))
        units = {"equity": "", "rates_level_bp": "bp", "credit_ig_bp": "bp"}
        for col, (factor, value) in zip(cols, rev["shocks"].items(), strict=True):
            label = COMPONENT_LABELS.get(factor.replace("_bp", ""), factor)
            shown = f"{value:.2%}" if factor == "equity" else f"{value:+.1f} {units.get(factor, '')}"
            col.metric(label, shown)
        st.write(
            f"Achieved loss **{rev['achieved_loss_frac']:.2%}** · "
            f"scaled magnitude {rev['scaled_magnitude']:.2f} · "
            f"{rev['n_iterations']} iterations"
        )
        if rev["binding_bounds"]:
            st.warning("Bounds binding at the optimum: " + ", ".join(rev["binding_bounds"]))

        st.subheader("Estimated factor sensitivities")
        st.dataframe(
            pd.DataFrame(
                {
                    "Factor": list(sens["betas"]),
                    "Beta (return per bp)": [f"{v:+.3e}" for v in sens["betas"].values()],
                    "Std error (HC1)": [f"{v:.1e}" for v in sens["stderrs"].values()],
                }
            ),
            use_container_width=True, hide_index=True,
        )
        for note in sens["notes"]:
            st.caption(f"• {note}")


def main() -> None:
    """Sidebar router."""
    page = st.sidebar.radio(
        "Pages",
        [
            "Portfolio Overview",
            "VaR & Backtesting",
            "VaR Method Comparison",
            "Stress Testing",
            "Narrative Risk (Week 4)",
        ],
    )
    try:
        portfolio, var_results, summary = load_outputs()
    except FileNotFoundError:
        st.error(
            "Pipeline outputs not found. Run:\n\n"
            "```\npython data/ingest_prices.py\n"
            "python -m risksense.pipelines.returns_pipeline\n"
            "python -m risksense.cli\n```"
        )
        return

    if page == "Portfolio Overview":
        page_overview(portfolio, var_results, summary)
    elif page == "VaR & Backtesting":
        page_var_backtest(var_results, summary)
    elif page == "VaR Method Comparison":
        page_comparison(var_results, summary)
    elif page == "Stress Testing":
        stress = load_stress()
        if stress is None:
            st.error(
                "Stress results not found. Run:\n\n"
                "```\npython data/ingest_macro.py\n"
                "python -m risksense.stress\n```"
            )
        else:
            page_stress(portfolio, stress)
    else:
        st.title(page)
        st.info("This page ships in a later weekly milestone — see README roadmap.")


if __name__ == "__main__":
    main()
