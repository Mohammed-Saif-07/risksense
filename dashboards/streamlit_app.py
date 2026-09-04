"""RiskSense Streamlit dashboard.

Week 1 scope: Portfolio Overview + VaR/Backtesting page for the Historical
Simulation engine. Weeks 2-4 add the engine comparison, stress testing and
narrative risk pages (placeholders shown until then).

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


def page_overview(portfolio: pd.DataFrame, var_results: pd.DataFrame, summary: dict) -> None:
    """Portfolio composition, cumulative P&L and headline risk metrics."""
    st.title("Portfolio Overview")
    st.caption(
        "Equal-weight S&P 500 constituent portfolio, daily-rebalanced. "
        "All metrics are one-day, log-return based."
    )

    latest = var_results.iloc[-1]
    basel = summary["basel_traffic_light_last_250d"]
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
    """VaR forecasts vs realised returns with exception markers + test results."""
    st.title("VaR & Backtesting — Historical Simulation")

    df = var_results
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
    fig.update_layout(height=520, yaxis_tickformat=".1%", title="Daily VaR vs realised P&L")
    st.plotly_chart(fig, use_container_width=True)

    kupiec = summary["kupiec_full_sample"]
    basel = summary["basel_traffic_light_last_250d"]
    left, right = st.columns(2)
    with left:
        st.subheader("Kupiec POF (full sample)")
        st.write(
            f"Exceptions: **{kupiec['n_exceptions']}** vs expected "
            f"**{kupiec['expected_exceptions']:.1f}** over {kupiec['n_obs']:,} days"
        )
        st.write(
            f"LR = {kupiec['lr_stat']:.2f}, p-value = {kupiec['p_value']:.4f} → "
            + (
                "**reject** correct coverage"
                if kupiec["reject_h0"]
                else "**cannot reject** correct coverage"
            )
        )
    with right:
        st.subheader("Basel traffic light (last 250 days)")
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


def main() -> None:
    """Sidebar router."""
    page = st.sidebar.radio(
        "Pages",
        [
            "Portfolio Overview",
            "VaR & Backtesting",
            "VaR Method Comparison (Week 2)",
            "Stress Testing (Week 3)",
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
    else:
        st.title(page)
        st.info("This page ships in a later weekly milestone — see README roadmap.")


if __name__ == "__main__":
    main()
