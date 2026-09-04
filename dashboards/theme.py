"""Shared visual language for the RiskSense dashboard.

One place for the palette, the Plotly template and the HTML components, so
every page renders as one system rather than six separately-styled charts.

Palette provenance
------------------
Categorical slots are assigned in fixed order and validated as a set against
the dashboard surface (``#12151c``): lightness band, chroma floor, adjacent
colour-vision-deficiency separation (worst ΔE 8.4), normal-vision separation
(worst ΔE 19.3) and ≥3:1 contrast all pass. Colour follows the *entity* —
an engine keeps its hue no matter how many are on screen — so a reader who
learns "GARCH is green" is never misled by a filter.

Status colours are a reserved set (good / warning / critical) never reused
for a data series, and always shipped with a text label so meaning is never
carried by hue alone.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import plotly.graph_objects as go
import streamlit as st

# --- Surfaces and ink -------------------------------------------------------
PAGE_BG = "#0b0e14"
SURFACE = "#12151c"
SURFACE_RAISED = "#161a23"
TEXT_PRIMARY = "#f2f4f8"
TEXT_SECONDARY = "#a8b0c0"
TEXT_MUTED = "#79839a"
GRIDLINE = "#1e2330"
BASELINE = "#2a3040"
BORDER = "rgba(255,255,255,0.09)"

# --- Categorical series (fixed order, validated as a set) -------------------
SERIES = ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300"]

# --- Reserved status colours (never used for a data series) -----------------
STATUS = {
    "good": "#0ca30c",
    "warning": "#fab219",
    "serious": "#ec835a",
    "critical": "#d03b3b",
}
ZONE_STATUS = {"green": "good", "yellow": "warning", "red": "critical"}

# --- Semantic roles ---------------------------------------------------------
LOSS = STATUS["critical"]
GAIN = STATUS["good"]
REALIZED = "#4a5468"  # recessive: the P&L backdrop, not a headline series

FONT = 'system-ui, -apple-system, "Segoe UI", sans-serif'

#: Engine → categorical slot. Fixed mapping: identity, not rank.
METHOD_COLORS = {
    "historical": SERIES[0],
    "parametric_normal": SERIES[1],
    "parametric_t": SERIES[2],
    "monte_carlo_normal": SERIES[3],
    "monte_carlo_t": SERIES[4],
    "monte_carlo_garch_t": SERIES[5],
}
METHOD_LABELS = {
    "historical": "Historical Simulation",
    "parametric_normal": "Parametric Normal",
    "parametric_t": "Parametric Student-t",
    "monte_carlo_normal": "Monte Carlo Normal",
    "monte_carlo_t": "Monte Carlo Student-t",
    "monte_carlo_garch_t": "Monte Carlo GARCH(1,1)-t",
}
METHOD_SHORT = {
    "historical": "Historical",
    "parametric_normal": "Param Normal",
    "parametric_t": "Param t",
    "monte_carlo_normal": "MC Normal",
    "monte_carlo_t": "MC t",
    "monte_carlo_garch_t": "MC GARCH-t",
}
COMPONENT_LABELS = {
    "equity": "Equity",
    "equity_residual": "Equity (residual)",
    "rates_level": "Rates level",
    "rates_slope": "Rates slope",
    "credit_ig": "Credit IG",
}


def style_fig(
    fig: go.Figure,
    height: int = 380,
    y_tickformat: str | None = None,
    x_tickformat: str | None = None,
    legend: bool = True,
    hovermode: str = "x unified",
) -> go.Figure:
    """Apply the shared chart chrome: recessive grid, hairline axes, legend.

    Gridlines and axes are solid hairlines one shade off the surface (dashed
    grids read as thresholds); the legend sits above the plot so it never
    competes with the marks for width.
    """
    fig.update_layout(
        height=height,
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"family": FONT, "size": 13, "color": TEXT_SECONDARY},
        margin={"l": 8, "r": 8, "t": 44 if legend else 16, "b": 8},
        hovermode=hovermode,
        hoverlabel={
            "bgcolor": SURFACE_RAISED,
            "bordercolor": BORDER,
            "font": {"family": FONT, "size": 12, "color": TEXT_PRIMARY},
        },
        showlegend=legend,
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "left",
            "x": 0,
            "font": {"size": 12, "color": TEXT_SECONDARY},
            "bgcolor": "rgba(0,0,0,0)",
        },
    )
    fig.update_xaxes(
        showgrid=False,
        zeroline=False,
        linecolor=BASELINE,
        tickcolor=BASELINE,
        tickfont={"size": 11, "color": TEXT_MUTED},
        tickformat=x_tickformat,
    )
    fig.update_yaxes(
        showgrid=True,
        gridcolor=GRIDLINE,
        gridwidth=1,
        zeroline=False,
        linecolor="rgba(0,0,0,0)",
        tickfont={"size": 11, "color": TEXT_MUTED},
        tickformat=y_tickformat,
    )
    return fig


CSS = f"""
<style>
  .stApp {{ background: {PAGE_BG}; }}
  html, body, [class*="css"] {{ font-family: {FONT}; }}

  /* Tighten Streamlit's default chrome */
  /* Clear Streamlit's floating toolbar, which otherwise crops the eyebrow. */
  .block-container {{ padding-top: 4.75rem; padding-bottom: 4rem; max-width: 1400px; }}
  #MainMenu, footer {{ visibility: hidden; }}
  header[data-testid="stHeader"] {{ background: transparent; }}

  h1, h2, h3 {{ color: {TEXT_PRIMARY}; letter-spacing: -0.01em; }}
  h1 {{ font-size: 1.9rem !important; font-weight: 650 !important; }}
  h2 {{ font-size: 1.15rem !important; font-weight: 600 !important;
        margin-top: 1.6rem !important; }}
  h3 {{ font-size: 0.95rem !important; font-weight: 600 !important; }}

  /* Page header */
  .rs-head {{ border-bottom: 1px solid {BORDER}; padding-bottom: 1rem;
              margin-bottom: 1.4rem; }}
  .rs-eyebrow {{ font-size: 0.72rem; font-weight: 600; letter-spacing: 0.09em;
                 text-transform: uppercase; color: {TEXT_MUTED}; }}
  .rs-sub {{ color: {TEXT_SECONDARY}; font-size: 0.88rem; line-height: 1.55;
             margin-top: 0.35rem; max-width: 74ch; }}

  /* KPI cards */
  .rs-kpis {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
              gap: 0.75rem; margin: 0.4rem 0 1.2rem; }}
  .rs-kpi {{ background: {SURFACE}; border: 1px solid {BORDER}; border-radius: 10px;
             padding: 0.85rem 1rem; }}
  .rs-kpi-label {{ font-size: 0.72rem; font-weight: 600; letter-spacing: 0.05em;
                   text-transform: uppercase; color: {TEXT_MUTED}; }}
  .rs-kpi-value {{ font-size: 1.75rem; font-weight: 620; color: {TEXT_PRIMARY};
                   line-height: 1.25; margin-top: 0.2rem; }}
  .rs-kpi-note {{ font-size: 0.76rem; color: {TEXT_SECONDARY}; margin-top: 0.1rem; }}

  /* Cards: Streamlit's own bordered container, restyled. Raw <div> wrappers
     can't work here — Streamlit auto-closes them and leaves an empty box. */
  [data-testid="stVerticalBlockBorderWrapper"]:has(> div > [data-testid="stVerticalBlock"]) {{
      background: {SURFACE}; border: 1px solid {BORDER} !important;
      border-radius: 12px; padding: 0.35rem 0.35rem 0.1rem;
  }}
  .rs-card-title {{ font-size: 0.95rem; font-weight: 600; color: {TEXT_PRIMARY}; }}
  .rs-card-sub {{ font-size: 0.78rem; color: {TEXT_MUTED}; margin-top: 0.15rem;
                  line-height: 1.5; }}

  /* Verdict pills — colour is paired with a text label, never alone */
  .rs-pill {{ display: inline-flex; align-items: center; gap: 0.4rem;
              padding: 0.2rem 0.6rem; border-radius: 999px; font-size: 0.78rem;
              font-weight: 600; border: 1px solid; }}
  .rs-dot {{ width: 7px; height: 7px; border-radius: 50%; }}

  /* Test result rows */
  .rs-test {{ display: flex; justify-content: space-between; align-items: baseline;
              padding: 0.42rem 0; border-bottom: 1px solid {BORDER}; }}
  .rs-test:last-child {{ border-bottom: none; }}
  .rs-test-name {{ font-size: 0.83rem; color: {TEXT_SECONDARY}; }}
  .rs-test-val {{ font-size: 0.83rem; color: {TEXT_PRIMARY};
                  font-variant-numeric: tabular-nums; }}

  .rs-note {{ font-size: 0.78rem; color: {TEXT_MUTED}; line-height: 1.6;
              border-left: 2px solid {BASELINE}; padding-left: 0.7rem;
              margin: 0.6rem 0; }}

  /* Sidebar */
  section[data-testid="stSidebar"] {{ background: {SURFACE}; border-right: 1px solid {BORDER}; }}
  section[data-testid="stSidebar"] .block-container {{ padding-top: 1.6rem; }}

  /* Tabs */
  .stTabs [data-baseweb="tab-list"] {{ gap: 1.4rem; border-bottom: 1px solid {BORDER}; }}
  .stTabs [data-baseweb="tab"] {{ padding: 0.4rem 0; font-size: 0.88rem; }}

  /* Tables */
  [data-testid="stDataFrame"] {{ border: 1px solid {BORDER}; border-radius: 10px; }}
</style>
"""


def header(title: str, eyebrow: str, subtitle: str) -> str:
    """Page header: eyebrow label, title, one-line orientation."""
    return (
        f'<div class="rs-head"><div class="rs-eyebrow">{eyebrow}</div>'
        f"<h1>{title}</h1>"
        f'<div class="rs-sub">{subtitle}</div></div>'
    )


def kpi_row(items: list[dict[str, Any]]) -> str:
    """Stat tiles. Each item: ``{label, value, note?, status?}``."""
    cards = []
    for it in items:
        color = STATUS[it["status"]] if it.get("status") else TEXT_PRIMARY
        note = f'<div class="rs-kpi-note">{it["note"]}</div>' if it.get("note") else ""
        cards.append(
            f'<div class="rs-kpi"><div class="rs-kpi-label">{it["label"]}</div>'
            f'<div class="rs-kpi-value" style="color:{color}">{it["value"]}</div>'
            f"{note}</div>"
        )
    return f'<div class="rs-kpis">{"".join(cards)}</div>'


def pill(text: str, status: str) -> str:
    """Status pill — dot plus text, so meaning never rests on colour alone."""
    c = STATUS[status]
    return (
        f'<span class="rs-pill" style="color:{c};border-color:{c}33;'
        f'background:{c}14"><span class="rs-dot" style="background:{c}"></span>'
        f"{text}</span>"
    )


def card_title(title: str, sub: str = "") -> str:
    """Chart card heading with optional sub-line."""
    s = f'<div class="rs-card-sub">{sub}</div>' if sub else ""
    return f'<div class="rs-card-title">{title}</div>{s}'


def test_row(name: str, value: str, verdict: str, status: str) -> str:
    """One statistical-test line: name, statistic, pass/fail pill."""
    return (
        f'<div class="rs-test"><span class="rs-test-name">{name}</span>'
        f'<span class="rs-test-val">{value} &nbsp; {pill(verdict, status)}</span></div>'
    )


def note(text: str) -> str:
    """Recessive caveat block."""
    return f'<div class="rs-note">{text}</div>'


PLOTLY_CONFIG = {"displayModeBar": False, "scrollZoom": False}


@contextmanager
def card(title: str = "", sub: str = "") -> Iterator[None]:
    """A bordered panel with an optional heading.

    Uses Streamlit's own bordered container rather than a raw ``<div>``:
    Streamlit auto-closes injected HTML, so a hand-rolled wrapper renders as
    an empty box with the content spilling out beneath it.
    """
    with st.container(border=True):
        if title:
            st.markdown(card_title(title, sub), unsafe_allow_html=True)
        yield


def chart_card(title: str, sub: str, fig: go.Figure) -> None:
    """The common case: a titled panel containing exactly one chart."""
    with card(title, sub):
        st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)
