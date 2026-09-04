#!/usr/bin/env bash
# RiskSense one-shot setup: venv -> deps -> data -> pipeline -> VaR -> backtest.
# Requirements: Python 3.10+, Java 11+ (for local PySpark), internet access.
# No cloud credentials, no API keys.
set -euo pipefail

cd "$(dirname "$0")"

echo "==> Creating virtual environment (.venv)"
python3 -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate

echo "==> Installing dependencies"
pip install --quiet --upgrade pip
pip install --quiet -r requirements.txt
pip install --quiet -e .

echo "==> Ingesting S&P 500 daily prices (yfinance, Stooq fallback)"
python data/ingest_prices.py

echo "==> Ingesting FRED macro series (keyless endpoint)"
python data/ingest_macro.py

echo "==> Running PySpark returns pipeline"
python -m risksense.pipelines.returns_pipeline

echo "==> Computing VaR/ES for all engines + backtest suite"
python -m risksense.cli

echo "==> Running stress tests (historical replay, scenarios, reverse)"
python -m risksense.stress

echo "==> Done. Launch the dashboard with:"
echo "    source .venv/bin/activate && streamlit run dashboards/streamlit_app.py"
