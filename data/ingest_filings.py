"""Ingest 10-K Item 1A Risk Factors from SEC EDGAR (Week 4 deliverable).

Planned flow: EDGAR ``submissions`` JSON per CIK → latest 10-K accession →
primary document HTML → Item 1A extraction (regex on item headers with
fallbacks) → paragraph-level parquet for FinBERT scoring on Colab.

SEC fair-access rules require an identifying User-Agent (set the
``EDGAR_USER_AGENT`` env var, e.g. "Name email@example.com") and <=10
requests/second. See https://www.sec.gov/os/accessing-edgar-data.

Status: stub — ships in Week 4 with the narrative overlay.
"""

from __future__ import annotations


def ingest() -> None:
    """Placeholder — implemented in Week 4. See module docstring."""
    raise NotImplementedError("EDGAR filings ingestion ships in Week 4.")


if __name__ == "__main__":
    ingest()
