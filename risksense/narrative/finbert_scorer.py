"""FinBERT narrative risk scoring (Month 4 — v2 differentiator).

Paragraph-level FinBERT (ProsusAI/finbert) sentiment inference over 10-K
Item 1A Risk Factors, batch-run on Google Colab's free GPU; embeddings and
scores saved to parquet so downstream steps stay laptop-runnable. Aggregated
monthly to a firm-level Narrative Risk Score (NRS).
"""

from __future__ import annotations


def score_filings(*args: object, **kwargs: object) -> None:
    """Placeholder — implemented in Month 4. See module docstring."""
    raise NotImplementedError("FinBERT scoring ships in Month 4.")
