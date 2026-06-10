"""LitChron configuration constants and runtime defaults.

Module-level constants intentionally — these are read-only across the
process; we don't need Pydantic Settings overhead. Select values support
environment overrides (e.g. ``LITCHRON_CONTACT_EMAIL``), read at import time.
"""
from __future__ import annotations

import os
from pathlib import Path

# --- Citation verification thresholds -------------------------------------
# Cosine floor for the relevance signal. Override via LITCHRON_CITATION_COSINE_THRESHOLD.
# Calibration on BEIR/SciFact (studies/citation_threshold_calibration.py) shows the
# current model (all-MiniLM-L6-v2) separates relevant vs. irrelevant abstracts well
# (AUC 0.990), but 0.55 is conservative — it gives precision 1.00 / recall 0.57, i.e.
# it drops ~43% of relevant citations. The data-derived precision>=0.95 operating
# point is ~0.40 (recall 0.86). The threshold is also model-specific, so re-run the
# study if you change EMBED_MODEL. Kept at 0.55 by default (high precision); lower it
# toward 0.40 to recover recall.
CITATION_COSINE_THRESHOLD: float = float(
    os.environ.get("LITCHRON_CITATION_COSINE_THRESHOLD", "0.55")
)
CITATION_YEAR_TOLERANCE: int = 1
CITATION_AUTHOR_OVERLAP_MIN: int = 1
CITATION_CACHE_TTL_DAYS: int = 30

# --- HTTP behavior --------------------------------------------------------
CITATION_HTTP_TIMEOUT_S: float = 5.0
CITATION_HTTP_RETRIES: int = 3

# Contact email embedded in the CrossRef polite-pool User-Agent. Override via
# the LITCHRON_CONTACT_EMAIL env var; defaults to a neutral, non-personal
# maintainers alias.
CROSSREF_CONTACT_EMAIL: str = os.environ.get(
    "LITCHRON_CONTACT_EMAIL", "litchron-maintainers@users.noreply.github.com"
)
CROSSREF_USER_AGENT: str = (
    f"LitChron/0.1.0 (mailto:{CROSSREF_CONTACT_EMAIL})"
)
CROSSREF_RATE_LIMIT_RPS: int = 40
PUBMED_RATE_LIMIT_RPS: int = 3

# --- Filesystem -----------------------------------------------------------
GLOBAL_CACHE_DIR: Path = Path.home() / ".cache" / "litchron"

# --- Embedding model ------------------------------------------------------
EMBED_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"


def ensure_cache_dir() -> Path:
    """Create the LitChron cache directory if missing and return its path."""
    GLOBAL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return GLOBAL_CACHE_DIR
