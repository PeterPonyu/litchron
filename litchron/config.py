"""LitChron configuration constants and runtime defaults.

Module-level constants intentionally — these are read-only across the
process; we don't need Pydantic Settings overhead. Select values support
environment overrides (e.g. ``LITCHRON_CONTACT_EMAIL``), read at import time.
"""
from __future__ import annotations

import os
from pathlib import Path

# --- Citation verification thresholds -------------------------------------
CITATION_COSINE_THRESHOLD: float = 0.55
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

# --- Reproducibility ------------------------------------------------------
# Global random seed for stochastic steps (PCA / UMAP / Leiden and baseline
# clustering), so the numerical comparators are reproducible across runs.
# Override via the LITCHRON_SEED env var.
RANDOM_SEED: int = int(os.environ.get("LITCHRON_SEED", "0"))


def ensure_cache_dir() -> Path:
    """Create the LitChron cache directory if missing and return its path."""
    GLOBAL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return GLOBAL_CACHE_DIR
