"""LitChron configuration constants and runtime defaults.

Module-level constants intentionally — these are read-only across the
process; we don't need Pydantic Settings overhead. Environment overrides,
if introduced later, can be layered on top by re-binding at import time.
"""
from __future__ import annotations

from pathlib import Path

# --- Citation verification thresholds -------------------------------------
CITATION_COSINE_THRESHOLD: float = 0.55
CITATION_YEAR_TOLERANCE: int = 1
CITATION_AUTHOR_OVERLAP_MIN: int = 1
CITATION_CACHE_TTL_DAYS: int = 30

# --- HTTP behavior --------------------------------------------------------
CITATION_HTTP_TIMEOUT_S: float = 5.0
CITATION_HTTP_RETRIES: int = 3

CROSSREF_USER_AGENT: str = (
    "LitChron/0.1.0 (mailto:litchron-maintainers@users.noreply.github.com)"
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
