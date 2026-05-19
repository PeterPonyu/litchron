"""CitationVerifier tests with mocked HTTP — no live network, no real model."""
from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

from litchron.citations import CitationInput, CitationVerifier

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_CROSSREF_PAYLOAD_OK = {
    "title": ["A Study of Single-Cell Pseudotime Ordering Methods"],
    "issued": {"date-parts": [[2019]]},
    "author": [
        {"given": "Jane", "family": "Smith"},
        {"given": "Bob", "family": "Jones"},
    ],
    "abstract": "We benchmarked several pseudotime ordering methods on single-cell RNA-seq data.",
}

_CROSSREF_RESPONSE_OK = MagicMock()
_CROSSREF_RESPONSE_OK.status_code = 200
_CROSSREF_RESPONSE_OK.is_success = True
_CROSSREF_RESPONSE_OK.json.return_value = {"message": _CROSSREF_PAYLOAD_OK}


def _make_verifier(monkeypatch: Any, cosine_value: float = 0.8) -> CitationVerifier:
    """Return a CitationVerifier with:
    - _get_with_retries monkeypatched to return _CROSSREF_RESPONSE_OK by default.
    - _compute_cosine monkeypatched to return cosine_value.
    - Cache disabled (fresh tmp dir via tmp_path is not available here; we
      patch _load_global_cache and _save_global_cache instead).

    The system may have proxy env vars (e.g. ALL_PROXY=socks://...) that the
    httpx version installed cannot handle. We strip them before constructing
    the client so the test never touches the network.
    """
    # Clear all proxy-related env vars so httpx.Client() does not choke on
    # SOCKS proxy schemes that are not supported by this httpx build.
    for var in ("ALL_PROXY", "all_proxy", "HTTPS_PROXY", "https_proxy",
                "HTTP_PROXY", "http_proxy", "NO_PROXY", "no_proxy"):
        monkeypatch.delenv(var, raising=False)

    v = CitationVerifier()
    monkeypatch.setattr(v, "_compute_cosine", lambda t1, t2: cosine_value)
    monkeypatch.setattr(v, "_load_global_cache", lambda: {})
    monkeypatch.setattr(v, "_save_global_cache", lambda cache: None)
    return v


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_200_crossref_verified(monkeypatch: Any) -> None:
    """Realistic CrossRef 200 → verified = True."""
    v = _make_verifier(monkeypatch, cosine_value=0.85)
    monkeypatch.setattr(v, "_get_with_retries", lambda url: _CROSSREF_RESPONSE_OK)

    result = v.verify(
        CitationInput(
            doi="10.1000/test.123",
            context="We used pseudotime ordering to reconstruct developmental trajectories.",
            year_claimed=2019,
            authors_claimed=["Jane Smith"],
        )
    )
    assert result.verified is True
    assert result.citation is not None
    assert result.confidence > 0.0


def test_404_drops_with_crossref_404(monkeypatch: Any) -> None:
    """CrossRef 404 → drop with reason crossref_404."""
    v = _make_verifier(monkeypatch)
    monkeypatch.setattr(v, "_get_with_retries", lambda url: None)

    result = v.verify(
        CitationInput(
            doi="10.9999/nonexistent",
            context="Some claim.",
        )
    )
    assert result.verified is False
    assert result.drop_reason == "crossref_404"


def test_cosine_below_threshold_drops(monkeypatch: Any) -> None:
    """Cosine < threshold → drop with reason cosine_below_threshold."""
    v = _make_verifier(monkeypatch, cosine_value=0.3)
    monkeypatch.setattr(v, "_get_with_retries", lambda url: _CROSSREF_RESPONSE_OK)

    result = v.verify(
        CitationInput(
            doi="10.1000/test.cosine",
            context="This paper is about photosynthesis in plants.",
        )
    )
    assert result.verified is False
    assert result.drop_reason == "cosine_below_threshold"


def test_year_mismatch_drops(monkeypatch: Any) -> None:
    """Year mismatch (claimed 2020, real 2019, tolerance=1) → year_mismatch."""
    v = _make_verifier(monkeypatch, cosine_value=0.9)
    monkeypatch.setattr(v, "_get_with_retries", lambda url: _CROSSREF_RESPONSE_OK)

    # Real year is 2019; claim 2025 (> tolerance of 1).
    result = v.verify(
        CitationInput(
            doi="10.1000/test.year",
            context="Single cell pseudotime benchmarking.",
            year_claimed=2025,
        )
    )
    assert result.verified is False
    assert result.drop_reason == "year_mismatch"


def test_author_overlap_zero_drops(monkeypatch: Any) -> None:
    """Author surname overlap = 0 when authors_claimed is non-empty → author_mismatch."""
    v = _make_verifier(monkeypatch, cosine_value=0.9)
    monkeypatch.setattr(v, "_get_with_retries", lambda url: _CROSSREF_RESPONSE_OK)

    # Real authors are Smith and Jones; claim completely different names.
    result = v.verify(
        CitationInput(
            doi="10.1000/test.author",
            context="Single cell pseudotime benchmarking.",
            year_claimed=2019,
            authors_claimed=["Nakamura Hiroshi", "Garcia Maria"],
        )
    )
    assert result.verified is False
    assert result.drop_reason == "author_mismatch"


def test_year_within_tolerance_passes(monkeypatch: Any) -> None:
    """Year within ±1 tolerance → not dropped for year mismatch."""
    v = _make_verifier(monkeypatch, cosine_value=0.9)
    monkeypatch.setattr(v, "_get_with_retries", lambda url: _CROSSREF_RESPONSE_OK)

    # Real year 2019, claimed 2020 → within tolerance=1.
    result = v.verify(
        CitationInput(
            doi="10.1000/test.tol",
            context="Single cell pseudotime benchmarking.",
            year_claimed=2020,
            authors_claimed=["Jane Smith"],
        )
    )
    assert result.verified is True
