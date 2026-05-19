"""Unit tests for search_crossref and search_europepmc — no live network."""
from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from litchron.citations import CitationVerifier
from mcp_litchron import tools as _tools


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_verifier(monkeypatch: Any) -> CitationVerifier:
    """Return a CitationVerifier with proxy env vars cleared and cache stubbed."""
    for var in (
        "ALL_PROXY", "all_proxy", "HTTPS_PROXY", "https_proxy",
        "HTTP_PROXY", "http_proxy", "NO_PROXY", "no_proxy",
    ):
        monkeypatch.delenv(var, raising=False)
    v = CitationVerifier()
    monkeypatch.setattr(v, "_load_global_cache", lambda: {})
    monkeypatch.setattr(v, "_save_global_cache", lambda cache: None)
    return v


def _mock_response(payload: dict[str, Any]) -> MagicMock:
    resp = MagicMock()
    resp.status_code = 200
    resp.is_success = True
    resp.json.return_value = payload
    return resp


# ---------------------------------------------------------------------------
# CrossRef fake payloads
# ---------------------------------------------------------------------------

_CROSSREF_SEARCH_PAYLOAD = {
    "message": {
        "items": [
            # Item 1: complete record
            {
                "DOI": "10.1000/hlf.marker.1",
                "title": ["HLF marks hematopoietic stem cells"],
                "issued": {"date-parts": [[2021]]},
                "author": [
                    {"given": "Alice", "family": "Brown"},
                    {"given": "Bob", "family": "Chen"},
                ],
                "abstract": "<jats:p>HLF is a key marker of long-term HSCs.</jats:p>",
            },
            # Item 2: missing DOI — should be filtered out
            {
                "title": ["Some other paper without a DOI"],
                "issued": {"date-parts": [[2020]]},
                "author": [{"given": "X", "family": "Y"}],
            },
            # Item 3: complete record
            {
                "DOI": "10.1000/hlf.marker.2",
                "title": ["Transcription factor HLF in HSC maintenance"],
                "issued": {"date-parts": [[2022]]},
                "author": [{"given": "Carol", "family": "Davis"}],
                "abstract": None,
            },
        ]
    }
}


# ---------------------------------------------------------------------------
# Europe PMC fake payloads
# ---------------------------------------------------------------------------

_EPMC_SEARCH_PAYLOAD = {
    "resultList": {
        "result": [
            # Item 1: PMID-based record
            {
                "pmid": "12345678",
                "doi": "10.1234/epmc.1",
                "title": "GATA2 expression in hematopoietic progenitors",
                "pubYear": "2019",
                "authorString": "Wang J, Li X, Zhang Y",
                "abstractText": "GATA2 controls HSC fate decisions.",
            },
            # Item 2: no pmid, has DOI
            {
                "doi": "10.5678/epmc.2",
                "title": "SCL drives erythroid differentiation",
                "pubYear": "2018",
                "authorString": "Smith A, Jones B",
                "abstractText": None,
            },
            # Item 3: no pmid, no doi — should be filtered out
            {
                "title": "Paper with no identifier",
                "pubYear": "2017",
                "authorString": "Anon",
            },
        ]
    }
}


# ---------------------------------------------------------------------------
# Tests: CitationVerifier.search_crossref
# ---------------------------------------------------------------------------

class TestSearchCrossref:
    def test_returns_two_citations_filters_missing_doi(self, monkeypatch: Any) -> None:
        """3 items in payload, 1 missing DOI → 2 Citations returned."""
        v = _make_verifier(monkeypatch)
        monkeypatch.setattr(
            v, "_get_with_retries",
            lambda url: _mock_response(_CROSSREF_SEARCH_PAYLOAD),
        )
        results = v.search_crossref("hematopoietic stem cell hlf marker", max_results=5)
        assert len(results) == 2
        dois = [c.id for c in results]
        assert "10.1000/hlf.marker.1" in dois
        assert "10.1000/hlf.marker.2" in dois

    def test_url_contains_encoded_query(self, monkeypatch: Any) -> None:
        """URL passed to _get_with_retries must URL-encode the query string."""
        captured_urls: list[str] = []

        def _fake_get(url: str) -> MagicMock:
            captured_urls.append(url)
            return _mock_response(_CROSSREF_SEARCH_PAYLOAD)

        v = _make_verifier(monkeypatch)
        monkeypatch.setattr(v, "_get_with_retries", _fake_get)
        v.search_crossref("hematopoietic stem cell hlf marker", max_results=5)

        assert len(captured_urls) == 1
        url = captured_urls[0]
        assert "api.crossref.org/works" in url
        # Spaces must be encoded (either as + or %20)
        assert " " not in url
        assert "hematopoietic" in url

    def test_confidence_is_zero(self, monkeypatch: Any) -> None:
        """Search results are unverified: confidence must be 0.0."""
        v = _make_verifier(monkeypatch)
        monkeypatch.setattr(
            v, "_get_with_retries",
            lambda url: _mock_response(_CROSSREF_SEARCH_PAYLOAD),
        )
        results = v.search_crossref("hlf", max_results=5)
        assert all(c.confidence == 0.0 for c in results)

    def test_source_is_crossref(self, monkeypatch: Any) -> None:
        v = _make_verifier(monkeypatch)
        monkeypatch.setattr(
            v, "_get_with_retries",
            lambda url: _mock_response(_CROSSREF_SEARCH_PAYLOAD),
        )
        results = v.search_crossref("hlf", max_results=5)
        assert all(c.source == "crossref" for c in results)

    def test_jats_tags_stripped_from_abstract(self, monkeypatch: Any) -> None:
        """Jats XML tags should be stripped from CrossRef abstracts."""
        v = _make_verifier(monkeypatch)
        monkeypatch.setattr(
            v, "_get_with_retries",
            lambda url: _mock_response(_CROSSREF_SEARCH_PAYLOAD),
        )
        results = v.search_crossref("hlf", max_results=5)
        r = next(c for c in results if c.id == "10.1000/hlf.marker.1")
        assert r.abstract is not None
        assert "<" not in r.abstract
        assert "HLF is a key marker" in r.abstract

    def test_network_error_returns_empty(self, monkeypatch: Any) -> None:
        """_get_with_retries returning None → empty list, no exception."""
        v = _make_verifier(monkeypatch)
        monkeypatch.setattr(v, "_get_with_retries", lambda url: None)
        results = v.search_crossref("hlf")
        assert results == []

    def test_year_filter_appended_to_url(self, monkeypatch: Any) -> None:
        captured_urls: list[str] = []

        def _fake_get(url: str) -> MagicMock:
            captured_urls.append(url)
            return _mock_response({"message": {"items": []}})

        v = _make_verifier(monkeypatch)
        monkeypatch.setattr(v, "_get_with_retries", _fake_get)
        v.search_crossref("hlf", max_results=5, year_min=2018, year_max=2023)

        assert "from-pub-date:2018" in captured_urls[0]
        assert "until-pub-date:2023" in captured_urls[0]


# ---------------------------------------------------------------------------
# Tests: CitationVerifier.search_europepmc
# ---------------------------------------------------------------------------

class TestSearchEuropePMC:
    def test_returns_two_citations_filters_no_identifier(self, monkeypatch: Any) -> None:
        """3 items; 1 has no pmid/doi → 2 Citations returned."""
        v = _make_verifier(monkeypatch)
        monkeypatch.setattr(
            v, "_get_with_retries",
            lambda url: _mock_response(_EPMC_SEARCH_PAYLOAD),
        )
        results = v.search_europepmc("hematopoietic progenitor GATA2", max_results=5)
        assert len(results) == 2

    def test_pmid_preferred_over_doi(self, monkeypatch: Any) -> None:
        """Item with pmid → scheme='pmid', id=pmid value."""
        v = _make_verifier(monkeypatch)
        monkeypatch.setattr(
            v, "_get_with_retries",
            lambda url: _mock_response(_EPMC_SEARCH_PAYLOAD),
        )
        results = v.search_europepmc("GATA2", max_results=5)
        pmid_results = [c for c in results if c.scheme == "pmid"]
        assert len(pmid_results) == 1
        assert pmid_results[0].id == "12345678"

    def test_doi_fallback_when_no_pmid(self, monkeypatch: Any) -> None:
        """Item without pmid but with doi → scheme='doi'."""
        v = _make_verifier(monkeypatch)
        monkeypatch.setattr(
            v, "_get_with_retries",
            lambda url: _mock_response(_EPMC_SEARCH_PAYLOAD),
        )
        results = v.search_europepmc("SCL erythroid", max_results=5)
        doi_results = [c for c in results if c.scheme == "doi"]
        assert len(doi_results) == 1
        assert doi_results[0].id == "10.5678/epmc.2"

    def test_source_is_pubmed(self, monkeypatch: Any) -> None:
        v = _make_verifier(monkeypatch)
        monkeypatch.setattr(
            v, "_get_with_retries",
            lambda url: _mock_response(_EPMC_SEARCH_PAYLOAD),
        )
        results = v.search_europepmc("GATA2", max_results=5)
        assert all(c.source == "pubmed" for c in results)

    def test_confidence_is_zero(self, monkeypatch: Any) -> None:
        v = _make_verifier(monkeypatch)
        monkeypatch.setattr(
            v, "_get_with_retries",
            lambda url: _mock_response(_EPMC_SEARCH_PAYLOAD),
        )
        results = v.search_europepmc("GATA2", max_results=5)
        assert all(c.confidence == 0.0 for c in results)

    def test_url_contains_encoded_query(self, monkeypatch: Any) -> None:
        captured_urls: list[str] = []

        def _fake_get(url: str) -> MagicMock:
            captured_urls.append(url)
            return _mock_response(_EPMC_SEARCH_PAYLOAD)

        v = _make_verifier(monkeypatch)
        monkeypatch.setattr(v, "_get_with_retries", _fake_get)
        v.search_europepmc("hematopoietic progenitor GATA2", max_results=5)

        assert len(captured_urls) == 1
        url = captured_urls[0]
        assert "europepmc" in url
        assert " " not in url
        assert "hematopoietic" in url

    def test_network_error_returns_empty(self, monkeypatch: Any) -> None:
        v = _make_verifier(monkeypatch)
        monkeypatch.setattr(v, "_get_with_retries", lambda url: None)
        results = v.search_europepmc("GATA2")
        assert results == []


# ---------------------------------------------------------------------------
# Tests: MCP tool wrappers
# ---------------------------------------------------------------------------

class TestMCPSearchCrossref:
    def test_returns_dict_with_query_and_results(self, monkeypatch: Any) -> None:
        v = _make_verifier(monkeypatch)
        monkeypatch.setattr(
            v, "_get_with_retries",
            lambda url: _mock_response(_CROSSREF_SEARCH_PAYLOAD),
        )
        monkeypatch.setattr(_tools, "_VERIFIER", v)
        result = _tools.search_crossref("hlf marker", max_results=5)
        assert isinstance(result, dict)
        assert result["query"] == "hlf marker"
        assert result["n_results"] == 2
        assert len(result["results"]) == 2

    def test_max_results_out_of_range_returns_error(self, monkeypatch: Any) -> None:
        v = _make_verifier(monkeypatch)
        monkeypatch.setattr(_tools, "_VERIFIER", v)
        result = _tools.search_crossref("hlf", max_results=0)
        assert hasattr(result, "code") or (isinstance(result, dict) and "code" in result)

    def test_max_results_51_returns_error(self, monkeypatch: Any) -> None:
        v = _make_verifier(monkeypatch)
        monkeypatch.setattr(_tools, "_VERIFIER", v)
        result = _tools.search_crossref("hlf", max_results=51)
        from mcp_litchron.errors import ErrorResult
        assert isinstance(result, ErrorResult)
        assert result.code == "invalid_input"


class TestMCPSearchEuropePMC:
    def test_returns_dict_with_query_and_results(self, monkeypatch: Any) -> None:
        v = _make_verifier(monkeypatch)
        monkeypatch.setattr(
            v, "_get_with_retries",
            lambda url: _mock_response(_EPMC_SEARCH_PAYLOAD),
        )
        monkeypatch.setattr(_tools, "_VERIFIER", v)
        result = _tools.search_europepmc("GATA2 hematopoietic", max_results=5)
        assert isinstance(result, dict)
        assert result["query"] == "GATA2 hematopoietic"
        assert result["n_results"] == 2

    def test_max_results_51_returns_error(self, monkeypatch: Any) -> None:
        v = _make_verifier(monkeypatch)
        monkeypatch.setattr(_tools, "_VERIFIER", v)
        result = _tools.search_europepmc("GATA2", max_results=51)
        from mcp_litchron.errors import ErrorResult
        assert isinstance(result, ErrorResult)
        assert result.code == "invalid_input"


# ---------------------------------------------------------------------------
# Registry count check
# ---------------------------------------------------------------------------

def test_tool_registry_count() -> None:
    """TOOL_REGISTRY should have 18 entries after R1+R2+R3 (12 base + 2 search + 2 figure + 2 LLM-pseudotime)."""
    assert len(_tools.TOOL_REGISTRY) == 18


def test_tool_registry_contains_search_tools() -> None:
    names = {t.name for t in _tools.TOOL_REGISTRY}
    assert "search_crossref" in names
    assert "search_europepmc" in names
