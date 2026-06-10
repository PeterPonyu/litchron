"""Citation verification — the LitChron trust boundary.

LLMs hallucinate DOIs and PMIDs constantly. Every citation that ends up
in the final report must be round-tripped through CrossRef or PubMed and
cross-checked against:

* **Cosine similarity** between the LLM's surrounding claim and the
  paper's abstract (falls back to the title when abstract is empty).
* **Year tolerance** of ±1 against the LLM-claimed year (if any).
* **Surname overlap** with LLM-claimed authors (if any).

Anything that fails any check is dropped to ``citations_dropped`` in the
run state with a machine-readable reason — never silently mutated.
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Literal, Optional

import httpx
from pydantic import BaseModel, Field

from .config import (
    CITATION_AUTHOR_OVERLAP_MIN,
    CITATION_CACHE_TTL_DAYS,
    CITATION_COSINE_THRESHOLD,
    CITATION_HTTP_RETRIES,
    CITATION_HTTP_TIMEOUT_S,
    CITATION_YEAR_TOLERANCE,
    CROSSREF_USER_AGENT,
    EMBED_MODEL,
    GLOBAL_CACHE_DIR,
    ensure_cache_dir,
)
from .state import file_lock


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
class Citation(BaseModel):
    """Verified citation record returned from CrossRef or PubMed."""

    scheme: Literal["doi", "pmid"]
    id: str
    title: str
    year: int
    authors: list[str] = Field(default_factory=list)
    abstract: Optional[str] = None
    confidence: float
    source: Literal["crossref", "pubmed"]


class CitationInput(BaseModel):
    """LLM-supplied citation candidate awaiting verification."""

    doi: Optional[str] = None
    pmid: Optional[str] = None
    context: str  # the surrounding claim text, used for cosine
    year_claimed: Optional[int] = None
    authors_claimed: list[str] = Field(default_factory=list)


class CitationVerdict(BaseModel):
    """Outcome of a single verification attempt."""

    verified: bool
    citation: Optional[Citation] = None
    drop_reason: Optional[str] = None
    confidence: float = 0.0
    signals: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _surname(name: str) -> str:
    """Extract a comparable surname token from a free-form author string.

    Handles "Smith, John A." and "John A. Smith" forms; falls back to the last
    whitespace-separated token otherwise. Lowercased and diacritic-folded
    (NFKD decomposition + combining-mark removal) so "Müller" and "Muller"
    compare equal. Base characters are preserved, so non-Latin surnames (e.g.
    CJK) still match exactly rather than being dropped.
    """
    if not name:
        return ""
    import unicodedata

    s = name.strip()
    if "," in s:
        s = s.split(",", 1)[0]
    else:
        s = s.split()[-1] if s.split() else s
    folded = "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))
    return folded.lower()


def _author_overlap(claimed: Iterable[str], verified: Iterable[str]) -> int:
    a = {_surname(n) for n in claimed if n}
    b = {_surname(n) for n in verified if n}
    a.discard("")
    b.discard("")
    return len(a & b)


# Inline LaTeX escape table for the bibtex emitter; covers the chars we
# most often see in biomedical author lists. unicode_to_latex is optional
# (not in pyproject deps) so we fall back to this table when unavailable.
_LATEX_UNICODE_MAP = {
    "ü": r"\"{u}",
    "Ü": r"\"{U}",
    "ö": r"\"{o}",
    "Ö": r"\"{O}",
    "ä": r"\"{a}",
    "Ä": r"\"{A}",
    "ß": r"\ss{}",
    "é": r"\'{e}",
    "É": r"\'{E}",
    "è": r"\`{e}",
    "È": r"\`{E}",
    "ê": r"\^{e}",
    "Ê": r"\^{E}",
    "ñ": r"\~{n}",
    "Ñ": r"\~{N}",
    "à": r"\`{a}",
    "À": r"\`{A}",
    "â": r"\^{a}",
    "Â": r"\^{A}",
    "ç": r"\c{c}",
    "Ç": r"\c{C}",
    "í": r"\'{i}",
    "Í": r"\'{I}",
    "ó": r"\'{o}",
    "Ó": r"\'{O}",
    "ú": r"\'{u}",
    "Ú": r"\'{U}",
}


def _latex_escape(text: str) -> str:
    r"""Escape reserved LaTeX chars + Unicode for a BibTeX field value.

    Order is load-bearing: we escape reserved chars on the *raw* string
    first, THEN substitute Unicode → LaTeX command sequences. Doing it
    the other way mangles backslashes/braces we just introduced for
    accents (turning ``\"{u}`` into ``\textbackslash{}\{\}"\{u\}``).
    """
    # Pass 1: reserved chars on the raw (still-Unicode) string.
    text = (
        text.replace("\\", r"\textbackslash{}")
        .replace("&", r"\&")
        .replace("%", r"\%")
        .replace("#", r"\#")
        .replace("_", r"\_")
        .replace("{", r"\{")
        .replace("}", r"\}")
    )
    # Pass 2: Unicode → LaTeX accent commands. unicode_to_latex (if
    # installed) handles a broader Unicode range; otherwise our small
    # inline table covers the most common biomedical author accents.
    try:
        from unicode_to_latex import unicode_to_latex as _utl  # type: ignore

        return _utl(text)
    except ImportError:
        out: list[str] = []
        for ch in text:
            out.append(_LATEX_UNICODE_MAP.get(ch, ch))
        return "".join(out)


def _ascii_fold(text: str) -> str:
    """Cheap ASCII fold for cite-key generation; not a full normalizer."""
    import unicodedata

    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(c for c in decomposed if not unicodedata.combining(c) and ord(c) < 128)


# ---------------------------------------------------------------------------
# Verifier
# ---------------------------------------------------------------------------
class CitationVerifier:
    """Verifies LLM-emitted citations against CrossRef and PubMed.

    The embed model is loaded lazily on first cosine call so that simply
    importing this module (e.g. for type checking) is cheap.
    """

    def __init__(
        self,
        embed_model_name: str = EMBED_MODEL,
        http_timeout: float = CITATION_HTTP_TIMEOUT_S,
    ) -> None:
        self.embed_model_name: str = embed_model_name
        self.http_timeout: float = http_timeout

        self._embed_model: Any = None  # sentence_transformers.SentenceTransformer
        self._http: httpx.Client = httpx.Client(
            timeout=http_timeout,
            headers={"User-Agent": CROSSREF_USER_AGENT},
            follow_redirects=False,  # CrossRef/PubMed do not legitimately redirect; treat 3xx as miss
            trust_env=False,  # ignore HTTP(S)_PROXY/SOCKS_PROXY — these academic APIs do not need them
        )

        ensure_cache_dir()
        self.cache_path: Path = GLOBAL_CACHE_DIR / "citations.json"
        self.cache_lock_path: Path = GLOBAL_CACHE_DIR / "citations.lock"

    # -- lazy embedding model ---------------------------------------------
    def _ensure_embed_model(self) -> Any:
        if self._embed_model is None:
            # Imported here so cold module import doesn't drag torch in.
            import os

            from sentence_transformers import SentenceTransformer

            # HF Hub uses httpx and respects env proxies; for a model that's
            # likely already cached locally, unset the proxy env vars so the
            # initial cache check doesn't blow up on unsupported SOCKS schemes.
            # This mirrors the verifier's own ``trust_env=False`` discipline.
            proxy_keys = (
                "ALL_PROXY",
                "all_proxy",
                "HTTP_PROXY",
                "http_proxy",
                "HTTPS_PROXY",
                "https_proxy",
            )
            saved = {k: os.environ.pop(k, None) for k in proxy_keys}
            try:
                self._embed_model = SentenceTransformer(self.embed_model_name)
            finally:
                for k, v in saved.items():
                    if v is not None:
                        os.environ[k] = v
        return self._embed_model

    # -- cache -------------------------------------------------------------
    def _load_global_cache(self) -> dict[str, dict[str, Any]]:
        """Lock-free read; corruption → empty dict (don't poison the run)."""
        if not self.cache_path.exists():
            return {}
        try:
            with self.cache_path.open("r", encoding="utf-8") as fh:
                data = json.load(fh)
            if isinstance(data, dict):
                return data
            return {}
        except (json.JSONDecodeError, OSError):
            return {}

    def _save_global_cache(self, cache: dict[str, dict[str, Any]]) -> None:
        """RMW under flock; prune entries older than TTL at write time."""
        cutoff = datetime.now(timezone.utc) - timedelta(days=CITATION_CACHE_TTL_DAYS)
        with file_lock(self.cache_lock_path):
            # Re-read to merge any concurrent writes done since our load.
            disk = self._load_global_cache()
            disk.update(cache)
            pruned: dict[str, dict[str, Any]] = {}
            for k, v in disk.items():
                vat = v.get("verified_at") if isinstance(v, dict) else None
                if not isinstance(vat, str):
                    # No timestamp → keep (will be re-stamped on next hit).
                    pruned[k] = v
                    continue
                try:
                    ts = datetime.fromisoformat(vat)
                except ValueError:
                    pruned[k] = v
                    continue
                if ts >= cutoff:
                    pruned[k] = v
            tmp = self.cache_path.with_suffix(".json.tmp")
            with tmp.open("w", encoding="utf-8") as fh:
                json.dump(pruned, fh, indent=2, sort_keys=True)
            tmp.replace(self.cache_path)

    # -- HTTP lookups ------------------------------------------------------
    def _get_with_retries(self, url: str) -> Optional[httpx.Response]:
        """GET with exponential backoff; return None on 404 or terminal fail."""
        backoff = 0.5
        for attempt in range(CITATION_HTTP_RETRIES):
            try:
                resp = self._http.get(url)
            except httpx.HTTPError:
                if attempt == CITATION_HTTP_RETRIES - 1:
                    return None
                time.sleep(backoff)
                backoff *= 2
                continue
            if resp.status_code == 404:
                return None
            if resp.status_code >= 500 and attempt < CITATION_HTTP_RETRIES - 1:
                time.sleep(backoff)
                backoff *= 2
                continue
            if resp.is_success:
                return resp
            # Other 4xx → terminal.
            return None
        return None

    def _crossref_lookup(self, doi: str) -> Optional[dict[str, Any]]:
        """Return CrossRef ``message`` dict or None on 404 / network error."""
        from urllib.parse import quote
        url = f"https://api.crossref.org/works/{quote(doi, safe='/')}"
        resp = self._get_with_retries(url)
        if resp is None:
            return None
        try:
            payload = resp.json()
        except ValueError:
            return None
        msg = payload.get("message")
        return msg if isinstance(msg, dict) else None

    def _pubmed_lookup(self, pmid: str) -> Optional[dict[str, Any]]:
        """Return PubMed ESummary result dict or None on 404 / network error."""
        from urllib.parse import quote
        url = (
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
            f"?db=pubmed&id={quote(pmid, safe='')}&retmode=json"
        )
        resp = self._get_with_retries(url)
        if resp is None:
            return None
        try:
            payload = resp.json()
        except ValueError:
            return None
        result = payload.get("result")
        if not isinstance(result, dict):
            return None
        record = result.get(pmid)
        return record if isinstance(record, dict) else None

    # -- normalization -----------------------------------------------------
    @staticmethod
    def _parse_crossref(msg: dict[str, Any]) -> dict[str, Any]:
        """Normalize a CrossRef message dict to title/year/authors/abstract."""
        title_list = msg.get("title") or []
        title = title_list[0] if title_list else ""
        # Year — prefer issued.date-parts; fall back to published-print etc.
        year: Optional[int] = None
        for key in ("issued", "published-print", "published-online", "created"):
            block = msg.get(key)
            if isinstance(block, dict):
                parts = block.get("date-parts")
                if isinstance(parts, list) and parts and isinstance(parts[0], list) and parts[0]:
                    try:
                        year = int(parts[0][0])
                        break
                    except (TypeError, ValueError):
                        continue
        authors: list[str] = []
        for a in msg.get("author") or []:
            if not isinstance(a, dict):
                continue
            given = a.get("given") or ""
            family = a.get("family") or ""
            full = (f"{given} {family}").strip()
            if full:
                authors.append(full)
        abstract = msg.get("abstract") or None
        return {"title": title, "year": year, "authors": authors, "abstract": abstract}

    @staticmethod
    def _parse_pubmed(record: dict[str, Any]) -> dict[str, Any]:
        """Normalize a PubMed ESummary record to title/year/authors/abstract."""
        title = record.get("title") or ""
        year: Optional[int] = None
        pubdate = record.get("pubdate") or record.get("epubdate") or ""
        if isinstance(pubdate, str) and len(pubdate) >= 4 and pubdate[:4].isdigit():
            try:
                year = int(pubdate[:4])
            except ValueError:
                year = None
        authors: list[str] = []
        for a in record.get("authors") or []:
            if isinstance(a, dict):
                name = a.get("name") or ""
                if name:
                    authors.append(name)
        # ESummary doesn't carry the abstract; leave None and let cosine
        # fall back to title.
        return {"title": title, "year": year, "authors": authors, "abstract": None}

    # -- similarity --------------------------------------------------------
    def _compute_cosine(self, text1: str, text2: str) -> float:
        """Sentence-Transformer cosine similarity in [0, 1]-ish range."""
        if not text2:
            return 0.0
        if not text1:
            return 0.0
        model = self._ensure_embed_model()
        import numpy as np

        emb = model.encode([text1, text2], normalize_embeddings=True)
        v1, v2 = np.asarray(emb[0]), np.asarray(emb[1])
        return float(np.dot(v1, v2))

    # -- public verify -----------------------------------------------------
    def verify(self, input: CitationInput) -> CitationVerdict:
        """Verify a single :class:`CitationInput`; never raises on bad data."""
        # --- cache key ---------------------------------------------------
        if input.doi:
            cache_key = f"doi:{input.doi.lower()}"
            scheme: Literal["doi", "pmid"] = "doi"
            source: Literal["crossref", "pubmed"] = "crossref"
        elif input.pmid:
            cache_key = f"pmid:{input.pmid}"
            scheme = "pmid"
            source = "pubmed"
        else:
            return CitationVerdict(
                verified=False,
                drop_reason="no_id",
                confidence=0.0,
                signals={"reason": "neither doi nor pmid provided"},
            )

        cache = self._load_global_cache()
        cached = cache.get(cache_key)
        if isinstance(cached, dict) and cached.get("verified") is True:
            # Re-validate by reconstructing the Citation from cache.
            try:
                cit = Citation.model_validate(cached["citation"])
                return CitationVerdict(
                    verified=True,
                    citation=cit,
                    confidence=cit.confidence,
                    signals=cached.get("signals", {"cached": True}),
                )
            except Exception:
                # Corrupt cache entry — fall through to re-fetch.
                pass

        # --- network lookup ----------------------------------------------
        if scheme == "doi":
            msg = self._crossref_lookup(input.doi or "")
            if msg is None:
                return CitationVerdict(
                    verified=False,
                    drop_reason="crossref_404",
                    confidence=0.0,
                    signals={"id": input.doi},
                )
            parsed = self._parse_crossref(msg)
        else:
            record = self._pubmed_lookup(input.pmid or "")
            if record is None:
                return CitationVerdict(
                    verified=False,
                    drop_reason="pubmed_404",
                    confidence=0.0,
                    signals={"id": input.pmid},
                )
            parsed = self._parse_pubmed(record)

        title: str = parsed["title"] or ""
        year: Optional[int] = parsed["year"]
        authors: list[str] = parsed["authors"]
        abstract: Optional[str] = parsed["abstract"]

        # --- cosine ------------------------------------------------------
        comparison_text = abstract if abstract else title
        cosine = self._compute_cosine(input.context or "", comparison_text or "")
        signals: dict[str, Any] = {
            "cosine": cosine,
            "year_match": False,
            "author_overlap": 0,
        }

        if cosine < CITATION_COSINE_THRESHOLD:
            return CitationVerdict(
                verified=False,
                drop_reason="cosine_below_threshold",
                confidence=max(0.0, min(1.0, cosine)),
                signals=signals,
            )

        # --- year check --------------------------------------------------
        year_match = True
        if input.year_claimed is not None and year is not None:
            year_match = abs(year - input.year_claimed) <= CITATION_YEAR_TOLERANCE
        signals["year_match"] = year_match
        if input.year_claimed is not None and year is not None and not year_match:
            return CitationVerdict(
                verified=False,
                drop_reason="year_mismatch",
                confidence=max(0.0, min(1.0, cosine)),
                signals=signals,
            )

        # --- author overlap ---------------------------------------------
        overlap = _author_overlap(input.authors_claimed, authors)
        signals["author_overlap"] = overlap
        if input.authors_claimed and overlap < CITATION_AUTHOR_OVERLAP_MIN:
            return CitationVerdict(
                verified=False,
                drop_reason="author_mismatch",
                confidence=max(0.0, min(1.0, cosine)),
                signals=signals,
            )

        # --- success -----------------------------------------------------
        confidence = max(0.0, min(1.0, cosine))
        citation = Citation(
            scheme=scheme,
            id=(input.doi if scheme == "doi" else input.pmid) or "",
            title=title,
            year=year if year is not None else 0,
            authors=authors,
            abstract=abstract,
            confidence=confidence,
            source=source,
        )
        verdict = CitationVerdict(
            verified=True,
            citation=citation,
            confidence=confidence,
            signals=signals,
        )

        # Persist to cache.
        cache[cache_key] = {
            "verified": True,
            "verified_at": _now_iso(),
            "citation": citation.model_dump(mode="json"),
            "signals": signals,
        }
        self._save_global_cache(cache)
        return verdict

    # -- search methods ---------------------------------------------------
    def search_crossref(
        self,
        query: str,
        max_results: int = 10,
        year_min: Optional[int] = None,
        year_max: Optional[int] = None,
    ) -> list[Citation]:
        """Search CrossRef for papers matching a free-text biological query.

        Calls ``https://api.crossref.org/works?query=...&rows=N`` with
        optional date filters. Returns up to ``max_results`` :class:`Citation`
        objects with ``confidence=0.0`` (unverified — pass each DOI through
        :meth:`verify` to compute a real confidence score).

        CrossRef abstracts are delivered as Jats XML.  We strip tags with a
        simple ``<[^>]+>`` regex; the result may still contain XML entity
        references (``&amp;``, ``&lt;``, etc.) that are not further decoded
        here — callers that need clean prose should post-process.

        On any HTTP error the method returns ``[]`` (never raises).
        """
        import re
        from urllib.parse import quote

        parts = [
            f"https://api.crossref.org/works?query={quote(query)}&rows={max_results}"
        ]
        filters: list[str] = []
        if year_min is not None:
            filters.append(f"from-pub-date:{year_min}")
        if year_max is not None:
            filters.append(f"until-pub-date:{year_max}")
        if filters:
            parts.append(f"&filter={','.join(filters)}")
        url = "".join(parts)

        resp = self._get_with_retries(url)
        if resp is None:
            return []
        try:
            payload = resp.json()
        except ValueError:
            return []

        message = payload.get("message")
        if not isinstance(message, dict):
            return []
        items = message.get("items")
        if not isinstance(items, list):
            return []

        _jats_tag = re.compile(r"<[^>]+>")

        results: list[Citation] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            doi_raw = item.get("DOI")
            if not doi_raw:
                continue  # skip items lacking a DOI
            doi = str(doi_raw)
            parsed = self._parse_crossref(item)
            title = parsed["title"] or ""
            year: int = parsed["year"] if parsed["year"] is not None else 0
            authors: list[str] = parsed["authors"]
            abstract_raw: Optional[str] = parsed["abstract"]
            # Strip Jats XML tags from abstract (CrossRef delivers Jats XML).
            abstract: Optional[str] = None
            if abstract_raw:
                abstract = _jats_tag.sub("", abstract_raw).strip() or None
            results.append(
                Citation(
                    scheme="doi",
                    id=doi,
                    title=title,
                    year=year,
                    authors=authors,
                    abstract=abstract,
                    confidence=0.0,
                    source="crossref",
                )
            )
            if len(results) >= max_results:
                break
        return results

    def search_europepmc(
        self,
        query: str,
        max_results: int = 10,
    ) -> list[Citation]:
        """Search Europe PMC for biomedical papers matching a free-text query.

        Calls ``https://www.ebi.ac.uk/europepmc/webservices/rest/search`` with
        ``format=json``.  Prefers PMID as the identifier; falls back to DOI
        when PMID is absent.  This endpoint may return clinical and biomedical
        hits that CrossRef misses.

        Returns up to ``max_results`` :class:`Citation` objects with
        ``confidence=0.0`` (unverified).  On any HTTP error returns ``[]``.
        """
        from urllib.parse import quote

        url = (
            "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
            f"?query={quote(query)}&format=json&pageSize={max_results}"
        )
        resp = self._get_with_retries(url)
        if resp is None:
            return []
        try:
            payload = resp.json()
        except ValueError:
            return []

        result_list = payload.get("resultList")
        if not isinstance(result_list, dict):
            return []
        items = result_list.get("result")
        if not isinstance(items, list):
            return []

        results: list[Citation] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            pmid_raw = item.get("pmid")
            doi_raw = item.get("doi")
            if pmid_raw:
                scheme: Literal["doi", "pmid"] = "pmid"
                ident = str(pmid_raw)
            elif doi_raw:
                scheme = "doi"
                ident = str(doi_raw)
            else:
                continue  # no usable identifier

            title = str(item.get("title") or "")
            pub_year_raw = item.get("pubYear") or item.get("firstPublicationDate") or ""
            year = 0
            if isinstance(pub_year_raw, int):
                year = pub_year_raw
            elif isinstance(pub_year_raw, str) and len(pub_year_raw) >= 4 and pub_year_raw[:4].isdigit():
                try:
                    year = int(pub_year_raw[:4])
                except ValueError:
                    year = 0

            # Europe PMC returns author list as a string: "LastA A, LastB B, ..."
            author_string = item.get("authorString") or ""
            authors: list[str] = (
                [a.strip() for a in author_string.split(",") if a.strip()]
                if author_string
                else []
            )
            abstract: Optional[str] = item.get("abstractText") or None

            results.append(
                Citation(
                    scheme=scheme,
                    id=ident,
                    title=title,
                    year=year,
                    authors=authors,
                    abstract=abstract,
                    confidence=0.0,
                    source="pubmed",
                )
            )
            if len(results) >= max_results:
                break
        return results

    # -- bibtex emitter ---------------------------------------------------
    @staticmethod
    def to_bibtex(citations: list[Citation]) -> str:
        """Render citations as a BibTeX string with stable, unique keys."""
        used_keys: dict[str, int] = {}
        entries: list[str] = []

        for cit in citations:
            first_author_surname = ""
            if cit.authors:
                first_author_surname = _ascii_fold(_surname(cit.authors[0]))
            first_author_surname = first_author_surname or "anon"

            # Keyword: first non-trivial word of the title (ASCII-folded,
            # lowercased) — cite keys must round-trip safely through BibTeX.
            keyword = "ref"
            for tok in (cit.title or "").split():
                t = "".join(c for c in _ascii_fold(tok) if c.isalnum()).lower()
                if t and t not in {"the", "a", "an", "of", "on", "in", "and", "for"}:
                    keyword = t
                    break

            base_key = f"{first_author_surname}_{cit.year or 'nd'}_{keyword}"
            # Collision handling: append -a, -b, ...
            if base_key in used_keys:
                used_keys[base_key] += 1
                suffix = chr(ord("a") + used_keys[base_key] - 1)
                key = f"{base_key}-{suffix}"
            else:
                used_keys[base_key] = 0
                key = base_key

            authors_field = " and ".join(_latex_escape(a) for a in cit.authors) or "Anonymous"
            title_field = _latex_escape(cit.title or "")
            year_field = str(cit.year) if cit.year else "n.d."

            id_kind = "doi" if cit.scheme == "doi" else "pmid"
            id_field = _latex_escape(cit.id)

            entries.append(
                "@article{" + key + ",\n"
                f"  author = {{{authors_field}}},\n"
                f"  title  = {{{title_field}}},\n"
                f"  year   = {{{year_field}}},\n"
                f"  {id_kind:<6} = {{{id_field}}}\n"
                "}\n"
            )

        return "\n".join(entries)
