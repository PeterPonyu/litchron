# Citation Verification

## Overview

Every DOI or PMID emitted by the LLM in `propose_ordering` must pass a three-signal verification pipeline before it enters `references.bib`. Citations that fail any signal are quarantined in `state.citations_dropped` with a structured reason code and never appear in the final PDF.

## Three Verification Signals

### Signal 1 — Cosine Similarity (relevance)

- **Model**: `sentence-transformers/all-MiniLM-L6-v2` (loaded from the `dl` conda env HuggingFace cache).
- **Input A**: The `context` string passed by the LLM when calling `verify_doi` / `verify_pmid`. This should be the sentence or claim the citation is meant to support.
- **Input B**: The abstract fetched from CrossRef (`abstract` field) or PubMed (`Summary` → `abstract`). Falls back to the title if no abstract is available.
- **Threshold**: cosine similarity ≥ 0.6.
- **Failure reason code**: `cosine_below_threshold`.

### Signal 2 — Year Match (temporal accuracy)

- **Condition**: Applied only when the LLM provides a `year` argument.
- **Tolerance**: ±1 calendar year (e.g., LLM says 2019, actual is 2018 → pass; actual is 2017 → fail).
- **Failure reason code**: `year_mismatch`.

### Signal 3 — Author Overlap (identity confirmation)

- **Condition**: Applied only when the LLM provides an `authors` list (surnames or full names).
- **Required overlap**: ≥ 1 surname in common (case-insensitive, diacritic-normalised via `unicodedata.normalize("NFC", ...)`).
- **Failure reason code**: `author_mismatch`.

## Reason Code Enum

| Code | Signal | Meaning |
|---|---|---|
| `cosine_below_threshold` | S1 | Abstract does not support the claimed context; wrong paper or hallucinated relevance |
| `year_mismatch` | S2 | LLM-stated year differs from authoritative record by more than ±1 |
| `author_mismatch` | S3 | No surname overlap between LLM-stated authors and authoritative record |
| `crossref_404` | Resolve | DOI not found in CrossRef; likely hallucinated or mistyped |
| `pubmed_404` | Resolve | PMID not found in PubMed E-utilities |
| `network_error` | Resolve | HTTP timeout or non-2xx response after 3 retries; treat as transient — may retry once |
| `title_mismatch` | Meta | Title provided by LLM differs substantially from resolved title (Levenshtein > 0.5); informational, does not block by itself but is logged |

## Configuration

All numeric thresholds are configurable in `litchron/config.py`:

```python
CITATION_COSINE_THRESHOLD: float = 0.6
CITATION_YEAR_TOLERANCE: int = 1
CITATION_AUTHOR_MIN_OVERLAP: int = 1
CITATION_HTTP_TIMEOUT_S: float = 5.0
CITATION_HTTP_RETRIES: int = 3
CITATION_CACHE_TTL_DAYS: int = 30
```

## Cache Architecture

**Global cache**: `~/.cache/litchron/citations.json`
- Keyed by `(scheme, id)` where `scheme` is `"doi"` or `"pmid"`.
- Each entry stores: resolved metadata, verification signals, verdict, `verified_at` timestamp.
- TTL: 30 days from `verified_at`. TTL pruning runs on each cache write (not on reads).
- **Concurrency**: writes acquire `fcntl.flock(LOCK_EX)` on `~/.cache/litchron/citations.lock` for the full RMW window; reads are lock-free with `try/except json.JSONDecodeError` fallback to cache-miss path.

**Per-run cache**: `runs/<run_id>/citation_cache.json`
- Holds in-flight citation contexts for the current run.
- Not shared across runs. Merged into the global cache on `finalize_run`.

## CrossRef Polite Pool

`verify_doi` uses the CrossRef polite pool. The HTTP `User-Agent` header is:

```
LitChron/0.1.0 (mailto:litchron-maintainers@users.noreply.github.com)
```

The contact email defaults to the neutral maintainers alias above and is configurable via the `LITCHRON_CONTACT_EMAIL` environment variable (see `CROSSREF_CONTACT_EMAIL` in `litchron/config.py`).

This grants higher rate limits (~40 req/s soft cap vs. 5 req/s anonymous). The `httpx.Client` applies a token-bucket limiter staying below 40 req/s.

## BibTeX Key Generation

Verified citations are written to `runs/<run_id>/references.bib` with stable keys:

```
<first_author_surname>_<year>_<keyword>
```

Where `keyword` is the first non-stopword from the title, lowercased, ASCII-only.

Collision handling: if the same key already exists in the `.bib` file, append `-a`, `-b`, etc. until unique.

Example: `trapnell_2014_pseudotime`, `trapnell_2014_pseudotime-a`.
