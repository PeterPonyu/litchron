# LitChron Architecture

## Hybrid Execution Model

LitChron uses a two-tier execution model to contain failure while maximising performance.

**Python-native baselines** (PAGA, Palantir, scVelo, pyslingshot) run in-process inside the MCP server. They share the `AnnDataCache` directly, avoiding repeated h5ad deserialization (each load costs ~3–8 s on the `dl` env cold start). These libraries are pure-Python stacks with no C-level abort risk.

**R-backed baselines** (Monocle3, Slingshot-R) run as isolated subprocesses via `python -m litchron.baselines._r_runner <method> <run_id> <h5ad_path>`. The subprocess loads h5ad, calls rpy2, emits a single JSON line on stdout, then exits. If the R session segfaults (Monocle3's `learn_graph` is a known offender on degenerate UMAP inputs), or if R's allocator raises `R_MAX_VSIZE` abort — both of which bypass Python `try/except` — the subprocess dies but the MCP server process survives. The wrapper (`litchron/baselines/monocle3.py`, `slingshot_r.py`) captures stderr and returns a structured `BaselineFailure` with reason code to the LLM.

**Dispatch rule** in `run_baseline`: `if method in R_BACKED: subprocess else: in-process`. The registry lives in `litchron/baselines/registry.py`.

**rpy2 import discipline**: `rpy2` is imported only inside `_r_runner.py`. No parent module (`litchron`, `litchron.baselines`, `mcp_litchron`) may import `rpy2` at module scope. This invariant is enforced by `tests/unit/test_no_rpy2_in_parent.py`.

## AnnDataCache and Fingerprint Protocol

`mcp_litchron/cache.py:AnnDataCache(max_entries=4)` holds up to four AnnData objects in memory, keyed by `run_id`, with LRU eviction.

**Fingerprint**: `(h5ad_path, st_mtime_ns, st_size, sha1[:16])`. Written to `runs/<run_id>/cache_fingerprint.json` after every load and after every recorded mutation. A fingerprint mismatch on `with_adata()` triggers a cache miss → reload path.

**Zarr-delta persistence**: When a Python baseline mutates `adata` (adds keys to `.uns`, `.obsp`, `.layers`), it reports `adata_delta_keys`. The cache persists the new values as a zarr store at `runs/<run_id>/baselines/<method>/adata_delta.zarr` and appends a `DeltaRef` to `state.adata_deltas`.

**Replay protocol**:
- Deltas are append-only. The `delta_keys` list in each `DeltaRef` is authoritative.
- Conflict resolution: last-writer-wins for overlapping keys across different baselines.
- Replay is idempotent: re-writing the same key with the same value is a no-op.
- Key removal is not supported in v1; attempts record a `quality_flag` but retain the key.

**Resume path**: On cache miss after restart, `AnnDataCache.resume(run_id)` reloads h5ad from disk, then replays all deltas from `state.adata_deltas` in recorded order.

## State as Filesystem

All run state lives under `runs/<run_id>/state.json`. The process holds no authoritative in-memory state beyond the `AnnDataCache`.

**Serialized writes**: `RunState.update(fn)` acquires an exclusive `fcntl.flock` on `runs/<run_id>/state.lock` around the full read-modify-write window. The commit uses `os.replace` (atomic rename) so readers never see a partial write.

**Schema versioning**: `state.json` carries `schema_version: int = 1`. Future migrations check this field before deserializing.

## Multi-Signal Citation Verifier

See `docs/citation-verification.md` for the full signal specification.

**Pipeline** for each LLM-emitted citation:
1. Resolve: CrossRef `api.crossref.org/works/{doi}` or NCBI `esummary.fcgi?id={pmid}`.
2. Cosine similarity between LLM-provided context string and the fetched abstract (all-MiniLM-L6-v2, threshold ≥ 0.6).
3. Year match: LLM-emitted year vs. CrossRef/PubMed year, tolerance ±1.
4. Author overlap: ≥ 1 surname in common when LLM emits authors.

Any signal failure quarantines the citation with a structured reason code. Verified citations enter `references.bib` with stable keys `firstauthor_year_keyword[-a|-b]`.

**Global cache**: `~/.cache/litchron/citations.json`, 30-day TTL, protected by `fcntl.flock` on `~/.cache/litchron/citations.lock`.

## Markdown → LaTeX Pipeline

1. Claude writes section markdown via `append_section`.
2. `litchron/sanitize.py` normalises Unicode dash variants to ASCII `-` outside fenced code blocks **and rejects unsafe raw TeX**. Pandoc is invoked with `--from gfm+tex_math_dollars --to latex`, so any raw TeX in the markdown flows through to `latexmk`. To prevent an LLM-emitted `\input{/etc/passwd}`, `\write18{...}` (shell-escape), `\openout`, `\catcode`, `\def`, `\let`, `\usepackage`, etc. from reaching the LaTeX compiler, the sanitizer raises `litchron.sanitize.UnsafeTexError` (code `unsafe_tex`) when any blocked control sequence appears outside a ` ``` ` fence. Math via `$...$` and `$$...$$` is allowed (that is the purpose of the `tex_math_dollars` extension), and arbitrary raw TeX inside fenced code blocks passes through verbatim since pandoc treats it as prose. The full block list lives in the module docstring of `litchron/sanitize.py`.
3. `pandoc --from gfm+tex_math_dollars --to latex+raw_tex` converts each section to a `.tex` fragment under `runs/<run_id>/tex_sections/`.
4. `latexmk -pdf -interaction=nonstopmode -halt-on-error` compiles `tex/litchron.tex`, which `\input`s each fragment. The `\runDir` macro is injected on the command line (`-usepretex='\def\runDir{runs/<run_id>}'`), keeping the template path-agnostic.

## Preflight Gate

`litchron/preflight.py:check_environment()` runs at MCP server startup and validates: `pandoc` in PATH, `latexmk` in PATH, `Rscript` in PATH (when R baselines requested), R version ≥ 4.1, `mcp` importable, `scanpy` importable. Missing critical deps cause a structured `PreflightFailure` abort with an actionable error message.
