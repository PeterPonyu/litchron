# MCP Tool Reference

All tools are exposed by `mcp_litchron/server.py` over stdio. Every tool is idempotent unless noted. All arguments are validated by Pydantic before execution.

---

## `start_run`

```python
start_run(
    h5ad_path: str,
    run_id: str | None = None,
    force: bool = False,
) -> StartRunResult
```

**Returns**: `{run_id: str, run_dir: str}`

Validates that `h5ad_path` exists before creating the run directory. Generates a default `run_id` of the form `2026-05-18T123456-abc12345` (hyphens only, no underscores/dots/whitespace) if not supplied. Custom `run_id` must match `^[A-Za-z0-9-]+$`; invalid IDs are rejected with `LitchronError(code="invalid_run_id")`. Refuses a non-empty run directory unless `force=True`.

**When to call**: Once, at the start of every new run. Creates `state.json` and empty artifact stubs.

---

## `load_h5ad`

```python
load_h5ad(run_id: str) -> LoadResult
```

**Returns**: `{n_cells: int, n_genes: int, modality: str, layers: list[str], cache_hit: bool}`

Loads the h5ad into `AnnDataCache` (or returns cached entry). On cache miss after restart, replays zarr deltas from `state.adata_deltas`.

**When to call**: After `start_run`, before `compute_observations`. Also call after session restart to restore the cache.

---

## `compute_observations`

```python
compute_observations(run_id: str) -> ObservationsResult
```

**Returns**: `{observations_md_path: str, n_clusters: int, modality: str}`

Runs clustering, marker detection, and centroid summary. Writes `runs/<run_id>/observations.md` in YAML front-matter + markdown table format (see `docs/llm-schema.md`). Idempotent: re-running overwrites `observations.md` but does not change state phase if already completed.

**When to call**: After `load_h5ad`. Read `observations.md` before calling `propose_ordering`.

---

## `propose_ordering`

```python
propose_ordering(
    run_id: str,
    proposal: OrderingProposal,
) -> ProposalResult
```

**`OrderingProposal`**:
```python
{
    per_cell_type_rank: list[CellTypeRankEntry],  # see llm-schema.md
    narrative_md: str,                            # biological rationale markdown
    citations: list[CitationInput],               # {doi?, pmid?, title?, year?, authors?}
}
```

**Returns**: `{persisted: bool, proposal_md_path: str, citation_count: int}`

Persists the proposal to `runs/<run_id>/proposals.md`. Citations are staged but not yet verified; call `verify_doi`/`verify_pmid` next for each citation.

**When to call**: After reading `observations.md`. May be called again to revise the proposal (overwrites previous proposal, resets citation verification state).

---

## `verify_doi`

```python
verify_doi(
    doi: str,
    context: str,
    year: int | None = None,
    authors: list[str] | None = None,
) -> CitationVerdict
```

**Returns**: `{status: "verified"|"dropped", reason?: str, confidence?: float, bibtex_key?: str}`

Calls CrossRef polite pool. Runs three-signal verification (cosine ≥ 0.6, year ±1, author overlap). On success, writes entry to `references.bib` and appends to `state.citations_verified`. On failure, appends to `state.citations_dropped` with reason code.

**When to call**: Once per DOI in `proposal.citations`. Never retry a DOI that returned a non-`network_error` reason code.

---

## `verify_pmid`

```python
verify_pmid(
    pmid: str,
    context: str,
    year: int | None = None,
    authors: list[str] | None = None,
) -> CitationVerdict
```

**Returns**: Same shape as `verify_doi`.

Calls NCBI E-utilities `esummary.fcgi`. Same three-signal verification as `verify_doi`.

**When to call**: Once per PMID in `proposal.citations`. Same retry discipline as `verify_doi`.

---

## `run_baseline`

```python
run_baseline(
    run_id: str,
    method: BaselineName,
) -> BaselineResult | BaselineFailure
```

**`BaselineName`**: `"paga" | "palantir" | "scvelo" | "pyslingshot" | "monocle3" | "slingshot_r"`

**`BaselineResult`**: `{method: str, ordering_path: str, figure_path: str, lineage_edges?: list, root_cell?: str}`

**`BaselineFailure`**: `{method: str, reason: str, detail: str, retryable: bool}`

Dispatches: R-backed methods (`monocle3`, `slingshot_r`) → subprocess via `_r_runner`; Python-backed methods → in-process. Persists results to `runs/<run_id>/baselines/<method>/`. Idempotent: re-calling a completed baseline returns the persisted result without recomputing.

**When to call**: After `propose_ordering`. Call once per desired method. Check `available_baselines` (returned in `compute_observations`) to know which methods are eligible for the loaded modality.

---

## `compare_orderings`

```python
compare_orderings(run_id: str) -> ComparisonReport
```

**Returns**: `{comparison_md_path: str, rows: list[ComparisonRow]}`

Computes rank correlation (Spearman, Kendall tau), lineage edge Jaccard, and root-cell agreement between the LLM ordering and each completed baseline. Writes `runs/<run_id>/comparison.md`.

**When to call**: After all desired `run_baseline` calls are complete.

---

## `append_section`

```python
append_section(
    run_id: str,
    section: SectionName,
    markdown: str,
) -> AppendResult
```

**`SectionName`**: `"observations" | "proposals" | "baselines" | "comparison"`

**Returns**: `{section_path: str, appended_bytes: int}`

Appends markdown to the named section file under `runs/<run_id>/tex_sections/`. The section file is later converted to TeX by `compile_pdf`.

**When to call**: After the relevant analysis phase is complete. May be called multiple times to append additional paragraphs.

---

## `report_status`

```python
report_status(run_id: str) -> StatusResult
```

**Returns**:
```python
{
    state: RunState,
    all_green: bool,
    suggested_next_tools: list[SuggestedTool],  # advisory only
    quality_flags: list[QualityFlag],
}
```

Returns the current run state without side effects. `suggested_next_tools` is advisory — the LLM retains oracle authority and may follow, reorder, or ignore suggestions. When `all_green == true`, `suggested_next_tools` is empty.

**When to call**: After any phase-completing tool call, to check what remains. Also the terminal condition check: stop when `all_green == true`.

---

## `compile_pdf`

```python
compile_pdf(run_id: str) -> CompileResult
```

**Returns**: `{pdf_path: str, pdf_size_bytes: int, latex_log_path: str}`

Runs: sanitize markdown → pandoc per section → latexmk on `tex/litchron.tex` with `\runDir` macro. Output at `runs/<run_id>/report.pdf`. Fails loud (`-halt-on-error`) with log path on LaTeX error.

**When to call**: After `compare_orderings` and all `append_section` calls are done.

---

## `finalize_run`

```python
finalize_run(run_id: str) -> FinalizeResult
```

**Returns**: `{finished_at: str, all_green: bool, quality_flags: list[QualityFlag]}`

Evicts the run's AnnData from cache, writes `finished_at` to `state.json`, and sets `all_green` once all verifier checks pass. Not idempotent in the sense that it locks the run; subsequent tool calls on a finalized run return `LitchronError(code="run_finalized")`.

**When to call**: Last tool in the sequence, after `compile_pdf` and `report_status().all_green == true`.
