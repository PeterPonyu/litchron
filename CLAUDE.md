# LitChron — Operator Contract for Claude Code

## Purpose

LitChron turns Claude Code into a biological oracle for single-cell chronology. Given an AnnData `.h5ad`, you propose a per-cell-type pseudotime ordering grounded in cited biological publications, verify every citation against CrossRef and PubMed, run classical trajectory inference methods as numerical comparators, and compile the results into a LaTeX PDF. You are the oracle; the MCP server is the verifier and persister.

## Tool Sequence Pattern

For each run, execute tools in this order. Steps marked OPTIONAL do not block `all_green`.

```
start_run(h5ad_path, run_id?)
  → load_h5ad(run_id)
  → compute_observations(run_id)
  → propose_ordering(run_id, proposal)          # includes citations list
  → verify_doi(doi, context, year?, authors?)   # one call per DOI
  → verify_pmid(pmid, context, year?, authors?) # one call per PMID
  → run_baseline(run_id, method)                # OPTIONAL: repeat for each method
  → compare_orderings(run_id)                   # OPTIONAL: only if baselines ran
  → append_section(run_id, section, markdown)   # once per section
  → compile_pdf(run_id)
  → finalize_run(run_id)
```

Check `report_status(run_id)` after each phase-completing tool to know what remains. Stop only when `report_status(run_id).all_green == true`.

## Citation Discipline

- Every DOI or PMID you emit in `propose_ordering` must be verified before `finalize_run`.
- If `verify_doi` or `verify_pmid` returns a `CitationVerdict` with `status != "verified"`, that citation is quarantined. Its `reason` code is recorded in `state.citations_dropped`.
- **Never re-emit a citation that appears in `state.citations_dropped`** — the reason code is a hard constraint, not a suggestion. If the reason is `cosine_below_threshold`, the paper is not relevant to your claim; find a different citation. If the reason is `year_mismatch`, the year you stated is wrong; either correct it or drop the paper. If the reason is `crossref_404` or `pubmed_404`, the identifier does not exist; do not retry the same ID.
- If all citations for a claim are dropped, either locate a verifiable replacement or remove the claim from the narrative. An unverified claim must not appear in the final report.
- Aim for at least one verified citation per major ordering decision (e.g., why cell type A precedes B).

## Advisory Tools (`suggested_next_tools`)

`report_status` returns a `suggested_next_tools` list. These are **advisory only**:

- Read the `rationale` field before deciding whether to follow the suggestion.
- You may reorder suggestions, combine steps, or ignore them entirely if your understanding of the biological context justifies it.
- An empty `suggested_next_tools` means `all_green == true` or no obvious next step — it does not mean you are blocked. Use your judgment.
- The server does not enforce suggestion order. You retain full oracle authority over the ordering proposal and the narrative.

## Stop Condition

The run is complete when:

```python
report_status(run_id).all_green == true
```

`all_green` requires: `llm_ordering_done`, `citations_verified` non-empty, and `latex_compiled`. Classical baselines (`baselines_all_done`, `comparison_done`) are **not** required — they are optional comparators. Do not call `finalize_run` before `all_green` is reachable; check `quality_flags` for any blocking issues.

## Resume After Crash

If the session is interrupted, restart with:

```
load_h5ad(run_id)   # restores cache from disk + replays zarr deltas
report_status(run_id)  # shows which phases are already done
```

Completed phases (baselines with persisted `ordering.parquet`, verified citations in `state.json`) do not need to be re-run. The server detects idempotent re-calls and returns the cached result.

## Quality Flags

If `report_status` returns non-empty `quality_flags`, address them before `finalize_run`:

| Flag | Blocking? | Action |
|---|---|---|
| `no_verified_citations` | yes | Provide at least one verifiable DOI/PMID |
| `baseline_disagreement_severe` | no | Acknowledge and discuss disagreement in the comparison narrative |
| `root_cell_ambiguous` | no | State your biological rationale for the chosen root cell type |
| `baseline_failure` | no | Informational only — note which method failed if you discuss baselines |
| `preflight_partial` | no | One or more system deps missing; some baselines may be unavailable |
