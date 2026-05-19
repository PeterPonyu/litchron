# LitChron

LitChron is a single-cell chronology system where a terminal LLM agent (Claude Code) acts as a biological oracle: given any AnnData `.h5ad`, the LLM proposes a per-cell-type pseudotime ordering grounded in cited biological publications (DOIs/PMIDs verified against CrossRef and PubMed), while classical trajectory inference methods (Monocle3, Slingshot, PAGA, Palantir, scVelo) run as numerical comparators. The system is exposed as a stdio MCP server driven by Claude Code; the in-session loop terminates when `report_status().all_green == true`. The final deliverable per run is a formally structured LaTeX PDF assembled from intermediate markdown artifacts.

## Setup

### 1. System dependencies

```bash
sudo apt install pandoc latexmk texlive-latex-extra
```

### 2. Conda environment

The project reuses the existing `dl` env. Install additional deps:

```bash
conda env update -f environment.yml
pip install -e .
```

Or, to create a fresh env from scratch using the `dl` base:

```bash
conda env create -f environment.yml
conda activate dl
pip install -e .
```

### 3. Register the MCP server with Claude Code

```bash
claude mcp add litchron --transport stdio --command 'conda run -n dl python -m mcp_litchron.server'
```

Verify registration:

```bash
claude mcp list
```

## Running a single h5ad

Paste the following prompt into a Claude Code session:

```
claude -p "Drive LitChron to completion on /path/to/data.h5ad. Stop when report_status().all_green is true."
```

Claude Code will call MCP tools in sequence — `start_run` → `load_h5ad` → `compute_observations` → `propose_ordering` (with citations) → `verify_doi`/`verify_pmid` → `run_baseline` (per method) → `compare_orderings` → `append_section` → `compile_pdf` → `finalize_run` — and stop autonomously when every phase is verified complete.

The compiled report appears at `runs/<run_id>/report.pdf`.

## Docs

- `docs/architecture.md` — hybrid execution model, AnnDataCache, state protocol
- `docs/mcp-tool-reference.md` — full tool signatures and when-to-call guidance
- `docs/citation-verification.md` — three-signal verifier and reason codes
- `docs/llm-schema.md` — YAML+markdown schema for observations and proposals
