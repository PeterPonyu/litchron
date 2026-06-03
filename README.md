# LitChron

LitChron is a single-cell chronology system where a terminal LLM agent (Claude Code) acts as a biological oracle: given any AnnData `.h5ad`, the LLM proposes a per-cell-type pseudotime ordering grounded in cited biological publications (DOIs/PMIDs verified against CrossRef and PubMed), while classical trajectory inference methods run as numerical comparators. The system is exposed as a stdio MCP server driven by Claude Code; the in-session loop terminates when `report_status().all_green == true`. The final deliverable per run is a formally structured LaTeX PDF assembled from intermediate markdown artifacts.

## Trajectory baselines: implementation status

| Method        | Backend         | Status                                                                                                                                                                                                              |
|---------------|-----------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| PAGA          | Python (scanpy) | Implemented and runnable.                                                                                                                                                                                           |
| Palantir      | Python          | Implemented and runnable.                                                                                                                                                                                           |
| scVelo        | Python          | Implemented and runnable (requires `Ms`/`Mu` or `spliced`/`unspliced` layers).                                                                                                                                      |
| pyslingshot   | Python          | Implemented and runnable.                                                                                                                                                                                           |
| **Monocle3**    | R (subprocess) | **Stub-only / not yet implemented.** The real rpy2 path in `litchron/baselines/_r_runner.py` returns the `monocle3_not_implemented` error. A deterministic linear-pseudotime stub is available behind `LITCHRON_STUB_R=1` for exercising the subprocess plumbing without a working R install. Tracking: [issue #5](https://github.com/PeterPonyu/litchron/issues/5). |
| **Slingshot-R** | R (subprocess) | **Stub-only / not yet implemented.** The real rpy2 path returns the `slingshot_r_not_implemented` error. Same `LITCHRON_STUB_R=1` stub as Monocle3. Tracking: [issue #5](https://github.com/PeterPonyu/litchron/issues/5).                                                                  |

## Setup

### 1. System dependencies

```bash
sudo apt install pandoc latexmk texlive-latex-extra texlive-bibtex-extra biber
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
claude mcp add litchron -- conda run -n dl python -m mcp_litchron.server
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
