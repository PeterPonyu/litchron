# LLM Schema Reference

This document describes the data shapes that Claude Code reads from and writes to during a LitChron run. All LLM-authored content is plain markdown (with optional YAML front matter). Claude never reads or writes TeX source directly.

---

## `observations.md` — Output of `compute_observations`

Written by `litchron/observations.py`. Structure:

```markdown
---
run_id: 2026-05-18T123456-abc12345
h5ad_path: /data/pbmc3k.h5ad
modality: scrna
n_cells: 2638
n_genes: 1838
generated_at: 2026-05-18T12:34:56Z
---

## Dataset Summary

| Field | Value |
|---|---|
| Modality | scRNA-seq |
| Cells | 2,638 |
| Genes | 1,838 |
| Clusters | 8 |

## Clusters and Markers

| Cluster | N cells | Top markers |
|---|---|---|
| 0 | 480 | CD3D, CD3E, IL7R |
| 1 | 340 | CD14, LYZ, CST3 |
| ... | ... | ... |

## Layer Summary

| Layer | Present | Shape |
|---|---|---|
| X (counts) | yes | (2638, 1838) |
| spliced | no | — |
| unspliced | no | — |
```

The YAML front matter is machine-readable. Claude should read the cluster table and marker list before calling `propose_ordering`.

---

## `OrderingProposal` — Input to `propose_ordering`

Pydantic model in `mcp_litchron/tools.py`:

```python
class CellTypeRankEntry(BaseModel):
    cell_type: str          # must match a cluster label from observations.md
    rank: int               # 1 = earliest in pseudotime; ties allowed
    confidence: float | None = None   # 0.0–1.0; omit if uncertain
    tied_with: list[str] | None = None  # other cell types at the same rank

class CitationInput(BaseModel):
    doi: str | None = None
    pmid: str | None = None
    title: str | None = None   # used for verification hinting only
    year: int | None = None
    authors: list[str] | None = None  # surnames or full names

class OrderingProposal(BaseModel):
    per_cell_type_rank: list[CellTypeRankEntry]
    narrative_md: str    # markdown paragraph(s) explaining the biological rationale
    citations: list[CitationInput]
```

**Constraints**:
- `per_cell_type_rank` must cover every cluster reported in `observations.md`. Missing clusters cause `LitchronError(code="incomplete_ordering")`.
- Ties are expressed by assigning the same `rank` to multiple entries and listing each in the other's `tied_with`.
- At least one `doi` or `pmid` must be present in `citations`. An empty citations list causes `LitchronError(code="no_citations")`.

**Example payload** (abridged):

```json
{
  "per_cell_type_rank": [
    {"cell_type": "HSC", "rank": 1, "confidence": 0.9, "tied_with": null},
    {"cell_type": "CMP", "rank": 2, "confidence": 0.8, "tied_with": null},
    {"cell_type": "GMP", "rank": 3, "confidence": 0.7, "tied_with": ["MEP"]},
    {"cell_type": "MEP", "rank": 3, "confidence": 0.7, "tied_with": ["GMP"]}
  ],
  "narrative_md": "HSCs are multipotent progenitors at the apex of the haematopoietic hierarchy...",
  "citations": [
    {"doi": "10.1038/nature14229", "year": 2015, "authors": ["Trapnell"]},
    {"pmid": "29700258", "year": 2018}
  ]
}
```

---

## `SuggestedTool` — Advisory Shape in `report_status`

Returned inside `StatusResult.suggested_next_tools`. Advisory only — the LLM may follow, reorder, or ignore.

```python
class SuggestedTool(BaseModel):
    tool: str        # name of the MCP tool to call
    args: dict       # suggested arguments (may be partial)
    rationale: str   # one sentence explaining why this tool is suggested next
```

**Example**:

```json
{
  "tool": "verify_doi",
  "args": {"doi": "10.1038/nature14229", "context": "HSC multipotency"},
  "rationale": "Citation proposed but not yet verified; must verify before finalize_run"
}
```

When `all_green == true`, `suggested_next_tools` is an empty list.

---

## `proposals.md` — Persisted by `propose_ordering`

Written to `runs/<run_id>/proposals.md`:

```markdown
---
proposal_version: 1
proposed_at: 2026-05-18T12:40:00Z
n_cell_types: 8
n_citations_staged: 3
---

## Proposed Ordering

| Cell Type | Rank | Confidence | Tied With |
|---|---|---|---|
| HSC | 1 | 0.90 | — |
| CMP | 2 | 0.80 | — |
| GMP | 3 | 0.70 | MEP |
| MEP | 3 | 0.70 | GMP |

## Biological Rationale

HSCs are multipotent progenitors at the apex of the haematopoietic hierarchy...

## Citations (staged, pending verification)

- DOI: 10.1038/nature14229 (Trapnell, 2015)
- PMID: 29700258 (2018)
```

---

## `comparison.md` — Output of `compare_orderings`

Written to `runs/<run_id>/comparison.md`:

```markdown
---
compared_at: 2026-05-18T13:10:00Z
methods_compared: ["paga", "palantir", "monocle3"]
---

## Ordering Comparison

| Method | Spearman | Kendall τ | Edge Jaccard | Root Cell Match |
|---|---|---|---|---|
| PAGA | 0.82 | 0.71 | 0.67 | yes |
| Palantir | 0.78 | 0.66 | — | yes |
| Monocle3 | 0.90 | 0.83 | 0.72 | yes |

## Interpretation

The LLM ordering shows strong correlation with all three baselines...
```
