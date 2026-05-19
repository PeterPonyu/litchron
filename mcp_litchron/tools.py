"""Pydantic-typed MCP tool surface for LitChron.

Every callable here is a thin wrapper around the analysis layer
(:mod:`litchron`) and the cache (:mod:`mcp_litchron.cache`). Three
invariants hold across all tools:

1. **Inputs are Pydantic-validated** at the call boundary so malformed
   payloads from the LLM are rejected with a structured
   :class:`mcp_litchron.errors.ErrorResult` instead of an arbitrary
   exception trace.
2. **Generic exceptions never escape** — every tool wraps its body in
   ``try / except Exception`` and converts unstructured failures into
   :class:`ErrorResult` so the MCP transport layer always sees JSON.
3. **State updates go through** :meth:`RunStateStore.update`, which
   takes an exclusive ``fcntl.flock`` for the whole read-modify-write
   window. Concurrent tool calls on the same ``run_id`` are serialized
   on ``runs/<run_id>/state.lock``.

The module-level :data:`_CACHE` and :data:`_VERIFIER` instances are
shared by every tool invocation. They are intentionally NOT request-
scoped: the LRU cache and the embed model are expensive to set up, and
the MCP server is a single-process stdio loop. Tests can monkeypatch
either binding to swap in stub implementations.

R-bridged baselines are dispatched via :func:`litchron.baselines.run`,
which **lazy-imports** the R-specific shim. ``rpy2`` is never imported
at module import time in this file (or anywhere reachable from the MCP
server entrypoint).
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, ValidationError

from litchron._runtime import (
    generate_run_id,
    project_root,
    run_dir,
    validate_run_id,
)
from litchron.citations import (
    Citation,
    CitationInput,
    CitationVerdict,
    CitationVerifier,
)
from litchron.compare import (
    ComparisonRow,
    comparison_to_markdown,
)
from litchron.compare import (
    compare as _compare_pair,
)
from litchron.embeddings import (
    recompute_embeddings as _recompute_embeddings_pure,
)
from litchron.figures import (
    make_litchron_annotation_figure as _make_annotation_figure_pure,
)
from litchron.io import detect_modality
from litchron.observations import (
    compute_observations as _compute_obs_pure,
)
from litchron.observations import (
    observations_to_markdown,
)
from litchron.report import compile_pdf as _compile_pdf_impl
from litchron.state import (
    CitationFingerprint,
    DeltaRef,
    DroppedCitation,
    RunState,
    RunStateStore,
    SuggestedTool,
    default_state,
)
from mcp_litchron.cache import AnnDataCache
from mcp_litchron.errors import (
    BaselineFailure,
    ErrorResult,
    LitchronError,
)

# ---------------------------------------------------------------------------
# Module-level singletons
# ---------------------------------------------------------------------------
_CACHE: AnnDataCache = AnnDataCache(max_entries=4)
"""Shared AnnData LRU cache. Tests may replace this binding."""

# The verifier is created lazily on first access so module import does
# not eagerly construct an :class:`httpx.Client` (which reads proxy env
# vars and would explode at import time on hosts with a non-HTTP proxy
# scheme set in the user environment). Tests can override by assigning
# to ``mcp_litchron.tools._VERIFIER`` directly.
_VERIFIER: Optional[CitationVerifier] = None


def _get_verifier() -> CitationVerifier:
    """Return the process-wide :class:`CitationVerifier` (lazy init)."""
    global _VERIFIER
    if _VERIFIER is None:
        _VERIFIER = CitationVerifier()
    return _VERIFIER


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _now_iso() -> str:
    """UTC ISO-8601 timestamp used in DroppedCitation / CitationFingerprint."""
    return datetime.now(timezone.utc).isoformat()


def _store_for(run_id: str) -> RunStateStore:
    """Return the :class:`RunStateStore` for ``run_id``'s run directory."""
    return RunStateStore(run_dir(run_id))


def _read_state(run_id: str) -> RunState:
    """Locked read of ``runs/<run_id>/state.json``."""
    return _store_for(run_id).read()


def _recompute_all_green(state: RunState) -> bool:
    """Single source of truth for the ``all_green`` flag.

    A run is "green" iff every *mandatory* gate has been satisfied:

    * the LLM proposed an ordering,
    * the LitChron LLM pseudotime was computed (the primary output),
    * LaTeX compiled,
    * at least one citation was verified.

    Classical-baseline runs (PAGA, scVelo) are sanity-check comparators,
    not requirements: ``baselines_all_done`` is *advisory only*. If the
    operator skipped them, the comparison step also becomes optional —
    the headline figure + LitChron LLM pseudotime is the primary product.

    The advisory ``suggested_next_tools`` list is recomputed against this
    same flag in :func:`report_status`.
    """
    has_llm_pt = any(d.baseline == "litchron_pseudotime" for d in state.adata_deltas)
    # When baselines did run, we still require comparison + latex; when they
    # were skipped, comparison_done is N/A and we don't demand it.
    baselines_ran = bool(state.baselines_done)
    comparison_ok = state.comparison_done if baselines_ran else True
    return bool(
        state.llm_ordering_done
        and has_llm_pt
        and state.latex_compiled
        and comparison_ok
        and len(state.citations_verified) > 0
    )


def _error_from_exception(exc: Exception) -> ErrorResult:
    """Convert any exception into a JSON-serializable :class:`ErrorResult`.

    Known :class:`LitchronError` subclasses carry their own ``code`` /
    ``hint`` / ``retryable`` fields; generic exceptions are wrapped under
    code ``"internal_error"`` with ``retryable=False``.
    """
    if isinstance(exc, LitchronError):
        return exc.to_result()
    return ErrorResult(
        code="internal_error",
        message=f"{type(exc).__name__}: {exc}",
        hint="Check the MCP server logs; this is likely a bug in LitChron.",
        retryable=False,
    )


# ---------------------------------------------------------------------------
# Pydantic input / output models
# ---------------------------------------------------------------------------
class StartRunResult(BaseModel):
    """Return type of :func:`start_run`."""

    run_id: str
    run_dir: str
    created: bool


class LoadResult(BaseModel):
    """Return type of :func:`load_h5ad`."""

    n_cells: int
    n_genes: int
    modality: str
    layers: list[str]
    cache_hit: bool


class ObservationsResult(BaseModel):
    """Return type of :func:`compute_observations`."""

    path: str
    n_clusters: int


class RecomputeEmbeddingsResult(BaseModel):
    """Return type of :func:`recompute_embeddings`."""

    path_pca: str
    path_umap: str
    n_clusters: int
    leiden_resolution: float


class AnnotationFigureResult(BaseModel):
    """Return type of :func:`make_annotation_figure`."""

    figure_path: str
    size_bytes: int


class CellTypeRankEntry(BaseModel):
    """One row of the per-cell-type pseudotime ordering.

    The ``tied_with`` list expresses biological ties (cell types the LLM
    cannot confidently rank against each other at the proposed
    position); ``confidence`` is an LLM-supplied scalar in [0, 1] when
    available.
    """

    cell_type: str
    rank: int
    confidence: Optional[float] = None
    tied_with: Optional[list[str]] = None
    cell_type_label: Optional[str] = None  # biological name e.g. "LT-HSC" (display only)


class OrderingProposal(BaseModel):
    """LLM-authored proposal: ranking + narrative + citations.

    The full proposal is persisted verbatim to ``proposals.md`` with
    YAML front-matter so downstream readers (humans and the comparison
    step) can re-parse it deterministically.
    """

    per_cell_type_rank: list[CellTypeRankEntry]
    narrative_md: str
    citations: list[CitationInput] = Field(default_factory=list)


class ProposalResult(BaseModel):
    """Return type of :func:`propose_ordering`."""

    path: str
    n_cell_types: int
    n_citations: int


class AppendResult(BaseModel):
    """Return type of :func:`append_section`."""

    path: str
    bytes_appended: int


SectionName = Literal["observations", "proposals", "baselines", "comparison"]


class StatusResult(BaseModel):
    """Return type of :func:`report_status`.

    ``suggested_next_tools`` is **advisory** — populated only when
    ``all_green == False`` and emptied when the run is complete. The
    LLM retains oracle authority and may ignore, reorder, or augment
    the suggestions.
    """

    state: RunState
    all_green: bool
    suggested_next_tools: list[SuggestedTool]
    quality_flags: list[str]


BaselineName = Literal[
    "monocle3",
    "slingshot_r",
    "paga",
    "palantir",
    "scvelo",
    "pyslingshot",
]


# ---------------------------------------------------------------------------
# Tool: start_run
# ---------------------------------------------------------------------------
def start_run(
    h5ad_path: str,
    run_id: Optional[str] = None,
    force: bool = False,
) -> StartRunResult | ErrorResult:
    """Create a fresh run directory and seed ``state.json``.

    When to call: once per dataset, before any other tool.
    Idempotent: no (unless ``force=True`` is passed).
    Expected next tool: ``load_h5ad``.
    """
    try:
        # 1) Resolve and validate run_id (auto-generate if omitted).
        rid = run_id if run_id is not None else generate_run_id()
        validate_run_id(rid)

        # 2) Validate the h5ad path *before* any side effects.
        src = Path(h5ad_path)
        if not src.is_file():
            raise LitchronError(
                code="h5ad_not_a_file",
                message=f"h5ad path is not a regular file: {h5ad_path}",
                hint="Provide an absolute path to an existing .h5ad regular file (not a directory or broken symlink).",
                retryable=False,
            )

        # 3) Inspect the target directory and enforce force= semantics.
        target = run_dir(rid)
        existing = target.exists() and any(target.iterdir()) if target.exists() else False
        if existing and not force:
            raise LitchronError(
                code="run_dir_non_empty",
                message=(
                    f"run directory {target!s} is non-empty; refusing to "
                    "overwrite without force=True"
                ),
                hint=(
                    "Pass force=True to start over, or choose a different "
                    "run_id."
                ),
                retryable=False,
            )

        # 4) Create the skeleton directories.
        target.mkdir(parents=True, exist_ok=True)
        (target / "baselines").mkdir(exist_ok=True)
        (target / "logs").mkdir(exist_ok=True)
        (target / "tex_sections").mkdir(exist_ok=True)

        # 5) Seed state.json via the locked update path.
        store = RunStateStore(target)
        store.update(fn=lambda _: default_state(run_id=rid, h5ad_path=str(src.resolve())))

        return StartRunResult(
            run_id=rid,
            run_dir=str(target),
            created=True,
        )
    except ValidationError as e:
        return ErrorResult(
            code="invalid_input",
            message=str(e),
            hint="Check the start_run input schema.",
            retryable=False,
        )
    except Exception as e:  # noqa: BLE001  (broad catch is the contract)
        return _error_from_exception(e)


# ---------------------------------------------------------------------------
# Tool: load_h5ad
# ---------------------------------------------------------------------------
def load_h5ad(run_id: str) -> LoadResult | ErrorResult:
    """Load (or restore from cache) the AnnData attached to a run.

    When to call: after ``start_run`` or to warm the cache after a
    server restart. Replays any zarr deltas recorded on the run state.
    Idempotent: yes (cache-hits return immediately).
    Expected next tool: ``compute_observations``.
    """
    try:
        validate_run_id(run_id)
        state = _read_state(run_id)
        target = run_dir(run_id)

        # Cache-hit detection runs against the same key the cache uses;
        # the fingerprint inside the cache decides hit vs. miss.
        cache_hit_before = run_id in _CACHE

        # If deltas are recorded, use resume() so they're replayed.
        if state.adata_deltas:
            adata = _CACHE.resume(
                run_id=run_id,
                h5ad_path=state.h5ad_path,
                run_dir=target,
                deltas=state.adata_deltas,
            )
            # resume() always drops the in-memory entry first, so this
            # is by definition not a cache hit.
            cache_hit = False
        else:
            adata = _CACHE.load(
                run_id=run_id,
                h5ad_path=state.h5ad_path,
                run_dir=target,
            )
            cache_hit = cache_hit_before

        layers = sorted(map(str, (getattr(adata, "layers", None) or {}).keys()))
        return LoadResult(
            n_cells=int(adata.n_obs),
            n_genes=int(adata.n_vars),
            modality=detect_modality(adata),
            layers=layers,
            cache_hit=cache_hit,
        )
    except Exception as e:  # noqa: BLE001
        return _error_from_exception(e)


# ---------------------------------------------------------------------------
# Tool: compute_observations
# ---------------------------------------------------------------------------
def compute_observations(run_id: str) -> ObservationsResult | ErrorResult:
    """Compute clusters + marker genes and write ``observations.md``.

    When to call: after ``load_h5ad`` and before ``propose_ordering``.
    Idempotent: re-running overwrites ``observations.md`` with fresh content.
    Expected next tool: ``propose_ordering``.
    """
    try:
        validate_run_id(run_id)
        state = _read_state(run_id)
        target = run_dir(run_id)

        with _CACHE.with_adata(
            run_id=run_id,
            h5ad_path=state.h5ad_path,
            run_dir=target,
        ) as adata:
            obs = _compute_obs_pure(adata)

        md = observations_to_markdown(obs)
        obs_path = target / "observations.md"
        obs_path.write_text(md)

        return ObservationsResult(
            path=str(obs_path),
            n_clusters=len(obs.clusters),
        )
    except Exception as e:  # noqa: BLE001
        return _error_from_exception(e)


# ---------------------------------------------------------------------------
# Tool: recompute_embeddings
# ---------------------------------------------------------------------------
def _write_embeddings_delta(
    run_dir: Path,
    adata: Any,
) -> tuple[Path, list[str]]:
    """Persist a zarr delta capturing recomputed PCA/UMAP/leiden.

    Side effect: writes ``<run_dir>/embeddings/adata_delta.zarr`` with
    the three arrays (``obsm/X_pca``, ``obsm/X_umap``, ``obs/leiden``).
    Returns the delta path and the list of keys it contains.
    """
    import numpy as np  # local — keeps tools.py import cheap
    import zarr  # heavy import, kept local

    embeddings_dir = run_dir / "embeddings"
    embeddings_dir.mkdir(parents=True, exist_ok=True)
    delta_path = embeddings_dir / "adata_delta.zarr"

    root = zarr.open(str(delta_path), mode="w")
    keys: list[str] = []
    obsm = getattr(adata, "obsm", None) or {}
    # zarr 3.x removed Group.array(name, ndarray, ...) positional; the
    # canonical replacement is create_array(name=, data=, overwrite=).
    # See line 1623 in this file for the same pattern.
    if "X_pca" in obsm:
        root.create_array(
            name="obsm/X_pca", data=np.asarray(obsm["X_pca"]), overwrite=True,
        )
        keys.append("obsm/X_pca")
    if "X_umap" in obsm:
        root.create_array(
            name="obsm/X_umap", data=np.asarray(obsm["X_umap"]), overwrite=True,
        )
        keys.append("obsm/X_umap")
    if "leiden" in adata.obs.columns:
        leiden_vals = np.asarray(adata.obs["leiden"].astype(str).values, dtype="U")
        root.create_array(
            name="obs/leiden", data=leiden_vals, overwrite=True,
        )
        keys.append("obs/leiden")
    return delta_path, keys


def recompute_embeddings(
    run_id: str,
    force: bool = False,
    leiden_resolution: float = 1.0,
) -> dict[str, Any]:
    """Recompute UMAP + PCA + leiden on the run's AnnData and persist a delta.

    When to call: after ``load_h5ad`` if the incoming h5ad should be
    treated as a new dataset (no trusted pre-existing embedding). Safe
    to skip if the upstream embedding is already canonical.
    Idempotent: yes — re-running with ``force=False`` is a no-op when
    PCA / UMAP / leiden are already present.
    Expected next tool: ``compute_observations``.

    The recomputed arrays are written to
    ``<run_dir>/embeddings/adata_delta.zarr`` and the delta is recorded
    on ``state.adata_deltas`` so the AnnDataCache can replay it after a
    restart.
    """
    try:
        validate_run_id(run_id)
        state = _read_state(run_id)
        target = run_dir(run_id)

        with _CACHE.with_adata(
            run_id=run_id,
            h5ad_path=state.h5ad_path,
            run_dir=target,
        ) as adata:
            _recompute_embeddings_pure(
                adata,
                force=force,
                leiden_resolution=float(leiden_resolution),
            )
            delta_path, delta_keys = _write_embeddings_delta(target, adata)
            n_clusters = int(adata.obs["leiden"].astype(str).nunique())

        if delta_keys:
            _CACHE.apply_delta(
                run_id=run_id,
                baseline="recompute_embeddings",
                delta_keys=delta_keys,
                delta_zarr_path=delta_path,
            )

            def _record(s: RunState) -> RunState:
                s.adata_deltas.append(
                    DeltaRef(
                        baseline="recompute_embeddings",
                        keys=delta_keys,
                        zarr_path=str(delta_path),
                        applied_at=_now_iso(),
                    )
                )
                return s

            _store_for(run_id).update(_record)

        return RecomputeEmbeddingsResult(
            path_pca=str(delta_path),
            path_umap=str(delta_path),
            n_clusters=n_clusters,
            leiden_resolution=float(leiden_resolution),
        ).model_dump(mode="json")
    except Exception as e:  # noqa: BLE001
        return _error_from_exception(e).model_dump(mode="json")


# ---------------------------------------------------------------------------
# Tool: make_annotation_figure
# ---------------------------------------------------------------------------
def _read_proposal_maps(target: Path) -> tuple[dict[str, str], dict[str, int], dict[str, float]]:
    """Parse ``proposals.md`` front-matter into label / rank / confidence dicts.

    Returns ``(label_map, rank_map, confidence_map)`` keyed by cluster id
    (the ``cell_type`` field in the YAML). Empty dicts if the proposal
    file is missing — the figure tool degrades gracefully.
    """
    label_map: dict[str, str] = {}
    rank_map: dict[str, int] = {}
    confidence_map: dict[str, float] = {}

    path = target / "proposals.md"
    if not path.exists():
        return label_map, rank_map, confidence_map

    text = path.read_text()
    if not text.startswith("---"):
        return label_map, rank_map, confidence_map
    end = text.find("\n---", 3)
    if end < 0:
        return label_map, rank_map, confidence_map
    fm = text[3:end].splitlines()

    in_rank_block = False
    current: dict[str, Any] = {}

    def _commit() -> None:
        if "cell_type" in current and "rank" in current:
            cid = str(current["cell_type"])
            rank_map[cid] = int(current["rank"])
            if "cell_type_label" in current:
                label_map[cid] = str(current["cell_type_label"])
            else:
                label_map[cid] = cid
            if "confidence" in current:
                try:
                    confidence_map[cid] = float(current["confidence"])
                except (TypeError, ValueError):
                    pass

    for raw in fm:
        line = raw.rstrip()
        if line.strip() == "per_cell_type_rank:":
            in_rank_block = True
            continue
        if line.strip().startswith("citations:"):
            in_rank_block = False
            _commit()
            current = {}
            continue
        if not in_rank_block:
            continue
        if line.startswith("  - cell_type:"):
            _commit()
            current = {}
            current["cell_type"] = line.split(":", 1)[1].strip().strip("'\"")
        elif line.startswith("    rank:"):
            try:
                current["rank"] = int(line.split(":", 1)[1].strip())
            except ValueError:
                pass
        elif line.startswith("    confidence:"):
            try:
                current["confidence"] = float(line.split(":", 1)[1].strip())
            except ValueError:
                pass
        elif line.startswith("    cell_type_label:"):
            current["cell_type_label"] = (
                line.split(":", 1)[1].strip().strip("'\"")
            )
    _commit()

    return label_map, rank_map, confidence_map


def make_annotation_figure(run_id: str) -> dict[str, Any]:
    """Render the 4-panel LitChron annotation figure for the run.

    When to call: after ``propose_ordering`` (so a label_map exists) and
    after embeddings + observations are computed. The figure is written
    to ``<run_dir>/figures/litchron_annotation.png``.
    Idempotent: yes — the PNG is overwritten on each call.
    Expected next tool: ``append_section`` or ``compile_pdf``.

    Panel D (the marker-gene dotplot) is wrapped in a try/except inside
    the figure builder: a missing ``rank_genes_groups`` or other plotting
    failure renders a "panel unavailable: …" placeholder so the figure
    as a whole always completes.
    """
    try:
        validate_run_id(run_id)
        state = _read_state(run_id)
        target = run_dir(run_id)
        label_map, rank_map, confidence_map = _read_proposal_maps(target)

        with _CACHE.with_adata(
            run_id=run_id,
            h5ad_path=state.h5ad_path,
            run_dir=target,
        ) as adata:
            fig_path = _make_annotation_figure_pure(
                adata=adata,
                run_dir=target,
                label_map=label_map,
                rank_map=rank_map,
                confidence_map=confidence_map,
            )

        size = fig_path.stat().st_size if fig_path.exists() else 0
        return AnnotationFigureResult(
            figure_path=str(fig_path),
            size_bytes=int(size),
        ).model_dump(mode="json")
    except Exception as e:  # noqa: BLE001
        return _error_from_exception(e).model_dump(mode="json")


# ---------------------------------------------------------------------------
# Tool: propose_ordering
# ---------------------------------------------------------------------------
def _render_proposal_markdown(proposal: OrderingProposal) -> str:
    """Render an :class:`OrderingProposal` as YAML+markdown.

    Layout: YAML front-matter (per-cell-type ranking + citation refs)
    followed by the LLM's narrative and a citation table for human
    review. Persisted verbatim to ``proposals.md``.
    """
    lines: list[str] = ["---", "per_cell_type_rank:"]
    for entry in proposal.per_cell_type_rank:
        lines.append(f"  - cell_type: {entry.cell_type!r}")
        lines.append(f"    rank: {entry.rank}")
        if entry.confidence is not None:
            lines.append(f"    confidence: {entry.confidence}")
        if entry.cell_type_label is not None:
            lines.append(f"    cell_type_label: {entry.cell_type_label!r}")
        if entry.tied_with:
            lines.append("    tied_with:")
            for t in entry.tied_with:
                lines.append(f"      - {t!r}")
    if proposal.citations:
        lines.append("citations:")
        for c in proposal.citations:
            lines.append("  -")
            if c.doi:
                lines.append(f"    doi: {c.doi!r}")
            if c.pmid:
                lines.append(f"    pmid: {c.pmid!r}")
            if c.year_claimed is not None:
                lines.append(f"    year_claimed: {c.year_claimed}")
            if c.authors_claimed:
                lines.append("    authors_claimed:")
                for a in c.authors_claimed:
                    lines.append(f"      - {a!r}")
            # Context can be multi-line; YAML literal block.
            lines.append("    context: |")
            for ln in (c.context or "").splitlines() or [""]:
                lines.append(f"      {ln}")
    lines.append("---")
    lines.append("")
    lines.append("# LLM Proposal")
    lines.append("")
    lines.append(proposal.narrative_md or "")
    lines.append("")

    lines.append("## Proposed Per-Cell-Type Ordering")
    lines.append("")
    lines.append("| Rank | Cell Type | Cluster ID | Confidence | Tied With |")
    lines.append("|---:|:---|:---:|---:|:---|")
    for entry in proposal.per_cell_type_rank:
        label = entry.cell_type_label if entry.cell_type_label else entry.cell_type
        conf = f"{entry.confidence:.2f}" if entry.confidence is not None else "-"
        ties = ", ".join(entry.tied_with) if entry.tied_with else "-"
        lines.append(
            f"| {entry.rank} | {label} | {entry.cell_type} | {conf} | {ties} |"
        )
    lines.append("")

    # The verified-citation list is rendered ONCE at the document level via
    # biblatex's \printbibliography — we deliberately do NOT emit a per-section
    # citations table to keep the report deduplicated. The proposal's
    # narrative refers to citations by their meaning, not by an inline table.

    return "\n".join(lines)


def propose_ordering(
    run_id: str,
    proposal: OrderingProposal,
) -> ProposalResult | ErrorResult:
    """Persist the LLM's per-cell-type ordering + narrative + citations.

    When to call: once after ``compute_observations`` (or again after
    receiving ``dropped_citations`` feedback). The narrative_md is the
    LLM's chance to ground the ordering in cited biology.
    Idempotent: re-running overwrites ``proposals.md`` wholesale.
    Expected next tool: ``verify_doi`` / ``verify_pmid`` per citation.
    """
    try:
        validate_run_id(run_id)
        # Pydantic-validate the proposal even if the caller passed a dict.
        if not isinstance(proposal, OrderingProposal):
            proposal = OrderingProposal.model_validate(proposal)

        target = run_dir(run_id)
        target.mkdir(parents=True, exist_ok=True)
        path = target / "proposals.md"
        md = _render_proposal_markdown(proposal)
        path.write_text(md)

        def _set_done(s: RunState) -> RunState:
            s.llm_ordering_done = True
            return s

        _store_for(run_id).update(_set_done)

        return ProposalResult(
            path=str(path),
            n_cell_types=len(proposal.per_cell_type_rank),
            n_citations=len(proposal.citations),
        )
    except ValidationError as e:
        return ErrorResult(
            code="invalid_input",
            message=str(e),
            hint="Check the OrderingProposal schema.",
            retryable=False,
        )
    except Exception as e:  # noqa: BLE001
        return _error_from_exception(e)


# ---------------------------------------------------------------------------
# Tool: verify_doi / verify_pmid
# ---------------------------------------------------------------------------
def _persist_verdict(run_id: str, verdict: CitationVerdict, raw_id: str) -> None:
    """Record the verdict on ``state.json`` (verified or dropped)."""

    def _apply(s: RunState) -> RunState:
        if verdict.verified and verdict.citation is not None:
            fp = CitationFingerprint(
                scheme=verdict.citation.scheme,
                id=verdict.citation.id,
                verified_at=_now_iso(),
                source=verdict.citation.source,
                confidence=verdict.confidence,
            )
            # Deduplicate by (scheme, id) so repeat-calls are idempotent.
            existing = {(c.scheme, c.id) for c in s.citations_verified}
            if (fp.scheme, fp.id) not in existing:
                s.citations_verified.append(fp)
        else:
            reason = verdict.drop_reason or "network_error"
            # Map to the closed Literal of DroppedCitation.reason.
            allowed: set[str] = {
                "title_mismatch",
                "year_mismatch",
                "author_mismatch",
                "crossref_404",
                "pubmed_404",
                "network_error",
                "cosine_below_threshold",
            }
            if reason not in allowed:
                reason = "network_error"
            dc = DroppedCitation(
                raw_id=raw_id,
                reason=reason,  # type: ignore[arg-type]
                detail=str(verdict.signals)[:500],
                dropped_at=_now_iso(),
            )
            existing_drops = {(d.raw_id, d.reason) for d in s.citations_dropped}
            if (dc.raw_id, dc.reason) not in existing_drops:
                s.citations_dropped.append(dc)
        return s

    _store_for(run_id).update(_apply)


def verify_doi(
    doi: str,
    context: str,
    year: Optional[int] = None,
    authors: Optional[list[str]] = None,
) -> dict[str, Any] | ErrorResult:
    """Verify a single DOI against CrossRef and append the verdict to state.

    When to call: once per DOI mentioned in ``propose_ordering`` (the
    LLM is expected to call this for every citation it emits).
    Idempotent: yes — repeat calls hit the global citation cache.
    Expected next tool: another ``verify_doi`` / ``verify_pmid`` until
    every citation is processed, then ``run_baseline``.

    Run ID is *not* a parameter because verification is independent of
    the run; however, when a ``run_id`` context is available, callers
    should use the higher-level helper that records the verdict on the
    run state. This wrapper requires the caller to invoke
    ``verify_doi_for_run`` if they need state side-effects.

    Returns the verdict's ``model_dump()`` so the LLM can read the
    drop_reason / confidence / signals directly.
    """
    try:
        cit_in = CitationInput(
            doi=doi,
            context=context,
            year_claimed=year,
            authors_claimed=authors or [],
        )
        verdict = _get_verifier().verify(cit_in)
        return verdict.model_dump(mode="json")
    except Exception as e:  # noqa: BLE001
        return _error_from_exception(e)


def verify_pmid(
    pmid: str,
    context: str,
    year: Optional[int] = None,
    authors: Optional[list[str]] = None,
) -> dict[str, Any] | ErrorResult:
    """Verify a single PMID against PubMed E-utilities.

    When to call: same pattern as :func:`verify_doi` but for PMIDs.
    Idempotent: yes (cache-backed).
    Expected next tool: continue verifying or proceed to ``run_baseline``.
    """
    try:
        cit_in = CitationInput(
            pmid=pmid,
            context=context,
            year_claimed=year,
            authors_claimed=authors or [],
        )
        verdict = _get_verifier().verify(cit_in)
        return verdict.model_dump(mode="json")
    except Exception as e:  # noqa: BLE001
        return _error_from_exception(e)


def verify_doi_for_run(
    run_id: str,
    doi: str,
    context: str,
    year: Optional[int] = None,
    authors: Optional[list[str]] = None,
) -> dict[str, Any] | ErrorResult:
    """Run-aware :func:`verify_doi`: verifies and updates ``state.json``.

    When to call: in place of ``verify_doi`` when the verdict should be
    appended to the run's ``citations_verified`` / ``citations_dropped``
    lists. Most LLM agents should use this.
    Idempotent: yes — state writes deduplicate by (scheme, id).
    Expected next tool: another verification or ``run_baseline``.
    """
    try:
        validate_run_id(run_id)
        cit_in = CitationInput(
            doi=doi,
            context=context,
            year_claimed=year,
            authors_claimed=authors or [],
        )
        verdict = _get_verifier().verify(cit_in)
        _persist_verdict(run_id, verdict, raw_id=doi)
        return verdict.model_dump(mode="json")
    except Exception as e:  # noqa: BLE001
        return _error_from_exception(e)


def verify_pmid_for_run(
    run_id: str,
    pmid: str,
    context: str,
    year: Optional[int] = None,
    authors: Optional[list[str]] = None,
) -> dict[str, Any] | ErrorResult:
    """Run-aware :func:`verify_pmid`: verifies and updates ``state.json``."""
    try:
        validate_run_id(run_id)
        cit_in = CitationInput(
            pmid=pmid,
            context=context,
            year_claimed=year,
            authors_claimed=authors or [],
        )
        verdict = _get_verifier().verify(cit_in)
        _persist_verdict(run_id, verdict, raw_id=pmid)
        return verdict.model_dump(mode="json")
    except Exception as e:  # noqa: BLE001
        return _error_from_exception(e)


# ---------------------------------------------------------------------------
# Tool: run_baseline
# ---------------------------------------------------------------------------
def run_baseline(run_id: str, method: BaselineName) -> dict[str, Any]:
    """Dispatch a trajectory baseline (in-process or subprocess).

    When to call: once per baseline method, after observations. Each
    invocation persists ``runs/<run_id>/baselines/<method>/`` artifacts
    and records a delta on the cache.
    Idempotent: re-running overwrites the baseline's artifacts and
    appends a duplicate marker to ``baselines_done`` only if missing.
    Expected next tool: another ``run_baseline`` or ``compare_orderings``.

    Returns ``BaselineResult.model_dump()`` on success or
    ``ErrorResult.model_dump()`` on :class:`BaselineFailure`.
    """
    try:
        validate_run_id(run_id)
        state = _read_state(run_id)
        target = run_dir(run_id)

        # Late-import: keeps R-bridging modules off the server cold path.
        from litchron import baselines as _baselines

        with _CACHE.with_adata(
            run_id=run_id,
            h5ad_path=state.h5ad_path,
            run_dir=target,
        ) as adata:
            try:
                result = _baselines.run(
                    method=method,
                    run_id=run_id,
                    h5ad_path=state.h5ad_path,
                    run_dir=target,
                    adata=adata,
                )
            except BaselineFailure as bf:
                # Record the failure on state, return ErrorResult.
                def _mark_failure(s: RunState) -> RunState:
                    if "baseline_failure" not in s.quality_flags:
                        s.quality_flags.append("baseline_failure")
                    return s

                _store_for(run_id).update(_mark_failure)
                return bf.to_result().model_dump(mode="json")

        # Success: record delta + mark baseline done.
        delta_keys = list(result.adata_delta_keys)
        delta_path: Optional[Path] = (
            Path(result.delta_zarr_path) if result.delta_zarr_path else None
        )
        if delta_path is not None and delta_keys:
            _CACHE.apply_delta(
                run_id=run_id,
                baseline=method,
                delta_keys=delta_keys,
                delta_zarr_path=delta_path,
            )

        def _mark_done(s: RunState) -> RunState:
            if method not in s.baselines_done:
                s.baselines_done.append(method)
            # Heuristic for baselines_all_done: at least one in-process
            # Python baseline successfully ran. The full spec checks
            # availability against preflight; we keep it simple here so
            # report_status remains the single source of truth.
            if s.baselines_done:
                s.baselines_all_done = True
            if delta_path is not None and delta_keys:
                s.adata_deltas.append(
                    DeltaRef(
                        baseline=method,
                        keys=delta_keys,
                        zarr_path=str(delta_path),
                        applied_at=_now_iso(),
                    )
                )
            return s

        _store_for(run_id).update(_mark_done)
        return result.model_dump(mode="json")
    except Exception as e:  # noqa: BLE001
        return _error_from_exception(e).model_dump(mode="json")


# ---------------------------------------------------------------------------
# Tool: compare_orderings
# ---------------------------------------------------------------------------
def _load_llm_ordering(target: Path) -> tuple[Any, Optional[list[tuple[str, str]]], Optional[str]]:
    """Parse ``proposals.md`` front-matter back into a pandas Series.

    Lazy-import pandas here so the cold MCP server doesn't pay the
    scientific-stack import cost on startup.
    """
    import pandas as pd  # local import

    path = target / "proposals.md"
    if not path.exists():
        raise LitchronError(
            code="proposals_missing",
            message=f"{path!s} does not exist; call propose_ordering first",
            hint="Invoke propose_ordering(run_id, proposal) before comparing.",
            retryable=False,
        )

    text = path.read_text()
    # Hand-roll a tiny YAML front-matter parser tuned to what
    # _render_proposal_markdown emits — we don't want a yaml dep just
    # for this. The structure we expect:
    #   ---
    #   per_cell_type_rank:
    #     - cell_type: 'name'
    #       rank: N
    #       ...
    #   citations:
    #     ...
    #   ---
    if not text.startswith("---"):
        raise LitchronError(
            code="proposals_unparseable",
            message="proposals.md does not start with a YAML front-matter block",
            hint="Regenerate via propose_ordering.",
            retryable=False,
        )
    end = text.find("\n---", 3)
    if end < 0:
        raise LitchronError(
            code="proposals_unparseable",
            message="proposals.md YAML front-matter not terminated",
            hint="Regenerate via propose_ordering.",
            retryable=False,
        )
    fm = text[3:end].splitlines()

    entries: list[dict[str, Any]] = []
    in_rank_block = False
    in_tied = False
    current: dict[str, Any] = {}

    def _commit() -> None:
        if "cell_type" in current and "rank" in current:
            entries.append(dict(current))

    for raw in fm:
        line = raw.rstrip()
        if line.strip() == "per_cell_type_rank:":
            in_rank_block = True
            continue
        if line.strip().startswith("citations:"):
            in_rank_block = False
            in_tied = False
            _commit()
            current = {}
            continue
        if not in_rank_block:
            continue
        if line.startswith("  - cell_type:"):
            _commit()
            current = {}
            in_tied = False
            val = line.split(":", 1)[1].strip().strip("'\"")
            current["cell_type"] = val
        elif line.startswith("    rank:"):
            in_tied = False
            try:
                current["rank"] = int(line.split(":", 1)[1].strip())
            except ValueError:
                pass
        elif line.startswith("    confidence:"):
            in_tied = False
            try:
                current["confidence"] = float(line.split(":", 1)[1].strip())
            except ValueError:
                pass
        elif line.startswith("    cell_type_label:"):
            in_tied = False
            current["cell_type_label"] = (
                line.split(":", 1)[1].strip().strip("'\"")
            )
        elif line.startswith("    tied_with:"):
            in_tied = True
            current["tied_with"] = []
        elif in_tied and line.startswith("      - "):
            val = line.split("- ", 1)[1].strip().strip("'\"")
            current.setdefault("tied_with", []).append(val)
        else:
            if in_tied and line.strip() and not line.startswith("      "):
                in_tied = False
    _commit()

    if not entries:
        raise LitchronError(
            code="proposals_unparseable",
            message="proposals.md front-matter parsed but yielded no ranks",
            hint="Re-emit propose_ordering with at least one CellTypeRankEntry.",
            retryable=False,
        )

    cell_types = [str(e["cell_type"]) for e in entries]
    ranks = [int(e["rank"]) for e in entries]
    series = pd.Series(ranks, index=cell_types, dtype="float64", name="pseudotime")
    # Attach full per-entry data (confidence, ties, label) on series.attrs
    # so downstream callers (compute_litchron_pseudotime) can consume the
    # complete proposal without re-parsing proposals.md.
    series.attrs["entries"] = entries
    # No tree/root extraction in v1 — the LLM emits flat rankings via
    # CellTypeRankEntry. (Edges-as-output is a P2 follow-up.)
    return series, None, None


def _load_baseline_ordering(
    target: Path, method: str
) -> tuple[Any, Optional[list[tuple[str, str]]], Optional[str]]:
    """Load a baseline's ``ordering.parquet`` into a pandas Series."""
    import json as _json

    import pandas as pd  # local import

    bdir = target / "baselines" / method
    ord_path = bdir / "ordering.parquet"
    if not ord_path.exists():
        raise LitchronError(
            code="baseline_ordering_missing",
            message=f"{ord_path!s} does not exist",
            hint=f"Re-run run_baseline(run_id, {method!r}).",
            retryable=False,
        )
    df = pd.read_parquet(ord_path)
    # Conventional columns: cell_id, pseudotime, cell_type. If a cell_type
    # column exists, aggregate pseudotime per cell_type (mean) so the index
    # aligns with the LLM's per-cell-type ranking. This is the right
    # comparison semantic: the LLM proposes ranks at the cluster level,
    # baselines compute pseudotime per cell, so we project the baseline
    # down to clusters by averaging.
    if "cell_type" in df.columns and "pseudotime" in df.columns:
        s = df.groupby(df["cell_type"].astype(str))["pseudotime"].mean().astype("float64")
    elif "cell_id" in df.columns and "pseudotime" in df.columns:
        s = df.set_index("cell_id")["pseudotime"].astype("float64")
    else:
        s = df.iloc[:, 0]
        s.index = df.index
    s.name = "pseudotime"

    edges: Optional[list[tuple[str, str]]] = None
    edges_path = bdir / "lineage_edges.json"
    if edges_path.exists():
        try:
            data = _json.loads(edges_path.read_text())
            if isinstance(data, list):
                edges = [
                    (str(a), str(b))
                    for a, b in data
                    if isinstance(a, str | int) and isinstance(b, str | int)
                ]
        except (ValueError, TypeError):
            edges = None

    root_cell: Optional[str] = None
    root_path = bdir / "root_cell.txt"
    if root_path.exists():
        try:
            root_cell = root_path.read_text().strip() or None
        except OSError:
            root_cell = None

    return s, edges, root_cell


def compare_orderings(run_id: str) -> dict[str, Any]:
    """Compare the LLM proposal against every persisted baseline.

    When to call: after at least one ``run_baseline`` has succeeded.
    Idempotent: yes — overwrites ``comparison.md`` each time.
    Expected next tool: ``append_section`` (narrative) and ``compile_pdf``.
    """
    try:
        validate_run_id(run_id)
        state = _read_state(run_id)
        target = run_dir(run_id)

        llm_series, llm_edges, llm_root = _load_llm_ordering(target)

        rows: list[ComparisonRow] = []
        bdir_root = target / "baselines"
        if bdir_root.exists():
            for bdir in sorted(bdir_root.iterdir()):
                if not bdir.is_dir():
                    continue
                method = bdir.name
                if method not in state.baselines_done:
                    continue
                try:
                    base_series, base_edges, base_root = _load_baseline_ordering(
                        target, method
                    )
                except LitchronError:
                    continue
                row = _compare_pair(
                    llm_ordering=llm_series,
                    baseline_ordering=base_series,
                    llm_edges=llm_edges,
                    baseline_edges=base_edges,
                    llm_root=llm_root,
                    baseline_root=base_root,
                    baseline_name=method,
                )
                rows.append(row)

        md = comparison_to_markdown(rows)
        cmp_path = target / "comparison.md"
        cmp_path.write_text(md)

        def _mark(s: RunState) -> RunState:
            s.comparison_done = True
            return s

        _store_for(run_id).update(_mark)

        return {
            "path": str(cmp_path),
            "n_baselines_compared": len(rows),
            "rows": [r.model_dump(mode="json") for r in rows],
        }
    except Exception as e:  # noqa: BLE001
        return _error_from_exception(e).model_dump(mode="json")


# ---------------------------------------------------------------------------
# Tool: append_section
# ---------------------------------------------------------------------------
_SECTION_FILENAME: dict[str, str] = {
    "observations": "observations.md",
    "proposals": "proposals.md",
    "baselines": "baselines.md",
    "comparison": "comparison.md",
}


def append_section(
    run_id: str,
    section: SectionName,
    markdown: str,
) -> dict[str, Any]:
    """Append LLM-authored prose to one of the four section markdown files.

    When to call: after the structured content for a section is written
    (e.g., after ``compute_observations`` you may want to append a
    narrative summary). The append is literal — the new bytes are
    placed at the end of the file with a separating blank line.
    Idempotent: no (each call adds bytes).
    Expected next tool: ``compile_pdf`` once all sections are populated.
    """
    try:
        validate_run_id(run_id)
        if section not in _SECTION_FILENAME:
            raise LitchronError(
                code="invalid_section",
                message=f"section {section!r} not in {sorted(_SECTION_FILENAME)}",
                hint="Pick one of: observations, proposals, baselines, comparison.",
                retryable=False,
            )
        target = run_dir(run_id)
        target.mkdir(parents=True, exist_ok=True)
        path = target / _SECTION_FILENAME[section]
        suffix = ("\n\n" if path.exists() and path.stat().st_size else "") + markdown
        if not suffix.endswith("\n"):
            suffix += "\n"
        before = path.stat().st_size if path.exists() else 0
        with path.open("a", encoding="utf-8") as fh:
            fh.write(suffix)
        after = path.stat().st_size
        return AppendResult(
            path=str(path),
            bytes_appended=after - before,
        ).model_dump(mode="json")
    except Exception as e:  # noqa: BLE001
        return _error_from_exception(e).model_dump(mode="json")


# ---------------------------------------------------------------------------
# Tool: report_status
# ---------------------------------------------------------------------------
def _build_suggestions(state: RunState, run_id: str) -> list[SuggestedTool]:
    """Emit advisory next-tool suggestions for false flags only.

    Empty when ``all_green == True``. The LLM retains oracle authority
    and may reorder / ignore these.
    """
    suggestions: list[SuggestedTool] = []
    args = {"run_id": run_id}

    if not state.llm_ordering_done:
        suggestions.append(
            SuggestedTool(
                tool="propose_ordering",
                args=args,
                rationale="No ordering proposed yet; emit a typed OrderingProposal.",
            )
        )
    if not state.baselines_all_done:
        suggestions.append(
            SuggestedTool(
                tool="run_baseline",
                args=args,
                rationale="Baselines not yet marked complete; run at least one method.",
            )
        )
    if not state.comparison_done:
        suggestions.append(
            SuggestedTool(
                tool="compare_orderings",
                args=args,
                rationale="LLM vs baseline comparison not yet generated.",
            )
        )
    if len(state.citations_verified) == 0:
        suggestions.append(
            SuggestedTool(
                tool="verify_doi",
                args=args,
                rationale=(
                    "No verified citations yet; every claim needs a "
                    "CrossRef- or PubMed-verified DOI/PMID."
                ),
            )
        )
    if not state.latex_compiled:
        suggestions.append(
            SuggestedTool(
                tool="compile_pdf",
                args=args,
                rationale="PDF not yet compiled; run after sections are populated.",
            )
        )
    return suggestions


def report_status(run_id: str) -> StatusResult | ErrorResult:
    """Return the full :class:`RunState` plus an advisory next-step list.

    When to call: after every phase-completing tool to learn what
    remains. Stop the loop when ``all_green == True``.
    Idempotent: yes (pure read + recompute).
    Expected next tool: whichever entry of ``suggested_next_tools`` the
    LLM judges most appropriate; an empty list means done.
    """
    try:
        validate_run_id(run_id)

        # Recompute all_green deterministically and persist it so
        # downstream tools share the same single-source-of-truth value.
        def _refresh(s: RunState) -> RunState:
            s.all_green = _recompute_all_green(s)
            s.suggested_next_tools = (
                [] if s.all_green else _build_suggestions(s, run_id)
            )
            # Maintain the "no_verified_citations" QualityFlag
            # transparently for the LLM.
            if (
                not s.citations_verified
                and "no_verified_citations" not in s.quality_flags
            ):
                s.quality_flags.append("no_verified_citations")  # type: ignore[arg-type]
            elif (
                s.citations_verified
                and "no_verified_citations" in s.quality_flags
            ):
                s.quality_flags = [
                    f for f in s.quality_flags if f != "no_verified_citations"
                ]
            return s

        new_state = _store_for(run_id).update(_refresh)
        return StatusResult(
            state=new_state,
            all_green=new_state.all_green,
            suggested_next_tools=new_state.suggested_next_tools,
            quality_flags=list(new_state.quality_flags),
        )
    except Exception as e:  # noqa: BLE001
        return _error_from_exception(e)


# ---------------------------------------------------------------------------
# Tool: compile_pdf
# ---------------------------------------------------------------------------
def compile_pdf(run_id: str) -> dict[str, Any]:
    """Assemble the per-section markdown into a final LaTeX PDF.

    When to call: after all sections are populated and at least one
    citation is verified.
    Idempotent: yes — re-running re-renders the same artifacts.
    Expected next tool: ``finalize_run``.
    """
    try:
        validate_run_id(run_id)
        target = run_dir(run_id)

        # Materialize references.bib from state.citations_verified + global
        # citation cache (which has full Citation records). biblatex requires
        # the .bib file to exist; even an empty one keeps the bibliography
        # section renderable.
        _write_references_bib(run_id, target)

        pdf_path = _compile_pdf_impl(
            run_dir=target,
            project_root=project_root(),
            run_id=run_id,
        )

        def _mark(s: RunState) -> RunState:
            s.latex_compiled = True
            return s

        _store_for(run_id).update(_mark)

        size = pdf_path.stat().st_size if pdf_path.exists() else 0
        return {"path": str(pdf_path), "size_bytes": size}
    except Exception as e:  # noqa: BLE001
        return _error_from_exception(e).model_dump(mode="json")


def _write_references_bib(run_id: str, target: Path) -> None:
    """Build ``<target>/references.bib`` from verified citations in state."""
    from litchron.citations import Citation, CitationVerifier

    state = _read_state(run_id)
    bib_path = target / "references.bib"

    if not state.citations_verified:
        # Empty bib is valid; biblatex tolerates it.
        bib_path.write_text("")
        return

    verifier = _get_verifier()
    cache = verifier._load_global_cache()
    citations: list[Citation] = []
    for fp in state.citations_verified:
        key = f"{fp.scheme}:{fp.id}"
        record = cache.get(key, {}).get("citation")
        if not record:
            continue
        try:
            citations.append(Citation(**record))
        except Exception:  # noqa: BLE001 — skip malformed cache entries
            continue

    bib = CitationVerifier.to_bibtex(citations) if citations else ""
    bib_path.write_text(bib)


# ---------------------------------------------------------------------------
# Tool: finalize_run
# ---------------------------------------------------------------------------
def finalize_run(run_id: str) -> dict[str, Any]:
    """Mark the run finished, evict the AnnData cache, recompute all_green.

    When to call: after ``compile_pdf`` returns successfully and
    ``report_status`` shows ``all_green == True``.
    Idempotent: yes (eviction of an absent entry is a no-op).
    Expected next tool: none — the run is complete.
    """
    try:
        validate_run_id(run_id)

        _CACHE.evict(run_id)

        def _finalize(s: RunState) -> RunState:
            s.finished_at = _now_iso()
            s.all_green = _recompute_all_green(s)
            s.suggested_next_tools = (
                [] if s.all_green else _build_suggestions(s, run_id)
            )
            s.phase = "finalized" if s.all_green else s.phase
            return s

        new_state = _store_for(run_id).update(_finalize)
        return new_state.model_dump(mode="json")
    except Exception as e:  # noqa: BLE001
        return _error_from_exception(e).model_dump(mode="json")


# ---------------------------------------------------------------------------
# Tool: search_crossref
# ---------------------------------------------------------------------------
def search_crossref(
    query: str,
    max_results: int = 10,
    year_min: Optional[int] = None,
    year_max: Optional[int] = None,
) -> dict[str, Any] | ErrorResult:
    """Search CrossRef for supporting papers given a free-text biological query. Returns up to max_results candidate Citations (DOI, title, year, authors, abstract). These are NOT verified — use verify_doi after to pass them through the multi-signal verifier. When to call: when the LLM wants to back a claim with literature beyond its parametric memory. Idempotent: yes. Expected next tool: verify_doi for each promising result.

    Validates max_results in [1, 50].
    """
    try:
        if not (1 <= max_results <= 50):
            return ErrorResult(
                code="invalid_input",
                message=f"max_results must be in [1, 50]; got {max_results}",
                hint="Pass an integer between 1 and 50 inclusive.",
                retryable=False,
            )
        results: list[Citation] = _get_verifier().search_crossref(
            query=query,
            max_results=max_results,
            year_min=year_min,
            year_max=year_max,
        )
        return {
            "query": query,
            "n_results": len(results),
            "results": [c.model_dump(mode="json") for c in results],
        }
    except Exception as e:  # noqa: BLE001
        return _error_from_exception(e)


# ---------------------------------------------------------------------------
# Tool: search_europepmc
# ---------------------------------------------------------------------------
def search_europepmc(
    query: str,
    max_results: int = 10,
) -> dict[str, Any] | ErrorResult:
    """Search Europe PMC for supporting biomedical papers given a free-text query. This endpoint is PMID-based and may return clinical/biomedical hits CrossRef misses. Returns up to max_results candidate Citations (PMID or DOI, title, year, authors, abstract). These are NOT verified — use verify_doi or verify_pmid after to pass them through the multi-signal verifier. When to call: when the LLM wants to find biomedical literature, especially clinical or PubMed-indexed papers. Idempotent: yes. Expected next tool: verify_pmid (or verify_doi) for each promising result.

    Validates max_results in [1, 50].
    """
    try:
        if not (1 <= max_results <= 50):
            return ErrorResult(
                code="invalid_input",
                message=f"max_results must be in [1, 50]; got {max_results}",
                hint="Pass an integer between 1 and 50 inclusive.",
                retryable=False,
            )
        results: list[Citation] = _get_verifier().search_europepmc(
            query=query,
            max_results=max_results,
        )
        return {
            "query": query,
            "n_results": len(results),
            "results": [c.model_dump(mode="json") for c in results],
        }
    except Exception as e:  # noqa: BLE001
        return _error_from_exception(e)


# ---------------------------------------------------------------------------
# Tool: compute_litchron_pseudotime
# ---------------------------------------------------------------------------
class LitchronPseudotimeResult(BaseModel):
    """Return type of :func:`compute_litchron_pseudotime`."""

    path: str
    n_cells: int
    method: str = "llm_continuous"
    spread_method: str


def _write_litchron_pseudotime_delta(
    run_dir_path: Path,
    pseudotime: Any,
) -> tuple[Path, list[str]]:
    """Persist a zarr delta with ``obs/litchron_pseudotime``."""
    import numpy as np  # local — keeps cold-import cheap
    import zarr  # heavy import, kept local

    embeddings_dir = run_dir_path / "embeddings"
    embeddings_dir.mkdir(parents=True, exist_ok=True)
    delta_path = embeddings_dir / "litchron_pseudotime.zarr"
    root = zarr.open(str(delta_path), mode="w")
    root.create_array(
        name="obs/litchron_pseudotime",
        data=np.asarray(pseudotime.values, dtype=np.float64),
        overwrite=True,
    )
    return delta_path, ["obs/litchron_pseudotime"]


def compute_litchron_pseudotime(
    run_id: str,
    spread_method: str = "diffmap",
) -> dict[str, Any]:
    """Lift the LLM's per-cluster ranks to a continuous per-cell pseudotime.

    When to call: after ``propose_ordering`` has persisted the LLM proposal
    and (optionally) after ``recompute_embeddings`` has populated
    ``X_diffmap`` / ``X_pca``. Produces the LitChron primary trajectory
    output that comparators (PAGA, scVelo) are aligned against.
    Idempotent: yes — the parquet + zarr delta are overwritten on re-run.
    Expected next tool: ``align_orderings`` or ``compare_orderings``.
    """
    try:
        import pandas as pd  # local

        from litchron.llm_pseudotime import compute_llm_pseudotime

        validate_run_id(run_id)
        state = _read_state(run_id)
        target = run_dir(run_id)

        # Reuse the existing front-matter parser; it returns entries with
        # confidence + ties + label via series.attrs["entries"].
        _llm_series, _edges, _root = _load_llm_ordering(target)
        per_cell_type_rank: list[dict[str, Any]] = _llm_series.attrs.get(
            "entries", []
        )
        if not per_cell_type_rank:
            raise LitchronError(
                code="proposals_unparseable",
                message="proposals.md parsed but yielded no per-cell-type entries",
                hint="Re-emit propose_ordering with at least one CellTypeRankEntry.",
                retryable=False,
            )

        with _CACHE.with_adata(
            run_id=run_id,
            h5ad_path=state.h5ad_path,
            run_dir=target,
        ) as adata:
            # Prefer the leiden column from recompute_embeddings; fall back
            # to "cell_type" inside compute_llm_pseudotime itself.
            pseudotime = compute_llm_pseudotime(
                adata=adata,
                per_cell_type_rank=per_cell_type_rank,
                cell_type_col="leiden",
                spread_method=spread_method,
            )

            # Build the per-cell parquet. Carry the cluster id (and optional
            # human label) so downstream tools can group by lineage stage.
            col = "leiden" if "leiden" in adata.obs.columns else "cell_type"
            cluster_series = adata.obs[col].astype(str)

        label_map: dict[str, str] = {}
        for entry in per_cell_type_rank:
            cid = str(entry.get("cell_type", ""))
            label = entry.get("cell_type_label") or cid
            label_map[cid] = str(label)

        df = pd.DataFrame(
            {
                "cell_id": pseudotime.index.astype(str),
                "pseudotime": pseudotime.values,
                "cell_type": cluster_series.values,
                "cell_type_label": [
                    label_map.get(c, c) for c in cluster_series.values
                ],
            }
        )
        ord_path = target / "litchron_pseudotime.parquet"
        df.to_parquet(ord_path, index=False)

        # Persist as a zarr delta on the adata + record it in run state.
        delta_path, delta_keys = _write_litchron_pseudotime_delta(
            target, pseudotime
        )
        _CACHE.apply_delta(
            run_id=run_id,
            baseline="litchron_pseudotime",
            delta_keys=delta_keys,
            delta_zarr_path=delta_path,
        )

        def _record(s: RunState) -> RunState:
            s.adata_deltas.append(
                DeltaRef(
                    baseline="litchron_pseudotime",
                    keys=delta_keys,
                    zarr_path=str(delta_path),
                    applied_at=_now_iso(),
                )
            )
            return s

        _store_for(run_id).update(_record)

        return LitchronPseudotimeResult(
            path=str(ord_path),
            n_cells=int(len(df)),
            method="llm_continuous",
            spread_method=spread_method,
        ).model_dump(mode="json")
    except Exception as e:  # noqa: BLE001
        return _error_from_exception(e).model_dump(mode="json")


# ---------------------------------------------------------------------------
# Tool: align_orderings
# ---------------------------------------------------------------------------
def align_orderings(run_id: str) -> dict[str, Any]:
    """Pairwise alignment between LitChron pseudotime and each baseline.

    When to call: after ``compute_litchron_pseudotime`` and at least one
    ``run_baseline``. Produces ``<run_dir>/alignment.md`` with a markdown
    table of Spearman, Kendall, Goodman-Kruskal gamma, and monotonic
    concordance for each (method_a, method_b) pair.
    Idempotent: yes — overwrites ``alignment.md`` on every call.
    Expected next tool: ``append_section`` or ``compile_pdf``.
    """
    try:
        import pandas as pd  # local

        from litchron.llm_alignment import (
            align_orderings as _align,
        )
        from litchron.llm_alignment import (
            alignment_to_markdown,
        )

        validate_run_id(run_id)
        state = _read_state(run_id)
        target = run_dir(run_id)

        orderings: dict[str, pd.Series] = {}

        # LitChron continuous pseudotime.
        lit_path = target / "litchron_pseudotime.parquet"
        if lit_path.exists():
            df = pd.read_parquet(lit_path)
            if "cell_id" in df.columns and "pseudotime" in df.columns:
                s = df.set_index("cell_id")["pseudotime"].astype("float64")
                s.name = "pseudotime"
                orderings["litchron"] = s

        # Each baseline's per-cell ordering.parquet (skip aggregated views).
        for method in state.baselines_done:
            ord_path = target / "baselines" / method / "ordering.parquet"
            if not ord_path.exists():
                continue
            try:
                df = pd.read_parquet(ord_path)
            except Exception:  # noqa: BLE001 — skip unreadable parquet
                continue
            if "cell_id" not in df.columns or "pseudotime" not in df.columns:
                continue
            s = df.set_index("cell_id")["pseudotime"].astype("float64")
            s.name = "pseudotime"
            orderings[method] = s

        if len(orderings) < 2:
            return {
                "path": None,
                "pairs": {},
                "n_methods": len(orderings),
                "note": (
                    "Need at least two named pseudotime series for "
                    "alignment; compute the LitChron pseudotime and at "
                    "least one per-cell baseline first."
                ),
            }

        result = _align(orderings)
        md = alignment_to_markdown(result)
        out_path = target / "alignment.md"
        out_path.write_text(md)

        # Serialize tuple keys as "a__b" strings for JSON return.
        pairs_json: dict[str, dict[str, Any]] = {
            f"{a}__{b}": dict(metrics) for (a, b), metrics in result.items()
        }
        return {
            "path": str(out_path),
            "pairs": pairs_json,
            "n_methods": len(orderings),
        }
    except Exception as e:  # noqa: BLE001
        return _error_from_exception(e).model_dump(mode="json")


# ---------------------------------------------------------------------------
# Tool: emit_figure_script
# ---------------------------------------------------------------------------
class EmitFigureScriptInput(BaseModel):
    """Input model for :func:`emit_figure_script`."""

    run_id: str
    figure_name: Literal["annotation", "comparison_strip"]


_ANNOTATION_DRIVER_TEMPLATE = '''\
"""Auto-generated driver: build the LitChron annotation figure for run {run_id!r}.

This script is self-contained: it loads run state, reconstructs the AnnData
cache, and builds the figure into pyplot state WITHOUT calling plt.close().
scivcd's runner picks up the live figure via plt.get_fignums().

Do NOT add an if __name__ == '__main__' guard — scivcd imports this module via
importlib.util.spec_from_file_location, so module-level code must run directly.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from pathlib import Path
from mcp_litchron.tools import _CACHE, _read_proposal_maps, _read_state, run_dir
from litchron.figures import make_litchron_annotation_figure

_RUN_ID = {run_id!r}
_state = _read_state(_RUN_ID)
_target = Path(run_dir(_RUN_ID))
_label_map, _rank_map, _confidence_map = _read_proposal_maps(_target)

# Suppress plt.close so the figure survives into pyplot state for scivcd.
_orig_close = plt.close
plt.close = lambda *_a, **_k: None  # type: ignore[assignment]

with _CACHE.with_adata(run_id=_RUN_ID, h5ad_path=_state.h5ad_path, run_dir=_target) as _adata:
    make_litchron_annotation_figure(
        adata=_adata,
        run_dir=_target,
        label_map=_label_map,
        rank_map=_rank_map,
        confidence_map=_confidence_map,
    )

# Restore plt.close so subsequent imports aren\'t affected.
plt.close = _orig_close  # type: ignore[assignment]
'''

_COMPARISON_STRIP_DRIVER_TEMPLATE = '''\
"""Auto-generated driver: build the LitChron comparison strip for run {run_id!r}.

This script is self-contained: it loads run state, reconstructs the AnnData
cache, and builds the figure into pyplot state WITHOUT calling plt.close().
scivcd's runner picks up the live figure via plt.get_fignums().

Do NOT add an if __name__ == '__main__' guard — scivcd imports this module via
importlib.util.spec_from_file_location, so module-level code must run directly.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import pandas as pd  # noqa: E402
from pathlib import Path
from mcp_litchron.tools import _CACHE, _read_state, run_dir
from litchron.figures import make_pseudotime_comparison_strip

_RUN_ID = {run_id!r}
_state = _read_state(_RUN_ID)
_target = Path(run_dir(_RUN_ID))

_lit_path = _target / "litchron_pseudotime.parquet"
if not _lit_path.exists():
    raise FileNotFoundError(
        f"comparison strip driver: litchron_pseudotime.parquet missing in {{_target}}"
    )

_lit_df = pd.read_parquet(_lit_path)
_llm_pt = _lit_df.set_index("cell_id")["pseudotime"].astype("float64")

_baseline_pts: dict = {{}}
_baselines_dir = _target / "baselines"
if _baselines_dir.is_dir():
    for _method_dir in sorted(_baselines_dir.iterdir()):
        if not _method_dir.is_dir():
            continue
        _ord_path = _method_dir / "ordering.parquet"
        if not _ord_path.exists():
            continue
        try:
            _df = pd.read_parquet(_ord_path)
        except Exception:
            continue
        if "cell_id" not in _df.columns or "pseudotime" not in _df.columns:
            continue
        _baseline_pts[_method_dir.name] = _df.set_index("cell_id")["pseudotime"].astype("float64")

if not _baseline_pts:
    raise RuntimeError(
        f"comparison strip driver: no baseline parquets found in {{_baselines_dir}}"
    )

# Suppress plt.close so the figure survives into pyplot state for scivcd.
_orig_close = plt.close
plt.close = lambda *_a, **_k: None  # type: ignore[assignment]

with _CACHE.with_adata(run_id=_RUN_ID, h5ad_path=_state.h5ad_path, run_dir=_target) as _adata:
    make_pseudotime_comparison_strip(
        adata=_adata,
        llm_pt=_llm_pt,
        baseline_pts=_baseline_pts,
        run_dir=_target,
    )

# Restore plt.close so subsequent imports aren\'t affected.
plt.close = _orig_close  # type: ignore[assignment]
'''


def _check_deps_pinned() -> bool:
    """Return True if pyproject.toml has a pinned scivcd in [audit] extras."""
    try:
        import tomllib  # Python 3.11+ stdlib

        toml_path = project_root() / "pyproject.toml"
        if not toml_path.exists():
            return False
        with toml_path.open("rb") as _f:
            _data = tomllib.load(_f)
        opt_deps = _data.get("project", {}).get("optional-dependencies", {})
        audit_deps = opt_deps.get("audit", [])
        return any("scivcd==" in dep for dep in audit_deps)
    except Exception:  # noqa: BLE001
        return False


def emit_figure_script(
    run_id: str,
    figure_name: Literal["annotation", "comparison_strip"],
) -> dict[str, Any]:
    """Emit a driver script that builds one figure into pyplot state. Args: run_id (str), figure_name ('annotation' or 'comparison_strip'). Returns {script_path, figure_name, deps_pinned}.

    The emitted script is written to ``<run_dir>/scripts/audit_<figure_name>.py``.
    scivcd's runner can execute it via ``importlib.util.spec_from_file_location``
    and pick up the resulting figure via ``plt.get_fignums()``.

    The MCP dispatcher unpacks JSON-RPC arguments as ``fn(**arguments)``, so
    callers MUST pass ``run_id`` and ``figure_name`` as flat keyword arguments,
    NOT wrapped in an ``inp`` envelope. Validation still flows through the
    :class:`EmitFigureScriptInput` model internally for shape consistency.
    """
    try:
        validated = EmitFigureScriptInput(run_id=run_id, figure_name=figure_name)
        validate_run_id(validated.run_id)

        target = Path(run_dir(validated.run_id))
        scripts_dir = target / "scripts"
        scripts_dir.mkdir(parents=True, exist_ok=True)

        script_path = scripts_dir / f"audit_{validated.figure_name}.py"

        if validated.figure_name == "annotation":
            content = _ANNOTATION_DRIVER_TEMPLATE.format(run_id=validated.run_id)
        else:
            content = _COMPARISON_STRIP_DRIVER_TEMPLATE.format(run_id=validated.run_id)

        script_path.write_text(content)

        return {
            "script_path": str(script_path),
            "figure_name": validated.figure_name,
            "deps_pinned": _check_deps_pinned(),
        }
    except Exception as e:  # noqa: BLE001
        return _error_from_exception(e).model_dump(mode="json")


# ---------------------------------------------------------------------------
# Public registry (consumed by server.py + tests)
# ---------------------------------------------------------------------------
class ToolSpec(BaseModel):
    """Compact descriptor for a single MCP tool entry.

    ``input_model`` is the Pydantic class (when an argument-object is
    used) or ``None`` for plain-argument tools. ``callable_name`` is
    the function name in this module — :mod:`mcp_litchron.server` looks
    it up at registration time.
    """

    name: str
    description: str
    callable_name: str


def _doc_first_line(fn: Any) -> str:
    """Pluck the first non-empty line of a callable's docstring."""
    doc = (fn.__doc__ or "").strip().splitlines()
    return doc[0] if doc else ""


TOOL_REGISTRY: list[ToolSpec] = [
    ToolSpec(
        name="start_run",
        description=_doc_first_line(start_run),
        callable_name="start_run",
    ),
    ToolSpec(
        name="load_h5ad",
        description=_doc_first_line(load_h5ad),
        callable_name="load_h5ad",
    ),
    ToolSpec(
        name="compute_observations",
        description=_doc_first_line(compute_observations),
        callable_name="compute_observations",
    ),
    ToolSpec(
        name="recompute_embeddings",
        description=_doc_first_line(recompute_embeddings),
        callable_name="recompute_embeddings",
    ),
    ToolSpec(
        name="make_annotation_figure",
        description=_doc_first_line(make_annotation_figure),
        callable_name="make_annotation_figure",
    ),
    ToolSpec(
        name="propose_ordering",
        description=_doc_first_line(propose_ordering),
        callable_name="propose_ordering",
    ),
    ToolSpec(
        name="verify_doi",
        description=_doc_first_line(verify_doi),
        callable_name="verify_doi",
    ),
    ToolSpec(
        name="verify_pmid",
        description=_doc_first_line(verify_pmid),
        callable_name="verify_pmid",
    ),
    ToolSpec(
        name="run_baseline",
        description=_doc_first_line(run_baseline),
        callable_name="run_baseline",
    ),
    ToolSpec(
        name="compute_litchron_pseudotime",
        description=_doc_first_line(compute_litchron_pseudotime),
        callable_name="compute_litchron_pseudotime",
    ),
    ToolSpec(
        name="align_orderings",
        description=_doc_first_line(align_orderings),
        callable_name="align_orderings",
    ),
    ToolSpec(
        name="compare_orderings",
        description=_doc_first_line(compare_orderings),
        callable_name="compare_orderings",
    ),
    ToolSpec(
        name="append_section",
        description=_doc_first_line(append_section),
        callable_name="append_section",
    ),
    ToolSpec(
        name="report_status",
        description=_doc_first_line(report_status),
        callable_name="report_status",
    ),
    ToolSpec(
        name="compile_pdf",
        description=_doc_first_line(compile_pdf),
        callable_name="compile_pdf",
    ),
    ToolSpec(
        name="finalize_run",
        description=_doc_first_line(finalize_run),
        callable_name="finalize_run",
    ),
    ToolSpec(
        name="search_crossref",
        description=_doc_first_line(search_crossref),
        callable_name="search_crossref",
    ),
    ToolSpec(
        name="search_europepmc",
        description=_doc_first_line(search_europepmc),
        callable_name="search_europepmc",
    ),
    ToolSpec(
        name="emit_figure_script",
        description=_doc_first_line(emit_figure_script),
        callable_name="emit_figure_script",
    ),
]


__all__ = [
    # Pydantic models
    "StartRunResult",
    "LoadResult",
    "ObservationsResult",
    "RecomputeEmbeddingsResult",
    "AnnotationFigureResult",
    "LitchronPseudotimeResult",
    "CellTypeRankEntry",
    "OrderingProposal",
    "ProposalResult",
    "AppendResult",
    "StatusResult",
    "ToolSpec",
    "SectionName",
    "BaselineName",
    # Tools
    "start_run",
    "load_h5ad",
    "compute_observations",
    "recompute_embeddings",
    "make_annotation_figure",
    "propose_ordering",
    "verify_doi",
    "verify_pmid",
    "verify_doi_for_run",
    "verify_pmid_for_run",
    "run_baseline",
    "compute_litchron_pseudotime",
    "align_orderings",
    "compare_orderings",
    "append_section",
    "report_status",
    "compile_pdf",
    "finalize_run",
    "search_crossref",
    "search_europepmc",
    "emit_figure_script",
    "EmitFigureScriptInput",
    # Registry + singletons
    "TOOL_REGISTRY",
    "_CACHE",
    "_VERIFIER",
    "_get_verifier",
]
