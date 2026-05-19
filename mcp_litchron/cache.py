"""Explicit AnnData LRU cache for the LitChron MCP server layer.

Replay protocol
---------------
Baseline-induced mutations to an in-memory ``AnnData`` are persisted as
zarr "deltas" under ``runs/<run_id>/baselines/<method>/adata_delta.zarr``.
The cache records each delta as a :class:`litchron.state.DeltaRef` on the
run state. Replay semantics, lifted from Plan §3 Phase 1:

* **Append-only.** Deltas are never removed from the recorded ordering.
* **Last-writer-wins.** When two deltas write the same key (``adata.uns``,
  ``adata.obsp``, or ``adata.layers``), the later write overrides the earlier
  one. The first write is *not* purged from disk but is shadowed in memory.
* **Idempotent.** Replaying the same delta twice is a no-op because zarr load
  + assignment to ``adata.uns[key] = value`` overwrites with the same value.
* **No deletes in v1.** A baseline that drops a key triggers a
  ``QualityFlag`` upstream; the key is retained in the materialized adata.

The cache is intentionally *not* the source of truth — ``runs/<run_id>/``
is. The cache is a fingerprinted in-memory accelerator that can always be
rebuilt by replaying state-recorded deltas.
"""
from __future__ import annotations

import hashlib
import json
import time
from collections import OrderedDict
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, NamedTuple, Optional

from anndata import AnnData

from litchron.io import load_h5ad
from litchron.state import DeltaRef


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------
class AnnDataCacheEntry(NamedTuple):
    """In-memory cache slot for a single run's AnnData."""

    adata: AnnData
    fingerprint: tuple[str, int, int, str]
    applied_deltas: list[str]
    last_access: float


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------
class AnnDataCache:
    """Bounded LRU cache mapping ``run_id`` → loaded :class:`AnnData`.

    Side effects
    ------------
    * :meth:`load` writes ``<run_dir>/cache_fingerprint.json`` on every
      successful load.
    * :meth:`apply_delta` is a *recording-only* operation: the caller is
      responsible for actually writing the zarr to disk. This method only
      updates the in-memory ``applied_deltas`` list. (The on-disk
      persistence is tracked in ``RunState.adata_deltas``.)
    * :meth:`resume` invokes ``load_h5ad`` (filesystem read) and reads each
      recorded zarr (filesystem read) but does not write anything.
    """

    # -- construction ------------------------------------------------------
    def __init__(self, max_entries: int = 4) -> None:
        if max_entries < 1:
            raise ValueError("max_entries must be >= 1")
        self._max_entries: int = max_entries
        self._entries: OrderedDict[str, AnnDataCacheEntry] = OrderedDict()

    # -- fingerprinting ----------------------------------------------------
    @staticmethod
    def _fingerprint(h5ad_path: str) -> tuple[str, int, int, str]:
        """Return ``(path, st_mtime_ns, st_size, sha1[:16])``.

        Trade-off: the SHA-1 digest is computed over the *first 1 MB* of the
        file rather than the full content. This is intentional — h5ad files
        can be 10+ GB, and a full digest would dwarf the load itself. The
        ``(mtime_ns, size)`` tuple catches almost every realistic mutation;
        the truncated digest defends against in-place edits that preserve
        mtime/size (rare but possible).
        """
        p = Path(h5ad_path)
        st = p.stat()
        h = hashlib.sha1()
        with p.open("rb") as fh:
            h.update(fh.read(1024 * 1024))
        return (str(p.resolve()), st.st_mtime_ns, st.st_size, h.hexdigest()[:16])

    @staticmethod
    def _write_fingerprint(run_dir: Path, fp: tuple[str, int, int, str]) -> None:
        """Persist the cache fingerprint as JSON in the run directory.

        Side effect: writes ``<run_dir>/cache_fingerprint.json`` (overwrites).
        """
        run_dir.mkdir(parents=True, exist_ok=True)
        target = run_dir / "cache_fingerprint.json"
        payload = {
            "path": fp[0],
            "st_mtime_ns": fp[1],
            "st_size": fp[2],
            "sha1_prefix": fp[3],
        }
        target.write_text(json.dumps(payload, indent=2, sort_keys=True))

    @staticmethod
    def _read_fingerprint(run_dir: Path) -> Optional[tuple[str, int, int, str]]:
        """Read the fingerprint JSON; return None if missing or corrupt."""
        target = run_dir / "cache_fingerprint.json"
        if not target.exists():
            return None
        try:
            payload = json.loads(target.read_text())
            return (
                str(payload["path"]),
                int(payload["st_mtime_ns"]),
                int(payload["st_size"]),
                str(payload["sha1_prefix"]),
            )
        except (json.JSONDecodeError, KeyError, ValueError, OSError):
            return None

    # -- LRU bookkeeping ---------------------------------------------------
    def _touch(self, run_id: str) -> None:
        """Move ``run_id`` to the most-recently-used end of the OrderedDict."""
        if run_id in self._entries:
            entry = self._entries.pop(run_id)
            refreshed = entry._replace(last_access=time.time())
            self._entries[run_id] = refreshed

    def _evict_lru_if_needed(self) -> None:
        """Drop the least-recently-used entry while over capacity."""
        while len(self._entries) >= self._max_entries:
            self._entries.popitem(last=False)

    def __contains__(self, run_id: object) -> bool:
        """Public membership check so callers can detect cache hits without poking _entries."""
        return run_id in self._entries

    # -- public API --------------------------------------------------------
    def load(self, run_id: str, h5ad_path: str, run_dir: Path) -> AnnData:
        """Return the cached AnnData or load it from disk on miss.

        Cache hit requires:
        1. ``run_id`` is present in the in-memory dict, and
        2. the on-disk h5ad fingerprint matches the recorded fingerprint.

        Side effects
        ------------
        * On miss: invokes :func:`litchron.io.load_h5ad` and writes the
          fingerprint JSON under ``run_dir``.
        * Always: updates ``last_access`` for LRU bookkeeping.
        """
        current_fp = self._fingerprint(h5ad_path)
        cached = self._entries.get(run_id)
        if cached is not None and cached.fingerprint == current_fp:
            self._touch(run_id)
            return cached.adata

        # Miss — evict to make room, then load.
        if run_id in self._entries:
            # Stale entry: drop it before re-loading.
            self._entries.pop(run_id)
        self._evict_lru_if_needed()

        adata = load_h5ad(h5ad_path)
        self._write_fingerprint(run_dir, current_fp)
        self._entries[run_id] = AnnDataCacheEntry(
            adata=adata,
            fingerprint=current_fp,
            applied_deltas=[],
            last_access=time.time(),
        )
        return adata

    def apply_delta(
        self,
        run_id: str,
        baseline: str,
        delta_keys: list[str],
        delta_zarr_path: Path,
    ) -> None:
        """Record that ``baseline`` wrote ``delta_keys`` onto the cached adata.

        This method is *bookkeeping only*. The caller is responsible for
        having already written the zarr at ``delta_zarr_path``. The cache
        merely records the baseline → keys provenance so that :meth:`resume`
        can replay them after a restart.

        Side effect: mutates ``applied_deltas`` for the entry.
        """
        entry = self._entries.get(run_id)
        if entry is None:
            raise KeyError(
                f"AnnDataCache.apply_delta: no cached entry for run_id={run_id!r}; "
                "call load() first"
            )
        new_marker = f"{baseline}:{delta_zarr_path}:{','.join(sorted(delta_keys))}"
        applied = list(entry.applied_deltas)
        applied.append(new_marker)
        self._entries[run_id] = entry._replace(
            applied_deltas=applied,
            last_access=time.time(),
        )

    def resume(
        self,
        run_id: str,
        h5ad_path: str,
        run_dir: Path,
        deltas: list[DeltaRef],
    ) -> AnnData:
        """Re-materialize the adata after a restart by replaying deltas.

        Algorithm
        ---------
        1. Drop any stale in-memory entry for ``run_id``.
        2. Re-load the base h5ad via :meth:`load`.
        3. For each :class:`DeltaRef` in recorded order, open its zarr and
           re-assign every recorded key into the adata. Last-writer-wins is
           the natural consequence of plain assignment.

        Side effects
        ------------
        * Re-reads the h5ad and every recorded zarr from disk.
        * Writes the fingerprint JSON via the inner :meth:`load` call.
        """
        if run_id in self._entries:
            self._entries.pop(run_id)
        adata = self.load(run_id, h5ad_path, run_dir)

        # Replay deltas in recorded order.
        import zarr  # local import — zarr is a heavyweight optional dep

        for delta in deltas:
            zarr_path = Path(delta.zarr_path)
            if not zarr_path.exists():
                # Missing delta on disk is non-fatal but worth a record.
                # Caller (server) is expected to convert this to a
                # QualityFlag if it cares.
                continue
            root = zarr.open(str(zarr_path), mode="r")
            for key in delta.keys:
                if key not in root:
                    continue
                value = root[key][...]
                # Route the value to the correct AnnData slot based on the
                # key's namespace prefix. Without this routing, every delta
                # key landed in adata.uns under its full prefixed name, so
                # downstream code expecting adata.obsm['X_umap'] never saw
                # the replayed embedding.
                if key.startswith("obsm/"):
                    adata.obsm[key[len("obsm/"):]] = value
                elif key.startswith("obs/"):
                    adata.obs[key[len("obs/"):]] = value
                elif key.startswith("layers/"):
                    adata.layers[key[len("layers/"):]] = value
                elif key.startswith("uns/"):
                    adata.uns[key[len("uns/"):]] = value
                else:
                    # Unknown namespace — stash in uns under the original
                    # key for forensic visibility rather than dropping.
                    adata.uns[key] = value
            self.apply_delta(run_id, delta.baseline, list(delta.keys), zarr_path)
        return adata

    @contextmanager
    def with_adata(
        self,
        run_id: str,
        h5ad_path: str,
        run_dir: Path,
        deltas: list[DeltaRef] | None = None,
    ) -> Iterator[AnnData]:
        """Context manager yielding the cached AnnData for ``run_id``.

        When ``deltas`` is provided AND the cache is cold for this run_id,
        replays the deltas via :meth:`resume` so the materialized AnnData
        carries every recorded obsm/obs/layers update from prior tool
        invocations. Without this, a fresh process loading a run whose
        embeddings live only in a zarr delta (e.g. after
        ``recompute_embeddings`` in a different process) would see the
        bare h5ad and miss adata.obsm['X_umap'] entirely.

        When ``deltas`` is omitted (existing call sites) behavior is
        unchanged — bare h5ad load via :meth:`load`.

        The cache holds a reference for the lifetime of the entry, so the
        context manager does not currently perform cleanup on exit — it
        exists to give baseline call-sites a uniform access pattern.
        """
        cache_cold = run_id not in self._entries
        if deltas and cache_cold:
            # Cold cache + deltas to replay → use resume which routes each
            # delta key into the correct AnnData slot.
            adata = self.resume(run_id, h5ad_path, run_dir, deltas)
        else:
            adata = self.load(run_id, h5ad_path, run_dir)
        try:
            yield adata
        finally:
            self._touch(run_id)

    def evict(self, run_id: str) -> None:
        """Drop the cache entry for ``run_id`` if present.

        Side effect: frees the in-memory AnnData reference. The on-disk
        fingerprint JSON is left untouched (re-used on next load).
        """
        self._entries.pop(run_id, None)
