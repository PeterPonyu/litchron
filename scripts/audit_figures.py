"""Routine scivcd audit for LitChron's headline figures.

Run after any change to ``litchron/figures.py`` (or as a pre-commit gate)
to make sure the headline annotation figure still passes scivcd's
publication-quality checks.

Usage:
    conda run -n dl python scripts/audit_figures.py
    conda run -n dl python scripts/audit_figures.py --run-id wtko-hspc-2026-05-18
    conda run -n dl python scripts/audit_figures.py --strict   # exit 2 on new MAJOR

Exit codes:
    0 — clean (no NEW CRITICAL/MAJOR vs baseline)
    1 — NEW CRITICAL finding vs baseline (always blocks commit)
    2 — NEW MAJOR finding vs baseline (only when --strict)
    3 — driver failed to build the figure (data missing, import error, etc.)

On first run for a run_id, baseline.json does not exist yet. All current
findings are treated as "new", which causes exit 1 if any CRITICAL findings
exist. The author then runs ``git add <run_dir>/audit/baseline.json`` to
bless the initial state. On subsequent runs the gate is delta-relative.

Why this script exists rather than `scivcd lint <fig.py>`:
    1. The figure builder calls plt.close(); raw `scivcd lint` would have
       nothing to audit. This driver disables the close before the builder
       runs, then enumerates the surviving Figure objects.
    2. Previously, scivcd's font_family_violation check was suppressed via
       IGNORED_CHECK_TYPES. That set is now empty — such findings surface in
       baseline.json as pre-existing on first run and do not block commits
       unless they are *new* relative to the saved baseline.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Disable plt.close so figures built below survive into plt.get_fignums()
# for the audit. Restored at the end so any code that imports this module
# (e.g. tests) doesn't get its close() neutered.
_ORIGINAL_CLOSE = plt.close
plt.close = lambda *_a, **_k: None  # type: ignore[assignment]

from scivcd import detect_all_conflicts  # noqa: E402

from litchron.figures import (  # noqa: E402
    make_litchron_annotation_figure,
    make_pseudotime_comparison_strip,
)
from mcp_litchron.tools import (  # noqa: E402
    _CACHE,
    _read_proposal_maps,
    _read_state,
    run_dir,
)

# Previously this set contained "font_family_violation" to suppress LitChron's
# intentional serif (Computer Modern) aesthetic from failing audits. That
# approach is decommissioned: the set is now empty, and any pre-existing
# findings (font_family_violation or otherwise) live in baseline.json as
# known-acceptable. They surface in the report but do not block commits unless
# they are *new* relative to the saved baseline.
IGNORED_CHECK_TYPES: set[str] = set()

def _resolve_default_run_id() -> str | None:
    """Resolve the default --run-id without hardcoding a specific dataset.

    Priority:
      1. ``LITCHRON_AUDIT_RUN_ID`` env var (explicit override)
      2. Newest directory under ``<project_root>/runs/`` that contains
         ``state.json`` (auto-discovery of the most recent run)
      3. ``None`` — argparse then requires the user to pass ``--run-id``
         explicitly rather than silently auditing the wrong run

    No literal run_id is baked in. This script is reusable across datasets
    and projects that follow LitChron's ``runs/<run_id>/state.json`` layout.
    """
    env_val = os.environ.get("LITCHRON_AUDIT_RUN_ID")
    if env_val:
        return env_val
    from litchron._runtime import project_root

    runs_root = project_root() / "runs"
    if not runs_root.is_dir():
        return None
    candidates = [
        p for p in runs_root.iterdir() if p.is_dir() and (p / "state.json").exists()
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime).name


DEFAULT_RUN_ID = _resolve_default_run_id()

# Ordered figure names matching the build sequence in main():
#   1st figure built → "annotation"
#   2nd figure built → "comparison_strip"
_FIGURE_NAMES = ["annotation", "comparison_strip"]


def _build_annotation_figure(run_id: str) -> None:
    """Materialize the headline annotation figure into pyplot state."""
    state = _read_state(run_id)
    target = run_dir(run_id)
    label_map, rank_map, confidence_map = _read_proposal_maps(target)
    with _CACHE.with_adata(
        run_id=run_id, h5ad_path=state.h5ad_path, run_dir=target
    ) as adata:
        make_litchron_annotation_figure(
            adata=adata,
            run_dir=Path(target),
            label_map=label_map,
            rank_map=rank_map,
            confidence_map=confidence_map,
        )


def _build_comparison_strip(run_id: str) -> None:
    """Materialize the pseudotime comparison strip into pyplot state.

    Soft-skips (with a printed warning) when the LitChron pseudotime parquet
    or baseline parquets are absent — this figure is only generated after
    ``compute_litchron_pseudotime`` and at least one ``run_baseline`` call.
    """
    import pandas as pd  # local — matches tools.py pattern

    state = _read_state(run_id)
    target = Path(run_dir(run_id))

    lit_path = target / "litchron_pseudotime.parquet"
    if not lit_path.exists():
        print(
            "comparison strip skipped: no baseline parquets"
            f" (missing {lit_path.name})"
        )
        return

    lit_df = pd.read_parquet(lit_path)
    if "cell_id" not in lit_df.columns or "pseudotime" not in lit_df.columns:
        print(
            "comparison strip skipped: litchron_pseudotime.parquet missing"
            " required columns (cell_id, pseudotime)"
        )
        return
    llm_pt = lit_df.set_index("cell_id")["pseudotime"].astype("float64")

    baseline_pts: dict[str, "pd.Series"] = {}
    baselines_dir = target / "baselines"
    if baselines_dir.is_dir():
        for method_dir in sorted(baselines_dir.iterdir()):
            if not method_dir.is_dir():
                continue
            ord_path = method_dir / "ordering.parquet"
            if not ord_path.exists():
                continue
            try:
                df = pd.read_parquet(ord_path)
            except Exception:  # noqa: BLE001
                continue
            if "cell_id" not in df.columns or "pseudotime" not in df.columns:
                continue
            s = df.set_index("cell_id")["pseudotime"].astype("float64")
            baseline_pts[method_dir.name] = s

    if not baseline_pts:
        print("comparison strip skipped: no baseline parquets found in run_dir/baselines/")
        return

    with _CACHE.with_adata(
        run_id=run_id, h5ad_path=state.h5ad_path, run_dir=target
    ) as adata:
        make_pseudotime_comparison_strip(
            adata=adata,
            llm_pt=llm_pt,
            baseline_pts=baseline_pts,
            run_dir=target,
        )


def _audit_open_figures() -> list[dict]:
    """Run scivcd on every figure currently in pyplot state."""
    issues: list[dict] = []
    for fig_num in plt.get_fignums():
        fig = plt.figure(fig_num)
        for issue in detect_all_conflicts(fig, verbose=False, profile="auto"):
            issue.setdefault("figure_num", fig_num)
            issues.append(issue)
    return issues


def _filter_ignored(issues: list[dict]) -> tuple[list[dict], int]:
    """Filter findings by IGNORED_CHECK_TYPES (currently empty — no-op).

    Previously this suppressed font_family_violation. Those findings now live
    in baseline.json as pre-existing; IGNORED_CHECK_TYPES is kept for future
    use but must not be populated without a documented justification.
    """
    kept = [i for i in issues if i.get("type") not in IGNORED_CHECK_TYPES]
    return kept, len(issues) - len(kept)


def _summarize(issues: list[dict]) -> Counter:
    return Counter(str(i.get("severity_level", "INFO")) for i in issues)


_VOLATILE_ADDR_RE = re.compile(r"0x[0-9a-fA-F]+")


def _normalize_detail(detail: str) -> str:
    """Remove volatile memory addresses (e.g. 'ax@0x7f...') from detail strings.

    scivcd embeds matplotlib object repr strings that include heap addresses;
    these change between runs and must be stripped before keying for delta
    comparison.
    """
    return _VOLATILE_ADDR_RE.sub("0x<addr>", detail)


def _finding_key(finding: dict) -> tuple[str, str, str]:
    """Stable match key for a finding: (severity_level, type, normalized-detail[:80]).

    detail is normalized to strip volatile memory addresses before slicing,
    ensuring the same logical finding matches across runs even when matplotlib
    object repr strings change.
    """
    detail = _normalize_detail(str(finding.get("detail", "")))
    return (
        str(finding.get("severity_level", "")),
        str(finding.get("type", "")),
        detail[:80],
    )


# Schema version for findings.json envelope. Bump when the on-disk shape
# changes in a way that's not backward-compatible with prior readers.
FINDINGS_SCHEMA_VERSION = "1"


def _make_findings_envelope(
    run_id: str,
    figure_name: str,
    audit_source: str,
    issues: list[dict],
) -> dict:
    """Wrap a list of scivcd findings in the canonical on-disk envelope.

    Both the CI path (this script) and the skill path (/audit-litchron-run via
    the plugin) MUST write this exact shape so consumers can read either
    without branching. ``audit_source`` is the only field that differs:
    ``"ci"`` here, ``"skill"`` from the orchestration skill.
    """
    counts = Counter(str(i.get("severity_level", "INFO")) for i in issues)
    return {
        "schema_version": FINDINGS_SCHEMA_VERSION,
        "run_id": run_id,
        "figure_name": figure_name,
        "audit_source": audit_source,
        "counts": {
            "CRITICAL": counts.get("CRITICAL", 0),
            "MAJOR": counts.get("MAJOR", 0),
            "MINOR": counts.get("MINOR", 0),
            "INFO": counts.get("INFO", 0),
        },
        "issues": issues,
    }


def _write_audit_artifacts(
    run_id: str,
    findings_by_figure: dict[str, list[dict]],
) -> dict[str, list[dict]]:
    """Write per-figure findings JSON and maintain baseline.json.

    Returns the loaded (or freshly written) baseline dict so the caller can
    compute the delta.

    Per-figure findings.json files use the canonical envelope (see
    ``_make_findings_envelope``). baseline.json keeps its own flat-by-figure
    shape (``{figure_name: [issues]}``) because it's a multi-figure record
    used only by the delta calculator, not by external consumers.
    """
    audit_dir = run_dir(run_id) / "audit"
    audit_dir.mkdir(parents=True, exist_ok=True)

    # Write per-figure findings files in the canonical envelope shape.
    for figure_name, findings in findings_by_figure.items():
        envelope = _make_findings_envelope(
            run_id=run_id,
            figure_name=figure_name,
            audit_source="ci",
            issues=findings,
        )
        findings_path = audit_dir / f"{figure_name}.findings.json"
        findings_path.write_text(
            json.dumps(envelope, indent=2, default=str), encoding="utf-8"
        )

    baseline_path = audit_dir / "baseline.json"
    if baseline_path.exists():
        try:
            baseline: dict[str, list[dict]] = json.loads(
                baseline_path.read_text(encoding="utf-8")
            )
        except (json.JSONDecodeError, OSError):
            baseline = {}
    else:
        # First run: bless current findings as the baseline.
        baseline = {}

    if not baseline:
        # Write initial baseline so subsequent runs are delta-relative.
        baseline_path.write_text(
            json.dumps(findings_by_figure, indent=2, default=str), encoding="utf-8"
        )
        return {}  # empty dict signals "no prior baseline" to the caller

    # Update baseline with any figure names that are new (e.g. comparison_strip
    # added later). Existing entries are *not* overwritten — they persist until
    # the author explicitly re-blesses via git add.
    updated = False
    for figure_name, findings in findings_by_figure.items():
        if figure_name not in baseline:
            baseline[figure_name] = findings
            updated = True

    if updated:
        baseline_path.write_text(
            json.dumps(baseline, indent=2, default=str), encoding="utf-8"
        )

    return baseline


def _compute_delta(
    findings_by_figure: dict[str, list[dict]],
    baseline: dict[str, list[dict]],
) -> dict[str, dict[str, list[dict]]]:
    """Compute new/removed findings per figure vs the saved baseline.

    Match is by (severity_level, type, detail[:80]) tuple.
    Returns {figure_name: {"new": [...], "removed": [...]}} for all figures.
    """
    delta: dict[str, dict[str, list[dict]]] = {}
    all_figures = set(findings_by_figure) | set(baseline)
    for figure_name in all_figures:
        current = findings_by_figure.get(figure_name, [])
        saved = baseline.get(figure_name, [])
        current_keys = [_finding_key(f) for f in current]
        saved_keys = [_finding_key(f) for f in saved]
        new_findings = [f for f, k in zip(current, current_keys) if k not in saved_keys]
        removed_findings = [f for f, k in zip(saved, saved_keys) if k not in current_keys]
        delta[figure_name] = {"new": new_findings, "removed": removed_findings}
    return delta


def _print_report(
    issues: list[dict],
    suppressed: int,
    summary: Counter,
    delta: dict[str, dict[str, list[dict]]] | None,
    first_run: bool,
    top_n: int = 15,
) -> None:
    print("== LitChron headline figure audit ==")
    print(
        f"CRITICAL={summary.get('CRITICAL', 0)} "
        f"MAJOR={summary.get('MAJOR', 0)} "
        f"MINOR={summary.get('MINOR', 0)} "
        f"INFO={summary.get('INFO', 0)} "
        f"(suppressed via IGNORED_CHECK_TYPES: {suppressed})"
    )

    if first_run:
        print("  [first run] baseline.json written — git add audit/baseline.json to bless")
    elif delta is not None:
        total_new = sum(len(v["new"]) for v in delta.values())
        total_removed = sum(len(v["removed"]) for v in delta.values())
        new_critical = sum(
            1
            for v in delta.values()
            for f in v["new"]
            if str(f.get("severity_level")) == "CRITICAL"
        )
        new_major = sum(
            1
            for v in delta.values()
            for f in v["new"]
            if str(f.get("severity_level")) == "MAJOR"
        )
        print(
            f"  baseline delta: +{total_new} new / -{total_removed} removed "
            f"(new CRITICAL={new_critical}, new MAJOR={new_major})"
        )
        if total_new == 0 and total_removed == 0:
            print("  → no change vs baseline")

    if not issues:
        print("  → clean")
        return

    severity_rank = {"CRITICAL": 0, "MAJOR": 1, "MINOR": 2, "INFO": 3}

    # Group issues by figure number for clearer per-figure reporting.
    figures_by_num: dict[int, list[dict]] = {}
    for issue in issues:
        fig_num = int(issue.get("figure_num", 0))
        figures_by_num.setdefault(fig_num, []).append(issue)

    printed = 0
    for fig_num in sorted(figures_by_num):
        fig_issues = sorted(
            figures_by_num[fig_num],
            key=lambda i: severity_rank.get(str(i.get("severity_level")), 9),
        )
        fig_obj = plt.figure(fig_num) if fig_num in plt.get_fignums() else None
        fname = ""
        if fig_obj is not None:
            canvas = getattr(fig_obj, "canvas", None)
            raw = getattr(canvas, "get_window_title", lambda: "")()
            if raw:
                fname = f" ({raw})"
        print(f"  -- Figure {fig_num}{fname}: {len(fig_issues)} issue(s) --")
        for i in fig_issues:
            if printed >= top_n:
                break
            detail = str(i.get("detail", ""))[:120]
            print(f"    [{i.get('severity_level')}] {i.get('type')}: {detail}")
            printed += 1
        if printed >= top_n:
            remaining = len(issues) - printed
            if remaining > 0:
                print(f"  … and {remaining} more")
            break


def _partition_findings_by_figure(
    issues: list[dict], fig_nums_before_annotation: int
) -> dict[str, list[dict]]:
    """Map issues to figure names based on figure_num order.

    Figure numbers in pyplot are assigned sequentially starting from 1.
    We capture the count of existing figures before each _build_* call, then
    assign findings to figure names by whether their fig_num is in the range
    added by each builder. The ordering matches _FIGURE_NAMES.
    """
    findings_by_figure: dict[str, list[dict]] = {name: [] for name in _FIGURE_NAMES}
    for issue in issues:
        fig_num = int(issue.get("figure_num", 0))
        if fig_num <= fig_nums_before_annotation:
            # Belongs to annotation figure (first built).
            findings_by_figure["annotation"].append(issue)
        else:
            # Belongs to comparison_strip (second built).
            findings_by_figure["comparison_strip"].append(issue)
    return findings_by_figure


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Routine scivcd audit for LitChron's headline figures."
    )
    default_hint = (
        DEFAULT_RUN_ID
        if DEFAULT_RUN_ID is not None
        else "<auto-discover from $LITCHRON_AUDIT_RUN_ID or newest dir in runs/>"
    )
    parser.add_argument(
        "--run-id",
        default=DEFAULT_RUN_ID,
        help=(
            "litchron run id whose headline figure to audit "
            f"(default resolved at startup: {default_hint})"
        ),
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="exit with code 2 when NEW MAJOR findings exist vs baseline (default: only new CRITICAL fails)",
    )
    args = parser.parse_args(argv)

    if args.run_id is None:
        print(
            "ERROR: could not resolve a run_id. Set $LITCHRON_AUDIT_RUN_ID, "
            "pass --run-id <id>, or create a run under <project_root>/runs/.",
            file=sys.stderr,
        )
        return 3

    # Snapshot figure count before any builds so we can assign findings to
    # figure names by build order.
    fignums_before_annotation = max(plt.get_fignums(), default=0)

    try:
        _build_annotation_figure(args.run_id)
    except Exception as exc:  # noqa: BLE001
        print(f"FAILED to build annotation figure for run_id={args.run_id!r}: {exc}", file=sys.stderr)
        return 3

    fignums_after_annotation = max(plt.get_fignums(), default=0)

    try:
        _build_comparison_strip(args.run_id)
    except Exception as exc:  # noqa: BLE001
        # Comparison strip failure is a soft skip, not a driver failure.
        print(f"WARNING: comparison strip raised unexpectedly for run_id={args.run_id!r}: {exc}", file=sys.stderr)

    raw_issues = _audit_open_figures()
    issues, suppressed = _filter_ignored(raw_issues)
    summary = _summarize(issues)

    # Partition findings by figure name using build-order boundaries.
    findings_by_figure = _partition_findings_by_figure(issues, fignums_after_annotation)

    # Persist audit artifacts and get baseline (empty dict = first run).
    saved_baseline = _write_audit_artifacts(args.run_id, findings_by_figure)
    first_run = not saved_baseline

    if first_run:
        # No prior baseline: treat all findings as "new" for exit-code purposes.
        delta: dict[str, dict[str, list[dict]]] | None = None
        new_critical = summary.get("CRITICAL", 0)
        new_major = summary.get("MAJOR", 0)
    else:
        delta = _compute_delta(findings_by_figure, saved_baseline)
        new_critical = sum(
            1
            for v in delta.values()
            for f in v["new"]
            if str(f.get("severity_level")) == "CRITICAL"
        )
        new_major = sum(
            1
            for v in delta.values()
            for f in v["new"]
            if str(f.get("severity_level")) == "MAJOR"
        )

    _print_report(issues, suppressed, summary, delta, first_run)

    # Restore plt.close so a caller importing this module doesn't get burned.
    plt.close = _ORIGINAL_CLOSE  # type: ignore[assignment]

    # Exit codes are delta-relative, not absolute-ceiling.
    if new_critical > 0:
        return 1
    if args.strict and new_major > 0:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
