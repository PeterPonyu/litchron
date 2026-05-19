"""Report assembly: markdown sections → unified markdown → LaTeX → PDF.

This module wires three filesystem-touching primitives together:

1. :func:`assemble_markdown` concatenates per-section markdown files into a
   single ``runs/<run_id>/report.md`` with ``# Section Title`` headers.
2. :func:`markdown_to_tex` invokes ``pandoc`` to convert a sanitized
   markdown file into a LaTeX fragment placed under ``<out_dir>/<stem>.tex``.
3. :func:`compile_pdf` orchestrates 1+2 across the four canonical sections
   (observations, proposals, baselines, comparison) and then spawns
   ``latexmk`` to produce the final PDF.

All three functions are side-effecting and document their effects in their
docstrings.
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from litchron.sanitize import sanitize_markdown
from mcp_litchron.errors import LitchronError, PreflightFailure

# ---------------------------------------------------------------------------
# Section order
# ---------------------------------------------------------------------------
_SECTION_FILES: tuple[tuple[str, str], ...] = (
    ("observations.md", "Observations"),
    ("annotation.md", "LitChron Annotation"),
    # Bibliography is rendered by biblatex \printbibliography directly after
    # the annotation figure section (see tex/litchron.tex).
    ("proposals.md", "LLM Proposal"),
    ("litchron_ordering.md", "LitChron LLM Pseudotime"),
    # Classical baselines run when configured but are NOT included in the
    # report — their ordering.parquet / plot.png / log.txt stay on disk for
    # external inspection. The headline product is the LitChron annotation
    # figure; baselines are sanity checks only.
)


def _strip_yaml_frontmatter(text: str) -> str:
    """Strip a leading ``---`` YAML front-matter block from a markdown string.

    The front matter is a machine-readable handshake between the renderer
    and the comparison parser; rendering it again in the final PDF is
    redundant noise. Only the leading block (starting at byte 0) is
    stripped; trailing YAML inside the body is left alone.
    """
    if not text.startswith("---"):
        return text
    end = text.find("\n---", 3)
    if end < 0:
        return text
    after = text[end + 4:]
    return after.lstrip("\n")


# ---------------------------------------------------------------------------
# 1. Concatenate per-section markdown
# ---------------------------------------------------------------------------
def _auto_generate_baselines_md(run_dir: Path) -> None:
    """Synthesize ``<run_dir>/baselines.md`` from the baselines/ directory.

    For each ``baselines/<method>/`` that has a ``plot.png``, emit a
    subsection with a markdown figure reference. Image paths are relative
    to ``run_dir`` so latexmk (which runs with ``-output-directory=run_dir``)
    resolves them correctly.

    Overwrites any existing ``baselines.md``. Called by :func:`assemble_markdown`
    each time so the figure list stays in sync with the on-disk baselines.
    """
    bdir = run_dir / "baselines"
    if not bdir.exists():
        return
    methods = sorted(
        p.name for p in bdir.iterdir()
        if p.is_dir() and (p / "plot.png").exists()
    )
    if not methods:
        return

    lines: list[str] = ["# Baselines", ""]
    lines.append(
        "Classical trajectory inference methods run as numerical comparators "
        "to the LLM proposal. Each baseline writes a `ordering.parquet`, a "
        "`plot.png` visualization, and a `log.txt`. Figures below; quantitative "
        "comparison appears in the next section."
    )
    lines.append("")
    for method in methods:
        rel_png = f"baselines/{method}/plot.png"
        log_path = bdir / method / "log.txt"
        log_excerpt = ""
        if log_path.exists():
            try:
                log_excerpt = log_path.read_text().strip().splitlines()[-1][:200]
            except OSError:
                log_excerpt = ""
        lines.append(f"## {method}")
        lines.append("")
        lines.append(f"![{method} trajectory]({rel_png})")
        lines.append("")
        if log_excerpt:
            lines.append(f"*Last log line:* `{log_excerpt}`")
            lines.append("")
    (run_dir / "baselines.md").write_text("\n".join(lines))


def _auto_generate_annotation_md(run_dir: Path) -> None:
    """If a LitChron annotation figure exists, synthesize annotation.md.

    The figure is the headline visual claim of LitChron: UMAP overlays
    showing leiden clusters, LLM biological labels, the LLM-derived
    continuous pseudotime, and a marker dotplot per cluster. The
    accompanying markdown points the PDF at the rendered PNG.
    """
    fig_path = run_dir / "figures" / "litchron_annotation.png"
    if not fig_path.exists():
        return
    lines: list[str] = [
        "# LitChron Annotation",
        "",
        "Single composed UMAP panel. Cells are colored by the LLM-inferred "
        "pseudotime rank of their cluster (discrete `tab20` palette, ordered "
        "by rank). Cluster IDs are overlaid at cluster centroids as circled "
        "labels. The legend on the right reads "
        "`rank R, cluster C - biological identity (n cells)` and is sorted "
        "by inferred pseudotime rank. The bottom strip shows the discrete "
        "palette ordered left-to-right by rank, with the rank number printed "
        "on each swatch.",
        "",
        "![LitChron annotation figure](figures/litchron_annotation.png)",
        "",
    ]
    strip = run_dir / "figures" / "pseudotime_comparison.png"
    if strip.exists():
        lines.extend([
            "## Pseudotime comparison (LitChron vs classical baselines)",
            "",
            "![Pseudotime comparison strip](figures/pseudotime_comparison.png)",
            "",
        ])
    (run_dir / "annotation.md").write_text("\n".join(lines))


def _auto_generate_litchron_ordering_md(run_dir: Path) -> None:
    """Summarize <run_dir>/litchron_pseudotime.parquet in markdown."""
    pq = run_dir / "litchron_pseudotime.parquet"
    if not pq.exists():
        return
    try:
        import pandas as pd  # local
        df = pd.read_parquet(pq)
    except Exception:  # noqa: BLE001
        return
    lines: list[str] = [
        "# LitChron LLM Pseudotime",
        "",
        "LitChron's primary trajectory output is a per-cell continuous "
        "pseudotime derived entirely from the LLM-proposed per-cluster "
        "ranking plus a small intra-cluster spread based on diffmap "
        "component 1. Classical methods (PAGA, scVelo) run only as "
        "sanity-check comparators.",
        "",
        f"- Cells with assigned pseudotime: **{len(df):,}**",
    ]
    if "pseudotime" in df.columns:
        lines.append(f"- Pseudotime range: **{df['pseudotime'].min():.3f} – {df['pseudotime'].max():.3f}**")
    if "cell_type_label" in df.columns:
        per_label = (
            df.groupby("cell_type_label")["pseudotime"].mean().sort_values()
        )
        lines.append("")
        lines.append("## Mean LitChron pseudotime per LLM-assigned cell type")
        lines.append("")
        lines.append("| Cell type | Mean pseudotime | n cells |")
        lines.append("|:---|---:|---:|")
        counts = df.groupby("cell_type_label").size()
        for label, mean in per_label.items():
            lines.append(f"| {label} | {mean:.3f} | {int(counts[label]):,} |")
        lines.append("")
    (run_dir / "litchron_ordering.md").write_text("\n".join(lines))


def assemble_markdown(run_dir: Path) -> Path:
    """Concatenate per-section markdown into ``<run_dir>/report.md``.

    Side effects
    ------------
    * Regenerates ``baselines.md``, ``annotation.md``, and
      ``litchron_ordering.md`` from on-disk artifacts before concatenation.
    * Writes ``<run_dir>/report.md`` (overwrites).
    * Strips leading YAML front-matter blocks from each section before
      writing so the same machine-readable handshake doesn't appear in
      the human-readable PDF.

    Missing sections are silently skipped so partial reports are still
    buildable.
    """
    run_dir = Path(run_dir)
    _auto_generate_annotation_md(run_dir)
    _auto_generate_litchron_ordering_md(run_dir)
    _auto_generate_baselines_md(run_dir)
    report_path = run_dir / "report.md"
    parts: list[str] = []
    for fname, title in _SECTION_FILES:
        src = run_dir / fname
        if not src.exists():
            continue
        body = _strip_yaml_frontmatter(src.read_text())
        # If section already has its own H1, don't add a wrapper title.
        if not body.lstrip().startswith("# "):
            parts.append(f"# {title}\n\n")
        parts.append(body)
        if not body.endswith("\n"):
            parts.append("\n")
        parts.append("\n")
    report_path.write_text("".join(parts))
    return report_path


# ---------------------------------------------------------------------------
# 2. markdown → tex via pandoc
# ---------------------------------------------------------------------------
def markdown_to_tex(md_path: Path, out_dir: Path) -> Path:
    """Convert a markdown file to a LaTeX fragment via pandoc.

    Side effects
    ------------
    * Writes ``<out_dir>/<stem>.tex``.
    * Spawns ``pandoc`` as a subprocess.

    Raises
    ------
    PreflightFailure
        If ``pandoc`` is not on ``PATH``.
    LitchronError
        If pandoc returns a non-zero exit code.
    """
    md_path = Path(md_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    from litchron.preflight import _which_with_env_bin
    pandoc = _which_with_env_bin("pandoc")
    if pandoc is None:
        raise PreflightFailure(
            code="pandoc_missing",
            message="pandoc binary not found on PATH",
            hint="Install pandoc (e.g. `apt install pandoc` or `conda install -c conda-forge pandoc`)",
            retryable=False,
        )

    raw = md_path.read_text()
    sanitized = sanitize_markdown(raw)

    out_path = out_dir / f"{md_path.stem}.tex"
    proc = subprocess.run(
        [
            pandoc,
            "--from",
            "gfm+tex_math_dollars",
            "--to",
            "latex",
            "-o",
            str(out_path),
        ],
        input=sanitized,
        text=True,
        capture_output=True,
    )
    if proc.returncode != 0:
        raise LitchronError(
            code="pandoc_failed",
            message=f"pandoc returned {proc.returncode} for {md_path.name}",
            hint=proc.stderr.strip()[:500] or "see pandoc stderr",
            retryable=False,
        )
    return out_path


# ---------------------------------------------------------------------------
# 3. Full PDF compile via latexmk
# ---------------------------------------------------------------------------
def compile_pdf(run_dir: Path, project_root: Path, run_id: str) -> Path:
    """Drive the full markdown→tex→PDF compile for one run.

    Side effects
    ------------
    * Writes ``<run_dir>/report.md`` (via :func:`assemble_markdown`).
    * Writes ``<run_dir>/tex_sections/<section>.tex`` for each section
      whose markdown exists.
    * Spawns ``latexmk``; resulting PDF lives at ``<run_dir>/litchron.pdf``,
      which this function renames to ``<run_dir>/report.pdf`` for spec
      compliance and returns.

    Raises
    ------
    LitchronError
        With code ``"latex_compile_failed"`` if ``latexmk`` returns non-zero.
        The ``hint`` field points to the latexmk log file.
    PreflightFailure
        If ``latexmk`` or ``pandoc`` is missing.
    """
    from litchron._runtime import validate_run_id

    validate_run_id(run_id)  # defense-in-depth: TeX macro safety
    run_dir = Path(run_dir)
    project_root = Path(project_root)

    # Phase 1: assemble the unified markdown (used as a side-artifact + the
    # source for pandoc per-section conversion).
    assemble_markdown(run_dir)

    # Phase 2: per-section markdown → tex
    tex_sections_dir = run_dir / "tex_sections"
    tex_sections_dir.mkdir(parents=True, exist_ok=True)
    for fname, _title in _SECTION_FILES:
        md_path = run_dir / fname
        if not md_path.exists():
            continue
        markdown_to_tex(md_path, tex_sections_dir)

    # Phase 3: latexmk
    from litchron.preflight import _which_with_env_bin
    latexmk = _which_with_env_bin("latexmk")
    if latexmk is None:
        raise PreflightFailure(
            code="latexmk_missing",
            message="latexmk binary not found on PATH",
            hint="Install texlive (e.g. `apt install texlive-latex-extra latexmk`)",
            retryable=False,
        )

    tex_main = project_root / "tex" / "litchron.tex"
    if not tex_main.exists():
        raise PreflightFailure(
            code="tex_template_missing",
            message=f"LaTeX template not found at {tex_main}",
            hint="Ensure the project_root points at the LitChron repo root containing tex/litchron.tex",
            retryable=False,
        )

    # \runDir is consumed by the LaTeX template to locate per-run inputs;
    # see Plan §3 Phase 3. No trailing slash to avoid double-slash on input.
    pretex = rf"\def\runDir{{runs/{run_id}}}"
    # -lualatex instead of -pdf (pdflatex): lualatex handles UTF-8 natively,
    # so the LLM-authored markdown can contain math symbols (≥, ±, α, …)
    # without needing per-character \DeclareUnicodeCharacter directives.
    cmd = [
        latexmk,
        "-lualatex",
        "-interaction=nonstopmode",
        "-halt-on-error",
        f"-output-directory={run_dir}",
        f"-usepretex={pretex}",
        str(tex_main),
    ]
    proc = subprocess.run(
        cmd,
        cwd=str(project_root),
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        # latexmk writes a .log next to the output; surface its path so the
        # LLM can request it directly via filesystem inspection.
        log_path = run_dir / "litchron.log"
        raise LitchronError(
            code="latex_compile_failed",
            message=f"latexmk returned {proc.returncode} for run {run_id}",
            hint=str(log_path) if log_path.exists() else proc.stderr.strip()[:500],
            retryable=False,
        )

    produced = run_dir / "litchron.pdf"
    final = run_dir / "report.pdf"
    if produced.exists():
        produced.replace(final)
    elif not final.exists():
        raise LitchronError(
            code="latex_compile_failed",
            message="latexmk exited 0 but no PDF was produced",
            hint=str(run_dir / "litchron.log"),
            retryable=False,
        )
    return final
