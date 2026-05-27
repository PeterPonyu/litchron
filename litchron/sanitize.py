"""Sanitize LLM-authored markdown for safe pandoc -> LaTeX conversion.

Pandoc already handles LaTeX-special characters (``&``, ``%``, ``#``,
``$``, ``_``, ``{``, ``}``, ``~``, ``^``) correctly when converting
markdown to LaTeX -- it escapes them in body text and leaves them as
markdown syntax (e.g., ``#`` for headings, ``%`` as literal percent)
where the markdown semantics require it.

Pre-escaping at the markdown layer is harmful: ``# Heading`` would
become ``\\# Heading`` which pandoc emits as the literal string
``\\# Heading`` rather than a section. ``50%`` would become ``50\\%``
which pandoc passes through as a literal backslash-percent in the PDF.

So sanitize_markdown does two things:

1. Normalize Unicode dash variants to ASCII ``-`` for stable typesetting
   (outside fenced code blocks; fenced content is preserved verbatim).
2. **Reject unsafe raw TeX constructs.** Pandoc consumes the markdown via
   ``--from gfm+tex_math_dollars --to latex`` and passes raw TeX through
   to ``latexmk``; without explicit filtering, an LLM-emitted
   ``\\input{/etc/passwd}`` or ``\\write18{...}`` would flow straight to
   the LaTeX compiler. :func:`sanitize_markdown` scans the markdown
   *outside fenced code blocks* and raises :class:`UnsafeTexError` if
   any of the disallowed control sequences appears.

Allowed Markdown / TeX subset
-----------------------------
- All standard GFM markdown: headings, lists, tables, links, emphasis,
  bold, inline code, fenced code blocks, blockquotes.
- TeX math via ``$ ... $`` (inline) and ``$$ ... $$`` (display), which
  is the purpose of the ``tex_math_dollars`` pandoc extension.
- Unicode dashes (normalized to ASCII ``-``).

Blocked TeX constructs (outside fenced code blocks)
---------------------------------------------------
The following control sequences are *always* unsafe because they read
or write to the host filesystem, redefine TeX semantics, or shell out:

- ``\\input``, ``\\include``, ``\\InputIfFileExists`` -- include arbitrary
  file paths into the compilation.
- ``\\write``, ``\\openout``, ``\\openin``, ``\\read``, ``\\newwrite``,
  ``\\closeout``, ``\\closein``, ``\\immediate`` -- TeX I/O primitives.
  ``\\write18{...}`` (with ``-shell-escape``) executes arbitrary shell
  commands.
- ``\\catcode``, ``\\def``, ``\\edef``, ``\\gdef``, ``\\xdef``,
  ``\\let`` -- redefine the meaning of characters or commands.
- ``\\usepackage``, ``\\RequirePackage`` -- load arbitrary LaTeX
  packages (and run their setup code).

Inside fenced code blocks these strings are still allowed (they are
prose / code examples and pandoc emits them as verbatim, not as TeX).
Math-mode ``$ ... $`` / ``$$ ... $$`` does not match the blocked list.
"""
from __future__ import annotations

import re

from mcp_litchron.errors import LitchronError

# Unicode dash variants we collapse to ASCII "-".
_DASH_MAP = str.maketrans(
    {
        "‐": "-",  # hyphen
        "‑": "-",  # non-breaking hyphen
        "‒": "-",  # figure dash
        "–": "-",  # en dash
        "—": "-",  # em dash
        "―": "-",  # horizontal bar
        "−": "-",  # minus sign
    }
)


class UnsafeTexError(LitchronError):
    """Raised when LLM-authored markdown contains a forbidden TeX construct.

    The MCP tool layer catches this (via the base :class:`LitchronError`
    handler in :mod:`mcp_litchron.tools`) and surfaces it to the LLM as a
    structured :class:`ErrorResult` with code ``"unsafe_tex"``.
    """


# Disallowed TeX control sequences. Each pattern uses ``(?![A-Za-z])``
# after the name so e.g. ``\input`` blocks both ``\input{...}`` and the
# bare ``\input ``, but a hypothetical safe ``\inputtable`` (not in our
# block list) would not match.
_BLOCKED_TEX_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("\\input", re.compile(r"\\input(?![A-Za-z])")),
    ("\\include", re.compile(r"\\include(?![A-Za-z])")),
    ("\\InputIfFileExists", re.compile(r"\\InputIfFileExists(?![A-Za-z])")),
    ("\\write", re.compile(r"\\write(?![A-Za-z])")),
    ("\\openout", re.compile(r"\\openout(?![A-Za-z])")),
    ("\\openin", re.compile(r"\\openin(?![A-Za-z])")),
    ("\\read", re.compile(r"\\read(?![A-Za-z])")),
    ("\\newwrite", re.compile(r"\\newwrite(?![A-Za-z])")),
    ("\\closeout", re.compile(r"\\closeout(?![A-Za-z])")),
    ("\\closein", re.compile(r"\\closein(?![A-Za-z])")),
    ("\\immediate", re.compile(r"\\immediate(?![A-Za-z])")),
    ("\\catcode", re.compile(r"\\catcode(?![A-Za-z])")),
    ("\\def", re.compile(r"\\def(?![A-Za-z])")),
    ("\\edef", re.compile(r"\\edef(?![A-Za-z])")),
    ("\\gdef", re.compile(r"\\gdef(?![A-Za-z])")),
    ("\\xdef", re.compile(r"\\xdef(?![A-Za-z])")),
    ("\\let", re.compile(r"\\let(?![A-Za-z])")),
    ("\\usepackage", re.compile(r"\\usepackage(?![A-Za-z])")),
    ("\\RequirePackage", re.compile(r"\\RequirePackage(?![A-Za-z])")),
)


def _find_blocked_tex(line: str) -> str | None:
    """Return the canonical name of the first blocked TeX construct, or ``None``."""
    for canonical, pattern in _BLOCKED_TEX_PATTERNS:
        if pattern.search(line):
            return canonical
    return None


def sanitize_markdown(md: str) -> str:
    """Normalize Unicode dashes and reject unsafe raw TeX outside code blocks.

    State machine over lines: a line whose stripped form starts with
    ``` toggles fenced-mode. Lines inside a fence are emitted verbatim.
    Lines outside a fence have Unicode dashes collapsed to ``-`` and are
    scanned for forbidden TeX control sequences (see module docstring
    for the full list). Any match raises :class:`UnsafeTexError`.
    """
    if not md:
        return md

    out: list[str] = []
    in_fence = False

    for lineno, line in enumerate(md.splitlines(keepends=True), start=1):
        stripped = line.lstrip()
        if stripped.startswith("```"):
            in_fence = not in_fence
            out.append(line)
            continue
        if in_fence:
            out.append(line)
            continue

        blocked = _find_blocked_tex(line)
        if blocked is not None:
            raise UnsafeTexError(
                code="unsafe_tex",
                message=(
                    f"line {lineno}: LLM-authored markdown contains the "
                    f"disallowed TeX control sequence {blocked!r}. "
                    "Raw TeX I/O / shell-escape / package-loading "
                    "primitives are blocked because the markdown flows "
                    "through pandoc into latexmk; an unfiltered "
                    "\\input{} or \\write18{} would read or execute "
                    "arbitrary files."
                ),
                hint=(
                    "Remove the raw TeX command. If you need math, use "
                    "$ ... $ or $$ ... $$. If you need to display the "
                    "literal text of the command, wrap it in a fenced "
                    "code block (```...```)."
                ),
                retryable=False,
            )

        out.append(line.translate(_DASH_MAP))

    return "".join(out)


__all__ = [
    "sanitize_markdown",
    "UnsafeTexError",
]
