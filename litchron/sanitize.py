"""Sanitize LLM-authored markdown for safe pandoc -> LaTeX conversion.

Pandoc already handles LaTeX-special characters (``&``, ``%``, ``#``,
``$``, ``_``, ``{``, ``}``, ``~``, ``^``) correctly when converting
markdown to LaTeX — it escapes them in body text and leaves them as
markdown syntax (e.g., ``#`` for headings, ``%`` as literal percent)
where the markdown semantics require it.

Pre-escaping at the markdown layer is harmful: ``# Heading`` would
become ``\\# Heading`` which pandoc emits as the literal string
``\\# Heading`` rather than a section. ``50%`` would become ``50\\%``
which pandoc passes through as a literal backslash-percent in the PDF.

So sanitize_markdown does only one thing: normalize Unicode dash
variants to ASCII ``-`` for stable typesetting. Code blocks are left
fully verbatim.
"""
from __future__ import annotations

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


def sanitize_markdown(md: str) -> str:
    """Normalize Unicode dashes outside fenced code blocks.

    State machine over lines: a line whose stripped form starts with
    ``` toggles fenced-mode. Lines inside a fence are emitted verbatim.
    Lines outside a fence have Unicode dashes collapsed to ``-``.
    """
    if not md:
        return md

    out: list[str] = []
    in_fence = False

    for line in md.splitlines(keepends=True):
        stripped = line.lstrip()
        if stripped.startswith("```"):
            in_fence = not in_fence
            out.append(line)
            continue
        if in_fence:
            out.append(line)
        else:
            out.append(line.translate(_DASH_MAP))

    return "".join(out)
