"""Tests for litchron.sanitize.sanitize_markdown.

After the v0.1.1 rewrite, sanitize only normalizes Unicode dashes —
pandoc handles LaTeX-special characters correctly during conversion.
Pre-escaping ``#``/``%``/``&`` at the markdown layer corrupts markdown
syntax (in particular section headings).
"""
from __future__ import annotations

from litchron.sanitize import sanitize_markdown


# ---------------------------------------------------------------------------
# Em / en dash normalization (the only behavior sanitize retains)
# ---------------------------------------------------------------------------

def test_em_dash_normalized_to_ascii_hyphen() -> None:
    result = sanitize_markdown("A—B\n")
    assert "—" not in result
    assert "A-B" in result


def test_en_dash_normalized() -> None:
    result = sanitize_markdown("2020–2021\n")
    assert "–" not in result
    assert "2020-2021" in result


def test_minus_sign_normalized() -> None:
    result = sanitize_markdown("x − y\n")
    assert "−" not in result
    assert "x - y" in result


# ---------------------------------------------------------------------------
# Markdown syntax must survive untouched (this is the regression fix)
# ---------------------------------------------------------------------------

def test_heading_hash_preserved() -> None:
    result = sanitize_markdown("# Observations\n")
    assert result == "# Observations\n"


def test_subheading_hash_preserved() -> None:
    result = sanitize_markdown("## Sub\n")
    assert "## Sub" in result
    assert "\\#" not in result


def test_ampersand_preserved_in_body() -> None:
    result = sanitize_markdown("Wilson & Trumpp 2008\n")
    assert "Wilson & Trumpp" in result
    assert "\\&" not in result


def test_percent_preserved_in_body() -> None:
    result = sanitize_markdown("100% coverage\n")
    assert "100%" in result
    assert "\\%" not in result


# ---------------------------------------------------------------------------
# Fenced code blocks untouched
# ---------------------------------------------------------------------------

def test_dash_in_fence_preserved() -> None:
    md = "```\nA—B\n```\n"
    result = sanitize_markdown(md)
    assert "—" in result  # em dash preserved inside fence


def test_hash_in_fence_preserved() -> None:
    md = "```python\n# comment\n```\n"
    result = sanitize_markdown(md)
    assert "# comment" in result


def test_mixed_dash_normalization() -> None:
    md = "Before — fence.\n\n```python\n# c—d\n```\n\nAfter – done.\n"
    result = sanitize_markdown(md)
    assert "Before - fence." in result
    assert "After - done." in result
    assert "c—d" in result  # inside fence preserved


# ---------------------------------------------------------------------------
# Empty / no-op
# ---------------------------------------------------------------------------

def test_empty_string_returns_empty() -> None:
    assert sanitize_markdown("") == ""


def test_plain_text_unchanged() -> None:
    md = "Hello world\n"
    assert sanitize_markdown(md) == "Hello world\n"
