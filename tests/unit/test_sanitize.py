"""Tests for litchron.sanitize.sanitize_markdown.

After the v0.1.1 rewrite, sanitize normalizes Unicode dashes (so
typesetting is stable) and rejects unsafe raw TeX constructs (so an
LLM-emitted ``\\input{/etc/passwd}`` cannot flow through pandoc into
``latexmk``). Pandoc handles LaTeX-special characters (``#``/``%``/
``&``) correctly during conversion; pre-escaping them at the markdown
layer would corrupt markdown syntax.
"""
from __future__ import annotations

import pytest

from litchron.sanitize import UnsafeTexError, sanitize_markdown


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


# ---------------------------------------------------------------------------
# Unsafe raw TeX rejection (one adversarial test per blocked construct)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "construct, payload",
    [
        ("\\input", "Read this: \\input{/etc/passwd}\n"),
        ("\\include", "Pulled in: \\include{secrets}\n"),
        (
            "\\InputIfFileExists",
            "\\InputIfFileExists{evil.tex}{}{}\n",
        ),
        ("\\write", "\\write18{rm -rf ~}\n"),
        ("\\openout", "\\openout15=output.txt\n"),
        ("\\openin", "\\openin0=secret\n"),
        ("\\read", "\\read0 to \\linex\n"),
        ("\\newwrite", "\\newwrite\\out\n"),
        ("\\closeout", "\\closeout15\n"),
        ("\\closein", "\\closein0\n"),
        # ``\immediate`` is paired with another I/O primitive in real TeX,
        # but for this test we use a standalone token so the assertion can
        # check that the construct is identified correctly even when it
        # appears first in the line.
        ("\\immediate", "Prefix \\immediate suffix\n"),
        ("\\catcode", "\\catcode`\\@=11\n"),
        ("\\def", "\\def\\evil#1{#1#1}\n"),
        ("\\edef", "\\edef\\x{\\y}\n"),
        ("\\gdef", "\\gdef\\global{...}\n"),
        ("\\xdef", "\\xdef\\x{\\y}\n"),
        ("\\let", "\\let\\foo\\bar\n"),
        ("\\usepackage", "\\usepackage{hyperref}\n"),
        ("\\RequirePackage", "\\RequirePackage{tikz}\n"),
    ],
)
def test_blocked_tex_construct_raises(construct: str, payload: str) -> None:
    """Each blocked control sequence triggers UnsafeTexError outside code blocks."""
    with pytest.raises(UnsafeTexError) as exc_info:
        sanitize_markdown(payload)
    err = exc_info.value
    assert err.code == "unsafe_tex"
    # The message must name the offending construct so the LLM knows what
    # to remove from its draft.
    assert construct in err.message
    # And the error must be marked non-retryable.
    assert err.retryable is False


def test_unsafe_tex_inside_fenced_code_block_is_allowed() -> None:
    """Raw TeX inside ``` ... ``` is prose / example and must pass through."""
    md = (
        "Here is an example of what NOT to write:\n\n"
        "```latex\n"
        "\\input{evil.tex}\n"
        "\\write18{rm -rf ~}\n"
        "```\n"
    )
    # Should not raise.
    out = sanitize_markdown(md)
    assert "\\input{evil.tex}" in out
    assert "\\write18{rm -rf ~}" in out


def test_unsafe_tex_after_fenced_code_block_still_blocked() -> None:
    """The fence state machine resets correctly after the closing fence."""
    md = (
        "```latex\n"
        "\\input{ok-here}\n"
        "```\n"
        "Outside the fence: \\input{not-ok}\n"
    )
    with pytest.raises(UnsafeTexError) as exc:
        sanitize_markdown(md)
    assert "\\input" in exc.value.message


def test_tex_math_dollars_allowed() -> None:
    """Inline / display math via ``$...$`` and ``$$...$$`` must pass through.

    The whole point of the ``tex_math_dollars`` pandoc extension is to
    let users write math. The sanitizer must not reject it.
    """
    md = "We compute $E = mc^2$ and $$\\sum_i x_i = 1$$.\n"
    out = sanitize_markdown(md)
    assert "$E = mc^2$" in out
    assert "$$\\sum_i x_i = 1$$" in out


def test_unrelated_backslash_command_not_blocked() -> None:
    """Latex-looking commands NOT on the block list pass through unchanged.

    For example ``\\textbf{...}`` is harmless and may legitimately appear
    in LLM output that opted into raw_tex semantics.
    """
    # ``\textbf`` is not in the block list -- it must not raise.
    out = sanitize_markdown("This is \\textbf{important}.\n")
    assert "\\textbf{important}" in out


def test_inputtable_does_not_false_positive_input() -> None:
    """``\\inputtable`` must NOT be confused with ``\\input``.

    The block patterns use ``(?![A-Za-z])`` after the name so longer
    identifiers that happen to share a prefix are not matched.
    """
    # \inputtable is hypothetical and not in our block list; the
    # negative-lookahead must prevent a false positive against \input.
    out = sanitize_markdown("custom \\inputtable command\n")
    assert "\\inputtable" in out


def test_blocked_tex_message_includes_line_number() -> None:
    """The error message names the line number to help the LLM locate the issue."""
    md = "Safe line one.\n\nNow the bad one: \\input{evil}\n"
    with pytest.raises(UnsafeTexError) as exc:
        sanitize_markdown(md)
    # The bad line is line 3 (1-indexed).
    assert "line 3" in exc.value.message
