"""Unit tests for diacritic-folded author-surname overlap (issue #22).

The verifier dropped legitimate citations when the LLM's author spelling and the
CrossRef/PubMed record differed only by diacritics (e.g. "Muller" vs "Müller").
These tests pin the documented behaviour: diacritics are folded, but non-Latin
base characters are preserved (so exact CJK surnames still match).
"""
from __future__ import annotations

from litchron.citations import _author_overlap, _surname


def test_diacritic_folded_to_ascii_base():
    assert _surname("Müller") == "muller"
    assert _surname("Müller, Hans") == "muller"
    assert _surname("Hans Müller") == "muller"
    assert _surname("MÜLLER") == "muller"
    assert _surname("García") == "garcia"


def test_overlap_matches_across_diacritic_spelling():
    # The exact failure mode from the issue: same author, different diacritics.
    assert _author_overlap(["Muller"], ["Müller"]) == 1
    assert _author_overlap(["Jose Garcia"], ["García, José"]) == 1


def test_non_latin_surnames_are_preserved_not_dropped():
    # NFKD + combining-mark removal must not strip CJK base characters.
    assert _surname("李") == "李"
    assert _author_overlap(["李"], ["李"]) == 1


def test_distinct_surnames_still_do_not_match():
    assert _author_overlap(["Smith"], ["Müller"]) == 0
    assert _author_overlap([], ["Müller"]) == 0
