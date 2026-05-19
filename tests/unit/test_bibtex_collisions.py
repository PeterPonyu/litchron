"""Spec §5.11: cite-key collision handling — -a / -b suffixes."""
from __future__ import annotations

from litchron.citations import Citation, CitationVerifier


def _make_citation(title: str, year: int, authors: list[str]) -> Citation:
    return Citation(
        scheme="doi",
        id=f"10.1000/test.{year}",
        title=title,
        year=year,
        authors=authors,
        abstract=None,
        confidence=0.9,
        source="crossref",
    )


def test_collision_produces_ab_suffixes() -> None:
    """Two citations normalizing to the same base key get -a and -b suffixes."""
    # Both will normalize to "smith_2019_pseudotime" as the base key.
    c1 = _make_citation(
        title="Pseudotime ordering in single cells",
        year=2019,
        authors=["Smith, Jane"],
    )
    c2 = _make_citation(
        title="Pseudotime ordering revisited",
        year=2019,
        authors=["Smith, Jane"],
    )

    bib = CitationVerifier.to_bibtex([c1, c2])

    # Both entries must be present.
    assert bib.count("@article{") == 2

    # Extract the cite keys.
    keys = [line.split("{", 1)[1].rstrip(",") for line in bib.splitlines() if line.startswith("@article{")]
    assert len(keys) == 2, f"Expected 2 keys, got: {keys}"

    # Keys must be distinct.
    assert keys[0] != keys[1], f"Keys are not distinct: {keys}"

    # The second key must carry the -a suffix (first collision).
    assert keys[1].endswith("-a"), f"Expected second key to end with -a, got: {keys[1]!r}"


def test_triple_collision_produces_abc_suffixes() -> None:
    """Three colliding citations → first key bare, second -a, third -b."""
    author = ["Doe, John"]
    c1 = _make_citation("Trajectory inference", 2020, author)
    c2 = _make_citation("Trajectory analysis", 2020, author)
    c3 = _make_citation("Trajectory methods review", 2020, author)

    bib = CitationVerifier.to_bibtex([c1, c2, c3])
    keys = [line.split("{", 1)[1].rstrip(",") for line in bib.splitlines() if line.startswith("@article{")]
    assert len(keys) == 3
    assert len(set(keys)) == 3, f"Not all keys are distinct: {keys}"

    # The first uses the bare key; second gets -a; third gets -b.
    assert not keys[0].endswith(("-a", "-b")), f"First key should be bare: {keys[0]!r}"
    assert keys[1].endswith("-a"), f"Second key should end with -a: {keys[1]!r}"
    assert keys[2].endswith("-b"), f"Third key should end with -b: {keys[2]!r}"
