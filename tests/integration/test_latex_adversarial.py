"""Spec §5.10: LaTeX adversarial sanity — Claude-shaped markdown with
Greek letters, math, Unicode authors, code blocks, and a long table.

Skips gracefully when pandoc or latexmk is absent.
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

pytestmark = pytest.mark.integration

# ---------------------------------------------------------------------------
# Skip guards
# ---------------------------------------------------------------------------

_PANDOC = shutil.which("pandoc")
_LATEXMK = shutil.which("latexmk")

if _PANDOC is None:
    pytestmark = pytest.mark.skip(reason="pandoc not installed")

if _LATEXMK is None:
    pytestmark = pytest.mark.skip(reason="latexmk not installed")


# ---------------------------------------------------------------------------
# Adversarial markdown fixture
# ---------------------------------------------------------------------------

_ADVERSARIAL_MD = r"""# Adversarial LaTeX Test

## Greek letters and inline math

Pseudotime follows $\alpha$-decay with rate $\lambda = 0.5$.

## Block math

$$
\sum_{i=1}^{n} x_i = \frac{n(n+1)}{2}
$$

## Unicode author names

Müller *et al.* (2021), Søren & Wang (2022).

## Code block with backslashes

```python
import os
path = "C:\\Users\\user\\data"
result = os.path.join(path, "file.h5ad")
```

## Long table

| Method | nDCG@10 | MRR | R@50 | Time (ms) |
|--------|---------|-----|------|-----------|
| PAGA | 0.119 | 0.169 | 0.412 | 42 |
| Palantir | 0.115 | 0.160 | 0.398 | 120 |
| scVelo | 0.101 | 0.145 | 0.371 | 95 |
| Monocle3 | 0.112 | 0.158 | 0.405 | 310 |
| Slingshot | 0.108 | 0.152 | 0.389 | 280 |

## Conclusion

The ordering $A \to B \to C$ is consistent across all methods.
"""


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------

@pytest.mark.skipif(_PANDOC is None, reason="pandoc not installed")
@pytest.mark.skipif(_LATEXMK is None, reason="latexmk not installed")
def test_adversarial_markdown_compiles_to_pdf(tmp_path: Path) -> None:
    """Adversarial markdown → pandoc → latexmk → PDF must succeed."""
    md_path = tmp_path / "adversarial.md"
    md_path.write_text(_ADVERSARIAL_MD, encoding="utf-8")

    tex_path = tmp_path / "adversarial.tex"

    # Step 1: pandoc markdown → LaTeX.
    pandoc_result = subprocess.run(
        [
            str(_PANDOC),
            str(md_path),
            "-o", str(tex_path),
            "--standalone",
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert pandoc_result.returncode == 0, (
        f"pandoc failed (rc={pandoc_result.returncode}):\n"
        f"stdout: {pandoc_result.stdout}\n"
        f"stderr: {pandoc_result.stderr}"
    )
    assert tex_path.exists(), "pandoc did not produce a .tex file"

    # Step 2: latexmk → PDF.
    latexmk_result = subprocess.run(
        [
            str(_LATEXMK),
            "-pdf",
            "-halt-on-error",
            "-interaction=nonstopmode",
            f"-outdir={tmp_path}",
            str(tex_path),
        ],
        capture_output=True,
        text=True,
        timeout=120,
        cwd=str(tmp_path),
    )
    assert latexmk_result.returncode == 0, (
        f"latexmk failed (rc={latexmk_result.returncode}):\n"
        f"stdout: {latexmk_result.stdout[-2000:]}\n"
        f"stderr: {latexmk_result.stderr[-2000:]}"
    )

    pdf_path = tmp_path / "adversarial.pdf"
    assert pdf_path.exists(), "latexmk did not produce a PDF"
    assert pdf_path.stat().st_size > 0, "PDF is empty"
