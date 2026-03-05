from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _heading_positions(text: str, headings: list[str]) -> list[int]:
    positions = []
    for heading in headings:
        pos = text.find(heading)
        assert pos != -1, f"missing heading: {heading}"
        positions.append(pos)
    return positions


def test_readme_top_fold_has_launch_contract():
    root = _repo_root()
    readme = (root / "README.md").read_text(encoding="utf-8")
    top = "\n".join(readme.splitlines()[:40])

    assert "# itbound: Causal Bounds Without Strong Identification Assumptions" in readme
    assert "https://arxiv.org/abs/2601.17160" in top
    assert "python -m pip install itbound" in top
    assert "itbound example --out /tmp/itbound_example.csv" in top
    assert (
        "itbound quick --data /tmp/itbound_example.csv --treatment a --outcome y --covariates x1,x2 --outdir /tmp/itbound_quick"
        in top
    )
    assert "ribbon_ihdp.png" in top
    assert (
        "Compute valid lower and upper causal bounds under unmeasured confounding when point identification is not credible."
        in top
    )
    assert "### Who is this for?" in readme

    positions = _heading_positions(
        readme,
        [
            "## Why it matters",
            "## One-command quickstart",
            "## What you get",
            "## Main figure",
            "## Reproduce paper",
            "## Citation",
            "## Links",
        ],
    )
    assert positions == sorted(positions)


def test_citation_metadata_matches_paper_source_of_truth():
    root = _repo_root()
    pyproject = (root / "pyproject.toml").read_text(encoding="utf-8")
    citation = (root / "CITATION.cff").read_text(encoding="utf-8")

    assert '{ name = "Bogyeong Kang" }' in pyproject
    assert "family-names: Kang" in citation
    assert "given-names: Bogyeong" in citation
    assert "https://arxiv.org/abs/2601.17160" in citation
