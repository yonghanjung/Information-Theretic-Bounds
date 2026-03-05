from __future__ import annotations

import re
from pathlib import Path


IMAGE_LINK_RE = re.compile(r"!\[[^\]]*\]\(([^)]+)\)")


def test_readme_images_use_absolute_urls():
    root = Path(__file__).resolve().parents[1]
    readme = (root / "README.md").read_text(encoding="utf-8")
    links = IMAGE_LINK_RE.findall(readme)

    relative = [
        link
        for link in links
        if not link.startswith("http://")
        and not link.startswith("https://")
        and not link.startswith("data:")
    ]
    assert not relative, f"Relative image links are not PyPI-safe: {relative}"
