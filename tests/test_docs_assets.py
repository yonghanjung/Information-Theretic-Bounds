from pathlib import Path


def test_readme_references_quick_demo_gif():
    root = Path(__file__).resolve().parents[1]
    readme = (root / "README.md").read_text(encoding="utf-8")
    assert (
        "https://raw.githubusercontent.com/yonghanjung/Information-Theretic-Bounds/main/docs/media/quick-demo-v5.gif"
        in readme
    )
    assert (root / "docs" / "media" / "quick-demo-v5.gif").is_file()
