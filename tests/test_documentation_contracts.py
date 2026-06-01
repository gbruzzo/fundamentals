"""Repository documentation drift checks."""

from __future__ import annotations

import re
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def _chapter_numbers() -> list[int]:
    return sorted(
        int(path.name.removeprefix("chapter_"))
        for path in (REPO_ROOT / "chapters").glob("chapter_*")
        if path.is_dir()
    )


def _chapter_scripts(chapter: int) -> list[Path]:
    chapter_dir = REPO_ROOT / "chapters" / f"chapter_{chapter:02d}"
    return sorted(path for path in chapter_dir.glob("*.py") if path.name != "__init__.py")


def test_live_chapters_have_docs_and_folder_contracts() -> None:
    for chapter in _chapter_numbers():
        suffix = f"{chapter:02d}"
        assert (REPO_ROOT / "chapters" / f"chapter_{suffix}" / "README.md").exists()
        assert (REPO_ROOT / "chapters" / f"chapter_{suffix}" / "AGENTS.md").exists()
        assert (REPO_ROOT / "docs" / "chapters" / f"chapter_{suffix}.md").exists()
        assert (REPO_ROOT / "output" / "figures" / f"chapter_{suffix}" / "README.md").exists()
        assert (REPO_ROOT / "output" / "data" / f"chapter_{suffix}" / "README.md").exists()
        figure_readme = (
            REPO_ROOT / "output" / "figures" / f"chapter_{suffix}" / "README.md"
        ).read_text(encoding="utf-8")
        assert f"output/data/chapter_{suffix}" in figure_readme
        assert "NPZ" in figure_readme and "JSON" in figure_readme


def test_reading_order_enumerates_all_live_chapters() -> None:
    text = (REPO_ROOT / "docs" / "reading_order.md").read_text(encoding="utf-8")
    for chapter in _chapter_numbers():
        assert f"chapters/chapter_{chapter:02d}.md" in text


def test_chapter_readme_coverage_counts_match_live_scripts() -> None:
    text = (REPO_ROOT / "chapters" / "README.md").read_text(encoding="utf-8")
    for chapter in _chapter_numbers():
        match = re.search(rf"^\| Chapter {chapter} \| (?P<count>\d+)\b", text, re.MULTILINE)
        assert match, f"Chapter {chapter} missing from chapters/README.md coverage table"
        assert int(match.group("count")) == len(_chapter_scripts(chapter))


def test_output_figures_overview_counts_match_rendered_artifacts_when_present() -> None:
    text = (REPO_ROOT / "output" / "figures" / "README.md").read_text(encoding="utf-8")
    saw_rendered_artifact = False

    for chapter in _chapter_numbers():
        suffix = f"{chapter:02d}"
        media_dir = REPO_ROOT / "output" / "figures" / f"chapter_{suffix}"
        png_count = len(list(media_dir.glob("*.png")))
        gif_count = len(list(media_dir.glob("*.gif")))
        if png_count + gif_count == 0:
            continue

        saw_rendered_artifact = True
        line = next(
            line for line in text.splitlines() if f"[`chapter_{suffix}/`]" in line
        )
        documented_png = re.search(r"(\d+)\s+PNGs?", line)
        documented_gif = re.search(r"(\d+)\s+GIFs?", line)
        if documented_png:
            assert int(documented_png.group(1)) == png_count
        else:
            assert png_count == 0
        if documented_gif:
            assert int(documented_gif.group(1)) == gif_count
        else:
            assert gif_count == 0

    assert saw_rendered_artifact


def test_root_readme_keeps_required_community_links() -> None:
    text = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    assert "https://activeinference.institute/" in text
    assert "https://textbook-group.activeinference.institute/" in text


def test_gitignore_ignores_nested_generated_media_but_preserves_docs() -> None:
    text = (REPO_ROOT / ".gitignore").read_text(encoding="utf-8")
    assert "output/figures/**/*.png" in text
    assert "output/figures/**/*.gif" in text
    assert "!output/figures/**/README.md" in text
    assert "!output/data/README.md" in text
    assert "!output/data/**/README.md" in text


def test_raw_data_contract_is_documented_for_agents_and_users() -> None:
    required = ("NPZ", "JSON", "save_chapter_data", "validate_raw_data_exports.py")
    docs = [
        REPO_ROOT / "AGENTS.md",
        REPO_ROOT / "README.md",
        REPO_ROOT / "output" / "data" / "README.md",
        REPO_ROOT / "output" / "data" / "AGENTS.md",
        REPO_ROOT / "scripts" / "README.md",
        REPO_ROOT / "docs" / "reference" / "utils.md",
    ]
    for path in docs:
        text = path.read_text(encoding="utf-8")
        for term in required:
            assert term in text, f"{term} missing from {path.relative_to(REPO_ROOT)}"
