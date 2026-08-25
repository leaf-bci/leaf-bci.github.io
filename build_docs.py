#!/usr/bin/env python3
"""Build the LEAF Markdown documentation as a static GitBook-style site."""

from __future__ import annotations

import html
import json
import os
import posixpath
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path


SITE_ROOT = Path(__file__).resolve().parent
SOURCE_ROOT = SITE_ROOT / "doc"
DOCS_ROOT = SITE_ROOT / "docs"


@dataclass(frozen=True)
class Page:
    source: str
    output: str
    title: str
    group: str
    nav_label: str


PAGES = [
    Page("overview.md", "index.html", "Overview", "Start", "Overview"),
    Page("datasets/README.md", "datasets/index.html", "Dataset specification", "Datasets", "Dataset Specification"),
    Page("datasets/instruction-tuning-datasets.md", "datasets/instruction-tuning-datasets.html", "Instruction tuning datasets", "Datasets", "Instruction Tuning Datasets"),
    Page("datasets/held-out.md", "datasets/held-out.html", "Held-out datasets", "Datasets", "Held-out Datasets"),
    Page("pretraining.md", "pretraining.html", "EEG pretraining", "Training", "Pretraining"),
    Page("instruction-tuning.md", "instruction-tuning.html", "Instruction tuning", "Training", "Instruction tuning"),
    Page("direct-inference.md", "direct-inference.html", "Direct inference", "Evaluation", "Direct Inference"),
    Page("single-dataset-finetuning.md", "single-dataset-finetuning.html", "Dataset-specific fine-tuning with a classification head", "Evaluation", "Dataset-specific Fine-tuning"),
]

GROUP_ORDER = ["Start", "Datasets", "Training", "Evaluation"]
PAGE_BY_SOURCE = {page.source: page for page in PAGES}


def relative_page_url(current: Page, target: Page) -> str:
    current_dir = (DOCS_ROOT / current.output).parent
    target_path = DOCS_ROOT / target.output
    if target_path.name == "index.html":
        result = os.path.relpath(target_path.parent, current_dir).replace(os.sep, "/")
        return "./" if result == "." else f"{result.rstrip('/')}/"
    return os.path.relpath(target_path, current_dir).replace(os.sep, "/")


def relative_asset_url(current: Page, asset: str) -> str:
    return os.path.relpath(DOCS_ROOT / asset, (DOCS_ROOT / current.output).parent).replace(os.sep, "/")


def project_url(current: Page, path: str = "index.html") -> str:
    return os.path.relpath(SITE_ROOT / path, (DOCS_ROOT / current.output).parent).replace(os.sep, "/")


def convert_markdown(source: Path) -> str:
    result = subprocess.run(
        ["pandoc", "--from=gfm", "--to=html5", "--wrap=none", str(source)],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout


def rewrite_links(fragment: str, current: Page) -> str:
    def replace(match: re.Match[str]) -> str:
        attribute, target = match.group(1), match.group(2)
        if target.startswith(("http://", "https://", "mailto:", "#", "data:")):
            return match.group(0)

        path_part, marker, anchor = target.partition("#")
        if attribute == "href" and path_part.endswith(".md"):
            resolved = posixpath.normpath(posixpath.join(posixpath.dirname(current.source), path_part))
            target_page = PAGE_BY_SOURCE.get(resolved)
            if target_page is None:
                raise ValueError(f"Unknown documentation link in {current.source}: {target}")
            url = relative_page_url(current, target_page)
            if marker:
                url += f"#{anchor}"
            return f'{attribute}="{url}"'

        if attribute == "src" and path_part == "../leaf-architecture.png":
            return f'{attribute}="{relative_asset_url(current, "assets/leaf-architecture.png")}"'

        return match.group(0)

    return re.sub(r'\b(href|src)="([^"]+)"', replace, fragment)


def strip_html(value: str) -> str:
    value = re.sub(r"<[^>]+>", " ", value)
    value = html.unescape(value)
    return re.sub(r"\s+", " ", value).strip()


def extract_toc(fragment: str) -> list[tuple[int, str, str]]:
    headings = []
    pattern = re.compile(r'<h([23]) id="([^"]+)">(.*?)</h\1>', re.DOTALL)
    for match in pattern.finditer(fragment):
        headings.append((int(match.group(1)), match.group(2), strip_html(match.group(3))))
    return headings


def sidebar(current: Page) -> str:
    sections = []
    for group in GROUP_ORDER:
        links = []
        for page in PAGES:
            if page.group != group:
                continue
            active = ' aria-current="page" class="active"' if page == current else ""
            links.append(
                f'<a href="{html.escape(relative_page_url(current, page))}"{active}>'
                f"{html.escape(page.nav_label)}</a>"
            )
        sections.append(
            f'<section class="nav-section"><h2>{html.escape(group)}</h2>'
            f'<div class="nav-items">{"".join(links)}</div></section>'
        )
    return "".join(sections)


def page_toc(headings: list[tuple[int, str, str]]) -> str:
    if not headings:
        return ""
    links = []
    for level, identifier, title in headings:
        links.append(
            f'<a class="toc-level-{level}" href="#{html.escape(identifier)}">'
            f"{html.escape(title)}</a>"
        )
    return '<nav class="page-toc" aria-label="On this page"><h2>On this page</h2>' + "".join(links) + "</nav>"


def breadcrumbs(current: Page) -> str:
    parts = [f'<a href="{html.escape(relative_page_url(current, PAGES[0]))}">Docs</a>']
    if current.group == "Datasets" and current.source != "datasets/README.md":
        parts.append(f'<a href="{html.escape(relative_page_url(current, PAGE_BY_SOURCE["datasets/README.md"]))}">Datasets</a>')
    if current != PAGES[0]:
        parts.append(f"<span>{html.escape(current.nav_label)}</span>")
    return '<nav class="breadcrumbs" aria-label="Breadcrumb">' + '<span aria-hidden="true">/</span>'.join(parts) + "</nav>"


def pager(current: Page) -> str:
    index = PAGES.index(current)
    items = []
    if index > 0:
        previous = PAGES[index - 1]
        items.append(
            f'<a class="pager-link previous" href="{html.escape(relative_page_url(current, previous))}">'
            f'<span>Previous</span><strong>← {html.escape(previous.nav_label)}</strong></a>'
        )
    else:
        items.append('<span class="pager-spacer"></span>')
    if index + 1 < len(PAGES):
        following = PAGES[index + 1]
        items.append(
            f'<a class="pager-link next" href="{html.escape(relative_page_url(current, following))}">'
            f'<span>Next</span><strong>{html.escape(following.nav_label)} →</strong></a>'
        )
    return '<nav class="page-pager" aria-label="Documentation pages">' + "".join(items) + "</nav>"


def render_page(current: Page, fragment: str) -> str:
    stylesheet = relative_asset_url(current, "assets/docs.css")
    script = relative_asset_url(current, "assets/docs.js")
    search_index = relative_asset_url(current, "search-index.json")
    favicon = project_url(current, "assets/leaf-mark.svg")
    project_home = project_url(current)
    docs_home = relative_page_url(current, PAGES[0])
    headings = extract_toc(fragment)

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta name="description" content="{html.escape(current.title)} — LEAF documentation">
  <meta name="theme-color" content="#f6f8f7">
  <title>{html.escape(current.title)} | LEAF Docs</title>
  <link rel="icon" href="{html.escape(favicon)}" type="image/svg+xml">
  <link rel="stylesheet" href="{html.escape(stylesheet)}?v=20260825-4">
  <script>try{{document.documentElement.dataset.theme=localStorage.getItem('leaf-doc-theme-v2')||'light'}}catch(e){{document.documentElement.dataset.theme='light'}}</script>
</head>
<body data-search-index="{html.escape(search_index)}" data-docs-home="{html.escape(docs_home)}">
  <a class="skip-link" href="#doc-content">Skip to content</a>
  <header class="docs-header">
    <button class="sidebar-toggle" type="button" aria-expanded="false" aria-controls="docs-sidebar" aria-label="Open documentation navigation"><span></span><span></span><span></span></button>
    <a class="docs-brand" href="{html.escape(docs_home)}" aria-label="LEAF documentation home">
      <span class="brand-mark" aria-hidden="true"><svg viewBox="0 0 40 40"><path d="M31 7C18 8 9 15 9 25c0 5 3 8 8 8 10 0 16-11 14-26Z"/><path d="M11 31c4-8 9-13 17-18"/></svg></span>
      <span><strong>LEAF</strong><small>Documentation</small></span>
    </a>
    <div class="header-actions">
      <label class="search-box"><span class="search-icon" aria-hidden="true">⌕</span><span class="sr-only">Search documentation</span><input id="doc-search" type="search" placeholder="Search docs…" autocomplete="off" aria-controls="search-results" aria-expanded="false"><kbd>/</kbd></label>
      <a class="header-link" href="{html.escape(project_home)}">Project <span aria-hidden="true">↗</span></a>
      <button class="theme-toggle" type="button" aria-label="Toggle color theme"><span class="theme-icon" aria-hidden="true">◐</span></button>
    </div>
    <div class="search-results" id="search-results" hidden></div>
  </header>
  <div class="sidebar-backdrop" hidden></div>
  <aside class="docs-sidebar" id="docs-sidebar">
    <div class="sidebar-scroll">{sidebar(current)}</div>
    <footer><a href="{html.escape(project_home)}">← Back to project</a><span>LEAF public release</span></footer>
  </aside>
  <div class="docs-layout">
    <main class="docs-main" id="doc-content">
      {breadcrumbs(current)}
      <article class="markdown-body">{fragment}</article>
      {pager(current)}
    </main>
    <aside class="toc-column">{page_toc(headings)}</aside>
  </div>
  <script src="{html.escape(script)}?v=20260825-4" defer></script>
</body>
</html>
"""


def main() -> None:
    if shutil.which("pandoc") is None:
        raise SystemExit("pandoc is required to build the documentation")
    missing = [page.source for page in PAGES if not (SOURCE_ROOT / page.source).is_file()]
    if missing:
        raise SystemExit(f"Missing documentation source files: {missing}")

    fragments: dict[str, str] = {}
    search_entries = []
    for page in PAGES:
        fragment = rewrite_links(convert_markdown(SOURCE_ROOT / page.source), page)
        fragments[page.source] = fragment
        output = DOCS_ROOT / page.output
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(render_page(page, fragment), encoding="utf-8")
        search_entries.append(
            {
                "title": page.title,
                "group": page.group,
                "url": page.output.replace("index.html", ""),
                "text": strip_html(fragment),
            }
        )

    architecture_source = SITE_ROOT / "assets" / "figures" / "leaf-architecture.png"
    if architecture_source.is_file():
        shutil.copy2(architecture_source, DOCS_ROOT / "assets" / "leaf-architecture.png")

    (DOCS_ROOT / "search-index.json").write_text(
        json.dumps(search_entries, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )

    # Preserve existing links while keeping Overview as the single docs landing page.
    (DOCS_ROOT / "overview.html").write_text(
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        '<meta http-equiv="refresh" content="0; url=./">'
        '<link rel="canonical" href="./"><title>LEAF Documentation</title>'
        '</head><body><p><a href="./">Go to the LEAF documentation overview</a>.</p>'
        '</body></html>\n',
        encoding="utf-8",
    )
    print(f"Built {len(PAGES)} documentation pages in {DOCS_ROOT}")


if __name__ == "__main__":
    main()
