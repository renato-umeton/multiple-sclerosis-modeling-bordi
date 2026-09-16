"""Checks on the documentation site, its API pages and the workflow files.

The site itself is built with ``mkdocs build --strict``; these tests cover what
a build cannot see: that every module of the package has a reference page, that
the pages written here keep the house style, and that the two workflow files
say what they are meant to say.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pytest
import yaml  # type: ignore[import-untyped]

import msrelapse

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
MKDOCS = ROOT / "mkdocs.yml"
WORKFLOWS = ROOT / ".github" / "workflows"
PACKAGE = ROOT / "src" / "msrelapse"
# The README holds the summary paragraph the home page has to quote word for
# word, so the two say the same thing. The local planning notes hold it too,
# but they are not in the repository and a checkout would not find them.
SUMMARY_SOURCE = ROOT / "README.md"
SUMMARY_MARKER = "In 70 untreated relapsing-remitting MS patients"

# The first line of a BibTeX entry, and a field line inside one.
BIBTEX_LINE = re.compile(r"^(@\w+\{|[a-z]+\s*=\s*\{)")

# Written as a code point so that the character itself never enters a source
# file of this repository, which is the rule the check below enforces.
EM_DASH = chr(0x2014)

# The reference page of each module, as the page name and the object the
# mkdocstrings directive on it addresses.
API_PAGES = {
    "model": "msrelapse.model",
    "simulate": "msrelapse.simulate",
    "renewal": "msrelapse.renewal",
    "cohort": "msrelapse.cohort",
    "fit": "msrelapse.fit",
    "stats": "msrelapse.stats",
    "io": "msrelapse.io",
    "plots": "msrelapse.plots",
    "datasets": "msrelapse.datasets",
    "cli": "msrelapse.cli",
    "params": "msrelapse._params",
}

# The prose pages written for the site. paper_facts.md is older than this task
# and is published as it stands.
PROSE_PAGES = (
    "index.md",
    "theory.md",
    "reproducing.md",
    "data.md",
    "citing.md",
    "CHANGELOG.md",
    "CONTRIBUTING.md",
)

# Pages that live under docs/ but are not part of the published site.
EXCLUDED = ("plan1.md", "IMPLEMENTATION_PLAN.md", "paper/", "pull_request_template.md")


def load_config() -> dict[str, Any]:
    """Return the parsed mkdocs configuration."""
    with MKDOCS.open(encoding="utf-8") as handle:
        config: dict[str, Any] = yaml.safe_load(handle)
    return config


def load_workflow(name: str) -> dict[Any, Any]:
    """Return one parsed workflow file of .github/workflows.

    The keys are not all strings: YAML reads the unquoted workflow key ``on``
    as the boolean True, which is how the triggers are addressed below.
    """
    with (WORKFLOWS / name).open(encoding="utf-8") as handle:
        workflow: dict[Any, Any] = yaml.safe_load(handle)
    return workflow


def nav_targets(nav: Any) -> list[str]:
    """Return every page a nav tree points at, in the order it appears."""
    if isinstance(nav, str):
        return [nav]
    if isinstance(nav, list):
        return [target for item in nav for target in nav_targets(item)]
    if isinstance(nav, dict):
        return [target for value in nav.values() for target in nav_targets(value)]
    raise TypeError(f"a nav entry is a string, a list or a mapping, got {nav!r}")


def authored_files() -> list[Path]:
    """Return every file of the site this task writes.

    The pages, prose and reference alike, plus the configuration, the MathJax
    loader and the two workflow files, because the house style rule below
    applies to all of them and not only to the Markdown.
    """
    return [
        *(DOCS / name for name in PROSE_PAGES),
        *(DOCS / "api" / f"{name}.md" for name in sorted(API_PAGES)),
        DOCS / "javascripts" / "mathjax.js",
        MKDOCS,
        WORKFLOWS / "ci.yml",
        WORKFLOWS / "docs.yml",
    ]


AUTHORED_FILES = authored_files()


def flatten(text: str) -> str:
    """Return text with blockquote markers dropped and the spacing flattened."""
    lines = [line.strip().removeprefix(">").strip() for line in text.splitlines()]
    return " ".join(" ".join(lines).split())


def summary_sentence() -> str:
    """Return the summary paragraph the home page has to quote from the README.

    The paragraph runs from the marker line to the next blank line, and comes
    back with its line breaks flattened so that the two files are free to wrap
    it differently.
    """
    if not SUMMARY_SOURCE.is_file():
        raise AssertionError(f"{SUMMARY_SOURCE} is gone, so the home page has nothing to match")
    lines = SUMMARY_SOURCE.read_text(encoding="utf-8").splitlines()
    for start, line in enumerate(lines):
        if SUMMARY_MARKER in line:
            end = start
            while end < len(lines) and lines[end].strip():
                end += 1
            return flatten("\n".join(lines[start:end]))
    raise AssertionError(f"{SUMMARY_SOURCE} no longer holds the summary sentence to quote")


def package_modules() -> set[str]:
    """Return the modules of the package that a reader can import by name."""
    return {
        path.stem
        for path in PACKAGE.glob("*.py")
        if path.stem != "__init__" and not path.stem.startswith("_")
    }


def test_site_is_named_after_the_package() -> None:
    config = load_config()
    assert config["site_name"] == "msrelapse"
    assert "renato-umeton/multiple-sclerosiss-modeling-bordi" in config["repo_url"]
    assert config["docs_dir"] == "docs"


def test_every_nav_entry_points_at_a_file_that_exists() -> None:
    targets = nav_targets(load_config()["nav"])
    missing = [target for target in targets if not (DOCS / target).is_file()]
    assert missing == []


def test_every_page_of_the_site_is_in_the_nav() -> None:
    targets = set(nav_targets(load_config()["nav"]))
    published = {
        path.relative_to(DOCS).as_posix()
        for path in DOCS.rglob("*.md")
        if not any(path.relative_to(DOCS).as_posix().startswith(name) for name in EXCLUDED)
    }
    assert published - targets == set()


@pytest.mark.parametrize(("page", "obj"), sorted(API_PAGES.items()))
def test_each_api_page_documents_its_module(page: str, obj: str) -> None:
    text = (DOCS / "api" / f"{page}.md").read_text(encoding="utf-8")
    assert f"::: {obj}" in text
    assert text.startswith("# ")


def test_every_importable_module_has_a_reference_page() -> None:
    documented = {obj.removeprefix("msrelapse.") for obj in API_PAGES.values()}
    assert package_modules() - documented == set()


def test_only_the_numbers_of_the_paper_are_documented_from_the_private_modules() -> None:
    """Private modules stay private, apart from the one holding the numbers.

    ``_params`` is documented because a reader checks the package against the
    article field by field. Any other module named with a leading underscore is
    internal, and a reference page for it would publish an interface nobody
    promised to keep.
    """
    documented = set()
    for path in sorted((DOCS / "api").glob("*.md")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.startswith("::: msrelapse."):
                documented.add(line.removeprefix("::: ").strip())
    private = {name for name in documented if name.rpartition(".")[2].startswith("_")}
    assert private == {"msrelapse._params"}, "docs/api/ documents a private module of its own"


@pytest.mark.parametrize(
    "path",
    AUTHORED_FILES,
    ids=[path.relative_to(ROOT).as_posix() for path in AUTHORED_FILES],
)
def test_files_keep_the_house_style(path: Path) -> None:
    lines = path.read_text(encoding="utf-8").splitlines()
    assert [line for line in lines if EM_DASH in line] == []
    assert [line for line in lines if line.strip() == "---"] == []


def test_the_private_pages_are_kept_out_of_the_site() -> None:
    excluded = load_config()["exclude_docs"].split()
    for name in EXCLUDED:
        assert name in excluded


def test_the_publisher_pdf_stays_out_of_the_published_site() -> None:
    """The copy of the article in docs/ is not republished from this site.

    It is a download from the publisher, stamp and all, and nothing on the site
    links to it. The article itself is open access and the citing page sends a
    reader to the DOI.
    """
    assert "*.pdf" in load_config()["exclude_docs"].split()


def test_mathjax_is_wired_up() -> None:
    scripts = load_config()["extra_javascript"]
    assert "javascripts/mathjax.js" in scripts
    assert (DOCS / "javascripts" / "mathjax.js").is_file()
    assert any("mathjax" in script and script.startswith("https://") for script in scripts)


def test_the_python_handler_reads_numpy_docstrings_from_the_source_tree() -> None:
    plugins = load_config()["plugins"]
    handlers = [
        plugin["mkdocstrings"]
        for plugin in plugins
        if isinstance(plugin, dict) and "mkdocstrings" in plugin
    ]
    assert handlers, "mkdocs.yml declares no mkdocstrings plugin"
    options = handlers[0]["handlers"]["python"]
    assert options["paths"] == ["src"]
    assert options["options"]["docstring_style"] == "numpy"


def test_display_math_is_rendered_by_arithmatex() -> None:
    extensions = load_config()["markdown_extensions"]
    arithmatex = [
        item["pymdownx.arithmatex"]
        for item in extensions
        if isinstance(item, dict) and "pymdownx.arithmatex" in item
    ]
    assert arithmatex, "mkdocs.yml declares no pymdownx.arithmatex extension"
    assert arithmatex[0]["generic"] is True
    assert "$$" in (DOCS / "theory.md").read_text(encoding="utf-8")


def test_the_theory_page_cites_the_work_it_rests_on() -> None:
    text = (DOCS / "theory.md").read_text(encoding="utf-8")
    for author in ("Bordi", "Benzi", "Kramers", "Day", "Zhu", "Keene"):
        assert author in text


def test_the_home_page_quotes_the_summary_sentence_word_for_word() -> None:
    page = flatten((DOCS / "index.md").read_text(encoding="utf-8"))
    assert summary_sentence() in page, "docs/index.md and README.md summarise the work differently"


def test_the_citing_page_carries_the_bibtex_the_package_prints() -> None:
    """The page and ``msrelapse.citation()`` give the same two entries.

    Both are written out in full, so nothing but this check stops the page from
    drifting away from the code as the version or the DOI changes.
    """
    page = flatten((DOCS / "citing.md").read_text(encoding="utf-8"))
    entries = [
        flatten(line)
        for line in msrelapse.citation().splitlines()
        if BIBTEX_LINE.match(line.strip())
    ]
    assert entries, "msrelapse.citation() prints no BibTeX"
    missing = [line for line in entries if line not in page]
    assert missing == [], "docs/citing.md has drifted from msrelapse.citation()"


def test_the_reproduction_page_lists_the_rows_of_the_closing_table() -> None:
    """The page names every row ``reproduction_table`` builds, as it names it."""
    table = msrelapse.reproduction_table(msrelapse.load_synthetic_bordi2013())
    page = flatten((DOCS / "reproducing.md").read_text(encoding="utf-8"))
    missing = [quantity for quantity in table["quantity"] if quantity not in page]
    assert missing == [], "docs/reproducing.md has drifted from reproduction_table"


def test_the_home_page_sends_each_audience_to_its_entry_point() -> None:
    lines = (DOCS / "index.md").read_text(encoding="utf-8").splitlines()
    assert any("10.1155/2013/910321" in line for line in lines)
    assert any(line.startswith("| Audience | What they need | Entry point |") for line in lines)
    for audience, entry_point in (
        ("Mathematical biologists", "msrelapse.model"),
        ("Trial statisticians", "msrelapse.fit"),
        ("In-silico trial and digital twin groups", "msrelapse.cohort"),
    ):
        rows = [line for line in lines if line.startswith(f"| {audience} |")]
        assert rows, f"the home page has no row for {audience}"
        assert entry_point in rows[0]


def test_ci_builds_the_docs_and_runs_the_notebooks_after_the_lint_job() -> None:
    jobs = load_workflow("ci.yml")["jobs"]
    assert {"lint", "test", "docs", "notebooks"} <= set(jobs)
    for name in ("docs", "notebooks"):
        # `needs` is a string or a list of strings; both spellings are correct.
        needs = jobs[name]["needs"]
        assert "lint" in ([needs] if isinstance(needs, str) else needs)
    docs_run = " ".join(step.get("run", "") for step in jobs["docs"]["steps"])
    assert "mkdocs build --strict" in docs_run
    assert "--group docs" in docs_run
    notebook_run = " ".join(step.get("run", "") for step in jobs["notebooks"]["steps"])
    assert "pytest -m notebook" in notebook_run


def test_the_docs_workflow_publishes_the_site_from_main() -> None:
    workflow = load_workflow("docs.yml")
    # PyYAML reads the unquoted key `on` as the boolean True.
    triggers = workflow[True]
    assert triggers["push"]["branches"] == ["main"]
    assert workflow["permissions"]["contents"] == "write"
    # Deploys queue instead of racing: a run cancelled inside gh-deploy would
    # leave the published branch on the previous commit with nothing to say so.
    assert workflow["concurrency"]["cancel-in-progress"] is False
    steps = " ".join(step.get("run", "") for step in workflow["jobs"]["deploy"]["steps"])
    # The deploy refuses to publish what the docs job of ci.yml refuses to
    # accept, so a warning cannot reach the published branch.
    assert "mkdocs gh-deploy --strict --force" in steps
