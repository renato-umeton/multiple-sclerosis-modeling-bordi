"""Checks on the documentation site, its API pages and the workflow files.

The site itself is built with ``mkdocs build --strict``; these tests cover what
a build cannot see: that every module of the package has a reference page, that
the pages written here keep the house style, and that the two workflow files
say what they are meant to say.

The house style sweep in the middle of this file reads every text file of the
repository rather than a hand written list, so that a module, a test, a
notebook and a workflow are held to the rules the prose is held to.
"""

from __future__ import annotations

import inspect
import os
import re
import shutil
import subprocess
from collections import Counter
from collections.abc import Callable
from importlib.util import find_spec
from pathlib import Path
from typing import Any

import pytest

import msrelapse
from _helpers import (
    ASSISTANT_NAMES,
    ASSISTANT_PHRASE,
    EM_DASH,
    ROOT,
    assistant_mention_lines,
    contains_em_dash,
    em_dash_lines,
    flatten,
    gitignored_paths,
    has_three_hyphen_line,
    load_workflow,
    load_yaml,
    read,
    single_trailing_space_lines,
    text_files,
    three_hyphen_lines,
)
from msrelapse import plots

DOCS = ROOT / "docs"
PACKAGE = ROOT / "src" / "msrelapse"
# The README holds the summary paragraph the home page has to quote word for
# word, so the two say the same thing. The local planning notes hold it too,
# but they are not in the repository and a checkout would not find them.
SUMMARY_SOURCE = ROOT / "README.md"
SUMMARY_MARKER = "In 70 untreated relapsing-remitting MS patients"

# The first line of a BibTeX entry, and a field line inside one.
BIBTEX_LINE = re.compile(r"^(@\w+\{|[a-z]+\s*=\s*\{)")

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
    "edss": "msrelapse.edss",
    "cli": "msrelapse.cli",
    "params": "msrelapse._params",
}

# The page of the illustrative disability extension and the page it follows in
# the nav, so that a reader meets the article first and the extension after it.
DISABILITY_PAGE = "disability.md"
THEORY_PAGE = "theory.md"

# A DOI, as a page of this site and an evidence record of ``msrelapse.edss``
# both write one. Case is not significant in a DOI, so the checks below compare
# them in lower case.
DOI = re.compile(r"10\.\d{4,9}/\S+")

# The header of the parameter table of the disability page, and a short
# citation as the Sources column of that table writes one: a surname, a year,
# and, where two references share both, the words in brackets that tell them
# apart, which have to be read off the entry they point at.
PARAMETER_TABLE_HEADER = "| Parameter | Central value | Range | Unit | Sources |"
SHORT_CITATION = re.compile(r"([A-Z][A-Za-z]+) ((?:19|20)\d{2})(?: \(([^)]+)\))?")

# The phrases the caption under the animation carries in the README and on the
# home page alike, so that the two describe the bottom panel the same way.
CAPTION_PHRASES = (
    "illustrative EDSS trajectory",
    "six months",
    "42 percent",
    "residual of at least 0.5",
    "not a typical one",
)

# Where one sentence of a caption ends and the next begins: a full stop, then
# the space before a capital or an opening bracket. A decimal point is followed
# by a digit, so 2.0 and 4.3 stay whole.
SENTENCE_BREAK = re.compile(r"(?<=\.)\s+(?=[A-Z\[])")

# What a Markdown link looks like in the middle of a sentence. The README ships
# as the package long description, where a relative link breaks, so its caption
# reaches the disability page by its address on the site while the home page
# reaches the file beside it. Those sentences are the ones the two captions are
# allowed to differ on.
MARKDOWN_LINK = "]("

# Pages that live under docs/ but are not part of the published site.
EXCLUDED = ("plan1.md", "paper/", "pull_request_template.md")

# The animation of the home page, as that page refers to it and as the
# repository holds it, with the contact sheet beside it. Both are pictures, so
# the house style sweep has to walk past them rather than read them as prose.
ANIMATION_IMAGE = "assets/double_well.gif"
ANIMATION_ASSETS = ("docs/assets/double_well.gif", "docs/assets/double_well_frames.png")

# The Sphinx cross reference roles. mkdocstrings reads numpy docstrings and
# interprets none of them, so one left in a docstring reaches the built page as
# the literal text a reader sees. The markers are built from their names rather
# than written out, so that this file keeps the rule it checks.
ROLE_NAMES = ("func", "class", "meth", "attr", "mod", "data", "const", "obj", "ref")
ROLE_MARKERS = tuple(f":{name}:" for name in ROLE_NAMES)

# The two other pieces of reStructuredText a docstring picks up out of habit and
# mkdocstrings renders as the characters themselves. Both are built rather than
# written, as the markers above are, so that this file keeps the rules it checks.
#
# The first is the marker that opens a literal block at the end of a line. In
# Markdown the two colons stay on the page and the block under them is run
# together with the prose; a single colon and an indented block give the code
# block the writer meant.
LITERAL_BLOCK_MARKER = ":" * 2

# The second is the rule of a table: the row of equals signs of a simple table
# or the row of plus signs and hyphens of a grid one. A line of nothing but
# those characters and spaces draws one, apart from the underline of a numpydoc
# section, which is a run of hyphens as long as the title above it.
TABLE_RULE = re.compile(r"[=+\- ]+")
# Two characters draw neither table: the shortest rule either of them can hold
# is the three character grid rule, a plus, a hyphen and a plus, and a simple
# table needs two columns and so a longer one still. The floor is the one
# ``_helpers.MINIMUM_HYPHENS`` sets for the same reason, so that a stray pair of
# hyphens or equals signs is not reported as the table it is not.
MINIMUM_RULE_CHARACTERS = 3

# What mkdocs-autorefs logs for a cross reference whose target it cannot find.
# Strict mode already turns it into a failed build; the phrase is held here so
# that a build which stops failing on it is still caught.
UNRESOLVED_REFERENCE = "Could not find cross-reference target"

# The variable an environment sets to say that it is meant to be able to build
# the site. The check that reads the built pages back is skipped where mkdocs is
# absent, which is right on a machine that never installed the documentation
# group and wrong in a job that is meant to have it: a silent skip there is how
# a guard stops guarding without anyone noticing. Set this to any value in such
# a job and the skip becomes a failure naming what is missing.
REQUIRE_DOCS = "MSRELAPSE_REQUIRE_DOCS"

# A line of hyphens, built rather than written, so that this file keeps the
# rule it checks. The same goes for the two fence markers below.
HYPHEN_LINE = "-" * 3
FENCE = "`" * 3
OTHER_FENCE = "~" * 3

# The chat prefixed spelling of one of the names, built in pieces here for the
# same reason the names themselves are built in pieces in ``_helpers``: no
# piece of it is a name, so this file keeps the rule it checks.
CHAT_ASSISTANT = "chat" + "gp" + "t"


def load_config() -> dict[str, Any]:
    """Return the parsed mkdocs configuration."""
    return load_yaml("mkdocs.yml")


def nav_targets(nav: Any) -> list[str]:
    """Return every page a nav tree points at, in the order it appears."""
    if isinstance(nav, str):
        return [nav]
    if isinstance(nav, list):
        return [target for item in nav for target in nav_targets(item)]
    if isinstance(nav, dict):
        return [target for value in nav.values() for target in nav_targets(value)]
    raise TypeError(f"a nav entry is a string, a list or a mapping, got {nav!r}")


def summary_sentence() -> str:
    """Return the summary paragraph the home page has to quote from the README.

    The paragraph runs from the marker line to the next blank line, and comes
    back with its line breaks flattened so that the two files are free to wrap
    it differently.
    """
    if not SUMMARY_SOURCE.is_file():
        raise AssertionError(f"{SUMMARY_SOURCE} is gone, so the home page has nothing to match")
    lines = read(SUMMARY_SOURCE).splitlines()
    for start, line in enumerate(lines):
        if SUMMARY_MARKER in line:
            end = start
            while end < len(lines) and lines[end].strip():
                end += 1
            return flatten("\n".join(lines[start:end]))
    raise AssertionError(f"{SUMMARY_SOURCE} no longer holds the summary sentence to quote")


def page_dois(text: str) -> list[str]:
    """Return every DOI written in one page, in lower case and in reading order.

    The punctuation that can close a sentence is dropped from the end of a
    match, since no DOI of this project ends in one.

    Parameters
    ----------
    text : str
        The page to read.

    Returns
    -------
    list of str
        One entry per DOI written, duplicates kept.
    """
    return [match.group(0).rstrip(".,;").lower() for match in DOI.finditer(text)]


def references_of(text: str) -> str:
    """Return the references section of a page, from its heading to the end.

    Parameters
    ----------
    text : str
        The page to read.

    Returns
    -------
    str
        The text under the references heading.

    Raises
    ------
    AssertionError
        If the page has no references section.
    """
    marker = "\n## References\n"
    if marker not in text:
        raise AssertionError("the page has no references section to resolve its citations in")
    return text[text.index(marker) + len(marker) :]


def reference_entries(text: str) -> list[str]:
    """Return the entries of the references section of a page, one line each.

    An entry is a paragraph of that section carrying a DOI, which leaves out the
    sentence that introduces the list.

    Parameters
    ----------
    text : str
        The page to read.

    Returns
    -------
    list of str
        One flattened entry per reference, in the order the page lists them.
    """
    blocks = [flatten(block) for block in references_of(text).split("\n\n")]
    return [block for block in blocks if "doi:" in block]


def parameter_table_citations(text: str) -> list[tuple[str, str, str, str]]:
    """Return every short citation of the Sources column of the parameter table.

    Parameters
    ----------
    text : str
        The disability page.

    Returns
    -------
    list of tuple of str
        One entry per citation written, in reading order, each holding the short
        form as the page writes it, the surname, the year, and the words in
        brackets that tell two references of the same surname and year apart,
        which is empty where the citation carries none.

    Raises
    ------
    AssertionError
        If the page holds no parameter table.
    """
    lines = text.splitlines()
    if PARAMETER_TABLE_HEADER not in lines:
        raise AssertionError("the page has no parameter table to read the citations of")
    citations: list[tuple[str, str, str, str]] = []
    for row in lines[lines.index(PARAMETER_TABLE_HEADER) + 2 :]:
        if not row.startswith("|"):
            break
        sources = row.strip().strip("|").split("|")[-1]
        citations.extend(
            (match.group(0), match.group(1), match.group(2), match.group(3) or "")
            for match in SHORT_CITATION.finditer(sources)
        )
    return citations


def caption_of(text: str) -> list[str]:
    """Return the sentences of the caption written under the animation.

    The caption is the paragraph under the line that shows the GIF. It comes
    back flattened and split into sentences, without the sentences that carry a
    Markdown link, since the README and the home page link the disability page
    differently and are free to differ there.

    Parameters
    ----------
    text : str
        The README or the home page.

    Returns
    -------
    list of str
        One entry per sentence of the caption, in reading order.

    Raises
    ------
    AssertionError
        If the page does not show the animation or writes no caption under it.
    """
    lines = text.splitlines()
    shown = [index for index, line in enumerate(lines) if f"{ANIMATION_IMAGE})" in line]
    if not shown:
        raise AssertionError("the page does not show the animation, so it captions nothing")
    start = shown[0] + 1
    while start < len(lines) and not lines[start].strip():
        start += 1
    end = start
    while end < len(lines) and lines[end].strip():
        end += 1
    caption = flatten("\n".join(lines[start:end]))
    if not caption:
        raise AssertionError("the page shows the animation and writes no caption under it")
    return [sentence for sentence in SENTENCE_BREAK.split(caption) if MARKDOWN_LINK not in sentence]


def package_modules() -> set[str]:
    """Return the modules of the package that a reader can import by name."""
    return {
        path.stem
        for path in PACKAGE.glob("*.py")
        if path.stem != "__init__" and not path.stem.startswith("_")
    }


def swept() -> set[str]:
    """Return the path of every file of the sweep, relative to the repository."""
    return {path.relative_to(ROOT).as_posix() for path in text_files()}


def offenders(numbers: dict[str, list[int]]) -> list[str]:
    """Return one ``path:line`` marker for every line a rule rejected."""
    return [f"{path}:{number}" for path, lines in numbers.items() for number in lines]


def sweep(rule: Callable[[str], list[int]], *, suffix: str | None = None) -> list[str]:
    """Return one ``path:line`` marker for every line of the repository a rule rejects.

    Parameters
    ----------
    rule : callable
        One of the house style rules of ``_helpers``, which takes the text of a
        file and returns the number of every line it rejects.
    suffix : str, optional
        Read only the files with this suffix, such as ``".md"``. Default None,
        which reads every text file of the repository.

    Returns
    -------
    list of str
        One marker per rejected line, empty when the repository keeps the rule.
    """
    return offenders(
        {
            path.relative_to(ROOT).as_posix(): rule(text)
            for path, text in text_files().items()
            if suffix is None or path.suffix == suffix
        }
    )


def role_markers(text: str) -> list[str]:
    """Return every Sphinx cross reference role marker a piece of text holds.

    Parameters
    ----------
    text : str
        The text to read, a module or a built page.

    Returns
    -------
    list of str
        One entry per marker found, without duplicates, in the order of
        ``ROLE_MARKERS``.
    """
    return [marker for marker in ROLE_MARKERS if marker in text]


def package_sources() -> dict[str, str]:
    """Return the text of every module of the package, by path relative to the root."""
    return {path.relative_to(ROOT).as_posix(): read(path) for path in sorted(PACKAGE.glob("*.py"))}


def python_sources() -> dict[str, str]:
    """Return the text of every Python file of the repository.

    The files come from the same walk the house style sweep uses, which prunes
    version control, the virtual environment, the built site and the caches and
    leaves out what .gitignore names. A module added under a new directory is
    therefore read without anybody remembering to add it here.

    Returns
    -------
    dict of str to str
        The content of each file, by path relative to the repository.
    """
    return {
        path.relative_to(ROOT).as_posix(): text
        for path, text in text_files().items()
        if path.suffix == ".py"
    }


def literal_block_lines(text: str) -> list[int]:
    """Return the number of every line that ends in a literal block marker.

    Parameters
    ----------
    text : str
        The text to read, a module of the package.

    Returns
    -------
    list of int
        Line numbers, counted from one.
    """
    return [
        number
        for number, line in enumerate(text.splitlines(), 1)
        if line.rstrip().endswith(LITERAL_BLOCK_MARKER)
    ]


def underlines_a_section(stripped: str, previous: str) -> bool:
    """Return whether a line is the underline of a numpydoc section.

    Parameters
    ----------
    stripped : str
        The line to read, without its indentation.
    previous : str
        The line above it, without its indentation.

    Returns
    -------
    bool
        True when the line is a run of hyphens as long as the title above it.
    """
    return set(stripped) == {"-"} and len(stripped) == len(previous)


def table_rule_lines(text: str) -> list[int]:
    """Return the number of every line that draws the rule of a reStructuredText table.

    Parameters
    ----------
    text : str
        The text to read, a module of the package.

    Returns
    -------
    list of int
        Line numbers, counted from one. A numpydoc section underline is the one
        line of that shape that is not a table, and it is left out.
    """
    found: list[int] = []
    previous = ""
    for number, line in enumerate(text.splitlines(), 1):
        stripped = line.strip()
        if (
            len(stripped) >= MINIMUM_RULE_CHARACTERS
            and TABLE_RULE.fullmatch(stripped)
            and not underlines_a_section(stripped, previous)
        ):
            found.append(number)
        previous = stripped
    return found


def missing_build_tools(mkdocs: bool, uv: bool) -> list[str]:
    """Return the names of the tools a site build needs that are not present.

    Parameters
    ----------
    mkdocs : bool
        Whether mkdocs can be imported.
    uv : bool
        Whether the uv executable is on the path.

    Returns
    -------
    list of str
        The missing names, empty when the site can be built here.
    """
    return [name for name, present in (("mkdocs", mkdocs), ("uv", uv)) if not present]


def site_build_tools_missing() -> list[str]:
    """Return what this environment lacks for a site build, empty when it lacks nothing."""
    return missing_build_tools(find_spec("mkdocs") is not None, shutil.which("uv") is not None)


def build_site(site_dir: Path) -> subprocess.CompletedProcess[str]:
    """Build the documentation site into a directory and return the finished run."""
    environment = dict(os.environ)
    # The virtual environment of the caller is not the one uv resolves the
    # project against, and leaving it set makes uv warn and then ignore it.
    environment.pop("VIRTUAL_ENV", None)
    return subprocess.run(
        ["uv", "run", "--no-sync", "mkdocs", "build", "--strict", "--site-dir", str(site_dir)],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )


def test_site_is_named_after_the_package() -> None:
    config = load_config()
    assert config["site_name"] == "msrelapse"
    assert "renato-umeton/multiple-sclerosis-modeling-bordi" in config["repo_url"]
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
    text = read(DOCS / "api" / f"{page}.md")
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
        for line in read(path).splitlines():
            if line.startswith("::: msrelapse."):
                documented.add(line.removeprefix("::: ").strip())
    private = {name for name in documented if name.rpartition(".")[2].startswith("_")}
    assert private == {"msrelapse._params"}, "docs/api/ documents a private module of its own"


def test_role_markers_reads_a_role_and_walks_past_a_plain_colon() -> None:
    assert role_markers(f"see {ROLE_MARKERS[0]}`calibrate`") == [ROLE_MARKERS[0]]
    assert role_markers("Returns: the mean duration in weeks") == []


def test_no_python_file_of_the_repository_holds_a_sphinx_role() -> None:
    """The docstrings are read by mkdocstrings, which renders a role as literal text.

    A cross reference is written ``[`name`][msrelapse.module.name]`` instead,
    which mkdocs-autorefs turns into a link, and anything that is not an object
    of this package is written as a code span. The whole file is read rather
    than its docstrings alone, so that the comments keep the same rule, and
    every Python file is read rather than the package alone, so that a role has
    nowhere left to sit and be copied from.
    """
    found = {
        path: markers for path, text in python_sources().items() if (markers := role_markers(text))
    }
    assert found == {}, f"a Sphinx role is rendered by nothing here and reads as itself: {found}"


def test_the_role_guard_reads_the_package_the_tests_and_the_notebook_builder() -> None:
    """The guard is worth nothing if it walks past the files that matter."""
    reached = set(python_sources())
    for relative in (
        "src/msrelapse/fit.py",
        "tests/_helpers.py",
        "tests/test_fit.py",
        "tests/test_metadata.py",
        "notebooks/build_notebooks.py",
    ):
        assert relative in reached, f"the role guard never reads {relative}"


def test_a_literal_block_marker_is_read_and_a_plain_colon_is_not() -> None:
    assert literal_block_lines(f"the scheme is{LITERAL_BLOCK_MARKER}\n\n    x = 1") == [1]
    assert literal_block_lines("Returns: the mean duration in weeks") == []


# This guard and the table one below it read the modules of the package, while
# the role guard above reads every Python file of the repository. The package is
# what mkdocstrings renders, so it is where either piece of reStructuredText
# does its damage, and it is also the only part of the repository that keeps
# these two rules today: the module docstring of notebooks/build_notebooks.py
# still opens two literal blocks of its own. Widening the two waits on that file
# being reworded.
def test_no_module_of_the_package_opens_a_literal_block() -> None:
    """Two colons at the end of a line are reStructuredText, which nothing here renders.

    mkdocstrings reads the docstrings as Markdown, where the marker stays on the
    page and the block under it runs together with the prose. A single colon
    with an indented block under it is the code block the writer meant.
    """
    found = offenders({path: literal_block_lines(text) for path, text in package_sources().items()})
    assert found == [], f"a line ends in the reStructuredText literal block marker: {found}"


def test_a_numpydoc_underline_is_not_read_as_the_rule_of_a_table() -> None:
    title = "Parameters"
    assert table_rule_lines(f"{title}\n{'-' * len(title)}\nseed : int") == []


def test_a_pair_of_rule_characters_is_too_short_to_draw_a_table() -> None:
    """Below the floor the guard would name a stray pair of hyphens a table."""
    for pair in ("-" * 2, "=" * 2):
        assert table_rule_lines(f"a first line\n{pair}\nthe next one") == []


def test_the_rule_of_a_simple_and_of_a_grid_table_are_both_read() -> None:
    simple = "=" * 5 + "  " + "=" * 5
    grid = "+" + "-" * 6 + "+" + "-" * 6 + "+"
    assert table_rule_lines(f"Quantity  Value\n{simple}\nrelapse   4.3") == [2]
    assert table_rule_lines(f"{grid}\n| a      | b      |\n{grid}") == [1, 3]


def test_no_module_of_the_package_draws_a_restructuredtext_table() -> None:
    """Neither table shape survives the trip through Markdown.

    A simple table and a grid table both reach the built page as the rule
    characters themselves, one long line of them, so a table written in a
    docstring is written as a Markdown table or as a bulleted list instead.
    """
    found = offenders({path: table_rule_lines(text) for path, text in package_sources().items()})
    assert found == [], (
        f"a reStructuredText table reaches the site as its own rule characters: {found}"
    )


def test_missing_build_tools_names_only_what_is_absent() -> None:
    assert missing_build_tools(mkdocs=True, uv=True) == []
    assert missing_build_tools(mkdocs=False, uv=True) == ["mkdocs"]
    assert missing_build_tools(mkdocs=True, uv=False) == ["uv"]
    assert missing_build_tools(mkdocs=False, uv=False) == ["mkdocs", "uv"]


@pytest.mark.slow
@pytest.mark.skipif(
    bool(site_build_tools_missing()) and not os.environ.get(REQUIRE_DOCS),
    reason="mkdocs or uv is missing here, so the site cannot be built",
)
def test_the_built_api_pages_carry_no_role_and_no_unresolved_reference(tmp_path: Path) -> None:
    """The site is built the way continuous integration builds it, then read back.

    The check above reads the sources; this one reads what a visitor of the site
    is served, so a role that survives the handler, and a cross reference that
    resolves to nothing, are both caught on the page itself.

    The build needs mkdocs, so the check is skipped where the documentation group
    is not installed. An environment that is meant to have it sets
    ``MSRELAPSE_REQUIRE_DOCS``, and the skip turns into the failure below, which
    names what is missing.
    """
    missing = site_build_tools_missing()
    assert missing == [], (
        f"{REQUIRE_DOCS} is set, so this environment has to be able to build the "
        f"site, and it is missing {', '.join(missing)}"
    )
    site = tmp_path / "site"

    completed = build_site(site)

    assert completed.returncode == 0, completed.stderr
    output = completed.stdout + completed.stderr
    assert UNRESOLVED_REFERENCE not in output, output
    # The site uses directory URLs, so each page is written as api/<name>/index.html.
    pages = sorted((site / "api").rglob("*.html"))
    assert {page.parent.name for page in pages} == set(API_PAGES)
    found = {
        page.relative_to(site).as_posix(): markers
        for page in pages
        if (markers := role_markers(read(page)))
    }
    assert found == {}, f"a built API page shows a Sphinx role as literal text: {found}"


def test_contains_em_dash_finds_the_character_and_nothing_else() -> None:
    assert contains_em_dash(f"a sentence broken {EM_DASH} in two")
    assert not contains_em_dash("a sentence broken, in two, with a hyphen-joined word")
    assert em_dash_lines(f"a first line\na sentence broken {EM_DASH} in two") == [2]


def test_a_bare_line_of_hyphens_is_a_violation() -> None:
    assert has_three_hyphen_line(f"a paragraph\n{HYPHEN_LINE}\nthe next one")
    assert three_hyphen_lines(f"a paragraph\n{HYPHEN_LINE}\nthe next one") == [2]


def test_a_markdown_table_separator_row_is_not_a_violation() -> None:
    table = f"| Quantity | Value |\n|{HYPHEN_LINE}|{HYPHEN_LINE}|\n| mean relapse | 4.3 |"
    assert not has_three_hyphen_line(table)


def test_a_line_of_hyphens_inside_a_fenced_block_is_not_a_violation() -> None:
    block = f"{FENCE}yaml\n{HYPHEN_LINE}\nname: CI\n{FENCE}"
    assert not has_three_hyphen_line(block)


def test_only_the_marker_that_opened_a_fence_closes_it() -> None:
    """A block opened with one marker runs to the matching one, not to the other."""
    mixed = f"{FENCE}text\n{OTHER_FENCE}\n{HYPHEN_LINE}\n{FENCE}"
    assert three_hyphen_lines(mixed) == []
    closed = f"{FENCE}text\nquoted\n{FENCE}\n{HYPHEN_LINE}"
    assert three_hyphen_lines(closed) == [4]


def test_a_numpydoc_underline_is_a_violation_only_outside_python() -> None:
    title = "Parameters"
    section = f"{title}\n{'-' * len(title)}\nseed : int"
    assert not has_three_hyphen_line(section, python_source=True)
    assert has_three_hyphen_line(section)


def test_an_underline_shorter_than_its_title_is_a_violation_in_python_too() -> None:
    """A section underline runs the length of the title, which is what tells it apart."""
    assert three_hyphen_lines(f"Parameters\n{HYPHEN_LINE}", python_source=True) == [2]


def test_a_line_of_hyphens_under_a_line_of_code_is_a_violation_in_python() -> None:
    code = "value = compute(seed)"
    assert three_hyphen_lines(f"{code}\n{HYPHEN_LINE}", python_source=True) == [2]


def test_a_line_of_hyphens_under_a_sentence_is_a_violation_in_python_too() -> None:
    prose = "a sentence long enough that no numpydoc section of any module opens with it"
    assert has_three_hyphen_line(f"{prose}\n{HYPHEN_LINE}", python_source=True)


def test_a_named_assistant_is_a_violation_whatever_it_runs_into() -> None:
    """The name is caught whatever the case, and a following digit does not hide it."""
    for name in ASSISTANT_NAMES:
        assert assistant_mention_lines(f"drafted with {name.upper()} at hand") == [1]
        assert assistant_mention_lines(f"drafted with {name}4 at hand") == [1]
    assert assistant_mention_lines(f"drafted with {CHAT_ASSISTANT.upper()} at hand") == [1]


def test_the_claim_that_a_machine_wrote_a_file_is_a_violation() -> None:
    assert assistant_mention_lines(f"this page was {ASSISTANT_PHRASE} last spring") == [1]


def test_a_word_of_the_domain_is_not_an_assistant() -> None:
    assert assistant_mention_lines("the model of the article is an asymmetric double well") == []
    assert assistant_mention_lines("the work was funded by a philanthropic donor") == []


def test_a_single_trailing_space_is_a_violation_and_a_hard_break_is_not() -> None:
    assert single_trailing_space_lines("a line that ends in one space \nand the next") == [1]
    assert single_trailing_space_lines("a Markdown hard break  \nand the next") == []


def test_the_sweep_reads_the_sources_the_tests_and_the_notebooks() -> None:
    """The sweep is worth nothing if it walks past the files that matter."""
    reached = swept()
    for relative in (
        "README.md",
        "pyproject.toml",
        "mkdocs.yml",
        "src/msrelapse/model.py",
        "src/msrelapse/data/PROVENANCE.txt",
        "tests/conftest.py",
        "notebooks/build_notebooks.py",
        "notebooks/01_reproduce_bordi2013.ipynb",
        "docs/index.md",
        "docs/api/model.md",
        ".github/workflows/ci.yml",
    ):
        assert relative in reached, f"the house style sweep never reads {relative}"


def test_the_sweep_leaves_out_the_publisher_pdf_and_the_directories_of_no_prose() -> None:
    reached = swept()
    assert [path for path in reached if path.endswith(".pdf")] == []
    assert [path for path in reached if path.startswith((".venv/", "site/", ".git/"))] == []


def test_the_local_planning_notes_are_named_by_gitignore() -> None:
    """What keeps the notes out of the sweep is the entry, not their absence."""
    assert "docs/plan1.md" in gitignored_paths()


@pytest.mark.skipif(
    not (ROOT / "docs" / "plan1.md").is_file(),
    reason="the local planning notes are not in the repository, so a checkout has none",
)
def test_the_local_planning_notes_stay_out_of_the_sweep() -> None:
    assert "docs/plan1.md" not in swept()


def test_the_walk_prunes_a_directory_that_holds_nothing_written_by_hand(tmp_path: Path) -> None:
    (tmp_path / "page.md").write_text("a page\n", encoding="utf-8")
    cache = tmp_path / "__pycache__"
    cache.mkdir()
    (cache / "page.md").write_text("a cached page\n", encoding="utf-8")
    reached = {path.relative_to(tmp_path).as_posix() for path in text_files(tmp_path)}
    assert reached == {"page.md"}


def test_the_walk_leaves_out_a_path_gitignore_names(tmp_path: Path) -> None:
    (tmp_path / ".gitignore").write_text("notes.md\n", encoding="utf-8")
    (tmp_path / "notes.md").write_text("local notes\n", encoding="utf-8")
    (tmp_path / "page.md").write_text("a page\n", encoding="utf-8")
    reached = {path.relative_to(tmp_path).as_posix() for path in text_files(tmp_path)}
    assert reached == {".gitignore", "page.md"}


def test_the_walk_skips_a_file_that_does_not_decode_as_text(tmp_path: Path) -> None:
    (tmp_path / "page.md").write_text("a page\n", encoding="utf-8")
    (tmp_path / "picture.bin").write_bytes(bytes(range(256)))
    reached = {path.relative_to(tmp_path).as_posix() for path in text_files(tmp_path)}
    assert reached == {"page.md"}


def test_the_decisions_page_is_held_to_the_house_style() -> None:
    """Only what git ignores is left out, and the decisions page is in the repository.

    The other page under docs/ that a reader of the nav reaches is swept as a
    matter of course; this one is named because it is the newest, and a page
    that is written but never committed is the way it goes missing.
    """
    reached = swept()
    assert "docs/theory.md" in reached
    assert "docs/decisions.md" in reached, (
        "docs/decisions.md is not in the working tree, so a fresh checkout has no "
        "page to sweep; mkdocs.yml navigates to it, so it belongs in the repository"
    )


def test_no_file_holds_an_em_dash() -> None:
    found = sweep(em_dash_lines)
    assert found == [], f"the em dash is not used anywhere in this repository: {found}"


def test_no_file_holds_a_line_of_hyphens() -> None:
    found = offenders(
        {
            path.relative_to(ROOT).as_posix(): three_hyphen_lines(
                text, python_source=path.suffix == ".py"
            )
            for path, text in text_files().items()
        }
    )
    assert found == [], (
        "a line of three or more hyphens is allowed only as a table separator row, "
        f"inside a fenced block, or under the title of a section of a docstring: {found}"
    )


def test_no_file_names_an_assistant_or_a_model() -> None:
    found = sweep(assistant_mention_lines)
    assert found == [], f"no file of this repository names an assistant or a model: {found}"


def test_no_markdown_line_ends_in_a_single_space() -> None:
    found = sweep(single_trailing_space_lines, suffix=".md")
    assert found == [], (
        f"a Markdown line ends in one trailing space, where two are a hard break: {found}"
    )


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
    assert "$$" in read(DOCS / "theory.md")


def test_the_theory_page_cites_the_work_it_rests_on() -> None:
    text = read(DOCS / "theory.md")
    for author in ("Bordi", "Benzi", "Kramers", "Day", "Zhu", "Keene"):
        assert author in text


def test_the_home_page_quotes_the_summary_sentence_word_for_word() -> None:
    page = flatten(read(DOCS / "index.md"))
    assert summary_sentence() in page, "docs/index.md and README.md summarise the work differently"


def test_the_home_page_shows_the_animation() -> None:
    assert f"]({ANIMATION_IMAGE})" in read(DOCS / "index.md")
    assert (DOCS / ANIMATION_IMAGE).is_file()


def test_the_animation_caption_describes_the_bottom_panel_the_same_way_twice() -> None:
    """The README and the home page say the same thing about the EDSS trace.

    The two captions are written out in full in their own pages, so nothing but
    this check stops one of them from drifting away from the other. They are
    compared sentence by sentence, with the sentences carrying a link left out,
    since only the address of the disability page differs between the two.
    """
    readme = caption_of(read(SUMMARY_SOURCE))
    home = caption_of(read(DOCS / "index.md"))
    assert readme == home, "README.md and docs/index.md caption the animation differently"
    for phrase in CAPTION_PHRASES:
        assert any(phrase in sentence for sentence in home), (
            f"the caption no longer says {phrase!r} about the bottom panel"
        )


def test_the_disability_page_follows_the_theory_page_in_the_nav() -> None:
    targets = nav_targets(load_config()["nav"])
    assert DISABILITY_PAGE in targets, "mkdocs.yml navigates to no disability page"
    assert targets.index(DISABILITY_PAGE) == targets.index(THEORY_PAGE) + 1


def test_the_disability_page_opens_by_saying_it_is_not_the_article() -> None:
    """The warning stands above everything else a reader of that page meets."""
    text = read(DOCS / DISABILITY_PAGE)
    assert text.startswith("# Disability trajectory (illustrative extension)")
    opening = text[: text.index("\n## ")]
    assert "!!! warning" in opening
    for phrase in ("no disability score", "prognosis", "clinical"):
        assert phrase in opening, f"the opening warning does not mention {phrase!r}"


def test_every_doi_of_the_evidence_records_is_cited_on_the_disability_page() -> None:
    """``msrelapse.edss.EVIDENCE`` and the page rest on the same literature.

    The module carries the numbers and the page explains them, so a source that
    reaches one of the two and not the other is a source a reader cannot check.
    """
    cited = set(page_dois(read(DOCS / DISABILITY_PAGE)))
    missing = sorted({record.doi.lower() for record in msrelapse.EVIDENCE} - cited)
    assert missing == [], f"docs/{DISABILITY_PAGE} cites none of these evidence DOIs: {missing}"


def test_every_doi_of_the_disability_page_is_listed_once_in_its_references() -> None:
    """Every DOI the page writes stands in its reference list exactly once.

    The body of the page cites by surname and year and writes no DOI of its
    own, so what this covers is that the list itself holds no entry twice. That
    the short citations reach the entries is the check below.
    """
    text = read(DOCS / DISABILITY_PAGE)
    listed = Counter(page_dois(references_of(text)))
    unresolved = sorted(doi for doi in set(page_dois(text)) if listed[doi] != 1)
    assert unresolved == [], (
        "each of these DOIs is cited on the page and is listed in its references "
        f"either not at all or more than once: {unresolved}"
    )


def test_every_short_citation_of_the_parameter_table_names_one_reference() -> None:
    """The Sources column cites by surname and year, and each form reaches one entry.

    The references at the foot of the page carry the DOIs, so a short form that
    names no entry, or two of them, is a source a reader cannot follow. Two
    surnames appear twice with the same year and are told apart by the words in
    brackets, which have to be read off the entry they point at.
    """
    text = read(DOCS / DISABILITY_PAGE)
    entries = reference_entries(text)
    assert entries, f"docs/{DISABILITY_PAGE} lists no references"
    citations = parameter_table_citations(text)
    assert citations, "the parameter table cites nothing"
    unresolved = set()
    for written, surname, year, bracketed in citations:
        named = [
            entry
            for entry in entries
            if entry.startswith(f"{surname} ")
            and year in entry
            and bracketed.lower() in entry.lower()
        ]
        if len(named) != 1:
            unresolved.add(f"{written} names {len(named)} of them")
    assert unresolved == set(), (
        "each of these citations of the parameter table reaches no reference of "
        f"the page or more than one: {sorted(unresolved)}"
    )


def test_the_animation_assets_are_never_read_as_prose() -> None:
    """The sweep reads text files, and these two are pictures."""
    reached = swept()
    for relative in ANIMATION_ASSETS:
        assert (ROOT / relative).is_file(), f"{relative} is not in the working tree"
        assert relative not in reached


def test_the_citing_page_carries_the_bibtex_the_package_prints() -> None:
    """The page and ``msrelapse.citation()`` give the same two entries.

    Both are written out in full, so nothing but this check stops the page from
    drifting away from the code as the version or the DOI changes.
    """
    page = flatten(read(DOCS / "citing.md"))
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
    page = flatten(read(DOCS / "reproducing.md"))
    missing = [quantity for quantity in table["quantity"] if quantity not in page]
    assert missing == [], "docs/reproducing.md has drifted from reproduction_table"


def test_the_reproduction_page_states_the_window_the_animation_runs_over() -> None:
    """The page and ``animate_double_well`` give the record the same length.

    The page tells a reader how long the committed file follows one patient
    for, in years and in the whole weeks that comes to, and the seed that wrote
    it was chosen over that window. Nothing but this check stops the page from
    keeping a window the animation no longer runs.
    """
    weeks = inspect.signature(plots.animate_double_well).parameters["n_weeks"].default
    page = flatten(read(DOCS / "reproducing.md"))
    assert f"{weeks} weeks" in page, "docs/reproducing.md has drifted from animate_double_well"


def test_the_home_page_sends_each_audience_to_its_entry_point() -> None:
    lines = read(DOCS / "index.md").splitlines()
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


def test_ci_builds_and_checks_the_distribution_on_every_change() -> None:
    """The build job is what keeps the sdist include list and the metadata honest.

    docs/CONTRIBUTING.md promises a contributor that a change to the version, the
    classifiers, the sdist include list or the README rendering fails there, so
    the steps that make the promise true are named here rather than left to
    review.
    """
    jobs = load_workflow("ci.yml")["jobs"]
    assert "build" in jobs, "ci.yml no longer builds the distribution"
    run = " ".join(step.get("run", "") for step in jobs["build"]["steps"])
    assert "uv build" in run
    assert "twine check --strict" in run
    # The shipped suite runs from the unpacked archive, so a file the include
    # list forgets fails here rather than after a download.
    assert "unpacked/msrelapse-*/tests" in run
    # And the wheel is installed away from the checkout, so a missing module or
    # a missing packaged data file fails here too.
    assert "--with dist/*.whl msrelapse cite" in run


def test_ci_hands_its_jobs_a_read_only_token() -> None:
    """These jobs run project code and third party code, and none of them writes.

    The workflow states the permissions rather than inheriting them, and the
    other two workflows state theirs, so the mapping is pinned the way
    tests/test_metadata.py pins the one in release.yml.
    """
    assert load_workflow("ci.yml")["permissions"] == {"contents": "read"}


def test_the_test_matrix_covers_the_platforms_and_versions_the_guide_promises() -> None:
    """docs/CONTRIBUTING.md tells a contributor which platforms the suite runs on."""
    matrix = load_workflow("ci.yml")["jobs"]["test"]["strategy"]["matrix"]
    assert matrix["os"] == ["ubuntu-latest", "macos-latest", "windows-latest"]
    assert matrix["python-version"] == ["3.10", "3.11", "3.12", "3.13", "3.14"]


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
