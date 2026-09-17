"""Tests of the repository metadata: README, citation files, templates, release.

The house style rules these files once carried of their own, no em dash and no
line of hyphens, are kept by the sweep in ``tests/test_docs.py``, which reads
every text file of the repository rather than the list below.

The Python examples written into the prose are run here as well, the README one
and the JOSS one and those of the site pages, because mkdocs renders a block
without ever running it and ruff is pointed away from docs/.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from datetime import date
from pathlib import Path
from typing import Any, NamedTuple, cast

import pytest
import yaml  # type: ignore[import-untyped]

import msrelapse
from _helpers import ROOT, WORKFLOWS, flatten, load_json, load_workflow, load_yaml, read, text_files
from msrelapse._citation import PAPER_BIBTEX
from msrelapse._params import PAPER

REPOSITORY = "https://github.com/renato-umeton/multiple-sclerosis-modeling-bordi"
DOCUMENTATION = "https://renato-umeton.github.io/multiple-sclerosis-modeling-bordi/"

# The files this module reads. The sdist ships the suite, so every one of them
# has to be on the sdist include list of pyproject.toml as well: the packaged
# run of the build job reads them from the unpacked archive, where a file the
# include list forgets fails here rather than passing.
OWNED_FILES = (
    "README.md",
    "CITATION.cff",
    "codemeta.json",
    ".github/ISSUE_TEMPLATE/bug_report.yml",
    ".github/ISSUE_TEMPLATE/feature_request.yml",
    ".github/ISSUE_TEMPLATE/config.yml",
    ".github/dependabot.yml",
    ".github/workflows/release.yml",
    "docs/pull_request_template.md",
    "docs/paper/paper.md",
    "docs/paper/paper.bib",
)

# The files this module reads that a checkout has and an archive has not. The
# pre commit configuration is a gate a contributor runs before a commit, so it
# is of no use to somebody who unpacked the sdist to build the package, and the
# include list of pyproject.toml leaves it out on purpose. The checks that read
# it are therefore skipped in the packaged run and run everywhere else, which is
# what ``unpacked_sdist`` below decides.
CHECKOUT_ONLY_FILES = (".pre-commit-config.yaml",)

# The nine authors of the article, in the order it prints them.
PAPER_AUTHORS = (
    ("Bordi", "Isabella"),
    ("Umeton", "Renato"),
    ("Ricigliano", "Vito A. G."),
    ("Annibali", "Viviana"),
    ("Mechelli", "Rosella"),
    ("Ristori", "Giovanni"),
    ("Grassi", "Francesca"),
    ("Salvetti", "Marco"),
    ("Sutera", "Alfonso"),
)

# The keys the JOSS draft cites and the bibliography has to define.
BIB_KEYS = ("bordi2013", "benzi1983", "kramers1940", "day1983", "zhu2014", "keene2007")

# The site pages whose Python blocks are run, below, the way the README block
# is. Without that, a renamed argument or a dropped export would leave a page
# example broken with every gate green: mkdocs renders a block without running
# it, and ruff is pointed away from docs/ so that the prose keeps its own
# spelling.
RUNNABLE_DOCS_PAGES = (
    "docs/index.md",
    "docs/citing.md",
    "docs/data.md",
    "docs/disability.md",
)

# The pages holding a Python block that is deliberately not run, with the
# reason each one is left out, so that the omission is a decision on the page.
UNRUN_DOCS_PAGES = {
    "docs/paper/paper.md": "run by test_paper_example_usage_runs, which reads its numbers",
    "docs/reproducing.md": "reads a registry export the reader supplies",
    "docs/paper_facts.md": "two formulas quoted from the article, calling nothing",
}

# One installation line of the README: the options it passes to ``uv sync`` and
# the comment that says what they bring in.
UV_SYNC_LINE = re.compile(r"^uv sync (?P<options>[^#\n]*?)\s+#\s*(?P<comment>.+)$", re.MULTILINE)

# A word of such a comment, hyphens and dots kept so that a distribution name
# arrives whole.
COMMENT_WORD = re.compile(r"[A-Za-z][\w.-]*")

# The words those comments use that name no package: "the API reference build"
# says what the docs group is for, and "the kernel" is how the README refers to
# ipykernel. Every other word of one of them is read as a package the line claims
# to install and is looked up in the group or the extra it names, which is what
# keeps a name the project has dropped, papermill for one, from sitting there
# unnoticed.
SETUP_COMMENT_PROSE = frozenset({"and", "the", "api", "reference", "build", "kernel"})

# The hooks a commit has to pass before it is written, which are the three
# gates continuous integration runs again. The notebooks are committed with
# their outputs in place, so no hook strips them.
PRE_COMMIT_HOOKS = ("ruff-check", "ruff-format", "mypy")

# The animation the README shows, the contact sheet beside it, and the ceiling
# each of the two is kept under. Both are committed, so a clone carries them,
# and the GIF is also shipped inside the sdist under docs/.
ANIMATION_GIF = "docs/assets/double_well.gif"
ANIMATION_CONTACT_SHEET = "docs/assets/double_well_frames.png"
MAX_GIF_BYTES = 3 * 1024 * 1024
MAX_CONTACT_SHEET_BYTES = 1024 * 1024

# The published page of the illustrative disability extension. The README ships
# as the package long description, where a relative link breaks, so its caption
# reaches that page by its address on the site.
DISABILITY_URL = f"{DOCUMENTATION}disability/"

# Every action the workflows are allowed to call, at the version the repository
# standardised on. A new action, or a bumped version, is named here first.
PINNED_ACTIONS = frozenset(
    {
        "actions/checkout@v7",
        "astral-sh/setup-uv@v10.1.0",
        "actions/upload-artifact@v7",
        "actions/download-artifact@v8",
        "codecov/codecov-action@v7",
        "pypa/gh-action-pypi-publish@release/v1",
    }
)


def unpacked_sdist(root: Path = ROOT) -> bool:
    """Return whether a directory is an unpacked sdist rather than a checkout.

    hatchling writes PKG-INFO into the root of every archive it builds and no
    checkout carries one, so the file is what tells the two apart. Reading the
    absence of a development file instead would turn a deleted file into a
    silent skip, which is how a guard stops guarding.

    Parameters
    ----------
    root : pathlib.Path, optional
        The directory to look at. Default the repository root; a test passes a
        directory of its own instead.

    Returns
    -------
    bool
        True when the directory holds the metadata file of a built archive.
    """
    return (root / "PKG-INFO").is_file()


def sdist_include() -> tuple[str, ...]:
    """Return the entries of the sdist include list of pyproject.toml.

    The file is read rather than parsed as TOML, for the reason
    ``pyproject_table`` gives: ``tomllib`` arrived in Python 3.11 and this suite
    runs from 3.10 on.

    Returns
    -------
    tuple of str
        One entry per line of the list, in the order it is written.
    """
    text = read("pyproject.toml")
    marker = "\n[tool.hatch.build.targets.sdist]\n"
    start = text.find(marker)
    assert start >= 0, "pyproject.toml declares no sdist build target"
    body = text[start + len(marker) :]
    opening = body.find("include = [")
    assert opening >= 0, "the sdist build target of pyproject.toml declares no include list"
    body = body[opening:]
    return tuple(re.findall(r'"([^"]+)"', body[: body.index("]")]))


def shipped_in_the_sdist(relative: str, include: Iterable[str]) -> bool:
    """Return whether an include list carries one file into the archive.

    Parameters
    ----------
    relative : str
        The path of the file, relative to the repository root, in POSIX form.
    include : iterable of str
        The entries of the include list. An entry is either the file itself or
        a directory, in which case it carries the whole tree under it.

    Returns
    -------
    bool
        True when at least one entry carries the file.
    """
    return any(relative == entry or relative.startswith(f"{entry}/") for entry in include)


def code_blocks(markdown: str, language: str) -> list[str]:
    """Return the body of every fenced block of one language."""
    pattern = re.compile(rf"^```{language}\n(.*?)^```", re.DOTALL | re.MULTILINE)
    return pattern.findall(markdown)


def section(markdown: str, heading: str) -> str:
    """Return the text under one heading, up to the next heading of the same level."""
    marker = f"\n{heading}\n"
    start = markdown.index(marker) + len(marker)
    level = heading.split(" ", 1)[0]
    rest = markdown[start:]
    # Blank out the fenced blocks, keeping every offset, so that a comment line
    # inside one is not mistaken for the next heading.
    masked = re.sub(
        r"```.*?```", lambda block: re.sub(r"[^\n]", " ", block.group(0)), rest, flags=re.DOTALL
    )
    match = re.search(rf"^{level} ", masked, re.MULTILINE)
    return rest if match is None else rest[: match.start()]


@pytest.mark.parametrize("relative", OWNED_FILES)
def test_metadata_file_exists(relative: str) -> None:
    assert (ROOT / relative).is_file()


@pytest.mark.skipif(
    unpacked_sdist(), reason="an archive carries no development configuration to read"
)
@pytest.mark.parametrize("relative", CHECKOUT_ONLY_FILES)
def test_checkout_only_file_exists(relative: str) -> None:
    assert (ROOT / relative).is_file()


def test_an_unpacked_sdist_is_told_apart_from_a_checkout(tmp_path: Path) -> None:
    """The skip above has to reach the packaged run and nothing else."""
    assert not unpacked_sdist(tmp_path)
    (tmp_path / "PKG-INFO").write_text("Metadata-Version: 2.4\n", encoding="utf-8")
    assert unpacked_sdist(tmp_path)


def test_an_include_entry_carries_the_file_itself_and_the_tree_under_it() -> None:
    assert shipped_in_the_sdist("CITATION.cff", ("CITATION.cff",))
    assert shipped_in_the_sdist("docs/paper/paper.md", ("docs",))
    assert not shipped_in_the_sdist("docs/paper/paper.md", ("doc",))
    assert not shipped_in_the_sdist(".pre-commit-config.yaml", ("docs", "CITATION.cff"))


def test_every_file_this_module_reads_is_shipped_in_the_sdist() -> None:
    """The rule stated above OWNED_FILES, checked here rather than left to review.

    The build job of ci.yml unpacks the archive and runs the suite from it, so a
    name added to the list above without a matching entry in pyproject.toml
    fails there, one job after this one and with nothing on the failure to say
    what the cause was. This reads the include list and names it here instead.
    """
    include = sdist_include()
    assert include, "pyproject.toml ships an empty sdist include list"
    missing = [name for name in OWNED_FILES if not shipped_in_the_sdist(name, include)]
    assert missing == [], (
        "the packaged run of the suite reads these and the sdist include list of "
        f"pyproject.toml carries none of them: {missing}"
    )


def test_a_file_the_archive_carries_is_not_left_among_the_checkout_only_ones() -> None:
    """The two tuples above are held apart, so that neither quietly becomes the other.

    A name that reaches the include list is read in the packaged run as well and
    has no business being skipped there, so it belongs in OWNED_FILES instead.
    """
    include = sdist_include()
    shipped = [name for name in CHECKOUT_ONLY_FILES if shipped_in_the_sdist(name, include)]
    assert shipped == [], (
        "the sdist carries these now, so the packaged run can read them and they "
        f"belong in OWNED_FILES rather than being skipped there: {shipped}"
    )


def test_readme_title_is_the_package_name() -> None:
    assert read("README.md").splitlines()[0] == "# msrelapse"


def test_readme_shows_the_ci_license_and_python_badges() -> None:
    readme = read("README.md")
    assert f"{REPOSITORY}/actions/workflows/ci.yml/badge.svg" in readme
    assert "img.shields.io/badge/license-MIT" in readme
    assert "img.shields.io/badge/python-3.10" in readme
    assert "3.14" in readme


def test_readme_keeps_the_pypi_badge_commented_out() -> None:
    readme = read("README.md")
    comments = re.findall(r"<!--(.*?)-->", readme, re.DOTALL)
    commented = "\n".join(comments)
    assert "img.shields.io/pypi/v/msrelapse" in commented
    # The badge may not be live before the first release puts the project there.
    assert "img.shields.io/pypi/v/msrelapse" not in re.sub(
        r"<!--.*?-->", "", readme, flags=re.DOTALL
    )


def test_readme_holds_the_summary_paragraph_verbatim() -> None:
    summary = (
        "In 70 untreated relapsing-remitting MS patients, relapse and remission durations "
        "are both exponentially distributed (means about 4.3 and about 100 weeks) with no "
        "detectable periodicity, so relapse onset is a memoryless process. A noise-driven "
        "asymmetric double-well model reproduces this, and the ratio of the logarithms of "
        "the two mean durations estimates the ratio of the two barrier heights (about 3). "
        f"Bordi, Umeton et al., Int J Genomics 2013, doi:{PAPER.paper_doi.value}."
    )
    assert summary in flatten(read("README.md"))


@pytest.mark.parametrize(
    ("audience", "entry_point"),
    [
        ("modellers", "DoubleWell"),
        ("trial statisticians", "compare_arr"),
        ("in silico trial", "generate"),
    ],
)
def test_readme_names_each_audience_with_an_entry_point(audience: str, entry_point: str) -> None:
    readme = flatten(read("README.md")).lower()
    assert audience.lower() in readme
    assert entry_point.lower() in readme


class InstallCommand(NamedTuple):
    """One commented ``uv sync`` line of the README."""

    line: str
    groups: tuple[str, ...]
    extras: tuple[str, ...]
    comment_words: tuple[str, ...]


def pyproject_table(header: str) -> dict[str, tuple[str, ...]]:
    """Return one table of arrays of pyproject.toml, as distribution names.

    The file is read rather than parsed as TOML, because ``tomllib`` arrived in
    Python 3.11 and this suite runs from 3.10 on. Only a table whose every value
    is an array of requirement strings can be read this way, which is what the
    two dependency tables are.

    Parameters
    ----------
    header : str
        The table to read, such as ``"dependency-groups"``.

    Returns
    -------
    dict of str to tuple of str
        The name of every requirement of each key of the table, with the version
        specifier, the environment marker and any extras dropped.
    """
    text = read("pyproject.toml")
    marker = f"\n[{header}]\n"
    start = text.find(marker)
    assert start >= 0, f"pyproject.toml holds no [{header}] table"
    body = text[start + len(marker) :]
    following = re.search(r"^\[", body, re.MULTILINE)
    if following is not None:
        body = body[: following.start()]
    body = "\n".join(line for line in body.splitlines() if not line.lstrip().startswith("#"))
    keys = list(re.finditer(r"^([A-Za-z][\w.-]*)\s*=", body, re.MULTILINE))
    ends = [key.start() for key in keys[1:]] + [len(body)]
    return {
        key.group(1): tuple(re.findall(r'"\s*([A-Za-z][\w.-]*)', body[key.end() : end]))
        for key, end in zip(keys, ends, strict=True)
    }


def readme_dependency_groups() -> set[str]:
    """Return every dependency group the README names, on a command line or in prose."""
    readme = read("README.md")
    return set(re.findall(r"--group\s+([A-Za-z][\w-]*)", readme)) | set(
        re.findall(r"`([A-Za-z][\w-]*)` dependency group", readme)
    )


def readme_extras() -> set[str]:
    """Return every extra the README names with ``--extra``."""
    return set(re.findall(r"--extra\s+([A-Za-z][\w-]*)", read("README.md")))


def readme_install_commands() -> list[InstallCommand]:
    """Return each commented ``uv sync`` line of the README, read into its parts."""
    commands = []
    for match in UV_SYNC_LINE.finditer(read("README.md")):
        options = match.group("options")
        commands.append(
            InstallCommand(
                line=match.group(0),
                groups=tuple(re.findall(r"--group\s+([A-Za-z][\w-]*)", options)),
                extras=tuple(re.findall(r"--extra\s+([A-Za-z][\w-]*)", options)),
                comment_words=tuple(
                    word.lower() for word in COMMENT_WORD.findall(match.group("comment"))
                ),
            )
        )
    return commands


def packages_not_installed(words: Iterable[str], installed: Iterable[str]) -> list[str]:
    """Return the words of an install comment that name none of the packages.

    A word names a package when it is the name of one of them, or the prefix of
    a family of them as mkdocs is of mkdocs-material. The prose words of
    ``SETUP_COMMENT_PROSE`` claim no package and are passed over.

    Parameters
    ----------
    words : iterable of str
        The words of the comment, in lower case.
    installed : iterable of str
        The distribution names the groups and extras of that line install.

    Returns
    -------
    list of str
        The words that name no installed package, in the order they were given.
    """
    packages = tuple(installed)
    return [
        word
        for word in words
        if word not in SETUP_COMMENT_PROSE
        and not any(package == word or package.startswith(f"{word}-") for package in packages)
    ]


def test_readme_documents_the_uv_setup() -> None:
    readme = flatten(read("README.md"))
    assert "uv sync" in readme
    for extra in ("plot", "fast"):
        assert f"--extra {extra}" in readme
    for group in ("docs", "notebooks"):
        assert f"--group {group}" in readme
    # Each of them has to be one pyproject.toml declares, the dev group the
    # README names in prose rather than on a command line included.
    assert sorted(readme_dependency_groups() - set(pyproject_table("dependency-groups"))) == []
    assert sorted(readme_extras() - set(pyproject_table("project.optional-dependencies"))) == []


def test_readme_setup_comments_name_packages_the_line_installs() -> None:
    """A package the README says a group or an extra brings in is really in it."""
    groups = pyproject_table("dependency-groups")
    extras = pyproject_table("project.optional-dependencies")
    checked = []
    for command in readme_install_commands():
        if not command.groups and not command.extras:
            continue
        installed = {package for group in command.groups for package in groups[group]}
        installed.update(package for extra in command.extras for package in extras[extra])
        assert packages_not_installed(command.comment_words, installed) == [], command.line
        checked.append(command.line)
    # The two group lines and the extras line, so that a reformatted README the
    # reader above no longer matches cannot leave this test checking nothing.
    assert len(checked) >= 3, checked


def test_a_setup_comment_naming_a_package_the_group_no_longer_holds_is_a_violation() -> None:
    """The check above reads a stale name for what it is, papermill for one."""
    notebooks = ("nbformat", "nbclient", "ipykernel", "matplotlib")
    assert packages_not_installed(("nbformat", "and", "papermill"), notebooks) == ["papermill"]
    assert packages_not_installed(("mkdocs",), ("mkdocs-material",)) == []


def test_readme_documents_every_cli_subcommand() -> None:
    readme = read("README.md")
    for subcommand in (
        "reproduce",
        "simulate",
        "animate",
        "fit",
        "test-memoryless",
        "test-periodicity",
        "cite",
        "params",
    ):
        assert f"msrelapse {subcommand}" in readme


def test_readme_says_the_shipped_records_are_synthetic() -> None:
    readme = flatten(read("README.md")).lower()
    assert "synthetic" in readme
    assert "never released" in readme or "was never released" in readme


def test_readme_links_the_documentation_site() -> None:
    readme = flatten(read("README.md"))
    assert DOCUMENTATION in readme
    assert "Pages" in readme


def test_readme_explains_how_to_cite() -> None:
    readme = read("README.md")
    assert "msrelapse.cite()" in readme
    assert PAPER.paper_doi.value in readme
    assert "CITATION.cff" in readme


def test_readme_shows_the_article_bibtex_the_package_prints() -> None:
    """The block on the page and the entry the package builds are the same entry.

    The README writes the entry out in full, so nothing but this check stops the
    page from drifting away from the code, as the check in tests/test_docs.py
    does for docs/citing.md.
    """
    readme = flatten(read("README.md"))
    missing = [line for line in PAPER_BIBTEX.splitlines() if flatten(line) not in readme]
    assert missing == [], "README.md has drifted from the article entry the package builds"


@pytest.mark.parametrize(
    ("name", "prints"),
    [
        ("cite", True),
        ("citation", False),
        ("provenance", False),
        ("load_synthetic_bordi2013", False),
    ],
)
def test_readme_describes_each_entry_point_with_the_verb_it_deserves(
    name: str, prints: bool, capsys: pytest.CaptureFixture[str]
) -> None:
    """The README says prints only of a function that prints, returns of the rest."""
    result = getattr(msrelapse, name)()
    printed = capsys.readouterr().out
    assert (printed != "") is prints
    assert (result is None) is prints
    match = re.search(rf"`msrelapse\.{name}\(\)` (prints|returns)", flatten(read("README.md")))
    assert match is not None
    assert match.group(1) == ("prints" if prints else "returns")


def test_readme_links_nothing_by_a_relative_path() -> None:
    """README.md ships as the package long description, where relative links break.

    An image is the one exception, and the animation is the one image of the
    README that is a file of this repository rather than a badge: GitHub
    resolves it against the commit the page belongs to, which an absolute link
    into a branch would not.
    """
    readme = read("README.md")
    images = set(re.findall(r"!\[[^\]]*\]\(([^)\s]+)", readme))
    targets = [target for target in re.findall(r"\]\(([^)\s]+)", readme) if target not in images]
    assert targets
    for target in targets:
        assert target.startswith(("https://", "#")), target


def test_readme_shows_the_animation() -> None:
    assert f"]({ANIMATION_GIF})" in read("README.md")
    assert (ROOT / ANIMATION_GIF).is_file()


def test_readme_caption_sends_the_reader_to_the_disability_page() -> None:
    """The caption names the bottom panel and links the page that explains it."""
    readme = flatten(read("README.md"))
    assert "illustrative EDSS trajectory" in readme
    assert f"]({DISABILITY_URL})" in readme


def test_the_animation_and_its_contact_sheet_stay_small_enough_to_commit() -> None:
    """Both are binary files in the history, so both carry a ceiling."""
    assert (ROOT / ANIMATION_GIF).stat().st_size <= MAX_GIF_BYTES
    assert (ROOT / ANIMATION_CONTACT_SHEET).stat().st_size <= MAX_CONTACT_SHEET_BYTES


def quick_start_block() -> str:
    """Return the one Python block of the README quick start."""
    blocks = code_blocks(section(read("README.md"), "## Quick start"), "python")
    assert len(blocks) == 1
    return blocks[0]


def test_readme_quick_start_runs(capsys: pytest.CaptureFixture[str]) -> None:
    exec(compile(quick_start_block(), "README.md quick start", "exec"), {"__name__": "__main__"})
    printed = capsys.readouterr().out
    assert "synthetic" in printed.lower()
    assert PAPER.paper_doi.value in printed
    # Both readings of the barrier ratio are printed: the estimate of equation
    # (7) and the ratio of the barriers of the potential calibrated to the same
    # two means. They are not the same number, which is the point of the note.
    estimate = re.search(r"barrier ratio dV1/dV2 (\d+\.\d+)", printed)
    potential = re.search(r"dV1 \d+\.\d+  dV2 \d+\.\d+  ratio (\d+\.\d+)", printed)
    assert estimate is not None
    assert potential is not None
    assert float(potential.group(1)) > float(estimate.group(1))


def test_readme_quick_start_says_why_the_two_barrier_ratios_differ() -> None:
    assert "prefactor" in quick_start_block()


def test_citation_cff_carries_the_required_keys() -> None:
    cff = load_yaml("CITATION.cff")
    assert cff["cff-version"] == "1.2.0"
    assert "cite" in cff["message"].lower()
    assert cff["title"].startswith("msrelapse")
    assert cff["license"] == "MIT"
    assert cff["repository-code"] == REPOSITORY
    assert str(cff["date-released"]) == "2026-09-15"
    assert cff["authors"] == [{"family-names": "Umeton", "given-names": "Renato"}]


def test_citation_cff_version_matches_the_package() -> None:
    assert str(load_yaml("CITATION.cff")["version"]) == msrelapse.__version__


def test_citation_cff_asks_for_both_the_software_and_the_article() -> None:
    message = load_yaml("CITATION.cff")["message"].lower()
    assert "software" in message
    assert "paper" in message or "article" in message


def test_citation_cff_prefers_the_article() -> None:
    preferred = load_yaml("CITATION.cff")["preferred-citation"]
    assert preferred["type"] == "article"
    assert preferred["title"].startswith("A mechanistic, stochastic model")
    assert preferred["journal"] == "International Journal of Genomics"
    assert int(preferred["volume"]) == 2013
    assert int(preferred["start"]) == 910321
    assert int(preferred["year"]) == 2013
    assert preferred["doi"] == PAPER.paper_doi.value


def test_citation_cff_lists_all_nine_authors_in_order() -> None:
    preferred = load_yaml("CITATION.cff")["preferred-citation"]
    names = [(author["family-names"], author["given-names"]) for author in preferred["authors"]]
    assert names == list(PAPER_AUTHORS)


def test_citation_cff_defers_the_orcid_to_a_comment_and_carries_no_doi_of_its_own() -> None:
    """The software is archived under no DOI, and the ORCID is still to hand in."""
    text = read("CITATION.cff")
    cff = load_yaml("CITATION.cff")
    assert "doi" not in cff
    assert all("orcid" not in author for author in cff["authors"])
    comments = [line.strip() for line in text.splitlines() if line.strip().startswith("#")]
    joined = " ".join(comments).lower()
    assert "orcid" in joined


def test_codemeta_describes_the_package() -> None:
    meta = load_json("codemeta.json")
    assert meta["@context"] == "https://w3id.org/codemeta/3.0"
    assert meta["name"] == "msrelapse"
    assert meta["version"] == msrelapse.__version__
    assert meta["license"] == "https://spdx.org/licenses/MIT"
    assert meta["codeRepository"] == REPOSITORY
    assert meta["programmingLanguage"] == "Python"
    assert meta["dateCreated"] == "2026-09-15"
    assert meta["developmentStatus"] == "active"
    assert meta["description"]
    assert meta["keywords"]


def test_codemeta_runs_on_every_supported_python() -> None:
    platforms = " ".join(load_json("codemeta.json")["runtimePlatform"])
    for minor in range(10, 15):
        assert f"3.{minor}" in platforms


def test_codemeta_lists_the_runtime_dependencies() -> None:
    assert load_json("codemeta.json")["softwareRequirements"] == ["numpy", "scipy", "pandas"]


def test_codemeta_names_the_author() -> None:
    author = load_json("codemeta.json")["author"]
    assert author == [{"@type": "Person", "givenName": "Renato", "familyName": "Umeton"}]


def test_codemeta_references_the_article() -> None:
    reference = load_json("codemeta.json")["referencePublication"]
    assert reference["identifier"] == f"https://doi.org/{PAPER.paper_doi.value}"


def test_codemeta_records_that_the_software_carries_no_identifier() -> None:
    """A harvester reads the absent field, and the comment says why it is absent."""
    meta = load_json("codemeta.json")
    assert "identifier" not in meta
    comment = meta["comment"].lower()
    assert "no identifier" in comment
    assert "article" in comment
    # The other two notes the key carries, which the dates test below relies on.
    assert "orcid" in comment
    assert "datepublished" in comment


def test_codemeta_dates_are_iso_and_the_last_change_is_not_before_the_release() -> None:
    """An aggregator reads both dates, and neither is written by a tool that checks it.

    The comment of codemeta.json says the release date is the one CITATION.cff
    carries as ``date-released`` and that the two are set together, so the pair
    is tied here rather than left to be remembered at the next release. PyYAML
    reads an unquoted ISO date as a ``datetime.date``, which is why the one from
    the citation file is compared as its string form.
    """
    meta = load_json("codemeta.json")
    published = date.fromisoformat(meta["datePublished"])
    modified = date.fromisoformat(meta["dateModified"])
    assert modified >= published
    assert meta["datePublished"] == str(load_yaml("CITATION.cff")["date-released"])


def test_no_file_of_the_repository_names_the_archive_service() -> None:
    """Every citation points at the article, and nothing promises an archive DOI.

    The maintainer decided against depositing the software, so the name of the
    service belongs in no file of the repository: not in a badge, not in a
    comment of a metadata file, not in a release step and not in a page. This
    module is the one exception, because the rule has to be written down
    somewhere to be checked, and the sweep therefore leaves its own file out.
    """
    here = Path(__file__).resolve()
    named = sorted(
        path.relative_to(ROOT).as_posix()
        for path, text in text_files().items()
        if "zenodo" in text.lower() and path != here
    )
    assert named == [], f"these files still name the archive service: {named}"


@pytest.mark.skipif(
    unpacked_sdist(), reason="an archive carries no development configuration to read"
)
def test_pre_commit_runs_the_gates_a_commit_has_to_pass() -> None:
    """The hooks are what a contributor gets before continuous integration sees the change."""
    config = load_yaml(".pre-commit-config.yaml")
    ids = {hook["id"] for repo in config["repos"] for hook in repo["hooks"]}
    assert set(PRE_COMMIT_HOOKS) <= ids, f".pre-commit-config.yaml declares only {sorted(ids)}"


@pytest.mark.parametrize("name", ["bug_report", "feature_request"])
def test_issue_form_is_a_github_form(name: str) -> None:
    form = load_yaml(f".github/ISSUE_TEMPLATE/{name}.yml")
    assert form["name"]
    assert form["description"]
    assert form["body"]
    for field in form["body"]:
        assert field["type"] in {"markdown", "input", "textarea", "dropdown", "checkboxes"}
        assert "label" in field["attributes"] or field["type"] == "markdown"


def test_bug_report_asks_what_a_maintainer_needs() -> None:
    form = load_yaml(".github/ISSUE_TEMPLATE/bug_report.yml")
    ids = {field.get("id") for field in form["body"]}
    assert {"version", "python", "os", "steps", "expected", "actual", "code"} <= ids


def test_feature_request_asks_for_the_problem_and_the_audience() -> None:
    form = load_yaml(".github/ISSUE_TEMPLATE/feature_request.yml")
    fields = {field.get("id"): field for field in form["body"]}
    assert {"problem", "proposal", "alternatives", "audience"} <= set(fields)
    assert fields["audience"]["type"] == "dropdown"
    assert len(fields["audience"]["attributes"]["options"]) >= 3


def test_issue_config_turns_blank_issues_off_and_offers_a_link() -> None:
    config = load_yaml(".github/ISSUE_TEMPLATE/config.yml")
    assert config["blank_issues_enabled"] is False
    links = config["contact_links"]
    assert links
    for link in links:
        assert link["name"]
        assert link["url"].startswith("https://")
        assert link["about"]
    assert any(DOCUMENTATION in link["url"] for link in links)


def test_pull_request_template_lists_the_quality_gates() -> None:
    template = flatten(read("docs/pull_request_template.md")).lower()
    for gate in ("tests", "ruff", "mypy", "changelog", "_params.py", "notebook"):
        assert gate.lower() in template
    assert "- [ ]" in read("docs/pull_request_template.md")


def release_workflow() -> dict[str, Any]:
    """Return release.yml, reading ``on`` past the YAML rule that turns it into True."""
    raw = load_workflow("release.yml")
    return {("on" if key is True else str(key)): value for key, value in raw.items()}


def test_release_workflow_triggers_on_version_tags() -> None:
    assert release_workflow()["on"] == {"push": {"tags": ["v*"]}}


def test_release_workflow_defaults_to_read_only_permissions() -> None:
    """The build job runs project code, so it inherits nothing beyond reading."""
    assert release_workflow()["permissions"] == {"contents": "read"}


def test_release_workflow_refuses_a_tag_that_does_not_match_the_built_version() -> None:
    build = release_workflow()["jobs"]["build"]
    guards = [step for step in build["steps"] if "GITHUB_REF_NAME" in step.get("run", "")]
    assert len(guards) == 1
    assert "exit 1" in guards[0]["run"]


def test_release_workflow_builds_and_uploads_the_distribution() -> None:
    build = release_workflow()["jobs"]["build"]
    runs = " ".join(step.get("run", "") for step in build["steps"])
    uses = [step.get("uses", "") for step in build["steps"]]
    assert "uv build" in runs
    assert any(action.startswith("actions/checkout@") for action in uses)
    assert any(action.startswith("astral-sh/setup-uv@") for action in uses)
    assert any(action.startswith("actions/upload-artifact@") for action in uses)


def test_release_workflow_publishes_by_trusted_publishing() -> None:
    publish = release_workflow()["jobs"]["publish"]
    assert publish["needs"] == "build"
    assert publish["environment"]["name"] == "pypi"
    assert publish["permissions"]["id-token"] == "write"
    steps = publish["steps"]
    assert any(step.get("uses", "").startswith("pypa/gh-action-pypi-publish@") for step in steps)
    # Trusted publishing carries no token; a password input would mean a secret.
    assert all("password" not in step.get("with", {}) for step in steps)


def test_release_workflow_creates_a_github_release_with_the_distribution() -> None:
    job = release_workflow()["jobs"]["github-release"]
    # The release waits for the publish, so that a failed publish cannot leave a
    # GitHub release, and an archive minted from it, standing on its own.
    assert job["needs"] == ["build", "publish"]
    assert job["permissions"]["contents"] == "write"
    runs = " ".join(step.get("run", "") for step in job["steps"])
    assert "gh release create" in runs
    assert "dist/" in runs


@pytest.mark.parametrize(
    "action",
    [
        "actions/checkout@v7",
        "astral-sh/setup-uv@v10.1.0",
        "pypa/gh-action-pypi-publish@release/v1",
    ],
)
def test_release_workflow_pins_the_actions(action: str) -> None:
    assert action in read(".github/workflows/release.yml")


def workflow_actions() -> dict[str, list[str]]:
    """Return the action every ``uses:`` line names, workflow file by workflow file."""
    paths = sorted(path for path in WORKFLOWS.iterdir() if path.suffix in {".yml", ".yaml"})
    pattern = re.compile(r"^\s*(?:-\s*)?uses:\s*(\S+)", re.MULTILINE)
    return {path.name: pattern.findall(read(path)) for path in paths}


def test_every_workflow_calls_only_the_pinned_actions() -> None:
    """release.yml is pinned above; most of the ``uses:`` lines live in the other two."""
    called = workflow_actions()
    assert called
    assert all(actions for actions in called.values()), called
    unpinned = sorted(
        f"{name} {action}"
        for name, actions in called.items()
        for action in actions
        if action not in PINNED_ACTIONS
    )
    assert unpinned == []


def test_dependabot_watches_the_workflow_actions_and_nothing_else() -> None:
    """The pinned versions above are bumped by a pull request, not by a drift.

    The actions are pinned word for word, so nothing bumps them on its own. This
    file is what says when a bump is due. The Python dependencies are locked in
    uv.lock, which uv writes, so exactly one ecosystem is watched and it is the
    actions one.
    """
    config = load_yaml(".github/dependabot.yml")
    assert config["version"] == 2
    updates = config["updates"]
    assert len(updates) == 1
    assert updates[0]["package-ecosystem"] == "github-actions"
    # The workflows live under the repository root, which is where the actions
    # ecosystem looks for them.
    assert updates[0]["directory"] == "/"
    assert updates[0]["schedule"]["interval"]


def test_release_workflow_opens_with_the_one_time_setup() -> None:
    lines = read(".github/workflows/release.yml").splitlines()
    header = " ".join(line for line in lines[: lines.index("name: Release")]).lower()
    assert header.startswith("#")
    assert "trusted publisher" in header
    assert "environment" in header
    # The two repository features the published metadata already points at.
    assert "pages" in header
    assert "discussions" in header


def test_paper_header_is_a_fenced_yaml_block() -> None:
    paper = read("docs/paper/paper.md")
    blocks = code_blocks(paper, "yaml")
    assert len(blocks) == 1
    header = cast("dict[str, Any]", yaml.safe_load(blocks[0]))
    assert header["title"].startswith("msrelapse")
    assert header["authors"]
    assert header["affiliations"]
    assert header["bibliography"] == "paper.bib"
    assert header["tags"]


def test_paper_notes_that_submission_waits_for_v1() -> None:
    comments = re.findall(r"<!--(.*?)-->", read("docs/paper/paper.md"), re.DOTALL)
    joined = " ".join(comments).lower()
    assert "v1.0.0" in joined
    assert "submission" in joined or "submitted" in joined


@pytest.mark.parametrize(
    "heading",
    [
        "# Summary",
        "# Statement of need",
        "# The model",
        "# Functionality",
        "# Example usage",
        "# Reproducibility and data",
        "# Acknowledgements",
        "# References",
    ],
)
def test_paper_has_the_joss_sections(heading: str) -> None:
    assert f"\n{heading}\n" in read("docs/paper/paper.md")


def test_paper_states_the_three_equations() -> None:
    model = section(read("docs/paper/paper.md"), "# The model")
    for symbol in (r"\alpha", r"\beta", r"\epsilon", r"\log"):
        assert symbol in model
    assert model.count("$$") >= 6


def test_paper_acknowledges_the_co_authors_of_the_article() -> None:
    acknowledgements = section(read("docs/paper/paper.md"), "# Acknowledgements")
    for family, _given in PAPER_AUTHORS:
        if family != "Umeton":
            assert family in acknowledgements


def test_paper_is_about_a_thousand_words() -> None:
    prose = re.sub(r"```.*?```", "", read("docs/paper/paper.md"), flags=re.DOTALL)
    prose = re.sub(r"<!--.*?-->", "", prose, flags=re.DOTALL)
    assert 700 <= len(prose.split()) <= 1400


def test_paper_claims_no_users_and_no_archive_doi() -> None:
    prose = read("docs/paper/paper.md").lower()
    assert "zenodo" not in prose
    for claim in ("is used by", "users at", "adopted by"):
        assert claim not in prose


def bib_keys() -> set[str]:
    return set(re.findall(r"^@\w+\{([^,]+),", read("docs/paper/paper.bib"), re.MULTILINE))


def test_bibliography_defines_every_required_key() -> None:
    assert bib_keys() == set(BIB_KEYS)


def test_bibliography_gives_every_entry_a_doi() -> None:
    """A reviewer of the draft asks for a DOI on every reference that has one."""
    entries = re.findall(r"@\w+\{([^,]+),(.*?)\n\}", read("docs/paper/paper.bib"), re.DOTALL)
    assert len(entries) == len(BIB_KEYS)
    for key, body in entries:
        assert re.search(r"^\s*doi\s*=", body, re.MULTILINE), key


def test_paper_cites_every_key_and_only_those_keys() -> None:
    prose = re.sub(r"```.*?```", "", read("docs/paper/paper.md"), flags=re.DOTALL)
    cited = set(re.findall(r"@([A-Za-z][A-Za-z0-9]*)", prose))
    assert cited == bib_keys()


def test_bibliography_records_the_article_with_its_doi() -> None:
    entry = re.search(r"@article\{bordi2013,.*?\n\}", read("docs/paper/paper.bib"), re.DOTALL)
    assert entry is not None
    assert PAPER.paper_doi.value in entry.group(0)
    for family, _given in PAPER_AUTHORS:
        assert family in entry.group(0)


def test_paper_example_usage_runs(capsys: pytest.CaptureFixture[str]) -> None:
    blocks = code_blocks(section(read("docs/paper/paper.md"), "# Example usage"), "python")
    assert len(blocks) == 1
    exec(compile(blocks[0], "paper.md example", "exec"), {"__name__": "__main__"})
    assert capsys.readouterr().out


@pytest.mark.parametrize("page", RUNNABLE_DOCS_PAGES)
def test_documentation_page_examples_run(page: str, capsys: pytest.CaptureFixture[str]) -> None:
    """Every Python block of these pages runs in a fresh namespace and prints something."""
    blocks = code_blocks(read(page), "python")
    assert blocks, page
    for index, block in enumerate(blocks):
        exec(compile(block, f"{page} block {index}", "exec"), {"__name__": "__main__"})
        assert capsys.readouterr().out, f"{page} block {index} printed nothing"


def test_every_documentation_page_with_an_example_is_run_or_named() -> None:
    """A page that grows a Python block cannot slip past the test above unnoticed."""
    docs = ROOT / "docs"
    pages = {
        path.relative_to(ROOT).as_posix()
        for path, text in text_files().items()
        if path.suffix == ".md" and path.is_relative_to(docs) and code_blocks(text, "python")
    }
    assert pages == set(RUNNABLE_DOCS_PAGES) | set(UNRUN_DOCS_PAGES)
