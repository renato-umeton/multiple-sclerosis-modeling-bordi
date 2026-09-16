"""Tests of the repository metadata: README, citation files, templates, release."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, cast

import pytest
import yaml  # type: ignore[import-untyped]

import msrelapse
from msrelapse._params import PAPER

ROOT = Path(__file__).resolve().parents[1]

REPOSITORY = "https://github.com/renato-umeton/multiple-sclerosiss-modeling-bordi"
DOCUMENTATION = "https://renato-umeton.github.io/multiple-sclerosiss-modeling-bordi/"

# Written as a code point so that the character itself never enters a source
# file of this repository, which is the rule the check below enforces.
EM_DASH = chr(0x2014)

OWNED_FILES = (
    "README.md",
    "CITATION.cff",
    "codemeta.json",
    ".github/ISSUE_TEMPLATE/bug_report.yml",
    ".github/ISSUE_TEMPLATE/feature_request.yml",
    ".github/ISSUE_TEMPLATE/config.yml",
    ".github/workflows/release.yml",
    "docs/pull_request_template.md",
    "docs/paper/paper.md",
    "docs/paper/paper.bib",
)

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


def read(relative: str) -> str:
    return (ROOT / relative).read_text(encoding="utf-8")


def load_yaml(relative: str) -> dict[str, Any]:
    return cast("dict[str, Any]", yaml.safe_load(read(relative)))


def load_json(relative: str) -> dict[str, Any]:
    return cast("dict[str, Any]", json.loads(read(relative)))


def flat(text: str) -> str:
    """Collapse every run of whitespace, so that a wrapped sentence still matches."""
    return " ".join(text.split())


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


@pytest.mark.parametrize("relative", OWNED_FILES)
def test_no_em_dash(relative: str) -> None:
    assert EM_DASH not in read(relative)


@pytest.mark.parametrize("relative", OWNED_FILES)
def test_no_line_of_three_hyphens(relative: str) -> None:
    lines = [line for line in read(relative).splitlines() if line.strip() == "---"]
    assert lines == []


def test_readme_title_is_the_package_name() -> None:
    assert read("README.md").splitlines()[0] == "# msrelapse"


def test_readme_shows_the_ci_license_and_python_badges() -> None:
    readme = read("README.md")
    assert f"{REPOSITORY}/actions/workflows/ci.yml/badge.svg" in readme
    assert "img.shields.io/badge/license-MIT" in readme
    assert "img.shields.io/badge/python-3.10" in readme
    assert "3.14" in readme


def test_readme_keeps_the_pypi_and_zenodo_badges_commented_out() -> None:
    readme = read("README.md")
    comments = re.findall(r"<!--(.*?)-->", readme, re.DOTALL)
    commented = "\n".join(comments)
    assert "img.shields.io/pypi/v/msrelapse" in commented
    assert "zenodo.org/badge" in commented
    # Neither badge may be live before the first release mints them.
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
    assert summary in flat(read("README.md"))


@pytest.mark.parametrize(
    ("audience", "entry_point"),
    [
        ("modellers", "DoubleWell"),
        ("trial statisticians", "compare_arr"),
        ("in silico trial", "generate"),
    ],
)
def test_readme_names_each_audience_with_an_entry_point(audience: str, entry_point: str) -> None:
    readme = flat(read("README.md")).lower()
    assert audience.lower() in readme
    assert entry_point.lower() in readme


def test_readme_documents_the_uv_setup() -> None:
    readme = flat(read("README.md"))
    assert "uv sync" in readme
    for extra in ("plot", "fast"):
        assert f"--extra {extra}" in readme
    for group in ("docs", "notebooks"):
        assert f"--group {group}" in readme


def test_readme_documents_every_cli_subcommand() -> None:
    readme = read("README.md")
    for subcommand in (
        "reproduce",
        "simulate",
        "fit",
        "test-memoryless",
        "test-periodicity",
        "cite",
        "params",
    ):
        assert f"msrelapse {subcommand}" in readme


def test_readme_says_the_shipped_records_are_synthetic() -> None:
    readme = flat(read("README.md")).lower()
    assert "synthetic" in readme
    assert "never released" in readme or "was never released" in readme


def test_readme_links_the_documentation_site() -> None:
    readme = flat(read("README.md"))
    assert DOCUMENTATION in readme
    assert "Pages" in readme


def test_readme_explains_how_to_cite() -> None:
    readme = read("README.md")
    assert "msrelapse.cite()" in readme
    assert PAPER.paper_doi.value in readme
    assert "CITATION.cff" in readme


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
    match = re.search(rf"`msrelapse\.{name}\(\)` (prints|returns)", flat(read("README.md")))
    assert match is not None
    assert match.group(1) == ("prints" if prints else "returns")


def test_readme_links_nothing_by_a_relative_path() -> None:
    """README.md ships as the package long description, where relative links break."""
    targets = re.findall(r"\]\(([^)\s]+)", read("README.md"))
    assert targets
    for target in targets:
        assert target.startswith(("https://", "#")), target


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


def test_citation_cff_defers_the_orcid_and_the_archive_doi_to_comments() -> None:
    text = read("CITATION.cff")
    cff = load_yaml("CITATION.cff")
    assert "doi" not in cff
    assert all("orcid" not in author for author in cff["authors"])
    comments = [line.strip() for line in text.splitlines() if line.strip().startswith("#")]
    joined = " ".join(comments).lower()
    assert "orcid" in joined
    assert "zenodo" in joined


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


def test_codemeta_defers_the_archive_doi_to_a_comment() -> None:
    meta = load_json("codemeta.json")
    assert "identifier" not in meta
    assert "zenodo" in meta["comment"].lower()


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
    template = flat(read("docs/pull_request_template.md")).lower()
    for gate in ("tests", "ruff", "mypy", "changelog", "_params.py", "notebook"):
        assert gate.lower() in template
    assert "- [ ]" in read("docs/pull_request_template.md")


def release_workflow() -> dict[str, Any]:
    """Return release.yml, reading ``on`` past the YAML rule that turns it into True."""
    raw = cast("dict[Any, Any]", yaml.safe_load(read(".github/workflows/release.yml")))
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


def test_release_workflow_opens_with_the_one_time_setup() -> None:
    lines = read(".github/workflows/release.yml").splitlines()
    header = " ".join(line for line in lines[: lines.index("name: Release")]).lower()
    assert header.startswith("#")
    assert "trusted publisher" in header
    assert "environment" in header
    assert "zenodo" in header


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


def test_paper_notes_that_submission_waits_for_the_first_release() -> None:
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
