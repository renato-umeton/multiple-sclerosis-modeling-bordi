from __future__ import annotations

import importlib.util
from collections.abc import Callable
from pathlib import Path
from types import ModuleType

import pytest

import msrelapse
from _helpers import ROOT, load_yaml

try:
    import nbformat
    from nbclient import NotebookClient
    from nbformat import NotebookNode
except ImportError:  # pragma: no cover - the notebooks dependency group is optional
    pytest.skip(
        "nbformat and nbclient are needed to run the notebooks; install the notebooks group",
        allow_module_level=True,
    )

# Every test here belongs to the notebook job of CI, which selects on this
# marker and is the only job that installs the notebooks group. Only the two
# tests that start a kernel are marked slow; the rest are file checks of a few
# milliseconds and carry the marker alone, so a local run of -m notebook gets
# them without waiting for a kernel. Nowhere else are they run: every other job,
# the build step that unpacks the sdist and runs the packaged suite with
# -m "not slow" included, syncs without the notebooks group, so nbformat is
# missing there and the guard above skips this module whole.
pytestmark = [pytest.mark.notebook]

# nbformat.read and nbformat.writes carry no annotations, so they are named once
# here with the types they are called with, rather than spreading Any through
# every call below.
_read: Callable[[Path, int], NotebookNode] = nbformat.read
_writes: Callable[[NotebookNode, int], str] = nbformat.writes

NOTEBOOK_DIR = Path(__file__).resolve().parents[1] / "notebooks"
BUILDER_PATH = NOTEBOOK_DIR / "build_notebooks.py"

REPRODUCTION_NOTEBOOK = "01_reproduce_bordi2013.ipynb"
NOTEBOOK_NAMES = (
    REPRODUCTION_NOTEBOOK,
    "02_sde_to_exponential.ipynb",
    "03_poisson_to_negative_binomial.ipynb",
    "04_virtual_cohort_for_trial_design.ipynb",
)

# The name of the column the closing table of notebook 01 is judged on, which is
# also how the two cells that report it are found among the others.
TOLERANCE_COLUMN = "within_tolerance"

# One of these has to appear in the opening cells of every notebook, which is the
# rule that the provenance of the data is stated before anything is computed.
PROVENANCE_PHRASES = ("synthetic", "no clinical data")

# The hook that clears the outputs and rewrites the cell identifiers, and the
# file that declares it. That file is not part of the sdist, so the check below
# reads it only where it is present.
PRE_COMMIT_CONFIG = ".pre-commit-config.yaml"
STRIP_HOOK = "nbstripout"

# The Sphinx cross reference roles, built from their names so that this file
# holds none of its own. tests/test_docs.py keeps them out of the package, where
# mkdocstrings renders one as the literal text a reader of the site sees; the
# builder is swept here, because the same docstring convention covers it and no
# other test reads it.
ROLE_NAMES = ("func", "class", "meth", "attr", "mod", "data", "const", "obj", "ref")
ROLE_MARKERS = tuple(f":{name}:" for name in ROLE_NAMES)

NBFORMAT_VERSION = 4
TIMEOUT_SECONDS = 900


def read_notebook(name: str) -> NotebookNode:
    return _read(NOTEBOOK_DIR / name, NBFORMAT_VERSION)


def execute_notebook(name: str) -> NotebookNode:
    client = NotebookClient(
        read_notebook(name),
        timeout=TIMEOUT_SECONDS,
        kernel_name="python3",
        resources={"metadata": {"path": str(NOTEBOOK_DIR)}},
    )
    return client.execute()


def code_cells(notebook: NotebookNode) -> list[NotebookNode]:
    return [cell for cell in notebook.cells if cell.cell_type == "code"]


def cell_text(cell: NotebookNode) -> str:
    """Return the text a code cell printed or displayed, in output order."""
    pieces: list[str] = []
    for output in cell.get("outputs", []):
        if output.get("output_type") == "stream":
            pieces.append(str(output.get("text", "")))
        else:
            pieces.append(str(output.get("data", {}).get("text/plain", "")))
    return "\n".join(pieces)


def cell_sources(notebook: NotebookNode) -> list[tuple[str, str]]:
    return [(str(cell.cell_type), str(cell.source)) for cell in notebook.cells]


def error_report(notebook: NotebookNode) -> str:
    """Return the tracebacks of every cell that raised, or an empty string."""
    reports: list[str] = []
    for index, cell in enumerate(code_cells(notebook)):
        for output in cell.get("outputs", []):
            if output.get("output_type") == "error":
                traceback = "\n".join(str(line) for line in output.get("traceback", []))
                reports.append(f"code cell {index}:\n{traceback}")
    return "\n".join(reports)


@pytest.fixture(scope="module")
def run_notebook() -> Callable[[str], NotebookNode]:
    """Return a function that executes a notebook once and remembers the result."""
    executed: dict[str, NotebookNode] = {}

    def run(name: str) -> NotebookNode:
        if name not in executed:
            executed[name] = execute_notebook(name)
        return executed[name]

    return run


@pytest.fixture(scope="module")
def builder() -> ModuleType:
    spec = importlib.util.spec_from_file_location("build_notebooks", BUILDER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"the notebook builder at {BUILDER_PATH} cannot be imported")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.slow
@pytest.mark.parametrize("name", NOTEBOOK_NAMES)
def test_notebook_runs_to_the_end(name: str, run_notebook: Callable[[str], NotebookNode]) -> None:
    assert error_report(run_notebook(name)) == ""


@pytest.mark.slow
def test_reproduction_table_rows_are_within_tolerance(
    run_notebook: Callable[[str], NotebookNode],
) -> None:
    executed = run_notebook(REPRODUCTION_NOTEBOOK)
    # The rendered table first, then the line the cell after it prints. Both are
    # found by what they display rather than by their source, so that the table
    # itself is read and not only the summary of it.
    reporting = [cell for cell in code_cells(executed) if TOLERANCE_COLUMN in cell_text(cell)]
    assert len(reporting) == 2, (
        f"expected the rendered table and the summary line to display "
        f"{TOLERANCE_COLUMN}, got {len(reporting)} cell(s)"
    )
    rendered_table, summary = (cell_text(cell) for cell in reporting)
    assert "True" in rendered_table
    assert "False" not in rendered_table
    assert "True" in summary


@pytest.mark.parametrize("name", NOTEBOOK_NAMES)
def test_committed_notebook_carries_no_output(name: str) -> None:
    cells = code_cells(read_notebook(name))
    assert cells, f"{name} holds no code cell"
    for index, cell in enumerate(cells):
        assert cell.outputs == [], f"code cell {index} of {name} was committed with output"
        assert cell.execution_count is None, f"code cell {index} of {name} carries a run count"


@pytest.mark.parametrize("name", NOTEBOOK_NAMES)
def test_builder_reproduces_the_committed_file(name: str, builder: ModuleType) -> None:
    """A rebuild must give the committed file back, so that it carries no diff."""
    built = builder.build(name)
    # The cells first, so that a changed cell is reported as a cell rather than
    # as a diff of the whole file, then the text, which covers the identifiers
    # and the metadata as well.
    assert cell_sources(built) == cell_sources(read_notebook(name))
    committed = (NOTEBOOK_DIR / name).read_text(encoding="utf-8")
    assert _writes(built, NBFORMAT_VERSION).rstrip("\n") == committed.rstrip("\n")


def test_the_strip_hook_the_identifiers_come_from_is_configured() -> None:
    """The test below rests on that hook, so a configuration without it fails here.

    nbstripout is what clears the outputs and rewrites the identifier of every
    cell to its position before a notebook is committed. Drop it and the two
    checks below keep passing on the notebooks that were stripped already, while
    the next notebook committed with output in it fails the rebuild test
    instead, a long way from the cause.
    """
    if not (ROOT / PRE_COMMIT_CONFIG).is_file():
        pytest.skip(f"{PRE_COMMIT_CONFIG} is not shipped in the sdist, so it cannot be read here")
    config = load_yaml(PRE_COMMIT_CONFIG)
    hooks = [hook["id"] for repo in config["repos"] for hook in repo.get("hooks", [])]
    assert STRIP_HOOK in hooks, f"{PRE_COMMIT_CONFIG} no longer runs {STRIP_HOOK}"


@pytest.mark.parametrize("name", NOTEBOOK_NAMES)
def test_committed_cell_ids_are_the_ones_the_strip_hook_leaves(name: str) -> None:
    """The identifiers have to be the sequential ones nbstripout writes.

    The hook checked above rewrites the identifier of every cell to its position
    before a notebook is committed, so a builder that stamped anything else
    would be undone by the first commit and the rebuild test above would then
    fail for good.
    """
    cells = read_notebook(name).cells
    assert [str(cell.id) for cell in cells] == [str(index) for index in range(len(cells))]


@pytest.mark.parametrize("name", NOTEBOOK_NAMES)
def test_notebook_opens_with_a_title_and_says_where_its_data_come_from(name: str) -> None:
    cells = read_notebook(name).cells
    assert cells[0].cell_type == "markdown"
    assert str(cells[0].source).startswith("# "), f"{name} does not open with a title"
    opening = " ".join(
        str(cell.source).lower() for cell in cells[:3] if cell.cell_type == "markdown"
    )
    assert any(phrase in opening for phrase in PROVENANCE_PHRASES), (
        f"none of the opening cells of {name} says where its data come from"
    )


@pytest.mark.parametrize("name", NOTEBOOK_NAMES)
def test_notebook_closes_with_the_citation(name: str) -> None:
    last = read_notebook(name).cells[-1]
    assert last.cell_type == "markdown"
    assert msrelapse.PAPER.paper_doi.value in str(last.source)


def test_the_builder_holds_no_sphinx_role() -> None:
    """The builder keeps the docstring convention the package keeps.

    A cross reference is written ``[`name`][msrelapse.module.name]`` and
    anything else as a code span, which is what mkdocstrings reads. Nothing
    publishes this file, so a role here breaks no page; it is the one script of
    the repository outside the package, and the convention is the same for it.
    """
    text = BUILDER_PATH.read_text(encoding="utf-8")
    found = [marker for marker in ROLE_MARKERS if marker in text]
    assert found == [], f"{BUILDER_PATH.name} holds a Sphinx role: {found}"


def test_the_suite_covers_every_notebook(builder: ModuleType) -> None:
    """Nothing reaches the notebooks directory without reaching these tests."""
    assert builder.NOTEBOOK_NAMES == NOTEBOOK_NAMES
    assert sorted(path.name for path in NOTEBOOK_DIR.glob("*.ipynb")) == sorted(NOTEBOOK_NAMES)
