from __future__ import annotations

import copy
import importlib.util
import sys
from collections.abc import Callable
from pathlib import Path
from types import ModuleType

import pytest

import msrelapse

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

# The reader, the writer and the output constructor of nbformat carry no
# annotations, so they are named once here with the types they are called with,
# rather than spreading Any through every call below.
_read: Callable[[Path, int], NotebookNode] = nbformat.read
_writes: Callable[[NotebookNode, int], str] = nbformat.writes
_new_output: Callable[..., NotebookNode] = nbformat.v4.new_output

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

# The two kinds of output a reader sees on GitHub: a rendered table and a
# figure. A printed line is a stream output and is not counted here, because a
# notebook that lost every figure and every table would still hold its prints.
RICH_OUTPUT_TYPES = ("display_data", "execute_result")

# The number of rich outputs each committed notebook holds today, counted from
# the files the builder writes. They are floors rather than exact counts, so a
# notebook that gains a figure passes and one that loses a figure or a table
# fails.
MINIMUM_RICH_OUTPUTS = {
    REPRODUCTION_NOTEBOOK: 14,
    "02_sde_to_exponential.ipynb": 7,
    "03_poisson_to_negative_binomial.ipynb": 5,
    "04_virtual_cohort_for_trial_design.ipynb": 4,
}

# The committed notebooks carry their figures as inline PNG, so each one has a
# size worth watching. The largest measures about half a megabyte today, so this
# ceiling leaves room for a figure or two while a notebook that started shipping
# an image at a printing resolution fails here.
MAXIMUM_BYTES = 3 * 1024 * 1024

# The Sphinx cross reference roles, built from their names so that this file
# holds none of its own. tests/test_docs.py keeps them out of the package, where
# mkdocstrings renders one as the literal text a reader of the site sees; the
# builder is swept here, because the same docstring convention covers it and no
# other test reads it.
ROLE_NAMES = ("func", "class", "meth", "attr", "mod", "data", "const", "obj", "ref")
ROLE_MARKERS = tuple(f":{name}:" for name in ROLE_NAMES)

NBFORMAT_VERSION = 4
TIMEOUT_SECONDS = 900
KERNEL_NAME = "python3"


def read_notebook(name: str) -> NotebookNode:
    return _read(NOTEBOOK_DIR / name, NBFORMAT_VERSION)


def execute_notebook(name: str) -> NotebookNode:
    client = NotebookClient(
        read_notebook(name),
        timeout=TIMEOUT_SECONDS,
        kernel_name=KERNEL_NAME,
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


def rich_cell_outputs(cell: NotebookNode) -> list[NotebookNode]:
    """Return the rendered tables and figures one code cell holds, in output order."""
    return [
        output
        for output in cell.get("outputs", [])
        if output.get("output_type") in RICH_OUTPUT_TYPES
    ]


def rich_outputs(notebook: NotebookNode) -> list[NotebookNode]:
    """Return every rendered table and figure of a notebook, in reading order."""
    return [output for cell in code_cells(notebook) for output in rich_cell_outputs(cell)]


def rich_output_types(notebook: NotebookNode) -> list[list[str]]:
    """Return the type of every rendered table and figure, cell by cell.

    Printed lines are left out on purpose, so that a warning one machine prints
    and another does not leaves the shape of a notebook unchanged, while a cell
    that stopped rendering a table or drawing a figure changes it.
    """
    return [
        [str(output.get("output_type")) for output in rich_cell_outputs(cell)]
        for cell in code_cells(notebook)
    ]


def without_outputs(notebook: NotebookNode) -> NotebookNode:
    """Return a copy of a notebook whose code cells carry no output and no run count."""
    cleared = copy.deepcopy(notebook)
    for cell in cleared.cells:
        if cell.cell_type == "code":
            cell.outputs = []
            cell.execution_count = None
    return cleared


def stderr_cells(notebook: NotebookNode) -> list[int]:
    """Return the position of every code cell that wrote to standard error."""
    return [
        index
        for index, cell in enumerate(code_cells(notebook))
        if any(
            output.get("output_type") == "stream" and output.get("name") == "stderr"
            for output in cell.get("outputs", [])
        )
    ]


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


@pytest.fixture(scope="module")
def unrun_build(builder: ModuleType, tmp_path_factory: pytest.TempPathFactory) -> tuple[int, Path]:
    """Write the four notebooks without running them, once, and report how it went.

    Returns the status the builder exited with and the directory it wrote into,
    so that the three tests that read this build share one run of it. Nothing
    here starts a kernel, which is why those tests carry no slow marker.
    """
    directory = tmp_path_factory.mktemp("no_execute")
    status: int = builder.main(["--no-execute", "--out-dir", str(directory)])
    return status, directory


@pytest.mark.slow
@pytest.mark.parametrize("name", NOTEBOOK_NAMES)
def test_notebook_runs_to_the_end(name: str, run_notebook: Callable[[str], NotebookNode]) -> None:
    assert error_report(run_notebook(name)) == ""


@pytest.mark.slow
@pytest.mark.parametrize("name", NOTEBOOK_NAMES)
def test_committed_outputs_have_the_shape_of_a_fresh_run(
    name: str, run_notebook: Callable[[str], NotebookNode]
) -> None:
    """A stale file is one whose cells no longer render what the committed file shows.

    The values are not compared, because a float printed to four decimals and a
    figure drawn on another platform are not the same twice. Which cell rendered
    a table and which drew a figure is the same everywhere, and it is what goes
    stale when the code under a notebook changes.
    """
    assert rich_output_types(run_notebook(name)) == rich_output_types(read_notebook(name)), (
        f"the tables and figures of a fresh run of {name} are not the ones the committed "
        f"file holds; rebuild it with notebooks/build_notebooks.py"
    )


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
def test_every_committed_code_cell_was_run(name: str) -> None:
    """The committed notebooks are the executed ones, so the counts run 1, 2, 3 and on.

    A file run in several sittings, or run out of order, carries counts that
    skip or double back, and its outputs are then not the ones one run from top
    to bottom produces. That is what the committed outputs are meant to be.
    """
    cells = code_cells(read_notebook(name))
    assert cells, f"{name} holds no code cell"
    assert [cell.execution_count for cell in cells] == list(range(1, len(cells) + 1)), (
        f"the run counts of {name} are not those of one run from top to bottom, so the "
        f"file was committed unexecuted, had its outputs stripped, or was run out of order"
    )


@pytest.mark.parametrize("name", NOTEBOOK_NAMES)
def test_committed_notebook_carries_its_tables_and_figures(name: str) -> None:
    """What a reader sees on GitHub is the point of committing the outputs at all."""
    found = len(rich_outputs(read_notebook(name)))
    assert found >= MINIMUM_RICH_OUTPUTS[name], (
        f"{name} holds {found} rendered table(s) and figure(s), below the "
        f"{MINIMUM_RICH_OUTPUTS[name]} it carried when the floor was measured"
    )


@pytest.mark.parametrize("name", NOTEBOOK_NAMES)
def test_committed_notebook_holds_no_error_output(name: str) -> None:
    """A committed traceback would be rendered as a result, so there must be none."""
    assert error_report(read_notebook(name)) == ""


@pytest.mark.parametrize("name", NOTEBOOK_NAMES)
def test_committed_notebook_holds_no_warning(name: str) -> None:
    """A warning a cell raised is committed as a standard error block and rendered.

    The suite turns several warning categories into errors under pytest, and a
    notebook kernel sits outside that configuration, so a warning an upstream
    release starts raising would otherwise reach the file unnoticed and be shown
    on GitHub above the result it belongs to.
    """
    noisy = stderr_cells(read_notebook(name))
    assert noisy == [], (
        f"code cells {noisy} of {name} wrote to standard error, which GitHub renders "
        f"as a warning block; fix or silence the warning where it is raised, then rebuild"
    )


@pytest.mark.parametrize("name", NOTEBOOK_NAMES)
def test_committed_notebook_stays_under_the_size_limit(name: str) -> None:
    """The committed figures are inline PNG, so the size of the file is worth a guard."""
    size = (NOTEBOOK_DIR / name).stat().st_size
    assert size <= MAXIMUM_BYTES, (
        f"{name} is {size / 1024 / 1024:.2f} MB, over the {MAXIMUM_BYTES / 1024 / 1024:.0f} MB "
        f"this repository keeps its notebooks under; draw smaller figures, or fewer of them, "
        f"in notebooks/build_notebooks.py"
    )


@pytest.mark.parametrize("name", NOTEBOOK_NAMES)
def test_committed_notebook_names_its_kernel_and_not_the_language_version(name: str) -> None:
    """A run leaves metadata of its own behind, and the builder takes it back out.

    The kernel reports the version of the language it ran under, which differs
    from one machine to the next and would put the file in the diff of every
    rebuild. What a notebook needs of that metadata is the kernel it opens
    under, and that is what is checked here, rather than equality with the
    constant the builder holds, which would keep passing the day a version key
    was added to it.
    """
    metadata = read_notebook(name).metadata
    assert metadata["kernelspec"]["name"] == KERNEL_NAME
    assert "version" not in metadata["language_info"], (
        f"{name} carries the language version its kernel reported, which differs from "
        f"one machine to the next and puts the file in the diff of every rebuild"
    )


@pytest.mark.parametrize("name", NOTEBOOK_NAMES)
def test_committed_cells_carry_no_timing(name: str) -> None:
    """The other thing a run leaves behind is a timing on every cell it ran.

    nbclient records one unless it is told not to, and the wall clock of a run
    is in the diff of every rebuild for as little reason as the language version
    is, so the committed cells carry no metadata at all.
    """
    timed = [index for index, cell in enumerate(read_notebook(name).cells) if cell.metadata]
    assert timed == [], f"cells {timed} of {name} carry metadata a run left behind"


@pytest.mark.parametrize("name", NOTEBOOK_NAMES)
def test_builder_reproduces_the_committed_file(name: str, builder: ModuleType) -> None:
    """A rebuild must give the committed file back, so that it carries no diff.

    The builder returns the notebook before it is run, so the committed file is
    compared with its outputs and its run counts taken out of the way. What is
    left covers the identifiers and the metadata as well as the cells.
    """
    built = builder.build(name)
    committed = read_notebook(name)
    # The cells first, so that a changed cell is reported as a cell rather than
    # as a diff of the whole file, then the text.
    assert cell_sources(built) == cell_sources(committed)
    assert _writes(built, NBFORMAT_VERSION) == _writes(without_outputs(committed), NBFORMAT_VERSION)


@pytest.mark.parametrize("name", NOTEBOOK_NAMES)
def test_committed_file_is_what_nbformat_writes(name: str) -> None:
    """A hand edited notebook fails here rather than in the rebuild test above.

    That test compares two serialisations, so a file holding the right cells in
    a layout of its own would pass it. This is the check on the bytes on disk,
    which are the ones the builder wrote and nothing else.
    """
    committed = (NOTEBOOK_DIR / name).read_text(encoding="utf-8")
    assert _writes(read_notebook(name), NBFORMAT_VERSION).rstrip("\n") == committed.rstrip("\n")


@pytest.mark.parametrize("name", NOTEBOOK_NAMES)
def test_committed_cell_ids_are_the_positions_the_builder_stamps(name: str) -> None:
    """The identifiers have to be the sequential ones the builder writes.

    nbformat draws a random identifier for every new cell, which would put every
    cell of every notebook in the diff of any rebuild. The builder stamps the
    position of the cell instead, and this is the check that the committed files
    carry that scheme rather than a random draw.
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


def test_writing_without_running_reports_success(unrun_build: tuple[int, Path]) -> None:
    """``--no-execute`` is a documented mode, so the status it exits with is checked."""
    status, _ = unrun_build
    assert status == 0


@pytest.mark.parametrize("name", NOTEBOOK_NAMES)
def test_notebook_written_without_running_carries_no_output(
    name: str, unrun_build: tuple[int, Path]
) -> None:
    """That mode is for reading a diff of the cells, so it writes nothing a run adds."""
    _, directory = unrun_build
    for index, cell in enumerate(code_cells(_read(directory / name, NBFORMAT_VERSION))):
        assert cell.outputs == [], f"code cell {index} of {name} was written with an output"
        assert cell.execution_count is None, (
            f"code cell {index} of {name} was written with a run count"
        )


@pytest.mark.parametrize("name", NOTEBOOK_NAMES)
def test_notebook_written_without_running_is_the_build(
    name: str, builder: ModuleType, unrun_build: tuple[int, Path]
) -> None:
    """What that mode writes is the notebook the builder returns, and nothing else."""
    _, directory = unrun_build
    written = (directory / name).read_text(encoding="utf-8")
    assert written.rstrip("\n") == _writes(builder.build(name), NBFORMAT_VERSION).rstrip("\n")


def test_writing_keeps_the_outputs_it_is_given(builder: ModuleType, tmp_path: Path) -> None:
    """A written notebook carries the outputs of the notebook handed to the writer.

    The writer used to clear them on the way out, and the committed files then
    rendered nothing on GitHub. No kernel is needed to check it: one cell
    holding one printed line stands in for an executed notebook.
    """
    notebook = builder.build(REPRODUCTION_NOTEBOOK)
    printed = "a printed line\n"
    cell = code_cells(notebook)[0]
    cell.outputs = [_new_output("stream", name="stdout", text=printed)]
    cell.execution_count = 1
    written = _read(builder.write(REPRODUCTION_NOTEBOOK, notebook, tmp_path), NBFORMAT_VERSION)
    assert cell_text(code_cells(written)[0]) == printed


def test_a_rebuild_that_fails_part_way_writes_nothing(
    builder: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A run that stops on a raising cell leaves the notebooks as they were.

    The committed files are large artefacts now, so a rebuild that wrote the
    first two and then stopped would leave a working tree half rebuilt and half
    stale. Every notebook is run before any of them is written. The run itself
    is stood in for here, so that no kernel starts.
    """
    runs: list[NotebookNode] = []

    def run_then_fail(notebook: NotebookNode, work_dir: Path) -> NotebookNode:
        runs.append(notebook)
        if len(runs) == 3:
            raise RuntimeError("a cell of the third notebook raised")
        return notebook

    monkeypatch.setattr(builder, "_execute", run_then_fail)
    with pytest.raises(RuntimeError, match="third notebook"):
        builder.main(["--out-dir", str(tmp_path)])
    written = sorted(path.name for path in tmp_path.glob("*.ipynb"))
    assert written == [], f"a rebuild that stopped on the third notebook wrote {written}"


def test_running_without_nbclient_names_the_dependency_group(
    builder: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Running the notebooks needs nbclient, which is optional, so a failure names it."""
    # None in sys.modules is how the import machinery is told that a module is
    # not available, so the import inside the builder fails the way it would on
    # a machine without the notebooks group.
    monkeypatch.setitem(sys.modules, "nbclient", None)
    with pytest.raises(RuntimeError, match="--group notebooks"):
        builder.main(["--out-dir", str(tmp_path)])


def test_running_without_the_kernel_names_the_dependency_group(
    builder: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A machine without the kernel gets the same remedy rather than a bare NoSuchKernel."""
    monkeypatch.setattr(builder, "KERNEL_NAME", "no-kernel-goes-by-this-name")
    with pytest.raises(RuntimeError, match="--group notebooks"):
        builder.main(["--out-dir", str(tmp_path)])


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
    assert set(MINIMUM_RICH_OUTPUTS) == set(NOTEBOOK_NAMES), (
        "a notebook has no floor on its rendered tables and figures; measure one from "
        "the file the builder writes and add it to MINIMUM_RICH_OUTPUTS"
    )
