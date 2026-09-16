"""Run the docstring examples of the package and of every module inside it."""

from __future__ import annotations

import contextlib
import doctest
import importlib
import io
import pkgutil
import sys
from collections.abc import Iterator

import pytest

import msrelapse

_OPTIONFLAGS = doctest.NORMALIZE_WHITESPACE | doctest.ELLIPSIS

# Modules whose examples need an optional dependency. A module named here is
# skipped rather than failed when that dependency is not installed.
_OPTIONAL_DEPENDENCIES = {"msrelapse.plots": "matplotlib"}

# Modules that carry examples today. One of them dropping out of the sweep, or
# losing its Examples section, would otherwise pass in silence.
_MUST_CARRY_EXAMPLES = frozenset(
    {"msrelapse", "msrelapse.model", "msrelapse.fit", "msrelapse.stats", "msrelapse.plots"}
)


def _stop_on_import_error(module_name: str) -> None:
    """Refuse to continue the discovery when a subpackage cannot be imported.

    Parameters
    ----------
    module_name : str
        Name of the subpackage that could not be imported.

    Raises
    ------
    ImportError
        Always. Without this, ``pkgutil.walk_packages`` swallows the error and
        the modules of that subpackage leave the sweep without a word.
    """
    raise ImportError(f"could not import {module_name} while discovering the package modules")


def _module_names() -> list[str]:
    """Return the package and every module inside it.

    Returns
    -------
    list of str
        Importable names, the package first and then its modules in
        alphabetical order. They are discovered rather than listed, so a module
        added to the package is covered without a change here, and the walk
        descends into a subpackage rather than stopping at its ``__init__``.
    """
    inside = sorted(
        name
        for _finder, name, _ispkg in pkgutil.walk_packages(
            msrelapse.__path__, prefix="msrelapse.", onerror=_stop_on_import_error
        )
    )
    return ["msrelapse", *inside]


def _prepare_optional_dependency(module_name: str) -> None:
    """Make the optional dependency of a module ready, or skip the module.

    Parameters
    ----------
    module_name : str
        Importable name of the module whose examples are about to run.

    Notes
    -----
    The drawing examples are run on the headless backend, chosen here rather
    than inherited from whichever test module happens to have been imported
    first, so that this file is safe to run on its own and on a machine with no
    window server.
    """
    dependency = _OPTIONAL_DEPENDENCIES.get(module_name)
    if dependency is None:
        return
    installed = pytest.importorskip(dependency)
    if dependency == "matplotlib":
        installed.use("Agg")


@pytest.fixture(autouse=True)
def _close_figures() -> Iterator[None]:
    yield
    pyplot = sys.modules.get("matplotlib.pyplot")
    if pyplot is not None:
        pyplot.close("all")


def test_the_sweep_covers_the_modules_that_must_carry_examples() -> None:
    missing = _MUST_CARRY_EXAMPLES - set(_module_names())

    assert missing == frozenset()


def test_the_plot_examples_run_without_a_window() -> None:
    matplotlib = pytest.importorskip("matplotlib")

    _prepare_optional_dependency("msrelapse.plots")

    assert matplotlib.get_backend().lower() == "agg"


@pytest.mark.parametrize("module_name", _module_names())
def test_docstring_examples_run(module_name: str) -> None:
    _prepare_optional_dependency(module_name)
    module = importlib.import_module(module_name)
    report = io.StringIO()

    with contextlib.redirect_stdout(report):
        results = doctest.testmod(
            module, optionflags=_OPTIONFLAGS, raise_on_error=False, verbose=False
        )

    assert results.failed == 0, f"{module_name} has failing examples:\n{report.getvalue()}"
    if module_name in _MUST_CARRY_EXAMPLES:
        assert results.attempted > 0, f"{module_name} carries no docstring example any more"
