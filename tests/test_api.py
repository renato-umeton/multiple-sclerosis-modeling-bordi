"""The public surface of the package: its exports, its version and its citation."""

from __future__ import annotations

import io
import math
import subprocess
import sys
from importlib import metadata

import numpy as np
import pandas as pd
import pytest

import msrelapse

DOI = msrelapse.PAPER.paper_doi.value

_Result = msrelapse.Cohort | msrelapse.FitResult | msrelapse.ARRResult


def _is_positive_number(value: float) -> bool:
    """Return whether a reported quantity is a finite number above zero."""
    return math.isfinite(value) and value > 0.0


@pytest.fixture(scope="module")
def cohort() -> msrelapse.Cohort:
    return msrelapse.generate(
        msrelapse.CohortSpec(n=6, tau_health=40.0, tau_relapse=4.0, followup_weeks=300.0),
        rng=0,
    )


@pytest.fixture(scope="module")
def durations(cohort: msrelapse.Cohort) -> pd.DataFrame:
    frame = cohort.durations
    assert frame is not None
    return frame


@pytest.fixture(scope="module")
def results(cohort: msrelapse.Cohort, durations: pd.DataFrame) -> dict[str, _Result]:
    return {
        "Cohort": cohort,
        "FitResult": msrelapse.fit_durations(durations, msrelapse.PAPER.state_no_health.value),
        "ARRResult": msrelapse.arr(cohort.events),
    }


def test_importing_the_package_prints_nothing_and_leaves_matplotlib_out() -> None:
    script = (
        "import sys\n"
        "import msrelapse\n"
        "assert 'matplotlib' not in sys.modules, 'importing msrelapse imported matplotlib'\n"
    )

    completed = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, check=False
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout == "", "importing msrelapse printed to standard output"
    assert completed.stderr == "", "importing msrelapse wrote to standard error"


def test_every_exported_name_resolves() -> None:
    assert [name for name in msrelapse.__all__ if not hasattr(msrelapse, name)] == []


def test_the_export_list_names_nothing_twice() -> None:
    names = list(msrelapse.__all__)

    assert sorted(names) == sorted(set(names))


def test_the_package_docstring_carries_the_reference() -> None:
    assert msrelapse.__doc__ is not None
    assert DOI in msrelapse.__doc__


def test_the_package_docstring_quotes_the_numbers_the_paper_reports() -> None:
    assert msrelapse.__doc__ is not None
    paper = msrelapse.PAPER
    # The summary paragraph quotes the article in prose, so the phrases it uses
    # are pinned to PAPER here rather than left free to drift from it.
    quoted = [
        f"{paper.n_patients.value:d} untreated",
        f"{paper.tau_no_health_cohort_weeks.value:g} weeks in relapse",
        f"{paper.tau_health_cohort_weeks.value:g} weeks in health",
        f"about {paper.barrier_ratio_cohort.value:g}",
    ]

    assert [phrase for phrase in quoted if phrase not in msrelapse.__doc__] == []


def test_cite_prints_the_article_and_names_the_package(capsys: pytest.CaptureFixture[str]) -> None:
    msrelapse.cite()

    printed = capsys.readouterr().out
    assert printed.count(DOI) == 2
    assert "msrelapse" in printed


def test_cite_writes_to_the_stream_it_is_given() -> None:
    stream = io.StringIO()

    msrelapse.cite(stream)

    assert stream.getvalue() == msrelapse.citation() + "\n"


def test_version_matches_the_installed_distribution() -> None:
    assert msrelapse.__version__ == metadata.version("msrelapse")


@pytest.mark.parametrize("name", ["Cohort", "FitResult", "ARRResult"])
def test_a_result_object_carries_the_citation(results: dict[str, _Result], name: str) -> None:
    assert DOI in results[name].citation


@pytest.mark.parametrize(
    "name",
    [
        "Cohort",
        "FitResult",
        pytest.param(
            "ARRResult",
            marks=pytest.mark.xfail(
                strict=True,
                reason=(
                    "msrelapse.stats.ARRResult keeps the citation on a property and takes the "
                    "repr the dataclass writes, which names the fields alone. Give it a "
                    "__repr__ in the manner of msrelapse.fit.FitResult and drop this marker"
                ),
            ),
        ),
    ],
)
def test_a_result_repr_names_the_article_once(results: dict[str, _Result], name: str) -> None:
    assert repr(results[name]).count(DOI) == 1


def test_a_short_analysis_runs_through_the_public_api(
    cohort: msrelapse.Cohort, durations: pd.DataFrame
) -> None:
    relapse_fit = msrelapse.fit_durations(durations, msrelapse.PAPER.state_no_health.value)
    health_fit = msrelapse.fit_durations(durations, msrelapse.PAPER.state_health.value)
    assert isinstance(relapse_fit, msrelapse.FitResult)
    assert isinstance(health_fit, msrelapse.FitResult)
    assert _is_positive_number(relapse_fit.mean)
    assert _is_positive_number(health_fit.mean)

    ratio = msrelapse.barrier_ratio(durations)
    assert isinstance(ratio, float)
    assert math.isfinite(ratio)

    rate = msrelapse.arr(cohort.events)
    assert isinstance(rate, msrelapse.ARRResult)
    assert rate.ci_low <= rate.arr <= rate.ci_high

    comparison = msrelapse.compare_arr(
        cohort.events, None, msrelapse.generate(cohort.spec, rng=1).events, None
    )
    assert isinstance(comparison, msrelapse.RateRatioResult)
    assert _is_positive_number(comparison.rate_ratio)
    assert math.isfinite(comparison.p_value)

    beta, sigma = msrelapse.calibrate(
        msrelapse.PAPER.tau_health_cohort_weeks.value,
        msrelapse.PAPER.tau_no_health_cohort_weeks.value,
    )
    assert 0.0 < beta < msrelapse.fold_beta(msrelapse.PAPER.alpha_reference.value)
    assert _is_positive_number(sigma)

    times = msrelapse.exit_times(
        msrelapse.DoubleWell(beta=beta), sigma, "relapse", 4, dt=0.05, rng=0
    )
    assert isinstance(times, np.ndarray)
    assert times.shape == (4,)
    assert bool(np.all(np.isfinite(times) & (times > 0.0)))
