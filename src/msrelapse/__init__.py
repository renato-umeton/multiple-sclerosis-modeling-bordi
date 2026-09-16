"""Reference implementation of the Bordi et al. 2013 double well model of multiple sclerosis.

The package reproduces the stochastic model of relapsing-remitting multiple
sclerosis of the article cited below. In the 70 untreated relapsing-remitting
patients the article followed, relapse and remission durations are both
exponentially distributed, with a mean of about 4.3 weeks in relapse and about
100 weeks in health, and neither carries a detectable periodicity, so relapse
onset is memoryless. A noise driven asymmetric double well reproduces that
record: the patient sits in a deep well of health or in a shallower well of no
health, the noise carries it over the barrier between the two, and the ratio of
the logarithms of the two mean durations estimates the ratio of the two barrier
heights, about 3.1.

Every number the article reports is collected in ``PAPER`` together with the
sentence it comes from, and the code reads them from there rather than writing
them down again. :func:`cite` prints the citation of the article and of this
package, and the duration fits, the cohorts and the annualised rate results
carry a one line citation of their own. What is re-exported here are the
functions and the result classes; the type aliases they are annotated with,
among them ``Side`` and ``Passage`` in :mod:`msrelapse.model`, ``Seed`` in
:mod:`msrelapse.simulate` and ``Schema`` in :mod:`msrelapse.io`, stay on their
own modules. Drawing lives in :mod:`msrelapse.plots`, which is imported
explicitly, so that importing this package needs no matplotlib.

Examples
--------
>>> import msrelapse
>>> msrelapse.PAPER.paper_doi.value in msrelapse.short_citation()
True
>>> round(msrelapse.fold_beta(msrelapse.PAPER.alpha_reference.value), 6)
0.3849

References
----------
I. Bordi, R. Umeton, V. A. G. Ricigliano, et al., "A mechanistic, stochastic
model helps understand multiple sclerosis course and pathogenesis",
International Journal of Genomics, 2013, doi 10.1155/2013/910321.
"""

from __future__ import annotations

from typing import TextIO

from msrelapse._citation import citation, short_citation
from msrelapse._params import PAPER
from msrelapse.cohort import (
    Cohort,
    CohortSpec,
    bordi2013_spec,
    constant,
    empirical,
    from_histogram,
    generate,
    lognormal_around,
    paper_patients,
    per_patient_params,
)
from msrelapse.datasets import (
    SYNTHETIC_SEED,
    load_synthetic_bordi2013,
    provenance,
    regenerate_synthetic_bordi2013,
    reproduction_table,
)
from msrelapse.fit import (
    FitResult,
    GammaFit,
    NBFit,
    PeriodicityResult,
    TestResult,
    barrier_ratio,
    fit_durations,
    fit_gamma_rates,
    fit_nb_counts,
    test_memoryless,
    test_periodicity,
)
from msrelapse.io import (
    durations_to_weekly,
    events_to_weekly,
    read_durations,
    read_events,
    read_weekly,
    validate,
    weekly_to_durations,
    weekly_to_events,
    write_csv,
)
from msrelapse.model import (
    Barriers,
    CriticalPoints,
    DoubleWell,
    barrier_ratio_from_durations,
    beta_from_barrier_ratio,
    calibrate,
    fold_beta,
    kramers_prefactor,
    kramers_time,
    mfpt,
    paper_exit_time,
    passage_endpoints,
)
from msrelapse.renewal import (
    alternating_renewal,
    effective_onset_rate,
    gamma_rates,
    nb_from_gamma,
    rates_from_means,
    relapse_counts,
    relapse_free,
)
from msrelapse.simulate import (
    HAS_NUMBA,
    Paths,
    durations,
    exit_times,
    simulate_paths,
    simulate_weekly,
    to_states,
    to_weekly,
)
from msrelapse.stats import (
    WEEKS_PER_YEAR,
    ARRResult,
    RateRatioResult,
    arr,
    compare_arr,
    relapse_free_curve,
    sample_size_arr,
)

__version__ = "0.1.0"

__all__ = [
    "HAS_NUMBA",
    "PAPER",
    "SYNTHETIC_SEED",
    "WEEKS_PER_YEAR",
    "ARRResult",
    "Barriers",
    "Cohort",
    "CohortSpec",
    "CriticalPoints",
    "DoubleWell",
    "FitResult",
    "GammaFit",
    "NBFit",
    "Paths",
    "PeriodicityResult",
    "RateRatioResult",
    "TestResult",
    "__version__",
    "alternating_renewal",
    "arr",
    "barrier_ratio",
    "barrier_ratio_from_durations",
    "beta_from_barrier_ratio",
    "bordi2013_spec",
    "calibrate",
    "citation",
    "cite",
    "compare_arr",
    "constant",
    "durations",
    "durations_to_weekly",
    "effective_onset_rate",
    "empirical",
    "events_to_weekly",
    "exit_times",
    "fit_durations",
    "fit_gamma_rates",
    "fit_nb_counts",
    "fold_beta",
    "from_histogram",
    "gamma_rates",
    "generate",
    "kramers_prefactor",
    "kramers_time",
    "load_synthetic_bordi2013",
    "lognormal_around",
    "mfpt",
    "nb_from_gamma",
    "paper_exit_time",
    "paper_patients",
    "passage_endpoints",
    "per_patient_params",
    "provenance",
    "rates_from_means",
    "read_durations",
    "read_events",
    "read_weekly",
    "regenerate_synthetic_bordi2013",
    "relapse_counts",
    "relapse_free",
    "relapse_free_curve",
    "reproduction_table",
    "sample_size_arr",
    "short_citation",
    "simulate_paths",
    "simulate_weekly",
    "test_memoryless",
    "test_periodicity",
    "to_states",
    "to_weekly",
    "validate",
    "weekly_to_durations",
    "weekly_to_events",
    "write_csv",
]


def cite(file: TextIO | None = None) -> None:
    """Print the citation of the article and of this package.

    Parameters
    ----------
    file : typing.TextIO, optional
        Stream to print to. The default prints to standard output.

    Returns
    -------
    None
        Nothing is returned; the citation is printed. Call
        :func:`msrelapse.citation` for the same text as a string.

    Examples
    --------
    >>> import io
    >>> stream = io.StringIO()
    >>> cite(stream)
    >>> stream.getvalue().startswith("Bordi I")
    True
    """
    print(citation(), file=file)
