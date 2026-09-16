"""Fits and tests for the durations, the counts and the rhythm of a record.

The paper reports two mean durations, 100 weeks in health and 4.3 weeks in no
health, and two qualitative claims about the record: that the durations carry no
typical scale, and that the relapses carry no typical period. It reports no fit,
no test and no interval of any kind, which section 7 of ``docs/paper_facts.md``
records in full. This module supplies the statistics the paper leaves out, so
that its claims can be checked rather than repeated:

durations
    :func:`fit_durations` fits an exponential or a geometric duration with or
    without the right censoring of a final remission,
    :func:`test_memoryless` asks in four different ways whether the durations
    of one state really are memoryless, and :func:`discrete_hazard` is the
    weekly hazard that test and the survival figure of :mod:`msrelapse.plots`
    both read.
counts
    :func:`fit_nb_counts` fits the negative binomial counts of a cohort and
    tests them against the Poisson counts of a single shared rate, and
    :func:`fit_gamma_rates` fits the gamma distribution of the per patient
    onset rates that would produce them.
rhythm
    :func:`test_periodicity` looks for a period in the relapse onsets of each
    weekly record and combines the per patient evidence.
barrier
    :func:`barrier_ratio` applies equation (7) to the observed durations.

Every number of the paper is imported from :mod:`msrelapse._params`, never
written here. Each of the five result types of this module, :class:`FitResult`,
:class:`TestResult`, :class:`PeriodicityResult`, :class:`NBFit` and
:class:`GammaFit`, carries the citation of the article it belongs to on a
``citation`` property, from :mod:`msrelapse._citation`, and repeats it once in
its repr.

References
----------
I. Bordi, R. Umeton, V. A. G. Ricigliano, et al., "A mechanistic, stochastic
model helps understand multiple sclerosis course and pathogenesis",
International Journal of Genomics, 2013, doi 10.1155/2013/910321.
"""

# test_memoryless and test_periodicity are statistical tests of a record, not
# pytest tests, so the pytest style rule that forbids a default argument on a
# function whose name starts with "test_" does not apply anywhere in this file.
# ruff: noqa: PT028

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from typing import Final, Literal, overload

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy import optimize, special, stats

from msrelapse import _citation
from msrelapse._params import PAPER
from msrelapse.io import validate
from msrelapse.model import barrier_ratio_from_durations

# The names the package re-exports through ``msrelapse``. discrete_hazard is
# public and documented and belongs in this list too. It is held out only
# because tests/test_api.py asks every function listed here to be re-exported
# from msrelapse or named as msrelapse.fit.discrete_hazard in the package
# docstring, and both of those live in src/msrelapse/__init__.py. Adding it
# there and adding the name below are the two lines that close this.
__all__ = [
    "MIN_AT_RISK",
    "Family",
    "FitResult",
    "GammaFit",
    "MemorylessMethod",
    "NBFit",
    "PeriodicityMethod",
    "PeriodicityResult",
    "Seed",
    "TestResult",
    "barrier_ratio",
    "fit_durations",
    "fit_gamma_rates",
    "fit_nb_counts",
    "test_memoryless",
    "test_periodicity",
]

Family = Literal["exponential", "geometric"]
"""Which duration law :func:`fit_durations` fits."""

MemorylessMethod = Literal["hazard", "cv", "ks", "ad"]
"""Which reading of memorylessness :func:`test_memoryless` applies."""

PeriodicityMethod = Literal["fisher_g", "lombscargle"]
"""Which periodogram :func:`test_periodicity` builds."""

Seed = np.random.Generator | int | None
"""What every random operation of this module accepts."""

MIN_AT_RISK: Final = 5
"""Fewest records still at risk for the hazard of a week to be worth reading.

Below this the ratio is one or two events over a handful of patients and says
nothing about the shape of the durations. It is the threshold
:func:`discrete_hazard` applies by default, and so the one behind both the
``hazard`` method of :func:`test_memoryless` and the inset of
:func:`msrelapse.plots.fig_survival_vs_exponential`.
"""

_Vector = npt.NDArray[np.float64]

_FAMILIES: Final = ("exponential", "geometric")
_MEMORYLESS_METHODS: Final = ("hazard", "cv", "ks", "ad")
_PERIODICITY_METHODS: Final = ("fisher_g", "lombscargle")

_NO_HEALTH: Final = PAPER.state_no_health.value
_HEALTH: Final = PAPER.state_health.value
_STATES: Final = (_NO_HEALTH, _HEALTH)

# A straight line through fewer points than this has no residual degrees of
# freedom left, so the slope carries no standard error.
_MIN_HAZARD_TIMES: Final = 3

# Longest duration the weekly hazard will read. It allocates one bin per week up
# to the longest duration it is handed, so a larger one asks for an array no
# record could fill: this bound is already nineteen thousand years of weekly
# follow up. Refusing above it also keeps the cast to int64 that builds the bins
# inside the range of that type.
_MAX_HAZARD_WEEKS: Final = 1_000_000

# A record shorter than this holds too few periodogram ordinates for the g test
# to say anything, so it is skipped and counted.
_MIN_PERIODICITY_WEEKS: Final = 8

# Fewest relapse onsets a record needs before its rhythm can be read. Two onsets
# leave a single gap, and any two relapses are one cycle apart whatever the
# cycle, so three is the first count that carries evidence of a period.
_MIN_ONSETS: Final = 3

# Share of the power of an onset series below which the ordinates the g test
# reads hold nothing but the rounding error of the transform. Only a series whose
# onsets fall in every other week falls that low, at 1e-32 or less, and only
# because the test leaves out the one frequency such a series has power at. Any
# other series keeps at least a part in a few thousand outside that frequency.
_SPECTRUM_FLOOR: Final = 1e-12

# Frequencies per Nyquist interval of the onset frequency grid. Five is the usual
# oversampling of a record of this length.
_LOMB_OVERSAMPLING: Final = 5

_DEFAULT_CI: Final = 0.95

# Where the two coordinates of the count fit sit in its parameter vector.
_LOG_MEAN: Final = 0
_LOG_DISPERSION: Final = 1

# Relative step of the central differences behind an observed information, the
# same as :mod:`msrelapse.stats` uses on the same likelihood. What the two
# difference is the analytic score, not the log likelihood, so the truncation
# error stays at the square of the step.
_HESSIAN_STEP: Final = 1e-5

# Below this a negative binomial is a Poisson and its dispersion reads 0, as in
# :mod:`msrelapse.stats`.
_DISPERSION_FLOOR: Final = 1e-8

# Bound on the log dispersion the count likelihood is read at, again as in
# :mod:`msrelapse.stats`. The likelihood divides by the dispersion, which
# underflows to zero below about -745 on the log scale, so a line search that
# probed that far would meet a bare arithmetic error rather than a value telling
# it to turn back. Every dispersion a count model means anything at lies well
# inside the bound.
_LOG_DISPERSION_LIMIT: Final = 50.0

# Largest exponent float64 can take: beyond it math.exp raises instead of
# returning a number, and an interval built on the log scale saturates.
_LARGEST_EXPONENT: Final = float(np.log(np.finfo(np.float64).max))

# Bound on the log mean the count likelihood is read at, for the reason the log
# dispersion has one. The likelihood forms the product of the mean and the
# dispersion, so the two bounds are set together, with a nat of headroom, to keep
# that product inside float64 at either corner.
_LOG_MEAN_LIMIT: Final = _LARGEST_EXPONENT - _LOG_DISPERSION_LIMIT - 1.0

# Largest number whose square still fits in float64. A variance and a likelihood
# both square a count and then add the squares up, so the bound a cohort of n
# counts has to respect is this over the square root of n.
_LARGEST_SQUARABLE: Final = math.sqrt(float(np.finfo(np.float64).max))

# A p value of exactly zero has no logarithm, so the combination floors it here.
_SMALLEST_P_VALUE: Final = 1e-300

_PER_PATIENT_COLUMNS: Final = (
    "patient_id",
    "n_weeks",
    "statistic",
    "p_value",
    "period_weeks",
    "n_onsets",
)


@dataclass(frozen=True, repr=False)
class FitResult:
    """One fitted duration law, with its interval and its log likelihood.

    Attributes
    ----------
    family : str
        ``'exponential'`` or ``'geometric'``.
    state : int
        The clinical code of the state whose durations were fitted, +1 for no
        health and -1 for health.
    n : int
        Number of runs of that state in the frame.
    n_censored : int
        How many of those runs were treated as right censored.
    mean : float
        Fitted mean duration, in weeks.
    rate : float
        Fitted rate, per week for the exponential family and the per week
        success probability for the geometric family. Always ``1 / mean``.
    ci_low : float
        Lower end of the confidence interval of `mean`, in weeks.
    ci_high : float
        Upper end of the confidence interval of `mean`, in weeks.
    ci_level : float
        Coverage the interval was built for, for example 0.95.
    ci_method : str
        ``'fisher'`` for the information based interval, ``'bootstrap'`` for
        the percentile interval.
    loglik : float
        Log likelihood at the fitted parameter, censoring included.
    continuity_correction : float
        Weeks subtracted from every complete duration before the exponential
        fit, 0 unless the caller asked for one.
    """

    family: str
    state: int
    n: int
    n_censored: int
    mean: float
    rate: float
    ci_low: float
    ci_high: float
    ci_level: float
    ci_method: str
    loglik: float
    continuity_correction: float

    @property
    def citation(self) -> str:
        """str: One line naming the article this fit belongs to."""
        return _citation.short_citation()

    def __repr__(self) -> str:
        """Return the fit, its interval and the citation, on one line.

        Returns
        -------
        str
            The family, the state, the sample size, the mean with its interval
            and the short citation.
        """
        return (
            f"{type(self).__name__}(family={self.family!r}, state={self.state:+d}, "
            f"n={self.n}, mean={self.mean:.6g} weeks, {self.ci_level:.0%} "
            f"{self.ci_method} CI {self.ci_low:.6g} to {self.ci_high:.6g}; "
            f"{self.citation})"
        )


@dataclass(frozen=True, repr=False)
class TestResult:
    """One hypothesis test, with the pieces it was built from.

    Attributes
    ----------
    method : str
        Name of the test that was run.
    statistic : float
        The test statistic.
    p_value : float
        The p value of `statistic` under the null hypothesis of the test.
    n : int
        Number of observations the test used.
    details : dict of str to float
        The intermediate quantities of the test, such as the sample mean or the
        slope of a regression.
    """

    method: str
    statistic: float
    p_value: float
    n: int
    details: dict[str, float]

    @property
    def citation(self) -> str:
        """str: One line naming the article this test belongs to."""
        return _citation.short_citation()

    def __repr__(self) -> str:
        """Return the test, its p value and the citation, on one line.

        Returns
        -------
        str
            The method, the sample size, the statistic, the p value and the
            short citation. The intermediate quantities of ``details`` are left
            out, and are read off that attribute itself.
        """
        return (
            f"{type(self).__name__}(method={self.method!r}, n={self.n}, "
            f"statistic={self.statistic:.6g}, p_value={self.p_value:.6g}; {self.citation})"
        )

    def reject(self, alpha: float = 0.05) -> bool:
        """Return whether the null hypothesis is rejected at a given level.

        Parameters
        ----------
        alpha : float, optional
            Significance level. Must lie strictly between 0 and 1.

        Returns
        -------
        bool
            True when `p_value` is at or below `alpha`.

        Raises
        ------
        ValueError
            If `alpha` is not strictly between 0 and 1.
        """
        if not 0.0 < alpha < 1.0:
            raise ValueError(f"alpha must lie strictly between 0 and 1, got {alpha!r}")
        return self.p_value <= alpha


@dataclass(frozen=True, repr=False)
class PeriodicityResult:
    """The search for a period, patient by patient and over the cohort.

    Attributes
    ----------
    per_patient : pandas.DataFrame
        One row per patient, with the columns ``patient_id``, ``n_weeks``,
        ``statistic``, ``p_value``, ``period_weeks`` and ``n_onsets``. A patient
        that was skipped keeps its row, with its week and onset counts and a
        missing statistic, p value and period.
    pooled : TestResult
        Fisher's combined probability test over the p values of the patients
        that were not skipped. Its ``details`` report how many patients there
        were and why any of them were skipped.
    """

    per_patient: pd.DataFrame
    pooled: TestResult

    @property
    def citation(self) -> str:
        """str: One line naming the article this search belongs to."""
        return _citation.short_citation()

    def __repr__(self) -> str:
        """Return how many patients were read, the pooled p value and the citation.

        Returns
        -------
        str
            The number of patients in the table, the number the pooled test
            could read, its p value and the short citation. The per patient
            table and the pooled result are read off the attributes themselves,
            so that the line stays a line and carries the citation once.
        """
        return (
            f"{type(self).__name__}(n_patients={len(self.per_patient)}, "
            f"n_tested={self.pooled.n}, method={self.pooled.method!r}, "
            f"pooled_p_value={self.pooled.p_value:.6g}; {self.citation})"
        )


@dataclass(frozen=True, repr=False)
class NBFit:
    """A negative binomial fit of relapse counts, against its Poisson null.

    Attributes
    ----------
    mean : float
        Fitted mean count.
    dispersion : float
        Fitted dispersion a of the NB2 parameterisation, in which the variance
        is ``mean + a mean**2``. Zero when the counts are not overdispersed.
    dispersion_ci : tuple of float
        95 percent Wald interval for `dispersion`, built on the log scale.
        ``(0.0, 0.0)`` when the counts are not overdispersed, and ``(0.0, inf)``
        when they are overdispersed by so little that the interval is wider than
        float64 can exponentiate.
    loglik_nb : float
        Log likelihood of the negative binomial fit.
    loglik_poisson : float
        Log likelihood of a Poisson fit at the sample mean.
    lrt_statistic : float
        ``2 (loglik_nb - loglik_poisson)``.
    p_value : float
        Evidence against the Poisson null, from half of a chi square with one
        degree of freedom because the dispersion sits on the boundary of the
        parameter space under that null.
    n : int
        Number of counts.
    """

    mean: float
    dispersion: float
    dispersion_ci: tuple[float, float]
    loglik_nb: float
    loglik_poisson: float
    lrt_statistic: float
    p_value: float
    n: int

    @property
    def citation(self) -> str:
        """str: One line naming the article this fit belongs to."""
        return _citation.short_citation()

    def __repr__(self) -> str:
        """Return the fit, its test against the Poisson null and the citation.

        Returns
        -------
        str
            The sample size, the mean, the dispersion with its interval, the p
            value against the Poisson null and the short citation.
        """
        low, high = self.dispersion_ci
        return (
            f"{type(self).__name__}(n={self.n}, mean={self.mean:.6g}, "
            f"dispersion={self.dispersion:.6g}, {_DEFAULT_CI:.0%} CI {low:.6g} to "
            f"{high:.6g}, p_value={self.p_value:.6g}; {self.citation})"
        )


@dataclass(frozen=True, repr=False)
class GammaFit:
    """A gamma fit of the onset rates of a cohort.

    Attributes
    ----------
    k : float
        Fitted shape. A small shape means the cohort is strongly heterogeneous.
    theta : float
        Fitted scale, in onsets per week, so that the mean rate is ``k theta``.
    per_patient_rates : pandas.Series
        The plug in rate of every patient the fit used, indexed by patient_id.
    n : int
        Number of patients the fit used.
    """

    k: float
    theta: float
    per_patient_rates: pd.Series[float]
    n: int

    @property
    def citation(self) -> str:
        """str: One line naming the article this fit belongs to."""
        return _citation.short_citation()

    def __repr__(self) -> str:
        """Return the fitted gamma, the cohort it was fitted to and the citation.

        Returns
        -------
        str
            The number of patients, the shape, the scale, the mean rate the two
            imply and the short citation. The per patient rates are read off the
            attribute itself.
        """
        return (
            f"{type(self).__name__}(n={self.n}, k={self.k:.6g}, theta={self.theta:.6g}, "
            f"mean_rate={self.k * self.theta:.6g} per week; {self.citation})"
        )


# The eight parameters are the fit itself, the two options that reproduce the
# paper's own arithmetic, the two that build the interval and the generator
# behind them. Grouping them into an options object would hide the plain call.
def fit_durations(  # noqa: PLR0917
    durations: pd.DataFrame,
    state: int,
    family: Family = "exponential",
    censoring: bool = True,
    ci: float = _DEFAULT_CI,
    bootstrap: int = 0,
    continuity_correction: float = 0.0,
    rng: Seed = None,
) -> FitResult:
    """Fit the durations of one state by maximum likelihood.

    The exponential family reads every duration as a number of weeks drawn from
    an exponential law, and the geometric family reads it as a number of whole
    weeks drawn from a geometric law on 1, 2, 3, ... The two give the same mean
    when no continuity correction is asked for, and differ only in their log
    likelihood and in what they claim about the weeks between the integers.

    Parameters
    ----------
    durations : pandas.DataFrame
        A frame in the durations schema of :mod:`msrelapse.io`. It is validated
        before use.
    state : int
        Clinical code of the state to fit, +1 for no health and -1 for health.
    family : {'exponential', 'geometric'}, optional
        Which duration law to fit.
    censoring : bool, optional
        With True, a run flagged as censored contributes its time but not its
        event, which is the right likelihood for a final remission cut short by
        the end of follow up. With False every run is treated as complete,
        which is the naive estimate and what the paper did.
    ci : float, optional
        Coverage of the confidence interval. Must lie strictly between 0 and 1.
    bootstrap : int, optional
        With 0, the interval comes from the Fisher information. With a positive
        number, that many resamples of the selected runs are drawn and the
        interval is the percentile interval of their means.
    continuity_correction : float, optional
        Weeks subtracted from every complete duration before the exponential
        fit. The default of 0 reproduces the arithmetic of the paper. The usual
        choice for whole week records is 0.5, on the argument that a duration
        recorded as k weeks lasted between k - 1 and k weeks, so that k - 0.5 is
        the midpoint. It must lie in [0, 1), and it is refused outright by the
        geometric family, which already lives on whole weeks.
    rng : numpy.random.Generator or int or None, optional
        Generator for the bootstrap, or a seed for
        :func:`numpy.random.default_rng`. Unused when `bootstrap` is 0.

    Returns
    -------
    FitResult
        The fitted mean and rate, the interval and the log likelihood.

    Raises
    ------
    ValueError
        If `durations` does not obey the durations schema, if `state`, `family`
        or `ci` is out of range, if `bootstrap` is negative, if
        `continuity_correction` is outside [0, 1) or is non zero for the
        geometric family, if no complete duration of `state` is available, or
        if every bootstrap resample turns out to hold only censored runs.

    Notes
    -----
    The exponential rate is the number of complete runs over the total time, the
    censored runs included, and the mean is its reciprocal. The geometric
    success probability is the same ratio, read on whole weeks. The Fisher
    interval is symmetric on the log rate, at the half width the information of
    that family gives: ``z / sqrt(n_complete)`` for the exponential and
    ``z sqrt((1 - rate) / n_complete)`` for the geometric, the shorter of the two
    by ``sqrt(1 - rate)``. At the 4.3 week relapse of the paper the exponential
    interval is 14 percent the wider. See :func:`_fisher_interval`.

    A weekly record is discrete, so the geometric family is the exact law of what
    was recorded and is the one to read a weekly duration with. The exponential
    family is kept because it is the law of the paper and of
    :mod:`msrelapse.model`, and because the mean it reports is the arithmetic the
    paper did; ``continuity_correction=0.5`` is the way to read an exponential on
    durations that were rounded to whole weeks.

    The continuity correction is taken off the complete runs only. A run
    recorded as k weeks and censored there is known to have lasted at least k
    weeks, not to have ended inside its k-th week, so it contributes its whole
    recorded time. Correcting it as well would take half a week of exposure off
    every censored run and bias the fitted mean downwards, which is the
    direction the correction exists to remove.

    A bootstrap resample that happens to hold only censored runs has no rate and
    is left out of the percentile interval. With a cohort of any size that
    outcome does not arise, since only the final run of a patient can be
    censored.

    Examples
    --------
    >>> import pandas as pd
    >>> frame = pd.DataFrame(
    ...     {
    ...         "patient_id": pd.Series(["p1", "p1"], dtype="str"),
    ...         "run_index": pd.Series([0, 1], dtype="int64"),
    ...         "state": pd.Series([1, -1], dtype="int64"),
    ...         "duration_w": pd.Series([4, 100], dtype="int64"),
    ...         "censored": pd.Series([False, False], dtype="bool"),
    ...     }
    ... )
    >>> round(fit_durations(frame, 1).mean, 6)
    4.0
    """
    _validate_choice("family", family, _FAMILIES)
    _validate_state(state)
    if not 0.0 < ci < 1.0:
        raise ValueError(f"ci must lie strictly between 0 and 1, got {ci!r}")
    if bootstrap < 0:
        raise ValueError(f"bootstrap must be zero or a positive count, got {bootstrap!r}")
    if not 0.0 <= continuity_correction < 1.0:
        raise ValueError(
            f"continuity_correction must lie in [0, 1) week, because every recorded "
            f"duration is at least one week, got {continuity_correction!r}"
        )
    if family == "geometric" and continuity_correction != 0.0:
        raise ValueError(
            f"a continuity correction turns a whole week count into the midpoint of the "
            f"week it stands for, which the geometric family already accounts for, so "
            f"continuity_correction must be 0 for family='geometric', got "
            f"{continuity_correction!r}"
        )
    validate(durations, "durations")

    selected = durations[durations["state"] == state]
    values = selected["duration_w"].to_numpy(dtype=np.float64)
    censored = (
        selected["censored"].to_numpy(dtype=bool)
        if censoring
        else np.zeros(values.size, dtype=bool)
    )
    n_complete = int(np.count_nonzero(~censored))
    if n_complete == 0:
        raise ValueError(
            f"the frame holds no complete run of state {state:+d}: {values.size} run(s) "
            f"of that state are present and none of them completed, so there is no "
            f"event to estimate a rate from"
        )

    rate, mean, loglik = _estimate(values, censored, family, continuity_correction)
    if bootstrap > 0:
        ci_low, ci_high = _bootstrap_interval(
            values, censored, family, continuity_correction, ci=ci, n_boot=bootstrap, rng=rng
        )
        ci_method = "bootstrap"
    else:
        ci_low, ci_high = _fisher_interval(mean, rate, family, n_complete, ci)
        ci_method = "fisher"
    return FitResult(
        family=family,
        state=int(state),
        n=int(values.size),
        n_censored=int(values.size - n_complete),
        mean=mean,
        rate=rate,
        ci_low=ci_low,
        ci_high=ci_high,
        ci_level=ci,
        ci_method=ci_method,
        loglik=loglik,
        continuity_correction=continuity_correction,
    )


def test_memoryless(
    durations: pd.DataFrame,
    state: int,
    method: MemorylessMethod = "hazard",
    n_boot: int = 500,
    rng: Seed = None,
) -> TestResult:
    """Test whether the durations of one state carry no memory.

    A memoryless duration is one whose chance of ending in the coming week does
    not depend on how long it has already lasted. The paper asserts this of both
    states on the shape of two histograms alone; the four methods here give it a
    number. Only the complete runs of `state` are used, because a censored run
    has no duration to test.

    Parameters
    ----------
    durations : pandas.DataFrame
        A frame in the durations schema of :mod:`msrelapse.io`. It is validated
        before use.
    state : int
        Clinical code of the state to test, +1 for no health and -1 for health.
    method : {'hazard', 'cv', 'ks', 'ad'}, optional
        ``hazard`` regresses the discrete hazard on time, so that a rising
        hazard means ageing and a falling one means a mixture of rates;
        ``cv`` compares the coefficient of variation with the one a rounded
        exponential of the same scale has; ``ks`` and ``ad`` are the
        Kolmogorov-Smirnov and the Anderson-Darling distance from a fitted
        exponential.
    n_boot : int, optional
        Number of bootstrap replicates behind the p value. All four methods draw
        them, under the same null and at the same scale.
    rng : numpy.random.Generator or int or None, optional
        Generator for the bootstrap, or a seed for
        :func:`numpy.random.default_rng`.

    Returns
    -------
    TestResult
        The statistic, its p value, the number of complete durations used and
        the pieces the statistic was built from.

    Raises
    ------
    ValueError
        If `durations` does not obey the durations schema, if `state` or
        `method` is unknown, if `n_boot` is below one, if fewer than two
        complete durations of `state` are available, or, for ``hazard``, if
        fewer than three times have enough records still at risk, if the hazard
        at those times lies exactly on a straight line, which leaves the slope
        with no standard error and its drift with nothing to measure, or if
        every one of the `n_boot` replicates was dropped for one of those two
        reasons, which leaves the drift with nothing to be read against.

    Notes
    -----
    The null hypothesis of all four methods is that the durations are an
    exponential sample rounded up to whole weeks, which is the rounding rule of
    the study and the only shape a durations frame can hold. Their p values come
    from replicates drawn under exactly that null and rounded the same way, a
    parametric bootstrap in the manner of Lilliefors, because the scale is
    estimated from the sample rather than given. Testing whole week durations
    against an unrounded exponential instead would reject a perfectly memoryless
    record on the rounding alone.

    The scale the replicates are drawn at is the scale of the exponential behind
    the rounding, reported as ``details['scale']`` by all four, and it is about
    half a week shorter than the mean of the durations themselves. That
    difference is the whole test at the four or five weeks the paper reports for
    a relapse: replicates drawn at the mean are rounded twice over, land further
    from an exponential than the data do, and reject a memoryless cohort of the
    paper's size almost every time. See :func:`_exponential_scale`.

    ``cv`` is read the same way. The coefficient of variation of a rounded
    exponential of mean m is near ``m / (m + 0.5)`` rather than the 1 of an
    unrounded one, so the sample value is compared with the mean of the
    replicates, reported as ``details['null_cv']``, and the p value is the share
    of replicates at least as far from it as the sample is. Comparing with a
    literal 1 instead rejects two thirds of memoryless cohorts of relapses.

    ``hazard`` reads only the drift of the weekly hazard, which is less than the
    other three ask: its statistic says nothing about the shape between the
    integers, and it is kept signed, positive for a rising hazard and negative
    for a falling one, so that the direction can be read off it. Its p value is
    the share of replicates whose drift is at least as far from zero, and not
    the p value the least squares fit prints. Least squares takes the hazard
    points as equally precise, while the variance of the hazard at a week is
    ``h (1 - h) / at_risk`` and grows sharply as the at risk set empties, so the
    printed p value is too small: on memoryless durations of the length and
    number the paper reports for a relapse it falls below 0.05 for nearly one
    cohort in ten rather than one in twenty, and worse as the cohort grows. See
    :func:`_hazard_test`.

    ``ks`` has little to say about short durations. Its distance from a
    continuous exponential is dominated there by the width of the weekly step,
    which is a function of the sample mean alone, so the statistic lands near the
    middle of its own null distribution whatever the shape of the record is and
    the p value comes back near a half. It keeps its power against an ageing
    record, whose steps do not follow the mean the same way, but read ``ad`` or
    ``cv`` beside it before concluding that a short duration is memoryless.

    Every p value here is ``(1 + exceedances) / (replicates + 1)``, so none of
    them is ever exactly 0 and the smallest one a run can report is set by
    ``details['n_boot']``. That count is `n_boot` itself for ``cv``, ``ks`` and
    ``ad``, and for ``hazard`` it is the replicates whose hazard could be read
    at all, which is `n_boot` less the ones too short to carry three weeks with
    enough still at risk.

    On a short cohort that loss is most of the replicates, so read
    ``details['n_boot']`` before the ``hazard`` p value beside it: half a dozen
    relapses of a few weeks each keep only about one replicate in eight. A run
    that keeps none at all is refused rather than reported, because the share of
    nothing that exceeds the statistic is nothing and the p value would come
    back as exactly 1 from a bootstrap that never ran.
    """
    _validate_choice("method", method, _MEMORYLESS_METHODS)
    _validate_state(state)
    if n_boot < 1:
        raise ValueError(f"n_boot must be a positive count, got {n_boot!r}")
    validate(durations, "durations")

    selected = durations[(durations["state"] == state) & ~durations["censored"]]
    values = selected["duration_w"].to_numpy(dtype=np.float64)
    if values.size < 2:
        raise ValueError(
            f"at least two complete durations of state {state:+d} are needed to test "
            f"memorylessness, got {values.size}"
        )
    if method == "hazard":
        return _hazard_test(values, n_boot, rng)
    if method == "cv":
        return _cv_test(values, n_boot, rng)
    return _distance_test(values, method, n_boot, rng)


def discrete_hazard(
    durations_complete: npt.ArrayLike, min_at_risk: int = MIN_AT_RISK
) -> tuple[_Vector, _Vector, npt.NDArray[np.int64]]:
    """Return the weekly hazard of a set of complete durations.

    The hazard of week k is the share of the durations still at risk at the
    start of that week which end in it. A memoryless duration has the same
    hazard in every week, an ageing one a rising hazard and a mixture of rates a
    falling one. This is the hazard the ``hazard`` method of
    :func:`test_memoryless` regresses on time, and the one the inset of
    :func:`msrelapse.plots.fig_survival_vs_exponential` draws.

    Parameters
    ----------
    durations_complete : array_like
        Complete durations, whole weeks and at least one week each, and no
        longer than a million weeks, which is past the length of any record. A
        censored run never ended, so it has no week to end in and belongs
        nowhere here.
    min_at_risk : int, optional
        Fewest durations that have to be still at risk in a week for the hazard
        of that week to be read. Must be at least one; the default is
        :data:`MIN_AT_RISK`.

    Returns
    -------
    times : numpy.ndarray
        The weeks that were read, in ascending order.
    hazard : numpy.ndarray
        The share of the durations at risk in each of those weeks that ended
        in it.
    at_risk : numpy.ndarray
        How many durations were at risk in each of them. This is what the
        precision of a hazard point is made of: the variance of the hazard of a
        week is ``hazard (1 - hazard) / at_risk``, so the late weeks of a record
        carry much the least precise points.

    Raises
    ------
    ValueError
        If `durations_complete` is not one dimensional or holds no duration at
        all, if a duration is not a whole number of weeks of at least one week
        or is longer than a million weeks, or if `min_at_risk` is below one.

    Notes
    -----
    All three arrays come back empty when no week holds `min_at_risk` durations
    still at risk, which is every set of fewer than that many durations. The
    callers read that emptiness rather than a hazard of one event over one
    patient: :func:`test_memoryless` refuses such a set outright and the
    survival figure writes a note in place of its inset.

    Examples
    --------
    >>> times, hazard, at_risk = discrete_hazard([1, 1, 2, 3, 3, 4], min_at_risk=2)
    >>> times.tolist(), at_risk.tolist()
    ([1.0, 2.0, 3.0], [6, 4, 3])
    >>> [round(value, 4) for value in hazard.tolist()]
    [0.3333, 0.25, 0.6667]
    """
    values = np.asarray(durations_complete, dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError(
            f"durations_complete must be one duration per completed run and hold at least "
            f"one of them, got shape {values.shape}"
        )
    if not np.all(np.isfinite(values) & (values >= 1.0) & (values == np.floor(values))):
        raise ValueError(
            "every duration must be a whole number of weeks of at least one week, which is "
            "all a weekly record can hold, but at least one of these is not"
        )
    longest = float(values.max())
    if longest > _MAX_HAZARD_WEEKS:
        raise ValueError(
            f"the hazard reads one week at a time, so a duration may be at most "
            f"{_MAX_HAZARD_WEEKS} weeks, which is longer than any record; the longest of "
            f"these is {longest!r}"
        )
    if min_at_risk < 1:
        raise ValueError(f"min_at_risk must be a positive count of records, got {min_at_risk!r}")
    weeks = values.astype(np.int64)
    deaths = np.bincount(weeks)[1:]
    times = np.arange(1, deaths.size + 1, dtype=np.float64)
    at_risk = weeks.size - np.concatenate(([0], np.cumsum(deaths)[:-1]))
    keep = at_risk >= min_at_risk
    hazard: _Vector = deaths[keep] / at_risk[keep]
    remaining: npt.NDArray[np.int64] = at_risk[keep]
    return times[keep], hazard, remaining


def test_periodicity(
    weekly: pd.DataFrame,
    method: PeriodicityMethod = "fisher_g",
    n_perm: int = 200,
    rng: Seed = None,
) -> PeriodicityResult:
    """Look for a period in the relapse onsets of each record and pool the evidence.

    The paper states that relapses have no typical periodicity, and this is the
    test of that statement. What is tested is the series of relapse onsets, one
    impulse in every week in which a relapse starts, and not the +1 and -1 state
    series itself. Each patient's onsets are turned into a periodogram, the
    largest ordinate is compared with what the null hypothesis of the method
    allows, and the per patient p values are combined by Fisher's method.

    Parameters
    ----------
    weekly : pandas.DataFrame
        A frame in the weekly schema of :mod:`msrelapse.io`. It is validated
        before use.
    method : {'fisher_g', 'lombscargle'}, optional
        ``fisher_g`` takes the discrete Fourier periodogram of the mean removed
        onset series and Fisher's exact g test; ``lombscargle`` takes the
        Rayleigh power of the onset times on a frequency grid from one cycle per
        record to the weekly Nyquist frequency, and a permutation p value. The
        Rayleigh power is the periodogram of the train of impulses the onsets
        form, which is what the Lomb-Scargle periodogram the option is named
        after becomes once the series read is a set of event times rather than a
        measurement taken at each of them.
    n_perm : int, optional
        Number of permutations behind the ``lombscargle`` p value. Unused by
        ``fisher_g``, whose p value is exact.
    rng : numpy.random.Generator or int or None, optional
        Generator for the permutations, or a seed for
        :func:`numpy.random.default_rng`.

    Returns
    -------
    PeriodicityResult
        The per patient table and the pooled test.

    Raises
    ------
    ValueError
        If `weekly` does not obey the weekly schema, if `method` is unknown, if
        `n_perm` is below one, or if no patient has a record this test can read.

    Notes
    -----
    Why the onsets and not the states. Fisher's g test asks whether the largest
    ordinate of a periodogram stands out from the rest, and its null hypothesis
    is white noise, a series whose weeks are independent of one another. The
    state series of a relapsing-remitting record is nothing like white noise: it
    stays in remission for about a hundred weeks and in relapse for about four,
    so its spectrum is concentrated at the low frequencies and its largest
    ordinate is large whatever the rhythm of the relapses is. Applied to the
    state series the test therefore rejects a record with no periodicity at all,
    which is a fault of the statistic and not a finding about the record. Under
    the memoryless hypothesis of the paper the relapse onsets are a renewal
    process with geometric gaps, and the weekly onset indicator is close to a
    Bernoulli series, whose spectrum is flat; periodic relapses put a peak in
    it. That is the series both methods read.

    A patient of fewer than eight weeks is skipped, and so is a patient of fewer
    than three onsets: two onsets leave one gap, and any two relapses are one
    cycle apart whatever the cycle. Under ``fisher_g`` a patient whose onsets
    fall in every other week is skipped as well, because all of the power of
    such a series sits at the two week Nyquist limit, which that test does not
    read. Every count is reported in the ``details`` of the pooled result.

    The limitation that is left is the refractory period: no relapse can start
    while the previous one is still running, so the onsets are slightly under
    dispersed compared with a Bernoulli series, and the test is conservative
    rather than liberal. At the durations of the paper, gaps of the order of a
    hundred weeks and relapses of about four, the effect on the size of the test
    is negligible, and ``tests/test_fit.py`` measures it on cohorts of the shape
    of the study.

    What a large p value says. Both methods are conservative on a record of few
    onsets, and a patient followed for a few hundred weeks at the rate of the
    paper has only a handful. Under ``fisher_g`` the reason is a ceiling: the
    periodogram of a train of n impulses cannot exceed n squared however the
    impulses are arranged, while the exponential null behind
    :func:`_fisher_g_p_value` has no such ceiling, so the largest ordinate of a
    sparse train falls short of what white noise would produce. Over records of
    400 weeks with the onsets placed at random, the share of p values at or
    below 0.05 is 0.000 at six onsets, 0.007 at ten, 0.026 at twenty and 0.039
    at forty, against the 0.05 a test of the nominal size would give. Read a
    large p value as no rhythm being visible in these few onsets, and not as
    evidence that there is none.

    The permutation p value of ``lombscargle`` shuffles the gaps between
    consecutive onsets, which keeps the distribution of the gaps and destroys
    any periodic arrangement of them. It is conservative on few onsets for a
    reason of its own: a patient of few onsets has few distinct shuffles, and a
    shuffle that reproduces the observed arrangement, or its mirror image, which
    carries exactly the same power at every frequency, ties with the observed
    statistic and is counted. Over the same random records the smallest p value
    a patient can reach is about 0.43 at three onsets, 0.14 at four and 0.05 at
    five, and only from about six onsets is the test of its nominal size. Such a
    patient is read for what it is, a record too short to carry evidence, rather
    than treated as evidence of no rhythm. The frequency grid of ``lombscargle``
    also keeps the two week limit that ``fisher_g`` drops, and on a record of few
    onsets the largest power lands near that limit often enough that a reported
    ``period_weeks`` beside a large p value says nothing; read a period beside
    the p value of its own row.
    """
    _validate_choice("method", method, _PERIODICITY_METHODS)
    if n_perm < 1:
        raise ValueError(f"n_perm must be a positive count, got {n_perm!r}")
    validate(weekly, "weekly")
    generator = _generator(rng)

    rows: list[tuple[str, int, float, float, float, int]] = []
    p_values: list[float] = []
    n_short = 0
    n_few_onsets = 0
    n_flat_spectrum = 0
    for patient, group in weekly.groupby("patient_id", sort=True):
        series = group["state"].to_numpy(dtype=np.int64)
        onsets = _onset_indicator(series)
        n_onsets = int(onsets.sum())
        skipped = (str(patient), series.size, math.nan, math.nan, math.nan, n_onsets)
        if series.size < _MIN_PERIODICITY_WEEKS:
            n_short += 1
            rows.append(skipped)
            continue
        if n_onsets < _MIN_ONSETS:
            n_few_onsets += 1
            rows.append(skipped)
            continue
        if method == "fisher_g":
            read = _fisher_g(onsets)
            if read is None:
                n_flat_spectrum += 1
                rows.append(skipped)
                continue
            statistic, p_value, period = read
        else:
            statistic, p_value, period = _rayleigh(
                np.flatnonzero(onsets).astype(np.float64), series.size, n_perm, generator
            )
        rows.append((str(patient), series.size, statistic, p_value, period, n_onsets))
        p_values.append(p_value)

    if not p_values:
        raise ValueError(
            f"no patient has a record this test can read: {n_short} were shorter than "
            f"{_MIN_PERIODICITY_WEEKS} weeks, {n_few_onsets} had fewer than {_MIN_ONSETS} "
            f"relapse onsets and {n_flat_spectrum} had their onsets in every other week, "
            f"which leaves the periodogram of this test no ordinate to read"
        )
    per_patient = pd.DataFrame(rows, columns=list(_PER_PATIENT_COLUMNS))
    pooled = _combine_p_values(
        p_values,
        method=f"fisher_combined:{method}",
        details={
            "n_patients": float(len(rows)),
            "n_skipped_short": float(n_short),
            "n_skipped_few_onsets": float(n_few_onsets),
            "n_skipped_flat_spectrum": float(n_flat_spectrum),
        },
    )
    return PeriodicityResult(per_patient=per_patient, pooled=pooled)


def fit_nb_counts(counts: pd.Series[int] | npt.ArrayLike) -> NBFit:
    """Fit relapse counts as negative binomial and test them against Poisson.

    The NB2 parameterisation is used: a mean m and a dispersion a, with variance
    ``m + a m**2``, so that a of zero is the Poisson case and a large a is a
    cohort in which a few patients carry most of the relapses. The fit maximises
    the log likelihood over the logarithms of the two parameters, started from
    the method of moments.

    Parameters
    ----------
    counts : pandas.Series or array_like
        One whole, non-negative count per patient.

    Returns
    -------
    NBFit
        The fitted mean and dispersion, the interval of the dispersion, both log
        likelihoods and the likelihood ratio test against the Poisson null.

    Raises
    ------
    ValueError
        If `counts` is empty or holds a single value, or if any count is
        missing, negative, not a whole number or so large that its square leaves
        float64.

    Notes
    -----
    The dispersion cannot be negative, so under the Poisson null it sits on the
    boundary of the parameter space and the likelihood ratio statistic follows a
    half and half mixture of a chi square with one degree of freedom and a point
    mass at zero. The p value is therefore half the chi square tail, as Chernoff
    showed and as every count regression package does. At a statistic of exactly
    zero the p value is 1 rather than a half, because the point mass of the
    mixture sits there and nothing at all is ruled out.

    Counts carry evidence of overdispersion only when the score of the
    dispersion at zero is positive, which happens exactly when their population
    variance exceeds their mean. That is the screen applied here. The sample
    variance, larger by a factor of ``n / (n - 1)``, is the wrong comparison:
    between the two lies a band of ordinary cohorts whose maximiser is on the
    boundary while their sample variance suggests otherwise, and about one
    Poisson cohort in twenty of the size of this study falls in it. A cohort
    inside the band is fitted as Poisson rather than pushed against the edge of
    the parameter space, where the curvature of the likelihood is of the order of
    the dispersion itself and float64 cannot resolve it.

    :mod:`msrelapse.stats` reaches the same cohorts through a related quantity
    rather than through this one. It screens on the method of moments dispersion
    around its own fitted Poisson mean, which for a model of an intercept alone
    read at equal follow up is the population variance less the mean over the
    square of the mean, so on such a cohort the two screens agree on which counts
    are overdispersed at all. What that module actually fits is an intercept and
    an arm, over a follow up that varies from patient to patient, so its fitted
    means vary with the follow up and the two quantities part company by that
    much. It then asks its own quantity to clear its dispersion floor rather than
    merely to be positive, because it starts its search from it and a start below
    the floor has nowhere to go.

    A cohort screened out that way, or one whose search settles on a dispersion
    no model can tell from zero, or one whose observed information leaves the log
    dispersion without positive curvature, gets a dispersion of exactly zero, a p
    value of 1 and the degenerate interval ``(0.0, 0.0)``. A cohort in which
    nobody relapsed is the extreme of that case and a legitimate one, of a short
    counting window or a mild cohort: its mean, dispersion and both log
    likelihoods are all zero.

    The mean reported is the sample mean, not the one the search stopped on. The
    score of an intercept only NB2 fit in the log mean is
    ``sum((y - m) / (1 + a m))``, whose root is the sample mean at every
    dispersion, so the maximiser of that coordinate is known in closed form and a
    quasi-Newton search can only add noise to it. Reporting the exact value keeps
    `mean` and `loglik_poisson`, which is read at the same place, consistent with
    one another and free of the release of the optimiser.

    The interval of the dispersion comes from the observed information, the
    Hessian of the negative log likelihood at the fitted parameter, differenced
    from the analytic score. The inverse Hessian the quasi-Newton search carries
    is not used: it is a record of the path the search took and can be out by a
    factor of five either way. An interval wider than float64 can exponentiate is
    reported as ``(0.0, inf)``, which is what a Wald half width of several
    hundred on the log scale says.

    The convergence flag of the quasi-Newton search is not reported, because the
    log likelihood is flat near its maximum and the search routinely stops on a
    loss of precision while sitting on the right answer. Read `loglik_nb` and
    `loglik_poisson`, which are both returned, to judge the fit instead.
    """
    values = _count_array(counts)
    mean = float(values.mean())
    loglik_poisson = _poisson_loglik(values, mean)
    poisson = NBFit(
        mean=mean,
        dispersion=0.0,
        dispersion_ci=(0.0, 0.0),
        loglik_nb=loglik_poisson,
        loglik_poisson=loglik_poisson,
        lrt_statistic=0.0,
        p_value=_boundary_p_value(0.0),
        n=values.size,
    )
    variance = float(values.var())
    if variance <= mean:
        return poisson

    start = np.array([math.log(mean), math.log((variance - mean) / mean**2)])
    best = optimize.minimize(
        _nb_negative_loglik, start, args=(values,), jac=_nb_negative_score, method="BFGS"
    )
    fitted = np.asarray(best.x, dtype=np.float64)
    # The screen above settles the question for data the likelihood agrees with,
    # since the maximum likelihood dispersion is zero exactly when the moment one
    # is. These two checks catch the search rather than the data: a cohort whose
    # two variances straddle its mean by a rounding bit passes the screen and is
    # then walked to the boundary, where the dispersion is no longer distinct
    # from zero and its curvature no longer distinct from noise. Both outcomes
    # are the Poisson fit, as in msrelapse.stats.
    dispersion = math.exp(
        _bounded_log_parameter(float(fitted[_LOG_DISPERSION]), _LOG_DISPERSION_LIMIT)
    )
    if dispersion <= _DISPERSION_FLOOR:
        return poisson
    standard_error = _log_dispersion_standard_error(fitted, values)
    if standard_error is None:
        return poisson
    loglik_nb = -float(best.fun)
    statistic = max(2.0 * (loglik_nb - loglik_poisson), 0.0)
    return NBFit(
        mean=mean,
        dispersion=dispersion,
        dispersion_ci=_log_scale_interval(dispersion, _z_value(_DEFAULT_CI) * standard_error),
        loglik_nb=loglik_nb,
        loglik_poisson=loglik_poisson,
        lrt_statistic=statistic,
        p_value=_boundary_p_value(statistic),
        n=values.size,
    )


def fit_gamma_rates(durations: pd.DataFrame) -> GammaFit:
    """Fit the spread of the per patient relapse onset rate across a cohort.

    Each patient contributes one plug in rate, the number of remissions that
    ended in a relapse over the weeks that patient spent in remission, censored
    remissions included. A gamma distribution is then fitted to those rates by
    maximum likelihood with the location held at zero.

    Parameters
    ----------
    durations : pandas.DataFrame
        A frame in the durations schema of :mod:`msrelapse.io`. It is validated
        before use.

    Returns
    -------
    GammaFit
        The fitted shape and scale, the rates that were fitted and how many
        patients they came from.

    Raises
    ------
    ValueError
        If `durations` does not obey the durations schema, if it holds no
        remission at all, or if fewer than two patients are left with a
        positive rate.

    Notes
    -----
    This is a plug in estimate and a noisy one: a patient with three remissions
    has a rate known to within a factor of two, and the gamma is fitted to those
    noisy rates as though they were the true ones. The spread it reports is
    therefore the spread of the estimates, which is wider than the spread of the
    rates. :func:`fit_nb_counts` is the usual alternative and the better one: it
    fits the same gamma mixture through the counts themselves, so the Poisson
    noise of a short record stays where it belongs.

    Two kinds of patient are left out. One with no remission run at all
    contributes no rate, since the denominator is the time spent in remission
    and every recorded run lasts at least a week. One whose remissions never
    ended in a relapse has a plug in rate of exactly zero, which a gamma
    likelihood cannot take; such a patient is dropped as well. Both are patients
    of few or no relapses, so the shape comes back a little larger, that is the
    cohort a little more uniform, than it is.
    """
    validate(durations, "durations")
    remissions = durations[durations["state"] == _HEALTH]
    if remissions.empty:
        raise ValueError(
            "the frame holds no remission, so no onset rate can be formed; a rate is the "
            "number of remissions that ended in a relapse over the weeks spent in remission"
        )
    patients = remissions["patient_id"]
    weeks = remissions["duration_w"].groupby(patients, sort=True).sum()
    completed = (~remissions["censored"]).groupby(patients, sort=True).sum()
    rates = (completed / weeks).astype(np.float64).rename("onset_rate").rename_axis("patient_id")
    positive = rates[rates > 0.0]
    if positive.size < 2:
        raise ValueError(
            f"a gamma fit needs at least two patients with a positive onset rate, got "
            f"{positive.size} of {rates.size}; the rest never completed a remission"
        )
    shape, _location, scale = stats.gamma.fit(positive.to_numpy(dtype=np.float64), floc=0)
    return GammaFit(k=float(shape), theta=float(scale), per_patient_rates=positive, n=positive.size)


@overload
def barrier_ratio(durations: pd.DataFrame, per_patient: Literal[False] = False) -> float: ...


@overload
def barrier_ratio(durations: pd.DataFrame, per_patient: Literal[True]) -> pd.Series[float]: ...


def barrier_ratio(durations: pd.DataFrame, per_patient: bool = False) -> float | pd.Series[float]:
    """Return the barrier ratio of equation (7) implied by observed durations.

    Equation (7) reads ``delta V1 / delta V2 = log(tau_x1) / log(tau_x2)``, with
    the two mean durations in weeks. The means are the naive ones, over every
    recorded run of a state with no censoring correction and no continuity
    correction, because those are the means the paper used to reach its 3.1.

    Parameters
    ----------
    durations : pandas.DataFrame
        A frame in the durations schema of :mod:`msrelapse.io`. It is validated
        before use.
    per_patient : bool, optional
        With False the ratio is computed once over the whole frame. With True
        one ratio is computed per patient.

    Returns
    -------
    float or pandas.Series
        The cohort ratio, or one ratio per patient indexed by patient_id. A
        patient missing either state, or whose mean duration in either state is
        one week or shorter, gets a missing value, because the logarithm of such
        a duration is zero or negative and equation (7) is then undefined.

    Raises
    ------
    ValueError
        If `durations` does not obey the durations schema, or, when
        `per_patient` is False, if the frame holds no run of one of the two
        states or a cohort mean of one week or shorter.

    See Also
    --------
    msrelapse.model.barrier_ratio_from_durations : The estimator itself.
    msrelapse.model.beta_from_barrier_ratio : The asymmetry that delivers a ratio.

    Notes
    -----
    Equation (7) follows from equations (5) and (6) only because those drop the
    prefactor of the exit time, which silently sets the prefactor of both wells
    to one week. Treat the number as the documented heuristic of the paper, as
    :func:`msrelapse.model.barrier_ratio_from_durations` explains at greater
    length, and not as a definition of the barrier ratio.
    """
    validate(durations, "durations")
    if not per_patient:
        return barrier_ratio_from_durations(
            _naive_mean(durations, _HEALTH), _naive_mean(durations, _NO_HEALTH)
        )
    ratios: dict[str, float] = {}
    for patient, group in durations.groupby("patient_id", sort=True):
        ratios[str(patient)] = _patient_barrier_ratio(group)
    return pd.Series(ratios, dtype=np.float64, name="barrier_ratio").rename_axis("patient_id")


def _validate_choice(name: str, value: str, allowed: tuple[str, ...]) -> None:
    """Raise if a string argument is not one of the spellings the API accepts."""
    if value not in allowed:
        raise ValueError(f"{name} must be one of {allowed}, got {value!r}")


def _validate_state(state: int) -> None:
    """Raise if a state is neither of the two clinical codes."""
    if state not in _STATES:
        raise ValueError(
            f"state must be {_NO_HEALTH:+d} for no health or {_HEALTH:+d} for health, got {state!r}"
        )


def _generator(rng: Seed) -> np.random.Generator:
    """Return the generator to draw from, building one from a seed if needed."""
    if isinstance(rng, np.random.Generator):
        return rng
    return np.random.default_rng(rng)


def _z_value(ci: float) -> float:
    """Return the normal quantile that two sided coverage `ci` asks for."""
    return float(stats.norm.ppf(0.5 + 0.5 * ci))


def _estimate(
    values: _Vector,
    censored: npt.NDArray[np.bool_],
    family: str,
    correction: float,
) -> tuple[float, float, float]:
    """Return the rate, the mean and the log likelihood of one duration fit.

    Parameters
    ----------
    values : numpy.ndarray
        Recorded durations in weeks, complete and censored alike.
    censored : numpy.ndarray
        True where the corresponding duration is right censored.
    family : str
        ``'exponential'`` or ``'geometric'``.
    correction : float
        Weeks subtracted from every complete duration, exponential family only.
        A censored duration keeps its whole recorded time, which is a lower
        bound on the true one and needs no midpoint.

    Returns
    -------
    tuple of float
        The rate, the mean in weeks and the log likelihood.

    Raises
    ------
    ValueError
        If the corrected total time is not positive.
    """
    complete = ~censored
    n_complete = int(np.count_nonzero(complete))
    total = (
        float(np.sum(values - correction * complete))
        if family == "exponential"
        else float(values.sum())
    )
    if total <= 0.0:
        raise ValueError(
            f"the total observed time is {total!r} weeks after a continuity correction of "
            f"{correction!r}, so no rate can be estimated"
        )
    rate = n_complete / total
    if family == "exponential":
        return rate, 1.0 / rate, n_complete * math.log(rate) - rate * total
    failures = total - n_complete
    loglik = n_complete * math.log(rate)
    if failures > 0.0:
        loglik += failures * math.log1p(-rate)
    return rate, 1.0 / rate, loglik


def _log_scale_interval(centre: float, half_width: float) -> tuple[float, float]:
    """Return a Wald interval of a positive parameter, built on the log scale.

    Parameters
    ----------
    centre : float
        The point estimate, which the interval is symmetric around on the log
        scale.
    half_width : float
        Half the width of the interval, in logarithms of `centre`.

    Returns
    -------
    tuple of float
        The two endpoints. A half width beyond :data:`_LARGEST_EXPONENT` has no
        exponential in float64, and the interval is then ``(0.0, inf)``, which is
        what a half width of several hundred says in any case: the parameter is
        not pinned down at all. Saturating rather than raising keeps a fit that
        succeeded from dying on its interval.
    """
    if half_width >= _LARGEST_EXPONENT:
        return 0.0, math.inf
    return centre * math.exp(-half_width), centre * math.exp(half_width)


def _fisher_interval(
    mean: float, rate: float, family: str, n_complete: int, ci: float
) -> tuple[float, float]:
    """Return the interval of `mean` from the information on the log rate.

    Parameters
    ----------
    mean : float
        The fitted mean in weeks, which the interval is centred on.
    rate : float
        The fitted rate: the weekly rate of the exponential family, or the
        success probability of the geometric one.
    family : str
        ``'exponential'`` or ``'geometric'``.
    n_complete : int
        Number of complete runs, the events the information counts.
    ci : float
        Coverage of the interval.

    Returns
    -------
    tuple of float
        The two endpoints. The interval is degenerate at a geometric rate of
        exactly 1, where every run ended in its first week: the information on
        the log rate is then infinite and the half width vanishes, which is the
        Wald interval at the boundary of the parameter and not a claim that the
        mean is known. Ask for a bootstrap interval there instead.

    Notes
    -----
    Each family carries the observed information of its own log likelihood on
    the log rate, and the mean is the reciprocal of that rate, so the half width
    reads across to the mean unchanged. The exponential likelihood is
    ``n_complete log(rate) - rate * total``, whose information on the log rate is
    `n_complete` however long the censored runs make the total, so the half width
    is ``z / sqrt(n_complete)``. The geometric likelihood is the binomial one of
    `n_complete` weeks that ended out of the total weeks recorded, whose
    information on the log success probability is ``n_complete / (1 - rate)``, so
    the half width is ``z sqrt((1 - rate) / n_complete)``. The two differ by
    ``sqrt(1 - rate)``: at the 4.3 week relapse of the paper the exponential half
    width is 14 percent the wider of the two, and at the 100 week remission half
    a percent.
    """
    half_width = _z_value(ci) / math.sqrt(n_complete)
    if family == "geometric":
        half_width *= math.sqrt(1.0 - rate)
    return _log_scale_interval(mean, half_width)


def _bootstrap_interval(
    values: _Vector,
    censored: npt.NDArray[np.bool_],
    family: str,
    correction: float,
    *,
    ci: float,
    n_boot: int,
    rng: Seed,
) -> tuple[float, float]:
    """Return the percentile interval of the mean over resampled runs.

    Raises
    ------
    ValueError
        If every resample holds only censored runs, so that no mean can be
        formed at all.
    """
    generator = _generator(rng)
    indices = generator.integers(0, values.size, size=(n_boot, values.size))
    means: list[float] = []
    for row in indices:
        resampled = censored[row]
        if bool(resampled.all()):
            continue
        means.append(_estimate(values[row], resampled, family, correction)[1])
    if not means:
        raise ValueError(
            f"all {n_boot} bootstrap resamples held only censored runs, so no interval "
            f"could be formed from {values.size} run(s)"
        )
    low, high = np.percentile(means, [50.0 * (1.0 - ci), 50.0 * (1.0 + ci)])
    return float(low), float(high)


def _hazard_drift(values: _Vector) -> float | None:
    """Return the slope of the hazard over its standard error, or None.

    Parameters
    ----------
    values : numpy.ndarray
        Complete durations, whole weeks and at least one week each.

    Returns
    -------
    float or None
        The signed ratio, positive when the hazard rises. None says these
        durations carry no readable drift at all: fewer than
        :data:`_MIN_HAZARD_TIMES` weeks have :data:`MIN_AT_RISK` still at risk,
        or the hazard at those weeks lies exactly on a straight line, which
        leaves the slope with no standard error. A replicate of the null that
        lands there is dropped rather than counted; :func:`_hazard_test` raises
        on either case for the durations themselves.
    """
    times, hazard, _at_risk = discrete_hazard(values)
    if times.size < _MIN_HAZARD_TIMES:
        return None
    line = stats.linregress(times, hazard)
    standard_error = float(line.stderr)
    if not standard_error > 0.0:
        return None
    return float(line.slope) / standard_error


def _hazard_test(values: _Vector, n_boot: int, rng: Seed) -> TestResult:
    """Return the drift of the discrete hazard, read against the rounded null.

    Raises
    ------
    ValueError
        If fewer than :data:`_MIN_HAZARD_TIMES` times have at least
        :data:`MIN_AT_RISK` records still at risk, if the hazard at those times
        lies exactly on a straight line, which leaves the slope with no standard
        error and the drift with nothing to measure, or if every replicate of
        the null was dropped for one of those two reasons, which leaves the
        drift with nothing to be read against.

    Notes
    -----
    The statistic is the slope of the hazard over time divided by its standard
    error, kept signed so that a rising hazard reads positive and a falling one
    negative. Its p value is not the one the regression prints. Ordinary least
    squares takes the points as equally precise, while the variance of a weekly
    hazard is ``h (1 - h) / at_risk`` and grows sharply as the at risk set
    empties, and those late imprecise points sit at the end of the time axis
    where they pull the line hardest. The printed p value is too small for that
    reason: on 218 durations drawn as ``ceil(Exp(4.3))``, memoryless by
    construction and the relapses the paper reports, it falls below 0.05 for
    nearly one cohort in ten rather than one in twenty, and the share grows with
    the number of durations rather than falling away.

    So the p value here is the share of replicates drawn under the rounded
    exponential null whose drift is at least as far from zero as the sample's,
    the same parametric bootstrap :func:`_cv_test` and :func:`_distance_test`
    draw, at the same scale and rounded the same way. That share holds its level
    at every sample size checked and keeps the power of the statistic against an
    ageing record.
    """
    times, hazard, _at_risk = discrete_hazard(values)
    if times.size < _MIN_HAZARD_TIMES:
        raise ValueError(
            f"a hazard regression needs at least {_MIN_HAZARD_TIMES} weeks with "
            f"{MIN_AT_RISK} or more durations still at risk, got "
            f"{times.size} from {values.size} duration(s)"
        )
    line = stats.linregress(times, hazard)
    standard_error = float(line.stderr)
    if not standard_error > 0.0:
        raise ValueError(
            f"the hazard of these durations lies exactly on the straight line of slope "
            f"{float(line.slope)!r} across the {times.size} week(s) with "
            f"{MIN_AT_RISK} or more at risk, so the slope has no standard error and its "
            f"drift cannot be measured; test the durations another way"
        )
    statistic = float(line.slope) / standard_error
    scale = _exponential_scale(values)
    draws = _null_draws(scale, (n_boot, values.size), _generator(rng))
    replicates = np.array(
        [drift for row in draws if (drift := _hazard_drift(row)) is not None], dtype=np.float64
    )
    if replicates.size == 0:
        raise ValueError(
            f"none of the {n_boot} replicates drawn at scale {scale!r} carried "
            f"{_MIN_HAZARD_TIMES} weeks with {MIN_AT_RISK} or more still at risk, so the "
            f"drift of these {values.size} duration(s) has nothing to be read against; "
            f"raise n_boot or test the durations another way"
        )
    exceeded = int(np.count_nonzero(np.abs(replicates) >= abs(statistic)))
    return TestResult(
        method="hazard",
        statistic=statistic,
        p_value=float(1 + exceeded) / (replicates.size + 1),
        n=int(values.size),
        details={
            "mean": float(values.mean()),
            "scale": scale,
            "slope": float(line.slope),
            "intercept": float(line.intercept),
            "n_times": float(times.size),
            "n_boot": float(replicates.size),
        },
    )


def _exponential_scale(values: _Vector) -> float:
    """Return the scale of the exponential whose weekly ceiling gave `values`.

    Parameters
    ----------
    values : numpy.ndarray
        Complete durations, whole weeks and at least one week each.

    Returns
    -------
    float
        The scale ``tau`` for which ``ceil(Exp(tau))`` has the mean of `values`.
        Rounding an exponential up to whole weeks gives a geometric law of
        success probability ``1 - exp(-1 / tau)``, whose maximum likelihood
        estimate is the number of durations over their total, so the scale is
        ``-1 / log(1 - p)``. A sample whose runs all lasted a single week puts
        that probability at 1 and the scale at 0, the degenerate exponential
        every draw of which is rounded up to the same one week.

    Notes
    -----
    The mean of `values` is not that scale and is about half a week longer than
    it, because every draw is rounded up. Drawing the replicates of a parametric
    bootstrap at the mean rather than at the scale rounds them twice over, which
    leaves them further from an exponential than the data are and turns a
    memoryless record into a rejection. See the Notes of :func:`test_memoryless`.
    """
    probability = float(values.size) / float(values.sum())
    if probability >= 1.0:
        return 0.0
    return -1.0 / math.log1p(-probability)


def _null_draws(scale: float, shape: tuple[int, int], generator: np.random.Generator) -> _Vector:
    """Return memoryless durations of the given scale, rounded up to whole weeks."""
    return np.maximum(np.ceil(generator.exponential(scale, size=shape)), 1.0)


def _cv_test(values: _Vector, n_boot: int, rng: Seed) -> TestResult:
    """Return the coefficient of variation against the one of the rounded null."""
    mean = float(values.mean())
    cv = float(values.std(ddof=1)) / mean
    scale = _exponential_scale(values)
    draws = _null_draws(scale, (n_boot, values.size), _generator(rng))
    replicates = draws.std(ddof=1, axis=1) / draws.mean(axis=1)
    null_cv = float(replicates.mean())
    exceeded = int(np.count_nonzero(np.abs(replicates - null_cv) >= abs(cv - null_cv)))
    return TestResult(
        method="cv",
        statistic=cv,
        p_value=float(1 + exceeded) / (n_boot + 1),
        n=int(values.size),
        details={
            "mean": mean,
            "scale": scale,
            "cv": cv,
            "null_cv": null_cv,
            "n_boot": float(n_boot),
        },
    )


def _distance_test(values: _Vector, method: str, n_boot: int, rng: Seed) -> TestResult:
    """Return a distance from the fitted exponential, with a bootstrap p value."""
    statistic = _ks_distance if method == "ks" else _ad_distance
    key = "ks_d" if method == "ks" else "ad_a2"
    scale = _exponential_scale(values)
    draws = _null_draws(scale, (n_boot, values.size), _generator(rng))
    with warnings.catch_warnings():
        # scipy.stats.anderson is on its way to requiring a choice of p value
        # method, and says so. Only its statistic is read here, and the p value
        # below is the bootstrap one, so the choice makes no difference.
        warnings.simplefilter("ignore", FutureWarning)
        observed = statistic(values)
        replicates = np.array([statistic(row) for row in draws])
    return TestResult(
        method=method,
        statistic=observed,
        p_value=float(1 + np.count_nonzero(replicates >= observed)) / (n_boot + 1),
        n=int(values.size),
        details={
            "mean": float(values.mean()),
            "scale": scale,
            key: observed,
            "n_boot": float(n_boot),
        },
    )


def _ks_distance(values: _Vector) -> float:
    """Return the Kolmogorov-Smirnov distance from the fitted exponential."""
    return float(stats.ks_1samp(values, stats.expon(scale=float(values.mean())).cdf).statistic)


def _ad_distance(values: _Vector) -> float:
    """Return the Anderson-Darling statistic against the fitted exponential."""
    return float(stats.anderson(values, dist="expon").statistic)


def _onset_indicator(series: npt.NDArray[np.int64]) -> _Vector:
    """Return one impulse per week in which a relapse starts.

    Parameters
    ----------
    series : numpy.ndarray
        The weekly states of one patient, in week order.

    Returns
    -------
    numpy.ndarray
        A series of the same length, 1.0 in a week of no health whose previous
        week was a week of health and 0.0 everywhere else. Week 0 is an onset
        when the record opens in no health, which is how the records of the
        study begin.
    """
    relapse = series == _NO_HEALTH
    previous = np.concatenate(([False], relapse[:-1]))
    onsets: _Vector = (relapse & ~previous).astype(np.float64)
    return onsets


def _fisher_g(series: _Vector) -> tuple[float, float, float] | None:
    """Return Fisher's g, its exact p value and the period of the largest ordinate.

    Parameters
    ----------
    series : numpy.ndarray
        The onset series of one patient, which is mean removed here.

    Returns
    -------
    tuple of float or None
        The statistic, its p value and the period in weeks, or None when the
        ordinates this test reads carry no power at all.

    Notes
    -----
    The zero frequency is left out because the series has its mean removed and
    there is nothing there. The Nyquist ordinate of an even length record is
    left out as well, because it is real rather than complex, so it is not
    distributed like the others and the finite sum behind
    :func:`_fisher_g_p_value` does not describe it: keeping it makes the exact p
    value anti-conservative, by half again at the eight week minimum of this
    module and by a few per cent at a record of several hundred weeks. What is
    left is the classical ``floor((n - 1) / 2)`` ordinates, for a record of
    either parity, which one slice gives for both.

    A series whose onsets fall in every other week carries all of its power at
    that Nyquist frequency and none anywhere else, so it has no ordinate this
    test can read and None is returned for it. Its rhythm, of exactly two weeks,
    is the one the g test does not look at. What the transform actually leaves at
    the other ordinates of such a series is its own rounding error, below 1e-32
    of the power of the series, which is why the emptiness is read against
    :data:`_SPECTRUM_FLOOR` rather than against an exact zero.
    """
    centred = series - series.mean()
    spectrum = np.abs(np.fft.rfft(centred)) ** 2
    power = spectrum[1 : (centred.size + 1) // 2]
    total = float(power.sum())
    if total <= _SPECTRUM_FLOOR * float(spectrum.sum()):
        return None
    index = int(np.argmax(power))
    g = float(power[index]) / total
    return g, _fisher_g_p_value(g, int(power.size)), centred.size / (index + 1.0)


def _fisher_g_p_value(g: float, m: int) -> float:
    """Return the probability that Fisher's g exceeds `g` on `m` ordinates.

    The classical finite sum is
    ``sum_j (-1)^(j-1) C(m, j) (1 - j g)^(m - 1)`` over ``j = 1 .. floor(1 / g)``.

    Parameters
    ----------
    g : float
        The observed ratio of the largest ordinate to their sum.
    m : int
        Number of periodogram ordinates the ratio was taken over.

    Returns
    -------
    float
        The p value, clipped to [0, 1].

    Notes
    -----
    The terms of the sum alternate and grow enormous for a small `g`, where they
    cancel to a p value near 1 that float64 cannot recover. The sum is therefore
    stopped as soon as a term is larger than the one before it, which is where
    that cancellation begins. The tail of the distribution, the part that
    decides a test, is summed in full: there the terms fall away steeply and the
    first one already carries nearly all of the answer.
    """
    if g <= 0.0:
        return 1.0
    if g >= 1.0:
        return 0.0
    log_choose = float(special.gammaln(m + 1))
    total = 0.0
    previous = math.inf
    for j in range(1, min(m, int(1.0 / g)) + 1):
        remainder = 1.0 - j * g
        if remainder <= 0.0:
            break
        log_term = (
            log_choose
            - float(special.gammaln(j + 1))
            - float(special.gammaln(m - j + 1))
            + (m - 1) * math.log(remainder)
        )
        if log_term > previous:
            break
        previous = log_term
        term = math.exp(log_term)
        total += term if j % 2 == 1 else -term
    return min(max(total, 0.0), 1.0)


def _rayleigh_power(times: _Vector, frequencies: _Vector) -> _Vector:
    """Return the Rayleigh power of a set of onset times at each frequency.

    The power at a frequency f is ``abs(sum_k exp(2 pi i f t_k))**2 / n``, the
    squared length of the sum of one unit phasor per onset, divided by the number
    of onsets. It is the periodogram of a train of unit impulses: onsets that
    fall at the same phase of a cycle of 1 / f add up and give a power of up to
    n, and onsets spread over the cycle cancel and give a power near 1.
    """
    phasors = np.exp(2.0j * math.pi * np.outer(frequencies, times))
    power: _Vector = np.abs(phasors.sum(axis=1)) ** 2 / times.size
    return power


def _rayleigh(
    times: _Vector, n_weeks: int, n_perm: int, generator: np.random.Generator
) -> tuple[float, float, float]:
    """Return the largest Rayleigh power of a set of onsets, its p value and its period.

    Parameters
    ----------
    times : numpy.ndarray
        The week of every relapse onset of one patient, in ascending order.
    n_weeks : int
        Length of the record, which sets the lowest frequency of the grid.
    n_perm : int
        Number of gap shuffles behind the p value.
    generator : numpy.random.Generator
        Generator the shuffles are drawn from.

    Returns
    -------
    tuple of float
        The largest power on the grid, its permutation p value and the period in
        weeks that goes with it.

    Notes
    -----
    The grid runs from one cycle per record to the weekly Nyquist frequency of
    0.5 cycles per week, oversampled by :data:`_LOMB_OVERSAMPLING`. It therefore
    keeps the two week limit that :func:`_fisher_g` drops, and on a train of few
    onsets the largest power lands near that limit often: over three cohorts of
    seventy memoryless records of four hundred weeks, 86 of the 191 patients read
    reported a period below three weeks and 19 of them exactly two. The p value
    is not moved by that, because the surrogates are read on the same grid and
    lean the same way, but the period beside a large p value is the grid talking
    rather than the record, and is worth reading only beside a small one.

    The surrogate that the observed maximum is compared with is the same onsets
    with the gaps between them shuffled, which keeps the first onset, the last
    onset and the distribution of the gaps, and destroys any periodic arrangement
    of them. The p value counts a surrogate whose maximum ties with the observed
    one, so a patient with few gaps, and therefore few distinct shuffles, cannot
    reach a small p value at all. See the Notes of :func:`test_periodicity`.
    """
    frequencies = np.linspace(1.0 / n_weeks, 0.5, max(2, _LOMB_OVERSAMPLING * n_weeks // 2))
    power = _rayleigh_power(times, frequencies)
    best = int(np.argmax(power))
    observed = float(power[best])
    gaps = np.diff(times)
    exceeded = 0
    for _ in range(n_perm):
        surrogate = np.concatenate(([times[0]], times[0] + np.cumsum(generator.permutation(gaps))))
        exceeded += int(float(_rayleigh_power(surrogate, frequencies).max()) >= observed)
    return observed, (1.0 + exceeded) / (n_perm + 1.0), 1.0 / float(frequencies[best])


def _combine_p_values(p_values: list[float], method: str, details: dict[str, float]) -> TestResult:
    """Return Fisher's combined probability test over per patient p values."""
    clipped = np.clip(np.array(p_values, dtype=np.float64), _SMALLEST_P_VALUE, 1.0)
    statistic = -2.0 * float(np.sum(np.log(clipped)))
    return TestResult(
        method=method,
        statistic=statistic,
        p_value=float(stats.chi2.sf(statistic, 2 * clipped.size)),
        n=int(clipped.size),
        details=details,
    )


def _count_array(counts: pd.Series[int] | npt.ArrayLike) -> _Vector:
    """Return the counts as whole non-negative doubles, or explain why not.

    Raises
    ------
    ValueError
        If the counts are not one dimensional, if they are empty or a single
        value, or if one of them is missing, negative, not a whole number or so
        large that its square leaves float64.
    """
    values = np.asarray(counts, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError(f"counts must be one count per patient, got shape {values.shape}")
    if values.size < 2:
        raise ValueError(f"at least two counts are needed to fit a dispersion, got {values.size}")
    if not np.all(np.isfinite(values)):
        raise ValueError("every count must be a finite number, but at least one is not")
    if np.any(values < 0.0):
        raise ValueError(f"every count must be non-negative, the smallest is {values.min()!r}")
    if np.any(values != np.floor(values)):
        raise ValueError("every count must be a whole number of relapses, but at least one is not")
    bound = _LARGEST_SQUARABLE / math.sqrt(values.size)
    if float(values.max()) > bound:
        raise ValueError(
            f"the largest count is {values.max()!r}, and the sum of the squares of "
            f"{values.size} counts leaves float64 above {bound!r}, so neither a variance nor "
            f"a likelihood can be formed from these counts"
        )
    return values


def _poisson_loglik(values: _Vector, mean: float) -> float:
    """Return the Poisson log likelihood of the counts at a given mean.

    Notes
    -----
    The mean passed here is the sample mean of whole non-negative counts, which
    is zero only when every count is zero. A Poisson law of mean zero gives
    those counts probability one, so their log likelihood is zero.
    """
    if mean <= 0.0:
        return 0.0
    return float(np.sum(values * math.log(mean) - mean - special.gammaln(values + 1.0)))


def _boundary_p_value(statistic: float) -> float:
    """Return the p value of a likelihood ratio statistic tested on a boundary.

    Under a null that puts the free parameter on the edge of its range, the
    statistic follows half a chi square with one degree of freedom and half a
    point mass at zero. The tail above a positive statistic is therefore half
    the chi square tail, and the mass at or above zero is all of it.
    """
    if statistic <= 0.0:
        return 1.0
    return 0.5 * float(stats.chi2.sf(statistic, 1))


def _bounded_log_parameter(value: float, limit: float) -> float:
    """Return a log parameter the NB2 likelihood can be evaluated at.

    See :data:`_LOG_DISPERSION_LIMIT` and :data:`_LOG_MEAN_LIMIT` for why the two
    bounds are there. The likelihood and its score both read the bounded value
    wherever the free parameter appears, so they describe the same model at a
    bound, and the likelihood is flat beyond it: a line search that reaches one
    turns back rather than meeting a division by an underflowed dispersion or an
    exponential of a log mean float64 has no number for.
    """
    return min(max(value, -limit), limit)


def _nb_negative_loglik(parameters: _Vector, values: _Vector) -> float:
    """Return the negative NB2 log likelihood at (log mean, log dispersion).

    Written with :func:`math.log1p`, as :func:`msrelapse.stats._nb_negative_loglik`
    is, so that the terms stay accurate as the dispersion approaches zero and the
    curvature read off them near the boundary means something.
    """
    log_mean = _bounded_log_parameter(float(parameters[_LOG_MEAN]), _LOG_MEAN_LIMIT)
    mean = math.exp(log_mean)
    log_dispersion = _bounded_log_parameter(
        float(parameters[_LOG_DISPERSION]), _LOG_DISPERSION_LIMIT
    )
    dispersion = math.exp(log_dispersion)
    size = 1.0 / dispersion
    terms = (
        special.gammaln(values + size)
        - special.gammaln(size)
        - special.gammaln(values + 1.0)
        + values * (log_dispersion + log_mean)
        - (values + size) * math.log1p(dispersion * mean)
    )
    return -float(np.sum(terms))


def _nb_negative_score(parameters: _Vector, values: _Vector) -> _Vector:
    """Return the gradient of :func:`_nb_negative_loglik`, computed analytically."""
    mean = math.exp(_bounded_log_parameter(float(parameters[_LOG_MEAN]), _LOG_MEAN_LIMIT))
    dispersion = math.exp(
        _bounded_log_parameter(float(parameters[_LOG_DISPERSION]), _LOG_DISPERSION_LIMIT)
    )
    size = 1.0 / dispersion
    residual = (values - mean) / (1.0 + dispersion * mean)
    by_log_mean = float(np.sum(residual))
    by_dispersion = float(
        np.sum(
            (special.digamma(size) - special.digamma(values + size) + math.log1p(dispersion * mean))
            / dispersion**2
            + residual / dispersion
        )
    )
    return -np.array([by_log_mean, dispersion * by_dispersion], dtype=np.float64)


def _numerical_hessian(parameters: _Vector, values: _Vector) -> _Vector:
    """Return the Hessian of :func:`_nb_negative_loglik` by central differences.

    Differencing the analytic score rather than the log likelihood itself keeps
    the truncation error at the square of the step while halving the number of
    evaluations, which is what a standard error near the boundary needs. There
    the log likelihood is a difference of gamma logarithms thousands of times
    larger than itself, and differencing it twice buries the curvature in that
    cancellation. This follows :func:`msrelapse.stats._numerical_hessian`, which
    differences the score of the same likelihood for the same reason.
    """
    size = parameters.size
    hessian = np.empty((size, size), dtype=np.float64)
    for index in range(size):
        step = _HESSIAN_STEP * max(1.0, abs(float(parameters[index])))
        forward = parameters.copy()
        forward[index] += step
        backward = parameters.copy()
        backward[index] -= step
        hessian[:, index] = (
            _nb_negative_score(forward, values) - _nb_negative_score(backward, values)
        ) / (2.0 * step)
    return (hessian + hessian.T) / 2.0


def _log_dispersion_standard_error(parameters: _Vector, values: _Vector) -> float | None:
    """Return the standard error of the fitted log dispersion, or None if there is none.

    Parameters
    ----------
    parameters : numpy.ndarray
        The fitted (log mean, log dispersion), the point the observed
        information is read at.
    values : numpy.ndarray
        The counts that were fitted.

    Returns
    -------
    float or None
        The square root of the log dispersion entry of the inverse observed
        information, or None when that information is singular or leaves the log
        dispersion with a variance that is not positive and finite.

    Notes
    -----
    None is not a failure to report: it says the likelihood has no curvature in
    the dispersion that float64 can resolve, so the data do not identify a
    dispersion at all and the Poisson fit is the answer. The caller returns that
    fit, in the manner of :func:`msrelapse.stats._fit_negative_binomial`, rather
    than a Wald interval built on a curvature that is not there.
    """
    hessian = _numerical_hessian(parameters, values)
    try:
        covariance = np.linalg.inv(hessian)
    except np.linalg.LinAlgError:
        # A singular observed information is one of the two ways the dispersion
        # can fail to be identified, and it is reported as such rather than
        # raised: the two are the same answer to the caller.
        return None
    variance = float(covariance[_LOG_DISPERSION, _LOG_DISPERSION])
    if not (math.isfinite(variance) and variance > 0.0):
        return None
    return math.sqrt(variance)


def _naive_mean(durations: pd.DataFrame, state: int) -> float:
    """Return the mean recorded duration of one state, censoring ignored.

    Raises
    ------
    ValueError
        If the frame holds no run of that state.
    """
    values = durations.loc[durations["state"] == state, "duration_w"]
    if values.empty:
        raise ValueError(
            f"the frame holds no run in state {state:+d}, so equation (7) has no mean "
            f"duration to take the logarithm of"
        )
    return float(values.mean())


def _patient_barrier_ratio(group: pd.DataFrame) -> float:
    """Return the barrier ratio of one patient, or NaN where it is undefined."""
    health = group.loc[group["state"] == _HEALTH, "duration_w"]
    no_health = group.loc[group["state"] == _NO_HEALTH, "duration_w"]
    if health.empty or no_health.empty:
        return math.nan
    tau_health = float(health.mean())
    tau_no_health = float(no_health.mean())
    if tau_health <= 1.0 or tau_no_health <= 1.0:
        return math.nan
    return barrier_ratio_from_durations(tau_health, tau_no_health)
