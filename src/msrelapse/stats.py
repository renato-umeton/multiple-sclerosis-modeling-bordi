"""Trial statistics on relapse event tables.

A relapse record is summarised in a clinical trial by the annualised relapse
rate, the number of relapses a patient has per year of follow up, and two arms
are compared by the ratio of their rates. This module computes both from an
events table, with the interval estimates the trial literature uses: the exact
Poisson interval of Garwood, a negative binomial interval that widens it when
patients differ from one another, and a patient level bootstrap. The rate ratio
is fitted by a log linear model with the follow up as an offset, as a Poisson
regression or as an NB2 negative binomial regression.

The module also wraps the relapse free curve of
[`msrelapse.renewal`][msrelapse.renewal] and gives a negative binomial sample
size formula, so that a reader can see what a cohort of the size of the paper's
can and cannot resolve. Nothing here is specific to the double well of
[`msrelapse.model`][msrelapse.model]: these are the statistics one would report
for any relapsing-remitting record, and they are what a simulated cohort has to
reproduce before the model is taken seriously. Numbers reported by the paper
are never written here; import ``PAPER`` from
[`msrelapse._params`][msrelapse._params] instead.

Both result types, [`ARRResult`][msrelapse.stats.ARRResult] and
[`RateRatioResult`][msrelapse.stats.RateRatioResult], carry the citation of the
article on a ``citation`` property and repeat it once in their repr, so that a
number copied out of a session names its source.

References
----------
I. Bordi, R. Umeton, V. A. G. Ricigliano, et al., "A mechanistic, stochastic
model helps understand multiple sclerosis course and pathogenesis,"
International Journal of Genomics, 2013, doi 10.1155/2013/910321.

F. Garwood, "Fiducial limits for the Poisson distribution," Biometrika, vol.
28, no. 3-4, pp. 437-442, 1936, doi 10.1093/biomet/28.3-4.437.

A. C. Cameron and P. K. Trivedi, Regression Analysis of Count Data, 2nd ed.,
Cambridge University Press, 2013.

H. Zhu and H. Lakkis, "Sample size calculation for comparing two negative
binomial rates," Statistics in Medicine, vol. 33, no. 3, pp. 376-387, 2014,
doi 10.1002/sim.5947.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final, Literal

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy import optimize, special, stats

from msrelapse import _citation
from msrelapse.io import EVENTS_COLUMNS, validate
from msrelapse.renewal import relapse_counts, relapse_free

__all__ = [
    "WEEKS_PER_YEAR",
    "ARRResult",
    "CIMethod",
    "ModelName",
    "RateRatioResult",
    "Seed",
    "arr",
    "compare_arr",
    "patient_followup",
    "relapse_free_curve",
    "sample_size_arr",
]

WEEKS_PER_YEAR: Final = 365.25 / 7.0
"""Weeks in a Julian year of 365.25 days, the divisor from weeks to years."""

CIMethod = Literal["poisson_exact", "nb", "bootstrap"]
ModelName = Literal["nb", "poisson"]
Seed = np.random.Generator | int | None

_CI_METHODS: Final = ("poisson_exact", "nb", "bootstrap")
_MODELS: Final = ("nb", "poisson")

_NEWTON_STEPS: Final = 100
_NEWTON_TOLERANCE: Final = 1e-12
"""Step the Poisson fit calls converged, which its exact Hessian can deliver."""

_POLISH_TOLERANCE: Final = 1e-9
"""Relative step the refinement of the negative binomial fit calls converged.

That refinement solves against a Hessian differenced from the analytic score
with a step of 1e-5, so the Hessian carries a relative error of order 1e-10:
the truncation error of a central difference, the square of the step, plus the
rounding error of the difference divided by the step. A Newton step cannot
shrink past that floor, and on a cohort whose likelihood is weakly curved it
stalls there and oscillates. Asking it for the 1e-12 an exact Hessian reaches
therefore reported a settled fit as a failure and spent every iteration of the
budget getting there.
"""

_BFGS_GRADIENT_TOLERANCE: Final = 1e-10
_BFGS_STEPS: Final = 1000
_HESSIAN_STEP: Final = 1e-5
_DISPERSION_FLOOR: Final = 1e-8
"""Numerical zero of the moment dispersion, below which the counts are a Poisson.

The maximum likelihood dispersion is zero exactly when the moment one is, so a
moment estimate no larger than this settles the question before the search
starts. Where the search itself stops is a different question and is judged
against ``_DISPERSION_COLLAPSE``, which is wider for the reason given there.
"""

_DISPERSION_COLLAPSE: Final = 1e-6
"""At or below this dispersion a fitted negative binomial is read as a Poisson.

The search stops where the likelihood stops moving, and on a cohort whose
dispersion is walking to nothing that is not the same place on every platform:
the same fit has been seen to arrive at 5e-9 on one and to stall at 2e-8 on
another, one either side of the floor above. The threshold is set fifty times
above that stall so that both read as the collapse they are. Nothing is lost by
it: a dispersion of 1e-6 inflates the variance of a count of mean m
from m to m (1 + 1e-6 m), a part in ten thousand even at a hundred relapses per
patient, which is far below the sampling noise of a variance measured on any
cohort.
"""

_LOGLIK_GAIN: Final = 1e-6
"""Log likelihood per patient a fitted dispersion has to buy over the Poisson fit.

The NB2 likelihood tends to the Poisson likelihood as the dispersion tends to
zero, so a fit with nothing to show against the Poisson maximum has collapsed
onto it, whatever the dispersion it stopped at reads.

The bound is counted per patient because the quantity it is compared with is a
difference of two log likelihoods, and the error of that difference grows with
the cohort. The negative binomial term gammaln(y + 1 / a) - gammaln(1 / a)
cancels two values of size (1 / a) log(1 / a), so the difference carries an
error of order the number of patients times the rounding of float64 times
(1 / a) log(1 / a), which is set by the dispersion and the size of the cohort
rather than by the size of either likelihood. Measured on 600 patients it is
3e-5 at a dispersion of 1e-8, and it falls as the dispersion rises, so it is a
few times 1e-7 at the dispersion of 1e-6 this rule is first consulted at. A
bound flat in the size of the cohort would therefore be read off rounding on a
cohort of a few thousand, while a bound per patient stays above the error on a
cohort of any size.

What the rule discards is a dispersion no cohort can resolve. Reading the
likelihood as a quadratic around its maximum, a gain of this size over n
patients puts the dispersion sqrt(2 n 1e-6) of a standard error from zero,
three hundredths of one on a cohort of 600. The smallest gain a fitted
dispersion of this package has been measured to buy is 1e-3, on the 60 patients
of the smallest cohort of the test suite, sixteen times the bound there.

In dispersion units the same quadratic puts the reach of the rule at about 2e-3
divided by the root mean square fitted count, whatever the size of the cohort:
the curvature of the likelihood in the dispersion grows with the cohort exactly
as the bound does, so n cancels and only the size of a count is left. That is
under two parts in a thousand on a cohort like the one the test suite fits,
where a patient has about one relapse over the record, and it means a fitted
dispersion is discarded some way above ``_DISPERSION_COLLAPSE`` rather than only
at it. Measured over 600 simulated Poisson cohorts of 200 patients, the largest
searched dispersion discarded was 1.7e-3 and the smallest kept 2.3e-3, against a
reach of 1.8e-3 predicted by that formula on those counts. The standard error
reading holds while the cohort stays below a few hundred thousand patients:
sqrt(2 n 1e-6) reaches one standard error at about 500000 of them and grows as
the square root of n after that.
"""

_SINGULAR_DISPERSION: Final = 1e-3
"""Dispersion below which a curvature that carries no interval reads as a collapse.

A Hessian with no curvature left in the dispersion direction is what a
dispersion on its way to zero leaves behind, and no Wald interval can be read
from it. The bound is three orders of magnitude above ``_DISPERSION_COLLAPSE``,
so that a fit stopping in the decades between the two is read as a collapse
where the curvature says so and as a fit where it does not.
"""

_LOG_DISPERSION_LIMIT: Final = 50.0
"""Bound on the log dispersion the negative binomial likelihood is read at.

The exponential of the free parameter underflows to zero below about -745 and
overflows above 709, and the likelihood divides by it, so a line search that
probed that far would meet a bare arithmetic error rather than a value telling
it to turn back. Between exp(-50) and exp(50) lies every dispersion a count
model means anything at, and the likelihood is flat beyond the bound, so a
search that reaches it stops rather than raises.
"""

_Vector = npt.NDArray[np.float64]
"""One value per patient, or one per parameter of a fitted model."""

_Matrix = npt.NDArray[np.float64]
"""A design matrix, one row per patient, or the Hessian of a fitted model."""


@dataclass(frozen=True, repr=False)
class ARRResult:
    """An annualised relapse rate with its interval estimate.

    Attributes
    ----------
    n_patients : int
        Number of patients the rate is computed from.
    n_relapses : int
        Number of relapse onsets across those patients.
    patient_years : float
        Total follow up, in patient-years.
    arr : float
        Annualised relapse rate, `n_relapses` divided by `patient_years`.
    ci_low : float
        Lower end of the confidence interval, in relapses per year.
    ci_high : float
        Upper end of the confidence interval, in relapses per year.
    ci_level : float
        Nominal coverage of the interval, for example 0.95.
    ci_method : str
        Which interval was computed: 'poisson_exact', 'nb' or 'bootstrap'.
    """

    n_patients: int
    n_relapses: int
    patient_years: float
    arr: float
    ci_low: float
    ci_high: float
    ci_level: float
    ci_method: str

    @property
    def citation(self) -> str:
        """Return the citation of the paper this package reproduces.

        Returns
        -------
        str
            The short citation of ``msrelapse._citation``.
        """
        return _citation.short_citation()

    def __repr__(self) -> str:
        """Return the rate, its interval and the citation, on one line.

        Returns
        -------
        str
            The cohort it was measured on, the rate with its interval and the
            short citation.
        """
        return (
            f"{type(self).__name__}(n_patients={self.n_patients}, "
            f"n_relapses={self.n_relapses}, patient_years={self.patient_years:.6g}, "
            f"arr={self.arr:.6g} per year, {self.ci_level:.0%} {self.ci_method} CI "
            f"{self.ci_low:.6g} to {self.ci_high:.6g}; {self.citation})"
        )


@dataclass(frozen=True, repr=False)
class RateRatioResult:
    """The relapse rate of one arm relative to another.

    Attributes
    ----------
    rate_ratio : float
        Fitted rate of arm b divided by the fitted rate of arm a.
    ci_low : float
        Lower end of the Wald confidence interval on the rate ratio.
    ci_high : float
        Upper end of the Wald confidence interval on the rate ratio.
    p_value : float
        Two sided Wald p value for the null hypothesis of an equal rate.
    arr_a : float
        Fitted annualised relapse rate of arm a.
    arr_b : float
        Fitted annualised relapse rate of arm b.
    dispersion : float
        Negative binomial dispersion, so that the variance of a count is
        mean + dispersion mean squared. Zero for a Poisson fit.
    model : str
        Which model was fitted, 'nb' or 'poisson'.
    converged : bool
        Whether the optimiser reported a converged fit.
    """

    rate_ratio: float
    ci_low: float
    ci_high: float
    p_value: float
    arr_a: float
    arr_b: float
    dispersion: float
    model: str
    converged: bool

    @property
    def citation(self) -> str:
        """Return the citation of the paper this package reproduces.

        Returns
        -------
        str
            The short citation of ``msrelapse._citation``.
        """
        return _citation.short_citation()

    def __repr__(self) -> str:
        """Return the rate ratio, the fit behind it and the citation, on one line.

        Returns
        -------
        str
            The model, the rate ratio with its interval and p value, the fitted
            rate of each arm, the dispersion and the short citation.
        """
        return (
            f"{type(self).__name__}(model={self.model!r}, rate_ratio={self.rate_ratio:.6g}, "
            f"CI {self.ci_low:.6g} to {self.ci_high:.6g}, p_value={self.p_value:.6g}, "
            f"arr_a={self.arr_a:.6g}, arr_b={self.arr_b:.6g}, "
            f"dispersion={self.dispersion:.6g}, converged={self.converged}; {self.citation})"
        )


@dataclass(frozen=True)
class _Fit:
    """One fitted log linear count model, with everything the caller reports.

    A negative binomial fit whose dispersion has collapsed is reported as the
    Poisson fit it was started from, the same object, so it carries a dispersion
    of 0 and the Poisson standard errors, which are the limit of the negative
    binomial ones. ``_fit_negative_binomial`` says which readings of a collapse
    it checks and why.
    """

    coefficients: _Vector
    standard_errors: _Vector
    dispersion: float
    converged: bool


def patient_followup(events: pd.DataFrame) -> pd.Series[float]:
    """Return the length of each patient's follow up window, in weeks.

    Parameters
    ----------
    events : pandas.DataFrame
        An events table, as described by the schema of
        [`msrelapse.io`][msrelapse.io].

    Returns
    -------
    pandas.Series
        Follow up in weeks, named 'followup_w', of dtype float64, indexed by
        patient_id in ascending order. Every patient of the table appears once,
        so a table with no rows gives an empty series. That is the one place
        this function and [`arr`][msrelapse.stats.arr] part company: a follow
        up of no patients is still a follow up, while a rate of no patients is
        nothing at all and [`arr`][msrelapse.stats.arr] refuses it.

    Raises
    ------
    ValueError
        If `events` lacks a column of the schema or fails the schema validator.

    Examples
    --------
    >>> import pandas as pd
    >>> events = pd.DataFrame(
    ...     {
    ...         "patient_id": ["p0001"],
    ...         "followup_start": [0.0],
    ...         "followup_end": [52.0],
    ...         "relapse_onset": [3.0],
    ...         "relapse_end": [5.0],
    ...     }
    ... )
    >>> float(patient_followup(events)["p0001"])
    52.0
    """
    _validate_events(events)
    return _followup_weeks(events)


# The six parameters are the estimate, the exposure it is divided by, and the
# four settings of the interval. Grouping the interval settings into an options
# object would hide the plain call for the two thirds of callers who take the
# default interval.
def arr(  # noqa: PLR0917
    events: pd.DataFrame,
    followup: pd.Series[float] | None = None,
    ci: CIMethod = "poisson_exact",
    level: float = 0.95,
    n_boot: int = 1000,
    rng: Seed = None,
) -> ARRResult:
    """Return the annualised relapse rate of a cohort with a confidence interval.

    The rate is the total number of relapse onsets divided by the total follow
    up in patient-years, which is the estimate a trial reports. Three intervals
    are available, and they answer different questions: the exact Poisson
    interval assumes every relapse is an independent event at a common rate,
    while the other two allow the rate to vary from patient to patient and are
    accordingly wider on a real cohort.

    Parameters
    ----------
    events : pandas.DataFrame
        An events table, as described by the schema of
        [`msrelapse.io`][msrelapse.io].
    followup : pandas.Series, optional
        Follow up in weeks per patient, indexed by patient_id, replacing the
        window recorded in `events` as the denominator of the rate. It does not
        restrict which relapses are counted: to analyse a window shorter than
        the record, trim `events` first, so that the numerator and the
        denominator describe the same window. It must hold exactly the patients
        of `events`: one that is missing and one that `events` does not hold are
        both refused, because either says the follow up belongs to some other
        cohort.
    ci : {'poisson_exact', 'nb', 'bootstrap'}, optional
        Which interval to compute. See the Notes.
    level : float, optional
        Nominal coverage of the interval, strictly between 0 and 1.
    n_boot : int, optional
        Number of bootstrap resamples, used only by the bootstrap interval.
    rng : numpy.random.Generator or int or None, optional
        Generator to draw the bootstrap from, or a seed for
        ``numpy.random.default_rng``.

    Returns
    -------
    ARRResult
        The rate, the interval, and the counts they come from.

    Raises
    ------
    ValueError
        If `events` fails the schema validator or holds no patient at all, if
        `followup` does not hold exactly the patients of `events` or is not
        positive and finite, if `ci` or `level` is out of range, if `n_boot` is
        below one, if the negative binomial interval is asked for a cohort with
        no relapse, or if either interval that allows for spread is asked for a
        single patient.

    Notes
    -----
    ``'poisson_exact'`` is the Garwood interval, the exact interval for the
    mean of a Poisson count: chi2((1 - `level`) / 2, 2 k) / 2 and
    chi2((1 + `level`) / 2, 2 k + 2) / 2 divided by the patient-years, with a
    lower end of 0 when no relapse was seen.

    ``'nb'`` is a Wald interval on the log rate whose variance is (1 + a m) /
    k, with m the mean count per patient, k the total count and a the method of
    moments dispersion max(0, (var(c) - mean(c)) / mean(c)^2) of the per
    patient counts c, where var is the unbiased sample variance, the one
    divided by n - 1, as the moment estimator behind a Wald interval
    conventionally takes.
    [`msrelapse.fit.fit_nb_counts`][msrelapse.fit.fit_nb_counts] weighs those
    same two quantities against one another with the population variance
    instead, divided by n, and its own Notes say why that is the right
    comparison there: it screens on whether the maximum likelihood dispersion
    is positive at all, and that turns exactly where the population variance
    meets the mean. The two variances differ by the factor n / (n - 1), so the
    formula above and that screen do not answer with the same number on the
    same counts, and neither of them is a slip. The approximation this interval
    makes is that patients are followed for comparable lengths of time: the
    moment estimator reads all the spread of the counts as a spread of rates,
    so with widely unequal follow up it also reads the spread of the exposure
    as over dispersion and the interval comes out too wide. Prefer the
    bootstrap in that case.

    ``'bootstrap'`` resamples patients with replacement, recomputes the rate
    from the resampled counts and their resampled follow up, and takes the
    percentile interval. It makes no assumption about the count distribution
    and it keeps each patient's count and exposure together, so unequal follow
    up costs it nothing. It needs at least two patients, as the negative
    binomial interval does: every resample of a single patient is that patient,
    so the interval would collapse onto the point estimate and read as an
    estimate of a spread that was never measured. It holds all the resamples at
    once, so it allocates on the order of `n_boot` times the number of patients
    values: a cohort of many thousands should be given a smaller `n_boot`.

    Examples
    --------
    >>> import pandas as pd
    >>> events = pd.DataFrame(
    ...     {
    ...         "patient_id": ["p0001", "p0001"],
    ...         "followup_start": [0.0, 0.0],
    ...         "followup_end": [104.0, 104.0],
    ...         "relapse_onset": [3.0, 60.0],
    ...         "relapse_end": [5.0, 62.0],
    ...     }
    ... )
    >>> round(arr(events).arr, 3)
    1.003
    """
    _check_choice(ci, _CI_METHODS, "ci")
    _check_unit_interval(level, "level")
    if n_boot < 1:
        raise ValueError(f"n_boot must be positive, got {n_boot!r}")
    counts = relapse_counts(events)
    # A table with no row has no patient block for the schema rules to reject,
    # so the empty cohort is caught here rather than by the validator.
    if counts.empty:
        raise ValueError("events has no patients, so there is no rate to report")
    # relapse_counts has already run the schema validator over this table, so
    # the follow up windows are read without a second pass over it.
    given = _followup_weeks(events) if followup is None else followup
    weeks = _aligned_followup(given, counts.index)
    years = weeks / WEEKS_PER_YEAR
    per_patient = np.asarray(counts.to_numpy(), dtype=np.int64)
    patient_years = float(years.sum())
    n_relapses = int(per_patient.sum())
    rate = n_relapses / patient_years

    if ci == "poisson_exact":
        low, high = _garwood_interval(n_relapses, patient_years, level)
    elif ci == "nb":
        low, high = _negative_binomial_interval(per_patient, rate, level)
    else:
        low, high = _bootstrap_interval(per_patient, years, level, n_boot, rng)

    return ARRResult(
        n_patients=int(per_patient.size),
        n_relapses=n_relapses,
        patient_years=patient_years,
        arr=rate,
        ci_low=low,
        ci_high=high,
        ci_level=level,
        ci_method=ci,
    )


# The six parameters are the two arms with their follow up, plus the model and
# the level. The arms come in pairs and cannot be collapsed into one argument
# without hiding which table belongs to which arm.
def compare_arr(  # noqa: PLR0917
    events_a: pd.DataFrame,
    followup_a: pd.Series[float] | None,
    events_b: pd.DataFrame,
    followup_b: pd.Series[float] | None,
    model: ModelName = "nb",
    level: float = 0.95,
) -> RateRatioResult:
    """Compare the relapse rate of two arms with a log linear count model.

    Per patient counts are regressed on an intercept and an indicator of arm b,
    with the logarithm of the follow up in years as an offset, so that the
    coefficient of the indicator is the logarithm of the rate ratio of arm b to
    arm a. The Poisson model assumes the counts of an arm have a common rate;
    the negative binomial model lets the rate vary between patients and is the
    model relapse trials report, because its interval does not narrow when the
    counts are more spread out than a Poisson allows.

    Parameters
    ----------
    events_a : pandas.DataFrame
        Events table of the reference arm.
    followup_a : pandas.Series or None
        Follow up in weeks per patient of arm a, indexed by patient_id, or None
        to take the window recorded in `events_a`. As in
        [`arr`][msrelapse.stats.arr], it is the exposure only and does not
        restrict which relapses are counted: to compare a window shorter than
        the record, trim the events tables first, and it must hold exactly the
        patients of its own arm, so that the follow up of the other arm is
        refused rather than silently aligned.
    events_b : pandas.DataFrame
        Events table of the arm compared with the reference.
    followup_b : pandas.Series or None
        Follow up in weeks per patient of arm b, or None as for `followup_a`.
    model : {'nb', 'poisson'}, optional
        Which count model to fit.
    level : float, optional
        Nominal coverage of the Wald interval, strictly between 0 and 1.

    Returns
    -------
    RateRatioResult
        The rate ratio with its interval and p value, the fitted rate of each
        arm, and the dispersion of the fitted model.

    Raises
    ------
    ValueError
        If either table fails the schema validator, if a follow up does not
        hold exactly the patients of its arm, if `model` or `level` is out of
        range, or if an arm has no relapse at all, where no finite log linear
        fit exists.
    numpy.linalg.LinAlgError
        If the Hessian of the fitted model is singular at the optimum, or
        inverts to a variance that is not finite and positive, so that no
        standard error exists. Both coefficients are identified as soon as each
        arm has a relapse, so this needs a degenerate cohort; it is raised
        rather than caught, because a fit whose curvature carries no interval
        is not reported as one with a nan interval. The one exception is a
        negative binomial fit whose dispersion has collapsed, where the missing
        curvature is that collapse and the Poisson fit is reported instead; see
        the Notes.

    Notes
    -----
    The Poisson model is fitted by Newton iteration on the exact score, which
    is the maximum likelihood fit to machine precision. The negative binomial
    model is the NB2 parameterisation, in which the variance of a count is
    mean + dispersion mean squared; its three parameters, the two coefficients
    and the logarithm of the dispersion, are fitted by maximising the joint log
    likelihood from the Poisson fit and a method of moments dispersion. A
    quasi-Newton search gets close and Newton steps on the analytic score
    finish, which matters because the dispersion is weakly identified and the
    search stops on a gradient the likelihood cannot resolve. The standard
    errors come from the inverse of the Hessian at the optimum, computed by
    central differences of the analytic score.

    `converged` reports the Newton refinement and not the quasi-Newton search
    that precedes it. The search is a starting point: at the gradient tolerance
    asked of it, it reports a loss of precision rather than success even when
    it has arrived, so its own flag would always read False. The refinement is
    the stricter test of the two, because it reports success only when a Newton
    step on the analytic score has become a part in a thousand million of the
    parameters, which a search that stopped short of the optimum does not
    produce. That is the precision a Hessian built by differencing the score
    can resolve and no more, so a cohort whose likelihood is too weakly curved
    for even that reads as not converged, which on a small cohort with a
    dispersion near zero is the honest answer rather than a fault.

    The maximum likelihood dispersion is zero exactly when the moment estimator
    is, because both have the sign of the score of the likelihood in the
    dispersion at zero. Data no more spread out than a Poisson therefore
    collapse the negative binomial onto the Poisson: the result then reports
    the model as 'nb' with a dispersion of 0 and the Poisson standard errors,
    which are the limit of the negative binomial ones, rather than a dispersion
    driven to zero by an optimiser that cannot reach it.

    A search that runs and then arrives at nothing is reported the same way,
    and it is read as having arrived at nothing on any of three counts: a
    dispersion at or below 1e-6, where the spread it adds to a count is far
    below the sampling noise of any cohort; a likelihood no better than the
    Poisson one by more than 1e-6 for each patient of the two arms; or a
    curvature that carries no standard errors while the dispersion is below
    1e-3, which is what a likelihood with no curvature left in the dispersion
    direction looks like. Three readings rather than one because the search
    stops where the likelihood stops moving, and that is not the same place on
    every platform, while a dispersion no model can tell from zero is a collapse
    wherever the search stopped. The second reading is counted per patient
    because the difference of the two log likelihoods carries a rounding error
    that grows with the cohort; what it discards is a dispersion a few
    hundredths of a standard error from zero, about 2e-3 divided by the root mean
    square fitted count whatever the size of the cohort, which no cohort below a
    few hundred thousand patients can resolve.

    `arr_a` and `arr_b` are the fitted rates exp(intercept) and
    exp(intercept + coefficient), so that `rate_ratio` is exactly their ratio.
    For the Poisson model these are also the crude rates of the two arms; for
    the negative binomial model they are close to but not exactly the crude
    rates, because the fit weights patients by their fitted variance.

    Examples
    --------
    >>> from msrelapse.renewal import alternating_renewal
    >>> arm_a = alternating_renewal(0.02, 0.25, 156.0, n=40, rng=1)
    >>> arm_b = alternating_renewal(0.01, 0.25, 156.0, n=40, rng=2)
    >>> compare_arr(arm_a, None, arm_b, None, model="poisson").rate_ratio < 1.0
    True
    """
    _check_choice(model, _MODELS, "model")
    _check_unit_interval(level, "level")
    counts_a, years_a = _counts_and_years(events_a, followup_a, "a")
    counts_b, years_b = _counts_and_years(events_b, followup_b, "b")
    counts = np.concatenate([counts_a, counts_b])
    exposure = np.concatenate([years_a, years_b])
    arm = np.concatenate([np.zeros(counts_a.size), np.ones(counts_b.size)])
    design = np.column_stack([np.ones(counts.size), arm])

    poisson = _fit_poisson(design, counts, exposure)
    if model == "poisson":
        return _rate_ratio_result(poisson, model, level)
    return _rate_ratio_result(
        _fit_negative_binomial(design, counts, exposure, poisson), model, level
    )


def relapse_free_curve(
    T_grid: npt.ArrayLike,
    lam: float | None = None,
    k: float | None = None,
    theta: float | None = None,
) -> npt.NDArray[np.float64]:
    """Return the probability of staying free of relapse over a grid of windows.

    This is the array valued face of
    [`msrelapse.renewal.relapse_free`][msrelapse.renewal.relapse_free]: the
    same curve, always returned as an array, so that it can be plotted or
    compared with an observed Kaplan-Meier curve without a further conversion.

    Parameters
    ----------
    T_grid : array_like
        Window lengths in weeks, one per point of the curve. Must not be
        negative.
    lam : float, optional
        Single onset rate per week, shared by every patient. Give this or the
        pair `k` and `theta`.
    k : float, optional
        Shape of the gamma distribution of onset rates.
    theta : float, optional
        Scale of the gamma distribution of onset rates, in onsets per week.

    Returns
    -------
    numpy.ndarray
        The probability at each window, of dtype float64 and with the shape of
        `T_grid`.

    Raises
    ------
    ValueError
        If neither or both parameterisations are supplied, if the gamma
        parameterisation is incomplete, or if any value is out of range.

    Examples
    --------
    >>> [round(float(value), 3) for value in relapse_free_curve([0.0, 100.0], lam=0.01)]
    [1.0, 0.368]
    """
    grid = np.asarray(T_grid, dtype=np.float64)
    curve = relapse_free(grid, lam=lam, k=k, theta=theta)
    return np.asarray(curve, dtype=np.float64)


# The seven parameters are the four quantities of the alternative hypothesis
# and the three settings of the test; every one of them appears in the formula.
def sample_size_arr(  # noqa: PLR0917
    arr_control: float,
    rate_ratio: float,
    dispersion: float,
    followup_years: float,
    power: float = 0.8,
    alpha: float = 0.05,
    allocation: float = 1.0,
) -> int:
    """Return an illustrative number of control patients for a two arm trial.

    The formula is the negative binomial sample size of Zhu and Lakkis with the
    variance evaluated under the alternative hypothesis: with rates r0 and
    r1 = r0 `rate_ratio`, a follow up of t years and a dispersion a, the number
    of control patients is

        (z(1 - alpha / 2) + z(power))^2 [(1 / (t r0) + a) + (1 / (t r1) + a) /
        allocation] / log(rate_ratio)^2,

    rounded up. The treated arm holds `allocation` times as many patients.

    This is an illustration of what the relapse rate, the dispersion and the
    length of follow up do to the size of a trial, and of how much of the
    burden comes from the between patient spread rather than the counting
    noise. It is not a trial design tool: it assumes a single fixed analysis of
    complete follow up, no dropout, no covariate adjustment and no interim
    look, and it does not replace a protocol statistician.

    Parameters
    ----------
    arr_control : float
        Annualised relapse rate of the control arm. Must be positive.
    rate_ratio : float
        Rate of the treated arm divided by the rate of the control arm, under
        the alternative hypothesis. Must be positive and not 1.
    dispersion : float
        Negative binomial dispersion of the per patient counts, so that the
        variance is mean + `dispersion` mean squared. Must not be negative;
        zero is the Poisson case.
    followup_years : float
        Length of follow up per patient, in years. Must be positive.
    power : float, optional
        Probability of rejecting the null hypothesis under the alternative,
        strictly between 0 and 1.
    alpha : float, optional
        Two sided significance level, strictly between 0 and 1.
    allocation : float, optional
        Patients in the treated arm per patient in the control arm. Must be
        positive.

    Returns
    -------
    int
        Number of patients in the control arm, at least 1.

    Raises
    ------
    ValueError
        If `rate_ratio` is 1, where no sample size answers the question, if
        `power` is at or below `alpha` / 2, where the two normal quantiles of
        the formula cancel and below which it runs backwards, or if any input
        is out of range.

    Examples
    --------
    >>> sample_size_arr(0.5, 0.7, 0.8, 2.0)
    249
    """
    if arr_control <= 0.0:
        raise ValueError(f"arr_control must be positive, got {arr_control!r}")
    if rate_ratio <= 0.0:
        raise ValueError(f"rate_ratio must be positive, got {rate_ratio!r}")
    if rate_ratio == 1.0:
        raise ValueError("rate_ratio must not be 1: equal rates need an infinite trial")
    if dispersion < 0.0:
        raise ValueError(f"dispersion must not be negative, got {dispersion!r}")
    if followup_years <= 0.0:
        raise ValueError(f"followup_years must be positive, got {followup_years!r}")
    if allocation <= 0.0:
        raise ValueError(f"allocation must be positive, got {allocation!r}")
    _check_unit_interval(power, "power")
    _check_unit_interval(alpha, "alpha")

    arr_treated = arr_control * rate_ratio
    z_alpha = float(stats.norm.ppf(1.0 - alpha / 2.0))
    z_power = float(stats.norm.ppf(power))
    # The formula squares the sum of the two quantiles, and that sum falls
    # through zero as the power falls through alpha / 2: below there squaring
    # sends the answer back up, so an ever weaker request would be met with an
    # ever larger trial. A power no better than the share of alpha a two sided
    # test spends on one tail is not a request a sample size answers, and it is
    # refused rather than answered with a number that runs the wrong way.
    if z_alpha + z_power <= 0.0:
        raise ValueError(
            f"power must be above alpha / 2, got power={power!r} with alpha={alpha!r}: "
            "at and below that point the two normal quantiles cancel and no sample size "
            "answers the question"
        )
    variance = (1.0 / (followup_years * arr_control) + dispersion) + (
        1.0 / (followup_years * arr_treated) + dispersion
    ) / allocation
    n_control = (z_alpha + z_power) ** 2 * variance / math.log(rate_ratio) ** 2
    # Every term of the formula is positive, so rounding a fraction of a
    # patient up already answers with at least one patient. The floor is the
    # postcondition of the docstring written down, for the one case the
    # arithmetic does not cover: a request extreme enough that the whole
    # expression underflows to zero, such as a rate ratio so far from one that
    # the square of its logarithm dwarfs everything above it.
    return max(1, math.ceil(n_control))


def _validate_events(events: pd.DataFrame) -> None:
    """Check that `events` follows the events schema.

    Only the five schema columns are handed to the validator, because it
    rejects any column it does not know and a caller may well be summarising a
    table that carries extra columns, such as the dated export of
    [`msrelapse.io.weekly_to_events`][msrelapse.io.weekly_to_events]. Every
    other rule of the schema still runs. The columns are checked first so that
    a missing one is reported rather than raising a bare lookup error.

    The same check is spelled out again in
    ``msrelapse.renewal._validate_events``, which that module keeps so that
    it needs nothing beyond numpy and pandas. The two messages are meant to
    stay identical, so a change to the wording here belongs in both.
    """
    missing = [name for name in EVENTS_COLUMNS if name not in events.columns]
    if missing:
        raise ValueError(f"events table is missing the columns {missing}")
    validate(events[list(EVENTS_COLUMNS)], "events")


def _followup_weeks(events: pd.DataFrame) -> pd.Series[float]:
    """Return the window length of each patient of an already validated table.

    This is the body of [`patient_followup`][msrelapse.stats.patient_followup]
    without the schema check, for the callers that have just run the validator
    through
    [`msrelapse.renewal.relapse_counts`][msrelapse.renewal.relapse_counts] and
    would otherwise pay for a second pass over the table.
    """
    windows = (events["followup_end"] - events["followup_start"]).astype(np.float64)
    # The schema gives a patient one window, repeated on each of its rows, so
    # the first row of a patient carries the whole answer.
    followup: pd.Series[float] = windows.groupby(events["patient_id"], sort=True).first()
    return followup.rename("followup_w").rename_axis("patient_id")


def _check_choice(value: str, allowed: tuple[str, ...], name: str) -> None:
    """Check that an argument names one of the choices this module offers."""
    if value not in allowed:
        raise ValueError(f"{name} must be one of {allowed}, got {value!r}")


def _check_unit_interval(value: float, name: str) -> None:
    """Check that a probability lies strictly inside the unit interval."""
    if not 0.0 < value < 1.0:
        raise ValueError(f"{name} must be between 0 and 1, got {value!r}")


def _aligned_followup(followup: pd.Series[float], patients: pd.Index[str]) -> _Vector:
    """Return the follow up of each patient in `patients`, in weeks.

    The index is checked in both directions: a follow up that misses a patient
    and a follow up that carries one the table does not hold are both the follow
    up of some other cohort, so both are reported rather than reindexed away.
    """
    aligned = followup.reindex(patients)
    absent = aligned.isna().to_numpy()
    if absent.any():
        names = [str(name) for name in patients[absent]]
        raise ValueError(
            f"followup has no entry for patient {names[0]}; it covers "
            f"{patients.size - len(names)} of the {patients.size} patients of the table"
        )
    unknown = followup.index.difference(patients)
    if unknown.size > 0:
        raise ValueError(
            f"followup has an entry for {unknown[0]}, which is not a patient of the table; "
            f"{unknown.size} of its {followup.index.size} entries are not patients of it"
        )
    weeks = np.asarray(aligned.to_numpy(), dtype=np.float64)
    offenders = np.flatnonzero(~(np.isfinite(weeks) & (weeks > 0.0)))
    if offenders.size > 0:
        first = int(offenders[0])
        raise ValueError(
            f"followup must be positive and finite; patient {patients[first]} has "
            f"{weeks[first]!r} weeks"
        )
    return weeks


def _counts_and_years(
    events: pd.DataFrame, followup: pd.Series[float] | None, arm: str
) -> tuple[_Vector, _Vector]:
    """Return the per patient counts and follow up years of one arm."""
    counts = relapse_counts(events)
    if int(counts.sum()) == 0:
        raise ValueError(
            f"arm {arm} needs at least one relapse: a log linear model has no finite fit "
            "for an arm whose rate is zero"
        )
    # relapse_counts has already run the schema validator over this table, so
    # the follow up windows are read without a second pass over it.
    given = _followup_weeks(events) if followup is None else followup
    weeks = _aligned_followup(given, counts.index)
    return np.asarray(counts.to_numpy(), dtype=np.float64), weeks / WEEKS_PER_YEAR


def _garwood_interval(n_relapses: int, patient_years: float, level: float) -> tuple[float, float]:
    """Return the exact Poisson interval for a rate, in relapses per year."""
    low = 0.0
    if n_relapses > 0:
        low = float(stats.chi2.ppf((1.0 - level) / 2.0, 2 * n_relapses)) / 2.0 / patient_years
    high = float(stats.chi2.ppf((1.0 + level) / 2.0, 2 * n_relapses + 2)) / 2.0 / patient_years
    return low, high


def _negative_binomial_interval(
    counts: npt.NDArray[np.int64], rate: float, level: float
) -> tuple[float, float]:
    """Return the Wald interval on the log rate that a moment dispersion widens."""
    n_relapses = int(counts.sum())
    if n_relapses == 0:
        raise ValueError(
            "the nb interval needs at least one relapse; use ci='poisson_exact' on a cohort "
            "with none"
        )
    if counts.size < 2:
        raise ValueError("the nb interval needs at least two patients to estimate a dispersion")
    mean = float(counts.mean())
    spread = float(counts.var(ddof=1))
    dispersion = max(0.0, (spread - mean) / mean**2)
    half_width = float(stats.norm.ppf((1.0 + level) / 2.0)) * math.sqrt(
        (1.0 + dispersion * mean) / n_relapses
    )
    return rate * math.exp(-half_width), rate * math.exp(half_width)


def _bootstrap_interval(
    counts: npt.NDArray[np.int64], years: _Vector, level: float, n_boot: int, rng: Seed
) -> tuple[float, float]:
    """Return the percentile interval of the rate over resampled patients."""
    if counts.size < 2:
        raise ValueError(
            "the bootstrap interval needs at least two patients: every resample of one "
            "patient is that patient, so the interval collapses onto the point estimate; "
            "use ci='poisson_exact' on a cohort of one"
        )
    # default_rng returns a generator it is given unaltered, so this accepts a
    # generator and a seed alike.
    generator = np.random.default_rng(rng)
    drawn = generator.integers(0, counts.size, size=(n_boot, counts.size))
    rates = counts[drawn].sum(axis=1) / years[drawn].sum(axis=1)
    low, high = np.percentile(rates, [50.0 * (1.0 - level), 50.0 * (1.0 + level)])
    return float(low), float(high)


def _standard_errors(covariance: _Matrix, model: ModelName) -> _Vector:
    """Return the standard errors an inverted curvature carries.

    Inverting the Hessian is not on its own proof that there is a Wald interval
    to read from it. A curvature that is singular to working precision inverts
    to variances that are not finite, and a point that is not a strict minimum
    of the negative log likelihood to variances that are negative; the square
    root of either is a nan that would travel silently into the rate ratio, its
    interval and its p value while the fit still reported itself converged.
    Both are refused here instead, in the manner of
    ``msrelapse.fit._log_dispersion_standard_error``, which reports the same
    condition to a caller that has a Poisson fit to fall back on. No cohort has
    been found that reaches this, here or in several hundred randomised fits, so
    it stands as insurance against the search and not as a description of any
    data.

    Parameters
    ----------
    covariance : numpy.ndarray
        The inverse of the Hessian of the negative log likelihood at the fit.
    model : {'nb', 'poisson'}
        The model that was fitted, which the message names.

    Returns
    -------
    numpy.ndarray
        The square root of each variance on the diagonal, one per parameter.

    Raises
    ------
    numpy.linalg.LinAlgError
        If a variance on the diagonal is not finite and positive.
    """
    variances = np.asarray(np.diag(covariance), dtype=np.float64)
    if not bool(np.all(np.isfinite(variances) & (variances > 0.0))):
        raise np.linalg.LinAlgError(
            f"the {model} fit has no standard errors: the curvature at its optimum inverts to "
            f"the variances {variances.tolist()}, which are not all finite and positive, so "
            "the fit does not sit at a minimum a Wald interval can be read from"
        )
    return np.sqrt(variances)


def _fitted_mean(design: _Matrix, coefficients: _Vector, exposure: _Vector) -> _Vector:
    """Return the mean count a log linear model fits to each patient.

    The linear predictor is a matrix by vector product, and on the numpy the
    lock resolves for the oldest supported Python that product reports
    floating point flags its own arithmetic never raised: an ordinary call
    with the finite 0/1 design of ``compare_arr`` and finite coefficients
    raises a divide by zero, an overflow and an invalid value warning at once
    and still returns the right answer. Three warnings per call reached the
    user of a plain comparison on that leg. They are silenced here, at the one
    place they are raised rather than through a filter over the whole run, in
    the manner of ``msrelapse.simulate.simulate_paths``. The exponential is
    left outside the guard, so an overflow of the mean itself is still
    reported.

    Parameters
    ----------
    design : numpy.ndarray
        The design matrix, one row per patient.
    coefficients : numpy.ndarray
        The coefficients of the linear predictor, one per column of `design`.
    exposure : numpy.ndarray
        Follow up in years per patient, the offset of the model.

    Returns
    -------
    numpy.ndarray
        The fitted mean count of each patient.
    """
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        linear = design @ coefficients
    return exposure * np.exp(linear)


def _fit_poisson(design: _Matrix, counts: _Vector, exposure: _Vector) -> _Fit:
    """Fit a log linear Poisson model by Newton iteration on the exact score."""
    coefficients = np.zeros(design.shape[1])
    coefficients[0] = math.log(counts.sum() / exposure.sum())
    converged = False
    for _ in range(_NEWTON_STEPS):
        fitted = _fitted_mean(design, coefficients, exposure)
        step = np.linalg.solve(design.T @ (design * fitted[:, None]), design.T @ (counts - fitted))
        coefficients = coefficients + step
        if np.max(np.abs(step)) < _NEWTON_TOLERANCE:
            converged = True
            break
    fitted = _fitted_mean(design, coefficients, exposure)
    covariance = np.linalg.inv(design.T @ (design * fitted[:, None]))
    errors = _standard_errors(covariance, "poisson")
    return _Fit(coefficients, errors, 0.0, converged)


def _moment_dispersion(design: _Matrix, counts: _Vector, exposure: _Vector, fit: _Fit) -> float:
    """Return the method of moments NB2 dispersion around a fitted Poisson mean."""
    fitted = _fitted_mean(design, fit.coefficients, exposure)
    return float(np.sum((counts - fitted) ** 2 - counts) / np.sum(fitted**2))


def _bounded_log_dispersion(value: float) -> float:
    """Return a log dispersion the NB2 likelihood can be evaluated at.

    See ``_LOG_DISPERSION_LIMIT`` for why the bound is there. The likelihood
    and its score both read the bounded value wherever the free parameter
    appears, so they describe the same model at the bound. Beyond it the
    likelihood is flat while the score still reports the slope at the bound, so
    the pair stops being a function and its gradient. That costs nothing: the
    likelihood at the bound is far below its value at any start this module
    uses, so a line search turns back rather than settles there.
    """
    return min(max(value, -_LOG_DISPERSION_LIMIT), _LOG_DISPERSION_LIMIT)


def _poisson_negative_loglik(
    coefficients: _Vector, design: _Matrix, counts: _Vector, exposure: _Vector
) -> float:
    """Return the negative Poisson log likelihood at one set of coefficients.

    This is the value the negative binomial fit is measured against: the same
    counts under the model the negative binomial reduces to as its dispersion
    goes to zero, at the coefficients the Poisson fit maximised it with.
    """
    fitted = _fitted_mean(design, coefficients, exposure)
    return -float(np.sum(counts * np.log(fitted) - fitted - special.gammaln(counts + 1.0)))


def _nb_negative_loglik(
    parameters: _Vector, design: _Matrix, counts: _Vector, exposure: _Vector
) -> float:
    """Return the negative NB2 log likelihood at (coefficients, log dispersion)."""
    log_dispersion = _bounded_log_dispersion(float(parameters[-1]))
    dispersion = math.exp(log_dispersion)
    size = 1.0 / dispersion
    fitted = _fitted_mean(design, parameters[:-1], exposure)
    terms = (
        special.gammaln(counts + size)
        - special.gammaln(size)
        - special.gammaln(counts + 1.0)
        + counts * (log_dispersion + np.log(fitted))
        - (counts + size) * np.log1p(dispersion * fitted)
    )
    return -float(np.sum(terms))


def _nb_negative_score(
    parameters: _Vector, design: _Matrix, counts: _Vector, exposure: _Vector
) -> _Vector:
    """Return the gradient of ``_nb_negative_loglik``, computed analytically."""
    dispersion = math.exp(_bounded_log_dispersion(float(parameters[-1])))
    size = 1.0 / dispersion
    fitted = _fitted_mean(design, parameters[:-1], exposure)
    residual = (counts - fitted) / (1.0 + dispersion * fitted)
    by_coefficient = design.T @ residual
    by_dispersion = np.sum(
        (special.digamma(size) - special.digamma(counts + size) + np.log1p(dispersion * fitted))
        / dispersion**2
        + residual / dispersion
    )
    return -np.concatenate([by_coefficient, [dispersion * by_dispersion]])


def _numerical_hessian(
    parameters: _Vector, design: _Matrix, counts: _Vector, exposure: _Vector
) -> _Matrix:
    """Return the Hessian of the negative log likelihood by central differences.

    Differencing the analytic score rather than the log likelihood keeps the
    truncation error at the square of the step while halving the number of
    evaluations, which is what the standard errors need to agree with an
    analytic fit to several digits.
    """
    size = parameters.size
    hessian = np.empty((size, size))
    for index in range(size):
        step = _HESSIAN_STEP * max(1.0, abs(float(parameters[index])))
        forward = parameters.copy()
        forward[index] += step
        backward = parameters.copy()
        backward[index] -= step
        hessian[:, index] = (
            _nb_negative_score(forward, design, counts, exposure)
            - _nb_negative_score(backward, design, counts, exposure)
        ) / (2.0 * step)
    return (hessian + hessian.T) / 2.0


def _polish(
    parameters: _Vector, design: _Matrix, counts: _Vector, exposure: _Vector
) -> tuple[_Vector, bool]:
    """Refine an optimum with Newton steps on the analytic score.

    A gradient tolerance of 1e-10 is below the precision a log likelihood of a
    few hundred can be evaluated with, so the quasi-Newton search reports a
    loss of precision rather than a converged fit even when it has arrived.
    Two or three Newton steps from there drive the score to rounding error,
    which both settles the weakly identified dispersion and gives the caller a
    convergence flag that means something.

    The step is judged against ``_POLISH_TOLERANCE`` relative to the size of
    the parameters, so that the test scales with them and stays above the floor
    the differenced Hessian of this loop can deliver.
    """
    polished = parameters
    converged = False
    for _ in range(_NEWTON_STEPS):
        score = _nb_negative_score(polished, design, counts, exposure)
        step = np.linalg.solve(_numerical_hessian(polished, design, counts, exposure), score)
        polished = polished - step
        scale = max(1.0, float(np.max(np.abs(polished))))
        if float(np.max(np.abs(step))) < _POLISH_TOLERANCE * scale:
            converged = True
            break
    return polished, converged


def _fit_negative_binomial(
    design: _Matrix, counts: _Vector, exposure: _Vector, poisson: _Fit
) -> _Fit:
    """Fit an NB2 model, falling back on the Poisson fit wherever the dispersion collapses.

    The moment dispersion settles the question for the data before the search
    starts, because the maximum likelihood dispersion is zero exactly when the
    moment one is. Three further readings settle it for the search, which stops
    where the likelihood stops moving and not at the same place on every
    platform: a dispersion at or below ``_DISPERSION_COLLAPSE``, a curvature the
    standard errors cannot be read from while the dispersion is below
    ``_SINGULAR_DISPERSION``, and a likelihood that beats the Poisson one by no
    more than ``_LOGLIK_GAIN`` for each patient of the two arms. Each of them
    hands back the Poisson fit it was given, which carries a dispersion of 0 and
    the Poisson standard errors, the limit of the negative binomial ones.

    A Hessian with no curvature left in the dispersion direction is the
    signature of a dispersion that has gone to nothing, and no Wald interval can
    be read from it, which is why it is read as a collapse rather than raised
    while the dispersion is small. At a dispersion the data can carry it is a
    fault in the fit instead, and it reaches the caller. That is why the
    curvature is read before the likelihood the fit arrived at: a fit that
    cannot be inverted at a real dispersion is reported as the fault it is
    rather than taken for a collapse because its likelihood is poor.

    ``msrelapse.fit.fit_nb_counts`` fits the same NB2 model on bare counts and
    reads a collapsed search off its own numerical floor alone, so the two
    modules do not answer the question the same way: a search this one reports as
    a Poisson can reach a dispersion there. Neither reading is wrong for what its
    own caller reports, but a change to either belongs in both.
    """
    start_dispersion = _moment_dispersion(design, counts, exposure, poisson)
    if start_dispersion <= _DISPERSION_FLOOR:
        return poisson
    start = np.concatenate([poisson.coefficients, [math.log(start_dispersion)]])
    found = optimize.minimize(
        _nb_negative_loglik,
        start,
        args=(design, counts, exposure),
        jac=_nb_negative_score,
        method="BFGS",
        options={"gtol": _BFGS_GRADIENT_TOLERANCE, "maxiter": _BFGS_STEPS},
    )
    parameters, converged = _polish(np.asarray(found.x, dtype=np.float64), design, counts, exposure)
    dispersion = math.exp(_bounded_log_dispersion(float(parameters[-1])))
    # The check above settles the question for data the likelihood agrees with.
    # The three that follow catch the optimiser rather than the data: a search
    # that walked to a dispersion no model can distinguish from zero reports the
    # Poisson fit instead of an interval built on curvature that is not there.
    if dispersion <= _DISPERSION_COLLAPSE:
        return poisson
    try:
        covariance = np.linalg.inv(_numerical_hessian(parameters, design, counts, exposure))
        errors = _standard_errors(covariance, "nb")
    except np.linalg.LinAlgError:
        if dispersion >= _SINGULAR_DISPERSION:
            raise
        return poisson
    # What the dispersion bought, in log likelihood units: the Poisson fit at
    # its own optimum against the negative binomial fit at this one. The gain is
    # weighed per patient, because the difference of two log likelihoods carries
    # a rounding error that grows with the cohort; see _LOGLIK_GAIN.
    poisson_loss = _poisson_negative_loglik(poisson.coefficients, design, counts, exposure)
    nb_loss = _nb_negative_loglik(parameters, design, counts, exposure)
    if poisson_loss - nb_loss <= _LOGLIK_GAIN * counts.size:
        return poisson
    return _Fit(parameters[:-1], errors[:-1], dispersion, converged)


def _rate_ratio_result(fit: _Fit, model: ModelName, level: float) -> RateRatioResult:
    """Turn a fitted log linear model into the reported rate ratio."""
    intercept = float(fit.coefficients[0])
    coefficient = float(fit.coefficients[1])
    error = float(fit.standard_errors[1])
    half_width = float(stats.norm.ppf((1.0 + level) / 2.0)) * error
    p_value = 2.0 * float(stats.norm.sf(abs(coefficient) / error))
    return RateRatioResult(
        rate_ratio=math.exp(coefficient),
        ci_low=math.exp(coefficient - half_width),
        ci_high=math.exp(coefficient + half_width),
        p_value=p_value,
        arr_a=math.exp(intercept),
        arr_b=math.exp(intercept + coefficient),
        dispersion=fit.dispersion,
        model=model,
        converged=fit.converged,
    )
