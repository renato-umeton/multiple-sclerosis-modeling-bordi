"""Virtual cohorts of relapsing-remitting records, in the three schemas.

A cohort is described by a :class:`CohortSpec`: how many patients, how long each
is followed, and how long that patient stays in each of the two clinical states
on average. :func:`generate` turns such a description into the frames of
:mod:`msrelapse.io`, through either of two engines.

``renewal``
    The phenomenological engine of :mod:`msrelapse.renewal`. Durations are drawn
    directly from the target means, geometric on whole weeks or exponential in
    continuous time. It is fast and it has no potential behind it.
``sde``
    The mechanistic engine. Each pair of target durations is turned into a
    potential and a noise amplitude by :func:`msrelapse.model.calibrate`, and the
    record is then integrated with :func:`msrelapse.simulate.simulate_weekly`.

Beside the records, the module reports the per patient parameters of equation
(7) with :func:`per_patient_params`, and the three worked examples of the paper
with :func:`paper_patients`, which is where the printed inconsistency of patient
23 can be read off a table.

The rounding correction
-----------------------
A weekly record of the study marks any week a relapse touches as a relapse week,
so a relapse of continuous length L covers L + 1 whole weeks on average and the
remission beside it loses that week. The means the paper prints, about 100 weeks
in health and 4.3 weeks in no health, are means of such rounded durations. A
continuous time engine therefore has to be aimed a week below the printed
relapse and a week above the printed remission for its rounded output to land on
them, which is what :func:`continuous_targets` returns and what the stochastic
engine is calibrated to. The renewal engine needs no such correction either way:
on whole weeks it draws durations of exactly the target means and no rounding
follows, and without them it produces continuous events that are never rounded
at all.

What the correction does not cover
----------------------------------
The rounding is not the only thing between a calibrated potential and a weekly
record, and :func:`msrelapse.simulate.simulate_weekly` documents the other two
for itself. An episode read off the integration grid runs long, by about a tenth
at the default step of 0.02 weeks, and two relapses separated by less than a week
merge into one longer weekly episode, which at the default band fraction of 0.3
happens to about one remission in seven. Together they leave the weekly relapse
mean of a stochastic cohort about a third above its target at the defaults. Both
shrink with a shorter `dt` and a wider `band_fraction`: at a step of 0.01 weeks
and a band fraction of 0.5 fewer than one remission in twenty falls inside a
week, and a cohort of a hundred patients over a thousand weeks lands within
about a seventh of its relapse target and a tenth of its remission target. The
defaults are kept as they are because they are cheap, and a caller who wants the
targets reproduced closely should pay for the finer grid.

A caution on naive means
------------------------
The mean of the recorded durations of a state is not the mean the cohort was
generated from, and the gap is the end of follow up rather than anything in this
module. A remission of about a hundred weeks rarely fits twice into a record of a
few hundred, so a short window records the short remissions plus one that the end
of follow up cut off, and the naive mean lands about a quarter below the target.
The cohort of :func:`bordi2013_spec` shows this plainly, and so, for the same
reason, does the 100 weeks the article prints, which is a naive mean over windows
of the same lengths and understates its own cohort. Read
:func:`msrelapse.fit.fit_durations` with censoring for the corrected estimate,
and treat every naive mean, here and in the paper, as a lower bound.

References
----------
I. Bordi, R. Umeton, V. A. G. Ricigliano, et al., "A mechanistic, stochastic
model helps understand multiple sclerosis course and pathogenesis",
International Journal of Genomics, 2013, doi 10.1155/2013/910321.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Final, Literal

import numpy as np
import numpy.typing as npt
import pandas as pd

from msrelapse import _citation
from msrelapse._params import PAPER
from msrelapse.io import events_to_weekly, validate, weekly_to_durations, weekly_to_events
from msrelapse.model import (
    DoubleWell,
    barrier_ratio_from_durations,
    beta_from_barrier_ratio,
    calibrate,
)
from msrelapse.renewal import alternating_renewal, rates_from_means
from msrelapse.simulate import MAX_DT, simulate_weekly

__all__ = [
    "Cohort",
    "CohortSpec",
    "Engine",
    "Sampler",
    "Seed",
    "StartState",
    "bordi2013_spec",
    "constant",
    "continuous_targets",
    "draw",
    "empirical",
    "from_histogram",
    "generate",
    "lognormal_around",
    "paper_patients",
    "per_patient_params",
]

_Vector = npt.NDArray[np.float64]
"""One dimensional array of doubles, the shape every per patient quantity takes."""

Seed = np.random.Generator | int | None
"""What every random operation of this module accepts in its ``rng`` argument."""

Sampler = float | Callable[[np.random.Generator, int], npt.NDArray[np.float64]]
"""One per patient quantity: a single value shared by the cohort, or a draw of them."""

Engine = Literal["renewal", "sde"]
"""Which of the two generators produces the records."""

StartState = Literal["relapse", "health"]
"""Which state every patient is in at the start of follow up."""

_Frames = tuple[pd.DataFrame | None, pd.DataFrame | None, pd.DataFrame]
"""The weekly, durations and events frames a generator returns."""

_ENGINES: Final = ("renewal", "sde")
_START_STATES: Final = ("relapse", "health")

_NO_HEALTH: Final = PAPER.state_no_health.value
_HEALTH: Final = PAPER.state_health.value
_WEEK: Final = PAPER.time_resolution_weeks.value

# Identifiers are spelled the way msrelapse.renewal and msrelapse.simulate spell
# them, so that a cohort keeps the same patient ids whichever engine made it.
_MIN_ID_DIGITS: Final = 4

# A record has to hold a relapse and something else for any duration to be read
# off it, so follow up is never shorter than this many weeks.
_MIN_FOLLOWUP_WEEKS: Final = 2

# Shortest continuous relapse the rounding correction may aim at. Below half a
# week the correction has taken away most of what it was given, and the
# calibration would be asked for a potential whose relapse well barely exists.
_MIN_CONTINUOUS_RELAPSE_WEEKS: Final = 0.5

# The three worked examples of Section 3.4, page 6: the number the article
# gives each patient, then its mean remission, its mean relapse, its printed
# barrier ratio and its printed asymmetry. The numbers 23, 32 and 53 are labels
# rather than measurements, and msrelapse._params holds no field for them; they
# appear there only inside the names of the four fields gathered on each line.
_PAPER_PATIENTS: Final = (
    (
        23,
        PAPER.tau_health_p23_weeks,
        PAPER.tau_no_health_p23_weeks,
        PAPER.barrier_ratio_p23,
        PAPER.beta_patient_23,
    ),
    (
        32,
        PAPER.tau_health_p32_weeks,
        PAPER.tau_no_health_p32_weeks,
        PAPER.barrier_ratio_p32,
        PAPER.beta_patient_32,
    ),
    (
        53,
        PAPER.tau_health_p53_weeks,
        PAPER.tau_no_health_p53_weeks,
        PAPER.barrier_ratio_p53,
        PAPER.beta_patient_53,
    ),
)


def constant(value: float) -> Sampler:
    """Return a sampler that gives the same value to every patient.

    A bare float is already a sampler, so this exists for symmetry with the
    three drawing samplers: it lets a caller write every field of a
    :class:`CohortSpec` in the same shape.

    Parameters
    ----------
    value : float
        The value every patient receives, in weeks. It is checked for
        positivity by :func:`draw`, not here.

    Returns
    -------
    callable
        A sampler of `value`.

    Raises
    ------
    ValueError
        If `value` is not a finite number.

    Examples
    --------
    >>> draw(constant(4.3), 0, 3).tolist()
    [4.3, 4.3, 4.3]
    """
    if not math.isfinite(value):
        raise ValueError(f"value must be a finite number, got {value!r}")

    def sample(rng: np.random.Generator, n: int) -> _Vector:
        return np.full(n, float(value), dtype=np.float64)

    return sample


def lognormal_around(mean: float, cv: float) -> Sampler:
    """Return a log normal sampler of a given mean and coefficient of variation.

    The log normal is the usual way to spread a positive quantity over a cohort:
    it cannot go negative and its spread is set as a fraction of its mean rather
    than in weeks. The parameters of the underlying normal follow from the two
    moments as ``s**2 = log(1 + cv**2)`` and ``m = log(mean) - s**2 / 2``.

    Parameters
    ----------
    mean : float
        Mean of the drawn values, in weeks. Must be positive.
    cv : float
        Coefficient of variation, the standard deviation over the mean. Must be
        positive.

    Returns
    -------
    callable
        A sampler with that mean and that coefficient of variation.

    Raises
    ------
    ValueError
        If `mean` or `cv` is not a positive finite number.
    """
    if not math.isfinite(mean) or mean <= 0.0:
        raise ValueError(f"mean must be positive and finite, got {mean!r}")
    if not math.isfinite(cv) or cv <= 0.0:
        raise ValueError(f"cv must be positive and finite, got {cv!r}")
    spread = math.sqrt(math.log1p(cv * cv))
    centre = math.log(mean) - 0.5 * spread * spread

    def sample(rng: np.random.Generator, n: int) -> _Vector:
        drawn: _Vector = rng.lognormal(centre, spread, n)
        return drawn

    return sample


def empirical(values: npt.ArrayLike) -> Sampler:
    """Return a sampler that resamples a pool of observed values with replacement.

    Parameters
    ----------
    values : array_like
        The pool to resample, one dimensional and not empty.

    Returns
    -------
    callable
        A sampler that draws from `values` with replacement.

    Raises
    ------
    ValueError
        If `values` is not one dimensional, is empty, or holds a value that is
        not a finite number.
    """
    pool = np.asarray(values, dtype=np.float64)
    if pool.ndim != 1:
        raise ValueError(f"values must be one dimensional, got shape {pool.shape}")
    if pool.size == 0:
        raise ValueError("values must hold at least one value to resample")
    if not np.all(np.isfinite(pool)):
        raise ValueError("every value must be a finite number, but at least one is not")

    def sample(rng: np.random.Generator, n: int) -> _Vector:
        return pool[rng.integers(0, pool.size, n)]

    return sample


def from_histogram(bin_edges: Sequence[float], counts: Sequence[int]) -> Sampler:
    """Return a sampler that reads a published histogram back into values.

    A bin is chosen with probability proportional to its count, and a value is
    then drawn uniformly inside that bin. This is how the distribution of a
    figure is turned back into a cohort when the underlying values were never
    published, as with Figure 3 of the paper.

    Parameters
    ----------
    bin_edges : sequence of float
        The `n` + 1 edges of `n` bins, strictly ascending.
    counts : sequence of int
        One count per bin, none of them negative and not all zero.

    Returns
    -------
    callable
        A sampler of values spread over the bins in proportion to `counts`.

    Raises
    ------
    ValueError
        If fewer than two edges are given, if `counts` does not have one entry
        per bin, if the edges do not ascend, or if the counts are negative or
        all zero.

    Examples
    --------
    >>> values = draw(from_histogram([0.0, 1.0, 2.0], [1, 3]), 0, 4)
    >>> bool(values.min() >= 0.0 and values.max() <= 2.0)
    True
    """
    edges = np.asarray(bin_edges, dtype=np.float64)
    weights = np.asarray(counts, dtype=np.float64)
    if edges.ndim != 1 or edges.size < 2:
        raise ValueError(f"bin_edges must hold at least two edges, got shape {edges.shape}")
    if weights.ndim != 1 or weights.size != edges.size - 1:
        raise ValueError(
            f"counts must hold one count per bin, got {weights.size} for {edges.size - 1} bin(s)"
        )
    if not np.all(np.isfinite(edges)) or np.any(np.diff(edges) <= 0.0):
        raise ValueError("bin_edges must be finite and in strictly ascending order")
    if np.any(weights < 0.0):
        raise ValueError("counts must not be negative, but at least one is")
    total = float(weights.sum())
    if total <= 0.0:
        raise ValueError("counts must hold at least one count, but every bin is empty")
    probabilities = weights / total

    def sample(rng: np.random.Generator, n: int) -> _Vector:
        chosen = rng.choice(weights.size, size=n, p=probabilities)
        lower = edges[chosen]
        return lower + rng.random(n) * (edges[chosen + 1] - lower)

    return sample


def draw(sampler: Sampler, rng: Seed, n: int) -> _Vector:
    """Return `n` values from a sampler, in either of the two shapes it may take.

    Parameters
    ----------
    sampler : float or callable
        A single value shared by the whole cohort, or a callable taking a
        generator and a count and returning that many values.
    rng : numpy.random.Generator or int or None
        Generator to draw from, or a seed for
        :func:`numpy.random.default_rng`.
    n : int
        Number of values wanted. Must be at least one.

    Returns
    -------
    numpy.ndarray
        The `n` values, as float64.

    Raises
    ------
    ValueError
        If `n` is below one, if a callable sampler returns something other than
        `n` values, or if any value is not a positive finite number, since every
        quantity a cohort is described by is a duration in weeks.
    """
    if n < 1:
        raise ValueError(f"n must be at least 1, got {n!r}")
    generator = _generator(rng)
    if callable(sampler):
        values = np.asarray(sampler(generator, n), dtype=np.float64)
    else:
        values = np.full(n, float(sampler), dtype=np.float64)
    if values.shape != (n,):
        raise ValueError(
            f"a sampler must return one value per patient, got shape {values.shape} for n = {n}"
        )
    offenders = np.flatnonzero(~(np.isfinite(values) & (values > 0.0)))
    if offenders.size:
        first = int(offenders[0])
        raise ValueError(
            f"every drawn duration must be positive and finite, because it is a number of "
            f"weeks; entry {first} of {values.size} is {values[first]!r}"
        )
    return values


def continuous_targets(
    tau_health: float,
    tau_relapse: float,
    weekly: bool = True,
) -> tuple[float, float]:
    """Return the targets a continuous time engine has to be aimed at.

    The weekly records of the study mark any week a relapse touches as a relapse
    week, so a relapse of continuous length L occupies L + 1 whole weeks on
    average and the remission beside it loses that week. To make the rounded
    output reproduce a printed pair of means, the continuous engine is therefore
    aimed a week below the printed relapse and a week above the printed
    remission. A record that is never rounded needs no correction at all, which
    is what `weekly` False asks for and what the renewal engine of
    :class:`CohortSpec` with ``weekly=False`` produces.

    Parameters
    ----------
    tau_health : float
        Mean remission duration wanted in the record, in weeks. Must be
        positive.
    tau_relapse : float
        Mean relapse duration wanted in the record, in weeks. Must be positive.
    weekly : bool, optional
        Whether the record the targets are wanted for is rounded to whole weeks.

    Returns
    -------
    tuple of float
        The remission and the relapse target to calibrate to, in weeks.

    Raises
    ------
    ValueError
        If either duration is not positive, or if `weekly` is True and the
        correction leaves a relapse shorter than half a week.

    Examples
    --------
    >>> continuous_targets(100.0, 4.3)
    (101.0, 3.3)
    """
    if not math.isfinite(tau_health) or tau_health <= 0.0:
        raise ValueError(f"tau_health must be a positive finite number, got {tau_health!r}")
    if not math.isfinite(tau_relapse) or tau_relapse <= 0.0:
        raise ValueError(f"tau_relapse must be a positive finite number, got {tau_relapse!r}")
    if not weekly:
        return float(tau_health), float(tau_relapse)
    corrected = tau_relapse - _WEEK
    if corrected < _MIN_CONTINUOUS_RELAPSE_WEEKS:
        raise ValueError(
            f"the rounding correction takes one week off a relapse of {tau_relapse!r} weeks "
            f"and leaves {corrected!r}, below the floor of "
            f"{_MIN_CONTINUOUS_RELAPSE_WEEKS} weeks; a continuous time engine cannot be "
            f"calibrated that short, so either raise tau_relapse or generate the cohort "
            f"with weekly=False"
        )
    return float(tau_health + _WEEK), float(corrected)


@dataclass(frozen=True)
class CohortSpec:
    """Everything needed to generate one cohort.

    Parameters
    ----------
    n : int
        Number of patients. Must be at least one.
    tau_health : float or callable
        Mean remission duration of each patient, in weeks.
    tau_relapse : float or callable
        Mean relapse duration of each patient, in weeks.
    followup_weeks : float or callable
        Length of each patient's record, in weeks. It is rounded up to whole
        weeks and never falls below two.
    engine : {'renewal', 'sde'}, optional
        Which generator produces the records.
    alpha : float, optional
        Control parameter of the potential, held fixed as the paper holds it.
        Used by the ``sde`` engine alone.
    weekly : bool, optional
        Whether the records are rounded to whole weeks. With False and the
        ``renewal`` engine the natural output is the events table and no weekly
        or durations frame is built. The ``sde`` engine produces a weekly record
        either way, because its integration grid is downsampled to whole weeks
        on the way out, so there False only turns the rounding correction off.
    band_fraction : float, optional
        Position of the two hysteresis thresholds that cut a simulated path into
        episodes, as a fraction of the distance from the saddle to each well
        bottom. Must lie strictly between 0 and 1. Used by the ``sde`` engine
        alone.
    dt : float, optional
        Integration step, in weeks. Must be positive and no larger than
        :data:`msrelapse.simulate.MAX_DT`. Used by the ``sde`` engine alone.
    start_state : {'relapse', 'health'}, optional
        State every patient is in at week 0. The records of the paper start at
        the first relapse, which is the default.

    Raises
    ------
    ValueError
        If `n` is below one, if `alpha` is not positive, if `dt` is outside
        ``(0, MAX_DT]``, if `band_fraction` is not strictly between 0 and 1, or
        if `engine` or `start_state` is not one of the accepted spellings.

    Notes
    -----
    The defaults of `dt` and `band_fraction` are the cheap ones, not the
    accurate ones. See the module docstring for what they cost a stochastic
    cohort and for the values that buy the targets back.
    """

    n: int
    tau_health: Sampler
    tau_relapse: Sampler
    followup_weeks: Sampler
    engine: Engine = "renewal"
    alpha: float = PAPER.alpha_reference.value
    weekly: bool = True
    band_fraction: float = 0.3
    dt: float = 0.02
    start_state: StartState = "relapse"

    def __post_init__(self) -> None:
        """Reject a description no cohort can be generated from.

        Raises
        ------
        ValueError
            If any field is outside the range named in the class docstring.
        """
        if self.n < 1:
            raise ValueError(f"n must be at least 1 patient, got {self.n!r}")
        if not math.isfinite(self.alpha) or self.alpha <= 0.0:
            raise ValueError(f"alpha must be a positive finite number, got {self.alpha!r}")
        if not math.isfinite(self.dt) or not 0.0 < self.dt <= MAX_DT:
            raise ValueError(
                f"dt must be a finite number of weeks in (0, {MAX_DT}], got {self.dt!r}"
            )
        if not 0.0 < self.band_fraction < 1.0:
            raise ValueError(
                f"band_fraction must lie strictly between 0 and 1, got {self.band_fraction!r}"
            )
        if self.engine not in _ENGINES:
            raise ValueError(f"engine must be one of {_ENGINES}, got {self.engine!r}")
        if self.start_state not in _START_STATES:
            raise ValueError(
                f"start_state must be one of {_START_STATES}, got {self.start_state!r}"
            )


@dataclass(frozen=True, repr=False)
class Cohort:
    """One generated cohort, in every schema its engine can express it.

    Attributes
    ----------
    spec : CohortSpec
        The description the cohort was generated from.
    weekly : pandas.DataFrame or None
        One row per patient-week, in the weekly schema of :mod:`msrelapse.io`.
        None when the records were not rounded to whole weeks and the engine was
        ``renewal``, where the natural output is the events table alone.
    durations : pandas.DataFrame or None
        The run length encoding of `weekly`, in the durations schema. None under
        the same rule as `weekly`.
    events : pandas.DataFrame
        One row per relapse, in the events schema. Always present.
    patients : pandas.DataFrame
        One row per patient, with the columns ``patient_id``, ``tau_health``,
        ``tau_relapse`` and ``followup_weeks`` that the patient was generated
        from, and the ``beta`` and ``sigma`` the ``sde`` engine calibrated for
        it, which are missing for the ``renewal`` engine.
    provenance : str
        One sentence saying that the records are synthetic and are not the
        clinical series of the paper.
    """

    spec: CohortSpec
    weekly: pd.DataFrame | None
    durations: pd.DataFrame | None
    events: pd.DataFrame
    patients: pd.DataFrame
    provenance: str

    @property
    def citation(self) -> str:
        """str: One line naming the article this cohort reproduces."""
        return _citation.short_citation()

    def __repr__(self) -> str:
        """Return the size, the engine and the citation, on one line.

        Returns
        -------
        str
            The number of patients, the engine that generated them, the number
            of rows in the events table and the short citation.
        """
        return (
            f"{type(self).__name__}(n={self.spec.n}, engine={self.spec.engine!r}, "
            f"{len(self.events)} event row(s); {self.citation})"
        )


def generate(spec: CohortSpec, rng: Seed = None) -> Cohort:
    """Generate one cohort from its description.

    The per patient durations and the length of each record are drawn first,
    then handed to the engine `spec` names. Every frame is validated against its
    schema before it is returned.

    Parameters
    ----------
    spec : CohortSpec
        The description to generate from.
    rng : numpy.random.Generator or int or None, optional
        Generator behind every draw of the call, or a seed for
        :func:`numpy.random.default_rng`. A given seed reproduces the cohort
        exactly.

    Returns
    -------
    Cohort
        The records, the per patient parameters and the provenance sentence.

    Raises
    ------
    ValueError
        If a sampler returns a duration that is not positive, if the ``sde``
        engine is asked for a relapse the rounding correction would empty, if no
        potential reproduces a patient's pair of targets, or if a generated
        frame breaks its schema.

    See Also
    --------
    bordi2013_spec : The description of the cohort of the paper.
    continuous_targets : The rounding correction the ``sde`` engine is aimed at.

    Examples
    --------
    >>> spec = CohortSpec(n=2, tau_health=100.0, tau_relapse=4.3, followup_weeks=200.0)
    >>> sorted(set(generate(spec, rng=0).events["patient_id"]))
    ['p0001', 'p0002']
    """
    generator = _generator(rng)
    tau_health = draw(spec.tau_health, generator, spec.n)
    tau_relapse = draw(spec.tau_relapse, generator, spec.n)
    followup = _whole_weeks(draw(spec.followup_weeks, generator, spec.n))
    identifiers = _patient_ids(spec.n)
    if spec.engine == "renewal":
        beta = np.full(spec.n, math.nan, dtype=np.float64)
        sigma = np.full(spec.n, math.nan, dtype=np.float64)
        frames = _renewal_frames(spec, tau_health, tau_relapse, followup, generator=generator)
    else:
        beta, sigma = _calibrated_parameters(spec, tau_health, tau_relapse)
        frames = _sde_frames(
            spec, beta, sigma, followup, identifiers=identifiers, generator=generator
        )
    weekly, durations, events = frames
    _validate_frames(weekly, durations, events)
    return Cohort(
        spec=spec,
        weekly=weekly,
        durations=durations,
        events=events,
        patients=_patients_frame(
            identifiers, tau_health, tau_relapse, followup, beta=beta, sigma=sigma
        ),
        provenance=_provenance(spec),
    )


def bordi2013_spec(engine: Engine = "renewal") -> CohortSpec:
    """Return the description of the cohort the paper studied.

    Every number comes from :data:`msrelapse._params.PAPER`: the cohort size,
    the two mean durations of Section 3.4, and the distribution of
    relapsing-remitting phase lengths of Figure 3, which supplies the length of
    each record.

    Parameters
    ----------
    engine : {'renewal', 'sde'}, optional
        Which generator to describe the cohort for.

    Returns
    -------
    CohortSpec
        The description, with weekly records as the study recorded them.

    Raises
    ------
    ValueError
        If `engine` is not one of the two spellings.

    Examples
    --------
    >>> bordi2013_spec().n == PAPER.n_patients.value
    True
    """
    return CohortSpec(
        n=PAPER.n_patients.value,
        tau_health=PAPER.tau_health_cohort_weeks.value,
        tau_relapse=PAPER.tau_no_health_cohort_weeks.value,
        followup_weeks=from_histogram(PAPER.fig3_bin_edges_weeks.value, PAPER.fig3_counts.value),
        engine=engine,
        weekly=True,
    )


def per_patient_params(durations: pd.DataFrame) -> pd.DataFrame:
    """Return the parameters equation (7) reads off each patient's record.

    The two mean durations are the naive ones, over every recorded run of a
    state with no censoring correction, because those are the means the paper
    used. The barrier ratio is then equation (7), and the asymmetry is the beta
    that delivers that ratio at the reference alpha, which is the fitting rule
    of page 6.

    Parameters
    ----------
    durations : pandas.DataFrame
        A frame in the durations schema of :mod:`msrelapse.io`. It is validated
        before use.

    Returns
    -------
    pandas.DataFrame
        One row per patient, sorted by ``patient_id``, with the columns
        ``patient_id``, ``n_relapses``, ``n_remissions``, ``tau_health``,
        ``tau_relapse``, ``barrier_ratio`` and ``beta``. The last two are
        missing for a patient with no run of one of the two states, or whose
        mean duration in either state is one week or shorter, where the
        logarithm of equation (7) is zero or negative and the estimator is
        undefined.

    Raises
    ------
    ValueError
        If `durations` does not obey the durations schema, or if a patient's
        barrier ratio is larger than any potential at the reference alpha can
        deliver.

    See Also
    --------
    msrelapse.fit.barrier_ratio : The same estimator over a whole cohort.
    """
    validate(durations, "durations")
    rows: list[tuple[str, int, int, float, float, float, float]] = []
    for patient, group in durations.groupby("patient_id", sort=True):
        relapses = group.loc[group["state"] == _NO_HEALTH, "duration_w"]
        remissions = group.loc[group["state"] == _HEALTH, "duration_w"]
        tau_health = float(remissions.mean()) if not remissions.empty else math.nan
        tau_relapse = float(relapses.mean()) if not relapses.empty else math.nan
        ratio = _barrier_ratio_or_missing(tau_health, tau_relapse)
        beta = beta_from_barrier_ratio(ratio) if ratio >= 1.0 else math.nan
        rows.append(
            (str(patient), relapses.size, remissions.size, tau_health, tau_relapse, ratio, beta)
        )
    frame = pd.DataFrame(
        rows,
        columns=[
            "patient_id",
            "n_relapses",
            "n_remissions",
            "tau_health",
            "tau_relapse",
            "barrier_ratio",
            "beta",
        ],
    )
    return frame.astype(
        {
            "n_relapses": np.int64,
            "n_remissions": np.int64,
            "tau_health": np.float64,
            "tau_relapse": np.float64,
            "barrier_ratio": np.float64,
            "beta": np.float64,
        }
    )


def paper_patients() -> pd.DataFrame:
    """Return the three worked examples of Section 3.4, recomputed beside the print.

    The paper prints, for patients 23, 32 and 53, a pair of mean durations, a
    barrier ratio and a fitted asymmetry. The table puts each printed value next
    to the value this package computes from its neighbours, so that the
    documented inconsistency of patient 23 can be read off a row: the ratio its
    two printed durations imply asks for an asymmetry of about 0.256, while the
    asymmetry printed beside it delivers a ratio of about 10.8 rather than the
    11.8 printed with it. The two rows below it agree to two decimals.

    Returns
    -------
    pandas.DataFrame
        One row per patient, with the columns ``patient_id`` (the number printed
        in the paper), ``tau_health``, ``tau_relapse``,
        ``barrier_ratio_printed``, ``beta_printed``, ``barrier_ratio_from_taus``
        (equation (7) applied to the two printed durations), ``beta_from_ratio``
        (the asymmetry that delivers that recomputed ratio) and
        ``barrier_ratio_from_beta_printed`` (the ratio the printed beta actually
        delivers).

    Examples
    --------
    >>> paper_patients()["patient_id"].tolist()
    [23, 32, 53]
    """
    alpha = PAPER.alpha_reference.value
    rows: list[tuple[int, float, float, float, float, float, float, float]] = []
    for number, health, relapse, printed_ratio, printed_beta in _PAPER_PATIENTS:
        tau_health = float(health.value)
        tau_relapse = float(relapse.value)
        beta = float(printed_beta.value)
        from_taus = barrier_ratio_from_durations(tau_health, tau_relapse)
        rows.append(
            (
                number,
                tau_health,
                tau_relapse,
                float(printed_ratio.value),
                beta,
                from_taus,
                beta_from_barrier_ratio(from_taus, alpha),
                DoubleWell(alpha, beta).barrier_ratio(),
            )
        )
    frame = pd.DataFrame(
        rows,
        columns=[
            "patient_id",
            "tau_health",
            "tau_relapse",
            "barrier_ratio_printed",
            "beta_printed",
            "barrier_ratio_from_taus",
            "beta_from_ratio",
            "barrier_ratio_from_beta_printed",
        ],
    )
    return frame.astype({"patient_id": np.int64})


def _barrier_ratio_or_missing(tau_health: float, tau_relapse: float) -> float:
    """Return equation (7) for one patient, or NaN where it is undefined.

    Parameters
    ----------
    tau_health : float
        Naive mean remission duration, in weeks, or NaN when the patient has
        no remission at all.
    tau_relapse : float
        Naive mean relapse duration, in weeks, or NaN when the patient has no
        relapse at all.

    Returns
    -------
    float
        The implied barrier ratio, or NaN when either mean is missing or is one
        week or shorter, where the logarithm is zero or negative.
    """
    if not (tau_health > _WEEK and tau_relapse > _WEEK):
        return math.nan
    return barrier_ratio_from_durations(tau_health, tau_relapse)


def _generator(rng: Seed) -> np.random.Generator:
    """Return the generator to draw from, building one from a seed if needed.

    Parameters
    ----------
    rng : numpy.random.Generator or int or None
        A generator, which is used as it is, or a seed.

    Returns
    -------
    numpy.random.Generator
        The generator every draw of the call goes through.
    """
    if isinstance(rng, np.random.Generator):
        return rng
    return np.random.default_rng(rng)


def _patient_ids(n: int) -> list[str]:
    """Return the ids p0001, p0002 and so on for a cohort of `n` patients.

    Parameters
    ----------
    n : int
        Number of patients.

    Returns
    -------
    list of str
        One identifier per patient, zero padded so that they sort in order, in
        the spelling :mod:`msrelapse.renewal` and :mod:`msrelapse.simulate` use.
    """
    digits = max(_MIN_ID_DIGITS, len(str(n)))
    return [f"p{number:0{digits}d}" for number in range(1, n + 1)]


def _whole_weeks(values: _Vector) -> npt.NDArray[np.int64]:
    """Return follow up lengths as whole weeks, rounded up and never below two.

    Parameters
    ----------
    values : numpy.ndarray
        Drawn follow up lengths, in weeks.

    Returns
    -------
    numpy.ndarray
        The same lengths as int64 weeks. Rounding up matches the rule of the
        study, under which a part of a week is recorded as a whole one.
    """
    rounded = np.maximum(np.ceil(values), float(_MIN_FOLLOWUP_WEEKS))
    return rounded.astype(np.int64)


def _renewal_frames(
    spec: CohortSpec,
    tau_health: _Vector,
    tau_relapse: _Vector,
    followup: npt.NDArray[np.int64],
    *,
    generator: np.random.Generator,
) -> _Frames:
    """Generate the records of a cohort with the alternating renewal engine.

    Parameters
    ----------
    spec : CohortSpec
        The description being generated from.
    tau_health : numpy.ndarray
        Mean remission duration of each patient, in weeks.
    tau_relapse : numpy.ndarray
        Mean relapse duration of each patient, in weeks.
    followup : numpy.ndarray
        Length of each record, in whole weeks.
    generator : numpy.random.Generator
        The generator every draw goes through.

    Returns
    -------
    tuple
        The weekly, durations and events frames. The first two are None when
        the records were not rounded to whole weeks, where the events table is
        the natural output and no weekly record exists.
    """
    rates = [
        rates_from_means(float(health), float(relapse))
        for health, relapse in zip(tau_health, tau_relapse, strict=True)
    ]
    events = alternating_renewal(
        np.array([lam for lam, _ in rates], dtype=np.float64),
        np.array([mu for _, mu in rates], dtype=np.float64),
        followup.astype(np.float64),
        n=spec.n,
        rng=generator,
        discretise="week" if spec.weekly else None,
        start_state=spec.start_state,
    )
    if not spec.weekly:
        return None, None, events
    weekly = events_to_weekly(events)
    return weekly, weekly_to_durations(weekly), events


def _calibrated_parameters(
    spec: CohortSpec,
    tau_health: _Vector,
    tau_relapse: _Vector,
) -> tuple[_Vector, _Vector]:
    """Return the potential and the noise of every patient of a stochastic cohort.

    Each distinct pair of targets is calibrated once and cached, so that a
    cohort whose patients share their targets pays for a single calibration.

    Parameters
    ----------
    spec : CohortSpec
        The description being generated from, which fixes alpha, the band
        fraction and whether the rounding correction applies.
    tau_health : numpy.ndarray
        Mean remission duration of each patient, in weeks.
    tau_relapse : numpy.ndarray
        Mean relapse duration of each patient, in weeks.

    Returns
    -------
    tuple of numpy.ndarray
        The asymmetry and the noise amplitude of each patient.

    Raises
    ------
    ValueError
        If the rounding correction would empty a relapse, or if no potential
        reproduces a patient's pair of targets.
    """
    cache: dict[tuple[float, float], tuple[float, float]] = {}
    beta = np.empty(spec.n, dtype=np.float64)
    sigma = np.empty(spec.n, dtype=np.float64)
    for index in range(spec.n):
        key = (float(tau_health[index]), float(tau_relapse[index]))
        if key not in cache:
            health, relapse = continuous_targets(key[0], key[1], spec.weekly)
            cache[key] = calibrate(
                health,
                relapse,
                spec.alpha,
                method="mfpt",
                passage="band",
                band_fraction=spec.band_fraction,
            )
        beta[index], sigma[index] = cache[key]
    return beta, sigma


def _sde_frames(
    spec: CohortSpec,
    beta: _Vector,
    sigma: _Vector,
    followup: npt.NDArray[np.int64],
    *,
    identifiers: Sequence[str],
    generator: np.random.Generator,
) -> _Frames:
    """Integrate one record per patient and derive the three frames.

    Parameters
    ----------
    spec : CohortSpec
        The description being generated from.
    beta : numpy.ndarray
        Asymmetry of each patient's potential.
    sigma : numpy.ndarray
        Noise amplitude of each patient.
    followup : numpy.ndarray
        Length of each record, in whole weeks.
    identifiers : sequence of str
        One patient identifier per record.
    generator : numpy.random.Generator
        The generator every increment is drawn from.

    Returns
    -------
    tuple
        The weekly, durations and events frames, none of them None: a simulated
        record is weekly whatever `spec` asks for, because the integration grid
        is downsampled to whole weeks on the way out.
    """
    frames: list[pd.DataFrame] = []
    for index, patient in enumerate(identifiers):
        well = DoubleWell(spec.alpha, float(beta[index]))
        frames.append(
            simulate_weekly(
                well,
                float(sigma[index]),
                int(followup[index]),
                n_paths=1,
                dt=spec.dt,
                rng=generator,
                band_fraction=spec.band_fraction,
                x0=well.well_bottom(spec.start_state),
                patient_ids=[patient],
            )
        )
    weekly = pd.concat(frames, ignore_index=True)
    return weekly, weekly_to_durations(weekly), weekly_to_events(weekly)


def _patients_frame(
    identifiers: Sequence[str],
    tau_health: _Vector,
    tau_relapse: _Vector,
    followup: npt.NDArray[np.int64],
    *,
    beta: _Vector,
    sigma: _Vector,
) -> pd.DataFrame:
    """Return the per patient table of what the cohort was generated from.

    Parameters
    ----------
    identifiers : sequence of str
        One patient identifier per row.
    tau_health : numpy.ndarray
        Mean remission duration of each patient, in weeks.
    tau_relapse : numpy.ndarray
        Mean relapse duration of each patient, in weeks.
    followup : numpy.ndarray
        Length of each record, in whole weeks.
    beta : numpy.ndarray
        Asymmetry of each patient's potential, all missing for the renewal
        engine, which has no potential behind it.
    sigma : numpy.ndarray
        Noise amplitude of each patient, missing under the same rule.

    Returns
    -------
    pandas.DataFrame
        One row per patient, in identifier order.
    """
    return pd.DataFrame(
        {
            "patient_id": pd.Series(list(identifiers), dtype=object),
            "tau_health": tau_health,
            "tau_relapse": tau_relapse,
            "followup_weeks": followup,
            "beta": beta,
            "sigma": sigma,
        }
    )


def _validate_frames(
    weekly: pd.DataFrame | None,
    durations: pd.DataFrame | None,
    events: pd.DataFrame,
) -> None:
    """Check every generated frame against the schema it claims to follow.

    Parameters
    ----------
    weekly : pandas.DataFrame or None
        The weekly record, or None when the engine produced none.
    durations : pandas.DataFrame or None
        The run length encoding, or None under the same rule.
    events : pandas.DataFrame
        The events table, which every engine produces.

    Returns
    -------
    None
        Nothing is returned; a frame that breaks its schema raises instead.

    Raises
    ------
    ValueError
        If any frame breaks its schema.
    """
    if weekly is not None:
        validate(weekly, "weekly")
    if durations is not None:
        validate(durations, "durations")
    validate(events, "events")


def _provenance(spec: CohortSpec) -> str:
    """Return the sentence that travels with a generated cohort.

    Parameters
    ----------
    spec : CohortSpec
        The description the cohort was generated from.

    Returns
    -------
    str
        One sentence naming the engine and saying plainly that the records are
        synthetic and are not the clinical series of the paper, which was never
        released.
    """
    return (
        f"Synthetic data: {spec.n} record(s) generated by msrelapse with the "
        f"{spec.engine} engine from the target durations of this cohort spec, and not the "
        f"{PAPER.n_patients.value} patient clinical series of Bordi et al. 2013, which the "
        f"article never released."
    )
