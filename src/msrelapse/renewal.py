"""Alternating renewal description of a relapsing-remitting record.

The paper describes a patient as alternating between a state of health and a
state of no health, spending on average tau_x1 weeks in health and tau_x2 weeks
in a relapse. Read as a phenomenological model rather than a mechanistic one,
that is an alternating renewal process: remission durations are exponential
with rate lambda = 1 / tau_x1, relapse durations are exponential with rate
mu = 1 / tau_x2, and the two alternate. This module generates event tables from
such a process, counts the relapse onsets they contain, and gives the Poisson
and negative binomial statistics that the counts follow when relapses are brief
and when the onset rate varies from patient to patient.

The module is the phenomenological twin of the stochastic differential equation
in :mod:`msrelapse.model`: it produces the same kind of record without any
double well behind it, which is what makes it useful as a null model. Numbers
reported by the paper are never written here; import ``PAPER`` from
:mod:`msrelapse._params` instead.

References
----------
I. Bordi, R. Umeton, V. A. G. Ricigliano, V. Annibali, R. Mechelli, G. Ristori,
F. Grassi, M. Salvetti, and A. Sutera, "A mechanistic, stochastic model helps
understand multiple sclerosis course and pathogenesis," International Journal
of Genomics, vol. 2013, 910321, 2013, doi 10.1155/2013/910321.
"""

from __future__ import annotations

import math
from typing import Final, Literal

import numpy as np
import numpy.typing as npt
import pandas as pd

from msrelapse.io import validate

__all__ = [
    "alternating_renewal",
    "effective_onset_rate",
    "gamma_rates",
    "nb_from_gamma",
    "rates_from_means",
    "relapse_counts",
    "relapse_free",
]

_EVENTS_COLUMNS: Final = (
    "patient_id",
    "followup_start",
    "followup_end",
    "relapse_onset",
    "relapse_end",
)
_TIME_COLUMNS: Final = _EVENTS_COLUMNS[1:]
_MIN_ID_DIGITS: Final = 4
_START_STATES: Final = ("relapse", "health")
_DISCRETISATIONS: Final = (None, "week")

Seed = np.random.Generator | int | None


def rates_from_means(tau_health: float, tau_relapse: float) -> tuple[float, float]:
    """Convert mean residence times into the rates of the renewal process.

    Parameters
    ----------
    tau_health : float
        Mean duration of a remission, in weeks. Must be positive.
    tau_relapse : float
        Mean duration of a relapse, in weeks. Must be positive.

    Returns
    -------
    tuple of float
        The remission rate lambda = 1 / `tau_health` and the relapse rate
        mu = 1 / `tau_relapse`, both per week.

    Raises
    ------
    ValueError
        If either mean duration is zero or negative.

    Examples
    --------
    >>> rates_from_means(100.0, 4.0)
    (0.01, 0.25)
    """
    if tau_health <= 0.0:
        raise ValueError(f"tau_health must be positive, got {tau_health!r}")
    if tau_relapse <= 0.0:
        raise ValueError(f"tau_relapse must be positive, got {tau_relapse!r}")
    return 1.0 / tau_health, 1.0 / tau_relapse


def effective_onset_rate(lambda_remission: float, mu_relapse: float) -> float:
    """Return the long run rate of relapse onsets of the alternating process.

    Onsets are separated by one remission and one relapse, so they arrive at
    1 / (1 / lambda + 1 / mu), which is lambda / (1 + lambda / mu). The rate is
    lower than lambda because the time spent in relapse is time in which no new
    relapse can start.

    Parameters
    ----------
    lambda_remission : float
        Rate at which a remission ends, per week. Must be positive.
    mu_relapse : float
        Rate at which a relapse ends, per week. Must be positive.

    Returns
    -------
    float
        Onset rate per week.

    Raises
    ------
    ValueError
        If either rate is zero or negative.

    Examples
    --------
    >>> effective_onset_rate(0.01, 0.01)
    0.005
    """
    if lambda_remission <= 0.0:
        raise ValueError(f"lambda_remission must be positive, got {lambda_remission!r}")
    if mu_relapse <= 0.0:
        raise ValueError(f"mu_relapse must be positive, got {mu_relapse!r}")
    return lambda_remission / (1.0 + lambda_remission / mu_relapse)


def nb_from_gamma(k: float, theta: float, T: float) -> tuple[float, float]:
    """Return the negative binomial moments of a gamma mixture of Poisson counts.

    If each patient has their own onset rate drawn from a Gamma distribution of
    shape `k` and scale `theta`, and their relapse count over a window of `T`
    weeks is Poisson given that rate, then the pooled counts are negative
    binomial with mean `k` `theta` `T` and dispersion 1 / `k`, so that the
    variance is mean + dispersion mean squared.

    Parameters
    ----------
    k : float
        Shape of the gamma distribution of onset rates. Must be positive.
    theta : float
        Scale of the gamma distribution of onset rates, in onsets per week.
        Must be positive.
    T : float
        Length of the counting window, in weeks. Must be positive.

    Returns
    -------
    tuple of float
        The mean count and the dispersion 1 / `k`.

    Raises
    ------
    ValueError
        If any argument is zero or negative.

    Examples
    --------
    >>> nb_from_gamma(2.0, 0.005, 500.0)
    (5.0, 0.5)
    """
    if k <= 0.0:
        raise ValueError(f"k must be positive, got {k!r}")
    if theta <= 0.0:
        raise ValueError(f"theta must be positive, got {theta!r}")
    if T <= 0.0:
        raise ValueError(f"T must be positive, got {T!r}")
    return k * theta * T, 1.0 / k


def relapse_free(
    T: float | npt.ArrayLike,
    lam: float | None = None,
    k: float | None = None,
    theta: float | None = None,
) -> float | npt.NDArray[np.float64]:
    """Return the probability of staying free of relapse over a window.

    With a single onset rate `lam` the waiting time to the first relapse is
    exponential and the probability is exp(-`lam` `T`). When the rate varies
    between patients as a Gamma distribution of shape `k` and scale `theta`,
    averaging the exponential over that distribution gives
    (1 + `theta` `T`) raised to minus `k`, which decays more slowly because the
    patients with a low rate dominate the survivors.

    Parameters
    ----------
    T : float or array_like
        Window length in weeks. May be an array of windows.
    lam : float, optional
        Single onset rate per week. Give this or the pair `k` and `theta`.
    k : float, optional
        Shape of the gamma distribution of onset rates.
    theta : float, optional
        Scale of the gamma distribution of onset rates, in onsets per week.

    Returns
    -------
    float or numpy.ndarray
        The probability, as a float when `T` is a scalar and as an array with
        the shape of `T` otherwise.

    Raises
    ------
    ValueError
        If neither or both parameterisations are supplied, if the gamma
        parameterisation is incomplete, or if any value is out of range.

    Examples
    --------
    >>> round(relapse_free(100.0, lam=0.01), 6)
    0.367879
    """
    gamma_given = k is not None or theta is not None
    if lam is not None and gamma_given:
        raise ValueError("give either lam or the pair k and theta, not both")
    if lam is None and not gamma_given:
        raise ValueError("give either lam or the pair k and theta")

    window = np.asarray(T, dtype=np.float64)
    if np.any(window < 0.0):
        raise ValueError("T must not be negative")

    if lam is not None:
        if lam <= 0.0:
            raise ValueError(f"lam must be positive, got {lam!r}")
        probability = np.exp(-lam * window)
    else:
        if k is None or theta is None:
            raise ValueError(f"gamma mixing needs both k and theta, got k={k!r}, theta={theta!r}")
        if k <= 0.0:
            raise ValueError(f"k must be positive, got {k!r}")
        if theta <= 0.0:
            raise ValueError(f"theta must be positive, got {theta!r}")
        probability = (1.0 + theta * window) ** -k

    if np.ndim(T) == 0:
        return float(probability)
    return probability


def gamma_rates(
    k: float,
    theta: float,
    n: int,
    rng: Seed = None,
) -> npt.NDArray[np.float64]:
    """Draw one onset rate per patient from a Gamma distribution.

    Parameters
    ----------
    k : float
        Shape of the distribution. Must be positive.
    theta : float
        Scale of the distribution, in onsets per week. Must be positive.
    n : int
        Number of patients. Must be at least one.
    rng : numpy.random.Generator or int or None, optional
        Generator to draw from, or a seed for
        :func:`numpy.random.default_rng`.

    Returns
    -------
    numpy.ndarray
        Array of `n` positive rates, in onsets per week.

    Raises
    ------
    ValueError
        If `k` or `theta` is not positive, or if `n` is below one.

    Examples
    --------
    >>> gamma_rates(2.0, 0.005, 3, rng=0).shape
    (3,)
    """
    if k <= 0.0:
        raise ValueError(f"k must be positive, got {k!r}")
    if theta <= 0.0:
        raise ValueError(f"theta must be positive, got {theta!r}")
    if n < 1:
        raise ValueError(f"n must be positive, got {n!r}")
    generator = _generator(rng)
    drawn: npt.NDArray[np.float64] = generator.gamma(k, theta, n)
    return drawn


# The seven parameters are the published signature of the generator: three per
# patient quantities, the cohort size, the generator and the two record shaping
# options. Splitting them into an options object would hide the plain call.
def alternating_renewal(  # noqa: PLR0917
    lambda_remission: float | npt.ArrayLike,
    mu_relapse: float | npt.ArrayLike,
    t_end: float | npt.ArrayLike,
    n: int = 1,
    rng: Seed = None,
    discretise: Literal[None, "week"] = None,
    start_state: Literal["relapse", "health"] = "relapse",
) -> pd.DataFrame:
    """Simulate an alternating renewal record for a cohort of patients.

    Each patient starts at time zero in `start_state` and then alternates
    between relapse and remission, with relapse durations of mean
    1 / `mu_relapse` weeks and remission durations of mean
    1 / `lambda_remission` weeks, until time `t_end`. A relapse still running
    at `t_end` is truncated to end there; a remission still running at `t_end`
    is simply cut, since it leaves no further event in the record.

    Parameters
    ----------
    lambda_remission : float or array_like
        Rate at which a remission ends, per week. A scalar shared by the whole
        cohort or one positive value per patient.
    mu_relapse : float or array_like
        Rate at which a relapse ends, per week. A scalar shared by the whole
        cohort or one positive value per patient.
    t_end : float or array_like
        End of follow up in weeks, measured from time zero. A scalar shared by
        the whole cohort or one positive value per patient.
    n : int, optional
        Number of patients. Must be at least one.
    rng : numpy.random.Generator or int or None, optional
        Generator to draw from, or a seed for
        :func:`numpy.random.default_rng`.
    discretise : {None, 'week'}, optional
        With ``'week'`` durations are drawn from a geometric distribution on
        1, 2, 3, ... with the same mean, so that every duration is a whole
        number of weeks, as in the paper's weekly records. Each mean duration
        must then be at least one week. Whole weeks survive the truncation at
        `t_end` only when `t_end` itself is a whole number of weeks.
    start_state : {'relapse', 'health'}, optional
        State at time zero. The paper's records start at the onset of the first
        relapse, which is the default; starting in health instead gives a
        record whose first onset is not forced to time zero.

    Returns
    -------
    pandas.DataFrame
        An events table with the columns patient_id, followup_start,
        followup_end, relapse_onset and relapse_end, one row per relapse,
        sorted by patient_id then relapse_onset. Patient ids are p0001, p0002
        and so on, zero padded to at least four digits. A patient with no
        relapse inside the window keeps one row with a missing onset and end,
        so that the follow up window is not lost.

    Raises
    ------
    ValueError
        If `n` is below one, if `start_state` or `discretise` is unknown, if a
        per patient array does not have length `n`, if a rate or a window is
        not positive and finite, or if a mean duration is below one week while
        `discretise` is ``'week'``.

    Examples
    --------
    >>> events = alternating_renewal(0.02, 0.5, 500.0, n=2, rng=0)
    >>> sorted(set(events["patient_id"]))
    ['p0001', 'p0002']
    """
    if n < 1:
        raise ValueError(f"n must be positive, got {n!r}")
    if start_state not in _START_STATES:
        raise ValueError(f"start_state must be one of {_START_STATES}, got {start_state!r}")
    if discretise not in _DISCRETISATIONS:
        raise ValueError(f"discretise must be one of {_DISCRETISATIONS}, got {discretise!r}")

    rates_remission = _per_patient(lambda_remission, n, "lambda_remission")
    rates_relapse = _per_patient(mu_relapse, n, "mu_relapse")
    horizons = _per_patient(t_end, n, "t_end")
    remission_means = 1.0 / rates_remission
    relapse_means = 1.0 / rates_relapse
    if discretise == "week":
        _check_whole_week_mean(remission_means, "remission")
        _check_whole_week_mean(relapse_means, "relapse")

    generator = _generator(rng)
    records: list[tuple[str, float, float, float, float]] = []
    for index, patient in enumerate(_patient_ids(n)):
        horizon = float(horizons[index])
        remission_mean = float(remission_means[index])
        relapse_mean = float(relapse_means[index])
        rows_before = len(records)
        clock = 0.0
        if start_state == "health":
            clock = _draw_duration(generator, remission_mean, discretise)
        while clock < horizon:
            onset = clock
            stop = onset + _draw_duration(generator, relapse_mean, discretise)
            records.append((patient, 0.0, horizon, onset, min(stop, horizon)))
            if stop >= horizon:
                break
            clock = stop + _draw_duration(generator, remission_mean, discretise)
        if len(records) == rows_before:
            records.append((patient, 0.0, horizon, math.nan, math.nan))

    frame = pd.DataFrame(records, columns=list(_EVENTS_COLUMNS))
    return frame.astype(dict.fromkeys(_TIME_COLUMNS, np.float64))


def relapse_counts(events: pd.DataFrame, window: float | None = None) -> pd.Series[int]:
    """Count the relapse onsets of each patient in an events table.

    Parameters
    ----------
    events : pandas.DataFrame
        An events table, as returned by :func:`alternating_renewal`.
    window : float, optional
        Length in weeks of the window that starts at each patient's
        followup_start. An onset counts when
        followup_start <= relapse_onset < followup_start + `window`. The
        default counts the whole follow up.

    Returns
    -------
    pandas.Series
        Counts named 'relapses', of dtype int64, indexed by patient_id. Every
        patient of the table appears, with a count of zero when the patient has
        no onset in the window.

    Raises
    ------
    ValueError
        If `events` lacks a column of the schema, if it fails the schema
        validator of :mod:`msrelapse.io`, or if `window` is not positive.

    Examples
    --------
    >>> events = alternating_renewal(0.02, 0.5, 500.0, n=2, rng=0)
    >>> relapse_counts(events).name
    'relapses'
    """
    _validate_events(events)
    onset = events["relapse_onset"]
    inside = onset.notna()
    if window is not None:
        if window <= 0.0:
            raise ValueError(f"window must be positive, got {window!r}")
        start = events["followup_start"]
        inside &= (onset >= start) & (onset < start + window)
    counts = inside.groupby(events["patient_id"], sort=True).sum()
    return counts.astype(np.int64).rename("relapses").rename_axis("patient_id")


def _generator(rng: Seed) -> np.random.Generator:
    """Return the generator to draw from, building one from a seed if needed."""
    if isinstance(rng, np.random.Generator):
        return rng
    return np.random.default_rng(rng)


def _patient_ids(n: int) -> list[str]:
    """Return the ids p0001, p0002 and so on for a cohort of `n` patients."""
    digits = max(_MIN_ID_DIGITS, len(str(n)))
    return [f"p{number:0{digits}d}" for number in range(1, n + 1)]


def _per_patient(values: npt.ArrayLike, n: int, name: str) -> npt.NDArray[np.float64]:
    """Return `values` as one positive finite float per patient."""
    array = np.asarray(values, dtype=np.float64)
    if array.ndim == 0:
        array = np.full(n, float(array))
    elif array.ndim != 1 or array.size != n:
        raise ValueError(
            f"{name} must be a scalar or an array of length n = {n}, got shape {array.shape}"
        )
    offenders = np.flatnonzero(~(np.isfinite(array) & (array > 0.0)))
    if offenders.size > 0:
        first = int(offenders[0])
        raise ValueError(f"{name} must be positive and finite; entry {first} is {array[first]!r}")
    return array


def _check_whole_week_mean(means: npt.NDArray[np.float64], name: str) -> None:
    """Check that every mean duration can be drawn on whole weeks."""
    offenders = np.flatnonzero(means < 1.0)
    if offenders.size > 0:
        first = int(offenders[0])
        raise ValueError(
            f"discretise='week' needs a mean {name} duration of at least one week; "
            f"entry {first} has mean {means[first]!r} weeks"
        )


def _draw_duration(
    generator: np.random.Generator,
    mean: float,
    discretise: Literal[None, "week"],
) -> float:
    """Draw one duration in weeks, exponential or geometric with the given mean.

    The geometric branch assumes `mean` is at least one week, which the caller
    checks once for the whole cohort.
    """
    if discretise is None:
        return float(generator.exponential(mean))
    return float(generator.geometric(1.0 / mean))


def _validate_events(events: pd.DataFrame) -> None:
    """Check that `events` follows the events schema.

    The missing columns are named here first, so that a table lacking one of
    them raises a plain message instead of the key error that selecting the
    schema columns would give.

    Only the five schema columns are handed to :func:`msrelapse.io.validate`,
    because it rejects any column it does not know and a caller may well be
    counting a table that carries extra columns, such as the dated export of
    :func:`msrelapse.io.weekly_to_events`. Every other rule of the schema, on
    dtypes, ordering, overlap and follow up windows, still runs.
    """
    missing = [name for name in _EVENTS_COLUMNS if name not in events.columns]
    if missing:
        raise ValueError(f"events table is missing the columns {missing}")
    validate(events[list(_EVENTS_COLUMNS)], "events")
