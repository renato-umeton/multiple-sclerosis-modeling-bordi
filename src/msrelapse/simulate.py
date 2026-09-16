"""Integration of the stochastic equation of motion and the records it produces.

Equation (4) of the paper is the stochastic differential equation::

    dx = [x (1 - alpha x^2) - beta] dt + sigma dW

with the model time read as weeks. This module integrates it with the Euler
Maruyama scheme::

    x_{k+1} = x_k + force(x_k) dt + sigma sqrt(dt) z_k

where ``z_k`` is a standard normal draw. The noise is additive, so the scheme
has strong order one and no Milstein correction term exists to add.

On top of the integrator sit the four steps that turn a continuous path into
the kind of record the paper prints: first passage times out of a well, the
mapping of a path to the two clinical states through a hysteresis band, the
downsampling to whole weeks under the rounding rule of the study, and the run
length encoding of the weekly record.

Three choices are made here that the paper leaves open, and none of them comes
from the article.

Time step
    The cubic drift is not globally Lipschitz, so a step that is too long sends
    the path to infinity instead of into a well. A convergence study at the
    calibrated parameters found that a step of half a week blows paths up while
    steps up to [`MAX_DT`][msrelapse.simulate.MAX_DT] produced no non finite
    path, which is where the ceiling of this module comes from.
Absorbing level
    Looking for a crossing only at the grid points misses the excursions
    between them, which overestimates an exit time by a term of order
    ``sqrt(dt)``. Moving the absorbing level towards the walker by
    [`BRIDGE_CONSTANT`][msrelapse.simulate.BRIDGE_CONSTANT] ``sigma sqrt(dt)``,
    the Brownian bridge correction, cancels that term and lets a step of 0.02
    weeks stand in for one of 0.001. Both
    [`exit_times`][msrelapse.simulate.exit_times] and
    [`to_states`][msrelapse.simulate.to_states] carry it, the first on its
    single absorbing level and the second on both thresholds of the band, and
    [`simulate_weekly`][msrelapse.simulate.simulate_weekly] turns it on by
    default.
Hysteresis band
    A bare threshold at the saddle counts every wobble of the path across the
    barrier top as a relapse. Two thresholds placed a fraction of the way from
    the saddle towards each well bottom remove those spurious switches, and the
    same fraction passed to
    [`msrelapse.model.calibrate`][msrelapse.model.calibrate] with
    ``passage='band'`` makes the simulated durations approach the calibration
    targets as the step shrinks. Read on the grid alone, both thresholds sit
    further from the walker than they are written, so an episode starts a step
    deep inside one state and ends a step past the other threshold and runs
    long at both ends. Measured on the band calibrated potential, over 50 paths
    of 12000 weeks holding about 5700 complete episodes of each side, the mean
    episode ran about 13 percent above its target on the relapse side and about
    11 percent above it on the health side at a step of 0.02 weeks. Passing the
    same Brownian bridge shift to [`to_states`][msrelapse.simulate.to_states]
    removes them: over twenty seeds of that run the corrected relapse mean
    stayed between 0.99 and 1.03 of its target and the corrected health mean
    between 0.95 and 1.01, the remaining deficit being the fixed window rather
    than the grid, since a window of a given length holds fewer of the long
    episodes than of the short ones and the pooled mean of the complete ones
    under-weights them.

The optional numba kernels are exactly that, optional:
[`HAS_NUMBA`][msrelapse.simulate.HAS_NUMBA] says whether they are available, a
pure numpy implementation runs whenever they are not, and a missing numba never
breaks the import.

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
from typing import Final, TypeVar

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray

from msrelapse._params import PAPER
from msrelapse.io import validate, weekly_to_durations
from msrelapse.model import DEFAULT_BAND_FRACTION, DoubleWell, Passage, Side, passage_endpoints

try:
    import numba

    HAS_NUMBA = True
    """bool: Whether the optional numba kernels could be compiled."""
except ImportError:  # pragma: no cover - depends on the installed extras
    HAS_NUMBA = False

__all__ = [
    "BRIDGE_CONSTANT",
    "HAS_NUMBA",
    "MAX_DT",
    "Paths",
    "Seed",
    "durations",
    "exit_times",
    "simulate_paths",
    "simulate_weekly",
    "to_states",
    "to_weekly",
]

Seed = np.random.Generator | int | None
"""What every random operation of this module accepts in its ``rng`` argument."""

_Kernel = Callable[..., None]
"""One inner loop, in its numpy and its numba spelling."""

_Scalar = TypeVar("_Scalar", bound=np.generic)
"""The numpy scalar type a series of records is read in."""

MAX_DT: Final = 0.3
"""Longest time step, in weeks, at which the scheme was found to stay finite.

An independent convergence study at the calibrated parameters sent paths to
infinity at a step of 0.5 weeks and saw no non finite path at steps up to this
value. The number is a property of the scheme and the potential, not of the
paper.
"""

BRIDGE_CONSTANT: Final = 0.5826
"""Mean overshoot of a Brownian bridge past a level, in units of ``sigma sqrt(dt)``.

The constant is ``-zeta(1/2) / sqrt(2 pi)``. A walk watched only at the grid
points behaves like a continuously watched walk whose level sits this much
further away, so bringing a simulated level this much closer to the walker
cancels the ``sqrt(dt)`` bias of a crossing test evaluated on the grid. Callers
of [`to_states`][msrelapse.simulate.to_states] pass
``BRIDGE_CONSTANT * sigma * sqrt(dt)`` as its ``level_shift``, which is what
[`simulate_weekly`][msrelapse.simulate.simulate_weekly] and
[`exit_times`][msrelapse.simulate.exit_times] do for themselves.
"""

# Normal draws are generated a block of steps at a time for all paths at once,
# which keeps the peak memory of a long run bounded. Neither bound changes the
# paths of simulate_paths: the generator fills the block in step order, so any
# blocking gives the same stream.
_MAX_CHUNK_STEPS: Final = 2048
_MAX_CHUNK_VALUES: Final = 2**22

# A count of steps or of weeks this close to a whole number is taken to be that
# whole number, so that a division such as 1.0 / 0.02 does not lose or gain a
# step to the last bit of a double.
_GRID_TOLERANCE: Final = 1e-9

_MIN_ID_DIGITS: Final = 4

_NO_HEALTH: Final = PAPER.state_no_health.value
_HEALTH: Final = PAPER.state_health.value
_WEEK: Final = PAPER.time_resolution_weeks.value


@dataclass(frozen=True)
class Paths:
    """A recorded solution of equation (4) for one or more paths.

    Attributes
    ----------
    t : numpy.ndarray
        The recorded times in weeks, of shape ``(n_records,)``, starting at
        zero and spaced by ``record_every`` steps.
    x : numpy.ndarray
        The recorded positions, of shape ``(n_paths, n_records)``.
    """

    t: NDArray[np.float64]
    x: NDArray[np.float64]


def simulate_paths(  # noqa: PLR0917
    well: DoubleWell,
    sigma: float,
    t_end: float,
    dt: float = 0.01,
    n_paths: int = 1,
    x0: float | NDArray[np.float64] | None = None,
    rng: Seed = None,
    record_every: int = 1,
    use_numba: bool | None = None,
) -> Paths:
    """Integrate equation (4) with the Euler Maruyama scheme.

    The path is advanced by ``ceil(t_end / dt)`` steps and recorded at time
    zero and then every `record_every` steps, so the last record sits at
    ``(ceil(t_end / dt) // record_every) * record_every * dt``. That is at or
    just past `t_end` when `record_every` divides the step count, which is
    always so at the default of one, and short of it otherwise: the steps of
    the unfinished last block are taken but not recorded.

    Parameters
    ----------
    well : DoubleWell
        The potential whose drift the path follows.
    sigma : float
        Noise amplitude, the square root of the paper's variance epsilon. Zero
        is allowed and gives the deterministic limit.
    t_end : float
        Length of the run, in weeks. Must be positive.
    dt : float, optional
        Time step, in weeks. Must be positive and no larger than
        [`MAX_DT`][msrelapse.simulate.MAX_DT].
    n_paths : int, optional
        Number of independent paths. Must be at least one.
    x0 : float or numpy.ndarray, optional
        Starting position, a scalar shared by every path or one value per path.
        Defaults to the bottom of the health well.
    rng : numpy.random.Generator or int or None, optional
        Generator to draw the normal increments from, or a seed for
        ``numpy.random.default_rng``.
    record_every : int, optional
        Keep one record every this many steps. Must be at least one.
    use_numba : bool or None, optional
        Whether to run the compiled kernel. The default uses it when numba is
        importable.

    Returns
    -------
    Paths
        The recorded times and positions, both float64.

    Raises
    ------
    ValueError
        If `dt` is not positive or is larger than
        [`MAX_DT`][msrelapse.simulate.MAX_DT], if `t_end` is not positive, if
        `n_paths` or `record_every` is below one, if `sigma` is negative, if
        `x0` is an array of a length other than `n_paths`, if `use_numba` is
        True while numba is not installed, or if any path left the finite range
        during the run.

    Notes
    -----
    [`MAX_DT`][msrelapse.simulate.MAX_DT] was measured at one potential and one
    noise amplitude, so it is not a ceiling that holds for every well and every
    `sigma`. A step inside it still sends paths to infinity at a larger noise,
    and a non-finite sample is neither a state nor an error further down the
    pipeline: it compares False against both hysteresis thresholds, so
    [`to_states`][msrelapse.simulate.to_states] would freeze the state and the
    weekly record would read as an ordinary clinical one. Every recorded path
    is therefore checked before it is returned.

    The potential and the noise amplitude are held fixed for the whole run:
    `well` is one frozen [`DoubleWell`][msrelapse.model.DoubleWell] and `sigma`
    is one number, and neither is read as a function of time. A within patient
    drift of the barrier or of the noise, a slowly varying beta(t) or sigma(t),
    is therefore outside what this module produces. The over-dispersion the
    package does carry is the between patient kind, the gamma mixture of
    relapse rates of
    [`msrelapse.renewal.gamma_rates`][msrelapse.renewal.gamma_rates].

    Examples
    --------
    >>> paths = simulate_paths(DoubleWell(1.0, 0.08), 0.36, 1.0, dt=0.1, rng=0)
    >>> paths.t.shape, paths.x.shape
    ((11,), (1, 11))
    """
    _check_step(dt)
    if not math.isfinite(t_end) or t_end <= 0.0:
        raise ValueError(f"t_end must be a positive finite number of weeks, got {t_end!r}")
    if n_paths < 1:
        raise ValueError(f"n_paths must be at least 1, got {n_paths!r}")
    if record_every < 1:
        raise ValueError(f"record_every must be at least 1, got {record_every!r}")
    if not math.isfinite(sigma) or sigma < 0.0:
        raise ValueError(f"sigma must be a non-negative finite number, got {sigma!r}")

    n_steps = _step_count(t_end, dt)
    n_records = n_steps // record_every + 1
    x = _starting_positions(x0, n_paths, well)
    recorded = np.empty((n_paths, n_records), dtype=np.float64)
    recorded[:, 0] = x
    advance = _advance_kernel(use_numba)
    generator = _generator(rng)
    noise_step = sigma * math.sqrt(dt)
    step = 0
    # A step too coarse for the noise sends the cubic drift to infinity, and
    # numpy reports that on the way as an overflow warning and then, once the
    # path is no longer a number, an invalid value one. Both outcomes are
    # expected here rather than exceptional, and _check_paths_finite below is
    # what turns them into an error, so they are silenced at the one place they
    # are raised rather than through a filter over the whole run.
    with np.errstate(over="ignore", invalid="ignore"):
        while step < n_steps:
            chunk = min(_chunk_steps(n_paths), n_steps - step)
            noise = generator.standard_normal((chunk, n_paths))
            advance(x, noise, recorded, well.alpha, well.beta, dt, noise_step, step, record_every)
            step += chunk
    _check_paths_finite(recorded, well, sigma, dt)
    times = np.arange(n_records, dtype=np.float64) * (record_every * dt)
    return Paths(t=times, x=recorded)


def exit_times(  # noqa: PLR0917
    well: DoubleWell,
    sigma: float,
    side: Side,
    n_paths: int,
    dt: float = 0.02,
    rng: Seed = None,
    passage: Passage = "bottom_to_saddle",
    band_fraction: float = DEFAULT_BAND_FRACTION,
    bridge_correction: bool = True,
    max_time: float = 20000.0,
    use_numba: bool | None = None,
) -> NDArray[np.float64]:
    """Measure the first passage time of one episode, once per path.

    Every path starts at the first endpoint of
    [`msrelapse.model.passage_endpoints`][msrelapse.model.passage_endpoints]
    and is followed until it reaches the second one. A health passage runs
    towards increasing x and a relapse passage towards decreasing x, and the
    time recorded is the first grid time at which the absorbing level is
    crossed. Paths are retired as they are absorbed, so a cohort whose exit
    times are spread over decades costs little more than its mean.

    Parameters
    ----------
    well : DoubleWell
        The potential.
    sigma : float
        Noise amplitude, the square root of the paper's variance epsilon.
    side : {'health', 'relapse'}
        Which state the episode is spent in.
    n_paths : int
        Number of independent paths to measure. Must be at least one.
    dt : float, optional
        Time step, in weeks.
    rng : numpy.random.Generator or int or None, optional
        Generator to draw the normal increments from, or a seed for
        ``numpy.random.default_rng``.
    passage : {'bottom_to_saddle', 'bottom_to_bottom', 'band'}, optional
        Which crossing counts as one episode.
    band_fraction : float, optional
        Threshold position of the ``band`` passage, ignored otherwise. The
        default is
        [`msrelapse.model.DEFAULT_BAND_FRACTION`][msrelapse.model.DEFAULT_BAND_FRACTION];
        the cohort engine cuts its paths at the wider
        [`msrelapse.cohort.SDE_ENGINE_BAND_FRACTION`][msrelapse.cohort.SDE_ENGINE_BAND_FRACTION]
        instead.
    bridge_correction : bool, optional
        Whether to move the absorbing level towards the walker by
        [`BRIDGE_CONSTANT`][msrelapse.simulate.BRIDGE_CONSTANT]
        ``sigma sqrt(dt)``, which removes the ``sqrt(dt)`` bias of a crossing
        test evaluated only at the grid points. Without it a step of 0.1 weeks
        runs about a fifth long; with it the same step lands within a few
        percent of the exact answer.
    max_time : float, optional
        How long, in weeks, a path is followed before the measurement is
        declared a failure.
    use_numba : bool or None, optional
        Whether to run the compiled kernel. The default uses it when numba is
        importable.

    Returns
    -------
    numpy.ndarray
        One first passage time in weeks per path, of shape ``(n_paths,)``.

    Raises
    ------
    ValueError
        If `dt`, `n_paths`, `sigma`, `side`, `passage` or `band_fraction` is
        out of range, if `max_time` is not positive, if `use_numba` is True
        while numba is not installed, if the bridge correction is wider than
        the passage itself, or if any path is still running at `max_time`.

    See Also
    --------
    msrelapse.model.mfpt : The exact mean of the same passage, by quadrature.

    Notes
    -----
    Retiring the absorbed paths is what keeps a long run cheap, and it also
    means the draws a given path receives depend on how many of its neighbours
    are still running. A call is reproducible from its seed, but a call with a
    different `n_paths`, `dt` or `max_time` is a different sample rather than
    the same one lengthened.
    """
    _check_step(dt)
    if n_paths < 1:
        raise ValueError(f"n_paths must be at least 1, got {n_paths!r}")
    if not math.isfinite(sigma) or sigma < 0.0:
        raise ValueError(f"sigma must be a non-negative finite number, got {sigma!r}")
    if not math.isfinite(max_time) or max_time <= 0.0:
        raise ValueError(f"max_time must be a positive finite number of weeks, got {max_time!r}")

    start, absorb = passage_endpoints(well, side, passage, band_fraction)
    rising = side == "health"
    overshoot = BRIDGE_CONSTANT * sigma * math.sqrt(dt) if bridge_correction else 0.0
    level = absorb - overshoot if rising else absorb + overshoot
    collapsed = level <= start if rising else level >= start
    if collapsed:
        raise ValueError(
            f"the bridge correction of {overshoot:.6g} moves the absorbing level from "
            f"{absorb:.6g} to {level:.6g}, which is at or past the start of the passage at "
            f"{start:.6g}, so every path would be absorbed in one step; shorten dt or turn "
            f"bridge_correction off"
        )

    n_steps = _step_count(max_time, dt)
    advance = _exit_kernel(use_numba)
    generator = _generator(rng)
    noise_step = sigma * math.sqrt(dt)
    x = np.full(n_paths, start, dtype=np.float64)
    active = np.arange(n_paths, dtype=np.int64)
    steps_taken = np.full(n_paths, -1, dtype=np.int64)
    step = 0
    while active.size and step < n_steps:
        chunk = min(_chunk_steps(active.size), n_steps - step)
        noise = generator.standard_normal((chunk, active.size))
        crossed = np.full(active.size, -1, dtype=np.int64)
        advance(x, noise, crossed, well.alpha, well.beta, dt, noise_step, level, rising, step)
        done = crossed >= 0
        if done.any():
            steps_taken[active[done]] = crossed[done]
            keep = ~done
            active = active[keep]
            x = np.ascontiguousarray(x[keep])
        step += chunk
    unfinished = int(np.count_nonzero(steps_taken < 0))
    if unfinished:
        raise ValueError(
            f"{unfinished} of {n_paths} paths had not reached x={level:.6g} within "
            f"max_time={max_time!r} weeks, so their {side} exit times are unknown; raise "
            f"max_time or check that sigma={sigma!r} can carry the path over the barrier"
        )
    return steps_taken.astype(np.float64) * dt


def to_states(
    x: NDArray[np.float64],
    well: DoubleWell,
    band_fraction: float = DEFAULT_BAND_FRACTION,
    initial: int | None = None,
    level_shift: float = 0.0,
) -> NDArray[np.int64]:
    """Map a path to the two clinical states through a hysteresis band.

    The path switches to no health only once it climbs above the relapse
    threshold and back to health only once it drops below the health
    threshold, so the band between the two thresholds carries no switch at all.
    Both thresholds come from ``passage_endpoints(well, 'health', 'band',
    band_fraction)``.

    Parameters
    ----------
    x : numpy.ndarray
        One path, of shape ``(n_records,)``, or a cohort of paths, of shape
        ``(n_paths, n_records)``.
    well : DoubleWell
        The potential the path was drawn from, which fixes the thresholds.
    band_fraction : float, optional
        Position of the two thresholds, as a fraction of the distance from the
        saddle to each well bottom. Must lie strictly between 0 and 1. The
        default is
        [`msrelapse.model.DEFAULT_BAND_FRACTION`][msrelapse.model.DEFAULT_BAND_FRACTION];
        the cohort engine cuts its paths at the wider
        [`msrelapse.cohort.SDE_ENGINE_BAND_FRACTION`][msrelapse.cohort.SDE_ENGINE_BAND_FRACTION]
        instead.
    initial : int, optional
        State to start every path in. The default starts a path in health when
        its first sample lies below the saddle and in no health otherwise.
    level_shift : float, optional
        The Brownian bridge correction for crossings read off a grid, in units
        of x. It moves the relapse entry threshold down by this much and the
        health entry threshold up by it, which brings each of them towards the
        walker that has to reach it. Must be finite and not negative, and it
        must be narrow enough to leave the two thresholds apart. Pass
        ``BRIDGE_CONSTANT * sigma * sqrt(dt)`` for the path this is mapping;
        the default of 0 leaves the thresholds where `band_fraction` puts them.

    Returns
    -------
    numpy.ndarray
        The states, of the same shape as `x` and of dtype int64, holding the
        two codes of ``PAPER.state_health`` and ``PAPER.state_no_health``.

    Raises
    ------
    ValueError
        If `x` is not one or two dimensional, is empty or holds a sample that
        is not a finite number, if `band_fraction` is not strictly between 0
        and 1, if `initial` is neither of the two state codes, or if
        `level_shift` is negative, is not finite, or is wide enough to close
        the band.

    Notes
    -----
    A sample that is not a finite number compares False against both
    thresholds, so it would neither switch the state nor raise. Such a path is
    refused by name instead, which also catches a caller who brings paths of
    their own from an integration that blew up.

    A threshold tested only at the grid points is crossed a step late, so a
    walk watched that way behaves like a continuously watched walk whose
    threshold sits [`BRIDGE_CONSTANT`][msrelapse.simulate.BRIDGE_CONSTANT]
    ``sigma sqrt(dt)`` further away. Left uncorrected that lengthens both ends
    of every episode, by about a tenth at the calibrated parameters and a step
    of 0.02 weeks; see the module docstring for the measurement.

    The shift brings the two thresholds together and does not move them apart,
    which is the direction [`exit_times`][msrelapse.simulate.exit_times]
    already moves its single level in. Measured on 50 records of 12000 weeks at
    a step of 0.02 weeks on the band calibrated potential, as the ratio of the
    mean complete episode to the exact first passage time over the same band:
    1.113 in health and 1.128 in no health with no shift, 0.988 and 1.005 with
    the thresholds brought together as they are here, and 1.233 and 1.244 with
    them moved apart, which roughly doubles the bias the shift is there to
    remove.

    Examples
    --------
    >>> well = DoubleWell(1.0, 0.08)
    >>> to_states(np.array([-1.0, 0.0, 1.0]), well).tolist()
    [-1, -1, 1]
    """
    rows = _as_rows(x, np.float64, "x")
    _check_samples_finite(rows, "x")
    return _to_states(rows, well, band_fraction, initial, level_shift).reshape(np.shape(x))


def _to_states(
    values: NDArray[np.float64],
    well: DoubleWell,
    band_fraction: float,
    initial: int | None,
    level_shift: float,
) -> NDArray[np.int64]:
    """Map rows of samples already known to be finite to the two clinical states.

    This is [`to_states`][msrelapse.simulate.to_states] without the scan for a
    sample that is not a finite number. The scan walks the whole integration
    grid and allocates a boolean temporary of the size of it, which is worth
    paying once for samples a caller brought from somewhere else and not worth
    paying twice for samples
    [`simulate_paths`][msrelapse.simulate.simulate_paths] has just checked. The
    rows come in already converted by ``_as_rows``, so that a path is reshaped
    once per call and one place decides how, and the caller puts the answer
    back in the shape it asked with.

    Parameters
    ----------
    values : numpy.ndarray
        A cohort of paths of shape ``(n_paths, n_records)``, holding finite
        samples only.
    well : DoubleWell
        The potential the path was drawn from, which fixes the thresholds.
    band_fraction : float
        Position of the two thresholds.
    initial : int or None
        State to start every path in, or None to read it off the first sample.
    level_shift : float
        The Brownian bridge correction, in units of x.

    Returns
    -------
    numpy.ndarray
        The states, of the same shape as `values` and of dtype int64.

    Raises
    ------
    ValueError
        If `band_fraction` is not strictly between 0 and 1, if `initial` is
        neither of the two state codes, or if `level_shift` is not a distance
        the band can carry.
    """
    threshold_health, threshold_relapse = passage_endpoints(well, "health", "band", band_fraction)
    threshold_health, threshold_relapse = _shifted_thresholds(
        threshold_health, threshold_relapse, level_shift
    )
    # A boolean is refused before the membership test, because True equals the
    # no health code and would otherwise start every path in relapse.
    if initial is not None and (
        isinstance(initial, bool | np.bool_) or initial not in (_HEALTH, _NO_HEALTH)
    ):
        raise ValueError(
            f"initial must be {_HEALTH:+d} for health or {_NO_HEALTH:+d} for no health, "
            f"got {initial!r}"
        )
    if initial is None:
        saddle = well.critical_points().saddle
        start = np.where(values[:, 0] < saddle, _HEALTH, _NO_HEALTH).astype(np.int64)
    else:
        start = np.full(values.shape[0], initial, dtype=np.int64)
    states = np.empty(values.shape, dtype=np.int64)
    _states_kernel()(values, states, threshold_health, threshold_relapse, start)
    return states


def to_weekly(
    states: NDArray[np.int64],
    dt: float,
    week: float = _WEEK,
) -> NDArray[np.int64]:
    """Downsample a state series to whole weeks under the rounding rule.

    Week ``k`` covers the samples whose time lies in ``[k week, (k + 1) week)``
    and is a relapse week when any sample inside it is a relapse, which is the
    rule the study applies to its own records: "shorter exacerbations have been
    rounded up to one week". Only complete weeks are returned.

    Parameters
    ----------
    states : numpy.ndarray
        One state series, of shape ``(n_records,)``, or a cohort of them, of
        shape ``(n_paths, n_records)``. Sample ``j`` is taken to sit at time
        ``j dt``.
    dt : float
        Spacing of the samples, in weeks. Must be positive.
    week : float, optional
        Length of a week in the same units, which the paper's weekly scale puts
        at one. Must be positive.

    Returns
    -------
    numpy.ndarray
        The weekly states, of dtype int64, with the last axis of length
        ``floor(n_records dt / week)``.

    Raises
    ------
    ValueError
        If `states` is not one or two dimensional or is empty, if `dt` or
        `week` is not positive, if `dt` is longer than `week`, or if the record
        does not hold one complete week.

    Examples
    --------
    >>> steps = np.array([-1, -1, -1, -1, -1, 1, -1, -1])
    >>> to_weekly(steps, 0.25).tolist()
    [-1, 1]
    """
    values = _as_rows(states, np.int64, "states")
    if not math.isfinite(dt) or dt <= 0.0:
        raise ValueError(f"dt must be a positive finite number of weeks, got {dt!r}")
    if not math.isfinite(week) or week <= 0.0:
        raise ValueError(f"week must be a positive finite number of weeks, got {week!r}")
    if dt > week:
        raise ValueError(
            f"dt={dt!r} is longer than week={week!r}, so a week could hold no sample at all "
            f"and its state would be unknown; downsample to a week that holds at least one "
            f"sample"
        )
    n_records = values.shape[1]
    n_weeks = math.floor(_snap(n_records * dt / week))
    if n_weeks < 1:
        raise ValueError(
            f"a record of {n_records} samples of dt={dt!r} weeks spans "
            f"{n_records * dt:.6g} weeks, which is not one complete week of {week!r}"
        )
    starts = np.array([_grid_index(index * week, dt) for index in range(n_weeks)])
    end = _grid_index(n_weeks * week, dt)
    weekly = np.maximum.reduceat(values[:, :end], starts, axis=1)
    return weekly.reshape((*np.shape(states)[:-1], n_weeks))


def durations(
    states: NDArray[np.int64],
    patient_ids: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Run length encode a weekly record into one row per episode.

    Parameters
    ----------
    states : numpy.ndarray
        Weekly states, of shape ``(n_weeks,)`` for a single patient or
        ``(n_patients, n_weeks)`` for a cohort.
    patient_ids : sequence of str, optional
        One identifier per row, which must be distinct and in ascending order.
        Defaults to p0001, p0002 and so on.

    Returns
    -------
    pandas.DataFrame
        A frame in the durations schema of [`msrelapse.io`][msrelapse.io], in
        which ``censored`` is True only on a final remission.

    Raises
    ------
    ValueError
        If `states` is not one or two dimensional or is empty, if
        `patient_ids` does not have one entry per row, or if the record breaks
        the weekly schema.

    Examples
    --------
    >>> runs = durations(np.array([-1, -1, 1, -1]))
    >>> runs["duration_w"].tolist()
    [2, 1, 1]
    """
    return weekly_to_durations(_weekly_frame(states, patient_ids))


def simulate_weekly(  # noqa: PLR0917
    well: DoubleWell,
    sigma: float,
    n_weeks: int,
    n_paths: int = 1,
    dt: float = 0.02,
    rng: Seed = None,
    band_fraction: float = DEFAULT_BAND_FRACTION,
    x0: float | NDArray[np.float64] | None = None,
    patient_ids: Sequence[str] | None = None,
    use_numba: bool | None = None,
    bridge_correction: bool = True,
) -> pd.DataFrame:
    """Simulate a cohort and return its weekly record.

    This is [`simulate_paths`][msrelapse.simulate.simulate_paths],
    [`to_states`][msrelapse.simulate.to_states] and
    [`to_weekly`][msrelapse.simulate.to_weekly] in a row, and it is the entry
    point a cohort engine calls.

    Parameters
    ----------
    well : DoubleWell
        The potential. Calibrating it with ``passage='band'`` and the same
        `band_fraction` brings the simulated episode durations towards the
        calibration targets, the more closely the shorter `dt` is; see Notes.
    sigma : float
        Noise amplitude, the square root of the paper's variance epsilon.
    n_weeks : int
        Length of the record, in whole weeks. Must be at least one.
    n_paths : int, optional
        Number of patients. Must be at least one.
    dt : float, optional
        Time step, in weeks.
    rng : numpy.random.Generator or int or None, optional
        Generator to draw the normal increments from, or a seed for
        ``numpy.random.default_rng``.
    band_fraction : float, optional
        Position of the two hysteresis thresholds. The default is
        [`msrelapse.model.DEFAULT_BAND_FRACTION`][msrelapse.model.DEFAULT_BAND_FRACTION];
        the cohort engine cuts its paths at the wider
        [`msrelapse.cohort.SDE_ENGINE_BAND_FRACTION`][msrelapse.cohort.SDE_ENGINE_BAND_FRACTION]
        instead.
    x0 : float or numpy.ndarray, optional
        Starting position of each path. Defaults to the bottom of the health
        well, so every patient starts in remission.
    patient_ids : sequence of str, optional
        One identifier per patient. Defaults to p0001, p0002 and so on.
    use_numba : bool or None, optional
        Whether to run the compiled integration kernel. The default uses it
        when numba is importable. The hysteresis kernel of
        [`to_states`][msrelapse.simulate.to_states] is always the compiled one
        when numba is importable, because
        [`to_states`][msrelapse.simulate.to_states] takes no such argument.
    bridge_correction : bool, optional
        Whether to hand [`to_states`][msrelapse.simulate.to_states] the
        Brownian bridge shift
        [`BRIDGE_CONSTANT`][msrelapse.simulate.BRIDGE_CONSTANT]
        ``sigma sqrt(dt)`` computed from this call's own `sigma` and `dt`,
        which removes the ``sqrt(dt)`` bias of a crossing tested only at the
        grid points. On by default; see Notes for what it is worth and for what
        it does not cover.

    Returns
    -------
    pandas.DataFrame
        A frame in the weekly schema of [`msrelapse.io`][msrelapse.io], holding
        `n_weeks` rows per patient.

    Raises
    ------
    ValueError
        If `n_weeks` is below one, if any argument of
        [`simulate_paths`][msrelapse.simulate.simulate_paths] is out of range,
        if `patient_ids` does not have one entry per patient, or if the bridge
        correction is wide enough to close the hysteresis band.

    Notes
    -----
    Two effects stand between a band calibrated potential and the weekly record
    this returns, and a caller correcting the record back to the calibration
    targets has to allow for both.

    The rounding rule of [`to_weekly`][msrelapse.simulate.to_weekly] lengthens
    every relapse: a relapse of continuous length L is marked in every week it
    touches and so occupies L + 1 whole weeks on average, and the remissions
    around it lose that week. A weekly record therefore reports a relapse
    burden shifted by about a week per relapse from the continuous durations
    the potential was calibrated to.

    Two relapses separated by less than a week fall in the same week and merge
    into a single weekly episode, which is why a weekly record holds fewer and
    longer episodes than the path it came from. The hysteresis band removes the
    chatter of a path wobbling across the barrier top, but it cannot remove a
    genuine short return to health. On the potential
    [`msrelapse.model.calibrate`][msrelapse.model.calibrate] returns for 100
    and 4.3 weeks with ``passage='band'`` and a band fraction of 0.3, the
    default here, about one remission in six is shorter than a week: measured
    over 20 paths of 20000 weeks at a step of 0.02 weeks and the three seeds 3,
    7 and 11, one in 6.1 to 6.6 of the complete remissions of the path. A wider
    band leaves fewer of them, which is why
    [`msrelapse.cohort.CohortSpec`][msrelapse.cohort.CohortSpec] widens it; the
    table is in the Notes of that class. The mean weekly episode duration is
    inflated by that merging while the weekly relapse burden is not, so a
    caller comparing durations with the paper should compare burdens rather
    than per episode means.

    A third effect used to sit beside them and no longer does. Read on the
    integration grid alone the two band thresholds are crossed a step late, and
    at the default step of 0.02 weeks that put the mean episode of the path
    about 13 percent above its target on the relapse side and about 11 percent
    above it on the health side. `bridge_correction` removes it before the
    rounding rule is applied. Turning it off restores the older behaviour, for
    a caller comparing with a record made before the correction existed.

    The whole integration grid is held in memory for the whole cohort, twice
    over: ``n_paths * n_weeks / dt`` samples as float64 and the same count
    again as int64. A hundred patients over 1500 weeks at a step of 0.02 weeks
    peaked at about 150 MB, and the cost grows linearly in every one of the
    three, so a cohort of a thousand patients of that length needs about 1.5
    GB. The `record_every` argument of
    [`simulate_paths`][msrelapse.simulate.simulate_paths] is no way out,
    because the rounding rule needs every sub-week sample. Split a larger
    cohort into batches of patients and concatenate the frames.

    Examples
    --------
    >>> frame = simulate_weekly(DoubleWell(1.0, 0.08), 0.36, 5, n_paths=2, rng=0)
    >>> frame["week"].tolist()
    [0, 1, 2, 3, 4, 0, 1, 2, 3, 4]
    """
    if n_weeks < 1:
        raise ValueError(f"n_weeks must be at least 1, got {n_weeks!r}")
    paths = simulate_paths(
        well,
        sigma,
        float(n_weeks) * _WEEK,
        dt=dt,
        n_paths=n_paths,
        x0=x0,
        rng=rng,
        use_numba=use_numba,
    )
    level_shift = BRIDGE_CONSTANT * sigma * math.sqrt(dt) if bridge_correction else 0.0
    # The private mapping, because simulate_paths has just scanned these samples
    # for a value that is not finite and the public one would scan them again.
    states = _to_states(_as_rows(paths.x, np.float64, "x"), well, band_fraction, None, level_shift)
    return _weekly_frame(to_weekly(states, dt), patient_ids)


def _check_step(dt: float) -> None:
    """Raise if the time step is not positive or is long enough to blow a path up.

    Parameters
    ----------
    dt : float
        Time step, in weeks.

    Raises
    ------
    ValueError
        If `dt` is not a positive finite number, or if it exceeds
        [`MAX_DT`][msrelapse.simulate.MAX_DT].
    """
    if not math.isfinite(dt) or dt <= 0.0:
        raise ValueError(f"dt must be a positive finite number of weeks, got {dt!r}")
    if dt > MAX_DT:
        raise ValueError(
            f"dt must not exceed {MAX_DT} weeks, got {dt!r}; the cubic drift of equation (4) "
            f"is not globally Lipschitz, so the Euler Maruyama path blows up to infinity at a "
            f"longer step instead of settling into a well"
        )


def _shifted_thresholds(
    threshold_health: float,
    threshold_relapse: float,
    level_shift: float,
) -> tuple[float, float]:
    """Return the two band thresholds with the bridge correction applied.

    Parameters
    ----------
    threshold_health : float
        Below this the path enters health, before the correction.
    threshold_relapse : float
        Above this the path enters no health, before the correction.
    level_shift : float
        How far to bring each threshold towards the walker that has to reach
        it, in units of x.

    Returns
    -------
    tuple of float
        The corrected health and relapse thresholds, the first raised by
        `level_shift` and the second lowered by it.

    Raises
    ------
    ValueError
        If `level_shift` is not a finite non-negative number, or if it is wide
        enough to bring the two thresholds together.
    """
    if not math.isfinite(level_shift) or level_shift < 0.0:
        raise ValueError(f"level_shift must be a non-negative finite number, got {level_shift!r}")
    health = threshold_health + level_shift
    relapse = threshold_relapse - level_shift
    if health >= relapse:
        raise ValueError(
            f"a level_shift of {level_shift!r} moves the health threshold from "
            f"{threshold_health:.6g} to {health:.6g} and the relapse threshold from "
            f"{threshold_relapse:.6g} to {relapse:.6g}, which closes the hysteresis band and "
            f"would let a single sample sit in both states; shorten dt or widen band_fraction"
        )
    return health, relapse


def _snap(value: float) -> float:
    """Return `value` rounded to a whole number when it is one to within a tolerance.

    Parameters
    ----------
    value : float
        A count of steps or of weeks that a division may have moved off a whole
        number by the last bit of a double.

    Returns
    -------
    float
        The nearest whole number when `value` is that close to it, else `value`
        itself.
    """
    nearest = round(value)
    if abs(value - nearest) <= _GRID_TOLERANCE * max(1.0, abs(value)):
        return float(nearest)
    return value


def _step_count(span: float, dt: float) -> int:
    """Return how many steps of `dt` weeks cover a span, rounding up.

    Parameters
    ----------
    span : float
        Length of the run, in weeks.
    dt : float
        Time step, in weeks.

    Returns
    -------
    int
        The number of steps, at least one.
    """
    return max(1, math.ceil(_snap(span / dt)))


def _grid_index(time: float, dt: float) -> int:
    """Return the index of the first grid step at or after a time.

    Parameters
    ----------
    time : float
        A time in weeks, measured from the start of the record.
    dt : float
        Spacing of the grid, in weeks.

    Returns
    -------
    int
        The step index.
    """
    return math.ceil(_snap(time / dt))


def _chunk_steps(n_paths: int) -> int:
    """Return how many steps of normal draws to generate at a time.

    Parameters
    ----------
    n_paths : int
        Number of paths the block has to cover.

    Returns
    -------
    int
        A block length of at least one step, capped both in steps and in the
        number of values it holds.
    """
    return max(1, min(_MAX_CHUNK_STEPS, _MAX_CHUNK_VALUES // n_paths))


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


def _starting_positions(
    x0: float | NDArray[np.float64] | None,
    n_paths: int,
    well: DoubleWell,
) -> NDArray[np.float64]:
    """Return one starting position per path.

    Parameters
    ----------
    x0 : float or numpy.ndarray or None
        A scalar shared by every path, one value per path, or None for the
        bottom of the health well.
    n_paths : int
        Number of paths.
    well : DoubleWell
        The potential, which supplies the default.

    Returns
    -------
    numpy.ndarray
        A fresh contiguous array of `n_paths` doubles, safe to advance in
        place.

    Raises
    ------
    ValueError
        If `x0` is an array of a length other than `n_paths`.
    """
    if x0 is None:
        return np.full(n_paths, well.well_bottom("health"), dtype=np.float64)
    values = np.asarray(x0, dtype=np.float64)
    if values.ndim == 0:
        return np.full(n_paths, float(values), dtype=np.float64)
    if values.ndim != 1 or values.size != n_paths:
        raise ValueError(
            f"x0 must be a scalar or an array of length n_paths = {n_paths}, got shape "
            f"{values.shape}"
        )
    # A fresh copy every time: the integrator advances this array in place and
    # must never write into an array the caller still holds.
    return np.ascontiguousarray(values).astype(np.float64, copy=True)


def _as_rows(values: ArrayLike, dtype: type[_Scalar], name: str) -> NDArray[_Scalar]:
    """Return a series or a cohort of series as a contiguous two dimensional array.

    Parameters
    ----------
    values : array_like
        One series, of shape ``(n_records,)``, or several, of shape
        ``(n_series, n_records)``.
    dtype : type
        The numpy scalar type the rows are wanted in.
    name : str
        Name of the argument, used in the error message.

    Returns
    -------
    numpy.ndarray
        A contiguous array of shape ``(n_series, n_records)``.

    Raises
    ------
    ValueError
        If `values` is not one or two dimensional, or if it holds no record.
    """
    # The dimensionality is read before the conversion, because
    # ascontiguousarray promotes a single number to shape (1,) and the check
    # would then never see the zero.
    if np.ndim(values) not in (1, 2):
        raise ValueError(f"{name} must be one or two dimensional, got shape {np.shape(values)}")
    rows: NDArray[_Scalar] = np.atleast_2d(np.ascontiguousarray(values, dtype=dtype))
    if rows.shape[1] == 0:
        raise ValueError(f"{name} must hold at least one record, got shape {rows.shape}")
    return rows


def _non_finite_report(values: NDArray[np.float64]) -> tuple[int, int, int] | None:
    """Return how many rows hold a sample that is not a finite number, and the first.

    Parameters
    ----------
    values : numpy.ndarray
        Samples of shape ``(n_rows, n_records)``.

    Returns
    -------
    tuple of int or None
        The number of offending rows, the index of the first of them and the
        index of its first offending record, or None when every sample is
        finite.
    """
    offending = np.flatnonzero(~np.isfinite(values).all(axis=1))
    if offending.size == 0:
        return None
    row = int(offending[0])
    return int(offending.size), row, int(np.argmin(np.isfinite(values[row])))


def _check_samples_finite(values: NDArray[np.float64], name: str) -> None:
    """Raise if a series of samples holds a value that is not a finite number.

    Parameters
    ----------
    values : numpy.ndarray
        Samples of shape ``(n_paths, n_records)``.
    name : str
        Name of the argument the samples came in as, used in the message.

    Raises
    ------
    ValueError
        If any sample is not a finite number.
    """
    report = _non_finite_report(values)
    if report is None:
        return
    count, row, record = report
    raise ValueError(
        f"{name} must hold finite samples, but {count} of its {values.shape[0]} path(s) leave "
        f"the finite range, the first at record {record} of path {row}, which is "
        f"{values[row, record]}; such a sample is below neither threshold and above neither, "
        f"so the state would freeze instead of switching"
    )


def _check_paths_finite(
    recorded: NDArray[np.float64],
    well: DoubleWell,
    sigma: float,
    dt: float,
) -> None:
    """Raise if the integration sent any path out of the finite range.

    Parameters
    ----------
    recorded : numpy.ndarray
        The recorded positions, of shape ``(n_paths, n_records)``.
    well : DoubleWell
        The potential the paths were drawn from, named in the message.
    sigma : float
        Noise amplitude, named in the message.
    dt : float
        Time step, in weeks, named in the message.

    Raises
    ------
    ValueError
        If any path holds a sample that is not a finite number.
    """
    report = _non_finite_report(recorded)
    if report is None:
        return
    count, row, record = report
    raise ValueError(
        f"{count} of {recorded.shape[0]} paths left the finite range at dt={dt!r} with "
        f"sigma={sigma!r} on the potential with alpha={well.alpha!r} and beta={well.beta!r}, "
        f"the first at record {record} of path {row}; MAX_DT={MAX_DT} was measured at one "
        f"potential and one noise amplitude alone, so it is no ceiling for these, and dt has "
        f"to be shortened until the run stays finite"
    )


def _patient_ids(n_patients: int) -> list[str]:
    """Return the identifiers p0001, p0002 and so on for a cohort.

    Parameters
    ----------
    n_patients : int
        Number of patients.

    Returns
    -------
    list of str
        One identifier per patient, zero padded so that they sort in order.
    """
    digits = max(_MIN_ID_DIGITS, len(str(n_patients)))
    return [f"p{number:0{digits}d}" for number in range(1, n_patients + 1)]


def _weekly_frame(
    states: NDArray[np.int64],
    patient_ids: Sequence[str] | None,
) -> pd.DataFrame:
    """Build and validate a weekly frame out of an array of weekly states.

    Parameters
    ----------
    states : numpy.ndarray
        Weekly states, of shape ``(n_weeks,)`` or ``(n_patients, n_weeks)``.
    patient_ids : sequence of str or None
        One identifier per row, or None for p0001, p0002 and so on.

    Returns
    -------
    pandas.DataFrame
        A frame in the weekly schema of [`msrelapse.io`][msrelapse.io].

    Raises
    ------
    ValueError
        If `states` is not one or two dimensional or is empty, if
        `patient_ids` does not have one entry per row, or if the frame breaks
        the weekly schema.
    """
    rows = _as_rows(states, np.int64, "states")
    n_patients, n_weeks = rows.shape
    identifiers = _patient_ids(n_patients) if patient_ids is None else list(patient_ids)
    if len(identifiers) != n_patients:
        raise ValueError(
            f"patient_ids must have one entry per patient, got {len(identifiers)} for "
            f"{n_patients} row(s) of states"
        )
    frame = pd.DataFrame(
        {
            "patient_id": np.repeat(np.array(identifiers, dtype=object), n_weeks),
            "week": np.tile(np.arange(n_weeks, dtype=np.int64), n_patients),
            "state": rows.reshape(-1),
        }
    )
    validate(frame, "weekly")
    return frame


def _use_numba(flag: bool | None) -> bool:
    """Decide whether the compiled kernels run.

    Parameters
    ----------
    flag : bool or None
        What the caller asked for. None means use numba when it is importable.

    Returns
    -------
    bool
        Whether to call the compiled kernel.

    Raises
    ------
    ValueError
        If the caller asked for numba and it is not installed.
    """
    if flag is None:
        return HAS_NUMBA
    if flag and not HAS_NUMBA:
        raise ValueError(
            "use_numba=True needs the numba package, which is not installed; install the "
            "'fast' extra of msrelapse, or leave use_numba at None to fall back to numpy"
        )
    return flag


def _advance_numpy(  # noqa: PLR0917
    x: NDArray[np.float64],
    noise: NDArray[np.float64],
    recorded: NDArray[np.float64],
    alpha: float,
    beta: float,
    dt: float,
    noise_step: float,
    first_step: int,
    record_every: int,
) -> None:
    """Advance every path one block of Euler Maruyama steps, recording as it goes.

    Parameters
    ----------
    x : numpy.ndarray
        Current position of each path, advanced in place.
    noise : numpy.ndarray
        Standard normal draws of shape ``(n_steps, n_paths)``, consumed one row
        per step.
    recorded : numpy.ndarray
        Output array of shape ``(n_paths, n_records)``, written wherever a step
        falls on a record.
    alpha : float
        Control parameter of the potential.
    beta : float
        Asymmetry parameter of the potential.
    dt : float
        Time step, in weeks.
    noise_step : float
        ``sigma sqrt(dt)``, the standard deviation of one increment.
    first_step : int
        Index of the step already taken before this block.
    record_every : int
        Keep one record every this many steps.

    Returns
    -------
    None
        Nothing is returned; `x` and `recorded` are written in place.
    """
    for row in range(noise.shape[0]):
        x += (x * (1.0 - alpha * x * x) - beta) * dt
        x += noise_step * noise[row]
        step = first_step + row + 1
        if step % record_every == 0:
            recorded[:, step // record_every] = x


def _advance_loop(  # noqa: PLR0917
    x: NDArray[np.float64],
    noise: NDArray[np.float64],
    recorded: NDArray[np.float64],
    alpha: float,
    beta: float,
    dt: float,
    noise_step: float,
    first_step: int,
    record_every: int,
) -> None:
    """Advance every path one block of steps, one scalar update at a time.

    This is the same arithmetic as ``_advance_numpy`` in the same order, so
    that the compiled kernel and the numpy one produce the same paths from the
    same draws. It is written for numba and is far too slow to run as Python.

    Parameters
    ----------
    x : numpy.ndarray
        Current position of each path, advanced in place.
    noise : numpy.ndarray
        Standard normal draws of shape ``(n_steps, n_paths)``.
    recorded : numpy.ndarray
        Output array of shape ``(n_paths, n_records)``.
    alpha : float
        Control parameter of the potential.
    beta : float
        Asymmetry parameter of the potential.
    dt : float
        Time step, in weeks.
    noise_step : float
        ``sigma sqrt(dt)``, the standard deviation of one increment.
    first_step : int
        Index of the step already taken before this block.
    record_every : int
        Keep one record every this many steps.

    Returns
    -------
    None
        Nothing is returned; `x` and `recorded` are written in place.
    """
    n_steps, n_paths = noise.shape
    for row in range(n_steps):
        for path in range(n_paths):
            value = x[path]
            value = value + (value * (1.0 - alpha * value * value) - beta) * dt
            value = value + noise_step * noise[row, path]
            x[path] = value
        step = first_step + row + 1
        if step % record_every == 0:
            index = step // record_every
            for path in range(n_paths):
                recorded[path, index] = x[path]


def _exit_numpy(  # noqa: PLR0917
    x: NDArray[np.float64],
    noise: NDArray[np.float64],
    crossed: NDArray[np.int64],
    alpha: float,
    beta: float,
    dt: float,
    noise_step: float,
    level: float,
    rising: bool,
    first_step: int,
) -> None:
    """Advance every path one block of steps, recording the first crossing.

    The block is abandoned as soon as every path has crossed, so a block longer
    than the episode costs nothing.

    Parameters
    ----------
    x : numpy.ndarray
        Current position of each path, advanced in place.
    noise : numpy.ndarray
        Standard normal draws of shape ``(n_steps, n_paths)``.
    crossed : numpy.ndarray
        Output array of one entry per path, which arrives all negative and
        comes back holding, for each path that crossed, the step at which it
        first did.
    alpha : float
        Control parameter of the potential.
    beta : float
        Asymmetry parameter of the potential.
    dt : float
        Time step, in weeks.
    noise_step : float
        ``sigma sqrt(dt)``, the standard deviation of one increment.
    level : float
        The absorbing level, already shifted by the bridge correction.
    rising : bool
        Whether the passage runs towards increasing x.
    first_step : int
        Index of the step already taken before this block.

    Returns
    -------
    None
        Nothing is returned; `x` and `crossed` are written in place.
    """
    remaining = crossed.size
    for row in range(noise.shape[0]):
        x += (x * (1.0 - alpha * x * x) - beta) * dt
        x += noise_step * noise[row]
        reached = x >= level if rising else x <= level
        newly = reached & (crossed < 0)
        hits = int(np.count_nonzero(newly))
        if hits:
            crossed[newly] = first_step + row + 1
            remaining -= hits
            if remaining == 0:
                return


def _exit_loop(  # noqa: PLR0917
    x: NDArray[np.float64],
    noise: NDArray[np.float64],
    crossed: NDArray[np.int64],
    alpha: float,
    beta: float,
    dt: float,
    noise_step: float,
    level: float,
    rising: bool,
    first_step: int,
) -> None:
    """Advance every path one block of steps, one scalar update at a time.

    This is the same arithmetic as ``_exit_numpy`` in the same order. It is
    written for numba and is far too slow to run as Python.

    Parameters
    ----------
    x : numpy.ndarray
        Current position of each path, advanced in place.
    noise : numpy.ndarray
        Standard normal draws of shape ``(n_steps, n_paths)``.
    crossed : numpy.ndarray
        Output array of one entry per path, which arrives all negative, as in
        ``_exit_numpy``.
    alpha : float
        Control parameter of the potential.
    beta : float
        Asymmetry parameter of the potential.
    dt : float
        Time step, in weeks.
    noise_step : float
        ``sigma sqrt(dt)``, the standard deviation of one increment.
    level : float
        The absorbing level, already shifted by the bridge correction.
    rising : bool
        Whether the passage runs towards increasing x.
    first_step : int
        Index of the step already taken before this block.

    Returns
    -------
    None
        Nothing is returned; `x` and `crossed` are written in place.
    """
    n_steps, n_paths = noise.shape
    remaining = n_paths
    for row in range(n_steps):
        for path in range(n_paths):
            value = x[path]
            value = value + (value * (1.0 - alpha * value * value) - beta) * dt
            value = value + noise_step * noise[row, path]
            x[path] = value
            if crossed[path] < 0 and (value >= level if rising else value <= level):
                crossed[path] = first_step + row + 1
                remaining -= 1
        if remaining == 0:
            return


def _states_numpy(
    x: NDArray[np.float64],
    states: NDArray[np.int64],
    threshold_health: float,
    threshold_relapse: float,
    initial: NDArray[np.int64],
) -> None:
    """Apply the hysteresis band to every path, one time step at a time.

    Parameters
    ----------
    x : numpy.ndarray
        The paths, of shape ``(n_paths, n_records)``.
    states : numpy.ndarray
        Output array of the same shape, written in place.
    threshold_health : float
        Below this the path enters health.
    threshold_relapse : float
        Above this the path enters no health.
    initial : numpy.ndarray
        The state each path is in before its first sample.

    Returns
    -------
    None
        Nothing is returned; `states` is written in place.
    """
    current = initial.copy()
    for step in range(x.shape[1]):
        column = x[:, step]
        current = np.where(
            column > threshold_relapse,
            _NO_HEALTH,
            np.where(column < threshold_health, _HEALTH, current),
        )
        states[:, step] = current


def _states_loop(
    x: NDArray[np.float64],
    states: NDArray[np.int64],
    threshold_health: float,
    threshold_relapse: float,
    initial: NDArray[np.int64],
) -> None:
    """Apply the hysteresis band to every path, one scalar at a time.

    This is the same rule as ``_states_numpy``. It is written for numba and
    is far too slow to run as Python.

    Parameters
    ----------
    x : numpy.ndarray
        The paths, of shape ``(n_paths, n_records)``.
    states : numpy.ndarray
        Output array of the same shape, written in place.
    threshold_health : float
        Below this the path enters health.
    threshold_relapse : float
        Above this the path enters no health.
    initial : numpy.ndarray
        The state each path is in before its first sample.

    Returns
    -------
    None
        Nothing is returned; `states` is written in place.
    """
    n_paths, n_records = x.shape
    for path in range(n_paths):
        current = initial[path]
        for step in range(n_records):
            value = x[path, step]
            if value > threshold_relapse:
                current = _NO_HEALTH
            elif value < threshold_health:
                current = _HEALTH
            states[path, step] = current


if HAS_NUMBA:
    _ADVANCE_FAST: _Kernel = numba.njit(cache=True)(_advance_loop)
    _EXIT_FAST: _Kernel = numba.njit(cache=True)(_exit_loop)
    _STATES_FAST: _Kernel = numba.njit(cache=True)(_states_loop)


def _advance_kernel(use_numba: bool | None) -> _Kernel:
    """Return the integration kernel the caller asked for.

    Parameters
    ----------
    use_numba : bool or None
        Whether to run the compiled kernel, or None to use it when available.

    Returns
    -------
    callable
        Either the compiled kernel or the numpy one.

    Raises
    ------
    ValueError
        If the caller asked for numba and it is not installed.
    """
    return _ADVANCE_FAST if _use_numba(use_numba) else _advance_numpy


def _exit_kernel(use_numba: bool | None) -> _Kernel:
    """Return the first passage kernel the caller asked for.

    Parameters
    ----------
    use_numba : bool or None
        Whether to run the compiled kernel, or None to use it when available.

    Returns
    -------
    callable
        Either the compiled kernel or the numpy one.

    Raises
    ------
    ValueError
        If the caller asked for numba and it is not installed.
    """
    return _EXIT_FAST if _use_numba(use_numba) else _exit_numpy


def _states_kernel() -> _Kernel:
    """Return the hysteresis kernel, compiled when numba is installed.

    Returns
    -------
    callable
        Either the compiled kernel or the numpy one.
    """
    return _STATES_FAST if HAS_NUMBA else _states_numpy
