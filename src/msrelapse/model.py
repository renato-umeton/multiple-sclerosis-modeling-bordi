"""The double well potential of Bordi et al. 2013 and the times to cross it.

The model is a single particle in a tilted double well, equation (4) of the
paper::

    dx = [x (1 - alpha x^2) - beta] dt + sigma dW

whose drift is the negative gradient of the potential of equations (2) and (3)::

    V(x) = -x^2 / 2 + alpha x^4 / 4 + beta x

The left minimum is the state of health (remission), the right minimum the
state of no health (relapse), and the middle stationary point is the barrier
top that separates them. The paper writes the noise term as ``epsilon^(1/2)
dw`` with ``epsilon`` the noise variance; this package uses ``sigma``
throughout, with ``sigma**2 = epsilon``.

This module holds everything that follows from the potential alone: the wells,
the barriers, the mean exit time estimators of equations (5) to (7), the exact
mean first passage time of the stochastic equation, and the calibration that
turns a pair of observed episode durations into a pair of model parameters.

References
----------
I. Bordi, R. Umeton, V. A. G. Ricigliano, et al., "A mechanistic, stochastic
model helps understand multiple sclerosis course and pathogenesis",
International Journal of Genomics, 2013, doi 10.1155/2013/910321.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final, Literal, overload

import numpy as np
from numpy.typing import NDArray
from scipy.integrate import cumulative_trapezoid, trapezoid
from scipy.optimize import brentq, least_squares

from msrelapse._params import PAPER

__all__ = [
    "DEFAULT_BAND_FRACTION",
    "Barriers",
    "CriticalPoints",
    "Curvatures",
    "DoubleWell",
    "Passage",
    "Side",
    "barrier_ratio_from_durations",
    "beta_from_barrier_ratio",
    "calibrate",
    "fold_beta",
    "kramers_prefactor",
    "kramers_time",
    "mfpt",
    "paper_exit_time",
    "passage_endpoints",
]

Side = Literal["health", "relapse"]
"""Which of the two wells an episode is spent in."""

Passage = Literal["bottom_to_saddle", "bottom_to_bottom", "band"]
"""Which crossing of the potential one episode is taken to be."""

DEFAULT_BAND_FRACTION: Final = 0.3
"""Where the two thresholds of a ``band`` passage sit, by default.

The number is the fraction of the distance from the saddle to each well bottom
at which that state is entered, so a small fraction puts the thresholds close to
the saddle and a large one deep inside the wells. It is the default of
:func:`passage_endpoints` and :func:`calibrate` here, and of every band taking
function of :mod:`msrelapse.simulate`, so that a potential calibrated with
``passage='band'`` is cut into episodes exactly as it was calibrated. The cohort
engine of :mod:`msrelapse.cohort` is the one place that departs from it; see
:data:`msrelapse.cohort.SDE_ENGINE_BAND_FRACTION`.
"""

_Vector = np.ndarray[tuple[int], np.dtype[np.float64]]
"""One dimensional array of doubles, the shape scipy hands a residual function."""

_SIDES: Final = ("health", "relapse")
_PASSAGES: Final = ("bottom_to_saddle", "bottom_to_bottom", "band")
_METHODS: Final = ("mfpt", "kramers")

# A root of the cubic counts as real when its imaginary part is below this.
_IMAGINARY_TOLERANCE: Final = 1e-10

# exp of an argument above about 709 overflows float64, so every exit time in
# this module is refused a little before that rather than returned as an
# infinity or raised as an OverflowError from the bare exponential.
_MAX_EXPONENT: Final = 700.0

# The reflecting boundary is placed where the potential has risen this many
# noise variances above the bottom of the well the walker starts in, which
# makes the inner integrand there smaller than its peak by exp(-40).
_BOUNDARY_RISE_IN_VARIANCES: Final = 20.0
_BOUNDARY_STEP: Final = 0.05
_BOUNDARY_MAX_STEPS: Final = 400

# Grid doubling schedule of the first passage quadrature.
_GRID_START: Final = 2**12
_GRID_MAX: Final = 2**21
_GRID_TOLERANCE: Final = 1e-8

# How far below the fold the bisection of beta_from_barrier_ratio stops, where
# the two barriers still solve cleanly.
_FOLD_MARGIN: Final = 1e-9

# Calibration search: bounds, starting points and the accepted mismatch.
_CALIBRATION_MARGIN: Final = 1e-6
# Below this relative gap between the two targets the calibration cannot
# separate them, because the search keeps beta strictly positive.
_SYMMETRIC_TARGET_GAP: Final = 1e-3
_CALIBRATION_BETA_STARTS: Final = (0.05, 0.15, 0.3)
_CALIBRATION_SIGMA_START: Final = 0.5
_SIGMA_MIN: Final = 0.02
_SIGMA_MAX: Final = 3.0
_CALIBRATION_TOLERANCE: Final = 1e-6
# Residual returned where the exponent would overflow, which is always the
# region of implausibly long times, so it points the search back towards
# shorter ones.
_UNREACHABLE_RESIDUAL: Final = 100.0
# Finite difference step of the calibration, relative to the unknowns. The
# first passage time is a quadrature and carries noise of order
# ``_GRID_TOLERANCE``, so the default step of about 1e-8 would be swamped.
_CALIBRATION_DIFF_STEP: Final = 1e-5


def _validate_choice(name: str, value: str, allowed: tuple[str, ...]) -> None:
    """Raise if a string argument is not one of the spellings the API accepts.

    Parameters
    ----------
    name : str
        Name of the argument, used in the error message.
    value : str
        Value that was passed in.
    allowed : tuple of str
        The spellings that are accepted.

    Raises
    ------
    ValueError
        If `value` is not in `allowed`.
    """
    if value not in allowed:
        raise ValueError(f"{name} must be one of {allowed}, got {value!r}")


def _is_positive_finite(value: float) -> bool:
    """Return whether a number is finite and above zero.

    Parameters
    ----------
    value : float
        Number to test.

    Returns
    -------
    bool
        True when the number is finite and strictly positive, which is what
        every quantity of this module that stands for a time, a noise or a
        control parameter has to be.
    """
    return math.isfinite(value) and value > 0.0


def _validate_sigma(sigma: float) -> None:
    """Raise if the noise amplitude is not a positive finite number.

    Parameters
    ----------
    sigma : float
        Noise amplitude, the square root of the paper's variance epsilon.

    Raises
    ------
    ValueError
        If `sigma` is zero, negative, infinite or not a number.
    """
    if not _is_positive_finite(sigma):
        raise ValueError(f"sigma must be a positive finite number, got {sigma!r}")


def _validate_alpha(alpha: float) -> None:
    """Raise if the control parameter is not a positive finite number.

    Parameters
    ----------
    alpha : float
        Control parameter of the potential.

    Raises
    ------
    ValueError
        If `alpha` is zero, negative, infinite or not a number, where the
        double well does not exist.
    """
    if not _is_positive_finite(alpha):
        raise ValueError(f"alpha must be a positive finite number, got {alpha!r}")


def _scalar_or_array(values: NDArray[np.float64]) -> float | NDArray[np.float64]:
    """Return a Python float for a zero dimensional result, the array otherwise.

    Parameters
    ----------
    values : numpy.ndarray
        Result of evaluating a potential function on ``numpy.asarray`` input.

    Returns
    -------
    float or numpy.ndarray
        The single value when the input was a scalar, else the array itself.
    """
    if values.ndim == 0:
        return float(values)
    return values


def fold_beta(alpha: float) -> float:
    """Return the asymmetry at which the shallower well disappears.

    The stationary points are the real roots of ``alpha x^3 - x + beta``. Three
    of them exist only while ``abs(beta)`` stays below ``2 / (3 sqrt(3 alpha))``,
    where two of the roots merge in a saddle-node fold and the double well
    becomes a single well.

    Parameters
    ----------
    alpha : float
        Control parameter of the potential. Must be positive.

    Returns
    -------
    float
        The fold value of the asymmetry, dimensionless.

    Raises
    ------
    ValueError
        If `alpha` is not a positive finite number, where the double well does
        not exist.

    Examples
    --------
    >>> round(fold_beta(1.0), 9)
    0.384900179
    """
    _validate_alpha(alpha)
    return 2.0 / (3.0 * math.sqrt(3.0 * alpha))


@dataclass(frozen=True)
class CriticalPoints:
    """The three stationary points of the potential, sorted from left to right.

    Attributes
    ----------
    health : float
        Bottom of the left well, the paper's x1, the state of health.
    saddle : float
        The middle stationary point, the paper's x0, the top of the barrier.
    relapse : float
        Bottom of the right well, the paper's x2, the state of no health.
    """

    health: float
    saddle: float
    relapse: float


@dataclass(frozen=True)
class Barriers:
    """The two barrier heights, each measured from the top of the barrier down.

    Attributes
    ----------
    health : float
        The paper's delta V1, ``V(saddle) - V(health)``, the climb out of the
        health well.
    relapse : float
        The paper's delta V2, ``V(saddle) - V(relapse)``, the climb out of the
        no health well.
    """

    health: float
    relapse: float

    @property
    def ratio(self) -> float:
        """float: The paper's delta V1 / delta V2, above one when beta is positive.

        Raises
        ------
        ValueError
            If the relapse barrier is not positive, which is where the shallow
            well has flattened onto the saddle. That is the case within about
            1e-12 of ``fold_beta(alpha)``, where the cubic still has three
            distinct roots but the two right ones no longer differ in the
            potential.
        """
        if self.relapse <= 0.0:
            raise ValueError(
                f"the barrier ratio is not defined when the relapse barrier is "
                f"{self.relapse!r}: the shallow well has flattened onto the saddle, which "
                f"is what happens as beta approaches fold_beta(alpha)"
            )
        return self.health / self.relapse


@dataclass(frozen=True)
class Curvatures:
    """The second derivative of the potential at each stationary point.

    Attributes
    ----------
    health : float
        ``V''`` at the bottom of the health well, positive.
    saddle : float
        ``V''`` at the top of the barrier, negative.
    relapse : float
        ``V''`` at the bottom of the no health well, positive.
    """

    health: float
    saddle: float
    relapse: float


@dataclass(frozen=True)
class DoubleWell:
    """The potential of equations (2) and (3) for one pair of parameters.

    Parameters
    ----------
    alpha : float, optional
        Control parameter, which sets the height of the barrier. Must be
        positive. Defaults to the reference value of the paper.
    beta : float, optional
        Asymmetry parameter. A positive value deepens the health well and makes
        the no health well shallower. Defaults to the symmetric case.

    Raises
    ------
    ValueError
        If `alpha` is not a positive finite number, or if `beta` is not finite.

    Examples
    --------
    >>> well = DoubleWell(1.0, 0.08)
    >>> round(well.barrier_ratio(), 6)
    1.914174
    """

    alpha: float = PAPER.alpha_reference.value
    beta: float = PAPER.beta_symmetric.value

    def __post_init__(self) -> None:
        """Reject parameters that do not describe a potential.

        Raises
        ------
        ValueError
            If `alpha` is not a positive finite number, or if `beta` is not
            finite.
        """
        _validate_alpha(self.alpha)
        if not math.isfinite(self.beta):
            raise ValueError(f"beta must be a finite number, got {self.beta!r}")

    @overload
    def V(self, x: float) -> float: ...

    @overload
    def V(self, x: NDArray[np.float64]) -> NDArray[np.float64]: ...

    def V(self, x: float | NDArray[np.float64]) -> float | NDArray[np.float64]:
        """Return the potential ``-x^2 / 2 + alpha x^4 / 4 + beta x``.

        Parameters
        ----------
        x : float or numpy.ndarray
            Position or positions at which to evaluate the potential.

        Returns
        -------
        float or numpy.ndarray
            The potential, a float for scalar input and an array of the same
            shape otherwise.
        """
        position = np.asarray(x, dtype=np.float64)
        values = -0.5 * position**2 + 0.25 * self.alpha * position**4 + self.beta * position
        return _scalar_or_array(values)

    @overload
    def dV(self, x: float) -> float: ...

    @overload
    def dV(self, x: NDArray[np.float64]) -> NDArray[np.float64]: ...

    def dV(self, x: float | NDArray[np.float64]) -> float | NDArray[np.float64]:
        """Return the gradient ``-x + alpha x^3 + beta`` of the potential.

        Parameters
        ----------
        x : float or numpy.ndarray
            Position or positions at which to evaluate the gradient.

        Returns
        -------
        float or numpy.ndarray
            The gradient, a float for scalar input and an array of the same
            shape otherwise.
        """
        position = np.asarray(x, dtype=np.float64)
        values = -position + self.alpha * position**3 + self.beta
        return _scalar_or_array(values)

    @overload
    def d2V(self, x: float) -> float: ...

    @overload
    def d2V(self, x: NDArray[np.float64]) -> NDArray[np.float64]: ...

    def d2V(self, x: float | NDArray[np.float64]) -> float | NDArray[np.float64]:
        """Return the curvature ``-1 + 3 alpha x^2`` of the potential.

        Parameters
        ----------
        x : float or numpy.ndarray
            Position or positions at which to evaluate the curvature.

        Returns
        -------
        float or numpy.ndarray
            The curvature, a float for scalar input and an array of the same
            shape otherwise. It does not depend on beta, which only tilts the
            potential.
        """
        position = np.asarray(x, dtype=np.float64)
        values = -1.0 + 3.0 * self.alpha * position**2
        return _scalar_or_array(values)

    @overload
    def force(self, x: float) -> float: ...

    @overload
    def force(self, x: NDArray[np.float64]) -> NDArray[np.float64]: ...

    def force(self, x: float | NDArray[np.float64]) -> float | NDArray[np.float64]:
        """Return the drift ``x (1 - alpha x^2) - beta`` of equation (4).

        This is minus the gradient of the potential, the deterministic part of
        the stochastic equation of motion.

        Parameters
        ----------
        x : float or numpy.ndarray
            Position or positions at which to evaluate the drift.

        Returns
        -------
        float or numpy.ndarray
            The drift, a float for scalar input and an array of the same shape
            otherwise.
        """
        position = np.asarray(x, dtype=np.float64)
        values = position * (1.0 - self.alpha * position**2) - self.beta
        return _scalar_or_array(values)

    @property
    def is_bistable(self) -> bool:
        """bool: Whether the potential still has two wells, that is ``abs(beta) < fold_beta``."""
        return abs(self.beta) < fold_beta(self.alpha)

    def critical_points(self) -> CriticalPoints:
        """Return the three stationary points of the potential.

        They are the real roots of ``alpha x^3 - x + beta``, found numerically
        and sorted from left to right as health, saddle and no health.

        Returns
        -------
        CriticalPoints
            The three positions, ascending.

        Raises
        ------
        ValueError
            If the cubic does not have three real roots, which happens once
            ``abs(beta)`` reaches ``fold_beta(alpha)``.
        """
        roots = np.roots(np.array([self.alpha, 0.0, -1.0, self.beta], dtype=np.float64))
        real = np.sort(roots[np.abs(roots.imag) < _IMAGINARY_TOLERANCE].real)
        if real.size != 3:
            raise ValueError(
                f"the potential with alpha={self.alpha!r} and beta={self.beta!r} has "
                f"{real.size} real critical points instead of three; three exist only "
                f"while abs(beta) stays below fold_beta(alpha) = {fold_beta(self.alpha)!r}"
            )
        return CriticalPoints(health=float(real[0]), saddle=float(real[1]), relapse=float(real[2]))

    def barriers(self) -> Barriers:
        """Return the two barrier heights, both measured from the saddle.

        Returns
        -------
        Barriers
            The paper's delta V1 and delta V2, both positive.

        Raises
        ------
        ValueError
            If the potential does not have two wells.
        """
        return _barriers_at(self, self.critical_points())

    def barrier_ratio(self) -> float:
        """Return the ratio of the two barriers, the paper's delta V1 / delta V2.

        Returns
        -------
        float
            The ratio, one for a symmetric potential and above one for a
            positive asymmetry.

        Raises
        ------
        ValueError
            If the potential does not have two wells, or if beta is so close to
            ``fold_beta(alpha)`` that the relapse barrier has already collapsed
            onto the saddle and the ratio has no value; see
            :attr:`Barriers.ratio`.
        """
        return self.barriers().ratio

    def well_bottom(self, side: Side) -> float:
        """Return the position of the bottom of one of the two wells.

        Parameters
        ----------
        side : {'health', 'relapse'}
            Which well to report.

        Returns
        -------
        float
            The position of that minimum.

        Raises
        ------
        ValueError
            If `side` is not one of the two spellings, or if the potential does
            not have two wells.
        """
        _validate_choice("side", side, _SIDES)
        points = self.critical_points()
        return points.health if side == "health" else points.relapse

    def curvatures(self) -> Curvatures:
        """Return the curvature of the potential at the three stationary points.

        Returns
        -------
        Curvatures
            ``V''`` at the health well, the saddle and the no health well. The
            two well entries are positive and the saddle entry is negative.

        Raises
        ------
        ValueError
            If the potential does not have two wells.
        """
        points = self.critical_points()
        return Curvatures(
            health=self.d2V(points.health),
            saddle=self.d2V(points.saddle),
            relapse=self.d2V(points.relapse),
        )


def _barriers_at(well: DoubleWell, points: CriticalPoints) -> Barriers:
    """Return the two barrier heights of a potential whose stationary points are known.

    Taking the points as an argument lets a caller that has already solved the
    cubic reuse that solution instead of solving it a second time.

    Parameters
    ----------
    well : DoubleWell
        The potential.
    points : CriticalPoints
        The three stationary points of that same potential.

    Returns
    -------
    Barriers
        The paper's delta V1 and delta V2, both measured from the saddle down.
    """
    top = well.V(points.saddle)
    return Barriers(health=top - well.V(points.health), relapse=top - well.V(points.relapse))


def _height_exponent(height: float, sigma: float) -> float:
    """Return ``2 * height / sigma**2``, infinite where the square underflows.

    A sigma below about 1.5e-162 has a square that underflows to exactly zero
    in float64, where the plain division would raise ZeroDivisionError. Such a
    noise puts every positive height far beyond the range of float64, which is
    what the callers' own ``_MAX_EXPONENT`` guard reports, so the exponent is
    handed back as infinite and that guard turns it into the ValueError the
    public functions document.

    Parameters
    ----------
    height : float
        A barrier to climb, or the largest departure of the potential over a
        passage, in units of the potential. Never negative.
    sigma : float
        Noise amplitude. Must already have passed :func:`_validate_sigma`.

    Returns
    -------
    float
        The exponent, infinite where the square of `sigma` underflows and the
        height is positive.
    """
    variance = sigma * sigma
    if variance == 0.0:
        return math.inf if height > 0.0 else 0.0
    return 2.0 * height / variance


def _escape_exponent(well: DoubleWell, sigma: float, side: Side) -> float:
    """Return ``2 barrier / sigma**2``, refusing a value float64 cannot carry.

    Both exit time estimators return e raised to this exponent, so past about
    709 the answer stops being a number at all. Refusing it here turns what
    would otherwise be an ``OverflowError`` out of :func:`math.exp` into a
    ValueError that names the noise, the barrier and the potential.

    Parameters
    ----------
    well : DoubleWell
        The potential.
    sigma : float
        Noise amplitude.
    side : {'health', 'relapse'}
        Which well the walker escapes from.

    Returns
    -------
    float
        The exponent of the exit time.

    Raises
    ------
    ValueError
        If the exponent puts the exit time beyond the range of float64.
    """
    barriers = well.barriers()
    barrier = barriers.health if side == "health" else barriers.relapse
    exponent = _height_exponent(barrier, sigma)
    if exponent > _MAX_EXPONENT:
        raise ValueError(
            f"2 * barrier / sigma^2 = {exponent:.6g} at sigma={sigma!r} puts the {side} "
            f"exit time beyond the range of float64 for the potential with "
            f"alpha={well.alpha!r} and beta={well.beta!r}"
        )
    return exponent


def passage_endpoints(
    well: DoubleWell,
    side: Side,
    passage: Passage = "bottom_to_saddle",
    band_fraction: float = DEFAULT_BAND_FRACTION,
) -> tuple[float, float]:
    """Return the start and the absorbing point of one episode of a given side.

    The paper does not say where an episode begins and ends, so the choice is
    made explicit here. A health episode always runs towards increasing x and a
    relapse episode towards decreasing x.

    Parameters
    ----------
    well : DoubleWell
        The potential the episode happens in.
    side : {'health', 'relapse'}
        Which state the episode is spent in.
    passage : {'bottom_to_saddle', 'bottom_to_bottom', 'band'}, optional
        How much of the crossing counts as the episode. ``bottom_to_saddle``
        runs from the well bottom to the top of the barrier,
        ``bottom_to_bottom`` from one well bottom to the other, and ``band``
        from the entry threshold of the state to the entry threshold of the
        other state.
    band_fraction : float, optional
        Position of the two thresholds of the ``band`` passage, as a fraction
        of the distance from the saddle to each well bottom. Must lie strictly
        between 0 and 1, and defaults to :data:`DEFAULT_BAND_FRACTION`. It
        matches the argument of the same name used to cut a simulated trajectory
        into episodes, so calibrating with ``passage='band'`` and the same
        fraction makes the simulated durations match the targets.

    Returns
    -------
    tuple of float
        The starting position and the absorbing position.

    Raises
    ------
    ValueError
        If `side` or `passage` is not one of the accepted spellings, if
        `band_fraction` is not strictly between 0 and 1, or if the potential
        does not have two wells.

    Examples
    --------
    >>> start, absorb = passage_endpoints(DoubleWell(1.0, 0.08), "health")
    >>> round(start, 6), round(absorb, 6)
    (-1.037827, 0.080522)
    """
    _validate_choice("side", side, _SIDES)
    _validate_choice("passage", passage, _PASSAGES)
    if not 0.0 < band_fraction < 1.0:
        raise ValueError(f"band_fraction must lie strictly between 0 and 1, got {band_fraction!r}")
    points = well.critical_points()
    if passage == "bottom_to_saddle":
        return well.well_bottom(side), points.saddle
    if passage == "bottom_to_bottom":
        if side == "health":
            return points.health, points.relapse
        return points.relapse, points.health
    threshold_health = points.saddle - band_fraction * (points.saddle - points.health)
    threshold_relapse = points.saddle + band_fraction * (points.relapse - points.saddle)
    if side == "health":
        return threshold_health, threshold_relapse
    return threshold_relapse, threshold_health


def kramers_prefactor(well: DoubleWell, side: Side) -> float:
    """Return the prefactor of the Kramers escape time, in weeks.

    The prefactor is ``2 pi / sqrt(V''(bottom) * abs(V''(saddle)))``. It carries
    the units of the problem: with the model time of equation (4) read as
    weeks, it is a number of weeks, and it is the shortest escape time the
    Kramers formula can produce for this potential.

    Parameters
    ----------
    well : DoubleWell
        The potential.
    side : {'health', 'relapse'}
        Which well the walker escapes from.

    Returns
    -------
    float
        The prefactor, in weeks.

    Raises
    ------
    ValueError
        If `side` is not one of the two spellings, or if the potential does not
        have two wells.
    """
    _validate_choice("side", side, _SIDES)
    curvatures = well.curvatures()
    bottom = curvatures.health if side == "health" else curvatures.relapse
    return 2.0 * math.pi / math.sqrt(bottom * abs(curvatures.saddle))


def kramers_time(well: DoubleWell, sigma: float, side: Side) -> float:
    """Return the classical Kramers escape time from one well, in weeks.

    The value is ``kramers_prefactor * exp(2 barrier / sigma**2)``. It is the
    well to well escape time in the small noise limit; the mean first passage
    time from the bottom of the well to the saddle is about half of it, because
    a walker that reaches the saddle still falls back with probability about a
    half. Equations (5) and (6) of the paper omit the prefactor altogether, for
    which see :func:`paper_exit_time`.

    Parameters
    ----------
    well : DoubleWell
        The potential.
    sigma : float
        Noise amplitude, the square root of the paper's variance epsilon. Must
        be positive.
    side : {'health', 'relapse'}
        Which well the walker escapes from.

    Returns
    -------
    float
        The escape time, in weeks.

    Raises
    ------
    ValueError
        If `sigma` is not a positive finite number, if `side` is not one of the
        two spellings, if the potential does not have two wells, or if
        ``2 barrier / sigma**2`` puts the escape time beyond the range of
        float64, which happens once that exponent passes about 700.
    """
    _validate_choice("side", side, _SIDES)
    _validate_sigma(sigma)
    return kramers_prefactor(well, side) * math.exp(_escape_exponent(well, sigma, side))


def paper_exit_time(well: DoubleWell, sigma: float, side: Side) -> float:
    """Return the mean exit time of equations (5) and (6), without a prefactor.

    The paper prints ``tau = exp(2 delta V / epsilon)`` with epsilon the noise
    variance, that is ``sigma**2`` here, and no prefactor. Dropping the
    prefactor makes equation (7) exact by construction: the ratio
    ``log(paper_exit_time(health)) / log(paper_exit_time(relapse))`` equals
    :meth:`DoubleWell.barrier_ratio` for every noise amplitude, which is why
    the noise cancels out of the estimator.

    Parameters
    ----------
    well : DoubleWell
        The potential.
    sigma : float
        Noise amplitude, the square root of the paper's variance epsilon. Must
        be positive.
    side : {'health', 'relapse'}
        Which well the walker escapes from.

    Returns
    -------
    float
        The exit time as the paper writes it, dimensionless.

    Raises
    ------
    ValueError
        If `sigma` is not a positive finite number, if `side` is not one of the
        two spellings, if the potential does not have two wells, or if
        ``2 barrier / sigma**2`` puts the exit time beyond the range of float64,
        which happens once that exponent passes about 700.
    """
    _validate_choice("side", side, _SIDES)
    _validate_sigma(sigma)
    return math.exp(_escape_exponent(well, sigma, side))


def _reflecting_boundary(
    well: DoubleWell,
    sigma: float,
    origin: float,
    reference: float,
    direction: float,
) -> float:
    """Return a position far enough out that the walker never reaches it.

    The search steps away from the well until the potential has risen
    ``_BOUNDARY_RISE_IN_VARIANCES`` noise variances above the bottom of that
    well, where the inner integrand of the first passage time has fallen below
    its peak by ``exp(-40)``.

    Parameters
    ----------
    well : DoubleWell
        The potential.
    sigma : float
        Noise amplitude.
    origin : float
        Where to start stepping, the outermost of the well bottom and the
        starting point of the passage.
    reference : float
        The potential at the bottom of the well the walker starts in.
    direction : float
        Minus one to step left, plus one to step right.

    Returns
    -------
    float
        The position of the reflecting boundary.

    Raises
    ------
    ValueError
        If the potential has not risen far enough within the search range.
    """
    rise = _BOUNDARY_RISE_IN_VARIANCES * sigma**2
    position = origin
    for _ in range(_BOUNDARY_MAX_STEPS):
        position += direction * _BOUNDARY_STEP
        if well.V(position) - reference > rise:
            return position
    raise ValueError(
        f"no reflecting boundary was found within "
        f"{_BOUNDARY_MAX_STEPS * _BOUNDARY_STEP} of {origin!r}; sigma={sigma!r} is too "
        f"large for the potential with alpha={well.alpha!r} and beta={well.beta!r}"
    )


def _largest_departure(
    well: DoubleWell,
    points: CriticalPoints,
    lower: float,
    upper: float,
) -> float:
    """Return how far the potential leaves its value at the saddle on an interval.

    Both first passage integrands are exponentials of the potential measured
    from the saddle, so this is the number that decides whether either of them
    can be represented at all. A smooth quartic attains its extremes on an
    interval either at an endpoint or at a stationary point inside it, so those
    are the only positions that have to be looked at.

    Parameters
    ----------
    well : DoubleWell
        The potential.
    points : CriticalPoints
        The three stationary points of that same potential.
    lower : float
        Left end of the interval the quadrature runs over.
    upper : float
        Right end of the interval.

    Returns
    -------
    float
        The largest value of ``abs(V(x) - V(saddle))`` on the interval.
    """
    saddle_value = well.V(points.saddle)
    inside = [x for x in (points.health, points.saddle, points.relapse) if lower < x < upper]
    return max(abs(well.V(x) - saddle_value) for x in (lower, upper, *inside))


def _passage_time_on_grid(
    well: DoubleWell,
    variance: float,
    shift: float,
    *,
    start: float,
    absorb: float,
    reflecting: float,
    n_points: int,
) -> float:
    """Return the double quadrature of the mean first passage time on one grid.

    Both exponentials are shifted by the potential at the saddle. The shift
    cancels in the product of the two integrands and keeps each of them inside
    the range of float64.

    Parameters
    ----------
    well : DoubleWell
        The potential.
    variance : float
        ``sigma**2``, the paper's epsilon.
    shift : float
        The potential at the saddle, subtracted inside both exponentials.
    start : float
        Where the walker starts.
    absorb : float
        The absorbing point.
    reflecting : float
        The reflecting boundary, beyond the well and on the far side from
        `absorb`.
    n_points : int
        Number of grid points spanning the whole interval.

    Returns
    -------
    float
        The mean first passage time on this grid.
    """
    rightward = start < absorb
    lower, upper = (reflecting, absorb) if rightward else (absorb, reflecting)
    below = max(2, round(n_points * (start - lower) / (upper - lower)))
    above = max(2, n_points - below)
    grid = np.concatenate(
        [
            np.linspace(lower, start, below, endpoint=False),
            np.linspace(start, upper, above),
        ]
    )
    offset = well.V(grid) - shift
    inner_integrand = np.exp(-2.0 * offset / variance)
    outer_factor = np.exp(2.0 * offset / variance)
    if rightward:
        inner = cumulative_trapezoid(inner_integrand, grid, initial=0)
        nodes, integrand = grid[below:], (outer_factor * inner)[below:]
    else:
        reversed_inner = cumulative_trapezoid(inner_integrand[::-1], -grid[::-1], initial=0)
        inner = reversed_inner[::-1]
        nodes, integrand = grid[: below + 1], (outer_factor * inner)[: below + 1]
    return float(2.0 / variance * trapezoid(integrand, nodes))


def mfpt(
    well: DoubleWell,
    sigma: float,
    side: Side,
    x0: float | None = None,
    x_absorb: float | None = None,
) -> float:
    """Return the exact mean first passage time between two points, in weeks.

    For ``dx = -V'(x) dt + sigma dW`` with an absorbing point at `x_absorb` and
    a reflecting boundary far out on the other side, the mean first passage
    time from `x0` is a double integral. For a health passage, which runs
    towards increasing x::

        T = (2 / sigma^2) int_{x0}^{x_absorb} dy e^{2 V(y) / sigma^2}
                          int_{-L}^{y} dz e^{-2 V(z) / sigma^2}

    and for a relapse passage, which runs towards decreasing x::

        T = (2 / sigma^2) int_{x_absorb}^{x0} dy e^{2 V(y) / sigma^2}
                          int_{y}^{+L} dz e^{-2 V(z) / sigma^2}

    The integrals are evaluated on a uniform grid that is doubled until the
    answer stops moving by more than 1e-8 in relative terms.

    Parameters
    ----------
    well : DoubleWell
        The potential.
    sigma : float
        Noise amplitude, the square root of the paper's variance epsilon. Must
        be positive.
    side : {'health', 'relapse'}
        Which state the episode is spent in, which also fixes the direction of
        the passage and the side the reflecting boundary sits on.
    x0 : float, optional
        Starting position. Defaults to the bottom of the well of `side`.
    x_absorb : float, optional
        Absorbing position. Defaults to the saddle. Use
        :func:`passage_endpoints` to get the pair that matches a given reading
        of what one episode is.

    Returns
    -------
    float
        The mean first passage time, in weeks, finite and positive.

    Raises
    ------
    ValueError
        If `sigma` is not a positive finite number, if `side` is not one of the
        two spellings, if the exponent ``2 max(barrier) / sigma**2`` would
        overflow float64, if `x0` and `x_absorb` are not ordered the way `side`
        requires, if they sit so far apart that the integrand overflows anyway,
        or if the quadrature does not converge.

    See Also
    --------
    kramers_time : The small noise approximation, without a quadrature.

    Examples
    --------
    >>> round(mfpt(DoubleWell(1.0, 0.08), 0.4, "health"), 4)
    151.8618
    """
    _validate_choice("side", side, _SIDES)
    _validate_sigma(sigma)
    points = well.critical_points()
    bottom = points.health if side == "health" else points.relapse
    start = bottom if x0 is None else x0
    absorb = points.saddle if x_absorb is None else x_absorb
    if side == "health" and start >= absorb:
        raise ValueError(
            f"a health passage runs towards increasing x, so x0 must be smaller than "
            f"x_absorb, got x0={start!r} and x_absorb={absorb!r}"
        )
    if side == "relapse" and start <= absorb:
        raise ValueError(
            f"a relapse passage runs towards decreasing x, so x0 must be larger than "
            f"x_absorb, got x0={start!r} and x_absorb={absorb!r}"
        )
    barriers = _barriers_at(well, points)
    exponent = _height_exponent(max(barriers.health, barriers.relapse), sigma)
    if exponent > _MAX_EXPONENT:
        raise ValueError(
            f"2 * barrier / sigma^2 = {exponent:.6g} overflows float64 inside the first "
            f"passage integrand at sigma={sigma!r}; kramers_time is the small noise formula "
            f"for this regime, but it too is representable only while its own exponent stays "
            f"below about 700"
        )
    direction = -1.0 if side == "health" else 1.0
    origin = min(start, bottom) if side == "health" else max(start, bottom)
    reflecting = _reflecting_boundary(well, sigma, origin, well.V(bottom), direction)
    lower, upper = (reflecting, absorb) if side == "health" else (absorb, reflecting)
    departure = _largest_departure(well, points, lower, upper)
    departure_exponent = _height_exponent(departure, sigma)
    if departure_exponent > _MAX_EXPONENT:
        raise ValueError(
            f"the potential departs from its value at the saddle by up to {departure:.6g} "
            f"on the passage from x0={start!r} to x_absorb={absorb!r}, so "
            f"2 * departure / sigma^2 = {departure_exponent:.6g} overflows float64 inside "
            f"the first passage integrand at sigma={sigma!r}; keep the endpoints inside the "
            f"two wells"
        )
    shift = well.V(points.saddle)
    previous: float | None = None
    n_points = _GRID_START
    while n_points <= _GRID_MAX:
        value = _passage_time_on_grid(
            well,
            sigma**2,
            shift,
            start=start,
            absorb=absorb,
            reflecting=reflecting,
            n_points=n_points,
        )
        if previous is not None and abs(value - previous) <= _GRID_TOLERANCE * abs(value):
            return value
        previous = value
        n_points *= 2
    raise ValueError(
        f"the first passage quadrature did not settle to {_GRID_TOLERANCE} in relative "
        f"terms within {_GRID_MAX} grid points for the passage from x0={start!r} to "
        f"x_absorb={absorb!r} at sigma={sigma!r} on the potential with alpha={well.alpha!r} "
        f"and beta={well.beta!r}"
    )


def barrier_ratio_from_durations(tau_health: float, tau_relapse: float) -> float:
    """Return the barrier ratio implied by two observed durations, equation (7).

    Equation (7) reads ``delta V1 / delta V2 = log(tau_x1) / log(tau_x2)``. It
    follows from equations (5) and (6) only because those drop the prefactor,
    so it silently takes the prefactor of both wells to be one week. Treat the
    result as the documented heuristic of the paper rather than as a definition
    of the barrier ratio. Any base of the logarithm gives the same number
    because the base cancels.

    Parameters
    ----------
    tau_health : float
        Mean duration of a health episode, in weeks. Must be above one week.
    tau_relapse : float
        Mean duration of a no health episode, in weeks. Must be above one week.

    Returns
    -------
    float
        The implied ratio ``delta V1 / delta V2``.

    Raises
    ------
    ValueError
        If either duration is one week or shorter, where its logarithm is zero
        or negative and the estimator is undefined.

    Examples
    --------
    >>> round(barrier_ratio_from_durations(100.0, 4.3), 6)
    3.157221
    """
    if tau_health <= 1.0 or tau_relapse <= 1.0:
        raise ValueError(
            f"both durations must be greater than 1 week, because the logarithm of a "
            f"shorter duration is zero or negative and the ratio of equation (7) is then "
            f"undefined, got tau_health={tau_health!r} and tau_relapse={tau_relapse!r}"
        )
    return float(np.log(tau_health) / np.log(tau_relapse))


def beta_from_barrier_ratio(ratio: float, alpha: float = PAPER.alpha_reference.value) -> float:
    """Return the asymmetry whose two barriers stand in a given ratio.

    This is the fitting rule of the paper, page 6: fix alpha and change beta
    until the two wells are asymmetric in the wanted ratio. The barrier ratio
    grows without bound as beta approaches the fold, so exactly one beta in
    ``[0, fold_beta(alpha))`` answers, and it is found by bisection.

    Parameters
    ----------
    ratio : float
        The wanted ``delta V1 / delta V2``. Must be a finite number of at least
        one.
    alpha : float, optional
        Control parameter, held fixed. Defaults to the reference value of the
        paper, which is the only value the paper ever fits with.

    Returns
    -------
    float
        The asymmetry beta, between zero and ``fold_beta(alpha)``.

    Raises
    ------
    ValueError
        If `ratio` is not a finite number, if `ratio` is below one, which would
        need a negative beta and would make the no health well the deeper of
        the two, if `ratio` is larger than any potential at this alpha can
        deliver, or if `alpha` is not a positive finite number.

    Examples
    --------
    >>> round(beta_from_barrier_ratio(3.157221141), 6)
    0.13749
    """
    if not math.isfinite(ratio):
        raise ValueError(f"the barrier ratio must be a finite number, got {ratio!r}")
    if ratio < 1.0:
        raise ValueError(
            f"the barrier ratio must be at least 1, because a positive beta makes the "
            f"health well the deeper one, got {ratio!r}"
        )
    upper = fold_beta(alpha) - _FOLD_MARGIN
    if ratio == 1.0:
        return 0.0
    largest = DoubleWell(alpha=alpha, beta=upper).barrier_ratio()
    if not ratio <= largest:
        raise ValueError(
            f"no beta below the fold delivers a barrier ratio of {ratio!r} at "
            f"alpha={alpha!r}; the largest ratio reachable is {largest:.6g}, at a beta just "
            f"under fold_beta(alpha) = {fold_beta(alpha)!r}"
        )

    def mismatch(beta: float) -> float:
        return DoubleWell(alpha=alpha, beta=beta).barrier_ratio() - ratio

    return float(brentq(mismatch, 0.0, upper))


def _calibration_bounds(alpha: float) -> tuple[list[float], list[float]]:
    """Return the search box of the calibration, as least_squares wants it.

    Parameters
    ----------
    alpha : float
        Control parameter, held fixed during the calibration.

    Returns
    -------
    tuple of list of float
        Lower and upper bounds on beta and on the logarithm of sigma.

    Raises
    ------
    ValueError
        If `alpha` is so large that the fold leaves no room for beta.
    """
    upper_beta = fold_beta(alpha) - _CALIBRATION_MARGIN
    if upper_beta <= _CALIBRATION_MARGIN:
        raise ValueError(
            f"alpha={alpha!r} puts the fold at {fold_beta(alpha)!r}, which leaves no room "
            f"for a positive asymmetry"
        )
    return (
        [_CALIBRATION_MARGIN, math.log(_SIGMA_MIN)],
        [upper_beta, math.log(_SIGMA_MAX)],
    )


def calibrate(
    tau_health: float,
    tau_relapse: float,
    alpha: float = PAPER.alpha_reference.value,
    *,
    method: Literal["mfpt", "kramers"] = "mfpt",
    passage: Passage = "bottom_to_saddle",
    band_fraction: float = DEFAULT_BAND_FRACTION,
) -> tuple[float, float]:
    """Return the asymmetry and the noise that reproduce two episode durations.

    The two unknowns are beta and sigma, and the two equations are that the
    mean time of a health episode is `tau_health` weeks and that of a relapse
    episode is `tau_relapse` weeks. The fit is a least squares problem in beta
    and ``log(sigma)`` on the logarithms of the two times, started from three
    values of beta and kept at the best of the three.

    Parameters
    ----------
    tau_health : float
        Target mean duration of a health episode, in weeks. Must be a positive
        finite number.
    tau_relapse : float
        Target mean duration of a no health episode, in weeks. Must be a
        positive finite number.
    alpha : float, optional
        Control parameter, held fixed. Defaults to the reference value of the
        paper.
    method : {'mfpt', 'kramers'}, optional
        Which reading of the episode time to match. ``mfpt`` uses the exact
        mean first passage time over the endpoints of `passage`; ``kramers``
        uses the small noise escape time, which ignores `passage` altogether
        because it is always a well to well time.
    passage : {'bottom_to_saddle', 'bottom_to_bottom', 'band'}, optional
        Which crossing counts as one episode, see :func:`passage_endpoints`.
        Ignored when `method` is ``kramers``.
    band_fraction : float, optional
        Threshold position of the ``band`` passage, which defaults to
        :data:`DEFAULT_BAND_FRACTION`. Ignored when `passage` is not ``band`` or
        `method` is ``kramers``.

    Returns
    -------
    tuple of float
        The asymmetry beta and the noise amplitude sigma.

    Raises
    ------
    ValueError
        If either target is not a positive finite number, if `method` or
        `passage` is not one of the accepted spellings, or if no parameter pair
        inside the search box reproduces both targets to 1e-6 in relative
        terms.

    Notes
    -----
    The Kramers escape time can never fall below its prefactor, and at the
    reference alpha that prefactor is never below ``2 pi / sqrt(2)``, about 4.44
    weeks. A relapse of 4.3 weeks therefore has no ``kramers`` solution at all,
    and the exact first passage time is the only reading of an episode that can
    be calibrated to the cohort of the paper.

    The search also keeps beta strictly positive, at 1e-6 or above, so the
    exactly symmetric potential sits just outside the box. Two targets that
    differ by less than about 3e-5 in relative terms therefore cannot be
    matched, and ``DoubleWell(alpha, 0.0)`` is the potential that answers
    there, with the noise that the symmetric first passage time asks for.

    Examples
    --------
    >>> beta, sigma = calibrate(100.0, 4.3)
    >>> round(beta, 6), round(sigma, 6)
    (0.210435, 0.507768)
    """
    _validate_choice("method", method, _METHODS)
    _validate_choice("passage", passage, _PASSAGES)
    if not _is_positive_finite(tau_health) or not _is_positive_finite(tau_relapse):
        raise ValueError(
            f"both target times must be positive finite numbers, got "
            f"tau_health={tau_health!r} and tau_relapse={tau_relapse!r}"
        )
    lower_bounds, upper_bounds = _calibration_bounds(alpha)

    def episode_times(beta: float, sigma: float) -> tuple[float, float]:
        well = DoubleWell(alpha=alpha, beta=beta)
        if method == "kramers":
            return kramers_time(well, sigma, "health"), kramers_time(well, sigma, "relapse")
        health = passage_endpoints(well, "health", passage, band_fraction)
        relapse = passage_endpoints(well, "relapse", passage, band_fraction)
        return (
            mfpt(well, sigma, "health", *health),
            mfpt(well, sigma, "relapse", *relapse),
        )

    def residuals(unknowns: _Vector) -> NDArray[np.float64]:
        beta = float(unknowns[0])
        sigma = math.exp(float(unknowns[1]))
        barriers = DoubleWell(alpha=alpha, beta=beta).barriers()
        if 2.0 * max(barriers.health, barriers.relapse) / sigma**2 > _MAX_EXPONENT:
            return np.array([_UNREACHABLE_RESIDUAL, _UNREACHABLE_RESIDUAL])
        health, relapse = episode_times(beta, sigma)
        return np.array([math.log(health / tau_health), math.log(relapse / tau_relapse)])

    fits = [
        least_squares(
            residuals,
            [
                min(max(beta_start, lower_bounds[0]), upper_bounds[0]),
                math.log(_CALIBRATION_SIGMA_START),
            ],
            bounds=(lower_bounds, upper_bounds),
            diff_step=[_CALIBRATION_DIFF_STEP, _CALIBRATION_DIFF_STEP],
            xtol=1e-12,
            ftol=1e-12,
            gtol=1e-12,
        )
        for beta_start in _CALIBRATION_BETA_STARTS
    ]
    best = min(fits, key=lambda fit: float(fit.cost))
    beta = float(best.x[0])
    sigma = math.exp(float(best.x[1]))
    health, relapse = episode_times(beta, sigma)
    mismatch = max(abs(health / tau_health - 1.0), abs(relapse / tau_relapse - 1.0))
    if mismatch > _CALIBRATION_TOLERANCE:
        raise ValueError(
            _calibration_failure(
                DoubleWell(alpha=alpha, beta=beta),
                method,
                (tau_health, tau_relapse),
                (health, relapse),
                mismatch,
            )
        )
    return beta, sigma


def _calibration_failure(
    well: DoubleWell,
    method: str,
    targets: tuple[float, float],
    achieved: tuple[float, float],
    mismatch: float,
) -> str:
    """Return the message of a calibration that did not reach its targets.

    Parameters
    ----------
    well : DoubleWell
        The closest potential the fit reached.
    method : str
        Which reading of the episode time was being matched.
    targets : tuple of float
        The wanted health and relapse times, in weeks.
    achieved : tuple of float
        The closest health and relapse times, in weeks.
    mismatch : float
        The largest of the two relative mismatches.

    Returns
    -------
    str
        A message naming the targets and the closest times, followed by the
        floor the prefactor puts on each time when the method is Kramers, and
        by the positive floor on beta when the two targets are nearly equal.
    """
    sentences = [
        f"no beta and sigma reproduce tau_health={targets[0]!r} and "
        f"tau_relapse={targets[1]!r} weeks; the closest achievable times are "
        f"{achieved[0]:.6g} and {achieved[1]:.6g} weeks, a relative mismatch of "
        f"{mismatch:.3g}"
    ]
    if method == "kramers":
        sentences.append(
            f"The Kramers prefactor puts a floor of "
            f"{kramers_prefactor(well, 'health'):.6g} weeks on the health time and "
            f"{kramers_prefactor(well, 'relapse'):.6g} weeks on the relapse time, so no "
            f"shorter time can be matched by this method"
        )
    if abs(targets[0] / targets[1] - 1.0) < _SYMMETRIC_TARGET_GAP:
        sentences.append(
            f"The two targets are all but equal and the search keeps beta at "
            f"{_CALIBRATION_MARGIN} or above, so the symmetric potential lies outside the "
            f"search box; DoubleWell(alpha, 0.0) is the potential that answers there"
        )
    return ". ".join(sentences)
