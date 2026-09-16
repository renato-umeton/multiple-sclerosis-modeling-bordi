from __future__ import annotations

import math
from collections.abc import Callable
from inspect import signature
from itertools import pairwise

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from msrelapse._params import PAPER
from msrelapse.model import (
    DEFAULT_BAND_FRACTION,
    Barriers,
    CriticalPoints,
    Curvatures,
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

# Regression values for alpha = 1, beta = 0.08, taken from docs/paper_facts.md
# section 6 and recomputed to full double precision.
X_HEALTH_008 = -1.037826654
X_SADDLE_008 = 0.080522090
X_RELAPSE_008 = 0.957304564
BARRIER_HEALTH_008 = 0.334751014
BARRIER_RELAPSE_008 = 0.174880107
BARRIER_RATIO_008 = 1.914174342

COHORT_BARRIER_RATIO = 3.157221141


@pytest.fixture
def symmetric() -> DoubleWell:
    return DoubleWell()


@pytest.fixture
def asymmetric() -> DoubleWell:
    return DoubleWell(1.0, 0.08)


@pytest.fixture
def calibrated() -> DoubleWell:
    return DoubleWell(1.0, 0.210435212)


def test_symmetric_critical_points(symmetric: DoubleWell) -> None:
    points = symmetric.critical_points()
    assert (points.health, points.saddle, points.relapse) == pytest.approx(
        (-1.0, 0.0, 1.0), abs=1e-12
    )


def test_symmetric_barriers_are_equal(symmetric: DoubleWell) -> None:
    barriers = symmetric.barriers()
    assert (barriers.health, barriers.relapse) == pytest.approx((0.25, 0.25), abs=1e-12)


def test_symmetric_barrier_ratio_is_one(symmetric: DoubleWell) -> None:
    assert symmetric.barrier_ratio() == pytest.approx(1.0, abs=1e-12)


def test_symmetric_well_is_bistable(symmetric: DoubleWell) -> None:
    assert symmetric.is_bistable


def test_low_alpha_wells_move_outward() -> None:
    points = DoubleWell(0.7).critical_points()
    assert points.relapse == pytest.approx(1.195229, abs=1e-6)
    assert points.health == pytest.approx(-1.195229, abs=1e-6)


def test_low_alpha_barrier_grows() -> None:
    assert DoubleWell(0.7).barriers().health == pytest.approx(0.357142857, abs=1e-9)


def test_half_alpha_barrier() -> None:
    assert DoubleWell(0.5).barriers().health == pytest.approx(0.5, abs=1e-9)


def test_half_alpha_health_well_position() -> None:
    assert DoubleWell(0.5).critical_points().health == pytest.approx(-1.41421356, abs=1e-8)


def test_high_alpha_barrier() -> None:
    assert DoubleWell(1.5).barriers().health == pytest.approx(1.0 / 6.0, abs=1e-9)


def test_barrier_decreases_as_alpha_increases() -> None:
    heights = [DoubleWell(alpha).barriers().health for alpha in (0.5, 0.75, 1.0, 1.25, 1.5)]
    assert all(later < earlier for earlier, later in pairwise(heights))


def test_asymmetric_critical_points(asymmetric: DoubleWell) -> None:
    points = asymmetric.critical_points()
    expected = (X_HEALTH_008, X_SADDLE_008, X_RELAPSE_008)
    assert (points.health, points.saddle, points.relapse) == pytest.approx(expected, abs=1e-8)


def test_asymmetric_barriers(asymmetric: DoubleWell) -> None:
    barriers = asymmetric.barriers()
    expected = (BARRIER_HEALTH_008, BARRIER_RELAPSE_008)
    assert (barriers.health, barriers.relapse) == pytest.approx(expected, abs=1e-8)


def test_asymmetric_barrier_ratio(asymmetric: DoubleWell) -> None:
    assert asymmetric.barrier_ratio() == pytest.approx(BARRIER_RATIO_008, abs=1e-7)


def test_barriers_ratio_property_divides_the_two_heights(asymmetric: DoubleWell) -> None:
    points = asymmetric.critical_points()
    top = asymmetric.V(points.saddle)
    expected = (top - asymmetric.V(points.health)) / (top - asymmetric.V(points.relapse))
    assert asymmetric.barriers().ratio == pytest.approx(expected, rel=1e-12)


def test_barriers_ratio_refuses_a_collapsed_relapse_barrier() -> None:
    with pytest.raises(ValueError, match="not defined"):
        _ = Barriers(health=0.75, relapse=0.0).ratio


def test_barrier_ratio_just_inside_the_fold_names_the_collapsed_well() -> None:
    # The cubic still has three distinct roots this close to the fold, but the
    # relapse barrier is already zero, so the ratio has no value to report.
    well = DoubleWell(1.0, fold_beta(1.0) - 1e-12)
    assert well.barriers().relapse == 0.0
    with pytest.raises(ValueError, match="fold_beta"):
        well.barrier_ratio()


def test_asymmetric_curvatures(asymmetric: DoubleWell) -> None:
    curvatures = asymmetric.curvatures()
    expected = (2.231252490, -0.980548578, 1.749296089)
    assert (curvatures.health, curvatures.saddle, curvatures.relapse) == pytest.approx(
        expected, abs=1e-8
    )


def test_positive_beta_deepens_the_health_well(asymmetric: DoubleWell) -> None:
    barriers = asymmetric.barriers()
    assert barriers.health > barriers.relapse


def test_negative_beta_deepens_the_relapse_well() -> None:
    barriers = DoubleWell(1.0, -0.08).barriers()
    assert barriers.relapse > barriers.health


def test_fold_beta_at_reference_alpha() -> None:
    assert fold_beta(1.0) == pytest.approx(0.3849001795, abs=1e-9)


def test_fold_beta_matches_the_recorded_paper_bound() -> None:
    assert fold_beta(1.0) == pytest.approx(PAPER.derived_beta_fold_alpha1.value, abs=1e-15)


@pytest.mark.parametrize("alpha", [0.0, -1.0, float("nan"), float("inf")])
def test_fold_beta_rejects_an_alpha_that_is_not_positive_and_finite(alpha: float) -> None:
    with pytest.raises(ValueError, match="alpha must be a positive finite number"):
        fold_beta(alpha)


def test_beta_beyond_the_fold_is_not_bistable() -> None:
    assert not DoubleWell(1.0, 0.39).is_bistable


def test_beta_beyond_the_fold_has_no_three_critical_points() -> None:
    with pytest.raises(ValueError, match="real critical points"):
        DoubleWell(1.0, 0.39).critical_points()


@pytest.mark.parametrize("alpha", [0.0, -1.0, float("nan"), float("inf")])
def test_double_well_rejects_an_alpha_that_is_not_positive_and_finite(alpha: float) -> None:
    with pytest.raises(ValueError, match="alpha must be a positive finite number"):
        DoubleWell(alpha)


def test_alpha_error_message_reports_the_value() -> None:
    with pytest.raises(ValueError, match=r"alpha must be a positive finite number, got -1\.0"):
        DoubleWell(-1.0)


@pytest.mark.parametrize("beta", [float("nan"), float("inf"), float("-inf")])
def test_double_well_rejects_a_non_finite_asymmetry(beta: float) -> None:
    with pytest.raises(ValueError, match="beta must be a finite number"):
        DoubleWell(1.0, beta)


@pytest.mark.parametrize("name", ["V", "dV", "d2V", "force"])
def test_potential_functions_preserve_array_shape(symmetric: DoubleWell, name: str) -> None:
    x = np.linspace(-2.0, 2.0, 12).reshape(3, 4)
    values = getattr(symmetric, name)(x)
    assert isinstance(values, np.ndarray)
    assert values.shape == x.shape


@pytest.mark.parametrize("name", ["V", "dV", "d2V", "force"])
def test_potential_functions_return_a_float_for_scalar_input(
    symmetric: DoubleWell, name: str
) -> None:
    assert isinstance(getattr(symmetric, name)(0.5), float)


def test_potential_matches_the_printed_formula(asymmetric: DoubleWell) -> None:
    x = 0.37
    expected = -0.5 * x**2 + 0.25 * asymmetric.alpha * x**4 + asymmetric.beta * x
    assert asymmetric.V(x) == pytest.approx(expected, abs=1e-15)


def test_force_matches_the_printed_drift(asymmetric: DoubleWell) -> None:
    x = np.linspace(-2.0, 2.0, 9)
    expected = x * (1.0 - asymmetric.alpha * x**2) - asymmetric.beta
    assert asymmetric.force(x) == pytest.approx(expected, abs=1e-15)


def test_force_is_minus_the_gradient(asymmetric: DoubleWell) -> None:
    x = np.linspace(-2.0, 2.0, 9)
    assert asymmetric.force(x) == pytest.approx(-asymmetric.dV(x), abs=1e-15)


def test_gradient_vanishes_at_every_critical_point(asymmetric: DoubleWell) -> None:
    points = asymmetric.critical_points()
    for x in (points.health, points.saddle, points.relapse):
        assert asymmetric.dV(x) == pytest.approx(0.0, abs=1e-10)


def test_saddle_sits_above_both_wells(asymmetric: DoubleWell) -> None:
    points = asymmetric.critical_points()
    assert asymmetric.V(points.saddle) > asymmetric.V(points.health)
    assert asymmetric.V(points.saddle) > asymmetric.V(points.relapse)


def test_well_bottom_returns_the_health_minimum(asymmetric: DoubleWell) -> None:
    assert asymmetric.well_bottom("health") == asymmetric.critical_points().health


def test_well_bottom_returns_the_relapse_minimum(asymmetric: DoubleWell) -> None:
    assert asymmetric.well_bottom("relapse") == asymmetric.critical_points().relapse


def test_well_bottom_rejects_an_unknown_side(asymmetric: DoubleWell) -> None:
    with pytest.raises(ValueError, match="side"):
        asymmetric.well_bottom("remission")  # type: ignore[arg-type]


def test_double_well_is_frozen(symmetric: DoubleWell) -> None:
    with pytest.raises(AttributeError):
        symmetric.alpha = 2.0  # type: ignore[misc]


def test_results_are_the_dedicated_record_types(asymmetric: DoubleWell) -> None:
    assert isinstance(asymmetric.critical_points(), CriticalPoints)
    assert isinstance(asymmetric.barriers(), Barriers)
    assert isinstance(asymmetric.curvatures(), Curvatures)


@pytest.mark.parametrize("name", ["critical_points", "barriers", "curvatures"])
def test_results_are_frozen(asymmetric: DoubleWell, name: str) -> None:
    record = getattr(asymmetric, name)()
    with pytest.raises(AttributeError):
        record.health = 0.0


@pytest.mark.parametrize(
    ("side", "expected"),
    [("health", 4.247867402), ("relapse", 4.797485255)],
)
def test_kramers_prefactor(asymmetric: DoubleWell, side: str, expected: float) -> None:
    assert kramers_prefactor(asymmetric, side) == pytest.approx(expected, abs=1e-7)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("side", "expected"),
    [("health", 278.8864593), ("relapse", 42.69548657)],
)
def test_kramers_time(asymmetric: DoubleWell, side: str, expected: float) -> None:
    assert kramers_time(asymmetric, 0.4, side) == pytest.approx(expected, rel=1e-7)  # type: ignore[arg-type]


@pytest.mark.parametrize("sigma", [0.0, -0.4, float("nan"), float("inf")])
def test_kramers_time_rejects_a_sigma_that_is_not_positive_and_finite(
    asymmetric: DoubleWell, sigma: float
) -> None:
    with pytest.raises(ValueError, match="sigma must be a positive finite number"):
        kramers_time(asymmetric, sigma, "health")


def test_kramers_time_refuses_a_time_beyond_the_range_of_float64(calibrated: DoubleWell) -> None:
    with pytest.raises(ValueError, match="float64"):
        kramers_time(calibrated, 0.03, "health")


def test_paper_exit_time_refuses_a_time_beyond_the_range_of_float64(
    calibrated: DoubleWell,
) -> None:
    with pytest.raises(ValueError, match="float64"):
        paper_exit_time(calibrated, 0.03, "health")


@pytest.mark.parametrize("function", [kramers_time, paper_exit_time, mfpt])
def test_a_sigma_whose_square_underflows_is_refused_by_name(
    asymmetric: DoubleWell, function: Callable[..., float]
) -> None:
    # Below about 1.5e-162 the square of sigma is exactly zero in float64, so
    # the exponent of the exit time would be a bare division by zero rather
    # than the named error the rest of that range gets.
    with pytest.raises(ValueError, match="float64"):
        function(asymmetric, 1e-200, "health")


def test_a_barrier_that_has_vanished_costs_nothing_at_any_noise() -> None:
    # Just inside the fold the relapse barrier is exactly zero, so the exit
    # time of equation (5) is exp(0) whatever the noise, even where the square
    # of that noise underflows and a positive barrier would be refused.
    well = DoubleWell(1.0, fold_beta(1.0) - 1e-12)
    assert well.barriers().relapse == 0.0
    assert paper_exit_time(well, 1e-200, "relapse") == 1.0


@pytest.mark.parametrize(
    ("side", "expected"),
    [("health", 151.8618466), ("relapse", 24.70239682)],
)
def test_mfpt_from_well_bottom_to_saddle(
    asymmetric: DoubleWell, side: str, expected: float
) -> None:
    assert mfpt(asymmetric, 0.4, side) == pytest.approx(expected, rel=1e-5)  # type: ignore[arg-type]


def test_mfpt_at_larger_noise(asymmetric: DoubleWell) -> None:
    assert mfpt(asymmetric, 0.5, "health") == pytest.approx(34.31647343, rel=1e-5)


def test_mfpt_is_a_finite_positive_float(asymmetric: DoubleWell) -> None:
    value = mfpt(asymmetric, 0.4, "health")
    assert isinstance(value, float)
    assert math.isfinite(value)
    assert value > 0.0


@pytest.mark.parametrize(
    ("side", "expected"),
    [("health", 1.83644849), ("relapse", 1.72839449)],
)
def test_kramers_time_is_about_twice_the_bottom_to_saddle_time(
    asymmetric: DoubleWell, side: str, expected: float
) -> None:
    ratio = kramers_time(asymmetric, 0.4, side) / mfpt(asymmetric, 0.4, side)  # type: ignore[arg-type]
    assert ratio == pytest.approx(expected, rel=1e-4)


def test_the_factor_two_is_approached_at_small_noise(asymmetric: DoubleWell) -> None:
    ratio = kramers_time(asymmetric, 0.2, "health") / mfpt(asymmetric, 0.2, "health")
    assert ratio == pytest.approx(1.9814, abs=0.01)


@pytest.mark.parametrize("sigma", [0.2, 0.3])
@pytest.mark.parametrize("side", ["health", "relapse"])
def test_bottom_to_bottom_time_agrees_with_kramers_at_small_noise(
    asymmetric: DoubleWell, sigma: float, side: str
) -> None:
    endpoints = passage_endpoints(asymmetric, side, "bottom_to_bottom")  # type: ignore[arg-type]
    exact = mfpt(asymmetric, sigma, side, *endpoints)  # type: ignore[arg-type]
    assert kramers_time(asymmetric, sigma, side) == pytest.approx(exact, rel=0.15)  # type: ignore[arg-type]


def test_bottom_to_bottom_cases_are_in_the_small_noise_regime(asymmetric: DoubleWell) -> None:
    # Both barriers are worth at least six noise variances at sigma = 0.2, which is
    # what makes the well to well agreement above a fair check of the small noise limit.
    barriers = asymmetric.barriers()
    assert 2.0 * min(barriers.health, barriers.relapse) / 0.2**2 >= 6.0


def test_mfpt_rejects_noise_that_overflows_the_exponent(calibrated: DoubleWell) -> None:
    with pytest.raises(ValueError, match="overflows"):
        mfpt(calibrated, 0.03, "health")


def test_mfpt_overflow_message_points_at_kramers_time(calibrated: DoubleWell) -> None:
    with pytest.raises(ValueError, match="kramers_time"):
        mfpt(calibrated, 0.03, "health")


def test_mfpt_rejects_endpoints_on_the_wrong_side_for_health(asymmetric: DoubleWell) -> None:
    with pytest.raises(ValueError, match="x0"):
        mfpt(asymmetric, 0.4, "health", 0.5, -0.5)


def test_mfpt_rejects_endpoints_on_the_wrong_side_for_relapse(asymmetric: DoubleWell) -> None:
    with pytest.raises(ValueError, match="x0"):
        mfpt(asymmetric, 0.4, "relapse", -0.5, 0.5)


@pytest.mark.parametrize("sigma", [0.0, -0.4, float("nan"), float("inf")])
def test_mfpt_rejects_a_sigma_that_is_not_positive_and_finite(
    asymmetric: DoubleWell, sigma: float
) -> None:
    with pytest.raises(ValueError, match="sigma must be a positive finite number"):
        mfpt(asymmetric, sigma, "health")


def test_mfpt_rejects_noise_too_large_for_a_reflecting_boundary(asymmetric: DoubleWell) -> None:
    with pytest.raises(ValueError, match="no reflecting boundary"):
        mfpt(asymmetric, 60.0, "health")


def test_mfpt_rejects_an_absorbing_point_far_outside_the_wells(asymmetric: DoubleWell) -> None:
    with pytest.raises(ValueError, match="x_absorb"):
        mfpt(asymmetric, 0.4, "health", asymmetric.well_bottom("health"), 6.0)


def test_mfpt_non_convergence_names_the_endpoints(asymmetric: DoubleWell) -> None:
    with pytest.raises(ValueError, match=r"x_absorb=2\.5"):
        mfpt(asymmetric, 0.4, "health", asymmetric.well_bottom("health"), 2.5)


def test_paper_exit_time_has_no_prefactor(asymmetric: DoubleWell) -> None:
    expected = math.exp(2.0 * asymmetric.barriers().health / 0.4**2)
    assert paper_exit_time(asymmetric, 0.4, "health") == pytest.approx(expected, rel=1e-12)


def test_paper_exit_time_log_ratio_is_the_barrier_ratio(asymmetric: DoubleWell) -> None:
    health = paper_exit_time(asymmetric, 0.4, "health")
    relapse = paper_exit_time(asymmetric, 0.4, "relapse")
    ratio = math.log(health) / math.log(relapse)
    assert ratio == pytest.approx(asymmetric.barrier_ratio(), rel=1e-12)


def test_barrier_ratio_from_cohort_durations() -> None:
    assert barrier_ratio_from_durations(100.0, 4.3) == pytest.approx(COHORT_BARRIER_RATIO, abs=1e-9)


def test_barrier_ratio_from_the_recorded_cohort_durations() -> None:
    ratio = barrier_ratio_from_durations(
        PAPER.tau_health_cohort_weeks.value, PAPER.tau_no_health_cohort_weeks.value
    )
    assert ratio == pytest.approx(PAPER.barrier_ratio_cohort.value, abs=0.06)


def test_barrier_ratio_is_independent_of_the_logarithm_base() -> None:
    natural = barrier_ratio_from_durations(100.0, 4.3)
    decimal = math.log10(100.0) / math.log10(4.3)
    assert natural == pytest.approx(decimal, rel=1e-12)


def test_barrier_ratio_rejects_a_unit_relapse_duration() -> None:
    with pytest.raises(ValueError, match="greater than 1"):
        barrier_ratio_from_durations(100.0, 1.0)


def test_barrier_ratio_rejects_a_health_duration_below_one() -> None:
    with pytest.raises(ValueError, match="greater than 1"):
        barrier_ratio_from_durations(0.5, 4.3)


def test_beta_from_the_cohort_barrier_ratio() -> None:
    assert beta_from_barrier_ratio(COHORT_BARRIER_RATIO) == pytest.approx(0.137489927, abs=1e-7)


def test_beta_from_barrier_ratio_round_trips() -> None:
    beta = beta_from_barrier_ratio(COHORT_BARRIER_RATIO)
    assert DoubleWell(1.0, beta).barrier_ratio() == pytest.approx(COHORT_BARRIER_RATIO, rel=1e-8)


def test_equal_barriers_need_no_asymmetry() -> None:
    assert beta_from_barrier_ratio(1.0) == 0.0


def test_beta_from_barrier_ratio_rejects_a_ratio_below_one() -> None:
    with pytest.raises(ValueError, match="at least 1"):
        beta_from_barrier_ratio(0.5)


def test_beta_from_barrier_ratio_rejects_a_ratio_the_potential_cannot_reach() -> None:
    with pytest.raises(ValueError, match="largest ratio reachable"):
        beta_from_barrier_ratio(1e14)


@pytest.mark.parametrize("ratio", [float("inf"), float("nan")])
def test_beta_from_barrier_ratio_rejects_a_ratio_that_is_not_finite(ratio: float) -> None:
    # A missing ratio is not a ratio the fold cannot reach, and saying so would
    # send the reader looking at alpha instead of at the ratio they passed.
    with pytest.raises(ValueError, match="must be a finite number"):
        beta_from_barrier_ratio(ratio)


def test_patient_32_asymmetry_matches_the_printed_beta() -> None:
    beta = beta_from_barrier_ratio(PAPER.barrier_ratio_p32.value)
    assert beta == pytest.approx(PAPER.beta_patient_32.value, abs=0.005)


def test_patient_53_asymmetry_matches_the_printed_beta() -> None:
    beta = beta_from_barrier_ratio(PAPER.barrier_ratio_p53.value)
    assert beta == pytest.approx(PAPER.beta_patient_53.value, abs=0.005)


def test_patient_23_printed_beta_does_not_deliver_the_printed_ratio() -> None:
    # The paper prints beta = 0.25 and a barrier ratio of 11.8 for patient 23, but
    # beta = 0.25 delivers 10.775. This is a documented inconsistency inside the
    # paper, recorded in docs/paper_facts.md section 6, and is asserted here so the
    # package never silently adopts one of the two numbers over the other.
    ratio = DoubleWell(1.0, PAPER.beta_patient_23.value).barrier_ratio()
    assert ratio == pytest.approx(10.775, abs=0.01)
    assert ratio != pytest.approx(PAPER.barrier_ratio_p23.value, abs=0.5)


def test_bottom_to_saddle_endpoints(asymmetric: DoubleWell) -> None:
    points = asymmetric.critical_points()
    assert passage_endpoints(asymmetric, "health") == (points.health, points.saddle)


def test_bottom_to_bottom_endpoints(asymmetric: DoubleWell) -> None:
    points = asymmetric.critical_points()
    start, absorb = passage_endpoints(asymmetric, "relapse", "bottom_to_bottom")
    assert (start, absorb) == (points.relapse, points.health)


def test_band_endpoints_for_a_remission_episode(asymmetric: DoubleWell) -> None:
    points = asymmetric.critical_points()
    expected = (
        points.saddle - 0.3 * (points.saddle - points.health),
        points.saddle + 0.3 * (points.relapse - points.saddle),
    )
    assert passage_endpoints(asymmetric, "health", "band", 0.3) == pytest.approx(expected)


def test_band_endpoints_for_a_relapse_episode_run_the_other_way(asymmetric: DoubleWell) -> None:
    forward = passage_endpoints(asymmetric, "health", "band", 0.3)
    backward = passage_endpoints(asymmetric, "relapse", "band", 0.3)
    assert backward == (forward[1], forward[0])


@pytest.mark.parametrize("band_fraction", [0.0, 1.0, -0.2, 1.5])
def test_band_fraction_must_be_strictly_inside_the_unit_interval(
    asymmetric: DoubleWell, band_fraction: float
) -> None:
    with pytest.raises(ValueError, match="band_fraction"):
        passage_endpoints(asymmetric, "health", "band", band_fraction)


def test_passage_endpoints_rejects_an_unknown_passage(asymmetric: DoubleWell) -> None:
    with pytest.raises(ValueError, match="passage"):
        passage_endpoints(asymmetric, "health", "bottom_to_top")  # type: ignore[arg-type]


def test_calibration_to_the_cohort_durations() -> None:
    beta, sigma = calibrate(100.0, 4.3)
    assert beta == pytest.approx(0.210435212, abs=1e-5)
    assert sigma == pytest.approx(0.507768007, abs=1e-5)


def test_calibrated_well_critical_points() -> None:
    beta, _ = calibrate(100.0, 4.3)
    points = DoubleWell(1.0, beta).critical_points()
    expected = (-1.09210262, 0.22126846, 0.87083416)
    assert (points.health, points.saddle, points.relapse) == pytest.approx(expected, abs=1e-6)


def test_calibrated_well_barriers() -> None:
    beta, _ = calibrate(100.0, 4.3)
    barriers = DoubleWell(1.0, beta).barriers()
    assert (barriers.health, barriers.relapse) == pytest.approx((0.49321674, 0.07482948), abs=1e-6)


def test_calibrated_well_barrier_ratio() -> None:
    beta, _ = calibrate(100.0, 4.3)
    assert DoubleWell(1.0, beta).barrier_ratio() == pytest.approx(6.5912090, abs=1e-4)


@pytest.mark.parametrize(("side", "target"), [("health", 100.0), ("relapse", 4.3)])
def test_calibration_round_trips_through_the_first_passage_time(side: str, target: float) -> None:
    beta, sigma = calibrate(100.0, 4.3)
    well = DoubleWell(1.0, beta)
    assert mfpt(well, sigma, side) == pytest.approx(target, rel=1e-6)  # type: ignore[arg-type]


@pytest.mark.parametrize(("side", "target"), [("health", 100.0), ("relapse", 4.3)])
def test_band_calibration_round_trips_over_the_band_endpoints(side: str, target: float) -> None:
    beta, sigma = calibrate(100.0, 4.3, passage="band", band_fraction=0.3)
    well = DoubleWell(1.0, beta)
    endpoints = passage_endpoints(well, side, "band", 0.3)  # type: ignore[arg-type]
    assert mfpt(well, sigma, side, *endpoints) == pytest.approx(target, rel=1e-6)  # type: ignore[arg-type]


def test_kramers_calibration_cannot_reach_a_relapse_of_four_weeks() -> None:
    with pytest.raises(ValueError, match="prefactor"):
        calibrate(100.0, 4.3, method="kramers")


@pytest.mark.parametrize(("side", "target"), [("health", 100.0), ("relapse", 10.0)])
def test_kramers_calibration_round_trips_above_the_prefactor_floor(
    side: str, target: float
) -> None:
    beta, sigma = calibrate(100.0, 10.0, method="kramers")
    well = DoubleWell(1.0, beta)
    assert kramers_time(well, sigma, side) == pytest.approx(target, rel=1e-6)  # type: ignore[arg-type]


@pytest.mark.parametrize(("tau_health", "tau_relapse"), [(0.0, 4.3), (100.0, -1.0)])
def test_calibrate_rejects_non_positive_targets(tau_health: float, tau_relapse: float) -> None:
    with pytest.raises(ValueError, match="must be positive"):
        calibrate(tau_health, tau_relapse)


@pytest.mark.parametrize(
    ("tau_health", "tau_relapse"),
    [
        (float("nan"), 4.3),
        (float("inf"), 4.3),
        (100.0, float("nan")),
        (100.0, float("inf")),
    ],
)
def test_calibrate_rejects_a_target_that_is_not_finite(
    tau_health: float, tau_relapse: float
) -> None:
    # A missing mean reaches this the moment a caller averages an empty group,
    # and the least squares driver has no idea which argument to name.
    with pytest.raises(ValueError, match="must be positive finite numbers"):
        calibrate(tau_health, tau_relapse)


def test_calibrate_rejects_an_alpha_that_leaves_no_room_for_beta() -> None:
    with pytest.raises(ValueError, match="no room"):
        calibrate(100.0, 4.3, 1e12)


def test_calibrate_rejects_an_unknown_method() -> None:
    with pytest.raises(ValueError, match="method"):
        calibrate(100.0, 4.3, method="eyeball")  # type: ignore[arg-type]


def test_calibrate_reports_the_closest_times_it_reached() -> None:
    with pytest.raises(ValueError, match="closest achievable times"):
        calibrate(100.0, 100.0)


def test_calibrate_explains_that_the_symmetric_case_is_outside_the_search_box() -> None:
    with pytest.raises(ValueError, match="symmetric potential lies outside the search box"):
        calibrate(100.0, 100.0)


@pytest.mark.parametrize("function", [passage_endpoints, calibrate])
def test_the_band_threshold_defaults_to_the_shared_constant(
    function: Callable[..., object],
) -> None:
    # A calibration and the segmentation that reads it have to cut at the same
    # band, so neither of these two functions can be called with a band of its
    # own by default: both resolve to the one constant of this layer.
    assert signature(function).parameters["band_fraction"].default == DEFAULT_BAND_FRACTION


def test_the_band_threshold_constant_holds_its_documented_value() -> None:
    # The number is quoted in the Notes of CohortSpec, in the module docstrings
    # and on the plan page, and the wider band the sde engine uses is set
    # against it. Pinning it here makes a change to it a deliberate edit that
    # has to move those readings too.
    assert DEFAULT_BAND_FRACTION == 0.3


def test_a_calibration_that_probes_an_unreachable_noise_turns_back() -> None:
    # Targets this long ask for a barrier the search can only reach by driving
    # the noise down until the exponent of the exit time leaves float64. The
    # residual there is a penalty that points the search back rather than an
    # overflow, so the calibration still lands on its targets.
    beta, sigma = calibrate(1e300, 1e250)
    well = DoubleWell(PAPER.alpha_reference.value, beta)
    health = passage_endpoints(well, "health")
    relapse = passage_endpoints(well, "relapse")

    assert mfpt(well, sigma, "health", *health) == pytest.approx(1e300, rel=1e-6)
    assert mfpt(well, sigma, "relapse", *relapse) == pytest.approx(1e250, rel=1e-6)


@settings(deadline=None, max_examples=200, derandomize=True)
@given(
    alpha=st.floats(min_value=0.3, max_value=3.0),
    magnitude=st.floats(min_value=0.01, max_value=0.9),
    sign=st.sampled_from([-1.0, 1.0]),
)
def test_critical_points_and_barriers_are_consistent(
    alpha: float, magnitude: float, sign: float
) -> None:
    beta = sign * magnitude * fold_beta(alpha)
    well = DoubleWell(alpha, beta)
    points = well.critical_points()
    barriers = well.barriers()
    for x in (points.health, points.saddle, points.relapse):
        assert well.dV(x) == pytest.approx(0.0, abs=1e-8)
    assert well.V(points.saddle) > well.V(points.health)
    assert well.V(points.saddle) > well.V(points.relapse)
    assert (barriers.health > barriers.relapse) == (beta > 0.0)
