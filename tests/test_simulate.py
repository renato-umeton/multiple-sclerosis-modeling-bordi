from __future__ import annotations

import doctest

import numpy as np
import numpy.typing as npt
import pandas as pd
import pytest
from scipy import stats

import msrelapse.simulate
from msrelapse._params import PAPER
from msrelapse.io import validate, weekly_to_durations
from msrelapse.model import DoubleWell, calibrate, mfpt, passage_endpoints
from msrelapse.simulate import (
    HAS_NUMBA,
    durations,
    exit_times,
    simulate_paths,
    simulate_weekly,
    to_states,
    to_weekly,
)

# No test here carries pytest.mark.slow: the whole file runs in about two
# seconds, and the heaviest of them, the 100 patient cohort over 1500 weeks,
# takes under half a second even with the numba kernels turned off. Marking a
# fast test slow would let "-m 'not slow'" drop the statistical checks that
# matter most.

RELAPSE = PAPER.state_no_health.value
HEALTH = PAPER.state_health.value
TAU_HEALTH = PAPER.tau_health_cohort_weeks.value
TAU_RELAPSE = PAPER.tau_no_health_cohort_weeks.value
WEEK = PAPER.time_resolution_weeks.value
BAND_FRACTION = 0.3


@pytest.fixture(scope="module")
def saddle_well() -> tuple[DoubleWell, float]:
    """Return the potential whose bottom to saddle times are the cohort durations."""
    beta, sigma = calibrate(TAU_HEALTH, TAU_RELAPSE)
    return DoubleWell(PAPER.alpha_reference.value, beta), sigma


@pytest.fixture(scope="module")
def band_well() -> tuple[DoubleWell, float]:
    """Return the potential whose band crossings are the cohort durations."""
    beta, sigma = calibrate(TAU_HEALTH, TAU_RELAPSE, passage="band", band_fraction=BAND_FRACTION)
    return DoubleWell(PAPER.alpha_reference.value, beta), sigma


@pytest.fixture(scope="module")
def health_exit_times(saddle_well: tuple[DoubleWell, float]) -> npt.NDArray[np.float64]:
    """Exit times out of the deep well, shared by the three tests that read them."""
    well, sigma = saddle_well
    return exit_times(well, sigma, "health", 2000, dt=0.02, rng=2)


def relapse_target(well: DoubleWell, sigma: float) -> float:
    """Return the exact bottom to saddle first passage time out of the shallow well."""
    return mfpt(well, sigma, "relapse", *passage_endpoints(well, "relapse"))


def health_target(well: DoubleWell, sigma: float) -> float:
    """Return the exact bottom to saddle first passage time out of the deep well."""
    return mfpt(well, sigma, "health", *passage_endpoints(well, "health"))


# Deterministic limit, where the paper's lettered list on pages 4 to 5 says a
# patient nearby a well stays there and a patient nearby the saddle falls into
# one of the two wells.


def test_zero_noise_path_stays_at_the_health_well_bottom(
    saddle_well: tuple[DoubleWell, float],
) -> None:
    well, _ = saddle_well
    paths = simulate_paths(well, 0.0, 50.0)
    assert np.max(np.abs(paths.x - well.well_bottom("health"))) < 1e-9


def test_zero_noise_path_above_the_saddle_falls_into_the_relapse_well(
    saddle_well: tuple[DoubleWell, float],
) -> None:
    well, _ = saddle_well
    points = well.critical_points()
    paths = simulate_paths(well, 0.0, 50.0, x0=points.saddle + 0.01)
    assert paths.x[0, -1] == pytest.approx(points.relapse, abs=1e-6)


def test_one_step_follows_the_drift_of_the_model(
    saddle_well: tuple[DoubleWell, float],
) -> None:
    # Every integration kernel writes the drift out rather than calling
    # DoubleWell.force, because the compiled kernel cannot call into the model.
    # This ties the copies to the model, so an edit to the drift of the model
    # that does not reach the kernel fails here.
    well, _ = saddle_well
    dt = 0.01
    paths = simulate_paths(well, 0.0, dt, dt=dt, x0=0.5)
    assert paths.x[0, 1] == pytest.approx(0.5 + well.force(0.5) * dt, abs=0.0)


# Shapes, validation and reproducibility of simulate_paths.


def test_a_step_that_blows_the_path_up_is_rejected(
    saddle_well: tuple[DoubleWell, float],
) -> None:
    well, sigma = saddle_well
    with pytest.raises(ValueError, match="blows up"):
        simulate_paths(well, sigma, 2.0, dt=0.5, n_paths=3)


@pytest.mark.parametrize(
    ("dt", "t_end", "n_paths", "record_every", "sigma", "message"),
    [
        (0.0, 2.0, 1, 1, 0.5, "dt must be"),
        (-0.25, 2.0, 1, 1, 0.5, "dt must be"),
        (0.25, 0.0, 1, 1, 0.5, "t_end must be"),
        (0.25, -2.0, 1, 1, 0.5, "t_end must be"),
        (0.25, 2.0, 0, 1, 0.5, "n_paths must be"),
        (0.25, 2.0, 1, 0, 0.5, "record_every must be"),
        (0.25, 2.0, 1, 1, -0.5, "sigma must be"),
    ],
)
def test_simulate_paths_rejects_out_of_range_arguments(
    saddle_well: tuple[DoubleWell, float],
    *,
    dt: float,
    t_end: float,
    n_paths: int,
    record_every: int,
    sigma: float,
    message: str,
) -> None:
    well, _ = saddle_well
    with pytest.raises(ValueError, match=message):
        simulate_paths(well, sigma, t_end, dt=dt, n_paths=n_paths, record_every=record_every)


def test_paths_that_leave_the_finite_range_are_rejected(
    saddle_well: tuple[DoubleWell, float],
) -> None:
    # MAX_DT was measured at the calibrated noise alone, so a step inside it
    # still blows the cubic drift up at a larger sigma. The count belongs in the
    # message, because a run where only some paths blew up looks ordinary.
    well, _ = saddle_well
    with pytest.raises(ValueError, match="4 of 4 paths left the finite range"):
        simulate_paths(well, 2.0, 20.0, dt=0.3, n_paths=4, rng=0)


def test_records_run_from_time_zero_to_the_end(
    saddle_well: tuple[DoubleWell, float],
) -> None:
    well, sigma = saddle_well
    paths = simulate_paths(well, sigma, 2.0, dt=0.25, n_paths=3, rng=0)
    assert paths.t == pytest.approx(np.arange(9) * 0.25)


def test_one_row_of_records_per_path(saddle_well: tuple[DoubleWell, float]) -> None:
    well, sigma = saddle_well
    paths = simulate_paths(well, sigma, 2.0, dt=0.25, n_paths=3, rng=0)
    assert paths.x.shape == (3, 9)


def test_record_every_thins_the_records(saddle_well: tuple[DoubleWell, float]) -> None:
    well, sigma = saddle_well
    paths = simulate_paths(well, sigma, 2.0, dt=0.25, n_paths=3, rng=0, record_every=2)
    assert paths.x.shape == (3, 5)


def test_thinned_records_are_the_full_records_sampled(
    saddle_well: tuple[DoubleWell, float],
) -> None:
    well, sigma = saddle_well
    full = simulate_paths(well, sigma, 2.0, dt=0.25, n_paths=3, rng=0)
    thinned = simulate_paths(well, sigma, 2.0, dt=0.25, n_paths=3, rng=0, record_every=2)
    assert thinned.x == pytest.approx(full.x[:, ::2], abs=0.0)


def test_x0_of_the_wrong_length_is_rejected(
    saddle_well: tuple[DoubleWell, float],
) -> None:
    well, sigma = saddle_well
    with pytest.raises(ValueError, match="x0 must be"):
        simulate_paths(well, sigma, 2.0, dt=0.25, n_paths=3, x0=np.zeros(2))


def test_one_start_per_path_is_honoured(saddle_well: tuple[DoubleWell, float]) -> None:
    well, sigma = saddle_well
    starts = np.array([-1.0, 0.0, 1.0])
    paths = simulate_paths(well, sigma, 2.0, dt=0.25, n_paths=3, x0=starts, rng=0)
    assert paths.x[:, 0] == pytest.approx(starts, abs=0.0)


def test_the_same_seed_reproduces_the_paths(
    saddle_well: tuple[DoubleWell, float],
) -> None:
    well, sigma = saddle_well
    first = simulate_paths(well, sigma, 5.0, dt=0.02, n_paths=4, rng=17)
    second = simulate_paths(well, sigma, 5.0, dt=0.02, n_paths=4, rng=17)
    assert first.x == pytest.approx(second.x, abs=0.0)


def test_different_seeds_give_different_paths(
    saddle_well: tuple[DoubleWell, float],
) -> None:
    well, sigma = saddle_well
    first = simulate_paths(well, sigma, 5.0, dt=0.02, n_paths=4, rng=17)
    second = simulate_paths(well, sigma, 5.0, dt=0.02, n_paths=4, rng=18)
    assert np.max(np.abs(first.x - second.x)) > 1e-6


def test_a_generator_is_used_as_it_is(saddle_well: tuple[DoubleWell, float]) -> None:
    well, sigma = saddle_well
    seeded = simulate_paths(well, sigma, 5.0, dt=0.02, n_paths=4, rng=17)
    generated = simulate_paths(well, sigma, 5.0, dt=0.02, n_paths=4, rng=np.random.default_rng(17))
    assert generated.x == pytest.approx(seeded.x, abs=0.0)


@pytest.mark.skipif(not HAS_NUMBA, reason="numba is not installed")
def test_the_numba_and_numpy_kernels_integrate_alike(
    saddle_well: tuple[DoubleWell, float],
) -> None:
    well, sigma = saddle_well
    fast = simulate_paths(well, sigma, 10.0, dt=0.02, n_paths=5, rng=7, use_numba=True)
    plain = simulate_paths(well, sigma, 10.0, dt=0.02, n_paths=5, rng=7, use_numba=False)
    assert np.max(np.abs(fast.x - plain.x)) < 1e-12


@pytest.mark.skipif(not HAS_NUMBA, reason="numba is not installed")
def test_the_numba_and_numpy_kernels_thin_the_records_alike(
    saddle_well: tuple[DoubleWell, float],
) -> None:
    well, sigma = saddle_well
    fast = simulate_paths(well, sigma, 10.0, dt=0.02, n_paths=5, rng=7, record_every=7)
    plain = simulate_paths(
        well, sigma, 10.0, dt=0.02, n_paths=5, rng=7, record_every=7, use_numba=False
    )
    assert np.max(np.abs(fast.x - plain.x)) < 1e-12


@pytest.mark.skipif(not HAS_NUMBA, reason="numba is not installed")
def test_the_numba_and_numpy_kernels_measure_the_same_exit_times(
    saddle_well: tuple[DoubleWell, float],
) -> None:
    well, sigma = saddle_well
    fast = exit_times(well, sigma, "relapse", 200, dt=0.02, rng=9, use_numba=True)
    plain = exit_times(well, sigma, "relapse", 200, dt=0.02, rng=9, use_numba=False)
    assert plain.tolist() == fast.tolist()


@pytest.mark.skipif(not HAS_NUMBA, reason="numba is not installed")
def test_the_kernels_agree_on_a_passage_that_outlasts_one_block_of_draws(
    saddle_well: tuple[DoubleWell, float],
) -> None:
    # A health passage lasts about 100 weeks, which is several blocks of draws
    # at dt = 0.02, so this is the case where a path carries over from one block
    # to the next with neither kernel having absorbed it.
    well, sigma = saddle_well
    fast = exit_times(well, sigma, "health", 20, dt=0.02, rng=6, use_numba=True)
    plain = exit_times(well, sigma, "health", 20, dt=0.02, rng=6, use_numba=False)
    assert plain.tolist() == fast.tolist()


@pytest.mark.skipif(not HAS_NUMBA, reason="numba is not installed")
def test_the_numba_and_numpy_kernels_map_the_same_states(
    saddle_well: tuple[DoubleWell, float],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    well, sigma = saddle_well
    paths = simulate_paths(well, sigma, 40.0, dt=0.02, n_paths=5, rng=11)
    fast = to_states(paths.x, well, BAND_FRACTION)
    monkeypatch.setattr(msrelapse.simulate, "HAS_NUMBA", False)
    assert to_states(paths.x, well, BAND_FRACTION).tolist() == fast.tolist()


def test_asking_for_a_kernel_that_is_not_installed_is_rejected(
    saddle_well: tuple[DoubleWell, float],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    well, sigma = saddle_well
    monkeypatch.setattr(msrelapse.simulate, "HAS_NUMBA", False)
    with pytest.raises(ValueError, match="needs the numba package"):
        simulate_paths(well, sigma, 1.0, use_numba=True)


# First passage times.


def test_relapse_exit_time_matches_the_first_passage_time(
    saddle_well: tuple[DoubleWell, float],
) -> None:
    well, sigma = saddle_well
    times = exit_times(well, sigma, "relapse", 4000, dt=0.02, rng=1)
    assert times.mean() == pytest.approx(relapse_target(well, sigma), rel=0.04)


def test_health_exit_time_matches_the_first_passage_time(
    saddle_well: tuple[DoubleWell, float],
    health_exit_times: npt.NDArray[np.float64],
) -> None:
    well, sigma = saddle_well
    assert health_exit_times.mean() == pytest.approx(health_target(well, sigma), rel=0.06)


def test_health_exit_times_have_the_spread_of_an_exponential(
    health_exit_times: npt.NDArray[np.float64],
) -> None:
    variation = health_exit_times.std(ddof=1) / health_exit_times.mean()
    assert abs(variation - 1.0) < 0.1


def test_health_exit_times_pass_an_exponential_goodness_of_fit(
    health_exit_times: npt.NDArray[np.float64],
) -> None:
    fitted = stats.expon(scale=float(health_exit_times.mean()))
    assert stats.kstest(health_exit_times, fitted.cdf).pvalue > 0.01


def test_a_coarse_step_without_the_bridge_correction_runs_long(
    saddle_well: tuple[DoubleWell, float],
) -> None:
    well, sigma = saddle_well
    times = exit_times(well, sigma, "relapse", 4000, dt=0.1, rng=5, bridge_correction=False)
    assert times.mean() > 1.1 * relapse_target(well, sigma)


def test_the_bridge_correction_rescues_a_coarse_step(
    saddle_well: tuple[DoubleWell, float],
) -> None:
    well, sigma = saddle_well
    times = exit_times(well, sigma, "relapse", 4000, dt=0.1, rng=5, bridge_correction=True)
    assert times.mean() == pytest.approx(relapse_target(well, sigma), rel=0.06)


@pytest.mark.parametrize(
    ("dt", "n_paths", "sigma", "max_time", "message"),
    [
        (0.0, 10, 0.5, 100.0, "dt must be"),
        (-0.02, 10, 0.5, 100.0, "dt must be"),
        (0.5, 10, 0.5, 100.0, "blows up"),
        (0.02, 0, 0.5, 100.0, "n_paths must be"),
        (0.02, 10, -0.5, 100.0, "sigma must be"),
        (0.02, 10, 0.5, 0.0, "max_time must be"),
        (0.02, 10, 0.5, -100.0, "max_time must be"),
    ],
)
def test_exit_times_rejects_out_of_range_arguments(
    saddle_well: tuple[DoubleWell, float],
    *,
    dt: float,
    n_paths: int,
    sigma: float,
    max_time: float,
    message: str,
) -> None:
    well, _ = saddle_well
    with pytest.raises(ValueError, match=message):
        exit_times(well, sigma, "relapse", n_paths, dt=dt, max_time=max_time)


def test_paths_that_do_not_finish_in_time_are_counted(
    saddle_well: tuple[DoubleWell, float],
) -> None:
    well, sigma = saddle_well
    with pytest.raises(ValueError, match="50 of 50"):
        exit_times(well, sigma, "health", 50, dt=0.02, rng=3, max_time=1.0)


def test_a_correction_wider_than_the_passage_is_rejected(
    saddle_well: tuple[DoubleWell, float],
) -> None:
    well, sigma = saddle_well
    with pytest.raises(ValueError, match="absorbed in one step"):
        exit_times(well, sigma, "relapse", 10, dt=0.3, passage="band", band_fraction=0.01)


def test_crossing_the_whole_well_takes_longer_than_reaching_the_saddle(
    saddle_well: tuple[DoubleWell, float],
) -> None:
    well, sigma = saddle_well
    to_saddle = exit_times(well, sigma, "relapse", 500, dt=0.02, rng=4)
    to_bottom = exit_times(well, sigma, "relapse", 500, dt=0.02, rng=4, passage="bottom_to_bottom")
    assert to_bottom.mean() > to_saddle.mean()


# The hysteresis band.


def test_a_path_that_stays_inside_the_band_never_switches(
    saddle_well: tuple[DoubleWell, float],
) -> None:
    well, _ = saddle_well
    low, high = passage_endpoints(well, "health", "band", BAND_FRACTION)
    inside = np.array([low + 0.01, high - 0.01] * 10)
    assert np.all(to_states(inside, well, BAND_FRACTION) == HEALTH)


def test_switches_land_on_the_crossing_steps(
    saddle_well: tuple[DoubleWell, float],
) -> None:
    well, _ = saddle_well
    low, high = passage_endpoints(well, "health", "band", BAND_FRACTION)
    series = np.zeros(30)
    series[10] = high + 0.01
    series[20] = low - 0.01
    expected = np.where((np.arange(30) >= 10) & (np.arange(30) < 20), RELAPSE, HEALTH)
    assert to_states(series, well, BAND_FRACTION).tolist() == expected.tolist()


def test_each_path_is_mapped_on_its_own(saddle_well: tuple[DoubleWell, float]) -> None:
    well, _ = saddle_well
    low, high = passage_endpoints(well, "health", "band", BAND_FRACTION)
    first = np.array([0.0, high + 0.01, 0.0, 0.0])
    second = np.array([0.0, 0.0, low - 0.01, 0.0])
    rows = to_states(np.vstack([first, second]), well, BAND_FRACTION)
    assert rows.tolist() == [
        to_states(first, well, BAND_FRACTION).tolist(),
        to_states(second, well, BAND_FRACTION).tolist(),
    ]


def test_an_explicit_initial_state_overrides_the_default(
    saddle_well: tuple[DoubleWell, float],
) -> None:
    well, _ = saddle_well
    inside = np.zeros(5)
    assert np.all(to_states(inside, well, BAND_FRACTION, initial=RELAPSE) == RELAPSE)


def test_the_default_initial_state_follows_the_first_sample(
    saddle_well: tuple[DoubleWell, float],
) -> None:
    well, _ = saddle_well
    saddle = well.critical_points().saddle
    above = np.full(5, saddle + 0.01)
    assert np.all(to_states(above, well, BAND_FRACTION) == RELAPSE)


@pytest.mark.parametrize("initial", [0, 2, True, np.True_])
def test_a_state_code_the_schema_does_not_know_is_rejected(
    saddle_well: tuple[DoubleWell, float],
    initial: int,
) -> None:
    # A boolean is rejected as well, although True equals the no health code:
    # a caller who reaches this argument with a mask element or the result of a
    # comparison means something else by it than "start every path in relapse".
    well, _ = saddle_well
    with pytest.raises(ValueError, match="initial must be"):
        to_states(np.zeros(5), well, BAND_FRACTION, initial=initial)


def test_a_three_dimensional_cohort_is_rejected(
    saddle_well: tuple[DoubleWell, float],
) -> None:
    well, _ = saddle_well
    with pytest.raises(ValueError, match="one or two dimensional"):
        to_states(np.zeros((2, 3, 4)), well, BAND_FRACTION)


def test_a_single_sample_with_no_series_around_it_is_rejected(
    saddle_well: tuple[DoubleWell, float],
) -> None:
    well, _ = saddle_well
    with pytest.raises(ValueError, match="one or two dimensional"):
        to_states(np.array(0.5), well, BAND_FRACTION)


def test_a_path_with_no_records_is_rejected(saddle_well: tuple[DoubleWell, float]) -> None:
    well, _ = saddle_well
    with pytest.raises(ValueError, match="at least one record"):
        to_states(np.zeros(0), well, BAND_FRACTION)


def test_a_sample_that_is_not_a_finite_number_is_rejected(
    saddle_well: tuple[DoubleWell, float],
) -> None:
    # A path that left the finite range compares False against both thresholds,
    # so the state would freeze at whatever it held instead of switching.
    well, _ = saddle_well
    series = np.zeros(5)
    series[2] = np.nan
    with pytest.raises(ValueError, match="x must hold finite samples"):
        to_states(series, well, BAND_FRACTION)


def test_a_zero_noise_health_path_never_leaves_health(
    saddle_well: tuple[DoubleWell, float],
) -> None:
    well, _ = saddle_well
    paths = simulate_paths(well, 0.0, 20.0)
    assert np.all(to_states(paths.x, well, BAND_FRACTION) == HEALTH)


# Weekly rounding.


def test_a_single_relapse_step_takes_its_whole_week() -> None:
    steps = np.full(16, HEALTH)
    steps[13] = RELAPSE
    assert to_weekly(steps, 0.25).tolist() == [HEALTH, HEALTH, HEALTH, RELAPSE]


# The three cases below pin where a week begins. Every other case here puts its
# relapse in the interior of a week, so a window shifted by one step would carry
# the relapse along and pass. The first two straddle the boundary at t = 1.0 and
# fail for a shift in either direction, and the third puts that boundary between
# two samples, where rounding the boundary down instead of up changes the answer.


def test_a_week_begins_at_its_own_multiple_of_a_week() -> None:
    steps = np.full(8, HEALTH)
    steps[4] = RELAPSE
    assert to_weekly(steps, 0.25).tolist() == [HEALTH, RELAPSE]


def test_the_last_step_before_a_boundary_still_belongs_to_the_earlier_week() -> None:
    steps = np.full(8, HEALTH)
    steps[3] = RELAPSE
    assert to_weekly(steps, 0.25).tolist() == [RELAPSE, HEALTH]


def test_a_step_that_does_not_divide_the_week_splits_at_the_boundary() -> None:
    # At dt = 0.4 the boundary at t = 1.0 falls between two samples, so week 0
    # holds three of them, at 0.0, 0.4 and 0.8, and week 1 only two. A week is
    # the samples whose time lies inside it and not a fixed count of them.
    steps = np.full(5, HEALTH)
    steps[2] = RELAPSE
    assert to_weekly(steps, 0.4).tolist() == [RELAPSE, HEALTH]


def test_only_complete_weeks_are_returned() -> None:
    assert to_weekly(np.full(17, HEALTH), 0.25).size == 4


def test_weekly_rounding_works_on_a_cohort() -> None:
    first = np.full(8, HEALTH)
    first[2] = RELAPSE
    second = np.full(8, HEALTH)
    second[7] = RELAPSE
    weekly = to_weekly(np.vstack([first, second]), 0.25)
    assert weekly.tolist() == [[RELAPSE, HEALTH], [HEALTH, RELAPSE]]


def test_a_record_shorter_than_a_week_is_rejected() -> None:
    with pytest.raises(ValueError, match="one complete week"):
        to_weekly(np.full(3, HEALTH), 0.25)


def test_a_week_shorter_than_the_step_is_rejected() -> None:
    with pytest.raises(ValueError, match="longer than week"):
        to_weekly(np.full(8, HEALTH), 0.25, week=0.1)


@pytest.mark.parametrize(
    ("dt", "week", "message"),
    [
        (0.0, 1.0, "dt must be"),
        (-0.25, 1.0, "dt must be"),
        (0.25, 0.0, "week must be"),
        (0.25, -1.0, "week must be"),
    ],
)
def test_to_weekly_rejects_out_of_range_arguments(*, dt: float, week: float, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        to_weekly(np.full(8, HEALTH), dt, week=week)


# Run length encoding of a weekly record.


def test_durations_obey_the_schema() -> None:
    weekly = np.array([[HEALTH, HEALTH, RELAPSE, RELAPSE, HEALTH, HEALTH]])
    validate(durations(weekly), "durations")


def test_only_a_final_remission_is_censored() -> None:
    weekly = np.array(
        [
            [HEALTH, HEALTH, RELAPSE, RELAPSE, HEALTH, HEALTH],
            [HEALTH, RELAPSE, HEALTH, HEALTH, RELAPSE, RELAPSE],
        ]
    )
    runs = durations(weekly)
    assert runs["censored"].tolist() == [False, False, True, False, False, False, False]


def test_durations_number_the_patients_from_one() -> None:
    weekly = np.array([[HEALTH, RELAPSE], [RELAPSE, HEALTH]])
    assert sorted(set(durations(weekly)["patient_id"])) == ["p0001", "p0002"]


def test_given_patient_ids_are_used() -> None:
    weekly = np.array([[HEALTH, RELAPSE], [RELAPSE, HEALTH]])
    runs = durations(weekly, patient_ids=["a", "b"])
    assert sorted(set(runs["patient_id"])) == ["a", "b"]


def test_the_wrong_number_of_patient_ids_is_rejected() -> None:
    weekly = np.array([[HEALTH, RELAPSE], [RELAPSE, HEALTH]])
    with pytest.raises(ValueError, match="patient_ids must"):
        durations(weekly, patient_ids=["a"])


def test_an_empty_weekly_record_is_rejected() -> None:
    with pytest.raises(ValueError, match="at least one record"):
        durations(np.zeros(0, dtype=np.int64))


# The cohort engine.


@pytest.fixture(scope="module")
def cohort(band_well: tuple[DoubleWell, float]) -> pd.DataFrame:
    well, sigma = band_well
    return simulate_weekly(
        well, sigma, 1500, n_paths=100, dt=0.02, rng=13, band_fraction=BAND_FRACTION
    )


def test_a_record_of_no_whole_weeks_is_rejected(
    saddle_well: tuple[DoubleWell, float],
) -> None:
    well, sigma = saddle_well
    with pytest.raises(ValueError, match="n_weeks must be"):
        simulate_weekly(well, sigma, 0)


def test_a_cohort_that_blows_up_is_refused_rather_than_recorded(
    saddle_well: tuple[DoubleWell, float],
) -> None:
    # A blown up path freezes its state instead of switching, and the frame that
    # comes out passes io.validate and reads as an ordinary clinical record, so
    # the run has to be refused here rather than handed on.
    well, _ = saddle_well
    with pytest.raises(ValueError, match="left the finite range"):
        simulate_weekly(well, 2.0, 20, n_paths=4, dt=0.3, rng=0)


def test_the_simulated_cohort_obeys_the_weekly_schema(cohort: pd.DataFrame) -> None:
    validate(cohort, "weekly")


def test_every_patient_is_followed_for_the_whole_run(cohort: pd.DataFrame) -> None:
    assert cohort.groupby("patient_id")["week"].size().unique().tolist() == [1500]


def test_the_simulated_relapse_burden_matches_the_cohort(cohort: pd.DataFrame) -> None:
    # A relapse of continuous length L is marked in every week it touches, so it
    # occupies L + 1 whole weeks on average, and the remissions around it lose
    # that week. One cycle of the calibrated potential is TAU_HEALTH +
    # TAU_RELAPSE weeks long and carries one relapse, so the rounding rule puts
    # (TAU_RELAPSE + 1) of those weeks in the no health state. The cohort module
    # will undo the shift to reproduce the mean durations the paper prints.
    cycle = TAU_HEALTH + TAU_RELAPSE
    burden = (cohort["state"] == RELAPSE).mean()
    assert burden == pytest.approx((TAU_RELAPSE + WEEK) / cycle, rel=0.15)


@pytest.fixture(scope="module")
def rounding_pair(
    band_well: tuple[DoubleWell, float],
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]:
    """Return a cohort of state series and the weekly record rounded from it."""
    well, sigma = band_well
    paths = simulate_paths(well, sigma, 3000.0, dt=0.02, n_paths=20, rng=29)
    states = to_states(paths.x, well, BAND_FRACTION)
    return states, to_weekly(states, 0.02)


def test_the_rounding_rule_adds_about_one_week_to_every_relapse(
    rounding_pair: tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]],
) -> None:
    states, weekly = rounding_pair
    continuous = float(np.count_nonzero(states == RELAPSE)) * 0.02
    expected = continuous + WEEK * count_relapses(states)
    assert float(np.count_nonzero(weekly == RELAPSE)) == pytest.approx(expected, rel=0.05)


def test_band_episodes_run_long_on_the_integration_grid(
    rounding_pair: tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]],
    band_well: tuple[DoubleWell, float],
) -> None:
    # to_states tests the two band thresholds at the grid points and carries no
    # Brownian bridge correction, unlike exit_times, so it finds a crossing a
    # step late and every episode of the path runs long before any rounding.
    # At dt = 0.02 the mean relapse episode measured about 13 percent above the
    # band target, over more than 60000 episodes with a standard error of half a
    # percent, falling to about 7 percent at dt = 0.005.
    states, _ = rounding_pair
    well, sigma = band_well
    endpoints = passage_endpoints(well, "relapse", "band", BAND_FRACTION)
    target = mfpt(well, sigma, "relapse", *endpoints)
    measured = complete_episode_lengths(states, 0.02, RELAPSE).mean()
    assert target < measured < 1.5 * target


def test_relapses_a_few_days_apart_merge_into_one_weekly_episode(
    rounding_pair: tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]],
) -> None:
    # The hysteresis band removes the chatter of a path wobbling across the
    # barrier top, but a genuine return to health that lasts less than a week
    # leaves no week of its own, so the relapses around it become one episode.
    # The weekly record therefore holds fewer and longer episodes than the path.
    states, weekly = rounding_pair
    assert count_relapses(weekly) < count_relapses(states)


def test_merging_lengthens_the_weekly_relapses(cohort: pd.DataFrame) -> None:
    runs = complete_runs(cohort)
    assert runs.loc[runs["state"] == RELAPSE, "duration_w"].mean() > TAU_RELAPSE + WEEK


def test_merging_lengthens_the_weekly_remissions(cohort: pd.DataFrame) -> None:
    runs = complete_runs(cohort)
    assert runs.loc[runs["state"] == HEALTH, "duration_w"].mean() > TAU_HEALTH - WEEK


# The two assertions below pin the weekly episode means where they were
# measured. A weekly record reproduces neither the continuous targets of the
# calibration, 4.3 and 100 weeks, nor those targets shifted by the one week the
# rounding rule adds, 5.3 and 99: the grid bias of to_states lengthens every
# episode before the rounding, and the merging of the two tests above lengthens
# the surviving ones again. Over seeds 7, 13, 21, 42 and 100 the mean weekly
# relapse measured 6.64 to 6.96 weeks and the mean weekly remission 111 to 123,
# so these bounds are two sided around the behaviour and a regression that moved
# either duration by half would fail them.


def test_the_weekly_relapses_run_about_seven_weeks(cohort: pd.DataFrame) -> None:
    runs = complete_runs(cohort)
    relapses = runs.loc[runs["state"] == RELAPSE, "duration_w"]
    assert relapses.mean() == pytest.approx(6.9, rel=0.15)


def test_the_weekly_remissions_run_about_a_hundred_and_ten_weeks(cohort: pd.DataFrame) -> None:
    runs = complete_runs(cohort)
    remissions = runs.loc[runs["state"] == HEALTH, "duration_w"]
    assert remissions.mean() == pytest.approx(112.0, rel=0.15)


def complete_episode_lengths(
    states: npt.NDArray[np.int64],
    dt: float,
    state: int,
) -> npt.NDArray[np.float64]:
    """Return the lengths in weeks of the episodes of one state that ran their course.

    The first and the last episode of every series are dropped, because the
    start of the record cuts the one and the end of follow up cuts the other.
    """
    lengths: list[float] = []
    for row in states:
        change = np.flatnonzero(row[1:] != row[:-1]) + 1
        starts = np.concatenate(([0], change))
        stops = np.concatenate((change, [row.size]))
        for index in range(1, starts.size - 1):
            if row[starts[index]] == state:
                lengths.append(float(stops[index] - starts[index]) * dt)
    return np.array(lengths, dtype=np.float64)


def count_relapses(states: npt.NDArray[np.int64]) -> int:
    """Count the runs of relapse across a cohort of state series."""
    padded = np.pad(states, ((0, 0), (1, 1)), constant_values=HEALTH)
    onsets = (padded[:, 1:] == RELAPSE) & (padded[:, :-1] != RELAPSE)
    return int(np.count_nonzero(onsets))


def complete_runs(weekly: pd.DataFrame) -> pd.DataFrame:
    """Return the runs of a weekly record that the end of follow up did not cut."""
    runs = weekly_to_durations(weekly)
    last = runs.groupby("patient_id")["run_index"].transform("max")
    return runs[runs["run_index"] < last]


def test_the_seed_alias_is_exported() -> None:
    # Seed names the rng argument of every public function here, so a caller
    # annotating a wrapper needs it advertised, as model.Side and model.Passage
    # are.
    assert "Seed" in msrelapse.simulate.__all__


def test_docstring_examples_run() -> None:
    results = doctest.testmod(msrelapse.simulate)
    assert results.attempted > 0
    assert results.failed == 0
