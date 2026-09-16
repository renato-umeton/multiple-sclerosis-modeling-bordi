from __future__ import annotations

import doctest
import math

import numpy as np
import numpy.typing as npt
import pandas as pd
import pytest

import msrelapse.cohort
from msrelapse._params import PAPER
from msrelapse.cohort import (
    Cohort,
    CohortSpec,
    Engine,
    Sampler,
    bordi2013_spec,
    constant,
    continuous_targets,
    draw,
    empirical,
    from_histogram,
    generate,
    lognormal_around,
    naive_mean_targets,
    paper_patients,
    per_patient_params,
)
from msrelapse.io import validate
from msrelapse.model import DoubleWell, barrier_ratio_from_durations, calibrate, fold_beta
from msrelapse.simulate import MAX_DT

RELAPSE = PAPER.state_no_health.value
HEALTH = PAPER.state_health.value
TAU_HEALTH = PAPER.tau_health_cohort_weeks.value
TAU_RELAPSE = PAPER.tau_no_health_cohort_weeks.value
ALPHA = PAPER.alpha_reference.value

# How far apart the two engines may land on the same spec. Over the eight seeds
# 5 to 12 of the cohort of sde_spec, the weekly naive mean of the stochastic
# records sat between 9.4 and 19.8 percent above the weekly naive mean of the
# renewal records on the remission side and between 9.5 and 18.6 percent above
# it on the relapse side; the table over both time steps and three band
# fractions is in the Notes of CohortSpec. That gap is the merging of two
# relapses on either side of a remission shorter than a week, which a weekly
# record cannot express and no calibration can undo: at the default band
# fraction of 0.4 about one simulated remission in ten is that short. The bound
# is the largest measured gap plus a quarter of it again for the sampling noise
# of a cohort of 120 records, and it is well inside the 18 to 32 percent the
# same comparison showed at a band fraction of 0.3, so a regression that
# narrowed the band back would fail here.
SDE_TOLERANCE = 0.25

CLOSE = 0.1

# How far the two naive means of a cohort of 20000 patients may sit from the
# targets naive_mean_targets was asked for. The Monte Carlo accuracy at that
# size is a few tenths of a percent, so two percent is a comfortable bound.
ROUND_TRIP_TOLERANCE = 0.02
N_CALIBRATION = 20000

# Length of the record a renewal cohort has to have for its naive means to be
# the means it was generated from. A record of a few hundred weeks holds only a
# handful of remissions of about a hundred weeks each, the last of them cut off
# by the end of follow up, and the naive mean of what is left runs a quarter
# low. See test_the_paper_follow_up_windows_bias_the_remission_mean_low.
LONG_RECORD_WEEKS = 6000.0

N_DRAWS = 20000


def naive_mean(durations: pd.DataFrame, state: int) -> float:
    """Return the mean recorded duration of one state, censoring ignored."""
    return float(durations.loc[durations["state"] == state, "duration_w"].mean())


def relative_gap(value: float, target: float) -> float:
    """Return how far `value` sits from `target`, as a fraction of `target`."""
    return abs(value / target - 1.0)


def only_value(frame: pd.DataFrame, column: str) -> float:
    """Return the single numeric value a one row frame holds in `column`."""
    values = frame[column].to_numpy(dtype=np.float64)
    assert values.size == 1
    return float(values[0])


def worked_example(number: int) -> pd.DataFrame:
    """Return the one row of paper_patients that belongs to one patient."""
    frame = paper_patients()
    return frame[frame["patient_id"] == number]


def sde_spec(engine: Engine = "sde", weekly: bool = True) -> CohortSpec:
    """Return the cohort the two engines are compared on, at the defaults of CohortSpec."""
    return CohortSpec(
        n=120,
        tau_health=TAU_HEALTH,
        tau_relapse=TAU_RELAPSE,
        followup_weeks=1000.0,
        engine=engine,
        weekly=weekly,
    )


def paper_cohort_durations() -> pd.DataFrame:
    """Return a durations frame whose two naive means are the cohort means of the paper.

    The schema records a duration as a whole number of weeks, so the paper's
    mean relapse of 4.3 weeks cannot be one row: it is the mean of ten relapses
    of four and five weeks instead. The remissions are ten runs of the paper's
    100 weeks, and the runs alternate as a run length encoding must.
    """
    relapses = [4, 4, 4, 4, 4, 4, 4, 5, 5, 5]
    remission = int(PAPER.tau_health_cohort_weeks.value)
    states: list[int] = []
    lengths: list[int] = []
    for length in relapses:
        states += [RELAPSE, HEALTH]
        lengths += [length, remission]
    return pd.DataFrame(
        {
            "patient_id": pd.Series(["p0001"] * len(states), dtype=object),
            "run_index": pd.Series(range(len(states)), dtype=np.int64),
            "state": pd.Series(states, dtype=np.int64),
            "duration_w": pd.Series(lengths, dtype=np.int64),
            "censored": pd.Series([False] * len(states), dtype=bool),
        }
    )


@pytest.fixture(scope="module")
def paper_cohort() -> Cohort:
    return generate(bordi2013_spec(), rng=20130910)


@pytest.fixture(scope="module")
def unmatched_paper_cohort() -> Cohort:
    return generate(bordi2013_spec(match_naive_means=False), rng=20130910)


@pytest.fixture(scope="module")
def sde_cohort() -> Cohort:
    return generate(sde_spec(), rng=5)


@pytest.fixture(scope="module")
def renewal_twin() -> Cohort:
    return generate(sde_spec(engine="renewal"), rng=5)


def test_paper_cohort_has_one_record_per_patient_of_the_study(paper_cohort: Cohort) -> None:
    assert paper_cohort.events["patient_id"].nunique() == PAPER.n_patients.value


def test_paper_cohort_frames_obey_their_schemas(paper_cohort: Cohort) -> None:
    assert paper_cohort.weekly is not None
    assert paper_cohort.durations is not None
    validate(paper_cohort.weekly, "weekly")
    validate(paper_cohort.durations, "durations")
    validate(paper_cohort.events, "events")


# The twin is measured the way the paper measured its own cohort, with the naive
# mean of every recorded run of a state, censored final remission included. At
# the seed of the shipped files it lands 4.3 percent above the printed remission
# and 4.2 percent below the printed relapse. Seventy records of these lengths
# hold 261 relapses and 261 remissions at that seed, so each mean carries a
# standard error of about six percent of itself, 5.8 percent on the relapse side
# and 6.3 percent on the remission side, and moves by about that much from one
# seed to the next: over the sixty consecutive seeds from this one the relapse
# mean has a standard deviation of 5.5 percent of itself and the remission mean
# one of 6.2 percent. The CLOSE bound of ten percent of the printed value is
# therefore about 1.8 standard errors wide on the relapse side and about 1.5 on
# the remission side, and neither mean meets it on every seed: over those sixty
# the relapse mean lands inside it on 57 and the remission mean on 52.
# msrelapse.datasets carries the same sweep beside the tolerances of the closing
# table. The seed is the one the shipped data carries, not a seed chosen to pass.


def test_paper_cohort_reproduces_the_printed_relapse_duration(paper_cohort: Cohort) -> None:
    assert paper_cohort.durations is not None
    assert relative_gap(naive_mean(paper_cohort.durations, RELAPSE), TAU_RELAPSE) < CLOSE


def test_paper_cohort_reproduces_the_printed_remission_duration(paper_cohort: Cohort) -> None:
    assert paper_cohort.durations is not None
    assert relative_gap(naive_mean(paper_cohort.durations, HEALTH), TAU_HEALTH) < CLOSE


def test_the_paper_follow_up_windows_bias_the_remission_mean_low(
    unmatched_paper_cohort: Cohort,
) -> None:
    # A remission of about a hundred weeks rarely fits twice into a record of a
    # few hundred, so the remissions a short window records are the short ones
    # plus one that the end of follow up cut off. The naive mean of a cohort
    # generated at exactly the printed hundred weeks therefore lands about a
    # quarter below it, and the printed hundred weeks of the study, a naive mean
    # over windows of the same lengths, understates its own cohort the same way.
    # This is what bordi2013_spec undoes by default, and turning the matching off
    # is what shows the size of it.
    assert unmatched_paper_cohort.durations is not None
    mean = naive_mean(unmatched_paper_cohort.durations, HEALTH)
    assert mean < 0.85 * TAU_HEALTH


def test_the_matched_spec_is_generated_above_the_printed_remission() -> None:
    spec = bordi2013_spec()
    assert isinstance(spec.tau_health, float)
    assert spec.tau_health > TAU_HEALTH


def test_the_matched_spec_records_the_printed_means_it_was_aimed_at() -> None:
    spec = bordi2013_spec()
    assert spec.naive_tau_health == TAU_HEALTH
    assert spec.naive_tau_relapse == TAU_RELAPSE


def test_an_unmatched_spec_records_no_naive_means() -> None:
    spec = bordi2013_spec(match_naive_means=False)
    assert spec.naive_tau_health is None
    assert spec.naive_tau_relapse is None


def test_the_matched_cohort_carries_its_naive_targets_per_patient(paper_cohort: Cohort) -> None:
    assert paper_cohort.patients["naive_tau_health"].eq(TAU_HEALTH).all()
    assert paper_cohort.patients["naive_tau_relapse"].eq(TAU_RELAPSE).all()


def test_an_unmatched_cohort_leaves_the_naive_columns_missing(
    unmatched_paper_cohort: Cohort,
) -> None:
    assert unmatched_paper_cohort.patients["naive_tau_health"].isna().all()
    assert unmatched_paper_cohort.patients["naive_tau_relapse"].isna().all()


def test_the_matched_provenance_names_the_means_it_was_aimed_at(paper_cohort: Cohort) -> None:
    assert str(TAU_HEALTH) in paper_cohort.provenance
    assert str(TAU_RELAPSE) in paper_cohort.provenance


def test_the_unmatched_provenance_claims_no_matching(unmatched_paper_cohort: Cohort) -> None:
    assert "naive means" not in unmatched_paper_cohort.provenance


def test_a_long_renewal_record_reproduces_both_printed_durations() -> None:
    spec = CohortSpec(
        n=10,
        tau_health=TAU_HEALTH,
        tau_relapse=TAU_RELAPSE,
        followup_weeks=LONG_RECORD_WEEKS,
    )
    cohort = generate(spec, rng=20130910)
    assert cohort.durations is not None
    assert relative_gap(naive_mean(cohort.durations, RELAPSE), TAU_RELAPSE) < CLOSE
    assert relative_gap(naive_mean(cohort.durations, HEALTH), TAU_HEALTH) < CLOSE


def test_paper_cohort_follow_up_spans_most_of_the_printed_range(paper_cohort: Cohort) -> None:
    # The test below pins that every window lies inside the histogram, which the
    # sampler guarantees by construction. This pins the other direction, which it
    # does not: that the seventy windows are spread across the bins rather than
    # collapsed onto the modal one. The printed range runs 40 to 1311 weeks and
    # the shipped seed draws 48 to 1272, so the drawn span is 96 percent of it,
    # against the half this asks for.
    follow_up = paper_cohort.patients["followup_weeks"]
    printed_span = PAPER.rr_phase_max_weeks.value - PAPER.rr_phase_min_weeks.value
    assert follow_up.max() - follow_up.min() > 0.5 * printed_span


def test_paper_cohort_follow_up_lies_inside_the_histogram(paper_cohort: Cohort) -> None:
    edges = PAPER.fig3_bin_edges_weeks.value
    follow_up = paper_cohort.patients["followup_weeks"]
    assert follow_up.min() >= edges[0]
    assert follow_up.max() <= edges[-1]


def test_paper_cohort_patients_frame_has_one_row_per_patient(paper_cohort: Cohort) -> None:
    assert len(paper_cohort.patients) == PAPER.n_patients.value


def test_paper_cohort_declares_itself_synthetic(paper_cohort: Cohort) -> None:
    assert "synthetic" in paper_cohort.provenance.lower()


def test_cohort_repr_names_the_size_the_engine_and_the_article(paper_cohort: Cohort) -> None:
    text = repr(paper_cohort)
    assert str(PAPER.n_patients.value) in text
    assert "renewal" in text
    assert text.count(paper_cohort.citation) == 1


def test_sde_cohort_frames_obey_their_schemas(sde_cohort: Cohort) -> None:
    assert sde_cohort.weekly is not None
    assert sde_cohort.durations is not None
    validate(sde_cohort.weekly, "weekly")
    validate(sde_cohort.durations, "durations")
    validate(sde_cohort.events, "events")


def test_sde_cohort_reproduces_the_relapse_duration(sde_cohort: Cohort) -> None:
    assert sde_cohort.durations is not None
    assert relative_gap(naive_mean(sde_cohort.durations, RELAPSE), TAU_RELAPSE) < SDE_TOLERANCE


def test_sde_cohort_reproduces_the_remission_duration(sde_cohort: Cohort) -> None:
    assert sde_cohort.durations is not None
    assert relative_gap(naive_mean(sde_cohort.durations, HEALTH), TAU_HEALTH) < SDE_TOLERANCE


def test_the_rounding_correction_shortens_the_weekly_relapse(sde_cohort: Cohort) -> None:
    # weekly=False turns the correction off while leaving everything else alone,
    # because a simulated record is downsampled to whole weeks whatever the spec
    # says. The uncorrected cohort therefore shows what the correction is worth:
    # its weekly relapses carry the extra week the rounding rule grants them.
    assert sde_cohort.durations is not None
    uncorrected = generate(sde_spec(weekly=False), rng=5)
    assert uncorrected.durations is not None
    corrected_mean = naive_mean(sde_cohort.durations, RELAPSE)
    uncorrected_mean = naive_mean(uncorrected.durations, RELAPSE)
    assert uncorrected_mean - corrected_mean > 0.5 * PAPER.time_resolution_weeks.value
    assert relative_gap(corrected_mean, TAU_RELAPSE) < relative_gap(uncorrected_mean, TAU_RELAPSE)


def test_constant_targets_are_calibrated_once_for_the_whole_cohort(sde_cohort: Cohort) -> None:
    patients = sde_cohort.patients
    assert patients["beta"].notna().all()
    assert patients["sigma"].notna().all()
    assert patients["beta"].nunique() == 1
    assert patients["sigma"].nunique() == 1


def test_continuous_targets_shift_a_week_between_the_two_states() -> None:
    health, relapse = continuous_targets(TAU_HEALTH, TAU_RELAPSE, weekly=True)
    week = PAPER.time_resolution_weeks.value
    assert health == pytest.approx(TAU_HEALTH + week)
    assert relapse == pytest.approx(TAU_RELAPSE - week)


def test_continuous_targets_leave_an_unrounded_record_alone() -> None:
    assert continuous_targets(TAU_HEALTH, TAU_RELAPSE, weekly=False) == (TAU_HEALTH, TAU_RELAPSE)


def test_continuous_targets_refuse_a_relapse_the_correction_would_empty() -> None:
    with pytest.raises(ValueError, match="rounding correction"):
        continuous_targets(TAU_HEALTH, 1.2, weekly=True)


def test_continuous_targets_refuse_a_remission_that_is_not_a_duration() -> None:
    with pytest.raises(ValueError, match="tau_health must be"):
        continuous_targets(0.0, TAU_RELAPSE)


def test_continuous_targets_refuse_a_relapse_that_is_not_a_duration() -> None:
    with pytest.raises(ValueError, match="tau_relapse must be"):
        continuous_targets(TAU_HEALTH, 0.0)


def paper_windows() -> Sampler:
    """Return the follow up sampler of Figure 3, the windows of the study."""
    return from_histogram(PAPER.fig3_bin_edges_weeks.value, PAPER.fig3_counts.value)


@pytest.fixture(scope="module")
def round_trip() -> Cohort:
    """Return a large cohort generated from the targets matched to the printed means.

    The cohort is generated through the public generator, with fresh follow up
    windows and a seed the calibration never saw, so what the two tests below
    measure is the naive means of a cohort the calibration has no memory of.
    """
    tau_health, tau_relapse = naive_mean_targets(TAU_HEALTH, TAU_RELAPSE, paper_windows())
    spec = CohortSpec(
        n=N_CALIBRATION,
        tau_health=tau_health,
        tau_relapse=tau_relapse,
        followup_weeks=paper_windows(),
    )
    return generate(spec, rng=777)


def test_naive_mean_targets_round_trip_the_remission_mean(round_trip: Cohort) -> None:
    assert round_trip.durations is not None
    measured = naive_mean(round_trip.durations, HEALTH)
    assert relative_gap(measured, TAU_HEALTH) < ROUND_TRIP_TOLERANCE


def test_naive_mean_targets_round_trip_the_relapse_mean(round_trip: Cohort) -> None:
    assert round_trip.durations is not None
    measured = naive_mean(round_trip.durations, RELAPSE)
    assert relative_gap(measured, TAU_RELAPSE) < ROUND_TRIP_TOLERANCE


def test_naive_mean_targets_aim_above_the_naive_mean_they_are_given() -> None:
    # Every run of a state that the end of follow up cuts short is recorded at
    # less than its length, so the naive mean is the shorter of the two and the
    # generative mean has to be above it.
    tau_health, _ = naive_mean_targets(TAU_HEALTH, TAU_RELAPSE, paper_windows())
    assert tau_health > TAU_HEALTH


def test_naive_mean_targets_refuse_a_target_no_window_can_reach() -> None:
    # No remission can be recorded as longer than the window it sits in, so a
    # naive mean of 10000 weeks is out of reach of windows of 300.
    with pytest.raises(ValueError, match="not reachable"):
        naive_mean_targets(10000.0, TAU_RELAPSE, 300.0, n_calibration=500)


def test_naive_mean_targets_refuse_a_duration_shorter_than_a_week() -> None:
    with pytest.raises(ValueError, match="naive_tau_relapse must be"):
        naive_mean_targets(TAU_HEALTH, 0.5, 300.0, n_calibration=500)


def test_naive_mean_targets_refuse_an_empty_calibration_cohort() -> None:
    with pytest.raises(ValueError, match="n_calibration must be"):
        naive_mean_targets(TAU_HEALTH, TAU_RELAPSE, 300.0, n_calibration=0)


def test_naive_mean_targets_refuse_a_window_too_short_to_hold_both_states() -> None:
    # Over windows of two weeks a relapse of fifty weeks fills every record, so
    # the calibration cohort holds no remission at all and its naive mean does
    # not exist. The message says which of the two states is missing.
    with pytest.raises(ValueError, match="remission run"):
        naive_mean_targets(1.0, 50.0, 2.0, n_calibration=500)


def test_naive_mean_targets_return_a_number_of_six_significant_figures() -> None:
    # The pair seeds a shipped file, so it is rounded to a number a platform
    # whose exponential differs in its last bit still returns. See the Notes of
    # cohort._solve_naive_mean for why the search is stopped tightly enough for
    # the rounding to reach.
    for value in naive_mean_targets(50.0, 4.0, 300.0, n_calibration=1000):
        assert value == float(f"{value:.6g}")


def test_naive_mean_targets_refuse_an_unknown_start_state() -> None:
    with pytest.raises(ValueError, match="start_state must be one of"):
        naive_mean_targets(
            TAU_HEALTH,
            TAU_RELAPSE,
            300.0,
            start_state="progressive",  # type: ignore[arg-type]
            n_calibration=500,
        )


def test_a_short_window_moves_the_relapse_target_too() -> None:
    # A relapse still running at the end of follow up is recorded truncated, so
    # the naive relapse mean falls short as well. Over windows of 300 weeks the
    # shortfall passes one percent and the second bisection runs.
    _, tau_relapse = naive_mean_targets(50.0, 4.0, 300.0, n_calibration=2000)
    assert tau_relapse > 4.0


def test_a_long_window_leaves_the_relapse_target_alone() -> None:
    # Over windows of 5000 weeks a record ends inside a relapse too rarely to
    # move the naive mean by a percent, so the relapse is left where it was.
    _, tau_relapse = naive_mean_targets(50.0, 4.0, 5000.0, n_calibration=500)
    assert tau_relapse == 4.0


def test_naive_mean_targets_are_reproducible_from_their_seed() -> None:
    first = naive_mean_targets(50.0, 4.0, 300.0, n_calibration=1000, seed=3)
    second = naive_mean_targets(50.0, 4.0, 300.0, n_calibration=1000, seed=3)
    assert first == second


def test_sde_engine_refuses_a_relapse_the_correction_would_empty() -> None:
    spec = CohortSpec(
        n=1,
        tau_health=TAU_HEALTH,
        tau_relapse=1.2,
        followup_weeks=50.0,
        engine="sde",
    )
    with pytest.raises(ValueError, match="rounding correction"):
        generate(spec, rng=0)


def test_renewal_engine_accepts_a_relapse_shorter_than_two_weeks() -> None:
    spec = CohortSpec(n=2, tau_health=TAU_HEALTH, tau_relapse=1.2, followup_weeks=200.0)
    cohort = generate(spec, rng=0)
    validate(cohort.events, "events")


def test_the_two_engines_agree_on_the_relapse_duration(
    sde_cohort: Cohort, renewal_twin: Cohort
) -> None:
    """The two engines are compared with each other, not with the targets.

    Both cohorts are generated from the same spec and so are cut short by the
    same distribution of follow up windows, which is what makes the comparison
    fair: the naive mean of either engine sits below the mean it was generated
    from, and by the same amount. What is left over is the merging of two
    relapses on either side of a sub-week remission, which only the stochastic
    engine produces. See SDE_TOLERANCE for the measured size of it.
    """
    assert sde_cohort.durations is not None
    assert renewal_twin.durations is not None
    stochastic = naive_mean(sde_cohort.durations, RELAPSE)
    renewal = naive_mean(renewal_twin.durations, RELAPSE)
    assert relative_gap(stochastic, renewal) < SDE_TOLERANCE


def test_the_two_engines_agree_on_the_remission_duration(
    sde_cohort: Cohort, renewal_twin: Cohort
) -> None:
    """The remission side of the same comparison, under the same bound.

    A merged pair of relapses swallows the remission between them, so the same
    effect lengthens the surviving remissions as well as the relapses, and both
    sides of the record are measured against the same tolerance.
    """
    assert sde_cohort.durations is not None
    assert renewal_twin.durations is not None
    stochastic = naive_mean(sde_cohort.durations, HEALTH)
    renewal = naive_mean(renewal_twin.durations, HEALTH)
    assert relative_gap(stochastic, renewal) < SDE_TOLERANCE


def continuous_cohort() -> Cohort:
    """Return a renewal cohort whose episodes are not rounded to whole weeks."""
    spec = CohortSpec(
        n=8,
        tau_health=TAU_HEALTH,
        tau_relapse=TAU_RELAPSE,
        followup_weeks=600.0,
        weekly=False,
    )
    return generate(spec, rng=4)


def test_an_unrounded_renewal_cohort_has_no_weekly_frame() -> None:
    cohort = continuous_cohort()
    assert cohort.weekly is None
    assert cohort.durations is None


def test_an_unrounded_renewal_cohort_still_has_an_events_frame() -> None:
    cohort = continuous_cohort()
    validate(cohort.events, "events")


def test_an_unrounded_renewal_cohort_records_fractions_of_a_week() -> None:
    events = continuous_cohort().events
    spans = (events["relapse_end"] - events["relapse_onset"]).to_numpy(dtype=np.float64)
    assert spans.size > 0
    assert np.all(spans != np.round(spans))


def test_constant_repeats_one_value() -> None:
    values = draw(constant(7.5), 0, 4)
    assert values.tolist() == [7.5, 7.5, 7.5, 7.5]


def test_a_bare_float_is_a_sampler_of_its_own() -> None:
    assert draw(7.5, 0, 3).tolist() == [7.5, 7.5, 7.5]


def test_lognormal_around_hits_the_mean_it_was_given() -> None:
    values = draw(lognormal_around(100.0, 0.6), 1, N_DRAWS)
    assert relative_gap(float(values.mean()), 100.0) < 0.05


def test_lognormal_around_hits_the_spread_it_was_given() -> None:
    values = draw(lognormal_around(100.0, 0.6), 1, N_DRAWS)
    cv = float(values.std(ddof=1)) / float(values.mean())
    assert relative_gap(cv, 0.6) < 0.15


def test_empirical_returns_only_the_values_it_was_given() -> None:
    pool = [2.0, 9.0, 40.0]
    values = draw(empirical(pool), 2, 500)
    assert set(values.tolist()) <= set(pool)


def test_from_histogram_stays_inside_the_outer_edges() -> None:
    edges = PAPER.fig3_bin_edges_weeks.value
    values = draw(from_histogram(edges, PAPER.fig3_counts.value), 3, N_DRAWS)
    assert values.min() >= edges[0]
    assert values.max() <= edges[-1]


def test_from_histogram_reproduces_the_bar_heights() -> None:
    edges = np.asarray(PAPER.fig3_bin_edges_weeks.value, dtype=np.float64)
    counts = np.asarray(PAPER.fig3_counts.value, dtype=np.float64)
    values = draw(from_histogram(edges.tolist(), PAPER.fig3_counts.value), 3, N_DRAWS)
    observed = np.histogram(values, bins=edges)[0] / values.size
    assert np.max(np.abs(observed - counts / counts.sum())) < 0.03


def test_draw_refuses_a_cohort_of_no_patients() -> None:
    with pytest.raises(ValueError, match="n must be at least 1"):
        draw(4.0, 0, 0)


def test_constant_refuses_a_value_that_is_not_a_number() -> None:
    with pytest.raises(ValueError, match="value must be a finite number"):
        constant(math.nan)


def test_draw_refuses_a_sample_that_is_not_a_positive_duration() -> None:
    def zeros(rng: np.random.Generator, n: int) -> npt.NDArray[np.float64]:
        return np.zeros(n, dtype=np.float64)

    with pytest.raises(ValueError, match="positive and finite"):
        draw(zeros, 0, 5)


def test_draw_refuses_a_negative_constant() -> None:
    with pytest.raises(ValueError, match="positive and finite"):
        draw(constant(-1.0), 0, 5)


def test_draw_refuses_a_sampler_of_the_wrong_length() -> None:
    def too_few(rng: np.random.Generator, n: int) -> npt.NDArray[np.float64]:
        return np.ones(n - 1, dtype=np.float64)

    with pytest.raises(ValueError, match="one value per patient"):
        draw(too_few, 0, 5)


def test_lognormal_around_refuses_a_mean_that_is_not_a_duration() -> None:
    with pytest.raises(ValueError, match="mean must be positive"):
        lognormal_around(0.0, 0.5)


def test_lognormal_around_refuses_a_spread_that_is_not_positive() -> None:
    with pytest.raises(ValueError, match="cv must be positive"):
        lognormal_around(10.0, 0.0)


def test_empirical_refuses_an_empty_pool() -> None:
    with pytest.raises(ValueError, match="at least one value"):
        empirical([])


def test_empirical_refuses_a_pool_that_is_not_a_flat_list_of_durations() -> None:
    with pytest.raises(ValueError, match="one dimensional"):
        empirical([[1.0, 2.0]])


def test_empirical_refuses_a_pool_holding_a_value_that_is_not_a_number() -> None:
    with pytest.raises(ValueError, match="every value must be a finite number"):
        empirical([math.nan])


def test_from_histogram_refuses_a_single_edge_with_no_bin_under_it() -> None:
    with pytest.raises(ValueError, match="at least two edges"):
        from_histogram([0.0], [])


def test_from_histogram_refuses_a_negative_count() -> None:
    with pytest.raises(ValueError, match="must not be negative"):
        from_histogram([0.0, 1.0], [-1])


def test_from_histogram_refuses_counts_that_do_not_match_the_edges() -> None:
    with pytest.raises(ValueError, match="one count per bin"):
        from_histogram([0.0, 1.0, 2.0], [3])


def test_from_histogram_refuses_edges_that_do_not_increase() -> None:
    with pytest.raises(ValueError, match="ascending"):
        from_histogram([0.0, 2.0, 1.0], [3, 4])


def test_from_histogram_refuses_bars_that_hold_no_patient() -> None:
    with pytest.raises(ValueError, match="at least one count"):
        from_histogram([0.0, 1.0, 2.0], [0, 0])


def test_per_patient_params_has_one_row_per_patient(paper_cohort: Cohort) -> None:
    assert paper_cohort.durations is not None
    params = per_patient_params(paper_cohort.durations)
    assert params["patient_id"].tolist() == sorted(paper_cohort.patients["patient_id"].tolist())


def test_per_patient_params_reads_a_barrier_ratio_for_most_patients(paper_cohort: Cohort) -> None:
    assert paper_cohort.durations is not None
    params = per_patient_params(paper_cohort.durations)
    assert params["barrier_ratio"].notna().mean() > 0.8


def test_per_patient_params_keeps_beta_below_the_fold(paper_cohort: Cohort) -> None:
    assert paper_cohort.durations is not None
    beta = per_patient_params(paper_cohort.durations)["beta"].dropna()
    assert not beta.empty
    assert beta.min() >= 0.0
    assert beta.max() < fold_beta(ALPHA)


def test_per_patient_params_counts_the_runs_of_each_state(paper_cohort: Cohort) -> None:
    assert paper_cohort.durations is not None
    params = per_patient_params(paper_cohort.durations)
    counts = paper_cohort.durations.groupby("patient_id", sort=True).size()
    totals = params["n_relapses"] + params["n_remissions"]
    assert totals.tolist() == counts.tolist()


def test_per_patient_params_reproduces_the_cohort_barrier_ratio() -> None:
    params = per_patient_params(paper_cohort_durations())
    expected = barrier_ratio_from_durations(TAU_HEALTH, TAU_RELAPSE)
    assert only_value(params, "barrier_ratio") == pytest.approx(expected, abs=1e-6)


def test_per_patient_params_reports_the_naive_means_the_paper_used() -> None:
    params = per_patient_params(paper_cohort_durations())
    assert only_value(params, "tau_health") == pytest.approx(TAU_HEALTH)
    assert only_value(params, "tau_relapse") == pytest.approx(TAU_RELAPSE)


def test_per_patient_params_leaves_an_undefined_ratio_missing() -> None:
    frame = pd.DataFrame(
        {
            "patient_id": pd.Series(["p0001", "p0001"], dtype=object),
            "run_index": pd.Series([0, 1], dtype=np.int64),
            "state": pd.Series([RELAPSE, HEALTH], dtype=np.int64),
            "duration_w": pd.Series([1, 100], dtype=np.int64),
            "censored": pd.Series([False, False], dtype=bool),
        }
    )
    params = per_patient_params(frame)
    assert math.isnan(only_value(params, "barrier_ratio"))
    assert math.isnan(only_value(params, "beta"))


def test_paper_patients_has_the_three_worked_examples() -> None:
    frame = paper_patients()
    assert len(frame) == 3


def test_paper_patients_recovers_the_printed_beta_of_patient_32() -> None:
    row = worked_example(32)
    assert only_value(row, "beta_from_ratio") == pytest.approx(
        PAPER.beta_patient_32.value, abs=0.005
    )


def test_paper_patients_recovers_the_printed_beta_of_patient_53() -> None:
    row = worked_example(53)
    assert only_value(row, "beta_from_ratio") == pytest.approx(
        PAPER.beta_patient_53.value, abs=0.005
    )


def test_paper_patients_shows_the_inconsistency_of_patient_23() -> None:
    row = worked_example(23)
    printed = DoubleWell(ALPHA, PAPER.beta_patient_23.value).barrier_ratio()
    assert only_value(row, "barrier_ratio_printed") == pytest.approx(PAPER.barrier_ratio_p23.value)
    assert only_value(row, "barrier_ratio_from_beta_printed") == pytest.approx(printed, abs=0.01)
    assert only_value(row, "barrier_ratio_from_beta_printed") < only_value(
        row, "barrier_ratio_printed"
    )


def test_bordi2013_spec_carries_the_cohort_of_the_paper() -> None:
    spec = bordi2013_spec()
    assert spec.n == PAPER.n_patients.value
    assert spec.engine == "renewal"
    assert spec.weekly is True


def test_bordi2013_spec_can_be_asked_for_the_stochastic_engine() -> None:
    assert bordi2013_spec("sde").engine == "sde"


# How far the stochastic spec of the paper must stay below the saddle-node fold,
# in units of beta. The calibration behind bordi2013_spec("sde") measures a beta
# of 0.359 at the default band fraction of 0.4 against a fold at 0.385, so a
# margin of 0.026, and the bound is a third of that. Asserting only that the
# calibration succeeds would say nothing, because the search box of
# msrelapse.model.calibrate already caps beta below the fold and raises rather
# than returning a value above it; a bound on the margin instead fails while the
# public path still works. Measured on the same spec at wider bands, the margin
# falls to 0.0117 at 0.44 and 0.0057 at 0.46, and at 0.5 there is no calibration
# at all, so this bound is met at the shipped band and crossed at 0.46.
SDE_FOLD_MARGIN = 0.008


def test_the_stochastic_spec_of_the_paper_stays_below_the_saddle_node_fold() -> None:
    # This is the calibration generate() performs for bordi2013_spec("sde"), the
    # matched targets with the rounding correction on top over the band the spec
    # carries. A change to the matched targets, to the rounding correction or to
    # the band default could spend the margin above and turn the documented
    # public path into a ValueError with nothing else noticing. Nothing is
    # integrated here, so the test costs milliseconds.
    spec = bordi2013_spec("sde")
    assert isinstance(spec.tau_health, float)
    assert isinstance(spec.tau_relapse, float)
    beta, _ = calibrate(
        *continuous_targets(spec.tau_health, spec.tau_relapse, spec.weekly),
        spec.alpha,
        method="mfpt",
        passage="band",
        band_fraction=spec.band_fraction,
    )
    assert fold_beta(ALPHA) - beta > SDE_FOLD_MARGIN


def test_cohort_spec_refuses_an_empty_cohort() -> None:
    with pytest.raises(ValueError, match="n must be at least 1"):
        CohortSpec(n=0, tau_health=100.0, tau_relapse=4.0, followup_weeks=100.0)


def test_cohort_spec_refuses_a_control_parameter_without_a_double_well() -> None:
    with pytest.raises(ValueError, match="alpha must be"):
        CohortSpec(n=1, tau_health=100.0, tau_relapse=4.0, followup_weeks=100.0, alpha=0.0)


def test_cohort_spec_refuses_a_step_that_blows_the_path_up() -> None:
    with pytest.raises(ValueError, match="dt must"):
        CohortSpec(n=1, tau_health=100.0, tau_relapse=4.0, followup_weeks=100.0, dt=MAX_DT + 0.1)


def test_cohort_spec_refuses_a_step_that_is_not_positive() -> None:
    with pytest.raises(ValueError, match="dt must"):
        CohortSpec(n=1, tau_health=100.0, tau_relapse=4.0, followup_weeks=100.0, dt=0.0)


def test_cohort_spec_refuses_a_band_outside_the_two_wells() -> None:
    with pytest.raises(ValueError, match="band_fraction"):
        CohortSpec(n=1, tau_health=100.0, tau_relapse=4.0, followup_weeks=100.0, band_fraction=1.0)


def test_cohort_spec_refuses_a_recorded_naive_remission_that_is_not_a_duration() -> None:
    with pytest.raises(ValueError, match="naive_tau_health must be"):
        CohortSpec(
            n=1,
            tau_health=100.0,
            tau_relapse=4.0,
            followup_weeks=100.0,
            naive_tau_health=0.0,
        )


def test_cohort_spec_refuses_a_recorded_naive_relapse_that_is_not_a_duration() -> None:
    with pytest.raises(ValueError, match="naive_tau_relapse must be"):
        CohortSpec(
            n=1,
            tau_health=100.0,
            tau_relapse=4.0,
            followup_weeks=100.0,
            naive_tau_relapse=math.inf,
        )


def test_cohort_spec_refuses_a_recorded_naive_remission_without_its_relapse() -> None:
    # The two are solved together and the provenance sentence names both, so half
    # a pair would travel with the cohort as "None weeks in health".
    with pytest.raises(ValueError, match="recorded together"):
        CohortSpec(
            n=1,
            tau_health=100.0,
            tau_relapse=4.0,
            followup_weeks=100.0,
            naive_tau_health=100.0,
        )


def test_cohort_spec_refuses_a_recorded_naive_relapse_without_its_remission() -> None:
    with pytest.raises(ValueError, match="recorded together"):
        CohortSpec(
            n=1,
            tau_health=100.0,
            tau_relapse=4.0,
            followup_weeks=100.0,
            naive_tau_relapse=4.3,
        )


def test_cohort_spec_refuses_an_unknown_engine() -> None:
    with pytest.raises(ValueError, match="engine must be one of"):
        CohortSpec(
            n=1,
            tau_health=100.0,
            tau_relapse=4.0,
            followup_weeks=100.0,
            engine="markov",  # type: ignore[arg-type]
        )


def test_cohort_spec_refuses_an_unknown_start_state() -> None:
    with pytest.raises(ValueError, match="start_state must be one of"):
        CohortSpec(
            n=1,
            tau_health=100.0,
            tau_relapse=4.0,
            followup_weeks=100.0,
            start_state="progressive",  # type: ignore[arg-type]
        )


def test_follow_up_is_rounded_up_to_whole_weeks() -> None:
    spec = CohortSpec(n=3, tau_health=100.0, tau_relapse=4.0, followup_weeks=120.4)
    cohort = generate(spec, rng=0)
    assert cohort.patients["followup_weeks"].tolist() == [121, 121, 121]


def test_a_very_short_follow_up_still_holds_two_weeks() -> None:
    spec = CohortSpec(n=2, tau_health=100.0, tau_relapse=4.0, followup_weeks=0.25)
    cohort = generate(spec, rng=0)
    assert cohort.patients["followup_weeks"].tolist() == [2, 2]


def test_generate_is_reproducible_from_its_seed() -> None:
    spec = CohortSpec(n=5, tau_health=100.0, tau_relapse=4.0, followup_weeks=300.0)
    first = generate(spec, rng=99).events
    second = generate(spec, rng=99).events
    pd.testing.assert_frame_equal(first, second)


def test_a_renewal_cohort_reports_no_potential_parameters() -> None:
    spec = CohortSpec(n=3, tau_health=100.0, tau_relapse=4.0, followup_weeks=300.0)
    patients = generate(spec, rng=1).patients
    assert patients["beta"].isna().all()
    assert patients["sigma"].isna().all()


def test_a_cohort_of_varied_patients_calibrates_each_target_pair() -> None:
    spec = CohortSpec(
        n=4,
        tau_health=empirical([80.0, 120.0]),
        tau_relapse=empirical([4.0, 6.0]),
        followup_weeks=300.0,
        engine="sde",
    )
    patients = generate(spec, rng=6).patients
    assert patients["beta"].notna().all()
    assert patients["sigma"].notna().all()
    assert patients["beta"].nunique() == 4


def test_docstring_examples_run() -> None:
    results = doctest.testmod(msrelapse.cohort)
    assert results.attempted > 0
    assert results.failed == 0
