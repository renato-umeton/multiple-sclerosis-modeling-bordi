"""The illustrative EDSS trajectory driven by the weekly relapse series."""

from __future__ import annotations

import dataclasses
import statistics
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
import pytest
from scipy import stats as scipy_stats

from msrelapse import cohort, edss
from msrelapse._params import PAPER
from msrelapse.stats import WEEKS_PER_YEAR

RELAPSE = PAPER.state_no_health.value
REMISSION = PAPER.state_health.value

# The published marginals the two laws were built to reproduce, and the
# tolerance each one is quoted to in the synthesis behind the module.
PEAK_THRESHOLDS = ((0.5, 0.84), (1.0, 0.61), (2.0, 0.17))
RESIDUAL_THRESHOLDS = ((0.5, 0.42), (1.0, 0.18))
LAW_TOLERANCE = 1e-9
THRESHOLD_TOLERANCE = 0.005
MEAN_TOLERANCE = 0.002

# The severity classes split the peak law in these shares.
MILD_SHARE = 0.39
SEVERE_SHARE = 0.61

# How often a residual comes out above the peak that was drawn with it, so that
# the peak is raised to it and the relapse steps up with no transient at all,
# and what that raising does to the mean peak a record actually draws. Both
# figures are quoted in the docstring of episode_draws and are summed exactly
# over the two published laws here.
RAISED_PEAK_SHARE = 0.09147
RAISED_MEAN_PEAK = 1.116295

# The record every kernel test runs on: one relapse from week 0 and a long
# remission behind it, so that the whole decay is inside the record.
KERNEL_RELAPSE_WEEKS = 4
KERNEL_REMISSION_WEEKS = 200

# How many seeds the two distribution tests draw over. Both are cheap because
# they draw the episodes alone and never build a trajectory.
FIRST_JUMP_SEEDS = 500
PEAK_SAMPLE_SEEDS = 2000
# How many seeds the tests that read a sample of episode draws run over.
DRAW_SAMPLE_SEEDS = 200
# Two disjoint runs of seeds, so that the two samples the test compares are
# drawn independently rather than being the same numbers twice.
SECOND_SAMPLE_START = 10_000

# The cohort the ten year anchor is read off: the paper spec at its own two
# mean durations, resized to 300 patients, with its follow up replaced rather
# than filtered. The paper spec draws the length of each record from the
# digitised Figure 3 histogram, under which only 76 of 300 patients reach the
# ten year week at all and those 76 hold 356 episodes up to it; fixing every
# record at 560 weeks keeps all 300 patients and 1491 episodes, four times as
# many, which is what makes both anchors readable. The anchor is a statement
# about the disability model rather than about that histogram, which
# tests/test_cohort.py covers.
#
# Do not restore the filter thinking it reads the anchor off more of the paper.
# It reads it off the quarter of the cohort that survives the cut, a mean over
# 76 records rather than 300, twice as variable (a standard error of 0.20
# against 0.10) and about 0.11 higher: over ten seeds the filtered mean
# averages 3.49 against 3.38 here, and it crosses the 3.5 bound of the
# tolerance below at five of those ten seeds, so the filtered form is not a
# test that holds. The relapse exposure is not what moves it: the episodes per
# patient up to the ten year week are about the same either way, 4.7 filtered
# against 5.0 here.
ANCHOR_PATIENTS = 300
ANCHOR_FOLLOWUP_WEEKS = 560.0
ANCHOR_SEED = 11
ANCHOR_DRAW_SEED = 2013
# The published EDSS at ten years from onset, and how far the cohort mean may
# sit from it. Pittock 2004, quoted in the evidence record of the module.
ANCHOR_EDSS = 3.0
ANCHOR_TOLERANCE = 0.5
RESIDUAL_ANCHOR_TOLERANCE = 0.05


def states_of(*runs: tuple[int, int]) -> npt.NDArray[np.int64]:
    """Return a weekly state array from (state, number of weeks) pairs."""
    codes = [state for state, _weeks in runs]
    lengths = [weeks for _state, weeks in runs]
    return np.repeat(codes, lengths).astype(np.int64)


def weekly_frame(states: npt.NDArray[np.int64], patient_id: str = "p1") -> pd.DataFrame:
    """Return a one patient frame in the weekly schema holding `states`."""
    return pd.DataFrame(
        {
            "patient_id": [patient_id] * states.size,
            "week": np.arange(states.size, dtype=np.int64),
            "state": states.astype(np.int64),
        }
    )


def forced(peak: float, residual: float, **overrides: Any) -> edss.EDSSSpec:
    """Return a spec whose two laws are degenerate, so every draw is known."""
    return dataclasses.replace(
        edss.EDSSSpec(),
        peak_law={peak: 1.0},
        residual_law_mild={residual: 1.0},
        residual_law_severe={residual: 1.0},
        **overrides,
    )


def kernel_record() -> npt.NDArray[np.int64]:
    """Return the one relapse record the kernel tests read."""
    return states_of((RELAPSE, KERNEL_RELAPSE_WEEKS), (REMISSION, KERNEL_REMISSION_WEEKS))


@pytest.fixture(scope="module")
def anchor_cohort() -> pd.DataFrame:
    spec = dataclasses.replace(
        cohort.bordi2013_spec(), n=ANCHOR_PATIENTS, followup_weeks=ANCHOR_FOLLOWUP_WEEKS
    )
    weekly = cohort.generate(spec, rng=ANCHOR_SEED).weekly
    assert weekly is not None
    return weekly


def test_every_law_of_the_default_spec_sums_to_one() -> None:
    spec = edss.EDSSSpec()

    totals = [
        sum(spec.peak_law.values()),
        sum(spec.residual_law_mild.values()),
        sum(spec.residual_law_severe.values()),
    ]

    assert totals == pytest.approx([1.0, 1.0, 1.0], abs=LAW_TOLERANCE)


@pytest.mark.parametrize(("threshold", "published"), PEAK_THRESHOLDS)
def test_the_peak_law_reproduces_the_published_nadir_thresholds(
    threshold: float, published: float
) -> None:
    spec = edss.EDSSSpec()

    mass = sum(p for value, p in spec.peak_law.items() if value >= threshold)

    assert mass == pytest.approx(published, abs=LAW_TOLERANCE)


def test_the_mean_peak_is_the_published_nadir_increase() -> None:
    # The synthesis quotes the mean of this law as 1.05, which is the exact
    # 1.0525 rounded to the two figures the table carries.
    mean = edss.expected_peak(edss.EDSSSpec())

    assert mean == pytest.approx(1.0525, abs=LAW_TOLERANCE)
    assert round(mean, 2) == 1.05


def test_the_severity_classes_split_the_peak_law_as_published() -> None:
    spec = edss.EDSSSpec()

    mild = sum(p for value, p in spec.peak_law.items() if value < edss.SEVERE_PEAK)

    assert mild == pytest.approx(MILD_SHARE, abs=LAW_TOLERANCE)
    assert 1.0 - mild == pytest.approx(SEVERE_SHARE, abs=LAW_TOLERANCE)


def mixed_residual_mass(spec: edss.EDSSSpec, threshold: float) -> float:
    """Return the marginal probability of a residual of at least `threshold`."""
    mild = sum(p for value, p in spec.residual_law_mild.items() if value >= threshold)
    severe = sum(p for value, p in spec.residual_law_severe.items() if value >= threshold)
    return MILD_SHARE * mild + SEVERE_SHARE * severe


@pytest.mark.parametrize(("threshold", "published"), RESIDUAL_THRESHOLDS)
def test_the_mixed_residual_law_reproduces_the_pooled_incomplete_recovery(
    threshold: float, published: float
) -> None:
    mass = mixed_residual_mass(edss.EDSSSpec(), threshold)

    assert mass == pytest.approx(published, abs=THRESHOLD_TOLERANCE)


def test_the_mean_residual_is_the_published_one() -> None:
    mean = edss.expected_residual(edss.EDSSSpec())

    assert mean == pytest.approx(0.256, abs=MEAN_TOLERANCE)


def test_the_median_residual_is_zero() -> None:
    spec = edss.EDSSSpec()

    at_or_above_zero = mixed_residual_mass(spec, 0.0)
    above_zero = mixed_residual_mass(spec, 0.5)

    # Fewer than half of relapses leave anything behind and more than half end
    # at or below the pre-relapse score, so the median residual is zero, which
    # is what the pooled placebo arms report.
    assert above_zero < 0.5
    assert at_or_above_zero >= 0.5


def test_every_evidence_record_carries_a_doi() -> None:
    assert [record.parameter for record in edss.EVIDENCE if not record.doi.strip()] == []


def test_every_evidence_record_is_filled_in() -> None:
    empty = [
        record.parameter
        for record in edss.EVIDENCE
        if not (record.parameter.strip() and record.value.strip() and record.source.strip())
    ]

    assert empty == []


def test_the_evidence_names_no_parameter_twice() -> None:
    names = [record.parameter for record in edss.EVIDENCE]

    assert sorted(names) == sorted(set(names))


def test_the_evidence_covers_what_the_model_rests_on() -> None:
    written = " ".join(record.parameter.lower() for record in edss.EVIDENCE)

    for subject in ("baseline", "nadir", "residual", "recovery", "severity", "10 years", "pira"):
        assert subject in written, f"no evidence record speaks to {subject!r}"


def test_relapse_episodes_are_the_maximal_runs_of_the_relapse_code() -> None:
    states = states_of((REMISSION, 2), (RELAPSE, 3), (REMISSION, 4), (RELAPSE, 1))

    assert edss.relapse_episodes(states) == [(2, 4), (9, 9)]


def test_a_record_without_a_relapse_holds_no_episode() -> None:
    assert edss.relapse_episodes(states_of((REMISSION, 10))) == []


def test_a_relapse_that_runs_to_the_end_of_the_record_is_an_episode() -> None:
    states = states_of((REMISSION, 2), (RELAPSE, 3))

    assert edss.relapse_episodes(states) == [(2, 4)]


def test_the_trace_rises_to_the_peak_within_the_first_week() -> None:
    spec = forced(peak=1.0, residual=0.5)

    trace = edss.edss_trajectory(kernel_record(), spec, rng=0)

    assert float(trace["edss"].iloc[0]) == pytest.approx(spec.baseline + 1.0)


def test_the_excess_above_the_residual_decays_with_the_recovery_constant() -> None:
    peak = 1.0
    residual = 0.5
    spec = forced(peak=peak, residual=residual)

    trace = edss.edss_trajectory(kernel_record(), spec, rng=0)

    weeks = trace["week"].to_numpy().astype(np.float64)
    kernel = trace["edss"].to_numpy() - spec.baseline
    # Past the nadir the excess above the residual falls by a factor of e every
    # tau weeks, counted from the nadir week and not from the onset week. The
    # two readings differ by a factor exp(1 / tau), 5 percent at this tau,
    # which is why the kernel is asserted exactly rather than within a margin
    # wide enough to cover a decay started a week early.
    after_nadir = weeks >= float(spec.time_to_nadir_weeks)
    decay = residual + (peak - residual) * np.exp(
        -(weeks[after_nadir] - spec.time_to_nadir_weeks) / spec.recovery_tau_weeks
    )
    assert kernel[after_nadir] == pytest.approx(decay, rel=1e-9)


def test_the_trace_never_rises_again_after_the_nadir() -> None:
    spec = forced(peak=1.0, residual=0.5)

    trace = edss.edss_trajectory(kernel_record(), spec, rng=0)

    steps = np.diff(trace["edss"].to_numpy()[1:])
    assert bool(np.all(steps <= 1e-12))


def test_a_relapse_that_leaves_no_residual_brings_the_trace_back_to_the_baseline() -> None:
    spec = forced(peak=1.0, residual=0.0)

    trace = edss.edss_trajectory(kernel_record(), spec, rng=0)

    assert float(trace["edss"].iloc[60]) == pytest.approx(spec.baseline, abs=0.01)


def test_progression_alone_raises_the_trace_by_a_point_over_ten_years() -> None:
    spec = dataclasses.replace(edss.EDSSSpec(), pira_per_year=0.1)
    weeks = round(10.0 * WEEKS_PER_YEAR) + 1

    trace = edss.edss_trajectory(states_of((REMISSION, weeks)), spec, rng=0)

    rise = float(trace["edss"].iloc[-1]) - float(trace["edss"].iloc[0])
    # The tolerance is a thousandth of a point rather than a hundredth so that
    # it separates the two year conventions: over these 522 weeks a rate of 0.1
    # per year raises the trace by 1.0004 at the package year of 365.25 over 7
    # weeks and by 1.0038 at a flat 52 week year, and only the first passes.
    assert rise == pytest.approx(1.0, abs=1e-3)


def test_the_first_jump_stays_on_the_scale_and_is_a_point_on_median() -> None:
    spec = edss.EDSSSpec()
    # A single relapse of 27 weeks, the longest first flare seen while choosing
    # the animation seed; the committed animation uses seed 6.
    states = states_of((RELAPSE, 27), (REMISSION, 200))

    tops = [
        float(edss.edss_trajectory(states, spec, rng=seed)["edss"].max()) - spec.baseline
        for seed in range(FIRST_JUMP_SEEDS)
    ]

    assert max(tops) <= max(spec.peak_law)
    assert statistics.median(tops) == pytest.approx(1.0)


def peaks_over_seeds(states: npt.NDArray[np.int64], seeds: range) -> list[float]:
    """Return the peak drawn for the one episode of `states`, one seed at a time."""
    spec = edss.EDSSSpec()
    return [float(edss.episode_draws(states, spec, rng=seed)["peak"].iloc[0]) for seed in seeds]


def test_the_length_of_a_relapse_does_not_change_the_peak_it_draws() -> None:
    short = peaks_over_seeds(states_of((RELAPSE, 2), (REMISSION, 200)), range(PEAK_SAMPLE_SEEDS))
    long_run = peaks_over_seeds(
        states_of((RELAPSE, 27), (REMISSION, 200)),
        range(SECOND_SAMPLE_START, SECOND_SAMPLE_START + PEAK_SAMPLE_SEEDS),
    )

    comparison = scipy_stats.ks_2samp(short, long_run)

    assert comparison.pvalue > 0.05


@pytest.mark.parametrize(("raw", "displayed"), [(2.25, 2.5), (2.24, 2.0)])
def test_the_displayed_score_rounds_half_a_point_up(raw: float, displayed: float) -> None:
    # A relapse that peaks at nothing and leaves nothing holds the trace on the
    # baseline, so the baseline alone decides what is displayed.
    spec = forced(peak=0.0, residual=0.0, baseline=raw)

    trace = edss.edss_trajectory(states_of((RELAPSE, 1), (REMISSION, 1)), spec, rng=0)

    assert float(trace["edss"].iloc[0]) == pytest.approx(raw)
    assert float(trace["edss_display"].iloc[0]) == pytest.approx(displayed)


def test_the_trace_is_clipped_to_the_top_of_the_scale() -> None:
    spec = forced(peak=3.0, residual=3.0, baseline=9.5)

    trace = edss.edss_trajectory(states_of((RELAPSE, 2), (REMISSION, 20)), spec, rng=0)

    assert float(trace["edss"].max()) == pytest.approx(spec.scale_max)
    assert float(trace["edss_display"].max()) == pytest.approx(spec.scale_max)


def test_the_trace_is_clipped_to_the_bottom_of_the_scale() -> None:
    spec = forced(peak=0.0, residual=-1.0, baseline=0.5)

    trace = edss.edss_trajectory(states_of((RELAPSE, 2), (REMISSION, 200)), spec, rng=0)

    assert float(trace["edss"].min()) == pytest.approx(spec.scale_min)


def test_the_same_seed_gives_the_same_trace() -> None:
    states = states_of((RELAPSE, 3), (REMISSION, 60), (RELAPSE, 2), (REMISSION, 60))

    first = edss.edss_trajectory(states, rng=7)
    second = edss.edss_trajectory(states, rng=7)

    pd.testing.assert_frame_equal(first, second)


def test_different_seeds_give_different_traces() -> None:
    states = states_of((RELAPSE, 3), (REMISSION, 60), (RELAPSE, 2), (REMISSION, 60))

    first = edss.edss_trajectory(states, rng=7)
    second = edss.edss_trajectory(states, rng=8)

    assert not first["edss"].equals(second["edss"])


def test_the_draws_and_the_trace_agree_under_one_seed() -> None:
    states = states_of((RELAPSE, 3), (REMISSION, 60), (RELAPSE, 2), (REMISSION, 60))

    draws = edss.episode_draws(states, rng=7)
    trace = edss.edss_trajectory(states, rng=7)

    # The nadir of the first episode is its peak, and the trace settles on the
    # residual of the last episode once the record has run long enough.
    first = draws.iloc[0]
    assert float(trace["edss"].iloc[0]) == pytest.approx(
        edss.EDSSSpec().baseline + float(first["peak"])
    )
    assert list(draws["start"]) == [0, 63]
    assert list(draws["end"]) == [2, 64]
    assert list(draws["duration_w"]) == [3, 2]


def test_the_trajectory_carries_the_documented_columns_and_dtypes() -> None:
    trace = edss.edss_trajectory(states_of((RELAPSE, 2), (REMISSION, 5)), rng=1)

    assert list(trace.columns) == ["week", "edss", "edss_display", "episode"]
    assert [trace[column].dtype.name for column in trace.columns] == [
        "int64",
        "float64",
        "float64",
        "int64",
    ]


def test_the_episode_column_names_the_episode_in_progress() -> None:
    states = states_of((REMISSION, 2), (RELAPSE, 3), (REMISSION, 4), (RELAPSE, 1))

    trace = edss.edss_trajectory(states, rng=1)

    assert list(trace["episode"]) == [-1, -1, 0, 0, 0, -1, -1, -1, -1, 1]


def test_the_draws_carry_one_row_per_episode() -> None:
    states = states_of((REMISSION, 2), (RELAPSE, 3), (REMISSION, 4), (RELAPSE, 1))

    draws = edss.episode_draws(states, rng=1)

    assert list(draws.columns) == [
        "episode",
        "start",
        "end",
        "duration_w",
        "peak",
        "residual",
        "severity_class",
    ]
    assert list(draws["episode"]) == [0, 1]
    assert list(draws["duration_w"]) == [3, 1]


def test_a_record_without_a_relapse_draws_nothing_and_holds_the_baseline() -> None:
    states = states_of((REMISSION, 30))

    draws = edss.episode_draws(states, rng=1)
    trace = edss.edss_trajectory(states, rng=1)

    assert draws.empty
    assert list(draws.columns) == [
        "episode",
        "start",
        "end",
        "duration_w",
        "peak",
        "residual",
        "severity_class",
    ]
    assert trace["edss"].to_numpy() == pytest.approx(edss.EDSSSpec().baseline)


def residual_law_of(spec: edss.EDSSSpec, peak: float) -> dict[float, float]:
    """Return the residual law the severity class of `peak` draws from."""
    law = spec.residual_law_severe if peak >= edss.SEVERE_PEAK else spec.residual_law_mild
    return dict(law)


def raised_peak_share(spec: edss.EDSSSpec) -> float:
    """Return how often a residual comes out above the peak drawn with it."""
    return sum(
        peak_p * sum(p for value, p in residual_law_of(spec, peak).items() if value > peak)
        for peak, peak_p in spec.peak_law.items()
    )


def realised_mean_peak(spec: edss.EDSSSpec) -> float:
    """Return the mean peak a record draws, after each peak is raised to its residual."""
    return sum(
        peak_p * sum(p * max(peak, value) for value, p in residual_law_of(spec, peak).items())
        for peak, peak_p in spec.peak_law.items()
    )


def test_raising_a_peak_to_its_residual_hits_the_documented_share_of_relapses() -> None:
    # The docstring of episode_draws quotes both of these figures, because a
    # relapse whose residual reaches its peak steps up with no transient at all
    # and a reader is owed how often that happens. Summing them exactly over
    # the two laws here is what stops the sentence going stale if a law is ever
    # retuned.
    spec = edss.EDSSSpec()

    assert raised_peak_share(spec) == pytest.approx(RAISED_PEAK_SHARE, abs=LAW_TOLERANCE)
    assert realised_mean_peak(spec) == pytest.approx(RAISED_MEAN_PEAK, abs=LAW_TOLERANCE)
    assert realised_mean_peak(spec) > edss.expected_peak(spec)


def test_the_peak_is_never_below_the_residual_it_settles_on() -> None:
    states = states_of((RELAPSE, 2), (REMISSION, 30))

    draws = pd.concat(
        [edss.episode_draws(states, rng=seed) for seed in range(DRAW_SAMPLE_SEEDS)],
        ignore_index=True,
    )

    assert bool(np.all(draws["peak"].to_numpy() >= draws["residual"].to_numpy()))


def test_the_severity_class_follows_the_peak_that_was_drawn() -> None:
    states = states_of((RELAPSE, 2), (REMISSION, 30))

    draws = pd.concat(
        [edss.episode_draws(states, rng=seed) for seed in range(DRAW_SAMPLE_SEEDS)],
        ignore_index=True,
    )

    classes = set(draws["severity_class"])
    assert classes == {"mild", "severe"}
    # Under the published laws a severe class means a drawn peak of at least
    # SEVERE_PEAK, and the peak a row carries is that draw raised to its own
    # residual, so it can only be higher. The reverse does not hold here: a
    # mild peak raised by a larger residual reaches SEVERE_PEAK too, which is
    # why only this direction is asserted. The test below pins both directions,
    # over laws under which no peak is ever raised.
    severe = draws[draws["severity_class"] == "severe"]
    assert float(severe["peak"].min()) >= edss.SEVERE_PEAK


def test_the_severity_class_is_the_peak_threshold_when_no_residual_raises_the_peak() -> None:
    # Every residual is a point below the pre-relapse score here, so no peak is
    # ever raised to its residual and the peak a row carries is the draw its
    # class was read off. That is what lets the rule be asserted in both
    # directions, so that a class read off the wrong quantity, or read off the
    # right one at the wrong side of the threshold, fails the test.
    spec = dataclasses.replace(
        edss.EDSSSpec(), residual_law_mild={-1.0: 1.0}, residual_law_severe={-1.0: 1.0}
    )
    states = states_of((RELAPSE, 2), (REMISSION, 30))

    draws = pd.concat(
        [edss.episode_draws(states, spec, rng=seed) for seed in range(DRAW_SAMPLE_SEEDS)],
        ignore_index=True,
    )

    assert set(draws["severity_class"]) == {"mild", "severe"}
    called_severe = draws["severity_class"] == "severe"
    peak_is_severe = draws["peak"] >= edss.SEVERE_PEAK
    assert list(called_severe) == list(peak_is_severe)


def test_a_weekly_frame_and_its_states_give_the_same_trace() -> None:
    states = states_of((RELAPSE, 3), (REMISSION, 40))

    from_frame = edss.edss_trajectory(weekly_frame(states), rng=3)
    from_states = edss.edss_trajectory(states, rng=3)

    pd.testing.assert_frame_equal(from_frame, from_states)


def test_a_frame_holding_two_patients_is_refused() -> None:
    first = weekly_frame(states_of((RELAPSE, 2), (REMISSION, 3)), "p1")
    second = weekly_frame(states_of((RELAPSE, 2), (REMISSION, 3)), "p2")

    with pytest.raises(ValueError, match="one patient"):
        edss.edss_trajectory(pd.concat([first, second], ignore_index=True))


def test_a_frame_without_the_state_column_is_refused() -> None:
    frame = weekly_frame(states_of((RELAPSE, 2), (REMISSION, 3))).drop(columns=["state"])

    with pytest.raises(ValueError, match="state"):
        edss.edss_trajectory(frame)


def test_a_record_of_no_weeks_is_refused() -> None:
    with pytest.raises(ValueError, match="at least one week"):
        edss.edss_trajectory(np.empty(0, dtype=np.int64))


def test_a_states_array_of_more_than_one_dimension_is_refused() -> None:
    with pytest.raises(ValueError, match="one dimensional"):
        edss.edss_trajectory(np.array([[RELAPSE, REMISSION]], dtype=np.int64))


def test_a_states_array_holding_an_unknown_code_is_refused() -> None:
    with pytest.raises(ValueError, match="state"):
        edss.edss_trajectory(np.array([RELAPSE, 0], dtype=np.int64))


@pytest.mark.parametrize("absent", [np.nan, np.inf, -np.inf])
def test_a_states_array_holding_a_value_that_is_not_a_number_is_refused(absent: float) -> None:
    # A float record is checked for finiteness before it is cast, because
    # casting a value that is not a number to an integer warns rather than
    # raises, and the warning says nothing about which week is at fault.
    with pytest.raises(ValueError, match="finite"):
        edss.edss_trajectory(np.array([float(RELAPSE), absent], dtype=np.float64))


def test_a_law_that_does_not_sum_to_one_is_refused() -> None:
    with pytest.raises(ValueError, match="sum to 1"):
        dataclasses.replace(edss.EDSSSpec(), peak_law={0.0: 0.5, 1.0: 0.4})


def test_a_law_off_the_half_point_grid_is_refused() -> None:
    with pytest.raises(ValueError, match="half point"):
        dataclasses.replace(edss.EDSSSpec(), peak_law={0.3: 1.0})


def test_a_negative_probability_is_refused() -> None:
    with pytest.raises(ValueError, match="negative"):
        dataclasses.replace(edss.EDSSSpec(), peak_law={0.0: 1.5, 1.0: -0.5})


def test_an_empty_law_is_refused() -> None:
    with pytest.raises(ValueError, match="at least one"):
        dataclasses.replace(edss.EDSSSpec(), peak_law={})


@pytest.mark.parametrize(
    ("name", "value"),
    [("recovery_tau_weeks", 0.0), ("baseline", -1.0), ("time_to_nadir_weeks", 0)],
)
def test_a_parameter_outside_its_range_is_refused(name: str, value: float) -> None:
    changed: dict[str, Any] = {name: value}

    with pytest.raises(ValueError, match="must be"):
        dataclasses.replace(edss.EDSSSpec(), **changed)


def test_a_scale_that_does_not_run_upwards_is_refused() -> None:
    with pytest.raises(ValueError, match="scale_max"):
        dataclasses.replace(edss.EDSSSpec(), scale_min=10.0, scale_max=0.0)


def test_summarise_reports_the_record_it_is_given() -> None:
    spec = forced(peak=1.0, residual=0.5)
    states = states_of((RELAPSE, 2), (REMISSION, 60))

    trace = edss.edss_trajectory(states, spec, rng=0)
    summary = edss.summarise(trace, edss.episode_draws(states, spec, rng=0))

    assert summary["max_edss"] == pytest.approx(spec.baseline + 1.0)
    assert summary["final_edss"] == pytest.approx(float(trace["edss"].iloc[-1]))
    assert summary["episodes"] == pytest.approx(1.0)
    assert summary["episodes_with_a_residual"] == pytest.approx(1.0)
    assert "edss_at_ten_years" not in summary


def test_summarise_reads_a_trace_given_on_its_own() -> None:
    spec = forced(peak=1.0, residual=0.5)
    states = states_of((RELAPSE, 2), (REMISSION, 60))

    summary = edss.summarise(edss.edss_trajectory(states, spec, rng=0))

    assert summary["max_edss"] == pytest.approx(spec.baseline + 1.0)
    assert summary["episodes"] == pytest.approx(1.0)
    # What each episode left behind is not in the trace, so the count of the
    # episodes that left a residual is reported only when the draws are given.
    assert "episodes_with_a_residual" not in summary


def test_summarise_refuses_draws_of_another_record() -> None:
    spec = forced(peak=1.0, residual=0.5)
    two_relapses = states_of((RELAPSE, 2), (REMISSION, 60), (RELAPSE, 2), (REMISSION, 60))
    one_relapse = states_of((RELAPSE, 2), (REMISSION, 60))

    trace = edss.edss_trajectory(two_relapses, spec, rng=0)
    draws = edss.episode_draws(one_relapse, spec, rng=0)

    with pytest.raises(ValueError, match="one record"):
        edss.summarise(trace, draws)


def test_summarise_reports_the_ten_year_level_of_a_long_enough_record() -> None:
    spec = forced(peak=1.0, residual=0.0)
    weeks = round(10.0 * WEEKS_PER_YEAR) + 1
    states = states_of((RELAPSE, 2), (REMISSION, weeks - 2))

    trace = edss.edss_trajectory(states, spec, rng=0)
    summary = edss.summarise(trace, edss.episode_draws(states, spec, rng=0))

    assert summary["edss_at_ten_years"] == pytest.approx(spec.baseline, abs=0.01)
    assert summary["episodes_with_a_residual"] == pytest.approx(0.0)


def test_ten_years_is_the_week_the_package_year_puts_it_at() -> None:
    # The module reads ten years at round(10 * WEEKS_PER_YEAR), which is week
    # 522 and not the 520 of ten 52 week years. A record whose last week is 521
    # is therefore one week short of the anchor and a record whose last week is
    # 522 reaches it, which pins the week in one pair of assertions: at 520 both
    # records would carry the level, and at 523 neither would.
    ten_years = round(10.0 * WEEKS_PER_YEAR)

    short = edss.summarise(edss.edss_trajectory(states_of((REMISSION, ten_years)), rng=0))
    long_enough = edss.summarise(edss.edss_trajectory(states_of((REMISSION, ten_years + 1)), rng=0))

    assert "edss_at_ten_years" not in short
    assert "edss_at_ten_years" in long_enough


def test_a_renewal_cohort_reaches_about_edss_three_at_ten_years(
    anchor_cohort: pd.DataFrame,
) -> None:
    spec = edss.EDSSSpec()
    generator = np.random.default_rng(ANCHOR_DRAW_SEED)

    # The level is read through summarise, so that ten years means here what it
    # means in the module: the round(10 * WEEKS_PER_YEAR) = 522 of the year the
    # whole package counts with, rather than the 520 of the ten 52 week years
    # the published anchor is quoted in. The two readings differ by 0.008 of a
    # point across this cohort.
    levels = [
        edss.summarise(edss.edss_trajectory(block["state"].to_numpy(), spec, generator))[
            "edss_at_ten_years"
        ]
        for _patient, block in anchor_cohort.groupby("patient_id", sort=True)
    ]

    assert len(levels) == ANCHOR_PATIENTS
    assert float(np.mean(levels)) == pytest.approx(ANCHOR_EDSS, abs=ANCHOR_TOLERANCE)
    # The published anchor is a level a cohort sits at rather than an average
    # over a long tail, and the median of the cohort lands on it.
    assert float(np.median(levels)) == pytest.approx(ANCHOR_EDSS, abs=ANCHOR_TOLERANCE)


def test_the_mean_residual_of_a_renewal_cohort_matches_the_law(
    anchor_cohort: pd.DataFrame,
) -> None:
    spec = edss.EDSSSpec()
    generator = np.random.default_rng(ANCHOR_DRAW_SEED)

    residuals = pd.concat(
        [
            edss.episode_draws(block["state"].to_numpy(), spec, generator)
            for _patient, block in anchor_cohort.groupby("patient_id", sort=True)
        ],
        ignore_index=True,
    )["residual"]

    assert len(residuals) > ANCHOR_PATIENTS
    assert float(residuals.mean()) == pytest.approx(
        edss.expected_residual(spec), abs=RESIDUAL_ANCHOR_TOLERANCE
    )


def test_the_module_says_it_is_not_the_article() -> None:
    assert edss.__doc__ is not None
    text = edss.__doc__.lower()

    assert "illustrative" in text
    assert "not part of the" in text or "not in the article" in text
