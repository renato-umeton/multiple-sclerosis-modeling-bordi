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

from msrelapse import cohort, datasets, edss
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

# What each severity class leaves behind at least half a point of, and how
# often a relapse of either class retains at least two points.
MILD_INCOMPLETE = 0.25
SEVERE_INCOMPLETE = 0.53
SEVERE_TAIL = 0.034

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

# The sample the tests of the accumulation rule read: one trace per seed, over
# the shipped records, taken in turn. A record of the twin runs from 40 to 1311
# weeks and holds a handful of flares, so the sample carries about seven
# hundred of them.
MONOTONE_SEEDS = 200
# How long after the last week of a flare its recovery is read, how many quiet
# weeks in front of a flare make the level before it a settled one, how few
# flares of the sample may clear that rule before the test stops meaning
# anything, and how far below its opening level a flare may end. The margin
# covers the last of the flare's own decay: over 150 quiet weeks the deepest
# transient the laws can draw falls to a millionth of a point. Of the 725
# flares the sample holds, 95 open after that many quiet weeks and are read
# before the next flare opens.
RECOVERY_WEEKS = 40
QUIET_WEEKS = 150
QUIET_FLARES = 50
SETTLED_MARGIN = 1e-5
# How few flares of the sample may be readable at all before the all flares
# form of the same rule stops meaning anything. A flare that opens in week 0
# has no week in front of it to read, which leaves 525 of the 725.
READABLE_FLARES = 500
# How many residuals the sign of the residual law is read over, as a record of
# many flares drawn under many seeds.
NEGATIVE_DRAW_EPISODES = 50
NEGATIVE_DRAW_SEEDS = 100
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
# about 80 records rather than 300, twice as variable and about 0.12 higher:
# over ten cohort seeds the filtered mean averages 3.98 with a spread of 0.086
# against 3.86 with a spread of 0.045 here, so the filtered form is the noisier
# test of the two. The relapse exposure is not what moves it: the episodes per
# patient up to the ten year week are about the same either way, 4.7 filtered
# against 5.0 here.
ANCHOR_PATIENTS = 300
ANCHOR_FOLLOWUP_WEEKS = 560.0
ANCHOR_SEED = 11
ANCHOR_DRAW_SEED = 2013
# The band the cohort level at ten years has to fall in. The published anchors
# sit near 3: Pittock 2004 puts a population based cohort at about that level
# ten years from onset, and the two natural history cohorts put the median time
# to EDSS 3 at ten years, so half a cohort is at or above 3 by then. The band is
# wide on purpose, because the spread at a fixed duration is enormous, because
# this model is a teaching illustration rather than a prognosis, and because a
# residual law that carries no improvements accrues faster than the published
# mean change per relapse: this cohort now averages 3.92 at ten years with a
# median of 3.50, which the disability page records as an overshoot.
ANCHOR_LOW = 2.5
ANCHOR_HIGH = 4.2
# The band alone is 1.7 points wide, so it would absorb a drift of a quarter of
# a point rather than report it. The measured mean of this one cohort, 3.9195
# under the seeds above, is therefore pinned beside the band: the band says the
# model still sits where the published anchors are read, and the pin says the
# laws behind it have not moved.
ANCHOR_MEAN = 3.92
ANCHOR_MEAN_TOLERANCE = 0.05
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
def twin_records() -> list[npt.NDArray[np.int64]]:
    """Return the weekly states of each shipped synthetic record, in order."""
    weekly = datasets.load_synthetic_bordi2013()
    return [
        block["state"].to_numpy() for _patient, block in weekly.groupby("patient_id", sort=True)
    ]


@pytest.fixture(scope="module")
def twin_traces(
    twin_records: list[npt.NDArray[np.int64]],
) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
    """Return one trace and its draws per seed, over the shipped records in turn."""
    spec = edss.EDSSSpec()
    sample = []
    for seed in range(MONOTONE_SEEDS):
        states = twin_records[seed % len(twin_records)]
        sample.append(
            (
                edss.edss_trajectory(states, spec, rng=seed),
                edss.episode_draws(states, spec, rng=seed),
            )
        )
    return sample


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


def test_each_severity_class_leaves_the_residual_its_class_was_built_to_leave() -> None:
    spec = edss.EDSSSpec()

    mild = sum(p for value, p in spec.residual_law_mild.items() if value >= 0.5)
    severe = sum(p for value, p in spec.residual_law_severe.items() if value >= 0.5)

    assert (mild, severe) == pytest.approx((MILD_INCOMPLETE, SEVERE_INCOMPLETE), abs=LAW_TOLERANCE)


def test_the_mixed_residual_law_keeps_the_published_severe_tail() -> None:
    mass = mixed_residual_mass(edss.EDSSSpec(), 2.0)

    assert mass == pytest.approx(SEVERE_TAIL, abs=THRESHOLD_TOLERANCE)


@pytest.mark.parametrize("name", ["residual_law_mild", "residual_law_severe", "peak_law"])
def test_no_law_of_the_default_spec_carries_a_value_below_zero(name: str) -> None:
    # Accumulated disability does not decrease, so no law of the model may put
    # any mass below zero: a negative residual would leave a patient
    # permanently better off after a relapse than before it.
    law: dict[float, float] = dict(getattr(edss.EDSSSpec(), name))

    assert [value for value in law if value < 0.0] == []


def test_the_mean_residual_is_the_one_the_two_laws_imply() -> None:
    spec = edss.EDSSSpec()

    mean = edss.expected_residual(spec)

    # Computed here from the laws themselves rather than quoted, so that the
    # figure the documentation carries, 0.361, cannot drift away from the laws
    # it is read off. It sits above the published mean change per relapse of
    # about 0.25, which is net of improvements that are not disability.
    mild = sum(value * p for value, p in spec.residual_law_mild.items())
    severe = sum(value * p for value, p in spec.residual_law_severe.items())
    assert mean == pytest.approx(MILD_SHARE * mild + SEVERE_SHARE * severe, abs=LAW_TOLERANCE)
    assert round(mean, 3) == 0.361


def test_the_median_residual_is_zero() -> None:
    spec = edss.EDSSSpec()

    at_or_above_zero = mixed_residual_mass(spec, 0.0)
    above_zero = mixed_residual_mass(spec, 0.5)

    # Fewer than half of relapses leave anything behind and the rest end back
    # at the pre-relapse score, so the median residual is zero, which is what
    # the pooled placebo arms report.
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


def test_the_floor_is_the_baseline_plus_the_residual_of_every_flare_so_far() -> None:
    spec = forced(peak=1.0, residual=0.5)
    states = states_of((RELAPSE, 2), (REMISSION, 61), (RELAPSE, 2), (REMISSION, 61))

    floor = edss.edss_trajectory(states, spec, rng=0)["floor"].to_numpy()

    # The first flare opens at week 0 and the second at week 63, and each one
    # raises the floor by the half point it leaves behind, from the week it
    # starts rather than from the week it ends.
    assert [floor[0], floor[62], floor[63], floor[-1]] == pytest.approx([2.5, 2.5, 3.0, 3.0])


def test_the_floor_carries_the_progression_term_beside_the_residuals() -> None:
    # The floor is written as E0 + k t / WEEKS_PER_YEAR + the residual of every
    # flare that has opened, and every other test of the floor runs at the
    # default rate of zero, where the middle term is invisible. This one pins
    # all three parts of it in one reading, at a rate high enough that dropping
    # the term would move the floor by a whole point.
    spec = forced(peak=1.0, residual=0.5, pira_per_year=0.1)
    weeks = round(10.0 * WEEKS_PER_YEAR) + 1
    states = states_of((RELAPSE, 2), (REMISSION, weeks - 2))

    trace = edss.edss_trajectory(states, spec, rng=0)

    last_week = float(trace["week"].iloc[-1])
    accrued = spec.baseline + 0.5 + spec.pira_per_year * last_week / WEEKS_PER_YEAR
    assert float(trace["floor"].iloc[-1]) == pytest.approx(accrued)


def test_the_trace_holds_its_floor_while_a_slow_deficit_is_still_climbing() -> None:
    # At the published time to nadir of one week a deficit is at its peak in the
    # week it starts, so it never passes below the residual it leaves. At a
    # longer time to nadir the climb would, and the floor is what holds the
    # trace up in those weeks.
    spec = forced(peak=1.0, residual=1.0, time_to_nadir_weeks=4)

    trace = edss.edss_trajectory(states_of((RELAPSE, 4), (REMISSION, 20)), spec, rng=0)

    assert float(trace["edss"].iloc[0]) == pytest.approx(spec.baseline + 1.0)


def test_the_floor_never_falls(twin_traces: list[tuple[pd.DataFrame, pd.DataFrame]]) -> None:
    falls = [
        seed
        for seed, (trace, _draws) in enumerate(twin_traces)
        if not bool(np.all(np.diff(trace["floor"].to_numpy()) >= 0.0))
    ]

    assert falls == []


def test_the_trace_never_sits_below_its_floor(
    twin_traces: list[tuple[pd.DataFrame, pd.DataFrame]],
) -> None:
    below = [
        seed
        for seed, (trace, _draws) in enumerate(twin_traces)
        if not bool(np.all(trace["edss"].to_numpy() >= trace["floor"].to_numpy()))
    ]

    assert below == []


def reading_week(end: int, last: int) -> int:
    """Return the week a flare's recovery is read in.

    The reading is taken ``RECOVERY_WEEKS`` after the last week of the flare,
    or at the last week of the record when the flare ends too close to it. Both
    of the helpers below read their level here, so that the rule lives in one
    place and the two cannot drift apart.
    """
    return min(int(end) + RECOVERY_WEEKS, last)


@pytest.mark.parametrize(
    "spec",
    [
        forced(peak=3.0, residual=3.0, baseline=9.5),
        forced(peak=0.0, residual=0.0, baseline=0.5, scale_min=1.0),
    ],
    ids=["accumulated above the top of the scale", "accumulated below the bottom of it"],
)
def test_the_trace_holds_the_floor_column_where_the_scale_clips_it(spec: edss.EDSSSpec) -> None:
    # The floor column is what the record has accumulated, clipped to the
    # scale, and the trace is held at or above that column rather than above
    # the unclipped total. The two readings can only part where the clip bites,
    # so the rule is read here at each end of the scale in turn.
    states = states_of((RELAPSE, 2), (REMISSION, 60))

    trace = edss.edss_trajectory(states, spec, rng=0)

    assert bool(np.all(trace["edss"].to_numpy() >= trace["floor"].to_numpy()))
    assert float(trace["floor"].min()) >= spec.scale_min
    assert float(trace["floor"].max()) <= spec.scale_max


def settled_shortfalls(trace: pd.DataFrame, draws: pd.DataFrame) -> list[float]:
    """Return how far below its opening level each flare of a record ends.

    The level after a flare is read at ``reading_week``, and the level before it
    is the week before it began. A flare is read only when both of those weeks
    are settled ones.

    In front, it has to open after ``QUIET_WEEKS`` with no flare in them: where
    an earlier flare is still recovering, the trace is still falling towards the
    level that earlier flare left, and that fall is a recovery rather than a
    gain. After that many quiet weeks the deepest transient the laws can draw
    has decayed to less than a millionth of a point, so what is read is the
    settled level and nothing else.

    Behind, the reading week has to fall before the next flare opens. A reading
    taken inside a later flare carries that flare's spike, which would lift the
    level read and hide a shortfall rather than report it.
    """
    levels = trace["edss"].to_numpy()
    last = levels.size - 1
    starts = [int(start) for start in draws["start"]]
    ends = [int(end) for end in draws["end"]]
    shortfalls = []
    previous_end = -QUIET_WEEKS
    for index, (start, end) in enumerate(zip(starts, ends, strict=True)):
        opening = start - 1
        reading = reading_week(end, last)
        # Nothing follows the last flare of a record, so no later spike can
        # reach its reading week wherever that week falls.
        next_start = starts[index + 1] if index + 1 < len(starts) else last + 1
        if opening >= 0 and start - previous_end >= QUIET_WEEKS and reading < next_start:
            shortfalls.append(float(levels[opening] - levels[reading]))
        previous_end = end
    return shortfalls


def floor_shortfalls(trace: pd.DataFrame, draws: pd.DataFrame) -> list[float]:
    """Return how far below its opening floor each flare of a record ends.

    The level after a flare is read at ``reading_week``, the same week
    ``settled_shortfalls`` reads, and the level before it is the floor of the
    week the flare opened on, which is what the record had accumulated for good
    by then. Every flare with a week in front of it is read, whatever else is in
    progress.
    """
    levels = trace["edss"].to_numpy()
    floors = trace["floor"].to_numpy()
    last = levels.size - 1
    shortfalls = []
    for start, end in zip(draws["start"], draws["end"], strict=True):
        opening = int(start) - 1
        if opening >= 0:
            shortfalls.append(float(floors[opening] - levels[reading_week(end, last)]))
    return shortfalls


def test_no_flare_leaves_the_record_below_the_level_it_began_from(
    twin_traces: list[tuple[pd.DataFrame, pd.DataFrame]],
) -> None:
    """A flare never ends below where it began, once its recovery is over.

    The two levels are read off the trace alone, so this is a check on what a
    reader of the figure sees rather than a second reading of the floor column.
    The tolerance covers what is left of the flare's own decay and of the
    arithmetic, and is a ten thousandth of the quarter point at which a
    displayed score moves.
    """
    shortfalls = [
        shortfall for trace, draws in twin_traces for shortfall in settled_shortfalls(trace, draws)
    ]

    assert len(shortfalls) > QUIET_FLARES, "the sample holds too few settled flares to read"
    assert max(shortfalls) <= SETTLED_MARGIN


def test_no_flare_ends_below_what_the_record_had_accumulated_before_it(
    twin_traces: list[tuple[pd.DataFrame, pd.DataFrame]],
) -> None:
    """Every flare of the sample ends at or above the floor it opened on.

    This is the rule above read over every flare rather than over the settled
    ones. A flare that opens while an earlier one is still recovering can end
    below the week in front of it, because that earlier transient goes on
    draining away in the meantime, and that fall is a recovery rather than a
    loss of ground. What such a flare can never do is leave the record below
    what it had already accumulated for good, so the level before it is read off
    the floor column here and the level after it off the trace. It holds
    exactly, with no tolerance at all.

    The assertion follows from the two invariants tested above, that the floor
    never falls and that the trace never sits below it, so it is not
    independent cover of the recovery rule and nothing can fail it that those
    two would not fail first. It is kept because it states the rule the model
    promises in the form a reader of the trace would check it, over every flare
    rather than over the settled ones. The test that carries independent weight
    is the one above, which reads both levels off the trace alone.
    """
    shortfalls = [
        shortfall for trace, draws in twin_traces for shortfall in floor_shortfalls(trace, draws)
    ]

    assert len(shortfalls) > READABLE_FLARES, "the sample holds too few readable flares"
    assert max(shortfalls) <= 0.0


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
    # No law of the model reaches below zero, so the only way under the bottom
    # of the scale is a scale that starts above the baseline.
    spec = forced(peak=0.0, residual=0.0, baseline=0.5, scale_min=1.0)

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

    assert list(trace.columns) == ["week", "edss", "edss_display", "floor", "episode"]
    assert [trace[column].dtype.name for column in trace.columns] == [
        "int64",
        "float64",
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


def test_a_drawn_residual_is_never_negative() -> None:
    # Five thousand residuals, as a record of many flares drawn under many
    # seeds. A negative one would let a flare leave the patient permanently
    # better than before it, which is what the model refuses to do.
    states = states_of(*([(RELAPSE, 1), (REMISSION, 3)] * NEGATIVE_DRAW_EPISODES))

    residuals = pd.concat(
        [edss.episode_draws(states, rng=seed)["residual"] for seed in range(NEGATIVE_DRAW_SEEDS)],
        ignore_index=True,
    )

    assert len(residuals) == NEGATIVE_DRAW_EPISODES * NEGATIVE_DRAW_SEEDS
    assert float(residuals.min()) >= 0.0


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
    # Every relapse recovers in full here, so no peak is ever raised to its
    # residual and the peak a row carries is the draw its class was read off.
    # That is what lets the rule be asserted in both directions, so that a class
    # read off the wrong quantity, or read off the right one at the wrong side
    # of the threshold, fails the test.
    spec = dataclasses.replace(
        edss.EDSSSpec(), residual_law_mild={0.0: 1.0}, residual_law_severe={0.0: 1.0}
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


@pytest.mark.parametrize("name", ["peak_law", "residual_law_mild", "residual_law_severe"])
def test_a_law_reaching_below_zero_is_refused(name: str) -> None:
    changed: dict[str, Any] = {name: {-0.5: 0.2, 0.0: 0.8}}

    with pytest.raises(ValueError, match="cannot decrease"):
        dataclasses.replace(edss.EDSSSpec(), **changed)


def test_a_negative_progression_rate_is_refused() -> None:
    # The floor is what the record has accumulated for good, and a progression
    # rate below zero would pull it down week by week, which is the same fault
    # the laws are checked for.
    with pytest.raises(ValueError, match="cannot decrease"):
        dataclasses.replace(edss.EDSSSpec(), pira_per_year=-0.5)


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
    # The residuals cannot be read back off a trace on its own, so the count of
    # the episodes that left one is reported only when the draws are given.
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
    """The cohort level at ten years sits in the band the two anchors set.

    Pittock 2004 reports a mean change of one point over ten years in a
    population based cohort, which puts a cohort that started at 2.0 near 3 at
    ten years. Confavreux and Leray report a median of about ten years from
    onset to EDSS 3, so half a cohort is at or above 3 by then. The band is wide
    because a mean over one draw per patient carries the whole spread of the
    model, and because the residual law excludes the visit to visit improvements
    the published mean change per relapse is net of, which lifts the level this
    cohort reaches to a mean of 3.92 and a median of 3.50, above the anchor
    rather than on it.
    """
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
    assert ANCHOR_LOW <= float(np.mean(levels)) <= ANCHOR_HIGH
    # The published anchor is a level a cohort sits at rather than an average
    # over a long tail, and the median of the cohort falls in the same band.
    assert ANCHOR_LOW <= float(np.median(levels)) <= ANCHOR_HIGH
    # The band is wide enough to absorb a drift the reader would want to hear
    # about, so the mean this cohort actually reaches is pinned as well.
    assert float(np.mean(levels)) == pytest.approx(ANCHOR_MEAN, abs=ANCHOR_MEAN_TOLERANCE)


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
