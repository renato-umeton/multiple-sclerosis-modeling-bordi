from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from pandas.testing import assert_frame_equal

from msrelapse._params import PAPER
from msrelapse.io import (
    DURATIONS_COLUMNS,
    EVENTS_COLUMNS,
    EVENTS_DATE_COLUMNS,
    WEEKLY_COLUMNS,
    durations_to_weekly,
    events_to_weekly,
    read_durations,
    read_events,
    read_weekly,
    validate,
    weekly_to_durations,
    weekly_to_events,
    write_csv,
)

NO_HEALTH = PAPER.state_no_health.value
HEALTH = PAPER.state_health.value

# One patient who starts with a relapse and ends with a remission, one who ends
# with a relapse, and one who never relapses.
EXAMPLE_STATES: dict[str, list[int]] = {
    "p1": [NO_HEALTH, NO_HEALTH, HEALTH, HEALTH, HEALTH],
    "p2": [NO_HEALTH, NO_HEALTH, HEALTH, HEALTH, NO_HEALTH],
    "p3": [HEALTH, HEALTH, HEALTH],
}


def weekly_frame(states_by_patient: dict[str, list[int]]) -> pd.DataFrame:
    patient_ids: list[str] = []
    weeks: list[int] = []
    states: list[int] = []
    for patient_id, sequence in states_by_patient.items():
        patient_ids.extend([patient_id] * len(sequence))
        weeks.extend(range(len(sequence)))
        states.extend(sequence)
    return pd.DataFrame(
        {
            "patient_id": patient_ids,
            "week": np.asarray(weeks, dtype="int64"),
            "state": np.asarray(states, dtype="int64"),
        }
    )


def durations_frame(rows: list[tuple[str, int, int, int, bool]]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "patient_id": [row[0] for row in rows],
            "run_index": np.asarray([row[1] for row in rows], dtype="int64"),
            "state": np.asarray([row[2] for row in rows], dtype="int64"),
            "duration_w": np.asarray([row[3] for row in rows], dtype="int64"),
            "censored": np.asarray([row[4] for row in rows], dtype=bool),
        }
    )


def events_frame(rows: list[tuple[str, float, float, float, float]]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "patient_id": [row[0] for row in rows],
            "followup_start": np.asarray([row[1] for row in rows], dtype="float64"),
            "followup_end": np.asarray([row[2] for row in rows], dtype="float64"),
            "relapse_onset": np.asarray([row[3] for row in rows], dtype="float64"),
            "relapse_end": np.asarray([row[4] for row in rows], dtype="float64"),
        }
    )


def example_weekly() -> pd.DataFrame:
    return weekly_frame(EXAMPLE_STATES)


def example_durations() -> pd.DataFrame:
    return durations_frame(
        [
            ("p1", 0, NO_HEALTH, 2, False),
            ("p1", 1, HEALTH, 3, True),
            ("p2", 0, NO_HEALTH, 2, False),
            ("p2", 1, HEALTH, 2, False),
            ("p2", 2, NO_HEALTH, 1, False),
            ("p3", 0, HEALTH, 3, True),
        ]
    )


def example_events() -> pd.DataFrame:
    return events_frame(
        [
            ("p1", 0.0, 5.0, 0.0, 2.0),
            ("p2", 0.0, 5.0, 0.0, 2.0),
            ("p2", 0.0, 5.0, 4.0, 5.0),
            ("p3", 0.0, 3.0, np.nan, np.nan),
        ]
    )


def test_column_tuples_are_the_documented_schemas() -> None:
    assert WEEKLY_COLUMNS == ("patient_id", "week", "state")
    assert DURATIONS_COLUMNS == (
        "patient_id",
        "run_index",
        "state",
        "duration_w",
        "censored",
    )
    assert EVENTS_COLUMNS == (
        "patient_id",
        "followup_start",
        "followup_end",
        "relapse_onset",
        "relapse_end",
    )


def test_validate_accepts_the_example_frames() -> None:
    validate(example_weekly(), "weekly")
    validate(example_durations(), "durations")
    validate(example_events(), "events")


def test_validate_accepts_empty_frames() -> None:
    validate(example_weekly().iloc[:0], "weekly")
    validate(example_durations().iloc[:0], "durations")
    validate(example_events().iloc[:0], "events")


def test_weekly_to_durations_encodes_the_example() -> None:
    assert_frame_equal(weekly_to_durations(example_weekly()), example_durations())


def test_durations_to_weekly_expands_the_example() -> None:
    assert_frame_equal(durations_to_weekly(example_durations()), example_weekly())


def test_weekly_to_durations_to_weekly_is_the_identity() -> None:
    weekly = example_weekly()
    assert_frame_equal(durations_to_weekly(weekly_to_durations(weekly)), weekly)


def test_weekly_to_events_encodes_the_example() -> None:
    assert_frame_equal(weekly_to_events(example_weekly()), example_events())


def test_events_to_weekly_expands_the_example() -> None:
    assert_frame_equal(events_to_weekly(example_events()), example_weekly())


def test_weekly_to_events_to_weekly_is_the_identity() -> None:
    weekly = example_weekly()
    assert_frame_equal(events_to_weekly(weekly_to_events(weekly)), weekly)


def test_conversions_keep_empty_frames_empty() -> None:
    empty_weekly = example_weekly().iloc[:0]
    assert_frame_equal(weekly_to_durations(empty_weekly), example_durations().iloc[:0])
    assert_frame_equal(durations_to_weekly(example_durations().iloc[:0]), empty_weekly)
    assert_frame_equal(weekly_to_events(empty_weekly), example_events().iloc[:0])
    assert_frame_equal(events_to_weekly(example_events().iloc[:0]), empty_weekly)


@pytest.mark.parametrize(
    ("onset", "end", "expected_relapse_weeks"),
    [
        (2.2, 2.6, [2]),
        (2.5, 3.5, [2, 3]),
        (2.0, 3.0, [2]),
        (0.1, 0.2, [0]),
        (0.0, 4.0, [0, 1, 2, 3]),
    ],
)
def test_events_to_weekly_rounds_a_relapse_up_to_whole_weeks(
    onset: float, end: float, expected_relapse_weeks: list[int]
) -> None:
    weekly = events_to_weekly(events_frame([("p1", 0.0, 5.0, onset, end)]))
    relapse_weeks = weekly.loc[weekly["state"] == NO_HEALTH, "week"].tolist()
    assert relapse_weeks == expected_relapse_weeks


def test_events_to_weekly_rounds_the_follow_up_length_up() -> None:
    weekly = events_to_weekly(events_frame([("p1", 0.0, 10.4, np.nan, np.nan)]))
    assert weekly["week"].tolist() == list(range(11))
    assert weekly["state"].tolist() == [HEALTH] * 11


def test_events_to_weekly_numbers_weeks_from_the_follow_up_start() -> None:
    weekly = events_to_weekly(events_frame([("p1", 3.0, 8.0, 5.2, 6.1)]))
    assert weekly["week"].tolist() == [0, 1, 2, 3, 4]
    assert weekly.loc[weekly["state"] == NO_HEALTH, "week"].tolist() == [2, 3]


def test_events_to_weekly_numbers_weeks_from_each_patients_own_start() -> None:
    # The two patients are followed over different calendar windows of the same
    # length, so each must come back numbered from its own follow up start.
    weekly = events_to_weekly(
        events_frame([("p1", 10.0, 13.0, 11.0, 12.0), ("p2", 0.0, 3.0, 1.0, 2.0)])
    )
    for patient in ("p1", "p2"):
        block = weekly.loc[weekly["patient_id"] == patient]
        assert block["week"].tolist() == [0, 1, 2]
        assert block.loc[block["state"] == NO_HEALTH, "week"].tolist() == [1]


def test_events_to_weekly_gives_a_window_shorter_than_the_rounding_tolerance_one_week() -> None:
    # A window this short is not a whole number of weeks, but a patient that
    # validate accepts must never fall out of the weekly record.
    weekly = events_to_weekly(events_frame([("p1", 0.0, 1e-10, np.nan, np.nan)]))
    assert weekly["week"].tolist() == [0]


def test_events_to_weekly_gives_a_relapse_shorter_than_the_rounding_tolerance_one_week() -> None:
    weekly = events_to_weekly(events_frame([("p1", 0.0, 5.0, 1.0, 1.0000000001)]))
    assert weekly.loc[weekly["state"] == NO_HEALTH, "week"].tolist() == [1]


def test_events_to_weekly_keeps_a_relapse_starting_just_before_the_follow_up_end() -> None:
    weekly = events_to_weekly(events_frame([("p1", 0.0, 5.0, 4.9999999999, 5.0)]))
    assert weekly.loc[weekly["state"] == NO_HEALTH, "week"].tolist() == [4]


def test_events_to_weekly_is_not_disturbed_by_a_fractional_follow_up_start() -> None:
    # 1.4 - 0.4 is exactly one week, but in binary floating point it falls just
    # short of it, which would stretch the relapse over two weeks.
    weekly = events_to_weekly(events_frame([("p1", 0.4, 2.4, 1.4, 1.9)]))
    assert weekly["week"].tolist() == [0, 1]
    assert weekly.loc[weekly["state"] == NO_HEALTH, "week"].tolist() == [1]


def test_events_to_weekly_does_not_invent_a_week_from_rounding_error() -> None:
    # 2.2 - 1.2 is exactly one week, but in binary floating point it just
    # exceeds it, which would add a spurious twelfth week.
    weekly = events_to_weekly(events_frame([("p1", 1.2, 2.2, np.nan, np.nan)]))
    assert weekly["week"].tolist() == [0]


def test_weekly_to_events_with_an_origin_adds_dates() -> None:
    origin = pd.Timestamp("2001-01-01")
    dated = weekly_to_events(example_weekly(), origin=origin)
    assert tuple(dated.columns) == EVENTS_COLUMNS + EVENTS_DATE_COLUMNS
    assert_frame_equal(dated[list(EVENTS_COLUMNS)], example_events())
    first = dated.iloc[0]
    assert first["followup_start_date"] == origin
    assert first["followup_end_date"] == origin + pd.Timedelta(days=35)
    assert first["relapse_onset_date"] == origin
    assert first["relapse_end_date"] == origin + pd.Timedelta(days=14)
    without_relapse = dated.loc[dated["patient_id"] == "p3"].iloc[0]
    assert pd.isna(without_relapse["relapse_onset_date"])
    assert pd.isna(without_relapse["relapse_end_date"])


def test_validate_rejects_the_dated_events_frame() -> None:
    dated = weekly_to_events(example_weekly(), origin=pd.Timestamp("2001-01-01"))
    with pytest.raises(ValueError, match="unexpected column"):
        validate(dated, "events")
    validate(dated.drop(columns=list(EVENTS_DATE_COLUMNS)), "events")


def test_validate_rejects_an_unknown_schema() -> None:
    with pytest.raises(ValueError, match="unknown schema"):
        validate(example_weekly(), "wide")  # type: ignore[arg-type]


def test_validate_rejects_a_missing_column() -> None:
    with pytest.raises(ValueError, match="missing column"):
        validate(example_weekly().drop(columns=["state"]), "weekly")


def test_validate_rejects_an_extra_column() -> None:
    frame = example_weekly().assign(comment="unexpected")
    with pytest.raises(ValueError, match="unexpected column"):
        validate(frame, "weekly")


def test_validate_rejects_a_repeated_column() -> None:
    frame = example_weekly()
    doubled = pd.concat([frame, frame["week"]], axis=1)
    with pytest.raises(ValueError, match="duplicated column"):
        validate(doubled, "weekly")


def test_validate_rejects_a_float_week() -> None:
    frame = example_weekly().astype({"week": "float64"})
    with pytest.raises(ValueError, match="'week' must have an int64 dtype"):
        validate(frame, "weekly")


def test_validate_rejects_a_narrow_integer_week() -> None:
    # A narrower integer survives validate but comes back widened from a round
    # trip, so the frame the caller gets is not the frame it handed over.
    frame = example_weekly().astype({"week": "int8"})
    with pytest.raises(ValueError, match="'week' must have an int64 dtype"):
        validate(frame, "weekly")


def test_validate_rejects_a_narrow_float_follow_up_end() -> None:
    frame = example_events().astype({"followup_end": "float32"})
    with pytest.raises(ValueError, match="'followup_end' must have a float64 dtype"):
        validate(frame, "events")


def test_validate_rejects_a_numeric_patient_id() -> None:
    frame = example_weekly().assign(patient_id=np.arange(13, dtype="int64"))
    with pytest.raises(ValueError, match="'patient_id' must have a string dtype"):
        validate(frame, "weekly")


def test_validate_rejects_an_object_patient_id_holding_numbers() -> None:
    frame = weekly_frame({"p1": [HEALTH, HEALTH]})
    frame["patient_id"] = pd.Series([1, 1], dtype=object)
    with pytest.raises(ValueError, match="'patient_id' must have a string dtype"):
        validate(frame, "weekly")


def test_validate_accepts_an_object_patient_id_holding_strings() -> None:
    frame = weekly_frame({"p1": [HEALTH, HEALTH]})
    frame["patient_id"] = pd.Series(["p1", "p1"], dtype=object)
    validate(frame, "weekly")


def test_validate_accepts_an_empty_object_patient_id() -> None:
    # An empty object column holds nothing to inspect, so it is taken on trust.
    frame = weekly_frame({"p1": [HEALTH]}).iloc[:0]
    frame["patient_id"] = frame["patient_id"].astype(object)
    validate(frame, "weekly")


def test_validate_rejects_an_object_patient_id_that_is_all_missing() -> None:
    frame = weekly_frame({"p1": [HEALTH, HEALTH]})
    frame["patient_id"] = pd.Series([None, None], dtype=object)
    with pytest.raises(ValueError, match="'patient_id' must not be missing"):
        validate(frame, "weekly")


def test_validate_rejects_an_integer_censored_flag() -> None:
    frame = example_durations().astype({"censored": "int64"})
    with pytest.raises(ValueError, match="'censored' must have a boolean dtype"):
        validate(frame, "durations")


def test_validate_rejects_an_integer_followup_end() -> None:
    frame = example_events().astype({"followup_end": "int64"})
    with pytest.raises(ValueError, match="'followup_end' must have a float64 dtype"):
        validate(frame, "events")


def test_validate_rejects_a_nullable_boolean_censored_flag() -> None:
    # A nullable extension dtype can hold a missing value, which no rule of the
    # durations schema can be checked against.
    frame = example_durations()
    frame["censored"] = pd.array([False, pd.NA, False, False, False, True], dtype="boolean")
    with pytest.raises(ValueError, match="'censored' must have a boolean dtype"):
        validate(frame, "durations")


def test_validate_rejects_a_nullable_integer_duration() -> None:
    # The message spells out which of the two dtypes the frame carries, because
    # 'Int64' and 'int64' differ only by a capital letter.
    frame = example_durations()
    frame["duration_w"] = pd.array([2, pd.NA, 2, 2, 1, 3], dtype="Int64")
    with pytest.raises(
        ValueError,
        match="'duration_w' must have an int64 dtype, got the pandas extension dtype Int64",
    ):
        validate(frame, "durations")


def test_validate_rejects_a_nullable_float_follow_up_end() -> None:
    frame = example_events()
    frame["followup_end"] = pd.array([5.0, 5.0, 5.0, pd.NA], dtype="Float64")
    with pytest.raises(ValueError, match="'followup_end' must have a float64 dtype"):
        validate(frame, "events")


def test_validate_rejects_a_state_outside_the_two_codes() -> None:
    frame = example_weekly()
    frame.loc[2, "state"] = 0
    with pytest.raises(ValueError, match=r"'state' must be \+1 or -1"):
        validate(frame, "weekly")


def test_validate_rejects_non_contiguous_weeks() -> None:
    frame = example_weekly()
    frame.loc[1, "week"] = 2
    with pytest.raises(ValueError, match="'week' must run contiguously from 0"):
        validate(frame, "weekly")


def test_validate_rejects_weeks_that_do_not_start_at_zero() -> None:
    frame = example_weekly()
    frame["week"] = frame["week"] + 1
    with pytest.raises(ValueError, match="'week' must run contiguously from 0"):
        validate(frame, "weekly")


def test_validate_rejects_non_contiguous_run_index() -> None:
    frame = example_durations()
    frame.loc[1, "run_index"] = 3
    with pytest.raises(ValueError, match="'run_index' must run contiguously from 0"):
        validate(frame, "durations")


def test_validate_rejects_a_zero_duration() -> None:
    frame = example_durations()
    frame.loc[0, "duration_w"] = 0
    with pytest.raises(ValueError, match="'duration_w' must be at least 1"):
        validate(frame, "durations")


def test_validate_rejects_two_neighbouring_runs_with_the_same_state() -> None:
    # A durations frame is a run length encoding, so two runs of the same state
    # side by side are really one run and a consumer counting rows would read
    # one episode as two.
    frame = durations_frame([("p1", 0, NO_HEALTH, 2, False), ("p1", 1, NO_HEALTH, 3, False)])
    with pytest.raises(ValueError, match="'state' must alternate between runs"):
        validate(frame, "durations")


def test_validate_accepts_the_same_state_at_the_join_between_two_patients() -> None:
    frame = durations_frame([("p1", 0, NO_HEALTH, 2, False), ("p2", 0, NO_HEALTH, 3, False)])
    validate(frame, "durations")


def test_validate_rejects_a_censored_flag_on_a_run_that_is_not_the_last() -> None:
    frame = example_durations()
    frame.loc[0, "censored"] = True
    with pytest.raises(ValueError, match="is not the final run"):
        validate(frame, "durations")


def test_validate_rejects_a_censored_relapse() -> None:
    frame = example_durations()
    frame.loc[4, "censored"] = True
    with pytest.raises(ValueError, match="only a final remission can be censored"):
        validate(frame, "durations")


def test_validate_rejects_an_empty_follow_up_window() -> None:
    frame = events_frame([("p1", 2.0, 2.0, np.nan, np.nan)])
    with pytest.raises(ValueError, match="'followup_end' must be greater than followup_start"):
        validate(frame, "events")


def test_validate_rejects_a_follow_up_window_that_varies_within_a_patient() -> None:
    frame = example_events()
    frame.loc[2, "followup_end"] = 6.0
    with pytest.raises(ValueError, match="must be constant within a patient"):
        validate(frame, "events")


def test_validate_rejects_a_missing_follow_up_window() -> None:
    frame = events_frame([("p1", 0.0, np.nan, np.nan, np.nan)])
    with pytest.raises(ValueError, match="'followup_end' must not be missing"):
        validate(frame, "events")


def test_validate_names_the_row_of_a_missing_follow_up_window() -> None:
    frame = events_frame([("p1", 0.0, 5.0, 1.0, 2.0), ("p1", 0.0, np.nan, 3.0, 4.0)])
    with pytest.raises(ValueError, match="row 1 of patient 'p1'"):
        validate(frame, "events")


def test_validate_rejects_an_infinite_follow_up_end() -> None:
    frame = events_frame([("p1", 0.0, np.inf, 1.0, 2.0)])
    with pytest.raises(ValueError, match="'followup_end' must be a finite number of weeks"):
        validate(frame, "events")


def test_validate_rejects_an_infinite_follow_up_start() -> None:
    frame = events_frame([("p1", -np.inf, 5.0, np.nan, np.nan)])
    with pytest.raises(ValueError, match="'followup_start' must be a finite number of weeks"):
        validate(frame, "events")


def test_validate_rejects_an_infinite_relapse_end() -> None:
    frame = events_frame([("p1", 0.0, 5.0, 1.0, np.inf)])
    with pytest.raises(ValueError, match="'relapse_end' must not be after followup_end"):
        validate(frame, "events")


def test_validate_rejects_a_missing_patient_id() -> None:
    frame = example_weekly()
    frame.loc[1, "patient_id"] = None
    with pytest.raises(ValueError, match="'patient_id' must not be missing"):
        validate(frame, "weekly")


def test_validate_rejects_a_relapse_before_the_follow_up_start() -> None:
    frame = events_frame([("p1", 1.0, 5.0, 0.5, 2.0)])
    with pytest.raises(ValueError, match="'relapse_onset' must not be before followup_start"):
        validate(frame, "events")


def test_validate_rejects_a_relapse_that_does_not_advance() -> None:
    frame = events_frame([("p1", 0.0, 5.0, 2.0, 2.0)])
    with pytest.raises(ValueError, match="'relapse_end' must be greater than relapse_onset"):
        validate(frame, "events")


def test_validate_rejects_a_relapse_after_the_follow_up_end() -> None:
    frame = events_frame([("p1", 0.0, 5.0, 4.0, 5.5)])
    with pytest.raises(ValueError, match="'relapse_end' must not be after followup_end"):
        validate(frame, "events")


def test_validate_rejects_a_half_missing_relapse() -> None:
    frame = events_frame([("p1", 0.0, 5.0, 2.0, np.nan)])
    with pytest.raises(ValueError, match="both be present or both be missing"):
        validate(frame, "events")


def test_validate_rejects_a_missing_relapse_beside_a_recorded_one() -> None:
    frame = events_frame([("p1", 0.0, 5.0, np.nan, np.nan), ("p1", 0.0, 5.0, 2.0, 3.0)])
    with pytest.raises(ValueError, match="must be the only row of that patient"):
        validate(frame, "events")


def test_validate_rejects_overlapping_relapses() -> None:
    frame = events_frame([("p1", 0.0, 9.0, 1.0, 4.0), ("p1", 0.0, 9.0, 3.0, 6.0)])
    with pytest.raises(ValueError, match="overlapping relapses"):
        validate(frame, "events")


def test_validate_rejects_unsorted_patients() -> None:
    frame = weekly_frame({"p2": [HEALTH], "p1": [HEALTH]})
    with pytest.raises(ValueError, match="must be sorted by patient_id"):
        validate(frame, "weekly")


def test_validate_rejects_a_patient_split_into_two_blocks() -> None:
    frame = pd.DataFrame(
        {
            "patient_id": ["p1", "p2", "p1"],
            "week": np.asarray([0, 0, 1], dtype="int64"),
            "state": np.asarray([HEALTH, HEALTH, HEALTH], dtype="int64"),
        }
    )
    with pytest.raises(ValueError, match="must be sorted by patient_id"):
        validate(frame, "weekly")


def test_validate_rejects_unsorted_relapses() -> None:
    frame = events_frame([("p1", 0.0, 9.0, 5.0, 6.0), ("p1", 0.0, 9.0, 1.0, 2.0)])
    with pytest.raises(ValueError, match="must be sorted by patient_id then relapse_onset"):
        validate(frame, "events")


def test_conversions_validate_their_input() -> None:
    frame = example_weekly()
    frame.loc[2, "state"] = 0
    with pytest.raises(ValueError, match=r"'state' must be \+1 or -1"):
        weekly_to_durations(frame)


@st.composite
def weekly_frames(draw: st.DrawFn) -> pd.DataFrame:
    n_patients = draw(st.integers(min_value=1, max_value=5))
    states_by_patient: dict[str, list[int]] = {}
    for index in range(n_patients):
        states_by_patient[f"p{index:02d}"] = draw(
            st.lists(
                st.sampled_from([NO_HEALTH, HEALTH]),
                min_size=1,
                max_size=60,
            )
        )
    return weekly_frame(states_by_patient)


@settings(deadline=None, max_examples=200, derandomize=True)
@given(
    tenths=st.integers(min_value=1, max_value=59),
    span=st.integers(min_value=1, max_value=20),
)
def test_events_to_weekly_counts_whole_weeks_from_any_follow_up_start(
    tenths: int, span: int
) -> None:
    start = tenths / 10
    weekly = events_to_weekly(events_frame([("p1", start, start + span, np.nan, np.nan)]))
    assert weekly["week"].tolist() == list(range(span))


@settings(deadline=None, max_examples=200, derandomize=True)
@given(
    tenths=st.integers(min_value=1, max_value=59),
    span=st.integers(min_value=1, max_value=20),
)
def test_events_to_weekly_marks_every_week_of_a_relapse_filling_follow_up(
    tenths: int, span: int
) -> None:
    start = tenths / 10
    end = start + span
    weekly = events_to_weekly(events_frame([("p1", start, end, start, end)]))
    assert weekly["state"].tolist() == [NO_HEALTH] * span


@settings(deadline=None, max_examples=200, derandomize=True)
@given(weekly=weekly_frames())
def test_weekly_and_durations_round_trip(weekly: pd.DataFrame) -> None:
    assert_frame_equal(durations_to_weekly(weekly_to_durations(weekly)), weekly)


@settings(deadline=None, max_examples=200, derandomize=True)
@given(weekly=weekly_frames())
def test_censoring_marks_exactly_the_patients_ending_in_remission(
    weekly: pd.DataFrame,
) -> None:
    durations = weekly_to_durations(weekly)
    last_run = durations.groupby("patient_id").tail(1)
    assert last_run["censored"].tolist() == (last_run["state"] == HEALTH).tolist()
    assert not durations.groupby("patient_id").head(-1)["censored"].any()


def test_write_csv_uses_the_schema_column_order(tmp_path: Path) -> None:
    path = tmp_path / "weekly.csv"
    write_csv(example_weekly()[["state", "week", "patient_id"]], path)
    assert path.read_text().splitlines()[0] == "patient_id,week,state"


def test_write_csv_keeps_the_date_columns_of_a_dated_events_frame(tmp_path: Path) -> None:
    path = tmp_path / "events.csv"
    dated = weekly_to_events(example_weekly(), origin=pd.Timestamp("2001-01-01"))
    write_csv(dated, path)
    header = path.read_text().splitlines()[0]
    assert header == ",".join(EVENTS_COLUMNS + EVENTS_DATE_COLUMNS)


def test_write_csv_rejects_a_repeated_column(tmp_path: Path) -> None:
    weekly = example_weekly()
    doubled = pd.concat([weekly, weekly["week"]], axis=1)
    with pytest.raises(ValueError, match="duplicated column"):
        write_csv(doubled, tmp_path / "weekly.csv")


def test_write_csv_rejects_a_frame_that_matches_no_schema(tmp_path: Path) -> None:
    frame = pd.DataFrame({"patient_id": ["p1"], "height": [1.7]})
    with pytest.raises(ValueError, match="matches no schema"):
        write_csv(frame, tmp_path / "unknown.csv")


def test_read_weekly_round_trips_a_written_frame(tmp_path: Path) -> None:
    path = tmp_path / "weekly.csv"
    weekly = example_weekly()
    write_csv(weekly, path)
    assert_frame_equal(read_weekly(path), weekly)


def test_read_durations_round_trips_a_written_frame(tmp_path: Path) -> None:
    path = tmp_path / "durations.csv"
    durations = example_durations()
    write_csv(durations, path)
    assert_frame_equal(read_durations(path), durations)


def test_read_events_round_trips_a_written_frame(tmp_path: Path) -> None:
    path = tmp_path / "events.csv"
    events = events_frame(
        [
            ("p1", 0.0, 5.1, 0.3, 2.7),
            ("p2", 0.0, 3.0, np.nan, np.nan),
        ]
    )
    write_csv(events, path)
    assert_frame_equal(read_events(path), events, check_exact=True)


def test_read_weekly_puts_the_columns_in_schema_order(tmp_path: Path) -> None:
    path = tmp_path / "weekly.csv"
    path.write_text("state,week,patient_id\n1,0,p1\n-1,1,p1\n")
    assert list(read_weekly(path).columns) == list(WEEKLY_COLUMNS)


def test_read_durations_puts_the_columns_in_schema_order(tmp_path: Path) -> None:
    path = tmp_path / "durations.csv"
    path.write_text("censored,duration_w,state,run_index,patient_id\nTrue,3,-1,0,p1\n")
    assert list(read_durations(path).columns) == list(DURATIONS_COLUMNS)


def test_read_events_puts_the_columns_in_schema_order(tmp_path: Path) -> None:
    path = tmp_path / "events.csv"
    path.write_text(
        "relapse_end,relapse_onset,followup_end,followup_start,patient_id\n2.0,1.0,5.0,0.0,p1\n"
    )
    assert list(read_events(path).columns) == list(EVENTS_COLUMNS)


def test_read_weekly_validates_what_it_reads(tmp_path: Path) -> None:
    path = tmp_path / "weekly.csv"
    path.write_text("patient_id,week,state\np1,0,1\np1,2,-1\n")
    with pytest.raises(ValueError, match="'week' must run contiguously from 0"):
        read_weekly(path)


def test_read_events_converts_dates_to_weeks(tmp_path: Path) -> None:
    # The two patients enter on different dates, so a reader that measured
    # weeks from one shared origin would place p2 more than a year in.
    path = tmp_path / "registry.csv"
    path.write_text(
        "patient_id,entry,exit,onset,recovery\n"
        "p1,2001-01-01,2001-03-12,2001-01-15,2001-01-29\n"
        "p2,2002-06-03,2002-07-08,,\n"
    )
    events = read_events(path, date_cols=("entry", "exit", "onset", "recovery"))
    assert_frame_equal(
        events,
        events_frame(
            [
                ("p1", 0.0, 10.0, 2.0, 4.0),
                ("p2", 0.0, 5.0, np.nan, np.nan),
            ]
        ),
        check_exact=True,
    )


def test_read_events_honours_the_week_length(tmp_path: Path) -> None:
    path = tmp_path / "registry.csv"
    path.write_text(
        "patient_id,entry,exit,onset,recovery\np1,2001-01-01,2001-01-11,2001-01-03,2001-01-06\n"
    )
    events = read_events(path, date_cols=("entry", "exit", "onset", "recovery"), week_length_days=5)
    assert events["followup_end"].tolist() == [2.0]
    assert events["relapse_onset"].tolist() == [0.4]


DAY_FIRST_REGISTRY = (
    "patient_id,entry,exit,onset,recovery\np1,01/01/2001,31/12/2001,03/02/2001,15/03/2001\n"
)


def test_read_events_rejects_dates_that_are_not_iso(tmp_path: Path) -> None:
    # Left to guess, pandas reads 'exit' and 'recovery' day first because their
    # day is above 12 and 'onset' month first because its day is not, so one
    # file would be read with two conventions.
    path = tmp_path / "registry.csv"
    path.write_text(DAY_FIRST_REGISTRY)
    with pytest.raises(ValueError, match="ISO8601"):
        read_events(path, date_cols=("entry", "exit", "onset", "recovery"))


def test_read_events_reads_a_day_first_file_when_its_format_is_named(tmp_path: Path) -> None:
    # Every column is read with the one named format, so the onset is 3 February,
    # 33 days after the entry, and not 2 March, which is 60 days after it.
    path = tmp_path / "registry.csv"
    path.write_text(DAY_FIRST_REGISTRY)
    events = read_events(
        path,
        date_cols=("entry", "exit", "onset", "recovery"),
        date_format="%d/%m/%Y",
    )
    assert events["relapse_onset"].tolist() == [33 / 7]
    assert events["relapse_end"].tolist() == [73 / 7]
    assert events["followup_end"].tolist() == [364 / 7]


def test_read_events_rejects_a_missing_date_column(tmp_path: Path) -> None:
    path = tmp_path / "registry.csv"
    path.write_text("patient_id,entry,exit,onset\np1,2001-01-01,2001-03-12,2001-01-15\n")
    with pytest.raises(ValueError, match="missing column"):
        read_events(path, date_cols=("entry", "exit", "onset", "recovery"))


def test_read_events_rejects_a_week_length_of_zero(tmp_path: Path) -> None:
    path = tmp_path / "registry.csv"
    path.write_text("patient_id,entry,exit,onset,recovery\np1,2001-01-01,2001-03-12,,\n")
    with pytest.raises(ValueError, match="week_length_days must be positive"):
        read_events(path, date_cols=("entry", "exit", "onset", "recovery"), week_length_days=0)


def test_read_events_rejects_a_negative_week_length_without_date_columns(tmp_path: Path) -> None:
    # The argument is unused here, but silently accepting a value the docstring
    # forbids would tell the caller it had been honoured.
    path = tmp_path / "events.csv"
    write_csv(example_events(), path)
    with pytest.raises(ValueError, match="week_length_days must be positive"):
        read_events(path, week_length_days=-3)
