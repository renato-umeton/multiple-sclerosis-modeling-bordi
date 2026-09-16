"""The three relapse record schemas, their validation, conversions and readers.

A patient record is held in one of three shapes, all of them plain
``pandas.DataFrame`` objects with fixed columns and fixed dtypes.

- ``weekly``: one row per patient-week: ``patient_id``, ``week`` (0-based and
  contiguous within a patient) and ``state`` (+1 no health, -1 health).
- ``durations``: the run-length encoding of ``weekly``: ``patient_id``,
  ``run_index``, ``state``, ``duration_w`` and ``censored``. A final remission
  is censored by the end of follow up; a final relapse is complete by
  construction.
- ``events``: the registry shape, one row per relapse: ``patient_id``,
  ``followup_start``, ``followup_end``, ``relapse_onset`` and ``relapse_end``,
  all in weeks. A patient with no relapse keeps a single row with a missing
  onset and end so that the follow up window survives.

``weekly`` and ``durations`` convert losslessly in both directions.
``events`` to ``weekly`` is lossy because it applies the weekly rounding rule
of the study, under which any week touched by a relapse counts as a whole
relapse week.

The two state codes are not written down here: they are read from
``PAPER.state_no_health`` and ``PAPER.state_health`` in
[`msrelapse._params`][msrelapse._params].
"""

from __future__ import annotations

import itertools
import math
import os
from typing import Any, Final, Literal

import numpy as np
import numpy.typing as npt
import pandas as pd

from msrelapse._params import PAPER

__all__ = [
    "DAYS_PER_WEEK",
    "DURATIONS_COLUMNS",
    "EVENTS_COLUMNS",
    "EVENTS_DATE_COLUMNS",
    "WEEKLY_COLUMNS",
    "Schema",
    "durations_to_weekly",
    "events_to_weekly",
    "read_durations",
    "read_events",
    "read_weekly",
    "validate",
    "weekly_to_durations",
    "weekly_to_events",
    "write_csv",
]

Schema = Literal["weekly", "durations", "events"]

WEEKLY_COLUMNS: Final = ("patient_id", "week", "state")
DURATIONS_COLUMNS: Final = ("patient_id", "run_index", "state", "duration_w", "censored")
EVENTS_COLUMNS: Final = (
    "patient_id",
    "followup_start",
    "followup_end",
    "relapse_onset",
    "relapse_end",
)
EVENTS_DATE_COLUMNS: Final = (
    "followup_start_date",
    "followup_end_date",
    "relapse_onset_date",
    "relapse_end_date",
)

DAYS_PER_WEEK: Final = 7
"""Calendar length of a week, used to turn dates into weeks and back."""

_WEEK_TOLERANCE: Final = 1e-9
"""How far from a whole week a span may fall and still count as a whole week."""

_NO_HEALTH: Final = PAPER.state_no_health.value
_HEALTH: Final = PAPER.state_health.value

_SCHEMA_COLUMNS: Final[dict[str, tuple[str, ...]]] = {
    "weekly": WEEKLY_COLUMNS,
    "durations": DURATIONS_COLUMNS,
    "events": EVENTS_COLUMNS,
}

_INTEGER_COLUMNS: Final = frozenset({"week", "run_index", "state", "duration_w"})
_BOOL_COLUMNS: Final = frozenset({"censored"})

_INT_DTYPE: Final = np.dtype(np.int64)
_FLOAT_DTYPE: Final = np.dtype(np.float64)
_BOOL_DTYPE: Final = np.dtype(np.bool_)

_WEEKLY_DTYPES: Final[dict[str, str]] = {
    "patient_id": "str",
    "week": "int64",
    "state": "int64",
}
_DURATIONS_DTYPES: Final[dict[str, str]] = {
    "patient_id": "str",
    "run_index": "int64",
    "state": "int64",
    "duration_w": "int64",
    "censored": "bool",
}
_EVENTS_DTYPES: Final[dict[str, str]] = {
    "patient_id": "str",
    "followup_start": "float64",
    "followup_end": "float64",
    "relapse_onset": "float64",
    "relapse_end": "float64",
}

_Block = tuple[str, int, int]
_Array = npt.NDArray[Any]


def validate(df: pd.DataFrame, schema: Schema) -> None:
    """Check that a frame obeys one of the three schemas, or explain why not.

    Every rule of the schema is checked, and the first violation raises with a
    message naming the schema, the column and the patient or row at fault.
    Extra columns are a violation as much as missing ones, so that a frame can
    be converted back and forth without picking up or losing information. A
    follow up window must be finite as well as present, because an unbounded
    window has no number of weeks for
    [`events_to_weekly`][msrelapse.io.events_to_weekly] to produce. The numeric
    columns must carry exactly the numpy dtypes the schema names, ``int64``,
    ``float64`` and ``bool``, and nothing else: a narrower or a wider width
    would come back changed from a conversion, and a pandas nullable extension
    dtype can hold a missing value where the schema allows none. The rules are
    listed under Notes.

    Parameters
    ----------
    df : pandas.DataFrame
        The frame to check.
    schema : {'weekly', 'durations', 'events'}
        Name of the schema the frame is meant to follow.

    Returns
    -------
    None
        Nothing is returned; a frame that breaks a rule raises instead.

    Raises
    ------
    ValueError
        If `schema` is not one of the three names, or if `df` breaks any rule
        of that schema.

    Notes
    -----
    A message that names a row names the position of that row in the frame,
    counting from 0, and not its index label, so 'row 3' is the fourth row
    whatever index the frame carries.

    Besides the columns and the dtypes, these are the rules.

    All three schemas
        Every row names a patient, the rows of a patient are together, and the
        patients are in ascending order of their identifier.
    weekly
        ``state`` is +1 or -1, and ``week`` runs 0, 1, 2, ... within a patient.
    durations
        ``state`` is +1 or -1 and differs between neighbouring runs of one
        patient, because a run length encoding has no two runs alike;
        ``run_index`` runs 0, 1, 2, ... within a patient; ``duration_w`` is at
        least one week; ``censored`` is True only on the final run of a patient
        and only when that run is a remission.
    events
        Each patient has one follow up window, present, finite and the same on
        all of its rows, with ``followup_end`` above ``followup_start``;
        ``relapse_onset`` and ``relapse_end`` are either both present or both
        missing, and a row missing both is the only row of its patient; a
        recorded relapse starts no earlier than ``followup_start``, ends after
        it starts and no later than ``followup_end``; and the relapses of a
        patient are sorted by onset and do not overlap.

    Examples
    --------
    >>> import pandas as pd
    >>> from msrelapse._params import PAPER
    >>> relapse = PAPER.state_no_health.value
    >>> remission = PAPER.state_health.value
    >>> frame = pd.DataFrame(
    ...     {"patient_id": ["p1", "p1"], "week": [0, 1], "state": [relapse, remission]}
    ... )
    >>> validate(frame, "weekly")
    """
    columns = _columns_for(schema)
    _check_columns(df, schema, columns)
    for column in columns:
        _check_dtype(schema, column, df[column])
    _check_patient_ids_present(df, schema)
    blocks = _patient_blocks(df)
    _check_patient_order(schema, blocks)
    if schema == "weekly":
        _check_state_codes(df, schema)
        _check_contiguous(df, schema, "week", blocks)
    elif schema == "durations":
        _check_state_codes(df, schema)
        _check_contiguous(df, schema, "run_index", blocks)
        _check_durations(df)
        _check_run_alternation(df, blocks)
        _check_censoring(df, blocks)
    else:
        _check_followup_windows(df, blocks)
        _check_missing_relapses(df, blocks)
        _check_relapse_bounds(df)
        _check_relapse_order(df, blocks)


def weekly_to_durations(df: pd.DataFrame) -> pd.DataFrame:
    """Run-length encode a weekly frame into runs of constant state.

    Parameters
    ----------
    df : pandas.DataFrame
        A frame in the weekly schema. It is validated before use.

    Returns
    -------
    pandas.DataFrame
        A frame in the durations schema. ``censored`` is True only on the
        final run of a patient whose state is health, because such a run is
        cut short by the end of follow up.

    Raises
    ------
    ValueError
        If `df` does not obey the weekly schema.
    """
    validate(df, "weekly")
    states = df["state"].to_numpy()
    patient_ids: list[_Array] = []
    run_indices: list[_Array] = []
    run_states: list[_Array] = []
    durations: list[_Array] = []
    censored: list[_Array] = []
    for patient, start, stop in _patient_blocks(df):
        block = states[start:stop]
        run_start, run_stop = _run_bounds(block)
        n_runs = run_start.size
        patient_ids.append(np.full(n_runs, patient, dtype=object))
        run_indices.append(np.arange(n_runs, dtype=np.int64))
        run_states.append(block[run_start].astype(np.int64))
        durations.append((run_stop - run_start).astype(np.int64))
        flags = np.zeros(n_runs, dtype=bool)
        flags[-1] = bool(block[-1] == _HEALTH)
        censored.append(flags)
    return pd.DataFrame(
        {
            "patient_id": _patient_id_series(patient_ids, df),
            "run_index": _concat(run_indices, np.int64),
            "state": _concat(run_states, np.int64),
            "duration_w": _concat(durations, np.int64),
            "censored": _concat(censored, bool),
        }
    )


def durations_to_weekly(df: pd.DataFrame) -> pd.DataFrame:
    """Expand a durations frame back into one row per patient-week.

    Parameters
    ----------
    df : pandas.DataFrame
        A frame in the durations schema. It is validated before use.

    Returns
    -------
    pandas.DataFrame
        A frame in the weekly schema, with weeks numbered from 0 within each
        patient. The censoring flag is dropped, because a weekly record ends
        where it ends.

    Raises
    ------
    ValueError
        If `df` does not obey the durations schema.
    """
    validate(df, "durations")
    lengths = df["duration_w"].to_numpy()
    weeks: list[_Array] = []
    for _patient, start, stop in _patient_blocks(df):
        weeks.append(np.arange(int(lengths[start:stop].sum()), dtype=np.int64))
    repeated_ids = np.repeat(df["patient_id"].to_numpy(), lengths)
    return pd.DataFrame(
        {
            "patient_id": _patient_id_series([repeated_ids], df),
            "week": _concat(weeks, np.int64),
            "state": np.repeat(df["state"].to_numpy(), lengths).astype(np.int64),
        }
    )


def events_to_weekly(df: pd.DataFrame) -> pd.DataFrame:
    """Turn a table of relapse events into a weekly record, rounding up.

    This applies the rounding rule of the study and is therefore lossy: week
    ``k`` covers the half-open interval ``[k, k + 1)`` weeks measured from the
    patient's ``followup_start``, and it is marked as a relapse when any
    relapse of that patient overlaps it at all, so a relapse shorter than a
    week still occupies a whole week. The record holds
    ``ceil(followup_end - followup_start)`` weeks, and never fewer than one:
    rounding up is what the study does, so a window and a relapse each take at
    least one week however short they are, and no patient and no relapse can
    fall out of the weekly record.

    Parameters
    ----------
    df : pandas.DataFrame
        A frame in the events schema. It is validated before use.

    Returns
    -------
    pandas.DataFrame
        A frame in the weekly schema.

    Raises
    ------
    ValueError
        If `df` does not obey the events schema.

    Notes
    -----
    The weekly record holds one row per week of every follow up window, so the
    memory it needs grows with the length of the windows.
    [`validate`][msrelapse.io.validate] puts no ceiling on a window it has
    found finite, and a window of many millions of weeks therefore ends this
    call with a ``MemoryError`` rather than a message naming the patient. A
    window of that length is not a clinical record, but nothing here rejects
    it.

    Examples
    --------
    A relapse from week 2.2 to week 2.6 lasts less than a week, and the
    rounding rule gives it the whole of week 2.

    >>> import pandas as pd
    >>> from msrelapse._params import PAPER
    >>> frame = pd.DataFrame(
    ...     {
    ...         "patient_id": ["p1"],
    ...         "followup_start": [0.0],
    ...         "followup_end": [5.0],
    ...         "relapse_onset": [2.2],
    ...         "relapse_end": [2.6],
    ...     }
    ... )
    >>> weekly = events_to_weekly(frame)
    >>> len(weekly)
    5
    >>> weekly.loc[weekly["state"] == PAPER.state_no_health.value, "week"].tolist()
    [2]
    """
    validate(df, "events")
    followup_start = df["followup_start"].to_numpy()
    followup_end = df["followup_end"].to_numpy()
    onsets = df["relapse_onset"].to_numpy()
    ends = df["relapse_end"].to_numpy()
    patient_ids: list[_Array] = []
    weeks: list[_Array] = []
    states: list[_Array] = []
    for patient, start, stop in _patient_blocks(df):
        origin = float(followup_start[start])
        n_weeks = max(math.ceil(_weeks_since(float(followup_end[start]), origin)), 1)
        state = np.full(n_weeks, _HEALTH, dtype=np.int64)
        for row in range(start, stop):
            if np.isnan(onsets[row]):
                continue
            # The study rounds every relapse up to a whole week, so one is kept
            # even where the span has collapsed to nothing in floating point.
            first = min(math.floor(_weeks_since(float(onsets[row]), origin)), n_weeks - 1)
            last = max(math.ceil(_weeks_since(float(ends[row]), origin)), first + 1)
            state[first:last] = _NO_HEALTH
        patient_ids.append(np.full(n_weeks, patient, dtype=object))
        weeks.append(np.arange(n_weeks, dtype=np.int64))
        states.append(state)
    return pd.DataFrame(
        {
            "patient_id": _patient_id_series(patient_ids, df),
            "week": _concat(weeks, np.int64),
            "state": _concat(states, np.int64),
        }
    )


def weekly_to_events(df: pd.DataFrame, origin: pd.Timestamp | None = None) -> pd.DataFrame:
    """Turn a weekly record into a table of relapse events.

    Each run of relapse weeks becomes one row, with ``relapse_onset`` at its
    first week and ``relapse_end`` one week past its last week. Follow up runs
    from 0 to the number of weeks in the record. A patient who never relapses
    keeps a single row with a missing onset and end.

    Parameters
    ----------
    df : pandas.DataFrame
        A frame in the weekly schema. It is validated before use.
    origin : pandas.Timestamp, optional
        Calendar date of week 0. When given, the four columns of
        ``EVENTS_DATE_COLUMNS`` are appended, each holding `origin` plus the
        corresponding number of weeks of
        [`DAYS_PER_WEEK`][msrelapse.io.DAYS_PER_WEEK] days.

    Returns
    -------
    pandas.DataFrame
        A frame in the events schema, with the four date columns appended when
        `origin` is given. The dated frame is meant for export and
        ``validate(frame, 'events')`` rejects it until the date columns are
        dropped, because the events schema has no room for them.

    Raises
    ------
    ValueError
        If `df` does not obey the weekly schema.
    """
    validate(df, "weekly")
    states = df["state"].to_numpy()
    patient_ids: list[_Array] = []
    starts: list[_Array] = []
    ends: list[_Array] = []
    onsets: list[_Array] = []
    offsets: list[_Array] = []
    for patient, start, stop in _patient_blocks(df):
        block = states[start:stop]
        n_weeks = stop - start
        run_start, run_stop = _run_bounds(block)
        is_relapse = block[run_start] == _NO_HEALTH
        onset = run_start[is_relapse].astype(np.float64)
        offset = run_stop[is_relapse].astype(np.float64)
        if onset.size == 0:
            onset = np.array([np.nan])
            offset = np.array([np.nan])
        patient_ids.append(np.full(onset.size, patient, dtype=object))
        starts.append(np.zeros(onset.size, dtype=np.float64))
        ends.append(np.full(onset.size, float(n_weeks), dtype=np.float64))
        onsets.append(onset)
        offsets.append(offset)
    events = pd.DataFrame(
        {
            "patient_id": _patient_id_series(patient_ids, df),
            "followup_start": _concat(starts, np.float64),
            "followup_end": _concat(ends, np.float64),
            "relapse_onset": _concat(onsets, np.float64),
            "relapse_end": _concat(offsets, np.float64),
        }
    )
    if origin is None:
        return events
    for column, dated in zip(EVENTS_COLUMNS[1:], EVENTS_DATE_COLUMNS, strict=True):
        events[dated] = origin + pd.to_timedelta(events[column] * DAYS_PER_WEEK, unit="D")
    return events


def read_weekly(path: str | os.PathLike[str]) -> pd.DataFrame:
    """Read a weekly record from a CSV file and validate it.

    Parameters
    ----------
    path : str or path-like
        File to read. It must carry exactly the columns of
        ``WEEKLY_COLUMNS``, in any order.

    Returns
    -------
    pandas.DataFrame
        A frame in the weekly schema, with the columns in schema order.

    Raises
    ------
    ValueError
        If the file does not obey the weekly schema.
    """
    frame = pd.read_csv(path, dtype=_WEEKLY_DTYPES)
    validate(frame, "weekly")
    return frame[list(WEEKLY_COLUMNS)]


def read_durations(path: str | os.PathLike[str]) -> pd.DataFrame:
    """Read a durations record from a CSV file and validate it.

    Parameters
    ----------
    path : str or path-like
        File to read. It must carry exactly the columns of
        ``DURATIONS_COLUMNS``, in any order.

    Returns
    -------
    pandas.DataFrame
        A frame in the durations schema, with the columns in schema order.

    Raises
    ------
    ValueError
        If the file does not obey the durations schema.
    """
    frame = pd.read_csv(path, dtype=_DURATIONS_DTYPES)
    validate(frame, "durations")
    return frame[list(DURATIONS_COLUMNS)]


def read_events(
    path: str | os.PathLike[str],
    date_cols: tuple[str, str, str, str] | None = None,
    week_length_days: int = DAYS_PER_WEEK,
    date_format: str = "ISO8601",
) -> pd.DataFrame:
    """Read a table of relapse events from a CSV file and validate it.

    The file may already be expressed in weeks, or it may be a registry export
    holding calendar dates. `week_length_days` belongs to this function alone:
    it is the only place where dates are turned into weeks, so the conversion
    functions of this module take no such argument.

    Parameters
    ----------
    path : str or path-like
        File to read.
    date_cols : tuple of str, optional
        Names of the date columns of the file, in the order
        ``(followup_start, followup_end, relapse_onset, relapse_end)``. When
        omitted, the file is expected to hold the events schema in weeks.
        When given, those four columns are parsed as dates, expressed as weeks
        since the first ``followup_start`` date of their patient, and renamed
        to the schema names; any other column of the file except
        ``patient_id`` is dropped.
    week_length_days : int, default 7
        Days per week used to convert dates into weeks. Must be positive, and
        it is checked on every call even though a file already expressed in
        weeks has no use for it.
    date_format : str, default 'ISO8601'
        The one format that every date column of the file follows, passed
        through to ``pandas.to_datetime``. The default reads any ISO 8601
        date. A file written in another convention has to name its format, for
        example ``'%d/%m/%Y'`` for a day first export, and a date that does not
        follow the named format is an error rather than a guess: left to guess,
        the reader would settle each column on its own, and one file could come
        back read two ways. Used only when `date_cols` is given.

    Returns
    -------
    pandas.DataFrame
        A frame in the events schema, with the columns in schema order.

    Raises
    ------
    ValueError
        If `week_length_days` is not positive, if a named column is absent, if
        a date does not follow `date_format`, or if the result does not obey
        the events schema.
    """
    if week_length_days <= 0:
        raise ValueError(f"week_length_days must be positive, got {week_length_days!r}")
    if date_cols is None:
        frame = pd.read_csv(path, dtype=_EVENTS_DTYPES, float_precision="round_trip")
        validate(frame, "events")
        return frame[list(EVENTS_COLUMNS)]
    raw = pd.read_csv(path, dtype={"patient_id": _EVENTS_DTYPES["patient_id"]})
    missing = [name for name in ("patient_id", *date_cols) if name not in raw.columns]
    if missing:
        raise ValueError(
            f"the events file is missing column(s) {missing}; it has {list(raw.columns)}"
        )
    seconds_per_week = pd.Timedelta(days=week_length_days).total_seconds()
    dates = {name: pd.to_datetime(raw[name], format=date_format) for name in date_cols}
    first_seen = pd.DataFrame(
        {"patient_id": raw["patient_id"], "followup_start": dates[date_cols[0]]}
    )
    start_date = first_seen.groupby("patient_id", sort=False)["followup_start"].transform("first")
    frame = pd.DataFrame({"patient_id": raw["patient_id"]})
    for column, name in zip(EVENTS_COLUMNS[1:], date_cols, strict=True):
        elapsed = (dates[name] - start_date).dt.total_seconds()
        frame[column] = (elapsed / seconds_per_week).astype("float64")
    validate(frame, "events")
    return frame


def write_csv(df: pd.DataFrame, path: str | os.PathLike[str]) -> None:
    """Write a frame to a CSV file with its columns in schema order.

    The schema is recognised from the columns of the frame, so that a file
    written here and read back by the matching reader comes back unchanged.
    An events frame carrying the four date columns of
    ``EVENTS_DATE_COLUMNS`` is written with those columns last.

    Parameters
    ----------
    df : pandas.DataFrame
        The frame to write. Its values are not validated, only its column
        names, which have to name a schema and to be distinct.
    path : str or path-like
        File to write. The row index is not written.

    Returns
    -------
    None
        Nothing is returned; the frame is written to `path`.

    Raises
    ------
    ValueError
        If `df` has a repeated column name, or if its columns match none of
        the three schemas.

    Notes
    -----
    Every line ends with a single newline, on every platform. Left to itself
    pandas ends a line with ``os.linesep``, so the same frame would write
    different bytes on Windows from the ones it writes elsewhere, and two files
    holding the same records could not be compared byte for byte. Naming the
    terminator is what makes the shipped files of
    [`msrelapse.datasets`][msrelapse.datasets] reproducible from a regeneration
    on any machine.
    """
    df[list(_output_columns(df))].to_csv(path, index=False, lineterminator="\n")


def _columns_for(schema: Schema) -> tuple[str, ...]:
    """Return the columns of a schema, or raise on an unknown schema name."""
    try:
        return _SCHEMA_COLUMNS[schema]
    except KeyError:
        known = ", ".join(repr(name) for name in _SCHEMA_COLUMNS)
        raise ValueError(f"unknown schema {schema!r}, expected one of {known}") from None


def _check_columns(df: pd.DataFrame, schema: str, columns: tuple[str, ...]) -> None:
    """Check that the frame carries exactly the columns of the schema, once each."""
    missing = [name for name in columns if name not in df.columns]
    if missing:
        raise ValueError(f"schema {schema!r} is missing column(s) {missing}")
    unexpected = [str(name) for name in df.columns if name not in columns]
    if unexpected:
        raise ValueError(
            f"schema {schema!r} has unexpected column(s) {unexpected}; extra columns are "
            "rejected so that a round trip between schemas keeps the frame unchanged"
        )
    duplicated = _duplicated_columns(df)
    if duplicated:
        raise ValueError(f"schema {schema!r} has duplicated column(s) {duplicated}")


def _duplicated_columns(df: pd.DataFrame) -> list[str]:
    """Return the names a frame carries more than once, in order of repetition."""
    return [str(name) for name in df.columns[df.columns.duplicated()]]


def _check_dtype(schema: str, column: str, values: pd.Series) -> None:
    """Check the dtype of one column against what the schema asks for."""
    dtype = values.dtype
    # A pandas extension dtype is spelled like the numpy one it shadows, apart
    # from its capital letter, so it is named in full rather than quoted.
    got = str(dtype) if isinstance(dtype, np.dtype) else f"the pandas extension dtype {dtype}"
    if column == "patient_id":
        wanted = "a string dtype, holding only strings"
        if pd.api.types.is_object_dtype(dtype):
            # An object column is accepted for what it holds, not for its dtype,
            # which says nothing. Missing values are reported separately.
            content = pd.api.types.infer_dtype(values, skipna=True)
            accepted = content in {"string", "empty"}
            got = f"object holding {content} values"
        else:
            accepted = pd.api.types.is_string_dtype(dtype)
    elif column in _BOOL_COLUMNS:
        # The dtype the schema names, and nothing else. A narrower or wider one
        # would come back changed from a conversion, and a nullable extension
        # dtype may hold a missing value, which answers none of the questions
        # the rules of the schema ask of this column.
        accepted = dtype == _BOOL_DTYPE
        wanted = "a boolean dtype"
    elif column in _INTEGER_COLUMNS:
        accepted = dtype == _INT_DTYPE
        wanted = "an int64 dtype"
    else:
        accepted = dtype == _FLOAT_DTYPE
        wanted = "a float64 dtype"
    if not accepted:
        raise ValueError(f"schema {schema!r} column {column!r} must have {wanted}, got {got}")


def _check_patient_ids_present(df: pd.DataFrame, schema: str) -> None:
    """Check that every row names a patient, so that rows can be grouped at all."""
    offending = np.flatnonzero(df["patient_id"].isna().to_numpy())
    if offending.size:
        raise ValueError(
            f"schema {schema!r} column 'patient_id' must not be missing, but row "
            f"{int(offending[0])} has no identifier"
        )


def _patient_blocks(df: pd.DataFrame) -> list[_Block]:
    """Return one ``(patient_id, start, stop)`` triple per stretch of equal ids."""
    ids = df["patient_id"].to_numpy()
    if ids.size == 0:
        return []
    change = np.flatnonzero(ids[1:] != ids[:-1]) + 1
    starts = np.concatenate(([0], change))
    stops = np.concatenate((change, [ids.size]))
    return [
        (str(ids[start]), int(start), int(stop)) for start, stop in zip(starts, stops, strict=True)
    ]


def _check_patient_order(schema: str, blocks: list[_Block]) -> None:
    """Check that patients come in ascending order and each in a single stretch."""
    for (earlier, _, _), (later, start, _) in itertools.pairwise(blocks):
        if later <= earlier:
            raise ValueError(
                f"schema {schema!r} rows must be sorted by patient_id, with all rows of a "
                f"patient together, but patient {later!r} starts at row {start}, after "
                f"patient {earlier!r}"
            )


def _first_offending(df: pd.DataFrame, offending: _Array) -> tuple[int, Any] | None:
    """Return the row and the patient of the first True of a mask, or None."""
    rows = np.flatnonzero(offending)
    if rows.size == 0:
        return None
    row = int(rows[0])
    return row, df["patient_id"].to_numpy()[row]


def _check_state_codes(df: pd.DataFrame, schema: str) -> None:
    """Check that every state is one of the two clinical codes."""
    states = df["state"].to_numpy()
    first = _first_offending(df, (states != _NO_HEALTH) & (states != _HEALTH))
    if first is not None:
        row, patient = first
        raise ValueError(
            f"schema {schema!r} column 'state' must be {_NO_HEALTH:+d} or {_HEALTH:+d}, got "
            f"{states[row]} at row {row} for patient {patient!r}"
        )


def _check_contiguous(df: pd.DataFrame, schema: str, column: str, blocks: list[_Block]) -> None:
    """Check that a counter runs 0, 1, 2, ... within every patient."""
    values = df[column].to_numpy()
    for patient, start, stop in blocks:
        expected = np.arange(stop - start)
        offending = np.flatnonzero(values[start:stop] != expected)
        if offending.size:
            offset = int(offending[0])
            raise ValueError(
                f"schema {schema!r} column {column!r} must run contiguously from 0 within a "
                f"patient, but patient {patient!r} has {values[start + offset]} at row "
                f"{start + offset} where {expected[offset]} was expected"
            )


def _check_durations(df: pd.DataFrame) -> None:
    """Check that every run lasts at least one week."""
    lengths = df["duration_w"].to_numpy()
    first = _first_offending(df, lengths < 1)
    if first is not None:
        row, patient = first
        raise ValueError(
            f"schema 'durations' column 'duration_w' must be at least 1 week, got "
            f"{lengths[row]} at row {row} for patient {patient!r}"
        )


def _check_run_alternation(df: pd.DataFrame, blocks: list[_Block]) -> None:
    """Check that neighbouring runs of one patient carry different states."""
    states = df["state"].to_numpy()
    for patient, start, stop in blocks:
        block = states[start:stop]
        repeated = np.flatnonzero(block[1:] == block[:-1])
        if repeated.size:
            offset = int(repeated[0])
            row = start + offset
            raise ValueError(
                f"schema 'durations' column 'state' must alternate between runs, but patient "
                f"{patient!r} has state {block[offset]:+d} at row {row} and row {row + 1}; a "
                "durations frame is a run length encoding, so two neighbouring runs cannot "
                "share a state"
            )


def _check_censoring(df: pd.DataFrame, blocks: list[_Block]) -> None:
    """Check that only a final remission carries the censoring flag."""
    censored = df["censored"].to_numpy()
    states = df["state"].to_numpy()
    for patient, start, stop in blocks:
        flags = censored[start:stop]
        offending = np.flatnonzero(flags[:-1])
        if offending.size:
            row = start + int(offending[0])
            raise ValueError(
                f"schema 'durations' column 'censored' is True at row {row} for patient "
                f"{patient!r}, which is not the final run of that patient"
            )
        last = stop - 1
        if flags[-1] and states[last] == _NO_HEALTH:
            raise ValueError(
                f"schema 'durations' column 'censored' is True at row {last} for patient "
                f"{patient!r}, whose state is {_NO_HEALTH:+d}; only a final remission can be "
                "censored, because a record ends with a completed relapse"
            )


def _check_followup_windows(df: pd.DataFrame, blocks: list[_Block]) -> None:
    """Check that each patient has one finite follow up window and that it is not empty."""
    starts = df["followup_start"].to_numpy()
    ends = df["followup_end"].to_numpy()
    for patient, start, stop in blocks:
        for column, values in (("followup_start", starts), ("followup_end", ends)):
            window = values[start:stop]
            missing = np.flatnonzero(np.isnan(window))
            if missing.size:
                row = start + int(missing[0])
                raise ValueError(
                    f"schema 'events' column {column!r} must not be missing, but row {row} of "
                    f"patient {patient!r} has no value for it"
                )
            unbounded = np.flatnonzero(np.isinf(window))
            if unbounded.size:
                row = start + int(unbounded[0])
                raise ValueError(
                    f"schema 'events' column {column!r} must be a finite number of weeks, but "
                    f"row {row} of patient {patient!r} has {values[row]}"
                )
        varies = np.any(starts[start:stop] != starts[start]) or np.any(
            ends[start:stop] != ends[start]
        )
        if varies:
            raise ValueError(
                "schema 'events' columns 'followup_start' and 'followup_end' must be constant "
                f"within a patient, but patient {patient!r} has more than one follow up window"
            )
        if not ends[start] > starts[start]:
            raise ValueError(
                "schema 'events' column 'followup_end' must be greater than followup_start, but "
                f"patient {patient!r} has followup_start {starts[start]} and followup_end "
                f"{ends[start]}"
            )


def _check_missing_relapses(df: pd.DataFrame, blocks: list[_Block]) -> None:
    """Check that a missing relapse is complete and stands alone for its patient."""
    onsets = df["relapse_onset"].to_numpy()
    ends = df["relapse_end"].to_numpy()
    half_missing = np.flatnonzero(np.isnan(onsets) != np.isnan(ends))
    if half_missing.size:
        row = int(half_missing[0])
        patient = df["patient_id"].to_numpy()[row]
        raise ValueError(
            "schema 'events' columns 'relapse_onset' and 'relapse_end' must both be present or "
            f"both be missing, but row {row} of patient {patient!r} has only one of them"
        )
    for patient, start, stop in blocks:
        missing = np.flatnonzero(np.isnan(onsets[start:stop]))
        if missing.size and stop - start > 1:
            row = start + int(missing[0])
            raise ValueError(
                f"schema 'events' row {row} records no relapse, so it must be the only row of "
                f"that patient, but patient {patient!r} has {stop - start} rows"
            )


def _check_relapse_bounds(df: pd.DataFrame) -> None:
    """Check that every recorded relapse advances in time and lies inside follow up."""
    starts = df["followup_start"].to_numpy()
    ends = df["followup_end"].to_numpy()
    onsets = df["relapse_onset"].to_numpy()
    offsets = df["relapse_end"].to_numpy()
    _raise_on_first(
        df,
        onsets < starts,
        "column 'relapse_onset' must not be before followup_start",
    )
    _raise_on_first(
        df,
        offsets <= onsets,
        "column 'relapse_end' must be greater than relapse_onset",
    )
    _raise_on_first(
        df,
        offsets > ends,
        "column 'relapse_end' must not be after followup_end",
    )


def _raise_on_first(df: pd.DataFrame, offending: _Array, complaint: str) -> None:
    """Raise about the first offending row, quoting its patient and its values."""
    first = _first_offending(df, offending)
    if first is None:
        return
    row, patient = first
    values = ", ".join(f"{name} {df[name].to_numpy()[row]}" for name in EVENTS_COLUMNS[1:])
    raise ValueError(
        f"schema 'events' {complaint}, but row {row} of patient {patient!r} has {values}"
    )


def _check_relapse_order(df: pd.DataFrame, blocks: list[_Block]) -> None:
    """Check that the relapses of a patient are sorted and do not overlap."""
    onsets = df["relapse_onset"].to_numpy()
    ends = df["relapse_end"].to_numpy()
    for patient, start, stop in blocks:
        onset = onsets[start:stop]
        end = ends[start:stop]
        unsorted_rows = np.flatnonzero(onset[1:] < onset[:-1])
        if unsorted_rows.size:
            row = start + int(unsorted_rows[0])
            raise ValueError(
                "schema 'events' rows must be sorted by patient_id then relapse_onset, but "
                f"patient {patient!r} has relapse_onset {onsets[row + 1]} at row {row + 1} "
                f"after {onsets[row]} at row {row}"
            )
        overlapping = np.flatnonzero(onset[1:] < end[:-1])
        if overlapping.size:
            row = start + int(overlapping[0])
            raise ValueError(
                f"schema 'events' has overlapping relapses for patient {patient!r}: the relapse "
                f"at row {row} ends at {ends[row]} and the next one starts at {onsets[row + 1]} "
                f"at row {row + 1}"
            )


def _run_bounds(states: _Array) -> tuple[_Array, _Array]:
    """Return the start and stop index of each run of equal states."""
    change = np.flatnonzero(states[1:] != states[:-1]) + 1
    return np.concatenate(([0], change)), np.concatenate((change, [states.size]))


def _weeks_since(value: float, origin: float) -> float:
    """Return `value` minus `origin`, snapped to a whole week when it is one.

    A follow up window written in tenths of a week subtracts inexactly in
    binary floating point, so a span of exactly one week can come out as
    0.999999999999999 or 1.000000000000001. Rounding such a value up or down
    would move a relapse across a week boundary, so a result within
    ``_WEEK_TOLERANCE`` of a whole number is returned as that whole number.
    """
    elapsed = value - origin
    nearest = round(elapsed)
    if abs(elapsed - nearest) <= _WEEK_TOLERANCE:
        return float(nearest)
    return elapsed


def _concat(parts: list[_Array], dtype: npt.DTypeLike) -> _Array:
    """Join the per-patient pieces of a column, keeping an empty frame empty."""
    if not parts:
        return np.empty(0, dtype=dtype)
    return np.concatenate(parts)


def _patient_id_series(parts: list[_Array], source: pd.DataFrame) -> pd.Series:
    """Build the identifier column, keeping the dtype the source frame used."""
    return pd.Series(_concat(parts, object), dtype=source["patient_id"].dtype)


def _output_columns(df: pd.DataFrame) -> tuple[str, ...]:
    """Return the schema order of the columns of a frame, for writing."""
    duplicated = _duplicated_columns(df)
    if duplicated:
        raise ValueError(
            f"the frame has duplicated column(s) {duplicated} and cannot be written, because no "
            "reader of this module could read the file back"
        )
    present = set(df.columns)
    for columns in _SCHEMA_COLUMNS.values():
        if present == set(columns):
            return columns
    if present == set(EVENTS_COLUMNS) | set(EVENTS_DATE_COLUMNS):
        return EVENTS_COLUMNS + EVENTS_DATE_COLUMNS
    raise ValueError(
        f"the frame matches no schema, its columns are {sorted(str(name) for name in present)}; "
        f"expected one of {list(_SCHEMA_COLUMNS.values())}"
    )
