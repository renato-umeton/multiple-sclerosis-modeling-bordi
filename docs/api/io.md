# io

The three record schemas, their validation, the conversions between them, and
the readers and the writer. All three are plain `pandas.DataFrame` objects with
fixed columns and fixed dtypes: `weekly` holds one row per patient-week,
`durations` is its run length encoding, and `events` is the registry shape with
one row per relapse.

`weekly` and `durations` convert losslessly in both directions. Going from
`events` to `weekly` is lossy, because it applies the weekly rounding rule of
the study, under which any week a relapse touches counts as a whole relapse
week. The [data page](../data.md) lists the columns of each schema.

::: msrelapse.io
