# datasets

The synthetic twin of the 2013 cohort, its provenance, and the closing table.
The clinical records of the article were never released, so what this package
ships is seventy simulated records generated from the numbers the article
prints, in the three schemas of `msrelapse.io`.

`reproduction_table` is the part that is not about the twin at all: it takes
any weekly record and measures on it, the way the article measured its own,
every aggregate the article reports. A reader who obtains the real series can
run the same function on it and read the same table. The [data
page](../data.md) describes the files and the [reproduction
page](../reproducing.md) the table.

::: msrelapse.datasets
