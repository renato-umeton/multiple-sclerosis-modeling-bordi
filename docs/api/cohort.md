# cohort

Virtual cohorts of relapsing-remitting records, written in the three schemas of
`msrelapse.io`. A `CohortSpec` says how many patients, how long each is
followed and how long each stays in either state on average; `generate` turns
that into records through either the renewal engine or the stochastic one.

The module also carries the two corrections that stand between a continuous
time model and a weekly clinical record, the rounding correction and the
inversion of the naive mean the article measured, and it returns the three
worked patients of the article as a table, which is where the printed
inconsistency of patient 23 can be read off.

::: msrelapse.cohort
