# Reproducing the paper

The whole reproduction is one command. It needs no arguments, no data of its
own and no network, and it prints the closing table on standard output.

```bash
msrelapse reproduce --out reproduction/
```

With no options it reads the synthetic twin shipped inside the package, the
cohort that the seed 20130910 generated, and measures on it every aggregate the
article reports. The options change the record, not the measurement.

| Option | Effect |
|---|---|
| `--data FILE` | Measure a weekly CSV of your own instead of the shipped twin |
| `--engine {renewal,sde}` | Generate a fresh cohort with the named engine, not to be given with `--data` |
| `--seed N` | Generate a fresh cohort under that seed, and draw from it every bootstrap of the run but the two of the closing table |
| `--out DIR` | Where to write, created if absent, `reproduction` by default |
| `--no-figures` | Skip the figures, which need the `plot` extra |

The run repeats itself exactly. Every random draw it makes, the cohort
included, comes from the one seed it reports, so two runs with the same
arguments write the same files. The two goodness of fit rows of the closing
table are the one exception: `msrelapse.datasets.reproduction_table` draws
their parametric bootstrap from a seed fixed inside that module, so that one
record always gives one table however the run around it was seeded. With
`--seed` given, those two p values therefore sit a little apart from the ones
the `memorylessness` block of `numbers.json` reports on the same durations.
They are two draws of five hundred replicates each from one bootstrap of an
identical statistic, and not two measurements. The exit status is 0 when every
judged row of the closing table falls inside its tolerance and 1 when one of
them does not, which makes the command usable as a check in a pipeline. A
single failing row is usually the sampling noise of seventy records rather than
a fault: see the tolerances below.

!!! warning "The shipped record is synthetic"

    The article released no data, so this command reads a synthetic twin unless
    you give it a record of your own. Every run says so on standard output, and
    the file it writes carries the provenance sentence. Do not present a number
    or a figure from a default run as a clinical result. [Data](data.md) has the
    whole story.

## What the run writes

Two data files, one record of the numbers, and nine figures.

| File | Contents |
|---|---|
| `numbers.json` | Every number of the run, described below |
| `weekly.csv` | The record that was measured, in the weekly schema |
| `durations.csv` | The same record as runs of each state, with the censoring flag |
| `fig2_sample_patients.png` | Three binary series, the article's Figure 2 |
| `fig3_rr_phase_histogram.png` | Record lengths, Figure 3 |
| `fig4_duration_histograms.png` | The two duration histograms, Figure 4 |
| `fig5_symmetric_potentials.png` | The symmetric potential at two values of alpha, Figure 5 |
| `fig6_asymmetric_potential.png` | The asymmetric potential and its two barriers, Figure 6 |
| `fig7_simulated_paths.png` | Two simulated paths of equation (4), Figure 7 |
| `fig8_patient_potentials.png` | The three sample patients, Figure 8 |
| `fig_survival_vs_exponential.png` | Observed survival against the fitted exponential, not in the article |
| `fig_poisson_to_nb.png` | Relapse counts against Poisson and negative binomial mass, not in the article |

The last two figures belong to the methods note rather than to the
reproduction, because the article reports no fit and no test to compare them
with.

## What `numbers.json` contains

One JSON object, with these keys.

| Key | Contents |
|---|---|
| `source` | Where the record came from, in words |
| `synthetic` | Whether the record is synthetic, true for every default run |
| `provenance` | The provenance note of the record, or null for a record read from a file |
| `seed`, `effective_seed` | The seed as given and the seed actually used |
| `engine` | `renewal`, `sde`, or null for a record read from a file |
| `citation` | The one line citation of the article |
| `closing_table` | One object per row of the table below, with `quantity`, `paper`, `reproduced`, `tolerance` and `within_tolerance` |
| `all_within_tolerance` | Whether every judged row is inside its tolerance, the value the exit status reports |
| `fits` | Six duration fits: naive, censored and geometric, for each of the two states |
| `memorylessness` | Four tests per state: increasing hazard, coefficient of variation, Kolmogorov-Smirnov and Anderson-Darling |
| `periodicity` | The pooled Fisher g test over the per patient relapse onsets |
| `barrier_ratio` | Equation (7) for the cohort, and for the three patients with the most relapses |
| `relapse_duration_range_weeks`, `longest_remission_weeks`, `followup_range_weeks` | The three ranges, read off the closing table |
| `calibration` | The alpha, beta and sigma fitted to the measured means |

Every value is a plain number, string, boolean or pair, so the file is readable
by anything that reads JSON and needs no library of ours to open.

**Which of the six fits to read.** A weekly record holds whole weeks and no run
shorter than one week, so the geometric family is the exact law of what was
recorded and is the one to read a weekly duration with:
`fit_durations(runs, state, family="geometric")`, which uses the censoring flag
by default. The exponential family is kept because it is the law of the article
and of `msrelapse.model`, and because it is the reading that reproduces the
numbers of the article: its default continuity correction of 0 takes every
recorded duration at face value, which is the arithmetic the article did, and
`censoring=False` completes that reading by averaging the censored final
remission of each record as the article averaged it. To read an exponential on
rounded weeks otherwise, pass `continuity_correction=0.5`, which places a
duration recorded as k weeks at the midpoint of the week it ended in; the
geometric family refuses a correction, since it already lives on whole weeks.

## The closing table and its tolerances

The table is built by `msrelapse.datasets.reproduction_table`, which takes any
weekly record and measures on it the way the article measured its own: naive
means over every recorded run, censored remissions included.

The first column below is the row name the table itself prints, so a row on
screen is found here by reading it off. The third column summarises the rule;
the full sentence for each rule is in the `tolerance` column of the table and
in `numbers.json`.

| Quantity | Article | Tolerance |
|---|---|---|
| mean relapse duration (weeks) | 4.3 | 5 percent of the printed value |
| mean remission duration (weeks) | about 100 | 10 percent of the printed value |
| relapse duration range (weeks) | 1 to about 24 | shortest at least 1 and longest at most 30 |
| longest remission (weeks) | about 1000 | between 0.1 and 3 times the printed value, an order of magnitude |
| relapsing-remitting phase range (weeks) | 40 to 1311 | inside 40 to 1312 weeks, and exact only for the real series |
| barrier ratio dV1/dV2, equation (7) | about 3.1 | within 0.2 of 3.157, the value the printed durations imply |
| exponential fit, relapse durations (KS p value) | not reported | Kolmogorov-Smirnov p above 0.05 |
| exponential fit, remission durations (KS p value) | not reported | Kolmogorov-Smirnov p above 0.05 |
| periodicity of the relapse onsets (pooled Fisher g p value) | reported as absent | pooled Fisher g p above 0.05 |

The barrier ratio row above is the cohort one, and it is the only barrier ratio
in the table. The per patient ratios of patients 23, 32 and 53 are rows of
`msrelapse.cohort.paper_patients` instead: this table measures a weekly frame,
while those three values are read off the durations the article prints for each
of the three patients, so no weekly record holds a counterpart to put beside
them.

Three of the tolerances are worth a sentence each.

**The barrier ratio is compared against 3.157, not against 3.1.** The article
prints 3.1 for a value its own two durations put at 3.157, so the printed
number is already 0.057 away from the arithmetic. The tolerance of 0.2 is set
above three measured quantities: the propagation of the two mean duration
tolerances (0.177), one standard error of the two means on a cohort of this
size (0.128), and the article's own rounding (0.057). A bound of 0.05 would
need about six times the cohort of the study.

**The last three rows are not comparisons.** The article reports no fit, no
test and no interval anywhere, which section 7 of
[Paper facts](paper_facts.md) records in full, so these rows are read as the
record declining to contradict the article's two claims in prose. Failing to
reject is not evidence for a null, and the periodicity row in particular has
almost no power on records that hold about four onsets each.

**A false row is usually the cohort size.** Regenerating the twin under sixty
consecutive seeds and building the table on each, every judged row comes out
true on 27 of them. The five percent rule on the mean relapse duration fails
most often, on 24 of the 60, because five percent is narrower than one standard
error of that mean on 261 relapses. The measurement sits beside each tolerance
in the source of `msrelapse.datasets`.

## The four notebooks

They live in `notebooks/` and are executed end to end in continuous
integration, by the tests carrying the `notebook` marker in
`tests/test_notebooks.py`. Open them with
`uv run --group notebooks --with jupyterlab jupyter lab`, which adds JupyterLab
for that one command since the project does not depend on it, or execute them
headlessly the way the tests do.

**`01_reproduce_bordi2013.ipynb`** is the reproduction, in the order the
article presents it: load the record and say loudly what it is, redraw Figures
2 to 4, fit the durations three ways (naive, censored and geometric), test
memorylessness and periodicity, apply equation (7), calibrate the stochastic
equation, simulate and overlay Figure 7, draw the three per patient potentials
of Figure 8, and close with the table above. It is the notebook whose closing
table has to stay inside tolerance in continuous integration.

**`02_sde_to_exponential.ipynb`** asks why the exit times look exponential. It
sweeps the noise amplitude and puts the Kramers time, the exact mean first
passage time and the simulated mean side by side, so that the gaps between the
three are visible at each noise level. It then reads the coefficient of
variation of the measured exit times against the exponential value of 1: every
well sits near that line, which rules out an exit time far from exponential,
but with three hundred paths behind each point the panel cannot separate the
deep health well from the shallow relapse one, and it says so. The separation
comes from the regime table and the survival curves beside it, which show that
the shallow relapse well of the calibrated potential is not in the memoryless
regime at all. It closes with the Brownian bridge correction of grid read
crossings.

**`03_poisson_to_negative_binomial.ipynb`** is the trial statistics story: from
an alternating renewal record to relapse counts, from counts at one shared rate
to Poisson, and from per patient rates to the negative binomial, with the
relapse free curve of two cohorts beside it and a closing comparison of the
predicted negative binomial mapping against the fitted one. The annualised
relapse rate with its intervals is in the fourth notebook rather than this one.

**`04_virtual_cohort_for_trial_design.ipynb`** is aimed at in-silico trial
work. It generates a two arm virtual trial with a given rate ratio, computes
the annualised relapse rate in each arm, compares them, and shows what a cohort
of the size of the study can and cannot resolve.

## The inconsistencies in the article

A faithful reproduction runs into three, and the package reports each of them
rather than smoothing it over. All three are recorded in
[Paper facts](paper_facts.md) with the measurement that established them.

**Patient 23.** The article prints, for that patient, durations of 117.7 and
1.5 weeks, a barrier ratio of 11.8 and a fitted asymmetry of
$\beta = 0.25$. Those numbers are not consistent with each other: at
$\alpha = 1$ the potential with $\beta = 0.25$ has a barrier ratio of 10.775,
and the ratio of 11.8 needs $\beta = 0.2565$. Rendering Figure 8(a) at 300 dots
per inch shows the plotted curve matching the $\beta = 0.25$ values exactly, so
the printed beta is the one the figure was drawn from and the mismatch is real
rather than a transcription slip. The other two sample patients agree to two
decimals. `msrelapse.cohort.paper_patients` returns all of it as one table,
with a column for the ratio implied by the durations, a column for the beta
that ratio needs, and a column for the ratio the printed beta actually
delivers. Section 6 of Paper facts has the derivation.

**3.1 against 3.157.** The cohort barrier ratio is printed as about 3.1, while
the printed durations of 100 and 4.3 weeks give 3.1572 through equation (7).
The rounding direction is not uniform elsewhere in the article either: 5.3764
is printed as 5.3 and 2.6396 as 2.7. The package compares against the
arithmetic rather than against the printed rounding, which is why the closing
table names 3.157 in its tolerance text.

**The Figure 4(b) bin count.** The panel states 266 health events, while the
bar heights measured off the rendered figure sum to 265. The first two bins are
the uncertain ones, by about one count. Figure 3 and Figure 4(a) both sum
exactly, to 70 and 218, so this is the one histogram in the article that cannot
be pinned. The digitised heights are in `PAPER` with that caveat attached, and
section 9 of Paper facts records the measurement.

## Substituting the real data

The article released no per patient series. Anyone who obtains them from the
corresponding authors, or who has a registry export of their own, can run the
same pipeline on it: nothing in the analysis knows whether the record it is
handed is synthetic.

If the record is already a weekly series of $+1$ and $-1$ with the columns
`patient_id`, `week` and `state`:

```bash
msrelapse reproduce --data my_cohort_weekly.csv --out reproduction/
```

If it is a registry export with one row per relapse and calendar dates, convert
it first:

```python
import msrelapse as ms

events = ms.read_events(
    "registry_export.csv",
    date_cols=("start", "end", "onset", "recovery"),
)
weekly = ms.events_to_weekly(events)         # applies the weekly rounding rule
ms.write_csv(weekly, "my_cohort_weekly.csv")
print(ms.reproduction_table(weekly))
```

`ms.validate(frame, "weekly")` checks a frame against the schema and explains
the first thing that is wrong with it rather than failing later inside an
analysis. [Data](data.md) documents all three schemas, and the events to weekly
direction is the one lossy conversion, because it applies the rounding rule of
the study.

The table is a cohort measurement and a handful of records does not carry it,
so `ms.reproduction_table` raises rather than returning a thin table when the
record is too small. Three of its rows set that floor. The two goodness of fit
rows need at least two complete, uncensored durations of each state to
bootstrap, and the periodicity row needs at least one patient whose periodogram
the pooled test can read, which asks for 8 weeks of follow up, 3 relapse onsets
and onsets that do not fall in every other week. A record holding no run of one
of the two states is refused before any of that, since the table then has no
mean duration to report for that state. None of this is a statement about how
many patients are enough for the numbers to mean something: seventy records
already leave every mean with a standard error near six percent, which
[Data](data.md) measures.

Two things change when the record is real. The relapsing-remitting phase range
row becomes an exact comparison rather than a range check, because 40 to 1311
weeks is a property of that particular follow up. And the whole table becomes a
statement about the study rather than about a generator, which is the point of
running it.
