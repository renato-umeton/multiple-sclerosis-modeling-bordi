# Data

## The decision

The article released no data. It carries no data availability statement, no
supplementary material and no deposited series, and the per patient records
behind its Figures 2, 3 and 4 were never published. Releasing the de-identified
weekly series is not this package's to decide, so the package ships a
**synthetic twin** instead: seventy simulated records generated from the
numbers the article prints, with the provenance attached everywhere, plus
instructions for asking the authors for the original.

!!! danger "Never present these records, or a plot of them, as clinical data"

    The twin reproduces the aggregates the article reports. It is not the
    cohort of the study, it contains no patient, and no clinical claim can be
    read off it. Every frame the loader returns carries a provenance sentence
    in `frame.attrs['provenance']`, every run of `msrelapse reproduce` prints
    it, and the file it writes records it. If you publish a figure drawn from
    these files, say in the caption that the data are synthetic.

## What is shipped

Three files, inside the installed package at `msrelapse/data/`, holding the
same seventy records in the three schemas below, plus the full provenance note.

| File | Schema | Rows |
|---|---|---|
| `synthetic_bordi2013_weekly.csv` | weekly | 28302, one per patient-week |
| `synthetic_bordi2013_durations.csv` | durations | 522, one per run of a state |
| `synthetic_bordi2013_events.csv` | events | 261, one per relapse |
| `PROVENANCE.txt` | plain text | the note `msrelapse.datasets.provenance()` returns |

They are read with the loader rather than by path, so that they work from an
installed wheel as well as from a checkout:

```python
import msrelapse as ms

weekly = ms.load_synthetic_bordi2013()               # or "durations", or "events"
print(weekly.attrs["provenance"])
print(ms.provenance())                               # the whole note
```

`ms.regenerate_synthetic_bordi2013("regenerated/")` writes the three files again
from the same seed, which is how they were made in the first place. Called with
no directory it writes them back into the data directory of the installed
package, which is how the shipped copies were produced and which needs an
install that can be written to: from a zip archive or a read-only install, name
a directory of your own. Any seed but 20130910 is refused unless a directory is
named, since a cohort under another seed is not the twin that the provenance
note and the closing table describe.

### How they were made

| Choice | Value |
|---|---|
| Generator | `msrelapse.cohort.generate(bordi2013_spec("renewal"), rng=20130910)` |
| Engine | `renewal`, the alternating renewal process, not the stochastic equation |
| Seed | 20130910, which opens with the year of the volume and closes with four arbitrary digits |
| Cohort size | 70 patients, because the study followed 70 |
| Follow up length | One draw per patient from the Figure 3 histogram of relapsing-remitting phase lengths, which spans 40 to 1312 weeks in eight bins of 159 weeks, one week above the printed 1311 week maximum |
| Generative mean remission | 134.21 weeks |
| Generative mean relapse | 4.34415 weeks |
| Start state | Relapse, because the records of the study start at the first relapse |

The two generative means are deliberately not the printed 100 and 4.3 weeks.
The article computed naive means, averaging every recorded run including the
last remission of each record, which the end of follow up cut short. Measured
that way over the Figure 3 windows, a cohort generated at a true mean of 100
weeks reports about 80. `msrelapse.cohort.naive_mean_targets` inverts that
measurement, so the naive means of this twin land on the printed pair: on these
seventy records they measure 104.3 and 4.12 weeks.

### What seventy records can pin down

Seventy records of these lengths hold about 260 relapses and about 260
remissions, so a mean duration measured on them carries a standard error near
six percent, and everything built on those means inherits it. The barrier ratio
of equation (7) moves by about 0.13 under one standard error of the two means,
which is why the closing table compares it under a tolerance of 0.2 rather than
a tighter one. Nothing about the size of the twin is a choice: it is seventy
records because the study followed seventy patients. A reader who wants tighter
comparisons should generate a larger cohort with
`msrelapse.cohort.generate`, not reach for a friendlier seed.

## The three schemas

All three are plain `pandas.DataFrame` objects with fixed columns and fixed
dtypes, validated by `msrelapse.validate(frame, schema)`, which names the first
thing that is wrong rather than failing later inside an analysis.

### weekly

One row per patient-week. This is the shape the article's Figure 2 plots and
the shape every measurement starts from.

| Column | Type | Meaning |
|---|---|---|
| `patient_id` | str | Patient identifier; rows of one patient are contiguous |
| `week` | int64 | Week number, starting at 0 and contiguous within a patient |
| `state` | int64 | $+1$ no health (relapse), $-1$ health (remission) |

### durations

The run length encoding of `weekly`. The two convert losslessly in both
directions, with `weekly_to_durations` and `durations_to_weekly`.

| Column | Type | Meaning |
|---|---|---|
| `patient_id` | str | Patient identifier |
| `run_index` | int64 | Position of the run within the patient, from 0, states alternating |
| `state` | int64 | $+1$ or $-1$, as above |
| `duration_w` | int64 | Length of the run in whole weeks, at least 1 |
| `censored` | bool | True for a final remission cut short by the end of follow up |

The censoring flag is the one piece of information the article did not use: it
averaged censored and complete runs alike. `fit_durations` uses it by default
and takes `censoring=False` to reproduce the article's arithmetic.

### events

The registry shape, one row per relapse, in weeks since the start of that
patient's follow up. A patient with no relapse keeps one row with a missing
onset and end, so that the follow up window survives.

| Column | Type | Meaning |
|---|---|---|
| `patient_id` | str | Patient identifier |
| `followup_start` | float64 | Start of the observation window, in weeks |
| `followup_end` | float64 | End of the observation window, in weeks |
| `relapse_onset` | float64 | Onset of one relapse, or missing |
| `relapse_end` | float64 | End of the same relapse, or missing |

`read_events` also reads a registry export that carries calendar dates, through
its `date_cols` argument, and converts them to weeks since each patient's first
follow up date. `weekly_to_events` and `events_to_weekly` convert in both
directions, but only the first is lossless: going from events to weeks applies
the weekly rounding rule of the study, under which any week a relapse touches
counts as a whole relapse week.

## Asking for the original series

Write to the corresponding authors of the article, Marco Salvetti
(marco.salvetti@uniroma1.it) and Alfonso Sutera
(alfonso.sutera@roma1.infn.it). The series are neither deposited nor
referenced anywhere, so there is no repository to try first, and this package
makes no claim on them. The data were collected before 1993 at the MS Clinic of
Sapienza University of Rome and the article records the ethics position that
applied to them.

A record obtained that way runs through the same pipeline with no change:
[Reproducing the paper](reproducing.md) has the two commands.

## Licence and reuse

The article is open access under the Creative Commons Attribution licence
(CC BY): "Copyright (c) 2013 Isabella Bordi et al. This is an open access
article distributed under the Creative Commons Attribution License, which
permits unrestricted use, distribution, and reproduction in any medium,
provided the original work is properly cited." Its equations, figures and
numbers may therefore be reused in a derived package, with citation, which is
what this package does: every number it takes is held in `PAPER` together with
the sentence it came from, and [Citing](citing.md) has the reference.

The package itself is MIT licensed. The synthetic files carry no separate
licence and no restriction beyond the one above them on this page: they are
synthetic, and they must be described as synthetic.
