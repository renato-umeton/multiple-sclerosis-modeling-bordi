# Disability trajectory (illustrative extension)

!!! warning "This is an extension, not the article"

    The 2013 article models one real number crossing a barrier and the weekly
    sequence of health and no health weeks that the crossings produce, and
    nothing else. It reports no EDSS, no disability score and no clinical
    outcome for any of its 70 patients. This page describes an extension built
    on top of that weekly series so that the bottom panel of the animation and
    one documented function, `msrelapse.edss_trajectory`, can be read. Every
    number below comes from other papers, and every distribution below was
    fitted here rather than there. Nothing here is fit for individual
    prognosis, for trial design, or for any clinical or regulatory use.

## The model

The weekly series is a sequence $s(w)$ of health and no health weeks for
$w = 1, \dots, W$. The maximal runs of no health weeks are the relapse
episodes, episode $i$ occupying weeks $[u_i, v_i]$, and time $t$ is counted in
weeks from clinical onset. Each episode contributes one kernel, which rises to
its peak $A_i$ over the time to nadir $T_n$ and then decays exponentially
towards its own residual $r_i$:

$$
g_i(t) =
\begin{cases}
0, & t < u_i \\[4pt]
A_i \min\!\left(1, \dfrac{t - u_i + 1}{T_n}\right), & u_i \le t < u_i + T_n \\[8pt]
r_i + (A_i - r_i)\, e^{-(t - u_i - T_n)/\tau}, & t \ge u_i + T_n
\end{cases}
$$

The trajectory is the baseline $E_0$, plus an optional linear progression term
at $k$ points a year, plus the sum of the kernels, held inside the scale:

$$
\mathrm{EDSS}(t) = \mathrm{clip}\!\left(E_0 + \frac{k\,t}{365.25/7}
+ \sum_i g_i(t),\; 0,\; 10\right) ,
$$

the denominator being the number of weeks in a year, 365.25 over 7, which is
what a year means everywhere else in the package and is why ten years is week
522 rather than week 520. What a figure shows is that value on the half point
grid the scale is scored on,

$$
\mathrm{EDSS}_{\text{display}}(t)
= \frac{\operatorname{round}\!\left(2\,\mathrm{EDSS}(t)\right)}{2} ,
$$

rounding a score that falls exactly between two steps upwards, so 2.25 is
displayed as 2.5 and 2.24 as 2.0.

As $t$ grows past a relapse, $g_i(t)$ tends to $r_i$, so the expression reduces
to the intended form: a baseline, plus the residuals of every past relapse,
plus the transient bump of the current one, plus the progression term. Note
that $\tau$ is far longer than the mean relapse episode of 4.3 weeks the
article reports, so the trace does not step back down in the week the weekly
series flips to remission.

### Drawing the peak and the residual

For each episode the peak $A_i$ is drawn from the peak law, then the residual
$r_i$ from the residual law of the severity class $A_i$ falls in, then $A_i$ is
raised to $\max(A_i, r_i)$ so that the kernel is monotone after the nadir. That
last step raises about 9 percent of peaks.

| Peak $A$ | 0.0 | 0.5 | 1.0 | 1.5 | 2.0 | 2.5 | 3.0 | 3.5 |
|---|---|---|---|---|---|---|---|---|
| probability | 0.16 | 0.23 | 0.26 | 0.18 | 0.09 | 0.04 | 0.025 | 0.015 |

The peak law has a mean of 1.05 and a median of 1.0, and by construction
$P(A \ge 0.5) = 0.84$, $P(A \ge 1.0) = 0.61$ and $P(A \ge 2.0) = 0.17$, which
are the three verified marginals. Those three describe the law the peak is
drawn from, not the peaks a trace ends up showing: the raise to the residual
lifts the realised figures to 0.88, 0.63 and 0.19, a little above each
published marginal.

| Residual $r$ | -1.0 | -0.5 | 0.0 | +0.5 | +1.0 | +1.5 | +2.0 | +2.5 | +3.0 |
|---|---|---|---|---|---|---|---|---|---|
| mild peak, $A \le 0.5$ | 0.03 | 0.15 | 0.57 | 0.21 | 0.04 | 0 | 0 | 0 | 0 |
| moderate or severe peak, $A \ge 1.0$ | 0.03 | 0.15 | 0.29 | 0.265 | 0.15 | 0.06 | 0.032 | 0.013 | 0.010 |

Marginally, over the 39 percent of relapses that are mild and the 61 percent
that are not, this gives $P(r \ge 0.5) = 0.42$, $P(r \ge 1.0) = 0.18$,
$P(r \ge 2.0) = 0.034$, a median of 0, a mean of 0.256 and an odds ratio of 3.4
for incomplete recovery in severe against mild relapses. The mass below zero,
18 percent of relapses ending at least half a point below the pre-relapse
score, is what lets the distribution satisfy both the published thresholds and
the published mean of 0.27 with a median of 0 at the same time.

### The defaults

```text
E0   = 2.0 EDSS points       baseline after the first attack
T_n  = 1 week                time to nadir
tau  = 10 weeks              recovery decay constant
k    = 0.0 EDSS per year     independent progression, off by default
clip = 0 to 10, displayed on the half point grid
```

## The parameters and their sources

Every row is a published quantity, apart from the four marked as derived, which
this model constructs from the rows above them. The citation in the last column
is the short form of an entry of the [references](#references) at the foot of
the page.

| Parameter | Central value | Range | Unit | Sources |
|---|---|---|---|---|
| Baseline EDSS after the first attack | 2.0 | 1.0 to 2.5 (interquartile range) | EDSS points | Tur 2025 |
| Mean EDSS increase at relapse nadir, untreated | 1.10 | 0.77 actively treated to 1.45 in a high disability clinic cohort | EDSS points | Lublin 2014; Hirst 2008 |
| Nadir increase of at least 0.5, 1.0 and 2.0 | 84, 61 and 17 percent | 0.5 threshold, 71 percent treated to 84 percent untreated; severe share 14 to 39 percent by grading scheme | proportion of relapses | Lublin 2014; Achiron 2019; Koch 2023; Naldi 2011 |
| Peak deficit law of the model (derived) | the half point grid above, mean 1.05, median 1.0 | pinned at the three verified marginals, free between them | EDSS points | from Lublin 2014 and Achiron 2019 |
| Time from relapse onset to nadir | 1 to 2 days | same day to about 3 days, longer for severe relapses, with a tail to about 2 weeks | days | Hosny 2023; Naldi 2011 |
| Relapse episode duration, the run of no health weeks | 4.3 weeks | 1 to about 24 weeks, exponentially distributed | weeks | Bordi 2013; Hosny 2023; Naldi 2011 |
| Median time to EDSS recovery after a relapse | 71 to 111 days | 71 (95 percent CI 66 to 75) to 111 (95 percent CI 99 to 138) | days | Koch 2023; Mostert 2025 |
| Relapses recovered by 1, 3, 6 and 12 months | 4, 38, 67 and 79 percent, 84 percent at any time | 52 and 55 percent if 12 or 24 weeks of confirmed recovery are required | proportion of relapses | Koch 2023; Iuliano 2008 |
| Recovery decay constant tau of the model (derived) | 10 weeks | 7 to 14 | weeks | from Koch 2023 and Mostert 2025 |
| Residual of at least 0.5 at 6 to 12 months | 0.42 | 0.31 to 0.54 (95 percent CI of the pooled estimate) | proportion of relapses | Ladeira 2025; Lublin 2003; Lublin 2014 |
| Residual of at least 1.0 at 6 to 12 months | 0.18 | 0.16 to 0.28 | proportion of relapses | Lublin 2003; Achiron 2019 |
| Mean net residual per relapse | 0.25 | 0.20 to 0.30, rising to 0.50 in a high disability cohort | EDSS points | Lublin 2003; Confavreux 2006 (Clinical Neurology and Neurosurgery); Lublin 2014; Hirst 2008; Stewart 2017 |
| Residual step law of the model (derived) | the two severity classes above, mean 0.256, median 0 | implies at least 0.5 in 0.42, at least 1.0 in 0.18 and at least 2.0 in 0.034 of relapses | EDSS points | from Ladeira 2025, Lublin 2003 and Achiron 2019 |
| Dependence of the residual on relapse severity | odds ratio about 3.4 for incomplete recovery, severe against mild | 2.4 to 17.2 across studies | odds ratio | Ladeira 2025; Koch 2023; Leone 2008 |
| Dependence of the residual on prior EDSS and on age | direction well supported, magnitude poorly quantified; odds ratio 2.9 for age 30 or more at relapse onset | 95 percent CI 1.5 to 5.7 for the age effect; no usable coefficient published for the EDSS effect | odds ratio | Lublin 2022; Kalincik 2014; Conway 2019; Leone 2008; Sotiropoulos 2021; Hirst 2008 |
| Dependence of the residual on relapse duration | odds ratio 3.2 for sequelae, long or intermediate against short | 95 percent CI 1.5 to 6.9 | odds ratio | Leone 2008; Hosny 2023; Naldi 2011 |
| Annualised relapse rate, untreated relapsing onset | 0.65 over the whole relapsing phase | 0.93 in the first 2 years, 0.41 from year 3 to progression onset, 0.015 at very long follow up | relapses per year | Scalfari 2010; Leray 2010; Skoog 2012; Bordi 2013 |
| Onset to an irreversible EDSS or DSS 3 | 10 years | 10.0 (95 percent CI 9.4 to 10.6) in Rennes, 10 in London Ontario | years | Leray 2010; Scalfari 2010 |
| Onset to an irreversible DSS 4 | 11.4 years | relapsing onset only; 0.0 for progressive onset | years | Confavreux 2000 |
| Onset to an EDSS or DSS 6 | about 20 years | 15 to 28 | years | Weinshenker 1989; Scalfari 2010; Leray 2010; Confavreux 2000; Tedeholm 2015; Tremlett 2006 |
| Age at an irreversible DSS 4, 6 and 7 | 44.3, 54.7 and 63.1 years | 95 percent CI 43.3 to 45.2, 53.5 to 55.8, 61.0 to 65.1 | years of age | Confavreux 2006 (Brain) |
| EDSS at 10 years from onset | about 3 | mean change of 1 point per 10 years; 20 percent worsen by 2 or more points | EDSS points | Pittock 2004 (10-year follow-up) |
| EDSS at 20 years from onset | 58 percent still relapsing remitting, of whom 39 percent at EDSS 3 or below; median about 3.25 at 14 years | EDSS 0 to 10; 31 percent at EDSS 6 or more at 14 years | EDSS points | Brex 2002; Fisniku 2008; Pittock 2004 (prevalence cohort) |
| Onset to secondary progression | 15 years | 15 to 16 | years | Scalfari 2010; Tedeholm 2015; Leray 2010; Bordi 2013 |
| Duration of the DSS 3 to DSS 6 phase | 7.4 years for relapsing onset, 7.0 for the whole cohort | 95 percent CI 6.8 to 8.0; 6.5 to 9.2 across every stratum of the first phase | years | Leray 2010; Confavreux 2000; Confavreux 2003 |
| Prognostic weight of relapses by time since onset | confined to roughly the first 2 years | the hazard contribution of a relapse to reaching EDSS 6 falls from 48 percent within 5 years of onset to 10 percent beyond 10 | hazard increase | Scalfari 2010; Tremlett 2009; Leray 2010; Confavreux 2000 |
| Relapse associated share of confirmed worsening | 27 percent | 18 to 33 percent measured; 35 to 50 percent untreated, which is an extrapolation | proportion of events | Lublin 2022; Kappos 2020; Muller 2023; Portaccio 2022 |
| Independent progression rate, the PIRA term of the model (derived) | 0.0 EDSS points per year by default | 0 to 0.05 | EDSS points per year | from Pittock 2004 (10-year follow-up), Leray 2010, Confavreux 2000 and Muller 2023 |
| Frequency of a catastrophic non-recovering relapse | 7 of 1078 patients, 0.6 percent, and about 0.3 percent of relapses | no clinical factor identified the patients at risk | proportion | Bejaoui 2010 |
| Annual probability of confirmed worsening in a placebo arm | 16.9 percent in year 1, 13.1 percent in year 2 | 16.0 and 11.2 percent in trials published after 2000; the odds fell 31 percent per decade of publication year | proportion per year | Rover 2015 |
| Long run cost of one incompletely recovered relapse | plus 0.6 EDSS at 10 years | also plus 0.5 seconds on the timed 25 foot walk | EDSS points | Sotiropoulos 2021 |
| Effect of poor early relapse recovery | the progressive phase begins at 8.3 years against 30.2 for good recoverers | p=0.001; 80.1 percent of a population based cohort were good recoverers and 15.7 percent poor | years from onset | Novotna 2015 |
| Effect of corticosteroids on the recovery curve | the curve moves left, not down | disability significantly lower at 1 and 4 weeks; no difference in visual acuity at 6 months | qualitative | Beck 1992; Beck 1994; Milligan 1987; Koch 2023 |

Four cautions on reading the table. The recovery times are the weakest
transport in it, because both cohorts are fully treated and no untreated
equivalent exists. The recovery proportions are re-denominated over all 240
CombiRx relapses rather than over the 202 patients who ever recovered, which is
the denominator the paper's own headline figures of 4, 45, 80 and 94 percent
use. The odds ratio of 17.2 at the top of the severity range should never be
used as a point estimate, since its reference stratum is small and its interval
spans two orders of magnitude. And the registry regression coefficient of about
0.16 to 0.42 EDSS points per relapse is a slope on relapse incidence rather
than a per-event step, so it must not be read as one.

## What rests on evidence and what is a judgement call

Well supported: the baseline of 2.0, the median first recorded EDSS in an
inception cohort of 1074 patients; the peak law, which reproduces three
verified marginals from the AFFIRM placebo arm and from 1672 graded relapses;
the residual probability of 0.42, the only meta-analytic estimate available and
drawn from 27,672 relapses; the half point grid itself; the dependence of the
residual on relapse severity; and a time to nadir of one week, since the
published median is 1 to 2 days and the series is weekly.

Judgement calls: $\tau = 10$ weeks, obtained by requiring that a typical
1.0 point excess falls below the 0.25 point display threshold at the published
median recovery time, which gives 7.3 weeks against the DECIDE median of 71
days and 11.4 weeks against the CombiRx median of 111 days, both from treated
cohorts; the split of the residual mass above 1.0, which is pinned only at the
2.0 threshold; the 18 percent of mass below zero, whose size is inferred from
the arithmetic rather than measured anywhere; two severity classes rather than
three; and $k = 0$, chosen so that the trace is driven by relapses alone, which
is the point of the exercise, and because adding a progression term on top of a
residual law already calibrated to the whole observed slope would count the
same accrual twice.

## What the first relapse looks like in the trace

The patient starts at EDSS 2.0. The first relapse draws a peak with a mean of
1.12 and a median of 1.0, so the modal trace rises from 2.0 to 3.0 inside the
first week of the episode, holds near the nadir only briefly, and then decays
with a ten week constant, losing about 86 percent of the recoverable excess by
week 20 and about 92 percent by week 26. The residual decides where it settles:
40 percent of first relapses leave nothing at all and the trace returns exactly
to 2.0, 18 percent end at 1.5 or lower, 42 percent leave 0.5 or more and 18
percent leave 1.0 or more. The expected sustained level after the first relapse
is 2.25.

The honest picture of a first attack is therefore a spike of about one EDSS
point that is visually gone within six months, with a slightly better than even
chance of leaving no permanent mark. A trace whose first relapse leaves a
visible permanent step every time is wrong.

## The long run anchors

Simulating 20,000 patients over 30 years by alternating renewal at the printed
means of 100 and 4.3 weeks, on a four week grid with $k = 0$, and reading the
sustained level with every transient bump decayed, which is the analogue of the
irreversible score the natural history cohorts measure:

| Quantity | This model | Published anchor |
|---|---|---|
| Median sustained EDSS at 10 years | 3.0 | about 3, from a mean change of 1 point per decade |
| Median sustained EDSS at 14 years | 3.5 | 3.25 |
| Median sustained EDSS at 20 years | 4.5 | wide; 39 percent at EDSS 3 or below |
| Median years to a sustained EDSS 3 | about 7 | 10.0 |
| Median years to a sustained EDSS 4 | about 14 | 11.4 |
| Median years to a sustained EDSS 6 | about 29 | 15 to 28 |
| Relapses per year | 0.50 | 0.65 to 0.93 |

With $k = 0.05$ the picture inverts: a median sustained EDSS 4 at about 11
years against a published 11.4 and EDSS 6 at about 22.5 years against a
published 21.7 to 23.1, but the 10 year level rises to 3.5 and overshoots the
mean change data. The two families of anchors cannot both be matched, because
the real curve accelerates, at about 0.1 EDSS points per year over the first
stage and about 0.4 over the DSS 3 to DSS 6 stage, while a model in which every
relapse contributes an independent residual is linear. The acceleration comes
from the secondary progressive transition, which begins at a median of 15 years
and is not in the weekly series at all.

## Caveats

- **This is an illustrative extension and is not part of the 2013 article.**
  That article models one real number crossing a barrier and the weekly
  sequence of health and no health weeks the crossings produce, and nothing
  else. It reports no EDSS, no disability score and no clinical outcome for any
  of its 70 patients. Every number here comes from other papers and every
  distribution here is fitted by this page rather than by the article, which is
  why the extension is labelled that way on this page, in the docstring of
  `msrelapse.edss` and in the caption of the animation.
- **The model is valid only over the relapsing phase**, which is roughly the
  first 15 years from onset. The article's own period of interest runs from the
  first relapse at onset to the last relapse before the shift to secondary
  progressive MS, and secondary progression begins at a median of 15 years in
  both London Ontario and Gothenburg. Run past that window the trace undershoots
  the natural history medians for EDSS 6, by design, because the mechanism that
  takes patients there is not in the weekly series.
- **The model cannot reproduce the accelerating shape of the real curve.**
  Observed accrual is about 0.1 EDSS points per year over the first stage and
  about 0.4 over the DSS 3 to DSS 6 stage, and the second stage lasts a near
  constant 7.0 to 7.4 years however fast the first stage was, confirmed
  separately in 900 untreated patients. A sum of independent per relapse
  residuals is linear and has no mechanism for that amnesia.
- **The model contradicts the untreated cohorts on when relapses matter.** Its
  residual law is per event and does not decay with disease duration, whereas
  London Ontario found that neither the total number of relapsing phase attacks
  nor attacks after year 2 had any deleterious effect on time to DSS 6, 8 or
  10, and British Columbia found the hazard contribution of a relapse falling
  from 48 percent within 5 years of onset to 10 percent beyond 10 years.
  Pulling the other way, the largest trial pool and the largest registry both
  find the residual left by a single relapse rising with age and with
  pre-existing EDSS. The two findings are reconcilable only by arguing that
  late relapses land on a trajectory already dominated by progression, and this
  model implements neither adjustment.
- **Almost every recovery and residual number comes from the treatment era.**
  The two untreated anchors are the pooled placebo arms of pre-DMT trials and
  the AFFIRM placebo arm, and AFFIRM placebo patients are of the 2000s, not of
  the years before 1993. Placebo arms have themselves become milder: the odds
  of confirmed progression in a placebo arm fell 31 percent per decade of trial
  publication year across 39 trials. The recovery time course, the only source
  for the decay constant, has no untreated equivalent at all and comes from two
  fully treated trial cohorts.
- **The cohort behind the article is free of disease modifying therapy rather
  than wholly untreated.** Its patients received short corticosteroid courses
  during relapses when deemed necessary. The randomised evidence says steroids
  move the recovery curve left without lowering its asymptote, which is why the
  residual law is left unchanged, but that is an assumption carried over from
  an optic neuritis trial with a non-EDSS endpoint and from two underpowered
  within-trial comparisons confounded by indication.
- **A single trace is one draw, never a median and never a prognosis.** The
  published spread at a fixed duration is enormous: EDSS 0 to 10 at 14 years in
  the London clinically isolated syndrome cohort, 39 percent still at EDSS 3 or
  below at 20 years, only 25 percent of Olmsted County relapsing remitting
  patients at EDSS 3 or more at 20 years, and 14 percent of the Gothenburg
  untreated relapsing onset cohort never entering a progressive phase at 50
  years. The model reproduces a wide spread, which is a feature, and any figure
  showing one patient has to say so.
- **EDSS is ordinal, not interval.** Adding residuals as though the scale were
  linear is an approximation, and the scale has a well known plateau around 4
  to 5.5 where the same nominal increment means something different. The
  inter-rater variability of the EDSS is of the order of the half point steps
  being modelled, which is part of why the published standard deviation of the
  residual, 1.04, is far larger than any distribution here.
- **The published mean residual of about 0.27 and the published thresholds of
  42 and 28 percent cannot both be satisfied by a non-negative residual.** The
  reconciliation used here, giving 18 percent of relapses a residual below
  zero, matches the reported median of 0, but its size is inferred from
  arithmetic rather than measured anywhere.
- **The model carries no relapse phenotype, no MRI, no sex, no functional
  systems and no treatment effect**, and it cannot separate relapse associated
  worsening from progression independent of relapse activity in a way any
  cohort has measured. With the default progression rate of zero its relapse
  associated share is 100 percent, against a measured 18 to 33 percent in the
  treatment era and a plausible but unmeasured 35 to 50 percent untreated.
- **Several citation limitations are worth recording.** The sample size of 224
  for the pooled placebo arms is not stated in that paper's own abstract and is
  taken from the CombiRx authors' description of it. The published abstract of
  the severe relapse residual paper is internally inconsistent, saying 1, 4 and
  12 months in its methods and 1, 2 and 12 months in its results, and the full
  text could not be opened, so its middle time point should be read as 2 to 4
  months. The day cut-offs for short, intermediate and long relapses are not
  readable in the paywalled paper that reports the odds ratio of 3.2 and come
  from the same group's companion paper. The widely quoted Lyon whole cohort
  milestones of 8.4, 20.1 and 29.9 years could not be verified from any
  reachable full text and are deliberately not used, the relapsing onset values
  of 11.4, 23.1 and 33.1 years being used instead; the MSBase EDSS by duration
  figures come from a conference abstract in a journal that assigns no DOI and
  are not used either; and the DECIDE analysis is cited at its verified online
  publication of October 2025.
- **Nothing here is fit for individual prognosis, for trial design, or for any
  clinical or regulatory use.** It is a teaching illustration, built to make one
  animation and one documented function legible.

## Using the function

`msrelapse.edss_trajectory` takes a weekly record of one patient, in the weekly
schema of [`msrelapse.io`](api/io.md) or as a plain array of states, and
returns the week, the continuous EDSS, the displayed half point EDSS and the
index of the episode in progress. It draws a peak and a residual for every
episode, so it takes a seed or a generator like every other random operation in
the package, and the same seed gives the same trace.

```python
import msrelapse as ms

weekly = ms.load_synthetic_bordi2013()               # 70 synthetic records
ten_years = round(10 * ms.stats.WEEKS_PER_YEAR)      # 522 weeks, as summarise reads it
weeks = weekly["patient_id"].value_counts()
chosen = weeks[weeks >= ten_years].index.min()       # one record past ten years
one = weekly[weekly["patient_id"] == chosen]
trajectory = ms.edss_trajectory(one, rng=5)          # illustrative, not clinical
draws = ms.episode_draws(one, rng=5)                 # the same draws, same seed
print(draws)
print(ms.summarise(trajectory, draws))
```

That record holds six relapses over about fifteen years. The draws say what the
trace does: a spike at every attack, a return towards the level it started
from, and a step left behind only where a residual was drawn. Another seed
gives another patient's story out of the same record, because a trace is one
draw from the laws above, never a median and never a prognosis.

The rest of the module is small. `msrelapse.episode_draws` gives one row per
episode, with its start, its end, its peak and its residual, which is how a
reader checks a trace against the laws above, and `msrelapse.summarise` reads a
finished trace and those draws together. `msrelapse.EDSSSpec` holds the
parameters, so a reader who wants a different decay constant or a non-zero
progression term changes one field and passes it in.
`msrelapse.expected_peak` and `msrelapse.expected_residual` return the means of
the two laws of such a specification, and `msrelapse.EVIDENCE` carries the
published quantities the defaults rest on, each with its unit, its source and
its DOI, so that the numbers in the code and the rows of the table above are
the same numbers. [The API page](api/edss.md) documents every one of them.

## References

Every DOI below was resolved against Crossref, and the load bearing counts
were read from the open access full texts where they exist.

Achiron A, Sarova-Pinhas I, Magalashvili D, Stern Y, Gal A, Dolev M, Menascu
S, Harari G, Gurevich M. Residual disability after severe relapse in people
with multiple sclerosis treated with disease-modifying therapy. Multiple
Sclerosis Journal 2019;25(13):1746-1753. doi:10.1177/1352458518809903

Beck RW, Cleary PA, Anderson MM Jr, et al. A randomized, controlled trial of
corticosteroids in the treatment of acute optic neuritis. New England Journal
of Medicine 1992;326(9):581-588. doi:10.1056/NEJM199202273260901

Beck RW, Cleary PA, Backlund JC. The course of visual recovery after optic
neuritis. Experience of the Optic Neuritis Treatment Trial. Ophthalmology
1994;101(11):1771-1778. doi:10.1016/s0161-6420(94)31103-1

Bejaoui K, Rolak LA. What is the risk of permanent disability from a multiple
sclerosis relapse? Neurology 2010;74(11):900-902.
doi:10.1212/wnl.0b013e3181d55ee9

Bordi I, Umeton R, Ricigliano VAG, Annibali V, Mechelli R, Ristori G, Grassi
F, Salvetti M, Sutera A. A mechanistic, stochastic model helps understand
multiple sclerosis course and pathogenesis. International Journal of Genomics
2013;2013:910321. doi:10.1155/2013/910321

Brex PA, Ciccarelli O, O'Riordan JI, Sailer M, Thompson AJ, Miller DH. A
longitudinal study of abnormalities on MRI and disability from multiple
sclerosis. New England Journal of Medicine 2002;346(3):158-164.
doi:10.1056/NEJMoa011341

Confavreux C, Vukusic S, Moreau T, Adeleine P. Relapses and progression of
disability in multiple sclerosis. New England Journal of Medicine
2000;343(20):1430-1438. doi:10.1056/NEJM200011163432001

Confavreux C, Vukusic S, Adeleine P. Early clinical predictors and progression
of irreversible disability in multiple sclerosis: an amnesic process. Brain
2003;126(4):770-782. doi:10.1093/brain/awg081

Confavreux C, Vukusic S. Age at disability milestones in multiple sclerosis.
Brain 2006;129(3):595-605. doi:10.1093/brain/awh714

Confavreux C, Vukusic S. Accumulation of irreversible disability in multiple
sclerosis: from epidemiology to treatment. Clinical Neurology and Neurosurgery
2006;108(3):327-332. doi:10.1016/j.clineuro.2005.11.018

Conway BL, Zeydan B, Uygunoglu U, Novotna M, Siva A, Pittock SJ, Atkinson EJ,
Rodriguez M, Kantarci OH. Age is a critical determinant in recovery from
multiple sclerosis relapses. Multiple Sclerosis Journal 2019;25(13):1754-1763.
doi:10.1177/1352458518800815

Cree BAC, Hollenbach JA, Bove R, et al (University of California San Francisco
MS-EPIC Team). Silent progression in disease activity-free relapsing multiple
sclerosis. Annals of Neurology 2019;85(5):653-666. doi:10.1002/ana.25463

Fisniku LK, Brex PA, Altmann DR, Miszkiel KA, Benton CE, Lanyon R, Thompson
AJ, Miller DH. Disability and T2 MRI lesions: a 20-year follow-up of patients
with relapse onset of multiple sclerosis. Brain 2008;131(3):808-817.
doi:10.1093/brain/awm329

Hauser SL, Bar-Or A, Comi G, et al. Ocrelizumab versus interferon beta-1a in
relapsing multiple sclerosis. New England Journal of Medicine
2017;376(3):221-234. doi:10.1056/NEJMoa1601277

Hirst C, Ingram G, Pearson O, Pickersgill T, Scolding N, Robertson N.
Contribution of relapses to disability in multiple sclerosis. Journal of
Neurology 2008;255(2):280-287. doi:10.1007/s00415-008-0743-8

Hosny HS, Shehata HS, Ahmed S, Ramadan I, Abdo SS, Fouad AM. Predictors of
severity and outcome of multiple sclerosis relapses. BMC Neurology
2023;23(1):67. doi:10.1186/s12883-023-03109-6

Iuliano G, Napoletano R, Esposito A. Multiple sclerosis: relapses and timing
of remissions. European Neurology 2008;59(1-2):44-48. doi:10.1159/000109260

Jokubaitis VG, Spelman T, Kalincik T, et al. Predictors of long-term
disability accrual in relapse-onset multiple sclerosis. Annals of Neurology
2016;80(1):89-100. doi:10.1002/ana.24682

Kalincik T, Buzzard K, Jokubaitis V, et al (MSBase Study Group). Risk of
relapse phenotype recurrence in multiple sclerosis. Multiple Sclerosis Journal
2014;20(11):1511-1522. doi:10.1177/1352458514528762

Kalincik T. Multiple sclerosis relapses: epidemiology, outcomes and
management. A systematic review. Neuroepidemiology 2015;44(4):199-214.
doi:10.1159/000382130

Kappos L, Wolinsky JS, Giovannoni G, et al. Contribution of relapse-
independent progression vs relapse-associated worsening to overall confirmed
disability accumulation in typical relapsing multiple sclerosis in a pooled
analysis of 2 randomized clinical trials. JAMA Neurology 2020;77(9):1132-1140.
doi:10.1001/jamaneurol.2020.1568

Koch MW, Moral E, Brieva L, et al. Relapse recovery in relapsing-remitting
multiple sclerosis: an analysis of the CombiRx dataset. Multiple Sclerosis
Journal 2023;29(14):1776-1785. doi:10.1177/13524585231202320

Ladeira F, Soares M, Faustino P, Leal Rato M, Gomes I, Caetano A, Taipa R, Sa
MJ. Multiple sclerosis relapse incomplete recovery and associated factors: a
systematic review and meta-analysis. Multiple Sclerosis and Related Disorders
2025;100:106507. doi:10.1016/j.msard.2025.106507

Leone MA, Bonissoni S, Collimedaglia L, Tesser F, Calzoni S, Stecco A, Naldi
P, Monaco F. Factors predicting incomplete recovery from relapses in multiple
sclerosis: a prospective study. Multiple Sclerosis Journal 2008;14(4):485-493.
doi:10.1177/1352458507084650

Leray E, Yaouanq J, Le Page E, Coustans M, Laplaud D, Oger J, Edan G. Evidence
for a two-stage disability progression in multiple sclerosis. Brain
2010;133(7):1900-1913. doi:10.1093/brain/awq076

Lublin FD, Baier M, Cutter G. Effect of relapses on development of residual
deficit in multiple sclerosis. Neurology 2003;61(11):1528-1532.
doi:10.1212/01.wnl.0000096175.39831.21

Lublin FD, Cutter G, Giovannoni G, Pace A, Campbell NR, Belachew S.
Natalizumab reduces relapse clinical severity and improves relapse recovery in
MS. Multiple Sclerosis and Related Disorders 2014;3(6):705-711.
doi:10.1016/j.msard.2014.08.005

Lublin FD, Haring DA, Ganjgahi H, et al. How patients with multiple sclerosis
acquire disability. Brain 2022;145(9):3147-3161. doi:10.1093/brain/awac016

Milligan NM, Newcombe R, Compston DA. A double-blind controlled trial of high
dose methylprednisolone in patients with multiple sclerosis: 1. Clinical
effects. Journal of Neurology, Neurosurgery and Psychiatry 1987;50(5):511-516.
doi:10.1136/jnnp.50.5.511

Mostert J, Strijbis EMM, D'Haeseleer M, Moral E, Brieva L, Comtois J, Repovic
P, Bowen JD, Cutter G, Koch MW. Extended window of relapse recovery in RRMS:
an analysis of the DECIDE dataset. Journal of Neurology, Neurosurgery and
Psychiatry, published online 9 October 2025. doi:10.1136/jnnp-2025-336660

Muller J, Cagol A, Lorscheider J, et al. Harmonizing definitions for
progression independent of relapse activity in multiple sclerosis: a
systematic review. JAMA Neurology 2023;80(11):1232-1245.
doi:10.1001/jamaneurol.2023.3331

Naldi P, Collimedaglia L, Vecchio D, Rosso MG, Perl F, Stecco A, Monaco F,
Leone MA. Predictors of attack severity and duration in multiple sclerosis: a
prospective study. The Open Neurology Journal 2011;5:75-82.
doi:10.2174/1874205X01105010075

Novotna M, Paz Soldan MM, Abou Zeid N, et al. Poor early relapse recovery
affects onset of progressive disease course in multiple sclerosis. Neurology
2015;85(8):722-729. doi:10.1212/WNL.0000000000001856

Pittock SJ, Mayr WT, McClelland RL, Jorgensen NW, Weigand SD, Noseworthy JH,
Weinshenker BG, Rodriguez M. Change in MS-related disability in a population-
based cohort: a 10-year follow-up study. Neurology 2004;62(1):51-59.
doi:10.1212/01.wnl.0000101724.93433.00

Pittock SJ, Mayr WT, McClelland RL, Jorgensen NW, Weigand SD, Noseworthy JH,
Rodriguez M. Disability profile of MS did not change over 10 years in a
population-based prevalence cohort. Neurology 2004;62(4):601-606.
doi:10.1212/wnl.62.4.601

Portaccio E, Bellinvia A, Fonderico M, et al. Progression is independent of
relapse activity in early multiple sclerosis: a real-life cohort study. Brain
2022;145(8):2796-2805. doi:10.1093/brain/awac111

Rover C, Nicholas R, Straube S, Friede T. Changing EDSS progression in placebo
cohorts in relapsing MS: a systematic review and meta-regression. PLoS One
2015;10(9):e0137052. doi:10.1371/journal.pone.0137052

Scalfari A, Neuhaus A, Degenhardt A, Rice GP, Muraro PA, Daumer M, Ebers GC.
The natural history of multiple sclerosis, a geographically based study 10:
relapses and long-term disability. Brain 2010;133(7):1914-1929.
doi:10.1093/brain/awq118

Skoog B, Runmarker B, Winblad S, Ekholm S, Andersen O. A representative cohort
of patients with non-progressive multiple sclerosis at the age of normal life
expectancy. Brain 2012;135(3):900-911. doi:10.1093/brain/awr336

Sotiropoulos MG, Lokhande H, Healy BC, Polgar-Turcsanyi M, Glanz BI, Bakshi R,
Weiner HL, Chitnis T. Relapse recovery in multiple sclerosis: effect of
treatment and contribution to long-term disability. Multiple Sclerosis Journal
Experimental Translational and Clinical 2021;7(2):20552173211015503.
doi:10.1177/20552173211015503

Stewart T, Spelman T, Havrdova E, et al (MSBase Study Group). Contribution of
different relapse phenotypes to disability in multiple sclerosis. Multiple
Sclerosis Journal 2017;23(2):266-276. doi:10.1177/1352458516643392

Tedeholm H, Skoog B, Lisovskaja V, Runmarker B, Nerman O, Andersen O. The
outcome spectrum of multiple sclerosis: disability, mortality, and a cluster
of predictors from onset. Journal of Neurology 2015;262(5):1148-1163.
doi:10.1007/s00415-015-7674-y

Tintore M, Rovira A, Rio J, et al. Baseline MRI predicts future attacks and
disability in clinically isolated syndromes. Neurology 2006;67(6):968-972.
doi:10.1212/01.wnl.0000237354.10144.ec

Tremlett H, Paty D, Devonshire V. Disability progression in multiple sclerosis
is slower than previously reported. Neurology 2006;66(2):172-177.
doi:10.1212/01.wnl.0000194259.90286.fe

Tremlett H, Yousefi M, Devonshire V, Rieckmann P, Zhao Y (UBC Neurologists).
Impact of multiple sclerosis relapses on progression diminishes with time.
Neurology 2009;73(20):1616-1623. doi:10.1212/WNL.0b013e3181c1e44f

Tur C, et al. The Barcelona baseline risk score to predict long-term prognosis
after a first demyelinating event: a prospective observational study. Lancet
Regional Health Europe 2025;53:101302. doi:10.1016/j.lanepe.2025.101302

Weinshenker BG, Bass B, Rice GP, Noseworthy J, Carriere W, Baskerville J,
Ebers GC. The natural history of multiple sclerosis: a geographically based
study. I. Clinical course and disability. Brain 1989;112(1):133-146.
doi:10.1093/brain/112.1.133
