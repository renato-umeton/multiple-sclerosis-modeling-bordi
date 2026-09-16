<!--
Draft of a submission to the Journal of Open Source Software. Submission is
planned for after v1.0.0, once the release is archived and the archive
identifier exists.

The block below is the JOSS header. It is kept inside a fenced code block here
so that this file stays ordinary Markdown; at submission time it becomes the
YAML front matter of paper.md, delimited the way JOSS asks for.
-->

```yaml
title: "msrelapse: a reference implementation of the Bordi et al. 2013 stochastic double well model of relapsing-remitting multiple sclerosis"
tags:
  - Python
  - multiple sclerosis
  - stochastic differential equations
  - double well potential
  - first passage time
  - reproducible research
authors:
  - name: Renato Umeton
    # orcid: added by the maintainer before submission
    affiliation: 1
affiliations:
  # Completed by the maintainer before submission.
  - name: Affiliation of the maintainer
    index: 1
date: 15 September 2026
bibliography: paper.bib
```

# Summary

Relapsing-remitting multiple sclerosis alternates between two clinical states,
relapse and remission, and the times spent in each look random. `msrelapse` is
a Python package that implements the mechanistic and stochastic model of that
alternation published by @bordi2013: a particle in an asymmetric double well
potential, pushed over the barrier between the wells by noise. The deep well is
health and the shallow one is relapse, so a patient spends long spells in
remission and short spells in relapse, and the height of each barrier sets how
long each spell lasts. The package provides the potential and its critical
points, the stochastic equation of motion, three readings of the time needed to
cross a barrier, maximum likelihood fits of the observed durations, the relapse
rate statistics a trial reports, and a generator of virtual cohorts. Every
number the article reports is held in one module beside the sentence it comes
from, and a single command reproduces the published aggregates and prints a
table that says, row by row, how close the reproduction came.

# Statement of need

The 2013 article is a compact and testable theory of relapse timing, but it was
published without code, without data and without a single fitted number: its
claims that the two duration distributions are exponential and carry no
periodicity rest on the shape of two histograms. Anyone who wants to build on
it has to re-derive the potential from three printed equations, guess an
integration scheme, and decide what to do about the several places where the
article is silent or inconsistent with itself. That is enough work to stop most
readers, and it is work that has to be repeated by each of them.

`msrelapse` does it once, in the open. It fixes the sign convention against the
published figures, states the consequences the article leaves implicit, and
documents the one place where the article disagrees with its own arithmetic, the
asymmetry parameter printed for one of its three sample patients. It is also
explicit that the exit time formula of @benzi1983 used in the article drops the
prefactor of the Kramers rate [@kramers1940], so that the barrier ratio computed
from observed durations is a documented heuristic rather than a definition. The
package therefore serves three audiences: modellers who want the dynamics, trial
statisticians who want the duration and rate estimators on a record of their
own, and groups building in silico cohorts who want relapse series that behave
the way the published ones do.

# The model

The potential is the quartic of the article, with a control parameter on the
quartic term and an asymmetry term linear in the state variable:

$$V(x) = -\frac{1}{2}x^{2} + \frac{\alpha}{4}x^{4} + \beta x$$

Its negative gradient drives the state, which is also pushed by a Wiener
process of variance $\epsilon$:

$$dx = \left[x\left(1 - \alpha x^{2}\right) - \beta\right]dt + \sqrt{\epsilon}\,dw$$

With $\beta > 0$ the left well, health, is deeper than the right well, relapse.
Writing $\Delta V_1$ and $\Delta V_2$ for the two barrier heights measured from
the saddle between the wells, and $\tau_{1}$ and $\tau_{2}$ for the mean times
spent in each state, the article estimates their ratio from the durations alone:

$$\frac{\Delta V_{1}}{\Delta V_{2}} \approx \frac{\log \tau_{1}}{\log \tau_{2}}$$

Because the base of the logarithm cancels, this is one number, and on the
published cohort means it comes to about 3.16, which the article prints as 3.1.
The package computes it, and also computes the exact mean first passage time of
the same equation, so that a reader can see how far the two readings are apart.
The asymptotic exponentiality of those passage times [@day1983] is why the
observed durations can be exponential at all.

# Functionality

`DoubleWell` holds a parameter pair and returns its critical points, its two
barriers and their ratio, and refuses to report wells for a parameter pair that
has lost bistability. `simulate_paths` and `simulate_weekly` integrate the equation of
motion, with a Brownian bridge correction on the level crossings and an optional
compiled kernel. `exit_times`, `mfpt` and `kramers_time` give the crossing time
by simulation, by quadrature and by the Kramers estimate, and `calibrate` inverts
a pair of observed mean durations into a parameter pair.

On the data side, `fit_durations` fits an exponential or geometric duration law
by maximum likelihood, with or without the censoring that the end of follow up
imposes on the last spell of each record, and `test_memoryless` and
`test_periodicity` put the two qualitative claims of the article to a test.
`arr` and `compare_arr` compute annualised relapse rates and rate ratios with
Poisson, negative binomial and bootstrap intervals, following the count
regression practice of relapse and exacerbation trials [@keene2007], and
`sample_size_arr` implements the negative binomial sample size of @zhu2014, so
that a reader can see what a cohort of the size of the published one can
resolve. `CohortSpec` and `generate` build virtual cohorts of any size, and a
command line exposes the reproduction, the cohort generator, the fits and the
tests without any Python.

# Example usage

```python
import msrelapse

weekly = msrelapse.load_synthetic_bordi2013()  # 70 synthetic records
runs = msrelapse.weekly_to_durations(weekly)
print(f"barrier ratio from the durations {msrelapse.barrier_ratio(runs):.2f}")

health = msrelapse.fit_durations(runs, msrelapse.PAPER.state_health.value, censoring=False)
relapse = msrelapse.fit_durations(runs, msrelapse.PAPER.state_no_health.value, censoring=False)
beta, sigma = msrelapse.calibrate(health.mean, relapse.mean)
well = msrelapse.DoubleWell(alpha=msrelapse.PAPER.alpha_reference.value, beta=beta)

# The same ratio read off the calibrated potential. It comes out larger here,
# because the estimator above drops the prefactor of the exit time.
print(f"barrier ratio from the potential {well.barrier_ratio():.2f}")
```

# Reproducibility and data

The clinical records of the 2013 study were never released, and the article
carries no data availability statement and no supplementary material. The
package therefore ships a synthetic twin: 70 generated records whose mean
durations reproduce the printed ones, labelled as synthetic everywhere they are
loaded or written, so that nothing clinical can be read off them. Running
`msrelapse reproduce` measures every published aggregate on a record and prints
a table comparing each with the printed value under a stated tolerance,
returning a non-zero exit code when a judged row falls outside. The same table
can be built on a real record by anyone who obtains one. Every random operation
takes a seed, the test suite pins the numerical results, and the type
annotations are checked under strict settings.

# Acknowledgements

This package implements a model published with Isabella Bordi, Vito A. G.
Ricigliano, Viviana Annibali, Rosella Mechelli, Giovanni Ristori, Francesca
Grassi, Marco Salvetti and Alfonso Sutera, whose article is the sole source of
every equation and every number implemented here. The study behind it was supported by Fondazione
Italiana Sclerosi Multipla and by Progetto Strategico 2007 of the Italian
Ministry of Health.

# References
