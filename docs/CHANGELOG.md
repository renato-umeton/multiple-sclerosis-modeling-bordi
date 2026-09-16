# Changelog

All notable changes to this project are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project
follows [semantic versioning](https://semver.org/spec/v2.0.0.html).

## Unreleased

### Added

- `msrelapse.model`: the potential of equations (2) and (3) as `DoubleWell`,
  its stationary points, barriers and curvatures, the fold value of the
  asymmetry, the mean exit times of equations (5) and (6) with and without the
  Kramers prefactor, the exact mean first passage time by quadrature, the
  barrier ratio of equation (7), and `calibrate`, which turns a pair of
  observed mean durations into a pair of model parameters.
- `msrelapse.simulate`: Euler and Maruyama integration of equation (4) with
  optional compiled kernels, first passage measurement, the hysteresis band
  that maps a path to the two clinical states, the Brownian bridge correction
  of grid read crossings, and downsampling to whole weeks.
- `msrelapse.renewal`: the alternating renewal twin of the model, the effective
  relapse onset rate, Poisson and negative binomial relapse counts through
  gamma mixing, and the relapse free curve.
- `msrelapse.fit`: exponential and geometric duration fits with and without
  censoring, four tests of memorylessness, negative binomial count fits with a
  Poisson comparison, gamma fits of per patient rates, and a periodicity test
  over relapse onsets.
- `msrelapse.stats`: annualised relapse rates with exact Poisson, negative
  binomial and bootstrap intervals, rate ratios by Poisson and negative
  binomial regression, the relapse free curve, and the negative binomial sample
  size of Zhu and Lakkis.
- `msrelapse.cohort`: virtual cohorts through either engine, the weekly
  rounding correction, the inversion of the naive mean measurement of the
  article, the per patient parameters of equation (7), and the three worked
  patients of the article as a table.
- `msrelapse.io`: the weekly, durations and events schemas, their validation
  with messages that name the first offending row, the conversions between
  them, and readers that accept a registry export carrying calendar dates.
- `msrelapse.plots`: Figures 2 to 8 of the article drawn from the data
  structures of the package, plus a survival figure and a count figure that
  belong to the methods note.
- `msrelapse.datasets`: the shipped synthetic twin of the 2013 cohort in the
  three schemas, its provenance note, and the closing table that measures every
  reported aggregate on any weekly record.
- `msrelapse._params`: every number the article reports, each with its unit and
  the sentence it comes from. No other module writes a number of the article.
- A command line with seven subcommands: `reproduce`, `simulate`, `fit`,
  `test-memoryless`, `test-periodicity`, `cite` and `params`.
- Four notebooks: the reproduction, the approach of the exit times to an
  exponential, the Poisson to negative binomial story, and a two arm virtual
  trial.
- Documentation site, including a theory page that derives every implemented
  equation and a reproduction page that records the three inconsistencies found
  in the article.

### Notes

- The clinical series of the article was never released, so the package ships a
  synthetic twin and says so on every frame, every run and every page.
- Nothing in the package is a clinical recommendation.
