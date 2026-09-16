# msrelapse

A reference implementation of the stochastic double-well model of
relapsing-remitting multiple sclerosis published by Bordi, Umeton and
co-authors in 2013. The package holds the potential and the stochastic
equation of the article, a seeded simulator, the duration fits and
memorylessness tests the article does not report, a virtual cohort generator
that writes registry-shaped relapse tables, and a command line that reproduces
the published numbers end to end.

> In 70 untreated relapsing-remitting MS patients, relapse and remission
> durations are both exponentially distributed (means about 4.3 and about 100
> weeks) with no detectable periodicity, so relapse onset is a memoryless
> process. A noise-driven asymmetric double-well model reproduces this, and the
> ratio of the logarithms of the two mean durations estimates the ratio of the
> two barrier heights (about 3). Bordi, Umeton et al., Int J Genomics 2013,
> doi:10.1155/2013/910321.

!!! warning "The records shipped here are synthetic"

    The clinical series behind the article was never released. What this
    package ships is a synthetic twin generated from the numbers the article
    prints, and no clinical claim can be read off it or off any figure drawn
    from it. [Data](data.md) says what the files are and how to ask the
    corresponding authors for the original series.

## Who this is for

| Audience | What they need | Entry point |
|---|---|---|
| Mathematical biologists | The stochastic equation, the potential, Kramers and first passage times, a simulator with seeds | `msrelapse.model`, `msrelapse.simulate` |
| Trial statisticians | Exponential and geometric fits with censoring, memorylessness tests, the Poisson to negative binomial mapping, annualised relapse rates with intervals | `msrelapse.fit`, `msrelapse.stats` |
| In-silico trial and digital twin groups | A calibrated virtual cohort generator that emits relapse event tables in registry format | `msrelapse.cohort`, `msrelapse.io` |

Every one of those entry points documents the article it comes from, and every
result object carries its citation, so that a number copied out of a session
can be traced back to the source. [Citing](citing.md) has the reference and the
BibTeX entries.

## Installation

```bash
uv add msrelapse          # in a uv project
pip install msrelapse     # anywhere else
```

The core install needs numpy, scipy and pandas only. Two extras are available:

```bash
pip install "msrelapse[plot]"   # matplotlib, for msrelapse.plots
pip install "msrelapse[fast]"   # numba, for the compiled integration kernels
pip install "msrelapse[all]"    # both
```

Neither extra changes a result. Without `plot` every function that draws raises
with the name of the extra to install; without `fast` the same integration runs
in numpy.

From a clone of the repository:

```bash
git clone https://github.com/renato-umeton/multiple-sclerosiss-modeling-bordi
cd multiple-sclerosiss-modeling-bordi
uv sync --all-extras
uv run --all-extras pytest
```

## Quick start

```python
import msrelapse as ms

weekly = ms.load_synthetic_bordi2013()                      # 70 synthetic records
runs = ms.weekly_to_durations(weekly)                       # run length encoding
relapse = ms.fit_durations(runs, state=ms.PAPER.state_no_health.value)
health = ms.fit_durations(runs, state=ms.PAPER.state_health.value)
beta, sigma = ms.calibrate(health.mean, relapse.mean)       # equation (4)
well = ms.DoubleWell(alpha=ms.PAPER.alpha_reference.value, beta=beta)
print(well.barriers(), well.barrier_ratio())                # the two barriers
simulated = ms.simulate_weekly(well, sigma, n_weeks=520, n_paths=5, rng=0)
print(ms.weekly_to_durations(simulated).head())             # the simulated runs
print(ms.reproduction_table(weekly))                        # the closing table
```

`fit_durations` corrects for the final remission that the end of follow up cut
short, which the article did not; pass `censoring=False` for the naive mean it
reported. [Reproducing the paper](reproducing.md) walks through the difference.

## Command line

Installing the package installs one command, `msrelapse`, with seven
subcommands. Each of them prints its own help.

| Subcommand | What it does |
|---|---|
| `msrelapse reproduce` | Measures every quantity the article reports on one weekly record, writes `numbers.json`, the record and the figures, and prints the closing table |
| `msrelapse simulate` | Generates a virtual cohort and writes it as a CSV file in any of the three schemas |
| `msrelapse fit` | Fits the duration law of one or both states of a durations CSV |
| `msrelapse test-memoryless` | Tests whether the durations of a state carry no memory |
| `msrelapse test-periodicity` | Looks for a period in the relapse onsets of a weekly record and pools the evidence |
| `msrelapse cite` | Prints the article and this package as a reference and as BibTeX |
| `msrelapse params` | Prints every number the article reports, with its unit and its source |

A first run needs no arguments and no data of its own:

```bash
msrelapse reproduce --out reproduction/
```

## Where to go next

- [Theory](theory.md) derives every equation the package implements, from the
  potential to the negative binomial relapse counts, and says which choices are
  the article's and which are this package's.
- [Reproducing the paper](reproducing.md) is the reproduction path: the command,
  what it writes, the closing table and its tolerances, the four notebooks, and
  the inconsistencies in the article that a reproduction runs into.
- [Data](data.md) describes the synthetic twin and the three schemas.
- [Paper facts](paper_facts.md) is the verified transcription of the article,
  equation by equation and figure by figure, which the package is checked
  against.
- The API pages document every module, generated from the docstrings.
