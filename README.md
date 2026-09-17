# msrelapse

[![CI](https://github.com/renato-umeton/multiple-sclerosis-modeling-bordi/actions/workflows/ci.yml/badge.svg)](https://github.com/renato-umeton/multiple-sclerosis-modeling-bordi/actions/workflows/ci.yml)
[![Docs](https://img.shields.io/badge/docs-live-brightgreen.svg)](https://renato-umeton.github.io/multiple-sclerosis-modeling-bordi/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/renato-umeton/multiple-sclerosis-modeling-bordi/blob/main/LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%20to%203.14-blue.svg)](https://www.python.org/downloads/)

<!-- Enable this badge at the first release, once the project exists on PyPI.
[![PyPI](https://img.shields.io/pypi/v/msrelapse.svg)](https://pypi.org/project/msrelapse/)
-->

In 70 untreated relapsing-remitting MS patients, relapse and remission
durations are both exponentially distributed (means about 4.3 and about 100
weeks) with no detectable periodicity, so relapse onset is a memoryless
process. A noise-driven asymmetric double-well model reproduces this, and the
ratio of the logarithms of the two mean durations estimates the ratio of the
two barrier heights (about 3). Bordi, Umeton et al., Int J Genomics 2013,
doi:10.1155/2013/910321.

`msrelapse` is a reference implementation of that model. It holds the
potential, the stochastic equation of motion, the exit time estimators, the
duration fits, the trial statistics and a cohort generator, and it keeps every
number the article reports in one module together with the sentence each one
comes from, so that no reported value is written down twice.

![One simulated patient over ten years: the particle in the double well, the weekly flare and remission series, and an illustrative EDSS trajectory](docs/assets/double_well.gif)

The three panels are one simulated record of 520 weeks: the particle moving
between the health and flare wells of the asymmetric double well, the weekly
series of flare and remission weeks that path produces, and an illustrative
EDSS trajectory the same series drives, the first two at the asymmetry and the
noise calibrated to the two mean durations the article reports, about 100 weeks
in remission and about 4.3 weeks in relapse, although the episodes drawn run
longer than those two means, since the calibration times the passage from the
bottom of a well to the saddle and every week the path touches the flare state
counts as a whole flare week. The bottom panel is an extension and not part of
the article, which reports no disability score for any of its patients. It
opens at a baseline of 2.0 after the first attack, gives every flare a nadir
deficit drawn from the published distribution and mostly recovered within six
months, and leaves a residual of at least 0.5 after 42 percent of them, every
number of it taken from the literature with its source. The record drawn here
is one draw of that model and not a typical one. Its first three flares all
recover in full, the first two of them a week apart, so together they leave one
spike, the slow decay after it and nothing more, and the last two leave half a
point each, which is where the trace ends after ten years, a point above the
baseline it opened at. A flare never leaves the trace below the level the
record has accumulated for good, because accumulated disability does not
decrease, so the only fall the panel shows is the recovery after a spike. The
[disability page](https://renato-umeton.github.io/multiple-sclerosis-modeling-bordi/disability/)
explains the model, the evidence behind it and what it cannot do, and no trace
it draws is a prognosis.

## What it gives you

**Modellers** get the potential and the dynamics as objects.
`msrelapse.DoubleWell` holds the critical points, the two barriers and their
ratio for any pair of control and asymmetry parameters;
`msrelapse.simulate_paths` and `msrelapse.simulate_weekly` integrate the
stochastic equation; `msrelapse.exit_times`, `msrelapse.mfpt` and
`msrelapse.kramers_time` give the times to cross the barrier by simulation, by
exact first passage and by the Kramers estimate; `msrelapse.calibrate` turns a
pair of observed mean durations into a parameter pair.

**Trial statisticians** get the estimators a relapse record is summarised with.
`msrelapse.fit_durations` fits each state's duration law with or without the
censoring the end of follow up imposes; `msrelapse.test_memoryless` and
`msrelapse.test_periodicity` put the two qualitative claims of the article to a
test; `msrelapse.arr` and `msrelapse.compare_arr` compute annualised relapse
rates and rate ratios with Poisson, negative binomial and bootstrap intervals;
`msrelapse.sample_size_arr` and `msrelapse.relapse_free_curve` say what a
cohort of a given size can resolve.

**In silico trial groups** get virtual cohorts. `msrelapse.CohortSpec` and
`msrelapse.generate` build a cohort of any size from two mean durations and a
follow up length, by drawing durations directly or by integrating the
potential; `msrelapse.per_patient_params` and `msrelapse.lognormal_around`
spread the parameters across patients; `msrelapse.write_csv` writes the result
in the weekly, durations or events schema, and the same cohort is available
from the command line without writing any Python.

## Installation

The package needs Python 3.10 or newer and depends on numpy, scipy and pandas.
From the first release on it will install from PyPI:

```bash
pip install msrelapse              # the library and the command line
pip install "msrelapse[plot]"      # adds matplotlib for the figures
pip install "msrelapse[fast]"      # adds numba for the path integrator
pip install "msrelapse[all]"       # both extras
```

Until then, and for any work on the package itself, install from source with
[uv](https://docs.astral.sh/uv/):

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
git clone https://github.com/renato-umeton/multiple-sclerosis-modeling-bordi
cd multiple-sclerosis-modeling-bordi
uv sync
```

`uv sync` downloads a Python if none is suitable, creates `.venv`, installs the
package in editable form and adds the `dev` dependency group, all from the
locked versions in `uv.lock`. The optional extras and the two other groups are
opt in:

```bash
uv sync --extra plot --extra fast   # matplotlib and numba
uv sync --group docs                # mkdocs and the API reference build
uv sync --group notebooks           # nbformat, nbclient and the kernel
uv sync --all-extras --all-groups   # everything at once
```

Day to day:

```bash
uv run python script.py    # run inside the project environment
uv run pytest              # the test suite
uv run ruff check .        # lint
uv run mypy                # type check
uv sync                    # re-sync after pulling a change to uv.lock
```

## Quick start

```python
import msrelapse

RELAPSE = msrelapse.PAPER.state_no_health.value  # +1, the "no health" state
REMISSION = msrelapse.PAPER.state_health.value  # -1, the "health" state

# A virtual cohort of 70 records with the durations the article reports.
# The records are synthetic and carry a note that says so.
cohort = msrelapse.generate(msrelapse.bordi2013_spec(), rng=20130910)
print(cohort.provenance)
durations = cohort.durations

# Fit the duration law of each state by maximum likelihood. censoring=False is
# the naive arithmetic the article used; the default honours the remissions cut
# short by the end of follow up and returns a longer mean.
relapse = msrelapse.fit_durations(durations, RELAPSE, censoring=False)
remission = msrelapse.fit_durations(durations, REMISSION, censoring=False)
print(f"mean relapse   {relapse.mean:.2f} weeks over {relapse.n} runs")
print(f"mean remission {remission.mean:.1f} weeks over {remission.n} runs")

# Equation (7): the ratio of the logarithms of the two mean durations estimates
# the ratio of the two barrier heights.
print(f"barrier ratio dV1/dV2 {msrelapse.barrier_ratio(durations):.2f}")

# Calibrate a potential to those two means, then read the barriers off it.
# Their ratio comes out larger than the estimate of equation (7) above, because
# equation (7) drops the prefactor of the exit time.
beta, sigma = msrelapse.calibrate(remission.mean, relapse.mean)
well = msrelapse.DoubleWell(alpha=msrelapse.PAPER.alpha_reference.value, beta=beta)
barriers = well.barriers()
print(f"beta {beta:.4f}  noise sigma {sigma:.4f}")
print(f"dV1 {barriers.health:.4f}  dV2 {barriers.relapse:.4f}  ratio {barriers.ratio:.2f}")

# Twenty first passage times out of the health well, in weeks.
times = msrelapse.exit_times(well, sigma, "health", n_paths=20, rng=0)
print(f"mean exit from health over 20 paths: {float(times.mean()):.0f} weeks")

msrelapse.cite()
```

## Command line

The package installs one command with eight subcommands. Every one of them
takes `--help`.

```bash
msrelapse reproduce --out reproduction   # the whole analysis, written out
msrelapse simulate --n 200 --tau-health 80 --tau-relapse 3 \
    --weeks 400 --seed 0 -o cohort.csv   # a virtual cohort as a CSV file
msrelapse animate --seed 6               # the animation above, as a GIF file
msrelapse fit durations.csv              # the duration law of each state
msrelapse test-memoryless durations.csv  # do the durations carry memory
msrelapse test-periodicity weekly.csv    # is there a period in the onsets
msrelapse cite                           # the article and this package
msrelapse params                         # every number the article reports
```

Under an install from source, put `uv run` in front of each line. The durations
above are illustrative; `msrelapse reproduce` uses the ones the article
reports.

## Data provenance

The clinical series of the study was never released: the article carries no
data availability statement, no supplementary material and no deposited
records. What this package ships instead is a **synthetic twin**, 70 generated
records whose mean durations are aimed at the printed ones and land within
about five percent of them, in the three schemas the package reads.
`msrelapse.load_synthetic_bordi2013()` returns it and `msrelapse.provenance()`
returns the full note, which names the generator, the seed and the generative
durations. Every command that touches it says so on standard output.

The twin is not the cohort of the study and no clinical claim can be read off
it. It exists so that the analysis pipeline can be run end to end by anyone. A
reader who obtains the real series from the corresponding authors of the
article can run exactly the same functions on it.

## Reproducing the paper

```bash
uv run msrelapse reproduce --out reproduction
```

The run measures on one weekly record every aggregate the article reports,
writes `numbers.json`, the record, the durations and the figures, and prints a
closing table. The figures need the `plot` extra; pass `--no-figures` to skip
them. Each row of that table puts the number the article printed beside the
number this record gives, the rule the two are compared under and whether the
rule is met, and the command exits non-zero when a judged row falls outside its
tolerance. The rows are the two mean durations, the relapse
duration range, the longest remission, the length of the relapsing-remitting
phase, the barrier ratio of equation (7), a goodness of fit p value for each
state's durations and a pooled periodicity p value. The last three rows have no
printed counterpart, because the article reports no fit and no test anywhere;
they are read as the record declining to contradict it.

`msrelapse.reproduction_table(weekly)` builds the same table in a session, on
the shipped twin or on any record in the weekly schema.

## Notebooks

Four executed notebooks under `notebooks/` show the package at work, and GitHub
renders their outputs in place:

- [Reproducing Bordi et al. 2013](https://github.com/renato-umeton/multiple-sclerosis-modeling-bordi/blob/main/notebooks/01_reproduce_bordi2013.ipynb): the
  paper's figures, the three duration fits, the memorylessness and periodicity
  tests, the barrier ratio and the closing table, all from the synthetic twin.
- [From the stochastic equation to exponential durations](https://github.com/renato-umeton/multiple-sclerosis-modeling-bordi/blob/main/notebooks/02_sde_to_exponential.ipynb):
  Kramers times, exact first passage times and simulated exits across noise levels.
- [From Poisson counts to a negative binomial](https://github.com/renato-umeton/multiple-sclerosis-modeling-bordi/blob/main/notebooks/03_poisson_to_negative_binomial.ipynb):
  how between patient heterogeneity turns Poisson relapse counts into over
  dispersed ones.
- [A virtual cohort for trial design](https://github.com/renato-umeton/multiple-sclerosis-modeling-bordi/blob/main/notebooks/04_virtual_cohort_for_trial_design.ipynb):
  a two arm virtual trial with annualised relapse rates, a rate ratio and a power curve.

Rebuild them with `python notebooks/build_notebooks.py`; see
[docs/CONTRIBUTING.md](https://github.com/renato-umeton/multiple-sclerosis-modeling-bordi/blob/main/docs/CONTRIBUTING.md).

## Documentation

The user guide, the theory notes, the reproduction walkthrough and the API
reference live at
<https://renato-umeton.github.io/multiple-sclerosis-modeling-bordi/>. It is
published with GitHub Pages, rebuilt from `main` by the docs workflow on every
push, and builds locally with `uv run --group docs mkdocs serve`.

## Citing

Cite the article. Cite the software as well if the code did work for you, and
never instead of the article.

> Bordi I, Umeton R, Ricigliano VAG, Annibali V, Mechelli R, Ristori G, Grassi
> F, Salvetti M, Sutera A. A mechanistic, stochastic model helps understand
> multiple sclerosis course and pathogenesis. International Journal of
> Genomics. 2013;2013:910321. doi:10.1155/2013/910321.

```bibtex
@article{bordi2013mechanistic,
  author    = {Bordi, Isabella and Umeton, Renato and Ricigliano, Vito A. G. and
               Annibali, Viviana and Mechelli, Rosella and Ristori, Giovanni and
               Grassi, Francesca and Salvetti, Marco and Sutera, Alfonso},
  title     = {A mechanistic, stochastic model helps understand multiple sclerosis
               course and pathogenesis},
  journal   = {International Journal of Genomics},
  volume    = {2013},
  pages     = {910321},
  year      = {2013},
  doi       = {10.1155/2013/910321},
  publisher = {Wiley}
}
```

> Umeton R. msrelapse: a reference implementation of the Bordi et al. 2013
> double well model of multiple sclerosis. Version 0.1.0. 2026.
> https://github.com/renato-umeton/multiple-sclerosis-modeling-bordi

`msrelapse.cite()` prints the article reference followed by a BibTeX entry for
each of the two, and `msrelapse.citation()` returns the same text as a string.
`CITATION.cff` and `codemeta.json` carry the same information for reference
managers and for software indexes. The software is deposited in no archive and
carries no DOI of its own: the repository URL is its stable address, and the
article is the reference to cite. The
[citing page](https://renato-umeton.github.io/multiple-sclerosis-modeling-bordi/citing/)
has both BibTeX entries in full.

## License

MIT, see
[LICENSE](https://github.com/renato-umeton/multiple-sclerosis-modeling-bordi/blob/main/LICENSE).
The article is open access under CC BY, which is what allows its equations,
figures and numbers to be reused here with citation.
