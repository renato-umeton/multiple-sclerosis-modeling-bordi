"""Build the four notebooks of this repository from their sources in this file.

The notebooks are generated rather than edited by hand, so that they stay small
in a diff, carry no execution state and can be rebuilt after the package moves
under them. Running this script writes the four ``.ipynb`` files next to it with
empty outputs, which is what the repository commits; ``--execute`` runs each of
them through nbclient first, so that a rebuild can be checked before it is
written.

Every notebook takes its numbers of the article from ``msrelapse.PAPER`` rather
than writing them down, fixes one seed for every random step, and says in its
opening cells whether the data it uses are synthetic.

``tests/test_notebooks.py`` checks that the committed files still hold the cells
this script builds, so a change here, and a change to what
:func:`msrelapse.citation` prints or to the version of the package, which the
closing cell of every notebook carries, both ask for a rebuild.

Examples
--------
Rebuild the four notebooks in place::

    uv run --no-sync python notebooks/build_notebooks.py

Rebuild them and run each one before writing it::

    uv run --no-sync python notebooks/build_notebooks.py --execute
"""

from __future__ import annotations

import argparse
import copy
from collections.abc import Callable
from pathlib import Path
from typing import Any, Final

import nbformat
from nbformat import NotebookNode

import msrelapse

__all__ = ["NOTEBOOK_DIR", "NOTEBOOK_NAMES", "build", "build_all", "main", "write"]

# The constructors and the writer of nbformat carry no annotations, so they are
# named once here with the types they use, rather than spreading Any through
# every call below.
_new_markdown_cell: Final[Callable[[str], NotebookNode]] = nbformat.v4.new_markdown_cell
_new_code_cell: Final[Callable[[str], NotebookNode]] = nbformat.v4.new_code_cell
_new_notebook: Final[Callable[..., NotebookNode]] = nbformat.v4.new_notebook
_write_notebook: Final[Callable[[NotebookNode, Any], None]] = nbformat.write

NOTEBOOK_DIR: Final = Path(__file__).resolve().parent
"""Directory the notebooks are written to, which is the one holding this file."""

KERNEL_NAME: Final = "python3"
EXECUTION_TIMEOUT: Final = 900

_NOTEBOOK_METADATA: Final = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": KERNEL_NAME},
    "language_info": {"name": "python"},
}
"""The only metadata the notebooks carry, kept minimal so that a run adds nothing."""


def _markdown(source: str) -> NotebookNode:
    """Return one markdown cell.

    Parameters
    ----------
    source : str
        The text of the cell. Leading and trailing blank lines are dropped, so
        that the source can be written as a block in this file.

    Returns
    -------
    nbformat.NotebookNode
        The cell.
    """
    return _new_markdown_cell(source.strip("\n"))


def _code(source: str) -> NotebookNode:
    """Return one code cell, with no output and no execution count.

    Parameters
    ----------
    source : str
        The code of the cell, written as a block in this file.

    Returns
    -------
    nbformat.NotebookNode
        The cell.
    """
    return _new_code_cell(source.strip("\n"))


def _citation_cell() -> NotebookNode:
    """Return the closing cell every notebook ends with.

    Returns
    -------
    nbformat.NotebookNode
        A markdown cell holding what :func:`msrelapse.citation` prints, so that
        a reader who opens one notebook alone still has the article in front of
        them.
    """
    return _markdown(
        "## How to cite\n\n"
        "This notebook reproduces the article below, and the package it uses carries the same\n"
        "reference on every result object. The text is what `msrelapse.citation()` prints.\n\n"
        "```text\n"
        f"{msrelapse.citation()}\n"
        "```"
    )


def _reproduce_bordi2013() -> list[NotebookNode]:
    """Return the cells of the reproduction notebook.

    Returns
    -------
    list of nbformat.NotebookNode
        The cells, in order.
    """
    return [
        _markdown("""
# Reproducing Bordi et al. 2013

This notebook walks through everything the article reports about its cohort of 70
relapsing-remitting patients and puts the number this package measures beside the number the
article prints. It draws Figures 2 to 4 from a record and from the bars digitised off the
paper, fits the two duration laws three ways each, tests the two qualitative claims the
article makes in prose, applies the barrier ratio of equation (7), calibrates the stochastic
model on the fitted means, and closes with the reproduction table, which fails the notebook
if any row falls outside its tolerance.
"""),
        _markdown("""
## Provenance of the data

**The records loaded below are synthetic.** The clinical series of the article was never
released: the paper carries no data availability statement, no supplementary material and no
deposited series. What ships with this package instead is a synthetic twin, 70 records
generated from the numbers the article prints. It is not the cohort of the study and no
clinical claim can be read off it.

To run the same notebook on a real cohort, replace the loading cell with

```python
weekly = msrelapse.read_weekly("my_cohort_weekly.csv")
durations = msrelapse.weekly_to_durations(weekly)
```

and leave every other cell as it is. The file has to obey the weekly schema of
`msrelapse.io`: one row per patient-week, with the columns `patient_id`, `week` and `state`,
the state being +1 in a relapse week and -1 in a remission week.
"""),
        _code("""
%matplotlib inline

import matplotlib.pyplot as plt
import pandas as pd

import msrelapse
from msrelapse import plots

# Every random step of this notebook is drawn from this one seed.
SEED = 20130910
# Bootstrap replicates behind each of the four memorylessness tests. Their p
# value is (1 + exceeded) / (n_boot + 1), a Monte Carlo estimate that moves with
# the seed: over thirty seeds the p value of the cv test on the health durations
# spans about 0.57 to 0.92 at 200 replicates and about 0.72 to 0.83 at the 2000
# used here. The eight tests cost about two seconds at this setting.
N_BOOT = 2000

RELAPSE = msrelapse.PAPER.state_no_health.value
HEALTH = msrelapse.PAPER.state_health.value
STATE_NAMES = {RELAPSE: "no health", HEALTH: "health"}

pd.set_option("display.width", 100)
print(f"msrelapse {msrelapse.__version__}")
"""),
        _code("""
weekly = msrelapse.load_synthetic_bordi2013("weekly")
durations = msrelapse.load_synthetic_bordi2013("durations")

print(weekly.attrs["provenance"])
print(f"{weekly['patient_id'].nunique()} patients over {len(weekly)} patient-weeks")
weekly.head()
"""),
        _code("""
print(msrelapse.provenance())
"""),
        _markdown("""
## Figures 2 to 4

Each figure is drawn twice: on the left the bars digitised from the article, on the right the
same figure measured on this record. Figure 2 shows the three patients of this cohort with
the most relapses, which is the counterpart of the three sample patients of the paper.
"""),
        _code("""
relapse_runs = durations[durations["state"] == RELAPSE]
per_patient_relapses = relapse_runs.groupby("patient_id", sort=True).size()
# The counts are already in patient id order and the sort below is stable, so a
# tie is broken by the patient id and these three names never move.
examples = list(per_patient_relapses.sort_values(ascending=False, kind="stable").head(3).index)
print(per_patient_relapses[examples])
"""),
        _code("""
plots.fig2_sample_patients(weekly, examples)
plt.show()
"""),
        _code("""
followup_weeks = weekly.groupby("patient_id", sort=True).size()
axes = plt.subplots(1, 2, figsize=(11.0, 3.6), layout="constrained")[1]
plots.fig3_rr_phase_histogram(ax=axes[0])
plots.fig3_rr_phase_histogram(followup_weeks.to_numpy(), ax=axes[1])
plt.show()
"""),
        _code("""
axes = plt.subplots(2, 2, figsize=(9.5, 7.2), layout="constrained")[1]
plots.fig4_duration_histograms(axes=axes[0])
plots.fig4_duration_histograms(durations, axes=axes[1])
plt.show()
"""),
        _markdown("""
## The two mean durations, fitted three ways

The article reports one mean per state and no interval. The naive fit is its arithmetic:
every recorded run counted as complete, the final remission of a record included although the
end of follow up cut it short. The censored fit gives that run its time but not its event,
which is the estimate of the process behind the record. The geometric fit reads the same runs
as whole weeks, which is what a weekly record holds, and it returns the same mean as the
censored exponential; the two differ in what they claim about the days between the integers,
not in what they estimate.

The last column is the log likelihood at the fitted parameter, and it is what separates rows
that agree in every other column. Read it down one family only: the two exponential fits
differ by what the censored runs contribute, while the geometric number is the probability of
a whole week rather than a density, so it does not sit on the same scale as the two above it.
The two relapse rows that stay identical are identical for a reason the table gives itself:
no relapse run of this record is censored, so the naive and the censored exponential fits are
one fit.
"""),
        _code("""
fit_rows = []
for state in (RELAPSE, HEALTH):
    fits = {
        "naive": msrelapse.fit_durations(durations, state, censoring=False),
        "censored": msrelapse.fit_durations(durations, state),
        "geometric": msrelapse.fit_durations(durations, state, family="geometric"),
    }
    for label, fit in fits.items():
        fit_rows.append(
            (
                STATE_NAMES[state],
                label,
                fit.n,
                fit.n_censored,
                fit.mean,
                fit.ci_low,
                fit.ci_high,
                fit.loglik,
            )
        )

duration_fits = pd.DataFrame(
    fit_rows,
    columns=["state", "fit", "n", "n_censored", "mean_w", "ci_low", "ci_high", "loglik"],
)
duration_fits
"""),
        _markdown("""
## Is the record memoryless, and does it carry a period?

The article asserts both claims on the shape of two histograms. The four tests below give the
memorylessness of each state a number, the figure under them puts the observed survival of
each state beside the exponential fitted to it, and the Fisher g test after that reads the
rhythm of the relapse onsets. A large p value means the record does not contradict the
article, and not that the claim is established: none of these tests has much power on a cohort
of this size, and the periodicity test has almost none at four onsets per patient.

All four tests read their p value off a bootstrap, so every one of those numbers carries a
Monte Carlo error of its own on top of the sampling error of the cohort, and the table shows
them to three decimals for that reason.
"""),
        _code("""
memoryless_rows = []
for state in (RELAPSE, HEALTH):
    for method in ("hazard", "cv", "ks", "ad"):
        test = msrelapse.test_memoryless(durations, state, method=method, n_boot=N_BOOT, rng=SEED)
        memoryless_rows.append((STATE_NAMES[state], method, test.n, test.statistic, test.p_value))

memoryless = pd.DataFrame(
    memoryless_rows,
    columns=["state", "method", "n", "statistic", "p_value"],
)
# Three decimals, which is as far as a bootstrap of this size can be read.
memoryless["p_value"] = memoryless["p_value"].round(3)
memoryless
"""),
        _code("""
axes = plt.subplots(1, 2, figsize=(11.0, 4.0), layout="constrained")[1]
plots.fig_survival_vs_exponential(durations, RELAPSE, ax=axes[0])
plots.fig_survival_vs_exponential(durations, HEALTH, ax=axes[1])
plt.show()
"""),
        _code("""
periodicity = msrelapse.test_periodicity(weekly, method="fisher_g")
print(periodicity.pooled)
print(periodicity.pooled.details)
periodicity.per_patient[["n_weeks", "n_onsets", "p_value"]].describe()
"""),
        _markdown("""
## The barrier ratio of equation (7)

Equation (7) reads the ratio of the two barriers off the two mean durations. The means are
the naive ones, because those are the means the article used to reach its 3.1. The table of
the three sample patients puts each printed value beside the value its neighbours imply: the
two lower rows agree to two decimals, while patient 23 does not. The asymmetry its two
printed durations ask for is about 0.256, and the 0.25 printed beside them delivers a ratio
of about 10.8 rather than the 11.8 printed with it. That is an inconsistency in the article
itself, confirmed against its own Figure 8(a), and not a transcription error here.
"""),
        _code("""
cohort_ratio = msrelapse.barrier_ratio(durations)
printed_ratio = msrelapse.barrier_ratio_from_durations(
    msrelapse.PAPER.tau_health_cohort_weeks.value,
    msrelapse.PAPER.tau_no_health_cohort_weeks.value,
)
print(f"this record: {cohort_ratio:.3f}")
print(f"the durations the article prints: {printed_ratio:.3f}")
print(f"the ratio the article prints: {msrelapse.PAPER.barrier_ratio_cohort.value}")
"""),
        _code("""
patient_params = msrelapse.per_patient_params(durations)
patient_params[patient_params["patient_id"].isin(examples)]
"""),
        _code("""
msrelapse.paper_patients()
"""),
        _markdown("""
## Calibrating the stochastic model on the fitted means

The calibration searches for the asymmetry and the noise whose exact mean first passage
times, from the bottom of each well to the saddle, are the two fitted means. The noise it
asks for is about twice the only noise value the article prints, which is the gap
`docs/paper_facts.md` records: the article's own epsilon does not reproduce its own durations
under an exact first passage reading.
"""),
        _code("""
tau_health = msrelapse.fit_durations(durations, HEALTH).mean
tau_relapse = msrelapse.fit_durations(durations, RELAPSE).mean
beta, sigma = msrelapse.calibrate(tau_health, tau_relapse, passage="bottom_to_saddle")
well = msrelapse.DoubleWell(msrelapse.PAPER.alpha_reference.value, beta)

print(f"fitted means: {tau_health:.1f} weeks in health, {tau_relapse:.2f} weeks in no health")
print(f"beta {beta:.4f}, sigma {sigma:.4f}, so epsilon = sigma^2 is {sigma**2:.4f}")
print(f"the article prints epsilon = {msrelapse.PAPER.epsilon_noise_variance.value}")
print(well.barriers())
"""),
        _code("""
plots.fig6_asymmetric_potential(beta=beta)
plt.show()
"""),
        _code("""
plots.fig7_simulated_paths(
    sigma=sigma,
    betas=(msrelapse.PAPER.beta_symmetric.value, beta),
    rng=SEED,
)
plt.show()
"""),
        _markdown("""
## The potential of each example patient, Figure 8

The article fits a patient by fixing alpha at its reference value and moving the asymmetry
until the two wells stand in the ratio equation (7) reads off that patient's durations. The
three panels below are that fit for the three patients of Figure 2 above.
"""),
        _code("""
example_betas = patient_params.set_index("patient_id").loc[examples, "beta"]
labels = [f"{patient} (beta {value:.3f})" for patient, value in example_betas.items()]
plots.fig8_patient_potentials(list(example_betas), labels)
plt.show()
"""),
        _markdown("""
## The closing table

Every aggregate the article reports, measured on this record the way the article measured its
own, with the rule each comparison is judged under. The two goodness of fit rows read a
bootstrap, which is drawn from the seed of this notebook like every other random step here.
The cell after the table raises if any judged row falls outside its tolerance, so that a
failed reproduction fails the notebook.
"""),
        _code("""
table = msrelapse.reproduction_table(weekly, rng=SEED)
table
"""),
        _code("""
judged = table[table["within_tolerance"].notna()]
outside = judged[~judged["within_tolerance"].astype(bool)]
if not outside.empty:
    raise AssertionError(
        f"{len(outside)} row(s) of the closing table fall outside tolerance:\\n{table.to_string()}"
    )
print(f"within_tolerance: True on every one of the {len(judged)} judged rows")
"""),
        _citation_cell(),
    ]


def _sde_to_exponential() -> list[NotebookNode]:
    """Return the cells of the notebook on the exit time of the stochastic model.

    Returns
    -------
    list of nbformat.NotebookNode
        The cells, in order.
    """
    return [
        _markdown("""
# From the stochastic equation to exponential durations

The article states that the durations of both states are exponential and reads the two
barriers off their means. This notebook asks what the model itself says: how the mean exit
time of a well grows as the noise falls, how close the classical Kramers formula and the
exact first passage time are to the times a simulated path actually spends, and when the exit
time is exponential at all. It ends with the Brownian bridge correction, which is what keeps
a crossing tested only at the grid points from running long.
"""),
        _markdown("""
## Provenance

**No clinical data are used in this notebook.** Every number below comes from the model: two
potentials, one calibrated to the two mean durations the article prints and one at the
asymmetry the article draws its own figures with, and paths integrated from them. The two
durations and the two parameters of the article are read from `msrelapse.PAPER`.
"""),
        _code("""
%matplotlib inline

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import msrelapse

# Every random step of this notebook is drawn from this one seed.
SEED = 20130910
# Paths behind each simulated point, and the integration step. A few thousand
# paths smooth the points at a cost that grows in proportion.
N_PATHS = 300
DT = 0.02

ALPHA = msrelapse.PAPER.alpha_reference.value
"""),
        _code("""
calibrated_beta, calibrated_sigma = msrelapse.calibrate(
    msrelapse.PAPER.tau_health_cohort_weeks.value,
    msrelapse.PAPER.tau_no_health_cohort_weeks.value,
)
calibrated = msrelapse.DoubleWell(ALPHA, calibrated_beta)
illustrative = msrelapse.DoubleWell(ALPHA, msrelapse.PAPER.beta_illustrative.value)
wells = {
    "calibrated": calibrated,
    f"illustrative, beta = {illustrative.beta:g}": illustrative,
}

print(f"calibrated beta {calibrated_beta:.4f}, sigma {calibrated_sigma:.4f}")
for name, well in wells.items():
    print(name, well.barriers())
"""),
        _markdown("""
## The sweep

For each well, each side and each noise amplitude the sweep collects five numbers: the
Kramers escape time, the exact mean first passage time from the bottom of the well to the
saddle, the same from one well bottom to the other, and the mean and the coefficient of
variation of a few hundred simulated first passage times. The grid starts at 0.40 rather than
lower because the health well of the calibrated potential takes thousands of weeks to leave
below that, and simulating a few hundred such paths costs more than this notebook is meant to.
"""),
        _code("""
SIGMAS = np.array([0.40, 0.45, 0.50, 0.55, 0.60, 0.70, 0.80])

# One generator runs through the whole sweep, so that every point is a sample of
# its own rather than the same noise stream measured again at another sigma. The
# sweep is still reproducible, because the generator starts from the seed above.
sweep_generator = np.random.default_rng(SEED)
sweep_rows = []
for name, well in wells.items():
    for side in ("health", "relapse"):
        for sigma in SIGMAS:
            times = msrelapse.exit_times(
                well, sigma, side, N_PATHS, dt=DT, rng=sweep_generator, bridge_correction=True
            )
            bottoms = msrelapse.passage_endpoints(well, side, "bottom_to_bottom")
            sweep_rows.append(
                (
                    name,
                    side,
                    float(sigma),
                    msrelapse.kramers_time(well, sigma, side),
                    msrelapse.mfpt(well, sigma, side),
                    msrelapse.mfpt(well, sigma, side, *bottoms),
                    float(times.mean()),
                    float(times.std(ddof=1) / times.mean()),
                )
            )

sweep = pd.DataFrame(
    sweep_rows,
    columns=[
        "well",
        "side",
        "sigma",
        "kramers",
        "mfpt_bottom_to_saddle",
        "mfpt_bottom_to_bottom",
        "simulated_mean",
        "simulated_cv",
    ],
)
sweep
"""),
        _code("""
axes = plt.subplots(2, 2, figsize=(10.5, 7.5), sharex=True, layout="constrained")[1]
for row, name in enumerate(wells):
    for column, side in enumerate(("health", "relapse")):
        panel = axes[row][column]
        block = sweep[(sweep["well"] == name) & (sweep["side"] == side)]
        inverse = 1.0 / block["sigma"] ** 2
        panel.semilogy(inverse, block["kramers"], marker="o", label="Kramers")
        panel.semilogy(inverse, block["mfpt_bottom_to_saddle"], marker="s", label="to the saddle")
        panel.semilogy(inverse, block["mfpt_bottom_to_bottom"], marker="^", label="well to well")
        panel.semilogy(
            inverse,
            block["simulated_mean"],
            marker="x",
            linestyle="none",
            color="black",
            label="simulated",
        )
        panel.set_title(f"{name}, {side} well")
        panel.set_xlabel("1 / sigma^2")
        panel.set_ylabel("mean exit time (weeks)")
axes[0][0].legend(fontsize="small")
plt.show()
"""),
        _markdown("""
The three analytic readings answer three different questions. Kramers is a well to well time
in the small noise limit. In the deep health well it sits above the exact first passage time
to the saddle by about a factor of two, because a walker that reaches the saddle still falls
back about half the time. The factor grows as a well gets shallower against the noise, since
the walker then comes back into the well more often: in the relapse well of the calibrated
potential, the shallowest of the four, it runs from about two at the smallest noise of the
sweep to about four at the largest, while the relapse well of the illustrative potential,
whose barrier is more than twice as high, only reaches about two and a half. The exact well to
well time is the one Kramers approximates, and over this range of noise the two stay within
about a fifth of one another; the formula is exact only as the noise goes to zero, which the
durations of the article do not reach. The simulated points follow the curve to the saddle,
which is the passage they were measured over.
"""),
        _code("""
panel = plt.subplots(figsize=(7.0, 4.2), layout="constrained")[1]
for name in wells:
    for side in ("health", "relapse"):
        block = sweep[(sweep["well"] == name) & (sweep["side"] == side)]
        panel.plot(block["sigma"], block["simulated_cv"], marker="o", label=f"{name}, {side}")
panel.axhline(1.0, linestyle="--", color="grey")
panel.set_xlabel("sigma")
panel.set_ylabel("coefficient of variation of the exit time")
panel.set_ylim(0.0, 1.3)
panel.legend(fontsize="small")
plt.show()
"""),
        _markdown("""
An exponential exit time has a coefficient of variation of 1, and every point of the panel
sits near that line. That is not evidence that all four wells are exponential. With 300 paths
behind each point the coefficient of variation of a sample that really is exponential already
scatters by about 0.06, and the points span roughly 0.82 to 1.12, so the panel cannot tell
the deep health well from the shallow relapse one. What it does rule out is an exit time far
from exponential, which would sit well off the line at every noise. The difference between
the two wells shows instead in the regime table below and in the shape of the survival curves
beside it, where the shallow well keeps far more paths inside at short times than a
memoryless law would.
"""),
        _markdown("""
## Why the relapse well of the calibrated potential is not exponential

An exit time is exponential when leaving the well is a rare event, that is when the barrier
is high against the noise and the walker relaxes to the bottom of the well many times before
it escapes. The exponent `2 dV / sigma^2` is that comparison, and the Kramers prefactor is
the relaxation time the exit time has to be long against. At the calibrated parameters the
health well clears both, by a barrier of about twice the noise variance and an exit time of
about twenty relaxation times. The relapse well clears neither: its barrier sits below the
noise variance and its mean exit time is shorter than the relaxation time itself, so there is
no memoryless law behind the record it produces, however the article reads its histogram.
"""),
        _code("""
barriers = calibrated.barriers()
regime_rows = []
for side, barrier in (("health", barriers.health), ("relapse", barriers.relapse)):
    regime_rows.append(
        (
            side,
            barrier,
            2.0 * barrier / calibrated_sigma**2,
            msrelapse.kramers_prefactor(calibrated, side),
            msrelapse.mfpt(calibrated, calibrated_sigma, side),
        )
    )

regime = pd.DataFrame(
    regime_rows,
    columns=["side", "barrier", "two_dV_over_sigma2", "prefactor_w", "mfpt_w"],
)
regime
"""),
        _code("""
N_SHAPE_PATHS = 2000
shape_generator = np.random.default_rng(SEED)
shapes = {
    side: msrelapse.exit_times(
        calibrated, calibrated_sigma, side, N_SHAPE_PATHS, dt=DT, rng=shape_generator
    )
    for side in ("health", "relapse")
}

panel = plt.subplots(figsize=(7.0, 4.2), layout="constrained")[1]
grid = np.linspace(0.0, 5.0, 200)
panel.semilogy(grid, np.exp(-grid), color="black", linestyle="--", label="exponential")
for side, times in shapes.items():
    scaled = np.sort(times / times.mean())
    survival = 1.0 - np.arange(scaled.size) / scaled.size
    cv = times.std(ddof=1) / times.mean()
    panel.semilogy(scaled, survival, label=f"{side} well, cv {cv:.2f}")
panel.set_ylim(1e-3, 1.05)
panel.set_xlabel("exit time, in units of its own mean")
panel.set_ylabel("share of paths still inside the well")
panel.legend()
plt.show()
"""),
        _markdown("""
## The Brownian bridge correction

A threshold tested only at the grid points is crossed a step late, because a path that
crossed and came back between two samples is never seen. The error grows like the square root
of the step, and the correction moves the absorbing level towards the walker by
`0.5826 sigma sqrt(dt)`, the mean overshoot of a Brownian bridge past a level. The sweep
below is the same passage measured at six steps with the correction on and off, against the
exact mean first passage time.
"""),
        _code("""
# More paths than the sweep above, because the bias this measures is a few
# percent and has to stand clear of the noise of the mean.
BRIDGE_PATHS = 6000
DT_GRID = np.array([0.005, 0.01, 0.02, 0.05, 0.1, 0.2])
bridge_sigma = msrelapse.PAPER.noise_amplitude.value
exact_relapse = msrelapse.mfpt(illustrative, bridge_sigma, "relapse")

# Here the seed is deliberately repeated rather than carried in a generator: the
# two series of each step start from the same noise stream, so what separates
# them is the correction and not a different draw.
bridge_rows = []
for dt in DT_GRID:
    for corrected in (True, False):
        times = msrelapse.exit_times(
            illustrative,
            bridge_sigma,
            "relapse",
            BRIDGE_PATHS,
            dt=dt,
            rng=SEED,
            bridge_correction=corrected,
        )
        bridge_rows.append((float(dt), corrected, float(times.mean())))

bridge = pd.DataFrame(bridge_rows, columns=["dt", "bridge_correction", "mean_w"])
bridge["relative_error"] = bridge["mean_w"] / exact_relapse - 1.0
bridge
"""),
        _code("""
panel = plt.subplots(figsize=(7.0, 4.2), layout="constrained")[1]
for corrected in (True, False):
    block = bridge[bridge["bridge_correction"] == corrected]
    panel.plot(
        np.sqrt(block["dt"]),
        block["mean_w"],
        marker="o",
        label=f"bridge_correction={corrected}",
    )
panel.axhline(exact_relapse, linestyle="--", color="grey", label="exact mean first passage time")
panel.set_xlabel("sqrt(dt)")
panel.set_ylabel("mean exit time (weeks)")
panel.legend()
plt.show()
"""),
        _markdown("""
## What to take away

The uncorrected mean rises with the step, close to a straight line in the square root of it
while the step is short and flattening at the longest steps. With the correction the same
measurement sits on the exact value until the step is long enough for the Euler step itself
to matter, which is where the corrected series turns down. Everything else in this package
that reads a crossing off a grid, `exit_times` and `simulate_weekly` alike, carries the same
correction by default.
"""),
        _citation_cell(),
    ]


def _poisson_to_negative_binomial() -> list[NotebookNode]:
    """Return the cells of the notebook on relapse counts.

    Returns
    -------
    list of nbformat.NotebookNode
        The cells, in order.
    """
    return [
        _markdown("""
# From Poisson counts to a negative binomial

A relapse trial counts the relapses of each patient over a window and compares the counts of
two arms. This notebook builds the count distribution from the bottom up: a cohort in which
every patient carries the same onset rate gives Poisson counts, a cohort whose rates are
spread over a gamma distribution gives negative binomial counts of dispersion 1 / k, and a
cohort whose rate changes with time inside each patient gives over-dispersed counts of its
own. The last section separates the two sources, which no single count can do.
"""),
        _markdown("""
## Provenance

**No clinical data are used in this notebook.** Every cohort below is simulated with the
alternating renewal engine of `msrelapse.renewal`, which draws durations from their target
means and has no potential behind it. The two mean durations it is aimed at are the ones the
article prints, read from `msrelapse.PAPER`; everything else, the window and the spread of
the rates, is chosen here and named as such.
"""),
        _code("""
%matplotlib inline

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import msrelapse
from msrelapse import plots

# Every random step of this notebook is drawn from this one seed.
SEED = 20130910
# Patients per cohort and the counting window, in weeks. Neither is from the
# article: five years of follow up puts a few relapses on each patient, which is
# what a count distribution needs to be visible at all.
N_PATIENTS = 400
WINDOW = 260.0

lam, mu = msrelapse.rates_from_means(
    msrelapse.PAPER.tau_health_cohort_weeks.value,
    msrelapse.PAPER.tau_no_health_cohort_weeks.value,
)
onset_rate = msrelapse.effective_onset_rate(lam, mu)
print(f"remission ends at {lam:.5f} per week, relapses end at {mu:.5f} per week")
print(f"relapse onsets arrive at {onset_rate:.5f} per week, {onset_rate * WINDOW:.2f} per window")
"""),
        _markdown("""
## One rate for everyone gives Poisson counts

The onsets of an alternating renewal record arrive at `1 / (tau_health + tau_relapse)`, and
over a window many times longer than one cycle their count is close to Poisson. It is not
exactly Poisson: no relapse can start while the previous one is still running, so the counts
are slightly under-dispersed, and the fit below reports a dispersion of zero rather than a
positive one.
"""),
        _code("""
fixed_events = msrelapse.alternating_renewal(
    lam, mu, WINDOW, n=N_PATIENTS, rng=SEED, start_state="health"
)
fixed_counts = msrelapse.relapse_counts(fixed_events)
fixed_fit = msrelapse.fit_nb_counts(fixed_counts)

print(f"mean {fixed_fit.mean:.3f}, variance {float(np.var(fixed_counts)):.3f}")
print(f"dispersion {fixed_fit.dispersion:.4f}, p value against Poisson {fixed_fit.p_value:.3f}")
plots.fig_poisson_to_nb(fixed_counts)
plt.show()
"""),
        _markdown("""
## A gamma spread of rates gives a negative binomial

Now give each patient their own onset rate, drawn from a gamma distribution of shape k and
scale theta. The counts are then Poisson given the rate and negative binomial once the rate
is averaged out, with mean `k theta T` and dispersion `1 / k`. The shape below is chosen to
give a strongly spread cohort while keeping its mean rate at the rate above.
"""),
        _code("""
K = 1.5
theta = onset_rate / K
rates = msrelapse.gamma_rates(K, theta, N_PATIENTS, rng=SEED)


def remission_rate(onset, relapse_rate):
    \"\"\"Return the remission rate whose long run onset rate is `onset`.

    One onset costs one remission and one relapse, so an onset rate is reachable
    only below the rate at which relapses end. A reader who raises K, or the
    number of patients, draws from further out in the tail of the gamma and can
    walk into that limit, which is what the check below reports.
    \"\"\"
    highest = float(np.max(onset))
    if highest >= relapse_rate:
        raise ValueError(
            f"an onset rate of {highest:.4g} per week is out of reach with relapses "
            f"ending at {relapse_rate:.4g} per week: one onset costs one remission and "
            f"one relapse, so the onset rate stays below the relapse rate"
        )
    return 1.0 / (1.0 / onset - 1.0 / relapse_rate)


gamma_events = msrelapse.alternating_renewal(
    remission_rate(rates, mu), mu, WINDOW, n=N_PATIENTS, rng=SEED + 1, start_state="health"
)
gamma_counts = msrelapse.relapse_counts(gamma_events)
gamma_fit = msrelapse.fit_nb_counts(gamma_counts)

print(f"mean {gamma_fit.mean:.3f}, variance {float(np.var(gamma_counts)):.3f}")
low, high = gamma_fit.dispersion_ci
print(f"dispersion {gamma_fit.dispersion:.3f}, 95% interval {low:.3f} to {high:.3f}")
print(f"p value against Poisson {gamma_fit.p_value:.3g}")
plots.fig_poisson_to_nb(gamma_counts)
plt.show()
"""),
        _markdown("""
## Two sources of over-dispersion, and how to tell them apart

Counts spread wider than a Poisson for two quite different reasons. Patients may differ from
one another, which is the gamma mixing above and persists for as long as the patient is
followed. Or a patient's own rate may change with time, which spreads the counts of a short
window while leaving the count of a long one alone. The third cohort below has no spread
between patients at all: every patient carries the same rate on average, but half of them run
fast in the first half of follow up and slow in the second, and half the other way round.
"""),
        _code("""
SWING = 0.6
half = WINDOW / 2.0

swing_generator = np.random.default_rng(SEED + 2)
swing = swing_generator.choice([-SWING, SWING], size=N_PATIENTS)
first_events = msrelapse.alternating_renewal(
    remission_rate(onset_rate * (1.0 + swing), mu),
    mu,
    half,
    n=N_PATIENTS,
    rng=SEED + 3,
    start_state="health",
)
second_events = msrelapse.alternating_renewal(
    remission_rate(onset_rate * (1.0 - swing), mu),
    mu,
    half,
    n=N_PATIENTS,
    rng=SEED + 4,
    start_state="health",
)
varying_counts = msrelapse.relapse_counts(first_events) + msrelapse.relapse_counts(second_events)
print(f"mean {float(varying_counts.mean()):.3f}, variance {float(np.var(varying_counts)):.3f}")
"""),
        _code("""
halves = {
    "one rate for everyone": (
        msrelapse.relapse_counts(fixed_events, window=half),
        fixed_counts - msrelapse.relapse_counts(fixed_events, window=half),
    ),
    "gamma spread between patients": (
        msrelapse.relapse_counts(gamma_events, window=half),
        gamma_counts - msrelapse.relapse_counts(gamma_events, window=half),
    ),
    "rate varying inside each patient": (
        msrelapse.relapse_counts(first_events),
        msrelapse.relapse_counts(second_events),
    ),
}

source_rows = []
for label, (first_half, second_half) in halves.items():
    whole = first_half + second_half
    source_rows.append(
        (
            label,
            msrelapse.fit_nb_counts(whole).dispersion,
            msrelapse.fit_nb_counts(first_half).dispersion,
            float(np.corrcoef(first_half, second_half)[0, 1]),
        )
    )

sources = pd.DataFrame(
    source_rows,
    columns=["cohort", "dispersion_whole_window", "dispersion_first_half", "correlation_halves"],
)
sources
"""),
        _markdown("""
The whole window tells the first two cohorts apart and says nothing about the third, whose
counts are as Poisson as the cohort that has one rate for everyone: every patient has the
same expected total, however that total is spread over the two halves. Read on a half window
the third cohort is over-dispersed as well, and the two sources then part company in the
correlation between the halves. A spread between patients makes a patient who relapsed often
in the first half relapse often in the second, so the correlation is positive; a rate that
swings inside the patient does the opposite.
"""),
        _markdown("""
## What the two cohorts do to the chance of staying relapse free

The same distinction shows in the share of patients who reach the end of a window with no
relapse at all. With one rate for everyone that share is `exp(-lam T)`; with rates spread
over a gamma it is `(1 + theta T)` raised to minus k, which falls more slowly because the
patients with a low rate dominate the survivors.

Both curves are laws of the waiting time to the first relapse, so the rate they carry is the
rate at which a remission ends and not the long run onset rate of the sections above: the two
differ by the few percent of each cycle that is spent in relapse, during which no new relapse
can start. The curve of the one rate cohort is drawn at that remission rate, which is exactly
what the engine draws the first waiting time from. The gamma curve keeps the spread of onset
rates the cohort was built with, which leaves it a few percent off the law its points follow,
far inside the scatter of 400 patients.
"""),
        _code("""
def first_onset_survival(events, grid):
    \"\"\"Return the share of patients whose first relapse comes after each window.\"\"\"
    first = events.groupby("patient_id", sort=True)["relapse_onset"].min()
    onsets = first.to_numpy(dtype=float)
    # A patient who never relapsed has a missing onset, which compares False
    # against every window and so counts as still free of relapse throughout.
    return np.array([float(np.mean(~(onsets <= point))) for point in grid])


grid = np.linspace(0.0, WINDOW, 60)
panel = plt.subplots(figsize=(7.0, 4.2), layout="constrained")[1]
panel.plot(grid, msrelapse.relapse_free_curve(grid, lam=lam), label="exp(-lam T)")
panel.plot(grid, msrelapse.relapse_free_curve(grid, k=K, theta=theta), label="(1 + theta T)^-k")
panel.plot(
    grid,
    first_onset_survival(fixed_events, grid),
    linestyle="none",
    marker="o",
    markersize=4,
    label="one rate for everyone",
)
panel.plot(
    grid,
    first_onset_survival(gamma_events, grid),
    linestyle="none",
    marker="s",
    markersize=4,
    label="gamma spread between patients",
)
panel.set_xlabel("window (weeks)")
panel.set_ylabel("share of patients with no relapse yet")
panel.legend(fontsize="small")
plt.show()
"""),
        _markdown("""
## The mapping, predicted and fitted

`nb_from_gamma` turns the two parameters of the rate distribution into the two moments of the
count distribution. The table puts what it predicts beside what the counts of the simulated
cohort give back.
"""),
        _code("""
predicted_mean, predicted_dispersion = msrelapse.nb_from_gamma(K, theta, WINDOW)
mapping = pd.DataFrame(
    [
        ("mean count", predicted_mean, gamma_fit.mean),
        ("dispersion", predicted_dispersion, gamma_fit.dispersion),
    ],
    columns=["quantity", "nb_from_gamma", "fit_nb_counts"],
)
ci_low, ci_high = gamma_fit.dispersion_ci
print(f"the fitted dispersion carries the 95% interval {ci_low:.3f} to {ci_high:.3f}")
mapping
"""),
        _citation_cell(),
    ]


def _virtual_cohort_for_trial_design() -> list[NotebookNode]:
    """Return the cells of the notebook on trial design.

    Returns
    -------
    list of nbformat.NotebookNode
        The cells, in order.
    """
    return [
        _markdown("""
# A virtual cohort for trial design

This notebook runs a two arm virtual trial on cohorts generated from the durations the
article reports: 150 patients per arm, two years of follow up, and a treatment that cuts the
relapse onset rate to seven tenths of the control rate. It reports the annualised relapse
rate of each arm under three intervals, compares the two arms with a negative binomial rate
ratio, and builds a power curve by repeating the whole trial, which is then put beside the
sample size formula for the same inputs.
"""),
        _markdown("""
## Provenance

**No clinical data are used in this notebook, and nothing in it is a trial protocol.** Both
arms are simulated with the alternating renewal engine, aimed at the two mean durations the
article prints. The treatment effect is imposed by hand and is not an effect the article
reports or a claim about any drug. The sample size formula assumes one fixed analysis of
complete follow up, with no dropout, no adjustment and no interim look, and is here as an
illustration of what the rate, the spread and the length of follow up cost, not as a design
tool.
"""),
        _code("""
%matplotlib inline

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import msrelapse

# Every random step of this notebook is drawn from this one seed.
SEED = 20130910
# The trial. None of these five numbers is from the article.
RATE_RATIO = 0.7
N_PER_ARM = 150
FOLLOWUP_WEEKS = 104.0
ALPHA_LEVEL = 0.05
N_BOOT = 1000

FOLLOWUP_YEARS = FOLLOWUP_WEEKS / msrelapse.WEEKS_PER_YEAR
TAU_HEALTH = msrelapse.PAPER.tau_health_cohort_weeks.value
TAU_RELAPSE = msrelapse.PAPER.tau_no_health_cohort_weeks.value
"""),
        _markdown("""
## The two arms

One relapse onset costs one remission and one relapse, so the onset rate is
`1 / (tau_health + tau_relapse)`. Cutting it to seven tenths therefore means lengthening the
cycle by that factor with the relapse itself untouched, which is what the treated arm is
aimed at below. The records are weekly, as the records of the study are, and every patient
starts in remission rather than in a relapse, so that the first onset of a record is not
forced to week zero.
"""),
        _code("""
TAU_HEALTH_TREATED = (TAU_HEALTH + TAU_RELAPSE) / RATE_RATIO - TAU_RELAPSE
control_onset = msrelapse.effective_onset_rate(*msrelapse.rates_from_means(TAU_HEALTH, TAU_RELAPSE))
treated_onset = msrelapse.effective_onset_rate(
    *msrelapse.rates_from_means(TAU_HEALTH_TREATED, TAU_RELAPSE)
)

print(f"control remission mean {TAU_HEALTH:.1f} weeks, treated {TAU_HEALTH_TREATED:.1f} weeks")
print(f"onset rates {control_onset:.5f} and {treated_onset:.5f} per week")
print(f"their ratio is {treated_onset / control_onset:.3f}")
"""),
        _code("""
def arm_spec(n, tau_health):
    \"\"\"Return the description of one trial arm.\"\"\"
    return msrelapse.CohortSpec(
        n=n,
        tau_health=tau_health,
        tau_relapse=TAU_RELAPSE,
        followup_weeks=FOLLOWUP_WEEKS,
        engine="renewal",
        weekly=True,
        start_state="health",
    )


trial_generator = np.random.default_rng(SEED)
control = msrelapse.generate(arm_spec(N_PER_ARM, TAU_HEALTH), rng=trial_generator)
treated = msrelapse.generate(arm_spec(N_PER_ARM, TAU_HEALTH_TREATED), rng=trial_generator)
print(control)
print(treated)
"""),
        _markdown("""
## The annualised relapse rate of each arm

The three intervals answer three questions. The exact Poisson interval takes every relapse to
be an independent event at a rate the whole arm shares. The negative binomial interval widens
it by the spread of the per patient counts. The bootstrap resamples patients and keeps each
patient's count and follow up together, which is the one to read when the follow up is
unequal. These arms are generated from a single shared rate, so the three come out close
together; a real cohort spreads them apart.
"""),
        _code("""
# One generator through the loop, so that the bootstrap of the two arms does not
# resample both of them on the same table of patient indices.
bootstrap_generator = np.random.default_rng(SEED)
arr_rows = []
for name, cohort in (("control", control), ("treated", treated)):
    for method in ("poisson_exact", "nb", "bootstrap"):
        rate = msrelapse.arr(cohort.events, ci=method, n_boot=N_BOOT, rng=bootstrap_generator)
        arr_rows.append(
            (
                name,
                method,
                rate.n_patients,
                rate.n_relapses,
                rate.patient_years,
                rate.arr,
                rate.ci_low,
                rate.ci_high,
            )
        )

rates = pd.DataFrame(
    arr_rows,
    columns=["arm", "ci", "n_patients", "n_relapses", "patient_years", "arr", "ci_low", "ci_high"],
)
rates
"""),
        _code("""
control_arr = msrelapse.arr(control.events)
comparison = msrelapse.compare_arr(control.events, None, treated.events, None, model="nb")

print(f"rate ratio {comparison.rate_ratio:.3f}")
print(f"95% interval {comparison.ci_low:.3f} to {comparison.ci_high:.3f}")
print(f"p value {comparison.p_value:.4g}, fitted dispersion {comparison.dispersion:.4f}")
print(f"arm rates {comparison.arr_a:.3f} and {comparison.arr_b:.3f} per year")
"""),
        _markdown("""
## The power curve

The whole trial is repeated 200 times at each of four sizes, and the share of repeats that
reject at the 5 percent level is the power. One generator runs through every repeat, so the
curve is reproducible from the seed above and no two arms and no two trials share a draw. The
sample size formula is drawn beside it at two dispersions: the one this comparison fitted,
which is zero because every simulated patient carries the same rate, and the dispersion of a
cohort spread as widely as the gamma mixture of the previous notebook.
"""),
        _code("""
N_GRID = (60, 100, 150, 250)
# Trials behind each point of the curve. The share it measures carries a
# standard error of about three percent here; four times as many trials halves
# it and costs four times as much.
N_TRIALS = 200


def one_trial(n, generator):
    \"\"\"Return the p value of one two arm trial of n patients per arm.\"\"\"
    arm_a = msrelapse.generate(arm_spec(n, TAU_HEALTH), rng=generator)
    arm_b = msrelapse.generate(arm_spec(n, TAU_HEALTH_TREATED), rng=generator)
    return msrelapse.compare_arr(arm_a.events, None, arm_b.events, None, model="nb").p_value


power_generator = np.random.default_rng(SEED)
power_rows = []
for n in N_GRID:
    p_values = np.array([one_trial(n, power_generator) for _ in range(N_TRIALS)])
    power_rows.append((n, float(np.mean(p_values < ALPHA_LEVEL))))

power = pd.DataFrame(power_rows, columns=["n_per_arm", "power"])
power
"""),
        _code("""
POWERS = (0.5, 0.6, 0.7, 0.8, 0.9, 0.95)
# The dispersion of a cohort whose rates follow a gamma of shape 1.5, which is
# the mixture of the counts notebook.
SPREAD_DISPERSION = 1.0 / 1.5

formula_rows = []
for target in POWERS:
    for dispersion in (comparison.dispersion, SPREAD_DISPERSION):
        formula_rows.append(
            (
                dispersion,
                msrelapse.sample_size_arr(
                    control_arr.arr,
                    RATE_RATIO,
                    dispersion,
                    FOLLOWUP_YEARS,
                    power=target,
                    alpha=ALPHA_LEVEL,
                ),
                target,
            )
        )

formula = pd.DataFrame(formula_rows, columns=["dispersion", "n_per_arm", "power"])
formula
"""),
        _code("""
panel = plt.subplots(figsize=(7.0, 4.2), layout="constrained")[1]
panel.plot(
    power["n_per_arm"],
    power["power"],
    marker="o",
    color="black",
    label=f"{N_TRIALS} simulated trials",
)
for dispersion, block in formula.groupby("dispersion"):
    panel.plot(
        block["n_per_arm"],
        block["power"],
        marker="s",
        linestyle="--",
        label=f"sample_size_arr, dispersion {dispersion:.2f}",
    )
panel.axhline(0.8, linestyle=":", color="grey", label="80 percent power")
panel.set_xlim(0, 400)
panel.set_xlabel("patients per arm")
panel.set_ylabel(f"share of trials with p < {ALPHA_LEVEL}")
panel.legend(fontsize="small")
plt.show()
"""),
        _markdown("""
## Which engine to use

Both engines in this package produce the same kind of record, and they answer different
questions.

The renewal engine draws durations straight from their target means. It has no potential
behind it, it costs almost nothing, and it is the right engine for a trial calculation, for a
null model of what a cohort of a given size can resolve, and for anything where the durations
themselves are the assumption being made.

The mechanistic engine calibrates a double well to the same pair of means and integrates the
stochastic equation. It costs far more, and what it buys is a mechanism: the durations are
then a consequence of a barrier, a noise and a shape rather than an assumption, so a question
about what would happen if the barrier moved, if the two wells were tilted differently, or if
the noise rose, has somewhere to be asked. It is also the engine that shows where the
phenomenological picture breaks down, as the exit time notebook does when it finds the
relapse well of the calibrated potential too shallow to give exponential durations at all.

Use the renewal engine to size a trial, and the mechanistic engine to ask why the record
looks the way it does.
"""),
        _citation_cell(),
    ]


_BUILDERS: Final = {
    "01_reproduce_bordi2013.ipynb": _reproduce_bordi2013,
    "02_sde_to_exponential.ipynb": _sde_to_exponential,
    "03_poisson_to_negative_binomial.ipynb": _poisson_to_negative_binomial,
    "04_virtual_cohort_for_trial_design.ipynb": _virtual_cohort_for_trial_design,
}

NOTEBOOK_NAMES: Final = tuple(_BUILDERS)
"""The file name of each notebook this script builds, in reading order."""


def build(name: str) -> NotebookNode:
    """Return one notebook, with empty outputs.

    Parameters
    ----------
    name : str
        File name of the notebook, one of :data:`NOTEBOOK_NAMES`.

    Returns
    -------
    nbformat.NotebookNode
        The notebook, validated against the nbformat schema. Every cell carries
        its position as its identifier, so that two builds of the same cells
        give the same file, byte for byte. The identifiers nbformat draws by
        default are random, which would put every cell of every notebook in the
        diff of any rebuild. The position is also exactly what the nbstripout
        hook of the repository writes over a cell identifier, so the hook leaves
        a freshly built notebook alone.

    Raises
    ------
    ValueError
        If `name` is not one of the four notebooks this script builds.
    """
    if name not in _BUILDERS:
        raise ValueError(f"name must be one of {NOTEBOOK_NAMES}, got {name!r}")
    notebook = _new_notebook(cells=_BUILDERS[name]())
    for index, cell in enumerate(notebook.cells):
        cell["id"] = str(index)
    notebook.metadata.update(_NOTEBOOK_METADATA)
    nbformat.validate(notebook)
    return notebook


def _without_outputs(notebook: NotebookNode) -> NotebookNode:
    """Return a copy of a notebook whose code cells carry no output.

    Parameters
    ----------
    notebook : nbformat.NotebookNode
        The notebook to copy.

    Returns
    -------
    nbformat.NotebookNode
        The copy, with an empty output list and no execution count on every
        code cell. This is what the repository commits, and it is what the
        nbstripout hook of the repository would leave behind in any case.
    """
    cleared = copy.deepcopy(notebook)
    for cell in cleared.cells:
        if cell.cell_type == "code":
            cell.outputs = []
            cell.execution_count = None
    return cleared


def build_all() -> dict[str, NotebookNode]:
    """Return every notebook this script builds, keyed by file name.

    Returns
    -------
    dict of str to nbformat.NotebookNode
        The four notebooks, in reading order.
    """
    return {name: build(name) for name in NOTEBOOK_NAMES}


def write(name: str, notebook: NotebookNode, out_dir: Path | None = None) -> Path:
    """Write one notebook to disk, with every output cleared.

    Parameters
    ----------
    name : str
        File name to write under.
    notebook : nbformat.NotebookNode
        The notebook to write. It is not changed; the outputs are cleared on a
        copy, so that a notebook that was run stays runnable in memory and the
        file on disk carries no output and no run count.
    out_dir : pathlib.Path, optional
        Directory to write into, created if it does not exist. The default is
        :data:`NOTEBOOK_DIR`.

    Returns
    -------
    pathlib.Path
        The file that was written.
    """
    directory = NOTEBOOK_DIR if out_dir is None else out_dir
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    notebook = _without_outputs(notebook)
    _write_notebook(notebook, path)
    return path


def _execute(notebook: NotebookNode, work_dir: Path) -> None:
    """Run every cell of a notebook and raise if one of them fails.

    Parameters
    ----------
    notebook : nbformat.NotebookNode
        The notebook to run. A copy is executed, so the outputs never reach the
        notebook that is written.
    work_dir : pathlib.Path
        Directory the kernel runs in.

    Raises
    ------
    nbclient.exceptions.CellExecutionError
        If any cell raises.
    """
    # Imported here rather than at the top of the file, because building the
    # notebooks needs nbformat alone and nbclient is only used by --execute.
    from nbclient import NotebookClient  # noqa: PLC0415

    client = NotebookClient(
        copy.deepcopy(notebook),
        timeout=EXECUTION_TIMEOUT,
        kernel_name=KERNEL_NAME,
        resources={"metadata": {"path": str(work_dir)}},
    )
    client.execute()


def main(argv: list[str] | None = None) -> int:
    """Build the notebooks, optionally run them, and write them without outputs.

    Parameters
    ----------
    argv : list of str, optional
        Command line arguments. The default reads them from ``sys.argv``.

    Returns
    -------
    int
        0 when every notebook was written.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--execute",
        action="store_true",
        help="run each notebook before writing it, to check that it still works",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=NOTEBOOK_DIR,
        help="directory to write the notebooks into (default: the notebooks directory)",
    )
    arguments = parser.parse_args(argv)
    arguments.out_dir.mkdir(parents=True, exist_ok=True)
    for name, notebook in build_all().items():
        if arguments.execute:
            print(f"running {name}")
            # Run where the test suite runs them, which is the directory the
            # notebooks live in, rather than in the directory they are written
            # to, so that a rebuild into a scratch directory is checked under
            # the conditions the committed files are checked under.
            _execute(notebook, NOTEBOOK_DIR)
        print(f"writing {write(name, notebook, arguments.out_dir)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
