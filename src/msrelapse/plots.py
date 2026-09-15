"""The figures of the paper, drawn from the data structures of this package.

matplotlib is an optional dependency, the ``plot`` extra, so it is imported
inside the functions that draw rather than at the top of the module. Importing
:mod:`msrelapse.plots` therefore works in an install without it, and only a call
that has to create a figure raises, naming the extra to install.

Every function takes the axes to draw on and gives them back, so that a caller
can compose the panels into a figure of their own; passing None instead creates
a figure of the size the panels want. A figure of several panels takes and
returns a tuple of axes, one per panel. Whatever a function is given is checked
before it creates anything, so a call that is refused leaves no figure behind in
the global registry of pyplot, which the caller has no handle on and would have
to close by number.

Numbers taken from the article, the bin edges of Figure 3 and the digitised bar
heights of Figures 3 and 4 among them, are read from ``PAPER`` in
:mod:`msrelapse._params` and are never written here. What the paper does not
fix, and what is therefore chosen here, is the resolution of the potential
curves, the time step of the simulated paths of Figure 7 and the placement of
the labels and arrows inside a panel.

Two of the figures are not in the article at all.
:func:`fig_survival_vs_exponential` puts the observed survival of one state
beside the exponential fitted to it, and :func:`fig_poisson_to_nb` puts the
relapse counts of a cohort beside the Poisson and the negative binomial mass.
The article reports no fit, no test and no interval of any kind, as section 7 of
``docs/paper_facts.md`` records, so these two belong to the methods note rather
than to the reproduction.

References
----------
I. Bordi, R. Umeton, V. A. G. Ricigliano, et al., "A mechanistic, stochastic
model helps understand multiple sclerosis course and pathogenesis",
International Journal of Genomics, 2013, doi 10.1155/2013/910321.
"""

from __future__ import annotations

import math
import os
from collections.abc import Callable, Sequence
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING, Final, cast

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy import stats

from msrelapse._params import PAPER
from msrelapse.fit import _MIN_AT_RISK, fit_durations, fit_nb_counts
from msrelapse.io import validate
from msrelapse.model import CriticalPoints, DoubleWell
from msrelapse.simulate import Seed, simulate_paths

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

__all__ = [
    "fig2_sample_patients",
    "fig3_rr_phase_histogram",
    "fig4_duration_histograms",
    "fig5_symmetric_potentials",
    "fig6_asymmetric_potential",
    "fig7_simulated_paths",
    "fig8_patient_potentials",
    "fig_poisson_to_nb",
    "fig_survival_vs_exponential",
    "save_all_paper_figures",
]

_Vector = npt.NDArray[np.float64]

_NO_HEALTH: Final = PAPER.state_no_health.value
_HEALTH: Final = PAPER.state_health.value

_X_LIMITS: Final = PAPER.potential_plot_x_limits.value
_V_LIMITS: Final = PAPER.potential_plot_v_limits.value

_FIG5_ALPHAS: Final = (PAPER.alpha_reference.value, PAPER.alpha_illustrative_low.value)
"""The two control parameters Figure 5 and Figure 6 are drawn at."""

_FIG7_BETAS: Final[tuple[float | None, ...]] = (PAPER.beta_symmetric.value, None)
"""The two asymmetries of Figure 7, the second of them the illustrative one."""

_SAMPLE_PATIENTS: Final = ("Patient 23", "Patient 32", "Patient 53")
"""The sample patients of Section 3.4, named as the paper names them.

Figure 8 draws their fitted potentials and titles its panels with these names.
"""

# How many patients Figure 2 of the paper shows, and so how many this figure
# takes from a frame when the caller names none.
_FIG2_N_PATIENTS: Final = 3

# How many identifiers an error message lists before it stops. A display choice,
# not a number of the paper.
_PREVIEW_PATIENTS: Final = 3

_PATIENT_BETAS: Final = (
    PAPER.beta_patient_23.value,
    PAPER.beta_patient_32.value,
    PAPER.beta_patient_53.value,
)
"""The asymmetry Figure 8 fits to each sample patient, in the order above."""

# The x axis of Figure 4(b) starts at zero, unlike Figure 4(a), whose first bin
# starts at the shortest relapse the weekly scale of the study can record.
_FIG4B_BIN_START: Final = 0.0

# Choices this module makes, none of them from the paper. The time step is the
# one msrelapse.simulate measured its bridge correction and its episode lengths
# at; simulate_paths itself defaults to a step of 0.01 and refuses any step above
# its MAX_DT of 0.3. The rest are cosmetic.
_FIG7_DT: Final = 0.02
_MAX_BIN_TICKS: Final = 11
_POTENTIAL_POINTS: Final = 601
_SURVIVAL_POINTS: Final = 200
_STATE_MARGIN: Final = 0.5
_LABEL_OFFSET: Final = 0.06
_INSET_BOX: Final = (0.52, 0.55, 0.45, 0.40)
_GUIDE_COLOUR: Final = "grey"
_CURVE_COLOUR: Final = "black"
_STACKED_PANEL_SIZE: Final = (7.0, 2.2)
_SIDE_PANEL_SIZE: Final = (4.2, 3.6)
_SINGLE_SIZE: Final = (6.0, 4.0)
_DPI: Final = 150


def fig2_sample_patients(
    weekly: pd.DataFrame,
    patient_ids: Sequence[str] | None = None,
    axes: Sequence[Axes] | None = None,
) -> tuple[Axes, ...]:
    """Draw the weekly record of a few patients as step functions, Figure 2.

    Each panel holds the plus one and minus one series of one patient against
    the week, with the two levels labelled as the paper labels them.

    Parameters
    ----------
    weekly : pandas.DataFrame
        A frame in the weekly schema of :mod:`msrelapse.io`. It is validated
        before use.
    patient_ids : sequence of str, optional
        Which patients to draw, one panel each. The default takes the first
        three of the frame, as the paper shows three sample patients.
    axes : sequence of matplotlib.axes.Axes, optional
        One Axes per patient. The default creates a stack of panels.

    Returns
    -------
    tuple of matplotlib.axes.Axes
        The panels that were drawn on, in the order of the patients.

    Raises
    ------
    ImportError
        If matplotlib is not installed and `axes` is None.
    ValueError
        If `weekly` does not obey the weekly schema, if it holds no patient, if
        `patient_ids` is empty or names a patient the frame does not hold, or
        if `axes` does not have one Axes per patient.
    """
    validate(weekly, "weekly")
    chosen = _chosen_patients(weekly, patient_ids)
    panels = _panel_axes(axes, len(chosen), stacked=True)
    for panel, patient in zip(panels, chosen, strict=True):
        record = weekly.loc[weekly["patient_id"] == patient]
        weeks = record["week"].to_numpy(dtype=np.float64)
        states = record["state"].to_numpy(dtype=np.float64)
        # Week k covers the interval from k to k + 1, so the step is closed one
        # week past the last one. Without that point the final week of a record
        # would be drawn with no width at all.
        end = float(weeks[-1]) + 1.0
        panel.step(
            np.append(weeks, end),
            np.append(states, states[-1]),
            where="post",
            linewidth=1.0,
            color=_CURVE_COLOUR,
        )
        panel.set_xlim(float(weeks[0]), end)
        panel.set_yticks([_HEALTH, _NO_HEALTH])
        panel.set_yticklabels(["Health", "No health"])
        panel.set_ylim(_HEALTH - _STATE_MARGIN, _NO_HEALTH + _STATE_MARGIN)
        panel.set_xlabel("Time (week)")
        panel.set_title(patient)
    return panels


def fig3_rr_phase_histogram(
    followup_weeks: npt.ArrayLike | None = None,
    ax: Axes | None = None,
) -> Axes:
    """Draw the histogram of relapsing-remitting phase lengths, Figure 3.

    The bins are the ones the paper's x axis is ticked at, eight bins of 159
    weeks from 40 to 1312.

    Parameters
    ----------
    followup_weeks : array_like, optional
        One record length in weeks per patient. The default draws the bar
        heights digitised from the figure of the paper instead.
    ax : matplotlib.axes.Axes, optional
        The panel to draw on. The default creates a figure.

    Returns
    -------
    matplotlib.axes.Axes
        The panel that was drawn on.

    Raises
    ------
    ImportError
        If matplotlib is not installed and `ax` is None.
    ValueError
        If `followup_weeks` is given and is not one finite record length per
        patient.

    Notes
    -----
    A record shorter than the first bin edge or longer than the last one falls
    outside the bins of the paper and is not drawn. The panel title reports how
    many of the records were drawn as well as how many were given, so that the
    bars and the count beside them always agree.

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> ax = fig3_rr_phase_histogram()
    >>> ax.get_xlabel()
    'Period of analysis (week)'
    >>> plt.close("all")
    """
    edges = np.asarray(PAPER.fig3_bin_edges_weeks.value, dtype=np.float64)
    if followup_weeks is None:
        counts = np.asarray(PAPER.fig3_counts.value, dtype=np.float64)
        title = "Digitised from Figure 3 of the paper"
    else:
        values = _record_lengths(followup_weeks)
        counts = _bin_counts(values, edges)
        title = f"This cohort, {int(counts.sum())} of {values.size} patient(s) inside the bins"
    # The bars are counted before the panel is created, so that a record length
    # this figure cannot bin is refused without leaving a figure open.
    panel = _single_axes(ax)
    _draw_bars(panel, edges, counts)
    panel.set_xticks(edges)
    panel.set_xlabel("Period of analysis (week)")
    panel.set_ylabel("Counts (number of patients)")
    panel.set_title(title)
    return panel


def fig4_duration_histograms(
    durations: pd.DataFrame | None = None,
    axes: Sequence[Axes] | None = None,
) -> tuple[Axes, Axes]:
    """Draw the two episode duration histograms, Figure 4.

    The first panel holds the no health durations in bins of 2.5 weeks from one
    week, the second the health durations in bins of 100 weeks from zero, and
    each carries the number of events and their mean duration.

    Parameters
    ----------
    durations : pandas.DataFrame, optional
        A frame in the durations schema of :mod:`msrelapse.io`, validated
        before use. The default draws the bar heights digitised from the figure
        of the paper instead.
    axes : sequence of matplotlib.axes.Axes, optional
        Two Axes, one per panel. The default creates a figure of two panels
        side by side.

    Returns
    -------
    tuple of matplotlib.axes.Axes
        The no health panel and the health panel.

    Raises
    ------
    ImportError
        If matplotlib is not installed and `axes` is None.
    ValueError
        If `durations` does not obey the durations schema, if it holds no run
        of one of the two states, or if `axes` does not hold two Axes.

    Notes
    -----
    Every recorded run of a state is counted, the censored final remission of a
    patient included, because those are the counts and the means the paper
    reports.
    """
    if durations is None:
        first, second = _panel_axes(axes, 2, stacked=False)
        _draw_digitised_durations(first, _NO_HEALTH)
        _draw_digitised_durations(second, _HEALTH)
        return first, second
    validate(durations, "durations")
    # Both states are selected before the panels are created, so that a frame
    # holding no run of one of them is refused without leaving a figure open.
    relapses = _state_durations(durations, _NO_HEALTH)
    remissions = _state_durations(durations, _HEALTH)
    first, second = _panel_axes(axes, 2, stacked=False)
    _draw_observed_durations(first, relapses, _NO_HEALTH)
    _draw_observed_durations(second, remissions, _HEALTH)
    return first, second


def fig5_symmetric_potentials(
    alphas: Sequence[float] = _FIG5_ALPHAS,
    axes: Sequence[Axes] | None = None,
) -> tuple[Axes, ...]:
    """Draw the symmetric double well at one or more control parameters, Figure 5.

    Each panel holds equation (2), that is the potential at an asymmetry of
    zero, with dashed guides at the top of the barrier and at the bottom of the
    two wells and a double headed arrow for the barrier between them.

    Parameters
    ----------
    alphas : sequence of float, optional
        One control parameter per panel. The default is the pair the paper
        draws, its reference value and the lower illustrative one.
    axes : sequence of matplotlib.axes.Axes, optional
        One Axes per control parameter. The default creates a figure of panels
        side by side.

    Returns
    -------
    tuple of matplotlib.axes.Axes
        The panels that were drawn on.

    Raises
    ------
    ImportError
        If matplotlib is not installed and `axes` is None.
    ValueError
        If `alphas` is empty or holds a value that is not positive and finite,
        or if `axes` does not have one Axes per control parameter.
    """
    # The potentials are built before the panels are created, so that a control
    # parameter the model refuses leaves no figure open.
    wells = tuple(_well_and_points(float(alpha), PAPER.beta_symmetric.value) for alpha in alphas)
    panels = _panel_axes(axes, len(wells), stacked=False)
    for panel, (well, points) in zip(panels, wells, strict=True):
        _draw_potential(panel, well)
        top = float(well.V(points.saddle))
        bottom = float(well.V(points.relapse))
        _guide_line(panel, top)
        _guide_line(panel, bottom)
        _barrier_arrow(panel, 0.5 * (points.saddle + points.relapse), top, bottom, r"$\Delta V$")
        _mark_critical_points(panel, well, points)
        _annotate_parameters(panel, well.alpha, well.beta)
    return panels


def fig6_asymmetric_potential(
    alpha: float = PAPER.alpha_reference.value,
    beta: float | None = None,
    ax: Axes | None = None,
) -> Axes:
    """Draw the asymmetric double well and both of its barriers, Figure 6.

    The panel holds equation (3), with the three stationary points marked, a
    dashed guide through each of them and a double headed arrow for each of the
    two barriers.

    Parameters
    ----------
    alpha : float, optional
        Control parameter. Defaults to the reference value of the paper.
    beta : float, optional
        Asymmetry parameter. Defaults to the illustrative value of the paper.
    ax : matplotlib.axes.Axes, optional
        The panel to draw on. The default creates a figure.

    Returns
    -------
    matplotlib.axes.Axes
        The panel that was drawn on.

    Raises
    ------
    ImportError
        If matplotlib is not installed and `ax` is None.
    ValueError
        If `alpha` is not positive and finite, or if the potential at `beta`
        has fewer than two wells.

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> ax = fig6_asymmetric_potential()
    >>> ax.get_xlabel(), ax.get_ylabel()
    ('x', 'V(x)')
    >>> plt.close("all")
    """
    asymmetry = PAPER.beta_illustrative.value if beta is None else float(beta)
    # The potential is built before the panel is created, so that one the model
    # refuses leaves no figure open.
    well, points = _well_and_points(alpha, asymmetry)
    panel = _single_axes(ax)
    _draw_asymmetric_potential(panel, well, points)
    return panel


def fig7_simulated_paths(  # noqa: PLR0917
    sigma: float | None = None,
    alpha: float = PAPER.alpha_reference.value,
    betas: Sequence[float | None] = _FIG7_BETAS,
    t_end: float = PAPER.fig7_time_span.value[1],
    dt: float = _FIG7_DT,
    rng: Seed = None,
    axes: Sequence[Axes] | None = None,
) -> tuple[Axes, ...]:
    """Draw one solution of the stochastic equation per asymmetry, Figure 7.

    Each panel holds a single path of equation (4) over the time span of the
    paper, at the noise the paper applies to both of its panels.

    Parameters
    ----------
    sigma : float, optional
        Noise amplitude. The default is the square root of the noise variance
        the paper reports, the only noise value in the article.
    alpha : float, optional
        Control parameter. Defaults to the reference value of the paper.
    betas : sequence of float or None, optional
        One asymmetry per panel, with None standing for the illustrative value
        of the paper. The default is the pair the paper draws, the symmetric
        potential and the illustrative asymmetric one.
    t_end : float, optional
        Length of each run, in the model time of the paper. The paper gives its
        time axis no unit and no conversion to weeks.
    dt : float, optional
        Time step of the Euler Maruyama integration. Not from the paper, which
        states no integration scheme and no step.
    rng : numpy.random.Generator or int or None, optional
        Generator to draw the noise from, or a seed for
        :func:`numpy.random.default_rng`. One generator is built for the whole
        call and drawn from once per panel, so the panels carry different
        noise and the figure as a whole is reproducible from a seed.
    axes : sequence of matplotlib.axes.Axes, optional
        One Axes per asymmetry. The default creates a stack of panels.

    Returns
    -------
    tuple of matplotlib.axes.Axes
        The panels that were drawn on.

    Raises
    ------
    ImportError
        If matplotlib is not installed and `axes` is None.
    ValueError
        If `betas` is empty, if any argument of
        :func:`msrelapse.simulate.simulate_paths` is out of range, or if `axes`
        does not have one Axes per asymmetry.

    Notes
    -----
    The traces of the paper cannot be reproduced sample for sample: it states
    no integration scheme, time step, seed, initial condition or number of
    realisations. What is reproduced is the behaviour, a path that switches
    between the two states at an asymmetry of zero and one that sits in the
    health well with brief excursions at the illustrative asymmetry.
    """
    amplitude = PAPER.noise_amplitude.value if sigma is None else float(sigma)
    resolved = tuple(
        PAPER.beta_illustrative.value if beta is None else float(beta) for beta in betas
    )
    generator = _generator(rng)
    # Every path is integrated before the panels are created, so that a step or
    # a time span the integrator refuses leaves no figure open. The generator is
    # still drawn from once per panel, in the order the panels are drawn.
    runs = tuple(
        simulate_paths(DoubleWell(alpha, beta), amplitude, t_end, dt=dt, rng=generator)
        for beta in resolved
    )
    panels = _panel_axes(axes, len(runs), stacked=True)
    for panel, beta, paths in zip(panels, resolved, runs, strict=True):
        panel.plot(paths.t, paths.x[0], linewidth=0.6, color=_CURVE_COLOUR)
        panel.set_xlim(0.0, float(paths.t[-1]))
        panel.set_ylim(*PAPER.fig7_x_limits.value)
        panel.set_xlabel("Time (model units, as in the paper)")
        panel.set_ylabel("x")
        panel.set_title(f"alpha = {alpha:g}, beta = {beta:g}, epsilon = {amplitude**2:g}")
    return panels


def fig8_patient_potentials(
    betas: Sequence[float] | None = None,
    labels: Sequence[str] | None = None,
    axes: Sequence[Axes] | None = None,
) -> tuple[Axes, ...]:
    """Draw the potential fitted to each sample patient, Figure 8.

    Every panel is drawn the way :func:`fig6_asymmetric_potential` draws one, at
    the reference control parameter of the paper, which is the only value the
    paper ever fits a patient with.

    Parameters
    ----------
    betas : sequence of float, optional
        One asymmetry per panel. The default is the three the paper prints, one
        per sample patient.
    labels : sequence of str, optional
        One panel title per asymmetry. The default names the three sample
        patients, and it may only be left out when `betas` is left out as well:
        a panel titled with a patient of the paper has to hold the asymmetry
        fitted to that patient.
    axes : sequence of matplotlib.axes.Axes, optional
        One Axes per asymmetry. The default creates a figure of panels side by
        side.

    Returns
    -------
    tuple of matplotlib.axes.Axes
        The panels that were drawn on.

    Raises
    ------
    ImportError
        If matplotlib is not installed and `axes` is None.
    ValueError
        If `betas` is given without `labels`, if `betas` and `labels` are of
        different lengths, if `betas` is empty, if a potential has fewer than
        two wells, or if `axes` does not have one Axes per asymmetry.
    """
    values = _PATIENT_BETAS if betas is None else tuple(float(beta) for beta in betas)
    if betas is not None and labels is None:
        raise ValueError(
            f"fig8_patient_potentials titles its panels with the sample patients of the "
            f"paper, so labels must be given whenever betas is; got {len(values)} asymmetry "
            f"value(s) and no labels"
        )
    names = _SAMPLE_PATIENTS if labels is None else tuple(str(label) for label in labels)
    if len(names) != len(values):
        raise ValueError(
            f"fig8_patient_potentials needs one label per panel, got {len(names)} label(s) "
            f"for {len(values)} asymmetry value(s)"
        )
    wells = tuple(_well_and_points(PAPER.alpha_reference.value, beta) for beta in values)
    panels = _panel_axes(axes, len(wells), stacked=False)
    for panel, (well, points), name in zip(panels, wells, names, strict=True):
        _draw_asymmetric_potential(panel, well, points)
        panel.set_title(name)
    return panels


def fig_survival_vs_exponential(
    durations: pd.DataFrame,
    state: int,
    ax: Axes | None = None,
    inset: bool = True,
) -> Axes:
    """Draw the observed survival of one state against the exponential fitted to it.

    The step is the Kaplan Meier estimate over the runs of `state`, in which a
    censored final remission contributes its time at risk but no event, and the
    smooth curve is the survival of the exponential
    :func:`msrelapse.fit.fit_durations` fits to the same runs. The figure is not
    in the paper, which reports no fit at all.

    Parameters
    ----------
    durations : pandas.DataFrame
        A frame in the durations schema of :mod:`msrelapse.io`. It is validated
        before use.
    state : int
        Clinical code of the state to draw, +1 for no health and -1 for health.
    ax : matplotlib.axes.Axes, optional
        The panel to draw on. The default creates a figure.
    inset : bool, optional
        Whether to add an inset holding the discrete hazard against time, which
        is flat when the durations carry no memory.

    Returns
    -------
    matplotlib.axes.Axes
        The panel that was drawn on. The inset, when there is one, is its only
        child axes.

    Raises
    ------
    ImportError
        If matplotlib is not installed and `ax` is None.
    ValueError
        If `durations` does not obey the durations schema, if `state` is
        neither clinical code, or if no run of `state` completed.

    Notes
    -----
    The inset repeats the hazard :func:`msrelapse.fit.test_memoryless` regresses
    on time: the complete durations alone, read at the weeks where at least five
    of them are still at risk. The step of the main panel is the wider estimate
    and counts a censored run as at risk until it is cut off. The dashed guide
    of the inset sits at one over the fitted mean, which is the weekly hazard of
    the fitted law however the law is read, since a duration rounded up to whole
    weeks is geometric with that success probability.

    The panel runs to the longest time any run of the state was at risk for,
    which is past the last event whenever a censored run outlived every one of
    them. Ending it at the last event instead would hide that run altogether,
    although it counts towards the fit, and would cut the fitted curve off in a
    range where it can look flat.

    A frame too small for any week to be read, which one patient of the paper's
    own size is, gets an inset that says so rather than an empty one. That is
    the same frame :func:`msrelapse.fit.test_memoryless` refuses outright; the
    step and the fitted curve of the main panel are drawn as usual.
    """
    fitted = fit_durations(durations, state)
    selected = durations[durations["state"] == state]
    values = selected["duration_w"].to_numpy(dtype=np.float64)
    censored = selected["censored"].to_numpy(dtype=bool)
    panel = _single_axes(ax)
    times, survival = _kaplan_meier(values, censored)
    # A censored run can outlive every event, and then the last step of the
    # estimate sits well short of the longest time the state was observed for.
    # The panel runs to that longest time instead, and the step is carried out
    # to it at the level it reached, so that the run is visible and the final
    # plateau has a width.
    right = float(max(times[-1], values.max()))
    if right > times[-1]:
        times = np.append(times, right)
        survival = np.append(survival, survival[-1])
    panel.step(
        times,
        survival,
        where="post",
        color=_CURVE_COLOUR,
        label=f"observed, {fitted.n} run(s), {fitted.n_censored} censored",
    )
    grid = np.linspace(0.0, right, _SURVIVAL_POINTS, dtype=np.float64)
    panel.plot(
        grid,
        np.exp(-grid / fitted.mean),
        color="tab:red",
        label=f"exponential, mean {fitted.mean:.3g} weeks",
    )
    panel.set_xlim(0.0, right)
    panel.set_ylim(0.0, 1.05)
    panel.set_xlabel("Duration (week)")
    panel.set_ylabel("Probability of lasting longer")
    panel.set_title(f"Duration of {_state_name(state)} events")
    # Lower left: the survival curve leaves that corner empty, and the inset,
    # when there is one, takes the upper right.
    panel.legend(loc="lower left")
    if inset:
        _draw_hazard_inset(panel, values[~censored], 1.0 / fitted.mean)
    return panel


def fig_poisson_to_nb(counts: npt.ArrayLike, ax: Axes | None = None) -> Axes:
    """Draw the relapse counts of a cohort against the Poisson and the NB mass.

    The bars are the share of patients at each count, the first line is the
    Poisson mass at the sample mean and the second is the negative binomial mass
    :func:`msrelapse.fit.fit_nb_counts` fits to the same counts. A cohort in
    which the onset rate varies from patient to patient sits above the Poisson
    line in both tails. The figure is not in the paper, which reports no fit.

    Parameters
    ----------
    counts : array_like
        One whole, non-negative relapse count per patient.
    ax : matplotlib.axes.Axes, optional
        The panel to draw on. The default creates a figure.

    Returns
    -------
    matplotlib.axes.Axes
        The panel that was drawn on.

    Raises
    ------
    ImportError
        If matplotlib is not installed and `ax` is None.
    ValueError
        If `counts` holds fewer than two counts, or a count that is missing,
        negative or not a whole number.

    Notes
    -----
    A cohort that carries no evidence of overdispersion is fitted as Poisson,
    with a dispersion of exactly zero, and the two masses are then the same. That
    is a result rather than a failure: the negative binomial line is dashed, so
    the Poisson line stays visible under it, and the legend reports the fitted
    dispersion.

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> ax = fig_poisson_to_nb([0, 1, 2, 1, 0])
    >>> len(ax.lines), ax.get_ylabel()
    (2, 'Share of patients')
    >>> plt.close("all")
    """
    fitted = fit_nb_counts(counts)
    observed = np.asarray(counts, dtype=np.int64).reshape(-1)
    panel = _single_axes(ax)
    grid = np.arange(int(observed.max()) + 1, dtype=np.int64)
    shares = np.bincount(observed, minlength=grid.size).astype(np.float64) / observed.size
    panel.bar(
        grid,
        shares,
        width=0.8,
        color="lightsteelblue",
        edgecolor="white",
        label=f"observed, {observed.size} patient(s)",
    )
    panel.plot(
        grid,
        _poisson_mass(grid, fitted.mean),
        marker="o",
        markersize=4,
        color=_CURVE_COLOUR,
        label=f"Poisson, mean {fitted.mean:.3g}",
    )
    panel.plot(
        grid,
        _nb_mass(grid, fitted.mean, fitted.dispersion),
        marker="s",
        markersize=4,
        # Dashed, so that the Poisson line stays visible underneath when the
        # cohort carries no overdispersion and the two masses are the same.
        linestyle="--",
        color="tab:red",
        label=f"negative binomial, dispersion {fitted.dispersion:.3g}",
    )
    panel.set_xticks(_thinned_ticks(grid.astype(np.float64)))
    panel.set_xlabel("Relapses per patient")
    panel.set_ylabel("Share of patients")
    panel.legend()
    return panel


def save_all_paper_figures(
    out_dir: str | os.PathLike[str],
    weekly: pd.DataFrame,
    durations: pd.DataFrame,
    rng: Seed = None,
) -> list[Path]:
    """Draw every figure of this module from one cohort and write them to disk.

    Figures 2 to 8 and the two figures of the methods note are drawn in that
    order, written as PNG files of fixed name at 150 dots per inch, and closed.

    Parameters
    ----------
    out_dir : str or path-like
        Directory to write into. It is created if it does not exist.
    weekly : pandas.DataFrame
        A frame in the weekly schema of :mod:`msrelapse.io`, which supplies the
        sample patients of Figure 2 and the record lengths of Figure 3.
    durations : pandas.DataFrame
        A frame in the durations schema of :mod:`msrelapse.io`, which supplies
        the histograms of Figure 4, the survival figure and the relapse counts.
    rng : numpy.random.Generator or int or None, optional
        Generator behind the simulated paths of Figure 7, or a seed for
        :func:`numpy.random.default_rng`.

    Returns
    -------
    list of pathlib.Path
        The files that were written, in the order they were drawn.

    Raises
    ------
    ImportError
        If matplotlib is not installed.
    ValueError
        If either frame breaks its schema, or if the cohort is too small for
        one of the figures, for example holding fewer than two patients for the
        count figure.

    Notes
    -----
    The survival figure is drawn for the no health state, whose durations are
    the ones the paper calls exponential in Section 3.1 and the ones its
    Figure 4(a) shows.

    A figure is drawn, written and closed before the next one is drawn, and any
    figure a failed draw left behind is closed as well, so that a call that
    raises part way through leaves no figure open that the caller has no handle
    on. The files written before the failure stay on disk.
    """
    validate(weekly, "weekly")
    validate(durations, "durations")
    directory = Path(out_dir)
    directory.mkdir(parents=True, exist_ok=True)
    lengths = weekly.groupby("patient_id", sort=True).size().to_numpy()
    counts = _relapse_counts(durations)
    drawings: list[tuple[str, Callable[[], Axes]]] = [
        ("fig2_sample_patients", lambda: fig2_sample_patients(weekly)[0]),
        ("fig3_rr_phase_histogram", lambda: fig3_rr_phase_histogram(lengths)),
        ("fig4_duration_histograms", lambda: fig4_duration_histograms(durations)[0]),
        ("fig5_symmetric_potentials", lambda: fig5_symmetric_potentials()[0]),
        ("fig6_asymmetric_potential", fig6_asymmetric_potential),
        ("fig7_simulated_paths", lambda: fig7_simulated_paths(rng=rng)[0]),
        ("fig8_patient_potentials", lambda: fig8_patient_potentials()[0]),
        ("fig_survival_vs_exponential", lambda: fig_survival_vs_exponential(durations, _NO_HEALTH)),
        ("fig_poisson_to_nb", lambda: fig_poisson_to_nb(counts)),
    ]
    plt = _pyplot()
    opened_before = set(plt.get_fignums())
    written: list[Path] = []
    try:
        for name, draw in drawings:
            written.append(_save_figure(draw(), directory / f"{name}.png"))
    finally:
        for number in sorted(set(plt.get_fignums()) - opened_before):
            plt.close(number)
    return written


def _pyplot() -> ModuleType:
    """Return the pyplot module, or explain which extra to install.

    Returns
    -------
    types.ModuleType
        The ``matplotlib.pyplot`` module.

    Raises
    ------
    ImportError
        If matplotlib is not installed.
    """
    try:
        import matplotlib.pyplot as plt  # noqa: PLC0415
    except ImportError as error:
        raise ImportError(
            "drawing a figure needs matplotlib, which is not installed; install the plot "
            "extra of this package, that is msrelapse[plot]"
        ) from error
    return plt


def _single_axes(ax: Axes | None) -> Axes:
    """Return the panel to draw on, creating a figure when none was given.

    Parameters
    ----------
    ax : matplotlib.axes.Axes or None
        The panel the caller passed in.

    Returns
    -------
    matplotlib.axes.Axes
        The panel to draw on.

    Raises
    ------
    ImportError
        If matplotlib is not installed and `ax` is None.
    """
    if ax is not None:
        return ax
    plt = _pyplot()
    created: Axes = plt.subplots(figsize=_SINGLE_SIZE, layout="constrained")[1]
    return created


def _panel_axes(axes: Sequence[Axes] | None, n_panels: int, *, stacked: bool) -> tuple[Axes, ...]:
    """Return one panel per piece of a figure, creating them when none were given.

    Parameters
    ----------
    axes : sequence of matplotlib.axes.Axes or None
        The panels the caller passed in.
    n_panels : int
        How many panels the figure has.
    stacked : bool
        Whether a created figure stacks its panels in a column, as a figure of
        time series does, rather than laying them out in a row.

    Returns
    -------
    tuple of matplotlib.axes.Axes
        The panels to draw on.

    Raises
    ------
    ImportError
        If matplotlib is not installed and `axes` is None.
    ValueError
        If `n_panels` is below one, or if `axes` holds a different number of
        panels.
    """
    if n_panels < 1:
        raise ValueError(f"a figure needs at least one panel, got {n_panels}")
    if axes is not None:
        given = tuple(axes)
        if len(given) != n_panels:
            raise ValueError(
                f"axes must hold one Axes per panel, got {len(given)} for {n_panels} panel(s)"
            )
        return given
    plt = _pyplot()
    if stacked:
        rows, cols = n_panels, 1
        size = (_STACKED_PANEL_SIZE[0], _STACKED_PANEL_SIZE[1] * n_panels)
    else:
        rows, cols = 1, n_panels
        size = (_SIDE_PANEL_SIZE[0] * n_panels, _SIDE_PANEL_SIZE[1])
    grid = plt.subplots(rows, cols, figsize=size, squeeze=False, layout="constrained")[1]
    panels: tuple[Axes, ...] = tuple(grid.ravel())
    return panels


def _save_figure(ax: Axes, path: Path) -> Path:
    """Write the figure a panel belongs to and close it.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Any panel of the figure to write.
    path : pathlib.Path
        File to write.

    Returns
    -------
    pathlib.Path
        The file that was written.

    Raises
    ------
    ImportError
        If matplotlib is not installed.
    """
    plt = _pyplot()
    # Every panel handed to this function was created by plt.subplots above, so
    # the figure it belongs to is a figure and not a sub figure.
    figure = cast("Figure", ax.get_figure())
    try:
        figure.savefig(path, dpi=_DPI)
    finally:
        plt.close(figure)
    return path


def _generator(rng: Seed) -> np.random.Generator:
    """Return the generator to draw from, building one from a seed if needed.

    Parameters
    ----------
    rng : numpy.random.Generator or int or None
        A generator, which is used as it is, or a seed.

    Returns
    -------
    numpy.random.Generator
        The generator every draw of the call goes through.
    """
    if isinstance(rng, np.random.Generator):
        return rng
    return np.random.default_rng(rng)


def _state_name(state: int) -> str:
    """Return the clinical name of a state code.

    Parameters
    ----------
    state : int
        Clinical code, +1 for no health and -1 for health.

    Returns
    -------
    str
        'no health' or 'health', as the paper names the two states.
    """
    return "no health" if state == _NO_HEALTH else "health"


def _chosen_patients(weekly: pd.DataFrame, patient_ids: Sequence[str] | None) -> list[str]:
    """Return the patients Figure 2 draws, in the order they will be drawn.

    Parameters
    ----------
    weekly : pandas.DataFrame
        A validated frame in the weekly schema.
    patient_ids : sequence of str or None
        The patients the caller asked for, or None for the first few of the
        frame.

    Returns
    -------
    list of str
        One identifier per panel.

    Raises
    ------
    ValueError
        If the frame holds no patient, or if `patient_ids` is empty or names a
        patient the frame does not hold.
    """
    present = [str(name) for name in weekly["patient_id"].drop_duplicates()]
    if not present:
        raise ValueError("the weekly frame holds no patient, so there is nothing to draw")
    if patient_ids is None:
        return present[:_FIG2_N_PATIENTS]
    chosen = [str(name) for name in patient_ids]
    if not chosen:
        raise ValueError("patient_ids must name at least one patient")
    known = set(present)
    missing = [name for name in chosen if name not in known]
    if missing:
        preview = present[:_PREVIEW_PATIENTS]
        raise ValueError(
            f"the weekly frame has no patient {missing}; it holds {len(present)} patient(s), "
            f"the first {len(preview)} of them {preview}"
        )
    return chosen


def _record_lengths(followup_weeks: npt.ArrayLike) -> _Vector:
    """Return the record lengths of Figure 3 as doubles, or explain why not.

    The same reading as :func:`msrelapse.fit` applies to its counts: one value
    per patient, in one dimension, and every one of them a real number of weeks.

    Parameters
    ----------
    followup_weeks : array_like
        One record length in weeks per patient.

    Returns
    -------
    numpy.ndarray
        The record lengths.

    Raises
    ------
    ValueError
        If the record lengths are not one dimensional, if they are empty, or if
        one of them is missing or infinite.
    """
    values = np.asarray(followup_weeks, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError(
            f"followup_weeks must be one record length per patient, got shape {values.shape}"
        )
    if values.size == 0:
        raise ValueError("followup_weeks must hold at least one record length in weeks")
    if not np.all(np.isfinite(values)):
        raise ValueError(
            "every record length in followup_weeks must be a finite number of weeks, "
            "but at least one is not"
        )
    return values


def _bin_counts(values: _Vector, edges: _Vector) -> _Vector:
    """Return how many values fall in each bin.

    Parameters
    ----------
    values : numpy.ndarray
        The observations.
    edges : numpy.ndarray
        The bin edges, one more than the number of bins.

    Returns
    -------
    numpy.ndarray
        One count per bin, as doubles so that it can be drawn directly.
    """
    counts, _ = np.histogram(values, bins=edges)
    return counts.astype(np.float64)


def _thinned_ticks(candidates: _Vector) -> _Vector:
    """Return the values to tick an axis at, at most :data:`_MAX_BIN_TICKS` of them.

    Ticking the bin edges themselves is what the paper does, so that a reader
    can see where a bin begins, and every other candidate is dropped as often as
    it takes to keep the labels apart. At the bins of Figure 4 this gives the
    axes of the paper: every edge on the no health panel and every second one,
    that is every 200 weeks, on the health panel. The relapse count axis of
    :func:`fig_poisson_to_nb` is thinned the same way, from every count.

    Parameters
    ----------
    candidates : numpy.ndarray
        The values that could be ticked, in ascending order.

    Returns
    -------
    numpy.ndarray
        The values to tick at.
    """
    step = max(1, math.ceil(candidates.size / _MAX_BIN_TICKS))
    return candidates[::step]


def _draw_bars(ax: Axes, edges: _Vector, counts: _Vector) -> None:
    """Draw one bar per bin, each spanning its own bin.

    The bars carry no legend entry, because a panel of one set of bars has
    nothing to tell apart and the title already says where they came from.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The panel to draw on.
    edges : numpy.ndarray
        The bin edges, one more than the number of bars.
    counts : numpy.ndarray
        One bar height per bin.

    Returns
    -------
    None
        Nothing is returned; the bars are drawn on `ax`.
    """
    ax.bar(
        edges[:-1],
        counts,
        width=np.diff(edges),
        align="edge",
        color="lightsteelblue",
        edgecolor="white",
    )


def _fig4_bins(state: int) -> tuple[float, float]:
    """Return the first bin edge and the bin width of one panel of Figure 4.

    Parameters
    ----------
    state : int
        Clinical code of the state the panel holds.

    Returns
    -------
    tuple of float
        The first edge and the width, both in weeks.
    """
    if state == _NO_HEALTH:
        return PAPER.relapse_duration_min_weeks.value, PAPER.fig4a_bin_width_weeks.value
    return _FIG4B_BIN_START, PAPER.fig4b_bin_width_weeks.value


def _draw_digitised_durations(ax: Axes, state: int) -> None:
    """Draw one panel of Figure 4 from the bar heights measured off the paper.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The panel to draw on.
    state : int
        Clinical code of the state the panel holds.

    Returns
    -------
    None
        Nothing is returned; the panel is drawn on `ax`.
    """
    if state == _NO_HEALTH:
        heights = PAPER.fig4a_counts.value
        total = PAPER.n_no_health_events.value
        mean = PAPER.tau_no_health_cohort_weeks.value
    else:
        heights = PAPER.fig4b_counts.value
        total = PAPER.n_health_events.value
        mean = PAPER.tau_health_cohort_weeks.value
    start, width = _fig4_bins(state)
    edges = start + width * np.arange(len(heights) + 1, dtype=np.float64)
    counts = np.asarray(heights, dtype=np.float64)
    title = "Digitised from Figure 4 of the paper"
    _draw_duration_panel(ax, state, edges, counts, total, mean, title)


def _state_durations(durations: pd.DataFrame, state: int) -> _Vector:
    """Return the recorded runs of one state, or explain that there are none.

    Parameters
    ----------
    durations : pandas.DataFrame
        A validated frame in the durations schema.
    state : int
        Clinical code of the state to select.

    Returns
    -------
    numpy.ndarray
        The durations of that state, in weeks.

    Raises
    ------
    ValueError
        If the frame holds no run of `state`.
    """
    values = durations.loc[durations["state"] == state, "duration_w"].to_numpy(dtype=np.float64)
    if values.size == 0:
        raise ValueError(
            f"the durations frame holds no run of state {state:+d}, so the {_state_name(state)} "
            f"panel of Figure 4 has nothing to bin"
        )
    return values


def _draw_observed_durations(ax: Axes, values: _Vector, state: int) -> None:
    """Draw one panel of Figure 4 from the runs of an observed cohort.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The panel to draw on.
    values : numpy.ndarray
        The recorded runs of that state, in weeks, at least one of them.
    state : int
        Clinical code of the state the panel holds.

    Returns
    -------
    None
        Nothing is returned; the panel is drawn on `ax`.
    """
    start, width = _fig4_bins(state)
    n_bins = max(1, math.ceil((float(values.max()) - start) / width))
    edges = start + width * np.arange(n_bins + 1, dtype=np.float64)
    _draw_duration_panel(
        ax,
        state,
        edges,
        _bin_counts(values, edges),
        int(values.size),
        float(values.mean()),
        f"This cohort, {values.size} event(s)",
    )


# The seven parameters are the panel, what it holds and the two numbers the
# paper prints inside it. Grouping them into an object would hide the plain call.
def _draw_duration_panel(  # noqa: PLR0917
    ax: Axes,
    state: int,
    edges: _Vector,
    counts: _Vector,
    total: int,
    mean: float,
    title: str,
) -> None:
    """Draw the bars, the axis labels and the two in-panel numbers of Figure 4.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The panel to draw on.
    state : int
        Clinical code of the state the panel holds.
    edges : numpy.ndarray
        The bin edges, in weeks.
    counts : numpy.ndarray
        One bar height per bin.
    total : int
        Number of events the panel reports.
    mean : float
        Mean duration the panel reports, in weeks.
    title : str
        Where the bars came from, written above the panel.

    Returns
    -------
    None
        Nothing is returned; the panel is drawn on `ax`.
    """
    name = _state_name(state)
    _draw_bars(ax, edges, counts)
    ax.set_xticks(_thinned_ticks(edges))
    ax.set_xlabel(f"Duration of {name} events (week)")
    ax.set_ylabel(f"Counts (no. of {name} events)")
    ax.set_title(title)
    _corner_text(
        ax,
        f"{name.capitalize()} events: total = {total}\n"
        f"Duration of {name} events: mean = {mean:.3g} weeks",
    )


def _draw_potential(ax: Axes, well: DoubleWell) -> None:
    """Draw one potential curve over the x range of the paper.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The panel to draw on.
    well : DoubleWell
        The potential to draw.

    Returns
    -------
    None
        Nothing is returned; the curve is drawn on `ax`.
    """
    grid = np.linspace(_X_LIMITS[0], _X_LIMITS[1], _POTENTIAL_POINTS, dtype=np.float64)
    ax.plot(grid, well.V(grid), color=_CURVE_COLOUR, linewidth=1.2)
    ax.set_xlim(*_X_LIMITS)
    ax.set_ylim(*_V_LIMITS)
    ax.set_xlabel("x")
    ax.set_ylabel("V(x)")


def _well_and_points(alpha: float, beta: float) -> tuple[DoubleWell, CriticalPoints]:
    """Return a potential and its three stationary points, or explain why not.

    The pair is built before any panel is created, so that a potential none of
    these figures can be drawn from is refused while there is still no figure to
    leave open.

    Parameters
    ----------
    alpha : float
        Control parameter of the potential.
    beta : float
        Asymmetry parameter of the potential.

    Returns
    -------
    tuple of DoubleWell and CriticalPoints
        The potential and its stationary points, from left to right.

    Raises
    ------
    ValueError
        If `alpha` is not positive and finite, if `beta` is not finite, or if
        the potential has fewer than two wells.
    """
    well = DoubleWell(alpha, beta)
    return well, well.critical_points()


def _draw_asymmetric_potential(ax: Axes, well: DoubleWell, points: CriticalPoints) -> None:
    """Draw a tilted double well with both of its barriers marked.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The panel to draw on.
    well : DoubleWell
        The potential to draw.
    points : CriticalPoints
        Its three stationary points, from :func:`_well_and_points`.

    Returns
    -------
    None
        Nothing is returned; the panel is drawn on `ax`.
    """
    _draw_potential(ax, well)
    top = float(well.V(points.saddle))
    health = float(well.V(points.health))
    relapse = float(well.V(points.relapse))
    for level in (top, health, relapse):
        _guide_line(ax, level)
    _barrier_arrow(ax, 0.5 * (points.health + points.saddle), top, health, r"$\Delta V_1$")
    _barrier_arrow(ax, 0.5 * (points.saddle + points.relapse), top, relapse, r"$\Delta V_2$")
    _mark_critical_points(ax, well, points)
    _annotate_parameters(ax, well.alpha, well.beta)


def _guide_line(ax: Axes, level: float) -> None:
    """Draw one dashed horizontal guide at a level of the potential.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The panel to draw on.
    level : float
        The value of the potential the guide runs through.

    Returns
    -------
    None
        Nothing is returned; the guide is drawn on `ax`.
    """
    ax.axhline(level, linestyle="--", linewidth=0.8, color=_GUIDE_COLOUR)


def _barrier_arrow(ax: Axes, x: float, top: float, bottom: float, label: str) -> None:
    """Draw a double headed arrow between two levels and label it.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The panel to draw on.
    x : float
        Where to stand the arrow.
    top : float
        The upper level, the potential at the top of the barrier.
    bottom : float
        The lower level, the potential at the bottom of a well.
    label : str
        Text to write beside the arrow.

    Returns
    -------
    None
        Nothing is returned; the arrow is drawn on `ax`.
    """
    ax.annotate(
        "",
        xy=(x, top),
        xytext=(x, bottom),
        arrowprops={"arrowstyle": "<->", "color": _CURVE_COLOUR, "linewidth": 0.9},
    )
    ax.text(x + _LABEL_OFFSET, 0.5 * (top + bottom), label, ha="left", va="center")


def _mark_critical_points(ax: Axes, well: DoubleWell, points: CriticalPoints) -> None:
    """Mark and name the three stationary points of a potential.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The panel to draw on.
    well : DoubleWell
        The potential the points belong to.
    points : CriticalPoints
        The three stationary points, from left to right.

    Returns
    -------
    None
        Nothing is returned; the marks are drawn on `ax`.
    """
    named = ((points.health, r"$x_1$"), (points.saddle, r"$x_0$"), (points.relapse, r"$x_2$"))
    for position, label in named:
        level = float(well.V(position))
        ax.plot([position], [level], marker="o", markersize=4, linestyle="none", color="tab:red")
        ax.text(position, level - _LABEL_OFFSET, label, ha="center", va="top")


def _annotate_parameters(ax: Axes, alpha: float, beta: float) -> None:
    """Write the parameters of a potential inside its panel, at the upper right.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The panel to draw on.
    alpha : float
        Control parameter of the potential.
    beta : float
        Asymmetry parameter of the potential.

    Returns
    -------
    None
        Nothing is returned; the text is drawn on `ax`.
    """
    _corner_text(ax, f"alpha = {alpha:g}, beta = {beta:g}")


def _corner_text(ax: Axes, text: str) -> None:
    """Write a note in the upper right corner of a panel, as the paper does.

    The note is given an opaque background, because the curve or the bars of a
    panel can run through that corner and would otherwise cross the text.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The panel to write on.
    text : str
        The note, which may hold several lines.

    Returns
    -------
    None
        Nothing is returned; the text is drawn on `ax`.
    """
    ax.text(
        0.97,
        0.95,
        text,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize="small",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85, "pad": 2.0},
    )


def _kaplan_meier(values: _Vector, censored: npt.NDArray[np.bool_]) -> tuple[_Vector, _Vector]:
    """Return the Kaplan Meier estimate of the survival of a set of durations.

    Parameters
    ----------
    values : numpy.ndarray
        Recorded durations in weeks, complete and censored alike.
    censored : numpy.ndarray
        True where the corresponding duration is right censored.

    Returns
    -------
    tuple of numpy.ndarray
        The times of the steps, starting at zero, and the survival just after
        each of them, starting at one.
    """
    times = [0.0]
    survival = [1.0]
    current = 1.0
    for time in np.unique(values[~censored]):
        at_risk = int(np.count_nonzero(values >= time))
        deaths = int(np.count_nonzero((values == time) & ~censored))
        current *= 1.0 - deaths / at_risk
        times.append(float(time))
        survival.append(current)
    return np.array(times, dtype=np.float64), np.array(survival, dtype=np.float64)


def _discrete_hazard(values: _Vector) -> tuple[_Vector, _Vector]:
    """Return the weekly hazard of a set of complete durations.

    This is the hazard :func:`msrelapse.fit.test_memoryless` regresses on time:
    the share of the durations still at risk at the start of week k that end in
    that week, read only at the weeks where at least
    :data:`msrelapse.fit._MIN_AT_RISK` of them are still at risk.

    Parameters
    ----------
    values : numpy.ndarray
        Complete durations, whole weeks and at least one week each.

    Returns
    -------
    tuple of numpy.ndarray
        The weeks that were read and the hazard at each of them. Both are empty
        when no week holds enough durations still at risk, which the inset says
        rather than drawing.

    Notes
    -----
    This is a copy of the hazard of :func:`msrelapse.fit._hazard_test` and has
    to be changed in lockstep with it, until that module offers the computation
    as a helper of its own. The inset is regressed against the slope and the
    intercept that module reports, so the two cannot drift apart unnoticed.
    """
    weeks = values.astype(np.int64)
    deaths = np.bincount(weeks)[1:]
    times = np.arange(1, deaths.size + 1, dtype=np.float64)
    at_risk = weeks.size - np.concatenate(([0], np.cumsum(deaths)[:-1]))
    keep = at_risk >= _MIN_AT_RISK
    return times[keep], deaths[keep] / at_risk[keep]


def _draw_hazard_inset(ax: Axes, complete: _Vector, fitted_hazard: float) -> None:
    """Draw the discrete hazard of a state in an inset of its survival panel.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The panel the inset goes inside.
    complete : numpy.ndarray
        The complete durations of the state, in whole weeks.
    fitted_hazard : float
        The constant weekly hazard of the fitted law, drawn as a dashed guide.

    Returns
    -------
    None
        Nothing is returned; the inset is added to `ax`.

    Notes
    -----
    A set of durations too small for any week to be read, which is every set of
    fewer than :data:`msrelapse.fit._MIN_AT_RISK` durations and a per patient
    call of the paper's own sample patients, gets an inset saying so rather than
    an empty pair of axes with a lone guide line across it.
    """
    weeks, hazard = _discrete_hazard(complete)
    inset = ax.inset_axes(_INSET_BOX)
    inset.tick_params(labelsize="x-small")
    if weeks.size == 0:
        # Ticks of an axis that holds nothing would read as data, so the inset
        # is left as a plain box around the note.
        inset.set_xticks([])
        inset.set_yticks([])
        inset.text(
            0.5,
            0.5,
            f"none of these {complete.size} duration(s)\n"
            f"reaches a week with {_MIN_AT_RISK} still at risk",
            transform=inset.transAxes,
            ha="center",
            va="center",
            fontsize="x-small",
        )
        return
    inset.plot(weeks, hazard, marker="o", markersize=3, linestyle="none", color=_CURVE_COLOUR)
    inset.axhline(fitted_hazard, linestyle="--", linewidth=0.8, color="tab:red")
    inset.set_xlabel("Week", fontsize="x-small")
    inset.set_ylabel("Hazard", fontsize="x-small")


def _poisson_mass(grid: npt.NDArray[np.int64], mean: float) -> _Vector:
    """Return the Poisson probability of each count of a grid.

    Parameters
    ----------
    grid : numpy.ndarray
        The counts to evaluate the mass at.
    mean : float
        Mean of the Poisson law.

    Returns
    -------
    numpy.ndarray
        One probability per count.
    """
    return np.asarray(stats.poisson.pmf(grid, mean), dtype=np.float64)


def _nb_mass(grid: npt.NDArray[np.int64], mean: float, dispersion: float) -> _Vector:
    """Return the negative binomial probability of each count of a grid.

    Parameters
    ----------
    grid : numpy.ndarray
        The counts to evaluate the mass at.
    mean : float
        Mean of the fitted law.
    dispersion : float
        Dispersion a of the NB2 parameterisation, in which the variance is
        ``mean + a mean**2``. A dispersion of zero is the Poisson law, which is
        what such a fit reports and what is drawn for it.

    Returns
    -------
    numpy.ndarray
        One probability per count.
    """
    if dispersion <= 0.0:
        return _poisson_mass(grid, mean)
    size = 1.0 / dispersion
    return np.asarray(stats.nbinom.pmf(grid, size, size / (size + mean)), dtype=np.float64)


def _relapse_counts(durations: pd.DataFrame) -> pd.Series[int]:
    """Return how many relapses each patient of a durations frame had.

    Parameters
    ----------
    durations : pandas.DataFrame
        A validated frame in the durations schema.

    Returns
    -------
    pandas.Series
        One count per patient, of dtype int64 and indexed by patient_id. A
        patient who never relapsed keeps a count of zero.
    """
    relapses = durations["state"] == _NO_HEALTH
    counts = relapses.groupby(durations["patient_id"], sort=True).sum()
    return counts.astype(np.int64).rename("relapses").rename_axis("patient_id")
