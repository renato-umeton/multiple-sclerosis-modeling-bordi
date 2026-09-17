"""The figures of the paper, drawn from the data structures of this package.

matplotlib is an optional dependency, the ``plot`` extra, so it is imported
inside the functions that draw rather than at the top of the module. Importing
[`msrelapse.plots`][msrelapse.plots] therefore works in an install without it,
and only a call that has to create a figure raises, naming the extra to
install. [`require_matplotlib`][msrelapse.plots.require_matplotlib] is that
check on its own, and hands back the pyplot module, for a caller that would
rather learn of a missing matplotlib before it starts work than at the figure.

Every figure function takes the axes to draw on and gives them back, so that a
caller can compose the panels into a figure of their own; passing None instead
creates a figure of the size the panels want. A figure of several panels takes
and returns a tuple of axes, one per panel. Whatever a function is given is
checked before it creates anything, so a call that is refused leaves no figure
behind in the global registry of pyplot, which the caller has no handle on and
would have to close by number.

[`animate_double_well`][msrelapse.plots.animate_double_well] is the one drawing
of the module that moves, so it takes a whole figure rather than a panel: it
lays out three panels of its own and hands back the animation over them, which
the caller plays or saves.
[`save_double_well_gif`][msrelapse.plots.save_double_well_gif] is that call
written out to a file, and it is what writes the animation the README shows.
Its bottom panel is the illustrative EDSS trajectory of
[`msrelapse.edss`][msrelapse.edss], which is an extension of this package and
not a quantity the article reports.

Numbers taken from the article, the bin edges of Figure 3 and the digitised bar
heights of Figures 3 and 4 among them, are read from ``PAPER`` in
[`msrelapse._params`][msrelapse._params] and are never written here. What the
paper does not fix, and what is therefore chosen here, is the resolution of the
potential curves, the time step of the simulated paths of Figure 7 and the
placement of the labels and arrows inside a panel.

The figures label the no health state of the article as Flare, the clinical
word for it, while the code keeps the names of the article, the state code
``PAPER.state_no_health`` among them.

Two of the figures are not in the article at all.
[`fig_survival_vs_exponential`][msrelapse.plots.fig_survival_vs_exponential]
puts the observed survival of one state beside the exponential fitted to it,
and [`fig_poisson_to_nb`][msrelapse.plots.fig_poisson_to_nb] puts the relapse
counts of a cohort beside the Poisson and the negative binomial mass. The
article reports no fit, no test and no interval of any kind, as section 7 of
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
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING, Any, Final, cast

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy import stats

from msrelapse._params import PAPER
from msrelapse.edss import EDSSSpec, edss_trajectory
from msrelapse.fit import MIN_AT_RISK, discrete_hazard, fit_durations, fit_nb_counts
from msrelapse.io import validate
from msrelapse.model import DEFAULT_BAND_FRACTION, CriticalPoints, DoubleWell, calibrate
from msrelapse.simulate import BRIDGE_CONSTANT, Seed, simulate_paths, to_states, to_weekly
from msrelapse.stats import WEEKS_PER_YEAR

if TYPE_CHECKING:
    from matplotlib.animation import FuncAnimation
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from matplotlib.lines import Line2D

__all__ = [
    "animate_double_well",
    "fig2_sample_patients",
    "fig3_rr_phase_histogram",
    "fig4_duration_histograms",
    "fig5_symmetric_potentials",
    "fig6_asymmetric_potential",
    "fig7_simulated_paths",
    "fig8_patient_potentials",
    "fig_poisson_to_nb",
    "fig_survival_vs_exponential",
    "require_matplotlib",
    "save_all_paper_figures",
    "save_double_well_gif",
]

_Vector = npt.NDArray[np.float64]
_States = npt.NDArray[np.int64]

_NO_HEALTH: Final = PAPER.state_no_health.value
_HEALTH: Final = PAPER.state_health.value
_WEEK: Final = PAPER.time_resolution_weeks.value

# What a panel calls the two states. The article writes the plus one state as
# no health; a figure of this module says flare, the clinical word for it.
_HEALTH_LABEL: Final = "Health"
_FLARE_LABEL: Final = "Flare"

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

# The animation and its three panels, none of it from the paper. The figure is
# sized so that the default run stays a file a README can carry, the time step
# is the one Figure 7 is integrated at, and the rest is placement.
_ANIMATION_SIZE: Final = (9.0, 6.0)
_ANIMATION_HEIGHT_RATIOS: Final = (1.0, 0.8)
_ANIMATION_DT: Final = _FIG7_DT

# How long the animation observes one patient for when it is asked for no
# window: ten years, written in the whole weeks the package counts a year in,
# which is the 522 weeks msrelapse.edss reads ten years at rather than the 520
# of ten flat 52 week years. The frame budget is what keeps the written file
# small enough for a README to carry, and one frame therefore covers a little
# over two weeks.
_ANIMATION_YEARS: Final = 10.0
_ANIMATION_WEEKS: Final = round(_ANIMATION_YEARS * WEEKS_PER_YEAR)
_ANIMATION_FRAMES: Final = 260
_ANIMATION_WEEKS_PER_FRAME: Final = _ANIMATION_WEEKS / _ANIMATION_FRAMES

_PARTICLE_SIZE: Final = 10
_CURRENT_WEEK_SIZE: Final = 6
_SADDLE_SIZE: Final = 5
# The bottom of the disability panel, and the room left above the highest score
# the trace reaches, so that a peak does not sit on the top of the panel. The
# panel never shrinks below the floor, so that a quiet record is not drawn as a
# dramatic one by a tight axis.
_EDSS_PANEL_FLOOR: Final = 6.0
_EDSS_HEADROOM: Final = 0.5
# The key of the disability panel is laid out in one flat strip, because the
# panel is wide and short and a stacked key would cover the opening weeks.
_EDSS_LEGEND_COLUMNS: Final = 3
# How many frames of a finished animation the contact sheet lays side by side.
_CONTACT_SHEET_PANELS: Final = 4

# The moving pieces carry a label, which is how a caller reading a panel back
# tells them from the static lines beside them.
_PARTICLE_LABEL: Final = "x(t)"
_CURRENT_WEEK_LABEL: Final = "current week"
_SADDLE_LABEL: Final = "saddle"
_EDSS_LABEL: Final = "EDSS"
_EDSS_DISPLAY_LABEL: Final = "displayed EDSS"
_EDSS_BASELINE_LABEL: Final = "baseline"
_EDSS_AXIS_LABEL: Final = "EDSS (illustrative model, see docs)"

# What the two time panels of the animation call their axis. Everything the
# animation computes is in weeks, and a year of them is what a panel divides by
# to place a week on that axis.
_TIME_AXIS_LABEL: Final = "Time (years)"


def fig2_sample_patients(
    weekly: pd.DataFrame,
    patient_ids: Sequence[str] | None = None,
    axes: Sequence[Axes] | None = None,
) -> tuple[Axes, ...]:
    """Draw the weekly record of a few patients as step functions, Figure 2.

    Each panel holds the plus one and minus one series of one patient against
    the week. The minus one level is ticked Health, as the article ticks it,
    and the plus one level Flare, the clinical word for the state the article
    calls no health. Neither tick carries the printed typo of the article,
    whose own Figure 2 reads "Health state" and "No health sate".

    Parameters
    ----------
    weekly : pandas.DataFrame
        A frame in the weekly schema of [`msrelapse.io`][msrelapse.io]. It is
        validated before use.
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
        panel.set_yticklabels([_HEALTH_LABEL, _FLARE_LABEL])
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
        A frame in the durations schema of [`msrelapse.io`][msrelapse.io],
        validated before use. The default draws the bar heights digitised from
        the figure of the paper instead.
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
    """Draw one panel of the asymmetric double well and both its barriers, Figure 6.

    The panel holds equation (3), with the three stationary points marked, a
    dashed guide through each of them and a double headed arrow for each of the
    two barriers. The article draws that panel twice, at the same two control
    parameters as its Figure 5: (a) at the reference value, which is the default
    here, and (b) at the lower illustrative one. A caller who wants panel (b)
    passes ``alpha=PAPER.alpha_illustrative_low.value``, and
    [`save_all_paper_figures`][msrelapse.plots.save_all_paper_figures] writes
    both panels side by side in the one file.

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
        ``numpy.random.default_rng``. One generator is built for the whole
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
        [`msrelapse.simulate.simulate_paths`][msrelapse.simulate.simulate_paths]
        is out of range, or if `axes` does not have one Axes per asymmetry.

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

    Every panel is drawn the way
    [`fig6_asymmetric_potential`][msrelapse.plots.fig6_asymmetric_potential]
    draws one, at the reference control parameter of the paper, which is the
    only value the paper ever fits a patient with. The one difference is the
    dashed guides: the article draws a single horizontal line per panel here,
    through V(x0), the level both barriers are measured down from, where its
    Figure 6 draws one line through each of the three stationary levels.

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
        _draw_asymmetric_potential(panel, well, points, guides=(points.saddle,))
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
    [`msrelapse.fit.fit_durations`][msrelapse.fit.fit_durations] fits to the
    same runs, read on the whole weeks the durations are recorded in. The
    figure is not in the paper, which reports no fit at all.

    Parameters
    ----------
    durations : pandas.DataFrame
        A frame in the durations schema of [`msrelapse.io`][msrelapse.io]. It
        is validated before use.
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
    The inset repeats the hazard
    [`msrelapse.fit.test_memoryless`][msrelapse.fit.test_memoryless] regresses
    on time: the complete durations alone, read at the weeks where at least
    five of them are still at risk. The step of the main panel is the wider
    estimate and counts a censored run as at risk until it is cut off. The
    dashed guide of the inset sits at one over the fitted mean, which is the
    weekly hazard of the fitted law however the law is read, since a duration
    rounded up to whole weeks is geometric with that success probability.

    The main curve reads the same fit the same way. It is ``(1 - rate) ** t``,
    whose weekly hazard is the rate the dashed guide sits at and whose survival
    at every whole week is that of the geometric of the fitted mean. As a
    continuous exponential it is the one of scale ``-1 / log(1 - rate)``, about
    half a week shorter than the fitted mean and the scale
    [`msrelapse.fit.test_memoryless`][msrelapse.fit.test_memoryless] draws its
    null at. Drawing ``exp(-t / mean)`` instead rounds the law a second time:
    it lifts the curve above the step where runs last only a few weeks, so the
    panel would show a record falling away from its own fit, and the main panel
    and the inset would describe two different laws. A state whose runs all
    lasted a single week puts the rate at one, and the curve is then the drop
    to zero after time zero that such a law gives.

    The panel runs to the longest time any run of the state was at risk for,
    which is past the last event whenever a censored run outlived every one of
    them. Ending it at the last event instead would hide that run altogether,
    although it counts towards the fit, and would cut the fitted curve off in a
    range where it can look flat.

    A frame too small for any week to be read, which one patient of the paper's
    own size is, gets an inset that says so rather than an empty one. That is
    the same frame
    [`msrelapse.fit.test_memoryless`][msrelapse.fit.test_memoryless] refuses
    outright; the step and the fitted curve of the main panel are drawn as
    usual.
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
        _fitted_survival(grid, fitted.rate),
        color="tab:red",
        label=f"exponential on whole weeks, mean {fitted.mean:.3g} weeks",
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
    Poisson mass at the sample mean and the second is the negative binomial
    mass [`msrelapse.fit.fit_nb_counts`][msrelapse.fit.fit_nb_counts] fits to
    the same counts. A cohort in which the onset rate varies from patient to
    patient sits above the Poisson line in both tails. The figure is not in the
    paper, which reports no fit.

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
    Every figure the article prints as a pair of panels is written as that pair
    in one file, Figure 6 among them, whose two control parameters are the ones
    of Figure 5.

    Parameters
    ----------
    out_dir : str or path-like
        Directory to write into. It is created if it does not exist.
    weekly : pandas.DataFrame
        A frame in the weekly schema of [`msrelapse.io`][msrelapse.io], which
        supplies the sample patients of Figure 2 and the record lengths of
        Figure 3.
    durations : pandas.DataFrame
        A frame in the durations schema of [`msrelapse.io`][msrelapse.io],
        which supplies the histograms of Figure 4, the survival figure and the
        relapse counts.
    rng : numpy.random.Generator or int or None, optional
        Generator behind the simulated paths of Figure 7, or a seed for
        ``numpy.random.default_rng``.

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
        ("fig6_asymmetric_potential", lambda: _fig6_panels()[0]),
        ("fig7_simulated_paths", lambda: fig7_simulated_paths(rng=rng)[0]),
        ("fig8_patient_potentials", lambda: fig8_patient_potentials()[0]),
        ("fig_survival_vs_exponential", lambda: fig_survival_vs_exponential(durations, _NO_HEALTH)),
        ("fig_poisson_to_nb", lambda: fig_poisson_to_nb(counts)),
    ]
    plt = require_matplotlib()
    opened_before = set(plt.get_fignums())
    written: list[Path] = []
    try:
        for name, draw in drawings:
            written.append(_save_figure(draw(), directory / f"{name}.png"))
    finally:
        for number in sorted(set(plt.get_fignums()) - opened_before):
            plt.close(number)
    return written


def animate_double_well(  # noqa: PLR0917
    well: DoubleWell | None = None,
    sigma: float | None = None,
    n_weeks: int = _ANIMATION_WEEKS,
    dt: float = _ANIMATION_DT,
    band_fraction: float = DEFAULT_BAND_FRACTION,
    weeks_per_frame: float = _ANIMATION_WEEKS_PER_FRAME,
    rng: Seed = None,
    fig: Figure | None = None,
    spec: EDSSSpec | None = None,
) -> FuncAnimation:
    """Animate one simulated record as the particle, the weekly series and the EDSS.

    The figure holds three panels. The potential at the top left carries the
    particle at the current x(t), with the two wells labelled Health and Flare
    and the barrier top between them marked. The weekly record at the top right
    is the step plot of Figure 2, the plus one and minus one series against the
    time, drawn up to the current week and marked there. The panel across the
    bottom is the illustrative EDSS trajectory of
    [`msrelapse.edss`][msrelapse.edss] that the same weekly series drives,
    drawn as a thin continuous line with the displayed half point score
    stepping over it and the baseline marked.

    The article prints no such figure. What the two top panels show is the
    model of the article at work, and every number behind them is one the
    package computes from the two mean durations the article reports.

    The bottom panel is an illustrative extension and not part of the article,
    which reports no disability score for any of its patients. Its parameters
    come from a separate literature synthesis, which
    [`msrelapse.edss.EVIDENCE`][msrelapse.edss.EVIDENCE] records with a source
    and a DOI for each one. A trace is one draw and never a prognosis.

    Parameters
    ----------
    well : DoubleWell, optional
        The potential the path is drawn on. The default is the potential
        [`msrelapse.model.calibrate`][msrelapse.model.calibrate] returns for the
        two mean durations the paper reports, at the reference control
        parameter.
    sigma : float, optional
        Noise amplitude. The default is the noise of that same calibration,
        which is not the noise the paper prints for its Figure 7.
    n_weeks : int, optional
        Length of the record, in whole weeks. Must be at least one. The default
        is ten years, that is ``round(10 * WEEKS_PER_YEAR)`` weeks of
        [`msrelapse.stats`][msrelapse.stats], which comes to 522.
    dt : float, optional
        Time step of the integration, in weeks. Not from the paper.
    band_fraction : float, optional
        Position of the two hysteresis thresholds that map the path to the two
        clinical states, as
        [`msrelapse.simulate.to_states`][msrelapse.simulate.to_states] takes it.
    weeks_per_frame : float, optional
        How many weeks one frame advances by. Must be positive and finite. The
        default is the ten year record over 260 frames, which is a little over
        two weeks a frame.
    rng : numpy.random.Generator or int or None, optional
        Generator to draw the noise from, or a seed for
        ``numpy.random.default_rng``. A seed is what makes a written file
        repeatable.
    fig : matplotlib.figure.Figure, optional
        The figure to lay the three panels out on, which is added to rather
        than cleared. The default creates one.
    spec : msrelapse.edss.EDSSSpec, optional
        The parameters of the disability trace across the bottom. The default
        is the published one,
        [`msrelapse.edss.EDSSSpec`][msrelapse.edss.EDSSSpec] as it comes.

    Returns
    -------
    matplotlib.animation.FuncAnimation
        The animation, which has to be kept alive by the caller for as long as
        it is played or saved.

    Raises
    ------
    ImportError
        If matplotlib is not installed.
    ValueError
        If `n_weeks` is below one, if `weeks_per_frame` is not a positive
        finite number of weeks, or if any argument of
        [`msrelapse.simulate.simulate_paths`][msrelapse.simulate.simulate_paths]
        or of [`msrelapse.simulate.to_states`][msrelapse.simulate.to_states] is
        out of range.

    See Also
    --------
    save_double_well_gif : The same animation, written to a file.

    Notes
    -----
    The path starts at the bottom of the health well, so every record opens in
    remission, and it is cut into the two states through the hysteresis band
    with the Brownian bridge shift
    [`msrelapse.simulate.simulate_weekly`][msrelapse.simulate.simulate_weekly]
    applies, so the episodes of the animation are the episodes a weekly record
    of this package holds. Those weekly episodes run longer than the two mean
    durations the default is calibrated to: the calibration times the passage
    from the bottom of a well to the saddle, while the weekly reading counts
    every week the path touches a state as a whole week of it, the rounding rule
    the study applies to its own records.

    The disability trace is drawn from the same generator as the path, after
    it, so a seed that gave a record gives that same record still and only the
    bottom panel is new. Its peak and its residual are drawn once per episode
    and the whole trace is computed before the first frame, so what the panel
    does as it plays is uncover a curve rather than draw a new one.

    What the bottom panel looks like is one draw and changes from seed to seed.
    A relapse that draws a residual of zero leaves a spike that decays back to
    where it started; a relapse that draws a residual equal to its peak leaves
    a step with no visible transient at all, since the trace never comes back
    down; and a record whose relapses happen to leave a residual more often
    than the published 42 percent climbs faster than a typical one. Read a
    trace as one draw of the model rather than as the shape it usually makes.

    Every axis is given its limits before the first frame, the two of the
    potential from the figures of the paper and the time axes from the whole
    record, so that nothing moves during playback except the pieces that are
    meant to.

    The two time panels are drawn in years, ticked at the whole years of the
    record and titled with the year the record has reached, while everything
    behind them stays in weeks: the path, the weekly series and the disability
    trace are computed, cut and read in weeks, and a panel divides a week
    number by [`msrelapse.stats.WEEKS_PER_YEAR`][msrelapse.stats.WEEKS_PER_YEAR]
    to place it. A record shorter than a year keeps the ticks matplotlib picks
    for it, since the only whole year inside it is zero.
    """
    record = _animation_record(well, sigma, n_weeks, dt, band_fraction, rng, spec)
    weeks = _animation_weeks(int(record.weekly.size), weeks_per_frame)
    # Everything that can be refused has been refused by now, so a figure
    # created here is a figure the call will hand back.
    plt = require_matplotlib()
    from matplotlib.animation import FuncAnimation  # noqa: PLC0415

    figure = plt.figure(figsize=_ANIMATION_SIZE, layout="constrained") if fig is None else fig
    return FuncAnimation(figure, _draw_animation(figure, record, weeks), len(weeks), blit=False)


def save_double_well_gif(
    path: str | Path,
    fps: int = 8,
    dpi: int = 80,
    contact_sheet: Path | None = None,
    **kwargs: Any,
) -> Path:
    """Write the animation of the double well as a GIF, and return the file.

    Parameters
    ----------
    path : str or pathlib.Path
        File to write. Its directory is created if it does not exist.
    fps : int, optional
        Frames per second of the written file, which fixes how long a frame is
        shown for.
    dpi : int, optional
        Dots per inch of each frame. With the figure of about nine by six
        inches this module animates, the default gives a frame of 720 by 480
        pixels and a default run under a megabyte.
    contact_sheet : pathlib.Path, optional
        Where to write a PNG of four evenly spaced frames side by side, so that
        the result can be read without playing it. The default writes none.
    **kwargs
        Passed to
        [`animate_double_well`][msrelapse.plots.animate_double_well], which is
        where the record, the potential and the seed are chosen.

    Returns
    -------
    pathlib.Path
        The file that was written.

    Raises
    ------
    ImportError
        If matplotlib is not installed.
    ValueError
        If an argument of
        [`animate_double_well`][msrelapse.plots.animate_double_well] is out of
        range.

    Notes
    -----
    The writer is the Pillow one, so no external program is needed: Pillow is a
    dependency of matplotlib itself. The contact sheet is read back out of the
    finished file rather than drawn a second time, so it shows the frames the
    file holds.

    The figure is created here rather than inside the animation, so that it is
    closed however the run ends and no figure is left behind in the registry of
    pyplot, which the caller has no handle on.
    """
    plt = require_matplotlib()
    from matplotlib.animation import PillowWriter  # noqa: PLC0415

    target = Path(path)
    # The directory is made before the animation is built, so that a mistyped or
    # an unwritable path costs a moment rather than a whole default run, which
    # integrates ten years of weeks and renders a couple of hundred frames.
    target.parent.mkdir(parents=True, exist_ok=True)
    figure = plt.figure(figsize=_ANIMATION_SIZE, layout="constrained")
    try:
        animation = animate_double_well(fig=figure, **kwargs)
        animation.save(target, writer=PillowWriter(fps=fps), dpi=dpi)
    finally:
        plt.close(figure)
    if contact_sheet is not None:
        _save_contact_sheet(target, Path(contact_sheet))
    return target


def require_matplotlib() -> ModuleType:
    """Return the pyplot module, or explain which extra to install.

    This is the hook a caller reaches for before a piece of work that ends in a
    figure, so that an install without the optional dependency is told so at
    the start rather than after the work. [`msrelapse.cli`][msrelapse.cli]
    calls it before a reproduction that draws.

    Returns
    -------
    types.ModuleType
        The ``matplotlib.pyplot`` module.

    Raises
    ------
    ImportError
        If matplotlib is not installed. The message names the extra to install,
        which is ``msrelapse[plot]``.

    Examples
    --------
    >>> require_matplotlib().__name__
    'matplotlib.pyplot'
    """
    try:
        import matplotlib.pyplot as plt  # noqa: PLC0415
    except ImportError as error:
        raise ImportError(
            "drawing a figure needs matplotlib, which is not installed; install the plot "
            "extra of this package, that is msrelapse[plot]"
        ) from error
    return plt


def _fig6_panels() -> tuple[Axes, ...]:
    """Draw both panels of Figure 6, one per control parameter of the article.

    [`fig6_asymmetric_potential`][msrelapse.plots.fig6_asymmetric_potential]
    draws a single potential, which is what a caller composing a figure of their
    own wants. Figure 6 of the paper is that panel at each of the two control
    parameters its Figure 5 also uses, so the reproduction composes the pair the
    way the module invites a caller to, by handing the figure function one of the
    panels it made.

    Returns
    -------
    tuple of matplotlib.axes.Axes
        The panels that were drawn on, in the order of the article.

    Raises
    ------
    ImportError
        If matplotlib is not installed.
    """
    panels = _panel_axes(None, len(_FIG5_ALPHAS), stacked=False)
    for panel, alpha in zip(panels, _FIG5_ALPHAS, strict=True):
        fig6_asymmetric_potential(alpha=alpha, ax=panel)
    return panels


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
    plt = require_matplotlib()
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
    plt = require_matplotlib()
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
    plt = require_matplotlib()
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
    """Return the name a panel of this module writes a state code as.

    Parameters
    ----------
    state : int
        Clinical code, +1 for no health and -1 for health.

    Returns
    -------
    str
        'flare' or 'health'. The article calls the plus one state no health;
        every label, title and note drawn here says flare instead, the word a
        clinic uses for it.
    """
    return _FLARE_LABEL.lower() if state == _NO_HEALTH else _HEALTH_LABEL.lower()


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

    The same reading as [`msrelapse.fit`][msrelapse.fit] applies to its counts:
    one value per patient, in one dimension, and every one of them a real
    number of weeks.

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
    """Return the values to tick an axis at, at most ``_MAX_BIN_TICKS`` of them.

    Ticking the bin edges themselves is what the paper does, so that a reader
    can see where a bin begins, and every other candidate is dropped as often
    as it takes to keep the labels apart. At the bins of Figure 4 this gives
    the axes of the paper: every edge on the no health panel and every second
    one, that is every 200 weeks, on the health panel. The relapse count axis
    of [`fig_poisson_to_nb`][msrelapse.plots.fig_poisson_to_nb] is thinned the
    same way, from every count.

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


def _draw_asymmetric_potential(
    ax: Axes,
    well: DoubleWell,
    points: CriticalPoints,
    *,
    guides: Sequence[float] | None = None,
) -> None:
    """Draw a tilted double well with both of its barriers marked.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The panel to draw on.
    well : DoubleWell
        The potential to draw.
    points : CriticalPoints
        Its three stationary points, from ``_well_and_points``.
    guides : sequence of float, optional
        The stationary points whose level gets a dashed horizontal guide. The
        default marks all three, which is what Figure 6 of the paper draws;
        Figure 8 marks the saddle alone.

    Returns
    -------
    None
        Nothing is returned; the panel is drawn on `ax`.
    """
    _draw_potential(ax, well)
    top = float(well.V(points.saddle))
    health = float(well.V(points.health))
    relapse = float(well.V(points.relapse))
    marked = (points.health, points.saddle, points.relapse) if guides is None else tuple(guides)
    for position in marked:
        _guide_line(ax, float(well.V(position)))
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


def _fitted_survival(grid: _Vector, rate: float) -> _Vector:
    """Return the survival of a fitted duration law over a grid of weeks.

    Parameters
    ----------
    grid : numpy.ndarray
        The times to read the survival at, in weeks.
    rate : float
        The weekly rate
        [`msrelapse.fit.fit_durations`][msrelapse.fit.fit_durations] returned,
        which is the number of complete runs over the weeks they were all at
        risk for.

    Returns
    -------
    numpy.ndarray
        ``(1 - rate) ** t`` at each time, the fit read on the whole weeks the
        durations are recorded in.

    Notes
    -----
    A rate of one, which only a state whose runs all lasted a single week gives,
    is kept out of the logarithm: the law then ends every run in its first week,
    so the survival is one at time zero and zero after it.
    """
    if rate >= 1.0:
        return np.where(grid > 0.0, 0.0, 1.0).astype(np.float64)
    return np.exp(grid * math.log1p(-rate))


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
    fewer than [`msrelapse.fit.MIN_AT_RISK`][msrelapse.fit.MIN_AT_RISK]
    durations and a per patient call of the paper's own sample patients, gets
    an inset saying so rather than an empty pair of axes with a lone guide line
    across it.
    """
    weeks, hazard, _at_risk = discrete_hazard(complete)
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
            f"reaches a week with {MIN_AT_RISK} still at risk",
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


@dataclass(frozen=True)
class _AnimationRecord:
    """One simulated path and the two weekly series the animation draws from it.

    Attributes
    ----------
    well : DoubleWell
        The potential the path was drawn on.
    dt : float
        Spacing of the samples of `x`, in weeks.
    x : numpy.ndarray
        The path itself, one row of
        [`msrelapse.simulate.simulate_paths`][msrelapse.simulate.simulate_paths].
    weekly : numpy.ndarray
        The weekly states, one entry per whole week of the record.
    edss : numpy.ndarray
        The continuous illustrative EDSS of
        [`msrelapse.edss`][msrelapse.edss], of the same length as `weekly`.
    edss_display : numpy.ndarray
        That same score on the half point grid of the scale.
    baseline : float
        The score the trace opens at, which the panel marks with a reference
        line.
    """

    well: DoubleWell
    dt: float
    x: _Vector
    weekly: _States
    edss: _Vector
    edss_display: _Vector
    baseline: float


def _animation_record(  # noqa: PLR0917
    well: DoubleWell | None,
    sigma: float | None,
    n_weeks: int,
    dt: float,
    band_fraction: float,
    rng: Seed,
    spec: EDSSSpec | None,
) -> _AnimationRecord:
    """Simulate the one record the animation plays back.

    Parameters
    ----------
    well : DoubleWell or None
        The potential, or None for the calibrated one.
    sigma : float or None
        Noise amplitude, or None for the calibrated one.
    n_weeks : int
        Length of the record, in whole weeks.
    dt : float
        Time step of the integration, in weeks.
    band_fraction : float
        Position of the two hysteresis thresholds.
    rng : numpy.random.Generator or int or None
        Generator to draw the noise from, or a seed.
    spec : EDSSSpec or None
        The parameters of the illustrative disability trace, or None for the
        published defaults.

    Returns
    -------
    _AnimationRecord
        The path, the weekly states and the disability trace they drive.

    Raises
    ------
    ValueError
        If `n_weeks` is below one, or if any argument of the integration or of
        the mapping to states is out of range.

    Notes
    -----
    One generator serves the whole record, and the path is drawn from it first,
    so that the two top panels hold what they held before the bottom one drew
    anything: a seed that gave a path gives that same path still.
    """
    if n_weeks < 1:
        raise ValueError(f"n_weeks must be at least 1, got {n_weeks!r}")
    parameters = EDSSSpec() if spec is None else spec
    generator = _generator(rng)
    potential, amplitude = _animation_parameters(well, sigma)
    paths = simulate_paths(potential, amplitude, float(n_weeks) * _WEEK, dt=dt, rng=generator)
    # The same reading of an episode simulate_weekly applies: the hysteresis
    # band, with the Brownian bridge shift that removes the bias of a crossing
    # tested only at the grid points.
    states = to_states(
        paths.x[0],
        potential,
        band_fraction,
        level_shift=BRIDGE_CONSTANT * amplitude * math.sqrt(dt),
    )
    weekly = to_weekly(states, dt)
    trace = edss_trajectory(weekly, parameters, generator)
    return _AnimationRecord(
        well=potential,
        dt=dt,
        x=paths.x[0],
        weekly=weekly,
        edss=trace["edss"].to_numpy(),
        edss_display=trace["edss_display"].to_numpy(),
        baseline=parameters.baseline,
    )


def _animation_parameters(well: DoubleWell | None, sigma: float | None) -> tuple[DoubleWell, float]:
    """Return the potential and the noise the animation runs at.

    Parameters
    ----------
    well : DoubleWell or None
        The potential the caller asked for, or None.
    sigma : float or None
        The noise amplitude the caller asked for, or None.

    Returns
    -------
    tuple of DoubleWell and float
        The potential and the noise. Whichever of the two the caller left out
        comes from the calibration of the two mean durations of the paper, at
        the reference control parameter.
    """
    if well is not None and sigma is not None:
        return well, float(sigma)
    beta, calibrated = calibrate(
        PAPER.tau_health_cohort_weeks.value, PAPER.tau_no_health_cohort_weeks.value
    )
    return (
        DoubleWell(PAPER.alpha_reference.value, beta) if well is None else well,
        calibrated if sigma is None else float(sigma),
    )


def _animation_weeks(n_weeks: int, weeks_per_frame: float) -> list[int]:
    """Return how many weeks of the record each frame of the animation shows.

    Parameters
    ----------
    n_weeks : int
        Length of the record, in whole weeks.
    weeks_per_frame : float
        How many weeks one frame advances by.

    Returns
    -------
    list of int
        One count of weeks per frame, rising and ending at the whole record, so
        that the last frame holds everything.

    Raises
    ------
    ValueError
        If `weeks_per_frame` is not a positive finite number of weeks.
    """
    if not math.isfinite(weeks_per_frame) or weeks_per_frame <= 0.0:
        raise ValueError(
            f"weeks_per_frame must be a positive finite number of weeks, got {weeks_per_frame!r}"
        )
    n_frames = math.ceil(n_weeks / weeks_per_frame)
    return [min(n_weeks, math.ceil((frame + 1) * weeks_per_frame)) for frame in range(n_frames)]


def _draw_animation(
    figure: Figure,
    record: _AnimationRecord,
    weeks: Sequence[int],
) -> Callable[[int], None]:
    """Lay the three panels out and return the function that advances them.

    Parameters
    ----------
    figure : matplotlib.figure.Figure
        The figure to draw the three panels on.
    record : _AnimationRecord
        The record to play back.
    weeks : sequence of int
        How many weeks each frame shows, from ``_animation_weeks``.

    Returns
    -------
    callable
        The function one frame index is passed to. It is called once here, so
        that the figure holds the opening frame before anything plays it.
    """
    grid = figure.add_gridspec(2, 2, height_ratios=_ANIMATION_HEIGHT_RATIOS)
    potential_panel = figure.add_subplot(grid[0, 0])
    series_panel = figure.add_subplot(grid[0, 1])
    edss_panel = figure.add_subplot(grid[1, :])
    n_weeks = int(record.weekly.size)
    particle = _draw_animated_potential(potential_panel, record.well)
    step, current = _draw_animated_series(series_panel, n_weeks)
    curve, displayed = _draw_animated_edss(edss_panel, record, n_weeks)

    def update(frame: int) -> None:
        drawn = weeks[frame]
        position = float(record.x[_sample_at(record, drawn)])
        particle.set_data([position], [float(record.well.V(position))])
        # The record is held in weeks and shown in years, so the week number of
        # each point is divided by the weeks of a year on its way to a panel.
        years = np.arange(drawn, dtype=np.float64) / WEEKS_PER_YEAR
        states = record.weekly[:drawn]
        # Week k covers the interval from k to k + 1, so the step is closed one
        # week past the current one, as Figure 2 closes a whole record. Without
        # that point the week the marker sits on would have no width at all.
        step.set_data(np.append(years, drawn / WEEKS_PER_YEAR), np.append(states, states[-1]))
        current.set_data([years[-1]], [states[-1]])
        curve.set_data(years, record.edss[:drawn])
        displayed.set_data(years, record.edss_display[:drawn])
        series_panel.set_title(
            f"Weekly record, year {drawn / WEEKS_PER_YEAR:.1f} of {n_weeks / WEEKS_PER_YEAR:.1f}"
        )

    update(0)
    return update


def _draw_animated_potential(ax: Axes, well: DoubleWell) -> Line2D:
    """Draw the potential of the animation and return the particle on it.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The panel to draw on.
    well : DoubleWell
        The potential the particle moves in.

    Returns
    -------
    matplotlib.lines.Line2D
        The particle, an empty line the caller moves one frame at a time.
    """
    _draw_potential(ax, well)
    points = well.critical_points()
    top = float(well.V(points.saddle))
    ax.plot(
        [points.saddle],
        [top],
        marker="o",
        markersize=_SADDLE_SIZE,
        linestyle="none",
        color="tab:blue",
        label=_SADDLE_LABEL,
    )
    ax.text(points.saddle, top + _LABEL_OFFSET, _SADDLE_LABEL, ha="center", va="bottom")
    for position, name in ((points.health, _HEALTH_LABEL), (points.relapse, _FLARE_LABEL)):
        ax.text(position, float(well.V(position)) - _LABEL_OFFSET, name, ha="center", va="top")
    ax.set_title("The particle in the double well")
    (particle,) = ax.plot(
        [],
        [],
        marker="o",
        markersize=_PARTICLE_SIZE,
        linestyle="none",
        color="tab:red",
        label=_PARTICLE_LABEL,
    )
    return particle


def _draw_animated_series(ax: Axes, n_weeks: int) -> tuple[Line2D, Line2D]:
    """Set the weekly panel up and return its step and its current week mark.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The panel to draw on.
    n_weeks : int
        Length of the whole record, in weeks, which fixes the time axis.

    Returns
    -------
    tuple of matplotlib.lines.Line2D
        The step of the record so far and the mark on the current week, both
        empty until the first frame.
    """
    (step,) = ax.plot([], [], drawstyle="steps-post", linewidth=1.0, color=_CURVE_COLOUR)
    (current,) = ax.plot(
        [],
        [],
        marker="o",
        markersize=_CURRENT_WEEK_SIZE,
        linestyle="none",
        color="tab:red",
        label=_CURRENT_WEEK_LABEL,
    )
    _time_axis_in_years(ax, n_weeks)
    ax.set_yticks([_HEALTH, _NO_HEALTH])
    ax.set_yticklabels([_HEALTH_LABEL, _FLARE_LABEL])
    ax.set_ylim(_HEALTH - _STATE_MARGIN, _NO_HEALTH + _STATE_MARGIN)
    return step, current


def _draw_animated_edss(ax: Axes, record: _AnimationRecord, n_weeks: int) -> tuple[Line2D, Line2D]:
    """Set the disability panel up and return the two traces that grow across it.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The panel to draw on.
    record : _AnimationRecord
        The record being played back, which fixes the height of the panel and
        the level the baseline is marked at.
    n_weeks : int
        Length of the whole record, in weeks, which fixes the time axis.

    Returns
    -------
    tuple of matplotlib.lines.Line2D
        The continuous score and the displayed half point score, both empty
        until the first frame.

    Notes
    -----
    The panel is the illustrative EDSS trajectory of
    [`msrelapse.edss`][msrelapse.edss]. It is an extension of this package and
    not a quantity the article reports.

    It is the one panel of the figure that carries a legend, because it is the
    one holding three lines a reader has to tell apart. The two panels above it
    hold a single curve each and name what it is in the title or the axis.
    """
    (curve,) = ax.plot([], [], linewidth=0.9, color=_CURVE_COLOUR, label=_EDSS_LABEL)
    (displayed,) = ax.plot(
        [],
        [],
        drawstyle="steps-post",
        linewidth=1.6,
        color="tab:red",
        label=_EDSS_DISPLAY_LABEL,
    )
    ax.axhline(
        record.baseline,
        linestyle=":",
        linewidth=1.0,
        color=_GUIDE_COLOUR,
        label=_EDSS_BASELINE_LABEL,
    )
    _time_axis_in_years(ax, n_weeks)
    # A quiet record keeps the whole of the lower half of the scale, so that a
    # small rise is not drawn as a large one by an axis fitted to it.
    top = max(_EDSS_PANEL_FLOOR, math.ceil(float(record.edss.max()) + _EDSS_HEADROOM))
    ax.set_ylim(0.0, top)
    ax.set_ylabel(_EDSS_AXIS_LABEL)
    # The key sits in the upper left, which the trace reaches only on a record
    # that has climbed most of the drawn scale.
    ax.legend(loc="upper left", ncols=_EDSS_LEGEND_COLUMNS, fontsize="small", framealpha=1.0)
    return curve, displayed


def _time_axis_in_years(ax: Axes, n_weeks: int) -> None:
    """Set the time axis of one panel of the animation, in years of follow up.

    The animation counts in weeks throughout, as the study does, and shows
    those weeks as the years of follow up they come to, which is the unit a
    reader of a ten year record thinks in.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The panel whose time axis to set.
    n_weeks : int
        Length of the whole record, in weeks.

    Returns
    -------
    None
        Nothing is returned; the axis of `ax` is set.
    """
    years = n_weeks / WEEKS_PER_YEAR
    ax.set_xlim(0.0, years)
    # A record of at least a year is ticked at every whole year it reaches,
    # which is the eleven ticks of the ten year default. A shorter one keeps the
    # ticks matplotlib picks for it, since the only whole year inside it is zero
    # and an axis ticked there alone carries no scale at all.
    if years >= 1.0:
        ax.set_xticks(np.arange(math.floor(years) + 1.0))
    ax.set_xlabel(_TIME_AXIS_LABEL)


def _sample_at(record: _AnimationRecord, weeks: int) -> int:
    """Return the index of the sample of a path at the end of a whole week.

    Parameters
    ----------
    record : _AnimationRecord
        The record being played back.
    weeks : int
        How many whole weeks have been drawn.

    Returns
    -------
    int
        The index into the path, never past its last sample.
    """
    return min(record.x.size - 1, round(weeks * _WEEK / record.dt))


def _save_contact_sheet(gif: Path, path: Path) -> Path:
    """Write four evenly spaced frames of a finished animation side by side.

    The frames are read back out of the written file, so the sheet shows what
    the file holds rather than a second drawing of the same record.

    Parameters
    ----------
    gif : pathlib.Path
        The animation to read.
    path : pathlib.Path
        The PNG file to write. Its directory is created if it does not exist.

    Returns
    -------
    pathlib.Path
        The file that was written.

    Raises
    ------
    ImportError
        If matplotlib is not installed.
    """
    from PIL import Image, ImageSequence  # noqa: PLC0415

    path.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(gif) as playback:
        # Each frame is converted while it is the current one, because the
        # iterator seeks through the file rather than holding the frames.
        frames = [frame.convert("RGB") for frame in ImageSequence.Iterator(playback)]
        chosen = _evenly_spaced(len(frames), _CONTACT_SHEET_PANELS)
        images = [np.asarray(frames[index]) for index in chosen]
    panels = _panel_axes(None, len(images), stacked=False)
    for panel, index, image in zip(panels, chosen, images, strict=True):
        panel.imshow(image)
        panel.set_axis_off()
        panel.set_title(f"frame {index + 1} of {len(frames)}", fontsize="small")
    return _save_figure(panels[0], path)


def _evenly_spaced(n_frames: int, n_panels: int) -> list[int]:
    """Return the indices of a few frames spread over the whole animation.

    Parameters
    ----------
    n_frames : int
        How many frames the animation holds.
    n_panels : int
        How many of them to pick, at least two.

    Returns
    -------
    list of int
        The indices, the first and the last frame among them. An animation of
        fewer frames than panels repeats one of them.
    """
    positions = np.rint(np.linspace(0.0, float(n_frames - 1), n_panels))
    return [int(position) for position in positions]
