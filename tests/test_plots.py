from __future__ import annotations

import sys
from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from msrelapse import fit, plots
from msrelapse._params import PAPER, symmetric_barrier
from msrelapse.io import events_to_weekly, weekly_to_durations
from msrelapse.model import DoubleWell
from msrelapse.renewal import alternating_renewal

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.lines import Line2D

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")
plt = pytest.importorskip("matplotlib.pyplot")

NO_HEALTH = PAPER.state_no_health.value
HEALTH = PAPER.state_health.value
ALPHA = PAPER.alpha_reference.value
FIG5_ALPHAS = (PAPER.alpha_reference.value, PAPER.alpha_illustrative_low.value)
PATIENT_BETAS = (
    PAPER.beta_patient_23.value,
    PAPER.beta_patient_32.value,
    PAPER.beta_patient_53.value,
)

# A small cohort: twenty patients over four hundred weeks, with remissions of
# forty weeks and relapses of four on average. Whole week durations keep the
# record in the shape the weekly scale of the study produces.
COHORT_PATIENTS = 20
COHORT_WEEKS = 400.0
COHORT_REMISSION_WEEKS = 40.0
COHORT_RELAPSE_WEEKS = 4.0
COHORT_SEED = 20130910

# A cohort whose onset rate plainly varies from patient to patient, so that the
# negative binomial fit keeps a dispersion above zero.
OVERDISPERSED_COUNTS = [0, 0, 0, 1, 1, 2, 12, 15, 20, 30]

# An asymmetry far past the fold of the potential at any control parameter this
# package draws, so that the well has one bottom rather than two and the figures
# that mark both of them refuse it.
BETA_BEYOND_FOLD = 100.0

EXPECTED_FILE_NAMES = [
    "fig2_sample_patients.png",
    "fig3_rr_phase_histogram.png",
    "fig4_duration_histograms.png",
    "fig5_symmetric_potentials.png",
    "fig6_asymmetric_potential.png",
    "fig7_simulated_paths.png",
    "fig8_patient_potentials.png",
    "fig_survival_vs_exponential.png",
    "fig_poisson_to_nb.png",
]


@pytest.fixture(autouse=True)
def close_figures() -> Iterator[None]:
    yield
    plt.close("all")


@pytest.fixture(scope="module")
def cohort() -> tuple[pd.DataFrame, pd.DataFrame]:
    events = alternating_renewal(
        1.0 / COHORT_REMISSION_WEEKS,
        1.0 / COHORT_RELAPSE_WEEKS,
        COHORT_WEEKS,
        n=COHORT_PATIENTS,
        rng=COHORT_SEED,
        discretise="week",
    )
    weekly = events_to_weekly(events)
    return weekly, weekly_to_durations(weekly)


@pytest.fixture
def weekly(cohort: tuple[pd.DataFrame, pd.DataFrame]) -> pd.DataFrame:
    return cohort[0]


@pytest.fixture
def runs(cohort: tuple[pd.DataFrame, pd.DataFrame]) -> pd.DataFrame:
    return cohort[1]


def bar_heights(ax: Axes) -> list[float]:
    return [float(patch.get_height()) for patch in ax.containers[0]]


def dashed_levels(ax: Axes) -> list[float]:
    return sorted(
        float(np.asarray(line.get_ydata())[0]) for line in ax.lines if line.get_linestyle() == "--"
    )


def solid_lines(ax: Axes) -> list[Line2D]:
    return [line for line in ax.lines if line.get_linestyle() == "-"]


def texts_of(ax: Axes) -> list[str]:
    return [text.get_text() for text in ax.texts]


def label_positions(ax: Axes) -> dict[str, float]:
    """Return where along x each labelled piece of text sits.

    The two barrier arrows carry no text of their own, so they share the empty
    key and neither of them is ever looked up here.
    """
    return {text.get_text(): float(text.get_position()[0]) for text in ax.texts}


def ydata(line: Line2D) -> Any:
    return np.asarray(line.get_ydata(), dtype=np.float64)


def xdata(line: Line2D) -> Any:
    return np.asarray(line.get_xdata(), dtype=np.float64)


def relapse_counts(runs: pd.DataFrame) -> pd.Series[int]:
    relapses = runs["state"] == NO_HEALTH
    return relapses.groupby(runs["patient_id"], sort=True).sum().astype("int64")


def record_lengths(weekly: pd.DataFrame) -> pd.Series[int]:
    return weekly.groupby("patient_id", sort=True).size()


def durations_frame(rows: list[tuple[str, int, int, int, bool]]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "patient_id": pd.Series([row[0] for row in rows], dtype="str"),
            "run_index": pd.Series([row[1] for row in rows], dtype="int64"),
            "state": pd.Series([row[2] for row in rows], dtype="int64"),
            "duration_w": pd.Series([row[3] for row in rows], dtype="int64"),
            "censored": pd.Series([row[4] for row in rows], dtype="bool"),
        }
    )


def relapse_only() -> pd.DataFrame:
    """Return a frame of one relapse and no remission at all."""
    return durations_frame([("p0001", 0, NO_HEALTH, 3, False)])


def four_relapses() -> pd.DataFrame:
    """Return four complete relapses, one per patient, of 1, 2, 3 and 4 weeks.

    Too few for any week to hold the number still at risk that a hazard is read
    at, which is the shape a single patient of the paper has.
    """
    return durations_frame([(f"p000{index}", 0, NO_HEALTH, index, False) for index in range(1, 5)])


def one_week_relapses() -> pd.DataFrame:
    """Return five complete relapses, one per patient, of a single week each.

    Every week at risk ends in an event, so the fitted rate is one and the law
    the survival panel draws is the degenerate one.
    """
    return durations_frame([(f"p000{index}", 0, NO_HEALTH, 1, False) for index in range(1, 6)])


def censored_remissions() -> pd.DataFrame:
    """Return five remissions of 1, 2, 2, 3 and 5 weeks, the 2nd and 5th censored.

    A remission is censored only when it is the last run of its patient, which
    is the only shape the durations schema allows, so a patient whose remission
    is complete carries a relapse after it.
    """
    return durations_frame(
        [
            ("p1", 0, HEALTH, 1, False),
            ("p1", 1, NO_HEALTH, 1, False),
            ("p2", 0, NO_HEALTH, 1, False),
            ("p2", 1, HEALTH, 2, True),
            ("p3", 0, HEALTH, 2, False),
            ("p3", 1, NO_HEALTH, 1, False),
            ("p4", 0, HEALTH, 3, False),
            ("p4", 1, NO_HEALTH, 1, False),
            ("p5", 0, NO_HEALTH, 1, False),
            ("p5", 1, HEALTH, 5, True),
        ]
    )


def step_line(ax: Axes) -> Line2D:
    return next(line for line in ax.lines if "steps" in line.get_drawstyle())


def test_fig2_draws_one_step_line_per_patient(weekly: pd.DataFrame) -> None:
    axes = plots.fig2_sample_patients(weekly)

    assert len(axes) == 3
    for ax in axes:
        assert len(ax.lines) == 1
        assert "steps" in ax.lines[0].get_drawstyle()


def test_fig2_labels_the_two_states(weekly: pd.DataFrame) -> None:
    ax = plots.fig2_sample_patients(weekly)[0]

    labels = [label.get_text() for label in ax.get_yticklabels()]
    assert labels == ["Health", "No health"]
    assert list(ax.get_yticks()) == [HEALTH, NO_HEALTH]


def test_fig2_plots_the_named_patients(weekly: pd.DataFrame) -> None:
    chosen = ["p0005", "p0011"]

    axes = plots.fig2_sample_patients(weekly, chosen)

    assert [ax.get_title() for ax in axes] == chosen
    states = weekly.loc[weekly["patient_id"] == chosen[0], "state"].to_numpy()
    # The step carries one closing point, so that the last week of the record is
    # as wide on the axis as every week before it.
    assert ydata(axes[0].lines[0]) == pytest.approx(np.append(states, states[-1]))


def test_fig2_gives_the_last_week_its_full_width(weekly: pd.DataFrame) -> None:
    chosen = str(weekly["patient_id"].iloc[0])

    ax = plots.fig2_sample_patients(weekly, [chosen])[0]

    last_week = float(weekly.loc[weekly["patient_id"] == chosen, "week"].max())
    assert xdata(ax.lines[0])[-1] == pytest.approx(last_week + 1.0)
    assert ax.get_xlim()[1] == pytest.approx(last_week + 1.0)


def test_fig2_refuses_a_frame_with_no_patient_in_it() -> None:
    empty = pd.DataFrame(
        {
            "patient_id": pd.Series([], dtype="str"),
            "week": pd.Series([], dtype="int64"),
            "state": pd.Series([], dtype="int64"),
        }
    )

    with pytest.raises(ValueError, match="no patient"):
        plots.fig2_sample_patients(empty)


def test_fig2_names_a_patient_that_is_not_in_the_frame(weekly: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="p9999"):
        plots.fig2_sample_patients(weekly, ["p9999"])


def test_fig3_draws_the_digitised_counts_by_default() -> None:
    ax = plots.fig3_rr_phase_histogram()

    assert bar_heights(ax) == pytest.approx(list(PAPER.fig3_counts.value))
    assert "Digitised" in ax.get_title()


def test_fig3_bins_a_cohort_on_the_paper_edges(weekly: pd.DataFrame) -> None:
    lengths = record_lengths(weekly)
    expected, _ = np.histogram(lengths, bins=np.asarray(PAPER.fig3_bin_edges_weeks.value))

    ax = plots.fig3_rr_phase_histogram(lengths)

    assert bar_heights(ax) == pytest.approx(expected)
    assert list(ax.get_xticks()) == pytest.approx(list(PAPER.fig3_bin_edges_weeks.value))


def test_fig2_refuses_an_empty_list_of_patients(weekly: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="at least one patient"):
        plots.fig2_sample_patients(weekly, [])


def test_fig3_refuses_a_cohort_with_no_record_in_it() -> None:
    with pytest.raises(ValueError, match="at least one record length"):
        plots.fig3_rr_phase_histogram([])


def test_fig3_reports_how_many_records_fall_inside_the_paper_bins() -> None:
    edges = PAPER.fig3_bin_edges_weeks.value
    inside = 0.5 * (edges[0] + edges[1])

    ax = plots.fig3_rr_phase_histogram([inside, edges[-1] + 100.0])

    assert sum(bar_heights(ax)) == pytest.approx(1.0)
    assert "1 of 2" in ax.get_title()


def test_fig3_refuses_a_two_dimensional_array_of_record_lengths() -> None:
    with pytest.raises(ValueError, match="followup_weeks"):
        plots.fig3_rr_phase_histogram(np.full((2, 2), 100.0))


def test_fig4_draws_the_digitised_counts_by_default() -> None:
    first, second = plots.fig4_duration_histograms()

    assert bar_heights(first) == pytest.approx(list(PAPER.fig4a_counts.value))
    assert bar_heights(second) == pytest.approx(list(PAPER.fig4b_counts.value))
    # The no health panel is ticked at every bin edge and the health panel at
    # every second one, which is every 200 weeks, as the paper ticks them.
    width_b = PAPER.fig4b_bin_width_weeks.value
    assert list(first.get_xticks()) == pytest.approx(
        [
            PAPER.relapse_duration_min_weeks.value + PAPER.fig4a_bin_width_weeks.value * step
            for step in range(len(PAPER.fig4a_counts.value) + 1)
        ]
    )
    assert list(second.get_xticks()) == pytest.approx(
        [2.0 * width_b * step for step in range(1 + len(PAPER.fig4b_counts.value) // 2)]
    )
    annotation = texts_of(first)[0]
    assert str(PAPER.n_no_health_events.value) in annotation
    assert f"{PAPER.tau_no_health_cohort_weeks.value:.3g}" in annotation


def test_fig4_bins_the_cohort_durations(runs: pd.DataFrame) -> None:
    first, second = plots.fig4_duration_histograms(runs)

    relapses = runs.loc[runs["state"] == NO_HEALTH, "duration_w"]
    remissions = runs.loc[runs["state"] == HEALTH, "duration_w"]
    assert sum(bar_heights(first)) == pytest.approx(relapses.size)
    assert sum(bar_heights(second)) == pytest.approx(remissions.size)
    assert first.containers[0][0].get_x() == pytest.approx(PAPER.relapse_duration_min_weeks.value)
    assert first.containers[0][0].get_width() == pytest.approx(PAPER.fig4a_bin_width_weeks.value)
    assert second.containers[0][0].get_width() == pytest.approx(PAPER.fig4b_bin_width_weeks.value)
    assert str(relapses.size) in texts_of(first)[0]
    assert "This cohort" in first.get_title()


def test_fig5_draws_one_symmetric_potential_per_alpha() -> None:
    axes = plots.fig5_symmetric_potentials()

    assert len(axes) == len(FIG5_ALPHAS)
    for ax, alpha in zip(axes, FIG5_ALPHAS, strict=True):
        assert len(solid_lines(ax)) == 1
        assert dashed_levels(ax) == pytest.approx([-symmetric_barrier(alpha), 0.0])
        assert r"$\Delta V$" in texts_of(ax)
        assert ax.get_xlim() == pytest.approx(PAPER.potential_plot_x_limits.value)
        assert ax.get_ylim() == pytest.approx(PAPER.potential_plot_v_limits.value)
        # The sign convention of the paper: x1, health, is the left well and x2,
        # no health, the right one. Reading the labels by position rather than by
        # membership is what catches a swap of the two.
        points = DoubleWell(alpha, PAPER.beta_symmetric.value).critical_points()
        by_label = label_positions(ax)
        assert by_label[r"$x_1$"] == pytest.approx(points.health)
        assert by_label[r"$x_0$"] == pytest.approx(points.saddle)
        assert by_label[r"$x_2$"] == pytest.approx(points.relapse)


def test_fig5_curve_is_the_potential_of_the_paper() -> None:
    ax = plots.fig5_symmetric_potentials((PAPER.alpha_illustrative_low.value,))[0]

    line = solid_lines(ax)[0]
    well = DoubleWell(PAPER.alpha_illustrative_low.value, PAPER.beta_symmetric.value)
    assert ydata(line) == pytest.approx(well.V(xdata(line)))


def test_fig6_marks_both_barriers() -> None:
    ax = plots.fig6_asymmetric_potential()

    well = DoubleWell(ALPHA, PAPER.beta_illustrative.value)
    points = well.critical_points()
    assert dashed_levels(ax) == pytest.approx(
        sorted(float(well.V(x)) for x in (points.health, points.saddle, points.relapse))
    )
    # delta V1 is the climb out of the deeper health well on the left and delta
    # V2 the climb out of the shallower no health well on the right, so the two
    # arrows and the two well labels are read by position rather than by
    # membership: a swap of either pair is exactly what this pins.
    by_label = label_positions(ax)
    assert by_label[r"$\Delta V_1$"] < points.saddle < by_label[r"$\Delta V_2$"]
    assert by_label[r"$x_1$"] == pytest.approx(points.health)
    assert by_label[r"$x_0$"] == pytest.approx(points.saddle)
    assert by_label[r"$x_2$"] == pytest.approx(points.relapse)


def test_fig6_draws_the_asymmetry_it_is_given() -> None:
    ax = plots.fig6_asymmetric_potential(beta=PAPER.beta_patient_53.value)

    line = solid_lines(ax)[0]
    well = DoubleWell(ALPHA, PAPER.beta_patient_53.value)
    assert ydata(line) == pytest.approx(well.V(xdata(line)))


def test_fig7_draws_one_path_per_beta() -> None:
    axes = plots.fig7_simulated_paths(t_end=20.0, dt=0.05, rng=3)

    assert len(axes) == 2
    for ax in axes:
        assert len(ax.lines) == 1
        assert ax.get_ylim() == pytest.approx(PAPER.fig7_x_limits.value)
        assert f"{PAPER.epsilon_noise_variance.value:g}" in ax.get_title()
    assert f"{PAPER.beta_illustrative.value:g}" in axes[1].get_title()


def test_fig7_is_reproducible_from_its_seed() -> None:
    first = plots.fig7_simulated_paths(t_end=20.0, dt=0.05, rng=11)
    second = plots.fig7_simulated_paths(t_end=20.0, dt=0.05, rng=11)

    for one, other in zip(first, second, strict=True):
        assert ydata(one.lines[0]) == pytest.approx(ydata(other.lines[0]))


def test_fig7_takes_a_generator_as_readily_as_a_seed() -> None:
    first = plots.fig7_simulated_paths(t_end=20.0, dt=0.05, rng=np.random.default_rng(7))
    second = plots.fig7_simulated_paths(t_end=20.0, dt=0.05, rng=7)

    for one, other in zip(first, second, strict=True):
        assert ydata(one.lines[0]) == pytest.approx(ydata(other.lines[0]))


def test_fig7_panels_carry_different_noise() -> None:
    axes = plots.fig7_simulated_paths(
        betas=(PAPER.beta_symmetric.value, PAPER.beta_symmetric.value),
        t_end=20.0,
        dt=0.05,
        rng=5,
    )

    assert not np.allclose(ydata(axes[0].lines[0]), ydata(axes[1].lines[0]))


def test_fig8_draws_the_three_sample_patients() -> None:
    axes = plots.fig8_patient_potentials()

    assert len(axes) == len(PATIENT_BETAS)
    for ax, beta in zip(axes, PATIENT_BETAS, strict=True):
        well = DoubleWell(ALPHA, beta)
        line = solid_lines(ax)[0]
        assert ydata(line) == pytest.approx(well.V(xdata(line)))
        assert ax.get_title().startswith("Patient")
        # Every fitted patient of the paper is left skewed, so each panel has to
        # carry delta V1 on the health side and delta V2 on the no health side.
        points = well.critical_points()
        by_label = label_positions(ax)
        assert by_label[r"$\Delta V_1$"] < points.saddle < by_label[r"$\Delta V_2$"]
        assert by_label[r"$x_1$"] == pytest.approx(points.health)
        assert by_label[r"$x_2$"] == pytest.approx(points.relapse)


def test_fig8_marks_the_saddle_level_alone() -> None:
    axes = plots.fig8_patient_potentials()

    # Figure 8 of the paper draws one dashed line per panel, through V(x0), the
    # level both barriers hang from, where Figure 6 draws one through each of
    # the three stationary levels.
    for ax, beta in zip(axes, PATIENT_BETAS, strict=True):
        well = DoubleWell(ALPHA, beta)
        saddle = well.critical_points().saddle
        assert dashed_levels(ax) == pytest.approx([float(well.V(saddle))])


def test_fig8_needs_one_label_per_asymmetry() -> None:
    with pytest.raises(ValueError, match="one label per panel"):
        plots.fig8_patient_potentials(
            betas=(PAPER.beta_illustrative.value, PAPER.beta_symmetric.value),
            labels=("the only label",),
        )


def test_fig8_refuses_asymmetries_that_are_not_the_ones_of_the_sample_patients() -> None:
    # Three asymmetries of somebody else's choosing, which the default titles
    # would otherwise hand the identifiers of the paper's own three patients.
    with pytest.raises(ValueError, match="labels must be given"):
        plots.fig8_patient_potentials(betas=(0.05, 0.10, 0.30))


def test_axes_that_are_passed_in_are_the_ones_returned() -> None:
    figure, grid = plt.subplots(1, 2)

    axes = plots.fig5_symmetric_potentials(axes=list(grid))

    assert axes == tuple(grid)
    assert axes[0].get_figure() is figure


def test_an_axes_passed_to_a_single_panel_figure_is_the_one_returned() -> None:
    ax = plt.subplots()[1]

    assert plots.fig6_asymmetric_potential(ax=ax) is ax


def test_a_panel_count_that_does_not_match_the_axes_is_refused() -> None:
    _figure, grid = plt.subplots(1, 2)

    with pytest.raises(ValueError, match="one Axes per panel"):
        plots.fig8_patient_potentials(axes=list(grid))


def test_a_figure_needs_at_least_one_panel() -> None:
    with pytest.raises(ValueError, match="at least one panel"):
        plots.fig5_symmetric_potentials(alphas=())


def test_fig3_leaves_no_figure_open_when_it_refuses_its_input() -> None:
    with pytest.raises(ValueError, match="followup_weeks"):
        plots.fig3_rr_phase_histogram([100.0, np.nan])

    assert plt.get_fignums() == []


def test_fig4_leaves_no_figure_open_when_it_refuses_its_input() -> None:
    with pytest.raises(ValueError, match="no run of state"):
        plots.fig4_duration_histograms(relapse_only())

    assert plt.get_fignums() == []


def test_fig5_leaves_no_figure_open_when_it_refuses_its_input() -> None:
    with pytest.raises(ValueError, match="alpha"):
        plots.fig5_symmetric_potentials(alphas=(-1.0,))

    assert plt.get_fignums() == []


def test_fig6_leaves_no_figure_open_when_it_refuses_its_input() -> None:
    with pytest.raises(ValueError, match="alpha"):
        plots.fig6_asymmetric_potential(alpha=0.0)

    assert plt.get_fignums() == []


def test_fig7_leaves_no_figure_open_when_it_refuses_its_input() -> None:
    with pytest.raises(ValueError, match="dt"):
        plots.fig7_simulated_paths(t_end=20.0, dt=0.0, rng=3)

    assert plt.get_fignums() == []


def test_fig8_leaves_no_figure_open_when_it_refuses_its_input() -> None:
    with pytest.raises(ValueError, match="critical points"):
        plots.fig8_patient_potentials(betas=(BETA_BEYOND_FOLD,), labels=("past the fold",))

    assert plt.get_fignums() == []


def test_survival_overlays_the_fitted_exponential(runs: pd.DataFrame) -> None:
    ax = plots.fig_survival_vs_exponential(runs, NO_HEALTH)

    empirical = step_line(ax)
    fitted_line = next(line for line in ax.lines if line.get_drawstyle() == "default")
    survival = ydata(empirical)
    assert survival[0] == pytest.approx(1.0)
    assert np.all(np.diff(survival) <= 0.0)
    # The curve is read at the points it was drawn at rather than interpolated
    # between them, so the comparison is with the fitted rate itself and carries
    # no tolerance beyond the arithmetic. The durations are whole weeks, so the
    # fit is read on whole weeks: the survival is (1 - rate) ** t rather than
    # exp(-t / mean), which is the same law rounded a second time.
    fitted = fit.fit_durations(runs, NO_HEALTH)
    assert ydata(fitted_line) == pytest.approx((1.0 - fitted.rate) ** xdata(fitted_line))


def test_survival_curve_stays_closer_to_the_step_than_the_twice_rounded_law(
    runs: pd.DataFrame,
) -> None:
    ax = plots.fig_survival_vs_exponential(runs, NO_HEALTH, inset=False)

    step = step_line(ax)
    curve = next(line for line in ax.lines if line.get_drawstyle() == "default")
    # The step carries information at the whole weeks the events fall on, and
    # holds its level in between, so the two readings are compared there. A post
    # step is read at the last jump at or before the week.
    weeks = np.arange(1.0, 9.0)
    times = xdata(step)
    observed = ydata(step)[np.searchsorted(times, weeks, side="right") - 1]
    drawn = np.interp(weeks, xdata(curve), ydata(curve))
    fitted = fit.fit_durations(runs, NO_HEALTH)
    rounded_twice = np.exp(-weeks / fitted.mean)
    assert np.max(np.abs(drawn - observed)) < np.max(np.abs(rounded_twice - observed))


def test_survival_curve_ends_every_run_in_its_first_week_when_they_all_did() -> None:
    frame = one_week_relapses()
    assert fit.fit_durations(frame, NO_HEALTH).rate == pytest.approx(1.0)

    ax = plots.fig_survival_vs_exponential(frame, NO_HEALTH, inset=False)

    curve = next(line for line in ax.lines if line.get_drawstyle() == "default")
    # A rate of one is the degenerate law, and the survival is one at time zero
    # and zero after it rather than the log(0) the general form would take.
    assert ydata(curve)[0] == pytest.approx(1.0)
    assert ydata(curve)[1:] == pytest.approx(0.0)


def test_survival_step_counts_a_censored_remission_as_at_risk_but_not_as_an_event() -> None:
    ax = plots.fig_survival_vs_exponential(censored_remissions(), HEALTH, inset=False)

    step = step_line(ax)
    # Events at 1, 2 and 3 weeks over 5, 4 and 2 remissions still at risk, the
    # two censored ones at risk until they are cut off but never an event. The
    # step is then carried out to the last time at risk, 5 weeks, at the level
    # it reached, so that the final plateau has a width.
    assert xdata(step) == pytest.approx([0.0, 1.0, 2.0, 3.0, 5.0])
    assert ydata(step) == pytest.approx([1.0, 0.8, 0.6, 0.3, 0.3])


def test_survival_axis_reaches_a_censored_run_that_outlives_every_event() -> None:
    frame = censored_remissions()
    longest = float(frame.loc[frame["state"] == HEALTH, "duration_w"].max())

    ax = plots.fig_survival_vs_exponential(frame, HEALTH, inset=False)

    assert ax.get_xlim()[1] == pytest.approx(longest)


def test_survival_step_is_the_plain_empirical_survival_when_nothing_is_censored(
    runs: pd.DataFrame,
) -> None:
    selected = runs.loc[runs["state"] == NO_HEALTH]
    assert not selected["censored"].any()
    values = selected["duration_w"].to_numpy(dtype=np.float64)

    ax = plots.fig_survival_vs_exponential(runs, NO_HEALTH, inset=False)

    step = step_line(ax)
    times = xdata(step)
    assert times[1:] == pytest.approx(np.unique(values))
    assert ydata(step) == pytest.approx([float(np.mean(values > time)) for time in times])


def test_survival_adds_the_hazard_inset_only_when_asked(runs: pd.DataFrame) -> None:
    with_inset = plots.fig_survival_vs_exponential(runs, NO_HEALTH)
    without_inset = plots.fig_survival_vs_exponential(runs, NO_HEALTH, inset=False)

    assert len(with_inset.child_axes) == 1
    assert without_inset.child_axes == []


def test_the_hazard_inset_reads_the_weeks_the_memoryless_test_reads(runs: pd.DataFrame) -> None:
    ax = plots.fig_survival_vs_exponential(runs, NO_HEALTH)

    inset = ax.child_axes[0]
    regression = fit.test_memoryless(runs, NO_HEALTH, "hazard")
    points = next(line for line in inset.lines if line.get_linestyle() != "--")
    assert ydata(points).size == int(regression.details["n_times"])
    guide = next(line for line in inset.lines if line.get_linestyle() == "--")
    assert float(ydata(guide)[0]) == pytest.approx(1.0 / fit.fit_durations(runs, NO_HEALTH).mean)


def test_the_hazard_inset_points_lie_on_the_memoryless_regression(runs: pd.DataFrame) -> None:
    ax = plots.fig_survival_vs_exponential(runs, NO_HEALTH)

    inset = ax.child_axes[0]
    points = next(line for line in inset.lines if line.get_linestyle() != "--")
    drawn = stats.linregress(xdata(points), ydata(points))
    regression = fit.test_memoryless(runs, NO_HEALTH, "hazard")
    assert float(drawn.slope) == pytest.approx(regression.details["slope"])
    assert float(drawn.intercept) == pytest.approx(regression.details["intercept"])


def test_the_hazard_inset_says_so_when_no_week_holds_enough_at_risk() -> None:
    ax = plots.fig_survival_vs_exponential(four_relapses(), NO_HEALTH)

    inset = ax.child_axes[0]
    assert list(inset.lines) == []
    assert "at risk" in inset.texts[0].get_text()


def test_survival_refuses_a_state_that_is_not_a_clinical_code(runs: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="state"):
        plots.fig_survival_vs_exponential(runs, 0)


def test_poisson_to_nb_draws_two_masses_over_the_counts(runs: pd.DataFrame) -> None:
    ax = plots.fig_poisson_to_nb(relapse_counts(runs))

    assert len(ax.lines) == 2
    assert len(ax.containers) == 1
    assert sum(bar_heights(ax)) == pytest.approx(1.0)


def test_poisson_to_nb_separates_an_overdispersed_cohort() -> None:
    counts = pd.Series(OVERDISPERSED_COUNTS, dtype="int64")

    ax = plots.fig_poisson_to_nb(counts)

    poisson = next(line for line in ax.lines if "Poisson" in str(line.get_label()))
    mixed = next(line for line in ax.lines if "negative binomial" in str(line.get_label()))
    assert not np.allclose(ydata(poisson), ydata(mixed))
    # A cohort whose onset rate varies puts more mass on no relapse at all than
    # a single rate does.
    assert ydata(mixed)[0] > ydata(poisson)[0]


def test_nb_mass_sits_at_the_fitted_parameters() -> None:
    counts = pd.Series(OVERDISPERSED_COUNTS, dtype="int64")
    fitted = fit.fit_nb_counts(counts)
    assert fitted.dispersion > 0.0

    ax = plots.fig_poisson_to_nb(counts)

    mixed = next(line for line in ax.lines if "negative binomial" in str(line.get_label()))
    size = 1.0 / fitted.dispersion
    expected = stats.nbinom.pmf(xdata(mixed), size, size / (size + fitted.mean))
    assert ydata(mixed) == pytest.approx(expected)


def test_poisson_to_nb_caps_the_number_of_count_ticks() -> None:
    counts = pd.Series(OVERDISPERSED_COUNTS, dtype="int64")

    ax = plots.fig_poisson_to_nb(counts)

    assert max(counts) + 1 > plots._MAX_BIN_TICKS
    assert len(ax.get_xticks()) <= plots._MAX_BIN_TICKS


def test_poisson_mass_sits_at_the_sample_mean(runs: pd.DataFrame) -> None:
    counts = relapse_counts(runs)

    ax = plots.fig_poisson_to_nb(counts)

    poisson = next(line for line in ax.lines if "Poisson" in str(line.get_label()))
    expected = stats.poisson.pmf(xdata(poisson), float(counts.mean()))
    assert ydata(poisson) == pytest.approx(expected)


def test_save_all_paper_figures_writes_every_panel(
    tmp_path: Path, weekly: pd.DataFrame, runs: pd.DataFrame
) -> None:
    paths = plots.save_all_paper_figures(tmp_path, weekly, runs, rng=5)

    assert [path.name for path in paths] == EXPECTED_FILE_NAMES
    for path in paths:
        assert path.is_file()
        assert path.stat().st_size > 0
    assert plt.get_fignums() == []


def test_save_all_paper_figures_leaves_no_figure_open_when_it_fails(
    tmp_path: Path, weekly: pd.DataFrame, runs: pd.DataFrame
) -> None:
    # One patient is too few to fit a dispersion, so the count figure raises
    # after the figures before it have been drawn.
    only = str(weekly["patient_id"].iloc[0])
    one_weekly = weekly.loc[weekly["patient_id"] == only].reset_index(drop=True)
    one_runs = runs.loc[runs["patient_id"] == only].reset_index(drop=True)

    with pytest.raises(ValueError, match="at least two counts"):
        plots.save_all_paper_figures(tmp_path, one_weekly, one_runs, rng=5)

    assert plt.get_fignums() == []


def test_save_all_paper_figures_closes_a_figure_a_failed_draw_left_behind(
    tmp_path: Path, weekly: pd.DataFrame, runs: pd.DataFrame, monkeypatch: pytest.MonkeyPatch
) -> None:
    def draw_then_fail(_counts: Any) -> Axes:
        plt.subplots()
        raise RuntimeError("the count figure gave up half way")

    monkeypatch.setattr(plots, "fig_poisson_to_nb", draw_then_fail)

    with pytest.raises(RuntimeError, match="gave up half way"):
        plots.save_all_paper_figures(tmp_path, weekly, runs, rng=5)

    assert plt.get_fignums() == []


def test_drawing_without_matplotlib_names_the_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "matplotlib", None)

    with pytest.raises(ImportError, match=r"msrelapse\[plot\]"):
        plots.fig5_symmetric_potentials()


def test_require_matplotlib_hands_back_pyplot() -> None:
    assert plots.require_matplotlib() is plt


def test_require_matplotlib_names_the_extra_when_matplotlib_is_absent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "matplotlib", None)

    with pytest.raises(ImportError, match=r"msrelapse\[plot\]"):
        plots.require_matplotlib()


def test_require_matplotlib_is_part_of_the_public_surface() -> None:
    assert "require_matplotlib" in plots.__all__
