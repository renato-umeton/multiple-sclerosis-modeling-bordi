"""Every number reported by Bordi et al. 2013, with its unit and its provenance.

This module is the single place in the package where a number taken from the
paper may be written down. No other module may hard-code a paper number:
import ``PAPER`` from here instead, so that every value keeps the citation
that justifies it and a reader can check the package against the article.

References
----------
C. Bordi, A. Speranza, and F. Bordi, "A mechanistic stochastic model helps
understand multiple sclerosis," Computational and Mathematical Methods in
Medicine, 2013, doi 10.1155/2013/910321.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, fields
from typing import Final, Generic, TypeVar

__all__ = [
    "PAPER",
    "Bordi2013",
    "Param",
    "symmetric_barrier",
    "symmetric_well_position",
]

T = TypeVar("T")


@dataclass(frozen=True)
class Param(Generic[T]):
    """One reported quantity together with its unit and its provenance.

    Attributes
    ----------
    value : T
        The quantity as reported by the paper, or as computed from reported
        quantities when the entry is derived.
    unit : str
        The unit of the value, or a short note when the value is dimensionless.
    source : str
        Where the value comes from. Reported quantities start with
        'Bordi 2013', derived ones with 'Derived from Bordi 2013'.
    """

    value: T
    unit: str
    source: str


@dataclass(frozen=True)
class Bordi2013:
    """The complete set of numbers the 2013 paper reports, one field per number.

    Each field holds a :class:`Param`, so a caller reading a value can also
    read the unit it is expressed in and the place in the paper it comes from.
    Use the module level instance ``PAPER`` rather than building your own.
    """

    alpha_reference: Param[float] = Param(
        value=1.0,
        unit="dimensionless",
        source=(
            "Bordi 2013, Section 3.2, page 4, 'the reference value alpha = 1'; used in "
            "Figures 5(a), 6(a), 7(a), 7(b) and all three panels of Figure 8"
        ),
    )
    alpha_illustrative_low: Param[float] = Param(
        value=0.7,
        unit="dimensionless",
        source=(
            "Bordi 2013, Section 3.2, page 4, 'two values of the control parameter: "
            "alpha = 1 and alpha = 0.7'; used only in Figures 5(b) and 6(b), never in a "
            "simulation or a patient fit"
        ),
    )
    beta_symmetric: Param[float] = Param(
        value=0.0,
        unit="dimensionless",
        source=(
            "Bordi 2013, Figure 5 (implicit, equation (2)) and Figure 7(a) caption, page 8, "
            "'the symmetric (beta = 0)'"
        ),
    )
    beta_illustrative: Param[float] = Param(
        value=0.08,
        unit="dimensionless",
        source=(
            "Bordi 2013, Section 3.2, page 6, 'Figure 6 illustrates an example of asymmetric "
            "double-well (beta = 0.08) for alpha = 1 and alpha = 0.7'; also Figure 7(b), page 8"
        ),
    )
    beta_patient_23: Param[float] = Param(
        value=0.25,
        unit="dimensionless",
        source=(
            "Bordi 2013, Figure 8 caption and Figure 8(a) in-panel annotation, page 8, "
            "'alpha = 1, beta = 0.25'"
        ),
    )
    beta_patient_32: Param[float] = Param(
        value=0.19,
        unit="dimensionless",
        source=(
            "Bordi 2013, Figure 8 caption and Figure 8(b) in-panel annotation, page 8, "
            "'alpha = 1, beta = 0.19'"
        ),
    )
    beta_patient_53: Param[float] = Param(
        value=0.12,
        unit="dimensionless",
        source=(
            "Bordi 2013, Figure 8 caption and Figure 8(c) in-panel annotation, page 8, "
            "'alpha = 1, beta = 0.12'"
        ),
    )
    epsilon_noise_variance: Param[float] = Param(
        value=0.13,
        unit="dimensionless, variance of the stochastic perturbation",
        source=(
            "Bordi 2013, Figure 7 caption and both in-panel annotations, page 8, 'a stochastic "
            "perturbation of the same variance is applied (epsilon = 0.13)'; the only numerical "
            "noise value anywhere in the paper"
        ),
    )
    noise_amplitude: Param[float] = Param(
        value=math.sqrt(0.13),
        unit="dimensionless, multiplies dw in equation (4)",
        source=(
            "Bordi 2013, equation (4), page 6, where the noise term is printed as "
            "epsilon^(1/2) dw; stored as the square root of the reported variance "
            "epsilon = 0.13"
        ),
    )
    n_patients: Param[int] = Param(
        value=70,
        unit="patients",
        source="Bordi 2013, Abstract page 1 and Section 2.1, page 2",
    )
    n_males: Param[int] = Param(
        value=28,
        unit="patients",
        source=(
            "Bordi 2013, Section 2.1, page 2, '70 patients (28 males and 42 females) with "
            "definite MS'"
        ),
    )
    n_females: Param[int] = Param(
        value=42,
        unit="patients",
        source="Bordi 2013, Section 2.1, page 2",
    )
    n_neurologists: Param[int] = Param(
        value=4,
        unit="clinicians",
        source="Bordi 2013, Section 2.1, page 2, 'by four experienced MS neurologists'",
    )
    data_cutoff_year: Param[int] = Param(
        value=1993,
        unit="calendar year",
        source="Bordi 2013, Section 2.1, page 2, 'all data are antecedent 1993'",
    )
    italy_ethics_committee_year: Param[int] = Param(
        value=1998,
        unit="calendar year",
        source=(
            "Bordi 2013, Section 2.1, page 2, 'before the institution of ethics committee in "
            "Italy in 1998'"
        ),
    )
    n_no_health_events: Param[int] = Param(
        value=218,
        unit="events across the whole cohort",
        source=(
            "Bordi 2013, Figure 4(a) in-panel annotation, page 5, 'No health events: total = 218'"
        ),
    )
    n_health_events: Param[int] = Param(
        value=266,
        unit="events across the whole cohort",
        source=(
            "Bordi 2013, Figure 4(b) in-panel annotation, page 5, 'Health events: total = 266'"
        ),
    )
    tau_health_cohort_weeks: Param[float] = Param(
        value=100.0,
        unit="weeks",
        source=(
            "Bordi 2013, Section 3.4, page 6, 'we have tau_x1 approximately 100 weeks'; also "
            "Figure 4(b) annotation 'Duration of health events: mean = 100 weeks'; also "
            "Section 3.1, page 3"
        ),
    )
    tau_no_health_cohort_weeks: Param[float] = Param(
        value=4.3,
        unit="weeks",
        source=(
            "Bordi 2013, Section 3.4, page 6, 'tau_x2 approximately 4.3 weeks'; also "
            "Figure 4(a) annotation and Section 3.1, page 3"
        ),
    )
    barrier_ratio_cohort: Param[float] = Param(
        value=3.1,
        unit="dimensionless",
        source=(
            "Bordi 2013, Section 3.4, page 6, 'it is found deltaV1/deltaV2 approximately 3.1. "
            "This implies that the well in x1 is about three times deeper than that in x2.'"
        ),
    )
    relapse_duration_min_weeks: Param[float] = Param(
        value=1.0,
        unit="weeks",
        source=(
            "Bordi 2013, Section 3.1, page 3, 'with a minimum of 1 week'; an artefact of the "
            "rounding rule in Section 2.1"
        ),
    )
    relapse_duration_max_weeks: Param[float] = Param(
        value=24.0,
        unit="weeks",
        source=(
            "Bordi 2013, Section 3.1, page 3, 'to a maximum of about 24 weeks (rare events)'; "
            "the paper gives this bound as approximate"
        ),
    )
    remission_duration_max_weeks: Param[float] = Param(
        value=1000.0,
        unit="weeks",
        source=(
            "Bordi 2013, Section 3.1, page 3, 'up to about 1000 weeks in exceptional cases'; "
            "the paper gives this bound as approximate"
        ),
    )
    rr_phase_min_weeks: Param[float] = Param(
        value=40.0,
        unit="weeks",
        source=(
            "Bordi 2013, Section 3.1, page 3, and Figure 3 caption, page 4, 'from a minimum of "
            "40 weeks'"
        ),
    )
    rr_phase_max_weeks: Param[float] = Param(
        value=1311.0,
        unit="weeks, about 27 years",
        source=(
            "Bordi 2013, Section 3.1, page 3, and Figure 3 caption, page 4, 'to a maximum of "
            "1311 weeks (about 27 years)'"
        ),
    )
    rr_phase_mode_weeks: Param[float] = Param(
        value=200.0,
        unit="weeks",
        source=(
            "Bordi 2013, Section 3.1, page 3, and Figure 3 caption, 'for 25 of 70 patients "
            "(about 36%) the relapsing-remitting phase lasts for about 200 weeks (about 4 "
            "years)'; the paper gives this value as approximate"
        ),
    )
    rr_phase_mode_n_patients: Param[int] = Param(
        value=25,
        unit="patients, about 36 percent of 70",
        source="Bordi 2013, Section 3.1, page 3, and Figure 3 caption, page 4",
    )
    relapse_min_symptom_hours: Param[float] = Param(
        value=24.0,
        unit="hours",
        source=(
            "Bordi 2013, Section 2.1, page 2, citing reference [14] Schumacher et al. 1965, "
            "'of at least 24 hours duration'"
        ),
    )
    relapse_max_symptom_months: Param[float] = Param(
        value=6.0,
        unit="months",
        source="Bordi 2013, Section 2.1, page 2, 'but less than 6 month' [sic]",
    )
    time_resolution_weeks: Param[float] = Param(
        value=1.0,
        unit="weeks",
        source=(
            "Bordi 2013, Section 2.1, page 2, 'Events are reported on a weekly scale (shorter "
            "exacerbations have been rounded up to one week).'"
        ),
    )
    state_no_health: Param[int] = Param(
        value=1,
        unit="dimensionless clinical code, relapse",
        source=(
            "Bordi 2013, Section 2.1, page 2, 'a sequence of plus one (+1) and minus one (-1) "
            "corresponding to the states of relapses and remissions, respectively'"
        ),
    )
    state_health: Param[int] = Param(
        value=-1,
        unit="dimensionless clinical code, remission",
        source=(
            "Bordi 2013, Section 2.1, page 2; model counterpart fixed on page 4, 'let x1 be the "
            "state of health and x2 the state of no health'"
        ),
    )
    tau_health_p23_weeks: Param[float] = Param(
        value=117.7,
        unit="weeks",
        source="Bordi 2013, Section 3.4 displayed list, page 6",
    )
    tau_no_health_p23_weeks: Param[float] = Param(
        value=1.5,
        unit="weeks",
        source="Bordi 2013, Section 3.4 displayed list, page 6",
    )
    barrier_ratio_p23: Param[float] = Param(
        value=11.8,
        unit="dimensionless",
        source="Bordi 2013, Section 3.4 displayed list, page 6",
    )
    tau_health_p32_weeks: Param[float] = Param(
        value=54.0,
        unit="weeks",
        source="Bordi 2013, Section 3.4 displayed list, page 6",
    )
    tau_no_health_p32_weeks: Param[float] = Param(
        value=2.1,
        unit="weeks",
        source="Bordi 2013, Section 3.4 displayed list, page 6",
    )
    barrier_ratio_p32: Param[float] = Param(
        value=5.3,
        unit="dimensionless",
        source="Bordi 2013, Section 3.4 displayed list, page 6",
    )
    tau_health_p53_weeks: Param[float] = Param(
        value=47.0,
        unit="weeks",
        source="Bordi 2013, Section 3.4 displayed list, page 6",
    )
    tau_no_health_p53_weeks: Param[float] = Param(
        value=4.3,
        unit="weeks",
        source="Bordi 2013, Section 3.4 displayed list, page 6",
    )
    barrier_ratio_p53: Param[float] = Param(
        value=2.7,
        unit="dimensionless",
        source="Bordi 2013, Section 3.4 displayed list, page 6",
    )
    fig7_time_span: Param[tuple[float, float]] = Param(
        value=(0.0, 1000.0),
        unit="dimensionless model time; no unit is stated and no conversion to weeks is given",
        source=(
            "Bordi 2013, Figure 7 both panels, page 8, x-axis labelled only 'Time', ticked "
            "every 100"
        ),
    )
    fig7_x_limits: Param[tuple[float, float]] = Param(
        value=(-2.0, 2.0),
        unit="dimensionless state variable",
        source="Bordi 2013, Figure 7 both panels, page 8, y-axis ticked -2, -1, 0, 1, 2",
    )
    potential_plot_x_limits: Param[tuple[float, float]] = Param(
        value=(-2.0, 2.0),
        unit="dimensionless state variable",
        source="Bordi 2013, Figures 5, 6 and 8, pages 5, 7 and 8, x-axis ticked -2, -1, 0, 1, 2",
    )
    potential_plot_v_limits: Param[tuple[float, float]] = Param(
        value=(-0.8, 0.8),
        unit="dimensionless potential",
        source="Bordi 2013, Figures 5, 6 and 8, pages 5, 7 and 8, V-axis ticked every 0.2",
    )
    fig3_bin_edges_weeks: Param[tuple[float, ...]] = Param(
        value=(40.0, 199.0, 358.0, 517.0, 676.0, 835.0, 994.0, 1153.0, 1312.0),
        unit="weeks, 8 bins of width 159",
        source="Bordi 2013, Figure 3 x-axis tick labels, page 4",
    )
    fig3_counts: Param[tuple[int, ...]] = Param(
        value=(25, 17, 9, 5, 7, 2, 2, 3),
        unit="patients, sums to exactly 70",
        source=(
            "Bordi 2013, Figure 3 bars, page 4; not printed as numbers, measured from a 400 dpi "
            "render, sum verified equal to the stated cohort size"
        ),
    )
    fig4a_bin_width_weeks: Param[float] = Param(
        value=2.5,
        unit="weeks, first bin starting at 1",
        source=(
            "Bordi 2013, Figure 4(a) x-axis ticks 1, 3.5, 6, 8.5, 11, 13.5, 16, 18.5, 21, 23.5, "
            "25, page 5"
        ),
    )
    fig4a_counts: Param[tuple[int, ...]] = Param(
        value=(127, 53, 16, 6, 1, 6, 0, 4, 5),
        unit="no health events, sums to exactly 218",
        source=(
            "Bordi 2013, Figure 4(a) bars, page 5; not printed as numbers, measured from a "
            "400 dpi render, sum verified equal to the printed total 218"
        ),
    )
    fig4b_bin_width_weeks: Param[float] = Param(
        value=100.0,
        unit="weeks, first bin starting at 0",
        source="Bordi 2013, Figure 4(b) x-axis 0 to 1200 ticked every 200, page 5",
    )
    fig4b_counts: Param[tuple[int, ...]] = Param(
        value=(193, 35, 18, 5, 0, 5, 4, 2, 0, 2, 2),
        unit="health events, sums to the printed total 266",
        source=(
            "Bordi 2013, Figure 4(b) bars, page 5; not printed as numbers, measured from a "
            "400 dpi render. The first bin measured 192.2 and was rounded up to 193 so that "
            "the total matches the printed 266"
        ),
    )
    glaciation_forcing_period_years: Param[float] = Param(
        value=100000.0,
        unit="years",
        source=(
            "Bordi 2013, Discussion, page 9, 'such a period is set to 100000 years'; context "
            "only, not a model parameter of this paper"
        ),
    )
    paper_doi: Param[str] = Param(
        value="10.1155/2013/910321",
        unit="DOI",
        source="Bordi 2013, title page header, page 1",
    )
    derived_symmetric_barrier: Param[float] = Param(
        value=0.25,
        unit="dimensionless, barrier height at the reference alpha = 1",
        source=(
            "Derived from Bordi 2013, equation (2), page 4; the barrier 1 / (4 alpha) is not "
            "printed in the paper. Verified against the plotted dashed lines of Figure 5(a) "
            "and 5(b). Call symmetric_barrier for other values of alpha"
        ),
    )
    derived_symmetric_well_position: Param[float] = Param(
        value=1.0,
        unit="dimensionless, positive well position at the reference alpha = 1",
        source=(
            "Derived from Bordi 2013, page 4, where the well positions are printed as plus or "
            "minus 1 / sqrt(alpha) (with a Latin 'a' under the radical); verified against "
            "Figure 5. Call symmetric_well_position for other values of alpha"
        ),
    )
    derived_beta_fold_alpha1: Param[float] = Param(
        value=2.0 / (3.0 * math.sqrt(3.0)),
        unit="dimensionless",
        source=(
            "Derived from Bordi 2013, equation (2), page 4; the value itself is not printed in "
            "the paper. At alpha = 1 the cubic x^3 - x + beta has three real roots only for "
            "beta below this value, above which the no health well vanishes in a saddle-node "
            "fold. Guard package input against it"
        ),
    )

    def as_records(self) -> list[dict[str, object]]:
        """Return one record per reported number, for documentation tables.

        Returns
        -------
        list of dict
            One dictionary per field, in declaration order, with the keys
            'name', 'value', 'unit' and 'source'.
        """
        records: list[dict[str, object]] = []
        for entry in fields(self):
            param: Param[object] = getattr(self, entry.name)
            records.append(
                {
                    "name": entry.name,
                    "value": param.value,
                    "unit": param.unit,
                    "source": param.source,
                }
            )
        return records


PAPER: Final = Bordi2013()


def symmetric_barrier(alpha: float) -> float:
    """Return the barrier height of the symmetric double well.

    The symmetric potential of equation (2) with beta = 0 has two wells of
    equal depth separated by a barrier of height 1 / (4 alpha).

    Parameters
    ----------
    alpha : float
        Control parameter of the potential. Must be positive.

    Returns
    -------
    float
        Barrier height, dimensionless.

    Raises
    ------
    ValueError
        If `alpha` is zero or negative, where the double well does not exist.

    Examples
    --------
    >>> symmetric_barrier(1.0)
    0.25
    """
    if alpha <= 0.0:
        raise ValueError(f"alpha must be positive, got {alpha!r}")
    return 1.0 / (4.0 * alpha)


def symmetric_well_position(alpha: float) -> float:
    """Return the positive well position of the symmetric double well.

    The symmetric potential of equation (2) with beta = 0 has its minima at
    plus and minus 1 / sqrt(alpha); this function returns the positive one.

    Parameters
    ----------
    alpha : float
        Control parameter of the potential. Must be positive.

    Returns
    -------
    float
        Position of the positive minimum, dimensionless.

    Raises
    ------
    ValueError
        If `alpha` is zero or negative, where the double well does not exist.

    Examples
    --------
    >>> symmetric_well_position(1.0)
    1.0
    """
    if alpha <= 0.0:
        raise ValueError(f"alpha must be positive, got {alpha!r}")
    return 1.0 / math.sqrt(alpha)
