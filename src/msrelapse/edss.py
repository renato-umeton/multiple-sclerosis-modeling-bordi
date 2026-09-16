"""An illustrative EDSS trajectory driven by the weekly relapse series.

This module is an extension and is not part of the 2013 article. That article
models one real number crossing a barrier and the weekly sequence of plus one
and minus one the crossings produce, and it reports no EDSS, no disability
score and no clinical outcome for any of its 70 patients. Every number here
comes from other papers, collected in ``EVIDENCE`` with the source and the DOI
of each one, and every distribution here was fitted to those papers rather than
taken from one. A trace is one draw, never a median and never a prognosis.

The model reads the weekly series as a list of relapse episodes, the maximal
runs of the no health state, and gives each episode one kernel. The kernel
rises to a peak deficit within the time to nadir, one week on this grid, and
then decays exponentially, with a recovery constant of about ten weeks, towards
the residual that episode leaves behind for good. The trajectory is the
baseline, plus an optional linear progression term, plus the sum of the
kernels, clipped to the scale and displayed on the half point grid. Long after
a relapse its kernel has decayed to its residual, so the trace is the baseline
plus the residuals of all past relapses plus the transient bump of the current
one.

The evidence behind the defaults. The baseline of 2.0 is the median first
recorded EDSS at a first demyelinating event in 1074 patients (Tur 2025). The
peak law reproduces three published marginals of the nadir increase, 84 percent
of relapses at or above 0.5, 61 percent at or above 1.0 (both from the AFFIRM
placebo arm, Lublin 2014) and 17 percent at or above 2.0 (Achiron 2019), and
its mean of 1.05 sits just below the AFFIRM placebo mean of 1.09. The residual
law reproduces the pooled incomplete recovery rate of 0.42 over 27,672 relapses
(Ladeira 2025) together with the mean of about 0.25 and the median of 0 of the
pooled placebo arms (Lublin 2003), which is what the mass below zero is for,
and it is drawn from one of two laws by the severity of the peak, the strongest
and most consistent predictor of incomplete recovery. The recovery constant of
ten weeks is fitted to published median recovery times of 71 and 111 days (Koch
2023, Mostert 2025). Progression independent of relapse activity is off by
default, so the trace is purely relapse driven.

The model is valid over the relapsing phase alone, roughly the first 15 years
from onset, because the mechanism that carries patients further, the secondary
progressive transition, is not in the weekly series at all. It cannot reproduce
the accelerating shape of the real curve, and the published spread at any fixed
duration is enormous. Nothing here is fit for individual prognosis, for trial
design, or for any clinical use.

Numbers reported by the 2013 article, the two state codes among them, are read
from ``PAPER`` in [`msrelapse._params`][msrelapse._params] and are never
written here. Weeks become years at the 365.25 over 7 weeks of
``msrelapse.stats.WEEKS_PER_YEAR``, which the whole package converts with. The
progression term is written with that year rather than with the flat 52 week
year the synthesis behind this module writes it with, so that a year means one
thing across the package; the two differ by 0.3 percent, which is a rate of 0.1
raising the trace by 0.9966 points over 520 weeks rather than by a point, and
ten years falling at week 522 rather than at week 520.

Examples
--------
>>> import numpy as np
>>> from msrelapse.edss import edss_trajectory, episode_draws
>>> states = np.array([1] * 4 + [-1] * 60)
>>> trace = edss_trajectory(states, rng=0)
>>> list(trace.columns)
['week', 'edss', 'edss_display', 'episode']
>>> draws = episode_draws(states, rng=0)
>>> list(draws["start"]), list(draws["duration_w"])
([0], [4])

References
----------
Tur C, et al. Lancet Regional Health Europe, 2025, doi
10.1016/j.lanepe.2025.101302. Lublin FD, et al. Multiple Sclerosis and Related
Disorders, 2014, doi 10.1016/j.msard.2014.08.005. Achiron A, et al. Multiple
Sclerosis Journal, 2019, doi 10.1177/1352458518809903. Ladeira F, et al.
Multiple Sclerosis and Related Disorders, 2025, doi
10.1016/j.msard.2025.106507. Lublin FD, et al. Neurology, 2003, doi
10.1212/01.wnl.0000096175.39831.21. Koch MW, et al. Multiple Sclerosis Journal,
2023, doi 10.1177/13524585231202320. Pittock SJ, et al. Neurology, 2004, doi
10.1212/01.wnl.0000101724.93433.00.

I. Bordi, R. Umeton, V. A. G. Ricigliano, et al., "A mechanistic, stochastic
model helps understand multiple sclerosis course and pathogenesis",
International Journal of Genomics, 2013, doi 10.1155/2013/910321.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Final

import numpy as np
import numpy.typing as npt
import pandas as pd

from msrelapse._params import PAPER
from msrelapse.io import validate
from msrelapse.simulate import Seed
from msrelapse.stats import WEEKS_PER_YEAR

__all__ = [
    "DEFAULT_SPEC",
    "EVIDENCE",
    "PEAK_LAW",
    "RESIDUAL_LAW_MILD",
    "RESIDUAL_LAW_SEVERE",
    "SEVERE_PEAK",
    "EDSSSpec",
    "Evidence",
    "edss_trajectory",
    "episode_draws",
    "expected_peak",
    "expected_residual",
    "relapse_episodes",
    "summarise",
]

_Vector = npt.NDArray[np.float64]
_States = npt.NDArray[np.int64]

_NO_HEALTH: Final = PAPER.state_no_health.value
_HEALTH: Final = PAPER.state_health.value

_HALF_POINT: Final = 0.5
"""float: The step of the EDSS scale, which every law here sits on."""

SEVERE_PEAK: Final = 1.0
"""float: The nadir increase at which a relapse leaves the mild class.

A peak below this draws its residual from ``RESIDUAL_LAW_MILD`` and a peak at
or above it from ``RESIDUAL_LAW_SEVERE``. Two classes rather than three is a
judgement call of the synthesis behind this module; the split carries an odds
ratio of 3.4 for incomplete recovery, inside the published range of 2.4 to
17.2.
"""

_LAW_TOLERANCE: Final = 1e-9
"""How far from one the mass of a law may fall and still count as a law."""

_TEN_YEARS_WEEKS: Final = round(10.0 * WEEKS_PER_YEAR)
"""Ten years from onset, in whole weeks, as ``summarise`` reads them."""

PEAK_LAW: Final[Mapping[float, float]] = MappingProxyType(
    {0.0: 0.16, 0.5: 0.23, 1.0: 0.26, 1.5: 0.18, 2.0: 0.09, 2.5: 0.04, 3.0: 0.025, 3.5: 0.015}
)
"""Mapping: The nadir increase of one relapse, on the half point grid.

Built to reproduce three verified marginals exactly, 84 percent of relapses at
or above 0.5 and 61 percent at or above 1.0 from the AFFIRM placebo arm and 17
percent at or above 2.0 from 1672 graded relapses. Its mean is 1.0525, which
the synthesis quotes as 1.05 and which sits 0.04 below the AFFIRM placebo mean
of 1.09. The shape between the pinned points is a judgement call.
"""

RESIDUAL_LAW_MILD: Final[Mapping[float, float]] = MappingProxyType(
    {-1.0: 0.03, -0.5: 0.15, 0.0: 0.57, 0.5: 0.21, 1.0: 0.04}
)
"""Mapping: What a relapse whose peak is below ``SEVERE_PEAK`` leaves behind.

Such relapses are 39 percent of the peak law. A quarter of them leave a
residual of at least half a point, and their mean residual is 0.04.
"""

RESIDUAL_LAW_SEVERE: Final[Mapping[float, float]] = MappingProxyType(
    {
        -1.0: 0.03,
        -0.5: 0.15,
        0.0: 0.29,
        0.5: 0.265,
        1.0: 0.15,
        1.5: 0.06,
        2.0: 0.032,
        2.5: 0.013,
        3.0: 0.010,
    }
)
"""Mapping: What a relapse whose peak reaches ``SEVERE_PEAK`` leaves behind.

Such relapses are 61 percent of the peak law. Marginally over the two classes
the residual is at least 0.5 in 42 percent of relapses and at least 1.0 in 18
percent, with a median of 0 and a mean of 0.256, which are the published
figures the pair of laws was built to satisfy at once. The mass below zero, 18
percent of relapses ending at least half a point below the pre-relapse score,
is what lets one distribution meet both the thresholds and the mean; its size
is inferred from that arithmetic rather than measured anywhere.
"""


@dataclass(frozen=True)
class Evidence:
    """One published quantity this illustrative model rests on.

    Attributes
    ----------
    parameter : str
        What the quantity is.
    value : str
        The central value, as the literature synthesis behind this module
        states it. Written as text because several of these are a set of
        figures rather than one number.
    unit : str
        The unit the value is expressed in.
    source : str
        The study or studies the value comes from, and what they measured.
    doi : str
        The DOI of the one study the value is credited to. A record whose value
        is derived from several studies names the others in `source` and
        carries the DOI of the one its central value rests on most.
    """

    parameter: str
    value: str
    unit: str
    source: str
    doi: str


EVIDENCE: Final[tuple[Evidence, ...]] = (
    Evidence(
        parameter="Baseline EDSS after the first attack",
        value="2.0",
        unit="EDSS points",
        source=(
            "Tur 2025, median first recorded EDSS at a first demyelinating event in a "
            "prospective inception cohort of 1074 patients, interquartile range 1.0 to 2.5. "
            "The first EDSS precedes any treatment, so the value transports to an untreated "
            "model"
        ),
        doi="10.1016/j.lanepe.2025.101302",
    ),
    Evidence(
        parameter="Mean EDSS increase at the relapse nadir, untreated",
        value="1.10",
        unit="EDSS points",
        source=(
            "Lublin 2014, AFFIRM placebo arm, mean increase from pre-relapse to at-relapse "
            "EDSS of 1.09 in 176 of 315 placebo patients who relapsed, against 0.77 on "
            "natalizumab. Hirst 2008 reports 1.45 from a pre-relapse EDSS of 3.73, far above "
            "a typical early cohort"
        ),
        doi="10.1016/j.msard.2014.08.005",
    ),
    Evidence(
        parameter="Proportion of relapses with a nadir increase of at least 0.5, 1.0 and 2.0",
        value="84 percent, 61 percent, 17 percent",
        unit="proportion of relapses",
        source=(
            "Lublin 2014, AFFIRM placebo arm, for the 0.5 and 1.0 thresholds; Achiron 2019, "
            "1672 graded relapses, for the 2.0 threshold. The published severe share runs "
            "from 14 to 39 percent because the grading schemes differ, not the cohorts"
        ),
        doi="10.1016/j.msard.2014.08.005",
    ),
    Evidence(
        parameter="Peak deficit law on the half point grid",
        value="0.16, 0.23, 0.26, 0.18, 0.09, 0.04, 0.025, 0.015 at 0.0 to 3.5",
        unit="probability per half point",
        source=(
            "Derived here from Lublin 2014 and Achiron 2019 to reproduce the three verified "
            "marginals exactly, with a monotone decreasing shape between them. Mean 1.05, "
            "median 1.0"
        ),
        doi="10.1177/1352458518809903",
    ),
    Evidence(
        parameter="Time from relapse onset to nadir",
        value="1 to 2 days",
        unit="days",
        source=(
            "Hosny 2023, median 1 day for mild and moderate relapses and 2 days for severe "
            "ones across 300 attacks in 223 patients, dated from a patient symptom diary. On "
            "a weekly grid this is the first week"
        ),
        doi="10.1186/s12883-023-03109-6",
    ),
    Evidence(
        parameter="Median time to EDSS recovery after a relapse",
        value="71 to 111 days",
        unit="days",
        source=(
            "Koch 2023, 240 first on-trial relapses in CombiRx, median 111 days; Mostert "
            "2025, 430 first relapses in DECIDE, median 71 days. Recovery is the first "
            "post-relapse EDSS at or below the pre-relapse EDSS. Both cohorts are fully "
            "treated and no untreated equivalent exists"
        ),
        doi="10.1177/13524585231202320",
    ),
    Evidence(
        parameter="Recovery decay constant of the model",
        value="10 weeks",
        unit="weeks",
        source=(
            "Judgement call fitted to the two published medians: an excess of one point falls "
            "below the quarter point display threshold at tau times the logarithm of four, "
            "which is 71 days at 7.3 weeks and 111 days at 11.4 weeks. Derived here from "
            "Mostert 2025 and Koch 2023"
        ),
        doi="10.1136/jnnp-2025-336660",
    ),
    Evidence(
        parameter="Probability a relapse leaves a residual of at least 0.5 EDSS",
        value="0.42",
        unit="proportion of relapses",
        source=(
            "Ladeira 2025, random effects pooled rate of incomplete recovery across 13 "
            "studies, 19,920 patients and 27,672 relapses followed at least 6 months, 95 "
            "percent interval 0.31 to 0.54. Two untreated anchors agree, 42 percent in the "
            "pooled placebo arms of pre-DMT trials and 45 percent in the AFFIRM placebo arm"
        ),
        doi="10.1016/j.msard.2025.106507",
    ),
    Evidence(
        parameter="Probability a relapse leaves a residual of at least 1.0 EDSS",
        value="0.18",
        unit="proportion of relapses",
        source=(
            "Lublin 2003, 28 percent at a mean of 64 days in the pooled placebo arms, shrunk "
            "here for the further recovery Achiron 2019 documents between 1 and 12 months"
        ),
        doi="10.1212/01.wnl.0000096175.39831.21",
    ),
    Evidence(
        parameter="Mean net residual EDSS per relapse",
        value="0.25",
        unit="EDSS points",
        source=(
            "Lublin 2003, mean 0.27 with a median of 0 at a mean of 64 days in the pooled "
            "placebo arms, and 0.28 in the AFFIRM placebo arm. Both are net of relapses that "
            "ended below the pre-relapse score, which is why the residual law carries mass "
            "below zero"
        ),
        doi="10.1212/01.wnl.0000096175.39831.21",
    ),
    Evidence(
        parameter="Residual step law on the half point grid, by severity class",
        value="mean 0.256, median 0, at least 0.5 in 0.42 and at least 1.0 in 0.18",
        unit="probability per half point",
        source=(
            "Derived here from Ladeira 2025, Lublin 2003 and Achiron 2019 to reproduce the "
            "pooled incomplete recovery rate, the published median and mean, and the severe "
            "tail of about 3.5 percent of relapses retaining at least 2.0 points"
        ),
        doi="10.1016/j.msard.2025.106507",
    ),
    Evidence(
        parameter="Dependence of the residual on relapse severity",
        value="odds ratio about 3.4 for incomplete recovery, severe against mild",
        unit="odds ratio",
        source=(
            "Ladeira 2025 and Koch 2023: severity is the strongest and most consistent "
            "predictor of incomplete recovery, and the only significant one in CombiRx. "
            "Published estimates run from 2.4 to 17.2, the upper figure from a small "
            "reference stratum that should never be used as a point estimate"
        ),
        doi="10.1016/j.msard.2025.106507",
    ),
    Evidence(
        parameter="EDSS at 10 years from onset",
        value="about 3",
        unit="EDSS points",
        source=(
            "Pittock 2004, mean EDSS change of 1 point over 10 years in the Olmsted County "
            "population based cohort, 161 of 162 patients followed up and only 15 percent "
            "ever on immunomodulatory therapy. From a baseline of 2.0 this puts the score "
            "near 3 at 10 years"
        ),
        doi="10.1212/01.wnl.0000101724.93433.00",
    ),
    Evidence(
        parameter="EDSS at 20 years from onset",
        value="median about 3.25 at 14 years, spread from 0 to 10",
        unit="EDSS points",
        source=(
            "Brex 2002, London clinically isolated syndrome cohort recruited 1984 to 1987 and "
            "reassessed at a mean of 14.1 and again at 20.2 years. At 20 years 58 percent "
            "were still relapsing remitting. The spread at one time point is the honest "
            "caveat on any single illustrative trajectory"
        ),
        doi="10.1056/NEJMoa011341",
    ),
    Evidence(
        parameter="Relapse associated share of confirmed disability worsening",
        value="27 percent",
        unit="proportion of worsening events",
        source=(
            "Lublin 2022, 474 relapse associated against 833 independent of relapse activity "
            "of 1761 confirmed events in the largest trial pool. Every published share comes "
            "from a population whose relapses were being suppressed, and no natural history "
            "cohort has ever computed the split with modern definitions"
        ),
        doi="10.1093/brain/awac016",
    ),
    Evidence(
        parameter="Independent progression rate, the PIRA term of the model",
        value="0.0",
        unit="EDSS points per year",
        source=(
            "Judgement call, derived here from Pittock 2004. At zero the model reproduces the "
            "observed mean slope of about one point per decade in a near untreated "
            "population; a positive rate instead matches the milestone medians and overshoots "
            "that slope. Zero keeps the trace purely relapse driven and avoids counting the "
            "same accrual twice"
        ),
        doi="10.1212/01.wnl.0000101724.93433.00",
    ),
)
"""tuple of Evidence: The published quantities the model and its defaults rest on.

Three of the sixteen records are parameters of the model rather than quantities
a study measured: the peak law, the residual step law and the recovery decay
constant. They are kept here beside the measured ones because the model uses
each of them directly and a reader is owed the same account of where each one
came from. The `source` of each of the three says that it is derived here and
names the measured records it was fitted to, so no study is credited with a
number it never reported.
"""


def _check_positive(name: str, value: float, unit: str) -> None:
    """Check that a parameter is a finite number above zero.

    Parameters
    ----------
    name : str
        The name of the parameter, for the message.
    value : float
        The value to check.
    unit : str
        The unit of the parameter, for the message.

    Returns
    -------
    None
        Nothing is returned; a value out of range raises instead.

    Raises
    ------
    ValueError
        If `value` is not a finite number above zero.
    """
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be a positive number of {unit}, got {value!r}")


def _check_law(name: str, law: Mapping[float, float]) -> None:
    """Check that a mapping is a law on the half point grid of the scale.

    Parameters
    ----------
    name : str
        The name of the law, for the message.
    law : collections.abc.Mapping
        The mapping to check.

    Returns
    -------
    None
        Nothing is returned; a mapping that is not a law raises instead.

    Raises
    ------
    ValueError
        If the mapping is empty, holds a value off the half point grid or a
        negative or non finite probability, or does not sum to one.
    """
    if not law:
        raise ValueError(f"{name} must hold at least one value, got an empty law")
    for value, probability in law.items():
        if not math.isfinite(value) or abs(value / _HALF_POINT - round(value / _HALF_POINT)) > 0.0:
            raise ValueError(
                f"{name} must sit on the half point grid of the EDSS scale, got {value!r}"
            )
        if not math.isfinite(probability):
            raise ValueError(
                f"{name} must hold a finite probability, got {probability!r} at {value!r}"
            )
        if probability < 0.0:
            raise ValueError(
                f"{name} must hold no negative probability, got {probability!r} at {value!r}"
            )
    total = sum(law.values())
    if abs(total - 1.0) > _LAW_TOLERANCE:
        raise ValueError(f"{name} must sum to 1 within {_LAW_TOLERANCE:g}, got {total!r}")


@dataclass(frozen=True)
class EDSSSpec:
    """The parameters of the illustrative trajectory, with published defaults.

    Attributes
    ----------
    baseline : float
        The score the record opens at, in EDSS points. The default of 2.0 is
        the median first recorded EDSS at a first demyelinating event.
    time_to_nadir_weeks : int
        How many weeks a relapse takes to reach its peak deficit. The default
        of one week is the whole of the published median of 1 to 2 days on a
        weekly grid.
    recovery_tau_weeks : float
        The constant of the exponential decay from the peak towards the
        residual, in weeks. The default of 10 is a judgement call fitted to
        published median recovery times of 71 and 111 days.
    pira_per_year : float
        Linear progression independent of relapse activity, in EDSS points per
        year. The default of 0 keeps the trace purely relapse driven, which is
        the point of the exercise, and avoids counting an accrual already
        inside the residual law a second time. A year here is the 365.25 over 7
        weeks of ``msrelapse.stats.WEEKS_PER_YEAR``, which the whole package
        converts with, and not the flat 52 weeks the published formula is
        usually written with: the two differ by 0.3 percent, so a rate of 0.1
        raises the trace by 0.9966 points over 520 weeks rather than by 1.
    peak_law : collections.abc.Mapping
        The law the nadir increase of each relapse is drawn from, as a mapping
        from a point of the half point grid to its probability. The default is
        ``PEAK_LAW``.
    residual_law_mild : collections.abc.Mapping
        The law the residual is drawn from when the peak falls below
        ``SEVERE_PEAK``. The default is ``RESIDUAL_LAW_MILD``.
    residual_law_severe : collections.abc.Mapping
        The law the residual is drawn from when the peak reaches
        ``SEVERE_PEAK``. The default is ``RESIDUAL_LAW_SEVERE``.
    scale_min : float
        The bottom of the EDSS scale, 0 by default.
    scale_max : float
        The top of the EDSS scale, 10 by default.

    Raises
    ------
    ValueError
        If a law is empty, holds a value off the half point grid or a negative
        probability, or does not sum to one; if `baseline` or
        `recovery_tau_weeks` is not a positive number; if
        `time_to_nadir_weeks` is below one week; if `pira_per_year` is not
        finite; or if `scale_max` is not above `scale_min`.

    Notes
    -----
    Well supported by the literature: the baseline, the peak law, which
    reproduces three verified marginals, the pooled probability of 0.42 that a
    relapse leaves at least half a point, the half point grid itself, the
    dependence of the residual on the severity of the relapse, and a time to
    nadir of one week on a weekly grid.

    Judgement calls, in the words of the synthesis behind this module: the
    recovery constant, obtained by requiring that a typical one point excess
    falls below the quarter point display threshold at the published median
    recovery time, which gives 7.3 weeks against one cohort and 11.4 against
    the other, both of them treated; the split of the residual mass above 1.0,
    which is pinned only at the 2.0 threshold; the size of the mass below zero,
    which is inferred from arithmetic rather than measured; two severity
    classes rather than three; and a progression rate of zero.

    Examples
    --------
    >>> spec = EDSSSpec()
    >>> spec.baseline, spec.recovery_tau_weeks
    (2.0, 10.0)
    >>> round(sum(spec.peak_law.values()), 12)
    1.0
    """

    baseline: float = 2.0
    time_to_nadir_weeks: int = 1
    recovery_tau_weeks: float = 10.0
    pira_per_year: float = 0.0
    peak_law: Mapping[float, float] = field(default_factory=lambda: PEAK_LAW)
    residual_law_mild: Mapping[float, float] = field(default_factory=lambda: RESIDUAL_LAW_MILD)
    residual_law_severe: Mapping[float, float] = field(default_factory=lambda: RESIDUAL_LAW_SEVERE)
    scale_min: float = 0.0
    scale_max: float = 10.0

    def __post_init__(self) -> None:
        """Check every parameter, and say which one is out of range.

        Raises
        ------
        ValueError
            If any parameter is out of the range its attribute documents.
        """
        _check_positive("baseline", self.baseline, "EDSS points")
        _check_positive("recovery_tau_weeks", self.recovery_tau_weeks, "weeks")
        if self.time_to_nadir_weeks < 1:
            raise ValueError(
                f"time_to_nadir_weeks must be at least 1 week, got {self.time_to_nadir_weeks!r}"
            )
        if not math.isfinite(self.pira_per_year):
            raise ValueError(
                f"pira_per_year must be a finite number of EDSS points per year, "
                f"got {self.pira_per_year!r}"
            )
        if not self.scale_max > self.scale_min:
            raise ValueError(
                f"scale_max must be above scale_min, got {self.scale_max!r} and {self.scale_min!r}"
            )
        _check_law("peak_law", self.peak_law)
        _check_law("residual_law_mild", self.residual_law_mild)
        _check_law("residual_law_severe", self.residual_law_severe)


DEFAULT_SPEC: Final = EDSSSpec()
"""EDSSSpec: The published defaults, which every call here uses when given none."""


def relapse_episodes(states: npt.ArrayLike) -> list[tuple[int, int]]:
    """Return the maximal runs of the no health state of a weekly record.

    Parameters
    ----------
    states : array_like
        A one dimensional weekly state array, each entry the no health or the
        health code of the paper.

    Returns
    -------
    list of tuple of int
        One ``(start, end)`` pair per relapse episode, both of them week
        numbers counted from zero and the end week inside the episode. The
        pairs are in ascending order of the start week.

    Raises
    ------
    ValueError
        If `states` is not one dimensional, holds no week, or holds a value
        that is neither of the two state codes.

    Examples
    --------
    >>> relapse_episodes([-1, 1, 1, -1, 1])
    [(1, 2), (4, 4)]
    """
    weekly = _states_array(states)
    relapse = weekly == _NO_HEALTH
    # A False on each side turns every run into one rising and one falling
    # edge, so the edges pair off into the episodes whatever the record does at
    # its two ends.
    padded = np.concatenate(([False], relapse, [False]))
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    return [
        (int(start), int(stop) - 1) for start, stop in zip(edges[0::2], edges[1::2], strict=True)
    ]


def episode_draws(
    weekly: pd.DataFrame | npt.ArrayLike,
    spec: EDSSSpec = DEFAULT_SPEC,
    rng: Seed = None,
) -> pd.DataFrame:
    """Draw the peak and the residual of every relapse episode of one record.

    Each episode draws its peak deficit from the peak law of `spec`, then its
    residual from the law of the severity class that peak falls in, and the
    peak is then raised to the residual if the residual is the larger of the
    two, so that the kernel never rises again after its nadir. That last step
    raises about nine percent of peaks, summed exactly over the published laws,
    and lifts the mean peak from the 1.0525 of the law to 1.1163. A relapse
    whose peak is raised all the way to its residual steps up with no transient
    at all.

    Parameters
    ----------
    weekly : pandas.DataFrame or array_like
        The record to read, either a frame in the weekly schema of
        [`msrelapse.io`][msrelapse.io] holding one patient, which is validated
        before use, or a one dimensional weekly state array.
    spec : EDSSSpec, optional
        The parameters of the model. The default is ``DEFAULT_SPEC``.
    rng : numpy.random.Generator or int or None, optional
        Generator to draw from, or a seed for ``numpy.random.default_rng``. A
        seed is what makes a trajectory repeatable.

    Returns
    -------
    pandas.DataFrame
        One row per episode, in the order the episodes occur, with the columns
        ``episode`` (its index), ``start`` and ``end`` (week numbers, the end
        week inside the episode), ``duration_w``, ``peak``, ``residual`` and
        ``severity_class``, which is 'mild' or 'severe'.

    Raises
    ------
    ValueError
        If `weekly` is a frame that breaks the weekly schema or holds more than
        one patient, or an array that is not a one dimensional record of the
        two state codes.

    See Also
    --------
    edss_trajectory : The trace these draws produce, from the same seed.

    Examples
    --------
    >>> draws = episode_draws([1, 1, -1, -1, 1], rng=3)
    >>> list(draws["start"]), list(draws["end"])
    ([0, 4], [1, 4])
    """
    states = _record_states(weekly)
    generator = _generator(rng)
    episodes = relapse_episodes(states)
    peaks: list[float] = []
    residuals: list[float] = []
    classes: list[str] = []
    for _episode in episodes:
        peak = _draw(spec.peak_law, generator)
        severe = peak >= SEVERE_PEAK
        law = spec.residual_law_severe if severe else spec.residual_law_mild
        residual = _draw(law, generator)
        peaks.append(max(peak, residual))
        residuals.append(residual)
        classes.append("severe" if severe else "mild")
    starts = np.array([start for start, _end in episodes], dtype=np.int64)
    ends = np.array([end for _start, end in episodes], dtype=np.int64)
    return pd.DataFrame(
        {
            "episode": np.arange(len(episodes), dtype=np.int64),
            "start": starts,
            "end": ends,
            "duration_w": ends - starts + 1,
            "peak": np.array(peaks, dtype=np.float64),
            "residual": np.array(residuals, dtype=np.float64),
            "severity_class": np.array(classes, dtype=object),
        }
    )


def edss_trajectory(
    weekly: pd.DataFrame | npt.ArrayLike,
    spec: EDSSSpec = DEFAULT_SPEC,
    rng: Seed = None,
) -> pd.DataFrame:
    """Return the illustrative EDSS trace one weekly record produces.

    Every episode of the record contributes one kernel. Writing the onset week
    of episode i as u, its peak as A, its residual as r, the time to nadir as
    T and the recovery constant as tau, the kernel is zero before u, it is
    ``A * min(1, (t - u + 1) / T)`` while the deficit rises, and it is
    ``r + (A - r) * exp(-(t - u - T) / tau)`` from the nadir onwards. The trace
    is the baseline, plus the progression term, plus every kernel, clipped to
    the scale.

    Parameters
    ----------
    weekly : pandas.DataFrame or array_like
        The record to read, either a frame in the weekly schema of
        [`msrelapse.io`][msrelapse.io] holding one patient, which is validated
        before use, or a one dimensional weekly state array.
    spec : EDSSSpec, optional
        The parameters of the model. The default is ``DEFAULT_SPEC``.
    rng : numpy.random.Generator or int or None, optional
        Generator to draw from, or a seed for ``numpy.random.default_rng``. The
        same seed gives the same trace, and the same draws as
        [`episode_draws`][msrelapse.edss.episode_draws].

    Returns
    -------
    pandas.DataFrame
        One row per week of the record, with the columns ``week``, ``edss``,
        the continuous score, ``edss_display``, that score on the half point
        grid, and ``episode``, the index of the episode in progress that week
        or -1 in remission.

    Raises
    ------
    ValueError
        If `weekly` is a frame that breaks the weekly schema or holds more than
        one patient, or an array that is not a one dimensional record of the
        two state codes.

    See Also
    --------
    episode_draws : The draws behind the trace, from the same seed.
    summarise : A few numbers read off a finished trace.

    Notes
    -----
    A displayed score rounds half a point up, so a continuous 2.25 shows as 2.5
    and 2.24 as 2.0, which is how a clinician records a score that sits between
    two steps of the scale.

    The recovery constant is far longer than a relapse episode, about ten weeks
    against a mean of 4.3, so the trace does not step back down when the weekly
    series flips to remission: it goes on decaying towards the residual of that
    episode for months afterwards.

    Examples
    --------
    >>> trace = edss_trajectory([1, 1, -1, -1, -1], rng=3)
    >>> list(trace["week"])
    [0, 1, 2, 3, 4]
    >>> list(trace["episode"])
    [0, 0, -1, -1, -1]
    """
    states = _record_states(weekly)
    draws = episode_draws(states, spec, rng)
    weeks = np.arange(states.size, dtype=np.int64)
    elapsed = weeks.astype(np.float64)
    raw: _Vector = np.full(states.size, spec.baseline, dtype=np.float64)
    raw += spec.pira_per_year * elapsed / WEEKS_PER_YEAR
    episode = np.full(states.size, -1, dtype=np.int64)
    starts = draws["start"].to_numpy()
    ends = draws["end"].to_numpy()
    peaks = draws["peak"].to_numpy()
    residuals = draws["residual"].to_numpy()
    for index in range(len(draws)):
        start = int(starts[index])
        raw += _kernel(elapsed, start, float(peaks[index]), float(residuals[index]), spec)
        episode[start : int(ends[index]) + 1] = index
    edss = np.clip(raw, spec.scale_min, spec.scale_max)
    return pd.DataFrame(
        {
            "week": weeks,
            "edss": edss,
            "edss_display": _to_half_points(edss, spec),
            "episode": episode,
        }
    )


def expected_peak(spec: EDSSSpec = DEFAULT_SPEC) -> float:
    """Return the mean nadir increase of the peak law of a spec.

    Parameters
    ----------
    spec : EDSSSpec, optional
        The parameters to read the law off. The default is ``DEFAULT_SPEC``.

    Returns
    -------
    float
        The mean of the peak law, in EDSS points. The peaks a record actually
        draws average a little above it, because a peak is raised to its own
        residual whenever the residual is the larger of the two.

    Examples
    --------
    >>> round(expected_peak(), 4)
    1.0525
    """
    return _mean_of(spec.peak_law)


def expected_residual(spec: EDSSSpec = DEFAULT_SPEC) -> float:
    """Return the mean residual one relapse leaves, over both severity classes.

    Parameters
    ----------
    spec : EDSSSpec, optional
        The parameters to read the laws off. The default is ``DEFAULT_SPEC``.

    Returns
    -------
    float
        The mean of the residual law of the mild class and that of the severe
        class, weighted by the share of the peak law that falls in each, in
        EDSS points.

    Examples
    --------
    >>> round(expected_residual(), 5)
    0.25594
    """
    severe_share = sum(p for value, p in spec.peak_law.items() if value >= SEVERE_PEAK)
    mild_share = sum(p for value, p in spec.peak_law.items() if value < SEVERE_PEAK)
    mild = _mean_of(spec.residual_law_mild)
    severe = _mean_of(spec.residual_law_severe)
    return mild_share * mild + severe_share * severe


def summarise(trajectory: pd.DataFrame, draws: pd.DataFrame | None = None) -> dict[str, float]:
    """Return a few numbers read off a finished trace.

    Parameters
    ----------
    trajectory : pandas.DataFrame
        A trace, as
        [`edss_trajectory`][msrelapse.edss.edss_trajectory] returns it.
    draws : pandas.DataFrame, optional
        The draws behind that trace, as
        [`episode_draws`][msrelapse.edss.episode_draws] returns them for the
        same record, the same spec and the same seed. Giving them adds the one
        number the trace does not hold.

    Returns
    -------
    dict
        ``final_edss``, the score the record ends on, ``max_edss``, the highest
        score it reaches, and ``episodes``, how many relapses it holds, which a
        trace on its own always gives. ``episodes_with_a_residual``, how many
        of those relapses leave a residual above zero, is there only when
        `draws` is given: the trace does not hold what each episode left
        behind, so a caller that wants that number has to pass the draws.
        ``edss_at_ten_years`` is there only when the record reaches ten years.

    Raises
    ------
    ValueError
        If either frame is missing a column this reads, or if the draws hold a
        different number of episodes from the trace and so belong to another
        record, another spec or another seed.

    Notes
    -----
    The draws are a second argument rather than something read back out of the
    trace, because what each episode left behind is not in the trace: its four
    columns are the week, the two scores and the episode in progress, and the
    level at any week is the sum of every residual so far plus whatever the
    current episode is still decaying through. Reading the residuals back out
    of that would be guessing, so the dependency is explicit instead, and
    optional, so that a trace on its own still summarises. The two frames of
    one record, one spec and one seed agree by construction, and a pair that
    disagrees on the number of episodes is refused rather than summarised.

    Ten years are read at the week
    ``round(10 * msrelapse.stats.WEEKS_PER_YEAR)``, which is 522 rather than
    the 520 of ten 52 week years, so that a year means here what it means
    everywhere else in the package.

    Examples
    --------
    >>> states = [1, 1] + [-1] * 40
    >>> summarise(edss_trajectory(states, rng=3))["episodes"]
    1.0
    >>> summary = summarise(edss_trajectory(states, rng=3), episode_draws(states, rng=3))
    >>> summary["episodes"]
    1.0
    >>> "edss_at_ten_years" in summary
    False
    """
    _check_columns("trajectory", trajectory, ("week", "edss", "episode"))
    in_progress = trajectory["episode"].to_numpy()
    # The episodes are numbered from zero in the order they occur, so the
    # highest number in the column is the last episode of the record.
    episodes = int(in_progress.max()) + 1 if in_progress.size else 0
    summary = {
        "final_edss": float(trajectory["edss"].iloc[-1]),
        "max_edss": float(trajectory["edss"].max()),
        "episodes": float(episodes),
    }
    if draws is not None:
        _check_columns("draws", draws, ("residual",))
        if len(draws) != episodes:
            raise ValueError(
                "a trajectory and its draws must be of one record, one spec and one seed, "
                f"but the trajectory holds {episodes} episode(s) and the draws {len(draws)}"
            )
        summary["episodes_with_a_residual"] = float((draws["residual"].to_numpy() > 0.0).sum())
    ten_years = trajectory.loc[trajectory["week"] == _TEN_YEARS_WEEKS, "edss"]
    if not ten_years.empty:
        summary["edss_at_ten_years"] = float(ten_years.iloc[0])
    return summary


def _kernel(elapsed: _Vector, start: int, peak: float, residual: float, spec: EDSSSpec) -> _Vector:
    """Return the contribution of one episode to every week of the record.

    Parameters
    ----------
    elapsed : numpy.ndarray
        The week number of each row of the record, as a float.
    start : int
        The onset week of the episode.
    peak : float
        The nadir increase of the episode, in EDSS points.
    residual : float
        What the episode leaves behind for good, in EDSS points.
    spec : EDSSSpec
        The parameters, for the time to nadir and the recovery constant.

    Returns
    -------
    numpy.ndarray
        The kernel, of the same length as `elapsed`: zero before the onset,
        rising to the peak over the time to nadir, then decaying towards the
        residual.
    """
    nadir = float(spec.time_to_nadir_weeks)
    since = elapsed - float(start)
    kernel = np.zeros_like(elapsed)
    rising = (since >= 0.0) & (since < nadir)
    kernel[rising] = peak * np.minimum(1.0, (since[rising] + 1.0) / nadir)
    settled = since >= nadir
    kernel[settled] = residual + (peak - residual) * np.exp(
        -(since[settled] - nadir) / spec.recovery_tau_weeks
    )
    return kernel


def _to_half_points(edss: _Vector, spec: EDSSSpec) -> _Vector:
    """Return a continuous score on the half point grid of the scale.

    Parameters
    ----------
    edss : numpy.ndarray
        The continuous score, already clipped to the scale.
    spec : EDSSSpec
        The parameters, for the two ends of the scale.

    Returns
    -------
    numpy.ndarray
        The score rounded to the nearest half point, with a score exactly
        between two steps rounded up, and kept inside the scale.
    """
    steps = np.floor(edss / _HALF_POINT + 0.5) * _HALF_POINT
    return np.clip(steps, spec.scale_min, spec.scale_max)


def _mean_of(law: Mapping[float, float]) -> float:
    """Return the mean of a law written as a mapping from value to probability.

    Parameters
    ----------
    law : collections.abc.Mapping
        The law, whose probabilities sum to one.

    Returns
    -------
    float
        The mean.
    """
    return float(sum(value * probability for value, probability in law.items()))


def _draw(law: Mapping[float, float], generator: np.random.Generator) -> float:
    """Draw one value from a law written as a mapping from value to probability.

    Parameters
    ----------
    law : collections.abc.Mapping
        The law to draw from, whose probabilities sum to one.
    generator : numpy.random.Generator
        The generator the draw goes through.

    Returns
    -------
    float
        One value of the law, drawn with its own probability. The values are
        offered to the generator in ascending order, so that a given seed gives
        a given draw whatever order the mapping was written in.
    """
    values = sorted(law)
    probabilities = [law[value] for value in values]
    return float(generator.choice(values, p=probabilities))


def _record_states(weekly: pd.DataFrame | npt.ArrayLike) -> _States:
    """Return the weekly states of one patient, from a frame or from an array.

    Parameters
    ----------
    weekly : pandas.DataFrame or array_like
        A frame in the weekly schema holding one patient, or a one dimensional
        weekly state array.

    Returns
    -------
    numpy.ndarray
        The states, one entry per week of the record.

    Raises
    ------
    ValueError
        If a frame breaks the weekly schema or holds more than one patient, or
        if an array is not a one dimensional record of the two state codes.
    """
    if isinstance(weekly, pd.DataFrame):
        validate(weekly, "weekly")
        patients = weekly["patient_id"].unique()
        if len(patients) != 1:
            raise ValueError(
                "a trajectory is drawn for one patient at a time, but the frame holds "
                f"{len(patients)} of them: {sorted(patients)[:3]}"
            )
        return _states_array(weekly["state"].to_numpy())
    return _states_array(weekly)


def _states_array(states: npt.ArrayLike) -> _States:
    """Return a weekly state array, checked against the two codes of the paper.

    Parameters
    ----------
    states : array_like
        The record to check.

    Returns
    -------
    numpy.ndarray
        The record as int64.

    Raises
    ------
    ValueError
        If the record is not one dimensional, holds no week, holds a value that
        is not a finite number, or holds a value that is neither of the two
        state codes.
    """
    weekly = np.asarray(states)
    if weekly.ndim != 1:
        raise ValueError(f"a weekly record must be one dimensional, got shape {weekly.shape}")
    if weekly.size == 0:
        raise ValueError("a weekly record must hold at least one week, got an empty one")
    # Finiteness is checked before the cast rather than after it, because
    # casting a value that is not a number to an integer warns and then hands
    # back whatever the platform makes of it, so the check below would report
    # the wrong fault or none at all.
    if weekly.dtype.kind == "f":
        absent = np.flatnonzero(~np.isfinite(weekly))
        if absent.size:
            first = int(absent[0])
            raise ValueError(
                f"a weekly record must hold the finite state codes {_NO_HEALTH} and {_HEALTH}, "
                f"but week {first} holds {float(weekly[first])!r}"
            )
    codes = weekly.astype(np.int64)
    if not np.array_equal(codes, weekly):
        raise ValueError(f"a weekly record must hold whole state codes, got {weekly.dtype}")
    unknown = codes[(codes != _NO_HEALTH) & (codes != _HEALTH)]
    if unknown.size:
        raise ValueError(
            f"a weekly record holds the state codes {_NO_HEALTH} and {_HEALTH}, "
            f"got {int(unknown[0])}"
        )
    return codes


def _check_columns(name: str, frame: pd.DataFrame, columns: tuple[str, ...]) -> None:
    """Check that a frame holds the columns a call is about to read.

    Parameters
    ----------
    name : str
        How to name the frame in the message.
    frame : pandas.DataFrame
        The frame to check.
    columns : tuple of str
        The columns that have to be there.

    Returns
    -------
    None
        Nothing is returned; a frame missing a column raises instead.

    Raises
    ------
    ValueError
        If any of `columns` is missing.
    """
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{name} is missing the column(s) {missing}")


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
