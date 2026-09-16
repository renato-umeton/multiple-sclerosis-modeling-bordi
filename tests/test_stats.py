from __future__ import annotations

import math

import numpy as np
import numpy.typing as npt
import pandas as pd
import pytest
from scipy import stats as scipy_stats
from statsmodels.discrete.discrete_model import NegativeBinomial, Poisson

from msrelapse import _citation
from msrelapse._params import PAPER
from msrelapse.renewal import alternating_renewal, gamma_rates, rates_from_means, relapse_counts
from msrelapse.stats import (
    WEEKS_PER_YEAR,
    ARRResult,
    _nb_negative_loglik,
    _nb_negative_score,
    arr,
    compare_arr,
    patient_followup,
    relapse_free_curve,
    sample_size_arr,
)

EVENTS_COLUMNS = (
    "patient_id",
    "followup_start",
    "followup_end",
    "relapse_onset",
    "relapse_end",
)

COHORT_LAMBDA, COHORT_MU = rates_from_means(
    PAPER.tau_health_cohort_weeks.value,
    PAPER.tau_no_health_cohort_weeks.value,
)

# Onset rates drawn from a gamma distribution of shape 2 give counts that are
# negative binomial with dispersion 1 / 2, which is over dispersed enough for
# the negative binomial machinery to have something to estimate. The scale is
# chosen so that the mean onset rate is the cohort rate of the paper.
GAMMA_SHAPE = 2.0
GAMMA_SCALE = COHORT_LAMBDA / GAMMA_SHAPE

N_PER_ARM = 300
ARM_FOLLOWUP_W = 156.0
ARM_FOLLOWUP_YEARS = ARM_FOLLOWUP_W / WEEKS_PER_YEAR
TRUE_RATE_RATIO = 0.7

# A cohort this small carries little information about the dispersion, so the
# likelihood is weakly curved around its maximum and the refinement of the fit
# has to settle for the precision its differenced Hessian can deliver.
SMALL_N_PER_ARM = 30
SMALL_FOLLOWUP_W = 104.0

# Both reference fits are asked to stop only at the maximum of the likelihood,
# rather than wherever their optimiser first slows down, so that the comparison
# is between two answers to the same question. The two models reach that
# maximum with different optimisers, which spell the tolerance differently: the
# Poisson model iterates with Newton and reads tol, the negative binomial model
# with a quasi-Newton search that reads gtol. A tolerance under the wrong name
# is passed through, ignored, and warned about, and leaves the reference
# dispersion a part in ten thousand short of the maximum.
REFERENCE_TOLERANCE = 1e-10

# Relative agreement asked of every comparison against a reference fit. The two
# fits agree far more closely than this on the cohort below: the negative
# binomial rate ratio to 2e-16, its dispersion to 5e-11 and its standard error
# to 9e-12, and the Poisson pair to 5e-16 and 2e-16. The bound is not drawn any
# tighter because the reference is an iterative fit whose own stopping point
# moves by up to 2e-8 when it is started somewhere else, so a tighter bound
# would be measuring the reference optimiser rather than this package.
REFERENCE_AGREEMENT = 1e-6


def events_from_counts(counts: dict[str, int], followup_w: float) -> pd.DataFrame:
    """Return an events table giving each patient the requested number of relapses."""
    rows: list[tuple[str, float, float, float, float]] = []
    for patient in sorted(counts):
        n_relapses = counts[patient]
        if n_relapses == 0:
            rows.append((patient, 0.0, followup_w, math.nan, math.nan))
            continue
        for index in range(n_relapses):
            onset = 1.0 + 2.0 * index
            rows.append((patient, 0.0, followup_w, onset, onset + 0.5))
    frame = pd.DataFrame(rows, columns=list(EVENTS_COLUMNS))
    return frame.astype(dict.fromkeys(EVENTS_COLUMNS[1:], np.float64))


def events_at_onsets(onsets: list[float], followup_w: float) -> pd.DataFrame:
    """Return a one patient events table with a half week relapse at each onset."""
    rows = [("p0001", 0.0, followup_w, onset, onset + 0.5) for onset in onsets]
    frame = pd.DataFrame(rows, columns=list(EVENTS_COLUMNS))
    return frame.astype(dict.fromkeys(EVENTS_COLUMNS[1:], np.float64))


def empty_events() -> pd.DataFrame:
    """Return an events table with the schema columns and no patient at all."""
    frame = pd.DataFrame({name: [] for name in EVENTS_COLUMNS})
    return frame.astype({"patient_id": str, **dict.fromkeys(EVENTS_COLUMNS[1:], np.float64)})


def ten_relapses_in_twenty_patient_years() -> pd.DataFrame:
    """Return four patients followed five years each, with ten relapses between them."""
    five_years_w = 5.0 * WEEKS_PER_YEAR
    return events_from_counts({"p0001": 4, "p0002": 3, "p0003": 2, "p0004": 1}, five_years_w)


def raw_sample_size(
    arr_control: float, rate_ratio: float, dispersion: float, followup_years: float
) -> float:
    """Return the unrounded control arm size of a balanced trial at power 0.8, alpha 0.05."""
    z_alpha = float(scipy_stats.norm.ppf(0.975))
    z_power = float(scipy_stats.norm.ppf(0.8))
    arr_treated = arr_control * rate_ratio
    variance = (1.0 / (followup_years * arr_control) + dispersion) + (
        1.0 / (followup_years * arr_treated) + dispersion
    )
    return (z_alpha + z_power) ** 2 * variance / math.log(rate_ratio) ** 2


def arm_events(rate_scale: float, rate_seed: int, record_seed: int) -> pd.DataFrame:
    """Return one simulated arm whose patient onset rates are scaled by `rate_scale`.

    The records start in health rather than at a relapse onset, which is the
    enrolment a trial sees. Starting every patient at an onset instead, as the
    paper's own records do, would add one relapse to every count and leave the
    counts no more spread out than a Poisson.
    """
    rates = gamma_rates(GAMMA_SHAPE, GAMMA_SCALE, N_PER_ARM, rng=rate_seed) * rate_scale
    return alternating_renewal(
        rates,
        COHORT_MU,
        ARM_FOLLOWUP_W,
        n=N_PER_ARM,
        rng=record_seed,
        start_state="health",
    )


def small_arms(seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return two small arms sharing an onset rate, arm b scaled by the true rate ratio."""
    arm_a = alternating_renewal(
        COHORT_LAMBDA,
        COHORT_MU,
        SMALL_FOLLOWUP_W,
        n=SMALL_N_PER_ARM,
        rng=seed,
        start_state="health",
    )
    arm_b = alternating_renewal(
        COHORT_LAMBDA * TRUE_RATE_RATIO,
        COHORT_MU,
        SMALL_FOLLOWUP_W,
        n=SMALL_N_PER_ARM,
        rng=seed + 1000,
        start_state="health",
    )
    return arm_a, arm_b


def reference_design(
    events_a: pd.DataFrame, events_b: pd.DataFrame
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Return the counts, the design matrix and the exposure the reference fit uses."""
    counts_a = relapse_counts(events_a).to_numpy(dtype=np.float64)
    counts_b = relapse_counts(events_b).to_numpy(dtype=np.float64)
    counts = np.concatenate([counts_a, counts_b])
    arm = np.concatenate([np.zeros(counts_a.size), np.ones(counts_b.size)])
    design = np.column_stack([np.ones(counts.size), arm])
    exposure = np.full(counts.size, ARM_FOLLOWUP_YEARS)
    return counts, design, exposure


def interval_width(result: ARRResult) -> float:
    """Return the width of a confidence interval on the annualised relapse rate."""
    return result.ci_high - result.ci_low


@pytest.fixture(scope="module")
def overdispersed_events() -> pd.DataFrame:
    return arm_events(1.0, rate_seed=7, record_seed=8)


@pytest.fixture(scope="module")
def two_arms() -> tuple[pd.DataFrame, pd.DataFrame]:
    return arm_events(1.0, 101, 103), arm_events(TRUE_RATE_RATIO, 102, 104)


@pytest.fixture(scope="module")
def reference_negative_binomial(
    two_arms: tuple[pd.DataFrame, pd.DataFrame],
) -> npt.NDArray[np.float64]:
    counts, design, exposure = reference_design(*two_arms)
    fitted = NegativeBinomial(counts, design, exposure=exposure).fit(
        disp=0, maxiter=500, gtol=REFERENCE_TOLERANCE
    )
    return np.concatenate([fitted.params, fitted.bse])


@pytest.fixture(scope="module")
def reference_poisson(
    two_arms: tuple[pd.DataFrame, pd.DataFrame],
) -> npt.NDArray[np.float64]:
    counts, design, exposure = reference_design(*two_arms)
    fitted = Poisson(counts, design, exposure=exposure).fit(
        disp=0, maxiter=500, tol=REFERENCE_TOLERANCE
    )
    return np.concatenate([fitted.params, fitted.bse])


def test_weeks_per_year_is_the_julian_year() -> None:
    days_in_a_year = WEEKS_PER_YEAR * 7.0
    assert days_in_a_year == pytest.approx(365.25)
    assert 51.0 < WEEKS_PER_YEAR < 53.0


def test_patient_followup_is_the_window_length_in_weeks() -> None:
    events = ten_relapses_in_twenty_patient_years()
    followup = patient_followup(events)
    assert followup.to_numpy() == pytest.approx(np.full(4, 5.0 * WEEKS_PER_YEAR))


def test_patient_followup_has_one_entry_per_patient() -> None:
    events = ten_relapses_in_twenty_patient_years()
    followup = patient_followup(events)
    assert list(followup.index) == ["p0001", "p0002", "p0003", "p0004"]
    assert followup.index.name == "patient_id"


def test_patient_followup_of_a_table_with_no_patients_is_empty() -> None:
    # arr refuses the empty cohort, because it has no rate to report, but a
    # follow up of no patients is a follow up of no patients and is answered.
    followup = patient_followup(empty_events())
    assert followup.empty
    assert followup.index.name == "patient_id"


def test_patient_followup_rejects_a_frame_that_is_not_an_events_table() -> None:
    with pytest.raises(ValueError, match="missing the columns"):
        patient_followup(pd.DataFrame({"patient_id": ["p0001"]}))


def test_arr_is_relapses_over_patient_years() -> None:
    result = arr(ten_relapses_in_twenty_patient_years())
    assert result.n_patients == 4
    assert result.n_relapses == 10
    assert result.patient_years == pytest.approx(20.0)
    assert result.arr == pytest.approx(0.5)


def test_garwood_interval_matches_the_textbook_example() -> None:
    result = arr(ten_relapses_in_twenty_patient_years())
    assert result.ci_low == pytest.approx(0.2398, abs=1e-3)
    assert result.ci_high == pytest.approx(0.9195, abs=1e-3)
    assert result.ci_method == "poisson_exact"
    assert result.ci_level == pytest.approx(0.95)


def test_garwood_interval_brackets_the_point_estimate() -> None:
    result = arr(ten_relapses_in_twenty_patient_years())
    assert result.ci_low < result.arr < result.ci_high


def test_garwood_interval_starts_at_zero_when_there_is_no_relapse() -> None:
    events = events_from_counts(dict.fromkeys(["p0001", "p0002"], 0), 100.0)
    result = arr(events)
    assert result.n_relapses == 0
    assert result.arr == pytest.approx(0.0)
    assert result.ci_low == 0.0
    assert result.ci_high > 0.0


def test_a_wider_level_gives_a_wider_interval() -> None:
    events = ten_relapses_in_twenty_patient_years()
    narrow = arr(events, level=0.90)
    wide = arr(events, level=0.99)
    assert interval_width(wide) > interval_width(narrow)


def test_the_supplied_follow_up_replaces_the_denominator_only() -> None:
    # The argument is the exposure, not an analysis window: it does not decide
    # which relapses are counted, and the documentation says so.
    record = events_at_onsets([10.0, 30.0, 60.0, 80.0], 100.0)
    half = patient_followup(record) / 2.0
    result = arr(record, followup=half)
    assert result.n_relapses == 4
    assert result.patient_years == pytest.approx(50.0 / WEEKS_PER_YEAR)


def test_a_shorter_window_needs_the_events_trimmed_so_the_rate_holds() -> None:
    record = events_at_onsets([10.0, 30.0, 60.0, 80.0], 100.0)
    first_half = events_at_onsets([10.0, 30.0], 50.0)
    full = arr(record)
    short = arr(first_half)
    assert (full.n_relapses, short.n_relapses) == (4, 2)
    assert short.patient_years == pytest.approx(full.patient_years / 2.0)
    assert short.arr == pytest.approx(full.arr)


def test_arr_rejects_a_follow_up_that_is_missing_a_patient() -> None:
    events = ten_relapses_in_twenty_patient_years()
    partial = patient_followup(events).drop("p0002")
    with pytest.raises(ValueError, match="p0002"):
        arr(events, followup=partial)


def test_arr_rejects_a_follow_up_that_carries_an_unknown_patient() -> None:
    # A follow up with a patient the table does not hold is the follow up of
    # some other cohort, so it is reported rather than quietly reindexed away.
    events = ten_relapses_in_twenty_patient_years()
    followup = patient_followup(events)
    followup["p9999"] = 10.0
    with pytest.raises(ValueError, match="p9999"):
        arr(events, followup=followup)


def test_arr_rejects_a_follow_up_that_is_not_positive() -> None:
    events = ten_relapses_in_twenty_patient_years()
    followup = patient_followup(events)
    followup["p0001"] = 0.0
    with pytest.raises(ValueError, match="positive"):
        arr(events, followup=followup)


def test_arr_pools_unequal_follow_up() -> None:
    one_year = events_from_counts({"p0001": 1}, WEEKS_PER_YEAR)
    three_years = events_from_counts({"p0002": 3}, 3.0 * WEEKS_PER_YEAR)
    result = arr(pd.concat([one_year, three_years], ignore_index=True))
    assert result.patient_years == pytest.approx(4.0)
    assert result.arr == pytest.approx(1.0)


def test_arr_rejects_an_unknown_interval_method() -> None:
    with pytest.raises(ValueError, match="ci must be one of"):
        arr(ten_relapses_in_twenty_patient_years(), ci="jackknife")  # type: ignore[arg-type]


@pytest.mark.parametrize("level", [0.0, 1.0, 1.5])
def test_arr_rejects_a_level_outside_the_unit_interval(level: float) -> None:
    with pytest.raises(ValueError, match="level must be between"):
        arr(ten_relapses_in_twenty_patient_years(), level=level)


def test_arr_rejects_a_non_positive_bootstrap_size() -> None:
    with pytest.raises(ValueError, match="n_boot must be positive"):
        arr(ten_relapses_in_twenty_patient_years(), ci="bootstrap", n_boot=0)


def test_arr_rejects_a_non_positive_bootstrap_size_on_every_interval() -> None:
    # The argument is checked whatever the interval, so that a caller who asks
    # for the exact interval today is not surprised by the bootstrap tomorrow.
    with pytest.raises(ValueError, match="n_boot must be positive"):
        arr(ten_relapses_in_twenty_patient_years(), n_boot=-5)


@pytest.mark.parametrize("method", ["poisson_exact", "nb", "bootstrap"])
def test_every_interval_brackets_the_point_estimate(
    overdispersed_events: pd.DataFrame, method: str
) -> None:
    result = arr(overdispersed_events, ci=method, rng=5)  # type: ignore[arg-type]
    assert result.ci_low < result.arr < result.ci_high
    assert result.ci_method == method


@pytest.mark.parametrize("method", ["nb", "bootstrap"])
def test_over_dispersion_widens_the_interval(
    overdispersed_events: pd.DataFrame, method: str
) -> None:
    exact = arr(overdispersed_events, ci="poisson_exact")
    spread = arr(overdispersed_events, ci=method, rng=5)  # type: ignore[arg-type]
    assert interval_width(spread) > interval_width(exact)


def test_the_negative_binomial_interval_is_the_moment_formula(
    overdispersed_events: pd.DataFrame,
) -> None:
    # Recomputing the half width here pins the whole expression: the total
    # count in the denominator, the product of the dispersion and the mean
    # count in the numerator, and the sample variance the dispersion uses.
    counts = relapse_counts(overdispersed_events).to_numpy(dtype=np.float64)
    mean = float(counts.mean())
    dispersion = max(0.0, (float(counts.var(ddof=1)) - mean) / mean**2)
    half_width = float(scipy_stats.norm.ppf(0.975)) * math.sqrt(
        (1.0 + dispersion * mean) / counts.sum()
    )
    result = arr(overdispersed_events, ci="nb")
    assert math.log(result.ci_high / result.arr) == pytest.approx(half_width)
    assert math.log(result.arr / result.ci_low) == pytest.approx(half_width)


def test_the_negative_binomial_interval_collapses_onto_the_poisson_wald_interval() -> None:
    # Counts 0, 1 and 2 have a sample variance equal to their mean, so the
    # moment dispersion is exactly zero and the interval is the Wald interval
    # on the log of a Poisson rate, whose half width is z / sqrt(k). That
    # formula is written nowhere in the module, so it anchors the numerator
    # 1 + a m and the total count in the denominator against an outside one.
    events = events_from_counts({"p0001": 0, "p0002": 1, "p0003": 2}, 100.0)
    result = arr(events, ci="nb")
    half_width = float(scipy_stats.norm.ppf(0.975)) / math.sqrt(result.n_relapses)
    assert math.log(result.ci_high / result.arr) == pytest.approx(half_width)
    assert math.log(result.arr / result.ci_low) == pytest.approx(half_width)


def test_the_bootstrap_repeats_for_one_seed(overdispersed_events: pd.DataFrame) -> None:
    first = arr(overdispersed_events, ci="bootstrap", rng=3)
    second = arr(overdispersed_events, ci="bootstrap", rng=3)
    assert (first.ci_low, first.ci_high) == (second.ci_low, second.ci_high)


def test_the_bootstrap_follows_the_generator_it_is_given(
    overdispersed_events: pd.DataFrame,
) -> None:
    seeded = arr(overdispersed_events, ci="bootstrap", rng=3)
    generated = arr(overdispersed_events, ci="bootstrap", rng=np.random.default_rng(3))
    assert (seeded.ci_low, seeded.ci_high) == (generated.ci_low, generated.ci_high)


def test_the_negative_binomial_interval_needs_a_relapse() -> None:
    events = events_from_counts(dict.fromkeys(["p0001", "p0002"], 0), 100.0)
    with pytest.raises(ValueError, match="at least one relapse"):
        arr(events, ci="nb")


def test_the_negative_binomial_interval_needs_two_patients() -> None:
    events = events_from_counts({"p0001": 3}, 100.0)
    with pytest.raises(ValueError, match="at least two patients"):
        arr(events, ci="nb")


def test_the_bootstrap_needs_two_patients() -> None:
    # Every resample of a single patient is that patient, so the percentile
    # interval would collapse onto the point estimate and read as an estimate.
    events = events_from_counts({"p0001": 3}, 100.0)
    with pytest.raises(ValueError, match="at least two patients"):
        arr(events, ci="bootstrap")


def test_arr_rejects_a_cohort_with_no_patients() -> None:
    with pytest.raises(ValueError, match="no patients"):
        arr(empty_events())


def test_the_result_carries_a_citation() -> None:
    result = arr(ten_relapses_in_twenty_patient_years())
    assert PAPER.paper_doi.value in result.citation


def test_the_rate_ratio_carries_a_citation(two_arms: tuple[pd.DataFrame, pd.DataFrame]) -> None:
    result = compare_arr(two_arms[0], None, two_arms[1], None, model="poisson")
    assert PAPER.paper_doi.value in result.citation


def test_the_arr_repr_cites_the_paper_once() -> None:
    text = repr(arr(ten_relapses_in_twenty_patient_years()))
    assert text.count(PAPER.paper_doi.value) == 1


def test_the_arr_repr_names_the_rate_it_reports() -> None:
    assert "arr=" in repr(arr(ten_relapses_in_twenty_patient_years()))


def test_the_rate_ratio_repr_cites_the_paper_once(
    two_arms: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    text = repr(compare_arr(two_arms[0], None, two_arms[1], None, model="poisson"))
    assert text.count(PAPER.paper_doi.value) == 1


def test_the_rate_ratio_repr_names_the_ratio_it_reports(
    two_arms: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    text = repr(compare_arr(two_arms[0], None, two_arms[1], None, model="poisson"))
    assert "rate_ratio=" in text


def test_both_results_carry_the_citation_of_the_citation_module(
    two_arms: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    # One citation is built in one place and read from everywhere else, so the
    # results of this module quote it rather than spelling it out again.
    expected = _citation.short_citation()
    assert arr(ten_relapses_in_twenty_patient_years()).citation == expected
    assert compare_arr(two_arms[0], None, two_arms[1], None, model="poisson").citation == expected


def test_negative_binomial_rate_ratio_matches_statsmodels(
    two_arms: tuple[pd.DataFrame, pd.DataFrame],
    reference_negative_binomial: npt.NDArray[np.float64],
) -> None:
    result = compare_arr(two_arms[0], None, two_arms[1], None)
    assert result.rate_ratio == pytest.approx(
        math.exp(reference_negative_binomial[1]), rel=REFERENCE_AGREEMENT
    )


def test_negative_binomial_dispersion_matches_statsmodels(
    two_arms: tuple[pd.DataFrame, pd.DataFrame],
    reference_negative_binomial: npt.NDArray[np.float64],
) -> None:
    result = compare_arr(two_arms[0], None, two_arms[1], None)
    assert result.dispersion == pytest.approx(
        reference_negative_binomial[2], rel=REFERENCE_AGREEMENT
    )


def test_negative_binomial_standard_error_matches_statsmodels(
    two_arms: tuple[pd.DataFrame, pd.DataFrame],
    reference_negative_binomial: npt.NDArray[np.float64],
) -> None:
    result = compare_arr(two_arms[0], None, two_arms[1], None)
    half_width = math.log(result.ci_high / result.rate_ratio)
    standard_error = half_width / scipy_stats.norm.ppf(0.975)
    assert standard_error == pytest.approx(reference_negative_binomial[4], rel=REFERENCE_AGREEMENT)


def test_poisson_rate_ratio_matches_statsmodels(
    two_arms: tuple[pd.DataFrame, pd.DataFrame],
    reference_poisson: npt.NDArray[np.float64],
) -> None:
    result = compare_arr(two_arms[0], None, two_arms[1], None, model="poisson")
    assert result.rate_ratio == pytest.approx(
        math.exp(reference_poisson[1]), rel=REFERENCE_AGREEMENT
    )
    assert result.model == "poisson"


def test_poisson_standard_error_matches_statsmodels(
    two_arms: tuple[pd.DataFrame, pd.DataFrame],
    reference_poisson: npt.NDArray[np.float64],
) -> None:
    result = compare_arr(two_arms[0], None, two_arms[1], None, model="poisson")
    half_width = math.log(result.ci_high / result.rate_ratio)
    standard_error = half_width / scipy_stats.norm.ppf(0.975)
    assert standard_error == pytest.approx(reference_poisson[3], rel=REFERENCE_AGREEMENT)


def test_the_interval_covers_the_simulated_rate_ratio(
    two_arms: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    result = compare_arr(two_arms[0], None, two_arms[1], None)
    assert result.ci_low < TRUE_RATE_RATIO < result.ci_high
    assert result.converged


def test_a_small_cohort_still_reports_a_converged_fit() -> None:
    # The refinement of the fit differences its Hessian, so its step cannot
    # shrink past the error of that difference. Judging convergence at the
    # precision of an exact Hessian asked for a step smaller than the
    # differenced one can deliver and reported a settled fit as a failure.
    arm_a, arm_b = small_arms(28)
    result = compare_arr(arm_a, None, arm_b, None)
    assert result.converged
    assert 0.0 < result.rate_ratio < 1.0
    assert result.dispersion > 0.0


def test_a_poisson_fit_that_never_meets_its_tolerance_reports_a_failure(
    two_arms: tuple[pd.DataFrame, pd.DataFrame],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The flag reports the iteration rather than the data. Asked for a step no
    # iteration can deliver, the loop spends its whole budget and comes back
    # saying so, with the estimate the settled fit stops on.
    settled = compare_arr(two_arms[0], None, two_arms[1], None, model="poisson")
    monkeypatch.setattr("msrelapse.stats._NEWTON_TOLERANCE", 0.0)
    result = compare_arr(two_arms[0], None, two_arms[1], None, model="poisson")
    assert result.converged is False
    assert result.rate_ratio == pytest.approx(settled.rate_ratio)


def test_a_negative_binomial_refinement_that_never_settles_reports_a_failure(
    two_arms: tuple[pd.DataFrame, pd.DataFrame],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # As above for the refinement of the negative binomial fit, which carries
    # the convergence flag of that model.
    settled = compare_arr(two_arms[0], None, two_arms[1], None)
    monkeypatch.setattr("msrelapse.stats._POLISH_TOLERANCE", 0.0)
    result = compare_arr(two_arms[0], None, two_arms[1], None)
    assert result.converged is False
    assert result.rate_ratio == pytest.approx(settled.rate_ratio)
    assert result.dispersion == pytest.approx(settled.dispersion)


def test_a_supplied_follow_up_is_the_exposure_of_the_arm_it_belongs_to(
    two_arms: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    # The Poisson fit reproduces each arm's crude rate, so halving the exposure
    # of arm b doubles its fitted rate exactly and leaves arm a untouched. The
    # two arms' follow up being swapped would move the other rate instead.
    recorded = compare_arr(two_arms[0], None, two_arms[1], None, model="poisson")
    halved = compare_arr(
        two_arms[0], None, two_arms[1], patient_followup(two_arms[1]) / 2.0, model="poisson"
    )
    assert halved.arr_a == pytest.approx(recorded.arr_a)
    assert halved.arr_b == pytest.approx(2.0 * recorded.arr_b)
    assert halved.rate_ratio == pytest.approx(2.0 * recorded.rate_ratio)


def test_halving_the_reference_arm_exposure_halves_the_rate_ratio(
    two_arms: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    recorded = compare_arr(two_arms[0], None, two_arms[1], None, model="poisson")
    halved = compare_arr(
        two_arms[0], patient_followup(two_arms[0]) / 2.0, two_arms[1], None, model="poisson"
    )
    assert halved.arr_a == pytest.approx(2.0 * recorded.arr_a)
    assert halved.arr_b == pytest.approx(recorded.arr_b)
    assert halved.rate_ratio == pytest.approx(recorded.rate_ratio / 2.0)


def test_compare_arr_rejects_a_follow_up_that_carries_an_unknown_patient(
    two_arms: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    followup = patient_followup(two_arms[1])
    followup["p9999"] = 10.0
    with pytest.raises(ValueError, match="p9999"):
        compare_arr(two_arms[0], None, two_arms[1], followup)


def test_the_rate_ratio_interval_widens_by_the_quantile_of_the_level(
    two_arms: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    # The half width on the log scale is the normal quantile of the level
    # times one standard error, so the ratio of two half widths is the ratio
    # of the two quantiles and nothing else.
    narrow = compare_arr(two_arms[0], None, two_arms[1], None, level=0.95)
    wide = compare_arr(two_arms[0], None, two_arms[1], None, level=0.99)
    widening = math.log(wide.ci_high / wide.rate_ratio) / math.log(
        narrow.ci_high / narrow.rate_ratio
    )
    expected = float(scipy_stats.norm.ppf(0.995) / scipy_stats.norm.ppf(0.975))
    assert widening == pytest.approx(expected)


@pytest.mark.parametrize("log_dispersion", [-800.0, 800.0])
def test_the_likelihood_survives_a_search_that_wanders_off(log_dispersion: float) -> None:
    # A line search is free to probe far from its start, and the exponential of
    # the free parameter there is zero or infinite. The likelihood and its
    # gradient answer with a finite number instead of an arithmetic error, so
    # that the optimiser turns back on its own.
    design = np.column_stack([np.ones(4), np.array([0.0, 0.0, 1.0, 1.0])])
    counts = np.array([2.0, 1.0, 0.0, 3.0])
    exposure = np.full(4, 2.0)
    parameters = np.array([-0.5, -0.2, log_dispersion])
    assert math.isfinite(_nb_negative_loglik(parameters, design, counts, exposure))
    assert np.all(np.isfinite(_nb_negative_score(parameters, design, counts, exposure)))


def test_the_rate_ratio_is_the_ratio_of_the_two_fitted_rates(
    two_arms: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    result = compare_arr(two_arms[0], None, two_arms[1], None)
    assert result.rate_ratio == pytest.approx(result.arr_b / result.arr_a)
    assert result.model == "nb"
    assert result.dispersion > 0.0


def test_the_poisson_fit_reproduces_the_crude_rates(
    two_arms: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    result = compare_arr(two_arms[0], None, two_arms[1], None, model="poisson")
    assert result.arr_a == pytest.approx(arr(two_arms[0]).arr)
    assert result.arr_b == pytest.approx(arr(two_arms[1]).arr)


def test_over_dispersion_widens_the_rate_ratio_interval(
    two_arms: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    negative_binomial = compare_arr(two_arms[0], None, two_arms[1], None)
    poisson = compare_arr(two_arms[0], None, two_arms[1], None, model="poisson")
    assert negative_binomial.ci_high / negative_binomial.ci_low > poisson.ci_high / poisson.ci_low
    assert negative_binomial.p_value > poisson.p_value


def test_identical_arms_show_no_difference() -> None:
    arm_a = arm_events(1.0, 201, 203)
    arm_b = arm_events(1.0, 202, 204)
    result = compare_arr(arm_a, None, arm_b, None)
    assert result.rate_ratio == pytest.approx(1.0, abs=0.15)
    assert result.p_value > 0.05
    assert result.ci_low < 1.0 < result.ci_high


def test_counts_without_over_dispersion_fall_back_to_the_poisson_errors() -> None:
    arm_a = events_from_counts(dict.fromkeys([f"p000{i}" for i in range(1, 7)], 2), 100.0)
    arm_b = events_from_counts(dict.fromkeys([f"p000{i}" for i in range(1, 7)], 1), 100.0)
    negative_binomial = compare_arr(arm_a, None, arm_b, None)
    poisson = compare_arr(arm_a, None, arm_b, None, model="poisson")
    assert negative_binomial.model == "nb"
    assert negative_binomial.dispersion == 0.0
    assert negative_binomial.ci_low == pytest.approx(poisson.ci_low)
    assert negative_binomial.ci_high == pytest.approx(poisson.ci_high)
    assert negative_binomial.rate_ratio == pytest.approx(0.5)


def test_compare_arr_rejects_an_unknown_model(
    two_arms: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    with pytest.raises(ValueError, match="model must be one of"):
        compare_arr(two_arms[0], None, two_arms[1], None, model="tweedie")  # type: ignore[arg-type]


def test_compare_arr_rejects_an_arm_without_a_relapse() -> None:
    arm_a = events_from_counts(dict.fromkeys(["p0001", "p0002"], 2), 100.0)
    arm_b = events_from_counts(dict.fromkeys(["p0001", "p0002"], 0), 100.0)
    with pytest.raises(ValueError, match="at least one relapse"):
        compare_arr(arm_a, None, arm_b, None)


def test_relapse_free_curve_starts_at_one_and_decreases() -> None:
    grid = np.linspace(0.0, 500.0, 26)
    curve = relapse_free_curve(grid, lam=COHORT_LAMBDA)
    assert curve[0] == pytest.approx(1.0)
    assert np.all(np.diff(curve) < 0.0)


def test_relapse_free_curve_is_the_exponential_survival() -> None:
    grid = np.array([0.0, 10.0, 100.0, 1000.0])
    curve = relapse_free_curve(grid, lam=COHORT_LAMBDA)
    assert curve == pytest.approx(np.exp(-COHORT_LAMBDA * grid))


def test_relapse_free_curve_mixes_over_the_gamma_rates() -> None:
    grid = np.array([0.0, 10.0, 100.0, 1000.0])
    curve = relapse_free_curve(grid, k=GAMMA_SHAPE, theta=GAMMA_SCALE)
    expected = (1.0 + GAMMA_SCALE * grid) ** -GAMMA_SHAPE
    assert curve == pytest.approx(expected)


def test_relapse_free_curve_returns_an_array() -> None:
    curve = relapse_free_curve([0.0, 50.0], lam=COHORT_LAMBDA)
    assert isinstance(curve, np.ndarray)
    assert curve.dtype == np.float64


def test_relapse_free_curve_rejects_a_missing_parameterisation() -> None:
    with pytest.raises(ValueError, match="lam"):
        relapse_free_curve([0.0, 50.0])


def test_sample_size_reproduces_the_formula() -> None:
    expected = math.ceil(raw_sample_size(0.5, 0.7, 0.8, 2.0))
    assert sample_size_arr(0.5, 0.7, 0.8, 2.0) == expected


def test_sample_size_grows_with_the_dispersion() -> None:
    low = sample_size_arr(0.5, 0.7, 0.2, 2.0)
    high = sample_size_arr(0.5, 0.7, 0.8, 2.0)
    assert high > low


def test_sample_size_falls_as_the_rate_ratio_moves_away_from_one() -> None:
    near = sample_size_arr(0.5, 0.9, 0.8, 2.0)
    far = sample_size_arr(0.5, 0.5, 0.8, 2.0)
    assert far < near


def test_sample_size_grows_with_the_power() -> None:
    modest = sample_size_arr(0.5, 0.7, 0.8, 2.0, power=0.8)
    strong = sample_size_arr(0.5, 0.7, 0.8, 2.0, power=0.9)
    assert strong > modest


def test_a_larger_treated_arm_needs_fewer_control_patients() -> None:
    balanced = sample_size_arr(0.5, 0.7, 0.8, 2.0)
    unbalanced = sample_size_arr(0.5, 0.7, 0.8, 2.0, allocation=2.0)
    assert unbalanced < balanced


def test_an_unlimited_treated_arm_leaves_only_the_control_variance() -> None:
    # As the treated arm grows the second variance term is divided away, so the
    # control arm size tends to the one the control term alone asks for.
    z_alpha = float(scipy_stats.norm.ppf(0.975))
    z_power = float(scipy_stats.norm.ppf(0.8))
    control_only = (z_alpha + z_power) ** 2 * (1.0 / (2.0 * 0.5) + 0.8) / math.log(0.7) ** 2
    assert sample_size_arr(0.5, 0.7, 0.8, 2.0, allocation=1e6) == math.ceil(control_only)


@pytest.mark.parametrize(("rate_ratio", "over_half"), [(0.7, True), (0.5, False)])
def test_sample_size_rounds_a_fraction_of_a_patient_up(rate_ratio: float, over_half: bool) -> None:
    # The two rate ratios straddle a half patient, so rounding to the nearest
    # whole patient rather than up would answer one patient short of the
    # second case and the pair rules that out.
    raw = raw_sample_size(0.5, rate_ratio, 0.8, 2.0)
    assert (raw - math.floor(raw) > 0.5) is over_half
    assert sample_size_arr(0.5, rate_ratio, 0.8, 2.0) == math.floor(raw) + 1


def test_sample_size_rejects_a_rate_ratio_of_one() -> None:
    with pytest.raises(ValueError, match="rate_ratio must not be 1"):
        sample_size_arr(0.5, 1.0, 0.8, 2.0)


def test_sample_size_rejects_a_power_at_or_below_half_the_significance_level() -> None:
    # The formula squares the sum of the two normal quantiles. They cancel at
    # this point and the sum turns negative below it, so answering would hand
    # back a trial that grows as the request for power gets weaker.
    with pytest.raises(ValueError, match="power must be above alpha"):
        sample_size_arr(0.5, 0.7, 0.8, 2.0, power=0.025, alpha=0.05)
    with pytest.raises(ValueError, match="power must be above alpha"):
        sample_size_arr(0.5, 0.7, 0.8, 2.0, power=0.01, alpha=0.99)


def test_sample_size_falls_with_the_power_all_the_way_to_that_point() -> None:
    # Everywhere the function answers, a weaker request asks for fewer
    # patients, which is the property the refusal below the turn protects.
    sizes = [sample_size_arr(0.5, 0.7, 0.8, 2.0, power=power) for power in (0.03, 0.1, 0.5, 0.9)]
    assert sizes == sorted(sizes)


@pytest.mark.parametrize(
    ("name", "kwargs"),
    [
        ("arr_control", {"arr_control": 0.0}),
        ("rate_ratio", {"rate_ratio": -0.5}),
        ("dispersion", {"dispersion": -0.1}),
        ("followup_years", {"followup_years": 0.0}),
        ("allocation", {"allocation": 0.0}),
        ("power", {"power": 1.0}),
        ("alpha", {"alpha": 0.0}),
    ],
)
def test_sample_size_rejects_impossible_inputs(name: str, kwargs: dict[str, float]) -> None:
    arguments: dict[str, float] = {
        "arr_control": 0.5,
        "rate_ratio": 0.7,
        "dispersion": 0.8,
        "followup_years": 2.0,
    }
    arguments.update(kwargs)
    with pytest.raises(ValueError, match=name):
        sample_size_arr(**arguments)
