from __future__ import annotations

import doctest
import math
from importlib import metadata

import numpy as np
import numpy.typing as npt
import pandas as pd
import pytest
from scipy import stats

from msrelapse import _citation, fit, io
from msrelapse._params import PAPER
from msrelapse.renewal import alternating_renewal, gamma_rates, rates_from_means

RELAPSE = PAPER.state_no_health.value
REMISSION = PAPER.state_health.value
DOI = PAPER.paper_doi.value

# One seed for the whole file. Every statistical assertion below was checked to
# hold with a comfortable margin at this seed, so a failure means a change of
# behaviour rather than an unlucky draw.
SEED = 20130910

COHORT_LAMBDA, COHORT_MU = rates_from_means(
    PAPER.tau_health_cohort_weeks.value,
    PAPER.tau_no_health_cohort_weeks.value,
)

MEMORYLESS_METHODS = ("hazard", "cv", "ks", "ad")
PERIODICITY_METHODS = ("fisher_g", "lombscargle")

# The result classes this module returns beside FitResult. Each of them carries
# the citation of the article and repeats it in its repr, once.
_Result = fit.TestResult | fit.PeriodicityResult | fit.NBFit | fit.GammaFit
RESULT_NAMES = ("TestResult", "PeriodicityResult", "NBFit", "GammaFit")

# A field of each result that its repr has to name beside the citation, so that
# a repr cut down to the citation alone fails here.
RESULT_FIELDS = {
    "TestResult": "method=",
    "PeriodicityResult": "n_patients=",
    "NBFit": "dispersion=",
    "GammaFit": "k=",
}

# All four methods read their p value off replicates drawn under the rounded
# exponential null. These three are the ones that read any sample at all: the
# hazard needs three weeks with enough runs still at risk, so it is left out of
# the tests that feed a degenerate sample.
BOOTSTRAP_METHODS = ("cv", "ks", "ad")

# Sample sizes and shapes of the memorylessness fixtures, from the task
# specification: 500 durations of mean 20 weeks, ageing (Weibull shape 2) in one
# case and memoryless (geometric) in the other.
N_DURATIONS = 500
MEAN_DURATION = 20.0
AGEING_SHAPE = 2.0

# The cohort this module exists to analyse: the 218 relapses of the paper, whose
# mean length is 4.3 weeks. Forty such cohorts are drawn as ceil(Exp(4.3)), which
# is memoryless by construction and rounded the way the study rounds, so a test
# of memorylessness may not reject many more of them than its own level allows.
CALIBRATION_COHORTS = 40
CALIBRATION_BOOT = 150
CALIBRATION_ALLOWED_REJECTIONS = 6

# Seed of the generator the hazard replicates of that calibration are drawn
# from. It is separate from SEED so that the cohorts and the other three methods
# read exactly the draws they read before the hazard method joined them.
CALIBRATION_HAZARD_SEED = 20130911

# The overdispersed counts of the specification: mean 3, dispersion 0.5.
NB_MEAN = 3.0
NB_DISPERSION = 0.5

# Cohorts of seventeen patients whose sample variance exceeds their mean while
# their population variance does not. The score of the dispersion at zero is the
# second comparison, not the first, so the maximum likelihood dispersion of these
# counts sits on the boundary and the fit is the Poisson one.
EDGE_COUNTS = (
    (13, 17, 11, 8, 9, 7, 7, 9, 6, 7, 8, 6, 11, 12, 9, 7, 5),
    (12, 14, 11, 14, 9, 3, 9, 6, 11, 7, 8, 12, 10, 7, 12, 5, 9),
)

# The injected cycle of the periodicity fixtures: a four week relapse once a
# year over ten years of weekly follow up, with a phase of its own per patient
# and an onset jittered by at most two weeks.
CYCLE_WEEKS = 52
RECORD_WEEKS = 520
RELAPSE_WEEKS = 4
ONSET_JITTER_WEEKS = 2

# Patients of the cohort carrying the injected cycle, and of the single
# memoryless cohort the tests read beside it. How far a reported period may sit
# from the injected one, as a fraction of it, and how many of the twenty patients
# have to report a period that close.
PERIODICITY_PATIENTS = 20
PERIOD_TOLERANCE = 0.1
PATIENTS_WITH_THE_RIGHT_PERIOD = 15

# Seeds the injected cycle is drawn at. Both power tests read every one of them,
# so the margin they clear belongs to the fixture rather than to one lucky draw,
# and the claim the docstring of yearly_cycle_weekly makes about the jitter is
# checked at each seed rather than asserted once.
PERIODICITY_FIXTURE_SEEDS = (SEED, 1, 99)

# The null cohort of the periodicity tests: the seventy records of the study over
# eight years, which at the durations of the paper is about four relapses each.
# Ten such cohorts may not be called periodic more than twice at the 0.05 level,
# which is already more than a correctly sized test produces.
NULL_PATIENTS = 70
NULL_WEEKS = 400.0
NULL_COHORTS = 10
ALLOWED_PERIODICITY_REJECTIONS = 2

# Permutations behind the lombscargle p value in the tests of this file. The
# default of the module is the same number and the tests do not vary it.
N_PERM = 200

# Shortest record the periodicity test reads, and the length at which leaving the
# Nyquist ordinate in distorts the exact p value the most.
_MIN_WEEKS = 8

# Record lengths the size of the exact g test is measured at: the eight week
# minimum, a record of the length the study follows patients for, and one long
# enough to carry hundreds of ordinates. All three are even, which is the parity
# that has a Nyquist ordinate to leave out.
WHITE_NOISE_WEEKS = (_MIN_WEEKS, 200, 1000)


def one_run_per_patient(values: npt.NDArray[np.int64], state: int) -> pd.DataFrame:
    """Return a durations frame with one complete run of `state` per patient."""
    n = values.size
    return pd.DataFrame(
        {
            "patient_id": pd.Series([f"p{index:04d}" for index in range(1, n + 1)], dtype="str"),
            "run_index": np.zeros(n, dtype=np.int64),
            "state": np.full(n, state, dtype=np.int64),
            "duration_w": values.astype(np.int64),
            "censored": np.zeros(n, dtype=bool),
        }
    )


def durations_frame(runs: dict[str, list[tuple[int, int]]]) -> pd.DataFrame:
    """Return a durations frame from a mapping of patient to (state, weeks) runs."""
    patients: list[str] = []
    indices: list[int] = []
    states: list[int] = []
    lengths: list[int] = []
    censored: list[bool] = []
    for patient in sorted(runs):
        patient_runs = runs[patient]
        for index, (state, weeks) in enumerate(patient_runs):
            patients.append(patient)
            indices.append(index)
            states.append(state)
            lengths.append(weeks)
            censored.append(index == len(patient_runs) - 1 and state == REMISSION)
    return pd.DataFrame(
        {
            "patient_id": pd.Series(patients, dtype="str"),
            "run_index": np.array(indices, dtype=np.int64),
            "state": np.array(states, dtype=np.int64),
            "duration_w": np.array(lengths, dtype=np.int64),
            "censored": np.array(censored, dtype=bool),
        }
    )


def weekly_frame(records: dict[str, npt.NDArray[np.int64]]) -> pd.DataFrame:
    """Return a weekly frame from a mapping of patient to a state series."""
    patients = sorted(records)
    return pd.DataFrame(
        {
            "patient_id": pd.Series(
                np.repeat(patients, [records[patient].size for patient in patients]), dtype="str"
            ),
            "week": np.concatenate(
                [np.arange(records[patient].size, dtype=np.int64) for patient in patients]
            ),
            "state": np.concatenate([records[patient].astype(np.int64) for patient in patients]),
        }
    )


def cohort_durations(n: int = 30, t_end: float = 600.0, seed: int = SEED) -> pd.DataFrame:
    """Return a durations frame simulated at the cohort rates of the paper."""
    events = alternating_renewal(COHORT_LAMBDA, COHORT_MU, t_end, n=n, rng=seed, discretise="week")
    return io.weekly_to_durations(io.events_to_weekly(events))


def ageing_durations() -> npt.NDArray[np.int64]:
    """Return Weibull durations of mean 20 weeks, rounded up to whole weeks."""
    rng = np.random.default_rng(SEED)
    scale = MEAN_DURATION / math.gamma(1.0 + 1.0 / AGEING_SHAPE)
    drawn = rng.weibull(AGEING_SHAPE, N_DURATIONS) * scale
    return np.maximum(np.ceil(drawn), 1.0).astype(np.int64)


def memoryless_durations() -> npt.NDArray[np.int64]:
    """Return geometric durations of mean 20 weeks."""
    rng = np.random.default_rng(SEED)
    return rng.geometric(1.0 / MEAN_DURATION, N_DURATIONS).astype(np.int64)


def rounded_exponential(mean: float, n: int, rng: np.random.Generator) -> npt.NDArray[np.int64]:
    """Return `n` memoryless durations of scale `mean`, rounded up to whole weeks."""
    return np.maximum(np.ceil(rng.exponential(mean, n)), 1.0).astype(np.int64)


def negative_binomial_counts(n: int = 2000) -> npt.NDArray[np.int64]:
    """Return counts of mean 3 and dispersion 0.5, the NB2 fixture of this file."""
    return np.asarray(
        np.random.default_rng(SEED).negative_binomial(
            1.0 / NB_DISPERSION, 1.0 / (1.0 + NB_DISPERSION * NB_MEAN), n
        ),
        dtype=np.int64,
    )


def yearly_cycle_weekly(
    n_patients: int = PERIODICITY_PATIENTS,
    seed: int = SEED,
) -> pd.DataFrame:
    """Return a weekly cohort whose relapses arrive once a year.

    Every patient gets a phase of its own, and every onset is moved by up to
    :data:`ONSET_JITTER_WEEKS` weeks either way. The jitter is what makes the
    record a noisy periodic process rather than a deterministic one, and both
    methods need it. A record whose gaps are all exactly 52 weeks is unchanged
    by a shuffle of those gaps, so the permutation p value of ``lombscargle``
    would have nothing to permute and would come back at 1 however periodic the
    record is; and the periodogram of a train of evenly spaced impulses carries
    exactly as much power at every harmonic of the cycle as at the cycle itself,
    so the period ``fisher_g`` reports would be decided by rounding error.
    """
    generator = np.random.default_rng(seed)
    records: dict[str, npt.NDArray[np.int64]] = {}
    for index in range(1, n_patients + 1):
        series = np.full(RECORD_WEEKS, REMISSION, dtype=np.int64)
        phase = int(generator.integers(0, CYCLE_WEEKS))
        for scheduled in range(phase, RECORD_WEEKS, CYCLE_WEEKS):
            jitter = int(generator.integers(-ONSET_JITTER_WEEKS, ONSET_JITTER_WEEKS + 1))
            onset = min(max(scheduled + jitter, 0), RECORD_WEEKS - 1)
            series[onset : onset + RELAPSE_WEEKS] = RELAPSE
        records[f"p{index:04d}"] = series
    return weekly_frame(records)


def renewal_weekly(n_patients: int, weeks: float, seed: int) -> pd.DataFrame:
    """Return a weekly cohort of memoryless records at the durations of the paper."""
    events = alternating_renewal(
        COHORT_LAMBDA, COHORT_MU, weeks, n=n_patients, rng=seed, discretise="week"
    )
    return io.events_to_weekly(events)


def onset_weeks(weekly: pd.DataFrame, patient: str) -> list[int]:
    """Return the weeks in which a relapse starts for one patient of a frame."""
    series = weekly.loc[weekly["patient_id"] == patient, "state"].to_numpy(dtype=np.int64)
    relapse = series == RELAPSE
    started = [week for week in range(series.size) if relapse[week]]
    return [week for week in started if week == 0 or not relapse[week - 1]]


def record_with_onsets(onsets: list[int], n_weeks: int) -> npt.NDArray[np.int64]:
    """Return a weekly record of one week relapses starting in the given weeks."""
    series = np.full(n_weeks, REMISSION, dtype=np.int64)
    series[onsets] = RELAPSE
    return series


@pytest.fixture(scope="module")
def coverage_counts() -> dict[int, int]:
    """Return how many of 200 replicates had a Fisher interval covering the truth."""
    generator = np.random.default_rng(SEED)
    targets = {
        REMISSION: PAPER.tau_health_cohort_weeks.value,
        RELAPSE: PAPER.tau_no_health_cohort_weeks.value,
    }
    covered = {REMISSION: 0, RELAPSE: 0}
    for _ in range(200):
        events = alternating_renewal(
            COHORT_LAMBDA, COHORT_MU, 600.0, n=30, rng=generator, discretise="week"
        )
        durations = io.weekly_to_durations(io.events_to_weekly(events))
        for state, target in targets.items():
            result = fit.fit_durations(durations, state)
            covered[state] += int(result.ci_low <= target <= result.ci_high)
    return covered


@pytest.fixture(scope="module")
def rejection_counts() -> dict[str, int]:
    """Return how many of forty memoryless cohorts each method rejected at 0.05."""
    generator = np.random.default_rng(SEED)
    # The hazard replicates are drawn from a generator of their own. The forty
    # cohorts and the replicates of the other three methods come off `generator`
    # in the order they always did, so the counts they had are unchanged and the
    # method added here reads the same cohorts they do.
    hazard_generator = np.random.default_rng(CALIBRATION_HAZARD_SEED)
    counts = {method: 0 for method in MEMORYLESS_METHODS}
    for _ in range(CALIBRATION_COHORTS):
        values = rounded_exponential(
            PAPER.tau_no_health_cohort_weeks.value,
            PAPER.n_no_health_events.value,
            generator,
        )
        frame = one_run_per_patient(values, RELAPSE)
        counts["hazard"] += int(
            fit.test_memoryless(
                frame, RELAPSE, method="hazard", n_boot=CALIBRATION_BOOT, rng=hazard_generator
            ).reject(0.05)
        )
        for method in BOOTSTRAP_METHODS:
            result = fit.test_memoryless(
                frame,
                RELAPSE,
                method=method,  # type: ignore[arg-type]
                n_boot=CALIBRATION_BOOT,
                rng=generator,
            )
            counts[method] += int(result.reject(0.05))
    return counts


@pytest.fixture(scope="module")
def periodicity_rejections() -> dict[str, int]:
    """Return how many of ten memoryless cohorts each method called periodic."""
    counts = dict.fromkeys(PERIODICITY_METHODS, 0)
    for seed in range(NULL_COHORTS):
        weekly = renewal_weekly(NULL_PATIENTS, NULL_WEEKS, seed)
        for method in PERIODICITY_METHODS:
            result = fit.test_periodicity(
                weekly,
                method=method,  # type: ignore[arg-type]
                n_perm=N_PERM,
                rng=seed,
            )
            counts[method] += int(result.pooled.reject(0.05))
    return counts


@pytest.fixture(scope="module")
def injected_cycle_readings() -> dict[tuple[str, int], fit.PeriodicityResult]:
    """Return the reading of the injected cycle, one per method and fixture seed."""
    return {
        (method, seed): fit.test_periodicity(
            yearly_cycle_weekly(seed=seed),
            method=method,  # type: ignore[arg-type]
            n_perm=N_PERM,
            rng=SEED,
        )
        for method in PERIODICITY_METHODS
        for seed in PERIODICITY_FIXTURE_SEEDS
    }


@pytest.fixture(scope="module")
def ageing_frame() -> pd.DataFrame:
    return one_run_per_patient(ageing_durations(), RELAPSE)


@pytest.fixture(scope="module")
def memoryless_frame() -> pd.DataFrame:
    return one_run_per_patient(memoryless_durations(), RELAPSE)


@pytest.fixture(scope="module")
def cohort_frame() -> pd.DataFrame:
    return cohort_durations()


@pytest.fixture(scope="module")
def result_objects(cohort_frame: pd.DataFrame) -> dict[str, _Result]:
    # One of every result class of this module, each built by the estimator that
    # returns it, so that what is read below is the object a caller gets.
    return {
        "TestResult": fit.test_memoryless(cohort_frame, RELAPSE, method="hazard"),
        "PeriodicityResult": fit.test_periodicity(renewal_weekly(10, 1000.0, SEED)),
        "NBFit": fit.fit_nb_counts(negative_binomial_counts(n=50)),
        "GammaFit": fit.fit_gamma_rates(cohort_frame),
    }


@pytest.fixture(scope="module")
def large_cohort_frame() -> pd.DataFrame:
    # Equation (7) takes the logarithm of a mean relapse of about four weeks, so
    # a one percent wobble in that mean moves the ratio by about 0.02. Thirty
    # patients leave the ratio with a spread of about 0.2, wider than the 0.15
    # this file asks of it, and three hundred bring the spread down to 0.06.
    return cohort_durations(n=300)


# Bounds on how many of the 200 replicates a nominal 95 percent interval may
# cover. At 95 percent the count has a binomial standard error of 3.1, so 182 is
# about three of them below 190 and the run here lands at 192 and 189. The upper
# bound is what makes this a calibration rather than a floor: an interval that
# lost its 1 / sqrt(n) factor, or a bootstrap that came back as (0, inf), would
# cover all 200 and has to fail here too.
COVERAGE_BOUNDS = (182, 199)


def test_fisher_intervals_cover_the_remission_mean(coverage_counts: dict[int, int]) -> None:
    low, high = COVERAGE_BOUNDS
    assert low <= coverage_counts[REMISSION] <= high


def test_fisher_intervals_cover_the_relapse_mean(coverage_counts: dict[int, int]) -> None:
    low, high = COVERAGE_BOUNDS
    assert low <= coverage_counts[RELAPSE] <= high


def test_the_exponential_fisher_interval_is_the_closed_form_of_its_information(
    cohort_frame: pd.DataFrame,
) -> None:
    # The width itself, not only its coverage. An exponential run carries one
    # unit of information on the log rate however long it lasted, so the half
    # width is z / sqrt(n_complete) whatever the censored runs add to the total
    # time, and the interval is that half width either side of the mean on the
    # log scale.
    result = fit.fit_durations(cohort_frame, REMISSION)
    z = float(stats.norm.ppf(0.5 * (1.0 + result.ci_level)))
    half_width = z / math.sqrt(result.n - result.n_censored)
    assert result.n_censored > 0
    assert result.ci_low == pytest.approx(result.mean * math.exp(-half_width), rel=1e-12)
    assert result.ci_high == pytest.approx(result.mean * math.exp(half_width), rel=1e-12)


def test_the_geometric_fisher_interval_is_narrower_by_its_own_information(
    cohort_frame: pd.DataFrame,
) -> None:
    # A geometric week that ends is one binomial success out of the weeks
    # recorded, so the information on the log success probability is
    # n_complete / (1 - p) rather than the n_complete of an exponential sample,
    # and the interval is shorter by sqrt(1 - p). At the four week relapse of the
    # paper the exponential interval is a seventh the wider of the two, which is
    # not a rounding detail.
    result = fit.fit_durations(cohort_frame, RELAPSE, family="geometric")
    exponential = fit.fit_durations(cohort_frame, RELAPSE)
    z = float(stats.norm.ppf(0.5 * (1.0 + result.ci_level)))
    half_width = z * math.sqrt((1.0 - result.rate) / (result.n - result.n_censored))

    assert result.ci_low == pytest.approx(result.mean * math.exp(-half_width), rel=1e-12)
    assert result.ci_high == pytest.approx(result.mean * math.exp(half_width), rel=1e-12)
    assert math.log(result.ci_high / result.ci_low) == pytest.approx(
        math.log(exponential.ci_high / exponential.ci_low) * math.sqrt(1.0 - result.rate),
        rel=1e-12,
    )


def test_the_geometric_interval_of_runs_that_all_ended_at_once_is_degenerate() -> None:
    # Every run ending in its first week puts the success probability at 1, where
    # the information on it is infinite and the Wald half width vanishes. The
    # interval is the point itself, which is what a Wald interval says at the
    # boundary of a parameter rather than a claim that the mean is known.
    frame = one_run_per_patient(np.ones(3, dtype=np.int64), RELAPSE)
    result = fit.fit_durations(frame, RELAPSE, family="geometric")
    assert result.rate == 1.0
    assert (result.ci_low, result.ci_high) == (1.0, 1.0)


def test_censored_remission_mean_is_at_least_the_naive_mean(cohort_frame: pd.DataFrame) -> None:
    censored = fit.fit_durations(cohort_frame, REMISSION, censoring=True)
    naive = fit.fit_durations(cohort_frame, REMISSION, censoring=False)
    assert censored.n_censored > 0
    assert naive.n_censored == 0
    assert censored.mean >= naive.mean


def test_bootstrap_interval_is_named_and_contains_the_point_estimate(
    cohort_frame: pd.DataFrame,
) -> None:
    result = fit.fit_durations(cohort_frame, REMISSION, bootstrap=200, rng=SEED)
    assert result.ci_method == "bootstrap"
    assert result.ci_low <= result.mean <= result.ci_high


def test_a_generator_gives_the_same_bootstrap_interval_as_its_seed(
    cohort_frame: pd.DataFrame,
) -> None:
    from_seed = fit.fit_durations(cohort_frame, REMISSION, bootstrap=50, rng=SEED)
    from_generator = fit.fit_durations(
        cohort_frame, REMISSION, bootstrap=50, rng=np.random.default_rng(SEED)
    )
    assert from_seed.ci_low == from_generator.ci_low
    assert from_seed.ci_high == from_generator.ci_high


def test_default_interval_is_named_after_the_fisher_information(
    cohort_frame: pd.DataFrame,
) -> None:
    result = fit.fit_durations(cohort_frame, REMISSION)
    assert result.ci_method == "fisher"
    assert result.ci_level == 0.95
    assert result.ci_low < result.mean < result.ci_high


def test_geometric_family_gives_the_exponential_mean_without_a_correction(
    cohort_frame: pd.DataFrame,
) -> None:
    exponential = fit.fit_durations(cohort_frame, REMISSION, family="exponential")
    geometric = fit.fit_durations(cohort_frame, REMISSION, family="geometric")
    assert geometric.mean == pytest.approx(exponential.mean)
    assert geometric.family == "geometric"


def test_geometric_fit_of_one_week_runs_is_certain_of_the_next_week() -> None:
    frame = one_run_per_patient(np.ones(3, dtype=np.int64), RELAPSE)
    result = fit.fit_durations(frame, RELAPSE, family="geometric")
    assert result.rate == 1.0
    assert result.mean == 1.0
    assert result.loglik == 0.0


def test_geometric_loglikelihood_counts_a_censored_run_as_a_whole_failure_week() -> None:
    # A complete run of d weeks ended in its d-th week and carries d - 1 weeks
    # in which it did not end; a censored run of d weeks is known only to have
    # lasted at least that long and carries d of them. Here the two remissions
    # hold 30 weeks and one event, so p is 1/30 and there are 29 failure weeks.
    frame = durations_frame(
        {
            "p0001": [(RELAPSE, 4), (REMISSION, 10), (RELAPSE, 4)],
            "p0002": [(RELAPSE, 4), (REMISSION, 20)],
        }
    )
    remissions = frame[frame["state"] == REMISSION]
    weeks = float(remissions["duration_w"].sum())
    complete = float((~remissions["censored"]).sum())
    probability = complete / weeks
    expected = complete * math.log(probability) + (weeks - complete) * math.log1p(-probability)
    result = fit.fit_durations(frame, REMISSION, family="geometric")
    assert result.n_censored == 1
    assert result.rate == pytest.approx(probability, rel=1e-12)
    assert result.loglik == pytest.approx(expected, rel=1e-12)


def test_exponential_loglikelihood_counts_a_censored_run_as_time_without_an_event() -> None:
    # The two remissions hold 30 weeks between them and one of them ended, so the
    # rate is 1/30 and the log likelihood is log(rate) taken once for that event
    # minus the rate over all 30 weeks, the censored 20 included.
    frame = durations_frame(
        {
            "p0001": [(RELAPSE, 4), (REMISSION, 10), (RELAPSE, 4)],
            "p0002": [(RELAPSE, 4), (REMISSION, 20)],
        }
    )
    remissions = frame[frame["state"] == REMISSION]
    weeks = float(remissions["duration_w"].sum())
    complete = float((~remissions["censored"]).sum())
    expected = complete * math.log(complete / weeks) - complete
    result = fit.fit_durations(frame, REMISSION)
    assert result.n_censored == 1
    assert result.mean == pytest.approx(weeks / complete, rel=1e-12)
    assert result.loglik == pytest.approx(expected, rel=1e-12)


def test_geometric_family_rejects_a_continuity_correction(cohort_frame: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="continuity correction"):
        fit.fit_durations(cohort_frame, REMISSION, family="geometric", continuity_correction=0.5)


def test_continuity_correction_shortens_the_mean(cohort_frame: pd.DataFrame) -> None:
    plain = fit.fit_durations(cohort_frame, RELAPSE)
    corrected = fit.fit_durations(cohort_frame, RELAPSE, continuity_correction=0.5)
    assert corrected.mean < plain.mean
    assert corrected.continuity_correction == 0.5


def test_a_censored_run_keeps_its_whole_time_under_a_continuity_correction() -> None:
    # The complete remission of 10 weeks is shortened to 9.5 because it is known
    # to have ended somewhere inside its tenth week. The censored one of 20 weeks
    # is not, because it is known only to have lasted at least that long, so the
    # single event sits on 9.5 + 20 = 29.5 weeks of observed time.
    frame = durations_frame(
        {
            "p0001": [(RELAPSE, 4), (REMISSION, 10), (RELAPSE, 4)],
            "p0002": [(RELAPSE, 4), (REMISSION, 20)],
        }
    )
    result = fit.fit_durations(frame, REMISSION, continuity_correction=0.5)
    assert result.n_censored == 1
    assert result.mean == pytest.approx(29.5)


def test_fit_durations_needs_a_complete_duration() -> None:
    frame = durations_frame({"p0001": [(REMISSION, 12)]})
    with pytest.raises(ValueError, match="no complete"):
        fit.fit_durations(frame, REMISSION)


def test_fit_durations_rejects_an_unknown_family(cohort_frame: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="family"):
        fit.fit_durations(cohort_frame, REMISSION, family="weibull")  # type: ignore[arg-type]


def test_fit_durations_rejects_an_unknown_state(cohort_frame: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="state"):
        fit.fit_durations(cohort_frame, 0)


def test_fit_durations_rejects_a_confidence_level_outside_the_unit_interval(
    cohort_frame: pd.DataFrame,
) -> None:
    with pytest.raises(ValueError, match="ci"):
        fit.fit_durations(cohort_frame, REMISSION, ci=1.0)


def test_fit_durations_rejects_a_negative_bootstrap_count(cohort_frame: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="bootstrap"):
        fit.fit_durations(cohort_frame, REMISSION, bootstrap=-1)


def test_fit_durations_rejects_a_correction_of_a_whole_week(cohort_frame: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="continuity_correction"):
        fit.fit_durations(cohort_frame, REMISSION, continuity_correction=1.0)


def test_a_bootstrap_that_only_ever_resamples_censored_runs_is_refused() -> None:
    # Two remission runs, one of them censored, and a single resample: at seed 0
    # that resample draws the censored run twice and has no event left to fit.
    frame = durations_frame(
        {
            "p0001": [(RELAPSE, 4), (REMISSION, 10), (RELAPSE, 4)],
            "p0002": [(RELAPSE, 4), (REMISSION, 20)],
        }
    )
    with pytest.raises(ValueError, match="only censored runs"):
        fit.fit_durations(frame, REMISSION, bootstrap=1, rng=0)


def test_exponential_loglikelihood_matches_the_closed_form() -> None:
    frame = one_run_per_patient(np.array([1, 2, 3, 4], dtype=np.int64), RELAPSE)
    result = fit.fit_durations(frame, RELAPSE)
    expected = 4.0 * math.log(result.rate) - result.rate * 10.0
    assert result.loglik == pytest.approx(expected)
    assert result.mean == pytest.approx(2.5)


def test_fit_result_repr_names_the_fit_and_cites_the_paper_once() -> None:
    frame = one_run_per_patient(np.array([1, 2, 3, 4], dtype=np.int64), RELAPSE)
    text = repr(fit.fit_durations(frame, RELAPSE))
    assert "exponential" in text
    assert text.count(DOI) == 1
    assert "n=4" in text


@pytest.mark.parametrize("method", MEMORYLESS_METHODS)
def test_ageing_durations_are_not_memoryless(ageing_frame: pd.DataFrame, method: str) -> None:
    result = fit.test_memoryless(ageing_frame, RELAPSE, method=method, rng=SEED)  # type: ignore[arg-type]
    assert result.reject(0.05)


def test_ageing_durations_have_a_rising_hazard(ageing_frame: pd.DataFrame) -> None:
    # Weibull durations of shape 2 age by construction: the longer a run has
    # lasted the likelier it is to end this week. The direction is the whole
    # reading of the test, and a two sided p value alone would not see a sign
    # flip in either the slope or the t statistic built from it.
    result = fit.test_memoryless(ageing_frame, RELAPSE, method="hazard")
    assert result.details["slope"] > 0.0
    assert result.statistic > 0.0


@pytest.mark.parametrize("method", MEMORYLESS_METHODS)
def test_geometric_durations_pass_as_memoryless(
    memoryless_frame: pd.DataFrame, method: str
) -> None:
    result = fit.test_memoryless(memoryless_frame, RELAPSE, method=method, rng=SEED)  # type: ignore[arg-type]
    assert not result.reject(0.05)
    assert result.n == N_DURATIONS


@pytest.mark.parametrize("method", MEMORYLESS_METHODS)
def test_relapses_of_the_length_the_paper_reports_are_not_called_ageing(
    rejection_counts: dict[str, int], method: str
) -> None:
    # Durations drawn as ceil(Exp(4.3)) are memoryless and rounded the way the
    # study rounds, so the only thing a test of memorylessness can find in them
    # is its own level. The replicates it calibrates against must therefore be
    # drawn at the scale of the exponential behind the rounding, not at the mean
    # of the durations after it, which is about half a week longer.
    #
    # The hazard method is read here beside the other three, and it is the one
    # this matters most for: it is the default of test_memoryless and of the
    # command line. Its p value used to come from the least squares fit of the
    # slope, which takes the weekly hazards as equally precise, and that reading
    # called nearly one memoryless cohort in ten ageing at the 0.05 level.
    assert rejection_counts[method] <= CALIBRATION_ALLOWED_REJECTIONS


def test_the_null_scale_reproduces_the_mean_of_the_durations_it_was_read_from() -> None:
    # The null is ceil(Exp(scale)), which lives on whole weeks with success
    # probability 1 - exp(-1 / scale) and therefore mean 1 / that probability.
    # Reading the scale back out of the durations has to return that mean.
    values = np.array([1.0, 1.0, 2.0, 3.0, 5.0, 8.0, 2.0, 1.0, 4.0, 7.0])
    scale = fit._exponential_scale(values)
    assert 1.0 / -math.expm1(-1.0 / scale) == pytest.approx(float(values.mean()), rel=1e-12)


def test_the_null_scale_of_runs_that_all_lasted_one_week_is_degenerate() -> None:
    # Every run ending in its first week leaves a success probability of exactly
    # 1, whose scale is 0: the exponential that is always rounded up to one week.
    assert fit._exponential_scale(np.ones(4)) == 0.0


@pytest.mark.parametrize("method", BOOTSTRAP_METHODS)
def test_runs_that_all_lasted_one_week_still_get_a_p_value(method: str) -> None:
    frame = one_run_per_patient(np.ones(6, dtype=np.int64), RELAPSE)
    result = fit.test_memoryless(frame, RELAPSE, method=method, n_boot=20, rng=SEED)  # type: ignore[arg-type]
    assert 0.0 < result.p_value <= 1.0


def test_cv_details_name_the_null_it_is_compared_with(memoryless_frame: pd.DataFrame) -> None:
    # A rounded exponential of mean m has a coefficient of variation near
    # m / (m + 0.5), not the 1 of an unrounded one, so the null value the sample
    # is judged against is reported rather than assumed.
    result = fit.test_memoryless(memoryless_frame, RELAPSE, method="cv", rng=SEED)
    assert result.details["null_cv"] == pytest.approx(
        MEAN_DURATION / (MEAN_DURATION + 0.5), abs=0.02
    )
    assert result.details["null_cv"] < 1.0


@pytest.mark.parametrize("method", MEMORYLESS_METHODS)
def test_details_name_the_scale_the_replicates_were_drawn_at(
    memoryless_frame: pd.DataFrame, method: str
) -> None:
    result = fit.test_memoryless(memoryless_frame, RELAPSE, method=method, rng=SEED)  # type: ignore[arg-type]
    values = memoryless_frame["duration_w"].to_numpy(dtype=np.float64)
    assert result.details["scale"] == pytest.approx(fit._exponential_scale(values), rel=1e-12)
    assert result.details["scale"] < result.details["mean"]


@pytest.mark.parametrize("method", MEMORYLESS_METHODS)
def test_memorylessness_details_carry_the_sample_mean(
    memoryless_frame: pd.DataFrame, method: str
) -> None:
    result = fit.test_memoryless(memoryless_frame, RELAPSE, method=method, rng=SEED)  # type: ignore[arg-type]
    assert result.details["mean"] == pytest.approx(memoryless_frame["duration_w"].mean(), rel=1e-12)


def test_memorylessness_uses_only_the_complete_durations() -> None:
    runs = {
        "p0001": [(REMISSION, 4), (RELAPSE, 4), (REMISSION, 9), (RELAPSE, 9), (REMISSION, 900)],
        "p0002": [(REMISSION, 4), (RELAPSE, 4), (REMISSION, 9), (RELAPSE, 9), (REMISSION, 900)],
    }
    frame = durations_frame(runs)
    result = fit.test_memoryless(frame, REMISSION, method="cv", n_boot=50, rng=SEED)
    assert result.n == 4
    assert result.details["mean"] == pytest.approx(6.5)


def test_memorylessness_rejects_an_unknown_method(memoryless_frame: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="method"):
        fit.test_memoryless(memoryless_frame, RELAPSE, method="chi2")  # type: ignore[arg-type]


def test_memorylessness_rejects_an_empty_bootstrap(memoryless_frame: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="n_boot"):
        fit.test_memoryless(memoryless_frame, RELAPSE, method="cv", n_boot=0)


def test_memorylessness_needs_two_complete_durations() -> None:
    frame = durations_frame({"p0001": [(RELAPSE, 4)]})
    with pytest.raises(ValueError, match="at least two complete durations"):
        fit.test_memoryless(frame, RELAPSE, method="cv")


def test_hazard_test_refuses_a_hazard_that_is_exactly_a_straight_line() -> None:
    # Sixteen durations of one week, eight of two, four of three and four of
    # four leave a hazard of exactly one half at every week with five or more
    # still at risk, which no regression can put a standard error on.
    values = np.repeat([1, 2, 3, 4], [16, 8, 4, 4]).astype(np.int64)
    frame = one_run_per_patient(values, RELAPSE)
    with pytest.raises(ValueError, match="exactly on the straight line"):
        fit.test_memoryless(frame, RELAPSE, method="hazard")


def test_hazard_test_needs_enough_times_at_risk() -> None:
    frame = one_run_per_patient(np.array([1, 1, 2, 2, 3, 3], dtype=np.int64), RELAPSE)
    with pytest.raises(ValueError, match="at risk"):
        fit.test_memoryless(frame, RELAPSE, method="hazard")


def test_the_hazard_p_value_counts_replicates_rather_than_reading_the_regression(
    memoryless_frame: pd.DataFrame,
) -> None:
    # The least squares fit takes every weekly hazard as equally precise, while
    # the variance of one is h (1 - h) / at_risk and grows as the at risk set
    # empties, so the p value the fit prints is too small. What the test reports
    # is the share of replicates of the rounded exponential null whose drift is
    # at least as far from zero, so it lands on a multiple of 1 / (n_boot + 1)
    # and is never exactly 0. The statistic stays the slope over its standard
    # error, signed, which is what the survival inset is drawn against.
    result = fit.test_memoryless(memoryless_frame, RELAPSE, method="hazard", n_boot=100, rng=SEED)
    values = memoryless_frame["duration_w"].to_numpy(dtype=np.float64)
    times, hazard = fit._hazard_points(values)
    line = stats.linregress(times, hazard)

    assert result.details["n_boot"] == 100.0
    assert result.details["n_times"] == float(times.size)
    assert result.p_value >= 1.0 / 101.0
    assert result.p_value * 101.0 == pytest.approx(round(result.p_value * 101.0))
    assert result.p_value != pytest.approx(float(line.pvalue))
    assert result.statistic == pytest.approx(float(line.slope) / float(line.stderr))
    assert result.details["slope"] == pytest.approx(float(line.slope))
    assert result.details["intercept"] == pytest.approx(float(line.intercept))


def test_a_generator_gives_the_same_hazard_p_value_as_its_seed(
    memoryless_frame: pd.DataFrame,
) -> None:
    from_seed = fit.test_memoryless(memoryless_frame, RELAPSE, method="hazard", n_boot=50, rng=SEED)
    from_generator = fit.test_memoryless(
        memoryless_frame,
        RELAPSE,
        method="hazard",
        n_boot=50,
        rng=np.random.default_rng(SEED),
    )
    assert from_seed.p_value == from_generator.p_value
    assert from_seed.statistic == from_generator.statistic


@pytest.mark.slow
@pytest.mark.parametrize("method", PERIODICITY_METHODS)
def test_memoryless_cohorts_are_rarely_called_periodic(
    periodicity_rejections: dict[str, int], method: str
) -> None:
    # The size of the test, measured on ten cohorts of the shape of the study.
    # Relapse onsets of a memoryless record are a renewal process with a flat
    # spectrum, which is the null both methods are built against, so a cohort of
    # them may not be called periodic more often than the level allows.
    assert periodicity_rejections[method] <= ALLOWED_PERIODICITY_REJECTIONS


@pytest.mark.parametrize("method", PERIODICITY_METHODS)
def test_a_memoryless_cohort_shows_no_periodicity(method: str) -> None:
    # Twenty memoryless records of ten years, relapses of about four weeks and
    # all. Read on the state series the g test called this very cohort periodic,
    # which is the reading this module no longer takes.
    result = fit.test_periodicity(
        renewal_weekly(PERIODICITY_PATIENTS, float(RECORD_WEEKS), SEED),
        method=method,  # type: ignore[arg-type]
        n_perm=N_PERM,
        rng=SEED,
    )
    assert result.pooled.p_value > 0.05


@pytest.mark.parametrize("seed", PERIODICITY_FIXTURE_SEEDS)
@pytest.mark.parametrize("method", PERIODICITY_METHODS)
def test_an_injected_yearly_cycle_is_detected(
    injected_cycle_readings: dict[tuple[str, int], fit.PeriodicityResult],
    method: str,
    seed: int,
) -> None:
    result = injected_cycle_readings[(method, seed)]
    assert result.pooled.n == PERIODICITY_PATIENTS
    assert result.pooled.p_value < 0.001


@pytest.mark.parametrize("seed", PERIODICITY_FIXTURE_SEEDS)
@pytest.mark.parametrize("method", PERIODICITY_METHODS)
def test_an_injected_yearly_cycle_reports_a_period_of_about_a_year(
    injected_cycle_readings: dict[tuple[str, int], fit.PeriodicityResult],
    method: str,
    seed: int,
) -> None:
    result = injected_cycle_readings[(method, seed)]
    periods = result.per_patient["period_weeks"].to_numpy(dtype=float)
    close = np.abs(periods - CYCLE_WEEKS) <= PERIOD_TOLERANCE * CYCLE_WEEKS
    assert int(np.count_nonzero(close)) >= PATIENTS_WITH_THE_RIGHT_PERIOD


@pytest.mark.parametrize("method", PERIODICITY_METHODS)
def test_a_patient_with_two_onsets_is_skipped_and_counted(method: str) -> None:
    # Two onsets leave one gap, which says nothing about a rhythm: any two
    # relapses are one cycle apart, whatever the cycle.
    records = {
        "p0001": record_with_onsets([10, 60], RECORD_WEEKS),
        "p0002": yearly_cycle_weekly(n_patients=1)["state"].to_numpy(dtype=np.int64),
    }
    result = fit.test_periodicity(
        weekly_frame(records),
        method=method,  # type: ignore[arg-type]
        n_perm=N_PERM,
        rng=SEED,
    )
    assert result.pooled.details["n_skipped_few_onsets"] == 1.0
    assert result.pooled.n == 1
    skipped = result.per_patient[result.per_patient["patient_id"] == "p0001"].iloc[0]
    assert int(skipped["n_onsets"]) == 2
    assert math.isnan(float(skipped["p_value"]))
    assert math.isnan(float(skipped["statistic"]))
    assert math.isnan(float(skipped["period_weeks"]))


@pytest.mark.parametrize("method", PERIODICITY_METHODS)
def test_a_cohort_in_which_every_patient_is_skipped_is_refused(method: str) -> None:
    records = {
        "p0001": np.full(4, REMISSION, dtype=np.int64),
        "p0002": record_with_onsets([10, 60], RECORD_WEEKS),
    }
    with pytest.raises(ValueError, match="no patient has a record this test can read"):
        fit.test_periodicity(
            weekly_frame(records),
            method=method,  # type: ignore[arg-type]
            n_perm=N_PERM,
            rng=SEED,
        )


def test_short_records_are_skipped_and_counted() -> None:
    cycle = yearly_cycle_weekly(n_patients=1)["state"].to_numpy(dtype=np.int64)
    records = {
        "p0001": np.full(4, REMISSION, dtype=np.int64),
        "p0002": cycle,
        "p0003": cycle,
    }
    result = fit.test_periodicity(weekly_frame(records))
    assert result.pooled.details["n_skipped_short"] == 1.0
    assert result.pooled.n == 2
    short = result.per_patient[result.per_patient["patient_id"] == "p0001"].iloc[0]
    assert math.isnan(float(short["p_value"]))
    assert int(short["n_weeks"]) == 4


def test_the_per_patient_table_counts_the_onsets_it_read() -> None:
    weekly = yearly_cycle_weekly(n_patients=2)
    result = fit.test_periodicity(weekly)
    counted = result.per_patient.set_index("patient_id")["n_onsets"]
    assert int(counted.loc["p0001"]) == len(onset_weeks(weekly, "p0001"))
    assert int(counted.loc["p0002"]) == len(onset_weeks(weekly, "p0002"))


def test_the_onset_of_a_record_that_opens_in_relapse_is_counted() -> None:
    # Week 0 has no week before it to have been a remission, and the study's
    # records start at the onset of a relapse, so a record that opens in relapse
    # opens on an onset.
    records = {"p0001": record_with_onsets([0, 20, 40], RECORD_WEEKS)}
    result = fit.test_periodicity(weekly_frame(records))
    assert int(result.per_patient["n_onsets"].iloc[0]) == 3


def test_the_onset_series_removes_the_red_spectrum_of_long_relapses() -> None:
    # The flaw the onset series exists to remove, on a record of three relapses
    # of thirty weeks each. Reading the state series, the g test rejects at 1e-29
    # and names a period of 173 weeks, which is only the spacing of three long
    # stretches in one state; reading the onsets, three impulses, it sees no
    # rhythm at all. A relapse counts once however long it goes on.
    series = np.full(RECORD_WEEKS, REMISSION, dtype=np.int64)
    for start in (10, 200, 400):
        series[start : start + 30] = RELAPSE
    from_states = fit._fisher_g(series.astype(np.float64))
    assert from_states is not None
    assert from_states[1] < 0.001
    result = fit.test_periodicity(weekly_frame({"p0001": series}))
    assert int(result.per_patient["n_onsets"].iloc[0]) == 3
    assert result.pooled.p_value > 0.05


def test_pooled_result_is_fishers_combination_of_the_per_patient_p_values() -> None:
    result = fit.test_periodicity(renewal_weekly(6, float(RECORD_WEEKS), SEED))
    per_patient = result.per_patient["p_value"].dropna().to_numpy(dtype=float)
    assert per_patient.size == result.pooled.n
    statistic = -2.0 * float(np.sum(np.log(per_patient)))
    assert result.pooled.statistic == pytest.approx(statistic, rel=1e-12)
    assert result.pooled.p_value == pytest.approx(
        float(stats.chi2.sf(statistic, 2 * per_patient.size)), rel=1e-12
    )


def test_fisher_g_leaves_the_nyquist_ordinate_out() -> None:
    # The Nyquist ordinate of an even length record is real rather than complex,
    # so it is not exchangeable with the others and the finite sum behind the
    # exact p value does not describe it. This series carries most of its power
    # there, which makes the two readings differ by a wide margin.
    series = np.array([1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0])
    power = np.abs(np.fft.rfft(series - series.mean())) ** 2
    read = fit._fisher_g(series)
    assert read is not None
    assert read[0] == pytest.approx(float(power[1:5].max() / power[1:5].sum()), rel=1e-12)
    # The largest ordinate of this series is the Nyquist one, so keeping it would
    # also report the two week period that goes with it instead of the 2.5 weeks
    # of the largest ordinate the test does read.
    assert int(np.argmax(power[1:])) == 4
    assert read[2] == pytest.approx(2.5, rel=1e-12)


@pytest.mark.parametrize("weeks", WHITE_NOISE_WEEKS)
def test_white_noise_series_of_even_length_are_not_over_rejected(weeks: int) -> None:
    # Keeping the Nyquist ordinate inflates the exact p value of the g test by
    # half again at the eight week minimum: 0.079 of white noise series are
    # called periodic at the 0.05 level instead of 0.05 of them.
    #
    # The three lengths are read because the exact p value is a finite
    # alternating sum that is cut as soon as a term stops falling, and then
    # clipped to [0, 1]. Neither the cut nor the clip bites at the eight week
    # minimum, where there are three ordinates, and both of them bite at the
    # hundreds of ordinates a real record carries: none of the p values at eight
    # weeks come back as exactly 1, against 0.045 of them at 200 weeks and 0.11
    # at 1000. The rejection rates measured here are 0.049, 0.055 and 0.053, so
    # the bound is about four binomial standard errors above the level of the
    # test at each length.
    generator = np.random.default_rng(SEED)
    rejected = 0
    trials = 3000
    for _ in range(trials):
        read = fit._fisher_g(generator.normal(size=weeks))
        assert read is not None
        rejected += int(read[1] <= 0.05)
    assert rejected / trials <= 0.065


def test_a_record_that_relapses_every_other_week_is_skipped_by_fisher_g() -> None:
    # Its onsets fall in every other week, so all the power of the onset series
    # sits at the Nyquist frequency, which the g test does not read, and there is
    # no ordinate left to compare.
    records = {
        "p0001": record_with_onsets(list(range(0, RECORD_WEEKS, 2)), RECORD_WEEKS),
        "p0002": yearly_cycle_weekly(n_patients=1)["state"].to_numpy(dtype=np.int64),
    }
    result = fit.test_periodicity(weekly_frame(records))
    assert result.pooled.details["n_skipped_flat_spectrum"] == 1.0
    assert result.pooled.n == 1


def test_periodicity_rejects_an_unknown_method() -> None:
    with pytest.raises(ValueError, match="method"):
        fit.test_periodicity(yearly_cycle_weekly(n_patients=2), method="welch")  # type: ignore[arg-type]


def test_periodicity_rejects_an_empty_permutation_count() -> None:
    with pytest.raises(ValueError, match="n_perm"):
        fit.test_periodicity(yearly_cycle_weekly(n_patients=2), method="lombscargle", n_perm=0)


def test_periodicity_needs_a_long_enough_record() -> None:
    records = {"p0001": np.full(4, REMISSION, dtype=np.int64)}
    with pytest.raises(ValueError, match="no patient"):
        fit.test_periodicity(weekly_frame(records))


def test_poisson_counts_show_no_overdispersion() -> None:
    counts = np.random.default_rng(SEED).poisson(3.0, 2000)
    result = fit.fit_nb_counts(counts)
    assert result.p_value > 0.05
    assert result.dispersion < 0.05
    assert result.mean == float(counts.mean())


def test_negative_binomial_mean_is_the_exact_sample_mean() -> None:
    # The score of an intercept only NB2 fit in the log mean is
    # sum((y - m) / (1 + a m)), whose root is the sample mean at every
    # dispersion, so the maximiser is known in closed form and the search adds
    # nothing to it but noise and a dependence on the release of the optimiser.
    counts = negative_binomial_counts()
    result = fit.fit_nb_counts(counts)
    assert result.dispersion > 0.0
    assert result.mean == float(counts.mean())


def test_negative_binomial_counts_recover_the_dispersion() -> None:
    result = fit.fit_nb_counts(negative_binomial_counts())
    assert result.dispersion == pytest.approx(NB_DISPERSION, rel=0.15)
    assert result.p_value < 0.001
    assert result.dispersion_ci[0] < result.dispersion < result.dispersion_ci[1]
    assert result.p_value == pytest.approx(0.5 * float(stats.chi2.sf(result.lrt_statistic, 1)))


def test_negative_binomial_poisson_loglikelihood_is_the_one_of_the_sample_mean() -> None:
    counts = negative_binomial_counts()
    result = fit.fit_nb_counts(counts)
    expected = float(np.sum(stats.poisson.logpmf(counts, counts.mean())))
    assert result.loglik_poisson == pytest.approx(expected, rel=1e-12)


def test_negative_binomial_interval_is_as_wide_as_the_information_allows() -> None:
    # The observed information of these 2000 counts puts the standard error of
    # the log dispersion at 0.0587, so the 95 percent interval spans about 0.11.
    # The inverse Hessian the quasi-Newton search happens to end on says 0.0115
    # and would give an interval five times too narrow, which a containment
    # assertion of its own cannot see, the interval being centred on the estimate.
    result = fit.fit_nb_counts(negative_binomial_counts())
    assert result.dispersion_ci[1] - result.dispersion_ci[0] > 0.08


@pytest.mark.parametrize("counts", EDGE_COUNTS)
def test_a_cohort_on_the_edge_of_overdispersion_is_fitted_as_poisson(
    counts: tuple[int, ...],
) -> None:
    values = np.array(counts, dtype=np.int64)
    assert float(values.var(ddof=1)) > float(values.mean()) > float(values.var())
    result = fit.fit_nb_counts(values)
    assert result.dispersion == 0.0
    assert result.dispersion_ci == (0.0, 0.0)
    assert result.p_value == 1.0
    assert result.mean == pytest.approx(float(values.mean()))


def test_counts_whose_population_variance_equals_the_mean_are_fitted_as_poisson() -> None:
    # The smallest cohort that tells the two variances apart: a mean of 1, a
    # population variance of 1 and a sample variance of 2.
    result = fit.fit_nb_counts(np.array([0, 2], dtype=np.int64))
    assert result.dispersion == 0.0
    assert result.dispersion_ci == (0.0, 0.0)
    assert result.p_value == 1.0


def test_no_poisson_cohort_of_the_size_of_the_study_fails_to_fit() -> None:
    # Four hundred cohorts of thirty patients at three relapses each. A few of
    # them land in the band where the sample variance exceeds the mean and the
    # population variance does not, and the fit has to read those as Poisson
    # rather than walk the dispersion down to the boundary and try to invert a
    # curvature that is not there.
    generator = np.random.default_rng(SEED)
    results = [fit.fit_nb_counts(generator.poisson(NB_MEAN, 30)) for _ in range(400)]
    assert all(math.isfinite(result.dispersion) for result in results)
    assert all(
        result.dispersion_ci[0] <= result.dispersion <= result.dispersion_ci[1]
        for result in results
    )
    assert any(result.dispersion > 0.0 for result in results)


def test_a_barely_identified_dispersion_does_not_break_the_fit() -> None:
    # The population variance of these seventeen counts exceeds their mean by so
    # little that the maximiser sits just inside the boundary. The log scale then
    # leaves the dispersion with a Wald half width of several hundred, which has
    # no exponential in float64, and the interval saturates at (0, inf) rather
    # than taking a fit that succeeded down with an OverflowError. Whether such a
    # cohort settles just inside the boundary or on it is the optimiser's
    # business; what it may not do is raise, and the likelihood ratio statistic
    # reports the same absence of overdispersion either way.
    counts = np.array([2, 3, 6, 6, 7, 8, 9, 9, 9, 9, 10, 10, 10, 11, 12, 12, 13], dtype=np.int64)
    assert float(counts.var()) > float(counts.mean())
    result = fit.fit_nb_counts(counts)
    assert result.dispersion_ci[0] <= result.dispersion <= result.dispersion_ci[1]
    assert result.lrt_statistic < 1e-3
    assert result.p_value >= 0.49
    assert result.mean == pytest.approx(float(counts.mean()), rel=1e-6)


def test_a_cohort_with_no_relapse_is_not_overdispersed() -> None:
    result = fit.fit_nb_counts(np.zeros(50, dtype=np.int64))
    assert result.mean == 0.0
    assert result.dispersion == 0.0
    assert result.dispersion_ci == (0.0, 0.0)
    assert result.p_value == 1.0
    assert result.loglik_nb == 0.0
    assert result.loglik_poisson == 0.0


def test_a_likelihood_ratio_statistic_of_zero_is_the_point_mass_of_the_mixture() -> None:
    # The half and half mixture of the boundary null puts half of its mass on a
    # statistic of exactly zero, so nothing at all is ruled out there. Reached
    # through the private function because an ordinary sample never lands on it.
    assert fit._boundary_p_value(0.0) == 1.0
    assert fit._boundary_p_value(3.0) == pytest.approx(0.5 * float(stats.chi2.sf(3.0, 1)))


def test_a_likelihood_flat_in_the_dispersion_has_no_standard_error() -> None:
    # Beyond the bound on the log dispersion the likelihood is clipped and
    # therefore flat, so the observed information carries no curvature in that
    # direction and inverting it leaves the log dispersion with a variance that
    # is not positive. There is then no interval to report, and the caller reads
    # that as the Poisson fit. Reached through the private function: a cohort
    # whose dispersion sits on the boundary is screened off well before here.
    values = np.arange(5.0)
    assert fit._log_dispersion_standard_error(np.array([math.log(2.0), -60.0]), values) is None


def test_an_interval_too_wide_for_float64_saturates_rather_than_overflowing() -> None:
    # A half width of several hundred on the log scale says the parameter is not
    # pinned down at all, and has no exponential in float64. Reported as (0, inf)
    # rather than raising an OverflowError out of a fit that otherwise succeeded.
    assert fit._log_scale_interval(0.5, 800.0) == (0.0, math.inf)
    assert fit._log_scale_interval(2.0, 1.0) == pytest.approx(
        (2.0 * math.exp(-1.0), 2.0 * math.exp(1.0))
    )


def test_underdispersed_counts_get_a_degenerate_interval() -> None:
    counts = pd.Series(np.tile([2, 3], 50), name="relapses")
    result = fit.fit_nb_counts(counts)
    assert result.dispersion == 0.0
    assert result.dispersion_ci == (0.0, 0.0)
    assert result.p_value == 1.0
    assert result.loglik_nb == result.loglik_poisson


def test_nb_counts_reject_a_negative_count() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        fit.fit_nb_counts(np.array([1, -1, 2]))


def test_nb_counts_reject_a_fractional_count() -> None:
    with pytest.raises(ValueError, match="whole"):
        fit.fit_nb_counts(np.array([1.0, 1.5, 2.0]))


def test_nb_counts_reject_a_frame_of_counts() -> None:
    with pytest.raises(ValueError, match="one count per patient"):
        fit.fit_nb_counts(np.zeros((2, 2)))


def test_nb_counts_reject_a_single_count() -> None:
    with pytest.raises(ValueError, match="at least two counts"):
        fit.fit_nb_counts(np.array([3]))


def test_nb_counts_reject_a_missing_count() -> None:
    with pytest.raises(ValueError, match="finite"):
        fit.fit_nb_counts(np.array([1.0, math.nan, 2.0]))


@pytest.mark.parametrize("largest", [1e300, 1e160])
def test_nb_counts_reject_a_count_too_large_to_square(largest: float) -> None:
    # Whole and non-negative but far beyond any relapse count: the variance and
    # the likelihood both square it, and the answer leaves float64. The failure
    # is named where the input is described rather than escaping as a bare
    # arithmetic error out of a private function.
    with pytest.raises(ValueError, match="sum of the squares"):
        fit.fit_nb_counts(np.array([largest, 1.0, 2.0]))


def test_the_count_likelihood_turns_an_absurd_log_mean_back() -> None:
    # A line search that probed a log mean of several thousand used to meet a
    # bare OverflowError from inside the likelihood. Both coordinates are now
    # bounded, so the likelihood is flat out there and its gradient finite,
    # which is what tells a search to turn back.
    values = np.arange(5.0)
    absurd = np.array([5000.0, 0.0])
    assert math.isfinite(fit._nb_negative_loglik(absurd, values))
    assert np.all(np.isfinite(fit._nb_negative_score(absurd, values)))
    assert fit._nb_negative_loglik(absurd, values) == fit._nb_negative_loglik(
        np.array([6000.0, 0.0]), values
    )


def test_gamma_shape_is_recovered_by_the_noisy_plug_in_within_thirty_percent() -> None:
    shape, scale, n_patients = 2.0, 0.005, 300
    rates = gamma_rates(shape, scale, n_patients, rng=SEED)
    events = alternating_renewal(
        rates, COHORT_MU, 400.0, n=n_patients, rng=SEED + 1, discretise="week"
    )
    durations = io.weekly_to_durations(io.events_to_weekly(events))
    result = fit.fit_gamma_rates(durations)
    assert result.k == pytest.approx(shape, rel=0.3)
    # A gamma maximum likelihood fit with the location held at zero matches the
    # sample mean exactly, so k theta pins the shape and the scale against each
    # other and would catch a swap or an inverted scale in the unpacking.
    assert result.k * result.theta == pytest.approx(
        float(result.per_patient_rates.mean()), rel=1e-9
    )
    assert np.all(result.per_patient_rates.to_numpy() > 0.0)
    # Only the patients whose remissions ended in a relapse carry a rate, and
    # the rest are dropped, so the fit is over fewer than the 300 patients.
    remissions = durations[durations["state"] == REMISSION]
    complete = remissions[~remissions["censored"]]
    assert result.n == complete["patient_id"].nunique()
    assert result.n < n_patients


def test_gamma_rates_need_a_remission() -> None:
    frame = durations_frame({"p0001": [(RELAPSE, 3)]})
    with pytest.raises(ValueError, match="remission"):
        fit.fit_gamma_rates(frame)


def test_gamma_rates_need_two_patients_with_a_positive_rate() -> None:
    frame = durations_frame(
        {
            "p0001": [(RELAPSE, 3), (REMISSION, 50), (RELAPSE, 2)],
            "p0002": [(RELAPSE, 3), (REMISSION, 50)],
        }
    )
    with pytest.raises(ValueError, match="positive onset rate"):
        fit.fit_gamma_rates(frame)


def test_barrier_ratio_of_exact_means_matches_equation_seven() -> None:
    relapses = [1, 2, 3, 4, 5, 6, 4, 6, 6, 6]
    runs = [(RELAPSE, weeks) for weeks in relapses]
    interleaved: list[tuple[int, int]] = []
    for index, run in enumerate(runs):
        interleaved.append(run)
        if index < len(runs) - 1:
            interleaved.append((REMISSION, 100))
    frame = durations_frame({"p0001": interleaved})
    assert float(np.mean(relapses)) == 4.3
    assert fit.barrier_ratio(frame) == pytest.approx(3.157221141, abs=1e-9)


def test_cohort_barrier_ratio_needs_both_states() -> None:
    frame = durations_frame({"p0001": [(RELAPSE, 4)]})
    with pytest.raises(ValueError, match="no run in state"):
        fit.barrier_ratio(frame)


def test_per_patient_barrier_ratio_is_missing_for_a_patient_without_remissions() -> None:
    frame = durations_frame(
        {
            "p0001": [(RELAPSE, 4), (REMISSION, 100), (RELAPSE, 5)],
            "p0002": [(RELAPSE, 4)],
        }
    )
    ratios = fit.barrier_ratio(frame, per_patient=True)
    assert math.isnan(float(ratios.loc["p0002"]))
    assert float(ratios.loc["p0001"]) == pytest.approx(math.log(100.0) / math.log(4.5), abs=1e-12)


def test_per_patient_barrier_ratio_is_missing_for_a_one_week_mean() -> None:
    frame = durations_frame({"p0001": [(RELAPSE, 1), (REMISSION, 100), (RELAPSE, 1)]})
    ratios = fit.barrier_ratio(frame, per_patient=True)
    assert math.isnan(float(ratios.loc["p0001"]))


def test_cohort_barrier_ratio_is_near_the_published_value(
    large_cohort_frame: pd.DataFrame,
) -> None:
    assert fit.barrier_ratio(large_cohort_frame) == pytest.approx(
        PAPER.barrier_ratio_cohort.value, abs=0.15
    )


def test_test_result_rejects_below_the_level() -> None:
    result = fit.TestResult(method="cv", statistic=2.0, p_value=0.02, n=10, details={})
    assert result.reject()
    assert not result.reject(0.01)


def test_test_result_rejects_an_impossible_level() -> None:
    result = fit.TestResult(method="cv", statistic=2.0, p_value=0.02, n=10, details={})
    with pytest.raises(ValueError, match="alpha"):
        result.reject(0.0)


def test_citation_quotes_the_doi_in_the_text_and_in_the_bibtex() -> None:
    assert _citation.citation().count(DOI) == 2


def test_citation_names_the_package() -> None:
    assert "msrelapse" in _citation.citation()


def test_citation_avoids_the_em_dash() -> None:
    assert chr(0x2014) not in _citation.citation()


def test_short_citation_carries_the_doi() -> None:
    assert _citation.short_citation().endswith(DOI)


def test_fit_result_citation_is_the_short_citation() -> None:
    frame = one_run_per_patient(np.array([1, 2, 3, 4], dtype=np.int64), RELAPSE)
    assert fit.fit_durations(frame, RELAPSE).citation == _citation.short_citation()


@pytest.mark.parametrize("name", RESULT_NAMES)
def test_a_result_carries_the_citation(result_objects: dict[str, _Result], name: str) -> None:
    assert DOI in result_objects[name].citation


@pytest.mark.parametrize("name", RESULT_NAMES)
def test_a_result_repr_cites_the_paper_once(
    result_objects: dict[str, _Result],
    name: str,
) -> None:
    assert repr(result_objects[name]).count(DOI) == 1


@pytest.mark.parametrize("name", RESULT_NAMES)
def test_a_result_repr_names_its_own_fields(
    result_objects: dict[str, _Result],
    name: str,
) -> None:
    assert RESULT_FIELDS[name] in repr(result_objects[name])


def test_software_bibtex_marks_the_doi_as_a_placeholder() -> None:
    lines = _citation.SOFTWARE_BIBTEX.splitlines()
    assert "10.5281/zenodo.XXXXXXX" in _citation.SOFTWARE_BIBTEX
    assert lines[0].startswith("%")
    assert "placeholder" in lines[0]


def test_software_bibtex_keeps_its_comment_outside_the_entry() -> None:
    # A per cent sign is a LaTeX comment, not a BibTeX one: inside an entry
    # BibTeX expects a field name, so a note written there stops the entry
    # parsing. Above the entry the note is ignored and still travels with it.
    lines = _citation.SOFTWARE_BIBTEX.splitlines()
    assert lines[1].startswith("@software{")
    assert "%" not in "\n".join(lines[1:])


def test_version_of_an_uninstalled_source_tree_is_named_rather_than_raised(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def absent(name: str) -> str:
        raise metadata.PackageNotFoundError(name)

    monkeypatch.setattr(metadata, "version", absent)
    assert _citation._software_version() == "0+unknown"


def test_docstring_examples_run() -> None:
    for module in (fit, _citation):
        results = doctest.testmod(module)
        assert results.attempted > 0
        assert results.failed == 0
