from __future__ import annotations

import math

import numpy as np
import numpy.typing as npt
import pandas as pd
import pytest

from msrelapse._params import PAPER
from msrelapse.io import weekly_to_events
from msrelapse.renewal import (
    alternating_renewal,
    effective_onset_rate,
    gamma_rates,
    nb_from_gamma,
    rates_from_means,
    relapse_counts,
    relapse_free,
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

# A relapse rate this large makes relapses effectively instantaneous, so the
# onsets of the alternating process are those of a Poisson process of rate
# lambda and the count statistics can be compared with the exact Poisson and
# negative binomial results without a correction for the time spent in relapse.
BRIEF_RELAPSE_RATE = 1.0e6


@pytest.fixture(scope="module")
def cohort_events() -> pd.DataFrame:
    return alternating_renewal(COHORT_LAMBDA, COHORT_MU, 600.0, n=25, rng=11)


def remission_durations(events: pd.DataFrame) -> npt.NDArray[np.float64]:
    """Return the gaps from the end of one relapse to the next onset."""
    gaps: list[npt.NDArray[np.float64]] = []
    for _, group in events.groupby("patient_id", sort=True):
        onset = group["relapse_onset"].to_numpy(dtype=float)
        end = group["relapse_end"].to_numpy(dtype=float)
        if onset.size > 1:
            gaps.append(onset[1:] - end[:-1])
    return np.concatenate(gaps)


def complete_relapse_durations(events: pd.DataFrame) -> npt.NDArray[np.float64]:
    """Return the durations of the relapses that end before follow up does."""
    complete = events[events["relapse_end"] < events["followup_end"]]
    durations = complete["relapse_end"] - complete["relapse_onset"]
    return durations.to_numpy(dtype=float)


def example_events() -> pd.DataFrame:
    """Return a small hand built events table with three patients."""
    return pd.DataFrame(
        {
            "patient_id": ["p0001", "p0001", "p0002", "p0003"],
            "followup_start": [0.0, 0.0, 0.0, 0.0],
            "followup_end": [100.0, 100.0, 100.0, 100.0],
            "relapse_onset": [1.0, 60.0, 5.0, math.nan],
            "relapse_end": [3.0, 62.0, 6.0, math.nan],
        }
    )


def test_rates_are_the_reciprocals_of_the_mean_durations() -> None:
    lam, mu = rates_from_means(100.0, 4.0)
    assert (lam, mu) == pytest.approx((0.01, 0.25))


@pytest.mark.parametrize(("tau_health", "tau_relapse"), [(0.0, 4.0), (100.0, -1.0)])
def test_rates_from_means_rejects_non_positive_means(tau_health: float, tau_relapse: float) -> None:
    with pytest.raises(ValueError, match="must be positive"):
        rates_from_means(tau_health, tau_relapse)


def test_events_table_has_exactly_the_schema_columns(cohort_events: pd.DataFrame) -> None:
    assert tuple(cohort_events.columns) == EVENTS_COLUMNS


def test_patient_id_is_a_string_column(cohort_events: pd.DataFrame) -> None:
    assert pd.api.types.is_string_dtype(cohort_events["patient_id"])


@pytest.mark.parametrize("column", EVENTS_COLUMNS[1:])
def test_time_columns_are_float64(cohort_events: pd.DataFrame, column: str) -> None:
    assert cohort_events[column].dtype == np.float64


def test_patient_ids_are_zero_padded_to_four_digits() -> None:
    events = alternating_renewal(COHORT_LAMBDA, COHORT_MU, 50.0, n=3, rng=0)
    assert sorted(set(events["patient_id"])) == ["p0001", "p0002", "p0003"]


def test_patient_ids_widen_when_the_cohort_needs_more_digits() -> None:
    events = alternating_renewal(COHORT_LAMBDA, COHORT_MU, 5.0, n=10000, rng=0)
    assert sorted(set(events["patient_id"]))[0] == "p00001"


def test_rows_are_sorted_by_patient_then_onset(cohort_events: pd.DataFrame) -> None:
    expected = cohort_events.sort_values(["patient_id", "relapse_onset"], kind="stable")
    pd.testing.assert_frame_equal(cohort_events, expected)


def test_onsets_strictly_increase_within_a_patient(cohort_events: pd.DataFrame) -> None:
    for _, group in cohort_events.groupby("patient_id", sort=True):
        onset = group["relapse_onset"].to_numpy(dtype=float)
        assert np.all(np.diff(onset) > 0.0)


def test_every_relapse_ends_after_it_starts(cohort_events: pd.DataFrame) -> None:
    assert np.all(cohort_events["relapse_end"] > cohort_events["relapse_onset"])


def test_every_relapse_lies_inside_the_follow_up_window(cohort_events: pd.DataFrame) -> None:
    assert np.all(cohort_events["relapse_onset"] >= cohort_events["followup_start"])
    assert np.all(cohort_events["relapse_end"] <= cohort_events["followup_end"])


def test_relapses_of_one_patient_do_not_overlap(cohort_events: pd.DataFrame) -> None:
    for _, group in cohort_events.groupby("patient_id", sort=True):
        onset = group["relapse_onset"].to_numpy(dtype=float)
        end = group["relapse_end"].to_numpy(dtype=float)
        assert np.all(onset[1:] >= end[:-1])


def test_follow_up_runs_from_zero_to_t_end(cohort_events: pd.DataFrame) -> None:
    assert np.all(cohort_events["followup_start"] == 0.0)
    assert np.all(cohort_events["followup_end"] == 600.0)


def test_the_same_seed_gives_the_same_table() -> None:
    first = alternating_renewal(COHORT_LAMBDA, COHORT_MU, 400.0, n=8, rng=123)
    second = alternating_renewal(COHORT_LAMBDA, COHORT_MU, 400.0, n=8, rng=123)
    pd.testing.assert_frame_equal(first, second)


def test_different_seeds_give_different_tables() -> None:
    first = alternating_renewal(COHORT_LAMBDA, COHORT_MU, 400.0, n=8, rng=123)
    second = alternating_renewal(COHORT_LAMBDA, COHORT_MU, 400.0, n=8, rng=456)
    assert not first["relapse_onset"].equals(second["relapse_onset"])


def test_a_generator_gives_the_same_table_as_its_seed() -> None:
    from_seed = alternating_renewal(COHORT_LAMBDA, COHORT_MU, 400.0, n=8, rng=99)
    from_generator = alternating_renewal(
        COHORT_LAMBDA, COHORT_MU, 400.0, n=8, rng=np.random.default_rng(99)
    )
    pd.testing.assert_frame_equal(from_seed, from_generator)


def test_mean_remission_duration_matches_the_cohort_mean() -> None:
    # A remission still running at the end of follow up leaves no next onset,
    # so the observed remissions are the ones that completed inside the window
    # and their mean sits a couple of percent below the true 100 weeks. Fifty
    # cycles per patient keep that truncation well inside the tolerance.
    events = alternating_renewal(COHORT_LAMBDA, COHORT_MU, 5000.0, n=400, rng=7)
    durations = remission_durations(events)
    assert durations.mean() == pytest.approx(PAPER.tau_health_cohort_weeks.value, rel=0.05)


def test_mean_relapse_duration_matches_the_cohort_mean() -> None:
    events = alternating_renewal(COHORT_LAMBDA, COHORT_MU, 5000.0, n=400, rng=7)
    durations = complete_relapse_durations(events)
    assert durations.mean() == pytest.approx(PAPER.tau_no_health_cohort_weeks.value, rel=0.05)


def test_counts_of_brief_relapses_are_poisson() -> None:
    # Follow up starts in health, so no relapse is forced at time zero and the
    # expected count is exactly the long run onset rate times the window.
    window = 500.0
    events = alternating_renewal(
        COHORT_LAMBDA, BRIEF_RELAPSE_RATE, window, n=10000, rng=4, start_state="health"
    )
    counts = relapse_counts(events).to_numpy(dtype=float)
    expected = effective_onset_rate(COHORT_LAMBDA, BRIEF_RELAPSE_RATE) * window
    assert counts.mean() == pytest.approx(expected, rel=0.03)
    assert counts.var() / counts.mean() == pytest.approx(1.0, abs=0.05)


def test_counts_of_gamma_distributed_rates_are_negative_binomial() -> None:
    shape, scale, window = 2.0, 0.005, 500.0
    rates = gamma_rates(shape, scale, 10000, rng=5)
    events = alternating_renewal(
        rates, BRIEF_RELAPSE_RATE, window, n=10000, rng=6, start_state="health"
    )
    counts = relapse_counts(events).to_numpy(dtype=float)
    mean, dispersion = nb_from_gamma(shape, scale, window)
    moment_dispersion = (counts.var() - counts.mean()) / counts.mean() ** 2
    assert counts.mean() == pytest.approx(mean, rel=0.03)
    assert moment_dispersion == pytest.approx(dispersion, rel=0.05)


def test_effective_onset_rate_tends_to_lambda_for_brief_relapses() -> None:
    rate = effective_onset_rate(COHORT_LAMBDA, BRIEF_RELAPSE_RATE)
    assert rate == pytest.approx(COHORT_LAMBDA, rel=1e-5)


def test_effective_onset_rate_halves_when_the_two_rates_are_equal() -> None:
    assert effective_onset_rate(COHORT_LAMBDA, COHORT_LAMBDA) == pytest.approx(COHORT_LAMBDA / 2.0)


@pytest.mark.parametrize(("lam", "mu"), [(0.0, COHORT_MU), (COHORT_LAMBDA, -1.0)])
def test_effective_onset_rate_rejects_non_positive_rates(lam: float, mu: float) -> None:
    with pytest.raises(ValueError, match="must be positive"):
        effective_onset_rate(lam, mu)


def test_weekly_durations_are_whole_weeks() -> None:
    events = alternating_renewal(COHORT_LAMBDA, COHORT_MU, 2000.0, n=20, rng=7, discretise="week")
    relapses = (events["relapse_end"] - events["relapse_onset"]).to_numpy(dtype=float)
    remissions = remission_durations(events)
    assert np.all(relapses >= 1.0)
    assert np.all(relapses == np.round(relapses))
    assert np.all(remissions >= 1.0)
    assert np.all(remissions == np.round(remissions))


def test_weekly_durations_keep_the_cohort_means() -> None:
    events = alternating_renewal(COHORT_LAMBDA, COHORT_MU, 5000.0, n=200, rng=7, discretise="week")
    assert remission_durations(events).mean() == pytest.approx(
        PAPER.tau_health_cohort_weeks.value, rel=0.05
    )
    assert complete_relapse_durations(events).mean() == pytest.approx(
        PAPER.tau_no_health_cohort_weeks.value, rel=0.05
    )


def test_weekly_discretisation_rejects_a_mean_below_one_week() -> None:
    with pytest.raises(ValueError, match="at least one week"):
        alternating_renewal(COHORT_LAMBDA, 2.0, 100.0, n=1, rng=0, discretise="week")


def test_relapse_free_uses_the_exponential_survival_for_a_single_rate() -> None:
    assert relapse_free(50.0, lam=COHORT_LAMBDA) == pytest.approx(math.exp(-0.5))


def test_relapse_free_accepts_an_array_of_windows() -> None:
    values = relapse_free(np.array([0.0, 50.0]), lam=COHORT_LAMBDA)
    assert isinstance(values, np.ndarray)
    assert values == pytest.approx([1.0, math.exp(-0.5)])


def test_relapse_free_uses_the_gamma_mixture_when_k_and_theta_are_given() -> None:
    assert relapse_free(50.0, k=2.0, theta=0.005) == pytest.approx(1.25**-2.0)


def test_relapse_free_gamma_mixture_tends_to_the_exponential_for_a_large_shape() -> None:
    shape = 1.0e6
    value = relapse_free(50.0, k=shape, theta=COHORT_LAMBDA / shape)
    assert value == pytest.approx(math.exp(-0.5), abs=1e-4)


def test_relapse_free_rejects_no_parameterisation() -> None:
    with pytest.raises(ValueError, match="lam"):
        relapse_free(50.0)


def test_relapse_free_rejects_both_parameterisations() -> None:
    with pytest.raises(ValueError, match="not both"):
        relapse_free(50.0, lam=0.01, k=2.0, theta=0.005)


def test_relapse_free_rejects_half_of_the_gamma_parameterisation() -> None:
    with pytest.raises(ValueError, match="needs both k and theta"):
        relapse_free(50.0, k=2.0)


def test_negative_binomial_moments_of_a_gamma_poisson_mixture() -> None:
    assert nb_from_gamma(2.0, 0.005, 500.0) == pytest.approx((5.0, 0.5))


@pytest.mark.parametrize(
    ("k", "theta", "window"), [(0.0, 0.005, 500.0), (2.0, -0.005, 500.0), (2.0, 0.005, 0.0)]
)
def test_nb_from_gamma_rejects_non_positive_inputs(k: float, theta: float, window: float) -> None:
    with pytest.raises(ValueError, match="must be positive"):
        nb_from_gamma(k, theta, window)


def test_a_relapse_still_running_at_the_end_is_truncated() -> None:
    lam, mu = rates_from_means(100.0, 100.0)
    events = alternating_renewal(lam, mu, 1.0, n=1, rng=3)
    assert len(events) == 1
    assert events["relapse_end"].iloc[0] == 1.0
    assert events["relapse_onset"].iloc[0] == 0.0


def test_a_patient_without_a_relapse_gets_one_empty_row() -> None:
    events = alternating_renewal(1e-9, COHORT_MU, 10.0, n=1, rng=3, start_state="health")
    assert len(events) == 1
    assert math.isnan(events["relapse_onset"].iloc[0])
    assert math.isnan(events["relapse_end"].iloc[0])
    assert relapse_counts(events).tolist() == [0]


def test_per_patient_arrays_give_per_patient_windows() -> None:
    horizons = np.array([50.0, 500.0, 5000.0])
    events = alternating_renewal(COHORT_LAMBDA, COHORT_MU, horizons, n=3, rng=1)
    ends = events.groupby("patient_id", sort=True)["followup_end"].max().to_numpy(dtype=float)
    assert ends == pytest.approx(horizons)


@pytest.mark.parametrize("name", ["lambda_remission", "mu_relapse", "t_end"])
def test_non_positive_values_are_rejected(name: str) -> None:
    arguments: dict[str, object] = {
        "lambda_remission": COHORT_LAMBDA,
        "mu_relapse": COHORT_MU,
        "t_end": 100.0,
    }
    arguments[name] = 0.0
    with pytest.raises(ValueError, match="positive and finite"):
        alternating_renewal(n=1, rng=0, **arguments)  # type: ignore[arg-type]


def test_an_empty_cohort_is_rejected() -> None:
    with pytest.raises(ValueError, match="n must be positive"):
        alternating_renewal(COHORT_LAMBDA, COHORT_MU, 100.0, n=0)


def test_an_unknown_start_state_is_rejected() -> None:
    with pytest.raises(ValueError, match="start_state"):
        alternating_renewal(COHORT_LAMBDA, COHORT_MU, 100.0, start_state="cured")  # type: ignore[arg-type]


def test_an_unknown_discretisation_is_rejected() -> None:
    with pytest.raises(ValueError, match="discretise"):
        alternating_renewal(COHORT_LAMBDA, COHORT_MU, 100.0, discretise="day")  # type: ignore[arg-type]


@pytest.mark.parametrize("name", ["lambda_remission", "mu_relapse", "t_end"])
def test_arrays_of_the_wrong_length_are_rejected(name: str) -> None:
    arguments: dict[str, object] = {
        "lambda_remission": COHORT_LAMBDA,
        "mu_relapse": COHORT_MU,
        "t_end": 100.0,
    }
    arguments[name] = np.array([1.0, 2.0])
    with pytest.raises(ValueError, match="length"):
        alternating_renewal(n=3, rng=0, **arguments)  # type: ignore[arg-type]


def test_relapse_counts_names_the_missing_columns() -> None:
    with pytest.raises(ValueError, match="relapse_onset"):
        relapse_counts(example_events().drop(columns=["relapse_onset"]))


def test_relapse_counts_counts_every_onset_of_the_follow_up() -> None:
    counts = relapse_counts(example_events())
    assert counts.tolist() == [2, 1, 0]
    assert counts.name == "relapses"
    assert counts.index.tolist() == ["p0001", "p0002", "p0003"]
    assert counts.dtype == np.int64


def test_relapse_counts_applies_the_schema_validator() -> None:
    overlapping = example_events()
    overlapping.loc[1, "relapse_onset"] = 2.0
    with pytest.raises(ValueError, match="overlapping"):
        relapse_counts(overlapping)


def test_relapse_counts_accepts_a_frame_carrying_extra_columns() -> None:
    enriched = example_events()
    enriched["relapse_onset_date"] = pd.Timestamp("2020-01-01") + pd.to_timedelta(
        enriched["relapse_onset"] * 7.0, unit="D"
    )
    assert relapse_counts(enriched).tolist() == [2, 1, 0]


def test_relapse_counts_accepts_the_dated_export_of_the_reader_module() -> None:
    weekly = pd.DataFrame(
        {
            "patient_id": ["p0001", "p0001", "p0001", "p0001"],
            "week": pd.Series([0, 1, 2, 3], dtype="int64"),
            "state": pd.Series(
                [
                    PAPER.state_no_health.value,
                    PAPER.state_health.value,
                    PAPER.state_no_health.value,
                    PAPER.state_health.value,
                ],
                dtype="int64",
            ),
        }
    )
    dated = weekly_to_events(weekly, origin=pd.Timestamp("2020-01-01"))
    assert relapse_counts(dated).tolist() == [2]


def test_relapse_counts_rejects_a_non_positive_window() -> None:
    with pytest.raises(ValueError, match="window must be positive"):
        relapse_counts(example_events(), window=0.0)


def test_relapse_free_rejects_a_negative_window() -> None:
    with pytest.raises(ValueError, match="T must not be negative"):
        relapse_free(-1.0, lam=COHORT_LAMBDA)


def test_relapse_free_rejects_a_non_positive_rate() -> None:
    with pytest.raises(ValueError, match="lam must be positive"):
        relapse_free(50.0, lam=0.0)


@pytest.mark.parametrize(("k", "theta"), [(0.0, 0.005), (2.0, -0.005)])
def test_relapse_free_rejects_a_non_positive_gamma_parameter(k: float, theta: float) -> None:
    with pytest.raises(ValueError, match="must be positive"):
        relapse_free(50.0, k=k, theta=theta)


def test_relapse_counts_restricts_to_the_window() -> None:
    counts = relapse_counts(example_events(), window=10.0)
    assert counts.tolist() == [1, 1, 0]


def test_gamma_rates_are_positive_and_reproducible() -> None:
    first = gamma_rates(2.0, 0.005, 1000, rng=2)
    second = gamma_rates(2.0, 0.005, 1000, rng=2)
    assert first.shape == (1000,)
    assert np.all(first > 0.0)
    assert first.mean() == pytest.approx(0.01, rel=0.1)
    assert np.array_equal(first, second)


@pytest.mark.parametrize(("k", "theta", "n"), [(0.0, 0.005, 10), (2.0, 0.0, 10), (2.0, 0.005, 0)])
def test_gamma_rates_rejects_non_positive_inputs(k: float, theta: float, n: int) -> None:
    with pytest.raises(ValueError, match="must be positive"):
        gamma_rates(k, theta, n, rng=0)
