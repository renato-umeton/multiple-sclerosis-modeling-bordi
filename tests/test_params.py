from __future__ import annotations

import dataclasses

import pytest

from msrelapse._params import (
    PAPER,
    Bordi2013,
    Param,
    symmetric_barrier,
    symmetric_well_position,
)

EXPECTED_N_PARAMS = 56
RECORD_KEYS = {"name", "value", "unit", "source"}


def field_names() -> list[str]:
    return [field.name for field in dataclasses.fields(Bordi2013)]


@pytest.mark.parametrize("name", field_names())
def test_every_field_is_a_param(paper: Bordi2013, name: str) -> None:
    assert isinstance(getattr(paper, name), Param)


@pytest.mark.parametrize("name", field_names())
def test_every_unit_is_non_empty(paper: Bordi2013, name: str) -> None:
    param: Param[object] = getattr(paper, name)
    assert param.unit.strip() != ""


@pytest.mark.parametrize("name", field_names())
def test_every_source_cites_the_paper(paper: Bordi2013, name: str) -> None:
    param: Param[object] = getattr(paper, name)
    assert param.source.startswith(("Bordi 2013", "Derived from Bordi 2013"))


@pytest.mark.parametrize("name", field_names())
def test_derived_fields_are_marked_as_derived(paper: Bordi2013, name: str) -> None:
    param: Param[object] = getattr(paper, name)
    is_derived_source = param.source.startswith("Derived from Bordi 2013")
    assert is_derived_source == name.startswith("derived_")


def test_rr_phase_histogram_counts_sum_to_the_cohort_size(paper: Bordi2013) -> None:
    assert sum(paper.fig3_counts.value) == paper.n_patients.value


def test_no_health_histogram_counts_sum_to_the_printed_total(paper: Bordi2013) -> None:
    assert sum(paper.fig4a_counts.value) == paper.n_no_health_events.value


def test_health_histogram_counts_sum_to_the_printed_total(paper: Bordi2013) -> None:
    assert sum(paper.fig4b_counts.value) == paper.n_health_events.value


def test_cohort_size_is_seventy(paper: Bordi2013) -> None:
    assert paper.n_patients.value == 70


def test_no_health_event_total_is_two_hundred_eighteen(paper: Bordi2013) -> None:
    assert paper.n_no_health_events.value == 218


def test_health_event_total_is_two_hundred_sixty_six(paper: Bordi2013) -> None:
    assert paper.n_health_events.value == 266


def test_cohort_health_residence_time(paper: Bordi2013) -> None:
    assert paper.tau_health_cohort_weeks.value == 100.0


def test_cohort_no_health_residence_time(paper: Bordi2013) -> None:
    assert paper.tau_no_health_cohort_weeks.value == 4.3


def test_patient_23_health_residence_time(paper: Bordi2013) -> None:
    assert paper.tau_health_p23_weeks.value == 117.7


def test_patient_23_no_health_residence_time(paper: Bordi2013) -> None:
    assert paper.tau_no_health_p23_weeks.value == 1.5


def test_patient_23_barrier_ratio(paper: Bordi2013) -> None:
    assert paper.barrier_ratio_p23.value == 11.8


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("beta_patient_23", 0.25),
        ("beta_patient_32", 0.19),
        ("beta_patient_53", 0.12),
    ],
)
def test_patient_asymmetry_values(paper: Bordi2013, name: str, expected: float) -> None:
    param: Param[float] = getattr(paper, name)
    assert param.value == expected


def test_noise_variance(paper: Bordi2013) -> None:
    assert paper.epsilon_noise_variance.value == 0.13


def test_noise_amplitude_squares_back_to_the_noise_variance(paper: Bordi2013) -> None:
    assert paper.noise_amplitude.value**2 == pytest.approx(paper.epsilon_noise_variance.value)


def test_reference_alpha(paper: Bordi2013) -> None:
    assert paper.alpha_reference.value == 1.0


def test_illustrative_low_alpha(paper: Bordi2013) -> None:
    assert paper.alpha_illustrative_low.value == 0.7


def test_paper_doi(paper: Bordi2013) -> None:
    assert paper.paper_doi.value == "10.1155/2013/910321"


def test_as_records_covers_every_reported_number(paper: Bordi2013) -> None:
    assert len(paper.as_records()) == EXPECTED_N_PARAMS


def test_as_records_entries_carry_the_four_documented_keys(paper: Bordi2013) -> None:
    assert all(set(record) == RECORD_KEYS for record in paper.as_records())


def test_as_records_names_match_the_field_names(paper: Bordi2013) -> None:
    assert [record["name"] for record in paper.as_records()] == field_names()


def test_stored_barrier_matches_the_helper_at_the_reference_alpha(paper: Bordi2013) -> None:
    expected = symmetric_barrier(paper.alpha_reference.value)
    assert paper.derived_symmetric_barrier.value == expected


def test_stored_well_position_matches_the_helper_at_the_reference_alpha(paper: Bordi2013) -> None:
    expected = symmetric_well_position(paper.alpha_reference.value)
    assert paper.derived_symmetric_well_position.value == expected


def test_symmetric_barrier_at_reference_alpha() -> None:
    assert symmetric_barrier(1.0) == 0.25


def test_symmetric_barrier_at_low_alpha() -> None:
    assert symmetric_barrier(0.7) == pytest.approx(0.357142857, abs=1e-9)


def test_symmetric_well_position_at_reference_alpha() -> None:
    assert symmetric_well_position(1.0) == 1.0


def test_symmetric_well_position_at_low_alpha() -> None:
    assert symmetric_well_position(0.7) == pytest.approx(1.195229, abs=1e-6)


@pytest.mark.parametrize("alpha", [0.0, -1.0])
def test_symmetric_barrier_rejects_non_positive_alpha(alpha: float) -> None:
    with pytest.raises(ValueError, match="alpha must be positive"):
        symmetric_barrier(alpha)


@pytest.mark.parametrize("alpha", [0.0, -1.0])
def test_symmetric_well_position_rejects_non_positive_alpha(alpha: float) -> None:
    with pytest.raises(ValueError, match="alpha must be positive"):
        symmetric_well_position(alpha)


def test_paper_is_frozen(paper: Bordi2013) -> None:
    replacement: Param[int] = Param(value=1, unit="patients", source="Bordi 2013")
    with pytest.raises(dataclasses.FrozenInstanceError):
        paper.n_patients = replacement  # type: ignore[misc]


def test_param_instances_are_frozen() -> None:
    with pytest.raises(dataclasses.FrozenInstanceError):
        PAPER.n_patients.value = 1  # type: ignore[misc]
