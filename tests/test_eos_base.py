"""Tests for the EoS table contract and its validation."""

from __future__ import annotations

import numpy as np
import pytest

from tov_solver.eos import EoSTable, Polytrope


@pytest.fixture
def table() -> EoSTable:
    """A well-behaved table from a causal polytrope."""
    return Polytrope(gamma=2.0, kappa=100.0, density_max=0.5).table()


# --- construction invariants ----------------------------------------------


def _columns(n: int = 20) -> dict[str, np.ndarray]:
    model = Polytrope(gamma=2.0, kappa=100.0)
    grid = model.default_grid(n)
    return {
        "baryon_density": grid,
        "energy_density": model.energy_density(grid),
        "pressure": model.pressure(grid),
    }


def test_rejects_non_monotonic_column() -> None:
    columns = _columns()
    columns["pressure"] = columns["pressure"][::-1]
    with pytest.raises(ValueError, match="strictly increasing"):
        EoSTable(**columns)


def test_rejects_negative_pressure() -> None:
    columns = _columns()
    columns["pressure"] = columns["pressure"] - columns["pressure"][5]
    with pytest.raises(ValueError, match="strictly positive"):
        EoSTable(**columns)


def test_rejects_non_finite_values() -> None:
    columns = _columns()
    columns["energy_density"] = columns["energy_density"].copy()
    columns["energy_density"][3] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        EoSTable(**columns)


def test_rejects_mismatched_lengths() -> None:
    columns = _columns()
    columns["pressure"] = columns["pressure"][:-1]
    with pytest.raises(ValueError, match="lengths disagree"):
        EoSTable(**columns)


def test_rejects_too_few_samples() -> None:
    columns = _columns(n=3)
    with pytest.raises(ValueError, match="at least 4 samples"):
        EoSTable(**columns)


# --- derived quantities ---------------------------------------------------


def test_chemical_potential_matches_closed_form(table: EoSTable) -> None:
    """The Euler relation must reproduce the polytrope's analytic mu."""
    model = Polytrope(gamma=2.0, kappa=100.0, density_max=0.5)
    expected = model.chemical_potential(table.baryon_density)
    np.testing.assert_allclose(table.chemical_potential, expected, rtol=1e-12)


def test_sound_speed_matches_closed_form(table: EoSTable) -> None:
    """Numerical (dp/dn)/(deps/dn) must reproduce Gamma P / (eps + P)."""
    model = Polytrope(gamma=2.0, kappa=100.0, density_max=0.5)
    expected = model.sound_speed_squared(table.baryon_density)
    np.testing.assert_allclose(table.sound_speed_squared, expected, rtol=1e-4)


# --- interpolation round trips --------------------------------------------


def test_pressure_energy_density_round_trip(table: EoSTable) -> None:
    mid = table.energy_density[5:-5]
    recovered = table.energy_density_at_pressure(table.pressure_at_energy_density(mid))
    np.testing.assert_allclose(recovered, mid, rtol=1e-10)


def test_interpolators_return_nan_outside_range(table: EoSTable) -> None:
    assert np.isnan(table.energy_density_at_pressure(1e-30))
    assert np.isnan(table.pressure_at_energy_density(1e30))


def test_potential_interpolators_agree_with_columns(table: EoSTable) -> None:
    mu = table.chemical_potential[3:-3]
    np.testing.assert_allclose(table.pressure_at_potential(mu), table.pressure[3:-3], rtol=1e-10)


# --- validation -----------------------------------------------------------


def test_causal_polytrope_validates(table: EoSTable) -> None:
    report = table.validate()
    assert report.ok
    assert report.is_causal
    assert report.is_consistent
    report.raise_if_invalid()


def test_first_law_residual_is_small(table: EoSTable) -> None:
    """deps/dn = mu is a real check: it is not enforced by construction."""
    assert table.validate().median_first_law_residual < 1e-6


def test_superluminal_polytrope_is_rejected() -> None:
    """A stiff polytrope taken to high density exceeds c, and must be caught."""
    table = Polytrope(gamma=3.0, kappa=2000.0, density_max=5.0).table()
    report = table.validate()
    assert not report.is_causal
    assert report.max_sound_speed_squared > 1.0
    with pytest.raises(ValueError, match="acausal"):
        report.raise_if_invalid()


def test_distorted_table_is_caught() -> None:
    """A tilt in eps(n) breaks the first law and must be detected.

    A *tilt* rather than a rescaling, deliberately. A constant factor on one
    column is largely absorbed by the Euler relation and is not what this
    check is for; see :class:`ValidationReport`. Getting the shape of the
    equation of state wrong is.
    """
    columns = _columns(200)
    n = columns["baryon_density"]
    columns["energy_density"] = columns["energy_density"] * (n / n[0]) ** 0.02
    report = EoSTable(**columns).validate()
    assert not report.is_consistent
    with pytest.raises(ValueError, match="inconsistent"):
        report.raise_if_invalid()


def test_uniform_rescaling_is_not_detected() -> None:
    """Documents a known blind spot rather than asserting a virtue.

    Scaling the energy density leaves the first law almost satisfied wherever
    p << eps. This test exists so that the limitation is recorded in the
    suite and does not have to be rediscovered.
    """
    columns = _columns(200)
    columns["energy_density"] = columns["energy_density"] * 197.0
    assert EoSTable(**columns).validate().is_consistent
