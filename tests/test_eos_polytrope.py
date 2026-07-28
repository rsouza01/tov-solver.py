"""Tests for the polytropic equation of state."""

from __future__ import annotations

import numpy as np
import pytest

from tov_solver.eos.polytrope import MASS_NEUTRON, Polytrope


def test_rejects_gamma_at_or_below_one() -> None:
    with pytest.raises(ValueError, match="gamma must exceed 1"):
        Polytrope(gamma=1.0, kappa=100.0)


def test_rejects_non_positive_kappa() -> None:
    with pytest.raises(ValueError, match="kappa must be positive"):
        Polytrope(gamma=2.0, kappa=0.0)


def test_rejects_inverted_density_range() -> None:
    with pytest.raises(ValueError, match="density_min < density_max"):
        Polytrope(gamma=2.0, kappa=100.0, density_min=1.0, density_max=0.1)


def test_pressure_is_the_power_law() -> None:
    model = Polytrope(gamma=2.5, kappa=42.0)
    n = np.array([0.1, 0.2, 0.4])
    np.testing.assert_allclose(model.pressure(n), 42.0 * n**2.5, rtol=1e-14)


def test_energy_density_reduces_to_rest_mass_at_low_density() -> None:
    """As n -> 0 the internal-energy term vanishes and eps -> m n."""
    model = Polytrope(gamma=2.0, kappa=100.0)
    n = np.array([1e-8])
    np.testing.assert_allclose(model.energy_density(n), MASS_NEUTRON * n, rtol=1e-6)


def test_euler_relation_holds_analytically() -> None:
    """eps + P = mu n, from the closed forms rather than the table."""
    model = Polytrope(gamma=2.75, kappa=30.0)
    n = np.linspace(0.05, 1.0, 50)
    lhs = model.energy_density(n) + model.pressure(n)
    rhs = model.chemical_potential(n) * n
    np.testing.assert_allclose(lhs, rhs, rtol=1e-13)


def test_sound_speed_closed_form_matches_numerical_derivative() -> None:
    """Gamma P / (eps + P) must equal dP/deps computed by finite difference."""
    model = Polytrope(gamma=2.0, kappa=100.0)
    n = 0.3
    h = 1e-6
    grid = np.array([n - h, n + h])
    eps = model.energy_density(grid)
    P = model.pressure(grid)
    numerical = (P[1] - P[0]) / (eps[1] - eps[0])
    assert model.sound_speed_squared(np.array([n]))[0] == pytest.approx(numerical, rel=1e-8)


def test_adiabatic_index_is_the_log_log_slope() -> None:
    """d ln P / d ln n = Gamma. That is what the adiabatic index means.

    Comparing raw pressures at a fixed density would be the wrong test:
    below n = 1 fm^-3 a larger Gamma gives a *smaller* P, since the
    dimensionless power law is evaluated on a number less than one.
    """
    for gamma in (1.5, 2.0, 2.75):
        model = Polytrope(gamma=gamma, kappa=100.0)
        n = np.array([0.1, 0.2])
        P = model.pressure(n)
        slope = np.log(P[1] / P[0]) / np.log(n[1] / n[0])
        assert slope == pytest.approx(gamma, rel=1e-12)


def test_default_grid_spans_the_declared_range() -> None:
    model = Polytrope(gamma=2.0, kappa=100.0, density_min=0.01, density_max=1.0)
    grid = model.default_grid(64)
    assert grid.size == 64
    assert grid[0] == pytest.approx(0.01)
    assert grid[-1] == pytest.approx(1.0)


def test_table_columns_agree_with_direct_evaluation() -> None:
    model = Polytrope(gamma=2.0, kappa=100.0, density_max=0.5)
    table = model.table()
    np.testing.assert_allclose(table.pressure, model.pressure(table.baryon_density), rtol=1e-14)
    np.testing.assert_allclose(
        table.energy_density, model.energy_density(table.baryon_density), rtol=1e-14
    )


def test_table_accepts_an_explicit_grid() -> None:
    model = Polytrope(gamma=2.0, kappa=100.0)
    grid = np.linspace(0.05, 0.5, 32)
    assert len(model.table(grid)) == 32
