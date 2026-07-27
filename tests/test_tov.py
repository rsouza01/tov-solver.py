"""Tests for the TOV integrator.

The Buchdahl tests are the important ones: they compare against a closed-form
solution of Einstein's equations rather than against the integrator itself.
Everything else here is a guard rail.
"""

from __future__ import annotations

import numpy as np
import pytest

from tov_solver.eos import Polytrope
from tov_solver.eos.buchdahl import (
    Buchdahl,
    central_energy_density,
    central_pressure,
    exact_mass_radius,
)
from tov_solver.structure import mass_radius_curve, solve_tov

P_STAR = 83.4
"""MeV/fm^3. Chosen so the resulting stars are about ten km across."""

CAUSAL_BETAS = [0.02, 0.05, 0.10, 0.14, 0.16]
"""Compactnesses below 1/6, where the Buchdahl sound speed stays under c.

p_c = 36 beta^2 p_star and the sound speed reaches c at p = p_star, so
beta = 1/6 is exactly the causal limit.
"""


def _buchdahl_star(beta: float, pressure_min: float = 1e-10):
    """A Buchdahl star on the causal branch, p_max = p_star."""
    eos = Buchdahl(p_star=P_STAR, pressure_min=pressure_min, pressure_max=P_STAR)
    return solve_tov(eos.table(), central_pressure(P_STAR, beta))


# --- the benchmark --------------------------------------------------------


@pytest.mark.parametrize("beta", CAUSAL_BETAS)
def test_mass_matches_buchdahl_exactly(beta: float) -> None:
    """Gravitational mass against the closed form, to nine digits."""
    expected, _ = exact_mass_radius(P_STAR, beta)
    assert _buchdahl_star(beta).mass == pytest.approx(expected, rel=1e-7)


@pytest.mark.parametrize("beta", CAUSAL_BETAS)
def test_radius_matches_buchdahl(beta: float) -> None:
    """Radius against the closed form.

    Looser than the mass because the integration stops at a small but finite
    surface pressure rather than at p = 0.
    """
    _, expected = exact_mass_radius(P_STAR, beta)
    assert _buchdahl_star(beta).radius == pytest.approx(expected, rel=1e-4)


def test_central_energy_density_matches_buchdahl() -> None:
    star = _buchdahl_star(0.10)
    assert star.central_energy_density == pytest.approx(
        central_energy_density(P_STAR, 0.10), rel=1e-8
    )


def test_compactness_matches_beta() -> None:
    """beta is defined as M/R, so the integrator must recover the input."""
    assert _buchdahl_star(0.16).compactness == pytest.approx(0.16, rel=1e-4)


def test_radius_converges_as_surface_pressure_falls() -> None:
    """Radius error must shrink with the truncation pressure, not plateau early."""
    _, exact = exact_mass_radius(P_STAR, 0.10)
    coarse = abs(_buchdahl_star(0.10, pressure_min=1e-4).radius - exact) / exact
    fine = abs(_buchdahl_star(0.10, pressure_min=1e-8).radius - exact) / exact
    assert fine < coarse / 10.0


def test_buchdahl_table_is_thermodynamically_consistent() -> None:
    """The reconstructed baryon density must satisfy Gibbs-Duhem."""
    eos = Buchdahl(p_star=P_STAR, pressure_min=1e-8, pressure_max=P_STAR)
    report = eos.table().validate()
    assert report.is_consistent
    assert report.is_causal


def test_buchdahl_refuses_turnover() -> None:
    """eps(p) stops increasing at 36 p_star / 25; a table cannot cross it."""
    with pytest.raises(ValueError, match="turnover"):
        Buchdahl(p_star=10.0, pressure_min=1e-6, pressure_max=15.0)


def test_buchdahl_above_p_star_is_acausal() -> None:
    """The sound speed reaches c at p = p_star exactly.

    Allowed to be built -- it still solves Einstein's equations, and is
    useful as an integrator benchmark -- but validate() must say so.
    """
    causal = Buchdahl(p_star=50.0, pressure_min=1e-6, pressure_max=50.0).table()
    assert causal.validate().is_causal

    superluminal = Buchdahl(p_star=50.0, pressure_min=1e-6, pressure_max=70.0).table()
    assert not superluminal.validate().is_causal


def test_exact_mass_radius_rejects_unphysical_compactness() -> None:
    with pytest.raises(ValueError, match="compactness"):
        exact_mass_radius(P_STAR, 0.6)


# --- general properties ---------------------------------------------------


def test_polytrope_produces_a_plausible_star() -> None:
    table = Polytrope(gamma=2.0, kappa=100.0).table()
    star = solve_tov(table, 50.0)
    assert 5.0 < star.radius < 25.0
    assert 0.1 < star.mass < 3.0


def test_compactness_respects_the_buchdahl_bound() -> None:
    """No static star may exceed 2M/R = 8/9, i.e. compactness 4/9."""
    table = Polytrope(gamma=2.0, kappa=100.0).table()
    for p_c in (1.0, 20.0, 100.0):
        assert solve_tov(table, p_c).compactness < 4.0 / 9.0


def test_profiles_are_monotonic_and_end_at_the_surface() -> None:
    table = Polytrope(gamma=2.0, kappa=100.0).table()
    star = solve_tov(table, 50.0)
    assert np.all(np.diff(star.pressures) < 0.0)
    assert np.all(np.diff(star.masses) > 0.0)
    assert star.radii[-1] == pytest.approx(star.radius, rel=1e-6)


def test_mass_increases_then_turns_over() -> None:
    """A sequence must show a maximum mass, not climb forever."""
    table = Polytrope(gamma=2.0, kappa=100.0, density_max=5.0).table()
    models = mass_radius_curve(table, np.logspace(0, 3.35, 50))
    masses = np.array([m.mass for m in models])
    assert masses.argmax() not in (0, masses.size - 1)
    assert masses.max() == pytest.approx(1.026, rel=1e-2)


# --- error handling -------------------------------------------------------


def test_rejects_central_pressure_outside_the_table() -> None:
    table = Polytrope(gamma=2.0, kappa=100.0).table()
    with pytest.raises(ValueError, match="outside the tabulated range"):
        solve_tov(table, 1e9)


def test_rejects_surface_pressure_below_the_table() -> None:
    table = Polytrope(gamma=2.0, kappa=100.0).table()
    with pytest.raises(ValueError, match="below the table minimum"):
        solve_tov(table, 50.0, surface_pressure=1e-30)


def test_rejects_surface_pressure_above_central() -> None:
    table = Polytrope(gamma=2.0, kappa=100.0).table()
    with pytest.raises(ValueError, match="not below the central pressure"):
        solve_tov(table, 10.0, surface_pressure=20.0)


def test_mass_radius_curve_skips_failures() -> None:
    """Out-of-range central pressures are dropped, not raised."""
    table = Polytrope(gamma=2.0, kappa=100.0).table()
    models = mass_radius_curve(table, np.array([1e-30, 10.0, 50.0, 1e9]))
    assert len(models) == 2
