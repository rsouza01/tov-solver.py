"""Tests for the tabulated SWRDP equation of state.

The maximum-mass test is the one that matters: it checks the loader, the
interpolation, the units and the TOV integrator together, against numbers
published in Gomes et al. (2015).
"""

from __future__ import annotations

import numpy as np
import pytest

from tov_solver.eos.swrdp import SWRDP, available_zeta
from tov_solver.structure import mass_radius_curve

TABULATED = (0.040, 0.049, 0.059, 0.071, 0.085)

PUBLISHED_MAX_MASS = {
    0.040: 2.15,
    0.049: 2.07,
    0.059: 1.99,
    0.071: 1.91,
    0.085: 1.83,
}
"""Gomes et al. 2015, ApJ 808, 8, Table 5, sigma-omega-rho-delta-phi column.

Hyperon stars with the potentials fixed to U_Lambda = -28 MeV,
U_Sigma = +30 MeV, U_Xi = -18 MeV, and a_sym = 32 MeV, L_0 = 97 MeV.
"""


# --- discovery and construction -------------------------------------------


def test_finds_all_tabulated_parametrizations() -> None:
    assert available_zeta() == TABULATED


def test_rejects_zeta_outside_the_tabulated_range() -> None:
    with pytest.raises(ValueError, match="outside the tabulated range"):
        SWRDP(zeta=0.02)
    with pytest.raises(ValueError, match="outside the tabulated range"):
        SWRDP(zeta=0.2)


@pytest.mark.parametrize("zeta", TABULATED)
def test_tabulated_zeta_is_not_interpolated(zeta: float) -> None:
    assert not SWRDP(zeta=zeta).is_interpolated


def test_intermediate_zeta_is_interpolated() -> None:
    assert SWRDP(zeta=0.045).is_interpolated


def test_interpolated_range_is_the_intersection() -> None:
    """Tables start at different densities; an interpolant needs both."""
    low, high = SWRDP(zeta=0.045).density_range
    assert low == pytest.approx(SWRDP(zeta=0.049).density_range[0])
    assert high == pytest.approx(SWRDP(zeta=0.040).density_range[1])


# --- the tables themselves -------------------------------------------------


@pytest.mark.parametrize("zeta", TABULATED)
def test_table_is_causal_and_consistent(zeta: float) -> None:
    report = SWRDP(zeta=zeta).table().validate()
    assert report.is_causal
    assert report.is_consistent


@pytest.mark.parametrize("zeta", TABULATED)
def test_euler_relation_matches_the_files_own_column(zeta: float) -> None:
    """The derived mu must agree with the mu the solver wrote out.

    This is the check that justifies not storing a chemical potential:
    the Euler relation reproduces the tabulated column to a few parts in
    10^5, so the derived value carries no less information and cannot fall
    out of step with p and eps.
    """
    model = SWRDP(zeta=zeta)
    table = model.table()
    tabulated = model.tabulated_chemical_potential(table.baryon_density)
    np.testing.assert_allclose(table.chemical_potential, tabulated, rtol=1e-4)


def test_tabulated_potential_refused_when_interpolating() -> None:
    with pytest.raises(ValueError, match="only available at a tabulated zeta"):
        SWRDP(zeta=0.045).tabulated_chemical_potential(np.array([0.1]))


# --- physics ---------------------------------------------------------------


def test_smaller_zeta_is_stiffer() -> None:
    """Lower zeta means stronger many-body repulsion, hence more pressure."""
    n = np.array([0.5])
    pressures = [SWRDP(zeta=z).pressure(n)[0] for z in TABULATED]
    assert np.all(np.diff(pressures) < 0.0)


def test_interpolated_pressure_lies_between_its_brackets() -> None:
    n = np.array([0.3, 0.5, 0.8])
    low = SWRDP(zeta=0.040).pressure(n)
    mid = SWRDP(zeta=0.045).pressure(n)
    high = SWRDP(zeta=0.049).pressure(n)
    assert np.all(high < mid)
    assert np.all(mid < low)


@pytest.mark.parametrize("zeta", TABULATED)
def test_maximum_mass_reproduces_gomes_2015(zeta: float) -> None:
    """The reproduction milestone.

    Published values carry two decimals, so the tolerance is half of the
    last digit. Every parametrization currently lands within 0.004.
    """
    table = SWRDP(zeta=zeta).table()
    pressures = np.logspace(0.5, np.log10(table.pressure[-1] * 0.999), 40)
    masses = [star.mass for star in mass_radius_curve(table, pressures)]
    assert max(masses) == pytest.approx(PUBLISHED_MAX_MASS[zeta], abs=0.005)


# --- physical scale --------------------------------------------------------
#
# The first-law check in EoSTable.validate is nearly blind to a uniform
# rescaling of a column. These are the tests that are not: a unit error
# anywhere in the loader moves these numbers by orders of magnitude.


@pytest.mark.parametrize("zeta", TABULATED)
def test_chemical_potential_starts_at_the_nucleon_mass(zeta: float) -> None:
    """At the lowest tabulated density matter is non-relativistic nucleons.

    The chemical potential is then just the nucleon rest mass, and the
    tables begin at exactly 940 MeV. Nothing about this follows from a
    thermodynamic identity -- it is the physical scale, and it is what
    would have caught a factor of hbar c.
    """
    table = SWRDP(zeta=zeta).table()
    assert table.chemical_potential[0] == pytest.approx(940.0, abs=0.5)


@pytest.mark.parametrize("zeta", TABULATED)
def test_central_densities_are_nuclear(zeta: float) -> None:
    """The tables must span roughly a hundredth of saturation to ~10 n_0."""
    low, high = SWRDP(zeta=zeta).density_range
    assert 1e-3 < low < 0.05
    assert 1.0 < high < 2.0