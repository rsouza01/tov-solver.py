"""Tests for the units module.

The point of these is not coverage. It is that every dimensional factor in
the project is pinned to an independently checkable number.
"""

import numpy as np
import pytest

from tov_solver.units import (
    HBARC,
    M_SUN_KM,
    fm_inv4_to_mev_fm3,
    fm_inv_to_mev,
    inv_km2_to_mev_fm3,
    km_to_m_sun,
    mev_fm3_to_dyn_cm2,
    mev_fm3_to_fm_inv4,
    mev_fm3_to_g_cm3,
    mev_fm3_to_inv_km2,
    mev_to_fm_inv,
)

# --- known values ---------------------------------------------------------


def test_hbarc_value() -> None:
    """hbar*c = 197.3269804 MeV.fm (PDG / CODATA)."""
    assert HBARC == pytest.approx(197.3269804, rel=1e-9)


def test_solar_mass_geometrized() -> None:
    """GM_sun/c^2 = 1.47662 km, a standard textbook figure."""
    assert M_SUN_KM == pytest.approx(1.47662, rel=1e-5)


def test_saturation_density_in_cgs() -> None:
    """n_0 = 0.16 fm^-3 of nucleons ~ 2.7e14 g/cm^3.

    Uses 939 MeV per nucleon, so this checks the g/cm^3 factor against the
    canonical 'nuclear density' quoted throughout the literature.
    """
    eps = 0.16 * 939.0  # MeV/fm^3
    assert mev_fm3_to_g_cm3(eps) == pytest.approx(2.68e14, rel=1e-2)


def test_one_mev_fm3_in_cgs_pressure() -> None:
    """1 MeV/fm^3 = 1.602177e33 dyn/cm^2."""
    assert mev_fm3_to_dyn_cm2(1.0) == pytest.approx(1.602176634e33, rel=1e-9)


# --- round trips ----------------------------------------------------------


@pytest.mark.parametrize(
    ("forward", "backward"),
    [
        (fm_inv_to_mev, mev_to_fm_inv),
        (fm_inv4_to_mev_fm3, mev_fm3_to_fm_inv4),
        (mev_fm3_to_inv_km2, inv_km2_to_mev_fm3),
    ],
)
def test_round_trip_is_identity(forward, backward) -> None:
    x = np.array([1e-6, 1.0, 3.7, 1e4])
    np.testing.assert_allclose(backward(forward(x)), x, rtol=1e-12)


def test_conversions_accept_scalars_and_arrays() -> None:
    assert isinstance(fm_inv_to_mev(1.0), float)
    assert isinstance(fm_inv_to_mev(np.array([1.0, 2.0])), np.ndarray)


# --- regression guard for the hbar*c bug class ----------------------------


def test_ultrarelativistic_gas_ratio() -> None:
    """eps/p = 3 for an ultrarelativistic gas -- only if units agree.

    This is the shape of the bug that survived seven passing tests in the
    previous codebase: pressure carried a factor of hbar*c, energy density
    did not, and the resulting eps/p of 0.016 was never checked against
    anything. Any future EoS should satisfy a ratio test like this one.
    """
    p_natural = 78.6  # fm^-4, arbitrary
    eps_natural = 3.0 * p_natural

    p = fm_inv4_to_mev_fm3(p_natural)
    eps = fm_inv4_to_mev_fm3(eps_natural)

    assert eps / p == pytest.approx(3.0, rel=1e-12)


def test_geometrized_density_scale() -> None:
    """Nuclear-density matter is ~1e-4 km^-2 in geometrized units.

    Sanity anchor for the TOV integrator: if central densities come out
    many orders from this, the geometrized conversion is wrong.
    """
    eps = 150.0  # MeV/fm^3, roughly saturation
    assert mev_fm3_to_inv_km2(eps) == pytest.approx(2.0e-4, rel=0.05)


def test_typical_neutron_star_mass_in_km() -> None:
    """A 2.08 M_sun star is ~3.07 km of geometrized mass."""
    assert km_to_m_sun(3.0714) == pytest.approx(2.08, rel=1e-3)
