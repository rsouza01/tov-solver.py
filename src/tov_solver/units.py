"""Physical constants and unit conversions.

Single source of truth for every dimensional factor in the project.

Conventions
-----------
Internal working units are *natural nuclear units*:

    energy / chemical potential   MeV
    pressure / energy density     MeV / fm^3
    number density                fm^-3
    Fermi momentum                fm^-1

Geometrized units (G = c = 1, lengths in km) appear **only** inside the TOV
integrator, and only via the explicit converters below.

Design notes
------------
1. Derived constants are *computed* from primitives, never transcribed.
   Transcription is how you end up with 5.07e-3 instead of 5.0677e-3.

2. Conversions are exposed as named, directional functions rather than bare
   multiplicative constants. A float called ``factor`` can be applied in the
   wrong direction, or forgotten entirely, and nothing complains. A function
   called ``fm_inv4_to_mev_fm3`` cannot.

3. Particle masses deliberately live with the models that use them
   (MFTQCD, SWRDP), not here. This module is about *dimensions*, not physics.
"""

from typing import Final

import numpy as np

__all__ = [
    "HBARC",
    "M_SUN_KM",
    "N_SATURATION",
    "fm_inv4_to_mev_fm3",
    "fm_inv_to_mev",
    "inv_km2_to_mev_fm3",
    "km_to_m_sun",
    "mev_fm3_to_dyn_cm2",
    "mev_fm3_to_fm_inv4",
    "mev_fm3_to_g_cm3",
    "mev_fm3_to_inv_km2",
    "mev_to_fm_inv",
]

# Every converter below is generic over this: it accepts a float or an
# ndarray and returns the same type. PEP 695 syntax (Python >= 3.12).
type _Numeric = float | np.ndarray


# --------------------------------------------------------------------------
# Primitive constants  (SI-2019 exact where marked; CODATA 2018 otherwise)
# --------------------------------------------------------------------------

C_LIGHT: Final[float] = 2.997_924_58e8  # m/s            (exact by definition)
MEV_TO_J: Final[float] = 1.602_176_634e-13  # J/MeV      (exact by definition)
G_NEWTON: Final[float] = 6.674_30e-11  # m^3 kg^-1 s^-2  (CODATA 2018)
GM_SUN: Final[float] = 1.327_124_4e20  # m^3 s^-2        (IAU 2015 nominal)

HBARC: Final[float] = 197.326_980_4
"""Reduced Planck constant times c, in MeV.fm.

The conversion factor between natural and physical units. If a quantity
computed in natural units has the wrong power of this, the result is off by
a factor of ~197 per power -- large enough to be obvious in a ratio test,
small enough to look plausible in isolation.
"""

N_SATURATION: Final[float] = 0.16
"""Nuclear saturation density, fm^-3. Reference scale, not a fitted value."""


# --------------------------------------------------------------------------
# Derived constants
# --------------------------------------------------------------------------

_FM_TO_M: Final[float] = 1e-15
_M_TO_KM: Final[float] = 1e-3

_MEV_FM3_TO_J_M3: Final[float] = MEV_TO_J / _FM_TO_M**3
_G_OVER_C4: Final[float] = G_NEWTON / C_LIGHT**4  # (m^-2) per (J m^-3)

_MEV_FM3_TO_INV_KM2: Final[float] = _MEV_FM3_TO_J_M3 * _G_OVER_C4 / _M_TO_KM**2
_MEV_FM3_TO_G_CM3: Final[float] = MEV_TO_J / C_LIGHT**2 / _FM_TO_M**3 * 1e3 / 1e6
_MEV_FM3_TO_DYN_CM2: Final[float] = _MEV_FM3_TO_J_M3 * 10.0

M_SUN_KM: Final[float] = GM_SUN / C_LIGHT**2 * _M_TO_KM
"""Solar mass in geometrized units (km). GM_sun/c^2 ~ 1.4766 km."""


# --------------------------------------------------------------------------
# Natural-unit conversions
# --------------------------------------------------------------------------


def fm_inv_to_mev[T: _Numeric](x: T) -> T:
    """Momentum, mass or chemical potential: fm^-1 -> MeV."""
    return x * HBARC  # type: ignore[return-value]


def mev_to_fm_inv[T: _Numeric](x: T) -> T:
    """Momentum, mass or chemical potential: MeV -> fm^-1."""
    return x / HBARC  # type: ignore[return-value]


def fm_inv4_to_mev_fm3[T: _Numeric](x: T) -> T:
    """Pressure or energy density: fm^-4 -> MeV/fm^3.

    Fermi-gas integrals evaluated with momenta in fm^-1 come out in fm^-4.
    This is the conversion to apply, exactly once, on the way out.
    """
    return x * HBARC  # type: ignore[return-value]


def mev_fm3_to_fm_inv4[T: _Numeric](x: T) -> T:
    """Pressure or energy density: MeV/fm^3 -> fm^-4."""
    return x / HBARC  # type: ignore[return-value]


# --------------------------------------------------------------------------
# Geometrized units (TOV integration)
# --------------------------------------------------------------------------


def mev_fm3_to_inv_km2[T: _Numeric](x: T) -> T:
    """Pressure or energy density: MeV/fm^3 -> km^-2 (G = c = 1)."""
    return x * _MEV_FM3_TO_INV_KM2  # type: ignore[return-value]


def inv_km2_to_mev_fm3[T: _Numeric](x: T) -> T:
    """Pressure or energy density: km^-2 -> MeV/fm^3."""
    return x / _MEV_FM3_TO_INV_KM2  # type: ignore[return-value]


def km_to_m_sun[T: _Numeric](x: T) -> T:
    """Geometrized mass: km -> solar masses."""
    return x / M_SUN_KM  # type: ignore[return-value]


# --------------------------------------------------------------------------
# CGS output (for comparison with the literature)
# --------------------------------------------------------------------------


def mev_fm3_to_g_cm3[T: _Numeric](x: T) -> T:
    """Mass-energy density: MeV/fm^3 -> g/cm^3."""
    return x * _MEV_FM3_TO_G_CM3  # type: ignore[return-value]


def mev_fm3_to_dyn_cm2[T: _Numeric](x: T) -> T:
    """Pressure: MeV/fm^3 -> dyn/cm^2."""
    return x * _MEV_FM3_TO_DYN_CM2  # type: ignore[return-value]
