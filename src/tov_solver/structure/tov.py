"""Tolman-Oppenheimer-Volkoff integration.

Hydrostatic equilibrium for a spherically symmetric, non-rotating star in
general relativity. With G = c = 1 and every quantity carrying units of a
power of length,

    dm/dr = 4 pi r^2 eps
    dp/dr = -(eps + p) (m + 4 pi r^3 p) / (r (r - 2m))

Radii and masses are in km inside the integrator; the public interface
reports mass in solar masses. Pressure and energy density are MeV/fm^3 on
the way in and out, converted at the boundary by :mod:`tov_solver.units`.

The centre is a coordinate singularity -- both equations divide by r -- so
integration starts at a small offset using the series expansion

    m(r)  = (4/3) pi eps_c r^3
    p(r)  = p_c - (2/3) pi (eps_c + p_c)(eps_c + 3 p_c) r^2

which is accurate to the order dropped, O(r^4), and utterly negligible at
the default offset of a metre against a ten-kilometre star.

The surface is located by a terminal event on the pressure rather than by
integrating to a fixed radius, so the radius comes out of the solver's own
root-finding rather than from the resolution of an output grid.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from scipy.integrate import solve_ivp

from tov_solver.units import inv_km2_to_mev_fm3, km_to_m_sun, mev_fm3_to_inv_km2

if TYPE_CHECKING:
    from tov_solver.eos.base import Array, EoSTable

__all__ = ["StellarModel", "mass_radius_curve", "solve_tov"]

# Offset from the centre at which the series expansion hands over to the
# integrator, in km. One metre.
_START_RADIUS = 1e-3

# Give up beyond this radius, in km. A star that has not found its surface
# by here is not a star.
_MAX_RADIUS = 200.0

_FOUR_PI = 4.0 * np.pi


@dataclass(frozen=True)
class StellarModel:
    """One equilibrium configuration.

    Attributes
    ----------
    radius
        Circumferential radius of the surface, km.
    mass
        Gravitational mass, solar masses.
    central_pressure, central_energy_density
        MeV/fm^3.
    radii, masses, pressures
        Profiles from centre to surface, in km, solar masses and MeV/fm^3.
        These are the integrator's own steps, so their spacing is adaptive.
    """

    radius: float
    mass: float
    central_pressure: float
    central_energy_density: float
    radii: Array
    masses: Array
    pressures: Array

    @property
    def compactness(self) -> float:
        """GM/Rc^2, dimensionless. Bounded above by 4/9 for any static star."""
        from tov_solver.units import M_SUN_KM

        return self.mass * M_SUN_KM / self.radius

    def __repr__(self) -> str:
        return (
            f"StellarModel(mass={self.mass:.4f} Msun, radius={self.radius:.4f} km, "
            f"p_c={self.central_pressure:.4g} MeV/fm^3)"
        )


def solve_tov(
    table: EoSTable,
    central_pressure: float,
    *,
    surface_pressure: float | None = None,
    rtol: float = 1e-9,
    atol: float = 1e-14,
) -> StellarModel:
    """Integrate the TOV equations outwards from a given central pressure.

    Parameters
    ----------
    table
        Equation of state. Only the p -> eps direction is used.
    central_pressure
        MeV/fm^3. Must lie within the tabulated range.
    surface_pressure
        Pressure defining the surface, MeV/fm^3. Defaults to the lowest
        pressure in the table, which is as far out as the equation of state
        can be trusted. Raising it shrinks the star slightly; lowering it
        below the table minimum is refused rather than silently
        extrapolated.
    rtol, atol
        Passed to the integrator. The defaults are tight because the
        maximum mass is a stationary point of M(p_c), so a loose tolerance
        shows up as a visibly ragged sequence long before it shows up in any
        single star.

    Returns
    -------
    StellarModel

    Raises
    ------
    ValueError
        If the central pressure lies outside the table, or if the surface
        pressure is below what the table supports.
    RuntimeError
        If no surface is found within the maximum radius.
    """
    p_min = float(table.pressure[0])
    p_max = float(table.pressure[-1])

    if not p_min <= central_pressure <= p_max:
        msg = (
            f"central pressure {central_pressure:.6g} MeV/fm^3 outside the "
            f"tabulated range [{p_min:.6g}, {p_max:.6g}]"
        )
        raise ValueError(msg)

    p_surface = p_min if surface_pressure is None else float(surface_pressure)
    if p_surface < p_min:
        msg = (
            f"surface pressure {p_surface:.6g} MeV/fm^3 is below the table minimum "
            f"{p_min:.6g}; the equation of state cannot be extrapolated there"
        )
        raise ValueError(msg)
    if p_surface >= central_pressure:
        msg = (
            f"surface pressure {p_surface:.6g} is not below the central pressure "
            f"{central_pressure:.6g}"
        )
        raise ValueError(msg)

    # Everything below is geometrized: km, km^-2.
    eps_floor = mev_fm3_to_inv_km2(float(table.energy_density[0]))
    p_floor_geom = mev_fm3_to_inv_km2(p_min)

    def energy_density(p_geom: float) -> float:
        """eps(p), clamped at the table floor.

        The clamp matters: an adaptive step can probe just below the surface
        pressure before the event fires, and an unclamped table returns NaN
        there, which poisons the step-size controller instead of simply
        ending the star.
        """
        if p_geom <= p_floor_geom:
            return eps_floor
        return float(
            mev_fm3_to_inv_km2(table.energy_density_at_pressure(inv_km2_to_mev_fm3(p_geom)))
        )

    def rhs(r: float, y: Array) -> list[float]:
        p, m = float(y[0]), float(y[1])
        eps = energy_density(p)
        dp_dr = -(eps + p) * (m + _FOUR_PI * r**3 * p) / (r * (r - 2.0 * m))
        dm_dr = _FOUR_PI * r**2 * eps
        return [dp_dr, dm_dr]

    def surface(r: float, y: Array) -> float:
        return float(y[0]) - p_surface_geom

    surface.terminal = True  # type: ignore[attr-defined]
    surface.direction = -1.0  # type: ignore[attr-defined]

    p_c = mev_fm3_to_inv_km2(central_pressure)
    eps_c = energy_density(p_c)
    p_surface_geom = mev_fm3_to_inv_km2(p_surface)

    # Series expansion from the centre out to the starting offset.
    r0 = _START_RADIUS
    m0 = (_FOUR_PI / 3.0) * eps_c * r0**3
    p0 = p_c - (2.0 / 3.0) * np.pi * (eps_c + p_c) * (eps_c + 3.0 * p_c) * r0**2

    solution = solve_ivp(
        rhs,
        (r0, _MAX_RADIUS),
        [p0, m0],
        method="DOP853",
        events=surface,
        rtol=rtol,
        atol=atol,
        dense_output=False,
    )

    if not solution.success:
        msg = f"TOV integration failed: {solution.message}"
        raise RuntimeError(msg)
    if solution.t_events is None or solution.t_events[0].size == 0:
        msg = (
            f"no surface found within {_MAX_RADIUS} km for central pressure "
            f"{central_pressure:.6g} MeV/fm^3"
        )
        raise RuntimeError(msg)

    radius = float(solution.t_events[0][0])
    mass_km = float(solution.y_events[0][0][1])

    return StellarModel(
        radius=radius,
        mass=float(km_to_m_sun(mass_km)),
        central_pressure=central_pressure,
        central_energy_density=float(inv_km2_to_mev_fm3(eps_c)),
        radii=np.asarray(solution.t, dtype=np.float64),
        masses=np.asarray(km_to_m_sun(solution.y[1]), dtype=np.float64),
        pressures=np.asarray(inv_km2_to_mev_fm3(solution.y[0]), dtype=np.float64),
    )


def mass_radius_curve(
    table: EoSTable,
    central_pressures: Array,
    **kwargs: float,
) -> list[StellarModel]:
    """Solve a sequence of stars, one per central pressure.

    Configurations that fail to converge are dropped rather than raising, so
    a sweep that runs off the end of the table still returns the part that
    worked.
    """
    models = []
    for p_c in np.asarray(central_pressures, dtype=np.float64):
        try:
            models.append(solve_tov(table, float(p_c), **kwargs))
        except (ValueError, RuntimeError):
            continue
    return models
