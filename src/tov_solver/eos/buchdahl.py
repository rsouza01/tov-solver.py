"""Buchdahl's analytic solution -- the oracle the integrator is checked against.

Buchdahl (1967) found a closed-form interior solution to Einstein's
equations for the one-parameter equation of state

    eps = 12 sqrt(p_star p) - 5 p

It has no nuclear physics in it whatsoever. Its value is that the resulting
stellar structure is known exactly, so a numerical TOV integrator can be
compared against truth rather than against itself.

For small p the relation reduces to eps = 12 sqrt(p_star p), which inverts
to p proportional to eps^2 -- the Newtonian n = 1 polytrope. Buchdahl's
solution is its relativistic generalisation, which makes it a natural
companion to :class:`~tov_solver.eos.polytrope.Polytrope`.

Exact results
-------------
Writing beta = M/R for the compactness, the surface sits at the first zero
of sin(A r')/(A r') with A^2 = 288 pi p_star / (1 - 2 beta), giving

    R     = (1 - beta) sqrt(pi / (288 p_star (1 - 2 beta)))
    M     = beta R
    eps_c = 72 p_star beta (1 - 5 beta / 2)

in geometrized units. :func:`exact_mass_radius` returns the first two in km
and solar masses.

Baryon density
--------------
The equation of state is specified as eps(p), with no baryon number in
sight. :class:`~tov_solver.eos.base.EoSTable` wants one, so it is
reconstructed from the first law at zero temperature,

    dn / n = d eps / (eps + p)

integrated along the pressure grid. The normalisation is arbitrary and
irrelevant -- nothing in the TOV equations depends on it -- but the
resulting column is thermodynamically consistent, which means the Buchdahl
table also exercises the Gibbs-Duhem check in
:meth:`~tov_solver.eos.base.EoSTable.validate`.

Validity
--------
Two distinct limits, and they are not the same number:

* d eps / d p vanishes at p = 36 p_star / 25, above which eps(p) turns over
  and stops being monotonic. Tables are refused there outright.

* d eps / d p = 6 sqrt(p_star / p) - 5, so the sound speed reaches c at
  exactly p = p_star. Above that the solution still solves Einstein's
  equations, but with superluminal sound. Tables are *not* refused --
  :meth:`~tov_solver.eos.base.EoSTable.validate` is left to report it -- so
  the equation of state remains usable as a pure integrator benchmark.

Since the central pressure of a Buchdahl star is p_c = 36 beta^2 p_star,
the causal branch corresponds to beta <= 1/6.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property

import numpy as np
from scipy.integrate import cumulative_trapezoid

from tov_solver.eos.base import Array, EoSTable
from tov_solver.units import inv_km2_to_mev_fm3, km_to_m_sun, mev_fm3_to_inv_km2

__all__ = ["Buchdahl", "exact_mass_radius"]

# eps(p) turns over at this multiple of p_star.
_TURNOVER = 36.0 / 25.0


def exact_mass_radius(p_star: float, beta: float) -> tuple[float, float]:
    """Closed-form mass and radius of a Buchdahl star.

    Parameters
    ----------
    p_star
        The equation-of-state parameter, MeV/fm^3.
    beta
        Compactness M/R, dimensionless. Must lie in (0, 1/2).

    Returns
    -------
    mass, radius
        Solar masses and km.
    """
    if not 0.0 < beta < 0.5:
        msg = f"compactness must lie in (0, 1/2), got {beta}"
        raise ValueError(msg)

    p_geom = mev_fm3_to_inv_km2(p_star)
    radius = (1.0 - beta) * np.sqrt(np.pi / (288.0 * p_geom * (1.0 - 2.0 * beta)))
    mass_km = beta * radius
    return float(km_to_m_sun(mass_km)), float(radius)


def central_energy_density(p_star: float, beta: float) -> float:
    """Exact central energy density, MeV/fm^3."""
    eps_geom = 72.0 * mev_fm3_to_inv_km2(p_star) * beta * (1.0 - 2.5 * beta)
    return float(inv_km2_to_mev_fm3(eps_geom))


def central_pressure(p_star: float, beta: float) -> float:
    """Exact central pressure, MeV/fm^3.

    Inverts eps = 12 sqrt(p_star p) - 5 p on the branch below the turnover.
    """
    eps_c = central_energy_density(p_star, beta)
    discriminant = 144.0 * p_star - 20.0 * eps_c
    if discriminant < 0.0:
        msg = f"no physical central pressure for p_star={p_star}, beta={beta}"
        raise ValueError(msg)
    root = (12.0 * np.sqrt(p_star) - np.sqrt(discriminant)) / 10.0
    return float(root**2)


@dataclass(frozen=True)
class Buchdahl:
    """The Buchdahl equation of state, sampled onto a table.

    Parameters
    ----------
    p_star
        Equation-of-state parameter, MeV/fm^3.
    pressure_min, pressure_max
        Range to tabulate, MeV/fm^3. ``pressure_max`` must stay below the
        turnover at 36 p_star / 25.
    num
        Number of log-spaced samples.
    """

    p_star: float
    pressure_min: float
    pressure_max: float
    num: int = 4000

    def __post_init__(self) -> None:
        if self.p_star <= 0.0:
            msg = f"p_star must be positive, got {self.p_star}"
            raise ValueError(msg)
        if not 0.0 < self.pressure_min < self.pressure_max:
            msg = (
                f"need 0 < pressure_min < pressure_max, got "
                f"({self.pressure_min}, {self.pressure_max})"
            )
            raise ValueError(msg)
        turnover = _TURNOVER * self.p_star
        if self.pressure_max >= turnover:
            msg = (
                f"pressure_max {self.pressure_max:.6g} reaches the turnover at "
                f"36 p_star / 25 = {turnover:.6g}, above which eps(p) is not monotonic"
            )
            raise ValueError(msg)

    def energy_density(self, pressure: Array) -> Array:
        """eps = 12 sqrt(p_star p) - 5 p."""
        p = np.asarray(pressure, dtype=np.float64)
        return np.asarray(12.0 * np.sqrt(self.p_star * p) - 5.0 * p, dtype=np.float64)

    @cached_property
    def _pressure_grid(self) -> Array:
        return np.logspace(
            np.log10(self.pressure_min),
            np.log10(self.pressure_max),
            self.num,
            dtype=np.float64,
        )

    @cached_property
    def _baryon_density(self) -> Array:
        """Reconstructed from d ln n = d eps / (eps + p), normalisation free.

        Integrated against log p rather than p. In the linear variable the
        integrand diverges as 1/(2p) at the low-pressure end, and the
        quadrature error there is large enough to swamp the true variation of
        mu = (eps + p)/n -- which is nearly flat, since both eps and n go as
        sqrt(p). The result is a chemical potential that drifts *downwards*
        by a part in 10^8, and a table that is rejected as non-monotonic.

        In log p the integrand is

            d ln n / d ln p = p (d eps / d p) / (eps + p)

        which tends smoothly to 1/2 as p -> 0 and integrates cleanly.
        """
        p = self._pressure_grid
        eps = self.energy_density(p)
        deps_dp = 6.0 * np.sqrt(self.p_star / p) - 5.0
        integrand = p * deps_dp / (eps + p)
        log_n = cumulative_trapezoid(integrand, np.log(p), initial=0.0)
        return np.asarray(np.exp(log_n), dtype=np.float64)

    def table(self) -> EoSTable:
        """Sample onto an :class:`~tov_solver.eos.base.EoSTable`."""
        p = self._pressure_grid
        return EoSTable(
            baryon_density=self._baryon_density,
            energy_density=self.energy_density(p),
            pressure=p,
        )
