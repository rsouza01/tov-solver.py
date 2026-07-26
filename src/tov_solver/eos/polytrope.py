"""Relativistic polytrope.

Physically this is a toy: no nuclear physics, no phase structure, no
composition. Its value is that everything about it is known in closed form,
which makes it the oracle the rest of the project is checked against.

Definition
----------
Pressure is a power law in baryon density,

    P = K n^Gamma

and the energy density carries the rest mass plus the internal energy
implied by that power law,

    eps = m n + P / (Gamma - 1)

This is the "rest-mass polytrope" of the neutron-star literature (the form
used for piecewise-polytropic fits), not the simpler ``P = K eps^Gamma``.
The difference matters here: only this version is thermodynamically
complete, so it has a well-defined chemical potential and can therefore
exercise the Gibbs-Duhem machinery in :class:`~tov_solver.eos.base.EoSTable`.

Closed forms worth knowing, all verified in the tests:

    mu     = m + Gamma / (Gamma - 1) * K n^(Gamma - 1)
    c_s^2  = Gamma P / (eps + P)

Causality is not automatic. For Gamma = 2 the sound speed reaches c at a
finite density, so a polytrope pushed high enough will fail
:meth:`~tov_solver.eos.base.EoSTable.validate`. That is a property of the
model, not a bug -- and it makes a convenient test that validation works.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from tov_solver.eos.base import Array, EoSModel

__all__ = ["MASS_NEUTRON", "Polytrope"]

MASS_NEUTRON = 939.565_421
"""Neutron rest mass in MeV (CODATA 2018).

Lives here rather than in :mod:`tov_solver.units` because it is physics
input to a specific model, not a dimensional conversion. Each model owns the
masses it uses; the previous codebase defined the electron mass twice, in
two places, with two different values.
"""


@dataclass(frozen=True)
class Polytrope(EoSModel):
    """P = K n^Gamma with rest mass included in the energy density.

    Parameters
    ----------
    gamma
        Adiabatic index, dimensionless. Must exceed 1; at Gamma = 1 the
        internal-energy term diverges.
    kappa
        Polytropic constant K, in MeV fm^(3 Gamma - 3), so that P comes out
        in MeV/fm^3 with n in fm^-3. Note this is *not* the cgs or
        geometrized K quoted in much of the literature.
    baryon_mass
        Rest mass per baryon, MeV. Defaults to the neutron mass.
    density_min, density_max
        Validity range in fm^-3. Defaults span roughly a hundredth of
        saturation to eight times saturation.
    """

    gamma: float
    kappa: float
    baryon_mass: float = MASS_NEUTRON
    density_min: float = 1e-3
    density_max: float = 1.3

    def __post_init__(self) -> None:
        if self.gamma <= 1.0:
            msg = f"gamma must exceed 1, got {self.gamma}"
            raise ValueError(msg)
        if self.kappa <= 0.0:
            msg = f"kappa must be positive, got {self.kappa}"
            raise ValueError(msg)
        if self.baryon_mass <= 0.0:
            msg = f"baryon_mass must be positive, got {self.baryon_mass}"
            raise ValueError(msg)
        if not 0.0 < self.density_min < self.density_max:
            msg = (
                f"need 0 < density_min < density_max, got ({self.density_min}, {self.density_max})"
            )
            raise ValueError(msg)

    @property
    def density_range(self) -> tuple[float, float]:
        return (self.density_min, self.density_max)

    def pressure(self, baryon_density: Array) -> Array:
        """P = K n^Gamma."""
        n = np.asarray(baryon_density, dtype=np.float64)
        return np.asarray(self.kappa * n**self.gamma, dtype=np.float64)

    def energy_density(self, baryon_density: Array) -> Array:
        """eps = m n + P / (Gamma - 1)."""
        n = np.asarray(baryon_density, dtype=np.float64)
        rest_mass = self.baryon_mass * n
        internal = self.pressure(n) / (self.gamma - 1.0)
        return np.asarray(rest_mass + internal, dtype=np.float64)

    # --- closed forms, for testing and for cheap evaluation ---------------

    def chemical_potential(self, baryon_density: Array) -> Array:
        """mu = m + Gamma / (Gamma - 1) K n^(Gamma - 1), exactly."""
        n = np.asarray(baryon_density, dtype=np.float64)
        return np.asarray(
            self.baryon_mass
            + self.gamma / (self.gamma - 1.0) * self.kappa * n ** (self.gamma - 1.0),
            dtype=np.float64,
        )

    def sound_speed_squared(self, baryon_density: Array) -> Array:
        """c_s^2 = Gamma P / (eps + P), exactly."""
        n = np.asarray(baryon_density, dtype=np.float64)
        P = self.pressure(n)
        return np.asarray(self.gamma * P / (self.energy_density(n) + P), dtype=np.float64)
