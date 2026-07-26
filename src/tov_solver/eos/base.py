"""Common interface for every equation of state.

Two types live here, and the split matters:

``EoSModel``
    The *physics*. Given parameters, produce pressure and energy density as
    functions of baryon density. Whether it root-solves a mean-field theory
    or interpolates a file on disk is nobody else's business.

``EoSTable``
    The *artifact*. Sampled arrays plus monotone interpolators. This is what
    the phase-transition constructions and the TOV integrator consume, and
    it is the only thing they need to understand.

Thermodynamics
--------------
At zero temperature with a single conserved charge the Euler relation reads

    eps = -p + mu n

so the baryon chemical potential is fully determined by the other three
quantities. ``EoSTable`` therefore *derives* it rather than storing it. A
table carrying its own mu column can drift out of step with its p and eps;
a derived one cannot.

The independent check that survives this is the Gibbs-Duhem relation,
dp/dmu = n, which :meth:`EoSTable.validate` evaluates numerically.

Naming
------
Public attributes and arguments are spelled out (``baryon_density``, not
``n_B``); local variables inside the physics use the notation of the papers
(``n``, ``mu``, ``eps``, ``P``). The linter enforces the first and permits
the second.

Units are those of :mod:`tov_solver.units`: MeV/fm^3 for pressure and energy
density, fm^-3 for number density, MeV for chemical potential.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import cached_property

import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import PchipInterpolator

__all__ = ["EoSModel", "EoSTable", "ValidationReport"]

Array = NDArray[np.float64]

# Tolerance on the numerical Gibbs-Duhem residual: loose enough to absorb
# interpolation error on a reasonable grid, tight enough to catch a table
# whose columns genuinely disagree.
GIBBS_DUHEM_TOLERANCE = 1e-3

# PCHIP needs four points to be well defined.
_MIN_SAMPLES = 4


@dataclass(frozen=True)
class ValidationReport:
    """Outcome of the thermodynamic checks on a table."""

    max_sound_speed_squared: float
    min_sound_speed_squared: float
    max_gibbs_duhem_residual: float

    @property
    def is_causal(self) -> bool:
        """True if 0 <= c_s^2 <= 1 everywhere."""
        return self.min_sound_speed_squared >= 0.0 and self.max_sound_speed_squared <= 1.0

    @property
    def is_consistent(self) -> bool:
        """True if dp/dmu = n holds to within tolerance."""
        return self.max_gibbs_duhem_residual <= GIBBS_DUHEM_TOLERANCE

    @property
    def ok(self) -> bool:
        return self.is_causal and self.is_consistent

    def raise_if_invalid(self) -> None:
        """Raise :class:`ValueError` describing the first failure found."""
        if not self.is_causal:
            msg = (
                f"acausal or unstable equation of state: c_s^2 in "
                f"[{self.min_sound_speed_squared:.4g}, {self.max_sound_speed_squared:.4g}], "
                f"expected [0, 1]"
            )
            raise ValueError(msg)
        if not self.is_consistent:
            msg = (
                f"thermodynamically inconsistent table: max |dp/dmu / n - 1| = "
                f"{self.max_gibbs_duhem_residual:.4g}, "
                f"tolerance {GIBBS_DUHEM_TOLERANCE:.4g}"
            )
            raise ValueError(msg)


@dataclass(frozen=True)
class EoSTable:
    """A sampled equation of state.

    Parameters
    ----------
    baryon_density
        fm^-3. Strictly increasing.
    energy_density
        Total energy density, MeV/fm^3. Strictly increasing, positive.
    pressure
        MeV/fm^3. Strictly increasing, positive.

    The invariants are enforced on construction because every consumer
    assumes them: monotone interpolation, the inversion p -> eps the TOV
    integrator needs, and the bracketing used by the phase-transition
    constructions all fail silently without them.

    Positivity of the pressure is part of the contract. Models whose raw
    output dips negative at low density -- MFTQCD does, thanks to the bag
    term -- must trim that region before emitting a table.
    """

    baryon_density: Array
    energy_density: Array
    pressure: Array

    def __post_init__(self) -> None:
        columns = {
            "baryon_density": self.baryon_density,
            "energy_density": self.energy_density,
            "pressure": self.pressure,
        }

        for name, column in columns.items():
            if column.ndim != 1:
                msg = f"{name} must be one-dimensional, got shape {column.shape}"
                raise ValueError(msg)
            if not np.all(np.isfinite(column)):
                msg = f"{name} contains non-finite values"
                raise ValueError(msg)
            if np.any(column <= 0.0):
                msg = f"{name} must be strictly positive"
                raise ValueError(msg)
            if np.any(np.diff(column) <= 0.0):
                msg = f"{name} must be strictly increasing"
                raise ValueError(msg)

        lengths = {name: column.size for name, column in columns.items()}
        if len(set(lengths.values())) != 1:
            msg = f"column lengths disagree: {lengths}"
            raise ValueError(msg)
        if self.baryon_density.size < _MIN_SAMPLES:
            msg = (
                f"need at least {_MIN_SAMPLES} samples for monotone interpolation, "
                f"got {self.baryon_density.size}"
            )
            raise ValueError(msg)

    def __len__(self) -> int:
        return int(self.baryon_density.size)

    # --- derived thermodynamics -------------------------------------------

    @cached_property
    def chemical_potential(self) -> Array:
        """Baryon chemical potential, MeV. Euler relation, not a stored column."""
        return (self.energy_density + self.pressure) / self.baryon_density

    @cached_property
    def sound_speed_squared(self) -> Array:
        """c_s^2 = dp/deps, in units of c^2.

        Evaluated as (dp/dn)/(deps/dn) rather than by differentiating an
        interpolant of p(eps), which is better conditioned when the two
        columns span very different dynamic ranges.
        """
        dp_dn = self._pressure_of_density.derivative()(self.baryon_density)
        deps_dn = self._energy_density_of_density.derivative()(self.baryon_density)
        return np.asarray(dp_dn / deps_dn, dtype=np.float64)

    # --- interpolators ----------------------------------------------------
    #
    # PCHIP rather than a cubic spline: it is monotonicity-preserving, so it
    # cannot manufacture a spurious extremum between samples. Cubic splines
    # can, and they do it exactly where a first-order phase transition puts a
    # kink -- producing negative c_s^2 out of clean input data.

    @cached_property
    def _pressure_of_density(self) -> PchipInterpolator:
        return PchipInterpolator(self.baryon_density, self.pressure, extrapolate=False)

    @cached_property
    def _energy_density_of_density(self) -> PchipInterpolator:
        return PchipInterpolator(self.baryon_density, self.energy_density, extrapolate=False)

    @cached_property
    def _pressure_of_energy_density(self) -> PchipInterpolator:
        return PchipInterpolator(self.energy_density, self.pressure, extrapolate=False)

    @cached_property
    def _energy_density_of_pressure(self) -> PchipInterpolator:
        return PchipInterpolator(self.pressure, self.energy_density, extrapolate=False)

    @cached_property
    def _pressure_of_potential(self) -> PchipInterpolator:
        return PchipInterpolator(self.chemical_potential, self.pressure, extrapolate=False)

    @cached_property
    def _energy_density_of_potential(self) -> PchipInterpolator:
        return PchipInterpolator(self.chemical_potential, self.energy_density, extrapolate=False)

    def pressure_at_energy_density(self, energy_density: Array | float) -> Array:
        """p(eps). NaN outside the tabulated range."""
        return np.asarray(self._pressure_of_energy_density(energy_density), dtype=np.float64)

    def energy_density_at_pressure(self, pressure: Array | float) -> Array:
        """eps(p). The direction the TOV integrator needs."""
        return np.asarray(self._energy_density_of_pressure(pressure), dtype=np.float64)

    def pressure_at_potential(self, chemical_potential: Array | float) -> Array:
        """p(mu). The direction the phase-transition constructions need."""
        return np.asarray(self._pressure_of_potential(chemical_potential), dtype=np.float64)

    def energy_density_at_potential(self, chemical_potential: Array | float) -> Array:
        """eps(mu)."""
        return np.asarray(self._energy_density_of_potential(chemical_potential), dtype=np.float64)

    # --- validation -------------------------------------------------------

    def validate(self) -> ValidationReport:
        """Check causality and thermodynamic consistency.

        Returns a report rather than raising, so a sampler can reject a
        parameter draw without exception handling. Use
        :meth:`ValidationReport.raise_if_invalid` where a failure would be a
        programming error rather than a bad sample.
        """
        cs2 = self.sound_speed_squared

        density_from_gibbs_duhem = self._pressure_of_potential.derivative()(self.chemical_potential)
        residual = np.abs(density_from_gibbs_duhem / self.baryon_density - 1.0)

        return ValidationReport(
            max_sound_speed_squared=float(np.max(cs2)),
            min_sound_speed_squared=float(np.min(cs2)),
            max_gibbs_duhem_residual=float(np.max(residual)),
        )


class EoSModel(ABC):
    """An equation of state that can be sampled onto a table.

    Implementations supply pressure and energy density as functions of
    baryon density, plus the range over which they are valid. Everything
    else -- chemical potentials, interpolation, validation -- is inherited.

    The abstraction deliberately does not distinguish computed models from
    tabulated ones. ``MFTQCD`` root-solves beta equilibrium; ``SWRDP``
    interpolates a file. Both are ``EoSModel``, and the Maxwell construction
    that joins them cannot tell the difference.
    """

    @property
    @abstractmethod
    def density_range(self) -> tuple[float, float]:
        """Inclusive (min, max) baryon density in fm^-3 where the model holds."""

    @abstractmethod
    def pressure(self, baryon_density: Array) -> Array:
        """Pressure in MeV/fm^3 at the given baryon densities."""

    @abstractmethod
    def energy_density(self, baryon_density: Array) -> Array:
        """Total energy density in MeV/fm^3 at the given baryon densities."""

    def default_grid(self, num: int = 500) -> Array:
        """Log-spaced densities spanning :attr:`density_range`.

        Log spacing because an equation of state covers several decades and
        the interesting structure sits at the low end.
        """
        low, high = self.density_range
        return np.logspace(np.log10(low), np.log10(high), num, dtype=np.float64)

    def table(self, baryon_density: Array | None = None) -> EoSTable:
        """Sample onto an :class:`EoSTable`, using :meth:`default_grid` if omitted."""
        grid = (
            self.default_grid()
            if baryon_density is None
            else np.asarray(baryon_density, dtype=np.float64)
        )
        return EoSTable(
            baryon_density=grid,
            energy_density=self.energy_density(grid),
            pressure=self.pressure(grid),
        )
