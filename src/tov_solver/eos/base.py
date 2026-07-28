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

The independent check that survives this is the first law at zero
temperature, deps/dn = mu, which :meth:`EoSTable.validate` evaluates
numerically. Given the Euler relation it is equivalent to Gibbs-Duhem,
dp/dmu = n, but far better conditioned: in the crust the chemical potential
barely moves -- a few MeV above the neutron mass across two decades in
pressure -- so differentiating with respect to it is hopeless, while
differentiating the energy density with respect to the baryon density is
not. On tabulated data the two forms differ by an order of magnitude in
accuracy for identical physics.

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

# Tolerance on the median first-law residual. Loose enough to absorb the
# quadrature error of a tabulated equation of state with real phase
# structure -- the SWRDP tables sit near 1e-3 -- and tight enough that a
# table whose columns genuinely disagree cannot pass. An analytic model
# clears it by five orders of magnitude.
FIRST_LAW_TOLERANCE = 1e-2

# PCHIP needs four points to be well defined.
_MIN_SAMPLES = 4


@dataclass(frozen=True)
class ValidationReport:
    """Outcome of the thermodynamic checks on a table.

    What the first-law residual does and does not catch is worth knowing
    precisely, since it is easy to over-trust. Writing the residual for a
    table whose energy density has been scaled by a constant c gives

        |deps/dn / mu - 1|  =  p (c - 1) / (c eps + p)

    so the check is weighted by p/eps and is nearly blind to a uniform
    rescaling in the crust, where that ratio is 1e-4. Rescaling the density
    column is invisible to it outright: Euler absorbs the factor exactly.

    What it does catch, decisively, is any error in the *shape* of eps(n)
    relative to p(n) -- a wrong exponent, a tilt, a misaligned column, a
    table stitched together from two sources. Those are the failures that
    are otherwise invisible.

    Gross unit errors are better caught by asserting a physical scale: the
    chemical potential of nuclear matter at low density must approach the
    nucleon mass, and no thermodynamic identity is needed to notice that it
    does not.
    """

    max_sound_speed_squared: float
    min_sound_speed_squared: float
    median_first_law_residual: float
    max_first_law_residual: float

    @property
    def is_causal(self) -> bool:
        """True if 0 <= c_s^2 <= 1 everywhere."""
        return self.min_sound_speed_squared >= 0.0 and self.max_sound_speed_squared <= 1.0

    @property
    def is_consistent(self) -> bool:
        """True if deps/dn = mu holds to within tolerance.

        Judged on the median rather than the maximum. A column that is
        wrong -- misread, mis-scaled, in the wrong units -- is wrong
        everywhere, so the median catches it decisively. An isolated spike
        is the signature of a kink, which is what a hyperon threshold or a
        first-order transition genuinely looks like, and rejecting a table
        for having correct physics would be the wrong call.
        Use :attr:`max_first_law_residual` to find those kinks.
        """
        return self.median_first_law_residual <= FIRST_LAW_TOLERANCE

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
                f"thermodynamically inconsistent table: median |deps/dn / mu - 1| = "
                f"{self.median_first_law_residual:.4g}, "
                f"tolerance {FIRST_LAW_TOLERANCE:.4g}"
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
    def _log_energy_density_of_log_density(self) -> PchipInterpolator:
        return PchipInterpolator(np.log(self.baryon_density), np.log(self.energy_density))

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

        # deps/dn evaluated in log-log, where both quantities are smooth over
        # the several decades a real equation of state spans:
        #     deps/dn = (eps/n) * dln(eps)/dln(n)
        log_slope = self._log_energy_density_of_log_density.derivative()(
            np.log(self.baryon_density)
        )
        derivative = self.energy_density / self.baryon_density * log_slope
        residual = np.abs(derivative / self.chemical_potential - 1.0)

        return ValidationReport(
            max_sound_speed_squared=float(np.max(cs2)),
            min_sound_speed_squared=float(np.min(cs2)),
            median_first_law_residual=float(np.median(residual)),
            max_first_law_residual=float(np.max(residual)),
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

        The endpoints are pinned exactly rather than left to ``logspace``.
        A round trip through log10 and back can land a fraction of an ulp
        outside the requested interval, which a tabulated model reports as
        NaN -- correctly, since it was asked to extrapolate -- and which
        then fails table construction for no physical reason.
        """
        low, high = self.density_range
        grid = np.logspace(np.log10(low), np.log10(high), num, dtype=np.float64)
        grid[0], grid[-1] = low, high
        return grid

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