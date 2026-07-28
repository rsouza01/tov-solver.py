"""SWRDP: tabulated hyperonic equation of state.

The sigma-omega-rho-delta-phi model of Gomes, Dexheimer, Schramm &
Vasconcellos (ApJ 808, 8, 2015), in the scalar version where many-body
forces enter through a single parameter zeta. Smaller zeta means a stiffer
equation of state: it lowers the nucleon effective mass at saturation and
raises the compressibility modulus, and the published maximum masses run
from 2.15 solar masses at zeta = 0.040 down to 1.83 at zeta = 0.085.

This module does not solve the mean-field equations. It reads tables
produced by that solver and presents them through
:class:`~tov_solver.eos.base.EoSModel`, so a tabulated equation of state is
indistinguishable to callers from a computed one.

Interpolation in zeta
---------------------
Only five parametrizations are tabulated. Since a posterior over zeta needs
a continuous model, values between them are interpolated: log-log in
density within each file, then linearly in log of the quantity between the
two bracketing files. Leave-one-out tests -- predicting a tabulated zeta
from its two *neighbours*, so twice the real bracket width -- give median
pressure errors of a few parts in a thousand, and log-space beats linear by
two to three times.

The caveat worth remembering: hyperon thresholds move with zeta, so at a
fixed density the two bracketing files may differ in composition. The
interpolant smears the kinks where new species appear. For a smooth
quantity like the maximum mass this is harmless; for anything that depends
on the derivative structure near a threshold it is not.

Extrapolation beyond the tabulated range is refused.

File format
-----------
Nine whitespace-separated columns. The first three describe the magnetic
case, which has pressure anisotropy, and column two is NaN throughout at
zero field. The columns used here are

    index 3   energy density      MeV/fm^3
    index 4   pressure            MeV/fm^3
    index 5   baryon density      fm^-3
    index 7   chemical potential  MeV        (cross-check only)

Column 7 is *not* loaded into the table. The Euler relation supplies the
chemical potential instead, and the file's own column is used only to
verify it -- the two agree to two parts in 10^5 across all five files.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import NamedTuple

import numpy as np
from scipy.interpolate import PchipInterpolator

from tov_solver.eos.base import Array, EoSModel

__all__ = ["DEFAULT_DATA_DIR", "SWRDP", "available_zeta"]

DEFAULT_DATA_DIR = Path(__file__).resolve().parents[3] / "data" / "swrdp"
"""Repository data directory.

Resolved relative to the source tree, which works for an editable install.
Packaging the tables inside ``tov_solver`` and using ``importlib.resources``
would be more robust; that is a change to make when the project is first
built as a wheel.
"""

_FILENAME = re.compile(r"^eos_swrdp_l(?P<zeta>\d{4})_L(?P<slope>\d+)a(?P<sym>\d+)\.dat$")

_COL_ENERGY_DENSITY = 3
_COL_PRESSURE = 4
_COL_BARYON_DENSITY = 5
_COL_CHEMICAL_POTENTIAL = 7


def _discover(data_dir: Path) -> dict[float, Path]:
    """Map zeta to file path for every table in a directory."""
    found: dict[float, Path] = {}
    for path in sorted(data_dir.glob("*.dat")):
        match = _FILENAME.match(path.name)
        if match is not None:
            found[int(match["zeta"]) / 1000.0] = path
    if not found:
        msg = f"no SWRDP tables matching 'eos_swrdp_lNNNN_LNNaNN.dat' in {data_dir}"
        raise FileNotFoundError(msg)
    return found


def available_zeta(data_dir: Path = DEFAULT_DATA_DIR) -> tuple[float, ...]:
    """Tabulated zeta values, ascending."""
    return tuple(sorted(_discover(data_dir)))


@lru_cache(maxsize=32)
def _load(path_str: str) -> tuple[Array, Array, Array, Array]:
    """Read one table: density, energy density, pressure, tabulated mu.

    Cached, because a sampler constructs many models over the same handful
    of files and reading ten thousand rows each time would dominate the
    likelihood evaluation.
    """
    raw = np.loadtxt(path_str)
    density = np.asarray(raw[:, _COL_BARYON_DENSITY], dtype=np.float64)
    order = np.argsort(density)
    return (
        density[order],
        np.asarray(raw[order, _COL_ENERGY_DENSITY], dtype=np.float64),
        np.asarray(raw[order, _COL_PRESSURE], dtype=np.float64),
        np.asarray(raw[order, _COL_CHEMICAL_POTENTIAL], dtype=np.float64),
    )


class _Interpolants(NamedTuple):
    """Log-log interpolants for one table, plus its density range."""

    energy_density: PchipInterpolator
    pressure: PchipInterpolator
    density_min: float
    density_max: float


@lru_cache(maxsize=32)
def _interpolants(path_str: str) -> _Interpolants:
    """Log-log PCHIP interpolants for eps(n) and p(n), plus the density range.

    Log-log because both quantities span several decades; PCHIP because it
    preserves monotonicity, and a cubic spline would manufacture spurious
    extrema at the hyperon thresholds -- exactly where the curvature is
    largest.
    """
    density, energy_density, pressure, _ = _load(path_str)
    log_n = np.log(density)
    return _Interpolants(
        energy_density=PchipInterpolator(log_n, np.log(energy_density), extrapolate=False),
        pressure=PchipInterpolator(log_n, np.log(pressure), extrapolate=False),
        density_min=float(density[0]),
        density_max=float(density[-1]),
    )


@dataclass(frozen=True)
class SWRDP(EoSModel):
    """Tabulated SWRDP equation of state at a given zeta.

    Parameters
    ----------
    zeta
        Many-body force parameter. Must lie within the tabulated range;
        values between tabulated points are interpolated.
    data_dir
        Directory holding the ``.dat`` tables.

    Examples
    --------
    Exact tabulated value, no interpolation::

        SWRDP(zeta=0.040).table()

    Interpolated between the 0.040 and 0.049 tables::

        SWRDP(zeta=0.045).table()
    """

    zeta: float
    data_dir: Path = DEFAULT_DATA_DIR
    _paths: dict[float, Path] = field(init=False, repr=False, compare=False, default_factory=dict)

    def __post_init__(self) -> None:
        paths = _discover(self.data_dir)
        object.__setattr__(self, "_paths", paths)

        tabulated = sorted(paths)
        if not tabulated[0] <= self.zeta <= tabulated[-1]:
            msg = (
                f"zeta = {self.zeta} outside the tabulated range "
                f"[{tabulated[0]}, {tabulated[-1]}]; extrapolation is not supported"
            )
            raise ValueError(msg)

    # --- bracketing -------------------------------------------------------

    @property
    def _bracket(self) -> tuple[float, float]:
        """The two tabulated zeta values spanning this one.

        Returns the same value twice when zeta sits exactly on a table, so
        that a tabulated parametrization is served from its file untouched
        rather than through an interpolant.
        """
        tabulated = sorted(self._paths)
        for value in tabulated:
            if np.isclose(self.zeta, value, rtol=0.0, atol=1e-12):
                return (value, value)
        index = int(np.searchsorted(tabulated, self.zeta))
        return (tabulated[index - 1], tabulated[index])

    @property
    def is_interpolated(self) -> bool:
        """True when zeta falls between tabulated values."""
        low, high = self._bracket
        return low != high

    @property
    def density_range(self) -> tuple[float, float]:
        """Intersection of the bracketing files' ranges, fm^-3.

        The tables do not all start at the same density -- the zeta = 0.040
        file reaches down to 4.2e-3 fm^-3 while zeta = 0.085 stops at
        2.7e-2 -- so an interpolated model is only valid where both
        bracketing files have data.
        """
        low, high = self._bracket
        spans = [_interpolants(str(self._paths[z])) for z in {low, high}]
        return (max(s.density_min for s in spans), min(s.density_max for s in spans))

    # --- the EoSModel interface -------------------------------------------

    def _evaluate(self, baryon_density: Array, *, pressure: bool) -> Array:
        """Interpolate one column, blending the bracketing files in log space."""
        n = np.asarray(baryon_density, dtype=np.float64)
        log_n = np.log(n)
        low, high = self._bracket

        span_low = _interpolants(str(self._paths[low]))
        log_low = (span_low.pressure if pressure else span_low.energy_density)(log_n)
        if low == high:
            return np.asarray(np.exp(log_low), dtype=np.float64)

        span_high = _interpolants(str(self._paths[high]))
        log_high = (span_high.pressure if pressure else span_high.energy_density)(log_n)
        weight = (self.zeta - low) / (high - low)
        return np.asarray(np.exp((1.0 - weight) * log_low + weight * log_high), dtype=np.float64)

    def default_grid(self, num: int = 500) -> Array:
        """The tables' own density grid, not a resampling of it.

        ``num`` is ignored at a tabulated zeta. The files carry some ten
        thousand points and already resolve the crust, where the equation of
        state has structure that a few hundred log-spaced points cannot
        follow: resampling there perturbs eps and p just enough that the
        derived chemical potential -- which climbs only a few MeV across two
        decades of pressure -- stops being monotonic, and the table is
        rejected. Using the native grid is both exact and cheaper.

        When interpolating between two files, the denser of the two grids is
        used, restricted to their common range.
        """
        low, high = self._bracket
        grids = [_load(str(self._paths[z]))[0] for z in {low, high}]
        grid = max(grids, key=len)
        first, last = self.density_range
        return np.asarray(grid[(grid >= first) & (grid <= last)], dtype=np.float64)

    def energy_density(self, baryon_density: Array) -> Array:
        """Energy density, MeV/fm^3. NaN outside the tabulated range."""
        return self._evaluate(baryon_density, pressure=False)

    def pressure(self, baryon_density: Array) -> Array:
        """Pressure, MeV/fm^3. NaN outside the tabulated range."""
        return self._evaluate(baryon_density, pressure=True)

    # --- provenance -------------------------------------------------------

    def tabulated_chemical_potential(self, baryon_density: Array) -> Array:
        """The file's own mu column, for cross-checking the Euler relation.

        Only defined for a tabulated zeta; interpolating a quantity that the
        table already determines twice over would defeat the purpose of the
        check.
        """
        low, high = self._bracket
        if low != high:
            msg = "tabulated chemical potential is only available at a tabulated zeta"
            raise ValueError(msg)
        density, _, _, mu = _load(str(self._paths[low]))
        interpolant = PchipInterpolator(np.log(density), mu, extrapolate=False)
        return np.asarray(interpolant(np.log(baryon_density)), dtype=np.float64)
