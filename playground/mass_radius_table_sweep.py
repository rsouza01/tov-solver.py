"""Mass-radius by sweeping the equation-of-state table row by row.

This mirrors the structure of the original Fortran code. The table serves
two purposes at once:

  1. it is interpolated between rows during the TOV integration, and
  2. each row is itself used as a central density for one star.

So the loop is simply "for every row in the table, build a star". No
separate grid of central pressures has to be invented, and the resolution
of the mass-radius curve is inherited directly from the resolution of the
equation of state.

Run:  .venv/bin/python playground/mass_radius_table_sweep.py
"""

import matplotlib.pyplot as plt
import numpy as np

from tov_solver.eos import SWRDP, Polytrope
from tov_solver.eos.base import EoSTable
from tov_solver.structure import solve_tov


def sweep(table: EoSTable, stride: int = 1):
    """One star per table row.

    Parameters
    ----------
    table
        The equation of state. Row ``i`` supplies the central pressure.
    stride
        Take every ``stride``-th row. The SWRDP tables carry ten thousand
        rows and one TOV integration each would take minutes, so for those
        a stride of fifty or so is sensible. The polytrope's five hundred
        rows are fine at stride one.

    Returns
    -------
    central_density, mass, radius
        Arrays over the rows that produced a valid star.
    """
    n_c, masses, radii = [], [], []

    # Row zero is the surface of the table: there is no pressure below it to
    # integrate down to, so a star centred there has nowhere to go.
    for i in range(1, len(table), stride):
        p_c = float(table.pressure[i])
        try:
            star = solve_tov(table, p_c)
        except (ValueError, RuntimeError):
            continue
        n_c.append(float(table.baryon_density[i]))
        masses.append(star.mass)
        radii.append(star.radius)

    return np.array(n_c), np.array(masses), np.array(radii)


# --------------------------------------------------------------------------
# 1. Polytrope -- 500 rows, sweep all of them
# --------------------------------------------------------------------------

poly = Polytrope(gamma=2.0, kappa=400.0, density_max=3.0).table()
n_poly, m_poly, r_poly = sweep(poly)

peak = int(m_poly.argmax())
print("POLYTROPE  Gamma = 2, K = 400")
print(f"  rows in table        : {len(poly)}")
print(f"  stars built          : {len(m_poly)}")
print(f"  maximum mass         : {m_poly[peak]:.4f} Msun")
print(f"  radius there         : {r_poly[peak]:.3f} km")
print(f"  central density there: {n_poly[peak]:.4f} fm^-3")

print(f"\n  {'row n_c':>10} {'M':>9} {'R':>9}")
for i in range(0, len(n_poly), len(n_poly) // 10):
    mark = "  <- max" if i <= peak < i + len(n_poly) // 10 else ""
    print(f"  {n_poly[i]:10.4f} {m_poly[i]:9.4f} {r_poly[i]:9.3f}{mark}")


# --------------------------------------------------------------------------
# 2. SWRDP -- ten thousand rows, so take every fiftieth
# --------------------------------------------------------------------------

print("\nSWRDP")
print(f"  {'zeta':>6} {'rows':>7} {'stars':>7} {'M_max':>9} {'R at max':>9} {'n_c at max':>11}")

curves = {}
for zeta in (0.040, 0.059, 0.085):
    table = SWRDP(zeta=zeta).table()
    n_c, mass, radius = sweep(table, stride=50)
    curves[zeta] = (n_c, mass, radius)
    i = int(mass.argmax())
    print(f"  {zeta:6.3f} {len(table):7d} {len(mass):7d} "
          f"{mass[i]:9.4f} {radius[i]:9.3f} {n_c[i]:11.4f}")


# --------------------------------------------------------------------------
# plot
# --------------------------------------------------------------------------

fig, (ax_mr, ax_nc) = plt.subplots(1, 2, figsize=(12.5, 5.2))

ax_mr.plot(r_poly, m_poly, color="0.6", lw=1.6, label=r"polytrope $\Gamma$=2, K=400")
ax_nc.plot(n_poly, m_poly, color="0.6", lw=1.6)

for zeta, colour in zip(sorted(curves), ["#1b7837", "#4393c3", "#b2182b"], strict=True):
    n_c, mass, radius = curves[zeta]
    i = int(mass.argmax())
    ax_mr.plot(radius, mass, color=colour, lw=2.0, label=rf"SWRDP $\zeta$={zeta:.3f}")
    ax_mr.plot(radius[i], mass[i], "o", color=colour, ms=6, mec="white")
    ax_nc.plot(n_c, mass, color=colour, lw=2.0)
    ax_nc.plot(n_c[i], mass[i], "o", color=colour, ms=6, mec="white")

ax_mr.set_xlabel("radius  [km]")
ax_mr.set_ylabel(r"mass  [$M_\odot$]")
ax_mr.set_title("one star per table row")
ax_mr.set_xlim(8, 20)
ax_mr.set_ylim(0, 2.4)
ax_mr.legend(frameon=False, fontsize=9, loc="lower left")

ax_nc.set_xlabel(r"central baryon density  [fm$^{-3}$]")
ax_nc.set_ylabel(r"mass  [$M_\odot$]")
ax_nc.set_title("the same stars, against their central density")
ax_nc.set_xscale("log")
ax_nc.set_ylim(0, 2.4)

for ax in (ax_mr, ax_nc):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=0.15, lw=0.6)

fig.tight_layout()
fig.savefig("./playground/mass_radius_table_sweep.png", dpi=160, bbox_inches="tight")
print("\nwrote mass_radius_table_sweep.png")