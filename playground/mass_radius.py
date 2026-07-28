"""Mass-radius diagrams for polytropic stars.

Run:  .venv/bin/python playground/mass_radius.py

Needs matplotlib, which is not a project dependency:
    .venv/bin/pip install matplotlib
"""

import matplotlib.pyplot as plt
import numpy as np

from tov_solver.eos import Polytrope
from tov_solver.structure import mass_radius_curve

GAMMA = 2.0
KAPPAS = [100.0, 200.0, 300.0, 400.0]

# PSR J0740+6620: Fonseca+ 2021 mass, Riley+ 2021 radius.
J0740 = {"M": 2.08, "dM": 0.07, "R": 12.39, "dR_lo": 0.98, "dR_hi": 1.30}


def sequence(kappa: float):
    """Solve a family of stars spanning the table, return arrays."""
    table = Polytrope(gamma=GAMMA, kappa=kappa, density_max=8.0).table()
    pressures = np.logspace(0.0, np.log10(table.pressure[-1] * 0.98), 120)
    stars = mass_radius_curve(table, pressures)
    return (
        np.array([s.radius for s in stars]),
        np.array([s.mass for s in stars]),
        np.array([s.central_energy_density for s in stars]),
    )


fig, (ax_mr, ax_rho) = plt.subplots(1, 2, figsize=(12.5, 5.4))
colours = plt.cm.viridis(np.linspace(0.15, 0.85, len(KAPPAS)))

for kappa, colour in zip(KAPPAS, colours, strict=True):
    R, M, eps_c = sequence(kappa)
    peak = int(M.argmax())

    # Stable branch: mass still rising with central density.
    # Past the maximum the star is unstable to radial oscillations.
    ax_mr.plot(R[: peak + 1], M[: peak + 1], color=colour, lw=2.2,
               label=f"$K$ = {kappa:.0f}")
    ax_mr.plot(R[peak:], M[peak:], color=colour, lw=1.4, ls=":", alpha=0.75)
    ax_mr.plot(R[peak], M[peak], "o", color=colour, ms=7,
               mec="white", mew=1.2, zorder=5)

    ax_rho.plot(eps_c[: peak + 1], M[: peak + 1], color=colour, lw=2.2)
    ax_rho.plot(eps_c[peak:], M[peak:], color=colour, lw=1.4, ls=":", alpha=0.75)
    ax_rho.plot(eps_c[peak], M[peak], "o", color=colour, ms=7,
                mec="white", mew=1.2, zorder=5)

# --- observational anchor --------------------------------------------------
ax_mr.errorbar(
    J0740["R"], J0740["M"],
    xerr=[[J0740["dR_lo"]], [J0740["dR_hi"]]], yerr=J0740["dM"],
    fmt="*", ms=16, color="crimson", ecolor="crimson",
    elinewidth=1.6, capsize=4, zorder=6, label="PSR J0740+6620",
)

ax_mr.set_xlabel("radius  [km]")
ax_mr.set_ylabel(r"gravitational mass  [$M_\odot$]")
ax_mr.set_title(rf"Mass-radius, $\Gamma$ = {GAMMA}")
ax_mr.set_xlim(6, 17)
ax_mr.set_ylim(0, 2.4)
ax_mr.legend(frameon=False, loc="lower left", fontsize=9)

ax_rho.set_xlabel(r"central energy density  [MeV fm$^{-3}$]")
ax_rho.set_ylabel(r"gravitational mass  [$M_\odot$]")
ax_rho.set_title("Stability: mass vs central density")
ax_rho.set_xscale("log")
ax_rho.set_ylim(0, 2.4)
ax_rho.text(
    0.97, 0.06,
    "solid: stable   dotted: unstable\ncircle: maximum mass",
    transform=ax_rho.transAxes, ha="right", va="bottom", fontsize=9,
)

for ax in (ax_mr, ax_rho):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=0.15, lw=0.6)

fig.tight_layout()
fig.savefig("./playground/mass_radius.png", dpi=160, bbox_inches="tight")
print("wrote mass_radius.png")

# --- numbers ---------------------------------------------------------------
print(f"\n{'K':>6} {'M_max':>8} {'R at max':>10} {'eps_c at max':>14}")
for kappa in KAPPAS:
    R, M, eps_c = sequence(kappa)
    i = int(M.argmax())
    print(f"{kappa:6.0f} {M[i]:8.3f} {R[i]:9.2f} {eps_c[i]:14.1f}")