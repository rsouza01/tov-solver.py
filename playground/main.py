"""Scratchpad. Not a test, not committed -- hack on this freely.

Run interactively so everything stays loaded afterwards:

    .venv/bin/python -i scratch/play.py

Then keep poking:

    >>> table.pressure_at_potential(1050.0)
    >>> soft.table().validate()
"""

import numpy as np

from tov_solver.eos import Polytrope

# --- look at the columns side by side --------------------------------------

def show(model: Polytrope, densities: np.ndarray) -> None:
    """Print the full thermodynamic state at the given densities."""
    t = model.table(densities)
    print(f"\ngamma={model.gamma}  kappa={model.kappa}")
    print(f"{'n':>8} {'eps':>12} {'p':>12} {'mu':>10} {'c_s^2':>9}")
    for n, eps, p, mu, cs2 in zip(
        t.baryon_density,
        t.energy_density,
        t.pressure,
        t.chemical_potential,
        t.sound_speed_squared,
        strict=True,
    ):
        print(f"{n:8.3f} {eps:12.3f} {p:12.3f} {mu:10.2f} {cs2:9.5f}")


def main():

    # --- build a couple of models ---------------------------------------------

    soft = Polytrope(gamma=2.0, kappa=100.0)
    stiff = Polytrope(gamma=2.75, kappa=100.0)

    table = soft.table()
    print(f"soft: {len(table)} rows over n = "
        f"{table.baryon_density[0]:.3g} .. {table.baryon_density[-1]:.3g} fm^-3")


    grid = np.linspace(0.1, 1.0, 6)
    show(soft, grid)
    show(stiff, grid)


    # --- interpolation ---------------------------------------------------------

    print("\ninterpolation")
    print("  eps(p=50)      =", table.energy_density_at_pressure(50.0))
    print("  p(mu=1100)     =", table.pressure_at_potential(1100.0))
    print("  out of range   =", table.energy_density_at_pressure(1e9), "(NaN by design)")


    # --- validation, including a deliberate failure ----------------------------

    print("\nvalidation")
    print("  soft  ->", soft.table().validate().ok)

    acausal = Polytrope(gamma=3.0, kappa=2000.0, density_max=5.0).table().validate()
    print("  acausal polytrope ok? ", acausal.ok)
    print("  max c_s^2            =", round(acausal.max_sound_speed_squared, 3))

    try:
        acausal.raise_if_invalid()
    except ValueError as exc:
        print("  raise_if_invalid() ->", exc)


    # --- things worth trying next ----------------------------------------------
    #
    #   * vary kappa at fixed gamma and watch mu at low density -- it should
    #     approach the neutron mass, 939.565 MeV, since eps -> m n there
    #   * find the density where c_s^2 crosses 1 for a given gamma
    #   * feed table(grid) a grid that is not sorted, and read the error
    #   * check that validate() still passes on a very coarse grid (num=10)

 
if __name__ == "__main__":
    main()