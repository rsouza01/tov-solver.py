import numpy as np

from tov_solver.utils.plotter import EoSPlotter
from tov_solver.utils.exporter import EoSExporter

from tov_solver.utils.plotter import EoSPlotter
from tov_solver.utils.exporter import EoSExporter

def run_merger_pipeline(hadronic_eos, quark_eos):

    # Perform the physics construction
    result = perform_maxwell_construction(hadronic_eos, quark_eos)
    p_merged, eps_merged, p_had, eps_had, p_quark, eps_quark, mu_grid = result

    cs2_had = compute_cs2(hadronic_eos, mu_grid)
    cs2_quark = compute_cs2(quark_eos, mu_grid)

    # 1. Export the merged data for Gnuplot
    exporter = EoSExporter(output_dir="outputs/data")
    exporter.export_to_csv(eps_merged, p_merged, filename="merged_eos.csv")
    
    # 2. Visualize the results
    plotter = EoSPlotter(output_dir="outputs/plots")
    
    # Add layers for visual verification
    plotter.add_phase_layer(eps_had, p_had, label="Hadronic", linestyle='--')
    plotter.add_phase_layer(eps_quark, p_quark, label="Quark", linestyle=':')
    plotter.add_phase_layer(eps_merged, p_merged, label="Merged (Maxwell)", linestyle='-')

    plotter.finalize_plot(filename="final_eos_comparison.png")

    # Calculate merged CS2 using the stitched arrays
    # Note: If you don't have a spline for the merged result yet, you can use np.gradient as a fallback
    cs2_merged = np.gradient(p_merged, mu_grid) / np.gradient(eps_merged, mu_grid)    
    plotter.plot_cs2(mu_grid, cs2_had, cs2_quark, cs2_merged=cs2_merged)
    
    # 1. Calculate the actual derivative arrays
    dp_dmu_had = hadronic_eos.p_interp.derivative()(mu_grid)
    dp_dmu_quark = quark_eos.p_interp.derivative()(mu_grid)

    # 2. Pass these arrays to the plotter
    plotter.plot_derivatives(mu_grid, dp_dmu_had, dp_dmu_quark)

    return p_merged, eps_merged

def find_transition_point(hadronic_eos, quark_eos):
    """
    Finds the intersection point where P_had == P_quark.
    Assumes hadronic_eos and quark_eos are objects with .p (pressure) 
    and .mu (chemical potential) attributes.
    """
    # Find the transition where pressures are equal
    # We look for the index where the difference is minimized
    diff = np.abs(hadronic_eos.p - quark_eos.p)
    transition_idx = np.argmin(diff)
    
    return transition_idx

import numpy as np

def perform_maxwell_construction(hadronic_eos, quark_eos, n_points=1000):
    """
    Evaluates splines on a shared mu_B grid and stitches them together.
    """
    # 1. Define a shared grid covering both models
    mu_min = max(hadronic_eos.mu_B.min(), quark_eos.mu_B.min())
    mu_max = min(hadronic_eos.mu_B.max(), quark_eos.mu_B.max())
    mu_grid = np.linspace(mu_min, mu_max, n_points)

    # 2. Evaluate models on the shared grid
    p_had = hadronic_eos.p_interp(mu_grid)
    eps_had = hadronic_eos.eps_interp(mu_grid)
    
    p_quark = quark_eos.p_interp(mu_grid)
    eps_quark = quark_eos.eps_interp(mu_grid)

    # 3. Find the transition point (where pressures intersect)
    diff = np.abs(p_had - p_quark)
    idx = np.argmin(diff)

    # 4. Stitch
    p_merged = np.concatenate([p_had[:idx], p_quark[idx:]])
    eps_merged = np.concatenate([eps_had[:idx], eps_quark[idx:]])
    
    return p_merged, eps_merged, p_had, eps_had, p_quark, eps_quark, mu_grid

def _compute_cs2(eos_model, mu_grid):
    # Check if the interpolator has a derivative method
    if hasattr(eos_model.p_interp, 'derivative'):
        dp_dmu = eos_model.p_interp.derivative()(mu_grid)
        deps_dmu = eos_model.eps_interp.derivative()(mu_grid)
    else:
        # Fallback for simple functions (optional, or raise descriptive error)
        # Numerical differentiation
        dp_dmu = np.gradient(eos_model.p_interp(mu_grid), mu_grid)
        deps_dmu = np.gradient(eos_model.eps_interp(mu_grid), mu_grid)
        
    return dp_dmu / deps_dmu

def compute_cs2(eos_model, mu_grid):
    # Instead of .derivative(), use np.gradient
    # This is much more stable for "real" simulation data
    p_values = eos_model.p_interp(mu_grid)
    eps_values = eos_model.eps_interp(mu_grid)
    
    dp_dmu = np.gradient(p_values, mu_grid)
    deps_dmu = np.gradient(eps_values, mu_grid)
    
    return dp_dmu / deps_dmu    