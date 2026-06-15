import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

class EoSPlotter:
    def __init__(self, output_dir="plots"):
        self.output_dir = output_dir
        # Ensure the directory exists
        import os
        os.makedirs(output_dir, exist_ok=True)

    def plot_eos(self, p_had, eps_had, p_quark, eps_quark, p_merged, eps_merged):
        """
        Base method to initialize the figure.
        We'll build the plot layers in the next chunks.
        """
        plt.figure(figsize=(8, 6))
        plt.xlabel(r"Energy Density $\epsilon$ [MeV/fm$^3$]")
        plt.ylabel(r"Pressure $P$ [MeV/fm$^3$]")
        plt.grid(True, linestyle="--", alpha=0.7)
        
        # We will add the plotting logic for the phases in the next step
        print("Figure initialized. Ready for data layers.")

    def add_phase_layer(self, eps, p, label, linestyle='-'):
        """
        Adds a single phase to the plot (Hadronic, Quark, or Merged).
        """
        plt.plot(eps, p, label=label, linestyle=linestyle, linewidth=2)

    def finalize_plot(self, filename="eos_comparison.png"):
        """
        Adds legend, saves the plot, and closes the figure.
        """
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, filename), dpi=300)
        plt.close()
        print(f"Plot saved to {self.output_dir}/{filename}")

    def plot_cs2(self, mu_grid, cs2_had, cs2_quark, cs2_merged):
        plt.figure(figsize=(8, 6))
        plt.plot(mu_grid, cs2_had, label="Hadronic", linestyle='--')
        plt.plot(mu_grid, cs2_quark, label="Quark", linestyle=':')

        if cs2_merged is not None:
            plt.plot(mu_grid, cs2_merged, label="Merged (Maxwell)", linestyle='-')

        plt.axhline(1/3, color='k', linestyle=':', label="Conformal Limit") # Useful reference
        plt.xlabel(r"Chemical Potential $\mu_B$ [MeV]")
        plt.ylabel(r"Speed of Sound $c_s^2$")
        plt.grid(True)
        plt.legend()
        plt.savefig(os.path.join(self.output_dir, "cs2_comparison.png"), dpi=300)        

    def plot_derivatives(self, mu_grid, dp_dmu_had, dp_dmu_quark):
        plt.figure(figsize=(8, 6))
        # Plot the arrays directly
        plt.plot(mu_grid, dp_dmu_had, label="dP/dmu (Had)", linestyle='--')
        plt.plot(mu_grid, dp_dmu_quark, label="dP/dmu (Quark)", linestyle=':')
        
        plt.xlabel(r"Chemical Potential $\mu_B$ [MeV]")
        plt.ylabel(r"Derivative $dP/d\mu$")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(self.output_dir, "derivatives_check.png"), dpi=300)